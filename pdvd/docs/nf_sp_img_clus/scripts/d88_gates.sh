#!/usr/bin/env bash
# doc pdvd/88 -- doc 78 action item 7b: the stop snap restricted to what the entry
# reaches (stop_snap_reachable), on candidates where doc 87's stop-local keep fired.
#   Fork by duplication (CLAUDE.md M10) of d87_gates.sh: the same zip / tree / branch
#   comparisons and movers by name against production, plus section 6: each ON arm
#   against ITS doc 87 twin (the same TLA without the new knob), so the knob's own
#   effect is isolated, and stop_snap_skipped by candidate.
#   OFF: p88voff vs p85vprod (PDVD production), p88hoff vs p85hoff (PDHD);
#        p88vr (the knob with no keep) vs p85vprod.
#   ON : p88v5fr (flip candidate) vs p87v5f, p88v5r vs p87v5, p88v20fr vs p87v20f.
#   The record is the MERGED smx1a+smx3+smx4+smx5 one (doc 86).
# Usage: bash d88_gates.sh > /home/xqian/tmp/p88/gates.log 2>&1
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p88
W87=/home/xqian/tmp/p87
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_verdicts.json
export STM_SCAN_RECORD=$REC
PROD=${PROD:-p85vprod}          # the production baseline arm (post doc-85 flip)
VOFF=${VOFF:-p88voff}; HOFF=${HOFF:-p88hoff}; HBASE=${HBASE:-p85hoff}
ARMS=${ARMS:-"p88v5fr p88vr p88v5r p88v20fr"}
PAIRS=${PAIRS:-"p88v5fr:p87v5f p88v5r:p87v5 p88v20fr:p87v20f"}
cd $IMG
echo "record: $REC"

echo "=== 0. completeness, pins, markers"
for spec in "pdvd $PROD /home/xqian/tmp/p85" "pdvd $VOFF $W" "pdhd $HBASE /home/xqian/tmp/p85" "pdhd $HOFF $W"; do
    set -- $spec; det=$1 a=$2 wd=$3
    n=$(ls -d $det/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in $det/work/*_$a; do [ -s $e/tracking-pr.root ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" $det/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $det $a dirs $n with tracking-pr.root $ok loader-deaths $lt"
done
for a in $ARMS; do
    n=$(ls -d pdvd/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in pdvd/work/*_$a; do [ -s $e/tracking-pr.root ] && echo 1; done | wc -l)
    echo "  pdvd $a dirs $n with tracking-pr.root $ok | $(grep -h 'md5' $W/arm_$a.log 2>/dev/null | tr '\n' ' ')"
done
zipcmp() {   # det arm base
DET=$1 ARM=$2 BASE=$3 python3 - <<'EOF'
import glob, hashlib, os, zipfile, collections, re
det, arm, base = os.environ["DET"], os.environ["ARM"], os.environ["BASE"]
same = diff = miss = 0; layers = collections.Counter(); ev_diff = []
def members(fn):
    out = {}
    with zipfile.ZipFile(fn) as z:
        for n in z.namelist():
            if n.endswith("/"): continue
            out[n] = hashlib.sha256(z.read(n)).hexdigest()
    return out
def layer(n): return re.sub(r"^\d+-", "", os.path.basename(n)).replace(".json", "")
for eb in sorted(glob.glob(det + "/work/*_" + base)):
    pre = eb[: -len(base) - 1]; ea = pre + "_" + arm
    if not (os.path.exists(eb + "/mabc-pr.zip") and os.path.exists(ea + "/mabc-pr.zip")):
        miss += 1; continue
    A, B = members(eb + "/mabc-pr.zip"), members(ea + "/mabc-pr.zip")
    d = [n for n in set(A) | set(B) if A.get(n) != B.get(n)]
    if d:
        diff += 1; ev_diff.append(os.path.basename(pre))
        for n in d: layers[layer(n)] += 1
    else: same += 1
cs = cd = 0
for eb in sorted(glob.glob(det + "/work/*_" + base)):
    pre = eb[: -len(base) - 1]; ea = pre + "_" + arm
    ja = glob.glob(eb + "/calib-pr-evt*.json"); jb = glob.glob(ea + "/calib-pr-evt*.json")
    if ja and jb:
        if open(ja[0], "rb").read() == open(jb[0], "rb").read(): cs += 1
        else: cd += 1
print("  %s %s vs %s: events with an identical zip %d, differing %d, missing %d | calib json same %d diff %d" % (det, arm, base, same, diff, miss, cs, cd))
if diff:
    print("    members that differ, by layer (events): %s" % dict(sorted(layers.items(), key=lambda kv: -kv[1])))
    print("    events: %s" % " ".join(ev_diff[:12]))
EOF
}
treecmp() {   # det arm base
DET=$1 ARM=$2 BASE=$3 python3 - <<'EOF'
import glob, os, uproot, collections, awkward as ak
det, arm, BASE = os.environ["DET"], os.environ["ARM"], os.environ["BASE"]
same = collections.Counter(); diff = collections.Counter(); diff_ev = collections.defaultdict(list); nev = 0
for eb in sorted(glob.glob(det + "/work/*_" + BASE)):
    pre = eb[: -len(BASE) - 1]; ea = pre + "_" + arm
    fa, fb = eb + "/tracking-pr.root", ea + "/tracking-pr.root"
    if not (os.path.exists(fa) and os.path.exists(fb)): continue
    nev += 1
    A, B = uproot.open(fa), uproot.open(fb)
    for t in sorted(set(k.split(";")[0] for k in A.keys()) | set(k.split(";")[0] for k in B.keys())):
        try:
            ta, tb = A[t].arrays(library="ak"), B[t].arrays(library="ak")
            def L(x):
                try: return ak.to_list(ak.nan_to_none(x))
                except Exception: return ak.to_list(x)
            ok = set(ta.fields) == set(tb.fields) and all(L(ta[n]) == L(tb[n]) for n in ta.fields)
        except Exception:
            ok = False
        (same if ok else diff)[t] += 1
        if not ok: diff_ev[t].append(os.path.basename(pre))
print("  %s %s vs %s over %d events: trees identical on every event: %s" % (det, arm, BASE, nev, sorted(t for t in same if t not in diff)))
for t in sorted(diff): print("    %s differs on %d events: %s" % (t, diff[t], " ".join(diff_ev[t][:10])))
EOF
}
G="matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED"

echo "=== 1. OFF gate, PDVD: $VOFF vs $PROD"
zipcmp pdvd $VOFF $PROD; treecmp pdvd $VOFF $PROD
python3 $X/d51g_branch_census.py --before "pdvd/work/*_$PROD" --after "pdvd/work/*_$VOFF" --before-arm $PROD --after-arm $VOFF --pts --out $W/g_$VOFF.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_$VOFF.txt

echo "=== 2. OFF gate, PDHD: $HOFF vs $HBASE"
zipcmp pdhd $HOFF $HBASE; treecmp pdhd $HOFF $HBASE
python3 $X/d51g_branch_census.py --before "pdhd/work/*_$HBASE" --after "pdhd/work/*_$HOFF" --before-arm $HBASE --after-arm $HOFF --pts --out $W/g_$HOFF.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_$HOFF.txt

echo "=== 3. every candidate on which any production branch or point-row set moved, by item name"
PROD=$PROD ARMS="$ARMS" REC=$REC python3 - <<'EOF'
import glob, os, uproot, numpy as np, json, collections
PROD = os.environ["PROD"]; ARMS = os.environ["ARMS"].split(); REC = os.environ["REC"]
R = {v["key"]: v for v in json.load(open(REC))}
NEW = {"n_floored_near_stop", "stop_snap_skipped"}
def read(arm):
    out = {}; pts = {}
    for d in sorted(glob.glob("pdvd/work/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        try:
            f = uproot.open(fn)
            t = f["T_stm_michel"].arrays(library="np")
            q = f["T_stm_michel_pts"].arrays(["cluster_id", "role", "seg_id", "x", "y", "z"], library="np")
        except Exception: continue
        for i in range(len(t["cluster_id"])):
            k = "%s/%d" % (ev, t["cluster_id"][i])
            out[k] = {kk: t[kk][i] for kk in t}
            sel = q["cluster_id"] == t["cluster_id"][i]
            # the rows of every role, in order, with their values: identical in value AND order off the keep-fire set
            pts[k] = (collections.Counter((int(r), int(s) // 1000) for s, r in zip(q["seg_id"][sel], q["role"][sel])),
                      [(int(r), int(s), round(float(x), 4), round(float(y), 4), round(float(z), 4))
                       for r, s, x, y, z in zip(q["role"][sel], q["seg_id"][sel], q["x"][sel], q["y"][sel], q["z"][sel])])
    return out, pts
def same(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)) and np.isnan(a) and np.isnan(b): return True
        return bool(a == b)
    except Exception:
        return False
def tag(k): return R.get(k, {}).get("verdict", "unjudged")
g = lambda r, b, d=0: r.get(b, d)
P, PP = read(PROD)
print("  production arm %s: %d candidates" % (PROD, len(P)))
for arm in ARMS:
    A, AP = read(arm)
    fire = {k for k, a in A.items() if int(g(a, "n_kept_near_stop_main")) + int(g(a, "n_kept_near_stop_comp")) > 0}
    print("  --- %s vs %s: %d candidates; keep fires on %d (residuals main %d comp %d); floored residuals %d on %d candidates" % (
        arm, PROD, len(A), len(fire), sum(int(g(a, "n_kept_near_stop_main")) for a in A.values()),
        sum(int(g(a, "n_kept_near_stop_comp")) for a in A.values()),
        sum(int(g(a, "n_floored_near_stop")) for a in A.values()), sum(1 for a in A.values() if int(g(a, "n_floored_near_stop")))))
    only = sorted({b for a in A.values() for b in a} - {b for p in P.values() for b in p})
    print("    branches only in %s: %s" % (arm, only))
    movers = []; bset = collections.Counter(); ord_moved = []
    for k, a in sorted(A.items()):
        if k not in P: print("    NEW candidate %s" % k); continue
        bs = sorted(b for b in P[k] if b not in NEW and not same(a.get(b), P[k].get(b)))
        (ca, la), (cp, lp) = AP[k], PP.get(k, (collections.Counter(), []))
        roles = sorted({r for (r, c) in set(ca) | set(cp) if ca.get((r, c)) != cp.get((r, c))})
        if la != lp: ord_moved.append(k)
        if bs or roles: movers.append((k, bs, roles)); bset.update(bs)
    for k in sorted(set(P) - set(A)): print("    DROPPED candidate %s" % k)
    for k, bs, roles in movers:
        a, p = A[k], P[k]
        print("    %-14s %-11s %s is_stm %d->%d mf %d->%d bits %d->%d conn %d->%d ke %.1f->%.1f | kept %d+%d floored %d | roles moved %s | branches %d" % (
            k, tag(k), "FIRE  " if k in fire else "NOFIRE", int(p["is_stm"]), int(a["is_stm"]), int(p["michel_found"]), int(a["michel_found"]),
            int(p["reject_bits"]), int(a["reject_bits"]), int(g(p, "michel_conn_type")), int(g(a, "michel_conn_type")),
            float(g(p, "michel_ke_best")), float(g(a, "michel_ke_best")),
            int(g(a, "n_kept_near_stop_main")), int(g(a, "n_kept_near_stop_comp")), int(g(a, "n_floored_near_stop")), roles, len(bs))
            + (" | SKIP" if int(g(a, "stop_snap_skipped")) else ""))
    print("    candidates moved: %d, of them is_stm 1 on the baseline: %d" % (len(movers), sum(1 for k, _, _ in movers if int(P[k]["is_stm"]))))
    nf = [k for k, _, _ in movers if k not in fire]
    print("    MOVED WITHOUT A KEEP FIRE: %d %s" % (len(nf), " ".join(nf)))
    print("    keep fires that moved nothing: %d %s" % (len(fire - {k for k, _, _ in movers}), " ".join(sorted(fire - {k for k, _, _ in movers}))))
    print("    branches that moved anywhere (candidates): %s" % dict(sorted(bset.items())))
    print("    candidates whose rows differ in value or order: %d, of them without a keep fire: %d %s" % (
        len(ord_moved), sum(1 for k in ord_moved if k not in fire), " ".join(k for k in ord_moved if k not in fire)))
    fl_s = [(k, int(P[k]["is_stm"]), int(a["is_stm"])) for k, a in sorted(A.items()) if k in P and int(P[k]["is_stm"]) != int(a["is_stm"])]
    fl_m = [(k, int(P[k]["michel_found"]), int(a["michel_found"])) for k, a in sorted(A.items()) if k in P and int(P[k]["michel_found"]) != int(a["michel_found"])]
    print("    is_stm flips %d: %s" % (len(fl_s), " ".join("%s(%s %d->%d)" % (k, tag(k), b, c) for k, b, c in fl_s)))
    print("    michel_found flips %d: %s" % (len(fl_m), " ".join("%s(%s %d->%d)" % (k, tag(k), b, c) for k, b, c in fl_m)))
EOF
for a in $ARMS; do
    python3 $X/d51g_branch_census.py --before "pdvd/work/*_$PROD" --after "pdvd/work/*_$a" --before-arm $PROD --after-arm $a --pts --out $W/g_$a.txt > /dev/null 2>&1
    echo "  branch census $a rc=$?"; grep -E "$G" $W/g_$a.txt | head -12
done

echo "=== 3b. the ON arms' Bee zips, calib and trees against production"
for a in $ARMS; do zipcmp pdvd $a $PROD; treecmp pdvd $a $PROD; done

echo "=== 4. prep + census score for every arm, against the record"
cd $IMG/pdhd/stm_michel_scan
[ -d $W/prep_$PROD ] || ln -s $W87/prep_$PROD $W/prep_$PROD
for a in $PROD $VOFF $ARMS; do
    [ -d $W/prep_$a ] || ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  --- $a: $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l) payloads"
    STM_SCAN_RECORD= python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_$a.json 2>&1 | grep -E "is_stm|michel|check|F1|purity" | head -12
    # the owner's grading record (smx1a + smx3 + smx4 + smx5, doc 86); the default above is the
    # frozen doc 55 record, kept because --check diffs doc 55's literals.  Doc 84 sec 7.3.
    echo "  --- $a on the merged record"
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_merged_$a.json 2>&1 | grep -E "^census|as shipped"
done
STM_SCAN_RECORD= python3 census_score.py --check 2>&1 | tail -3

echo "=== 5. d87_keep_census.py: each ON arm against production (record $REC)"
cd $IMG
for a in $ARMS; do
    python3 $X/d87_keep_census.py --arm $a --prep $W/prep_$a --base $PROD --base-prep $W/prep_$PROD > $W/keep_$a.txt 2>&1
    echo "  keep census $a rc=$? -> $W/keep_$a.txt"; grep -v "^  0[0-9]" $W/keep_$a.txt | head -60
done

echo "=== 6. each ON arm against its doc 87 twin (the knob's own effect), and stop_snap_skipped"
cd $IMG
for pr in $PAIRS; do
    a=${pr%%:*}; b=${pr##*:}
    echo "  --- $a vs $b"
    zipcmp pdvd $a $b
    python3 $X/d51g_branch_census.py --before "pdvd/work/*_$b" --after "pdvd/work/*_$a" --before-arm $b --after-arm $a --pts --out $W/g_${a}_vs_$b.txt > /dev/null 2>&1
    echo "  branch census rc=$?"; grep -E "$G" $W/g_${a}_vs_$b.txt | head -12
    python3 $X/d87_keep_census.py --arm $a --prep $W/prep_$a --base $b --base-prep $W87/prep_$b > $W/keep_${a}_vs_$b.txt 2>&1
    echo "  keep census vs twin rc=$? -> $W/keep_${a}_vs_$b.txt"; sed -n '/=== C/,$p' $W/keep_${a}_vs_$b.txt
done
ARMS="$ARMS" python3 - <<'PYEOF'
import glob, os, uproot, numpy as np
for arm in os.environ["ARMS"].split():
    sk = []; n = 0
    for d in sorted(glob.glob("pdvd/work/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        f = uproot.open(fn)
        if "T_stm_michel" not in f: continue
        t = f["T_stm_michel"].arrays(library="np")
        if "stop_snap_skipped" not in t: continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        for i in range(len(t["cluster_id"])):
            n += 1
            if t["stop_snap_skipped"][i]: sk.append("%s/%d" % (ev, t["cluster_id"][i]))
    print("  %s: stop_snap_skipped on %d of %d candidates: %s" % (arm, len(sk), n, " ".join(sk)))
PYEOF
echo GATES_DONE

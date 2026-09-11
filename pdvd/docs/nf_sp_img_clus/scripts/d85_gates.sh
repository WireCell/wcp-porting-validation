#!/usr/bin/env bash
# doc pdvd/85 -- doc 78 action item 5: a capture gamma only on a stopper
# (stop_gamma_require_stm), and the capture ring widened to 50 cm.
#   Fork by duplication (CLAUDE.md M10) of d84_gates.sh: same zip / tree / branch
#   comparisons; section 3 names every candidate on which ANY production branch or
#   point-row set moved, with the capture-gamma fields, and checks the non-role-5
#   rows for value AND order; section 5 runs d85_gamma_census.py per arm.
#   Every gate arm is BARE PRODUCTION plus this round's key (no d53 survey bag).
#   OFF: p85voff vs p84vprod (PDVD production after the doc 84 flip),
#        p85hoff vs p84hoff (PDHD).
#   ON : p85vwh / p85vsg50 vs p84vprod; the census against the merged record.
#   Smoke: p85vwhs (knob + survey bag) -- role-6 rej-16 rows (section 6).
# Usage: bash d85_gates.sh > /home/xqian/tmp/p85/gates.log 2>&1
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p85
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
PROD=${PROD:-p84vprod}          # the production baseline arm (post doc-84 flip)
VOFF=${VOFF:-p85voff}; HOFF=${HOFF:-p85hoff}; HBASE=${HBASE:-p84hoff}
ARMS=${ARMS:-"p85vwh p85vsg50"}
cd $IMG

echo "=== 0. completeness, pins, markers"
for spec in "pdvd $PROD /home/xqian/tmp/p84" "pdvd $VOFF $W" "pdhd $HBASE /home/xqian/tmp/p84" "pdhd $HOFF $W"; do
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
NEW = {"n_stop_gammas_withheld"}
GB = ["n_stop_gammas", "stop_gamma_ke_tot", "stop_gamma_dis_min", "stop_gamma_dis_max", "n_michel_gammas", "michel_ke_gamma"]
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
            # the rows of every role but 5, in order, with their values: must be identical in value AND order
            pts[k] = (collections.Counter((int(r), int(s) // 1000) for s, r in zip(q["seg_id"][sel], q["role"][sel])),
                      [(int(r), int(s), round(float(x), 4), round(float(y), 4), round(float(z), 4))
                       for r, s, x, y, z in zip(q["role"][sel], q["seg_id"][sel], q["x"][sel], q["y"][sel], q["z"][sel]) if int(r) != 5])
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
    print("  --- %s vs %s: %d candidates; n_stop_gammas total %d -> %d; n_stop_gammas_withheld total %d" % (
        arm, PROD, len(A), sum(int(g(p, "n_stop_gammas")) for p in P.values()), sum(int(g(a, "n_stop_gammas")) for a in A.values()),
        sum(int(g(a, "n_stop_gammas_withheld")) for a in A.values())))
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
        print("    %-14s %-11s is_stm %d->%d mf %d->%d bits %d->%d | stop gammas %d->%d withheld %d | michel gammas %d->%d | roles moved %s | branches %d" % (
            k, tag(k), int(p["is_stm"]), int(a["is_stm"]), int(p["michel_found"]), int(a["michel_found"]), int(p["reject_bits"]), int(a["reject_bits"]),
            int(g(p, "n_stop_gammas")), int(g(a, "n_stop_gammas")), int(g(a, "n_stop_gammas_withheld")),
            int(g(p, "n_michel_gammas")), int(g(a, "n_michel_gammas")), roles, len(bs)))
    print("    candidates moved: %d, of them is_stm 1 on the baseline: %d" % (len(movers), sum(1 for k, _, _ in movers if int(P[k]["is_stm"]))))
    print("    branches that moved anywhere (candidates): %s" % dict(sorted(bset.items())))
    print("    candidates whose non-role-5 rows differ in value or order: %d %s" % (len(ord_moved), " ".join(ord_moved[:20])))
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
for a in $PROD $VOFF $ARMS; do
    [ -d $W/prep_$a ] || ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  --- $a: $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l) payloads"
    python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_$a.json 2>&1 | grep -E "is_stm|michel|check|F1|purity" | head -12
    # the owner's grading record since doc 68 (smx1a + smx3 + smx4); the default above is the
    # frozen doc 55 record, kept because --check diffs doc 55's literals.  Doc 84 sec 7.3.
    echo "  --- $a on the merged record"
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_merged_$a.json 2>&1 | grep -E "^census|as shipped"
done
python3 census_score.py --check 2>&1 | tail -3

echo "=== 5. d85_gamma_census.py: production, then each ON arm against the knob-off arm"
cd $IMG
python3 $X/d85_gamma_census.py --arm $VOFF --prep $W/prep_$VOFF > $W/census_$VOFF.txt 2>&1; echo "  census $VOFF rc=$?"
for a in $ARMS; do
    python3 $X/d85_gamma_census.py --arm $a --prep $W/prep_$a --base $VOFF --base-prep $W/prep_$VOFF --sections BE > $W/census_$a.txt 2>&1
    echo "  census $a rc=$? -> $W/census_$a.txt"; sed -n '/=== E/,$p' $W/census_$a.txt | grep -v "^      " | head -40
done

echo "=== 6. smoke p85vwhs (knob + survey bag): withheld segments as role-6 rej-16 rows"
python3 - <<'EOF'
import glob, os, uproot, numpy as np
n16 = nc = n5r = ev = 0; wh = 0
for d in sorted(glob.glob("pdvd/work/*_p85vwhs")):
    fn = d + "/tracking-pr.root"
    if not os.path.exists(fn): continue
    ev += 1
    f = uproot.open(fn)
    if "T_stm_michel" not in f: continue     # 039252_11: no STM candidate, no tree (as on every arm)
    t = f["T_stm_michel"].arrays(["cluster_id", "is_stm", "n_stop_gammas", "n_stop_gammas_withheld"], library="np")
    q = f["T_stm_michel_pts"].arrays(["cluster_id", "role", "rej"], library="np")
    for i in range(len(t["cluster_id"])):
        sel = q["cluster_id"] == t["cluster_id"][i]
        r16 = int(np.sum((q["role"][sel] == 6) & (q["rej"][sel] == 16)))
        wh += int(t["n_stop_gammas_withheld"][i])
        if r16: nc += 1; n16 += r16
        if not t["is_stm"][i]: n5r += int(np.sum(q["role"][sel] == 5))
print("  p85vwhs: %d events; withheld gammas %d; candidates with rej-16 rows %d (%d rows); role-5 rows on is_stm-0 candidates %d" % (ev, wh, nc, n16, n5r))
EOF

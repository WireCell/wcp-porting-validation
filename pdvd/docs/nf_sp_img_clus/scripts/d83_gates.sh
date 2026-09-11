#!/usr/bin/env bash
# doc pdvd/83 -- doc 78 action item 4, the attached arm near the stop.
#   Fork by duplication (CLAUDE.md M10) of d82_gates.sh: same zip / tree / branch
#   comparisons; section 3 reads the Michel instead of the stop.
#   Every arm is BARE PRODUCTION plus this round's key (no d53 survey bag).
#   OFF: p83voff vs p82vprod (PDVD production after the doc 82 split_kink flip),
#        p83hoff vs p82bhoff (PDHD) -- every Bee zip member, calib json, every
#        tracking-pr.root tree, every T_stm_michel branch and point row.
#   ON : p83v5 / p83v7 / p83v10 vs p82vprod -- every candidate whose Michel changed, by
#        item name, with the segment the rule took; every branch that moved on any other
#        candidate; the is_stm and michel_found census against the record.
# Usage: bash d83_gates.sh > /home/xqian/tmp/p83/gates.log 2>&1
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p83
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
PROD=${PROD:-p82vprod}          # the production baseline arm (post doc-82 flip)
VOFF=${VOFF:-p83voff}; HOFF=${HOFF:-p83hoff}; HBASE=${HBASE:-p82bhoff}
ARMS=${ARMS:-"p83v5 p83v7 p83v10"}
cd $IMG

echo "=== 0. completeness, pins, markers"
for spec in "pdvd $PROD /home/xqian/tmp/p82" "pdvd $VOFF $W" "pdhd $HBASE /home/xqian/tmp/p82" "pdhd $HOFF $W"; do
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

echo "=== 3. every candidate whose Michel changed, by item name; every branch that moved elsewhere"
PROD=$PROD ARMS="$ARMS" REC=$REC python3 - <<'EOF'
import glob, os, uproot, numpy as np, json, collections
PROD = os.environ["PROD"]; ARMS = os.environ["ARMS"].split(); REC = os.environ["REC"]
R = {v["key"]: v for v in json.load(open(REC))}
NEW = {"michel_near_arm", "near_arm_dist_cm", "n_near_arms_examined"}
def read(arm):
    out = {}; pts = {}
    for d in sorted(glob.glob("pdvd/work/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        try:
            f = uproot.open(fn)
            t = f["T_stm_michel"].arrays(library="np")
            q = f["T_stm_michel_pts"].arrays(["cluster_id", "role", "seg_id"], library="np")
        except Exception: continue
        for i in range(len(t["cluster_id"])):
            k = "%s/%d" % (ev, t["cluster_id"][i])
            out[k] = {kk: t[kk][i] for kk in t}
            sel = q["cluster_id"] == t["cluster_id"][i]
            pts[k] = sorted({int(s) for s, r in zip(q["seg_id"][sel], q["role"][sel]) if int(r) == 3})
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
def seg_tag(k, s): return R.get(k, {}).get("tags", {}).get(str(s), "-")
P, PP = read(PROD)
print("  production arm %s: %d candidates" % (PROD, len(P)))
for arm in ARMS:
    A, AP = read(arm)
    fired = [k for k, a in sorted(A.items()) if int(a["michel_near_arm"])]
    exam = sum(int(a["n_near_arms_examined"]) for a in A.values())
    print("  --- %s vs %s: %d candidates; the rule fired on %d; kOther arms offered the gate %d" % (arm, PROD, len(A), len(fired), exam))
    for k in fired:
        a, p = A[k], P.get(k)
        new3 = [s for s in AP[k] if s not in PP.get(k, [])]
        print("    %-14s %-11s seg %d (tag %s) at %.2f cm | len %.1f cm mip %.2f kink %.1f far %.1f cm | KE best %.1f MeV | conn %d->%d  is_stm %d->%d  michel_found %d->%d  bits %d->%d | new role-3 segs %s" % (
            k, tag(k), int(a["michel_seg_id"]), seg_tag(k, int(a["michel_seg_id"])), float(a["near_arm_dist_cm"]),
            float(a["michel_len"]), float(a["michel_mip"]), float(a["michel_kink_deg"]), float(a["michel_far_len"]),
            float(a["michel_ke_best"]), int(p["michel_conn_type"]), int(a["michel_conn_type"]),
            int(p["is_stm"]), int(a["is_stm"]), int(p["michel_found"]), int(a["michel_found"]),
            int(p["reject_bits"]), int(a["reject_bits"]), ["%d(%s)" % (s, seg_tag(k, s)) for s in new3]))
    # every candidate the rule did NOT fire on: is every production branch identical?
    moved = collections.Counter(); moved_ev = collections.defaultdict(list); n_other = 0
    for k, a in sorted(A.items()):
        if k in fired or k not in P: continue
        n_other += 1
        for b in P[k]:
            if b in NEW: continue
            if not same(a.get(b), P[k][b]):
                moved[b] += 1; moved_ev[b].append(k)
        if AP[k] != PP.get(k): moved["<role-3 rows>"] += 1; moved_ev["<role-3 rows>"].append(k)
    print("    the other %d candidates: production branches that moved: %s" % (n_other, dict(moved) if moved else "NONE"))
    for b in moved: print("      %s: %s" % (b, " ".join(moved_ev[b][:10])))
    fb = collections.Counter()
    for k in fired:
        for b in P.get(k, {}):
            if b not in NEW and not same(A[k].get(b), P[k][b]): fb[b] += 1
    print("    branches that moved on the fired candidates: %s" % sorted(fb))
    fl_s = [(k, int(P[k]["is_stm"]), int(a["is_stm"])) for k, a in sorted(A.items()) if k in P and int(P[k]["is_stm"]) != int(a["is_stm"])]
    fl_m = [(k, int(P[k]["michel_found"]), int(a["michel_found"])) for k, a in sorted(A.items()) if k in P and int(P[k]["michel_found"]) != int(a["michel_found"])]
    print("    is_stm flips %d: %s" % (len(fl_s), " ".join("%s(%s %d->%d)" % (k, tag(k), b, c) for k, b, c in fl_s)))
    print("    michel_found flips %d: %s" % (len(fl_m), " ".join("%s(%s %d->%d)" % (k, tag(k), b, c) for k, b, c in fl_m)))
EOF
for a in $ARMS; do
    python3 $X/d51g_branch_census.py --before "pdvd/work/*_$PROD" --after "pdvd/work/*_$a" --before-arm $PROD --after-arm $a --pts --out $W/g_$a.txt > /dev/null 2>&1
    echo "  branch census $a rc=$?"; grep -E "$G" $W/g_$a.txt | head -12
done

echo "=== 4. prep + census score for every arm, against the record"
cd $IMG/pdhd/stm_michel_scan
for a in $PROD $ARMS; do
    [ -d $W/prep_$a ] || ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  --- $a: $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l) payloads"
    python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_$a.json 2>&1 | grep -E "is_stm|michel|check|F1|purity" | head -12
done
python3 census_score.py --check 2>&1 | tail -3

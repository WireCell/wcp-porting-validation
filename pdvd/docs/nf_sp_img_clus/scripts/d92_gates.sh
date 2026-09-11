#!/usr/bin/env bash
# doc pdvd/92 -- the gates for the wide Bragg-peak read (doc 91 sec 10 item 1).
#
# Fork by duplication (CLAUDE.md M10) of d88_gates.sh (its zipcmp / treecmp / branch census).  This
# round changes C++, so the OFF gate is the real bar:
#   1  OFF, PDVD : p92voff (new binary, knob absent) vs p90vprod (production) -- must be identical.
#   2  OFF, PDHD : p92hoff vs p88hoff -- PDHD stays OFF and its config has not moved.
#   3  ON        : p92v13 and p92v15 against the twin written BEFORE they ran
#                  (/home/xqian/tmp/p92/twin.json), item by item -- the same firing items, the same
#                  reject_bits / topology_cleared_bits on each, and bragg_wide_shift_cm equal to the
#                  predicted shift.  A one-row window mis-resolution shows up there as ~0.6 cm.
#   4  ON        : the branches that must NOT move anywhere (rec.bragg is deliberately not replaced),
#                  and any branch that moved outside the expected set.
#   5  ON        : the census on the judged record, against the twin's prediction.
# Usage: bash d92_gates.sh > /home/xqian/tmp/p92/gates.log 2>&1
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p92
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json
export STM_SCAN_RECORD=$REC
PROD=${PROD:-p90vprod}
VOFF=${VOFF:-p92voff}; HOFF=${HOFF:-p92hoff}; HBASE=${HBASE:-p88hoff}
ARMS=${ARMS:-"p92v13 p92v15"}
cd $IMG
echo "record: $REC"
echo "twin  : $W/twin.json (written before the arms ran)"

echo "=== 0. completeness, pin, loader deaths"
for spec in "pdvd $PROD" "pdvd $VOFF" "pdhd $HBASE" "pdhd $HOFF" "pdvd p92v13" "pdvd p92v15"; do
    set -- $spec; det=$1 a=$2
    n=$(ls -d $det/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in $det/work/*_$a; do [ -s $e/tracking-pr.root ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" $det/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $det $a dirs $n with tracking-pr.root $ok loader-deaths $lt"
done
md5sum $W/libpin_p92/*.so* > $W/libpin_md5_after.txt
cmp -s $W/libpin_md5_before.txt $W/libpin_md5_after.txt \
    && echo "  pin libpin_p92: $(wc -l < $W/libpin_md5_after.txt) libraries, md5 identical before and after" \
    || echo "  *** PIN CHANGED between launch and gates -- every arm is unattributable"

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
print("  %s %s vs %s: events with an identical zip %d, differing %d, missing %d" % (det, arm, base, same, diff, miss))
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

echo "=== 1. OFF gate, PDVD: $VOFF vs $PROD  (must be identical on everything)"
zipcmp pdvd $VOFF $PROD; treecmp pdvd $VOFF $PROD
python3 $X/d51g_branch_census.py --before "pdvd/work/*_$PROD" --after "pdvd/work/*_$VOFF" \
    --before-arm $PROD --after-arm $VOFF --pts --out $W/g_$VOFF.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_$VOFF.txt

echo "=== 2. OFF gate, PDHD: $HOFF vs $HBASE  (PDHD stays OFF)"
zipcmp pdhd $HOFF $HBASE; treecmp pdhd $HOFF $HBASE
python3 $X/d51g_branch_census.py --before "pdhd/work/*_$HBASE" --after "pdhd/work/*_$HOFF" \
    --before-arm $HBASE --after-arm $HOFF --pts --out $W/g_$HOFF.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_$HOFF.txt

echo "=== 3-5. the ON arms against the twin, item by item"
PROD=$PROD ARMS="$ARMS" REC=$REC TWIN=$W/twin.json python3 - <<'EOF'
import collections, glob, json, os
import numpy as np, uproot
import sys
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C
PROD = os.environ["PROD"]; ARMS = os.environ["ARMS"].split()
TW = json.load(open(os.environ["TWIN"]))
R = C.load_record(); J = [k for k in R if C.judged(R[k])]

# rec.bragg is deliberately NOT replaced by the wide read, so every one of these must be identical
# on EVERY candidate, movers included.
FROZEN = ["michel_found", "michel_ke_best", "michel_len", "michel_conn_type", "bragg_anchor_shift_cm",
          "bragg_anchor_fallback", "ks_mu", "ks_flat", "ratio_mu", "ratio_flat", "tail_med",
          "plateau_med", "n_live_pts", "dead_frac_cmp"] + \
         ["comp_fwd%d" % i for i in range(4)] + ["comp_bwd%d" % i for i in range(4)]
# the only branches allowed to move, and the new ones
MOVERS_OK = {"reject_bits", "topology_cleared_bits", "is_stm"}
NEW_OK = {"bragg_wide_fired", "bragg_wide_shift_cm"}
# pred.txt pre-registered this family as expected and allowed: stop_gamma_require_stm is ON in PDVD
# production (doc pdvd/85), so a candidate whose verdict stops rejecting may now publish a capture
# gamma.  It is a consequence of the verdict moving, not a second effect of the rule.
GAMMA_OK = lambda b: b.startswith("stop_gamma") or b.startswith("n_stop_gamma")


def read(arm):
    out = {}
    for d in sorted(glob.glob("pdvd/work/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        f = uproot.open(fn)
        if "T_stm_michel" not in f: continue
        t = f["T_stm_michel"].arrays(library="np")
        for i in range(len(t["cluster_id"])):
            out["%s/%d" % (ev, t["cluster_id"][i])] = {kk: t[kk][i] for kk in t}
    return out


def same(a, b):
    try:
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            if np.isnan(a) and np.isnan(b): return True
        return bool(np.asarray(a == b).all())
    except Exception:
        return False


P = read(PROD)
print("  production arm %s: %d candidates" % (PROD, len(P)))
for arm in ARMS:
    A = read(arm)
    pred = TW["arms"][arm]; items = pred["items"]
    print("\n  --- %s (W %.1f R %.1f D %.1f) vs %s: %d candidates" % (arm, pred["W"], pred["R"], pred["D"], PROD, len(A)))
    fails = []
    if set(A) != set(P):
        print("    *** candidate sets differ: only in arm %s; only in prod %s" % (sorted(set(A) - set(P))[:5], sorted(set(P) - set(A))[:5]))
        fails.append("candidate set")
    newb = sorted({b for a in A.values() for b in a} - {b for p in P.values() for b in p})
    print("    branches only in the arm: %s  (expected: %s)" % (newb, sorted(NEW_OK)))
    if set(newb) != NEW_OK: fails.append("new branch set")

    # 3. every candidate: the published bits, is_stm, and the wide fields against the twin
    nfire = nquiet = nmove = 0; bad = []
    for k in sorted(set(A) & set(P)):
        a, p = A[k], P[k]
        exp = items.get(k)
        fired = int(a.get("bragg_wide_fired", 0))
        if (exp is not None) != bool(fired):
            bad.append("%s: fired %d, twin says %d" % (k, fired, 1 if exp else 0)); continue
        if exp is None:
            if not same(a["reject_bits"], p["reject_bits"]) or not same(a["topology_cleared_bits"], p["topology_cleared_bits"]):
                bad.append("%s: did not fire but its bits moved" % k)
            continue
        nfire += 1
        if abs(float(a["bragg_wide_shift_cm"]) - exp["bragg_wide_shift_cm"]) > 0.005:
            bad.append("%s: shift %.3f cm, twin %.3f cm" % (k, float(a["bragg_wide_shift_cm"]), exp["bragg_wide_shift_cm"]))
        if int(a["reject_bits"]) != exp["reject_bits_after"]:
            bad.append("%s: reject_bits %d, twin %d" % (k, int(a["reject_bits"]), exp["reject_bits_after"]))
        if int(a["topology_cleared_bits"]) != exp["topology_cleared_after"]:
            bad.append("%s: topology_cleared_bits %d, twin %d" % (k, int(a["topology_cleared_bits"]), exp["topology_cleared_after"]))
        if int(a["is_stm"]) != exp["is_stm_after"]:
            bad.append("%s: is_stm %d, twin %d" % (k, int(a["is_stm"]), exp["is_stm_after"]))
        if exp["is_stm_after"] != exp["is_stm_before"]: nmove += 1
        else: nquiet += 1
    print("    fired on %d (twin %d); is_stm movers %d (twin %d); bits-only %d (twin %d)" % (
        nfire, len(items), nmove, len(pred["movers"]), nquiet, len(pred["bits_only"])))
    if bad:
        fails.append("item-by-item")
        print("    *** %d disagreement(s) with the twin:" % len(bad))
        for b in bad[:20]: print("        " + b)
    else:
        print("    every firing item matches the twin exactly (bits, is_stm, shift)")

    # 4. the frozen branches, and anything else that moved
    moved = collections.Counter()
    for k in sorted(set(A) & set(P)):
        for b in set(A[k]) & set(P[k]):
            if not same(A[k][b], P[k][b]): moved[b] += 1
    print("    branches that moved anywhere: %s" % dict(sorted(moved.items(), key=lambda kv: -kv[1])))
    froze = [b for b in FROZEN if moved.get(b)]
    if froze:
        fails.append("frozen branch moved")
        print("    *** these must NOT move (rec.bragg is not replaced): %s" % froze)
    else:
        print("    frozen branches (michel_*, ks_*, ratio_*, tail_med, plateau_med, comp_*, anchor) unmoved on all %d" % len(set(A) & set(P)))
    gam = sorted(b for b in moved if GAMMA_OK(b))
    if gam:
        print("    capture-gamma family moved on %d candidate(s) (pre-registered in pred.txt as a"
              " consequence of the verdict moving): %s" % (max(moved[b] for b in gam), gam))
    extra = sorted(b for b in set(moved) - MOVERS_OK - NEW_OK if not GAMMA_OK(b))
    if extra:
        fails.append("unexpected branch moved")
        print("    *** moved outside the expected set: %s" % extra)

    # 5. the census on the judged record
    def stm(d, k): return bool(d[k]["is_stm"]) if k in d else False
    tp = sum(1 for k in J if stm(A, k) and C.is_stopper(R[k])); fp = sum(1 for k in J if stm(A, k) and not C.is_stopper(R[k]))
    fn = sum(1 for k in J if not stm(A, k) and C.is_stopper(R[k]))
    print("    census on the %d judged: %d/%d/%d (eff %.3f pur %.3f) | twin predicted %s" % (
        len(J), tp, fp, fn, tp / (tp + fn), tp / (tp + fp), pred["census"]))
    if [tp, fp, fn] != pred["census"]: fails.append("census")
    print("    VERDICT %s: %s" % (arm, "PASS" if not fails else "FAIL (%s)" % ", ".join(fails)))
EOF
echo "GATES_DONE"

#!/usr/bin/env bash
# doc pdvd/93 -- the gates for the flip of doc 92's wide Bragg-peak read.
#
# Two questions, in this order:
#   sec 1  IDENTITY.  p93vprod (the flipped file, no TLA) vs p92v13 (the arm that MEASURED the
#          trade, keys via TLA).  Everything must be identical: branches, point rows, trees, zips,
#          calib.  Any difference means the file does not compile to what was measured.
#   sec 2  EFFECT.  p93vprod vs p90vprod (pre-flip production).  Exactly 5 is_stm flips are
#          expected: +4 stoppers (039252_9/101, 039253_12/93, 039349_44/28, 039349_51/29) and
#          +1 THRU (039349_71/37).  Named, so an extra mover cannot hide.
#   sec 3  CENSUS against the smx1a..smx9 record -> 242/8/44, eff 0.846, pur 0.968.
#          Note this record is the smx9 one, NOT the smx8 one doc 92 sec 6 used.  Getting the same
#          numbers is an independent re-check of d92_fold_smx9.py's own "0 class changes" assertion.
#
# Fork by duplication (CLAUDE.md M10) of d90_gates.sh: the same read()/same()/members() comparison
# core, with the twin section dropped (this round predicts identity, not movers) and section 2's
# expected movers named inline.
# Usage: bash d93_gates.sh > /home/xqian/tmp/p93/gates.log 2>&1; echo rc=$?
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p93
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json
export STM_SCAN_RECORD=$REC
ARM=${ARM:-p93vprod}
TWIN_ARM=${TWIN_ARM:-p92v13}
PRE_ARM=${PRE_ARM:-p90vprod}
cd $IMG
echo "record: $REC"
[ -s "$REC" ] || { echo "ABORT: record missing"; exit 2; }

echo "=== 0. completeness, pin, loader deaths"
for a in $PRE_ARM $TWIN_ARM $ARM; do
    n=$(ls -d pdvd/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in pdvd/work/*_$a; do [ -s $e/tracking-pr.root ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" pdvd/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  pdvd $a dirs $n with tracking-pr.root $ok loader-deaths $lt"
done
md5sum /home/xqian/tmp/p92/libpin_p92/*.so* > $W/libpin_md5_after.txt
cmp -s $W/libpin_md5_before.txt $W/libpin_md5_after.txt \
    && echo "  pin libpin_p92: $(wc -l < $W/libpin_md5_after.txt) libraries, md5 identical before and after" \
    || echo "  PIN CHANGED between launch and gates -- the comparison is VOID"

ARM=$ARM TWIN_ARM=$TWIN_ARM PRE_ARM=$PRE_ARM REC=$REC python3 - <<'EOF'
import collections, glob, hashlib, json, os, sys, zipfile
import numpy as np, uproot
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C
ARM, TWIN_ARM, PRE_ARM = os.environ["ARM"], os.environ["TWIN_ARM"], os.environ["PRE_ARM"]
R = {v["key"]: v for v in json.load(open(os.environ["REC"]))}
GAIN = ["039252_9/101", "039253_12/93", "039349_44/28", "039349_51/29"]
COST = ["039349_71/37"]
# doc 92 sec 4 / the offline twin: two candidates whose shape bits this rule clears but whose is_stm
# does NOT move, because P1 (stm_michel_topology_clear) would have cleared them anyway.  The doc 91
# sizing could not see this class; it is pre-registered here so it cannot read as an unexplained mover.
BITS_ONLY = ["039253_8/65", "039349_48/54"]
def read(arm):
    out, pts = {}, {}
    for d in sorted(glob.glob("pdvd/work/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        f = uproot.open(fn)
        if "T_stm_michel" not in f: continue
        t = f["T_stm_michel"].arrays(library="np")
        q = f["T_stm_michel_pts"].arrays(["cluster_id", "role", "seg_id", "x", "y", "z"], library="np")
        for i in range(len(t["cluster_id"])):
            k = "%s/%d" % (ev, t["cluster_id"][i])
            out[k] = {kk: t[kk][i] for kk in t}
            sel = q["cluster_id"] == t["cluster_id"][i]
            pts[k] = [(int(r), int(s), round(float(x), 4), round(float(y), 4), round(float(z), 4))
                      for r, s, x, y, z in zip(q["role"][sel], q["seg_id"][sel], q["x"][sel], q["y"][sel], q["z"][sel])]
    return out, pts
def same(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)) and np.isnan(a) and np.isnan(b): return True
        return bool(a == b)
    except Exception:
        return False
def members(fn):
    with zipfile.ZipFile(fn) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith("/")}
tag = lambda k: R.get(k, {}).get("verdict", "unjudged")
g = lambda r, b: int(r.get(b, 0))
A, AP = read(ARM)
def compare(base, label, expect_identical):
    B, BP = read(base)
    print("\n=== %s: %s vs %s (%d vs %d candidates)" % (label, ARM, base, len(A), len(B)))
    # d90_gates.sh reported the branch SETS and the first fork of this script dropped the line.
    # It has to be here: the per-candidate loop below iterates the BASELINE's branches, so a branch
    # present only in ARM is structurally invisible to it -- which is exactly what a default-OFF
    # writer does the moment the knob is flipped ON (doc 92 writes bragg_wide_fired and
    # bragg_wide_shift_cm only when bragg_wide_anchor_cm > 0, so the flip changes the schema).
    ba = {b for v in A.values() for b in v}; bb = {b for v in B.values() for b in v}
    print("    T_stm_michel branches: %s %d, %s %d | only in %s: %s | only in %s: %s" % (
        ARM, len(ba), base, len(bb), ARM, sorted(ba - bb) or "none", base, sorted(bb - ba) or "none"))
    for k in sorted(set(A) - set(B)): print("    NEW candidate %s" % k)
    for k in sorted(set(B) - set(A)): print("    DROPPED candidate %s" % k)
    moved, bset = {}, collections.Counter()
    for k in sorted(set(A) & set(B)):
        bs = sorted(b for b in B[k] if not same(A[k].get(b), B[k].get(b)))
        rows = AP[k] != BP[k]
        if bs or rows:
            moved[k] = (bs, rows); bset.update(bs)
    ident = len(set(A) & set(B)) - len(moved)
    print("    candidates bit-identical on every branch and point row: %d / %d" % (ident, len(set(A) | set(B))))
    for k, (bs, rows) in moved.items():
        a, b = A[k], B[k]
        print("    %-14s %-13s is_stm %d->%d mf %d->%d bits %d->%d | rows moved %s | branches %s" % (
            k, tag(k), g(b, "is_stm"), g(a, "is_stm"), g(b, "michel_found"), g(a, "michel_found"),
            g(b, "reject_bits"), g(a, "reject_bits"), rows, " ".join(bs)))
    fl = sorted(k for k in moved if g(A[k], "is_stm") != g(B[k], "is_stm"))
    print("    candidates moved %d; branches moved: %s" % (len(moved), dict(sorted(bset.items()))))
    print("    is_stm flips %d: %s" % (len(fl), " ".join("%s(%s %d->%d)" % (k, tag(k), g(B[k], "is_stm"), g(A[k], "is_stm")) for k in fl)))
    zd = []
    for eb in sorted(glob.glob("pdvd/work/*_" + base)):
        ev = os.path.basename(eb)[: -len(base) - 1]; ea = "pdvd/work/%s_%s" % (ev, ARM)
        if os.path.exists(eb + "/mabc-pr.zip") and os.path.exists(ea + "/mabc-pr.zip") and members(eb + "/mabc-pr.zip") != members(ea + "/mabc-pr.zip"):
            zd.append(ev)
    print("    zips differing on %d events: %s" % (len(zd), " ".join(zd) or "none"))
    if expect_identical:
        ok = not moved and not zd and set(A) == set(B)
        print("    VERDICT: %s" % ("IDENTICAL -- the flipped file runs exactly what %s ran" % base if ok
                                   else "*** NOT IDENTICAL: the file does not reproduce the measured arm ***"))
    else:
        # A false positive is an is_stm 0->1 on a THRU-judged item -- it RISES, exactly like a gain.
        # The first version of this script classified by flip DIRECTION, expecting the cost to fall
        # 1->0; all five rises therefore landed in "gains", the cost list came back empty, and the
        # gate reported a spurious FAIL on a run that was in fact exactly as predicted.  Classify by
        # the RECORD'S VERDICT instead: a rise on a stopper is a gain, a rise on a THRU is the FP.
        rise = [k for k in fl if g(A[k], "is_stm") > g(B[k], "is_stm")]
        fall = [k for k in fl if g(A[k], "is_stm") < g(B[k], "is_stm")]
        got_g = [k for k in rise if k in R and C.is_stopper(R[k])]
        got_c = [k for k in rise if not (k in R and C.is_stopper(R[k]))]
        bits_only = sorted(k for k in moved if k not in fl)
        print("    expected gains %s -> got %s" % (GAIN, sorted(got_g)))
        print("    expected cost  %s -> got %s" % (COST, sorted(got_c)))
        print("    is_stm FELL on: %s (none expected)" % (sorted(fall) or "nothing"))
        print("    bits-only movers (is_stm unchanged): expected %s -> got %s" % (BITS_ONLY, bits_only))
        ok = (sorted(got_g) == sorted(GAIN) and sorted(got_c) == sorted(COST)
              and not fall and bits_only == sorted(BITS_ONLY))
        print("    VERDICT: %s" % ("EXACTLY THE PREDICTED 5 is_stm MOVERS (+4 stoppers, 1 FP) PLUS THE 2 PRE-REGISTERED BITS-ONLY MOVERS"
                                   if ok else "*** MOVERS DIFFER FROM THE PREDICTION ***"))
compare(TWIN_ARM, "1. IDENTITY", True)
compare(PRE_ARM, "2. EFFECT", False)
EOF

echo ""
echo "=== 3. trees and calib: $ARM vs $TWIN_ARM (must be identical on every event)"
DET=pdvd ARM=$ARM BASE=$TWIN_ARM python3 - <<'EOF'
import glob, os, uproot, collections, awkward as ak
det, arm, BASE = os.environ["DET"], os.environ["ARM"], os.environ["BASE"]
same = collections.Counter(); diff = collections.Counter(); diff_ev = collections.defaultdict(list); nev = 0
cs = cd = 0; cd_ev = []
for eb in sorted(glob.glob(det + "/work/*_" + BASE)):
    pre = eb[: -len(BASE) - 1]; ea = pre + "_" + arm
    fa, fb = eb + "/tracking-pr.root", ea + "/tracking-pr.root"
    ja = glob.glob(eb + "/calib-pr-evt*.json"); jb = glob.glob(ea + "/calib-pr-evt*.json")
    if ja and jb:
        if open(ja[0], "rb").read() == open(jb[0], "rb").read(): cs += 1
        else: cd += 1; cd_ev.append(os.path.basename(pre))
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
print("  %s vs %s over %d events: trees identical on every event: %s | calib json same %d diff %d %s" % (
    arm, BASE, nev, sorted(t for t in same if t not in diff), cs, cd, " ".join(cd_ev)))
for t in sorted(diff): print("    %s differs on %d events: %s" % (t, diff[t], " ".join(diff_ev[t][:12])))
EOF

echo ""
echo "=== 4. prep + census against the smx1a..smx9 record"
cd $IMG/pdhd/stm_michel_scan
for a in $PRE_ARM $ARM; do
    [ -d $W/prep_$a ] || ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  --- $a: $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l) payloads"
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_$a.json 2>&1 | grep -E "^census|as shipped"
done
STM_SCAN_RECORD= python3 census_score.py --check 2>&1 | tail -3
echo GATES_DONE

#!/usr/bin/env bash
# doc pdvd/95 -- the gates for the region-based Michel charge (doc 78 action item 9).
#
# Three questions, in this order.  The first two must PASS before the third is worth reading.
#   sec 1  OFF IDENTITY, PDVD.  p95voff (the new binary, no q2d key) vs p93vprod (today's
#          production).  Everything identical: branches, point rows, trees, zips, calib.
#   sec 2  OFF IDENTITY, PDHD.  p95hoff vs p85hoff.  Same bar.  PDHD gets no flip this round, so
#          this gate is the whole of its exposure.
#   sec 3  ADDITIVE.  p95vq2d (region 40 cm + control 35 cm + cells) vs p95voff.  A DIFFERENT
#          assertion in kind: every PRE-EXISTING branch bit-identical, exactly the pre-registered
#          new branches appearing and no branch lost, 0 is_stm flips, 0 michel_found flips, and the
#          census unmoved.  That is what makes the flip additive rather than a physics change.
#
# Why the branch-set line is not optional (doc pdvd/93's defect, inherited deliberately): the
# per-candidate loop iterates the BASELINE's branches, so a branch present only in the new arm is
# structurally invisible to it.  A default-OFF writer does exactly that the moment its knob is
# flipped on.  Here the ON arm adds 49 branches, so a gate without the set comparison would report
# a clean pass while being blind to the entire deliverable.
#
# Fork by duplication (CLAUDE.md M10) of d93_gates.sh.
# Usage: bash d95_gates.sh > /home/xqian/tmp/p95/gates.log 2>&1; echo rc=$?
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p95
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json
export STM_SCAN_RECORD=$REC
VOFF=${VOFF:-p95voffb}; VON=${VON:-p95vq2db}; VPROD=${VPROD:-p93vprod}
HOFF=${HOFF:-p95hoffb}; HBASE=${HBASE:-p85hoff}
# the pin the arms actually ran on.  libpin_p95b is binary B (the own_blob column); the first
# wave ran on libpin_p95 = binary A and its arms are superseded -- a gate must be taken on the
# binary that SHIPS, not on a near-identical earlier one.
PINDIR=${PINDIR:-libpin_p95b}
cd $IMG
echo "record: $REC"
[ -s "$REC" ] || { echo "ABORT: record missing"; exit 2; }

echo "=== 0. completeness, pin, loader deaths"
for a in $VPROD $VOFF $VON; do
    n=$(ls -d pdvd/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in pdvd/work/*_$a; do [ -s $e/tracking-pr.root ] && echo 1; done 2>/dev/null | wc -l)
    lt=$(grep -l "file too short" pdvd/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  pdvd $a dirs $n with tracking-pr.root $ok loader-deaths $lt"
done
for a in $HBASE $HOFF; do
    n=$(ls -d pdhd/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in pdhd/work/*_$a; do [ -s $e/tracking-pr.root ] && echo 1; done 2>/dev/null | wc -l)
    echo "  pdhd $a dirs $n with tracking-pr.root $ok"
done
# DEFECT FIXED (doc 95): this line hard-coded libpin_p95 (binary A) while the before-manifest
# was written from libpin_p95b (binary B, the one the arms actually ran).  It therefore compared
# two DIFFERENT pin directories and printed "PIN CHANGED ... every comparison below is VOID"
# over three gates that had in fact passed.  The tell was that exactly ONE of 572 hashes differed
# -- the Clus library -- which is two pins of the same tree, not a library swapped mid-run.
# Same species as doc pdvd/93's gate defect: it failed SAFE, and it still had to be fixed rather
# than argued away in prose.  Both logs are kept (gates.log = the defect, gates2.log = corrected).
md5sum $W/$PINDIR/*.so* > $W/libpin_md5_after.txt 2>/dev/null
cmp -s $W/libpin_md5_before.txt $W/libpin_md5_after.txt \
    && echo "  pin $PINDIR: $(wc -l < $W/libpin_md5_after.txt) libraries, md5 identical before and after" \
    || echo "  *** PIN CHANGED between launch and gates -- every comparison below is VOID ***"

VOFF=$VOFF VON=$VON VPROD=$VPROD HOFF=$HOFF HBASE=$HBASE REC=$REC python3 - <<'EOF'
import collections, glob, hashlib, json, os, sys, zipfile
import numpy as np, uproot
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C
E = os.environ
R = {v["key"]: v for v in json.load(open(E["REC"]))}

# PRE-REGISTERED: the branches the ON arm adds.  doc pdvd/81's group (23) plus doc pdvd/95's
# region (16) and body control (10).  Listed so that an UNEXPECTED new branch fails the gate
# instead of being waved through as "additive".
Q2D81 = ["michel_q2d_valid", "michel_q2d_reason", "michel_q2d_dropped_plane",
         "michel_q2d_u", "michel_q2d_v", "michel_q2d_w",
         "michel_q2d_mu_u", "michel_q2d_mu_v", "michel_q2d_mu_w",
         "michel_q2d_n_u", "michel_q2d_n_v", "michel_q2d_n_w",
         "michel_q2d_nx_u", "michel_q2d_nx_v", "michel_q2d_nx_w",
         "michel_q2d_raw_u", "michel_q2d_raw_v", "michel_q2d_raw_w",
         "michel_q2d", "michel_q2d_gamma",
         "michel_ke_q2d", "michel_ke_q2d_gamma", "michel_ke_q2d_total"]
REGION = ["michel_q2d_region_u", "michel_q2d_region_v", "michel_q2d_region_w",
          "michel_q2d_region_mu_u", "michel_q2d_region_mu_v", "michel_q2d_region_mu_w",
          "michel_q2d_region_n_u", "michel_q2d_region_n_v", "michel_q2d_region_n_w",
          "michel_q2d_region_nd_u", "michel_q2d_region_nd_v", "michel_q2d_region_nd_w",
          "michel_q2d_region_dropped_plane", "michel_q2d_n_role0",
          "michel_q2d_region", "michel_ke_q2d_region"]
CTL = ["michel_q2d_ctl_u", "michel_q2d_ctl_v", "michel_q2d_ctl_w",
       "michel_q2d_ctl_n_u", "michel_q2d_ctl_n_v", "michel_q2d_ctl_n_w",
       "michel_q2d_ctl_dropped_plane", "michel_q2d_ctl_valid",
       "michel_q2d_ctl", "michel_ke_q2d_ctl"]
NEW_EXPECTED = sorted(Q2D81 + REGION + CTL)

def read(det, arm):
    out, pts = {}, {}
    for d in sorted(glob.glob(det + "/work/*_" + arm)):
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

def compare(det, arm, base, label, mode):
    A, AP = read(det, arm)
    B, BP = read(det, base)
    print("\n=== %s: %s vs %s (%d vs %d candidates)" % (label, arm, base, len(A), len(B)))
    ba = {b for v in A.values() for b in v}; bb = {b for v in B.values() for b in v}
    only_a, only_b = sorted(ba - bb), sorted(bb - ba)
    print("    T_stm_michel branches: %s %d, %s %d | only in %s: %s | only in %s: %s" % (
        arm, len(ba), base, len(bb), arm, only_a or "none", base, only_b or "none"))
    for k in sorted(set(A) - set(B)): print("    NEW candidate %s" % k)
    for k in sorted(set(B) - set(A)): print("    DROPPED candidate %s" % k)
    # the loop iterates the BASELINE's branches -- for the additive mode that is exactly the
    # question ("did any pre-existing branch move?"), and the set line above covers the rest.
    moved, bset = {}, collections.Counter()
    for k in sorted(set(A) & set(B)):
        bs = sorted(b for b in B[k] if not same(A[k].get(b), B[k].get(b)))
        rows = AP[k] != BP[k]
        if bs or rows:
            moved[k] = (bs, rows); bset.update(bs)
    ident = len(set(A) & set(B)) - len(moved)
    print("    candidates bit-identical on every PRE-EXISTING branch and point row: %d / %d" % (ident, len(set(A) | set(B))))
    for k, (bs, rows) in list(moved.items())[:25]:
        a, b = A[k], B[k]
        print("    %-14s %-13s is_stm %d->%d mf %d->%d bits %d->%d | rows moved %s | branches %s" % (
            k, tag(k), g(b, "is_stm"), g(a, "is_stm"), g(b, "michel_found"), g(a, "michel_found"),
            g(b, "reject_bits"), g(a, "reject_bits"), rows, " ".join(bs)))
    fl = sorted(k for k in moved if g(A[k], "is_stm") != g(B[k], "is_stm"))
    mf = sorted(k for k in moved if g(A[k], "michel_found") != g(B[k], "michel_found"))
    print("    candidates moved %d; branches moved: %s" % (len(moved), dict(sorted(bset.items()))))
    print("    is_stm flips %d: %s" % (len(fl), " ".join(fl) or "none"))
    print("    michel_found flips %d: %s" % (len(mf), " ".join(mf) or "none"))
    zd = []
    for eb in sorted(glob.glob(det + "/work/*_" + base)):
        ev = os.path.basename(eb)[: -len(base) - 1]; ea = "%s/work/%s_%s" % (det, ev, arm)
        if os.path.exists(eb + "/mabc-pr.zip") and os.path.exists(ea + "/mabc-pr.zip") and members(eb + "/mabc-pr.zip") != members(ea + "/mabc-pr.zip"):
            zd.append(ev)
    print("    zips differing on %d events: %s" % (len(zd), " ".join(zd) or "none"))
    if mode == "identical":
        ok = not moved and not zd and set(A) == set(B) and not only_a and not only_b
        print("    VERDICT: %s" % ("BYTE-IDENTICAL -- the knob-off path is unchanged by this round's C++"
                                   if ok else "*** NOT IDENTICAL: the OFF path moved, the round is not safe ***"))
    else:
        ok = (not moved and not zd and set(A) == set(B) and not only_b and only_a == NEW_EXPECTED)
        print("    new branches expected %d, got %d; unexpected: %s; missing: %s" % (
            len(NEW_EXPECTED), len(only_a),
            sorted(set(only_a) - set(NEW_EXPECTED)) or "none", sorted(set(NEW_EXPECTED) - set(only_a)) or "none"))
        print("    VERDICT: %s" % ("PURELY ADDITIVE -- every pre-existing branch, point row, zip and verdict unmoved"
                                   if ok else "*** NOT ADDITIVE: the ON arm moved something that already existed ***"))
    # what the new branches actually say (ON arm only)
    if mode == "additive" and A:
        v = [a for a in A.values() if "michel_q2d_valid" in a]
        if v:
            nval = sum(1 for a in v if int(a["michel_q2d_valid"]) == 1)
            reasons = collections.Counter(int(a["michel_q2d_reason"]) for a in v)
            print("    michel_q2d_valid == 1 on %d / %d candidates; reason codes %s" % (nval, len(v), dict(sorted(reasons.items()))))
        r0 = [int(a.get("michel_q2d_n_role0", 0)) for a in A.values() if "michel_q2d_n_role0" in a]
        if r0:
            print("    role-0 cells per candidate (the population the association drops): median %d, max %d, zero on %d" % (
                int(np.median(r0)), max(r0), sum(1 for x in r0 if x == 0)))

compare("pdvd", E["VOFF"], E["VPROD"], "1. OFF IDENTITY (PDVD)", "identical")
compare("pdhd", E["HOFF"], E["HBASE"], "2. OFF IDENTITY (PDHD)", "identical")
compare("pdvd", E["VON"], E["VOFF"], "3. ADDITIVE (the ON arm)", "additive")
EOF

echo ""
echo "=== 4. trees and calib: $VOFF vs $VPROD (the OFF path must be identical on every event)"
DET=pdvd ARM=$VOFF BASE=$VPROD python3 - <<'EOF'
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
# census_score prints the 546 items that HAVE a candidate, where today's production reads
# is_stm 242 / 8 / 34 (eff 0.877) and michel 144 / 12 / 20.  Doc pdvd/93's headline 242 / 8 / 44
# (eff 0.846) is the OTHER denominator -- all 576 judged items, where the 10 the tagger never
# hands on count as misses (doc pdvd/89 sec 1).  Both are correct; an earlier version of this
# comment quoted the 576 figure beside the 546 output and so looked like a mismatch.
echo "=== 5. prep + census against the smx1a..smx9 record (546 with a candidate: is_stm 242/8/34, michel 144/12/20)"
cd $IMG/pdhd/stm_michel_scan
for a in $VOFF $VON; do
    [ -d $W/prep_$a ] || ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  --- $a: $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l) payloads"
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_$a.json 2>&1 | grep -E "^census|as shipped"
done
echo GATES_DONE

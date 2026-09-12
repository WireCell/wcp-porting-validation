#!/usr/bin/env bash
# doc pdvd/96 -- the gates for the region's SCOPE (doc pdvd/95 sec 9 item 1).
#
# FOUR questions, in this order.  The first must PASS before any other is worth reading.
#   sec 1  OFF IDENTITY, PDVD.  p96voff (the new binary, no q2d key) vs p95vprodb.  NOTE the
#          baseline is NOT p93vprod any more: doc pdvd/95 flipped four keys into
#          wct-pr-perevt.jsonnet, so today's production is what p95vprodb ran.  Using the old
#          baseline here would compare against a config that no longer exists.
#   sec 2  OFF IDENTITY, PDHD.  p96hoff vs p95hoffb.  PDHD gets no flip this round, so this
#          gate is the whole of its exposure.
#   sec 3  CROSS-BINARY INERTNESS (nice-to-have).  p96vbase vs p95vq2db -- the same R=40 scope-0
#          config on two different binaries, which also proves michel_q2d_region_scope written
#          explicitly at its C++ default 0 changes nothing.  Registered in pred.txt as X2 with
#          its fallback: doc 95's pin was DELETED, so a diff here cannot be attributed to my
#          knob rather than to compiler/env drift.  On a diff, scope 0 is derived offline from
#          p96vscope's own unfiltered cell table and the scope comparison rests on ONE arm and
#          ONE binary.  A diff is REPORTED, not chased, and does not by itself block the flip.
#   sec 4  THE SCOPE EFFECT.  p96vscope vs p96vbase: same binary, same radius, only the knob.
#          This is deliberately NOT the "additive" assertion doc 95 used -- the whole point of
#          this round is that the region scalars MOVE.  The assertion is instead that they are
#          the ONLY things that move, against a list pre-registered before the arm ran.
#
# TWO BRANCHES ARE FROZEN AND THE GATE SAYS SO (pred.txt X3).  michel_q2d_region_nd_* (dead
# cells) and michel_q2d_n_role0 are deliberately hoisted OUT of the scope filter so they keep
# meaning "the region", not "the cells summed".  If either moves, the hoist is wrong and the
# round stops -- so they are checked as FROZEN rather than merely omitted from the allowed list.
#
# Why the branch-set line is not optional (doc pdvd/93's defect, inherited deliberately): the
# per-candidate loop iterates the BASELINE's branches, so a branch present only in the new arm is
# structurally invisible to it.  This round adds no branch, so the set line must come back EMPTY
# on both sides -- a non-empty one means I changed the schema without meaning to.
#
# Fork by duplication (CLAUDE.md M10) of d95_gates.sh.
# Usage: bash d96_gates.sh > /home/xqian/tmp/p96/gates.log 2>&1; echo rc=$?
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p96
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json
export STM_SCAN_RECORD=$REC
VOFF=${VOFF:-p96voff}; VBASE=${VBASE:-p96vbase}; VSCOPE=${VSCOPE:-p96vscope}
VPROD=${VPROD:-p95vprodb}; VOLD=${VOLD:-p95vq2db}
HOFF=${HOFF:-p96hoff}; HBASE=${HBASE:-p95hoffb}
# the pin the arms actually ran on.  Read from the VARIABLE, never hard-coded: doc 95's gate
# hard-coded libpin_p95 while its manifest came from libpin_p95b and printed "PIN CHANGED ...
# VOID" over three passing gates.  Exactly one of 572 hashes differed -- the tell of two pins of
# the same tree, not a library swapped mid-run.
PINDIR=${PINDIR:-libpin_p96}
cd $IMG
echo "record: $REC"
[ -s "$REC" ] || { echo "ABORT: record missing"; exit 2; }

echo "=== 0. completeness, pin, loader deaths"
for a in $VPROD $VOLD $VOFF $VBASE $VSCOPE; do
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
md5sum $W/$PINDIR/*.so* > $W/libpin_md5_after.txt 2>/dev/null
cmp -s $W/libpin_md5_before.txt $W/libpin_md5_after.txt \
    && echo "  pin $PINDIR: $(wc -l < $W/libpin_md5_after.txt) libraries, md5 identical before and after" \
    || echo "  *** PIN CHANGED between launch and gates -- every comparison below is VOID ***"

VOFF=$VOFF VBASE=$VBASE VSCOPE=$VSCOPE VPROD=$VPROD VOLD=$VOLD HOFF=$HOFF HBASE=$HBASE REC=$REC python3 - <<'EOF'
import collections, glob, hashlib, json, os, sys, zipfile
import numpy as np, uproot
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C
E = os.environ
R = {v["key"]: v for v in json.load(open(E["REC"]))}

# PRE-REGISTERED in /home/xqian/tmp/p96/pred.txt (X3), BEFORE the arms ran: the only branches the
# scope knob may move.  An unexpected mover FAILS the gate rather than being explained afterwards.
ALLOWED = sorted([
    "michel_q2d_region_u", "michel_q2d_region_v", "michel_q2d_region_w",
    "michel_q2d_region_mu_u", "michel_q2d_region_mu_v", "michel_q2d_region_mu_w",
    "michel_q2d_region_n_u", "michel_q2d_region_n_v", "michel_q2d_region_n_w",
    "michel_q2d_region_dropped_plane", "michel_q2d_region", "michel_ke_q2d_region",
    "michel_q2d_ctl_u", "michel_q2d_ctl_v", "michel_q2d_ctl_w",
    "michel_q2d_ctl_n_u", "michel_q2d_ctl_n_v", "michel_q2d_ctl_n_w",
    "michel_q2d_ctl_dropped_plane", "michel_q2d_ctl", "michel_ke_q2d_ctl"])
# these must NOT move: hoisted out of the filter on purpose so they keep counting the REGION
FROZEN = ["michel_q2d_region_nd_u", "michel_q2d_region_nd_v", "michel_q2d_region_nd_w",
          "michel_q2d_n_role0", "michel_q2d_ctl_valid"]

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
    if not A or not B:
        print("    *** ONE SIDE IS EMPTY -- this is a dead net, not a pass ***"); return
    ba = {b for v in A.values() for b in v}; bb = {b for v in B.values() for b in v}
    only_a, only_b = sorted(ba - bb), sorted(bb - ba)
    print("    T_stm_michel branches: %s %d, %s %d | only in %s: %s | only in %s: %s" % (
        arm, len(ba), base, len(bb), arm, only_a or "none", base, only_b or "none"))
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
    for k, (bs, rows) in list(moved.items())[:15]:
        a, b = A[k], B[k]
        print("    %-14s %-13s is_stm %d->%d mf %d->%d bits %d->%d | rows moved %s | branches %s" % (
            k, tag(k), g(b, "is_stm"), g(a, "is_stm"), g(b, "michel_found"), g(a, "michel_found"),
            g(b, "reject_bits"), g(a, "reject_bits"), rows, " ".join(bs[:8])))
    fl = sorted(k for k in moved if g(A[k], "is_stm") != g(B[k], "is_stm"))
    mf = sorted(k for k in moved if g(A[k], "michel_found") != g(B[k], "michel_found"))
    rowmoved = sorted(k for k in moved if moved[k][1])
    print("    candidates moved %d; branches moved: %s" % (len(moved), dict(sorted(bset.items()))))
    print("    is_stm flips %d: %s" % (len(fl), " ".join(fl) or "none"))
    print("    michel_found flips %d: %s" % (len(mf), " ".join(mf) or "none"))
    print("    point rows moved on %d candidates" % len(rowmoved))
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
    elif mode == "crossbin":
        ok = not moved and not zd and set(A) == set(B) and not only_a and not only_b
        print("    VERDICT: %s" % ("BYTE-IDENTICAL ACROSS BINARIES -- the new key at its default 0 is inert"
                                   if ok else "NOT IDENTICAL -- NICE-TO-HAVE gate (pred.txt X2): doc 95's pin was deleted so "
                                              "drift cannot be excluded; fall back to deriving scope 0 offline from the "
                                              "scope arm's own cell table.  Reported, not chased."))
    else:  # scoped
        unexpected = sorted(set(bset) - set(ALLOWED))
        froze = sorted(b for b in FROZEN if b in bset)
        print("    movers allowed by pre-registration: %d of %d seen" % (len(set(bset) & set(ALLOWED)), len(ALLOWED)))
        print("    UNEXPECTED movers: %s" % (unexpected or "none"))
        print("    FROZEN branches that moved (must be none): %s" % (froze or "none"))
        ok = (not unexpected and not froze and not zd and set(A) == set(B)
              and not only_a and not only_b and not fl and not mf and not rowmoved)
        print("    VERDICT: %s" % ("SCOPED -- only the pre-registered region/control branches moved; "
                                   "no verdict, no point row, no zip, no frozen counter"
                                   if ok else "*** THE SCOPE KNOB MOVED SOMETHING IT WAS NOT REGISTERED TO MOVE ***"))

# RUNCMP selects which comparisons to run, so gate 1 -- the load-bearing OFF-identity check --
# can be taken the moment its two arms are complete, without waiting for the later waves.  The
# final, authoritative run uses the default (all four) and is the one quoted in the doc.
sel = set(x.strip() for x in os.environ.get("RUNCMP", "1,2,3,4").split(",") if x.strip())
if "1" in sel: compare("pdvd", E["VOFF"],   E["VPROD"], "1. OFF IDENTITY (PDVD, vs today's production)", "identical")
if "2" in sel: compare("pdhd", E["HOFF"],   E["HBASE"], "2. OFF IDENTITY (PDHD)", "identical")
if "3" in sel: compare("pdvd", E["VBASE"],  E["VOLD"],  "3. CROSS-BINARY INERTNESS (nice-to-have)", "crossbin")
if "4" in sel: compare("pdvd", E["VSCOPE"], E["VBASE"], "4. THE SCOPE EFFECT (same binary, same radius)", "scoped")
EOF

echo ""
echo "=== 4b. the deliverable is in the CELL table, which the branch gate above cannot see"
VSCOPE=$VSCOPE VBASE=$VBASE python3 - <<'EOF'
import glob, os, collections
import numpy as np, uproot
E = os.environ
def owncounts(arm):
    c = collections.Counter(); nfile = 0
    for d in sorted(glob.glob("/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn) or os.path.getsize(fn) < 100000: continue
        try: f = uproot.open(fn)
        except Exception: continue
        if "T_stm_michel_2d" not in f: continue
        t = f["T_stm_michel_2d"]
        if t.num_entries == 0: continue
        ob = t.arrays(["own_blob"], library="np")["own_blob"]
        u, n = np.unique(ob, return_counts=True)
        for a, b in zip(u, n): c[int(a)] += int(b)
        nfile += 1
    return c, nfile
for arm in (E["VBASE"], E["VSCOPE"]):
    c, nf = owncounts(arm)
    tot = sum(c.values())
    if not tot:
        # an `or 1` fallback here printed "0 files, 1 cells" -- a fabricated count in a
        # doc-facing log.  Say plainly that there is nothing to compare instead.
        print("  %-11s %3d files, NO CELLS -- arm has no output yet, nothing to compare" % (arm, nf))
        continue
    print("  %-11s %3d files, %8d cells, own_blob %s" % (
        arm, nf, tot, {k: "%d (%.1f%%)" % (v, 100.0 * v / tot) for k, v in sorted(c.items())}))
print("  bit 4 (a preloaded FITTED companion) exists only at scope > 0.  doc pdvd/95 shipped bits")
print("  1 and 2 only, and bit 2 never fired on any of 596 PDVD candidates -- so if bit 4 is also")
print("  absent here, 'own' means the MAIN CLUSTER ALONE and the doc says so rather than claiming")
print("  doc 78 item 9's scope.  This line is the positive control for the whole round.")
EOF

echo ""
echo "=== 5. trees and calib: $VOFF vs $VPROD (the OFF path must be identical on every event)"
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
# hands on count as misses (doc pdvd/89 sec 1).  Both are correct; they are different questions.
echo "=== 6. prep + census (546 with a candidate: is_stm 242/8/34, michel 144/12/20 -- must not move)"
cd $IMG/pdhd/stm_michel_scan
for a in $VOFF $VSCOPE; do
    # Refuse to build a prep dir for an arm that has no output yet.  Without this, running the
    # gate early (to take gate 1 as soon as its arms are complete) would create an EMPTY
    # prep_<arm>/, and the later authoritative run would see the directory exist and SKIP
    # rebuilding it -- scoring the round against nothing while looking like a pass.  Same
    # species as a killed table-builder leaving a short file that `[ -f ]` then reuses.
    n=$(ls $IMG/pdvd/work/*_$a/tracking-pr.root 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "  --- $a: NO ARM OUTPUT -- skipping prep (a dir built now would be reused empty later)"
        continue
    fi
    [ -d $W/prep_$a ] || ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  --- $a: $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l) payloads from $n events"
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_$a.json 2>&1 | grep -E "^census|as shipped"
done
echo GATES_DONE

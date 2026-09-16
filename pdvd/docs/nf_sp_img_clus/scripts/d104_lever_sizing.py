#!/usr/bin/env python3
"""doc pdvd/104 sec 2 -- size every candidate Michel-admission lever on the real population (read-only, no chain run).

Doc 103 sec 14.4b set round 7 the task "build an admission rule on sec 10.4's class -- default OFF with a
byte-identical gate, graded on both detectors so PDVD keeps its D1".  This script tests whether such a rule EXISTS
before any C++ is written.  It is an OFFLINE PREDICATE: michel_found is zeroed after the fact when the lever vetoes;
is_stm is left as the arm wrote it.  A real knob at the admission site can differ (a rejected Michel may change what
else the tagger decides), so a lever that passed here would still have to be re-derived from an arm -- but a lever
that FAILS here cannot be rescued by moving it into C++, which is the result sec 2 reports.

FRAMING (doc 103 sec 12.4 reports two; only one describes a shippable unit).  The lever would ship WITH the flip, so
production A0 has no knob and stays exactly as it runs today.  Every row below is therefore

    A1 + lever   vs   A0 unchanged,

and the bar is doc 103 figs/103_pred.txt: every metric >= A0 - 0.020.  The "floor on both arms" framing of doc 103
sec 12.4 is NOT repeated here -- it measures a knob nobody would ship, and it always fails for the wrong reason
(the floor also cleans up A0's false positives and so raises the bar it is judged against).

THE FOUR LEVERS, and why each one is the natural next thing to try:
  L1 energy floor      michel_ke_best >= E.  Refuted already in doc 103 sec 12.4; repeated here at the same
                       settings as the V1 cross-check that this script's population matches the grader's.
  L2 length floor      michel_len >= L.  The obvious successor to L1 -- "a Michel must be a resolvable object".
                       Sized twice: pooled over all conn types, and restricted to conn 1, because michel_len is ~0
                       BY CONSTRUCTION for conn 2/3 (no fitted seed arm).  The pooled version is the trap.
  L3 continuation veto conn 1 and len >= L and mip >= M and kink < K.  The physics: a MIP-like arm that barely
                       turns is muon continuation, not a Michel.  This is the same statement the gate's own
                       continuation clause makes (StmMichelFunctions.cxx:620-625, kink < 20 deg, len > 3 cm,
                       0.7 <= mip <= 1.3); the lever asks whether widening it catches the false positives.
  L4 bragg at stop     require the muon's own stopping signature where the Michel is attached:
                       bragg_valid and contrast >= 0.6 * contrast_expected -- the tagger's own bragg_here test
                       (CheckSTM_Michel.cxx:3505-3506).  The physics: if the muon did not stop here, whatever is
                       attached here is not a Michel.

SCOPE TRAP the table prints for each lever: michel_mip and michel_kink_deg are measured only on the attached path.
For conn 2 (bridged) and conn 3 (charge only) they are 0.00 and -1 for every row, TP and FP alike, so any lever
using them silently exempts those objects -- 3 of PDHD's 9 false positives and 21 of its 77 true Michels.

    python3 d104_lever_sizing.py > figs/104_lever_sizing.txt
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d104_michel_features as F

LIMIT = -0.020
# doc 103 figs/103_union_grade_pdhd_own103h_stmonlyneg.txt and figs/103_union_grade_pdvd_own103v.txt
V1 = {"pdhd": dict(tp0=70, fp0=4, tp1=77, fp1=9, pop=196, pos=116),
      "pdvd": dict(tp0=158, fp0=13, tp1=171, fp1=11, pop=606, pos=239)}


def grade(D, keep):
    """purity/efficiency of A1 after the lever, against production A0 unchanged."""
    tp, fp = F.split(D)
    t = sum(1 for k, v in tp if keep(v))
    f = sum(1 for k, v in fp if keep(v))
    return t, f, t / max(1, t + f), t / max(1, D["nP"]), len(tp) - t, len(fp) - f


def line(D, label, keep):
    t, f, p, e, tl, fr = grade(D, keep)
    dp, de = p - D["p0"], e - D["e0"]
    ok = "PASSES" if min(dp, de) >= LIMIT else "fails"
    print(f"   {label:52s} TP {t:3d} FP {f:2d} (lost {tl:2d}, removed {fr}) purity {p:.3f} ({dp:+.3f}) "
          f"eff {e:.3f} ({de:+.3f})  {ok}")


def anatomy(D, L):
    """The one lever that clears the arithmetic bar on both detectors -- and what it actually does.

    L2b at L = 3.0 cm passes doc 103's bar on PDHD (purity -0.012, efficiency +0.009) and on PDVD (+0.028, +0.000).
    Before that is called a fix, ask the three questions the bar cannot ask, because a lever found by SCANNING the
    same data it is graded on is exactly what a pre-registered threshold does not protect against:

      (1) does it remove the class round 7 was aimed at -- the A1-only false positives -- or the ones production
          already makes?  Removing a SHARED false positive improves the A1 - A0 delta without touching the
          regression, because A0 keeps it.
      (2) how many true Michels does it discard per false positive removed?  A discriminator separates; a cut that
          removes more true objects than false ones is just shrinking a region that is dense in both.
      (3) what does it cost the detector that already passes?  PDVD banked Michel efficiency +0.054 in doc 103
          sec 12; a lever that hands it back is a regression there to fix a regression here.
    """
    tp, fp = F.split(D)
    A0 = D["A0"]
    keep = lambda x: x["michel_conn_type"] != 1 or x["michel_len"] >= L
    rm_only = [k for k, v in fp if not keep(v) and not A0.get(k, {}).get("michel_found", 0)]
    rm_shared = [k for k, v in fp if not keep(v) and A0.get(k, {}).get("michel_found", 0)]
    surv = [k for k, v in fp if keep(v) and not A0.get(k, {}).get("michel_found", 0)]
    lost = sum(1 for _, v in tp if not keep(v))
    nrm = len(rm_only) + len(rm_shared)
    print(f"\n   -- anatomy of the one lever that clears the bar on both detectors (L2b, L = {L} cm)")
    print(f"      false positives removed: {nrm}")
    print(f"        of the A1-ONLY class this round targets: {len(rm_only)}  {rm_only}")
    print(f"        SHARED with production A0, which keeps them: {len(rm_shared)}  {rm_shared}")
    print(f"      A1-only false positives SURVIVING the lever: {len(surv)}  {surv}")
    print(f"      true Michels discarded: {lost}  ({lost / max(1, nrm):.1f} per false positive removed)")
    print(f"      joint passing window across both detectors: L in [3.0, 3.2] cm (PDHD fails at 2.8 and at 4.0, "
          f"PDVD fails at 3.5) -- 0.2 cm wide, on a sample of 61 + 120 events")


def main():
    print("# doc pdvd/104 sec 2: can ANY cut on the persisted admission geometry fix PDHD's Michel purity?")
    print("# Offline predicate, michel_found := michel_found and <lever>; is_stm untouched.")
    print("# Every row is A1 + lever vs production A0 UNCHANGED (the shippable framing); bar is every metric >= A0 - 0.020.")
    for det in ("pdhd", "pdvd"):
        D = F.load(det)
        tp, fp = F.split(D)
        v = V1[det]
        print(f"\n===== {det}: A0 {D['cells'][0]} purity {D['p0']:.3f} eff {D['e0']:.3f} "
              f"| A1 {D['cells'][1]} TP {len(tp)} FP {len(fp)} =====")
        # V1: the population and truth join must reproduce the committed grade before any lever is read
        got = (D["tp0"], D["fp0"], len(tp), len(fp), len(D["P"]), D["nP"])
        exp = (v["tp0"], v["fp0"], v["tp1"], v["fp1"], v["pop"], v["pos"])
        assert got == exp, f"V1 FAILED for {det}: A0 TP/FP, A1 TP/FP, pop, positives = {got}, expected {exp}"
        print(f"   V1 ok: reproduces the committed grade (A0 {v['tp0']}/{v['fp0']}, A1 {v['tp1']}/{v['fp1']}, "
              f"population {v['pop']}, hand positives {v['pos']})")
        for conn in (1, 2, 3):
            n_t = sum(1 for _, x in tp if x["michel_conn_type"] == conn)
            n_f = sum(1 for _, x in fp if x["michel_conn_type"] == conn)
            if n_t or n_f:
                meas = any(x["michel_kink_deg"] >= 0 for _, x in tp + fp if x["michel_conn_type"] == conn)
                print(f"   conn {conn}: TP {n_t:3d} FP {n_f}  mip/kink measured on this path: {'yes' if meas else 'NO'}")

        print("\n   -- L1 energy floor: michel_ke_best >= E")
        for E in (1, 2, 4, 6, 8, 10, 12, 15):
            line(D, f"ke >= {E:4.1f} MeV", lambda x, E=E: x["michel_ke_best"] >= E)

        print("\n   -- L2a length floor pooled over all conn types: michel_len >= L  (the trap)")
        for L in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            line(D, f"len >= {L:4.1f} cm (all conn)", lambda x, L=L: x["michel_len"] >= L)
        print("   -- L2b the same floor applied only where michel_len means something (conn 1)")
        for L in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            line(D, f"len >= {L:4.1f} cm (conn 1 only)",
                 lambda x, L=L: x["michel_conn_type"] != 1 or x["michel_len"] >= L)

        print("\n   -- L3 continuation veto: reject conn 1 with len >= L, mip >= M, kink < K")
        for L, M, K in ((3, 0.7, 20), (3, 0.7, 30), (6, 1.0, 40), (8, 0.9, 45), (10, 1.0, 40), (12, 1.0, 40)):
            line(D, f"veto len>={L} mip>={M} kink<{K}",
                 lambda x, L=L, M=M, K=K: not (x["michel_conn_type"] == 1 and x["michel_len"] >= L
                                               and x["michel_mip"] >= M and 0 <= x["michel_kink_deg"] < K))

        print("\n   -- L4 require the muon's own Bragg peak at the stop (the tagger's bragg_here test)")
        for frac in (0.6, 0.8, 1.0):
            line(D, f"bragg_valid and contrast >= {frac} * expected",
                 lambda x, f=frac: x["bragg_valid"] > 0 and x["contrast_expected"] > 0
                 and x["contrast"] >= f * x["contrast_expected"])

        anatomy(D, 3.0)


if __name__ == "__main__":
    main()

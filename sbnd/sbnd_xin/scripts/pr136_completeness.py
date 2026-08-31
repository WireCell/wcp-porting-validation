#!/usr/bin/env python3
"""doc pr/136 -- combine the two em117_score runs and cross them with the pi0 residual.

em117_score.py scores ONE label tag against ONE arm.  The hand-marked
population spans both tags (98-set `emscan-0827`, 141-set
`emscan-0828-agent5`), so the campaign-level completeness number needs the
two runs merged.  This script does that, and then does the thing neither
half can do alone: it asks which of the charge-attribution failures are
also pi0 blockers, by joining on `pr136-mass-closure.tsv`.

The metric is em117_score's, unchanged: per hand-marked shower,
`target = (members | marked-in) - marked-out`, charge-weighted against the
segments the arm actually gave that shower, giving `q_comp` (completeness),
`q_pur` (purity) and their harmonic mean `q_f1`.  Under-clustering is
`q_miss`, over-clustering is `q_extra`, both in raw charge.

OPERATING POINT.  Pass --src98/--src141 to choose it, and READ THE BANNER:
the defaults are the pr130r1-probe arms (before the NC chain, K24 and the
0.86 EM scale), while `pr136-completeness-f086-*.tsv` are the f086
PRODUCTION point minted by proposal 0's probe arm (scripts/pr136_arms.sh).
Match --closure to the same arm or the join is a cross-arm join.  The
sidecar matters: the dump's `segments[].shower_id` is single-valued, so a
segment held by two showers is credited to one and the lossy join invents
misses that are not there.

Repro (the two scoring runs, then this):
  cd em_display
  ./em117_score.py --tag emscan-0827 --manifest em117-pr130q98-manifest.tsv \
      --prepdir emprep-pr130q98  --tsv ../docs/pr/pr136-completeness-98.tsv
  ./em117_score.py --tag emscan-0828-agent5 --manifest em114c-pr130q141-manifest.tsv \
      --prepdir emprep-pr130q141 --tsv ../docs/pr/pr136-completeness-141.tsv
  cd .. && scripts/pr136_completeness.py --tsv docs/pr/pr136-completeness-pr130arms.tsv

READ-ONLY apart from --tsv.
"""
import argparse
import csv
import os
import statistics as st

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)
D = os.path.join(SX, "docs", "pr")

DEFAULT_SRC98 = "pr136-completeness-98.tsv"
DEFAULT_SRC141 = "pr136-completeness-141.tsv"


def num(r, k):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    # doc pr/136 proposal 0: the f086 probe arm mints a SECOND pair of
    # em117_score outputs.  Defaults reproduce the pr130-arm run exactly,
    # so an argument-free call is byte-identical to before these existed.
    ap.add_argument("--src98", default=DEFAULT_SRC98)
    ap.add_argument("--src141", default=DEFAULT_SRC141)
    ap.add_argument("--closure", default=os.path.join(D, "pr136-mass-closure.tsv"))
    a = ap.parse_args()

    rows = []
    for setlab, name in (("98", a.src98), ("141", a.src141)):
        p = os.path.join(D, name)
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run the em117_score commands in the docstring first" % p)
        for r in csv.DictReader(open(p), delimiter="\t"):
            r["set"] = setlab
            rows.append(r)

    ev = {(r["set"], r["event"]) for r in rows}
    qt = sum(num(r, "q_target") for r in rows)
    qm = sum(num(r, "q_miss") for r in rows)
    qe = sum(num(r, "q_extra") for r in rows)
    f1 = [num(r, "q_f1") for r in rows if num(r, "q_f1") == num(r, "q_f1")]

    # Never hard-code the arm in the banner: this script now runs on two
    # operating points and a stale label is how a cross-arm join hides.
    print("EM SHOWER CHARGE ATTRIBUTION vs THE HAND SCAN")
    print("  inputs: %s + %s" % (a.src98, a.src141))
    print("  pi0 join: %s" % os.path.basename(a.closure))
    print("  hand-marked showers %d over %d events (98-set + 141-set)" % (len(rows), len(ev)))
    print("  sum q_target %.4g" % qt)
    print("  q_miss  (UNDER: charge the scanner says belongs, the shower does not hold) %.4g = %.1f%%"
          % (qm, 100 * qm / qt))
    print("  q_extra (OVER : charge the shower holds, the scanner says it should not)   %.4g = %.1f%%"
          % (qe, 100 * qe / qt))
    print("  charge-weighted F1 per shower: median %.3f  mean %.3f  min %.3f"
          % (st.median(f1), st.mean(f1), min(f1)))
    for t in (0.90, 0.80, 0.50):
        print("     F1 < %.2f : %2d of %d" % (t, sum(1 for v in f1 if v < t), len(f1)))

    pu = [r for r in rows if num(r, "q_miss") > 0 and num(r, "q_extra") == 0]
    po = [r for r in rows if num(r, "q_extra") > 0 and num(r, "q_miss") == 0]
    pb = [r for r in rows if num(r, "q_extra") > 0 and num(r, "q_miss") > 0]
    pc = [r for r in rows if num(r, "q_extra") == 0 and num(r, "q_miss") == 0]
    print("\n  SHAPE OF THE ERROR -- it is not one-sided:")
    print("     pure UNDER %2d | pure OVER %2d | BOTH %2d | clean %2d"
          % (len(pu), len(po), len(pb), len(pc)))

    print("\n  WORST 12 SHOWERS BY CHARGE-WEIGHTED F1")
    print("  %-4s %-8s %-9s %7s %7s %7s %11s %11s"
          % ("set", "event", "matched", "q_f1", "q_comp", "q_pur", "q_miss", "q_extra"))
    worst = sorted(rows, key=lambda r: (num(r, "q_f1") if num(r, "q_f1") == num(r, "q_f1") else 9))[:12]
    for r in worst:
        print("  %-4s %-8s %-9s %7.3f %7.3f %7.3f %11.3g %11.3g"
              % (r["set"], r["event"], r.get("matched", ""), num(r, "q_f1"),
                 num(r, "q_comp"), num(r, "q_pur"), num(r, "q_miss"), num(r, "q_extra")))

    # ---- the synthesis: which attribution failures are also pi0 blockers
    if os.path.exists(a.closure):
        cl = {r["event"]: r for r in csv.DictReader(open(a.closure), delimiter="\t")}
        print("\n  INTERSECTION WITH THE pi0 RESIDUAL (join on event, pr136-mass-closure.tsv)")
        print("  %-8s %7s %7s %8s %8s %8s  %s"
              % ("event", "q_f1", "q_comp", "R_prod", "m_prod", "R_marks", "note"))
        hits = 0
        for r in sorted(rows, key=lambda r: num(r, "q_f1")):
            c = cl.get(r["event"])
            if not c or float(c["R_prod"]) >= 1.0:
                continue
            hits += 1
            note = ("rescued by hand marks" if float(c["R_marks"]) >= 1.0
                    else "still impossible with marks")
            print("  %-8s %7.3f %7.3f %8s %8s %8s  %s"
                  % (r["event"], num(r, "q_f1"), num(r, "q_comp"),
                     c["R_prod"], c["m_prod"], c["R_marks"], note))
        print("  --> %d hand-marked showers sit in events whose hand pi0 pair is"
              " kinematically impossible" % hits)
    else:
        print("\n  (no %s -- run pr136_mass_closure.py for the intersection)" % a.closure)

    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        cols = list(rows[0].keys())
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        print("\nwrote %s (%d rows)" % (p, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

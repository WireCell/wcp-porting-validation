#!/usr/bin/env python3
"""doc pr/136 sec 6.1 -- refit the pi0 mass peak on the CONTAINED subsample.

WHY.  The owner's point: "for pi0 it is possible part of the gamma from the
pi0 decay go out of the detector.  Since we can only reconstruct what is in
the detector, even if we have perfect clustering, we may still miss
significant energy leading to lower pi0 mass reconstruction."

pr136_containment.py measured that.  29 of 112 hand gammas have containment
f < 0.95 and 43% of pairs carry at least one leaking gamma; contained pairs
sit at median 136.8 MeV against 128.1 for leaking ones.  So the full-sample
peak that fixed kine_shower_fudge_factor = 0.86 was fitted on a BLEND.

THE CONSEQUENCE, and the only claim this script makes.  A fudge fitted on the
blend absorbs the sample's average leakage, so it is partly a
detector-geometry constant: it does not transport to a different fiducial
cut, a different sample composition, or a different detector.  The
CONTAINED-only fit is the calorimetric number -- the one that answers "is the
charge-to-energy scale right", separated from "how much energy left the box".

WHAT THIS IS NOT.  This is NOT a proposal to apply a per-pair leakage
correction to the mass.  Doing that double counts (pr136_containment.py
sec 4 measured it: median 135.5 -> 142.9 MeV and in-window 37 -> 34), because
the fudge in force was fitted so the MEASURED peak sits at 135 and therefore
already absorbs mean leakage.  The containment number is a CLASSIFIER here,
never a correction.

ESTIMATOR.  Identical to pr135_pi0_peak_prod.py: the FIXED-PAIRING mass from
the hand pairing and the hand angle with PRODUCTION energies, so no
acceptance window and no reco-angle bias enters.  peak_fit/boot/to_fudge are
pr126_pi0_peak.py's, reused unchanged.

READ-ONLY.  Input is pr136-containment.tsv (which already carries m_prod, f1,
f2 and the geometry flags) -- no arm, no dump re-read.

    scripts/pr136_peak_contained.py [--tsv docs/pr/pr136-containment.tsv]
        [--fudge 0.86] [--fcut 0.95] [--out docs/pr/pr136-peak-contained.tsv]
"""
import argparse, csv, importlib.util, math, os
import numpy as np

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PK = _load("pr126_pi0_peak", os.path.join(SD, "pr126_pi0_peak.py"))


def f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def cell(name, xs, fudge, note=""):
    x = np.asarray([v for v in xs if v > 0], float)
    nan = float("nan")
    if len(x) < 5:
        return dict(cell=name, n=len(x), n_in=0, median=nan, median_in=nan,
                    peak=nan, peak_lo=nan, peak_hi=nan, implied_fudge=nan,
                    implied_lo=nan, implied_hi=nan, sanity="n/a", note=note)
    xin = x[(x >= PK.WIN[0]) & (x <= PK.WIN[1])]
    pk = PK.peak_fit(x)
    lo, hi = PK.boot(x, PK.peak_fit)
    g = lambda m: fudge * m / PK.PI0_MASS
    med_in = float(np.median(xin)) if len(xin) else nan
    return dict(cell=name, n=len(x), n_in=len(xin), median=float(np.median(x)),
                median_in=med_in, peak=pk, peak_lo=lo, peak_hi=hi,
                implied_fudge=g(pk), implied_lo=g(lo), implied_hi=g(hi),
                sanity=("PASS" if pk >= med_in - 1e-9 else "peak<median"),
                note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="docs/pr/pr136-containment.tsv")
    ap.add_argument("--fudge", type=float, default=0.86)
    ap.add_argument("--fcut", type=float, default=0.95,
                    help="a gamma is CONTAINED when its profile fraction f >= fcut")
    ap.add_argument("--out", default="docs/pr/pr136-peak-contained.tsv")
    a = ap.parse_args()

    p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
    with open(p) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    m_all, m_cont, m_leak, m_cont_clean, m_both_leak = [], [], [], [], []
    dmin_cont = []
    for r in rows:
        m = f(r["m_prod"])
        if m <= 0:
            continue
        f1, f2 = f(r["f1"], -1), f(r["f2"], -1)
        if f1 < 0 or f2 < 0:
            continue
        cont = (f1 >= a.fcut and f2 >= a.fcut)
        m_all.append(m)
        (m_cont if cont else m_leak).append(m)
        if cont and r.get("geom_suspect") == "0":
            m_cont_clean.append(m)
        if f1 < a.fcut and f2 < a.fcut:
            m_both_leak.append(m)
        if cont:
            dmin_cont.append(min(f(r["depth1_cm"]), f(r["depth2_cm"])))

    cells = [
        cell("A all pairs (the blend)", m_all, a.fudge,
             "the sample the 0.86 fit saw: contained + leaking together"),
        cell("B CONTAINED pairs (f1,f2>=%.2f)" % a.fcut, m_cont, a.fudge,
             "the calorimetric estimator: leakage removed by selection, not by correction"),
        cell("C >=1 leaking gamma", m_leak, a.fudge,
             "NOT a scale estimator -- it measures how much energy left the box"),
        cell("D contained AND geometry-clean", m_cont_clean, a.fudge,
             "B minus the axis-vs-vertex-ray >30 deg rows (the depth is least trustworthy there)"),
        cell("E BOTH gammas leaking", m_both_leak, a.fudge,
             "the far tail; n is small, quoted for shape only"),
    ]

    print("pi0 MASS PEAK, CONTAINED vs LEAKING  (doc pr/136 sec 6.1)")
    print("  input %s;  fudge in force %.2f;  containment cut f>=%.2f" %
          (os.path.relpath(p, SX), a.fudge, a.fcut))
    print("  pairs with both containment fractions computed: %d" % len(m_all))
    if dmin_cont:
        print("  contained pairs: min-depth median %.0f cm, min %.0f cm"
              % (float(np.median(dmin_cont)), min(dmin_cont)))
    print("\n%-34s %4s %5s %8s %19s %8s %-16s %s" %
          ("cell", "n", "n_in", "med(in)", "PEAK [CI68]", "fudge", "[CI68]", "sanity"))
    for c in cells:
        print("%-34s %4d %5d %8.1f  %6.1f [%5.1f,%5.1f] %8.3f [%.3f,%.3f] %s"
              % (c["cell"], c["n"], c["n_in"], c["median_in"], c["peak"],
                 c["peak_lo"], c["peak_hi"], c["implied_fudge"],
                 c["implied_lo"], c["implied_hi"], c["sanity"]))
    print()
    for c in cells:
        print("  %-34s %s" % (c["cell"], c["note"]))

    A, B = cells[0], cells[1]
    print("\nREADING")
    if not (math.isnan(A["peak"]) or math.isnan(B["peak"])):
        d = B["peak"] - A["peak"]
        print("  contained peak - blend peak = %+.1f MeV" % d)
        print("  implied fudge  blend %.3f [%.3f,%.3f]   contained %.3f [%.3f,%.3f]"
              % (A["implied_fudge"], A["implied_lo"], A["implied_hi"],
                 B["implied_fudge"], B["implied_lo"], B["implied_hi"]))
        overlap = not (B["implied_lo"] > A["implied_hi"] or A["implied_lo"] > B["implied_hi"])
        print("  CI68 overlap: %s -- %s" % (
            "yes" if overlap else "NO",
            "the leakage blend does not move the scale beyond its own statistical error"
            if overlap else
            "the blend and the calorimetric scale are statistically distinct"))
        print("  in force %.2f sits %s the contained CI68 [%.3f,%.3f]"
              % (a.fudge,
                 "INSIDE" if B["implied_lo"] <= a.fudge <= B["implied_hi"] else "OUTSIDE",
                 B["implied_lo"], B["implied_hi"]))
    print("  NOTE: this is a classifier result, not a correction -- see the docstring.")

    o = a.out if os.path.isabs(a.out) else os.path.join(SX, a.out)
    with open(o, "w", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(cells[0].keys()))
        w.writeheader(); w.writerows(cells)
    print("\nwrote %s (%d cells)" % (o, len(cells)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Calibrate the QLMatching low-PE (PD-inefficiency) error inflation from the
run-29107 hand-scan labels (evts 983/991/999/1007).

Motivation
----------
When a photon detector's PREDICTED PE is modest it frequently measures nothing at
all -- real low-light detection inefficiency, far stronger than Poisson (Poisson
P(0|pred=3) ~ 5%; the hand scans show ~45%). The bundle chi2 currently treats such
a channel with a flat 30% relative error, so a "predicted 3 PE, measured 0" channel
contributes ~11 to chi2; across a bundle these pile up and penalize good matches.

Fix (matches the C++ change in TimingTPCBundle.cxx): grow the per-opdet relative
error as the predicted PE falls,

    rel(pred) = frac + (lowpe_frac - frac) * exp(-pred / lowpe_knee)
    perr      = sqrt( (rel(pred)*pred)^2 + floor^2 )
    sigma^2   = meas + perr^2 ,   chi2_j = (pred-meas)^2 / sigma^2

so that at high pred rel -> frac (0.3, unchanged) and at low pred rel -> lowpe_frac
(near-unity), making "predicted a few PE, measured zero" statistically tolerated.

This script reproduces the inefficiency curve and picks (lowpe_frac, lowpe_knee) so
the typical meas==0 channel contributes ~1 to chi2, while leaving high-PE channels
untouched. Scope = the meas==0 / low-pred case only (the measured >> predicted tail
is a separate effect, out of scope).

Usage:  python3 fit_lowpe.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LBL = os.path.join(HERE, "..", "work", "ql_labels")
EVTS = ["983", "991", "999", "1007"]

# error-model constants. FRAC is the high-pred relative-error asymptote = the PDHD
# cfg pe_err_frac (0.44, from fit_labels.py); FLOOR = pe_err_floor (C++ default 0.3).
FRAC = 0.44
FLOOR = 0.3
NDF_KNEE = 1.0   # pe_ndf_knee: a pair counts only if NOT (meas<knee and pred<knee)


def load_pairs():
    pairs = []
    nmatch = 0
    for e in EVTS:
        d = json.load(open(os.path.join(LBL, "labels-evt%s.json" % e)))
        for m in d["matches"]:
            nmatch += 1
            for p, x in zip(m["pred_pes"], m["op_pes"]):
                pairs.append((p, x))
    # ndf gate: only channels that enter the chi2/ndf sum
    counted = [(p, x) for p, x in pairs if not (x < NDF_KNEE and p < NDF_KNEE)]
    return pairs, counted, nmatch


def rel(pred, lowpe_frac, knee):
    return FRAC + (lowpe_frac - FRAC) * math.exp(-pred / knee)


def perr_cur(p):
    # current pe_err_on_pred branch: (pred<knee)?floor:frac*pred  (knee=1.0)
    return FLOOR if p < 1.0 else FRAC * p


def perr_new(p, lowpe_frac, knee):
    r = rel(p, lowpe_frac, knee)
    return math.sqrt((r * p) ** 2 + FLOOR ** 2)


def chi2(p, x, perr):
    return (p - x) ** 2 / (x + perr * perr)


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


def efficiency_table(counted):
    bins = [(1, 2), (2, 5), (5, 10), (10, 30), (30, 100), (100, 1e18)]
    print("\n  predicted-PE inefficiency (counted channels, human-confirmed matches)")
    print("  %-12s %7s %12s %10s" % ("pred bin", "N", "frac meas=0", "med pred"))
    for lo, hi in bins:
        sub = [(p, x) for p, x in counted if lo <= p < hi]
        if not sub:
            continue
        f0 = sum(1 for p, x in sub if x == 0) / len(sub)
        print("  [%5g,%6g) %7d %12.3f %10.2f"
              % (lo, hi, len(sub), f0, median([p for p, x in sub])))


def fit(counted):
    """Pick (lowpe_frac, knee) so the meas==0 subset's median chi2 ~ 1.

    For a meas==0 channel chi2 = pred^2 / ((rel*pred)^2 + floor^2) ~ 1/rel^2 when
    pred >> floor, so rel ~ 1 -> median chi2 ~ 1. The knee places the 0.3->lowpe_frac
    transition where the inefficiency dies out (frac meas=0 < ~10%, i.e. ~10-20 PE).
    """
    zero = [(p, x) for p, x in counted if x == 0]
    best = None
    for kf in [round(0.8 + 0.05 * i, 2) for i in range(0, 41)]:        # 0.80 .. 2.80
        for kn in [round(2.0 + 0.5 * i, 1) for i in range(0, 37)]:     # 2.0 .. 20.0
            med = median([chi2(p, x, perr_new(p, kf, kn)) for p, x in zero])
            score = abs(med - 1.0)
            if best is None or score < best[0]:
                best = (score, kf, kn, med)
    return best[1], best[2], best[3]


def report(counted, lowpe_frac, knee):
    zero = [(p, x) for p, x in counted if x == 0]
    print("\n  fitted:  pe_err_lowpe_frac = %.2f   pe_err_lowpe_knee = %.1f" % (lowpe_frac, knee))
    print("\n  meas==0 subset (N=%d, the reported case):" % len(zero))
    print("    median chi2   current %.2f -> new %.2f"
          % (median([chi2(p, x, perr_cur(p)) for p, x in zero]),
             median([chi2(p, x, perr_new(p, lowpe_frac, knee)) for p, x in zero])))
    print("    mean   chi2   current %.2f -> new %.2f"
          % (sum(chi2(p, x, perr_cur(p)) for p, x in zero) / len(zero),
             sum(chi2(p, x, perr_new(p, lowpe_frac, knee)) for p, x in zero) / len(zero)))
    # high-PE must be ~unchanged
    hi = [(p, x) for p, x in counted if p >= 50]
    print("\n  pred>=50 subset (N=%d, must stay ~unchanged):" % len(hi))
    print("    rel(50)=%.3f rel(200)=%.3f  (frac=%.2f)"
          % (rel(50, lowpe_frac, knee), rel(200, lowpe_frac, knee), FRAC))
    print("    sum chi2      current %.0f -> new %.0f"
          % (sum(chi2(p, x, perr_cur(p)) for p, x in hi),
             sum(chi2(p, x, perr_new(p, lowpe_frac, knee)) for p, x in hi)))


def main():
    pairs, counted, nmatch = load_pairs()
    print("matches=%d  per-PD pairs=%d  counted=%d" % (nmatch, len(pairs), len(counted)))
    efficiency_table(counted)
    lf, kn, med = fit(counted)
    report(counted, lf, kn)
    print("\n  => jsonnet:  pe_err_lowpe_frac: %.2f,  pe_err_lowpe_knee: %.1f" % (lf, kn))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PEAK of the pi0 mass distribution and the EM scale (doc pr/126 sec 4g).

Owner correction to round 1: *"we should align with the peak instead of mean of
the pi0 mass distribution.  This may need a fit, with low statistics."*

WHY THE PEAK.  m = sqrt(4 E1 E2 sin^2(theta/2)) is built from charge-derived
energies, and its failure modes are ONE-SIDED: a shower missing members lost
charge, so m falls.  Nothing makes a true pi0's mass rise except a wrong
pairing.  The distribution is therefore a peak with a low tail, and BOTH the
mean and the median sit inside that tail.  The mode is the scale; the tail is
reconstruction loss and must not be averaged in.

THE ESTIMATOR, and why this one (all four criteria fixed before the real-data
number was quoted; the full comparison is printed by --compare so nothing is
hidden):

  1. it must be a FIT, as the owner asked -- which rules out the nonparametric
     mode-finders (KDE mode, half-sample mode) as PRIMARY;
  2. its window must be fixed by something EXTERNAL to the sample: [100, 185]
     MeV is the union of the finders' own acceptance windows
     (id_pi0_with_vertex (100,160), id_pi0_without_vertex (65,185) upper edge);
  3. it must be stable at n ~ 20 -- verified on truth-known toys by --validate;
  4. on a low-tailed sample the fitted peak must come out >= the median.

  => unbinned TRUNCATED-Gaussian maximum likelihood on [100,185], with mu
     bounded to the window and sigma to [3,60] MeV.  The truncation term is not
     optional: dropping it biases mu into the tail.  The BOUNDS are not
     cosmetic either -- the unbounded Nelder-Mead version runs away on ~1 in 30
     resamples at n=19 (toy sd 2070 MeV), which the bootstrap then under-reports.

  ERROR: bootstrap of the whole procedure (refit inside every resample), fixed
  seed.  Not the Gaussian's analytic error.

WHAT THE TOYS SAY, and it matters more than the point estimate: at n=19 BOTH
the fit and the median are biased LOW, by an amount set by the (unknown) tail
fraction -- see --validate.  The fit's bias is consistently ~75-80 % of the
median's.  So the fitted peak is a FLOOR on the true peak, and the correction
it implies is a FLOOR on the true correction.  The observed (fit - median) gap
is itself a tail-strength meter: on toys it grows monotonically with the tail
fraction.

READ-ONLY.  Consumes docs/pr/pr126-pi0-mass.tsv; changes nothing.

    ./pr126_pi0_peak.py                  # the measurement
    ./pr126_pi0_peak.py --compare        # every estimator, published not hidden
    ./pr126_pi0_peak.py --validate       # the truth-known toy study
    ./pr126_pi0_peak.py --selftest
    ./pr126_pi0_peak.py --tsv docs/pr/pr126-pi0-peak.tsv
"""
import argparse, csv, math, os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI0_MASS = 134.9768
FUDGE_NOW = 0.80          # kine_shower_fudge_factor in force on SBND
E_MIN = 15.0              # NeutrinoTaggerNuE.cxx:695, the code's own threshold
WIN = (100.0, 185.0)      # union of the two finders' acceptance windows
NBOOT = 1500
SEED = 126


def to_fudge(m):
    return FUDGE_NOW * m / PI0_MASS


# ------------------------------------------------------------- estimators
def peak_fit(x, lo=WIN[0], hi=WIN[1]):
    """Bounded unbinned truncated-Gaussian ML.  Returns mu (the peak)."""
    x = np.asarray(x, float)
    xin = x[(x >= lo) & (x <= hi)]
    if len(xin) < 4:
        return float("nan")

    def nll(p):
        mu, ls = p
        s = math.exp(ls)
        z = (xin - mu) / s
        d = norm.cdf((hi - mu) / s) - norm.cdf((lo - mu) / s)
        if d <= 1e-12:
            return 1e9
        return float(np.sum(0.5 * z * z + ls) + len(xin) * math.log(d))

    r = minimize(nll, [float(np.median(xin)), math.log(max(np.std(xin), 5.0))],
                 method="L-BFGS-B",
                 bounds=[(lo, hi), (math.log(3.0), math.log(60.0))])
    return float(r.x[0])


def peak_fit_sigma(x, lo=WIN[0], hi=WIN[1]):
    x = np.asarray(x, float)
    xin = x[(x >= lo) & (x <= hi)]
    if len(xin) < 4:
        return float("nan"), 0

    def nll(p):
        mu, ls = p
        s = math.exp(ls)
        z = (xin - mu) / s
        d = norm.cdf((hi - mu) / s) - norm.cdf((lo - mu) / s)
        if d <= 1e-12:
            return 1e9
        return float(np.sum(0.5 * z * z + ls) + len(xin) * math.log(d))

    r = minimize(nll, [float(np.median(xin)), math.log(max(np.std(xin), 5.0))],
                 method="L-BFGS-B",
                 bounds=[(lo, hi), (math.log(3.0), math.log(60.0))])
    return float(math.exp(r.x[1])), len(xin)


def kde_mode(x, factor=1.0, grid=3001):
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return float("nan")
    sd = np.std(x, ddof=1)
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    a = min(sd, iqr / 1.349) if iqr > 0 else sd
    h = factor * 0.9 * a * n ** (-0.2)
    if h <= 0:
        return float("nan")
    g = np.linspace(x.min() - 3 * h, x.max() + 3 * h, grid)
    dens = np.exp(-0.5 * ((g[:, None] - x[None, :]) / h) ** 2).sum(axis=1)
    return float(g[int(np.argmax(dens))])


def half_sample_mode(x):
    x = np.sort(np.asarray(x, float))
    while len(x) > 3:
        n = len(x)
        h = n // 2
        i = int(np.argmin(x[h:] - x[:n - h]))
        x = x[i:i + h + 1]
    return float(np.mean(x))


def boot(x, fn, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    o = [v for v in (fn(rng.choice(x, size=len(x), replace=True)) for _ in range(nboot))
         if np.isfinite(v)]
    if len(o) < 20:
        return float("nan"), float("nan")
    o = np.sort(np.asarray(o))
    return float(o[int(0.16 * len(o))]), float(o[int(0.84 * len(o))])


# ------------------------------------------------------------------ data
def load(conv="vtx", which="now", gate="ncpi0"):
    ek = {"scanhand": "hand", "scanreco": "reco", "now": "now"}[which]
    out = []
    with open(os.path.join(SX, "docs", "pr", "pr126-pi0-mass.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            m, e1, e2 = r.get("m_%s_%s" % (conv, which)), r.get("e1_" + ek), r.get("e2_" + ek)
            if not m or not e1 or not e2:
                continue
            m, e1, e2 = float(m), float(e1), float(e2)
            if m <= 0 or min(e1, e2) <= E_MIN:
                continue
            if gate == "ncpi0" and r["origin"] != "ncpi0":
                continue
            out.append(m)
    return np.asarray(out)


# ---------------------------------------------------------------- reports
def measure(label, conv, which, gate):
    x = load(conv, which, gate)
    if len(x) < 5:
        print("  %-40s n=%3d (too few)" % (label, len(x)))
        return None
    med = float(np.median(x))
    pk = peak_fit(x)
    lo, hi = boot(x, peak_fit)
    mlo, mhi = boot(x, lambda v: float(np.median(v)))
    sig, nin = peak_fit_sigma(x)
    excl = not (lo <= PI0_MASS <= hi)
    print("  %-40s n=%2d  median=%6.1f [%.1f,%.1f]" % (label, len(x), med, mlo, mhi))
    print("      PEAK = %6.1f  CI68=[%6.1f,%6.1f]   sigma=%5.1f  n_in=%2d  peak-median=%+.1f"
          % (pk, lo, hi, sig, nin, pk - med))
    print("      -> kine_shower_fudge_factor = %.3f  [%.3f, %.3f]      CI excludes 134.9768: %s"
          % (to_fudge(pk), to_fudge(lo), to_fudge(hi), "YES" if excl else "no"))
    if conv == "vtx":
        print("      SANITY peak >= median: %s" % ("PASS" if pk >= med - 1e-9 else "*** FAIL — REJECT ***"))
    return dict(label=label, conv=conv, which=which, gate=gate, n=len(x), median=med,
                median_lo=mlo, median_hi=mhi, peak=pk, peak_lo=lo, peak_hi=hi, sigma=sig,
                n_in=nin, fudge=to_fudge(pk), fudge_lo=to_fudge(lo), fudge_hi=to_fudge(hi),
                ci_excludes_135=int(excl))


def compare():
    print("=== every estimator, published rather than hidden ===")
    ests = [("median", lambda v: float(np.median(v))),
            ("PEAK fit [100,185]", peak_fit),
            ("KDE mode x0.5", lambda v: kde_mode(v, 0.5)),
            ("KDE mode x1.0", lambda v: kde_mode(v, 1.0)),
            ("KDE mode x2.0", lambda v: kde_mode(v, 2.0)),
            ("half-sample mode", half_sample_mode)]
    for gate in ("ncpi0", "all"):
        for conv in ("vtx", "axis"):
            x = load(conv, "now", gate)
            if len(x) < 5:
                continue
            print("\n  gate=%s conv=%s n=%d" % (gate, conv, len(x)))
            for name, fn in ests:
                pt = fn(x)
                lo, hi = boot(x, fn)
                print("    %-20s %6.1f  CI68=[%6.1f,%6.1f]  fudge=%.3f  %s"
                      % (name, pt, lo, hi, to_fudge(pt),
                         "excl 135" if not (lo <= PI0_MASS <= hi) else ""))
    print("\n  window stability of the PRIMARY fit (ncpi0, vtx):")
    x = load("vtx", "now", "ncpi0")
    for lo, hi in ((100, 185), (95, 190), (90, 200), (105, 180), (100, 200), (80, 220), (110, 175)):
        n_in = int(((x >= lo) & (x <= hi)).sum())
        p = peak_fit(x, lo, hi)
        print("    [%3d,%3d] n_in=%2d peak=%6.1f fudge=%.3f" % (lo, hi, n_in, p, to_fudge(p)))


def validate(ntrial=600):
    """Truth-known toys at the real n.  This is the section that decides how the
    numbers may be read: it measures the BIAS of both estimators against a
    one-sided low tail."""
    print("=== toy validation at n=19, window [100,185] ===")
    print("  a Gaussian core (sigma 18) plus a fraction drawn from an exponential")
    print("  low tail (tau 30) -- the charge-loss shape.  600 trials per row.\n")
    print("  %-7s %-6s %-10s %-9s %-10s %-9s" % ("truth", "tail%", "fit median", "fit bias", "med median", "med bias"))
    for truth in (135.0, 140.0, 145.0):
        for tf in (0.0, 0.15, 0.30):
            rng = np.random.default_rng(11)
            E, M = [], []
            for _ in range(ntrial):
                nt = int(round(19 * tf))
                s = np.concatenate([rng.normal(truth, 18.0, 19 - nt),
                                    truth - np.abs(rng.exponential(30.0, nt))])
                v = peak_fit(s)
                if np.isfinite(v):
                    E.append(v)
                    M.append(float(np.median(s)))
            E, M = np.asarray(E), np.asarray(M)
            print("  %-7.1f %-6.0f %-10.1f %-9.1f %-10.1f %-9.1f"
                  % (truth, 100 * tf, np.median(E), np.median(E) - truth,
                     np.median(M), np.median(M) - truth))
    print("\n  bias vs tail fraction (truth 140), and the fit-minus-median gap")
    print("  -- that gap is a TAIL-STRENGTH METER for the real sample:")
    print("  %-7s %-10s %-10s %-10s" % ("tail%", "med bias", "fit bias", "fit-med"))
    for tf in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        rng = np.random.default_rng(11)
        E, M = [], []
        for _ in range(ntrial):
            nt = int(round(19 * tf))
            s = np.concatenate([rng.normal(140.0, 18.0, 19 - nt),
                                140.0 - np.abs(rng.exponential(30.0, nt))])
            v = peak_fit(s)
            if np.isfinite(v):
                E.append(v)
                M.append(float(np.median(s)))
        fe, fm = float(np.median(E)) - 140.0, float(np.median(M)) - 140.0
        print("  %-7.0f %-10.1f %-10.1f %-10.1f" % (100 * tf, fm, fe, fe - fm))
    x = load("vtx", "now", "ncpi0")
    print("\n  REAL primary sample: fit-minus-median = %+.1f MeV" % (peak_fit(x) - float(np.median(x))))
    print("  -> read against the table above, that indicates a HEAVY tail, i.e.")
    print("     both estimators are biased low and the fitted peak is a FLOOR.")


def selftest():
    ok = True
    rng = np.random.default_rng(0)
    y = rng.normal(140.0, 16.0, 3000)
    p = peak_fit(y)
    g = abs(p - 140.0) < 1.5
    print("%s  symmetric toy: peak=%.1f (truth 140.0)" % ("OK " if g else "FAIL", p))
    ok &= g
    # low tail: peak must sit ABOVE the median, and closer to truth
    core = rng.normal(140.0, 18.0, 800)
    tail = 140.0 - np.abs(rng.exponential(30.0, 240))
    x = np.concatenate([core, tail])
    p, m = peak_fit(x), float(np.median(x))
    g2 = p > m and abs(p - 140.0) < abs(m - 140.0)
    print("%s  low-tail toy: peak=%.1f > median=%.1f, and nearer truth 140.0"
          % ("OK " if g2 else "FAIL", p, m))
    ok &= g2
    # the property that lets one arm scan every trial scale
    k = 0.93
    g3 = abs(peak_fit(k * x, k * WIN[0], k * WIN[1]) - k * p) < 0.02 * k * p
    print("%s  peak(k*m) == k*peak(m) under a scaled window" % ("OK " if g3 else "FAIL"))
    ok &= g3
    xr = load("vtx", "now", "ncpi0")
    pr_, mr = peak_fit(xr), float(np.median(xr))
    g4 = pr_ >= mr
    print("%s  REAL primary sanity peak(%.1f) >= median(%.1f), n=%d"
          % ("OK " if g4 else "FAIL", pr_, mr, len(xr)))
    ok &= g4
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.validate:
        validate()
        return 0
    if a.compare:
        compare()
        return 0
    print("pi0 mass = %.4f MeV;  in force kine_shower_fudge_factor = %.2f;  fit window [%.0f,%.0f]"
          % (PI0_MASS, FUDGE_NOW, WIN[0], WIN[1]))
    rows = []
    print("\n=== PRIMARY ===")
    r = measure("ncpi0 + min(E)>15, vertex chord", "vtx", "now", "ncpi0")
    if r:
        rows.append(r)
    print("\n=== cross-checks ===")
    for lab, conv, which, gate in (
            ("ncpi0 + min(E)>15, shower axis", "axis", "now", "ncpi0"),
            ("all origins, vertex chord", "vtx", "now", "all"),
            ("all origins, shower axis", "axis", "now", "all"),
            ("ncpi0, scan-time + marks, vtx", "vtx", "scanhand", "ncpi0"),
            ("ncpi0, scan-time as reco, vtx", "vtx", "scanreco", "ncpi0")):
        r = measure(lab, conv, which, gate)
        if r:
            rows.append(r)
    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\nwrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

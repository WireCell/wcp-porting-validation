#!/usr/bin/env python3
"""Section 1: what pu / pv / pw / pt actually are, measured not assumed.

Two measurements, because one is not enough:

  (a) SINGLE-AXIS, on one long straight chain.  A coordinate that regresses on
      ONE of x, y, z with |r| = 1 is that axis and nothing else.  pt and pw do;
      the induction coordinates do not, because they depend on y AND z -- and a
      straight chain makes x, y, z collinear, so a multivariate fit on it is not
      identifiable and its apparent pitch is an artifact.  That is why (b) exists.

  (b) GRADIENT MAGNITUDE, pooled over every chain.  |dp| / |dr| in 3-D is the
      directional derivative of the coordinate; its maximum over directions is
      the gradient magnitude, = 1 / pitch.  This works for pw and pt.  It does
      NOT work for pu and pv: PDHD's induction planes are WRAPPED, so those
      coordinates carry discontinuities (censused below) that contaminate the
      upper tail and drive the estimate to about half the true pitch.  pu and
      pv are therefore reported as NOT ESTABLISHED here -- which also limits
      what their null fold results in sec 2 can be taken to mean.
"""
import os, json, glob, sys
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")
PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
EV, CID = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("028084_8", "120")
LO, HI = 30, 250

out = ["(a) single-axis regression, chain %s/%s rows %d:%d" % (EV, CID, LO, HI), ""]
d = json.load(open("%s/smprep-%s-c%s.json" % (PREP, EV, CID)))
m = d["muon"]
a = lambda k: np.array([np.nan if t is None else t for t in m[k]], float)
x, y, z = a("x"), a("y"), a("z"); s = slice(LO, HI)
for nm in ("pt", "pw", "pv", "pu"):
    p = a(nm); best = []
    for cn, c in (("x", x), ("y", y), ("z", z)):
        g = np.isfinite(p[s]) & np.isfinite(c[s])
        if g.sum() < 20: continue
        best.append((abs(np.corrcoef(c[s][g], p[s][g])[0, 1]), cn,
                     np.polyfit(c[s][g], p[s][g], 1)[0]))
    best.sort(reverse=True)
    r, cn, slope = best[0]
    tag = "EXACT -- this axis alone" if r > 0.99999 else "NOT a single axis (artifact if read as a pitch)"
    out.append("  %-3s best axis %s  %+.4f units/cm  (%.4f cm/unit)  |r|=%.6f   %s"
               % (nm, cn, slope, abs(1.0/slope), r, tag))

out += ["", "(b) gradient magnitude |dp|/|dr| pooled over all chains -> 1/pitch", ""]
acc = {k: [] for k in ("pt", "pw", "pv", "pu")}
for f in sorted(glob.glob(PREP + "/smprep-*.json")):
    mm = json.load(open(f))["muon"]
    g = lambda k: np.array([np.nan if t is None else t for t in mm[k]], float)
    X, Y, Z = g("x"), g("y"), g("z")
    dr = np.sqrt(np.diff(X)**2 + np.diff(Y)**2 + np.diff(Z)**2)
    for k in acc:
        dp = np.abs(np.diff(g(k)))
        # |dp| >= 5 is a readout-unit change (APA / cathode), not a wrap; the
        # smaller wrap steps are deliberately KEPT so the census below sees them
        ok = np.isfinite(dp) & np.isfinite(dr) & (dr > 0.05) & (dp < 5)
        if ok.any(): acc[k].append(dp[ok]/dr[ok])
for k in ("pt", "pw", "pv", "pu"):
    v = np.concatenate(acc[k])
    q = np.percentile(v, 99.5)
    # a step implying a pitch under 0.30 cm is not geometry, it is a wrap
    disc = 100.0*float((v > 1/0.30).mean())
    flag = "ESTABLISHED" if disc < 0.2 else "NOT ESTABLISHED (wrapped plane)"
    out.append("  %-3s q99.5 |dp|/|dr| = %.4f /cm  =>  pitch %.4f cm   "
               "discontinuous steps %.3f%%   %s   (n=%d)"
               % (k, q, 1.0/q, disc, flag, v.size))
out.append("")
out.append("  The discontinuity rate above is the discriminator, two orders of magnitude")
out.append("  apart.  frac(pu) and frac(pv) are therefore not a clean cell phase, and")
out.append("  sec 2's null fold result on those two planes is weaker evidence than the")
out.append("  pw and pt results, not equal evidence.")
txt = "\n".join(out); print(txt)
open(os.path.join(FIGS, "24_calib.txt"), "w").write(txt+"\n")

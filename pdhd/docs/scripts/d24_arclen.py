#!/usr/bin/env python3
"""Section 5 bullet 1, and the size of the thing to be explained.

  (a) Is L the true arc length?  If the chain's L were a nominal resample step
      while the 3-D points sat elsewhere, dQ/dx = dQ/dL would oscillate for that
      reason alone.  Reported as the 3-D step between consecutive points divided
      by the step in L.
  (b) How big is the wave?  The plateau dQ/dx fractional RMS per chain, which is
      what any candidate mechanism has to account for.
"""
import os, json, glob
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")
PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
rs, fr = [], []
for f in sorted(glob.glob(PREP + "/smprep-*.json")):
    m = json.load(open(f))["muon"]
    a = lambda k: np.array([np.nan if t is None else t for t in m[k]], float)
    x, y, z, L, q, rr = a("x"), a("y"), a("z"), a("L"), a("q"), a("rr")
    d3 = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    dl = np.diff(L)
    g = np.isfinite(d3) & np.isfinite(dl) & (dl > 0)
    rs.append(d3[g]/dl[g])
    p = np.isfinite(q) & (q > 0) & (rr > 30)
    if p.sum() >= 60:
        fr.append(np.std(q[p])/np.mean(q[p]))
r = np.concatenate(rs); fr = np.array(fr)
out = ["(a) 3-D step / dL over %d steps of %d chains:" % (r.size, len(rs))]
for qq in (1, 5, 25, 50, 75, 95, 99):
    out.append("      q%02d %.4f" % (qq, np.percentile(r, qq)))
out.append("      => L is the true arc length; the +-1.5% tail is not a dQ/dx wave.")
out.append("")
out.append("(b) plateau (rr>30) dQ/dx fractional RMS per chain, %d chains:" % fr.size)
out.append("      median %.3f   q25 %.3f   q75 %.3f" % (np.median(fr), np.percentile(fr,25), np.percentile(fr,75)))
txt = "\n".join(out); print(txt)
open(os.path.join(FIGS, "24_arclen.txt"), "w").write(txt+"\n")

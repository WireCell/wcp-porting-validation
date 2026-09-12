#!/usr/bin/env python3
"""Zoom: dQ/dx against each clock on one contiguous stretch, integer wires marked."""
import json, sys
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
def arr(v): return np.array([np.nan if t is None else t for t in v], float)

ev, cid, lo, hi, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
d = json.load(open("%s/smprep-%s-c%s.json" % (PREP, ev, cid)))
m = d["muon"]
L, q = arr(m["L"]), arr(m["q"])
sel = (L >= lo) & (L <= hi)
L, q = L[sel], q[sel]
fig, ax = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
ax[0].plot(L, q/1000., ".-", lw=1.0, ms=4, color="k")
ax[0].set_ylabel("dQ/dx [ke/cm]")
ax[0].set_title("%s/%s   L in [%g, %g] cm" % (ev, cid, lo, hi))
ax[0].grid(alpha=0.3)
for i, nm in enumerate(("pw", "pu", "pv", "pt")):
    p = arr(m[nm])[sel]
    a = ax[i+1]
    a.plot(L, p, "-", lw=1.2, color="tab:blue")
    a.set_ylabel(nm, fontsize=9)
    p0, p1 = np.nanmin(p), np.nanmax(p)
    if np.isfinite(p0) and (p1-p0) < 120:
        for k in np.arange(np.floor(p0), np.ceil(p1)+1):
            a.axhline(k, color="grey", lw=0.5, alpha=0.6)
    a.text(0.99, 0.86, "%.2f units over %.1f cm => %.2f cm/unit"
           % (p1-p0, hi-lo, (hi-lo)/max(p1-p0, 1e-9)),
           transform=a.transAxes, ha="right", fontsize=8)
    a.grid(axis="x", alpha=0.3)
ax[-1].set_xlabel("L [cm]")
fig.tight_layout(); fig.savefig(out, dpi=110); print("wrote", out)

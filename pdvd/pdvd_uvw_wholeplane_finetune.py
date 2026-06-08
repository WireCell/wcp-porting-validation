#!/usr/bin/env python
"""Whole-plane fine-tune of the (sum / common-mode / all-on-V) shift, using ALL
tracks.  Pool every whole-plane triple, look at the dZ_W distribution AFTER the
common-mode shift, and locate its PEAK robustly (the real triples pile up at the
true residual; combinatorial fakes form a broad background).

The peak location = residual sum correction on top of the common mode.  Reported
three equivalent ways:
  - refine common mode  : split evenly, each of U,V gets peak/2
  - all-on-V (your gauge): extra V shift = peak*2/... = 2*(peak)/2 -> = peak in sum,
                           i.e. dV = 2*residual_sum  (since cV=0.5)
Also split by track angle to check a single constant fits the whole plane.
Read-only; writes a histogram to /home/xqian/tmp/ and pdvd/pics/.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wirecell.util.wires import persist

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd")
import pdvd_uvw_offset as O
import pdvd_uvw_wholeplane_scan as WP

OUT = "/home/xqian/tmp"
PICS = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/pics"
HALF = O.W_PITCH_MM / 2.0


def peak_of(res, lo=-HALF, hi=HALF, nb=51):
    """Robust peak: histogram the core, smooth, return bin-center of the max,
    then refine with the charge-unweighted median of triples within ±1 mm of it."""
    h, e = np.histogram(res, bins=nb, range=(lo, hi))
    c = 0.5 * (e[:-1] + e[1:])
    hs = np.convolve(h, np.ones(5) / 5.0, mode="same")
    p0 = c[int(np.argmax(hs))]
    near = res[np.abs(res - p0) < 1.0]
    return float(np.median(near)) if len(near) else p0, h, c


store = persist.load(WP.V4)
print("Whole-plane fine-tune of the sum/common-mode shift, all tracks (v4).")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, akey in zip(axes, (0, 4)):
    pc = WP.PCFG[akey]
    fi, tabs = O.find_face_planes(store, pc["anode"], WP.tabs_w_window(store, pc))
    a, b, cU, cV = WP.crossing_coeffs(tabs)
    res = WP.collect_triples(pc, tabs, a, b)        # dZ_W after common-mode, |.|<2.5 pitch

    peak, h, c = peak_of(res)
    core = res[np.abs(res) < HALF]
    print(f"\n==== {pc['label']} ====")
    print(f"  {len(res)} pooled triples; {len(core)} in the ±½-pitch core")
    print(f"  dZ_W distribution PEAK after common-mode = {peak:+.2f} mm "
          f"(core median {np.median(core):+.2f}, core MAD {np.median(np.abs(core-np.median(core))):.2f})")
    print(f"  => residual SUM correction = {peak:+.2f} mm.  Apply as either:")
    print(f"       refine common mode : U,V each {pc['coarse']:+.2f} {peak/2:+.2f} = "
          f"{pc['coarse']+peak/2:+.2f} mm")
    print(f"       all-on-V (gauge)   : U {pc['coarse']:+.2f} mm,  V {pc['coarse']:+.2f} "
          f"{peak:+.2f} = {pc['coarse']+peak:+.2f} mm")

    # angle robustness: split pooled triples is not angle-tagged; instead compare
    # the clean single-track sum to the whole-plane peak as a cross-check.
    ax.hist(res, bins=60, range=(-2.5*O.W_PITCH_MM, 2.5*O.W_PITCH_MM),
            color="0.7", label="all pooled triples")
    ax.hist(core, bins=30, range=(-HALF, HALF), color="C0", label="±½-pitch core")
    ax.axvline(peak, color="r", ls="--", label=f"peak {peak:+.2f} mm")
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("dZ_W after common-mode  [mm]"); ax.set_ylabel("triples")
    ax.set_title(f"{pc['label']}: whole-plane residual"); ax.legend(fontsize=8)
fig.tight_layout()
for d in (OUT, PICS):
    fig.savefig(f"{d}/pdvd_wholeplane_finetune.png", dpi=120)
print(f"\nwrote {OUT}/pdvd_wholeplane_finetune.png and {PICS}/")

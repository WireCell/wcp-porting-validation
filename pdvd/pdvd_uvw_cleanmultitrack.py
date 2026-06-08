#!/usr/bin/env python
"""Clean multi-track fine-tune using ALL tracks but NO combinatorial fakes.

Pool only the ticks where each plane has ONE dominant charge peak (a single real
track is active there) -> the (U,V,W) triple is unambiguous.  This samples every
track in the plane across the ticks where it is isolated, with no fake pairings.

Report, from this clean pooled set:
  - the optimal common-mode (sum) shift s* (all-tracks), vs the single-track value;
  - the residual after the current common mode (=> additional-V under the gauge);
  - whether one constant fits (spread of dZ_W).
Read-only; writes a histogram to /home/xqian/tmp/ and pdvd/pics/.
"""
import sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from wirecell.util.wires import persist

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd")
import pdvd_uvw_offset as O
import pdvd_uvw_wholeplane_scan as WP

OUT = "/home/xqian/tmp"
PICS = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/pics"
HALF = O.W_PITCH_MM / 2.0
DOM = 3.0          # a peak is "dominant" if >= DOM x the next-highest peak


def clean_triples(pc, tabs, a, b):
    fn = O.MAGBASE.format(E=pc["event"], A=pc["anode"])
    f = uproot.open(fn)
    prof = {}
    for pl in ("U", "V", "W"):
        h = f[f"h{pl.lower()}_gauss{pc['anode']}"]
        v = np.clip(h.values(), 0, None)
        xe = h.axis(0).edges(); ch = (xe[:-1] + xe[1:]) / 2.0
        lo, hi = tabs[pl]["chan"].min(), tabs[pl]["chan"].max()
        sel = (ch >= lo) & (ch <= hi)
        prof[pl] = (v[sel], ch[sel])
    t0, t1 = pc["tickwin"]
    thr = {pl: 0.08 * np.percentile(prof[pl][0][prof[pl][0] > 0], 99.5)
           for pl in ("U", "V", "W")}

    base = []
    for t in range(t0, t1):
        ok = True; cen = {}
        for pl in ("U", "V", "W"):
            v, ch = prof[pl]
            col = v[:, t]
            idx, _ = find_peaks(col, height=thr[pl], distance=3)
            if len(idx) == 0:
                ok = False; break
            hgt = col[idx]
            order = np.argsort(hgt)[::-1]
            if len(idx) > 1 and hgt[order[0]] < DOM * hgt[order[1]]:
                ok = False; break                 # not dominant -> ambiguous, skip
            cen[pl] = ch[idx[order[0]]]
        if not ok:
            continue
        pU = O.chan_to_pitch(tabs["U"], cen["U"])
        pV = O.chan_to_pitch(tabs["V"], cen["V"])
        zc = a * pU + b * pV
        base.append(zc - O.w_chan_to_z(tabs["W"], cen["W"]))
    return np.array(base)


store = persist.load(WP.V4)
print("Clean multi-track fine-tune (single-dominant-peak ticks, no fakes), v4.")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, akey in zip(axes, (0, 4)):
    pc = WP.PCFG[akey]
    fi, tabs = O.find_face_planes(store, pc["anode"], WP.tabs_w_window(store, pc))
    a, b, cU, cV = WP.crossing_coeffs(tabs)
    base = clean_triples(pc, tabs, a, b)               # dZ_W at (0,0)

    # optimal common-mode (sum): s* maximizes consistency  |base + s| < half
    g = np.arange(-25, 25.001, 0.05)
    frac = [np.mean(np.abs(base + s) < HALF) for s in g]
    s_star = float(g[int(np.argmax(frac))])
    resid = base + s_star
    # single-track value for reference
    cfg = O.CONFIGS[akey]
    print(f"\n==== {pc['label']} ====")
    print(f"  {len(base)} clean single-track-dominant triples pooled")
    print(f"  all-tracks optimal common-mode s* = {s_star:+.2f} mm  "
          f"(consistency {np.max(frac)*100:.1f}%)")
    print(f"  residual after s*: median {np.median(resid):+.3f} mm, "
          f"robust-sigma {1.4826*np.median(np.abs(resid-np.median(resid))):.2f} mm")
    print(f"  => additional-V (gauge, fix U at s*): dV = {2*np.median(resid):+.3f} mm  "
          f"(~0 by construction once s* is optimal)")
    print(f"  cf. single calibration track s* (from doc/2dscan): "
          f"{'+13.1' if akey==0 else '-9.8'} mm")

    ax.hist(base, bins=80, range=(-25, 5) if akey==0 else (-25, 5), color="0.7",
            label="dZ_W at (0,0), clean triples")
    ax.axvline(-s_star, color="r", ls="--", label=f"-s* = {-s_star:+.2f} mm (peak)")
    ax.set_xlabel("dZ_W (uncorrected)  [mm]"); ax.set_ylabel("clean triples")
    ax.set_title(f"{pc['label']}: clean multi-track"); ax.legend(fontsize=8)
fig.tight_layout()
for d in (OUT, PICS):
    fig.savefig(f"{d}/pdvd_cleanmultitrack.png", dpi=120)
print(f"\nwrote {OUT}/pdvd_cleanmultitrack.png and {PICS}/")

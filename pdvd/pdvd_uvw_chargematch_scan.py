#!/usr/bin/env python
"""Charge-weighted fine separability test (user's refinement).

Same whole-plane triples as pdvd_uvw_wholeplane_scan.py, but now:
  - weight each matched (U,V,W) triple by its PEAK CHARGE, and
  - use a smooth charge-alignment score  S = sum_tri  q * exp(-r^2 / 2 sigma^2)
    (how well the U,V,W peaks line up, charge-weighted),
  - zoom into a SMALL fine window (+-3 mm) for the U-vs-V differential, since the
    common-mode shift is already close and the differential is expected small.

Decisive cut: at the best sum, slide along the DIFFERENCE (ddzU - ddzV) and show
the score is flat -> the charge alignment does not prefer any U-vs-V split.

Read-only; writes plot+summary to /home/xqian/tmp/.
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
HALF = O.W_PITCH_MM / 2.0
GATE = 2.5 * O.W_PITCH_MM
SIGMA = HALF                                  # soft-match width ~ half W-pitch


def collect_triples_q(pc, tabs, a, b):
    """Like WP.collect_triples but also return a per-triple charge weight
    (min of the three peak heights = the charge common to all three views)."""
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

    res, wgt = [], []
    for t in range(t0, t1):
        pk = {}; ph = {}
        for pl in ("U", "V", "W"):
            v, ch = prof[pl]
            col = v[:, t]
            idx, props = find_peaks(col, height=thr[pl], distance=3)
            pk[pl] = ch[idx]; ph[pl] = col[idx]
        if min(len(pk["U"]), len(pk["V"]), len(pk["W"])) == 0:
            continue
        pU = O.chan_to_pitch(tabs["U"], pk["U"])
        pV = O.chan_to_pitch(tabs["V"], pk["V"])
        zc = a * pU[:, None] + b * pV[None, :] + pc["coarse"]
        wz = O.w_chan_to_z(tabs["W"], pk["W"])
        d = zc[:, :, None] - wz[None, None, :]
        k = np.argmin(np.abs(d), axis=2)
        nearest = np.take_along_axis(d, k[:, :, None], axis=2)[:, :, 0]
        wq = ph["W"][k]                                   # W charge of nearest peak
        qmin = np.minimum(np.minimum(ph["U"][:, None], ph["V"][None, :]), wq)
        good = np.abs(nearest) < GATE
        res.extend(nearest[good].ravel().tolist())
        wgt.extend(qmin[good].ravel().tolist())
    return np.array(res), np.array(wgt)


def analyse(pc, store):
    fi, tabs = O.find_face_planes(store, pc["anode"], WP.tabs_w_window(store, pc))
    a, b, cU, cV = WP.crossing_coeffs(tabs)
    r0, q = collect_triples_q(pc, tabs, a, b)
    print(f"\n==== {pc['label']}  face {fi} ====")
    print(f"  {len(r0)} charge-weighted triples; cU={cU:.4f} cV={cV:.4f}")

    def score(ddU, ddV):
        r = r0 + cU * ddU + cV * ddV
        return float(np.sum(q * np.exp(-r ** 2 / (2 * SIGMA ** 2))))

    g = np.arange(-3.0, 3.001, 0.1)
    S = np.array([[score(du, dv) for du in g] for dv in g])    # rows dv, cols du
    jb, ib = np.unravel_index(S.argmax(), S.shape)
    best_sum = g[ib] + g[jb]
    print(f"  best charge-match at (ddzU,ddzV)=({g[ib]:+.2f},{g[jb]:+.2f}); "
          f"sum={best_sum:+.2f} mm")

    # decisive cut: hold sum at best_sum, slide the difference; score should be flat
    diffs = np.arange(-3.0, 3.001, 0.25)
    cut = []
    for dd in diffs:
        ddU = (best_sum + dd) / 2.0; ddV = (best_sum - dd) / 2.0
        cut.append(score(ddU, ddV))
    cut = np.array(cut)
    span = (cut.max() - cut.min()) / cut.mean() * 100
    print(f"  difference-direction cut at best sum: score varies by only "
          f"{span:.3f}% across ddzU-ddzV in [-3,+3] mm  (flat => no U-vs-V preference)")
    return pc, g, S, diffs, cut, best_sum


def plot(pc, g, S, diffs, cut, best_sum):
    GU, GV = np.meshgrid(g, g)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    cs = ax[0].contourf(GU, GV, S / S.max(), levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax[0], label="charge-match score (norm.)")
    ax[0].plot(g, best_sum - g, "r--", lw=1, label=f"ddzU+ddzV={best_sum:+.2f} (best sum)")
    ax[0].set_xlabel("ddzU (mm)"); ax[0].set_ylabel("ddzV (mm)")
    ax[0].set_title(f"{pc['label']}: charge-match vs fine (ddzU,ddzV)")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
    ax[1].plot(diffs, cut / cut.mean(), "o-")
    ax[1].set_xlabel("ddzU - ddzV  (mm)  [sum held at best]")
    ax[1].set_ylabel("charge-match score / mean")
    ax[1].set_ylim(0.9, 1.1)
    ax[1].set_title("difference-direction cut (flat = degenerate)")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    p = f"{OUT}/pdvd_chargematch_anode{pc['anode']}.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  wrote {p}")


def main():
    store = persist.load(WP.V4)
    print("Charge-weighted fine separability test (small differential window).")
    for akey in (0, 4):
        out = analyse(WP.PCFG[akey], store)
        plot(*out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""V fine-tune diagnostic (user's request).

After the common-mode U/V shift, FIX U (and W) and ask whether V wants to move:
for each tick, intersect the U wire (U peak) with the W wire (W peak) to get the
crossing point, predict the V coordinate there, and compare to the MEASURED V
peak.  Histogram (V_predicted - V_measured) over all ticks of each clean track.

  - distribution centred on 0  => V is already where U+W put it (no room to move);
  - a nonzero centre           => V could be shifted by that amount.

Caveat kept honest: this residual is the same three-plane closure scalar as the
W test, so its centre = the residual SUM after the common mode (it moves the same
whether you slide V or U); but it is exactly the picture asked for, in V units.

Writes the plot to /home/xqian/tmp and pdvd/pics.  Read-only on inputs.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wirecell.util.wires import persist

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd")
import pdvd_uvw_offset as O
import pdvd_uvw_2dscan as S2

OUT = "/home/xqian/tmp"
PICS = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/pics"
HALF_V = O.UV_PITCH_MM / 2.0            # 3.825 mm; V consistency tolerance


def predict_v_pitch(tabs, u_chan, w_chan, sU):
    """Intersect the (U-shifted) U wire and the (fixed) W wire; return the V pitch
    coordinate at that crossing."""
    pdU = tabs["U"]["pdir"]; pdW = tabs["W"]["pdir"]; pdV = tabs["V"]["pdir"]
    pU = O.chan_to_pitch(tabs["U"], u_chan) + sU * pdU[1]      # U shifted in z
    pW = O.chan_to_pitch(tabs["W"], w_chan)                    # W fixed
    A = np.vstack([pdU, pdW])
    P = np.linalg.solve(A, np.array([pU, pW]))
    return pdV @ P


def analyse(akey, store):
    cfg = O.CONFIGS[akey]
    fn = O.MAGBASE.format(E=cfg["event"], A=cfg["anode"])
    _, res = O.centroids(fn, cfg["anode"], cfg)
    gm = O.good_mask(res)
    uc = res["U"]["cen"][gm]; vc = res["V"]["cen"][gm]; wc = res["W"]["cen"][gm]
    fi, tabs = O.find_face_planes(store, cfg["anode"], cfg["win"]["W"])
    pdV_z = tabs["V"]["pdir"][1]

    # common-mode shift that centres the W-closure (the established v5 basis)
    z0fn, cU, cV = S2.crossing_z_coeffs(tabs)
    baseW = z0fn(uc, vc) - O.w_chan_to_z(tabs["W"], wc)
    s_star = -float(np.median(baseW))

    # fix U at s*, V at s*; predict V from (U,W); residual vs measured V (pitch mm)
    pV_meas = O.chan_to_pitch(tabs["V"], vc) + s_star * pdV_z
    pV_pred = np.array([predict_v_pitch(tabs, u, w, s_star) for u, w in zip(uc, wc)])
    rV = pV_pred - pV_meas                                     # mm along V pitch
    med = float(np.median(rV)); rms = float(np.std(rV))
    # V fine-tune that would null the median (moves residual by -pdV_z per mm of dzV)
    dV_tune = med / (-pdV_z)
    frac_in = float(np.mean(np.abs(rV - 0.0) < HALF_V))

    print(f"\n==== {cfg['label']} ====")
    print(f"  common-mode basis s* = {s_star:+.2f} mm (each U,V)")
    print(f"  V residual (V_pred from U,W  -  V_meas): median {med:+.3f} mm, RMS {rms:.2f} mm")
    print(f"  within ±½ V-pitch (3.83 mm): {frac_in*100:.1f}%")
    print(f"  => V fine-tune that would re-centre it: dV = {dV_tune:+.3f} mm "
          f"({dV_tune/O.UV_PITCH_MM:+.3f} V-strip)  [~0 => no room]")
    return cfg, rV, med, rms, dV_tune, frac_in


def main():
    store = persist.load(S2.V4)
    print("V fine-tune diagnostic: predicted-V (from U∩W) minus measured-V, after common mode.")
    results = [analyse(akey, store) for akey in (0, 4)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (cfg, rV, med, rms, dV, frac) in zip(axes, results):
        ax.hist(rV, bins=60, range=(-3*O.UV_PITCH_MM, 3*O.UV_PITCH_MM),
                color="C0", alpha=.85)
        ax.axvspan(-HALF_V, HALF_V, color="g", alpha=.15, label="±½ V-pitch")
        ax.axvline(0, color="k", lw=1)
        ax.axvline(med, color="r", ls="--", lw=1.5,
                   label=f"median {med:+.2f} mm  (V fine-tune {dV:+.2f} mm)")
        ax.set_xlabel("V$_{pred}$(from U∩W) − V$_{meas}$   [mm along V pitch]")
        ax.set_ylabel("ticks")
        ax.set_title(f"{cfg['label']}\nRMS {rms:.2f} mm, {frac*100:.0f}% within ½ V-pitch")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Does V have room to fine-tune?  (U and W fixed, common mode applied)",
                 fontsize=11)
    fig.tight_layout()
    for d in (OUT, PICS):
        fig.savefig(f"{d}/pdvd_v_finetune_residual.png", dpi=120)
    print(f"\nwrote {OUT}/pdvd_v_finetune_residual.png and {PICS}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Doc 23 section 7d companion figure: cathode-channel raw vs decon waveform
maps around the split flash pair of tracks B (evt 298609) and C (evt 298651).

Per track one figure, two panels: LEFT the ped-subtracted RAW full-stream ADC,
RIGHT the wiener-inspired decon (validated python port of Flash::OpDecon,
production templates pdvd-spe-templates.json + cathode wi_sigma = 1.25 MHz).
X = the 16 cathode DAPHNE channels (10xx), Y = raw light time (us,
PDVDOpWaveformSource t0), color = amplitude (log scale).  The two
reconstructed flash times are drawn as horizontal lines: the finder cuts the
single physical pulse between the fast peak and its slow (triplet) tail.

Run from pdvd/docs/qlmatch:  python3 scripts/aca_flash_wf.py
-> ../../docs/pics/pdvd_flash_split_wf_{B,C}.png  (~1 min: two 468864-pt FFT
   decons x 16 channels)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(PDVD, "pd_plot"))
import pdvd_light as pl                              # noqa: E402
from wct_wi_validate import wi_decon                 # noqa: E402

CFG = os.environ.get(
    "PDVD_CFG_DIR",
    "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd")
RAW = os.path.join(PDVD, "input_data_light",
    "np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root")
OUT = os.path.join(PDVD, "docs", "pics")
TICK_US = pl.TICK_NS / 1000.0
N_DECON = 468864              # production cathode branch samples (zero-pad)
WI_SIGMA = 1.25               # MHz, production cathode branch

# track -> (event, window us, [(flash raw us, PE, label), ...])
CASES = {
    "B": (298609, (5396.0, 5408.0),
          [(5399.991, 24384, "fast member (cathode fast peak)"),
           (5401.156, 19942, "late member (slow tail + self-trig PDs)")]),
    "C": (298651, (2798.0, 2806.0),
          [(2800.636, 13858, "fast member (cathode fast peak)"),
           (2801.947, 7382, "late member = matched #84")]),
}


def templates():
    j = json.load(open(os.path.join(CFG, "pdvd-spe-templates.json")))
    tmpl = {ch: np.asarray(j["templates"][ti]["values"], dtype=float)
            for ch, ti in zip(j["channels"], j["template_index"])}
    return tmpl


def main():
    import uproot
    tmpl = templates()
    d = uproot.open(RAW)["rawdump/raw_waveform"].arrays(
        ["event", "opchannel", "nsamp", "timestamp", "adc"], library="np")
    for trk, (evt, (lo, hi), flashes) in CASES.items():
        # raw-axis t0: min over ALL records (snippets shifted by trig_sample)
        sel = np.nonzero(d["event"] == evt)[0]
        starts = {i: int(round(d["timestamp"][i] * 62.5))
                  - (0 if d["nsamp"][i] > 1024 else 64) for i in sel}
        t0 = min(starts.values())

        cath = sorted((i for i in sel if d["opchannel"][i] // 1000 == 1),
                      key=lambda i: d["opchannel"][i])
        chans = [int(d["opchannel"][i]) for i in cath]
        raw_img, dec_img, t_us = [], [], None
        for i in cath:
            adc = np.asarray(d["adc"][i], dtype=np.float64)
            ped, _ = pl.pedestal_robust(adc)
            s0 = starts[i] - t0
            tt = (np.arange(adc.size) + s0) * TICK_US
            m = (tt >= lo) & (tt <= hi)
            if t_us is None:
                t_us = tt[m]
            raw_img.append(adc[m] - ped)
            dec = wi_decon(adc, tmpl[int(d["opchannel"][i])], N_DECON, WI_SIGMA)
            dec_img.append(dec[:adc.size][m])
            print(f"track {trk} ch{int(d['opchannel'][i])}:"
                  f" raw max {raw_img[-1].max():7.0f}"
                  f"  decon max {dec_img[-1].max():8.1f}")
        raw_img = np.array(raw_img).T    # [time, chan]
        dec_img = np.array(dec_img).T

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.5), sharey=True)
        for ax, img, name, unit in [
                (axes[0], raw_img, "raw (ped-sub)", "ADC"),
                (axes[1], dec_img, "decon (WI, prod templates)", "PE / 16 ns tick")]:
            vmax = np.percentile(img, 99.9)
            pos = np.clip(img, vmax * 1e-3, None)   # log floor at 0.1 % of max
            pc = ax.pcolormesh(np.arange(len(chans) + 1) - 0.5,
                               np.append(t_us, t_us[-1] + TICK_US),
                               pos, norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax),
                               cmap="viridis", rasterized=True)
            fig.colorbar(pc, ax=ax, label=unit, pad=0.01)
            for (tf, pe, lab), col in zip(flashes, ("w", "r")):
                ax.axhline(tf, color=col, ls="--", lw=1.3)
                ax.text(-0.35, tf, f"{tf:.2f} us, {pe} PE — {lab}",
                        color=col, fontsize=8, va="bottom", ha="left")
            ax.set_xticks(range(len(chans)))
            ax.set_xticklabels([str(c) for c in chans], rotation=90, fontsize=7)
            ax.set_xlabel("cathode DAPHNE channel")
            ax.set_title(name, fontsize=10)
        axes[0].set_ylabel("raw light axis (us, PDVDOpWaveformSource t0)")
        axes[0].set_ylim(lo, hi)
        fig.suptitle(f"track {trk} (evt {evt}): the two reconstructed flashes "
                     "cut one cathode pulse at its fast/slow-tail boundary",
                     fontsize=11)
        fig.tight_layout()
        p = os.path.join(OUT, f"pdvd_flash_split_wf_{trk}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"-> {p}\n")


if __name__ == "__main__":
    main()

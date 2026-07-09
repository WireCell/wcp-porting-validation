#!/usr/bin/env python3
"""Cathode XA long-time single-pulse response from bright stream pulses.

Evidence behind the template tail repair (spe_build.repair_fs_tail): median
of amplitude-normalized isolated bright pulses (>=30 PE) over a 51 us
window, per cathode channel.  Shows the true response = fast peak + positive
slow component (tau ~ 0.6-1.2 us, late light + slow XA response) decaying to
zero by ~25 us, with NO measurable AC undershoot (<5e-4 of the prompt) --
i.e. the negative near-constant tail of the raw 1-PE-averaged templates is a
harvest baseline bias (local pre-peak pedestals sit on falling tails of
earlier activity in the busy stream), not electronics.

Output: docs/pds/spe_cathode_longtail.png
Usage: spe_longtail.py [file_index]   (default 0 = run 039252)
"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdvd_light as pl
import spe_build as sb
import spe_compare as sc

TICK_US = pl.TICK_NS / 1000.0
NPRE, NPOST = 100, 3200        # -1.6 .. +51.2 us
PE_MIN = 30.0


def channel_median(f, ch, mode):
    """median amplitude-normalized shape over isolated bright pulses."""
    stack = []
    for evt in f.events:
        for r in f.records(evt, fullstream_only=True):
            if r["opch"] != ch:
                continue
            adc = r["adc"]
            med, _ = pl.pedestal_robust(adc)
            w = adc - med
            hp = sb.hpf(w)
            dsig = np.median(np.abs(np.diff(adc))) * 1.4826 / np.sqrt(2)
            pks, _ = find_peaks(hp, height=8 * dsig, distance=400)
            for p in pks:
                if p < NPRE + 50 or p + NPOST > len(w):
                    continue
                base = w[p - 90:p - 20].mean()
                amp = w[p] - base
                if amp / mode < PE_MIN or adc[p - 5:p + 5].max() >= pl.ADC_RAIL:
                    continue
                if hp[p - NPRE:p - 8].max() > 0.3 * amp:
                    continue
                stack.append((w[p - NPRE:p + NPOST] - base) / amp)
    return np.asarray(stack)


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    f = pl.LightFile(pl.data_files()[idx])
    harvest = sc.load_harvest(f.run)
    cath = [ch for ch in sorted(harvest) if ch < 2000]

    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    tt = np.arange(-NPRE, NPOST) * TICK_US
    print(f"run {f.run}: cathode bright-pulse (>= {PE_MIN:.0f} PE) median "
          f"shape / prompt amplitude")
    for ax, ch in zip(axes.flat, cath):
        stack = channel_median(f, ch, float(harvest[ch]["mode"]))
        if len(stack) < 5:
            ax.set_title(f"ch{ch}: n={len(stack)} (too few)", fontsize=9)
            ax.axis("off")
            continue
        avg = np.median(stack, axis=0)
        ax.plot(tt, avg, lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(alpha=0.3)
        # AC-recovery probe: fit beyond 12 us where the light has died
        m = tt > 12.0
        try:
            popt, _ = curve_fit(lambda t, a, tau: a * np.exp(-t / tau),
                                tt[m], avg[m], p0=(avg[m][0], 20.0),
                                maxfev=20000)
            ax.plot(tt[m], popt[0] * np.exp(-tt[m] / popt[1]), "r--", lw=1)
        except Exception:
            pass
        ax.set_title(f"ch{ch}  n={len(stack)}", fontsize=9)
        print(f"  ch{ch}: n={len(stack):4d}  @6.4us={avg[NPRE + 400]:+.4f}  "
              f"@12us={avg[NPRE + 750]:+.4f}  @25us={avg[NPRE + 1600]:+.4f}  "
              f"@50us={avg[NPRE + 3150]:+.4f}")
    for ax in axes[-1]:
        ax.set_xlabel("us from peak")
    fig.suptitle(f"PDVD run {f.run}: cathode XA median bright-pulse shape / "
                 f"prompt amplitude (symlog) — positive slow tail, no AC "
                 f"undershoot down to ~5e-4")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(pl.PICS, "spe_cathode_longtail.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()

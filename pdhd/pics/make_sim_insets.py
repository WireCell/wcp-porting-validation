#!/usr/bin/env python3
"""Generated physics insets for the PD-HD sim-chain diagram.

Counterpart of the PD-VD sim-chain inset sources.  PD-VD copied its
cold-electronics and noise-spectra panels from the DNN_ROI_SP study; PD-HD has
no such study panels, so both are generated here directly from the production
configuration inputs:

  elec_response.png       - the ColdElecResponse impulse e(t) used by
                            DepoTransform (single model, all 4 APAs):
                            gain 14 mV/fC, shaping 2.2 us, postgain 1.0
                            (cfg/pgrapher/experiment/pdhd/params.jsonnet).
  noise_input_spectra.png - the measured amplitude spectra that seed
                            EmpiricalNoiseModel + AddNoise:
                            protodunehd-noise-spectra-14mVfC-v1.json.bz2,
                            one normalized curve per plane (U/V/W).

Output: pdhd/pics/sim_chain_src/{elec_response,noise_input_spectra}.png
"""
import bz2
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wirecell import units
from wirecell.sigproc.response import electronics
from wirecell.util.fileio import wirecell_path

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sim_chain_src")
NOISE_FILE = "protodunehd-noise-spectra-14mVfC-v1.json.bz2"


def find_wcdata(fn):
    for d in wirecell_path():
        c = os.path.join(d, fn)
        if os.path.exists(c):
            return c
    raise FileNotFoundError(fn)


def make_elec_response(out="elec_response.png"):
    """ColdElecResponse impulse: 14 mV/fC peak gain, 2.2 us shaping."""
    gain = 14.0 * units.mV / units.fC
    shaping = 2.2 * units.us
    t = np.linspace(0, 10 * units.us, 1000)
    e = electronics(t, gain, shaping)          # [voltage/charge]
    e_mVfC = e / (units.mV / units.fC)
    t_us = t / units.us

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(t_us, e_mVfC, lw=2.0, color="#1f4e9b")
    ax.axhline(0, color="0.4", lw=0.6)
    pk = int(np.argmax(e_mVfC))
    ax.annotate("peak gain 14 mV/fC",
                xy=(t_us[pk], e_mVfC[pk]),
                xytext=(t_us[pk] + 2.2, e_mVfC[pk] * 0.9),
                fontsize=10, color="#b22222",
                arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.0))
    ax.axvline(2.2, color="0.6", lw=0.8, ls="--")
    ax.text(2.32, e_mVfC.max() * 0.12, "shaping 2.2 µs", fontsize=9.5,
            color="0.35", rotation=90, va="bottom")
    ax.set_xlabel("time  [µs]", fontsize=11)
    ax.set_ylabel("e(t)  [mV/fC]", fontsize=11)
    ax.set_title("cold electronics impulse — ColdElecResponse (all APAs)",
                 fontsize=10.5)
    ax.tick_params(labelsize=10)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(os.path.join(SRC, out), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "peak %.2f mV/fC @ %.2f us" % (e_mVfC[pk], t_us[pk]))


def make_noise_spectra(out="noise_input_spectra.png"):
    """Measured amplitude spectra seeding AddNoise — one curve per plane."""
    spectra = json.load(bz2.open(find_wcdata(NOISE_FILE)))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    style = {0: ("#1f77b4", "U (induction)"),
             1: ("#d62728", "V (induction)"),
             2: ("#2ca02c", "W (collection)")}
    for plane in (0, 1, 2):
        # longest-wire entry of each plane (the curves within a plane are
        # nearly identical across the two wire-length bins)
        e = max((x for x in spectra if x["plane"] == plane),
                key=lambda x: x["wirelen"])
        f_MHz = np.asarray(e["freqs"]) / units.megahertz
        amp = np.asarray(e["amps"])
        half = len(f_MHz) // 2 + 1
        col, lab = style[plane]
        ax.plot(f_MHz[:half], amp[:half] / amp.max(), lw=1.8, color=col,
                label="%s, wire %.1f m" % (lab, e["wirelen"] / units.meter))
    ax.set_xlabel("frequency  [MHz]", fontsize=11)
    ax.set_ylabel("spectral amplitude  [a.u.]", fontsize=11)
    ax.set_title("noise input spectra — measured, per plane", fontsize=10.5)
    ax.tick_params(labelsize=10)
    ax.margins(x=0)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(SRC, out), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "planes U/V/W longest-wire entries")


def main():
    os.makedirs(SRC, exist_ok=True)
    make_elec_response()
    make_noise_spectra()


if __name__ == "__main__":
    main()

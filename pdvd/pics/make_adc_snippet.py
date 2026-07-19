#!/usr/bin/env python3
"""Time-domain raw-ADC waveform inset for the PD-VD sim-chain diagram.

Uses a REAL ProtoDUNE-VD *data* raw-ADC frame (run 41189, anode 0) as the
illustrative digitizer-output waveform.  Reads the pre-existing raw frame
archive, auto-selects a clean mid-readout collection pulse (max peak-over-noise
with the peak away from the readout edges), baseline-subtracts, and plots
ADC counts vs tick with the flat electronics-noise floor visible on both sides.

Input : pdvd/input_data/run41189/protodune-sp-frames-raw-anode0.tar.bz2
        (member frame_raw0_4.npy = [nchan, nticks] ADC, channels_raw0_4.npy)
Output: pdvd/pics/sim_chain_src/adc_snippet.png
"""
import io
import os
import tarfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(HERE)
TAR = os.path.join(PDVD, "input_data", "run41189",
                   "protodune-sp-frames-raw-anode0.tar.bz2")
OUT = os.path.join(HERE, "sim_chain_src", "adc_snippet.png")
TICK_US = 0.5  # PD-VD bottom-CRP readout: 500 ns/tick


def load_frame(tar_path):
    with tarfile.open(tar_path, "r:bz2") as tf:
        fr = ch = None
        for m in tf.getmembers():
            if m.name.startswith("frame_"):
                fr = np.load(io.BytesIO(tf.extractfile(m).read()))
            elif m.name.startswith("channels_"):
                ch = np.load(io.BytesIO(tf.extractfile(m).read()))
    return fr, ch


V_ROWS = (476, 952)   # V-plane (induction) channel band, anode-0 raw frame


def main():
    a, c = load_frame(TAR)
    amp = a - np.median(a, axis=1)[:, None]
    rms = np.std(a[:, :400], axis=1)
    lo_r, hi_r = V_ROWS
    seg = amp[lo_r:hi_r]
    pos = seg.max(axis=1)
    neg = -seg.min(axis=1)
    ptk = np.abs(seg).argmax(axis=1)
    # clean bipolar induction pulse (both lobes strong) away from edges
    bip = np.minimum(pos, neg) / (rms[lo_r:hi_r] + 1e-6)
    ok = (ptk > 1500) & (ptk < 6000)
    j = np.argmax(np.where(ok, bip, -1))
    row = lo_r + j
    ch_id, noise = int(c[row]), float(rms[row])
    w = amp[row]
    p = int(np.abs(w).argmax())
    ppk, ntk = int(w[:p + 40].argmax()), int(w.argmin())
    lo, hi = max(0, p - 190), min(a.shape[1], p + 250)
    x = np.arange(lo, hi)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(x, w[lo:hi], lw=0.8, color="#1f3b73")
    ax.axhline(0, color="0.4", lw=0.6)
    ax.axhspan(-3 * noise, 3 * noise, color="0.85", zorder=0,
               label="noise floor  ±3σ (σ≈%.0f ADC)" % noise)
    ax.annotate("bipolar induction\nsignal", xy=(ntk, w[ntk]),
                xytext=(ntk + 55, w[ntk] * 0.62), fontsize=8, color="#b22222",
                arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.0))
    ax.set_xlabel("readout tick  (500 ns / tick)", fontsize=9)
    ax.set_ylabel("ADC (pedestal-subtracted)", fontsize=9)
    ax.set_title("raw-ADC waveform — V plane (induction), "
                 "ProtoDUNE-VD data run 41189", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.margins(x=0)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("row=%d ch=%d +%.0f/%.0f ADC @tick %d  noise σ=%.1f ADC -> %s"
          % (row, ch_id, w[ppk], w[ntk], p, noise, OUT))


if __name__ == "__main__":
    main()

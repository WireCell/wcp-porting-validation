#!/usr/bin/env python3
"""PD-HD Wire-Cell TPC signal+noise simulation-chain presentation diagram.

Counterpart of pdvd/pics/make_sim_chain_diagram.py (PD-VD original) with the
ProtoDUNE-HD configuration facts.

Wide 16:9 figure for pptx.  Draws the per-APA simulation dataflow
(G4 depos -> Drifter -> Fanout -> DepoTransform [signal] -> Reframer
-> AddNoise [noise] -> Digitizer -> raw ADC).  Drifter + Fanout are boxed as a
"DRIFT SIMULATION" group (Wire-Cell stages — only the G4 deposits are external
LArSoft input); the signal path is blue, the noise path orange.  Embeds five
physics "inset" panels (drift+diffusion, field response, cold-electronics
impulse, noise input spectra, a real data raw-ADC waveform) with leader lines
to the node each explains.

Inset sources live in sim_chain_src/ (committed).  Output:
pdhd/pics/pdhd_sim_chain.{png,pdf}.
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sim_chain_src")
OUT = os.path.join(HERE, "pdhd_sim_chain")

# ---- palette -------------------------------------------------------------
C_SIG = "#1f4e9b"      # signal path
C_NOI = "#d3781f"      # noise path
C_DRF = "#6a3d9a"      # drift simulation (Wire-Cell)
C_IN = "#4a4a4a"       # external input (LArSoft/G4)
C_OUT = "#2e7d4f"      # output
BG_SIG = "#e8f0fb"
BG_NOI = "#fbeede"
BG_DRF = "#efe8f7"
INK = "#20242b"

# 16 x 9 world; figure is 19.2x10.8in => exactly 1.2 in per world-unit on
# BOTH axes, so world units are physically square (image aspect preserved).
W, H = 16.0, 9.0
FIGIN = (19.2, 10.8)


def crop(name, box):
    """box = (l, t, r, b) as fractions of the source image."""
    im = Image.open(os.path.join(SRC, name)).convert("RGB")
    w, h = im.size
    l, t, r, b = box
    return im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


# V plane illustrates throughout.
INSETS = {
    "drift": crop("drifter_arrival.png", (0.0, 0.0, 1.0, 1.0)),
    "field": crop("field_response_2d_time_V.png", (0.0, 0.0, 1.0, 1.0)),
    "elec": crop("elec_response.png", (0.0, 0.0, 1.0, 1.0)),
    "noise": crop("noise_input_spectra.png", (0.0, 0.0, 1.0, 1.0)),
    "adc": crop("adc_snippet.png", (0.0, 0.0, 1.0, 1.0)),
}


def main():
    fig = plt.figure(figsize=FIGIN)
    fig.patch.set_facecolor("white")
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, W)
    ov.set_ylim(0, H)
    ov.axis("off")

    def box(cx, cy, w, h, text, fc, ec, fs=13, tc=None, bold=True, pad=0.02):
        p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                           boxstyle="round,pad=%.3f,rounding_size=0.14" % pad,
                           linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=5)
        ov.add_patch(p)
        ov.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=tc or INK, zorder=6,
                fontweight="bold" if bold else "normal")

    def arrow(p0, p1, color, lw=2.6, ms=22, ls="-"):
        ov.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                     mutation_scale=ms, linewidth=lw,
                                     color=color, linestyle=ls,
                                     shrinkA=2, shrinkB=2, zorder=4))

    def leader(p_from, p_to, color):
        ov.add_patch(FancyArrowPatch(p_from, p_to, arrowstyle="-",
                                     linewidth=1.4, color=color, alpha=0.85,
                                     linestyle=(0, (5, 3)), zorder=3))
        ov.add_patch(Circle(p_to, 0.055, color=color, zorder=6))

    # ---- title band ------------------------------------------------------
    ov.text(W / 2, 8.62, "ProtoDUNE Horizontal Drift — Wire-Cell TPC "
            "Signal + Noise Simulation Chain", ha="center", va="center",
            fontsize=24, fontweight="bold", color=INK)
    ov.text(W / 2, 8.06,
            r"per APA:   "
            r"$\mathrm{raw\ ADC}(t)=\mathcal{D}\left[\,\left(Q_{\rm drift}"
            r"\ast\,\mathrm{FR}\ast E\right)(t)\;+\;n(t)\,\right]$",
            ha="center", va="center", fontsize=16, color="#444")

    # ---- group backgrounds ----------------------------------------------
    yspine = 6.35
    # Drift simulation groups Drifter + Fanout: these ARE Wire-Cell stages
    # (only the G4 deposits are external LArSoft input).
    ov.add_patch(FancyBboxPatch((2.30, yspine - 0.95), 3.05, 1.9,
                 boxstyle="round,pad=0.02,rounding_size=0.16",
                 linewidth=1.6, edgecolor=C_DRF, facecolor=BG_DRF,
                 alpha=0.9, zorder=1))
    ov.text(3.83, yspine + 1.18, "DRIFT SIMULATION", ha="center",
            fontsize=13.5, fontweight="bold", color=C_DRF)
    ov.add_patch(FancyBboxPatch((5.72, yspine - 0.95), 3.86, 1.9,
                 boxstyle="round,pad=0.02,rounding_size=0.16",
                 linewidth=1.6, edgecolor=C_SIG, facecolor=BG_SIG,
                 alpha=0.9, zorder=1))
    ov.text(7.65, yspine + 1.18, "SIGNAL SIMULATION", ha="center",
            fontsize=13.5, fontweight="bold", color=C_SIG)
    ov.add_patch(FancyBboxPatch((9.74, 4.02), 1.86, 3.28,
                 boxstyle="round,pad=0.02,rounding_size=0.16",
                 linewidth=1.6, edgecolor=C_NOI, facecolor=BG_NOI,
                 alpha=0.9, zorder=1))
    ov.text(10.67, yspine + 1.18, "NOISE SIMULATION", ha="center",
            fontsize=13.5, fontweight="bold", color=C_NOI)

    # ---- spine nodes -----------------------------------------------------
    n_depos = (1.25, yspine)
    n_drift = (3.15, yspine)
    n_fan = (4.75, yspine)
    n_depotr = (6.9, yspine)
    n_refr = (8.75, yspine)
    n_add = (10.67, yspine)
    n_digi = (12.55, yspine)
    n_raw = (14.55, yspine)
    n_noise = (10.67, 4.72)

    box(*n_depos, 1.7, 1.15, "G4 energy\ndeposits\n(LArSoft)", "#eef1f5",
        C_IN, fs=12.5)
    box(*n_drift, 1.5, 1.2, "Drifter\n$v_d$=1.565\nmm/µs\n+ diffusion",
        BG_DRF, C_DRF, fs=12, tc=C_DRF)
    box(*n_fan, 1.4, 1.05, "Fanout\n×4 APAs", BG_DRF, C_DRF, fs=12.5,
        tc=C_DRF)
    box(*n_depotr, 2.5, 1.5, "DepoTransform\n\nfield response\n∗ "
        "electronics", "white", C_SIG, fs=13, tc=C_SIG)
    box(*n_refr, 1.45, 1.1, "Reframer\n6000 tick", "white", C_SIG, fs=12,
        tc=C_SIG)
    box(*n_digi, 1.7, 1.15, "Digitizer\n14-bit ADC", "#eaf5ee", C_OUT,
        fs=13, tc=C_OUT)
    box(*n_raw, 1.7, 1.15, "raw ADC\nframe", "#eaf5ee", C_OUT, fs=13.5,
        tc=C_OUT)
    # WCT tag of the digitized frame (same convention note as the NF slide)
    ov.text(n_raw[0], n_raw[1] - 0.40, "WCT tag: orig{N}", ha="center",
            va="center", fontsize=9.5, color="#6b7178", style="italic",
            zorder=6)
    box(*n_noise, 2.1, 1.35, "Empirical\nNoiseModel\n(measured\nspectra)",
        "white", C_NOI, fs=12, tc=C_NOI)

    # AddNoise merge symbol
    ov.add_patch(Circle(n_add, 0.42, facecolor="white", edgecolor=C_NOI,
                        linewidth=2.4, zorder=5))
    ov.text(n_add[0], n_add[1], "+", ha="center", va="center", fontsize=26,
            fontweight="bold", color=C_NOI, zorder=6)
    ov.text(n_add[0], n_add[1] - 0.72, "AddNoise", ha="center", fontsize=10.5,
            color=C_NOI, zorder=6)

    # ---- spine arrows ----------------------------------------------------
    def rt(node, w):   # right edge
        return (node[0] + w / 2, node[1])

    def lf(node, w):
        return (node[0] - w / 2, node[1])

    arrow(rt(n_depos, 1.7), lf(n_drift, 1.5), C_IN)
    arrow(rt(n_drift, 1.5), lf(n_fan, 1.4), C_DRF)
    arrow(rt(n_fan, 1.4), lf(n_depotr, 2.5), C_DRF)
    arrow(rt(n_depotr, 2.5), lf(n_refr, 1.45), C_SIG)
    arrow(rt(n_refr, 1.45), (n_add[0] - 0.42, n_add[1]), C_SIG)
    arrow((n_noise[0], n_noise[1] + 0.675), (n_add[0], n_add[1] - 0.42), C_NOI)
    arrow((n_add[0] + 0.42, n_add[1]), lf(n_digi, 1.7), C_OUT)
    arrow(rt(n_digi, 1.7), lf(n_raw, 1.7), C_OUT)

    # ---- legend (top-left margin, clear of the flow band) ---------------
    lx, ly, dy = 0.55, 8.32, 0.27
    for i, (col, lab) in enumerate([
            (C_IN, "G4 input (LArSoft)"), (C_DRF, "drift sim"),
            (C_SIG, "signal path"), (C_NOI, "noise path"),
            (C_OUT, "digitized output")]):
        ov.add_patch(FancyArrowPatch((lx, ly - i * dy),
                     (lx + 0.5, ly - i * dy), arrowstyle="-|>",
                     mutation_scale=13, lw=2.4, color=col, zorder=6))
        ov.text(lx + 0.65, ly - i * dy, lab, va="center", fontsize=10.5,
                color=INK)

    # ---- insets (one bottom-aligned row) --------------------------------
    YB = 1.55   # common bottom of every inset image

    def place(key, cx, wd, caption, target, color):
        im = INSETS[key]
        iw, ih = im.size
        hd = wd * ih / iw
        ax = fig.add_axes([(cx - wd / 2) / W, YB / H, wd / W, hd / H])
        ax.imshow(np.asarray(im))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(1.8)
        ov.text(cx, YB - 0.22, caption, ha="center", va="top",
                fontsize=12, color=color, fontweight="bold")
        leader((cx, YB + hd + 0.04), target, color)

    place("drift", 1.90, 2.85, "drift + longitudinal diffusion",
          (n_drift[0], n_drift[1] - 0.6), C_DRF)
    place("field", 5.05, 2.85, "field response — V plane",
          (n_depotr[0] - 0.55, n_depotr[1] - 0.75), C_SIG)
    place("elec", 8.05, 2.85, "cold-electronics impulse  $e(t)$",
          (n_depotr[0] + 0.55, n_depotr[1] - 0.75), C_SIG)
    place("noise", 11.05, 2.85, "noise input spectra — per plane",
          (n_noise[0] - 0.5, n_noise[1] - 0.675), C_NOI)
    place("adc", 14.10, 2.85, "raw-ADC waveform — V plane  (data)",
          (n_raw[0] - 0.15, n_raw[1] - 0.575), C_OUT)

    # ---- provenance footer ----------------------------------------------
    ov.text(W / 2, 0.42,
            "Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/pdhd/sim.jsonnet  "
            "(per-APA signal+noise)  ·  Drifter $D_L$=6.2 / $D_T$=16.3 cm²/s"
            "  ·  FR dune-garfield-1d565 (APA0 own fit)  ·  ColdElec "
            "14 mV/fC, 2.2 µs  ·  noise protodunehd-noise-spectra-14mVfC-v1"
            "  ·  waveform: PDHD data run 027409",
            ha="center", va="center", fontsize=9.3, color="#8a8a8a")

    fig.savefig(OUT + ".png", dpi=200, facecolor="white")
    fig.savefig(OUT + ".pdf", facecolor="white")
    print("wrote", OUT + ".png / .pdf")


if __name__ == "__main__":
    main()

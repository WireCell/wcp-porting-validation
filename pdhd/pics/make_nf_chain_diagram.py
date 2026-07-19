#!/usr/bin/env python3
"""ProtoDUNE-HD Noise-Filtering algorithm diagram (wide 16:9, for slides).

Draws the live OmnibusNoiseFilter cascade as the toolkit actually runs it on
PDHD (nf.jsonnet / wct-nf-sp.jsonnet):

  orig raw ADC -> [Resampler 512->500 ns, data only]
   -> OmnibusNoiseFilter:
        1 PDHDOneChannelNoise   (per channel)
        2 PDHDFEMBNoiseSub      (per FEMB, multigroup)
        3 PDHDCoherentNoiseSub  (per FEMB group)
   -> raw{N} cleaned ADC

Only live code paths are drawn as active; configured-but-dormant machinery is
listed in a greyed "OFF in this build" panel (see nf.md).  Two real-data insets
(coherent-noise 2D before/after; per-channel noise RMS pre/post) sit below.

Output: pdhd/pics/pdhd_nf_chain.{png,pdf}
"""
import os
from diagram_helpers import Canvas

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "nfsp_src")

C_NF = "#c85a11"      # noise-filter algorithms
BG_NF = "#fbeede"
C_IN = "#4a4a4a"      # input / passthrough
BG_IN = "#eef1f5"
C_OUT = "#2e7d4f"     # cleaned output
BG_OUT = "#eaf5ee"
C_OFF = "#9aa0a8"     # dormant / OFF
BG_OFF = "#f1f1f3"


def main():
    c = Canvas()
    c.title("ProtoDUNE-HD Wire-Cell Noise Filtering  ·  OmnibusNoiseFilter (per APA)",
            r"raw ADC  →  per-channel baseline  →  FEMB negative-pulse  →  "
            r"coherent common-mode subtraction  →  cleaned ADC")

    ys = 6.75
    # ---- input + data-only resampler ------------------------------------
    c.box(1.30, ys, 1.95, 1.15, "orig raw ADC\nframe\n(per APA)", BG_IN, C_IN,
          fs=11.5)
    c.box(3.35, ys, 1.75, 1.0, "Resampler\n512→500 ns\n(data only)", BG_OFF,
          C_OFF, fs=10.5, tc="#6b7178")

    # ---- OmnibusNoiseFilter group ---------------------------------------
    gx0, gy0, gw, gh = 4.55, 5.05, 9.55, 2.92
    c.group_bg(gx0, gy0, gw, gh, C_NF, BG_NF,
               label="OmnibusNoiseFilter   (one per APA)", ly=gy0 + gh - 0.24,
               fs=13)

    yb = 6.25
    c.algobox(6.05, yb, 2.55, 2.30,
              "① OneChannelNoise", [
                  "per channel:",
                  "• FFT → zero DC bin → IFFT",
                  "• dynamic baseline:",
                  "   clip ±6σ, subtract",
                  "   binned median",
              ], "white", C_NF, title_fs=12)
    c.algobox(8.95, yb, 2.35, 2.30,
              "② FEMBNoiseSub", [
                  "per FEMB (multigroup):",
                  "• detect coherent",
                  "   negative-pulse dips",
                  "• width 50, 3.5σ",
                  "   → restore baseline",
              ], "white", C_NF, title_fs=12)
    c.algobox(12.05, yb, 3.35, 2.30,
              "③ CoherentNoiseSub", [
                  "per FEMB group (40 ch U/V, 48 ch W):",
                  "• A  CalcMedian — per-tick group median",
                  "• B  SignalProtection — shield real signal",
                  "        (ADC-domain + deconv-domain ROI)",
                  "• C  Subtract_WScaling — per-ch coef",
                  "        Σ(s·m)/Σ(m²), clip [0,1.5]",
              ], "white", C_NF, title_fs=12, bullet_fs=9.3)

    # ---- output ---------------------------------------------------------
    c.box(15.0, ys, 1.7, 1.15, "raw{N}\ncleaned\nADC frame", BG_OUT, C_OUT,
          fs=11.5, tc=C_OUT)

    # ---- spine arrows ----------------------------------------------------
    c.arrow((2.275, ys), (2.475, ys), C_IN)
    c.arrow((4.225, ys), (4.78, yb + 0.4), C_IN)    # resampler -> group/box1
    c.arrow((7.325, yb), (7.775, yb), C_NF)         # box1 -> box2
    c.arrow((10.125, yb), (10.375, yb), C_NF)       # box2 -> box3
    c.arrow((13.725, yb + 0.4), (14.15, ys), C_OUT) # box3 -> output

    # ---- OFF-in-this-build panel (bottom-left) --------------------------
    ox0, oy0, ow, oh = 0.45, 1.25, 3.35, 2.55
    c.group_bg(ox0, oy0, ow, oh, C_OFF, BG_OFF,
               label="available in code — OFF here", ly=oy0 + oh - 0.24,
               fs=10.5)
    off_lines = [
        "PDHD cold electronics is DC-coupled,",
        "so these MicroBooNE-era paths are",
        "left at their C++ defaults:",
        "",
        "• adaptive baseline / IS_RC partial",
        "• RC+RC undershoot deconvolution",
        "• per-channel freq-notch masks",
        "• sticky-bit / ledge detection",
        "• min/max-RMS noisy tagging",
    ]
    yy = oy0 + oh - 0.58
    for ln in off_lines:
        c.ov.text(ox0 + 0.20, yy, ln, ha="left", va="center", fontsize=9.2,
                  color="#6b7178", zorder=2)
        yy -= 0.232

    # ---- insets (bottom) -------------------------------------------------
    ybi = 1.30
    c.place_image(os.path.join(SRC, "nf_coherent_2d.png"), 7.55, 6.1, ybi,
                  "coherent-noise subtraction (data)", (12.05, 5.10), C_NF)
    c.place_image(os.path.join(SRC, "nf_noise_rms.png"), 13.15, 5.0, ybi,
                  "noise RMS: pre-NF → post-NF (data)", (15.0, 5.95), C_OUT)

    c.footer("Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/pdhd/{nf,chndb-base,"
             "chndb-resp}.jsonnet + wct-nf-sp.jsonnet  ·  impl sigproc/src/"
             "ProtoduneHD.cxx  ·  insets: ProtoDUNE-HD data run 027409 evt 0, APA1 V")
    c.save(os.path.join(HERE, "pdhd_nf_chain"))


if __name__ == "__main__":
    main()

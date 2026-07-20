#!/usr/bin/env python3
"""ProtoDUNE-HD Noise-Filtering algorithm diagram (wide 16:9, for slides).

Draws the live OmnibusNoiseFilter cascade as the toolkit actually runs it on
PDHD (nf.jsonnet / wct-nf-sp.jsonnet):

  raw ADC (DAQ or WCT sim) -> [Resampler 512->500 ns, data only]
   -> OmnibusNoiseFilter:
        1 PDHDOneChannelNoise   (per channel)
        2 PDHDFEMBNoiseSub      (per FEMB, multigroup)
        3 PDHDCoherentNoiseSub  (per FEMB group)
   -> cleaned ADC frame

WCT tag convention (a known source of confusion): the *input* raw ADC is tagged
`orig` in WCT, and the NF-*cleaned* output is tagged `raw{N}` — i.e. WCT's "raw"
is everyone else's "cooked".  The boxes therefore carry the physical name big
and the WCT tag as a small secondary line.

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
C_IN = "#3a3f47"      # input / raw ADC
BG_IN = "#eef1f5"
C_WC = "#2f6ba6"      # WC-domain driver stage (Resampler)
BG_WC = "#e7f0f9"
C_OUT = "#2e7d4f"     # cleaned output
BG_OUT = "#eaf5ee"
C_OFF = "#9aa0a8"     # dormant / OFF
BG_OFF = "#f1f1f3"
INK = "#20242b"


def main():
    c = Canvas()
    c.title("ProtoDUNE-HD Wire-Cell Noise Filtering  ·  OmnibusNoiseFilter (per APA)",
            r"raw ADC  →  per-channel baseline  →  FEMB negative-pulse  →  "
            r"coherent common-mode subtraction  →  cleaned ADC")

    ys = 6.40
    # ---- input (raw ADC, either DAQ or WCT sim) -------------------------
    c.stack_box(1.55, ys, 2.30, 1.85, [
        ("raw ADC frame", 15, True, INK, False),
        ("(per APA)", 11.5, False, INK, False),
        ("source: DAQ or WCT sim", 11.5, False, C_IN, False),
        ("WCT tag: orig", 11, False, "#6b7178", True),
    ], BG_IN, C_IN, gap=0.42)

    # ---- data-only resampler (a WC-domain driver stage — colored) -------
    c.stack_box(4.15, ys, 1.95, 1.55, [
        ("Resampler", 14, True, C_WC, False),
        ("512 → 500 ns", 12, False, INK, False),
        ("data only", 11, False, C_WC, True),
    ], BG_WC, C_WC, gap=0.44)

    # ---- OmnibusNoiseFilter group ---------------------------------------
    gx0, gy0, gw, gh = 5.35, 4.55, 9.05, 3.30
    c.group_bg(gx0, gy0, gw, gh, C_NF, BG_NF,
               label="OmnibusNoiseFilter   (one per APA)", ly=gy0 + gh - 0.30,
               fs=15.5)

    yb = 5.95
    c.algobox(6.75, yb, 2.55, 2.62,
              "① OneChannelNoise", [
                  "per channel:",
                  "• FFT → zero DC bin → IFFT",
                  "• dynamic baseline:",
                  "   clip ±6σ, subtract",
                  "   binned median",
              ], "white", C_NF, title_fs=14, bullet_fs=11, dy=0.36)
    c.algobox(9.55, yb, 2.55, 2.62,
              "② FEMBNoiseSub", [
                  "per FEMB (multigroup):",
                  "• detect coherent",
                  "   negative-pulse dips",
                  "• width 50, 3.5σ",
                  "   → restore baseline",
              ], "white", C_NF, title_fs=14, bullet_fs=11, dy=0.36)
    c.algobox(12.55, yb, 3.35, 2.62,
              "③ CoherentNoiseSub", [
                  "per FEMB group (40 ch U/V, 48 ch W):",
                  "• A  CalcMedian — per-tick group median",
                  "• B  SignalProtection — shield real signal",
                  "        (ADC-domain + deconv-domain ROI)",
                  "• C  Subtract_WScaling — per-ch coef",
                  "        Σ(s·m)/Σ(m²), clip [0,1.5]",
              ], "white", C_NF, title_fs=14, bullet_fs=10.3, dy=0.365)

    # ---- output ---------------------------------------------------------
    c.stack_box(15.15, ys, 1.75, 1.7, [
        ("cleaned ADC", 14, True, C_OUT, False),
        ("frame (per APA)", 12, False, INK, False),
        ("WCT tag: raw{N}", 11, False, "#5f8f72", True),
    ], BG_OUT, C_OUT, gap=0.44)

    # ---- spine arrows ----------------------------------------------------
    c.arrow((2.70, ys), (3.175, ys), C_IN)          # input -> resampler
    c.arrow((5.125, ys), (5.62, yb + 0.4), C_WC)    # resampler -> group/box1
    c.arrow((8.025, yb), (8.275, yb), C_NF)         # box1 -> box2
    c.arrow((10.825, yb), (11.075, yb), C_NF)       # box2 -> box3
    c.arrow((14.225, yb + 0.4), (14.275, ys), C_OUT)  # box3 -> output

    # ---- OFF-in-this-build panel (bottom-left) --------------------------
    ox0, oy0, ow, oh = 0.45, 1.05, 3.55, 3.00
    c.group_bg(ox0, oy0, ow, oh, C_OFF, BG_OFF,
               label="available in code — OFF here", ly=oy0 + oh - 0.28,
               fs=12.5)
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
    yy = oy0 + oh - 0.70
    for ln in off_lines:
        c.ov.text(ox0 + 0.22, yy, ln, ha="left", va="center", fontsize=11,
                  color="#6b7178", zorder=2)
        yy -= 0.262

    # ---- insets (bottom) -------------------------------------------------
    ybi = 1.05
    c.place_image(os.path.join(SRC, "nf_coherent_2d.png"), 7.30, 5.20, ybi,
                  "coherent-noise subtraction (data)", (12.55, 4.60), C_NF)
    c.place_image(os.path.join(SRC, "nf_noise_rms.png"), 13.10, 5.30, ybi,
                  "noise RMS: pre-NF → post-NF (data)", (15.15, 5.55), C_OUT)

    c.footer("Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/pdhd/{nf,chndb-base,"
             "chndb-resp}.jsonnet + wct-nf-sp.jsonnet  ·  impl sigproc/src/"
             "ProtoduneHD.cxx  ·  insets: ProtoDUNE-HD data run 027409 evt 0, APA1 V")
    c.save(os.path.join(HERE, "pdhd_nf_chain"))


if __name__ == "__main__":
    main()

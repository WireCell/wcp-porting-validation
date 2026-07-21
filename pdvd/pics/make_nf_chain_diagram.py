#!/usr/bin/env python3
"""ProtoDUNE-VD Noise-Filtering algorithm diagram (wide 16:9, for slides).

Counterpart of pdhd/pics/make_nf_chain_diagram.py (PD-HD original) with the
ProtoDUNE-VD cascade.  Draws the live OmnibusNoiseFilter chain as the toolkit
actually runs it on PDVD (protodunevd/nf.jsonnet / wct-nf-sp*.jsonnet):

  raw ADC (DAQ or WCT sim) -> [Resampler 512->500 ns, bottom CRPs, data only]
   -> OmnibusNoiseFilter:
        1 PDVDOneChannelNoise    (per channel)
        2 PDVDCoherentNoiseSub   (per conduit group)
        3 PDVDShieldCouplingSub  (top-CRP U strips only)
   -> cleaned ADC frame

WCT tag convention (a known source of confusion): the *input* raw ADC is tagged
`orig` in WCT, and the NF-*cleaned* output is tagged `raw{N}` — i.e. WCT's "raw"
is everyone else's "cooked".  The boxes therefore carry the physical name big
and the WCT tag as a small secondary line.

Only live code paths are drawn as active; configured-but-dormant machinery is
listed in a greyed "OFF in this build" panel.  Two real-data insets
(coherent-noise 2D before/after; per-channel noise RMS pre/post) sit below.

Output: pdvd/pics/pdvd_nf_chain.{png,pdf}
"""
import os
from diagram_helpers_v2 import Canvas

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
    c.title("ProtoDUNE-VD Wire-Cell Noise Filtering  ·  OmnibusNoiseFilter "
            "(per CRP)",
            r"raw ADC  →  per-channel baseline  →  coherent common-mode "
            r"subtraction  →  shield coupling (top U)  →  cleaned ADC")

    ys = 6.40
    # ---- input (raw ADC, either DAQ or WCT sim) -------------------------
    c.stack_box(1.35, ys, 2.20, 1.85, [
        ("raw ADC frame", 15, True, INK, False),
        ("(per CRP anode)", 11.5, False, INK, False),
        ("source: DAQ or WCT sim", 11.5, False, C_IN, False),
        ("WCT tag: orig", 11, False, "#6b7178", True),
    ], BG_IN, C_IN, gap=0.42)

    # ---- data-only resampler (a WC-domain driver stage — colored) -------
    c.stack_box(3.55, ys, 1.90, 1.70, [
        ("Resampler", 14, True, C_WC, False),
        ("512 → 500 ns", 12, False, INK, False),
        ("bottom CRPs · data", 10.5, False, C_WC, True),
        ("(top: tick relabel)", 10, False, "#6b7178", True),
    ], BG_WC, C_WC, gap=0.40)

    # ---- OmnibusNoiseFilter group ---------------------------------------
    gx0, gy0, gw, gh = 4.70, 4.55, 9.00, 3.30
    c.group_bg(gx0, gy0, gw, gh, C_NF, BG_NF,
               label="OmnibusNoiseFilter   (one per CRP anode)",
               ly=gy0 + gh - 0.30, fs=15.5)

    yb = 5.95
    c.algobox(6.05, yb, 2.50, 2.62,
              "① OneChannelNoise", [
                  "per channel:",
                  "• FFT → zero DC bin → IFFT",
                  "• dynamic baseline:",
                  "   clip ±6σ, subtract",
                  "   binned median",
                  "• RMS ∉ [1,60] → \"noisy\"",
              ], "white", C_NF, title_fs=14, bullet_fs=10.6, dy=0.335)
    c.algobox(8.75, yb, 2.50, 2.62,
              "② CoherentNoiseSub", [
                  "per conduit group",
                  "(16–48 ch):",
                  "• per-tick group median",
                  "• signal protection —",
                  "   U/V via FR⊛E deconv",
                  "• subtract, per-ch coef",
              ], "white", C_NF, title_fs=13.5, bullet_fs=10.6, dy=0.335)
    c.algobox(11.80, yb, 3.35, 2.62,
              "③ ShieldCouplingSub", [
                  "top CRPs only, U plane:",
                  "• shield-strip pickup groups keyed",
                  "   by strip length (PDVD_strip_length)",
                  "• per-strip common-mode estimate",
                  "• subtract from the coupled U chans",
                  "   (bottom CRPs: stage skipped)",
              ], "white", C_NF, title_fs=14, bullet_fs=10.3, dy=0.365)

    # ---- output ---------------------------------------------------------
    c.stack_box(14.85, ys, 1.90, 1.7, [
        ("cleaned ADC", 14, True, C_OUT, False),
        ("frame (per CRP)", 12, False, INK, False),
        ("WCT tag: raw{N}", 11, False, "#5f8f72", True),
    ], BG_OUT, C_OUT, gap=0.44)

    # ---- spine arrows ----------------------------------------------------
    c.arrow((2.45, ys), (2.60, ys), C_IN)            # input -> resampler
    c.arrow((4.50, ys), (4.80, yb + 0.4), C_WC)      # resampler -> group/box1
    c.arrow((7.30, yb), (7.50, yb), C_NF)            # box1 -> box2
    c.arrow((10.00, yb), (10.125, yb), C_NF)         # box2 -> box3
    c.arrow((13.475, yb + 0.4), (13.90, ys), C_OUT)  # box3 -> output

    # ---- OFF-in-this-build panel (bottom-left) --------------------------
    ox0, oy0, ow, oh = 0.45, 1.05, 3.55, 3.00
    c.group_bg(ox0, oy0, ow, oh, C_OFF, BG_OFF,
               label="available in code — OFF here", ly=oy0 + oh - 0.28,
               fs=12.5)
    off_lines = [
        "PDVD front-end is DC-coupled, so",
        "these MicroBooNE-era paths are",
        "left at their C++ defaults:",
        "",
        "• adaptive baseline / IS_RC partial",
        "• RC+RC undershoot deconvolution",
        "• per-channel freq-notch masks",
        "",
        "no PDHD-style FEMB neg-pulse stage",
    ]
    yy = oy0 + oh - 0.70
    for ln in off_lines:
        c.ov.text(ox0 + 0.22, yy, ln, ha="left", va="center", fontsize=11,
                  color="#6b7178", zorder=2)
        yy -= 0.262

    # ---- insets (bottom) -------------------------------------------------
    ybi = 1.05
    c.place_image(os.path.join(SRC, "nf_coherent_2d.png"), 7.30, 5.20, ybi,
                  "coherent-noise subtraction (data)", (8.75, 4.60), C_NF)
    c.place_image(os.path.join(SRC, "nf_noise_rms.png"), 13.10, 5.30, ybi,
                  "noise RMS: pre-NF → post-NF (data)", (14.85, 5.55), C_OUT)

    c.footer("Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/protodunevd/"
             "{nf,chndb-base,chndb-resp-bot,chndb-resp-top}.jsonnet + "
             "wct-nf-sp(-dnnroi).jsonnet  ·  impl sigproc/src/ProtoduneVD.cxx"
             "  ·  insets: ProtoDUNE-VD data run 039252 evt 298567, CRP0 V")
    c.save(os.path.join(HERE, "pdvd_nf_chain"))


if __name__ == "__main__":
    main()

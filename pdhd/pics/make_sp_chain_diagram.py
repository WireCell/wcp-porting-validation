#!/usr/bin/env python3
"""ProtoDUNE-HD Signal-Processing algorithm diagram (wide 16:9, for slides).

Draws the live OmnibusSigProc -> L1SPFilterPD cascade as the toolkit runs it on
PDHD (sp.jsonnet / sp-filters.jsonnet / wct-nf-sp.jsonnet):

  raw{N} NF frame
   -> OmnibusSigProc:
        1 2D deconvolution   (remove field+elec response, apply SP filters)
        2 ROI finding        (tight/loose ROI + refinement cascade)
        3 charge extraction  (gauss{N}, wiener{N})
   -> L1SPFilterPD           (per-ROI LASSO unipolar-induction correction)
   -> SP frame (gauss{N}, wiener{N}) -> imaging

Four real insets sit below: the 2D field-response kernel, the analytic SP
frequency filters, a data raw->deconvolved single-channel waveform, and the
L1SP bipolar/unipolar response bases.

Output: pdhd/pics/pdhd_sp_chain.{png,pdf}
"""
import os
from diagram_helpers import Canvas

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "nfsp_src")

C_SP = "#1f4e9b"      # signal-processing algorithms
BG_SP = "#e8f0fb"
C_L1 = "#7a3fb0"      # L1SP correction
BG_L1 = "#f1eafa"
C_IN = "#4a4a4a"      # input
BG_IN = "#eef1f5"
C_OUT = "#2e7d4f"     # charge output
BG_OUT = "#eaf5ee"


def main():
    c = Canvas()
    c.title("ProtoDUNE-HD Wire-Cell Signal Processing  ·  "
            "OmnibusSigProc → L1SPFilterPD (per APA)",
            r"$\mathrm{charge}(t)=\mathrm{IFFT}\{\,\mathrm{FFT}[\mathrm{raw}]"
            r"\cdot G_{\rm filter}(\omega)\,/\,[\mathrm{FR}\ast E](\omega)\,\}"
            r"\ \rightarrow\ $ROI gating $\rightarrow$ L1SP unipolar fit")

    ys = 6.55
    # ---- input ----------------------------------------------------------
    c.box(1.05, ys, 1.6, 1.15, "raw{N}\nNF frame", BG_IN, C_IN, fs=11.5)

    # ---- OmnibusSigProc group -------------------------------------------
    gx0, gy0, gw, gh = 2.05, 5.15, 8.55, 2.90
    c.group_bg(gx0, gy0, gw, gh, C_SP, BG_SP,
               label="OmnibusSigProc", ly=gy0 + gh - 0.24, fs=13)

    yb = 6.35
    c.algobox(3.45, yb, 2.55, 2.45,
              "① 2D deconvolution", [
                  "• FFT(raw)  ÷  [FR ⊛ ColdElec](ω)",
                  "   remove detector response",
                  "• × Wiener / Gaus filter (ω)",
                  "• × wire-domain filter",
                  "  → deconvolved charge",
              ], "white", C_SP, title_fs=11.5, bullet_fs=9.0)
    c.algobox(6.20, yb, 2.65, 2.45,
              "② ROI finding", [
                  "• tight ROI  (col 5σ / ind 3σ)",
                  "• loose ROI  (rebin 6; LF filters)",
                  "• refine: CleanupROI,",
                  "   BreakROI×2, Shrink/Extend ROI",
                  "• fake-signal rejection",
              ], "white", C_SP, title_fs=11.5, bullet_fs=9.0)
    c.algobox(9.00, yb, 2.40, 2.45,
              "③ charge extract", [
                  "two filtered",
                  "estimates:",
                  "• gauss{N}",
                  "   (Gaussian)",
                  "• wiener{N}",
                  "   (Wiener SNR)",
              ], "white", C_SP, title_fs=11.5, bullet_fs=9.3)

    # ---- L1SPFilterPD (downstream) --------------------------------------
    c.algobox(12.55, yb, 3.05, 2.45,
              "L1SPFilterPD", [
                  "induction U/V, per ROI:",
                  "• LASSO fit — bipolar +",
                  "   unipolar response bases",
                  "• cross-channel adjacency (≤3 hops)",
                  "• 5-arm trigger gate",
                  "  → replace gauss/wiener",
              ], BG_L1, C_L1, title_fs=12, bullet_fs=9.0)

    # ---- output ---------------------------------------------------------
    c.box(15.15, ys, 1.55, 1.35, "SP frame\ngauss/wiener\n→ imaging", BG_OUT,
          C_OUT, fs=10.5, tc=C_OUT)

    # ---- spine arrows ----------------------------------------------------
    c.arrow((1.85, ys), (2.175, ys), C_IN)          # input -> group
    c.arrow((4.725, yb), (4.875, yb), C_SP)         # 1 -> 2
    c.arrow((7.525, yb), (7.80, yb), C_SP)          # 2 -> 3
    c.arrow((10.20, yb), (11.025, yb), C_L1)        # 3 -> L1SP
    c.arrow((14.075, yb), (14.375, ys), C_OUT)      # L1SP -> output

    # ---- OFF / APA0 note -------------------------------------------------
    c.ov.text(6.30, 4.74,
              "OFF in this build: multi-plane protection (MP3/MP2).     "
              "APA0 specials: own field-response file · U↔V plane2layer swap · "
              "L1SP on U only.", ha="center", va="center", fontsize=10,
              color="#8a8a8a", style="italic")

    # ---- insets (bottom row) --------------------------------------------
    ybi = 1.25
    c.place_image(os.path.join(SRC, "sp_decon_kernel_2d.png"), 2.55, 3.55, ybi,
                  "field response kernel (V)", (3.45, 5.15), C_SP)
    c.place_image(os.path.join(SRC, "sp_filters.png"), 6.30, 3.55, ybi,
                  "SP deconvolution filters", (3.75, 5.15), C_SP)
    c.place_image(os.path.join(SRC, "sp_waveform.png"), 10.05, 3.55, ybi,
                  "raw ADC → deconvolved charge (data)", (9.00, 5.15), C_OUT)
    c.place_image(os.path.join(SRC, "l1sp_kernel.png"), 13.75, 4.35, ybi,
                  "L1SP response bases (V bipolar / W unipolar)",
                  (12.55, 5.15), C_L1)

    c.footer("Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/pdhd/{sp,sp-filters}."
             "jsonnet + wct-nf-sp.jsonnet  ·  impl sigproc/src/{OmnibusSigProc,"
             "L1SPFilterPD}.cxx  ·  data insets: ProtoDUNE-HD run 027409 evt 0, APA1 V")
    c.save(os.path.join(HERE, "pdhd_sp_chain"))


if __name__ == "__main__":
    main()

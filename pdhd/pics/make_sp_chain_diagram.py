#!/usr/bin/env python3
"""ProtoDUNE-HD Signal-Processing algorithm diagram — DNN-ROI chain (wide 16:9).

Draws the live production DNN-ROI signal-processing cascade
(wct-nf-sp-dnnroi.jsonnet / dnnroi_pp.jsonnet):

  raw{N} NF frame  (WCT tag raw{N} = the NF-cleaned ADC, i.e. "cooked")
   -> ① 2D deconvolution  (OmnibusSigProc: remove field+elec response, SP
        filters; also emits the ROI feature frames loose/tight LF, MP2/MP3)
   -> ② DNN-ROI           (DNNROIFinding + TorchService: a 6-channel CNN ROI
        finder; per-pixel score -> binary mask x decon_charge -> dnnsp)
   -> L1SPFilterPD        (per-ROI LASSO unipolar-induction correction, after DNN)
   -> SP frame (gauss{N}, wiener{N}) -> imaging

The DNN consumes the traditional SP ROI products as its input channels — it does
not delete the ROI finder, it makes the final ROI decision from those features.
MP2/MP3 protection is therefore ON in this chain.

Insets: 2D field-response kernel and a data raw->deconvolved waveform (both under
deconvolution), the real DNN-ROI event display (deconvolved charge -> CNN score),
and the L1SP bipolar/unipolar bases.

Output: pdhd/pics/pdhd_sp_chain.{png,pdf}
"""
import os
from diagram_helpers import Canvas

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "nfsp_src")

C_SP = "#1f4e9b"      # deconvolution
BG_SP = "#e8f0fb"
C_DNN = "#0f8a8a"     # DNN-ROI
BG_DNN = "#e2f3f3"
C_L1 = "#7a3fb0"      # L1SP correction
BG_L1 = "#f1eafa"
C_IN = "#3a3f47"      # input
BG_IN = "#eef1f5"
C_OUT = "#2e7d4f"     # charge output
BG_OUT = "#eaf5ee"
INK = "#20242b"


def main():
    c = Canvas()
    c.title("ProtoDUNE-HD Wire-Cell Signal Processing  ·  "
            "OmnibusSigProc → DNN-ROI → L1SP  (per APA)",
            r"$\mathrm{deconvolve}\ \frac{\mathrm{FFT}[\mathrm{raw}]\cdot "
            r"G_{\rm filter}(\omega)}{[\mathrm{FR}\ast E](\omega)}\ \rightarrow\ $"
            r"neural-network ROI $\rightarrow$ L1SP unipolar fit",
            mfs=21.5)

    ys = 6.20
    yb = 6.20
    # ---- input (NF output = SP input; same WCT tag as the NF slide) ------
    c.stack_box(1.30, ys, 2.00, 1.75, [
        ("NF-cleaned ADC", 14, True, C_IN, False),
        ("(input to SP)", 11.5, False, INK, False),
        ("WCT tag: raw{N}", 11, False, "#6b7178", True),
    ], BG_IN, C_IN, gap=0.42)

    # ---- ① 2D deconvolution --------------------------------------------
    c.algobox(4.05, yb, 3.45, 3.00,
              "① 2D deconvolution", [
                  "OmnibusSigProc:",
                  "• FFT(raw)  ÷  [FR ⊛ ColdElec](ω)",
                  "• × Wiener / Gaus & wire filter",
                  "• IFFT → decon charge + gauss",
                  "• + ROI feature frames",
                  "   (loose / tight LF, MP2 / MP3)",
              ], "white", C_SP, title_fs=14.5, bullet_fs=11, dy=0.42)

    # ---- ② DNN-ROI ------------------------------------------------------
    c.algobox(8.60, yb, 4.40, 3.00,
              "② DNN-ROI", [
                  "DNNROIFinding + TorchService:",
                  "• inputs: 6 SP channels (loose / tight",
                  "   LF, MP2 / MP3 ROI, decon_charge, gauss)",
                  "• CNN ROI finder — TorchScript .ts model",
                  "• per-pixel score → binary mask",
                  "• mask × decon_charge → dnnsp",
                  "• U+V via model · W = gauss · APA0: U only",
              ], BG_DNN, C_DNN, title_fs=15, bullet_fs=10.6, dy=0.375)

    # ---- L1SPFilterPD ---------------------------------------------------
    c.algobox(12.85, yb, 3.05, 3.00,
              "L1SPFilterPD", [
                  "after DNN-ROI, induction U/V:",
                  "• per-ROI LASSO fit —",
                  "   bipolar + unipolar bases",
                  "• adjacency expansion (≤3 hops)",
                  "• refine → gauss / wiener",
              ], BG_L1, C_L1, title_fs=14, bullet_fs=10.8, dy=0.46)

    # ---- output ---------------------------------------------------------
    c.stack_box(15.25, ys, 1.60, 1.65, [
        ("SP frame", 13.5, True, C_OUT, False),
        ("gauss / wiener", 11.5, False, INK, False),
        ("→ imaging", 11, False, C_OUT, True),
    ], BG_OUT, C_OUT, gap=0.44)

    # ---- spine arrows ----------------------------------------------------
    c.arrow((2.30, ys), (2.325, ys), C_IN)          # input -> ①
    c.arrow((5.775, yb), (6.375, yb), C_DNN)        # ① -> ②
    c.arrow((10.80, yb), (11.30, yb), C_L1)         # ② -> L1SP
    c.arrow((14.375, yb), (14.45, ys), C_OUT)       # L1SP -> output

    # ---- per-APA / plane note (MP is ON in the DNN chain) ----------------
    c.ov.text(8.0, 4.55,
              "W plane: standard SP gauss (no DNN).      APA0 specials: own "
              "field-response file · U↔V plane2layer swap · DNN-ROI on U only "
              "(V anomalous).", ha="center", va="center", fontsize=11.5,
              color="#8a8a8a", style="italic")

    # ---- insets (bottom row) --------------------------------------------
    # The regenerable insets (decon kernel, waveform) stay legible at smaller
    # width, so cede width to the two wide rasters — dnn_roi is a baked crop
    # whose internal text only grows with placement size, and l1sp is now a
    # fresh large-font panel.
    ybi = 1.35
    c.place_image(os.path.join(SRC, "sp_decon_kernel_2d.png"), 1.90, 2.95, ybi,
                  "field-response kernel (V)", (4.05, 4.86), C_SP, cap_fs=12)
    c.place_image(os.path.join(SRC, "sp_waveform.png"), 4.95, 2.95, ybi,
                  "raw ADC → deconvolved charge", (5.10, 4.86), C_OUT,
                  cap_fs=12)
    # crop the clipped partial-title strips (top/bottom) off the dnn_roi raster
    c.place_image(os.path.join(SRC, "dnn_roi.png"), 9.20, 5.20, ybi,
                  "DNN-ROI: charge → CNN score", (8.60, 4.86), C_DNN,
                  box=(0.0, 0.035, 1.0, 0.94), cap_fs=12)
    c.place_image(os.path.join(SRC, "l1sp_kernel.png"), 13.90, 4.10, ybi,
                  "L1SP response bases", (12.85, 4.86), C_L1, cap_fs=12)

    c.footer("Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/pdhd/{sp,sp-filters,"
             "dnnroi_pp}.jsonnet + wct-nf-sp-dnnroi.jsonnet  ·  model "
             "wire-cell-data/dnnroi/pdhd/*.ts  ·  data insets: ProtoDUNE-HD run "
             "027409 evt 0 (APA1 V; DNN-ROI panel APA0 U)")
    c.save(os.path.join(HERE, "pdhd_sp_chain"))


if __name__ == "__main__":
    main()

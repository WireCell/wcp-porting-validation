#!/usr/bin/env python3
"""ProtoDUNE-HD Signal-Processing algorithm diagram — DNN-ROI chain (wide 16:9).

Draws the live production DNN-ROI signal-processing cascade
(wct-nf-sp-dnnroi.jsonnet / dnnroi_pp.jsonnet):

  raw{N} NF frame
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
C_IN = "#4a4a4a"      # input
BG_IN = "#eef1f5"
C_OUT = "#2e7d4f"     # charge output
BG_OUT = "#eaf5ee"


def main():
    c = Canvas()
    c.title("ProtoDUNE-HD Wire-Cell Signal Processing  ·  "
            "OmnibusSigProc → DNN-ROI → L1SPFilterPD (per APA)",
            r"$\mathrm{deconvolve}\ \frac{\mathrm{FFT}[\mathrm{raw}]\cdot "
            r"G_{\rm filter}(\omega)}{[\mathrm{FR}\ast E](\omega)}\ \rightarrow\ $"
            r"neural-network ROI $\rightarrow$ L1SP unipolar fit")

    ys = 6.30
    # ---- input ----------------------------------------------------------
    c.box(1.05, ys, 1.65, 1.2, "raw{N}\nNF frame", BG_IN, C_IN, fs=11.5)

    yb = 6.30
    # ---- ① 2D deconvolution --------------------------------------------
    c.algobox(4.00, yb, 3.40, 2.60,
              "① 2D deconvolution", [
                  "OmnibusSigProc:",
                  "• FFT(raw)  ÷  [FR ⊛ ColdElec](ω)",
                  "• × Wiener / Gaus & wire filter",
                  "• IFFT → decon charge + gauss",
                  "• + ROI feature frames",
                  "   (loose / tight LF, MP2 / MP3)",
              ], "white", C_SP, title_fs=12.5, bullet_fs=9.3)

    # ---- ② DNN-ROI ------------------------------------------------------
    c.algobox(8.55, yb, 4.35, 2.60,
              "② DNN-ROI", [
                  "DNNROIFinding + TorchService:",
                  "• inputs: 6 SP channels (loose / tight",
                  "   LF, MP2 / MP3 ROI, decon_charge, gauss)",
                  "• CNN ROI finder — TorchScript .ts model",
                  "• per-pixel score → binary mask",
                  "• mask × decon_charge → dnnsp",
                  "• U+V via model · W = gauss · APA0: U only",
              ], BG_DNN, C_DNN, title_fs=13, bullet_fs=9.0, dy=0.285)

    # ---- L1SPFilterPD ---------------------------------------------------
    c.algobox(12.85, yb, 3.05, 2.60,
              "L1SPFilterPD", [
                  "after DNN-ROI, induction U/V:",
                  "• per-ROI LASSO fit —",
                  "   bipolar + unipolar bases",
                  "• adjacency expansion (≤3 hops)",
                  "• refine → gauss / wiener",
              ], BG_L1, C_L1, title_fs=12, bullet_fs=8.9)

    # ---- output ---------------------------------------------------------
    c.box(15.15, ys, 1.55, 1.35, "SP frame\ngauss/wiener\n→ imaging", BG_OUT,
          C_OUT, fs=10.5, tc=C_OUT)

    # ---- spine arrows ----------------------------------------------------
    c.arrow((1.875, ys), (2.30, ys), C_IN)          # input -> ①
    c.arrow((5.70, yb), (6.375, yb), C_DNN)         # ① -> ②
    c.arrow((10.725, yb), (11.325, yb), C_L1)       # ② -> L1SP
    c.arrow((14.375, yb), (14.375, ys), C_OUT)      # L1SP -> output

    # ---- per-APA / plane note (no MP-off: MP is ON in the DNN chain) -----
    c.ov.text(8.0, 4.74,
              "W plane: standard SP gauss (no DNN).      APA0 specials: own "
              "field-response file · U↔V plane2layer swap · DNN-ROI on U only "
              "(V anomalous).", ha="center", va="center", fontsize=10,
              color="#8a8a8a", style="italic")

    # ---- insets (bottom row) --------------------------------------------
    ybi = 1.20
    c.place_image(os.path.join(SRC, "sp_decon_kernel_2d.png"), 2.30, 3.40, ybi,
                  "field-response kernel (V)", (3.40, 5.00), C_SP)
    c.place_image(os.path.join(SRC, "sp_waveform.png"), 6.00, 3.40, ybi,
                  "raw ADC → deconvolved charge (data)", (4.60, 5.00), C_OUT)
    c.place_image(os.path.join(SRC, "dnn_roi.png"), 9.95, 4.25, ybi,
                  "DNN-ROI: deconvolved charge → CNN score (data)",
                  (8.55, 5.00), C_DNN)
    c.place_image(os.path.join(SRC, "l1sp_kernel.png"), 14.00, 3.45, ybi,
                  "L1SP response bases (V bipolar / W unipolar)",
                  (12.85, 5.00), C_L1)

    c.footer("Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/pdhd/{sp,sp-filters,"
             "dnnroi_pp}.jsonnet + wct-nf-sp-dnnroi.jsonnet  ·  model "
             "wire-cell-data/dnnroi/pdhd/*.ts  ·  data insets: ProtoDUNE-HD run "
             "027409 evt 0 (APA1 V; DNN-ROI panel APA0 U)")
    c.save(os.path.join(HERE, "pdhd_sp_chain"))


if __name__ == "__main__":
    main()

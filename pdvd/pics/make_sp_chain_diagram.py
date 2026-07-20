#!/usr/bin/env python3
"""ProtoDUNE-VD Signal-Processing algorithm diagram — DNN-ROI chain (wide 16:9).

Counterpart of pdhd/pics/make_sp_chain_diagram.py (PD-HD original) with the
ProtoDUNE-VD cascade.  Draws the live production DNN-ROI signal-processing
chain (wct-nf-sp-dnnroi.jsonnet / protodunevd/{sp,dnnroi_pp}.jsonnet):

  raw{N} NF frame  (WCT tag raw{N} = the NF-cleaned ADC, i.e. "cooked")
   -> ① 2D deconvolution  (OmnibusSigProc: remove field+elec response, SP
        filters, traditional tight/loose ROI; also emits the ROI feature
        frames loose/tight LF, MP2/MP3)
   -> ② DNN-ROI           (DNNROIFinding + TorchService: a 6-channel CNN ROI
        finder; per-pixel score -> binary mask x decon_charge -> dnnsp)
   -> retag dnnsp{N} -> gauss{N} -> imaging

PD-VD specifics vs PD-HD: per-side electronics (bottom analytic ColdElec 7.8
mV/fC; top JSON response x1.36 postgain, 2.0 V fullscale), roi_mad_rms ON,
BreakROI disabled on the collection plane, and L1SPFilterPD present in the
builder but OFF in this chain (drawn greyed).

Insets: 2D field-response kernel, a data raw->deconvolved waveform, the SP
frequency filters, and a real traditional-ROI vs DNN-ROI A/B comparison.

Output: pdvd/pics/pdvd_sp_chain.{png,pdf}
"""
import os
from diagram_helpers_v2 import Canvas

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "nfsp_src")

C_SP = "#1f4e9b"      # deconvolution
BG_SP = "#e8f0fb"
C_DNN = "#0f8a8a"     # DNN-ROI
BG_DNN = "#e2f3f3"
C_OFF = "#9aa0a8"     # dormant / OFF (L1SP here)
BG_OFF = "#f1f1f3"
C_IN = "#3a3f47"      # input
BG_IN = "#eef1f5"
C_OUT = "#2e7d4f"     # charge output
BG_OUT = "#eaf5ee"
INK = "#20242b"


def main():
    c = Canvas()
    c.title("ProtoDUNE-VD Wire-Cell Signal Processing  ·  "
            "OmnibusSigProc → DNN-ROI  (per CRP)",
            r"$\mathrm{deconvolve}\ \frac{\mathrm{FFT}[\mathrm{raw}]\cdot "
            r"G_{\rm filter}(\omega)}{[\mathrm{FR}\ast E](\omega)}\ "
            r"\rightarrow\ $"
            r"traditional ROI features $\rightarrow$ neural-network ROI "
            r"decision",
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
                  "• FFT(raw)  ÷  [FR ⊛ E](ω)",
                  "   E per side: bottom analytic",
                  "   · top JSON ×1.36",
                  "• × Wiener / Gaus & wire filter",
                  "• IFFT → decon charge + gauss",
                  "• + ROI features (loose / tight",
                  "   LF, MP2 / MP3) · roi_mad_rms",
              ], "white", C_SP, title_fs=14.5, bullet_fs=10.2, dy=0.315)

    # ---- ② DNN-ROI ------------------------------------------------------
    c.algobox(8.60, yb, 4.40, 3.00,
              "② DNN-ROI", [
                  "DNNROIFinding + TorchService:",
                  "• inputs: 6 SP channels (loose / tight",
                  "   LF, MP2 / MP3 ROI, decon_charge, gauss)",
                  "• CNN ROI finder — TorchScript .ts model",
                  "• per-pixel score → binary mask",
                  "• mask × decon_charge → dnnsp",
                  "• U+V via model · W = gauss passthrough",
              ], BG_DNN, C_DNN, title_fs=15, bullet_fs=10.6, dy=0.375)

    # ---- L1SPFilterPD — present but OFF in this chain -------------------
    # shorter + lower than the live boxes so the ② -> output arrow passes
    # visibly OVER it (the stage is not in the production dataflow)
    c.algobox(12.85, 5.85, 3.05, 2.05,
              "L1SPFilterPD — OFF", [
                  "available in the builder",
                  "(PDVD-tuned kernels, U+V)",
                  "but disabled in the DNN",
                  "production chain",
              ], BG_OFF, C_OFF, tc=C_OFF, title_fs=13.5, bullet_fs=10.8,
              dy=0.36)

    # ---- output ---------------------------------------------------------
    c.stack_box(15.25, ys, 1.60, 1.65, [
        ("SP frame", 13.5, True, C_OUT, False),
        ("dnnsp → gauss", 11.5, False, INK, False),
        ("→ imaging", 11, False, C_OUT, True),
    ], BG_OUT, C_OUT, gap=0.44)

    # ---- spine arrows ----------------------------------------------------
    c.arrow((2.30, ys), (2.325, ys), C_IN)          # input -> ①
    c.arrow((5.775, yb), (6.375, yb), C_DNN)        # ① -> ②
    # ② -> output, passing over the (inert) L1SP box
    c.arrow((10.80, 7.45), (14.45, 6.75), C_OUT)

    # ---- per-CRP / plane note (per-side split) ---------------------------
    c.ov.text(8.0, 4.55,
              "per-side: bottom (idents 0–3) ColdElec 7.8 mV/fC · top (4–7) "
              "JSON elec ×1.36, 2.0 V fullscale.      W plane: BreakROI off "
              "· no DNN (gauss passthrough).", ha="center", va="center",
              fontsize=11.5, color="#8a8a8a", style="italic")

    # ---- insets (bottom row) --------------------------------------------
    ybi = 1.35
    c.place_image(os.path.join(SRC, "sp_decon_kernel_2d.png"), 1.90, 2.95,
                  ybi, "field-response kernel (V)", (3.60, 4.72), C_SP,
                  cap_fs=12)
    c.place_image(os.path.join(SRC, "sp_filters.png"), 4.95, 2.95, ybi,
                  "SP deconvolution filters", (4.60, 4.72), C_SP, cap_fs=12)
    c.place_image(os.path.join(SRC, "dnn_roi.png"), 9.20, 5.20, ybi,
                  "traditional ROI vs DNN-ROI (data A/B)", (8.60, 4.72),
                  C_DNN, cap_fs=12)
    c.place_image(os.path.join(SRC, "sp_waveform.png"), 13.90, 3.30, ybi,
                  "NF ADC → deconvolved charge", (15.00, 5.42), C_OUT,
                  cap_fs=12)

    c.footer("Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/protodunevd/"
             "{sp,sp-filters,dnnroi_pp}.jsonnet + wct-nf-sp-dnnroi.jsonnet  ·  "
             "model wire-cell-data/dnnroi/pdvd/pipe_distill_nestedunet_6ch.ts"
             "  ·  data insets: ProtoDUNE-VD runs 039252/039253, CRP0 V")
    c.save(os.path.join(HERE, "pdvd_sp_chain"))


if __name__ == "__main__":
    main()

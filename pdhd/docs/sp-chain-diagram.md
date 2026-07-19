# PD-HD Signal-Processing algorithm diagram (DNN-ROI chain)

A wide (16:9) presentation diagram of the ProtoDUNE-HD Wire-Cell **signal
processing** stage as run in production — `OmnibusSigProc` → **DNN-ROI**
(`DNNROIFinding`) → `L1SPFilterPD` — with four physics insets.  Companion to the
reference doc [sp.md](sp.md).  See also [nf-chain-diagram.md](nf-chain-diagram.md)
for the noise-filtering counterpart and [nf_sp_workflow.md](nf_sp_workflow.md)
for the end-to-end run.

![PD-HD SP chain](../pics/pdhd_sp_chain.png)

Deliverable: `pdhd/pics/pdhd_sp_chain.png` (3840×2160) and `.pdf`.

## Repro block

Run in the toolkit-dev direnv python env (matplotlib, numpy, PIL; `WIRECELL_PATH`
set) from `pdhd/pics/`:

```bash
python3 make_nfsp_insets.py        # generates the NF+SP data insets into nfsp_src/
# the DNN-ROI inset is a 2-panel crop (decon_charge -> dnn_score) of
#   DNN_ROI_SP/simulation/../pdhd_27409_evt0_debug/evdisp_region.png -> nfsp_src/dnn_roi.png
# the L1SP inset is a crop of nf_plot/track_response_l1sp_pdhd_V.png -> nfsp_src/l1sp_kernel.png
python3 make_sp_chain_diagram.py   # -> pdhd_sp_chain.png / .pdf
```

`nfsp_src/` holds every committed inset PNG so the master build is self-contained.

## The cascade (what the boxes are)

Traced from `cfg/pgrapher/experiment/pdhd/{sp,sp-filters,dnnroi_pp}.jsonnet` and
`pdhd/wct-nf-sp-dnnroi.jsonnet` (production defaults `use_dnnroi=true`,
`use_l1sp_dnn=true`).

| stage | node | what it does |
|---|---|---|
| ① 2D deconvolution | `OmnibusSigProc` | `FFT(raw) ÷ [FR ⊛ ColdElec](ω)` to remove the detector response, × the SP frequency filter (Wiener/Gaussian) and a wire-domain filter, then IFFT → deconvolved charge + `gauss`. In the DNN chain it also runs with `use_roi_debug_mode` + `use_multi_plane_protection` **ON**, so it additionally emits the ROI feature frames (`loose_lf`, `tight_lf`, `mp2_roi`, `mp3_roi`, `decon_charge`). |
| ② DNN-ROI | `DNNROIFinding` + `TorchService` | a **6-channel CNN ROI finder** (TorchScript `.ts` model in `wire-cell-data/dnnroi/pdhd/`) whose input channels are the six SP products above; it outputs a per-pixel ROI score, thresholds it to a binary mask, and applies `mask × decon_charge → dnnsp`. U and V are run through the model (APA1–3); W passes through as standard SP `gauss`; APA0 runs the model on U only (V anomalous). |
| L1SP | `L1SPFilterPD` | runs **after** DNN-ROI on induction U/V: per-ROI LASSO fit against bipolar + unipolar response bases, cross-channel adjacency expansion (≤3 hops), refining the DNN-ROI charge. |
| out | `gauss{N}`, `wiener{N}` | deconvolved-charge frame (a `Retagger` relabels `dnnsp{N}` → `gauss{N}`/`wiener{N}`) → imaging. |

**The DNN does not delete the traditional ROI finder** — it *consumes* the SP
ROI products (loose/tight LF, MP2/MP3 ROI, decon_charge, gauss) as its input
feature channels and makes the final ROI decision.  This is why multi-plane
protection is ON in this chain.  To recover the traditional (non-DNN) ROI path,
run with `--tla-code use_dnnroi=false` (falls back to the `OmnibusSigProc`
tight/loose ROI + BreakROI/Shrink/Extend refinement described in [sp.md](sp.md)).

### Per-APA / plane notes

- **W plane**: always standard SP `gauss` (no DNN).
- **APA0**: own field-response file (`np04hd-garfield` fit vs the
  `dune-garfield-1d565` template for APA1–3), U↔V `plane2layer=[0,2,1]` swap, and
  DNN-ROI on the U plane only.

## Physics insets

| inset | source | shows |
|---|---|---|
| field-response kernel (V) | `make_nfsp_insets.py:make_decon_kernel_2d` (`dune-garfield-1d565.json.bz2`) | the 2D time-domain field response deconvolved in ① — the induction kernel SP inverts. |
| raw ADC → deconvolved charge (data) | `make_nfsp_insets.py:make_sp_waveform` (NF-raw vs SP-gauss frames) | one V-plane channel (ch 3757): bipolar induction ADC → the unipolar deconvolved charge — the same channel threaded through the NF diagram. |
| DNN-ROI: deconvolved charge → CNN score (data) | 2-panel crop of `pdhd_27409_evt0_debug/evdisp_region.png` | the noisy deconvolved charge (top, DNN input) and the clean CNN sigmoid ROI score (bottom) that isolates the track — a real DNN-ROI inference. |
| L1SP response bases | crop of `nf_plot/track_response_l1sp_pdhd_V.png` | the bipolar (V) and unipolar (collection-on-induction) response bases the L1SP LASSO fits per ROI. |

The DNN-ROI inset is APA0 U (that is the event region with a saved DNN debug
frame); the other three insets are APA1 V.  All are ProtoDUNE-HD **data**, run
027409 evt 0 — the same event.

## Verification

- The DNN-ROI inset shows the network cleanly recovering the track from a
  deconvolved-charge image full of coherent-baseline structure — the score map
  isolates the diagonal track and rejects the horizontal artifacts.
- `make_sp_chain_diagram.py` runs clean → 3840×2160 PNG + PDF; three algorithm
  boxes, arrows, leader lines and four insets legible at slide scale, 16:9, no
  text overflow.
- No toolkit C++/cfg touched — docs/figure deliverable only; nothing in the
  reconstruction path changes, so no build or A/B gate is required.

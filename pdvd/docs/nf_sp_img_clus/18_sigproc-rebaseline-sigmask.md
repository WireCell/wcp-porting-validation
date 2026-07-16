# `rebase_waveform` (sigmask "rebaseline") across the PDVD/PDHD SP chains

Diagnostic note (2026-06-08): what the `OmnibusSigProc` (OSP) baseline-rebase commits mean for
our PDVD/PDHD data and `DNN_ROI_SP` simulation chains.

## The commits

- **`70bd3015`** *sigproc/OSP: signal-safe rebase_waveform anchor options* — adds
  `rebase_method` = `mean|median|sigmask` (+ `rebase_nsigma`); default still `mean`.
- **`97f9d233`** *sigproc/OSP: drop the biased plain-mean rebase anchors* — **removes `mean`**,
  makes **`sigmask` the default**, and makes any config passing `rebase_method:"mean"` throw a
  `ValueError`.

`rebase_waveform` (added 2025-09-04 in `0c1d9f72` "…rebaseline…") subtracts a per-channel linear
baseline tilt fit from the front/back `rebase_nbins` (=200) tick windows. The old plain-mean
anchor is biased by any signal pulse inside a window, tilting the whole channel and corrupting
downstream ROI thresholds. `sigmask` masks `|x − median| > nσ` outliers before averaging (σ from
16/50/84 percentiles), widening the window inward when signal leaves too few clean samples.

## Key facts

1. **Always-on, no toggle.** `rebase_waveform` runs unconditionally on every plane in
   `m_rebase_planes` (default `{0,1,2}` = all three planes) in `OmnibusSigProc::load_data`
   (`sigproc/src/OmnibusSigProc.cxx:445`). The only way to disable it is `rebase_planes: []`.
   So these commits affect **every OSP run** in all our chains; the change is not gated.

2. **No config overrides it.** Nothing in toolkit `cfg/`, the PDVD/PDHD `wct-nf-sp*.jsonnet`, or
   the `DNN_ROI_SP/simulation/stageB*` jsonnet sets any `rebase_*` key — everything uses the code
   default, which after `97f9d233` is **`sigmask`**.

3. **Provenance of existing runs** (pinned by log-line format: post-`70bd3015` lines carry
   `method = N (...)`, the pre-`70bd3015` line is a bare `… m_rebase_nbins = 200`):

   | Chain | Example run | When | Rebase anchor |
   |---|---|---|---|
   | PDHD **data** (run 027409, `wct_nfspdnn`) | `pdhd/work/027409_*/…_a0.log` | 2026-06-08 | **sigmask (new)** |
   | PDVD / PDHD **sim_sp** (evt0–9) | `pd{vd,hd}/sim_sp/evt*/sim_sp.log` | 2026-06-07 | **sigmask (new)** |
   | **DNN_ROI_SP** training sim (stageB) | `DNN_ROI_SP/simulation/stageB*/stageB*.log` | 2026-05-19/20 | **plain-mean (old)** |

## The situation

- The **data** SP (PDVD + PDHD) and recent **sim_sp** runs already use the new **sigmask**
  rebaseline — consistent with current `HEAD`.
- The **DNN-ROI training sample** in `../DNN_ROI_SP` (stageB, May 19–20) was generated with the
  **old plain-mean** rebaseline. The U-Net trains on the 6 OSP output channels
  (`loose_lf, mp2_roi, mp3_roi, tight_lf, decon_charge, gauss`), all of which pass through
  `rebase_waveform`. So a model trained on that sim and applied (via `DNNROIFinding`) to the new
  sigmask data SP has a **baseline / input-distribution mismatch**.
- **No breakage**: none of our configs pass `rebase_method:"mean"`, so the `97f9d233` removal does
  not error anything we run. (FYI: `97f9d233` changed a **global** OSP default for the whole
  branch, not just PD — counter to the usual default-OFF/bit-identical convention.)

## How big is the mismatch? (cheap check first)

mean vs sigmask only diverge where signal sits inside the front/back 200-tick windows; on
clean-baseline channels mean ≈ median ≈ sigmask, so the magnitude is **empirical**. Before any
expensive regen + retrain:

1. Re-run stageB SP on a few `DNN_ROI_SP` sim events with the **current** toolkit build, producing
   new-sigmask 6-channel tensors.
2. Diff the 6 DNN input channels (esp. `decon_charge`) old-vs-new per plane.
3. Negligible delta → no retrain needed; significant delta → regenerate the training sample with
   the new code and retrain + re-export the TorchScript model.

## Files referenced

- `sigproc/src/OmnibusSigProc.cxx` (`configure` ~213-241, `load_data` 445-450, `rebase_waveform`
  1057+), `sigproc/inc/WireCellSigProc/OmnibusSigProc.h` (~260-280)
- Data: `pd{vd,hd}/wct-nf-sp-dnnroi.jsonnet`
- Sim: `DNN_ROI_SP/simulation/stageB_pdvd/wct-sim-nf-sp-dnnroi-pdvd.jsonnet` + `run_stageB_sp_dnnroi.sh`

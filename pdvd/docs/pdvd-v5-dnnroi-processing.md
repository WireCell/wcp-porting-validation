# PDVD data processing — DNN-ROI + L1SP on v5 geometry (2026-06-08)

Record of processing all PDVD beam/cosmic data events through the DNN-ROI
signal-processing chain and 3-D imaging/clustering on the **v5** wire geometry.

## Configuration

- **Geometry: v5** (`protodunevd-wires-larsoft-v5.json.bz2`) — now the default
  for all PDVD processing **and** simulation. Set in
  `cfg/pgrapher/experiment/protodunevd/params.jsonnet` (`files.wires`);
  `simparams.jsonnet` inherits it. (toolkit commit `f3a986dc`.)
  v5 = v4 + the U/V endpoint z-shift calibration (`make_v5_uvwcal.py`); it
  changes 3-D wire crossings, so it affects imaging/clustering, not the
  per-plane 2-D SP waveforms.
- **SP recipe:** `run_nf_sp_dnnroi_evt.sh -D cpu -P fp32 -r data -N hybrid --loose-heur`
  (DNN-ROI fp32 KD model + post-DNN `L1SPFilterPD` in hybrid/loose mode, which
  supplies the `wiener` tag imaging needs). L1SP now defaults on.
- **Imaging/clustering:** `run_img_evt.sh` / `run_clus_evt.sh`, `PDVD_MAX_JOBS=16`.

## Events processed (32)

| run | condition | events |
|---|---|---|
| 039252 | 8 GeV beam | 0–4 (5) |
| 039253 | 8 GeV beam | 0–4 (5) |
| 039324 | 1 GeV beam | 0–10 (11) |
| 040475 | cosmic     | 0–10 (11) |

`run41189` excluded — flat layout with only pre-made non-DNN SP frames (no
`protodune-orig-frames`), so not DNN-able.

## Bee links (per run, v5 reco)

- **039252** (5 evt): https://www.phy.bnl.gov/twister/bee/set/4a837d02-acd0-4550-b35c-b4d46d6bf2ab/event/list/
- **039253** (5 evt): https://www.phy.bnl.gov/twister/bee/set/6bd59c47-2e82-4476-9e58-daf0d6087e66/event/list/
- **039324** (11 evt): https://www.phy.bnl.gov/twister/bee/set/ae0b5c74-9de3-4cb0-a304-1b1f8d154bd8/event/list/
- **040475** (11 evt, cosmic): https://www.phy.bnl.gov/twister/bee/set/bf58a846-67a5-4402-ade7-6c9c8e4cc4bc/event/list/

Built with `./run_bee_combined_evt.sh <run>` (all events of a run → one set;
imaging-group0123/4567 + clustering/dead from `mabc-all-apa.zip`).

## Operational notes

- **DNN frame → imaging bridge.** The DNN runner writes
  `protodune-sp-dnnroi-frames-anode{N}.tar.bz2`; imaging expects
  `protodune-sp-frames-anode{N}`. Symlink the former to the latter in the work
  dir before imaging. Missing symlink ⇒ imaging silently falls back to the
  non-DNN `input_data` SP frames (valid-looking but wrong).
- **Parallel `wire-cell` gets SIGKILLed.** Running ~4 DNN+L1SP SP jobs at once
  on the shared box had them killed in lockstep (not memory — a single event
  peaks <2 GiB; apparently an external watchdog/process cap). 9 events were lost
  this way and recovered by running SP **serially**. Imaging/clustering at 16-way
  parallel were fine (lighter, no L1SP).

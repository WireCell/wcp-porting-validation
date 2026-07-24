# save_stm_fit: persist the STM tagger's track fits (doc 41, implements doc 40 phases 0–3)

One default-OFF knob makes the STM tagger's trajectory + dQ/dx fitting
observable end-to-end.  Everything below is inert (byte-identical outputs)
unless opted in with `run_nusel_evt.sh -stm-fit` / `run_pr_evt.sh -stm-fit`
(env `SBND_STM_FIT=1`).

**Commits** (all pushed 2026-07-24): toolkit `3db191e9` on `apply-pointcloud`
(knob + `SbndMagnifyTrackingVisitor` + jsonnet, 10 files); wcp-porting-validation
`6099ed0` on `main` (runner flags, viewer panel, `stm_ref_dqdx.json`, docs
40–41); Magnify-tracking-SBND `b78b255` on `master` (GUI geometry).

## What the knob does (toolkit side)

`TaggerCheckSTM` config key `save_stm_fit` (C++ default false; jsonnet
key-suppressed; threaded `cfg/pgrapher/common/clus.jsonnet
tagger_check_stm(save_stm_fit=)` → `sbnd/clus.jsonnet clus_pr/pr()` →
`wct-pr-perevt.jsonnet` TLA):

- Records, per main cluster and per fitting **pass** (0 = forward, 1 =
  backward), the FINAL round-2 fit — **including rejected passes** — plus a
  status code:
  `0` accepted (STM), `1` TGM, `2` rejected long-leftover ("Mid Point A"),
  `3` rejected dQ/dx eval, `4` rejected extra-tracks, `5` rejected proton
  end, `6` fitted-no-decision.  Round-1 rough fits and passes aborted with
  ≤3 fit points are not recorded.  FC-contained and already-TGM clusters
  never reach the fitter (no record).
- Persists onto the cluster node (same-job consumers only; the PR job does
  not re-save the pctree tarball):
  - `stm_fit` PC: per point x,y,z, dQ, dx, L, rr (residual range from the
    path end = candidate stopping end), pu,pv,pw,pt, apa, face,
    reduced_chi2, pass, status;
  - `stm_pass` PC: per pass status/kink_num/npoints/exit_L/left_L/
    exit_dqdx/left_dqdx;
  - `stm_eval` PC: one row per `eval_stm_core` call (combo parameters, ks1,
    ks2, ratio1, ratio2, res_length, ave_res_dqdx, verdict) — captured via a
    thin wrapper around the eval; the KS numbers previously existed only as
    trace logs.
- Hands an accumulator `TrackFitting` (all recorded segments + the per-pass
  pred/meas 2D charge, merged last-writer-wins) to the **named grouping slot
  `"stm"`** — new additive API
  `Facade::Grouping::{set,get}_track_fitting(name)`; the unnamed slot the
  neutrino-PR chain uses is untouched.  New additive
  `TrackFitting::merge_fitted_charge_2d()`.
- Emits one INFO line per pass:
  `persist_stm_fit: cluster N stmfit pass=P status=S kink=K exit_L=... left_L=... npts=...`.

## Consumers

1. **Magnify-tracking ROOT dump** — new `SbndMagnifyTrackingVisitor`
   (fork-by-duplication of `UbooneMagnifyTrackingVisitor`, which reads the
   PR graph and stays untouched), pipeline name `stm_magnify` (appended
   automatically by `-stm-fit`), writes `nusel_evt*/tracking-stm.root`:
   - `T_rec_charge`: per fitted point; **block id `ndf` = cluster_id*10 +
     pass** so fwd/bwd fits display as separate tracks; extra `pass`/`status`
     branches; `rr` filled; NO PR-stage branches (flag_vertex etc.).
   - **Two-TPC channel convention (owner-chosen)**: per-plane concatenation,
     `global = base[plane] + apa*nch[plane] + wire`, nch computed from the
     anodes at run time (SBND per TPC: U 1984, V 1984, W 1670 → planes span
     U [0,3968), V [3968,7936), W [7936,11276); 857 4-tick time slices).
   - `T_proj_data` (meas/pred/err per channel-slice, from the "stm" slot;
     entries duplicated under each pass block id of the tagged cluster),
     `T_bad_ch`, `Trun` (dQdx_scale 0.1 / offset −1000), empty `T_proj`.
   - `T_stm_pass` / `T_stm_eval`: flat copies of the decision PCs (cm units)
     so python reads need no log parsing.
   - Converter: the existing `wire-cell-uboone-magnify-tracking-convert`
     works as-is in data mode (`-f2`) → five-tree Magnify file.  Its MC mode
     (`-f1`) applies a hard-coded uBooNE x transform — do NOT use for SBND
     truth pairing until a pass-through flag is added.
2. **Bee layer** — `bee_points_sets` entry `stm_fit` (visitor-keyed
   `TaggerCheckSTM:pr`, present in the compiled config only when the knob is
   on) adds `*-stm_fit-global.json` to `mabc-pr.zip` with
   `q = dQ*0.1 − 1000` (same convention as the PR `track_fit` layer).
   wire-cell-bee3 itself is NOT modified (owner directive); the layer name
   avoids the `-track` substring bee3 filters.  Implemented as an `stm_fit`
   pcname branch in `MultiAlgBlobClustering::fill_bee_points_from_cluster`
   (signature gained defaulted dQdx_scale/offset params).
3. **nusel_display panel** — new "STM fit: dQ/dx vs residual range" figure +
   scalar Div (`render_dqdx()`), reading `tracking-stm.root` via uproot
   (panel reports when uproot or the file is absent):
   - measured dQ/dx (e/cm) vs rr for the focused bundle's main cluster;
     color = pass (fwd blue / bwd red), marker = TPC from drift-x sign
     (TPC0 circle / TPC1 triangle) so two-TPC coverage is visible per track;
   - overlays: the muon expectation (`nusel_display/stm_ref_dqdx.json`,
     extracted from the compiled config's MuonDeDx LinterpFunction — the
     exact table `eval_stm` uses; plateau ~49 ke/cm) and the flat 50 ke/cm
     line;
   - the scalar Div lists per-pass status and every eval row (ks1/ks2/
     ratios/res_length → PASS/fail);
   - the fitted trajectory is overlaid as black crosses on the three 2-D
     projections.
   TSV columns are unchanged (verdict schema untouched); decision scalars
   live in the ROOT file.
4. **Magnify-tracking-SBND GUI** (separate repo,
   `~/work/scratch_wcgpu1/toolkit-dev/Magnify-tracking-SBND`, also reachable
   as `sbnd_xin/Magnify-tracking-SBND`): geometry constants ported off
   MicroBooNE — `Data.cc` nChannel U/V/W = 3968/3968/3340, nTime = 857,
   `DrawBadCh` splits now use the nChannel variables; `ControlWindow.cc`
   ranges `{0,0,3968,7936}` / `{857,3968,7936,11276}`.  `rebin` stays 4 (the
   uBooNE value is already the SBND one).  Committed `b78b255` on `master`
   and pushed; visual validation still pending (needs an X/VNC session).

## Verification

- Compiled-config proofs (wcsonnet; pre-edit baselines from stashed HEAD
  bf6ddc92, files `/home/xqian/tmp/stmfit/{base,post}_*.json`):
  - knob off at the doc-39 op point AND at production-default TLAs:
    `cmp`-identical;
  - knob on: `save_stm_fit: true` on the tagger, `stm_fit` bee entry,
    `SbndMagnifyTrackingVisitor:pr` at the pipeline tail, `WireCellRoot` in
    plugins.
- `wcdoctest-clus`: 518/518 pass.  Freshness proof done (M1 was actually
  caught once: a wrongly-cwd'd wcbuild left stale libs; re-run properly).
- **Knob-off runtime gate PASS**: 30-event PR-only rerun
  (`work-mcp{10,1000,1000b}-stmoff`, QL symlinked from `*-fvxy`) vs the
  pre-change `*-fvxy` outputs — 30/30 TSVs identical, 30/30 `mabc-pr.zip`
  member-content hashes identical (`abtest/hash_archive.py`).
- **Knob-on smoke** (evt284657, `work-mcp10-stmon`): TSV identical to
  knob-off (the dump does not perturb verdicts); `mabc-pr.zip` gains only
  `0-stm_fit-global.json` (554 pts, clusters 14/20/27); `tracking-stm.root`
  checks — blocks 140/200/270, statuses {0,4,2} matching the log, pu/pv/pw
  inside the concatenated ranges with U hits in BOTH TPCs, pt < 857, rr up
  to 193 cm, `T_stm_pass`/`T_stm_eval` populated; converter run produces the
  five-tree Magnify file with correct per-track lengths.
  One crash was found+fixed during smoke: the hand-off fitter needed
  `set_detector_volume`/`set_pc_transforms` before `add_segment`
  (BuildGeometry segfault otherwise).
- **qlport uboone gate** (A = stashed HEAD bf6ddc92 build, labels
  `sweep/pre_stmfit` vs `sweep/post_stmfit`, 35 events, 0 failures/side):
  - Gate 1 (Bee `mabc_*.zip` member-content hashes): **35/35 identical —
    PASS**.
  - Gate 2 (tagger-compare verbose logs): 32/35 differ — **shown to be the
    pre-existing cross-recompile fragility, not this change**:
    (a) same-binary back-to-back reruns differ by order-only permutations of
    the same value multisets (16 lines, 0 value diffs — live run-to-run
    ordering flap in `shw_sp_pio_2_v_*`-class vector fills, e.g. the
    `graph_nodes` loop in `NeutrinoTaggerSinglePhoton.cxx`, a file this
    change does not touch);
    (b) rebuilding the SAME source with one inert throwaway line
    (`sweep/stmfit_probe_sweep` vs `post_stmfit`, identical sweep context)
    differs on 33/35 events with 2072 value-level line diffs — an order of
    magnitude MORE than this change's A-vs-B (32 events, 102 value diffs).
    Gate 2 is therefore currently insensitive at recompile granularity
    (matches the accepted FP-fragile class in
    memory `project_ql_matching_pointer_nondeterminism`; its zips-level
    physics products are what gate 1 checks, and those are identical).
    **Pre-existing finding reported to the owner, not fixed here.**
- **abtest (pdhd+pdvd) clus-only — PASS**: labels
  `snap/pre_stmfit_clus` vs `snap/post_stmfit_clus`; `ab_compare` OVERALL
  PASS, 70/70 archive comparisons identical on the 3 events with intact
  inputs (pdhd 027305_0, pdvd 039252_5, pdvd 039349_0; wall/RSS flat).
  The other 3 pdhd manifest events have no cluster tarballs in work/ and
  the img stage cannot rerun anywhere (NF+SP frames removed from work/
  dirs — pre-existing cleanup, symmetric on both sides; img/ is untouched
  by this round).

## Phase-4 first pass: 30-event knob-on round (`work-mcp{10,1000,1000b}-stmon`)

30/30 events processed with `-stm-fit` at the doc-39 op point (QL symlinked
from `*-fvxy`).  TSV verdicts vs the knob-off round: **0 real differences**
(12 column flips, all the known torn-log 0/−1 parse artifact — the extra
`persist_stm_fit` INFO lines raise the concurrent-write tear probability;
verified line-by-line against the raw logs).  Viewer re-served on :5010,
tag `mcp10-stmfit`, prevs fvxy(×3) → mainreal → mainpair → fvzi → lm2.

STM fit inventory — `python3 stmon_stats.py` from `sbnd_xin/` (committed
alongside these docs; it reads the three `*-stmon` roots and reprints every
number in this section):

- 36 fitted (cluster, pass) records over 30 events, all forward passes:
  11 accepted-STM, 13 rejected long-leftover, 5 rejected dQ/dx eval,
  6 rejected extra-tracks, 1 rejected proton-end.
- Two-TPC coverage: 18 fits span both TPCs, 13 TPC1-only, 5 TPC0-only —
  both TPCs well populated (per-point TPC from drift-x sign).
- dx: median 0.60 cm (p10 0.56, p90 0.70); reduced_chi2 p90 2.7, max 18.8.
- **x frame check** (doc 40 §2 left this as a CONFIRM): all 18561 fitted
  points fall in x ∈ [−201.3, +198.2] cm against SBND's 200 cm half-drift
  (0.17 % marginally past 200, fit overshoot at the anode).  The fits are
  therefore in the T0-corrected frame that `switch_scope` installs, not the
  raw one — a raw frame would spread over the ~2.7 m readout window.  The
  per-point TPC assignment by `sign(x)` used here and in the viewer is sound.
- **MIP plateau flag (hand-scan/calibration item, NOT tuned here)**:
  median fitted dQ/dx for rr > 40 cm on accepted-STM tracks is
  TPC0 59.5 ke/cm (p25–p75 50.5–82.8), TPC1 55.8 ke/cm (48.2–72.1) vs the
  ~50 ke/cm flat / ~49 ke/cm muon-table reference — the CORE (p25) sits at
  the reference but a high-side tail drags the medians ~10–20% up, with a
  mild ~6% TPC0>TPC1 asymmetry.  Over ALL fits (rejected included) the
  medians are 72/59 ke/cm.  Candidate explanations to check in the
  hand-scan: delta rays/overlaps on the fit path, rr measured from the
  wrong end on rejected fits, dx underestimation on angled segments, or a
  real calibration-scale offset feeding the eval_stm ratio cuts.

Final unit tests on the shipped binary: `wcdoctest-clus` 518/518 pass.

**Still open in phase 4** (doc 40 §Phase 4 has the per-check table): the
Magnify/Bee trajectory hand-scan and the viewer dQ/dx scan (both need an
X/VNC session, owner-side), the Bragg-shape review per stopping candidate,
the determinism repeat check under `setarch -R`, and MC truth pairing — the
last blocked until the converter's `-f1` mode gets a pass-through
x-transform flag (its current transform is hard-coded uBooNE).

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# knob-on nusel chain (one event; 'all' for the batch):
F="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit"
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-stmon ./run_nusel_evt.sh data 2 $F
# Magnify conversion (data mode; -f1 is uBooNE-only):
wire-cell-uboone-magnify-tracking-convert \
  -bwork-mcp10-stmon/nusel_evt284657/tracking-stm.root -tT_rec_charge \
  -o/path/track_com_284657.root -f2
# GUI: cd ~/work/scratch_wcgpu1/toolkit-dev/Magnify-tracking-SBND && ./magnify.sh /path/track_com_284657.root
# viewer with the panel: nusel_display/serve_nusel_scan.sh ... (reads
# nusel_evt*/tracking-stm.root automatically when present)
# fit inventory / plateau / x-frame numbers quoted above:
python3 stmon_stats.py
```

# Validating the STM tagger's track trajectory + dQ/dx fitting on SBND — plan (doc 40)

Status: **IMPLEMENTED — phases 0–3 landed 2026-07-24, phase 4 in progress.**
As-built record + gates: **doc 41** (`41_stm-fit-dump.md`).  Commits:
toolkit `3db191e9` (apply-pointcloud), wcp-porting-validation `6099ed0`
(main), Magnify-tracking-SBND `b78b255` (master) — all pushed.  The plan text
below is left as written; `[x]/[ ]` marks and **AS-BUILT** notes record what
actually happened, including where the implementation diverged from the plan.

Owner request 2026-07-24: before
validating the STM tagger verdicts themselves, validate the track-trajectory +
dQ/dx fitting that the tagger is built on, via (1) a dedicated
Magnify-tracking file, (2) wire-cell-bee display, and (3) new panels in
`nusel_display` (dQ/dx near the track end vs stopping-particle expectation),
with explicit two-TPC coverage.  This doc records the investigation findings
and the detailed plan.

## Repro (current state)

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# STM-only PR pipeline on one event (fit runs, but persists NOTHING except flags):
./run_pr_evt.sh -stm data <evt>       # pipeline=switch_scope,steiner,fiducialutils,tagger_check_stm
# full nusel chain (TGM+STM+FC) as used for the 30-event scans:
./run_nusel_evt.sh data 1 <op-point flags>   # op point per docs/39
# AS-BUILT: the same with the fit dump on (doc 41 has the full Repro block):
./run_nusel_evt.sh data 1 <op-point flags> -stm-fit   # + tracking-stm.root, stm_fit Bee layer
# TrackFitting parameters actually used by SBND:
cat sbnd_track_fitting.json           # sigmas coupled to cfg/.../sbnd/sp-filters.jsonnet
```

## 1. Findings (what exists today)

### 1.1 The STM tagger persists nothing about its fit

`clus/src/TaggerCheckSTM.cxx` (2265 lines, single-file component):

- `check_stm_conditions()` (`:1879`) drives, per main cluster:
  `cluster_fc_check` → endpoint choice → `run_pass` lambda (`:1989-2183`)
  executed **forward, and backward when double-ended** (`:2186-2191`).  Each
  pass does **two fitting rounds**: `do_rough_path` (Steiner shortest path) +
  `do_single_tracking` round 1, then `adjust_rough_path` (mid-track break
  detection + Steiner crawl re-route) + round-2 `do_single_tracking`
  (`:2001-2016`).  Then `find_first_kink`, exit/left split, TGM short-circuit,
  `eval_stm_core` KS tests, `search_other_tracks`/`check_other_tracks`,
  `detect_proton`.
- **Outputs today: only `Flags::STM` / `Flags::TGM` cluster flags** (`:165`,
  `:2073/:2086`) plus one INFO line per cluster and trace-level diagnostics
  (kink table `:852/:951`, `eval_stm` KS values `:1375`, proton detection
  `:1193/:1207`, exit/left summary `:2045`).
- The fit results (`fits()` = per-point x,y,z, dQ, dx, pu/pv/pw/pt, paf,
  reduced_chi2) live on **transient** `PR::Segment` objects;
  `m_track_fitter.clear_segments()` runs between rounds and the private
  fitter is dropped on return.  **STM never calls
  `grouping.set_track_fitting()`** — only `TaggerCheckNeutrino.cxx:555` does.
- A commented-out dump harness at `:186-227` and `:2201-2259` marks the
  intended hook: `create_segment_fit_point_cloud(segment, m_dv, "fit")` →
  `segment->dpcloud("fit")`.

Everything we need is reachable from `TrackFitting`
(`clus/inc/WireCellClus/TrackFitting.h`):

| Quantity | Accessor |
|---|---|
| fitted trajectory | `get_fine_tracking_path()` |
| per-point dQ, dx | `get_dQ()`, `get_dx()` |
| per-point wire/tick projections | `get_pu/pv/pw/pt()`, `get_paf()` (apa/face) |
| per-point fit quality | `get_reduced_chi2()` |
| measured 2D charge | `get_charge_data()` keyed `(apa,time,channel)` |
| predicted vs measured 2D charge | `get_fitted_charge_2d()` keyed `(apa,face,plane)→(wire,time)`, `FittedCharge2D{charge, charge_err, pred_charge, flag}` |
| 3D↔2D hit association | `m_3d_to_2d` (`Point3DInfo`, per-plane associated `Coord2D` sets) |

The trajectory fit and dQ/dx fit are BiCGSTAB sparse solves with Gaussian
charge division (`div_sigma`) and Tikhonov regularization (`lambda`), ported
from prototype `PR3DCluster_trajectory_fit.h` / `PR3DCluster_dQ_dx_fit.h`.
SBND parameters: `sbnd_xin/sbnd_track_fitting.json` (smearing sigmas MUST
match `sbnd/sp-filters.jsonnet`; derivation in
`sbnd/docs/sbnd-pattern-recognition.md §6.2`), threaded via
`--tla-str trackfitting_config=...` → `tagger_check_stm` in
`cfg/pgrapher/experiment/sbnd/clus.jsonnet:522-526`, with
`sbnd_box_recomb` (Box A=1.0, B=0.255, Efield 0.5) and the NIST/PDG
`particle_dataset` for the muon dE/dx reference.

### 1.2 The uBooNE Magnify-tracking dump chain (the model to follow)

- `root/src/UbooneMagnifyTrackingVisitor.cxx` reads
  `grouping.get_track_fitting()` (i.e. the **neutrino** tagger's persisted
  fitter) and writes per event: `T_rec_charge` (flat, per fitted point:
  `x,y,z` cm, `q = dQ*dQdx_scale + dQdx_offset`, `nq = dx/cm`,
  `ndf` = cluster id, `pu,pv,pw` with uBooNE plane offsets 0/2400/4800, `pt`
  in time slices, `reduced_chi2`, PR-stage extras `flag_vertex, flag_shower,
  rr, sub_cluster_id, particle_id`), `T_proj_data` (per cluster:
  `channel, time_slice, charge, charge_err, charge_pred` from
  `FittedCharge2D`), `T_bad_ch`, `Trun` (`dQdx_scale/dQdx_offset`).
  Documented in `clus/docs/magnify_tracking_output.md`.
- `root/apps/wire-cell-uboone-magnify-tracking-convert.cxx` regroups
  `T_rec_charge` by `ndf` into the nested-vector `T_rec`
  (`rec_x/y/z/dQ/dx/L/u/v/w/t`, `rec_cluster_id`, optional
  `reduced_chi2/sub_cluster_id/flag_vertex`, MC truth pairing
  `com_dis/com_dtheta/stat_*`) and clones `T_proj_data`/`T_bad_ch` — exactly
  the five-tree format the Magnify-tracking GUI reads.  **Caveat:** its MC
  mode (`-f1`) applies a hard-coded uBooNE x transform
  (`(x+0.6)/1.098*1.1009999-0.1101`, lines 148/183) — SBND must run data mode
  (`-f2`) or add a pass-through flag.

### 1.3 Magnify-tracking-SBND is not yet SBND

`/home/xqian/work/scratch_wcgpu1/toolkit-dev/Magnify-tracking-SBND` is a fork
of BNLIF/Magnify-tracking with feature/cleanup commits only — **geometry is
still pure MicroBooNE**:

- `event/Data.cc:93-96`: `nChannel_u=2400, nChannel_v=2400, nChannel_w=3456,
  nTime=2400` (8256 global channels, single TPC).  Plane routing is by global
  channel range (`Data.cc:497-510`, `:995-1022`); `DrawBadCh` uses literal
  2400/4800 splits (`:850-853`); `ControlWindow.cc:80-81` hard-codes the same
  ranges; `LoadBadCh` hard-codes `rebin=4`.
- Input = one ROOT file with `T_true` (optional; absence = data mode),
  `T_rec` (nested-vector), `T_proj_data`, `T_bad_ch` (`T_proj` loaded but
  unused).  Channels are **globally concatenated**, not per-plane/per-TPC.
- Panel inventory (3×3 canvas): dQ/dx vs L (pad 1), MC-compare (pad 2), 3D
  (pad 3), per-plane measured channel×time with fit points (pads 4-6),
  per-plane `(pred−meas)/meas` residual (pads 7-9).  The **generic
  trajectory/dQ/dx panels are exactly what STM validation needs**; the
  PR-stage extras (sub-cluster coloring, all-cluster overlay, vertex flags)
  can be left dormant — STM emits one track per pass with no
  `sub_cluster_id`/`flag_vertex` branches, and the reader already guards on
  branch existence.
- No two-TPC concept anywhere.  This port is the largest new work item of
  avenue 1.

### 1.4 Bee (bee3) display capabilities and the SBND wiring

- Bee3's layer schema (`wire-cell-bee3/events/static/js/bee/physics/sst.js:42-79`)
  is exactly `{x,y,z,q,cluster_id,real_cluster_id}` + RSE; coloring by
  cluster id or by a `q` HSL ramp.  **No polylines, no dQ/dx axis, no
  track/shower or residual-range concepts** — the uBooNE chain smuggles
  semantics into `q`: `track_fit` layer `q = dQ*0.1 - 1000` (fitted dQ/dx),
  `shower_track` layer `q = 15000` shower / `0` track, `vertices` layer main
  vertex `q=15000`.  `events/models.py:76-84` filters out filenames
  containing `-track` — layer names must avoid that substring.
- SBND already declares these three PR layers + `bee_pf` "mc" tree in
  `cfg/pgrapher/experiment/sbnd/clus.jsonnet:633-696`, but all keyed
  `visitor: 'TaggerCheckNeutrino:pr'` — **inert in the STM-only pipeline**.
  The `mabc-pr.zip` mechanism (MultiAlgBlobClustering `bee_points_sets`,
  visitor-keyed dump-after) is the ready-made transport; we need an
  STM-keyed variant.

### 1.5 nusel_display plug-in points

`nusel_display/nusel_scan_viewer.py` (1317 lines): per event it reads the
verdict TSV, the pctree tarball (`parse_pctree`), and
`ql_evt*/mabc-all-apa.zip` `*-clustering-global.json`
(`x,y,z,q,cid,rid` — the only per-point source; `mabc-pr.zip` is currently
**never opened**).  A dQ/dx panel plugs in as: extra read in
`Event.__init__` (`:341-367` mirror), a new `render_dqdx()` fanned out from
`refresh()` (`:1005-1011`), a new figure in the `layout` (`:1296-1314`);
focus plumbing (`bundle_cluster_ids`, `:371-378`) already exists.  Residual
range is not stored anywhere today — it must be dumped or computed as arc
length from the tagged end.

### 1.6 Prototype reference for A/B of the physics

`prototype_base/pid/src/ToyFiducial.cxx`: `check_stm` (`:405`), `eval_stm`
(same `(peak_range, offset_length, com_range)` combos as the toolkit),
`find_first_kink` (`:1024`), `detect_proton` (`:1268`), with per-function
docs under `prototype_base/pid/docs/ToyFiducial/`.  Standalone apps
`wire-cell-prod-stm{,-port}.cxx` exist for the uBooNE chain, and the qlport
harness already gates that chain byte-identically — giving us a
**cross-detector sanity anchor**: the same toolkit fitting code validated
against the prototype on uBooNE events.

## 2. The central design decision: one dump, three consumers

All three validation avenues need the same data, which today dies inside the
tagger.  So the backbone is a single **default-OFF dump knob on
`TaggerCheckSTM`** (working name `save_stm_fit`), and three thin consumers.

What to record, per main cluster and per pass (fwd / bwd):

- **Per fitted point**: x, y, z (fitter output; already T0-corrected —
  confirm against `switch_scope`'s `x_t0cor` convention), dQ, dx, cumulative
  L, residual range from the identified track end, pu, pv, pw, pt, apa/face
  (`get_paf`), reduced_chi2, cluster_id, pass index, round index (keep only
  the final round-2 fit per pass; the round-1 rough fit is diagnostic noise).
- **Per plane 2D**: measured + predicted charge (`get_fitted_charge_2d`),
  i.e. `T_proj_data` content, keyed (apa, face, plane, wire/channel, time).
- **Per cluster scalars** (the tagger's decision inputs, today trace-only):
  kink position/index, exit_L/left_L, exit and left dQ/dx, `eval_stm` KS
  values (ks1 vs muon, ks2 vs flat) and ratios per tested combo, michel
  `res_length`/`ave_res_dQ_dx`, `detect_proton` verdict, FC-check exit
  points, and the final STM/TGM outcome — so the display can say *why*.

**AS-BUILT (§2 record list).** Per-point: all of the above, in PC `stm_fit`
(x,y,z,dQ,dx,L,rr,pu,pv,pw,pt,apa,face,reduced_chi2,pass,status); round-1
rough fits dropped as planned.  The **x frame** ("confirm against
`switch_scope`'s `x_t0cor`") is resolved: persisted x is
`Segment::fits()[i].point.x()`, i.e. the fitter's working frame = the
cluster's default scope, which the PR pipeline's `switch_scope` sets to the
T0-corrected coordinates.  Empirical confirmation over the 30-event knob-on
round: all 18561 fitted points lie in x ∈ [−201.3, +198.2] cm against SBND's
200 cm half-drift, only 0.17 % marginally past 200 (fit overshoot at the
anode) — a raw-x frame would spill across the full ~2.7 m readout window.
TPC assignment by `sign(x)` (used by the viewer and `scripts/analysis/stm/stmon_stats.py`) is
therefore sound.
Per-cluster scalars landed as two PCs: `stm_pass`
(status/kink_num/npoints/exit_L/left_L/exit_dqdx/left_dqdx) and `stm_eval`
(one row per `eval_stm_core` call: combo params, ks1, ks2, ratio1, ratio2,
res_length, ave_res_dqdx, verdict).  **Divergences**: `detect_proton` has no
separate column — a proton-end rejection is folded into the pass status code
(5); FC-check exit points were **not** recorded (the FC decision is already
visible in the verdict TSV and the FC Bee layers).

Persistence targets (both behind the same knob):

1. **Grouping hand-off** for the ROOT visitor: after the passes, before
   clearing, snapshot the accepted segments into the grouping the same way
   `TaggerCheckNeutrino.cxx:555` does (or a dedicated
   `set_track_fitting("stm")` slot to avoid colliding with the neutrino
   tagger when both run — to be settled at implementation review).
   **AS-BUILT**: named slot `"stm"` (owner Q2), new additive
   `Facade::Grouping::{set,get}_track_fitting(name)`; the unnamed slot the PR
   chain uses is untouched.  The holder fitter accumulates every recorded
   pass plus the per-pass 2D charge (new `TrackFitting::merge_fitted_charge_2d`)
   — needed because `clear_segments()` drops it between rounds.
2. **pctree point clouds + cluster scalars** via the commented
   `create_segment_fit_point_cloud` harness (`TaggerCheckSTM.cxx:186-227`):
   an `stm_fit` PC on the cluster node (columns above) plus scalar entries.
   These survive the TensorDM tarball → `nusel_extract`/viewer and the Bee
   dump can both feed from it.  GOTCHA from doc 38 applies: TensorDM
   `as_tensors` drops heterogeneous same-named PC keys and
   `separate()/from()` drops node-local PCs — the doc-38
   `save_real_cluster_id` plumbing is the template for doing this right.
   **AS-BUILT divergence**: the PCs are written on the cluster node and are
   consumed **in the same job only** (Bee layer + the ROOT visitor) — the PR
   job does not re-save the pctree tarball, so `nusel_extract`/the viewer do
   NOT see them.  The viewer therefore reads `tracking-stm.root` (uproot)
   instead of the tarball, and the doc-38 TensorDM gotcha never came into
   play.

Knob-off must be byte-identical (standard gate on the 30-event manifests +
pdhd/pdvd abtest since TaggerCheckSTM is shared code — any change must be
additive-only under the knob).

## 3. Phase plan

### Phase 0 — dump backbone (prerequisite)  — **DONE** (toolkit `3db191e9`)

- [x] `save_stm_fit` knob in `TaggerCheckSTM` implementing §2 (C++ +
      key-suppressed jsonnet threading + runner flag `-stm-fit`).
      As-built: also `-no-stm-fit` / env `SBND_STM_FIT`; rejected passes
      recorded with a status column (owner Q4).
- [x] Residual-range definition fixed here once: arc length from the
      candidate stopping end (the end `eval_stm` tests), recorded per point.
      As-built: `rr = L_tot − L_i`, i.e. measured from the path end.
- [x] Gates: knob-off byte-identical (SBND 30-event roots + abtest
      pdhd/pdvd + qlport, since clus/ is shared); knob-on smoke on 1 event
      per TPC with sentinel log.
      As-built (details + labels in doc 41): SBND 30/30 TSV + `mabc-pr.zip`
      hashes identical vs `*-fvxy`; qlport gate 1 35/35 identical; abtest
      clus-only OVERALL PASS on the 3 events with intact inputs;
      `wcdoctest-clus` 518/518.  **Two caveats recorded in doc 41, not
      fixed**: qlport gate 2 is insensitive at recompile granularity
      (pre-existing), and the abtest manifest's NF+SP frames are missing
      from the pdhd/pdvd work dirs so the img stage could not be re-run.

### Phase 1 — Magnify-tracking file + GUI (avenue 1)  — **DONE** (toolkit `3db191e9`, Magnify `b78b255`)

- [x] **`SbndMagnifyTrackingVisitor`** — fork-by-duplication (M10) of
      `UbooneMagnifyTrackingVisitor`; production uBooNE file untouched.
      Reads the STM fitter snapshot; writes `T_rec_charge` (no
      `flag_vertex`/`sub_cluster_id`/`particle_id` — STM stage has none;
      `rr` filled), `T_proj_data`, `T_bad_ch`, `Trun`.
      As-built: block id `ndf` = `cluster_id*10 + pass` so the fwd/bwd fits
      display as separate tracks; extra `pass`/`status` branches; two extra
      flat trees `T_stm_pass`/`T_stm_eval` so python needs no log parsing.
- [x] **Two-TPC channel convention** (the key SBND decision): globally
      concatenated channels `U = [0, 2·N_u)`, `V`, `W` with TPC1 offset by
      the per-plane TPC0 count — i.e. within each plane pad, TPC0 and TPC1
      appear side by side, satisfying "data in both TPCs" in one view.
      Channel counts to be read from the SBND wires file at implementation
      (expected 1984/1984/1664 per TPC → 5632 per TPC, 11264 total —
      CONFIRM, do not trust this line).  Time axis: SBND readout ticks /
      rebin (confirm slice count from the SP config; uBooNE's `rebin=4` and
      `nTime=2400` are both hard-coded in the GUI and must become SBND
      values).
      **CONFIRMED at implementation** (do not use the guessed line above):
      per TPC U 1984, V 1984, **W 1670** (not 1664) → globals U [0,3968),
      V [3968,7936), W [7936,11276), 11276 channels total; time axis 857
      slices (4-tick rebin of 3427 ticks).  Counts are computed from the
      anodes at run time, not hard-coded in the visitor.
- [x] **Converter**: reuse `wire-cell-uboone-magnify-tracking-convert` in
      data mode (`-f2`) initially (MC-truth pairing needs the uBooNE x
      transform neutralized — small follow-up flag, or an SBND copy).
      As-built: `-f2` works unmodified; the pass-through flag for `-f1` is
      still **not written**, so MC truth pairing stays blocked (phase 4).
- [x] **Magnify-tracking-SBND GUI port** (in that repo, not the toolkit):
      `nChannel_u/v/w`, `nTime`, `ControlWindow` ranges, `DrawBadCh` splits,
      `rebin` — all currently MicroBooNE literals (`Data.cc:93-96`,
      `:850-853`, `ControlWindow.cc:80-81`).  Follow `Magnify-PDVD/docs/`
      (input_format.md, porting notes) as the template.  PR-stage panels
      (sub-clusters, all-cluster overlay) left dormant.
      As-built (`b78b255`): `Data.cc` nChannel 3968/3968/3340 + nTime 857,
      `DrawBadCh` splits now use the nChannel variables instead of literal
      2400/4800, `ControlWindow.cc` ranges `{0,0,3968,7936}`/`{857,3968,7936,11276}`.
      `rebin` stays 4 — uBooNE's value happens to be the SBND one.
- [ ] **OPEN** — Validation use: hand-scan fit-vs-charge overlay per plane
      per TPC, pred/meas residual panels, dQ/dx vs L with Bragg rise.
      The GUI is ROOT-graphical: needs an X/VNC session, so this is an
      owner-side scan (phase 4 check 1).

### Phase 2 — Bee display (avenue 2)  — **DONE** (toolkit `3db191e9`)

**wire-cell-bee3 itself is NOT modified** (owner directive): we only save
results in the Bee JSON format it already reads
(`{x,y,z,q,cluster_id,real_cluster_id}` layers inside the mabc zip), exactly
as the MicroBooNE chain does.

- [x] Add STM-keyed `bee_points_sets` entries in
      `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (visitor key
      `TaggerCheckSTM:<name>`, dump-after mechanism already generic):
      - `stm_fit` layer: fit points, `q = dQ·0.1 − 1000` (same
        scale/offset convention as the PR `track_fit` layer);
      - optionally `stm_marks`: kink + endpoints (`q` flag values), the
        cheapest way to see the tagger's decision geometry.
      Names must avoid the substring `-track` (bee3 `models.py:76-84`
      filters it).  Layers inert unless `tagger_check_stm` in pipeline AND
      knob on (key-suppression keeps `mabc-pr.zip` byte-identical
      otherwise).
      As-built: only the `stm_fit` layer was added (`*-stm_fit-global.json`
      in `mabc-pr.zip`, `q = dQ·0.1 − 1000`), via a new `stm_fit` pcname
      branch in `MultiAlgBlobClustering::fill_bee_points_from_cluster`
      (signature gained defaulted `dQdx_scale`/`dQdx_offset`).  The optional
      `stm_marks` kink/endpoint layer was **not** implemented — the kink and
      endpoint scalars are in `stm_pass`/`tracking-stm.root` and the viewer
      panel shows them; add the layer only if the Bee hand-scan wants it.
- [x] Accept Bee's limits knowingly: points-only (no polyline), no residual
      range, `q`-ramp coloring — good for trajectory sanity + Bragg-peak
      visibility, mirroring how MicroBooNE used `WireCell-charge` vs
      tracking layers.  Track-vs-shower / subcluster layers are PR-stage;
      deferred until the PR campaign.
- [x] Any upload to the public Bee server stays owner-gated
      (`upload-to-bee.sh`, escalation rule 6); local bee3 serving is the
      default.  As-built: nothing was uploaded; bee3 untouched.

### Phase 3 — nusel_display panels (avenue 3)  — **DONE** (wcp `6099ed0`)

- [ ] **NOT IMPLEMENTED (deliberate deviation)** — `nusel_extract.py`: parse
      the `stm_fit` PC + STM scalar set from the pctree tarball; add TSV
      columns for the headline scalars (ks1, ks2, kink L, exit/left dQ/dx,
      proton flag) so the table can sort/filter on them.
      Why: the STM PCs never reach the tarball (see the §2 AS-BUILT
      divergence), and the decision scalars are already in
      `tracking-stm.root` as `T_stm_pass`/`T_stm_eval`, which the viewer
      reads directly.  `nusel_extract.py` is **untouched** and the TSV/label
      schema is unchanged, so every earlier scan round stays comparable.
      Revisit only if table-level sorting on ks1/ks2 becomes necessary.
- [x] Viewer: new bottom panel `render_dqdx()` for the focused bundle —
      dQ/dx vs residual range of the main cluster's STM fit, overlaid with
      (a) the stopping-muon expectation from the same `particle_dataset`
      dE/dx table pushed through `sbnd_box_recomb` (precomputed into a small
      static JSON/npz so the viewer stays self-contained), (b) the flat
      50 ke/cm MIP reference used by `eval_stm_core`; annotate kink
      position, KS values, and the per-point TPC (apa) via marker shape so
      two-TPC coverage is visible per track.  Fit trajectory also overlaid
      on the existing 2D projections (distinct glyph on `f_xy/f_yz/f_xz`).
      As-built: implemented as specified, except the source is
      `tracking-stm.root` via uproot (not the pctree), the reference curve is
      pre-extracted to `nusel_display/stm_ref_dqdx.json` from the compiled
      config's MuonDeDx `LinterpFunction` (the exact table `eval_stm` uses),
      and the kink is reported in the scalar Div rather than annotated on the
      curve.  Marker shape = TPC (circle/triangle), color = pass (fwd/bwd).
- [x] `--prev` chain untouched; new panel is read-only decoration, labels
      schema unchanged.

### Phase 4 — physics validation protocol (what "validated" means)  — **IN PROGRESS**

**AS-BUILT status of the six checks** (first pass = the 30-event knob-on
round `work-mcp{10,1000,1000b}-stmon`, inventory in doc 41; 36 fitted
(cluster, pass) records, all forward passes; two-TPC coverage 18 both /
13 TPC1-only / 5 TPC0-only, so the two-TPC requirement is met):

| # | Check | Status |
|---|---|---|
| 1 | Trajectory sanity (Magnify + Bee hand-scan) | **OPEN** — products exist (`tracking-stm.root`, converter `-f2`, `stm_fit` Bee layer, GUI ported); the scan itself needs an X/VNC session ⇒ owner-side |
| 2 | Fit-quality distributions | **DONE, with an open finding** — dx median 0.60 cm (p10 0.56, p90 0.70), reduced_chi2 p90 2.7 / max 18.8 (doc 41); **12.8 % of all 18561 fitted points have negative dQ/dx** (doc 42 §5), status 4 worst at 20.9 %, one accepted track 56 % negative |
| 3 | dQ/dx absolute scale per TPC | **DONE, with an open finding** — rr > 40 cm median on accepted-STM tracks TPC0 59.5 / TPC1 55.8 ke/cm vs the ~50 reference; p25 sits at the reference, high tail pulls the median ~10–20 % up, ~6 % TPC asymmetry.  Reported, **not tuned** (escalation rule 7) |
| 4 | Stopping-particle shape vs expectation | **STARTED, doc 42** — one accepted-STM track (evt286241 c8) reproduces the Bragg rise in shape, +10–25 % in normalization; but **only 1 of 11 accepted-STM long tracks shows any rise at all**.  Per-candidate hand scan still open |
| 5 | Determinism (repeat identity under `setarch -R`) | **NOT RUN** |
| 6 | Cross-detector anchor (uBooNE) | **PARTIAL** — qlport gate 1 35/35 zips identical, so the shared fitting path is unperturbed; gate 2 is currently insensitive at recompile granularity (pre-existing finding, doc 41) |

Also still open from phase 1: MC truth pairing needs a pass-through x-transform
flag in the converter's `-f1` mode (uBooNE transform hard-coded).



Sample: the standard 30-event roots (mcp10/mcp1000/mcp1000b) at the doc-39
op point, plus a **stopping-track-enriched selection** (clusters whose FC
check passes with one end well inside the FV).  Explicit two-TPC coverage
requirement: the summary must count fitted tracks per apa and include
TPC0-only, TPC1-only, and cathode-crossing cases (doc-38/xtpc machinery
gives us known crossers, e.g. the evt298567-class events on PDVD have SBND
analogues in the scan set).

Checks, in order of increasing physics content:

1. **Trajectory sanity** (Magnify + Bee): fit follows the charge in all
   three planes, both TPCs; no plane-flipping or dead-region derailments;
   pred/meas residual panels structureless along healthy track sections.
2. **Fit quality distributions**: reduced_chi2 per point/track; dx
   distribution vs `low_dis_limit`; fraction of points with dead-plane
   flags.
3. **dQ/dx absolute scale**: MIP plateau of long through-going muons per
   TPC and per plane-combination vs the ~50 ke/cm reference used by
   `eval_stm_core` — a TPC-asymmetric plateau would invalidate the shared
   cut values.
4. **Stopping-particle shape**: dQ/dx vs residual range for hand-confirmed
   stopping muons vs the muon dE/dx+recombination expectation (the exact
   curve `eval_stm` uses); Michel-tail cases inspected; `detect_proton`
   spot-checked on any proton-like enders.
5. **Determinism**: repeat-run identity of the fit output (the solvers are
   deterministic BiCGSTAB; verify under `setarch -R` like the qlport gate).
6. **Cross-detector anchor**: the identical fitting code path already
   passes prototype-vs-toolkit gates on uBooNE (qlport); any SBND-only
   anomaly therefore points at SBND config (sigmas vs `sp-filters`,
   recombination, geometry), not the fitter core — check
   `sbnd_track_fitting.json` couplings first.

Verdict-level STM validation (labels, efficiency vs hand scan) is the
follow-on campaign once the fitting is trusted; this doc deliberately stops
at the fitting.

## 4. Open questions for the owner — **ALL FOUR ANSWERED 2026-07-24**

The owner took the recommendation in every case; the as-built answers are:

1. **Two-TPC display**: per-plane concatenated channels, TPC0 then TPC1
   (confirmed counts 1984/1984/1670 per TPC).
2. **Grouping slot**: named slot `"stm"`, PR slot untouched.
3. **Sample**: MC first — the 30-event mcp10/mcp1000/mcp1000b roots at the
   doc-39 op point.
4. **Rejected passes**: yes, recorded, with a status column (codes 0–6).

The original wording is kept below for the record.



1. **Two-TPC display convention** for Magnify-tracking-SBND: side-by-side
   concatenated channels per plane pad (recommended above) vs per-TPC files/
   canvases?
2. **Grouping slot**: reuse `grouping.set_track_fitting()` (collides if
   `tagger_check_neutrino` later runs in the same job) or add a named slot?
   Recommendation: named slot, keeps the PR stage untouched.
3. Which stopping-muon sample anchors check 4 — MC (truth-tagged stoppers,
   enables the converter's truth-pairing panels) or data in-time cosmics?
   Recommendation: MC first (truth available), data second.
4. Does the STM dump also record the **rejected** passes (fits that failed
   eval_stm) — useful for false-negative debugging, at ~2× output size?
   Recommendation: yes, with a pass-status column, since STM events are the
   minority.

## 5. Effort/ordering summary

Phase 0 is the gate for everything and touches shared clus/ code (full
multi-detector byte-identical gates).  Phase 1 (ROOT visitor + converter
reuse) and Phase 2 (Bee layers) are independent consumers and can proceed in
parallel once 0 lands; the Magnify GUI geometry port is the one piece of
work outside this repo pair.  Phase 3 rides on the same pctree products and
is pure python.  Phase 4 is the actual campaign and produces the numbered
follow-up docs (41+: per-check results with Repro blocks).

**AS-BUILT**: phases 0–3 all landed in one day and in one toolkit commit
(`3db191e9`) plus one wcp commit (`6099ed0`) — the consumers turned out small
enough not to need separate rounds — with the Magnify GUI port in
`b78b255`.  Phase 4's first pass (checks 2, 3, and half of 6) is written up
in doc 41; the remaining checks are hand-scan/determinism items and will
extend doc 41 or start doc 42.

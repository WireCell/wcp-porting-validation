# qlport → SBND downstream port plan

Author: porting session 2026-05-27. Status: implementing as of this commit.

## Goal

Adopt the MABC downstream-after-clustering chain from
`qlport/uboone-mabc.jsonnet` into `sbnd/clus.jsonnet` so SBND clusters
go through `clustering_recovering_bundle → steiner → fiducialutils →
tagger_check_neutrino` (plus uboone-weighted BDT scorers as a smoke
test) and write the matching set of Bee point sets / particle-flow
output.

## How qlport runs (for reference)

```bash
source sbnd/setup-local-opt.sh
# qlport additionally needs the WCT cfg tree (uboone simparams)
export WIRECELL_PATH=$WIRECELL_PATH:/exp/sbnd/app/users/yuhw/wire-cell-toolkit/cfg

wire-cell -l stderr -L debug \
  -A kind=both -A beezip=mabc_0.zip \
  -A initial_runNo=5384 -A initial_subRunNo=130 -A initial_eventNo=6501 \
  -A infiles=qlport/rootfiles/nuselEval_5384_130_6501.root \
  qlport/uboone-mabc.jsonnet
```

Output: `mabc_0.zip` Bee archive with six point sets (`regular`,
`steiner`, `track_fit`, `shower_track`, `vertices`, `mc`) plus an
optional `track_com_<run>_<event>.root` tracking dump.

## qlport pipeline (the bit we port)

```jsonnet
local cm_pipeline = [
    cm.tagger_flag_transfer("tagger"),
    cm.clustering_recovering_bundle("recover_bundle", graph_name="relaxed_pid"),
    cm.switch_scope(),
    cm.steiner(retiler=improve_cluster_2, perf=perf),
    cm.fiducialutils(),
    cm.tagger_check_neutrino(trackfitting_config_file=..., recombination_model=...,
                             particle_dataset=..., dl_weights=..., clus_geom_helper=...),
] + numu_bdt + nue_bdt + (uboone_tracking_visitor_pair);
```

| # | Visitor | C++ type | Needs |
|---|---|---|---|
| 1 | `tagger_flag_transfer` | ClusteringTaggerFlagTransfer | upstream `tagger_info` PC on each cluster |
| 2 | `clustering_recovering_bundle` | ClusteringRecoveringBundle | dv, pcts, `relaxed_pid` graph |
| 3 | `switch_scope` | ClusteringSwitchScope | pcts (`T0Correction` scope) |
| 4 | `steiner` | CreateSteinerGraph + ImproveCluster_2 retiler | dv, pcts, blob sampler with `disable_mix_dead_cell:false` |
| 5 | `fiducialutils` | MakeFiducialUtils | CompositeFiducial (Poly XY ∧ Poly ZX) |
| 6 | `tagger_check_neutrino` | TaggerCheckNeutrino | trackfitting json, BoxRecombination, ParticleDataSet, (opt) DL/SCE |
| 7–8 | `numu_bdt_scorer`, `nue_bdt_scorer` | UbooneNumuBDTScorer / UbooneNueBDTScorer | per-BDT XML weights + xgboost XML |

## Decisions baked into the v1 SBND port

1. **Enable `tagger_flag_transfer`.** Re-enabled 2026-05-28: the
   `match`-branch WireCellMatch port now tags `main_clus`, writing the
   upstream `tagger_info` PC that this step consumes. Pipeline starts
   with `cm.tagger_flag_transfer("tagger")` before
   `clustering_recovering_bundle`, matching qlport ordering.
2. **Append to existing `clus_all_apa.cm_pipeline`** (one MABC node)
   after the current `neutrino()`/`isolated()`.
3. **Reuse uboone weights as smoke test.** Wire BDT XML and DL `.pth`
   to the uboone files in `wire-cell-data/uboone/{weights,scn_vtx}/`
   to verify the C++ paths run; scores are meaningless for SBND.
4. **Out of scope:** UbooneMagnifyTrackingVisitor and
   UbooneTaggerOutputVisitor (uboone-only ROOT writer/format), real
   SBND SCE (`clus_geom_helper=""` disables).

## Implementation checklist (this commit)

### A. New jsonnet building blocks in `sbnd/clus.jsonnet`

1. `sbnd_box_recomb` — `BoxRecombination` with `Efield: 0.5` (SBND
   nominal vs uboone 0.273); other constants copied from uboone.
2. `sbnd_particle_dataset` — `ParticleDataSet` aggregating 10
   `LinterpFunction`s. Tables (NIST/PDG, detector-agnostic) come from
   the extracted `sbnd/particle_dataset.jsonnet`.
3. Composite fiducial = AND(PolyFiducial XY, PolyFiducial ZX). v1
   polygons are rectangles built from the existing `dvm` bounds:
   X ∈ [−202.5, +201.45] cm, Y ∈ [−200.312, +200.312] cm, Z ∈ [4.05,
   505.35] cm.
4. `bs_live_no_dead_mix_face(apa,face)` — clone of `bs_live_face`
   with `disable_mix_dead_cell:false` to let `ImproveCluster_2` mix
   dead-region info.
5. `improve_cluster_2_sbnd` — `cm.improve_cluster_2(anodes, samplers)`
   where samplers cover apa=0..N-1 face=0.
6. `sbnd/sbnd_track_fitting.json` — first pass = verbatim copy of
   `qlport/uboone_track_fitting.json`. Wire pitches differ; tune
   after v1 runs.

### B. Append to `clus_all_apa.cm_pipeline` (after `cm.isolated()`)

```jsonnet
cm.clustering_recovering_bundle("recover_bundle", graph_name="relaxed_pid"),
cm.steiner(retiler=improve_cluster_2_sbnd, perf=true),
cm.fiducialutils(),
cm.tagger_check_neutrino(
    trackfitting_config_file="sbnd_track_fitting.json",
    recombination_model=wc.tn(sbnd_box_recomb),
    particle_dataset=wc.tn(sbnd_particle_dataset),
    perf=true,
    dl_weights="uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth",
    dQdx_scale=0.1, dQdx_offset=-1000,
    clus_geom_helper=""),
cm.numu_bdt_scorer(... uboone numu_tagger*.xml, cos_tagger_10, numu_xgboost ...),
cm.nue_bdt_scorer(... 30 uboone BDT XMLs + XGB_nue ...),
```

Pass `fiducial=sbnd_fid` into the existing
`clus.clustering_methods(...)` so `cm.fiducialutils()` resolves it.

### C. `clus_all_apa` MABC node data — add bee outputs

```jsonnet
bee_points_sets: [
  ...existing,
  { name:"regular",      visitor:"CreateSteinerGraph",  pcname:"3d",         coords:["x_t0cor","y","z"], filter:1 },
  { name:"steiner",      visitor:"CreateSteinerGraph",  pcname:"steiner_pc", coords:["x_t0cor","y","z"] },
  { name:"track_fit",    visitor:"TaggerCheckNeutrino", grouping:"live", pcname:"3d", coords:["x","y","z"], individual:false, dQdx_scale:0.1, dQdx_offset:-1000 },
  { name:"shower_track", visitor:"TaggerCheckNeutrino", grouping:"live", pcname:"3d", coords:["x","y","z"], individual:false, use_associate_points:true },
  { name:"vertices",     visitor:"TaggerCheckNeutrino", grouping:"live", pcname:"3d", coords:["x","y","z"], individual:false, use_graph_vertices:true },
],
bee_pf: [ { name:"mc", visitor:"TaggerCheckNeutrino", grouping:"live" } ],
```

Extend `uses=` to cover the new components.

## Validation

1. `cd sbnd && ./js.sh all wcls-img-clus-matching` — must regenerate the
   `.json` graph and `.pdf` without jsonnet errors.
2. `lar -c sbnd/wcls-img-clus-matching.fcl -s sbnd/input-10ev/... -n 1`
   — single-event smoke test; confirm `mabc-all-apa.zip` now contains
   `regular`, `steiner`, `track_fit`, `shower_track`, `vertices`, `mc`.
3. Upload via `sbnd/upload-to-bee.sh` and eyeball the new point sets in Bee.

## Risk register / follow-ups

- `TaggerCheckNeutrino` with `clus_geom_helper=""`: verify in
  `clus/src/TaggerCheckNeutrino.cxx` that SCE block is short-circuited.
- Track-fitting JSON copied from uboone: wire pitch / orientations
  differ; expect mediocre dQ/dx until tuned for SBND.
- v1 fiducial polygons are loose rectangles — tighten with real
  cryostat geometry after v1 runs.
- BDT scores are meaningless until SBND retrains; scorers must not
  crash on uboone XMLs (their docstrings say missing XMLs are
  tolerated — verify).
- `numu_bdt_scorer` and `nue_bdt_scorer` must come **after**
  `tagger_check_neutrino` (input features are filled there). Order is
  enforced in the proposed pipeline.

## Reference

- qlport entry: `qlport/uboone-mabc.jsonnet`
- overview: `qlport/uboone-mabc-overview.md`
- shared method defs: `/exp/sbnd/app/users/yuhw/opt/share/wirecell/pgrapher/common/clus.jsonnet`
- env: `sbnd/setup-local-opt.sh` (puts `/exp/sbnd/app/users/yuhw/wire-cell-data` on WIRECELL_PATH).

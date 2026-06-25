# SBND wcls img+clus+QL-matching+patrec (Xin-faithful) — status 2026-06-08

Status of the artROOT-input wcls chain that mirrors Xin's standalone
imaging+clustering+Q/L-matching+pattern-recognition chain.

## Config / how to run

- Env: `source sbnd/setup-ap.sh` inside the SL7 apptainer. setup-ap.sh =
  setup-local-opt.sh + prepend `wire-cell-toolkit/cfg` (toolkit img/clus/
  qlmatching/simparams win, Xin's env; same wires geom as sbndcode) + sbnd_xin +
  `wire-cell-data/sbnd/photodet` (QLMatching semi-analytical model).
- Job: `sbnd/wcls-img-clus-matching-xin.{fcl,jsonnet}` (committed). Inputs are
  artROOT: `wclsCookedFrameSource` (charge) + `wclsOpFlashSource` (light, feeds
  `FlashTensorToOpticalPCs` directly — no opflash file dump).
- Imaging: toolkit `img.jsonnet` `multi-3view` + `full_deghost=true`
  (live port0 + `multi_masked_2view` dead port1). Clustering: toolkit
  `clus.jsonnet` per_apa + joint `QLMatching` (premerged all_apa). Output: one
  shared Bee zip `mabc.zip`.
- Run: `time lar --nskip 0 -n N -c wcls-img-clus-matching-xin.fcl -s standalone-sample/2025f-mc.root --no-output`
  then `BROWSER=echo bash sbnd_xin/upload-to-bee.sh mabc.zip`.

## Downstream pattern-recognition toggle

`wire-cell-toolkit/cfg/pgrapher/experiment/sbnd/clus.jsonnet` top:
`local enable_downstream_pr = true;`
- true  => full chain: tagger_flag_transfer -> clustering_recovering_bundle ->
  steiner -> fiducialutils -> tagger_check_neutrino -> numu/nue BDT scorers,
  plus steiner/track_fit/shower_track/vertices Bee point sets + `mc` particle-flow.
- false => matching-only all-APA MABC (img/clustering/op views only). Robust for
  all events; use this for bulk runs until issue #3 below is fixed.

## Local WCT changes (apply-pointcloud, NOT committed — review pending)

All under `/exp/sbnd/app/users/yuhw/wire-cell-toolkit`:
1. `match/src/QLMatching.cxx` + `inc/WireCellMatch/QLMatching.h`: port of
   larwirecell beam_flash/main_cluster tagging. `m_max_beam_flash_time` (5 us);
   in `apply_matched_t0s`, set "beam_flash" on in-beam matched clusters and
   "main_cluster" on the smallest-|flash_time| one (per APA run); call
   `Clus::Facade::normalize_cluster_flags` before tensor output (single + joint
   branches) so the flags survive the as_tensors schema. **WORKS.**
2. `clus/cfg/.../sbnd/clus.jsonnet`: added the qlport-style downstream chain to
   clus_all_apa, gated by `enable_downstream_pr`, with the supporting locals
   (pds/particle_dataset, sbnd_box_recomb, sbnd_fid PolyFiducial from dvm FV,
   smoke-test BDT/DL weights, improve_cluster_2 retiler).
3. `clus/src/TaggerCheckNeutrino.cxx` — two robustness fixes:
   - early-return when no cluster is flagged `main_cluster` (was a null deref at
     the old line 118 log call);
   - reset the persistent `m_track_fitter` (`clear_graph()` + `clear_segments()`)
     at the top of `visit()` — it kept dangling Facade::Cluster*/Blob* into the
     previous event's destroyed pctree -> sync_from_graph segfault.

## TGM tagger (ported from WCP, validated 2026-06-23)

WCP `pid/src/Cosmic_tagger.h::check_tgm` ported to
`wire-cell-toolkit/clus/src/TaggerCheckTGM.cxx` (apply-pointcloud, NOT committed).
- Faithful CASE A (both-ends-exit + 3-point through check + flag_check_again
  fallback) + CASE B (Hough push-out, prolonged-track→`check_signal_processing`,
  dead-region→`check_dead_volume`) over `get_extreme_wcps()`. NOT ported:
  `check_neutrino_candidate` (all flashes treated as type != 2; see file header).
- Tags ALL live clusters. Endpoints SCE-corrected via
  `SCECorrection::forward(p, t0=0, face, apa)` then box-tested against the overall
  FV box (no-op SCE today: no `sce_field` in DetectorVolumes metadata).
- Debug mode writes per-point charge (`tgm_charge` in `tgm_debug` PC; endpoints
  10000, body 100). Generic MABC support added:
  `bee_points_sets[].charge_array`/`charge_pcname` — dump only clusters carrying
  that array, charge read from it (in `MultiAlgBlobClustering.{h,cxx}`).
- Wired in `clus_all_apa` (always-on): pipeline tail
  `examine_bundles -> MakeFiducialUtils -> TaggerCheckTGM(debug=true)`, with a
  `BoxFiducial:all-overall-fv` (dvm.overall shrunk by margins) and a `tgm` Bee set
  (visitor=`TaggerCheckTGM:all`). Helper `tagger_check_tgm` added to
  `pgrapher/common/clus.jsonnet`.
- Validation: `sbnd/tgm-validation/` (summarize→npz, analyze XY/XZ/YZ + box,
  Bokeh `tgm_viewer.py`/`serve_tgm_viewer.sh`, README). 100-event corsika run
  (`mc_paths-10files.lst`): 966 tagged tracks, 1861 endpoints; **86.7% within
  5 cm of an FV box face (median 2.7 cm)** — endpoints outline the FV box as
  expected. Plots: `tgm-validation/tgm_views_100evt*.png`.
- Caveat: ~1% of endpoints (18, in 2 events) have absurd `x_t0cor` (±1e8 cm) from
  a bad-t0 flash match upstream (Q/L matching), not a TGM bug; filtered with
  `|x|<300` in the analysis.

## Known-good results (uploaded to BEE)

- Matching-only, 10 events (enable_downstream_pr=false): clean, valid zip.
  https://www.phy.bnl.gov/twister/bee/set/6e85d423-7ddd-492b-94a5-7694ba21f6db/event/list/
- Full chain, 1 event: clean.

## OPEN issue #3 (full chain, multi-event) — NOT fixed, stopped here

Running the FULL chain over the 13-event sample (`-n 10`) aborts on event 31
(7th processed): `TrackFitting::get_anode(0)` returns null for that event's
merged all-APA grouping (logs "TrackFitting: Error getting anode 0", non-fatal),
then track-fit geometry raises `ValueError "Anode is null"` and art aborts the
job (exit 1, not a segfault). First 6 events pass full patrec.

Event order (this sample): 2(full), 9(skip,no-main), 11(full), 14(full),
18(full), <one more full>, 31(ABORT).

Hypothesis: event 31's merged grouping is missing anode 0 (single-TPC event, or
the grouping anode-set / merge doesn't carry both anodes), and the track fitter
asks for the absent anode. Candidate fixes for later (see options in session):
make TaggerCheckNeutrino skip the event gracefully when an anode is missing
(guard get_anode-null like the no-main-cluster skip), or fix the merged-grouping
anode set.

UPDATE: the downstream crash is NONDETERMINISTIC. `-n 10` reached event 31 and
threw "Anode is null" (exit 1) after 6 passes incl. event 18; `-n 6` (same first
6 events, same order, same build) SEGFAULTED (exit 139) on event 18 in
TaggerCheckNeutrino with no anode-null. Same input/order/build giving different
failures => residual heap/memory unsafety in the downstream track-fitting beyond
the two fixes above (likely more dangling pointers / uninitialized reads). So a
clean full-patrec run can't be guaranteed by event-count selection; small
prefixes pass most of the time. Bulk/production: use enable_downstream_pr=false.

## OPEN issue #4 — steiner_pc required by tagger pattern-rec (data-dependent)

Full patrec on `rb_none/sp.root` event 2 (1st) aborts (exit 1, RuntimeError
"Steiner point cloud 'steiner_pc' not found"). That event's main cluster has <2
steiner terminals so CreateSteinerGraph returns an empty graph (no steiner_pc),
and TaggerCheckNeutrino's vertex examination requires it:
  build_steiner_kd_cache <- kd_steiner_knn <- PatternAlgorithms::
  examine_main_vertices <- determine_overall_main_vertex <- TaggerCheckNeutrino::visit
Not a Bee-dump issue (disabling the steiner/regular Bee point sets does NOT help;
the throw is in core pattern-rec).  Fix later: guard examine_main_vertices /
build_steiner_kd_cache to handle a cluster with no steiner_pc.  Data-dependent:
the no-SP-rebase dnnsp of this event produces the degenerate cluster; the
original 2025f-mc dnnsp events (1- and 5-event runs) passed full patrec.

## Practical recipe until #3 is fixed

Full patrec is good for the first ~6 events of this sample: run `-n 6` with
enable_downstream_pr=true for a valid full-chain mabc.zip; use
enable_downstream_pr=false for all-event bulk runs.

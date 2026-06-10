# PDHD clustering algorithm (baseline reference)

This documents the PDHD clustering chain as configured in
`cfg/pgrapher/experiment/pdhd/clus.jsonnet` (moved there from the local
`pdhd/clus.jsonnet`), driven by `pdhd/wct-clustering.jsonnet` /
`run_clus_evt.sh`.  It is the baseline to diff against when the algorithm is
updated.

## Data flow

```
imaging output (per APA):
  clusters-apa-apa{N}-ms-active.tar.gz   (3-view live blobs, post-deghosting)
  clusters-apa-apa{N}-ms-masked.tar.gz   (2-view blobs over dead regions)
        |  ClusterFileSource x2 per APA
        v
  per-APA stage (per_apa) -- internally runs the per-face stage on face 0/1
        |  x2 APAs per drift side
        v
  per-drift-group stage (per_group) -- {APA0,APA2} (drift -x) / {APA1,APA3} (drift +x)
        |  x2 groups
        v
  all-TPC stage (all_tpc) -> mabc-all-apa.zip (Bee) + trash-all-apa.tar.gz
```

Every stage is one `MultiAlgBlobClustering` (MABC) node executing a configured
`pipeline` of `Clustering*` components (from
`cfg/pgrapher/common/clus.jsonnet` `clustering_methods`).

## Geometry / point-cloud building

`DetectorVolumes` metadata (`dvm` in clus.jsonnet):

| key | meaning |
|---|---|
| overall FV | x ±357.985 cm, y [7.61, 606.0] cm, z [0.234, 462.297] cm (margins 2/2.5/3 cm) |
| a0f0pA / a2f0pA | active face, drift −x: FV_x [−357.985, −2.54] cm |
| a1f1pA / a3f1pA | active face, drift +x: FV_x [+2.54, +357.985] cm |
| a0f1pA, a1f0pA, ... | wall-facing (degenerate) faces: FV_x pinned at ±357.985 cm |
| per-face params | `drift_speed` 1.6 mm/µs, `tick` 0.5 µs, `nticks_live_slice` 4, `time_offset` (see T0 below) |

Per face, two `ClusterScopeFilter`s (live/dead) feed one `PointTreeBuilding`
with two `BlobSampler`s:

* **live** sampler: `strategy: ["stepped"]`, with `drift_speed` and
  `time_offset`; extras: wire_index, charge_val, charge_unc, wpid.
* **dead** sampler: `strategy: ["center"]`, all extras, no time offset.

The drift coordinate of every sampled point is

```
x = xorig + dirx * (t_slice + time_offset) * drift_speed
```

with `xorig` = collection-plane wire x of the (anode,face) and `dirx` =
`anodeface->dirx()` (BlobSampler::time2drift, `clus/src/BlobSampler.cxx`).

## Stage 1 — per-face pipeline (`clus_per_face`)

Coordinates: raw `["x","y","z"]`.  Methods in order
(jsonnet method → WCT component, key parameters):

| # | method | component | parameters / role |
|---|---|---|---|
| 1 | `pointed()` | ClusteringPointed | initial cluster formation from the point tree (live grouping) |
| 2 | `live_dead(dead_live_overlap_offset=2)` | ClusteringLiveDead | merge live clusters bridged by dead regions |
| 3 | `extend(flag=4, length_cut=60cm, num_try=0, length_2_cut=15cm, num_dead_try=1)` | ClusteringExtend | track-extension merge toward boundaries/dead regions |
| 4 | `regular("-one", length_cut=60cm, no extend)` | ClusteringRegular | general proximity/direction merge, long range |
| 5 | `regular("_two", length_cut=30cm, extend)` | ClusteringRegular | second pass, shorter range with extension enabled |
| 6 | `parallel_prolong(length_cut=35cm)` | ClusteringParallelProlong | merge parallel/prolonged track fragments |
| 7 | `close(length_cut=1.2cm)` | ClusteringClose | merge clusters in close contact |
| 8 | `extend_loop(num_try=3)` | ClusteringExtendLoop | iterate the extend family 3× |
| 9 | `connect1()` | ClusteringConnect1 | final connection pass |

(`separate(use_ctpc)` used to run between extend_loop and connect1; it moved
to the per-drift-group stage.  `isolated()` and `retile(...)` exist in the
toolbox but are commented out at this stage.)

Per-face MABC dumps `mabc-apa{N}-face{F}.zip` (Bee set "clustering",
`individual: true`) when run with `dump=true` — in the standard chain the
per-face nodes run with `dump=false` inside the per-APA stage.

## Stage 2 — per-APA (`clus_per_apa`)

The two per-face outputs are merged by `PointTreeMerging` (multiplicity 2),
then a per-APA MABC runs:

| # | method | component | role |
|---|---|---|---|
| 1 | `deghost()` | ClusteringDeghost | cross-face ghost removal (PDHD-specific; PDVD does not run it here) |
| 2 | `protect_overclustering()` | ClusteringProtectOverClustering | undo pathological merges |

Dump (when standalone): `mabc-apa{N}.zip`.

## Stage 3 — per-drift-group (`clus_per_group`)

`PointTreeMerging` over the 2 APAs of one drift side (`tolerate_missing:
true` — an absent APA input does not abort): group02 = {APA0, APA2} (drift
−x) and group13 = {APA1, APA3} (drift +x).  Coordinates: raw
`["x","y","z"]`.  Then:

| # | method | component | role |
|---|---|---|---|
| 1 | `extend(flag=4, 60cm/15cm, num_dead_try=1)` | ClusteringExtend | as stage 1, across the drift side |
| 2 | `regular("1", 60cm, no extend)` | ClusteringRegular | |
| 3 | `regular("2", 30cm, extend)` | ClusteringRegular | |
| 4 | `parallel_prolong(35cm)` | ClusteringParallelProlong | |
| 5 | `close(1.2cm)` | ClusteringClose | |
| 6 | `extend_loop(3)` | ClusteringExtendLoop | |
| 7 | `separate(use_ctpc=true)` | ClusteringSeparate | split over-merged clusters (moved here from the per-face stage) |
| 8 | `examine_x_boundary()` | ClusteringExamineXBoundary | split clusters at the drift-x fiducial boundary (newly enabled; the C++ accepts multi-wpid groupings whose wpids share identical FV_x metadata, true within one drift side) |
| 9 | `neutrino()` | ClusteringNeutrino | neutrino-candidate tagging/merge |
| 10 | `isolated()` | ClusteringIsolated | small→big isolated-cluster absorption |
| 11 | `examine_bundles()` | ClusteringExamineBundles | bundle examination/final merge |

Dump (when standalone): `mabc-group02.zip` / `mabc-group13.zip`.

## Stage 4 — all-TPC (`clus_all_tpc`)

`PointTreeMerging` over the 2 drift groups (`tolerate_missing: true`), then:

| # | method | component | coords | role |
|---|---|---|---|---|
| 1 | `switch_scope()` | ClusteringSwitchScope (T0Correction) | x→x_t0cor | computes the T0-corrected coordinate set (`["x_t0cor","y","z"]`) and applies a containment scope filter per (apa,face) volume |
| — | `cathode_connect(...)` | ClusteringCathodeConnect | x_t0cor | **commented out for now**: cathode-crossing connector with the SBND-tuned parameter set (cathode_x_cut=5cm, drift_cut=8cm, min_length_short=2cm, short_dir_len=25cm, conn_short_cut=30) as placeholder, plus `use_flash_t0=false` because PDHD has no flash matching (the default flash-coincidence gate would veto every pair).  PDHD's cathode is central at x=0 (the C++ default `cathode_x`); dimensions to be confirmed before enabling. |

A `retile` block (ClusteringRetile with per-face stepped samplers) is present
but commented out — the designated hook for re-tiling-based refinement.

Without a flash/T0 association, `x_t0cor` equals the (time_offset-shifted)
apparent x.

## Bee output (`mabc-all-apa.zip`)

Two point sets + dead areas, all with `use_config_rse: true` (runNo/subRunNo/
eventNo come from the `run_clus_evt.sh` TLAs):

| Bee instance | source |
|---|---|
| `clustering-group02` / `clustering-group13` | `name:"img"` hook: the live grouping **before** the all-APA pipeline (i.e. the per-APA clustering result), grouped by drift side via `apa_drift_groups` (APAs 0+2 / 1+3) |
| `clustering-global` | `name:"clustering"`: the **end** dump after the full all-APA pipeline |
| `channel-deadarea-group02/13` | `save_deadarea` with `dead_area_version: 2` (tpc=apa wrapper), grouped by `dead_apa_groups` |

Coordinates dumped are the uncorrected `["x","y","z"]` (x_t0cor would need a
flash-associated T0).  `run_bee_combined_evt.sh` merges these with the
imaging-stage `imaging-group02/13` instances (wirecell-img `bee-blobs` on the
`-ms-active` tarballs) into one upload.

## Where T0 enters (all zeroed as of 2026-06)

There is **no per-event T0 determination for PDHD**; the chain now uses
`time_offset = 0` everywhere, so a blob's x is its apparent drift position in
the readout frame (250 µs ≡ 40 cm at 1.6 mm/µs):

1. `cfg/pgrapher/experiment/pdhd/clus.jsonnet` — `time_offset` function
   parameter (default **0**; was −250 µs = readout-tick0-relative-to-trigger
   compensation).  Feeds both the live `BlobSampler` and the
   `DetectorVolumes` per-face metadata consumed by T0Correction.
2. `pdhd/wct-clustering.jsonnet` — `time_offset` TLA (default 0) passed
   through to the cfg file.
3. Imaging-Bee conversion `--t0` (wirecell-img bee-blobs):
   `run_bee_combined_evt.sh`, `run_bee_img_evt.sh`, `build_apa0_bee.sh`,
   `build_perapa_bee.sh`, `wct-img-2-bee.py` — all now `--t0 "0*us"` (were
   250 µs, the sign-flipped equivalent of the clustering offset) so imaging
   and clustering instances overlay in the combined display.
4. `pdhd/img_plot/preprocess_event.py` — `TIME_OFFSET_NS = 0.0` (viewer-side
   reproduction of time2drift; `img_viewer.py` reads the per-event meta, so
   caches preprocessed before the change stay self-consistent).

Restore a real value (clus jsonnet param + the bee `--t0`s, opposite signs)
once a T0 measurement exists.

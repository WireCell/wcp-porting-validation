# PDVD clustering: current algorithm and per-pass scope capabilities

This documents (1) the clustering pipeline PDVD currently runs and (2) for
every clustering pass available in `clus/src/`, whether it can operate on a
**single APA-face**, a **single APA (both faces)**, or **all APAs** — the
reference for updating the PDVD clustering algorithm.

Configuration: `cfg/pgrapher/experiment/protodunevd/clus.jsonnet` (library,
in the toolkit cfg tree), driven by `pdvd/wct-clustering.jsonnet`.
Companion docs: [clus-workflow.md](clus-workflow.md) (graph topology, RSE,
dead channels), [clustering-boundary-merge.md](clustering-boundary-merge.md)
(the no-T0 scope-filter exclusion this config now relaxes).

## 1. The current PDVD clustering algorithm

Three hierarchical stages.  Each stage is one `MultiAlgBlobClustering`
(MABC) instance running an ordered pipeline of passes over a point-tree
grouping.

### Stage 1 — per APA-face (16 instances: 8 anodes x 2 faces)

Input: live + dead imaging clusters, scope-filtered to one face
(`ClusterScopeFilter`), sampled into point trees (`PointTreeBuilding` with
the stepped live sampler and center dead sampler).  Coordinates:
`["x","y","z"]`; DetectorVolumes: that face only.

| # | pass | key parameters |
|---|------|----------------|
| 1 | `pointed` | prune empty blobs/clusters |
| 2 | `live_dead` | dead_live_overlap_offset=2 — merge live clusters bridged by dead regions |
| 3 | `extend` | flag=4, length_cut=60cm, num_dead_try=1 — extend near dead regions |
| 4 | `regular` ("-one") | length_cut=60cm, no extend — first distance/angle merge |
| 5 | `regular` ("_two") | length_cut=30cm, with extend — second merge |
| 6 | `parallel_prolong` | length_cut=35cm — parallel/prolonged track merge |
| 7 | `close` | length_cut=1.2cm — tiny-gap merge |
| 8 | `extend_loop` | num_try=3 — iterative extension (4 internal passes per try) |
| 9 | `separate` | use_ctpc=true — split over-merged clusters |
| 10 | `connect1` | isochronous-aware final connection |

Bee output: `mabc-anode{N}-face{F}.zip`.

### Stage 2 — per APA (8 instances)

`PointTreeMerging` (multiplicity 2) merges the two faces' point trees.
DetectorVolumes: both faces of the anode.  Single pass:

| # | pass | purpose |
|---|------|---------|
| 1 | `protect_overclustering` | split clusters whose blob connectivity does not support the merge |

Note: there is **no cross-face merge pass** at this stage — the two faces'
clusters coexist in one grouping but are only ever merged later, in the
all-APA stage.  Bee output: `mabc-anode{N}.zip`.

### Stage 3 — all APAs (1 instance)

`PointTreeMerging` (multiplicity 8) merges all per-APA trees.
DetectorVolumes: full detector (all 8 anodes + `overall` cryostat box).

| # | pass | coords | key parameters |
|---|------|--------|----------------|
| 1 | `switch_scope` | x,y,z → x_t0cor,y,z | T0Correction; registers the corrected scope and scope-filters clusters by volume containment (see §3) |
| 2 | `extend` | x_t0cor | flag=4, length_cut=60cm, num_dead_try=1 |
| 3 | `regular` ("1") | x_t0cor | length_cut=60cm, no extend |
| 4 | `regular` ("2") | x_t0cor | length_cut=30cm, with extend |
| 5 | `parallel_prolong` | x_t0cor | length_cut=35cm |
| 6 | `close` | x_t0cor | length_cut=1.2cm |
| 7 | `extend_loop` | x_t0cor | num_try=3 |
| 8 | `separate` | x_t0cor | use_ctpc=true |
| 9 | `neutrino` | x_t0cor | neutrino-interaction pattern recognition |
| 10 | `isolated` | x_t0cor | isolated small-cluster classification/merge |
| 11 | `examine_bundles` | x_t0cor | graph-based bundle re-examination |

With no flash matching in PDVD every `cluster_t0` is 0, so `x_t0cor == x`
numerically; the corrected scope still matters because it carries the
containment scope-filter.  Bee output: `mabc-all-apa.zip` containing
`clustering-group0123` / `clustering-group4567` (the **per-anode** result,
dumped from the merged input *before* the all-APA passes via the
`name:"img"` bee points set) and `clustering-global` (the final result).

## 2. Scope capability of every clustering pass

Scope legend — **face**: single APA-face grouping; **APA**: one anode, both
faces in one grouping; **all**: full-detector grouping.  "PDVD use" = where
the current pipeline runs it.  Evidence cites `clus/src/`.

| pass | face | APA | all | PDVD use | evidence / limiting mechanism |
|---|---|---|---|---|---|
| `pointed` | yes | yes | yes | stage 1 | structure-only pruning, no geometry (clustering_pointed.cxx) |
| `live_dead` | yes | untested | untested | stage 1 | per-wpid wire-geometry maps built for all wpids present, but live↔dead bridging is a per-face concept (dead regions are per-face); only exercised per-face |
| `extend` | yes | yes* | yes | stages 1+3 | pairwise distance/Hough logic, per-wpid lookups; scope-agnostic |
| `regular` | yes | yes* | yes | stages 1+3 | pairwise closest-point + wire-angle via `get_wireplaneid()`; scope-agnostic |
| `parallel_prolong` | yes | yes* | yes | stages 1+3 | as `regular`; NB hard-coded drift axis (1,0,0) — fine for PDVD/x-drift |
| `close` | yes | yes* | yes | stages 1+3 | distance/direction only, no DetectorVolumes at all (clustering_close.cxx) |
| `extend_loop` | yes | yes* | yes | stages 1+3 | wrapper looping `extend` 4 ways per iteration |
| `separate` | yes | yes | yes | stages 1+3 | `select_scope_fv()` (clustering_separate.cxx:27-94) picks per-APA FV for single-APA DV, cryostat `overall` FV for multi-APA — explicitly scope-aware |
| `connect1` | yes | **no** | **no** | stage 1 only | hard assertion: `wpids().size() > 1` → ValueError (clustering_connect.cxx:69-83, "This is for only one APA/face") |
| `deghost` | yes | yes | **no** | not used | "all faces within a single APA"; `apas.size() > 1` → ValueError (clustering_deghost.cxx:118-174); needs CTPC point clouds when use_ctpc |
| `examine_x_boundary` | yes | **no** | **no** | not used (commented out) | hard assertion: `wpids().size() > 1` → ValueError (clustering_examine_x_boundary.cxx:42-54); reads FV of the single wpid |
| `protect_overclustering` | yes | yes | yes | stage 2 | intra-cluster blob-graph analysis; reads per-wpid `nticks_live_slice` for every wpid present |
| `switch_scope` | yes | yes | yes | stage 3 | applies a named IPCTransform (T0Correction) per cluster; scope-independent mechanics, but only meaningful where a corrected scope is wanted (post-merge, all-APA) |
| `neutrino` | yes | yes | yes | stage 3 | "handle all APA/Face" (clustering_neutrino.cxx:61); builds per-(apa,face) geometry maps; uses scope-selected fiducial volume |
| `isolated` | yes | yes | yes | stage 3 | "Handle all APA/Faces" (clustering_isolated.cxx:75) |
| `examine_bundles` | yes | yes | yes | stage 3 | "All APA Faces" (clustering_examine_bundles.cxx:79); needs CTPC when use_ctpc |
| `cathode_connect` | n/a | (yes) | (yes) | not used | SBND-specific cathode-crosser connector: pairs clusters in *different* APAs whose ends meet at a configured shared-cathode x.  PDVD's two drift volumes meet at the cathode plane (x≈±25.4mm), so the *mechanism* could in principle be retargeted, but all cuts are SBND-tuned — treat as SBND-only until validated |
| `retile` | yes | yes | yes | not used (commented out) | re-tiles clusters through an IPCTreeMutate; defaults to and warns unless the T0-corrected scope is used (clustering_retile.cxx:67-81) |
| `ctpointcloud` | — | — | — | not used | test/diagnostic only (clustering_ctpointcloud.cxx:51) |

`yes*` = no scope-limiting code found (same pairwise logic PDVD already runs
per-face and all-APA), but the per-APA two-faces-in-one-grouping case is not
exercised by any current PDVD stage — validate before relying on it.

## 3. T0 handling and the containment scope filter (2026-06 changes)

Two knobs in `cfg/pgrapher/experiment/protodunevd/clus.jsonnet` (also exposed
as TLAs of `pdvd/wct-clustering.jsonnet`):

- **`time_offset` (default 0, was -250us)** — the live BlobSampler converts
  slice time to drift position as
  `x = xorig + dirx*(t_slice + time_offset)*drift_speed`.  The readout tick0
  sits 250us before the trigger, so -250us pins *trigger-time* activity to
  its true x — i.e. it presets T0 = trigger for every cluster.  PDVD has no
  per-event T0, and the -250us pushed early-window activity up to 40cm
  *behind* the anode planes (the "clusters broken outside the detector box"
  seen in Bee) while the dead-area sampler carried no such offset.  With 0,
  no activity maps behind the anode and live/dead samplers agree.  Apparent
  x of late activity can still extend past the cathode into the opposite
  drift volume — with a 3000us readout window and ~2100us full drift this is
  unavoidable physics until a real T0 exists.  Restore a measured value here
  once PDVD T0 determination is available.

- **`relax_containment_filter` (default true; PCTransformSet option, C++
  default false)** — `switch_scope`'s T0Correction filter normally accepts a
  point only if it is contained by its *own* (apa,face) volume; clusters with
  no accepted point are excluded from **all** subsequent all-APA merge passes
  (the 17.5% / 66.6% point exclusion documented in
  [clustering-boundary-merge.md](clustering-boundary-merge.md)).  Relaxed, a
  point passes if contained by *any* sensitive volume, so out-of-time
  clusters keep participating in cross-CRP merging.  Other detectors are
  unaffected (C++ default off; the key is set only in the PDVD config).

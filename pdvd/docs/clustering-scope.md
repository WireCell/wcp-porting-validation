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

## 1. The current PDVD clustering algorithm (4 stages, 2026-06 reorg)

Four hierarchical stages (same structure as the PDHD 4-stage reorg, toolkit
`fe1927e3`).  Each stage is one `MultiAlgBlobClustering` (MABC) instance
running an ordered pipeline of passes over a point-tree grouping.

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
| 9 | `connect1` | isochronous-aware final connection |

`separate` moved to stage 3 (the per-drift-group scope sees the whole track
before splitting it).  Bee output: `mabc-anode{N}-face{F}.zip`.

### Stage 2 — per APA, two faces (8 instances)

`PointTreeMerging` (multiplicity 2) merges the two faces' point trees.
DetectorVolumes: both faces of the anode.

| # | pass | purpose |
|---|------|---------|
| 1 | `deghost` | remove ghost clusters (multi-face within one APA; use_ctpc=true) |
| 2 | `protect_overclustering` | split clusters whose blob connectivity does not support the merge |

Note: there is **no cross-face merge pass** at this stage — the two faces'
clusters coexist in one grouping but are only ever merged in stage 3.  Bee
output: `mabc-anode{N}.zip`.

### Stage 3 — per drift-side group (2 instances: anodes 0-3, anodes 4-7)

`PointTreeMerging` (multiplicity 4, tolerate_missing) merges the per-APA
trees of one drift side.  DetectorVolumes: the group's 4 anodes (8 wpids,
all sharing the same FV_x metadata).  Coordinates: raw `["x","y","z"]` —
with no per-event T0, `x_t0cor == x`, so running the merge family before
`switch_scope` is equivalent.

| # | pass | key parameters |
|---|------|----------------|
| 1 | `extend` | flag=4, length_cut=60cm, num_dead_try=1 |
| 2 | `regular` ("1") | length_cut=60cm, no extend |
| 3 | `regular` ("2") | length_cut=30cm, with extend |
| 4 | `parallel_prolong` | length_cut=35cm |
| 5 | `close` | length_cut=1.2cm |
| 6 | `extend_loop` | num_try=3 |
| 7 | `separate` | use_ctpc=true |
| 8 | `connect1` | **added 2026-06-10** — MicroBooNE order after separate; reconnects dashed-line fragments (e.g. drift-direction tracks) at group scope; allow_mixed_faces=true |
| 9 | `deghost` | **added 2026-06-10** — group-scope ghost removal; allow_mixed_faces=true, empty_view_unique=true (required at group scope, see clus/docs/clustering-group-connect1-deghost.md) |
| 10 | `examine_x_boundary` | multi-wpid since toolkit `f68e5f6a` (all wpids must share FV_x — true within one drift side) |
| 11 | `neutrino` | neutrino-interaction pattern recognition |
| 12 | `isolated` | isolated small-cluster classification/merge |
| 13 | ~~`examine_bundles`~~ | **DISABLED 2026-06-14** (commented out) — see note below |

**`examine_bundles` disabled at this stage (2026-06-14).** It only rewrites the
`isolated`/`perblob` array (never cluster membership at this stage,
`use_flash_t0=false`), and PDVD runs no Q/L matching, so that array has no
consumer — it is dropped at `switch_scope` (stage 4, see note in §Stage 4) either
way. Removing the pass therefore leaves the cluster grouping **byte-identical**
(verified A/B on run 039252 evt 0: `mabc-all-apa.zip` member content unchanged).
This mirrors the PDHD change, where the same pass was disabled to fix a Q/L
boundary-flag bug (it split crossing-cosmic tracks into a main spine + associated
continuation, and `QLMatching` flagged the main only); see
`cfg/.../pdhd/clus.jsonnet` and `pdhd/docs/clustering-algorithm.md`. Re-enable in
the all-TPC stage if the main/associated structure is ever needed downstream.

Bee output: `mabc-group0123.zip` / `mabc-group4567.zip`.

### Stage 4 — all-TPC (1 instance)

`PointTreeMerging` (multiplicity 2, tolerate_missing) merges the two
drift-side groups.  DetectorVolumes: full detector (all 8 anodes +
`overall` cryostat box).

| # | pass | coords | key parameters |
|---|------|--------|----------------|
| 1 | `switch_scope` | x,y,z → x_t0cor,y,z | T0Correction; registers the corrected scope and scope-filters clusters by volume containment (see §3) |
| 2 | `cathode_connect` | x_t0cor | **enabled 2026-06-09** — SBND-tuned cuts (cathode_x=0 default, cathode_x_cut=5cm, drift_cut=8cm, use_flash_t0=false since PDVD has no flash matching); the only cross-drift-side merge pass |

With no flash matching in PDVD every `cluster_t0` is 0, so `x_t0cor == x`
numerically; the corrected scope still matters because it carries the
containment scope-filter.  `cathode_connect` is the only merge pass here, so
the global cluster count equals the sum of the two stage-3 group counts
except where it connects a cathode crosser.  Without a per-event T0 only
near-trigger-time crossers qualify: a crosser's cathode tips appear at
apparent |x| ≈ t0·v_drift on *opposite* sides of x=0, so out-of-time cosmics
fail the 5 cm cathode window while a trigger-coincident crosser's tips sit
at the ±2.54 cm sensitive-volume edges and pass.  (On the two locally imaged
events — 039252/298567, 039253/49686 — it fires zero times; global = sum
held.)  Stage-3 prerequisite: `examine_x_boundary(allow_mixed_faces=true)` —
the same-face check added for PDHD drift-side groups (9a41546a) would
otherwise raise on PDVD's per-group groupings, whose 8 wpids legitimately
mix faces (an anode's faces are the y-halves of one CRP, identical FV_x).  Note: `switch_scope` destroys and recreates every cluster,
which drops the per-cluster `isolated`/`perblob` array produced by stage
3's `isolated` (since 2026-06-14 `examine_bundles` is disabled at stage 3, so
the array is `isolated`'s, with no `-1` main) — accepted, as nothing in the
PDVD chain reads it; SBND re-runs `examine_bundles` after `switch_scope` when
that array is needed downstream.

Bee output: `mabc-all-apa.zip` containing `clustering-group0123` /
`clustering-group4567` (the **stage-3 per-drift-group output**, dumped from
the merged input *before* the all-TPC passes via the `name:"img"` bee points
set) and `clustering-global` (the final result).

## 2. Scope capability of every clustering pass

Scope legend — **face**: single APA-face grouping; **APA**: one anode, both
faces in one grouping; **group**: 4 anodes of one drift side; **all**:
full-detector grouping.  "PDVD use" = where the current pipeline runs it.
Evidence cites `clus/src/`.

| pass | face | APA | group/all | PDVD use | evidence / limiting mechanism |
|---|---|---|---|---|---|
| `pointed` | yes | yes | yes | stage 1 | structure-only pruning, no geometry (clustering_pointed.cxx) |
| `live_dead` | yes | untested | untested | stage 1 | per-wpid wire-geometry maps built for all wpids present, but live↔dead bridging is a per-face concept (dead regions are per-face); only exercised per-face |
| `extend` | yes | yes* | yes | stages 1+3 | pairwise distance/Hough logic, per-wpid lookups; scope-agnostic |
| `regular` | yes | yes* | yes | stages 1+3 | pairwise closest-point + wire-angle via `get_wireplaneid()`; scope-agnostic |
| `parallel_prolong` | yes | yes* | yes | stages 1+3 | as `regular`; NB hard-coded drift axis (1,0,0) — fine for PDVD/x-drift |
| `close` | yes | yes* | yes | stages 1+3 | distance/direction only, no DetectorVolumes at all (clustering_close.cxx) |
| `extend_loop` | yes | yes* | yes | stages 1+3 | wrapper looping `extend` 4 ways per iteration |
| `separate` | yes | yes | yes | stage 3 | `select_scope_fv()` (clustering_separate.cxx:27-94) picks per-APA FV for single-APA DV, cryostat `overall` FV for multi-APA — explicitly scope-aware |
| `connect1` | yes | yes | group | stages 1+3 | generalized 2026-06-10: multi-wpid drift groups validated via `validate_drift_group()` (identical FV_x; same face unless allow_mixed_faces).  Per-point wpid routing for dead maps + 2D queries, per-cluster OR'd prolonged test over occupied volumes, extrapolations re-bucketed per volume (`make_points_linear_extrapolation` seed_wpid).  See clus/docs/clustering-group-connect1-deghost.md |
| `deghost` | yes | yes | group | stages 2+3 | generalized 2026-06-10: multi-APA accepted when LIVE wpids form one drift volume (`validate_drift_group()`, allow_mixed_faces for PDVD).  Group instances REQUIRE empty_view_unique=true: an empty per-volume 2D index returns −1 which reads as overlap, else the longest cluster of each unseeded volume is wrongly destroyed.  Needs CTPC point clouds when use_ctpc |
| `examine_x_boundary` | yes | yes | group | stage 3 | multi-wpid since `f68e5f6a`: all wpids must share identical FV_x metadata or ValueError (clustering_examine_x_boundary.cxx:74-87) — true within a PDVD drift side, false across the cathode, so group scope is its ceiling |
| `protect_overclustering` | yes | yes | yes | stage 2 | intra-cluster blob-graph analysis; reads per-wpid `nticks_live_slice` for every wpid present |
| `switch_scope` | yes | yes | yes | stage 4 | applies a named IPCTransform (T0Correction) per cluster; scope-independent mechanics, but only meaningful where a corrected scope is wanted (post-merge) |
| `neutrino` | yes | yes | yes | stage 3 | "handle all APA/Face" (clustering_neutrino.cxx:61); builds per-(apa,face) geometry maps; uses scope-selected fiducial volume |
| `isolated` | yes | yes | yes | stage 3 | "Handle all APA/Faces" (clustering_isolated.cxx:75) |
| `examine_bundles` | yes | yes | yes | stage 3 | "All APA Faces" (clustering_examine_bundles.cxx:79); needs CTPC when use_ctpc |
| `cathode_connect` | n/a | no | all | stage 4 (enabled 2026-06-09) | cathode-crosser connector: pairs clusters in *different* APAs whose ends meet at a configured shared-cathode x (clustering_cathode_connect.cxx; requires wpid.apa() to differ).  PDVD's two drift volumes meet at x≈±25.4mm, matching the default cathode_x=0; `use_flash_t0=false` (`f68e5f6a`) disables the flash-coincidence gate PDVD cannot satisfy.  Cuts are SBND-tuned placeholders |
| `retile` | yes | yes | yes | not used (commented out) | re-tiles clusters through an IPCTreeMutate; defaults to and warns unless the T0-corrected scope is used (clustering_retile.cxx:67-81) |
| `ctpointcloud` | — | — | — | not used | test/diagnostic only (clustering_ctpointcloud.cxx:51) |

`yes*` = no scope-limiting code found (same pairwise logic PDVD already runs
per-face and per-group), but the per-APA two-faces-in-one-grouping case is
not exercised by any current PDVD merge stage — validate before relying on
it.

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
  [clustering-boundary-merge.md](clustering-boundary-merge.md)).  Relaxed,
  the filter is **disabled entirely**: every point passes, every cluster
  participates in cross-CRP merging, and `switch_scope` never splits a
  cluster on filter results.  An earlier relaxed form (accept if contained by
  *any* sensitive volume) proved insufficient: with no T0 the apparent x is
  unreliable in both directions, and clusters can sit **entirely outside all
  sensitive volumes** — early activity in the band between the anode-face
  boundary (|x| = 335.835 cm) and the wire planes (|x| = 341.55 cm), late
  activity in the cathode gap (|x| < 2.54 cm).  Run 039324 evt 0: 25 of 81
  global clusters (10 anode-band + 15 cathode-band) were still excluded by
  the any-volume form — e.g. a 92-point track tip at x ∈ [−341.6, −336.1]
  left 0.57 cm from its parent track but never merged; with the filter
  disabled the global cluster count drops 81 → 51.  Other detectors are
  unaffected (C++ default off; the key is set only in the PDVD config).

  Under the 4-stage reorg the merge family runs *before* `switch_scope`
  (stage 3, raw coords), so the scope filter can no longer exclude clusters
  from those passes; relaxing it still matters so that `switch_scope` does
  not split clusters on filter results and so any post-correction pass
  (`cathode_connect`, now enabled) sees every cluster.

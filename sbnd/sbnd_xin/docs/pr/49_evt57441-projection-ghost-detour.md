# doc pr/49 — evt 57441: trajectory-fit detour is a cross-track projection ghost

**Status: investigation only. No code or config changed.** This document
records findings and a *proposed* (not implemented) fix design for a
trajectory-fit distortion the owner spotted by eye. The multi-track fitter
(`TrackFitting`) is fundamental to the whole PR chain, so per the operating
manual (§5 rule 1/4) nothing here is applied without its own default-OFF
knob and full byte-identical gate round.

## Repro block

All evidence below comes from an already-run, final-binary arm — no rebuild
was needed and none was done.

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-pr48-on1kc/pr_evt57441
ls -la ../../../../local/lib/libWireCellClus.so   # 2026-08-07 19:57 -- predates every file here (freshness OK)
python3 - <<'PY'
import zipfile, json, math
import numpy as np
from collections import Counter

z = zipfile.ZipFile('mabc-pr.zip')
tf = json.loads(z.read('data/0/0-track_fit-global.json'))      # fitted trajectory (all segments/vertices)
cg = json.loads(z.read('data/0/0-clustering-global.json'))     # raw 3D image (all clusters)
d  = json.load(open('calib-pr-evt57441.json'))                 # richer per-segment/per-plane dump

# 1. fit trajectory around the owner's coordinates, cluster 20 (real_cluster_id 20000)
X, Y, Z, R = map(np.array, (tf['x'], tf['y'], tf['z'], tf['real_cluster_id']))
P = (-48.1, -144.0, 409.6)

# 2. per-plane 2D charge-cell ownership near the fit, from the calib-pr dump
proj = {(p['apa'], p['face'], p['plane']): p for p in d['proj']}
seg  = [s for s in d['segments'] if s['id'] == 20000][0]['points']

# 3. true 3D position of the contaminating cluster (real_cluster_id 13)
m13 = R == 13
print('cluster 13 min 3D distance to detour region:',
      np.sqrt((X[m13]+45.05)**2 + (Y[m13]+146.3)**2 + (Z[m13]-411.0)**2).min(), 'cm')
PY
```

The full interactive session (fit-vs-image comparison, per-plane cell
ownership table, cluster-13 3D extent, wire-range pad check) is reproduced
step by step in **Root cause** below; every number quoted there was obtained
this way, against this arm.

## Symptom

Owner: run 18255 evt 57441, `(x, y, z) = (-48.1, -144.0, 409.6)` — "the track
trajectory fitted is not consistent with the original 3D image... in one of
the projection views (measurement), at that location, there is an overlap
between another 3D track with this track."

Cluster 20 (`is_main_cluster`, particle_id 13/muon-like, one segment,
102.7 cm long, 172 fit points) is the reconstructed track. Its fitted
trajectory (`track_fit-global.json`, `real_cluster_id == 20000`) runs in a
clean, near-straight 0.60 cm step from fit index ~95 to ~104, then visibly
kinks/wobbles from index ~105 through ~114 before straightening out again —
exactly bracketing the owner's coordinates (nearest fit point is index 117,
1.13 cm away; the kink itself is centered a few points earlier):

| idx | x | y | z | step (cm) |
|---|---|---|---|---|
| 104 | -44.20 | -149.19 | 410.98 | 0.60 |
| 105 | -44.59 | -148.74 | 410.92 | 0.60 |
| 106 | -44.95 | -148.26 | 410.91 | 0.60 |
| 107 | -45.31 | -147.78 | 410.90 | 0.60 |
| 108 | -45.11 | -147.22 | 411.00 | 0.60 (y jumps back) |
| 109 | -45.07 | -146.62 | 411.00 | 0.60 |
| 110 | -45.03 | -146.02 | 410.99 | 0.60 |
| 111 | -45.33 | -145.53 | 410.86 | 0.60 |
| 112 | -45.88 | -145.27 | 410.87 | 0.60 |
| 113 | -46.37 | -144.94 | 410.80 | 0.60 |
| 114 | -46.94 | -144.77 | 410.72 | 0.60 |

Step size is pinned at 0.60 cm by the fixed sampling in
`organize_segments_path` (expected — that's how the trajectory is resampled
between passes), so the "detour" is not a step-size artifact; the fitted
*positions* wander off the smooth line the neighboring points define,
particularly in `y`.

Crucially, the cluster's own raw 3D image (`clustering-global.json`,
`real_cluster_id == 20`) does **not** show this wander: the nearest image
points to each of these fit indices stay within 3.7 cm of the fit at every
index in this range, and their own local scatter is smooth. So the *image*
is fine; the *fit* is what's distorted — consistent with the owner's
description.

## Root cause

**Short answer: yes, exactly what you suspected — a second, physically
unrelated track (`real_cluster_id 13`, 163 cm away in 3D) supplies real
measured charge that aliases with cluster 20's own track in the V-plane
(collection-view) projection only, over fit indices ~108-113. U and W are
clean. The fitter has no way to tell the two apart because its charge
lookup is ownership-blind by design, in both codebases.**

### 1. The contamination is real, localized to the V-plane, and traced to a specific foreign cluster

`calib-pr-evt57441.json`'s `proj` array holds, per `(apa, face, plane)`, the
full set of `(wire, slice, charge, cluster_id)` cells the fitter's charge
lookup produced for this event. Cross-referencing against the fitted
segment's own per-point projections (`pu`, `pv`, `pw`, `pt`, all in
`segments[0].points`) at fit indices 108-113:

| idx | plane U cells (cluster) | plane V cells (cluster) | plane W cells (cluster) |
|---|---|---|---|
| 107 | 12/12 → cid 20 | 11/11 → cid 20 | all → cid 20 |
| 108 | 5/5 → cid 20 | 2× cid 20, **2× cid 13** | all → cid 20 |
| 109 | 1/1 → cid 20 | 1× cid 20, **5× cid 13** | all → cid 20 |
| 110 | 0 cells | **10/10 → cid 13** | all → cid 20 |
| 111 | 0 cells | **11/11 → cid 13** | all → cid 20 |
| 113 | 2/2 → cid 20 | 1× cid 20, **12× cid 13** | all → cid 20 |

Every U- and W-plane cell in this window belongs to cluster 20. Every
contaminated cell is in the **V plane** (plane index 1), and every one of
those foreign cells belongs to `cluster_id == 13`.

### 2. Cluster 13 is a real, distant, distinct track — not a clustering bug

`clustering-global.json`'s raw image points for `real_cluster_id == 13`
(3947 points) occupy `z ∈ [155, 320] cm`, `y ∈ [-193, -18] cm`, versus
cluster 20's `z ∈ [408, 417] cm`, `y ∈ [-196, -119] cm`. The minimum 3D
distance from any cluster-13 point to the detour region
(`(-45.05, -146.3, 411.0)`) is **163 cm**. These are two genuinely separate
physical tracks; clustering correctly kept them apart. The overlap is purely
a projective coincidence: in one of the three 2D views, the two tracks'
`(wire, time)` coordinates happen to land on top of each other over an ~8 cm
arc, while the other two views (which use different wire angles and
therefore different linear combinations of `y`/`z`) resolve them cleanly.
This is a classic "ghost" ambiguity in wire-chamber reconstruction: three 2D
projections of two 3D tracks are, in general, four apparent 3D positions,
and nothing local to one view can tell the real combination from the ghost.

### 3. It is not the toolkit's bounding-box padding

`TrackFitting::prepare_data` (`clus/src/TrackFitting.cxx:728-816`, in
`toolkit/`) pads the cluster's own `(u,v,w,t)` extent by `±5` wires and
`±20` ticks before querying the grouping's charge map (`:764-767`):

```cpp
u_min -= 5; v_min -= 5; w_min -= 5;   u_max += 5; v_max += 5; w_max += 5;
t_min -= 20;  t_max += 20;
```

The prototype's equivalent, `prepare_data`
(`prototype_base/pid/src/PR3DCluster_trajectory_fit.h:1861-1901`), uses the
cluster's own projected-hit `min`/`max` **with no pad**. This looked like a
plausible toolkit-only divergence worth checking — but it is not the cause
here: the contaminating V-plane wires (834-841) fall well inside cluster
20's own *unpadded* V-wire range, `[703, 909]` (checked directly against the
`proj` block's own `cluster_id == 20` entries). The aliasing is intrinsic to
where cluster 20's real track sits in the V view, not an artifact of the pad
reaching outside the cluster's true footprint. (The pad is still a real,
documented port divergence — see "Not the fix" below — just not the cause
of this event.)

### 4. Mechanism, traced through both codebases

The chain that lets this happen, with the toolkit's file:line and the
prototype's structurally identical counterpart:

- **`prepare_data`** builds a flat, **cluster-ownership-blind** charge map
  keyed only by `(apa, time, channel)`:
  - toolkit: `TrackFitting::prepare_data`, `clus/src/TrackFitting.cxx:728-816`,
    via `Facade::Grouping::get_overlap_good_ch_charge`
    (`clus/src/Facade_Grouping.cxx:882-920`) — a rectangular scan of the
    *global* per-plane CTPC point cloud. Whatever charge was measured in
    that `(wire, time)` cell is returned, independent of which cluster's
    blobs produced it.
  - prototype: `WCPPID::PR3DCluster::prepare_data`,
    `prototype_base/pid/src/PR3DCluster_trajectory_fit.h:1804-1931`, calling
    `ct_point_cloud.get_overlap_good_ch_charge(...)` — same shape, same
    ownership blindness, over the cluster's own (unpadded) hit range.

- **`form_point_association`** generates *candidate* `(wire, time)` cells
  near a fit point purely geometrically: nearest own-cluster point,
  `nlevel`-hop neighbors on the cluster's own blob/Steiner graph, a window
  sized by `dis_cut` (`clus/src/TrackFitting.cxx:2109-2557`; prototype
  `pid/src/PR3DCluster_multi_track_fitting.h`, documented in
  `pid/docs/track_fitting/form_point_association.md`). It has no way to know
  a candidate cell's charge came from a different physical track — the
  charge value for that cell is whatever `prepare_data`'s ownership-blind
  map returned.

- **`examine_point_association`** applies a charge-magnitude cut and a
  dead-plane rescue (`clus/src/TrackFitting.cxx:2709-3068`; prototype
  `pid/docs/track_fitting/examine_point_association.md`), but nothing
  cross-plane or cross-cluster. A cell that passes the charge cut is kept
  regardless of whether the other two planes corroborate it.

- **The one defense that exists is structurally unable to help here.**
  `update_association` / the `flag_exclusion` knob
  (`clus/src/TrackFitting.cxx:2563-2680`; prototype
  `pid/src/PR3DCluster_multi_track_fitting.h:970-1075`, both documented in
  `pid/docs/track_fitting/update_association.md`) strips a candidate cell if
  it is closer, in that projection, to some *other segment* than to the
  segment being fit. But the comparison set (`all_segments` / `segments`) is
  built only from the current PR pattern's own segment graph:
  - toolkit: `TrackFitting::form_map_graph` collects `segments` from
    `get_segment_edges()` over `m_graph` — this pattern's own boost graph
    (`clus/src/TrackFitting.cxx:3122-3129`).
  - prototype: `form_map_multi_segments` filters
    `map_segment_vertices` by `get_cluster_id() != cluster_id` before
    building its `segments` vector
    (`pid/src/PR3DCluster_multi_track_fitting.h:729-746`).

  **Cluster 13 has zero segments anywhere in this event's PR output** —
  `calib-pr-evt57441.json`'s `segments` array lists only cluster ids
  20/25/44-51 (cluster 20's own track plus small shower stubs). Cluster 13
  is not part of this neutrino candidate's pattern at all, so it can never
  appear in `update_association`'s comparison set — turning
  `m_fit_exclusion` on (it currently defaults **false**, doc pr/30 §11 P1/F1
  — see `clus/inc/WireCellClus/NeutrinoPatternBase.h:292-316`) would not
  change this event's outcome. This was checked directly against the
  mechanism, not assumed.

### 5. Sharper still: the association window overshoots cluster 20's own blob footprint

Section 3 ruled out the `prepare_data` bounding-box pad as the entry point.
A finer look at exactly where the contamination enters narrows it further.
Restricting the `proj` block's V-plane cells to the ones each cluster's
*own* fit actually claimed (`cluster_id == 20` vs. `cluster_id == 13`) shows
the two sets are **completely disjoint — zero cells in common** — even
though they sit inside the same coarse wire-index bounding box:

```
wire 830: cid20 slices 604-607
wire 834: cid20 slices 599-603   |   cid13 slices 607-612
wire 837: cid20 slices 597-599   |   cid13 slices 603-607
wire 839: cid20 slices 595-597   |   cid13 slices 598-604   (one tick apart)
wire 840: cid20 slice  595       |   cid13 slices 596-603
```

The two tracks' real footprints are **diagonally adjacent, not overlapping**
— cluster 20's own cells end at wire 839/slice 597, cluster 13's begin at
wire 839/slice 598, a one-tick gap. So the contamination does not enter
because cluster 20's own track genuinely occupies those cells; it enters
because `form_point_association`'s candidate-cell search window (`dis_cut`,
the `nlevel`-hop neighbor expansion, `clus/src/TrackFitting.cxx:2109-2557`)
**overshoots cluster 20's own blob footprint by a few wires/ticks** into
cluster 13's immediately adjacent, exclusive territory. "The charge lookup
is ownership-blind" (§4) is the structural reason this is *possible*;
"the association window overshoots the true footprint by 1-3 cells" is the
precise reason it *happens on this event*. This distinction is the basis for
the blob-coverage fix design in the Fix section below.

(Caveat on this evidence, carried into the Fix section: `cluster_id` here is
**post-fit fit-claim provenance** — which cells each cluster's finished fit
happened to use — not a direct query of `Facade::Blob` geometry. It is
strong proxy evidence, not proof; see the Fix section for the direct check
that still needs doing.)

### 6. Answering the two questions asked directly

- **"Does this have something to do with retiling?" — No.**
  `CreateSteinerGraph::mutate`'s retiled scratch cluster only donates
  `steiner_pc`/`steiner_graph` back to the real cluster
  (`clus/src/CreateSteinerGraph.cxx:292-296`) and is destroyed immediately
  after (`:313`). `TrackFitting` never reads retiled blobs — it queries the
  grouping's live CTPC directly (`TrackFitting.cxx:786`,
  `get_overlap_good_ch_charge`). Retiling changes which 3D points and
  Steiner terminals exist *upstream* of the fit; it does not touch the
  charge values the fit itself looks up.

- **Is this a port regression?" — No.** Both codebases build the charge
  lookup identically (ownership-blind, rectangular, per-plane) and both have
  the identical structurally-scoped-away defense (§4). This is a
  shared, pre-existing limitation of single-view charge lookup design in
  both the prototype and the port: when two physically separate tracks
  alias in exactly one of three projections over a shared arc, nothing in
  either codebase currently notices.

## Why it hid

Nothing in the routine PR output surfaces per-cell cluster ownership before
the fit runs. `PrDisplayDump::dump_proj`
(`toolkit/clus/src/PrDisplayDump.cxx:924-990`) is the *only* place a
`cluster_id` is ever attached to a charge cell, and even there it's a
**post-fit, display-only** annotation (`fc.clusters`, populated during
`TrackFitting::assemble_fitted_charge_2d`, `TrackFitting.cxx:1139` —
itself flagged in `PrDisplayDump.cxx:905-923` as carrying an unrelated,
already-documented nondeterminism in `charge_pred` for cells claimed by more
than one cluster). Bee's own display layers, the nusel TSVs, and every
tagger verdict are all downstream of the finished fit and never expose which
2D cells were shared. Finding this required deliberately cross-referencing
three independent dumps (fit trajectory, raw image, per-plane charge
ownership) against each other by coordinate — exactly the kind of check
nobody runs unless a human notices the fit looks visually wrong, as the
owner did here.

## Fix (proposed — NOT implemented)

Both options below are default-OFF-knob proposals requiring their own
validation round (byte-identical off-gate + on-footprint mover census) per
the operating manual §4, and neither has been prototyped or coded. The
owner reviewed both and prefers a refined version of option 2 — reuse the
cluster's own already-tiled 3D blobs as the consistency check, rather than
comparing candidate cells directly across planes. That refinement is
written up first, as the recommended design; the cross-cluster
`update_association` generalization is kept below it as the larger,
structural alternative.

### 1. Own-blob-coverage down-weighting (recommended)

**Idea**: a candidate 2D `(wire, time)` cell that is not covered by any of
*this cluster's own* tiled blobs is untrustworthy — down-weight it — without
needing to look at the other two planes at all. Blobs are already
3-plane-consistent by construction (`RayGrid` tiling only forms a blob where
a wire-range overlap exists across all three planes simultaneously), so
"does this cell belong to one of my own blobs" is a cheaper proxy for
"is this cell 3D-consistent with my own track" than re-deriving cross-plane
consistency from scratch.

**Empirical support, on this exact event**: §5 above shows cluster 20's own
claimed V-plane cells and cluster 13's are completely disjoint — a
membership test against cluster 20's own blob coverage would have excluded
every contaminating cell while keeping every legitimate one. This is proxy
evidence (post-fit fit-claim provenance, not a direct blob-geometry query —
see §5's caveat), so the first step of any implementation is to confirm it
directly: build the coverage index below from cluster 20's actual
`children()` blobs (data already extracted read-only at
`/home/xqian/tmp/pr49/pctree/` for this investigation) and re-check that
wires 834-841 at slices 607-612 fall outside it.

**The per-plane wire bound is exact, not a loose bounding box — checked
directly against the tiling code, not assumed.** `Facade::Blob`
(`clus/inc/WireCellClus/Facade_Blob.h:78-86`) caches
`slice_index_min()/max()` and `u/v/w_wire_index_min()/max()`, copied
verbatim from the RayGrid shape's strip bounds
(`aux/src/SamplingHelpers.cxx:92-106`). `RayGrid::prune()`
(`util/src/RayTiling.cxx:449-502`) tightens each wire-layer bound to the
floor/ceil of the corner-polygon's projection onto that layer's pitch axis
(`:498-499`). Because a blob region is a convex intersection of pitch-index
slabs, the projection onto any *single* plane's axis is an interval, and
`prune` sets the bound to exactly that interval — so every wire in
`[u_wire_index_min, u_wire_index_max)` genuinely intersects the blob's 2D
footprint in that one plane. The looseness one would normally worry about
(bbox ⊋ true polygon) only shows up if you AND all three planes' ranges
together as a 3D containment test — this design deliberately doesn't do
that, it tests one plane's range at a time, so no polygon/corner refinement
tier is needed. (Correcting an earlier assumption in this investigation:
`Blob::corners()` — `Facade_Blob.h:94` — looked like an available exact-tier
fallback, but the loading code that would populate `corners_` is commented
out in `fill_cache`, `clus/src/Facade_Blob.cxx:137-148`, so `corners()`
always returns empty at the facade level today. Moot here since the
per-plane range is already exact, but worth not repeating as a design
option elsewhere.)

**The aggregation index this needs already exists, cached, and is already
shared across the codebase — no duplication needed at all.**
`Cluster::time_blob_map()` (`clus/inc/WireCellClus/Facade_Cluster.h:640-641`,
type `apa → face → slice_start_tick → BlobSet`,
`clus/inc/WireCellClus/Facade_ClusterCache.h:23-26`) is built lazily on
first call and cached in the cluster's `ClusterCache`
(`clus/src/Facade_Cluster.cxx:317-329`), scoped to that cluster's own
`children()` blobs only. It is already a public, shared Facade::Cluster
accessor — five other components already call it directly
(`retile_cluster.cxx:492-533`, `improvecluster_1.cxx:613-730`,
`connect_graph_closely.cxx:174-221,625-678`,
`clustering_separate.cxx:3943-3952`, `SteinerGrapher.cxx:160-199,369`), so
`TrackFitting` calling it too is the sanctioned reuse pattern, not an M10
fork-by-duplication situation — M10 is about not extracting shared code out
of a *production pass*, not about calling an existing shared data-facade
method that many components already use. `TrackFitting.cxx` does not call
it today (grep confirms). Two related, but not directly callable,
precedents already exist on `Cluster` and shaped the design of the new
predicate rather than being reused verbatim:
`is_point_spatially_related_to_time_blobs` and `check_wire_ranges_match`
(`clus/src/Facade_Cluster.cxx:3243-3327`, `:3333-3390`) — both take a point
*index* (not a bare `(plane,wire,time)` triple) and AND all three planes
together (this design needs a single-plane test), and the latter is
`private` (`Facade_Cluster.h:744`). So the actual new code needed is small
— roughly a 10-line single-plane predicate built on top of
`time_blob_map()`, living directly in `TrackFitting.cxx` — not a duplicated
20-line loop.

**A genuine correctness blocker, not yet resolved, on par with the
tolerance question below: the time key is not uniformly aligned across
`form_point_association`'s three candidate-generation paths.** Candidate
cells from the blob-neighbor branch carry `Coord2D.time =
blob->slice_index_min()` (`TrackFitting.cxx:2206,2221,2258,2263,2268`) —
this matches `time_blob_map()`'s key exactly. Candidates from the Steiner
branch and the fallback branch instead carry a *floor-quantized* tick value
(`floor(tick / cur_ntime_ticks) * cur_ntime_ticks`,
`TrackFitting.cxx:2377,2521`), which is only guaranteed to land on a blob
slice boundary if slice starts happen to be aligned to that grid on every
face — not guaranteed (`Facade_Grouping.cxx:759-767`; the same class of
misalignment already documented for `slice_stride` at
`Facade_Cluster.h:680-687`). A naive `time_blob_map().at(time)` exact-key
lookup would then silently read "not covered" for every Steiner- and
fallback-derived candidate on a misaligned face — which would look like the
fix working (contamination gone) while it was actually just discarding two
of the three candidate sources wholesale. **This must be resolved before
the design is trustworthy**: either snap via `std::map::lower_bound()` to
find the blob-slice interval containing `coord.time`, or verify and assert
alignment explicitly. Not yet done — flagged here as the concrete first
technical task for an implementation, alongside the tolerance question.

**Cost.** `time_blob_map()` build is O(nblobs·log nblobs), once per
cluster, amortized across the whole `do_multi_tracking` call (the cache
guard is a "have I built it" sentinel, not a validity check — safe here
because `TrackFitting` only registers clusters and never restructures a
cluster's blob children mid-call, `TrackFitting.cxx:7908-7917`, but this
should be stated as an explicit assumption in any implementation). Each
candidate-cell test is then a handful of map lookups plus a linear scan of
the `BlobSet` at that slice (typically order 1-10 blobs) — not a scan of
all blobs in the cluster. `RetileCluster::get_activity`
(`clus/src/retile_cluster.cxx:117-186`, same own-`children()`-only pattern,
building a dense per-slice-per-plane wire-hit array instead of a `BlobSet`
map) is available as a cheaper alternative if per-cell map lookups ever
prove to be a hot path.

**One more composition caveat, from the same investigation**: dead/2-view
blobs' re-derived shapes can spill past their true wire boundary
(`sio/inc/WireCellSio/ClusterFileSource.h:86-90`), so the coverage check
should exempt or special-case dead-plane blobs — consistent with the
existing dead-plane rescue interaction noted below, and with
`TrackFitting::prepare_data` already treating bad planes specially via
`Grouping::is_blob_plane_bad` (`Facade_Grouping.cxx:1136-1171`, used at
`TrackFitting.cxx:821-840`).

**Landing site: `examine_point_association`, not `form_point_association`.**
`TrackFitting::examine_point_association` (`clus/src/TrackFitting.cxx:2709-3068`)
already owns exactly this kind of accept/reject-and-weight decision — the
`charge_cut` filter and the `PlaneData::quantity` weight it computes already
flow into the fit as a per-plane scaling factor in `fit_point`
(`TrackFitting.cxx:3517-3526`, reduced when `quantity` is low). So
"down-weight" already has a mechanism to hook into: reduce `quantity` for a
plane whose kept cells are mostly outside the cluster's own blob coverage,
rather than inventing a new weighting path. **One interaction that must be
preserved, not overridden**: the dead-plane rescue already inside
`examine_point_association` (`:2810-3068`) deliberately keeps single-plane
charge when the *other two* planes are dead/empty. The blob-coverage check
must compose with that rescue (run after it, or exempt rescued cells), not
replace it — otherwise a legitimately single-plane-dominant point (a dead
channel in the other two views) could be wrongly down-weighted.

**The open parameter that decides whether this works is tolerance, and it
must be measured, not guessed.** The adjacency in §5 is tight — cluster 20's
own cells and cluster 13's exclusive cells are one tick apart at wire 839.
A membership test with wire/slice tolerance ≥ 1-2 would re-admit exactly the
contamination being excluded; a test with zero tolerance risks dropping
legitimate edge-of-blob cells the fit currently benefits from (the existing
`nlevel`-hop neighbor extension in `form_point_association` exists to reach
real charge near blob boundaries and dead channels). This constant has to be
tuned against real events, not asserted, and is the central open question
for any implementation.

**Complexity, honestly stated**: cheaper than option 2 below — no
fit-ordering dependency, no access to other clusters needed, everything is
local to the cluster already being fit. Still a real behavior change,
needing the same default-OFF knob + full byte-identical gate bar as any
other option here before any implementation or flip.

### 2. Cross-cluster generalization of `update_association` (larger, structural alternative)

Extend the comparison set inside `TrackFitting::update_association`
(`clus/src/TrackFitting.cxx:2563-2680`) from "other segments of *this*
cluster's PR graph" to also include other **already-fitted** clusters'
segments in the same grouping — reusing the exact same
`segment_get_closest_2d_distances` metric already implemented
(`clus/src/PRSegmentFunctions.cxx:142-174`) and the same keep rule
(`min_dis_track < min_dis1_track || min_dis_track < 0.3 cm`).

**Explicit blocker, not glossed over:** this requires an ordering guarantee
— the "other" cluster must already have a completed fit (a `dpcloud("fit")`
to measure distance against) before *this* cluster's `do_multi_tracking`
runs. No such fit-order dependency exists today; `do_multi_tracking` calls
are per-cluster with no cross-cluster sequencing. Introducing one is a
larger change to the pattern-recognition driver, not a local patch to
`TrackFitting`, and would need its own design review before implementation.

### Explicitly not the fix

- **Shrinking/removing the `±5`/`±20` `prepare_data` pad**
  (`TrackFitting.cxx:764-767`) — a real, documented divergence from the
  prototype's unpadded bounding box, but §3 above shows the contaminating
  wires are inside cluster 20's own *unpadded* range. Removing the pad would
  not change this event's outcome. (It may still be worth its own,
  independent look someday as a port-fidelity item — flagged here, not
  pursued.)
- **Flipping `m_fit_exclusion` to true** — structurally a no-op for this
  event (§4 above); cluster 13 is not in cluster 20's segment graph.

## Verification

None yet — no code was changed, so there is nothing to gate. If either fix
above is implemented, it must clear, before any default flip, the same bar
every other change in this tree does (CLAUDE.md §4):

- Knob-off byte-identical gate on the standard manifests (nueCC48,
  ncpi0-19) via `hash_archive.py` member comparison.
- Knob-on smoke test on evt 57441 itself, showing the fit no longer detours
  at the V-plane-ghost location, quoted as a fit-vs-image position diff the
  way §"Symptom" did above. If the blob-coverage design (option 1) is what
  ships, additionally show the V-plane `PlaneData::quantity` for the
  contaminated cells drop (or the cells disappear from the kept set
  entirely) at fit indices 108-113, while U/W `quantity` is unchanged —
  the direct evidence that the down-weighting fired where intended and
  nowhere else.
- A mover census against the current 1k production arm
  (`work-pr48-on1kc` or its successor) to confirm the change does not
  disturb the already-validated pr/48 back-to-back-break population
  (69/1000 movers, all individually examined) — any new movers from this
  knob would need the same individual-examination treatment.
- `./build/clus/wcdoctest-clus` passing, plus new synthetic-track test
  coverage for whichever mechanism is chosen (a two-track-crossing-in-one-view
  synthetic case is the natural regression test).

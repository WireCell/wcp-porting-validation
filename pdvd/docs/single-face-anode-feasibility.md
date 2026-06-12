# Feasibility: merging the two PDVD faces per anode into a single face

Evaluation of the idea to re-describe each PDVD CRP as ONE anode face
(SBND-style: U/V wires spanning the full volume, the collection plane split
handled inside a single face) instead of the current two-faces-per-anode
encoding.  All geometry numbers below were measured directly from the wire
files (`protodunevd-wires-larsoft-v5.json.bz2`,
`sbnd-wires-geometry-v0206.json.bz2`); code claims cite toolkit `file:line`.

**TL;DR** — Merging is *partially* possible: the U/V planes merge trivially
(the cross-face strips are collinear halves of one physical strip, and a
joined plane is exactly SBND-like, one wire per channel).  The collection
plane does **not** merge: PDVD's W strips are split **along the wire
direction** (two y-halves at the *same* pitch positions), while SBND's W
split is **along the pitch direction** (every wire keeps a unique pitch
position).  Ray-grid tiling treats every wire as an infinite line clipped
only by the face's active rectangle, so "which y-half of the strip fired" can
only be expressed by splitting the face — the two-face encoding *is* the WCT
idiom for wire-direction-segmented planes (same mechanism as wrapped-APA
detectors).  A single 3-plane face would silently extend every W channel
across the full 3.36 m height (false coverage → ghost blobs, wrong dead
regions); a 4-plane face (W-bottom + W-top) breaks both the tiling
requirement that a blob has activity in *all* planes and the 3-view (U,V,W)
assumption hard-coded throughout the clus stack.  The practical benefits the
merge would buy — no mid-CRP cluster splits, 8 instead of 16 volumes, a
2-tier instead of 3-tier clustering chain — are mostly reachable *without*
touching geometry, by consolidating the clustering stages (see §6).

## 1. What the two PDVD faces actually are

Wire file: 8 anodes × 2 faces × 3 planes (16 faces, 48 planes, 13 840 wires).
Per anode: 1536 channels = 476 U + 476 V + 584 W.

The two faces of an anode are the **two y-halves of one CRP**, both facing
the same drift direction (anode 0, mm units):

| | face 0 | face 1 |
|---|---|---|
| y extent | [−1685, −6] | [−3364, −1685] |
| U plane x | −3415.1 | −3415.1 |
| V plane x | −3415.3 | −3415.3 |
| W plane x | −3415.5 | −3415.5 |
| U wires / channels | 287 / 287 | 286 / 286 |
| V wires / channels | 286 / 286 | 287 / 287 |
| W wires / channels | 292 / 292 | 292 / 292 |

Strip directions (unit vectors): U = (0, 0.5, 0.866), V = (0, 0.5, −0.866)
(±60° from y), W = (0, 1, 0) — the collection strips run **along y**, with
pitch ~5.1 mm along z.  Top anodes 4–7 mirror in x (planes at +3415.1/.3/.5),
same y split.

**Cross-face channels**: 194 of the 1536 channels per anode (97 U + 97 V)
have a wire segment on *both* faces — segment 0 on face 1, segment 1 on
face 0 (`Wire.segment` field, util/inc/WireCellUtil/WireSchema.h:42).  These
are the induction strips that physically cross the y = −1685 mm boundary.
Measured over all 194 pairs of anode 0: the two segments are collinear to
≤ 0.001°, with a 0.2 mm endpoint gap and 0.17 mm perpendicular offset —
i.e. **one continuous straight strip, artificially cut in two** at the face
boundary.  W has zero cross-face channels: each W strip lives entirely in
one y-half, and the two faces' 292 W wires sit at the **same 292 z (pitch)
positions**.

## 2. How SBND really looks in WCT (the model being proposed)

`sbnd-wires-geometry-v0206`: 2 anodes × **1 face** × 3 planes.  Every channel
is exactly one wire, all segments 0.  U/V: 1984 full-span single wires each
(lengths 10–5789 mm, clipped by the active rectangle).  W: 1670 full-height
wires (y ∈ [−2000, 2000]), one per z position, **perfectly uniform 3.00 mm
pitch straight through the detector middle, contiguous channel numbering, no
dummy wires and no visible split**.

The "two W sets with 6 dummy channels in the middle" exists only at the
LArSoft channel-map level; the WCT wire file has already flattened it.  Its
WCT footprint is the 6-channel full-vertical dead band handled by the
dead-gap registry in clustering, not by geometry.  Crucially, SBND's physical
W split is **along z = the pitch direction**: each half-frame's wires occupy
*distinct* pitch positions, so a single ray-grid plane represents them with
no information loss.

## 3. Why the W plane blocks a single 3-plane PDVD face

Ray-grid tiling (util RayGrid/RayTiling, img/src/GridTiling.cxx) models a
plane as an infinite family of parallel rays at uniform pitch; a wire's
finite extent *along its own direction* is not represented.  The only
clipping is the face's active rectangle (the two horizontal/vertical bounds
layers, GridTiling.cxx:94: `nlayers = m_face->nplanes() + nbounds_layers`).
This is exactly why WCT splits detectors into faces: SBND needs none (every
wire spans the full rectangle), wrapped-APA detectors (PDHD) use front/back
faces, and PDVD uses two y-half faces because its W strips span only half
the rectangle.

Folding PDVD into one 3-plane face would mean the merged W plane has **two
wires and two channels per ray position** (584 channels on 292 rays).  Two
sub-options, both bad:

- **OR the two channels onto one ray**: every W ray then claims coverage of
  the full 3.36 m height.  Activity on a top-half W channel could support
  blobs in the bottom half at the same z (ghost blobs the face split vetoes
  today), and a dead bottom-half W channel would wrongly kill (or a live one
  wrongly validate) the whole column.  A strict resolution/deghosting
  regression.
- **Keep 584 wires at 292 pitch positions in one plane**: ray-grid pitch
  construction requires strictly ordered unique pitch positions; two wires
  per ray is not representable.

A **4-plane face** (U, V, W-bottom, W-top) is the geometrically honest
single-face encoding, and RayGrid itself is N-layer generic — but:

- each W half-plane's rays are still infinite, so the false-coverage problem
  returns unless wire extent is added to ray-grid (a core-engine change);
- GridTiling skips a slice unless *every* plane has activity
  (img/src/GridTiling.cxx:101: `nactivities < m_face->nplanes()`), and a real
  blob only ever touches 3 of the 4 planes;
- the 3-view assumption is hard-coded far beyond tiling: `Facade_Blob`
  caches `u/v/w_wire_index_min/max` (clus/inc/WireCellClus/Facade_Blob.h:36),
  and ~20 clus sources key logic on `kUlayer/kVlayer/kWlayer`
  (ClusteringFuncs, clustering_regular/extend/deghost/neutrino/separate,
  Facade_Cluster, TrackFitting with ~99 references, …).  Charge solving,
  BlobSampler, and the dead-channel machinery all assume 3 planes per face.

So the single-face conversion is not merely "massive config work" — it
collides with a structural assumption of the imaging engine (infinite rays +
3 views per face).  The two-face description is the supported way to encode
PDVD's half-height collection strips.

## 4. What *does* merge cleanly: the U/V planes

If one nevertheless regenerated the wire file with one face per anode:

- join each of the 194 cross-face segment pairs into a single wire (the
  0.2 mm gap is negligible against the 7.65 mm U/V pitch);
- merged U plane: 476 wires = 476 channels; merged V likewise — exactly
  one wire per channel, the SBND shape;
- W is the blocker of §3.

This confirms the user's intuition for the induction planes — they are
already "SBND-like" strips cut administratively in two — but the collection
plane is the opposite case.

## 5. Full ripple list (if the W problem were solved anyway)

For completeness, a merge would additionally touch:

- `params.jsonnet` faces arrays (cfg/pgrapher/experiment/protodunevd/
  params.jsonnet:49-92) — one face per anode;
- imaging: the per-anode `SliceFanout → 2 × GridTiling(face) → BlobSetSync`
  pattern (cfg img.jsonnet:126-163) collapses to one GridTiling;
- DetectorVolumes metadata: 16 → 8 blocks (cfg clus.jsonnet:52-101) and all
  `a{N}f{F}p{P}` key plumbing;
- clustering: 3-tier → 2-tier (the per-face MABC tier and the per-APA
  PointTreeMerging disappear); ClusterScopeFilter face indexing; per-face
  dead samplers (`bs_dead_face`);
- chndb / coherent-group channel ranges revalidation against the regenerated
  file; the U/V channel↔wire assignment audit (already flagged unverified in
  img/docs/protodune-wire-geometry-channel-mapping-audit.md) would need to be
  redone for the merged numbering;
- magnify/Bee tooling that names per-face artifacts.

## 6. Recommendation: get the benefits without the geometry change

The practical costs of the two-face encoding are (a) blob/cluster splits at
the mid-CRP y = −1685 mm line, (b) 16-volume bookkeeping, (c) the 3-tier
clustering chain.  Most of this is addressable in clustering configuration
alone, keeping geometry and imaging untouched:

1. **Drop the per-face MABC tier**: feed both faces of an anode into one
   grouping and run the full merge pipeline per-APA.  Per the scope table in
   [clustering-scope.md](clustering-scope.md), every pass in the current
   per-face pipeline is scope-agnostic (`yes*`) except `connect1` and
   `examine_x_boundary`, which hard-assert single-face
   (clus/src/clustering_connect.cxx:69) — run those per-face first or extend
   them.  Cross-boundary U/V strips already give the two faces correlated
   activity, and the per-APA merge passes would heal the y = −1685 splits the
   same way the all-APA stage heals CRP-boundary splits today.
2. The 16 metadata blocks are jsonnet-generated boilerplate; they cost
   nothing at run time.
3. Imaging keeps the per-face tiling — which is also what protects W
   dead-channel locality and deghosting.

This is the staged path: cheap, reversible, and it can be validated against
the current chain event-by-event.  The single-face geometry rewrite should
only be reconsidered if ray-grid ever grows first-class support for
wire-extent (segmented planes), which would be a toolkit-core project, not a
PDVD config project.

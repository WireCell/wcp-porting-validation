# PDVD over-clustering: vertical track + crossing isochronous bands (run 39324 evt 0)

## Symptom

In drift group 4567 (anodes 4–7), one cluster (id 51, 32268 points) contained
three structures that should be separate:

- a steep "vertical" track (large drift-x extent): points `(-78.3,-187.7,133.0)`
  and `(13.6,-203.0,95.9)` both lie on it;
- two broad isochronous bands (~constant x ≈ −150 cm, wide ribbons in Y–Z that
  cross each other in an X): point `(-150.6,-96.9,109.6)` is on one of them.

## What actually merged what (traced with temporary merge-decision prints)

Per-anode (stage-2) components of cluster 51: anode4 {cl46 = vertical, cl6 =
X-shaped 11k-pt blob containing BOTH bands' CRP4 segments, cl25}, anode5
{cl6, cl28, cl33}, anode6 {cl57}, plus ~12 small fragments.

| link | pass / branch | measured |
|---|---|---|
| vertical ↔ iso band | stage-3 `extend(flag=4)` dead-region pass, non-parallel `is_angle_consistent(dir1,dir2) && is_angle_consistent(dir3,dir2)` branch (`E-dead-angcons-both`) | dis = 6.07 cm |
| band pieces across CRP boundaries | `clustering_regular` cross-APA/face long-track branch (`len>100cm && dis<5cm`, **no angle check**) | dis = 1.57 / 3.81 cm |
| band pieces across CRP boundaries | `clustering_regular` parallel branch, `para_angle_cut=60°` at dis<5cm | angle_diff 28–40° |

Key findings:

1. The two iso bands **cross inside CRP4** — anode4 cl6 is already a spatial
   X-merge at the per-face stage (PCA planarity ratio 0.63).  Since the 4-stage
   restructure, `separate` no longer runs per-face, and no merge cut can keep
   apart clusters that physically overlap.  The stage-3 iso↔iso merges above are
   *legitimate continuations* of the bands across CRP boundaries (pieces touch
   at 1–5 cm); tightening those cuts is not the fix.
2. `connect1`'s `iso_max_dis` guard (the SBND isochronous fix) is not involved:
   `connect1` only runs per-face (stage 1) and all the merges here happen at
   stage 3.
3. The **`separate` safety net never ran**: `Cluster::get_hull` returns an empty
   hull for clusters above `max_hull_points` (default 10000), which makes
   `JudgeSeparateDec_1` bail — `separate` silently skipped the 32k-pt cluster.

## Fix

`cm.separate(use_ctpc=true, max_hull_points=100000)` in
`cfg/pgrapher/experiment/protodunevd/clus.jsonnet` (and the same for PDHD),
mirroring SBND.  See `clus/docs/clustering-separate-hull-cap.md` in the toolkit
repo for the full analysis.

Result (run 39324 evt 0): the vertical track separates cleanly (its own
cluster, all 1409 points); the X of crossing bands is carved into band-aligned
pieces (imperfectly — a `separate`-algorithm limitation for wide crossing
ribbons, not a cut-tuning issue).  The three reported points go from one
cluster to two (vertical / iso).

## Refinement: collinear tip recovery + two-band re-carve

The post-hull-fix carve had three reported defects (group4567 output ids):

- the vertical track (cluster 53) was missing its tip — 190 points lying ON
  its PCA axis (3.2° collinear, <4 cm perpendicular) across a real **6.07 cm
  imaging gap**, larger than `Separate_2`'s 5 cm relink, stranded in cluster
  51: pair `(-127.3,-178.1,153.5)` / `(-118.6,-178.8,148.8)` split;
- the band complex was carved arbitrarily: 51 was the "leftover bin" (27
  disconnected components at 3 cm, chunks of BOTH bands), 52/57 arbitrary
  slices of one band: pairs `(-154.4,-307.7,132.0)` / `(-153.8,-253.8,135.1)`
  (57/52) and `(-155.1,-277.6,235.1)` / `(-153.8,-281.1,210.6)` (51/52) split.
  The bands cross at only ~13° in y–z and overlap heavily; local directions
  inside a band vary 40–60°, so only a global 2-line model can assign
  fragments per band.

Fix: two knob-gated post-separation steps in `ClusteringSeparate`
(`collinear_recover` + `band_recarve`, default OFF, enabled for PDVD and PDHD)
— see `clus/docs/clustering-separate-refine.md` in the toolkit repo for the
algorithms and tunables.

Result (39324 evt 0, group4567): all three flagged pairs land in one cluster
each; vertical track whole (1451 → 1565 pts, the 114-pt tip recovered) and
separate; the band complex is exactly **2 clusters, one per physical band**,
with the carve boundary a clean fit line through the genuinely ambiguous
overlap region.  Group0123 of the same event: one clean tip recovery (a long
drift track adopts 42 collinear continuation points from a 380-pt leftover
fragment).

### Refinement regression (same set as below, plus PDHD 027409/027380)

- Knobs OFF: byte-identical to pre-refinement outputs (md5 of all three
  clustering JSONs, 39324 evt 0).
- PDVD 39252 evts 0–4, 39253 evts 0–4: identical.
- PDVD 39324: evts 1, 3, 5, 6, 9, 10 identical; changes confined to the
  separated families of evts 0, 2, 4, 7, 8 and qualitatively all improvements:
  - evt 2: a long drift track (x 50→341 cm) reassembled from 3 carve fragments
    via tip recovery (final pca eval ratio 0.044);
  - evt 4: two crossing band complexes consolidated from 5 arbitrary pieces to
    2 (one per band);
  - evt 7: a horizontal track (x −67→320 cm) reassembled from 3 fragments
    (ratio 0.008); the 21.5k-pt band complex re-partitioned per band;
  - evt 8: a thin inclined track (ratio 0.026) extracted from a ratio-0.57
    mash piece; the band complex consolidated.
- PDHD 027409: evts 0, 2, 3, 5, 7, 8 identical (evt 6 relabel-only); evts 1, 4
  show the same two flavors (band-pair re-partition; band recarve + tip
  donations to two tracks).
- PDHD 027380: pre-existing references were stale (point *sets* differ —
  sampled points can't be affected by the refinement, which only moves blobs
  between clusters), so verified by a same-binary ON-vs-OFF comparison
  instead: across all 8 events the only difference is one 20-point tip
  recovery in evt 2.
- Determinism: two knobs-ON runs of 39324 evt 8 (the event exercising both
  steps on the same family: `band_recarve` pooled 5 members at seed angle
  38.8°, sides 24381/12715 blob-npoints; `collinear_recover` claimed 177 blobs
  for a 191 cm track) are byte-identical.

## Regression (runs 39252 evts 0–4, 39253 evts 0–4, 39324 evts 0–10)

- All group0123 outputs identical; 39252/39253 fully identical
  (39253_1..4 baseline zips predate the current config — stale, not a fix
  effect; their inputs are byte-duplicates of evt 0, which is identical
  old-vs-new under the current config).
- 39324 evts 1, 5, 6, 9, 10: identical.
- 39324 evts 0, 2, 4, 7, 8: only diffs are splits of >10k-pt over-merged
  clusters (32k→6, 77k→12, 16k→3 + 25k→4, 39k→7, 42k→5 pieces) plus re-grouping
  of the split pieces.
- 39324_3: both group jsons identical; global differs only by stage-4
  `cathode_connect` now joining a genuine cathode-crossing pair (old clusters
  30+70, two halves reaching x = +4.7 / −4.4 cm from opposite drift sides).
  Reproducible (two post-fix runs byte-identical) — a second-order side effect
  of `separate` engaging at stage 3 (internal cluster state/order differs even
  when the partition is unchanged), and an improvement: the crosser is real.

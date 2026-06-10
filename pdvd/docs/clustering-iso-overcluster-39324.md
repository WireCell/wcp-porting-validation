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

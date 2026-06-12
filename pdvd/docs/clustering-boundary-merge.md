# PDVD clustering: why CRP-boundary cluster pairs stay separated

Run 39324 evt 0.  Three cluster pairs in the Bee `0-clustering-group0123` view
look like single tracks but carry different cluster ids:

| pair | user points (x,y,z) cm | ids (fallback-off zip) | closest gap | boundary | anode/face |
|---|---|---|---|---|---|
| 1 | (46.0,−6.8,226.6) / (60.4,1.0,226.0) | 45 / 109 | **2.39 cm** | y ≈ 0 | a1f0 ↔ a3f1 |
| 2 | (−306.4,175.5,188.3) / (−298.7,161.7,178.1) | 101 / 116 | **2.01 cm** | y ≈ 168.5 | a3 ↔ y-neighbor |
| 3 | (−69.6,149.7,204.1) / (−82.7,176.5,171.5) | 107 / 102 | **0.90 cm** | y ≈ 168.5 | a3f0 ↔ y-neighbor |

(The quoted points are where the user clicked; the *closest* approach between
the two point sets is mm-to-2 cm in every case, sitting exactly on a CRP
anode/face boundary.)

## Q1 — are they separated from the start (imaging gap)?

Effectively yes, but not because of a charge gap: **imaging is per anode**, so a
track crossing a CRP boundary is imaged as two blob sets that can approach each
other only down to the boundary dead band (the measured 0.9–2.4 cm).  There is
no cross-anode blob, hence no single imaging cluster.

## Q2 — should the per-APA clustering merge them?

It **cannot, by construction**:

* the full merge pipeline (`pointed, live_dead, extend, regular ×2,
  parallel_prolong, close, extend_loop, separate, connect1`) runs **per face**
  (`pdvd/clus.jsonnet` `clus_per_face`);
* the per-APA stage only merges the two faces' point *trees* and runs
  `protect_overclustering` — **no merging pass at all**
  (`pdvd/clus.jsonnet:328-331`).

The Bee `0-clustering-group0123/4567` dumps are the **input** of the all-APA
instance (the `name:"img"` hook), i.e. exactly this per-anode result — so
boundary pairs are *always* separate there.  Cross-boundary merging is the job
of the **all-APA** stage that follows.

## Q3 — what merges (or fails to merge) them in the all-APA stage?

Instrumented run (temporary printouts in `Clustering_1st_round`,
`merge_clusters`, `clustering_switch_scope`, `T0Correction::filter`; all
reverted afterwards):

* **Pair 2 (101/116)** — merged by **`ClusteringExtend:all`** (first all-APA
  merge pass).
* **Pair 3 (107/102)** — merged by **`ClusteringRegular:all1`**: dis = 0.90 cm,
  `flag_para=1`, `flag_regular=1`, hough angle-diff 3.1°/1.2° well inside the
  `dis ≤ 3 cm` branch (`angle_cut*1.5 = 18°`).
* `ClusteringIsolated:all` later merges both into bigger cosmics; final global
  clusters 2 and 62.  **Working as intended.**
* **Pair 1 (45/109) — never merged, and never even *evaluated*.**

### Root cause for pair 1: the switch-scope volume filter vs. no-T0 PDVD

`ClusteringSwitchScope` (first thing the all-APA pipeline does) splits every
cluster by a per-blob test: a blob survives only if one of its **T0-corrected**
points is contained by the detector volume **of its own (apa,face)**
(`clustering_switch_scope.cxx:64-89`, `PCTransforms.cxx` `T0Correction::filter`
— `contained_by(point)` must return the blob's own apa/face).  Clusters whose
blobs all fail get `scope_filter = false` and **every subsequent pass skips
them** (`if (!cluster->get_scope_filter(scope)) continue;` at the top of each
pair loop).

PDVD has **no flash matching**, so `cluster_t0 = 0` and the "corrected" x is
the raw apparent x — which for an out-of-time cosmic lies in the *wrong drift
volume*.  Measured at the pair-1 points:

```
target (46.0,−6.8,226.6): blob apa/face=1/0  contained_by → apa/face=5/0  → filter=0
target (60.4, 1.0,226.0): blob apa/face=3/1  contained_by → apa/face=7/1  → filter=0
```

Anodes 5/7 are the **opposite drift side**.  Consequences for pair 1:

* cluster 109 (514 pts, x ∈ [58.5, 126.9]) is **entirely** filtered out — it
  passes through the whole all-APA pipeline untouched (514 pts in, 514 out);
* cluster 45 (2096 pts) is **split** by switch_scope: its 495-point boundary
  segment (the part near y≈0) is filtered out, the rest participates normally.

So the two halves of the crossing track are invisible to every merge pass —
nothing "re-separates" them; they are excluded before any merging happens.

### How big is the effect?

Apparent-x on the wrong drift side (sign test, approximating the volume
containment), run 39324 evt 0:

* group0123: 4444 / 25371 points (**17.5 %**); 13 of 116 clusters entirely,
  5 more partially;
* group4567: 35704 / 53647 points (**66.6 %**); 123 of 284 clusters entirely,
  7 more partially.

All of these clusters/fragments skip the all-APA merge passes.

## Status: FIXED (2026-06, two rounds)

The switch-scope x-containment test was designed for chains where a flash-T0
correction has already placed clusters at their true x (SBND).  For PDVD (no
T0) it silently excludes a large fraction of clusters from cross-boundary
merging.

**Round 1** — `relax_containment_filter` option on `PCTransformSet` (C++
default off = production bit-identical; PDVD config sets it true): the filter
accepted a point contained by *any* sensitive volume instead of its own
(apa,face).  This recovered the opposite-drift-volume exclusions above
(evt 0: 229 → 81 global clusters; pair 45/109 merged).

**Round 2** — the any-volume form still excluded clusters sitting **entirely
outside all sensitive volumes**: early-time activity in the band between the
anode-face boundary (|x| = 335.835 cm) and the wire planes (|x| = 341.55 cm),
and late-time activity in the cathode gap (|x| < 2.54 cm).  Evt 0: 25 of 81
global clusters (10 anode-band + 15 cathode-band) — e.g. a 92-point track tip
at x ∈ [−341.6, −336.1] separated by only 0.57 cm from its parent track.
Relaxed mode now **disables the filter entirely** (every point passes, no
switch_scope splitting); evt 0 global clusters drop 81 → 51.  See
[clustering-scope.md](clustering-scope.md) §3 for the knob semantics.

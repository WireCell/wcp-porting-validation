# doc pr/73 — why the fitted trajectory zigzags, and the low dQ/dx that follows

**Scope: diagnosis only. No C++ and no jsonnet is changed by this round, so no
A/B gate is run or claimed.** Every number below is read off Bee archives that
already exist on disk.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the three owner cases: step size, the ribbon, the arm sweep, per-point dQ
python3 scripts/analysis/pr73/zigzag_anatomy.py --png docs/pr/73_evt57903_zigzag.png
#   -> docs/pr/73_anatomy_output.txt  (committed verbatim)

# the population
python3 scripts/analysis/pr73/zigzag_census.py work-pr51r7-on50 \
        --tsv docs/pr/73_zigzag_census_r7on50.tsv
#   -> docs/pr/73_census_output.txt   (committed verbatim)
```

Both scripts are read-only: they open each arm's `pr_evt*/mabc-pr.zip` and run
nothing.

## 1. The three cases

| event | Bee | owner point (x, y, z) cm | arm holding that Bee content |
|---|---|---|---|
| 18255-53427 | `f8203fcd` / `event/13` | (−24.6, −17.1, 446.1) | `work-pr64r4-on1k` |
| 18255-54351 | `f8203fcd` / `event/17` | (−150.3, 82.4, 196.2) | `work-pr64r4-on1k` |
| 18255-57903 | `deb8abf5` / `event/0` | (−16.5, −65.6, 293.6) | `work-pr51r6-flip50` (round-6 production) |

Bee index ↔ event confirmed against `docs/pr/mcp1k50-prod0811.index.txt`
(`13 → 53427`, `17 → 54351`), following doc pr/67 §1 — the owner's links are
right. `deb8abf5` is the *before* set of doc pr/51 round 7, i.e. round-6
production; current production is round 7, arm `work-pr51r7-on50`, and both are
measured below.

The owner's reading of 57903 — "clearly isochronous, the easy answer is a
straight line between the two ends, but the fit zigzags and the fitted dQ/dx is
low in places" — is the case this doc is built around. 53427 and 54351 turn out
to be a **different** failure and are separated out in §4.4.

## 2. Answer in one line

**The fit does not create the zigzag — it fails to remove it.** `fit_point`
solves each trajectory point independently with no smoothness term and no ridge,
so along any poorly-constrained direction the answer is whatever the *seed* said;
and every anti-zigzag guard downstream measures the fitted point against that
same seed, so a zigzagging seed shields itself. On 57903 the seed changed under
pr/51 **round 6** and the smooth answer that survived round 5 intact was lost
(§4.3; round 5 moved no point by more than 1 cm, §4.8 says why).

**And the three cases are not the same shape.** §4.7 splits `path/chord` into a
smooth large-amplitude excursion (a **bow**) and a small-amplitude sawtooth
(**jitter**). 57903 is bow-dominated (1.225 of its 1.347); 53427 and 54351 are
almost pure jitter (1.040 of 1.055, 1.037 of 1.042). Because
`multi_trajectory_fit` **pins both segment endpoints to `vertex->fit().point` and
never fits them** (`:4246-4259`), a vertex sitting off the charge ridge *forces*
a bow — no amount of interior smoothing can remove it. That is why the fix
ranking in §6 puts the seed/vertex fix ahead of trajectory smoothing.

## 3. The chain, with the code

Stage by stage, `clus/src/TrackFitting.cxx` unless stated:

1. **Imaging.** A near-isochronous stretch lands in a handful of drift slices, so
   its blobs are wide ribbons in y–z rather than points (§4.2).
2. **Seed.** `organize_segments_path` (`:8234`) turns the Steiner/shortest path
   into the trajectory seed; `organize_segments_path_3rd` (`:8571`) re-spaces it.
   Note `:8570` reassigns `low_dis_limit = 0.6*units::cm` first, so the final
   polyline — the one Bee writes and the one whose `local_dx` feeds the dQ/dx
   fit — is uniformly spaced at **0.600 cm** (§4.1).
3. **Position fit.** `fit_point` (`:3697-3942`) builds, per associated 2-D cell,
   a wire row and a time row weighted by `s_k = (Q/σ_Q) · div · quality · ghost`,
   and solves
   `A = RUᵀRU + RVᵀRV + RWᵀRW`, `b = RUᵀd_u + RVᵀd_v + RWᵀd_w`
   as an independent **3×3** system via `BiCGSTAB::solveWithGuess(b, seed)`
   (`:3926`). There is **no term coupling neighbouring points** anywhere in this
   function, and no damping. Where a plane degenerates, the code does not detect
   it geometrically: `examine_point_association` (`:2862-3334`) computes a
   statistical `PlaneData::quantity`, and where that is 0 it substitutes a
   synthetic cell **at the seed point's own projection** and drops the weight to
   `scaling_ratio = 0.05` (`:3778-3784`). The comment at `:4498-4501` states the
   consequence outright: such a point "has no 2D associations, so the solve keeps
   (regularizes around) the seed."
4. **Guards — all measured against the seed.** In
   `examine_segment_trajectory` (`:5004-5188`):
   * `temp_fine_tracking_path` is filled from `init_ps_vec`, i.e. the **pre-fit
     seed** (`:5070`);
   * the triangle-area smoothing replaces the fitted point (`:5175`,
     `fine_tracking_path[i] = temp_fine_tracking_path[i]`) only when
     `area1 > area_ratio1 * c && area1 > area_ratio2 * area2` (`:5112`, `:5140`,
     `:5168`, for the three neighbour triples), where `area1` is the sagitta of
     the *fitted* point and `area2` the sagitta of the *seed* point (SBND
     `area_ratio1 = 1.8 mm`, `area_ratio2 = 1.7`). **A seed that already zigzags
     makes `area2` large and the test never fires** — the seed shields the fit,
     and the replacement is the seed point anyway;
   * `skip_trajectory_point`'s charge veto (`:5189-5520`) reverts `p = ps_point`,
     i.e. *to the seed*;
   * the dead-plane angle veto (`:5501`) needs ≥2 planes at `quantity == 0`,
     which does not happen on any of these three events — all three planes carry
     charge.
5. **dQ/dx.** `dQ_dx_multi_fit` (`:6119-7292`) solves for dQ per point, then
   divides by `local_dx` = the half-sum of the two adjacent chords (`:6446-6463`)
   — so a zigzag inflates the denominator directly. The only `lambda`-weighted
   regulariser in the whole chain (`FᵀF`, `:7129-7177`) smooths **dQ/dx**, using
   that already-inflated `dx`; it does not regularise geometry.

## 4. Evidence

![evt 57903 zigzag anatomy](73_evt57903_zigzag.png)

### 4.1 What is being measured

Step length on the three owner segments: min 0.599 / median 0.600 cm on all
three. 53427 seg 14002 has a max of 3.000 cm — the polyline is uniform 0.6 cm
**except across gaps**, so `path/chord` on that long segment carries a little
gap-bridging as well as zigzag. `path/chord` is otherwise a direct zigzag
measure.

`q` in the Bee `track_fit-global` layer is
`fit.dQ * dQdx_scale + dQdx_offset`, clipped at 0
(`MultiAlgBlobClustering.cxx:955-957`), and SBND uses `0.1 / −1000`
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet:1768`). So
`dQ[e] = (q_bee + 1000)/0.1`, and **`q_bee == 0` means `dQ ≤ 10⁴ e` over that
0.6 cm step** — a real dQ/dx deficit, not the absence of charge. All statements
below are phrased that way.

### 4.2 18255-57903: the image is a ribbon, and the segment is one drift slice thick

Segment 14001's charge occupies **6 distinct drift slices** over a 24.10 cm
chord (slice pitch 0.3120 cm) — **4.0 cm of track per slice**. For comparison,
segment 14006 in the same cluster (26.4° out of the drift-⊥ plane, `path/chord`
1.029) occupies 62 slices, 0.54 cm of track per slice.

Selecting image points **fit-independently**, by an 8 cm cylinder about the
segment's own chord: 226 points, **drift-direction rms 0.33 cm** (one slice)
versus **in-plane transverse rms 2.60 cm**. The charge is a ribbon roughly 5 cm
wide lying in a constant-drift plane. There is no per-point 3-D anchor for
`fit_point` to hold onto.

**A caveat that must not be dropped.** Charge within 3 cm of the *zigzagging fit*
is 2 671 084 e (36.6 % of the in-band cluster charge); within 3 cm of the
*straight chord between the current two ends* it is 1 687 880 e (23.1 %). The
straight chord captures **less**, because the endpoints themselves are displaced
from the charge ridge (the ridge sits 3–5 cm off the chord along its whole
length, top-right panel). So "a straight line connecting the two ends" is not
literally the better answer for this event as currently segmented; the target is
a **smooth line through the charge ridge**, which is a stronger requirement than
straightness and is what §5 is written against.

### 4.3 The discriminator: the zigzag tracks the seed, not the fit

Same event, same fitting code, different **upstream** knobs:

| arm | what changed upstream | seg | n | chord | path | path/chord | max in-plane |
|---|---|---|---:|---:|---:|---:|---:|
| `work-pr67f-off50` | pre-round-5 (`steiner_gap_penalty` off) | 14003 | 83 | 48.13 | 49.38 | **1.026** | 2.15 |
| `work-pr51r5-flip50` | round 5 (`steiner_gap_penalty = 2.0`) | 14003 | 108 | 60.80 | 63.92 | 1.051 | 4.17 |
| `work-pr51r6-flip50` | round 6 (+ `sgp_weak_scale = 5.0`) | 14001 | 55 | 24.10 | 32.46 | **1.347** | 6.85 |
| `work-pr51r7-on50` | round 7 = current production (+ `mvfit_robust`) | 14001 | 52 | 23.36 | 30.78 | 1.318 | 3.93 |

and the vertex the segment ends on:

| arm | main vertex (x, y, z) cm | opening angle of the two legs at it | cluster-14 charge within 1.5 / 3.0 cm of any fit point |
|---|---|---:|---|
| `work-pr67f-off50` | (−16.40, −58.05, 265.90) | 154.9° (near-collinear) | 44.3 % / 57.7 % |
| `work-pr51r5-flip50` | (−20.30, −52.70, 253.54) | 176.5° (collinear) | 44.3 % / 57.5 % |
| `work-pr51r6-flip50` | (−16.50, −70.36, 293.54) | **27.2° (hairpin)** | 44.2 % / 58.5 % |
| `work-pr51r7-on50` | (−16.52, −65.92, 292.26) | 69.3° | 44.9 % / 57.5 % |

**The per-segment `path/chord` in that first table is not like-for-like** — the
arms cut the same corridor into segments at different places, so a segment can
inflate purely by absorbing a curved piece from its neighbour. Fixing the window
to the pre-round-5 segment's own z extent, [265.90, 314.03] cm, and re-measuring
every arm's polyline inside it:

| arm | npts | chord | path | `path/chord` | bow | jitter rms |
|---|---:|---:|---:|---:|---:|---:|
| `work-pr67f-off50` (pre-R5) | 83 | 48.13 | 49.38 | **1.026** | 1.90 | 0.081 |
| `work-pr51r5-flip50` (R5) | 83 | 47.57 | 49.20 | **1.034** | 1.93 | 0.082 |
| `work-pr51r6-flip50` (R6) | 54 | 23.52 | 31.86 | **1.355** | 5.91 | 0.259 |
| `work-pr51r6-flip50` (R6) | 64 | 31.33 | 37.80 | **1.206** | 5.18 | 0.303 |
| `work-pr51r7-on50` (R7) | 51 | 22.76 | 30.18 | 1.326 | 2.52 | 0.342 |
| `work-pr51r7-on50` (R7) | 60 | 28.28 | 35.40 | 1.252 | 2.94 | 0.273 |

and the pointwise divergence of the whole cluster-14 fit:

| comparison | max deviation | points moved > 1 cm |
|---|---:|---|
| pre-R5 → R5 | **0.69 cm** | **0 of 176** |
| R5 → R6 | 11.02 cm | 70 of 176 |
| pre-R5 → R6 | 10.91 cm | 72 of 176 |

Read together:

* **Round 5 is not the culprit.** It leaves the corridor alone — 0.69 cm max
  deviation, *zero* points moved by more than 1 cm — and like-for-like it costs
  1.026 → 1.034 with the bow and jitter unchanged (1.90 → 1.93 cm, 0.081 →
  0.082 cm rms). The 1.026 → 1.051 in the per-segment table is a **segmentation
  artifact**: round 5 slid the main vertex 15 cm *along* that unchanged corridor,
  so segment 14003 grew 48.13 → 60.80 cm by swallowing a curved piece that had
  belonged to 14004.
* **Round 6 is the whole regression** — see §4.8 for the mechanism.
* Before pr/51 round 5, this region was **one smooth 48 cm segment at 0.6° from
  isochronous with `path/chord` 1.026** — *more* isochronous than today's
  and smooth. **Isochronicity alone therefore does not predict the zigzag.**
  The seed does.
* Round 6 moved the main vertex 43.9 cm from where round 5 left it (30.2 cm from
  its pre-round-5 position) and converted a near-collinear vertex into a 27°
  hairpin; both resulting legs zigzag (1.36 and 1.21). Round 7 opened the hairpin to 69° and
  reduced the excursion 6.85 → 3.93 cm — but those two residuals are measured
  about different chords (24.10 vs 23.36 cm), so read that as "smaller", not as
  a factor. The like-for-like number is the bow amplitude in §4.7: 5.94 → 2.52 cm.
* **Charge coverage is flat across all four arms** (44.2–44.9 % within 1.5 cm).

The whole-cluster coverage number includes segment 14006, which no knob moves,
so it dilutes a local change. Restricting to the **changed region only** —
cluster-14 image points with z in [249.64, 314.23] cm, the union z-range of
round-6 segments 14001 and 14007, an identical 963-point / 1.043 × 10⁷ e sample
in every arm:

| arm | charge within 1.5 cm of any fit point | within 3.0 cm |
|---|---:|---:|
| `work-pr67f-off50` | 30.3 % | 47.0 % |
| `work-pr51r5-flip50` | 30.2 % | 46.8 % |
| `work-pr51r6-flip50` | 30.1 % | 48.0 % |
| `work-pr51r7-on50` | 31.0 % | 46.8 % |

Flat there too. Whatever the topology change bought elsewhere, on this event it
bought no coverage — locally or globally — and cost trajectory smoothness.

**This is a regression signal on shipped, owner-flipped SBND production knobs**
(specifically `sgp_weak_scale`, doc pr/51 round 6 — **not** round 5, §4.3/§4.8), stated here
rather than buried. It is *not* a claim that those rounds were wrong: they were
validated on other events and their gates were clean. It is a claim that this
event was collateral and that nobody looked at trajectory smoothness as a
criterion at the time. Per CLAUDE.md §5.7 this is reported, not tuned.

### 4.4 18255-53427 and 18255-54351 are a *different* failure

At both owner points the fitted trajectory is **on** the charge — nearest image
point 0.08–0.67 cm, with ~10⁵ e within 1.5 cm at every step — yet dQ collapses
about 3× and recovers:

| event | seg | dQ before → at the dip → after (e) | turn angle at the dip | nearest image point there |
|---|---|---|---:|---|
| 53427 | 14002 | 33 714 → **11 277** → 33 731 | **130° and 136°** (i = 165, 166) | 0.10 / 0.47 cm |
| 54351 | 17007 | 33 841 → **13 002** → 33 574 | **39° and 70°** (i = 70, 72) | 0.36 / 0.67 cm |

The dQ minima coincide **point for point** with fold-back kinks in the
trajectory. Geometrically these are small — sub-centimetre excursions on a 0.6 cm
step — so the `local_dx` inflation (local `path/chord` 1.108 and 1.038) cannot
account for a 3× drop on its own. The dQ *assignment* is what fails.

57903 is not like this. There the trajectory walks **off** the ridge: `q_bee`
is 0 for eleven consecutive points (i = 42…52, 6.6 cm), the nearest image point
reaches 1.19 cm, and the charge within 1.5 cm falls from 50–95 k e to 19–38 k e.

So the owner's three points are two distinct defects: **57903 = geometry**
(the trajectory leaves the charge), **53427 / 54351 = dQ assignment** (the
trajectory stays on the charge and the charge is not assigned to it). The named
suspect for the second is the compact-matrix overlap down-weighting in the dQ/dx
fit — `calculate_compact_matrix_multi` (`:5729`) and the `MU/MV/MW` diagonals it
rewrites (`:7105-7107`) — which fires exactly when consecutive 3-D points share
2-D pixels, i.e. at a fold-back. **That is a hypothesis with a probe (F5), not a
conclusion**: nothing here measures the matrix.

### 4.5 The population (50-event manifest, `work-pr51r7-on50`)

124 segments with ≥10 points and ≥5 cm chord:

| bin | n | median `path/chord` | p90 | median fold-back fraction (turn > 30°) | median `q_bee == 0` fraction |
|---|---:|---:|---:|---:|---:|
| iso < 10° | 22 | 1.070 | 1.242 | 24.8 % | 0.0 % |
| iso ≥ 10° | 102 | 1.031 | 1.111 | 9.1 % | 0.0 % |

Two bins deliberately: a finer binning produced cells of n = 4 that cannot
support a trend. Fold-back kinks are **endemic** — 9–25 % of all fitted points
turn by more than 30° between consecutive 0.6 cm steps — not an exceptional
condition of these three events.

**Negative result, stated because it constrains the fix.** Spearman rank
correlation of the `q_bee == 0` fraction against fold fraction / `path/chord` /
isochronicity is only **+0.14 / +0.26 / +0.17**, and the median is 0 % in both
bins. The zigzag ↔ dQ-hole link is solid in the named events but is **not** a
population-level effect. Anyone selling a fix on "it will fix the dQ/dx" needs
per-event evidence, not the census.

For orientation, fold fraction and `path/chord` are near-redundant (Spearman
+0.87); isochronicity against `path/chord` is +0.44.

**A second limitation of this census, from §4.7.** `path/chord` sums a bow and a
sawtooth, and the fold-back fraction sees only the sawtooth — a 40° turn at a
0.6 cm step is a 0.4 cm lateral kick, so the fold metric is blind to a 6 cm bow
entirely. The two bins above therefore merge two phenomena that need different
fixes. Any population number used to size a fix should be recomputed with the
§4.7 split; the F1 probe should emit `ratio_bow` and `ratio_jit` separately.

### 4.6 The existing isochronous guard is cluster-level; the defect is segment-level

`skip_revert_iso_xext_cut` (`:5392-5407`, SBND production 20 cm) abstains from
`skip_trajectory_point`'s charge revert when the **cluster's** drift extent is
small, on the reasoning (doc pr/28 round 10) that on an isochronous cluster the
two charge samples being compared are the same overlapping blob.

57903's cluster 14 spans **21.26 cm** in drift — 1.26 cm *above* the cut — so
`abstain = false`, the revert is active, and the zigzag survives anyway. The
extent is a max − min, so a per-cluster T0 offset cancels; the residual
blob-centre-versus-point-cloud uncertainty is about one slice (0.31 cm), which
leaves the conclusion standing but marginal.

The reason the cluster is not isochronous while the segments are: segments 14001
(1.0° from the drift-⊥ plane) and 14007 (6.3°) share cluster 14 with segment
14006, which spans 19 cm in drift by itself. **The condition that matters is a
property of the segment, and every isochronous test in the fit today is a
property of the cluster.**

### 4.7 Bow versus jitter — the three cases are not the same shape

Fit a degree-4 polynomial in arclength to each transverse component, then report
`path/chord` **of** that smooth curve (`ratio_bow`) and `path/chord` of the raw
path **about** it (`ratio_jit`); the two multiply to the total.

| arm / event | seg | chord | `path/chord` | bow amplitude | `ratio_bow` | `ratio_jit` | jitter rms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pr64r4-on1k` / 53427 | 14002 | 190.00 | 1.055 | 10.80 | 1.015 | **1.040** | 0.314 |
| `pr64r4-on1k` / 54351 | 17007 | 52.03 | 1.042 | 1.42 | 1.005 | **1.037** | 0.159 |
| `pr51r6-flip50` / 57903 | 14001 | 24.10 | 1.347 | 5.94 | **1.225** | 1.099 | 0.258 |
| `pr51r7-on50` / 57903 | 14001 | 23.36 | 1.318 | 2.52 | 1.136 | 1.159 | 0.342 |
| `pr67f-off50` / 57903 | 14003 | 48.13 | 1.026 | 1.90 | 1.005 | 1.021 | 0.081 |

* **57903 is a bow**, not a sawtooth: a monotone 5.94 cm excursion out and back,
  with only 0.26 cm rms of jitter riding on it. Round 7 more than halved the bow
  (5.94 → 2.52 cm) but raised the jitter (0.258 → 0.342 cm rms). The pre-round-5
  segment had neither (1.005 / 1.021, jitter rms 0.081 cm).
* **53427 and 54351 are jitter**, and their bows are physical: 53427's 10.80 cm
  bow is spread over a 190 cm track and costs only 1.5 % of path length — that is
  a real curving track, not a defect.

**Why this changes the fix.** `multi_trajectory_fit` sets
`init_ps.front()/back()` from `start_v->fit().point` / `end_v->fit().point`
(`:4246-4247`) and then pushes those same vertex points into `final_ps` without
ever calling `fit_point` on them (`:4253-4259`). The endpoints are *pinned*. With
the main vertex 3–5 cm off the charge ridge (§4.2, top-right panel), the
trajectory has no choice but to bow out to reach it. **Interior-only smoothing
with pinned endpoints cannot fix 57903** — it would produce a cleaner curve that
still has to arrive at the same displaced point. Fixing the vertex can.

The prototype pins endpoints the same way (`multi_track_fitting.h:369-380`), so
this is shared behaviour, not a porting divergence.

### 4.8 Why round 6 and not round 5 — the two penalties price different things

Both rounds reweight edges of the same lazily-built `"steiner_graph_gap"` flavor
consumed only by `do_rough_path` (`clus/src/NeutrinoSteinerGapGraph.cxx`), and
the per-cluster build line in the event log says how many edges each one touched.
Cluster 14, event 57903:

| round | edges | scanned | penalized | of which weak-charge |
|---|---:|---:|---:|---:|
| 5 (`steiner_gap_penalty = 2.0`) | 1115 | 1033 | **119** (11.5 %) | — |
| 6 (+ `sgp_weak_scale = 5.0`, `qref = 6000`) | 1115 | 1033 | **311** (30.1 %) | 223 |

The two terms measure different things (`:191`, `:200`):

* **Round 5 — `gap_edge_bad_fraction`** samples the chord interior and asks
  whether each sample point is *supported in 2-D*: `classify_point` (`:68-80`)
  calls `grouping->test_good_point`, and returns "unsupported" only if no plane
  has charge there. **An isochronous ghost ribbon is by construction fully
  supported in 2-D everywhere** — that is exactly what makes it a ghost. So
  `bad = 0` throughout the ribbon and round 5's penalty is a *no-op inside it*.
  The 119 penalized edges must lie elsewhere in the cluster, and indeed round 5
  moved the vertex 15 cm without moving the corridor at all (§4.3).
* **Round 6 — `weak_charge_deficit`** (`:105-111`) prices an edge by the charge
  at its two *endpoints* against `qref`:
  `0.5·[max(0, 1 − q_a/q_ref) + max(0, 1 − q_b/q_ref)]`. That **does**
  discriminate between routes inside the ribbon, because a ribbon's charge is not
  uniform — it has a bright core and a broad low-charge ghost fringe. Cluster
  14's own numbers: the log's steiner-vertex quantiles are
  q25 = 6802, q50 = 10926, q75 = 15386 against `qref = 6000` (so `qref` sits near
  the 20th percentile, and 223 of 1033 scanned edges = 22 % are weak); and image
  charge per point in the isochronous stretch has q25 = 3228 / median = 10767
  versus q25 = 4702 / median = 7068 in the rest of the cluster — a wider
  distribution with a much heavier low-charge tail, over 112 image points per
  drift slice versus 7.2.

So round 6 is the first round whose penalty can re-route the path *within* the
isochronous ribbon, and inside a ghost ribbon charge is not a reliable guide to
which route is the real track — the fringe carries genuine reconstructed charge
too.

**Status of that last sentence: it is a mechanism hypothesis consistent with the
measurements, not a proof.** What is measured is: which round moved the corridor
(round 6, §4.3), how many edges each round penalized, what the two penalty
formulas are, and the charge distributions above. What is *not* measured is where
the penalized edges sit. Closing it needs a per-edge sentinel from the
`ensure_steiner_gap_graph` scan (source/target position, `bad`, `deficit`) — a
one-line log addition, and the natural companion to F1.

## 5. The owner's prototype question: is there special code in WCP?

**No — not in the fit.** `grep -in 'isochronous|prolonged|flag_para|degenerate|drift_dir'`
returns **0 matches in each** of the four prototype fitting files:

```
prototype_base/pid/src/PR3DCluster_trajectory_fit.h        (2166 lines)
prototype_base/pid/src/PR3DCluster_multi_track_fitting.h   (1837 lines)
prototype_base/pid/src/PR3DCluster_dQ_dx_fit.h             (1123 lines)
prototype_base/pid/src/PR3DCluster_multi_dQ_dx_fit.h       (1085 lines)
```

Everything WCP names "isochronous" sits **upstream** of the fit, in pattern
recognition and graph connection:

| prototype site | what it does |
|---|---|
| `PR3DCluster_graph.h:1445-1520` | `search_for_connection_isochronous` — anisotropic metric tolerating displacement along the degenerate direction |
| `NeutrinoID_proto_vertex.h:1597-1694` | `modify_segment_isochronous` — re-root a segment/vertex when near-isochronous |
| `NeutrinoID_proto_vertex.h:1696-1770` | `modify_vertex_isochronous` |
| `NeutrinoID_proto_vertex.h:1410-1462` | the call sites, gated on `|angle(dir, drift) − 90°| < 15°` |
| `PR3DCluster_path.h:288-315` | `get_local_extension` bails out entirely within 7.5° of isochronous |
| `PR3DCluster_graph.h:220-255, 1310-1327` | Hough radius widened 10 → 50 → 80 cm near-isochronous |

WCT already carries the first three of these
(`clus/src/NeutrinoOtherSegments.cxx:721`, gated by `iso_snap_min_dir_mag`, SBND
production 4.0 cm — doc pr/67 round 3).

Inside the prototype fit there are only two relatives, and neither is isochronous
handling:

* an explicit prior-toward-initial-position term, `PMatrix`, which was **written
  and then commented out** — `// + PMatrixT * pos_3D_init` and
  `// + PMatrixT*PMatrix` at `multi_track_fitting.h:690, 692` and
  `trajectory_fit.h:302, 304`. WCT never ported it at all: `grep -r PMatrix clus/`
  returns 0 lines. So this is **new code in both trees, not a re-enable**;
* `cal_compact_matrix{,_multi}`, which detects projection degeneracy by counting
  3-D points sharing a 2-D pixel — the one place in WCP that finds the
  isochronous condition without an angle test — and it is wired only into the
  **dQ/dx** fit, not the position fit.

Per M15: anything built here is an **addition**, not a port, and should not cite
prototype parity as justification.

## 6. Proposed fixes — shape and gate, none implemented

Ranked by what the evidence above supports. **The ranking is split by §4.7**:
57903 is a bow forced by a pinned endpoint and only F3 can reach it; 53427 and
54351 are jitter and dQ assignment, which F2 and F5 can reach. There is no single
fix for all three.

### F1 — segment-level trajectory-smoothness probe, log-only, default OFF

Emit per segment: chord, path, `path/chord`, fold-back fraction, angle out of the
drift-⊥ plane, drift-slice occupancy, and — the important one — a counter for how
often the triangle-area test was **shielded by `area2`**, i.e. `area1 >
area_ratio1 * c` held but `area1 > area_ratio2 * area2` did not. That counter is
the direct test of §3.4's claim, which is currently an inference from the code
rather than a measurement. Emit `ratio_bow` and `ratio_jit` (§4.7) separately,
not just `path/chord`. Zero footprint; this should come first regardless of what
follows.

### F3 — the fix for 57903: the seed, i.e. where the vertex sits

*Promoted above F2 by §4.7.* The endpoints are pinned to the vertex fit points,
so a vertex off the charge ridge forces the bow, and the bow is 1.225 of 57903's
1.347. §4.3 shows the smooth answer survived pr/51 round 5 intact and was lost in round 6
to a vertex move that bought no charge coverage either locally or globally.

Two sub-options, in increasing order of intrusiveness:

* **F3a — add trajectory smoothness as a criterion in the near-vertex graph
  work.** A candidate vertex that turns a near-collinear junction into a tight
  hairpin *and* raises the incident segments' `ratio_bow` without raising local
  charge coverage is a bad candidate. This is the honest fix and it lives where
  the decision is made.
* **F3b — let the endpoint move.** Unpin `init_ps.front()/back()` for segments
  flagged isochronous and let `fit_point` place them, then re-seat the vertex.
  Much more invasive: it changes the vertex, hence downstream selection.

Both are entangled with pr/51 round 6, which is production and was validated
on other events. Flagged as an **owner decision**, with F1's numbers as the input.

### F2 — seed-independent smoothing on isochronous *segments*

The criterion is per-**segment** (§4.6), not per-cluster: chord within
`traj_iso_angle` (10°) of the drift-⊥ plane **and** chord ≥ `traj_iso_min_len`
(10 cm). On such a segment, smooth only the transverse in-plane coordinate,
keeping the drift coordinate (which is well determined — 0.05 cm rms) and the
along-track parameterisation, then re-run `dQ_dx_multi_fit` on the smoothed path.

**Scope, stated honestly:** with endpoints pinned this addresses `ratio_jit`, not
`ratio_bow`. It is therefore a fix for the 53427/54351 shape (jitter 1.037–1.040,
rms 0.16–0.31 cm) and for the residual jitter round 7 left on 57903 (0.342 cm
rms) — **not** for 57903's 5.94 cm bow, which needs F3.

Two further constraints from §4.2:

* the acceptance test cannot be "is it straighter" — a straight chord between the
  current endpoints *loses* charge here (23.1 % vs 36.6 %). It must be a
  charge-coverage test: accept the smoothed path only if the charge within a
  fixed tube does not drop;
* the pre-round-5 arm sets the target: jitter rms 0.081 cm is what a well-seeded
  isochronous segment looks like in this same detector and event.

All knobs default OFF ⇒ key-suppressed in jsonnet ⇒ byte-identical when off.
Gate: the standard three-manifest off-gate, a knob-on census with a per-fire
sentinel, before/after Bee.

### F4 — anisotropic prior in `fit_point` (the never-ported `PMatrix`)

Add `A += PᵀP`, `b += PᵀP·p_seed` with `P` strong only along the degenerate
direction. **Ranked last on purpose:** it pulls toward the seed, and §4.3 shows
the seed is where the zigzag comes from, so it can only damp jitter the fit adds
on top. Worth keeping on the list because it is cheap and because F1 might show
the fit *does* add meaningful jitter.

### F5 — separate item: the dQ collapse at fold-back kinks (§4.4)

Instrument `calculate_compact_matrix_multi` and the `MU/MV/MW` rewrite: log, per
point, the overlap fractions and the diagonal it lands on, then check whether the
53427/54351 dips coincide with the overlap term firing. Independent of F1–F4 and
independent of isochronicity — the census says fold-backs are endemic
everywhere.

## 7. What this doc does not establish

* Whether the fit adds jitter of its own on top of the seed. F1 measures it; §4.3
  only shows the seed dominates the *event-to-event* variation. §4.7 gives the
  jitter its own number (rms 0.08–0.34 cm) but does not attribute it.
* Where exactly the bow/jitter boundary sits. The degree-4 polynomial of §4.7 is
  a working split, not a physics model, and **it is not degree-independent**:
  sweeping the degree 2 / 4 / 6 gives `ratio_bow`/`ratio_jit` of
  1.010/1.044 → 1.015/1.040 → 1.017/1.037 for 53427 and
  1.004/1.038 → 1.005/1.037 → 1.007/1.035 for 54351 (stable), but
  **1.036/1.272 → 1.225/1.099 → 1.216/1.107 for 57903** — a quadratic cannot
  represent the excursion at all and dumps it into "jitter". Degrees ≥ 4 agree.
  The bow claim for 57903 does *not* rest on the polynomial: it rests on the raw
  transverse profile (§4.2 table — a monotone 0 → −6.8 → 0 cm excursion with
  within-bin spread ≤ 1 cm) and on the endpoint pinning at `:4246-4259`, neither
  of which involves a fit.
* Whether the compact-matrix overlap term is what collapses dQ at fold-backs
  (§4.4). Named as a suspect from code reading only.
* Whether the round-5/6 vertex on 57903 is physically wrong. This doc measures
  trajectory smoothness and charge coverage; it does not adjudicate the vertex.
  Doc pr/51 §"footprint" already records 57903 as a round-7 mover.
* Anything about the other 49 events' *correctness* — the census measures
  smoothness, not truth.

## 8. Status

* Diagnosis only. No code, no config, no gate.
* Records committed: `73_evt57903_zigzag.png`, `73_anatomy_output.txt`,
  `73_census_output.txt`, `73_zigzag_census_r7on50.tsv`, and the two scripts
  under `scripts/analysis/pr73/`.
* Related: doc pr/67 (isochronous coverage, `iso_snap_min_dir_mag`), doc pr/51
  rounds 5–7 (the knobs in §4.3), docs pr/49–50 (`fit_blob_coverage` — the
  precedent for a knob living inside the trajectory fit), doc pr/28 round 10
  (`skip_revert_iso_xext_cut`, §4.6).

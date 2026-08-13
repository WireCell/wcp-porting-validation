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
pr/51 rounds 5–6 and the smooth answer that existed before them was lost (§4.3).

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

Read together:

* Before pr/51 round 5, this region was **one smooth 48 cm segment at 0.6° from
  isochronous with `path/chord` 1.026** — *more* isochronous than today's
  and smooth. **Isochronicity alone therefore does not predict the zigzag.**
  The seed does.
* Round 6 moved the main vertex 30.2 cm from its pre-round-5 position (43.9 cm
  from round 5's) and converted a near-collinear vertex into a 27° hairpin; both
  resulting legs zigzag (1.35 and 1.21). Round 7 opened
  the hairpin to 69° and halved the excursion (6.85 → 3.93 cm) but did not
  remove it.
* **Charge coverage is flat across all four arms** (44.2–44.9 % within 1.5 cm).
  Whatever the topology change bought elsewhere, on this event it bought no
  coverage and cost trajectory smoothness.

**This is a regression signal on shipped, owner-flipped SBND production knobs**
(`steiner_gap_penalty`, `sgp_weak_scale`, doc pr/51 rounds 5–6), stated here
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

Ranked by what the evidence above supports.

### F1 — segment-level trajectory-smoothness probe, log-only, default OFF

Emit per segment: chord, path, `path/chord`, fold-back fraction, angle out of the
drift-⊥ plane, drift-slice occupancy, and — the important one — a counter for how
often the triangle-area test was **shielded by `area2`**, i.e. `area1 >
area_ratio1 * c` held but `area1 > area_ratio2 * area2` did not. That counter is
the direct test of §3.4's claim, which is currently an inference from the code
rather than a measurement. Zero footprint; this should come first.

### F2 — recommended: seed-independent smoothing on isochronous *segments*

The criterion is per-**segment** (§4.6), not per-cluster: chord within
`traj_iso_angle` (10°) of the drift-⊥ plane **and** chord ≥ `traj_iso_min_len`
(10 cm). On such a segment, smooth only the transverse in-plane coordinate,
keeping the drift coordinate (which is well determined — 0.05 cm rms) and the
along-track parameterisation, then re-run `dQ_dx_multi_fit` on the smoothed path.

Two constraints from §4.2 that the implementation must respect:

* the acceptance test cannot be "is it straighter" — a straight chord between the
  current endpoints *loses* charge here. It must be a charge-coverage test:
  accept the smoothed path only if the charge within a fixed tube does not drop;
* the endpoints may themselves need to move, which reaches back into the vertex,
  so the first version should smooth the interior only and leave endpoints pinned.

All knobs default OFF ⇒ key-suppressed in jsonnet ⇒ byte-identical when off.
Gate: the standard three-manifest off-gate, a knob-on census with a per-fire
sentinel, before/after Bee.

### F3 — the source: revisit the isochronous seed

§4.3 says the smooth answer for 57903 existed before pr/51 rounds 5–6 and was
lost to a vertex move that bought no charge coverage on this event. Adding
trajectory smoothness (or slice occupancy) as a *criterion* in the near-vertex
graph work would be the honest fix rather than repairing the symptom downstream.
It is also the most entangled: those rounds are production and were validated on
other events. Flagged as an owner decision, not proposed.

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
  only shows the seed dominates the *event-to-event* variation.
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

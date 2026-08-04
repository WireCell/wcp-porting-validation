# doc pr/28 — vertex fit / trajectory fit / dQ-dx fit: prototype-fidelity audit

**Why.** The owner observed that the **SBND neutrino vertex position looks a bit
off**, and asked first to check one specific recalled prototype behaviour —
*after the 3-D vertex fit, the vertex is held fixed and only the other
trajectory points are fitted* — then widened it to *"anything weird on vertex
fitting, track trajectory and dQ/dx fitting compared to prototype code?"*

**Status.** Audit + measurement, and — after the owner read it and asked for
exactly these two — **§3.1 and §3.2 are now FIXED in the toolkit**, unconditionally
(no knob: they are port-fidelity bugs, not legacy behaviour to preserve). See
**§7** for the change and its measured effect on evt 388. Everything else below
is still reported, not fixed.

**Headline.** The vertex-fixing mechanism the owner asked about is a **faithful
port and it fires on SBND** — so it does *not* explain an off vertex (§1, §2).
The audit found the vertex fit was reading the **wrong point cloud** (§3.1) —
now fixed, and on evt 388 it moves the neutrino vertex **0.765 cm**. It also
turned up one confirmed behaviour-changing divergence in the **dQ/dx** fit
(§4.1) and a missing uBooNE calibration chain (§4.2); neither moves the vertex
directly, both reach it only through PID. Those two remain open.

---

## Repro

```bash
# Trees this doc was verified against
cd /nfs/data/1/xqian/toolkit-dev/toolkit && git rev-parse --short HEAD
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img && git rev-parse --short HEAD
# prototype_base -> /nfs/data/1/xqian/prototype-dev/wire-cell/  (package pid/, WCPPID)

# The measurement in section 2 (fresh dir; campaign dirs untouched)
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
SBND_WCT_LOGLEVEL=trace PR_EXTRA_STAGES=pr_display PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-vtxfit-388 data 388
grep -E "improve_vertex: cluster 23 (fitting vertex|fit_vertex)" \
  work-vtxfit-388/pr_evt388/wct_pr_evt388.log
```

`SBND_WCT_LOGLEVEL` is an existing runner env (`run_pr_chain_batch.sh:306`) —
no runner change was needed.

---

## §1 The question asked: is the vertex held fixed? — **yes, faithfully**

Prototype `ProtoVertex::flag_fit_fix` ↔ toolkit `PR::Fit::flag_fix` on
`Vertex::fit()`. Mechanism for mechanism:

| behaviour | prototype (`pid/`) | toolkit (`clus/`) |
|---|---|---|
| flag storage | `inc/WCPPID/ProtoVertex.h:75` | `PRCommon.h:126`, accessors `PRVertex.h:73-74` |
| **set true — the only site** | `src/NeutrinoID_improve_vertex.h:973` (`MyFCN::UpdateInfo`) | `clus/src/MyFCN.cxx:471` (`UpdateInfo`) |
| don't snap the fit point back to wcpt | `PR3DCluster_multi_track_fitting.h:9` | `TrackFitting.cxx:7778-7783` |
| **skip `fit_point` for the vertex** | `:328-333` | `TrackFitting.cxx:3839-3842` |
| segments copy the endpoint, never re-solve it | `:361-392` | `TrackFitting.cxx:3950-3963` |
| path organizers anchor to a fixed vertex | `:1425,1448,1601,1628` | `TrackFitting.cxx:1379/1389, 1532/1561, 1679/1708` |
| save/restore across `reset_fit_prop()` | `:178-180, :732-734` | `TrackFitting.cxx:3089-3091, 8246-8248` |
| cleared for a brand-new vertex | `ProtoVertex.cxx:14`, `ProtoSegment.cxx:1773` | `Fit::reset()` `PRCommon.h:136` |

It is a **hard exclusion from the unknowns**, not a soft pin. `fit_point`
(`TrackFitting.cxx:3415`) is an independent 3×3 normal-equation solve per point
in both trees — no coupled system, no Lagrange rows — and the
pin-to-initial-position (`PMatrix`) terms are commented out in both. Segments
meeting at a vertex share **one** unknown by construction via a single
`fit_index` (`TrackFitting.cxx:3222-3238`).

**No vertex is ever spuriously fixed.** The discriminating check is a grep for
*all* writes including through references (an enumeration of `->fit(...)` call
sites misses `auto& f = vtx->fit(); f.flag_fix = true;`):

```
grep -rn "\.flag_fix\s*=\|flag_fix(" clus/src clus/inc
```

The only `= true` reachable on a **vertex** is `MyFCN.cxx:471`.
`TrackFitting.cxx:3222/3235` and `PRSegment.cxx:34` write *segment* fits, and
the `PRVertex.h:73` setter has exactly two callers (`:3091`, `:8248`), both
save/restore pairs.

### Divergences in this area — none

Two things that *looked* like toolkit-only divergences on a first pass are not.
Both are recorded here because each cost a wrong conclusion:

1. **`flag_fix` on segment endpoint fits is a faithful port, not an extra.**
   `form_map_graph:3222/3235` sets it `true` on each segment's first and last
   fit — and so does the prototype, as `saved_skip`:
   `PR3DCluster_multi_track_fitting.h:879` and `:887` push `true` for the first
   and last point, `:873` pushes `false` for interior, and `:895`
   `set_fit_associate_vec(saved_pts, saved_index, saved_skip)` stores them.
   Consumed identically on both sides (`:367/:381` ↔ `TrackFitting.cxx:3944/:3966`).
   **Do not conclude "all-false" from `ProtoSegment.cxx:1009`** — that
   `resize(..., false)` is `clear_fit`'s *initialisation*, overwritten by
   `form_map` on the next pass. That is exactly the mistake made here first.
2. **`check_and_reset_close_vertices` is not toolkit-only.** It is an extraction
   of the prototype's inline block — `PR3DCluster_multi_track_fitting.h:1383-1392`
   (`_2nd`) and `:1158-1167` (`_3rd`) — with the same 0.01 cm threshold, the same
   degree-1 condition, and the same two call sites. Only its *tail* diverges,
   see §3b D5.

Separately, all eight vertex-`Fit` assignment sites (`TrackFitting.cxx:1216/1221,
1383/1393, 1555/1584, 1702/1731`) are copy-then-mutate of the vertex's **own**
fit, so a segment fit is never copied onto a vertex and the endpoint flags cannot
leak into vertex fixing.

### Two things that look like bugs and are not

* The second `fit_vertex` at `NeutrinoVertexFinder.cxx:2098` re-solves an
  already-fixed vertex. The prototype does the same — `FitVertex` never checks
  the flag. The vertex is fixed *with respect to the trajectory fit*, not
  globally.
* `do_single_tracking` / `trajectory_fit` never consult `flag_fix`, in **either**
  tree.

### Ruled out as a cause of a misplaced *main* vertex

The 0.6 cm outward endpoint extension in `organize_segments_path{,_2nd}`
(`TrackFitting.cxx:1535`, `:1564`) fires only for unfixed vertices of
**degree 1** (`:1499-1500`). It cannot move a multi-leg interaction vertex.

---

## §2 Measurement — the fit does fire on SBND (evt 18255/388)

Code path ≠ firing rate, so this was measured rather than assumed. Full ordered
outcome from the trace log:

```
pass 1  improve_vertex: flag_search_vertex_activity=false flag_final_vertex=false
        fitting vertex (-163.08, 32.19, 426.52) nsegs=3
        fit_vertex done, vertex moved 0.693 cm -> (-162.45, 31.93, 426.37)
        second fit_vertex done -> (-162.45, 31.93, 426.37)
        fitting vertex (-168.70, 32.19, 441.22) nsegs=3
        fit_vertex done, vertex moved 1.039 cm -> (-168.70, 31.67, 442.12)
        second fit_vertex done -> (-169.02, 31.67, 442.72)
pass 2  improve_vertex: flag_search_vertex_activity=true flag_final_vertex=true
        fitting vertex (-162.45, 31.93, 426.37) nsegs=3
        fit_vertex done, vertex moved 0.000 cm -> (-162.45, 31.93, 426.37)
        fitting vertex (-169.02, 31.67, 442.72) nsegs=3
        fit_vertex made no update
```

| | count |
|---|---|
| fit attempts | 4 |
| fitted, `UpdateInfo` ran ⇒ `flag_fix = true` | 3 |
| `MyFCN` solver failures (`Fit Vertex Failed`) | 0 |
| skipped by SBND's `fit_vertex_min_seg_length` | **0** |

**The main vertex was fitted and is fixed.** It is the first one: dump
`main_vertex` = (−162.53, 32.00, 426.17), `wcpt` = (−162.45, 31.93, 426.37).
It moved **0.693 cm** on pass 1, was re-tracked and re-fitted (the >0.5 cm
branch), and re-fitting it on the final pass moved it **0.000 cm** — i.e. it is
sitting at the `MyFCN` optimum, exactly as a fixed vertex should.

**SBND's `fit_vertex_min_seg_length = 1.0` cm never fired here.** All 18 logged
candidate segments are ≥ 2.11 cm (`fit_vertex: cluster 23 candidate segment
wcpt_len=… (cut 1.00 cm)`), and there are zero `skipping vertex fit` and zero
`excluding … short segment` lines. So on this event the knob is inert. It
remains a live suspect on *other* events — the trace lines above are how to
check one.

The single non-fit (pass 2, the second vertex) came from the **`MyFCN.cxx:220`
gate** — `(ntracks > 2 && n_large_angles > 1) || (ntracks >= 2 &&
enforce_two_track_fit && n_large_angles >= 1)`, large angle = >15°. It was not
the length cut (0 skips) and it was **not** the charge veto: the veto
(`NeutrinoVertexFinder.cxx:1995-2001`) never returns false — it reverts
`results.second` to the old position and still calls `UpdateInfo`, so a
charge-vetoed vertex is still *fixed*, just at its old location. That vertex had
already been fixed on pass 1, so it retained `flag_fix` regardless.

> **Correction to doc pr/27 §12.** Its "three reasons a vertex does not move" is
> accurate as written, but do not read reason 3 as a reason the vertex is not
> *fixed*. The charge veto blocks the *move*, not the fix.

---

## §3 Vertex fit internals (`MyFCN`) — **two confirmed defects that move the vertex**

§1 answered "is the vertex *held* fixed". This section audits the fit that
*produces* the position, and it is where the audit found real problems. Both
were re-verified by hand in both trees.

Everything else in `MyFCN` matches: constructor constants
(`0.43 / 1.5 / 0.9 / 6 cm`), the annulus and its `length > 3 cm` short-track
branch, the `(0.15 cm)²` eigenvalue floor, the descending-eigenvalue convention,
`center` = nearest surviving point, the normal equations (row 0 zeroed,
rows 1–2 weighted `sqrt(λ0/λk)`), the isotropic prior, `BiCGSTAB` +
`solveWithGuess` with the `isnan(error())` gate, `default_dis_cut = 4 cm`, and
the 5000/8000 e⁻ charge-veto constants (FP-exact via `43000 × 5/43`, `× 8/43`).
The `improve_vertex` call sequence was compared step by step: **no reordering,
no missing step, no extra step.**

### §3.1 **The PCA is computed on the wrong point cloud** (headline)

```cpp
// prototype  NeutrinoID_improve_vertex.h:538
WCP::PointVector& pts = sg->get_point_vec();
//               ProtoSegment.h:16 ->  { return fit_pt_vec; }      <- FITTED points
```
```cpp
// toolkit  clus/src/MyFCN.cxx:57-58
// Get raw steiner points from segment (consistent with prototype's get_point_vec())
const auto& wcpts = sg->wcpts();                                 // <- RAW STEINER path
```

**The comment is wrong.** `get_point_vec()` returns `fit_pt_vec`, the fitted
trajectory cloud; the raw Steiner path is a *different* member exposed as
`get_wcpt_vec()` (`ProtoSegment.h:15`). So the prototype fits the vertex against
the **smooth fitted trajectory**, while the toolkit fits it against the
**blob-quantised Steiner skeleton**.

The toolkit already has the right object — `Segment::fits()`
(`PRSegment.h:85-88`), whose `m_fits` is resized from `m_wcpts` in `clear_fit`
exactly as the prototype resizes `fit_pt_vec` from `wcpt_vec`. `MyFCN` simply
does not use it.

One line, but it cascades through everything the fit depends on: which points
survive the annulus, `nsum`, `center` (the nearest surviving point — the fit's
anchor), the covariance and therefore **all three eigenvalues and eigenvectors**,
and `length`, which can straddle the 3 cm short-track branch differently.

**This is the most plausible mechanism found for a systematically-off vertex**:
the transverse constraint directions and the anchor point are derived from a
skeleton whose nodes sit at blob centres, not from the trajectory the rest of the
chain uses.

### §3.2 Degenerate legs open the fit gate that should stay closed

A segment with ≤1 point surviving the annulus pushes a `(0,0,0)` PCA direction
placeholder (`MyFCN.cxx:96-116`). Both trees then loop over *all* entries when
counting large angles — but they disagree on what a null vector means:

```cpp
// prototype :688   ROOT TVector3::Angle -> if (ptot2 <= 0) return 0.0;  and clamps arg to [-1,1]
if (dir1.Angle(dir2)/3.1415926*180. > 15) n_large_angles++;
// toolkit MyFCN.cxx:214
double angle = std::acos(dir1.dot(dir2)) * 180.0 / M_PI;   // acos(0) = 90 degrees
if (angle > 15) n_large_angles++;                          // ... so it INCREMENTS
```

`std::acos(0)` is 90°, so every degenerate leg contributes a spurious "large
angle" and pushes `n_large_angles` up.

**Reachability — stated precisely, because the naive version overclaims.** A
stub does *not* also inflate `ntracks`: the PCA gate (`MyFCN.cxx:100`) and
`get_fittable_tracks` (`:184`) use the **same** `vec_points[i].size() > 1`
predicate, so a leg with ≤1 surviving point is excluded from `ntracks` while
still occupying a slot in `vec_PCA_dirs`, which the angle loop iterates in full.
The divergence therefore needs a stub *plus* enough real legs to satisfy the
`ntracks` half of the gate. Two cases:

* **Main vertex** (`enforce_two_track_fit = true`, set only for the main vertex,
  `NeutrinoVertexFinder.cxx:1953`) — gate is `ntracks >= 2 && n_large_angles >= 1`.
  **Two near-collinear fittable legs plus one stub is enough**: the prototype
  counts 0 large angles (the real pair is collinear, and `Angle()` returns 0 for
  the stub pairs) and declines; the toolkit counts 2 × 90° from the stub pairs
  and fits. This is a straight track with a stub at the vertex — a common
  topology, and it is the *main* vertex, which is what the off-vertex report is
  about.
* **Any other vertex** — gate is `ntracks > 2 && n_large_angles > 1`, so it needs
  ≥3 fittable legs, mutually near-collinear enough that the prototype counts ≤1
  large angle among the real pairs, plus ≥1 stub. Narrower, but reachable.

So the accurate statement is: **for the main vertex, a stub leg alone can open a
gate the prototype keeps closed.** This is live in exactly the geometry
`m_fit_vertex_min_seg_length` was invented to fight — a sub-cm vertex-activity
stub has all its points inside the 0.9/1.5 cm inner radius, so it yields a zero
direction.

The PCA axes *are* explicitly unit-normalised (`MyFCN.cxx:141-147`), so `dot` is
a true cosine for real segments — but there is **no `[-1,1]` clamp**, which ROOT
also provides. On near-collinear pairs `|dot|` can exceed 1 by an epsilon,
giving `NaN`; `NaN > 15` is false, so that one silently *misses* an increment.

### §3.3 Other vertex-fit divergences

| what | prototype | toolkit | severity |
|---|---|---|---|
| **`UpdateInfo` → `clear_fit` also wipes PID/direction**: toolkit's `Segment::clear_fit` additionally resets `m_dirsign=0`, `m_dir_weak=false`, `m_particle_score=100`, `m_particle_info=nullptr` | `ProtoSegment.cxx:1012-1040` (touches none of these) | `PRSegment.cxx:80-118`, called `MyFCN.cxx:460` | **behaviour-changing** — every vertex fit silently discards PID on all attached legs |
| **Silent no-op reported as success**: three early `return`s in `UpdateInfo` (fit_pos outside a DV; missing wpid offsets; missing `steiner_pc`) abort before any write, yet `fit_vertex` still returns `true` | no counterpart | `MyFCN.cxx:294-298, 310-313, 351-355` | **behaviour-changing** — `improve_vertex` then re-tracks and may re-fit on an unmoved vertex |
| `flag_front` criterion: index identity vs distance comparison (the distance form is present in the prototype but **explicitly commented out**) | `:857-862` | `MyFCN.cxx:373-376` | behaviour-changing |
| charge-veto sampling radius | 0.3 cm (the default) | **0.6 cm** passed explicitly | behaviour-changing (different charge integral, same veto logic) |
| `+0.5` half-wire offset on stored `pu/pv/pw/pt` | present `:971` | dropped `:466-469` | benign *if* consumers agree — it is a global port convention (`TrackFitting.cxx:1315` also drops it) |
| identity-by-index → identity-by-0.01 cm distance throughout `UpdateInfo` | `:900-935` | `MyFCN.cxx:417-451` | forced — toolkit `WCPoint` has no index field (`PRCommon.h:96-99`); two Steiner points within 0.01 cm would alias |
| `m_fit_vertex_min_seg_length` | **no counterpart** (grep: zero hits) | `NeutrinoPatternBase.h:203` | toolkit-only; inert at its `0` default, but **SBND sets 1.0** |
| second-loop `vertex_segments` refresh uses unsorted `boost::out_edges` where the first loop uses `sorted_out_edges` | — | `NeutrinoVertexFinder.cxx:2222-2229` | benign today (feeds only a `size()==3` test) — but it is an iterated unordered edge set; keep it off any path that affects output |

## §3b Trajectory fitting

The skeleton is faithful. **MATCH**, checked rather than assumed: the three-pass
structure (`organize_segments_path → form_map → multi_trajectory_fit` ×2, then
`organize_segments_path_3rd`), the pass parameters at default config
(`low_dis_limit = 1.2 cm`, `end_point_limit = 0.6 cm`, halved on pass 2,
`0.6 cm` on pass 3, `div_sigma = 0.6 cm`, `charge_div_method = 1`), `fit_point`'s
`RU/RV/RW` construction and charge weighting, the `charge < 100 → (100,1000)`
clamp, the `quantity < 0.5` down-weight, `BiCGSTAB` + `solveWithGuess` with the
`isnan` fallback, the commented-out `PMatrix` pin in **both**, `form_map`'s
`end_point_factor 0.6 / mid_point_factor 0.9 / nlevel 3 / charge_cut 2000`, the
time cut (prototype `time_cut = 5` slices ↔ toolkit `time_tick_cut = 20` ticks,
reconciled at `TrackFitting.cxx:2499-2506`), the endpoint pin and the
`examine_trajectory` smoothing windows, and the whole single-track path.

**The prototype does have a third path organiser** —
`PR3DCluster_multi_track_fitting.h:1100`, called at `:160`.

### How far to trust this section

Two tiers, following pr/27 §0's convention, because they are **not** equally
solid and a reader deciding what to fix must know which is which:

* **Independently re-read in both trees by the author**: everything in §1, §2,
  §3.1, §3.2, §3.3's first row, §4.1, §4.2, and **T5** below — plus the
  `saved_skip` MATCH that corrected §1.
* **Reported by an audit pass and *not* independently re-verified**: **T1, T2,
  T3, T4, T6, T7, T8**. The anchors and reasoning are recorded as given. Treat
  them as leads to confirm before acting, not as established the way §3.1 is.

That distinction is not pedantry: this document's first draft carried a
`saved_skip` "divergence" that a second look disproved (see §1). Re-read before
you fix.

### Behaviour-changing divergences

| # | what | prototype | toolkit | note |
|---|---|---|---|---|
| **T1** | **The charge-ratio veto in `skip_trajectory_point` is dead in the multi-track path.** `pss_vec` is filled from `final_ps_vec` (already-fitted points), so the comparison point *is* the point under test ⇒ `ratio == 3`, `ratio_1 == 1` exactly ⇒ the cut is unreachable and the `p = ps_point` revert is a no-op. | `:429` passes `init_ps_vec`; cut at `PR3DCluster_trajectory_fit.h:745-750` | `TrackFitting.cxx:4704`, consumed `:5041` | The **single**-track caller (`:4513-4520`) passes pre-fit points and *is* correct — same function, two callers, one right. That asymmetry is what makes this a port slip rather than a design choice. Constants (`0.97`, `0.75`, `160/90/45°`, `0.5 cm`) all match. |
| **T2** | `angle1` computed from fitted instead of initial points — same root cause as T1 | `PR3DCluster_trajectory_fit.h:768-769` | `TrackFitting.cxx:5182-5186` | |
| **T3** | **Dead-channel plane-quantity lookup uses the loop position, not the fit index.** `m_3d_to_2d.find(i)` where `i` is the per-segment point position, but the map is keyed by the **global** `count`. For every segment after the first this *hits a valid but wrong point* rather than missing. | `PR3DCluster_trajectory_fit.h:780-782` uses `init_indices.at(i)` | `TrackFitting.cxx:5100-5105` | The toolkit's `skip_trajectory_point` signature (`:4849`) carries no index parameter at all. |
| **T4** | **An extra `form_map_graph` runs before `dQ_dx_multi_fit`**, and it calls `set_fit_associate_vec`, which drops interior points whose summed plane quantity is 0 — i.e. it re-runs a point-dropping pass on the *final* post-`_3rd` trajectory, changing the output point count. | no `form_map` anywhere in `PR3DCluster_multi_dQ_dx_fit.h`; `:174-188` resets only | `TrackFitting.cxx:8257` | |
| **T5** | **Fixed vertices no longer get their projections refreshed.** The prototype's `vtx->set_fit(...)` sits *outside* the `if (!flag_fit_fix)`; the toolkit puts the whole `pu/pv/pw/pt/paf/index` update *inside* it. | `:333-342` (verified: `set_fit` outside the guard) | `TrackFitting.cxx:3842-3901` | A fixed vertex's 3-D point does not move, so its projections are only *stale* if the transform context changed — but the prototype also resets `dQ`/`dx`/`reduced_chi2`/`index` here on every pass and the toolkit does not. Effect is subtler than T1–T4; flagged, not ranked with them. |
| **T6** | `check_and_reset_close_vertices` **rebuilds the segment's fits to 2 points** (`segment->fits(generate_fits_with_projections(...))`); the prototype's inline equivalent resets only the two vertex fit points. | `:1383-1392` | `TrackFitting.cxx:1222-1227` | |
| **T7** | `charge_div_method == 2` missing-key fallback: toolkit prefills `1/N` and `continue`s on unknown wpid, leaving `1/N`; the prototype's `if/else if` leaves the key absent so `operator[]` gives `0.0` (zero weight). Reachable in the single-track 2nd pass. | `:249, :269` | `TrackFitting.cxx:3670-3690`, `:4077` | |
| **T8** | **`associated_2d_points` ordering differs**: `Coord2D` orders by `(apa,face,time,wire,channel,plane)`, the prototype's `pair<int,int>` by `(wire,time)`. Deterministic run-to-run, but it permutes the rows of `RU/RV/RW` and hence the FP accumulation order. | `:575,613,648` | `TrackFitting.h:282-290` | **Fidelity, not determinism** — it means the toolkit cannot be bit-identical to the prototype even with everything else fixed. |

### Benign / unknown

Start-end vertex identification by geometric distance rather than WCP index
identity (`TrackFitting.cxx:3123-3133`); a toolkit-only degenerate-collapse
fallback in `organize_segments_path_2nd` (`:1522-1527`); an extra 2-point-collapse
reset between the single-track passes (`:8377-8388`); a toolkit-only vertex guard
in `form_map_graph` (`:3274`) that covers a prototype degenerate case; no
counterpart for `collect_charge_multi_trajectory`; `RW` carrying a Y-term the
prototype omits (`:3612`, benign at `angle_w == 0`, which the toolkit asserts);
and the `+0.5` / plane-offset projection convention (D9 above), self-consistent
within each tree.

### Determinism

Clean, with one exception already known: `m_cluster_fitted_charge_2d`
(`TrackFitting.h:673-675`) is `std::map<Facade::Cluster*, …>` with the **default
pointer comparator** and is iterated at `:1139`, last-writer-wins. Everything
else checked out — `ordered_edges`/`ordered_nodes` are index-ordered, `m_clusters`
uses `PR::ClusterPtrCmp` (compares `get_cluster_id()`), `m_2d_to_3d`/`m_3d_to_2d`
are value-keyed, and `m_segments`/`m_blobs` are membership-tested but never
iterated. **The fix is one word**: give `m_cluster_fitted_charge_2d` the
`PR::ClusterPtrCmp` comparator that already exists and is already used for
`m_clusters`.

---

## §4 dQ/dx fitting

Structurally the port is close: same `cal_gaus_integral_seg` erf smearing with
`nsigma = 4` on all three planes, same charge whitening, same collection-plane
down-weighting (`rel_uncer_col` 0.05 vs 0.075, `add_uncer_col` 300 vs 0), same
`calculate_compact_matrix_multi` with `cut_pos` 3/3/2 for U/V/W, same
`Eigen::BiCGSTAB` + `solveWithGuess` with an `isnan(error())` fallback, same
`dx` construction with no minimum floor.

`reduced_chi2` matches **character for character**, including the deliberate
collection-plane `/4`:

```cpp
// prototype PR3DCluster_multi_dQ_dx_fit.h:859  ==  toolkit TrackFitting.cxx:6851
traj_reduced_chi2.push_back(sqrt((sum[0] + sum[1] + sum[2]/4.)/(sum1[0]+sum1[1]+sum1[2])));
```

A suspected divergence was **disproved**: `reduced_chi2` *is* written to
segment-interior points in both trees (`TrackFitting.cxx:6891-6893` in the multi
fit; `:8688-8702` in `do_single_tracking`). The commented-out block at
`NeutrinoPatternBase.cxx:809-812` is a **dead duplicate**, not a missing step —
and re-enabling it would be strictly worse, since it omits `index`, `range` and
`paf`. (This supersedes the "flagged, not fixed" note in doc pr/27 §2.)

### §4.1 CONFIRMED behaviour-changing: the multi-fit close weights were not scaled up

The prototype uses **different regularisation weights in the multi-track fit than
in the single-track fit**. The toolkit has only one set, and it is the
single-track one.

| | prototype single | prototype multi | toolkit (both fits) |
|---|---|---|---|
| `close_ind_weight` | 0.15 | **0.25** | 0.15 |
| `close_col_weight` | 0.45 | **0.75** | 0.45 |
| `lambda` | 0.0005 | **0.0008** | 0.0005 → ×8/5 = 0.0008 ✅ |

Anchors: prototype `PR3DCluster_dQ_dx_fit.h:870-874` (single) and
`PR3DCluster_multi_dQ_dx_fit.h:759-763` (multi), `:793` (lambda);
toolkit `TrackFitting.cxx:6738-6739` (multi reads `m_params`), `:7584-7585`
(single reads the same `m_params`), header defaults
`inc/WireCellClus/TrackFitting.h:94-95`, SBND config
`cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json:43-44,47`.

The prototype scales **both** knobs going single → multi: lambda ×8/5
(0.0005→0.0008) and the close weights ×5/3 (0.15→0.25, 0.45→0.75). **The port
carried the lambda scale-up and missed the close-weight one** — and the toolkit
comment at `TrackFitting.cxx:6779` (`// adjusted for multi-track fitting`) shows
the author knew the multi fit needs stronger regularisation. So the toolkit's
multi-track dQ/dx fit is under-regularised on the close-wire (overlap) term by a
factor 5/3 relative to the prototype.

**Not fixed — escalation rule 1** (it changes a constant, i.e. production output
unconditionally). Note also that there is currently **no way to express the two
separately**: one `set_parameter("close_ind_weight", …)` feeds both fits, so a
fix needs either a second parameter pair or an internal ×5/3 in
`dQ_dx_multi_fit` mirroring the existing lambda line.

**Does this explain the off vertex? Not directly.** `dQ_dx_multi_fit` is the
charge pass that runs *after* the three geometry passes; it does not move
trajectory points. It reaches the vertex only indirectly, through dQ/dx → track
PID (`segment_do_track_pid`) → direction → the §6 vertex score.

### §4.2 The uBooNE calibration chain is absent (flagged, not a defect claim)

Present in the prototype, with **no counterpart anywhere** in
`TrackFitting.cxx` (verified: `get_corr_factor`, `attenuation`,
`SCE_correction`, `PosEfield`, `/= 0.7` all return zero hits):

| prototype step | anchor |
|---|---|
| `/0.7` hardcoded U-wire dead-region patch (7 ranges) | `PR3DCluster_multi_dQ_dx_fit.h:875-882` |
| `get_corr_factor` YZ/position calibration | `:901-903`, `:929-931` |
| electron-lifetime attenuation | `:906-908`, `:933-935` |
| SCE correction on **dx** | `:211-223`, `:505-517` |
| SCE correction on final dQ/dx | `:942-952` |

These are uBooNE-specific calibrations (uBooNE YZ maps, uBooNE SCE map, uBooNE
lifetime), so **their absence may well be correct** for SBND rather than a
porting gap — that is the owner's call, not this document's. Two reasons it
still deserves a decision:

1. the SCE **dx** correction feeds `local_dx`, which feeds the regulariser — so
   it is not merely an output rescale;
2. track PID compares the measured dQ/dx profile against **stopping-particle
   templates** that came from uBooNE. If those templates presume a corrected
   dQ/dx and SBND supplies an uncorrected one, PID carries a systematic — which
   is a path from here to the vertex, via §5→§6.

### §4.3 Other dQ/dx divergences

| what | prototype | toolkit | severity |
|---|---|---|---|
| `connected_vec` end-vertex case pushes `indices.size()-2` (a **size** where an **index** belongs) | `PR3DCluster_multi_dQ_dx_fit.h:723` | `TrackFitting.cxx:6708` — correct index | **toolkit fixes a prototype bug** |
| vertex-adjacency `paf` guard: toolkit drops a neighbour on a different APA/face | none | `:6414`, `:6417` | behaviour-changing; multi-APA necessity, no prototype counterpart |
| skips trajectory points with `face == -1` | none | `:6103-6108`, `:6459-6464` | behaviour-changing in edge cases |
| smearing constants hardcoded to uBooNE literals in C++ | derived from `TPCParams` | header `:41-43,52` | benign for SBND — config overrides all four (`sbnd_track_fitting.json:7-9,14`); latent for any detector run without such a file |
| `pred <= 0` guards inside the chi2 sums | none | `:6837,6842,6847` | benign (prototype would give inf/NaN) |
| wire search window `\|w−c\|≤10` vs `round(c)±10` | `:371,390,411` | `:6288-6289` | benign (the `nsigma=4` gate dominates) |
| `assemble_fitted_charge_2d` iterates a **pointer-keyed** `std::map<Cluster*,…>`, last-writer-wins | no counterpart | `:1136-1152` | **toolkit-only nondeterminism**, already known — 10.2% of cells move between `setarch -R` runs (`PrDisplayDump.cxx:772-777`, doc pr/26 §5.2) |

### §4.4 Open question, not settled

The time-bin convention differs and could not be closed from the fitting code
alone. Prototype integrates `[tbin, tbin+1]` with `t_center` carrying `+0.5`
(`PR3DCluster_dQ_dx_fit.h:165-168`); the toolkit integrates
`[tbin − 0.5·nt, tbin + 0.5·nt]` with `tbin` a raw tick and no `+0.5`
(`TrackFitting.cxx:5162-5168`). The dead-channel snap at `:6361` implies stored
times are slice-grid multiples, i.e. slice *starts*, which would put the erf
window half a slice off centre. **To close it:** establish whether the
`time_slice` in `CoordReadout(apa, time_slice, channel)` (`:802`) is the slice
*start* tick or its *centre*. If start, the toolkit Gaussian is offset by half a
slice — behaviour-changing. If centre, the two agree.

---

## §5 What this means for the off-vertex worry

**Cleared** — not the explanation:

* the vertex-fixing mechanism (§1): faithful, and it fires (§2);
* the 0.6 cm endpoint extension: degree-1 only;
* `fit_vertex_min_seg_length` **on evt 388**: never fired.

**Prime suspects**, in order — and note the first two are in the vertex fit
itself, which is where the owner's instinct pointed:

1. **§3.1 — the `MyFCN` PCA runs on the Steiner skeleton instead of the fitted
   trajectory.** One line (`MyFCN.cxx:57-58`), directly biases the fitted vertex
   position, and the toolkit already holds the correct cloud in
   `Segment::fits()`. This is the single most likely cause of a systematically
   off vertex.
2. **§3.2 — degenerate legs count as 90° and open the fit gate.** The toolkit
   fits vertices the prototype declines to. Combined with §3.1, the vertex both
   moves more often than it should and moves to a slightly wrong place.
3. **§3.3 — `UpdateInfo` wipes PID/direction on every attached leg** via the
   toolkit's heavier `clear_fit`. That feeds §5→§6, i.e. *which* vertex is then
   chosen.
4. §3.3 — the "silent no-op reported as success" early returns.
5. `fit_vertex_min_seg_length = 1.0` on *other* events (SBND-only, doc pr/9);
   check with the §2 trace recipe on an event where the vertex looks wrong.
   Inert on evt 388.
6. `UpdateInfo` re-snapping `wcpt` to the nearest Steiner point
   (`MyFCN.cxx:474`) — the seed is quantised onto the skeleton, and the 0.22 cm
   `fit_distance` on evt 388's main vertex *is* that residual.
7. **§3b T1/T2 — the charge-ratio veto is dead in the multi-track path.** The
   guard that reverts a badly-fitted trajectory point to its pre-fit position
   never fires, because the point is compared against itself. Trajectory points
   feed the vertex through `organize_segments_path` and through `MyFCN`'s PCA.
8. **§3b T3 — the dead-channel lookup reads the wrong point** for every segment
   after the first, silently rather than by failing.
9. §3b T4 — an extra association rebuild drops points from the final trajectory
   before dQ/dx.
10. §4.1 / §4.2 via PID → direction → vertex score — indirect, but real.

Note that §3.1 and T1 compound: `MyFCN` builds its PCA from a point cloud whose
own quality guard (T1) is inoperative — except that under §3.1 it reads the
Steiner path rather than that cloud at all. Fixing either one alone changes what
the other sees, so they want separate gates and separate events.

**§3.1 and §3.2 are now fixed** — the owner read this section and asked for both
(§7). Items 3-10 remain open and are still escalation rule 1.

**The single most useful next step** is making `flag_fix` observable. It is the
one flag that answers "did the vertex fit run for this vertex", and today **no
artifact carries it** (doc pr/27 §14) — the only source is the trace log used in
§2. One JSON key in `PrDisplayDump` (which is default-OFF, so a diagnostic
schema change, not a physics change) would put it in the display.

## §6 Correction issued alongside this audit

*(this section predates §7 and is unaffected by it)*

`fit_distance` is **not** "how far the 3-D vertex fit moved the vertex", and 0
does **not** mean the fit did not run. It is `|fit().point − wcpt().point|`
(`PRVertex.h:84`): the trajectory fit moves `fit().point` for every *unfixed*
vertex, and `UpdateInfo` re-snaps `wcpt()` for every *fixed* one, so it is
nonzero either way — **127 of 127** vertices on evt 388, degree-1 track ends
included. Corrected in `pr_display/README.md`, doc pr/26 and doc pr/27's payload
tables. The viewer chip label is left to a follow-up (another session had the
file open).

---

## §7 §3.1 and §3.2 FIXED — the change and its effect on evt 18255/388

Owner instruction, after reading §3.1/§3.2 above: *"Can you fix these two? Double
check it against Prototype, and then redo for the event 388 ... No need to do
large A/B check. We just want to see its impact on this event first."*

**No knob.** Both are port-fidelity defects — the toolkit read the wrong array
and counted an angle the prototype does not count. A default-OFF knob would ship
the owner's requested fix as a no-op. CLAUDE.md §1's prime directive protects
*legacy behaviour the owner wants preserved*; escalation rule 1 was satisfied by
asking, and the owner answered. Both changes are unconditional, in one file.

### 7.1 The change — `clus/src/MyFCN.cxx`, toolkit commit `e6c51cc5`

**§3.1 — `AddSegment` now reads the fitted trajectory.**

```cpp
-    // Get raw steiner points from segment (consistent with prototype's get_point_vec())
-    const auto& wcpts = sg->wcpts();
+    const auto& fits = sg->fits();
```

Re-verified in both trees before editing: prototype `NeutrinoID_improve_vertex.h:538`
is `WCP::PointVector& pts = sg->get_point_vec();`, and `ProtoSegment.h:16` is
`std::vector<WCP::Point>& get_point_vec(){return fit_pt_vec;};` — the *fitted*
cloud. The toolkit counterpart is `Segment::fits()` (`PRSegment.h:85-88`),
populated by `multi_trajectory_fit` via `generate_fits_with_projections`
(`TrackFitting.cxx:1226/1477/1645/1787`).

**No fallback to `wcpts()` when `fits()` is empty**, deliberately. Every path
into `fit_vertex` runs `do_multi_tracking` first (`NeutrinoVertexFinder.cxx:497,
2042, 2099, 2115`), so a populated `fits()` is guaranteed by construction — the
prototype relies on the same guarantee harder still, since its `:539`
`pts.front()` on an empty `fit_pt_vec` is UB. A fallback would let a segment
silently revert to the old buggy behaviour and would be invisible in the
measurement. The pre-existing empty-guard is kept, now pointing at `fits()`: a
segment with no fit contributes a zero direction and nothing to the solve, and —
with §3.2 in place — nothing to `n_large_angles` either, exactly as the prototype
treats a null direction.

**Two-effect change, worth remembering.** `length` (`MyFCN.cxx:71-78`) is also
computed front-to-back over the same cloud, and it selects the 1.5 cm vs 0.9 cm
inner annulus radius at `:86`. Switching the cloud can therefore change the
*inner radius* as well as the points. This matches the prototype, which computes
`length` from the same `get_point_vec()` at `:539` — and both are straight-line
front-to-back distances, not path sums. If a future event shows a surprise, this
is the second place to look.

**§3.2 — the angle now follows ROOT's `TVector3::Angle` exactly.**

```cpp
-    double angle = std::acos(dir1.dot(dir2)) * 180.0 / M_PI;
+    const double ptot2 = dir1.squaredNorm() * dir2.squaredNorm();
+    double angle = 0.0;
+    if (ptot2 > 0) {
+        double arg = dir1.dot(dir2) / std::sqrt(ptot2);
+        if (arg >  1.0) arg =  1.0;
+        if (arg < -1.0) arg = -1.0;
+        angle = std::acos(arg) * 180.0 / M_PI;
+    }
```

The `ptot2` normalisation is redundant for real segments (the PCA axes are
explicitly unit-normalised at `:141-147`) but is kept so the arithmetic shape
matches the prototype's and no question remains.

**This is not simply a stricter gate.** It moves `n_large_angles` in *both*
directions: null-direction pairs go 90° → 0° (fewer large angles, fewer fits),
while near-anti-parallel pairs whose `dot` rounds below −1 go `NaN` → 180° (more
large angles, more fits). The net on any given event is empirical.

Both sites carry inline prototype file+line citations per CLAUDE.md §2.

### 7.2 Repro

```bash
wcbuild && ./build/clus/wcdoctest-clus
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for arm in work-vtxfit388-fix work-vtxfit388-fix31 work-vtxfit388-fix-rep; do
  PR_EXTRA_STAGES=pr_display PR_JOBS=1 SBND_WCT_LOGLEVEL=trace \
    ./run_pr_chain_batch.sh work-nuecc48-prod0803 $arm data 388
done
```

`./build/clus/wcdoctest-clus` — **71/71 passed, 0 failed** (1 skipped) with both
fixes in. Freshness proof (M1): `local/lib/libWireCellClus.so` 18:48:31 >
`clus/src/MyFCN.cxx` 18:48:02.

Arms, all on the same input `work-nuecc48-prod0803/ql_evt388`:

| label | binary |
|---|---|
| `work-vtxfit-388` | baseline, pre-fix (the §2 arm, reused unchanged — M13) |
| `work-vtxfit388-fix31` | §3.1 only (angle guard temporarily reverted) |
| `work-vtxfit388-fix` | §3.1 + §3.2 — what is committed |
| `work-vtxfit388-fix-rep` | repeat of `-fix`, same binary — the nondeterminism yardstick |

> `work-vtxfit388-fix31-VOID-badbuild` is a **void** arm: its `wcbuild` hit M3
> (the install race left `local/lib/libWireCellClus.so` as `data`, not ELF) while
> the run still exited 0, so its binary is unknown. Renamed rather than deleted,
> and re-run cleanly as `work-vtxfit388-fix31`. **Do not use it.**

### 7.3 Result — the neutrino vertex moves 0.765 cm

```
main_vertex   baseline  (-162.5332, 31.9991, 426.1712) cm
              fixed     (-163.1355, 31.5351, 426.2557) cm
              delta     dx=-0.602  dy=-0.464  dz=+0.084   |d| = 0.765 cm
```

The trace shows *why*, and it is the §3.1 mechanism directly:

| | baseline | §3.1 only | §3.1+§3.2 |
|---|---|---|---|
| fit attempts (cluster 23) | 4 | 4 | 4 |
| **main vertex, 1st pass** | **moved 0.693 cm** | **moved 0.000 cm** | **moved 0.000 cm** |
| main vertex, triggered >0.5 cm refit? | **yes** | no | no |
| other vertex, 1st pass | moved 1.039 cm | 1.039 cm | 1.039 cm |
| main vertex, final pass | 0.000 cm | 0.000 cm | 0.000 cm |
| `fit_vertex made no update` | 1 | 1 | 1 |
| `skipping vertex fit` (the 1.0 cm cut) | 0 | 0 | 0 |

Read this carefully: the trace number is how far `wcpt()` — the Steiner seed —
was re-snapped, so **0.000 cm means the fit's optimum already coincides with the
current position**, not that the fit declined (it ran; `fit_vertex made no
update` is the "declined" line and its count is unchanged at 1). Pre-fix, the
vertex fit was *dragging the main vertex 0.693 cm off* and that drag was large
enough to trigger a second re-track-and-refit. Post-fix it converges in place.
That is exactly the shape of the owner's "the vertex looks a bit off" report.

Topology is unchanged: **84 segments, 127 vertices, 13 showers** in all arms.

Selection:

| field | baseline | fixed |
|---|---|---|
| `nue_score` | 4.30094 | 4.30094 (identical) |
| `numu_score` | −2.48440 | −2.37287 |
| `cosmict_flag` | 0 | 0 |

Both scores keep their sign and the event keeps its classification; `numu_score`
shifts by 0.11 and stays well negative.

### 7.4 Attribution: §3.1 does all of it, §3.2 is inert on this event

`work-vtxfit388-fix31` and `work-vtxfit388-fix` agree on `main_vertex` to all
printed digits and on every selection score exactly. Their calib dumps do differ
— 1343 leaf values — but the difference is **entirely inside the known
run-to-run nondeterminism envelope**, established by re-running the *same*
binary:

| comparison | leaf diffs | where |
|---|---|---|
| `-fix` vs `-fix-rep` (**same binary**) | 825 | 815 `proj/charge_pred`, plus `kine_energy_particle`, `kine_particle_type`, `showers/kine_dQdx`, `showers/total_length` |
| `-fix31` vs `-fix` (**§3.2 in/out**) | 1343 | 1339 `proj/charge_pred`, 2 `showers/kine_dQdx`, 2 `showers/total_length` |
| either comparison, `main_vertex` or any `*_score` | **0** | — |

The §3.2 comparison touches a strict subset of the field families that two
identical runs already disagree on, and the surviving shower numbers differ only
in the last few ulp (`634.643497382151` vs `634.6434973821506`). This is
`assemble_fitted_charge_2d`'s documented `charge_pred` nondeterminism (doc pr/26
§5.2) propagating into dQ/dx, not an effect of the angle guard.

So on evt 388, **§3.2 changes nothing measurable**. It is kept because it is the
correct port and because §3.2's reachability argument is about topologies (a stub
at a two-leg main vertex) that this event does not happen to present.

### 7.5 Scope and what is NOT claimed

* **One event, no gate.** Per the owner's instruction there is no valfast /
  `abtest` run behind this. The fix is **not bit-identical** to the pre-fix
  chain and will move other events; a population gate is still owed before it can
  be called validated.
* **"Moves 0.765 cm" is not "is now 0.765 cm more correct."** Evt 388 has no
  truth here. What is established is that the fit no longer drags the main vertex
  and that the point cloud it fits is now the one the prototype fits. Judging
  the new position is a look at the display (§7.6) or a truth-matched sample.
* The other open items — §3.3, §3b T1-T8, §4.1, §4.2 — are untouched.

### 7.6 The display

Evt 388 on port 5017 has been re-served from the post-fix arm:

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./pr_display/serve_pr_display.sh 5017 work-vtxfit388-fix/pr_evt388/calib-pr-evt388.json
```

The previous instance was serving `work-prdisp-cosscan2/pr_evt388/...`, which is
a **pre-fix** dump; that arm is left on disk untouched, so the two can be served
side by side on different ports to compare vertex placement directly.

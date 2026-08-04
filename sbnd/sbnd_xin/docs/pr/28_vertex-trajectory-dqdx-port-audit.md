# doc pr/28 — vertex fit / trajectory fit / dQ-dx fit: prototype-fidelity audit

**Why.** The owner observed that the **SBND neutrino vertex position looks a bit
off**, and asked first to check one specific recalled prototype behaviour —
*after the 3-D vertex fit, the vertex is held fixed and only the other
trajectory points are fitted* — then widened it to *"anything weird on vertex
fitting, track trajectory and dQ/dx fitting compared to prototype code?"*

**Status.** Audit + measurement, plus **three rounds of fixes** the owner ordered
after reading it. All seven are unconditional — no knob: they are port-fidelity
bugs or plain defects, not legacy behaviour to preserve.

| round | items | toolkit commit | section |
|---|---|---|---|
| 1 | **§3.1** wrong point cloud · **§3.2** angle guard | `e6c51cc5` | **§7** |
| 2 | **§3.3a** `clear_fit` wipes PID · **§3.3b** silent no-op reported as success · **§3.3c** charge-veto radius `0.6 → 0.3 cm` · **§3.3d** unsorted `out_edges` | `c89cb7b4` | **§8** |
| 3 | **§3.3e** `improve_vertex`'s remaining unsorted `out_edges` — the two that mutate segments | `ea1a7e3d` | **§9** |

Each fixed item is marked **FIXED** at its own section/table row below — do not
read §3.1, §3.2 or §3.3 rows a–e as open defects. **Still open** (all escalation
rule 1, untouched): §3.3's `flag_front` row and the rows after it, §3b T1–T8,
§4.1, §4.2.

**Headline.** The vertex-fixing mechanism the owner asked about is a **faithful
port and it fires on SBND** — so it does *not* explain an off vertex (§1, §2).
The audit found the vertex fit was reading the **wrong point cloud** (§3.1) and,
separately, that `clear_fit` was **discarding the PID of every leg of every
fitted vertex** (§3.3a). Both are now fixed; together they move the evt 388
neutrino vertex **0.854 cm** and remove **4 spurious vertex fits**. One confirmed
behaviour-changing divergence in the **dQ/dx** fit (§4.1) and a missing uBooNE
calibration chain (§4.2) remain open; neither moves the vertex directly, both
reach it only through PID.

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

## §3 Vertex fit internals (`MyFCN`) — **the confirmed defects that move the vertex**

> **All defects in §3.1, §3.2 and the first four rows of §3.3 are FIXED**
> (§7 round 1, §8 round 2). The findings are kept in their original form below
> because they document *why* each change was made; the FIXED markers say which.

§1 answered "is the vertex *held* fixed". This section audits the fit that
*produces* the position, and it is where the audit found real problems. All
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

### §3.1 **The PCA is computed on the wrong point cloud** (headline) — **FIXED** (`e6c51cc5`, §7)

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

### §3.2 Degenerate legs open the fit gate that should stay closed — **FIXED** (`e6c51cc5`, §7)

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

**Rows a–d are FIXED** in round 2 (§8); the rest are open.

| what | prototype | toolkit | severity |
|---|---|---|---|
| **(a) FIXED §8** — **`UpdateInfo` → `clear_fit` also wipes PID/direction**: toolkit's `Segment::clear_fit` additionally resets `m_dirsign=0`, `m_dir_weak=false`, `m_particle_score=100`, `m_particle_info=nullptr` | `ProtoSegment.cxx:1012-1040` (touches none of these) | `PRSegment.cxx:80-118`, called `MyFCN.cxx:460` | **behaviour-changing** — every vertex fit silently discards PID on all attached legs. **Largest mover of the two rounds**; see §8.1a for the `:2327` mechanism |
| **(b) FIXED §8** — **Silent no-op reported as success**: three early `return`s in `UpdateInfo` (fit_pos outside a DV; missing wpid offsets; missing `steiner_pc`) abort before any write, yet `fit_vertex` still returns `true` | no counterpart | `MyFCN.cxx:294-298, 310-313, 351-355` | **behaviour-changing** — `improve_vertex` then re-tracks and may re-fit on an unmoved vertex. Toolkit-only correctness fix, *not* port fidelity — the prototype has no such guards |
| **(c) FIXED §8** — charge-veto sampling radius | 0.3 cm (the default, `ToyCTPointCloud.h:35`) | **0.6 cm** passed explicitly | **behaviour-changing** — 8× the sampled volume, so a different charge integral on *every* vertex fit; same veto logic |
| **(d) FIXED §8** — second-loop `vertex_segments` refresh uses unsorted `boost::out_edges` where the first loop uses `sorted_out_edges` | — | `NeutrinoVertexFinder.cxx:2222-2229` | benign (feeds only a `size()==3` test) — fixed as hygiene, byte-identical by construction |
| `flag_front` criterion: index identity vs distance comparison (the distance form is present in the prototype but **explicitly commented out**) | `:857-862` | `MyFCN.cxx:373-376` | behaviour-changing — **OPEN** |
| `+0.5` half-wire offset on stored `pu/pv/pw/pt` | present `:971` | dropped `:466-469` | benign *if* consumers agree — it is a global port convention (`TrackFitting.cxx:1315` also drops it) |
| identity-by-index → identity-by-0.01 cm distance throughout `UpdateInfo` | `:900-935` | `MyFCN.cxx:417-451` | forced — toolkit `WCPoint` has no index field (`PRCommon.h:96-99`); two Steiner points within 0.01 cm would alias |
| `m_fit_vertex_min_seg_length` | **no counterpart** (grep: zero hits) | `NeutrinoPatternBase.h:203` | toolkit-only; inert at its `0` default, but **SBND sets 1.0** |
| **(e) FIXED §9** — **the `fitted_vertices` consumer loop** also iterates unsorted `boost::out_edges` — and unlike row (d) this one calls `segment_is_shower_topology` and `determine_dir_*`, which **do** affect output | — | `NeutrinoVertexFinder.cxx:2323` | **order-dependent output path**, i.e. genuine run-to-run nondeterminism (`setS` ⇒ *pointer* order). Found while fixing (a)–(d), fixed in its own round together with the two sibling loops at `:2374` and `:2176` |

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

1. ~~**§3.1 — the `MyFCN` PCA runs on the Steiner skeleton instead of the fitted
   trajectory.**~~ **FIXED, §7.** One line (`MyFCN.cxx:57-58`), directly biases
   the fitted vertex position, and the toolkit already holds the correct cloud in
   `Segment::fits()`. Confirmed the largest single mover of round 1: 0.765 cm on
   evt 388.
2. ~~**§3.2 — degenerate legs count as 90° and open the fit gate.**~~ **FIXED,
   §7** — but measured **inert on evt 388**; kept as correct-port insurance for
   topologies this event does not present.
3. ~~**§3.3a — `UpdateInfo` wipes PID/direction on every attached leg**~~
   **FIXED, §8.** This prediction was right and *understated*: it does feed
   "which vertex is chosen", via the `:2327` gate — see §8.1a. It removed 4
   spurious vertex fits on evt 388 and is round 2's entire effect.
4. ~~§3.3b — the "silent no-op reported as success" early returns.~~ **FIXED,
   §8**; measured inert on evt 388 (no guard ever fired).
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

**Items 1–4 are now fixed** — the owner read this section and asked for §3.1/§3.2
first (§7), then §3.3a–d (§8). **Items 5–10 remain open and are still escalation
rule 1.** The compounding note above still stands for the open items: §3b T1 and
the now-fixed §3.1 both feed the same cloud, so T1 must be gated on its own.

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

---

## §8 §3.3a–d FIXED — round 2 (toolkit `c89cb7b4`), and its effect on evt 18255/388

Owner instruction, after reading §3.3: *"Can you then fix these ... for the
relevant ones, please double check with prototype code. After that, please
update the md file, commit and push, rerun this 388 event served on display 5017
port, restart the display."*

**No knob**, for the same reason as §7 — and note (b) is a *toolkit-only
correctness* fix rather than a port-fidelity one: the prototype has no
counterpart guards to be faithful to.

### 8.1 The changes

Four files, `+53 / −26`. Every claim below was re-read in both trees before
editing; prototype anchors are cited inline in the code.

#### 8.1a — `Segment::clear_fit` no longer discards PID (`clus/src/PRSegment.cxx`)

The prototype's `clear_fit()` (`ProtoSegment.cxx:1012-1040`) clears **only** the
fit arrays and rebuilds `pcloud_fit`. `particle_type`, `flag_dir`, `dir_weak`
and `particle_score` survive untouched. The toolkit additionally reset all four.

The decisive scoping fact: **`clear_fit` has exactly one caller in each tree** —
`MyFCN::UpdateInfo` (`MyFCN.cxx:482`; prototype `NeutrinoID_improve_vertex.h:943`).
So this was never general bookkeeping: it discarded the PID of every leg of every
**fitted vertex**, and nothing else.

That is observable, and it is why this is the round's big mover:

```cpp
// NeutrinoVertexFinder.cxx:2327   (prototype :259 -> `if (sg->get_particle_type()==0)`)
if (!sg->particle_info()) segment_is_shower_topology(sg, ...);
```

With `m_particle_info` nulled by `clear_fit`, that gate was **always true**, so
`segment_is_shower_topology` re-ran on legs the prototype skips.

But the re-run is **not** what removed the 4 fits — be precise here, because the
loose version of this claim does not survive checking. The `is_shower` test is

```cpp
// NeutrinoVertexFinder.cxx:2070
bool is_shower = sg->flags_any(kShowerTrajectory) || sg->flags_any(kShowerTopology)
                 || (sg->particle_info() && std::abs(sg->particle_info()->pdg()) == 11);
```

`clear_fit` never touched either flag, so only the **third** term can flip when
PID is preserved. And the flags were demonstrably *not* set on these legs: under
the old code `:2327` did re-run `segment_is_shower_topology` on them, and they
still counted as tracks — so that re-run did not set `kShowerTopology`. The
measured chain is therefore:

> PID survives `clear_fit` → `|pdg| == 11` at `:2070` is true → the leg is a
> **shower**, not a track → `ntracks == 0` → the `:2077`
> `ntracks == 0 && vtx != main_vertex` filter excludes the vertex.

An electron leg had been coming back as `pdg = 0` and being counted as a track,
so the toolkit fitted vertices the prototype never fits (measured: 4 such fits on
evt 388, §8.3). Stopping the `:2327` re-run is a real second consequence of the
same fix, but on this event it was inert.

#### 8.1b — `UpdateInfo` reports whether it actually updated (`MyFCN.{h,cxx}`, `NeutrinoVertexFinder.cxx`)

`UpdateInfo` is now `bool`. The three guards return `false`; the tail returns
`true`. Both call sites became

```cpp
return results.first && fcn.UpdateInfo(results.second, cluster, track_fitter, dv);
```

The `&&` short-circuit reproduces the previous `if (results.first)` exactly —
`UpdateInfo` is still never called on a failed fit.

`fit_vertex`'s return feeds **three** consumers, all of which assume the vertex
really moved: the `> 0.5 cm` re-track at `:2099`, `refit_vertices` at `:2229`,
and `fitted_vertices` at `:2087` / `:2262` (whose consumer is the `:2318`
direction/PID loop). When a guard fires, nothing was written — no `clear_fit`,
no path change, no position change — so `false` is correct for all three at once.
The prototype's `UpdateInfo` is `void` and cannot fail, so `flag_update == true`
there always did mean "the vertex was updated"; this restores that invariant.

#### 8.1c — charge-veto sampling radius `0.6 → 0.3 cm` (`NeutrinoVertexFinder.cxx`)

```cpp
// prototype NeutrinoID_improve_vertex.h:22-23  -- no radius argument
double old_charge = ct_point_cloud->get_ave_3d_charge(vtx->get_fit_pt());
// ToyCTPointCloud.h:35   double get_ave_3d_charge(WCP::Point& p, double radius = 0.3*units::cm);
```

`Facade_Grouping.h:315` carries the **same 0.3 cm default**; the toolkit passed
`0.6` explicitly, present since `6b66163e` ("implement the fit_vertex function")
with no comment or rationale. That is 8× the sampled volume, averaging in charge
from well outside the vertex, on *every* vertex fit. The argument is now left
implicit so the two defaults cannot drift apart.

> This is the change with the widest reach across a population even though it is
> measured inert on evt 388 (§8.3) — it perturbs the veto decision at every
> vertex fit, and events sitting near the 5000/8000 e⁻ thresholds will flip.

#### 8.1d — `sorted_out_edges` in the second `vertex_segments` refresh

`NeutrinoVertexFinder.cxx:2223`, matching the first loop at `:2247`.
**Byte-identical by construction**: the set feeds only a `size() == 3` test,
which cannot depend on iteration order. Fixed as determinism hygiene.

### 8.2 Repro

```bash
wcbuild && ./build/clus/wcdoctest-clus
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for arm in work-vtxfit388-r2 work-vtxfit388-r2-rep work-vtxfit388-r2-noC; do
  PR_EXTRA_STAGES=pr_display PR_JOBS=1 SBND_WCT_LOGLEVEL=trace \
    ./run_pr_chain_batch.sh work-nuecc48-prod0803 $arm data 388
done
```

`./build/clus/wcdoctest-clus` — **71/71 passed, 0 failed** (1 skipped).
Freshness proof (M1): `local/lib/libWireCellClus.so` 19:10:44 >
`NeutrinoVertexFinder.cxx` 19:10:17, and `file` confirms ELF (M3).

| label | binary |
|---|---|
| `work-vtxfit-388` | original baseline, pre-§7 (reused unchanged — M13) |
| `work-vtxfit388-fix` | §7 only (§3.1+§3.2) — round 2's baseline |
| `work-vtxfit388-r2-noC` | §7 + a + b + d, **radius left at 0.6** — isolates (c) |
| `work-vtxfit388-r2` | §7 + a + b + c + d — what is committed |
| `work-vtxfit388-r2-rep` | repeat of `-r2`, same binary — the nondeterminism yardstick |

### 8.3 Result — the vertex moves a further 0.137 cm, and 4 spurious fits vanish

```
main_vertex   pre-§7 baseline  (-162.5332, 31.9991, 426.1712) cm
              after §7        (-163.1355, 31.5351, 426.2557) cm   |d| = 0.765 cm
              after §8        (-163.1962, 31.4605, 426.1582) cm   |d| = 0.854 cm cumulative
                                                                        (+0.137 cm this round)
```

The bigger structural change is in the fit loop:

| | after §7 | after §8 |
|---|---|---|
| `fit_vertex` calls in the traced loop | **8** | **4** |
| distinct vertices fitted | 3 | **1** (the main vertex only) |
| `fit_vertex made no update` | 2 | **0** |
| `UpdateInfo` guard messages | 0 | 0 |
| main vertex, every pass | 0.000 cm | 0.000 cm |
| `skipping vertex fit` (the 1.0 cm cut) | 0 | 0 |

The two extra vertices — at `(-168.70, 32.19, 441.22)` and
`(-169.02, 31.67, 442.72)`, `nsegs=3` each, one of which was moving **1.039 cm**
per pass — are no longer fitted at all. That is 8.1a working exactly as
predicted: with PID preserved, their legs are correctly seen as showers,
`ntracks` is 0, and the `:2077` filter excludes them. The prototype never fitted
these vertices.

Topology is unchanged across **all five** arms: **84 segments, 127 vertices,
13 showers**.

Selection:

| field | pre-§7 | after §7 | after §8 |
|---|---|---|---|
| `nue_score` | 4.30094 | 4.30094 | 4.30094 (identical throughout) |
| `numu_score` | −2.48440 | −2.37287 | **−1.85489** |
| `cosmict_flag` | 0 | 0 | 0 |

`numu_score` moves 0.518 this round — larger than round 1's 0.11 — but stays
well negative and the event keeps its classification. `nue_score` has not moved
at all across either round.

### 8.4 Attribution: (a) does all of it; (b), (c), (d) are inert here

| item | evidence | verdict on evt 388 |
|---|---|---|
| **(a) `clear_fit` PID** | the only remaining change once b/c/d are excluded; mechanism visible in the trace as the 8→4 fit-loop collapse. **Inferred by elimination, not measured directly** — no arm was run with (a) alone reverted | **the entire effect** |
| **(b) `UpdateInfo` bool** | `grep "UpdateInfo: Warning"` → **0 hits** in every arm: no guard ever fires | **provably inert** |
| **(c) charge radius** | `-r2-noC` vs `-r2` agree on `main_vertex` **to all printed digits** and on every score exactly | **inert** (see leaf-diff below) |
| **(d) `sorted_out_edges`** | feeds only `size()==3` | **byte-identical by construction** |

The (c) comparison needs the nondeterminism control, because its calib dumps are
*not* leaf-identical:

| comparison | leaf diffs (of 176 956) | where |
|---|---|---|
| `-r2` vs `-r2-rep` (**same binary**) | **356** | 351 `proj/charge_pred`, 2 `kine_energy_particle`, 2 `showers/kine_dQdx`, 1 `kine_reco_Enu` |
| `-r2-noC` vs `-r2` (**radius in/out**) | **357** | 347 `proj/charge_pred`, 3 `kine_energy_particle`, 2 `kine_energy_info`, 2 `kine_particle_type`, 2 `showers/kine_dQdx`, 1 `kine_reco_Enu` |
| either, `main_vertex` or any `*_score` | **0** | — |

357 ≈ 356, in the same field families: the radius change sits **entirely inside
the noise floor** two identical runs already produce (`assemble_fitted_charge_2d`
`charge_pred` nondeterminism, doc pr/26 §5.2). `-r2` and `-r2-rep` are identical
on `main_vertex` and every score.

> **Method note, worth repeating from §7.4:** never attribute a calib-JSON diff
> to a code change without re-running the *same* binary first. Here the noise
> floor is 356 leaves; a 357-leaf diff would otherwise have looked like a result.

### 8.5 Scope and what is NOT claimed

* **One event, no gate.** Per the owner's instruction there is no valfast /
  `abtest` run behind this. The change is **not bit-identical** and *will* move
  other events — (c) especially, since it perturbs the veto at every vertex fit
  even though it happened not to flip one here. A population gate is still owed
  for §7 **and** §8 together, and the next valfast baseline must be regenerated.
* **Inert on evt 388 ≠ inert.** (b) and (c) are unfired here, not proven
  harmless. (b) fires only on pathological geometry (fit position outside every
  detector volume, missing wpid entry, missing `steiner_pc`).
* **The 4 removed fits are the prototype's behaviour, not an improvement claim.**
  Evt 388 has no truth for the vertex position. What is established is that the
  toolkit now fits the vertices the prototype fits, from the cloud the prototype
  uses, with the prototype's charge-sampling radius.
* Found but deliberately **not** fixed here: the `fitted_vertices` consumer loop
  at `:2323` iterates unsorted `boost::out_edges` on a path that *does* affect
  output (§3.3, last row). Outside the four listed items.

### 8.6 The display

Evt 388 on port 5017 re-served from the round-2 arm:

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./pr_display/serve_pr_display.sh 5017 work-vtxfit388-r2/pr_evt388/calib-pr-evt388.json
```

`work-vtxfit-388` (pre-§7) and `work-vtxfit388-fix` (post-§7, pre-§8) are both
left on disk, so all three vertex positions can be served side by side.

---

## §9 §3.3e FIXED — `improve_vertex` has no unordered edge iteration left (toolkit `ea1a7e3d`)

Owner instruction, after reading §8.5's "found but deliberately not fixed" note:
*"can you fix [the `fitted_vertices` consumer loop] as well? Update the md file,
commit and push."*

### 9.1 Why this one is not cosmetic

`sorted_out_edges` exists precisely because of this (`PRTrajectoryView.h:149-152`):

> `boost::out_edges()` with `setS` iterates in **pointer order**.

Pointer order is not stable across runs. Row (d) of §3.3 was safe to call
byte-identical because its result feeds only a `size() == 3` test — a count
cannot depend on order. The loop the owner flagged is a different animal: it
**mutates** every segment it visits.

Three loops were left in `improve_vertex`. All three are now
`sorted_out_edges`:

| site | what it does | order-sensitive? |
|---|---|---|
| `:2333` — the `fitted_vertices` consumer | `segment_is_shower_topology` (sets `kShowerTopology`) then `segment_determine_dir_track` (writes `dirsign`, `particle_info`, `particle_score`) | **yes — mutating** |
| `:2374` — the `main_vertex` special case | `unset_flags(kShowerTopology)` plus `segment_determine_dir_track` on three separate branches | **yes — mutates harder** |
| `:2176` — collects `main_vertex_segments` | feeds only a `size() == 3` test | no — byte-identical, fixed for consistency |

The `:2374` sibling was not in the owner's message. It is the *same defect in the
same function*, three screens below the flagged one, and leaving it would have
left the nondeterminism hole open while the doc claimed it closed — so it is
fixed here rather than logged as a fourth round. The `*it` dereferences inside it
(`boost::source`/`boost::target`, six sites) were rewritten to the loop's
`edesc`.

**`improve_vertex` now contains zero `boost::out_edges` calls.** The rest of
`NeutrinoVertexFinder.cxx` still has many; they are outside this audit's scope
and are *not* claimed clean.

### 9.2 Repro

```bash
wcbuild && ./build/clus/wcdoctest-clus
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for arm in work-vtxfit388-r3 work-vtxfit388-r3-rep; do
  PR_EXTRA_STAGES=pr_display PR_JOBS=1 SBND_WCT_LOGLEVEL=trace \
    ./run_pr_chain_batch.sh work-nuecc48-prod0803 $arm data 388
done
```

`./build/clus/wcdoctest-clus` — **71/71 passed, 0 failed** (1 skipped).

### 9.3 Result — inert on evt 388

| | `-r2` (round 2) | `-r3` | `-r3-rep` |
|---|---|---|---|
| `main_vertex` | `(-163.1962, 31.4605, 426.1582)` | **identical** | **identical** |
| `nue_score` / `numu_score` | 4.30094 / −1.85489 | **identical** | **identical** |
| segments / vertices / showers | 84 / 127 / 13 | 84 / 127 / 13 | 84 / 127 / 13 |

| comparison | leaf diffs (of 176 956) | where |
|---|---|---|
| `-r3` vs `-r3-rep` (**same binary**) | **268** | 264 `proj/charge_pred`, 2 `showers/kine_dQdx`, 2 `showers/total_length` |
| `-r2` vs `-r3` (**sorting in/out**) | **275** | 265 `proj/charge_pred`, 2 each `kine_energy_info`, `kine_energy_particle`, `kine_particle_type`, `showers/kine_dQdx`, `showers/total_length` |

275 against a 268-leaf same-binary floor, same field families: **inside the
nondeterminism envelope**. On evt 388 the segments attached to each fitted vertex
do not compete — the mutations are independent per segment — so the order never
mattered here.

> The noise floor is an *estimate*, and it moves: 356 leaves in §8.4, 268 here.
> Always re-measure it in the same session as the comparison you are judging.

### 9.4 What is and is not claimed

* **Inert on this event is not "no-op".** The whole point is that the previous
  code's output depended on heap addresses. An event where two legs of one
  fitted vertex are classified in competition — one `segment_is_shower_topology`
  call changing what the next leg's `calculate_num_daughter_showers` sees — would
  have been genuinely irreproducible before this and is reproducible after.
* **This makes results stable, not correct.** Sorting by edge index is *an*
  order; the prototype's `map_vertex_segments` iteration order is a different
  one. No claim is made that the two now agree — only that the toolkit's no
  longer varies run to run.
* **Not bit-identical** to `c89cb7b4` output in general (it is on evt 388). The
  population gate owed for §7 and §8 now covers §9 too.
* `NeutrinoVertexFinder.cxx` outside `improve_vertex` still iterates unsorted
  out-edges in ~40 places. Unaudited, deliberately untouched.

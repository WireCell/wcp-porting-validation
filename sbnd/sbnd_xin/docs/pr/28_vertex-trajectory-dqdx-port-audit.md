# doc pr/28 — vertex fit / trajectory fit / dQ-dx fit: prototype-fidelity audit

**Why.** The owner observed that the **SBND neutrino vertex position looks a bit
off**, and asked first to check one specific recalled prototype behaviour —
*after the 3-D vertex fit, the vertex is held fixed and only the other
trajectory points are fitted* — then widened it to *"anything weird on vertex
fitting, track trajectory and dQ/dx fitting compared to prototype code?"*

**Status.** Audit + measurement, plus **eight rounds of fixes** the owner ordered
after reading it. All are unconditional — no knob: they are port-fidelity
bugs or plain defects, not legacy behaviour to preserve.

**This doc is now CLOSED.** Round 8 (§14) fixed the last item the owner listed,
measured both container classes §11.8 left, and closed §4.2/§4.3/§4.4. What
remains is recorded as *new* items with their own starting points, not as
unfinished business here: the shower-quantity nondeterminism (§14.5), the
`SIZE_MAX` index hazard (§14.7), the unconfirmed §11.7 mechanism (§14.8), the
global slice-start `x` convention (§14.9), and the owed valfast/1000 population
gate.

| round | items | toolkit commit | section |
|---|---|---|---|
| 1 | **§3.1** wrong point cloud · **§3.2** angle guard | `e6c51cc5` | **§7** |
| 2 | **§3.3a** `clear_fit` wipes PID · **§3.3b** silent no-op reported as success · **§3.3c** charge-veto radius `0.6 → 0.3 cm` · **§3.3d** unsorted `out_edges` | `c89cb7b4` | **§8** |
| 3 | **§3.3e** `improve_vertex`'s remaining unsorted `out_edges` — the two that mutate segments | `ea1a7e3d` | **§9** |
| 4 | **the whole `clus/` sweep** — all 117 remaining raw `boost::out_edges` sites in 13 files | `4f2e7303` | **§10** |
| 5 | **the residual 10** — `boost::edges` / `boost::vertices` / `graph_nodes` (32 loops, 12 files). **`T_tagger` is now fully deterministic on the 48-event manifest.** | `c05bc5f7` | **§11** |
| 6 | **§4.1** the multi-track dQ/dx close weights get the prototype's ×5/3 scale-up. **Vertex-fit area closed by owner decision (§12.7).** | `01ff88b1` | **§12** |
| 8 | **§4.3 last row** `assemble_fitted_charge_2d` iterated a pointer-keyed map, making `charge_pred` run-dependent — the same-binary noise floor drops **593 → 2** leaves and `proj[]/charge_pred[]` to **zero**. **Both §11.8 container classes measured; §4.2/§4.3/§4.4 closed, §4.4 *settled*; and one newly-found pointer-ordered MUTATING traversal (`NeutrinoVertexFinder.cxx:2934`) fixed (§14.12).** | `22249ff4` | **§14** |
| 7 | **§3b T1+T2** the multi-track charge veto was structurally dead · **T3** the dead-channel lookup used the loop position, not the global index · **T6** a close-vertex reset destroyed the segment's trajectory. **T4 kept as-is + made non-silent; T5/T7/T8 dropped (§13.7). Owner accepted the result from the event display, and there is no knob to flip: the round is unconditional (§13.12).** | `23bd6783` | **§13** |

Each fixed item is marked **FIXED** at its own section/table row below — do not
read §3.1, §3.2, §3.3 rows a–e, §4.1, or §3b T1/T2/T3/T6 as open defects.

**The vertex-fit area is CLOSED.** After round 6 the owner ruled: *"I do not
think that we need to worry about the remaining vertex fitting divergencies."*
That closes §3.3's remaining rows (`flag_front`, the `+0.5` offset, identity-by-
distance, `m_fit_vertex_min_seg_length`) and §5's vertex-fit suspects 5–6. They
stay documented as **known, accepted divergences** — not as pending work, and
not as defects to be rediscovered by a later audit. See §12.7.

**The trajectory-fit area is closed too, after round 7.** All of §3b T1–T8 were
re-read in both trees and triaged; the owner ordered **T1, T2, T3, T6** fixed as
bugs, kept **T4**'s toolkit behaviour (asking only that it be made robust /
non-silent), and **dropped T5, T7, T8**. See §13.7.

~~**Still open** (untouched by any ruling): §4.2, §4.3, §4.4, and §11.8's two
container classes — including `m_cluster_fitted_charge_2d`'s pointer
comparator.~~ **All closed in round 8 — see §14.** `m_cluster_fitted_charge_2d`
is FIXED (§14.1–14.4); §4.4 is SETTLED, not merely closed (§14.9); the two
container classes are measured and classified, with nothing converted
(§14.6, §14.7).

**Headline.** The vertex-fixing mechanism the owner asked about is a **faithful
port and it fires on SBND** — so it does *not* explain an off vertex (§1, §2).
The audit found the vertex fit was reading the **wrong point cloud** (§3.1) and,
separately, that `clear_fit` was **discarding the PID of every leg of every
fitted vertex** (§3.3a). Both are now fixed; together they move the evt 388
neutrino vertex **0.854 cm** and remove **4 spurious vertex fits**. One confirmed
behaviour-changing divergence in the **dQ/dx** fit (§4.1) and a missing uBooNE
calibration chain (§4.2) were both flagged; **§4.1 is now fixed (§12)** and §4.2 is the
owner's call and set aside. Neither moves the vertex directly — measured, not
assumed: §12 confirms the vertex is bit-identical across the §4.1 fix, which
reaches selection only through dQ/dx → PID.

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

**Rows a–d are FIXED** in round 2 (§8), row (e) in round 3 (§9).

> **The remaining rows are CLOSED by owner decision, not open work** (§12.7).
> They were investigated, are understood, and are accepted as-is. The severity
> labels below describe what each divergence *is*, not a queue. `flag_front` in
> particular reads "behaviour-changing — **OPEN**" for the historical record;
> read it as **accepted**.

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
* ~~**Reported by an audit pass and *not* independently re-verified**: **T1, T2,
  T3, T4, T6, T7, T8**.~~ **Superseded — round 7 (§13.1) re-read all seven in
  both trees.** Every row below is now first-tier. Two of them changed under
  re-reading: T4 gained a *cause* (it compensates for a `reset_fit_prop`
  divergence, so it is not gratuitous) and T5's claim that the prototype resets
  `index` turned out to be wrong (see the corrected row).

That distinction is not pedantry: this document's first draft carried a
`saved_skip` "divergence" that a second look disproved (see §1). Re-read before
you fix.

### Behaviour-changing divergences

**Round 7 disposition** (§13): T1, T2, T3, T6 **FIXED** (`23bd6783`); T4 kept
with a non-silent counter; T5, T7, T8 **dropped** by owner decision. The rows
below are the audit as written, annotated.

| # | what | prototype | toolkit | note |
|---|---|---|---|---|
| **T1** — **FIXED §13** | **The charge-ratio veto in `skip_trajectory_point` is dead in the multi-track path.** `pss_vec` is filled from `final_ps_vec` (already-fitted points), so the comparison point *is* the point under test ⇒ `ratio == 3`, `ratio_1 == 1` exactly ⇒ the cut is unreachable and the `p = ps_point` revert is a no-op. | `:429` passes `init_ps_vec`; cut at `PR3DCluster_trajectory_fit.h:745-750` | `TrackFitting.cxx:4704`, consumed `:5041` | The **single**-track caller (`:4513-4520`) passes pre-fit points and *is* correct — same function, two callers, one right. That asymmetry is what makes this a port slip rather than a design choice. Constants (`0.97`, `0.75`, `160/90/45°`, `0.5 cm`) all match. |
| **T2** — **FIXED §13** | `angle1` computed from fitted instead of initial points — same root cause as T1 | `PR3DCluster_trajectory_fit.h:768-769` | `TrackFitting.cxx:5182-5186` | |
| **T3** — **FIXED §13** | **Dead-channel plane-quantity lookup uses the loop position, not the fit index.** `m_3d_to_2d.find(i)` where `i` is the per-segment point position, but the map is keyed by the **global** `count`. For every segment after the first this *hits a valid but wrong point* rather than missing. | `PR3DCluster_trajectory_fit.h:780-782` uses `init_indices.at(i)` | `TrackFitting.cxx:5100-5105` | The toolkit's `skip_trajectory_point` signature (`:4849`) carries no index parameter at all. |
| **T4** — **KEPT §13.5** | **An extra `form_map_graph` runs before `dQ_dx_multi_fit`**, and it calls `set_fit_associate_vec`, which drops interior points whose summed plane quantity is 0 — i.e. it re-runs a point-dropping pass on the *final* post-`_3rd` trajectory, changing the output point count. | no `form_map` anywhere in `PR3DCluster_multi_dQ_dx_fit.h`; `:174-188` resets only | `TrackFitting.cxx:8257` | |
| **T5** — **DROPPED §13.7** | **Fixed vertices no longer get their projections refreshed.** The prototype's `vtx->set_fit(...)` sits *outside* the `if (!flag_fit_fix)`; the toolkit puts the whole `pu/pv/pw/pt/paf/index` update *inside* it. | `:333-342` (verified: `set_fit` outside the guard) | `TrackFitting.cxx:3842-3901` | A fixed vertex's 3-D point does not move, so its projections are only *stale* if the transform context changed. **Correction (round 7):** this row originally said the prototype also resets `index` — it does not. `set_fit(p, 0, -1, pu, pv, pw, pt, -1)` zeroes `dQ`, sets `dx` and `reduced_chi2` to −1 and refreshes the projections; `index` is never touched. And `dQ_dx_multi_fit` rewrites vertex `dQ`/`dx`/`reduced_chi2` **unconditionally** for every vertex in `vertex_index_map`, fixed or not (`TrackFitting.cxx:6888-6891`), so no stale value reaches the output or PID. That is what closed T5 — see §13.7. |
| **T6** — **FIXED §13** | `check_and_reset_close_vertices` **rebuilds the segment's fits to 2 points** (`segment->fits(generate_fits_with_projections(...))`); the prototype's inline equivalent resets only the two vertex fit points. | `:1383-1392` | `TrackFitting.cxx:1222-1227` | |
| **T7** — **DROPPED §13.7** | `charge_div_method == 2` missing-key fallback: toolkit prefills `1/N` and `continue`s on unknown wpid, leaving `1/N`; the prototype's `if/else if` leaves the key absent so `operator[]` gives `0.0` (zero weight). Reachable in the single-track 2nd pass. | `:249, :269` | `TrackFitting.cxx:3670-3690`, `:4077` | |
| **T8** — **DROPPED §13.7** | **`associated_2d_points` ordering differs**: `Coord2D` orders by `(apa,face,time,wire,channel,plane)`, the prototype's `pair<int,int>` by `(wire,time)`. Deterministic run-to-run, but it permutes the rows of `RU/RV/RW` and hence the FP accumulation order. | `:575,613,648` | `TrackFitting.h:282-290` | **Fidelity, not determinism** — it means the toolkit cannot be bit-identical to the prototype even with everything else fixed. |

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

### §4.1 CONFIRMED behaviour-changing: the multi-fit close weights were not scaled up — **FIXED** (§12)

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

~~**Not fixed — escalation rule 1**~~ **FIXED in round 6, §12** — the owner read
this section and asked for it. It changes a constant, so escalation rule 1 was
satisfied by asking, not by a knob. Of the two options noted here — a second
parameter pair, or an internal ×5/3 in `dQ_dx_multi_fit` mirroring the existing
lambda line — §12.1 took the second, for the reasons given there.

**Does this explain the off vertex? Not directly.** `dQ_dx_multi_fit` is the
charge pass that runs *after* the three geometry passes; it does not move
trajectory points. It reaches the vertex only indirectly, through dQ/dx → track
PID (`segment_do_track_pid`) → direction → the §6 vertex score.

### §4.2 The uBooNE calibration chain is absent (flagged, not a defect claim) — **CLOSED (§14.10)**

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

### §4.3 Other dQ/dx divergences — **CLOSED (§14.10); its last row is FIXED (§14.4)**

| what | prototype | toolkit | severity |
|---|---|---|---|
| `connected_vec` end-vertex case pushes `indices.size()-2` (a **size** where an **index** belongs) | `PR3DCluster_multi_dQ_dx_fit.h:723` | `TrackFitting.cxx:6708` — correct index | **toolkit fixes a prototype bug** |
| vertex-adjacency `paf` guard: toolkit drops a neighbour on a different APA/face | none | `:6414`, `:6417` | behaviour-changing; multi-APA necessity, no prototype counterpart |
| skips trajectory points with `face == -1` | none | `:6103-6108`, `:6459-6464` | behaviour-changing in edge cases |
| smearing constants hardcoded to uBooNE literals in C++ | derived from `TPCParams` | header `:41-43,52` | benign for SBND — config overrides all four (`sbnd_track_fitting.json:7-9,14`); latent for any detector run without such a file |
| `pred <= 0` guards inside the chi2 sums | none | `:6837,6842,6847` | benign (prototype would give inf/NaN) |
| wire search window `\|w−c\|≤10` vs `round(c)±10` | `:371,390,411` | `:6288-6289` | benign (the `nsigma=4` gate dominates) |
| `assemble_fitted_charge_2d` iterates a **pointer-keyed** `std::map<Cluster*,…>`, last-writer-wins | no counterpart | `:1136-1152` | **FIXED (§14)** — was toolkit-only nondeterminism, 10.2% of cells moving between `setarch -R` runs (`PrDisplayDump.cxx:772-777`, doc pr/26 §5.2). Now `PR::ClusterPtrCmp`; the same-binary `charge_pred` floor is **0**. |

### §4.4 Open question — **SETTLED in §14.9: `slice_index` is a slice START tick, and the 3-D side is on the same grid, so the window is not off centre**

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
5. ~~`fit_vertex_min_seg_length = 1.0` on *other* events~~ **CLOSED by owner
   decision (§12.7).** SBND-only, doc pr/9; inert on evt 388. The §2 trace
   recipe remains the way to check it if a future event warrants.
6. ~~`UpdateInfo` re-snapping `wcpt` to the nearest Steiner point
   (`MyFCN.cxx:474`)~~ **CLOSED by owner decision (§12.7).** The seed is
   quantised onto the skeleton, and the 0.22 cm `fit_distance` on evt 388's main
   vertex *is* that residual — understood and accepted.
7. **§3b T1/T2 — the charge-ratio veto is dead in the multi-track path.** The
   guard that reverts a badly-fitted trajectory point to its pre-fit position
   never fires, because the point is compared against itself. Trajectory points
   feed the vertex through `organize_segments_path` and through `MyFCN`'s PCA.
8. **§3b T3 — the dead-channel lookup reads the wrong point** for every segment
   after the first, silently rather than by failing.
9. §3b T4 — an extra association rebuild drops points from the final trajectory
   before dQ/dx.
10. ~~§4.1~~ **FIXED, §12** / §4.2 via PID → direction → vertex score — indirect,
    but real. §12.3 measures the §4.1 half: on evt 388 the main vertex is
    **bit-identical** across the fix, so the "indirect, but real" path is real
    for *selection* (`numu_score` −0.173) and **not** for the vertex position
    on this event.

Note that §3.1 and T1 compound: `MyFCN` builds its PCA from a point cloud whose
own quality guard (T1) is inoperative — except that under §3.1 it reads the
Steiner path rather than that cloud at all. Fixing either one alone changes what
the other sees, so they want separate gates and separate events.

**Items 1–4 are fixed** — the owner read this section and asked for §3.1/§3.2
first (§7), then §3.3a–d (§8). **Items 5–6 are CLOSED by owner decision**
(§12.7) — the vertex-fit area is done. **Item 10's §4.1 half is fixed** (§12),
and it moved selection but *not* the vertex. **Items 7–9 (§3b T1–T4) and §4.2
remain open** — trajectory and calibration, not vertex fit, so the ruling in
§12.7 does not reach them. The compounding note above still stands for those:
§3b T1 and the now-fixed §3.1 both feed the same cloud, so T1 must be gated on
its own.

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

---

## §10 The full `boost::out_edges` sweep — 117 sites, 13 files (toolkit `4f2e7303`)

Owner instruction, after §9 flagged the rest of the file as unaudited: *"I think
we should just go ahead fixing all these, these can only make things better,
right?"* — and, mid-round: *"I do not worry about the changing the output at this
point. This is a newly ported code, and we are doing the validation and
improvement. After doing it, please run some events to check the cost."*

### 10.1 Why "can only make things better" needed one qualification

Two caveats were raised before starting, and the measurement settled both:

1. **It changes output.** Sorting does not restore a known-correct answer; it
   replaces an unstable choice with a stable one. §10.4 shows the change is real
   and — on this manifest — confined to per-candidate diagnostic vectors.
2. **`sorted_out_edges` allocates and sorts per call**, where the old code was a
   bare iterator walk. Measured in §10.3: **no detectable cost**.

### 10.2 What was converted, and the one strategy that worked

**All 117** raw `boost::out_edges` iteration sites in `clus/src` (13 files) are
now `sorted_out_edges`, joining the 50 already converted. **`clus/src` contains
zero raw `boost::out_edges` iterations.**

| file | sites |
|---|---|
| `NeutrinoVertexFinder.cxx` | 36 |
| `NeutrinoTaggerNuE.cxx` | 22 |
| `NeutrinoTaggerSinglePhoton.cxx` | 14 |
| `NeutrinoTrackShowerSep.cxx` | 13 |
| `NeutrinoStructureExaminer.cxx` | 9 |
| `NeutrinoTaggerSSM.cxx` | 8 |
| `TrackFitting.cxx` | 5 |
| `NeutrinoPatternBase.cxx` | 3 |
| `NeutrinoKinematics.cxx`, `NeutrinoTaggerNuMu.cxx` | 2 each |
| `NeutrinoTaggerCosmic.cxx`, `NeutrinoShowerClustering.cxx`, `NeutrinoOtherSegments.cxx` | 1 each |

The transformation that survived contact: **keep the range variable and change
its type**, rather than deleting the declaration and inlining the call.

```cpp
-   auto edge_range = boost::out_edges(vd, graph);
-   for (auto eit = edge_range.first; eit != edge_range.second; ++eit) {
-       SegmentPtr sg = graph[*eit].segment;
+   const auto edge_range = sorted_out_edges(vd, graph);
+   for (auto eit : edge_range) {
+       SegmentPtr sg = graph[eit].segment;
```

Two earlier attempts that *deleted* the declaration both failed, because several
functions declare one range and consume it in **two or more** later loops
(`edge_range1`/`edge_range2` in `NeutrinoTrackShowerSep.cxx`). Keeping the
variable makes reuse work automatically, preserves the original scoping exactly,
and sorts **once** per declaration instead of once per loop.

Two sites needed hands:

* `NeutrinoStructureExaminer.cxx:40` — a `boost::edges` (all-edges) loop whose
  binding names collided with a converted one. Restored; **out of scope**.
* `NeutrinoStructureExaminer.cxx:2132` — already hand-rolled
  `assign(begin,end)` + `std::sort` by `graph[e].index`, i.e. `sorted_out_edges`
  written out longhand. Replaced by the helper.

**Verification that no loop bound to the wrong vector:** every introduced
`for (auto X : Y)` was checked to have a `const auto Y = sorted_out_edges(...)`
declaration inside its own function (75 range-form + 117 inline-form). A
name resolving to another function's vector cannot compile, and the build is
clean.

### 10.3 Cost — none measurable

Serial, pinned, single-thread (`PR_JOBS=1`), the runner's comparable-wall mode:

| event | baseline `ea1a7e3d` | swept | delta |
|---|---|---|---|
| 388 | 14 994 ms | 14 992 ms | −0.01 % |
| 163543 | 10 080 ms | 10 093 ms | +0.13 % |
| 172230 | 11 652 ms | 11 770 ms | +1.01 % |
| 138009 | 10 807 ms | 10 706 ms | −0.93 % |
| 116962 | 7 582 ms | 7 480 ms | −1.35 % |
| 214469 | 22 440 ms | 22 357 ms | −0.37 % |
| **total** | **77 555 ms** | **77 398 ms** | **−0.20 %** |

48-event batch (`PR_JOBS=6`): **116 s → 117 s**.

Both are inside run-to-run noise, and the sign is not even consistent. The
reason the allocation does not show: PR-graph vertices have ~1–4 out-edges, and
the type-change strategy sorts once per declaration where the old code walked
the range once per loop. **Record the number as ≈0 %, not as "small".**

### 10.4 Effect — 25 of 35 unstable T_tagger branches become deterministic

The right measurement here is not A-vs-B, it is **run-to-run instability of each
binary against itself**. Four 48-event arms, `work-oe{base,sweep}-nuecc48{,-rep}`,
47 events with a `T_tagger`:

| branch | baseline: events unstable | swept |
|---|---|---|
| `shw_sp_lol_1_v_angle` | 41 | **0** |
| `numu_cc_1_length` / `_direct_length` / `_dQ_dx_cut` / `_medium_dQ_dx` | 24 each | **0** |
| `numu_cc_1_n_daughter_all` | 23 | **0** |
| `br3_6_v_length` / `_angle` / `_direct_length` | 20 each | **0** |
| `numu_cc_1_particle_type`, `shw_sp_br3_6_v_*` | 18 each | **0** |
| `lol_2_v_*`, `shw_sp_lol_2_v_*` | 9 each | **0** |
| **`numu_cc_flag_1`** — an actual tagger **flag**, not a diagnostic | 1 | **0** |
| `shw_sp_pio_2_v_dis2` / `_angle2` / `_acc_length` | 40 each | 40 |
| `pio_2_v_dis2` / `_angle2` / `_acc_length` | 38 each | 38 |
| `stw_3_v_angle` / `_dir_length` | 27 each | 22 |
| `pio_2_v_flag`, `shw_sp_pio_2_v_flag` | 1 / 2 | 1 / 2 |

**Unstable branches: 35 → 10. Branches made fully deterministic: 25.**

A worked example, because it shows the hazard class precisely.
`NeutrinoTaggerSinglePhoton.cxx:1857` (`lol_1`) collects a vertex's shower legs
into `vtx_ss` and then does

```cpp
Vector dv1 = segment_cal_dir_3vector(vtx_ss.front(), ...);
Vector dv2 = segment_cal_dir_3vector(vtx_ss.back(),  ...);
double open_angle = dv1.angle(dv2) / M_PI * 180.0;
```

`front()`/`back()` of a vector filled in **pointer order** — so *which two legs
define the opening angle* was decided by heap layout. This is the "collect into a
vector" shape that pr/11's triage listed as order-*insensitive*; it is not, once
anything downstream indexes the vector. It is unstable in 41/47 events on the
baseline and stable in 0/47 after. `open_angle` feeds `flag_ov1` at a 36° cut,
so it is a decision input, not only a dump.

**No scalar decision branch moved between baseline and swept** across all 47
events, and `nusel-table.tsv` / `nusel-events.tsv` (546 / 49 rows) are
**byte-identical** between the two arms.

### 10.5 The residual 10 — first diagnosis, **since shown wrong** (see §11.1)

> ⚠️ **This subsection's diagnosis did not survive §11's measurement.** It is
> kept because the reasoning error is instructive, not because it is correct.
> The FP-accumulation mechanism described here is real but was **not** what made
> these ten branches unstable. The actual cause was a pointer-ordered *vertex*
> loop three lines further down. §11.1 has the correction and the evidence.

`pio_2` accumulates per-cluster track length over **all** graph edges
(`NeutrinoTaggerNuE.cxx`, `NeutrinoTaggerSinglePhoton.cxx`):

```cpp
std::map<Facade::Cluster*, double> cluster_acc_length;
for (auto [eit, eend] = boost::edges(ctx.graph); eit != eend; ++eit) {
    ...
    cluster_acc_length[sg1->cluster()] += segment_track_length(sg1);
}
```

The map is only `find()`-ed, so pointer-keyed *iteration* is not the problem —
the `+=` is. Floating-point addition is not associative, and `boost::edges`
returns edges in pointer order for a `setS` graph, so `acc_length` differs in the
last ulp between runs and moves candidates across the cut that gates
`pio_2_v_dis2`/`_angle2`. This is pr/11's **float-accumulation** hazard applied
to `boost::edges`, which this sweep did not touch.

**Follow-up candidates, in order:** (1) accumulate `cluster_acc_length` in a
stable order or over a sorted edge list — kills 6 of the 10; (2) `stw_3`, which
improved 27 → 22 and so has a second source; (3) the 36 pointer-keyed
`std::map`/`std::set<T*>` declarations still in `clus/src` (CLAUDE.md §2 forbids
iterating these).

### 10.6 Scope and what is NOT claimed

* **Not bit-identical**, deliberately, and the owner waived the concern for this
  round. The population gate owed for §7–§9 now covers §10.
* **The remaining instability is larger than what was fixed, by event count.**
  45/47 events still differ run-to-run, because `pio_2` alone touches ~40. The
  honest headline is *25 of 35 branches*, not "determinism fixed".
* **Stable ≠ correct** (as in §9.4): edge-index order is *an* order. The
  prototype iterates pointer-keyed `std::set<ProtoSegment*>` at the equivalent
  spots, so its order is equally arbitrary and no parity is broken — the same
  argument pr/11 used at `broken_muon_id`.
* `boost::edges`, `boost::in_edges` and `boost::adjacent_vertices` are untouched.
  → **swept in round 5, §11**, which also corrects this section's diagnosis of
  the residual 10.

---

## §11 The residual 10 closed — `T_tagger` is deterministic on this manifest (toolkit `c05bc5f7`)

Owner instruction: *"Can you focus on this issue [the residual 10 unstable
branches] ... Please update the md file, commit and push."*

**Result up front.** On the 48-event nueCC manifest, two runs of the same binary
now produce **byte-identical `T_tagger` on every event**: 0 unstable branches,
0 events differing, down from 10 branches / 45 of 47 events. `nusel-table.tsv`
and `nusel-events.tsv` are identical between the repeat runs, and — separately —
**identical to round 4's**, with **zero** scalar-branch differences anywhere in
`T_tagger` or `T_kine` across all 47 events. The entire A/B effect of this round
is the *ordering* of four per-candidate diagnostic vectors.

### 11.1 §10.5 was wrong: it was loop order, not float accumulation

§10.5 attributed the residual to `cluster_acc_length[...] += ...` accumulating
over pointer-ordered `boost::edges`, i.e. last-ulp drift crossing a cut. That
mechanism is real, but it was not the cause. **Three lines below the
accumulation sits the loop that actually fills the output vectors:**

```cpp
std::map<Facade::Cluster*, double> cluster_acc_length;
for (auto [eit, eend] = boost::edges(ctx.graph); eit != eend; ++eit) {   // <- §10.5 blamed this
    ...  cluster_acc_length[sg1->cluster()] += segment_track_length(sg1);
}

for (const auto& vd : graph_nodes(ctx.graph)) {                          // <- the actual cause
    ...
    ti.pio_2_v_dis2.push_back(dis2 / units::cm);
    ti.pio_2_v_angle2.push_back(back_angle);
    ti.pio_2_v_acc_length.push_back(acc_length / units::cm);
}
```

`PR::graph_nodes()` returns the **raw `boost::vertices()` order packaged as a
vector** (`PRGraphType.cxx:6-8`) — pointer order. `PR::ordered_nodes()` is the
sorted one. Despite the name, `graph_nodes` is *not* a determinism helper, and a
sweep keyed on `boost::edges`/`boost::vertices` misses it entirely. So the loop
that decides *the order of every `pio_2_v_*` entry* was pointer-ordered.

Three call sites, and they map **one-to-one** onto the ten residual branches:

| site | branches |
|---|---|
| `NeutrinoTaggerNuE.cxx:638` | `pio_2_v_dis2` `_angle2` `_acc_length` `_flag` |
| `NeutrinoTaggerSinglePhoton.cxx:2058` | `shw_sp_pio_2_v_dis2` `_angle2` `_acc_length` `_flag` |
| `NeutrinoTaggerNuE.cxx:2444` (`stw_3`) | `stw_3_v_angle` `_dir_length` |

**Three sites, exactly ten branches, no remainder.**

**The evidence that settles it** — the round-4 census was re-run with a
classifier that asks whether two differing branches are *permutations* of each
other (same multiset ⇒ pure reorder) or genuinely different numbers:

```
work-oesweep-nuecc48  vs  work-oesweep-nuecc48-rep
   40  shw_sp_pio_2_v_acc_length   reorder= 40 value=  0
   40  shw_sp_pio_2_v_angle2       reorder= 40 value=  0
   40  shw_sp_pio_2_v_dis2         reorder= 40 value=  0
   38  pio_2_v_acc_length          reorder= 38 value=  0
   38  pio_2_v_angle2              reorder= 38 value=  0
   38  pio_2_v_dis2                reorder= 38 value=  0
   22  stw_3_v_angle               reorder= 22 value=  0
   22  stw_3_v_dir_length          reorder= 22 value=  0
    2  shw_sp_pio_2_v_flag         reorder=  2 value=  0
    1  pio_2_v_flag                reorder=  1 value=  0
```

**`value = 0` everywhere.** Not one number ever changed — only their order. An
FP last-ulp drift crossing a cut would have shown up as `value > 0` (a candidate
entering or leaving the vector), and it never did. In hindsight the event counts
alone should have been enough: a last-ulp coin flip landing in 38–40 of 47 events
is not credible; a pointer-ordered `push_back` reordering a multi-entry vector in
85 % of events is exactly what one expects.

Also explained by the same mechanism: `stw_3_v_energy` and `_medium_dQ_dx` sit in
the *same* pointer-ordered loop but were always stable — they push the **same
scalar** on every iteration, so a permutation of them is invisible. And
`pio_2_v_flag` moved in only 1–2 events because it is almost always all-`1.0`.

**Method note for next time.** *Classify a difference before diagnosing it.*
Reorder-vs-value is one line of code in the comparator and it would have pointed
at the right loop immediately.

### 11.2 What was converted

**32 loops in 12 files** — 22 `boost::edges` → `ordered_edges`, 7
`boost::vertices` → `ordered_nodes`, **3 `graph_nodes` → `ordered_nodes`**.
`clus/src` now contains **zero** raw `boost::edges` / `boost::vertices` /
`graph_nodes` iterations of the PR graph.

| file | loops |
|---|---|
| `NeutrinoVertexFinder.cxx` | 10 |
| `NeutrinoTrackShowerSep.cxx` | 5 |
| `NeutrinoTaggerNuE.cxx` | 4 |
| `NeutrinoPatternBase.cxx`, `NeutrinoTaggerCosmic.cxx` | 3 each |
| `NeutrinoStructureExaminer.cxx`, `NeutrinoTaggerSinglePhoton.cxx` | 2 each |
| `NeutrinoDeghoster.cxx`, `NeutrinoTaggerNuMu.cxx`, `NeutrinoTaggerSSM.cxx`, `TrackFitting.cxx` | 1 each |

Supporting changes:

* `PR::graph_nodes` / `ordered_nodes` / `ordered_edges` now take `const Graph&`
  (bodies were already read-only); several tagger call sites hold a const graph
  and could not bind otherwise. No `const_cast` was added.
* `PRGraphType.h` — `graph_nodes()`'s doc comment now **warns in capitals** that
  its order varies between runs and that it is not a determinism helper. That
  misleading name is what hid these three sites through two prior rounds.
* `TrackFitting.cxx:279` (`sync_from_graph`) had
  `m_grouping = (*segments_set.begin())->cluster()->grouping()` over a
  `std::set<shared_ptr<Segment>>` — i.e. *the lowest heap address decided which
  grouping the fitter adopted*. Replaced with the first segment in edge-index
  order. (Benign in practice — one grouping per graph — but it is the exact
  shape CLAUDE.md §2 forbids.)
* `NeutrinoVertexFinder.cxx:1837` — the Case-5 block re-walked **every graph
  edge** once per (trajectory point × existing segment) pair just to ask "is this
  segment still in the graph". Hoisted to one `unordered_set` snapshot per
  segment. This is a pure speed fix; the test was already order-free.

### 11.3 Verified no-ops — the graphs that were never at risk

`boost::edges` / `boost::vertices` / `boost::adjacent_vertices` also appear in
`SteinerGrapher.cxx`, `Graphs.cxx`, `PatternDebugIO.cxx`, `Facade_Util.cxx` and
`PointTreeBuilding.cxx`. **These were checked and deliberately not touched** —
they walk *different graph types*, both of which already have integer
descriptors and therefore content-stable iteration:

| file | graph | declaration |
|---|---|---|
| `SteinerGrapher`, `Graphs.cxx`, `PatternDebugIO`, `Facade_Util` | `Clus::Graphs::Weighted::Graph` (reached via `Cluster::find_graph()`, `Facade_Mixins.h:347`) | `adjacency_list<vecS, vecS, …>`, `Graphs.h:22-27` |
| `PointTreeBuilding` | `ICluster::cluster_graph_t` (`icluster->graph()`) | `adjacency_list<setS, vecS, …>`, `ICluster.h:167` — `setS` is the *out-edge* list; the **vertex** list is `vecS` |

Converting either would have been churn. Only `PR::Graph`
(`adjacency_list<setS, setS, …>`, `PRGraphType.h:91-98`) has pointer descriptors.

### 11.4 The sort key is sound — new unit test

`ordered_nodes`/`ordered_edges` sort on `NodeBundle::index`/`EdgeBundle::index`.
`std::sort` is not stable, so **a duplicate index would silently fall back to the
input order — pointer order — and make every conversion in rounds 3–5 a no-op
while still looking converted.** New `clus/test/doctest_pr_graph_order.cxx`
(5 cases, 25 assertions) pins:

* node and edge indices are pairwise distinct;
* both helpers really come back strictly ascending;
* indices stay distinct across a `remove_vertex` + re-add cycle — indices come
  from a monotone counter in `GraphBundle` that never decrements
  (`PRGraph.cxx:24-26, 52-61`), so a removed index is never reissued;
* **the `EdgeBundle` index survives the aliasing path of §11.7** — adding a
  second segment between an already-connected vertex pair leaves
  `first->get_graph_index() == second->get_graph_index()` (asserted) while the
  edge indices `ordered_edges` sorts on stay pairwise unique (asserted). The two
  facts are separate and this case pins both in one place;
* `graph_nodes` and `ordered_nodes` hold the same *set*, so converting a call
  site changes order and nothing else.

`./build/clus/wcdoctest-clus`: **76 cases / 836 assertions, all pass.**

### 11.5 Repro

```bash
wcbuild
ls -la ../local/lib/libWireCellClus.so          # M1 freshness proof
./build/clus/wcdoctest-clus

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-oe5c-nuecc48     data
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-oe5c-nuecc48-rep data

# same-binary repeat census, with reorder/value classification
python3 /home/xqian/tmp/ttag_cmp5.py work-oe5c-nuecc48 work-oe5c-nuecc48-rep
# A/B against round 4
python3 /home/xqian/tmp/ttag_cmp5.py work-oesweep-nuecc48 work-oe5c-nuecc48
```

Arms (M13, all fresh): `work-oe5c-nuecc48{,-rep}` (48 events),
`work-oe5c-r{1,2}-t<evt>` (serial timing). Superseded intermediates kept for the
record: `work-oe5-*` (before the hoist) and `work-oe5b-*` (the reverted
comparator experiment in §11.7).

### 11.6 Result

**Determinism — same binary, run vs. its own repeat, 47 events with a `T_tagger`:**

| | round 4 (`4f2e7303`) | round 5 |
|---|---|---|
| unstable `T_tagger` branches | 10 | **0** |
| events with any `T_tagger` diff | 45 / 47 | **0 / 47** |
| `nusel-table.tsv` / `nusel-events.tsv` | identical | identical |

**A/B — round 4 vs round 5, same 47 events:**

| | |
|---|---|
| scalar branches differing, `T_tagger` + `T_kine`, all events | **0** |
| `nusel-table.tsv` (546 rows), `nusel-events.tsv` (49 rows) | **byte-identical** |
| branches differing at all | the same 10, **`reorder` in 100 % of cases, `value` in 0** |

So the round changes exactly what it was supposed to and nothing else: four
per-candidate diagnostic vectors are now emitted in a stable order. No
reconstructed quantity, flag, score or label moved on any event.

**Cost — serial, pinned, `PR_JOBS=1`, WCT `Timer: Total wall-sec`, two replicates:**

| event | round 4 | round 5 run 1 | round 5 run 2 |
|---|---|---|---|
| 388 | 8.892 | 8.708 | 8.738 |
| 163543 | 4.032 | 3.919 | 3.876 |
| 172230 | 5.723 | 5.784 | 5.661 |
| 138009 | 4.653 | 4.739 | 4.724 |
| 116962 | 1.448 | 1.499 | 1.522 |
| 214469 | 16.293 | 16.211 | 16.640 |
| **total** | **41.041 s** | **40.860 s (−0.44 %)** | **41.161 s (+0.29 %)** |

Sign flips between replicates, and within-arm spread on evt 388 alone is ~12 %.
**Record the cost as ≈0 %.** (The §11.2 hoist is why: an early build without it
measured a consistent +1 % from re-sorting the whole edge list inside a
triple-nested loop.)

### 11.7 One near-miss worth recording

An intermediate build also changed `existing_segments` in
`eliminate_short_vertex_activities` from `std::set<SegmentPtr>` to the
index-ordered `PR::IndexedSegmentSet`, on the reasoning that CLAUDE.md §2 forbids
iterating pointer-keyed containers. **That moved `kine_reco_Enu` on SBND evt
239794 from 2930 to 1687 MeV** and 376 other branches with it — a genuine physics
change, caught only because the A/B was run.

Why: `SegmentIndexCmp` compares `Segment::get_graph_index()`, so the swap
silently changed `find()`/`count()` from **pointer identity to index identity** —
and **that index is not unique across live `Segment` objects**:

```cpp
// PRGraph.cxx:86-89 -- PR::add_segment, "edge already existed" path
g[desc].segment = seg;                  // displaces the previous segment
seg->set_graph_index(g[desc].index);    // ...and hands the new one its index
```

> ⚠️ **The mechanism below did NOT reproduce — see §14.8.** Probing this exact
> container at this exact call site on six events, **including evt 239794**,
> finds *no* two segments sharing a graph index. The observation (the swap moved
> `kine_reco_Enu` by 1.2 GeV) stands; the conclusion (judge each site by use,
> never bulk-convert) stands and is reinforced; the *stated cause* is
> **unconfirmed** and should not be cited as established.

Adding a segment between a vertex pair that already carries an edge does not
create an edge; it *overwrites* the bundle and copies the existing index into the
new segment. The displaced segment keeps that same index, so any `SegmentPtr`
still held to it — which is exactly what `existing_segments` holds — now compares
**equal** to a different live segment. Pinned by the fifth case of
`doctest_pr_graph_order.cxx` (§11.4). Reverted; the container is now documented
in `NeutrinoPatternBase.h` as pointer-keyed **on purpose**, with the reason.

Its *iteration* needs no fix either: the loop body takes a running `min` over
three distances, which is order-insensitive. **CLAUDE.md §2's rule is about
iteration, and swapping a comparator to satisfy it can change lookup semantics —
check what the container is used for before changing how it is ordered.**

### 11.8 Scope and what is NOT claimed

* **Deterministic on this manifest, not proven deterministic.** 47 nueCC events,
  one repeat each. Other topologies can still hold pointer-ordered paths this
  manifest never reaches.
* **Not bit-identical to round 4** — though the difference is now confined to
  vector ordering, with zero scalar movement measured. The owner's waiver for
  rounds 1–4 covers it; the population gate owed for §7–§10 now covers §11 too,
  and its baseline must be regenerated.
  **Round 5 makes that gate cheaper to read.** Because the 48-event A/B showed
  *zero* scalar movement, the expected §7–§11 diff against the pre-round-1
  baseline is rounds 1–4's effects **plus vector reordering and nothing else**.
  A checkable PASS criterion: every scalar matches the rounds-1–4 baseline, and
  the only branches that differ are the ten `pio_2_v_*` / `shw_sp_pio_2_v_*` /
  `stw_3_v_*` vectors, each a **permutation** of its baseline value
  (`ttag_cmp5.py` reports this as `reorder=N value=0`).
  > ⚠️ **Superseded by round 6.** §12 changes the dQ/dx regulariser, which moves
  > scalars throughout the dQ/dx → PID → kine chain by construction. The
  > criterion above can no longer PASS and must not be read as a live FAIL
  > signal. See §12.5 for what the gate should expect once §7–§12 are gated
  > together.
* **Stable ≠ correct**, as in §9.4/§10.6. Index order is *an* order; the
  prototype's equivalent loops walk pointer-keyed `std::map<ProtoVertex*, …>`,
  so its order is equally arbitrary and no parity is broken.
* **Was "still open — and it is now TWO classes, not one". Both are now
  MEASURED in §14.6/§14.7 — nothing converted, a verdict per site, and a
  sharper rule (`SIZE_MAX`, not the inherit path, is the real hazard):**
  1. **Pointer-keyed** `std::map`/`std::set<T*>` declarations in `clus/src`
     (CLAUDE.md §2). Round 5 audited only those on the paths it touched
     (`m_clusters` already had `ClusterPtrCmp`; `segments_set` removed;
     `existing_segments` documented). ~~The rest are unaudited.~~
     **Audited in §14.6: 66 declarations, 63 never traversed, 3 traversed —
     one benign, one tie-only, one outside this doc's subsystem.**
  2. **Index-keyed** containers — the class §11.7 discovered, and the one the
     obvious fix for (1) walks straight into. Any `std::set`/`std::map` ordered
     by `Segment::get_graph_index()` can alias two *live* segments, so it is
     unsafe for identity lookup even though it is perfectly deterministic.
     Pre-existing sites: `PRShower.h:223` (`ShowerSegmentMap`),
     `NeutrinoShowerClustering.cxx:1676, 2167`, `NeutrinoTaggerSSM.cxx:608`,
     `NeutrinoKinematics.cxx:93`, `PRSegmentFunctions.cxx:1916-1918` (three).
     Related: `NeutrinoPatternBase.cxx:2159` does `seg->set_id(edge_bundle.index)`,
     which hands two segments the same id under the same collision.
     **Not touched in this round** (CLAUDE.md: an unrelated defect does not ride
     along) — recorded so the next round starts from the right list.

  §11.7 is why each site in both classes must be judged on how it is **used**,
  never converted in bulk.
* `boost::in_edges` does not occur anywhere in `clus/src`.

---

## §12 §4.1 FIXED — the multi-track dQ/dx close weights get their ×5/3 (toolkit `01ff88b1`, evt 18255/388)

Owner instruction, after reading the open-items summary: *"We want to fix [§4.1]
... we do not need to worry about these two [§3.3 `flag_front`, §4.2] ... please
double check with prototype code, and note, we focus on this one event for now.
No need to worry about the full sample validation. We will need to continue do
some improvements on the algorithm before we need to look at a larger sample."*

So: §4.1 only, prototype re-verified, evt 388 only, no population gate. The two
set-aside items are marked as such in the status table rather than deleted.

### 12.1 Prototype re-verification — all four numbers re-read this round

Not taken from §4.1's earlier table; re-grepped in
`/nfs/data/1/xqian/prototype-dev/wire-cell/pid/src/`:

| | single (`PR3DCluster_dQ_dx_fit.h`) | multi (`PR3DCluster_multi_dQ_dx_fit.h`) | ratio |
|---|---|---|---|
| `dead_ind_weight` | `:870` 0.3 | `:759` 0.3 | 1 |
| `dead_col_weight` | `:871` 0.9 | `:760` 0.9 | 1 |
| **`close_ind_weight`** | `:873` **0.15** | `:762` **0.25** | **5/3** |
| **`close_col_weight`** | `:874` **0.45** | `:763` **0.75** | **5/3** |
| `lambda` | `:933` 0.0005 | `:793` 0.0008 | 8/5 |

Two facts this re-read established that the earlier table did not state, and
both matter:

1. **The dead weights are *not* scaled.** 0.3/0.9 in both fits. So the fix is
   specifically the *close* pair — scaling all four would have been wrong.
2. **The functional forms legitimately differ between the two fits, and the
   toolkit already ports each one correctly.** Single uses
   `pow(2*overlap-1, 2)` (`:892-894`), multi uses `pow(overlap-0.5, 2)`
   (`:783-785`) — the first is exactly 4× the second. The toolkit matches:
   `TrackFitting.cxx:7604-7614` (single) and `:6779-6781` (multi). This is worth
   pinning because `pow(2x-1,2)` vs `(x-0.5)²` *looks* like a porting slip and
   is not; only the coefficient was wrong.

**Call-path check** (that the scale-up lands on exactly the prototype's multi
path and no further): `dQ_dx_multi_fit` (`TrackFitting.cxx:5746`) has exactly one
caller, `do_multi_tracking:8267`; `dQ_dx_fit` (`:6904`) has exactly one caller,
`do_single_tracking:8613`. No single-track case routes through the multi fit, so
nothing outside the prototype's multi fit is touched.

### 12.2 The change — `clus/src/TrackFitting.cxx`

Internal ×5/3 in `dQ_dx_multi_fit`, **not** a second parameter pair:

```cpp
-    const double close_ind_weight = m_params.close_ind_weight;
-    const double close_col_weight = m_params.close_col_weight;
+    const double close_ind_weight = m_params.close_ind_weight * 5.0 / 3.0;
+    const double close_col_weight = m_params.close_col_weight * 5.0 / 3.0;
```

Three reasons for that choice over a new knob:

* it is the **same shape as the line 8 rows below**, `double lambda =
  m_params.lambda*8.0/5.0; // adjusted for multi-track fitting` — the port
  already expresses "multi needs stronger regularisation" this way, and the
  single/multi ratio is a property of the algorithm, not of a detector;
* the SBND config (`sbnd_track_fitting.json:43-44`) carries the **single-track**
  values 0.15/0.45, so it stays byte-identical — no config change, nothing for
  another detector to have to learn about;
* a second knob pair would let the two drift apart, which is exactly how the
  original divergence happened.

**FP-exactness, verified rather than assumed.** Written left-to-right so it is
`(0.15*5.0)/3.0`, which is exactly 0.25, and `(0.45*5.0)/3.0`, exactly 0.75 —
checked with a standalone `printf("%.20g")` program, both `== 0.25` / `== 0.75`
comparing true. (Here `*(5.0/3.0)` happens to be exact too, but the left-to-right
form is what the lambda line uses and what §3's `43000 × 5/43` convention
expects.) The multi fit therefore now runs on the prototype's literals bit for
bit, not on a rounded approximation of them.

**One liveness trace added**, at TRACE level, reporting how many overlap terms
actually fired and with which weights. It exists because of the failure mode
§12.3 guards against, and it is what makes that section's claim checkable.

### 12.3 Liveness first — proving the term fires before reading any diff

Every prior round in this doc had to defend against *attributing noise to a
change*. Round 6 carries the opposite risk: a within-noise result read as
success when the real cause is a stale library (M1) or a dead code path. So the
liveness proof came first, from the trace line:

```
dQ_dx_multi_fit: close-wire regulariser fired on 562 of 864 plane-pair terms
                 (145 3D positions); close_ind=0.25 close_col=0.75
```

Summed over all **145** `dQ_dx_multi_fit` calls on evt 388:

| | fired / total overlap terms | weights in use |
|---|---|---|
| baseline | 62 685 / 88 464 (**70.9 %**) | `close_ind=0.15 close_col=0.45` |
| fixed | 62 685 / 88 464 (**70.9 %**) | `close_ind=0.25 close_col=0.75` |

Two things at once. The term is **live on 71 % of terms**, so a null result would
have meant something was wrong. And the fired count is **identical to the unit**
between arms — the `overlap > overlap_th` gate is untouched, only the
coefficient moved. That is the cleanest possible isolation: the two binaries
differ in two literals and nothing else.

> **Arm construction (deliberate).** The baseline arm was built from the *same
> source as the fixed arm with only the `* 5.0 / 3.0` removed* — liveness counter
> and trace line included. It is behaviourally identical to `f07c0299` (counters
> and a TRACE log cannot change FP arithmetic) but guarantees the A/B isolates
> one edit. The `close_ind=0.15` line in the baseline log is the proof that the
> intended baseline is what actually ran.

### 12.4 Repro

```bash
wcbuild && ./build/clus/wcdoctest-clus
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for arm in work-dqdx388-base work-dqdx388-base-rep work-dqdx388-w53 work-dqdx388-w53-rep; do
  PR_EXTRA_STAGES=pr_display PR_JOBS=1 SBND_WCT_LOGLEVEL=trace \
    ./run_pr_chain_batch.sh work-nuecc48-prod0803 $arm data 388
done
python3 /home/xqian/tmp/leafdiff.py work-dqdx388-base/pr_evt388/calib-pr-evt388.json \
                                    work-dqdx388-w53/pr_evt388/calib-pr-evt388.json
```

`./build/clus/wcdoctest-clus` — **76 cases / 836 assertions, all pass.**
Freshness proof (M1): `local/lib/libWireCellClus.so` 07:57:05 >
`clus/src/TrackFitting.cxx` 07:56:31, `file` confirms ELF (M3).

| label | binary |
|---|---|
| `work-dqdx388-base` | close weights 0.15/0.45 (= `f07c0299` behaviour) |
| `work-dqdx388-base-rep` | repeat, same binary — noise floor |
| `work-dqdx388-w53` | close weights ×5/3 ⇒ 0.25/0.75 — what is committed |
| `work-dqdx388-w53-rep` | repeat, same binary — noise floor |

### 12.5 Result — dQ/dx moves well outside the noise floor; the vertex does not move at all

**The effect clears the floor by a factor ~17, and — more decisively — it lands
in field families the floor never touches:**

| comparison | leaf diffs (of 176 956) | top families |
|---|---|---|
| `base` vs `base-rep` (**same binary**) | **357** | 356 `proj/charge_pred`, 1 `showers/kine_dQdx` |
| `w53` vs `w53-rep` (**same binary**) | **651** | 647 `proj/charge_pred`, 2 `showers/kine_dQdx`, 2 `showers/total_length` |
| **`base` vs `w53`** (**the fix**) | **10 799** | 8912 `proj/charge_pred`, **812 `segments/points/dQ`**, **810 `segments/points/reduced_chi2`**, **122 `vertices/fit/dQ`**, **120 `vertices/fit/reduced_chi2`**, 12 `showers/kine_dQdx`, 2 `kine_energy_particle`, 1 `kine_reco_Enu`, 1 `showers/kine_best` |

`segments/points/dQ`, `reduced_chi2` and `vertices/fit/dQ` **never appear in
either same-binary comparison**. The count alone would have been suggestive; the
field families make it conclusive. Magnitude: **812 of 817** segment trajectory
points changed `dQ`, median relative change **1.68 %**.

> **Do not quote a "max 124 %".** The largest *relative* changes are
> small-denominator artifacts: the top five sit on points whose baseline `dQ` is
> 1022–3716, against a **median |dQ| of 30 663** — i.e. 3–12 % of a typical
> point, several of them straddling zero (`−1992.9 → 421.2`). The honest
> scale-free statement is the median, plus: **126 of 812 points move by more
> than 10 % of a typical `dQ`.**

**Determinism is intact on both sides** — `base` == `base-rep` and `w53` ==
`w53-rep` on every scalar below, and `nusel-evt388.tsv` is identical in *all
three* comparisons including the fix.

**The vertex is bit-identical, as §4.1 predicted:**

```
main_vertex   base  (-163.1961948517943, 31.460485459387076, 426.15822898239566)
              w53   (-163.1961948517943, 31.460485459387076, 426.15822898239566)   identical to the last digit
```

Topology unchanged: **84 segments, 127 vertices, 13 showers** in all four arms.

**What did move — selection and kinematics, via dQ/dx → PID:**

| field | base | w53 | delta |
|---|---|---|---|
| `nue_score` | 4.300936 | 4.300936 | **0** (identical) |
| `numu_score` | −1.854887 | −2.028229 | **−0.173** |
| `sig_2_score` | 0.587352 | 0.575374 | −0.012 |
| `stw_2_score` | 0.518467 | 0.526259 | +0.008 |
| `tro_1_score` | 0.143150 | 0.137883 | −0.005 |
| `tro_4_score` | 0.182464 | 0.181392 | −0.001 |
| `cosmict_flag` / `cosmic_flag` | 0 / 1 | 0 / 1 | 0 |
| `kine_reco_Enu` | 2109.351 MeV | 2109.283 MeV | −0.068 MeV |
| `kine_energy_particle` | — | — | 2 of 13 entries move (19.075→19.009, 0.2365→0.2345 MeV) |

Exactly the five tagger scores that consume track PID move; every other tagger
field is unchanged. The event keeps its classification and `nue_score` does not
move at all — as in rounds 1 and 2.

**Cost — none.** `Timer: Total wall-sec`, evt 388, `PR_JOBS=1`: base
9.120 / 9.308 s, fixed 9.186 / 9.070 s. Within-arm spread exceeds the
between-arm difference; the change is two multiplications hoisted out of the
loop. **≈0 %.**

### 12.6 Scope and what is NOT claimed

* **One event, no gate — by instruction.** The owner explicitly waived
  population validation for this round ("we focus on this one event for now ...
  We will need to continue do some improvements on the algorithm before we need
  to look at a larger sample"). Same standing as §7.5/§8.5.
* **Not bit-identical**, and unlike round 5 this one moves **scalars** by
  design. **This supersedes §11.8's gate PASS criterion** — that criterion said
  every scalar must match the rounds-1–4 baseline, which a dQ/dx regulariser
  change cannot satisfy. When §7–§12 are finally gated together, expect scalar
  movement throughout the dQ/dx → PID → kine chain, and judge it against a
  regenerated baseline rather than against §11.8's text.
* **"Now matches the prototype" is not "now more correct."** Evt 388 has no
  truth for dQ/dx. What is established is that the toolkit's multi-track fit now
  uses the prototype's multi-track regularisation strength, FP-exactly, on the
  path the prototype uses it. Whether SBND *wants* uBooNE's tuning is the same
  open question §4.2 raises, and it is not settled here.
* **The dQ/dx effect is large enough to matter for PID.** A 1.68 % median shift
  on segment `dQ`, with 126 of 812 points moving more than 10 % of a typical
  `dQ`, is not cosmetic; `numu_score` moving
  0.173 on a single event is the visible consequence. On a population this will
  move events across selection cuts.
* Untouched, as before: §3.3's `flag_front` and later rows, §3b T1–T8, §4.2,
  §4.3, §4.4, and §11.8's two container classes. `flag_front` and §4.2 are now
  explicitly **set aside by the owner**, not merely unqueued.
* **Noticed, not fixed** (an unrelated defect does not ride along): `:6779-6781`
  gates on `m_params.overlap_th` but hardcodes the matching `-0.5` offset in the
  quadratic. At SBND's configured `overlap_th = 0.5` the two agree, which is why
  this is invisible today; any other configured threshold would desynchronise
  gate and offset. Pre-existing, one line, worth a look in a later round.

### 12.7 Owner decision — the vertex-fit area is closed

After round 6 the owner ruled: *"I do not think that we need to worry about the
remaining vertex fitting divergencies, please feel free to make a note on the md
file about this."*

**What that closes.** Every remaining vertex-fit divergence in this document.
They are **accepted known divergences**, not pending work:

| item | where | status |
|---|---|---|
| `flag_front`: index identity vs 0.01 cm distance comparison (the distance form exists in the prototype but is commented out) | `MyFCN.cxx:395-398` ↔ prototype `:857-862` | accepted |
| `+0.5` half-wire offset dropped on stored `pu/pv/pw/pt` | `MyFCN.cxx:466-469` ↔ `:971` | accepted — global port convention, `TrackFitting.cxx:1315` drops it too |
| identity-by-index → identity-by-distance throughout `UpdateInfo` | `MyFCN.cxx:417-451` ↔ `:900-935` | accepted — forced, toolkit `WCPoint` has no index field |
| `m_fit_vertex_min_seg_length` (toolkit-only; **SBND sets 1.0 cm**) | `NeutrinoPatternBase.h:203` | accepted — inert on evt 388, never fired |
| `wcpt` re-snap to the nearest Steiner point (seed quantisation) | `MyFCN.cxx:474` | accepted — it *is* the 0.22 cm `fit_distance` residual |

**What that does *not* close**, stated explicitly so a later reader does not
over-apply the ruling: it is about *vertex fitting*. Still open and untouched —
~~**§3b T1–T8** (trajectory fitting: the dead charge-ratio veto, the wrong-index
dead-channel lookup, the extra `form_map_graph`, …)~~ **— closed by round 7,
§13**, ~~**§4.2** (the absent uBooNE calibration chain, separately set aside),
**§4.3**, **§4.4** (the unresolved time-bin convention), and **§11.8's two
container classes**~~ **— all closed by round 8, §14**.

**Why this is a reasonable place to stop, on the evidence.** The area was not
abandoned — it was worked and measured. The question that opened this document
("the SBND neutrino vertex looks a bit off") produced four vertex-fit fixes
across rounds 1–3, which together moved evt 388's neutrino vertex **0.854 cm**
and removed **4 spurious vertex fits**; §1/§2 cleared the mechanism the owner
originally suspected; and round 6 confirmed that even a substantial dQ/dx change
leaves the vertex **bit-identical**. The remaining divergences are each either
forced by a toolkit data-structure difference, a deliberate global port
convention, or measured inert on the event in hand.

**If the vertex is ever suspected again**, the recipes are all here and still
valid: the §2 trace recipe (which vertices were fitted, skipped, or vetoed, and
by which gate), the §7.6/§8.6 side-by-side display arms, and the row-by-row
prototype anchors above. Reopening any row is a matter of re-reading the two
trees at the cited lines, not of redoing the audit.

---

## §13 §3b T1/T2/T3/T6 FIXED — the trajectory fit's dead veto, wrong index, and destroyed trajectory (toolkit `23bd6783`, evt 18255/388)

The owner read the §3b triage and ruled: **fix T1, T2, T3 and T6**; leave **T4**
alone (*"the toolkit logic is OK, if you see a place to improve to make it more
robust, it is fine to do"*); **drop T5, T7, T8**. The framing was explicit —
*"I want to focus on bugs, not exactly the behavior change, since the latter, if
improved, is OK in toolkit."* That is the line this section applies: a
divergence where the toolkit does something *different and defensible* stays; a
divergence where toolkit code cannot do what it was written to do is a bug and
goes.

Unconditional, no knob, same as rounds 1–6.

### 13.1 The triage — all seven re-read, in both trees

This supersedes §3b's second trust tier. Every row below was read at the cited
lines in `prototype_base` and in the toolkit at `01ff88b1` before any edit.

| # | verdict | why |
|---|---|---|
| **T1** | **bug — fix** | `examine_segment_trajectory` fills `pss_vec` from `final_ps_vec` (`:4709`) and passes `p = final_ps_vec[i]` (`:4715`). The point under test **is** the comparison point ⇒ same projections ⇒ `c1 == c2` on all three planes ⇒ `ratio` is exactly 3 and `ratio_1` exactly 1, for every point, always. The cut `ratio/3 < 0.97 \|\| ratio_1 < 0.75` is arithmetically unreachable and the `p = ps_point` revert is a no-op. Prototype passes `init_ps_vec` (`multi_track_fitting.h:429`). |
| **T2** | **bug — fix (same line)** | `angle1` reads `pss_vec` (`:5084-5086`), so it measured the fold-back comparison on the *fitted* path instead of the initial one. Fixing T1's fill fixes T2. |
| **T3** | **bug — fix** | `m_3d_to_2d.find(i)` uses the **per-segment loop position** while the map is keyed by the **global** `count` from `form_map_graph`. Not an "index missing in the toolkit" case: `init_indices` is built at `:3948`, is correct, and is handed to `fit_point` at `:3979` — it simply was not passed on. |
| **T6** | **bug — fix** | `check_and_reset_close_vertices` replaced the segment's whole fit vector with **two** points (`:1231-1234`). The prototype (`:1383-1392`, and again at `:1158`) resets only the two vertex fit points. |
| **T4** | **keep, make non-silent** | Not gratuitous: it *compensates*. See §13.5. |
| **T5** | **drop** | Closed on evidence, and the audit row was wrong. See §13.7. |
| **T7** | **drop** | The `continue`s at `:3717`/`:3747` are labelled multi-APA crash guards (S1.15, S2.2) with no prototype counterpart to diverge from; `1/N` is a saner fallback than zero weight. |
| **T8** | **drop** | Consumers of `associated_2d_points` are accumulation loops and `.size()` checks only — no order-dependent branching. ULP-level fidelity, unfixable without changing the key type. |

**The strongest evidence that T1 is a port slip and not a design choice** is
inside the toolkit itself. Two independent tells:

1. The **face-crossing guard** at `:4862-4872` exists to catch *"the fit may have
   moved p to a different face than the reference point `pss_vec[i]`"*. Nobody
   writes a guard against a case they intend to make impossible.
2. The **single-track caller** (`:4499-4521`) passes pre-fit points and is
   correct. Same function, two callers, one right — an asymmetry, not a policy.

### 13.2 The change — `clus/src/TrackFitting.cxx`, `clus/inc/WireCellClus/TrackFitting.h`

**T1 + T2 — one line.** `examine_segment_trajectory`'s comparison path:

```cpp
// was: pss_vec.push_back(std::make_pair(final_ps_vec[i], segment));
pss_vec.push_back(std::make_pair(init_ps_vec[i], segment));
```

with the reference `(apa, face)` now taken from that same comparison point,
mirroring the single-track caller and making the face-crossing guard live:

```cpp
// was: auto test_wpid = m_dv->contained_by(p);
auto test_wpid = m_dv->contained_by(pss_vec[i].first);
```

`pss_vec` is not read again after this loop — the smoothing pass works on
`fine_tracking_path` / `temp_fine_tracking_path` — so the blast radius is
exactly `skip_trajectory_point`.

**T3 — thread the global index.** `skip_trajectory_point` gains an `int index`
parameter (the prototype has had one since `trajectory_fit.h:479`), and
`examine_segment_trajectory` gains `const std::vector<int>& init_indices`:

```cpp
bool flag_skip = skip_trajectory_point(p, apa_face, i,
                                       i < init_indices.size() ? init_indices[i] : -1,
                                       pss_vec, fine_tracking_path);
...
if (m_3d_to_2d.find(index) != m_3d_to_2d.end()) {
```

The single-track caller passes `i` explicitly, because there `form_map`
compacted `ptss` to the surviving points (`ptss = saved_pts`, `:3397`) so the
loop position *is* the key — that path is unchanged by construction.

**T6 — delete the rebuild.** The four lines that replaced `segment->fits()` with
`generate_fits_with_projections(segment, {start, end})` are gone; only the two
vertex fit points are reset, as in the prototype. This matters because
`organize_segments_path_{2nd,3rd}` rebuild `curr_pts` **from
`segment->fits()`** (`:1358-1361`), so the two-point collapse came back as a
straight line between the vertices. Both organisers already take their endpoints
from the vertex fits (`:1384-1401`), so the consistency the rebuild was reaching
for is supplied downstream anyway.

### 13.3 Liveness first — every fixed path proven to fire

Round 6's rule again, and it matters more here: this round **turns on a cut that
had never once fired**, so a null result would be indistinguishable from a
stale library (M1). Trace lines, from `work-tfix388-fix-trace`:

```
examine_segment_trajectory: segment 2 -- 42 point(s) in, 42 kept, 3 charge-reverted,
                            42 with a global index != loop position
```

Summed over **1168** `examine_segment_trajectory` calls / **14 840** points on
evt 388:

| what | count | share |
|---|---|---|
| **T1** — points the revived charge veto reverted (survivors only) | **2 146** | 14.5 % of points |
| **T3** — points whose global index ≠ loop position | **12 373** | **83.4 %** of points, in 926 of 1168 calls |
| points skipped by the angle/dead-plane gates | 456 | 3.1 % |
| **T6** — `check_and_reset_close_vertices` firings | **2** | both on 2-point segments |
| **T4** — points dropped by the pre-dQ/dx `form_map_graph` | **0** | inert on this event |

> **On the 2 146.** The per-segment counter is incremented *after* the skip
> gates, so it counts reverted points that **survived**. The raw trace carries
> **2 668** `charge revert` lines, which additionally include reverts on points
> that were then skipped, and reverts in the **single**-track path — where the
> cut was already live and correct before this round. 2 146 is the number to
> quote for "what the fix turned on"; do not read 2 668 − 2 146 as a discrepancy.

Read that T3 row carefully: the dead-channel lookup was reading **the wrong
point's plane quantities for 83 % of trajectory points**, or missing the map
entirely — and a miss sets all three planes "dead", which opens the
`angle > 45°` skip on any point past 45°. It failed in both directions.

T6's two firings both landed on segments that already had exactly 2 fit points,
so the collapse was a no-op *on this event*. The fix is still correct: the
condition is on the two **vertex fit** points, which says nothing about how many
interior points the segment has.

### 13.4 Repro and arms

```bash
wcbuild && ./build/clus/wcdoctest-clus
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for arm in base base-rep t3only t2only t12only fix fix-rep final final2; do
  PR_EXTRA_STAGES=pr_display PR_JOBS=1 \
    ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-tfix388-$arm data 388
done
# liveness (trace is needed for the clus.TrackFitting counters)
PR_EXTRA_STAGES=pr_display PR_JOBS=1 SBND_WCT_LOGLEVEL=trace \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-tfix388-fix-trace data 388
```

`./build/clus/wcdoctest-clus` — **76 cases / 896 assertions, all pass.**
Freshness proof (M1): `local/lib/libWireCellClus.so` **08:33:23** >
`clus/src/TrackFitting.cxx` **08:32:42**. The two baseline arms ran at 08:0x
against the **07:57:05** library, i.e. `01ff88b1` unmodified.

| label | binary |
|---|---|
| `work-tfix388-base` / `-base-rep` | `01ff88b1` — noise floor |
| `work-tfix388-t12only` | T1+T2 only (index still `i`) |
| `work-tfix388-t3only` | T3 only (`pss_vec` still from `final_ps_vec`) — with T6, which is a no-op here |
| `work-tfix388-t2only` | T2 (+T3, T6) but **not** T1: `ps_point = p` and `apa_face` from `p` restore base's dead veto and inert face guard (§13.6a) |
| `work-tfix388-fix` / `-fix-rep` | all four — what is committed |
| `work-tfix388-final` | rebuilt from the restored source after the attribution edits — **proves no `TEMP ATTRIBUTION` residue survived** |
| `work-tfix388-final2` | rebuilt again after the §13.6a split, from `git checkout`-restored source — 469 leaf diffs vs `final`, all in the noise families |
| `work-tfix388-fix-trace` | same binary at `SBND_WCT_LOGLEVEL=trace`, liveness only |

### 13.5 T4 — kept, and no longer silent

The owner kept the toolkit's behaviour. Round 7 supplies the *reason* it is
defensible, which the audit row lacked:

| | prototype | toolkit |
|---|---|---|
| `reset_fit_prop()` on a segment | `fit_index_vec.resize(size, -1)` — `resize` only *extends*, so existing indices **survive** (`ProtoSegment.cxx:1007-1010`) | `for (auto& fit : m_fits) fit.reset();` — `PR::Fit::reset()` sets `index = -1` **on every point** (`PRCommon.h:131-135`) |
| consequence before dQ/dx | indices reused directly by `dQ_dx_multi_fit` | indices must be rebuilt ⇒ the extra `form_map_graph` (`:8280`) |

So the extra pass is **compensating, not gratuitous**. It is also arguably the
better of the two: it re-derives the associations for the final post-`_3rd`
positions, whereas the prototype feeds post-`_3rd` points with **pre-`_3rd`
indices** — which, since `organize_segments_path_3rd` can *add* points
(`multi_track_fitting.h:1265-1274`), would leave a `-1` and throw on
`traj_pts.at(-1)`. That is a latent prototype-side fragility, recorded here and
not acted on.

The one real side effect is that `form_map_graph`'s zero-quantity point drop
(`:3201`) re-runs on the **final** trajectory, which the prototype never does.
The robustness improvement the owner allowed is to make that drop impossible to
miss:

```cpp
if (n_fits_after != n_fits_before) {
    SPDLOG_LOGGER_DEBUG(s_log,
        "do_multi_tracking: pre-dQ/dx form_map_graph dropped {} of {} "
        "trajectory point(s) with zero plane quantity", ...);
}
```

Measured on evt 388: **0 dropped**. T4 is inert here, and if it ever is not, the
log says so.

### 13.6 Result — the fix is large, and it is all T1/T2

Determinism holds on both sides (`base ≡ base-rep`, `fix ≡ fix-rep ≡ final` on
every scalar below).

| arm | segments | vertices | showers | seg points | main vertex (cm) | Δvtx | `nue_score` | `numu_score` |
|---|---|---|---|---|---|---|---|---|
| `base` | 84 | 127 | 13 | 817 | (−163.1962, 31.4605, 426.1582) | — | 4.3009 | −2.0282 |
| `t3only` | 84 | 127 | 13 | 817 | (−163.1962, 31.4605, 426.1582) | 0.0000 | 4.3009 | −2.0282 |
| `t12only` | 75 | 119 | 13 | 728 | (−163.2313, 31.4259, 426.2929) | **0.1434** | 4.3009 | **−2.8349** |
| `fix` | 75 | 119 | 13 | 728 | (−163.2313, 31.4259, 426.2929) | **0.1434** | 4.3009 | **−2.8349** |

**Attribution is clean: T1+T2 carry the entire effect on this event** — and
§13.6a splits that pair, because "T1+T2" was as far as the arms above could
resolve it and the answer turned out not to be the obvious one.

* **T3 alone is within the noise floor** — `base` vs `t3only` is 633 leaf diffs,
  all in the known-noisy families (`proj[]/charge_pred` 631, `showers[]/kine_dQdx` 2),
  against a same-binary floor of 662 (`base`/`base-rep`) and 584 (`fix`/`fix-rep`).
  Despite 83 % of points having had a wrong key.
* **T3 on top of T1/T2 is *not* inert**: `t12only` vs `fix` differs structurally
  (179 254 vs 179 266 leaves) — 8 segments of **cluster 92** change point count
  (33→18, 12→11, 18→13, 9→19, 13→27, 19→9, 27→20, 9→6) and segment ids
  renumber. Cluster 92 is not the neutrino cluster; `main_vertex`, `tagger` and
  the topology counts are identical between the two arms. The interaction is
  real — with T1/T2 fixed the fitted points differ, so the dead-plane gate lands
  on different points — it just does not reach this event's answer.
* **T6 is a no-op here** (both firings on 2-point segments).

**What moved in the physics.** Nine segments and eight vertices are gone, the
neutrino vertex moves **0.1434 cm**, and:

| field | base | fix |
|---|---|---|
| `numu_score` | −2.0282 | **−2.8349** |
| `nue_score` | 4.3009 | 4.3009 (unchanged) |
| `kine_reco_Enu` | 2109.28 MeV | **2909.80 MeV** |
| `kine_pio_flag` | 0 | **1** |
| `kine_pio_mass` | 0.0 | **125.71 MeV** |
| sub-BDT scores moved | — | **16** (`br3_3`, `br3_5`, `br3_6`, `lol_2`, `numu_1`, `pio_2`, `sig_1`, `sig_2`, `stw_2`, `stw_3`, `stw_4`, `tro_1`, `tro_2`, `tro_4`, `tro_5`) |

> ⚠️ **This is a big physics move and it is flagged, not tuned.** Evt 388 now
> reconstructs a **π⁰** (`kine_pio_flag` 0 → 1, mass 125.7 MeV) and its
> reconstructed neutrino energy rises **800 MeV**. That follows from the fix, it
> was not fitted to: the charge veto that reverts a fitted point onto charge was
> dead, and turning it on changes 14.5 % of trajectory points, which changes the
> shower/track topology feeding `kine`. Whether 2.9 GeV is the *better* answer
> for this event is a physics question this document does not settle — it is a
> hand-scan / population question, deferred with the rest.
> **RESOLVED (§13.12):** the owner compared the before/after Bee sets and ruled
> the new reconstruction *"much better"*. The π⁰ and 2 909 MeV are accepted.
> §13.11 separately establishes that the mechanism behind them is sound.

`nusel-evt388.tsv` is **identical** — the bundle table, flash assignment, and
the `nu-candidate` label do not move. Cost: **15 s → 13 s** wall (fewer
segments), peak RSS flat at 1.53 GB.

### 13.7 Owner decision — T4 kept, T5/T7/T8 dropped, trajectory area closed

The owner's instruction was to separate *bugs* from *behaviour changes*: a
toolkit divergence that is an improvement stays. Applying that:

| # | ruling | standing status |
|---|---|---|
| **T4** | **kept** — the toolkit logic is OK; robustness improvement accepted (§13.5) | accepted divergence, now instrumented |
| **T5** | **dropped** | and independently *closed on evidence*: `dQ_dx_multi_fit` rewrites vertex `dQ`/`dx`/`reduced_chi2` unconditionally (`:6888-6891`), so nothing stale reaches output or PID. The audit row's claim that the prototype resets `index` was wrong — corrected in place. |
| **T7** | **dropped** | multi-APA crash guards with no prototype counterpart; `1/N` beats zero weight |
| **T8** | **dropped** | ULP-level ordering; not fixable without changing `Coord2D` |

With §12.7 closing the vertex fit and this section closing the trajectory fit,
**doc pr/28's §3, §3b and §4.1 are all disposed of.** ~~What remains open is
§4.2 (set aside), §4.3, §4.4, and §11.8's container classes.~~ **Those went in
round 8 (§14); the doc is closed.**

### 13.8 Scope and what is NOT claimed

* **One event.** Everything above is evt 18255/388. No population statement is
  made or implied — and after a change of this size the owed valfast/1000 gate
  needs a **regenerated** baseline, since the pre-round-7 one no longer
  describes this code.
* **NOT bit-identical, and not intended to be.** This is a behaviour change by
  construction: it revives a cut that could not fire. §11.8's per-scalar PASS
  criterion is superseded here exactly as it was by §12.
* The `proj[]/charge_pred` family remains run-to-run noisy (~600 leaves,
  same-binary) — a known open item, unrelated to this round and not touched.
* The T3 interaction was measured only in the direction `t12only → fix`. A
  T3-alone-then-T1 ordering was not run; the fixes ship together.
* T6's correctness rests on the prototype reading, not on a measured effect —
  its two firings on this event were no-ops.
* Display arm regenerated on port 5017: **`work-tfix388-final`** (§13.9).

### 13.9 The display

Port **5017** now serves the post-fix event (M13: a fresh arm, the round-2 arm
`work-vtxfit388-r2` it replaced is untouched):

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./pr_display/serve_pr_display.sh 5017 work-tfix388-final/pr_evt388/calib-pr-evt388.json
# laptop: ssh -L 5017:localhost:5017 wcgpu1.phy.bnl.gov
#         http://localhost:5017/pr_display_viewer
```

What to look at, given §13.6: the PR graph is **75 segments / 119 vertices**
(was 84 / 127), the neutrino vertex has moved 0.1434 cm, and the particle-flow
panel now reports a **π⁰** with `kine_reco_Enu` 2.91 GeV. The pre-fix arm for a
side-by-side is `work-tfix388-base` — serve it on any free port.

### 13.10 Bee links — evt 18255/388 before and after

Uploaded on request, so the round-7 change can be examined in the 3-D display.
Both sets are the PR chain's own `mabc-pr.zip` (layers: `clustering-global`,
`shower_track-global`, `track_fit-global`, `vertices-global`, dead areas, and
the `mc.json` particle-flow tree).

| | binary | arm | Bee set |
|---|---|---|---|
| **before** | `01ff88b1` | `work-tfix388-base` | https://www.phy.bnl.gov/twister/bee/set/62ecc490-8ca7-4541-b08f-db2220f55b1f/event/list/ |
| **after** | `23bd6783` | `work-tfix388-final` | https://www.phy.bnl.gov/twister/bee/set/7511876b-f244-48b5-8982-c71b95736408/event/list/ |

What to compare, from §13.6: **84 → 75 segments**, **127 → 119 vertices**, the
neutrino vertex 0.1434 cm away, and a **π⁰** now reconstructed
(`kine_reco_Enu` 2109 → 2910 MeV). The `track_fit-global` layer is where the
trajectory change lives — it shrank 38.6 kB → 34.7 kB while `shower_track-global`
grew 290.7 kB → 315.7 kB, i.e. the moved points are track-to-shower
re-attributions, not points vanishing.

**Only this one event exists for round 7.** No population run was made, so there
is no set of "events with a large move" to link — evt 388 is the whole sample.
Finding others requires a small-group run against a pre-round-7 baseline.

### 13.6a Splitting T1 from T2 — the π⁰ is **T2**, not T1

§13.6 could only resolve the pair, because T1 and T2 share one line: both read
`pss_vec`. The obvious reading — that the headline move comes from T1, the
spectacular one that was arithmetically dead — is **wrong**.

A fourth arm separates them. `work-tfix388-t2only` keeps the corrected
`pss_vec` fill (so `angle1` is measured on the initial path = **T2 fixed**) but
restores base behaviour for T1's two mechanisms: `ps_point = p` (charge veto
dead again) and `apa_face` from `p` (face guard inert again). T3 and T6 are on
in both it and `t3only`, and `t3only ≡ base`, so the two step-deltas below are
exact isolations.

| arm | fixes on | seg | vtx | pts | Δvtx (cm) | `numu_score` | `kine_reco_Enu` | `kine_pio_flag` |
|---|---|---|---|---|---|---|---|---|
| `base` | — | 84 | 127 | 817 | — | −2.0282 | 2109.3 | 0 |
| `t3only` | T3+T6 | 84 | 127 | 817 | 0.0000 | −2.0282 | 2109.3 | 0 |
| `t2only` | **T2**+T3+T6 | 83 | 125 | 786 | 0.2043 | −2.1573 | **2907.4** | **1** |
| `t12only` | T1+T2+T6 | 75 | 119 | 728 | 0.1434 | −2.8349 | 2909.7 | 1 |
| `fix` | all four | 75 | 119 | 728 | 0.1434 | **−2.8349** | 2909.8 | 1 |

Step-deltas (`nue_score` is 4.3009 in every arm):

| step | isolates | segments | vertices | vertex moves | `numu_score` | `kine_reco_Enu` |
|---|---|---|---|---|---|---|
| `t3only → t2only` | **T2** | 84 → 83 | 127 → 125 | 0.2043 cm | −2.028 → −2.157 | **2109 → 2907 MeV**, π⁰ appears |
| `t2only → fix` | **T1** | 83 → 75 | 125 → 119 | 0.1973 cm | **−2.157 → −2.835** | 2907 → 2910 MeV |

So the two bugs do different jobs, and neither is a subset of the other:

* **T2 — the fold-back comparison angle — produces the π⁰ and the +800 MeV.**
  One segment and two vertices change, 31 trajectory points move, and that is
  enough to re-attribute a shower and put `kine_pio_flag` at 1. A *small*
  topology change with a *large* downstream consequence.
* **T1 — the revived charge veto — produces the bulk of the topology change**
  (8 more segments, 6 more vertices, 58 more points) and the bulk of the
  `numu_score` shift, but adds only 3 MeV to the energy. A *large* topology
  change with a *small* energy consequence.
* The vertex is not monotone: T2 moves it 0.2043 cm from base, then T1 moves it
  0.1973 cm again, landing 0.1434 cm from base. The net is smaller than either
  step — do not read the 0.1434 cm as "the size of the change".

**Why this matters for review.** T1 is the dramatic finding (a cut that could
never fire, 2 146 points reverted); T2 is one line of the same edit and reads
like a footnote to it. On this event the footnote is what moved the physics.
Any future population study should carry both labels separately rather than
attributing to "the T1 fix".

### 13.11 Is T2 supported by physics? — yes, and the old code was self-referential

Asked directly by the owner after §13.6a showed T2 carries the π⁰. The answer
is not a matter of taste; the cut has a definite meaning and the old code
destroyed it.

**What the cut is for.** Inside `skip_trajectory_point`,

```
angle   = kink at this point in the ACCEPTED OUTPUT path   (fine_tracking_path, fitted)
angle1  = kink at the same place in the REFERENCE path     (ps_vec)
if (angle > 160)          -> skip   // absolute: a fold-back is unphysical
if (angle > angle1 + 90)  -> skip   // relative: did the FIT bend it far more than the input?
```

The relative arm is a **regression guard**, and the reason it exists is
physical: real tracks *do* kink — scatters, decay vertices, a shower's first
branch. An absolute angle cut alone would shave those off. The relative form
tolerates a large `angle` **when the seed path already bent there** (a real
feature), and fires when the fit invented the bend (an artifact). That only
works if `angle1` comes from a reference the fit did not produce.

**What the toolkit was doing.** With `pss_vec` filled from `final_ps_vec`,
`angle1` was computed from the *fitted* path — the same path `angle` measures.
Probe build (`work-tfix388-t2probe`, both references computed side by side),
9 972 points on evt 388:

| | median | mean | within 5° of `angle` |
|---|---|---|---|
| `angle` (fitted output) | 35.00° | 41.73° | — |
| `angle1` from the **seed** path (prototype, now fixed) | **11.05°** | 19.72° | **10.9 %** |
| `angle1` from the **fitted** path (old toolkit) | 39.16° | 50.59° | **61.7 %** |

**`angle1_fit` was *bit-identical* to `angle` for 5 301 of 9 972 points (53 %)** —
whenever no point had been skipped, `fine_tracking_path`'s last two entries
*are* `final_ps_vec[i-1]` and `[i-2]`, so the two vectors are the same vectors.
For those points the test was literally `angle > angle + 90`, which cannot fire.

**The decisive slice.** Take the 180 points where the fit produced a severe kink
(`angle > 120°`) — exactly the population the guard exists for:

| reference for `angle1` | median `angle1` there | relative cut fires |
|---|---|---|
| seed path (fixed) | **42.04°** | **103** |
| fitted path (old) | **128.25°** | 40 |

The seed says *"the input was smooth here, the fit bent it 120°+"* — artifact.
The fitted path says *"everything around here is bent 128°"* — no anomaly. **The
artifact was excusing itself.** That is the failure mode, measured, not argued.

Over all points the relative cut fires **372** times with the seed reference vs
**46** with the fitted one, and the change is nearly one-directional:
**327 seed-only, 1 fitted-only, 45 both.** The fix is a recovery of sensitivity,
not a trade of one population for another.

**Two honest qualifications.** The absolute `angle > 160°` fold-back arm never
depended on `angle1` and fired 53 times either way — the old code was not
defenceless, it had lost one of two arms. And none of this proves evt 388's π⁰
is *correct*; it establishes that the mechanism producing it is the one the
algorithm was designed around. The event-display check below is what settles
the outcome.

### 13.12 Owner verdict — accepted from the display; nothing to flip

Having compared the §13.10 Bee sets, the owner ruled: *"the new one is actually
much better than the old one. So these changes are all improvements."*

**No configuration change is required, and none was made.** Round 7 shipped
**unconditionally**, like rounds 1–6: `23bd6783` touches only
`clus/src/TrackFitting.cxx` and `clus/inc/WireCellClus/TrackFitting.h` — it adds
no `get(config, ...)` key, no `m_params` field, and no jsonnet. There is no
default-OFF knob to turn on. **The SBND production chain has had the fixed
behaviour since `23bd6783` was pushed**; `cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json`
is untouched by this round (its last change is `564012fe`, unrelated), and every
existing `skip_*` parameter keeps its value.

So the production status of the four fixes is simply:

| item | production state |
|---|---|
| T1 charge veto revived | **on** — unconditional, `23bd6783` |
| T2 fold-back reference restored to the seed path | **on** — unconditional, `23bd6783` |
| T3 dead-channel lookup keyed by the global index | **on** — unconditional, `23bd6783` |
| T6 close-vertex reset keeps the trajectory | **on** — unconditional, `23bd6783` |
| T4 zero-quantity drop | unchanged behaviour + a DEBUG counter |

What the verdict *does* change is the standing of the numbers in §13.6/§13.6a:
evt 388's **π⁰ and 2 909 MeV are now the accepted reconstruction**, not a
flagged anomaly. The ⚠️ block in §13.6 stands as the record of what moved and
why it was flagged at the time; it is resolved here, on the display, by the
owner.

**Still owed, and now more clearly:** a population pass. One event was accepted
by eye; the valfast/1000 gate needs a regenerated baseline (stale since round 6,
doubly so after round 7) before any efficiency statement can be made.

---

## §14 The last open items — nondeterminism FIXED, both container classes measured, §4.2–§4.4 closed (toolkit `22249ff4`, evt 18255/388 + 5 more)

Owner instruction, after reading the round-7 report: *"the only thing left that
we should look at is [the `m_cluster_fitted_charge_2d` nondeterminism] … Then
the two container classes §11.8 left for a next round … for the rest, I think we
can mark them closed. Unless you see a major bug etc."*

So this round does three things: **fix** the nondeterminism, **measure** the two
container classes rather than convert anything, and **close** §4.2/§4.3/§4.4 —
with §4.4 *settled* rather than merely marked closed, because it was the only
remaining open item that could have been a physics defect.

### 14.1 The change — `clus/inc/WireCellClus/TrackFitting.h`, `clus/src/TrackFitting.cxx`

Three edits, no new knob, no config:

```cpp
// TrackFitting.h -- the per-cluster snapshot map
 std::map<Facade::Cluster*,
          std::map<APAFacePlane, std::map<WireTime, FittedCharge2D>>,
+         PR::ClusterPtrCmp>
     m_cluster_fitted_charge_2d;

// TrackFitting.h -- the per-cell cluster association
-    std::set<Facade::Cluster*> clusters;
+    std::set<Facade::Cluster*, PR::ClusterPtrCmp> clusters;
```

`PR::ClusterPtrCmp` (`PRShower.h:227-231`) orders by `get_cluster_id()`. It is
the comparator `m_clusters` / `m_loaded_clusters` in the *same class* already
use (`TrackFitting.h:591-592`), so nothing new is introduced.

The second edit changes no output today, and for a stronger reason than "the
consumer sorts": `PrDisplayDump::dump_proj` emits
`min(cids)` (`:841-845` — sort, then `cids.front()`), and an ident-dedup cannot
change a minimum. What it does **not** cover is worth stating, because the guard
below does not reach it: this set's keys come from `global_rb_map`'s blobs, not
from `m_cluster_filter`, and `enumerate_idents()` restarts at 1 **per grouping**
(`MultiAlgBlobClustering.cxx:2444-2446` loops `ensemble.children()`), so a
cross-grouping ident collision is structurally possible here in a way it is not
for the map. Harmless for a minimum; **not** harmless for the next consumer that
counts or enumerates the set. The comparator is there to make that consumer's
order deterministic, not to make the set collision-proof.

Third edit — an explicit guard rather than an assumption
(`TrackFitting.cxx`, end of `fill_fitted_charge_2d`):

```cpp
auto held = m_cluster_fitted_charge_2d.find(m_cluster_filter);
if (held != m_cluster_fitted_charge_2d.end() && held->first != m_cluster_filter) {
    SPDLOG_LOGGER_WARN(s_log,
        "fill_fitted_charge_2d: cluster ident {} is shared by two live clusters; "
        "the earlier snapshot ({} plane group(s)) is being discarded", ...);
}
```

**Why the guard, and why it can't fire.** Keying a *set* by ident degrades
gracefully — a duplicate silently drops one cluster. Keying this *map of maps*
by ident does not: the second cluster would discard the first's entire snapshot,
every cell of it. That severity does not transfer from `m_clusters`' precedent,
so it is checked rather than argued. It cannot fire because the map lives
entirely inside one visitor's `visit()` — `TaggerCheckNeutrino` fills it, calls
`assemble_fitted_charge_2d()` (`:886`) and hands the fitter over (`:889`) — and
`Grouping::enumerate_idents()` runs only *between* visitors
(`MultiAlgBlobClustering.cxx:2445`), never while entries are held. Idents are
dense and unique at the instant of either write (doc 53).

### 14.2 Blast radius — verified here, not inherited from `PrDisplayDump`

`PrDisplayDump.cxx:784-787` asserts the merged map is diagnostic-only. That
sentence was written in doc pr/26 and has been carried ever since; it is the one
claim in this item nobody re-derived. Re-checked at this HEAD —
`get_fitted_charge_2d()` has exactly **two** callers:

| caller | what it does with it | reaches a verdict? |
|---|---|---|
| `PrDisplayDump.cxx:798` | the display's 2-D `proj[]` panels | no — dumper, default OFF |
| `TaggerCheckSTM.cxx:746` | accumulates into `m_acc_fitted_charge`, merged at `:528` into the named `"stm"` `TrackFitting` slot for `SbndMagnifyTrackingVisitor` | no — gated on `save_stm_fit`, **C++ default false**, and `wct-pr-perevt.jsonnet:9` classes it as a *"pure diagnostic output"* |

No tagger flag, no Bee layer, no pctree tensor reads it. **Confirmed
diagnostic-only.** That is also why no A/B gate ever caught the defect.

### 14.3 Repro and arms

```bash
wcbuild && ./build/clus/wcdoctest-clus
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for arm in work-p14-fix work-p14-fix-rep work-p14-fix-rep2; do
  PR_EXTRA_STAGES=pr_display PR_JOBS=2 \
    ./run_pr_chain_batch.sh work-nuecc48-prod0803 $arm data 388
done
python3 /home/xqian/tmp/leafdiff.py work-p14-fix/pr_evt388/calib-pr-evt388.json \
                                    work-p14-fix-rep/pr_evt388/calib-pr-evt388.json
```

`./build/clus/wcdoctest-clus` — **76 cases / 896 assertions, all pass.**
Freshness proof (M1): `local/lib/libWireCellClus.so` **09:58:38** >
`clus/src/TrackFitting.cxx` **09:46:46** > `TrackFitting.h` **09:46:42**.

The "before" arms are round 7's own same-binary repeats
`work-tfix388-final2` / `-final3`, i.e. `23bd6783` unmodified — no new baseline
run was needed and none of them was overwritten (M13).

### 14.4 Result — the `charge_pred` noise floor goes to **zero**

The defect's claim is run-to-run movement, so the demonstration is a
**same-binary repeat**, not an A/B:

| pair | binary | leaf diffs (of 179 266) | where |
|---|---|---|---|
| `final2` vs `final3` | `23bd6783` (**before**) | **593** | 591 `proj[]/charge_pred[]`, 1 `showers[]/kine_dQdx`, 1 `showers[]/total_length` |
| `p14-fix` vs `-fix-rep` | fixed | **2** | 2 `showers[]/kine_dQdx` |
| `p14-fix` vs `-fix-rep2` | fixed | **3** | 2 `showers[]/kine_dQdx`, 1 `showers[]/total_length` |
| `-fix-rep` vs `-fix-rep2` | fixed | **3** | 2 `showers[]/kine_dQdx`, 1 `showers[]/total_length` |

**`proj[]/charge_pred[]` is 0 in all three fixed pairs**, from 591. Every
leaf-diff table in §7.4, §8.4, §9.3, §11.6, §12.5 and §13.6 had to carry a
`charge_pred`-dominated noise floor of 268–825 leaves; on `23bd6783` itself the
same-binary floor measured **413, 469 and 593** on different pairs (§13.4's
arms and this round's) — the point is the *range*, not the largest. From here
it is **2–3**.

**Selection-level check, the same one rounds 4 and 5 closed with.** `cmp` on the
artifacts the owner actually reads:

| file | `work-tfix388-final3` (pre-fix) vs `work-p14-fix{,-rep,-rep2}` |
|---|---|
| `nusel-table.tsv` | **byte-identical**, all three |
| `nusel-events.tsv` | **byte-identical**, all three |

So "diagnostic-only" is not just a two-caller grep — the selection output is
unchanged across the fix, and unchanged between repeats of it.

**The A/B against pre-fix moves `charge_pred`, by construction —
that is the fix, not a FAIL.** `final3` vs `p14-fix` is 655 leaf diffs, 651 of
them `charge_pred`, the remaining 4 in the residual families below. The winner
of a cross-cluster overlap is now *defined* (lowest cluster ident last) instead
of arbitrary; a diff there is the expected signature.

### 14.5 The residual — a *second*, smaller nondeterminism, and it is not diagnostic

2–3 leaves survive: **`showers[]/kine_dQdx`** (2, every pair) and
**`showers[]/total_length`** (0–1). These are **physics** leaves, not display
ones, and they were already inside the old floor (§8.4 listed 2 `kine_dQdx` and
1 `kine_reco_Enu`; §12.5 listed 1). Fixing `assemble_fitted_charge_2d` does not
touch them, and this round makes them visible for the first time by removing the
591 leaves that were burying them.

**Not diagnosed here** — it is a new item, not one of the three the owner
listed, and CLAUDE.md is explicit that an unrelated defect does not ride along.
Recorded as the next determinism item with its own starting point: two shower
quantities move while every vertex, segment, trajectory and tagger leaf is
bit-stable, which points at shower *membership* order rather than at any fit.

### 14.6 Class 1 — pointer-keyed containers: 116 declarations, 18 traversal sites, **2** that can move a result

§11.8 left "the rest are unaudited". Censused at this HEAD: every ordered
`std::map`/`std::set` in `clus/` keyed on `Cluster*` / `Blob*` / `Segment*` /
`Vertex*` / `Shower*` **or on the `…Ptr` aliases** — `std::less<shared_ptr<T>>`
compares `get()`, so a `std::set<SegmentPtr>` is address-ordered exactly like a
`std::set<Segment*>` — minus those already carrying a comparator
(`ClusterPtrCmp`, `cluster_less_functor`, `ClusterLess`, `SegmentIndexCmp`, …),
then cross-checked for **both** range-`for` and iterator-based traversal:

| | count |
|---|---|
| bare pointer/`Ptr`-keyed ordered containers | **116** |
| distinct traversal loops over one of them | **18** |
| …of which false positives (name reused in another scope, or the loop is commented out) | 2 |
| …**order cannot be observed** | **11** |
| …**tie-only** | **3** |
| …**can move a result** | **2** |

The 98 never-traversed declarations are compliant as they stand: CLAUDE.md §2's
rule is about *iteration*, and they are pure `find`/`count`/`[]`/`erase`.

**Order cannot be observed (11).** Removal/erase loops where each element is
independent — `NeutrinoVertexFinder.cxx:1875` (`remove_segment`/`remove_vertex`),
`:3718`, `:3873`, `NeutrinoDeghoster.cxx:617`,
`NeutrinoShowerClustering.cxx:452/1960/2072` (`showers.erase`); pure counting
(`:394`); a body that acts only on the single element equal to a
previously-chosen maximum, where that maximum's tie-break is **id-based, not
pointer-based** (`:407-410` compares `start_segment()->id()`), so the selection
is order-independent too (`:418`); an independent per-segment side effect
(`PRSegmentFunctions.cxx:2190`, one point cloud per segment); a running `min`
over distances (`:1756`, and `NeutrinoVertexFinder.cxx:1850` — the §11.7 loop);
an accumulation of exactly-representable `0.5`/`1.0` terms, so the FP sum is
order-independent (`NeutrinoVertexFinder.cxx:602`); and one that copies into a
vector which is **then explicitly sorted by segment id**
(`NeutrinoShowerClustering.cxx:661`).

**Tie-only (3)** — a strict `>` selection, so the *first* maximum wins and only
an exact tie can flip:

| site | selection |
|---|---|
| `NeutrinoVertexFinder.cxx:689` | nested loop over `map_in_segment_dirs` × `map_out_segment_dirs`, keeps the pair with `angle > max_angle` |
| `NeutrinoVertexFinder.cxx:3511` | `snap_map` → `snapped`, later scanned for a min-z and scored |
| `NeutrinoDeghoster.cxx:61` | copies into a vector, `std::sort`s by total length — `std::sort` is not stable, so equal-length clusters can swap, and the result is the deghost **processing order** |

**Can move a result (2)** — the first is **FIXED in §14.12** on the owner's
instruction; the second is reported, not fixed:

* **`NeutrinoVertexFinder.cxx:2934` — the one worth calling a bug.**
  `used_segments` (`:2839`, a bare `std::set<SegmentPtr>`) is traversed in
  **address order** and the body **mutates**: `particle_info()->set_pdg(13)`,
  `set_mass`, `unset_flags(kShowerTopology)`, and
  `change_daughter_type(graph, vtx, sg1, 13, …)` at `:2942`/`:2945` — which
  propagates the type change through the graph. A later iteration reads
  `current_pdg` (`:2953`) that an earlier one may have written, so the outcome
  depends on heap layout. This is the same defect class rounds 3–5 fixed at
  `:2333`/`:2374`, and it survived because those rounds swept
  `boost::out_edges` / `boost::edges` / `boost::vertices` / `graph_nodes` — not
  `std::set<SegmentPtr>`. **The fix is `PR::IndexedSegmentSet`**, and §14.7's
  census says it is safe here (the members come from `vertex_segments`, i.e.
  graph edges, so every index is valid and unique). **FIXED — §14.12**, by
  sorting the *iteration* rather than retyping the container, so `find()` at
  `:2905` keeps pointer identity.
* **`GroupingHelper.cxx:27`** — `orig_to_shadow` is traversed and each iteration
  calls `original.separate(...)`, which **creates clusters**; creation order sets
  their idents. Real, but a different subsystem (retile/shadow), not
  vertex/trajectory/dQ-dx.

So the "unaudited rest" is not a sweep waiting to happen — but it is not empty
either, and the first census this round ran **missed the `…Ptr` aliases
entirely** (66/3 instead of 116/18). Recorded so the next reader does not repeat
the mistake: `std::set<SegmentPtr>` is a pointer-keyed container.

### 14.7 Class 2 — index-keyed containers: the hazard is **`SIZE_MAX`**, not the inherit path

§11.8 named six pre-existing sites and warned that the obvious fix for class 1
walks into this class. Classified by **use**, per §11.7's rule:

| site | container | used for | aliasing-sensitive? |
|---|---|---|---|
| `NeutrinoKinematics.cxx:93` | `map<SegmentPtr,ShowerPtr,SegmentIndexCmp>` | `find()` at `:175`, `:219` | yes — identity lookup |
| `PRSegmentFunctions.cxx:1916-1918` | three caches | `find()` at `:1962`, `:1990`, `:2007` | yes — identity lookup |
| `PRShower.h:223` `ShowerSegmentMap` (`TaggerCheckNeutrino.cxx:557`) | segment → shower | `count()`/`find()` throughout the taggers | yes — identity lookup |
| `NeutrinoShowerClustering.cxx:1677` `map_segment_new_shower` | segment → shower | `operator[]` read at `:1803` | yes — identity lookup |
| `NeutrinoShowerClustering.cxx:2167` `map_merge_seg_shower` | segment → shower | insert-dedup + iterate | yes — insert-dedup |
| `NeutrinoTaggerSSM.cxx:608` `all_ssm_sg` | segment → flags | insert at `:701`, iterate at `:901` | insert-dedup only |

Then measured, with a temporary probe (`WCT_SEGIDX_PROBE=1`, since reverted —
`git checkout` of `PRGraph.{h,cxx}`, `NeutrinoKinematics.cxx`,
`PRSegmentFunctions.cxx`, `NeutrinoVertexFinder.cxx`; `grep -rn "TEMP
PROBE\|WCT_SEGIDX" clus/` is empty) on **six** events —
**388, 239794, 172230, 271851, 54095, 163543**:

| measurement | result |
|---|---|
| `PR::add_segment` "edge already existed" path fires (the §11.7 aliasing source) | **2 times in 6 events** — once in evt 388 (index 19), once in evt 163543; both displacing a *distinct* segment |
| distinct `SegmentPtr` reachable at the kinematics stage (graph edges ∪ every shower's member sets) vs distinct `get_graph_index()` | **equal in all 6** — 75/78/46/57/81/57. **Zero** shadowed segments |
| comparisons through `SegmentIndexCmp`/`VertexIndexCmp` involving an **unindexed** node | **0**, whole PR chain, all 6 events |

So every currently index-keyed container is safe *on this manifest*: the
displaced segment never survives to co-occur with its replacement.

**But the census turned up a bigger hazard than the inherit path.**
`PRSegment.h:153` and the vertex equivalent default `m_graph_index` to
`std::numeric_limits<size_t>::max()`. **Every segment that has not yet been
added to a graph carries the same index**, so *any* number of them compare
equal. That is a far broader collision source than the rare "edge already
existed" path, and it has a provable live site:

```cpp
// NeutrinoOtherSegments.cxx:451-460
// Create segment (not yet in graph)
auto new_seg = create_segment_for_cluster(cluster, dv, path_points);
...
existing_segments.push_back(new_seg);        // index is still SIZE_MAX here
```

`existing_segments` there is a `std::vector<SegmentPtr>` (iterated at `:332`,
`:707` in insertion order — deterministic, correct as written). Converting *it*
to an `IndexedSegmentSet` would collapse **every** not-yet-added segment into
one entry. So the rule §11.7 stated by example has a sharper form:

> **An index-keyed segment container is unsafe exactly when a segment can enter
> it before `PR::add_segment` has given it an index.** `SIZE_MAX` is shared by
> all of them. Check that before checking anything else.

### 14.8 §11.7's *mechanism* did not reproduce — flagged, not silently corrected

§11.7 attributes the 1.2 GeV `kine_reco_Enu` swing on SBND evt 239794 to index
aliasing in `eliminate_short_vertex_activities`'s `existing_segments`. Probing
that exact container at that exact call site
(`NeutrinoVertexFinder.cxx:2289`, immediately before the call):

| event | `existing_segments` distinct ptr | distinct index | shadowed | unindexed |
|---|---|---|---|---|
| 388, **239794**, 172230, 271851, 54095, 163543 | 1 call each | equal | **0** | **0** |

On the cited event the set holds no two segments sharing an index, so an
`IndexedSegmentSet` would have held exactly the same elements, and `find()` at
`:1703`/`:1827`/`:2460` would have answered identically. The remaining
difference is iteration order at `:1850` — and that loop takes a running `min`
over three distances, which §11.7 itself calls order-insensitive.

**What stands and what doesn't.** The *observation* (the swap moved
`kine_reco_Enu` by 1.2 GeV, caught by the A/B) is a measurement and stands. The
*conclusion* — judge each site by how it is used, never bulk-convert — is the
right rule and is reinforced by §14.7. The *stated cause* is **not confirmed**
and should not be cited as established. The likeliest alternative, given §14.7,
is that the intermediate build also converted a container that holds
not-yet-added segments (`NeutrinoOtherSegments.cxx:460` is the one such site,
and it is confusingly *also* called `existing_segments`), where the `SIZE_MAX`
collapse is certain rather than rare.

Coverage limit, stated rather than glossed: the probe reports one call per
event, at one site, on six events. It cannot prove the aliasing never happens —
it shows it did not happen where the doc says it did.

### 14.9 §4.4 SETTLED — `slice_index` is a slice **start** tick, and the two grids agree

§4.4 asked one discriminating question: is the `time_slice` in
`CoordReadout(apa, time_slice, channel)` the slice *start* tick or its *centre*?

**It is the start tick**, and both producers agree:

```cpp
// PointTreeBuilding.cxx:326  and  aux/src/SamplingHelpers.cxx:286
const auto& slice_index = slice->start() / tick;      // NOT an ordinal, NOT a centre
```

`Grouping::get_overlap_good_ch_charge` (`Facade_Grouping.cxx:901-913`) reads
that array straight through, and `TrackFitting.cxx:795` uses it as the
`CoordReadout` key — so `row.time` is a multiple of the slice span (4 ticks for
SBND, `img.jsonnet:133`).

**The answer to "is the erf window half a slice off centre" is no, because the
3-D side is on the same grid**:

| | anchor | grid |
|---|---|---|
| readout key `tbin = row.time` | `SamplingHelpers.cxx:286` | slice start tick |
| ctpc point `x` | `SamplingHelpers.cxx:306` — `time2drift(…, slice->start())` | slice start tick |
| the fit's data term | `TrackFitting.cxx:3518` — `scaling * (it->time - offset_t)` | slice start tick |
| `t_center = offset_t + slope_t·x` | `TrackFitting.cxx:522-523` | same, no `+0.5·nt` |

So a point where the fit puts it has `t_center == tbin`, and the symmetric window
`[tbin − nt/2, tbin + nt/2]` (`TrackFitting.cxx:5216-5217`) is centred on it.
Self-consistent.

**What genuinely differs from the prototype is the bin convention, not an
offset.** Put the two side by side and the point is one sentence: the prototype
integrates `[tbin, tbin+1]` and gates on `fabs((tbin+0.5) − t_center)`
(`PR3DCluster_dQ_dx_fit.h:164-168`) — a one-bin-wide window centred on
`tbin+0.5`, the bin centre under **lower-edge** labelling. The toolkit
integrates `[tbin ± nt/2]` and gates on `fabs(tbin − t_center)` — a one-bin-wide
window centred on `tbin`, the bin centre under **centre** labelling. **Both are
one-bin-wide windows centred on the bin centre.** They differ only in how bins
are *labelled*, and the labelling is measurably consistent between the data side
(`row.time`) and the model side (`t_center`) in the toolkit. This is the *same*
deliberate convention change the toolkit already applies to wires, where it is
documented in place (`TrackFitting.cxx:5205-5208`, `:5231` "All boundaries shift
by -0.5 due to bin convention change") and compensated in the offset
(`offset_u = -(center_u + 0.5·pitch_u)/pitch_u`, `:504`, `:510`, `:516`). The
time offset carries no such term (`:523`) — and correctly so, because the time
grid was never shifted in the first place.

**Residual, flagged for the owner, deliberately not decided here (escalation
rule 4).** Representing a slice's charge at its **start** rather than its centre
is a *global imaging-time* convention (`SamplingHelpers.cxx:306`), applied
identically to blob `x`, to the fit's data term and to this window. If the
physically right choice is the slice centre, then every reconstructed `x` is
biased by half a slice — 2 ticks = 1 µs ≈ **0.156 cm** at SBND's 1.563 mm/µs —
uniformly, everywhere, and the dQ/dx window is merely one of many places it
shows. It is **not** a dQ/dx port defect and fixing it here would be wrong.
Two readings, both defensible: (a) the start tick is the correct label because
the charge in a slice is attributed to when the slice *opened*, and the
convention is at least uniform; (b) the charge is deposited over the whole span,
so the centre is the unbiased estimator and the toolkit carries a systematic
half-slice `x` offset. **And it is not a free correction either way**: a uniform
half-slice offset is exactly the kind of term a *calibrated* drift velocity and
`time_offset` absorb, so "fixing" the convention in isolation would desync the
calibration that was tuned on top of it. Whoever schedules this has to move the
convention and the calibration together. **§4.4 is closed as a dQ/dx question; the global
convention is a separate item, and the owner's call.**

### 14.10 §4.2 and §4.3 — closed

Per the owner's instruction, with the state each is closed *in*:

* **§4.2 — the uBooNE calibration chain.** Set aside by the owner in round 6 and
  closed here. The absence stays documented as **accepted**, not as a pending
  gap: these are uBooNE YZ / SCE / lifetime maps and their absence may well be
  right for SBND. The one consequence worth keeping visible is the second
  bullet of §4.2 — the stopping-particle PID templates came from uBooNE, so if
  they presume a corrected dQ/dx, SBND carries a systematic through PID. Closed
  as a *port* question; live as a *calibration* question if PID is ever retuned.
* **§4.3 — the other dQ/dx divergences.** Closed. Its last row (the
  `assemble_fitted_charge_2d` nondeterminism) is **FIXED** in §14.4. The rest
  were already triaged benign, a toolkit *fix* of a prototype bug
  (`connected_vec` at `:6708`), or multi-APA necessities with no prototype
  counterpart.

### 14.11 Scope and what is NOT claimed

* **Six events, one detector.** The class-2 census is SBND nueCC only, one run
  each. It shows no collapse *there*; the mechanism remains real (pinned by the
  fifth case of `doctest_pr_graph_order.cxx`) and can bite on another topology.
* **Nothing was converted.** No class-1 or class-2 container changed in this
  round — deliberately. The 18 class-1 traversals and the `SIZE_MAX` hazard are
  recorded, with a verdict each, for the owner to schedule.
* **The class-1 census was wrong once and is corrected in place.** The first
  pass matched only literal `T*` keys and reported 66 declarations / 3
  traversals; it silently excluded every `std::set<SegmentPtr>` /
  `std::map<VertexPtr,…>`, which are address-ordered just the same. The numbers
  in §14.6 are the corrected 116 / 18. Any earlier quotation of 66/3 is void.
* **`NeutrinoVertexFinder.cxx:2934` is fixed but its effect is unmeasured**
  (§14.12): the enclosing branch never executed on these six events, so the
  change is provably inert here and the fix rests on the code, not on a number.
* **`GroupingHelper.cxx:27` is still open** — a real ordering dependence, left
  alone because it is a different subsystem (CLAUDE.md: an unrelated defect does
  not ride along).
* **The fix is not bit-identical to `23bd6783`** and cannot be: it *defines* a
  value that was previously arbitrary. Diagnostic-only (§14.2), which is why it
  ships without a knob, on the same footing as rounds 1–7.
* **`showers[]/kine_dQdx` / `total_length` still move run-to-run** (§14.5).
  Every determinism claim in this doc, including this one, is bounded by that.
* **Still owed, unchanged:** the valfast/1000 population gate with a regenerated
  baseline. Rounds 6 and 7 both moved it; this round does not (diagnostic-only),
  but it does not discharge it either.

### 14.12 `NeutrinoVertexFinder.cxx:2934` FIXED — the pointer-ordered mutating traversal

Owner, after reading §14.6: *"I think we need to fix [it], right?"* Yes.

**The defect.** `used_segments` (`:2839`) is a bare `std::set<SegmentPtr>`, i.e.
**address-ordered**, and the `:2934` loop body mutates:

```cpp
sg1->particle_info()->set_pdg(13);  sg1->particle_info()->set_mass(muon_mass);
sg1->unset_flags(SegmentFlags::kShowerTopology);
change_daughter_type(graph, vtx,       sg1, 13, muon_mass, …);   // :2942
change_daughter_type(graph, other_vtx, sg1, 13, muon_mass, …);   // :2945
```

`change_daughter_type` propagates the type change through the graph, and a later
iteration reads `current_pdg` at `:2953` that an earlier one may have written.
So the outcome depended on heap layout. Same class as round 3's `:2333`/`:2374`;
it survived rounds 3–5 because those swept `boost::out_edges` / `boost::edges` /
`boost::vertices` / `graph_nodes`, never `std::set<SegmentPtr>`.

**The fix, and why it is not the obvious one.** The obvious fix — retype the
container as `PR::IndexedSegmentSet` — would also change `find()` at `:2905`
from **pointer identity to index identity**, which is precisely the swap §11.7
warns about. So the set stays pointer-keyed and only the *iteration* is ordered:

```cpp
std::vector<SegmentPtr> ordered_used_segments(used_segments.begin(), used_segments.end());
std::sort(ordered_used_segments.begin(), ordered_used_segments.end(),
          [](const SegmentPtr& a, const SegmentPtr& b) {
              return a->get_graph_index() < b->get_graph_index();
          });
for (auto sg1 : ordered_used_segments) { … }
```

`find()` semantics are untouched; the sort key is unique here because every
member came from `vertex_segments`, i.e. live graph edges (§14.7's census). A
TRACE line reports the set size at every entry, so the >1 case — the only case
where order can matter — is countable from a log rather than assumed.

**Liveness — the honest result: the block never fires on this manifest.**

```bash
wcbuild && ./build/clus/wcdoctest-clus
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
SBND_WCT_LOGLEVEL=trace PR_EXTRA_STAGES=pr_display PR_JOBS=6 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-p15-live data \
  388 239794 172230 271851 54095 163543
grep -c "back-to-back retype over" work-p15-live/pr_evt*/wct_pr_evt*.log
```

**0 occurrences on all six events** — the enclosing branch needs back-to-back
muon or proton legs at one vertex (`angle > 165°`/`170°`, pdg 13 or 2212, length
> 30/20 cm, `:2880-2893`) *and* `flag_skip` to survive, which no event here
produces. So the change is **provably inert on this manifest**, and the A/B
confirms it:

| comparison | leaf diffs | where |
|---|---|---|
| evt 388, `work-p14-fix` → `work-p15-live` | **2** | 1 `showers[]/kine_dQdx`, 1 `showers[]/total_length` |
| evts 239794 / 172230 / 271851 / 54095 | **2** each | same two families |
| evt 163543 | **1** | `showers[]/total_length` |
| `work-p15-idx` vs `work-p15-live` (two builds differing **only** in a TRACE guard) | **3** | same two families |
| `work-p15-live` vs `-live-rep` (**genuinely the same binary**, `22249ff4`) | **1** | 1 `showers[]/kine_dQdx` |

Every diff is inside §14.5's 2–3-leaf residual floor. **`nusel-evt<ID>.tsv` is
byte-identical on all six events**, and on the same-binary repeat too (the
arm-level `nusel-table.tsv` differs only because the arms hold 1 vs 6 events).

Two labelling points, because this doc's credibility rests on them (M1):
`work-p15-idx` and `work-p15-live` are **not** the same binary — the TRACE guard
was widened from `size() > 1` to unconditional between them, so the second build
could report "the block never fires" rather than only "never fires with >1".
The TRACE has no side effects and sits in a branch that never executes, but a
"same binary" label that is not one is exactly the sort of thing a later reader
builds on. `work-p15-live` vs `-live-rep` is the real same-binary repeat of
`22249ff4`, and its floor is **1 leaf**.

**Stable ≠ prototype-agreement**, as in §9.4, §10.6 and §11.8: graph-index order
is *an* order. The prototype's counterpart loop walks its own pointer-keyed
`std::map<ProtoSegment*, …>`, so its order is equally arbitrary — this buys
reproducibility, not parity, and no prototype-agreement claim is made.

**What is and is not claimed.** The defect was real and is closed. Its *effect*
is unmeasured, because the path never executed here — this is a correctness fix
bought on the code, not on a number. **It also ships without a test**, and so
does §14.1's ident-ordered merge: round 5 added `doctest_pr_graph_order.cxx` for
its sort key, round 8 adds nothing. A revert-proven doctest on the merge order
would need `TrackFitting` scaffolding that does not exist, and the `:2934` path
is unexercised by any event in this manifest — but upstream `CLAUDE.md`'s "new
code ships with tests" applies, and the gap is the owner's to accept, not
mine to leave silent. A `numu`-rich sample with back-to-back
tracks at the neutrino vertex is where it would show; that is the population the
owed valfast/1000 gate covers anyway.

# doc pr/30 — Proto-vertex and track-segment finding: prototype-fidelity audit

**Why.** The owner asked for a pr/28-style prototype↔toolkit audit of the
**proto-vertex and track-segment finding** stage — the code that turns one
cluster's Steiner point cloud into the vertex/segment graph everything
downstream (direction, PID, main-vertex selection, energy) reads. This is the
third audit in the series: pr/28 covered vertex fitting and trajectory/dQ-dx,
pr/29 covered Steiner graph building, and this one covers the stage that
consumes pr/29's output and produces pr/28's input.

**Status.** **Audit only. No code is changed and none is proposed.** Every
divergence below is reported with both readings, per CLAUDE.md §5 rule 4 — the
porting dictionary has no proto-vertex section either (see §7.5). P1, P2, P3 and
P4 would each alter production output unconditionally, which is §5 rule 1:
the owner's call, not mine.

> **STATUS 2026-08-04: implemented and closed.** §12 has the five knobs, the
> byte-identical gate and the 48-event measurements; **§12.10 records the owner
> decision** — F2 flipped to a **SBND production default ON** (gated 48/48
> byte-identical), P8 and P4 **closed with no fix owed**, P2 left at its
> production default, and **P1 the only item still open**, blocked on the §3.1
> unit question. Read §12.10 first, then §12.8.
>
> **→ For how the list got to four, read §10.** The owner applied a filter on
> 2026-08-04 — *skip what is an improvement over the prototype, keep only bugs
> or gaps in the port* — plus the clarification that where the toolkit lacks the
> prototype's id/index information and substitutes geometry, that is accepted.
> Four of the fourteen survive: **F1 = P1** (`flag_exclusion` never passed
> `true`), **F2 = P9** (the out-of-volume skip, now three sites with three
> different undeclared biases), **F3 = P8** (the endpoint check dropped, not
> translated), **F4** (the two `find_vertices(...).first` callers, a narrowed
> P5). **P14 is resolved** at `f1e29f19`. §10.8 corrects three counts in §3/§8.

**Headline.** The **control flow is a faithful port** — `find_proto_vertex`'s
nine phases run in the prototype's order, and every cut constant I could pair
across `init_first_segment`, `break_segments`, `examine_segment`,
`examine_vertices_{1,2,3,4}`, `examine_partial_identical_segments`,
`crawl_segment`, `check_end_point`, `find_vertex_other_segment`,
`modify_{segment,vertex}_isochronous` and `find_other_segments` matches, as do
every default argument. The fourteen divergences are **not** in the
skeleton; they are (a) one dropped flag that changes how *every* fit in this
stage associates 2-D charge (**P1**), (b) four unconditional toolkit-only
behaviours added on top of the prototype (**P2**, **P3**, **P4**, **P6**), and
(c) a changed ordering convention in a helper that every caller uses (**P5**).
The determinism answer inverts pr/28's: the **prototype** is the clean one here
(its maps are id-ordered by construction), and the toolkit at HEAD still has 13
raw pointer-ordered out-edge loops — none of them reachable, and a concurrent
session is converting them as this was written (**P14**, §6).

**Which toolkit was read.** Every toolkit line number below is against **commit
`ea1a7e3d`**, not the working tree. This matters here: while this audit was
being written a **concurrent session had uncommitted edits to
`NeutrinoPatternBase.cxx`, `NeutrinoStructureExaminer.cxx`,
`NeutrinoOtherSegments.cxx` and `TrackFitting.cxx`** — a `boost::out_edges` →
`sorted_out_edges` determinism sweep (see §6 / **P14**). Line numbers drifted by
5-7 lines mid-audit; all anchors were re-derived from `git show HEAD:` at the
end. Prototype at `prototype_base/` → `/nfs/data/1/xqian/prototype-dev/wire-cell`.

---

## Repro

No event was run for this document (see §9). Every number below comes from
reading the two trees; these are the commands that regenerate the counts.

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit

# P1 -- the flag_exclusion split.
grep -rhn "do_multi_tracking(" prototype_base/pid/src/NeutrinoID*.h \
  | grep -v "^\s*[0-9]*:\s*//\|/\* " | grep -c "true, true, true)"    # -> 28
grep -rhn "do_multi_tracking(" prototype_base/pid/src/NeutrinoID*.h \
  | grep -v "^\s*[0-9]*:\s*//\|/\* " | grep -c "true, true, false)"   # -> 2
grep -rn  "do_multi_tracking(" clus/src/*.cxx \
  | grep -v "void TrackFitting::" | wc -l                             # -> 32
grep -rn  "do_multi_tracking(" clus/src/*.cxx \
  | grep -v "void TrackFitting::" | grep -c ", true, [a-z]*, true"    # -> 0

# section 6 -- the determinism sweep.  Read HEAD, NOT the working tree: a
# concurrent session is mid-sweep converting these to sorted_out_edges (P14).
for f in clus/src/NeutrinoPatternBase.cxx clus/src/NeutrinoStructureExaminer.cxx \
         clus/src/NeutrinoOtherSegments.cxx; do
  echo -n "HEAD $f raw="; git show HEAD:$f | grep -c "boost::out_edges"
  echo -n "     worktree raw="; grep -c "boost::out_edges" $f
done                                          # HEAD -> 3 9 1 ; worktree -> 0 0 0

# section 5 -- prototype code that is commented out or never called.
grep -rn "check_end_segments\|clean_up_maps_vertices_segments\|residual_segment_candidates" \
     prototype_base/pid/src/ prototype_base/pid/inc/

# provenance of the three toolkit-only behaviours.
git log -1 --format="%h %an %ad %s" --date=short 1eb097a9   # P2
git log --oneline -3 -L 1458,1476:clus/src/NeutrinoPatternBase.cxx   # P3
```

---

## §0 Scope

**In scope** — the prototype's `find_proto_vertex` and everything it calls that
belongs to this stage:

| prototype | lines | toolkit | lines |
|---|---|---|---|
| `NeutrinoID::find_proto_vertex` | `NeutrinoID_proto_vertex.h:69-197` | `PatternAlgorithms::find_proto_vertex` | `NeutrinoPatternBase.cxx:1945-2106` |
| `init_first_segment` | `:419-572` | `init_first_segment` | `NeutrinoPatternBase.cxx:523-823` |
| `init_point_segment` | `:398-417` | `init_point_segment` | `NeutrinoPatternBase.cxx:2109-2148` |
| `break_segments` | `:577-792` | `break_segments` | `NeutrinoPatternBase.cxx:1192-1544` |
| `find_other_segments` | `:797-1519` | `find_other_segments` | `NeutrinoOtherSegments.cxx:31-834` |
| `check_end_point` | `:1776-1932` | `check_end_point` | `NeutrinoOtherSegments.cxx:835-947` |
| `find_vertex_other_segment` | `:1520-1596` | `find_vertex_other_segment` | `NeutrinoOtherSegments.cxx:948-1051` |
| `modify_segment_isochronous` | `:1597-1693` | `modify_segment_isochronous` | `NeutrinoOtherSegments.cxx:1178-1309` |
| `modify_vertex_isochronous` | `:1696-1775` | `modify_vertex_isochronous` | `NeutrinoOtherSegments.cxx:1052-1177` |
| `examine_segment` | `:2044-2157` | `examine_segment` | `NeutrinoStructureExaminer.cxx:963-1156` |
| `crawl_segment` | `:2158-2377` | `crawl_segment` | `NeutrinoStructureExaminer.cxx:702-962` |
| `examine_vertices` | `:2378-2406` | `examine_vertices` | `NeutrinoStructureExaminer.cxx:2079-2107` |
| `examine_vertices_1` (cluster / pair) | `:2877-3006` / `:3007-3226` | `examine_vertices_1` / `_1p` | `NeutrinoStructureExaminer.cxx:1388-1493` / `:1162-1387` |
| `examine_vertices_2` | `:2543-2613` | `examine_vertices_2` | `NeutrinoStructureExaminer.cxx:1494-1617` |
| `examine_vertices_3` | `:2413-2542` | `examine_vertices_3` | `NeutrinoStructureExaminer.cxx:2432-2627` |
| `examine_vertices_4` (cluster / pair) | `:2614-2825` / `:2826-2876` | `examine_vertices_4` / `_4p` | `NeutrinoStructureExaminer.cxx:1705-2078` / `:1618-1704` |
| `examine_partial_identical_segments` | `:200-367` | `examine_partial_identical_segments` | `NeutrinoStructureExaminer.cxx:2109-2378` |
| `add/del_proto_connection`, `del_proto_vertex/segment` | `:1933-2042` | `add_segment`, `remove_segment`, `remove_vertex` | `PRGraph.cxx` |
| `find_vertices/find_segments/find_other_vertex/find_segment` | `:3227-3285` | same names | `PRGraph.cxx:105-170`, `NeutrinoPatternBase.cxx:25-54` |

**Out of scope, and explicitly NOT claimed clean:**

* **`examine_structure*`** — `find_proto_vertex` calls `examine_structure` and
  `examine_structure_3`, but `NeutrinoStructureExaminer.cxx` is 3,517 lines and
  is its own audit. Not read beyond the two call sites.
* **`do_multi_tracking` / `do_single_tracking` internals** — see doc pr/28.
  Read here only far enough to establish P1's mechanism (§3.1).
* **Main-vertex selection** (`compare_main_vertices*`, `examine_main_vertex*`,
  `search_for_vertex_activities`, the bulk of `NeutrinoVertexFinder.cxx`) — a
  later stage that consumes this one's output.
* **`segment_search_kink`, `proto_extend_point`, `proto_break_tracks`,
  `do_rough_path`** — the geometric primitives `break_segments` drives. Their
  call signatures are compared (§3.3, §7.1); their bodies are not.
* **The Steiner inputs** — doc pr/29, whose eleven divergences all land
  upstream of the first line of `init_first_segment`.

---

## §1 How far to trust each finding

Two tiers, stated so a reader deciding what to act on knows which is which.

**Tier A — read in both trees, line by line, both directions.**
`find_proto_vertex`, `init_first_segment`, `init_point_segment`,
`break_segments`, `modify_segment_isochronous`, `examine_vertices`, the
connection bookkeeping, `find_vertices`, and the `do_multi_tracking` flag
plumbing down to `form_map_graph`. P1, P2, P3, P5, P6, P7, P8, P10, P11 and all
of §5 rest on Tier A reading.  P14's thirteen sites were each opened and read
at HEAD, but see the caveat in §9.

**Tier B — orchestration, call signatures and cut constants compared; branch
bodies sampled, not exhaustively read.** `find_other_segments` (723 prototype
lines against ~800 toolkit lines), `crawl_segment`, `examine_segment`,
`examine_vertices_{1,2,3,4}`, `examine_partial_identical_segments`,
`check_end_point`, `find_vertex_other_segment`, `modify_vertex_isochronous`.
P4, P9 and P12 are Tier B: the specific lines quoted were read, the surrounding
branch was not fully traced. **A Tier-B "matches" means the constants and the
signature match, not that every branch was compared.**

---

## §2 What matches — the skeleton is a faithful port

Establishing this first is what makes the divergence list readable: these are
wrong or added *inputs* to correct machinery, not a broken port.

**§2.1 The nine phases of `find_proto_vertex`, in order.** Prototype `:69-197`
against toolkit `:1945-2106`:

| # | phase | prototype | toolkit |
|---|---|---|---|
| 0 | bail if no Steiner cloud or < 2 points | `:71-72` | `:1952-1955` |
| 1 | `init_first_segment` | `:81` | `:1959` |
| 2 | record the main cluster's initial vertex pair | `:89` | `:1969-1971` |
| 3 | bail if the first segment has ≤ 1 wcpt | `:91` | `:1975-1977` |
| 4 | `break_segments` + `examine_structure`, **or** a bare refit | `:93-111` | `:1984-2044` |
| 5 | `find_other_segments` × `nrounds` | `:128-132` | `:2048-2052` |
| 6 | main cluster only: `examine_structure_3`, refit if it changed anything | `:136-143` | `:2057-2063` |
| 7 | `examine_vertices` | `:150` | `:2067` |
| 8 | `examine_partial_identical_segments` | `:157` | `:2072` |
| 9 | main cluster only: `examine_vertices_3`, then the final refit | `:164-170` | `:2076-2084` |

**§2.2 Every default argument is identical** — 16 parameters across six functions. Prototype
`NeutrinoID.h:1620-1702` against toolkit `NeutrinoPatternBase.h:336-409`:
`find_proto_vertex(flag_break_track=true, nrounds=2, flag_back_search=true)`;
`init_first_segment(flag_back_search=true)`; `break_segments(dis_cut=0)`;
`find_other_segments(flag_break_track=true, search_range=1.5 cm,
scaling_2d=0.8)`; `check_end_point(flag_front=true, vtx_cut1=0.9 cm,
vtx_cut2=2.0 cm, sg_cut1=2.0 cm, sg_cut2=1.2 cm)`;
`modify_segment_isochronous(dis_cut=6 cm, angle_cut=15, extend_cut=15 cm)`.
This is the check that caught pr/29's D2, so it is worth stating that it came
back clean here — the dropped flag (P1) is at a *call site*, not a default.

**§2.3 The cut constants match.** Histogramming every numeric literal and
comparison threshold per function pair (the `dump()` helper in §Repro) leaves no
unexplained difference in `examine_segment` (105, 150, 2 cm, 4),
`examine_vertices_1p` (2.5, 30, 8, 35, 0.6 cm, 0.2 cm, 9.0),
`examine_vertices_2` (0.45, 1.5), `examine_vertices_4` (0.2/0.3/2.0/2.5/3.0 cm,
2, 3.5, 10), `examine_partial_identical_segments` (0.3, 2, 5),
`crawl_segment` (0.2/0.3/3.0 cm, 1, 2), `check_end_point` (2, 5, 90),
`find_vertex_other_segment` (0.9, 1.2/1.5/2.5/3.0 cm),
`modify_vertex_isochronous` (0.2/0.6/15 cm), and `find_other_segments`
(0.25/0.4/0.6/0.78, 10/15/18/30 cm, 1.1/1.6, 3.5, 6, 7, 8, 13, 36). The four
apparent extras all resolve — three to commented-out prototype lines (§5.6,
§5.7) and one to a real toolkit addition (**P4**).

**§2.4 The graph-mutation contract is equivalent.** The prototype maintains six
hand-rolled maps (`map_vertex_segments`, `map_segment_vertices`,
`map_vertex_cluster`, `map_cluster_vertices`, `map_segment_cluster`,
`map_cluster_segments`) through `add_proto_connection` / `del_proto_connection`
/ `del_proto_vertex` / `del_proto_segment` (`:1933-2042`). The toolkit replaces
all six with one BGL graph in which a segment *is* an edge and a vertex *is* a
node, plus `seg->cluster()` / `vtx->cluster()` back-pointers; `add_segment`,
`remove_segment` and `remove_vertex` (`PRGraph.cxx`) maintain the same
invariants. Deleting a vertex detaches it from its segments in both; deleting a
segment detaches it from both endpoints in both. The one lost check is **P8**.

**§2.5 Both trees iterate deterministically** — see §6. This is a *different*
answer from pr/29's `setS` finding and worth stating plainly: the prototype's
`Map_Proto_Vertex_Segments` is keyed with an explicit `ProtoVertexCompare` that
orders by `get_id()`, not by pointer.

---

## §3 The divergences

Ranked by how much of the delivered graph each one can move. Severity is my
judgement of reach, not a measurement.

### §3.1 P1 — `flag_exclusion` is dropped at every toolkit `do_multi_tracking` call site

| | |
|---|---|
| prototype | `NeutrinoID_proto_vertex.h:108,140,170,319,355,415,1363,1479,2358,2471,2537,2604,2724,2813,2995` (+13 more in `improve_vertex` / `final_structure` / `examine_structure` / `track_shower`) — `..., true, true, **true**)` |
| toolkit | `NeutrinoPatternBase.cxx:1491,1507,1692,2042,2060,2084,2147`; `NeutrinoStructureExaminer.cxx` ×15; `NeutrinoOtherSegments.cxx:550,670`; `NeutrinoVertexFinder.cxx` ×6 — `do_multi_tracking(true, true, ?, **false**, false, cluster)` |
| severity | **highest — every fit in the stage** |

The counts: **28 of 30 live prototype call sites pass `flag_exclusion = true`**.
The two that pass `false` are `:722` and `:751` — both inside `break_segments`,
where the prototype's own comment says *"fit dQ/dx here, do not exclude
others"*. **All 32 toolkit call sites pass `false`.** So the toolkit matches the
prototype exactly where the prototype turns exclusion *off*, and diverges
everywhere the prototype turns it *on*.

The mechanism is confirmed on both sides and is the same code:

```cpp
// prototype PR3DCluster_multi_track_fitting.h:820
if (flag_exclusion)
  update_association(temp_2dut, temp_2dvt, temp_2dwt, sg, segments);

// toolkit TrackFitting.cxx:3187
if (flag_exclusion) {
    update_association(segment, segments, temp_2dut, temp_2dvt, temp_2dwt);
}
```

Both sit inside `form_map_graph`, between `form_point_association` and
`examine_point_association`, and both strip from this segment's 2-D
associations the (wire, tick) cells that belong to *other* segments. With
exclusion off, two segments crossing the same wire region both claim the same
charge; the multi-track fit then distributes it differently, which moves fitted
point positions, `dQ`, `dx` and `reduced_chi2` — i.e. the inputs to almost
every cut in §2.3.

**The callee is a faithful port too**, which is what makes this a dropped
argument rather than an abandoned feature. `TrackFitting::update_association`
(`TrackFitting.cxx:2535-2680`) reproduces
`PR3DCluster::update_association` (`PR3DCluster_multi_track_fitting.h:970-1075`)
plane for plane: the same per-cell loop, the same "distance to *this* segment
vs. the minimum distance to every *other* segment" comparison, and the same
keep rule to the constant —
`min_dis_track < min_dis1_track || min_dis_track < 0.3*units::cm`. The machinery
is complete, correct and never invoked.

**One thing I could not resolve, recorded rather than ruled on.** The two
implementations reconstruct the transverse coordinate differently — prototype
`y = (wire - offset_u) * pitch_u` (`:1013`), toolkit
`raw_y = (coord.wire - offset_u) / slope_yu` with
`slope_yu = -sin(angle_u)/pitch_u` (`TrackFitting.cxx:2568`) — and the two feed
different closest-2-D-distance helpers, each of which projects again. Whether
the compositions agree is a `segment_get_closest_2d_distances` question, i.e.
doc pr/28's territory, and it is **unobservable today precisely because this
path is dead.** If P1 is ever resolved by turning exclusion on, this is the
first thing to check — a dead code path is exactly where a unit error survives.

**Both readings.** (i) The toolkit is missing an argument and should pass
`true` at the 28 sites where the prototype does. (ii) The toolkit deliberately
standardised on the non-excluding fit — but nothing in the code says so: there
is no comment at any of the 32 sites, and the parameter is threaded all the way
down and honoured, so it is not vestigial.

**Reach is not measured.** How many SBND points actually have a cross-segment
association to strip is exactly the question one instrumented run answers, and
this document does not answer it.

### §3.2 P2 — a local-PCA endpoint refinement with no prototype counterpart

| | |
|---|---|
| prototype | none — `init_first_segment:426` takes `get_two_boundary_wcps(2)` and uses it |
| toolkit | `NeutrinoPatternBase.cxx:567-676`, commit `1eb097a9` (Xin Qian, 2026-03-31, *"fix a bug and randomness"*) |
| severity | **high — moves the seed of the whole stage** |

After taking the same two Steiner boundary points as the prototype, the toolkit
gathers every Steiner point within `r_local = 10 cm` of each endpoint, filters
to terminal nodes (falling back to all points when fewer than 3 terminals),
computes their local principal axis by 10 power iterations seeded with the
global first→second direction, and replaces the endpoint with whichever
neighbour projects furthest along that axis. The stated motivation is a curving
cluster whose true tip lies "around a corner" from the global boundary search.

This is unconditional and has no knob. Everything downstream of
`init_first_segment` is derived from these two points — the Dijkstra path, the
kink search, every break point, and hence every vertex.

**Both readings.** (i) It is a genuine improvement the prototype lacks, and the
prototype is the thing that is wrong. (ii) It is an undocumented, un-knobbed
departure from the reference implementation in the single most load-bearing
function of the stage. Either way it belongs in the porting dictionary, which
does not mention it.

**Interaction worth knowing:** when `m_iso_endpoint` fires
(`NeutrinoPatternBase.cxx:538`), the entire block — boundary search *and* this
refinement — is bypassed in favour of `find_iso_first_segment_endpoints`.
`iso_endpoint` is the **SBND production default** (doc pr/24 round 3, flip
`12d65f1d`), so on isochronous-sheet clusters SBND runs neither the prototype's
path nor P2's.

### §3.3 P3 — a second, wider terminus-stub absorption branch in `break_segments`

| | |
|---|---|
| prototype | `NeutrinoID_proto_vertex.h:705-725` — `if (min_dis/cm < 1.5 && angle > 120)` then, *only* if `end_v` has degree 1, absorb; else do nothing |
| toolkit | `NeutrinoPatternBase.cxx:1472-1495`, commits `a0572c3b` → `7725a9c7` → `6127bc31` |
| severity | **high — changes the segment count** |

```cpp
// toolkit :1472-1475
bool use_replace = (end_is_terminus && min_dis / units::cm < 2.0
                                    && kink_angle_at_break > 30.0
                                    && angle < 45.0) ||
                   (!end_is_terminus && min_dis / units::cm < 1.5 && angle > 120);
```

The second clause is the prototype's, with one change: the prototype applies the
`1.5 cm / 120°` test *first* and then checks the degree, so on a degree-1 end
vertex it absorbs; the toolkit routes degree-1 vertices to the new first clause
instead and leaves the old clause for degree ≠ 1 (where the prototype did
nothing — the toolkit preserves that, see the comment at `:1495`).

The first clause is new: a degree-1 endpoint with a tail shorter than 2 cm, a
real (> 30°) kink angle in the Steiner path at the break, and a stub roughly
aligned (< 45°) with the parent segment gets absorbed instead of becoming its
own segment. The 45° guard was added later (`6127bc31`) explicitly to stop the
absorption of genuine short secondaries — the comment cites a 12.94 MeV Michel
at ~79° as the case it must not eat.

**Both readings.** (i) It fixes a real fold-back artefact at track endpoints
that the prototype materialises as a spurious stub segment. (ii) It changes
how many segments a track ends up with, unconditionally, on a branch the
prototype reaches with different geometry. The `kink_angle_at_break` quantity
has no prototype counterpart at all.

### §3.4 P4 — an extra acceptance clause in `find_other_segments`

| | |
|---|---|
| prototype | `NeutrinoID_proto_vertex.h:1368` — `length > 30 cm \|\| (direct_length < 0.78·length && length > 10 cm && medium_dQ_dx/MIP > 1.6)` |
| toolkit | `NeutrinoOtherSegments.cxx:558-560` — the same, **plus** `\|\| (direct_length < 0.72·length && length > 15 cm && medium_dQ_dx/MIP > 1.05)` |
| severity | medium — admits more segments |

The new clause accepts a *less* curved (0.72 vs 0.78 direct/total ratio),
*longer* (15 cm vs 10 cm), *less* ionising (1.05 vs 1.6 × MIP) candidate than
the prototype's. It is a strict widening: nothing the prototype accepts is
rejected, and some things it rejects are now accepted, so this can only add
segments. Tier B — I read the clause and its immediate context, not every path
that reaches it.

The *second* acceptance test in the same function (toolkit `:676-681`,
prototype `:1486-1493`) is a faithful port including the operator-precedence
grouping — see §5.9.

### §3.5 P5 — `find_vertices(segment)` returns its pair in a different order

| | |
|---|---|
| prototype | `NeutrinoID_proto_vertex.h:3227-3243` — `.first` / `.second` are `map_segment_vertices[sg]`'s first two elements, i.e. **ascending vertex id** |
| toolkit | `PRGraph.cxx:105-141` — `.first` is whichever endpoint's `wcpt()` is **closer to the segment's first wcpt** |
| severity | medium — every caller, but only the order-sensitive ones |

The toolkit's ordering is documented in the header (`PRGraph.h:117-118`) and is
arguably the more useful contract: `.first` is the vertex at the segment's
start. But it is not the prototype's contract, and any call site ported
literally from a prototype that assumed id order now sees a different vertex in
`.first`.

`break_segments` is unaffected because both trees re-derive start/end
explicitly (§3.11), and `modify_segment_isochronous` treats the pair
symmetrically. Whether any other caller is order-sensitive is **not measured
here** — that is a sweep of every `find_vertices(graph, seg)` call in `clus/`,
which this audit did not do.

### §3.6 P6 — a `merge_nearby_vertices` post-pass at the end of `break_segments`

| | |
|---|---|
| prototype | none |
| toolkit | `NeutrinoPatternBase.cxx:1519-1530` → `merge_nearby_vertices` at `:1547` |
| severity | medium |

After the break loop, the toolkit merges any two vertices of the entry cluster
within **0.1 cm** and refits. The comment attributes it to an oscillating-break
pattern that leaves duplicate vertex objects at identical positions.

The prototype has a function that does something similar —
`clean_up_maps_vertices_segments` (`:3287-3314`), which merges vertices whose
`wcpt().index` coincide — but it is **never called**: both of its call sites,
including the one right inside `break_segments` at `:718`, are commented out
(§5.3). So the toolkit re-enabled, in a different and looser form (0.1 cm
proximity rather than exact index equality), a cleanup the prototype
deliberately switched off.

### §3.7 P7 — the walk-halfway override in `break_segments`

| | |
|---|---|
| prototype | none |
| toolkit | `NeutrinoPatternBase.cxx:1299-1321` |
| severity | medium |

If the current `proto_extend_point` walk's `dir1` is anti-parallel
(`dot < -0.5`) to the previous break iteration's `dir1`, the toolkit discards
the walker's endpoint and substitutes `walk_hist[walk_hist.size()/2]` — the
midpoint of the walk history — then re-snaps to the nearest Steiner point.
`proto_extend_point` gained a `walk_history` out-parameter for this; the
prototype's has no such parameter.

Unlike the two loop guards in §3.13 this is not a "only fires on pathological
input" guard: it changes the break point whenever the direction test trips.

### §3.8 P8 — the vertex/segment endpoint check is not ported

| | |
|---|---|
| prototype | `NeutrinoID_proto_vertex.h:1953-1956` — refuses the connection and prints `"Error! Vertex and Segment does not match"` unless the vertex's `wcpt().index` equals the segment's front or back |
| toolkit | `PRGraph.cxx: add_segment` — links any two vertices with no such test |
| severity | low-medium — a lost invariant, not a changed algorithm |

Because most prototype callers ignore `add_proto_connection`'s return value,
the practical prototype behaviour on a mismatch is *the connection is silently
not made*; the toolkit's is *the connection is made*. Whether any live path
produces a mismatch is unknown — the prototype's diagnostic would have printed
if one did, which is itself weak evidence that none does.

### §3.9 P9 — points outside every detector volume are skipped, not evaluated

| | |
|---|---|
| prototype | `is_good_point(test_p, 0.2 cm, 0, 0)` / `get_closest_dead_chs(...)` are called for every step point |
| toolkit | `NeutrinoOtherSegments.cxx:1251-1257`, `NeutrinoStructureExaminer.cxx:1264-1278` — `if (wpid.apa() == -1 \|\| wpid.face() == -1) continue;` |
| severity | low, but note the sign flip |

The guard exists because `contained_by` returns `-1` outside the detector and
the downstream `m_trigger_offsets.at(-1)` throws (the class-C crash shape, doc
pr/11 §6.3) — so the guard is necessary. What is worth recording is that its
*bias* differs by site:

* in `modify_segment_isochronous` the skipped point does **not** increment
  `n_bad`, so an out-of-volume path is treated as **connected** → more
  isochronous modifications than the prototype;
* in `examine_vertices_1p` the skipped point does **not** clear `flag_dead`, so
  an out-of-volume segment is treated as **dead** → the opposite polarity.

Neither matches the prototype, which evaluates the point.

### §3.10 P10 — a toolkit-only "no segments survived" return

| | |
|---|---|
| prototype | returns `true` after the final refit unconditionally |
| toolkit | `NeutrinoPatternBase.cxx:2090-2101` — scans the graph and returns `false` if no segment of this cluster remains |
| severity | low — a recovery path |

The caller is `TaggerCheckNeutrino.cxx:617`, `if (!find_proto_vertex(..., false,
1, false)) init_point_segment(...)` — the same shape as prototype
`NeutrinoID.cxx:183`. So the toolkit can fall back to `init_point_segment` in a
case where the prototype would leave the cluster with no segments at all. The
comment names the trigger: a Type II merge that removed the only segment.

### §3.11 P11 — start/end vertex determination in `break_segments`

| | |
|---|---|
| prototype | `:595-610` — matches `wcpt().index` against the segment's front/back; if neither ordering matches, `start_v`/`end_v` stay **null**, an error prints, and `:615` dereferences the null pointer |
| toolkit | `:1231-1236` — takes `find_vertices`' pair and swaps if the second is closer to the segment front |
| severity | low |

Same answer in the normal case. The toolkit cannot reach the prototype's
null-dereference, and it also cannot detect the inconsistency the prototype was
testing for. Reported as a divergence rather than a fix because the prototype's
branch is reachable in principle and its behaviour there is a crash, which is
not something to reproduce.

### §3.12 P12 — `examine_vertices*` gained a `main_vertex` parameter

| | |
|---|---|
| prototype | `examine_vertices(cluster)`, `examine_vertices_{1,2,4}(cluster)` |
| toolkit | all four take a trailing `VertexPtr main_vertex = nullptr` (`NeutrinoPatternBase.h:397-401`) |
| severity | inert **at this call site** |

`find_proto_vertex` passes no `main_vertex`, so within this stage the parameter
is `nullptr` and the extension is inert. It is live at the *other* call sites
(the final-structure pass), which are out of scope. Listed so the next reader
does not mistake the signature change for a no-op everywhere.

### §3.13 P13 — two infinite-loop guards in the kink search

| | |
|---|---|
| prototype | none — `while(...)` with no iteration bound |
| toolkit | `NeutrinoPatternBase.cxx:1269-1272` (1000-pass cap) and `:1372-1385` (exact-stationarity detector) |
| severity | lowest — fires only where the prototype would hang |

Both break out of the kink loop rather than changing its result, and the
stationarity test is exact (same `test_start_p`, same `break_wcp`, same
`break_idx` two passes running). Recorded for completeness; the shape is the
class-A hang of doc pr/11 §6.1.

---

## §4 Knobbed divergences already on the record

These are real prototype/toolkit differences in this stage, but each is a
documented, config-visible knob rather than a silent drift. Listed so they are
not re-discovered as findings.

* **`m_mip_dqdx_median`** (`NeutrinoPatternBase.h:114`, default `43000/cm`) —
  the prototype hard-codes `43e3/units::cm` at each comparison
  (`:1368`, `:1483`). Doc pr/8.
* **`m_cathode_x` / `m_cathode_kink_xcut`** — extra arguments to
  `segment_search_kink` (`NeutrinoPatternBase.cxx:1275`) with no prototype
  counterpart. Doc pr/20 Part II §B0; **SBND default ON**.
* **`m_iso_endpoint`** and its six companions — the `init_first_segment`
  bypass described under P2. Doc pr/24; **SBND default ON**.
* **`m_perf`** — the prototype prints its per-phase timings unconditionally
  (`std::cout`); the toolkit gates them behind `m_perf` and `SPDLOG_TRACE`. Not
  physics.

---

## §5 Things that look like divergences and are not

Each of these cost real reading to resolve. Recording them so the next reader
does not pay again.

**§5.1 `count < 2` in `break_segments` is dead in BOTH trees.** Prototype
`:586-589` declares `int count = 0` and loops `while(remaining_segments.size()
!= 0 && count < 2)` — and never increments `count`. The toolkit reproduces this
exactly (`:1194`, `:1212`, no increment anywhere in the body). The loop is
`while(!remaining_segments.empty())` in both. This is the direct analogue of
pr/29's commented-out kruskal: a faithful port of a prototype bug, not a
divergence. **Do not "fix" one side without the other.**

**§5.2 `check_end_segments` is not a port gap.** The prototype defines it
(`:368-394`) and its only call site is commented out (`:100`, behind a dead
`flag_check_end_segments`). The toolkit has no such function. Faithful.

**§5.3 `clean_up_maps_vertices_segments` is not a port gap.** Defined at
`:3287`, both call sites commented out (`:718`, `:1361`). The toolkit's
`clean_up_graph` is a different thing (it deletes a cluster's whole subgraph).
See P6 for what the toolkit does instead.

**§5.4 `residual_segment_candidates` is write-only in the prototype.** Declared
`NeutrinoID.h:2022`, pushed at `NeutrinoID_proto_vertex.h:1473`, **read
nowhere**. The toolkit's omission is a faithful drop. Same shape as pr/29's
`point_cloud_steiner_terminal` GOTCHA.

**§5.5 The commented-out `seg->fits(...)` block in toolkit `init_first_segment`
(`:799-812`) is not a lost assignment.** `do_single_tracking` already does it:
`TrackFitting.cxx:405`, `segment->fits(segment_fits);`. The prototype's
explicit `sg1->set_fit_vec(...)` at `:525` is the same work done at the call
site instead.

**§5.6 The `30*units::cm` "missing" from `modify_segment_isochronous`** is
prototype lines `:1603-1604`, which are commented out. The live prototype uses
`extend_cut` exactly as the toolkit does.

**§5.7 The uBooNE-looking constants in `find_other_segments`** — `index_u ∈
(800, 820)`, `index_w ∈ (5505-4800, 5525-4800)`, `x ∈ (193, 200) cm`, and
`ncounts[i] >= 40` — are all inside commented-out prototype debug lines
(`:848`, `:1024`). Nothing detector-specific was dropped.

**§5.8 `flag_back_search = false` for non-main clusters is inert.** Prototype
`NeutrinoID.cxx:169,183` relies on the default `true`; toolkit
`TaggerCheckNeutrino.cxx:604,617` passes `false` explicitly. It makes no
difference: `flag_back_search` is consulted **only** inside the
`temp_cluster == main_cluster` branch of `init_first_segment` (prototype
`:428-444`, toolkit `:686-700`), and these calls are for non-main clusters.

**§5.9 The `||` / `&&` precedence in the second acceptance test is preserved.**
Prototype `:1486-1488` writes `A && B && C || D && E` with no parentheses,
relying on `&&` binding tighter. The toolkit writes `(A && B && C) || (D && E)`
— the same grouping, made explicit. Not a change.

**§5.10 `examine_vertices`' short-circuit is faithful.** Both trees write
`flag_continue = flag_continue || examine_vertices_N(...)`, so once `_1`
returns true, `_2` and `_4` are **not called** that round. It looks like a bug;
it is the prototype's behaviour, reproduced exactly.

**§5.11 The `2.5` two-dimensional threshold in `examine_vertices_1p` is in the
same units on both sides.** The prototype passes precomputed
`offset_t + slope_xt·x` and `offset_u + slope_yu·y + slope_zu·z` — time-slice
index and wire index. The toolkit calls `convert_3Dpoint_time_ch` (which
returns a *raw tick* index) and divides the time component by
`ntime_ticks = slice_index_max - slice_index_min` of the cluster's first blob
(`:1208-1210`) to recover slice units. The comment at `:1204-1207` states the
assumption — the first blob spans exactly one readout bin — and what breaks if
it does not (the comparison becomes loose by that factor). Worth knowing, not a
divergence.

**§5.12 `init_point_segment`'s different graph name is not a divergence.** The
prototype uses `get_two_boundary_wcps(1)` and `dijkstra_shortest_paths(wcp, 1)`
— flag 1 selects the *regular* point cloud, in contrast to flag 2 (Steiner)
used by `init_first_segment`. The toolkit's `get_two_boundary_wcps(false)` and
`do_rough_path_reg_pc(..., "relaxed_pid")` make the same choice by name.

---

## §6 Determinism — the prototype is clean; the toolkit at HEAD is not, and is being fixed right now

**Prototype: clean, and for a reason worth knowing.**
`Map_Proto_Vertex_Segments` and `Map_Proto_Segment_Vertices`
(`Map_Proto_Vertex_Segment.h`) are keyed with explicit comparators
(`ProtoVertex.h:82-94`, `ProtoSegment.h:192-204`) that order by `get_id()` and
fall back to the raw pointer **only when two ids are equal** — which cannot
happen, since `acc_vertex_id` / `acc_segment_id` increment monotonically. So
every `for (auto it = map_vertex_segments.begin(); ...)` in this file — and
there are dozens — is id-ordered, not heap-ordered. **This is the opposite of
the `setS` result in pr/28**: the prototype's proto-vertex bookkeeping was
written to be deterministic.

**Toolkit at `ea1a7e3d`: 13 raw pointer-ordered out-edge iterations remain in
the in-scope files** — `NeutrinoPatternBase.cxx` ×3 (`:1580`, `:1899`, `:2440`),
`NeutrinoStructureExaminer.cxx` ×9 (`:721`, `:1005`, `:1067`, `:1286`, `:1648`,
`:2135`, `:2452`, `:3075`, `:3262`), `NeutrinoOtherSegments.cxx` ×1 (`:1119`).
`boost::out_edges` on a `setS` edge list iterates in **pointer order**, which is
allocation-address order, which is not stable across runs. That is exactly why
`sorted_out_edges` exists (`PRTrajectoryView.h:154-165`) and is used at 32 other
sites in these files, alongside 31 uses of `ordered_nodes` / `ordered_edges`.

**I inspected all 13 and none is demonstrably order-sensitive**, which is why
this is P14 and not P1:

* `:2135` (`examine_partial_identical_segments`) and `:721` (`crawl_segment`)
  copy the edges into a vector and **sort by `graph[e].index` themselves** —
  the same fix, written out longhand.
* `:1286` (`examine_vertices_1p`) and `:2452` (`examine_vertices_3`) take the
  **first** out-edge and `break` — genuinely order-sensitive in shape, but both
  are guarded by an exact-degree test upstream (`degree != 2 → return false` at
  `:1176`; `degree != 1 → continue` at `:2447`), so the set they choose from
  has exactly one eligible member.
* The remaining nine accumulate over every out-edge independently (angles,
  removal lists, per-segment path rewrites) with no first-wins or best-wins
  selection, so the result does not depend on the order.

So the verdict is **"no reachable nondeterminism found, resting on invariants
that are not enforced where they are relied on."** A future edit that relaxes
either degree guard turns `:1286` or `:2452` into a live pointer-order
dependency with nothing to catch it.

**P14 — and it is already being fixed.** The working tree (uncommitted, another
session) converts all 13 of these to `sorted_out_edges`, along with 36 of the 38
raw sites in `NeutrinoVertexFinder.cxx`. So the answer to "is this stage
deterministic" is *yes at HEAD by accident, yes after that sweep by
construction*. I have not read that sweep beyond the four hunks quoted in §Repro
and make no claim about its correctness.

**One nuance, not a finding.** The two orderings are **keyed differently** —
prototype by object id, toolkit by graph insertion index. Each is reproducible
run to run, which is what the gates test; they are not guaranteed to be the
*same* order as each other.

**One place the toolkit already got this right explicitly:**
`break_segments:1209-1210` captures `entry_cluster` before the loop, because
picking a cluster off "the first edge" of the shared multi-cluster graph used to
select a different cluster run to run.

---

## §7 Loose ends found while reading, not acted on

**§7.1** `proto_extend_point`'s toolkit signature gained a
`std::vector<geo_point_t>* walk_history` out-parameter used only by P7.
Harmless, but it means the two `proto_extend_point` calls in `break_segments`
differ (`:1296` passes it, `:1344` does not) — the "seen this break before"
re-search path gets no walk-halfway protection.

**§7.2** `init_first_segment` compares the main-cluster **flag** against the
main-cluster **pointer** and warns when they disagree (`:682-685`), then uses
the pointer. `find_proto_vertex` (`:1967`) uses the flag alone. If they can
ever disagree, the two functions would make different choices — and the warning
implies someone thought they could.

**§7.3** The dead-region check in `examine_vertices_1p` scans **all** of the
segment's fit points and requires every one to be dead
(`flag_dead = false; break;` on the first live one). That matches the
prototype. Noted only because the loop is O(npoints) inside an O(vertices²)
scan.

**§7.4** `find_segment(v1, v2)` in the prototype (`:3256-3267`) leaves `sg`
holding the last-examined segment when the loop exits without a match — but the
final `if` sets it to 0 in that case, so it is correct. It is one edit away
from not being.

**§7.5** `clus/docs/porting/porting_dictionary.md` has **no proto-vertex /
segment-finding section**, exactly as it had no Steiner section (pr/29 GOTCHA
5). Every divergence above is therefore undocumented by construction.

---

## §8 Summary

| rank | id | what | prototype | toolkit | kind |
|---|---|---|---|---|---|
| 1 | **P1** | `flag_exclusion` dropped — 28 of 30 live prototype sites pass `true`, all 32 toolkit sites pass `false` | `NeutrinoID_*.h` ×28 | `clus/src/*.cxx` ×32 | dropped argument |
| 2 | **P2** | local-PCA endpoint refinement, unconditional, no knob | — | `NeutrinoPatternBase.cxx:567-676` | toolkit-only |
| 3 | **P3** | second terminus-stub absorption branch (`<2 cm`, kink `>30°`, `<45°`) | `:705-725` | `NeutrinoPatternBase.cxx:1472-1495` | toolkit-only |
| 4 | **P4** | extra acceptance clause `0.72 / 15 cm / 1.05` | `:1368` | `NeutrinoOtherSegments.cxx:560` | toolkit-only |
| 5 | **P5** | `find_vertices(seg)` ordered by proximity, not by vertex id | `:3227-3243` | `PRGraph.cxx:105-141` | changed contract |
| 6 | **P6** | `merge_nearby_vertices` (0.1 cm) post-pass after breaking | — (disabled `:718`) | `NeutrinoPatternBase.cxx:1519-1530` | toolkit-only |
| 7 | **P7** | walk-halfway override of the break point | — | `NeutrinoPatternBase.cxx:1299-1321` | toolkit-only |
| 8 | **P8** | endpoint-index validation on connection lost | `:1953-1956` | `PRGraph.cxx add_segment` | lost check |
| 9 | **P9** | out-of-volume points skipped, with opposite bias at two sites | — | `NOS:1251`, `NSE:1272` | added guard |
| 10 | **P10** | "no segments survived" → `false` → caller falls back | — | `NeutrinoPatternBase.cxx:2090-2101` | toolkit-only |
| 11 | **P11** | start/end by distance instead of wcpt-index equality | `:595-610` | `NeutrinoPatternBase.cxx:1231-1236` | changed method |
| 12 | **P12** | `main_vertex` parameter added to `examine_vertices*` | — | `NeutrinoPatternBase.h:397-401` | inert here |
| 13 | **P13** | kink-loop pass cap + stationarity detector | — | `NeutrinoPatternBase.cxx:1269,1372` | added guard |
| 14 | **P14** | 13 raw pointer-ordered `boost::out_edges` at HEAD where the prototype's maps are id-ordered; none reachable, all degree-guarded or self-sorted; an uncommitted concurrent sweep is converting them | id-ordered maps | `NPB:1580,1899,2440`, `NSE:721,1005,1067,1286,1648,2135,2452,3075,3262`, `NOS:1119` | latent |

P1–P4 and P6–P7 change production output unconditionally → CLAUDE.md §5 rule 1
→ reported, not fixed. P14 changes nothing today (§6) and is already being
addressed by someone else.

---

## §9 What is NOT claimed

* **No event was run.** Every "this changes the output" above is an argument
  from the source, not an observed diff. In particular P1's *reach* — how many
  SBND points actually carry a cross-segment 2-D association for
  `update_association` to strip — is unmeasured, and one instrumented run
  settles it.
* **No recommendation is made on any of the thirteen.** The porting dictionary
  has no section for this stage (§7.5), so per §5 rule 4 both readings are
  presented and neither is picked.
* **Tier B coverage is what §1 says it is.** For `find_other_segments`,
  `crawl_segment`, `examine_segment`, `examine_vertices_{1,2,3,4}`,
  `examine_partial_identical_segments`, `check_end_point`,
  `find_vertex_other_segment` and `modify_vertex_isochronous`, "matches" means
  the signature, the default arguments and the numeric constants match. It does
  **not** mean every branch was compared. A divergence that changes control
  flow without changing a constant would not have been caught in those
  functions.
* **The out-of-scope list in §0 is not claimed clean.** `examine_structure*`
  (3,517 lines) is called twice from the middle of `find_proto_vertex` and is
  entirely unaudited; a divergence there would sit inside this stage's results
  even though it is outside this document.
* **P5's reach is a sweep this audit did not do.** The mechanism is certain;
  the number of order-sensitive `find_vertices` callers is not stated because
  it was not counted.
* **P14's 13 sites are judged, not proven.** "Not order-sensitive" rests on
  reading each loop's body and its upstream degree guard, not on running the
  same event twice under `setarch -R` and diffing. The two first-edge sites
  (`NSE:1286`, `NSE:2452`) are safe *because of* a guard several lines away;
  that is an invariant nothing enforces.
* **The concurrent working-tree sweep is not reviewed.** Four hunks are quoted
  in §Repro; the rest of it was not read, and this document makes no statement
  about whether it is correct or complete.
* **Nothing here is a knob-off byte-identicality claim.** No gate was run;
  none is owed, because no code changed.

---

## §10 Owner filter, 2026-08-04 — the four that are bugs or gaps

**What was asked.** The owner read §3/§8 and applied a filter: *"skip the ones
that are improvements over the previous prototype, and only focus on the ones
that are bugs or missing from the port."* Two clarifications came with it, and
both dispose of findings:

1. **"In the toolkit we are missing id information for some data, so we have to
   use positions to do it — this is not a problem."** The prototype identifies
   graph endpoints by `wcpt().index` (a Steiner point-cloud index) and orders
   its maps by `get_id()`. Where the toolkit substitutes a **geometric**
   equivalent for either, that is an accepted design consequence, not a
   divergence.
2. Re-check both trees at current HEAD rather than trusting §3's anchors.

**Method.** Every claim below was re-verified against toolkit **`f1e29f19`**
(§3 was written at `ea1a7e3d`) and `prototype_base/`. Three §3 statements did
not survive re-verification unchanged and are corrected in place below.
**No code is changed by this section**, so no gate is owed and none was run.

### §10.1 The filtered list — four items

| # | id | what it is | why it survives the filter | severity |
|---|---|---|---|---|
| **F1** | P1 | `flag_exclusion` is never passed `true` | the toolkit does **less** than the prototype at 28 of 30 sites; the callee is fully ported and dead | **highest** |
| **F2** | P9 | the out-of-volume skip has **three different, undeclared biases** at three in-scope sites | the guard is necessary, but no two sites agree on what an unevaluable point *means*, and none matches the prototype | **medium-high** |
| **F3** | P8 | the endpoint-consistency check is not ported **and has no positional replacement** | a check the prototype had, dropped rather than translated — the one place clarification (1) does *not* cover | medium |
| **F4** | P5, narrowed | the **two** callers in either tree that read `find_vertices(sg).first` as "the" vertex | at these two sites the prototype's `.first` is not expressing a positional concept, so the id→position clearance does not reach them; the two conventions name **opposite ends** of the second half of every broken segment (§10.5) | medium |

Everything else is dropped — see §10.6.

### §10.2 F1 (P1) — `flag_exclusion` — re-verified, one count corrected

Re-run at `f1e29f19`:

```bash
grep -rhn "do_multi_tracking(" prototype_base/pid/src/NeutrinoID*.h \
  | grep -v "^\s*[0-9]*:\s*//" | grep -c "true, true, true)"      # -> 28
grep -rn "track_fitter\.do_multi_tracking(\|local_fitter\.do_multi_tracking(" \
     clus/src/*.cxx | wc -l                                       # -> 31
grep -rhn "track_fitter\.do_multi_tracking(\|local_fitter\.do_multi_tracking(" \
     clus/src/*.cxx | grep -c "true, true, [a-z]*, true"          # -> 0
```

**Correction to §3.1 and §8: there are 31 toolkit call sites, not 32.** The
32nd match in the original grep is a *comment* at `TrackFitting.cxx:1135`.
Nothing else moves: 31 real sites, **all** passing `flag_exclusion = false`
(`NeutrinoVertexFinder.cxx:500` calls the 3-argument form, whose default is
also `false`). The prototype still shows 28 sites passing `true` and exactly
**two live** sites passing `false` (`NeutrinoID_proto_vertex.h:722`, `:751`,
both inside `break_segments`); a third such line at `:776` is inside a `/* */`
block and does not count.

One fact worth adding, because it strengthens the "dropped argument" reading
over the "deliberate standardisation" one: **both signatures default
`flag_exclusion` to `false`** —

```cpp
// prototype  PR3DCluster.h:182
..., bool flag_dQ_dx_fit = true, bool flag_exclusion = false);
// toolkit    TrackFitting.h:446
..., bool flag_force_load_data = false, bool flag_exclusion = false, ...);
```

— so the prototype's 28 `true`s are all *explicit overrides of the same
default*. The toolkit did not inherit a different default; it stopped writing
the override. That is the signature of a lost argument during translation, not
of a design decision, and there is still no comment at any of the 31 sites
saying otherwise.

**This is F1 and it is the item to act on first.** The recommendation is
unchanged from §3.1: it is not a fix to apply silently, because turning
exclusion on changes every fit in the stage. It wants a default-OFF knob and an
A/B, plus the unit check flagged in §3.1 (the transverse-coordinate
reconstruction inside `update_association` has never been exercised).

### §10.3 F2 (P9) — the out-of-volume skip: a **third** site, and three different meanings

§3.9 reported two sites with opposite bias. Re-reading at `f1e29f19` finds
**three in-scope sites**, with **three distinct semantics** for the same
unevaluable point:

| site | function | code | an out-of-volume point is treated as… |
|---|---|---|---|
| `NeutrinoOtherSegments.cxx:1253-1259` | `modify_segment_isochronous` | `if (face()!=-1 && apa()!=-1) { … n_bad++ }` | **good** — it cannot increment `n_bad`, so the bridge is judged *connected* |
| `NeutrinoStructureExaminer.cxx:1273` | `examine_vertices_1p` | `continue` before the `flag_dead=false; break;` | **dead** — `flag_dead` stays `true`, so the segment is judged to lie in a dead region |
| `NeutrinoStructureExaminer.cxx:2552` | `examine_vertices_3` | `continue` before the uniqueness test | **not unique** — the point cannot contribute `num_unique`, so the segment is more likely to be *removed* |

The prototype evaluates every step point in all three
(`is_good_point(test_p, 0.2 cm, 0, 0)` /
`get_closest_dead_chs(...)`, e.g. `NeutrinoID_proto_vertex.h:1645`), so none of
the three matches it.

**Why this is a bug and not an improvement.** The guard itself is *required* —
`contained_by` returns `-1` outside the detector and the downstream
`m_trigger_offsets.at(-1)` / `m_anodes.at(-1)` throws and aborts the job (the
class-C crash shape, doc pr/11 §6.3; the comment at `NSE:1313-1317` names SBND
MCP2025C evt 49951). Nobody disputes the guard. What is defective is that the
**failure semantics were never chosen**: three sites in the same stage silently
picked three different answers to "what does an unevaluable point mean", and at
least two of them are the *permissive* answer, which is the direction that
quietly admits geometry the prototype would have rejected.

`examine_structure_4` (`NSE:539-546`) shows what the considered version looks
like — it has the same guard *with a stated reason* ("terminals outside every
TPC are rejected immediately — without this guard, `min_dis_{u,v,w}` stay at
`1e9` and the 2D-distance criterion trivially passes"). That site is out of
scope per §0 but is the model: the other three should each state, and justify,
their polarity.

**Reach is not measured.** How often an SBND fit point falls outside every
volume is one instrumented count, and this document does not have it. The
places it would matter most are the cathode region and the readout-window
edges.

### §10.4 F3 (P8) — the endpoint check is dropped, not translated

Confirmed unchanged at `f1e29f19`: `PRGraph.cxx:41-92` `add_segment` links any
two vertices with no consistency test, while the prototype refuses:

```cpp
// prototype NeutrinoID_proto_vertex.h:1952-1956
if (pv->get_wcpt().index != ps->get_wcpt_vec().front().index &&
    pv->get_wcpt().index != ps->get_wcpt_vec().back().index){
  std::cout << "Error! Vertex and Segment does not match " << … ;
  return false;
}
```

**This is the one item the id→position clearance does not cover, and the reason
is worth stating.** Clarification (1) accepts substituting geometry for an
index *where the toolkit does the substitution*. Here it does not: there is no
positional analogue of this check anywhere in `PRGraph.cxx` — the check is
simply absent. The prototype's practical behaviour on a mismatch is *the
connection is silently not made* (most callers ignore the return value); the
toolkit's is *the connection is made*.

**It interacts with F4.** `find_vertices` (`PRGraph.cxx:105-141`) decides its
ordering by asking which endpoint's `wcpt()` is nearer the segment's **first**
wcpt. That question is only meaningful if both vertices really do sit at the
segment's two ends — exactly the invariant `add_segment` no longer enforces. A
mis-attached vertex therefore does not announce itself; it silently produces a
wrong `.first`, which is then consumed at the two sites in §10.5.

**Not measured, and the evidence is weak in both directions.** No live path is
known to produce a mismatch. The prototype's diagnostic would have printed if
one did, which is weak evidence that none does — weak because nobody has
confirmed the prototype was ever run with that path live on SBND-like geometry.
The cheap version of this finding is a `SPDLOG_LOGGER_WARN` in `add_segment`
when neither endpoint is within a tolerance of the segment's two ends: it costs
nothing when the invariant holds and converts a silent corruption into a log
line when it does not.

### §10.5 F4 — the narrowed P5: exactly two order-sensitive callers, in both trees

§9 recorded that P5's reach was never swept. It has been swept now.

**The sweep.** `find_vertices(` has ~60 call sites in `clus/src`. Every one was
checked for whether it uses `.first` / `.second` asymmetrically:

* the large majority bind both and **re-derive** start/end from position
  anyway — `NeutrinoTrackShowerSep.cxx:83,347`, `NeutrinoPatternBase.cxx:1218`
  (`break_segments`, §3.11), `NeutrinoStructureExaminer.cxx:429-431`
  (compares `.first` against a known vertex by pointer identity, which is
  order-free);
* `NeutrinoVertexFinder.cxx:3467` writes `bool flag_start = (v1 == min_vertex)`.
  This *looks* order-sensitive and is **not a finding**: it is the positional
  translation of the prototype's own idiom
  `sg->get_wcpt_vec().front().index == vtx->get_wcpt().index`
  (`NeutrinoID_DL.h:94-98`, and identically at
  `NeutrinoID_final_structure.h:44`, `NeutrinoID_shower_clustering.h:974`,
  `NeutrinoID_nue_tagger.h:1442`). This is precisely clarification (1), and the
  toolkit version is in fact safer — the prototype leaves `flag_start`
  **uninitialised** when neither branch matches;
* **two sites read `.first` alone and use it as "the" vertex**, and they are the
  same two in both trees:

| | prototype | toolkit |
|---|---|---|
| | `NeutrinoID_singlephoton_tagger.h:167` `auto shw_vtx_main = find_vertices(sg_start).first;` | `NeutrinoTaggerSinglePhoton.cxx:2361` |
| | `NeutrinoID_singlephoton_tagger.h:342` `auto shw_vtx = find_vertices(sg).first;` | `NeutrinoTaggerSinglePhoton.cxx:2440` |

At both, the chosen vertex is passed straight to `bad_reconstruction_2_sp` and
`bad_reconstruction_3_sp` as a geometric anchor, and the resulting flags feed
the single-photon tagger.

**Why this is not covered by clarification (1).** The prototype's `.first` here
is *not* a positional concept being approximated. `find_vertices` in the
prototype (`NeutrinoID_proto_vertex.h:3227-3243`) returns the first two
elements of `map_segment_vertices[sg]`, ordered by `ProtoVertexCompare` on
`get_id()` — i.e. **whichever endpoint was created first**. The toolkit returns
the endpoint nearest the segment's front wcpt. Those are different vertices,
and the substitution is not "position instead of id" — it is "a meaningful
choice instead of an arbitrary one".

**How often do the two conventions actually disagree? On exactly half of every
break.** They coincide more often than "arbitrary vs. geometric" suggests,
because prototype ids are handed out monotonically and vertices are usually
created in path order — so for a segment whose start vertex is the older one,
"lowest id" and "nearest the front wcpt" name the same vertex. `break_segments`
makes this precise (`NeutrinoID_proto_vertex.h:732-741`):

```cpp
ProtoVertex *v3  = new ProtoVertex(acc_vertex_id, break_wcp, …); acc_vertex_id++;
ProtoSegment *sg2 = new ProtoSegment(acc_segment_id, wcps_list1, …);  // start_v → break
ProtoSegment *sg3 = new ProtoSegment(acc_segment_id, wcps_list2, …);  // break → end_v
add_proto_connection(start_v, sg2, …);  add_proto_connection(v3, sg2, …);
add_proto_connection(v3,      sg3, …);  add_proto_connection(end_v, sg3, …);
```

* **`sg2`** is bounded by `start_v` (pre-existing, lower id) and `v3` (created
  just now, highest id). `sg2`'s wcpts run `start_v → break`, so the lowest-id
  endpoint *is* the front one. **The two conventions agree.**
* **`sg3`** is bounded by `v3` (highest id) and `end_v` (pre-existing, lower
  id). `sg3`'s wcpts run `break → end_v`, so the lowest-id endpoint is the
  **back** one. The prototype's `.first` is `end_v`; the toolkit's is `v3`.
  **The two conventions name opposite ends.**

So this is not a rare re-attachment corner case: **every break produces one
segment of each kind**, and the second half of every broken track is a segment
on which prototype `.first` and toolkit `.first` are the two different
endpoints. If a single-photon shower's start segment is a `sg3`-type half —
and a shower trunk broken off a parent track is exactly that — the two trees
anchor `bad_reconstruction_2_sp` / `_3_sp` at opposite ends of it.

**Both readings, and neither is obviously right.** (i) The toolkit is better:
`.first` now has a defined geometric meaning, and a tagger anchoring on "the
lower-id endpoint" was never intentional. (ii) The prototype's arbitrary choice
is the one the single-photon tagger's cuts were **tuned against**, so the
toolkit silently re-anchors two live tagger tests; and "nearest the front wcpt"
is not the same as "the shower start" either, so the toolkit's choice is
principled but not obviously the intended one. Deciding this needs the
single-photon tagger's own audit, not this document.

**Scope note — F4's two sites are outside §0.** They live in
`NeutrinoTaggerSinglePhoton.cxx`, which §0 puts out of scope along with the
rest of the taggers. They surface here anyway because **P5's reach question is
global by nature**: the changed contract is in `PRGraph.cxx`, a helper this
stage owns, so bounding its consequences means leaving the stage. Read F4 as
*"the reach sweep P5 owed, completed"* — **not** as a claim about tagger code
that was audited. Nothing else in `NeutrinoTaggerSinglePhoton.cxx` was read,
and per §10.9 the adjudication belongs to that file's own audit.

**P5 as a whole is otherwise closed** by clarification (1), and §9's open
"P5's reach is a sweep this audit did not do" is now discharged: the answer is
**two sites**, both listed above.

### §10.6 What was dropped, and why — one line each

Recorded as the **owner's** decision of 2026-08-04, not as a conclusion of this
audit. §3's both-readings posture (CLAUDE.md §5 rule 4) is left intact above;
this is the filter applied on top of it.

**Accepted as improvements over the prototype** — toolkit-only additions, each
verified still present at `f1e29f19`:

* **P2** local-PCA endpoint refinement (`NeutrinoPatternBase.cxx:567-676`) —
  the in-code rationale is explicit ("when the cluster curves near its end, the
  true tip may lie around a corner and be missed"), commit `1eb097a9` *"fix a
  bug and randomness"*.
* **P3** the wider terminus-stub absorption (`:1472-1476`) — three commits of
  refinement, the last (`6127bc31`) adding the 45° guard specifically so a
  12.94 MeV Michel at ~79° is **not** eaten. That is the fingerprint of a fix
  tuned against a real case, not of a drift.
* **P4** the extra acceptance clause `0.72 / 15 cm / 1.05`
  (`NeutrinoOtherSegments.cxx:558-560`) — a strict widening; it can only admit
  segments the prototype rejected.
* **P6** `merge_nearby_vertices` (0.1 cm) after breaking — re-enables, in
  looser form, a cleanup the prototype defined and then commented out (§5.3).
* **P7** the walk-halfway override (`:1299-1321`) — a break-oscillation fix
  with no prototype counterpart.
* **P10** the "no segments survived → return false" recovery — gives the
  caller a fallback the prototype never had.
* **P13** the kink-loop pass cap and stationarity detector — fire only where
  the prototype would hang.

**Accepted under clarification (1) — id information replaced by geometry:**

* **P5** (except the two sites in §10.5) — `find_vertices` ordered by
  proximity instead of by vertex id.
* **P11** — `break_segments` determines start/end by distance instead of
  `wcpt().index` equality. The toolkit also cannot reach the prototype's
  null-dereference at `:615`, which is a second reason not to reproduce it.

**Not a divergence to act on:**

* **P12** — the `main_vertex` parameter is `nullptr` throughout this stage;
  inert here by construction, and live only at the out-of-scope call sites.

### §10.7 P14 is **resolved**, verified at `f1e29f19`

§6 reported 13 raw pointer-ordered `boost::out_edges` at `ea1a7e3d` and noted a
concurrent uncommitted sweep. That sweep has landed:

```bash
for f in clus/src/NeutrinoPatternBase.cxx clus/src/NeutrinoStructureExaminer.cxx \
         clus/src/NeutrinoOtherSegments.cxx; do
  echo -n "$f raw="; grep -c "boost::out_edges" $f
done
# -> 0, 0, 0
```

All 13 in-scope sites now use `sorted_out_edges`. `NeutrinoVertexFinder.cxx`
retains 2 raw uses; that file is outside this audit's scope (§0) and is doc
pr/28's territory. **P14 is closed** — the latent risk §6 described (the two
first-edge sites resting on a degree guard several lines away) no longer
exists, because the iteration is now ordered by construction rather than by
invariant.

### §10.8 Corrections to earlier sections

Three §3/§8 statements did not survive re-verification and are corrected here
rather than edited in place, so the original record stays readable:

1. **§3.1 / §8 row 1: "32 toolkit call sites" → 31.** The 32nd grep match is a
   comment at `TrackFitting.cxx:1135` (§10.2).
2. **§3.9 / §8 row 9: "opposite bias at two sites" → three sites, three
   distinct biases.** `NeutrinoStructureExaminer.cxx:2552`
   (`examine_vertices_3`) was missed; it treats an unevaluable point as *not
   unique*, a third polarity (§10.3).
3. **§8 row 14 / §6: P14 is no longer latent — it is resolved** (§10.7).

§9's "P5's reach is a sweep this audit did not do" is discharged by §10.5.
Everything else in §9 still stands, in particular: **no event was run for this
document**, so every reach claim above remains an argument from the source.

### §10.9 What this section does not claim

* **No code changed**, so no A/B gate is owed and none was run.
* **F1's reach is still unmeasured.** How many SBND points carry a
  cross-segment 2-D association for `update_association` to strip is one
  instrumented run, and it has not been done.
* **F2's rate is unmeasured.** How often a fit point lands outside every
  detector volume on SBND is a count nobody has taken.
* **F3 is unobserved.** No live path is known to attach a vertex to a segment
  it does not end on; that is not the same as knowing none does.
* **F4 is not adjudicated.** Both readings are given and neither is picked; the
  single-photon tagger's own audit is where that decision belongs.
* **Tier B coverage is unchanged** (§1, §9). Dropping P2/P3/P4/P6/P7/P10/P13
  from the action list does not upgrade the reading behind them.

---

## §12 Implementation and validation — five knobs, 48 nueCC data events

**What was asked.** The owner asked for fixes to **P1** and **P8**, for **P2**
and **P4** to be improved using the 48 nueCC data events to inform the
decision (*"for P4, we want to avoid major bugs"*), and for **F2** to be
resolved by reading each of the three cases and choosing the best treatment.

**What shipped.** Five config knobs, every one of them defaulting to the
pre-pr/30 behaviour, plus unconditional log-only instrumentation. **The
knob-off path is proven byte-identical** (§12.2). Nothing is turned on; every
"should this ship on?" question is answered with a measurement below and left
to the owner.

### Repro

```bash
wcbuild && ./build/clus/wcdoctest-clus
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# knob-off arm (production defaults) and the pre-change baseline
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr30-final data
#   baseline = the same command with the pr/30 diff stashed and rebuilt at 6206c46b
#   -> work-pr30-baseHEAD

# knob-on arms.  The SBND operating point lives only in cfg (doc 68), so each
# arm is a COPY of $TK/cfg with exactly one line of
# cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet changed -- production cfg
# is never edited.  ARMDIR below is scratch and is meant to be regenerated:
#   p1    : fit_exclusion            = false -> true
#   f2    : oov_prototype_parity     = false -> true
#   p8    : graph_endpoint_strict    = false -> true
#   p2off : first_seg_local_pca      = null  -> false
#   p4off : other_seg_relaxed_accept = null  -> false
# tmp_run_pr_chain_pr30.sh is run_pr_chain_batch.sh with line 54 changed to
# honour $PR30_CFG, and nothing else (committed alongside this doc).
ARMDIR=/home/xqian/tmp/pr30cfg          # regenerate: tar -cf - cfg | tar -x -C $ARMDIR/<arm>
for a in p1 f2 p8 p2off p4off; do
  PR30_CFG=$ARMDIR/$a/cfg PR_JOBS=6 \
    ./tmp_run_pr_chain_pr30.sh work-nuecc48-prod0803 work-pr30-$a data
done

EVTS=$(ls work-pr30-final | sed -n 's/^pr_evt//p')
python3 scripts/analysis/misc/pr_arm_compare.py work-pr30-baseHEAD work-pr30-final $EVTS
python3 pr_scores_table.py --root work-pr30-<arm> --out <arm>.tsv
```

### §12.1 The five knobs

Config keys on `TaggerCheckNeutrino`, threaded to `PatternAlgorithms`
(and, for P8, to a process-wide policy struct because `PR::add_segment` is a
free function with no component config in reach).

| key | finding | C++ default | what ON does |
|---|---|---|---|
| `fit_exclusion` | **P1** | `false` | 27 `do_multi_tracking` sites pass `flag_exclusion=true` |
| `graph_endpoint_strict` | **P8** | `false` | `add_segment` refuses an inconsistent connection |
| `graph_endpoint_tol` | **P8** | `0.3` cm | positional stand-in for the prototype's wcpt-index equality |
| `oov_prototype_parity` | **F2** | `false` | all three out-of-volume guards vote the prototype's way |
| `first_seg_local_pca` | **P2** | `true` | *(default = production)*; `false` drops the local-PCA refinement |
| `other_seg_relaxed_accept` | **P4** | `true` | *(default = production)*; `false` restores the prototype's clause |

The first three default to `false` because they introduce **new** behaviour.
The last two default to `true` because the behaviour they gate is **already
production** — for those, the knob's purpose is that an unconditional,
un-knobbed departure from the prototype could not previously be measured at
all. Either way the defaults reproduce the pre-pr/30 tree exactly.

**Not knobbed, deliberately, and commented at each site:**
`NeutrinoPatternBase.cxx:1498` and `:1516` (`break_segments`) stay hard
`false` — they are the toolkit's *correct* match to the prototype's own two
`false` sites (`NeutrinoID_proto_vertex.h:722/751`, *"fit dQ/dx here, do not
exclude others"*), and knobbing them would break the one place parity already
holds. `NeutrinoPatternBase.cxx:1708` (`merge_nearby_vertices`) stays hard
`false` because that function is **toolkit-only** (P6) and the prototype
therefore has no answer for it — inventing one is what §5 rule 4 forbids.
`NeutrinoVertexFinder.cxx:505` uses the 3-argument form on a **single-segment**
local graph, where `update_association` has no other segment to exclude.

### §12.2 The gate — knob-off is byte-identical

Baseline arm produced by stashing the pr/30 diff, rebuilding at **`6206c46b`**,
and running the 48 events; then the diff restored and rebuilt.

| pair | mabc | pctree | `T_tagger`+`T_kine` | `nusel-evt*.tsv` |
|---|---|---|---|---|
| `work-pr30-baseHEAD` vs `work-pr30-final` | **48/48** | **48/48** | **48/48** | **48/48 byte-identical** |

Compiled config, with the real `pipeline_names` TLA: the default job JSON is
**byte-identical** to the pre-change tree (`cmp` clean) and contains **none**
of the five keys; each knob-on cfg copy shows exactly its own key
(`"fit_exclusion" : true`, …). `./build/clus/wcdoctest-clus`: **91 cases /
963 assertions**, including two new revert-proven P8 cases (§12.7).
Freshness proof done before every arm.

### §12.3 What the divergences actually do — measured at the production point

Counters are unconditional and run in the knob-**off** arm, which is what makes
them a measurement of *reachability* rather than of the fix. 48 events, 47 of
which build a PR graph (evt 116962 is TGM-tagged and builds none, by design).

| finding | counter | 48-event total | events > 0 |
|---|---|---|---|
| **F2** site 1 `modify_segment_isochronous` | `oov_iso` | **0** | 0 |
| **F2** site 2 `examine_vertices_1p` | `oov_dead` | **0** | 0 |
| **F2** site 3 `examine_vertices_3` | `oov_uniq` | **1** | 1 (evt 54095) |
| **P8** connections made | `addseg` | 5284 | 47 |
| **P8** inconsistent | `ep_mismatch` | **108** | **22** |
| **P2** endpoint refinements attempted | `pca_calls` | 2152 | 47 |
| **P2** endpoint actually moved | `pca_moved` | **635 (29.5%)** | 47 |
| **P2** largest single move | `pca_max_cm` | **9.90 cm** *(max over events, not a sum)* | — |
| **P4** accepted by a prototype clause | `oseg_proto` | 29 | 19 |
| **P4** accepted by the toolkit clause ALONE | `oseg_relaxed_only` | **9** | **8** |
| **P4** rejected by all clauses | `oseg_reject` | 592 | 47 |

### §12.4 Knob-on effects, 48 events

| arm | events changed (mabc) | `nue_score` differs | sign flips +→− / −→+ | selected-vertex shift |
|---|---|---|---|---|
| `fit_exclusion=true` (**P1**) | **47/48** | 17/48 | **7 / 3** | median 0.70 cm, max **97.0 cm** |
| `oov_prototype_parity=true` (**F2**) | **0/48** | 0/48 | 0 / 0 | 0 |
| `graph_endpoint_strict=true` (**P8**) | 22/48 | 9/48 | **5 / 1** | median 0, max 77.5 cm |
| `first_seg_local_pca=false` (**P2**) | 47/48 | 13/48 | 1 / 0 | median 0, max 86.8 cm |
| `other_seg_relaxed_accept=false` (**P4**) | 7/48 | 2/48 | 0 / 0 | median 0, max 85.2 cm |

**Arm provenance.** `work-pr30-final` (knob-off gate arm) and the
`graph_endpoint_strict` arm ran on the **shipped** binary. The
`fit_exclusion`, `oov_prototype_parity`, `first_seg_local_pca` and
`other_seg_relaxed_accept` arms ran on an earlier build of the same branch that
differs **only inside `PR::add_segment`'s P8 block** (where the consistency
check sits relative to the `descriptor_valid()` early return, plus one
diagnostic counter). None of those four knobs reaches that code, and the P8
check is a no-op with `graph_endpoint_strict=false`, so the difference cannot
touch their numbers — but the arm labels differ from a fresh re-run, and that
is bookkeeping, not a defect.

`numu_score` moves on 47/48 under P1 and `kine_reco_Enu` by a median of
35.5 MeV (max 429 MeV). **No arm flips any event's `event_label`** — the
cosmic/nu-candidate verdict is stable throughout; what moves is the selected
vertex, the energy and the BDT scores.

**Sign flips are counted on a nueCC sample**, so "+→−" is a lost signal event
and "−→+" a recovered one. That framing is why P1 is reported below as a
regression at this operating point rather than a fix.

### §12.5 F2 — implemented, and it is **inert on this manifest**

Each of the three sites was read against the prototype's own helper, and the
result was not a judgement call: **all three toolkit defaults are the opposite
of what the prototype actually returns** for a point with no readout.

| site | toolkit's implicit vote | prototype's helper | parity |
|---|---|---|---|
| `NeutrinoOtherSegments.cxx:1253` | good / connected | `is_good_point` → `num_planes==0` → **false** (`ToyCTPointCloud.cxx:399-431`) | `n_bad++` — **exact** |
| `NeutrinoStructureExaminer.cxx:1273` | dead | `get_closest_dead_chs` → channel absent from `dead_uchs/vchs/wchs` → **false** | `flag_dead=false` — **exact** |
| `NeutrinoStructureExaminer.cxx:2552` | not unique (⇒ segment **removed**) | `get_closest_2d_dis` is a pure kd-tree 2-D distance, **no volume check** (`ProtoSegment.cxx:1094-1103`) | `num_unique++` — **directional inference**, flagged as such in the code |

**And it fires essentially never.** Two of the three guards did not trip once
in 48 events; the third tripped **once**, and the knob-on arm is byte-identical
to the knob-off arm on all 48 events and all four artifact families. So F2 is a
real code defect with **zero measured effect on nueCC data** — correctness
bought on the code, not on a number. The population where it could matter is
the cathode region and the readout-window edges, which this manifest barely
populates.

### §12.6 P8 — the check works, and what it caught is **not** a lost invariant

108 firings across 22 of 48 events looked like the headline result. It is not,
and the correction matters more than the original number.

Running one high-count event (**38856**, 13 firings) under `WCT_DET_DEBUG=2`
and correlating each WARN against the `add_segment` creation backtraces
attributes **9 of 9 matched firings to a single call site**:
`PatternAlgorithms::crawl_segment`. Reading it explains the rest
(`NeutrinoStructureExaminer.cxx:892` and `:944`):

```cpp
auto path_points = do_rough_path(cluster, vtx_new_point, min_wcpt_point); // ends AT vtx_new_point
...
add_segment(graph, new_other_seg, flag_front ? vertex : other_v2, ...);   // attached to `vertex`
...
vertex->wcpt().point = vtx_new_point;                                     // :948 -- AFTER
```

The rebuilt segments terminate at `vtx_new_point`; the vertex is only *moved*
to `vtx_new_point` after every connection has been made. **The inconsistency is
transient and self-healing** — by the time `crawl_segment` returns, every one of
those connections is consistent. The measured offsets fit: in all 104 logged
calls exactly one vertex is at 0.000 cm and the other at 0.3–2.9 cm
(median 1.18 cm), the scale of a crawl step, with nothing piled just above the
0.3 cm tolerance.

**Therefore `graph_endpoint_strict` is a false positive as placed** — it
refuses legitimate connections in the middle of a repair, and the measurement
shows the damage: 22/48 events change, **5 nue candidates lost against 1
gained**. It must stay OFF. A check for a *persistent* violation would have to
run at end-of-stage, not inside `add_segment`.

The WARN and counter are retained as a **tripwire**: they cost nothing, they
are on in production, and a firing from any call site other than
`crawl_segment` would be the real thing. That is stated in the code so the next
reader does not re-derive it.

**Two corrections to my own work on the way to this, recorded because they
each nearly became the conclusion:** (i) the first backtrace correlation
matched zero firings and I read that as "these are all no-op re-calls" — it was
a **unit error in the correlation script** (`WCT_DETA` prints internal units,
the WARN prints cm); the measured re-entry count is **0**, so no call ever
arrives with the segment already in the graph. (ii) The check was nevertheless
moved after the `descriptor_valid()` early return, because guarding *a
connection being made* is the correct semantics regardless — measured to change
nothing here, and pinned by a doctest.

### §12.7 Tests

`clus/test/doctest_pr_graph_order.cxx` gains two cases, both revert-proven:

* *"pr30 P8 inconsistent endpoints are counted only when a connection is
  made"* — moving the check back above the `descriptor_valid()` early return
  makes it fail (`CHECK( 2 == 1 )`, verified by actually reverting and
  rebuilding).
* *"pr30 P8 strict mode withholds the edge but keeps the vertices"* — pins that
  strict refuses the **edge** (`num_edges == 0`) while still recording both
  vertices (`num_vertices == 2`), which is what the prototype does.

### §12.8 Recommendations — and what I am NOT recommending

Nothing is proposed for a default flip. Concretely:

* **P1 (`fit_exclusion`) — do NOT turn on at this operating point.** It is the
  clearest port-fidelity gap in the document and the fix is now one config key,
  but on 48 nueCC data events it is a **net regression**: 7 nue candidates lost
  against 3 gained, `numu_score` moved on 47/48, the selected vertex moved by
  up to 97 cm. That is not evidence that exclusion is wrong — it is evidence
  that every cut downstream was tuned against the non-excluding fit. The
  §3.1 unit question (prototype `y = (wire-offset)*pitch_u` vs toolkit
  `raw_y = (wire-offset)/slope_yu`) is now **reachable** and should be settled
  *before* anyone reads these numbers as physics: a dead code path is exactly
  where a unit error survives.
* **P8 (`graph_endpoint_strict`) — do NOT turn on.** §12.6. Keep the tripwire.
* **F2 (`oov_prototype_parity`) — safe to turn on whenever you like**, and it
  buys nothing measurable here (byte-identical on 48/48). The argument for
  turning it on is fidelity and future-proofing, not a number.
* **P2 (`first_seg_local_pca`) — keep it ON (production), now measurable.**
  It moves an endpoint in 29.5% of attempts, by up to 9.9 cm, and turning it
  off changes 47/48 events and loses 1 nue candidate net. It is doing real
  work, and the data give no reason to remove it. It is now knobbed and
  counted, which is what it lacked.
* **P4 (`other_seg_relaxed_accept`) — keep it ON, and the "avoid major bugs"
  question is answered: it is not admitting garbage.** Across 48 events the
  toolkit-only clause admitted **9 segments in 8 events** — 9 out of 630
  candidates (1.4%), against 592 rejections and 29 prototype-clause accepts.
  Turning it off changes 7/48 events, moves `nue_score` on 2, and flips **no**
  sign and **no** label. It is a small, well-behaved widening, not a leak.

### §12.9 What this section does NOT claim

* **The manifest is 48 nueCC data events.** It is not a population gate. The
  owed valfast/1000 census is not discharged by anything here, and F2's
  "0 firings" in particular is a statement about *this* manifest.
* **No truth labels.** Sign flips are counted against the *current* production
  answer, not against truth. "7 lost, 3 gained" under P1 means seven events
  whose `nue_score` sign changed from + to −; on a nueCC sample that is the
  bad direction, but it is not an efficiency measurement.
* **P8's attribution rests on one event.** 9 of 13 firings in evt 38856 were
  matched to a backtrace, all to `crawl_segment`; the remaining 4 and the other
  21 events were not individually traced. The code reading generalises it, the
  measurement does not.
* **P1's reach is now measured; its correctness is not.** That the knob changes
  47/48 events says nothing about whether the excluded fit is *better*.
* **Nothing here revisits F4** (doc §10.5) — out of scope for this round and
  still unadjudicated.

### §12.10 Owner decision, 2026-08-04 — F2 flipped ON, P8 and P4 closed

The owner read §12.8 and adopted the recommendation. This section records what
changed and what is now closed, so nothing here is re-litigated.

**F2 `oov_prototype_parity` — SBND PRODUCTION DEFAULT ON.**
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` now sets it `true`. The
C++ default stays `false`, so no other detector moves and the pre-flip arm is
one edit away.

| | |
|---|---|
| gate | `work-pr30-baseHEAD` vs **`work-pr30-f2on`** |
| mabc / pctree / `T_tagger`+`T_kine` | **48/48 / 48/48 / 48/48** |
| `nusel-evt*.tsv` | **48 compared, 0 differ** |
| knob engaged | `PR30AUDIT … oov_parity=true` in **47/47** events that build a PR graph |
| compiled config | one added key, `"oov_prototype_parity" : true`, and nothing else (`diff` against the knobs-off job JSON) |

**The reason to land it now is that it is currently free.** The three guards
fire once in total across 48 nueCC events (§12.3), so the flip is measurably
byte-identical. The population where they *would* fire — the cathode region and
the readout-window edges — is barely represented in this manifest. Landing the
fidelity fix while it costs nothing is strictly cheaper than flipping it later,
once it has become a real behaviour change needing its own A/B. The knob is
engaged and inert, not absent: the `oov_parity=true` count above is the proof
that this is a live-and-quiet flip rather than a no-op configuration.

**P8 — CLOSED, no fix owed.** §12.6 established that the 108 firings are the
`crawl_segment` transient and self-healing by the time that function returns.
There is no invariant being lost, so there is nothing to repair. Two decisions
follow and are now implemented:

* the log line is **demoted from WARN to DEBUG** (`PRGraph.cxx`). At WARN it
  emitted ~108 lines per 48 events of known-benign output, which trains a
  reader to ignore the one channel that would matter if it ever fired from a
  call site other than `crawl_segment`. The counter is unchanged, so the rate
  is still in every `PR30AUDIT` line;
* **`crawl_segment` is deliberately NOT "fixed".** Moving
  `vertex->wcpt().point = vtx_new_point` (`NeutrinoStructureExaminer.cxx:948`)
  above the `add_segment` calls would not be behaviour-preserving — `flag_front`
  at `:901` is computed against the *old* vertex position — and there is no
  defect to justify the risk. The check stays as a tripwire.

`graph_endpoint_strict` remains available and remains **OFF**, with §12.6's
measurement (22/48 events change, 5 nue candidates lost against 1 gained) as
the standing reason not to use it.

**P4 — CLOSED.** The owner's *"we want to avoid major bugs"* question is
answered by the numbers rather than by an argument: the toolkit-only clause
admitted **9 segments of 630 candidates in 8 of 48 events**, and turning it off
flips **no** `nue_score` sign and **no** `event_label`. It is a small,
well-behaved widening. `other_seg_relaxed_accept` stays at its `true` default
and no further investigation is planned.

**P2 — no action, and that is the decision.** It stays at its `true` default.
It is now knobbed and counted, which was the actual gap; the data give no
reason to remove something that moves an endpoint in 29.5% of attempts and
whose removal costs a nue candidate.

**P1 — the one item still open, and it is now blocked on a specific check.**
`fit_exclusion` stays OFF. Before anyone interprets §12.4's first row as
physics, the §3.1 transverse-coordinate question inside `update_association`
must be settled:

```cpp
// prototype PR3DCluster_multi_track_fitting.h:1013
double y = (wire - offset_u) * pitch_u;
// toolkit  TrackFitting.cxx:2568
double raw_y = (coord.wire - offset_u) / slope_yu;   // slope_yu = -sin(angle_u)/pitch_u
```

and the two feed different closest-2-D-distance helpers, each of which projects
again. Until this round that whole path was **dead code** — precisely where a
unit error survives unnoticed. The branch decides what §12.4 means:

* compositions **agree** ⇒ "7 nue lost, 3 gained" is a real statement about a
  fit whose downstream cuts were tuned without exclusion, and the next step is
  the valfast/1000 manifest, because ten flips out of 48 does not establish the
  sign of the net effect;
* compositions **disagree** ⇒ the P1-on arm measured a unit bug, §12.4's first
  row is void, and the fix is upstream of any A/B.

No compute should be spent on P1 until that is known. This is read-only work
plus a targeted unit test, not a campaign.

**Still owed, unchanged by this round:** the valfast/1000 population gate with
a regenerated baseline. Five more knobs now ride on it, all default-off except
the SBND `oov_prototype_parity` flip recorded above, which is gated
byte-identical here.

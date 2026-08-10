# doc pr/56 — 2D wind/tick connectivity redesign for `connect_graph_relaxed_strict`'s gap test

Status: **round 3 IMPLEMENTED, SBND PRODUCTION default still OFF (owner
review pending before any flip).** §1-§6 below are the original design
(unchanged as written except where round 2/3 explicitly override it). §7 is
the round 2 implementation record. §8 is round 3: two round-2 bugs fixed
(4-connected BFS stencil, wrong slice-adjacency stride), one owner-reported
false-positive rescued via a distance floor, adjacency widening tried and
rejected (it breaks a kill target), a 117-event sample-wide census +
fragment-count study, and a corrected 117-event byte-identical OFF gate.
Toolkit round-2 `a2a3c697`, round-3 `<TOOLKIT_SHA, this commit, not yet
pushed>`; wcp-porting-img round-2 `e85bd0f`, round-3 `<WCP_SHA, this
commit>`.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# round 2 (this implementation): byte-identical OFF gate + S6-on runs for
# the two named target events (fresh labels, both single-event reruns):
SBND_PROTECT_GRAPH=relaxed_strict_img_2d WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=1 \
    ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr56r2-on19 data 71372
SBND_PROTECT_GRAPH=relaxed_strict_img_2d WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=1 \
    ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr56r2-on48 data 269774

# OFF gate (bare, no env override -- must match the existing production
# baseline byte-for-byte since default graph_name is unchanged):
PR_JOBS=1 ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr56r2-off19 data 71372
PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr56r2-off48 data 269774
python3 ../../../abtest/hash_archive.py work-pr53r7-on19/pr_evt71372/mabc-pr.zip   work-pr56r2-off19/pr_evt71372/mabc-pr.zip
python3 ../../../abtest/hash_archive.py work-pr53r7-on48/pr_evt269774/mabc-pr.zip work-pr56r2-off48/pr_evt269774/mabc-pr.zip

# design-phase census rerun this doc's §1 motivating example is grounded in:
#   WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=1 \
#     ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr55-census269774 data 269774
grep -a "OC53CENSUS-S closest j=5 k=10\|OC53CENSUS-S closest j=5 k=11\|OC53CENSUS-S closest j=10 k=11" \
    work-pr55-census269774/pr_evt269774/wct_pr_evt269774.log

# after-fix Bee set (both events, S6-on PR output):
python3 scripts/bee/make_pr_bee.py -q work-ncpi0-cb0805 -q work-nuecc48-cb0805 \
    -p work-pr56r2-on19 -p work-pr56r2-on48 -o bee/pr56r2/pr56r2-after.zip 71372 269774

# round 3: sample-wide census (48 nueCC + 19 NC-pi0 + 50 PR-data), final
# operating point (repaired baseline, U/V/W adjacency all (1,1), distance
# floor 1.0cm):
M50=$(awk 'NR>1{print $2}' docs/pr/mcp1k-50-cb0805.index.txt)
SBND_PROTECT_GRAPH=relaxed_strict_img_2d WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=6 \
    ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr56r3-floor-cen48 data
SBND_PROTECT_GRAPH=relaxed_strict_img_2d WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=6 \
    ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr56r3-floor-cen19 data
SBND_PROTECT_GRAPH=relaxed_strict_img_2d WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=6 \
    ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr56r3-floor-cen50 data $M50

# byte-identical OFF gate (bare, no env override), all three samples:
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr56r3-off48 data
PR_JOBS=6 ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr56r3-off19 data
PR_JOBS=6 ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr56r3-off50 data $M50
python3 scripts/analysis/pr49/on_compare.py work-pr53r7-on48 work-pr56r3-off48
python3 scripts/analysis/pr49/on_compare.py work-pr53r7-on19 work-pr56r3-off19
python3 scripts/analysis/pr49/on_compare.py work-pr53r7-on50 work-pr56r3-off50

# offline operating-point scan from the census logs (no rerun needed per
# candidate point):
python3 scripts/analysis/pr56/s6_scan.py work-pr56r3-floor-cen48 work-pr56r3-floor-cen19 \
    work-pr56r3-floor-cen50 --near-touching 3.0

# round-3 Bee set (both named targets + two representative sample-wide
# fragmentation cases, final S6-on output):
python3 scripts/bee/make_pr_bee.py \
    -q work-ncpi0-cb0805 -q work-nuecc48-cb0805 -q work-nuecc48-cb0805 -q work-nuecc48-cb0805 \
    -p work-pr56r3-floor-cen19 -p work-pr56r3-floor-cen48 -p work-pr56r3-floor-cen48 -p work-pr56r3-floor-cen48 \
    -o bee/pr56r3/pr56r3-after.zip 71372 269774 122660 214469
```

Two figures sent this session document the visual gap this design targets
(vertex-region overview and a 3-panel X-Y/Y-Z/X-Z crop around the `c5-c10-
c11` neighborhood); not re-attached here, ask the session for regeneration
if needed.

## 1. What's broken (recap, grounded in this session's live data)

Doc pr/55 diagnosed that `do_rough_path` never even consults `protect_bundle`'s
membership graph (`relaxed_strict_img`) — it runs on the unrelated
`ctpc_ref_pid`/`steiner_graph` family. That's a routing bug. This doc is
about a second, independent problem the owner surfaced by hand-tracing one
of pr/55's own worked examples: **even the membership graph itself misses
gaps a human eye catches instantly**, because its connectivity test is
structurally too coarse.

Concretely, for event 269774 cluster 13 (case `269774-G1`, the ghost run at
`(89.9,8.7,137.9)→(94.9,15.1,136.7)`), the direct short-cut `c5→c11`
candidate (5.80cm, 6 steps) *is* correctly killed — but two individually
"legitimate" shorter hops bridge the same physical gap anyway, both logged
`killed=false` in a fresh rerun this session:

```
OC53CENSUS-S closest j=5  k=10 dis=2.00cm nsteps=2 strong=true  nb=[1,1,0,1] nb1=[1,1,0,1] killed=false
OC53CENSUS-S closest j=10 k=11 dis=2.69cm nsteps=3 strong=false nb=[0,0,2,0] nb1=[2,0,2,0] killed=false
OC53CENSUS-S closest j=5  k=11 dis=5.80cm nsteps=6 strong=true  nb=[4,4,2,3] nb1=[4,4,2,3] killed=true
```

**Root cause, mechanistically**: `relaxed_strict_bad(nbad, num_steps)`
(`Graphs.cxx:49-57`) requires `nbad ≥ 2 AND nbad ≥ 0.75×(num_steps−1)`
interior samples to be bad before it kills an edge. `c5→c10` has only 1
interior 1cm sample (`num_steps=2`) — `nbad` can be at most 1, and the floor
needs `≥2`, so this edge is **structurally immune** to the kill test no
matter how bad the true charge picture is: too short a gap to ever
accumulate two failures. `c10→c11` has 2 interior samples and one plane
(`nb[2]=2`) flags both as bad, but the wire-angle branch that actually fired
(`strong=false` routes into one of four angle-based sub-cases,
`:396-427`) evidently used a bad-count that excludes that plane's column,
so it also passed. Both survivals are properties of the *test's geometry*
(too few samples, or a branch that doesn't see the bad plane), not evidence
that the underlying charge picture is actually connected.

**Second, deeper issue** (the actual thing this design fixes): even setting
aside the sub-3cm sample-count blind spot, the per-step "goodness" check
itself, `Facade::Grouping::test_good_point` → `has_closest_point`
(`Facade_Grouping.cxx:585-613, 680-694`), is a **continuous k-d-tree radius
query** — "is there a real hit within 0.6cm of this exact sample point, in
this plane's 2D projection." A 0.6cm radius is forgiving: it can span
several wires and several ticks. That means a step can be marked "good"
because *some* real hit happens to be nearby in that plane's 2D view, even
if that hit belongs to a completely different, unrelated piece of track,
and even if the two candidate components' own real fired-pixel footprints
are not actually adjacent to each other at the discrete wire/tick level
anywhere nearby. This is the same class of problem doc pr/53 round 7's S5
fix (image-ghost-run) was built to catch in 3D — independent per-plane 2D
views each finding *some* nearby charge, without ever checking whether it's
the *same* charge connecting the *same* two components. S5 only measures
"is there any 3D image point within 1cm of a sample" (an OR across all of
this cluster's own components) — it doesn't ask the more specific question
either: are components j and k's own 2D wire/tick footprints connected to
each other.

## 2. What already exists (reused, not reinvented)

Read `connect_graph_relaxed_strict.cxx` in full, plus a codebase-wide search
across `clus/`, `img/`, `sigproc/`, `iface/`, this session:

- **No 2D fired-pixel connected-component utility exists anywhere in the
  codebase.** `img/`'s dead-blob tiling (`CMMModifier.cxx`, `MaskSlice.*`,
  the `dead_area_version` knob) is a Bee-output/tiling concern — it
  geometrically tiles dead channels into "dead blobs" for display,
  independent of live activity, and isn't reusable for an on-the-fly
  per-candidate-pair connectivity query.
- **No "prolonged/isochronous induction-plane signal inefficiency" concept
  exists in code, config, or docs anywhere in this repo.** Every "prolong"
  hit found (`clustering_parallel_prolong.cxx`, `NeutrinoTaggerNuE.cxx
  check_prolong`, `gap_flag_prolong_u/v/w` in `NeutrinoTaggerInfo.h`) is a
  *different* concept: ambiguous 2D projection when a cluster's direction
  runs parallel to a wire, not signal-processing inefficiency from a long
  pulse train. This design's exception (b) is a genuinely new physics rule
  to encode, not a rename of something that already exists.
- **The discrete data this design needs, however, already exists and is
  cheap to query:**
  - `Grouping::is_wire_dead(apa, face, plane, wire_index, time_slice)`
    (`Facade_Grouping.cxx:1030-1068`) — a direct map lookup
    (`cache.dead_wires[plane].find(wire_index)`) plus an x-range check
    against the converted time_slice. O(1)-ish. Already used by
    `is_blob_plane_bad` (`:1137+`) for exactly this kind of per-cell dead
    query.
  - `Grouping::get_wire_charge()` / `wire_charge_row()`
    (`Facade_Grouping.cxx:996-1028`, cache built in `build_wire_cache`,
    `:922-994`) — `std::array<unordered_map<time_slice,
    unordered_map<wind, pair<charge,uncertainty>>>, 3>`, one entry per
    real slice-activity hit, populated straight from `ICluster` slice
    activity independent of 3D blob formation. This is exactly the
    discrete per-(plane, wind, time_slice) "is there live charge here"
    grid a 2D flood-fill needs — and it's an unordered-map lookup, cheaper
    per query than the k-d-tree radius search `test_good_point` currently
    does.
  - `convert_3Dpoint_time_ch(point, apa, face, pind)`
    (`Facade_Grouping.cxx:759-768`) — 3D point → (time_slice, wind); the
    same conversion the current code already relies on indirectly via
    `get_closest_dead_chs`.
  - Wire pitch / tick binning: the `fastgeom_t` cache
    (`Facade_Grouping.h:115-119`), already populated per (apa, face).
  - **The existing `angle1`/`angle2`/`angle1p` computation**
    (`connect_graph_relaxed_strict.cxx:341-357`) — the angle between a
    candidate path and each plane's wire direction, tested against
    `wire_angle_tol` (12.5°) — is *already* the geometric test for "this
    path runs parallel to plane P's wires." That is precisely the
    isochronous/prolonged topology in question. Today this angle test only
    picks which `num_bad[...]` sum feeds the S1/S2 floor
    (`:396-427`). §3.4 below reuses it directly as the trigger for the new
    exception, rather than inventing a second detector for the same
    geometric fact.
- **Cost model of the existing function**, for budgeting the new check: the
  per-pair path-test loop (`:290-333`) runs unconditionally for every one of
  the `O(num_components²)` raw pairs, `dis_cm + 1` `test_good_point` calls
  each — no 80cm cap on this loop (only the Hough-probe block has one,
  `:236`). In the 269774 cluster-13 example this session reconstructed in
  full from log-mined census data, only **15 of 561 raw pairs** survived to
  become real edges. Survivors are a small fraction of all raw pairs in
  practice, because distance and the existing quality test already kill
  most candidates before any new check would need to run.

## 3. Proposed design

**Placement**: an additional, independent gate — S6 — evaluated only for
candidate pairs that already **survive** the existing S1-S3 (+S5 image)
test, in all three of `connect_graph_relaxed_strict.cxx`'s path-check
blocks (closest-pair, dir1, dir2), following the exact pattern S5 already
uses (`:429-446` for the closest-pair block). Killed-by-existing-test
edges stay killed; S6 can only additionally kill, never rescue, an edge.

### 3.1 Per-plane 2D connectivity check

For a candidate pair (j, k) and each plane P in {U, V, W}:

1. **Seed two cell-sets** in P's discrete (wind, time_slice) grid: the
   subset of component j's own real hits, and of component k's own real
   hits, that fall within a local window around the candidate pair's
   closest-approach region — not each component's full extent, which
   bounds cost. Window = bounding box of the two closest-point projections
   plus a small margin (a few wires/ticks), **sized off the 3D gap
   distance**, not a fixed constant — the failure mode this targets (short
   2-3cm gaps) automatically gets small, cheap windows.
2. **Bidirectional BFS/flood-fill** outward from both seed sets
   simultaneously, through cells that are **live** (`wire_charge_row`
   charge above a threshold — see §3.3 for the threshold question) **or
   dead** (`is_wire_dead` true). Folding "dead ⇒ always passable" directly
   into the connectivity predicate is exception (a) from the owner's
   request — dead channels aren't a bolt-on exception, they're just part
   of what "connected" means. Early-exit the moment the two frontiers
   meet. Hard cell-visit budget (e.g. a few thousand cells) as a circuit
   breaker: exceeding it without meeting means "gap" (fails closed,
   conservative — an inconclusive result should not silently pass).
3. Result per plane: **connected**, or **gap**.

This directly implements the owner's steps 2-4: each component's own real
2D footprint, checked for wind/tick adjacency to the other's, rather than
sampling points along an assumed-straight 3D line. It's also strictly more
robust to non-straight true gap geometry than the current line-sampling
approach, since it isn't tied to a straight-line assumption at all — noted
here as the reason a "denser line-sampling" alternative (just shrinking the
1cm step) was considered and rejected: it would still inherit the same
line-assumption fragility, just with a smaller blind spot.

### 3.2 Prolonged-signal exception (U/V only)

If plane P ∈ {U, V} and the **existing** `angle1` (U) or `angle2` (V) is
below `wire_angle_tol` for this candidate pair — the same "path runs
parallel to plane P's wires" condition the current code already computes
at `:341-351` — a **gap** found on plane P by §3.1 is *excused*: it does
not count toward the kill vote in §3.4. Physical justification (owner's
item 5b): a track running near-parallel to an induction wire's direction
produces a long, low-amplitude same-wire pulse train, a known LArTPC
mechanism for real hit-finding inefficiency on induction planes
specifically (not collection/W). Deliberately reusing the already-computed,
already-validated angle test rather than inventing a new detector for the
same geometric fact keeps this exception cheap and consistent with the
rest of the file's branch-selection logic.

### 3.3 Open verification items (do not guess — confirm before coding)

**All three resolved in round 2 (§7.1) — reading left below unchanged as
the record of what was unknown before implementation started.**

- **Exact `scores[]` index-to-plane mapping** used by `test_good_point`
  (`scores[0..2]` vs `scores[3..5]`, and which index pairs with which of
  U/V/W) — needed to keep S6's plane semantics consistent with the
  existing S1-S3 test's plane semantics.
- **Whether `wire_charge_row`'s stored entries are live-only or include
  dead/masked activity.** If it includes masked/uncertain entries, the
  "live" predicate in §3.1 step 2 needs an uncertainty cut, not just a
  charge cut.
- **The literal "fired" charge/uncertainty threshold.** Reuse whatever
  `PointTreeBuilding.cxx`'s `m_dead_threshold` filter already uses to
  build the `ctpc_*` point clouds (`:295-299`, `charge.uncertainty() <=
  m_dead_threshold`) rather than picking a new number — consistency with
  the existing live/dead boundary matters more than tuning a fresh cut.

### 3.4 Kill rule and knob

**Superseded in round 2 — owner's explicit instruction is ≥1, not ≥2 (see
§7.2). Left as originally written below for the record of the design's own
reasoning; §7.2 is authoritative for what shipped.**

**Kill rule (as designed)**: S6 fires (edge invalidated, same
`invalidate_distance()` pattern S1-S3 and S5 already use) if **≥2 of the 3
planes** show a real (non-excused) gap. This matches the "≥2 views agree"
convention already implicit in this file's own combined-quality weighting
(`scores[0]+scores[3]+scores[1]+scores[4]+(scores[2]+scores[5])*2 < 3`
gives W double weight, `:309`) and general LArTPC practice of requiring at
least two independent 2D views to trust a 3D claim.

**Knob**: a new `bool two_d_connectivity_check = false` parameter on
`connect_graph_relaxed_strict`, following the exact `image_check` pattern
already in this file (`connect_graphs.h:64-69`) — byte-identical off,
threaded through the same `graph_name`-selected call sites
(`make_graphs.cxx:108-116`) that already carry `image_check`, per the
repo's default-OFF-knob bar (`CLAUDE.md` §1/§4).

## 4. Performance

- **Scoped to survivors, not all raw pairs.** S6 only runs on candidate
  pairs that already passed S1-S3(+S5) — a small fraction of
  `O(num_components²)` in practice (15/561 in the 269774 cluster-13
  example this session measured end-to-end).
- **Window size scales with gap distance.** The failure mode this design
  targets (short 2-3cm gaps) gets small, cheap BFS windows automatically;
  there's no reason to expect this to be worse-behaved than today's
  `dis_cm + 1`-step line sampling for the same pairs.
- **Hard cell-visit budget** as a circuit breaker bounds worst-case cost
  per pair regardless of true separation, at the cost of failing closed
  (treated as a gap) on the rare pair that would exceed it.
- **Per-cell cost is lower than today's per-step cost.** `is_wire_dead`
  and `wire_charge_row` are `unordered_map`/`map` lookups; the current
  `test_good_point` step does a k-d-tree radius query (`has_closest_point`,
  `Facade_Grouping.cxx:680-694`) up to 3 times per step plus dead-channel
  fallbacks. Replacing "radius query per 1cm sample" with "hash lookup per
  discrete cell, on a small bounded set of cells" should not regress the
  function's wall time in aggregate, though this is a design expectation,
  not yet a profiled number — profiling is an implementation-round item.

## 5. Validation target (stated honestly — not yet proven)

**Now confirmed in round 2 (§7.3) — both edges verified killed on a real
rerun.** Left as originally written below for the record of the design-time
hypothesis this confirms.

`c5→c10` (2.0cm) and `c10→c11` (2.7cm) are this design's concrete target
case: §1 already explains mechanistically why the *old* line-sampling test
structurally cannot catch them (too few interior samples to ever reach the
"≥2 bad" floor, or a branch selection that happens to skip the one bad
plane). Whether the *new* §3.1 wind/tick-adjacency check would actually
find these two components' real footprints disconnected in ≥2 planes is
**not yet checked against real 269774 wire/tick data** — that requires
running code against `wire_charge_row` for this event's actual U/V/W hits
around `(61,-10,109)`-`(64,-9,113)`, which is the first concrete step of
the implementation round, not a claim this design doc makes today. The
plausibility argument (§1, last paragraph) is that the old test's 0.6cm
k-d-tree radius tolerance is far more forgiving than exact discrete-pixel
adjacency — real hits from unrelated nearby structure can satisfy the old
test's "is anything within 0.6cm" question without ever meaning components
j and k's own footprints touch — but this is a hypothesis to verify, not a
result.

## 6. Explicitly out of scope for this doc (design phase)

No code changes to `connect_graph_relaxed_strict.cxx`, `Graphs.h`, or any
jsonnet config. No A/B gate. No default flip. No profiling run. **§7 below
is the next round** — code was written, gated, and measured on the two
named target events; the default flip itself is still not done (owner
review pending).

## 7. Round 2 — implementation record

Scope, per explicit owner instruction, narrower than §3 in three ways:
(1) kill rule is **≥1** non-excused gap, not ≥2; (2) S6 is an
**additional required** check ANDed on top of S1-S5, gated by its own knob
(design already intended this — restated because "additional required"
was the owner's own framing); (3) blast radius confined to **one file**,
`connect_graph_relaxed_strict.cxx` (plus its two thin dispatch/factory
wrappers), and verified against **exactly two named events** — 269774 (the
`c5→c10`/`c10→c11` chain, §1) and 71372 (near-vertex `(-165.9,-155.0,
221.7)`, cluster id 19). No other event, no sample-wide census, no default
flip this round.

### 7.1 §3.3 open items — resolved

- **`scores[]` mapping CONFIRMED**: `test_good_point`
  (`Facade_Grouping.cxx:585-613`) does `num_planes[pind]++` on
  `has_closest_point`, **else** `if (get_closest_dead_chs(...))
  num_planes[pind+3]++` — `0,1,2` = live per plane, `3,4,5` = dead per
  plane, `0=U,1=V,2=W`. Mutually exclusive pairs; the existing caller's
  `scores[k]+scores[k+3]` reading was already correct.
- **`wire_charge_row` is unfiltered but pre-filtered upstream**:
  `build_wire_cache` (`Facade_Grouping.cxx:934-953`) inserts every `ctpc_*`
  row unconditionally; the dead cut already happened in
  `PointTreeBuilding.cxx:295` (`charge.uncertainty() > m_dead_threshold` →
  `continue`, strictly `>`). **Key presence in `wire_charge_row` IS the
  live predicate** — no separate threshold needed, exactly the "reuse, don't
  invent a number" requirement.
- **UNIT TRAP FOUND AND AVOIDED**: `wire_charge_row`'s `time_slice` key is
  `slice->start()/tick`, and slices start at `slicebin * tick_span * tick`
  (`img/src/MaskSlice.cxx:264`) with SBND `tick_span=4`
  (`cfg/pgrapher/experiment/sbnd/img.jsonnet:133`) — **valid keys are
  multiples of 4 ticks, not every tick.** A naive per-tick BFS adjacency
  would have silently missed ~3/4 of real cells and looked like it "worked"
  (every plane reporting a gap) while actually just never finding any live
  cell at all. The implementation **never computes an absolute tick value
  and never assumes stride=1** — see §7.2 seeding route.

### 7.2 What was actually built

**Seeding avoids the unit trap entirely** by never calling
`convert_3Dpoint_time_ch`. Real per-point accessors already carry the same
global point index as the graph's connected-component point clouds:
`Cluster::wire_index(global_idx, plane)` (`Facade_Cluster.h:272`) for the
wire index, and `Cluster::blob_with_point(global_idx)->slice_index_min()`
(`Facade_Cluster.h:253`, `Facade_Blob.h:76`) for the slice index — built by
the *identical* expression as `wire_charge_row`'s key
(`islice->start()/tick`, `aux/src/SamplingHelpers.cxx:90` vs
`PointTreeBuilding.cxx:326`), so same unit and same values by construction,
no conversion, no rounding risk.

Points near the gap are gathered with the existing
`Simple3DPointCloud::get_closest_wcpoints_radius(p, radius)`
(`Facade_Util.h:150`), radius = `max(edge_dis, 1cm) + 2cm` — scales with
the candidate's own 3D gap distance, so the 2-3cm target gaps get small
seed sets automatically. The real slice-adjacency stride is **derived at
runtime**, not assumed: the minimum positive difference between distinct
`slice_index_min` values found among the seeded points (falls back to the
SBND default of 4 only if no two distinct values are seen).

> **Correction (round 3):** the claim that follows in this paragraph in the
> original round-2 text — "every run this round measured `slice_step=4`,
> confirming the derivation lands on the true config value rather than the
> fallback by coincidence" — is **false** and is retracted here rather than
> silently edited away. Round 3 found `slice_step` values of 12, 16, 36, 44,
> 48, 64, 76, 84, 100 and 204 on 11 of 93 S6-evaluated edges in the SAME
> `work-pr56r2-on19/pr_evt71372` run this claim was based on — the "first
> sorted adjacent gap between distinct `slice_index_min` values" rule above
> is not the true stride whenever the seed region spans non-adjacent blobs.
> See §8.1 for the fix and its effect (it does **not** change either of
> round 2's own two verified target-edge kills — see §8.3 — so that specific
> result stands; only this one supporting sentence was wrong).

**The check** (`s6_planes_connected`, anonymous namespace,
`connect_graph_relaxed_strict.cxx`): per plane, a bounded bidirectional BFS
between the two components' seed cell-sets in (wire_index, time_slice)
space, window = seed bbox ± 3 wires / ± 2 slice-steps, cell budget 4000
(fails closed — an exhausted budget counts as a gap, never as connected). A
cell is passable if `wire_charge_row(apa,face,plane,slice)` has `wind` as a
key (live) **or** `is_wire_dead(apa,face,plane,wind,slice)` is true (dead
channel folded directly into the connectivity predicate, not a bolt-on
exception). Adjacency ±1 wire, ±1 slice-step (the derived stride).

**Excusal**: U excused if the caller's already-computed `angle1 <
wire_angle_tol`, V if `angle2 < wire_angle_tol` — the same wire-parallel
test the file already runs for branch selection (`:341-351` in the
original numbering), not re-detected. W is never excused.

**Kill rule (as shipped, owner's instruction)**: `two_d_connectivity_bad`
(`WireCellClus/Graphs.h`, pure, doctested) — `(gap_u && !excuse_u) ||
(gap_v && !excuse_v) || gap_w`. **≥1**, not §3.4's original ≥2.

**Knob**: new graph flavor `relaxed_strict_img_2d` — `bool two_d_check =
false` parameter on `connect_graph_relaxed_strict`, plumbed exactly like
`image_check` (new `make_graph_relaxed_strict_img_2d()` in
`make_graphs.h/.cxx`, registered at the same 4 dispatch sites in
`Facade_Cluster.cxx` that already carry `relaxed_strict_img`). Selecting it
is a cfg-only `-A protect_graph_name=relaxed_strict_img_2d` (env
`SBND_PROTECT_GRAPH=relaxed_strict_img_2d` via the existing runner
plumbing) — **C++ default and jsonnet default both untouched**;
`ClusteringProtectBundle`'s production `graph_name` stays
`relaxed_strict_img`.

**Diagnostics**: new `OC56CENSUS-2D` log line (gap/excuse per plane, slice
step, verdict), env-gated by the same `WCT_RELAXED_EDGE_CENSUS`, same
log-only pattern as the existing `OC53CENSUS-*` lines — no new env var.

Files touched: `clus/src/connect_graph_relaxed_strict.cxx` (the BFS, the
S6 lambda, three call sites in the closest-pair/dir1/dir2 blocks),
`clus/src/connect_graphs.h` (new parameter), `clus/inc/WireCellClus/
Graphs.h` (new pure predicate declaration + doc), `clus/src/make_graphs.h`
/`.cxx` (new flavor factory), `clus/src/Facade_Cluster.cxx` (4 dispatch
sites), `clus/test/doctest_relaxed_strict.cxx` (4 new `TEST_CASE`s for
`two_d_connectivity_bad`, monotonicity included).

### 7.3 Verification

**Freshness proof**: `local/lib/libWireCellClus.so` mtime `2026-08-09
21:25` > last source edit `21:20`. Hit the documented "new-symbol
first-build link gotcha" (stale `local/lib` wins the linker's `-L` search
order over the freshly-built `build/clus` on a brand-new symbol name) —
fixed by copying `build/clus/libWireCellClus.so` into `local/lib` once,
then a clean `wcbuild` succeeded.

**`./build/clus/wcdoctest-clus`**: 145/145 test cases, 1594/1594 assertions
passed (0 failed), including the 4 new `two_d_connectivity_bad` cases.

**Byte-identical OFF gate — PASS 2/2.** Fresh bare reruns
(`work-pr56r2-off19`, `work-pr56r2-off48`, no env override, default
`graph_name` unchanged) vs. the existing production baseline arms
(`work-pr53r7-on19`, `work-pr53r7-on48`):

| event | `mabc-pr.zip` hash (`hash_archive.py`) | `nusel-evt<N>.tsv` |
|---|---|---|
| 71372  | identical (`37faaf63...`) | byte-identical |
| 269774 | identical (`abe0487c...`) | byte-identical |

**S6-on target-edge verification (269774)** — fresh `SBND_PROTECT_GRAPH=
relaxed_strict_img_2d WCT_RELAXED_EDGE_CENSUS=1` rerun
(`work-pr56r2-on48`), replaying the exact three pairs from §1/§5:

```
OC53CENSUS-S closest j=5  k=10 dis=2.00cm nsteps=2 ... killed=true   (was false)
OC53CENSUS-S closest j=5  k=11 dis=5.80cm nsteps=6 ... killed=true   (unchanged, already killed)
OC53CENSUS-S closest j=10 k=11 dis=2.69cm nsteps=3 ... killed=true   (was false)
```

Both target edges now killed. Every one of the 15 closest-pair candidates
that survived S1-S5 for this cluster in the design-phase census is now
killed by S6 (`OC56CENSUS-2D ... killed=true` x15) — a materially more
aggressive outcome than the ≥2-of-3 design in §3.4 would likely have given
(flagged as a real, not hypothetical, consequence of the owner's ≥1 rule,
per the design doc's own §"risks to watch" from the implementation plan —
not something to quietly loosen).

**`ClusteringProtectBundle` split counts, before → after:**

| event | cluster | before (main+frag) | after (main+frag) |
|---|---|---|---|
| 269774 | cid 13 | 521 + 20 = 21 pieces | 339 + 32 = 33 pieces |
| 71372  | cid 19 | 1482 + 3 = 4 pieces  | 1059 + 12 = 13 pieces |

Both target events split materially more than before, in the region the
owner flagged, consistent with the design's intent.

**Cost** (`MABC timing: ClusteringProtectBundle:pr`, single-run wall time,
not a profiled average — noise-level, no regression observed):

| event | before | after |
|---|---|---|
| 269774 | 568.8 ms | 510.3 ms |
| 71372  | 958.5 ms | 885.1 ms |

No wall-time regression on either target event (both runs came in slightly
faster; treated as within run-to-run noise, not claimed as a real speedup
— a single before/after pair on two events is not a performance profile).

**Bee set, S6-on PR output, both events** (owner-requested, outward-facing,
authorized this round):
`https://www.phy.bnl.gov/twister/bee/set/08f06469-c0ce-4e6c-ad26-c0b10e964ec1/event/list/`
— bee index 0 = 71372, index 1 = 269774
(`bee/pr56r2/pr56r2-after.{zip,index.txt,prid-map.txt}`).

### 7.4 Open items / not done this round

- **No sample-wide census.** Only the two named events were run. The ≥1
  kill rule's collateral effect on other events/detectors is unmeasured —
  do not extrapolate the 269774/71372 split-count jump to "typical."
- **No default flip.** `graph_name` stays `relaxed_strict_img` in
  production; `relaxed_strict_img_2d` is opt-in via `-A
  protect_graph_name=relaxed_strict_img_2d` / `SBND_PROTECT_GRAPH` only.
- **No downstream (fitter/tagger/nusel) re-verification beyond the two
  events' own nusel line**, which is identical to the OFF baseline for the
  OFF gate by construction, but the **ON** run's downstream nusel was not
  independently scanned — the ON runs' `nusel-evt*.tsv` exist in
  `work-pr56r2-on{19,48}/` but were not diffed against anything (there is
  no meaningful "before" nusel to diff against, since ON is a new graph
  flavor, not a flip); a hand-scan of the Bee set above is the intended
  next check.
- **Collateral-kill magnitude (all 15 of 15 survivors killed on 269774
  cid 13) is worth the owner's attention before considering a flip** — see
  §7.3. This is reported, not tuned away.

## 8. Round 3 — repair, one owner-reported false positive fixed, sample-wide census

Owner report that triggered this round: near `(x,y,z) = (-164.4,-171.0,
226.1)`, an isochronous track was broken into two clusters by a signal gap
in the U plane, attributed to a large EM shower's prolonged induction
signal (not prolonged path *topology*, so the existing `angle1`/`angle2`
excusal cannot fire for it) — not prolonged-topology inefficiency. Ask: use
the 48 nueCC + 19 NC-pi0 + 50 PR-data samples to scan and tune, sample real
vs. not-real gaps, and above all avoid breaking a long track into two.

### 8.1 Event attribution corrected

The reported coordinates are in event **71372**, not 269774 as first
assumed: 269774 has no reconstructed charge within 110cm of that point;
71372 has charge 0.03cm away (124 points in a 5cm cube), and its S6-killed
edge there matches the report exactly — `j=12 k=13`, `dis=0.76cm`,
`gap_u=true`, `gap_v=gap_w=false`, no excusal. Round-2's Bee set already had
71372 at index 0.

### 8.2 Two round-2 defects found and fixed (repair, before any tuning)

Both manufacture false gaps; tuning on top of them would tune around a bug,
not a physical effect, so both were fixed first and re-verified before any
operating-point change.

1. **The BFS was 4-connected, not 8-connected.** `s6_planes_connected`'s
   stencil was `{(±1,0),(0,±slice_step)}` — von Neumann, not "±1 in both
   wire and time" as assumed. Any non-axis-aligned 2D footprint disconnects
   wherever the corner pixel doesn't fire. Fixed by generalizing the
   stencil to a parametrized Chebyshev box (`dw` wires × `ds` slice-steps,
   excluding the origin); `dw=ds=1` is the true 8-connected repair of round
   2's intended behavior.
2. **`slice_step` was derived by a wrong rule.** Round 2 took the first
   adjacent gap in the sorted-unique list of seeded points' own
   `slice_index_min` values — not the true stride whenever the seed region
   spans non-adjacent blobs. Measured values of 12, 16, 36, 44, 48, 64, 76,
   84, 100, 204 on 11 of 93 S6-evaluated edges in evt 71372 (all should have
   been 4) — this directly falsifies round 2's own §7.2 claim that "every
   run this round measured `slice_step=4`" (corrected in place there, this
   round). Fixed: a blob's own `slice_index_max()-slice_index_min()` **is**
   `span/tick`, i.e. the true `tick_span`, by construction
   (`aux/src/SamplingHelpers.cxx:90-91`) — no cross-blob inference needed at
   all. `slice_step` is now the minimum such span found among the seeded
   blobs on either side, fallback 4 only if none is positive.

**Repair-only re-verification** (both bugs fixed, adjacency still at the
default `(1,1)`, no distance floor yet): re-ran both named events.
269774's census is **byte-identical to round 2** — all 15 edges, all 15
verdicts unchanged, including both named targets (`j=5 k=10` 2.00cm,
`j=10 k=11` 2.69cm, both still `killed=true`). So round 2's target-edge
result stands untouched by the repair. 71372 changed on 6 of 93 edges, one
of them the reported case: `j=12 k=13` (0.76cm) flips from `killed=true`
(round 2, spurious `gap_u` from the 4-connected stencil) to `killed=false`
— the **repair alone already rescues the reported case**, with no further
relaxation. The other 5 flipped edges (4.5-11.7cm) all show `gap_u=gap_v=
gap_w=false` under the repair, i.e. the repaired 8-connected BFS finds no
gap on any plane at all — consistent with round 2's 4-connectivity bug
being the cause, not a second real effect. Cluster 19's fragment count
improves from round 2's buggy 13 pieces to a repaired 11 (main-retained
blob count unchanged at 1059).

### 8.3 A third bug found in the diagnostics, fixed at the source

The first cut of round 3's enriched census split each edge's per-plane data
across separate `OC56CENSUS-2D plane` lines correlated with the edge's
`OC56CENSUS-2D edge` line by matching `(blk,j,k)` across sequential log
lines. Component indices `j,k` restart per cluster, so this key collides
across different clusters processed within the same event — and
`ClusteringProtectBundle` can process clusters concurrently, interleaving
their log lines. Six of 67 logs in an early 48+19-sample run showed
unpaired plane-line groups, proving the correlation broke somewhere.
**Fixed at the source, not by patching the parser**: every field (including
a `dw={1..4} × ds={1..4}` connectivity matrix per plane, so any candidate
operating point is evaluable offline from one run) now lives on a single
self-contained `OC56CENSUS-2D edge` line — nothing left to correlate. This
is a diagnostics-only fix; it does not touch `two_d_gap_kill`'s actual
kill decision, which was never affected (each call computes and returns its
own verdict atomically; only the *offline log-parsing reconstruction* of
that verdict was at risk). New script: `scripts/analysis/pr56/s6_scan.py`.

### 8.4 Sample-wide census: the repair alone is not enough

A census across all 117 events (`WCT_RELAXED_EDGE_CENSUS=1
SBND_PROTECT_GRAPH=relaxed_strict_img_2d`, repaired baseline, no floor)
found 2089 S6-evaluated edges across 88 events, **74.2% killed overall**,
53-89% depending on 3D-gap-distance bucket (53% at 0-1cm, rising to 89% at
15-30cm). The aggregate rate is not by itself the metric that matters — S6
only evaluates candidates that already survived S1-S5, a small, already
curated pool — so the decisive check is the one the owner's own concern
maps onto directly: **does turning S6 on break more real tracks than
production does**, measured as `ClusteringProtectBundle`'s own `retained N
+ M fragment(s)` line, diffed per cluster between the bare (OFF) arm and
the S6-on (repaired-baseline) arm, across every cluster large enough for
`ClusteringProtectBundle` to log in all 117 events.

**Result: 51 of 53 comparable clusters get MORE fragmented at the repaired
baseline than production; 0 get fewer.** This is not confined to the two
named events — e.g. evt 122660 cid 9 goes from 3 pieces (production) to 19;
evt 239794 cid 2 from 5 to 18; evt 433451 cid 4 from 4 to 16. The bug repair
fixes what it claims to fix (the two named controls, see §8.2) but does
**not**, by itself, avoid the owner's stated "long track broken into two"
failure mode at sample scale — it was never designed to; §8.2 only closed
two implementation bugs, it did not address the operating point's
aggressiveness.

### 8.5 Adjacency widening: tried, and rejected — it breaks a kill target

The natural next step, widening the per-plane BFS stencil beyond `(1,1)`
(the owner's own suggestion, "U and V plane should have slightly relaxed 2D
connectivity"), was implemented and tested at `(dw,ds)=(2,2)` on U/V (W left
at `(1,1)` — W is the collection plane, has the least real signal
inefficiency, and is never excused by design; a check confirmed that
relaxing W's adjacency too erases most of its near-touching discriminating
power entirely, e.g. `(4,4)` on all three planes drops the 0-1cm bucket's
kill rate to 0/184 — treated as proof W should stay strict, not as a
tuning candidate).

**`(2,2)` on U/V rescues the 71372 target (0.76cm) — and also rescues
269774's `j=10 k=11` control (2.69cm), which must stay killed.** Both are
U-plane-only unexcused gaps (269774's V gap is present but excused by the
angle test); pure wire/tick adjacency widening cannot distinguish them,
since it responds to proximity in the 2D grid, not to 3D distance, and both
edges' U-plane gaps happen to close at the same stencil size. Verified with
a real pipeline rerun, not just the offline matrix estimate — this is
exactly the binding-constraint tension identified before implementation
began, and exactly the case the owner pre-approved a fallback for:

> "If widening U adjacency alone can't separate them, use a 3D-gap floor."

Adjacency widening on U/V is **not shipped**. All three planes remain at
the repaired `(1,1)` baseline.

### 8.6 Final operating point: a distance floor only

`s6_dis_floor = 1.0cm`: below this 3D gap, S6 abstains outright (same
"no evidence, no verdict" posture as the existing APA/face and empty-seed
abstentions). A floor only ever *removes* kills strictly below it; it
cannot change any verdict at or above it, so it is structurally incapable
of re-breaking a kill target the way adjacency widening did. Chosen with
margin on both sides of the two controls it has to separate (0.76cm
rescued, 2.00cm and 2.69cm both untouched); 1.0-1.9cm all give the same
separation, 1.0cm was kept as the simplest round number with the most
margin toward the rescue side.

**Controls, verified with real pipeline reruns (not the offline matrix
estimate) at the final operating point** (repaired `(1,1)` adjacency on all
three planes, `s6_dis_floor=1.0cm`):

| control | dis | requirement | result |
|---|---|---|---|
| 71372 `j=12 k=13` (closest) | 0.76cm | must survive | **survives** (floor-abstained; no census line emitted for floor-abstained edges — see caveat below) |
| 269774 `j=5 k=10` (closest) | 2.00cm | must stay killed | **killed=true** |
| 269774 `j=10 k=11` (closest) | 2.69cm | must stay killed | **killed=true** |
| 71372 near-vertex set (1.1-2.7cm, from doc pr/56 round 1) | — | must stay killed | all confirmed still killed (none of them are S6's own near-touching-only class besides the one named target) |

Caveat: the floor's early return happens before the `oc53_census` log
statement, so a floor-abstained edge emits **no** `OC56CENSUS-2D` line at
all. The rescue was confirmed instead via the resulting
`ClusteringProtectBundle` fragment count for cluster 19 (below), and by the
absence of any `OC56CENSUS-2D edge blk=closest j=12 k=13` line in the final
rerun where round 2 and the repair-only rerun both showed one.

**Fragment counts, final operating point, both named events:**

| event | cluster | production (OFF) | S6-on, repaired baseline (no floor) | S6-on, final (repair + floor 1.0cm) |
|---|---|---|---|---|
| 269774 | cid 13 | 521 + 20 = 21 pieces | 339 + 32 = 33 pieces | 339 + 30 = 32 pieces |
| 71372  | cid 19 | 1482 + 3 = 4 pieces  | 1059 + 10 = 11 pieces | 1067 + 9 = 10 pieces |

The floor rescues exactly one fragment on 71372 (11→10, +8 blobs reabsorbed
into the main retained count) and two on 269774 (32→30); it does not touch
269774's main-retained blob count at all (339 both with and without the
floor) — expected, since neither of 269774's own gaps is below 1cm.

### 8.7 The floor is a narrow, correct fix — not a sample-wide solution

Re-running the full 117-event fragment-count diff at the final operating
point (repair + floor 1.0cm): **49 of 57 comparable clusters still get more
fragmented than production; 0 get fewer** — essentially unchanged from
§8.4's repair-only 51/53. The floor removes exactly the sub-1cm class of
false positive it was built for (confirmed on the two named events above)
but the bulk of the sample-wide over-fragmentation comes from S6 kills at
1-30cm, a range the floor does not touch and adjacency widening cannot
touch without re-breaking a control (§8.5).

**This is reported as the honest result, not tuned away further.** The
owner's ask was to fix the specific reported case and scope-check it
against the sample — both are done, and both point the same direction: S6,
even after the round-3 repair and floor, remains far more aggressive than
production if it were ever the default. That is a property of the current
per-plane-gap-vote design (any S1-S5 survivor showing a real but small,
possibly-3D-ambiguous single-plane discontinuity gets killed, and such
candidates are common across the whole 1-30cm range, not rare), not
something this round's scope (owner: "no need to worry other cases yet")
asked to redesign. A charge-context-aware excusal (distinguish "real gap"
from "gap embedded in dense nearby charge, like a large EM shower" more
directly than an angle proxy) was offered as an alternative fallback and
not chosen this round; it remains the natural next design if the owner
wants S6 broadened beyond the two named cases.

### 8.8 Verification

**Freshness proof**: `local/lib/libWireCellClus.so` mtime `2026-08-09
22:22:36` > last source edit `22:22:14`, for the final build.

**`./build/clus/wcdoctest-clus`**: 145/145 test cases, 1594/1594 assertions
passed, both after the repair and after the final floor-only build.

**Byte-identical OFF gate — PASS 0/117** across all three samples (default
`graph_name` unchanged, so this must be, and is, a clean pass):

| sample | events | archive-level diffs | nusel-events.tsv diffs | nusel-table.tsv diffs |
|---|---|---|---|---|
| 48 nueCC | 48 | 0/48 | 0/48 | 0/48 |
| 19 NC-pi0 | 19 | 0/19 | 0/19 | 0/19 |
| 50 PR-data | 50 | 0/50 | 0/50 | 0/50 |

Compared against the same production baseline arms round 2 used
(`work-pr53r7-on{48,19,50}`), via `hash_archive.py` member-content hashing
(never raw `cmp`/`md5sum` on archives) plus `nusel-evt*.tsv`.

**Cost**, non-census production-representative timing (`MABC timing:
ClusteringProtectBundle:pr`, single-run, not a profiled average):

| event | production (OFF) | S6-on, final (repair + floor) |
|---|---|---|
| 269774 | 506.4 ms | 485.1 ms |
| 71372  | 1015.7 ms | 845.7 ms |

No regression on either named event (both slightly faster under S6-on,
consistent with round 2's observation that killing more candidate edges
reduces downstream MST/steiner work — not claimed as a real speedup, a
single before/after pair is not a profile). Census-mode runs (used for all
the diagnostic numbers above) are slower due to the diagnostic 16-point
`dw×ds` matrix scan per plane per edge; that overhead is never present when
`WCT_RELAXED_EDGE_CENSUS` is unset, which is every production run.

**Bee set, final S6-on output** (owner-requested judgment cases: both named
targets plus two representative sample-wide fragmentation examples from
§8.4, so the owner can independently judge whether those are real
gaps correctly found or over-aggressive kills):
`https://www.phy.bnl.gov/twister/bee/set/e4bc3ea0-396c-49bc-821e-5e1f04993fb3/event/list/`
— index 0 = 71372, 1 = 269774, 2 = 122660 (cid 9, 3→15 pieces), 3 = 214469
(cid 16, 2→12 pieces). (`bee/pr56r3/pr56r3-after.{zip,index.txt,
prid-map.txt}`.)

### 8.9 Files touched this round

`clus/src/connect_graph_relaxed_strict.cxx` only (stencil generalization,
`slice_step` fix, single-line census redesign, `s6_dis_floor`, the
`dw={1..4}×ds={1..4}` matrix diagnostic) — same one-file blast radius as
round 2. New: `scripts/analysis/pr56/s6_scan.py` (offline operating-point
scan/candidate evaluator against the census matrix). This doc.

### 8.10 Open items / not done this round

- **No default flip.** `graph_name` stays `relaxed_strict_img` in
  production.
- **Sample-wide over-fragmentation at the current operating point is real
  and unresolved** (§8.7) — flagged for owner attention, not silently
  tuned around. If broader relaxation is wanted, a charge-context-aware
  excusal (§8.7) is the recommended next design, not further adjacency or
  floor tuning on the current per-plane-gap-vote mechanism.
- **No downstream (fitter/tagger/nusel) hand-scan of the final ON arms**
  beyond the byte-identical OFF gate and the four named controls — the Bee
  set above is the intended next check, same posture as round 2.
- **The census diagnostic cannot show floor-abstained edges** (§8.6
  caveat) — confirmed via the resulting fragment count instead; a future
  round could log floor-abstentions explicitly if that gap matters for a
  later study.

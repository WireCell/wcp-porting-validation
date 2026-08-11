# doc pr/59 — shower_track_global hole at cluster 7 seg 20 (run 18255-142421): a segment created after its cluster's only association pass, orphaned by a silent main-cluster swap

Status: **root cause confirmed by instrumented rerun. Log-only diagnostic
sentinels added, byte-identical proven (§4). No fix, no new knob shipped this
round** — owner chose "diagnose first" explicitly; the rescue fix is a
follow-up (doc pr/60 or later), designed against the mechanism below instead
of the earlier inference. Prototype comparison (§6, M15 check) confirms this
is a genuine defect shared by both trees, not a toolkit divergence — the
Round-2 fix needs no owner "which reading" call.

**Headline finding**: segment graph-index 20 in cluster 7 (encoded `7020`,
109 fitted points, `(155.8,-39.5,250.5) -> (117.9,-69.1,209.9)` cm — the
64.9 cm track running straight through the owner's reported gap) is **never
once passed into `clustering_points_segments`** for the entire event. It is
created by one of `separate_track_shower` / `determine_direction` /
`shower_determining_in_main_cluster` / `determine_main_vertex`, all of which
run strictly *after* the only `clustering_points` call that ever touches
cluster 7 (`TaggerCheckNeutrino.cxx:1012`, before `find_proto_vertex`'s later
effects have produced segment 20). The *second* `clustering_points` call
(`:1202`, meant to re-associate the main cluster after `improve_vertex` /
shower clustering) does not run on cluster 7 at all — `determine_overall_
main_vertex_DL`'s rerank silently repointed `main_cluster` from 7 to 106 in
between (`swap_main_cluster`, `NeutrinoPatternBase.cxx:2963`, no log line at
any call site until this round). Cluster 7 is left as a permanent
"other_cluster" with a stale, first-pass-only `associate_points` state;
segment 20, created after that first pass, never gets one at all.

This is **not display-only**. `NeutrinoEnergyReco.cxx:289-293` falls back to
the sparse `fit` point cloud when `associate_points` is null, so cluster-7 seg
20's reconstructed charge/energy is currently computed from its ~109 polyline
points instead of the true charge cloud that should feed it.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# off-gate (sentinels present, WCT_PR59_ASSOC_CENSUS unset -- byte-identical proof):
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue PR_OC56_SCAN_DUMP=1 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr59-gate1 data 142421

# knob-on smoke run (sentinels emit):
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue PR_OC56_SCAN_DUMP=1 \
  WCT_PR59_ASSOC_CENSUS=1 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr59-gate2 data 142421

# per-segment creation backtrace (pre-existing WCT_DET_DEBUG=2 facility,
# PRGraph.cxx:20-24/193-212 -- not new this round, stderr-only fprintf,
# no behavior/log-file effect; used for the "why segment 20 specifically"
# analysis in sec 3.1):
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue PR_OC56_SCAN_DUMP=1 \
  WCT_PR59_ASSOC_CENSUS=1 WCT_DET_DEBUG=2 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr59-gate3 data 142421
grep -A6 "WCT_DETA seg idx=20 " work-pr59-gate3/pr_evt142421/stdout.log

# the swap, named directly:
grep "pr59 assoc-census" work-pr59-gate2/pr_evt142421/wct_pr_evt142421.log

# confirm segment 20 never entered clustering_points_segments at all:
grep "pr59 assoc-census stage.*segment 20 fits_size" \
  work-pr59-gate2/pr_evt142421/wct_pr_evt142421.log   # empty

# byte-identical check (member-content hash, not raw md5 -- M2):
python3 ../../abtest/hash_archive.py work-pr57r6-scan19/pr_evt142421/mabc-pr.zip
python3 ../../abtest/hash_archive.py work-pr59-gate1/pr_evt142421/mabc-pr.zip
python3 ../../abtest/hash_archive.py work-pr59-gate2/pr_evt142421/mabc-pr.zip
```

## 1. Symptom

Owner hand-scanned Bee set `22ca051a-d032-4b2b-9454-69a7bf85630e` (event index
4) and reported: "I do not understand why there is no shower_track_global
points around (x, y, z) = (146.7, -46.1, 239.0) ... there are clearly imaging
points, somehow just these points were not clustered to track."

Bee set index 4 was byte-proven (member-hash match on all 7 zip entries) to
be `work-pr57r6-scan19/pr_evt142421/mabc-pr.zip`, run 18255 evt 142421, the
production round-6 `relaxed_strict_img_2d_rescue` arm (owner-flipped SBND
default as of this same day, doc pr/57).

Measured on that zip, both layers in the same post-QL-corrected frame (doc
pr/13 — `img`/`clustering-global` and `shower_track-global` are NOT in
different coordinate frames, so a same-frame distance query is valid):

| layer | points within 10 cm of (146.7,-46.1,239.0) | nearest point |
|---|---|---|
| `clustering-global` (= `img`, corrected frame) | 209, all `cluster_id==7` | 0.06 cm |
| `shower_track-global` | 0 (also 0 within 20 cm) | 13.0 cm |
| `track_fit-global` | 34 (segment `real_cluster_id=7020`) | 1.35 cm |

The one and only `pr55 shower_track layer` sentinel (doc pr/55, existing,
`MultiAlgBlobClustering.cxx:887`) in the entire run log fires for exactly this
segment:

```
pr55 shower_track layer: segment 7020 (cluster 7) has no associate_points
dpcloud -- contributes 0 points to 'shower_track'
```

## 2. Ruled out

**Not a round-6 (`relaxed_strict_img_2d_rescue`) regression.** The identical
symptom (same missing segment 20, same sentinel line) reproduces byte-for-byte
across five independent arms that differ only in `protect_bundle`'s graph
flavor:

| arm | flavor | seg-20 hole? |
|---|---|---|
| `work-pr57r6-scan19` (today's production) | `relaxed_strict_img_2d_rescue` | yes |
| `work-pr57r4-scan19` | `relaxed_strict_img_2d_wfloor` | yes |
| `work-pr58-scan19` | `relaxed_strict_img_2d_wfloor` | yes |
| `work-pr56r4b-off19` | pr/56-era, S6 off | yes |
| `work-pr56r3-off19` | pr/56-era, S6 off | yes |

`protect_bundle`'s graph flavor governs blob-to-blob *merging* before PR runs;
this gap is entirely inside PR's own association pass, downstream of and
unrelated to which flavor built the cluster. No revert is implicated.

**Not a coordinate-frame or `real_cluster_id`-numbering artifact.** Confirmed
early that `track_fit`'s `real_cluster_id` encodes `cluster_id*1000 +
segment->get_graph_index()` while `shower_track`'s encodes `cluster_id*1000 +
id()` (start segment's, if shower-absorbed) — different numbering schemes
naively cross-referenced would manufacture false "missing segment" lists. All
measurements above use the geometric point-cloud comparison instead (distance
queries, not id matching), so this asymmetry does not affect the conclusion —
flagged here only as a gotcha for future readers replaying this analysis.

## 3. Root cause (instrumented, `WCT_PR59_ASSOC_CENSUS=1`)

`TaggerCheckNeutrino.cxx` calls `pattern_algos.clustering_points(*pr_graph,
*main_cluster, m_dv)` (→ `PatternAlgorithms::clustering_points` →
`clustering_points_segments`, which paints image points onto each segment's
`associate_points` cloud) **exactly twice**:

- `:1012` — right after `find_proto_vertex`, before shower/vertex work.
- `:1202` — after `improve_vertex` / `main_vertex_graph_audit`, right before
  `shower_clustering_with_nv`.

New sentinel output on the smoke run (`WCT_PR59_ASSOC_CENSUS=1`):

```
[20:21:03.004] pr59 assoc-census: first clustering_points call, main_cluster=7
[20:21:05.425] pr59 assoc-census: swap_main_cluster 7 -> 106
[20:21:05.481] pr59 assoc-census: second clustering_points call, main_cluster=106
```

Between the two calls, `determine_overall_main_vertex_DL` (DL vertex path,
on for SBND production) holds `Cluster*& main_cluster` **by reference** and
called `swap_main_cluster` (`NeutrinoPatternBase.cxx:2963`) when its rerank
preferred cluster 106's vertex. `swap_main_cluster` had **no log line at any
call site** before this round (confirmed: grepping the pre-existing log for
`swap`, `mvsa`, or `rerank selected` on this event returns nothing at DEBUG
level) — the swap was completely invisible without the new sentinel.

The new per-segment Stage-A/Stage-C sentinels inside
`clustering_points_segments` (`PRSegmentFunctions.cxx`) confirm the
consequence directly: cluster 7 receives **exactly 10** Stage-A lines
(segments `4, 9, 10, 11, 12, 15, 16, 17, 18, 19`) — all from the first call,
none after the swap. Segment **20 never appears in any Stage-A or Stage-C
line for the whole event**, under any cluster id:

```
$ grep "pr59 assoc-census stage.*segment 20 fits_size" wct_pr_evt142421.log
(empty)
```

### 3.1 Why segment 20 specifically, and not its siblings (`WCT_DET_DEBUG=2`)

Cluster 7's final segments split into two populations by *how* they were
created, not merely *when*:

**Pre-existing, correctly associated**: segments `9, 10, 11, 16, 17, 18, 19`
(plus `4, 12, 15`, which later merge away and are not in the final graph)
were all created by `find_proto_vertex` — confirmed via
`WCT_DET_DEBUG=2`'s per-segment-creation backtrace+coordinate dump
(`PRGraph.cxx:193-212`, `boost::add_edge` succeeds) — **before**
`clustering_points:1012` ran, so they were in that call's `segs` list and
won real points (Stage-A `terminals_seeded` 6-356 each, all nonzero).

**Segment 20** is created *after* `:1012`, but by a specific mechanism that
matters: the `WCT_DETA` backtrace for graph-index 20 is

```
WCT_DETA seg idx=20 nw=123 v1=(1176.7,-694.3,2097.8) v2=(1558.1,-394.7,2505.8) ...
  bt[1] PatternAlgorithms::examine_vertices_1
  bt[2] PatternAlgorithms::examine_vertices
  bt[3] PatternAlgorithms::improve_vertex
  bt[4] PatternAlgorithms::determine_main_vertex
  bt[5] TaggerCheckNeutrino::visit
```

i.e. it is built *inside* the first round's `determine_main_vertex` call
(`:1026`), by `examine_vertices_1`'s internal `improve_vertex` sub-step. That
function (`NeutrinoStructureExaminer.cxx:1414-1521`) is a vertex-cleanup
pass: when a degree-2 vertex sits between two short (<4 cm) segments that
represent the same physical point, it **deletes both old segments and that
vertex**, then builds one brand-new replacement via a fresh Steiner-graph
search (`do_rough_path`) and `create_segment_for_cluster` +
`add_segment` — a segment with **zero inherited history**, `associate_points`
null by construction, same as any freshly `add_segment`-ed edge. Since the
second `clustering_points` call never reaches cluster 7 (swapped to 106
first), segment 20 never gets a chance to compete for points at all. Zero
Stage-C "won zero points" lines fired anywhere in the event
(`grep -c "stageC" == 0`), which rules out the 2D ghost-removal cascade
(`PRSegmentFunctions.cxx:2991-3031`, a live gap-source in general, §5) for
this segment specifically — it never entered the competition, one level
simpler than losing it.

**Segments 111 and 112** are created even later — after the swap, inside
`shower_clustering_with_nv`'s call to `break_segment` — and are NOT missing
points (9600 cluster-7 points in `shower_track-global` total, no null/empty
sentinel for either). Their `WCT_DETA` backtrace shows the difference:
`break_segment` (`PRGraph.cxx` via `make_segment`, called from
`shower_clustering_with_nv_from_vertices`) takes an **already-associated**
parent segment and splits its *existing* `associate_points` cloud in half by
nearest-point (`PRSegmentFunctions.cxx:1087-1153`, lossless redistribution,
no fresh competition needed) — so both children inherit real charge points
regardless of how late they're born.

**The rule, stated plainly**: a segment created by `add_segment`-ing a
brand-new polyline (a merge/re-route, as `examine_vertices_1` does) after the
cluster's last association pass is orphaned; a segment created by
`break_segment` splitting an already-associated parent is not, because it
inherits rather than competes. Segment 20 is the *only* segment in this event
born the first way, in the narrow window between the one association pass
and the swap — a coincidence of how many such re-routes happened to fire in
that window, not a property of "graph-index 20" itself. A different event
could just as easily orphan two or three such segments, or none.

## 4. Why it hid

- `swap_main_cluster` logs nothing, at any of its three call sites
  (`determine_overall_main_vertex_DL`'s rerank, `check_switch_main_cluster`,
  `check_switch_main_cluster_2`), despite the DL-path comment at
  `TaggerCheckNeutrino.cxx:1104-1113` explicitly noting the *traditional*
  path's swap is made visible via an `mvsa:` log line — the DL path (the one
  actually taken here, `dl=on` in production) has no equivalent.
- The only downstream symptom is the pr/55 `shower_track layer` sentinel,
  which fires at Bee-dump time, in a completely different function
  (`MultiAlgBlobClustering.cxx`), long after the swap and long after
  `clustering_points_segments` has already run and returned. Nothing at the
  point of loss (Stage A/B/C inside `clustering_points_segments`) or at the
  point of the swap itself said anything.
- `track_fit`'s `real_cluster_id` numbering (`get_graph_index()`-based) and
  `shower_track`'s (`id()`-based, shower-collapsed) differ, so a naive
  cross-layer segment-id diff (as this investigation's first pass did) reports
  false positives/negatives and has to be redone as a geometric point-cloud
  comparison (§2) to trust.

## 5. Fix — not shipped this round

Owner-selected shape for the follow-up round (not implemented here): an
**additive, default-OFF knob** — after the PR stages complete, identify any
segment in the final PR graph with a non-empty `fits()` but a null/absent
`associate_points` cloud, and run one association pass for it; adopt the
result **only** into segments that were previously null. Every
already-associated segment (including cluster 106's, and cluster 7's
first-pass segments 4/9/10/11/12/15/16/17/18/19) stays byte-identical. This
needs its own gate and an owner flip, since — per `NeutrinoEnergyReco.cxx`
§0 above — turning it on will move reconstructed charge/energy for affected
segments, not just Bee display.

Two mechanisms this fix must NOT silently also touch (documented so a future
implementer doesn't reach for the wrong lever):
- `swap_main_cluster`'s behavior itself is not being questioned here — the
  swap may be entirely correct (106's vertex may genuinely be the better
  neutrino vertex); only its *silence*, and its side effect of stranding the
  demoted cluster's post-swap segments, is the gap.
- Segments born via `break_segment` (e.g. cluster 7's 111/112, §3.1) already
  carry a real, correctly-inherited `associate_points` cloud — a Round-2 fix
  that indiscriminately re-associates "every segment created after the last
  pass" rather than specifically "every segment with a null/absent cloud"
  would re-compete for and potentially reshuffle points that are already
  correct, not just fill genuine gaps.
- The 2D ghost-removal cascade (`PRSegmentFunctions.cxx:2991-3031`) is a
  second, independent, pre-existing way a point can be silently dropped
  (Voronoi-cell owner loses the 2-plane 2D-nearest contest and no other
  segment reclaims it) — it did not fire for segment 20 in this event (§3),
  but any Round-2 fix that re-runs association broadly, rather than only for
  null-cloud segments, would also re-expose that class and needs its own
  scoping decision.

## 6. Prototype comparison (M15 check — not a toolkit-introduced defect)

CLAUDE.md M15: before treating a toolkit behavior as a bug, confirm it isn't
an intentional prototype divergence, checking `porting_dictionary.md` /
`neutrino_id_function_map.md` first. Neither documents this pattern — but
the check below establishes something stronger than "unlisted, surface it":
there is **no divergence to reconcile at all**. The prototype
(`prototype_base/pid/src/`) has the identical structural gap, in all three
places that matter.

**`swap_main_cluster` is a byte-for-byte match**, `NeutrinoID.cxx:735-740`:

```cpp
void WCPPID::NeutrinoID::swap_main_cluster(WCPPID::PR3DCluster *new_main_cluster){
  other_clusters.push_back(main_cluster);
  main_cluster = new_main_cluster;
  auto it1 = find(other_clusters.begin(), other_clusters.end(), main_cluster);
  other_clusters.erase(it1);
}
```

Same demote-and-erase, no re-association call, no log line.
`neutrino_id_function_map.md:164` records the port as `explicit args` only —
no behavior difference noted, and none exists: the toolkit's `main_cluster`
(a by-reference local threaded through the call chain) plays exactly the
role of the prototype's `main_cluster` (a class member `swap_main_cluster`
reassigns directly).

**`clustering_points(main)` runs on exactly the same two occasions.**
Exhaustively grepping every `clustering_points(` call site across the whole
prototype `pid/src/` tree finds four total, matching the toolkit's
two-for-main/two-for-others structure — no third or final catch-all pass
anywhere in the reconstruction chain. The toolkit's own call-order
comparison table, `neutrino_id_function_map.md:520-542`, documents both:
line 522 before shower/vertex work, line 542 "again" after `improve_vertex`
— with `determine_overall_main_vertex_DL()` (the swap-capable call) sitting
between both times, same as the toolkit.

**`examine_vertices_1`'s new-segment path is also identical**,
`NeutrinoID_proto_vertex.h:2966-2995`:

```cpp
temp_cluster->dijkstra_shortest_paths(v3->get_wcpt(),2);
temp_cluster->cal_shortest_path(v2->get_wcpt(),2);
ProtoSegment *sg2 = new WCPPID::ProtoSegment(acc_segment_id, temp_cluster->get_path_wcps(), temp_cluster->get_cluster_id()); acc_segment_id++;
...
add_proto_connection(v2, sg2, temp_cluster);
add_proto_connection(v3, sg2, temp_cluster);
del_proto_vertex(v1);
del_proto_segment(sg);
del_proto_segment(sg1);
temp_cluster->do_multi_tracking(map_vertex_segments, map_segment_vertices, *ct_point_cloud, global_wc_map, flash_time*units::microsecond, true, true, true);
```

Deletes the two short segments, builds the replacement via a fresh
Dijkstra/Steiner path (the direct analog of `do_rough_path`), refits
(`do_multi_tracking`) — and never touches point-cloud association, matching
`examine_vertices_1`'s `add_segment`-a-brand-new-polyline behavior in §3.1
exactly. And `determine_main_vertex` (`NeutrinoID_track_shower.h:1286`)
calls `improve_vertex(temp_cluster, false)` internally, in the first round —
matching the toolkit's `determine_main_vertex → improve_vertex →
examine_vertices → examine_vertices_1` chain that built segment 20 (§3.1) —
so the prototype can create this exact class of orphan-prone segment
*before* any swap has even happened, the same way segment 20 was.

**Conclusion**: this is a genuine, previously-undocumented defect in the
original WCP prototype algorithm, faithfully inherited by the port — not a
toolkit regression, and not a case where a fix would fight an intentional
prototype convention. Per the doc pr/54 precedent (a toolkit-only extension
of an unfinished/buggy prototype behavior, not a parity fix), the Round-2
fix in §5 is free to diverge from the prototype's behavior here without an
owner "which reading do you want" call under M15 — there is only one
reading, and both trees share it.

## 7. Verification (this round: sentinels only, log-only)

- `wcdoctest-clus`: **152/152 test cases, 1614/1614 assertions PASS**
  (`./build/clus/wcdoctest-clus`, post-`wcbuild`).
- Freshness proof: `libWireCellClus.so` mtime `20:18:57` after `wcbuild`, all
  three edited sources (`PRSegmentFunctions.cxx` `20:16:44`,
  `TaggerCheckNeutrino.cxx` `20:17:51`, `NeutrinoPatternBase.cxx` `20:18:11`)
  strictly older.
- **Byte-identical gate, off (env unset)**: `work-pr59-gate1/pr_evt142421`
  vs. the pre-existing production `work-pr57r6-scan19/pr_evt142421` —
  `hash_archive.py` member-content hash
  `3312574e108ea29f7b34824d10ddfa13e80502ee825630a8b5bb18e3a3ddf75e` on both;
  `nusel-evt142421.tsv` and `oc56scan-evt142421.jsonl` (round-4-era
  per-edge S6 dump) `diff` clean.
- **Byte-identical gate, on (`WCT_PR59_ASSOC_CENSUS=1`)**: `work-pr59-gate2`
  vs. `work-pr59-gate1` — same `hash_archive.py` hash
  (`3312574e...`) on both; the new sentinels are pure `SPDLOG_LOGGER_DEBUG`
  additions with no control-flow effect, confirmed empirically, not just by
  code inspection.
- **Knob-on smoke run**: `work-pr59-gate2`, 83 `pr59 assoc-census` lines
  emitted, including the swap (`7 -> 106`) and the two `clustering_points`
  call markers (`main_cluster=7` then `main_cluster=106`) — this doc's §3 is
  built directly from this run's log, not inference.
- No iterated pointer-keyed containers introduced (all new maps keyed by
  `SegmentIndexCmp`/cluster id, matching existing file convention).

## Gotchas carried forward

- `track_fit` vs `shower_track` `real_cluster_id` numbering asymmetry (§2) —
  always verify by geometric point-cloud distance, not id cross-reference.
- Bee-zip `cluster_id` on PR-stage layers is NOT remapped (unlike the Bee
  *upload* copy's `make_pr_bee.py` renumbering, doc pr/55) — read
  `pr_evt<N>/mabc-pr.zip` directly for segment-keyed analysis, as done here.
- `swap_main_cluster` fires silently on at least the DL-rerank path; anyone
  debugging a "segment X exists in track_fit but nowhere else" symptom should
  check `WCT_PR59_ASSOC_CENSUS=1` first before assuming a ghost-removal
  (Stage-B) cause.

Related: [[project_pr55_fit_vs_image]] (the existing `shower_track layer`
sentinel that named the symptom but not the mechanism),
[[project_pr54_isolated_residual_keep]] (a different, unrelated way a
segment's points/existence can be silently dropped in the same general
neighborhood of code — ruled out here, `pr54 isolated-residual drop` did not
fire on segment 20), [[project_pr57_separation_scan]] (the production arm
this symptom was found in; confirmed unrelated to its S6 flavor, §2).

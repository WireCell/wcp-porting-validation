# doc pr/59 — shower_track_global hole at cluster 7 seg 20 (run 18255-142421): a segment created after its cluster's only association pass, orphaned by a silent main-cluster swap

Status: **Round 1: root cause confirmed by instrumented rerun, log-only
sentinels, byte-identical proven (§7). Round 2 (§8-§10): fix SHIPPED as
`assoc_full_recluster`. SBND PRODUCTION ON — owner flip 2026-08-10 (§11),
C++ knob default itself stays `false`.** Prototype comparison (§6, M15
check) established this is a genuine defect shared by both trees, not a
toolkit divergence, so the Round-2 fix needed no owner "which reading" call.
Round 2 gates: off-gate byte-identical 19/19 (§9), on-gate rescues all 12
orphans across the 19-event manifest with zero regressions and zero
nusel-verdict movers (§9), Bee links for both events the owner asked about
(§10). Bare production now IS the validated `work-pr59r2-on19` arm,
byte-proven directly (§11); legacy escape `-A assoc_full_recluster=false`
restores the pre-flip byte-identical behavior.

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

Round 2 (§8-§10 — the shipped `assoc_full_recluster` fix):

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# measurement before any fix code -- locate 71372's root orphans (§8.3):
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue WCT_PR59_ASSOC_CENSUS=1 \
  WCT_DET_DEBUG=2 PR_JOBS=2 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr59r2-probe data 71372 142421
grep -A6 "WCT_DETA seg idx=52 \|WCT_DETA seg idx=53 \|WCT_DETA seg idx=199 " \
  work-pr59r2-probe/pr_evt71372/stdout.log

# off-gate, full 19-event manifest (byte-identical proof, §9):
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue PR_JOBS=6 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr59r2-off19 data

# on-gate, same manifest, knob on + census sentinel:
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue SBND_ASSOC_FULL_RECLUSTER=true \
  WCT_PR59_ASSOC_CENSUS=1 PR_JOBS=6 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr59r2-on19 data

# per-event byte-identical hash check, off19 vs the pre-existing production baseline:
for e in 105946 114446 142421 180801 18625 21073 259542 285567 314838 359980 \
         37112 399860 463565 506114 506746 521075 56982 71372 84229; do
  python3 ../../abtest/hash_archive.py work-pr57r6-scan19/pr_evt${e}/mabc-pr.zip
  python3 ../../abtest/hash_archive.py work-pr59r2-off19/pr_evt${e}/mabc-pr.zip
done

# on-gate rescue evidence (pr55 sentinel count, must drop to 0 for a rescued event):
for e in 142421 71372 285567 314838 506114 506746 399860; do
  grep -ac "pr55 shower_track layer" work-pr59r2-off19/pr_evt${e}/wct_pr_evt${e}.log
  grep -ac "pr55 shower_track layer" work-pr59r2-on19/pr_evt${e}/wct_pr_evt${e}.log
done  # -a: these logs trip grep's binary-file heuristic without it (extended-ASCII long lines)

# census: zero "[lost]" tags anywhere in the manifest is the regression watch:
grep -a "pr59r2 recluster:.*\[lost\]" work-pr59r2-on19/pr_evt*/wct_pr_evt*.log  # must be empty

# nusel verdict-mover check:
diff <(sort work-pr59r2-off19/nusel-table.tsv) <(sort work-pr59r2-on19/nusel-table.tsv)  # empty

# Bee bundles for the owner's two events, both arms:
python3 scripts/bee/make_pr_bee.py -q work-ncpi0-cb0805 -p work-pr59r2-off19 \
  -o /home/xqian/tmp/pr59r2-off-71372-142421.zip 71372 142421
python3 scripts/bee/make_pr_bee.py -q work-ncpi0-cb0805 -p work-pr59r2-on19 \
  -o /home/xqian/tmp/pr59r2-on-71372-142421.zip  71372 142421
./upload-to-bee.sh /home/xqian/tmp/pr59r2-off-71372-142421.zip
./upload-to-bee.sh /home/xqian/tmp/pr59r2-on-71372-142421.zip

# compiled-config proof (off/on key presence), from the toolkit repo:
cd ../../../../toolkit
./build/apps/wcsonnet --tla-str input=dummy.tar.gz --tla-code "anode_indices=[0,1]" \
  --tla-str output_dir=/tmp --tla-code run=1 --tla-code subrun=1 --tla-code event=1 \
  --tla-str reality=data --tla-code "pipeline_names=['tagger_check_neutrino']" \
  --tla-str save_tensors=/tmp/x.tar.gz \
  cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet | grep -c assoc_full_recluster  # 0
./build/apps/wcsonnet ... --tla-code assoc_full_recluster=true \
  cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet | grep assoc_full_recluster    # present
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

## 5. Fix — design (Round 1; superseded/shipped in §8-§9)

Owner-selected shape (Round 1's plan, before the owner's two Round-2
constraints below refined it further): an **additive, default-OFF knob** —
after the PR stages complete, identify any segment with a non-empty `fits()`
but a null/absent `associate_points` cloud, and run an association pass for
it, adopting the result only into previously-null segments. This needs its
own gate and an owner flip, since — per `NeutrinoEnergyReco.cxx` §0 above —
turning it on moves reconstructed charge/energy for affected segments, not
just Bee display.

Two mechanisms this fix must NOT silently also touch (documented so a future
implementer doesn't reach for the wrong lever):
- `swap_main_cluster`'s behavior itself is not being questioned here — the
  swap may be entirely correct (106's vertex may genuinely be the better
  neutrino vertex); only its *silence*, and its side effect of stranding the
  demoted cluster's post-swap segments, is the gap.
- The 2D ghost-removal cascade (`PRSegmentFunctions.cxx:2991-3031`) is a
  second, independent, pre-existing way a point can be silently dropped
  (Voronoi-cell owner loses the 2-plane 2D-nearest contest and no other
  segment reclaims it) — it did not fire for segment 20 in this event (§3),
  but any fix that re-runs association broadly, rather than only for
  null-cloud segments, would also re-expose that class and needs its own
  scoping decision. **Round 2 measured this class directly for the first
  time — see §8.3.**

The owner pushed back on "isolated rescue" (call `clustering_points_segments`
with just the orphan) before implementation started: `clustering_points_segments`
is a *competition* (Voronoi + 2D ghost-removal) among exactly the segments
handed to it, so an isolated orphan would win points by default with no
already-good sibling able to contest — the fix has to re-compete the WHOLE
cluster, not just the gap. That is what shipped; see §8.

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
fix (§8) is free to diverge from the prototype's behavior here without an
owner "which reading do you want" call under M15 — there is only one
reading, and both trees share it.

## 7. Verification (Round 1: sentinels only, log-only)

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

## 8. Fix — Round 2, shipped (`assoc_full_recluster`, DEFAULT OFF)

### 8.1 Owner's two constraints

The owner reviewed §5's original "isolated rescue" shape and set two
constraints before implementation:

1. **The (re-)association must happen before track/shower separation** — not
   bolted on at the very end of the PR chain, so the segment's shower/track
   classification can actually consume the new cloud. Verified in code this
   round: `segment_is_shower_topology` (`PRSegmentFunctions.cxx:3449-3450`)
   hard-returns `false` on a null `associate_points` cloud, and
   `segment_is_shower_trajectory` (`:1738`) returns false for any segment
   longer than 50 cm. Segment 20 is 64.9 cm — **under the pre-fix behavior it
   can never be classified as a shower by either route, ever, regardless of
   when a rescue ran** — it is silently forced to read as a track. This is
   the owner's constraint 1, made concrete.
2. **When it fires, recreate the point cloud for the whole cluster** — delete
   the old `associate_points` and establish new ones, not a rescue scoped to
   the orphan alone. This matches the concern already on record in §5:
   `clustering_points_segments` is a *competition* (Voronoi ownership by
   graph-geodesic distance, then a 2-of-3-plane 2D ghost-removal contest)
   among exactly the segments handed to it — an isolated orphan would win
   points by default with no already-good sibling able to contest.

### 8.2 Design and implementation

New method `PatternAlgorithms::reassociate_cluster_orphans(Graph&,
Facade::Cluster&, IDetectorVolumes::pointer)`
(`clus/src/NeutrinoTrackShowerSep.cxx`, declared
`clus/inc/WireCellClus/NeutrinoPatternBase.h`), gated on a new member
`m_assoc_full_recluster` checked *inside* the function (matching the
`main_vertex_graph_audit` idiom — the caller invokes it unconditionally):

1. Collect the cluster's current segments via `ordered_edges(graph)`
   (deterministic edge-index order, never pointer order).
2. Record each segment's `associate_points` point count. **If none is zero,
   return 0 immediately** — an untouched cluster is a byte-identical no-op
   even with the knob on.
3. Otherwise clear `associate_points` on **every** segment in the cluster
   (owner constraint 2's "delete the old") and call
   `clustering_points_segments(segments, dv)` once over the whole set — a
   fresh full-cluster competition, reusing the existing, unmodified entry
   point.
4. For exactly the segments that were orphaned before the clear (owner
   constraint 1's target, scoped so an already-correctly-classified sibling
   is never touched): re-run the same two calls `separate_track_shower`'s
   loop body makes — `segment_is_shower_topology(...)`, then, if not
   topology-shower, `segment_is_shower_trajectory(...)` — with the identical
   arguments. This is *before* every later consumer of `associate_points` or
   the shower/track flags: `determine_direction`,
   `shower_determining_in_main_cluster`, `deghosting`,
   `shower_clustering_with_nv`.
5. Emit a per-segment before/after point-count census
   (`WCT_PR59_ASSOC_CENSUS=1`, the Round 1 sentinel) tagged `rescued` /
   `moved` / `lost` whenever the helper does work — covering the *whole*
   cluster, not just the orphans, because a clear-then-recompete can in
   principle leave a previously-good segment at zero (a manufactured new
   orphan) and that must be visible, not just the intended fix.

Two call sites in `TaggerCheckNeutrino.cxx`, both unconditional (the knob
check lives inside the method):

- **P1**, immediately after each `determine_main_vertex` call (main cluster,
  and both branches of the other-clusters loop). `determine_main_vertex`'s
  internal `examine_structure_final_1/2/3` (measured this round, §8.3) and
  `examine_vertices_1` (Round 1, §3.1) are exactly the mechanisms that
  delete-and-replace segments with a brand-new polyline and no inherited
  association — P1 catches the result the moment it can exist, before
  `determine_direction`/`shower_determining_in_main_cluster`/`deghosting`
  consume it.
- **P2**, immediately after the existing second `clustering_points` call
  (which only ever touched `main_cluster`), looped over `main_cluster` +
  `other_clusters`. A safety net for two things P1 cannot reach: a segment
  created inside `improve_vertex`/`main_vertex_graph_audit` (both run
  between P1 and P2), and the original bug's own mechanism — `main_cluster`
  silently repointed by `swap_main_cluster` since the first
  `clustering_points` call, which otherwise leaves the demoted original main
  cluster on its stale, first-round-only state forever. Still before
  `shower_clustering_with_nv`.

New config knob `assoc_full_recluster` (C++ default `false`), wired with the
house `get(config, "assoc_full_recluster", m_assoc_full_recluster)` /
`default_configuration()` round-trip / `pattern_algos.m_assoc_full_recluster =
...` copy idiom (`TaggerCheckNeutrino.h`/`.cxx`, modeled on
`other_seg_keep_isolated`), threaded through three jsonnet layers with the
key-suppression idiom (`+ (if assoc_full_recluster then {
assoc_full_recluster: true } else {})` — key omitted, byte-identical, when
off):
`cfg/pgrapher/common/clus.jsonnet`'s `tagger_check_neutrino(...)` →
`cfg/pgrapher/experiment/sbnd/clus.jsonnet`'s two `clus_pr`-shaped functions →
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s top-level TLA
(`assoc_full_recluster = false`). Runner: `SBND_ASSOC_FULL_RECLUSTER=1` (or
`-A assoc_full_recluster=true` directly) in
`run_pr_chain_batch.sh`, same tri-state-env-to-TLA contract as every other
knob in that file.

New doctest `TEST_CASE("pattern_recognition reassociate_cluster_orphans
[A]")` (`clus/test/doctest_pattern_recognition.cxx`): manufactures an orphan
by nulling one already-associated segment's cloud after a real
`AfterClusteringPoints` run, confirms (a) knob off is a true no-op — returns
0, cloud stays null — and (b) knob on rescues it (returns > 0, cloud
non-null), and (c) a cluster with no orphan left is a true no-op on a second
call.

### 8.3 Round-2 measurements

**71372's three orphans are born the same way as 142421's segment 20.**
Round 1 only measured 142421 (segment 20, born inside `determine_main_vertex`
via `examine_vertices_1`). Before writing any fix code this round, the same
`WCT_DET_DEBUG=2` backtrace facility was run on 71372
(`work-pr59r2-probe/pr_evt71372/stdout.log`): all three orphans (`19052`,
`19053`, `136199`, i.e. graph indices 52, 53, 199) are created *inside*
`determine_main_vertex`, via `examine_structure_final_1`, `_3`, and `_2`
respectively (`create_segment_from_vertices` /
`merge_two_segments_into_one` / `merge_vertex_into_another` — sibling
structural-cleanup sub-steps of the same function `examine_vertices_1` came
from). **Every orphan measured this round, across both events, is born
at-or-before `determine_main_vertex` returns** — confirmed empirically, not
assumed: P1 alone is sufficient for the whole manifest; P2 fired as a
genuine no-op (0 rescued) everywhere in this 19-event run except where P1's
own re-competition needed a second, post-deghosting pass (§9's evt399860
note).

**`break_segment`'s associate_points redistribution has no else branch** —
correcting §3.1's stated rule. Reading `PRSegmentFunctions.cxx:1087-1153`
this round: the entire redistribution sits inside `if
(seg->dpcloud("associate_points"))`, with nothing outside it. A null-cloud
parent therefore yields **two** null children, not "not orphaned" as §3.1
implied from the single 142421 case (which happened to have no
`break_segment`-born orphan to test this on). This is exactly what explains
71372's `19052`/`19053` pair: `examine_structure_final_1` first creates one
new segment (`52`) with no cloud, and it is subsequently split by
`break_segment` into `52`/`53` — both inherit nothing, because there was
nothing to inherit.

**A distinct, pre-existing failure class survived the fix, correctly.**
`evt399860`'s single residual `pr55 shower_track layer` sentinel (cluster 17,
segment `17005`) is a *different* bug from the one this fix targets: the
`WCT_PR59_ASSOC_CENSUS` Stage-A/Stage-C sentinels show it **does** enter
`clustering_points_segments` (Stage-A seeds 2 terminals from its 7-9 fit
points) but **loses the 2-of-3-plane 2D ghost-removal contest** every single
time — the original pass and both of this fix's re-competes (P1 pre-deghost,
P2 post-deghost). This is precisely the "2D ghost-removal cascade" flagged as
a second, independent, out-of-scope gap in §5 (and confirmed there as *not*
firing for segment 20 in 142421). `reassociate_cluster_orphans` handles this
correctly by design: it attempts the full recompete (since the cluster has
at least one orphan), the segment legitimately loses again, and the helper
leaves it null rather than forcing a result — no crash, no infinite retry,
and (per §9) zero effect on any other segment's final point cloud in that
event.

## 9. Verification (Round 2)

- `wcdoctest-clus`: **153/153 test cases (+1 new), 1620/1620 assertions PASS**
  (`./build/clus/wcdoctest-clus`, post-`wcbuild`); the new
  `reassociate_cluster_orphans [A]` case rescues 1 segment on the uBooNE `[A]`
  fixture and passes all three sub-checks (§8.2).
- Freshness proof: `local/lib/libWireCellClus.so` mtime `21:10:34` after
  `wcbuild`, all five edited/added sources (`NeutrinoTrackShowerSep.cxx`
  `21:08:53`, `TaggerCheckNeutrino.cxx` `21:09:58`, `NeutrinoPatternBase.h`
  `21:08:47`, `TaggerCheckNeutrino.h` `21:08:22`,
  `doctest_pattern_recognition.cxx` `21:13:29`) strictly older.
- **Compiled-config proof**: `wcsonnet` on
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` with the job's minimal
  required TLAs — `assoc_full_recluster` key absent with the knob off (0
  occurrences), present as `"assoc_full_recluster" : true` with `-A
  assoc_full_recluster=true` on; `diff` of the two compiled JSONs is exactly
  that one added line.
- **Off-gate, byte-identical**: full 19-event manifest
  (`work-pr59r2-off19`, `PR_JOBS=6`) vs. the pre-existing production
  `work-pr57r6-scan19` — `hash_archive.py` member-content hash **19/19
  match** (M2: member-content hash, not raw md5/cmp on the zip); merged
  `nusel-table.tsv` `diff` clean across all 19 events.
- **On-gate** (`work-pr59r2-on19`, `SBND_ASSOC_FULL_RECLUSTER=true
  WCT_PR59_ASSOC_CENSUS=1`, same 19 events):
  - `pr55 shower_track layer` sentinel count, off → on:
    142421 1→0, 71372 3→0, 285567 1→0, 314838 2→0, 506114 1→0, 506746 4→0,
    **399860 1→1** (the separate ghost-removal-loss case, §8.3 — correctly
    left alone). The other 12 events had 0 both ways (never touched this
    manifest's `hash_archive.py` result at all — see next bullet).
  - Per-event `mabc-pr.zip` hash, on vs. off: **DIFFERS for exactly the 6
    events with a rescued orphan** (142421, 71372, 285567, 314838, 506114,
    506746), **SAME for all 13 others including 399860** — the knob is a
    true no-op on every cluster it does not need to touch, and even on
    399860's cluster 17 (which *was* re-competed twice, since it still had
    an orphan both times) the final per-segment point counts came out
    byte-identical to the un-recompeted baseline.
  - `pr59r2 recluster` census tally across the whole manifest: **12 segments
    rescued** (142421:1, 71372:3, 285567:1, 314838:2, 506114:1, 506746:4 —
    matches the sentinel deltas exactly), dozens more tagged `moved` (small
    point-count shifts on already-good siblings from the fresh Voronoi
    competition), **zero tagged `lost`** anywhere in the manifest — no
    previously-populated segment was driven to zero.
  - Direct per-segment point-count check on the two events the owner asked
    about (`shower_track-global.json` inside `mabc-pr.zip`,
    `real_cluster_id`-keyed): 142421 cluster 7 — `7020` (segment 20) 0 → 619
    points, all six siblings shift by at most 21 points (`7009` 155→161,
    `7010` 3244→3244, `7016` 679→680, `7017` 51→54, `7019` 702→681, `7111`
    4853→4815), none reach zero; 71372 cluster 19 — shower-collapsed
    `real_cluster_id` `19032` (the start segment the absorbed segments
    report under, per the `id()`-based numbering gotcha) 2794 → 3404 points.
  - `nusel-table.tsv`, on vs. off: **byte-identical across all 19 events** —
    `diff` of the sorted merged tables is empty. Zero verdict movers; the
    fix touches only `associate_points`/shower-classification flags, not any
    variable nusel's selection reads.

## 10. Bee links (both arms, both requested events)

Built with `scripts/bee/make_pr_bee.py -q work-ncpi0-cb0805 -p <pr_root> -o
<zip> 71372 142421` (both events evaluated a neutrino candidate — no
degenerate-dump refusal) and uploaded via `upload-to-bee.sh`. Both sets carry
the same two events in the same order (index 0 = 71372, index 1 = 142421),
so the same Bee URL suffix (`/event/0/`, `/event/1/`) compares directly
across links.

- **Knob OFF** (`work-pr59r2-off19`, today's production behavior, the
  `shower_track_global` hole still present on both events):
  https://www.phy.bnl.gov/twister/bee/set/0dc8f7ac-d832-433b-92f9-e9c2c4c1c295/event/list/
- **Knob ON** (`work-pr59r2-on19`, `assoc_full_recluster=true`, the fix from
  §8):
  https://www.phy.bnl.gov/twister/bee/set/720c176a-976d-4fb0-8d99-38c671d2189b/event/list/

## 11. SBND production flip (owner, 2026-08-10)

Owner, after reviewing §9's gate numbers and §10's Bee links: "flip it on for
default for SBND production." `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s
top-level TLA default changed `assoc_full_recluster = false` →
`assoc_full_recluster = true` — the same flip mechanism as doc pr/54's
`other_seg_keep_isolated` (TLA default flips; the C++ knob's own default
inside `TaggerCheckNeutrino` stays `false`, so any OTHER caller of
`tagger_check_neutrino(...)` that doesn't pass the arg is unaffected).

- **Compiled-config proof**: bare `wcsonnet` compile (no `-A` override) now
  emits `"assoc_full_recluster" : true`; `-A assoc_full_recluster=false`
  (the legacy escape) omits the key entirely, and the resulting compiled
  JSON is `diff`-clean against the pre-Round-2 compiled config saved during
  §9's compiled-config proof — the escape hatch is byte-exact, not just
  "close."
- **Bare-production byte-proof**: reran evt142421 with zero env overrides
  (`work-pr59-flip-bare19`, `PR_JOBS=1`) — `hash_archive.py` member-content
  hash of `mabc-pr.zip` is **identical** to `work-pr59r2-on19`'s
  (`d223c5c4...`). Bare production is not merely "expected to behave like"
  the validated on-gate arm; it is now provably that exact config.
- No further gate is needed beyond §9's: the flip changes which value a
  jsonnet default arg carries, not any code path, and §9 already exercised
  the knob at `true` across the full 19-event manifest.

This is **not a byte-identical change to bare production going forward** —
by design: any event with an orphaned segment (measured this round: 6 of 19
in the current manifest) now gets it rescued. The legacy escape above
reproduces the pre-flip byte-identical behavior for any A/B that needs it.

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
- `break_segment`'s `associate_points` redistribution has no else branch
  (§8.3) — a null-cloud parent yields two null children, not "safe by
  construction" as an earlier draft of §3.1 implied from the single 142421
  case. Check both children, not just the parent, when auditing a
  `break_segment`-born pair.
- A segment can enter `clustering_points_segments` and still end up null by
  *losing* the 2-of-3-plane 2D ghost-removal contest (Stage-C) — this is a
  different, still-open gap from the one `assoc_full_recluster` fixes
  (§8.3's 399860/cluster-17/segment-5 case). `WCT_PR59_ASSOC_CENSUS=1`'s
  Stage-A/Stage-C lines distinguish the two: "never appears in Stage-A" is
  this doc's bug; "Stage-C 'won ZERO points'" is the other one.
- `wct_pr_evt<N>.log` files trip `grep`'s binary-file heuristic (extended-ASCII,
  very long lines) — the count silently comes back empty instead of `0`
  unless you pass `-a`/`--text`. Cost real time working out why a `grep -c`
  loop was returning blank instead of zero.

Related: [[project_pr55_fit_vs_image]] (the existing `shower_track layer`
sentinel that named the symptom but not the mechanism),
[[project_pr54_isolated_residual_keep]] (a different, unrelated way a
segment's points/existence can be silently dropped in the same general
neighborhood of code — ruled out here, `pr54 isolated-residual drop` did not
fire on segment 20), [[project_pr57_separation_scan]] (the production arm
this symptom was found in; confirmed unrelated to its S6 flavor, §2).

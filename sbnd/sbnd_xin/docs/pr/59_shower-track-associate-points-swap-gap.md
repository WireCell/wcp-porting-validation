# doc pr/59 — shower_track_global hole at cluster 7 seg 20 (run 18255-142421): a segment created after its cluster's only association pass, orphaned by a silent main-cluster swap

Status: **root cause confirmed by instrumented rerun. Log-only diagnostic
sentinels added, byte-identical proven (§4). No fix, no new knob shipped this
round** — owner chose "diagnose first" explicitly; the rescue fix is a
follow-up (doc pr/60 or later), designed against the mechanism below instead
of the earlier inference.

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

Cross-checked against the *original* (pre-instrumentation) log's
`print_segs_info` output: the "After first round of main cluster PR" block
(printed after `determine_main_vertex`, i.e. after `clustering_points:1012`
already ran) already lists segment 20 (`Track -1 13 105.658 176.529 1 0`,
64.9 cm) — so segment 20 is created somewhere inside
`separate_track_shower` / `determine_direction` /
`shower_determining_in_main_cluster` / `determine_main_vertex`, all of which
execute strictly after cluster 7's one and only `clustering_points_segments`
invocation. The second invocation, which would have picked it up, ran on
cluster 106 instead because of the swap above. Zero Stage-C "won zero points"
lines fired anywhere in the event (`grep -c "stageC" == 0`) — this rules out
the 2D ghost-removal cascade (`PRSegmentFunctions.cxx:2991-3031`, documented
as a live gap-source in general, see §5) as the mechanism *for this
particular segment*; segment 20's gap is strictly a "never entered the
competition at all" defect, one level simpler than a ghost-removal loss.

**Chain, stated plainly**: cluster 7 is main at `:1012` (gets associated) →
segment 20 is created after `:1012` while cluster 7 is still main → cluster 7
stops being main (silent DL swap to 106) before `:1202` → `:1202` associates
106, not 7 → cluster 7, including segment 20, is permanently stuck with
whatever `associate_points` state existed at `:1012` — which for segment 20
is nothing, since it did not exist yet.

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
- The 2D ghost-removal cascade (`PRSegmentFunctions.cxx:2991-3031`) is a
  second, independent, pre-existing way a point can be silently dropped
  (Voronoi-cell owner loses the 2-plane 2D-nearest contest and no other
  segment reclaims it) — it did not fire for segment 20 in this event (§3),
  but any Round-2 fix that re-runs association broadly, rather than only for
  null-cloud segments, would also re-expose that class and needs its own
  scoping decision.

## 6. Verification (this round: sentinels only, log-only)

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

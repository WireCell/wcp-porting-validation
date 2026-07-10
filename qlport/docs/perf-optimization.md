# uBooNE MABC (qlport) — performance & memory optimization log

Master document for the profiling/optimization campaign over the uBooNE
pattern-recognition jobs driven from `qlport/` (`uboone-mabc.jsonnet`, 35
run-5384 events).  Style and ground rules follow
`toolkit/clus/docs/imgclus-optimization-log.md`.

## Ground rule

Every optimization must keep outputs **byte-identical**:

- Bee zips (`mabc_<idx>.zip`) compared via `abtest/hash_archive.py`
  (member-content hash, mtime-insensitive) — 35/35 must match the baseline.
- `track_com_5384_<EV>.root` is never byte-stable (ROOT timestamps/UUIDs);
  it is validated through `wire-cell-uboone-tagger-compare` against the
  prototype reference — the verbose comparison logs must be identical.
- Anything behavior-affecting becomes a jsonnet knob, default OFF.

## Methodology

- **Harness**: `qlport/scripts/` — `run_one.sh` (single event, isolated
  scratch dir: `track_com_*.root` is cwd-relative in the jsonnet),
  `sweep_5384.sh` (all 35 events, 6-way concurrency cap — box policy),
  `parse_timing.sh` (MABC pass / TaggerCheckNeutrino sub-stage / [timer]
  node aggregation), `summary.py` (population stats + before/after),
  `profile_mabc.sh` (gperftools CPU/heap), `ab_check.sh` (identity gate).
- **Timing/RSS**: `abtest/timecmd.py` — wall + `getrusage(RUSAGE_CHILDREN)`
  peak RSS (== VmHWM).  Headline numbers always from un-preloaded runs.
- **Profiling**: linux `perf` is unavailable on this host (not installed,
  `perf_event_paranoid=3`); CPU/heap profiles use gperftools
  (`LD_PRELOAD=libtcmalloc_and_profiler.so.4`, `CPUPROFILE`/`HEAPPROFILE`),
  analyzed with `google-pprof`.  The jsonnet is pre-compiled with `wcsonnet`
  first (SIGPROF corrupts the gojsonnet GC).  Note tcmalloc-preloaded wall
  times are not comparable to production (glibc malloc) numbers.
- **Build**: toolkit at `-O2 -ggdb3`, non-stripped; `./wcb && ./wcb install`
  after every change.

## Round 0 — baseline

Toolkit at `cd482273` (branch `apply-pointcloud`), 35 events, 6-way
concurrency, warm NFS cache.  Two back-to-back sweeps (`base1`, `base2`):

| metric | min | median | mean | p90 | max |
|---|---|---|---|---|---|
| wall_s | 11 | 14 | 17 | 26 | 33 |
| node_exec_s | 8.50 | 12.08 | 14.65 | 23.76 | 30.78 |
| peak RSS MB | 1738 | 1980 | 2002 | 2213 | 2346 |

base2 medians identical (wall 14 s, node 12.11 s, RSS 1970 MB); per-event
wall noise floor ≈ ±2 s.  Process wall − node exec ≈ 5–7 s is config +
geometry startup (wires JSON, SCE offsets).

**Stage breakdown across the population** (sum over 35 events; node total
512.7 s):

| node | s | % |
|---|---|---|
| MultiAlgBlobClustering | 391.0 | 76.3 |
| UbooneClusterSource | 79.0 | 15.4 |
| UbooneBlobSource (×8) | 22.4 | 4.4 |
| ClusterFlashDump | 13.4 | 2.6 |
| BlobSetMerge | 6.4 | 1.2 |

MABC passes (241.9 s instrumented): TaggerCheckNeutrino 111.2 s (46.0%),
CreateSteinerGraph 82.1 s (33.9%), ClusteringRecoveringBundle 16.8 s (6.9%),
"done" (Bee zip write) 15.4 s (6.4%), "loaded" 10.6 s (4.4%).
TaggerCheckNeutrino sub-stages: main_cluster initial PR 51.0 s, overall main
vertex 31.4 s, other_clusters PR 15.2 s, improve_vertex + examine_direction
7.5 s.

**Representative events**: median-node idx=24 ev=6821; worst-wall and
worst-RSS idx=12 ev=6604 (30.8 s node, 2.35 GB).

### Run-to-run nondeterminism found (and neutralized for A/B)

base1 vs base2 (ASLR on): **12/35 events differ** — always and only in the
pattern-recognition outputs (`track_fit`, `vertices`, `shower_track`, `mc`
Bee members; e.g. ev 6520 fits 917 vs 918 points, a vertex moves ~0.02 cm).
Clustering/steiner members are stable in all 35.  With ASLR disabled
(`setarch -R`) the full zip is bit-identical across repeats, proving the PR
stage is **pointer-order dependent** (heap addresses feed container
iteration order somewhere in the track-fit/vertex code).  Consequences:

- `run_one.sh` disables ASLR by default (`ASLR=1` opts out); the identity
  gate baselines are `det1`/`det2`.
- The pointer-order dependence itself is a pre-existing reproducibility bug,
  out of scope for this perf campaign but documented here as a known issue
  (same family as the QLMatching pointer nondeterminism fixed earlier).

## Round 1 — fix PR run-to-run nondeterminism (DONE)

User-directed priority: fix the nondeterminism before any perf work.

**Diagnosis chain** (event idx21/6804 primary, idx12/6604 secondary):
- ASLR off did NOT stabilize under sweep conditions → not pure address-space
  layout; heap layout perturbations (timing-derived allocation history) leak
  through pointer-ordered container iteration in the PR stage.
- Full-precision compare of `track_com` T_rec_charge: positions bit-identical,
  only fitted charge q differed → localized to the dQ/dx chain.
- gperftools-independent env-gated instrumentation added (`WCT_DET_DEBUG=1`):
  FNV checksums of every BiCGSTAB solve input (A, b, x0) + component hashes
  (RU/MU/F/data/local_dx/traj_pts) in TrackFitting, PR-graph content
  checksums per PR phase in TaggerCheckNeutrino, per-stage fit-point hashes
  in do_multi_tracking, wcpt-level checksums in improve_vertex.

**Root causes found and fixed so far** (all in toolkit clus, byte-behavior =
one canonical order replaces a run-random order):
1. `NeutrinoVertexFinder.cxx improve_vertex`: three `vertex_segments` fills
   iterated raw `boost::out_edges` (pointer order) feeding `fit_vertex` →
   MyFCN normal-equation FP accumulation, and `search_for_vertex_activities`.
   → `sorted_out_edges`.
2. `MultiAlgBlobClustering.cxx fill_bee_points_from_pr_graph`: raw
   `boost::edges` ordering the emitted Bee arrays → `PR::ordered_edges`.
   `fill_bee_pf_tree` BFS seeded/walked via raw out_edges → mc.json sibling
   order → `PR::sorted_out_edges`.
3. `TrackFitting.{h,cxx}`: `m_clusters`/`m_loaded_clusters`/`new_clusters`
   were `std::set<Cluster*>` (pointer order); prepare_data dead-channel pass
   FP-accumulates shared coords in that order → `PR::ClusterPtrCmp`.
4. `TrackFitting.cxx dQ_dx_multi_fit`: two `boost::adjacent_vertices` loops
   (vertex dx sum via connected_pts; connected_vec entries whose order feeds
   F-matrix setFromTriplets duplicate summation) → `sorted_out_edges`.
5. `NeutrinoVertexFinder.cxx examine_main_vertex_candidate`: early `break`
   left ntracks/nshowers as PARTIAL counts in pointer order (feeds
   flag_save_only_showers + candidate map) → `sorted_out_edges`.
   `compare_main_vertices_all_showers`: pts collection via raw edges/vertices
   → `ordered_edges`/`ordered_nodes`.

**Systematic audit round** (three parallel file audits of the PR call path)
produced the remaining fixes:

6. `NeutrinoPatternBase.cxx`: `merge_vertex_into_another` reconnect order
   (re-adds assign NEW graph indices); `break_segments` picked its
   post-process cluster off the FIRST pointer-ordered edge of the SHARED
   multi-cluster graph (could be a different cluster run-to-run) → cluster
   captured at entry; `merge_nearby_vertices` pass-2 first-covered-wins scan;
   `vertex_get_dir` + `calc_dir_cluster` FP coordinate sums;
   `print_segs_info` output order.
7. `NeutrinoVertexFinder.cxx`: `eliminate_short_vertex_activities`
   first-removable-wins scan; `examine_direction` BFS frontier order (assigns
   direction/PDG on first discovery); `calc_conflict_maps` BFS;
   `determine_overall_main_vertex_DL` candidate list + DL input point cloud.
8. `NeutrinoStructureExaminer.cxx`: `examine_vertices_4` merge scan **and its
   reconnect loops (both branches)** — this was the confirmed graph-index
   relabeling source for ev 6604 (creation backtrace via WCT_DET_DEBUG=2);
   `examine_structure_2`/`_3` and `examine_structure_final_1p` degree-2
   sg1/sg2 extraction (decides merged-segment direction);
   `examine_structure_final_1/2/3` first-wins merges; `examine_segment`
   duplicate-survivor pick; `examine_partial_identical_segments` argmin;
   `examine_vertices_1` first-qualifying-segment merge.
9. `NeutrinoTrackShowerSep.cxx`: `fix_maps_shower_in_track_out` vertex order
   (dirsign flips couple vertices); `improve_maps_no_dir_tracks` reclassify
   convergence; `find_cont_muon_segment_nue` + `examine_all_showers` argmax
   ties and length-sum order; `calculate_num_daughter_showers/tracks` BFS
   `acc_length` sum order.
10. `PRShowerFunctions.cxx`: `shower.edges()` is pointer-hashed —
    `shower_get_dis` argmin chain and `shower_cal_dir_3vector` FP sum now use
    a local index-sorted helper.
11. `NeutrinoShowerClustering.cxx`: `map_shower_showers` map+set →
    `ShowerIndexCmp` (acc_energy FP sum order).

Not applied (negligible, tie-only on continuous FP argmax):
`compare_main_vertices` max-muon pick, `find_cont_muon_segment`,
`examine_direction` muon-length argmax (NVF 766/978/1467/1492).

**Diagnostic method that cracked it**: env-gated FNV checksums.
`WCT_DET_DEBUG=1` prints per-solve input hashes (A/b/x0) in TrackFitting,
per-PR-phase graph hashes, per-do_multi_tracking-stage fit hashes with BOTH
an index-ordered hash `h` and an order-independent XOR content hash `hx` —
`hx` match + `h` mismatch proves pure graph-index relabeling.
`WCT_DET_DEBUG=2` additionally logs every segment creation (index, endpoints,
wcpt orientation) with a mini backtrace; diffing two runs' creation logs
attributes the first swapped index to its creating function.  The
instrumentation is kept (zero cost when unset) for future regressions.

**Gate status: PASSED.**  The first full double sweep (`gate1`/`gate2`,
ASLR on) was 33/35; the two residuals (idx18, idx21) were localized with the
WCT_DET_DEBUG=2 creation-log diff to `examine_vertices_2` and four more
remove-and-re-add reconnect loops in NeutrinoStructureExaminer, plus two
first-wins picks in NeutrinoTaggerNuE — all fixed.  Final gate
(`gate3`/`gate4`, two independent 35-event sweeps, ASLR on): **35/35
byte-identical Bee zips**.  repeat_check on the historically worst events
(idx18 ×6, idx21 ×6, idx12 ×4): one distinct hash each.

**Cost**: zero.  gate3 medians — wall 14 s, node 12.15 s, RSS 1980 MB —
match the round-0 baseline exactly.

**Tagger-compare sanity** (vs prototype, base1 outputs vs gate3 outputs):
population total of mismatched branches vs the prototype is 20435 (pre-fix)
→ 20591 (post-fix), a +0.8 % wash with per-event shifts in both directions
(max +71 / −28 on a ~600/event scale).  Most events shift at the
FP-noise level only.  Two events show a discrete tagger flip — the canonical
order re-draws a boundary decision that was previously stable-by-accident
(base1 == base2 on both): ev6528 nue_score −4.30 → −15.0 (proto −3.62,
moves away) and ev6786 nue_score −4.30 → +4.30 (proto +10.7, moves toward).
Both old and new values are equally valid draws of the previously random
process; net prototype agreement is unchanged.

Note: these fixes legitimately CHANGE outputs relative to the pre-fix
baselines (a canonical iteration order replaces a run-random one) — the
det1/base1 hashes are superseded; **gate3 is the reference for all
subsequent perf rounds**.

## Result-changing ideas (not applied)

TBD.

## Gotchas encountered

- The pre-existing `qlport/mabc_*.zip` (2026-04) have 6 Bee members; the
  current binary emits 7 (adds `data/0/0-channel-deadarea-apa0-face0.json`).
  Old in-place outputs are stale — baselines are regenerated, never compared
  against April artifacts.

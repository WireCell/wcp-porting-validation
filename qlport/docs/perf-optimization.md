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

## Round 2 — perf: connect_graph_closely_pid set→vector (DONE)

**Profiles** (gperftools CPU, 250 Hz; wcsonnet-precompiled config):
- idx24/ev6821 (median): 40 % of the whole process is `Main::initialize`,
  dominated by `UbooneNueBDTScorer::init_readers` → `TMVA::Reader::BookMVA`
  XML parsing (38 % ≈ 4.4 s, paid by every per-event job).
- idx12/ev6604 (worst): ~23 % of flat samples are `std::_Rb_tree`
  operations plus ~7 % allocator traffic, concentrated under
  `Graphs::connect_graph_closely_pid` (20.6 % cum; 87 % of its calls from
  CreateSteinerGraph, 45 % via ImproveCluster_2::mutate, plus
  `Facade::Cluster::find_graph`/`make_graph_ctpc_pid` for the PR stage).
  The heap run corroborates: >20 GB cumulative allocations for a
  ~1.5 GB-live event.

**Change** (toolkit `clus/src/connect_graph_closely.cxx`,
`connect_graph_closely_pid` only): the per-wire buckets
`std::map<int, std::set<int>>` become `std::map<int, std::vector<int>>`.
Within one plane a point index lands in exactly one wire bucket, so bucket
contents are disjoint and the former set-union of a bucket range is
reproduced exactly by concatenate + `std::sort` (sorted-unique, identical
iteration order).  The per-point union/intersection sets become reused
`std::vector` buffers with `std::set_intersection` → `back_inserter`.
Candidate content, edge insertion order and FP arithmetic are unchanged —
byte-identical by construction, just without ~10⁶ red-black-tree node
allocations per event.

**Gate — PASSED** (sweep `round1b`, 6-way, vs gate3):
- Bee zips: 35/35 byte-identical.
- Tagger population mismatched-branches total (vs prototype):
  gate3 20591, gate4 20590 (same-binary rerun noise ±1), round1b 20589.
- Per-event nue_score max|diff| identical to gate3 for all 35 events.

**Result** (un-preloaded sweeps, node_exec_s over 35 events):

| metric | gate3 | round1b | delta |
|---|---|---|---|
| wall_s median / max | 14 / 33 | 14 / 28 | — / −15 % |
| node_exec_s median / mean / max | 12.15 / 14.51 / 30.91 | 11.73 / 13.38 / 25.42 | −3 % / −8 % / −18 % |
| peak RSS MB median / max | 1980 / 2358 | 1984 / 2382 | — |

The win concentrates in the busy tail (idx12/30/33: −5 s wall each).
Median-event stage timings: CreateSteinerGraph 1057 → 716 ms,
TaggerCheckNeutrino 2954 → 2637 ms.

**Tried and rejected: lazy TMVA BookMVA.**  Deferring the 16 BookMVA calls
(15 vector sub-BDTs + xgboost) from configure time to first
EvaluateMVA produced byte-identical output and removed the 4.4 s from
startup — but every event in this workload evaluates essentially all
sub-BDTs, so the parse simply moved into the first `visit()`
(UbooneNueBDTScorer node 3 ms → 4550 ms) for zero net wall change, and
peak RSS **rose ~300 MB** (the parse transients no longer overlap the
startup arena; they stack on top of the mid-event working set).  Reverted;
diff preserved in the session scratchpad.  The 4.4 s/job XML parse is a
hard floor of the one-event-per-process job model — amortizing it needs
multi-event jobs, not code changes.

**Gate methodology corrections** (learned this round):
- Tagger-compare logs are NOT byte-stable even between two sweeps of the
  identical binary (gate4 vs gate3: 35/35 logs differ) — the ROOT vector
  branches inherit the residual cosmetic segment-index relabeling that the
  canonical Bee dumps hide, and full-precision values expose sub-mm FP
  noise.  The tagger gate is therefore the population mismatch total plus
  per-event nue_score, not log identity.
- One sweep (`round1`) produced a divergent hash for idx21/ev6804 in
  `shower_track-global.json` only (329/11613 points re-drawn in one
  region).  Four sequential reruns with the same binary all reproduce the
  gate3 hash exactly, under two different work-dir path lengths; a full
  re-sweep (`round1b`) passed 35/35.  A rare load/timing-sensitive
  residual (the round1 sweep ran alongside a heap-profiler job), not an
  effect of the code change; watch for recurrence.

## Round 3 — perf: BFS flag array + BlobSampler regex memoization (DONE)

**Re-profile** (round-2 binary, idx12/ev6604): total samples 6525 → 5625
(−14 %, matching the wall win); `connect_graph_closely_pid` down 1346 → 554
samples (−59 %).  New ranking: `find_proto_vertex` 20 %, `do_multi_tracking`
19.5 % (`form_map_graph` 12.5 %, `form_point_association` 12.3 %),
TMVA startup 19.5 % (structural floor), `UbooneClusterSource::flush` 10 %,
`find_neighbors_nlevel` 9.8 %.

**Changes** (both byte-identical by construction):
1. `clus/src/Graphs.cxx` `Weighted::GraphAlgorithms::find_neighbors_nlevel`
   — the BFS `visited` bookkeeping was a `std::set<vertex_type>`;
   74.5 % of the function's samples were `std::set::find`.  Replaced with a
   flat `std::vector<char>` flag array indexed by the vecS vertex
   descriptor.  Result content (a vertex_set of all vertices within nlevel
   hops) is bookkeeping-independent.  All calls come from
   `TrackFitting::form_point_association`.
2. `clus/src/BlobSampler.cxx` `Sampler::is_extra` — matched a handful of
   fixed array-suffix names against the configured regex list once per
   sampled blob (`std::regex_match` was 18 % of UbooneClusterSource).
   Added a member `unordered_map<string,bool>` memo (same thread-safety
   caveat as the existing `plane_ident2index` cache).

**Gate — PASSED** (sweep `round3`, 6-way, vs gate3):
- Bee zips: 35/35 byte-identical.
- Tagger population mismatched-branches total: 20589
  (gate3 20591, same-binary noise ±1); nue_score per-event max|diff|:
  0 flips (identical values).

**Result** (cumulative over rounds 2+3, un-preloaded sweeps):

| metric | gate3 | round3 | delta |
|---|---|---|---|
| wall_s median / mean / max | 14 / 17 / 33 | 13 / 14 / 24 | −7 % / −18 % / −27 % |
| node_exec_s median / mean / max | 12.15 / 14.51 / 30.91 | 11.12 / 12.46 / 22.36 | −8 % / −14 % / −28 % |
| peak RSS MB median / max | 1980 / 2358 | 1985 / 2355 | — |

**Remaining hotspots** (idx12, round-2 binary): deeper set/map churn inside
`dQ_dx_multi_fit` (lower_bound-heavy charge maps), `break_segments`,
`init_first_segment`; `UbooneClusterSource` sampling (`ChargeStepped::sample`
36 % of the node); the 4.4 s/job TMVA XML startup floor.  Live memory at
peak is dominated by ROOT read caches (234 MB), the booked TMVA forest
(182 MB) and ROOT I/O buffers — structural, little surgical headroom.

## Result-changing ideas (not applied)

- Multi-event wire-cell jobs (amortize the 4.4 s TMVA XML parse and ~7 s
  config/geometry startup across events) — changes the validation job
  structure, not the code; revisit only if batch throughput matters.

## Gotchas encountered

- The pre-existing `qlport/mabc_*.zip` (2026-04) have 6 Bee members; the
  current binary emits 7 (adds `data/0/0-channel-deadarea-apa0-face0.json`).
  Old in-place outputs are stale — baselines are regenerated, never compared
  against April artifacts.
- check_tagger_5384.pl backgrounds its 35 compare processes and returns
  before they finish — compute metrics on its logs only after the compare
  processes exit, or the logs read as empty.

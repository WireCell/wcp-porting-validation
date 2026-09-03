# 28 — PDVD PR job: running time and memory, perf round 1

**Status (2026-09-03).** Measured the 120-event `d27fresh` PR arm, profiled the busy events
(CPU + heap), compared with SBND, and shipped four byte-identical levers plus one
output-neutral memory visitor: toolkit `b53d19c2` (S1+S2), `fc751659` (T1), `9c5d40c7` (D1),
`e7650c7d` (M2+M3), pushed to `apply-pointcloud`. Gate labels and numbers in §6–§8. Everything that was
measured and *not* taken is in §9; the next round's plan is §10.

**Scope.** The PR job = everything `run_pr_evt.sh` runs after Q/L matching
(`pdvd/wct-pr-perevt.jsonnet` → `cfg/pgrapher/experiment/protodunevd/pr.jsonnet`).
The three PDVD scope choices that set the *count* of work — the readout-wide beam window,
`flag_mains`, and the two `*_stm_only` gates (docs 25 §13.10/13.11) — are by design and were
not touched. This round only lowers the *per-unit* cost, so every output stays byte-identical.

## 0. Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/pdvd
# 0. arm profile from what the runner already leaves on disk (no rerun)
python3 stm/perf/pr_perf_profile.py --tag d27fresh --png      # docs/perf/pr_d27fresh_{events,stages}.tsv, docs/pics/doc28_*.png
# 1. busy set (top-5 wall + median + the ncomp=509 witness) on a pinned base lib, sequential
LIB=/home/xqian/tmp/doc28/lib_base   # cp -p local/lib/libWireCell*.so at toolkit 4e2bd2f1
for ev in "039349 6" "039252 8" "039349 7" "039252 15" "039253 11" "039253 15" "039252 5"; do set -- $ev
  ./scripts/stage_pr_tag.sh $1 $2 d28base && LD_LIBRARY_PATH=$LIB PDVD_MAX_JOBS=1 ./run_pr_evt.sh -stm-fit -s d28base $1 $2; done
# 2. CPU profile of one event (precompiled cfg, tcmalloc+profiler, never under setarch -R)
LD_LIBRARY_PATH=$LIB ./profile_pr.sh 039252 5 /home/xqian/tmp/doc28/pr_039252_5.prof
google-pprof --text --cum $(which wire-cell) /home/xqian/tmp/doc28/pr_039252_5.prof | head -40
# 3. heap profile (jemalloc sampling; gperftools HEAPPROFILE is ~25x slower)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
MALLOC_CONF=prof:true,prof_prefix:/home/xqian/tmp/doc28/heap/jeprof,lg_prof_sample:19,lg_prof_interval:30,prof_final:true \
GOGC=off wire-cell -l stderr -l wct.log:debug -L debug -c .wct-pr_<tag>.json     # then jeprof --text --cum --inuse_space
# 4. censuses (log-only env gates) on the new lib
WCT_RELAXED_EDGE_TIMING=1 WCT_TF_COVERAGE_CENSUS=1 WCT_DQDX_SOLVE_CENSUS=1 ./run_pr_evt.sh -stm-fit -s d28cen 039252 5
python3 stm/perf/pr_census_summary.py work/039252_5_d28cen/wct_pr_039252_5.log
# 5. gates: two tags, bee zip members + calib json (minus dual_chain timer) + ROOT branch buffers
python3 stm/gates/r28_gate.py d28base d28l2
```

Instruments: `pr_resource_*.txt` (bash wall + exact VmHWM), `pr_rss_*.csv` (2-s sampler),
the `MABC timing:` per-stage table and the `MEM: total … res=` per-stage summary that
`MultiAlgBlobClustering` writes in `perf` mode (already on for PDVD), the `Timer:` node
table. Two caveats that decide how numbers are read:

- **Batch wall is contention-inflated.** Under `PDVD_MAX_JOBS=6` the median event spends
  22.9 s of its 50.6 s wall inside `TensorFileSource` reading the 22 MB point-tree tarball
  (1.15 core-s). Quote `node_core_s` (MultiAlgBlobClustering core-sec); the PR node is
  single-threaded and CPU-bound on the busy events (core ≈ wall).
- **`MEM:` and the sampler report RSS, not live heap.** With tcmalloc the live heap on
  039252/5 is 1.65 GB where RSS reads 3.26 GB (§5).

## 1. The arm profile (`d27fresh`, 120 events)

| quantity | sum | p25 | p50 | p90 | max (event) |
|---|---|---|---|---|---|
| wall_s (batch) | 8488 | 32.8 | 51.5 | 147 | 744 (039252/5) |
| node_core_s | 7807 | 27.8 | 43.8 | 133 | 742 (039252/5) |
| TensorFileSource wall | 627 | 1.2 | 2.2 | 22.1 | 23.0 (039253/2) |
| peak RSS GB (VmHWM) | — | 1.10 | 1.42 | 2.77 | 3.61 (039252/5) |

| stage | sum s | share | worst event |
|---|---|---|---|
| TaggerCheckNeutrino | 3762 | 48.4 % | 328.6 (039253/15) |
| TaggerCheckSTM | 1666 | 21.4 % | 62.6 (039252/16) |
| CreateSteinerGraph | 1386 | 17.8 % | 55.9 (039252/8) |
| ClusteringProtectBundle | 504 | 6.5 % | **450.7 (039252/5)** — one `connect_graph_relaxed_strict` call, ncomp=385 |
| TaggerCheckTGM | 156 | 2.0 % | 11.2 |
| everything else (writers, load, scope, FC, done) | 303 | 3.9 % | |

Per-unit costs over the arm: steiner 194 ms per kept cluster (7135), STM 300 ms per
evaluated main (5543), TGM 22 ms per main, neutrino PR **6.4 s per candidate** (590).
Pictures: `docs/pics/doc28_d27fresh_stage_shares.png`, `doc28_d27fresh_rss_vs_core.png`.
Tables: `docs/perf/pr_d27fresh_events.tsv`, `pr_d27fresh_stages.tsv`.

## 2. SBND comparison (prod0901 medians, `sbnd_xin/docs/pr/pr142-perf-*.tsv`)

| | SBND νe MC (nuecc48) | SBND data (mcp1k) | PDVD data (d27fresh) |
|---|---|---|---|
| event core-s, median | 15.7 | 1.55 | 43.8 |
| peak RSS, median | 1.05 GiB | 0.45 GiB | 1.42 GB |
| TaggerCheckNeutrino | 10.1 s | 0.6 ms | 31 s (mean) |
| CreateSteinerGraph | 0.60 s | 0.07 s | 11.6 s (mean) |
| TaggerCheckSTM | 0.01 s | 1.8 ms | 13.9 s (mean) |
| ProtectBundle | 0.12 s | 0.5 ms | 4.2 s (mean; 0.5 ms median) |
| clusters through steiner / mains tagged / ν candidates | 12 / 1 / 1 | 12 / 1 / 0 | 59 / 59 / 4.9 |

Per unit the two detectors are the same code at the same cost: SBND's ν candidate takes
10 s, PDVD's 6.4 s; SBND's steiner build 50 ms per cluster, PDVD's 194 ms (PDVD clusters are
longer). PDVD is *lighter* on the two things SBND pays and PDVD does not (the SCN vertex
network and the two BDT scorers, ~4 s and ~0.9 GB). The whole gap is the count: 60–80
flash-matched mains per PDVD event through steiner/TGM/STM versus SBND's one in-window
bundle, and 2–12 ν candidates versus one — the readout-wide window is the design (doc 25
§2.1), so the round targets per-unit cost.

## 3. Busy-set CPU profiles (base lib, `docs/perf/doc28_prof_*.txt`)

Three events, three different shapes — there is no single hot spot:

- **039252/5 (742 s): 57 % of all samples are `Simple3DPointCloud::get_closest_dis`** inside
  the strict connector's S5 ghost test (`max_ghost_run`): for every 1 cm step of every
  candidate walk it queried each of the 385 component kd-trees. 60 % of the event was one
  `connected_blobs` call. TrackFitting 19 %, steiner 6 %.
- **039253/15 (369 s): 74 % is Eigen's sparse×dense product inside `BiCGSTAB` in
  `dQ_dx_multi_fit`** — the multi-track dQ/dx solve with the default tolerance
  (machine epsilon) runs to its iteration cap on the large systems (max n3d 2093 on
  039252/5). This is the numerics; only its constant factor is touchable (§9).
- **039349/7 (158 s): map-bound.** `dQ_dx_fit` 22.5 % (std::map/unordered_map build over
  every 2D cell of the loaded cluster — up to 90k cells to fit a 3-point track),
  `dQ_dx_multi_fit` 21.8 %, `update_association`→`exclusion_closest_2d_dis` 10.9 %
  (per-call `sincos` + a `std::map<WirePlaneId>` lookup), `fill_fitted_charge_2d` 11.9 %.

Log-gap attribution (§0 of the plan) had charged 81 s to the `fit_blob_coverage` debug
line: that line is emitted once per trajectory point and the cost is the association loop
after it, not logging (171k lines ≈ 1.4 s). The retile (`CreateSteinerGraph` 18 % of the
arm) is diffuse: sampling 24 %, `ctpc_pid` graph 21 %, `closely_pid` 13 %, activity 10 %,
tree node removal 7 % — nothing above 1 % of an event.

## 4. Censuses (new, log-only, env-gated; `stm/perf/pr_census_summary.py`)

On 039252/5 with the round-1 library (`d28cen`):

- `WCT_RELAXED_EDGE_TIMING=1` — strict connector, 8 calls, 9.5 s total (was 454 s): the
  ncomp=385 call 9.3 s = 54189 closest-pair walks 8.6 s; dir1/dir2 walks 31, 0.02 s; ghost
  tests 1360 (29043 steps) **16 ms**. With S1+S2 the connector is walk-bound, and the walks
  are the killed bridges of a genuinely disconnected cluster (F1 in the design note).
- `WCT_DQDX_SOLVE_CENSUS=1` — multi-fit solver 1329 calls 6.6 s (max 143 ms, 207671
  iterations, max n3d 2093); single-fit 1023 calls 0.1 s. On this event the solver is not
  the cost; on 039253/15 it is 74 %.
- `WCT_TF_COVERAGE_CENSUS=1` — `is_cell_covered_by_foreign_blobs` 3.96 M calls; with the
  index 18.5 clusters visited per call instead of 280 (the whole grouping), 4.27 M box
  survivors, 3.18 M covering.

## 5. Heap (jemalloc sampling on 039252/5, base lib; `docs/perf/doc28_heap_039252_5_i*.txt`)

Live heap 1653 MB after the STM stage, 2403 MB at the end of the neutrino stage, 329 MB
after teardown; the sampler's RSS at the same points 3.26 / 3.27 / — GB (tcmalloc retains
the churn: 574 GiB were allocated over the event).

| owner at end of neutrino stage | live MB |
|---|---|
| loaded point-cloud tree (`as_pctree` → `Dataset::slice`, 16 volumes) | 707 |
| TaggerCheckNeutrino fitters (12 `nu<i>` slots): `fill_global_rb_map` 650, 2D↔3D maps | 850 |
| CreateSteinerGraph: `steiner_graph` copies 192 + GraphAlgorithms caches 160 | 366 |
| TaggerCheckSTM member fitter scratch | 243 |
| TaggerCheckTGM `ctpc` graphs | 95 |
| raw tensors (`TFSTensor`) | 99 |

The writers add +360 MB RSS (PrDisplayDump builds the whole 24 MB JSON DOM) and `done`
+77 MB (`as_tensors` of every grouping, discarded by the sink). VmHWM 3.61 GB = the
end-of-job RSS; the STM-stage RSS is already 3.26 GB.

## 6. Levers shipped (toolkit, one commit each; all default paths byte-identical)

| id | site | what | identity argument | effect |
|---|---|---|---|---|
| S1 | `connect_graph_relaxed_strict.cxx` `max_ghost_run` | one `cluster.kd3d().knn1` per step instead of one `get_closest_dis` per component | component clouds are filled from `cluster.points()` = `kd3d().points()` over every graph vertex, so their union is the kd3d set; both trees are nanoflann L2 over the same doubles, knn1 is exact, sqrt monotone ⇒ `sqrt(min metric)` == `min sqrt(metric)` bit for bit; only `img_dis > radius` is consumed | 039252/5 connector 454 s → 9.5 s |
| S2 | same, three S5 call sites | skip the ghost walk when S1–S3 already killed the candidate (census mode keeps it) | its only effect there is `invalidate_distance()` on an invalid tuple | part of the above |
| T1 | `TrackFitting::is_cell_covered_by_foreign_blobs` | per-(apa,face) index of cluster bounding boxes in (slice, u/v/w wire), children order; box is a necessary condition of the unchanged predicate | superset visited in the same order with the verbatim predicate ⇒ same first claimant, same OR; (ptr, nblobs) signature re-checked every pass | 280 → 18.5 cluster visits per call; ~1 % of 039252/5 |
| D1 | `DynamicPointCloud` 2D queries | cache {cos, sin} of the resolved plane angles per query volume | same `resolve_wpid_key` + `cos()/sin()` on the same doubles, computed once; cleared when the params map changes | `exclusion_closest_2d_dis` ~11 % of 039349/7: removes its sincos + map lookup |
| M3 | `TrackFitting::release_fit_scratch()` | drop charge maps, 2D↔3D maps, `global_rb_map`, per-cluster caches, edge caches, coverage index | keeps every consumer getter's data (graph, segments, fitted 2D charge, showers, tagger/kine, scoreboard, pi0 maps); the writers call only those | called by M2 |
| M2 | new visitor `ClusteringReleaseCaches` (`release_post_nu` in `pr.jsonnet`, runner `PDVD_PR_RELEASE=1`) | after `tagger_check_neutrino`: `take_graph` + `remove_graph_algorithms` for every stored graph/cache (the ProtectBundle split sequence), `release_fit_scratch` on the unnamed, `nu<i>` and `stm` slots | nothing downstream reads graphs/caches; local PCs stay; absent from `pipeline_names` ⇒ absent from the compiled config (proved diff-to-zero) | §8 |
| M4 | runner env | `TCMALLOC_RELEASE_RATE` passthrough | allocator policy only | §8 |

Censuses (log-only, no code path when unset): `WCT_RELAXED_EDGE_TIMING`,
`WCT_TF_COVERAGE_CENSUS`, `WCT_DQDX_SOLVE_CENSUS`. Runner: `PDVD_LOG_LEVEL` /
`PDVD_LOG_LOGGERS` (expose the TRACE phase timers without editing the script),
`scripts/stage_pr_tag.sh` (PR-only tag from a point-tree tag), `profile_pr.sh`.

## 7. Byte-identity gates

Every gate compares two arms built on pinned library snapshots (`/home/xqian/tmp/doc28/lib_base`
= `local/lib` at toolkit `4e2bd2f1`, copied 13:37; `lib_l2` = the round's cumulative build,
14:18) and was verified to map those libraries through `/proc/<pid>/maps`. Hash
scripts: `stm/gates/r28_gate.py` (PDVD: `mabc-pr.zip` member content, `calib-pr` JSON minus
the `dual_chain` timer, both ROOT files as awkward buffers per branch), the doc 25
`shared_gate.sh` (SBND: bee + calib + nusel TSV; uBooNE: `ab_check.sh`).

| gate | arms | result | record |
|---|---|---|---|
| PDVD busy set, S1+S2+T1 | `d28base` vs `d28l1` (lib_l1) | **28/28** identical | `stm/gates/r28_d28l1_vs_d28base.txt` |
| PDVD busy set, + D1 (cumulative) | `d28base` vs `d28l2` | **28/28** | `r28_d28l2_vs_d28base.txt` |
| PDVD busy set, + release visitor | `d28base` vs `d28m1` (`PDVD_PR_RELEASE=1`) | **28/28** | `r28_d28m1_vs_d28base.txt` |
| PDVD busy set, + `TCMALLOC_RELEASE_RATE=10` | `d28base` vs `d28m2` | **28/28** | `r28_d28m2_vs_d28base.txt` |
| PDVD 120 events, cumulative | `d28fbase` vs `d28fpost` (8 jobs each) | **480/480** (bee, calib, `tracking-pr.root`, `tracking-stm.root` × 120). Two events (039253/3, 039349/71) write the two `T_rec_charge` rows of one shared vertex in the opposite order — same rows, same values; the qlport repeat check hashes that tree as a multiset for this reason (doc 90 §4/7) and `r28_gate.py` now does the same, reporting the swap here rather than hiding it. Verified not a physics change by rebuilding the peer's `d2edd63a` state with and without the four commits (`lib_base2`/`lib_post2`): the order is a function of the binary, the multiset is not | `r28_d28fpost_vs_d28fbase.txt` |
| SBND nuecc48 + ncpi0 (`d99fix` Q/L roots, DL off) | `work-*-doc25_28base` vs `_28post` | **201/201** (67 events × bee, calib, nusel) | `r28_sbnd_compare_28post_vs_28base.txt` |
| uBooNE 5384 sweep (35 events) | `sweep/doc25_28base` vs `_28post` | **35/35 zips**; tagger 34/35 — the one is idx 22 / event 6805, the documented layout-bistable event (doc 26 round 2) — `repeat_check.sh 22 4` under each library gives the same two tagger states (2 of 4 runs each side, Bee zip identical in all 8), so the diff is the bistability, not the code | `r28_ub_ab_check_28post_vs_28base.txt` |

The busy set is 039252/5, 039253/15, 039253/11, 039349/7, 039252/15 (the five costliest
walls), 039349/6 (the median) and 039252/8 (the ncomp=509 witness). Note that
`d27fresh` could not serve as the base: the peer's BlobSampler commit `4e2bd2f1` (13:43)
landed between it and this round and sits in every library here, so the base arm was rerun.
The `lib_l2` snapshot also carries the peer's then-uncommitted SteinerGrapher edit (14:09,
later `d2edd63a`), so the cross-detector gates cover that commit too.

## 8. Timing and memory, before/after

**Time** (busy set, the clean pair: `d28seqb` on `lib_base` and `d28seqp` on `lib_l2`, each
sequential, the two side by side on an otherwise quiet box; `stm/gates/r28_d28seqp_vs_d28seqb.txt`,
28/28 identical):

| event | base core-s | post core-s | speed-up | where |
|---|---|---|---|---|
| 039252/5 | 679.7 | 270.9 | **2.51×** | ProtectBundle 415 → 8 s; Neutrino 173 → 171, STM 39 → 40, steiner 33 → 33 |
| 039253/15 | 349.8 | 346.7 | 1.01× | BiCGSTAB-bound (Neutrino 319 → 317), untouched by design |
| 039253/11 | 224.1 | 210.8 | 1.06× | |
| 039349/7 | 161.4 | 150.0 | 1.08× | Neutrino 109 → 100 (T1+D1), ProtectBundle 9 → 1 |
| 039252/15 | 154.9 | 148.7 | 1.04× | |
| 039252/8 | 148.7 | 149.7 | 0.99× | the ncomp=509 witness: its connector is already lazy |
| 039349/6 (median) | 24.8 | 25.3 | 0.98× | noise |
| **sum** | **1743.3** | **1302.1** | **1.34×** | |

The parallel arms (`d28l1`, `d28l2`, seven events at once) agree: 765.7 → 343.9 core-s on
039252/5 under 7-way contention. Arm-wide the levers remove the 454 s tail (5.8 % of the
7807 core-s arm) and 5–8 % of the neutrino-heavy events; the three big stages keep their
per-unit cost, because their cost is the fit numerics, the map building (T2, deferred) and
a diffuse retile. Run-to-run noise on this box is ±2–3 % (039252/8, 039349/6).

**Memory.** The sampler's VmHWM under tcmalloc is not usable as a metric: the identity arm
alone moves the same event by up to 0.8 GB between runs (039253/11: 4.16 / 3.41 / 3.34 GB
in `d28base` / `d28l1` / `d28l2`; 039349/7: 1.98 / 2.09 / 2.20). Matched jemalloc runs of
039252/5 (`d28h0` = base lib without the visitor, `d28h1` = `lib_l2` with `PDVD_PR_RELEASE=1`;
`docs/perf/` + `/home/xqian/tmp/doc28/heap{0,1}`) give the defensible numbers:

| 039252/5 under jemalloc | without visitor | with visitor |
|---|---|---|
| live heap at end of TaggerCheckNeutrino | 2.85–3.06 GB | 2.85 GB |
| live heap after `release_post_nu` | — | **1.78 GB** (−1.07 GB) |
| live heap peak during the writers | 3.06 GB | 1.98 GB |
| RSS after neutrino → after writers → done | 3.15 → 3.54 → 3.50 GB | 3.18 → 2.84 → 3.00 GB |
| VmHWM (runner) | 3.75 GB | **3.10 GB** |

Under tcmalloc on the busy set the visitor's VmHWM effect is visible where the peak is the
writer stage (039252/5 3.54 → 3.34, 039253/11 3.34 → 3.36, 039252/15 3.02 → 2.92) and lost in
the run-to-run noise on the small events. `TCMALLOC_RELEASE_RATE=10` (`d28m2`) changed
nothing systematic and is not recommended. The plateau itself (STM-stage live 1.65 GB,
RSS 3.26 GB) is the loaded tree (0.71 GB) plus allocator retention of the churn (574 GiB
allocated over the event): round-2 material (§9).

## 9. Measured and not taken

- **T2 — `dQ_dx_fit` fit-side row window.** The single-track fit builds `std::map`s and
  per-row `std::set<Coord2D>` over *every* 2D cell of the loaded cluster (up to 90k rows
  for a 3-point fit; `dQ_dx_fit` is 22.5 % of 039349/7 and the STM stage's main term). Rows
  not coupled to any trajectory point contribute no term to `RUT·MU·RU` or `b`, so
  restricting the fit-side structures to the coupled rows is exact — but the product map
  (`fill_fitted_charge_2d`, the Bee `stm_fit` layer, `T_proj_data`) records *every* row
  with pred 0, and that per-row lambda is itself 9 % of the event. Net saving ~10 % of an
  STM-heavy event for a ~300-line rewrite of an 800-line function with a delicate identity
  argument: deferred to round 2 with this design.
- **The BiCGSTAB solve** (74 % of 039253/15): default tolerance = machine epsilon, so the
  large systems run to the iteration cap; any tolerance/solver change moves results. The
  matvec on a symmetric column-major `A` is the same summation order as a row-major dot per
  row, which would allow thread-parallel rows bit-identically, but the toolkit is not built
  with OpenMP and threads do not cut CPU time. Left for the owner as a build-level option.
- **R1 `get_overlap_good_ch_charge`**: 0.05 % of 039252/5 — the whole-CTPC scan is not the
  cost on PDVD. **R2/R3** (activity-array window, `get_uvwt_min/max` sets): the retile is
  diffuse (§3), each below 1 %. Not worth an A/B (the round-1 lesson: do the arithmetic).
- **S3/S4** (doc 78 early exits, lazy dir1/dir2): after S1+S2 the connector is 9.5 s on the
  worst event and 0.05 s median; the remaining 8.6 s is 54k killed-bridge walks of a
  disconnected cluster at 0.16 ms each. S4's ceiling is the dir walks (0.02 s). Not
  pursued.
- **Log volume**: the 171k `fit_blob_coverage` lines cost ~1.4 s of 745; census scripts
  anchor on the prefix. Left.
- **Loaded tree = 707 MB live** (`Dataset::slice` per blob: ~16 KB per blob for ~12 arrays
  whose payload is ~100 B — jsoncpp metadata and per-array objects). A util-level change to
  share metadata across sliced arrays would halve the plateau on every detector; out of
  scope for a clus perf round (touches the base package), recorded for the owner.
- **M1 PrDisplayDump streaming** (~360 MB of the writer spike, byte-identical by
  FastWriter's compositional grammar): not implemented this round; M2/M3 lower the
  plateau it sits on (§8).

## 10. Next steps

1. Round 2 = T2 (§9) with the census as the sizing instrument, gated on the same busy set.
2. Q/L-stage `ctpc_fast` / `busy_num_threshold` for PDVD (SBND docs 78/79; the PDVD Q/L
   job passes neither) — a separate Q/L round, same identity bar.
3. Point-tree metadata sharing in `util` (the 707 MB) — owner decision, cross-detector.
4. The batch input stall (23 s of the median event's wall under 6 jobs): a zstd point-tree
   codec (the `sp_sink_ext` precedent, 27×) or staggered starts — runner work, no reco
   change.

## Milestone log

- 2026-09-03 — arm profile, three CPU profiles, one heap profile, SBND comparison;
  S1/S2/T1/D1/M2/M3/M4 shipped; gates §7; doc + memory note.

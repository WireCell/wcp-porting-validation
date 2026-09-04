# 28 — PDVD PR job: running time and memory, perf round 1

**Status (2026-09-03).** Measured the 120-event `d27fresh` PR arm, profiled the busy events
(CPU + heap), compared with SBND, and shipped four byte-identical levers plus one
output-neutral memory visitor: toolkit `b53d19c2` (S1+S2), `fc751659` (T1), `9c5d40c7` (D1),
`e7650c7d` (M2+M3), pushed to `apply-pointcloud`. Gate labels and numbers in §6–§8. Everything that was
measured and *not* taken is in §9; the next round's plan is §10.

**Round 2 (2026-09-03, §11–§18).** The three items §10 named: T2 (the single-track dQ/dx
fit's per-row containers, plus the compact-matrix dense array and a channel→wires memo),
the point-tree footprint in `util` (Array 144 → 88 bytes), and the Q/L-stage fast
flavors for PDVD (wired, gated identical, measured not worth turning on). Toolkit
`66831770`, `34d0a5f5`, `1a7e1b66`. Busy set 1283 → 1176 node core-s (1.09×; the 120-event
arm 8194 → 6935), STM stage 1.5–1.7× on every event, live heap at the STM stage 2290 →
1587 MB; PDVD 28/28 and 478/480, SBND 201/201, uBooNE 35/35 — the one differing archive is
a pre-existing out-of-range read in the STM eval that the gate exposed (§13, §15).

**Round 3 (2026-09-03, §19–§26).** The owner's asks: fix the STM eval's out-of-range read
(done, toolkit `8c577c4b`; fires on 2 of 120 PDVD events, `T_stm_eval` rows only, no
verdict moves; SBND/uBooNE untouched), and explain why PDVD events cost so much more than
SBND's, borrowing SBND's busy-event gating where it applies. The neutrino stage was 57 %
of the arm; 57 % of *it* was doc pr/112's dual-chain OFF pass, copied from SBND's production
settings and, on a detector with no DL vertex net, never read by anything — 2263 of 6935
node core-s. It is now skipped when nothing can consume it (`a94ce32e`, output-identical by
construction, PDVD 477/480 with the two STM archives above the only movers, SBND 201/201,
uBooNE 35/35). Busy set 1135 → 660 node core-s (1.72×), 039252/5 2.37×, the 120-event arm
6935 → 4742 (1.46×). The rest of the gap to SBND is the count and the length of PDVD's
candidates (§21); the busy-event gate that fits is a scope knob for the owner (§23).

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

## 10. Next steps (as written at the end of round 1; 1–3 done in round 2, §11–§18)

1. Round 2 = T2 (§9) with the census as the sizing instrument, gated on the same busy set. **Done, §13.**
2. Q/L-stage `ctpc_fast` / `busy_num_threshold` for PDVD (SBND docs 78/79; the PDVD Q/L
   job passes neither) — a separate Q/L round, same identity bar. **Done, §14: identical, not worth it on PDVD.**
3. Point-tree metadata sharing in `util` (the 707 MB) — owner decision, cross-detector. **Round 2 took the exact part, §12.**
4. The batch input stall (23 s of the median event's wall under 6 jobs): a zstd point-tree
   codec (the `sp_sink_ext` precedent, 27×) or staggered starts — runner work, no reco
   change.

---

# Round 2

## 11. Round 2 — scope, repro, labels

Three items from §10, in the order the owner listed them: T2, the point-tree footprint,
the Q/L-stage fast connectors. Same bar as round 1: every default path byte-identical,
proven by gates on pinned library snapshots (`/home/xqian/tmp/doc28/lib_r2base` = toolkit
`8bf5b1bc`, the peer's doc 31 round-7 state, which is why the round-1 arms could not serve
as this round's base; `lib_r2post` = that plus the round-2 commits). The 32-process licence
was used: the two 120-event arms ran 14–16 jobs each with the pairs alongside.

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/pdvd
# census of the loaded tree: arrays per blob straight from the point-tree tarball
python3 - <<'EOF'   # see sec 12: 64 414 live blobs x (21 "3d" + 18 "scalar") arrays
import tarfile, json, io, numpy as np
tf = tarfile.open('work/039252_5_d27fresh/pctree-evt298637.tar.gz')
for m in tf.getmembers():
    if m.name.endswith('metadata.json'):
        md = json.load(tf.extractfile(m))
        if md.get('datatype') == 'pcarray' and 'lpcmaps' in md['datapath']:
            a = np.load(io.BytesIO(tf.extractfile(m.name.replace('metadata.json', 'array.npy')).read()))
            print(md['datapath'].split('/')[-1], 'blobs', int((a > 0).sum()), 'points', int(a.sum()))
EOF
# busy-set clean pair (sequential per arm, the two side by side) and its gate
/home/xqian/tmp/doc28/run_r2seqpair2.sh          # tags d28r2sb2 (lib_r2base) / d28r2sp2 (lib_r2post)
python3 stm/gates/r28_gate.py d28r2sb2 d28r2sp2 --out stm/gates/r28_d28r2sp2_vs_d28r2sb2.txt
# matched jemalloc pair of 039252/5, both with PDVD_PR_RELEASE=1 (round-1 operating point)
/home/xqian/tmp/doc28/run_r2heappair2.sh         # tags d28r2hb2 / d28r2hp2, dumps in /home/xqian/tmp/doc28/heap2{b,p}2
# 120-event arms and gate
/home/xqian/tmp/doc28/run_full.sh d28r2fb /home/xqian/tmp/doc28/lib_r2base   # and d28r2fp on lib_r2post
python3 stm/gates/r28_gate.py d28r2fb d28r2fp --out stm/gates/r28_d28r2fp_vs_d28r2fb.txt
# Q/L fast flavors: knobs off vs on, same lib, five worst Deghost events
/home/xqian/tmp/doc28/run_r2qlpair.sh            # tags r2qloff / r2qlon (PDVD_CLUS_TLA="-S ql_po_fast=true -S ql_dg_fast=true")
python3 stm/gates/r28_ql_gate.py r2qloff r2qlon --out
# compiled-config proof for the knobs (runner compile-only mode, knob off vs base, knob on)
PDVD_CLUS_COMPILE_ONLY=1 PDVD_KEEP_CFG=1 ./run_clus_evt.sh -s r2cfgpost -calib -op -save-pctree 039252 8
# shared-component gates
QL_SUFFIX=d99fix OLD_ARM=_r2base NEW_ARM=_r2post OLD_LIB=/home/xqian/tmp/doc28/lib_r2base NEW_LIB=/home/xqian/tmp/doc28/lib_r2post ./stm/gates/shared_gate.sh arms
QL_SUFFIX=d99fix OLD_ARM=_r2base NEW_ARM=_r2post ./stm/gates/shared_gate.sh compare
(cd ../qlport/scripts && LD_LIBRARY_PATH=/home/xqian/tmp/doc28/lib_r2post:$LD_LIBRARY_PATH ./repeat_check.sh 22 4 rep28r2post 4)
```

Toolkit commits (`apply-pointcloud`): `66831770` (util Array footprint), `34d0a5f5` (T2 + memo +
compact matrix), `1a7e1b66` (PDVD Q/L knobs). wcp-porting-img: this doc, the Q/L
runner knobs, `stm/gates/r28_ql_gate.py`, the gate records.

## 12. The loaded point tree: what the 707 MB is, and the exact part taken

**Census (039252/5).** The live tree has 64 414 blobs. Every blob carries two local point
clouds: `3d` with 21 arrays (x, y, z, t, x_t0cor, wpid, three wire indices, six charge
val/unc, six 2D projections) over 2.7 points on average, and `scalar` with 18 one-element
arrays. That is 39 `Array` objects per blob, 2.51 M for the event, and the per-object cost
dominates the payload: jeprof under `as_pctree` splits the 707 MB into 373 MB of
`make_shared<Array>` objects (144-byte `Array` + 16-byte control block → the 160-byte size
class), 190 MB of `std::map` nodes in `Dataset` (80 bytes each), 104 MB of payload copies,
25 MB of `local_pcs` map nodes. So "metadata" in §5 was the wrong word: the arrays' jsoncpp
metadata is null; the cost is 2.5 M small objects. The same shape holds for every detector
that loads a tree (SBND: `3d` + `scalar` per blob too).

**What round 2 changed (util, exact, API unchanged; the only `util` edit of the campaign):**
`Array` keeps its dtype as a pointer into a process-wide interned table (`dtype()` and
`shape()` already return by value) and allocates its `Json::Value` metadata lazily (a const
read of an array that never had metadata set returns a shared null value, which is what the
always-present member held). `sizeof(Array)` 144 → 88, `Dataset` 112 → 88; the
`make_shared<Array>` allocation moves from the 160- to the 112-byte class. Measured on the
jemalloc pairs: at equal allocation volume during the load `as_pctree` 676 → 606 MB (the
Array objects 350 → 266 MB); at the STM stage stamp of the final pair 676 → 650 MB. The
sampled estimates spread by tens of MB; the arithmetic says −48 bytes × 2.5 M = −120 MB. All util/aux/clus doctests pass (276 / 22 / 277). The
`Array` move-assignment swaps `m_bytes` but not `m_store` (pre-existing; a moved-from owner's
span dangles) — left as is, noted for the owner.

**Tried and dropped: `boost::container::flat_map` for `Dataset`'s store** (would have taken
the 190 MB of map nodes to ~110). Boost 1.85's `flat_map` in this install mis-inserts under
`-O2`: after `m.insert({"x", p})` the size is 1 but `find("x")` fails and `begin()==end()`
(`/home/xqian/tmp/claude-25225/.../scratchpad/fm.cxx`, correct at `-O0`). It took the
util/aux/clus doctests down with segfaults; reverted, plain `std::map` kept. Not a WCT bug,
but a trap for anyone reaching for that container here.

**Not taken, with the numbers to decide later:** (U1) sharing the aggregate arrays' bytes
into the per-blob slices (`Dataset::slice(..., share=true)`) saves the 104 MB of payload
copies but needs a keep-alive for the aggregate on the tree root — a `Points` API addition.
(U3) the real fix is columnar: one `Array` per (point cloud, column) held by the tree with
per-node views, which removes the 2.5 M objects entirely (~560 MB of the 707) but changes
`local_pcs()` from a map of `Dataset` to a view type — a util redesign, cross-detector, the
owner's call.

## 13. T2 — the single-track dQ/dx fit without its per-row maps

`TrackFitting::dQ_dx_fit` (the STM tagger's per-segment fit) built, for every segment it
fitted, three `std::map<CoordReadout, pair<ChargeMeasurement, std::set<Coord2D>>>` over
**every** 2D cell of the loaded cluster (up to ~90k rows, ~4 heap nodes per row), then
`std::set` lookup tables and a wire index from them, then tore it all down; 22.5 % of
039349/7's samples, 35 % of that in red-black-tree walks and 11 % in map destruction. The
rows now live in flat per-plane tables (`DqdxRow`: key, measurement, and a `[c0,c1)` range
into one shared `Coord2D` vector) built by one pointer sort of the unordered charge source:

| change | identity argument |
|---|---|
| `build_dqdx_rows`: rows in `CoordReadout` order per plane, coords per row sorted+unique | the `std::map` iterated in `CoordReadout` order and each row's `std::set<Coord2D>` in `Coord2D` order; rows with no wires or a last-wire plane outside U/V/W were dropped by the old `switch` and are dropped here |
| response fill reads the same `PlaneRow` wire index (row index `n` = position in the plane table) | identical `RU/RV/RW` rows, identical `reg_flag` OR |
| `(wire,time)` membership tests (`set_UT.find`) → `binary_search` on sorted+unique vectors | same boolean |
| `fill_fitted_charge_2d(rows, coords, ...)` overload mirrors the map flavour line for line; both flavours end in the shared `record_cluster_fitted_charge_2d()` | same product map, same per-cluster snapshot |
| `get_wires_for_channel` memoised per (apa, channel), returns a const ref; cleared with the grouping in `reset_for_new_event` | geometry is fixed for the fitter's lifetime |
| multi-track fit keeps its maps but builds them in key order with an `end()` hint from the memo | unique keys ⇒ identical maps |
| `calculate_compact_matrix(_multi)`: the **dense** `n_3d × n_2d` double array `pair_values` (a 1000-point STM fit over a 90k-row cluster: 720 MB, zero-filled per call) → `(row·n_2d+col, value)` vector over the nonzeros in iteration order, **stable**-sorted, read as the **last** entry with the key | every read returns the same double: absent → 0.0 as the zero-filled array did, and a duplicated key → the last write, as the array did (see below) |

**The one gate failure of the round, and what it taught.** The first post binary passed the
busy set 28/28, SBND 201/201 and uBooNE 35/35 but failed the 120-event PDVD gate on ONE
archive: `tracking-stm.root` of 039253/3, tree `T_stm_eval` (the STM tagger's per-pass
evaluation record, `-stm-fit` only), rows 18–25 of 49 — `ratio1/ratio2/res_length/
ave_res_dqdx` of a pass that both binaries went on to reject; Bee, calib and every other
tree identical. Two reruns per library (`d28r2rb1/2`, `d28r2rp1/2`) were 4/4 identical
within each library, so this was a real binary difference, not the bistability of §7.
Bisecting it took three steps, each recorded under `/home/xqian/tmp/doc28/`:

1. util-only build (`lib_utilonly`, tag `d28r2ua`): identical to base 4/4 — the Array
   shrink is exact; the fitter change carries the difference.
2. Identical env-gated hashes in both binaries (`WCT_DQDX_AB`, builds `dump*P/B`, tags
   `d28r2d*`): every one of the 215 single-track fits agrees bit for bit on the row tables,
   data vectors, `RU/RV/RW`, `MU`, regularisation flags, `F`, `b`, `A`, the solution and `dx`;
   every one of the 215 `do_single_tracking` calls agrees on the trajectory, dQ, dx, the
   whole loaded charge table and `global_rb_map`. So T2 is exact, and a first guess —
   duplicate (row, col) entries from a wrapped channel taking a different one of two values
   in the sparse `pair_values` lookup — was wrong (`std::stable_sort` + last-duplicate read
   is kept anyway: it is what the dense array did).
3. Hashes inside `eval_stm_core_impl`: all eight differing evals have identical points, `L`,
   dQ/dx, kink and parameters and differ ONLY in `end_L`: base −2022.41 mm, post 2 mm. That
   eval has `kink_num = 417 = num_pts` (the `short_track` branch sets `kink_num = dQ.size()`),
   so `max_num = L.size()`; no bin qualified, `max_bin = max_num`, and the code reads
   `L[max_bin]` — **one past the end of the vector** (`TaggerCheckSTM.cxx`, the line
   `end_L = L[max_bin] + 0.2 * units::cm` after `if (max_bin == -1) max_bin = max_num;`).
   The value is whatever sits after the vector on the heap; my change moved the heap around
   it. Downstream, `res_length`/`ave_res_dqdx` for those rows are garbage in both binaries
   and the pass was rejected in both, so nothing else moved.

This is a pre-existing out-of-range read in the STM tagger, not a T2 defect, and it makes
`T_stm_eval` binary-dependent on any event whose short-track branch finds no peak bin. It
is **not fixed here**: a bounds guard changes that tree's values (and could change a pass
verdict where the garbage happened to pass), so it is a behaviour change for the owner to
decide (CLAUDE.md: report, do not fix in the same change; doc 92's `T_cluster`
uninitialised-flash read is the precedent). The 120-event gate below therefore reads
479/480 with this one archive explained to the byte; the busy-set, SBND and uBooNE gates
are clean. All gates in §15 are on the final binary (`lib_r2post5`, source identical to
`lib_r2post3`).

Effect (busy set, clean sequential pair, `MABC timing:` stage seconds; §16 for the totals):

| event | STM stage base → post | Neutrino stage | peak RSS (runner) |
|---|---|---|---|
| 039252/5 | 47.1 → 28.9 | 365.7 → 363.3 | 3.35 → 3.20 GB |
| 039253/11 | 45.7 → 27.5 | 164.7 → 153.0 | 3.76 → 3.19 |
| 039252/15 | 40.1 → 25.3 | 111.6 → 103.4 | 3.20 → 2.78 |
| 039349/7 | 34.7 → 20.5 | 85.1 → 84.0 | 2.03 → 1.92 |
| 039252/8 | 15.0 → 9.4 | 19.4 → 16.1 | 3.14 → 2.44 |
| 039253/15 | 14.5 → 10.1 | 49.4 → 48.1 | 2.32 → 1.89 |
| 039349/6 | 7.5 → 5.0 | 26.3 → 24.7 | 1.06 → 1.00 |

The STM stage is 1.5–1.7× faster on every event (`d28r2sb5` vs `d28r2sp5`, the final pair;
the first pair `d28r2sb2/sp2` read the same within noise); the neutrino stage moves 0–17 %
(its single-track fits and the multi-fit map build); the RSS column is under tcmalloc and
therefore only directionally meaningful (§8), but 7/7 down is the dense `pair_values`
gone. Note the base here is HEAD `8bf5b1bc`, whose doc 31 steiner-threshold sync changed
these events since round 1 (039253/15 is no longer BiCGSTAB-bound: 349.8 → 85.7 core-s).

## 14. Q/L-stage fast connectors on PDVD: wired, identical, not worth turning on

The SBND Q/L-tail flavors (docs 78/79: `ProtectOverclustering busy_num_threshold=200`,
`Deghost` on the `ctpc_fast` graph) are now reachable for PDVD: `po_fast` / `dg_fast` args
on `clus_per_apa` (and `dg_fast` on `clus_per_group`) in
`cfg/pgrapher/experiment/protodunevd/clus.jsonnet`, TLAs `ql_po_fast` / `ql_dg_fast` in
`pdvd/wct-clustering.jsonnet`, default **off** (ExamineBundles is disabled in the PDVD
group stage, so `eb_fast` has no site). Compiled-config proof with the runner's own TLAs
(`PDVD_CLUS_COMPILE_ONLY=1`, event 039252/8): knob-off JSON byte-identical to the
pre-edit compile; knob-on puts the keys on all 18 Deghost/ProtectOverclustering nodes.

Where the PDVD Q/L job's time goes (120 `d27fresh` events, `MABC timing:` sums, 1594 s):
`done` (writers) 15.3 %, **Deghost 14.8 %**, Separate 13.2 %, `loaded live` 9.1 %,
Connect1 7.4 %, ExtendLoop 7.0 %, Neutrino 4.4 %, **ProtectOverclustering 4.4 %**; worst
Deghost 7.8 s (039252/8). Knobs on vs off, same `lib_r2base`, five worst Deghost events
(`stm/gates/r28_ql_r2qlon_vs_r2qloff.txt`): **140/140 archives identical** (point tree,
27 Bee zips and the calib dump per event), and

| event | wall off/on | Deghost off/on | ProtectOverclustering off/on |
|---|---|---|---|
| 039252/8 | 74 / 72 | 8.4 / 6.9 | 5.6 / 3.7 |
| 039252/14 | 51 / 52 | 5.2 / 5.3 | 1.9 / 1.8 |
| 039252/2 | 48 / 48 | 5.4 / 5.4 | 1.9 / 1.9 |
| 039252/5 | 57 / 56 | 5.0 / 4.6 | 1.7 / 1.7 |
| 039253/5 | 54 / 54 | 5.1 / 5.1 | 1.7 / 1.7 |

The lazy path engages only on clusters with more than 200 components; PDVD's Deghost
seconds are spread over many small clusters (the readout-wide window again), so only the
one event with a busy cluster moves, by 3.4 s of 74. The knobs stay off; the SBND
production values are one TLA away if a future PDVD sample has the SBND tail shape.

## 15. Round-2 gates

All on pinned snapshots: `lib_r2base` (toolkit `8bf5b1bc`) vs `lib_r2post5` (the final
source state; `lib_r2post3` is the same source). Every arm's `/proc/<pid>/maps` was checked.

| gate | arms | result | record |
|---|---|---|---|
| PDVD busy set (7 events × bee, calib, tracking-pr, tracking-stm) | `d28r2sb5` / `d28r2sp5` | **28/28 identical** | `stm/gates/r28_d28r2sp5_vs_d28r2sb5.txt` |
| PDVD 120 events (480 archives) | `d28r2fb` / `d28r2fp5` | **478/480**; 039253/3 `tracking-pr.root` identical, `tracking-stm.root` differs in `T_stm_eval` rows 18–25 only — the `L[L.size()]` read of §13, deterministic per binary (reruns `d28r2rb1/2` = `d28r2rp1/2` 4/4) | `stm/gates/r28_d28r2fp5_vs_d28r2fb.txt` |
| PDVD Q/L fast flavors off vs on (5 events × point tree + 27 Bee zips + calib) | `r2qloff` / `r2qlon`, both `lib_r2base` | **140/140 identical** | `stm/gates/r28_ql_r2qlon_vs_r2qloff.txt` |
| PDVD Q/L compiled config, knobs off | runner compile-only, 039252/8 | byte-identical to the pre-edit compile; knobs on: keys on 18 nodes | §14 |
| SBND nuecc48 (48) + ncpi0 (19): Bee, calib, nusel | `doc25_r2base` / `doc25_r2post5` (d99fix Q/L, `dl_weights=''`) | **201/201 identical** | `stm/gates/r28_sbnd_compare_r2post5_vs_r2base.txt` |
| uBooNE 35 events, Bee zip + tagger ROOT | `sweep/doc25_r2base` / `doc25_r2post5b` | **35/35 zips, 35/35 tagger identical**; `repeat_check.sh 22 4` on 6805 under each lib: Bee identical 4/4, tagger 1 and 2 states (the doc 90 bistability) | `stm/gates/r28_ub_ab_check_r2post5b_vs_r2base.txt` |
| util-only bisect build | `d28r2ua` vs `d28r2fb` (039253/3) | 4/4 identical | `/home/xqian/tmp/doc28/split_gate_d28r2ua.txt` |

Doctests on the final build: util 276/276, aux 22/22, clus 277/277 (1 skipped). The 120-event
arm gives the arm-wide numbers: node core-s 8194 → 6935, stage sums STM 1834 → 1107 s,
Neutrino 4377 → 3950, CreateSteinerGraph 1480 → 1376 (the arms ran under different job
counts, so quote the clean pair of §16 for per-event factors).

## 16. Round-2 timing and memory, before/after

**Time** (busy set, `d28r2sb5` on `lib_r2base` vs `d28r2sp5` on `lib_r2post5`, each
sequential, side by side while the 120-event arms ran; `Timer:` node core-s of
`MultiAlgBlobClustering`; the quieter first pair `d28r2sb2/sp2` gave 1315.7 → 1187.3, 1.108×):

| event | base core-s | post core-s | speed-up |
|---|---|---|---|
| 039252/5 | 475.9 | 454.4 | 1.05× |
| 039253/11 | 250.2 | 220.2 | 1.14× |
| 039252/15 | 177.3 | 154.9 | 1.15× |
| 039349/7 | 137.9 | 121.9 | 1.13× |
| 039252/8 | 116.4 | 108.5 | 1.07× |
| 039253/15 | 83.7 | 78.1 | 1.07× |
| 039349/6 (median) | 41.7 | 37.9 | 1.10× |
| **sum** | **1283.2** | **1175.8** | **1.091×** |

Against round 1's 1.34× on this set this is the smaller step the §9 arithmetic predicted
(~10 % of an STM-heavy event); unlike round 1 it moves every event, including the median.

**Memory** (matched jemalloc pair of 039252/5, both arms with `PDVD_PR_RELEASE=1`; live
heap from the dump nearest each stage stamp):

| 039252/5, live heap at the stage stamp | base `d28r2hb2` | post `d28r2hp5` |
|---|---|---|
| after CreateSteinerGraph | 1262 MB | 1290 MB |
| after TaggerCheckSTM | 2290 MB | **1587 MB** (−703: the dense compact-matrix array of the last STM fit was live at the stamp) |
| end of TaggerCheckNeutrino | 2705 MB | 2616 MB |
| after `release_post_nu` | — (same-second dump) | 1656 MB |
| PrDisplayDump | 1733 MB | 1577 MB |
| peak of the sampled dumps (every 12th) | 2629 MB | **2033 MB** |
| loaded tree (`as_pctree`) | 676 MB | 650 MB |
| VmHWM (runner, jemalloc) | 2.94 GB | 2.77 GB |

Under tcmalloc on the busy set (§13 table) the runner's peak RSS is lower on 7/7 events,
by 0.1–0.7 GB, which is the dense `n_3d × n_2d` array no longer being allocated.

## 17. Round 2 — measured and not taken

- The `flat_map` store (§12) — a working idea blocked by the installed boost.
- U1/U3 tree designs (§12) — owner's call; U3 is the one that halves the plateau.
- `fill_fitted_charge_2d`'s remaining cost (6 % of 039349/7: a hash find, a
  `std::set<Cluster*>` and 1–3 map inserts per row, then a deep copy of the 90k-entry
  snapshot per fit): the copy could go if `get_fitted_charge_2d()` aliased the per-cluster
  snapshot; deferred — it changes a getter's lifetime contract.
- `update_dQ_dx_data` walks the fitter's whole `m_charge_data` per fit; not hot on the
  profiled events, left.
- The Q/L flavors' threshold (200 components) was not tuned for PDVD: the census says the
  time is not in one busy cluster, so no threshold rescues it.

## 18. Next steps after round 2 (as written then; the STM read is fixed and the neutrino stage re-examined in round 3, §19–§26)

1. U3 (columnar tree storage in `util`) is the only remaining lever on the plateau
   (~560 MB of 707 per event, every detector) — design in §12, owner decision.
2. The neutrino stage is now 60–80 % of every busy event and is fit numerics
   (BiCGSTAB to the iteration cap, §9): a build-level threading option, or a tolerance
   change *as a knob* judged on physics, are the only levers left there.
3. The batch input stall (§10 item 4) is unchanged: 23 s of a median event's wall under
   contention is the point-tree read; a zstd codec is runner-side work.

## 19. Round 3 — scope, repro, labels

The owner's three asks after round 2: fix the STM eval's out-of-range read that the round-2
gate exposed (§13, §15); explain why some PDVD events cost so much more than SBND's and take
what SBND's perf rounds can offer, confining any behaviour change to the busy events; doc,
commit, push. Base for every gate is the round-2 final binary (`lib_r2post5`, toolkit
`1a7e1b66`); the round-3 snapshots are `lib_r3fix` (STM fix only) and `lib_r3post` (STM fix +
the OFF-pass skip of §22). Every arm's `/proc/<pid>/maps` was checked against its snapshot.

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/pdvd
# where the neutrino stage goes, per candidate, from the round-2 arm's logs (sec 21)
python3 /home/xqian/tmp/doc28/r3/nu_census.py d28r2fp5
# CPU profiles of the five heaviest neutrino-stage events, round-2 binary (sec 21, sec 23)
LD_LIBRARY_PATH=/home/xqian/tmp/doc28/lib_r2post5:$LD_LIBRARY_PATH TAG=profpr3_039252_5 ./profile_pr.sh 039252 5 /home/xqian/tmp/doc28/r3/pr_039252_5.prof
google-pprof --text --focus=run_dual_chain_off_pass build/apps/wire-cell /home/xqian/tmp/doc28/r3/pr_039252_5.prof | head -3
# busy-set clean pair, the 120-event arm, the shared-detector arms (one launcher, 26 processes)
/home/xqian/tmp/doc28/r3/run_r3.sh          # d28r3sb (lib_r2post5) / d28r3sp (lib_r3post); d28r3fp; SBND/uB _r3post
python3 stm/gates/r28_gate.py d28r3sb d28r3sp --out stm/gates/r28_d28r3sp_vs_d28r3sb.txt
python3 stm/gates/r28_gate.py d28r2fp5 d28r3fp --out stm/gates/r28_d28r3fp_vs_d28r2fp5.txt
QL_SUFFIX=d99fix OLD_ARM=_r2post5 NEW_ARM=_r3post ./stm/gates/shared_gate.sh compare
(cd ../qlport/scripts && ./ab_check.sh doc25_r3post doc25_r2post5b)
python3 stm/perf/pr_perf_profile.py --tag d28r3fp     # docs/perf/pr_d28r3fp_{events,stages}.tsv
grep -l "end bin falls back" work/*_d28r3fp/wct_pr_*.log   # which events the STM fix touches
```

Toolkit commits (`apply-pointcloud`): `8c577c4b` (STM eval fallback), `a94ce32e` (dual-chain
OFF pass skipped without DL weights). wcp-porting-img: this doc, `stm/perf/nu_census.py`,
the gate records and perf TSVs named below.

## 20. The STM eval's out-of-range read — fixed

**Symptom.** Round 2's 120-event gate (§15) found one archive of 480 differing between two
binaries built from sources that a three-level hash bisect proved arithmetically identical:
039253/3 `tracking-stm.root`, `T_stm_eval` rows 18–25 of a rejected pass, with `end_L`
−2022.41 mm under the base library and 2 mm under the post library.

**Root cause.** `TaggerCheckSTM::eval_stm_core_impl` ends the dQ/dx comparison window at the
bin of maximal 5-point mean dQ/dx inside `[end_L − peak_range, end_L + 0.5 cm)`. When no bin
in that window carries positive dQ/dx (all fitted charges zero) `max_bin` stays −1 and the
fallback was `max_bin = max_num`. In the kink branch `max_num` is the kink index — a valid
point. In the no-kink branch `max_num = L.size()`, so the next line, `end_L = L[max_bin] +
0.2 cm`, read one element past the end of `L`. The toolkit reaches the no-kink branch more
often than the prototype did: the short-track reset (`TaggerCheckSTM.cxx:3692`) sets
`kink_num = dQ.size()`, which the `kink_num >= num_pts` test treats as "no kink".
`detect_proton` has the same window search with no fallback at all — `max_bin == −1` would
index `L[-1]`; it has never fired on a gated event (the new DEBUG line would show it).

**Why it hid.** The value read is whatever the allocator left after the vector's storage:
deterministic for one binary and one heap layout (four reruns per library agreed), so every
same-binary repeat check passed, and the pass it sat in was rejected on other grounds so the
STM verdict never moved. Only a gate between two different binaries could see it, and round
2's was the first such gate on that event.

**Fix** (toolkit `8c577c4b`, `clus/src/TaggerCheckSTM.cxx`, both sites). When no window bin
qualifies, the end bin is `min(max_num, L.size() − 1)`: the kink point when there is a kink,
the track's last point when there is none — which is what `end_L = L.back()` at the top of
the function already says the no-kink end is. A DEBUG line names every fire
(`eval_stm: no positive dQ/dx window, end bin falls back to …`). No knob: the previous
behaviour was an undefined read, not a legacy path anyone can want back. **NOT
bit-identical where it fires**; everywhere else it is a no-op by construction (the branch
only runs when `max_bin == −1`).

**Verification.** Doctests clus 277/277 (1 skipped) on both builds (`lib_r3fix`, `lib_r3post`). Gates in §24: on the 120 PDVD events the
fallback fired on two events — 039253/3 (8 evaluations, end bin 416 of 417, `kink_num` 417 = the short-track reset) and 039349/33 (4 evaluations, 628 of 629, `kink_num` 629); the only archives that differ between `d28r2fp5` and
`d28r3fp` are exactly those two events' `tracking-stm.root`, and inside them only `T_stm_eval` (rows 18–25 and 13–16: `ks1`/`ks2` were 0 with `res_length` 273 / 400 mm and a garbage `ave_res_dqdx`, now real KS values with an empty residual, as an end at the last point implies); `T_stm_pass` and every other tree identical, so no verdict moved. SBND and uBooNE arms identical (201/201, 35/35 zips, 34/35 tagger with 6805 the one difference — the doc 90 bistable event, 35/35 against the partial `doc25_r2post5` arm),
i.e. the branch never fired there. The reproducing test is the event itself: the same
hash gate that exposed the bug now shows the rows carrying `L.back() + 0.2 cm`.

## 21. Why a PDVD event costs more than an SBND event

Round 1 (§2) answered this at the stage level: the same code at the same per-unit cost, and
PDVD does 5 ν candidates and 60–80 mains per event where SBND does one. Round 3 looked one
level down, into the neutrino stage, which after round 2 is 57 % of the arm (3950 of 6935
node core-s over 120 events; 583 candidates). Two things there are PDVD-specific.

**The candidates are long cosmics, and the cost is superlinear in their length.** From the
round-2 arm's logs (`nu_census.py`; per candidate = the selected main's `initial PR` +
`other_clusters PR` + the small stages, OFF pass excluded — see below):

| main length | candidates | production PR, sum | share | median per candidate | max |
|---|---|---|---|---|---|
| < 100 cm | 188 | 69 s | 4 % | 0.2 s | 5.1 s |
| 100–200 cm | 119 | 116 s | 7 % | 0.7 s | 10.5 s |
| 200–400 cm | 178 | 498 s | 31 % | 1.7 s | 37.3 s |
| 400–600 cm | 72 | 542 s | 33 % | 5.9 s | 30.4 s |
| 600–1000 cm | 26 | 398 s | 25 % | 12.8 s | 52.0 s |

98 candidates longer than 4 m — through-going cosmic muons up to 9.2 m (039252/14 cluster
113) — are 58 % of the production PR time; the median PDVD candidate costs 0.2–1.7 s. SBND's
candidates on the same binary (the `_r2post5` gate arms, 136 candidates, `dl_weights=''`):
median length 87 cm, maximum 264 cm, and **7.0 s** median for a 100–200 cm candidate against
PDVD's 0.7 s — a νe or NCπ0 shower breaks into many segments and refits each; a straight
cosmic does not. So per centimetre PDVD's PR is ten times cheaper than SBND's; what makes
the PDVD event slow is five candidates per event and the 4–9 m lengths the readout-wide
window admits.

**The dual-chain OFF pass ran on every candidate and was never read.** PDVD's PR TLA file
carries SBND's production operating point for doc pr/112's dual chain (`dl_vtx_dual_chain =
true`, mode `snap`, transfer on, D = 2 cm — `pdvd/wct-pr-perevt.jsonnet:2839`). The OFF pass
is a full exclusion-free PR of the main cluster on its own graph and fitter; its product is a
`DualChainHint` whose only consumer is `determine_overall_main_vertex_DL`
(`TaggerCheckNeutrino.cxx:2955-2962`: the snap transfer, the voxels/union re-rank and the
scoreboard's `dual_chain` block all live inside it), and that function runs only `if
(!m_dl_weights.empty())`. PDVD has no DL vertex net (`dl_weights = ''`, doc 25 §4). So on
PDVD every OFF pass — 583 of them, **2263 s = 57 % of the neutrino stage and 33 % of the
arm** — computed a vertex that nothing read; no PDVD calib dump has ever contained a
`dual_chain` block (grep over the 120 dumps: 0). The pass is also more expensive than the
production pass it shadows because exclusion is off: 1.1–4.7× (039252/5 cluster 84, 6.6 m:
OFF 262 s, production 52 s; the profile of that event is 62 % OFF pass, and 57 % of the
whole event is the BiCGSTAB solve of the exclusion-free multi-fit running to its iteration
cap). On SBND the same pass is consumed (DL weights configured) and costs 1.58× the visit
(pr/112 §11); on the SBND gate arms, which run `dl_weights=''` (M4), it was likewise
unconsumed — 882 s of their 1809 s of candidate time.

## 22. The OFF pass is skipped when nothing can consume it

Toolkit `a94ce32e`, `clus/src/TaggerCheckNeutrino.cxx`: with `dl_vtx_dual_chain` on and
`dl_weights` empty the pass is not run and one INFO line per candidate says why
(`dual_chain: OFF pass skipped for cluster N -- no DL weights, so nothing consumes the
hint`). Output-identical by construction — the hint's every reader is inside a function that
does not run — and pr/112 §11.2's probe gate had already shown the pass leaks nothing into
production (graph, fitter, flags all its own; 96/96 + 200/200). No knob: there is no output
the legacy path produced that this removes. SBND production, with its DL weights, is
untouched; the SBND gate arms (`dl_weights=''`) exercise the skip and are the identity
proof on 67 more events (§24).

Whether PDVD *should* have a dual chain is a separate, physics question for the owner: the
design's value is an exclusion-free suggestion for the neutrino vertex, and wiring the snap
into the traditional `determine_overall_main_vertex` path would make it live on a no-DL
detector — at the 2263 s this round removed, or at the fraction of it a length-gated
version would cost. Not done here.

## 23. SBND-style busy-event gating: what is left to gate, and its ceiling

After §22 the neutrino stage is ~1690 s of a ~4670 s arm (36 %), CreateSteinerGraph 1376 s
(29 %), TaggerCheckSTM 1107 s (24 %). Inside the production PR of the busy candidates
(profiles of the five heaviest events, samples outside `run_dual_chain_off_pass`):

| event (profile s) | ν stage outside the OFF pass | `dQ_dx_multi_fit` | of which `BiCGSTAB` | `form_point_association` | `exclusion_closest_2d_dis` | STM stage | steiner |
|---|---|---|---|---|---|---|---|
| 039252/5 (439) | 17.8 % | 9.9 % | 7.5 % | 2.0 % | 1.7 % | 6.0 % | 9.5 % |
| 039252/14 (261) | 17.0 % | 6.2 % | 3.5 % | 2.8 % | 3.6 % | 6.6 % | 11.3 % |
| 039253/11 (193) | 37.6 % | 10.3 % | 0.6 % | 4.9 % | 10.1 % | 12.2 % | 14.7 % |
| 039349/72 (147) | 26.0 % | 7.3 % | 0.3 % | 4.0 % | 3.7 % | 6.3 % | 6.9 % |
| 039252/10 (122) | 34.7 % | 7.1 % | 0.4 % | 5.8 % | 12.3 % | 4.8 % | 7.6 % |

(percent of the whole profiled event, which still includes its OFF pass — 50–62 % of these
five; `pprof --focus=<f> --ignore=run_dual_chain_off_pass`, files `/home/xqian/tmp/doc28/r3/pprof_rest.txt`.)

The two exact levers with any weight are the association step (`form_point_association` +
`exclusion_closest_2d_dis`, the kd-2D queries and per-point containers) and the multi-fit's
response build (`cal_gaus_integral` / `erf`, the row maps) — both already through SBND's
pr/98 perf round, each worth a few percent of an event for a delicate rewrite. The solver
itself, outside the OFF pass, is 7.5 % of 039252/5 — the round-1 §9 candidate for
a busy-gated tolerance knob (`n_3d > N` only) is now small.

What SBND's tail-gating idea (doc 76 §9.3) maps onto here is not an approximation but a
scope gate: 98 candidates over 4 m are 58 % of the remaining PR time, and every one of them
is a through-going cosmic that the neutrino chain fits, breaks and refits as if it might
hold a vertex. A `nu_max_main_length_cm` knob (default 0 = today) that leaves such
candidates without a PR pass would be the "limit the change to the busy events" shape —
but it removes their PF/kine rows, i.e. it changes what PDVD's products *are*, and doc 25's
STM/Michel work reads those rows. Owner decision; the arithmetic is in the §21 table
(≥ 400 cm: −940 s of 4670; ≥ 600 cm: −400 s, 26 candidates).

## 24. Round-3 gates

| gate | arms | result | record |
|---|---|---|---|
| PDVD busy set (7 events × bee, calib, tracking-pr, tracking-stm) | `d28r3sb` (`lib_r2post5`) / `d28r3sp` (`lib_r3post`) | **28/28 identical** | `stm/gates/r28_d28r3sp_vs_d28r3sb.txt` |
| PDVD 120 events (480 archives) | `d28r2fp5` / `d28r3fp` | **477/480**: the two `tracking-stm.root` of §20's two events, `T_stm_eval` only; every Bee zip, calib dump and `tracking-pr.root` identical, so the OFF-pass skip changed nothing downstream | `stm/gates/r28_d28r3fp_vs_d28r2fp5.txt` |
| SBND nuecc48 (48) + ncpi0 (19): Bee, calib, nusel; `dl_weights=''` so the skip fires on every candidate | `doc25_r2post5` / `doc25_r3post` | 201/201 | `stm/gates/r28_sbnd_compare_r3post_vs_r2post5.txt` |
| uBooNE 35 events, Bee zip + tagger ROOT | `sweep/doc25_r2post5b` / `doc25_r3post` | 35/35 zips, 34/35 tagger with 6805 the one difference — the doc 90 bistable event, 35/35 against the partial `doc25_r2post5` arm | `stm/gates/r28_ub_ab_check_r3post_vs_r2post5b.txt` |

Doctests on the final build: clus 277/277 (1 skipped) on both builds (`lib_r3fix`, `lib_r3post`) (the STM fix and the skip touch no other
package). Freshness proofs: `libWireCellClus.so` newer than the last source edit before
each snapshot.

## 25. Round-3 timing

Busy set, `d28r3sb` (`lib_r2post5`) vs `d28r3sp` (`lib_r3post`), each sequential, side by
side while the 120-event arm and the SBND arms ran; `Timer:` node core-s:

| event | base core-s | post core-s | speed-up | TaggerCheckNeutrino s | peak RSS GB |
|---|---|---|---|---|---|
| 039252/5 | 434.7 | 183.2 | 2.37× | 350.2 → 86.6 | 3.20 → 3.21 |
| 039253/11 | 223.6 | 148.8 | 1.50× | 154.1 → 81.8 | 3.19 → 3.19 |
| 039252/15 | 148.8 | 97.0 | 1.53× | 97.1 → 46.0 | 2.79 → 2.78 |
| 039349/7 | 126.4 | 68.6 | 1.84× | 85.4 → 29.4 | 1.92 → 1.91 |
| 039252/8 | 93.9 | 88.1 | 1.07× | 13.6 → 6.8 | 2.45 → 2.45 |
| 039253/15 | 71.8 | 50.7 | 1.42× | 44.5 → 20.0 | 1.90 → 1.74 |
| 039349/6 (median) | 36.2 | 23.9 | 1.51× | 23.5 → 11.6 | 1.13 → 1.13 |
| **sum** | **1135.3** | **660.3** | **1.719×** | | |

TSVs: `docs/perf/pr_d28r3{sb,sp,fp}_{events,stages}.tsv`. The 039252/8 event is the one whose
time is the steiner build and the strict connector, not the neutrino stage.

120-event arm: node core-s 6935 → 4742; TaggerCheckNeutrino 3950 → 1721 s.
The removed time is the OFF pass exactly (predicted 2263 s from the base arm's `[dual-off]`
timers). Memory is unchanged in kind — the OFF pass's graph and fitter were released at its
end — but the peak on the events where the OFF pass held the largest fit is lower
(3 of 7 busy events lower, none higher by more than 0.01 GB; the arm's maximum 3.26 GB (039252/16) unchanged; median event 39.2 → 30.4 core-s).

## 26. Next steps after round 3

1. Owner: should PDVD have a dual chain at all? If yes, the snap belongs in the traditional
   vertex path (§22); if no, `dl_vtx_dual_chain = false` in `wct-pr-perevt.jsonnet` makes
   the config say what the code now does.
2. Owner: the length gate of §23 — a scope knob, not a perf lever.
3. U3 (columnar tree storage, §12) remains the memory lever; the per-candidate exact levers
   of §23 are the remaining CPU ones, each a few percent.
4. `detect_proton`'s fallback (§20) is instrumented; if the DEBUG line ever fires on a gated
   event, that event needs a look.

## Milestone log

- 2026-09-03 — arm profile, three CPU profiles, one heap profile, SBND comparison;
  S1/S2/T1/D1/M2/M3/M4 shipped; gates §7; doc + memory note.
- 2026-09-03 (round 2) — tree census + Array footprint (util `66831770`), T2 + compact-matrix
  dense array + channel→wires memo (`34d0a5f5`), PDVD Q/L fast-flavor knobs, off
  (`1a7e1b66`); the 039253/3 bisect found the STM eval's out-of-range read; gates §15.
- 2026-09-03 (round 3) — STM eval fallback fixed (`8c577c4b`); neutrino-stage census by
  candidate length, five CPU profiles; the unconsumed dual-chain OFF pass skipped
  (`a94ce32e`); gates §24, timing §25.

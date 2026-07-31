# Doc pr/11 — full PR chain at population scale: scores, runtime, memory, perf

**Status: DONE.** All 1071 events run on one binary
(`apply-pointcloud` @ `289d78e4`) with **1071/1071 `rc=0`**.

**Scope change mid-round, on the owner's instruction.** This started as a pure
measurement round ("no toolkit C++ or jsonnet is changed"). The first pass
crashed or hung on **76/1071 events (7.1%)**, and the owner asked for those to
be fixed rather than merely reported. Sec 6 therefore ships **five crash/hang
fixes plus two shower point-cloud correctness defects** (commit `289d78e4`);
every number in secs 2-5 comes from the fixed binary. Classes A-E are proven
byte-identical on events that already completed; defects F and G do change
shower point clouds and are flagged **NOT bit-identical** (sec 6.6/6.7).

Goal: run the FULL 13-stage SBND PR chain — `switch_scope, unmerge_bundle,
unmerge_assoc, steiner, fiducialutils, tagger_check_tgm, tagger_check_stm,
tagger_check_fc, tagger_check_neutrino, numu_bdt_scorer, nue_bdt_scorer,
tracking_visitor, tagger_output` — over three samples (1000-event MCP2025C
data, 48-event Lynn nueCC data, 23 MC events) and record, for every event: (1)
tagger verdict + `nue_score`/`numu_score`/cosmic-side quantities, (2) wall
time, (3) memory, (4) a profiling round naming the bottleneck.

**Headline results**

| | |
|---|---|
| Completion | 1071/1071 `rc=0` (was 995/1071 before the sec-6 fixes) |
| Verdicts | 53.3% ν-candidate, 28.6% cosmic-tagged, 11.1% no-beam-flash, 7.0% no-bundle |
| Scored (`T_tagger` row) | 629/1071 = 58.7% |
| νe BDT actually evaluated | **193/1071 = 18.0%** — the rest carry a `−15` *not-evaluated sentinel*, not a score (sec 2.3) |
| νe BDT separation | 32 `+4.301` saturations on the 48 hand-picked νe events, **0** on 1000 generic data events |
| Bottleneck #1 | **TMVA weight-file loading — 49% of campaign wall.** One process per event ⇒ 5704 s of configure vs 6013 s of physics; `UbooneNueBDTScorer` alone 4.85 s/event. A *harness* property, not a toolkit defect (sec 3.3, 5.3) |
| Bottleneck #2 | **SCN (DL) vertex inference: ~1.16 s fixed per evaluated event**, independent of event size — 629 s of the census, 69% of `TaggerCheckNeutrino` (sec 5.2) |
| Both are fixed overheads | the genuinely algorithmic part of the chain is ~0.7 s on a median event — the smallest of the three (sec 5.4) |
| DL vertex accepted | 82.4% of events reaching the vertex stage (75.7% data, 97.8% νe) |
| Peak RSS | 1.13–1.60 GB per process, essentially flat across all 1071 events |
| Dead branches | `cosmict_score` ≡ 0 **and** `cosmict_10_score` ≡ 0.700 — both structurally dead (sec 2.4) |

No prior driver ran this chain past `tagger_check_fc` at any scale (docs
pr/2-3 smoke-tested single events only); this doc is the first population
run.

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# All four samples, one binary (e154d50b), 20-way concurrency.
# PR_JOBS=20 is safe HERE only because each event is pinned to a single
# thread (OMP_NUM_THREADS=MKL_NUM_THREADS=1 inside the driver) and peaks at
# ~1.53 GB RSS: 20 x 1 core on a 64-core box, ~31 GB against 209 GB free.
# This is a deliberate, owner-approved departure from the CLAUDE.md M5
# rule-of-thumb of ~6, which assumes the usual multi-threaded wire-cell job.
export PR_JOBS=20 PR_TIMEOUT=1800

mkdir -p work-nuecc48-pr11v3 work-r1qlmc-pr11v3 work-r2mc-pr11v3 work-mcp1kall-pr11v3
./run_pr_chain_batch.sh work-nuecc48-nuf    work-nuecc48-pr11v3  data   # 48 events
./run_pr_chain_batch.sh work-r1ql-first10   work-r1qlmc-pr11v3   sim    # 10 (doc 67)
./run_pr_chain_batch.sh work-r2patrec-f1    work-r2mc-pr11v3     sim    # 13
./run_pr_chain_batch.sh work-mcp1kall-d59k  work-mcp1kall-pr11v3 data   # 1000

# per-sample score tables, then merge
python3 pr_scores_table.py --root work-nuecc48-pr11v3   --sample nuecc48 --no-header --out /tmp/nuecc48.tsv
python3 pr_scores_table.py --root work-r1qlmc-pr11v3    --sample r1qlmc  --no-header --out /tmp/r1qlmc.tsv
python3 pr_scores_table.py --root work-r2mc-pr11v3      --sample r2mc    --no-header --out /tmp/r2mc.tsv
python3 pr_scores_table.py --root work-mcp1kall-pr11v3  --sample mcp1k   --no-header --out /tmp/mcp1k.tsv
python3 pr_scores_table.py --merge /tmp/nuecc48.tsv /tmp/r1qlmc.tsv /tmp/r2mc.tsv /tmp/mcp1k.tsv \
    --out docs/pr/11_scores-table.tsv --summary

# label breakdown (sec 2.1), score percentiles (2.2/2.5), wall/core (3.1),
# peak RSS (4) -- every distribution table in this doc comes from this one call
python3 pr11_analyze_census.py docs/pr/11_scores-table.tsv

# stage-cost breakdown (population, free at -L debug -- perf=true is already
# the SBND default for MABC and TaggerCheckNeutrino)
python3 pr_stage_totals.py --root work-mcp1kall-pr11v3

# --- sec 5: DL-vertex acceptance arm (248 events at TRACE) -------------------
# The DL decision lines are SPDLOG_LOGGER_TRACE, so they do NOT appear in the
# -L debug census logs.  200 mcp1k events stratified over the core_s ordering
# (deterministic every-Nth pick, no RNG) + all 48 nueCC48.
python3 pr11_pick_events.py docs/pr/11_scores-table.tsv dlrate > /tmp/dlrate_picks.sh
. /tmp/dlrate_picks.sh
SBND_WCT_LOGLEVEL=trace PR_JOBS=20 ./run_pr_chain_batch.sh work-mcp1kall-d59k /tmp/dlrate_mcp1k data $MCP1K_DL
SBND_WCT_LOGLEVEL=trace PR_JOBS=20 ./run_pr_chain_batch.sh work-nuecc48-nuf   /tmp/dlrate_nuecc data $NUECC_DL
grep -l "switching to DL vertex" /tmp/dlrate_*/pr_evt*/wct_pr_evt*.log | wc -l   # accepted
grep -ho "scn_inference=[0-9.]*ms" /tmp/dlrate_*/pr_evt*/wct_pr_evt*.log         # cost

# --- sec 3/4: sequential latency + peak-RSS arm (30 events, PR_JOBS=1) -------
# Events are ranked by core_s, NOT wall_s: under 20-way concurrency wall is
# contention-dominated (events with identical 6.23 s wall span core 0.73-3.63 s),
# so wall is the wrong key for picking events whose latency we want.
python3 pr11_pick_events.py docs/pr/11_scores-table.tsv phase4 > /tmp/phase4_picks.sh
. /tmp/phase4_picks.sh
PR_JOBS=1 ./run_pr_chain_batch.sh work-mcp1kall-d59k /tmp/seq_mcp1k data $HEAVY_MCP1K $MEDIAN_MCP1K
PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-nuf   /tmp/seq_nuecc data $HEAVY_NUECC48

# seq-vs-census comparison (sec 3.2) and the per-process configure cost (3.3)
python3 pr11_seq_compare.py /tmp/seq_table.tsv docs/pr/11_scores-table.tsv
python3 pr11_config_cost.py work-mcp1kall-pr11v3

# --- sec 2.3: nue_score buckets (br_filled read straight from T_tagger) ------
python3 pr11_br_filled_census.py .

# --- sec 5.3: CPU profile.  DL_WEIGHTS= is REQUIRED: libpython + ------------
# libtcmalloc_and_profiler co-preloaded deadlocks (see sec 5.3 caveat 1).
DL_WEIGHTS= EVT=397480 ROOT=work-mcp1kall-d59k OUTDIR=/tmp/prof_geo \
    ./profile_pr11.sh /tmp/prof_geo/cpu.prof
google-pprof --text --cum $(which wire-cell) /tmp/prof_geo/cpu.prof | head -32
```

Freshness proof (CLAUDE.md M1): `local/lib/libWireCellClus.so` /
`libWireCellRoot.so` md5 recorded before **and after every arm**, not just
once per campaign — a concurrent session was live in this tree and advanced
`apply-pointcloud` by six commits while this round ran, so a mid-arm `wcbuild`
by that session would silently void a timing measurement with nothing to show
for it. All arms in secs 2-5 ran on
`e154d50b3ea87959addba8b3ef4e56fe` / `1e0908bee5c2369265886552dea70bfb`,
verified unchanged at both ends of each arm.

## 1. What was run, and the DL-engagement proof

- Binary: toolkit `apply-pointcloud` @ `289d78e4` = `135e5f38` **plus the five
  crash/hang fixes and the two shower point-cloud fixes of sec 6**
  (`e154d50b3ea87959addba8b3ef4e56fe libWireCellClus.so`,
  `1e0908bee5c2369265886552dea70bfb libWireCellRoot.so`).
  Every number in secs 2-5 comes from this single binary; the pre-fix binary
  `8f91c486` produced only the failure census of sec 6, and none of its
  timings are quoted here. md5 re-checked before and after each arm (sec 3) —
  a concurrent session was active in this tree and moved `apply-pointcloud`
  six commits during the campaign, so this fence is not a formality.
- Every event runs the SAME production knob set `run_nusel_evt.sh` uses for
  TGM/STM/FC (chord+rescue+rescue-chord, fvz 5/fvzi 3/fvx 2.5/fvy 3, mip
  56000, unmerge+unmerge-assoc real, doc-63/66 STM guards all ON), plus the 5
  neutrino-PR stages (doc pr/2-3), plus the **SCN (DL) neutrino vertex ON**
  (owner decision this round — the production default since `e3d46c91`;
  every prior PR-chain measurement used the geometric vertex only).
  `-stm-fit`/`stm_magnify` is left OFF (doc pr/3: `save_stm_fit` gates only a
  diagnostic dump, never the STM verdict).
- `OMP_NUM_THREADS=MKL_NUM_THREADS=1`: the DL inference is multithreaded by
  default. A pinned-vs-unpinned probe on the same event (172230) showed
  5.74 s wall / 5.60 s core (pinned) vs 5.33 s wall / 9.54 s core (unpinned,
  default thread count). Pinned everywhere in this doc so wall time reads as
  single-thread latency, comparable across every event and arm.
- **DL-engagement proof.** The DL decision lines are `SPDLOG_LOGGER_TRACE`
  (`NeutrinoVertexFinder.cxx:3374`), so they are **absent from the `-L debug`
  census logs by construction** — their absence there is not evidence of
  anything. Engagement is proven two ways:
  1. *Negative, over the whole population:* a failed SCN import emits a WARN
     (`determine_overall_main_vertex_DL: DL vertex failed: SCN_Vertex: import
     failed for module ...: undefined symbol: PyTuple_Type` — what a
     **missing** `LD_PRELOAD=libpython...so.1.0 RTLD_GLOBAL` looks like,
     confirmed against the prior-session artifact
     `work-nuecc48-prsmoke2/nupr_evt172230_dl_importfail`). Every event's log
     is grepped for it (`dl_warn` column): **0/1071**.
  2. *Positive, on a sized subset:* a dedicated 248-event trace arm (sec 5)
     shows `scn_inference=1150.8 ms` (median) actually executing.

  **Do NOT use `cw_ratio = core_s/wall_s` as the DL discriminator.** The plan
  for this round proposed flagging rows below 1.3 on the theory that DL is
  multithreaded (core ≈ 1.8 × wall). That threshold was derived from an
  *unpinned* probe and is unreachable here: this campaign pins
  `OMP_NUM_THREADS=MKL_NUM_THREADS=1`, which makes `core_s ≤ wall_s`
  identically, so `cw_ratio < 1.3` on **1068/1068** rows regardless of what DL
  did (measured max 1.266). The rule is **retired** — read the trace arm
  instead. A reader who computes the ratio and takes `below1.3 = 1068` at face
  value would wrongly conclude the DL vertex failed on every event.
  (`cw_ratio` is not a column in `11_scores-table.tsv`; `pr_scores_table.py`
  never implemented it. It exists only in the round's plan and in
  `pr11_analyze_census.py`, which recomputes it from `wall_s`/`core_s` — this
  paragraph is the reason it is reported there but used nowhere.)
- **A `nu_evaluated` flag, not the nusel bundle table, decides whether scores
  are meaningful.** `unmerge_bundle`/`unmerge_assoc` (both production-ON) run
  *before* `tagger_check_neutrino` and can split a bundle into new,
  PR-job-internal cluster idents that never appear in `nusel_extract.py`'s
  (pre-PR) bundle census. Seen directly on evt 285201
  (`work-mcp1kall-d59k`): the nusel table lists exactly one in-window bundle
  (10.0 cm, tagged STM); the PR log's
  `TaggerCheckNeutrino: selected main cluster 1 (t0 1.444 us, L 2.5 cm, 0
  associated)` is a *different*, unmerge-split 2.5 cm cluster with no row in
  the nusel table at all — and it got real, independent BDT scores
  (`numu_score -1.94`). `pr_scores_table.py` therefore never joins a score
  row to a nusel `main_id`; it reports scores only when the PR log's own
  `selected main cluster` line fired (`nu_evaluated=1`), and separately
  reports the event's cosmic/nu verdict (`event_label`) from the independent
  flag-based nusel rows. See the script's module docstring for the full
  reasoning; this finding is worth the owner's attention as a real gap in how
  `nusel_extract.py`'s bundle accounting relates to the neutrino stage's own
  (post-unmerge) selection.

## 2. Score + verdict census

Per-event rows: `docs/pr/11_scores-table.tsv` (1071 rows; `1000 + 48 + 10 + 13`
reconciles, `rc=0` on all 1071).

### 2.1 Event verdicts

`event_label` is the event-level verdict from the flag-based nusel rows (see
sec 1 on why it is *not* joined to a score row).

| sample | n | nu-candidate | cosmic-tagged | no-beam-flash | no-bundle |
|---|---:|---:|---:|---:|---:|
| mcp1k (data)   | 1000 | 516 (51.6%) | 297 (29.7%) | 117 (11.7%) | 70 (7.0%) |
| nuecc48 (data) |   48 |  45 (93.8%) |   3 ( 6.2%) |   0 ( 0.0%) |  0 (0.0%) |
| r1qlmc (MC)    |   10 |   5 (50.0%) |   3 (30.0%) |   2 (20.0%) |  0 (0.0%) |
| r2mc (MC)      |   13 |   5 (38.5%) |   3 (23.1%) |   0 ( 0.0%) |  5 (38.5%) |
| **ALL**        | 1071 | 571 (53.3%) | 306 (28.6%) | 119 (11.1%) | 75 (7.0%) |

`nu_skip_cosmic` is ON (production), so cosmic-tagged events legitimately
produce no scores; that is a result, not a gap. On the nueCC48 sample — 48
hand-picked νe candidates — 93.8% survive as ν candidates, which is the
expected sanity check.

**A `TaggerInfo` row exists for 629/1071 events (58.7%)** — 572/1000 mcp1k,
47/48 nueCC48, 5/10 r1qlmc, 5/13 r2mc. Two independent counts agree exactly:
`nu_evaluated=1` (from the PR log's `selected main cluster` line) and the
presence of a non-empty `T_tagger` in `tracking-pr.root` both give 629.

### 2.2 `numu_score` (uBooNE-trained, ranking only)

On `nu_evaluated==1`. These are **uBooNE-trained weights**
(`sbnd/clus.jsonnet:1145-1152`) applied to SBND — uncalibrated, so the scale
is a ranking, not a probability.

| sample | n | min | p50 | p90 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| mcp1k   | 572 | −3.793 | −0.151 | 3.290 |  4.301 |  0.204 |
| nuecc48 |  47 | −2.289 | −0.597 | 0.646 |  1.608 | −0.532 |
| r1qlmc  |   5 |  0.708 |  1.223 | 3.797 |  3.797 |  1.670 |
| r2mc    |   5 | −1.715 |  0.000 | 3.116 |  3.116 |  0.102 |
| **ALL** | 629 | −3.793 | −0.217 | 3.116 |  4.301 |  0.160 |

`±4.301` is the XGBoost log-odds clamp (`val1` clipped to `±0.9999`,
`UbooneNueBDTScorer.cxx:1922`), already seen in doc pr/3 — a saturated score,
not a measured extreme.

Note the νµ score is *lower* on the νe sample (p50 −0.597) than on generic
data (−0.151), which is the correct direction for a νe-enriched selection.

### 2.3 `nue_score` — **most events never reach the BDT**

**This is the finding that a naive percentile table hides.** `nue_score = −15`
is not a low score: it is `default_val` written at
`UbooneNueBDTScorer.cxx:1655/1925` whenever `ti.br_filled != 1`, i.e. the νe
tagger never filled its BDT variable block and **the BDT never ran**. Quoting
a median of `−15` as if it were a score would be wrong. `br_filled` is itself
a `T_tagger` branch (`UbooneTaggerOutputVisitor.cxx:889`), so the buckets
below are counted directly, not inferred from the sentinel value.

| sample | n | no `T_tagger` | `−15` not filled | BDT ran | of which −4.301 | +4.301 | interior |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcp1k   | 1000 | 428 | 426 | **146 (14.6%)** | 132 |  0 | 14 |
| nuecc48 |   48 |   1 |   4 |  **43 (89.6%)** |   0 | 32 | 11 |
| r1qlmc  |   10 |   5 |   4 |   **1 (10.0%)** |   1 |  0 |  0 |
| r2mc    |   13 |   8 |   2 |   **3 (23.1%)** |   3 |  0 |  0 |
| **ALL** | 1071 | 442 | 436 | **193 (18.0%)** | 136 | 32 | 25 |

So the honest denominator for `nue_score` is **193, not 629 and not 1071**.

**The νe BDT does separate the two data samples cleanly.** On 1000 generic
MCP2025C data events it produced **zero** `+4.301` (νe-like) saturations; on
the 48 hand-picked νe candidates it produced **32**. Unsaturated values:
mcp1k n=14, median −1.765; nueCC48 n=11, median −0.338. This is the first
population-scale evidence that the ported νe BDT behaves sensibly on SBND,
and it is a ranking statement only — the weights are uBooNE's.

### 2.4 Two structurally-dead cosmic quantities

- **`cosmict_score` ≡ 0 on all 629 rows.** Nothing in the tree writes it —
  only the struct default (`NeutrinoTaggerInfo.h:1363`) and the branch
  (`UbooneTaggerOutputVisitor.cxx:1061`). Cause: the un-ported `cal_bdts()`
  TMVA combination, which also leaves ~20 other `*_score` branches at 0.
  Reported as a porting gap, **not** papered over with a proxy.
- **`cosmict_10_score` ≡ 0.700 on all 629 rows** — and 0.700 is exactly the
  `default_val` argument at `UbooneNumuBDTScorer.cxx:412`. `cal_cosmict_10_bdt`
  returns it unchanged whenever `ti.cosmict_10_length` is empty
  (`UbooneNumuBDTScorer.cxx:285`), i.e. always: the upstream-dirt cluster
  vectors are never populated in SBND. A second dead cosmic quantity, same
  status as `cosmict_score`.

The live cosmic-side quantities are therefore the flags, not the scores:
`cosmic_flag` = 1 on 627/629 rows, `cosmict_flag` = 1 on 132/629. **The
event-level cosmic verdict today is the TGM/STM/LM label (sec 2.1), not a
score.**

### 2.5 `kine_reco_Enu`

| sample | n | p50 (MeV) | p90 | max |
|---|---:|---:|---:|---:|
| mcp1k   | 572 |  421.6 | 1010.8 | 2746.1 |
| nuecc48 |  47 | 1644.2 | 2551.8 | 3902.5 |
| r1qlmc  |   5 |  592.8 | 1034.3 | 1034.3 |
| r2mc    |   5 |  628.1 |  900.4 |  900.4 |

The νe sample sits ~4× higher than generic data, as expected for selected
νe CC events over a cosmic-dominated population.

### 2.6 The doc pr/3 spot-check does **not** reproduce — and cannot

The plan asked to reproduce doc pr/3's `numu_score` −1.992 / +0.277 and
`kine_reco_Enu` 1765.9 / 1470.1 MeV on evts 172230 / 444187. This round gives
**+0.121 / +0.889** and **1694.3 / 1368.9 MeV**. That is *not* a DL-vs-geometric
delta and must not be reported as one: at least six commits landed between doc
pr/3's binary and this one, including the pr/10 recombination round
(`6d0396a2`, `db625c81`) which changes `kine_reco_Enu` **by construction**, the
SSM ×10 unit fixes (`1628328e`), and the MIP-scale/`dl_vtx` threads
(`2ebb0177`, `135e5f38`). The comparison is confounded on both axes at once, so
it closes no verification item. **Isolating the DL-vs-geometric delta needs a
same-binary geometric arm, which this round did not run** — recorded as gap G3
in sec 7.

## 3. Runtime

Two arms, and they answer different questions. **The census arm
(`PR_JOBS=20`) is throughput under load and carries no latency claim; the
sequential arm (`PR_JOBS=1`, 30 events) carries every latency number here.**

### 3.1 Census arm — throughput, 1071 events at 20-way

`wall_s`/`core_s` are the WCT `Timer: Total` line (graph execution only — see
3.3, this is *not* the process wall).

| sample | n | min | p50 | p90 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcp1k   | 998 | 0.81 | 6.24 | 8.43 | 11.13 | 51.58 | 6.03 |
| nuecc48 |  47 | 4.47 | 10.85 | 21.65 | 45.56 | 45.56 | 12.36 |
| r1qlmc  |  10 | 4.32 | 4.67 | 5.70 | 5.92 | 5.92 | 4.99 |
| r2mc    |  13 | 5.33 | 6.73 | 7.11 | 7.28 | 7.28 | 6.58 |
| **ALL** | 1068 | 0.81 | 6.29 | 8.71 | 15.48 | 51.58 | 6.30 |

Whole-campaign throughput: 1071 events in ~11 min wall at 20-way.

`n=1068`, not 1071: **three events lose their `Timer` line to WCT's known
non-atomic multi-line log write** (`project_wct_log_line_tearing`) — e.g.
evt 52672's line reads `Timer: Total 11.81[07:18:25.338] D [ pgraph ] ...`,
truncated mid-word and interleaved with an earlier Pgrapher line. All three
are `rc=0` with a normal `maxrss`; this is a logging artifact at 0.28%, not a
run failure, and the partial values (11.81 / 5.480 / 3.027 s wall) are
consistent with their samples. A fourth instance (evt 239794) appeared in the
sequential arm.

### 3.2 Sequential arm — latency, 30 events at `PR_JOBS=1`

Events ranked by **`core_s`, not `wall_s`** (sec 7 explains why: at 20-way,
events with identical 6.23 s wall span 0.73–3.63 s of core).

| group | n | seq wall p50 | census wall p50 | census/seq ratio (min / p50 / max) |
|---|---:|---:|---:|---|
| mcp1k heaviest+median | 20 | 5.07 s | 8.44 s | 1.17 / **1.96** / 4.14 |
| nueCC48 heaviest      | 10 | 14.51 s | 22.26 s | 1.13 / **1.63** / 1.96 |
| **all 30**            | 30 | 6.82 s | 12.64 s | 1.13 / **1.76** / 4.14 |

So **20-way concurrency inflates per-event wall by ~1.8× at the median and up
to 4.1×** on light events, where queueing dominates. Sequential wall ≈ core to
within 1% on every event (e.g. evt 407170: 44.06 wall / 43.91 core), as
expected under single-thread pinning.

Latency extremes, sequential: heaviest data event 44.06 s (evt 407170),
heaviest νe event 40.42 s (evt 256587), **median data event ≈ 1.85 s**
(the ten median-`core_s` events span 1.78–1.97 s).

Note `core_s` *itself* inflates under load — the ten median events measure
2.72–2.74 s core at 20-way but 1.68–1.82 s sequentially (~1.5×), which is
memory-bandwidth and cache contention, not queueing. Even "CPU time" is not
concurrency-independent here.

### 3.3 The number that matters most: **half the wall is not physics**

The WCT `Timer` covers graph execution only. `timecmd.py`'s process wall is
several seconds larger on *every* event — evt 397480: `Timer` 1.890 s, process
7 s. That gap is **component configuration**, and because
`run_pr_chain_batch.sh` runs **one wire-cell process per event**, it is paid
1071 times.

Measured from the `-L debug` `configuring/configured component` timestamps
across all 1000 mcp1k events:

| component | p50 | mean | **total over 1000 events** |
|---|---:|---:|---:|
| `UbooneNueBDTScorer`  | 4.849 s | 4.848 s | **4848.5 s** |
| `UbooneNumuBDTScorer` | 0.550 s | 0.562 s | 561.7 s |
| `WireSchemaFile`      | 0.234 s | 0.239 s | 239.1 s |
| everything else       | — | — | ~54 s |
| **configure total**   | **5.686 s** | 5.704 s | **5703.9 s** |
| *(graph execution, for comparison)* | *6.24 s* | *6.03 s* | *6013.0 s* |

**5704 s of configuration against 6013 s of physics: 49% of the campaign's
wall time was spent re-parsing the same BDT weight files 1000 times.**
`UbooneNueBDTScorer` alone is 85% of that and ~40% of total campaign wall.

For a *median* data event the ratio is worse still: **5.0 s of configure
against 1.85 s of physics — 73% overhead.** It only looks benign on the
heaviest events (evt 280972: 4.84 s configure vs 20.5 s physics, 19%).

**Caveat, stated up front:** this is a property of *this harness*, not of the
PR chain. One process per event is what makes a per-event A/B and a
per-event crash census possible. A production art/LArSoft job configures once
and streams many events, amortising the 5.7 s to ~0. So this is **not** a
defect to fix in the toolkit — it is the dominant cost of *this measurement
workflow*, and it is why the sec 3.1/3.2 tables quote the `Timer` line rather
than process wall.

## 4. Memory

Peak RSS is `timecmd.py`'s `RUSAGE_CHILDREN` maximum, i.e. **per process**.

| sample | n | min | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| mcp1k   | 1000 | 1.127 | 1.456 | 1.459 | 1.461 | 1.597 |
| nuecc48 |   48 | 1.137 | 1.458 | 1.460 | 1.479 | 1.479 |
| r1qlmc  |   10 | 1.132 | 1.138 | 1.459 | 1.460 | 1.460 |
| r2mc    |   13 | 1.131 | 1.137 | 1.458 | 1.459 | 1.459 |
| **ALL** | 1071 | **1.127** | **1.456** | 1.459 | 1.461 | **1.597** |

(GB. `1 GB = 1048576 kB`.)

**The distribution is remarkably flat: a 1.13–1.60 GB range across 1071
events spanning a 40× spread in processing time.** Peak RSS is essentially
decoupled from event complexity — it is dominated by the fixed cost of the
loaded BDT models and geometry, not by the point clouds. The two modes visible
in the table (≈1.13 GB and ≈1.46 GB) are exactly the events that never reach
`tagger_check_neutrino` versus those that do.

**Concurrency-independence, verified rather than assumed.** The same 30 events
measured at `PR_JOBS=1` and `PR_JOBS=20` agree to ~0.003 GB
(evt 407170: 1.142 vs 1.141; evt 269774: 1.458 vs 1.460; evt 280972: 1.558 vs
1.556). So 20-way concurrency costs ~31 GB total, matching the headroom
argument in the Repro block.

For sizing: **~1.6 GB per concurrent job** is a safe figure, and it does not
grow with event size.

*(Before the sec-6 fixes, class D reached a 224 GB peak on evt 399118 — that
event now peaks at 1.53 GB, inside the distribution above.)*

## 5. Profile / bottleneck

### 5.1 Where the time goes, by stage (free, from the census logs)

`perf: true` is already the SBND default for MABC and `TaggerCheckNeutrino`,
so the whole 1000-event census carries a substage breakdown at `-L debug`
(`pr_stage_totals.py`). Top of the table, totals over 1000 events:

| step | total | n_evt | median | max |
|---|---:|---:|---:|---:|
| `MABC:TaggerCheckNeutrino:pr` | 911.7 s | 1000 | 1136 ms | 21270 ms |
| ↳ `TaggerCheckNeutrino:visit() TOTAL` | 911.6 s | 572 | 1385 ms | 21267 ms |
| ↳ **`TaggerCheckNeutrino:overall main vertex`** | **629.1 s** | 572 | **1152 ms** | 1427 ms |
| ↳ `TaggerCheckNeutrino:main_cluster initial PR` | 183.6 s | 572 | 123 ms | 17358 ms |
| `MABC:CreateSteinerGraph:pr` | 139.4 s | 1000 | 61 ms | 22853 ms |
| `MABC:done` | 112.2 s | 975 | 109 ms | 588 ms |
| `MABC:SbndPrMagnifyTrackingVisitor:pr` | 66.4 s | 1000 | 63 ms | 131 ms |
| `MABC:UbooneNumuBDTScorer:pr` | 54.7 s | 1000 | 1.1 ms | 659 ms |
| `MABC:UbooneNueBDTScorer:pr` | **0.5 s** | 1000 | 0.09 ms | 5 ms |

**`overall main vertex` is 69% of `TaggerCheckNeutrino` and the single
largest stage in the chain.** Note the shape: median 1152 ms with min 0.01 ms
and max only 1427 ms — a near-constant cost, not one that scales with event
size. That is the signature of a fixed-size neural-network inference.

Note also that *scoring* is nearly free (`UbooneNueBDTScorer:pr` = 0.5 s over
1000 events); the expensive part of the BDTs is loading them (sec 3.3), not
evaluating them.

### 5.2 Bottleneck #1 — SCN (DL) vertex inference, ~1.16 s fixed per event

Confirmed directly by the 248-event trace arm (the DL timing lines are
`SPDLOG_LOGGER_TRACE`, so they do not appear in the census logs):

```
determine_overall_main_vertex_DL timing: collect_pc=0.009ms
  scn_inference=966.619ms selection=0.101ms examine_direction=0.000ms
  proton_tagging=0.000ms cleanup_long_muon=0.000ms TOTAL=966.732ms
```

| | mcp1k (200) | nueCC48 (48) |
|---|---|---|
| events reaching the vertex stage | 107 | 46 |
| `scn_inference` p50 | 1150.8 ms | 1182.6 ms |
| range | 982.3 – 1349.0 ms | 1042.6 – 1319.5 ms |
| total | 124.1 s | 54.2 s |

`scn_inference` accounts for essentially all of the 629 s `overall main
vertex` total (572 × 1.15 s ≈ 658 s). **It is a flat per-event cost with a
1.37× spread between the fastest and slowest event in a sample whose
processing times span 100×** — it does not scale with event complexity, so it
dominates cheap events and is negligible on expensive ones.

**Is that cost being used?** Yes, mostly — this needed measuring because an
initial n=2 look happened to catch two rejections and suggested otherwise:

| | accepted | rejected | never reached vertex stage |
|---|---:|---:|---:|
| mcp1k (200)   | 81 (75.7% of 107) | 26 | 93 |
| nueCC48 (48)  | 45 (97.8% of 46)  |  1 |  2 |
| **combined**  | **126 (82.4% of 153)** | 27 | 95 |

Rejections are `rerank rejected (best_score=… < threshold=4.0000), staying
with traditional vertex` — i.e. `dl_vtx_min_accept_score`, whose default is
4.0. When accepted, the DL vertex usually moves the vertex only slightly
(`snap_dis` p50 0.52 cm) but sometimes a lot (p90 5.67 cm, max 31.6 cm).

Acceptance is much higher on the νe sample (97.8%) than on generic data
(75.7%), which is the expected direction: νe CC events have well-formed
neutrino vertices for the SCN to lock onto.

**Not tuned here.** `dl_vtx_min_accept_score` is an existing knob's default;
changing it would alter production output unconditionally (CLAUDE.md §5
rule 1) and is the owner's call. Reported, not touched.

### 5.3 Bottleneck #2 — TMVA weight-file XML parsing, confirmed by profile

gperftools CPU profile of a **median** data event (397480), 1300 samples.
Cumulative:

```
84.8%  WireCell::Main::initialize
81.0%    TMVA::Reader::BookMVA
72.9%      WireCell::Root::UbooneNueBDTScorer::init_readers
59.9%      TMVA::MethodBase::ReadStateFromFile
33.7%        TMVA::MethodBDT::ReadWeightsFromXML
33.0%          TXMLEngine::ParseFile
30.2%            TXMLEngine::ReadNode       (flat 10.5%)
 8.1%      WireCell::Root::UbooneNumuBDTScorer::init_readers
11.7%  WireCell::Pgraph::Pgrapher::execute   <-- the actual physics
```

Flat top: `TXMLEngine::ReadNode` 10.5%, `__strcmp_evex` 7.3%,
`__dynamic_cast` 6.5%, `std::istream::get` 5.8% — all XML tokenising.

This is an **independent confirmation of sec 3.3 by a different instrument**:
the log-timestamp method said 5.0 s configure vs 1.9 s physics on this event
(73%/27%); the profiler says 84.8% vs 11.7% of CPU samples. Same conclusion,
two methods.

**Profiling caveats, both material:**
1. **This is the geometric arm** (`DL_WEIGHTS=`). The DL arm could not be
   profiled: `LD_PRELOAD` carrying both libpython (needed `RTLD_GLOBAL` for
   the SCN import) and `libtcmalloc_and_profiler` **deadlocks** — the process
   sat alive at ~0% CPU with a frozen log for 10 minutes and had to be
   killed. `profile_pr11.sh`'s header already anticipated this and prescribes
   exactly this fallback. The DL cost is therefore quoted from the trace arm
   (sec 5.2), which needs no profiler, and is *excluded* from these
   percentages. Adding it back, **using the sequential arm's numbers only**
   (evt 397480: 5.00 s configure, 1.89 s `Timer` wall with DL on): configure
   is 5.00/(5.00+1.89) ≈ **73%** of the event, not 85%. That 73% is the same
   figure sec 3.3 reports for a median event by the log-timestamp method — the
   two agree. Do **not** compute this from the census `core_s` of 2.73 s: that
   is the contention-inflated value sec 3.2 explicitly warns against quoting,
   and it yields a spuriously low ~62%.
2. Profiled under tcmalloc (production preloads it), per CLAUDE.md M17, with
   the config pre-compiled by `wcsonnet` and **no** `setarch -R`.

### 5.4 What this means

For **this per-event harness**, the ranked costs on a median data event are:

| | cost | scales with event size? |
|---|---:|---|
| TMVA weight-file loading (configure) | ~5.0 s | no — fixed |
| SCN vertex inference | ~1.16 s | no — fixed |
| everything else (the actual PR chain) | ~0.7 s | **yes** |

Both dominant costs are **fixed per-process/per-event overheads, not
algorithmic hot spots**, and the genuinely algorithmic part of the chain is
the smallest of the three on a typical event. The two large-max stages
(`main_cluster initial PR` 17.4 s, `CreateSteinerGraph` 22.9 s) are what make
the *tail* expensive, and they are the right targets if tail latency ever
matters — but they are ~123 ms and ~61 ms at the median.

No optimisation is attempted here (CLAUDE.md §5 tie-breaker: a profile finding
in a measurement round gets named, not fixed). The actionable items, in order:
1. If the PR chain is ever run per-event at scale again, amortise the
   configure step — one process over many events, not one per event.
2. `UbooneNueBDTScorer::init_readers` at 4.85 s is worth a look on its own
   (it is 8.6× `UbooneNumuBDTScorer` for a comparable model count).
3. `dl_vtx_min_accept_score=4.0` discards 24% of a 1.16 s inference on data —
   an owner decision, not a bug.

## 6. Failure census: five crash/hang classes and two correctness defects

The first pass over the 1071 events (binary `8f91c486`, pre-fix) failed on
**76 events = 7.1%**: 73/1000 MCP2025C, 2/48 nueCC48, 1/13 r2patrec, 0/10
r1qlmatch. Every one of the 76 fell into **five** classes, each traced to a
specific line. All five are now fixed; the verification is in sec 6.6.

| class | events | symptom | rc |
|---|---|---|---|
| A | 1 | infinite loop, 100% CPU, byte-flat RSS | (none — killed after 8h17m) |
| B | 69 | `RuntimeError: Steiner point cloud 'steiner_pc' not found` | 1 |
| C | 1 | `std::out_of_range: map::at` | 250 (SIGABRT) |
| D | 4 | `std::bad_alloc`, one at a **224 GB** peak RSS | 250 (SIGABRT) |
| E | 1 | `runtime_error: do_single_tracking: inconsistent vector sizes` | 250 (SIGABRT) |

`rc=250`/`241`/`124` are `timecmd.py`'s encoding of a fatal signal N as
256−N (SIGABRT / SIGTERM / timeout).

**A note on how class E was nearly missed.** The first census binned failures
by grepping `stdout.log` for the three signatures then known, and 3 events
matched none of them. That was an artefact of the driver, not of the events:
`run_pr_chain_batch.sh` ran `nusel_extract.py` unconditionally, and on a
crashed event it hit a zero-byte `mabc-pr.zip` and appended a
`zipfile.BadZipFile` traceback that buried the real error at the tail of the
file. The driver now skips extraction when wire-cell's rc is non-zero (and
takes a `PR_TIMEOUT`, default 3600 s, so a class-A hang can never again stall
a batch on its last slot). With that masking removed, class E stood out
immediately.

### 6.1 Class A — unbounded `flag_continue` loop (evt 352365)

Found live: the 1000-event batch sat on its last slot for **8 h 17 m** while
`wire-cell` burned one core at 99.9% with `VmRSS` byte-flat at 1451340 kB
across a 20 s window, `syscw=1656`, and no log line since
`MABC timing: start clustering`. Flat RSS + 1.00 s/s utime + no I/O is an
infinite loop, not a slow event — and the pctree is *median*-sized (2.67 MB
vs a 2.31 MB sample median), so it is pathology, not scale.

gdb on the busy thread:

```
TaggerCheckNeutrino::visit                    TaggerCheckNeutrino.cxx:603
→ shower_clustering_with_nv                   NeutrinoShowerClustering.cxx:3201
→ shower_clustering_with_nv_from_vertices     NeutrinoShowerClustering.cxx:1157
→ PR::segment_get_closest_point               PRSegmentFunctions.cxx:120
→ nanoflann knn
```

**Root cause.** `Shower::add_segment()` returns `void` and silently no-ops
when the segment's descriptor is invalid — `TrajectoryView::add_segment`
(`clus/src/PRTrajectoryView.cxx:52`) returns `false`, and the `Shower`
wrapper (`clus/src/PRShower.cxx:141`) discards it. But the caller sets
`flag_continue = true` unconditionally. So the segment never enters
`map_segment_in_shower`, the next pass re-selects it, and the loop never
terminates. Nothing accumulates (a set-insert of nothing), which is exactly
why RSS was flat.

**Fix.** A termination guard at both `flag_continue` loops
(`shower_clustering_with_nv_from_main_cluster` :595-629 and
`shower_clustering_with_nv_from_vertices` :1142-1178): capture
`map_segment_in_shower.size()` before `update_shower_maps` and stop if a pass
registered nothing, with a WARN. The guard is *exact* rather than heuristic:
`Shower::fill_maps()` is `return *this;`, `edges()` returns `m_edges`
directly, and `add_vertex` touches `m_nodes` only — so a vertex-only add
cannot grow the segment count, and the size rises by exactly one per genuine
add.

Proof of mechanism (not merely of timing) — the WARN fires on the rerun:

```
W [clus.NeutrinoPattern] shower_clustering_with_nv_from_vertices: a pass
requested >=1 add_segment but registered none (9 segments mapped, unchanged);
stopping to avoid an unbounded loop
```

evt 352365: **8 h 17 m → 13 s.**

### 6.2 Class B — steiner consumer does not honour the producer’s contract (69 events)

`CreateSteinerGraph` *deliberately* leaves a cluster without steiner products
when `create_steiner_tree` finds fewer than 2 terminals — the run log carries
`create_steiner_tree: only 0 steiner terminal(s) found (need >=2), returning
empty graph` (`SteinerGrapher.cxx:38`) followed by
`create_steiner_tree produced no steiner_graph for assoc 47, skipping
transfer` (`CreateSteinerGraph.cxx:226`), whose comment states "all consumers
guard on that".

They do not. `Cluster::kd_steiner_knn` → `ensure_steiner_kd_cache` →
`build_steiner_kd_cache` **raises** `RuntimeError` when the PC is absent
(`Facade_Cluster.cxx:3821`). The exception's own stacktrace tag names the
consumer exactly:

```
build_steiner_kd_cache        Facade_Cluster.cxx:3821
← kd_steiner_knn              Facade_Cluster.cxx:3885
← examine_main_vertices       NeutrinoPatternBase.cxx:2076
← determine_overall_main_vertex NeutrinoVertexFinder.cxx:3763
← TaggerCheckNeutrino::visit  TaggerCheckNeutrino.cxx:553
```

The give-away is that both call sites already wrap the result in
`if (!knn.empty())` — the author expected "no steiner answer" to be a
skippable non-answer, and the throw defeats that intent.

**Fix.** Guard the two `examine_main_vertices` sites (:2076, :2092) with
`main_cluster->has_pc("steiner_pc")`, letting the existing `!knn.empty()`
blocks fall through. Deliberately *narrow*: the ~25 other steiner consumers
keep the loud throw, and 3 of them (`MyFCN.cxx:358`, `:407`,
`NeutrinoPatternBase.cxx:88-89`) index `[0]` without an empty check, so a
blanket API-level "return empty instead of throw" would convert a clean
exception into UB there. Scope was decided empirically, not by preference:
all 68 complete with the narrow guard, so no site beyond these two is
reached.

### 6.3 Class C — unguarded `contained_by()` (evt 49951)

```
Grouping::fastgeom (apa=-1, face=1)   Facade_Grouping.cxx:741  → m_anodes.at(-1)
← convert_3Dpoint_time_ch             Facade_Grouping.cxx:760
← examine_vertices_1p                 NeutrinoStructureExaminer.cxx:1305
```

A fit point lying outside every detector volume makes
`dv->contained_by(p)` return an invalid `WirePlaneId` (`apa() == -1`), and
`fastgeom` does `m_anodes.at(-1)` — `std::out_of_range: map::at`. (Note
`const size_t key = apa*2 + face` also wraps to a huge value, so the cache
short-circuit above it never fires.)

**Fix.** Skip the point. This is the file's own idiom, already applied to the
*vertex* points 120 lines earlier at `:1185-1188` — the inner loop over
`sg1->fits()` simply omits it.

### 6.4 Class D — self-append blows the heap to 224 GB (3 events)

```
examine_showers                              NeutrinoShowerClustering.cxx:2317
→ Shower::complete_structure_with_start_segment PRShower.cxx:299
→ Shower::add_segment                        PRShower.cxx:172
→ DynamicPointCloud::add_points              DynamicPointCloud.cxx:112/117
→ DPCBatch::append                           DynamicPointCloud.cxx:41  → bad_alloc
```

**Root cause.** `dpcloud(name, ptr)` stores the shared_ptr **by reference**
(`m_dpcs[name] = dpc_ptr`, `PRCommon.h:78`). So the `if (!shower_dpc)` branch
does not copy the first segment's cloud into the shower — it *aliases* the
shower's cloud to that segment's cloud. When the same segment comes round
again, `shower_dpc` and `seg_dpc` are the same object and
`shower_dpc->add_points(*seg_dpc)` appends a `DPCBatch` **to itself**:
`DPCBatch::append` does `x.insert(x.end(), other.x.begin(), other.x.end())`
on the very same vectors. The size doubles per pass — hence 224 GB.

**Fix.** Skip the merge when `shower_dpc == seg_dpc` (pointer equality) at
both merge sites (`PRShower.cxx` `add_segment` and the duplicated block in
`complete_structure_with_start_segment`): the points are already present, so
the correct merge is a no-op. evt 399118: **224 GB bad_alloc → 1.53 GB, 10 s.**

The guard is placed at the two `PRShower` sites rather than inside
`DynamicPointCloud::add_points` on purpose — a guard in `add_points` would
mask self-append for every caller in the codebase and hide the underlying
aliasing defect (sec 7).

### 6.5 Class E — fatal consistency check on a legitimately-degenerate fit (evt 386354)

```
TaggerCheckNeutrino::visit           TaggerCheckNeutrino.cxx (find_proto_vertex)
→ find_proto_vertex                  NeutrinoPatternBase.cxx:1570
→ init_first_segment                 NeutrinoPatternBase.cxx:431
→ TrackFitting::do_single_tracking   TrackFitting.cxx:8589   → throw
```

`do_single_tracking` ends by asserting that `dQ/dx/pu/pv/pw/pt/reduced_chi2`
all match `fine_tracking_path.size()`, and **throws** an uncaught
`std::runtime_error` otherwise. Doc 60 already met a version of this (evt
278794) and added a `pts.size() <= 1` guard at `:8517` — but that guards the
*input*. The 2nd-pass projection loop drops points on its own at `:8533`
(`if (test_wpid.apa()==-1) continue;` — the same "point outside every
detector volume" condition as class C), so the path can fall to one entry
even when `pts.size() > 1`; `dQ_dx_fit`/`dQ_dx_fill` then bail at their own
`fine_tracking_path.size() <= 1` guard, leaving the charge vectors empty
against a length-1 geometry. The instrumented rerun prints exactly that:

```
W [clus.TrackFitting] TrackFitting::do_single_tracking: inconsistent
fit-output sizes (path=1 dQ=0 dx=0 pu=1 pv=1 pw=1 pt=1 chi2=0); dropping
this segment's fit
```

**Fix.** Make the check non-fatal: WARN, clear every output vector, return —
the contract doc 60 itself described ("the prototype carries the empty dQ and
lets its caller drop the track", `PR3DCluster_pattern_recognition.h:262`).
Both callers already cope: `init_first_segment` tests `fine_path.size() > 1`
and reads only `.front()`/`.back()` behind `!vec.empty()`
(`NeutrinoPatternBase.cxx:449-466`), and `TaggerCheckSTM` tests
`fits().size() > 1`. Clearing is what makes the early return safe — it forces
those filters to fire rather than leaving a half-populated fit behind.

Scoping note: this is the one fix whose guard is not a size/validity test but
a *replacement of a throw*. That is deliberate — it is entered only where the
old code aborted the process, so it cannot alter any fit that previously
succeeded. Guarding on `ptss.size() <= 1` instead would have been the more
obvious patch but is **not** equivalent, and the difference was verified in
code rather than assumed:

1. `do_single_tracking` clears `dQ`/`dx`/`reduced_chi2` on entry
   (`TrackFitting.cxx:8241-8243`).
2. With `ptss.size() == 0`: `npoints == 0`; `dQ_dx_fill` returns at its
   `fine_tracking_path.size() <= 1` guard (`:5612`) *without* refilling, so
   those stay empty; `pu..pt` were cleared and got no pushes. Every size is
   therefore 0 == `npoints`, and the old consistency check **passes** — no
   throw.
3. Execution then reaches `segment->fits(segment_fits)` with an **empty**
   vector, i.e. it actively clears the segment's fits.

A `ptss.size() <= 1` guard would have returned early and left the segment's
previous fits intact — observably different on events that complete today.

### 6.6 Verification

Binary lineage — three builds, each gated separately:

| md5 `libWireCellClus.so` | contents |
|---|---|
| `8f91c486` | pre-fix baseline (the 1071-event first pass) |
| `7420c767` | classes A–D |
| `5c53f7d0` | + class E |
| `e154d50b` | + defects F/G (sec 6.7) — **the binary this doc's census runs on** |

`libWireCellRoot.so` stays `1e0908be` throughout: no `root/` file was touched.

| gate | result |
|---|---|
| `wcdoctest-clus` | 49/49 test cases, 565/565 assertions — on every build above |
| freshness proof (M1) | installed libs newer than the newest source edit before each arm |
| **no-op A/B for classes A–E**, 30 size-spread events drawn from the 927 that already succeeded (264 KB → 9.1 MB pctree), `work-mcp1kall-ab30fix` vs `work-mcp1kall-pr11` | `hash_archive.py` on `mabc-pr.zip` + `pctree-pr-evt*.tar.gz`: **60/60 content-identical, 0 DIFF**; `pr_scores_table.py` rows (timing/RSS columns excluded): **30/30 identical** |
| **intentional-change A/B for defects F/G** | sec 6.7 — `pctree` 30/30 identical; `mabc-pr.zip` 10/30 changed; **1/30** score row changed |
| all 76 previously-failing events rerun on `e154d50b` | **73/73** (MCP2025C) + 2/2 (nueCC48) + 1/1 (r2patrec) complete, rc=0 |

The A–E A/B is the gate that licenses those five fixes: each only adds a
branch that fires where the old code hung, threw, or aborted, so on any event
that completed before, the code path is unchanged — and the 60/60 + 30/30
result is the evidence, not the argument. Defects F/G are deliberately
outside that guarantee and are gated by attribution instead (sec 6.7).

### 6.7 Two further defects behind class D — confirmed by instrumentation, FIXED

Tracing class D exposed two defects that are *not* confined to the crashing
events — they are on the normal path for every shower. They were initially
reported unfixed (this doc's earlier sec 7.0) because they change output
unconditionally. The owner asked for them to be investigated and, if
confirmed, fixed; both were confirmed by temporary runtime instrumentation
and are now fixed **default ON, no knob**, shipped together.

**Method.** Temporary `SPDLOG_LOGGER_WARN` probes (tagged `TEMP-DPCDIAG`,
since removed) were added at the three cloud-seeding branches, at both merge
branches, and at the per-segment charge consumer, printing shower id, segment
id, cloud *pointer* and point counts. Run on SBND nueCC evt 172230.

**Defect F — the shower's cloud IS the start segment's cloud (aliasing).**
`dpcloud(name, ptr)` stores the shared_ptr verbatim (`m_dpcs[name] = dpc_ptr`,
`clus/inc/WireCellClus/PRCommon.h:78`), so the `if (!shower_dpc)` seeding
branch does not copy — it aliases. The instrumentation is unambiguous:

```
ALIAS set_start_segment shower=0 seg=... cloud=0x5555650b4230 npts=44
...
cloud 0x5555650b4230 (shower 0 start-seg own cloud, started 44 pts)
    -> grew to 786 pts via 67 merges
merges targeting an aliased cloud: 67 / 67
```

Every merge into every shower wrote into a segment-owned object; the start
segment ended up holding the entire shower (44 → 786 points, 17.9×). That
matters because consumers read a *segment's* cloud as a proximity mask —
`NeutrinoEnergyReco.cxx:266-267` (`cal_kine_charge(SegmentPtr)`),
`PRSegmentFunctions.cxx`, `MultiAlgBlobClustering.cxx:861`.

*Fix*: `clone_dpc()` — seed the shower with a deep copy. `DynamicPointCloud`
holds a `unique_ptr` k-d tree so it is not copy-constructible; the clone is
built through the public API (`get_wpid_params()` + `add_points(const
DynamicPointCloud&)`).

**Defect G — every segment merged twice.**
`complete_structure_with_start_segment` called `this->add_segment(seg)` at
`PRShower.cxx:299`, which already performs the fit/associate merge internally
(`:163-188`), and then repeated the merge by hand at `:303-326`. All 7 callers
use the default cloud names and `add_segment`'s defaults are the same two
names, so the second merge was a pure duplicate — and a *worse* one, since it
omitted the `merge_wpid_params` call. Instrumented: **22/22** second merges
duplicated a first. With defect F this is what turned a duplicate into a
doubling and produced the 224 GB peak.

*Fix*: thread `cloud_name_fit`/`cloud_name_associate` into the `add_segment`
call and delete the hand-rolled block, so each segment merges exactly once
(and now with `merge_wpid_params`).

**Measured effect (30 size-spread events, attributed by building each fix
alone).**

| variant | Bee `shower_track` changed | score rows changed |
|---|---|---|
| G alone (single merge) | 9/30 | **0/30** |
| F on top of G (un-aliased) | 10/30 | **1/30** |

- **G is physics-neutral by construction**, and the measurement confirms it:
  `kine_charge_from_maps` (`NeutrinoEnergyReco.cxx:48`) loops over the *charge
  map* — one entry per (time, channel) hit — and uses the point cloud only for
  a nearest-neighbour distance test. Duplicate points at identical coordinates
  give the same nearest-neighbour distance, so the charge sum is unchanged.
  What it does change is size: evt 166870's shower cloud drops 5012 → 2208
  points (−56%), evt 350121 5139 → 4921.
- **F does change physics**: evt 166870 `nue_score` **2.6070938 → 2.4717388**.
  The new number is the correct one — it comes from a segment cloud holding
  only that segment's points instead of the whole shower's.

`pctree-pr-evt*.tar.gz` is **identical in all 30** under both variants; only
`mabc-pr.zip`'s `data/0/0-shower_track-global.json` moves.

**Status: NOT bit-identical — needs revalidation.** This is the one change in
this doc that alters reconstruction output on events that already succeeded.

## 7. Caveats and gaps

### 7.0 (moved) — the two shower-point-cloud defects are confirmed and FIXED

These were first reported here as "found but not fixed". They were then
investigated with runtime instrumentation, confirmed, and fixed with the
owner's explicit approval (default ON, no knob, shipped together). See
**sec 6.7**.

- **`cosmict_score` is structurally dead — always 0.** Nothing in the tree
  writes it: `clus/inc/WireCellClus/NeutrinoTaggerInfo.h:1363` only sets the
  struct default, and `root/src/UbooneTaggerOutputVisitor.cxx:1061` only
  copies that default into the branch. The live cosmic-side quantities are
  the **flags only** — `cosmic_flag` (struct default 1, i.e. "cosmic" unless
  the neutrino stage overwrites it) and `cosmict_flag`. **Corrected from an
  earlier draft:** `cosmict_10_score` is *not* live either. It is ≡ 0.700 on
  all 629 scored events, and 0.700 is exactly the `default_val` argument at
  `root/src/UbooneNumuBDTScorer.cxx:412`, returned unchanged by
  `cal_cosmict_10_bdt` whenever `ti.cosmict_10_length` is empty (`:285`) —
  which it always is in SBND, because the upstream-dirt cluster vectors are
  never populated. Two dead cosmic scores, not one (sec 2.4). Cause of the
  first: the un-ported `cal_bdts()`
  TMVA combination (`root/src/UbooneNueBDTScorer.cxx` header comment `:37`),
  which also leaves ~20 other `*_score` branches (`mipid_score`, `gap_score`,
  `hol_lol_score`, `cme_anc_score`, `mgo_mgt_score`, `br1_score`, `br3_score`,
  `stemdir_br2_score`, `trimuon_score`, `br4_tro_score`, `mipquality_score`,
  `pio_1_score`, `stw_spt_score`, `vis_1_score`, `vis_2_score`,
  `cosmict_2_4_score`, `cosmict_3_5_score`, `cosmict_6_score`,
  `cosmict_7_score`, `cosmict_8_score`) structurally 0. This is a porting gap,
  reported rather than papered over with a proxy (CLAUDE.md §5 rule 7): the
  event-level cosmic verdict this doc actually uses is the independent
  TGM/STM/LM label from `nusel_extract.py`, not a score.
- Scores come from **uBooNE-trained weights**
  (`cfg/pgrapher/experiment/sbnd/clus.jsonnet:1145-1152`) — uncalibrated on
  SBND, ranking only (doc pr/2 gap G1, SBND BDT retraining, out of scope
  here). `±4.300936` is the XGB log-odds clamp, already documented in doc
  pr/3. **Correction to an earlier draft of this bullet:** it is *not* true
  that `nue_score` saturates on most nu-candidate events. Only 193/1071
  events run the νe BDT at all, and of those only 32 saturate positive — all
  32 in the nueCC48 sample. The dominant `nue_score` value population-wide is
  the `−15` *not-evaluated sentinel*, which is not a score (sec 2.3).
- Wall under `PR_JOBS=20` concurrency is throughput-under-load, not isolated
  latency; peak RSS is per-process and concurrency-independent (same framing
  as `clus/docs/imgclus-resource-profile.md`). The sequential 30-event arm
  (sec 3/4) carries the latency/RSS claim — and at 20-way it carries more of
  the weight than it would have at 6-way, so no per-event wall figure from
  the census arm should be quoted as a latency.
- `nu_skip_cosmic=ON` (SBND production default) means a real fraction of
  events legitimately produce no score — that fraction is a result (sec 2),
  not a defect.
- `--prtree`/`save_tensors` is enabled per event (needed for authoritative
  `flag_TGM/STM/FC` in `nusel_extract.py`, avoiding the documented log-tearing
  gotcha) but `-stm-fit`/`stm_magnify`/`tracking-stm.root` is not — no STM
  Magnify display artifacts were produced this round.
- **Superseded.** An earlier draft of this bullet listed the steiner
  `RuntimeError` and the `std::bad_alloc` as pre-existing gaps "reported, not
  patched in a measurement round". The owner subsequently asked for them to be
  fixed; all five crash/hang classes and both shower point-cloud defects are
  now fixed and shipped (sec 6, commit `289d78e4`), and every number in secs
  2-5 comes from the fixed binary. This round is therefore **no longer a
  pure measurement round** — see the Status banner.
- **G3 — the DL-vs-geometric delta is unmeasured.** The owner's decision this
  round was "SCN DL vertex, production default", and sec 5 shows the DL vertex
  is *accepted* on 82.4% of events that reach the vertex stage — so the census
  scores really are DL-vertex scores. But no geometric arm was run on **this**
  binary, and the doc pr/3 spot-check cannot substitute for one (sec 2.6). The
  size of the score/energy shift attributable to DL alone is therefore
  unknown. A same-binary `-A dl_weights=` (empty) arm over the same 1071
  events would close it; it is a separate round.

# Doc 76 — SBND production chain, perf round 1 (2026-08-23)

Owner ask: "for the current SBND production, use the perf tools to evaluate
the bottleneck on running time and memory; improve; repeat until no major
benefit; gated by no change in the final results."

**Result.**  The per-event cost of the production chain (imaging → Q/L →
pattern recognition, one `wire-cell` process per stage per event) is
dominated not by physics but by a fixed per-process cost in the PR stage:
booking 35 TMVA BDT weight files (358 MB of XML, 199 MB of it one file)
took **4.8 s of the PR job's 6.5 s solo wall and 0.9 GB of its 1.17 GB
peak RSS**, on every event, including the 54 % of data events that never
score a candidate.  Two rounds fix that with **byte-identical outputs
(gate: 308 events × {mabc-pr.zip, pctree, tracking-pr.root branches,
nusel table}, all PASS)**:

| PR job, solo | base | round 1 (lazy) | round 1+2 (fast forest) |
|---|---:|---:|---:|
| non-scoring data event (166650): wall | 6.5 s | **1.4 s** | 1.4 s |
| non-scoring data event: peak RSS | 1.17 GB | **0.42 GB** | 0.42 GB |
| scoring event (nueCC48 10550, solo): wall | 13 s | 13 s | **10 s** |
| scoring event: nue-scorer step (books + scores) | 4.4 s (in configure) | 4.6 s | **1.8 s** |
| scoring event: peak RSS | 1.5 GB | 1.8 GB | **1.24 GB** |

Population-weighted (mcp1k/mcp2k data: 54 % non-scoring), the PR stage
drops from ~6.5 s to ~2.8 s per event and from 1.2–1.5 GB to 0.4–1.2 GB;
on the 308-event gate manifest (85 % scoring) the summed wall is −24 %.
`fast_xgb_forest` is **SBND production ON** (sec 6.2); C++ default off.
The Q/L and imaging stages were profiled too (sec 2, sec 7): their cost is
physics-shaped (heavy tail in `ClusteringExamineBundles`'s relaxed-graph
Hough transforms) and already sits on a previous optimization; nothing
there is a byte-identical lever this round.

Toolkit commit: see `git log` for `root: doc 76` (this doc) — the
`TmvaGradForest` evaluator, the lazy reader booking, the `fast_xgb_forest`
knob and its doctest.  Runner: `run_pr_chain_batch.sh` gains
`SBND_FAST_XGB_FOREST` (tri-state, same idiom as every other knob).

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
S=<scratch>   # this round: /home/xqian/tmp/claude-25225/.../scratchpad/perf

# 1. Where the time goes today (prod0823 outputs, no rerun needed)
python3 - <<'EOF'        # .time.meta wall/RSS per stage  (sec 1)
...see sec 1 tables; the one-liners are in this doc's history
EOF

# 2. Solo PR baseline on a non-scoring data event (base binary f3666c28 built 12:40)
PR_JOBS=1 ./run_pr_chain_batch.sh work-mcp1k-ql0819 $S/pr_solo2 data 166650
#   -> .time.meta wall 6-7 s, maxrss 1.17 GB; log: configure 5.4 s, graph 1.1 s

# 3. Memory map of that process at steady state (sec 3)
#   sample /proc/<pid>/smaps every 0.25 s, keep the largest: [heap] 919 MB,
#   libCling+pch 150 MB, everything else < 10 MB each

# 4. CPU profile of a scoring event's configure phase (sec 4; no setarch, M17)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4:$PYLIB \
  CPUPROFILE=pr.prof CPUPROFILE_FREQUENCY=500 wire-cell -l stderr -L info -c <compiled cfg>
google-pprof --text $(which wire-cell) pr.prof | grep -i "TMVA\|TXMLEngine"

# 5. Gate manifest: the 241 mcp1k events that score (UbooneNumuBDTScorer step
#    > 1 ms in prod0823) plus the first 60, all 48 nueCC48, all 19 NCpi0
#    -> $S/mcp1k_gate.txt
# 6. Arms (each: base binary first, then the changed binary; PR_JOBS=8)
PR_JOBS=8 ./run_pr_chain_batch.sh work-<s>-ql0819 $S/base-<s>  data [<events>]   # base
PR_JOBS=8 ./run_pr_chain_batch.sh work-<s>-ql0819 $S/lazy-<s>  data [<events>]   # round 1
PR_JOBS=8 ./run_pr_chain_batch.sh work-<s>-ql0819 $S/off3-<s>  data [<events>]   # round 2, knob off
SBND_FAST_XGB_FOREST=1 \
PR_JOBS=8 ./run_pr_chain_batch.sh work-<s>-ql0819 $S/fast3-<s> data [<events>]   # round 2, knob on
# 7. Gates (sec 5, sec 6)
python3 scripts/pr85_hash_gate.py --jobs 8 $S/base-<s> $S/<arm>-<s>   # mabc-pr.zip + pctree member hashes
python3 scripts/pr94_root_gate.py        $S/base-<s> $S/<arm>-<s>   # every branch of tracking-pr.root
diff <(sort $S/base-<s>/nusel-table.tsv) <(sort $S/<arm>-<s>/nusel-table.tsv)
# 8. Unit test of the evaluator against TMVA itself (4013 assertions)
(cd toolkit && TMPDIR=$S ./build/root/wcdoctest-root)
```

## 1. Where the production chain spends its time

The current production products are `work-<s>-prod0823` (PR stage,
3067 events, 2026-08-23), `work-<s>-ql0819` (Q/L) and `work-img-<s>`
(imaging).  Every stage writes enough to be costed without a rerun:
`.time.meta` (timecmd.py wall + maxrss), the WCT `Timer:` lines and the
`MABC timing:` per-step lines.

### 1.1 The recorded numbers are oversubscription artefacts

prod0823's PR stage reads **46 s median wall / 1.15 GB** per mcp1k event
(sum 40 913 s over 1000 events, i.e. 1000 events in 22 min ⇒ ~35 jobs in
flight) and 20 s on mcp2k.  Inside those jobs the graph's `Timer:` lines
say the TensorFileSource took a **median 23.8 s wall for 0.36 core-s** and
the sink 11.9 s for 0.2 core-s — pure waiting.  Re-running 24 of the same
events at `PR_JOBS=6` on the idle box: **6–11 s wall each** (source 0.6 s,
sink 0.4 s), and solo: 6.5 s.  So the batch numbers say nothing about the
code; the per-event cost below is measured solo or at `-j8` on an idle
64-core box (CLAUDE.md M5).

### 1.2 Per-event cost, solo, typical data event (mcp1k 166650)

| stage | wall | of which fixed (startup + configure + IO) | physics | peak RSS |
|---|---:|---:|---:|---:|
| imaging (`wct-img-all`) | 8.3 s median log span | ~1.5 s | ~7 s (8 core-s, TBB) | n/a (no meta) |
| Q/L (`wct-clus-matching-perevt`) | 8 s median (9 mean; tail 191 s) | ~1.7 s | 6.4 s | 431 MB median, 960 max |
| PR (`wct-pr-perevt`) | **6.5 s** | **5.4 s configure + 0.5 s startup** | **1.1 s** | **1.17 GB** |

The PR job is the only stage where the fixed cost dwarfs the work, and it
is the stage every A/B round in this project re-runs by the hundreds.

### 1.3 Inside the PR job's 5.4 s configure

Consecutive `configuring component` timestamps (solo log):

| component | configure time |
|---|---:|
| `UbooneNueBDTScorer:pr` | **4.41 s** |
| `UbooneNumuBDTScorer:pr` | 0.40 s |
| `LinterpFunction:MuonRange` | 0.30 s |
| `WireSchemaFile` (bz2 json) | 0.18 s |
| `SCEFieldTH3` | 0.10 s |
| everything else (60 components) | < 0.05 s |

The two scorers book 35 TMVA weight files from `wire-cell-data/uboone/weights`
totalling **358 MB of XML**:

| file(s) | size | model | notes |
|---|---:|---|---|
| `XGB_nue_seed2_0923.xml` | **199.0 MB** | xgboost2TMVA, BoostType Grad, 600 trees, **1 490 732 nodes**, 258 vars, no transformations | the nue combiner |
| `numu_scalars_scores_0923.xml` | 13.6 MB | same shape, 400 trees, 102 754 nodes, 74 vars | the numu combiner |
| 30 nue sub-BDTs + 4 numu sub-BDTs | 145 MB | AdaBoost, 850 trees each, **5–6 input transformations** (Decorrelation/PCA/Gauss/Gauss/Decorrelation/Normalize) | stay on TMVA (sec 6) |

## 2. Memory: 0.9 GB of heap before the first event

`/proc/<pid>/smaps` sampled through a solo PR run, largest snapshot:

| mapping | RSS |
|---|---:|
| `[heap]` | **919 MB** |
| `allDict.cxx.pch` + `libCling.so` (ROOT) | 150 MB |
| every other library | < 8 MB each |

RSS over time (0.25 s samples): 70 MB → 104 → 288 → 486 → 687 → 888 →
960 MB across the 2.5 s in which the two scorers configure, flat
afterwards; the MABC `MEM:` lines confirm `res=1.16 GB` at "starting
MultiAlgBlobClustering", before any physics.  The heap is the TMVA DOM +
`DecisionTreeNode` objects of the 1.6 M nodes.

## 3. Who scores?  54 % of data events never reach the BDTs

`UbooneNueBDTScorer::visit` returns early when the grouping has no
`TrackFitting` (no neutrino candidate was built).  Counting the
`no TrackFitting in grouping` warning in prod0823 logs:

| sample | events | skip nue scoring |
|---|---:|---:|
| mcp1k (data) | 1000 | **537 (54 %)** |
| mcp2k (data) | 2000 | **1093 (55 %)** |
| NCpi0 MC | 19 | 0 |
| nueCC48 MC | 48 | 0 |

(The numu-scorer-step > 1 ms criterion used to build the gate manifest
picks a superset — 206 of the 241 manifest events score — which is why the
manifest is biased toward scoring events; that is what a gate wants.)

## 4. CPU profile of the load (gperftools, scoring nueCC48 event, 500 Hz)

`TMVA::Reader::BookMVA` = 41.5 % of the samples.  Inside it:

| function | samples | share of BookMVA |
|---|---:|---:|
| `TMVA::Reader::GetMethodTypeFromFile` | 271 | 25 % — parses the WHOLE file into a DOM just to read `Method="BDT::BDT"`, frees it |
| `TXMLEngine::ParseFile` (both parses) | 432 | 40 % |
| `TMVA::Node::ReadXML` → `DecisionTreeNode::ReadAttributes` (stringstream reads per node) | 456 | 43 % |
| `TransformationHandler::ReadFromXML` (the 34 sub-BDTs' Gauss PDFs) | 101 | 9 % |

So the 199 MB combiner is parsed twice into a DOM and then walked node by
node through `std::stringstream`.  Nothing about the *evaluation* needs
any of that: the forest is 1.49 M `(ivar, cut, cType, response)` tuples.

## 5. Round 1 — book the readers lazily (byte-identical, no knob)

`init_readers()` ran from `configure()`.  It is now `ensure_readers()`,
`std::call_once` from the first `visit()` that has a `TrackFitting` to
score — after the same grouping/TrackFitting checks that already gate the
scoring loop, so an event that never scored never pays.  The readers, the
variable bindings and every `EvaluateMVA` are untouched; the early-return
warning for an unset `nue_xgboost_xml` now tests the resolved path (the
condition under which the reader used to be created).  Both scorers.

Gate, base binary vs round-1 binary, **308 events** (241 mcp1k + 48
nueCC48 + 19 NCpi0), `PR_JOBS=8`:

| sample | pr85 hash gate (mabc-pr.zip + pctree) | nusel-table | pr94 ROOT branch gate |
|---|---|---|---|
| mcp1k | PASS 482/482 archives | IDENTICAL | PASS 241/241 |
| nueCC48 | PASS 96/96 | IDENTICAL | PASS 48/48 |
| NCpi0 | PASS 38/38 | IDENTICAL | PASS 19/19 |

Effect: a non-scoring event's PR job goes from 6.5 s / 1.17 GB to **1.4 s
/ 0.42 GB** (solo, 166650).  A scoring event is unchanged in wall (the
4.6 s parse moved from `configure` into the scorer step) and its peak RSS
*rises* ~300 MB (1.49 → 1.75 GB median on the mcp1k manifest) because the
transient DOM now sits on top of the event's point clouds — which is what
round 2 removes.

## 6. Round 2 — `TmvaGradForest`: evaluate the XGB combiners without TMVA

`root/inc/WireCellRoot/TmvaGradForest.h`, `root/src/TmvaGradForest.cxx`.
One `mmap` of the weight file, one linear scan for `<Node ...>` tags into
a 24-byte-per-node array (36 MB for the nue combiner instead of ~700 MB of
`DecisionTreeNode`s), and an `evaluate()` that is a literal transcription
of ROOT 6.32.02's evaluation path, each step cited in the header:

* `Tools::ReadAttr(float&)` = `atof` then narrow to `Float_t`; `int`/`short`
  = `atoi`; `Bool_t` = `stringstream >>`;
* `Node::ReadXML` attaches children by `pos="l"/"r"`;
* `DecisionTreeNode::GoesRight`: `value[IVar] >= Cut`, inverted when
  `cType` is false (note `>=`, not `>` — this is the 6.32 source, which is
  why the source was fetched rather than recalled);
* `DecisionTree::CheckEvent` on a regression tree (`Weights
  AnalysisType="1"`) returns the leaf's `Float_t` response;
* `MethodBDT::GetGradBoostMVA`: `sum` (double) of the responses, return
  `2.0/(1.0+exp(-2.0*sum))-1`;
* `Reader::EvaluateMVA`: any NaN input ⇒ `-999`.

It refuses (throws `ValueError`) anything outside that class — variable
transformations, a non-Grad BoostType, preselection cuts, Fisher-cut nodes,
non-regression trees, a variable list that differs from the caller's — so
the 34 transformation-carrying sub-BDTs cannot silently take this path;
they stay on `TMVA::Reader`.

Wired behind **`fast_xgb_forest`** (C++ default `false`; `cm.numu_bdt_scorer`
/ `cm.nue_bdt_scorer` in `pgrapher/common/clus.jsonnet` emit the key only
when true, threaded through SBND `clus_pr()` / `pr()` and the
`wct-pr-perevt.jsonnet` TLA; runner env `SBND_FAST_XGB_FOREST=0|1`).
Compiled-config proof: with the knob off the compiled JSON for evt 166650
is byte-identical to the pre-change compile (paths normalised); with it on
both scorers carry `"fast_xgb_forest": true`.

### 6.1 Unit test

`root/test/doctest_tmva_grad_forest.cxx` writes a 40-tree forest in the
xgboost2TMVA layout (same attribute spellings and number formats as the
production files, `cType` 0 and 1, cut values that the inputs hit
exactly), books it with **both** `TmvaGradForest` and a real
`TMVA::Reader`, and requires `==` on 4000 random inputs (ties on `>=`,
±1e30, NaN ⇒ −999): **4013/4013 assertions pass** (`wcdoctest-root`).  A
second case checks every refusal.

### 6.2 Gate

Two arms on the round-2 binary against the base arm, same 308-event
manifest, `PR_JOBS=8`: **knob off** (`fast_xgb_forest` absent — the
legacy `TMVA::Reader` path, proves the code motion is inert) and **knob
on** (`SBND_FAST_XGB_FOREST=1`).

| sample | arm | pr85 hash gate (mabc-pr.zip + pctree) | nusel-table | pr94 ROOT branch gate |
|---|---|---|---|---|
| mcp1k (241) | off | PASS 482/482 | IDENTICAL | PASS 241/241 |
| mcp1k (241) | **on** | **PASS 482/482** | **IDENTICAL** | **PASS 241/241** |
| nueCC48 (48) | off | PASS 96/96 | IDENTICAL | PASS 48/48 |
| nueCC48 (48) | **on** | **PASS 96/96** | **IDENTICAL** | **PASS 48/48** |
| NCpi0 (19) | off | PASS 38/38 | IDENTICAL | PASS 19/19 |
| NCpi0 (19) | **on** | **PASS 38/38** | **IDENTICAL** | **PASS 19/19** |

Every `nue_score` / `numu_score` and the fifteen nue sub-scores are in
`tracking-pr.root` (T_tagger) and the nusel table, so the ROOT branch gate
is the direct check that the forest evaluates identically on 273 scoring
events (206 mcp1k + 48 + 19) — on top of the 4000-input unit test.

**SBND production default: ON** (`wct-pr-perevt.jsonnet`
`fast_xgb_forest = true`, flipped after the gate).  Compiled-config proof
on nueCC48 10550: the default compile is byte-identical to the knob-on
arm's compile; `SBND_FAST_XGB_FOREST=0` compiles byte-identically to the
pre-change base compile (paths normalised); the outputs of both hash
identically to the base arm's.  Every other detector's job compiles
unchanged (the common builders omit the key unless asked).

### 6.3 Effect

Per-event PR-stage wall / peak RSS, `PR_JOBS=8` on the idle box, medians
over the manifest (`.time.meta`; wall is 1-s resolution):

| events | base | round 1 (lazy) | **round 1+2 (fast, production)** |
|---|---:|---:|---:|
| mcp1k scoring (206) — wall | 10 s | 10.5 s | **7 s** |
| mcp1k scoring — peak RSS med/max | 1488 / 1603 MB | 1759 / 2092 | **1203 / 1536** |
| mcp1k non-scoring (35) — wall | 8 s | 2 s | **2–3 s** |
| mcp1k non-scoring — peak RSS | 1148 MB | 387 | **386** |
| nueCC48 (48) — wall | 18 s | 18 s | **14.5 s** |
| nueCC48 — peak RSS med/max | 1488 / 1508 MB | 1796 / 2016 | **1242 / 1486** |
| NCpi0 (19) — wall | 16 s | 16 s | **13 s** |
| NCpi0 — peak RSS | 1488 MB | 1783 | **1228** |
| sum of wall, all 308 | 4015 s | 3861 s | **3040 s (−24 %)** |
| `UbooneNueBDTScorer:pr` step (books + scores) | 0 (booked in configure, 4.4 s) | 4.6 s | **1.8 s** |
| `UbooneNumuBDTScorer:pr` step | 0 (0.4 s in configure) | 0.42 s | **0.25 s** |

Solo, nueCC48 10550: 13 s / 1.81 GB (TMVA path) → **10 s / 1.24 GB**.
Solo, non-scoring 166650: 6.5 s / 1.17 GB → **1.4 s log span (2 s wall) /
0.42 GB**.  The manifest is 85 % scoring events; on the production data
mix (54 % non-scoring) the PR stage goes from ~6.5 s to ~2.8 s per event
and the fleet-wide peak RSS from 1.2–1.5 GB to 0.4–1.2 GB.

What is left in the scoring events' 1.8 s nue step is the 30 sub-BDTs
(~1.4 s of `TMVA::Reader` with their transformation chains) plus ~0.4 s
for the 199 MB scan; the peak RSS of a scoring event is now set by the
DL-vertex python stack it loads, not by the BDTs.

## 7. What was looked at and left alone

* **Q/L heavy tail.**  Median 8 s, but 191 s (mcp1k evt 280884), 116 s
  (286617), 59 s, 57 s … — all `ClusteringExamineBundles` (137 of 187 s on
  280884: `apa1-0` 67 s + `all` 70 s), then `ProtectOverclustering` 19 s and
  `Deghost` 18 s.  gperftools on 280884 (42 705 samples): 44 % in
  `Cluster::hough_transform` (nanoflann radius search 28 %, `acos` 9 %),
  30 % in `Grouping::has_closest_point`, both from `connect_graph_relaxed`
  (the pairwise `vhough_transform` loops over the merged flash-group's
  sub-clouds).  `hough_transform` already carries the sparse stamped-grid
  optimisation from an earlier round; what is left is the kd-tree radius
  queries and per-point `acos`, whose arithmetic sets bin boundaries —
  not a byte-identical lever.  The tail is ~8 % of the Q/L stage's
  summed wall; parked.
* **Q/L `TensorFileSink`: 1.0 s wall for 0.18 core-s per event**
  (`m_out.flush()` after every one of the ~412 tensors).  `/nfs/data/1` is
  local ext4, so this is not NFS latency; not chased this round (a
  candidate for the next one — removing the per-member flush changes the
  gzip framing but not the member content the gate hashes).
* **The remaining PR-stage fixed cost after round 2**: the 34 sub-BDTs
  (~1.3 s, ~200 MB), of which 25 % is TMVA's double parse
  (`BookMVA(tag, file)` → `GetMethodTypeFromFile`); avoidable with
  `BookMVA(Types::kBDT, xmlstring)` at ~0.3 s/event — marginal, parked.
  `LinterpFunction:MuonRange` 0.3 s (irregular interpolation table build).
* **Per-event process model.**  Imaging + Q/L + PR spawn three
  `wire-cell` processes per event, each re-loading plugins, ROOT/Cling
  (150 MB, ~0.5 s), the SCE map and — in PR — the BDTs.  A multi-event
  job would amortise all of it; that is a runner/config restructuring, not
  a code round (docs 20 and 65 reached the same conclusion).

## 8. Verification summary

* Unit tests: `wcdoctest-clus` 2397/2397 (round 1), `wcdoctest-root`
  4013/4013 (round 2).
* Freshness: `build/root/libWireCellRoot.so` 18:01:19 > last edit; the
  runtime loads `build/<pkg>/` (smaps), `local/lib` installed in lockstep.
* Determinism: no pointer-keyed iteration introduced; `std::call_once`
  guards the lazy booking; `evaluate()` is pure.
* Other experiments: `pgrapher/common/clus.jsonnet` builders default the
  new key off and omit it ⇒ uBooNE / every other job compiles byte-identically.

## 9. Follow-up review of the §7 parked items (2026-08-24, analysis only, no code)

Owner asked how much each §7 item is actually worth, and whether the Q/L
tail carries any free (byte-identical) redundancy before reaching for an
approximation.  No code changed in this section — sizing and a code read
only, recorded so the next round doesn't re-derive it.

### 9.1 Ceiling on each §7 item, against the current (post round-1+2)
per-event chain: imaging 8.3 s + Q/L 8 s (median) + PR 2.8 s
(population-weighted) = **19.1 s**

| item | save | applies to | % of full chain |
|---|---:|---|---:|
| Q/L `TensorFileSink` per-tensor flush (1.0 s wall, 0.18 core-s) | ~0.8 s | every event | **~4 %** |
| sub-BDT double parse (`BookMVA(tag,file)`→`GetMethodTypeFromFile`, sec 7) | ~0.3 s | only the ~46 % of data events that score | **~0.7 %** weighted |
| multi-event process (amortise ROOT/Cling 150 MB × 3 processes, wire geometry, SCE map) | ~1.0–1.5 s in the limit of a large batch | every event, but only pays off batched | **~5–8 %**, ceiling ~20–26 % if all remaining fixed cost were amortised away |
| Q/L heavy tail (`hough_transform`/`has_closest_point`, sec 7) | bounded by its own share | rare outlier events | **~3–4 %** (doc's own "~8 % of Q/L stage" ÷ chain) |

None of the four is in the same class as round 1+2 (which cut the PR
job's own solo cost 55–80 % and the 308-event gate manifest −24 %); all
four are chasing single-digit-percent residual margin.  The multi-event
process model is the only one whose ceiling is comparable in size, and
it is a runner/config restructuring, not a component change (docs 20, 65
reached the same conclusion previously).

### 9.2 Q/L tail: checked for a free (byte-identical) win, found none

Read `clus/src/connect_graph_relaxed.cxx`, `make_graphs.cxx`,
`Facade_Cluster.cxx`'s `find_graph`/`give_graph`, and
`Facade_Grouping.cxx`'s `has_closest_point`:

* `Facade::Cluster::find_graph()` (`Facade_Cluster.cxx:2795`) already
  memoizes by `flavor` (`if (this->has_graph(flavor)) return
  get_graph(flavor);`) — a second call for the same flavor on the same
  cluster is free.  No duplicate-graph-build bug across the many
  `make_graph_relaxed*` entry points in `make_graphs.cxx`.
* Inside `connect_graph_relaxed`'s O(num²) component-pair loop, every
  `vhough_transform`/`hough_transform` call uses a different origin
  point (each pair's own closest points) and the different radii
  (15/30/50/80 cm at different call sites) are genuinely different
  queries, not restatements of one query.
* `has_closest_point` (`Facade_Grouping.cxx:680`) is called once per
  1 cm step along each pair's connecting path
  (`grouping->test_good_point`, via `connect_graph_relaxed.cxx`'s
  "check points along path" loop) — cost is O(num² × path length),
  which is what the algorithm is defined to check, not redundant work.
  `has_closest_point` itself is already at its floor: `exists_within`
  early-terminates on the first in-radius point rather than resolving
  a true kNN.

Conclusion: unlike the PR-stage TMVA double-parse, there is no
zero-risk lever here.  Any further speedup (coarser Hough bins, cheaper
trig, approximate/early-truncated radius search, reordering the
per-bin `+=` accumulation) changes output bits — precision traded for
speed, not waste removed — and would need a knob plus real physics
validation, not a hash gate, for a ≤4 % average win.  Consistent with
the sec 7 parked verdict; not chased.

### 9.3 Idea: gate an approximation on `num` to confine it to the tail (not implemented)

The O(num²) pair loop means cost is predicted by `num` (the connected-
component count) — already computed before any of the expensive work
runs, so it is a free, existing signal to gate on:

```
if (num > relaxed_fast_threshold) {
    // coarser-bin / cheaper-trig approximate path — new
} else {
    // today's exact path — untouched, byte-identical
}
```

Why this shape is better than a blanket knob: the legacy path stays
provably bit-identical for the large majority of events (only clusters
over the threshold ever reach the approximation), so validation effort
shrinks to just the handful of events that cross it — e.g. the two
named in sec 7 (280884, 286617) — rather than a general sample.  It
also targets the actual pain (worst-case/straggler latency in a batch),
which is what this lever is good for, rather than average throughput
(recall: even fully eliminating the tail only buys ~8 % of the Q/L
stage on average, sec 9.1).

Caveat that keeps this from being low-stakes despite the small event
count: events that cross a `num` threshold are disproportionately the
busy, multi-track topologies — often the physically interesting ones.
Confining the blast radius shrinks the *validation scope*, not the
*rigor required per affected event*; each event that takes the fast
path would still need the same handscan-level check as a general
knob's gate, just on far fewer events.

Not scoped further this round — would need the distribution of `num`
across a normal mcp1k/mcp2k sample (to find where the cost cliff
actually sits and how many production events would cross a candidate
threshold) before deciding whether to pursue it.
## 10. Round 2 — many events per process (2026-08-24)

Owner ask: "from the Rec1 file … imaging/clustering/ql-matching/event-taggers
… one single output, and the code can do multiple events together, not per
event … there should also be a way to run a single event or a single step if
needed like now.  From the output … we can then run PR … in groups … also the
capability to do PR for a single event in the large group."

**Result.**  The chain now runs a GROUP of events per `wire-cell` process
instead of one event per process, in two stages, and the per-event path is
untouched and still the default.  Getting there needed C++ after all: the
PR/tagger stages carried a **use-after-free that fires on the second event of
any process** (sec 10.1), unreachable one-event-per-process and therefore
invisible to every gate this project has ever run.  On the 100 numuCC events,
stage B goes from **100 processes / 526 s of summed process wall to 7 / 373 s
(-29 %)** with peak RSS per process flat (sec 10.7).  Everything is
byte-identical -- legacy path, other detectors, and group-vs-per-event -- with
**one exception, which is the owner's to rule on**: with the DL (SCN) vertex ON,
26 of 167 events differ in vertex-position-derived kinematics, while `T_tagger`
(the tagger verdicts and BDT scores) and all 1938 bundle labels are unchanged
(sec 10.6).

This is §7's parked "per-event process model", the last item whose ceiling
(§9.1: ~1.0–1.5 s/event, ~5–8 % of the chain, 20–26 % if all fixed cost went)
was comparable to rounds 1+2.

**§7 called it "a runner/config restructuring, not a code round".  That was
wrong**, and the correction is the most important result here: the PR/tagger
stages carry a **use-after-free that fires on the second event of any process**.
It cannot be reached one-event-per-process, so no gate in this project has ever
been in a position to see it.

### 10.1  What actually blocked it (and why no earlier gate could have seen it)

Probe: two events' post-Q/L trees concatenated into one archive, fed to today's
PR job.  Event 1 completes; **event 2 SIGSEGVs** in
`TrackFitting::form_point_association` (`clus/src/TrackFitting.cxx:2316`) from
`TaggerCheckSTM::visit`.

`TrackFitting::m_grouping` is set exactly once, ever
(`TrackFitting.cxx:317`, `if (m_grouping == nullptr && segments_first)`), but the
`Grouping` it points at lives on the `Points::node_t root` that
`MultiAlgBlobClustering::operator()` builds as a **local** and destroys when the
event ends.  `TaggerCheckNeutrino` and `TaggerCheckSTM` each hold their fitter
for the life of the process, so from event 2 the guard is false,
`BuildGeometry()` is skipped, and ~30 sites dereference freed memory.  Two
quieter members ride along: `global_rb_map` refuses to rebuild while non-empty
(`:1030`), so event N attributes charge with event 1's `Blob*` map; and
`sync_from_graph()` only ever *inserts* into `m_clusters`/`m_blobs` (`:300-329`)
— a fact the code already knows *within* an event
(`TaggerCheckNeutrino.cxx:1646-1651`: "sync_from_graph() **ACCUMULATES** …
reusing one fitter would leak bundle i-1's charge data into bundle i") and that
applies verbatim *across* events.

Fix: `TrackFitting::reset_for_new_event()`, called once at the top of
`TaggerCheckNeutrino::visit()` and `TaggerCheckSTM::visit()`.

**The invariant that matters: the reset is per `visit()`, never per candidate.**
`TaggerCheckNeutrino.cxx:1628-1660` loops over neutrino candidates within one
event, and candidate 0 deliberately reuses the member fitter while candidates
>=1 get a fresh one.  A reset placed inside that loop (or in `add_graph()`)
would change candidate-0 behaviour on *every* event today — the whole gate would
flip, and it would read like "multi-event changed physics" rather than "the
reset is in the wrong place".  This change adds freshness ACROSS events without
removing it WITHIN one, and is a no-op on a one-event process, which is what
makes the legacy path byte-identical.

Everything else the multi-event path needed was already there and simply never
wired: the pgraph engine has no event concept at all (`pgraph/src/Graph.cxx`
is a data-driven drain loop), `TensorFileSource`/`ClusterFileSource`/
`FrameFileSource` have always streamed a whole archive to EOS on an ident
boundary, `TensorFileSink`/`ClusterFileSink`/`FrameFileSink` already name every
member by ident, `wct-img-all.jsonnet` has **no event TLA at all**, and
`MultiAlgBlobClustering` already has a per-ident boundary path.  What pinned the
chain to one event per process was only the shape of the input handed to it:
`_runlib.sh:105-116` carves one event's frames, `run_ql_evt.sh:351-373` builds a
one-event `$QLDIR`, `run_pr_chain_batch.sh` names one `pctree-evt<ID>.tar.gz`.
`--tla-code event=<ID>` is **not** a selector — it is only `eventNo` for RSE
labelling, which is exactly why a group needs a way to say otherwise.

### 10.2  Two stages, and why the boundary is where it is

The owner asked for stage A to end after the event taggers.  **It cannot.**
Probe (3 events, one per sample): save the tree after `steiner_refresh` and
resume from it with `tagger_check_neutrino` onward — all three abort with
`std::runtime_error: Empty Steiner point cloud`.  `CreateSteinerGraph`'s point
cloud, its boost graphs and the `GraphAlgorithms` the STM fit caches are
in-memory only and are not in the pctree.  Re-running `steiner` at the head of
stage B does not restore identity either: the first pass builds on *unsplit*
clusters and `steiner_refresh` (`replace=false`) rebuilds only what
`protect_bundle` purged, which no fresh full pass reproduces.

What the same probe measured decided the shape (solo, idle box):

| PR job | wall | peak RSS |
|---|---:|---:|
| stages 1-10 only | 2 s | 0.41 GB |
| all 15 stages | 7 / 9 / 10 s | 1.23-1.28 GB |

Per stage on nueCC48 10550: stages 1-10 = **1.10 s** (Steiner 0.72 + TGM 0.29),
stages 11-15 = **6.62 s** (TaggerCheckNeutrino 4.57 + nue BDT 1.74).  All the
cost is in the second half, so duplicating the first half is cheap — and the
`nusel-evt<ID>.tsv` written by the 1-10 pipeline is **byte-identical** to the
one from the full 15, so it is a complete selection product on its own.

Boundary as shipped (owner-confirmed):

* **Stage A** — reco1 -> imaging -> clustering -> Q/L, one process per stage per
  GROUP.  Hands stage B the **post-Q/L pctree**, the boundary the chain already
  had and the only one proven lossless.  Optionally also runs stages 1-10 in
  memory to emit the cosmic-tagger table used to choose what goes on.
* **Stage B** — all 15 stages from that pctree, in groups.

### 10.3  What changed

**C++ (toolkit).**  Each item is inert for a one-event process.

| what | where | why |
|---|---|---|
| `TrackFitting::reset_for_new_event()` + one call per visitor | `clus/src/TrackFitting.cxx`, `TaggerCheckNeutrino.cxx`, `TaggerCheckSTM.cxx` | the UAF above.  No knob: the current behaviour on event >=2 is *undefined*, and event 1 is unchanged |
| `Ensemble::set_ident()` / `set_rse()` | `clus/inc/WireCellClus/Facade_Ensemble.h`, set in `MultiAlgBlobClustering` | the one place a visitor can learn WHICH event it is looking at.  Published only in a multi-event mode, so per-event jobs keep using their own configured numbers |
| `event_from_ident` + `rse_map` | `MultiAlgBlobClustering` | event number from the tensor ident, run/subrun per event.  `rse_from_ident` (which forces run=subrun=0) already existed but throws the real run away, and a group can span many runs -- SBND nueCC48 spans 12 |
| `flush()` returns early when nothing was loaded since the last flush | `MultiAlgBlobClustering` | EOS-then-finalize called it twice, and the second call still advanced the shared Bee event index |
| one output file per event when the configured name carries a `%` | `TensorFileSink`, `BeeSink`, MABC's own `bee_zip`, `SbndPrMagnifyTrackingVisitor`, `UbooneTaggerOutputVisitor`, `SbndMagnifyTrackingVisitor`, `PrDisplayDump` | this is what makes a group job write **exactly today's per-event files**, which is in turn what makes the gate a file-for-file comparison instead of a new bespoke one.  Same idiom as `QLMatching`'s calib dump.  The conversion is `boost::format`, so `%1%` where the id appears twice in one name |
| `"evt<ID> "` prefix on the log lines `nusel_extract.py` parses | `TaggerCheck{TGM,STM,FC}` | see 10.5 -- empty, and the log text unchanged, unless a group is streaming |

**Config (toolkit).**  Three default-OFF TLAs on the two SBND jobs
(`multi_event`, `rse_map`, and `evt_subdir` on the PR job), threaded through
`clus.jsonnet` with the key-suppression idiom.  `evt_subdir` and
`event_from_ident` are a pair and jsonnet now *asserts* they are set together:
the tensor sink resolves its `%` from the tensor ident while MABC's own zip and
the ROOT writers resolve theirs from the event number MABC published, so one
without the other would scatter one event's output across two paths.

**Runners (wcp-porting-img).**  `run_pr_chain_batch.sh` gains `PR_GROUP_SIZE`
(unset/0 = today's per-event path, untouched) and `PR_PIPELINE`; new
`run_chain_group.sh` for stage A; `scripts/multi/` holds the group-archive
builder, the RSE-map builder, the log slicer and the Q/L legacy gate.
`nusel_extract.py` gains `--bw-gate` (default: read from the log, as today).

### 10.3a  Stage A, and what "one output" means

`run_chain_group.sh <reco1.root> <out> <data|sim> --size G [--group K]` runs, per
group, ONE `wire-cell` process per stage instead of one per stage per event:

```
reco1 art file --(entry range)--> g<K>/frames-dnn.tar.bz2 + opflash_apa{0,1}.tar.gz
                     |
                     +-- imaging  ---> g<K>/icluster-apa{0,1}-{active,masked}.npz
                     +-- clus+Q/L ---> g<K>/pctree-ql.tar.gz + g<K>/mabc-all-apa.zip
```

One archive per product kind for the whole group, members keyed by event id --
which is how `TensorFileSink`, `ClusterFileSink` and `Bee::Sink` already write,
so "one output" needed no new sink.  Smoke run, entries [0,4) of the 48-event
reco1 file: 4 events, one dump + one imaging + one Q/L process, `pctree-ql.tar.gz`
carrying `clustering_tensorset_{172230,444187,388,10550}`, Bee indices 0-3, and
an `rse.json` of `{"172230": [18253,1], ..., "10550": [18255,1]}` -- **two
different runs in one group**, which is the case `rse_map` exists for.  Feeding
that group directly to stage B produced the four `pr_evt<ID>/` directories with
`mabc-pr.zip`, `tracking-pr.root`, `pctree-pr-evt<ID>.tar.gz` and
`nusel-evt<ID>.tsv` -- i.e. exactly the per-event layout, from a group process.

Reading the art file by entry range needed `entry_begin`/`entry_count` on
`SBNDReco1{Frame,OpFlash}Source`, which live in the separate
`WireCell/wire-cell-sbnd-reco1` repo (WCT master moved them out).  Defaults
0 / -1 are the whole file, i.e. today's behaviour.

**Single event and single step still work exactly as before**, because the
per-event drivers are byte-untouched: `run_img_evt.sh`, `run_ql_evt.sh`,
`run_pr_evt.sh`, `run_nusel_evt.sh`, and `run_pr_chain_batch.sh` with
`PR_GROUP_SIZE` unset.  Stage A additionally takes `--group K` for one group and
`--from/--to {img,ql}` for one step; stage B takes an explicit event list, so a
single event out of a large group is just that event named on the command line.

**What stage A does NOT do:** it does not run the stages-1-10 tagger pass that
would emit the selection table (sec 10.2).  The pass itself is proven -- its
`nusel-evt<ID>.tsv` is byte-identical to the full chain's, and
`PR_PIPELINE=<first ten stages> PR_GROUP_SIZE=G ./run_pr_chain_batch.sh` runs it
today -- but it is not yet wired into `run_chain_group.sh` as a fourth step.
Until it is, the selection table comes from stage B, which is where it has
always come from.

### 10.4  Group size

Fixed cost is ~1.0-1.5 s/event (sec 9.1); a group of G captures (1-1/G) of it:
G=8 -> 87 %, G=16 -> 94 %.  Past ~16 there is nothing left to buy and restart
granularity and straggler tolerance get worse -- docs 78/79 record single Q/L
events at 37-190 s, and one of those blocks its slot for the whole time.
**G=16** is the default, with #groups >= #concurrent slots as the constraint
that actually matters for a production sample.  A group that fails is retried
event by event, so one bad event costs one event, not a group.

### 10.5  Two traps this round walked into, recorded so the next one does not

**A group archive must preserve each event's member ORDER.**
`TensorFileSource` finds event boundaries by watching the ident change as it
reads the stream.  Building the group archive with a sorted `tar czf *` splits
every event into two runs of members, and the source then re-emits each ident
TWICE -- the second time with 10 tensors instead of 228.  It does not error: the
second, near-empty copy simply overwrites the first in the output archive, and
the result reads as "event 1 changed in a multi-event run" when nothing about
multi-event was wrong.  `scripts/multi/make_group_pctree.py` copies members in
each source archive's own order for exactly this reason.

**A group's log cannot be sliced by file position.**  The log carries lines from
more than one spdlog sink and they do not arrive in time order: on a measured
run the `TaggerCheckTGM` line stamped 16:28:01.926 sat at file line 272, ahead
of the `loading tensor set ident=48367` line stamped 16:28:01.479 at line 318.
`nusel_extract.py` reads tagger verdicts from that log, and cluster idents
restart every event, so a mis-attributed line lands on a COLLIDING ident and
rewrites a physics column -- measured cost of getting it wrong: 140 differing
rows in a 100-event table.  Fixed at the source: `TaggerCheck{TGM,STM,FC}` stamp
`"evt<ID> "` into exactly the lines that are parsed (empty, and the log text
unchanged, in a one-event job), and `scripts/multi/slice_group_log.py` keys on
that, falling back to TIMESTAMP -- never file order -- for everything else.  The
beam-window gate is passed explicitly (`nusel_extract.py --bw-gate`) rather than
recovered from the log, because without it an out-of-window main reads "0
evaluated, clean" instead of "-1 not evaluated".

### 10.6  What is byte-identical, and the one thing that is not

**Legacy path (production, one event per process) — unchanged.**

| gate | result |
|---|---|
| compiled config, every live consumer of the SBND/common clustering config | **21/21 byte-identical** (`scripts/cfg/compile_consumers.sh` + `cmp_consumers.sh`, doc 77's harness) |
| PR job, 100 mcp1k events, this binary vs the recorded pre-round arm | `pr85` **PASS 200/200 archives**, `pr94` **PASS 100/100 ROOT files**, nusel table **0 differing lines** |
| unit tests | `wcdoctest-clus` 2390, `-root` 4013, `-sio` 11 (new), `-util` 42566 — all pass |

**Group vs per-event, 167 events (100 numuCC + 48 nueCC48 + 19 NCpi0).**
With the **DL (SCN) neutrino vertex OFF**, a group of 8 is **byte-identical** to
eight per-event processes — `pr85` 16/16 archives, `pr94` 8/8 ROOT files —
*including the three events that differ with it on*.

With the DL vertex **ON** (the production default for this driver), **26 of 167
events differ**: mcp1k 5/100, nueCC48 17/48, NCpi0 4/19.  Characterised:

* the differing branches are `T_kine` (66 + 25 + …), `T_rec_charge` and
  `T_proj_data` — every one of them downstream of the vertex POSITION;
* **`T_tagger` never differs** (0 events on mcp1k, 0 on nueCC48): the tagger
  verdicts and the nue/numu BDT scores are unchanged;
* across all **1938 matched bundles** the nusel `label` changes **0 times**, and
  no bundle appears or disappears.

The cause is not new and not this round's: the DL vertex is a python/torch
inference, CLAUDE.md M4 already records it as not bit-stable and keeps it out of
byte-identity gates (`qlport` runs `-A dl_weights=` for exactly this reason),
and doc pr/111 measured its argmax as unstable at 1/10 of a voxel.  A per-event
process gets a fresh interpreter each time; a group process does not, so
whatever state carries between calls is enough to move that argmax on ~16 % of
events.  `SBND_NO_DL=1` on the driver is the switch that demonstrates it.

**This is the owner's call, not a defect to tune away** (escalation rule 7).
Group mode is opt-in (`PR_GROUP_SIZE` unset = today's per-event path, byte for
byte), so nothing in production changes by shipping it.  What the owner decides
is whether a selection-preserving, vertex-position-level difference on ~16 % of
events is acceptable in exchange for the throughput below — and if not, whether
to spend a round making the DL vertex reproducible across events in one process.

**Shared components — the other detectors.**  `clus`, `util` and `sio` are
shared, so:

| gate | result |
|---|---|
| `abtest/ab_compare.sh post_doc79 post_doc76r2c` — PDHD + PDVD clustering, 6 events | **OVERALL PASS** |
| `qlport/ab_check.sh doc76r2 doc79` — uBooNE MABC, 35 events | **zips 35/35 content-identical**; tagger 34/35, the one diff is **ev 6805**, the known binary-LAYOUT-dependent case of doc pr/97 that docs 78 and 79 both recorded before this round |
| SBND Q/L job, 4 events per event, fresh work root vs the recorded `work-mcp1k-ql0819` products | **16/16 archives SAME** (`mabc-all-apa.zip`, both per-APA zips, `pctree-evt<ID>.tar.gz`) |

The PDHD/PDVD **imaging** leg could not be re-run: its NF+SP input frames
(`protodunehd-sp-dnnroi-frames-anode*.tar.bz2`) are no longer in the pdhd work
tree, so `run_events.sh <label> both` skips every event.  The clustering leg was
run instead, which is the leg that exercises everything this round touched in
`clus`/`util` (`MultiAlgBlobClustering`, `BeeSink`, the Bee index).  Imaging's
own config compiles byte-identically (gate 1) and no imaging component was
touched.

The Q/L gate is called out separately because **no PR gate can stand in for it**:
the Q/L job is the one production job that drives `MultiAlgBlobClustering`
through a SHARED `BeeSink` across the per-APA and all-APA nodes -- the exact path
the `flush()` early-return changes -- and the PR gates never open
`mabc-all-apa.zip`.

### 10.7  Throughput

Measured on the 100 numuCC PR events, per-PROCESS wall and peak RSS from
`.time.meta` (concurrency-insensitive, which end-to-end wall is not -- sec 1.1
is this doc's own warning about reading a batch number as a code number):

| arm | processes | summed process wall | median | peak RSS med / max |
|---|---:|---:|---:|---:|
| per-event (`PR_JOBS=8`) | 100 | 526 s | 4 s | 0.49 / **1.34 GB** |
| group of 16 (`PR_JOBS=6`) | **7** | **373 s** | 60 s | 1.19 / **1.26 GB** |

**Summed process wall -29 %**, and 100 process spawns become 7.  The RSS line is
the one to read twice: peak RSS per process does **not** grow with group size
(1.26 GB max at G=16 vs 1.34 GB max per event), i.e. the round did not trade
memory for wall -- which is also independent evidence that the per-event reset
of sec 10.1 is complete, because leaked per-event state would show up here
first.

The two arms ran at different concurrency (8 vs 6) and the box had a peer
session on it, so the end-to-end wall of either arm is not quoted: the summed
per-process numbers above are the comparison that means something.

### 10.8  Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
S=<scratch>

# --- stage A: reco1 -> imaging -> clustering -> Q/L, ONE process per stage per GROUP
./run_chain_group.sh <reco1.root> $S/stageA data --size 16          # all groups
./run_chain_group.sh <reco1.root> $S/stageA data --size 16 --group 3   # one group
./run_chain_group.sh <reco1.root> $S/stageA data --size 4 --to img     # one step
# per-event/per-step working exactly as before: run_img_evt.sh, run_ql_evt.sh,
# run_pr_evt.sh, run_nusel_evt.sh are all byte-untouched.

# --- stage B: the 15-stage PR chain, in groups
PR_GROUP_SIZE=16 PR_JOBS=6 ./run_pr_chain_batch.sh <ql_root> $S/prB data [evts...]
PR_GROUP_SIZE=0  PR_JOBS=8 ./run_pr_chain_batch.sh <ql_root> $S/prA data [evts...]  # per event (default)
# one event out of a large group: name it, and it is its own group of one
PR_GROUP_SIZE=1  PR_JOBS=1 ./run_pr_chain_batch.sh <ql_root> $S/one data 48367

# --- gates
python3 scripts/pr85_hash_gate.py --jobs 8 $S/prA $S/prB     # mabc-pr.zip + pctree
python3 scripts/pr94_root_gate.py          $S/prA $S/prB     # every tracking-pr.root branch
diff <(sort $S/prA/nusel-table.tsv) <(sort $S/prB/nusel-table.tsv)
SBND_NO_DL=1 ...                                             # the DL-vertex control of sec 10.6
./scripts/multi/ql_legacy_gate.sh work-mcp1k-ql0819 $S/qlgate 48301 48367 48565 48895
./scripts/cfg/compile_consumers.sh <cfgroot> <out> && ./scripts/cfg/cmp_consumers.sh <before> <after>
(cd ../../abtest   && ./run_events.sh post_doc76r2c clus && ./ab_compare.sh post_doc79 post_doc76r2c)
(cd ../../qlport/scripts && ./sweep_5384.sh doc76r2 6 && ./ab_check.sh doc76r2 doc79)
```

Arms: `$S/mevt/{m1,m2}-{mcp1k,nuecc48,ncpi0}` (per-event / group-16),
`$S/mevt/{dlA,dlB}` (the DL-off control), `$S/mevt/qlgate2` (Q/L),
`abtest/snap/post_doc76r2c`, `qlport/scripts/sweep/doc76r2`.

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

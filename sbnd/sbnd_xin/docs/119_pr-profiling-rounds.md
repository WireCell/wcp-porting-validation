# doc sbnd_xin/119 — the cross-detector PR profiling campaign: rounds 0, 1 and 2

**Status: round 0 measured. Round 1 measured, gated and FLIPPED. Round 2 measured, gated, and
FLIPPED on the owner's explicit instruction of 2026-09-21 — against this doc's own recommendation,
and for a different reason than doc 30's. Read §6 before quoting any number from it: on SBND the
knob buys no CPU and no memory, and it does cost 12.1 % of the 2-D display.**
Doc 118 part B planned this campaign; this doc executes it. It is the round-numbered perf doc for
the SBND Neutrino chain and the PDHD/PDVD STM+Michel chain, and it supersedes doc 118 part B's
sizing wherever the two disagree — three of doc 118's own premises turned out to be wrong, and
they are corrected in section 2 rather than quietly dropped.

Nothing here re-opens the doc-118 trajectory flip. That flip is production and stays production;
this campaign pays for it.

---

## 0. Repro

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
PD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd

# --- round 0, SBND: per-stage ranking, pre- vs post-flip, from logs already on disk
python3 $SX/scripts/d119/perf_rank.py nuecc work-r3nue-d116s0rep work-r3nue-d116tfull
python3 $SX/scripts/d119/perf_rank.py cv    work-r3cv-d116s0rep  work-r3cv-d116tfull
python3 $SX/scripts/d119/perf_rank.py off   work-r3off-d116s0rep work-r3off-d116tfull

# --- round 0, SBND: the allocator control (3 repetitions of each arm, one event)
for rep in 1 2 3; do for m in glibc tcmalloc; do
  PRDIR=$SX/work-r3nue-d118flip/f002/pr_evt2925 OUTDIR=~/tmp/d119/alloc_${m}_r$rep MODE=$m \
    $SX/scripts/perf/profile_pr118.sh
done; done

# --- round 0, PDHD: the -nu arm re-measured on the current tree (61 events), and the
#     first CPU attribution CheckSTM_Michel has ever had
ARM=d119hnu MODE=-nu STMFIT=1 PIN=~/tmp/d119-libpin JOBS=12 \
  $PD/docs/scripts/d30_run_pr_arm.sh
python3 $PD/stm/perf/d30_pr_census.py --tsv ~/tmp/d119/d119hnu.tsv $PD/work d119hnu
TAG=d119nu MODE=-nu $PD/profile_pr.sh 029107 12 ~/tmp/d119/pdhd/michel_029107_12.prof
google-pprof --text --cum --focus=CheckSTM_Michel \
  /nfs/data/1/xqian/toolkit-dev/toolkit/build/apps/wire-cell ~/tmp/d119/pdhd/michel_029107_12.prof

# --- round 0, PDVD: the -nu arm, 120 events.  d48nu7 is gone; p100flip holds exactly the
#     120 rows of pdvd/scripts/perf_manifest.tsv (verified row for row).
ARM=d119vnu MODE=-nu STMFIT=1 SRC=p100flip PIN=~/tmp/d119-libpin JOBS=12 \
  $PD/stm/perf/d30_run_pdvd_arm.sh
python3 $PD/stm/perf/d30_pr_census.py --tsv ~/tmp/d119/d119vnu.tsv \
  /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work d119vnu

# --- round 0, SBND: CPU attribution against the arm's OWN precompiled production config
PRDIR=$SX/work-r3nue-d118flip/f002/pr_evt2925 OUTDIR=~/tmp/d119/sbnd_cpu MODE=cpu \
  $SX/scripts/perf/profile_pr118.sh
# sanity: exactly ONE "PROFILE: interrupts/evictions/bytes" line in the log.  Two means the
# wrapper profiled itself and clobbered the real profile -- see sec 3.1.

# --- rounds 1 and 2: four arms on the same 62 events, one lever each.  ctl2 is the NULL PAIR
#     and is not optional: it is what says a 2 % delta is not a measurement.
#     Run them back to back on an idle box -- a concurrent arm contaminates the comparison.
for L in ctl ctl2 tcm pad; do for S in nuecc cv off; do
  LEVER=$L JOBS=8 $SX/scripts/d119/stageB_lever.sh $S
done; done
python3 $SX/scripts/d119/lever_gate.py ctl2     # the null pair -- read this first
python3 $SX/scripts/d119/lever_gate.py tcm      # round 1 product identity
python3 $SX/scripts/d119/lever_gate.py pad      # round 2 product identity
python3 $SX/scripts/d119/lever_cost.py          # cost: null pair, then both levers

# --- round 2's FLIP, and the two gates that carry the measurement onto production.
#     The measurement used SBND_TRACKFIT_JSON; production reads the in-tree file.  Different
#     file, different path -- so the 62-event gate does not transfer without these.
python3 $SX/scripts/d119/tf_key_gate.py                  # G2: 49 live keys identical
LEVER=flip JOBS=8 $SX/scripts/d119/stageB_lever.sh nuecc # a 62-evt arm with NO override at all
python3 $SX/scripts/d119/lever_gate.py --vs pad flip     # G1: flipped default == measured arm

# --- the tripwire, at every stage of the round
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-20   # PASS 25 before the flip,
                                                                     # DRIFT: sbnd_track_fitting.json after
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21   # PASS 25 on the new reference
```

Toolkit `d7d4da83` (the round-2 flip; round 0 and round 1 changed no toolkit file).
Pin: `~/tmp/d119-libpin`, `libWireCellClus.so` md5 `71ebd5aeb386` — the same binary doc 115, doc
116, doc 117 and doc 118 ran. No C++ was rebuilt in this campaign; every lever is configuration or
process environment.

---

## 1. What round 0 found, in one table

The campaign was planned around a target list taken from doc pdvd/30 and doc 116. Round 0 measured
that list on the current tree. **The ranking is not what the plan assumed.**

| chain | doc 118 part B expected the top target to be | round 0 measured |
|---|---|---|
| SBND Neutrino, post-flip | `TaggerCheckNeutrino` (67 % of the job) | true for the *job*, but **`CreateSteinerGraph` is 67 % of what the flip ADDED** |
| PDHD STM+Michel (`-nu`) | `CheckSTM_Michel`, 28 % of the arm | **`CreateSteinerGraph` 50.3 %**; Michel is 17.4 % and its worst event is 25.7 s, not 174.3 s |

Both detectors now point at the same component. That is the finding of round 0:

> **`CreateSteinerGraph` is the single largest CPU consumer of the PDHD `-nu` arm (50.3 %) and the
> PDVD one (56.8 %), and it is where two-thirds of the SBND flip's added cost went. It is the same
> code on all three detectors.**

**And more than half of it is not the Steiner graph.** `--focus=CreateSteinerGraph::visit` on three
independent profiles puts **`ImproveCluster_2::mutate` — the retile sampler — at 53.3 % (SBND),
57.8 % and 60.2 % (PDHD)** of that stage, with `Aux::sample_live` a further 6–14 % and
`Grapher::find_peak_point_indices`, the graph step proper, at only 5–10 %.

That split matters because the two halves are different targets with different blast radii.
Sampling (`charge_stepped`, flipped for SBND by doc 118 and for PDHD/PDVD by doc pdvd/103 and 108)
decides which points exist; the Steiner graph decides how they are connected. A future round aiming
at "Steiner" should aim at the **sampler** first, and should know that doing so re-opens a
trajectory decision the owner has already made. Nothing in this campaign touches either.

---

## 2. Three corrections to doc 118 part B

Recorded as corrections because doc 118 is committed and someone will read it.

**2.1 `want_2d` is not an SBND lever.** Doc 118 §8 item 1 says of the double-fit in
`check_stm_conditions`: *"SBND has no pad knob, so on SBND it is at full value."* That is wrong.
`check_stm_conditions` is a method of **`TaggerCheckSTM`** (`TaggerCheckSTM.cxx:3605`), and the
SBND Neutrino chain reaches `TrackFitting::do_single_tracking` through
`NeutrinoPatternBase.cxx:1514` and `NeutrinoOtherSegments.cxx:592` instead. SBND *does* run a
`TaggerCheckSTM:pr` stage, and round 0 measured it at **0.056 s of a 17.1 s job — 0.3 %**. So the
ceiling of `want_2d` on SBND is three tenths of one percent, not "full value". It remains a
PDHD/PDVD lever and it is **not taken in round 1**; see §7.

**2.2 The PDHD "174 s / 5.54 GB max event" is two different events.** Doc 118 §7 item 4 asks for a
heap profile of "the PDHD `-nu` max event (174 s / 5.54 GB)". No such event exists. From doc 30's
own committed census (`pdvd/docs/perf/doc30_pdhd_nu.tsv`, read by column name, not index):

| | event | `ms_michel` | `peak_gb` |
|---|---|---:|---:|
| the 174 s | 029107/12 | **174.3 s** | 4.34 |
| the 5.54 GB | 028084/2 | 7.8 s | **7.51 → 5.54** |

The slowest Michel event and the largest-memory event are different events, and the memory one
barely runs Michel at all.

**2.3 `CheckSTM_Michel` is no longer 28 % of the PDHD `-nu` arm.** It is 17.4 %. See §4 — and see
§4.1 for why the honest statement is "the ranking moved", not "we made it faster".

---

## 3. Round 0, SBND — what the flip actually cost, per stage

Instrument: the in-job `TICK:` ladder, already on disk for every doc-116 arm. Control arm is
`s0rep` (byte-identically the baseline configuration run inside the doc-116 environment), never
the doc-115 baseline — doc 116 §14 measured `wall_s` carrying an environment term up to 2.5×, and
the TICK ladder is the instrument that survives it. **Paired per (sub-root, event)**: event ids
restart in every stage-A input file, and 137 ids are duplicated across buckets in the nuecc arm,
so a bare-id join silently compares different events.

`scripts/d119/perf_rank.py`, 2001 nuecc / 2017 cv / 1000 off paired events:

| stage | nuecc before | nuecc after | Δ | cv Δ | off Δ |
|---|---:|---:|---:|---:|---:|
| `TaggerCheckNeutrino:pr` | 10.581 s | 11.508 s | +0.927 | +0.038 | +0.000 |
| `CreateSteinerGraph:pr` | 0.546 s | 1.625 s | **+1.079** | **+0.210** | **+0.221** |
| `CreateSteinerGraph:prrefresh` | 0.323 s | 1.102 s | **+0.780** | **+0.068** | +0.001 |
| `UbooneNueBDTScorer:pr` | 1.875 s | 1.861 s | −0.013 | −0.002 | +0.001 |
| **sum of stages** | **14.348 s** | **17.117 s** | **+2.769 (+19.3 %)** | +0.316 (+10.5 %) | +0.246 (+25.0 %) |

**The two Steiner passes are +1.859 s of the +2.769 s — 67 % of the flip's entire CPU cost**, and
on `cv` and beam-off they are essentially all of it. The tagger's own +0.927 s on nuecc is the
minority share, and it is zero on the other two samples. Doc 116 §14 reported the +19 % correctly;
what it did not do is say where the 19 % went.

Memory, same pairing (`maxrss_kb`, getrusage CHILDREN high-water):

| sample | before p50 / max | after p50 / max |
|---|---|---|
| nuecc | 1.163 / **1.522** GiB | 1.165 / **2.208** GiB |
| cv | 0.443 / 1.347 | 0.466 / 1.369 |
| off | 0.377 / 1.381 | 0.385 / 1.366 |

The memory cost is entirely a nuecc tail: the median does not move on any sample, and cv and
beam-off do not move at all. Exactly **5 of 2001** nuecc events exceed 2 GiB, and they are
`f060/pr_evt11239` (2.208), `f049/pr_evt12202` (2.203), `f188/pr_evt11811` (2.050),
`f100/pr_evt5265` (2.021), `f196/pr_evt6248` (2.006). The MEM ladder's own peak moves far less
(0.978 → 0.990 GiB mean, max 1.445 → 2.038), i.e. part of the tail is outside the clustering node.

### 3.1 SBND CPU attribution, post-flip

`scripts/perf/profile_pr118.sh MODE=cpu` on the heaviest post-flip nuecc event
(`f002/pr_evt2925`, 9 775 samples). This is the first SBND PR profile taken against the arm's own
`.wct-cfg-evt<ID>.json`, i.e. the 15-stage pipeline production actually runs — `profile_pr11.sh`
hardcodes 13 stages and would have profiled a job nobody runs.

| | % of job |
|---|---:|
| `TaggerCheckNeutrino::visit` | 67.5 % |
| → `TrackFitting::do_multi_tracking` | 53.0 % |
| → `PatternAlgorithms::find_proto_vertex` | 50.3 % |
| → `TaggerCheckNeutrino::run_dual_chain_off_pass` | **31.1 %** |
| `CreateSteinerGraph::visit` | 21.5 % |
| → `ImproveCluster_2::mutate` (the retile sampler) | 11.4 % |

The 67.5 % reproduces the TICK ladder's 67 % independently, and `run_dual_chain_off_pass` at
31.1 % confirms doc 117 §7's 27 % estimate of the dual second pass from a different instrument.

**An instrument defect found and fixed in the process.** The first run of this profile produced a
`.prof` with **5 samples**, all Python import machinery. Cause: `profile_pr118.sh` had the
`timecmd.py` wrapper *inside* the profiling environment, so the python process inherited
`LD_PRELOAD` + `CPUPROFILE`, linked `libtcmalloc_and_profiler` itself, and — exiting after its
child — wrote its own 5-sample profile over wire-cell's 10 207-sample one at the same path. The
symptom is **two** `PROFILE: interrupts/evictions/bytes` lines in the log, the second tiny. Fixed by
putting `timecmd.py` outside and the profiling env inside, on `wire-cell` alone; the fixed run
emits one `PROFILE:` line. A profile that small is obviously wrong — one merely *truncated* would
not have been, which is why the fix is commented at the call site rather than just made.

---

## 4. Round 0, PDHD — the `-nu` arm re-measured, and `CheckSTM_Michel` attributed at last

Doc 30's `-nu` profiles are from 2026-09-07. Since then PDHD took the doc-108 trajectory flip
(`f9665bea`), the doc-30-round-3 `proj_pad` flip, and — at **14:53 on the very day doc 30 was
committed at 11:08** — `a01f2d38`, *"turn on unmerge_assoc in the default PR chain"*. So doc 30's
arm and today's production chain are not the same pipeline. A fresh 61-event arm (`d119hnu`, same
inputs, same pin) was run rather than comparing across that gap.

**Current PDHD `-nu`, 61 events, arm sum of stages 1952 s:**

| stage | arm sum | % arm | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| `CreateSteinerGraph` | **982.0 s** | **50.3 %** | 14.05 s | 23.33 s | 56.91 s |
| `TaggerCheckSTM` | 396.2 s | 20.3 % | 5.64 s | 11.09 s | 17.24 s |
| `CheckSTM_Michel` | 339.8 s | 17.4 % | 4.31 s | 11.15 s | 25.72 s |
| `TaggerCheckTGM` | 89.5 s | 4.6 % | 1.29 s | 2.34 s | 5.07 s |

For reference, doc 30's `d30hnupre` on the same 61 events: `ms_stm` 38.9 %, `ms_michel` 28.7 %,
`ms_steiner` 21.8 %, arm sum 3024 s, peak RSS p50 2.69 / max 7.51 GiB (now 1.67 / 3.43 GiB).

### 4.1 What that comparison does and does not license

The arm is 35 % cheaper and its memory tail less than half. **This campaign did not do that**, and
the two numbers are not a before/after of anything measured here: the pipelines differ by
`unmerge_assoc`, and three production flips land in between. The honest claims are:

- **On the chain PDHD runs today**, the ranking is Steiner ≫ STM > Michel. That is measured.
- **Doc 118's premise that Michel is 28 % of the arm is stale.** That is measured.
- *Why* it fell from 28 % to 17 % is **not** established here and would need doc 30's exact
  pipeline re-run, whose arm directories no longer exist. Not worth a round.

### 4.1b PDVD — the cleaner comparison, and the same answer

PDVD's `-nu` pipeline did **not** change (`unmerge_assoc` was a PDHD commit), so PDVD's 120-event
arm is comparable to doc 30's like for like — same events, same pipeline, same source pctrees
(`d48nu7` is gone; `p100flip` holds exactly the 120 manifest rows and was verified row-for-row):

| stage | doc 30 (`d30vnupre`) | now (`d119vnu`) | Δ |
|---|---:|---:|---|
| `CreateSteinerGraph` | 1114.9 s (42.0 %) | **1585.1 s (56.8 %)** | **+42 %** |
| `TaggerCheckSTM` | 564.1 s (21.2 %) | 383.7 s (13.8 %) | −32 % |
| `CheckSTM_Michel` | 546.3 s (20.6 %) | 385.0 s (13.8 %) | −29 % |
| arm sum | 2656 s | 2788 s | +5 % |
| peak RSS p50 / max | 1.41 / 3.44 GiB | 1.28 / 3.39 GiB | — |

On the one detector where the comparison is clean, the trajectory flips **moved cost into
`CreateSteinerGraph`** (+42 %) and out of the two taggers, for a roughly flat arm total. That is
the same direction as the SBND measurement in §3, arrived at independently, and it is why §1's
conclusion is stated as a cross-detector one rather than an SBND one.

### 4.2 `CheckSTM_Michel` CPU attribution — the deliverable doc 118 named

Every gperftools profile in the tree was `-stm`; this stage had never been attributed. Two events,
`--focus=CheckSTM_Michel`:

| inside `CheckSTM_Michel::visit` | 029107/12 | 028084/28 |
|---|---:|---:|
| `PatternAlgorithms::find_proto_vertex` | 94.3 % | 97.0 % |
| → `TrackFitting::do_multi_tracking` | 64.7 % | 83.8 % |
| → → `TrackFitting::dQ_dx_multi_fit` | 34.5 % | 42.1 % |
| `PatternAlgorithms::find_other_segments` | 11.0 % | 47.6 % |
| `TrackFitting::form_map_graph` | 25.7 % | 36.3 % |
| `TrackFitting::do_single_tracking` | 27.0 % | 11.4 % |

**`CheckSTM_Michel` is not a Michel algorithm by cost — it is a full pattern-recognition re-run.**
Essentially 100 % of it is `find_proto_vertex` on the re-seeded cluster, and two-thirds to
five-sixths of that is the multi-track fit. Any future attempt to make it cheaper is an attempt to
avoid or reuse a PR pass, not to tune a Michel cut. Whole-job context on 028084/28:
`CreateSteinerGraph` 40.7 %, `do_single_tracking` 30.2 %, `TaggerCheckSTM` 29.2 %,
`ImproveCluster_2::mutate` 24.5 %, `CheckSTM_Michel` 18.8 %.

---

## 5. Round 1 — tcmalloc on the SBND PR chain

**The finding.** SBND's PR runners have never preloaded tcmalloc, while PDHD, PDVD and SBND's own
`run_clus_evt.sh` all do. Doc 118 §6 established that the *conditional shape* of
`run_pr_evt.sh:307-310` is deliberate — the `-stm` / `-tgm` / bare `-p` arms are frozen A/B
comparison arms — and that the absence of tcmalloc is a separate, unexplained thing.

**Measurement**, before any change: SBND nuecc event `f002/pr_evt2925`, **three repetitions of each
arm**, same pinned binary, same compiled config, nothing else on the box.

| | rep 1 | rep 2 | rep 3 | mean |
|---|---:|---:|---:|---:|
| in-job TICK total, glibc | 45.611 s | 45.366 s | 43.960 s | **44.98 s** |
| in-job TICK total, tcmalloc | 37.524 s | 38.075 s | 37.060 s | **37.55 s** |
| peak RSS, glibc | 1.388 | 1.430 | 1.430 GiB | 1.416 |
| peak RSS, tcmalloc | 1.443 | 1.446 | 1.443 GiB | 1.444 |

**−16.5 % CPU for +2.0 % peak RSS**, with the two sets of three non-overlapping. The `wall_s` of
the first glibc rep was 68 s against a 45.6 s TICK — a cold page cache — which is precisely why the
claim is made on the in-job ladder and not on wall.

**Arm-level confirmation**, all 62 gate events, ctl vs tcm arms run back to back at `JOBS=8`:

| sample | arm TICK sum, ctl → tcm | maxrss mean |
|---|---|---|
| nuecc (24 evt) | 341.0 → 303.8 s (**−10.9 %**) | 1.187 → 1.223 GiB (+3.0 %) |
| cv (18 evt) | 58.3 → 53.6 s (**−8.1 %**) | 0.813 → 0.824 GiB (+1.3 %) |
| beam-off (20 evt) | 23.2 → 19.3 s (**−16.7 %**) | 0.448 → 0.446 GiB (−0.5 %) |

The arm figures are smaller than the single-event −16.5 % because the arms ran 8-wide and the
controlled experiment ran alone; the production-realistic number is the arm one. `wall_s` on the
nuecc arm reads **+19.6 %** over the same pair — its p50 goes 11 → 23 s while TICK goes 7.33 →
6.78 s — which is batch scheduling, not the allocator, and is exactly why no claim here is made on
wall.

**Gate.** `scripts/d119/lever_gate.py tcm`, allowance = *nothing* but the stopwatch:

```
cv    : 82 files compared, 10 differ (stopwatch), 0 NOT allowed
nuecc : 120 files compared, 24 differ (stopwatch), 0 NOT allowed
off   : 82 files compared,  1 differs (stopwatch), 0 NOT allowed
PASS
```

284 files across 62 events; every reconstruction product byte-identical. An allocator that moved a
physics product would be a toolkit defect, so round 1's allowance was set to zero deliberately.

**And the null-pair column, which is what makes that PASS mean something.** `lever_gate.py` is a
new fork with new allowance logic; a gate that sees nothing passes everything. Running it on
`ctl` vs `ctl2` — the same configuration twice — gives **exactly the same counts as tcm**:

| | cv | nuecc | off |
|---|---|---|---|
| ctl vs ctl2 (one configuration, twice) | 10 | 24 | 1 |
| ctl vs tcm (the allocator) | **10** | **24** | **1** |
| ctl vs pad (a lever that really moves a product) | 28 | 48 | 21 |

The first two rows are identical, so tcmalloc's differences are precisely what run-to-run variation
produces; the third row shows the same gate does report a real change when there is one. Sensitivity
and specificity, both measured, neither assumed.

**What shipped.** `run_pr_chain_batch.sh` gains `SBND_PR_TCMALLOC` (default on) at its two
full-PR-chain preload sites, `:1963` and `:2156`. It **joins** the preload
(`LD_PRELOAD="$PYLIB:$SBND_TCMALLOC_LIB"`) — assigning would drop libpython and silently demote the
job to the geometric vertex. `run_pr_evt.sh:318` is deliberately **not** touched, so the frozen A/B
arms keep their exact process environment. Escape hatch: `SBND_PR_TCMALLOC=0`.

---

## 6. Round 2 — `proj_pad` for SBND: measured, recommended against, FLIPPED anyway

C++ default −1 = OFF. `TrackFitting.cxx` is shared, so the *absent key* is what kept SBND on the
untrimmed path while PDHD and PDVD have carried `proj_pad_wire 3 / proj_pad_time 3` since doc 30
round 3. It is read by `TaggerCheckNeutrino` as well as `TaggerCheckSTM`, so SBND is in scope, and
the owner pre-authorised the flip on 2026-09-20 **on the doc-30 terms**.

**The doc-30 terms are not the terms SBND gets.** Measured on the same 62 events, ctl vs pad:

| | PDHD (doc 30 round 3) | SBND (this round) |
|---|---|---|
| core-s / arm TICK | **−28.8 %** | **+2.2 % nuecc, +1.3 % cv, +1.5 % off** |
| worst-event peak RSS | **5.00 → 2.19 GB (−56 %)** | 1.428 → 1.424 GiB (**−0.3 %**) |
| mean peak RSS | — | 1.187 → 1.186 GiB (**−0.1 %**) |
| proj cells kept | **2.4 – 2.9 %** | **87.9 %** |

**The mechanism.** The knob only pays when the fitted-charge map is mostly cells the fit does not
predict. On PDHD that was a 40× over-storage — ~362 k cells per call, 2.4 % of them carrying a
prediction. On SBND the map holds **7 968 cells per event** and the pad removes 12 % of them. For
scale, PDHD's `-nu` arm *with the pad already on* stores 9 153 cells per event: **SBND without the
knob is already tighter than PDHD is with it.** There is nothing for the filter to remove, so all
it does is add a dilate-and-test pass over cells it then keeps.

**The apparent +2 % CPU is not even a real cost — it is inside the noise floor.** It appears
uniformly across stages `proj_pad` cannot touch (`UbooneNueBDTScorer` +0.034 s,
`CreateSteinerGraph:pr` +0.021 s, `:prrefresh` +0.026 s), which is the signature of run-to-run
variation. The null-pair arm `ctl2` measures that variation directly:

| arm TICK sum vs ctl | nuecc | cv | off |
|---|---:|---:|---:|
| **ctl2 — the same configuration, twice** | **+0.2 %** | **−1.6 %** | **−2.6 %** |
| pad | +2.2 % | +1.3 % | +1.5 % |
| tcm (round 1, for contrast) | −10.9 % | −8.1 % | −16.7 % |

`pad` is inside the null pair on every sample; `tcm` is outside it on every sample. So round 2's
CPU effect is **not measurable**, not "+2 %".

**Gate result (for the record, since it was run before the decision):**
`scripts/d119/lever_gate.py pad` — 284 files over 62 events, **0 differences outside the
allowance**; `T_rec_charge`, every tagger tree, `mabc-pr.zip`, the pctree and the non-`proj` calib
dump are byte-identical, and only `T_proj_data`, the calib `proj` block and the provenance move.
So the flip was *available*. It was not taken.

**Reconciling this with what doc 118 actually wrote.** Doc 118 §9 set the condition literally:
*"Round 2 therefore measures **and** flips, on one explicit condition: only if the per-tree gate
shows every physics product byte-identical."* That condition **was met**. What doc 118 did not
state, because it assumed it, was that the PDHD payoff would carry over. It does not.

### 6.1 The recommendation, and the owner's decision against it

**This doc recommended not flipping.** The authorisation of 2026-09-20 was for the doc-30 trade —
give up display fidelity, get back a quarter of the CPU and half the memory. SBND's measured trade
is: give up 12.1 % of the display cells and get back nothing measurable. Spending a real product
for a saving that was assumed and then measured to be absent is not the trade that was authorised.

**The owner flipped it on 2026-09-21 regardless**, with the reason *"since the output did not
change"*. That reason needs one correction on the record, and it does not change the decision:

> The **physics** output does not change — `T_rec_charge`, every tagger tree, `mabc-pr.zip`, the
> pctree, `nusel` and the non-`proj` calib dump are byte-identical across 62 events. The
> **display** output does change: `T_proj_data` and the calib `proj` block lose **12.1 %** of
> their cells (7 968 → 7 002 per event). That is the product the knob exists to trim, so it is not
> a free change; it is a cheap one.

**The defensible reason, and the one recorded in the config comment, is cross-detector
consistency.** After doc pdvd/103, doc pdvd/108 and doc sbnd_xin/118 all three detectors run the
same trajectory family, and SBND was the only one whose 2-D display settings still differed — which
is a real friction for anyone comparing scans across detectors. The config comment states plainly
that there is **no CPU or memory saving on SBND** and that the 12.1 % display loss is real, so that
nobody later cites this key for a benefit it does not deliver here.

### 6.2 Gates on the flip itself

Flipping moves the knob from a runtime override (`SBND_TRACKFIT_JSON`, how it was measured) to the
in-tree default (`cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json`, how production reads it).
Those are different files and different code paths, so the 62-event gate above does not by itself
transfer. Two checks close that:

**G2 — fit-JSON key identity** (`scripts/d119/tf_key_gate.py`,
`docs/119_figs/119_gate_tfkeys.txt`): after stripping `_`-prefixed comment keys, which
`load_trackfitting_config` skips, the flipped production file and the measured file carry **all 49
live keys identical in name and value**. So the measured configuration *is* the production
configuration.

**G1 — the flipped default reproduces the measured arm** (`lever_gate.py --vs pad flip`,
`docs/119_figs/119_gate_flip.txt`): a fresh 62-event arm run with **no `SBND_TRACKFIT_JSON` at
all**, gated against the `pad` arm at provenance-only allowance — **0 differences outside
provenance** on all three samples (284 files, 62 events). A key list proves the values agree; only
this proves the path that reads them does. And because "forgiven by an allowance" is not the same
as "identical", a direct tree-level comparison on 8 nuecc events confirms the crux: the only
differing tree is `Trun`, and **`T_proj_data` itself is byte-identical** between the flipped default
and the measured arm.

*An instrument note, because it cost a re-run.* The first flip arm was launched while
`sbnd_track_fitting.json` was still being rewritten, so some of its events may have read a torn
file; it is kept as `work-r3nue-d119flip-torn` and used for nothing. `stageB_lever.sh LEVER=flip`
now prints the fit JSON's md5 **before and after** the arm, and the arm that G1 uses shows
`97668454d5cb` at both ends.

**G3 — the tripwire fires, on exactly one artifact.** `prod_cfg_gate.py --ref ref/prod-2026-09-20`
returns `DRIFT: sbnd_track_fitting.json` and nothing else — uBooNE (a frozen reference), PDHD and
PDVD byte-identical. This is the hole doc 118 opened the tripwire to cover (the runtime fit JSONs
are artifacts 22–24), working as designed and bounding the blast radius to one file. New reference
**`ref/prod-2026-09-21`** is PASS 25/25; exactly **1 of 25** hashes differs from `prod-2026-09-20`,
which is kept and still reports the drift, so it remains a valid record of the pre-flip point.

### 6.3 If this is ever revisited

`docs/119_figs/119_tf_sbnd_pad.json` and `scripts/d119/stageB_lever.sh LEVER=pad` re-run the whole
measurement in about three minutes. Deleting the two keys restores the pre-flip display exactly
(C++ default −1 = OFF). If SBND's fitted-charge map ever grows — a larger readout window, a wider
fit, a denser sampler — the saving that is absent today could appear, and the keep fraction
(87.9 % now) is the number to re-measure.

---

## 6.4 The tripwire, and a hole round 1 opens in it

`prod_cfg_gate.py --ref ref/prod-2026-09-20` was run **before** any edit in this round and gave
**PASS, 25 artifacts** (`docs/119_figs/119_gate_pre.txt`), so nothing this round reports has
inherited drift mixed into it. It was run again after round 0 and round 1 — still PASS 25
(`119_gate_post.txt`), because neither touched a compiled artifact — and a third time after the
round-2 flip, where it reports the single expected drift (§6.2 G3).

**A new reference generation IS needed, because round 2 flipped.** `sbnd_track_fitting.json` is
consumer artifact 22, so the tripwire reports `DRIFT: sbnd_track_fitting.json` against
`prod-2026-09-20` — and nothing else, which is the blast-radius statement (§6.2 G3). The new
reference is **`ref/prod-2026-09-21`**; `prod-2026-09-20` is kept, as every earlier generation is.
No jsonnet changed, so the other 24 artifacts are bit-identical across the two generations.

Had round 2 gone the other way — measured and declined — no new generation would have been needed
at all, and that is worth knowing: this round's *measurements* moved nothing, only its flip did.

**But round 1 shipped a production change the tripwire cannot see.** `run_pr_chain_batch.sh` is a
runner script, and **no runner is among the 25 consumers** — the set holds compiled configs and the
three runtime fit JSONs. So `SBND_PR_TCMALLOC` defaulting on changes what production does while
`prod_cfg_gate.py` reports 25/25. That is the *same shape* as the two holes doc 118 closed (the
runtime fit JSONs, which no compile could see; and the LArSoft 1-step chain, which no compiled
artifact covered).

It is named here rather than closed, because closing it is a real decision with its own cost: the
PR runners are large, frequently edited, and mostly contain A/B scaffolding whose churn would make
the tripwire noisy. The honest statement is that **the process environment of the PR chain is
currently ungated**, and that the argument for leaving it so is convenience, not safety. A minimal
version — hashing only the preload/allocator block of the three PR runners — would close it without
the churn, and is the recommended next tripwire change.

---

## 7. What this campaign did not take, and why

- **`CreateSteinerGraph`, and inside it the retile sampler** — round 0's own finding, and the
  largest target on all three detectors. Not attempted: it is the component the doc-118 and
  doc-108 trajectory flips deliberately made more expensive in exchange for trajectory and vertex
  accuracy, so making it cheaper means changing what it computes. A physics round with its own
  owner decision, not a perf lever. §1 says which half to aim at if it is ever opened.
- **`want_2d`** — PDHD/PDVD only (§2.1), and its value there is now bounded by a `TaggerCheckSTM`
  share of 20.3 %, of which it can reach only the duplicated round-1 fits. Worth a measurement, not
  worth a C++ change on an unmeasured payoff — doc 30 §12.2 is the standing lesson about that.
- **SBND's dual second pass** (doc 118 §9.1) — **31.1 %** of the post-flip nuecc job by direct
  profile (§3.1), against doc 117 §7's 27 % from the scoreboard; but gating it
  changes which events get the snap, so it needs grading against doc 107's metrics, not a
  byte-identical gate. §5 rule 1: a separate round with its own owner decision.
- **The structural `fill_fitted_charge_2d` restriction** (doc 118 §9.2) — changes which cells exist
  in every downstream product.
- **Doc 30 §12.2's three measured-and-rejected levers** — the `APAFacePlane` pointer cache, the
  row-set `std::move`, and the `merge_fitted_charge_2d` "third copy" move. Do not re-propose.

---

## 8. Recommended next step

**Round 3 as doc 118 §9.1 framed it — making SBND's dual second pass cheaper — is now the best-sized
remaining target**, and round 0 strengthened the case: it is 31.1 % of the post-flip nuecc job
measured directly, larger than doc 117's 27 % estimate, and it is the only large consumer that is
not the trajectory the owner just chose. It needs grading against doc 107's metrics rather than a
byte-identical gate, so it is an owner decision, not a perf round — the census it needs
(`vertex_scoreboard.dual_chain.{agree,transferred,d,off_ms}`) is already recorded per event in every
arm and costs nothing to read.

Two smaller items, both cheap:

1. **Close the runner tripwire hole** (§6.1) in its minimal form — hash the preload/allocator block
   of the three PR runners into the consumer set. Round 1 is the first production change in this
   arc that `prod_cfg_gate.py` cannot see.
2. **Retention.** This round added five SBND arms of 62 events (`d119ctl`, `d119ctl2`, `d119tcm`,
   `d119pad`, `d119flip`), a 61-event PDHD arm and a 120-event PDVD arm. Every number from them is
   in `docs/119_figs/`, so they are re-derivable and none is a scan record. Two are kept only as
   records of measurements that had to be discarded, and are named so nobody reads them as data:
   `d119ctl2contended` (the null pair, run while the PDVD arm shared the box) and
   `work-r3nue-d119flip-torn` (a flip arm that read `sbnd_track_fitting.json` while that file was
   being rewritten — see the md5-before/after lines `stageB_lever.sh LEVER=flip` now prints).

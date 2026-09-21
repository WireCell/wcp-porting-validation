# doc sbnd_xin/119 — the cross-detector PR profiling campaign: rounds 0, 1, 2, 3, 4 and 5

**Status: round 0 measured. Round 1 measured, gated and FLIPPED. Round 3 (§9) measured, gated and
TAKEN — the first round of this campaign to rebuild C++: two byte-identical levers in the
projection hot path, gated on **all three detectors** (SBND 62 evt, PDHD 61, PDVD 120) and worth
−14.9 / −9.5 / −6.7 % in-job CPU on SBND nuecc / cv / beam-off and −4.1 / −4.4 % on the PDHD and
PDVD `-nu` arms, no knob because there is no behaviour to switch off.
The round-3 the campaign had recommended (sharing the dual chain's association lattice) was
REFUTED by the code read and is closed, not deferred — §9.1. Round 2 measured, gated, and
FLIPPED on the owner's explicit instruction of 2026-09-21 — against this doc's own recommendation,
and for a different reason than doc 30's. Read §6 before quoting any number from it: on SBND the
knob buys no CPU and no memory, and it does cost 12.1 % of the 2-D display.**
Doc 118 part B planned this campaign; this doc executes it. It is the round-numbered perf doc for
the SBND Neutrino chain and the PDHD/PDVD STM+Michel chain, and it supersedes doc 118 part B's
sizing wherever the two disagree — three of doc 118's own premises turned out to be wrong, and
they are corrected in section 2 rather than quietly dropped.

**Round 4 (§10) is the memory round — the first of this campaign to measure memory at all, and it
measures rather than changes anything.** It corrects a claim made in the round-3 hand-off: the
62-event gate manifest showing no event above 2 GiB is a manifest-size artefact, not evidence that
doc 116's 2.2 GiB tail is gone. Re-running the five named tail events at current production puts
three of them still above 2.0 GiB (all 14 arm runs `rc=0`). Attribution: `CreateSteinerGraph::visit` is 70–78 % of peak
live heap on both detectors profiled, and the retile sampler — already round 0's #1 CPU target —
is a stable 14–21 % of it. **The CPU target and the memory target are the same object.** §10.5
records two readings this round formed and then retracted, and the rule that kills both.

**Round 5 (§11) is the allocator round, and it opens by retracting its own premise.** The
−0.74 GiB that round 4 used to justify it was an instrument mix — a ladder number from a
*profiled* jemalloc run set against a `getrusage` number from an unprofiled tcmalloc run — and one
sentence of §10.4 is retracted outright (§11.1). Measured like for like, on four allocator arms
through one invocation: **jemalloc 5.3.0 is −18.8 % mean and −19.1 % max peak RSS on the seven tail
events at no CPU cost, taking all three events above 2 GiB below it, and it is byte-identical on the
62-event manifest** with a null pair proving the gate is specific. It buys **nothing on the gate
manifest** (+0.1 % on nuecc) — the win exists only where a cap is threatened — and it is **+4 to
+7 % CPU on jobs of a few seconds**, which is a few hundredths of a second each. **Not flipped** — changing an
existing production default is §5 rule 1, the owner's call, as round 1's was. A mechanism control
(tcmalloc at `TCMALLOC_RELEASE_RATE=10`) shows page-return policy is only about a quarter of it and
is a cheaper partial option in its own right. Shipping regardless: **§6.4's runner tripwire hole is
closed**, and with it a second hole found *inside* `prod_cfg_gate.py` — adding a consumer to the
gate's own input set had silently never taken effect (§11.4).

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

```bash
# --- round 3: the C++ round.  Its arms differ from `flip` in the BINARY and nothing else, so
#     stageB_lever.sh now REFUSES r3/r3b on the pre-round-3 pin and the old levers on the new one.
LEVER=r3  JOBS=8 $SX/scripts/d119/stageB_lever.sh nuecc   # and cv, off; then LEVER=r3b likewise
python3 $SX/scripts/d119/lever_gate.py --vs flip r3       # sec 9.4 G1  -- PASS, 62 events
python3 $SX/scripts/d119/lever_gate.py --vs r3   r3b      # sec 9.4 G2  -- the null pair
python3 $SX/scripts/d119/lever_gate.py --vs ctl  r3 nuecc # sec 9.4 G3  -- MUST FAIL (sensitivity)
python3 $SX/scripts/d119/r3_cost.py                       # sec 9.3
# --- the other two detectors: round 0's arms re-run on the round-3 pin, then gated (sec 9.4 G4)
ARM=d119hr3 MODE=-nu STMFIT=1 PIN=~/tmp/d119r3-libpin JOBS=10 $PD/docs/scripts/d30_run_pr_arm.sh
ARM=d119vr3 MODE=-nu STMFIT=1 SRC=p100flip PIN=~/tmp/d119r3-libpin JOBS=10 \
    $PD/stm/perf/d30_run_pdvd_arm.sh
python3 $PD/stm/perf/d30_hash_gate.py $PD/work  d119hnu d119hr3   # PASS  61 / FAIL 0
python3 $PD/stm/perf/d30_hash_gate.py $PV/work  d119vnu d119vr3   # PASS 120 / FAIL 0
/nfs/data/1/xqian/toolkit-dev/toolkit/build/clus/wcdoctest-clus   # sec 9.4 G4
# before/after profile on ONE event, same config, two pins (sec 9.2):
for a in flip r3; do PIN=$HOME/tmp/$([ $a = flip ] && echo d119-libpin || echo d119r3-libpin)
  LD_LIBRARY_PATH=$PIN PRDIR=$SX/work-r3nue-d119$a/f002/pr_evt2925 OUTDIR=~/tmp/d119r3-prof/$a \
      $SX/scripts/perf/profile_pr118.sh; done
```

```bash
# --- round 5, the allocator.  FOUR arms through ONE invocation on the SAME events, back to back
#     on an idle box.  MALLOC_CONF is cleared inside the script: round 4's jemalloc runs had the
#     heap sampler on, and that perturbation is exactly what sec 11.1 retracts.
for A in tcm glibc jem rel; do ALLOC=$A JOBS=7 $SX/scripts/d119/stageB_alloc.sh tail; done
ALLOC=jem JOBS=8 $SX/scripts/d119/stageB_alloc.sh nuecc    # and cv, off -- the 62-event gate arm
ALLOC=tcm JOBS=8 $SX/scripts/d119/stageB_alloc.sh nuecc    # and cv, off -- its same-binary control
python3 $SX/scripts/d119/r5_alloc.py            > docs/119_figs/119_r5_alloc.txt   # sec 11.2
python3 $SX/scripts/d119/lever_gate.py --vs r3 r3b   # sec 11.5 NULL PAIR -- run it FIRST
python3 $SX/scripts/d119/lever_gate.py --vs r3 jem   # sec 11.5 the byte gate -- PASS, 62 events
# --- round 5, the tripwire hole (sec 11.4).  The NEW detection is the point: before this round
#     adding a consumer to compile_consumers.sh was silently never adopted.
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21    # NEW: runner_alloc.txt, rc=1
cp -a $SX/ref/prod-2026-09-21 $SX/ref/prod-2026-09-21b                # M13: never refresh in place
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21b --refresh   # 26 (1 added)
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21b   # PASS 26/26
# --- round 5, the dual-chain census sec 8 kept deferring.  Read-only, nothing re-run.
python3 $SX/scripts/d117/dual_chain_census.py \
    "today (d119r3) = current production:nuecc:work-r3nue-d119r3:" \
    "today (d119r3) = current production:cv:work-r3cv-d119r3:" \
    "today (d119r3) = current production:off:work-r3off-d119r3:" \
    > docs/119_figs/119_r5_dual_census.txt                            # sec 11.6
```

Toolkit `d7d4da83` for rounds 0-2 (the round-2 flip; rounds 0 and 1 changed no toolkit file).
Pin `~/tmp/d119-libpin`, `libWireCellClus.so` md5 `71ebd5aeb386` — the same binary doc 115, doc
116, doc 117 and doc 118 ran. **Round 3 is the first round of this campaign that rebuilds C++**:
pin `~/tmp/d119r3-libpin`, md5 `4ff75274e43e`. Rounds 0-2 remain readable against the old pin, and
the driver enforces the pairing in both directions so neither can be re-run on the wrong one.

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

> **SUPERSEDED BY §11.7 — read that ranking, not this one.** Round 5 finally ran the census this
> section kept deferring (§11.6) and it *demoted* the dual chain: the second pass is right about
> two times in three on the events where it moves the pick (p = 1.1e-10 on 2 001 nuecc events), so
> it is the largest number and the worst lever, and it cannot be had byte-identically at all.
> Round 5 also closed item 1 below — the runner tripwire hole (§11.4). Item 2, retention, is still
> open and round 5 adds ~0.6 GB of arms to it.
>
> **Still open after round 4.** §10 is a memory round and does not touch this ranking. The dual
> chain remains the largest CPU target and still needs doc 107 grading rather than a byte gate.
>
> **EXECUTED, and the recommendation below was partly refuted — see §9.** The concrete round-3A
> proposal (share the association lattice between the two passes) died on the code read: the
> lattice is rebuilt from the fit points, the OFF pass owns a separate fitter, and it runs with
> exclusion off. §9 took a different, byte-identical target found by the same read, for
> −14.9 / −9.5 / −6.7 % in-job CPU. What §8 says below about the dual chain itself still stands.

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

---

## 9. Round 3 — the projection hot path

**Status: TAKEN. Byte-identical on all three detectors — SBND 62 events, PDHD 61, PDVD 120 —
and worth −14.9 / −9.5 / −6.7 % (SBND) and −4.1 / −4.4 % (PDHD / PDVD) in-job CPU. CPU only.**
Toolkit commit adds no config key and no knob: there is nothing to turn off, because there is no
behaviour to turn off.

### 9.1 The recommended round 3 was refuted by the code read, and that is the round's first result

§8 recommended attacking `run_dual_chain_off_pass` (31.1 % of the nuecc job), and the concrete
proposal was **3A: share the point↔cell association lattice between the production pass and the
OFF pass, byte-identically, if that lattice is seed-independent.** The first step was a code read
answering exactly that question. The answer is **no**, for three independent reasons:

1. **The lattice is a function of the fit points, not of cluster geometry.** `form_map_graph`
   *clears* `m_3d_to_2d` and `m_2d_to_3d` at entry (`TrackFitting.cxx:4482-4483`) and rebuilds them
   from `segment->fits()` — the current fitted trajectory. A different vertex seed gives a
   different trajectory gives a different lattice. It is also re-cleared and rebuilt several times
   *within* one pass (`do_multi_tracking` runs three `form_map_graph` passes).
2. **There is no shared object to populate.** The OFF pass constructs its own `TrackFitting`, its
   own `PR::Graph` and its own `PatternAlgorithms` copy (`TaggerCheckNeutrino.cxx:4639-4655`).
3. **The two passes do not even ask the same question.** The OFF pass sets
   `pattern_algos.m_fit_exclusion = false` (`:4657`) where production runs `fit_exclusion=true`,
   and `form_map_graph(flag_exclusion, …)` branches on precisely that. Even at identical fit
   points the two lattices would differ.

So 3A cannot be built, and 3B (skip the pass by a predicate) remains what §8 said it was — a
physics decision needing doc 107 grading, not a perf round. **3A is closed, not deferred.**

**But the same code read re-aimed the round at something better.** The functions 3A wanted to share
are expensive for a reason that has nothing to do with the dual chain, and fixing that reason helps
the production pass, the OFF pass, and PDHD/PDVD, all at once.

### 9.2 Where the time in `form_point_association` actually went

Profiles: `docs/119_figs/119_prof_sbnd_nuecc.txt` (round 0) and
`docs/119_figs/119_prof_r3_before_after.txt` (this round). `form_point_association` was **19.1 %
of the SBND nuecc job**, and it is called once per projected fit point from both passes. Inside it:

| callee | samples (of 9775) | what it was doing |
|---|---:|---|
| `convert_3Dpoint_time_ch` | 820 (44 % of the function) | the projection |
| ⤷ `drift2time` | 405 | **re-deriving two per-face constants, per call** |
| ⤷ `point2wind` | 220 | of which ~139 is `cos`/`sin` **of a per-face constant angle** |
| `find_neighbors_nlevel` | 461 | of which **327 (71 %) is `std::set` insertion**, not graph work |

Two specific defects, both of them work that did not need doing at all:

- **`drift2time` re-derived a job constant on every call.** Its body was
  `xsign = anodeface->dirx(); xorig = anodeface->planes()[2]->wires().front()->center().x();` —
  `planes()` and `wires()` both return **by value**, so each call cloned two containers of
  shared pointers, took and dropped a refcount, and walked to a wire centre, to recover two numbers
  that are fixed for the whole job. The line-level profile shows the cost as
  `std::vector::~vector` 140, `_Sp_counted_base::_M_release` 88, `IWire::center` 120.
- **`point2wind` called `cos(angle)` and `sin(angle)` per point**, where `angle` is a
  per-(apa, face, plane) constant. 139 of its 223 samples were the two transcendentals and the
  `libc_feholdsetround_sse_ctx` fenv save/restore they drag in.

The call site that makes this expensive is **not** the obvious one. `convert_3Dpoint_time_ch` is
called three times per projected point — but also **three times per Steiner-graph neighbour**,
inside `for (auto vertex_idx : total_vertices_found)` at `TrackFitting.cxx:3346`. That inner loop
is why `convert_3Dpoint_time_ch` carries 845 samples while `convert_3Dpoint_wire_cont` — called
once per point on the same branch, under the same `assoc_cont_center` that is on for all three
detectors — carries 1. The ratio is a call count, not a knob being inert; that was checked before
anything was built on it.

### 9.3 The two levers, and what they cost

**L1 — the projection constants move into the memo that already exists.** `Grouping::fastgeom_t`
was built for exactly this purpose and says so in its own comment ("a per-call `std::vector` for
the angles and the by-value `IAnodePlane::faces()` vector dominated their cost. Output-identical:
the memo holds exactly the values those lookups return"). It memoised angle/pitch/centre and
stopped there. L1 adds `cos_angle[3]`, `sin_angle[3]`, `xsign`, `xorig` to the same struct, filled
in the same builder, and adds `drift2time(xsign, xorig, …)` / `point2wind_cs` / `point2wind_cont_cs`
overloads that take them. **Each function body is carried across textually**; only `cos(angle)` and
`sin(angle)` become parameters. The angle-taking forms remain for callers with no memo and now
delegate, so there is one copy of each expression, not two that could drift apart.

**L2 — the BFS stops paying for an order it already has.** `find_neighbors_nlevel` returned
`std::set<vertex_type>`, one red-black node allocation per neighbour, and its `visited` flag array
already guaranteed uniqueness — the set was supplying **order only**. It now returns a vector
sorted once at the end. The ascending order is load-bearing and not incidental: `TrackFitting`
walks the result to build a blob set whose iteration order reaches the output, so returning raw BFS
discovery order would **not** have been byte-identical. Both call sites already used `auto`.

**Cost, 62 events, TICK total** (`docs/119_figs/119_cost_r3.txt`). The baseline is the **`flip`**
arm — production as it runs after round 2 — not `ctl`, which is pre-`proj_pad` and would fold
round 2's delta into round 3's number:

| sample | arm TICK sum | round 3 | **null-pair floor, new binary** |
|---|---|---:|---:|
| nuecc (24 evt) | 352.4 → 300.0 s | **−14.9 %** | +3.0 % |
| cv (18 evt) | 62.1 → 56.2 s | **−9.5 %** | +0.9 % |
| off (20 evt) | 24.2 → 22.6 s | **−6.7 %** | −0.1 % |

All three sit clearly outside the floor, which is why the floor was re-measured on the **new**
binary (`r3b`) rather than inherited from round 2's — a noise floor does not survive a rebuild.

**And on the other two detectors** (`docs/119_figs/119_cost_r3_xdet.txt`; the edited files are
shared `clus` code, so PDHD and PDVD are affected whether or not anyone measured them). Baseline is
round 0's own arm, re-run on the round-3 pin, **claim instrument `node_core_s`** — the before-arms
ran at `JOBS=12` and the after-arms at `JOBS=10`, so `wall_s` is not comparable and is not quoted:

| arm | in-job compute | | peak RSS |
|---|---|---:|---|
| PDHD `-nu`, 61 evt | 1982.9 → 1902.4 s | **−4.1 %** | 1.765 → 1.765 GB (−0.0 %) |
| PDVD `-nu`, 120 evt | 2835.0 → 2710.8 s | **−4.4 %** | 1.421 → 1.420 GB (−0.1 %) |

Per stage, both detectors move in the same places and by the same mechanism: `TaggerCheckSTM`
−7.3 / −8.5 %, `CheckSTM_Michel` −8.0 / −9.0 %, `CreateSteinerGraph` only −2.2 / −2.9 %. **The
smaller headline is expected and is not a weaker result**: PDHD and PDVD spend 50–57 % of the job
in `CreateSteinerGraph`, which barely touches the projection, where SBND's nuecc job is dominated
by the PR/fit path that does. The lever is the same size; the job mix is different.

**Memory: this is a CPU round and memory did not improve.** Peak RSS moved **+1.1 / +0.8 / +0.7 %**
(nuecc max 1.402 → 1.429 GiB) against a null-pair floor of +0.1 / +0.4 / +0.0 %. The nuecc figure
is small but probably real. My first explanation — a `std::vector` grown by doubling replacing
exactly-sized red-black nodes — **does not survive the cross-detector arms**, which are flat
(−0.0 % PDHD, −0.1 % PDVD) on the same code. What separates them is the allocator: the SBND arms
run glibc (`SBND_PR_TCMALLOC=0`, to match `flip`), PDHD and PDVD preload tcmalloc. So the drift
correlates with the allocator, not with the change, and it is recorded as an observation and not
as a mechanism. Either way **the 2.2 GiB nuecc tail doc 118 introduced is untouched by this
round.**

**Attribution — the same event, the same config, two binaries**
(`docs/119_figs/119_prof_r3_before_after.txt`, evt 2925, one `PROFILE:` line in each run):

| function | before | after | |
|---|---:|---:|---|
| **total samples** | 9440 | 8526 | −9.7 % |
| `form_point_association` | 1793 | 870 | −923 |
| ⤷ `convert_3Dpoint_time_ch` | 802 | **137** | −665 |
| ⤷ ⤷ `drift2time` | 405 | **2** | −403 |
| ⤷ ⤷ `point2wind` | 258 | **4** | −254 |
| ⤷ `find_neighbors_nlevel` | 446 | 250 | −196 |
| `do_cos` / `__sincos_fma` / `feholdsetround` | 33 / 33 / 104 | 0 / 3 / 19 | −148 |
| `CreateSteinerGraph::visit` — **not touched** | 2083 | 2076 | **−7** |

The whole −914-sample total delta is accounted for by the −923 in `form_point_association`, and
`CreateSteinerGraph` — the largest stage this round does not touch — moves by 0.3 %. That is the
internal control: the saving is where the change is, not a global shift.

### 9.4 Gates

**G1 — output equivalence, 62 events** (`119_gate_r3.txt`): `r3` vs `flip`, 284 files compared,
**0 differences outside the declared allowance**. The allowance is deliberately **narrow, not
blanket**: `toolkit_git` and `wcp_git` are forgiven (the two arms are necessarily run at different
working-tree states), while `op_config_sha256` and `trackfitting_config` stay under comparison —
and both are byte-identical, `6089ecad115d…` on both sides. What the arms actually differ by, with
provenance forbidden entirely, is only:

```
  x 24  calib-pr-evt*.json : 1 of up to 523 940 keys -- vertex_scoreboard.dual_chain.off_ms
  x 24  tracking-pr.root   : Trun['toolkit_git', 'wcp_git']
```

i.e. the dual chain's own stopwatch and the two git strings. `T_rec_charge`, every tagger tree,
`T_proj_data`, `mabc-pr.zip`, the pctree, `nusel` and every other calib key are identical.

**G2 — the null pair, on the new binary** (`119_gate_r3null.txt`): `r3b` vs `r3` returns
**10 / 24 / 1**, and `r3` vs `flip` returns **28 / 48 / 21**. The difference is exactly
**18 / 24 / 20** — one `tracking-pr.root` per event in each sample. The gate's output is the floor
plus one git string per event and nothing else.

**G3 — sensitivity** (`119_gate_r3_sens.txt`): the *same* `r3` allowance applied to a pair that
genuinely differs (`ctl`, which is pre-`proj_pad`) **FAILS**, reporting `T_proj_data` and
20 979–121 764 `proj` charge keys per event. The PASS above is the gate working, not the gate being
blind — the round-2 lesson applied to a new allowance code path.

**G4 — the other two detectors, gated not argued** (`119_gate_r3_xdet.txt`). `Facade_Util.cxx`,
`Facade_Grouping.cxx` and `Graphs.cxx` are shared across all three detectors, so a PASS on SBND
alone would not have discharged CLAUDE.md §4's "every affected detector". Round 0's PDHD and PDVD
arms re-run on the round-3 pin and gated with `d30_hash_gate.py` (five products per event,
including `T_proj_data` through the jagged-safe `d30_hash_proj.py` — `hash_root_trees.py` alone
cannot read that tree, doc 30's round-3 correction):

```
GATE d119hnu vs d119hr3 (pdhd/work): PASS  61  FAIL 0  MISSING 0  of  61 events, 5 products each
GATE d119vnu vs d119vr3 (pdvd/work): PASS 120  FAIL 0  MISSING 0  of 120 events, 5 products each
```

Independently of the hashes, the census agrees: `nclus`, `nstm`, `nfit`, `nseg`, `sum_npts` and
`max_npts` are equal **on every one of the 181 events**
(`119_pdhd_r3_census.tsv`, `119_pdvd_r3_census.tsv`). This also covers the one non-fit caller of
`point2wind`, `DynamicPointCloud.cxx:1010`.

**G5 — unit tests**: `./build/clus/wcdoctest-clus` **446 cases, 716 750 assertions, 0 failed.** The
new `clus/test/doctest_projection_constant_hoist.cxx` writes out the *legacy* bodies literally and
asserts the hoisted forms reproduce them **bit for bit** — 140 000 random points over seven
SBND/PDHD/PDVD plane geometries, 6 006 cases driven onto the `.5` rounding boundary on purpose, and
120 000 `drift2time` cases. The *delegating* angle-taking forms get the same full sweep rather than a spot check, because `DynamicPointCloud` calls them and their results reach production outside the fit. This is what retires the one real risk in L1: `cos(a)*z − sin(a)*y` can
be contracted to an FMA in one inlining context and not another, and a value landing exactly on
`.5` would then round to a different wire. It does not, and now a future refactor that breaks it
fails here instead of failing an A/B three hours into an arm.

**G6 — the tripwire** (`119_gate_r3_cfg.txt`): `prod_cfg_gate.py --ref ref/prod-2026-09-21` is
**PASS 25/25**. Round 3 changes no jsonnet and no runtime JSON, so **no new reference generation**;
`prod-2026-09-21` remains current.

**M1, the stale-library trap, is the failure this round was most exposed to** — the first in this
campaign whose arms differ by the binary. `stageB_lever.sh` now refuses `r3`/`r3b` on the
pre-round-3 pin ("this would compare the old binary against itself and PASS for nothing") *and*
refuses `ctl`/`ctl2`/`tcm`/`pad`/`flip` on the new one. Both directions were fired once on purpose
before the arms were launched.

### 9.5 What round 3 did not take, and why

1. **The plane-independent drift time.** `convert_3Dpoint_time_ch` is called three times per point
   and the time index is identical for all three planes — `NeutrinoStructureExaminer.cxx:1513`
   already documents and exploits this. Two thirds of the `drift2time` calls are therefore dead.
   **Subsumed by L1**: with `xsign`/`xorig` memoised, `drift2time` is a subtract, a divide and a
   subtract (2 samples after, from 405), so removing two thirds of it saves nothing worth a diff.
2. **Deriving the rounded wire from the continuous one** (`cur_wire = std::round(cen)` when
   `assoc_cont_center` is on, instead of calling `point2wind` again). Dropped deliberately. After
   L1 memoises the trig it is worth ~nothing, and it was the only lever in the round whose
   correctness rested on two separately-compiled expressions agreeing to the last ulp. Removing it
   removed the round's only FP-identity risk.
3. **The `std::vector<char> visited(num_vertices)` allocated per `find_neighbors_nlevel` call** —
   O(N) in the whole cluster point cloud for a BFS that touches a neighbourhood. A reusable
   stamp buffer would fix it, but `GraphAlgorithms` methods are `const` and the object is shared,
   so the buffer would have to be `mutable` — a thread-safety hazard in a multi-threaded node for a
   minority of the 461 samples. Left, with the reason.
4. **Hinted `std::set` insertion in the wire loops.** The `j` loops insert ascending runs into
   `associated_2d_points`, so `insert(hint, …)` would be O(1) amortised. The runs are short and the
   remaining `std::set::insert` is 563 samples across many call sites; it needs its own measurement
   before it is worth a diff.
5. **`time2drift`, which has the identical defect.** Left alone on purpose: `Aux::time2drift` is a
   different package with imaging-stage consumers, and the `clus` one does not appear in this
   profile at all. Round-4 material, named here so it is not lost.
6. **`libWireCellMatch.so` and `libWireCellRoot.so` also changed in the rebuild** — checked, not
   waved through. `nm -C` shows neither references `point2wind`, `drift2time` or
   `find_neighbors_nlevel`, defined or undefined, and `fastgeom_t` is only ever held behind a
   `std::unique_ptr` so growing it does not move `Grouping`'s layout. Both are pure recompile
   artifacts of including the edited headers; the blast radius is `clus`. All three libraries were
   pinned together anyway, so the gated arms carry a consistent set.
7. **A pre-existing order dependence, named not fixed** (CLAUDE.md §5 tie-breaker). The blob set
   `form_point_association` builds is an `unordered_set<const Blob*>`, and the `blobs_by_face`
   vectors — hence `face_blobs.front()->wpid()` — take their order from iterating it. That is a
   pointer-keyed iteration order reaching production output today. L2 preserves it exactly (the
   insertion sequence is unchanged), and this round does not touch it.

### 9.6 What is left

**Round 3 does not reduce the dual chain, it makes everything cheaper.** `run_dual_chain_off_pass`
went 2955 → 2508 samples on evt 2925 purely because it runs the same projection code — it is still
29 % of the job. §8's ranking is unchanged, and §8's recommendation still stands as the largest
remaining target, still needing doc 107 grading rather than a byte gate.

The two cheap items in §8 (the runner tripwire hole; retention) are untouched by this round.
Round 3 adds two SBND arms of 62 events each (`d119r3`, `d119r3b`) plus a 61-event PDHD arm
(`pdhd/work/*_d119hr3`) and a 120-event PDVD arm (`pdvd/work/*_d119vr3`) to the retention list.
Every number from all of them is in `docs/119_figs/`, so none is a scan record and all are
re-derivable.

**One defect this round deliberately left standing.** `time2drift` has the *identical* shape as
`drift2time` — it re-derives `xsign`/`xorig` from the `IAnodeFace` on every call. It was not
touched because `Aux::time2drift` lives in a different package with imaging-stage consumers, and
the `clus` one does not appear anywhere in this profile. That makes it a round of its own with its
own manifests, not something to fold into a PR-chain round. Named here so it is not lost.

## 10. Round 4 — memory, attributed for the first time

**No code changes.** This round measures. It is the first round of the campaign to look at memory
at all: rounds 0–3 were CPU, and doc 118 §B.1 item 4 planned jemalloc sampling that was never run.

Repro:

```bash
# the tail, at current production and at the pre-round-1 allocator
ALLOC=prod  scripts/d119/stageB_tail.sh          # -> work-r3nue-d119tail
ALLOC=glibc scripts/d119/stageB_tail.sh          # -> work-r3nue-d119tailg
# per-stage memory, from instruments already on disk -- nothing re-run
scripts/d119/r4_stage_mem.py     > docs/119_figs/119_r4_stage_mem.txt
# live-heap attribution
PRDIR=$PWD/work-r3nue-d119tail/f060/pr_evt11239 OUTDIR=~/tmp/d119r4/sbnd_11239 \
    LG_INTERVAL=30 scripts/perf/profile_pr119r4.sh
LD_LIBRARY_PATH=~/tmp/d119r3-libpin LG_INTERVAL=30 ../../pdvd/profile_pr_heap.sh 39252 8
scripts/d119/heap_rank.py ~/tmp/d119r4/sbnd_11239 11239
```

### 10.1 The correction this round exists to make

Rounds 1–3 all gated on the 62-event d118 manifest, whose 24 `nuecc` events top out at 1.43 GiB.
It is tempting — and I did it in the round-3 hand-off — to read "0 events above 2 GiB" there as
the doc-116 memory tail having gone away. It is not evidence of anything. Doc 116 §14 measured
that tail over **2001** events and found **5** above 2 GiB: a 0.25 % tail. Twenty-four draws
cannot contain it.

So this round names those five events from the doc-116 arm's own `.time.meta` records and re-runs
them, plus the p99 shoulder and the p50 as a composition control.

### 10.2 The tail today: still there, and neither round 3 nor `proj_pad` touched it

`docs/119_figs/119_r4_tail.txt`. Peak RSS, `getrusage(RUSAGE_CHILDREN).ru_maxrss`:

| sub/event | doc 116 (glibc) | now, glibc | now, **production (tcmalloc)** |
|---|---:|---:|---:|
| f060/11239 | 2.208 | 2.204 | **2.084** |
| f049/12202 | 2.203 | 2.197 | **2.063** |
| f188/11811 | 2.050 | 2.048 | **2.162** |
| f100/5265 | 2.021 | 2.019 | **1.915** |
| f196/6248 | 2.006 | 2.001 | **1.910** |
| f075/9393 (p99) | 1.602 | 1.734 | 1.698 |
| f054/7582 (p50) | 1.165 | 1.160 | 1.195 |

Three readings, and one thing reported rather than explained:

1. **The tail is still there.** Three of the five still exceed 2.0 GiB under today's production.
2. **Round 3 and `proj_pad` are neutral on it** — glibc-to-glibc, four of the five moved by less
   than 0.005 GiB. `proj_pad` buying SBND nothing in memory is consistent with §6: SBND keeps
   87.9 % of its proj cells where PDHD kept 2.4 %.
3. **tcmalloc is not a memory lever here** — −6.1 % to +5.6 %, both signs. Round 1 flipped it for
   CPU; it should not be cited for memory in either direction.
4. `f075/9393` moved 1.602 → 1.734 GiB (+8 %) glibc-to-glibc while its four neighbours moved by
   under 0.3 %. One event, unexplained, recorded (§5 rule 7).

**A trap this round fell into and had to back out of.** The first version of `stageB_tail.sh` set
`SBND_PR_TCMALLOC=0`, copied from the `flip` lever where glibc is deliberate — and would have
reported a glibc arm as "current production". Round 1 made tcmalloc the SBND PR default, so
production has been tcmalloc since. Both arms are kept, because the allocator is the one term
that differs between doc 116's tail and today's. And the proof that the prod arm really preloads
it is a `/proc/<pid>/maps` read, not the arm-to-arm delta: `run_pr_chain_batch.sh:1978` falls
back to glibc *silently* if `SBND_TCMALLOC_LIB` is missing, so a null delta would have two
readings.

### 10.3 The mean and the tail name different stages

`docs/119_figs/119_r4_stage_mem.txt`, from the in-job `MEM:` ladder — nothing re-run.

| | SBND gate arm (24 evt) | SBND tail arm (7 evt) |
|---|---:|---:|
| `CreateSteinerGraph:pr` | mean **+0.155**, max +0.604 | mean **+1.010**, max +1.336 |
| `TaggerCheckNeutrino:pr` | mean **+0.394**, max +0.461 | mean +0.291, max +0.365 |

On the gate manifest the tagger is the biggest mean and Steiner is a quarter of it. On the tail
events Steiner is 6–9× its own gate-arm mean and the tagger is *flat*. **A memory target ranked
on means is the wrong target for every event that breaks a cap.** The tagger's increment sits in
a narrow band everywhere; Steiner's max/mean is 3.7× on PDHD and 5.6× on PDVD. The stage that
varies is the stage that matters, and it is the same stage on all three detectors.

### 10.4 Where the live heap actually is

`docs/119_figs/119_r4_heap_sbnd.txt`, `119_r4_heap_pdvd.txt`, `119_r4_heap_sbnd_p50.txt`.

Each row's ladder peak is read from *that jemalloc run's own* job log, not from the production
`.time.meta` — a different allocator, so it would not be a ratio of like to like.

| | peak live | ladder peak res | live/RSS | released by exit |
|---|---:|---:|---:|---:|
| SBND f060/evt11239 (tail) | 0.910 GiB | 1.343 GiB | 67.8 % | 58.6 % |
| SBND f054/evt7582 (p50) | 0.525 | 1.065 | 49.3 % | 40.9 % |
| PDVD 039252_8 (max Steiner) | 2.686 | 2.966 | **90.6 %** | **95.4 %** |

> **RETRACTED by §11.1 — do not quote the paragraph below.** Its two comparisons mix instruments:
> the jemalloc figures are `MEM:`-ladder peaks taken *with the heap sampler active*, the glibc and
> tcmalloc figures are `getrusage` high-water from unprofiled chain runs. Measured like for like
> (§11.2), the p50 comparison **reverses sign** — jemalloc 1.197 equals production tcmalloc 1.197
> and is *above* glibc's 1.162 — and the tail gap is −20.3 %, not the ~1.6× implied here. The
> conclusion the paragraph draws, that the effect is tail-specific and absent at the median, is the
> one part that survives, and §11.2 states it on the right numbers.

On SBND the allocator overhead **grows into the tail**: at the p50 jemalloc peaks at 1.065 GiB
against glibc's 1.160 and production tcmalloc's 1.195, a few per cent; on the tail event it is
1.343 against 2.204 and 2.084, a factor of ~1.6. Whatever those extra 0.7–0.9 GiB are, they are
not live data and they are not present at the median. (One event; not a lever, and nothing here
gates a byte-identical output.)

PDVD's 90 % live / 95.4 % released says this is a **working set, not retained state**: a lever has
to shrink what is *simultaneously* live, not free something sooner.

**What is stable across both detectors and every phase:**

1. `CreateSteinerGraph::visit` is the stage — 78.0 % and 69.9 % of peak live heap at two different
   SBND peaks, 71.6 % on PDVD.
2. `ImproveCluster_2::mutate` — the **retile sampler** (`CreateSteinerGraph.cxx:322`) — is 14–21 %
   of peak live in *every* phase sampled, including ones where the stage itself is small. The
   **fraction** is what is stable; the absolute is not. On SBND evt11239 it is 158.5 MB at i6,
   98.9 at i16, 61.7 at i32 — a 2.5× range while the share holds. It **scales with the cluster**,
   so it is not a fixed block that could be deleted.

(2) is the round's result. Round 0 named the retile sampler as the #1 CPU target on all three
detectors (53–60 % of Steiner's time). It is now also a stable sixth-to-fifth of peak live heap on
two detectors. **The CPU target and the memory target are the same object.**

**What is *not* a finding: the within-stage split.** The peak dumps are spikes above an
oscillating plateau — one oscillation per cluster's retile→graph→Steiner cycle — so the dump
position selects a phase, and the sub-attribution follows it. Within one SBND event:

| dump | `CreateSteinerGraph::visit` | `connect_graph_closely_pid` | `create_steiner_tree` |
|---|---:|---:|---:|
| SBND i6 (932 MB) | 78.0 % | **59.2 %** | absent |
| SBND i16 (687 MB) | 69.9 % | 2.6 % | **24.2 %** |
| PDVD i32 (2750 MB) | 71.6 % | 2.1 % | **29.4 %** |

The split swaps between two dumps of the same event on the same detector. These are two phases of
one stage, not two targets, and this instrument cannot separate them.

**Checked and clean:** retiled child clusters do *not* accumulate. Every exit path of
`process_cluster_steiner` calls `grouping.destroy_child` (`CreateSteinerGraph.cxx:330, 378, 391,
422`), which is what the oscillating trajectory shows. The peak is one cluster's working set, not
a leak. Recorded so round 5 does not chase it.

**An incidental floor:** `UbooneNueBDTScorer::ensure_readers` → `TMVA::MethodBase::ReadStateFromFile`
holds **145 MB live at exit** — 45.7 % of the median event's final live heap — allocated once under
`call_once` and never released. Not a tail contributor (it is the same 145 MB on every event), but
it is the largest single thing alive at exit and a floor under every SBND PR job.

### 10.5 Two readings this round retracted, and the rule they give

Both came from `google-pprof` frames marked `(inline)`. Neither reached a conclusion — each was
refuted by primary evidence within minutes — and both are recorded so the next round does not
re-derive them.

**(a) "the doc-118 base-weight flip is the memory tail."** `boost::vec_adj_list_impl::copy_impl`
at 1254 MB, 45.6 % of PDVD's peak, sits under `add_edge` in the inline chain, and
`Steiner::reweight_base_graph` (`SteinerBaseWeight.h:84`) really does copy a whole graph —
`Graph out = base;` — only when `base_weight_blank_alpha > 0`, which is exactly what doc 118
flipped on. It fits doc 116 §16's finding that the 2.2 GiB events need *both* knobs.
**Refuted by the job's own log**: 409 pricing calls in this event, largest 62 vertices / 450
edges. All of them together are a few MB. The copy is real code; it is not this memory.

**(b) "the Steiner graph stores a red-black-tree node per edge."** `_Rb_tree::_M_create_node` at
809 MB and `list::push_back` at 632 MB both appear under `boost::add_edge`.
**Refuted by the typedef**: `Graphs.h:23` is `boost::adjacency_list<vecS, vecS, undirectedS, …>`.
Both vertex and edge storage are vectors; there is no per-edge tree or list node in this graph at
all.

**The rule.** In this profile, trust **named non-inline frames** — `CreateSteinerGraph::visit`,
`ImproveCluster_2::mutate`, `connect_graph_closely_pid`. Treat every `(inline)` frame and every
bare `std::`/`boost::` container frame as **unattributed**: the inline chain is reconstructed by
the symbolizer, not observed, and it was wrong twice in one profile. Going deeper needs a
different instrument — a frame-pointer build — not a closer reading of this one.

### 10.6 What a round 5 would have to be

Nothing here is a lever yet, and this round deliberately stops short of proposing one. What it
fixes is the aim:

- The target is **`CreateSteinerGraph`'s per-cluster working set on the largest cluster**, not a
  container and not a leak. Shrinking the mean buys nothing operationally; the tail is the value.
- The retile sampler is the one object that is both the #1 CPU cost and a stable *fraction* of
  peak live heap on two detectors. A lever there pays twice — which also means it needs both
  gates. But budget it as a proportional saving, not as a removable 17 %: its absolute footprint
  tracks the cluster (158.5 → 61.7 MB across phases of one event), so the win is "the largest
  cluster's retile gets smaller", not "a fifth of the peak disappears".
- **Doc 30 §12.4's target is not confirmed on these arms and should not be carried forward
  unexamined.** Its 20.5 %-of-live-heap figure for the per-cell `std::set<Cluster*>` is a `-stm`
  measurement; on these `-nu` arms the `stm` stage is 4th on PDHD (+0.184 mean) and 6th on PDVD
  (+0.069). Whatever is true in `-stm` mode, it is not where `-nu` memory goes.
- Deciding *what inside the retile is large* needs an instrument this round showed is not
  available: a frame-pointer build, so the inline chain is observed rather than reconstructed.
  That is the first step of a round 5, and it is a build question before it is a physics one.

### 10.7 What is left

§8's CPU ranking is unchanged by this round. The two cheap items there (the runner tripwire hole;
retention) are still untouched, and round 4 adds `work-r3nue-d119tail{,g}` (7 events each) plus
the throwaway `pdvd/work/039252_8_heappr_039252_8` tag to the retention list. Every number is in
`docs/119_figs/`, so none of it is a scan record.

The `time2drift` item named at the end of §9.6 is also still open, and is still a round of its own.

---

## 11. Round 5 — the allocator, and the tripwire hole round 1 opened

**Status: MEASURED and GATED, NOT flipped.** jemalloc 5.3.0 is **−18.8 % mean and −19.1 % max peak
RSS against production tcmalloc on the seven tail events, at −4.0 % in-job CPU** — it removes every
one of the three events above 2 GiB and is byte-identical on the 62-event manifest. Changing the
allocator default is changing an existing production default (§5 rule 1), so this round measures,
gates and recommends; the flip is the owner's, exactly as round 1's was.

Two things ship regardless, because neither is an operating-point change: the **tripwire hole §6.4
named is closed** (§11.5), and **a second hole found inside the tripwire itself** is closed with it.

### 11.1 Round 5 begins by retracting round 5's own premise

The hand-off out of round 4 recommended this round on a number that was **wrong by about 40 %**, and
wrong for a reason round 4 had already been burned by once.

§10.4 reported jemalloc peaking at **1.343 GiB** on `f060/evt11239` where production tcmalloc peaked
at **2.084** — a gap of 0.74 GiB. Those two numbers are **different instruments on different
invocations**:

| | 1.343 GiB | 2.084 GiB |
|---|---|---|
| instrument | in-job `MEM:` ladder — VmRSS sampled at stage boundaries | `.time.meta maxrss_kb` — `getrusage(RUSAGE_CHILDREN)` high-water |
| invocation | `scripts/perf/profile_pr119r4.sh`, a direct `wire-cell -c` | the full `run_pr_chain_batch.sh` chain |
| allocator state | **`MALLOC_CONF=prof:true,…,lg_prof_sample:19`** — the heap sampler active | production, unperturbed |

Both corrections matter and they push in **opposite** directions:

- **Instrument.** On these seven events getrusage reads **+0.08 to +0.18 GiB above** the ladder, so
  the ladder-vs-getrusage comparison inflates any advantage held by whichever arm was read by the
  ladder.
- **Perturbation.** The sampler is not neutral. Clean jemalloc's ladder peak on `f060/evt11239` is
  **1.589 GiB**, not 1.343: heap sampling **understated** jemalloc by 0.246 GiB (15.5 %) there,
  while at the p50 it *overstated* it by 0.022. Round 4's own §10 header says the sampler perturbs
  allocation; it did, in both directions.

Read like for like, the real number is **2.093 → 1.669 GiB, −0.424 GiB, −20.3 %** on that event.
Still large, still the largest single memory effect this campaign has found, and **not** 0.74 GiB.

**One sentence of §10.4 is retracted outright.** It said that at the p50 "jemalloc peaks at 1.065
GiB against glibc's 1.160 and production tcmalloc's 1.195, a few per cent". Like-for-like at the
p50 the three are **tcmalloc 1.197, glibc 1.162, jemalloc 1.197** — jemalloc is *exactly* production
and *above* glibc. The direction of that comparison was an artefact of the instrument mix. What it
was used to argue — that the effect is tail-specific and absent at the median — is **strengthened**
by the correction, not weakened: at the median jemalloc buys precisely nothing.

This is the third time in two rounds that an instrument, not the code, produced the finding
(§10.5's two `(inline)` retractions, round 4's own corrected p50 row, this). The rule that comes out
of it is in §11.7.

### 11.2 Four allocators, one invocation, one instrument

`scripts/d119/stageB_alloc.sh` runs the same seven events through the **same**
`run_pr_chain_batch.sh` invocation four times, back to back at one fan-out on an idle box, with
`MALLOC_CONF` and `TCMALLOC_RELEASE_RATE` explicitly cleared so the arms differ in the preload and
in nothing else. All 28 runs `rc=0`. The allocator of every arm is proved twice: from the runner's
new log line in each event's own `stdout.log`, and from `/proc/<pid>/maps` of a live job
(`docs/119_figs/119_r5_maps.txt`) — **not** inferred from the arm's name, because the runner falls
back to glibc silently when the requested library is absent (§11.5).

Peak RSS, `getrusage` CHILDREN high-water, GiB — the number an operational memory cap is set
against:

| event | tcmalloc (production) | glibc | **jemalloc** | tcmalloc + release_rate 10 |
|---|---:|---:|---:|---:|
| f188/11811 | 2.156 | 2.049 | **1.745** | 2.073 |
| f060/11239 | 2.093 | 2.206 | **1.669** | 1.979 |
| f049/12202 | 2.059 | 2.201 | **1.670** | 1.921 |
| f100/5265 | 1.915 | 2.017 | **1.499** | 1.803 |
| f196/6248 | 1.910 | 1.981 | **1.488** | 1.862 |
| f075/9393 (p99) | 1.705 | 1.599 | **1.316** | 1.630 |
| f054/7582 (p50) | 1.197 | 1.162 | **1.197** | 1.197 |
| **mean** | 1.862 | 1.888 (+1.4 %) | **1.512 (−18.8 %)** | 1.781 (−4.4 %) |
| **max** | 2.156 | 2.206 (+2.3 %) | **1.745 (−19.1 %)** | 2.073 (−3.9 %) |
| **events > 2.0 GiB** | **3 of 7** | 4 of 7 | **0 of 7** | 1 of 7 |

In-job CPU on the same arms (TICK total, the instrument every cost claim in this doc is made on):

| | mean | vs production | max | vs production |
|---|---:|---:|---:|---:|
| tcmalloc (production) | 81.9 s | — | 164.5 s | — |
| glibc | 97.3 s | **+18.8 %** | 191.7 s | +16.6 % |
| **jemalloc** | 78.7 s | **−4.0 %** | 159.1 s | −3.3 % |
| tcmalloc + release_rate 10 | 82.3 s | +0.4 % | 168.1 s | +2.2 % |

**Read the CPU column conservatively.** Round 3's null pair put the run-to-run floor on this machine
at +0.2 / −1.6 / −2.6 % (§9.3). −4.0 % is only just outside it, and there is one run per event. The
defensible statement is **"jemalloc costs no CPU"**, not "jemalloc is 4 % faster". The glibc column
is a different matter — +18.8 % is far outside any floor, and it **independently reproduces round 1
on a different event set**: round 1 measured tcmalloc at −16.5 % on one median event, and the
inverse of +18.8 % is −15.8 % here on the tail.

**The 62-event gate manifest says the same thing from the other side.** The same two allocators on
the same binary over the d118 manifest — 24 nuecc + 18 cv + 20 beam-off, which §10.1 established
contains **no tail event** (max 1.449 GiB):

| sample | job size (TICK mean) | peak RSS, mean | peak RSS, max | TICK |
|---|---:|---:|---:|---:|
| nuecc (24) | 11.7 s | **+0.1 %** | +2.1 % | −2.4 % |
| cv (18) | 3.0 s | −1.4 % | +1.0 % | **+6.6 %** |
| beam-off (20) | 1.0 s | +3.0 % | −2.9 % | **+4.5 %** |
| *tail (7), for contrast* | 81.9 s | **−18.8 %** | **−19.1 %** | −4.0 % |

**Memory: jemalloc buys nothing here.** Not "a little" — nothing, inside the noise in both
directions. Taken with the p50 event, that is the clearest form of the round's central claim: the
win exists **only** on the events that threaten a cap, which is also the only place it is wanted.
It also means a flip cannot be justified by an average — the average is flat.

**CPU: report the caveat.** On the two *short* samples jemalloc is **worse in relative terms**,
+6.6 % on cv and +4.5 % on beam-off. In absolute core-seconds that is +0.20 s and +0.04 s per
event against −0.28 s on nuecc and −3.2 s on the tail, so jemalloc wins where the time actually is
— but the honest shape is "jemalloc's CPU benefit grows with job size and is slightly negative on
jobs of a few seconds", not "jemalloc is faster".

**Two free by-products of running the control arm.**

1. **The memory instrument's repeatability.** `work-r3nue-d119ttcm` is byte-identically the
   configuration of round 4's `work-r3nue-d119tail`, run again three weeks of edits later. Worst
   per-event disagreement in `maxrss`: **0.009 GiB, ≤ 0.5 %.** So a 19 % memory delta is not noise,
   and neither is a 4 % one — the earlier caution in this doc about RSS being unrepeatable applies
   to *cross-allocator* comparison, not to this instrument.
2. **Round 4's tail result is confirmed independently**: three events above 2.0 GiB at current
   production, same three, same values to 0.01 GiB.

### 11.3 The mechanism control, and why it was not optional

jemalloc 5.x returns dirty pages to the OS on a decay timer (`opt.dirty_decay_ms`, 10 s by default);
tcmalloc holds freed spans in its page heap and releases at `TCMALLOC_RELEASE_RATE`, default 1. If
the tail gap were simply *page-return policy*, the finding would not be "jemalloc" at all — it would
be "one environment variable", available on production's existing allocator with no flip, no byte
gate and no owner decision. An arm that cannot tell those two apart is not worth running, so the
fourth arm is production tcmalloc at `TCMALLOC_RELEASE_RATE=10`.

**It is part of the mechanism and not most of it.** Aggressive release recovers **−4.4 % of the
−18.8 %** — about a quarter — and costs nothing in CPU (+0.4 %, inside the floor). On the worst
event jemalloc still sits **0.310 GiB below** the aggressively-releasing tcmalloc (1.669 vs 1.979).
So roughly three quarters of the gap is jemalloc's allocation *layout* — its size classes and
per-arena extent reuse against a heap that, per §10.4, is a genuinely live working set — and not
merely when pages go back.

`TCMALLOC_RELEASE_RATE=10` is therefore a **real, smaller, much cheaper option in its own right**:
one env var in the runner, no allocator change, no byte gate needed beyond what round 1 already has,
worth about 0.1 GiB off the worst event and one of the three events above 2 GiB. It is recorded here
as an alternative the owner can take instead of the flip, not as a recommendation on its own.

### 11.4 The tripwire hole §6.4 named, closed — and a second one found inside the tripwire

This part ships whatever the owner decides about the allocator, because nothing in it changes a
production operating point. §6.4 left the hole open with an explicit condition: close it *minimally*
or not at all, because the PR runners are large, churn constantly with A/B scaffolding, and a noisy
tripwire is worse than none.

**The hole.** Round 1 made `SBND_PR_TCMALLOC` default on, which preloads `libtcmalloc_minimal` into
every PR job for −10.9 to −16.7 % in-job CPU (§5; §11.2 reproduces it at −15.8 % on the tail). That
is a production operating-point change living in a **runner**, and no runner was among the 25
consumers, so `prod_cfg_gate.py` reported PASS 25/25 while it landed. Worse than the edit is the
**silent fallback**:

```bash
if [ "${SBND_PR_TCMALLOC:-$SBND_PR_TCMALLOC_DEFAULT}" = 1 ] && [ -e "$SBND_TCMALLOC_LIB" ]; then
```

If that library ever stops existing — a package upgrade renaming a soname is enough — every PR job
drops to glibc. Before this round there was **no log line, no compiled-config change and no gate
failure**: round 4 had to read `/proc/<pid>/maps` to establish which allocator a running job
actually had. A ~16 % CPU regression could land on production with nothing anywhere recording it.

**What was added.** Three small things:

1. `run_pr_chain_batch.sh` now **logs the allocator it chose**, on both the per-event and the group
   path, and **warns loudly** when the requested library is missing instead of falling back in
   silence. The line lands in each event's own `stdout.log`, next to the products it produced —
   `lever_gate.py`'s SKIP already excludes `stdout.log`, so it is a record, not a product. Log-only:
   no product, no compiled config, no TLA moves. §11.2's four arms are proved from it.
2. `compile_consumers.sh` emits `runner_alloc.txt`: the allocator decision of the three SBND runners
   that set `LD_PRELOAD` (`run_pr_chain_batch.sh`, `run_pr_evt.sh`, `run_clus_evt.sh`), extracted by
   pattern, with comments and leading whitespace stripped and **no line numbers**. Moving the block,
   re-indenting it or rewriting its comments does not fire; changing which library is preloaded, or
   deleting the new warning, does. That is §6.4's minimal form, and it is what keeps the tripwire
   quiet enough to stay trusted. The one line in it that looks like a false positive in waiting —
   `PYLIB=…/libpython3.11.so.1.0` — is deliberately included: a python bump moving that soname is a
   production change of exactly this family, because doc 118 §8 records that dropping `$PYLIB`
   silently disables the SCN import and the job falls back to the **geometric** vertex. A quiet
   upgrade there would be worse than a quiet allocator swap.
3. A new reference generation, **`ref/prod-2026-09-21b`**, 26 artifacts. `prod-2026-09-21` is kept.
   **All 25 of its hashes are bit-identical here** — the manifests differ by exactly one added line
   (`119_r5_ref_diff.txt`). Round 5 changed what is watched, not what runs.

**The second hole, found while closing the first.** `prod_cfg_gate.py` only ever looked up the names
already in the reference manifest, and `--refresh` rewrote it over `sorted(want)`. So **adding a
consumer to `compile_consumers.sh` did nothing at all**: the new artifact was compiled, ignored, and
never adopted, while the gate went on reporting PASS on the old set. Doc 118 added three runtime fit
JSONs and the LArSoft 1-step chain and got away with it only because it built its reference
generation by hand — had it used `--refresh`, those four would silently not be gated today.

Fixed in the same change: the gate computes what the compile **produced**, prints
`NEW : <name>` for anything the reference does not list, **fails** on it, and `--refresh` writes the
manifest from the produced set while reporting what it added and dropped. `runner_alloc.txt` is
also kept in full in the reference dir, so a drift in it is *named* by a line diff and not merely
detected — the same reason `prod_prjob.json` is kept.

This one is worth stating plainly: **the instrument that exists to catch unseen production changes
had a blind spot of exactly that shape.** It is the same failure mode as the runner hole, one level
up.

Records: `119_r5_gate_pre.txt` (PASS 25/25 at unmodified HEAD, before the round's first edit),
`119_r5_gate_new.txt` (`NEW : runner_alloc.txt`, rc=1), `119_r5_gate_post.txt` (PASS 26/26),
`119_r5_ref_diff.txt`, and `ref/prod-2026-09-21b/README.md`.

### 11.5 The byte gate — 62 events, and a null pair that makes the PASS mean something

An allocator change is **not** structurally byte-identical in this codebase, and saying so is the
reason this gate exists rather than being waved through. §9.5 item 7 names a live
`unordered_set<const Blob*>` whose *iteration order* reaches production output; allocator changes
are exactly what perturbs pointer values. Round 1's PASS for tcmalloc is empirical evidence about
tcmalloc, not a proof that covers a third allocator.

`scripts/d119/lever_gate.py --vs r3 jem` compares the jemalloc arm against `work-r3*-d119r3` — the
**same binary, same compiled config, same fit JSON**, differing in the malloc implementation and
nothing else. Allowance: **nothing** beyond the dual chain's own stopwatch and the two git-sha
provenance strings (the arms were produced at different working-tree states; `op_config_sha256` and
`trackfitting_config` stay under comparison, so the gate would still notice if the configuration
had moved).

| | files compared | differ, within allowance | differ, NOT allowed |
|---|---:|---:|---:|
| cv (18 evt) | 82 | 28 | **0** |
| nuecc (24 evt) | 120 | 48 | **0** |
| beam-off (20 evt) | 82 | 21 | **0** |

**PASS**, `119_r5_gate_jem.txt`. And the null pair makes it specific rather than merely green:
`lever_gate.py --vs r3 r3b` — one configuration run twice on the same binary — reports **10 / 24 /
1** differences at provenance-off allowance (`119_r5_gate_null.txt`). The jemalloc columns are
exactly those counts **plus one provenance record per event** (18 + 10 = 28, 24 + 24 = 48,
20 + 1 = 21). So jemalloc moved **nothing that two runs of one configuration do not also move**.
That is the statement a bare PASS cannot make.

Scope of the claim, stated so it is not over-read: 62 events, one detector, one binary. It is the
same bar round 1 cleared for tcmalloc and round 3 cleared for the projection hoist — not a proof of
pointer-order independence, which no gate here provides.

### 11.6 The dual-chain census §8 has been deferring — read, and it changes the recommendation

§8 has named SBND's dual second pass the largest remaining CPU target since round 0 (31.1 % of the
post-flip nuecc job by direct profile, §3.1) and has three times deferred the census that would size
it, on the grounds that it costs nothing to read. It does cost nothing:
`scripts/d117/dual_chain_census.py` reads `vertex_scoreboard.dual_chain` out of calib dumps already
on disk. `119_r5_dual_census.txt`, on the **2001-event** post-flip `nuecc` arm (`t2`, the trajectory
that is production today) and the 2017-event `cv` arm:

| | nuecc (1 869 scoreboards) | cv (952) | beam-off (85) |
|---|---:|---:|---:|
| second pass ran | 1 865 (99.8 %) | 931 (97.8 %) | 58 (68.2 %) |
| chains agreed | 1 762 (94.5 %) | 901 (97.1 %) | 48 |
| transfer accepted | 611 (32.8 %) | 452 | 47 |
| **and it MOVED the pick** | **572 (30.7 %)** | 349 | 38 |
| transfer distance, median / p90 | 0.413 / 1.80 cm | 0.272 / 1.27 | 0.000 / 2.25 |
| second-pass cost, median / p90 | 2.9 / 13.4 s | 1.6 / 2.9 s | 1.5 / 4.3 s |
| **total second-pass cost** | **3.16 core-h / 2 001 evt** | 0.53 core-h / 2 017 | 0.04 core-h / 1 000 |

The 62 events of the gate manifest agree on the rates (second pass on **100 %** of nuecc, median
2.0 s) and are quoted only as a consistency check — 24 events cannot measure a rate to better than
a few per cent, and its lighter median cost is the same manifest-composition effect §10.1 is about.

**What this does to §8's recommendation: it weakens it, and that is the useful result.** The census
was expected to size a target. It also scores it, through the proxy doc 117 built: on the 572 nuecc
events where the transfer moved the pick, the transferred vertex is **closer** to a true in-FV
vertex than the composite winner on 294 and **further** on 157 (sign test p = 1.1 × 10⁻¹⁰); on `cv`,
172 closer against 51 further (p = 1.6 × 10⁻¹⁶). The pass is not merely expensive — **it is
consistently right**, on both samples, by a wide margin.

So a lever here is not "skip work that does nothing". 30.7 % of the time it moves the answer, and
when it moves it, it moves it the right way about two times in three. Three consequences worth
writing down before anyone opens this:

1. **`agree` is not a skip predicate.** 94.5 % of nuecc events agree, which looks like an enormous
   saving — but agreement is an *output* of the second pass. It is known only after paying for it.
2. **The saving is bounded by the whole pass.** 3.16 core-h over 2 001 events is ≈ 5.7 s/event
   against a job whose post-flip nuecc median is ~17 s; that is the 31.1 % §3.1 measured, and there
   is no smaller sub-piece the census can point at — `off_ms` covers the pass end to end.
3. **It is a physics decision, not a perf round.** Any predicate that skips the pass changes which
   events get the snap, so it needs grading against doc 107's metrics. §5 rule 1, owner's call —
   which is what §8 already said, now with a number attached to what would be given up.

**The honest summary: this is a worse lever than §8 has been implying for five rounds.** It remains
the largest CPU item, and it is now the *least* attractive of the campaign's remaining targets,
because unlike round 3's hoist or round 5's allocator it cannot be had byte-identically at all.

### 11.7 What round 5 recommends

**To the owner, one decision:** flip `SBND_TCMALLOC_LIB` to jemalloc 5.3.0 in the SBND PR runners,
or do not. Everything needed to decide is above:

- **For.** −18.8 % mean / −19.1 % max peak RSS on the tail, three of three events above 2 GiB taken
  below it, −4.0 % CPU there (read as "no CPU cost"), byte-identical on 62 events with a specific
  null pair, one line in one runner, and an exact precedent in round 1.
- **Against.** It buys **nothing where there is no tail** — +0.1 % mean RSS on the 24 gate-manifest
  nuecc events, and 1.197 vs 1.197 GiB on the p50 event. This is a tail-only change, worth taking
  only if the operational constraint is a per-process cap; an average will not justify it. On jobs
  of a few seconds it is **+4.5 to +6.6 % CPU** (absolutely, +0.04 to +0.20 s/event). It adds a
  third allocator to a chain that already has two histories. And the byte gate is 62 events on one
  detector: §9.5 item 7's pointer-keyed `unordered_set` iteration is a real mechanism, and no gate
  here proves its absence — only that it did not fire.
- **A cheaper partial.** `TCMALLOC_RELEASE_RATE=10` keeps production's allocator, needs no new byte
  gate, costs no CPU, and takes one of the three events below 2 GiB (−4.4 % mean). If the aim is
  "fewer cap breaches" rather than "less memory", it may be enough.

**Nothing else in this round is a decision.** The tripwire changes (§11.4) are shipped; they change
no operating point and `prod_cfg_gate.py` reports PASS 26/26 against the new reference.

**For the campaign, the ranking after round 5:**

1. **The allocator** — sized, gated, waiting on one decision.
2. **`CreateSteinerGraph`'s per-cluster working set** (§10.6) — still the largest object on all
   three detectors in both CPU and memory, still blocked on a frame-pointer build before anything
   inside the retile sampler can be resolved, and still likely a physics trade rather than a lever.
3. **The dual second pass** — biggest number, worst lever, now measured (§11.6). Do not re-open it
   as a perf round.
4. **`time2drift`** (§9.6) and **retention** (§8 item 2) — both still open, both still small.

**And one rule, earned three times in two rounds.** Round 4 retracted two attributions taken from
`(inline)` frames (§10.5); round 4's p50 row mixed a glibc `.time.meta` into a jemalloc table; round
5 opened by retracting the very number that motivated it. All three are the same failure: **a
quantity was compared against another quantity produced by a different instrument.** The discipline
that catches it is cheap and is now built into `scripts/d119/r5_alloc.py` — print every arm under
every instrument, label the claim instrument, and never let a table hold two.

The same script carries the other lesson of this round in a guard: `work-r3cv-d119tcm` already
exists and is **round 1's** arm on the **pre-round-3 binary**, so reading round 5's jemalloc against
it would have folded round 3's −14.9 % CPU win into what was reported as an allocator delta. The
M13 arm marker would have refused a *write* there; nothing stopped a *read*, so `r5_alloc.py` now
checks the marker before it reads, and round 5's own 62-event arms carry an `r5` tag.

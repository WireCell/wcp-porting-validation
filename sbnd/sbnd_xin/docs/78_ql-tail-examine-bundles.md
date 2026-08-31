# Doc 78 — Q/L heavy tail: ExamineBundles / connect_graph_relaxed (perf round 2)

Owner ask (2026-08-24): *"follow [doc 76] and attack the Q/L tail issue.
What we want is 1. gated only for busy events, no need to do the
approximation for the regular events. 2. for the busy events, use the final
PR results to help to determine if it is good enough. if they are please
turn them on for default for SBND running. Note, we want to explore the
major improvements on this area for these events. It is OK to perform
multiple rounds."*  Owner also confirmed (plan questions): include an exact
(byte-identical) round first; round-2 accept bar = **no nusel selection-flag
flips** on any affected event, score drift reported per event.

**Status.**  Rounds 0-3 DONE, all gates PASS.  Round 1 (exact
restructurings, byte-identical, no knob): the worst tail event's
ExamineBundles drops 123 s → 49 s, tail population −33%.  Round 2
(`eb_fast` → the `relaxed_fast` graph flavor: busy-cluster lazy-Kruskal
walk in ExamineBundles) and round 3 (`po_fast` → the same mode inside
ProtectOverclustering's duplicate): the worst event's Q/L job drops
**167 s → 37 s** (EB 123 → 6 s, PO 17.5 → 2.4 s), the 86-event tail
population −48% summed wall, regular events untouched — and the knob-ON
outputs are **byte-identical to the legacy path on all 186 gate events**
(43 of them engaging the lazy path), so the owner's accept bar (final PR
results unchanged) is met in the strongest form.  SBND production defaults
flipped ON (sec 7).

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
S=<scratch>/qltail   # this round: /home/xqian/tmp/claude-25225/.../scratchpad/qltail

# 0a. Tail distribution from the existing ql0819 production logs (no rerun)
for d in work-mcp1k-ql0819/ql_evt*/; do e=${d#*ql_evt}; e=${e%/}
  grep -h "MABC timing: ClusteringExamineBundles" $d/wct_ql_evt*.log \
   | sed 's/.*took \([0-9.]*\) ms.*/\1/' | awk -v e=$e '{s+=$1} END{print e, s/1000}'
done   # -> $S/../eb_times_mcp1k.txt (same for mcp2k)

# 0b. Census of the worst event (log-only env probe, doc pr/53 F0)
WCT_RELAXED_EDGE_CENSUS=1 ROOT=$S/census-r0 ./run_ql_batch.sh -j 2 280884 286617

# 0c. CPU profile (M17: precompiled cfg; needs the photodet dir on WIRECELL_PATH)
export WIRECELL_PATH=$TK/cfg:$WD:$WD/sbnd/photodet
cd $S/census-r0/ql_evt280884
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4 \
  CPUPROFILE=$S/ql280884.prof CPUPROFILE_FREQUENCY=250 \
  wire-cell -l stderr -L info -c .wct-cfg-evt280884.json
google-pprof --text --cum $(which wire-cell) $S/ql280884.prof | grep -Ei "hough|good_point|connect_graph"

# 1. Round-1 A/B (base = pre-edit HEAD 0e8b7334, then the round-1 build)
QL_EXTRA=-save-pctree ROOT=$S/base-ql ./run_ql_batch.sh -j 6 -f $S/ql_manifest.txt
# (build round 1) then:
QL_EXTRA=-save-pctree ROOT=$S/r1-ql   ./run_ql_batch.sh -j 6 -f $S/ql_manifest.txt
python3 scripts/analysis/qltail/ql_gate_r1.py $S   # all 6 archives x 186 events
# PR gate (base arms = doc 77's rm10-{mcp1k,nuecc48,ncpi0}, same binary as HEAD):
PR_JOBS=8 ./run_pr_chain_batch.sh work-mcp1k-ql0819   $S/r1pr-mcp1k   data $(cat <gate>/mcp1k241.txt)
PR_JOBS=8 ./run_pr_chain_batch.sh work-nuecc48-ql0819 $S/r1pr-nuecc48 sim
PR_JOBS=8 ./run_pr_chain_batch.sh work-ncpi0-ql0819   $S/r1pr-ncpi0   sim
python3 scripts/pr85_hash_gate.py --jobs 8 <rm10-arm> $S/r1pr-<s>
python3 scripts/pr94_root_gate.py          <rm10-arm> $S/r1pr-<s>
diff <(sort <rm10-arm>/nusel-table.tsv) <(sort $S/r1pr-<s>/nusel-table.tsv)
# PDHD+PDVD: abtest run_events.sh pre_doc78r1c/post_doc78r1c clus + ab_compare.sh
# uBooNE:    qlport sweep_5384.sh doc78r1base/doc78r1 6 + ab_check.sh
```

## 1. The tail, measured (production ql0819 logs, summed ExamineBundles ms)

| sample | n | sum | median | ≥5 s | share of sum |
|---|---:|---:|---:|---:|---:|
| mcp1k | 1000 | 1241 s | 1 s | 30 evts → 574 s | 46% |
| mcp2k | 2000 | 2138 s | 1 s | 56 evts → 844 s | 39% |
| nueCC48 | 48 | — | ~2 s | 3 (174637 13 s, 90055 6 s, 256587 5 s) | — |
| NCpi0 | 19 | — | ~2 s | 1 (285567 6 s) | — |

~3% of events carry ~40–45% of the step's cost.  MC **signal** events do
cross a ≥5 s cut, so any busy-event approximation touches signal — that is
why the round-2 accept bar is per-event selection stability, not averages.
On the worst event (mcp1k 280884, 187 s job) ExamineBundles is 137 s (74%):
apa1-0 67 s + all 70 s; then ProtectOverclustering 19 s, Deghost 18 s.

## 2. Round 0 — where the time actually goes

### 2.1 Census (`WCT_RELAXED_EDGE_CENSUS=1`, evt 280884)

One merged flash-bundle cluster dominates: **npoints 41 979, ncomp = 1004**
→ 503 506 pairs (twice: apa1-0 and all).  Everything else has ncomp ≤ 53.
`num` (the connected-component count, computed before any expensive work)
separates the monster from everything else by a factor ~20 — exactly the
gate signal doc 76 sec 9.3 predicted.  Per-pair census over 1.01 M walked
pairs (both instances):

* total walk steps 105.8 M (1 cm `test_good_point` steps, 3 kd probes each);
* only 7.5% of pairs are killed by the walk; **80.2% of steps belong to
  pairs whose verdict is "keep" under every predicate branch** — an exact
  early-exit cannot touch those;
* 45.9% of steps are on pairs ≥ 180 cm apart (paths of metres through
  live charge — kept, and almost never used: the MST keeps ~num of 503 k).

### 2.2 CPU profile (250 Hz, full Q/L job, 40 055 samples = 160 s)

| function (cumulative) | share |
|---|---:|
| `connect_graph_relaxed` | **73.5%** |
| — `vhough_transform` (all call sites) | 43.9% |
| — `test_good_point` (main pair walk) | 25.2% |
| — `is_good_point` (dir1/dir2 walks + PO + ctpc) | 16.5% |
| — `Simple3DPointCloud::get_closest_points` (pair distances) | **3.9%** |
| `Separate_overclustering` (PO's inline duplicate) | 10.7% |
| `connect_graph_ctpc` (Deghost) | 9.4% |

Three sites inside `connect_graph_relaxed`, differently gated:
site A (30 cm per-component Hough probes, already `close_enough` +
size-filtered — rarely fires on the monster whose components average 42
points), site B (**two 15 cm whole-cluster Hough transforms per pair**,
`clus/src/connect_graph_relaxed.cxx` old :273-284, gated only by an angle
window, whose only effect is choosing WHICH kill predicate applies), and
site C (the 1 cm walk, ungated, O(num² × path-cm)).  The O(num²) closest-
pair kd queries everyone suspects are 3.9% — NOT the problem.

## 3. Round 1 — exact restructurings (byte-identical, no knob, all events)

Four transformations in `clus/src/connect_graph_relaxed.cxx` + the mirror
in `clus/src/clustering_protect_overclustering.cxx` (its header comment
requires mirroring).  Each is waste-removal with a proof of exactness, not
an approximation:

1. **Site-B lazy skip.**  Both kill verdicts (`kill_strong` from
   `num_bad1[0]`, `kill_nonstrong` from the angle-branch ladder — a
   verbatim transcription of the old :296-332) are evaluated BEFORE the
   two whole-cluster Houghs; the Houghs run only when the verdicts
   disagree (census: they agree on 93.8% of pairs).  When they agree the
   flag they set is provably moot.
2. **Main-walk early exit.**  All counters are monotone and `num_steps`
   is fixed, so once `num_bad1[0]>7 && num_bad[0]>7 && nb[2]+nb[3]>9 &&
   nb[1]+nb[3]>9 && nb[2]+nb[1]>9` every branch's predicate is satisfied
   — kill is forced regardless of angles/strong-check; break.  (The
   `>=3`/ratio alternatives only ADD kill reasons — not needed for
   sufficiency.)  Saves the kill-side 7.5–20% of walk steps; the 80%
   keep-side is untouchable exactly (sec 2.1).
3. **dir1/dir2 walks**: break at `num_bad == 8` (`num_bad1 >= num_bad`
   because `is_good_point(allowed_bad=0)` is strictly tighter, so both
   branch predicates fire); and when the loose `is_good_point` fails the
   tight second call is skipped (same monotonicity, same counters).
4. **PO mirror**: the same `num_bad > 7` early exit in
   `Separate_overclustering`'s `check_path` (single OR-predicate).

All guarded by `!oc53_census` where an early exit would truncate the
census log's counters — census mode keeps the legacy path bit-for-bit.
`num_bad2[]` was checked to be write-only (dead) before touching the loop.

### 3.1 Effect (byte-identical, so this is pure waste removal)

| metric | base | round 1 |
|---|---:|---:|
| evt 280884 ExamineBundles (apa1-0 + apa0-0 + all) | 123 s | **49 s** |
| evt 280884 Q/L job wall (solo-ish, -j2) | 167 s | 96 s |
| evt 286617 ExamineBundles | 72 s | **25 s** |
| 86 tail events (≥5 s), summed Q/L wall (-j6) | 2534 s | **1702 s (−33%)** |
| tail max wall | 175 s | 99 s |
| 100 regular mcp1k events, summed wall | 516 s | 510 s (unchanged) |
| PDHD heavy clus events (abtest) | 158/144 s | 155/141 s |

The tail's residual is now ~40% site-C keep-side walk + ~25% ambiguous-pair
Houghs + PO/Deghost — see sec 8.

## 4. Round-1 verification (all PASS)

* Freshness: `build/clus/libWireCellClus.so` + `local/lib` 02:35:14 >
  last edit; base rebuilds for the A-sides at 02:21:51 / 02:32:18
  (`.build-fingerprint`-style header lines are quoted in each launch log).
* `./build/clus/wcdoctest-clus`: 232 cases, 2379/2379 assertions, twice
  (after first r1 build and after the final one).
* **SBND Q/L gate** (`scripts/analysis/qltail/ql_gate_r1.py`, rescued into
  the repo 2026-08-25 from `$S/qltail/` when that arm was deleted): 186-event manifest
  (30 mcp1k tails + 56 mcp2k tails + first 100 regular mcp1k), 6 archives
  per event incl. the two per-face mabc zips that `ql_arm_compare.py`
  does not hash: **185/186 events byte-identical**.  The 1 "FAIL"
  (mcp2k 283833, pctree only, single member `cluster_scalar/lm_flag`,
  one cluster 2→0) is a **pre-existing run-to-run nondeterminism**:
  two repeats on the SAME round-1 binary produced BOTH hashes, one of
  them exactly matching the base arm (`$S/qltail/r1-rep{1,2}`).  Known
  class (LM verdict), not introduced by this change; left as a reported
  bug, not fixed here.
* **SBND PR gate**: r1 binary PR arms on the doc-76/77 308-event manifest
  vs doc 77's `rm10-*` arms (same base binary as HEAD; arm labels
  `$S/qltail/r1pr-{mcp1k,nuecc48d,ncpi0d}`):
  pr85 **PASS 482/482 + 96/96 + 38/38 archives byte-identical**, pr94
  **PASS 241 + 48 + 19 events**, sorted nusel-table diff **0 lines** for
  all three samples.  (First attempt ran the MC arms with `reality=sim`
  and every event differed wholesale — the doc-76/77 gate arms use
  `reality=data` for ALL samples including MC; the reruns
  `r1pr-nuecc48d`/`r1pr-ncpi0d` match that convention.  Lesson: the PR
  A/B reality flag is part of the arm definition; always read the base
  arm's `.lineage_reality` before launching the counterpart.)
* **PDHD+PDVD abtest**: `ab_compare.sh pre_doc78r1c post_doc78r1c`
  (clus stage, 6-event events.txt manifest; imaging could not be re-run —
  SP frames retired — and the change touches only clus):
  **OVERALL PASS**, wall/RSS flat.
* **uBooNE qlport**: `ab_check.sh doc78r1 doc78r1base` (35 events):
  **ZIPS 35/35 content-identical; TAGGER identical=35 diff=0** (both gates).

### 4.1 A-side integrity note (mistake made and repaired)

The round-1 binary was rebuilt at 02:29:26 while the base Q/L arm was
still running (the exact "never wcbuild while a batch runs" mistake,
pr/103).  Damage: 4 events crashed at dlopen (`invalid ELF header`) and
14 events started after the overwrite ran the wrong binary.  All 18 were
identified from `.status` mtimes minus wall_s and rerun on a freshly
rebuilt base binary into `base-ql-fix/` (18/18 rc=0); the gate reads
those events from the fix arm.  Events already in flight at the overwrite
kept their mapped pages, completed rc=0, and hash-match the r1 arm.

## 5. Files touched (round 1)

* `clus/src/connect_graph_relaxed.cxx` — sites B/C + dir walks (sec 3).
* `clus/src/clustering_protect_overclustering.cxx` — `check_path` early
  exit.
* No config, no knobs, no schema change; no other package.

## 6. Rounds 2+3 — busy-gated lazy walk (IMPLEMENTED, SBND ON)

The doc 76 sec 9.3 shape: per-cluster gate `num > busy_num_threshold`
(threshold 200; the ExamineBundles monster is num=1004, the
ProtectOverclustering counterpart num=988, vs ≤53 for everything else on
the census events).  At or below the threshold the legacy path runs
bit-for-bit.

Above it, the per-pair walk is evaluated **lazily by a single-pass
Kruskal**: pairs are considered in ascending (distance, j, k) order and
the walk verdict (`eval_pair_verdict` / `check_path` — the round-1 text,
factored into a lambda) is computed only when a pair would bridge two
union-find components; killed bridges are skipped, and the accepted set
is the spanning forest of the walk-filtered graph.  Near pairs (< 3 cm,
the direct-edge rule) and pairs with a directional candidate (whose MAIN
entry gates the dir edges through `process_mst_deterministically`'s
recording) are walked eagerly.  On 280884 this walks 15 941 of 503 506
pairs and accepts 988 bridges.

Two design notes recorded for the next reader:
* A kill-and-rebuild Prim fixpoint was tried first and REJECTED: killed
  bridges cluster, each rebuild proposes more killable edges, and the
  iteration cap then walks everything — measurably SLOWER than eager
  (280884 apa1-0: 41 s vs 23.5 s).  The single-pass Kruskal replaces it.
* Kruskal vs the legacy per-component Prim can differ only on exact
  distance ties.  Empirically the knob-ON arm is **byte-identical to the
  legacy arm on all 186 gate events** (sec 7) — ties did not bite — but
  because identity is not PROVEN in general the mode stays behind the
  busy gate and the knobs, exactly as the owner specified.

Wiring: round 2 = new graph flavor `relaxed_fast`
(`connect_graph_relaxed(..., const RelaxedFastCfg*)` +
`make_graph_relaxed_fast` + 4 registrations in `find_graph` /
`graph_algorithms`), selected by ClusteringExamineBundles' existing
`graph_name` config key; SBND cfg arg `eb_fast` (all three EB instances),
TLA + runner env `SBND_EB_FAST`.  Round 3 = config key
`busy_num_threshold` (C++ default 0 = off) on
ClusteringProtectOverclustering, threaded into its inline duplicate;
SBND cfg arg `po_fast` (=> threshold 200, per-face instances), TLA +
runner env `SBND_PO_FAST`.  Gotcha found on the way: `Configuration` has
no `convert<size_t>` specialization — `get<size_t>` silently returns the
default; the component reads the key as int and casts (a debug log
`Separate_overclustering: npoints=.. ncomp=.. lazy=..` now shows the
gate decision per multi-component cluster).

## 7. Final verification (rounds 1-3 combined, final binary — all PASS)

The complete battery was re-run on the final binary (the one carrying
rounds 1+2+3; freshness proof: `local/lib` == `build/clus` lib mtime
03:45:05, sizes equal; `wcdoctest-clus` 2379/2379 twice more).

* **SBND Q/L, knob OFF** (`$S/qltail/r3off-ql` vs the base composite):
  **PASS 186/186 events**, 6 archives each — incl. 283833, which this
  time landed on the base lm_flag mode.
* **SBND Q/L, knob ON** (`SBND_EB_FAST=1 SBND_PO_FAST=1`,
  `$S/qltail/r3on-ql`): **PASS 186/186 — byte-identical to the legacy
  arm** with the lazy path engaged on 43 events.  The "approximation"
  is output-preserving on every tested event, so the owner's accept bar
  (final PR results, no nusel flag flips) is met with zero movers.
* **SBND PR** (`$S/qltail/r3pr-{mcp1k,nuecc48,ncpi0}`, reality=data, vs
  doc 77's `rm10-*`): pr85 **PASS 482+96+38 archives**, pr94 **PASS
  241+48+19 events**, nusel diffs **0** — all three samples.
* **PDHD+PDVD abtest**: `ab_compare.sh pre_doc78r1c post_doc78r3` —
  **OVERALL PASS** (clus stage, 6 events).
* **uBooNE qlport**: `ab_check.sh doc78r3 doc78r1base` — ZIPS **35/35
  content-identical**; TAGGER 34/35 identical + **1 event (6805) proven
  binary-LAYOUT-dependent, not a semantic effect**: its four
  `kine_pio_*_2` scalars flip between two modes; the r3 binary
  reproduces its own mode across repeats, and a control build that adds
  ONLY an unused padding string to an unrelated TU flips the event BACK
  to the base-binary mode with cmp-identical clustering zips throughout.
  This is the pr/97 class (residual address-dependence in the uBooNE
  second-pi0-photon selection) — reported for a pr/97-style follow-up,
  not chased here.

### 7.1 Perf summary (per-event `.status` walls, -j6/-j2 idle box)

| metric | base | r1 (exact) | **r2+r3 ON (production)** |
|---|---:|---:|---:|
| evt 280884 Q/L wall | 167 s | 96 s | **37-53 s** (38 s in the ON arm) |
| — ExamineBundles (3 inst.) | 123 s | 49 s | **6.3 s** |
| — ProtectOverclustering (apa1-0) | 17.5 s | 17.5 s | **2.4 s** |
| evt 286617 Q/L wall | 105 s | 52 s | **24-32 s** |
| 86 tail events, summed wall | 2534 s | 1702 s (−33%) | **1330 s (−48%)** |
| 100 regular mcp1k events, summed | 516 s | 510 s | **510 s** (unchanged) |
| 186-event arm, batch wall (-j6) | 581 s | 375 s | **315 s** |

The residual on 280884 is now Deghost (16.5 s, `connect_graph_ctpc` —
different mechanism, parked) plus ordinary per-event work.

## 8. SBND production ON + escapes

`wct-clus-matching-perevt.jsonnet` defaults flipped: `eb_fast = true`,
`po_fast = true` (doc 76 `fast_xgb_forest` precedent).  Compiled-config
proof: the default compile carries `graph_name: relaxed_fast` ×3 and
`busy_num_threshold: 200` ×2; the escapes `SBND_EB_FAST=0` /
`SBND_PO_FAST=0` compile with zero occurrences of either key (legacy),
and the runner-compiled OFF config was proven byte-identical to the
pre-change compile (paths normalized).  C++ defaults are OFF
(`graph_name="relaxed"`, `busy_num_threshold=0`), so uBooNE, PDHD, PDVD
and every other consumer compile and run unchanged.

## 9. What was deliberately not done

* No change to `connect_graph_relaxed_pid` (PR-stage NeutrinoPatternBase
  flavor) — not in the Q/L tail profile.
* Deghost (`connect_graph_ctpc`, 16.5 s on 280884) — parked, different
  mechanism (shortest-path), the natural next round if the owner wants
  the last of the straggler.
* The lm_flag bimodal nondeterminism (sec 4) and the uBooNE 6805
  layout-dependence (sec 7) — both pre-existing, reported for follow-up,
  not fixed in this change.

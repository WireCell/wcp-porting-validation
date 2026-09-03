# PDVD doc 26 — the non-terminating `examine_partial_identical_segments` loop behind the 039349/14, /53 hang

**Status:** FIXED, unknobbed. NOT bit-identical on the two PDVD events that never
terminated (they now finish); byte-identical everywhere the guard does not fire
(gates in §5, §7.4). Toolkit commits `441b6de4` (round 1) and `87903977`
(round 2, §7); this document, the gate scripts and their results are the
wcp-porting-img record. The owner's decisions on the §7.5 questions are in
§8 (2026-09-03): the duplicate-pair merge rule is NOT pursued; 039349/14 goes
to a separate Steiner-terminals campaign, and the next round on this thread is
the over-clustering behind 039349/53.

**Correction (round 2, §7.1).** Round 1 attributed the hang to the
`fetch_channel_from_anode` crash fix (toolkit `3e1854a8`) because the two
events hung on the first arm run after it. That was a coincidence of timing:
the pre-fix binary hangs on both events too once the PDVD production config
of `228f1c39` (07:51, E = 450 V/cm defaults) is in place, and on that config
the crash fix itself moves zero tagger verdicts and is byte-identical on
117/120 events (the other three: 41 no longer crashes, 14/53 hang either way).
§3 below is written as it now stands; the round-1 text it replaces is in the
git history of this file.

**Correction 2 (doc 27, same day).** The 039349/53 reading in §7.5 ("the void
is real in the charge too") and the owner's §8 decision 2 built on it are
WRONG: the input cluster is one continuous 200 cm track; the PR job had
re-tiled it one face height off in y because the point tree was sampled with
the v6 wires file and the PR job compiled the v7-uvwfit one (07:51 commit
`228f1c39`; the two files order the faces of anodes 2, 3, 6, 7 oppositely).
There is no over-clustering to separate. The same mismatch sits in every PDVD
PR arm run after 07:51 on a pre-07:51 point tree (`d25r13*`, `d25r14*`), so
the §7.1 attribution of the pre/post-07:51 verdict churn to the E-field
operating point is at best partial. Details, guards and the self-consistent
replacement arm: `27_pdvd-stale-geometry-face-swap.md`. The loop fix itself
(§2, §4) is unaffected.

**Scope.** One C++ change in `clus/src/NeutrinoStructureExaminer.cxx` (plus a
declaration in `NeutrinoPatternBase.h`) and one new doctest. No config, no
knob, no other experiment's files. Follows doc 25 §13.4 item 12, which shipped
the `fetch_channel_from_anode` lower-bound fix and left this problem open.

## 0. Repro

```bash
# The hang (post-crash-fix binary, before this fix): event 14 never finishes.
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
mkdir -p work/039349_14_<tag>; cp -n work/039349_14_d25r12eager/pctree-evt19689.{tar.gz,tlas} work/039349_14_<tag>/
LD_LIBRARY_PATH=/home/xqian/tmp/doc25r12/lib_fix:$LD_LIBRARY_PATH PDVD_MAX_JOBS=1 ./run_pr_evt.sh -s <tag> -stm-fit 39349 14
# ~40 s in, the log repeats this triple forever (one iteration ~0.42 s):
grep -c 'do_rough_path: cluster 36' work/039349_14_<tag>/wct_pr_039349_14.log
gdb -p $(pgrep -f 'wire-cell.*039349_14_<tag>') -batch -ex 'bt 8'   # frame 4 = examine_partial_identical_segments

# The unit test that reproduces the decision (fails before the fix, §4):
./build/clus/wcdoctest-clus -tc='partial_identical_split_point*'

# The 120-event gate + guard census (both arms must exist, §5):
python3 stm/gates/r13_loop_gate.py            # -> stm/gates/r13_compare.txt, r13_census.tsv
```

## 1. Symptom

Doc 25 §13.4 item 12 fixed a SIGSEGV in `TrackFitting::fetch_channel_from_anode`
(negative wire index, PDVD 039349/32). On the first 120-event `_keep` rerun
after it — which, unnoticed at the time, was also the first rerun after the
config commit `228f1c39` — two unrelated events stopped finishing:

| event | last pre-fix arm (07:14, old config) | first post-fix arm (08:02, new config) | log |
|---|---|---|---|
| 039349/14 | 13 s | killed at 2891 s | 44 MB, `do_rough_path` ×20652 (20652 on cluster 36), `do_multi_tracking` ×6935 |
| 039349/53 | 53 s | killed at 2649 s | 306 MB, `do_rough_path` ×18671 on cluster 53, `do_multi_tracking` ×6271 |

Not a deadlock: 100 % CPU, RSS creeping (1.17 → 1.81 GB), log timestamps
advancing. The tell is in the `do_rough_path` lines: on cluster 36 every
iteration recomputes the same three paths from the same point,

```
first=(2732.6,-1189.0,866.1) last=(2732.6,-1189.0,866.1)     # zero length
first=(2732.6,-1189.0,866.1) last=(1968.6,-1677.6,1514.8)
first=(2732.6,-1189.0,866.1) last=(1968.6,-1674.7,1509.8)
```

~6880 times (event 53: the same shape, ~6220 times). A search that merely had
more to do would not recompute an identical triple.

## 2. Root cause

`gdb -p` on a rerun (frame 4) names the loop: `PatternAlgorithms::
examine_partial_identical_segments`, the `while (flag_continue)` in
`NeutrinoStructureExaminer.cxx`, called once per `find_proto_vertex` from
`TaggerCheckNeutrino::run_dual_chain_off_pass`. The function looks for a
vertex of degree > 2 whose two segments overlap (fit points within 0.3 cm) for
more than 5 cm leaving the vertex, and splits the shared trunk at the far end
of the overlap (`max_point`): if an existing vertex sits within 0.3 cm of
`max_point` it merges there, otherwise it creates a new vertex at the steiner
point closest to `max_point`, moves both segments to it, refits, and sets
`flag_continue`.

A temporary debug line inside the branch (removed before commit) gave the
per-iteration state on event 14:

```
EPIS cluster 36 vtx deg=3 wcpt=(2732.6,-1189.0,866.1) fit=(2729.0,-1196.2,862.8) max_dis=8.65 cm max_point=(2666.1,-1226.2,913.9) min_dis=8.65 cm branch=new nv=5 ne=5
EPIS new: vtx_new=(2732.6,-1189.0,866.1) sg3_path=2
EPIS new sg1: tmp=(1968.6,-1677.6,1514.8) path=3      EPIS new sg2: tmp=(1968.6,-1674.7,1509.8) path=4
EPIS new done: vtx deg=2 vtx2 deg=3 nv=6 ne=6
EPIS cluster 36 vtx deg=3 wcpt=(2732.6,-1189.0,866.1) fit=(2732.6,-1189.0,866.1) max_dis=8.69 cm ... branch=new nv=6 ne=6
...
EPIS new done: vtx deg=2 vtx2 deg=3 nv=145 ne=145
```

Read it as a fixed point:

1. The vertex V has three segments; two of them run to points A and B that are
   5.8 mm apart and 111 cm away, and their fits coincide for the first 8.69 cm.
   Both leave V into a charge gap: the rough path from V to A has 3 steiner
   points, V to B has 4 (a 111 cm track routed on `steiner_graph_gap` edges).
2. `max_point` is 8.69 cm out. No vertex is near it (`min_dis` = `max_dis`), so
   the new-vertex branch runs. `kd_steiner_knn(1, max_point)` returns **V's own
   steiner point** — there is nothing in the steiner cloud between V and the
   split location, because there is no charge there.
3. The branch therefore creates a clone vertex at V, a two-point zero-length
   segment (V → V) between the clone and V, and re-attaches both overlapping
   segments to the clone. `flag_continue = true`.
4. The clone now has degree 3 with the same two overlapping segments and the
   same `max_point`; the closest steiner point is still V. Every pass adds one
   vertex and one zero-length edge (`nv` 5 → 145 in the 700 instrumented lines)
   and re-runs `do_multi_tracking` on a graph one segment larger. Forever.

The prototype (`NeutrinoID_proto_vertex.h`, `examine_partial_identical_segments`)
has exactly this logic — `pcloud->get_closest_wcpoint(max_point)` with no check
that the point differs from the vertex — and the same fixed point. It never
arose there because on uBooNE and SBND the steiner cloud is dense along every
track, so the closest steiner point to `max_point` is always within a few mm of
`max_point`, never 8.7 cm away at the vertex.

The port had already added `path.size() >= 2` guards the prototype lacks; they
did not help here (the V → V path has two identical points) and a second latent
variant — an empty kNN falling through to `flag_continue = true` — would loop the
same way. Both are closed by the fix.

## 3. Why it hid

Three layers, each necessary:

- **Before 07:51 the two events never reached this code — and it was the
  config, not the crash fix, that changed that.** The last pre-fix arm and the
  first post-fix arm differ in two things: the crash fix and toolkit `228f1c39`
  ("adopt E=450 V/cm production defaults": `pr.jsonnet`, `pdvd_track_fitting.json`,
  `particle_dataset.jsonnet`, `params.jsonnet`, all read by the PR job). Between
  those two arms 802 of 5434 STM verdicts, 646 FC and 437 TGM verdicts moved on
  119 of 120 events, so different bundles became neutrino candidates
  (`selected main cluster` 23/18/45 before vs 48/36 after on event 14) and
  cluster 36 (event 14) / cluster 53 (event 53) entered `find_proto_vertex`.
  Re-running both binaries on today's config (§7.1) separates the two: the
  crash fix moves **zero** verdicts and is byte-identical on 117 events, and the
  **pre-fix binary hangs on 14 and 53 as well** (13228 and 10666 rough paths at
  the kill). The loop was always reachable; the new operating point reaches it.
- **The geometry is PDVD-specific — and it is a hole in the steiner cloud, not
  (on event 14) in the charge.** The trigger is two near-duplicate segments
  leaving a vertex into a region with no steiner points, which
  `steiner_graph_gap` bridges with long edges. §7.5 shows what that region is:
  on 039349/14 the imaged charge is continuous along the whole track but the
  steiner cloud (632 points) covers only the half above V — 0 steiner points
  along the 111 cm from V to A; on 039349/53 charge and steiner cloud both stop
  75 cm short of V. SBND and uBooNE steiner clouds do not have such holes along
  tracks.
- **The loop's own guard is unconditional.** `flag_continue` is raised whether
  or not the graph changed in a way that can converge; nothing in the function
  or its callers bounds the iteration count, and the per-iteration
  `do_multi_tracking` is cheap enough (0.42 s) that the job looks alive.

## 4. Fix

`clus/src/NeutrinoStructureExaminer.cxx`, `NeutrinoPatternBase.h`:

- New helper `PatternAlgorithms::partial_identical_split_point(cluster,
  max_point, vtx_point)` — the closest steiner point to `max_point`, exactly
  the prototype's `get_closest_wcpoint(max_point)`, returning `std::nullopt`
  when the cluster has no steiner points or when that point lies within 0.1 cm
  of the vertex's own position.
- The new-vertex branch asks the helper; on `nullopt` it logs one DEBUG line
  (`examine_partial_identical_segments: cluster N degenerate split skipped:
  vtx=... max_dis=... nv=... ne=...`) and `continue`s to the next vertex
  **without** raising `flag_continue`. The `while` loop then ends when a full
  pass makes no change, as designed.

**Why unknobbed.** The guard fires only in a state that is a fixed point of the
loop: after the clone is made the configuration on the clone is identical
(same wcpts, same segments, same closest steiner point), so from the second
pass on the iteration is deterministic and infinite. The only case in which
the guard could alter a run that would have terminated is if the single refit
after the first cloning happened to shrink the overlap below 5 cm — the
production instrumentation shows the opposite (8.65 → 8.69 cm, then constant).
Same category as the crash-path fixes in doc 25 §13.4 items 9–12: an
unambiguous intended answer (terminate), not a documented divergence, held to
the byte-identity gates in §5 rather than to a knob. The 0.1 cm tolerance only
matters when a vertex is not itself exactly a steiner point; the split the
algorithm wants is > 5 cm away, and steiner spacing is several mm.

**Test first** (`clus/test/doctest_partial_identical_split_point.cxx`, synthetic
steiner cloud on the x axis, vertex at the origin, `max_point` 8.69 cm out):

| case | helper with the prototype's semantics (step 1) | with the guard (step 2) |
|---|---|---|
| dense cloud (every 5 mm to 30 cm) | 8.5 cm point | 8.5 cm point |
| gap: only the vertex point below 20 cm (the production shape) | **returns the vertex itself — `CHECK_FALSE(has_value()) FAILED`** | `nullopt` |
| vertex 0.05 cm off the steiner point, same gap | **FAILED** | `nullopt` |
| split beyond the gap (`max_point` 21.2 cm) | 21.0 cm point | 21.0 cm point |
| no `steiner_pc`; present-but-pointless `steiner_pc` | `nullopt` | `nullopt` |

Step 1 (helper extracted with the original behaviour, test added): 3 cases,
2 passed, 1 failed, 2 of 14 assertions failed — the two gap sub-cases. Step 2
(guard added): 3/3, 14/14. Full `wcdoctest-clus`: 267/267 test cases, 2901
assertions.

## 5. Verification

Freshness: `local/lib/libWireCellClus.so` 09:23:15 > `NeutrinoStructureExaminer.cxx`
09:22:50 (2026-09-03); the new symbol is present in the installed library
(`nm -DC | grep partial_identical_split_point` → 1). Binaries pinned for every
arm below: `/home/xqian/tmp/doc25r13/lib` (PDVD fix arm), `lib_base` / `lib_post`
(uBooNE + SBND gates, built minutes apart from the same tree with and without
the two changed files; the peer session's uncommitted edits — all dated before
either build — are in both).

**PDVD, 120-event `_keep` manifest** (`stm/gates/r13_loop_gate.py`, results
committed as `r13_compare.txt` / `r13_census.tsv`). Arm B `d25r13fix` = the
guard, all 120 events. Arm A `d25r13base` = the same tree without the guard
(`lib_base`), 118 events — 039349/14 and /53 are left out because that binary
never terminates on them. The first attempt used `d25r12eager` (the arm that
hung) as arm A and every `mabc-pr.zip` differed while 116/116 calib dumps
matched; the single divergent member was `data/0/0-stm_fit-global.json`'s
`real_cluster_id` array (all 0 vs real ids) — toolkit `ab0762c6` (08:30, doc
pdvd/30) fixed that field between the 07:47 crash-fix build and this one. A
three-binary rerun of 039349/13 confirmed it: 07:47 build ≠ tree-without-guard
= tree-with-guard. Hence the clean pairing below. (Lesson, twice in one day:
an arm is a baseline only for the commits and config it actually ran with —
see §7.1 for the same trap on the config side.)

```
events_B=120 same=234 diff=0 missing=0 absent_in_A=2 no_dump_both=2
guard fired on 2 event(s): [('039349_14', 1), ('039349_53', 2)]
039349_14 fires=1 wall -1s -> 26s  ABSENT in arm A (never terminated there)
039349_53 fires=2 wall -1s -> 27s  ABSENT in arm A (never terminated there)
```

| | |
|---|---|
| common events | 118 (116 with a calib dump; 2 have no STM tag and therefore no dump on either side, doc 25 §13.10) |
| `mabc-pr.zip` + calib hash pairs | **234/234 identical, 0 diffs** |
| guard fires | exactly the two events that never terminated: 039349/14 ×1, 039349/53 ×2 |
| 039349/14 | killed at 2891 s → **26 s** (13 s before the crash fix) |
| 039349/53 | killed at 2649 s → **27 s** (53 s before the crash fix) |

→ **PASS** for the claim in §4: the guard changes nothing where it does not
fire, and where it fires the job was never going to finish. Wall-time sums
over the 118 common events (5146 s vs 5518 s) are arm-to-arm noise — the fix
arm ran concurrently with the uBooNE and SBND gates.

**uBooNE MABC two-gate** (`qlport/scripts/sweep_5384.sh`, 35 events, labels
`doc25r13base` vs `doc25r13fix`, `ab_check.sh`):

```
ZIPS: 35/35 content-identical
TAGGER: identical=35 diff=0 skip=0
```
→ **PASS** (gate 1 + gate 2). Sentinel census: `degenerate split skipped`
appears in 0 of the 106 log files under `sweep/doc25r13fix/`.

**SBND PR chain** (`sbnd_xin/run_pr_chain_batch.sh` on the `d99fix` Q/L roots,
`scripts/pr85_hash_gate.py`, arms `work-{nuecc48,ncpi0}-doc25r13{base,fix}`):

```
nuecc48: # events in A: 48  compared archives: 96   PASS all 96 archives byte-identical
ncpi0:   # events in A: 19  compared archives: 38   PASS all 38 archives byte-identical
```
→ **PASS** on both samples. Sentinel census: 0 of 67 PR logs.

Gates not run: the PDHD/PDVD img+clus `abtest` gate (this code runs only in the
PR stage, which that harness does not reach), and the 3067-event SBND
production sample (the guard's DEBUG sentinel is the cheap census for it —
`grep -c 'degenerate split skipped'` over any arm's logs).

## 6. Left open after round 1 (each taken up in §7)

- **Ping-pong variant.** The guard covers the closest steiner point coinciding
  with the vertex being split. If it coincided with a *different* existing
  vertex the branch would create a duplicate vertex there instead; whether that
  converges depends on the refit. Not observed; the DEBUG line does not cover
  it. A broader criterion — accept the split only if its steiner point is
  closer to `max_point` than to the vertex — would close both and is the
  natural next step if the sentinel ever fires elsewhere.
- **No iteration cap.** `examine_partial_identical_segments` and the other
  `while (flag_continue)` examiners still have no bound. A cap with a WARN would
  be a belt-and-braces safety net; it was left out to keep this change to the
  root cause.
- **The two segments themselves.** Cluster 36's V → A and V → B segments are a
  111 cm gap-bridged pair whose far ends are 5.8 mm apart. That `find_other_
  segments` produced both is a separate question about gap-bridged duplicates
  on PDVD, not addressed here.
- **The upstream verdict changes** from the crash fix (§3, STM 1 → 0 on four
  clusters across the two events) are real physics changes from removing
  garbage channel lookups near plane edges. They were disclosed in doc 25 §13.4
  item 12 as NOT bit-identical; a census of how many `_keep` bundles moved is
  still owed.

## 7. Round 2 — the four open items

### 7.1 Census of the crash fix's verdict flips (item 4 of §6)

`stm/gates/r13_verdict_census.py` diffs every `TaggerCheck{STM,TGM,FC}`
verdict line per (event, cluster) between two arms and marks flipped clusters
that are in the doc 25 dQ/dx sample (`stm/sample_index.tsv`).

The first pairing was contaminated and is kept as a warning: the last pre-fix
arm (`d25r12fast`, 07:14) vs `d25r13fix` gave 802 of 5434 STM verdicts moved on
119 of 120 events — far too many for an edge-wire effect. Toolkit `228f1c39`
(07:51, "adopt E=450 V/cm production defaults": `pr.jsonnet`,
`pdvd_track_fitting.json`, `particle_dataset.jsonnet`, `params.jsonnet`) sits
between those two arms, and the PR job reads all four files. The census was
therefore redone on today's config with two pinned binaries that differ only by
the crash fix: `d25r14pre` (round-12 pre-fix snapshot, `/home/xqian/tmp/doc25r12/lib`)
vs `d25r14cf` (`lib_fix`, crash fix only; 14 and 53 skipped — that binary hangs):

```
events compared: 118; verdict lines compared per tagger: {'FC': 6897, 'STM': 5763, 'TGM': 6944}
flips: {} on 0 event(s)
flipped clusters that are in the dQ/dx sample (stm/sample_index.tsv, 5 entries): 0
```

Zero flips. The companion hash gate (`r13_loop_gate.py d25r14pre d25r14cf`,
`stm/gates/r14_compare_crashfix.txt`): 232/232 hash pairs identical on 117
events; 039349/41 differs because the **pre-fix binary segfaults there** on
today's config (`fetch_channel_from_anode`, the item-12 crash moved from event
32 to event 41 with the new operating point) and its zip is empty; 14 and 53
never finish on either binary (killed at 1543 s / 1249 s pre-fix, 13228 /
10666 rough paths). So on PDVD the crash fix is byte-identical except where it
prevents a crash — doc 25 §13.4 item 12's "NOT byte-identical" and the
verdict-flip lists in round 1 of this document were the config change. The
first, contaminated census is kept in the git history of
`r13_verdict_census.tsv` (802 STM / 646 FC / 437 TGM flips); the committed
file is the clean one.

### 7.2 Pass budget for the `while (flag_continue)` examiners (item 2)

`clus/inc/WireCellClus/ExaminerPassBudget.h`: `ExaminerPassCounter` at the ten
PatternAlgorithms examiner loops (`examine_structure_2/3`, `examine_vertices`,
`examine_partial_identical_segments`, `examine_structure_final_1/2/3`,
`eliminate_short_vertex_activities`, `shower_clustering_with_nv_from_main_cluster`
/`_from_vertices`). Past `kExaminerPassBudget = 1000` passes it logs a WARN and
breaks; its destructor logs one DEBUG line per call (`examiner passes: <who>
cluster <id> passes=<n>`), which is the census input. Not covered: the
`while (flag_continue)` loops in `NeutrinoTaggerNuE.cxx` (`broken_muon_id`) and
`PRShower.cxx`, which walk segment chains rather than re-run a graph pass, and
`connect_graph_relaxed.cxx` (clustering, not PR).

Observed maximum over the round-2 fix arms (120 PDVD + 67 SBND + 35 uBooNE
events):

```
examiner                                       calls   max passes   where
eliminate_short_vertex_activities              1419        4        pdvd 039253/5 cluster 97
examine_partial_identical_segments             7342        9        pdvd 039252/3 cluster 62
examine_structure_2                            2651        8        sbnd nuecc48 evt 46363 cluster 19
examine_structure_3                            1419        9        pdvd 039349/46 cluster 64
examine_structure_final_1                      7523        4        sbnd ncpi0 evt 433451 cluster 4
examine_structure_final_2                      7523        3        pdvd 039253/6 cluster 270
examine_structure_final_3                      7523        4        pdvd 039349/32 cluster 63
examine_vertices                               8125       15        sbnd nuecc48 evt 256587 cluster 11
shower_clustering_with_nv_from_main_cluster     527        2        pdvd 039252/2 cluster 118
shower_clustering_with_nv_from_vertices         636        5        sbnd ncpi0 evt 389538 cluster 11
```
(222 logs; `stm/gates/r14_census_round2.tsv` has the PDVD rows.) Maximum 15;
the budget of 1000 is 60× above it. The WARN never fired.

The budget is therefore a no-op on every event that terminates today; a change
to the constant is pinned by the doctest so it is a decision, not drift.

### 7.3 The ping-pong variant (item 1)

The prototype tests the existing vertices against `max_point`, but the new
vertex is created at the steiner point closest to `max_point` — 8.7 cm away on
039349/14. `examine_partial_identical_segments` now decides the merge target
before branching: when no vertex sits within 0.3 cm of `max_point`, it also
asks `closest_cluster_vertex()` (nearest wcpt of this cluster) about the split
point itself; a vertex within 0.3 cm of it is the merge target (the prototype's
merge block, unchanged, runs against it), and the split vertex itself there is
the round-1 skip. On a dense cloud the split point is near `max_point`, no
existing vertex is within 0.3 cm of it unless it was already within 0.3 cm of
`max_point`, and the path is unchanged. DEBUG sentinel: `split point coincides
with an existing vertex ... merging there`.

Tests (`doctest_partial_identical_split_point.cxx`, appended): `closest_cluster_vertex`
picks this cluster's nearest wcpt and ignores another cluster's nearer vertex;
`ExaminerPassCounter` is silent for exactly `kExaminerPassBudget` passes and
refuses the next. `wcdoctest-clus` 270/270, 3918 assertions.

### 7.4 Gates (round 2)

Libraries: `lib_base2` = tree at `441b6de4`+peers without the round-2 files;
`lib_post2` = with them; built 10:28 / 10:30, no peer edits in between.

| gate | result |
|---|---|
| PDVD 120 events, `d25r14base2` vs `d25r14fix2` (`stm/gates/r14_compare_round2.txt`) | **238/238 hash pairs identical**, 0 diffs; 14 and 53 finish in 29 s / 32 s on both (the round-1 guard is in both libraries) |
| uBooNE `doc25r14base` vs `doc25r14fix`, 35 events | Bee zips **35/35** identical; tagger logs 34/35 — the one diff is 5384-136-6805, the bistable event documented in `repeat_check.sh` (`kine_pio_angle` 14.81 vs 109.51). `repeat_check.sh 22 4` under ASLR-on gives **2 distinct tagger states on BOTH libraries** (`rep14base2_*`, `rep14post2_*`, the same two hashes), so the diff is the known layout-dependent bistability, not this change |
| SBND nueCC48 / NCpi0, `doc25r14base` vs `doc25r14fix` | **96/96 and 38/38** archives byte-identical |
| sentinels over all three fix arms | pass-budget WARN: 0. `degenerate split skipped`: 3 (14 ×1, 53 ×2, as in round 1). `merging there instead`: **3**, all PDVD — 039252/15 cluster 107, 039349/52 cluster 24, 039349/53 cluster 50 — each with the steiner point 0.00 cm from an existing vertex and 0.31–0.37 cm from `max_point`, i.e. a near-miss of the prototype's 0.3 cm merge test. All three fall inside `dual_chain` OFF-pass windows (log timestamps), whose graph is discarded, and the outputs are byte-identical |

The redirect therefore changed no output on 222 events while firing three
times where it would have created a duplicate vertex; the WARN is armed and
silent.

### 7.5 The duplicate pair, looked at (item 3) — for the owner

`stm/gates/r13_duplicate_pair_png.py` draws, for each event, the imaged charge
near the fits (grey), the steiner cloud of the PR cluster (black), the
candidate's PR segments from the calib dump, and the three points from the
round-1 log (V = the vertex, A/B = the two far ends):

- `docs/pics/doc26_039349_14_cluster36_duplicate_pair.png`
- `docs/pics/doc26_039349_53_cluster53_duplicate_pair.png`

What they show, and what changes the round-1 reading of "charge gap":

- **039349/14, cluster 36** is one straight cosmic track from (z 5, x 392, y −83) cm
  to A (z 151, x 197, y −168) cm. The imaged charge is continuous along it. The
  **steiner cloud stops exactly at V** (x 273, z 87): 632 steiner points, all on
  the upper half, none along the 111 cm from V to A (0 within 3 cm of the V→A
  line; the 35 that are near the line all lie behind V). The vertex sits at the
  edge of the steiner cloud, not at a kink or a gap in the charge. Of the three
  segments at V, `36001` (116 cm) is a real fit that follows the charge to A;
  `36007` (115 cm) is a straight chord from V to B, 5.8 mm from A — the
  duplicate; `36008` (145 cm) is a straight chord from V to the far upper end
  that ignores the bend the steiner points make at z ≈ 40.
- **039349/53, cluster 53**: steiner points and imaged charge both run from A
  (z 40, x 343) to z ≈ 87, then nothing for ~75 cm, then a small isolated piece
  at V (z ≈ 161–165, x ≈ 210). Three chords (`53002` 181 cm, `53004` 142 cm,
  `53003` 112 cm) cross the void to V. ~~Here the void is real in the charge
  too.~~ **Wrong — see doc 27.** The "charge near the fits" in this picture is
  a coincidental cluster; the cluster's own points are 168 cm away in y (one
  face height) and continuous from z 27 to 166. The steiner cloud was re-tiled
  into the wrong face of anodes 6/7 because the point tree (v6 wires) and the
  PR job (v7-uvwfit wires) disagreed on the face order.

So the two events are different cases. On 14 the defect upstream is a steiner
cloud that covers only part of a continuous track (the boundary is not a
readout-time cut: x_V + t0·v and x_V − t0·v differ between the two clusters);
this is the same territory as doc 25 §13.4 item 8 (the Steiner terminal floor)
and doc pdvd/30's two-point fits, and it is what made both the loop and the
duplicate. On 53 the question is why an isolated piece 75 cm away is in the
same cluster and gets chords drawn to it.

To judge (answered in §8):
1. On 14: should the chord duplicate `36007` be dropped in favour of the fitted
   `36001` (the obvious merge rule: when a split is degenerate, keep the segment
   with more fit points on charge and remove the chord)? And is the missing
   steiner coverage below V a bug to chase first — it would remove the
   duplicate at the source?
2. On 53: should chords across a void with no imaged charge exist at all, or is
   the association of V's piece with this cluster the thing to fix?
3. Whether either merge rule should be a knob (default OFF, SBND gate) or a
   PDVD-only behaviour.

The Bee zips are `work/039349_14_d25r13fix/mabc-pr.zip` and
`work/039349_53_d25r13fix/mabc-pr.zip` (layers `track_fit`, `shower_track`,
`vertices` carry cluster 36's / 53's PR; not uploaded — upload is yours).

## 8. Owner decisions (2026-09-03) and the next round

The owner answered the three §7.5 questions the same day:

1. **039349/14 (steiner cloud covering half a continuous track):** improve the
   Steiner terminals. This is a **separate campaign**, not a merge rule in
   `examine_partial_identical_segments`. The chord duplicate `36007` is a
   symptom of the missing terminals below V and disappears when the cloud
   covers the whole track; dropping it by a segment-level rule would treat the
   symptom. Nothing in this thread touches it further. Starting points for that
   campaign: doc 25 §13.4 item 8 (the Steiner terminal floor), doc pdvd/30's
   two-point fits, and the measurement in §7.5 (632 points above V, 0 within
   3 cm of the 111 cm V→A line, boundary not a readout-time cut).
2. **039349/53 (isolated piece 75 cm away, chords across a charge void)** —
   *withdrawn by doc 27: the piece and the void were a stale-geometry artifact;
   the input cluster is continuous and there is nothing to separate.* The
   decision as taken: the gap is real, so the two pieces should **not** be connected. The defect is
   **over-clustering** — V's piece belongs to a different cluster (or its own),
   and the three chords `53002`/`53003`/`53004` only exist because the cluster
   boundary is wrong. The owner's framing: when clusters are passed along the
   chain we sometimes have to merge them, but then we have to separate them
   properly again, and the chain has graph-building devices for exactly this
   (over-clustering avoidance). SBND had the same class of problem early on and
   its chain was updated with proper separation.
3. **Knob vs PDVD-only:** moot for this thread, since no merge rule ships.
   Whatever the over-clustering round changes follows the standard bar
   (default-OFF knob, key suppression, SBND + uBooNE byte-identity gates) — the
   separation components are shared with SBND.

**Order of work:** the over-clustering case (item 2, 039349/53) first; the
Steiner-terminals campaign (item 1) after, as its own doc.

### 8.1 What the next round has to establish (039349/53)

Not started here; this is the scope, written down so the round opens with a
Repro block and a measurable target.

- **Which pass joined V's piece to the A→z 87 body.** PDVD's clustering runs
  `separate()` at the drift-group stage with the full SEPARATION operating
  point (`protodunevd/clus.jsonnet` around line 511: `track_recarve`,
  `far_point_x_cut` 14 cm, `far_point_mid_dis` 60 cm, `dec1_guard_main_angle`
  45, `iso_slab_split`, `collinear_*`), followed by `connect1(allow_mixed_faces,
  respect_separate_family)`, `deghost`, `examine_x_boundary`, `neutrino`,
  `isolated`. A 75 cm charge void with no collinearity argument should not
  survive `separate()`; so either an earlier merge pass produced a shape
  `separate()` does not split (a per-APA/face `connect1`, `parallel_prolong` at
  35 cm, `extend_loop`), or a later pass (`connect1` at group scope, `neutrino`,
  or the Q/L-side bundle grouping) re-joined them, or the PR-side
  `find_other_segments` chords are what associates the piece. The first task is
  the census: run the `_keep` clustering on 039349/53 with the per-pass Bee
  dump (`trace_bee`-style) and name the pass whose output first contains both
  pieces under one cluster id. Doc 96 did this for SBND (`96_over-clustering-
  and-missing-trackfit.md`: over-clustering fused in imaging vs in clustering;
  `dec2` dead for in-time clusters; `track_recarve` already splitting one case)
  and doc 97 shipped the rescue (`sep_fv_point`, now SBND production).
- **The SBND precedent to port, not re-invent.** SBND's PR job has
  `unmerge_bundle` and `unmerge_assoc` (`sbnd/clus.jsonnet` ~2003–2035; docs
  `45_unmerge-bundle-main-associated.md`, `50_stm-fit-scope-and-unmerge.md`,
  `53_unmerge-vs-cathode-crossers.md`) which undo the flash-collapse merge
  before pattern recognition and re-separate main from associated pieces. PDVD's
  `pr.jsonnet` deliberately has neither (its header, line 10: PDVD never runs
  `examine_bundles`, doc 25 §4). If the census shows V's piece is attached at
  the bundle/association stage rather than in clustering proper, the fix is an
  unmerge-type stage for PDVD (fork by duplication, default OFF); if it is in
  clustering, it is a `separate()`/`connect1` operating-point question, gated
  on SBND because the components are shared.
- **Success criterion.** On 039349/53 cluster 53: V's piece and the A→z 87
  body carry different cluster ids in the pre-PR Bee layer; the PR tail then
  produces no segment crossing the void (no `53002`/`53003`/`53004`
  equivalents), and `examine_partial_identical_segments` has no duplicate pair
  to split, so the round-1 guard does not fire on this event (its DEBUG
  "degenerate split skipped" line disappears from the log). Everything else on
  the 120-event manifest byte-identical with the knob off (`r13_loop_gate.py`
  style hash pairs), and a census of how many other `_keep` events change with
  the knob on, each looked at.
- **Inputs on disk.** `work/039349_53_keep/` (Q/L outputs + pctree),
  `work/039349_53_d25r13fix/` (PR outputs, `mabc-pr.zip`, the calib dump used
  for the §7.5 picture), pinned libs `/home/xqian/tmp/doc25r13/lib`. Picture
  script `stm/gates/r13_duplicate_pair_png.py` (CASES list) can be reused to
  draw the two pieces with their new cluster ids.


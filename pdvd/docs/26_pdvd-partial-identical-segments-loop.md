# PDVD doc 26 — the non-terminating `examine_partial_identical_segments` loop that the wire-bound crash fix exposed

**Status:** FIXED, unknobbed. NOT bit-identical on the two PDVD events that never
terminated (they now finish); byte-identical everywhere the guard does not fire
(gates in §5). Toolkit commit `441b6de4`; this document, the gate script and its
results are the wcp-porting-img record.

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
(negative wire index, PDVD 039349/32). Rerunning the 120-event `_keep` manifest
with that fix, two unrelated events stopped finishing:

| event | before the crash fix | after the crash fix | log |
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

- **Before the crash fix the two events never reached this code.** The old
  `fetch_channel_from_anode` returned an out-of-bounds read for negative wires
  (cached, so stable run to run), and those garbage channels entered the STM
  fits' 2D point associations near wire-plane edges. Fixing it changed the
  cosmic-tagger verdicts upstream of the neutrino tagger. Event 14:
  `TaggerCheckSTM` clusters 18 and 23 went STM=1 → STM=0, and FC flipped on
  clusters 28, 29, 34, 49; event 53: clusters 14 and 19 STM=1 → 0, FC flipped on
  29, 35, 36. Different bundles therefore became neutrino candidates
  (`selected main cluster` 23/18/45 before vs 48/36 after on event 14), and
  cluster 36 (event 14) / cluster 53 (event 53) entered `find_proto_vertex`
  for the first time. The crash fix did not create the loop; it removed the
  accident that kept these clusters away from it.
- **The geometry is PDVD-specific.** The trigger is two near-duplicate
  segments leaving a vertex into a region with no steiner points — a track
  crossing a charge gap (CRP boundary, readout edge, dead region), which
  `steiner_graph_gap` bridges with long edges. SBND and uBooNE steiner clouds
  do not have 9 cm holes along tracks.
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
= tree-with-guard. Hence the clean pairing below.

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

## 6. Left open

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

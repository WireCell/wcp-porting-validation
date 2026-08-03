# Doc 54 — TGM/STM tagger performance, rounds 1–2

Goal: reduce the running cost (wall + memory) of the PR tagger stage
(`tagger_check_tgm` / `tagger_check_stm`) on the standard 30-event MCP2025C
scan, with **zero** change to any output — the same bar as any perf round:
member-content-hash identical archives, identical TSVs, identical
tracking-ROOT branch content.

**Result: TaggerCheckSTM 29.73 s → 17.49 s (−41%) over the 30 events (1.7×),
every other pipeline step within ±1%, peak RSS unchanged.  Gate 120/120
identical.**  Toolkit commit `077a48d1`
(`clus: wire-index dQ_dx_fit's response fill ...`).

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# arm 1: baseline binary (toolkit HEAD 11bec7f4), 30 events, sequential,
# per-event wall/RSS via abtest/timecmd.py, setarch x86_64 -R
./scripts/runners/run_perf54_nusel.sh base            # -> work-{mcp10,mcp1000,mcp1000b}-p54base

# apply toolkit commit 077a48d1, then wcbuild (build + install; freshness
# proof: local/lib/libWireCellClus.so mtime > source edit), wcdoctest-clus
./scripts/runners/run_perf54_nusel.sh opt             # -> work-*-p54opt

python3 scripts/perf/p54_ab_report.py              # gate + timing tables
```

The runner reruns ONLY the nusel (PR tagger) stage: imaging `evt*` and Q/L
`ql_evt*` products are symlinked from `work-*-d55ton` (M11/M13 — never
regenerated), and the tagger flag set is the d55ton arm's exactly (doc 53:
the d52 NUF set + `-unmerge-assoc`, i.e. chord/rescue/rescue-chord,
main-pair-real, fvz 5 / fvzi 3 / fvx 2.5 / fvy 3, `-stm-fit`, `-mip 56000`,
unmerge real).  Both arms run under `setarch x86_64 -R` (this chain is
ASLR-non-deterministic at ±7 STM tags, doc 49 §4a).

Harness sanity: the 30 `p54base` TSVs are byte-identical to the `d55ton`
production ones (same binary, same inputs) — 30/30.

## Baseline: where the PR job spends its time

Summed `MABC timing` lines over all 30 events (work-*-p54base logs):

| step | total s | share of MABC |
|---|---:|---|
| CreateSteinerGraph:pr | 42.5 | 48% (not a tagger — see "next levers") |
| **TaggerCheckSTM:pr** | **29.7** | **34%** |
| TaggerCheckTGM:pr | 4.5 | 5% |
| SbndMagnifyTrackingVisitor:pr | 4.2 | |
| TaggerCheckFC:pr | 1.5 | |
| everything else (load/save/switch_scope/unmerge/bee) | ~6 | |

Whole-job wall (config compile + tensor IO + python table included):
164 s / 30 events; peak RSS 626 MB (evt 285185).

## Profile evidence (CPU, gperftools, evt 287517 = heaviest STM event)

`profile_pr.sh` (scratchpad; mirrors the runner's wire-cell call,
wcsonnet-precompiled config per M17, outputs to scratch).  NB: the profiled
process must NOT run under `setarch x86_64 -R` — the SIGPROF-based profiler
dies at startup with "Profiling timer expired".  Profiles ran without
setarch; the gates ran with it.

Baseline (1579 samples total):

```
TaggerCheckSTM::visit                 810  51.3%   (all in check_stm_conditions)
  WireCell::Clus::TrackFitting::do_single_tracking   782  49.5%
    WireCell::Clus::TrackFitting::dQ_dx_fit          673  42.6%
      std::_Rb_tree_increment (flat)                 202  12.8%
CreateSteinerGraph::visit             469  29.7%
TaggerCheckTGM::visit                  41   2.6%
```

Line-level (`--list=TrackFitting::dQ_dx_fit`): the samples concentrate in
the response-matrix fill at TrackFitting.cxx:7330–7400 — for **every**
trajectory point the code iterated the **entire** `map_U/V/W_charge_2D`
(std::map) and each entry's `Coord2D_set`, discarding almost everything on
`plane/apa/face` mismatch and a wire/time window test:
O(n_3D_pos × n_2D) with rb-tree iteration as the leading term.

## The change (toolkit `077a48d1`, clus/src/TrackFitting.cxx)

`dQ_dx_multi_fit` already solved the same problem with a wire-indexed
lookup (`wire_idx_U/V/W`: (apa,face) → wire → rows).  This round gives
`dQ_dx_fit` the same index, with one deliberate difference kept for
byte-identity:

- multi_fit enumerates wires as `round(center) ± ceil(search_range)` and
  applies no further wire test — a window that is not literally the original
  predicate.  Here the index is only used to enumerate a
  `floor(center−sr) … ceil(center+sr)` **superset**, and the **original
  predicate is re-applied verbatim** on each candidate
  (`abs(wire − centers.front()) <= search_range &&
  abs(time − centers_T.front()) <= search_range * cur_ntime_ticks`), so the
  accepted (row, value) set is provably the original one.
- Row visiting order can differ from the map order.  That cannot leak into
  the output: `Eigen::SparseMatrix::insert` keeps each column's storage
  row-sorted whatever the insertion order, so the assembled RU/RV/RW are
  bit-identical; `reg_flag_*` are idempotent flag sets.

No knob: the transformation is an identity on outputs (and gate-verified
below), so it ships default-on like every pure perf change.

## Results (30 events, sequential, setarch x86_64 -R)

| step | base s | opt s | change |
|---|---:|---:|---:|
| **TaggerCheckSTM:pr** | **29.73** | **17.49** | **−41%** |
| CreateSteinerGraph:pr | 42.50 | 42.10 | −1% |
| TaggerCheckTGM:pr | 4.47 | 4.44 | −1% |
| SbndMagnifyTrackingVisitor:pr | 4.18 | 4.13 | −1% |
| TaggerCheckFC:pr | 1.46 | 1.46 | +1% |

Heaviest events (STM s): 287517 3.49→1.68, 289849 2.98→1.63,
284657 2.45→1.44, 288859 2.19→1.23, 285185 2.08→1.08, 287825 1.47→0.85.

Whole-job wall 164 s → 154 s (−6%; the job is dominated by steiner +
config compile + IO).  Re-profile of evt 287517 on the optimized binary:
event total 1579 → 1121 samples (−29%), `dQ_dx_fit` 673 → 193 (42.6% →
17.2%), `TaggerCheckSTM::visit` 810 → 325 (51.3% → 29.0%).

## Memory

Peak RSS is **unchanged** (626 MB base / 627 MB opt; per-event values agree
within ±2 MB) — expected, since the change only replaces a scan with an
index of the same data.  Heap profile of the baseline binary
(evt 285185, tcmalloc `HEAPPROFILE`, in-use at the 238 MB dump):

```
TensorDM::as_pctree                89.0 MB  37%   (pctree deserialization)
PointCloud::Dataset::slice/add     85+17 MB 35%   (point-cloud arrays)
Aux::sample_live                   25.0 MB  10%
```

The PR job's memory is set by the loaded point-cloud tree and its sampled
point clouds, not by the taggers (the fit's transient Eigen/charge-map state
is small and freed per pass).  No memory lever inside TGM/STM worth taking
this round; RSS work would have to target the pctree load/steiner side.

## Gate (byte-identicality)

`scripts/perf/p54_ab_report.py` — labels `work-{mcp10,mcp1000,mcp1000b}-p54base` vs
`-p54opt`, full report at `/home/xqian/tmp/p54_ab_report.txt`:

- per event: `mabc-pr.zip` and `pctree-pr-evt*.tar.gz` member-content hash
  (`abtest/hash_archive.py`, never `cmp` on archives — M2), `nusel-evt*.tsv`
  as text, `tracking-stm.root` compared **numerically** (uproot, bitwise
  per branch — ROOT files embed timestamps, so byte compare is void).
- **GATE: 120/120 comparisons identical → PASS.**
- `wcdoctest-clus`: 565/565 assertions pass.

---

# Round 2 — the Steiner-graph stage (toolkit `6c7192b1`)

Round 1 left `CreateSteinerGraph:pr` as the top step (42.1 s / 30 events,
44% of the optimized evt-287517 profile).  Under it:
`ImproveCluster_2::mutate` 68%, and as leaf costs the graph builders
(`make_graph_closely/ctpc_pid` ~200 samples), `get_two_boundary_wcps` 89,
`get_activity_improved` 82, blob re-sampling (`sample_live`+`sample_blob`)
105, `estimate_total_charge` → `get_wire_charge` 56/48.

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
# baseline = the round-1 products already on disk (work-*-p54opt)
# apply toolkit 6c7192b1, wcbuild, freshness proof, wcdoctest-clus
./scripts/runners/run_perf54_nusel.sh opt p55         # -> work-*-p55opt
python3 scripts/perf/p54_ab_report.py --base-tag p54opt --opt-tag p55opt
```

## The change

`Cluster::get_two_boundary_wcps` re-ran `Blob::estimate_total_charge()` for
**every point** of every blob — a pure per-blob function that walks all the
blob's wires through the grouping's hash caches.  And inside it,
`get_wire_charge()` repeated the `cache()/find(time_slice)` chain per wire.
Both fixed byte-identically (toolkit `6c7192b1`):

- per-blob memo of the charge estimate inside `get_two_boundary_wcps`
  (pointer-keyed `unordered_map`, keyed lookups only, never iterated —
  determinism-safe);
- `estimate_total_charge` fetches the charge row once per
  (apa, face, plane, time_slice) via the new read-only
  `Grouping::wire_charge_row()` and looks wires up in it (missing row ==
  `get_wire_charge()`'s `{0, 1e12}` for every wire: no positive charge).

`estimate_total_charge`'s only caller is the `get_two_boundary_wcps` loop,
so the memo covers every consumer — including STM's and FC's
`cluster_fc_check` and TGM, which call `get_two_boundary_wcps` on the same
clusters.

## Results (30 events, sequential, setarch x86_64 -R)

| step | p54base | p54opt (r1) | p55opt (r2) | r2 change |
|---|---:|---:|---:|---:|
| CreateSteinerGraph:pr | 42.50 | 42.10 | **36.88** | **−12%** |
| TaggerCheckSTM:pr | 29.73 | 17.49 | **16.72** | −4% |
| TaggerCheckFC:pr | 1.46 | 1.46 | **0.44** | **−70%** |
| TaggerCheckTGM:pr | 4.47 | 4.44 | 4.41 | −1% |

Whole-job wall 154 → 146 s (164 s at the round-0 baseline: **−11%
combined**); peak RSS 625 MB (unchanged, as expected).  Profile on
evt 287517 (round-2 binary): event total 1121 → 1040 samples;
`get_two_boundary_wcps` 89 → 41, `estimate_total_charge` 56 → 9,
`cluster_fc_check` 27 → 11.

## Gate

`scripts/perf/p54_ab_report.py --base-tag p54opt --opt-tag p55opt` — full report at
`/home/xqian/tmp/p55_ab_report.txt`:
**GATE: 120/120 comparisons identical → PASS** (same scope as round 1).
`wcdoctest-clus` 565/565.

# Next levers (not taken)

1. Remaining `CreateSteinerGraph` cost (36.9 s): the graph builders
   (`connect_graph_closely_pid` — already vector-bucket-optimized in an
   earlier campaign, residue spread across rb-tree range scans) and the
   retile re-sampling (`Aux::sample_live`/`sample_blob`, ~105 samples) —
   central Aux code shared with imaging, high blast radius for a perf-only
   change.
2. `ImproveCluster_1::get_activity_improved` (82 samples): kd2d knn probes
   per dead/good channel and per-wire set inserts; a contained but fiddly
   target.
3. Remaining `do_single_tracking` cost outside `dQ_dx_fit`
   (`trajectory_fit`, rough-path/graph work).  Second-order.
4. `TaggerCheckTGM` is 4.4 s / 30 events; its nucand path already uses the
   indexed `dQ_dx_multi_fit`.  Not worth a round.
5. Whole-job overhead (config compile, tensor IO, per-event process spawn)
   is ~half the wall — a runner/workflow change, not a code round.

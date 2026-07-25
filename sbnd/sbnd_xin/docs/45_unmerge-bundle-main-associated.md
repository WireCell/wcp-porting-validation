# 45 — Restoring the prototype "main cluster + associated clusters" data product

`ClusteringUnmergeBundle` (new, default OFF), SBND MC 10-event sample.

**Read the Root cause section before the Fix section.** The reported symptom
(`showcase-stmfit-mc-evt18/track_com_18.root`, cluster id 80) turns out to
have **two independent causes**, and the fix in this document addresses only
one of them.  **The evt18 cluster 80 fit is unchanged by this fix.**

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# doc-39 op point + the STM fit dump; the two roots share the Q/L pctrees of
# work-mcsim-stmon by symlink (ql_evt*/ and evt*/), so only the PR tail reruns.
F="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit"
SBND_WORK_ROOT=$PWD/work-mcsim-unmoff SBND_MAX_JOBS=4 ./run_nusel_evt.sh mc all $F
SBND_WORK_ROOT=$PWD/work-mcsim-unmon  SBND_MAX_JOBS=4 ./run_nusel_evt.sh mc all $F -unmerge

# component-mode probe (single event, NOT a label set):
SBND_WORK_ROOT=$PWD/work-mcsim-unmcomp ./run_nusel_evt.sh mc 6 $F -unmerge-comp
```

Labels: `work-mcsim-unmoff` (knob off) / `work-mcsim-unmon` (knob on) /
`work-mcsim-unmcomp` (evt18 component-mode probe only).

## Symptom

`showcase-stmfit-mc-evt18/track_com_18.root`, block 80 (= STM cluster id 8,
pass 0; the Magnify branch is `cid*10+pass`): the STM tagger fits an 83.3 cm
"track" through 142 fit points.  In the Magnify GUI the underlying cluster is
visibly not one track but a handful of small, widely separated clumps.

## Root cause

There are two, and only the first is a Q/L artifact.

### (a) The post-Q/L tree has no main/associated structure — real, general

uBooNE hands the PR chain one **main cluster** plus a vector of **additional
clusters** per matched flash bundle
(`prototype_base/pid/apps/wire-cell-prod-stm.cxx:815-855`):

```cpp
temp_clusters = map_parentid_clusters[it->second];
main_cluster  = the member whose get_cluster_id() == parent id;
additional_clusters = every other member;
main_cluster->create_steiner_graph(...);            // main only
fid->check_stm(main_cluster, additional_clusters, ...);
```

`check_stm()` fits the main as one track and only *counts* the companions
(`check_other_clusters()`).

On SBND the Q/L stage runs
`clustering_examine_bundles(use_flash_t0=true, flags_from_longest=true)`,
which **merges every flash-time-coincident member of a bundle into one
cluster** carrying `flag_main_cluster`.  Consequences in the PR tail:

* `TaggerCheckSTM` fits that composite as a single track;
* `check_other_clusters()` has no companions left to count (inert);
* `TaggerCheckNeutrino`'s "companions sharing `matched_flash_gid`" is empty.

`TaggerCheckTGM` was adapted *internally* instead (`main_component_pairs`,
docs 36/38), which works for a pair-selection veto but does not generalise to
anything that fits or walks the main.

Measured on the 10-event MC sample (`-unmerge` log lines): 23 mains carry
co-merged members, the largest being evt41 cluster 22 at **1187 blobs →
559 main + 628 in one merged-in companion**.

### (b) evt18 cluster 8 is not a flash merge at all

For the specific cluster in the report the flash merge is **not** implicated:

```
work-mcsim-unmoff/nusel-table.tsv, evt18 main_id 8:
  n_bundle = 1, npts_main = npts_bundle = 77, len_main_cm = 84.2, STM
mabc-pr.zip clustering layer: cluster 8 real_cluster_id set = {8}
```

`n_bundle = 1` and a single `real_cluster_id` mean this main was never
flash-merged with anything — so `ClusteringUnmergeBundle` in `real` mode
correctly leaves it alone, and the fit is bit-for-bit the same off and on
(142 points, 83.3 cm, same endpoints).

Its 84 cm span comes from the **clustering chain**, which joined four small,
detached clumps (single-linkage over the Bee points of cluster 8):

| clump | points | own extent | centroid (x,y,z) cm |
|---|---|---|---|
| A | 54 | 9.6 cm | (−154, 196, 263) |
| B | 13 | 1.1 cm | (−120, 172, 232) |
| C | 7 | 1.0 cm | (−110, 143, 220) |
| D | 3 | 0.6 cm | (−126, 161, 232) |

Gaps A→B ≈ 52 cm, B→C ≈ 32 cm; total end-to-end 84.2 cm.  The all-APA merge
passes (`extend` 60 cm, `regular` 60/30 cm, `parallel_prolong` 35 cm — all
gated on `use_flash_t0`) merged them, and the STM fit then bridged the gaps.

The flash-time gate can only **suppress** merges, so the prototype's ungated
passes would have merged these clumps too.  Whether the prototype's
`create_steiner_graph` + `do_tracking` would likewise bridge 30–50 cm gaps is
**an open question left to the owner** — see §Open.

## Fix

New `clus/src/ClusteringUnmergeBundle.cxx` — a fork of
`ClusteringRecoveringBundle` (fork-by-duplication; the uBooNE component is
untouched).  For each in-scope `flag_main_cluster` cluster it splits the blobs
into the pre-merge **main** (retained: keeps its ident, `flag_main_cluster`
and the cluster scalars incl. `matched_flash_gid` / `cluster_t0` / `flash`)
and one new **`flag_associated_cluster`** cluster per other member.  That
restores the prototype layout in the *data product*, so `TaggerCheckSTM`,
`check_other_clusters()` and `TaggerCheckNeutrino` all get it without each
re-deriving it.

Two modes (`unmerge_bundle_mode`, C++ default `"real"`):

* **`real`** (exact) — the per-blob `real_cluster_main` / `real_cluster_id`
  provenance written by `merge_clusters()` and persisted through the pctree
  by `save_real_cluster_id` (doc 38).  Blobs with `real_cluster_main != 0`
  come from the merge's *representative* member — the same member whose flags
  (including `flag_main_cluster`) and flash the merged cluster carries — so
  "the main" is by construction the cluster the bundle verdicts describe.  The
  rest are grouped by their pre-merge `real_cluster_id`.  A cluster whose
  provenance says it was **never merged is left alone** (not a fallback case:
  splitting it would undo the clustering chain's own deliberate long-range
  merges, which the prototype keeps inside one `PR3DCluster`).
* **`component`** (proxy, **opt-in only**) — relaxed-graph connected
  components, longest = main.  Mode `real` deliberately does **not** fall back
  to this: undoing the flash merge is bookkeeping, but splitting on graph
  connectivity is a clustering decision (the relaxed graph does not join the
  two halves of a cathode crosser).  A cluster with no provenance is therefore
  left **unsplit with a warning**, so running `-unmerge` against a pctree
  saved before `-save-rcid` is a no-op rather than a silent re-clustering.
  Ask for the proxy explicitly with `-unmerge-comp`.

  Runner caveat: `run_nusel_evt.sh` only appends `-save-rcid` to a Q/L step it
  *launches*; an existing `ql_evt*/pctree-*.tar.gz` is reused as-is.  Point
  `SBND_WORK_ROOT` at a root whose pctrees came from a `-save-rcid` run (the
  doc-42 `work-mcsim-stmon` trees do), or `-unmerge` will warn and do nothing.

Placement: **between `switch_scope` and `steiner`**, enforced by the runner
flag.  `separate()` does not carry node-local point clouds, so the split must
precede `steiner_pc` creation — otherwise the *retained* main would keep a
`steiner_pc` built from the pre-split blob set and the fit would silently use
it.  The visitor erases such a stale `steiner_pc` and logs a warning.

Implementation notes worth keeping:

* `separate(remove=false)` leaves the retained cluster holding **full-length**
  `perblob` arrays over a now-shorter blob list.  Both the retained cluster
  and every part get their `perblob` dataset re-carved by row
  (`Dataset::subset`), so `real_cluster_id` (Bee colors) and
  `real_cluster_main` (TGM `main_component_mode="real"`) stay parallel to blob
  order.  This is *not* the `switch_scope` pattern — that one uses
  `remove=true` and has no retained cluster.
* `from()` copies flags, so every part inherits `flag_main_cluster`; it is
  cleared explicitly, otherwise STM/TGM/FC would evaluate every fragment as a
  bundle main.
* Part idents are `main_ident*100 + sub`, with collision avoidance against
  every ident already in the grouping.
* Blob conservation is asserted per split (`retained + parts == nb`); the
  logs show no `BLOB LOSS` lines on the whole sample.

Files: `clus/src/ClusteringUnmergeBundle.cxx`,
`cfg/pgrapher/common/clus.jsonnet` (`unmerge_bundle()` builder),
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` (`cm_by_name.unmerge_bundle` +
`unmerge_bundle_mode` arg), `sbnd_xin/wct-pr-perevt.jsonnet`,
`sbnd_xin/run_nusel_evt.sh` (`-unmerge` / `-unmerge-comp` /
`-no-unmerge`, `SBND_UNMERGE` / `SBND_UNMERGE_MODE`).

## Verification

**Compiled config, knob off — byte-identical.**  `wcsonnet` on
`wct-pr-perevt.jsonnet` with the production pipeline
(`switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc`),
new tree vs a `git archive HEAD` baseline tree: identical after JSON
normalisation **and** `cmp`-identical raw bytes.  The visitor is absent from
the compiled config unless named in `pipeline_names`, so "off" is a no-op by
construction — the byte gate is necessary, not sufficient, hence the verdict
diff below.

**Compiled config, knob on.**  `ClusteringUnmergeBundle:pr` appears with
`{mode: "real", graph_name: "relaxed", pcarray_name: "perblob",
require_in_scope: true}` and the MABC pipeline reads
`[ClusteringSwitchScope:pr, ClusteringUnmergeBundle:pr, CreateSteinerGraph:pr,
MakeFiducialUtils:pr, TaggerCheckTGM:pr, TaggerCheckSTM:pr, TaggerCheckFC:pr]`.

**Unit tests.** `./build/clus/wcdoctest-clus` — 41/41 cases, 518 assertions,
`rc=0`.

**Coverage, knob on.** 23 mains split across the 10 events, **100 % via the
exact provenance** (0 proxy fallbacks); every split conserves blobs.

**Verdict diff, `work-mcsim-unmoff` vs `work-mcsim-unmon`, 128 bundles.**
Nine rows differ.  Four of those are a **pre-existing log-writing race**, not
behavior: the per-cluster verdict line is truncated by an interleaved
`MultiAlgBlobClustering` timing line (e.g. evt41 ON:
`visit: TaggerCheckSTM: ciAlgBlobClustering.cxx:2082`), so the extractor
records the `-1` "not evaluated" sentinel.  Those are evt2 gid1000000 `fc
-1→0`, evt31 gid1000002 `stm -1→0`, evt41 gid8 `stm 0→-1`, evt9 gid11 `stm
0→-1` — verified present in both logs by hand.

Real changes:

| event | main | change | note |
|---|---|---|---|
| 2  | 11 | FC 0 → 1 | main was `{rcid 9: 1347 pts} + {rcid 4: 32 pts}`; the 32-pt speck was pushing the extremes out of the FV |
| 9  | 11 | FC 0 → 1 | main was `{rcid 4: 405} + {rcid 8: 38}` |
| 9  | 10 | FC 0 → 1 | |
| 11 | 22 | FC 1 → 0 | the only FC loss |
| 31 |  8 | TGM 0 → 1, label nu-candidate → **TGM** | **needs a hand scan** — 34 pts, 1.2 cm; not characterised here as an improvement |

Totals: TGM 35 → 36, FC 26 → 29, STM tagged 2 → 2 (no STM verdict flipped).

**STM fits actually change.**  Comparing the `stm_fit` Bee layer of
`mabc-pr.zip` per event, two long fits over merged composites **disappear**
(their mains are now fully contained, so `check_stm_conditions` returns at the
FC gate):

| event | cluster | off | on |
|---|---|---|---|
| 2 | 11 | 378 fit pts, 153.9 cm, (−27, 200, 51) → (−46, 63, 119) | no fit |
| 9 | 11 | 515 fit pts, 267.6 cm, (−197, 195, 24) → (−113, 25, 214) | no fit |

Every other event's fit blocks are identical, **including evt18** (blocks 80
and 150 unchanged: 142 pts / 83.3 cm and 346 pts / 215.9 cm).

**`tagger_check_neutrino` exercised, companion list now non-empty.**  The
production `PIPELINE` in `run_nusel_evt.sh` stops at `tagger_check_fc`, so the
neutrino tagger was run by hand with `pipeline_names` extended
(`beam_window_us=[0.2,2.2]`, `dl_weights=''`).  It completes in both modes.
On evt9, whose in-beam main *is* one of the split mains:

```
off: TaggerCheckNeutrino: selected main cluster 11 (t0 0.656 us, L 31.8 cm, 0 associated)
on : TaggerCheckNeutrino: selected main cluster 11 (t0 0.656 us, L 26.6 cm, 1 associated)
```

That is the prototype layout arriving at the PR code: a shorter main plus a
companion.  On evt18 (whose in-beam mains 4 and 9 were not split) the tagger
is unchanged, `0 associated` both ways, as expected.

**`check_other_clusters()` still returns false everywhere.**  Trace-level runs
over all 10 events in both modes: 5 (off) / 3 (on) mains reach the companion
check at all (the rest exit at the FC gate), and `flag_other_clusters=false`
in every case.  The companions this sample produces are below the function's
`length > 5 cm` cut.  So the structural fix is in place, but on **this**
sample its observable effect is confined to the fit *inputs* and the FC/TGM
consequences above — companion counting is not yet demonstrated live.

## Open — for the owner

1. **evt18 cluster 8 (the reported case) is untouched.**  Fixing it means
   deciding what a "main cluster" is when the clustering chain has merged
   detached clumps across 30–50 cm gaps.  `-unmerge-comp` demonstrates the
   effect on evt18 (`work-mcsim-unmcomp`): cluster 8 goes `28 blobs → main 11
   + 4 associated`, i.e. the STM fit would run on the 54-point / 10 cm clump
   alone.  But relaxed-graph connectivity is known to **break cathode
   crossers**, so shipping `component` as the operating mode is a clustering
   decision, not a bookkeeping one, and is deliberately left as an opt-in
   probe.  A 30 cm-step *path* linkage (the notion `TaggerCheckTGM`'s
   `main_component_pairs` already uses, where a cathode crosser is one
   component, doc 36) is the obvious middle ground — **not implemented**, on
   purpose, because it changes clustering-level behavior.
2. **The alternative reading** is that the fit, not the cluster, is at fault:
   the toolkit's steiner/tracking bridges 30–50 cm charge-free gaps.  Whether
   the prototype does the same has not been checked against
   `prototype_base/`.  Documented per §5.4 rather than picked.
3. **evt31 main 8 nu-candidate → TGM** needs a hand scan before this knob is
   turned on in any label campaign.

## Notes on the labels

* `work-mcsim-unmon` was **re-run** after a defect in the first ON pass: the
  `real` mode fell through to the component proxy for clusters that were never
  flash-merged, so that run's "8/9 proxy" counts are superseded by the 100 %
  exact counts above.
* The `work-mcsim-unmoff` / `work-mcsim-unmon` label pair predates the
  no-fallback change below (`Prov::unusable` → skip instead of proxy).  It is
  unaffected: every cluster in that sample carried the provenance, so zero
  fallbacks were taken either way.
* `work-mcsim-unmcomp` holds **evt18 only** — a probe for §Open item 1, not a
  label set.
* All three roots symlink `ql_evt*/` and `evt*/` from `work-mcsim-stmon`;
  nothing under that doc-42/44 record was written to.

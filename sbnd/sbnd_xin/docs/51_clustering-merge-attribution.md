# Doc 51 — Which clustering pass merged the isolated clump into the main trunk?

**`clustering_isolated`, the last pass of the per-APA tail.** Every gap-jumping
pass that exists to bridge inefficiency declined these two pairs; the pass that
fused them is the one whose job is to put main + associated clusters in the same
bucket.

**And the associated-cluster information really is dropped.** `clustering_isolated`
does record which member each blob came from — then the very next pass,
`examine_bundles`, throws that partition away because the two ends disagree on
how the main is marked, and `switch_scope` drops the array outright at the
all-APA stage. §4 names the lines.

**Status:** investigation + a default-OFF diagnostic knob (`trace_bee`). No
reconstruction behaviour is changed. The remedies in §7 are not implemented.

## Repro

```bash
cd sbnd_xin

# 1. per-step Bee layers (default OFF; fresh work root, imaging symlinked in)
mkdir -p work-mcp10-trace51
for e in 285185 284657; do ln -sfn $PWD/work-mcp10-d49son/evt$e work-mcp10-trace51/evt$e; done
for i in 2 3; do                                   # idx 2 = 284657, 3 = 285185
  SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-trace51 setarch x86_64 -R \
    ./run_ql_evt.sh data $i -trace-bee -save-pctree -save-rcid
done

# 2. attribute the merge
python3 stm_merge_attribution.py --pr-root work-mcp10-d49son work-mcp10-trace51 285185 21
python3 stm_merge_attribution.py --pr-root work-mcp10-d49son work-mcp10-trace51 284657 27
```

Follows doc 50, which measured the *consequence*: 26 % of STM fits stray >5 cm
from any charge of their own cluster; 12 of 44 tags end in a detached clump.

---

## 1. Method

`MultiAlgBlobClustering` already supports a per-visitor Bee dump: a
`bee_points_sets` entry carrying `visitor: <type:name>` is written right after
that step runs (`MultiAlgBlobClustering.cxx:2270`). New jsonnet knob `trace_bee`
(`cfg/.../sbnd/clus.jsonnet` `trace_sets()`, threaded through
`per_apa()`/`all_apa()` and the runner's `-trace-bee`) emits one layer per step,
named `tr<NN>_<Type>` — 16 per per-APA stage, 9 for all-APA.

Two traps the tool handles, both verified rather than assumed:

- **Cluster ids are meaningless across layers.** `cluster_id_order: 'tree'`
  renumbers after *every* step, so the pieces are identified by point
  coordinates from the final post-un-merge geometry.
- **The per-APA layers are in a different scope.** They are raw; every all-APA
  layer from `switch_scope` onward carries the T0 drift correction. That offset
  is not small — evt284657's bundle sits at t0 = −841.412 µs — and the tool
  recovers **dx = +131.51 cm**, which is 841.412 µs × 1.563 mm/µs = 131.51 cm to
  four digits. That agreement is the check that cross-scope matching is sound.

**Reproduction check.** This trace run is a *separate* Q/L job with extra flags,
so it has to be shown to reproduce the clustering being attributed: its final
all-APA cluster is **1211 pts (evt285185)** and **337 pts (evt284657)**, matching
the `d49son` round's pre-un-merge numbers in doc 50 §Q2 exactly.

Match verdicts use a 20 %-majority rule, so a few stray nearest-neighbour hits
from an adjacent cluster cannot fake a merge.

## 2. Result — the same pass on both events

| per-APA apa1 step | evt285185 body / clump | evt284657 body / clump | joined |
|---|---|---|---|
| tr00 ClusteringPointed | 31 / 16 | 54,57 / 56 | no |
| tr01 ClusteringLiveDead | 31 / 16 | 54,57 / 56 | no |
| tr02 ClusteringExtend | 31 / 16 | 54,57 / 56 | no |
| tr03 ClusteringRegular (60 cm) | 27 / 14 | 50,53 / 52 | no |
| tr04 ClusteringRegular (30 cm) | 27 / 14 | 50,53 / 52 | no |
| tr05 ClusteringParallelProlong | 27 / 14 | 50,53 / 52 | no |
| tr06 ClusteringClose | 25 / 12 | 46,54 / 45 | no |
| tr07 ClusteringExtendLoop | 23 / 12 | 46,54 / 45 | no |
| tr08 ClusteringSeparate | 23 / 12 | 46,54 / 45 | no |
| tr09 ClusteringConnect1 | 19 / 10 | 40,48 / 39 | no |
| tr10 ClusteringDeghost | 16 / 8 | 36,42 / 35 | no |
| tr11 ClusteringExamineXBoundary | 23 / 7 | 42,44 / 28 | no |
| tr12 ClusteringProtectOverclustering | 21 / 7 | 42,44 / 28 | no |
| tr13 ClusteringNeutrino | 21 / 7 | 46 / 28 | no |
| **tr14 ClusteringIsolated** | **7 / 7** | **10 / 10** | **YES** |
| tr15 ClusteringExamineBundles | 7 / 7 | 10 / 10 | yes |

Exact point counts across that one step:

| | body at tr13 | clump at tr13 | joined at tr14 | clusters in the APA |
|---|---|---|---|---|
| evt285185 | 1164 | 8 | **1199** | 25 → **9** |
| evt284657 | 279 | 21 | **300** | 46 → **11** |

evt284657 is exact: 279 + 21 = 300. evt285185 gains 27 further points — the same
pass absorbed other small clusters into the same trunk. And it collapses
**25 → 9** and **46 → 11** clusters in a single pass: this one step, not the
generic merge passes, is the dominant source of the composite mains doc 50
counted.

Everything after is inherited. The all-APA stage sees them already joined (its
pre-pipeline `img` layer, in raw coords, already shows one cluster) and never
separates them.

Also visible: evt284657's *body* is itself two clusters (54,57) until
`ClusteringNeutrino` (tr13) fuses them. That merge is faithful — the prototype's
`ToyClustering_neutrino.h:1278` really does `live_clusters.erase(...)` — and it
joined two pieces of the same track, so it is not the problem here.

## 3. Why `clustering_isolated` fired

`clus/src/clustering_isolated.cxx`:

1. **Classification** (`:158-232`): "small" = largest U/V/W-wire or time-slice
   extent `< range_cut` (150) **and** `get_length() < length_cut` (SBND 15 cm,
   `docs/overclustering-evt11-gamma.md`). Both clumps are 8 and 21 blob points
   spanning 1.3 and 1.9 cm — small by a wide margin.
2. **Merge** (`:249-278`): for each small cluster, find the **nearest** big
   cluster and merge if the closest-point distance is `< small_big_dis_cut = 80
   cm`. **No direction test, no gap-quality test, no dead-region test** —
   nearest-and-under-80-cm is the whole criterion. Our gaps are 18.3 and 20.3 cm.

I did not confirm which branch each pair took: if a body classified "small" too
(evt284657's spans only 22.4 cm, near the 15 cm boundary) the small–small 50 cm
branch (`:302-330`) applies instead. Both branches are equally distance-only, so
the conclusion is unaffected.

The cuts are a faithful port — the prototype has the identical
`small_big_dis_cut = 80 cm`, `small_small_dis_cut` 5 then 50 cm,
`big_dis_cut = 3 cm`, `big_dis_range_cut = 16 cm`, `range_cut = 150`,
`length_cut = 20 cm`
(`2dtoy/src/ToyClustering_isolated.h:85,113,145,164,165,15,16`). **The criteria
are not the divergence.**

## 4. The divergence, and where the associated-cluster information dies

### 4a. The prototype groups; the toolkit merges

`Clustering_isolated` (`2dtoy/src/ToyClustering_isolated.h:292-325`) builds the
same member sets, then:

```cpp
map_cluster_cluster_vec results;                   // main -> [(associated, dis)]
for (auto it = merge_clusters.begin(); ...) {      // local variable name only
    max_cluster = the longest member;              // the MAIN
    results[max_cluster] = {};
    for (each other member) results[max_cluster].push_back({temp_cluster, dis});
}
return results;                                    // live_clusters UNTOUCHED
```

Nothing is merged, nothing deleted, and the gap distance to the main is kept.
The grouping travels as a per-cluster `parent_cluster_id` branch, and the STM app
rebuilds **separate cluster objects**: `wire-cell-prod-stm.cxx:517-527` fills
`map_parentid_clusters`; `:816-828` picks `main_cluster` = the member whose
`get_cluster_id()` equals the parent id, `additional_clusters` = the rest, then
calls `create_steiner_graph` on the **main only**.

The toolkit instead calls, at `clustering_isolated.cxx:507`:

```cpp
merge_clusters(g, live_grouping, "isolated");
```

one cluster object per group, with the member partition recorded per blob in
`("isolated","perblob")`.

### 4b. The partition is recorded — then discarded one pass later

This is the part that answers "the information was dropped somehow", and it is a
concrete producer/consumer mismatch, not a vague loss:

| # | where | what happens to `("isolated","perblob")` |
|---|---|---|
| 1 | `clustering_isolated.cxx:507` → `ClusteringFuncs.cxx:205-208` | written as a per-member index **0, 1, 2, …** — the main is **not** marked |
| 2 | `clustering_examine_bundles.cxx:167` | reads it back as `old_cc_array` |
| 3 | `:177-181` | `has_main = (find(old_cc_array, -1) != end())` → **false**, because step 1 never writes −1 |
| 4 | `:206-240` | so `flag_largest = true`: the array is **recomputed from scratch** as relaxed-graph connectivity (`connected_blobs`, `graph_name = "relaxed"`), longest component marked −1 |
| 5 | `:243` | `put_pcarray(b2groupid, "isolated", "perblob")` — the isolated partition is **overwritten** |
| 6 | all-APA `clustering_switch_scope.cxx:88-95` | carries forward **only** `real_cluster_id` / `real_cluster_main`; the `isolated` array is dropped |
| 7 | all-APA `examine_bundles` | writes a fresh connectivity partition of the flash-merged cluster |

The consumer at step 3 is clearly written *expecting* the main to be flagged −1 —
that is also the convention the SBND config documents ("carried as the
`isolated`/`perblob` per-blob array (main blobs tagged -1)",
`cfg/.../sbnd/clus.jsonnet`). The producer at step 1 does not follow it. So the
overlap-preserving branch at `:181-205`, which exists precisely to carry an
incoming main forward, can never run after `clustering_isolated`.

Measured consequence at the PR stage (from the trace run's own pctree,
`pointtrees/.../namedpcs/perblob/arrays/isolated`): **93.1 % of the 7802 blobs
are −1** with only 11 non-main groups in the entire event. What reaches the
taggers is relaxed-graph connectivity of flash-merged clusters — and the relaxed
graph bridges gaps, which is why almost everything is "main". The record of which
blobs were an *associated* cluster is gone.

Note this is **not** the merge doc 45 undoes. There are **two** merges of
"main + associated" and only **one** un-merge:

| merge | where | provenance | inverted? |
|---|---|---|---|
| isolated grouping | per-APA `clustering_isolated` | member index, **overwritten at the next pass** | **no — and no longer recoverable** |
| flash bundle | all-APA `examine_bundles` | `real_cluster_id` / `real_cluster_main`, persisted (doc 38) | yes — `ClusteringUnmergeBundle`, doc 45 |

QLMatching's `decompose_cluster_groups` reads the isolated array to build
prototype-style bundles, but by then the array is the connectivity partition, the
decomposition is internal to matching, and the objects are recomposed afterwards
(`project_group_aware_ql_matching`) — the tagger stage receives the fused object.

### 4c. Why it was probably built this way

The toolkit pctree has no cluster-level parent-id field; uBooNE's output tree
does (`parent_cluster_id`). Encoding the grouping as merge + per-blob array was
plausibly a way around that rather than an oversight. But the mechanism for a
cluster-level parent id does exist — `matched_flash_gid` is exactly such a scalar
— so the prototype's representation is reachable. Nothing in
`clus/docs/porting/porting_dictionary.md` records this divergence, which by M15
makes it one to surface, not to silently pick.

## 5. The designed gap-jumpers are innocent

The passes that exist to cross dead channels and SP inefficiency had 13
consecutive chances and all declined: `extend` (60 cm, `num_dead_try=1`),
`regular` ×2 (60 and 30 cm), `parallel_prolong` (35 cm), `close` (1.2 cm),
`extend_loop`, `connect1` (`iso_max_dis` 5 cm). Their cuts are far larger than
the 18–20 cm gaps here, so they *could* have merged and chose not to — they apply
direction/Hough and charge-continuity tests. Consistent with doc 50's finding
that **0 %** of these gaps lie in a dead region (this MC sample has ~94 cm² of
dead area per TPC).

That is the cleanest evidence that this is not the inefficiency-bridging
machinery doing its job: the machinery said no, and a grouping step said yes.

## 6. What it costs

From doc 50, unchanged here: 69 % of STM-fitted mains are multi-component, 26 % of
fits have a trajectory point >5 cm from any charge of their own cluster, and 12 of
44 STM tags rest on a fit whose endpoint is in a detached clump. §2 shows where
those composites come from.

## 7. Remedies — described, NOT implemented

1. **Mark the main, so the existing carry-forward path works.** Have the isolated
   merge write **−1** for the longest member instead of index 0..N−1 (i.e. pass
   the group's main through `merge_clusters`, or post-process `cc`). Then
   `examine_bundles:181-205` — code that already exists and is currently dead —
   preserves the grouping instead of recomputing it, and the array reaching
   QLMatching means what its config comment says. Smallest change, fixes a
   documented-convention violation, and it is a prerequisite for anything else.
   It does change what `decompose_cluster_groups` sees, so it needs an A/B.
2. **Also persist it past `switch_scope`** if the PR stage is to see it: add
   `isolated` to the carry list at `clustering_switch_scope.cxx:88-95`, under a
   different array name to avoid the collision with the all-APA
   `examine_bundles` write. This is the exact analogue of doc 38's
   `save_real_cluster_id`, including its two gotchas (TensorDM drops
   heterogeneous same-named PC keys; `separate()`/`from()` drops node-local PCs).
   With 1 + 2 in place, a late un-merge in the style of doc 45 becomes possible —
   the worked example is already sitting commented out at
   `clustering_isolated.cxx:510-535`.
3. **Do not merge at all; carry a parent id** (closest to the prototype, §4c).
   The honest end state: `clustering_isolated` writes a cluster scalar and leaves
   the objects separate, so main-only fitting falls out for free everywhere. Most
   invasive — it changes the cluster count entering QLMatching.
4. **Guard the 80 cm cut** with a direction or gap-quality test. Not recommended:
   it deviates from criteria that are otherwise a clean port, and the criteria are
   not what is wrong.

A **late un-merge alone is not an option** — I checked, and the information it
would need no longer exists at that point (§4b steps 4-7). It requires 1 + 2
first.

## 8. Caveats and open items

- Two events. The mechanism is general (§3-§4) and the collapse ratios (25→9,
  46→11) say it is common, but "`clustering_isolated` is the merger" is
  established here only for these two clusters. Running
  `stm_merge_attribution.py` over doc 50's 36 end-in-clump cases would make it a
  rate; not done.
- Which of the two distance branches fired is not pinned down (§3).
- The 80 cm merge was already implicated once, in
  `docs/overclustering-evt11-gamma.md`, and mitigated by tightening `length_cut`
  20 → 15 cm. This doc says that treated a symptom: any genuinely small fragment
  within 80 cm still gets fused, and the record of it being separate is then lost.
- Whether uBooNE's mains are as fragmented as ours is still unmeasured (doc 50 §6).
- `trace_bee` adds ~25 Bee layers per event; diagnostic only, not for scan rounds.

# Why event 269774's neutrino looked fragmented on the `tgm` branch but coherent on `apply-pointcloud`/Xin

**TL;DR** — The BEE "Cluster" color is `real_cluster_id`, not `cluster_id`. On the
`tgm` WCT branch `QLMatching` splits each matched cluster into a main + sub-clusters
(`decompose_cluster_groups`, to anchor bundle geometry) but **never merges them back**.
The `apply-pointcloud` branch added the inverse step **`recompose_cluster_groups`**.
So on `tgm` the split leaks into the output tree as extra `real_cluster_id`s that the
geometry-based all-APA merge cannot rejoin → the neutrino is painted as ~67 color
fragments; on `apply-pointcloud` it is ~13. **The physical clustering (`cluster_id`
9-group partition + all points) is byte-identical between the branches** — the
difference is purely the display sub-labeling. Root cause: the missing
`recompose_cluster_groups` on `tgm` ("the QLMatching split-cluster bug").

- Exact file: **`match/src/QLMatching.cxx`** (function `recompose_cluster_groups`,
  present on `apply-pointcloud`, absent on `tgm`).
- Event: run 18255 / subRun 1 / **event 269774** (a nueCC), the 48-event data
  frameshift sample.

---

## 1. Symptom

BEE set `93907c09` (our old run, 1-step larwirecell chain, `tgm` branch), event 20:
the nueCC neutrino is one clean cluster in `clustering-apa1-face0` but appears broken
into many colors in `img-global` / `clustering-global`. Xin's set `0fbcecd1` and a
fresh `apply-pointcloud` run (`082629b0`) looked coherent. The question: what makes
the two differ?

## 2. Gotchas found along the way

- **BEE event index is 0-based and follows file order.** `.../event/20/` is the
  **21st** record = `lar --nskip 20` = event **269774**. `--nskip 19` is the 20th
  record = event **268784** (a different event). Verify the file's RSE order with a
  minimal source-only job in a CLEAN sbndcode env (no WCT `opt` libs on
  `LD_LIBRARY_PATH`, or `libWireCellRoot.rootmap` dict-clashes with the sbndcode PTB
  dict):
  ```fcl
  # rsedump.fcl
  process_name: rsedump
  source: { module_type: RootInput }
  physics: {}
  ```
  ```
  lar -c rsedump.fcl -s <file>.root -n 48   # grep "Begin processing ... run: ... event:"
  ```

- **BEE paints the "Cluster" color by `real_cluster_id`, not `cluster_id`**
  (`wire-cell-bee3/events/static/js/bee/physics/sst.js:152`):
  ```js
  let color_id = this.data.real_cluster_id[ind] > 0
                 ? this.data.real_cluster_id[ind] : this.data.cluster_id[ind];
  ```
  The hover readout (`scene.js:567`) also prints `cluster = real_cluster_id`. Comparing
  `cluster_id` (identical) hid the difference for a while.

- **`apply-pointcloud` WCT is ABI-incompatible with the `tgm`-built larwirecell in
  `opt`** (`match/` diverged +4970 lines, `clus.jsonnet` +496). The larwirecell dump /
  1-step chain **segfaults** on `apply-pointcloud` WCT; it works on `tgm` WCT. To dump
  icluster/opflash you must build `tgm` WCT, dump, then rebuild `apply-pointcloud` for
  the (pure-wire-cell) matching. `tgm`'s `match/` does, however, compile cleanly
  against `apply-pointcloud`'s `clus/`, which made the bisection swaps possible.

## 3. What the fields mean

In the all-APA (`use_flash_t0=true`) clustering, `clustering_examine_bundles.cxx`
groups every base cluster sharing a flash-time group and **merges** them into one —
that merged identity is **`cluster_id`** (the 9 flash-t0 groups here). Before merging,
each blob's **original pre-merge base-cluster id** is stored per-blob as
**`real_cluster_id`** (`merge_clusters(..., "perblob", "real_cluster_id", ...)`), so
BEE can still color the individual base clusters. The Bee writer reads it back at
`clus/src/MultiAlgBlobClustering.cxx:1575`.

So `cluster_id` = coarse flash-time group; `real_cluster_id` = individual base cluster
(what you see colored).

## 4. Measurements (all on event 269774, same input)

| set | `cluster_id` | `real_cluster_id` |
|---|---|---|
| OLD `93907c09` (tgm, 1-step) | 9 | **67** |
| Xin `0fbcecd1` (apc, 2-step) | 9 | **13** |
| NEW `082629b0` (apc, 2-step) | 9 | **13** |

`clustering-global` points + `cluster_id` partition are **byte-identical** across all
three (the `cluster_id` integers are only relabeled). `clustering-apa1-face0` (per-APA)
is byte-identical too. Only `real_cluster_id` differs.

`examine_bundles` diagnostic ("in-scope matched clusters → flash-time groups"):

| run | code | workflow | line |
|---|---|---|---|
| OLD `93907c09` | tgm | 1-step | `68 → 9` |
| tgm-2-step (same dumps) | tgm | 2-step | `68 → 9` |
| apc-2-step (same dumps) | apply-pointcloud | 2-step | `15 → 9` |

## 5. Elimination

- **Not the config.** `clus.jsonnet` per-APA (`clus_per_face`) **and** all-APA
  `cm_pipeline`s are byte-identical between branches (same steps, same params). The
  `cathode_connect` step (config toggle `cathode_connect_on`) — toggled OFF in apc —
  still gives 13. `examine_bundles(flags_from_longest=true)` is the only pipeline-config
  difference and is flag bookkeeping.
- **Not the workflow.** `tgm` gives 67 in **both** 1-step and 2-step (rebuilt tgm, ran
  the standalone perevt on the same dumps). Dump/reload is not it.
- **Not the flash/t0 matching.** The matched **bundles are byte-identical**: 16 bundles,
  same 12 flash t0s (in-time 717/735 ns; out-of-time cosmics +1107 µs / −831 µs /
  −979 µs / …). So `apply_matched_t0s` stamps the same t0 on the same clusters in both.
- **Not `clus/`.** See the bisection below — swapping `match/` alone flips the result.

## 6. Bisection (clean source swaps + full rebuild, no runtime ABI mixing)

| build (same dumps) | `real_cluster_id` |
|---|---|
| apc-clus + apc-match (baseline) | 13 |
| apc-clus + **tgm-match** (whole library) | **67** |
| apc-clus + apc-helpers + **only tgm `QLMatching.{cxx,h}`** | **67** |

Swapping **only** `QLMatching.cxx` (keeping every other `match/` file — `Opflash`,
`TimingTPCBundle`, `PhotonLibraryModel` — and all of `clus/` at apply-pointcloud) flips
13↔67. **`match/src/QLMatching.cxx` is the file** (the +2709-line rewrite; 93% of the
whole `match/` diff).

## 7. Root cause — the missing `recompose`

`QLMatching::operator()` (both branches identical through step 5):

1. `decompose_cluster_groups(run)` — for each matched cluster with several internal
   connected-components, **split** it into a *main* (cc id < 0) + *associated
   sub-clusters* (id ≥ 0) via `Grouping::separate()`, to anchor each bundle's geometry
   on its main component.
2. `build_bundles` / `build_bundle_maps` — form the 16 flash bundles (identical).
3. `fit_round1/2` — LASSO match (identical).
4. `apply_matched_t0s(run)` — stamp `cluster_t0`/`flash`/`matched_flash_gid` on each
   bundle's main **and** its `other_clusters` (identical: same t0 to the same sets).
5. `write_opflash_pc`.
6. **`recompose_cluster_groups(run)` — apply-pointcloud only** (`tgm`: 0 references):
   ```cpp
   void QLMatching::recompose_cluster_groups(ApaRun& run) {
       auto* grouping = run.grouping;
       for (auto& [main_cluster, others] : run.match_groups) {
           if (others.empty()) continue;
           grouping->merge(others.begin(), others.end(), main_cluster);  // undo decompose's split
       }
   }
   ```

`tgm` performs the split (step 1) but never the merge-back (step 6) — the code comment
in `apply-pointcloud` calls this "the match/QLMatching split-cluster bug."

**Why 67 vs 13:**
- **tgm:** the `Grouping::separate()` split persists into the tree `QLMatching` hands to
  MABC. Those sub-clusters are *separate connected components* (that is why `separate()`
  split them), so the all-APA `use_flash_t0` merge steps — which merge on **geometry +
  shared flash-t0** — cannot rejoin them even though they share the same t0. They survive
  → **68 base clusters → `real_cluster_id` 67**.
- **apply-pointcloud:** `recompose_cluster_groups` merges each split group back by
  **match-group membership** (not geometry) before serializing → whole clusters →
  **15 base clusters → `real_cluster_id` 13**.

Because `recompose` only re-merges clusters that were one cluster to begin with (same
match group, same flash-t0), it changes neither the 9-group `cluster_id` partition nor
any point — only the `real_cluster_id` sub-labeling (the BEE colors).

## 8. Conclusion

- The apparent difference between "our old run" and "Xin's run" for the event-269774
  neutrino is **entirely `real_cluster_id` (BEE color) granularity**, not a difference
  in the physical clustering. The `cluster_id` grouping and all points are identical.
- It is a **`tgm`-branch bug**: `QLMatching` decomposes clusters for bundle building
  and forgets to recompose them. `apply-pointcloud` fixed it with
  `recompose_cluster_groups`.
- **Nothing to fix on `apply-pointcloud`** for this. If a `tgm`-derived config is ever
  used for production display, either rebase onto the `apply-pointcloud` `QLMatching`
  or back-port `recompose_cluster_groups`.
- Reminders that saved/cost time: BEE colors by `real_cluster_id`; BEE event index is
  0-based; `apply-pointcloud` WCT ⟂ `tgm` larwirecell ABI (dump on tgm, match on apc).

## 9. Artifacts

- Bisection runs: `sbnd/TensorSetLabeler/data-frameshift/evt269774-xin/`
  `{work (dumps), bp-on, tgm-2step, hybrid-tgmmatch, qlm-isolate}` — each holds
  `mabc-all-apa.zip` (inspect `real_cluster_id` in `data/0/0-clustering-global.json`).
- BEE (apply-pointcloud, correct event 269774, per-APA + global):
  https://www.phy.bnl.gov/twister/bee/set/082629b0-86f1-4d53-b9fa-1cc571f36efa/event/list/
- Compared against OLD `93907c09` (event/20) and Xin `0fbcecd1` (event/20).
- Memory: `project_evt20_neutrino_fragmentation`.

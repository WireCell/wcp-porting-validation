# Clustering Algorithms in the SBND Chain

> For the clustering *stage* / driver scripts see **[clustering.md](clustering.md)**.
> For the QL (charge–light) matching chain see **[ql-chain.md](ql-chain.md)**.

## Scope

This document answers four questions about the `sbnd_xin` clustering chain:

1. Which clustering algorithms run in the **per-APA** (single-TPC) case.
2. Which clustering algorithms run in the **combined / all-APA** (multi-TPC) case.
3. The role and scope of five functions that are **not** (or only partly) wired
   into the current `sbnd_xin` chain: `deghost`, `examine_x_boundary`,
   `protect_over_clustering`, `examine_bundles`, `retile`.
4. Specifically: what case `examine_x_boundary` handles, and whether
   `examine_bundles` uses the light/flash signal.

Throughout, `clus.jsonnet` refers to the canonical in-tree SBND clustering
module `cfg/pgrapher/experiment/sbnd/clus.jsonnet`; `sbnd_xin/clus.jsonnet` is a
thin re-export of it. Each pipeline step is a `Clustering*` C++
`IEnsembleVisitor` constructed by `clus.clustering_methods()` in
`cfg/pgrapher/common/clus.jsonnet`. Source lives under `clus/src/`.

The chain runs in two sequential stages: per-APA `MultiAlgBlobClustering`
(MABC) → `PointTreeMerging` → all-APA MABC.

---

## 1. Per-APA clustering (single TPC / single anode+face)

`cm_pipeline` at `cfg/pgrapher/experiment/sbnd/clus.jsonnet:135-151`. Runs in
local (anode-relative) `x`.

| # | jsonnet step | C++ class | purpose |
|---|---|---|---|
| 1 | `pointed()` | `ClusteringPointed` | mark/select pointed clusters |
| 2 | `live_dead(dead_live_overlap_offset=2)` | `ClusteringLiveDead` | merge live blobs with overlapping dead-channel regions |
| 3 | `extend(flag=4, length_cut=60cm, length_2_cut=15cm, num_dead_try=1)` | `ClusteringExtend` | extend cluster trajectories (dead-region aware) |
| 4 | `regular(name='-one', length_cut=60cm, flag_enable_extend=false)` | `ClusteringRegular` | pairwise merge within 60 cm, no extension |
| 5 | `regular(name='_two', length_cut=30cm, flag_enable_extend=true)` | `ClusteringRegular` | tighter pairwise merge with extension |
| 6 | `parallel_prolong(length_cut=35cm)` | `ClusteringParallelProlong` | merge parallel / prolonged segments |
| 7 | `close(length_cut=1.2cm)` | `ClusteringClose` | merge spatially adjacent clusters |
| 8 | `extend_loop(num_try=3)` | `ClusteringExtendLoop` | iterate extension until convergence |
| 9 | `separate(use_ctpc=true)` | `ClusteringSeparate` | split over-merged clusters |
| 10 | `connect1()` | `ClusteringConnect1` | first-pass connectivity using detector-volume info |

**Inactive tail.** `clus.jsonnet:146-150` carries a commented-out tail —
`deghost()`, `examine_x_boundary()`, `isolated()` — inherited from the original
cfg `func_cfgs` list. These are **intentionally not active** in the `sbnd_xin`
per-APA chain (left in place to re-enable for testing later).

---

## 2. Combined / all-APA clustering (TPCs merged)

`PointTreeMerging` merges the per-APA point-tree outputs into one global
grouping; a second MABC then runs on it. `cm_pipeline` at
`cfg/pgrapher/experiment/sbnd/clus.jsonnet:226-237`. Runs in the **T0-corrected**
coordinate scope (`common_corr_coords`).

| # | jsonnet step | C++ class | purpose |
|---|---|---|---|
| 1 | `switch_scope()` | `ClusteringSwitchScope` | insert `x_t0cor`; switch the active PC scope to T0-corrected coords |
| 2 | `extend(flag=4, length_cut=60cm, length_2_cut=15cm, num_dead_try=1)` | `ClusteringExtend` | re-run extension in the T0-corrected scope |
| 3 | `regular(name='1', length_cut=60cm, flag_enable_extend=false)` | `ClusteringRegular` | pairwise merge, 60 cm |
| 4 | `regular(name='2', length_cut=30cm, flag_enable_extend=true)` | `ClusteringRegular` | tighter pairwise merge with extension |
| 5 | `parallel_prolong(length_cut=35cm)` | `ClusteringParallelProlong` | parallel / prolonged merge |
| 6 | `close(length_cut=1.2cm)` | `ClusteringClose` | merge adjacent clusters |
| 7 | `extend_loop(num_try=3)` | `ClusteringExtendLoop` | iterative extension |
| 8 | `separate(use_ctpc=true)` | `ClusteringSeparate` | split over-merged clusters |
| 9 | `neutrino()` | `ClusteringNeutrino` | neutrino-candidate clustering pass |
| 10 | `isolated()` | `ClusteringIsolated` | flag isolated (unconnected) clusters |

**Per-APA-only vs combined-only.** The shared core (`extend`, `regular×2`,
`parallel_prolong`, `close`, `extend_loop`, `separate`) is identical. Per-APA
additionally has `pointed`, `live_dead`, `connect1` and starts in local `x`;
combined starts with `switch_scope` (T0 correction) and ends with `neutrino` +
`isolated`.

---

## 3. With vs without QL (charge–light) matching

| | config | flow |
|---|---|---|
| **Without QL** | `sbnd_xin/wct-clustering.jsonnet` | per-APA MABC → `PointTreeMerging` → all-APA MABC → Bee. No optical layer. |
| **With QL** | `sbnd_xin/wct-clus-matching-perevt.jsonnet` (per-event) or `wct-clus-matching-standalone.jsonnet` (all-10) | per-APA MABC → **QL matching** → `PointTreeMerging` → all-APA MABC → Bee (+ optical layer). |

QL matching is inserted **after** per-APA clustering and **before** all-APA
clustering. Each per-APA output feeds `FlashTensorToOpticalPCs` (expands the
opflash matrix into flash/light PCs) then `QLMatching`, which writes a
per-cluster flash scalar + `cluster_t0` back onto the point-tree.
`PointTreeMerging` carries the `opflash` root PC into the merged grouping
(`clus.jsonnet:217`), so the all-APA MABC can dump the optical Bee display.
Helper nodes are defined in `cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet`.

**The clustering algorithms themselves do not change.** Both the per-APA and
all-APA `cm_pipeline` arrays are identical in the with-QL and without-QL chains.
QL matching is transparent to clustering: it reads matched clusters and writes a
scalar match + T0; it does not alter clustering topology. (The downstream
all-APA `switch_scope` does consume the per-cluster T0 written by matching, so
QL matching influences the *coordinates* the combined stage sees, but not the
list of algorithms run.)

---

## 4. Reference: five functions not (fully) in the `sbnd_xin` chain

None of `deghost`, `examine_x_boundary`, `protect_over_clustering`,
`examine_bundles`, `retile` run in the current `sbnd_xin` per-APA chain.
`isolated` (same family) is the only one that *is* live — in the all-APA chain.

| function | source | runs per-APA / combined / both | what it does |
|---|---|---|---|
| `deghost` | `clus/src/clustering_deghost.cxx` (`apas.size()>1` raises) | **per-APA, single APA only** (loops all faces within the APA) | removes ghost / duplicate clusters by comparing each cluster's 2D U/V/W wire-plane projections against a reference cloud built from longer, already-validated clusters |
| `examine_x_boundary` | `clus/src/clustering_examine_x_boundary.cxx`; impl `Facade_Cluster.cxx:2337-2428` (`wpids().size()>1` raises) | **per-APA, single face only** | drift-boundary cleanup — see below |
| `protect_over_clustering` | `clus/src/clustering_protect_overclustering.cxx` | **both** — accepts a multi-APA grouping but processes per `(apa,face)` via `time_blob_map`, using `get_nticks_per_slice().at(apa).at(face)` | prevents over-fragmentation by rebuilding connectivity with the "relaxed" graph topology (same physics as `connect_graph_relaxed`) |
| `examine_bundles` | `clus/src/clustering_examine_bundles.cxx` ("All APA Faces", no apa/face restriction) | **both / all faces** | per-blob *labeling* pass (not a split) — see below |
| `retile` | `clus/src/clustering_retile.cxx`; core `clus/src/retile_cluster.cxx` (per-`(apa,face)` samplers) | **both** — multi-APA capable | re-tiles clusters from refined ray-grid activity: rebuild blobs → resample point clouds → form new "shadow" clusters |

### 4a. What case does `examine_x_boundary` handle?

It is a **single-face drift-direction (x) boundary cleanup** — *not* a cross-TPC
or cross-cathode merge. For each cluster of length 5–150 cm
(`Facade_Cluster.cxx:2337-2428`):

- Each 3D point is binned into three regions relative to the per-face fiducial
  bounds `[FV_xmin - FV_xmin_margin, FV_xmax + FV_xmax_margin]` (bounds read from
  `dv->metadata(...)`, the drift extent of the one drift volume):
  `num_points[0]` = below the low limit, `num_points[1]` = in fiducial,
  `num_points[2]` = above the high limit.
- If the out-of-bounds points are only a **small fraction** of the in-bounds
  points (`num_points[0] + num_points[2] < num_points[1] * 0.075`), the cluster
  is split into up to three sub-clusters: group 1 (`x` below low), group 2
  (in-fiducial), group 3 (`x` above high).

In effect it peels a small tail of blobs that pokes past the single-face drift
fiducial boundary off the main in-volume cluster. Because it raises on
`wpids().size() > 1`, it can only be applied per single APA/face (the per-APA
stage), never on the merged all-APA grouping.

### 4b. What `examine_bundles` examines (and what it does *not* do)

It does **not** examine every cluster, and it does **not** split clusters. It
loops over all live clusters (`clustering_examine_bundles.cxx:82`) but **skips
any cluster that fails the scope filter** (`get_scope_filter(scope)`, line 85).
For each in-scope cluster it runs `connected_blobs(dv, pcts, graph_name)`
(line 98), which groups that cluster's blobs into connected components — the
"bundles" — via the wire-overlap graph. It then picks the **main** bundle
(maximum overlap with the previous main, lines 103-130, else the **longest**
component, lines 139-166), marks it `-1`, and writes the per-blob labels back
with `put_pcarray(b2groupid, "isolated", "perblob")` (line 168). It is a
per-blob **labeling / annotation pass** — it records which connected-component
bundle each blob belongs to and which bundle is the main one; it never calls
`live_grouping.separate()`. The actual split into separate `Cluster` objects is
left to the `isolated()` step.

### 4c. Does `examine_bundles` use the light/flash signal?

**No.** `examine_bundles` is purely **charge-based geometry**. It calls
`Cluster::connected_blobs()`, which builds a wire-overlap connectivity graph and
returns connected components; there are no references to flash / opflash /
optical / PE / light anywhere in `clustering_examine_bundles.cxx`. The "bundles"
in the name are blob (charge) bundles, not optical bundles.

Contrast with `retile`, which **is** flash-gated: `retile_cluster.cxx:559-564`
skips any cluster without a matched `get_flash()` or whose flash time falls
outside `[m_cut_time_low, m_cut_time_high]`. So among these functions, only
`retile` depends on QL/flash information.

### 4d. Prototype (MicroBooNE) comparison — is light used there?

Both functions were checked against the MicroBooNE prototype to see whether the
*original* code used optical/light (flash, PE, PMT) information.

**`examine_bundles` — prototype `ExamineBundles` / `ExamineBundle`**
(`prototype_base/2dtoy/src/ExamineBundles.cxx`,
header `prototype_base/2dtoy/inc/WCP2dToy/ExamineBundles.h`):

```cpp
FlashTPCBundleSelection ExamineBundles(WCP::FlashTPCBundleSelection bundles,
                                       WCP::ToyCTPointCloud& ct_point_cloud);
```

The prototype runs *inside the charge–light matching framework*: it operates on
`FlashTPCBundle` objects, each of which pairs a `PR3DCluster` with its matched
`Opflash` plus `pred_pmt_light`, `flag_close_to_PMT`, etc. **But the actual
splitting decision is purely charge/geometry**: it merges the bundle's clusters,
calls `PR3DCluster::Examine_graph(ct_point_cloud)` (`ExamineBundles.cxx:77`) to
find connected components — the prototype counterpart of the toolkit's
`connected_blobs()` — and keeps the component with the largest overlap with the
old main cluster. The flash / PE / PMT fields are only **carried through** onto
the re-formed bundle (`ExamineBundles.cxx:107-118`: `get_flash()`,
`set_pred_pmt_light()`, `set_flag_close_to_PMT()`), never consulted to decide
the split.

So the conclusion is the **same in both code bases**: the `examine_bundles`
splitting logic is charge-only. The only difference is bookkeeping — the
prototype re-attaches the result to its flash (because it lives in the bundle /
matching stage), whereas the toolkit version is decoupled from flash and just
writes per-blob `("isolated","perblob")` labels.

**`retile` — no prototype equivalent.** There is no prototype re-tiling routine
that uses light. The closest prototype function is
`Improve_PR3DCluster[_1/_2]` (`prototype_base/pid/src/ImprovePR3DCluster.cxx`),
whose signature takes only a cluster, the `ToyCTPointCloud`, the geometry, and a
holder — **no flash argument** — and whose body has no flash / PE / T0 / drift-
offset references. The toolkit's `retile` is therefore where the flash
dependence was *added* (the `get_flash()` time-window gate of §4c, plus running
in the T0-corrected `x_t0cor` scope).

# Point sampling in PDHD / PDVD imaging

How 3-D points are produced from imaging blobs, what point clouds exist, and which
ones feed clustering vs. the Bee event display. §8 contrasts this with how
MicroBooNE (`qlport`) samples points (`charge_stepped`); §9 records why PDHD/PDVD
keep `stepped` (not `charge_stepped`) for the clustering stage.

## TL;DR

* Imaging (`img.jsonnet`) produces **blobs** — RayGrid shapes (per-plane wire
  strips + their geometric corners). It does **no point sampling and no Bee
  output**. Blobs are written to a cluster file.
* Live blobs are sampled **exactly once**, with the `stepped` strategy, into a
  point cloud named `"3d"`. **Both** the clustering algorithms **and** every Bee
  charge dump read those same points. There is no separate "imaging sampler" and
  no separate "Bee sampler".
* "Imaging for Bee" vs "clustering for Bee" is **not** two samplings. It is the
  *same* `"3d"` stepped points dumped at two different points in the pipeline:
  * `name:"img"`  → dumped **before** the all-APA clustering pipeline (the per-APA
    result, grouped by drift side).
  * `name:"clustering"` → dumped **after** the pipeline (full-detector result).
* The genuinely *different* point clouds are dead-blob `center`/`corner`,
  `ctpc_*` (charge), `scalar` (per-blob summary), `dead_winds_*`, and optional
  `steiner_pc`. See the table below.

PDHD and PDVD use identical sampling strategy (`stepped` for live, `center`
for dead, `time_offset = -250 µs`) and detector-specific calibrated
`drift_speed` (1.565 mm/µs PDHD, 1.57 mm/µs PDVD; both were 1.6 before
calibration). They also differ in geometry: PDHD has 4 APAs in 2 drift groups
`{0,2}`/`{1,3}`; PDVD has 8 CRPs in 2 drift groups `{0,1,2,3}` (bottom) /
`{4,5,6,7}` (top).

---

## 1. Where sampling happens in the pipeline

```
  RAW → NF → SP                            (signal processing)
   │
   ▼
  IMAGING  (pdhd/img.jsonnet)
   ├─ GridTiling           → blobs (RayGrid strips + corners) per slice/face
   ├─ BlobClustering       → group blobs across slices
   ├─ GlobalGeomClustering → geometry merge
   └─ ClusterFileSink      → write ICluster (blobs+slices+wires+channels)
                             ── NO sampling, NO Bee here ──
   │
   ▼
  CLUSTERING (pdhd/clus.jsonnet)
   ├─ ClusterScopeFilter
   ├─ PointTreeBuilding    → ***SAMPLING HAPPENS HERE***
   │     ├─ BlobSampler "3d"   strategy=["stepped"]  (LIVE blobs)
   │     ├─ BlobSampler "dead" strategy=["center"]   (DEAD blobs)
   │     ├─ add_ctpc()         → per-channel charge point clouds (NOT sampling)
   │     └─ add_dead_winds()   → dead-wire x-ranges
   └─ MultiAlgBlobClustering (MABC)
         ├─ runs clustering algorithms on the "3d" point cloud
         └─ writes Bee (.zip): charge points, dead-area patches, flashes
```

`img.jsonnet` contains no `BlobSampler` and no `bee` references (verified:
`pdhd/img.jsonnet` lines 121–207 define `GridTiling`/`BlobClustering`/
`GlobalGeomClustering`, line 291 `ClusterFileSink`). All sampling is downstream in
`clus.jsonnet`.

---

## 2. What a blob is before sampling

`GridTiling` (`img.jsonnet:131`) produces a RayGrid blob: a set of per-plane wire
**strips** (`[wire_index_min, wire_index_max)` per U/V/W plane) and the geometric
**corners** where the bounding rays cross. A blob is a 2-D transverse shape at a
given drift slice — it is *not yet* a set of 3-D points. Sampling is the act of
placing discrete 3-D points inside/around that shape and projecting them along the
drift (x) axis using the slice time.

---

## 3. The `BlobSampler` strategies

Source: `clus/src/BlobSampler.cxx`, header `clus/inc/WireCellClus/BlobSampler.h`
(strategy docs at `BlobSampler.h:70-226`). Full reference:
`img/docs/blob-sampling.org`.

A sampler runs a pipeline of one or more **strategies**; the result is their
union. The strategies differ only in how they place points in the transverse
(wire) plane. The available menu:

| strategy        | transverse placement                                            |
|-----------------|------------------------------------------------------------------|
| `center`        | one point = average of the blob corners (the default)            |
| `corner`        | the blob corner points                                          |
| `edge`          | centers of the blob boundary edges                              |
| `grid`          | uniform points on a ray grid (`step`, `planes`)                |
| `stepped`       | sub-sampled ray-grid of two views (see below)                  |
| `bounds`        | points walked along the boundary edges (`step`)                |
| `charge_stepped`| `stepped` + per-wire charge filtering & dead-plane handling     |

**PDHD/PDVD use only `stepped` (live) and `center` (dead).** The rest exist but
are not wired into the PDHD/PDVD configs. **MicroBooNE (`qlport`) uses
`charge_stepped` (live) and `center` (dead)** — see §8.

### 3.1 The `stepped` strategy (live blobs)

`BlobSampler.cxx:703-885`. This reproduces the Wire-Cell *prototype* sampling.

Algorithm:

1. Of the three plane strips, find the one with the **most** wire coverage
   (`smax`), the **least** (`smin`), and the remaining "mid" (`smid`)
   (`BlobSampler.cxx:746-771`).
2. Step sizes along the two directions
   (`BlobSampler.cxx:774-775`):
   ```
   nmin = max(min_step_size, max_step_fraction · swidth(smin))
   nmax = max(min_step_size, max_step_fraction · swidth(smax))
   ```
   with defaults `min_step_size = 3` wires, `max_step_fraction = 1/12`. So a step
   is at least 3 wires and at most ~1/12 of the blob's extent in that view. Small
   blobs ⇒ a 3-wire grid; large blobs ⇒ a coarser grid that scales with size.
3. Build the candidate wire sets in the min/max views (first wire, every step,
   and the last wire), then take every (min-wire × max-wire) crossing
   (`BlobSampler.cxx:815-834`).
4. Keep a crossing only if it also lands inside the third ("mid") plane's strip,
   within `tolerance` (`BlobSampler.cxx:847-855`).
5. **`offset = 0.5`** (default) shifts each kept point along the cell diagonal from
   the *ray* crossing to the *wire* crossing — i.e. to the physical wire-center
   intersection. `offset = 0` would leave points on ray crossings. The `0.5`
   value is what makes this byte-equivalent to the prototype
   (`BlobSampler.h:127-132`, `BlobSampler.cxx:797-799`).

The strategy also emits an **aux** dataset (one row per blob), later folded into
the `scalar` PC: `max_wire_interval` (=nmax), `min_wire_interval` (=nmin),
`max_wire_type`, `min_wire_type` (plane id 0/1/2 of the max/min-coverage view)
(`BlobSampler.cxx:880-883`).

### 3.2 The `center` strategy (dead blobs)

`BlobSampler.h:95-96`. A single point at the average of the blob corners. Dead
("bad") blobs only need one representative point; their *shape* for display comes
from the separate `corner` dataset (§4.2).

### 3.3 Drift (x) projection — common to all strategies

After transverse placement, every point is projected along drift using the slice
time and a time→x conversion (`Aux::time2drift`,
`aux/src/SamplingHelpers.cxx:247-256`):

```
x = x_origin + xsign · (t_slice + time_offset) · drift_speed
```

The number of x-samples per blob is set by the time binning `(tbins, tmin, tmax)`,
default `(1, 0.0, 1.0)` → **one** sample taken at the blob's start time
(`BlobSampler.h:147-165`). PDHD/PDVD use the default, so each transverse point
yields one 3-D point at the slice start. `time_offset = -250 µs`,
`drift_speed = 1.565 mm/µs` (PDHD; PDVD 1.57; calibrated from A–C crossers, was
1.6) (`clus.jsonnet:7-8,113-114`).

---

## 4. The point clouds attached to each blob

`PointTreeBuilding` builds a point-cloud tree:
`root (Grouping) → cluster nodes → blob nodes`. Each blob node carries one or more
named local point clouds; the grouping (root) node carries detector-wide ones.

### 4.1 Live blob node — `Aux::sample_live` (`SamplingHelpers.cxx:9-32`)

* **`"3d"`** — the stepped sample points. Columns:
  * `x, y, z, t` — 3-D position + time.
  * Per-plane (matched by the sampler's `extra` regex
    `[".*wire_index", ".*charge_val", ".*charge_unc", "wpid"]`,
    `clus.jsonnet:116`): `u/v/w wire_index`, `u/v/w charge_val`,
    `u/v/w charge_unc`, `wpid`.
  * 2-D projections per plane added by `fill_2dpcs`
    (`SamplingHelpers.cxx:49-78`): for each plane angle, `x` and a rotated
    `y = cos(angle)·z − sin(angle)·y`. These give the per-view 2-D coordinates the
    clustering uses for plane-wise distance tests.
* **`"scalar"`** — one row summarizing the whole blob
  (`SamplingHelpers.cxx:81-148`): `charge`, `wpid`, `slice_index_min/max`,
  `u/v/w_wire_index_min/max`, `center_x/y/z`, `npoints`, and the stepped aux
  (`max/min_wire_interval`, `max/min_wire_type`).

### 4.2 Dead blob node — `Aux::sample_dead` (`SamplingHelpers.cxx:34-46`)

* **`"scalar"`** — same per-blob summary (with zeroed center/aux).
* **`"corner"`** — the blob's geometric corner points
  (`make_corner_dataset`, `SamplingHelpers.cxx:171-245`). These corners are what
  Bee draws as **dead-area patches** (§6.3). For reloaded 2-view dead blobs the
  *stored* imaging-time corners are used, since re-derived RayGrid corners can run
  past the wire boundary.

### 4.3 Grouping (root) node — charge & dead-wire clouds

These are **not** blob sampling; they are built directly from slice activity and
attached to the root.

* **`ctpc_a{apa}f{face}p{U|V|W}`** — the "charge tiling point cloud"
  (`PointTreeBuilding::add_ctpc`, `PointTreeBuilding.cxx:272-360`; library twin
  `SamplingHelpers.cxx:260-359`). One 2-D point **per live channel per slice**:
  `x` = drift position, `y` = wire-pitch position `pitch·(wind+0.5) + proj_center`,
  plus `charge`, `charge_err`, `cident`, `wind`, `slice_index`. This is the
  charge-aware geometry that algorithms like `separate(use_ctpc=true)`
  (`clus.jsonnet:208,427`) query for nearby charge — distinct from the sparse
  stepped `"3d"` points.
* **`dead_winds_a{apa}f{face}p{U|V|W}`** — dead-wire drift ranges
  (`PointTreeBuilding.cxx:362-516`): per dead wire, `xbeg`, `xend`, `wind`. Used by
  `Grouping::is_good_point` / dead-channel tests during clustering.

A channel is "dead" when its charge uncertainty exceeds the dead threshold
(`1e10`), the same convention used throughout (`PointTreeBuilding.cxx:295`,
`MultiAlgBlobClustering.cxx:1577`).

### 4.4 Summary table

| point cloud            | scope/node | producer                       | strategy / source        | consumed by                                  |
|------------------------|------------|--------------------------------|--------------------------|----------------------------------------------|
| `3d`                   | per blob   | BlobSampler "3d"               | `stepped`                | clustering algorithms **and** Bee charge dump |
| `scalar`               | per blob   | `fill_scalar_*`                | blob summary             | clustering (blob-level queries)              |
| `corner`               | dead blob  | `make_corner_dataset`          | blob corners             | Bee dead-area patches                        |
| `ctpc_a*f*p{UVW}`      | grouping   | `add_ctpc`                     | slice activity (charge)  | charge-aware clustering (`use_ctpc`)         |
| `dead_winds_a*f*p{UVW}`| grouping   | `add_dead_winds`               | dead-channel slices      | dead-point / good-point tests                |
| `steiner_pc`           | per cluster| clustering (optional)          | Steiner-tree vertices    | Bee (only if `pcname:"steiner_pc"`)          |

---

## 5. How sampled points feed clustering

`MultiAlgBlobClustering` (MABC) loads the point-cloud tree and runs a pipeline of
clustering methods (`clus.jsonnet:198-212` per-face, `:417-444` all-APA). The
methods operate on the **`"3d"` scope** — the stepped sample points — using the
2-D per-plane projections for plane-wise proximity and the `ctpc_*` clouds for
charge. Example methods in the pipeline: `live_dead`, `extend`, `regular`,
`parallel_prolong`, `close`, `extend_loop`, `separate(use_ctpc=true)`,
`connect1`, `neutrino`, `isolated`, `examine_bundles`.

The all-APA pipeline runs in the `x_t0cor` coordinate set (`clus.jsonnet:415`) —
the same stepped points, with drift-x corrected by the flash-associated t0.

---

## 6. How sampled points reach the Bee display

All Bee output is produced by MABC and written to the `mabc-*.zip` files
(`clus.jsonnet:224,327,455`). Bee draws three kinds of objects.

### 6.1 Charge points — the `bee_points_sets`

Configured per MABC node (`clus.jsonnet:236-245` per-face,
`:467-498` all-APA). Each entry selects a point-cloud scope to dump:

```jsonnet
{ name: "clustering", detector: "protodunehd",
  algorithm: "clustering", pcname: "3d",
  coords: ["x","y","z"], individual: false }
```

`fill_bee_points_from_cluster` (`MultiAlgBlobClustering.cxx:1497-1619`) reads the
scoped `"3d"` point cloud (one sub-PC per blob) and appends **every sampled
point** to a `Bee::Points`. Each point gets a per-point charge computed as the
**mean of the non-dead plane charges** for the wires defining that point
(`MultiAlgBlobClustering.cxx:1601-1612`) — the prototype formula. So the Bee
charge view shows exactly the stepped sample points used for clustering, colored
by charge.

### 6.2 `img` vs `clustering` — same points, different pipeline stage

The all-APA config defines two sets (`clus.jsonnet:467-498`):

* **`name:"img"`** — dumped **before** the all-APA pipeline runs
  (`MultiAlgBlobClustering.cxx:2087-2096`). At that moment the live clusters are
  the per-APA clustering result. With `apa_groups` it is grouped by drift side →
  Bee instances `clustering-group02` / `clustering-group13` (PDHD) or
  `clustering-group0123` / `clustering-group4567` (PDVD).
* **`name:"clustering"`** — dumped **after** the pipeline
  (`MultiAlgBlobClustering.cxx:2177-2189`) → `clustering-global`, the
  full-detector clustering.

Both read `pcname:"3d"`. The difference is *when* and *how grouped*, not a
different sampling. (A set may also pin to a specific visitor via `visitor`, in
which case it is dumped right after that algorithm runs,
`MultiAlgBlobClustering.cxx:2126-2154`.)

### 6.3 Dead-area patches

When `save_deadarea` is on (`clus.jsonnet:231`), `fill_bee_patches_from_grouping`
(`MultiAlgBlobClustering.cxx:2073-2083, 1624-1725`) draws the dead blobs as Bee
**patches** built from the dead-blob `corner` point clouds (§4.2), grouped per
APA/face or per drift group (`dead_apa_groups`, `clus.jsonnet:464,475`).

### 6.4 Flashes and Steiner points

* Optical flashes are dumped pre-pipeline (`MultiAlgBlobClustering.cxx:2104-2113`)
  while the 1:1 cluster↔flash mapping still holds.
* If a `bee_points_sets` entry uses `pcname:"steiner_pc"`, Bee dumps Steiner-tree
  vertices instead of sampled points, tagging terminals vs. non-terminals
  (`MultiAlgBlobClustering.cxx:1505-1536`). Not used in the default PDHD/PDVD
  configs.

---

## 7. PDHD vs PDVD

Identical sampling configuration:

| item              | PDHD                       | PDVD                              |
|-------------------|----------------------------|-----------------------------------|
| live strategy     | `stepped`                  | `stepped`                         |
| dead strategy     | `center`                   | `center`                          |
| `drift_speed`     | 1.565 mm/µs (calib, was 1.6) | 1.57 mm/µs (calib, was 1.6)     |
| `time_offset`     | −250 µs                    | −250 µs                          |
| `bee_detector`    | `protodunehd`              | `protodunevd`                     |
| anodes / groups   | 4 APAs, `{0,2}` / `{1,3}`  | 8 CRPs, `{0,1,2,3}` / `{4,5,6,7}` |

Source: `pdhd/clus.jsonnet:7-8,22-25,109-126`;
`pdvd/clus.jsonnet:7-8,23-25,127,135,237`.

---

## 8. MicroBooNE (`qlport`)

MicroBooNE (`qlport/uboone-mabc.jsonnet`) runs the **same toolkit
`WireCellClus` components** (BlobSampler, MABC, the `Aux::sample_live`/
`sample_dead` helpers) but with a different upstream, a different live strategy,
and different physics constants. The point-cloud *structure* attached to each
blob (`3d`, `scalar`, `corner`, `ctpc_*`, `dead_winds_*`) is identical to
PDHD/PDVD — only how the `3d` points are *chosen* and *projected* differs.

### 8.1 Different pipeline shape — imaging is upstream, sampling is in the source

PDHD/PDVD image in WCT (`img.jsonnet` → `GridTiling`) and sample later in
`PointTreeBuilding`. MicroBooNE does **neither in WCT**: the blobs are already
tiled (the WCP "TC" tiling, done in LArSoft/`wire-cell-pid`) and stored in a
ROOT file. The toolkit pipeline *reads* them and samples immediately:

```
  [ROOT file: pre-tiled TC blobs + optical]
   │
   ├─ UbooneBlobSource × view-combos        (qlport:871)
   │     live: uvw, uv, vw, wu               (qlport:1415)
   │     dead: uv, vw, wu                     (qlport:1419)
   ├─ BlobSetMerge                           → one blob stream
   │
   ▼
  UbooneClusterSource                        ***SAMPLING HAPPENS HERE***
   │  (root/src/UbooneClusterSource.cxx)
   ├─ Aux::sample_live(bs_live,  …)   strategy=["charge_stepped"]   (:586)
   ├─ Aux::sample_dead(iblob, …)      strategy=["center"]            (:600)
   ├─ Aux::add_ctpc(…)                                              (:658)
   └─ Aux::add_dead_winds(…)                                        (:659)
   │
   ▼
  MultiAlgBlobClustering (MABC)  → Bee (.zip)
```

* The **view combinations** are the MicroBooNE analogue of live/dead imaging:
  `uvw` is a normal 3-plane live blob; `uv`/`vw`/`wu` are 2-plane blobs where the
  third plane is dead (so a blob is built from only two views). PDHD/PDVD instead
  carry dead information per-channel via dead blobs + `dead_winds`.
* Sampling lives in an `IClusterSource` (`UbooneClusterSource`), not in
  `PointTreeBuilding`, but it calls the **same** `Aux::sample_live`/`sample_dead`
  library helpers, so the resulting `"3d"`/`"scalar"`/`"corner"` columns match
  PDHD/PDVD.

### 8.2 The `charge_stepped` strategy (live blobs)

`BlobSampler.cxx:901-1438`. This is an enhanced `stepped` that ports
`WCPPID::calc_sampling_points()` from `wire-cell-pid` — i.e. it *is* the original
MicroBooNE sampling. `stepped` (§3.1) is the stripped-down version PDHD/PDVD use;
`charge_stepped` adds charge awareness on top of the identical geometric skeleton:

1. **Same min/max/mid skeleton.** Find the max/min/mid-coverage strips, compute
   `nmin`/`nmax` from `min_step_size=3` and `max_step_fraction=1/12`, and apply the
   same `offset=0.5` diagonal shift to wire-center crossings
   (`BlobSampler.cxx:982-1009,1073-1079`) — identical to `stepped`.
2. **Small-blob densification — `use_all_wires`** (`:1053-1070`). If
   `swidth(smax)·swidth(smin) ≤ max_wire_product_threshold` (default **2500**),
   *every* wire in the min/max views is used instead of the stepped subset. Small
   blobs therefore get a dense, every-wire grid; only large blobs fall back to the
   `nmin`/`nmax` step. The original stepped wire set is still tracked as the
   **"must"** set (`min_wires_set`/`max_wires_set`).
3. **Charge filtering** (`:1134-1224`). For each candidate min×max crossing, the
   per-wire charge is looked up in the slice activity (`get_wire_charge`,
   `:1382-1437`). A crossing is **dropped** unless it is a "must" wire or its wire
   charge clears the threshold (defaults `charge_threshold_max/min/other = 4000`).
   The third ("mid") plane is also charge-gated. Crossings with all-zero charge are
   dropped. This is what makes the MicroBooNE point cloud track real charge instead
   of filling the whole RayGrid shape.
4. **Dead-plane handling — `disable_mix_dead_cell`** (default **true**,
   `:1016-1029`, `is_plane_bad` `:1254-1378`). When on, each of the three planes is
   tested for "bad" (charge uncertainty `> dead_threshold=1e10` on the blob's first
   or last wire); a bad plane's charge threshold is set to **0** so its (absent)
   charge can't veto points. The `(charge != 0 || disable_mix_dead_cell)` clauses
   then let zero-charge wires through on dead planes — i.e. dead cells are *not*
   mixed into the charge requirement. Setting it **false** (the
   `bs_live_no_dead_mix` sampler) lets dead-cell charge participate; that variant is
   used only by the `improve_cluster_2` retiler (§8.4).
5. **Aux** output is the same four fields as `stepped`
   (`max/min_wire_interval`, `max/min_wire_type`, `:1238-1241`).

`charge_stepped` also supports a **runtime config override**
(`sample_blob_with_config` → `apply_runtime_config`, `:944-961,1444-1466`): a
caller can pass per-call `charge_threshold_*`/`disable_mix_dead_cell`/`dead_threshold`
that override the configured values for that one blob. This is how retiling
(§8.4) re-samples improved clusters with different charge settings.

### 8.3 Physics constants

| constant            | PDHD/PDVD            | MicroBooNE (`qlport`)                         |
|---------------------|----------------------|-----------------------------------------------|
| live strategy       | `stepped`            | `charge_stepped`                              |
| `drift_speed`       | 1.6 mm/µs            | **1.101 mm/µs**                              |
| `time_offset`       | −250 µs              | **−1600 µs + 6 mm/drift_speed**             |
| `tick`              | (slice default)      | **0.5 µs** (0.5 mm/tick)                     |
| ticks per slice     | —                    | **4** (`nticks_live_slice`)                  |
| `extra` regex       | split charge fields  | `[".*wire_index", ".*charge.*", "wpid"]`     |

Source: `qlport/uboone-mabc.jsonnet:44-71` (samplers), `:141-145` (DetectorVolumes).
The x-projection formula is the same `x = x_origin + xsign·(t_slice + time_offset)·drift_speed`
(`Aux::time2drift`); only the constants change. Time binning is still the default
`(1,0,1)` → one x-sample per blob at the slice start.

### 8.4 Retiling re-samples (`improve_cluster_2`)

Unlike PDHD/PDVD (which sample once), the MicroBooNE MABC pipeline re-samples
during Steiner-graph construction. `cm.steiner(retiler=improve_cluster_2)`
(`qlport:1244`) hands the clustering an `improve_cluster_2` retiler configured with
the **`bs_live_no_dead_mix`** sampler (`charge_stepped` with
`disable_mix_dead_cell=false`, `qlport:1171-1173`). When a cluster is improved, its
blobs are re-tiled and re-sampled with dead-cell charge allowed to participate —
densifying points across dead regions for the downstream PID/tracking. The
`cut_time_low/high` retiler also exists (`qlport:1167-1169`) but is not wired into
the default pipeline.

### 8.5 Bee output

MicroBooNE's `bee_points_sets` (`qlport:1267-1336+`) differ from PDHD/PDVD's
`img`/`clustering` pair. The `img`/`clustering`/`retiled`/`examine` sets are
commented out; the active sets are dumped **after specific visitors**:

| name           | visitor / grouping       | pcname        | coords                | content                                  |
|----------------|--------------------------|---------------|-----------------------|------------------------------------------|
| `regular`      | `CreateSteinerGraph`     | `3d`          | `x_t0cor,y,z`         | the `charge_stepped` points (scope-filtered) |
| `steiner`      | `CreateSteinerGraph`     | `steiner_pc`  | `x_t0cor,y,z`         | Steiner-tree vertices                    |
| `track_fit`    | `TaggerCheckNeutrino`    | (PRGraph)     | —                     | track-fitted points, dQ/dx colored       |
| `shower_track` | `TaggerCheckNeutrino`    | (PRGraph)     | —                     | points colored by shower/track class     |
| `vertices`     | `TaggerCheckNeutrino`    | (PRGraph)     | —                     | PR-graph vertices                        |
| `mc` (`bee_pf`)| `TaggerCheckNeutrino`    | (PF tree)     | —                     | particle-flow tree (Bee `mc` format)     |

So MicroBooNE's "charge points" Bee view (`regular`) is the `charge_stepped`
`"3d"` cloud — the same point-cloud-to-Bee path as PDHD/PDVD (§6.1), just dumped
after `CreateSteinerGraph` and in `x_t0cor`. Unlike PDHD/PDVD it additionally
ships `steiner_pc`, track-fit, and PID/PF Bee sets, because the MicroBooNE MABC
runs the full neutrino-ID/tracking tail.

### 8.6 Summary: MicroBooNE vs PDHD/PDVD

| aspect              | PDHD/PDVD                          | MicroBooNE (`qlport`)                       |
|---------------------|-----------------------------------|---------------------------------------------|
| imaging             | in WCT (`img.jsonnet`)            | upstream (WCP TC tiling), read from ROOT    |
| sampling node       | `PointTreeBuilding`               | `UbooneClusterSource`                        |
| live strategy       | `stepped` (geometry only)         | `charge_stepped` (geometry + charge + dead) |
| dead handling       | dead blobs + `dead_winds`         | 2-view blob combos (`uv`/`vw`/`wu`)         |
| re-sampling         | none                              | `improve_cluster_2` retiler (no-dead-mix)   |
| constants           | 1.6 mm/µs, −250 µs                | 1.101 mm/µs, −1600 µs +6 mm offset          |
| Bee `3d` view       | `img` + `clustering`              | `regular` (post-SteinerGraph) + PID sets    |

---

## 9. Why PDHD/PDVD use `stepped` (not `charge_stepped`) for clustering

`charge_stepped` is the sampler for the **pattern-recognition / tracking stage**
(MicroBooNE's neutrino-ID and 3-D tracking tail, where a dense, charge-confirmed
point cloud feeds Steiner-tree building and PID). For the **clustering stage**
that PDHD/PDVD run, plain `stepped` is good enough — and was confirmed so
empirically. This section records that evaluation (2026-06-09); `stepped`
remains the default live sampler.

### 9.1 `charge_stepped` is a no-op vs `stepped` on PD (any positive threshold)

Switching the live sampler to `charge_stepped` and sweeping `charge_threshold_*`
on one event per detector (PDVD 039324 evt0, PDHD 027409 evt0) gives a clustering
`"3d"` point cloud that is **point-for-point identical to `stepped`** (same
multiset, 0 added / 0 removed) for every positive threshold tested:

| `charge_threshold`                       | PDVD global pts | PDHD global pts |
|------------------------------------------|-----------------|-----------------|
| 4000 (MicroBooNE default)                | 79 018          | 102 363         |
| 2000                                     | 79 018          | 102 363         |
| 1500 / 1000 / 500 / 200 / 50 / 1         | 79 018          | 102 363         |
| **0**                                    | 129 958 (+65 %) | 111 227 (+9 %)  |
| `stepped` (reference)                    | 79 018          | 102 363         |

Only `charge_threshold ≤ 0` changes the output, and the points it adds are
charge-less (§9.3) — so no positive threshold makes `charge_stepped` differ from
`stepped` on PD.

### 9.2 Mechanism

`charge_stepped` = the `stepped` geometry + a `use_all_wires` densification, gated
by per-wire charge. Two facts make it collapse to `stepped` on PD:

1. The original `stepped` crossings are tagged **"must"** and **bypass the charge
   gate entirely** (`clus/src/BlobSampler.cxx:1195-1196`) — they are never removed.
2. The extra densified crossings land on wires that carry **zero charge on ≥1
   plane**: PD imaging already runs 3-plane `ChargeSolving`, so blobs contain only
   charge-confirmed cells — there is no extra coincident charge for
   `charge_stepped` to recover. Any positive threshold drops all of them.

Runtime instrumentation of the gate (PDVD evt0, threshold 2000) confirms this: the
configured threshold reaches the gate verbatim on all 35 106 sampled blobs, and of
the 6 558 drops at the **final, point-determining** gate **0 (0.000 %)** had
positive charge — every point-affecting drop is a zero-charge phantom crossing.
(Dead planes are handled correctly: a bad plane's threshold is zeroed,
`clus/src/BlobSampler.cxx:1027-1029`.)

### 9.3 The threshold-0 densification is not an improvement

At `charge_threshold = 0` the gate keeps the densified crossings (`0 < 0` is
false). These extra points are **geometric grid-fills inside the existing blob
envelope with no 3-plane charge confirmation** — they fatten tracks rather than
extending or sharpening them. For clustering they add no information and risk
noise, so they are not used.

### 9.4 Conclusion

`stepped` is the default live sampler for the PDHD/PDVD **clustering** chain
(paired with the zero imaging activity threshold, `nthreshold = 1e-6`).
`charge_stepped` belongs to the downstream **pattern-recognition** stage (as in
MicroBooNE) and, on PD, offers nothing over `stepped` for clustering.

---

## Source map

| topic                         | file:lines                                                |
|-------------------------------|-----------------------------------------------------------|
| imaging (no sampling/Bee)     | `pdhd/img.jsonnet:121-207,291`                            |
| sampler config (live/dead)    | `pdhd/clus.jsonnet:109-126`; `pdvd/clus.jsonnet:127-140`  |
| PointTreeBuilding samplers    | `pdhd/clus.jsonnet:165-179`                               |
| bee_points_sets              | `pdhd/clus.jsonnet:236-245,467-498`                       |
| strategy docs                 | `clus/inc/WireCellClus/BlobSampler.h:70-226`             |
| `stepped` impl                | `clus/src/BlobSampler.cxx:703-885`                       |
| `charge_stepped` impl (uBooNE)| `clus/src/BlobSampler.cxx:901-1438`                      |
| uBooNE sampler config         | `qlport/uboone-mabc.jsonnet:44-82,141-145`              |
| uBooNE sampling node          | `root/src/UbooneClusterSource.cxx:579-659`             |
| uBooNE blob views / graphs    | `qlport/uboone-mabc.jsonnet:871-922,1414-1420`         |
| uBooNE retiler / bee sets     | `qlport/uboone-mabc.jsonnet:1167-1173,1267-1336`       |
| `center`/`corner`/`scalar`    | `aux/src/SamplingHelpers.cxx:9-245`                      |
| `time2drift`                  | `aux/src/SamplingHelpers.cxx:247-256`                    |
| `ctpc` / `dead_winds`         | `clus/src/PointTreeBuilding.cxx:272-516`                 |
| Bee charge points             | `clus/src/MultiAlgBlobClustering.cxx:1497-1619`          |
| Bee fill timing (img/clust)   | `clus/src/MultiAlgBlobClustering.cxx:2087-2189`          |
| Bee dead patches              | `clus/src/MultiAlgBlobClustering.cxx:1624-1725`          |
| full reference                | `img/docs/blob-sampling.org`                             |

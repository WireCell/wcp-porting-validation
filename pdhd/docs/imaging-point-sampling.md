# Point sampling in PDHD / PDVD imaging

How 3-D points are produced from imaging blobs, what point clouds exist, and which
ones feed clustering vs. the Bee event display.

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

PDHD and PDVD use identical sampling configuration (`stepped` for live, `center`
for dead, `drift_speed = 1.6 mm/µs`, `time_offset = -250 µs`). They differ only
in geometry: PDHD has 4 APAs in 2 drift groups `{0,2}`/`{1,3}`; PDVD has 8 CRPs in
2 drift groups `{0,1,2,3}` (bottom) / `{4,5,6,7}` (top).

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
are not wired into the production configs.

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
`drift_speed = 1.6 mm/µs` (`clus.jsonnet:7-8,113-114`).

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
| `drift_speed`     | 1.6 mm/µs                  | 1.6 mm/µs                         |
| `time_offset`     | −250 µs                    | −250 µs                          |
| `bee_detector`    | `protodunehd`              | `protodunevd`                     |
| anodes / groups   | 4 APAs, `{0,2}` / `{1,3}`  | 8 CRPs, `{0,1,2,3}` / `{4,5,6,7}` |

Source: `pdhd/clus.jsonnet:7-8,22-25,109-126`;
`pdvd/clus.jsonnet:7-8,23-25,127,135,237`.

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
| `center`/`corner`/`scalar`    | `aux/src/SamplingHelpers.cxx:9-245`                      |
| `time2drift`                  | `aux/src/SamplingHelpers.cxx:247-256`                    |
| `ctpc` / `dead_winds`         | `clus/src/PointTreeBuilding.cxx:272-516`                 |
| Bee charge points             | `clus/src/MultiAlgBlobClustering.cxx:1497-1619`          |
| Bee fill timing (img/clust)   | `clus/src/MultiAlgBlobClustering.cxx:2087-2189`          |
| Bee dead patches              | `clus/src/MultiAlgBlobClustering.cxx:1624-1725`          |
| full reference                | `img/docs/blob-sampling.org`                             |

# nugraph HDF5 test records

A running log of `wclsTensorSetLabeler` nugraph-HDF5 test outputs that are
worth keeping (indicated by the user).  Each entry records the output file
location, the corresponding BEE link, and how the test was produced.

The component and schema are documented in the larwirecell tree:
`larwirecell/aiml/docs/TensorSetLabeler-notes.md` (§2.11 = nugraph HDF5).
Build/run recipes: `sbnd/docs/0-build-...md`, `sbnd/docs/1-run-tests-...md`.

---

## v1 — 10-event corsika+GENIE rockbox (2026-07-16)

- **File:** `sbnd/TensorSetLabeler/nugraph-sample-v1/nugraph.h5` (14 MB, 10 events)
- **BEE (3D `sp` nodes; q = 1 nu / 0 cosmic / −1 ghost; cluster_id = trackid):**
  https://www.phy.bnl.gov/twister/bee/set/97ded3b3-8fbe-4949-871b-cfeb3a14ad37/event/list/

### How it was produced

Local SL7 builds from `/exp/sbnd/app/users/yuhw/opt` (setup-ap.sh), run from
`sbnd/TensorSetLabeler/`:

```bash
lar -n 10 -c wcls-img-clus-matching-xin.fcl \
    -S /exp/sbnd/app/users/yuhw/2025-fall-prod-sample/mc_paths-10files.lst \
    --no-output
# -> nugraph.h5 (moved into nugraph-sample-v1/), plus mabc.zip + trash-all-apa.tar.gz
# BEE from the HDF5:
python3 h5_sp_to_bee.py nugraph-sample-v1/nugraph.h5 nugraph_sp_bee.zip
BROWSER=echo bash ../sbnd_xin/upload-to-bee.sh TensorSetLabeler/nugraph_sp_bee.zip   # from sbnd/
```

- **Input:** MCP2025C corsika+GENIE rockbox reco1, `mc_paths-10files.lst`
  (runs 31/32).
- **Chain:** XIN-faithful imaging + clustering + joint QLMatching
  (`wcls-img-clus-matching-xin.fcl`, `reality=sim`, truth labeler auto-on).

### Build state (commits used; none pushed at record time)

| repo | branch | commit |
|---|---|---|
| larwirecell | `dev-v10_14_02_02` | `190830e` (nugraph HDF5 output) |
| wire-cell-toolkit | `tgm` | `7d0ab7b6` (both-TPC ctpc merge + dv/pcts threading) |
| wcp-porting-img | `main` | `646fb53` (h5_sp_to_bee.py validation script) |

### Contents / configuration notes

- **Container:** pynuml `H5DataModule`; `/datasize [10,0,0]` (all events in
  `train`), one scalar COMPOUND record per event at `/dataset/<sample>`.
- **Truth is EXACT** (SED→blob `trackid`), not point-distance matching.
  `sp/y_semantic` {0 nu, 1 cosmic, −1 ghost}, `sp/y_instance` = trackid.
- **2D `u/v/y` nodes cover BOTH TPCs** (ctpc of both anodes merged at
  `qlmatching.jsonnet matching_joint`).
- **`sp_nexus_sp`** (blob-blob) = intra-cluster blob-center **kNN** fallback:
  the WCT `ctpc` graph flavor throws `map::at` on the `as_pctree()`-restored
  tree (clustering-time maps not reconstructed).
- **`{p}_nexus_sp`** (2D-hit → blob) = TRUE wire/slice-box overlap.
- **`{p}_plane_{p}` intra-plane edges are NOT present** (added downstream in
  post-processing).

### Per-event stats

| # | run/subrun/event | sp | nu / cosmic / ghost | u/v/y 2D | sp-sp |
|---|---|---|---|---|---|
| 0 | 31/88/12 | 5903 | 162 / 5184 / 557 | 1553/1434/1429 | 20056 |
| 1 | 31/88/5  | 2167 | 282 / 1532 / 353 | 580/576/463 | 7573 |
| 2 | 32/10/10 | 2454 | 156 / 1654 / 644 | 541/551/437 | 8618 |
| 3 | 32/10/13 | 5584 | 547 / 2760 / 2277 | 912/909/696 | 19966 |
| 4 | 32/10/14 | 4438 | 143 / 3429 / 866 | 1147/1068/1012 | 14990 |
| 5 | 32/10/16 | 5832 | 240 / 5282 / 310 | 1747/1683/1564 | 19354 |
| 6 | 32/10/21 | 2120 | 201 / 1664 / 255 | 532/637/503 | 7337 |
| 7 | 32/10/39 | 3405 | 168 / 2832 / 405 | 875/946/762 | 11508 |
| 8 | 32/10/43 | 6284 | **0** / 4805 / 1479 | 1399/1513/1257 | 21702 |
| 9 | 32/10/6  | 4031 | 6 / 3894 / 131 | 1258/1195/1121 | 13489 |

Totals: **1905 nu / 33036 cosmic / 7277 ghost** blobs.
Note evt 32/10/43 has 0 nu blobs — its in-detector interaction deposits
only ~2.8 MeV and its `nu_idx 0` interaction is out-of-TPC rock (see the
`nu_edep` array in the event metadata).

---

## v2 — sim + data ("reality" mode), 10 events each (2026-07-16)

First test of the `wclsTensorSetLabeler` `reality="data"` mode (input-only
HDF5 for inference on real data) alongside the unchanged sim path.  Workdir
`sbnd/TensorSetLabeler/nugraph-sample-v2/`.

### Files

| | HDF5 | events |
|---|---|---|
| MC (sim) | `nugraph-sample-v2/nugraph-mc-10evt.h5` (14 MB) | 10 |
| data     | `nugraph-sample-v2/nugraph-data-10evt.h5` (17.7 MB) | 10 |

(1-event smokes also kept: `nugraph-{mc,data}-1evt.h5`.)

### BEE

| view | MC | data |
|---|---|---|
| reco Bee (clus.jsonnet `mabc.zip`) | https://www.phy.bnl.gov/twister/bee/set/f2040249-8afc-4c2a-a083-63191f8ddfd6/event/list/ | https://www.phy.bnl.gov/twister/bee/set/cc83735e-2b23-47b4-8735-e931cc0bca65/event/list/ |
| nugraph HDF5 `sp` nodes (`h5_sp_to_bee.py`) | https://www.phy.bnl.gov/twister/bee/set/f8e30646-edc0-4907-954b-17e0e6b69717/event/list/ | https://www.phy.bnl.gov/twister/bee/set/d3c0b9b1-8acd-4579-b81f-9ac130768d98/event/list/ |

- reco Bee MC has the truth sets (`truth_trackid_labeled`/`truth_unlabeled`/
  `sed-*`/`mc`) + reco sets; data has ONLY the reco sets (labeler adds no Bee
  in data mode) — the gating check.
- nugraph `sp` MC coloured by truth (q = 1 nu/0 cosmic/-1 ghost, cluster_id =
  trackid); data coloured by reco (q = charge, cluster_id = reco_cluster_id)
  since there is no truth.

### How it was produced

```bash
cd sbnd/TensorSetLabeler/nugraph-sample-v2
# MC (truth):
lar -n 10 -c wcls-img-clus-matching-xin.fcl \
    -S /exp/sbnd/app/users/yuhw/2025-fall-prod-sample/mc_paths-10files.lst --no-output
# data (input-only, reality=data):
lar -n 10 -c wcls-img-clus-matching-xin-data.fcl \
    -s /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/samples/filtered-reco1/\
data_filtered_decoded_reco1-fe6033f3-07a0-4971-cea5-16ce59269fba_eventidfiltered.root --no-output
# BEE from the HDF5 sp nodes (auto: truth colours for sim, reco-cluster for data):
python3 ../h5_sp_to_bee.py nugraph-mc-10evt.h5   nugraph-mc-bee.zip
python3 ../h5_sp_to_bee.py nugraph-data-10evt.h5 nugraph-data-bee.zip
```

### What's new vs v1 (the `reality` option)

- **wclsTensorSetLabeler `reality`** (default "sim"): "data" writes ONLY the
  RSE into the ITensorSet metadata, NO Bee sets, NO `truth_per_track`, and an
  INPUT-ONLY HDF5 (same 32-field schema; reco features real, truth fields =
  sentinels `y_semantic=-1`/`y_instance=-1`/`vtx_*=-1/0`/`edge_y=labelable=0`).
- **Reco-reality grouping** in `clus.jsonnet` (`reco` local keyed by reality):
  sim → use_sce=true, pos_offset_on=false; data → use_sce=false,
  pos_offset_on=true.  The labeler pseudo-sim is independent of this.
- The data fcl keeps the `wclsTensorSetLabeler` inputer (reads RSE only).

### Verification

- MC: 10 records, keys 31/88/12 … 32/10/6, truth totals 1905 nu / 33036
  cosmic / 7277 ghost.  Same as v1 (grouping didn't change sim).
- data: 10 records, real data keys 18253/1/172230 … 18255/1/90055, all
  `y_semantic=-1`, `evt/y=0`, real reco features (charge, reco_cluster_id),
  both TPCs (u/v/y ≈ per-event hundreds), edges present & unlabelable.
  Identical 32-field schema to MC, no `{p}_plane_{p}` edges.

### Build state (commits; none pushed)

| repo | branch | commit |
|---|---|---|
| larwirecell | `dev-v10_14_02_02` | reality sim/data mode |
| wire-cell-toolkit | `tgm` | reco-reality grouping + labeler-both + reality thread |
| wcp-porting-img | `main` | entry jsonnet + data fcl + h5_sp_to_bee data mode + this record |

---

## v3 — real data + frameshift, 48 events (2026-07-21)

Full real-data run: first the sbndcode FrameShift job, then the WCT data
chain (`reality=data`) on the `_frameshift.root`.  Workdir
`sbnd/TensorSetLabeler/data-frameshift/`.

### Files

- Frameshift input file (48 events):
  `sbnd/samples/filtered-reco1/data_filtered_decoded_reco1-fe6033f3-07a0-4971-cea5-16ce59269fba_eventidfiltered_frameshift.root`
  (produced by `run_frameshift.fcl`; see `sbnd/samples/docs/gen2-data-frameshift.md`).
- HDF5 (input-only, data mode): `data-frameshift/nugraph-data-frameshift-48evt.h5` (92 MB, 48 records).

### BEE

reco Bee (clus.jsonnet `mabc.zip`, 48 events):
https://www.phy.bnl.gov/twister/bee/set/93907c09-4d67-4cd1-902c-4246b8e18820/event/list/

### How it was produced

```bash
# 1) frameshift (sbndcode; SL7 + setup-local-opt.sh):
lar -c run_frameshift.fcl -s <..._eventidfiltered>.root      # -> ..._frameshift.root
# 2) WCT data chain on the frameshift file (SL7 + setup-ap.sh), 48 events:
cd sbnd/TensorSetLabeler/data-frameshift
lar -n 48 -c wcls-img-clus-matching-xin-data.fcl -s <..._frameshift>.root --no-output
BROWSER=echo bash ../../sbnd_xin/upload-to-bee.sh TensorSetLabeler/data-frameshift/mabc-data-frameshift-48evt.zip
```

### Verification

- **Gen2 data frameshift**: for all Gen2 real data, run `run_frameshift.fcl`
  first (reminder: `sbnd/samples/docs/gen2-data-frameshift.md`).
- **reality=data reco config confirmed** (compiled with wcsonnet):
  `ClusteringSwitchScope` correction = **T0Correction**, scope **x_t0cor**
  (`use_sce=false`); **`pos_offset_on=true`** (nonzero `[0,±1.1,∓6.7]` shifts).
- Labeler in DATA mode all 48 events (RSE-only metadata, no Bee, input-only
  HDF5).  The WCT chain reads the same sptpc2d/opflash products (copied
  through by the frameshift RootOutput); it does not itself consume the new
  frameshift product (persisted for downstream timing use).

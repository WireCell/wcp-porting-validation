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

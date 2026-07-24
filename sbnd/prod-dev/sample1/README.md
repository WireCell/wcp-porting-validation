# prod-dev / sample1 — 10-event MC + data (img + clus + Q/L matching)

10 events each of MC and real data through the 1-step larwirecell
imaging + clustering + joint Q/L matching chain
(`wcls-img-clus-matching-xin[-data].fcl`), for production-development review.

## Contents

```
sample1/
  mc/    mabc.zip, trash-all-apa.tar.gz, nugraph.h5, run.log   (MC,  reality=sim)
  data/  mabc.zip, trash-all-apa.tar.gz, nugraph.h5, run.log   (data, reality=data)
```

## Inputs

- **MC**: first 10 events of the 100-file list
  `2025-fall-prod-sample/round2-patrec/mc_paths-v10_14_02_03-100files.lst`
  (SAM def `aurora_SBND2026A_gen2_BNBLight_prodgenie_corsika_proton_rockbox0p1_sbnd_CV_v10_14_02_03_reco1_sbnd`).
- **data**: first 10 events of
  `2025-fall-prod-sample/round2-patrec/data_MCP2025C_reco1_frameshift_first1000ev.root`
  — the first 1000 events of the data 100-file list
  (`data_MCP2025C_Fall25-Run1_BNB_FixedDev_bnblight_v10_14_02_reco1_sbnd`) with
  the **FrameShift** product added via `run_frameshift.fcl` (Gen2 real data
  must be frameshifted first; see `sbnd/samples/docs/gen2-data-frameshift.md`).

## How it was generated

SL7 apptainer + `sbnd/setup-ap.sh`, with `sbnd/` on `FHICL_FILE_PATH`:

```bash
# MC (reality=sim; truth labeler on)
cd prod-dev/sample1/mc
lar -n 10 -c wcls-img-clus-matching-xin.fcl \
    -S .../round2-patrec/mc_paths-v10_14_02_03-100files.lst --no-output

# data (reality=data; input-only nugraph, no truth Bee)
cd prod-dev/sample1/data
lar -n 10 -c wcls-img-clus-matching-xin-data.fcl \
    -s .../round2-patrec/data_MCP2025C_reco1_frameshift_first1000ev.root --no-output

# upload
BROWSER=echo bash sbnd/sbnd_xin/upload-to-bee.sh prod-dev/sample1/mc/mabc.zip
BROWSER=echo bash sbnd/sbnd_xin/upload-to-bee.sh prod-dev/sample1/data/mabc.zip
```

- `reality` selects the grouped reco config: MC `sim` → `use_sce=true`,
  `pos_offset_on=false`; data `data` → `use_sce=false`, `pos_offset_on=true`.
- MC `mabc.zip` carries reco sets (`clustering`/`img`/`op`/`tgm`) + the
  labeler truth sets (`truth_trackid_labeled`/`truth_unlabeled`/`sed-*`/`mc`);
  data `mabc.zip` has reco sets only (labeler adds no Bee in data mode).
- Each run also writes `nugraph.h5` (truth in MC, input-only in data) — see
  `sbnd/TensorSetLabeler/docs/nugraph-hdf5-tests.md`.

## BEE

| | link |
|---|---|
| MC   | https://www.phy.bnl.gov/twister/bee/set/f12193e9-f33a-4287-8a0d-e152994c5add/event/list/ |
| data | https://www.phy.bnl.gov/twister/bee/set/008f3c7a-359f-46b3-8bd0-f52b421914c3/event/list/ |

(10 events each, generated 2026-07-22.)

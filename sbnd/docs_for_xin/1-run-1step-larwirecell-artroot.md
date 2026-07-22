# 1-step: one larwirecell job over artROOT (img + clus + Q/L matching)

A single `lar` job runs the whole WireCell chain (imaging → per-APA clustering
→ joint charge-light matching → all-APA clustering) via the `WireCellToolkit`
art module reading the reco1 artROOT products.  Output is a Bee zip
(`mabc.zip`).

## Environment

Inside the SL7 apptainer, once per shell:

```bash
source /nashome/y/yuhw/.bashrc
source /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/setup-ap.sh
# the entry fcls live in sbnd/ -> put it on the fcl search path:
export FHICL_FILE_PATH=/exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd:$FHICL_FILE_PATH
```

(The full container invocation is in
`sbnd/docs/1-run-tests-sl7-local-builds-sbnd.md`.)  Run from an output
directory — `mabc.zip` and a `trash-all-apa.tar.gz` land in the cwd.  Remove a
stale `mabc.zip` before each run (it is opened once per job).

## sim (MC)

```bash
lar -n 1 -c wcls-img-clus-matching-xin.fcl \
    -s <mc_reco1>.root --no-output
# or over a list:  -S <mc_paths>.lst
BROWSER=echo bash /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/sbnd_xin/upload-to-bee.sh mabc.zip
```

- `wcls-img-clus-matching-xin.fcl` sets `reality="sim"` and the MC signal-
  processing product tags (`simtpc2d:*`).  The truth labeler runs (adds
  neutrino/truth metadata, a `truth_per_track` tensor, per-blob trackid, the
  `truth_*`/`sed-*`/`mc` Bee sets, and a nugraph HDF5).

## data (real)

**First: frameshift.**  For Gen2 real data, if the input artROOT file does not
already carry the FrameShift product, produce it:

```bash
lar -c run_frameshift.fcl -s <decoded_reco1_data>.root     # -> <...>_frameshift.root
```

(see `sbnd/samples/docs/gen2-data-frameshift.md`).  Then run the data chain on
the `_frameshift.root`:

```bash
lar -n 1 -c wcls-img-clus-matching-xin-data.fcl \
    -s <..._frameshift>.root --no-output
BROWSER=echo bash /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/sbnd_xin/upload-to-bee.sh mabc.zip
```

- `wcls-img-clus-matching-xin-data.fcl` `#include`s the sim fcl and overrides
  `reality="data"` + the data product tags (`sptpc2d:*`).  `reality="data"`
  selects `use_sce=false` (T0-corrected reco scope) + `pos_offset_on=true`
  (per-TPC calibration).  The labeler runs in data mode: RSE metadata only
  (no truth Bee), plus an input-only nugraph HDF5.

## Output

- `mabc.zip` — Bee display: `img`, `clustering-{apa0,apa1,global}`, `op`,
  `tgm` (+ the labeler's truth sets in sim).
- `trash-all-apa.tar.gz` — the tensor output (labeled pctree + metadata).
- `nugraph.h5` — the nugraph heterogeneous graph (truth in sim, input-only in
  data); see `sbnd/TensorSetLabeler/docs/`.

## Toggles

| toggle | where | meaning |
|---|---|---|
| `reality` | fcl `params` (`"sim"`/`"data"`) | grouped reco config (use_sce, pos_offset) + labeler truth on/off |
| `run_labeler` | entry jsonnet (`true`) | insert the wclsTensorSetLabeler after the all-APA MABC |
| `enable_downstream_pr` | toolkit `pgrapher/experiment/sbnd/clus.jsonnet` | full pattern-rec tail; keep FALSE for bulk |

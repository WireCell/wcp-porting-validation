# SBND Test

## Setup

```bash
source setup-local-opt.sh
```

## Run

```bash
rm -rf data-sep
time lar --nskip 0 -n 1 -c wcls-img-clus-matching.fcl -s standalone-sample/2025f-mc.root --no-output >& wcls-img-clus-matching.log
```

## Upload to Bee

```bash
./bee-upload.sh
```

## Xin's approach (full WCT-native img + clus + QL matching)

A wcls job that reads the artROOT directly (`wclsCookedFrameSource` charge +
`wclsOpFlashSource` light) but otherwise faithfully follows Xin's standalone
chain (`sbnd_xin/wct-clus-matching-standalone.jsonnet` + `wct-img-all.jsonnet`),
using the in-tree canonical toolkit modules (nothing in `sbnd_xin` is modified):

- imaging: toolkit `img.jsonnet` `multi-3view` with `full_deghost=true`
  (live + `multi_masked_2view` dead, recovers charge across dead W channels);
- clustering: toolkit `clus.jsonnet` `per_apa` (`rse_from_ident=true`);
- matching: canonical `FlashTensorToOpticalPCs` + **joint** `QLMatching`
  (`WireCellMatch`) feeding the `premerged` all-APA clustering;
- output: ONE shared Bee zip `mabc.zip` for every MABC node.

Config: `wcls-img-clus-matching-xin.{fcl,jsonnet}`.

Setup — `setup-ap.sh` builds on `setup-local-opt.sh` but prepends the toolkit
cfg (so the toolkit img/clus/qlmatching/simparams win, Xin's environment) plus
the photodet dir (so `QLMatching` finds `semi-analytical-sbnd.json`):

```bash
source setup-ap.sh
```

Run:

```bash
rm -f mabc.zip
time lar --nskip 0 -n 1 -c wcls-img-clus-matching-xin.fcl -s standalone-sample/2025f-mc.root --no-output >& wcls-img-clus-matching-xin.log
```

Upload `mabc.zip` to Bee (Xin's direct uploader; prints the event-list URL):

```bash
BROWSER=echo bash ../sbnd_xin/upload-to-bee.sh mabc.zip
```

# SBND Test

Default workflow: imaging + clustering + charge/light (Q/L) matching (+ optional
pattern recognition) run as a wcls job reading the artROOT directly
(`wclsCookedFrameSource` charge + `wclsOpFlashSource` light), faithfully
following Xin's standalone chain but using the in-tree canonical toolkit modules
(nothing in `sbnd_xin` is modified):

- imaging: toolkit `img.jsonnet` `multi-3view` with `full_deghost=true`
  (live + `multi_masked_2view` dead, recovers charge across dead W channels);
- clustering: toolkit `clus.jsonnet` `per_apa` (`rse_from_ident=true`);
- matching: canonical `FlashTensorToOpticalPCs` + **joint** `QLMatching`
  (`WireCellMatch`) feeding the `premerged` all-APA clustering;
- output: ONE shared Bee zip `mabc.zip` for every MABC node.

Config: `wcls-img-clus-matching-xin.{fcl,jsonnet}`.

## Setup

`setup-ap.sh` builds on `setup-local-opt.sh` but prepends the toolkit cfg (so the
toolkit img/clus/qlmatching/simparams win, Xin's environment) plus the photodet
dir (so `QLMatching` finds `semi-analytical-sbnd.json`):

```bash
source setup-ap.sh
```

## Run

```bash
rm -f mabc.zip
time lar --nskip 0 -n 1 -c wcls-img-clus-matching-xin.fcl -s standalone-sample/2025f-mc.root --no-output >& wcls-img-clus-matching-xin.log
```

Downstream pattern recognition (tagger / steiner / vertices / mc particle-flow)
is toggled by `enable_downstream_pr` at the top of the toolkit
`pgrapher/experiment/sbnd/clus.jsonnet`:

- `true`  — full pattern recognition;
- `false` — matching-only all-APA MABC (use for bulk runs; full patrec has
  data-dependent failures on some events — see `cm-2606/STATUS-xin-chain.md`).

## Upload to Bee

```bash
BROWSER=echo bash ./sbnd_xin/upload-to-bee.sh mabc.zip
```

Prints the event-list URL.

---

The old larwirecell-`wclsQLMatching` chain has been retired to `obsolete/`
(`obsolete/wcls-img-clus-matching.{fcl,jsonnet}`); it was run with
`setup-local-opt.sh` and uploaded via `./bee-upload.sh`.

# TGM tagger validation

Validation of the ported Through-Going-Muon tagger
(`wire-cell-toolkit/clus/src/TaggerCheckTGM.cxx`, a port of WCP
`pid/src/Cosmic_tagger.h::check_tgm`).

## What runs

`wcls-img-clus-matching-xin.fcl` (the default img + clustering + Q/L-matching
chain) now ends the all-APA MABC pipeline with:

```
... -> ClusteringExamineBundles -> MakeFiducialUtils -> TaggerCheckTGM
```

- `MakeFiducialUtils` attaches a `FiducialUtils` carrying the overall FV **box**
  (`BoxFiducial:all-overall-fv`, the `dvm.overall` active-volume bounds shrunk
  by the per-face margins) to the live grouping.
- `TaggerCheckTGM` (debug ON) flags every live cluster whose two ends both exit
  the SCE-corrected FV box, and writes a per-point charge array `tgm_charge`
  (endpoints=10000, body=100) into the `tgm_debug` PC of each tagged cluster.
- MABC dumps the `tgm` Bee point set (algorithm `tgm-global`) reading that
  charge array, so only tagged tracks appear, endpoints highlighted.

All wiring is in `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (`clus_all_apa`)
and the `tagger_check_tgm` / `tagger_check_stm` helpers in
`cfg/pgrapher/common/clus.jsonnet`.

## Run

```bash
# inside SL7, after `source sbnd/setup-ap.sh`
# 1 event smoke test (local sample):
lar -n 1 -c wcls-img-clus-matching-xin.fcl -s standalone-sample/2025f-mc.root --no-output
# N events from the prod list (corsika cosmics -> lots of TGM):
lar -n 100 -S /exp/sbnd/app/users/yuhw/2025-fall-prod-sample/mc_paths-10files.lst \
    -c wcls-img-clus-matching-xin.fcl --no-output
```

The shared Bee zip is `mabc.zip`.

## Analyse

```bash
# cumulate endpoints/body across all events -> npz
python3 tgm-validation/summarize_tgm.py mabc.zip -o tgm-validation/tgm_points.npz
# static XY/XZ/YZ sanity plots (endpoints should outline the FV box)
python3 tgm-validation/analyze_tgm.py tgm-validation/tgm_points.npz \
    -o tgm-validation/tgm_views.png --body
# interactive 3-panel Bokeh viewer (per-event navigation)
./tgm-validation/serve_tgm_viewer.sh 5011 tgm-validation/tgm_points.npz
#   ssh -L 5011:localhost:5011 <user>@<build node>  ; open http://localhost:5011/tgm_viewer
```

## Expectation

In SCE-corrected true space the FV boundary is a box, so the cumulated TGM
**endpoints** populate the box faces: each of XY / XZ / YZ should look like the
outline of a rectangle (the dashed FV box), with the track bodies (grey)
crossing straight through it.

## Notes / caveats

- SCE is currently a no-op (no `sce_field` in the SBND DetectorVolumes
  metadata), so the box test runs on `x_t0cor` points.  `TaggerCheckTGM` already
  calls `SCECorrection::forward(p, t0=0, ...)`, so it becomes correct
  automatically once an `ISCEField` is wired.
- The WCP `check_neutrino_candidate` topology veto is not ported; all flashes
  are treated as normal (type != 2).  See the header of `TaggerCheckTGM.cxx`.

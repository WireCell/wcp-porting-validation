# SBND SCE Correction — WCT 0.36 migration & validation

Migration of the SBND Space-Charge-Effect (SCE) correction from the patched WCT 0.33
pipeline (which used a `PCTransforms` reflection trick) into a clean, WCT-native
`IPCTransform` for 0.36, validated against the per-TPC SCE TH3 displacement maps.

## What

- New class `WireCell::Clus::SCECorrection` — an `IPCTransform` that applies T0
  and the per-TPC SCE displacement maps in one step:
  `x_sce = x_t0 + sign * dx_map(x_t0, y, z)`, apa0 → East TH3, apa1 → West TH3.
- Reads the SBND dualmap
  (`SCEoffsets_SBND_E500_dualmap_CV_voxelTH3.root`, `TrueBkwd_Displacement_X_{E,W}`)
  via DV metadata `sce_map_file` (axes in cm, sign default −1).
- Wired through the existing `PCTransformSet` and `ClusteringSwitchScope`
  (`correction_name="SCECorrection"`); merged-APA clustering uses
  `common_sce_coords = ["x_sce","y","z"]`.
- Toolkit code lives on branch `sbnd-sce-correction-036` off Haiwang's `match`.
  Upstream PR to `WireCell/wire-cell-toolkit` is pending the `clus → root`
  cleanup (the `TFile`/`TH3F` machinery belongs in the `root/` subpackage so
  `clus` stays ROOT-free).

## Result

50 crossing-muon events (apacross 3 GeV detsim), 228 404 paired 3D points,
per-point comparison of reco displacement against the TH3 map prediction:

| Quantity                              | Value                |
|---------------------------------------|----------------------|
| residual (reco − map) rms             | E 1.8 µm,  W 2.2 µm  |
| residual max                          | 8.4 µm               |
| East mean&#124;Δx&#124;               | 0.4021 cm            |
| West mean&#124;Δx&#124;               | 0.5338 cm            |
| **pooled W/E**                        | **1.327**            |
| reference (map volume-avg)            | 1.276                |
| reference (0.33-era reco)             | 1.271                |

The map application is exact to interpolation precision (sub-µm rms vs the
~5000 µm signal). The track-sampled aggregate ratio is consistent with the
volume-averaged map reference; the small offset (~4 %) is the crossing-muon
trajectories preferentially sampling the high-displacement mid-drift region.

Plots: `pics/01_residual.png` (headline, residual histogram),
`pics/02_dx_vs_drift.png` (slide-style 2D Δx vs drift, per TPC),
`pics/03_profile_ratio.png` (drift profile + W/E ratio panel).

## Reproduce

```bash
# Env (FNAL SBND gpvm, SL7 container, our 0.36 + SCECorrection install)
export SCE_TOP=/exp/sbnd/data/users/abhat/wct_sce
source $SCE_TOP/restore_sce_env_036_sce.sh

cd $SCE_TOP/sce_test
export WIRECELL_PATH="$SCE_TOP/sce_test:$WIRECELL_PATH"
export FHICL_FILE_PATH="$SCE_TOP/sce_test:$FHICL_FILE_PATH"
lar -n 50 -c wcls-img-clus.fcl \
    -s $SCE_TOP/validation_crossing_muons/chain_v3/apacross_3gev_detsim_50.root \
    --no-output

python3 make_validation_plots.py
cat validation_plots/SUMMARY.txt
```

## Files

- `SCECorrection.h` — the new `IPCTransform` (header-only v1).
  Destination in WCT toolkit: `clus/inc/WireCellClus/SCECorrection.h`.
- `PCTransforms.cxx.patch` — 4-line registration in `clus/src/PCTransforms.cxx`.
- `wscript_build.patch` — adds `ROOTSYS` to clus link list (will move to `root/`
  subpackage in the upstream PR).
- `clus.jsonnet` — patched copy of canonical `../clus.jsonnet`; the SCE wiring
  is: `common_sce_coords` local, `sce_map_file` in DV metadata, all-APA coords
  set to `common_sce_coords`, `switch_scope(correction_name="SCECorrection")`,
  an extra `bee_points_set` named `sce` (dumps `x_sce` for validation), and
  `use_config_rse: false` so per-event folders survive multi-event runs.
- `clus.jsonnet.patch` — diff against canonical `../clus.jsonnet`.
- `wcls-img-clus.fcl` — symlink to canonical `../wcls-img-clus.fcl`.
- `make_validation_plots.py` — pairs `img`/`sce` BEE points by `(y,z,q)`,
  re-evaluates the TH3 maps per point, produces the three figures.
- `pics/` — validation PNGs.
- `SUMMARY.txt` — headline numbers.

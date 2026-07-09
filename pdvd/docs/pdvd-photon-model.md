# PDVD photon model — what the simulation uses, and the light model for Q/L matching

Investigation (2026-07-09) answering: *what optical-photon model does the PDVD
simulation use, can we calibrate a semi-analytical model for PDVD QLMatching
from the existing photon library, or should we use the library directly?*

**Answers, in one paragraph.**  The official PDVD simulation does **not** use
the semi-analytical model (PDHD's choice) and does not use a classic voxel
photon library either: dunesw's nominal PDVD optical fast-sim is
**`PDFastSimANN`**, a TensorFlow MLP ("computable graph") that maps a 3D
position to the 40 per-OpDet visibilities.  A coarse (25³) voxel library
exists as a disabled fallback.  Both were built on the **v4** geometry whose
Arapuca layout turns out to be **y-mirrored relative to the as-built detector**
(v5); the raw-data opdet positions match v5 exactly.  We therefore sampled the
**v5 ANN** on a 10 cm grid, wired a knob-gated **gridded-library backend into
QLMatching** (`light_model: "library"`), and *also* calibrated the WCT
semi-analytical form against the same ANN as a fallback/cross-check
(`semi-analytical-pdvd.json`).  The fit quality (cathode XAs ~±13% but PMTs
~±40% and membrane XAs needing an unported lateral branch) is exactly why the
library backend is the recommended primary.

Everything below is reproducible from `pdvd/photlib/` (wcp-porting repo).

## 1. What the PDVD simulation uses

dunesw `v10_05_00d00` (the release of our MC workflow docs; models unchanged
in later releases):

| chain | optical config | model |
|---|---|---|
| `protodunevd_g4_stage2.fcl` (stock, nominal) | `protodune_vd_pdfastsim_ann_ar` → `protodune_vd_v4_pdfastsim_ann_ar` | **PDFastSimANN**: TFLoaderMLP, `PhotonPropagation/ComputableGraph/protodune_vd_v4_128nm_tf2.6` (Ar); `..._175nm` (Xe 10 ppm) |
| `protodunevd_refactored_g4_stage2.fcl` (our DNN-ROI training chain) | `#PDFastSim: @local::protodune_pdfastsim_pvs` **commented out** | no light at all — training MC stops at `IonAndScint` |
| `protodunevd_refactored_g4_stage2_pureAr.fcl` | `protodunevd_v4_Ar_photonvisibilityservice` | **PDFastSimPVS** voxel library `libext_protodunevd_v4_Ar_Baseline_v09_69_00d00_5e7_25x25x25_landau_20231216.root` (1 m RSL, 20 m AbsL, reflectivities ON); Xe variant alongside |

Definitions: `duneopdet/v10_05_00d00/fcl/PDFastSim_dune.fcl:236-259`,
`photpropservices_dune.fcl:534-556`.  Files on StashCache cvmfs
(`/cvmfs/dune.osgstorage.org/.../PhotonPropagation/{ComputableGraph,LibraryData}/`),
which also carries **v5** graphs `protodune_vd_v5_{128,175}nm_tf2.6`.
For contrast, PDHD's stock chain runs `PDFastSimPAR` (the semi-analytical
model) in the active volume + a PVS library for the external buffer.

The voxel library: TTree `PhotonLibraryData(Voxel,OpChannel,Visibility)`,
**40 OpChannels**, 25×25×25 voxels over the cryostat bounding box
(790×854.8×854.8 cm centered at world (20, 0, 149.65) cm ⇒ **31.6/34.2/34.2 cm
voxels** — very coarse), x-fastest voxel ids.

## 2. Geometry finding: v4 is Arapuca-mirrored; the data is v5

Cross-matching the library's hot-voxel centroids, the v4/v5 GDML
`volOpDetSensitive_*` world positions, the official channel map
(`duneprototypes .../PDVD_PDS_Mapping_v04152025.json`), and our
`pdvd-opdet-geom.json` (raw-data TTree positions) gives
(`extract_photlib.py`, gates all pass):

- **Raw-data opdet positions ≡ v5 GDML (`protodunevd_v5_ggd`), max Δ = 0.0 mm**
  over all 36 live channels — same world frame, no offsets.
- **v4 vs as-built: all 16 Arapucas are y-mirrored** (cathode y-sets are exact
  mirrors ± 3.5–7.5 cm as-built shifts; membrane pairs swap walls), the v4
  cathode row even carries a y=+205.65 cm duplicate (a GDML typo where +297.85
  is expected).  Several PMTs moved up to ~37 cm (TCO coated PMTs z
  381→409 cm).
- Consequently **any light model for real data must come from v5**, not from
  the v4 library or v4 ANN.
- The **v5 ANN channel order = our flash-chain OpDet order (identity)** for
  all 36 live channels; dead = {24, 27, 28, 34} exactly as in data
  (`sample_ann.py`).

**Open item (hardware-level):** jjo's DAPHNE→module assignment and the
April-2025 official PDS map disagree by the y-mirror partner for the Arapucas
(e.g. offline 1020/1021 = module at y=+124 cm per jjo vs −131 cm per the
official map read against v5 positions); PMTs agree exactly.  Our whole light
chain is self-consistent under jjo's map, but before production Q/L results
the pairing should be confirmed **from data**: cathode-crosser events, charge
cluster (y, z) vs which cathode XA lights up — a mirror swap is a maximal-
contrast signature.

## 3. ANN ↔ voxel library consistency (v4 vs v4)

The v4 128 nm ANN evaluated at the 15 000 filled voxel centers against the Ar
voxel library: **log-visibility correlation 0.991, ANN/library ratio
16/50/84% = 0.78/0.99/1.17** (`sample_ann.py checkv4`).  The ANN is a faithful
smooth of the Geant4-built library (which is MC-noisy at 5×10⁷ photons/voxel),
also validating input units (cm), world frame, and channel order.  Ar↔Xe
libraries correlate at 0.944 in log-visibility.

## 4. The shipped model: v5 ANN sampled onto a grid

`sample_ann.py` samples `protodune_vd_v5_128nm_tf2.6` (and `175nm` for
Xe-doped running) on a **10 cm grid over the active volume**
(71×69×33 nodes × 40 channels, f32 ≈ 26 MB), written by
`export_wct_photlib.py` to

    wire-cell-data/pdvd/photodet/pdvd-photlib-vis-v5-128nm.{json,npy}   (+175nm)

The meta JSON carries grid origin/step/shape and 32 random self-check points
with python-computed trilinear values.  Confirmed physics: cathode XAs are
**double-sided** (visibility symmetric across x=0), cathode opacity is encoded
(negligible cross-cathode visibility for membrane/PMT channels).

## 5. QLMatching library backend (toolkit)

New `match/PhotonLibraryModel` (meta JSON + npy via cnpy, trilinear
interpolation, boundary clamp) selected by two new QLMatching knobs:

```jsonnet
light_model: 'library',            // default 'semi' -> existing path untouched
photon_library_file: 'pdvd/photodet/pdvd-photlib-vis-v5-128nm.json',
```

In library mode the `semimodel_file` is still loaded (it supplies the OpDet
table used by masks and cathode-side logic) but per-point visibilities come
from the library; the reflected term is 0 (the ANN visibility is total photon
arrival) and **no same-TPC x-sign gate** is applied — the library itself
encodes cathode opacity and the double-sided cathode XAs (the semi path's
gate at `SemiAnalyticalModel.cxx` would wrongly blank the cathode XAs for
x<0 points).

Verification: `match/test/doctest_photonlibrarymodel.cxx` (synthetic-grid
exactness + clamping + masking, and `PHOTLIB_META=... ` runs the 32
self-check points of the real file at ≤1e-6); default-OFF byte-identity A/B
on the PDHD matching chain (see §7).

`cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet` (skeleton, NOT
production) collects the PDVD constants: nchan 40, `active_opdet_types
[0, 1]`, ch_mask = dead {24,27,28,34} + Ar-blind {16,29,32,39} (no/quenched
WLS at 128 nm; unmask with the 175 nm library for Xe running), per-OpDet
`VUVEfficiency` from the official map (XA 0.03, TPB PMT 0.12, PEN PMT 0.036).

## 6. Semi-analytical calibration (fallback / cross-check)

`fit_semi_analytical.py` replicates the exact WCT formula (rectangle/dome
solid angles, 10°-binned Gaisser-Hillas, Norm border correction, λ fixed at
2000 cm = the sim absorption length; Rayleigh migration absorbed by GH as in
LArSoft) and fits it per PD group to the v5 128 nm ANN samples.  pred/ANN
ratio 16/50/84%:

| group | ratio | usable in current C++? |
|---|---|---|
| cathode XAs (8) | 0.88 / 0.99 / 1.13 | yes (`GH_PARS_flat`), **needs the x=0 both-sides gate fix** |
| bottom PMTs (16) | 0.75 / 1.01 / 1.46 | yes (`GH_PARS_dome`) |
| TCO PMTs (8) | 0.68 / 1.03 / 1.80 | poorly — table kept under `_GH_PARS_pmt_tco`; suggest masking in semi mode |
| membrane XAs (8) | 0.82 / 0.97 / 1.32 | **no** — port fixes `cosine=|Δx|/d` (orientation-0), wrong/divergent for y-normal PDs; correct-physics table under `_GH_PARS_membrane_lateral` awaits a lateral-branch port |

Output `semi-analytical-pdvd.json` (also installed to
`wire-cell-data/pdvd/photodet/`) is in the exact schema QLMatching loads;
per-channel ANN/pred medians (0.84–1.24) are recorded in the JSON for
efficiency shimming.  Fit plots: `pdvd/photlib/pics/fit_*.png`.

**Verdict**: a semi-analytical PDVD model is *possible* but second-best — it
needs two C++ ports (both-sides cathode gate, lateral cosine branch) to cover
all PD types and still carries 30–80% point-level spread for the PMTs, vs the
library's direct representation of the official model.  Tuning "the photon
library model's parameters" is best done as (a) regenerating the grid from a
retrained/newer ANN, and (b) PDHD-style data-driven efficiency/λ-type scale
calibration on top of either backend.

## 7. Byte-identity (default OFF)

The QLMatching changes are config-gated (`light_model` default `"semi"`).
A/B on the PDHD matching chain (run 29107 event 0, imaging inputs fixed,
`run_clus_evt.sh` with Q/L on): all `mabc-*.zip` inner members byte-identical
between the pre-change binary and the new binary with the knob at default.

## 8. What remains before PDVD Q/L matching runs

1. `offset_us` light↔charge time-base calibration (flash `t` + drift
   consistency on A–C crossers), as noted in `pdvd-light-chain.md`.
2. Absolute PE scale / QtoL (and per-PD gain spread) against data.
3. The Arapuca DAPHNE↔module mirror-pair confirmation from data (§2).
4. Graph wiring: opflash tensors → `FlashTensorToOpticalPCs{nchan:40}` →
   QLMatching with the §5 constants; PDVD-specific matching knobs (FV, ladder,
   rescues) to be tuned after that.

## Reproduce

```bash
cd pdvd/photlib
python3 extract_photlib.py                  # PVS libraries + mapping gates
python3 gdml_opdets.py                      # v4/v5 GDML opdet dumps (needs PyROOT)
/home/xqian/tmp/tfvenv/bin/python sample_ann.py       # ANN order/checkv4/sample
/home/xqian/tmp/tfvenv/bin/python fit_semi_analytical.py
python3 export_wct_photlib.py               # WCT library files + selfcheck
# C++ interpolation check against the real file:
PHOTLIB_META=$WCDATA/pdvd/photodet/pdvd-photlib-vis-v5-128nm.json \
    ./build/match/wcdoctest-match
```

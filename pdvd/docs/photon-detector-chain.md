# ProtoDUNE VD Photon Detector: Simulation, Geometry & Reconstruction

A reference on the ProtoDUNE Vertical-Drift (PDVD) **Photon Detector (PD / optical / PDS)** chain,
as implemented in the DUNE LArSoft code (`dunecore`, `duneprototypes`, `dunereco`) and as surfaced
in the `wire-cell-bee3` event display. Paths are relative to `toolkit-dev/` (the parent of this
`toolkit` checkout), so e.g. `dunecore/...` means `toolkit-dev/dunecore/...`.

This is the VD companion to `pdhd/docs/photon-detector-chain.md`; a PDVD-vs-PDHD comparison is in
§6. Like that doc, it surveys upstream DUNE code to orient WCP/toolkit work — it records what
exists and what is wired in, and changes no code.

---

## 0. One-paragraph summary

PDVD light is produced in simulation by the standard LArSoft **FastOptical** path
(`IonAndScint` → `PDFastSim`) using a **photon library** (`PhotonVisibilityService =
protodune_photonvisibilityservice`, inherited from ProtoDUNE-SP), with the back-tracker keyed on
**both `PDFastSimAr` and `PDFastSimXe`** (argon + xenon-doped scintillation). The detector is small
and architecturally different from HD: **40 OpDet channels = 8 cathode X-ARAPUCAs + 8 membrane
X-ARAPUCAs + 24 PMTs** (56 DAPHNE hardware channels), mapped via a JSON file
(`PDVD_PDS_Mapping_v09162025.json`) through the `PDVDPDMapAlg` tool. On the **data** side a
`DAPHNEReaderPDVD` module exists to decode the three PDS sub-streams into `raw::OpDetWaveform`, but
it is **not wired into any job fcl**, and there is **no PDVD OpHit/OpFlash reconstruction, no flash
matcher** at all — the PDVD optical reco is even less wired than PDHD's. In `wire-cell-bee3`, PDVD
(like PDHD) has **no optical-detector geometry**: TPC boxes only, no `updateOPLocation`/`opTPC`, and
`showPD()` has no `protodunevd` branch (and `op.js` would fall back to MicroBooNE-style PMT circles).

---

## 1. Photon Detector Simulation Chain (incl. photon library)

### 1.1 Service stack

PDVD simulation services are in `dunecore/dunecore/Utilities/services_protodunevd.fcl`:

```fcl
protodunevd_simulation_services:
{
  @table::protodunevd_minimal_simulation_services
  SignalShapingServiceDUNE:     @local::protodunesp_signalshapingservice
  PhotonVisibilityService:      @local::protodune_photonvisibilityservice   # photon library (PDSP-inherited)
  OpDetResponseInterface:       @local::standard_opdetresponse
}
protodunevd_simulation_services.OpDetResponseInterface.QuantumEfficiency: 1
...
protodunevd_services.LArPropertiesService.ScintPreScale: 0.2
```

- **`PhotonVisibilityService = protodune_photonvisibilityservice`** — PDVD reuses the **ProtoDUNE-SP
  photon library** (a precomputed per-voxel→per-OpDet visibility lookup), **not** a VD-specific
  service. (Contrast: PDHD selects `protodune_hd_photonvisibilityservice`.) The library `.root`
  table itself is defined in external `larsim`/`dunecore`.
- **`OpDetResponseInterface = standard_opdetresponse`**, `QuantumEfficiency = 1`.
- **`ScintPreScale = 0.2`** for PDVD (vs `1` for PDHD, `0.0287` for the PDSP base) — the photon
  down-sampling factor that stands in for QE during production.

The refactored-LArG4 stack (`dunecore/dunecore/Utilities/services_refactored_pdune.fcl`) adds the
back-tracker and notably keys it on **two** producers:

```fcl
protodunevd_refactored_simulation_services:
{
  @table::protodunevd_simulation_services
  ParticleInventoryService:  @local::standard_particleinventoryservice
  PhotonBackTrackerService:  @local::dunefd_photonbacktrackerservice
}
protodunevd_larg4_services.LArG4Detector: @local::protodunevd_larg4detector
protodunevd_refactored_simulation_services.PhotonBackTrackerService.PhotonBackTracker.G4ModuleLabels:
    [ "PDFastSimAr", "PDFastSimXe" ]
```

The **`PDFastSimAr` + `PDFastSimXe`** pair reflects VD's dual argon/xenon-doped scintillation
handling (the PD2HD doc keys only on `PDFastSim`).

### 1.2 Optical physics parameters

Inherited from `dunecore/dunecore/Utilities/services_protodune_singlephase.fcl`: `ScintYield=24000`
γ/MeV, Cerenkov off, and `FastOptical` in the `EnabledPhysics` list — i.e. photons are sampled from
the library rather than tracked step-by-step. PDVD overrides `ScintPreScale` to `0.2`.

### 1.3 Module order

```
Generator → LArG4 → IonAndScint(+External) → PDFastSim(Ar/Xe) → OpDetDigitizer(DAPHNE) → raw::OpDetWaveform
```

The `protodunevd_ionandscint` config drives IonAndScint; `PDFastSim` uses the photon library.
As with PDHD, charge-only workflows skip the optical branch — e.g. the local
`DNN_ROI_SP/simulation/stageA/protodunevd_refactored_g4_stage2.fcl` runs `IonAndScint` with
`PDFastSim` commented out.

---

## 2. Photon Detector Geometry (channel → location, shape)

### 2.1 GDML & detector shape

- Geometry: `dunecore/dunecore/Geometry/gdml/protodunevd_v5_ggd.gdml` (selected by
  `protodunevd_v5_geo` in `geometry_dune.fcl`; a `driftY` v3 variant also exists).
- Optics are **large square X-ARAPUCA modules**, not HD's long bars. From the GDML:
  - Module outer envelope **65.3 (x) × 2.5 (y) × 65.3 (z) cm**.
  - **Single-window** modules (`XARAPUCA_window_shape` 60×1×60 cm) — used on the **membrane** walls.
  - **Double-window** modules (`XARAPUCA_double_window_shape` 60×2.48×60 cm) — used on the
    **cathode** (double-sided collection).
- Sensitive volumes: `volOpDetSensitive_XARAPUCA*` (cathode/membrane) and PMT volumes
  (`volOpDetSensitive_pmt*`). Cathode modules sit on the central cathode plane; membrane modules on
  the lateral field-cage / membrane walls.

### 2.2 Channel count and grouping

Authoritative counts from the mapping JSON
(`dunecore/dunecore/ChannelMap/PDVD_PDS_Mapping_v09162025.json`, 40 entries):

| Type | OpDet channels | DAPHNE HW channels |
|---|---|---|
| Cathode X-ARAPUCA | 8 | 16 (2 per module) |
| Membrane X-ARAPUCA | 8 | 16 (2 per module) |
| PMT | 24 | 24 |
| **Total** | **40** | **56** |

Each map entry carries `channel`, `pd_type` (`Cathode`/`Membrane`/`PMT`), module `name`
(`C1…C8`, `M1…M8`), wavelength shifter (`wls: "PTP"`), efficiencies (`eff_Ar`, `eff_Xe` ≈ 0.03),
and a `HardwareChannel` list of `{Slot, Link, DaphneChannel, OfflineChannel}`. Example:

```json
{ "channel": 0, "pd_type": "Membrane", "name": "M1", "wls": "PTP",
  "eff_Ar": 0.03, "eff_Xe": 0.03,
  "HardwareChannel": [ {"Slot":7,"Link":0,"DaphneChannel":47,"OfflineChannel":2010},
                       {"Slot":7,"Link":0,"DaphneChannel":45,"OfflineChannel":2011} ] }
```

### 2.3 Channel mapping service

Wired in `dunecore/dunecore/Geometry/geometry_dune.fcl`:

```fcl
protodunevd_wire_readout: {
  ...
  ChannelsPerOpDet: 1
  PDMapTool: { tool_type: "PDVDPDMapAlg"
               MappingFile: "PDVD_PDS_Mapping_v09162025.json" }   # v04/v07 older
}
```

Implemented by `dunecore/dunecore/ChannelMap/PDVDPDMapAlg.hh` + `PDVDPDMapAlg_tool.cc` (loads the
JSON; provides `pdType`, `ArgonEfficiency`/`XenonEfficiency`, channel lookups). PDVD selects this
via `protodunevd_wire_readout` in `services_protodunevd.fcl`.

### 2.4 Runtime access

Standard `geo::Geometry::OpDetGeoFromOpChannel(ch).GetCenter()` for the OpDet position; PD-type and
efficiency come from the `PDVDPDMapAlg` PD-map tool rather than the geometry alone.

---

## 3. Photon Detector Reconstruction Chain

### 3.1 Raw decoding (data)

`duneprototypes/duneprototypes/Protodune/vd/RawDecoding/DAPHNEReaderPDVD_module.cc` decodes DAPHNE
into `std::vector<raw::OpDetWaveform>`:

- `produces<std::vector<raw::OpDetWaveform>>(fOutputLabel)`, `OutputLabel` default `"daq"`.
- Handles **three PDS sub-streams** via `DaphneInterface3`:
  `SubDetString = {"VD_Membrane_PDS", "VD_Cathode_PDS", "VD_PMT_PDS"}`.
- Supporting interface: `…/vd/RawDecoding/PDVDDataInterfaceWIBEth_tool.cc`.

**Wiring status:** the module compiles (referenced only in `…/vd/RawDecoding/CMakeLists.txt` and its
own source) but is **not invoked by any `.fcl` job** in the tree. The VD RawDecoding job fcls
(`run_pdvd_beamevent.fcl`, `run_pdvd_timing_decoder.fcl`, `run_pdvd_wibeth_tpc_decoder.fcl`, …)
cover TPC / timing / trigger / beam — **none decode optical**. So even raw PDS decoding is not part
of a standard PDVD job today (PDHD at least ships a standalone `pdhd_daphne_decoder_job.fcl`).

### 3.2 Higher-level reco (OpHit / OpFlash)

**None exists for PDVD.** No `OpHitFinder`/`Deconvolution`/`OpFlash`/`OpSlicer` configuration or
producer is defined for PDVD anywhere in `duneprototypes/.../Protodune/vd` or `dunereco`. (HD at
least *declares* `opdec`/`ophitspe`/`opflash`/`opslicer` in its master job, albeit disabled.)

For reference, the **VD Coldbox** decoder
`duneprototypes/duneprototypes/Coldbox/vd/VDColdboxPDSDecoder_module.cc` does
`produces<std::vector<recob::OpHit>>(...)` — but it emits **empty placeholder `recob::OpHit()`
objects**, i.e. the OpHit slot exists structurally without real hit-finding. That is the closest
VD-lineage precedent for an OpHit producer.

### 3.3 Flash matching / T0

**No PDVD flash matcher.** PDVD beam timing comes from the beam-line module
(`pdvd_beamevent`, from the SP `BeamReco/BeamEvent.fcl`), which is **not** optical flash–TPC
matching. CRT reconstruction/trigger exists (`…/Protodune/vd/CRT/`) but is independent of PDS.

### 3.4 Calibration

`services_protodunevd.fcl`:
`protodunevd_data_reco_services.IPhotonCalibrator: @local::protodunesp_photoncalibrator` — PDVD
reuses the ProtoDUNE-SP photon calibrator (defined but unused without OpHit reco).

---

## 4. Current Status & Planned Usage in PDVD

| Component | Status | Where |
|---|---|---|
| Simulation (FastOptical + photon library, Ar+Xe) | ✅ implemented | §1 |
| DAPHNE raw decode → `OpDetWaveform` | ⚠️ module exists, **not in any job fcl** | `DAPHNEReaderPDVD_module.cc` |
| OpHit finding | ❌ none for PDVD (Coldbox emits empty placeholders) | — |
| OpFlash | ❌ none | — |
| Flash ↔ TPC T0 matching | ❌ none (beam-line T0 only, not optical) | `pdvd_beamevent` |
| CRT ↔ PDS matching | ❌ not wired (CRT reco itself exists) | `…/vd/CRT/` |
| Optical calibration | ✅ defined, reused from SP, unused | `protodunesp_photoncalibrator` |

**Reading of the situation.** PDVD light *simulation* is mature; the *data/reco* side is the least
developed of the ProtoDUNEs surveyed — a single decoder module exists but isn't even wired into a
job, and there is no OpHit/OpFlash/flash-matching at all. The implied next steps mirror PDHD but
start one rung lower: (1) ship a PDVD DAPHNE decoder *job* fcl that runs `DAPHNEReaderPDVD`;
(2) add OpHit/OpFlash producers (the SP `protoDUNE_optical_reco.fcl` chain is the natural template,
adapted to VD's cathode+membrane+PMT mix); (3) build a flash matcher; (4) wire CRT↔PDS. For
WCP/toolkit purposes the only available product today is `raw::OpDetWaveform` (and only by running
the decoder module manually).

---

## 5. Photon Detector handling in wire-cell-bee3

PDVD's situation in bee3 is **identical to PDHD: no optical geometry**.

### 5.1 Generic infra (exists, detector-agnostic)

`wire-cell-bee3/events/static/js/bee/physics/op.js` loads per-event `op.json`
(`op_t`, `op_pes`, `op_peTotal`, `op_pes_pred`, `op_cluster_ids`, optional `apa`, optional
`op_flash_group`) and steps flashes with `<`/`>`; charge-light matching (reco vs detector frame) is
documented in `wire-cell-bee3/docs/charge-light-matching-true-frame.md`. This machinery would load
PDVD flash data if provided.

### 5.2 PDVD gap

In `experiment.js`, the `ProtoDUNEVD` class defines **8 TPC anode boxes** (central cathode at x=0,
two drift volumes, `driftDir(i) = i<4 ? 1 : -1`) and **nothing optical**:

- **no `updateOPLocation()`** → `op.location = {}`, `nDet = 0`;
- **no `opTPC()` override** → falls back to TPC 0;
- **no `showPD()` branch** for `protodunevd` in `helper.js` (it covers only `protodune`, `icarus`,
  `sbnd`);
- `op.js buildGroup()` has an explicit `sbnd` branch and otherwise renders **MicroBooNE-style PMT
  circles** — so even with data, PDVD's X-ARAPUCAs would be mis-rendered as generic PMTs.
- `wire-cell-bee3/docs/protodune_geometry.md` documents PDVD **TPC boxes only** — zero optical
  content.

### 5.3 What full PDVD support would need

1. `updateOPLocation(...)` in `ProtoDUNEVD` from the GDML X-ARAPUCA/PMT positions (§2), with a
   per-channel `detType` (X-ARAPUCA square vs PMT circle, plus cathode/membrane);
2. an `opTPC()` mapping (cathode modules are shared between the two drift volumes — needs a
   convention);
3. a `showPD()` / `op.js` branch for `protodunevd` (X-ARAPUCA rectangles + PMT circles);
4. doc the optics in `protodune_geometry.md`.

The charge-light matching machinery itself is detector-agnostic and would work once geometry is
supplied.

---

## 6. PDVD vs PDHD at a glance

| | **PDVD** | **PDHD** |
|---|---|---|
| OpDet channels | **40** (8 cathode + 8 membrane X-ARAPUCA + 24 PMT) | **160** (4 APA × 10 bars × 4 windows, X-ARAPUCA) |
| OpDet shape/placement | large 65.3 cm square modules on cathode + membrane walls; PMTs | long ~209.6 cm bars (4 windows each) on APA frames |
| PhotonVisibilityService | `protodune_photonvisibilityservice` (SP-inherited) | `protodune_hd_photonvisibilityservice` (HD-specific) |
| Backtracker labels | `PDFastSimAr`, `PDFastSimXe` | `PDFastSim` |
| ScintPreScale | 0.2 | 1 |
| Channel map | JSON `PDVD_PDS_Mapping_v09162025.json` via `PDVDPDMapAlg` | text `PD2HDChannelMap_v5.txt` via `PD2HDChannelMapService` |
| DAPHNE decode | `DAPHNEReaderPDVD` (3 substreams) — **not in any job fcl** | `DAPHNEReaderPDHD` — has standalone job fcl, ✅ usable |
| OpHit/OpFlash | none defined | declared in master job but **disabled** |
| Flash matcher | none | none |
| bee3 optical geometry | none (TPC only) | none (TPC only) |

---

## Appendix — key file index

| Topic | File |
|---|---|
| Sim/data service tables | `dunecore/dunecore/Utilities/services_protodunevd.fcl` |
| Refactored sim + backtracker (Ar/Xe) | `dunecore/dunecore/Utilities/services_refactored_pdune.fcl` |
| Optical physics base | `dunecore/dunecore/Utilities/services_protodune_singlephase.fcl` |
| Geometry GDML | `dunecore/dunecore/Geometry/gdml/protodunevd_v5_ggd.gdml` |
| Geometry / wire_readout / PD-map wiring | `dunecore/dunecore/Geometry/geometry_dune.fcl` |
| PD channel map (JSON) | `dunecore/dunecore/ChannelMap/PDVD_PDS_Mapping_v09162025.json` |
| PD-map algorithm tool | `dunecore/dunecore/ChannelMap/PDVDPDMapAlg.hh` + `PDVDPDMapAlg_tool.cc` |
| DAPHNE decode (VD) | `duneprototypes/.../Protodune/vd/RawDecoding/DAPHNEReaderPDVD_module.cc` |
| VD Coldbox PDS decoder (empty OpHit) | `duneprototypes/.../Coldbox/vd/VDColdboxPDSDecoder_module.cc` |
| bee3 optical loader/schema | `wire-cell-bee3/events/static/js/bee/physics/op.js`, `wire-cell-bee3/docs/overview.md` |
| bee3 charge-light matching | `wire-cell-bee3/docs/charge-light-matching-true-frame.md` |

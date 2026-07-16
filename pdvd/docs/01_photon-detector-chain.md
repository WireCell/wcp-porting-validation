# ProtoDUNE VD Photon Detector: Simulation, Geometry & Reconstruction

A reference on the ProtoDUNE Vertical-Drift (PDVD) **Photon Detector (PD / optical / PDS)** chain,
as implemented in the DUNE LArSoft code (`dunecore`, `duneprototypes`, `dunereco`) and as surfaced
in the `wire-cell-bee3` event display. Paths are relative to `toolkit-dev/` (the parent of this
`toolkit` checkout), so e.g. `dunecore/...` means `toolkit-dev/dunecore/...`.

This is the VD companion to `pdhd/docs/01_photon-detector-chain.md`; a PDVD-vs-PDHD comparison is in
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
(`C1…C8`, `M1…M8`), wavelength shifter (`wls`), efficiencies (`eff_Ar`, `eff_Xe`), and a
`HardwareChannel` list of `{Slot, Link, DaphneChannel, OfflineChannel}`. The wavelength shifter and
efficiency are **type-dependent**: X-ARAPUCAs are `wls: "PTP"` with `eff ≈ 0.03`; PMTs are
`wls: "TPB"` with `eff ≈ 0.12`. Examples:

```json
{ "channel": 0, "pd_type": "Membrane", "name": "M1", "wls": "PTP",
  "eff_Ar": 0.03, "eff_Xe": 0.03,
  "HardwareChannel": [ {"Slot":7,"Link":0,"DaphneChannel":47,"OfflineChannel":2010},
                       {"Slot":7,"Link":0,"DaphneChannel":45,"OfflineChannel":2011} ] }
{ "channel": 14, "pd_type": "PMT", "name": "", "wls": "TPB",
  "eff_Ar": 0.12, "eff_Xe": 0.12,
  "HardwareChannel": [ {"Slot":10,"Link":0,"DaphneChannel":4,"OfflineChannel":3010} ] }
```

### 2.3 Channel mapping convention

The PDVD mapping is **OpChannel-centric** (unlike PDHD's electronics-first text map): the JSON's
**`channel` (0–39) is the OpDet / OpChannel index** — the primary key consumed by the geometry and
the photon library — and each entry hangs the hardware list *off* that key. The convention chain is
the inverse direction of PDHD's:

```
JSON entry, keyed by  channel (0..39)  ==  OpDet / OpChannel index
   ├─ pd_type / name / wls / eff_Ar / eff_Xe        (physics attributes)
   └─ HardwareChannel[] : {Slot, Link, DaphneChannel, OfflineChannel}
                                 │  the DAPHNE electronics that read this OpDet
                                 ▼
        OfflineChannel is block-coded by type:  Membrane ≈ 20xx,  Cathode ≈ 10xx,  PMT ≈ 30xx
   channel (OpChannel)
      │  ChannelsPerOpDet = 1
      ▼
   geo::OpDetGeoFromOpChannel(channel).GetCenter()  →  (x,y,z)
      │  same index
      ▼
   photon-library column  (indexed by OpChannel — see §8)
```

Note an X-ARAPUCA OpDet maps to **two** DAPHNE channels (the `HardwareChannel` list has two
entries), a PMT to one — hence 40 OpDet channels but 56 DAPHNE hardware channels (§2.2). The
README also pins the **geometric** module→channel layout, e.g. membranes
`M1(0) M3(1) / M2(2) M4(3) / M5(12) M7(13) / M6(18) M8(19)` (NO-TCO | TCO halves) and the cathode
`C1(6)…C8(11)` layout. PMT `HardwareChannel`s are flagged "not fully defined yet" in the JSON.

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
JSON; provides `pdType`, `ArgonEfficiency`/`XenonEfficiency`, and the
`OfflineChannel ↔ OpDet` lookups). PDVD selects this via `protodunevd_wire_readout` in
`services_protodunevd.fcl`.

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
| Photon-library type (§8) | full voxel library `lib_Protodunev7_merged_avg.root` 140×120×140 — **SP geometry, not VD** | full voxel library `…protoDUNEhd_v2_refactored_nonActive.root` 122×67×93 (HD geometry) |
| Backtracker labels | `PDFastSimAr`, `PDFastSimXe` | `PDFastSim` |
| ScintPreScale | 0.2 | 1 |
| PDS channel map | JSON `PDVD_PDS_Mapping_v09162025.json` (OpChannel-keyed) via `PDVDPDMapAlg` | text `DAPHNE_test5_ChannelMap_v1.txt` (electronics-keyed) via `DAPHNEChannelMapService` |
| TPC/PDS time offset (§7) | −250 µs (copy of HD) | −250 µs |
| DAPHNE decode | `DAPHNEReaderPDVD` (3 substreams) — **not in any job fcl** | `DAPHNEReaderPDHD` — has standalone job fcl, ✅ usable |
| OpHit/OpFlash | none defined | declared in master job but **disabled** |
| Flash matcher | none | none |
| bee3 optical geometry | none (TPC only) | none (TPC only) |

---

## 7. Timing: PDS (light) vs TPC (charge) readout offset

> *Analogue of "SBND's 250 µs".* PDVD's detector-clocks config is a **verbatim copy of PDHD's**, so
> the light-vs-charge offset is **−250 µs**, same as HD.

From `dunecore/dunecore/Utilities/detectorclocks_dune.fcl`:

```fcl
# "Implement new 6000 tick readout window (500 before trigger, or 250 us)"
protodune_detectorclocks.G4RefTime:        -250.   # G4 time [us] where the electronics clock starts
protodune_detectorclocks.TriggerOffsetTPC: -250.   # TPC readout start w.r.t. trigger (= 500 ticks @ 2 MHz)
protodune_detectorclocks.DefaultTrigTime:   250.
protodune_detectorclocks.DefaultBeamTime:   250.

protodunehd_detectorclocks: @local::protodune_detectorclocks
protodunehd_detectorclocks.ClockSpeedTPC:     2.0
protodunehd_detectorclocks.ClockSpeedOptical: 62.5  # MHz → 16 ns optical tick

# "PD VD config - copy config from PD HD"
protodunevd_detectorclocks: @local::protodunehd_detectorclocks
```

`protodunevd_detectorclocks` is wired into every PDVD service table
(`services_protodunevd.fcl:14,21,27,35`).

**What the offset means.** The TPC readout window opens **250 µs (500 ticks @ 2 MHz) before** the
trigger; the PDS/optical readout shares the trigger reference (`DefaultBeamTime = 250 µs`). So
**light leads the start of the TPC window by 250 µs** — an interaction at trigger time is clocked
out starting at TPC tick **500**. A flash matcher converts flash time → drift-x with this 250 µs
anchor plus `drift_v · (t_flash − t_trigger)`.

- **Optical clock granularity:** `ClockSpeedOptical = 62.5 MHz` → **16 ns** per optical tick
  (inherited from HD; the base `protodune_detectorclocks` value of 150 MHz is overridden).
- **SBND for comparison** (`sbndcode/.../detectorclocks_sbnd.fcl`): `TriggerOffsetTPC = −205 µs`,
  `G4RefTime = −1700 µs`, optical 500 MHz (2 ns tick). SBND's "≈250 µs" is really −205 µs; PDVD's
  analogue is −250 µs.
- **Toolkit side:** PDVD's `cfg/pgrapher/experiment/protodunevd/clus.jsonnet` currently sets
  **`time_offset = 0`** — the readout tick0 is at −250 µs, but PDVD has **no per-event T0** (no flash
  matcher yet), so the imaging frame is left offset-free; restore −250 µs once a T0 measurement
  exists (comment block at `protodunevd/clus.jsonnet:9-16`). SBND, which *does* have a flash T0, uses
  `time_offset = −205 µs`.

| Quantity | PDVD | PDHD | SBND |
|---|---|---|---|
| `TriggerOffsetTPC` | −250 µs (copy of HD) | −250 µs (500 ticks) | −205 µs |
| `G4RefTime` | −250 µs | −250 µs | −1700 µs |
| Optical clock | 62.5 MHz (16 ns) | 62.5 MHz (16 ns) | 500 MHz (2 ns) |
| Toolkit `time_offset` | 0 (no T0; "true" = −250 µs) | 0 (no T0; "true" = −250 µs) | −205 µs |

---

## 8. Photon library: technology and wire-cell integration

### 8.1 What the PDVD library is

The named service `protodune_photonvisibilityservice` (selected in `services_protodunevd.fcl:45`,
"PDSP-inherited") is **defined in external `duneopdet`**, not in the checked-out repos — it lives in
`duneopdet/.../fcl/photpropservices_dune.fcl` (on cvmfs,
`/cvmfs/dune.opensciencegrid.org/products/dune/duneopdet/<ver>/fcl/photpropservices_dune.fcl:301,335`):

```fcl
# ProtoDUNE Single Phase with arapucas
protodunev7_photonvisibilityservice:
{
  NX: 140   NY: 120   NZ: 140
  UseCryoBoundary: true
  DoNotLoadLibrary: false
  LibraryFile: "PhotonPropagation/LibraryData/lib_Protodunev7_merged_avg.root"
  XMin: -120  XMax: 120  YMin: -120  YMax: 120  ZMin: 0  ZMax: 1200
}
# "Make the v7 visibility service the default"
protodune_photonvisibilityservice: @local::protodunev7_photonvisibilityservice
```

- **Technology: a full voxelized lookup library** (a `phot::PhotonLibrary` ROOT table), *not* a
  semi-analytical model — **140 × 120 × 140 ≈ 2.35 M voxels**, each storing per-OpChannel
  visibility, with `UseCryoBoundary: true` (grid spans the cryostat).
- **Important caveat:** this is the **ProtoDUNE *Single-Phase* "v7" library**
  (`lib_Protodunev7_merged_avg.root`), generated for the **SP geometry**, *not* the vertical-drift
  geometry. PDVD reuses it as an inherited placeholder. Its OpDet count/positions and detector
  bounds (a single-drift SP volume, `Z` 0–1200) **do not correspond** to PDVD's 40 cathode/membrane
  X-ARAPUCAs + PMTs — so light predictions from this library are not physically faithful to PDVD. A
  VD-specific library (or semi-analytical model) is the real prerequisite for PDVD optical reco.
- The `.root` file resolves at runtime via `FW_SEARCH_PATH` under `PhotonPropagation/LibraryData/`;
  it is **not** in the source repos.

### 8.2 How LArSoft uses it

`PhotonVisibilityService` is consumed by **`PDFastSimAr`** and **`PDFastSimXe`** (the dual
argon/xenon `FastOptical` path, §1.1): each scintillation deposit's photons are scaled by the
deposit-voxel's per-OpChannel visibility, Poisson-sampled into `sim::OpDetBacktrackerRecord`, and
digitized into `raw::OpDetWaveform`. Access is
`phot::PhotonVisibilityService::GetAllVisibilities(point)` / `GetCounts(...)`, indexed by the
**OpChannel** of §2.3.

### 8.3 Integrating PDVD light into wire-cell

As for PDHD (see that doc's §7.3), the toolkit's matcher (`match/`) does **not** read a ROOT
library — it predicts light with `WireCell::Match::SemiAnalyticalModel`
(`match/src/SemiAnalyticalModel.cxx`), configured by `VUVHits`/`VISHits` blocks + a `Geometry`
struct + a JSON list of `OpticalDetector{center, h, w, type, orientation}`. The two integration
routes are the same — but PDVD is **substantially harder** than HD:

1. **Semi-analytical route.** Would need a PDVD OpDet-geometry JSON (40 channels from §2.3) plus
   VD `VUVHits`/`VISHits` parameters. **But the current toolkit port supports only flat (X)Arapuca
   (`type 0`) and dome PMT (`type 1`) at anode/cathode orientation, and explicitly *omits*** lateral
   PD corrections, Xe absorption, and field-cage/vertical-border corrections
   (`SemiAnalyticalModel.h:14-20`). PDVD breaks all three: **membrane** X-ARAPUCAs sit on the lateral
   walls (not anode/cathode orientation), the **cathode** modules are double-sided, the **PMTs** are
   a third population, and the **Xe-doped** scintillation (the `PDFastSimAr`+`PDFastSimXe` split)
   wants the Xe-absorption branch. So this route requires porting those missing larsim branches
   first.
2. **Library route.** Add a `match/` reader for a VD photon library — but the inherited SP `v7`
   library (§8.1) is geometrically wrong, so a **VD-specific library must be generated first**
   (a `LibraryBuildJob` over `protodunevd_v5` geometry).

Either way the surrounding plumbing must also be built — and PDVD starts one rung lower than HD
(§4): there is **no OpHit/OpFlash producer wired at all**, so steps are:

3. **Flashes:** wire a PDVD DAPHNE decode → OpHit → OpFlash chain (none exists today, §3), then
   export per-event `op.json` (`op_t`, `op_pes` per OpChannel) in the bee3/toolkit schema (§5.1).
4. **Time anchor:** apply the §7 offset (−250 µs + `drift_v·(t_flash − t_trigger)`).
5. **Index alignment:** keep the `op_pes` vector, the light-model OpDet order, and the geometry
   positions all in **OpChannel order** (§2.3) — the single shared index.

In short: PDVD optical-to-wire-cell integration is gated on (a) a VD-faithful light model
(library *or* extended semi-analytical), (b) an OpFlash producer, neither of which exists yet.

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
| Detector clocks (TPC/PDS time offset, §7) | `dunecore/dunecore/Utilities/detectorclocks_dune.fcl` (`protodunevd_detectorclocks`) |
| Photon library definition (§8, external) | `duneopdet/.../fcl/photpropservices_dune.fcl` → `protodune_photonvisibilityservice` (= `protodunev7_…`) |
| Toolkit light model (semi-analytical) | `toolkit/match/{src,inc/WireCellMatch}/SemiAnalyticalModel.{cxx,h}` |
| DAPHNE decode (VD) | `duneprototypes/.../Protodune/vd/RawDecoding/DAPHNEReaderPDVD_module.cc` |
| VD Coldbox PDS decoder (empty OpHit) | `duneprototypes/.../Coldbox/vd/VDColdboxPDSDecoder_module.cc` |
| bee3 optical loader/schema | `wire-cell-bee3/events/static/js/bee/physics/op.js`, `wire-cell-bee3/docs/overview.md` |
| bee3 charge-light matching | `wire-cell-bee3/docs/charge-light-matching-true-frame.md` |

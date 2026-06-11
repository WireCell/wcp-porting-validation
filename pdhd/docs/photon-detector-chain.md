# ProtoDUNE HD Photon Detector: Simulation, Geometry & Reconstruction

A reference on the ProtoDUNE HD (PDHD) **Photon Detector (PD / optical / PDS)** chain, as
implemented in the DUNE LArSoft code (`dunecore`, `duneprototypes`, `dunereco`) and as surfaced in
the `wire-cell-bee3` event display. Paths are given relative to `toolkit-dev/` (i.e. the parent of
this `toolkit` checkout), so e.g. `dunecore/...` means `toolkit-dev/dunecore/...`.

> Scope note: this is a *survey of the upstream DUNE code*, written to orient WCP/toolkit work. It
> records what exists, what is wired into production, and what is not. It does not change any code.

---

## 0. One-paragraph summary

PDHD light is produced in simulation by the standard LArSoft **FastOptical** path
(`IonAndScint` → `PDFastSim`, using a **photon library** via `PhotonVisibilityService =
protodune_hd_photonvisibilityservice`), then digitized into `raw::OpDetWaveform`. The detector is
**160 X-ARAPUCA optical channels** (4 APAs × 10 bars × 4 acceptance windows). On the **data** side,
DAPHNE raw decoding into `raw::OpDetWaveform` is production-ready, but **OpHit/OpFlash
reconstruction and flash–TPC T0 matching are defined-but-disabled** in the master PDHD reco job —
the optical reco is essentially "decode only" today, with the higher-level reco existing as code
(largely inherited from ProtoDUNE-SP) but not yet enabled. In `wire-cell-bee3` there is generic
optical/flash display infrastructure, but **no PDHD optical-detector geometry is defined**, so PD
bars do not render for PDHD even though `op.json` flash data would load.

---

## 1. Photon Detector Simulation Chain (incl. photon library)

### 1.1 Service stack

The PDHD simulation service table is in
`dunecore/dunecore/Utilities/services_protodunehd.fcl`:

```fcl
protodunehd_simulation_services:
{
  @table::protodunehd_minimal_simulation_services
  SignalShapingServiceDUNE:     @local::protodunesp_signalshapingservice
  PhotonVisibilityService:      @local::protodune_hd_photonvisibilityservice   # the photon library
  OpDetResponseInterface:       @local::standard_opdetresponse
  DetectorClocksService:        @local::protodunehd_detectorclocks
}
protodunehd_simulation_services.OpDetResponseInterface.QuantumEfficiency: 1
```

- **`PhotonVisibilityService = protodune_hd_photonvisibilityservice`** — this is the *photon
  library*: a precomputed lookup of per-voxel → per-OpDet visibility. It is consumed by `PDFastSim`
  to convert each scintillation energy deposit into expected detected photons without full Geant4
  optical tracking. (The library table itself is defined in external `larsim`/`dunecore` photon-lib
  fcl; PDHD simply selects the HD variant here.)
- **`OpDetResponseInterface = standard_opdetresponse`**, with **`QuantumEfficiency = 1`** — QE is
  set to 1 in sim because the photon down-scaling is applied earlier (see `ScintPreScale` below) and
  at digitization, not here.

The table layers up as `protodunehd_simulation_services` → `protodunehd_minimal_simulation_services`
→ `protodune_minimal_simulation_services`. The refactored-LArG4 variant adds the back-tracker and is
in `dunecore/dunecore/Utilities/services_refactored_pdune.fcl`:

```fcl
protodunehd_refactored_simulation_services:
{
  @table::protodunehd_larg4_services
  @table::protodunehd_simulation_services
  ParticleInventoryService:     @local::standard_particleinventoryservice
  PhotonBackTrackerService:     @local::dunefd_photonbacktrackerservice
}
```

### 1.2 Optical physics parameters

From `dunecore/dunecore/Utilities/services_protodune_singlephase.fcl` (inherited base):

```fcl
...LArPropertiesService.ScintYield:        24000      # photons / MeV
...LArPropertiesService.ScintPreScale:     0.0287     # pre-scale to save memory (QE-like)
...LArPropertiesService.EnableCerenkovLight: false
...LArG4Parameters.UseCustomPhysics: true
...LArG4Parameters.EnabledPhysics: [ "Em", "FastOptical", "SynchrotronAndGN", "Ion",
                                     "Hadron", "Decay", "HadronElastic", "Stopping" ]
```

The key is **`FastOptical`** in the physics list: photons are not tracked step-by-step through
Geant4; instead `PDFastSim` samples the photon library. Cerenkov light is off; scintillation yield
is 24000 γ/MeV with a `ScintPreScale` down-sampling factor.

### 1.3 Module order (depo → detected light → waveform)

```
Generator (e.g. cosmics)
   │
   ▼
LArG4 (refactored)                      energy deposits in LAr
   │
   ▼
IonAndScint / IonAndScintExternal       ionization + scintillation photons per depo
   │
   ▼
PDFastSim / PDFastSimExternal           photon-library propagation → OpDetBacktrackerRecord
   │                                    (semi-analytical visibility; "active" + "external" volumes)
   ▼
OpDetDigitizer (DAPHNE response)        SiPM + electronics → raw::OpDetWaveform
```

- `IonAndScint`/`PDFastSim` come in **active** and **External** instances (in-TPC vs. surrounding
  volumes). The module configs (`IonAndScint_dune.fcl`, `PDFastSim_dune.fcl`) live in external
  `larsim`/`dunecore`.
- **PhotonBackTrackerService** (`dunecore/dunecore/Utilities/photonbacktrackerservice_dune.fcl`)
  ties detected light back to truth:

  ```fcl
  dunefd_photonbacktrackerservice.PhotonBackTracker:
  {
    G4ModuleLabels:           ["PDFastSim"]   # producer of the OpDetBacktrackerRecords
    MinimumHitEnergyFraction: 0.1
    Delay:                    260
  }
  ```

> Practical note: in charge-only workflows the optical branch is often skipped to save CPU — e.g.
> the local `DNN_ROI_SP/simulation/stageA_pdhd` depo-extraction logs show
> `IonAndScint (PDFastSim photon sim disabled)`. The photons are produced/measured only when the PDS
> branch is actually needed.

---

## 2. Photon Detector Geometry (channel → location, shape)

### 2.1 GDML & detector shape

- Geometry: `dunecore/dunecore/Geometry/gdml/protodunehd_v6_refactored.gdml` (with-wires) and
  `…_nowires.gdml`; generated by
  `dunecore/dunecore/Geometry/gdml/generate_protodunehd_v6_refactored.pl`.
- Detector type: **single-sided X-ARAPUCA bars** mounted on the APA frames.
  - Bar outer envelope ≈ **2.3 (x) × 12.0 (y) × 209.6 (z) cm**.
  - Each bar has **4 acceptance windows** ≈ **1.0 × 10.0 × 47.75 cm**, spaced along z.

### 2.2 Channel count and naming

| Quantity | Value |
|---|---|
| APAs | 4 (2×2 layout) |
| Bars per APA | 10 |
| Acceptance windows per bar | 4 |
| **Total OpDet channels** | **160** (= 4 × 10 × 4) |

Sensitive volumes are named `volOpDetSensitive_[APA]-[BAR]-[WINDOW]` (APA 0–3, bar 0–9, window 0–3),
with positions `posOpArapuca[APA]-[BAR]-[WINDOW]-…` in the GDML. Cathode-side vs. anode-side APAs sit
at the two extreme drift-x values; bars span the full APA height in y and the bar length in z.

### 2.3 Channel maps (offline ↔ hardware)

- TPC/PDS offline mapping service:
  `duneprototypes/duneprototypes/Protodune/hd/ChannelMap/PD2HDChannelMapService.h` with map files
  `…/ChannelMap/PD2HDChannelMap_v5.txt` (v5 = current physics; v0–v4 historical).
- DAPHNE (PDS electronics) mapping:
  `duneprototypes/duneprototypes/Protodune/hd/ChannelMap/DAPHNEChannelMapService.h`
  + `DAPHNE_test5_ChannelMap_v1.txt`.

### 2.4 Runtime access (OpDet position from a channel)

```cpp
auto const& geo = *lar::providerFrom<geo::Geometry>();
geo::OpDetGeo const& od = geo.OpDetGeoFromOpChannel(opChannel);
geo::Point_t center = od.GetCenter();   // (x,y,z) cm
```

The ProtoDUNE/DUNE APA op-channel routing lives in
`dunecore/dunecore/Geometry/DuneApaWireReadoutGeom.h` and
`dunecore/dunecore/Geometry/ProtoDUNEWireReadoutGeomv8.h`
(`OpDetFromOpChannel`, `NOpChannels`, `SSPfromOpDet`, …).

---

## 3. Photon Detector Reconstruction Chain

### 3.1 Raw decoding (data) — production-ready

`duneprototypes/duneprototypes/Protodune/hd/RawDecoding/DAPHNEReaderPDHD_module.cc` converts DAPHNE
raw frames into `std::vector<raw::OpDetWaveform>`:

- `produces<std::vector<raw::OpDetWaveform>>(fOutputLabel)` where `OutputLabel` defaults to `"daq"`;
  the sub-detector tag is `SubDetString = "HD_PDS"` (this is a tag, *not* the product instance name).
- Frame-format variants handled by tools `DAPHNEInterface1/2/3_tool.cc`
  (`DAPHNEInterfaceBase.h`), with helpers in `DAPHNEUtils.{h,cxx}`.
- Standalone job: `…/hd/RawDecoding/pdhd_daphne_decoder_job.fcl`; module config
  `DAPHNEReaderPDHD.fcl`.
- **Feb-2026 fix** in `DAPHNEInterface3_tool.cc`: timestamps converted from DAPHNE ticks → µs via
  `OpticalClock().TickPeriod()` so downstream `OpHitFinder` reads correct times; an open caveat is
  noted that a 64-bit→double conversion does not preserve the full 16 ns PDS precision.

### 3.2 Higher-level reco (OpHit → OpFlash) — defined but **not** enabled

In the master PDHD data job
`duneprototypes/duneprototypes/Protodune/spsbsm/fcl/run_master_np04data_processor.fcl`, optical
producers are *declared*:

```fcl
pdhddaphne: @local::DAPHNEReaderPDHD
opdec:      @local::dune_deconvolution        # waveform deconvolution
ophitspe:   @local::dune_ophit_finder_deco    # OpDetWaveform → recob::OpHit
opflash:    @local::protodune_opflash         # OpHit → recob::OpFlash
opslicer:   @local::protodune_opslicer        # flash slicing
```

…but the executed `produce:` path (lines ~145–169) contains **none of them** — not even
`pdhddaphne`. It runs only TPC decoding, signal processing (`wclsdatahd`), hit finding, and Pandora.
So in the current master configuration, **no optical reconstruction runs at all** for PDHD data; the
modules are staged for future enabling.

### 3.3 Reference chain (ProtoDUNE-SP) — the template PDHD inherits from

`duneprototypes/duneprototypes/Protodune/singlephase/PhotonDetectors/protoDUNE_optical_reco.fcl`
shows the intended shape of a full optical reco path:

```fcl
produceIt: [ ssprawdecoder,
             ophitInternal, ophitExternal,
             opflashInternal, opflashExternal ]
```

i.e. raw-decode → OpHit (internal/external) → OpFlash (internal/external), with `opflashana`
analyzers. Flash–CRT T0 matching for SP exists in
`…/singlephase/PhotonDetectors/RunPDSPMatch.fcl` (module `PDSPmatch`, with `CRTCTBOffset`,
`CRTWindow`). **There is no PDHD-specific equivalent yet** — these are the templates a PDHD
flash-matcher would be ported from.

### 3.4 Calibration

`dunecore/dunecore/Utilities/services_protodunehd.fcl`:

```fcl
protodunehd_data_reco_services.IPhotonCalibrator: @local::protodunesp_photoncalibrator
```

PDHD currently **reuses the ProtoDUNE-SP photon calibrator** (SPE/gain); no PDHD-specific optical
calibration service was found.

---

## 4. Current Status & Planned Usage in PDHD

| Component | Status | Where |
|---|---|---|
| DAPHNE raw decode → `OpDetWaveform` | ✅ implemented, production-ready | `DAPHNEReaderPDHD_module.cc` |
| OpHit finding (`OpDetWaveform`→`OpHit`) | ⚠️ code present, **disabled** | declared in `run_master_np04data_processor.fcl`, absent from `produce:` |
| OpFlash (`OpHit`→`OpFlash`) | ⚠️ code present, **disabled** | same |
| Flash ↔ TPC T0 matching | ❌ no PDHD module (SP `PDSPmatch` exists as template) | `RunPDSPMatch.fcl` (SP) |
| CRT ↔ PDS matching | ❌ not wired for PDS (CRT reco itself exists) | — |
| Optical calibration | ✅ basic, **reused from SP** | `protodunesp_photoncalibrator` |
| Simulation (FastOptical + photon library) | ✅ implemented | §1 |

**Reading of the situation.** The PDHD light *simulation* is mature and standard. On the *data /
reconstruction* side the system is deliberately staged at "decode only": waveforms are produced, but
OpHit/OpFlash and any flash-based T0 are not yet run in the master job. The Feb-2026 DAPHNE
timestamp-precision fix is the kind of groundwork that precedes enabling OpHit finding, which
suggests renewed/active effort. The natural next steps implied by the code are:

1. add `opdec`/`ophitspe`/`opflash`/`opslicer` to the `produce:` path once timing is trusted;
2. port a PDHD flash-matcher from the SP `PDSPmatch` template (flash → TPC T0 / cluster matching);
3. wire CRT ↔ PDS coincidence for absolute timing.

For WCP/toolkit purposes the immediately useful product is `raw::OpDetWaveform` (and, once enabled,
`recob::OpHit`/`recob::OpFlash`) — i.e. the same flash objects the SBND toolkit flash-matching
consumes, but not yet produced by the standard PDHD chain.

---

## 5. Photon Detector handling in wire-cell-bee3

`wire-cell-bee3` is the 3-D Bee event display. It has **generic** optical/flash infrastructure but,
crucially, **no PDHD optical geometry**.

### 5.1 Generic optical/flash infrastructure (exists)

- Loader/renderer: `wire-cell-bee3/events/static/js/bee/physics/op.js`; helper objects in
  `…/js/bee/helper.js`; detector classes in `…/js/bee/physics/experiment.js`.
- Per-event `op.json` schema (documented in `wire-cell-bee3/docs/overview.md`): `op_t` (flash times,
  µs, ascending), `op_pes` (per-channel measured PE), `op_peTotal`, optional `op_pes_pred`
  (predicted PE), `op_cluster_ids` (matched clusters per flash), optional `apa` (per-flash TPC), and
  optional `op_flash_group` (ties TPC0/TPC1 flashes within ±80 ns for grouped display).
- Display: step flashes with `<`/`>`; measured PE = red circles (radius ∝ √PE), predicted PE = green
  circles; TPC boxes shift along drift by `driftV·t·driftDir`. Charge-light matching has a "reco
  frame" vs "detector frame" side-panel mode, documented in
  `wire-cell-bee3/docs/charge-light-matching-true-frame.md`.

### 5.2 PDHD gap (what is missing)

The `ProtoDUNEHD` class in `experiment.js` defines **TPC boxes only**. Specifically it has:

- **no `updateOPLocation()`** call → no optical-detector positions registered;
- **no `opTPC()` override** → cannot map an OpDet/flash to its APA (falls back to TPC 0);
- **no `showPD()` branch** for `exp.name == 'protodunehd'` in `helper.js`.

Consequence: `op.json` *loads* and flashes *step*, but PDHD's 160 X-ARAPUCA bars are **not drawn**,
and per-detector PE has no place to render. Contrast with the full implementations: **SBND** = 312
optical detectors (120 PMT + 192 X-ARAPUCA), original single-drift **ProtoDUNE** = ~90 bars.

### 5.3 What full PDHD support would need

1. add `updateOPLocation({id:[x,y,z]}, 160)` in `ProtoDUNEHD`, derived from the GDML
   `volOpDetSensitive_[APA]-[BAR]-[WINDOW]` positions (§2);
2. add an `opTPC()` override mapping OpDet → APA (0–3);
3. add a `showPD()` branch for `protodunehd` to draw the X-ARAPUCA bars;
4. document the optics in `wire-cell-bee3/docs/protodune_geometry.md` (currently TPC-only).

The charge-light matching machinery itself is detector-agnostic and would work for PDHD once the
geometry above is supplied.

---

## Appendix — key file index

| Topic | File |
|---|---|
| Sim service table | `dunecore/dunecore/Utilities/services_protodunehd.fcl` |
| Optical physics / FastOptical | `dunecore/dunecore/Utilities/services_protodune_singlephase.fcl` |
| Refactored sim + backtracker | `dunecore/dunecore/Utilities/services_refactored_pdune.fcl` |
| Photon backtracker | `dunecore/dunecore/Utilities/photonbacktrackerservice_dune.fcl` |
| Geometry GDML (+gen) | `dunecore/dunecore/Geometry/gdml/protodunehd_v6_refactored.gdml` / `generate_protodunehd_v6_refactored.pl` |
| OpDet/op-channel routing | `dunecore/dunecore/Geometry/{DuneApaWireReadoutGeom,ProtoDUNEWireReadoutGeomv8}.h` |
| Offline channel map (PDS) | `duneprototypes/.../Protodune/hd/ChannelMap/PD2HDChannelMapService.h` + `PD2HDChannelMap_v5.txt` |
| DAPHNE map | `duneprototypes/.../Protodune/hd/ChannelMap/DAPHNEChannelMapService.h` + `DAPHNE_test5_ChannelMap_v1.txt` |
| DAPHNE decode | `duneprototypes/.../Protodune/hd/RawDecoding/DAPHNEReaderPDHD_module.cc` + `DAPHNEInterface{1,2,3}_tool.cc` |
| Master data job (produce list) | `duneprototypes/.../Protodune/spsbsm/fcl/run_master_np04data_processor.fcl` |
| SP optical reco template | `duneprototypes/.../Protodune/singlephase/PhotonDetectors/protoDUNE_optical_reco.fcl` |
| SP flash–CRT match template | `duneprototypes/.../Protodune/singlephase/PhotonDetectors/RunPDSPMatch.fcl` |
| bee3 optical loader/schema | `wire-cell-bee3/events/static/js/bee/physics/op.js`, `wire-cell-bee3/docs/overview.md` |
| bee3 charge-light matching | `wire-cell-bee3/docs/charge-light-matching-true-frame.md` |

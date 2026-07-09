# PDVD TPC geometry & QLMatching fiducial volume

Cross-source comparison of the ProtoDUNE-VD (PDVD) TPC geometry, and the
definition of the fiducial volume (FV) that charge–light matching (QLMatching)
uses **per drift-TPC region**. Written to prepare PDVD for QLMatching, which is
**not yet wired into the PDVD chain**
(`cfg/pgrapher/experiment/protodunevd/clus.jsonnet:486` "PDVD runs no Q/L
matching"; `flash.jsonnet:49-50` names `QLMatching{nchan:40}` only as a future
consumer). The three sources compared are:

1. **Toolkit WCT config** — `cfg/pgrapher/experiment/protodunevd/params.jsonnet`
   (+ `simparams.jsonnet`, `clus.jsonnet`).
2. **wire-cell-bee3** display — `events/static/js/bee/physics/experiment.js`
   (class `ProtoDUNEVD`), derivation in `wire-cell-bee3/docs/protodune_geometry.md`.
3. **LArSoft simulation** — GDML `protodunevd_v4_refactored.gdml`
   (`dunecore/dunecore/Geometry/gdml/`), used by the sim chain in
   `DNN_ROI_SP/simulation/stageA/` (`depo_extract.fcl:407,488` GDML =
   `protodunevd_v4_refactored.gdml`, Name `protodunevd_v4`).

There is **no PDVD GDML inside the toolkit repo itself**; the WCT geometry is
derived from the LArSoft wire dump `protodunevd-wires-larsoft-v5.json.bz2`
(`params.jsonnet:189`). The GDML values below were read directly from
`dunecore`'s v4 GDML (both `dunecore` and `DNN_ROI_SP` `git pull`ed, already
up to date at time of writing).

---

## 1. Coordinate convention

PDVD is a **vertical-drift** detector: charge drifts along **x** toward two
Charge-Readout Planes (CRPs), one above (top) and one below (bottom) a central
cathode. In the **WCT / larsoft-v5 wire frame** used everywhere downstream:

- Cathode membrane centred at **x = 0**.
- W (collection) plane at **|x| = 341.55 cm**.
- 8 anodes / CRP quadrants, idents **0–7**, 2 faces each (the two y-halves of a
  CRP) → 16 faces, 48 planes, 13 840 wires.
- **Anodes 0–3 = bottom CRP**, centerline x = **−341.55 cm**, drift **+x**.
- **Anodes 4–7 = top CRP**, centerline x = **+341.55 cm**, drift **−x**.
- The two drift volumes are mirror-symmetric about the cathode at x = 0.

> **GDML frame offset.** The raw `protodunevd_v4` GDML cryostat frame is shifted:
> `posCathode-0/1` sit at **GDML x = −20 cm** (not 0). Consequently the GDML
> top-CRP collection plane is at GDML x ≈ **+321.55** and the bottom at ≈ **−361.55**
> (top CRPs placed at box-center x=+152.28 identity, bottom at −192.28 with
> `rPlus180AboutY`; the collection Z-plane sits at local x=+169.27 inside `volTPC`).
> Both are **341.55 cm from the cathode** — i.e. once the −20 cm offset is removed
> the GDML agrees with the WCT frame. The larsoft-v5 wire dump already applies
> this shift, so all WCT/bee3 numbers are cathode-at-0.

---

## 2. TPC geometry — three-source comparison

All distances are |x| from the cathode centre (x = 0 in the WCT frame) unless noted.

| Quantity | Toolkit WCT | wire-cell-bee3 | LArSoft v4 GDML |
|---|---|---|---|
| W collection plane \|x\| | **341.55 cm** (`apa_cpa`, `params.jsonnet:29`) | **341.55 cm** (box X edge, `experiment.js:795-803`) | **341.55 cm** from cathode (collection Z-plane local +169.27, CRP center ±341.55−cathode; verified via −20 cm offset) |
| Anode grid / field plane \|x\| | **335.835 cm** (`apa_plane`=0.5·`apa_g2g`=57.15 mm inside W, `params.jsonnet:33,35,66`) | 335.835 cm (FV anode-band edge) | `AnodePlate` 0.01 cm thin (GDML L12772), y=337, z=299.3 cm |
| Response plane \|x\| | **319.164 cm** (`res_plane`=0.5·`apa_w2w`+18.1 cm, `params.jsonnet:41,43`) | — (n/a) | — (n/a) |
| Cathode surface \|x\| (drift stop) | **2.54 cm** (`cpa_plane`=`apa_cpa`−0.5·`cpa_thick`⇒ surface at 2.54, `params.jsonnet:47`) | 2.54 cm (FV gap edge) | **~2.94–3.0 cm** (mesh at ±2.937/±2.975 about −20; `CathodeBlock` = 6 cm thick, GDML L12778) |
| Cathode thickness | **5.08 cm** (`cpa_thick`=50.8 mm, `params.jsonnet:30`) | 5.08 cm | **6.0 cm** (`CathodeBlock` x, GDML L12778-79) |
| Effective drift distance (W → cathode surface) | **339.01 cm** (341.55−2.54) | 339.01 cm | **338.5 cm** (`CRMActive` box x, GDML L1201; ≈341.55−3.0) |
| Drift speed | reco **1.57 mm/µs** (`params.jsonnet:112`, calibrated from A–C crossers) | 0.16 cm/µs base default | sim **1.473 mm/µs** (`simparams.jsonnet:12`); Efield **500 V/cm** (GDML `volTPCActive` aux) |
| U / V / W plane x-stacking | W at 341.55, V 341.53, U 341.51 (0.2 mm steps; PCB strips) | — | `CRMUPlane/VPlane/ZPlane` 0.02 cm thick (GDML L1206-1224) |
| Y extent | y ∈ [−342, 342] cm rough box (`bounds`, `params.jsonnet:100-103`); wire y ±336.4 | y ±336.4 cm, per-CRP split at \|y\|=0.6 cm | CRM module y = 168.5 cm; 4 rows at y = ±252.75, ±84.25 (GDML posTPC) → full ±337 |
| Z extent | z ∈ [0, 304] cm rough box | bottom [0.855, 298.445], top [−0.36, 300.0] cm | CRM module z = 149.65 cm; 2 columns at z = ±74.825 |
| nticks (readout) | **6000** (data, `params.jsonnet:117`) | — | **6400** (sim, `simparams.jsonnet:22`) |
| Wire file | `protodunevd-wires-larsoft-v5.json.bz2` (`params.jsonnet:189`) | v5 (`experiment.js:782`) | GDML `protodunevd_v4_refactored.gdml` (its wire dump → v5) |

### Per-drift-volume face planes (WCT, `params.jsonnet:49-92`)

| Drift volume | Anodes | Centerline (W) x | Anode/grid face x | Response plane x | Cathode surface x | Drift dir |
|---|---|---|---|---|---|---|
| Bottom CRP | 0–3 | −341.55 | −335.835 | −319.164 | −2.54 | +x (→ cathode) |
| Top CRP | 4–7 | +341.55 | +335.835 | +319.164 | +2.54 | −x (→ cathode) |

---

## 3. Reconciliation & discrepancies

The three sources **agree on the load-bearing number**: the W collection plane
sits **341.55 cm from the cathode centre** in every source (once the GDML −20 cm
frame offset is removed). The remaining differences all trace to the **cathode
thickness** and to sim-vs-reco parameter choices:

- **Cathode thickness / surface:** WCT `cpa_thick` = 5.08 cm (surface at |x|=2.54)
  vs GDML `CathodeBlock` = 6.0 cm (surface at |x|≈3.0). The ~0.46 cm/side
  difference in where the drift stops is the *entire* cause of the drift-distance
  gap below.
- **Effective drift distance:** WCT 339.01 cm (341.55−2.54) vs GDML `CRMActive`
  338.5 cm (≈341.55−3.0). Same 341.55 anchor; difference = the cathode
  half-thickness discrepancy. **This ~0.5 cm directly moves the cathode-end FV
  edge** and should be kept in mind when setting QLMatching's `cathode_ext1/2`.
- **Drift speed:** sim uses **1.473 mm/µs** (`simparams.jsonnet`), whereas data
  reconstruction uses the **1.57 mm/µs** value calibrated from anode→cathode
  crossers (`params.jsonnet:112`; see `pdvd/docs/clus-workflow.md` drift-velocity
  calibration). A cluster's apparent drift-x therefore differs between MC-truth
  and reco frames — relevant to any MC-based FV/velocity residual study.
- **Readout window:** 6000 ticks (data) vs 6400 ticks (sim).

> **Provenance note.** Unlike PDHD — whose op-detector geometry we verified
> end-to-end against the official GDML on dunegpvm (see the bee3 geometry memory)
> — the PDVD physical constants (`cpa_thick`, `apa_cpa`) are hand-set in
> `params.jsonnet` and derived from the v5 wire dump; the GDML cross-check here is
> the first direct comparison and it exposes the 5.08 vs 6.0 cm cathode difference.

---

## 4. Fiducial volume for QLMatching

### Two distinct FV consumers — do not conflate

1. **`clus.jsonnet` `dvm`** (per drift region) — bottom-drift `a0f0pA` x ∈
   [−3358.35, −25.4] mm, top-drift `a4f0pA` x ∈ [25.4, 3358.35] mm, y/z with
   15 cm insets (`clus.jsonnet:56-104`). This feeds **`clustering_separate` /
   `clustering_neutrino`** via `select_scope_fv`. It is **not** the QLMatching FV.

2. **QLMatching FV** (`match/src/QLMatching.cxx::compute_geometry`, ~L890-989) —
   the box comes from **`IDetectorVolumes::inner_bounds(wpid)` =
   `iface->sensitive()`** (the wire-geometry sensitive bounding box,
   `aux/src/DetectorVolumes.cxx:187-193`), **unioned over all anodes on the drift
   side** (`grouping_anodes`), then expressed in a per-side drift coordinate:

   ```
   u = s * (x − anode_x),   u = 0 at anode,   u = u_cathode at cathode
   ```

   with signed cushions applied in `u`: `anode_ext1` (−2 cm, PE-inclusion lower
   edge), `anode_ext2` (+4 cm, close-to-PMT flag), `cathode_ext1` (+ past cathode,
   containment/PE edge), `cathode_ext2` (− short of cathode, flag-only), plus
   `y_cushion`/`z_cushion` (default 0). Two consumers: the PE-inclusion gate
   (`build_bundles`, ~L1216) and the containment/boundary flags
   (`compute_endpoint_flags`, ~L2733). `require_containment` (default OFF) drops
   uncontained bundles. Optional per-TPC 3-D CPA structure-exclusion `IFiducial`
   (`cathode_fiducial`) exists for **SBND only**
   (`cfg/.../sbnd/cathode_fiducial.jsonnet`); PDHD/PDVD use the flat-cathode test.

**Both are per drift-TPC region** (each side of the cathode): QLMatching runs one
`ApaRun` per drift side, with the FV box unioned over that side's anodes and the
OpDet set masked to that side by comparing each OpDet's x to `m_cathode_x`
(`compute_geometry` ~L967-973). Cross-cathode crossers are handled by a separate
**joint** node (`matching_joint`, `xtpc_flag`, `cull_cross_tpc`) that pairs a
crosser's two halves across the two sides. Confirms the intended definition:
**the fiducial volume is defined for each drift TPC region separately.**

### PDHD vs PDVD

| Aspect | PDHD (`cfg/.../pdhd/qlmatching.jsonnet`) | PDVD (target) |
|---|---|---|
| QLMatching wired? | Yes | **No** (future; `flash.jsonnet` names `QLMatching{nchan:40}`) |
| Cathode | Central, **opaque**, x = 0 | Central, x = 0; 5.08 cm (WCT) / 6.0 cm (GDML) |
| Photon detectors | Anode-mounted X-ARAPUCAs (160 ch, `active_opdet_types:[0]`) | **Cathode-mounted** X-ARAPUCAs, `nchan:40`, double-sided |
| FV per drift side | Yes (`inner_bounds` union over the side's 2 APAs → z ≈ 0–4.6 m) | Yes (union over the side's 4 CRP quadrants) |
| `cathode_ext1 / ext2` | **1.5 / −3.0 cm** (`qlmatching.jsonnet:276-277`; tuned to the ±1.75 cm crosser residual at reco v=1.576) | provisional (see §5) |
| `require_containment` | true | provisional true |
| `cross_side_filter` | **true** (opaque cathode: a flash lights only its own side) | **does NOT transfer** — see caveat |

> **Cathode-mounted-PD caveat (central to QLMatching prep).** PDHD's
> `cross_side_filter` / opaque-cathode side-assignment assumes a flash illuminates
> only clusters on its own drift side. **PDVD's X-ARAPUCAs are mounted on the
> cathode and face both drift volumes** (`nchan:40`, single all-PD flash spanning
> both sides), so a flash sees **both** sides. The PDHD side-assignment /
> `cross_side_filter` logic therefore **cannot be carried over unchanged** to
> PDVD. The *charge-containment* FV stays per-drift-region in both; it is the
> *light*-side assignment that differs.

---

## 5. Translating PDVD geometry into QLMatching parameters

For when QLMatching is wired into PDVD, the geometry inputs are:

- `anode_x = ±341.55 cm` (W collection plane; blob `xorig`), sign per drift side.
- `cathode_x = 0` (central cathode).
- `u_cathode ≈ 339.01 cm` (WCT drift distance; **338.5 cm** if anchored to the
  GDML `CRMActive` active volume — the ~0.5 cm gap is the cathode-thickness
  difference, §3).
- Per-side drift box from `inner_bounds` unioned over the 4 CRP quadrants on that
  side; y ≈ ±336.4 cm, z ≈ [0, 300] cm.
- Drift speed: **1.57 mm/µs** for data reco (sim MC is 1.473).

Suggested starting `cathode_ext1 / cathode_ext2` **by analogy to PDHD (1.5 / −3.0
cm)** — but flagged **provisional**: PDHD's values were tuned to its ±1.75 cm
cathode-crossing residual at its calibrated velocity, and PDVD's velocity /
t0 / SCE residual spread is not yet characterised. Do a PDVD-specific
crosser-residual study before freezing these. The ~0.5 cm cathode-surface
discrepancy (§3) is comparable to these cushions, so pin down whether PDVD reco
anchors the cathode at 2.54 or ~3.0 cm before setting the containment edge.

---

## 6. Open items / limitations

- GDML cross-check done against **v4** (`protodunevd_v4_refactored.gdml`, the
  version the DNN_ROI_SP sim uses). Newer `v5_ggd` GDMLs exist in `dunecore` and
  were not compared; confirm the production reco geometry version before freezing.
- The 5.08 vs 6.0 cm cathode-thickness difference is unresolved here — it is a
  real geometry-definition mismatch between WCT config and the GDML, worth
  reconciling with the PDVD geometry owners (does reco expect the drift to stop at
  2.54 or ~3.0 cm?).
- PDVD light-side (`cross_side_filter` replacement for cathode-mounted PDs) is a
  design task, not just a parameter — the single double-sided flash means a PDVD
  bundle's light model must sum contributions from both drift volumes.
- No PDVD-specific `cathode_fiducial` (CPA structure-exclusion) exists; SBND-only
  today.

## References (read-only sources)

- `cfg/pgrapher/experiment/protodunevd/{params,simparams,clus,flash}.jsonnet`
- `wire-cell-bee3/events/static/js/bee/physics/experiment.js`,
  `wire-cell-bee3/docs/protodune_geometry.md`
- `dunecore/dunecore/Geometry/gdml/protodunevd_v4_refactored.gdml`;
  `DNN_ROI_SP/simulation/stageA/{depo_extract,gen_protodunevd_cosmics}.fcl`
- `match/src/QLMatching.cxx`, `match/inc/WireCellMatch/QLMatching.h`,
  `cfg/pgrapher/experiment/{pdhd,sbnd}/qlmatching.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/cathode_fiducial.jsonnet`,
  `aux/src/DetectorVolumes.cxx`
- Related PDVD docs: `pdvd-wire-geometry-v3-v4.md`, `single-face-anode-feasibility.md`,
  `clus-workflow.md` (drift-velocity calibration), `photon-detector-chain.md`.

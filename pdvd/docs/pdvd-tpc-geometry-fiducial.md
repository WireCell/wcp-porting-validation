# PDVD TPC geometry & QLMatching fiducial volume

> **Update 2026-07 — cathode correction applied.** The toolkit's legacy cathode
> thickness (`cpa_thick = 50.8 mm`, a DocDB 203 / ProtoDUNE-SP copy — see §3) has
> been **corrected to the GDML value `60.0 mm`**. Drift-facing cathode surface
> 2.54 → **3.0 cm**; drift distance `cpa_plane` 339.01 → **338.55 cm** (≈ GDML
> CRMActive 338.5); `dvm` FV cathode edges ±25.4 → **±30.0 mm**; reco drift speed
> rescaled `v ∝ D`: 1.57 → **1.568 mm/µs**. `apa_cpa = 341.55 cm` (position) was
> already correct and is unchanged. Files: `params.jsonnet`, `clus.jsonnet`,
> `drift_calib/calib_drift_velocity.py`. **Data not yet reprocessed** — the new
> velocity applies on the next reprocessing pass; the ~0.15 % / ≤0.5 cm change is
> within the calibration's pile-up bin resolution. The comparison below is written
> as of the pre-correction state to show the reconciliation; corrected toolkit
> values are flagged inline.

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
| Cathode surface \|x\| (drift stop) | 2.54 cm (legacy) → **3.0 cm** (corrected, `cpa_plane`=`apa_cpa`−0.5·`cpa_thick`) | 2.54 cm (FV gap edge) | **~2.94–3.0 cm** (mesh at ±2.937/±2.975 about −20; `CathodeBlock` = 6 cm thick, GDML L12778) |
| Cathode thickness | 5.08 cm (legacy) → **6.0 cm** (corrected, `cpa_thick`=60 mm) | 5.08 cm | **6.0 cm** (`CathodeBlock` x, GDML L12778-79) |
| Effective drift distance (W → cathode surface) | 339.01 cm (legacy) → **338.55 cm** (corrected) | 339.01 cm | **338.5 cm** (`CRMActive` box x, GDML L1201; ≈341.55−3.0) |
| Drift speed | reco 1.57 → **1.568 mm/µs** (corrected, `v ∝ D`; A–C crossers) | 0.16 cm/µs base default | sim **1.473 mm/µs** (`simparams.jsonnet:12`); Efield **500 V/cm** (GDML `volTPCActive` aux) |
| U / V / W plane x-stacking | W at 341.55, V 341.53, U 341.51 (0.2 mm steps; PCB strips) | — | `CRMUPlane/VPlane/ZPlane` 0.02 cm thick (GDML L1206-1224) |
| Y extent | y ∈ [−342, 342] cm rough box (`bounds`, `params.jsonnet:100-103`); wire y ±336.4 | y ±336.4 cm, per-CRP split at \|y\|=0.6 cm | CRM module y = 168.5 cm; 4 rows at y = ±252.75, ±84.25 (GDML posTPC) → full ±337 |
| Z extent | z ∈ [0, 304] cm rough box | bottom [0.855, 298.445], top [−0.36, 300.0] cm | CRM module z = 149.65 cm; 2 columns at z = ±74.825 |
| nticks (readout) | **6000** (data, `params.jsonnet:117`) | — | **6400** (sim, `simparams.jsonnet:22`) |
| Wire file | `protodunevd-wires-larsoft-v5.json.bz2` (`params.jsonnet:189`) | v5 (`experiment.js:782`) | GDML `protodunevd_v4_refactored.gdml` (its wire dump → v5) |

### Per-drift-volume face planes (WCT, `params.jsonnet:49-92`)

| Drift volume | Anodes | Centerline (W) x | Anode/grid face x | Response plane x | Cathode surface x | Drift dir |
|---|---|---|---|---|---|---|
| Bottom CRP | 0–3 | −341.55 | −335.835 | −319.164 | −3.0 (was −2.54) | +x (→ cathode) |
| Top CRP | 4–7 | +341.55 | +335.835 | +319.164 | +3.0 (was +2.54) | −x (→ cathode) |

---

## 3. Reconciliation & discrepancies

The three sources **agree on the load-bearing number**: the W collection plane
sits **341.55 cm from the cathode centre** in every source (once the GDML −20 cm
frame offset is removed). The one real geometry discrepancy was the **cathode
thickness**, now corrected (see banner + provenance below); the rest are sim-vs-reco
parameter choices:

- **Cathode thickness / surface — RESOLVED.** The toolkit carried `cpa_thick` =
  5.08 cm (surface |x|=2.54) vs the GDML `CathodeBlock` = 6.0 cm (surface |x|≈3.0).
  **Corrected 2026-07 to 6.0 cm** so the surface is at |x|=3.0 cm and the drift
  distance `cpa_plane` = 338.55 cm (≈ GDML `CRMActive` 338.5). The ~0.5 cm shift
  directly moves the cathode-end FV edge — hence it feeds into QLMatching's
  `cathode_ext1/2`.
- **Effective drift distance:** now WCT 338.55 cm (341.55−3.0) vs GDML `CRMActive`
  338.5 cm — agree to 0.05 cm (the collection-plane vs active-volume-edge offset).
- **Drift speed:** sim uses **1.473 mm/µs** (`simparams.jsonnet`, the true MC
  velocity — left unchanged), whereas data reconstruction uses the calibrated value
  **1.568 mm/µs** (rescaled from 1.57 by `v ∝ D`; `params.jsonnet`; see
  `pdvd/docs/clus-workflow.md`). A cluster's apparent drift-x therefore differs
  between MC-truth and reco frames — relevant to any MC-based FV/velocity residual
  study.
- **Readout window:** 6000 ticks (data) vs 6400 ticks (sim).

### Where do the toolkit cathode numbers come from?

The two cathode constants had **different provenance** — this is *why* one was
correct and the other needed the 2026-07 fix:

- **`apa_cpa = 341.55 cm` (cathode *position* / W-plane→cathode-centre) — from
  LArSoft.** `params.jsonnet:25` derives it from `protodune-wires-larsoft-v3.json.bz2`,
  and it matches the v4 GDML directly (W plane 341.55 cm from cathode, verified
  §1). Trustworthy.

- **`cpa_thick` was `50.8 mm` (cathode *thickness*) — NOT from any PDVD GDML;
  corrected to 60 mm.** Every PDVD GDML version (v1–v4) models the cathode as a
  **6 cm `CathodeBlock`**; none is 5.08 cm. 50.8 mm = exactly 2.00 inches, a
  **pre-GDML nominal from DocDB 203 / the ProtoDUNE-SP template**. The whole PDVD
  `det` block (`apa_w2w=85.725`, `apa_g2g=114.3`, `cpa_thick=50.8`) was copied from
  `pdsp/params.jsonnet`, whose comment reads
  `apa_w2w = 85.725*wc.mm, // DocDB 203 calls "W" as "X"`. The proof it was stale:
  `dune10kt-1x2x6/params.jsonnet:16-17` keeps `// cpa_thick = 50.8*wc.mm, // DocDB 203`
  commented out and replaces it with `3.175*wc.mm, // 1/8", from Bo Yu (BNL) and
  confirmed with LArSoft`. **PDHD (`pdhd/params.jsonnet:31`) and iceberg had already
  moved to the corrected value; PDVD had never been updated** and still carried the
  2-inch legacy value until this fix.

**So the 5.08 vs 6.0 cm cathode difference was not a version mismatch** — no GDML
version has 5.08 cm; the toolkit value was an inherited ProtoDUNE-SP/DocDB constant
that sidestepped the GDML entirely. The fix adopts the GDML `CathodeBlock` = 60 mm.
Note that other sibling constants copied in the same block (`apa_w2w`, `apa_g2g` —
the ProtoDUNE-SP *wire-plane-stack* spacings) still do not physically apply to a
PDVD PCB-strip CRP (U/V/W are 0.2 mm apart, not tens of mm), so the
`apa_plane`/`res_plane`-derived faces remain template constructs, not measured PDVD
planes — a separate, unaddressed issue.

> **Impact & why re-calibration is a rescale.** `cpa_thick` sets
> `cpa_plane = apa_cpa − 0.5·cpa_thick`, which is both the FV cathode edge and the
> collection→cathode-surface distance `D` the drift-velocity calibration uses. The
> calibration is `v_true = v_reco · D / S` (S = measured crosser reco x-span, from
> data, *independent of D*), so **v is exactly proportional to D** — the data does
> not independently measure v. Moving the cathode surface 2.54 → 3.0 cm changes
> D 339.01 → 338.55 cm, so v rescales 1.57 → **1.568 mm/µs** (−0.15 %). The
> re-run closure (`calib_drift_velocity.py`, 142 evts, 51 A–C crossers) gives a
> point estimate of 1.561 at the new D — consistent with 1.568 within the pile-up
> bin resolution (~0.9 %/bin). Net reco effect: absolute x ≤0.5 cm at full drift;
> clustering/matching behaviour is essentially unchanged — the value is corrected
> for geometric honesty and QLMatching-edge accuracy, not to fix a reco defect.

---

## 4. Fiducial volume for QLMatching

### Two distinct FV consumers — do not conflate

1. **`clus.jsonnet` `dvm`** (per drift region) — bottom-drift `a0f0pA` x ∈
   [−3358.35, −30.0] mm, top-drift `a4f0pA` x ∈ [30.0, 3358.35] mm (cathode edge
   corrected from ±25.4), y/z with 15 cm insets (`clus.jsonnet:56-104`). This feeds
   **`clustering_separate` / `clustering_neutrino`** via `select_scope_fv`. It is
   **not** the QLMatching FV.

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
- `u_cathode ≈ 338.55 cm` (corrected WCT drift distance; = GDML `CRMActive` 338.5
  to 0.05 cm — the cathode surface now sits at |x|=3.0 cm, §3).
- Per-side drift box from `inner_bounds` unioned over the 4 CRP quadrants on that
  side; y ≈ ±336.4 cm, z ≈ [0, 300] cm.
- Drift speed: **1.568 mm/µs** for data reco (sim MC is 1.473).

Suggested starting `cathode_ext1 / cathode_ext2` **by analogy to PDHD (1.5 / −3.0
cm)** — but flagged **provisional**: PDHD's values were tuned to its ±1.75 cm
cathode-crossing residual at its calibrated velocity, and PDVD's velocity /
t0 / SCE residual spread is not yet characterised. Do a PDVD-specific
crosser-residual study before freezing these. The cathode surface is now anchored
at 3.0 cm (§3), so tune the cushions against that edge.

---

## 6. Open items / limitations

- **Cathode thickness — DONE (2026-07).** Corrected from the legacy 5.08 cm to the
  GDML 6.0 cm (surface 3.0 cm, D=338.55, v 1.57→1.568). Config + calib updated;
  **data not yet reprocessed** — v=1.568 applies on the next reprocessing pass.
- GDML cross-check done against **v4** (`protodunevd_v4_refactored.gdml`, the
  version the DNN_ROI_SP sim uses). Newer `v5_ggd` GDMLs exist in `dunecore` and
  were not compared; confirm the production reco geometry version before freezing.
- The sibling `apa_w2w`/`apa_g2g` ProtoDUNE-SP wire-stack spacings copied into the
  PDVD `det` block are still non-physical for a PCB-strip CRP (§3) — the
  `apa_plane`/`res_plane` faces remain template constructs; not addressed here.
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

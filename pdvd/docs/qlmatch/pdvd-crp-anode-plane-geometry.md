# PDVD CRP anode plane geometry (W / V / U / Shield) + fiducial volume at the shield plane

> **Status: IN PROGRESS (2026-07-13).** This document is being filled in
> step-by-step as the geometry correction lands. Each section is marked with its
> current state and byte-identical status.

## Repro / where the numbers come from

- Confirmed hardware stacking: user (CRP owner), 2026-07-13.
- GDML source: `dunecore/dunecore/Geometry/gdml/protodunevd_v4_refactored.gdml`
  (official `git@github.com:DUNE/dunecore.git`), planes `posPlaneU*/Y*/Z*`,
  solids `CRMUPlane/VPlane/ZPlane`, active box `CRMActive`/`posActive*`.
- Wire file: `wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2`
  (referenced by `cfg/pgrapher/experiment/protodunevd/params.jsonnet:207`).
- Toolkit FV: `params.jsonnet` (`apa_plane`), `clus.jsonnet` (`dvm` `FV_x*`),
  consumed by `aux/src/DetectorVolumes.cxx::inner_bounds` and the `dvm`
  metadata; QLMatching FV in `match/src/QLMatching.cxx::compute_geometry`.
- Inspect the regenerated wire file:
  `wirecell-util wires-info protodunevd-wires-larsoft-v6.json.bz2`.

## 1. The confirmed physical stack

The PDVD Charge Readout Plane (CRP) is a PCB-strip anode. Describing the
**bottom** CRP (the top CRP is mirror-symmetric about the central cathode),
from the collection plane going **up toward the drift volume**:

```
W (collection)                          <- KEEP FIXED
   | 3.2 mm  (PCB thickness)
V (induction)
   | 10 mm   (gap)
U (induction)
   | 3.2 mm  (PCB thickness)
Shield plane                            <- drift-volume boundary (no wires)
```

- Shield-to-W total = 3.2 + 10 + 3.2 = **16.4 mm**.
- The shield plane is the drift-facing boundary of the **active LAr**. The
  16.4 mm between the shield and W is PCB stack, **not** active drift volume.

### Why this is a correction

The existing toolkit/LArSoft geometry modelled U/V/W as **0.2 mm** apart (a
simplified LArSoft convention) with **no shield plane** — off by ~16× for the
real CRP. This document records moving U/V to their physical positions, adding
the shield plane, and anchoring the fiducial volume at the shield.

## 2. Derived positions (W kept at |x| = 341.55 cm)

| Plane  | WCT \|x\| (cm) | Wire-file signed x (mm) bottom / top | GDML local x (cm) |
|--------|----------------|---------------------------------------|-------------------|
| W (Z)  | 341.55 (fixed) | −3415.5 / +3415.5 (unchanged)         | 169.27 (fixed)    |
| V      | 341.23         | −3412.3 / +3412.3  (Δ 3.0 mm inward)   | 168.95            |
| U      | 340.23         | −3402.3 / +3402.3  (Δ 12.8 mm inward)  | 167.95            |
| Shield | 339.91         | (no wires — FV boundary only)         | 167.63 (new)      |

"Inward" = toward the cathode (smaller |x|). Sign per CRP side: anodes 0–3
(bottom) at −x, anodes 4–7 (top) at +x.

## 3. Change log (filled per step)

| Step | Repo | File(s) | State |
|---|---|---|---|
| 0 | wcp-porting-img | this doc | scaffolded |
| 1 | dunecore (local only) | `protodunevd_v*.gdml` | pending |
| 2 | wire-cell-data | `protodunevd-wires-larsoft-v6.json.bz2`, `params.jsonnet:207` | pending |
| 3 | toolkit | `params.jsonnet` (`apa_plane`), `clus.jsonnet` (`dvm`) | pending |
| 4 | — | few-event reprocess sanity check | pending |
| 5 | wcp-porting-img | existing geometry docs + memory | pending |

## 4. Byte-identical / reproducibility status

- **NOT byte-identical.** Moving U/V physically (Step 2) changes sigproc/imaging;
  moving the FV anode boundary to the shield (Step 3) changes QLMatching FV and
  clustering containment. Applied as an **unconditional geometry correction**
  (per the prior U-plane FV move `b8f7f3d6`), not a default-OFF knob — user
  decision 2026-07-13.
- **Field-response caveat:** `protodunevd_FR_*.json.bz2` were computed for the
  0.2 mm geometry. A faithful U/V move should regenerate them; that is handled
  separately by the user ("other things"). The few-event reprocess here is a
  **sanity check, not production validation**.
- **Anode-cushion caveat:** the FV move interacts with QLMatching
  `anode_ext1/anode_ext2` and the ctoffset u=0 pinning; re-examined post-reprocess.

## 5. Open items

- GDML `volTPCActive` overlap when U/V move inside the active box (Step 1).
- Field-response regeneration for the new U/V geometry (user, separate).
- `anode_ext1/2` retune after the FV move (data-driven, post-reprocess).
- bee3 (`wire-cell-bee3`) also encodes the 341.55 anode edge — coordinated
  update noted, not done here.

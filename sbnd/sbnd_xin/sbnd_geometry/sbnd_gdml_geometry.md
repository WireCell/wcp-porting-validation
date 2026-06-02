# SBND Geometry Summary — v02_06

Extracted from `sbnd_v02_06_base.gdml`, `sbnd_v02_06.gdml`, `sbnd_v02_06_nowires.gdml`, and `sbnd-wires-geometry-v0206.json.bz2` (tag `v10_20_05` of `SBNSoftware/sbndcode`, `main` of `SBNSoftware/sbnd_data`).

All lengths in **mm** unless noted. Cross-check between GDML and wire JSON: **no discrepancies** (see Section 7).

---

## 1. Coordinate Systems

Two frames appear in these files:

| Frame | Label | Origin | Used in |
|-------|-------|--------|---------|
| Cryostat frame | (C) | Center of LAr membrane | GDML physvol placements |
| Wire-Cell / TPC frame | (J) | Upstream (−z) face of TPC | `sbnd-wires-geometry-v0206.json` |

Conversion: **z_J = z_C + 2917.5 mm**; x and y are identical in both frames.

The offset 2917.5 = 5010/2 + 412.5 moves the origin from the cryostat center to the upstream TPC face. In the J frame the TPC spans z_J = 0 → 5010 mm with its center at z_J = 2505 mm.

---

## 2. Two TPCs

Solid `boxTPC`; drift direction is ±x from cathode (x = 0) toward each anode wall.

| TPC | GDML volume | x_C center (mm) | y_C center (mm) | z_C center (mm) | z_J center (mm) | dx (mm) | dy (mm) | dz (mm) |
|-----|-------------|----------------:|----------------:|----------------:|----------------:|--------:|--------:|--------:|
| East (TPC 0) | `volTPC_East` | −1011.0 | 0.0 | −412.5 | 2505.0 | 2022.0 | 4074.645 | 5010.0 |
| West (TPC 1) | `volTPC_West` | +1011.0 | 0.0 | −412.5 | 2505.0 | 2022.0 | 4074.645 | 5010.0 |

**Active LAr volume** (`boxTPCActive`): dx = 2013.0 mm, dy = 4074.645 mm, dz = 5010.0 mm.  
Active volume center is shifted 4.5 mm toward the cathode relative to the TPC box center (the three 3 mm wire planes occupy the anode-facing edge):  
East active x_C = −1006.5 mm, West active x_C = +1006.5 mm.

Drift distance (cathode → outermost collection plane) = **2020.5 mm ≈ 202 cm**.

---

## 3. Anode Planes (Wire Planes)

Each TPC has three planes. Solid `boxTPCPlane`: thickness 3.0 mm, y-span 4000.15 mm, z-span 5013.15 mm.  
Wire pitch = 3.0 mm for all planes. Angles measured from the +Y axis.

> **Note on U/V naming in East TPC:** The GDML explicitly swaps `volTPCPlane_U` and `volTPCPlane_V` in the East TPC (the East U plane is rotated w.r.t. West). The JSON `ident` field labels planes by drift order (0 = closest to cathode, 2 = collection), which is the convention used here.

| TPC | Plane ident | x_C (mm) | Angle from +Y | Role | Channels | Distance from cathode (mm) |
|-----|-------------|----------:|--------------:|------|:--------:|---------------------------:|
| East | 0 | −2014.5 | +60° | First induction | 1984 | 2014.5 |
| East | 1 | −2017.5 | −60° | Second induction | 1984 | 2017.5 |
| East | 2 | −2020.5 | 0° (vertical) | Collection | 1670 | 2020.5 |
| West | 0 | +2014.5 | −60° | First induction | 1984 | 2014.5 |
| West | 1 | +2017.5 | +60° | Second induction | 1984 | 2017.5 |
| West | 2 | +2020.5 | 0° (vertical) | Collection | 1670 | 2020.5 |

All planes: y_C = 0, z_C = −412.5 mm (z_J = 2505.0 mm), same center as TPC.

---

## 4. Cathode Plane (CPA)

The CPA is a steel tube-frame structure, not a plain box. Its electrical surface is a mesh + TPB-coated foil panel assembly.

| Property | Value |
|----------|-------|
| x position (C frame) | **0.0 mm** (midway between anodes at ±2020.5 mm) |
| y position | 0.0 mm |
| z position (C) | −412.5 mm |
| z position (J) | 2505.0 mm |
| Individual foil panel size | 0.0508 mm (thick) × 932 mm (y) × 1172 mm (z) |
| Panels per TPC half-wall | 16 (arranged ~4 rows × 4 columns) |
| Overall frame y half-span | ~2070 mm (±2 × 1035 mm) |
| Overall frame z half-span | ~1310 mm from center |

---

## 5. Photon Detection System (PDS) Modules

There are **24 PDS modules** mounted on the APA frames: 12 on the East wall (x < 0), 12 on the West wall (x > 0). Each module holds **5 PMTs** and **8 X-Arapucas** (192 X-Arapucas total, 120 PMTs total).

Module case (`PDS_moduleCase`): 133.5 mm (x) × 1040 mm (y) × 904.8 mm (z).

### 5.1 PDS Module Layout

There are 3 y-groups and 4 z-positions, giving 12 modules per wall.

**y-group A (top):** PMTs at y = +1750, +1350, +950 mm  
**y-group B (middle):** PMTs at y = +400, 0, −400 mm  
**y-group C (bottom):** PMTs at y = −950, −1350, −1750 mm

**Module z-center positions (z_J):** 536.74, 1869.58, 3140.42, 4473.26 mm

| East module | West module | y-group | z_J center (mm) | z_C center (mm) |
|:-----------:|:-----------:|:-------:|----------------:|----------------:|
| 1 | 13 | A (top) | 536.74 | −2380.76 |
| 2 | 14 | A (top) | 1869.58 | −1047.92 |
| 3 | 15 | B (mid) | 536.74 | −2380.76 |
| 4 | 16 | B (mid) | 1869.58 | −1047.92 |
| 5 | 17 | C (bot) | 536.74 | −2380.76 |
| 6 | 18 | C (bot) | 1869.58 | −1047.92 |
| 7 | 19 | A (top) | 3140.42 | −77.08 |
| 8 | 20 | A (top) | 4473.26 | +1555.76 |
| 9 | 21 | B (mid) | 3140.42 | −77.08 |
| 10 | 22 | B (mid) | 4473.26 | +1555.76 |
| 11 | 23 | C (bot) | 3140.42 | −77.08 |
| 12 | 24 | C (bot) | 4473.26 | +1555.76 |

---

## 6. PMT Positions

**PMT geometry:** hemispherical R = 102 mm. Photocathode center x = ±2085.5 mm (East −, West +).

Within each module, relative to the module z-center, the 5 PMT z-offsets are ±300 mm (for the outer pair at the same y) and 0 mm (for the center PMT). The East wall modules face +x; West wall modules are rotated 180° around Y, so z-offsets within a module are negated for West.

### East Wall PMTs (x = −2085.5 mm)

| Module | PMT | y (mm) | z_J (mm) | z_C (mm) |
|:------:|:---:|-------:|---------:|---------:|
| 1 | 1 | +1750 | 236.74 | −2680.76 |
| 1 | 2 | +1750 | 836.74 | −2080.76 |
| 1 | 3 | +1350 | 536.74 | −2380.76 |
| 1 | 4 | +950 | 236.74 | −2680.76 |
| 1 | 5 | +950 | 836.74 | −2080.76 |
| 2 | 1 | +1750 | 1569.58 | −1347.92 |
| 2 | 2 | +1750 | 2169.58 | −747.92 |
| 2 | 3 | +1350 | 1869.58 | −1047.92 |
| 2 | 4 | +950 | 1569.58 | −1347.92 |
| 2 | 5 | +950 | 2169.58 | −747.92 |
| 3 | 1 | +400 | 236.74 | −2680.76 |
| 3 | 2 | +400 | 836.74 | −2080.76 |
| 3 | 3 | 0 | 536.74 | −2380.76 |
| 3 | 4 | −400 | 236.74 | −2680.76 |
| 3 | 5 | −400 | 836.74 | −2080.76 |
| 4 | 1 | +400 | 1569.58 | −1347.92 |
| 4 | 2 | +400 | 2169.58 | −747.92 |
| 4 | 3 | 0 | 1869.58 | −1047.92 |
| 4 | 4 | −400 | 1569.58 | −1347.92 |
| 4 | 5 | −400 | 2169.58 | −747.92 |
| 5 | 1 | −950 | 236.74 | −2680.76 |
| 5 | 2 | −950 | 836.74 | −2080.76 |
| 5 | 3 | −1350 | 536.74 | −2380.76 |
| 5 | 4 | −1750 | 236.74 | −2680.76 |
| 5 | 5 | −1750 | 836.74 | −2080.76 |
| 6 | 1 | −950 | 1569.58 | −1347.92 |
| 6 | 2 | −950 | 2169.58 | −747.92 |
| 6 | 3 | −1350 | 1869.58 | −1047.92 |
| 6 | 4 | −1750 | 1569.58 | −1347.92 |
| 6 | 5 | −1750 | 2169.58 | −747.92 |
| 7 | 1 | +1750 | 2840.42 | −77.08 |
| 7 | 2 | +1750 | 3440.42 | +522.92 |
| 7 | 3 | +1350 | 3140.42 | +222.92 |
| 7 | 4 | +950 | 2840.42 | −77.08 |
| 7 | 5 | +950 | 3440.42 | +522.92 |
| 8 | 1 | +1750 | 4173.26 | +1255.76 |
| 8 | 2 | +1750 | 4773.26 | +1855.76 |
| 8 | 3 | +1350 | 4473.26 | +1555.76 |
| 8 | 4 | +950 | 4173.26 | +1255.76 |
| 8 | 5 | +950 | 4773.26 | +1855.76 |
| 9 | 1 | +400 | 2840.42 | −77.08 |
| 9 | 2 | +400 | 3440.42 | +522.92 |
| 9 | 3 | 0 | 3140.42 | +222.92 |
| 9 | 4 | −400 | 2840.42 | −77.08 |
| 9 | 5 | −400 | 3440.42 | +522.92 |
| 10 | 1 | +400 | 4173.26 | +1255.76 |
| 10 | 2 | +400 | 4773.26 | +1855.76 |
| 10 | 3 | 0 | 4473.26 | +1555.76 |
| 10 | 4 | −400 | 4173.26 | +1255.76 |
| 10 | 5 | −400 | 4773.26 | +1855.76 |
| 11 | 1 | −950 | 2840.42 | −77.08 |
| 11 | 2 | −950 | 3440.42 | +522.92 |
| 11 | 3 | −1350 | 3140.42 | +222.92 |
| 11 | 4 | −1750 | 2840.42 | −77.08 |
| 11 | 5 | −1750 | 3440.42 | +522.92 |
| 12 | 1 | −950 | 4173.26 | +1255.76 |
| 12 | 2 | −950 | 4773.26 | +1855.76 |
| 12 | 3 | −1350 | 4473.26 | +1555.76 |
| 12 | 4 | −1750 | 4173.26 | +1255.76 |
| 12 | 5 | −1750 | 4773.26 | +1855.76 |

### West Wall PMTs (x = +2085.5 mm)

West modules 13–24 correspond to East modules 1–12 with x → +2085.5 mm. Due to the 180° rotation around Y, z-offsets within each module are negated (±300 mm offsets swap). The y coordinates and module z-centers are identical to the corresponding East module.

| Module | PMT | y (mm) | z_J (mm) | z_C (mm) |
|:------:|:---:|-------:|---------:|---------:|
| 13 | 1 | +1750 | 836.74 | −2080.76 |
| 13 | 2 | +1750 | 236.74 | −2680.76 |
| 13 | 3 | +1350 | 536.74 | −2380.76 |
| 13 | 4 | +950 | 836.74 | −2080.76 |
| 13 | 5 | +950 | 236.74 | −2680.76 |
| 14 | 1 | +1750 | 2169.58 | −747.92 |
| 14 | 2 | +1750 | 1569.58 | −1347.92 |
| 14 | 3 | +1350 | 1869.58 | −1047.92 |
| 14 | 4 | +950 | 2169.58 | −747.92 |
| 14 | 5 | +950 | 1569.58 | −1347.92 |
| 15 | 1 | +400 | 836.74 | −2080.76 |
| 15 | 2 | +400 | 236.74 | −2680.76 |
| 15 | 3 | 0 | 536.74 | −2380.76 |
| 15 | 4 | −400 | 836.74 | −2080.76 |
| 15 | 5 | −400 | 236.74 | −2680.76 |
| 16 | 1 | +400 | 2169.58 | −747.92 |
| 16 | 2 | +400 | 1569.58 | −1347.92 |
| 16 | 3 | 0 | 1869.58 | −1047.92 |
| 16 | 4 | −400 | 2169.58 | −747.92 |
| 16 | 5 | −400 | 1569.58 | −1347.92 |
| 17 | 1 | −950 | 836.74 | −2080.76 |
| 17 | 2 | −950 | 236.74 | −2680.76 |
| 17 | 3 | −1350 | 536.74 | −2380.76 |
| 17 | 4 | −1750 | 836.74 | −2080.76 |
| 17 | 5 | −1750 | 236.74 | −2680.76 |
| 18 | 1 | −950 | 2169.58 | −747.92 |
| 18 | 2 | −950 | 1569.58 | −1347.92 |
| 18 | 3 | −1350 | 1869.58 | −1047.92 |
| 18 | 4 | −1750 | 2169.58 | −747.92 |
| 18 | 5 | −1750 | 1569.58 | −1347.92 |
| 19 | 1 | +1750 | 3440.42 | +522.92 |
| 19 | 2 | +1750 | 2840.42 | −77.08 |
| 19 | 3 | +1350 | 3140.42 | +222.92 |
| 19 | 4 | +950 | 3440.42 | +522.92 |
| 19 | 5 | +950 | 2840.42 | −77.08 |
| 20 | 1 | +1750 | 4773.26 | +1855.76 |
| 20 | 2 | +1750 | 4173.26 | +1255.76 |
| 20 | 3 | +1350 | 4473.26 | +1555.76 |
| 20 | 4 | +950 | 4773.26 | +1855.76 |
| 20 | 5 | +950 | 4173.26 | +1255.76 |
| 21 | 1 | +400 | 3440.42 | +522.92 |
| 21 | 2 | +400 | 2840.42 | −77.08 |
| 21 | 3 | 0 | 3140.42 | +222.92 |
| 21 | 4 | −400 | 3440.42 | +522.92 |
| 21 | 5 | −400 | 2840.42 | −77.08 |
| 22 | 1 | +400 | 4773.26 | +1855.76 |
| 22 | 2 | +400 | 4173.26 | +1255.76 |
| 22 | 3 | 0 | 4473.26 | +1555.76 |
| 22 | 4 | −400 | 4773.26 | +1855.76 |
| 22 | 5 | −400 | 4173.26 | +1255.76 |
| 23 | 1 | −950 | 3440.42 | +522.92 |
| 23 | 2 | −950 | 2840.42 | −77.08 |
| 23 | 3 | −1350 | 3140.42 | +222.92 |
| 23 | 4 | −1750 | 3440.42 | +522.92 |
| 23 | 5 | −1750 | 2840.42 | −77.08 |
| 24 | 1 | −950 | 4773.26 | +1855.76 |
| 24 | 2 | −950 | 4173.26 | +1255.76 |
| 24 | 3 | −1350 | 4473.26 | +1555.76 |
| 24 | 4 | −1750 | 4773.26 | +1855.76 |
| 24 | 5 | −1750 | 4173.26 | +1255.76 |

---

## 7. X-Arapuca Positions

**X-Arapuca geometry:** 20 mm (x) × 240 mm (y) × 96 mm (z) box. Center x = ±2145.5 mm (further from TPC than PMTs). Each module has 8 X-Arapucas.

Within a module there are two z-clusters per row. For the center PMT row (y = ±1350 or 0 mm) there are 4 X-Arapucas; for the two outer rows there are 2 X-Arapucas each. Local z offsets relative to module z-center: −376.2, −223.8, +223.8, +376.2 mm (approximate; the exact computed values for module 1 are given below).

### East Wall X-Arapucas — Module 1 (representative, x = −2145.5 mm, z_J center = 536.74 mm)

| XA | y (mm) | z_J (mm) | z_C (mm) |
|:--:|-------:|---------:|---------:|
| 1 | +1750 | 460.54 | −2456.96 |
| 2 | +1750 | 612.94 | −2304.56 |
| 3 | +950 | 460.54 | −2456.96 |
| 4 | +950 | 612.94 | −2304.56 |
| 5 | +1350 | 160.54 | −2756.96 |
| 6 | +1350 | 312.94 | −2604.56 |
| 7 | +1350 | 760.54 | −2156.96 |
| 8 | +1350 | 912.94 | −2004.56 |

For all other East modules, apply the same z-offset pattern to the respective module z-center. For West modules (x = +2145.5 mm), the z offsets are negated (180° Y rotation), same as for PMTs.

---

## 8. GDML ↔ Wire JSON Cross-Check

| Quantity | GDML (computed) | JSON (measured) | Match |
|----------|:--------------:|:--------------:|:-----:|
| East plane 0 x | −2014.5 mm | −2014.5 mm | ✓ |
| East plane 1 x | −2017.5 mm | −2017.5 mm | ✓ |
| East plane 2 (collection) x | −2020.5 mm | −2020.5 mm | ✓ |
| West plane 0 x | +2014.5 mm | +2014.5 mm | ✓ |
| West plane 1 x | +2017.5 mm | +2017.5 mm | ✓ |
| West plane 2 (collection) x | +2020.5 mm | +2020.5 mm | ✓ |
| Wire pitch (collection) | 3.0 mm | 3.0 mm | ✓ |
| Wire pitch (induction, projected) | 3.0 mm | 2.999 mm | ✓ (< 0.1% diff) |
| Channels per U or V plane | 1984 | 1984 | ✓ |
| Channels per Y (collection) plane | 1670 | 1670 | ✓ |
| Wire plane y span | 4000 mm | 4000 mm | ✓ |
| Wire plane z span (collection) | 5007 mm | 5007 mm | ✓ |

**All quantities agree. No discrepancies found.**

---

## 9. Notes and Caveats

1. **U/V naming swap in East TPC.** The GDML comment at line 6483 of `sbnd_v02_06_base.gdml` states that `volTPCPlane_V` is placed at the U position in the East TPC because the East induction planes are rotated 180° relative to the West. The wire JSON uses the drift-order convention (`ident` 0 = closest to cathode) rather than the U/V name, so East ident=0 is at +60° and West ident=0 is at −60°.

2. **CPA is a structural assembly, not a box.** The electrically active surface (foil panels) covers roughly y ∈ [−2070, +2070] mm and z_J ∈ [1223, 3787] mm, but the exact active area is determined by the 16-panel mosaic, not a single bounding box.

3. **PMT photocathode vs. volume center.** The coordinates in Section 6 are the centers of the PMT optical volume (`vol_PMT_in`), not the hemispherical photocathode face, which is shifted slightly in x toward the TPC interior.

4. **X-Arapuca local z offsets.** The full 192-position table was not computed exhaustively; the pattern from Module 1 applies to all modules with the appropriate module-center z substitution and East/West z-flip.

5. **JSON z frame.** Any overlay of GDML-derived coordinates onto Wire-Cell outputs must use z_J (= z_C + 2917.5 mm), not z_C.

---

## 10. v02_02 vs v02_06 Differences

The `wire-cell-bee3` BEE geometry is based on **v02_02**. Files downloaded from the same `v10_20_05` tag: `sbnd_v02_02_base.gdml`, `sbnd_v02_02.gdml`, `sbnd_v02_02_nowires.gdml`.

### Changed quantities

| Quantity | v02_02 | v02_06 | Δ |
|----------|:------:|:------:|:-:|
| TPC box dz (`dimDetector_Z`) | **5094.0 mm** | 5010.0 mm | −84 mm |
| Active LAr dz | **5094.0 mm** | 5010.0 mm | −84 mm |
| z_J TPC-frame origin offset | z_J = z_C + **2959.5** mm | z_J = z_C + 2917.5 mm | −42 mm |
| PDS module z_J centers | **578.74, 1911.58, 3182.42, 4515.26** mm | 536.74, 1869.58, 3140.42, 4473.26 mm | −42 mm each |

The z_J shift of −42 mm is a purely geometric consequence of the TPC being 84 mm shorter in z (42 mm removed from each end). The PDS world-frame (z_C) positions are **identical** in both versions — only the z_J values differ because z_J is measured from the TPC upstream face, which moved.

### Unchanged quantities

All other geometric quantities are identical between v02_02 and v02_06:

| Category | Quantity | Value (both versions) |
|----------|----------|-----------------------|
| TPC | dx, dy | 2022.0, 4074.645 mm |
| TPC | East/West center x | ±1011.0 mm |
| TPC | y, z_C center | 0, −412.5 mm |
| Active LAr | dx, dy | 2013.0, 4074.645 mm |
| Wire planes | Thickness | 3.0 mm |
| Wire planes | y-span | 4000.15 mm |
| Wire planes | z-span | 5013.15 mm |
| Wire planes | East x positions | −2014.5, −2017.5, −2020.5 mm |
| Wire planes | West x positions | +2014.5, +2017.5, +2020.5 mm |
| Wire planes | Pitch | 3.0 mm |
| Wire planes | U/V channels | 1984 per plane |
| Wire planes | Y (collection) channels | 1670 |
| Cathode | x position | 0.0 mm |
| Cathode | Foil panel size | 932 × 1172 mm |
| Cathode | Panels per half-wall | 16 |
| PMT | x position | ±2085.5 mm |
| PMT | Count | 120 (24 modules × 5) |
| PMT | y-group centers | ±1350, 0 mm |
| PMT | z-offsets within module | ±300, 0 mm |
| X-Arapuca | x position | ±2145.5 mm |
| X-Arapuca | Count | 192 (24 modules × 8) |
| X-Arapuca | z-offsets within module (outer rows) | ±76.2 mm |
| X-Arapuca | z-offsets within module (center row) | ±223.8, ±376.2 mm |

### Summary

The sole geometric difference is a **−84 mm reduction in TPC z-length** going from v02_02 to v02_06 (509.4 cm → 501.0 cm). Every other structural element — wire geometry, cathode, and the entire PDS in world coordinates — is unchanged. When using v02_02-based BEE geometry alongside v02_06 wire or simulation data, only the TPC z-extent and z_J frame offset need correction.

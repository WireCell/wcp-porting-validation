# PDVD wire geometry: U/V vs W, top vs bottom, v3 vs v4

Measured directly from the shipped wire files (read-only inspection, 2026-06-07):

- `wire-cell-data/protodunevd-wires-larsoft-v3.json.bz2` (production, Apr 21 2025, 193 082 B)
- `wire-cell-data/protodunevd-wires-larsoft-v4.json.bz2` (experimental, Jun 7 2026, 194 054 B)

> **A local `v5` now exists** = v4 with the induction U/V planes shifted along the W pitch
> to fix the U/V-vs-W registration measured from real tracks, assigned **by the two
> registration types of this table** (+13.2 mm for type A {0,2,5,7}, −9.8 mm for type B
> {1,3,4,6}) so the bottom↔top mirror symmetry is preserved. Validated + corroborated by a
> per-anode blob-count recovery. See `pdvd-uvw-wire-offset-calibration.md` (built with
> `pdvd/make_v5_uvwcal.py`, kept local).

Both files have **8 anodes (idents 0–7), 16 faces, 48 planes, 13 840 wires,
12 288 channels**. `v4` = **v3's channel assignment + the v5 GDML wire positions** (see the full
audit in the toolkit repo: `img/docs/protodune-wire-geometry-channel-mapping-audit.md`).
The production config currently uses **v3**, but **v4 is clearly better in imaging**
(see the §3 correction) and is preferred going forward.

## TL;DR

- **U/V relative to W.** Within each face the planes stack in drift (X) order
  **U → V → W**, separated by **0.2 mm** steps in X, with W (collection) the
  plane behind. Wire **angles** (measured in the Y–Z plane, from the +Z axis):
  **W = 90°** (vertical), **U and V at ±30° off vertical**. Pitch:
  **U = V = 7.65 mm**, **W = 5.10 mm**. U/V have ~286 wires/plane, W has 292.
- **Top vs bottom.** Anodes **0–3 = bottom CRP** (at X ≈ −3415 mm),
  anodes **4–7 = top CRP** (at X ≈ +3415 mm). They are **mirror images** about the
  cathode (X = 0): the U/V wire angles are **swapped** between top and bottom, and
  the U–W / V–W X-offsets flip sign. Angles/pitch/counts are otherwise identical.
- **v3 → v4.** The **large (~5.5 mm) transverse shift is on the bottom CRP only**
  (anodes 0–3). The top CRP (4–7) is **not** untouched, though: its **U and V
  planes also move ~0.1–0.4 mm** (while top W stays fixed), so the U/V-vs-W
  registration changes on *both* CRPs. No change in the drift (X) direction, in
  angles, in counts, or in the channel↔plane map.

---

## 1. U/V wires relative to W

Wire direction is measured in the transverse Y–Z plane; angle is `atan2(dy, dz)`
reduced to `[0,180)`, so **90° = vertical (along Y)**. Numbers below are the
v3 file (v4 angles/pitch are identical to ≤0.01°):

| plane | angle (Y–Z) | pitch | wires/plane | role |
|---|---|---|---|---|
| U | bottom 30.0° / top 150.0° | 7.65 mm | 286–287 | induction (1st seen) |
| V | bottom 150.0° / top 30.0° | 7.65 mm | 286–287 | induction (2nd seen) |
| W | 90.0° (both) | 5.10 mm | 292 | collection |

- **Orientation.** W wires are **vertical** (along Y). U and V are the two
  induction planes at **±30° from vertical** — i.e. ±60° from the W wire's
  perpendicular. U and V are mirror images of each other across the vertical.
- **Drift-direction stacking (X), per face.** The three planes sit at slightly
  different X, ordered the way charge crosses them — **U closest to the drift
  volume, then V, then W behind**:

  | plane | bottom X (mm) | top X (mm) | offset from W |
  |---|---|---|---|
  | U | −3415.1 | +3415.1 | bottom +0.4 / top −0.4 mm |
  | V | −3415.3 | +3415.3 | bottom +0.2 / top −0.2 mm |
  | W | −3415.5 | +3415.5 | 0 (reference) |

  So the three planes are stacked **0.2 mm apart in X** (U–V–W), as expected for
  the thin CRP PCB stack. The sign flips between top and bottom because the two
  CRPs drift in opposite directions.

### Pitch of the first U/V wire relative to W

The §1 table above gave the **drift (X)** offsets. The *transverse* relationship —
the pitch direction and where the first wire of each plane sits — is the following.
Here "relative to W" means measured along **W's pitch direction**, which for the
vertical W wires is the **+Z axis**. Wires are numbered in increasing pitch, so
"first wire" = wire index 0.

- **Pitch direction (robust).** U's and V's pitch directions each make **60° with
  W's pitch** (= 30° off the horizontal Z axis), tilted to opposite sides in Y.
  The Y-component **flips between top and bottom** — i.e. U and V swap which side
  they lean, exactly mirroring the wire-angle swap:

  | CRP | U pitch dir (Ŷ, Ẑ) | V pitch dir (Ŷ, Ẑ) | angle to W pitch |
  |---|---|---|---|
  | **bottom (0–3)** | (−0.866, +0.5) | (+0.866, +0.5) | 60° each |
  | **top (4–7)** | (+0.866, +0.5) | (−0.866, +0.5) | 60° each |

- **First-wire offset (Z position of U[0]/V[0] vs W[0]).** The first U/V wire
  centre sits a small distance **ahead of the first W wire** along +Z. This offset
  is **sub-pitch but not uniform** — the two faces of an anode swap the U↔V value,
  and each CRP contains two registration "types" of anode. Measured for **v3** and
  **v4** (mm, along +Z; shown as `v3 → v4`):

  | CRP | anode type | face | U[0]−W[0] | V[0]−W[0] |
  |---|---|---|---|---|
  | **bottom** | 0, 2 | front / back | 1.78→1.43 / 1.22→1.05 | 1.22→1.05 / 1.78→1.43 |
  | **bottom** | 1, 3 | front / back | 11.69→11.89 / 3.48→3.85 | 3.48→3.85 / 11.69→11.89 |
  | **top** | 4, 6 | front / back | 3.48→3.85 / 11.68→11.89 | 11.68→11.89 / 3.48→3.85 |
  | **top** | 5, 7 | front / back | 1.22→1.05 / 1.78→1.43 | 1.78→1.43 / 1.22→1.05 |

  The takeaway: the **pitch direction** of U/V vs W is a clean, fixed 60° that
  mirror-flips top↔bottom (identical in v3 and v4); the absolute **first-wire
  transverse offset** is small (≈1–12 mm, i.e. ≤ ~1.5 pitch) and depends on the
  specific anode and face (front/back), because which physical wire becomes
  "wire 0" follows the wrapping.

  > The **data-measured correction** to this U/V-vs-W registration is given in
  > [`pdvd-uvw-wire-offset-calibration.md`](pdvd-uvw-wire-offset-calibration.md),
  > including the conversion between a W-plane shift and a +Z shift of the first
  > U/V wires: **move U & V toward +Z by 3.3 mm (bottom CRP) / −2.45 mm (top CRP)**
  > — a *common-mode* U=V shift on top of the differential offsets tabulated here.

- **v3 → v4 (this offset does change).** The offset shifts by **−0.35 to +0.37 mm**
  on *every* anode — including the top CRP. This is because the v3→v4 (v5-geometry)
  move is **not a perfectly rigid per-CRP translation of all three planes
  together**: on the bottom CRP, W shifts ~5.5 mm while U/V shift ~5.32 mm (along
  slightly different transverse directions), and on the top CRP only U/V get the
  tiny ~0.27 mm v5 residual while W stays put. So the U/V-vs-W *registration*
  changes at the few-tenths-of-a-mm level even though the overall geometry move is
  near-rigid. (The numbers in §1 and §2 — angles, pitch, X-stacking — are identical
  in v3 and v4 to <0.01°.)

---

## 2. Top (anodes 4–7) vs bottom (anodes 0–3)

PDVD is two CRPs of 4 anodes each, mirror-symmetric about the cathode at X = 0:

| | anodes | X position | U angle | V angle | W angle |
|---|---|---|---|---|---|
| **bottom CRP** | 0, 1, 2, 3 | −3415 mm | 30° | 150° | 90° |
| **top CRP** | 4, 5, 6, 7 | +3415 mm | 150° | 30° | 90° |

Key differences:

- **X side / drift direction is opposite** (bottom at −X, top at +X).
- **U and V wire angles are swapped** between top and bottom (a consequence of the
  mirror): a U wire on the bottom is parallel to a V wire on the top.
- **U–W / V–W X-offsets flip sign** (table in §1).
- Pitch, wire counts, wrapping, and the channel↔plane assignment are **identical**
  across all 8 anodes and both CRPs.

---

## 3. v3 → v4 wire shift (measured, matched wire-by-wire)

`v4` carries the **v5 GDML positions** on top of v3's channels. The shift is
purely transverse (no drift-X component, no angle change). `mean |shift|` is the
3-D midpoint displacement; `pitch-comp` is its projection onto each plane's pitch
direction (the component that actually moves the image):

### By CRP group

| group | plane | mean \|shift\| | max | pitch-direction comp. |
|---|---|---|---|---|
| **bottom (0–3)** | U | 5.32 mm | 5.52 | **2.48 mm** |
| | V | 5.32 mm | 5.52 | **2.48 mm** |
| | W | 5.50 mm | 5.50 | **5.50 mm** |
| **top (4–7)** | U | 0.32 mm | 0.42 | 0.27 mm |
| | V | 0.32 mm | 0.42 | 0.27 mm |
| | W | 0.00 mm | 0.00 | 0.00 mm |

### Per anode

The shift is **uniform within each CRP** — all four bottom anodes move the same,
all four top anodes move the same:

| anode | U pitch-comp | V pitch-comp | W pitch-comp |
|---|---|---|---|
| 0 | 2.48 mm | 2.48 mm | 5.50 mm |
| 1 | 2.48 mm | 2.48 mm | 5.50 mm |
| 2 | 2.48 mm | 2.48 mm | 5.50 mm |
| 3 | 2.48 mm | 2.48 mm | 5.50 mm |
| 4 | 0.27 mm | 0.27 mm | 0.00 mm |
| 5 | 0.27 mm | 0.27 mm | 0.00 mm |
| 6 | 0.27 mm | 0.27 mm | 0.00 mm |
| 7 | 0.27 mm | 0.27 mm | 0.00 mm |

### What actually moved (per-plane shift vector, dy, dz in mm)

Resolving the shift into its Y/Z components per plane (face-0, representative
anode of each CRP) shows it is **not** a single rigid bottom-CRP translation:

| | plane | bottom (anode 0) | top (anode 4) |
|---|---|---|---|
| | U | (+0.275, +5.257) | (+0.275, +0.243) |
| | V | (−0.143, +5.375) | (−0.143, +0.126) |
| | W | (0, +5.500) | (0, **0**) |

Two distinct things are happening:

1. **A large ~5.5 mm Z shift of the bottom CRP** (all three of its planes;
   direction alternates sign by anode). Top W does **not** move.
2. **A small in-plane adjustment of the U and V planes that is common to BOTH
   CRPs** — note the U dy = +0.275 and V dy = −0.143 are *identical* on top and
   bottom, plus a ~0.13–0.25 mm Z piece. This is why the **top CRP's U/V planes
   move ~0.13–0.37 mm even though top W is fixed**.

**Interpretation.** So "v5 moved only the bottom CRP" is only true for the *large*
shift. The bottom CRP gets the ~5.5 mm (W) / ~2.5 mm (U/V pitch-direction) move
(near-rigid, ~0.2 mm per-CRP residual). But v5 *also* nudges the **U/V planes on
both CRPs** by a few tenths of a mm relative to a fixed W, so the U/V-vs-W
registration changes everywhere — exactly the sub-mm offset changes seen in §1.

**Correction (2026-06-07): v4 is clearly better than v3 in imaging.** An earlier
version of this note reported that re-imaging run 39324 with v4 gave blob counts
"essentially identical to v3" and concluded v4 "does not close the gaps" — that was
a **mistake in the comparison run**, not a real result. v4 (v5 positions) **does**
improve the image. The benefit is in the **absolute wire positions**; a **separate,
larger U/V-vs-W registration offset (~2–2.6 W-pitch) survives into v4 unchanged** and
is what still limits the imaging. That offset has now been measured directly from
real tracks (run 39324) and turned into a concrete fix — see
[`pdvd-uvw-wire-offset-calibration.md`](pdvd-uvw-wire-offset-calibration.md).

---

## 4. Caveat — the open issue is channel→wire, not geometry

The wire **positions** (this doc) are well understood and v3↔v4↔v5 agree up to the
bottom-CRP shift above. The unresolved PDVD issue is the **within-plane
channel→wire assignment**: the shipped v3's U/V channel numbering follows a
drift-order / channel-block convention that the current wire-cell converter does
**not** reproduce (it differs on 12/16 U, 16/16 V, 12/16 W face-planes), and no
upstream test validates it (`test_gdml_integration_v4` checks only Z-plane
geometry). This is *not* a ±3-style shift and *not* a position error. v3's
channel→**plane** map and channel count match `PD2VDTPCChannelMap_v2` 100%; the
open question is the channel→**wire** ordering, which can only be settled with a
LArSoft `PD2VDChannelMapService` wire-dump. Full analysis in the toolkit repo:
`img/docs/protodune-wire-geometry-channel-mapping-audit.md`.

**Now measured from data.** This residual U/V-vs-W offset has since been pinned down
directly from real tracks (run 39324): it is **~2.5 W-pitch (bottom CRP) / ~1.8
(top)**, present in both v3 and v4, and removable by a per-CRP W-plane (or
symmetric U/V) shift — see
[`pdvd-uvw-wire-offset-calibration.md`](pdvd-uvw-wire-offset-calibration.md).

---

*Numbers measured with a read-only traversal of the v3/v4 wire schema files
(`wirecell.util.wires.persist.load`); angles in the Y–Z plane from +Z, pitch as
median adjacent-wire perpendicular spacing, v3→v4 shift matched wire-by-wire in
lockstep plane order.*

# SBND cathode-plane (CPA) structure-exclusion fiducial volume

A simple, retunable description of the region occupied by the SBND **cathode
(CPA) mechanical structure**, approximated by a few overlapping axis-aligned
boxes, defined **per TPC** in the **wire-cell-toolkit coordinate system**.

This file (`cathode_fiducial.py`) remains the **design/validation source of
truth** for the box geometry. It has now been **ported into the toolkit** as the
cushion-configurable jsonnet helper
`cfg/pgrapher/experiment/sbnd/cathode_fiducial.jsonnet` (re-exported via
`sbnd_xin/cathode_fiducial.jsonnet`), whose box bounds match this `.py` to 3
decimals at cushion 0 and 0.5. It is consumed by `QLMatching` (cathode-end
`flag_at_x_boundary`, via the `cathode_fiducial` tn; see
`match/docs/qlmatching-code.md` §4.1a) and is reusable by any code needing a CPA
structure-exclusion volume via the toolkit `IFiducial` interface. See
[§5](#5-how-to-implement-this-in-the-toolkit) for the mechanics.

| File | Role |
|---|---|
| [`cathode_fiducial.py`](cathode_fiducial.py) | single source of truth — box numbers, `inside()` test, jsonnet emitter, drawing |
| [`cathode_fiducial.png`](cathode_fiducial.png) | XZ / XY / YZ drawing of the exclusion boxes (pad slab + tube lattice + knuckles) |
| `cathode_fiducial.md` | this document |

Related: [`geoDisplay.C`](geoDisplay.C) (view the GDML in ROOT),
[`sbnd_gdml_geometry.md`](sbnd_gdml_geometry.md) (full geometry breakdown).

---

## 1. Purpose

The CPA is not a flat sheet. It is a 4×4 grid of mesh/foil panels mounted on a
steel **pipe lattice** (Ø54 mm), with **knuckle joints**, a **center boss** and
**corner elbows** that stick out from the cathode plane into the drift volume.
The full GDML solid is far too detailed for analysis use. The current toolkit
config (`cfg/pgrapher/experiment/sbnd/clus.jsonnet`) only cuts a flat ±2.5 cm
slab at the cathode and is unaware of this structure.

Here we describe the **region the structure occupies** as a small set of boxes,
so it can be used as an **exclusion / veto region** near the cathode
(`contained() == true` ⟹ point is in the CPA structure ⟹ exclude it).

**Core design principle:** the 16 mesh/foil **pads are thin** (reach ~0.6 cm), so
they get only a thin slab — *no* ~2.2 cm cut. The deep (~2.7 cm) exclusion follows
**only the steel tube lattice in the gaps between the pads**, and the deepest
(~4.1 cm) only the four knuckle joints. See [§4](#4-box-decomposition--tube-lattice--thin-pads).

---

## 2. Coordinate system (wire-cell-toolkit / `sbnd_xin`)

Verified against `cfg/pgrapher/experiment/sbnd/params.jsonnet`,
`.../sbnd/clus.jsonnet`, and the v02_06 GDML. **Units: cm** (toolkit jsonnet
uses `wc.cm`; GDML is mm).

- **X = drift**, **cathode at X = 0**.
  - **TPC0 = East**: structure pokes into **X < 0**; anode/W (collection) plane
    at X = −202.05 cm.
  - **TPC1 = West**: structure pokes into **X > 0**; W plane at X = +202.05 cm.
- **Y = vertical**, centered at 0 (wires bbox Y ∈ [−200.3, +200.3] cm).
- **Z = beam** (wires bbox Z ∈ [−0.15, +501.15] cm).

**Frame conversion** GDML `volTPCActive` local → toolkit/wire frame:

| axis | conversion | note |
|---|---|---|
| X | `X = local_x − 1006.5 mm` | cathode-centered; TPC0 keeps the negative half, TPC1 is the mirror (×−1) |
| Y | `Y = local_y` | both centered at 0 |
| Z | `Z = 250.5 cm + local_z` | `volTPCActive` local z=0 → detector center (z_J = z_C + 291.75 cm, TPC center z_C = −41.25 cm) |

**TPC0 / TPC1 symmetry — open question.** The CPA is expected to be *asymmetric*
between TPC0 and TPC1. In **this v02_06 GDML**, however, the CPA structure is
*realized* as **mirror-symmetric about X = 0**: only `volCPA_East` is placed, and
`volTPCActive` is reused under `volTPC_West` with a pure 180°-Y rotation, so each
side pokes ≤ 41 mm into its own drift. (Even the separately-defined
`solidCPAInner_East` / `_West` solids are mirror images — the only difference is
the ±54 mm divide-plane offset.) The one genuine East/West difference in this
file is the **anode** U/V wire-plane swap (GDML comment: *"Plane V is placed at U
position as this wire plane is rotated w.r.t. TPC West"*), which is **not** the
cathode.

So the boxes below are currently **X-mirrored** per TPC. **If a real cathode
asymmetry is intended** (a different geometry version, the as-designed intent, or
the physical detector), the per-TPC reaches are trivial to make independent — see
the `PER_TPC_REACH` note in `cathode_fiducial.py`. This point needs confirmation.

---

## 3. CPA structure measurements

Obtained by sampling the v02_06 CPA solids (`volCPA_East` frame + mesh/foil
panels) with ROOT `TGeoShape::Contains`. All in **mm, cathode-centered**
(X = 0 at the cathode; the magnitude is the reach into one TPC's drift). These
numbers are the constants block at the top of `cathode_fiducial.py`.

| Feature | X reach (mm) | Y | Z | notes |
|---|---|---|---|---|
| Mesh panels (16) | ±0.5 | grid ±2009 | grid ±2518.5 | thin, spans whole plane |
| Foil + TPB (16) | −6.0 … −6.05 | ±2018.5 | ±2528 | ~6 mm behind the mesh |
| Pipe lattice / frame | **±27** | ±2097 | ±2606.5 | Ø54 mm pipes, whole plane |
| Center boss | ±13.5 | ±13.5 | ±29.5 | at the plane center |
| **Knuckles (×4)** | **±41** | 100 each, at y = ±517.5, ±1552.5 | 130 each, at z = 0 | deepest; on the vertical centerline |

The knuckles are the deepest-protruding parts (±41 mm), located where the
central vertical pipe crosses the horizontal pipes — i.e. on the z = 0
centerline at four y positions.

**The lattice lines** (verified by sampling at X = −2.0 cm, a depth only the
tubes/knuckles reach — the pads stop at −0.6 cm). The 16 pads are centered at
y = ±517.5, ±1552.5 mm and z = ±667, ±1942 mm; the steel pipe lattice runs in the
gaps between/around them:

| lattice member | positions (toolkit) | reach (X) |
|---|---|---|
| horizontal pipes (along Z) | y = 0, ±103.5, ±207 cm | ±2.7 cm |
| vertical pipes (along Y) | z = 250.5, 250.5±127.5 cm (= 123, 378) | ±2.7 cm |
| knuckles | (y = ±51.75, ±155.25 cm) × (z = 250.5 cm) | ±4.1 cm |

---

## 4. Box decomposition — tube lattice + thin pads

**Key design point:** the pads need only a *thin* cut; the deep (~2.7 cm) cut
belongs **only on the tube lattice between the pads**, and the deeper (~4.1 cm)
cut only at the knuckles. (A uniform 2.7 cm slab over the whole plane would
wrongly veto the thin pad area.) Boxes overlap by design — "inside the CPA
region" = inside **any** box.

Tunable parameters. The whole exclusion region is **dilated by an independent
per-side cushion in each dimension, default 0.5 cm**, applied to the pad slab,
tube bars and knuckles alike:

- `cx` — drift (X) cushion, cm; added to every reach (one-sided, deeper). Default **0.5**.
- `cy` — vertical (Y) cushion, cm; added to every Y half-extent (per side). Default **0.5**.
- `cz` — beam (Z) cushion, cm; added to every Z half-extent (per side). Default **0.5**.
- `pad` — include the thin pad slab (default `True`; set `False` for no pad cut).
- `tube_hw_cm` — tube-bar transverse half-width (default **2.7 cm** = pipe radius;
  use ~**6.1 cm** to fill the whole inter-pad gap instead of just the pipe).

Numbers below are TPC0 at **bare geometry** (`cx = cy = cz = 0`); negate X for
TPC1, `Z = 250.5 + local_z`.

### Pad slab (thin, full plane) — 1 box
X [ −(0.6 + cx), 0 ]; Y and Z span the **bounding box of the whole tube/knuckle
lattice** (cushions included), i.e. Y ±(209.7 + cy), Z 250.5 ± (252.8 + cz).
Covers the mesh (0.05 cm) + foil (0.6 cm) over the whole cathode **out to and
including the edge tube bars** — not just the foil-grid envelope (±201.85 cm).
**This is the only box over the pad area, and it is thin** — no 2.2 cm cut on the
pads.

### Horizontal tube bars (along Z) — 5 boxes
At **y = 0, ±103.5, ±207 cm**, each: X [ −(2.7 + cx), 0 ], Y = y ± (2.7 + cy),
Z = full plane. Covers the Ø54 mm pipes and the corner elbows (same ±2.7 cm).

### Vertical tube bars (along Y) — 3 boxes
At **z = 250.5, 123, 378 cm**, each: X [ −(2.7 + cx), 0 ], Y = full plane,
Z = z ± (2.7 + cz).

### Knuckle boxes — 4 boxes
At **(y = ±51.75, ±155.25 cm, z = 250.5 cm)**, each: X [ −(4.1 + cx), 0 ],
Y = y ± (5.0 + cy), Z = 250.5 ± (6.5 + cz). The deepest features, on the central
vertical line.

**Total: 13 boxes per TPC, 26 total** (plus an optional center-boss box via
`include_boss=True`, normally subsumed by the central tube crossing).

The drawing ([`cathode_fiducial.png`](cathode_fiducial.png)) shows the **exclusion
boxes only** — the physical CPA pads are *not* drawn, since they are not part of
the FV definition: **(left)** the YZ cathode face — the full-plane pad slab with
the blue tube lattice in the gaps and red knuckles; **(middle)** an XZ slice
through a pad row — thin pad band with deep notches only at the vertical
tubes/knuckle; **(right)** an XY slice through a pad column — thin pad band
(extending out past the edge tube bars) with deep notches at the horizontal tubes.

---

## 5. How to implement this in the toolkit

> Not done here — this section is the recipe. Reuse existing components; no new
> C++ should be needed.

The toolkit already has the right primitives in
`wire-cell-toolkit/aux/src/`, all implementing
`IFiducial::contained(const Point&)`
(`wire-cell-toolkit/iface/inc/WireCellIface/IFiducial.h`):

- **`BoxFiducial`** (`aux/src/BoxFiducial.cxx`) — one axis-aligned box, config
  key `bounds` is a `Ray` (`{tail:{x,y,z}, head:{x,y,z}}`).
- **`CompositeFiducial`** (`aux/src/CompositeFiducial.cxx`) — combine named
  fiducials with `logic` ∈ `and` / `or` / `nand` / `nor`.

**Recipe:** one `BoxFiducial` per box (13 per TPC = 26 total: 1 pad + 5 horizontal
tubes + 3 vertical tubes + 4 knuckles), combined by a
`CompositeFiducial { logic: "or" }`. A point is in the CPA structure when the
composite `contained()` is true → exclude it (or, for a good fiducial, AND with
`NOT` this region).

`cathode_fiducial.py` emits the exact jsonnet via `to_jsonnet(cx, cy, cz)`
(printed when you run the script). Excerpt at the default cushions
(`cx = cy = cz = 0.5`):

```jsonnet
local cpa_boxes = [
  { type: 'BoxFiducial', name: 'cpa-tpc0-pad',           // thin full-plane pad
    data: { bounds: {
      tail: { x: -1.100*wc.cm, y: -210.200*wc.cm, z:  -2.800*wc.cm },
      head: { x:  0.000*wc.cm, y:  210.200*wc.cm, z: 503.800*wc.cm } } } },
  { type: 'BoxFiducial', name: 'cpa-tpc0-htube_y+0000',  // deep horizontal tube
    data: { bounds: {
      tail: { x: -3.200*wc.cm, y:  -3.200*wc.cm, z:  -2.800*wc.cm },
      head: { x:  0.000*wc.cm, y:   3.200*wc.cm, z: 503.800*wc.cm } } } },
  // ... 4 more h-tubes, 3 v-tubes, 4 knuckles, then the tpc1 mirror (x>0) ...
];
local cpa_exclusion = {
  type: 'CompositeFiducial', name: 'cpa-exclusion',
  data: { logic: 'or', fiducials: [
    'BoxFiducial:cpa-tpc0-pad', 'BoxFiducial:cpa-tpc0-htube_y+0000',
    /* ...all 26 box names... */ ] },
};
```

Where to wire it eventually: alongside the existing SBND fiducial in
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` (which today holds the flat per-TPC
`FV_xmin/FV_xmax` cut at ±1.5 cm "data CPA face"). The cushions `cx`/`cy`/`cz`
should become jsonnet parameters so the exclusion can be loosened without
re-deriving geometry.

---

## 6. Usage

```bash
source ../venv/bin/activate          # needs matplotlib + numpy
python3 cathode_fiducial.py          # writes PNG, prints jsonnet, runs self-checks
```

In code:

All public functions default to the **0.5 cm per-side cushion** (`cx=cy=cz=0.5`);
pass `cx=cy=cz=0` for the bare geometry.

```python
import cathode_fiducial as cf
cf.inside((-0.3, 51.75, 183.8))               # -> True  (shallow, on a pad)
cf.inside((-2.0, 51.75, 183.8), cx=0, cy=0, cz=0)  # -> False (deep on a pad, bare geom)
cf.inside((-2.0, 0.0,   183.8))               # -> True  (deep, on a horizontal tube)
cf.inside((-3.5, 51.75, 250.5))               # -> True  (in a knuckle)
cf.inside((-2.0, 51.75, 183.8), cx=2.0)       # -> True  (pad slab widened by cushion)
cf.cathode_boxes(0, cx=1.0, cy=0.5, cz=0.5)   # -> list of Box for TPC0 (13 boxes)
cf.cathode_boxes(0, pad=False)                # -> drop the pad slab (tubes only)
cf.cathode_boxes(0, tube_hw_cm=6.1)           # -> widen tube bars to the full gap
cf.to_jsonnet(cx=1.0, cy=0.5, cz=0.5)         # -> toolkit config string
cf.draw("cathode_fiducial.png", cx=1.0)       # -> redraw with a cushion
```

All numbers derive from the constants block in `cathode_fiducial.py`; edit there
to retune, then re-run to regenerate the drawing, jsonnet, and this section's
numbers.

---

## 7. Out of scope (for later)

- Actual toolkit C++/jsonnet wiring and integration into `clus.jsonnet`.
- Space-charge-boundary / dead-region polygon modeling (the WCP `ToyFiducial`
  approach) — this artifact only describes the cathode-structure boxes.

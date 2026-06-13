# PDHD PDS channel→position mapping: end-to-end validation

Does an optical channel's measured PE get drawn / fit at the **correct physical
photon detector**? This note validates the PDHD OpChannel/OpDet → (x,y,z) chain
three ways — **toolkit vs DUNE code vs the input ROOT file** — and specifically
tests for a *permutation* (relabeling) bug, the failure mode where PE for one PD
is silently attributed to another.

Companion to [`pds-opchannel-opdet-mapping.md`](pds-opchannel-opdet-mapping.md)
(which explains the 256-hw-channel ↔ 160-OpDet hierarchy and per-run
instrumentation). This note is the **correctness proof** for the position join.

Source file: `/nfs/data/1/xning/wirecell-working/data/hd/onevent_run27305_final.root`
(the same `onevent_run27305_final.root` that seeded
`cfg/pgrapher/experiment/pdhd/pdhd-opdet-geom.json`). Run 27305, event 150.

---

## TL;DR

**No mapping problem.** The chain is faithful end to end:

1. **Toolkit == ROOT, exactly.** `pdhd-opdet-geom.json` reproduces
   `flashopdet/opdet_geo` to **0 mm** over all 160 OpDets, and
   `PDHDOpFlashSource` keys per-OpDet PE on the ROOT's `opdet` index directly
   (no remapping table in between).
2. **No permutation — proven independently.** The flash finder's own
   reconstructed centroid `YCenter`/`ZCenter` equals the PE-weighted centroid of
   the `opdet_geo` positions to **0.00 cm for every flash**. This is an
   *independent* quantity (not one of the position columns), so it rules out any
   relabeling between the PE-carrying `opdet` and its geometry — the exact thing
   a "mapping problem" would be.
3. **Confirmed in dunecore source.** `protodunehd` uses `ChannelsPerOpDet: 1`
   (`OpChannel == OpDet`, identity numbering, no PD-map tool). The 4 × 40
   (10 y × 4 z) layout, x = ±356 cm and the z-window pattern match the GDML
   generators and are stable across versions v2–v8.
4. **Versions *do* differ — in the X-ARAPUCA Y pitch.** run-27305's data uses
   pitch `APAphys_y/10 = 60.75 cm`; current dunecore `develop` uses 58.72 cm and
   v8 (Jul 2025) 59.2 cm — up to ~18 cm apart at the top window. The toolkit is
   safe **because it takes geometry from the data's own `opdet_geo`**, not a
   standalone GDML. Do not pair these flashes with a current dunecore PDHD GDML.

The earlier Bee3 within-block ordering fix (z-window outer descending, y-bar
inner descending) is **independently confirmed** by the raw `opdet_geo` ordering.

---

## The three sources

| Leg | What it is | Role |
|---|---|---|
| **Input ROOT** | `flashopdet/{opdet_geo, flash_opdet, opch_map}`, `opflashana/PerFlashTree`, `opflashana/PerOpHitTree` | ground truth — produced by the DUNE light-data dumper running LArSoft geometry on the PDHD GDML |
| **DUNE code** | LArSoft `geo::OpDetGeo` positions + `ChannelsPerOpDet = 1` ⇒ offline `OpChannel == OpDet` | the convention the toolkit relies on |
| **Toolkit** | `pdhd-opdet-geom.json` (extracted by `flash/test/extract_pdhd_light_maps.py`); `PDHDOpFlashSource` reads PE per OpDet | consumer under test |

### What each ROOT tree carries

```
flashopdet/opdet_geo   : opdet, x, y, z, nchan          (160 rows, opdet 0..159)
flashopdet/flash_opdet : flash_id, opdet, x, y, z, pe    (per-flash PE per OpDet)
flashopdet/opch_map    : opch, opdet, x, y, z            (224 hw rows, hardware subset)
opflashana/PerFlashTree: FlashID, YCenter, ZCenter, ...  (reco flash centroids)
opflashana/PerOpHitTree: OpChannel, PE, ...              (per-hit, OpChannel basis)
```

Critically, `flash_opdet` carries **both** the PE *and* a redundant (x,y,z) per
`opdet`, and `PerFlashTree` carries the flash finder's **independently
reconstructed** centroid. That redundancy is what makes a real validation
possible.

---

## Leg 1 — Toolkit faithfully copies the ROOT

`extract_pdhd_light_maps.py` dumps `flashopdet/opdet_geo` (cm → mm) verbatim
into `pdhd-opdet-geom.json`, keeping `opdet` as the key. Re-running the
comparison:

```
max |pdhd-opdet-geom.json − 10·opdet_geo| over 160 OpDets = 0 mm
```

`PDHDOpFlashSource.cxx` builds the per-flash PE matrix by reading
`flashopdet/flash_opdet` and indexing **directly** on its `opdet` branch:

```cpp
matrix[row*ncol + 1 + o_opdet] += o_pe;   // o_opdet from flash_opdet.opdet, 0..159
```

There is **no intermediate channel-remap table** — the column index *is* the
ROOT `opdet`, the same key as the geometry. So the toolkit cannot introduce a
PE↔position permutation; if one existed it would have to be upstream, in the
ROOT labeling itself. Leg 2 tests exactly that.

---

## Leg 2 — No permutation in the ROOT (the decisive test)

Comparing `flash_opdet`'s own (x,y,z) against `opdet_geo`'s (x,y,z) for the same
`opdet` gives **0 mismatches** — but that is the same geometry-service lookup
written twice by the same producer, so on its own it only proves
*self-consistency*, not that PE labeled `opdet N` truly came from PD *N*.

The independent test uses `PerFlashTree.YCenter`/`ZCenter`, the flash finder's
own centroid, which is **not** a position column I controlled. For each flash,
compute the PE-weighted geometry centroid from `flash_opdet.pe × opdet_geo.(y,z)`
and compare to the reco centroid:

```
ΣᵢpeᵢYᵢ / Σpeᵢ   vs   YCenter        (Yᵢ from opdet_geo[opdetᵢ])
ΣᵢpeᵢZᵢ / Σpeᵢ   vs   ZCenter
```

Result over all flashes in event 150:

```
max |ΔY| = 0.00 cm     max |ΔZ| = 0.00 cm
```

| ev | fid | recoY | geomY | recoZ | geomZ | totPE | nOpDet |
|---|---|---|---|---|---|---|---|
| 150 | 0 | 199.16 | 199.16 | 357.04 | 357.04 | 2061.8 | 4 |
| 150 | 3 | 429.03 | 429.03 | 377.25 | 377.25 | 2530.3 | 6 |
| 150 | 13 | 325.66 | 325.66 | 390.06 | 390.06 | 16528.6 | 4 |
| 150 | 24 | 275.76 | 275.76 | 310.62 | 310.62 | 5243.5 | 4 |
| … | | exact for all flashes | | | | | |

The flash finder's centroid equals Σ(pe·pos)/Σpe over these very
`opdet → position` pairs, to the last digit. That can only happen if the PE
labeling and the geometry labeling are the **same** labeling. **This rules out
the permutation/relabeling class of bug** — the one the question was really
about — using a quantity outside the self-consistent (x,y,z) columns.

---

## Leg 3 — DUNE geometry sanity

`opdet_geo` is the LArSoft geometry-service output on the PDHD GDML, so
"ROOT == DUNE geometry" holds **by construction** (not independently re-derived
here). What is independently checkable is that the resulting layout matches the
known detector:

- **160 OpDets = 4 APAs × 40**, each APA a regular **10 (y) × 4 (z)** grid —
  matches the ProtoDUNE PDS design (40 X-ARAPUCA windows behind each APA;
  arXiv:2412.15154).
- **x takes only two values**, `+356.246` / `−356.446` cm (just behind the
  ±353 cm wire planes): OpDets **0–79 on +x**, **80–159 on −x**.
- Block structure (confirmed from raw `opdet_geo`):

| OpDet | x (cm) | z-windows (cm), in index order | APA region |
|---|---|---|---|
| 0–39 | +356.246 | 427.1, 377.9, 316.7, 267.5 | +x downstream (z > 231) |
| 40–79 | +356.246 | 195.0, 145.9, 84.6, 35.5 | +x upstream (z < 231) |
| 80–119 | −356.446 | 427.1, 377.9, 316.7, 267.5 | −x downstream |
| 120–159 | −356.446 | 195.0, 145.9, 84.6, 35.5 | −x upstream |

- **Within a block the index runs z-window OUTER (descending z), y-bar INNER
  (descending y):** OpDet 0–9 share z = 427.1 cm with y = 578.9 → 32.2 cm,
  OpDet 10–19 the next z window down, etc. This is exactly the ordering the
  wire-cell-bee3 within-block fix now reproduces.

`ChannelsPerOpDet = 1 ⇒ OpChannel == OpDet` is the DUNE/LArSoft convention the
chain assumes. It is *supported* by the data — this one-sided +x event has every
`PerOpHitTree.OpChannel < 80` and every fired `flash_opdet.opdet < 80`, both
bases in the same 0–159 space — and **confirmed directly in dunecore source**
below.

---

## Leg 3b — Direct check against dunecore source (and versions)

Checked against [`DUNE/dunecore`](https://github.com/DUNE/dunecore) `develop`
(the geometry GDML generators + `geometry_dune.fcl`), to answer "could a
different DUNE geometry version remap channels?".

**OpChannel == OpDet is in the source, not just assumed.** In
`dunecore/Geometry/geometry_dune.fcl`, protodunehd resolves to
`dune_wire_readout` which sets **`ChannelsPerOpDet: 1`**. (The `ChannelsPerOpDet:
12` values in the same file belong to the legacy `dune35t` detector, not PDHD;
PDVD also uses 1 but with an explicit `PDVDPDMapAlg` + `PDVD_PDS_Mapping_*.json`
— PDHD has **no** offline PD-map tool, so its offline OpChannel is the
identity-sorted OpDet.) So the offline channel↔detector numbering is the
identity map, by construction.

**The numbering convention and z/x layout are version-stable.** The X-ARAPUCA
constants in `generate_protodunehd_v*_refactored.pl` are identical across
**v2–v8**: window 10 cm (y) × 47.75 cm (z), 4 windows/bar at z-offsets
±79.775 / ±30.625 cm about the bar centre (→ 427.1 / 377.9 / 316.7 / 267.5 cm),
10 bars/APA. These reproduce `opdet_geo`'s z-windows and x = ±356 cm exactly.

**One real version difference — the X-ARAPUCA Y pitch.** This is the thing that
*does* drift between geometry generations:

| Geometry | `PaddleYInterval` (y pitch) | value | matches run-27305 data? |
|---|---|---|---|
| **run-27305 `opdet_geo`** | `APAphys_y / 10` | **60.7499 cm** (uniform, std 0) | — (this is the data) |
| dunecore v2–v7 (`develop`) | `(APAphys_y − 2·APAFrameZSide_y) / 10` | 58.7179 cm | **no** |
| dunecore v8 (Jul 2025) | hardcoded `59.2` | 59.2 cm | **no** |

`APAphys_y = 607.49875 cm` (same `APAFrame_y = 606`, `G10 = 0.335`,
`WrapCover = inch/16` in all versions), so the data's pitch is **exactly**
`APAphys_y/10` — an *older* placement that did not inset by the two frame sides.
Current `develop` (and v8) would place the same OpDets up to
`9 × (60.75 − 58.72) ≈ 18 cm` lower at the top window. So **yes, the absolute
OpDet Y positions are version-dependent**; x and the z-window pattern are not.

**Why this does not break the toolkit.** The toolkit reads geometry from the
**same ROOT file** that carries the flash PE (`opdet_geo` and `flash_opdet`,
both keyed by `opdet`, same production). So it automatically matches whatever
geometry version the data was reconstructed with — verified by the 0 mm copy
and the 0.00 cm centroid test above. A mapping error would only appear if these
flashes were paired with a **different standalone GDML** (e.g. the current
dunecore v7/v8), which would shift Y by up to ~18 cm.

> **Practice:** keep using the data's embedded `opdet_geo` (as
> `extract_pdhd_light_maps.py` / `pdhd-opdet-geom.json` do). Do **not** substitute
> a current dunecore PDHD GDML for these 2024 flashes. Pinning the exact
> production geometry tag (which generator emitted the `APAphys_y/10` pitch)
> would require dumping the geometry from the dunesw release used for run 27305
> on a DUNE GPVM — not needed for correctness here, since the data carries its
> own positions.

---

## Observations that are *not* mapping bugs

- **`flash_opdet` in event 150 populates only odd OpDets 1–35**, while
  `PerOpHitTree` carries a denser even+odd set. These are different reco
  products (per-flash OpDet breakdown vs per-hit), and the population reflects
  the **partial / alternating instrumentation** documented in the companion doc
  (e.g. z-row 4 is odd-only). It is orthogonal to the position mapping: the
  centroid test passes regardless of *which* OpDets fire, because every OpDet
  that appears is correctly positioned.
- **`opch_map` lists `opdet` 0–59 only** (54 unique), a different range than
  `opdet_geo`'s 0–159. This is the **hardware-instrumented subset** of this run,
  *in the same OpDet numbering* — verified: `opch_map[o]` position equals
  `opdet_geo[o]` position for all 54. `opch_map` (hardware DAPHNE channel →
  OpDet) is informational; `PDHDOpFlashSource` never uses it.

---

## Reproduce

```python
import uproot, numpy as np, json
f = uproot.open(".../onevent_run27305_final.root")
geo = f["flashopdet/opdet_geo"].arrays(library="np")
G = {int(o):(float(x),float(y),float(z))
     for o,x,y,z in zip(geo["opdet"],geo["x"],geo["y"],geo["z"])}

# Leg 1: toolkit geom.json == 10*opdet_geo
gj = json.load(open(".../cfg/pgrapher/experiment/pdhd/pdhd-opdet-geom.json"))
print(max(max(abs(o["x"]-10*G[o["opdet"]][0]),
              abs(o["y"]-10*G[o["opdet"]][1]),
              abs(o["z"]-10*G[o["opdet"]][2])) for o in gj["opdets"]))  # -> 0.0

# Leg 2: PE-weighted geom centroid == reco YCenter/ZCenter (the decisive test)
from collections import defaultdict
pf = f["opflashana/PerFlashTree"].arrays(["FlashID","YCenter","ZCenter"],library="np")
fo = f["flashopdet/flash_opdet"].arrays(["flash_id","opdet","pe"],library="np")
byflash = defaultdict(list)
for fid,o,pe in zip(fo["flash_id"],fo["opdet"],fo["pe"]):
    byflash[int(fid)].append((int(o),float(pe)))
for fid,yc,zc in zip(pf["FlashID"],pf["YCenter"],pf["ZCenter"]):
    ods = byflash.get(int(fid));  W = sum(pe for _,pe in ods) if ods else 0
    if W<=0: continue
    gy = sum(pe*G[o][1] for o,pe in ods)/W
    gz = sum(pe*G[o][2] for o,pe in ods)/W
    assert abs(gy-yc)<1e-2 and abs(gz-zc)<1e-2          # holds for every flash
```

---

## Conclusion

The PDHD optical channel→position mapping is **correct end to end**. The toolkit
copies the ROOT geometry exactly and indexes PE on the same `opdet` key; the
ROOT's PE↔position join is itself unscrambled, proven by the flash finder's
independent centroid matching the PE-weighted geometry centroid to 0.00 cm;
`ChannelsPerOpDet = 1` (`OpChannel == OpDet`) is confirmed in dunecore source;
and the 4×40 layout matches the DUNE PDS design. No relabeling bug exists at any
stage. The within-block ordering used by the wire-cell-bee3 viewer (z-window
outer / y-bar inner, both descending) matches the raw `opdet_geo` index order.

The one genuine version sensitivity is the X-ARAPUCA **Y pitch** (data 60.75 cm
vs current dunecore 58.72 / v8 59.2 cm). It does not affect the toolkit because
geometry travels with the data in `opdet_geo`; the only hazard is substituting a
mismatched standalone GDML, which this validation explicitly warns against.

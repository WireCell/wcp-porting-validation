# PDHD: why do the photon detectors light up on the "APA1/3 side"?

**Symptom.** The beam enters PDHD near APA0, so one expects the lit photon
detectors (PDs) to be on the APA0 side. In the Bee display the lit PDs are on
the **APA1/3 side** instead. Is the OpChannel → PD-position mapping flipped?

**Answer: no — there is no side flip.** Two independent things explain it, and
neither is a mapping bug:

1. In **run 27305 only the +x face is cabled**, and +x **is** the WCT APA1/3
   side. So flashes can *only* appear there.
2. The PD that fire depend on which side is **instrumented**, not on where the
   beam is.

Companion docs: [`pds-channel-position-validation.md`](pds-channel-position-validation.md)
(the mapping correctness proof), [`pds-opchannel-opdet-mapping.md`](pds-opchannel-opdet-mapping.md)
(per-run instrumentation), [`qlmatching-chain.md`](qlmatching-chain.md)
(charge↔light frames), and wire-cell-bee3 `docs/protodune_geometry.md`
(the viewer geometry + the op-channel side fix).

---

## 1. The two frames agree on which side is "APA1/3"

| frame | side | which APAs | which channels |
|---|---|---|---|
| charge (WCT clustering) | `x < 0` | `group02` = APA0, APA2 | anode edge x = −353 |
| charge (WCT clustering) | `x > 0` | `group13` = APA1, APA3 | anode edge x = +353 |
| light (OpDet/OpChannel)  | `x > 0` | the APA1/3 volume       | **OpDets 0–79** |
| light (OpDet/OpChannel)  | `x < 0` | the APA0/2 volume       | **OpDets 80–159** |

So **"APA1/3 side" = +x** in *both* the charge frame and the light frame. The
charge-side assignment is confirmed empirically from the per-event Bee dumps:
`pdhd/data/<ev>/<ev>-clustering-group02.json` has its anode edge at x = −353,
`group13` at x = +353.

The OpChannel→position chain (OpDets 0–79 = +x) is proven correct end to end in
[`pds-channel-position-validation.md`](pds-channel-position-validation.md):
`OpChannel == OpDet` (`ChannelsPerOpDet:1`, no PD-map tool), the toolkit copies
the ROOT `opdet_geo` to 0 mm, and the flash finder's *independent* reco
centroid matches the PE-weighted geometry centroid to **0.00 cm** for every
flash — which rules out the permutation/relabel class of bug.

## 2. In run 27305 only +x is cabled — so only APA1/3 can light

The PDS was still being installed in 2024, so the instrumented set is
**run-dependent** (see [`pds-opchannel-opdet-mapping.md`](pds-opchannel-opdet-mapping.md)).
In **run 27305 only the +x face (OpDets 0–79) has electronics**; the −x face
(80–159) reads out nothing.

Verified directly in the `op.json` flash dumps — every measured flash puts
**100% of its PE on ch 0–79 (+x) and exactly 0 on ch 80–159 (−x)**, even though
*both* drift volumes carry heavy cosmic charge in the same events:

```
pdhd/data/{3,5,8}/<ev>-op.json   (run 27305)
  → every flash: PE(ch 0–79) = total,  PE(ch 80–159) = 0
  → meanwhile clustering-group02 (−x) AND group13 (+x) each hold ~10^5 charge points
```

The light is therefore forced onto the +x = APA1/3 wall regardless of where the
charge is. This is a **missing-cabling asymmetry, not a flip**. (Run 27980 *does*
cable the −x side; the static `ch_mask` + per-event `auto_mask` scheme absorbs
the run-to-run variation.)

## 3. Reconciling the beam premise

"Beam enters APA0 ⇒ APA0 PDs should be the lit ones" conflates two independent
things:

- **Which volume the charge is in** (physics / beam), versus
- **Which PDs are cabled** (2024 installation state).

In run 27305 the cabled side is +x = APA1/3, *independently of the beam*. A beam
(or cosmic) track on the −x / APA0 side produces **no recorded flash** in 27305.

Also note the **WCT APA index is not necessarily the official DUNE APA number**:
WCT APA0/2 = x<0, APA1/3 = x>0. If "APA0" in the expectation is the official
beam-side APA, it may correspond to a WCT APA1/3 (+x) index. The data here can't
(and need not) assert the physical beam side — the lit side is fully explained
by cabling.

**Refutation of a flip, from a working fit:** Q/L matching was *calibrated on
run-27305 anode–cathode crossers* (light-model λ and `vuv_eff` retuned; see
`project_pdhd_light_reco`). A crosser's predicted light comes from its charge;
if the optical side were flipped relative to the charge, the fit would be
nonsense across the opaque cathode. The normalization converged — so light and
charge land on the **same** physical side.

## 4. Likely cause of *this* report: the BNL twister/bee deploy is stale

The op-channel→side fix (bee3 `681617a`) and within-block y/z fix (`1a7d187`)
landed on **wire-cell-bee3 `main` on 2026-06-13**. Local repo HEAD is correct
(`events/static/js/bee/physics/experiment.js`:
`hdX = [353.002, 353.002, -353.202, -353.202]  // blocks 0,1 = +x ; 2,3 = -x`
⇒ ch 0–79 → +x).

**The phy.bnl.gov/twister/bee deployment is a separate manual step** and renders
from its own deployed copy of `experiment.js`. If it predates the fix it runs
the **old** layout (`ch = 40*APA`, blocks 0,2→−x, 1,3→+x), where a flash's drawn
side is keyed by its **z-half block, not its physical side**. Under that old map
a run-27305 flash (firing ch 0–79 = block0 ch0–39 + block1 ch40–79) is drawn on
**both** cathode sides — or on the wrong side if only one z-block fired —
producing exactly the "PDs lit on the side I don't expect" symptom.

The underlying data and toolkit mapping are correct regardless of viewer; the
artifact is purely in the deployed JS.

> **Action:** redeploy the fixed bee3 to phy.bnl.gov/twister/bee, then re-open
> the same run-27305 event — the lit PDs should now sit only on the +x / APA1/3
> wall, matching the local bee3.

## Reproduce the side check

```python
import json, numpy as np
op = json.load(open("pdhd/data/3/3-op.json"))     # run 27305
for i, v in enumerate(op["op_pes"]):
    a = np.array(v, float)
    if a.sum() <= 0: continue
    print(i, "totPE=%.0f  +x(0-79)=%.0f  -x(80-159)=%.0f" %
          (a.sum(), a[:80].sum(), a[80:].sum()))     # -x column is 0 for every flash
```

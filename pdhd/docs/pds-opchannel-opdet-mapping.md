# PDHD PDS: OpChannel vs OpDet — 256 vs 160 Explained

Investigation based on `pdhd/example_light_data/onevent_run27305_final.root`
(`flashopdet/opdet_geo` and `flashopdet/opch_map` trees).

---

## Detector hierarchy

PDHD's photon detection system uses **X-ARAPUCA bars**. Each bar contains
**4 acceptance windows** spaced along z (the long axis of the bar). In the
LArSoft geometry each *window* is one `OpDet` object:

```
1 X-ARAPUCA bar  =  4 acceptance windows (along z, ~47.75 cm each)
                 =  4 OpDet geometry objects
                 =  4 × (4 DAPHNE hardware channels)  [see below]
```

Whole-detector count:

| Level | Count | Formula |
|---|---|---|
| APAs | 4 | 2×2 layout |
| Bars per APA | 10 | |
| Windows per bar | 4 | along z |
| **Total OpDets (geometry)** | **160** | 4 × 10 × 4 |
| Bars per drift face | 20 | 10 bars × 2 bars/column in z |
| Windows per drift face | 80 | 20 bars × 4 |

The 80 x > 0 windows are OpDets 0–79; the 80 x < 0 windows are OpDets 80–159.

---

## Why 256 OpChannels

The DAQ electronics has **4 DAPHNE digitizer boards × 64 channels/board = 256**
hardware channel addresses (0–255). Each instrumented X-ARAPUCA window is connected
to **4 independent DAPHNE hardware channels** (4 SiPM readout gangs per window).
These 4 channels are summed in the DAPHNE decoder into a single offline OpChannel
(0–159), so after decoding OpChannel == OpDet index (1:1).

The `flashopdet/opch_map` tree records the pre-decoding mapping:

```
opch 0,1,2,3   → opdet 1   (same x,y,z — 4 SiPM readout channels for one window)
opch 4,5,6,7   → opdet 3
opch 8,9,10,11 → opdet 5
...
```

So:

```
256 DAPHNE hw channels (0–255)
  ├── 32 unconnected (gaps: 40–47, 88–95, 184–191, 208–215)
  └── 224 connected  →  54 windows
        (53 windows × 4 hw-ch  +  1 window [opdet 47] × 12 hw-ch  =  224)
```

Opdet 47 (y=153.66 cm, z=195.01 cm) shows `nchan=12` (3 DAPHNE quads), likely a
special test wiring at that window location.

---

## What is actually instrumented in these data

Only the **x > 0 drift face** (OpDets 0–79) has electronics in runs 27305/27980.
The x < 0 face (OpDets 80–159) is completely unequipped.

Layout of the x > 0 face (10 y-columns × 8 z-rows = 80 windows):

| z-row | z (cm) | Windows instrumented (of 10) | Bar group |
|---|---|---|---|
| 1 | 427.07 | 10/10 ✓ | bar-far |
| 2 | 377.92 | 10/10 ✓ | bar-far |
| 3 | 316.67 | 9/10 (opdet 28 absent) | bar-far |
| 4 | 267.52 | 5/10 (alternating: odd only) | bar-far |
| 5 | 195.01 | 10/10 ✓ | bar-near |
| 6 | 145.86 | 10/10 ✓ | bar-near |
| 7 | 84.61  | 0/10 — not equipped | bar-near |
| 8 | 35.46  | 0/10 — not equipped | bar-near |

Rows 1–4 (z=267–427 cm) are the 4 windows of the **far bar** for each y-column;
rows 5–8 (z=35–195 cm) are the **near bar**.  Only the top 2 windows of the near
bar (rows 5–6) are instrumented; the bottom 2 (rows 7–8) are dark.

Summary:

| Category | Windows | Bars equivalent |
|---|---|---|
| Total geometry | 160 | 40 |
| x < 0 face — no electronics | 80 | 20 |
| x > 0 row 7–8 — unequipped | 20 | 10 full bars |
| x > 0 row 4 — 5 missing | 5 | ~1.25 bars |
| x > 0 row 3 — 1 missing | 1 | 0.25 bar |
| **Instrumented** | **54** | **~13.5** |

The 54 are individual acceptance **windows**, not whole bars. Some bars are
partially instrumented (e.g. only 2 of their 4 windows cabled). The partial
instrumentation reflects the ongoing installation state during the 2024 run period.

---

## Mapping chain summary

```
Physical scintillation photon
       │  semi-analytical model (or photon library) over all 160 OpDets
       ▼
Predicted PE at each of 160 OpDet positions
       │  discard 106 unequipped (nchan=0)
       ▼
Predicted PE at 54 instrumented windows           ← model side

DAPHNE digitizer (256 hw channels, 224 connected)
       │  4 hw channels summed per window  →  DAPHNEChannelMapService
       ▼
Offline OpChannel 0–159 (== OpDet index)          ← data side
       │  opflash reconstruction
       ▼
Measured PE at ≤54 offline OpChannels
       │  opch_map: offline ch → (x,y,z) centre
       ▼
Q/L matching: compare measured vs predicted at same 54 positions
```

The `opch_map` tree provides the bridge from the DAQ hw-channel space (0–255)
to geometry positions (opdet 0–159, x/y/z).  For Q/L matching, what matters is
that measured and predicted PE are aligned on the same **OpDet index**.

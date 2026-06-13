# PDHD PDS: OpChannel vs OpDet — 256 vs 160 Explained

Investigation based on `pdhd/input_data_7p8_new_coh_grouping/onevent_run27305_final.root`
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

The instrumented set is **run-dependent** — it reflects the ongoing 2024
installation, so it differs between runs. The numbers below are **run 27305**'s
`flashopdet/opdet_geo` snapshot (the file this doc is based on): only the
**x > 0 drift face** (OpDets 0–79) has electronics there — 54 of its 80 windows
have `nchan>0`; the x < 0 face (OpDets 80–159) is entirely `nchan=0`.

> **Caveat — this is not true for every run.** Verified directly in the WCT
> opflash across 115 events (runs 27305/27980/28084/29107), the x < 0 side
> (OpDets 80–119) **does** carry real flashes in **run 27980** (11/31 events,
> PE up to several thousand); it is dark in 27305/28084/29107. So "x < 0 is
> unequipped" holds for 27305 but **not** for 27980. The Q/L matching handles
> this run-to-run variation with a per-event `auto_mask` (see
> "Dead / masked channels" below) rather than a fixed list. OpDets **120–159**
> are a separate case: they are read out in DAPHNE full-stream mode and never
> appear in the WCT opflash in any run (global max PE = 0 / 115 events).

Layout of the x > 0 face in **run 27305** (10 y-columns × 8 z-rows = 80 windows):

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

## Dead / masked channels for Q/L matching

`QLMatching` declares all **160** OpChannels (`nchan=160`,
`cfg/pgrapher/experiment/pdhd/qlmatching.jsonnet`). A channel that the photon
model predicts light on but that carries **no measured PE** biases the per-flash
χ²/ndf, so dead channels must be masked. PDHD uses the same two-tier scheme as
SBND: a **static** `ch_mask` for permanently-dead / never-readout channels, plus
a per-event **`auto_mask`** for run-dependent ones. All of this is plain config —
the machinery lives in the shared `match/src/QLMatching.cxx`.

**Static `ch_mask` (47 channels, permanent):**

| Channels | Reason |
|---|---|
| `3` | noisy (LArSoft `IgnoreChannels`) |
| `86, 87, 97, 107, 116, 117` | dead (LArSoft v1 `IgnoreChannels`; confirmed 0 PE in all 115 events) |
| `120–159` | DAPHNE **full-stream** readout — skipped by the snippet decoder, so they never enter the WCT opflash (global max PE = 0 / 115 events). They exist only as always-zero columns in the `nchan=160` matrix. |

Empirical basis: across 115 WCT opflash events (runs 27305/27980/28084/29107),
the seven listed 0–119 channels never fire, and every channel in 120–159 is
exactly 0 in every event.

**Per-event `auto_mask` (run-dependent):** within one event/TPC, a channel whose
event-max PE stays below `auto_mask_pe_low` while its nearest **live** neighbours
see light (neighbour-median PE > `auto_mask_pe_bright` in ≥ `min_contrast`
flashes) is dropped for that event. This catches a channel that is dead in
**this** run but absent from the static list — e.g. the x < 0 side, which is
cabled in run 27980 but not in 27305/28084/29107. PDHD thresholds:

| knob | value | rationale (from the 115-event study) |
|---|---|---|
| `auto_mask_pe_low` | `10` | a *live* channel's event-max PE is ≥ 31.6 (p1) / ≥ 71 (p5); a dead one is 0. 10 sits in the empty gap (cleared by 99.83% of live channels). |
| `auto_mask_pe_bright` | `50` | per-flash nonzero PE p75 = 55 — a real flash easily lifts the neighbour median above 50. |
| `auto_mask_neighbors` | `4` | K nearest live channels for the brightness reference. |
| `auto_mask_min_contrast` | `1` | ≥ 1 bright-neighbour flash required to mask. |
| `auto_mask_min_flash` | `3` | skip auto-masking below 3 flashes (too little evidence). |

Both tiers fold into the same per-channel `opdet_mask`, so prediction, χ², KS and
ndf all inherit them. The `-calib` dump marks each channel's `auto_masked` flag
(dynamic) vs the static `ch_mask`.

> Note on the colleague's list (`-1 (disconnected); 86,87,97,107,116,117,147
> (dead); 3 (noisy); 120–159 (full-stream)`): `-1` is a sentinel, not a real
> OpChannel — nothing to mask. `147` falls inside the 120–159 full-stream block,
> already covered. The rest map 1:1 onto the static `ch_mask` above.

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

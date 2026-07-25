# Geometry, Timing, and Detector Constants (`sbnd_xin/`)

The local jsonnets in `sbnd_xin/` are thin pipeline wrappers. Most detector
knowledge is imported from the shared `cfg/pgrapher/experiment/sbnd/` tree.
This document lists every physics constant used by the standalone pipeline:
its canonical source, any local override, and the reason for the override.

> For how the constants affect the Bee display see the **time offset** and
> **BEE undrift convention** sections below.

---

## SBND detector geometry summary

| Property | Value |
|---|---|
| APAs | 2 (APA0 and APA1) |
| Faces per APA | 2 |
| APA0 anode plane X | −201.45 cm |
| APA1 anode plane X | +201.45 cm |
| APA0 W collection plane X | −202.05 cm |
| APA1 W collection plane X | +202.05 cm |
| APA0 response plane X | −192.05 cm (= W + 10 cm) |
| APA1 response plane X | +192.05 cm (= W − 10 cm) |
| Cathode X (APA0 side) | −0.45 cm |
| Cathode X (APA1 side) | +0.45 cm |
| Max drift distance | ≈ 201 cm |
| Channels per APA (real) | 5638 |
| Frame length | 3427 ticks |
| Tick period | 0.5 µs |
| `tick0_time` (absolute time at tick 0) | −205 µs |

---

## Anode-plane X positions

**Source of truth:** `cfg/pgrapher/experiment/sbnd/params.jsonnet:20-25`

```jsonnet
uplane_left  =  201.45*wc.cm    // APA1 anode (positive-x drift volume)
uplane_right = -201.45*wc.cm    // APA0 anode (negative-x drift volume)
cpa_left     =   0.45*wc.cm     // cathode, APA1 side
cpa_right    =  -0.45*wc.cm     // cathode, APA0 side
```

Verification: `wirecell-util wires-info <wires-sbnd.json.bz2>` reports
`anode:0 face:0` at X ≈ [−2020.5, −2014.5] mm and `anode:1 face:1` at
X ≈ [+2014.5, +2020.5] mm.

**Local uses:**

| File | Line | Value | Context |
|---|---|---|---|
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | — | `FV_xmax: 201.05*wc.cm` | overall fiducial volume |
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | — | `FV_xmax: 201.05*wc.cm` | `a1f0pA` drift-volume block |
| `wct-img-2-bee.py` | 19, 21 | `--x0 "-201.45*cm"` / `"201.45*cm"` | bee-blobs undrift origin |

---

## Response plane and field response

**Source of truth:** `cfg/pgrapher/experiment/sbnd/params.jsonnet:27-32`

```jsonnet
// The "response" plane is where the field response functions
// start.  Garfield calcualtions start somewhere relative to
// something, here's where that is made concrete.  This MUST
// match what field response functions also used.
response_plane: 10*wc.cm, // relative to collection wires
```

The response plane sits **10 cm in front of the W collection plane**, i.e.
at X = ∓192.05 cm for APA0/APA1 (= `wplane ∓ 10*wc.cm`). The Garfield field
response file `garfield-sbnd-v1.json.bz2` (`params.jsonnet:154`) traces
drifting electrons from this plane to the wires, so without further
correction the deconvolved hit time would stamp **when the charge crossed
the response plane**, not when it was collected at W.

The simulation pipeline cancels this 10 cm offset via a ductor-open-early /
reframer-chop-front round-trip — see the **Deconvolution timing chain**
section below for how this cancellation works and what tick 0 of the
deconvolved frame actually means.

---

## Drift speed

**Source of truth (simulation):** `cfg/pgrapher/experiment/sbnd/simparams.jsonnet:16`

```jsonnet
drift_speed : 1.563*wc.mm/wc.us
```

**Local copies (all synced to 1.563 mm/µs):**

| File | Line | Value | Context |
|---|---|---|---|
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | — | `1.563 mm/us` | `BlobSampler` drift speed; `time_offset` scaling |
| `wct-clustering.jsonnet` | 32 | TLA default `1.563` | overlaid onto `simparams.lar.drift_speed` |
| `run_clus_evt.sh` | 121 | `--tla-code "driftSpeed=1.563"` | explicit TLA forwarded to above |
| `wct-img-2-bee.py` | 19, 21 | `±1.563 mm/us` | bee-blobs `--speed` arg |

Previously these four sites held three different values (1.56 / 1.563 / 1.565)
which were historical hand-me-downs from different calibration passes. The
largest discrepancy (1.56 vs 1.565) produced ≈ 1 mm of position error over the
full ~201 cm drift.

**Upstream sites also synced** (toolkit cfg tree, separate repo / commit):

| File | Line | Was | Now |
|---|---|---|---|
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | 12 | `1.56` | `1.563` |
| `cfg/pgrapher/experiment/sbnd/fhicl/standard_detsim_sbnd.fcl` | 110 | `1.565` | `1.563` |
| `cfg/pgrapher/experiment/sbnd/fhicl/wirecell_pgrapher_detsim_sbnd.fcl` | 99 | `1.59` | `1.563` |
| `cfg/pgrapher/experiment/sbnd/fhicl/wirecell_tbb_deposet_detsim_sbnd.fcl` | 98 | `1.59` | `1.563` |

The two `1.59` values were a much larger discrepancy than the others (~2 %).
The accompanying FHICL comment says only `# Electron drift speed, assumes a
certain applied E-field` — no specific E-field is named, and the value appears
in two unrelated FHICL files, both consistent with the value being a stale
default rather than a tuned non-nominal-field setting.

**LAr parameter sync (DL / DT / lifetime).** Synced across all sites to
`DL=6.5781 cm²/s`, `DT=13.1349 cm²/s`, `lifetime=35 ms` (the diffusion pair was
`4.0`/`8.8` before 2026-07-25 — see
[47_stm-bragg-reference-sbnd-retune.md](47_stm-bragg-reference-sbnd-retune.md) §6a). Touches
`cfg/.../sbnd/simparams.jsonnet`, all three `cfg/.../sbnd/fhicl/*.fcl`,
`sbnd_xin/run_clus_evt.sh`, and `sbnd_xin/wct-clustering.jsonnet`
(previously held the older `DL=6.2 DT=9.8 lifetime=10` tuning).

---

## Time offset

**Local definition:** `cfg/pgrapher/experiment/sbnd/clus.jsonnet`

```jsonnet
local time_offset = -205 * wc.us;
```

Used by:
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (`dvm.a0f0pA.time_offset`) — passed into `BlobSampler`
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (`bs_live_face data.time_offset`) — applied when sampling blob points into (x,y,z) space

### BEE undrift convention — why `wct-img-2-bee.py` uses `--t0 "200*us"` (positive)

There is a **sign-convention mismatch** between the C++ `BlobSampler` and the
Python `wirecell-img bee-blobs` converter.

`BlobSampler` (C++, `pgrapher/common/clus.jsonnet:82`) **adds** `time_offset`:

```
x = anode_x + drift_sign * drift_speed * (t + time_offset)
```

`wirecell-img bee-blobs` (`wirecell/img/converter.py:undrift_blobs`) **subtracts** `--t0`:

```
x = x0 - speed * (t - t0)
```

To make the two formulas consistent, `--t0` must be the **negative** of
`time_offset`. Since `time_offset = -205 us`, the correct argument is
`--t0 "+205*us"`.

Passing `--t0 "-205*us"` (i.e. the same sign as `time_offset`) shifts every
blob by 2 × 205 us × 1.563 mm/us ≈ 64 cm — that was the kind of bug fixed
on this branch.

**APA-specific drift direction** (`cfg/pgrapher/experiment/sbnd/clus.jsonnet`):

```jsonnet
local drift_sign = if anode.data.ident%2 == 0 then 1 else -1;
```

- APA0 (`drift_sign = +1`): drift is from anode (−201.45 cm) toward cathode (−0.45 cm), i.e. toward **+x**. `--speed` must be negative so that `x0 - speed*dt` increases toward the cathode.
- APA1 (`drift_sign = −1`): drift is from anode (+201.45 cm) toward cathode (+0.45 cm), i.e. toward **−x**. `--speed` must be positive.

Final `wct-img-2-bee.py` arguments (correct values on this branch):

| APA | `--x0` | `--speed` | `--t0` |
|---|---|---|---|
| 0 | `-201.45*cm` | `-1.563*mm/us` | `205*us` |
| 1 | `+201.45*cm` | `+1.563*mm/us` | `205*us` |

---

## Deconvolution timing chain

This section explains where the local `time_offset = -205 µs` (above) comes
from, what tick 0 of the deconvolved frame physically means, and how the
final tick → X conversion is assembled. The chain has three pieces:
(a) where tick 0 is anchored, (b) the ductor + reframer cancellation of the
response-plane offset, and (c) `OmnibusSigProc` per-plane alignment.

### (a) Where tick 0 is anchored

**Source:** `cfg/pgrapher/experiment/sbnd/params.jsonnet:130-133`

```jsonnet
// The "absolute" time (ie, relative to trigger time?) that the lower edge
// of final readout tick #0 should correspond to.
// this is the default value unless overridden with extVar in main
tick0_time: -205 * wc.us,
```

Tick 0 of the final readout frame corresponds to absolute time **−205 µs**
relative to trigger. The 205 µs pre-trigger buffer is a hardware/readout
convention and is present in both simulation and data. Note that `tick0_time`
is now a **member of `sim`** (not a `local`), so other components reference
it as `$.sim.tick0_time`; older configs used a `local tick0_time = -200*wc.us`
and the value was −200 µs.

### (b) Ductor + reframer cancellation

**Source:** `cfg/pgrapher/experiment/sbnd/params.jsonnet:135-150`

```jsonnet
// Open the ductor's gate a bit early.
local response_time_offset = $.det.response_plane / $.lar.drift_speed,
local response_nticks = wc.roundToInt(response_time_offset / $.daq.tick),

ductor : {
    nticks: $.daq.nticks + response_nticks,
    readout_time: self.nticks * $.daq.tick,
    start_time: $.sim.tick0_time - response_time_offset,
},

// To counter the enlarged duration of the ductor, a Reframer
// chops off the little early, extra time.
reframer: {
    tbin: response_nticks,
    nticks: $.daq.nticks,
}
```

Walk-through:
- `response_time_offset = 10 cm / 1.563 mm/µs ≈ 64 µs`.
- The ductor opens 64 µs earlier than `tick0_time` so it can compute the
  full field response from the response plane to W.
- The reframer chops those 64 µs back off the front of the frame, restoring
  the 3427-tick window.
- **Net effect:** the response-plane offset opens and closes; after this
  round-trip the deconvolved hit time effectively stamps **W-arrival time**,
  expressed in the absolute clock where tick 0 = −205 µs.

### (c) `OmnibusSigProc` per-plane alignment

**Source:** `cfg/pgrapher/experiment/sbnd/sp.jsonnet:53-54`

```jsonnet
ftoffset: 0.0,                      // fine-time offset (sub-tick)
ctoffset: 1.0*wc.microsecond,       // coarse-time offset (SBND override)
```

`OmnibusSigProc` combines three offsets to circular-shift the deconvolved
waveform in time:

| Offset | Source | Purpose |
|---|---|---|
| `fine_time_offset` | `ftoffset` (sub-tick) | sub-tick alignment |
| `coarse_time_offset` | `ctoffset` (integer ticks) | per-detector tuning |
| `intrinsic_time_offset` | computed per-plane from the field response | absorbs U/V vs W plane-separation delay |

The sum is applied as a circular time-shift in
`sigproc/src/OmnibusSigProc.cxx:1203-1211` (declarations in
`sigproc/inc/WireCellSigProc/OmnibusSigProc.h:110-117`). The result is that
U, V, W deconvolved hits from a single physical track land at the same
tick. SBND uses `ctoffset = +1 µs` (the common default is −8 µs); the
value is detector-tuned.

### Tick → X formula

The pieces above motivate the final formula used by both
`clus/src/BlobSampler.cxx::time2drift` and
`clus/src/Facade_Util.cxx::Facade::time2drift`:

```
X = X_W + drift_sign · (t + time_offset) · drift_speed
```

with:
- `X_W = ±202.05 cm` (W plane X, from the wire schema).
- `time_offset = -205 µs` — exactly the negative of `tick0_time`, so
  `(t + time_offset)` becomes drift duration measured from trigger time.
- `drift_speed = 1.563 mm/µs` (matches `simparams.jsonnet`; see the
  **Drift speed** section for the sync history and the remaining upstream
  stragglers).
- `drift_sign = (anode.data.ident % 2 == 0 ? +1 : -1)` — from
  `pgrapher/common/clus.jsonnet:27`. APA0 drifts toward +x, APA1 toward −x.

**Sanity check** (APA1, `drift_sign = -1`, `X_W = +202.05 cm`,
`drift_speed = 1.563 mm/µs`):

- Charge collected at W at trigger time ⇒ `t = 205 µs` ⇒ `(t + time_offset) = 0`
  ⇒ `X = X_W = +202.05 cm`. ✅
- Charge originating at the cathode (≈ +0.45 cm), full max drift ≈ 1290 µs
  ⇒ `t ≈ 1495 µs` ⇒ drift = 1290 × 1.563 ≈ 201.6 cm ⇒ `X ≈ +0.45 cm`. ✅

---

## Per-APA channel count

**Real value:** 5638 channels per APA = 1984 U + 1984 V + 1670 W (confirmed by
`wirecell-util wires-info sbnd-wires-geometry-v0206.json.bz2`).

**Previously a production bug:** `cfg/pgrapher/experiment/sbnd/img.jsonnet:47`
used `5632*ident`, dropping the last 6 W-plane wires of APA0 and the last 12
W-plane wires of APA1 (because `img.jsonnet`'s `chsel_pipes` channel selection
restricted each anode to `[0,5631]` for APA0 and `[5638,11263]` for APA1).

**Fixed (this branch):**

| File | Line | Value |
|---|---|---|
| `cfg/pgrapher/experiment/sbnd/img.jsonnet` | 47 | `std.range(5638*ident, 5638*(ident+1)-1)` |
| `wct-sp-to-magnify.jsonnet` | 118 | Same `ChannelSelector` before `MagnifySink` |

---

## Frame length and tick

| Constant | Value | Source |
|---|---|---|
| `nticks` | 3427 | `wct-sp-to-magnify.jsonnet:45` TLA default; passed to `MagnifySink` `runinfo.total_time_bin`. Matches actual SP-frame readout window (input frames are 11276 × 3427). |
| tick period | 0.5 µs | `cfg/pgrapher/experiment/sbnd/clus.jsonnet` `tick: 0.5 * wc.us` |
| `tick_drift` | `drift_speed * tick` | `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (= 0.78 µm per tick at 1.56 mm/µs) |

**Imaging tick clipping (also fixed on this branch):**
`cfg/pgrapher/experiment/sbnd/img.jsonnet` previously hardcoded `MaskSlices.max_tbin: 3400`
(line 145) and `CMMModifier.org_hlimit: [3400]` (line 89), silently dropping the last
27 ticks of every event. Both raised to 3427.

---

## Clustering physics knobs (TLAs)

Passed by `run_clus_evt.sh:121-125` and applied in `wct-clustering.jsonnet:36-43`
as an overlay on `simparams.lar`:

| Knob | Value | Units | Description |
|---|---|---|---|
| `DL` | 6.5781 | cm²/s | longitudinal diffusion coefficient (SBND physical; was 4.0 before 2026-07-25) |
| `DT` | 13.1349 | cm²/s | transverse diffusion coefficient (SBND physical; was 8.8) |
| `lifetime` | 35 | ms | electron lifetime |
| `driftSpeed` | 1.565 | mm/µs | overrides `simparams.lar.drift_speed` for clustering |
| `reality` | `'sim'` | — | `'sim'` or `'data'`; controls dead-channel treatment |

---

## Scope-aware fiducial volume (`clustering_separate` / `clustering_neutrino`)

`clustering_separate` (its `JudgeSeparateDec_2` boundary/exit test) and
`clustering_neutrino` choose their fiducial-volume bounds to match the **scope** of the
clustering pass, via the shared helper `select_scope_fv` (`clus/src/clustering_separate.cxx`,
declared in `clus/inc/WireCellClus/ClusteringFuncs.h`). The scope is read from the dv's
**configured** drift volumes (`dv->wpident_faces()`) — i.e. which APAs/faces the
`DetectorVolumes` node was built for — **not** from which TPCs happen to have live activity
in a given event. This matters: an all-APA event with charge in only one TPC must still use
the cryostat FV, which a live-activity (`Grouping::wpids()`) check would get wrong.

| Pass | configured volumes (SBND) | FV used | x-range |
|---|---|---|---|
| per-APA TPC0 | `{a0f0pA}` (1 APA) | that drift volume's FV | `[-201.05, -2.5]` cm |
| per-APA TPC1 | `{a1f0pA}` (1 APA) | that drift volume's FV | `[2.5, 201.05]` cm |
| all-APA | `{a0f0pA, a1f0pA}` (>1 APA) | `overall` cryostat envelope | `[-201.05, 201.05]` cm |

Rules: **>1 configured APA (or none) → `overall`** (cryostat); this reproduces the legacy
`dv->metadata(WirePlaneId(0))` reads bit-for-bit, so all-APA behavior is unchanged
regardless of per-event activity. **Single configured APA → the union (outermost envelope)
of that APA's configured per-`(APA,face)` blocks** (for a single-face APA like SBND, just
that one block; for a multi-face APA, the full APA even if a face is quiet). Any field
missing from a per-face block falls back to `overall`; `vertical_dir`/`beam_dir` are
detector-global and always come from `overall`.

This reverses the previous "always cryostat" convention: in a per-APA pass, "exiting"
now means leaving the **clustered drift volume**, which is the physically correct notion
when the grouping holds only one TPC's blobs.

Config (`cfg/pgrapher/experiment/sbnd/clus.jsonnet`, `dvm`): the per-`(APA,face)` blocks
`a0f0pA`/`a1f0pA` now define the full FV (x is genuinely per-TPC; y/z bounds + margins are
sourced from `overall` since SBND TPCs span the full height/length). The change is in
shared C++ but **SBND is the only config that actively runs `separate`/`neutrino`** (DUNE-VD
has them commented out), so in practice only SBND is affected. The union/multi-face path is
exercised only on multi-face detectors (untested on SBND).

---

## Shared config file map

| Local file | Imported shared config |
|---|---|
| `wct-sp-to-magnify.jsonnet` | `pgrapher/experiment/sbnd/simparams.jsonnet` |
| `wct-img-all.jsonnet` | `pgrapher/experiment/sbnd/simparams.jsonnet`, `pgrapher/experiment/sbnd/img.jsonnet` |
| `wct-clustering.jsonnet` | `pgrapher/experiment/sbnd/simparams.jsonnet` (with TLA overlay) |
| `clus.jsonnet` | `pgrapher/experiment/sbnd/clus.jsonnet` |
| `magnify-sinks.jsonnet` | *(no sbnd-specific imports)* |

All shared configs live under `$WIRECELL_PATH` → `toolkit/cfg/pgrapher/...`.

---

## Consistency with external geometry constants

For cross-reference, here is how the WCT-internal values above compare to
the externally-quoted SBND geometry numbers (standard analysis write-ups
and 2024 data calibration notes):

| External constant | WCT value | Match |
|---|---|---|
| U plane X = ±201.45 cm | ±201.45 cm (`params.jsonnet:20-23`) | ✅ |
| W plane X = ±202.05 cm | ±202.05 cm (`params.jsonnet:21-23`) | ✅ |
| Drift velocity = 1.563 mm/µs | 1.563 mm/µs (`simparams.jsonnet:16`); all `sbnd_xin/` sites now synced (see **Drift speed**) | ✅ |
| Max drift time in data ≈ 1281–1282 µs | sim geometry gives ≈ 1290 µs (= (W − CPA) / v_d); the ~8 µs gap is consistent with data's wider cathode | ✅ |
| CPA thickness in data ~±1.5 cm (~3 cm gap; DENT issue) | sim is ±0.45 cm (~1 cm gap; `params.jsonnet:24-25`) | ⚠ — known sim/data mismatch, not yet modelled |
| TPC active volume Y ±200 cm, Z 0.15–500.85 cm (wire-readout overlap) | Not in this doc; cf. `wirecell-util wires-info $WIRECELL_DATA/sbnd-wires-geometry-v0206.json.bz2` and FV_y/FV_z in `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (±199.312 cm / 0.85–500.15 cm with margins) | (verify via `wires-info`) |

**Implication for WCT geometry:** none — the only ⚠ row (CPA thickness) is a
known upstream sim/data study item, not a WCT configuration bug.

# PDVD light-flash ↔ charge-readout time chain (and where an unaccounted shift could hide)

**Purpose.** One end-to-end account of how a light flash's time is put onto the charge
(drift) time base in the *current* PDVD Q-L matching chain — from the raw-waveform DAQ
timestamps, through the preprocessing that stamps per-crate offsets into the opflash
metadata, into how `QLMatching` uses them to place charge in drift-x. Written to answer a
specific worry: **is there still an overall light↔charge time shift the chain does not
account for?** The relation differs for the **Top (TDE)** and **Bottom (BDE)** crates, which
is exactly where a residual could sit — so the top/bottom split is tracked throughout.

This doc **synthesizes** material already spread across `pdvd-ql-pending.md §1`,
`pdvd-anode-time-consistency.md §3/§6/§8.13`, `pdvd-flash-dt.md`, and
`pdhd-anode-time-consistency.md`; it does not duplicate their derivations. Its new content
is the single-picture chain, the **worked numeric trace** (§6), and the **defect-hunt +
next-check** (§7).

## Repro

```bash
cd wcp-porting-img/pdvd
python3 docs/qlmatch/trace_light_charge_time.py
# other event / flash:
python3 docs/qlmatch/trace_light_charge_time.py work/039252_6/calib-evt298651.json 88
```
Reads an existing calib dump (no wire-cell run) and prints the term-by-term trace in §6;
writes `trace_light_charge_298651_gid88.png`.

---

## 1. The three clocks (raw DAQ side — Jay's numbers, the input boundary)

All times live on the DTS 16 ns clock, masked to 40 bits (rollover 17 592.186 s). Three
windows matter, and they are **not** locked to each other the same way:

| window | placement vs trigger | stability |
|---|---|---|
| **Trigger** (CTB beam, `tc_us`) | — | the common anchor |
| **Light** full-stream window start `fs_t0_us` | trigger − fs_start = **5000.4 / 5000.6 / 2500.5 µs** (runs 039252/039253/039349) | **trigger-locked to ±0.3 µs** (per-run DAQ constant) |
| **Charge** window start (per crate) | trigger − charge_start | **floats ±15 µs event-to-event**, σ 9–12 µs |

Two facts drive everything below:

- It is the **charge** window that floats, not the light window. A per-*run* constant offset
  can therefore never do better than ±15 µs (≈ ±2.5 cm of drift) — which is why the old
  statistical A/C-crosser fit stalled at ~3σ. The fix is a **per-event** offset.
- The two charge crates (**TDE** = top CRPs, **BDE** = bottom CRPs) open on **independent
  64-sample frame boundaries, up to ~32 µs apart** (≈5 cm of drift). So "the charge window
  start" is a **per-crate** quantity, and the top/bottom offsets genuinely differ.

**Caveat (Jay).** In the PDVD decoder `raw::RDTimeStamp` carries **different units per crate**:
the 6 144 TDE channels are ns-since-epoch (frame-aligned to 32 µs), the 6 144 BDE channels
are 16 ns DTS ticks — a factor-16 apart. Within a crate all channels share one window start;
between crates the starts differ. This unit split is why §7 keeps the **top-vs-bottom**
alignment as an explicit candidate.

The raw-DAQ per-run/per-event values (`tc_us`, `charge_bde_us`, `charge_tde_us`,
`fs_t0_us`, …) come from Jay's `PDVDTriggerLightAna` and are the deterministic replacement
for the statistical fit; full derivation and the per-run summary table live in
`pdvd-ql-pending.md §1`.

## 2. Preprocessing — measuring the per-crate offsets and stamping them

The per-crate light↔charge offsets are computed once, per event, at light-reco time:

- **`pdvd/run_light_evt.sh`** (L71–116) reads the `trigoff/trigger_offset` tree jjo added to
  `*_rawwf.root`, recomputes the light chain t0 anchor identically
  (`ticks = round(ts·62.5) − where(nsamp≤1024, 64, 0)`, `t0 = min·0.016 µs`), and forms
  ```
  offset_bot_us = light_chain_t0 − charge_bde_window_start     # bottom / BDE
  offset_top_us = light_chain_t0 − charge_tde_window_start     # top    / TDE
  ```
  (≈ −2.1…−2.5 ms; **sign convention: add the offset to the flash time** to land on that
  crate's charge axis).
- These pass via `-S offset_bot_us / offset_top_us` into `wct-light-reco.jsonnet`;
  `cfg/.../protodunevd/flash.jsonnet` (L178–197) stamps them into the opflash **archive
  metadata** through `metadata_extra`. (The legacy scalar `offset_us` stays 0 — inert for
  PDVD.)
- **`data/ql_trigger_offset.txt`** and `data/jjo_triglight_offsets.txt` are referenced by the
  runner/scripts as an *optional per-run residual constant added to both crates*, but are
  **not present on disk** — so `RUN_OFF` defaults to 0 and the live term is the per-crate
  metadata offset only. `ql_light_calib/dump_light_t0.py` and `fit_trigger_offset.py` are the
  **superseded** cross-checks (the statistical fit is BLIND at ~150 flashes/window).

## 3. Units — where nanoseconds become microseconds

- The OpFlash matrix **col0 = flash time in NANOSECONDS** (PE-weighted mean of OpHit
  `peak_time`, WCT native ns): `flash/src/OpFlashFinder.cxx` L157–167, L658–668. It passes
  through `FlashTensorToOpticalPCs` and `Opflash::get_time()` **unscaled** (still ns).
- The `offset_*_us` metadata is in µs; `pdvd/wct-clustering.jsonnet` L90–91 multiplies by
  `wc.us` before handing it to `QLMatching`, so internally the offset is ns too.
- The **only** ns→µs conversions are the `/ us` at QLMatching **dump** time
  (`QLMatching.cxx` L2776/2782). So the calib-dump `f["time"]`/`f["time1"]` you read in §6
  are in **µs**, already offset-folded.

## 4. Consumption — how QLMatching places charge (the one formula)

`match/src/QLMatching.cxx` L1297–1302 (identical form at L2934, L3230):

```
flash_x_offset = sign_offset · (flash_time + trigger_offset_for(input_idx)) · drift_speed_for(input_idx)
x_drift        = x_raw + flash_x_offset
u              = s · (x_drift − anode_x)          // u=0 at anode, u=u_cathode at cathode
```

- **Per crate.** `trigger_offset_for(idx)` returns `trigger_offsets = [bottom(input0),
  top(input1)]` (`QLMatching.h` L178–189); `drift_speed_for(idx)` returns the per-side speed
  (`drift_speeds`, commit `15a7b3bf`). Input 0 = bottom/BDE volume, input 1 = top/TDE.
- **Geometry.** `sign_offset`, `s`, `anode_x`, `u_cathode` come from `compute_geometry`
  (L1049–1095) and are echoed into the calib dump's `geometry` block per side.
- **Dump/display re-basing.** `f["time"]` folds the **bottom/BDE** offset for both volumes;
  `f["time1"]` (and the opflash `op_t1`) carry the **top/TDE** clock — emitted **only** when
  per-input `trigger_offsets` are set (commit `e587f357`, so PDHD/SBND scalar output stays
  byte-identical). Consumers read `f["time1"]`; older dumps fall back to
  `f["time"] + (off_top − off_bot)`.
- **Clustering twin.** The clustering-side drift correction `x_t0cor` in
  `clus/src/PCTransforms.cxx` uses the **same** per-side `trigger_offset` (threaded through
  `clus.jsonnet`), so a matched cluster's `x_t0cor` and QLMatching's matching geometry share
  one light time base.

## 5. Top vs Bottom — the full accounting

| term | where | per-crate? | enters |
|---|---|---|---|
| `flash_time` (`get_time()`) | opflash col0 (ns) | no (one flash) | light side |
| `trigger_offset` (`offset_bot/top_us`) | metadata → `trigger_offsets` | **YES** (BDE vs TDE, per event) | folded into `f["time"]`/`f["time1"]` |
| `drift_speed` | `drift_speeds` knob | **YES** (per side, optional) | `flash_x_offset` |
| `sign_offset`, `s`, `anode_x` | `compute_geometry` | **YES** (geometry) | `u = s·(x_drift − anode_x)` |
| `ctoffset` (SP tick-shift) | `sigproc` (see §7) | **YES** (`ctoffset_b/_t`) | **x_raw**, upstream — *not* here |
| BlobSampler `time_offset = 0` | `clus.jsonnet` | n/a | x_raw (0 for data) |

Net path of one point to drift-x: **bottom** cluster → `x_drift = x_raw + (−1)·f["time"]·v`;
**top** cluster → `x_drift = x_raw + (+1)·f["time1"]·v`. The *only* light-side quantity that
differs top-vs-bottom is the offset already baked into `time` vs `time1`.

## 6. Worked numeric trace — run 039252 evt 298651, flash gid=88

A single cosmic crossing the cathode: its two halves are read out by the two independent
crates but matched to **one shared flash**. Bottom half = cluster uid 33 (2494 pts, BDE);
top half = uid 4000166 (1138 pts, TDE). (This dump used **v = 0.148073 cm/µs** — the
single-crosser estimate, *not* a canonical convention; the arithmetic is what matters.)

**The flash time, decomposed** (all µs):
```
get_time()               =  2880.002     (light full-stream window relative)
+ offset_bot (−2515.344) =   364.658     f["time"]  → BOTTOM (BDE) charge axis
+ offset_top (−2507.744) =   372.258     f["time1"] → TOP    (TDE) charge axis
Δ = time1 − time = +7.600 µs  = offset_top − offset_bot   →  Δ·v = +1.13 cm
```
Δ is **event-specific** (charge windows float per event; crates up to ~32 µs apart) — not a
fixed skew.

**Per-crate `flash_x_offset` and placement** (robust p1/p99 endpoints; anode u=0,
cathode u=336.91):

| crate | n | clock (µs) | `flash_x_offset` = sign·t·v | u_anode_end | u_cath_end |
|---|---|---|---|---|---|
| BDE bottom | 2494 | 364.66 (`time`) | −1·364.66·0.148073 = **−54.00 cm** | −1.7 | 324.9 |
| TDE top | 1138 | 372.26 (`time1`) | +1·372.26·0.148073 = **+55.12 cm** | 18.6 | 345.9 |

Both halves span anode→cathode in the common `u` frame (see the PNG). Two checks:

- **Top-vs-bottom is real.** Using the *wrong* (bottom) clock for the top half moves its
  cathode end from u = 345.89 to 347.01 — a **−1.13 cm** mis-placement, exactly Δ·v. This is
  why `time1`/`op_t1` exist.
- **Self-consistency with the code.** A small `at_cathode`-flagged fragment (flash gid 137,
  bot uid 4) computed by this script lands at u = [336.68, 336.98] vs u_cathode = 336.91 —
  i.e. the script's `u` reproduces the flag QLMatching itself set. The arithmetic above *is*
  the code's arithmetic.

**Read this the right way:** every term in the table is `trigger_offset` (folded into
`time`/`time1`) and `drift_speed`. The whole light↔charge registration rides on those two.
Nothing else appears — which is the point of §7.

## 7. Where an unaccounted overall shift could hide — and the next-check

The §6 arithmetic is the **flash-side** registration and it is closed and self-consistent. An
unaccounted overall light↔charge shift therefore cannot live there; it can only live on the
**charge side**, in how `x_raw` itself is anchored in time — plus one non-time confound to
keep separate.

1. **Frame-tick-0 vs DAQ charge-window-start alignment — the primary candidate.**
   `offset_bot/top` are defined as `light_t0 − charge_{bde,tde}_window_start`. But the
   drift-x arithmetic assumes `x_raw`'s anode edge corresponds to the WCT **frame tick 0**.
   If WCT's frame tick 0 does **not** coincide with the DAQ `charge_window_start` that
   `offset_bot/top` reference, a **constant** shift enters — same sign on both crates. And a
   **bottom-vs-top difference** in that alignment is exactly what the `RDTimeStamp` unit
   inconsistency (§1: TDE ns-epoch vs BDE 16 ns-DTS ticks) would produce. This is the
   top/bottom differential the original worry is about, and it is **not** captured by
   `trigger_offsets` (which only encode the *window-start* difference, not any tick-0-vs-
   window-start gap).

2. **SP tick-shift (`ctoffset` + intrinsic FR term) — a fixed per-crate charge registration.**
   `OmnibusSigProc` shifts the deconvolved charge by
   `time_shift = (m_coarse_time_offset + m_intrinsic_time_offset) / m_period`
   (`sigproc/src/OmnibusSigProc.cxx` L1321), where `m_intrinsic_time_offset = fr.origin /
   fr.speed` (L937). So the SP registration is **`ctoffset` (currently +4 µs per side, §8.13)
   plus** an intrinsic field-response term — *not* `ctoffset` alone, and **not derived from
   any flash**. If either crate needs different centering, the per-side `ctoffset_b/_t` is the
   only handle; §8.13 fixed +4 µs by requiring the gauss peak to sit on the raw collection
   rising edge (median residual +2.07 µs = real SP shaping, not a clock error).

3. **`BlobSampler time_offset = 0` is correct, not a leak.** For data there is no per-event
   T0 baked into `x_raw`; the sim's `tick0_time = −250 µs` term is **replaced** by the flash
   mechanism (§4), so keeping it would double-count. This is *not* a place a residual lives —
   it is called out only to forestall that misreading.

4. **Velocity mismatch is an x error, not a time offset.** The known §6.2 "+6.6 µs ≈ +1 cm
   cathode-ward" excess is **entangled with the undecided drift velocity** (1.53 FR / 1.568
   recon / 1.586 cathode-pinned / 1.481 single-crosser). A wrong `v` shifts `x` without any
   clock being wrong. When hunting a *time* shift, hold `v` fixed and read at the **anode**
   (where drift distance → 0, so a velocity error vanishes and only a true time offset
   survives). Do not let a velocity-induced x-shift masquerade as a timing defect.

### Next-check (the discriminating one)

Do **not** use `PDVD_QL_DIAG=1` (forcing `trigger_offset = 0`) as the test: the same code
path applies whatever offset value it is handed, so that only verifies *application*, and
will pass regardless of whether the value — or the frame-tick-0 anchor — is physically right.

Use the **anode-stop** test instead, for which the infrastructure already exists
(`pdvd-anode-time-consistency.md §8.14`, `check_anode_stop_ensemble.py`): take
anode-touching tracks, place their charge end by the matched flash, and ask **where the anode
end lands relative to u = 0**. Then:

- a **systematic same-sign offset on both crates** = the unaccounted **constant**
  (candidate 1a and/or the SP term 2);
- a **bottom-vs-top difference** at the anode = the **RDTimeStamp / crate-alignment**
  asymmetry (candidate 1b).

§8.14 already reports the W-plane anode signal stopping within **0.5–0.8 cm of u = 0 on both
crates across 120 events / 3 runs** — i.e. any unaccounted *constant* light↔charge shift is
bounded at the ~½-cm (~3 µs) level today. The next-check is to re-read that ensemble
**anode** placement holding `v` fixed and reporting the **bottom-vs-top** split explicitly:
if the two crates disagree by more than the ~0.5 cm read precision, candidate 1b (the
top-vs-bottom alignment the original worry names) is the culprit; if they agree but both sit
off u = 0, it is a common constant (1a/2).

---

## Cross-references

- `pdvd-ql-pending.md §1` — raw-DAQ derivation, per-run offset summary, beam-flash closure
  (99/120 events, residual −0.9 µs).
- `pdvd-anode-time-consistency.md` — **§3** full clock chain formula-by-formula; **§3.4** the
  per-event BDE/TDE Δ table (039252); **§6.2** the +6.6 µs / drift-velocity open item;
  **§8.13** ctoffset = +4 µs verdict; **§8.14** the anode-stop ensemble the next-check builds on.
- `pdvd-flash-dt.md` — per-PD-group time coincidence (≤64 ns, no per-type correction needed).
- `pdhd-anode-time-consistency.md` — the PDHD precedent (constant 249.808 µs offset, measured
  not fitted).

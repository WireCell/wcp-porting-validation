# PDHD run 29107 — event 1015 light anomaly (ADC saturation on the −x wall)

Event 1015 of run 29107 reconstructs with far more flashes and far higher total
PE than any other event in the run, while the TPC **charge looks normal**. This
note explains what is wrong: a bright light flash **saturated the 14-bit DAPHNE
ADC** on the −x photon-detector wall. The clipped (flat-topped) waveforms then
(a) over-integrate in deconvolution → the huge PE, and (b) fragment/ring into many
spurious hits → the huge flash count. It is a **detector dynamic-range effect in
the raw data, not a reconstruction bug** — it reproduces identically in the LArSoft
`opflashana` product and in our independent WCT light chain.

> Companion docs: `pdhd-light-flash-run-comparison.md` (cross-run LArSoft flash
> stats), `run27980-processing-status.md`, `pdhd-light-raw-data.md`.

Data used (all under `pdhd/work/`):
- `029107_1015/opflash_pdhd.tar.gz` — LArSoft `opflashana` opflash/ophit tensors.
- `029107_allpd1015_nocut/opflash_pdhd-allpd-wct.tar.gz` — our WCT `wct-flash` all-PD reco.
- `029107_allpd1015_nocut/snippet_decoana.root` — **raw + deconv snippet waveforms** (the smoking gun).
- `029107_1047*` — a representative normal event for comparison.

OpDet side map: **+x = 0–79**, **−x self-trigger = 80–119**, **−x full-stream = 120–159**.

---

## 1. Event 1015 is a dramatic outlier

LArSoft `opflashana` product, event-by-event over the 29107 sample:

| quantity | evt 1015 | run median | ratio |
|---|---|---|---|
| n flashes | **1347** | 398 | ≈3.4× |
| total PE | **2.09 M** | 226 k | ≈9.2× |
| n ophits | **16 154** | 10 291 | ≈1.5× |

The hit count is only ~1.5× but the PE is ~9×, so the **PE-per-hit is hugely
inflated** — a few hits carry enormous PE.

## 2. The excess is localized to the −x self-trigger wall (80–119)

Total PE per APA block (evt 1015 vs normal evt 1047):

| block | OpDets | PE evt 1015 | PE evt 1047 | ratio |
|---|---|---|---|---|
| +x upper | 0–39 | 422 k | 73 k | ~6× |
| +x lower | 40–79 | 145 k | 127 k | ~1× |
| **−x self** | **80–119** | **1.53 M** | **58 k** | **≈26×** |
| −x full-stream | 120–159 | 0 | 0 | — |

The +x wall is roughly normal; essentially the entire excess is on the **−x
self-trigger block**, where 32–33 of the 40 channels fire one giant pulse at
the same time.

## 3. The PE is dominated by one bright prompt burst with unphysical width

~86 % of the event's total PE sits in a single ~150 µs window. The dominant hits
have **widths of 11–16 µs** — i.e. they span essentially the *entire* 1024-tick
(16.4 µs) self-trigger snippet record. That is physically impossible for liquid-argon
scintillation (slow component τ ≈ 1.6 µs; even very late light is gone within a few µs).
A hit as wide as the whole readout record is the signature of a **clipped/over-integrated
waveform**, not real light.

## 4. Root cause — 14-bit DAPHNE ADC saturation (confirmed in the raw ADC)

Reading the raw snippet waveforms from `snippet_decoana.root` for the −x channels:

- The brightest −x snippets **rail hard at 16383 = 2¹⁴ − 1**, the DAPHNE 14-bit ADC
  ceiling, and stay **clipped flat-top for ~211 samples (~3.4 µs)**. Pedestal ≈ 8000,
  so the pulse overflows the full ~8000-ADC headroom and saturates.
- Example, raw `ch80`: pedestal ≈ 7996, max = 16383, with the peak region a hard flat
  plateau of constant 16383 across tens of consecutive samples.

| event | saturated −x snippets (raw ≥ 16383) | distinct −x channels saturated |
|---|---|---|
| **1015** | **35** | **33** |
| 1047 (normal) | **0** | **0** |

A clipped flat-top deconvolves to a broad plateau, so the OpHitFinder integrates the
whole record as one ~16 µs hit → ~40 k PE per saturated channel → the ~9× total-PE
blow-up. Normal events have **zero** saturated snippets.

## 5. Both reconstructions blow up the same way → raw-data effect, not a reco bug

The LArSoft `opflashana` chain and our WCT all-PD chain run **independent
deconvolutions off the same raw ADC**. Both explode on event 1015 and both are
normal on 1047:

| reco | quantity | evt 1015 | evt 1047 |
|---|---|---|---|
| WCT `wct-flash` | n flashes | 2128 | 1061 |
| WCT `wct-flash` | total PE | 3.47 M | 0.35 M |
| WCT `wct-flash` | −x PE | **3.15 M** | 0.15 M |

Because the anomaly is shared upstream of both deconvolutions, it lives in the raw
−x waveforms (the saturation of §4), not in any one reconstruction's settings.

## 6. The big PE does **not** by itself explain the big flash count

The flash-count inflation is a *second symptom of the same saturation event*, not a
consequence of the bright-burst PE:

- Only **158 of 1347** flashes lie inside the bright burst; the other **1189** are
  spread across the readout — already ≈3× a normal event's *entire* flash total (~400).
- The −x snippet *count* is only **+26 %** (1479 vs 1173) → the extra flashes are
  **not** from extra self-triggers.
- The excess flashes are overwhelmingly **−x-dominant** (939 vs 250 outside the burst;
  a normal event is +x-dominant, 294 vs 107) and elevated in **every** PE bin (≈3×).

The PE inflation and the flash-count inflation are therefore **two largely separate
effects**, confirmed by the fix below:

- **PE inflation** = over-integration of each clipped pulse into a few giant wide hits.
  Removing the saturated hits drops the event's total PE by ~76 % (§9), so the PE is
  almost entirely this artifact.
- **Flash inflation** = many *small* hits spread across the readout (post-saturation
  baseline recovery on the saturated channels plus a genuinely busier event). These are
  **not** the saturated/over-integrated hits: removing the latter leaves the flash count
  essentially unchanged (793 → 787, §9). The flash count is high because the event is
  genuinely active on the −x wall, not because of the clipping per se.

## 7. The fix — saturation detect + veto in the WCT light chain

A toggle-able, default-OFF saturation veto was added to the WCT light chain so saturated
snippets stop over-integrating. Default off → every existing config is bit-identical.

**`OpDecon`** (`flash/src/OpDecon.cxx`, knobs `detect_saturation`, `saturation_adc=16383`,
`saturation_min_samples=1`, `saturation_pad`): when enabled, it scans each input snippet's
**raw** ADC (the only place the un-pedestal-subtracted 0–16383 values are visible) and, for
every contiguous run of ≥ `saturation_min_samples` samples at/above `saturation_adc`, adds
`[tbin+run−pad, tbin+run+pad)` to a `"saturation"` `ChannelMaskMap` on the output frame.
Marking the *run* (padded), not the whole trace, lets a 343808-sample full-stream channel
lose only its clipped region, not the whole 5.5 ms.

**`OpHitFinder`** (`flash/src/OpHitFinder.cxx`, knob `veto_saturation`): when enabled, it
drops any reconstructed hit whose tick span overlaps a saturated range for that channel —
i.e. the broad over-integrated pulse around the clip — while real narrow light elsewhere on
the trace survives. `OpRoi` was patched to forward `in->masks()` so the full-stream branch
carries the flag through ROI cleaning.

**Config** (`cfg/.../pdhd/flash.jsonnet` builders; enabled in
`pdhd/wct-light-allpd-reco.jsonnet` on both branches): PDHD uses `saturation_pad=1024`
(≈ one snippet record / the over-integration window). The pad is the lever that determines
how much of the over-integration plateau around each clip is removed.

## 8. Validation (all-PD WCT chain, run 29107)

Reprocessed with the veto on (`saturation_pad=1024`) vs off, reusing the existing decoana
inputs. The veto flags ~72 saturated runs on the snippet branch and ~78 on the full-stream
branch in evt 1015; ~0–2 in a normal event.

| quantity (evt 1015) | OFF | ON (pad 1024) | change |
|---|---|---|---|
| total PE | 3.31 M | **0.81 M** | **−76 %** |
| −x full-stream PE (120–159) | 1.73 M | **0.15 M** | −91 % (→ ~normal 0.08 M) |
| −x snippet PE (80–119) | 1.19 M | 0.39 M | −67 % |
| +x PE (0–79) | 0.32 M | 0.23 M | −28 % |
| brightest flash at the burst | 1.33 M PE | **0.21 M PE** | de-inflated, **not deleted** |
| n flashes | 793 | 787 | ≈unchanged |
| n ophits | 58 548 | 58 057 | −491 (the giant hits) |

- The real bright flash **survives** at the burst time (built from the +x and unsaturated
  −x channels), just with a physically sane PE instead of the over-integrated value.
- **Normal events are untouched:** evts 1047/1031/999/1191 change by ≤0.5 % total PE and
  0 flashes (their handful of saturated samples carry negligible PE).
- The fix removes the **PE inflation** (the clear artifact). It does **not** reduce the
  flash count (§6): only ~491 of 58 548 hits are removed, so the flash multiplicity — which
  reflects genuine −x activity / post-saturation recovery — is essentially unchanged.

## 9. Why the charge looks normal

Saturation here is a **light-readout dynamic-range** effect — the photon-detector ADC clips
at 16383 — and is independent of the TPC charge ADC. A track passing close to the −x PD
plane dumps locally intense light (∝ 1/r²) that overflows the PD range, while the total
ionization (charge) it deposits stays ordinary and unsaturated. Hence huge light, normal
charge.

## 10. Per-PD "fired" threshold for the flash multiplicity cut (`min_fired_pe`)

The `min_fired_pds` quality cut counted a PD as "fired" if its PE ≥ `refine_fired_pe` = **0.5
PE** — sub-single-photoelectron (noise level). A new `OpFlashFinder` knob **`min_fired_pe`**
(member default −1 = use `refine_fired_pe` ⇒ bit-identical; PDHD sets **1.0**) raises that to
one detected photoelectron, so a PD must really fire to count. It is decoupled from the
refinement-merge logic, which still uses `refine_fired_pe`.

This was motivated by evt 1015's high flash count, but a scan showed it is a **general quality
improvement, not a 1015 fix**: the per-PD PE spectrum is a smoothly falling continuum (no
noise/signal valley) and raising the threshold tightens *normal* events more than 1015 (whose
extra flashes have a *higher* median 5th-PD PE, 3.2 vs 2.1 — i.e. genuine activity, §6). At
1.0 PE the effect is small: evt 1015 **787 → 768** flashes (−2 %), normal events ≈ −7 %
(1047 237→220, 1031 207→191, 999 240→226, 1191 223→209), matching the post-hoc scan exactly.

> Flash counts above are **after** all three cuts: `min_total_pe ≥ 20`, `min_fired_pds ≥ 5`,
> and now per-PD `min_fired_pe ≥ 1.0`.

## 11. Known residual / follow-up

- **Flash count** for this event stays ~3.5× a normal event after the fixes (768 vs ~210),
  driven by spread-out small hits (post-saturation baseline recovery on the saturated
  channels + a genuinely busier −x wall), which are not over-integration artifacts. Reducing
  it would need a recovery-window or baseline-aware veto that risks removing real light — left
  as follow-up.
- **Residual −x snippet PE** (0.39 M, ~7× normal) is the same post-saturation activity, not
  clipping; left in place for the same reason.
- A **mean-PE-per-fired-PD** cut (the nPD–totPE diagonal in the 2D distribution) would be a
  sharper signal/noise discriminator than the flat per-PD floor, if ever needed.

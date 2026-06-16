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

Mechanism: deconvolving 33 clipped/ringing flat-top snippets produces many fragmented,
over-threshold crossings spread in time across the readout → many spurious OpHits →
inflated flash multiplicity. So:

- **PE inflation** = over-integration of each clipped pulse (one giant hit/channel).
- **Flash inflation** = fragmentation/ringing of the same clipped pulses (many small hits).

These are two faces of the one saturation event; the bright-burst PE term alone accounts
for only ~12 % of the extra flashes.

## 7. Why the charge looks normal

Saturation here is a **light-readout dynamic-range** effect — the photon-detector ADC
clips at 16383 — and is independent of the TPC charge ADC. A track passing close to the
−x PD plane dumps locally intense light (∝ 1/r²) that overflows the PD range, while the
total ionization (charge) it deposits stays ordinary and unsaturated. Hence huge light,
normal charge.

## 8. Recommendation (follow-up, not done here)

This is real detector behaviour; nothing is miscomputed given the clipped input. To stop
saturated events from inflating PE/flash counts, the light chain could **detect clipped
snippets** (raw at/near 16383) and either flag them, cap/skip the over-integration, or
veto hits from saturated records. That is a separate change to the OpDecon/OpHitFinder
chain and is out of scope for this diagnosis.

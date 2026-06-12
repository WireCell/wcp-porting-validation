# Imaging activity threshold — PDVD and PDHD

What "the threshold" is in the 3-D imaging step, where its value comes from in
the earlier (signal-processing) chain, and whether PDVD and PDHD differ in
principle.

Short answer: the imaging threshold is `nthreshold × (per-channel noise RMS)`,
with the per-channel RMS *measured live, per channel and per event*, by signal
processing. PDVD and PDHD use the **identical mechanism and the identical
`nthreshold` value** — they are not different in principle. What differs is only
the absolute RMS each channel reports, which the rule self-adapts to.

> **Current default (both detectors): `nthreshold = [1e-6, 1e-6, 1e-6]`** — i.e.
> effectively **zero threshold (charge > 0)**. This was changed from the former
> `3.6σ` default on 2026-06-09. **Why `1e-6` and not literal `0`:** `MaskSlice`
> falls back to a *high* fixed bar when the threshold evaluates to exactly zero
> (`if (threshold == 0) threshold = m_default_threshold`, see §1), so a literal
> `0` would do the opposite of "charge > 0". `1e-6` is the charge>0 surrogate.
> The change affects every standalone PDVD/PDHD imaging run (it is **not**
> bit-identical to the old 3.6σ reco).

## 1. What the threshold is (in the imaging code)

Imaging turns SP waveforms into 3-D blobs. The first stage, **`MaskSlices`**
(`img/src/MaskSlice.cxx`), decides for every (channel, tick) whether there is
enough charge to count as *activity*. Only activity is tiled into blobs, so this
is the gate that defines what imaging "sees".

The per-channel threshold is computed at `MaskSlice.cxx:327`:

```cpp
double threshold = m_nthreshold[planeid.index()] * summary[idx];
if (threshold == 0) threshold = m_default_threshold[planeid.index()];
```

- `m_nthreshold` — the per-plane multiplier, set in the imaging config
  (`pdvd/img.jsonnet`, `pdhd/img.jsonnet`):

  ```jsonnet
  nthreshold: [1e-6, 1e-6, 1e-6],   // U, V, W — current default (charge>0)
  // nthreshold: [3.6, 3.6, 3.6],   // former default: 3.6 sigma
  ```

  It is **dimensionless** — a number of sigma. Same value in both detectors and
  for all three planes. The current `1e-6` is "as low as the cut can go without
  tripping the `threshold == 0` fallback below".

- `summary[idx]` — the **per-channel noise RMS** of the deconvolved waveform,
  carried on the `wiener` trace as its *trace summary* (`summary_tag: wiener`).
  This is the value supplied by the earlier SP chain (Section 2).

A (channel, tick) is **active** when, roughly, the Wiener-deconvolved charge
exceeds `threshold` (`MaskSliceBase::thresholding`, `MaskSlice.cxx:173`):

```cpp
if (q_wiener > threshold) return true;            // primary cut: nthreshold·RMS
// else: neighbour-slice rescue using the gauss waveform
if ((q_gauss > q_next/3. && q_next > threshold) ||
    (q_gauss > q_prev/3. && q_prev > threshold)) is_active = true;
```

So the operative cut is **`nthreshold` × per-channel RMS** on the Wiener charge
(currently `1e-6 × RMS`, i.e. charge > 0), with a softer neighbour-slice rescue
so that a tick adjacent to a strong slice is not dropped on a fluctuation.

`m_default_threshold` (`= {587.8, 836.6, 567.97} × 4 ≈ {2351, 3346, 2272}`) is a
**fallback used only when the product `nthreshold × summary` is exactly 0** —
in practice when `summary == 0` (a dead/empty channel that reports no RMS).
This is exactly why the new default is `1e-6` and **not** `0`: with `1e-6`, a
live channel gives `1e-6 × RMS > 0` so the fallback is skipped and the cut is
the intended charge>0; a literal `0` would force `0 × RMS = 0` on *every*
channel and substitute this high bar instead. Neither `pdvd/img.jsonnet` nor
`pdhd/img.jsonnet` overrides `m_default_threshold`, so both inherit this header
default. Note it is **MicroBooNE-inherited**, i.e. *not* derived from PD data —
it applies only to zero-product channels, never to the live charge>0 path.

## 2. How the threshold value is obtained in the earlier (SP) chain

The `summary[idx]` that imaging multiplies by 3.6 is **not a tuned constant** —
it is computed from the data itself during `OmnibusSigProc`.

1. **Per-wire RMS of the deconvolved waveform** — `ROI_formation::cal_RMS`
   (`sigproc/src/ROI_formation.cxx:364`) estimates each wire's noise robustly:

   ```cpp
   par[0] = percentile(signal, 0.5 - 0.34);   // ~ -1σ  (16th pct)
   par[1] = percentile(signal, 0.5);          //   median
   par[2] = percentile(signal, 0.5 + 0.34);   // ~ +1σ  (84th pct)
   float rms = sqrt(((par[2]-par[1])^2 + (par[1]-par[0])^2)/2);  // quantile σ
   // refine: ordinary RMS over samples within |x| < 5·rms (reject signal tails)
   ```

   This is a **quantile-based, signal-robust** noise estimate: the 16/50/84
   percentile spread gives a first σ, and the final RMS is the plain RMS over
   the samples inside ±5σ (so real pulses don't inflate the noise estimate).

2. **Stored as the `wiener` trace summary** — `OmnibusSigProc::save_data`
   (`sigproc/src/OmnibusSigProc.cxx:530, 548/560`) pushes that per-wire RMS into
   the trace summary of every saved `wiener` trace:

   ```cpp
   const float thresh = perwire_rmses[och.wire];   // = uplane_rms / vplane_rms / wplane_rms
   ...
   threshold.push_back(thresh);
   ```

   (`perwire_rmses` = `roi_form.get_{u,v,w}plane_rms()`, `OmnibusSigProc.cxx:1865`.)

3. **Written into the per-anode SP archive** alongside the traces, so the
   downstream imaging job reads it back. This is confirmed in the imaging log:

   ```
   <FrameFileSource> read type=summary tag=wiener7 ident=339910 ext=npy
   <MaskSlice:slicing-anode7-ms-active_0> nslices=2000 from ntraces=1536 ...
   ```

   The `wiener` tag carries a populated `summary`, and slices are produced (no
   `default_threshold` fallback, no size-mismatch) — i.e. the live 3.6σ path is
   the one in use.

**Net:** the imaging threshold is *self-calibrating*. Each channel contributes
its own measured noise level; imaging asks for charge ≥ 3.6× that level. A noisy
channel automatically gets a higher absolute bar; a quiet channel a lower one.

### Aside — a *different* threshold inside SP (don't confuse the two)

SP also has its own internal **ROI-forming** threshold, the *tight* ROI cut
`th_factor × rms + 1` (`ROI_formation.cxx:448-457`), governed by
`troi_ind_th_factor` / `troi_col_th_factor`. This decides where ROIs are drawn
*inside* SP; it is a separate cut from the imaging activity cut. It happens to
share the same per-wire `rms`, but a different multiplier and purpose. Values:

| | `th_factor_ind` (U,V) | `th_factor_col` (W) |
|---|---|---|
| PDVD (`protodunevd/sp.jsonnet`) | 3.0 (explicit) | 5.0 (explicit) |
| PDHD (`protodunehd/sp.jsonnet`) | 3 (code default) | 5 (code default) |

Both detectors are effectively ind=3 / col=5. Lead with `nthreshold` for the
*imaging* question; these `th_factor` numbers belong to SP, not imaging, and are
**unaffected** by the imaging `nthreshold` change.

## 3. Are PDVD and PDHD different in principle?

**No.** The principle, the code, and the value are the same:

| | PDVD | PDHD |
|---|---|---|
| imaging cut | `nthreshold × RMS` (Wiener charge) | `nthreshold × RMS` (Wiener charge) |
| `nthreshold` (U,V,W) | `[1e-6, 1e-6, 1e-6]` (was `[3.6,3.6,3.6]`) | `[1e-6, 1e-6, 1e-6]` (was `[3.6,3.6,3.6]`) |
| RMS source | `wiener` trace summary from SP | `wiener` trace summary from SP |
| RMS definition | `ROI_formation::cal_RMS` (quantile σ) | same code |
| `default_threshold` fallback | header default (zero-product chans only) | header default |
| neighbour-slice rescue | yes | yes |

The same `MaskSlice.cxx` / `ROI_formation.cxx` code serves both; the only config
knob (`nthreshold`) is set to the same value in each (now `1e-6`).

What **does** differ between the two detectors is the *absolute* per-channel RMS
— different electronics, noise spectra, and field responses give each detector
(and each channel) its own noise level. But because the threshold is built as
`nthreshold × (that channel's measured RMS)` and the RMS is measured live per
event, the rule **self-adapts**: the significance cut is identical even though
the absolute charge bar is not. That is the clean answer — same principle,
detector differences absorbed by the per-channel RMS.

## 4. Effect of the zero-threshold default (empirical)

A 0σ / 1σ / 2σ comparison on one event per detector (PDVD run 039324 evt 0,
PDHD run 027409 evt 0) showed that **`nthreshold = 1e-6` (charge>0) does *not*
flood the image**: active-blob output grows only ~25–40 % versus 2σ, and
imaging+clustering stay at the ~minute scale. The reason is that the active
tiling requires multi-plane coincidence (3-view + 2-of-3), so isolated
single-plane noise excursions rarely form blobs even when charge > 0. Output
blob volume is cleanly monotonic 0σ > 1σ > 2σ on every anode/APA.

Bee links from that comparison (0σ / 1σ / 2σ):

- PDVD 039324 evt0: [0σ](https://www.phy.bnl.gov/twister/bee/set/2b6cac28-0557-4851-bb75-f1b50a2a63b7/event/list/) ·
  [1σ](https://www.phy.bnl.gov/twister/bee/set/07fc4c80-5f0a-4e15-926e-83eef52eab2e/event/list/) ·
  [2σ](https://www.phy.bnl.gov/twister/bee/set/7f793bd3-e563-4505-a4e6-30d208b4c9a8/event/list/)
- PDHD 027409 evt0: [0σ](https://www.phy.bnl.gov/twister/bee/set/7c7fa285-5f1f-4d81-8185-b94502c7d44b/event/list/) ·
  [1σ](https://www.phy.bnl.gov/twister/bee/set/42f84510-430f-492e-8a89-b3ae91eed3e1/event/list/) ·
  [2σ](https://www.phy.bnl.gov/twister/bee/set/21916fda-fbe8-4f1a-9a4e-ac8ccd4e20d9/event/list/)

## Files referenced

- `img/src/MaskSlice.cxx` — `:327` threshold formula, `:173` `thresholding()`,
  `:328` `default_threshold` fallback
- `img/inc/WireCellImg/MaskSlice.h` — `:77` `m_nthreshold{3.6,3.6,3.6}`,
  `:79` `m_default_threshold`
- `pdvd/img.jsonnet`, `pdhd/img.jsonnet` — `slicing()` (`nthreshold`, `summary_tag`)
- `sigproc/src/ROI_formation.cxx` — `:364` `cal_RMS`, `:445-457` per-plane RMS fill
- `sigproc/src/OmnibusSigProc.cxx` — `:530/548/560` summary push, `:1865` per-plane RMS
- `cfg/pgrapher/experiment/protodune{vd,hd}/sp.jsonnet` — `troi_*_th_factor` (SP-internal ROI cut, distinct)

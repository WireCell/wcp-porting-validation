# Imaging activity threshold — PDVD and PDHD

What "the threshold" is in the 3-D imaging step, where its value comes from in
the earlier (signal-processing) chain, and whether PDVD and PDHD differ in
principle.

Short answer: **the imaging threshold is `3.6 × (per-channel noise RMS)`**, with
the per-channel RMS *measured live, per channel and per event*, by signal
processing. PDVD and PDHD use the **identical mechanism and the identical 3.6σ
value** — they are not different in principle. What differs is only the absolute
RMS each channel reports, which the rule self-adapts to.

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
  nthreshold: [3.6, 3.6, 3.6],   // U, V, W
  ```

  It is **dimensionless** — a number of sigma. Same value in both detectors and
  for all three planes.

- `summary[idx]` — the **per-channel noise RMS** of the deconvolved waveform,
  carried on the `wiener` trace as its *trace summary* (`summary_tag: wiener`).
  This is the value supplied by the earlier SP chain (Section 2).

A (channel, tick) is **active** when, roughly, the Wiener-deconvolved charge
exceeds `threshold` (`MaskSliceBase::thresholding`, `MaskSlice.cxx:173`):

```cpp
if (q_wiener > threshold) return true;            // primary cut: 3.6σ
// else: neighbour-slice rescue using the gauss waveform
if ((q_gauss > q_next/3. && q_next > threshold) ||
    (q_gauss > q_prev/3. && q_prev > threshold)) is_active = true;
```

So the operative cut is **3.6 × per-channel RMS** on the Wiener charge, with a
softer neighbour-slice rescue so that a tick adjacent to a strong slice is not
dropped on a fluctuation.

`m_default_threshold` (`= {587.8, 836.6, 567.97} × 4 ≈ {2351, 3346, 2272}`) is a
**fallback used only when `summary == 0`** (e.g. a dead/empty channel that
reports no RMS). Neither `pdvd/img.jsonnet` nor `pdhd/img.jsonnet` overrides it,
so both inherit this header default. Note it is **MicroBooNE-inherited**, i.e.
*not* derived from PD data — it applies only to zero-RMS channels, never to the
live 3.6σ path.

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

Both detectors are effectively ind=3 / col=5. Lead with `nthreshold=3.6` for the
*imaging* question; these `th_factor` numbers belong to SP, not imaging.

## 3. Are PDVD and PDHD different in principle?

**No.** The principle, the code, and the value are the same:

| | PDVD | PDHD |
|---|---|---|
| imaging cut | `3.6 × RMS` (Wiener charge) | `3.6 × RMS` (Wiener charge) |
| `nthreshold` (U,V,W) | `[3.6, 3.6, 3.6]` | `[3.6, 3.6, 3.6]` |
| RMS source | `wiener` trace summary from SP | `wiener` trace summary from SP |
| RMS definition | `ROI_formation::cal_RMS` (quantile σ) | same code |
| `default_threshold` fallback | header default (zero-RMS chans only) | header default |
| neighbour-slice rescue | yes | yes |

The same `MaskSlice.cxx` / `ROI_formation.cxx` code serves both; the only config
knob (`nthreshold`) is set to 3.6σ in each.

What **does** differ between the two detectors is the *absolute* per-channel RMS
— different electronics, noise spectra, and field responses give each detector
(and each channel) its own noise level. But because the threshold is built as
`3.6 × (that channel's measured RMS)` and the RMS is measured live per event,
the rule **self-adapts**: the 3.6σ significance cut is identical even though the
absolute charge bar is not. That is the clean answer — same principle, detector
differences absorbed by the per-channel RMS.

## Files referenced

- `img/src/MaskSlice.cxx` — `:327` threshold formula, `:173` `thresholding()`,
  `:328` `default_threshold` fallback
- `img/inc/WireCellImg/MaskSlice.h` — `:77` `m_nthreshold{3.6,3.6,3.6}`,
  `:79` `m_default_threshold`
- `pdvd/img.jsonnet`, `pdhd/img.jsonnet` — `slicing()` (`nthreshold`, `summary_tag`)
- `sigproc/src/ROI_formation.cxx` — `:364` `cal_RMS`, `:445-457` per-plane RMS fill
- `sigproc/src/OmnibusSigProc.cxx` — `:530/548/560` summary push, `:1865` per-plane RMS
- `cfg/pgrapher/experiment/protodune{vd,hd}/sp.jsonnet` — `troi_*_th_factor` (SP-internal ROI cut, distinct)

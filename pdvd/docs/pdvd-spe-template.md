# PDVD SPE templates — determination, calibration and deconvolution demonstration

Milestone 1 of the light-reconstruction plan (`light-reco-plan.md` §4): determine the
single-photo-electron (SPE) response template for **every available photon detector**, and
demonstrate it by deconvolution — the acceptance criterion is a **flat deconvolved baseline**.
All results are from run 039252 (18 events), Python studies with the validated NumPy port of
`flash/src/OpDecon.cxx` (fixed Wiener filter; PDHD settings `fixed_snr=0.005`, 1.5 MHz Gauss
post-filter as the starting point — the dedicated filter scan is milestone 2).

> Scripts: `pdvd/pd_plot/pdvd_light.py` (loader + decon port), `spe_build.py` (pulse harvest +
> template averaging + cathode tail repair), `spe_compare.py` (try-and-compare + per-PD
> figures), `spe_longtail.py` (cathode bright-pulse long-window medians).
> Figures: `pdvd/docs/pds/` — `pd_ch<CCCC>.png` (one per DAPHNE channel: template |
> amplitude spectrum | raw and deconvolved small/medium/large example pulses),
> `spe_amplitude_spectra.png`, `spe_templates_by_population.png`, `spe_template_compare.png`,
> `spe_cathode_longtail.png`, `spe_summary.json` (per-channel numbers).
> **Which `pd_ch*.png` is which physical PD: §7.**

---

## TL;DR

- **Data-driven templates win, per channel.** Out-of-sample decon-tail residual (build the
  template on half the pulses, deconvolve the held-out half): **0.4–1.2 %** for the
  data-driven templates vs **8–31 %** for the PDHD `SPE_NP04_{FBK,HPK}_2024` templates and
  **3.4–5.0 %** for the analytic `SPE_NP02_estimate`. The PDVD pulse shapes are simply
  different from PDHD (much slower decay, no deep undershoot) — the PDHD SiPM templates are
  ruled out as-is.
- **Cathode template tails are repaired for a harvest baseline bias** (§2 step 4): the raw
  1-PE averages carried a spurious near-constant −2…−4 ADC tail; bright-pulse medians show the
  true response is a positive slow component (τ ≈ 0.6–1.2 µs) with **no AC undershoot** down
  to 5×10⁻⁴ of the prompt out to 50 µs. Without the repair, bright cathode pulses deconvolve
  to a non-decaying ~12 % plateau instead of returning to baseline.
- **All 51 live DAPHNE channels are demonstrated** (`pd_ch*.png`): 42 with their own
  template, 9 low-statistics channels (8 cathode + ch2010) with the population-average shape
  scaled to the channel 1-PE amplitude.
- **PE normalization anchors**: membrane XA have textbook 1-PE amplitude peaks (36–64 ADC,
  channel-dependent); **PMTs show a clear SPE bump too** (92–160 ADC — better than initially
  assumed); the **cathode XA PE scale is provisional** (spectrum is threshold-limited, no PE
  structure — see §5).
- PE closure: deconvolving a held-out average 1-PE pulse with the channel template gives
  integrated area ≈ 1 PE (0.8–1.2 across the tested channels) — the AutoScale +
  `area/spe_area` convention will carry through unchanged.

## 1. Populations and pulse shapes (measured)

| population | channels | readout | pulse shape | noise RMS (ADC) |
|---|---|---|---|---|
| cathode XA (10xx, OpDet 4–11) | 16 | full-stream 468800/468864 | fast peak (FWHM ~12 ticks ≈ 0.2 µs) + positive slow tail (τ ≈ 0.6–1.2 µs); **no measurable AC undershoot** (< 5×10⁻⁴ of prompt out to 50 µs) | 2.1–3.2 (quiet); two noisy pairs (§6) |
| membrane XA (20xx) | 15 (ch2031 absent) | self-trigger 1024 | slow unipolar decay, FWHM 22–54 ticks, tiny undershoot | bottom ~2–3, **top 5–10** |
| PMT (30xx) | 20 live | self-trigger 1024 | very fast (FWHM 2–3 ticks, undersampled at 16 ns) + small afterpulse bump | 1.4–2.1 |

Notes:
- All waveforms **positive-going** in this extraction. Pedestals: XA ~1700–3600, PMT ~7500–9300.
- The full-stream records carry **low-frequency baseline wander** (plain MAD 27–90 ADC) while
  the true sample-to-sample noise is only ~2 ADC (adjacent-difference estimator). Pulse
  finding/isolation on the cathode therefore runs on a high-passed trace (OpRoi-style
  `1−exp(−(f/0.05 MHz)²)`), while templates are extracted from the **raw** trace.
- The PDVD membrane XA pulse decays several times slower than the PDHD NP04 templates and has
  no deep AC undershoot; the cathode (full-stream) XA shows **no PDHD-style AC undershoot at
  all** down to the 5×10⁻⁴ level (bright-pulse medians out to 50 µs,
  `spe_cathode_longtail.png`) — on reconstruction timescales it behaves DC-coupled.

## 2. Template construction

Per DAPHNE channel (`spe_build.py`):

1. **Harvest**: membrane XA + PMT from the 1024-sample self-trigger snippets; cathode XA from
   isolated small pulses found in the 7.5 ms full stream (HPF trace for finding/isolation,
   raw trace + local pre-peak pedestal for extraction).
2. **1-PE selection**: peak-amplitude spectrum mode = 1-PE estimate; accept
   0.75–1.30 × mode, unsaturated, flat pre-peak baseline, no second comparable pulse
   (prominence test on a boxcar-smoothed trace — plain height tests miscount noise wiggles on
   the broad XA top) in the −0.96…+6.4 µs template window.
3. **Average**: align on the peak sample; mean in the peak region, per-sample **median**
   beyond +1.6 µs (robust to residual pileup); split-half (odd/even pulses) copies kept for
   out-of-sample validation.
4. **Cathode tail repair** (`repair_fs_tail`; cathode channels only). The raw 1-PE averages
   carry a spurious near-constant negative tail (−2…−4 ADC ≈ −10…−15 % of the 1-PE peak *per
   sample* over 6 µs): the local pre-peak pedestals of the harvest windows sit on slowly
   falling tails of earlier activity in the busy stream, biasing the whole window low. Used
   as a decon kernel this corrupts the low frequencies — **bright pulses deconvolve to a
   non-decaying positive plateau** (~12 % of the prompt spike per tick; this is what the
   original `pd_ch10xx.png` showed). The truth is measured from bright pulses, where the tail
   is far above noise: median amplitude-normalized shape of isolated ≥30 PE pulses over 51 µs
   windows (n ≈ 700–1100 per channel, `spe_cathode_longtail.png` / `spe_longtail.py`) shows a
   **positive** slow component (τ ≈ 0.6–1.2 µs — late light plus slow XA response) decaying
   to zero by ~25 µs and **no AC undershoot above 5×10⁻⁴** of the prompt. Repair: fit
   `a·e^(−t/τ) + c` to the template beyond +0.32 µs and keep only the exponential (`c` = the
   baseline bias, dropped). Validation on ch1010: decon tail/peak of ≥8 PE pulses drops
   **0.115 → 0.020**, at/below the membrane late-light benchmark (0.026–0.075) and flat vs
   amplitude (8 → >150 PE), i.e. the remaining tail is real late light, as on the membrane.

The stored template is the **mean 1-PE response in raw ADC** — its absolute scale IS the PE
calibration (§4).

## 3. Try-and-compare result

Metric: deconvolve the **held-out** average 1-PE pulse with each candidate; measure the
post-pulse residual 0.5–4 µs after the decon spike, relative to the candidate's own
**matched kernel** (template deconvolved with itself — the intrinsic tail a perfect template
gives; PDHD convention, `pdhd-spe-template-tuning.md`). Mean |residual| in % of the decon peak:

| candidate | cathode XA | membrane XA | PMT |
|---|---|---|---|
| **data-driven (own channel)** | **1.0** | **1.2** | **0.4** |
| data-driven (population avg) | 2.1 | 2.2 | 5.1 |
| SPE_NP04_FBK_2024 (PDHD) | 7.8 | 17.7 | – |
| SPE_NP04_HPK_2024 (PDHD) | 31.1 | 24.6 | – |
| SPE_NP02_estimate (larsoft VD) | 5.0 | 3.4 | – |

(`spe_template_compare.png` shows the per-channel bars. Cathode numbers are after the tail
repair of §2 step 4 — kernels *and* held-out probes; before the repair, own-channel read
2.8 % and the biased probe inflated all candidates.)

**Decision:** per-channel data-driven templates for all populations; population-average shape
(scaled to the channel 1-PE mode) for the 9 channels without enough clean 1-PE pulses. After
the tail repair the own-channel cathode templates beat the population average (1.0 vs 2.1 %) —
the final chain may still use one template per population for the XAs, mirroring the PDHD
per-TYPE lesson; to be frozen after the filter scan (milestone 2) and a cross-run stability
check.

The 6-sample ProtoDUNE-DP PMT waveform (`protoDUNEDP_waveform_20180927.txt`, area exactly 410
= the larsoft `SPEArea`) is unusable as a 16 ns-sampled shape; it remains only a historical
area anchor. Not needed: the PDVD PMT SPE peak is measured directly (§4).

## 4. Normalization (the PE scale)

How the chain turns ADC into PE (verified against `OpDecon.cxx` + `OpHitFinder.cxx`):

1. `OpDecon` loads the template **as-is in ADC** and never normalizes it.
2. **AutoScale** applies the Wiener+post filter to the template itself, integrates the
   positive lobe around its peak and divides the decon output by that area → a pulse that
   matches the template deconvolves to **unit area** (1 PE).
3. `OpHitFinder` integrates the decon pulse and divides by `spe_area=100` after a `scale=100`
   pre-scale → OpHit `pe` = decon area.

Therefore the template **must be stored as the mean single-PE response in raw ADC**: its
amplitude is the gain calibration. Closure was verified per channel (held-out 1-PE average →
decon area 0.8–1.2 PE).

Two conventions adopted (both matter for the C++ chain):

- **Pre-trigger trim**: templates are used as decon kernels with the peak at sample ~10 (the
  NP04 `without_pretrigger` convention). A kernel with N pre-peak samples shifts every decon
  spike (and OpHit time) N ticks early — this was reproduced explicitly with the 60-tick
  harvest window.
- **1-PE anchor per population**: membrane XA = clean amplitude-spectrum peak (36–64 ADC);
  PMT = SPE bump (92–160 ADC; ch3010/OpDet14 has no usable gain); cathode XA = **provisional**
  (§5).

## 5. Cathode XA: the open PE-scale question

**Shape is measured; gain is not.** The full stream gives ample statistics for the pulse
*shape* (2.8 % held-out residual), but the PE *scale* is a **gain** number — ADC per
photo-electron — and measuring it data-drivenly needs a **resolved single-PE feature**: a
1-PE bump separated from the noise below and the 2-PE bump above. More data does not create
that feature; the detector either resolves it or it does not. On the cathode it is buried, so
the templates have the **shape** right but an **uncertain absolute scale** on these 16
channels. (This is why the full stream, despite its length, does not settle the scale — a
common point of confusion.)

Why it is buried (the amplitude spectrum is a smooth threshold-limited continuum; apparent
mode ~16–28 ADC is a turn-on artifact, not a peak):

- **Pile-up** — the cathode sits between the two drift volumes and sees light from **both**
  continuously, so overlapping sub-PE pulses fill in the valley that would define a 1-PE peak;
- **Threshold turn-on** — the pulse finder seeds at 5σ ≈ 10 ADC over the baseline wander; if
  the true 1-PE amplitude is at or below that, we only ever harvest pulses above threshold, so
  the spectrum climbs to the seed rather than showing a peak.

### Can we just use the membrane XA scale?

It is the same **sensor** (X-ARAPUCA / SiPM), so borrowing the membrane 1-PE scale is a
reasonable **interim default** — but *not* a free transfer, because the cathode PDs use a
**different signal path**. They sit on the HV cathode and (in the VD design) are powered and
read out over fiber (power-/signal-over-fiber) rather than the directly-cabled membrane
modules — a different front-end gain. The measured pulse shape is the fingerprint of this:
the cathode peak is far **faster** (FWHM ≈ 0.2 µs) than the membrane's slow unipolar decay
(0.35–0.9 µs). Same photons, different electronics ⇒ ADC-per-PE almost certainly differs
(this is also the likely origin of the "lower effective gain").

Consequences if you do transfer:

- Transfer the 1-PE **area, not the amplitude** — the two shapes distribute a given charge
  very differently in time (narrow cathode peak vs broad membrane pulse), so matching peak
  heights would be wrong.
- Even area transfer is only approximate: the shapes split their charge differently between
  the fast lobe and the slow tail, so equal *total* charge ≠ equal charge inside the AutoScale
  integration window. And it is valid only if the cathode SoF/DAPHNE gain and OV are confirmed
  equal to the membrane's.

### Anchors, in order of effort

1. **Decon-area spectrum** of isolated stream pulses (cleanest data-driven shot, not yet
   exhausted): the Wiener decon is a near-matched filter, so PE quantization may emerge in
   *area* where amplitude fails.
2. **Lower the finder threshold** (e.g. 3σ) on the HPF'd cathode trace and check whether a
   1-PE bump is hiding just under the current 5σ seed — cheap, but risks noise-triggered fakes.
3. **Area gain-transfer from the membrane XAs** — the interim number above, flagged
   ±(SoF gain unknown), valid only if the hardware settings match.
4. **Ask the PDS group** for the cathode SoF/DAPHNE gain settings — the authoritative answer.

(An earlier caveat here claimed the AC recovery extends past the 6.4 µs template window. That
negative tail was the **harvest baseline bias**, now measured and repaired — §2 step 4. The
bright-pulse medians show no undershoot at all down to 5×10⁻⁴ of the prompt out to 50 µs, so
on reconstruction timescales the cathode behaves DC-coupled and AC-recovery truncation drops
out of the transfer question entirely.)

## 6. Channel flags

| channel | flag |
|---|---|
| ch2031 (OpDet 1, 2nd ch) | absent from the data entirely |
| ch2010 (OpDet 0) | no clean 1-PE peak, 0 selected pulses → fallback template; noisy (RMS 8.9) |
| ch2020 (OpDet 2) | **nonlinear undershoot — no linear kernel fix.** The decon shows the same positive-plateau symptom as the (repaired) cathode channels, but the cause differs: the undershoot is real electronics AND its relative depth shrinks with amplitude (−0.24…−0.30 of peak at 1 PE from n=192 dark counts, −0.18 at 2–5 PE, −0.14 at 7.5 PE, ≈−0.05 at 5–10 PE), so any single kernel is wrong at some amplitude — kernels matching the 1-PE response leave a ~0.2·peak plateau on 3–8 PE pulses; shallow-tail kernels only "win" the tail metric by absorbing mean late light. Recommend **masking this channel in the C++ chain** (top membrane is noisy anyway); if kept, PE from it carries an amplitude-dependent bias |
| membrane-top group (20xx, x>0) | systematically noisier (RMS 5–10 vs 2–3 bottom) |
| ch1050/1051 (OpDet 7), ch1070/1071 (OpDet 9) | noisy cathode pairs (RMS 3.2/12.6, 3.2/2.1); ch1051 1-PE mode (82) is a threshold artifact |
| ch3010 (OpDet 14) | near-dead PMT (matches hand-over note); "template" is garbage, exclude |
| ch3020 (OpDet 15) | anomalous ~30 % slow tail in the template + large noise-trigger population — needs a dedicated re-selection |
| OpDet 24/27/28/34 | dead PMTs, absent from the data |

## 7. Channel ↔ physical PD map (which `pd_ch*.png` is which)

Each `pd_ch<CCCC>.png` is one DAPHNE channel `CCCC` (= the `opchannel` branch). Coordinates
are the PD-centre `(x, y, z)` in cm from the data file. In this frame **x is the drift
coordinate**: the cathode plane is at `x ≈ 0`, the **top** drift volume is `x > 0` and the
**bottom** drift volume is `x < 0`; the two long cryostat **membrane walls** are at
`y = ±417.6`, and the **bottom PMT array** sits below the bottom volume at `x ≤ −206`.
Cathode and most membrane OpDets read out through **two** DAPHNE channels (two supercells);
PMTs use one. (Mapping v09162025, cross-checked against the `x/y/z` branches of run 039252.)

### Cathode XA — on the cathode plane (`x ≈ 0`), 8 OpDets × 2 ch, full-stream

Tile a 4(z) × 2(y) grid across the cathode:

| OpDet | ch A (`.png`) | ch B (`.png`) | y (cm) | z (cm) | note |
|---|---|---|---|---|---|
| 4  | 1020 | 1021 | +123.8 | +258.5 | |
| 5  | 1060 | 1061 | −213.2 | +258.5 | |
| 6  | 1010 | 1011 | +290.4 | +187.3 | |
| 7  | 1050 | 1051 | −46.6  | +187.3 | noisy pair; ch1051 mode is a threshold artifact |
| 8  | 1030 | 1031 | +42.6  | +112.0 | |
| 9  | 1070 | 1071 | −213.2 | +112.0 | noisy pair |
| 10 | 1040 | 1041 | +209.1 | +40.8  | |
| 11 | 1080 | 1081 | −127.9 | +40.8  | |

(All 16 cathode channels use the population-average shape scaled to the channel; the PE
scale is provisional — §5.)

### Membrane XA — on the ±y cryostat walls (`|y| = 417.6`, `z = 149.7`)

Top volume (`x > 0`) and bottom volume (`x < 0`), each wall carrying two OpDets stacked along x:

| volume | wall | OpDet | ch A (`.png`) | ch B (`.png`) | x (cm) | note |
|---|---|---|---|---|---|---|
| top    | +y | 0  | 2010 | 2011 | +305.6 | ch2010 no clean 1-PE → fallback template; noisy |
| top    | +y | 2  | 2020 | 2021 | +229.0 | ch2020 distorted template (flagged) |
| top    | −y | 1  | 2030 | —    | +305.6 | ch2031 **absent** from the data |
| top    | −y | 3  | 2040 | 2041 | +229.0 | |
| bottom | +y | 12 | 2050 | 2051 | −201.1 | |
| bottom | +y | 18 | 2060 | 2061 | −277.7 | |
| bottom | −y | 13 | 2070 | 2071 | −201.1 | |
| bottom | −y | 19 | 2080 | 2081 | −277.7 | |

(The top-wall membrane — the `x > 0` rows — are the systematically noisier ones, RMS 5–10 vs
2–3 on the bottom wall.)

### PMT — bottom PMT array, one ch each, in three x-planes

| OpDet | ch (`.png`) | x (cm) | y (cm) | z (cm) | note |
|---|---|---|---|---|---|
| 14 | 3010 | −205.9 | +221.0 | +409.0 | near-dead (exclude) |
| 15 | 3020 | −205.9 | −221.0 | +409.0 | anomalous slow tail |
| 16 | 3030 | −205.9 | +256.0 | −96.1  | |
| 17 | 3040 | −205.9 | −221.0 | −109.7 | |
| 20 | 3050 | −281.7 | +221.0 | +409.0 | |
| 21 | 3060 | −281.7 | −221.0 | +409.0 | |
| 22 | 3070 | −281.7 | +256.0 | −96.1  | |
| 23 | 3080 | −281.7 | −221.0 | −109.7 | |
| 25 | 3100 | −336.5 | +0.0   | +455.6 | |
| 26 | 3110 | −336.5 | −170.0 | +455.6 | |
| 29 | 3140 | −336.5 | −170.0 | +353.6 | |
| 30 | 3150 | −336.5 | +405.3 | +217.8 | |
| 31 | 3160 | −336.5 | −405.3 | +217.8 | |
| 32 | 3170 | −336.5 | +405.3 | +149.7 | |
| 33 | 3180 | −336.5 | −405.3 | +149.7 | |
| 35 | 3200 | −336.5 | +0.0   | −54.4  | |
| 36 | 3210 | −336.5 | −170.0 | −54.4  | |
| 37 | 3220 | −336.5 | +170.0 | −156.3 | |
| 38 | 3230 | −336.5 | +0.0   | −156.3 | |
| 39 | 3240 | −336.5 | −170.0 | −156.3 | |

(Dead PMTs OpDet 24/27/28/34 are absent from the data — no `.png`. PMTs occupy three
x-planes: −205.9 and −281.7 carry OpDet 14–17 and 20–23; the deepest floor plane −336.5
carries the rest.)

## 8. Reproduce

```bash
cd pdvd
python3 pd_plot/spe_build.py 0          # harvest run 039252 (+ cathode tail repair) -> work/light_spe/harvest_r039252.npz
python3 pd_plot/spe_compare.py 39252    # comparison + all docs/pds figures + spe_summary.json
python3 pd_plot/spe_longtail.py 0       # cathode bright-pulse long-window medians (repair evidence)
```

## 9. Open items

- Cathode PE-scale anchor (§5). (The former "AC-recovery tail beyond 6.4 µs" item is
  resolved: it was the harvest baseline bias, repaired in §2 step 4 — no undershoot exists.)
- ch3020 template re-selection; ch2010 needs a quieter run or longer statistics. (ch2020
  re-selection was attempted and is a dead end — amplitude-nonlinear undershoot, §6; decide
  mask-vs-keep at chain assembly.)
- Cross-run stability: repeat on 039253 / 039349 before freezing `pdvd-spe-templates.json`.
- SiPM vendor (FBK vs HPK) per module still unknown — now moot for the template choice
  (data-driven), but useful metadata.
- Filter parameters (`fixed_snr`, post-filter cutoff, per-population noise RMS) — milestone 2.

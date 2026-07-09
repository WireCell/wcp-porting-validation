# PDVD SPE templates — determination, calibration and deconvolution demonstration

Milestone 1 of the light-reconstruction plan (`light-reco-plan.md` §4): determine the
single-photo-electron (SPE) response template for **every available photon detector**, and
demonstrate it by deconvolution — the acceptance criterion is a **flat deconvolved baseline**.
All results are from run 039252 (18 events), Python studies with the validated NumPy port of
`flash/src/OpDecon.cxx` (fixed Wiener filter; PDHD settings `fixed_snr=0.005`, 1.5 MHz Gauss
post-filter as the starting point — the dedicated filter scan is milestone 2).

> Scripts: `pdvd/pd_plot/pdvd_light.py` (loader + decon port), `spe_build.py` (pulse harvest +
> template averaging), `spe_compare.py` (try-and-compare + per-PD figures).
> Figures: `pdvd/docs/pds/` — `pd_ch<CCCC>.png` (one per DAPHNE channel: template |
> amplitude spectrum | raw and deconvolved small/medium/large example pulses),
> `spe_amplitude_spectra.png`, `spe_templates_by_population.png`, `spe_template_compare.png`,
> `spe_summary.json` (per-channel numbers). **Which `pd_ch*.png` is which physical PD: §7.**

---

## TL;DR

- **Data-driven templates win, per channel.** Out-of-sample decon-tail residual (build the
  template on half the pulses, deconvolve the held-out half): **0.4–2.8 %** for the
  data-driven templates vs **18–41 %** for the PDHD `SPE_NP04_{FBK,HPK}_2024` templates and
  **3.4–7.7 %** for the analytic `SPE_NP02_estimate`. The PDVD pulse shapes are simply
  different from PDHD (much slower decay, no deep undershoot) — the PDHD SiPM templates are
  ruled out as-is.
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
| cathode XA (10xx, OpDet 4–11) | 16 | full-stream 468800/468864 | fast peak (FWHM ~12 ticks ≈ 0.2 µs) + slow **AC-coupling recovery** (negative tail, > 6 µs) | 2.1–3.2 (quiet); two noisy pairs (§6) |
| membrane XA (20xx) | 15 (ch2031 absent) | self-trigger 1024 | slow unipolar decay, FWHM 22–54 ticks, tiny undershoot | bottom ~2–3, **top 5–10** |
| PMT (30xx) | 20 live | self-trigger 1024 | very fast (FWHM 2–3 ticks, undersampled at 16 ns) + small afterpulse bump | 1.4–2.1 |

Notes:
- All waveforms **positive-going** in this extraction. Pedestals: XA ~1700–3600, PMT ~7500–9300.
- The full-stream records carry **low-frequency baseline wander** (plain MAD 27–90 ADC) while
  the true sample-to-sample noise is only ~2 ADC (adjacent-difference estimator). Pulse
  finding/isolation on the cathode therefore runs on a high-passed trace (OpRoi-style
  `1−exp(−(f/0.05 MHz)²)`), while templates are extracted from the **raw** trace.
- The PDVD membrane XA pulse decays several times slower than the PDHD NP04 templates and has
  no deep AC undershoot; the cathode (full-stream) XA is AC-coupled like the PDHD full-stream
  channels but with a much longer, shallower recovery.

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

The stored template is the **mean 1-PE response in raw ADC** — its absolute scale IS the PE
calibration (§4).

## 3. Try-and-compare result

Metric: deconvolve the **held-out** average 1-PE pulse with each candidate; measure the
post-pulse residual 0.5–4 µs after the decon spike, relative to the candidate's own
**matched kernel** (template deconvolved with itself — the intrinsic tail a perfect template
gives; PDHD convention, `pdhd-spe-template-tuning.md`). Mean |residual| in % of the decon peak:

| candidate | cathode XA | membrane XA | PMT |
|---|---|---|---|
| **data-driven (own channel)** | **2.8** | **1.2** | **0.4** |
| data-driven (population avg) | 1.6 | 2.2 | 5.1 |
| SPE_NP04_FBK_2024 (PDHD) | 23.3 | 17.7 | – |
| SPE_NP04_HPK_2024 (PDHD) | 40.7 | 24.6 | – |
| SPE_NP02_estimate (larsoft VD) | 7.7 | 3.4 | – |

(`spe_template_compare.png` shows the per-channel bars.)

**Decision:** per-channel data-driven templates for all populations; population-average shape
(scaled to the channel 1-PE mode) for the 9 channels without enough clean 1-PE pulses. For
the cathode the population average is even marginally better than own-channel (the per-channel
tails are statistics-limited) — the final chain may use one template per population for the
XAs, mirroring the PDHD per-TYPE lesson; to be frozen after the filter scan (milestone 2) and
a cross-run stability check.

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

The cathode amplitude spectrum is a smooth threshold-limited continuum (seed = 5σ ≈ 10 ADC);
the apparent mode ~16–28 ADC is a turn-on artifact, not a resolved 1-PE peak. Possibilities:
the true 1-PE amplitude is below/near the noise (lower effective gain of the double-window
supercells), or pileup fills the valley (the cathode sees both volumes continuously). The
current templates therefore have the **shape** right (2.8 % held-out residual) but an
**uncertain absolute scale** on these 16 channels.

Planned anchors (milestone-2/3 work): (a) decon-area spectrum of isolated stream pulses — the
Wiener decon is a near-matched filter, so PE quantization may emerge in area where amplitude
fails; (b) gain transfer from the membrane XAs if the hardware settings (OV, DAPHNE gain) are
confirmed equal; (c) ask the PDS group for the DAPHNE gain settings of the cathode modules.

A second cathode caveat: the AC recovery is **longer than the 6.4 µs template window** (the
negative tail has not returned to zero at the window end). Isolated-medium-pulse probes of the
full recovery failed on this cosmic-rich data (every long window contains later light); the
truncation shows up only as a ≲2 % slow residual after bright pulses and will be revisited
together with the OpRoi settings for the 7.5 ms stream.

## 6. Channel flags

| channel | flag |
|---|---|
| ch2031 (OpDet 1, 2nd ch) | absent from the data entirely |
| ch2010 (OpDet 0) | no clean 1-PE peak, 0 selected pulses → fallback template; noisy (RMS 8.9) |
| ch2020 (OpDet 2) | distorted template (oscillating undershoot, only 15 pulses) — keep flagged |
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
python3 pd_plot/spe_build.py 0          # harvest run 039252 -> work/light_spe/harvest_r039252.npz
python3 pd_plot/spe_compare.py 39252    # comparison + all docs/pds figures + spe_summary.json
```

## 9. Open items

- Cathode PE-scale anchor (§5) and AC-recovery tail beyond 6.4 µs.
- ch3020 / ch2020 template re-selection; ch2010 needs a quieter run or longer statistics.
- Cross-run stability: repeat on 039253 / 039349 before freezing `pdvd-spe-templates.json`.
- SiPM vendor (FBK vs HPK) per module still unknown — now moot for the template choice
  (data-driven), but useful metadata.
- Filter parameters (`fixed_snr`, post-filter cutoff, per-population noise RMS) — milestone 2.

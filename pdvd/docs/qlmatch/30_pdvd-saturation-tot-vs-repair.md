# PDVD cathode saturation: a time-over-threshold (ToT) estimator, developed in simulation and judged against the two-sided repair in simulation and on data

**Repro:**
```
cd pdvd/docs/qlmatch && python3 scripts/saturation_tot_study.py --all
```
- Stages run in order: `--shape`, `--sim`, `--develop`, `--bench`, `--data`. The whole run takes about 4 minutes.
- Output: `d30/{shape,sim,develop,bench,data}.txt`, `pics/saturation_tot_sim.png` and
  `pics/saturation_tot_data_synthetic.png`.
- Cache: `/home/xqian/tmp/sat_tot/`, holding the harvested pulses, quiet windows, fitted shape and ToT table.
- Inputs: the raw-waveform files of `input_data_light_rawwf/` (runs 039252, 039253 and 039349; the pulse set is doc
  11's, from the first two) and the SPE templates `pgrapher/experiment/protodunevd/pdvd-spe-templates.json`.
- The run is seeded and deterministic: two `--bench` runs gave identical files.
- `saturation_recovery_study.py` (doc 11) is imported read-only. Its estimators, decon replica and pulse harvest are
  reused unchanged.

**Scope:** a study only. No toolkit, C++, jsonnet or runner change; production PDVD still runs doc 11's twoside repair.

## 0. Question and answer

A colleague suggested estimating a saturated pulse's charge from **how long the waveform stays at the rail** (time
over threshold). The owner asked for:
1. a search for that method in LArSoft;
2. if it isn't there, a ToT estimator built in simulation;
3. a comparison with our two-sided repair (doc 11, production `OpDecon::saturation_repair`), judged **separately**
   in simulation and on PDVD data.

**Answer.**
1. **Not in LArSoft** (§1). Every piece of saturation code found only flags, skips, clamps or excludes saturated
   waveforms. The one PDVD module that counts saturated ticks uses the count only to switch baseline removal off.
2. **Verdict S (simulation): ToT wins** at every depth from d = 2 to 20. Its bias is ≤ 1 %. Twoside passes only to
   d = 2 and reaches 1.35 at d = 4 and 2.5 at d = 20.
3. **Verdict D (data): ToT wins on each data test that can separate the methods.**
   - **Synthetic clips of held-out real pulses:** ToT passes to d = 6.7 with ≤ 5 % bias. Twoside fails from d = 4.
   - **Real rails where both ganged sub-channels saturate:** ToT keeps the pair ratio within 1–5 % out to the deepest
     rails (d ≳ 10). Twoside's band opens to ×1.4–4.
   - **Real rails whose partner did not saturate** only reach d ≈ 1.1–1.4. There all methods tie.
4. **The two verdicts agree.** The main weakness of ToT is also clear: it relies on the light's time profile. A
   ±20 % shift in the LAr slow constant biases it by up to ±15 % at d = 20, and a pileup pulse inside the rail run
   biases it by +20–50 %.

## 1. Search: ToT in LArSoft

| Where | What saturation code does | ToT → charge? |
|---|---|---|
| `duneprototypes` (local) `Protodune/vd/RawDecoding/DAPHNEReaderPDVD_module.cc` | PDVD DAPHNE decoding; no saturation logic (threshold/baseline fields commented out) | no |
| `duneopdet` v10_26_00d00 (CVMFS + GitHub) `OpticalDetector/WaveformPreProcessing_module.cc`, fcl `protodunevd_wvfpreprocessing` (L. Paulucci / A. Paudel, 2026; used by no standard workflow fcl yet) | `CheckSaturation()` counts **consecutive ticks ≥ 16383**. If a run exceeds `MaxTicksSat` 500, it **skips** the baseline-fluctuation removal. `DentCorrection()` refills a dip in the flat top. The denoiser keeps railed samples. | **no**: the count only toggles baseline removal |
| `duneopdet` `OpDetDigitizerProtoDUNE{VD,HD}_module.cc` | simulation clamps at `DynamicRangeSaturation` | no |
| `DUNE/waffles` (GitHub) `scripts/np02comm/beam_first_look/beam_saturated_check_*.py` | `remove_saturated`: **drops** waveforms > 15500 ADC in the beam window | no |
| `DUNE/waffles` `np04_analysis/ground_shakes/scripts/plots_talk.py` | histograms the width of saturated waveforms (PDHD), as a diagnostic | no |
| `duneprototypes` `spsbsm/SPSFilter/PDHDChannelSaturationFilter_module.cc` | TPC: counts consecutive negative ticks to **reject** events | no |
| `sbndcode` `OpDeconvolutionAlgWiener*_tool.cc`; `sbncode` `QLLMatch.cxx:166-184` | skip saturated waveforms (`UseSaturated: false`); flash matching zeroes the prediction for saturated PMTs | no |

Also searched: `dunecore`, `dunereco`, `dunesw` and `dunepdlegacy` (local or CVMFS), plus GitHub code search over the
`DUNE` organisation. **The colleague's exact recipe was not found.** If they can point to a talk or code, their
calibration can be dropped into the same bench (`methods_pe` in the script) and scored against the same tables.

## 2. The two approaches

| | **twoside** (doc 11, production) | **ToT** (this doc) |
|---|---|---|
| Measured input | the ≈ 4 samples before and 8 samples after the rail run | the rail-run length ToT (ticks) and the rail height R = 16383 − baseline |
| Model | single exponentials: rise τ and fall τ fitted to the **SPE template** | the channel's **bright-pulse shape** = SPE template ⊗ LAr scintillation profile (§3) |
| Reconstruction | fill = min(rise exp, fall exp), then deconvolve and integrate | **`tot_cal`**: PE = R · g_c(ToT), a per-channel table learned from simulation. **`tot_fill`**: solve the level λ at which the shape is ToT wide, fill the run with (R/λ)·s(t), then deconvolve and integrate. |
| Output | PE in the OpRoi window | same window, same decon, so the methods are directly comparable |

Why twoside overshoots: it anchors on the pulse **exit** and extrapolates backwards with the SPE's fall τ (38–137
ticks). A bright pulse's exit sits on the LAr slow component, which falls more slowly than it rose. The back-projection
therefore runs above the true top, and further above the deeper the rail.

ToT uses the whole run instead: a measured **time**, set by where the pulse crosses the rail on the way up and on the
way down. That is quantised at one tick, but it is not extrapolated.

## 3. The toy MC and its validation (`d30/shape.txt`, `d30/sim.txt`)

No LArSoft install is available on this machine, and WCT has no light simulation. The simulation is therefore a
waveform-level toy built only from measured PDVD inputs:
- **Pulse shape per channel:** SPE template ⊗ [f_p·δ + (1−f_p)·(b_i·e^{−t/τ_i}/τ_i + (1−b_i)·e^{−t/τ_s}/τ_s)] ⊗
  Gauss(σ).
  - It is fitted to the median shape of the run-039252 bright pulses only (857 pulses, window −30…+700 ticks; samples
    above 0.3 of the peak weighted ×3).
  - Fitted values: τ_s = 62–117 ticks (1.0–1.9 µs, the LAr triplet), τ_i = 2.5–8.4 ticks (the intermediate
    component), prompt fraction 0–0.32. Ch 1051's fit is degenerate (prompt fraction 1).
  - Residual rms is 1.2–2.1 %, except ch 1051 at 7 %: its SPE template is anomalous, with area/amp 113 against about
    40 elsewhere.
  - A two-component version (no τ_i) made the peak too narrow: 4 against 7 ticks wide at 90 % of the peak. The τ_i
    term fixed that.
- **Pulse-to-pulse variation:** the prompt fraction is refitted per pulse, giving σ = 0.032.
- **Noise and baseline:** real quiet 16384-sample windows of the same channel (40 per channel per run).
  - Training uses run 039252's windows; every test uses run 039253's.
  - Baselines vary by channel from 1040 to 5500 ADC, so R varies too.
- **Digitisation:** round, then clamp to [0, 16383].

**Validation on held-out data** (run 039253, 864 pulses):

| d | width above peak/d, data | sim | sim/data |
|---|---|---|---|
| 1.11 | 7 [6, 9] | 7 [5, 8] | 1.00 |
| 1.43 | 32 [24, 42] | 32 [18, 40] | 1.00 |
| 2.00 | 71 [62, 81] | 69 [62, 78] | 0.97 |
| 4.00 | 141 [128, 155] | 143 [134, 157] | 1.01 |
| 6.67 | 194 [180, 210] | 200 [187, 214] | 1.03 |

The raw and clip estimators match data within 1.5 % at every depth.

**One sim/data difference:** twoside overshoots more in sim than in data at d ≥ 2.9, reading 1.53 against 1.36 at
d = 5.6. The sim is therefore somewhat pessimistic for twoside, and Verdict D is the one to weigh.

## 4. ToT development (`d30/develop.txt`)

- **Training sample:** 300 simulated pulses per channel on run-039252 noise, d log-uniform in [1.05, 25].
- **`tot_cal`:** per channel, PE/R in 24 quantile bins of ToT, made monotone, then interpolated in log-log and
  extrapolated linearly in log-log beyond the table.
- **`tot_fill`:** uses the model shape directly. Self-check: the integer width at the solved level reproduces the input
  ToT within 3 ticks, and the fractional crossing makes it exact.
- **No data enters the calibration.** The only data input is the §3 shape fit on run 039252. Every data evaluation
  below is on run 039253 or on real rails.

## 5. Verdict S: held-out simulation (`d30/bench.txt`, `pics/saturation_tot_sim.png`)

Real 16383 rail; 320 pulses per depth; values are PE_rec/PE_true, median [16, 84].

| d | twoside | tot_cal | tot_fill | winner |
|---|---|---|---|---|
| 1.11 | 0.998 [0.997, 1.001] | 0.982 [0.953, 1.020] | 0.999 [0.999, 1.000] | tot_fill |
| 1.43 | 1.012 [0.997, 1.075] | 1.003 [0.975, 1.041] | 0.996 [0.989, 1.001] | tot_fill |
| 2.00 | 1.106 [1.058, 1.286] | 1.003 [0.998, 1.008] | 0.994 [0.987, 1.004] | tot_cal |
| 4.00 | 1.348 [1.203, 2.298] ✗ | 1.002 [0.986, 1.016] | 0.992 [0.976, 1.007] | tot_cal |
| 6.70 | 1.627 [1.386, 3.999] ✗ | 1.005 [0.987, 1.028] | 0.993 [0.975, 1.015] | tot_cal |
| 20.0 | 2.548 [1.885, 12.64] ✗ | 1.001 [0.977, 1.019] | 0.987 [0.964, 1.010] | tot_cal |

- Pass bar: |bias| < 20 % and half-band < 30 %.
- Deepest passing depth: twoside 2.0, tail 2.0, tmpl 1.43, raw and clip 2.0; **tot_cal and tot_fill 20**.
- The nominal row is the favourable case, because the test pulses come from the calibration's own model. The stress
  arms perturb the test pulses only; the calibration stays nominal.

**Stress arms**, tot_fill (tot_cal is within 0.02 of it) against twoside:

| arm | d = 4 | d = 6.7 | d = 20 | twoside at d = 4 |
|---|---|---|---|---|
| prompt fraction +0.08 (≈ 2.5σ) | 0.955 | 0.947 | 0.923 | 1.24 |
| prompt fraction −0.08 | 1.029 | 1.039 | 1.039 | 1.52 |
| τ_s +20 % | 1.003 | 1.051 | **1.164** | 1.40 |
| τ_s −20 % | 0.960 | 0.926 | **0.841** | 1.30 |
| extra 5-tick arrival spread | 1.014 | 1.019 | 1.031 | 1.51 |
| pileup: 30 % pulse 40–200 ticks later | **1.172 [0.94, 1.52]** | **1.465** | **1.469** | 2.00 |

- The shape arms stay inside the pass bar at every depth; the worst is τ_s ±20 % at d = 20.
- **Pileup inside the rail run lengthens ToT** and fails ToT from d = 4. It fails twoside even worse.
- `tot_cal` alone breaks at shallow depth under pileup (0.78 at d = 2), because the second pulse's own rail run is
  merged in. `tot_fill` does not break there.

## 6. Verdict D: PDVD data (`d30/data.txt`, `pics/saturation_tot_data_synthetic.png`)

### 6.1 (a) Synthetic clips of real unrailed pulses

- **Closure:** on all 1727 pulses, the raw, clip, tail, twoside and tmpl columns reproduce doc 11's §2.1 table exactly
  at d = 1.11–4.00. For example, twoside at d = 2 is 1.086 [1.017, 1.254] and clip at d = 4 is 0.602 [0.579, 0.626].
- Held-out run 039253, 864 pulses per depth:

| d | twoside | tot_cal | tot_fill | winner |
|---|---|---|---|---|
| 1.11 | 0.998 [0.996, 1.001] | 1.018 [0.955, 1.101] | 0.999 [0.998, 1.000] | tot_fill |
| 1.43 | 1.018 [0.993, 1.057] | 1.029 [0.979, 1.096] | 0.988 [0.981, 0.999] | tot_fill |
| 2.00 | 1.087 [1.015, 1.253] | 1.023 [0.973, 1.084] | 0.995 [0.968, 1.020] | tot_fill |
| 2.86 | 1.142 [1.033, 1.563] | 1.009 [0.954, 1.079] | 0.988 [0.945, 1.028] | tot_fill |
| 4.00 | 1.234 [1.068, 2.024] ✗ | 0.990 [0.919, 1.055] | 0.973 [0.917, 1.026] | tot_cal |
| 5.60 | 1.366 [1.114, 2.920] ✗ | 0.971 [0.892, 1.056] | 0.959 [0.894, 1.034] | tot_fill |
| 6.70 | 1.448 [1.151, 3.590] ✗ | 0.963 [0.878, 1.065] | 0.951 [0.875, 1.045] | tot_cal |

Deepest passing depth on data: twoside 2.86; **tot_cal and tot_fill 6.7**, the deepest depth tested. d > 6.7 cannot be
tested this way, because the bright pulses peak at 5600–11700 ADC.

The data ToT drifts about 3–5 % low at d ≥ 4, against about 1 % in sim. This is the same size as the sim's 3 % overshoot
in width at d = 6.7 (§3), so it is a residual model-shape error, not noise.

### 6.2 (c) Real rails whose ganged partner did not rail

- **Idea:** the two sub-channels of a cathode X-Arapuca (`1000+10x` and `1001+10x`) see the same light. Where one
  rails and its partner doesn't, the truth is the partner's PE × the pair's PE ratio.
- **Pair ratios:** measured on 687–720 bright pulses per pair where neither rails. The ratio spread is ±4.2–6.5 %,
  which is the truth's own resolution.
- **Coverage:** 5 files, 1137 such rails. But the truth-implied d is 1.11 at the median and 1.38 at the 95th
  percentile. A partner avoids the rail only when the pulse is barely saturated, so this test **cannot reach deep
  rails**.

| d | n | twoside | tot_cal | tot_fill | winner |
|---|---|---|---|---|---|
| 1.0–1.25 | 701 | 0.997 [0.955, 1.043] | 1.017 [0.908, 1.348] | 1.000 [0.958, 1.048] | tot_fill |
| 1.25–1.6 | 226 | 0.963 [0.931, 1.004] | 0.918 [0.862, 0.976] | 0.969 [0.935, 1.008] | tmpl |

All methods tie within the ±5 % truth resolution. The exception is `tot_cal`, whose few-tick ToT is quantised at this
depth. `tot_fill` avoids that because it fills the run and then measures the unrailed rest of the pulse directly.

### 6.3 (c2) Real rails where both sub-channels rail

- **Metric:** q = (PE_a/PE_b)_rec / (PE_a/PE_b)_controls. It should be 1 for any method whose bias does not depend on
  depth.
- **Selection:** only pairs whose peak ratio is outside [0.87, 1.15], so that the two channels saturate to clearly
  different depths. 1050/1051 is excluded.
- **Binning:** by the more-saturated channel's measured ToT, which is estimator-free. Selected bins are shown; the
  full table is in `d30/data.txt`.

| ToT (ticks) | ~d | n | clip | twoside | tot_cal | tot_fill |
|---|---|---|---|---|---|---|
| < 40 | < 1.4 | 709 | 1.016 [0.950, 1.069] | 1.003 [0.953, 1.055] | 1.027 [0.857, 1.130] | 1.007 [0.958, 1.056] |
| 71–108 | 2–2.9 | 340 | 1.059 [0.909, 1.148] | 0.958 [0.872, 1.091] | 0.943 [0.856, 1.084] | 0.974 [0.911, 1.054] |
| 141–194 | 4–6.7 | 313 | 1.093 [0.913, 1.175] | 0.906 [0.816, **1.392**] | 0.946 [0.843, 1.058] | 0.948 [0.874, 1.054] |
| 194–260 | 6.7–10 | 226 | 1.133 [0.879, 1.176] | 0.959 [0.771, **1.506**] | 1.006 [0.902, 1.111] | 1.010 [0.909, 1.099] |
| ≥ 260 | ≳ 10 | 163 | 1.179 [1.125, 1.232] | 0.919 [0.674, **4.361**] | 1.004 [0.861, 1.119] | 1.002 [0.833, 1.114] |

- **Clip** drifts steadily, because its under-count grows with depth. That shows the metric is sensitive.
- **Twoside's** median stays near 1, but its upper band opens to ×1.4–4.4 at depth. That is the same growing overshoot
  tail seen in (a) and in the sim.
- **ToT** stays within 1–5 % of 1, with a half-band of 0.09–0.14. This is roughly the √2 × ±6 % expected from the
  control ratio alone.
- This is the only data test of deep **real** rails in this doc, and ToT passes it. It is a **relative** test: a shape
  error common to both sub-channels partly cancels in q. The absolute depth dependence is covered by (a) up to 6.7.

### 6.4 (d) Census of real cathode rail runs

- **Sample:** 13,357 runs across 5 files.
- **Edges:** 12 touch a stream edge. The cathode is a full stream, so edge truncation of ToT is negligible.
- **Run-length (ToT) quantiles** 16/50/84/95/99: 8 / 43 / 162 / 254 / 379 ticks.
- Mapped through the model widths at level 1/d:

| real rail runs deeper than | d > 2 | d > 4 | d > 6.7 | d > 10 | d > 20 |
|---|---|---|---|---|---|
| fraction of runs | 39.0 % | 20.0 % | 10.6 % | 5.6 % | 1.9 % |

About a fifth of real rail runs are in the regime where twoside fails. Bright flashes collect exactly these runs: doc
11 §2.2 found a median of d ≈ 5.6 for the deepest rail of a railed flash.

## 7. The two verdicts side by side

| test | twoside | ToT (tot_fill) | better |
|---|---|---|---|
| **S** nominal, d 2 / 4 / 20 | 1.11 / 1.35 / 2.55 | 0.99 / 0.99 / 0.99 | ToT |
| **S** worst shape arm (τ_s ±20 %) at d = 20 | 1.94–3.29 | 0.84–1.16 | ToT |
| **S** pileup at d = 4 | 2.00 [1.28, 3.75] | 1.17 [0.94, 1.52] | ToT (both fail) |
| **D (a)** held-out real pulses, d 4 / 6.7 | 1.23 / 1.45, fails | 0.97 / 0.95, passes | ToT |
| **D (c)** real rails, partner truth, d ≤ 1.4 | 0.997 / 0.963 | 1.000 / 0.969 | tie |
| **D (c2)** real deep rails, pair consistency | band to ×1.4–4.4 | ±10 % | ToT |

**The two judgements agree.** The sim predicts that ToT removes twoside's growing overshoot, and the data show exactly
that, with a slightly smaller margin: ToT's data bias is 3–5 % low at d ≥ 4, against 1 % in sim, and twoside's data
overshoot is a little smaller than in sim.

## 8. Pros and cons

| | twoside (production) | ToT |
|---|---|---|
| **+** | no shape model, only the SPE template's τ; purely local; already in production behind a knob, byte-gated | uses a measured *time* over the whole run, not a few anchor samples; bias flat to d = 20 in sim and ≤ 5 % to d = 6.7 on data; consistent on deep real rails |
| **+** | exact at d ≲ 1.5 | `tot_fill` is exact at shallow depth too, so there is no hand-over |
| **−** | overshoot grows with depth (×1.2 at d = 4, ×1.45 at d = 6.7 on data), with a long upper tail (×2–4); fails for the ≈ 20 % of real runs beyond d ≈ 4 | needs a per-channel bright-pulse shape (template ⊗ LAr profile). The profile depends on the LAr triplet τ (purity, N₂ quenching; any xenon doping would change it completely) and on particle type (prompt fraction). ±20 % in τ_s gives ±15 % at d = 20, ±5 % at d = 6.7. |
| **−** | | a second pulse inside the run lengthens ToT: +17 % at d = 4, +47 % at d ≥ 6.7 in sim |
| **−** | | the calibration must be kept current per run period (shape fit plus quiet windows; 4 minutes here). Channels with an anomalous SPE template (1051) have a poor shape model. |
| **−** | | tested here only on **cathode full-stream** rails. Membrane and PMT self-trigger snippets are short, and doc 14's overflow-to-0 runs may be truncated by the snippet end, making ToT a lower bound there. Not assessed. |
| **−** | | `tot_cal` alone is quantised at d ≲ 1.4 and mis-reads pileup at shallow depth; `tot_fill` is the variant to keep |

## 9. Recommendation and scope

- **The better estimator for PDVD cathode rails is ToT, in its `tot_fill` form.** It fills the rail run with the
  channel's bright-pulse shape, scaled so that it is exactly ToT wide at the rail, then goes through the existing
  decon and ROI path. It is a drop-in replacement for the fill step of `OpDecon::repair_runs`.
- **It is not adopted here.** Adopting it needs the owner's go on these steps:
  1. a default-OFF `OpDecon` knob (for example `saturation_repair_mode: "tot"`), with the per-channel shape parameters
     of `d30/shape.txt` shipped as a data file next to `pdvd-spe-templates.json`;
  2. a doctest that pins `tot_fill` against this script's Python on a few pulses;
  3. a byte-identical knob-off gate on the PDVD light chain;
  4. a knob-on rerun of the doc 11 §6 and doc 12 numbers (the railed chi2 terms, meas/pred, QtoL), since QLMatching's
     `chi2_sat_inflate` 0.5 was chosen for twoside's error, not ToT's.
- **Open items before a flip:**
  - (i) repeat the shape fit on a later run, to see whether τ_s drifts with purity;
  - (ii) check membrane and PMT self-trigger snippets for ToT truncation;
  - (iii) decide whether a run that is visibly two pulses (a dip back below the rail > 2 ticks) should fall back to
    twoside.

## 10. Files
- `scripts/saturation_tot_study.py`: this study.
- `d30/shape.txt`, `sim.txt`, `develop.txt`, `bench.txt`, `data.txt`: records (§3–6).
- `pics/saturation_tot_sim.png`, `pics/saturation_tot_data_synthetic.png`.
- Related docs:
  - `11_pdvd-saturation-recovery.md`: the estimator bench and the twoside choice;
  - `14_pdvd-lightpattern-sp-investigation.md`: the saturation signature and overflow-to-rail;
  - `12_pdvd-qtol-recalibration.md`.

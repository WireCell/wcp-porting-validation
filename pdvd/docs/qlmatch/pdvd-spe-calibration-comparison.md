# PDVD SPE calibration comparison: our templates vs LArSoft/duneopdet vs independent amplitudes

**Repro:**
```
cd pdvd/docs/qlmatch && python3 spe_calibration_comparison.py
# prints the tables below + writes spe_calibration_comparison.png
```
Inputs are read-only: `cfg/pgrapher/experiment/protodunevd/pdvd-spe-templates.json`
(our run-039252 1-PE harvest) and values transcribed verbatim from cvmfs
`duneopdet v10_21_02d00` (read 2026-07-14).

## 1. What is being compared, and why the units line up

Both chains digitize DAPHNE at 62.5 MHz (16 ns ticks), so SPE **areas in
ADC·ticks** and **amplitudes in ADC** are directly comparable numbers:

| source | quantity | role |
|---|---|---|
| **ours** — `pdvd-spe-templates.json` | per-channel 1-PE **template** (amp = max, area = sum) | with the F(0)=1 wiener-inspired OpDecon, the template IS our PE scale: `pe_ours ≈ raw_area / template_area` |
| **keepup** — `standard_reco_protodunevd_keepup.fcl` → `protodunevd_ophit` | flat `SPEArea: 410` ADC·ticks, PE = raw area/410 + 0.43 (AlgoSiPM on raw waveform, no decon) | what the official chain actually uses for every channel (`UseCalibrator: false`) |
| **PhotonCalibratorDUNE.fcl** (v10_21_02d00) | six dated per-channel `SPESizes` sets (ADC·ticks) | **not used** by keepup; header "STILL TO BE OBTAINED. DO NOT USE"; v300/v400 membrane from finger plots (Henrique) "most likely underestimated by a factor 2-3"; v400_2 cathode from Ajib (indico 69308); all 24 PMT entries placeholder 100 in every set |
| **colleague** case statement | per-channel SPE **amplitudes** (ADC): cathode `0.6·x`, membrane 2050–2081 `0.6·x` (2010–2041 flat 50 = placeholder), no PMTs | fully independent of both chains |

Known-bad templates excluded from statistics (kept in tables): ch1051
(threshold artifact, area 8557) and ch2020 (distorted, negative area) —
`pdvd-spe-template.md` §6.

## 2. Results (medians [16,84] over channels)

### Amplitudes — the cleanest cross-check, and it validates our templates

| block | our_amp / colleague_amp | our_amp / colleague_raw (÷0.6 undone) |
|---|---|---|
| cathode (n=15) | **1.07 [0.80, 1.25]**, Spearman 0.69 | 0.64 [0.48, 0.75] |
| membrane (n=8) | 1.58 [1.46, 1.72], Spearman 0.69 | **0.95 [0.88, 1.03]** |

Reading: the colleague's `0.6·x` prefactor evidently applies to the
**cathode only** (consistent with the cathode's different signal path —
power/signal over fiber); their membrane `x` values are direct amplitudes.
With that reading, **both PD types agree with our template amplitudes at
unity to ~±25%, with per-channel rank correlation ~0.7**. This is the
strongest external validation the provisional cathode PE scale has received:
our cathode 1-PE amplitude mode (~16–28 ADC), which we feared was a
threshold turn-on artifact, matches an independent calibration channel by
channel at the tens-of-percent level.

### Areas — the official per-channel sets disagree with each other far more than with us

| block | vs v100 ("do not use") | vs v300 (finger plot, "×2-3 under") | vs v400_2 (cathode = Ajib) | vs keepup flat 410 |
|---|---|---|---|---|
| cathode | 0.32 [0.25, 0.44], ρ=0.72 | 5.32 [3.93, 7.02], ρ=0.72 | **1.88 [1.30, 2.58], ρ=0.20** | 2.23 [1.53, 2.61] |
| membrane | 0.47 [0.43, 0.77], ρ=0.67 | 7.41 [6.59, 11.14], ρ=0.24 | 7.56 [6.96, 12.36], ρ=0.67 | 3.98 [3.02, 5.33] |
| PMT | — | — | — (all placeholder) | 3.37 [2.62, 4.55] |

Reading, in decreasing order of confidence:

1. **The official sets span ×16 among themselves** (v100 vs v300 for the
   same channels) — they cannot all be areas of the same pulse in the same
   units. Whatever convention each used (integration window, filtering,
   prefactor) is not recoverable from the fcl.
2. **Membrane is our control sample**: our membrane templates have textbook
   1-PE peaks (36–64 ADC) *and* now match the colleague's amplitudes at
   0.95. Yet our membrane areas are ×7.4 the finger-plot v300/v400 values —
   more than their own stated ×2-3 underestimate. So the finger-plot
   SPESizes are unusable as absolute areas, which also caps the trust in
   v400_2's membrane half.
3. **Ajib's cathode areas (v400_2)** would imply our cathode PE is
   *underestimated* ×1.88 median — but the per-channel correlation with our
   areas is only ρ=0.20 (vs 0.7 everywhere else), and the colleague-amplitude
   route (their amp × our per-channel pulse width) reproduces **our** areas,
   not Ajib's (implied/Ajib = 1.24–2.7 per channel). Two independent
   amplitude-based estimates agreeing with each other and disagreeing with
   the area-based one suggests Ajib's integration convention differs
   (shorter window would lose the cathode's long tail — our templates carry
   an exp-repaired tail with effective width ~46 ticks vs PMTs' ~11).
4. **Keepup's flat 410** sits ×2.2 (cathode) / ×4.0 (membrane) / ×3.4 (PMT)
   below our per-channel areas: the official keepup chain would *over*-count
   PE relative to us by those factors, channel-type-dependently. Nothing in
   the official chain today is a per-channel calibration we could adopt.

### Per-channel table

Full 56-channel table (our amp/area, colleague amp, v300, v400_2, flat 410,
implied corrections) is printed by the script; the PNG
(`spe_calibration_comparison.png`) shows ours-vs-Ajib areas, ours-vs-colleague
amplitudes, and the cathode per-channel area bars.

## 3. Conclusions for our chain

- **No LArSoft-side per-channel SPE set is mature enough to correct our
  templates.** The keepup chain itself uses none of them. Our run-039252
  data-driven templates remain the best available calibration for PDVD.
- **The provisional cathode PE scale is in better shape than feared**: two
  independent amplitude calibrations (colleague; and via them, effectively
  the membrane control) agree with it at ~unity ±25%. A coherent cathode-wide
  gain error of the ×2–3 kind Ajib's areas would imply is disfavored but not
  excluded (area conventions unverifiable); the ×13 per-channel meas/pred
  spread seen in QL matching is **not** explained by any external SPE set
  (nothing coherent at ρ=0.2).
- Follow-up (decision gate 3 of this campaign): refit the per-channel
  `measured_pe_scale` on saturation-repaired dumps; compare *that* per-channel
  pattern against Ajib's — if they correlate, the SPE-area route gets a
  second life.

## 4. Provenance quotes

- `protodunevd_ophit` (opticaldetectormodules_dune.fcl v10_21_02d00):
  `SPEArea: 410`, `SPEShift: 4.3e-1`, `AreaToPE: true`, `UseCalibrator:
  false`, HitAlgoPset `Name: "SiPM"` — PE = raw pulse area/410 + 0.43.
- `PhotonCalibratorDUNE.fcl` v400_2 comment: "Values from
  waffles/np02_data/calibration_data/np02-config-v4.0.0.csv for membrane
  modules from finger plot (Henrique), values most likely underestimated by a
  factor 2-3. Cathode values obtained by Ajib (indico 69308 ...
  Preliminary_LowPE_background_Study.pdf)".
- Keepup chain has **no deconvolution and no saturation handling**; the
  rail-aware `WaveformPreProcessing` and Wiener `Deconvolution` modules exist
  in duneopdet but are not wired for VD. (Context for the companion
  saturation study, `pdvd-saturation-recovery.md`.)

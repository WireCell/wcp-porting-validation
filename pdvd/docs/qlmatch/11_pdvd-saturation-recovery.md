# PDVD cathode saturation: PE-estimator study and the chosen processing scheme

**Repro** (step 1 makes the validation input for step 2):
```
cd pdvd && export WIRECELL_PATH=/nfs/data/1/xqian/toolkit-dev/toolkit/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data
wcsonnet -A input_file=$PWD/input_data_light_rawwf/np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root \
         -A output_file=/home/xqian/tmp/satstudy/decon-dump-298567.tar.gz \
         -S run=39252 -S event=298567 \
         -o /home/xqian/tmp/satstudy/decon-dump.json docs/qlmatch/scripts/wct-decon-dump.jsonnet
wire-cell -l stderr -c /home/xqian/tmp/satstudy/decon-dump.json
cd docs/qlmatch && python3 scripts/saturation_recovery_study.py --all
```

Context: `09_pdvd-pd-mapping-investigation.md` §6-7 established that the DAPHNE
saturation veto zeroes railed cathode channels in 42% of bright (≥1000 PE)
flashes. This study answers the follow-up: **given saturation, what is the
best signal processing** — OpHit without deconvolution? deconvolution through
the clipped waveform? waveform repair? — and what must happen downstream.

## 1. Method

**Validated replica.** A NumPy replica of the production
`Flash::OpDecon` wiener-inspired deconvolution (F(0)=1, σ=1.25 MHz cathode,
per-channel run-039252 SPE templates) reproduces the pipeline decon dump
(`wct-decon-dump.jsonnet`, evt298567, 16 cathode channels) to **3×10⁻⁷
relative** (float32 round-off) both at waveform level and on pulse integrals.
The study's PE estimator = decon integral over `[peak−50, peak+300]` ticks
(the OpRoi pads) with a local decon-baseline correction.

**Ground truth = synthetic clipping of real unrailed pulses.** 1727 bright
UNRAILED cathode pulses (peak 3000–12000 ADC over baseline, runs
039252/039253) are clipped at fake rails `base + f·(peak−base)`,
f ∈ {0.9, 0.7, 0.5, 0.35, 0.25} (clip depth d = 1/f up to 4). PE_true = the
production estimator on the unclipped pulse. Same channel, same electronics,
same pileup and noise — a non-circular, quantitative benchmark.

**Estimators.**
- `raw` — LArSoft-keepup style: baseline-subtracted RAW area / template area,
  no deconvolution (the official `protodunevd_ophit` runs AlgoSiPM on raw
  waveforms; the keepup chain has **no** deconvolution and **no** saturation
  handling — clipped pulses silently under-report there too).
- `clip` — deconvolve straight through the rail (our current veto-off mode).
- `tail` — repair: fill the rail run with the falling-edge single-τ
  exponential back-extrapolation (τ from the channel SPE-template tail).
- `twoside` — repair: `min(rising-edge exp, falling-edge exp)`, clamped ≥
  the rail; the exp intersection approximates the peak.
- `tmpl` — repair: fill with the channel's own SPE template shape anchored on
  the exit samples.

## 2. Results

### 2.1 Bias vs clip depth (median [16,84] of PE_rec/PE_true, n=1727)

| d = 1/f | raw | clip | tail | twoside | tmpl |
|---|---|---|---|---|---|
| 1.11 | 0.961 | 0.995 [0.995,0.996] | 0.998 | 0.998 | 1.023 |
| 1.43 | 0.927 | 0.962 [0.953,0.968] | 1.039 | 1.017 [0.993,1.058] | 1.051 |
| 2.00 | 0.824 | 0.859 [0.840,0.876] | 1.176 | 1.086 [1.017,1.254] | 1.252 |
| 2.86 | 0.691 | 0.727 [0.703,0.748] | 1.331 | 1.149 [1.032,1.573] | 1.452 |
| 4.00 | 0.567 | 0.602 [0.579,0.626] | 1.511 [1.25,3.41] | 1.232 [1.060,2.047] | 1.603 |

(figure: `saturation_recovery_bias.png`)

Readings:
- **"OpHit without deconvolution" does not help saturation.** `raw` loses the
  same clipped charge as `clip` plus a few % (and gives up the decon's pileup
  separation). On unclipped pulses raw-area/decon = 0.966 [0.951,0.979] —
  the two scales agree; there is no hidden robustness win.
- **`clip` is a tight deterministic underestimate** (±2% band): −4% at
  d=1.4, −14% at d=2, −40% at d=4. Bounded and predictable — good as a
  conservative lower bound.
- **Repairs overshoot with growing scatter.** The single-exp `tail` misses
  the fast decay component right after exit; `tmpl` (pure SPE shape) is too
  peaked because a bright pulse is SPE ⊗ scintillation-time-profile (LAr slow
  component ≈ 90 ticks), broader than a lone SPE near the peak. `twoside` is
  the best repair: +2% at d=1.4, +9% at d=2, +23% median at d=4 but with a
  [1.06, 2.05] band.
- Nothing is trustworthy beyond d ≈ 3.

### 2.2 How deep are REAL rails? (120 events, rails within ±4 µs of a ≥1000 PE flash)

13042 rail runs: median d ≈ 1.9, but the distribution is heavy-tailed —
37% above d=3, 25% above d=5 (d estimated by tail back-extrapolation, itself
an over-estimator for long runs, so these are upper bounds — but the hand
checks below confirm genuinely deep rails). Per bright railed flash
(n=2922), the **deepest** rail:

| pass bar | flashes fully "repairable" |
|---|---|
| max-d ≤ 1.5 | 23% |
| max-d ≤ 2 | 30% |
| max-d ≤ 3 | 38% |
| max-d ≤ 5 | 47% |

**The median railed bright flash has a d ≈ 5.6 rail.** Deep saturation is the
norm, not the exception, for exactly the flashes that anchor Q/L matching.

### 2.3 Real-rail cross-check (evt298567 hand-confirmed crossers, satoff dump)

Scaling the prediction on the unrailed cathode OpDets and comparing the
railed ones (`--realrail`): repairs move railed channels in the right
direction but do NOT close deep holes — e.g. gid37 od7 (C5, 320 rail
samples): veto-off dump 2468, clip 2737, twoside 5124, tmpl 6827 vs
prediction-scaled ≈ 10600. The unrailed controls also show the known
per-channel meas/pred spread (×0.4–2.4), so this check is qualitative;
the quantitative verdict rests on §2.1-2.2.

## 3. Conclusions — the chosen scheme

1. **Keep the deconvolution.** Raw-waveform hits (the official-keepup mode)
   have no saturation advantage and lose pileup separation. Deconvolving
   through the rail (`veto_saturation=false`) is a well-behaved, tightly
   bounded underestimate.
2. **Repair is a second-order improvement, not the fix.** Only ~30% of
   railed bright flashes are entirely in the d ≤ 2 regime where `twoside`
   repair is accurate to ~10%. For the deep-rail majority no waveform
   estimator is reliable.
3. **The physics-critical fix is flag propagation** (option iv of
   `09_pdvd-pd-mapping-investigation.md` §7): keep the (clipped or repaired) hit
   with `veto_saturation=false`, propagate a per-flash per-channel saturation
   flag through OpHitFinder → OpFlashFinder (`flash_sat` companion tensor) →
   QLMatching, and **exclude flagged channels from the chi2/KS/LASSO terms**
   the way dead channels are masked — but per flash, not per run. The
   measured PE stays in the flash (totals, display, thresholds) as a
   conservative estimate; it just cannot mislead the matching.
4. Recommended operating point (all default-OFF knobs):
   `detect_saturation=true, veto_saturation=false, flag_saturation=true`
   (+ QLMatching `use_saturation_flag=true`), with the `twoside` repair
   (`saturation_repair`) optional on top for better flash totals. The QtoL
   recalibration (companion doc `12_pdvd-qtol-recalibration.md`) must likewise
   exclude railed channels from its Σmeas/Σpred anchors.

## 4. Files

- `saturation_recovery_study.py` — this study (validate / study / realrail /
  prevalence parts); `wct-decon-dump.jsonnet` — pipeline decon dump job.
- `saturation_recovery_bias.png` — §2.1 figure.
- Logs quoted above: /home/xqian/tmp/satstudy/{study2,realrail}.log
  (scratch; regenerate via the Repro block).

## 5. Implementation (toolkit d29d5f67, all default-OFF)

The keep-and-mark chain:
- `OpHitFinder.flag_saturation` — keep saturated hits, append a 10th ophit
  column with the rail-overlap flag (9 cols / bit-identical off).
- `OpFlashFinder` — with a 10-col input, emit the per-flash per-OpDet
  `flash_sat` tensor (4th archive member; absent otherwise).
- `Aux::FlashTensorToOpticalPCs` — carries `flash_sat` on the light-PC
  `error` field (previously unconsumed, always 0). Data-driven, no knob.
- `Match::Opflash::get_sat(ch)`; `QLMatching.use_saturation_flag` — flagged
  channels leave that flash's opdet mask (pred/chi2/KS via the existing
  `bundle_mask_ks: true`) and their LASSO rows are zeroed (round1/round2/
  joint fills). Calib dump gains a per-flash `sat` array (knob-on only).
- `OpDecon.saturation_repair` (+`repair_fit_samples`, default 8) — the
  `twoside` bridge of §2.1 before decon; mask emission unchanged.

Config/runner plumbing (wcp repo): `wct-light-reco.jsonnet` tla args
`flag_saturation` / `saturation_repair` (env `PDVD_FLAG_SATURATION=1` /
`PDVD_SAT_REPAIR=1` in `run_light_evt.sh`); `wct-clustering.jsonnet` arg
`ql_use_saturation_flag` (env `PDVD_QL_USE_SAT_FLAG=1` in
`run_clus_evt.sh`); key-suppression in `cfg/.../protodunevd/
{flash,qlmatching}.jsonnet`.

**Gates (all PASS):**
- Compiled configs knobs-off byte-identical: light vs the stored
  `work/039252_light298567{,_satoff}/.wct-light.json`; clustering old-vs-new
  jsonnet (git-stash diff). Knob-on keys present (3 branches / QLMatching).
- `hash_archive.py` member-identical knobs-off: PDVD light evt298567 vs
  canonical (`work/039252_light298567_abflag`), PDHD allpd 29107 evt1015 vs
  canonical (`work/029107_allpd1015_abflag`).
- QL knobs-off calib dump **byte-identical**: `work/039252_0_satoffab/`
  vs `work/039252_0_satoff/` (covers the FlashTensorToOpticalPCs / Opflash /
  QLMatching code paths shared with SBND/uboone).
- `wcdoctest-{flash,match,aux}` pass.

**Knob-on smoke (evt298567, operating point veto-off + flag + repair
= `_satrep`):** `flash_sat` flags exactly the railed cathode OpDets
({5,7,9} for gid37, {4,6,7,8} for gid57); C++ repair od7 2468 → 4843
(Python `twoside` prototype 5124, −5% from estimator details); gid37 total
13924 → 16330. Full QL with `PDVD_QL_USE_SAT_FLAG=1`
(`work/039252_0_satrep/`): 32/198 flashes carry sat flags, the three hand
crossers keep their matches (KS 0.15–0.25).

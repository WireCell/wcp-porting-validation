# PDVD QtoL recalibration on saturation-fixed dumps (crosser anchors)

**Repro:**
```
# inputs: the 120-event _satrep reprocess (all three runs)
cd pdvd && export PDVD_VETO_SATURATION=0 PDVD_FLAG_SATURATION=1 PDVD_SAT_REPAIR=1
./run_light_all.sh -s _satrep 39252 ; ./run_light_all.sh -s _satrep 39253
for f in input_data_light/np02vd_raw_run039349_*_rawwf.root; do ./run_light_all.sh -f "$f" -s _satrep 39349; done
# QL (imaging symlinked per index dir from the plain dirs):
export OMP_NUM_THREADS=8 PDVD_MAX_JOBS=6 PDVD_LIGHT_SUFFIX=_satrep PDVD_QL_USE_SAT_FLAG=1
./run_clus_evt.sh -s satrep -calib 039252 all   # + 039253, 039349
# fits:
cd ql_light_calib && python3 fit_qtol_crossers.py              # satrep dumps, dump preds (128 nm)
python3 fit_qtol_crossers.py --tag ""                          # veto-ON baseline (bias check)
python3 fit_qtol_crossers.py --relib 128                       # replica validation (must == dump)
python3 fit_qtol_crossers.py --relib 175                       # Xe/175 nm hypothesis
```

## 1. Setup

Anchors = **strict geometric cathode-crosser triplets** (both halves' cathode
ends within 3 cm of the cathode at the flash time, end (y,z) within 20 cm,
flash ≥1500 PE dominating ±30 µs — the `pd_mapping_audit.py` harvest). They
are external geometric ground truth, independent of the LASSO/auto selection
(the `fit_qtol_gold.py` ×350 trap is respected). 120 events, three runs →
**192 anchors** (150 with both halves bundled). Estimator = median over
anchors of Σmeas/Σpred on channels that are active, unmasked, and **not
saturation-flagged in that flash** (the new dump `sat` array —
`pdvd-saturation-recovery.md`); QtoL_new = 0.094 × median.

The reprocess itself works as designed: bright (≥1000 PE) flashes with an
exactly-zero active cathode channel drop from **42% (veto-ON) to 8%**
(residual = genuinely dim channels); 188/192 anchors carry ≥1 flagged
channel, properly excluded from both sums.

## 2. Results

### 2.1 The veto had been hiding a ×5 cathode bias

| | veto-ON dumps (old anchors) | satrep dumps |
|---|---|---|
| global median Σmeas/Σpred | 1.76 [0.92, 4.09] n=172 | **5.99 [3.43, 17.2] n=192** |
| cathode-XA-restricted | 1.86 | **10.1** |
| membrane-XA-restricted | 1.74 | 1.66 |
| PMT-restricted | 0.34 | 0.35 |

On veto-ON dumps the railed (= nearest, highest-prediction) cathode channels
sat at exactly 0 in Σmeas while their large pred stayed in Σpred — dragging
the cathode ratio down ×5. With rails excluded from both sums, the unrailed
cathode channels show meas/pred ≈ 10 (directly confirmed by the raw-waveform
scale of the evt298567 hand crossers: unrailed-od scale 12.2). Membrane and
PMT groups (rarely railed) are unchanged — the veto bias was cathode-specific.

### 2.2 A single QtoL does not exist under the current Ar/128 nm default

Naively, QtoL = 0.094 × 5.99 = **0.56 [0.32, 1.62]** — but the scatter is
model structure, not statistics:
- per-type flash-level medians span ×30 (cathode 10.1 / membrane 1.66 / PMT 0.35);
- per-channel ratios **rise toward dim (far) channels** in both XA groups
  (cathode 7.9→31, membrane 2.3→14 across the pred-PE quintiles): the
  128 nm library's visibility falls too steeply with distance.

### 2.3 Under the 175 nm library the far-field reference group is consistent as-is

Recomputing every anchor's prediction with the Python vis-loop replica
(`--relib`; the 128 nm recompute reproduces the dump predictions exactly,
median 5.985 = 5.985):

| (175 nm) | flash-level median | per-channel vs pred-PE (bright→dim) |
|---|---|---|
| membrane XA | 0.87 | **1.15 / 0.97 / 1.05** / 1.64 / 3.62 — flat ≈ 1 |
| cathode XA | 8.5 | 6.9 / 7.6 / 7.9 / 10.6 / 20 — uniform ×7 + mild tail |
| PMTs | 0.21 | — |
| global | 4.41 → "QtoL 0.41" | |

The membrane X-Arapucas — the group whose SPE calibration has textbook 1-PE
peaks and passed the external amplitude cross-check
(`pdvd-spe-calibration-comparison.md`) — are consistent with the **175 nm
model at the current QtoL 0.094 with no correction**, flat in brightness.
Under 128 nm the same group needs ×2-4 with a distance slope. This re-affirms
the earlier Xe/175 data verdict (`pdvd-questions-dune.md` §3) on
saturation-fixed data — the previous verdict AND the later revert to Ar/128
were both computed on hole-punched measurements.

The cathode excess is now a clean, nearly **uniform ×7** (175 nm), i.e. an
efficiency-like factor, not a distance shape: candidates are the official
eff_Xe for the cathode XAs being low for the as-built modules, a
double-sided-visibility accounting issue in the sampled grid at the x=0
plane, near-field bias of the 10 cm grid for tracks lying in the cathode
plane, or (less likely after the SPE comparison) a cathode PE-scale error.
PMTs measure ×5 below membrane on the same anchors — an eff_TPB/PEN-level
question.

## 3. Recommendation as first reported (superseded by the §4 owner ruling)

1. **Do not adopt a global QtoL from these numbers under Ar/128 nm** — the
   model's distance shape makes any single value topology-dependent. For the
   record, that fit is 0.56 [0.32, 1.62].
2. **Revisit the 128↔175 nm choice on the satrep dumps** (the machinery is a
   one-flag rerun now): under 175 nm the calibrated reference group says
   QtoL ≈ 0.094 is already right, and the remaining work becomes per-type
   `VUVEfficiency` corrections — cathode ×~7, PMT ×~0.2 relative to membrane
   — which are efficiency knobs, not QtoL.
3. Before touching cathode efficiency, a dedicated near-field/double-sided
   check of the sampled grid at the cathode plane (the anchors are maximally
   near-field for exactly this group).
4. The chi2/KS ladder + `flash_minPE` retune should follow whichever model is
   blessed (flash totals grew ×2-4 with the veto off).

## 4. Adopted calibration (owner decision, 2026-07-14)

The owner ruled: **these data are Argon-only — keep the 128 nm library**, run
the keep-and-mark saturation chain as the production signal processing, and
absorb the per-type residuals as data scale factors.  Applied as:

- **Toolkit `f7c66ab8`** (`cfg/.../protodunevd/qlmatching.jsonnet`): the
  official eff_Ar values are multiplied by the fitted per-type factors —
  cathode XA **x10.116** (0.03 → 0.3035), membrane XA **x1.655** (0.03 →
  0.0497), PMT **x0.352** (TPB 0.12 → 0.0422, PEN 0.036 → 0.0127; official
  relative weighting kept).  QtoL stays 0.094; Ar-blind zeros kept.  NOT
  byte-identical by construction (PDVD predictions change).
- **wcp `4d398d5`**: the keep-and-mark chain becomes the PDVD **runner
  default** — `PDVD_VETO_SATURATION=0`, `PDVD_FLAG_SATURATION=1`,
  `PDVD_SAT_REPAIR=1`, `PDVD_QL_USE_SAT_FLAG=1` (toolkit C++/jsonnet
  defaults stay legacy/OFF, byte-identical for other consumers).

**Closure** (repro: `python3 fit_qtol_crossers.py --tag arcal`): rerunning QL
on the 18 candle events (run 039252, `_satrep` light, tag `arcal`) and
refitting the strict-crosser anchors gives global median Σmeas/Σpred =
**1.011** [0.67, 3.54] n=42 and cathode **1.027** (was 10.1) — the dominant
group closes at the calibration point.  On this single-run subset membrane =
0.445 (n=24) and PMT = 0.704 (n=31) with wide bands — run-to-run spread
around the 3-run medians the factors were fitted on (the 128 nm far-channel
distance slope also remains, §2.2); per-channel work stays an open item.

**Candle round `arcal`** (`ql_display/decisions-*-arcal/`): find_crossers
280 keep/add + find_boundary 155 over the 18 events (vcal round: 198/127);
labels `work/ql_labels/candles-arcal/`.  The ql_scan viewer on **port 5019**
now serves this round (`serve_ql_scan.sh 5019 --tag candles-arcal
work/039252_*_arcal/calib-evt*.json`), showing the cathode-crosser pairs and
anode boundary tracks together, replacing the `candles-pull2c2` display
(those work dirs and decisions are untouched).

## 5. Files

- `ql_light_calib/fit_qtol_crossers.py` — harvest + estimator + per-type /
  per-brightness / library-swap diagnostics (this doc's tables).
- Dumps: `work/<run6>_<idx>_satrep/calib-evt*.json` (120; operating point
  veto-off + flag + twoside repair + QL per-flash masking).
- Fit logs: /home/xqian/tmp/satstudy/fit-relib{128,175}.log (scratch;
  regenerate via the Repro block).

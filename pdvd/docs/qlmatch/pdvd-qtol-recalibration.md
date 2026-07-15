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

## 5. Coverage-aware re-examination on the `spcov` round (2026-07-14, SP-fix reprocess)

**Repro:**
```
# 120-event light reprocess at the SP-fixed operating point (keep-and-mark +
# coverage + SPE templates v2 -- all run_light_evt.sh defaults):
cd pdvd && PDVD_MAX_JOBS=6 ./run_light_all.sh -s _spcov 39252   # + 039253, + 039349 x3 files
# QL rerun (imaging symlink farms from the plain dirs):
export OMP_NUM_THREADS=8 PDVD_MAX_JOBS=6 PDVD_LIGHT_SUFFIX=_spcov
./run_clus_evt.sh -s spcov -calib 039252 all   # + 039253, 039349
cd ql_light_calib && python3 fit_qtol_crossers.py --tag spcov
```

After the three light-pattern SP fixes
(`pdvd-lightpattern-sp-investigation.md`: unmasked dump prediction, per-flash
readout-coverage masking, SPE templates v2), the fitter now also drops
channels with dump `cov < 1` from both sums.  195 strict-crosser anchors
(154 both-halves; every anchor has ≥1 uncovered channel).

**Global / cathode: the adopted calibration closes.**  Global median
Σmeas/Σpred = **1.078** [0.77, 2.66] n=195 (per-run 1.057/1.088/1.070);
cathode-restricted **1.006** [0.77, 2.45].  Per-cathode-opdet bright-end
medians (pred ≥ 5 PE, covered+unsat) are 0.89–1.48 with ~0 zero-fraction,
and od7 — pinned at ~0.5 by the ch1051 template latch before v2 — now sits
at 1.14.  QtoL 0.094 and the cathode ×10.116 factor need no change.

**Membrane / PMT: the previous ×1.655 / ×0.352 factors were fit on ~70%
fake zeros, and honest coverage does NOT yield a replacement single factor.**
Anchor-level medians move to 3.85 (membrane) / 2.31 (PMT), but these carry a
self-trigger selection bias: a snippet channel is only covered when it
triggered, so at dim predictions the covered sample is exactly the
upward-fluctuating tail (per-channel ratio slope 10.8 → 2.4 from dim to
bright prediction).  At the unbiased bright end the picture inverts —
per-opdet medians over covered, unsaturated entries with pred ≥ 5 PE:

| od (membrane) | n | med m/p | frac meas=0 |
|---|---|---|---|
| 0 (repaired pair) | 83 | 1.43 | 0.42 |
| 1 | 65 | 0.00 | 0.55 |
| 3 | 61 | 0.00 | 0.54 |
| 12 | 60 | 0.00 | 0.65 |
| 18 | 60 | 0.00 | 0.55 |
| 19 | 71 | 0.00 | 0.66 |

i.e. 5 of 6 membrane walls measure NOTHING in most covered flashes against
12–25 PE predictions while other entries overshoot ×4–10; PMTs scatter
0.02–3.3 per tube.  No scalar rescale fixes a bimodal residual — this is the
128 nm-library wall-channel shape problem (§2.2 distance slope, §3 type
spread) now visible per channel with fake zeros removed.

**Recommendation (as reported):** keep the adopted f7c66ab8 factors as-is.
The cathode group (which closes at unity and dominates every fit) plus
coverage masking now carry the matching; the membrane/PMT factors only shape
chi2 on the minority of covered wall channels.  Refitting them from this data
would launder the selection bias into the calibration.  The per-channel
census (fit log /home/xqian/tmp/perod_spcov.log, regenerate per Repro) is the
starting point if the Xe/175 nm wall-channel question or per-channel
efficiencies are ever revisited.

**Owner ruling (2026-07-15): factors kept as-is.**  The f7c66ab8 values
(QtoL 0.094, cathode ×10.116, membrane ×1.655, PMT ×0.352) remain the
adopted calibration; no change to `qlmatching.jsonnet`.  In the same ruling
the owner accepted the `candles-spcov` hand-scan round as reasonable and
promoted the **spcov operating point to the PDVD production default**
(saturation keep-and-mark + repair, coverage flag chain, QL sat/cov masking,
**SPE templates v2**) — i.e. the runner env defaults already in
`run_light_evt.sh` (`PDVD_FLAG_SATURATION=1 PDVD_SAT_REPAIR=1
PDVD_EMIT_COVERAGE=1 PDVD_SPE_V2=1`, veto off) and `run_clus_evt.sh`
(`PDVD_QL_USE_SAT_FLAG=1 PDVD_QL_USE_COV_FLAG=1`) are ratified as the
standing operating point, and the `_spcov` dumps supersede `_satrep` as the
canonical 120-event record.  Toolkit C++/jsonnet defaults stay OFF
(byte-identical) as always.

**Candle round `spcov`** (`ql_display/decisions-*-spcov/`): find_crossers
204 keep + 76 add, candles union 222 keep + 203 add + 1911 reject over the
18 run-039252 events; labels `work/ql_labels/candles-spcov/`; port **5019**
now serves this round (`serve_ql_scan.sh 5019 --tag candles-spcov
work/039252_*_spcov/calib-evt*.json`), replacing the `candles-arcal` display
(arcal work dirs, decisions, and labels untouched).  The viewer shows sat
channels as orange rings (full prediction visible) and uncovered channels as
grey no-data markers.

## 6. Files

- `ql_light_calib/fit_qtol_crossers.py` — harvest + estimator + per-type /
  per-brightness / library-swap diagnostics (this doc's tables).
- Dumps: `work/<run6>_<idx>_spcov/calib-evt*.json` (120; CANONICAL since the
  2026-07-15 ruling: veto-off + flag + twoside repair + coverage chain + QL
  sat/cov masking + SPE v2).  Previous round
  `work/<run6>_<idx>_satrep/calib-evt*.json` (no coverage/SPE-v2) kept as
  record.
- Fit logs: /home/xqian/tmp/satstudy/fit-relib{128,175}.log (scratch;
  regenerate via the Repro block).

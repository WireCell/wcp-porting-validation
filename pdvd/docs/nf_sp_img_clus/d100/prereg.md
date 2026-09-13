# doc pdvd/100 — pre-registration (written 2026-09-13, before any refit arm ran)

Owner, 2026-09-13: flip `top_gain_scale=0.889` with the C refit and the Michel-threshold refits, gated against `p99wflip`.
Owner answers the same day: Michel thresholds = **the two data-sized knobs only** (`topology_michel_ke_min`, `plateau_mip_hi`);
QtoL = **measure, don't flip**. Scans need not be blind.

## Inputs fixed before this file
- Pre-refit candidate: `p99rwon` (gain ON, v7-uvwfit SP wires, real window, PR `-nu -stm-fit`, pin `libpin_p96`).
- Production (baseline and null): `p99wflip` (== `p99rwprod` 120/120, doc 99 §4.5).
- Knob gate G1 (`g1_compiled_config.txt`): toolkit `pr.jsonnet` `stm_recomb_C` (default 0.7941) and the driver's forward.
  Compiled PR config md5 before and after the edit at the default: 447e0f65 = 447e0f65. With `stm_recomb_C=0.8630` the
  diff is exactly one key, `pdvd_stm_recomb` `data.C` 0.7941 → 0.863.

## 1. C (measured before this file; `c_refit_*.txt`)
`d16_stm_energy_scales.py --det pdvd --chain-C 0.7941`:

| arm | fit C | chain reproduction (G1 python vs chain) | bottom / top ratio median at the fit | bottom/top |
|---|---|---|---|---|
| `p99wflip` (null) | 0.7923 ± 0.0051 (254 tracks) | 1.0017 | 1.1089 / 0.9716 | 1.141 |
| `p99rwon` | **0.8630 ± 0.0075** (259 tracks) | 1.0023 | 1.0389 / 0.9946 | 1.045 |

- Null: production's fit is 0.35σ from the chain's 0.7941.
- **C_new = 0.8630** (the `p99rwon` fit; the same value doc 99 fitted on `p98vonq`).

**The refit arm `p100c`** = PR only on `p99rwon`'s pctree with `-S stm_recomb_C=0.8630` (`d100_arms.sh WAVE=pr`).
Criteria, judged on `p100c` with `d16 --chain-C 0.8630`:
- **C1** the script reproduces the chain at 0.8630 (G1 python/chain median within 1.005 of 1);
- **C2** the fit on `p100c` is within 1σ (0.0075) of 0.8630. The STM set can move only through cuts in MeV (which carry C);
  every is_stm mover against `p99rwon` is named. A second C pass is NOT taken unless C2 fails (then once, and reported);
- **C3** bottom/top ratio medians at 0.8630 within 5 % (1.141 on production).

## 2. The two data-sized Michel thresholds
Sized in doc 89 §5 as the "(a1) bundle": two existing settings that clear missed stoppers on the record with **no false
stopper**; flipped in doc 90 (`topology_michel_ke_min` 10 → 3.0 MeV, `plateau_mip_hi` 1.6 → 2.0). Re-run of that rule on
the flipped scale:

- **Population:** every STM candidate of `p100c` (prep with `--sheetdir` in scratch), graded on the latest carried record
  (`pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json`) carried from `p98vonq` onto `p99rwon` keys with `d99_match` (its
  pre-registered thresholds; `ok` matches only), plus the owner's `own100` verdicts if folded by then (named if used).
- **Twin:** a fork of `d90_twin.py` with the record and the values as arguments. It must first reproduce `p100c` itself
  (0 mismatches at 3.0 / 2.0) before predicting anything.
- **Grid, fixed here:** `topology_michel_ke_min` ∈ {2.0, 2.5, 3.0, 3.5, 4.0, 5.0} MeV × `plateau_mip_hi` ∈ {1.6, 1.8,
  2.0, 2.25, 2.5}.
  - The old-scale equivalents are inside it: top plateau reads ×1.125, so 2.0 → 2.25 on top; top Michel energy ×1.125 /
    1.087 (C) ≈ ×1.035, bottom ×0.920, so 3.0 stays 2.8–3.1.
- **Rule (doc 89's):** keep 3.0 / 2.0 unless another grid point gives **more** judged stoppers accepted with **no more**
  judged non-stoppers accepted than 3.0 / 2.0. Among such points take the one closest to 3.0 / 2.0 (fewest steps). A tie
  keeps the current values. The twin's plateau predictions are a lower bound (doc 90: the geometric re-read is not
  recorded), so a changed value is measured exactly by a PR-only arm `p100thr` before adoption, and adopted only if that
  arm confirms the twin's count.
- **Success criterion for the round** (the doc 99 §6 symptom: Michel purity 0.831 → 0.785 OFF → ON on the carried
  record): on the final arm, Michel purity on the carried record is within 1.5σ (binomial) of `p99wflip`'s on its own
  carried record, and the top plateau/MIP centering is as doc 99 §7 (top plateau closer to bottom's). If purity does not
  recover, the "thresholds fitted on the old scale" reading is wrong: reported, not retuned.

## 3. QtoL: measured only
`pdvd/ql_light_calib/fit_qtol_crossers.py --tag p99rwon` and `--tag p99wflip`, plus a per-half fork (top half vs bottom
half meas/pred). Prediction: the top-half ratio ON/OFF ≈ 1/1.125, the bottom half unchanged. Production keeps 0.094.

## 4. The flip gate
Production defaults changed to the adopted values (SP entry TLA default 0.889; production imaging tag `pvdimg`, hard links
of `p98von`'s imaging; driver `stm_recomb_C`; thresholds if changed). `d100_arms.sh WAVE=flip ARM=p100flip` (no
overrides) must equal the final measured arm on **120/120** events in pctree, tlas, every PR branch and `mabc-pr`
(`d99rw_identity.py`). Its clustering must equal `p99rwon`'s (the table window == the env window on the gain-ON side).

## 5. The owner's edge scan (`own100`, 19 objects, not blind)
Reported: stoppers vs non-stoppers among the objects the readout-edge guard removed, by arm and volume. The guard is not
changed in this round.

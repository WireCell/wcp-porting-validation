# doc 32 pre-registration, round 1 (written 2026-09-23 before any sheet was rendered or scanned)

Goal (owner): make the ToT light viable and flip it. The existing owner/AI scan record (`wfresc` gold + `decisions-cathxa`)
was taken on twoside light, so on cathode-railed flashes it is not a neutral judge of a light change. This round
builds a same-scanner record on the clusters whose matching the ToT light changes.

## Arms
- `q31ctl`: production Q/L on light `_g31off` (twoside, short hits; hash-identical to production `_keep`).
- `q32ti`: identical Q/L on light `_q32ti` = `PDVD_SAT_REPAIR_MODE=tot PDVD_HIT_INT_SAMPLES=1` (ToT fill + the
  OpHitFinder int_samples fix).
- Clustering pin `/home/xqian/tmp/p100/libpin_p100b` for both; cluster uids are identical between the arms.

## Items (`scripts/d32_scan_items.py`, seed 32)
- **Movers**: long clusters (the scorer's filter) whose auto-selected flash set differs between the arms after the
  doc 31 time map (`d32/time_map_ctl_to_q32ti.json`) translates control times. Count on a dry run: 100. Only
  movers can change the comparison. Each mover is rendered in both lights (200 sheets).
- **Calibration**: 40 non-mover clusters with owner verdicts, 20 on cathode-railed flashes and 20 on unrailed ones,
  rendered in the control light (the light the owner saw).
- **Duplicates**: 20 mover sheets re-rendered under new ids.
- 260 sheets, shuffled, split into 5 waves of 52 (seeded). One general-purpose scanner agent per wave.

## Blinding
- Sheets show geometry, measured PE, this cluster's predicted PE and railed marks.
- They hide auto marks, ks, chi2, strength, flags, gid, uid and light name.
- The key lives in `/home/xqian/tmp/p32/scan_key/`, which scanners are told never to read.
- An audit of each scanner transcript discards any scanner that read the key, calib dumps, scores, labels or
  decisions.
- Known leak: ToT railed channels read higher.

## Verdict per sheet
- The scanner picks one letter (the flash this cluster produced), `none` (no candidate is its flash) or `unsure`,
  with confidence high/med/low.
- Low confidence counts as `unsure`, matching the scorer's objective tiers.

## Truth for a mover (light-independent, "consensus")
- **Positive.** Both lights' sheets pick a flash, and the two picks are the same physical flash: the ctl pick's time
  maps (time map, or equal within 0.5 µs) to the ToT pick's time within 0.5 µs. Then the truth is positive at each
  light's own pick time, and negative at every other drawn candidate.
- **None.** Both sheets say `none`: negatives at every drawn candidate in both lights.
- **Anything else** (disagreement, or an `unsure`): unresolved. The owner entries for that cluster are dropped in both
  arms, so it scores alike in both. Count reported.
- **Scoring.** For each arm, the owner record, with all owner entries of scanned mover clusters replaced by the
  consensus truth, goes through `ql_agree_score.py --override-truth` (new option, default off = byte-identical).
- **Secondary, reported.** Per-light truth: each arm scored with its own light's verdicts.

## Decision rule for this round (a flip additionally needs the owner's go)
- **Scanner calibration.** Agreement with the owner on calibration sheets must be ≥ 85 %.
  - Agreement: the scanner's pick equals the owner's positive flash, or the scanner says `none` when the owner has only
    negatives among the drawn candidates.
  - `unsure` is excluded and counted.
  - Duplicate-pair consistency is reported.
- **Doc 23 rule on the consensus-truth scores.** `q32ti` vs `q31ctl`: at least one of agree / phantom / missed
  improves and none worsens, with a sign test on the paired movers. A pass carried by movers with p > 0.2 is stated as
  "not separable from churn".
- **Split-half.** Odd and even events are reported separately. Round 1 tunes nothing, so there is no hold-out yet;
  the Q/L ladder of later rounds tunes on odd events and confirms on even.
- **If the scan does not pass.** Loss-stage attribution (`d32/loss_stage.txt`) names the Q/L knobs for the round-2
  ladder, with values taken from the movers' margins.

## Additions before rendering (same date, after the design review; still before any sheet existed)
- **Railed channels.** The rubric (`d32/RUBRIC.md`) makes railed (R) channels a sanity check only, identically for
  both lights. The verdict rests on geometry at the T0 plus the pattern and amplitude on the unrailed channels. Each
  sheet prints pred/meas on unrailed channels separately. This keeps the truth light-independent. ToT's gain has to
  show up as Q/L choosing the right flash, not as the scanner preferring ToT's R heights.
- **Waves.** The two light sheets of one mover always go to different scanners. A duplicate goes to a third scanner
  (neither of its pair's). Duplicates feed only the consistency number, never the truth.
- **Calibration** is reported separately by verdict source:
  - `gold`: the owner's own scan, evt298567, 6 items per class forced;
  - `cathxa`: the earlier AI scan with owner review, on the unblinded display with saturated channels excluded.

  Agreement is counted only on candidates the owner record judged. A scanner pick of an unjudged candidate is "not
  comparable" and is counted separately. The ≥ 85 % bar applies to gold + cathxa pooled, with gold also reported
  alone.
- **Minimum resolution.** The round's consensus-truth verdict counts only if at least 60 % of the 100 movers reach
  consensus (positive or none). Below that, the round reports "scan too noisy to judge" and no flip reading is taken.

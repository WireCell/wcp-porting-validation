# doc 33 pre-registration, round 2 (written 2026-09-23 after the lever arms ran, BEFORE any sheet was rendered or any arm was scored against a scan)

**Goal (owner):** make the ToT light viable and flip it. The owner does not scan (short tracks cannot be judged by
hand; geometry + light pattern decide), so an AI blind scan is the tuning target.

## Frozen arm set (the target covers exactly these; a later arm needs a labelled top-up scan)
All arms run on the clustering pin `/home/xqian/tmp/p100/libpin_p100b` with `libWireCellMatch` from the pin named.

| arm | light | Q/L lever | pin | role |
|---|---|---|---|---|
| `q31ctl` | `_g31off` (production twoside, short hits) | production | p100b | **control** |
| `q32ti` | `_q32ti` (ToT + int_samples) | production | p100b | L0 ToT baseline |
| `q33ts` / `q33cs` | ToT / control | L1 `sat_skip_round2_shared` | q33 | ToT arm / diagnostic twin |
| `q33tm` / `q33cm` | ToT / control | L2 `saturation_mask_fit=true` (`PDVD_QL_SAT_MASK_FIT=1`) | q33 | ToT arm / diagnostic twin |
| `q33tu` / `q33cu` | ToT / control | L3 `lasso_weight_unrailed` | q33b | ToT arm / diagnostic twin |

Knob-off gates: `g33pin` (q33 vs q31ctl 120/120 calib-identical), `g33bpin` (q33b likewise).

No combo arm: L2 alone already reduces ToT-vs-control movers from 100 to 40 (L3 only to 96), so an L2+L3 arm would
add little; it is left for a top-up if L2 is the candidate.

Mover counts vs `q31ctl` before scanning (for sizing only; no truth involved): L0 100, L1 102 (twin 2), L2 252
(twin 250), L3 125 (twin 48); union 329 (odd events 155, even 174).

## Population and sizing
- Long clusters (the scorer's filter, >= 25 cm or >= 100 points) of the 18 run-039252 events whose auto flash set
  differs from `q31ctl`'s in ANY frozen arm (mover test in the arm's own frame, control picks mapped forward by
  `d32/time_map_ctl_to_q32ti.json` for ToT-light arms; identity for control-light arms).
- Cap 360 movers (the union, 329, is scanned whole). Above the cap, strata in arm-list order would be kept whole
  while they fit and the first that does not fit sampled (seed 33).

## Sheets (light-neutral, `scripts/d33_blind_sheet.py`)
- Drawn from the control dump. Railed channels: hatched bar, no value. Unrailed PE, the prediction, and the cluster's
  own group are (near-)identical between lights: `d33/neutrality.txt` quantifies it; the residual differences
  (pred <= 6 % from the flash-time shift; unrailed PE > 1 % different on ~6 % of bundles and sat flags different on
  ~2 %, from hits changing flash membership) are the listed leaks.
- Candidates <= 5: any frozen arm's pick (control frame), the owner's objective verdict flashes, then |log pred/meas|.
  Letters shuffled per look.
- Stacked prediction: co-matched clusters = those auto-selected on that physical flash in the control AND every
  frozen arm (arm-neutral intersection), plus this cluster.
- Rubric `d33/RUBRIC.md`; answers letter | `tie:X,Y` | none | unsure; low confidence counts as unsure.

## Scanners
- 9 primary waves (agents 1-9), each cluster seen by 2 different scanners (independent shuffles); 40 owner-judged
  calibration clusters (same selection rule as round 1, seed 33) one look each; 5 % of mover looks duplicated to a
  third scanner (consistency only).
- Agent 10: adjudication of committed contradictions (a third look, fresh shuffle, new id).
- Verdicts only via `scripts/d33_record.py`; every transcript audited (`d32_audit.py`, adapted to the d33 recorder).

## Target (merge rule, `scripts/d33_merge_record.py`)
- Resolved when >= 1 look is committed (med/high) and no two committed looks contradict. Compatible: same letter's
  flash; a pick inside a tie set (-> the pick); identical tie sets; none & none.
- Contradiction: the adjudication look decides if it is compatible with exactly one side; else unresolved.
- Ties: resolved-NEUTRAL (dropped from the paired score), rate reported separately.

## Scoring and decision (`scripts/d33_paired_score.py`)
- PRIMARY: per resolved non-tie mover, an arm is right iff its auto set is exactly the truth flash (or empty for
  `none`). Paired wins/losses of each ToT arm vs `q31ctl`, two-sided sign test.
- SECONDARY: agree/phantom/missed on the resolved movers; and `ql_agree_score.py --override-truth` (owner record with
  the target replacing every scanned mover; unresolved/tie dropped).
- **Tune on odd events** (idx 1, 3, ..., 17): the candidate is the ToT arm with the largest net wins vs `q31ctl` on odd
  events (ties broken by fewer phantoms).
- **Confirm on even events** (idx 0, 2, ..., 16). The candidate PASSES only if ALL hold:
  1. calibration >= 85 % (committed verdicts on owner-judged candidates, gold + cathxa pooled);
  2. resolution >= 60 % of movers (ties count as resolved, reported separately);
  3. power floor: the even half has >= 10 discordant resolved movers between the candidate and `q31ctl`; below it the
     answer is "not separable";
  4. even-half net wins >= +1 with sign-test p < 0.05, AND no worsening of agree/phantom/missed on even events
     (doc 23 rule: one improves, none worsens);
  5. the owner-record secondary does not worsen by more than the round-1 churn (reported).
- **Governing comparison: candidate (ToT + lever) vs `q31ctl`.** The same lever on control light is a diagnostic. If
  the control twin scores >= the candidate on the even half, the pre-registered reading is "offer the lever for
  production on its own; it is not ToT evidence".
- A PASS is presented to the owner for the flip go; no default changes in this round. A FAIL of (2) or (3) is
  "target too thin", not a verdict on ToT.

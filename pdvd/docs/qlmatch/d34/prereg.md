# doc 34 pre-registration, round 3

**Part 1** was written 2026-09-23, before step A was run, before any round-3 arm ran and before any round-3 sheet
existed. **Part 2** (the frozen arm values) is appended after step A. The SHA of each part is kept in
`prereg_sha.txt`.

**Goal (owner).** Make the ToT light viable and flip it. Q/L need only **not be worse** than production.

## Part 1 — rules

### Decision margin (owner-accepted 2026-09-23)
**PASS.** A ToT candidate passes when **both** of these hold:
1. On the **held-out even half** (events idx 0, 2, …, 16), the one-sided 95 % lower bound of the paired net (wins −
   losses vs `q31ctl`, resolved non-tie movers) is **≥ −6.46 clusters**.
   - The bound is net − 1.645·√(W+L).
   - 6.46 = 2 % of the 323 owner-confirmed matched long clusters in those 9 events.
2. **Phantoms** of the candidate on the owner+target record (`ql_agree_score.py --override-truth`) are **≤ those of
   `q31ctl`**.

**Also reported.** Superiority (a two-sided sign test) and the control-twin read-out (whether the gain is
ToT-specific).

**Truth for scoring.** The round-2 target (`d33/scan_r2/truth.jsonl`, unchanged) ∪ the round-3 top-up
(`d34/scan_r3/truth.jsonl`).

**Disclosure.** The round-2 even half is partly spent:
- L3 (`lasso_weight_unrailed`) became the base of every round-3 arm because of its even-half result in doc 33 §7.
- Post hoc, `q33tu` already meets this margin: bound −6.39, phantoms 154 vs 163. That pass is by 0.06 of a cluster on
  data already read.
- The **fresh** part of the even half is the top-up movers. They are reported separately as well as in the union.

### Step A — rail measurement (`scripts/d34_rail_calib.py`)
**Anchors.** Strict cathode-crosser anchors, using the `fit_qtol_crossers.py` harvest, from the 18 run-039252 dumps
of `q32ti` (ToT) and `q31ctl` (production).
- These anchors are geometric truth, independent of the scan verdicts.
- The events overlap the scan target; this is stated, not hidden.

**Per anchor.**
- s = Σmeas/Σpred over the kept unrailed channels.
- For each railed channel with pred > 0.5 PE: r = (meas/pred)/s.

**Tolerance.** f = exp(q84 of |ln r|) − 1 on the **ToT** arm.
- This needs ≥ 30 railed samples.
- Below 30, f = 1.0 (a prior: the ×2 trace scale of doc 33).

**QtoL trigger.**
- Quantity: the rail-inclusive global median Σmeas/Σpred ÷ the rail-excluded one, on the ToT anchors.
- If it differs from 1 by more than 10 %, one extra arm pair runs with QtoL = 0.094 × that ratio (`PDVD_QTOL`, on top
  of the ladder's middle rung).
- Otherwise there is no QtoL arm.

### Arms (values fixed in Part 2 by the rule above)
**Common settings.** Every arm uses:
- the clustering pin `libpin_p100b` + `libWireCellMatch` from `libpin_q34`;
- `lasso_weight_unrailed` (L3) on.

**Ladder.** `ks_sat_tol` ∈ {f/2, f, 2f}.
- ToT arms `q34tk1`, `q34tk2`, `q34tk3`, on `_q32ti` light.
- Control twins `q34ck1`, `q34ck2`, `q34ck3`, on `_g31off` light. They are diagnostic only.

**Frozen references.** `q31ctl` (control) and `q33tu` (L3, no tolerance; **the reference, not a rung**).

### Top-up target
**Population.** Long-cluster movers vs `q31ctl` in any round-3 arm, excluding the 329 round-2 movers. Cap 300; above
the cap, sample by arm order (seed 34).

**Procedure: the same as round 2.**
- Light-neutral sheets drawn from the control dump.
- Co-matched intersection over round-2 plus round-3 arms.
- Shuffled letters, ≤ 5 candidates.
- 2 looks from different scanners, 5 % duplicates.
- The round-2 calibration rule (seed 34) for owner-judged calibration clusters.
- `d33/RUBRIC.md` unchanged, the `d33_record.py` recorder, `d33_audit.py` on every transcript.
- `d33_merge_record.py` (the same merge rule), with a `d33_adjudicate_items.py` wave for contradictions.

**Bars for the top-up.**
- Calibration ≥ 85 %.
- Resolution ≥ 60 %.
- Clean audit.

A failed bar reads "top-up too thin", and the union score is then reported on the round-2 truth alone.

### Selection and confirmation
**Selection** on the **odd** half (idx 1, 3, …, 17):
- The ToT ladder rung with the largest net vs `q31ctl` is chosen, **provided it exceeds `q33tu`'s odd net**.
- Ties are broken by fewer phantoms on the resolved movers, then by the smaller tolerance.
- If no rung exceeds `q33tu` on odd events, the outcome is **"no improvement over L3"**. `q33tu` then remains the
  candidate, with its margin pass stated as post hoc.

**Confirmation** on the **even** half: the margin above.
- A pass is presented to the owner as the flip candidate, together with the pre-flip list: doc 11 §6 / doc 12 rerun,
  the 102 other events, a flip-equivalence gate, and the owner's go.
- No default changes in this round.

## Part 2 — frozen values (appended after step A, before any round-3 arm ran)
Source: `d34/rail_calib.txt`, 49 ToT anchors with 164 railed samples (≥ 30).

**ToT rails vs the prediction.** Median r = 1.040 [16 % 0.690, 84 % 1.354], and q84 of |ln r| = 0.479.
- **f = 0.615.**
- The ladder is `ks_sat_tol` = **0.3075 / 0.615 / 1.23**: arms `q34tk1` / `q34tk2` / `q34tk3`, with twins
  `q34ck1` / `q34ck2` / `q34ck3`.

**QtoL trigger.** Rail-inclusive / rail-excluded = 1.082. That is within 10 %, so there is **no QtoL arm**.

**Comparison, not used.** Production rails (`q31ctl`, 158 samples) have median r 0.920 and f would be 1.005. At the
brightest quintile (pred > 13 k PE) they read r = 0.550.

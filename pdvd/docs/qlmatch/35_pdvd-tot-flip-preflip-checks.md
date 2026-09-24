# PDVD ToT flip: the pre-flip checks — C1/C2 pass, C3 passes on Q/L, C5 transfers exactly, the STM gate is undecided, NOT flipped

**Status 2026-09-23. NOT FLIPPED.** The owner's go was conditional on the pre-flip checks (`d35/prereg.md`,
sha in `d35/prereg_sha.txt`, written before any doc-35 arm ran). C1 and C2 pass their bars. C3 passes on Q/L, though its
PR sub-clause fired literally on one zero-candidate event (§4.2). C5, which carried no bar, shows the production binary
transfers exactly. The STM gate (C4) is **UNDECIDED on both purities**. Its point values are inside the bar, but the ToT arm brings 33 STM candidates that no record has
labelled, against 10 for production. The pre-registered rule makes UNDECIDED a STOP, so no runner default was changed.
The owner's choice of how to label those items is in §6.

**Owner request (2026-09-23):** "Can you proceed as you recommended? After confirmation, please update the production
default for PDVD, please update the relevant md file, commit and push." The recommendation was doc 34 §7: rerun doc 11
§6 and doc 12 on `q34tk1`, check the other 102 events for crashes and match-rate tails, run a flip-equivalence gate,
then change the default. Following the doc 29 §7 precedent for a PDVD Q/L flip, the 120-event STM gate on the
production binary was added (§2, C4). That gate goes beyond the four steps recommended in doc 34, and it is the one
that stopped the flip.

**In one screen.**

| check | what | result |
|---|---|---|
| C1 | doc 11 §6: railed terms on selected matches, 120 events | **PASS**, but close. ToT rails read 1.91× the prediction (bar ≤ 2.0; production's clipped rails read 1.57) |
| C2 | doc 12: crosser closure, all three runs | **PASS**. Rail-excluded ratio to production 0.979 (1.000 on the 189 common anchors); rail-inclusive/excluded 1.038 (bar ≤ 1.10) |
| C3 | the 102 unscanned events: crashes and match-rate tails | **PASS on the Q/L bars**: the unscanned runs move like the scanned one (mover fraction 0.126 vs 0.127). The registered PR sub-clause fired literally on one zero-candidate event, not a crash (§4.2) |
| C4 | STM gate: `is_stm` / Michel purity and efficiency vs production | **UNDECIDED → STOP.** Point values −0.004 / −0.009 / −0.010 / −0.009 (bar −0.020); NEG bounds on the purities −0.043 / −0.042 |
| C5 | does the doc 34 margin transfer to the production binary? (report only, no bar) | **Exactly.** Production-binary calib dumps are byte-identical to the measured arms, 120 / 120, for both production and ToT |
| F1 | flip equivalence | **Not run**: the flip was stopped before the edit |

**Repro:**
```
# pin: /home/xqian/tmp/p35/libpin_prod = cp -a local/lib (2026-09-23; md5 list d35/libpin_prod.md5, Clus 129461a1991e,
#      Match f3cfd9b19072 = the ks_sat_tol lib of libpin_q34)
cd pdvd/docs/qlmatch/scripts
ARM=q35ctl LIGHT=_keep  CLUS_JOBS=6 PR_JOBS=5 ./d35_arms.sh                                                     # production
ARM=q35tk  LIGHT=_q32ti CLUS_JOBS=6 PR_JOBS=5 ENVS="PDVD_QL_LASSO_W_UNRAILED=1 PDVD_QL_KS_SAT_TOL=0.3075" ./d35_arms.sh   # ToT
cd ../../.. && W=work && for t in q31ctl q34tk1; do python3 docs/qlmatch/scripts/d31_sat_terms.py $t \
    $W/0392{52,53}_*_$t/calib-evt*.json $W/039349_*_$t/calib-evt*.json; done                 # C1 -> d35/c1_sat_terms.txt
cd docs/qlmatch/scripts
python3 d35_crossers.py --cand q34tk1 --ctl q31ctl > ../d35/c2_crossers.txt                                      # C2
python3 d35_match_tails.py --ctl q31ctl --cand q34tk1 > ../d35/c3_tails_pinned.txt                                # C3
python3 d35_match_tails.py --ctl q35ctl --cand q35tk --pr > ../d35/c3_tails_prod_literal.txt
python3 d35_match_tails.py --ctl q35ctl --cand q35tk --pr --zero-candidate-ok > ../d35/c3_tails_prod_zero_ok.txt
(cd ../../nf_sp_img_clus/scripts && for t in T_stm_michel T_rec_charge; do
   python3 d103_flip_gate.py --det pdvd --a d116vflip --b q35ctl --tree $t > ../../qlmatch/d35/c4_control_vs_d116vflip_$t.txt; done)
(cd ../../nf_sp_img_clus/scripts && python3 d116_grade.py --det pdvd --cells A0=q35ctl,T=q35tk,P=d116vflip \
   --extra-record ../../scan/pdvd_stm_michel_smx116_verdicts.json --owner-record ../../scan/pdvd_stm_michel_own116v_verdicts.json \
   --split-half --movers-out /home/xqian/tmp/p35/c4_movers) > ../d35/c4_grade_uncarried.txt                   # C4, as registered
python3 d35_stm_grade.py --a0 q35ctl --t q35tk --unlabelled-out ../d35/c4_unlabelled.tsv > ../d35/c4_grade.txt   # C4, carried
python3 d35_stm_grade.py --a0 q35ctl --t d116vflip > ../d35/c4_grade_carry_null.txt                              # carry null
# C5: cmp of calib dumps q35ctl vs q31ctl and q35tk vs q34tk1 over pdvd/stm/events.txt -> d35/c5_transfer.txt
```

## 1. What was checked
`d35/prereg.md` fixed the arms, every bar, and what a STOP means before anything ran. Two arms ran on the production
binary (a private copy of `local/lib`), both clustering + PR (`-calib -save-pctree`, then `-nu -stm-fit`), 120 / 120
events each, under `setarch -R`:
- **`q35ctl`**: production today (`pvdimg` staging, `_keep` light, no Q/L env);
- **`q35tk`**: the doc 34 candidate (`_q32ti` ToT light, `PDVD_QL_LASSO_W_UNRAILED=1`, `PDVD_QL_KS_SAT_TOL=0.3075`).

Checks C1–C3 read the measured pinned arms `q31ctl` / `q34tk1` (120 events each). C5 shows that those arms and the
production-binary arms are the same thing.

## 2. Results

### C1 — doc 11 §6 (`d35/c1_sat_terms.txt`)
Railed-channel chi2 terms on selected matches, pooled over 120 events:

| | production `q31ctl` | ToT `q34tk1` |
|---|---|---|
| railed terms on selected matches | 8997 | 9839 |
| median meas / pred | 1.57 | **1.91** (bar [0.5, 2.0]: PASS) |
| meas > pred | 68.9 % | 76.5 % |
| chi2 at `chi2_sat_inflate` 0.5: median / p90 / max | 0.63 / 3.24 / 4.00 | 0.77 / 3.32 / 4.00 |

- **ToT lifts the railed measurement toward the truth.** The raw ratio is above 1 in both lights because a railed
  flash is usually shared: the flash holds other clusters' light and the bundle predicts only its own. On geometric
  crossers, where the flash is one object, the ToT rails agree with the prediction (doc 34 §2).
- The 0.5 inflation still bounds every term at 4.0.
- The pass is close to its bar (1.91 vs 2.0). The bar measures the shared-flash effect as much as the rails.

### C2 — doc 12 (`d35/c2_crossers.txt`)
Strict crosser anchors on all three runs:

| | production `q31ctl` | ToT `q34tk1` |
|---|---|---|
| anchors | 192 (doc 29's 0.833 reproduced) | 215 |
| rail-excluded global median Σmeas/Σpred | 0.833 | 0.816 (ratio 0.979: PASS) |
| rail-inclusive / rail-excluded | 0.844 (clipped rails pull it down) | **1.038** (bar ≤ 1.10: PASS, no QtoL change) |
| railed r, median [16, 84 %] | 0.917 [0.506, 1.295] | 1.024 [0.684, 1.368] |

- **The arms harvested different anchor sets.** On the 189 anchors present in both (same event, flash within 1.5 µs)
  the rail-excluded ratio is **1.000**. The 26 ToT-only anchors are bright flashes (median 51k PE), so the 1500 PE
  harvest cut is not why they appear. What admits them is not traced.
- **The tolerance ladder may sit a little low.** On 120 events the doc 34 read-out gives f = 0.760, not 18-event
  0.615. The ladder's rungs came from 0.615, and the winning rung was already its lowest (0.3075), so this changes
  nothing selected.

### C3 — the 102 unscanned events (`d35/c3_tails_pinned.txt`, `c3_tails_prod_*.txt`)
Reading no truth. A mover is a long cluster whose auto-selected flash set moves by more than 1.5 µs; on the scanned run
this definition gives 162 against doc 34's time-mapped 161.

| | run 039252 (scanned, 18 events) | runs 039253 + 039349 (102 events) |
|---|---|---|
| matched long clusters, production → ToT | 1185 → 1189 (+0.34 %) | 4540 → 4581 (+0.90 %) |
| movers / long clusters | 162 / 1272 = **0.127** | 618 / 4894 = **0.126** (bar ≤ 0.20: PASS) |
| events keeping < 75 % of their matches | — | 0 |

- **No crash.** All 120 events completed clustering, Q/L, pctree and PR in both arms.
- **One flag, by rule not a stop.** In 039349_41, 11 of 35 long clusters move (0.314) while its matches rise 29 → 31.
- **The unscanned runs behave like the scanned one,** so the doc 34 scan covers them as well as it covers itself.
- The prereg quoted the scanned run's "pooled 0.137". That figure used matched clusters as the denominator; over long
  clusters it is 0.127, as the tables here print. The bar (0.20) is unaffected.

### C4 — the STM gate (`d35/c4_grade.txt`; as registered `c4_grade_uncarried.txt`)
The rule is doc 113's, with records own116v > own103v2 > own103v > p99rwon-carried-corrected > smx11 > smx116. Each of
the four metrics must be ≥ −0.020 vs production in both the NEG and POS scenarios for the unlabelled candidates.

| metric | production `q35ctl` | ToT `q35tk` | point Δ | NEG | POS | reading |
|---|---|---|---|---|---|---|
| `is_stm` purity | 0.982 (273 / 5) | 0.978 (270 / 6) | −0.004 | **−0.043** | −0.003 | UNDECIDED |
| `is_stm` efficiency | 0.776 | 0.767 | −0.009 | −0.009 | +0.023 | PASS |
| Michel purity | 0.942 (180 / 11) | 0.932 (178 / 13) | −0.010 | **−0.042** | −0.008 | UNDECIDED |
| Michel efficiency | 0.779 | 0.771 | −0.009 | −0.009 | +0.019 | PASS |

- **Split halves** (reported): the worst is −0.026, on Michel efficiency in the odd half; doc 116's guard is −0.040.
- **Sensitivity** (reading 2, the 23 uncarried keys dropped from both cells): the same picture, with purity NEG
  −0.046 / −0.042.
- **Every point value is inside the bar.** The NEG bounds fail because the ToT arm has **33 candidates with no label
  in any record** (17 tagged `is_stm`, 10 with a Michel) against production's 10 (5 and 3). NEG counts every one of
  them as a false positive.
- **Most are not relabellings.** Only 3 of the 33 are the geometric images of a labelled cluster that failed to carry.
  The other ~30 are clusters that became STM candidates only under ToT Q/L: a changed flash (t0) moves a cluster in x,
  and matching-driven merges change what the tagger sees.
- The item list for any labelling round is `d35/c4_unlabelled.tsv` (43 rows: 33 ToT, 10 production).

### C5 — binary transfer (`d35/c5_transfer.txt`)
`q35ctl` vs `q31ctl` and `q35tk` vs `q34tk1`: calib dumps byte-identical on **120 / 120** each.
- The Q/L stage of today's production binary is the pinned binary of docs 31–34, whatever changed in Clus, Img or Root
  since.
- **The doc 34 hand-scan margin therefore applies to production as it is.** The movers on run 039252 are the same 161.

## 3. The production reference moved since doc 116 (reported before the grade)
Fresh production `q35ctl` differs from doc 116's graded production `d116vflip`:
- in 21 of 119 events on `T_stm_michel`, and in 31 on `T_rec_charge`;
- because `d116vflip`'s PR reused doc 103-era clustering (`SRC=d103vflip` pctrees), while `q35ctl` re-clusters with
  today's Q/L defaults, including doc 29's boundary weight.

The four metrics still come out identical in the uncarried grade (`P` cell of `c4_grade_uncarried.txt`), and within
±0.006 in the carried null (§4.1). So the records apply to `q35ctl` as they applied to `d116vflip`.

## 4. Disclosures (what was decided after the data were seen)

### 4.1 The carry
The command the prereg named, `d116_grade.py --cells A0=q35ctl,T=q35tk`, turned out to be structurally invalid. The ToT
flash association renumbers the PR cluster ids, so that grade compares different clusters under one key:
- 228 unlabelled ToT candidates and 251 production-only ones;
- efficiency 0.776 → 0.388 (`c4_grade_uncarried.txt`, kept).

Doc 29 met the same thing and carried the record by geometry. `d35_stm_grade.py` does that inside the grade:
- it uses `d100_carry_verdicts.one()` and doc 99's pre-registered match thresholds, unchanged;
- 865 of 888 keys carry;
- 16 fail as merged, 6 as lost and 1 as split, so the Q/L change did alter matching-driven merges;
- it then calls d116's `reading()` unchanged.

**Null control**, production vs `d116vflip` through the same carry (`c4_grade_carry_null.txt`): 887 / 888 keys carry,
deltas are within ±0.006, and all four metrics PASS in both scenarios.

The carry is a post-hoc repair of the gate's plumbing, not of its rule, and both grades are kept. Either way the
outcome is a STOP.

### 4.2 The C3 PR sub-clause
As registered, the clause requires the CheckSTM candidate line in every ToT PR log where production has one. It fires
on **039349_55**: production has one candidate there, which fails every check; ToT has none
(`c3_tails_prod_literal.txt`: STOP).

The PR itself completed (rc 0, all outputs, "PR done") and logged CheckSTM_Michel's zero-candidate end line, "no
STM-tagged main cluster; nothing to reconstruct". The mirror case is in production, at 039252_17 (0 candidates; ToT 2).
The clause was written for crashes and did not foresee the zero-candidate line. Counting that line as a completed
CheckSTM gives PASS (`c3_tails_prod_zero_ok.txt`). Both outputs are kept. This does not change the outcome, which C4
already stops.

## 5. The flip that was prepared and not made
Nothing in `run_light_evt.sh`, `run_clus_evt.sh` or `stm/run_campaign.sh` was edited. The unit is fixed in
`d35/prereg.md` §0; after a pass it goes in as follows:
- **light:** `PDVD_SAT_REPAIR_MODE` defaults to `tot`, only while `PDVD_SAT_REPAIR=1` and `PDVD_SPE_V2=1`, so that the
  legacy-chain escapes still compile. `PDVD_HIT_INT_SAMPLES` follows the mode, and twoside + int stays explicit, with a
  warning;
- **Q/L:** `PDVD_QL_LASSO_W_UNRAILED` defaults to 1 and `PDVD_QL_KS_SAT_TOL` to 0.3075;
- **light record:** the campaign's light suffix changes from `_keep` to a new `_tot` record;
- **guard:** a stderr warning when the ToT-tuned Q/L defaults meet non-ToT light, because that combination is the
  control twin `q34ck1`, not production;
- **F1 gates:** the compiled-config proof (`scripts/d35_flip_compiled.sh`; its pre-flip half was taken, with compiles
  in `/home/xqian/tmp/p35/cfg/`), the light gate `_tot` == `_q32ti` (`scripts/d35_light_f1.py`, whose comparator was
  checked on the null pair `_g31offb`/`_g31off`, PASS 120 / 120, and on a negative, FAIL 0 / 120), and the runtime gate
  `q35flip` == `q35tk` (`scripts/d35_f1_check.sh`, with the `qp.ks_sat_tol` deploy tell).
- **The 2026-09-23 F1 pre-half is scratch.** The compiles in `/home/xqian/tmp/p35/cfg/` are tied to today's runner,
  and `~/tmp` gets swept. Re-run `d35_flip_compiled.sh pre` immediately before the edit. Likewise, F1(c) compares
  against arms run on `/home/xqian/tmp/p35/libpin_prod`: check that pin still exists and matches
  `d35/libpin_prod.md5`, or re-run both arms on a fresh pin.

## 6. The owner's decision
The flip now waits only on labels for 43 items (`d35/c4_unlabelled.tsv`). Three ways to proceed:

- **(a) A blind scan by Opus agents**, the doc 103/116 procedure (`d116_scan_items.py`, the same rubric, audit and
  V2 checks). It is folded at the lowest precedence, like smx116, and re-graded by the same rule. It needs a prereg
  amendment with its sha recorded before any sheet is rendered. **Recommended:** it is how doc 116 resolved exactly
  this. The owner's go on the result is still needed; a pass under an amended rule is not the owner's confirmation.
- **(b) The owner looks at the items**, own116v-style (not blind, highest precedence): 43 items, against own116v's 25.
- **(c) An explicit override on the point values**, as in docs 108 and 116. All four point deltas are inside −0.010.

Whichever route is taken, F1 (§5) runs before any default changes.

**Still open from doc 34:**
- the top-up calibration shortfall (80 %);
- a lower tolerance rung (the 120-event read-out, f = 0.760, does not argue for one);
- the 26-cluster owner-record deficit;
- the runner's `PDVD_FLASH_TAIL_MERGE=1` default, which neither the production light record (`_keep`) nor `_q32ti`
  ever used.

## 7. Files
| file | what |
|---|---|
| `d35/prereg.md`, `prereg_sha.txt`, `libpin_prod.md5` | pre-registration and the production-binary pin |
| `d35/c1_sat_terms.txt` | C1 |
| `d35/c2_crossers.txt` | C2 (pre-registered read-out + the common-anchor line) |
| `d35/c3_tails_pinned.txt`, `c3_tails_prod_literal.txt`, `c3_tails_prod_zero_ok.txt` | C3 |
| `d35/c4_grade.txt`, `c4_grade_uncarried.txt`, `c4_grade_carry_null.txt`, `c4_control_vs_d116vflip_*.txt`, `c4_unlabelled.tsv` | C4 |
| `d35/c5_transfer.txt` | C5 |
| `scripts/d35_arms.sh` | clustering + PR arms with a light suffix (fork of `d29_stm_arms.sh`) |
| `scripts/d35_crossers.py`, `d35_match_tails.py`, `d35_stm_grade.py` | C2, C3, C4 |
| `scripts/d35_flip_compiled.sh`, `d35_light_f1.py`, `d35_f1_check.sh` | F1, prepared, not run |

Arms (new tags, M13): `work/<run6>_<idx>_q35ctl`, `_q35tk` (120 each); `_q35cfg` holds compile-only scratch for 3
events. Logs are in `/home/xqian/tmp/p35/arm_*`.

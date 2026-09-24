# PDVD ToT flip: the pre-flip checks pass, and ToT light is now the PDVD production default

**Status 2026-09-23. NOT FLIPPED.** The owner's go was conditional on the pre-flip checks (`d35/prereg.md`,
sha in `d35/prereg_sha.txt`, written before any doc-35 arm ran). C1 and C2 pass their bars. C3 passes on Q/L, though its
PR sub-clause fired literally on one zero-candidate event (§4.2). C5, which carried no bar, shows the production binary
transfers exactly. The STM gate (C4) is **UNDECIDED on both purities**. Its point values are inside the bar, but the ToT arm brings 33 STM candidates that no record has
labelled, against 10 for production. The pre-registered rule makes UNDECIDED a STOP, so no runner default was changed.
The owner's choice of how to label those items is in §6.

**Status 2026-09-24: FLIPPED. ToT light + `lasso_weight_unrailed` + `ks_sat_tol` 0.3075 is the PDVD production default
(§11).** The owner applied the staged runner edits, and every flip-equivalence check passes. Production with no
overrides reproduces the graded ToT arm `q35tk` byte-for-byte on all 120 events: pctree, TLA sidecars, PR trees, Bee
zips and calib dumps. The new production light record `_tot` equals the measured ToT light on 120 / 120.

*Earlier update, 2026-09-24:* with the owner's larger calibration (amendment 2, §10), C4 passed, but the flip was not
yet made.
- The combined calibration is 30 items with 4 disagreements (13.3 %, bar 25 %), so the blind labels are usable.
- Folded by amendment 1's rule, the STM gate passes every metric in both scenarios. Purity changes by −0.006 / −0.013
  and efficiency by +0.022 / +0.017.
- The runner edit was refused by the session's permission classifier ("Modify Shared Resources"). The staged edits
  are unapplied in `/home/xqian/tmp/p35/flip/`, and F1 cannot run until the owner applies them or grants the
  permission (§10.4).

*Earlier update, 2026-09-23 evening:* the blind scan of route (a) ran and failed its calibration check. Still NOT
flipped then.
The owner chose route (a). It was pre-registered as amendment 1 (`d35/prereg_amend1.md`), with its sha recorded before
any item was drawn.
- **The scan ran cleanly (§9).** Five blind Opus scanners judged 37 items and all five audits are clean.
- **Calibration failed.** On the 4 calibration items the scanners disagreed with the existing record on 2
  (50 %, bar 25 %). By the amendment the labels (`smx35`) are **not used for grading**, and no fold was computed.
- **A defect was found and fixed (§8).** The C4 grade had keyed production's clusters as if they shared the record's
  cluster ids. In two events they do not. The corrected grade changes nothing material: the purity point values are
  −0.007 / −0.010, the NEG bounds −0.043 / −0.042, so both purities are still UNDECIDED. The ToT arm still brings 30
  unlabelled candidates, against 8 for production.
- **The flip is still stopped.** The prepared edits are unapplied (`/home/xqian/tmp/p35/flip/`). The owner's options
  are in §6.

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
| C4 | STM gate: `is_stm` / Michel purity and efficiency vs production | First reading UNDECIDED → STOP (purity NEG −0.043 / −0.042 from unlabelled ToT candidates). **PASS after the calibrated blind scan (§10):** purity −0.006 / −0.013, efficiency +0.022 / +0.017, in both NEG and POS |
| C5 | does the doc 34 margin transfer to the production binary? (report only, no bar) | **Exactly.** Production-binary calib dumps are byte-identical to the measured arms, 120 / 120, for both production and ToT |
| F1 | flip equivalence | **PASS (§11)**: compiled config, the light record `_tot` == `_q32ti` 120 / 120, and production `q35flip` == `q35tk` 120 / 120 |

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
# sec 8 -- the corrected C4 (both cells re-keyed into the record lineage d116vflip) and its nulls
python3 d35_stm_grade.py --a0 q35ctl --t q35tk --unlabelled-out ../d35/c4k_unlabelled.tsv > ../d35/c4k_grade.txt
python3 d35_stm_grade.py --a0 q35ctl --t d116vflip > ../d35/c4k_grade_null.txt      # and --t q35ctl: exactly 0
# sec 9 -- the blind scan (amendment 1; round /home/xqian/tmp/p35scan, scratch)
python3 d35_scan_items.py --out /home/xqian/tmp/p35scan/items/pdvd_items.tsv          # 43 new + 4 calibration
(cd ../../../../pdhd/stm_michel_scan && for arm in q35ctl q35tk; do ./prep_stm_michel_scan.py --det pdvd --arm $arm \
   --ctx-cells --outdir /home/xqian/tmp/p35scan/round/prep_$arm --sheetdir /home/xqian/tmp/p35scan/round/sheet_$arm --redraw; done)
(cd ../../nf_sp_img_clus/scripts && python3 d103_scan_set.py --det pdvd --items /home/xqian/tmp/p35scan/items/pdvd_items.tsv \
   --prep-root /home/xqian/tmp/p35scan/round --out /home/xqian/tmp/p35scan/round_pdvd/set)
bash d35_shoot_round.sh pdvd round_pdvd 2          # RUBRIC.md = nf_sp_img_clus/d99/swap_scan_rubric.md; AGENT_TASK.md = 116's, paths changed
#   de-duplicate set/items_all.txt (2 same-object keys), then nextwave.py <round> <items_all> w1 --agents 5 --per 8 --seed 35
#   one blind Opus agent per wave file; python3 d35_audit.py <transcript> pdvd w1_a<i>   (every scanner) -> ../d35/scan_audit.txt
python3 d35_scan_record.py --det pdvd --round /home/xqian/tmp/p35scan/round_pdvd --items /home/xqian/tmp/p35scan/items/pdvd_items.tsv \
   --tag smx35 --record-out ../../scan/pdvd_stm_michel_smx35_verdicts.json > ../d35/scan_smx35.txt     # V2 FAIL
# sec 10 -- amendment 2 (round /home/xqian/tmp/p35scan/round_pdvd2, scratch): 26 calibration + 1 new item
python3 d35_calib2_items.py --out /home/xqian/tmp/p35scan/items/pdvd_items2.tsv
(cd ../../nf_sp_img_clus/scripts && python3 d103_scan_set.py --det pdvd --items /home/xqian/tmp/p35scan/items/pdvd_items2.tsv \
   --prep-root /home/xqian/tmp/p35scan/round --out /home/xqian/tmp/p35scan/round_pdvd2/set --seed 1036)
bash d35_shoot_round.sh pdvd round_pdvd2 2      # nextwave.py ... w1 --agents 4 --per 7 --seed 36; 4 blind Opus scanners
#   python3 d35_audit.py <transcript> pdvd w1_a<i> round_pdvd2 -> ../d35/scan2_audit.txt
python3 d35_scan_record.py --det pdvd --round /home/xqian/tmp/p35scan/round_pdvd2 --items /home/xqian/tmp/p35scan/items/pdvd_items2.tsv \
   --tag smx35c --record-out ../../scan/pdvd_stm_michel_smx35c_verdicts.json > ../d35/scan2_smx35c.txt
python3 d35_v2_combined.py ../../scan/pdvd_stm_michel_smx35_verdicts.json ../../scan/pdvd_stm_michel_smx35c_verdicts.json > ../d35/scan_v2_combined.txt
python3 d35_stm_grade.py --a0 q35ctl --t q35tk --scan-record ../../scan/pdvd_stm_michel_smx35_verdicts.json \
   ../../scan/pdvd_stm_michel_smx35c_verdicts.json --unlabelled-out ../d35/c4f_unlabelled.tsv > ../d35/c4f_grade.txt   # C4 PASS
# sec 11 -- the flip and F1 (pre-half BEFORE the runner edit, the rest after)
bash d35_flip_compiled.sh pre; <the runner edit>; bash d35_flip_compiled.sh post > ../d35/f1_compiled.txt
ARM=_tot ./d32_light_arms.sh; ARM=_q35esc ENVS="PDVD_SAT_REPAIR_MODE=twoside" ./d32_light_arms.sh
python3 d35_light_f1.py > ../d35/f1_light.txt
(cd ../../.. && LD_LIBRARY_PATH=/home/xqian/tmp/p35/libpin_prod:$LD_LIBRARY_PATH STM_PR_MODE=-nu setarch x86_64 -R ./stm/run_campaign.sh q35flip all)
bash d35_f1_check.sh                                        # -> ../d35/f1_identity.txt, f1_calib.txt
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

### C4 — the STM gate (`d35/c4k_grade.txt`, corrected in §8; first reading `c4_grade.txt`; as registered `c4_grade_uncarried.txt`)
The rule is doc 113's, with records own116v > own103v2 > own103v > p99rwon-carried-corrected > smx11 > smx116. Each of
the four metrics must be ≥ −0.020 vs production in both the NEG and POS scenarios for the unlabelled candidates.

| metric | production `q35ctl` | ToT `q35tk` | point Δ | NEG | POS | reading |
|---|---|---|---|---|---|---|
| `is_stm` purity | 0.986 (275 / 4) | 0.978 (272 / 6) | −0.007 | **−0.043** | −0.006 | UNDECIDED |
| `is_stm` efficiency | 0.777 | 0.768 | −0.008 | −0.008 | +0.021 | PASS |
| Michel purity | 0.938 (180 / 12) | 0.927 (178 / 14) | −0.010 | **−0.042** | −0.008 | UNDECIDED |
| Michel efficiency | 0.779 | 0.771 | −0.009 | −0.009 | +0.019 | PASS |

The table is the corrected reading (§8). The first reading, which keyed production directly, gave −0.004 / −0.009 /
−0.010 / −0.009 with the same NEG bounds and the same verdicts.

- **Split halves** (reported): the worst is −0.026, on Michel efficiency in the odd half; doc 116's guard is −0.040.
- **Sensitivity** (reading 2, the keys whose carry failed dropped from both cells): the same picture, with purity NEG
  −0.043 / −0.042.
- **Every point value is inside the bar.** The NEG bounds fail because the ToT arm has **30 candidates with no label
  in any record** (15 tagged `is_stm`, 9 with a Michel) against production's 8 (4 and 2); the first reading counted
  33 and 10. NEG counts every one of them as a false positive.
- **Most are not relabellings.** Only 3 of the 33 are the geometric images of a labelled cluster that failed to carry.
  The other ~30 are clusters that became STM candidates only under ToT Q/L: a changed flash (t0) moves a cluster in x,
  and matching-driven merges change what the tagger sees.
- The item list for any further labelling is `d35/c4k_unlabelled.tsv` (38 rows: 30 ToT, 8 production), the corrected
  successor of `c4_unlabelled.tsv` (43 rows), from which the §9 scan was drawn.

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
**Route (a) has been run (§9) and did not give usable labels.** The flip still waits on labels for the corrected 38
items (`d35/c4k_unlabelled.tsv`). What the owner can do now:
- **(i) Adjudicate the two calibration disagreements.** They are 039349_3/50 (record STM_MICHEL, blind THRU at low
  confidence) and 039252_7/29 (record THRU, blind STM_ONLY, FLAT_STOP, at medium). Both are hard items. If the owner
  sides with the scanners on either, the miss was the record's. V2 passing would then be the owner's ruling, not an
  inference, and the pre-registered fold and re-grade of amendment 1 §4 follow.
- **(ii) A larger calibration**, pre-registered as amendment 2 with the owner's approval. The combined sample (the 4
  plus the new items) must pass; the first 4 cannot be dropped.
- **(iii) The owner looks at the 38 items directly** (route b below).
- **(iv) An override on the point values** (route c below).

The original routes, kept for the record:

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
| `d35/c4k_grade.txt`, `c4k_grade_null.txt`, `c4k_unlabelled.tsv` | C4 corrected (§8) and its null |
| `d35/prereg_amend1.md` (sha in `prereg_sha.txt`), `scan_smx35.txt`, `scan_audit.txt`, `scan_posthoc_agreement.txt` | §9 blind scan |
| `pdvd/docs/scan/pdvd_stm_michel_smx35_verdicts.json` | the smx35 record (37 rows): V2 failed alone (2 / 4); **used after the combined V2 passed (§10)** |
| `scripts/d35_scan_items.py`, `d35_shoot_round.sh`, `d35_audit.py`, `d35_scan_record.py` | §9 |
| `d35/prereg_amend2.md`, `scan2_audit.txt`, `scan2_smx35c.txt`, `scan_v2_combined.txt`, `c4f_grade.txt`, `c4f_unlabelled.tsv`, `scan_bias_sensitivity.txt` | §10 |
| `pdvd/docs/scan/pdvd_stm_michel_smx35c_verdicts.json` | the smx35c record (27 rows); with smx35 it passes the combined V2 |
| `scripts/d35_calib2_items.py`, `d35_v2_combined.py` | §10 |
| `d35/f1_compiled.txt`, `f1_light.txt`, `f1_identity.txt`, `f1_calib.txt`, `f1_warning.txt` | §11 F1 |
| `pdvd/run_light_evt.sh`, `pdvd/run_clus_evt.sh`, `pdvd/stm/run_campaign.sh` | §11 **the production flip** |
| `scripts/d35_arms.sh` | clustering + PR arms with a light suffix (fork of `d29_stm_arms.sh`) |
| `scripts/d35_crossers.py`, `d35_match_tails.py`, `d35_stm_grade.py` | C2, C3, C4 |
| `scripts/d35_flip_compiled.sh`, `d35_light_f1.py`, `d35_f1_check.sh` | F1, prepared, not run |

Arms (new tags, M13): `work/<run6>_<idx>_q35ctl`, `_q35tk` (120 each); `_q35cfg` holds compile-only scratch for 3
events. Logs are in `/home/xqian/tmp/p35/arm_*`.

## 8. Correction: the C4 grade keyed production on the wrong ids in two events
`d35_stm_grade.py` first applied the records directly to production's (`q35ctl`) cluster ids and carried only the ToT
arm. The records are keyed on the doc 103–116 lineage (`d116vflip`, the `d103vflip` pctrees). `q35ctl` re-clusters
with today's Q/L defaults, and in **039349_39 and 039349_46 its ids are shifted by 2**.
- Example: `q35ctl` 039349_39/58 is the record's THRU 039349_39/56.
- So about 6 labels sat on the wrong production clusters, and the same objects were drawn into the §9 scan as
  "unlabelled".

Found while checking the V2 disagreements, which proved to be genuine: both calibration keys name the same object in
both lineages.

**Fix.** Both cells are now re-keyed into the record lineage by the same geometric carry (`--key-arm d116vflip`,
default). An unlabelled object present in both arms shares one private key. The first reading is kept as
`c4_grade.txt`.

**Nulls.**
- `q35ctl` vs itself: exactly 0 on every metric.
- `q35ctl` vs `d116vflip` (`c4k_grade_null.txt`): within ±0.006, and all PASS.

**Corrected C4** (`c4k_grade.txt`): the table in §2. The verdict is unchanged (UNDECIDED on both purities → STOP), and
the unlabelled count is 30 ToT / 8 production.

## 9. The blind scan (amendment 1, route (a)) — calibration failed on its own 4 items; the labels were used only after amendment 2 (§10)
**Pre-registration.** `d35/prereg_amend1.md`, sha appended to `prereg_sha.txt` at 21:12, before the item list was
drawn.

**Items** (`scripts/d35_scan_items.py`, private list):
- the 43 rows of `c4_unlabelled.tsv`, each on its own arm under its native key, plus 4 calibration items (seed 35)
  drawn from judged `q35ctl` candidates;
- the prep's own filters left **6 unscannable** (8 rows), which stay unlabelled as pre-registered: 039349_15/41,
  039349_16/41, 039349_52/53, 039349_64/68, 039349_64/70, 039349_77/40;
- **two keys, 039349_46/63 and 039349_82/19, were drawn on both arms.** The carry maps the ToT cluster onto the
  production cluster with the same id, so each is one object, shown once (the second set of shots is set aside);
- that leaves **37 unique items** (33 new + 4 calibration).

**Procedure.** Doc 116 §5, unchanged in substance:
- prep with `--ctx-cells`, `d103_scan_set.py`, blind headless shots at 2 processes: `check_shots` clean, 37 dirs, 0
  blank frames;
- the rubric is `d99/swap_scan_rubric.md` (sha d760e223) unchanged, and the agent task is doc 116's with the round
  paths changed;
- five Opus scanners, 8 / 8 / 8 / 8 / 5 items (`nextwave.py`, seed 35).

**Audit** (`scan_audit.txt`, `scripts/d35_audit.py`; its selftest flags 13 / 13 bad lines and 0 / 4 good ones): 77 /
76 / 77 / 76 / 50 tool calls, **0 flagged**.

**In-place rewrites.** Two scanners each rewrote one of their own records through `mkv.py`, so there are no double
scans:
- 039349_39/56: STM_MICHEL → THRU, after re-reading rule 7;
- 039252_8/99: dropped a `CONTINUES:` prefix; the verdict was unchanged.

**Record** (`scan_smx35.txt`, `pdvd/docs/scan/pdvd_stm_michel_smx35_verdicts.json`, 37 rows):
- new items: 11 STM_MICHEL, 10 STM_ONLY, 1 FRAG_STM_ONLY, 8 THRU, 1 MESSY, 2 UNCLEAR;
- confidence: 17 high, 14 medium, 2 low.
- `scripts/d35_scan_record.py` is `d103_scan_record.py` with one change: V2 reads the truth union the calibration
  was drawn from. The unchanged script reads only the doc 103 lineage record, which lacks one calibration key
  (039253_6/124), and crashed on it.

**V2: 2 of 4 disagree (50 %, bar 25 %) → FAIL; the labels are not used for grading, and no fold was computed.**
- 039349_3/50: record STM_MICHEL vs blind THRU, low confidence. The scanner read the fit end as the upper end of a
  near-vertical track, with the peak 6–15 cm back and a long decline. Its competing reading was an upward-going
  stopper.
- 039252_7/29: record THRU vs blind STM_ONLY (FLAT_STOP), medium confidence. The end is 8 cm above the bottom anode.
  The competing reading was THRU/ANODE, if the last 8 cm were not imaged.
- Four items give little power (the binomial interval on 2/4 is about 7–93 %). That argues for a larger calibration
  next time, not for re-reading this one.

**Post hoc, NOT governing** (`scan_posthoc_agreement.txt`). Because of the §8 defect, 3 scanned objects are in fact
record-labelled: 039349_39/46 (an owner review), 039349_39/56 and 039349_46/65. The blind verdicts agree with the
record on all 3, stopper or not. With the formal 4 that is 5 of 7, which would still be above the bar. It enters no
decision.

**Rubric gaps the scanners raised** (for the owner; keys in their reports):
- the rubric's hard rule 1 still names the old `p99scan` path; they followed the task file;
- a Michel bridged inside the muon's own long segment cannot be recorded as STM_MICHEL, because the split-row rule
  forces a `muon` tag (039252_6/39, 039349_3/50);
- unfitted C-rows carry no dQ/dx, which collides with the degenerate-row clause (039349_80/62, 039252_8/111);
- there is no prefix for an end at the readout-window edge (039349_26/56, 039252_17/75);
- there are no rules for tracks along the drift (039252_12/100);
- the pin position is undefined when the decline stays above the plateau (039253_2/74, 039253_2/76);
- an arm leaving a body vertex ~5 cm before the stop (039252_17/29).

## 10. The larger calibration (amendment 2, owner 2026-09-24) — V2 PASSES, C4 PASSES
**Pre-registration.** `d35/prereg_amend2.md`, sha appended to `prereg_sha.txt` before any item was drawn. Before the
draw the advisor review tightened it in four ways:
- the new calibration items come only from owner-derived sources: grade-truth source `record` (the p99rwon
  owner-corrected record and own116v) or `owner_review`, not the AI scans smx11 / smx116;
- the concrete threshold: at most 7 disagreements of 30;
- a note that round 1's 4 keys were untouched by the §8 defect;
- **one new item**: a coverage check of `c4k_unlabelled.tsv` by key membership only found `q35ctl` 039252_8/102
  unscanned. It is unlabelled only because its record label fails to carry (the cluster split).

### 10.1 The round
- **Items** (`scripts/d35_calib2_items.py`): 26 calibration items (seed 36, from a pool of 381) plus the one new item,
  shuffled together and all scannable.
- **Procedure.** Round dir `/home/xqian/tmp/p35scan/round_pdvd2`, with the same prep, rubric and agent task (path
  changed). Shots clean (27 dirs, 0 blank).
- **Scanners and audit.** Four fresh Opus scanners (7 / 7 / 7 / 6 items). The audit (`scan2_audit.txt`) found 67 /
  70 / 67 / 61 tool calls, 0 flagged. Three scanners rewrote records in place to fix tags or evidence, so there are no
  double scans.
- **Record** `pdvd/docs/scan/pdvd_stm_michel_smx35c_verdicts.json` (27 rows; `scan2_smx35c.txt`).

### 10.2 The combined V2 (`scan_v2_combined.txt`, `scripts/d35_v2_combined.py`)

| | calibration items judged on both sides | disagreements |
|---|---|---|
| amendment 1 (smx35) | 4 | 2 |
| amendment 2 (smx35c) | 26 | 2 (039252_12/110 record THRU vs blind STM_ONLY, low; 039253_14/113 record THRU vs blind STM_MICHEL, low) |
| **combined** | **30** | **4 = 13.3 % → PASS** (bar 25 %) |

**The direction of the misses.** 3 of the 4 are record THRU called a stopper by the blind scan. That is the
direction that would flatter ToT's purity, and it is sized in §10.3.

### 10.3 The fold and C4 (`c4f_grade.txt`, `c4f_unlabelled.tsv`)
Amendment 1 §4, unchanged:
- smx35's non-calibration rows and smx35c's new item are folded at the lowest precedence into the corrected grade;
- 27 items are folded; the rest were already labelled under the corrected keys, including both arms' views of
  039253_2/76 and 039349_14/44;
- 4 unlabelled candidates are left per arm, the unscannable ones.

| metric | production `q35ctl` | ToT `q35tk` | point Δ | NEG | POS | reading |
|---|---|---|---|---|---|---|
| `is_stm` purity | 0.982 (277 / 5) | 0.976 (285 / 7) | −0.006 | −0.006 | −0.006 | PASS |
| `is_stm` efficiency | 0.747 | 0.768 | **+0.022** | +0.022 | +0.021 | PASS |
| Michel purity | 0.938 (181 / 12) | 0.925 (185 / 15) | −0.013 | −0.013 | −0.013 | PASS |
| Michel efficiency | 0.754 | 0.771 | **+0.017** | +0.017 | +0.016 | PASS |

- **C4 PASS.** Split halves are worst −0.020 (Michel purity, odd half), within doc 116's −0.040. Reading 2 is the same.
- **What changes.** ToT tags more real stoppers: 8 more `is_stm` true positives and 4 more Michels. It costs 2 more
  `is_stm` and 3 more Michel false positives.
- **Post hoc, NOT governing** (`scan_bias_sensitivity.txt`). 15 folded ToT items are blind-labelled stoppers and
  tagged `is_stm`. The `is_stm` purity pass survives up to 4 of them being THRU. The calibration's rate of wrong
  stopper calls (3 of 23 = 13 %) predicts about 2.

### 10.4 The flip: blocked on a permission
Amendment 1 §5 was followed as far as it could go:
- **The F1 pre-half was re-taken** immediately before the edit. It reproduces the 2026-09-23 compiles
  byte-for-byte (same md5s).
- **The production pin was re-checked.** `libpin_prod` and `local/lib` both still match `libpin_prod.md5`.
- **The edit itself was refused by the session's permission classifier** ("Modify Shared Resources"): copying the
  three staged files over `run_light_evt.sh`, `run_clus_evt.sh` and `stm/run_campaign.sh`. It was not worked around.

**What remains, in order, once the owner applies the edit or grants the permission:**
1. `d35_flip_compiled.sh post` (F1a);
2. the light records `_tot` (flipped runner, `_keep`'s argument set) and `_q35esc` (`PDVD_SAT_REPAIR_MODE=twoside`),
   then `d35_light_f1.py` (F1b);
3. `setarch -R stm/run_campaign.sh q35flip all` on the pin, with `STM_PR_MODE=-nu` so PR runs the chain the arms ran
   (the campaign's own default is `run_pr_evt.sh`'s `stm` mode; its header comment saying "-nu" is stale), then
   `d35_f1_check.sh` (F1c).

Any F1 failure reverts the edit.

## 11. The flip (2026-09-24)
The owner applied the three staged files after the classifier refusal of §10.4:
```
cp /home/xqian/tmp/p35/flip/run_light_evt.sh /home/xqian/tmp/p35/flip/run_clus_evt.sh pdvd/ && cp /home/xqian/tmp/p35/flip/run_campaign.sh pdvd/stm/
```

**What changed** (wcp only; the toolkit C++ and jsonnet defaults stay OFF, the doc 12 / doc 29 precedent):

| file | change | escape to the pre-flip point |
|---|---|---|
| `run_light_evt.sh` | `PDVD_SAT_REPAIR_MODE` defaults to `tot`, only while `PDVD_SAT_REPAIR=1` and `PDVD_SPE_V2=1`; `PDVD_HIT_INT_SAMPLES` follows the mode (1 under `tot`); twoside + int warns | `PDVD_SAT_REPAIR_MODE=twoside` (int then defaults to 0) |
| `run_clus_evt.sh` | `PDVD_QL_LASSO_W_UNRAILED` defaults to 1, `PDVD_QL_KS_SAT_TOL` to 0.3075; a stderr warning when they meet light whose `.wct-light.json` is not `tot` | `PDVD_QL_LASSO_W_UNRAILED=0 PDVD_QL_KS_SAT_TOL=` |
| `stm/run_campaign.sh` | `PDVD_LIGHT_SUFFIX` defaults to `_tot` instead of `_keep` | `PDVD_LIGHT_SUFFIX=_keep` |

**Scope.** The same as doc 29: this repository's PDVD chain (`run_light_evt.sh`, `run_clus_evt.sh`, the campaign and
every arm script built on them). A job that compiles the toolkit jsonnet directly still gets twoside light and the
legacy Q/L. **Re-running any pre-flip arm** (`q31ctl`, `_g31off`, the d29 / d31–d35 arms) now needs the escape
assignments above. The new warning names the twin case when only the light is old.

**F1, all PASS:**
- **(a) compiled config** (`d35/f1_compiled.txt`, events 039252_0 / 039253_15 / 039349_7; the pre-half was re-taken
  just before the edit): post-flip bare equals pre-flip plus the two Q/L assignments, and post-flip escape equals
  pre-flip bare. The only leaves that move are `ks_sat_tol` (absent → 0.3075) and `lasso_weight_unrailed`
  (absent → true).
- **(b) light** (`d35/f1_light.txt`): the new record `_tot`, made by the flipped runner with `_keep`'s argument set,
  equals `_q32ti` on 120 / 120 archives (member content) and configs, and says `tot` + `int_samples`. The escape
  `_q35esc` equals the pre-flip `_g31off` on 120 / 120, with neither key.
- **(c) runtime** (`d35/f1_identity.txt`, `f1_calib.txt`):
  - the run was `setarch -R stm/run_campaign.sh q35flip all` on `libpin_prod` (== `local/lib`), with no PDVD_* env
    and `STM_PR_MODE=-nu` (the chain the arms ran);
  - it is identical to `q35tk` on 120 / 120 events (pctree, TLAs, PR trees with 343 branches, mabc-pr) and its calib
    dumps are byte-identical on 120 / 120;
  - it differs from pre-flip production `q35ctl` on 120 / 120;
  - `ks_sat_tol` is present in 120 / 120 flip dumps and 0 / 120 control dumps.
- **Warning** (`d35/f1_warning.txt`): silent on ToT light; one line on `_keep` light with the new defaults; silent
  again with the escape.

**The flip was verified for the `-nu` PR chain.** `stm/run_campaign.sh` with no env runs `run_pr_evt.sh`'s `stm` mode
(its header comment saying "-nu" is stale). The Q/L and light change is upstream of PR, but the byte identity was only
shown for `-nu`.

**What production now is, against the old default, in one line each:**
- Q/L hand-scan margin (doc 34): non-inferior, even-half worst case −0.16 clusters vs a −6.46 margin; not shown
  better.
- STM gate (§10): `is_stm` efficiency +0.022 and Michel +0.017, purity −0.006 / −0.013. **All four numbers rest on the
  blind labels** of the ~27 folded items. Their calibration missed mostly in the stopper direction (3 of 4), so read
  the efficiency gain as a scan-dependent estimate, not an independent measurement.
- Light closure (C2): unchanged on common crossers; rail-inclusive / rail-excluded 1.038 (was 0.844 with clipped rails).

**Still open:**
- the owner-record deficit of doc 34 (26 clusters on the twoside-era record);
- a lower `ks_sat_tol` rung;
- the runner's `PDVD_FLASH_TAIL_MERGE=1` default, which neither `_keep` nor `_tot` used. A fresh
  `run_light_evt.sh` with no env now gives ToT **plus** tail merge, a combination never graded. The production
  record `_tot` is made with `PDVD_FLASH_TAIL_MERGE=0`, like `_keep` before it (`d32_light_arms.sh`);
- the scanners' rubric gaps (§9, and the round-2 reports: a knife-edge 0.3 direction cut, the `pin_rr` origin when
  `context.pin` ≠ `ends.stop`, and grey-only crossers at the stop's drift time).

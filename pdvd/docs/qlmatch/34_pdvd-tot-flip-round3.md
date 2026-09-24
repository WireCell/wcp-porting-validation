# PDVD ToT flip campaign, round 3: rails measured on geometric truth, a KS tolerance ladder, and the owner's margin

**Status 2026-09-23. Round 3 of the doc 32/33 campaign. Nothing is flipped; the new knob is default OFF.**

**Update 2026-09-23 (doc 35): the pre-flip checks of §7 ran; NOT flipped.** The Q/L stage of today's production binary
reproduces `q34tk1` and `q31ctl` byte for byte (120 / 120), so this margin applies to production as is. Doc 11 §6
and doc 12 pass. The 102 unscanned events pass on Q/L (a PR sub-clause fired literally on one zero-candidate event). The 120-event STM gate is UNDECIDED on both purities (point values inside the
bar, NEG bounds −0.043 / −0.042 from 33 unlabelled ToT candidates), which the pre-registration makes a STOP. See
`35_pdvd-tot-flip-preflip-checks.md` §6 for the owner's choice. Later the same evening the owner chose a blind agent scan of those items (doc 35 §9).
It ran cleanly but failed its calibration check (2 of 4), so its labels are not used, and the flip is still stopped.

**Goal (owner).** Flip PDVD light to the ToT saturation fill (docs 30–33). Q/L need only **not be worse** than
production. The owner accepted the non-inferiority margin on 2026-09-23, before any round-3 scoring (`d34/prereg.md`).

**Round 3 in one paragraph.**
- **A correction first.** The round-2 plan to "recalibrate on ToT light" rested on a wrong premise (§1). Measured on
  geometric crossers, the ToT rails already **agree** with the prediction: median 1.04 [0.69, 1.35] (§2).
- **What round 3 did instead.** It added a KS tolerance for repaired rails, `ks_sat_tol` (default OFF, byte-identical
  on PDVD and uBooNE), calibrated from those crossers at f = 0.615. It then ran a three-rung ladder {f/2, f, 2f} on
  ToT light, every rung on top of `lasso_weight_unrailed`.
- **Result by the pre-registered rule: `q34tk1` PASSES the owner's margin.** `q34tk1` is ToT + int_samples + L3 +
  `ks_sat_tol` 0.3075.
  - **Selection.** It was chosen on odd events: net +7 vs the reference `q33tu` at +2.
  - **Confirmation.** On the even half it scores +9 (20–11) vs production. The one-sided 95 % worst case is −0.16
    clusters, against a margin of −6.46.
  - **The even half is not fully held out.** Round 2's even half is partly spent: L3 became every arm's base
    because of its even result there (prereg disclosure). The only fresh even data is the top-up, where `q34tk1` is
    0 (3–3).
  - **Phantoms** on the owner+target record: 150 vs 163.
  - **Twin.** Its control-light twin is +5 on even and −2 on odd events, so most of the gain is ToT-specific.
- **What it is not.** It is **not** significantly better: p = 0.15 on the even half and 0.036 over all events, where
  the tuning half is included.
- **The top-up scan failed its calibration bar** (80 % vs 85 %). The governing score is therefore on the round-2
  target, as pre-registered. The union and the fresh top-up are reported alongside: the union also passes, and on the
  88 fresh movers `q34tk1` is neutral (−1).
- **Nothing is flipped.** The pre-flip list is in §7.

**Repro:**
```
# toolkit (apply-pointcloud) `b5a897f6`: match ks_sat_tol (default OFF)
wcbuild && ./wcb build --target=wcdoctest-match && ./build/match/wcdoctest-match          # 10/10
# pin: /home/xqian/tmp/p34/libpin_q34 = libpin_q33b with the ks_sat_tol libWireCellMatch
cd pdvd/docs/qlmatch/scripts
W=/home/xqian/tmp/p31/wroot   # + symlinks 039252_<idx>_<arm> -> pdvd/work/039252_<idx>_<arm>
python3 d34_rail_calib.py --work-root $W --arms q32ti,q31ctl > ../d34/rail_calib.txt           # step A
P=/home/xqian/tmp/p34/libpin_q34
ARM=q34ctlp LIGHT=_g31off PIN=$P ./d31_ql_arms.sh                                                # gate g34pin
for k in 1:0.3075 2:0.615 3:1.23; do i=${k%%:*}; t=${k#*:}
  ARM=q34tk$i LIGHT=_q32ti  PIN=$P ENVS="PDVD_QL_LASSO_W_UNRAILED=1 PDVD_QL_KS_SAT_TOL=$t" ./d31_ql_arms.sh
  ARM=q34ck$i LIGHT=_g31off PIN=$P ENVS="PDVD_QL_LASSO_W_UNRAILED=1 PDVD_QL_KS_SAT_TOL=$t" ./d31_ql_arms.sh; done
(cd ../../../../qlport/scripts && LD_LIBRARY_PATH=/home/xqian/tmp/p34/ubbase ./sweep_5384.sh d34ub_base 6 \
   && ./sweep_5384.sh d34ub 6 && ./ab_check.sh d34ub d34ub_base)                                  # uBooNE knob-off gate
for arm in q34ctlp q34tk1 q34tk2 q34tk3 q34ck1 q34ck2 q34ck3; do for i in $(seq 0 17); do
  ln -sfn $PWD/../../../work/039252_${i}_$arm $W/039252_${i}_$arm; done; done
python3 d34_mover_counts.py --work-root $W --time-map ../d32/time_map_ctl_to_q32ti.json --pairs <see d34/mover_counts.txt> > ../d34/mover_counts.txt
A=q32ti:T,q33ts:T,q33tm:T,q33tu:T,q33cs:C,q33cm:C,q33cu:C,q34tk1:T,q34tk2:T,q34tk3:T,q34ck1:C,q34ck2:C,q34ck3:C
python3 d34_scan_items.py --exclude-key ../d33/scan_r2/key.json --work-root $W --arms $A --time-map ../d32/time_map_ctl_to_q32ti.json \
    --sheet-dir /home/xqian/tmp/p34/scan/r3 --key /home/xqian/tmp/p34/scan_key_r3/key.json
#   9 scanner agents (d33/RUBRIC.md unchanged; D33_SCAN_ROOT=/home/xqian/tmp/p34/scan/r3 d33_record.py), d33_audit.py per transcript
python3 d33_merge_record.py --key <key.json> --scan-root /home/xqian/tmp/p34/scan/r3 --out ../d34/scan_r3 --adjudicate-out ../d34/scan_r3/contradicted.json
python3 d33_adjudicate_items.py --key <key.json> --contradicted ../d34/scan_r3/contradicted.json --work-root $W \
    --time-map ../d32/time_map_ctl_to_q32ti.json --scan-root /home/xqian/tmp/p34/scan/r3 --wave 9 --out-key <key_adj.json>   # agent 10
python3 d33_merge_record.py --key <key.json> --key-extra <key_adj.json> --scan-root /home/xqian/tmp/p34/scan/r3 --out ../d34/scan_r3
S=q33tu:T,q34tk1:T,q34tk2:T,q34tk3:T,q34ck1:C,q34ck2:C,q34ck3:C
python3 d33_paired_score.py --work-root $W --ctl q31ctl --arms $S --time-map ../d32/time_map_ctl_to_q32ti.json \
    --truth ../d33/scan_r2/truth.jsonl --key ../d33/scan_r2/key.json --key-extra ../d33/scan_r2/key_adj.json --out ../d34/scan_r3/r2truth
#   union: the same with truth = d33 truth + d34 truth and key = all four keys concatenated (--out ../d34/scan_r3/union);
#   top-up alone: --truth ../d34/scan_r3/truth.jsonl --key <key.json> --key-extra <key_adj.json> --out ../d34/scan_r3/topup
cd ../../.. && python3 ql_display/ql_agree_score.py --tag <arm> --work-root $W --truth-uid-map-tag keep \
    [--override-truth docs/qlmatch/d34/scan_r3/<r2truth|union>/override_target_<arm>.jsonl] --out-root /home/xqian/tmp/p34/scores_...
cd docs/qlmatch/scripts && python3 d34_margin.py --union ../d34/scan_r3/r2truth/paired_score.txt --fresh ../d34/scan_r3/topup/paired_score.txt \
    --scores-root /home/xqian/tmp/p34/scores_target_r2truth --ladder q34tk1,q34tk2,q34tk3 --ref q33tu \
    --twins q34tk1=q34ck1,q34tk2=q34ck2,q34tk3=q34ck3 > ../d34/scan_r3/margin_r2truth.txt      # (and _union)
```

## 1. Correction: the round-2 "recalibrate on ToT light" premise was wrong
Doc 33 §8 item 1 proposed refitting QtoL and the cathode visibility on ToT light. It claimed production's calibration
"saw rails at clipped values" and is therefore biased low. That claim was mine, and it is wrong:
- doc 12's per-type factors (`ql_light_calib/fit_qtol_crossers.py`) **exclude railed channels from both sums**;
- the unrailed PE is the same in both lights.

A refit on ToT light therefore returns the same factors. What ToT newly allows is putting the **railed channels
themselves** against the prediction, so round 3 starts there.

## 2. Step A: the ToT rails agree with the prediction on geometric crossers (`d34/rail_calib.txt`)
**Method** (`scripts/d34_rail_calib.py`).
- **Anchors.** The 18 run-039252 dumps, using the `fit_qtol_crossers.py` strict cathode-crosser anchors, which are
  geometric truth, independent of the LASSO and the scan.
- **Per anchor.** s = Σmeas/Σpred over the kept unrailed channels. For each railed channel, r = (meas/pred)/s.
- **Pre-registration.** The rule was fixed in `d34/prereg.md` part 1 before the script ran (SHAs in
  `d34/prereg_sha.txt`).

| | ToT `q32ti` | production `q31ctl` |
|---|---|---|
| anchors / railed samples | 49 / 164 | 47 / 158 |
| median r [16 %, 84 %] | **1.040 [0.690, 1.354]** | 0.920 [0.500, 1.253] |
| r, brightest quintile (pred > 13 k PE) | 1.024 | **0.550** |
| rail-inclusive / rail-excluded global ratio | 1.082 | 0.921 |

**Reading.**
- **Repaired rails match the prediction on truth anchors,** flat in brightness. The per-channel medians run 0.67–1.21.
  The prediction is not mis-calibrated for ToT light.
- **Production's clipped rails fall to half the prediction at the brightest channels,** as expected for a clip.
- **Where doc 33 §2's "rails about 2× the prediction" comes from.** It is per bundle in shared flashes: the other
  clusters' light is on the rail but not in that bundle's prediction. The KS compares shapes, so this is a shape
  mismatch, not an offset. A tolerance attacks it; a rescale would not.
- **Pre-registered read-out.**
  - 164 ≥ 30 samples, so **f = exp(q84 |ln r|) − 1 = 0.615**, and the ladder is 0.3075 / 0.615 / 1.23.
  - The QtoL trigger reads 1.082, within 10 %, so **there is no QtoL arm**.
- **Overlap.** The anchors come from the same 18 events as the scan target. They are geometric, not scan verdicts;
  this is stated rather than hidden.

## 3. The knob: `ks_sat_tol` (toolkit `b5a897f6`, default OFF)
**What it does.** `BundleQualityParams::ks_sat_tol` and the free function `Match::ks_sat_clamp`
(`match/{inc,src}/TimingTPCBundle.*`) act in both `examine_bundle` KS loops.
- **The clamp.** On a channel the flash flags railed (left in the KS by `saturation_mask_fit=false`), the measured
  value entering the KS becomes `clamp(pred·s, pe/(1+tol), pe·(1+tol))`.
  - s is the bundle's Σpe/Σpred over its unmasked **unrailed** channels, the same normaliser as step A.
  - A rail agreeing with the scaled prediction within (1+tol) counts as agreeing.
  - A rail further off moves toward it by at most that factor.
  - With no unrailed light, there is no clamp.
- **Reach.** The KS feeds the LASSO weight (`δ_shape·nopdet·ks/λ`), the ks cuts and the bundle merges.
- **Caveat.** In a shared flash the unrailed Σpe includes the other clusters' light, so s is an upper estimate.
- **Plumbing.**
  - QLMatching `ks_sat_tol` (member default 0, round-trips in `default_configuration`; the calib-dump `qp` key is
    written only when > 0).
  - PDVD argument `ql_ks_sat_tol` (null ⇒ key suppressed); runner `PDVD_QL_KS_SAT_TOL=<tol>`.

**Gates.**
- **Tests.** `wcdoctest-match` 10/10 cases, 103 assertions. The new cases are:
  - the default 0;
  - `ks_sat_clamp` on a shared flash (bundle pred 0.3× the flash): a matching rail is a no-op, a 3× rail is pulled by
    exactly 1+tol, a rail inside the tolerance lands on the scaled prediction, and no unrailed light means no clamp.
- **Freshness.** `local/lib/libWireCellMatch.so` 16:43:24 is newer than the source edits (16:42:34).
- **Compiled clustering config, knob off.** Identical to a HEAD overlay of both jsonnet files on 3 argument sets
  (default; saturation flag + keep rails + inflate 0.5; the same + L3). The HEAD overlay rejects the new argument
  (rc 134). With the knob on, exactly one `ks_sat_tol` key appears.
- **PDVD gate `g34pin`** (`d34/gate/g34pin.txt`): pin `libpin_q34` with the knob off gives calib dumps `cmp`-identical
  to `q31ctl` on **120/120** events.
- **uBooNE gate** (`d34/gate/ub_d34ub.txt`), needed because TimingTPCBundle is shared: `qlport/scripts/ab_check.sh
  d34ub d34ub_base` gives 35/35 Bee zips content-identical and 35/35 tagger logs identical. The base sweep ran with
  the `5376f739` Match library prepended.
- **PDHD gate** (`d34/gate/pdhd_d34q.txt`): run 029107 events 10–13 with Q/L on, in tagged work dirs. Compared with
  the `5376f739` Match library, calib dumps are identical 4/4 and mabc zips are content-identical 60/60.
- **SBND is not gated.** `abtest/sbndgate_run.sh` has no input set left. What remains is the structural argument: the
  new branch runs only for `ks_sat_tol > 0`, and the dump key is written only then.
- **Knob-on smoke** (evt298567, `q34ck1` vs `q33cu`, which differ only by tol 0.3075): 972 of 5371 bundles change
  their KS, 876 of them downward (median −0.024). The run log carries `QLMatching ks_sat_tol=0.3075`.

## 4. Arms
All arms run on the clustering pin `libpin_p100b` + `libWireCellMatch` from `libpin_q34`, with
`PDVD_QL_LASSO_W_UNRAILED=1`; 120/120 events each.

| arm | light | `ks_sat_tol` | role |
|---|---|---|---|
| `q34tk1` / `q34ck1` | ToT `_q32ti` / control `_g31off` | 0.3075 (f/2) | ladder rung / twin |
| `q34tk2` / `q34ck2` | ToT / control | 0.615 (f) | ladder rung / twin |
| `q34tk3` / `q34ck3` | ToT / control | 1.23 (2f) | ladder rung / twin |
| `q33tu` (round 2) | ToT | – | reference (L3, no tolerance) |

**Mover counts over the 18 events** (`d34/mover_counts.txt`, `scripts/d34_mover_counts.py`; the definition
reproduces doc 33's 100 / 96 / 125):

| | vs `q31ctl` | ToT vs its twin | vs the no-tolerance arm |
|---|---|---|---|
| tol 0.3075 | ToT 161, control 121 | 101 | 81 (ToT), 83 (control) |
| tol 0.615 | ToT 193, control 167 | 96 | 129, 136 |
| tol 1.23 | ToT 242, control 224 | 113 | 180, 197 |

- **The tolerance is an operating-point lever in both lights.** It does not make the light choice irrelevant: ToT vs
  its twin stays at about 100 (as with L3 alone, 96).
- **Owner record alone** (`d34/owner_record_scores.txt`; the record was taken on twoside light, doc 33):
  - `q31ctl` 683/97/158, `q33tu` 653/82/188;
  - `q34tk1` 657/84/184, `q34tk2` 648/84/193, `q34tk3` 634/85/207;
  - twins 677/89/164, 663/88/178, 640/88/201.
  - Larger tolerances lose agreement in both lights.
  - `q34tk1` is 4 better than `q33tu` on agree and on missed, and still 26 short of production on this light-biased
    record. Doc 33 §5 reads that deficit as not separable from the record's bias.

## 5. Top-up blind scan
**Population.** Movers vs `q31ctl` over the round-2 + round-3 arms, minus the 329 already scanned:
- **88 new movers** (46 odd, 42 even), under the cap of 300, so no sampling;
- 224 sheets rendered: 176 mover looks, 40 calibration, 9 duplicates. One item had no bundle.

**Procedure.** The same as round 2: light-neutral sheets, `d33/RUBRIC.md` byte-identical, 9 scanners of 24–25 sheets,
and an adjudicator for the one contradiction.

**Audit** (`d34/scan_r3/audit.txt`): every transcript read only its own wave, the rubric and the recorder. The one
flag per wave is the agent's final hand-back call, not a file access.

**Target** (`d34/scan_r3/merge_report.txt`):
- 69 of 88 resolved = **78.4 %** (bar 60 %): 62 positive, 7 ties;
- 19 unresolved as unsure, 0 contradictions left after adjudication.

**Calibration against the owner record: 80.0 % (20/25 committed), which FAILS the 85 % bar.**
- The split is cathxa 14/17 and gold 6/8.
- Round 2 was 38/42 on different calibration items. *Post hoc, and not used for any decision:* pooled over both
  rounds it is 58/67 = 86.6 %.
- Per the prereg, the governing score is **the round-2 truth alone**, with the union reported alongside. The 5
  disagreements are listed in the merge report: 3 are owner-negative flashes the scanner picked, and 2 are different
  positives.

**Duplicates:** 5 compatible, 3 with one side unsure, 1 contradiction.

## 6. Scores and the owner's margin
**Governing: the round-2 target** (`d34/scan_r3/r2truth/paired_score.txt`, `margin_r2truth.txt`). Paired against
`q31ctl` on resolved non-tie movers (n = 91 odd / 111 even); bound = net − 1.645·√(W+L).

| arm | odd net (W–L) | even net (W–L) | even bound | all net, p |
|---|---|---|---|---|
| `q33tu` (reference) | +2 (12–10) | +2 (14–12) | −6.39 | +4, 0.67 |
| **`q34tk1`** tol 0.3075 | **+7 (14–7)** | **+9 (20–11)** | **−0.16** | **+16, 0.036** |
| `q34tk2` tol 0.615 | +4 (15–11) | +8 (24–16) | −2.40 | +12, 0.18 |
| `q34tk3` tol 1.23 | +2 (15–13) | +10 (26–16) | −0.66 | +12, 0.19 |
| `q34ck1` (twin) | −2 (5–7) | +5 (11–6) | −1.78 | +3, 0.71 |
| `q34ck2` (twin) | −7 | +9 | | +2 |
| `q34ck3` (twin) | −7 | +3 | | −4 |

**Owner record with the round-2 target replacing every scanned mover** (agree / phantom / missed):

| arm | score |
|---|---|
| `q31ctl` | 634 / 163 / 195 |
| `q33tu` | 635 / 154 / 199 |
| **`q34tk1`** | **645 / 150 / 189** |
| `q34tk2` | 638 / 150 / 195 |
| `q34tk3` | 632 / 144 / 201 |
| twins | 631 / 163 / 198, 625 / 155 / 204, 610 / 151 / 219 |

**Pre-registered read-out** (`d34_margin.py`).
- **Selection (odd):** `q34tk1` +7 > `q33tu` +2, so `q34tk1` is selected.
- **Margin (even):** the bound is −0.16 ≥ −6.46 (**OK**), and phantoms are 150 ≤ 163 (**OK**). **`q34tk1` PASSES.**
- **Superiority is not shown:** even-half p = 0.15. The all-events p = 0.036 includes the tuning half and is not a
  confirmation.
- **Twin:** `q34ck1` is +5 on the even half and −2 on the odd half, against `q34tk1`'s +9 / +7. The gain is mostly
  ToT-specific, and the tolerance also helps production light a little on the even half.

**Cross-checks (descriptive).**
- **Union** (round-2 target ∪ top-up, n = 124 / 140; `union/`, `margin_union.txt`). `q34tk1` is selected again (odd
  +6) and passes: even +9 (23–14), bound −1.01. Owner+union phantoms are 171 vs 181, and agree/missed are also better
  (648/206 vs 633/214).
- **Fresh top-up movers alone** (`topup/`). `q34tk1` is −1 on odd events (2–3) and 0 on even (3–3), which is neutral.
  `q34tk3` is **−9 / −8** (p 0.035 / 0.096). The largest tolerance loses on the clusters it newly moves, which fits
  its owner-record drop. The ladder has a dose tail, and the smallest rung is the safe one.
- **`q34tk1` is the lowest rung,** so the optimum may lie lower still. That would be a further ladder, not something
  this round tested.

## 7. Next
The candidate for the owner's flip decision is **ToT light (`saturation_repair_mode tot` + `int_samples`) with
QLMatching `lasso_weight_unrailed` + `ks_sat_tol` 0.3075.** Before a go, in order:
1. **Doc 11 §6 and doc 12** rerun on the `q34tk1` operating point (light closure; rail-inclusive crosser check).
2. **The other 102 events** (runs 39253 / 39349) through ToT light + `q34tk1` Q/L, for crashes and match-rate tails.
   They carry no hand-scan record.
3. **A flip-equivalence gate.** Flipping the defaults (runner and/or jsonnet) must reproduce `q34tk1`'s calib dumps
   byte-identically.
4. **The owner's go.** Only then does any default change.

**Done 2026-09-23 in doc 35** (`35_pdvd-tot-flip-preflip-checks.md`): item 1 passes and item 2 passes on Q/L, and the production binary
reproduces this doc's arms exactly. The added STM gate stopped the flip before item 3 (UNDECIDED purities: unlabelled
ToT STM candidates).

**Open items.**
- The top-up calibration shortfall (80 %).
- A lower-tolerance rung.
- `q34tk1`'s 26-cluster owner-record deficit, which is read as that record's twoside bias (doc 33 §5) but not proven
  to be.

## 8. Files
- **toolkit** `b5a897f6`:
  - `match/{inc,src}/TimingTPCBundle.*`, `match/{inc,src}/QLMatching.*`;
  - `match/test/doctest_qlmatching_config.cxx`;
  - `cfg/.../protodunevd/{qlmatching,wct-clustering}.jsonnet`.
- **wcp:**
  - `run_clus_evt.sh` (`PDVD_QL_KS_SAT_TOL`);
  - scripts `d34_{rail_calib,scan_items,margin}.py`;
  - `d34/`: `prereg.md`, `prereg_sha.txt`, `rail_calib.txt`, `gate/`, `mover_counts.txt`, `scan_r3/`;
  - doc 33 §7 (the accepted margin) and §8 (the correction).
- **Scratch** (not committed): `/home/xqian/tmp/p34/` (pin, sheets, logs, scores).

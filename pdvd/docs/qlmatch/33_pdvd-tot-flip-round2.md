# PDVD ToT flip campaign, round 2: an AI blind-scan target, the Q/L mechanism, and a lever ladder

**Status 2026-09-23. Round 2 of the doc 32 campaign. Nothing is flipped; every new knob is default OFF.**

**Goal (owner).** Flip PDVD light to the ToT saturation fill (docs 30–32). The owner does not hand-scan this round:
short tracks cannot be judged by hand, and the evidence that counts is geometry plus the light pattern. So an AI blind
scan with ten agents became the tuning target.

**Round 2 in one paragraph.**
- **Mechanism.** ToT changes Q/L matching through the railed PE that enters chi2/KS and the LASSO weights, not through
  the LASSO rows. The ToT-repaired rails read about 2× the prediction on bright cathode channels, so the right flash
  gets a worse KS and a larger LASSO penalty.
  - *Round 3 note:* the 2× is per bundle in shared flashes. On geometric crossers, ToT rails agree with the
    prediction (doc 34 §1).
  - A real gap exists: `fit_round2_shared` never skipped railed rows (§2). Closing it moves only 2–3 clusters.
- **The target.** 329 contested clusters, seen by two blind scanners each on a light-neutral sheet.
  - 74.5 % resolved (bar 60 %), 13 % of them as neutral ties.
  - Scanner calibration against the owner 90.5 % (bar 85 %).
  - Clean audit on all 10 transcripts.
- **Against that target no ToT arm is significantly better or worse than production:**
  - ToT alone: net −4 of 202 (p = 0.66);
  - the best lever, ToT + `lasso_weight_unrailed`: net +4 (p = 0.67);
  - dropping railed channels from chi2/KS (the pre-07-16 operating point) is worse in **both** lights: net −12.
- **The pre-registered flip rule (superiority, p < 0.05) FAILS.**
- **Owner's non-inferiority question (post hoc, §7).** On the held-out even half, the one-sided 95 % worst cases vs
  production are:
  - ToT + `lasso_weight_unrailed`: 6.4 clusters in 9 events (about 2 % of the 323 clusters the owner record confirms
    there);
  - ToT alone: 12.4 clusters.

  The margin is the owner's to set.

**Repro:**
```
# toolkit (apply-pointcloud) 5376f739: match sat_skip_round2_shared + lasso_weight_unrailed (default OFF)
wcbuild && ./wcb build --target=wcdoctest-match && ./build/match/wcdoctest-match          # 8/8
# pins: libpin_q33 = libpin_q32 + Match(sat_skip); libpin_q33b = + Match(both knobs)   (/home/xqian/tmp/p33/)
cd pdvd/docs/qlmatch/scripts
P=/home/xqian/tmp/p33/libpin_q33; PB=/home/xqian/tmp/p33/libpin_q33b
ARM=q33ctlp LIGHT=_g31off PIN=$P ./d31_ql_arms.sh                                  # gate g33pin
ARM=q33bctlp LIGHT=_g31off PIN=$PB ./d31_ql_arms.sh                                # gate g33bpin
ARM=q33cs LIGHT=_g31off PIN=$P ENVS="PDVD_QL_SAT_SKIP_R2=1" ./d31_ql_arms.sh       # L1 (ToT: q33ts, LIGHT=_q32ti)
ARM=q33cm LIGHT=_g31off PIN=$P ENVS="PDVD_QL_SAT_MASK_FIT=1" ./d31_ql_arms.sh      # L2 (ToT: q33tm)
ARM=q33cu LIGHT=_g31off PIN=$PB ENVS="PDVD_QL_LASSO_W_UNRAILED=1" ./d31_ql_arms.sh # L3 (ToT: q33tu)
W=/home/xqian/tmp/p31/wroot   # + symlinks 039252_<idx>_<arm> -> pdvd/work/039252_<idx>_<arm>
A=q32ti:T,q33ts:T,q33tm:T,q33tu:T,q33cs:C,q33cm:C,q33cu:C; TM=../d32/time_map_ctl_to_q32ti.json
python3 d33_lasso_trace.py --work-root $W --arms q31ctl:C,q32ti:T,q33cs:C,q33ts:T,q33cm:C,q33tm:T,q33cu:C,q33tu:T --time-map $TM --n 4 > ../d33/lasso_trace.txt
python3 d33_neutrality.py --work-root $W --arms $A --time-map $TM > ../d33/neutrality.txt
python3 d33_scan_items.py --work-root $W --arms $A --time-map $TM --sheet-dir /home/xqian/tmp/p33/scan/r2d --key /home/xqian/tmp/p33/scan_key_r2d/key.json
#   9 scanner agents (d33/RUBRIC.md, d33_record.py), then d33_audit.py <transcript> <wave>  -> d33/scan_r2/audit.txt
python3 d33_merge_record.py --key <key.json> --scan-root /home/xqian/tmp/p33/scan/r2d --out ../d33/scan_r2 --adjudicate-out ../d33/scan_r2/contradicted.json
python3 d33_adjudicate_items.py --key <key.json> --contradicted ../d33/scan_r2/contradicted.json --work-root $W --time-map $TM \
    --scan-root /home/xqian/tmp/p33/scan/r2d --wave 9 --out-key <key_adj.json>                       # agent 10
python3 d33_merge_record.py --key <key.json> --key-extra <key_adj.json> --scan-root ... --out ../d33/scan_r2 \
    --r1-key /home/xqian/tmp/p32/scan_key_r1q/key.json --r1-scan-root /home/xqian/tmp/p32/scan/r1q --time-map $TM
python3 d33_paired_score.py --work-root $W --ctl q31ctl --arms $A --time-map $TM --truth ../d33/scan_r2/truth.jsonl \
    --key <key.json> --key-extra <key_adj.json> --out ../d33/scan_r2
python3 ../../../ql_display/ql_agree_score.py --tag <arm> --work-root $W --truth-uid-map-tag keep \
    [--override-truth ../d33/scan_r2/override_target_<arm>.jsonl] --out-root /home/xqian/tmp/p33/scores_{owner,target}
```
(The key files are committed as `d33/scan_r2/key.json` and `key_adj.json`.)

## 1. What was asked
- **Owner, 2026-09-23.** Repeat the round-1 procedure to build a new hand-scan target, then fine-tune the algorithm
  against it. Use up to 10 scanning agents, write a new md, commit and push. The owner will not judge.
- **Round-1 plan (doc 32 §8):**
  - a consensus rule that does not square abstention;
  - a trace of what the LASSO sees;
  - a Q/L ladder tuned on odd events and confirmed on even ones.

## 2. Mechanism: what makes ToT and twoside match differently
**The round-2 fit gap (toolkit `5376f739`, `sat_skip_round2_shared`).**
- **What the code does.** Commit `d29d5f670` ("zero their LASSO rows in fit_round1/round2/joint fills") put the
  railed-row skip in `fit_round1`, `fit_round2` and `fit_round1_shared`. It left it out of `fit_round2_shared`, which
  already existed and is the round-2 joint solve PDVD runs.
  - With `saturation_mask_fit=false` (PDVD production), railed PE and its prediction enter the solve that sets
    `strength`.
  - That contradicts the header comment "LASSO rows stay zeroed at every fit site". It is an omission, not a choice.
- **Knob.** `sat_skip_round2_shared` adds the same skip.
  - It is **default OFF**; with it on, the log line `QLSATR2` counts the skipped rows (56 of 1826 on evt298567).
  - PDVD argument `ql_sat_skip_round2`, runner `PDVD_QL_SAT_SKIP_R2=1`.
- **Effect (`d33/mover_counts.txt`).** Only 2 clusters change on control light, and ToT-vs-control movers stay at
  100. **The gap is real, but it is not the mechanism.**
- **Erratum to doc 32 §4/§6**, which said railed rows are zeroed at every fit site. A pointer is added there, and the
  doc 32 readings stand.

**Where the difference really comes from** (`d33/lasso_trace.txt`, 4 clusters lost and 4 gained, in every arm):
- **On a mover's flash, 70–99 % of the active-channel PE sits on railed cathode channels.**
- **ToT raises that PE to its true value,** while the prediction stays where it was calibrated (on production light).
  - pred/meas on the same physical flash drops about 2× (e.g. 0.51 → 0.25, 0.70 → 0.10).
  - The KS, which includes the railed channels when `saturation_mask_fit=false`, worsens (e.g. 0.18 → 0.42,
    0.34 → 0.78).
- **The LASSO column weight rises through both terms.**
  - The base `|pred-meas|/meas` uses the all-channel flash total.
  - The round-2 term `δ_shape·nopdet·ks/λ` = 1.1·ks.
  - Example: 0.49 → 0.75 plus 0.27 from the KS, so the cluster moves to a flash it fits less badly.

**Levers tested** (mover counts vs `q31ctl` over the 18 events; no truth involved):

| lever | control light | ToT light | ToT vs control, same lever |
|---|---|---|---|
| none (production Q/L) | – | `q32ti` 100 | 100 |
| L1 `sat_skip_round2_shared` | `q33cs` 2 | `q33ts` 102 | 100 |
| L2 `saturation_mask_fit=true` (railed dropped from chi2/KS) | `q33cm` 250 | `q33tm` 252 | **40** |
| L3 `lasso_weight_unrailed` (new, weight base from unrailed channels) | `q33cu` 48 | `q33tu` 125 | 96 |

- **L2 is the switch that makes the light choice nearly irrelevant.** The chi2/KS railed term is the main route.
- **But L2 is also a large operating-point change:** 250 clusters move on production light.

`lasso_weight_unrailed` (same commit, default OFF): in the shared fits, the weight base's totals are summed over the
unrailed LASSO-row channels. A flash with no unrailed light keeps the legacy totals. PDVD argument
`ql_lasso_weight_unrailed`, runner `PDVD_QL_LASSO_W_UNRAILED=1`.

**Gates.**
- `d33/gate/g33pin.txt`: pin `libpin_q33`, knob off, is 120/120 calib-identical (`cmp`) to `q31ctl`.
- `d33/gate/g33bpin.txt`: `libpin_q33b`, both knobs off, is 120/120 identical.
- Compiled clustering config with the knobs off is identical to a HEAD overlay on 3 argument sets; the HEAD overlay
  rejects the new argument (rc 134), which proves it was HEAD. Each key appears once when on.
- `wcdoctest-match`: 8/8 test cases, 89 assertions.

## 3. The new target: design (pre-registered in `d33/prereg.md` before any sheet was rendered)
The round-1 failure was structural: consensus needed a committed verdict in both lights from two scanners. Round 2
changes the unit and the sheet, following the owner's own procedure (`ql_display/docs/ql-scan-criteria.md` §3):
- **Population.** The union of movers vs `q31ctl` over the **frozen** arm set (the 7 arms of §2): 329 long clusters,
  all of them scanned. A later arm needs a labelled top-up scan.
- **One light-neutral sheet per look** (`scripts/d33_blind_sheet.py`, example `pics/33_blind_sheet_example.png`).
  - **Source.** Everything is drawn from the control dump, and railed channels are drawn as hatched bars with **no
    value**, so no sheet shows which light an arm used.
  - **Neutrality proof, run before rendering** (`d33/neutrality.txt`, 15 504 bundles). These are the listed leaks:
    - the cluster's group (orange) is identical;
    - the prediction differs by at most 6 %, from the flash-time shift;
    - unrailed PE differs by more than 1 % on 6.8 % of bundles, and the railed flags on 1.6 %, because a few hits
      change flash membership.
  - **Stacked prediction.** The clusters that the control **and every** frozen arm match to that flash are shaded,
    and this cluster's prediction sits on top. This is the owner's "many clusters per flash".
  - **Zoom.** A zoom of the cluster with the anode and cathode planes.
  - **Letters** are shuffled per look.
  - **Candidates** (at most 5): every arm's pick, the owner's verdict flashes, then |log pred/meas|.
- **Rendering fixes, all made before any scanner saw a sheet.**
  - `r2`: 31 arm picks had no bundle in the control dump, so they were missing from the sheet. They are now drawn from
    the dump of an arm that has the bundle, control-light arms first.
  - `r2c`: zoom labels drew outside the axes.
  - The final render is `r2d`. The `r2`, `r2b` and `r2c` directories are unscanned scratch.
- **Answers.** A letter, `tie:X,Y` (equally good and better than the rest), `none`, or `unsure`; low confidence
  counts as unsure (`d33/RUBRIC.md`).
  - The rubric's only change after its first hash was the key-path line (`r2` → `r2d`), made before launch.
  - `d33/prereg_sha_before_render.txt` shows `prereg.md` unchanged since then.
- **Scanners.** Nine agents did about 81 sheets each: every cluster had 2 looks by different agents, plus 40
  owner-judged calibration clusters and 33 duplicates. A tenth agent adjudicated the 5 committed contradictions.
- **Merge rule.**
  - A mover is resolved when at least one look is committed and no committed look contradicts it; a pick inside a tie
    is compatible.
  - Ties are resolved but neutral.

## 4. The target: results (`d33/scan_r2/merge_report.txt`, `audit.txt`)

| pre-registered bar | required | round 2 | round 1 (doc 32) |
|---|---|---|---|
| calibration vs owner (committed) | ≥ 85 % | **90.5 %** (19/21): cathxa 13/13, gold 6/8 | 83.3 % |
| movers resolved | ≥ 60 % | **74.5 %** (245/329): 200 pos, 2 none, 43 tie; 84 unsure; 5 contradictions all adjudicated | 28 % |
| duplicates | reported | 11 compatible, 22 one side unsure, **0 contradict** | 6 / 14 / 0 |
| audit | clean | 10/10 transcripts: one Read and one recorder call per sheet, 0 reads outside the wave | clean |

- **Verdicts.** 731 in total: 367 letters at med/high, 72 ties at med, 4 `none`, and 288 low or unsure.
- **Where the gold disagreements are.** Both are in the owner's gold event, on owner **negatives** that the scanner
  picked. The scanners agree with cathxa (the AI + owner record) 13/13.
- **Reproducibility.** The round-1 verdicts re-merged under the new rule resolve 59 of 100 movers. Where both rounds
  resolve a positive, they agree on 31 and differ on 6 (84 %), across different sheets, scanners and rules.
- **What scanners found hard.** All ten reported the same three things:
  - predicted peaks railed at 2–3 candidates;
  - clusters that touch no plane;
  - a few stray points inflating "outside drift box" (they judged the main body in the zoom).

## 5. Scores against the target (`d33/scan_r2/paired_score.txt`)
**Primary.** Per resolved non-tie mover, an arm is right if its auto set is exactly the truth. Paired against
`q31ctl` (n: 91 odd events, 111 even, 202 all):

| arm | odd net (W–L) | even net (W–L) | all net | p (all) |
|---|---|---|---|---|
| `q32ti` ToT | 0 (10–10) | −4 (11–15) | −4 | 0.66 |
| `q33ts` ToT + L1 | +2 (12–10) | −4 (11–15) | −2 | 0.89 |
| `q33tm` ToT + L2 | −8 (23–31) | −4 (33–37) | −12 | 0.32 |
| `q33tu` ToT + L3 | +2 (12–10) | **+2 (14–12)** | **+4** | 0.67 |
| `q33cs` ctl + L1 | +2 | 0 | +2 | 0.50 |
| `q33cm` ctl + L2 | −7 | −5 | −12 | 0.32 |
| `q33cu` ctl + L3 | 0 | +2 | +2 | 0.77 |

**Secondary**, agree / phantom / missed on the resolved movers: `q31ctl` 90/101/110, `q32ti` 85/99/115, `q33tu`
91/92/109, `q33tm` 71/121/129.

**Owner record with the target replacing every scanned mover** (`secondary_owner_plus_target.txt`):

| arm | agree / phantom / missed |
|---|---|
| `q31ctl` | 634 / 163 / 195 |
| `q32ti` | 629 / 161 / 206 |
| `q33tu` | **635 / 154 / 199** |
| `q33tm` | 615 / 183 / 221 |

**Owner record alone** (`d33/owner_record_scores.txt`; the record taken on twoside light):

| arm | agree / phantom / missed |
|---|---|
| `q31ctl` | 683 / 97 / 158 |
| `q32ti` | 652 / 88 / 189 |
| `q33tu` | 653 / 82 / 188 |
| `q33tm` | 595 / 96 / 246 |

**Reading, by the pre-registered rule.**
- **Tuning half.** `q33ts` and `q33tu` tie at +2, and the pre-registered tie-break (fewer phantoms) also ties (38 each).
  The confirmation is therefore reported for both.
- **Confirmation (even half).**
  - `q33tu`: +2 with 26 discordant, above the power floor of 10, but p = 0.85. It **fails** item 4.
  - `q33ts`: −4. It **fails**.
- **Pre-registered verdict: FAIL, "not separable".** No ToT arm is better than production on the blind target.
- **Governing comparison.** The L3 twin on control light is +2 (even half) as well, so L3 is not ToT-specific
  evidence. Per the pre-registration it reads "offer the lever on its own".
- **L2 is worse in both lights.** It is −12 on the blind target (p = 0.32, consistent with the owner's 2026-07-16
  choice to keep railed channels in chi2/KS) and clearly worse on the owner record (595 vs 683 agreed).
- **The owner record's 31-cluster ToT deficit is not separable from zero on the neutral target at this n.** The
  point estimate still leans toward production: ToT alone is −4 of 202 [−8.6 %, +4.6 %], and on the owner+target
  record it misses 11 more (206 vs 195). The record's bias toward twoside light may explain part of the deficit;
  this round cannot say how much.
- **Both arms are wrong on most contested clusters.** They are right on about 41 % of the resolved movers. Contested
  clusters are the hard ones, and that ceiling is shared by every arm.

## 6. Doc 11 / doc 12
Not rerun: no arm passed the pre-registered rule. A flip under §7 reruns both first.

## 7. Owner's non-inferiority reading (post hoc; asked 2026-09-23 after the results)
**The owner's question.** If ToT is not clearly better, is "not worse" enough? ToT is the better reconstruction of
saturated light (docs 30–32).

**Answer.**
- Yes, that is a legitimate criterion: the light-level case for ToT stands on its own, and Q/L need only not degrade.
- It is a **change of rule after the data**, recorded here as the owner's decision. It needs a margin.
- **The held-out even half governs.** The tuning half tied (`q33ts` and `q33tu` both +2, phantoms 38 each), so any
  preference between those two already uses even-half data. That is stated here, not hidden.
- **Numbers.** Paired against `q31ctl` on the resolved non-tie movers of the 9 even events (n = 111; one-sided 95 %
  lower bound from the discordant pairs):

| arm | even half: wins–losses | net | worst case (one-sided 95 %) |
|---|---|---|---|
| `q32ti` ToT | 11–15 | −4 | −12.4 clusters |
| `q33ts` ToT + L1 | 11–15 | −4 | −12.4 clusters |
| `q33tu` ToT + L3 | 14–12 | +2 | **−6.4 clusters** |

- **Scale.** The owner record confirms 323 matched long clusters in those 9 events. A worst case of 6.4 clusters is
  about 2 % of them, and 12.4 clusters about 4 %.
- **Descriptive only** (all events, including the tuning half): `q33tu` +4 (26–22), `q32ti` −4 (21–25).
- **Why absolute clusters, not "% of movers".** The 202 movers are the union over every frozen arm, and many move
  only under L2. A percentage on that denominator changes with the arm set.
- **The owner sets the margin.** The margin should be stated in clusters per event set (or as a share of matched
  clusters) before round 3 scores anything.
- **Margin accepted by the owner (2026-09-23), before any round-3 scoring** (`d34/prereg.md`):
  - on the held-out even half, the one-sided 95 % lower bound of the paired net vs `q31ctl` must be ≥ −6.46
    clusters (2 % of 323);
  - **and** phantoms on the owner+target record must be ≤ `q31ctl`'s.
  - `q33tu` meets it **post hoc**: bound −6.39 and phantoms 154 vs 163. That is a margin of 0.06 of a cluster, on an
    even half already read, so round 3 treats `q33tu` as the reference to beat, not as a confirmed candidate.
- **Most favourable configuration so far:** ToT + int_samples + `lasso_weight_unrailed` (`q33tu`).
  - It has the best held-out net.
  - It has the fewest phantoms on both combined records (154 vs 163; 82 vs 97).
  - Its missed count is within +4 on the owner+target record.
  - Its control twin is also +2, so the lever is not ToT-specific.
- **Before any flip** (round 3), in this order:
  1. doc 11 §6 and doc 12 on the chosen arm;
  2. the 102 other events (runs 39253 / 39349) for crashes and match-rate tails;
  3. a flip-equivalence gate;
  4. the owner's go.

## 8. Round 3: tune the ToT case (the owner's second question)
The trace names the lever: a truly measured bright rail exceeds a prediction calibrated on production light.

1. **Recalibrate the prediction on ToT light.**
   - Refit QtoL (doc 12 `fit_qtol_crossers.py`) and the cathode-channel visibility scale on `_q32ti` light.
   - Production's calibration saw rails at their clipped or wrapped values, so it is biased low on the channels ToT
     fixes.
   - This is the one lever that should let ToT's truer PE help rather than hurt.

   **Correction (round 3, doc 34 §1).** The premise is wrong. Doc 12's factors exclude railed channels from both
   sums, and unrailed PE is the same in both lights, so a refit on ToT light returns the same numbers.
   - Measured on geometric crossers, ToT rails **agree** with the prediction: median ratio 1.04 [0.69, 1.35]
     (`d34/rail_calib.txt`).
   - The "about 2×" in §2 is per bundle in shared flashes, where the other clusters' light is not in that bundle's
     prediction. It is not a calibration offset.
2. **ToT-only railed-channel error.** Extend `chi2_sat_inflate` to also enter the KS / LASSO weight on railed
   channels, as a new default-OFF knob. Railed channels then stay in (L2 showed dropping them is worse) but with an
   honest error on a repaired value.
3. **L3 as the base** of every ToT ladder arm.
4. **Scanning.** Each new arm adds its own movers, scanned by the same procedure: the same rubric and light-neutral
   sheet, with a top-up wave labelled as such. The 84 unresolved movers stay unresolved; they are the hard 26 %.
5. **Split and bars.** The same odd/even split, and the pre-registered bars plus the owner's non-inferiority margin,
   stated before the arms run.

## 9. Files
- **toolkit** `5376f739`:
  - `match/{inc,src}/QLMatching.*` (`sat_skip_round2_shared`, `lasso_weight_unrailed`);
  - `match/test/doctest_qlmatching_config.cxx`;
  - `cfg/.../protodunevd/{qlmatching,wct-clustering}.jsonnet`.
- **wcp:**
  - `run_clus_evt.sh` (`PDVD_QL_SAT_SKIP_R2`, `PDVD_QL_LASSO_W_UNRAILED`);
  - scripts `d33_{scan_items,blind_sheet,record,audit,merge_record,adjudicate_items,paired_score,lasso_trace,neutrality}.py`;
  - `d33/`: `prereg.md`, `prereg_sha_before_render.txt`, `RUBRIC.md`, `gate/`, `lasso_trace.txt`, `neutrality.txt`,
    `mover_counts.txt`, `owner_record_scores.txt`;
  - `d33/scan_r2/`: keys, verdicts and INDEX per wave, `audit.txt`, `merge_report.txt`, `truth.jsonl`,
    `contradicted.json`, `paired_score.txt`, `override_target_*.jsonl`, `secondary_owner_plus_target.txt`;
  - `pics/33_blind_sheet_example.png`;
  - doc 32 erratum pointer.
- **Scratch** (not committed): sheets `/home/xqian/tmp/p33/scan/r2d/`, pins `/home/xqian/tmp/p33/libpin_q33{,b}`, arm
  logs `/home/xqian/tmp/p31/arm_q33*`.

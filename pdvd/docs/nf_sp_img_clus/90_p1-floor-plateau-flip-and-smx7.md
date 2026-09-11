# 90 — P1's energy floor at 3 MeV and the plateau edge at 2.0, flipped; and the owner's blind re-judge of doc 89's 22 (smx7)

Doc 89 ended with two next steps. The owner said go ahead with the first: flip the 0-FP bundle, which lowers the energy floor of P1's Michel override (the rule that a found Michel of enough size overrides the dQ/dx shape tests) and widens the plateau window. The owner chose to do the second, the blind re-judge of the 22 medium- and low-confidence agent verdicts, themselves, on :5018. Both are here.

**Status (2026-09-11): FLIPPED in PDVD production.** The keys are `topology_michel_ke_min: 3.0` and `plateau_mip_hi: 2.0` in `pdvd/wct-pr-perevt.jsonnet`; PDHD stays OFF. No C++ changed.

- **The flip** (§1–§4):
  - it did exactly what the offline twin, written before the arms, said it would: 5 missed stoppers gained, one MESSY item flips, and two boundary items lose one of their two reject bits;
  - on doc 89's record: `is_stm` 233 / 7 / 55 → 238 / 7 / 50 on the 576 judged items, 0 new FP;
  - `michel_found` identical;
  - 588 of 596 candidates bit-identical to production;
  - the Bee zips differ on one event only.
  - The confirmation arm on the flipped file: **§4.2**.
- **The re-judge** (§5–§6):
  - Of the 22 record stoppers, the owner kept **18** as stoppers: 13 with a Michel, 5 without. **4** are through-going.
  - Of the 8 through-going controls mixed in blind, 1 is a stopper.
  - Folded into a new record, `pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json`, production reads `is_stm` 233 / 7 / 52 and, with the flip, **238 / 7 / 47**. `michel_found` is 144 / 12 / 25: the owner found 3 more Michels than the agent had.
- **Why production sees nothing at the owner's 13 Michels** (§6): it builds no Michel object on any of them. It finds no stop arm on 11 and only a non-Michel ("other") arm on 2, and records no local piece and no unfitted dots. Two mechanisms:
  - **the fit runs through the Michel on 5.** The owner's pin sits 3.6–7.2 cm before the fit end; production's Bragg-peak anchor moved the stop 0–2.8 cm, since its search window is 3 cm;
  - **unfitted charge on 8.** The pin stays at the fit end, and 3–35 of the muon cluster's own points lie more than 2 cm off the fit.

  Off-fit charge alone does not separate the owner's Michels from the rest: 3 of 5 STM_ONLY and 3 of 4 THRU items have it too.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts; S=$IMG/pdvd/docs/scan
export STM_SCAN_RECORD=$S/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_verdicts.json
# The record each script demands: d90_twin, d90_build_smx7 and d90_score_smx7 refuse anything but the
# smx6 record (set here; d90_gates.sh sets it itself).  d90_offfit refuses anything but the smx7 record
# that d90_score_smx7 writes; it is set inline on its line below.
# the twin, before any arm (pred.txt 14:10:16; arms launched 14:10:21)
python3 $X/d90_twin.py --prep /home/xqian/tmp/p89/prep_p88vprod --log-arm p88vprod --json /home/xqian/tmp/p90/twin.json
# the arms: no C++ this round; doc 88's full pin libpin_p88 (the one p88vprod ran on); bare production + TLA
WAVE=1 nohup bash $X/d90_arms.sh > /home/xqian/tmp/p90/arms_wave1.log 2>&1 < /dev/null & disown   # p90vb p90vk p90vp
bash $X/d90_gates.sh > /home/xqian/tmp/p90/gates.log 2>&1              # sections 0, 2-4
ONLY1=1 bash $X/d90_gates.sh > /home/xqian/tmp/p90/gates_s1.log 2>&1   # section 1, after the no-candidate fix
# the flip, its proofs, every other live job, the confirmation arm
PRE_EVT=/home/xqian/tmp/p90/pre_wct-pr-perevt.jsonnet bash $X/d90_proofs.sh      # run from $IMG/pdvd
(cd $IMG/abtest && ./compile_all_cfg.sh /home/xqian/tmp/p90/cfg_after && ./cmp_cfg.sh /home/xqian/tmp/p90/cfg_before /home/xqian/tmp/p90/cfg_after)
WAVE=2 nohup bash $X/d90_arms.sh > /home/xqian/tmp/p90/arms_wave2.log 2>&1 < /dev/null & disown   # p90vprod
PROD=p90vb ARMS=p90vprod bash $X/d90_gates.sh > /home/xqian/tmp/p90/gates_confirm.log 2>&1
# smx7: build (blind, production payloads), serve, score + fold, the off-fit reading
cd $IMG/pdhd/stm_michel_scan && python3 $X/d90_build_smx7.py --census /home/xqian/tmp/p89/census.json \
    --prep /home/xqian/tmp/p89/prep_p88vprod --outprep $PWD/prep-pdvd-smx7 --sheet $S/pdvd_stm_michel_smx7_sheet.tsv \
    --questions $S/pdvd_stm_michel_smx7_questions.json --key $S/pdvd_stm_michel_smx7_key.tsv
./serve_stm_michel_scan.sh 5018 --det pdvd --scan-tag smx7 --manifest $S/pdvd_stm_michel_smx7_sheet.tsv \
    --prepdir $PWD/prep-pdvd-smx7 --questions $S/pdvd_stm_michel_smx7_questions.json
python3 $X/d90_score_smx7.py --labels $S/pdvd_stm_michel_smx7_labels.json --key $S/pdvd_stm_michel_smx7_key.tsv \
    --arm p88vprod:/home/xqian/tmp/p89/prep_p88vprod --arm p90vb:/home/xqian/tmp/p90/prep_p90vb \
    --write-merged $S/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json
STM_SCAN_RECORD=$S/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json python3 $X/d90_offfit.py \
    --prep /home/xqian/tmp/p89/prep_p88vprod --labels $S/pdvd_stm_michel_smx7_labels.json --key $S/pdvd_stm_michel_smx7_key.tsv
```

Outputs: `/home/xqian/tmp/p90/{twin,pred,gates,gates_s1,proofs,cfg_cmp,score_smx7,offfit_smx7}.txt|log`.

## 1. What was flipped, and why it was the owner's call

These are doc 89 §5's two settings, both existing knobs:
- **`topology_michel_ke_min`: 10 → 3 MeV.** This is P1's energy floor (doc 70), with one consumer, at `CheckSTM_Michel.cxx:3989`, applied last on recorded inputs. It is a new key: before, it was absent from the compiled config and took the C++ default of 10.
- **`plateau_mip_hi`: 1.6 → 2.0.** This is the plateau window's upper edge, a value edit. It is read at `:2729` (the anchored profile) and `:2813` (doc 75's geometric re-read), and by nothing outside `CheckSTM_Michel`.

Doc 89 §5.1 named what the owner accepted with the go:
- **the threshold moved on the record it is graded on;**
- **the docs 72 / 84 invariant no longer holds.** That invariant said the moved-stop Michel veto (T2c) and its exemptions cannot move `is_stm` (`CheckSTM_Michel.cxx:664–678`). A T2c change must now be graded on `is_stm` too;
- **the energy is the fit-based `michel_ke_best`,** which at 3–10 MeV inherits doc 78 item 9's fit dependence. The 3 cm length floor, which is what protects purity, is unchanged.

## 2. The twin, written before the arms

`scripts/d90_twin.py` applies the rule offline to every one of production's 585 candidates, judged or not. It reproduces production itself with 0 mismatches.

Two things it cannot see are listed from production's own DEBUG lines (`anchor_geo_fallback`) rather than guessed:
- the geometric re-read can newly stand only where its sole failure was the plateau test: 6 candidates;
- where the re-read stood, the anchored reading can take over only if its sole failure was the plateau: 1 candidate, `039349_76/75`.

| # | prediction (`/home/xqian/tmp/p90/pred.txt`) | result |
|---|---|---|
| P1 | `p90vb`: `is_stm` movers exactly `039252_6/106`, `039349_19/52`, `039349_48/21`, `039349_63/55`, `039349_70/61` and the MESSY `039349_77/52`; bits-only movers `039349_27/41`, `039349_35/30`; 233/7/55 → 238/7/50, 0 new FP | **held exactly** (0 missing, 0 unpredicted) |
| P2 | none of the 6 possible re-read gains fires, and `039349_76/75` does not move: their anchored plateaus are all on the low side (0.21–0.50 MIP) | **held**: none moved |
| P3 | `p90vk` (the energy floor alone): the same minus `039252_6/106`; 237/7/51 | **held exactly** |
| P4 | `p90vp` (the plateau edge alone): `039252_6/106` only; 234/7/54 | **held exactly** |
| P5 | `michel_found` identical on every candidate of all three arms | **held** |
| P6 | nothing outside the movers moves; zips, calib and T_rec_charge move only on the events of `is_stm` 0 → 1 movers, through their un-withheld capture gammas | **held**. The one such event is `039349_77`: the capture gamma belongs to the MESSY mover `039349_77/52`, 2 role-5 points. |

## 3. Gates (arms `p90vb p90vk p90vp` vs `p88vprod`; `/home/xqian/tmp/p90/gates.log`, `gates_s1.log`)

| gate | result |
|---|---|
| completeness, pin | 120 / 120 events per arm, 0 loader deaths; `libpin_p88` (572 libraries) md5 identical before and after |
| candidates bit-identical on all 147 shared branches (`d51g_branch_census`) | `p90vb` 588 / 596, `p90vk` 589 / 596, `p90vp` 595 / 596 |
| the branches that moved (`p90vb`) | `reject_bits` 8, `topology_cleared_bits` 7, `is_stm` 6, and the `stop_gamma_*` / `n_stop_gammas*` fields on 1 candidate (`039349_77/52`) |
| trees (`p90vb`) | `T_bad_ch`, `T_cluster`, `T_proj`, `T_proj_data`, `Trun` identical on all 120 events. `T_stm_michel` differs on the 8 mover events, and `T_stm_michel_pts`, `T_rec_charge` and calib on `039349_77` only. `p90vp`: `T_stm_michel` on `039252_6` only. |
| Bee zips (member content) | `p90vb` and `p90vk`: 1 event differs, `039349_77`, an `is_stm` mover event. `p90vp`: 0. |
| census, smx6 record (`census_score`, with a candidate) | `p90vb` 238 / 7 / 40 (eff 0.856, F1 0.910); `michel_found` 144 / 12 / 17 unchanged. `census_score --check` 0 of 14 differ. |

- **A gate-script bug.** Section 1 of the first gate run crashed: an event with no candidate writes no `T_stm_michel`, and the fork had dropped doc 88's skip. It was fixed and section 1 re-run alone (`ONLY1=1`); sections 0 and 2–4 were unaffected.
- **A prediction I made in conversation, not in `pred.txt`.** I expected the capture gamma to be on one of the four energy-floor stoppers. It is on the MESSY `039349_77/52`.

## 4. The flip

### 4.1 Proofs (`scripts/d90_proofs.sh`, `/home/xqian/tmp/p90/proofs.txt`)

| proof | result |
|---|---|
| A: PRE + the arm's TLA vs POST | 0 lines |
| B: POST with both forced back (`topology_michel_ke_min:10.0, plateau_mip_hi:1.6`) vs PRE | one line, `"topology_michel_ke_min": 10`, present vs absent. Its value equals the C++ initializer `m_topology_michel_ke_min{10.0}`, so it is inert (the docs 58 / 61 / 82 form). |
| C: PRE vs POST | exactly the two keys: `plateau_mip_hi` 1.6 → 2, `topology_michel_ke_min` absent → 3 |
| D: PDHD | `pdhd/wct-pr-perevt.jsonnet` unchanged against git HEAD; no `topology_michel_ke_min`; `plateau_mip_hi` 1.6 |
| the other 16 live jobs (SBND / PDHD / PDVD clustering, imaging, NF+SP, simulation) | `compile_all_cfg.sh` before and after, then `cmp_cfg.sh`: every job NORMDIFF 0, **OVERALL PASS** (`/home/xqian/tmp/p90/cfg_{before,after}`, `cfg_cmp.txt`) |

### 4.2 Confirmation arm (`p90vprod`: the flipped file, no TLA)

`PROD=p90vb ARMS=p90vprod bash scripts/d90_gates.sh` (`/home/xqian/tmp/p90/gates_confirm.log`; launched 14:33:23, done 14:42:23, rc 0, the same pin, md5 identical before and after).

| comparison, `p90vprod` vs `p90vb` | result |
|---|---|
| candidates bit-identical on all 147 branches | **596 / 596**; point geometry 596 / 596; 0 movers, 0 `is_stm` or `michel_found` flips |
| trees, 120 events | all 8 identical on every event, including `T_rec_charge`, `T_stm_michel` and `T_stm_michel_pts` |
| Bee zips / calib | 0 events differ / 119 same, 0 differ |
| census, smx6 record | 238 / 7 / 40 with a candidate, 238 / 7 / 50 on all judged items; `census_score --check` 0 of 14 differ |

The flipped file runs exactly what `p90vb` ran. PDVD production is now `p90vprod`; its payloads are `/home/xqian/tmp/p90/prep_p90vprod`.

## 5. smx7: the owner's blind re-judge

### 5.1 The set (`scripts/d90_build_smx7.py`)

- **The 22 items** are doc 89's (b1): record stoppers that production rejects on the shape tests alone, with no Michel object, on a medium (20) or low (2) confidence verdict from doc 55's tranche 2. Tranche 2 was an agent scan (doc 55 §13: "9 waves of 5 agents").
- **A set of record stoppers alone would give the answer away**, so 8 controls are mixed in. They are record-THRU items at medium confidence from the same scan with the same production reading (`is_stm` 0, `michel_found` 0, shape bits only), drawn with seed 90 from a pool of 97.
- **The build asserts that all 30 read alike.** The order is shuffled with seed 90.
- **What the panel hides:** the record's verdict, its evidence and its pin. The group is kept in `pdvd_stm_michel_smx7_key.tsv`.
- **Payloads** are production's (`p88vprod`), taken before the flip. The flip moves none of the 30 (§2).
- **Serving:** on :5018 at the owner's request. A headless check showed the 30 items and the panel with 0 errors, and the owner's forward reached it (`[::1]` connections). The owner finished at 14:28. The labels are committed as `pdvd_stm_michel_smx7_labels.json` (md5 `97f51eac`); the live file was unchanged when the server stopped.

### 5.2 The owner's answers (`scripts/d90_score_smx7.py`, `/home/xqian/tmp/p90/score_smx7.txt`)

| group | n | record → owner | the stopper / THRU call held |
|---|---:|---|---|
| (b1) record stoppers | 22 | STM_MICHEL → STM_MICHEL 8, STM_MICHEL → THRU 2, STM_ONLY → STM_MICHEL 5, STM_ONLY → STM_ONLY 5, STM_ONLY → THRU 2 | **18 of 22**; THRU: `039349_82/25`, `039349_81/25`, `039349_32/63`, `039349_26/34` |
| controls, record THRU | 8 | THRU → THRU 7, THRU → STM_ONLY 1 | **7 of 8**; stopper: `039349_38/33` |

- **The agent's stopper calls held on 18 of the 22, and its THRU calls on 7 of the 8: 5 overturned in 30.**
  - This measures this population only. The 22 were selected because production rejects them on the shape bits with no Michel object, and the controls because they read the same.
  - It is not a rate for agent verdicts in general, and with n = 30 it is not a constant to carry into later rounds.
  - It is not comparable with the "about a third" doc 89 §4 cited from smx3 / smx4 either. Those sets were drawn differently: the hardest items by construction.
- **Michels at the stop: 13 by the owner** against 10 on the record. Five of the agent's STM_ONLY items carry a Michel; two of its STM_MICHEL items are through-going.
- **The owner moved the pin on 5 items, all STM_MICHEL:**

  | item | owner's pin rr (cm) | production's anchor shift (cm) | the agent's reading |
  |---|---:|---:|---|
  | `039349_71/37` | 5.4 | 0.0 | `pin_rr` 6.5 |
  | `039349_29/40` | 3.6 | 0.0 | — |
  | `039349_44/28` | 4.3 | 0.0 | 4.3 |
  | `039349_3/47` | 7.2 | 2.8 | — |
  | `039252_12/123` | 7.2 | 0.0 | — |

  On the agent's other six `pin_rr` items the owner left the pin at the fit end.

### 5.3 The fold, and the census on the new record

The rules are those of smx4–smx6: the owner's verdict, Michel kind and pin; confidence "owner"; the old `pin_rr` carried. The stopper / Michel / judged class changes on 10 of 601 records.

| arm | record | population | `is_stm` TP / FP / FN | `michel_found` TP / FP / FN |
|---|---|---|---|---|
| production `p88vprod` | smx6 | all judged (576) | 233 / 7 / 55 (eff 0.809) | 144 / 12 / 22 |
| production `p88vprod` | **smx7** | all judged | **233 / 7 / 52** (eff 0.818) | **144 / 12 / 25** |
| flip `p90vb` | smx6 | all judged | 238 / 7 / 50 (eff 0.826) | 144 / 12 / 22 |
| flip `p90vb` | **smx7** | all judged | **238 / 7 / 47** (eff 0.835, F1 0.898) | 144 / 12 / 25 |
| flip `p90vb` | smx7 | with a candidate (546) | 238 / 7 / 37 (eff 0.865, F1 0.915) | 144 / 12 / 20 |

**Doc 89's partition on the new record, with the flip.** 47 missed = 10 (the tagger's) + 8 (the owner's fiducial / continuation preference) + 6 (Michel under the 3 cm floor) + 4 (high confidence, no lever) + **18** (b1, now owner-confirmed: 13 with a Michel, 5 without) + **1** (`039349_38/33`, `shape_flat`, no Michel). The 5 of (a1) are gained by the flip.

**Doc 89's census script stays on the smx6 record.** `d89_miss_census.py` refuses any record but the smx6 one by design, so doc 89's numbers are frozen there, and the table above supersedes them on the smx7 record. The guard is not changed: doc 89's script is its reproducibility record. A later census on the smx7 record needs a fork.

## 6. Why production sees nothing at the owner's 13 Michels (`scripts/d90_offfit.py`, `/home/xqian/tmp/p90/offfit_smx7.txt`)

Fork of doc 86's off-fit reading, run on all 22 (b1) items grouped by the owner's call. Columns:
- **off**: the muon cluster's own points within 10 cm of the fit end that lie more than 2 cm off the fit;
- **past**: own points more than 0.5 cm past the fit end along its last 5 cm;
- **other10**: points of other clusters within 10 cm;
- **dead**: planes with a dead channel within 15 channels of the fit end.

| owner's call | n | off-fit | own points past the end (median) | other clusters within 10 cm (median) | owner pin moved | a dead plane at the end |
|---|---:|---:|---:|---:|---:|---:|
| STM + MICHEL | 13 | 9 | 2 | 0 | **5** | 9 |
| STM, no Michel | 5 | 3 | 3 | 0 | 0 | 4 |
| THRU | 4 | 3 | 6 | 0 | 0 | 4 |

- **What production records on all 13.**
  - `michel_found` 0 and `n_local_pieces` 0.
  - No stop arm on 11; one non-Michel arm on `039252_5/73` and `039252_9/101`.
  - No unfitted-dot energy.
  - No other cluster near the end, except `039253_2/13` (7 points) and `039349_0/28` (11).
- **Two mechanisms on the 13.**
  - **The fit runs through the Michel: 5 items,** those of §5.2's pin table. Four of them have no own-cluster point more than 2 cm off the fit. The Michel is the fit's own last 3.6–7.2 cm; the 3 cm Bragg-peak anchor (doc 68) cannot reach back that far. This is doc 86's `fit-through` class and doc 78 item 8's population.
  - **Unfitted charge off the fit: the other 8 items.** The pin stays at the fit end, and 3–35 own points lie off the fit: `039349_51/29` 32, `039349_0/28` 35, `039252_5/73` 34, `039349_64/52` 27, `039252_8/102` 18, `039252_9/101` 12, `039349_81/44` 9, `039253_2/13` 3. This is doc 86's `unfitted` class.
  - **Doc 88's floored residual keep does not reach either class.** It keeps a residual PR fitted and dropped within 5 cm of the stop, with ≥ 5 terminals and ≥ 5 cm.
    - Production kept or floored no residual on any of the 13: `n_kept_near_stop_main` and `_comp` 0, no `floored` log line, `n_local_pieces` 0.
    - The production logs show a `pr54 isolated-residual drop` for the cluster on only 3 of them (`039252_5/73` 2, `039349_64/52` 1, `039349_71/37` 2). None was inside the keep's radius, since a residual there would have been kept or floored.
    - On the other 10, PR fitted no residual on the cluster at all.
- **What does not separate the owner's Michels from the rest.**
  - Off-fit charge sits on 3 of the 5 STM_ONLY items (`039349_16/70` 44, `039349_34/48` 61, `039349_48/63` 7) and 3 of the 4 THRU items. So "off-fit charge near the stop means a Michel" would not be pure on this sample.
  - Dead planes at the end (9/13, 4/5, 4/4) do not separate either, as in doc 56 §5.
- **What does separate, with a caveat.** The owner moved the pin only in the Michel group, 5 of 13 against 0 of 9.
  - That is an asymmetry in what the owner did, not a measurement. A pin left at the fit end is not evidence that the fit end is right, so the other 17 are not a clean negative control for overshoot.
  - The independent corroboration is this section's own reading: 4 of the 5 pinned items have no own-cluster point more than 2 cm off the fit within 10 cm of its end; 039252_12/123 has 5. On those four the Michel sits inside the fit, not beside it.

## 7. What the record cannot grade

- The MESSY `039349_77/52`, now `is_stm` 1.
- The 16 unjudged candidates doc 79 added, and the unjudged movers of earlier flips (`039253_8/81`, `039349_64/80`, doc 88 §9.5).
- Whether the 13 owner Michels are recoverable without new false Michels. §6 says the off-fit signal alone is not pure; that needs a rule and an arm.

## 8. Next, ranked

1. **The 5 fit-through Michels (§6):** measure first.
   - The owner's pins (3.6–7.2 cm) lie beyond the 3 cm anchor search.
   - Doc 68's 10 cm anchor cost is_stm purity (9 → 22 FP, on smx1a). The question is whether a wider search restricted to candidates the owner's rule already favours keeps purity: e.g. only where the fit's tail past the peak reads below the plateau, as in doc 78 item 8.
   - Size it offline on the new record (the 5 targets against the smx7 controls and the record's THRU) before any build. Doc 78 item 8's blind re-judge of its 10 fires is part of the same question.
2. **The 8 unfitted Michels:** find a discriminator before any rule. Charge amount, direction relative to the muon and compactness, measured against the 6 non-Michel items that also carry off-fit charge.
3. **Doc 88 §9.5 item 2:** grade the PDVD flips of docs 83–90 on PDHD's smx18 record.
4. **Doc 78 item 9**, the Michel charge energy.
5. **Housekeeping:** `039349_77/52` (MESSY, now a stopper) and `039349_38/33` (a new owner stopper the chain misses on `shape_flat`) go into the next scan tranche's notes.

**Recommendation: 1.** It is the only owner-graded population whose mechanism is corroborated two ways: the owner's pins at 3.6–7.2 cm, and no off-fit charge on 4 of the 5, so the Michel sits inside the fit. It can be sized offline without new C++. The sizing must not use the un-pinned items as negatives (§6).

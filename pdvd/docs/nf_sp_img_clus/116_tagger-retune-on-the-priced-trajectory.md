# 116 — the STM / Michel tagger retune on the priced trajectory: the un-run PDVD arm, the nature of the tag regressions, a pre-registered tagger ladder, the blind fold, and the flip decision

**Status: NOT FLIPPED. No tagger level passes the frozen rule on both detectors (PDVD Michel purity −0.037…−0.039 and `is_stm` purity −0.021 at every level; PDHD passes all four metrics at T / R1 after the blind fold but fails the split-half guard). Both Steiner knobs and the tagger keys stay at their production values; no C++ change; no toolkit commit. PDHD 61 / PDVD 120 events.**

Owner request (2026-09-18, after doc 115 round 2): (1) run the one arm round 2 did not — the `tree+path` pricing at
α 0.5 *without* `prefer3` on PDVD — to settle the v3 / U2 / U1b misses; (2) remove the output-less `d115?bwa` /
`d115?bwb` dirs; (3) study the nature of the PDHD and PDVD tag regressions on the α 0.5 trajectories, retune the STM
and Michel taggers on both, and if the validation passes turn on the knobs for both the trajectory and the taggers.
Owner decisions taken before any new arm ran: PDVD carries the no-`prefer3` arm into the retune and the flip if it
passes every round-2 clause (a per-detector configuration is acceptable), else `prefer3` + α 0.5; the unlabelled
candidates are resolved by a doc-103-style blind scan run by Opus agents before the grade is read.

**In one screen.** (1) The un-run PDVD arm (`tree+path` α 0.5 without `prefer3`) fixes the v3 spot (0.54 cm) but
fails U1b outright (+4.5 %: the pricing moves length into FIT, 3.0 → 6.2 m), U2 (−35.7 %), the paired shape count and
the Bee holes, and costs the tags twice as much (Michel −0.07) — so by the owner's rule PDVD stays on `prefer3` + α 0.5,
the same trajectory as PDHD (sec 1). (2) The 362 empty round-1 dirs are gone (sec 7). (3) The tag regressions are
boundary flips, not a scale shift: the tagger's inputs are unchanged at the median on the common candidates, and the
20 % of candidates that change tags sit at a 2 cm stop displacement (sec 2). The one-way channels have keys — the
proton veto un-vetoed by `proton_muon_guard`, the Michel arm at the stop re-classified below `michel_min_kink_deg` 30
or over `michel_max_len_cm` 25, and the range-energy veto — and were sized on their eligible population before a
three-rung ladder was frozen (sec 3). The ladder does what the sizing said (sec 4): R2 (`proton_muon_guard` +
`michel_min_kink_deg` 20 + `michel_max_len_cm` 30) restores both Michel efficiencies to production (PDVD TP 171 → 173,
PDHD 73 → 77) for one false positive each; R3's range-energy floor admits false positives faster than true ones. A
blind Opus scan labelled 44 of the 76 unlabelled candidates (the rest are below the display's own filters), V2
agreement 7 / 7, every transcript audited clean (sec 5). **After the fold, PDHD passes all four metrics on the new
trajectory with the production tagger** (`is_stm` −0.006 / +0.032, Michel −0.017 / +0.019: doc 115's PDHD
"regression" was a labelling gap plus boundary noise) but fails the rule's split-half guard; **PDVD fails purity at
every rung** (`is_stm` FP 5 → 11, Michel FP 11 → 18–19, both sides labelled) while R2 brings its efficiencies back.
Under the rule nothing is flipped (sec 6). The remaining cost is the false-positive side on PDVD — ten THRU-labelled
clusters the new trajectory tags as stoppers — and the owner's look at those, or an explicit override as in doc 108,
is the next step.

## 0. Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs; P=/home/xqian/tmp/d115/libpin_d115
# Part 1 -- the un-run PDVD arm (figs/115r2_pred_amend1.txt; its sha appended to 115r2_pred.sha256)
(cd $F && sha256sum -c 115r2_pred.sha256)
bash $S/d102_compile_pr.sh pdvd d116_bwp05 -S steiner_base_weight_blank_alpha=0.5 -S "steiner_base_weight_scope='tree+path'"
python3 $S/d116_cfgdiff.py /home/xqian/tmp/d102/cfg/d115knoboff_pdvd.json /home/xqian/tmp/d102/cfg/d116_bwp05_pdvd.json --expect CreateSteinerGraph/pr/base_weight_blank_alpha,CreateSteinerGraph/pr/base_weight_scope,CreateSteinerGraph/prrefresh/base_weight_blank_alpha,CreateSteinerGraph/prrefresh/base_weight_scope   # figs/116_gate_config.txt
(cd $S && DET=pdvd SERIES=d115 PRED=115r2_pred.sha256 LEVELS=bwp05 JOBS=8 bash d116_run_levels.sh)     # arm d115vbwp05, 120 evt
bash /home/xqian/tmp/d116/post_bwp05.sh          # = the six clause scripts of d115r2_post_arms.sh for L=bwp05 (115r2_ prefix) + d113_grade -> figs/116_grade_pdvd_bwp05.txt
python3 $S/d113_spot_figs.py --out $F/116_spot --logd-root /home/xqian/tmp/d115 --spots v1,v2,v3 --arms pdvd=d115voff:production,d115vp3bwp05:prefer3+alpha0.5+path,d115vbwp05:alpha0.5+path
python3 $S/d115r2_verdict.py --levels BWP05:bwp05:0.5:115r2_,P3BWP05:p3bwp05:0.5:115r2_ --spot 116_spot.tsv > $F/116_verdict_bwp05.txt

# Part 2 -- the cleanup (figs/116_cleanup.txt): the verified list /home/xqian/tmp/d116/rm_list.txt, then xargs -a <list> rm -r

# sec 2 -- the regression study on the alpha-0.5 arms (read-only)
for d in pdhd:h pdvd:v; do det=${d%%:*}; x=${d#*:}
  python3 $S/d116_movers.py --det $det --a d115${x}off --b d115${x}p3bwp05 --out $F/116_movers_$det
  python3 $S/d116_mechanism.py --det $det --a d115${x}off --b d115${x}p3bwp05 --movers $F/116_movers_$det.tsv --out $F/116_mechanism_$det
  python3 $S/d116_michel_terminals.py --det $det --a d115${x}off --b d115${x}p3bwp05 --mechanism $F/116_mechanism_$det.tsv --out $F/116_michel_terminals_$det.txt
  python3 $S/d116_arm_sizing.py --det $det --arm d115${x}p3bwp05 --out $F/116_arm_sizing_$det.txt; done
python3 $S/d116_case_figs.py --det pdvd --a d115voff --b d115vp3bwp05 --keys 039349_10/66,039252_12/117,039252_14/37,039349_16/42,039349_45/49,039253_16/104,039252_4/104,039349_50/49 --out $F/116_case_pdvd
python3 $S/d116_case_figs.py --det pdhd --a d115hoff --b d115hp3bwp05 --keys 028084_21/132,028084_0/36,028084_20/65,029107_11/33,028084_2/116,029107_25/123 --out $F/116_case_pdhd

# sec 3 -- the frozen rule (written after sec 2, before any tagger arm; mtimes in sec 3)
(cd $F && sha256sum -c 116_pred.sha256)
for det in pdhd pdvd; do for L in 1 2 3; do bash $S/d102_compile_pr.sh $det d116_r${L}_p3 <the level's TLAs>; done; done   # figs/116_gate_config.txt (d116_cfgdiff.py vs d115r2_p3bwp05_<det>.json)
(cd $S && DET=pdhd SERIES=d116 PRED=116_pred.sha256 TRAJ=p3bwp05 LEVELS="r1 r2 r3" JOBS=5 bash d116_run_levels.sh)   # d116hr1..3
(cd $S && DET=pdvd SERIES=d116 PRED=116_pred.sha256 TRAJ=p3bwp05 LEVELS="r1 r2 r3" JOBS=5 bash d116_run_levels.sh)   # d116vr1..3

# sec 5 -- the blind scan (scratch round /home/xqian/tmp/d116; records only through mkv.py)
(cd $IMG/pdhd/stm_michel_scan && ./prep_stm_michel_scan.py --det pdhd --arm d115hoff --ctx-cells --outdir /home/xqian/tmp/d116/round/prep_d115hoff --sheetdir /home/xqian/tmp/d116/round/sheet_d115hoff --redraw)   # and every cell arm
python3 $S/d116_scan_items.py --det pdhd --cells d115hoff,d115hp3bwp05 --out /home/xqian/tmp/d116/items/pdhd_items.tsv
python3 $S/d103_scan_set.py --det pdhd --items /home/xqian/tmp/d116/items/pdhd_items.tsv --prep-root /home/xqian/tmp/d116/round --out /home/xqian/tmp/d116/round_pdhd/set
bash $F/116_shoot_round.sh pdhd round_pdhd 3        # headless blind shots; RUBRIC.md = pdhd/docs/scan/pdhd_stm_michel_scan_rubric.md (750751ea), AGENT_TASK.md = figs/116_agent_task_pdhd.md
python3 $IMG/pdhd/stm_michel_scan/campaign/nextwave.py /home/xqian/tmp/d116/round_pdhd /home/xqian/tmp/d116/round_pdhd/set/items_all.txt w1 --agents 2 --per 9 --seed 116
#   one blind Opus agent per wave file; python3 $S/d116_audit.py <transcript> pdhd w1_a0   (every scanner)
python3 $S/d103_scan_record.py --det pdhd --round /home/xqian/tmp/d116/round_pdhd --items /home/xqian/tmp/d116/items/pdhd_items.tsv --tag smx116 --record-out $IMG/pdhd/docs/scan/pdhd_stm_michel_smx116_verdicts.json > $F/116_scan_smx116_pdhd.txt
#   PDVD likewise (round_pdvd, RUBRIC.md = d99/swap_scan_rubric.md d760e223, items from the PDVD cells)

# sec 4 / 6 -- the grade with the fold, the split-half guard, the movers
python3 $S/d116_grade.py --self-test
python3 $S/d116_grade.py --det pdhd --cells A0=d115hoff,T=d115hp3bwp05,R1=d116hr1,R2=d116hr2,R3=d116hr3 --extra-record $IMG/pdhd/docs/scan/pdhd_stm_michel_smx116_verdicts.json --movers-out $F/116_movers_r_pdhd --split-half > $F/116_grade_pdhd.txt
python3 $S/d116_grade.py --det pdvd --cells A0=d115voff,T=d115vp3bwp05,R1=d116vr1,R2=d116vr2,R3=d116vr3 --extra-record $IMG/pdvd/docs/scan/pdvd_stm_michel_smx116_verdicts.json --movers-out $F/116_movers_r_pdvd --split-half > $F/116_grade_pdvd.txt
# sec 6 -- the rule FAILS on both detectors: no flip, no config edit; the flip unit and its proofs are listed in sec 6.3 for an owner override
```

## 1. The un-run PDVD arm: `tree+path` α 0.5 without `prefer3` (`d115vbwp05`)

The arm `d115vbwp05` (`-S steiner_base_weight_blank_alpha=0.5 -S steiner_base_weight_scope='tree+path'`, no
`prefer3`; PDVD 120 events, SRC `d103vflip`, the doc-115 pin, JOBS 8, 120 / 120 complete, pin md5 `2efa7fa09325`
before and after) was pre-registered as amendment 1 of the round-2 rule (`figs/115r2_pred_amend1.txt`, sha
`4157048f…` appended to `115r2_pred.sha256`; the compile proof `figs/116_gate_config.txt` shows the two pricing keys
of the two `CreateSteinerGraph` nodes as the only difference from the OFF config) and graded under every round-2
clause against `d115voff` (`figs/116_verdict_bwp05.txt`, next to P3BWP05 re-read from the same files):

| clause (PDVD) | base `d115voff` | BWP05 (α 0.5 `tree+path`, no `prefer3`) | P3BWP05 (`prefer3` + α 0.5) |
|---|---|---|---|
| U1 DETOUR length (≤ −30 %) | 8.2 m / 367 | 5.5 m / 253 (−32.9 %) PASS | −43 % PASS |
| U1b DETOUR + FIT (≤ −20 %) | 11.2 m | **11.7 m (+4.5 %) FAIL**; FIT 110 / 3.0 m → 122 / 6.2 m | −3.6 % FAIL (one 1.79 m spur) |
| U2 blank-carried stretches (≤ −40 %) | 221 | 142 (−35.7 %) FAIL; term 114 → 89, int 107 → 53 | −39.8 % FAIL |
| U3 D-crawl ≤ base | 99 | 62 PASS | PASS |
| P5f spot fits v1 / v2 / v3 (v3 ≤ 0.7) | 0.53 / 1.27 / 0.91 | 0.38 / 1.43 / **0.54 PASS** | 1.07 FAIL |
| D1 median `f_low` ≤ base, paired | 0.0160 | 0.0145 (−9.7 %), 193 / 182 PASS | PASS |
| D2 shape median ≤ base, paired better ≥ worse | 0.1643 | 0.1615 but paired 234 / 238 FAIL | PASS |
| D3 `k_pop` ±2 % / D4 Bee holes ≤ base | 1.0250 / 5.90 | 1.0249 PASS / 5.93 FAIL | PASS / PASS |
| S1, P1, R2D, W (×1.10), P2, P4, P6', Ga–Gd, C, R, G0 | | all PASS (S1 −15 %, W +9.4 %, R 0.857) | all PASS |
| tags (reported) `is_stm` purity / eff, Michel purity / eff vs A0 | 0.982 / 0.719, 0.940 / 0.731 | −0.019 / −0.024, **−0.070 / −0.073** | −0.018 / −0.011, −0.039 / −0.034 |

The arm settles the v3 question exactly as round 2 predicted (`prefer3` costs the v3 spot, the pricing alone fixes it:
0.54), and it does *not* settle the PDVD misses: without `prefer3` the pricing moves length from DETOUR into FIT
(3.0 → 6.2 m, round 1's BWP1 pattern; `figs/116_newstretch_pdvd_bwp05.txt`: 33 new FIT stretches of 2.69 m, the
largest the same painted-stripe cluster 039349_61 / 67 as in round 2, now a 1.81 m spur) so U1b fails outright, the blank-term
count stays at 89 so U2 misses by 4 points, the paired shape count is 234 / 238, and the tags move twice as far as
under `prefer3` (Michel −0.07). **By the owner's rule the PDVD trajectory for the retune and the flip is P3BWP05
(`d115vp3bwp05`), the same configuration as PDHD's**; the per-detector option is not taken. (The `pdhd:C` line of
BWP05 in the verdict file only records that no PDHD arm of that level exists.)

## 2. The nature of the tag regressions on the α 0.5 trajectories

The study compares each detector's production output (`d115?off`: production trajectory, production tagger) with
the α 0.5 arm (`d115?p3bwp05`: `prefer3` + `tree+path` α 0.5, production tagger) on the doc-103 population and truth
(PDHD own103h2 > own103h > smx27 > smx28; PDVD own103v2 > own103v > p99rwon carried corrected > smx11). Every number
below is in `figs/116_movers_<det>.txt`, `116_mechanism_<det>.txt`, `116_michel_terminals_<det>.txt`,
`116_arm_sizing_<det>.txt` and the case figures `116_case_<det>_*.png`.

### 2.1 The movers with their truth

| det | metric | TP lost | TP new | FP new | FP gone | candidates A0 / A1 / common | unlabelled candidates |
|---|---|---|---|---|---|---|---|
| PDHD | `is_stm` | 18 (9 not a candidate in A1) | 20 (6 new candidates) | 3 | 2 | 333 / 325 / 277 | 36 |
| PDHD | `michel_found` | 12 (6 not a candidate) | 10 (3 new candidates) | 4 | 2 | | |
| PDVD | `is_stm` | 43 (22 not a candidate) | 39 (19 new candidates) | 9 (4 new candidates) | 4 | 563 / 543 / 478 | 25 |
| PDVD | `michel_found` | 33 (10 not a candidate) | 25 (10 new candidates) | 10 (4 new candidates) | 3 | | |

The tag movement is nearly symmetric on both detectors: every metric's net change is a handful of items on a
population of 328 / 550 (PDHD Michel purity −0.024 is one false positive; PDVD Michel purity −0.039 is seven).

### 2.2 The tagger's inputs did not move; the decisions on the boundary did

On the candidates common to both arms (277 / 478) the quantities the tagger reads are unchanged at the median and
narrow at the tails (`116_mechanism_<det>.txt`, "scale check"):

| quantity (B relative to A) | PDHD p10 / p50 / p90 | PDVD p10 / p50 / p90 |
|---|---|---|
| `plateau_med` B/A | 0.856 / **1.000** / 1.215 | 0.933 / **0.998** / 1.050 |
| (`ks_mu` − `ks_flat`) B − A | −0.079 / **0.000** / +0.051 | −0.033 / **0.000** / +0.037 |
| `ratio_mu` B − A | −0.386 / −0.007 / +0.180 | −0.069 / +0.001 / +0.059 |
| `muon_len` B/A | 0.939 / **1.000** / 1.042 | 0.965 / **1.000** / 1.030 |
| \|Δ stop\| (cm) | 0.00 / 0.50 / 4.6 | 0.00 / 0.35 / 3.4 |
| `michel_ke_best` B/A (Michel in both) | 0.73 / 1.00 / 1.60 | 0.58 / 1.00 / 1.48 |

80 % of the common candidates keep both tags (221 / 277, 386 / 478); the 20 % that change sit at a median stop
displacement of 2.2 / 1.6 cm against 0.3 cm for the unchanged. So the regressions are **boundary flips of decisions
the tagger was already sitting on**, not a scale shift: a re-centring of the plateau / KS windows is not warranted and
none is attempted.

### 2.3 Which decision moved each tag

The first differing decision, in the tagger's order (candidacy = `T_stm_pass.status` per pass index, doc 107; then the
reject bits; then the Michel object), for the labelled movers (`116_mechanism_<det>.txt`, class table):

| channel | PDHD `is_stm` lost / new | PDHD Michel lost / new | PDVD `is_stm` lost / new | PDVD Michel lost / new |
|---|---|---|---|---|
| candidacy: STM evaluation (status 0 ↔ 3; prototype constants, no key) | 5 / 0 | 3 / 0 | 8 / 14 | 3 / 7 |
| candidacy: `detect_proton` (status 0 ↔ 5) | 1 / 1 | 0 / 0 | **10 / 2** | **5 / 0** |
| candidacy: mid-point track, left charge, pre-fit exit, no row | 3 / 5 | 3 / 3 | 4 / 3 | 2 / 2 |
| reject bits: `no_bragg` \| `shape_flat` (+ combos) | 7 / 8 | 2 / 1 (via `is_stm`) | 12 / 14 | 5 / 5 (via `is_stm`) |
| reject bits: `continuation` | 0 / 2 | 1 / 0 | **4 / 0** | 2 / 0 |
| reject bits: `plateau_off_mip`, `profile_sparse`, `stop_near_boundary`, `vertex_hadron` | 2 / 4 | 0 / 0 | 5 / 6 | 1 / 0 |
| Michel object: the stop arm re-classified Michel → other (kink below `michel_min_kink_deg` 30, or arm + far subtree above `michel_max_len_cm` 25) | | 1 / 3 | | **4 / 1** |
| Michel object: no arm found at the stop (`n_stop_arms` 1 → 0) | | 1 / 2 | | **8 / 4** |
| Michel object: the piece detached and vetoed by the range-energy guard (`n_michel_range_veto`) | | 0 / 1 | | **5 / 3** |

Two-way channels (the STM evaluation, the Bragg / shape bits, the exits) are noise on the boundary: a threshold move
would trade one side for the other. The **one-way channels** are on PDVD: the proton veto (10 hand stoppers lost,
7 of them with an attached Michel of 8–42 MeV, against 2 gained; `116_case_pdvd_039349_16_42.png`), the Michel arm
at the stop re-classified below the kink threshold (039349_10/66 39.6 → 22.6°, 039252_12/117 30.8 → 26.9°; PDHD
028084_21/132 44.3 → 24.9°; `116_case_pdvd_039349_10_66.png`, `116_case_pdhd_028084_21_132.png`) or over the length
cap (039349_50/49: 28.0 cm with its far subtree), the arm not found at all (`116_case_pdvd_039252_14_37.png`: the same
stop, the Michel segment no longer attached to the stop vertex), and the range-energy veto of a Michel piece that
the new trajectory detached from the stop (conn 1 → 2/3, 3–15 cm, < 10 MeV).

### 2.4 Two hypotheses tested and refuted

* **`prefer3` drops the Michel's Steiner terminals.** No: within 1.5 cm of the A-arm's Michel members the B arm
  holds 97 % of the terminals (PDVD lost Michels 182 / 188; control of 25 Michels kept in both arms 98 %), and no
  cluster lost every terminal (`116_michel_terminals_<det>.txt`). The Michel points are in the graph; the change is
  in the PR segmentation and the arm classification at the stop.
* **A re-centring is needed.** No (sec 2.2).

### 2.5 The levers, sized on their eligible population before pre-registration

`figs/116_arm_sizing_<det>.txt` reads every `stop-arm:` DEBUG line of the α 0.5 arms (kind, length, far length, MIP,
kink, shower, terminal) on the accepted stoppers *without* a Michel, joined to the truth:

| lever (production value) | PDVD admits | PDHD admits |
|---|---|---|
| `michel_min_kink_deg` 30 → 20 | 3 stoppers, all hand-positive (039349_57/25, 039252_12/117, 039349_10/66) | 2 (1 positive, 1 negative: 029107_19/111) |
| `michel_max_len_cm` 25 → 30 | 3, all positive (039349_54/33, 039253_14/45, 039349_50/49) | 1 positive |
| `michel_range_energy_ke_min` 10 → 5 MeV | 6 (4 positive, 2 negative) | 0 |
| `proton_muon_guard` OFF → ON | not sizable offline (detect_proton's ks1 / ratio3 are not persisted): 10 lost hand stoppers eligible | 1 |

The "no stop-arm" class (PDVD 8 lost Michels; PDHD 39 of the 53 stoppers without a Michel have no arm line) has no
key. The current Michel false positives (PDVD 10, PDHD 9) sit at kink p50 75–77°, length p50 3–7 cm, KE p50 8–11 MeV
against true Michels at 10–11 cm and 23–28 MeV — an energy or length floor would cut both classes (doc 104), and
none is built.

## 3. The frozen rule (`figs/116_pred.txt`, sha256 `ad24efda…`, written 17:36:15 before the first tagger arm 17:36:54)

Cells per detector: A0 = `d115?off`; T = the trajectory arm with the production tagger; the ladder R1 = T +
`proton_muon_guard`; R2 = R1 + `michel_min_kink_deg` 20 + `michel_max_len_cm` 30; R3 = R2 + `michel_range_energy_ke_min`
5 MeV; one ladder, the same keys on both detectors, compile-proven per level (`figs/116_gate_config.txt`: only the
named keys of the `TaggerCheckSTM` / `CheckSTM_Michel` nodes differ from T's compile). Metrics: `is_stm` and
`michel_found` purity / efficiency of R_k vs A0 (`d116_grade.py`), the blind record `smx116` folded at the lowest
precedence. **PASS** = every metric ≥ A0 − 0.020 on the labelled point value and its NEG bound ≥ −0.030, on both
detectors; plus C / R ≤ 1.20 / G0 and the split-half guard (no metric < A0 − 0.040 on either half of the events).
Selection = the smallest passing level. No level passing on both ⇒ nothing is flipped, the trajectory knobs
included. Predictions were written down (PDVD Michel purity predicted to fail at every level: the levers add true
positives and remove no false positive).

## 4. The tagger ladder

Six arms (`d116{h,v}r{1,2,3}`, PDHD 61 / PDVD 120 events, the doc-115 pin, JOBS 5, launched 17:36 / 17:46 after
the rule's sha; every runner summary rc 0, pin md5 `2efa7fa09325` before and after; the one "incomplete" event per
detector, 028084_13 and 039252_17, has all three outputs and no candidate, as in doc 115). Wall and memory against
the T arm (`figs/116_resources_<det>.txt`): PDHD wall ×0.89–0.90, PDVD ×0.94–0.96, peak RSS ×1.000 — R passes
everywhere.

### 4.1 What each rung of the ladder did (`figs/116_movers_ladder_<det>_*.txt`, labelled items)

| rung | PDVD | PDHD |
|---|---|---|
| T → R1 (`proton_muon_guard`) | +3 `is_stm` TP, all new candidates (the guard un-vetoed 3 of the 10 eligible hand stoppers), +1 Michel TP; no FP | no change at all (T ≡ R1 on every tag) |
| R1 → R2 (`michel_min_kink_deg` 20, `michel_max_len_cm` 30) | +2 `is_stm` TP, **+7 Michel TP, +1 Michel FP** | +2 Michel TP, +1 Michel FP |
| R2 → R3 (`michel_range_energy_ke_min` 5) | +1 `is_stm` TP / +1 FP, +4 Michel TP / **+6 Michel FP** | +1 Michel FP |

The sizing of sec 2.5 held to the item: R2 admitted 3 + 3 of the 6 hand-positive PDVD stoppers it named and 1 of
the eligible negatives, R3's range-energy floor admits false positives faster than true ones (its eligible
population was 4 : 2), and the proton guard fires only where the end also matches the muon hypothesis.

### 4.2 The grade before the fold (`figs/116_grade_<det>_prefold.txt`; point value, NEG / POS bounds)

| det | cell | `is_stm` purity | `is_stm` eff | Michel purity | Michel eff | TP / FP (Michel) |
|---|---|---|---|---|---|---|
| PDHD | A0 `d115hoff` | 0.959 | 0.646 | 0.880 | 0.716 | 73 / 10 |
| PDHD | T `d115hp3bwp05` | −0.007 (NEG −0.037) | +0.011 | −0.024 (NEG −0.024) | −0.020 | 71 / 12 |
| PDHD | R1 | = T | = T | = T | = T | 71 / 12 |
| PDHD | R2 | −0.007 | +0.011 | −0.031 | **+0.000** | 73 / 13 |
| PDHD | R3 | −0.007 | +0.011 | −0.040 | +0.000 | 73 / 14 |
| PDVD | A0 `d115voff` | 0.982 | 0.725 | 0.940 | 0.734 | 171 / 11 |
| PDVD | T `d115vp3bwp05` | −0.018 (NEG −0.032) | −0.011 | −0.039 | −0.034 | 163 / 18 |
| PDVD | R1 | −0.018 | −0.003 | −0.038 | −0.030 | 164 / 18 |
| PDVD | R2 | −0.018 (NEG −0.031) | **+0.003** | −0.040 | **+0.000** | 171 / 19 |
| PDVD | R3 | −0.021 | +0.005 | −0.065 | +0.017 | 175 / 25 |

R2 restores both efficiencies to production on both detectors (PDVD Michel TP 171 = A0's 171; PDHD 73 = 73) at a
cost of one Michel false positive each. What no rung touches is the Michel purity: the false positives the
trajectory brought (PDVD 18 vs 11, PDHD 12 vs 10) stay, exactly as the rule predicted. The `is_stm` purity point
values pass on both detectors (−0.007 / −0.018) but their NEG bounds sit below −0.030 before the fold — the
unlabelled candidates decide them, which is what the blind scan is for.

The split-half guard reads the same story on both detectors: the odd half passes every metric at every rung
(PDVD R2: −0.008 / −0.016 / −0.022 / −0.009), the even half carries the whole Michel purity cost (PDVD −0.055, PDHD
−0.072 at R2) — the cost is a small number of clusters, not a systematic shift.

## 5. The blind scan of the unlabelled candidates

Per detector (`figs/116_scan_smx116_<det>.txt`, records `<det>/docs/scan/<det>_stm_michel_smx116_verdicts.json`, new tag):

| | PDHD | PDVD |
|---|---|---|
| unlabelled candidates of A0 ∪ T (∪ the bwp05 arm on PDVD) + calibration | 36 + 4 | 33 + 3 |
| scannable (the prep's own filters: `has_pass`, ≥ 20 profile points, muon ≥ 10 cm) | 17 (23 "not in sheet") | 27 (9) |
| shots | 17, `check_shots` clean | 27; 6 frames smeared by a lost WebGL context (3 processes), re-shot with one process, then clean |
| scanners (Opus, one per wave, `nextwave.py` seed 116) | 2 (9 + 8) | 3 (9 + 9 + 9) |
| records / double scans | 17 / 0 | 27 / 0 |
| V2 calibration (stopper-or-not vs the existing record) | 4 / 4 agree, PASS | 3 / 3, PASS |
| verdicts on the new items | 3 STM_MICHEL, 1 FRAG_STM_MICHEL, 1 STM_ONLY, 4 THRU, 2 FRAG_THRU, 1 MESSY, 1 UNCLEAR | 4 STM_MICHEL, 5 STM_ONLY, 12 THRU, 1 FRAG_THRU, 1 MESSY, 1 UNCLEAR |
| confidence | 7 high, 6 medium | 10 high, 12 medium, 2 low |
| transcript audit (`d116_audit.py`, every scanner) | 86 + 79 tool calls, 0 flagged | 83 + 86 + 81 tool calls, 0 flagged |
| unlabelled candidates left after the fold | 23 (A0 20, T 21; below the display filters) | 9 (A0 9, T 5) |

What the fold did to the grade (`figs/116_movers_r_<det>_T.tsv`, source `smx116`): on PDHD four of T's unlabelled
tags are hand stoppers (028084_17/122, 028084_5/136, 029107_28/42 with a Michel; 029107_7/28 STM_ONLY) and one more
Michel (029107_18/68) — T's `is_stm` TP 119 → 123 and Michel TP 71 → 75, no new false positive; on PDVD one new
stopper with a Michel (039349_3/74), one Michel (039349_34/47) and one new `is_stm` false positive (039252_9/48, a
cathode-crossing THRU). The pins moved on 2 items (028084_17/122 at rr 8.5, 039349_34/47 at 11.28), both overshoots.

Rubric feedback the scanners raised (for the owner, with keys in their reports): the michel / gamma radius rule
leaves 5–10 cm undefined (three PDHD and two PDVD items landed there); the overshoot clause does not say what governs
when the post-peak charge lands *on* the plateau (029107_28/42) or collapses over only 2 cm (028084_17/122); rule 7
has no minimum track length (029107_7/28, 029107_25/23); at a cathode end the three-plane grey test is inapplicable by
construction (039349_9/42) and the `f_meas` window is truncated a few slices past some ends (039349_62/28, 33/18,
38/19); one PDHD prep item drew empty V / W `f_meas` panels (028084_23/125). None of these changes a verdict that the
grade turns on.

## 6. The decision

### 6.1 The reading of the rule (`figs/116_grade_{pdhd,pdvd}.txt`, after the fold)

| det | cell | `is_stm` purity | `is_stm` eff | Michel purity | Michel eff | NEG ≥ −0.030 | split-half (bar −0.040 per half) | rule |
|---|---|---|---|---|---|---|---|---|
| PDHD | T = R1 | −0.006 PASS | +0.032 PASS | −0.017 PASS | +0.019 PASS | all | odd PASS; **even FAIL** (Michel purity −0.056) | **FAIL** (split-half only) |
| PDHD | R2 | −0.006 | +0.032 | **−0.024 FAIL** | +0.038 | | even FAIL (−0.064) | FAIL |
| PDHD | R3 | −0.006 | +0.032 | −0.033 FAIL | +0.038 | | even FAIL (−0.079) | FAIL |
| PDVD | T | **−0.022 FAIL** | −0.008 | **−0.038 FAIL** | **−0.026 FAIL** | all | even FAIL (−0.062) | FAIL |
| PDVD | R1 | −0.021 FAIL | +0.000 | −0.037 FAIL | −0.021 FAIL | | even FAIL (−0.060) | FAIL |
| PDVD | R2 | −0.021 FAIL | +0.005 | −0.039 FAIL | +0.009 PASS | | even FAIL (−0.054) | FAIL |
| PDVD | R3 | −0.024 FAIL | +0.008 | −0.063 FAIL | +0.026 | | odd and even FAIL | FAIL |

C, R (≤ 1.20: PDHD 0.89–0.90, PDVD 0.94–0.96) and G0 pass on every arm. **No level passes on both detectors, so
nothing is flipped — the two Steiner knobs and the tagger keys stay at their production values** (rule sec 3: the
trajectory knobs are tied to the tagger's verdict). The fewest-failing levels are R1 and R2 (five failing clauses
each); R2 is the operating point that restores both efficiencies on both detectors, R1 the one that changes nothing
on PDHD.

### 6.2 What the reading means

* **PDHD.** With the blind labels in, the new trajectory with the *production* tagger is within the bar on all four
  metrics on the full sample. Doc 115's PDHD tag cost was, to that extent, a labelling gap: the trajectory tags
  stoppers the old scan never saw as candidates. The split-half guard fails because the Michel false positives (12
  vs 10) sit in one half of the events — the 4 new PDHD Michel false positives are owner-STM_ONLY stoppers with a
  small attached object (028084_2/116, 029107_15/40) and two smx27 STM_ONLY items (029107_11/33, 029107_20/17); 3
  false positives in 31 events is −0.056 on a 90-item half. A bar of −0.040 on a 30-event half is a small-number
  test, and the owner may judge it too tight; the rule is applied as written.
* **PDVD.** Purity fails on both metrics at every rung and the levers cannot reach it: they add true positives and
  remove no false positive, exactly as predicted. The `is_stm` false positives (5 → 11) and the Michel false positives
  (11 → 18) are labelled THRU or STM_ONLY on both sides (`figs/116_movers_r_pdvd_R2.tsv`: `is_stm` 039252_8/102,
  039252_9/48, 039253_16/104, 039253_4/96, 039349_21/27, 039349_24/24, 039349_37/55, 039349_7/57, 039349_70/58,
  039349_81/22; Michel adds 039252_12/84, 039252_4/104, 039253_12/98, 039253_17/107, 039349_23/55, 039349_48/59,
  039349_57/21) and come through the symmetric channels (sec 2.3: the STM evaluation, `no_bragg | shape_flat`
  cleared by the topology rule, and 0.5–2 cm arms at the stop). Five of the ten carry blind-agent (`new_agent`)
  THRU labels from doc 103's scan of the *old* trajectory; none has been looked at by the owner on the new one.
* **The levers.** `proton_muon_guard` recovered 3 PDVD stoppers and nothing on PDHD (no false positive either);
  `michel_min_kink_deg` 20 + `michel_max_len_cm` 30 recovered 7 + 2 Michels for 1 + 1 false positives; the
  range-energy floor is refuted (4 true for 6 false on PDVD). The remaining Michel losses (the arm not found at the
  stop at all, PDVD 8) have no key.

### 6.3 Recommendation

1. **Nothing is flipped; the C++ defaults and the production configs are untouched.** The trajectory (`prefer3` +
   `tree+path` α 0.5) plus the tagger operating point R2 is the configuration this round would carry, and doc 108's
   precedent is an explicit owner override of a purity miss — that is the owner's call, not this round's.
2. **The next measurement is the PDVD false-positive side, not another lever:** an owner look (the doc-103 `own103v`
   pattern, served on :5017) at the 10 `is_stm` and 7 additional Michel false positives above, on the *new*
   trajectory. Five of the ten carry only a blind-agent THRU label; if the owner's reading moves a third of them the
   PDVD purities clear the bar at R2 (Michel needs FP ≤ 14 at TP 173; `is_stm` needs FP ≤ 10).
3. **If the owner accepts R2 by override**, the flip unit is: toolkit `pdhd/pr.jsonnet` and `protodunevd/pr.jsonnet`
   `cm.steiner` + `steiner_refresh` call sites (`terminal_blank_plane_mode` `'prefer3'`, `base_weight_blank_alpha`
   0.5, `base_weight_scope` `'tree+path'` in the `if x == null then … else x` form of doc 108) and the driver bags
   (`stm_proton_muon_guard=true`, `michel_min_kink_deg: 20.0`, `michel_max_len_cm: 30.0` in `stm_michel_knobs`),
   proven exactly as sec 3 of the rule lists (compiled diff vs the R2 arms' TLA compile, the escape hatch, the
   SBND / uBooNE compiles, and a no-TLA runtime arm per detector against `d116?r2` on every `T_stm_michel` branch).
4. The 32 unlabelled candidates the display cannot draw (short or sparse tracks below the prep's filters) stay
   unlabelled; they bound the PDHD `is_stm` purity by −0.006 only and are not what the decision turns on.

## 7. The cleanup

The 362 output-less tag dirs of doc 115 round 1 (`pdhd/work/*_d115hbw{a,b}` 61 + 61, `pdvd/work/*_d115vbw{a,b}`
120 + 120; 0 regular files, 724 pctree symlinks, 4.3 MB) were listed with the predicate "name matches, `find -type f`
= 0, every entry a pctree symlink", re-verified at removal, and removed from the list file (`figs/116_cleanup.txt`).
The pctree targets (`d51hclus`, `p100flip`) were not touched; `ls -d pd?d/work/*_d115?bw[ab]` is empty.

## 8. Files

| path | what |
|---|---|
| `scripts/d116_run_levels.sh` | launcher (fork of `d115r2_run_levels.sh`): SERIES d115 (`bwp05`) / d116 (`r1 r2 r3` on TRAJ), sha + pin + placeholder guards |
| `scripts/d116_cfgdiff.py` | compiled-config proof: node / key diff of two compiled PR jobs with an expected key set |
| `scripts/d116_movers.py`, `d116_mechanism.py`, `d116_michel_terminals.py`, `d116_arm_sizing.py`, `d116_case_figs.py` | the regression study (sec 2) |
| `scripts/d116_grade.py` | the grade of the rule (fork of `d113_grade.py`): `--extra-record`, `--movers-out`, `--split-half`, `--neg-bar` |
| `scripts/d116_scan_items.py`, `d116_audit.py`, `figs/116_shoot_round.sh`, `figs/116_agent_task_{pdhd,pdvd}.md` | the blind scan (sec 5) |
| `scripts/d115r2_verdict.py` (`--levels`, `--spot`, additive) | Part 1's verdict under the round-2 clauses |
| `figs/115r2_pred_amend1.txt` (+ sha line in `115r2_pred.sha256`), `figs/116_pred.txt` + `.sha256` | the frozen rules |
| `figs/116_gate_config.txt`, `116_verdict_bwp05.txt`, `116_grade_pdvd_bwp05.txt`, `116_spot*`, `115r2_*_bwp05*`, `115r2_steiner_d115vbwp05.json` | Part 1 |
| `figs/116_movers_*`, `116_mechanism_*`, `116_michel_terminals_*`, `116_arm_sizing_*`, `116_case_*.png` | sec 2 |
| `figs/116_grade_{pdhd,pdvd}.txt`, `116_movers_r_*`, `116_scan_smx116_*`, `116_cleanup.txt` , `116_grade_<det>_prefold.txt`, `116_movers_ladder_*`, `116_resources_*`, `116_newstretch_pdvd_bwp05*`, `figs/116_verdict_bwp05.txt`; `scripts/d116_scan_record.py` (record builder on the doc-113 precedence), `d116_resources.py` (R clause) | secs 4–7 |
| `<det>/docs/scan/<det>_stm_michel_smx116_verdicts.json` | the blind records (new tag) |

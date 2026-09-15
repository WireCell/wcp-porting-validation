# 103 — Why a better track trajectory "costs" the STM/Michel taggers (PDHD, PDVD)

**Status (2026-09-15, round 1: measurement only).**
* **Question.** The owner wants both trajectory levers on: the doc 101 fit knobs (`fit_weight_pow` 1.5 +
  `assoc_cont_center` 1) and the doc 102 retile `charge_stepped`. What blocks them is the STM/Michel grade on the
  hand-scan records, which falls with both on:
  - STM efficiency 0.776 → 0.605 (PDHD), 0.877 → 0.732 (PDVD);
  - Michel purity 0.971 → 0.839 (PDHD), 0.923 → 0.837 (PDVD).
* **This round.** It finds the reason and measures the true cost, before any tagger change.
  - No toolkit code and no config change; every chain output is from the doc 101/102 arms.
  - New in this round: 337 verdict-blind agent labels (PDHD `smx28`, PDVD `smx10`) on the candidates the old
    records lack.
* **Answer, provisional (agent labels; owner review pending, sec 6).**
  1. **The efficiency loss is an artifact of the records.**
     - The records hold only clusters the old chain passed. The both-on chain swaps about a quarter of its
       candidates, and most of what it gains are real stoppers the records could not see.
     - On the union record the both-on STM efficiency is within the pre-registered 0.02 band of production:
       PDHD 0.605 → 0.590, PDVD 0.657 → 0.657.
  2. **Purity falls; how much of that is real differs by detector and chain.**
     - STM purity 0.976 → 0.929 (PDHD), 0.965 → 0.916 (PDVD); Michel purity 0.973 → 0.843, 0.924 → 0.827.
     - **PDVD:** the cost is partly owner-backed. Owner THRU labels carry 4 of the 17 new STM and 6 of the 23 new
       Michel false positives.
     - **PDHD Michel:** the cost rests entirely on unadjudicated agent labels (12 of 12). The scan reads 7 as
       stoppers with detached dots and 5 with no Michel, while A1 finds a short one (median 1.9 cm). That is the
       short / detached-piece class where the scanners reported rubric gaps (sec 6). So it is not yet a measured
       cost.
     - **Where the new STM false positives enter:** mostly through-going tracks that production's CheckSTM_Michel
       rejected on a shape or plateau bit, which clears with the new fit. PDVD `no_bragg` / `shape_flat`, PDHD
       `plateau_off_mip`.
  3. **The better fit separates at candidacy.** TaggerCheckSTM accepts more real stoppers and fewer non-stoppers
     (PDHD 0.840 → 0.862 vs 0.679 → 0.635; PDVD 0.758 → 0.830 vs 0.709 → 0.620).
  4. **Both taggers decide on zero- or small-margin KS sign tests.** Their refit noise equals the distance to the
     cut for half the clusters, so any refit re-deals verdicts in both directions (sec 4).
* **Pre-registered reading** (`figs/103_pred.txt` + amendment 1): **D2, a real cost, on both detectors**, carried
  by purity; efficiency is within 0.02. The losing class is named in sec 7, with a round-2 proposal aimed at it.
* **Not flipped. No production change.**
* **Round 2 (sec 10, same day).**
  - PDHD truth is corrected to `smx27`.
  - The owner adjudicated 27 PDHD items. STM is now within 0.02. Michel purity still fails (−0.051): stoppers the
    owner reads as having no visible Michel, where the chain attaches a piece. That is Michel admission, not the KS
    test.
  - PDVD's round-1 cells were not production. They are re-run on the production lineage, with labels in progress.
* **Round 3 (sec 11, same day).** PDVD on the production lineage, provisional (agent labels):
  - 239 verdict-blind labels (`smx11`), calibration disagreement 11 %.
  - Both purities fail: `is_stm` −0.022 (one false positive over the limit), Michel −0.044. Both efficiencies rise.
  - The readout-window-edge rubric gap (37 items) does not carry the purity cost.
  - The owner's adjudication `own103v` (40 items) is in progress.

## 0. Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs
export STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json

# sec 3-4 -- record-free census, candidate swap, TaggerCheckSTM status transitions, clause attribution, scale
python3 $S/d103_churn_census.py --det pdhd --base d101hnew --arms d101hkf,d102hocs,d102hcs  > $F/103_churn_pdhd.txt
python3 $S/d103_churn_census.py --det pdvd --base d101vnew --arms d101vkf,d102vocs,d102vcsall > $F/103_churn_pdvd.txt
for a in d101hkf d102hocs d102hcs; do python3 $S/d103_eval_attrib.py --det pdhd --base d101hnew --arm $a > $F/103_eval_attrib_pdhd_$a.txt; done
for a in d101vkf d102vocs d102vcsall; do python3 $S/d103_eval_attrib.py --det pdvd --base d101vnew --arm $a > $F/103_eval_attrib_pdvd_$a.txt; done
python3 $S/d103_scale.py --det pdhd --base d101hnew --arms d101hkf,d102hocs,d102hcs  > $F/103_scale_pdhd.txt
python3 $S/d103_scale.py --det pdvd --base d101vnew --arms d101vkf,d102vocs,d102vcsall > $F/103_scale_pdvd.txt
# sec 2 -- separation on the old records (all six cells, appended to one file)
python3 $S/d103_separation.py --det pdhd --base d101hnew --arm d102hcs      # ... > $F/103_separation.txt
# sec 3.3 -- bound; sec 6 -- tag moves on the old records
python3 $S/d103_bounds.py  --det pdhd --base d101hnew --arm d102hcs    > $F/103_bounds_pdhd.txt
python3 $S/d103_bounds.py  --det pdvd --base d101vnew --arm d102vcsall > $F/103_bounds_pdvd.txt
python3 $S/d103_fp_list.py --det pdhd --base d101hnew --arm d102hcs    > $F/103_moves_pdhd.tsv
python3 $S/d103_fp_list.py --det pdvd --base d101vnew --arm d102vcsall > $F/103_moves_pdvd.tsv

# sec 5.0 -- frozen rule (before any new label existed), V1
sha256sum -c <(head -1 $F/103_pred.sha256 | sed "s#pred.txt#$F/103_pred.txt#")
python3 $S/d103_union_grade.py --det pdhd --fixed-only > $F/103_v1_fixed_pdhd.txt   # must reproduce doc 102 13.2.1
python3 $S/d103_union_grade.py --det pdvd --fixed-only > $F/103_v1_fixed_pdvd.txt

# sec 5.1 -- the blind scan (scratch round dirs; prep with scratch --outdir/--sheetdir, never the committed sheet)
R=/home/xqian/tmp/d103; P=$IMG/pdhd/stm_michel_scan; C=$P/campaign
(cd $P && ./prep_stm_michel_scan.py --det pdhd --arm d102hcs --ctx-cells --outdir $R/round/prep_d102hcs \
    --sheetdir $R/round/sheet_d102hcs --redraw)                                # and d102hocs d101hkf d101hnew, pdvd likewise
python3 $S/d103_scan_set.py --det pdhd --items $R/items/pdhd_items.tsv --prep-root $R/round --out $R/round_pdhd/set
bash $F/103_shoot_all.sh pdhd 3        # every display arm through campaign/shoot.sh, then mkzoom + check_shots (needs $R/round_pdhd/set)
#   scanner instructions: $F/103_agent_task_{pdhd,pdvd}.md, copied to $R/round_<det>/AGENT_TASK.md with the frozen rubric
#   ($R/round_pdhd/RUBRIC.md = pdhd/docs/scan/pdhd_stm_michel_scan_rubric.md, sha 750751ea...;
#    $R/round_pdvd/RUBRIC.md = pdvd/docs/nf_sp_img_clus/d99/swap_scan_rubric.md, sha d760e223...)
python3 $C/nextwave.py $R/round_pdhd $R/round_pdhd/set/items_all.txt w1 --agents 6 --per 21 --seed 103
#   one blind agent per wave file, AGENT_TASK.md in the round dir; records only through mkv.py
python3 $S/d103_audit.py --selftest
python3 $S/d103_audit.py <transcript> pdhd w1_a0                              # every scanner
python3 $S/d103_scan_record.py --det pdhd --round $R/round_pdhd --items $R/items/pdhd_items.tsv --tag smx28 \
    --record-out $IMG/pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json > $F/103_scan_smx28_pdhd.txt    # V2 inside
python3 $S/d103_scan_record.py --det pdvd --round $R/round_pdvd --items $R/items/pdvd_items.tsv --tag smx10 \
    --record-out $IMG/pdvd/docs/scan/pdvd_stm_michel_smx10_verdicts.json > $F/103_scan_smx10_pdvd.txt

# sec 5.2-7 -- union grade, separation on the union record, FP classes and paths, owner queue
N28=$IMG/pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json; N10=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx10_verdicts.json
python3 $S/d103_union_grade.py --det pdhd --new-record $N28 > $F/103_union_grade_pdhd.txt
python3 $S/d103_union_grade.py --det pdvd --new-record $N10 > $F/103_union_grade_pdvd.txt
python3 $S/d103_separation.py --det pdhd --base d101hnew --arm d102hcs --extra-record $N28   # > 103_separation_union_pdhd.txt (3 cells)
python3 $S/d103_separation.py --det pdvd --base d101vnew --arm d102vcsall --extra-record $N10 # > 103_separation_union_pdvd.txt
python3 $S/d103_fp_classes.py --det pdhd --new-record $N28 > $F/103_fp_classes_pdhd.txt     # and pdvd
python3 $S/d103_fp_path.py    --det pdhd --new-record $N28 > $F/103_fp_path_pdhd.txt        # and pdvd
python3 $S/d103_owner_queue.py --det pdhd --new-record $N28 --items $R/items/pdhd_items.tsv > $F/103_owner_queue_pdhd.md   # and pdvd
```

Cells, on the same pctrees:

| Cell | PDHD (61 events) | PDVD (120 events; 116 readable in all) |
|---|---|---|
| A0 production | `d101hnew` | `d101vnew` |
| K fit knobs only | `d101hkf` | `d101vkf` |
| S sampler only | `d102hocs` | `d102vocs` |
| A1 both on | `d102hcs` | `d102vcsall` |

## 1. The question

The fixed-denominator grades of doc 102 sec 13.2.1 (`d101_stm_grade_{pdhd,pdvd}.py`):

| Detector | STM purity / efficiency A0 → A1 | Michel purity / efficiency A0 → A1 |
|---|---|---|
| PDHD | 0.974 / 0.776 → 0.937 / 0.605 | 0.971 / 0.791 → 0.839 / 0.605 |
| PDVD | 0.968 / 0.877 → 0.922 / 0.732 | 0.923 / 0.872 → 0.837 / 0.689 |

The owner's objection: a trajectory and dQ/dx that follow the image better should make the Bragg-peak-versus-flat
KS test work better, not worse.

## 2. Does the better fit separate stoppers from non-stoppers better?

AUC is the probability that a hand stopper scores more muon-like than a hand non-stopper, on the same labelled
clusters in both cells:
* **TaggerCheckSTM:** the best discriminant D over the eval calls of the lowest pass (sec 4).
* **CheckSTM_Michel:** `ks_flat − ks_mu` and `contrast / contrast_expected`.

**On the old records** (`103_separation.txt`):

| Test | PDHD A0 → A1 | PDVD A0 → A1 |
|---|---|---|
| TaggerCheckSTM D | 0.837 → 0.843 | 0.859 → 0.838 |
| CheckSTM_Michel shape | 0.686 → **0.743** | 0.882 → 0.838 |
| Bragg contrast | 0.730 → 0.748 | 0.893 → 0.842 |

**On the union record** (old records + this round's blind labels; `103_separation_union_{pdhd,pdvd}.txt`):

| Test | PDHD A0 → A1 (both records blind) | PDVD A0 → A1 (old record verdict-visible) |
|---|---|---|
| TaggerCheckSTM D | 0.814 → **0.846** | 0.807 → **0.820** |
| TaggerCheckSTM accepts stoppers | 0.840 → **0.862** | 0.758 → **0.830** |
| TaggerCheckSTM accepts non-stoppers | 0.679 → **0.635** | 0.709 → **0.620** |
| CheckSTM_Michel shape (candidates in both cells) | 0.686 → 0.743 *(same items as the old-record row)* | 0.869 → 0.834 |
| Bragg contrast (candidates in both cells) | 0.730 → 0.748 *(same items)* | 0.884 → 0.838 |

The CheckSTM_Michel rows need a candidate with a shape test in both cells. A gained candidate has none in A0, so the
new blind labels add no item to them on PDHD, where they equal the old-record numbers. They are not independent
corroboration. On PDVD they gain the few blind-labelled items that were candidates in both cells.

* **PDHD** has two verdict-blind records (smx18/smx22 and this round's smx28). On them the both-on fit separates
  better at both stages.
* **PDVD's** old record was scanned with the chain's verdict on screen (doc pdvd/55 sec 16.3), which leans its
  labels toward production. Its CheckSTM_Michel AUCs favour production by an unknown amount, and they are the only
  numbers that go the other way. Candidacy, which the new blind labels dominate on the gained side, separates better
  on PDVD too.
* **Answer to the owner's objection.** Yes: where it can be measured cleanly, the better fit makes the taggers
  separate better. The drop in the grades is not a loss of separation (sec 3); the part that is real is a purity
  class (sec 7).

## 3. Why the old grades fall

### 3.1 The records were drawn from clusters the old chain had already passed

Among the clusters labelled in the old records, production's TaggerCheckSTM accepted (lowest-pass status 0):
* PDHD: **100 %** of the stoppers and 95 % of the non-stoppers;
* PDVD: 96 % and 90 %.

On this population production's pass rate is close to one by construction, so any change can only remove clusters.

The both-on chain keeps:

| Detector | Stoppers | Non-stoppers |
|---|---|---|
| PDHD | 135 of 152 (−11 %) | 69 of 106 (**−35 %**) |
| PDVD | 239 of 273 (−12 %) | 172 of 256 (**−33 %**) |

It drops non-stoppers three times as often as stoppers, and the old grade barely credits that. Most of those
non-stoppers were already rejected downstream by CheckSTM_Michel, while each dropped stopper is a lost true positive.

### 3.2 A quarter of the candidates swap, and the gained ones were unlabelled

`103_churn_{pdhd,pdvd}.txt`, sections A and B.

| Detector | Cell | Candidates | `is_stm` | `michel_found` | Lost / gained vs A0 |
|---|---|---|---|---|---|
| PDHD | A0 | 341 | 129 | 133 | — |
| PDHD | K | 320 | 120 | 118 | 70 / 49 |
| PDHD | S | 328 | 124 | 131 | 89 / 76 |
| PDHD | A1 | 333 | 129 | 131 | 80 / 72 |
| PDVD | A0 | 593 | 264 | 169 | — |
| PDVD | K | 587 | 266 | 176 | 102 / 96 |
| PDVD | S | 598 | 283 | 189 | 124 / 129 |
| PDVD | A1 | 581 | 277 | 184 | 133 / 121 |

* **Different objects, same clusters.** Lost and gained candidates are different objects: 78/80 (PDHD) and 132/133
  (PDVD) have no other-cell candidate within 20 cm. Their `T_cluster` rows are identical in both cells (79/80,
  129/133), so imaging and clustering are the same. What changed is TaggerCheckSTM's verdict (sec 4).
* **What the old records held of the gains.** PDHD: 0 of the 72 gained. PDVD: 15 of 121.
* **Symmetric totals are not evidence of no cost.** Doc pdhd/stm-tagger-chain §13 had a knob whose added tags were
  73 % through-going. Only labels on the gained side decide, which is why sec 5 exists.

### 3.3 How far labels on the unlabelled candidates could move the grade (before the scan)

`103_bounds_{pdhd,pdvd}.txt`.
* The graders' counts are re-derived first: PDHD A0 TP 114 FP 3 FN 33 TN 107, A1 89 / 6 / 58 / 104; PDVD A0
  242 / 8 / 34 / 262, A1 202 / 17 / 74 / 253. All identical to doc 102.
* The unlabelled candidates of either cell (PDHD 96, PDVD 129) are also clusters the other cell missed, so they enter
  both cells' denominators.
* Carried-over records and the doc-99 swap scan are not counted as labels (`figs/103_pred.txt`).

| Detector | Case | A0 purity / efficiency | A1 purity / efficiency | A1 − A0 efficiency |
|---|---|---|---|---|
| PDHD | best for A1 | 0.974 / 0.655 | 0.951 / 0.667 | +0.011 |
| PDHD | worst for A1 | 0.974 / 0.528 | 0.730 / 0.412 | −0.116 |
| PDVD | best for A1 | 0.965 / 0.766 | 0.937 / 0.772 | +0.006 |
| PDVD | worst for A1 | 0.942 / 0.683 | 0.754 / 0.567 | −0.115 |

The measured union-record changes (sec 5.2: −0.015 PDHD, 0.000 PDVD) lie inside these bounds, near the best case.

### 3.4 The cuts were tuned on the old fit, against the same records

CheckSTM_Michel's operating point was tuned knob by knob against these same records:
* `ks_margin` −0.10 / −0.02;
* the 3 cm and 8 cm Bragg anchors;
* the plateau window;
* the topology clears.

The tuning ran through docs pdhd/21–25 and pdvd/48–96, all on the q²-weighted, stepped-retile trajectory. An
operating point fitted to one realisation of the per-point noise, re-measured on another, reads worse on the records
it was fitted to.

## 4. The mechanism of the swap

### 4.1 Candidacy is TaggerCheckSTM's verdict, and the flips are status 0 ↔ 3

* **Candidacy.** CheckSTM_Michel evaluates only mains that TaggerCheckSTM flagged STM (CheckSTM_Michel.cxx:2618-2631,
  `require_stm_flag` true), one row each. A missing row means TaggerCheckSTM did not pass the cluster.
  `T_cluster.stm` is never filled.
* **Flips.** From the job log (`visit: TaggerCheckSTM`, `persist_stm_fit`), 4041 PDHD / 4449 PDVD clusters are
  evaluated identically in every cell. The dominant flip of the lowest pass is status **0 (accepted) ↔ 3
  (`eval_stm_core` failed)** (codes TaggerCheckSTM.cxx:922-935), in both directions:

  | Detector | Cell | 0→3 | 3→0 |
  |---|---|---|---|
  | PDHD | K | 38 | 28 |
  | PDHD | S | 54 | 47 |
  | PDHD | A1 | 45 | 47 |
  | PDVD | A1 | 91 | 79 |

* **The rest:** 0 ↔ 5 (proton), 0 ↔ 2 (long leftover), 0 ↔ 4 (other tracks).

### 4.2 The deciding clause is a zero-margin sign test

`tracking-stm.root:T_stm_eval` records every `eval_stm_core` call: `ks1` (data vs the muon template), `ks2` (data vs
a flat `mip_dqdx`), `ratio1`, `ratio2`, `res_length`, `ave_res_dqdx` and the verdict
(TaggerCheckSTM.cxx:2966-2974, :3047).

`d103_eval_attrib.py` re-evaluates the clauses of TaggerCheckSTM.cxx:2980-3036 from those values. It reproduces
**every** recorded accept (0 predicted rejects recorded as accepts, over 8301 PDHD / 14664 PDVD calls). 187 / 386
predicted accepts are recorded rejects: the residual-straightness clause, whose inputs are not recorded.

The flips are not a change of ladder rung: `left_L` changes by a median 0 mm, and no call accepts in the rejecting
cell. On the call that accepted in the other cell, the rejecting clause is:

| Clause | PDHD A1 0→3 | PDHD A1 3→0 | PDVD A1 0→3 | PDVD A1 3→0 |
|---|---|---|---|---|
| `ks1 − ks2 ≥ 0` (:2987) | **28** | **24** | **57** | **43** |
| no accept clause (:3025-3035) | 10 | 6 | 12 | 18 |
| near-flat (:2988) | 4 | 7 | 2 | 8 |
| predicted accept (straightness) | 1 | 1 | 15 | 3 |
| rung not reached | 3 | 11 | 8 | 12 |

The discriminant D = ks1 − ks2 + (|ratio1−1| − |ratio2−1|)/1.5·0.3 on that call:
* PDHD: −0.003 in the accepting cell vs +0.067 in the rejecting one;
* PDVD: −0.006 vs +0.058.

### 4.3 Half of the STM-like clusters sit inside the refit noise

| Measure | PDHD | PDVD |
|---|---|---|
| Refit noise on best (ks1 − ks2), same-status clusters: p68 / p90 of the change | 0.028 / 0.060 | 0.020 / 0.043 |
| Share within 0.06 of the edge, status 0 (best D) | 47 % | 57 % |
| Share within 0.06 of the edge, status 3 (best D) | 65 % | 73 % |

* **Why the test is fragile.** `kslike_compare` (util/src/KSTest.cxx:216) is the largest gap between two normalised
  running sums over **fit points**. Each point weighs the same whatever its dx, and the window starts at a peak
  chosen by a 5-point running mean (TaggerCheckSTM.cxx:2849-2906). A refit moves points, changes which survive the
  charge-zero cleanup, and moves the peak by a point or two.
* **The sign term carries little.** On the labelled clusters, `ks1 − ks2` alone separates at AUC 0.46–0.58; the
  discrimination lives in the ratio term.
* **CheckSTM_Michel repeats the recipe in series** (`shape_flat`, CheckSTM_Michel.cxx:3206-3220, with `ks_margin`),
  together with `no_bragg`. Its kept-candidate flips are symmetric on the same bits (`103_churn_*` section D):
  - PDHD A1: 19 1→0, mostly `no_bragg` 13; 12 0→1, clearing `plateau_off_mip` 7 and `no_bragg` 4.
  - PDVD A1: 24 each way, on `shape_flat` / `no_bragg`.

### 4.4 A small systematic dQ/dx rise rides on top

`103_scale_{pdhd,pdvd}.txt`, A1 vs A0:

| Measure | PDHD | PDVD |
|---|---|---|
| Paired `plateau_med` | +3.6 % | +2.0 % |
| All-candidate plateau median | 50.9k → 55.6k | 53.8k → 55.3k |
| Summed profile charge | ×1.025 | ×1.014 |
| `muon_len` | ×0.978 | ×0.979 |

* **Charge, not dx.** The plateau ratio does not follow the length ratio (correlation +0.02 / +0.07). The better
  trajectory collects more charge.
* **The ends do not move.** Stop and entry shifts along the track have a median of ≤ 0.01 cm.
* **Scale vs references.** The MIP references (`mip_dqdx` 56000 / 55000, `mip_dqdx_median` 48000 / 47000) were set on
  the old scale. The shift is small next to the KS noise, but it has one visible consequence on PDHD (sec 7).

## 5. The true grade

### 5.0 The frozen rule

`figs/103_pred.txt` (sha256 `8e832396…`, 2026-09-15T09:56:44), written before any new label existed. It fixes:
* **labels:** old record first, then the new blind record;
* **population:** judged items that are candidates in any cell, plus the graders' fixed populations;
* **metrics;** V1 reproduction of the old grades; V2 calibration (≤ 25 % disagreement); V3 consistency with the bound;
* **decision:** D1 if every A1 metric ≥ A0 − 0.02, else D2.

**Amendment 1** (`figs/103_pred_amend1.txt`, 2026-09-15T10:05:44, still before any label) removed the PDVD owner
records `own100*`. The ordering is checkable: the earliest `written` stamp in the new records is 2026-09-15T10:20:03
(smx28) and 10:26:06 (smx10). Their keys are cluster ids on other pctree lineages (p96vprod / p98vonq / p99 / p100). Their 8
candidates were added to the blind scan instead.

**V1 passes** (`103_v1_fixed_{pdhd,pdvd}.txt`). With the old records only, the union grader reproduces doc 102
13.2.1 exactly, in all four cells and both chains.

### 5.1 The blind scan

* **Items** (`/home/xqian/tmp/d103/items/<det>_items.tsv`, private):
  - every candidate of any cell with no label (PDHD 136, PDVD 196 + 8), plus 20 calibration items each drawn with
    `random.Random(103)`, shuffled;
  - displayed from the both-on cell where it is a candidate, else S, K, A0;
  - 34 PDHD / 10 PDVD candidates have no scan payload in their display cell: the prep keeps `n_profile_pts` ≥ 20 and
    `muon_len` ≥ 10 cm (`103_no_payload_{pdhd,pdvd}.tsv`). One of them, PDHD 028084_2/66, had a payload in `d101hkf`
    and was scanned from there. The other 33 + 10 stay unlabelled and outside the population. None is `is_stm` 1 in
    A0 or A1.
* **Frames.**
  - The real viewer driven headless: `shoot.sh`, `scan_harness.py --blind --hide-selection`.
  - PDHD 123 and PDVD 214 items; `check_shots.py` clean.
  - The saved context carries no `is_stm`, `reject_names`, `in_fv` or `flow`. The residual leak of doc pdhd/18 stays:
    the object table's reconstruction typing.
* **Scanners.**
  - 6 PDHD and 10 PDVD agents, 18–22 items each. Rubrics: PDHD v5 (`750751ea…`), PDVD doc-99 blind port
    (`d760e223…`).
  - Each read only the rubric, the task file, its item list and its own items' frames, and wrote only through
    `mkv.py`.
  - `d103_audit.py` (self-test 13/13 bad lines flagged, 0/8 good) passes on all 16 transcripts.
* **Records** (new files; the old records' sha256 are unchanged):
  - `pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json` (123);
  - `pdvd/docs/scan/pdvd_stm_michel_smx10_verdicts.json` (214).

| | PDHD smx28 (103 new) | PDVD smx10 (194 new) |
|---|---|---|
| STM_MICHEL / STM_ONLY | 27 / 16 | 49 / 49 |
| THRU (+FRAG) | 52 | 84 |
| MESSY / UNCLEAR | 1 / 7 | 9 / 3 |
| confidence high / medium / low | 57 / 39 / 7 | 78 / 108 / 8 |
| **V2 calibration**: stopper-or-not disagreements | **2 / 18 (11 %) PASS** | **2 / 20 (10 %) PASS** |

Notes on the table:
* **V2 vs queue section D.** V2 counts only items judged on both sides. Queue section D also lists a stopper that the
  blind scan called UNCLEAR, hence 3 there for PDHD.
* **Consistency, not accuracy.** On PDHD 15 of the 20 calibration items carry an agent label in smx22 (5 owner). On
  PDVD 5 of 20 carry an owner-confidence label in the verdict-visible record. So V2 mostly measures agent-to-agent
  label consistency.
* **Unlabelled counts.** Sec 5.2's "unlabelled 33" (PDHD) is the 34 no-payload candidates minus 028084_2/66, which
  had a payload in `d101hkf` and was scanned from there.

### 5.2 Union-record grade (provisional: agent labels)

`103_union_grade_{pdhd,pdvd}.txt`; bootstrap 68 % intervals in brackets.

**PDHD**
* Population 365: fixed 257, plus 108 judged candidates. Labels: smx22 233, new blind 95, owner 37.
* Unlabelled 33 (no payload); unjudged 55.

| Cell | `is_stm` purity | `is_stm` efficiency | Michel purity (pop. 197) | Michel efficiency |
|---|---|---|---|---|
| A0 production | 0.976 [0.962, 0.992] | 0.605 [0.572, 0.642] | 0.973 | 0.615 |
| K knobs | 0.938 | 0.530 | 0.920 | 0.590 |
| S sampler | 0.934 | 0.570 | 0.895 | 0.658 |
| **A1 both** | **0.929** [0.908, 0.952] | **0.590** [0.557, 0.625] | **0.843** | **0.641** |
| A1 − A0 | **−0.047** | −0.015 | **−0.130** | +0.026 |

**PDVD**
* Population 751: fixed 546, plus 205. Labels: old record 569, new blind 182.
* Unlabelled 10; unjudged 37.

| Cell | `is_stm` purity | `is_stm` efficiency | Michel purity | Michel efficiency |
|---|---|---|---|---|
| A0 production | 0.965 [0.954, 0.977] | 0.657 [0.635, 0.683] | 0.924 | 0.676 |
| K knobs | 0.927 | 0.628 | 0.818 | 0.625 |
| S sampler | 0.911 | 0.673 | 0.799 | 0.662 |
| **A1 both** | **0.916** [0.899, 0.933] | **0.657** [0.633, 0.683] | **0.827** | **0.662** |
| A1 − A0 | **−0.049** | 0.000 | **−0.097** | −0.014 |

**The gained side, labelled** (A1-only candidates):

| Detector | Stoppers | Non-stoppers | Unjudged / unlabelled | Of those A1 tags: stoppers / non-stoppers |
|---|---|---|---|---|
| PDHD | 32 | 33 | 1 / 6 | **24 / 3** |
| PDVD | 67 | 50 | 5 / 2 | **40 / 5** |

The lost side, for comparison:

| Detector | Stoppers | Non-stoppers | A0 tags: stoppers / non-stoppers |
|---|---|---|---|
| PDHD | 22 | 39 | 16 / 0 |
| PDVD | 37 | 87 | 28 / 2 |

The both-on chain gains more real stoppers than it loses (PDHD +24 vs −16 tagged; PDVD +40 vs −28). What it gains
also carries a few more non-stoppers.

**V3** passes. The union A1 − A0 efficiency (−0.015 / 0.000) lies inside the pre-scan bound (sec 3.3).

**Reading of the frozen rule: D2 on both detectors.** Efficiency is within 0.02, but STM purity falls by 0.047 /
0.049 and Michel purity by 0.130 / 0.097.

## 6. What the owner needs to review

Owner queue, `figs/103_owner_queue_{pdhd,pdvd}.md`:

| Section | Content | PDHD | PDVD |
|---|---|---|---|
| A | False positives of A1 on the union record (`is_stm` / Michel), with label source, confidence and which cells tag them | 9 / 14 | 23 / 30 |
| B | New blind labels on items some cell tags | 69 | 92 |
| C | Other medium / low new labels | 19 | 68 |
| D | Calibration disagreements | 3 | 2 |
| E | New labels whose notes name the readout-window edge | 4 | 88 |

**Where the purity cost rests** (`103_fp_classes_{pdhd,pdvd}.txt`, A1-only false positives):

| Detector | Chain | n | Label source | Hand class |
|---|---|---|---|---|
| PDHD | `is_stm` | 8 | owner 1, existing agent 4, new blind agent 3 | all THRU |
| PDHD | Michel | 12 | existing agent 9, new blind agent 3, owner **0** | STM_ONLY with detached dots 7, none 5 |
| PDVD | `is_stm` | 17 | owner 4, existing agent 8, new blind agent 5 | all THRU |
| PDVD | Michel | 23 | owner 6, existing agent 9, new blind agent 8 | THRU 17, STM_ONLY 6 |

* **PDHD Michel.** Its entire Michel purity cost sits on agent labels of real stoppers whose Michel the scan reads as
  absent or detached. That is the 5–10 cm / short-piece boundary the scanners flagged as ambiguous in the rubric
  (below). `feedback_adjudicate_fps_before_holding_a_knob` applies directly.
* **Owner-labelled false positives.** 1 of 8 PDHD STM, 4 of 17 PDVD STM and 6 of 23 PDVD Michel carry the owner's own
  THRU. Those are firm.

**Rubric gaps the scanners reported** (`103_scan_reports_{pdhd,pdvd}.md`), most frequent first:
1. **The readout-window edge** has no rule or prefix.
   - Run 039349 frames end at slice ~1600 (6400 ticks, doc pdvd/99). Many top-volume tracks end there.
   - Most PDVD scanners called a flat end at the edge THRU. One applied rule 3 throughout and named the edge only as
     the competing reading. At least two others decided margin cases (an end 7–38 slices from the edge, or a short
     rise at it) as stops.
   - The owner should rule once. Queue section E lists the 88 PDVD labels whose notes mention the edge.
2. **Unfitted C rows** have no dQ/dx. The degenerate-row clause then forbids the capture-gamma tag the rubric's own
   example uses; this moves `michel_kind` on several items.
3. **A detached piece 5–10 cm from the stop** has no michel-or-gamma rule; it decides both vs attached vs detached
   dots.
4. **An overshoot collapse inside one fitted segment** cannot be recorded as STM_MICHEL: `mkv.py` needs a michel row.
   Such items were recorded STM_ONLY with the pin moved (PDHD 029107_13/111, PDVD 039253_17/23).
5. **A straight-on stub under a clear rise.** Rule 1 v2 ("weak") conflicts with the hot-tip section ("a real Michel").

## 7. The decision and the round-2 proposal

**The losing class is purity**, and `103_fp_path_{pdhd,pdvd}.txt` shows where it enters.

| Detector | Chain | A1-only FPs | How they get through |
|---|---|---|---|
| PDHD | `is_stm` | 8 | 5 were A0 candidates that CheckSTM_Michel rejected on `plateau_off_mip` 3, `stop_near_boundary` 2, `no_bragg` 1, `shape_flat` 1; the bit clears with the new fit. 3 were not A0 candidates (TaggerCheckSTM status 3 → 0). |
| PDVD | `is_stm` | 17 | 12 were A0 candidates rejected on **`no_bragg` 11 / `shape_flat` 6**, which clear. 5 were status 3 → 0. |
| PDHD | Michel | 12 | 9 were A0 candidates with `michel_found` 0. A1's Michel is short (median `michel_len` 1.9 cm; conn attached 7, bridged 4). |
| PDVD | Michel | 23 | 16 were A0 candidates with `michel_found` 0, 7 new candidates. Median `michel_len` 3.5 cm, attached 18. |

So the purity cost is mostly **CheckSTM_Michel's shape / Bragg / plateau tests re-dealing through-going tracks**.
These are the same knife-edge tests as sec 4, now on the reject side. On PDHD the `plateau_off_mip` clears match
the +3.6 % plateau rise (sec 4.4): low-plateau through-goers move into the [0.6, 1.6] MIP window.

**Proposed round 2, in order:**
1. **The owner reviews queue A:** 23 PDHD and 53 PDVD entries, the PDHD Michel ones first, since no owner label
   carries them. Also rule on gaps 1–3 above. Then re-grade (`d103_union_grade.py` reads owner fields first). With
   1–3 false positives per class a single label is the measurement.
2. **If the purity cost survives,** build the lever the owner already chose: a refit-stable input to the KS/Bragg
   tests. It must be aimed at the class above: CheckSTM_Michel `shape_flat` and the `no_bragg` contrast, as well as
   TaggerCheckSTM's eval sign test.
   - Prototype offline on the existing arms first (reproduce the recorded `ks_mu` / `ks_flat` and `ks1` / `ks2`).
     Then a default-OFF knob pair with doctests and the byte-identical gate.
   - Graded on this union record plus labels for any new candidates.
   - Rule: flips between trajectory-off and trajectory-on ≤ 0.6× today's, and every A1 metric ≥ A0 − 0.02.
3. **PDHD `plateau_off_mip`.** Re-deriving `mip_dqdx_median` for the both-on scale is a config-only arm aimed at 3 of
   8 PDHD STM false positives. The owner excluded the MIP-reference arm from the lever choice, so it is listed, not
   proposed.

The combined flip (fit knobs + `charge_stepped`, with any lever) would again be graded as one unit and presented,
not pushed.

## 8. Not concluded

* **Owner review.** Every sec 5–7 number uses agent labels on 95 PDHD / 182 PDVD items. It is restated after the
  owner reviews queue A.
* **PDVD run 039349 readout window in these cells.**
  - The d51vclus pctrees behind all four PDVD cells carry `readout_window_ticks` = 10000. Run 039349's frames are
    6400 ticks, and current production already uses 6400 (doc pdvd/99; owner 2026-09-13).
  - So TaggerCheckSTM's readout-edge guard never fires at the real edge in these cells. Production-cell TaggerCheckSTM
    accepts 142 stops at ticks 6280–6420 on run 039349, against 1–2 on the other runs.
  - The old record calls 119 of those non-stoppers, and CheckSTM_Michel rejects 115 of them.
  - All four cells share it, so A1 vs A0 is fair. Absolute PDVD numbers on that run are not today's production, and
    part of the PDVD candidate churn is frame-edge exits.
* **PDVD CheckSTM_Michel separation** against the verdict-visible old record is biased toward production by an
  unknown amount (sec 2).
* **Items the scan could not label.** 43 candidates with no payload are outside the population; none is tagged in A0
  or A1. The short-track prep cut of the scan tool sets that boundary.
* **Per-lever cells K and S** are reported, not decided on. On PDVD the sampler alone costs Michel purity most (0.799).
* **No PDVD rubric version for the frame edge.** A label rule change there is the owner's.

## 9. Files

* **Scripts** (`scripts/`):
  - diagnosis: `d103_churn_census.py`, `d103_eval_attrib.py`, `d103_separation.py`, `d103_scale.py`,
    `d103_bounds.py`, `d103_fp_list.py`;
  - the scan: `d103_scan_set.py`, `d103_audit.py`, `d103_scan_record.py`;
  - the grade: `d103_union_grade.py`, `d103_fp_classes.py`, `d103_fp_path.py`, `d103_owner_queue.py`.
* **Records** (new): `pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json`,
  `pdvd/docs/scan/pdvd_stm_michel_smx10_verdicts.json`.
* **Figures and tables** (`figs/`):
  - rule: `103_pred.txt` / `.sha256`, `103_pred_amend1.txt` / `.sha256`;
  - validation: `103_v1_fixed_{pdhd,pdvd}.txt`;
  - census and mechanism: `103_churn_*`, `103_eval_attrib_*` (6), `103_separation*.txt`, `103_scale_*`,
    `103_bounds_*`, `103_moves_*.tsv`;
  - the scan: `103_no_payload_*.tsv`, `103_scan_smx28_pdhd.txt`, `103_scan_smx10_pdvd.txt`,
    `103_scan_reports_{pdhd,pdvd}.md`;
  - the grade: `103_union_grade_*`, `103_fp_classes_*`, `103_fp_path_*`, `103_owner_queue_{pdhd,pdvd}.md`.
* **Scratch** (not committed): `/home/xqian/tmp/d103/`, holding the item lists with roles, preps, frames, agent
  records, task files, rubric copies and transcripts map.

## 10. Round 2 (2026-09-15): the owner's adjudication and the production lineage

The owner asked what it takes to turn both levers on. This round:
* found two defects in round 1's comparison;
* ran the owner's adjudication of the contested PDHD false positives;
* re-ran PDVD on its production inputs.

### 10.0 Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs; O=/home/xqian/tmp/d103/own
R27=$IMG/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json; N28=$IMG/pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json
OWN=$IMG/pdhd/docs/scan/pdhd_stm_michel_own103h_verdicts.json
for n in 2 3 4 5; do (cd $F && head -1 103_pred_amend$n.sha256 | sha256sum -c); done
# 10.1 PDHD truth smx27 (amend3).  With D103_* unset every round-1 figure reproduces byte for byte.
D103_PDHD_RECORD=$R27 python3 $S/d103_union_grade.py --det pdhd --new-record $N28 > $F/103_union_grade_pdhd_smx27.txt
D103_PDHD_RECORD=$R27 python3 $S/d103_fp_classes.py  --det pdhd --new-record $N28 > $F/103_fp_classes_pdhd_smx27.txt
# 10.2 the owner set (amend2), the display, the fold
D103_PDHD_RECORD=$R27 python3 $S/d103_owner_scan_set.py --det pdhd --new-record $N28 --out $O/set_pdhd_smx27
(cd $IMG/pdhd/stm_michel_scan && ./serve_stm_michel_scan.sh 5017 --det pdhd --scan-tag own103h --manifest $O/set_pdhd_smx27/manifest.tsv \
    --prepdir $O/set_pdhd_smx27/prep --questions $O/set_pdhd_smx27/questions.json --dead-points)
python3 $S/d103_owner_scan_score.py --set $O/set_pdhd_smx27 --labels $IMG/pdhd/work/stm_michel_labels/own103h/labels.json \
    --record-out $OWN > $F/103_own103h_pdhd.txt
# 10.3 the re-grade: headline with amend5 (--stm-only-unset-negative), the reading without it, the Michel FP list
D103_PDHD_RECORD=$R27 python3 $S/d103_union_grade.py --det pdhd --new-record $N28 --owner-record $OWN --stm-only-unset-negative \
    > $F/103_union_grade_pdhd_own103h_stmonlyneg.txt
D103_PDHD_RECORD=$R27 python3 $S/d103_union_grade.py --det pdhd --new-record $N28 --owner-record $OWN > $F/103_union_grade_pdhd_own103h.txt
D103_PDHD_RECORD=$R27 python3 $S/d103_michel_fp_list_own.py > $F/103_michel_fp_pdhd_own103h.txt
# 10.5 PDVD on the production lineage (pin = doc 102's libpin_d102, clus md5 091e142b9481)
PIN=/home/xqian/tmp/d102/libpin_d102
ARM=d103v0 DET=pdvd SRC=p100flip JOBS=3 PIN=$PIN LOGD=/home/xqian/tmp/d103/arms/arm_d103v0 bash $S/d102_run_arms.sh
ARM=d103v1 DET=pdvd SRC=p100flip JOBS=3 PIN=$PIN LOGD=/home/xqian/tmp/d103/arms/arm_d103v1 \
    PR_TLA="-A trackfitting_config=$F/101_tf_prod_pdvd_kf.json -S retile_sampler_strategy='charge_stepped'" bash $S/d102_run_arms.sh
python3 $S/d103_pdvd_prod_look.py > $F/103_pdvd_prod_look.txt      # reads the carry in /home/xqian/tmp/p100 (== the committed copy)
D103_PDVD_CELLS=d103v0,d103v1 D103_PDVD_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_p99rwon_carried_corrected_verdicts.json \
    python3 $S/d103_pdvd_items.py --out /home/xqian/tmp/d103/items/pdvd2_items.tsv                  # amend4 sec 3
```

### 10.1 Two defects in round 1's comparison

* **The PDHD truth record.**
  - Sec 5 graded PDHD on `smx22`. The PDHD truth record since doc pdhd/25 is `smx27`: the same 317 keys plus 20
    owner rulings from 2026-09-12. Example: 028084_3/72, agent THRU → owner STM_MICHEL.
  - Amendment 3 corrects it, and the smx22 figures stay.
  - On smx27, before any new owner label, PDHD still reads D2:
    - `is_stm` purity 0.984 → 0.929 (−0.055);
    - efficiency 0.605 → 0.585 (−0.020, at the limit);
    - Michel purity 0.958 → 0.818 (−0.140); Michel efficiency +0.027.
  - V2 on smx27 is 2 / 18 (11 %).
* **The PDVD cells were not production.**
  - `d101vnew` sits on the `d16vnu` / `d51vclus` pctrees: readout window 10000, pre-gain-flip imaging, 596 candidates.
  - PDVD production since doc 100 §7.5 is `p100flip`: window 6400, SP top gain 0.889, 540 candidates.
  - Only 36 of the 596 keys map by cluster id and a stop within 2 cm. So sec 5's PDVD grade says nothing about a
    PDVD production flip.
* **PDHD's A0 is production.** `d101hnew` and `h28prod` have identical `T_stm_michel` on 61 / 61 events.

### 10.2 The owner's adjudication, PDHD `own103h`

* **Set** (amendment 2, rebuilt on smx27 by amendment 3):
  - 19 tier-1 items, each an agent-labelled item that is a false positive in exactly one of A0 / A1;
  - 8 controls, shuffled in;
  - all shown on A1, behind one identical question panel.
* **The session.** Served on :5017; 27 / 27 labelled; no other label tag changed. **Not blind**: every item was
  revealed before it was labelled.
* **Tier 1**, read with amendment 5:

  | Charged with | FP confirmed | Now a TP | Leaves the population |
  |---|---|---|---|
  | `is_stm` (7) | 2 | 1 | 4 (MESSY / UNCLEAR) |
  | Michel (12) | 4 | 4 | 4 (2 THRU, 2 MESSY) |

* **Controls.** None of the 8 changed stopper class. On the Michel call 2 of 8 flipped, one each way:
  - 028084_25/84: Michel → detached dots;
  - the shared false positive 029107_4/58: detached dots → attached.

  So the attached-versus-detached call is itself unstable at this size.
* **The owner's rulings** (amendment 5):
  - *"STM with no Michel, there is no Michel there."* An owner STM_ONLY with no kind counts as no Michel.
  - *"when I say not Michel, likely the Michel is too low energies, or we cannot see them."* The chain's Michel on
    such a stopper stays a false positive.

### 10.3 PDHD re-grade after the adjudication

`103_union_grade_pdhd_own103h_stmonlyneg.txt` (the headline, amendment 5). Truth: own103h > smx27 > smx28.

| A0 → A1 | Value | Change | Limit −0.02 |
|---|---|---|---|
| `is_stm` purity | 0.984 → 0.967 | −0.017 | pass |
| `is_stm` efficiency | 0.614 → 0.599 | −0.015 | pass |
| Michel purity | 0.946 → 0.895 | **−0.051** | **fail** |
| Michel efficiency | 0.603 → 0.664 | +0.060 | pass |

* **Reading: D2, on PDHD Michel purity only.** Without amendment 5 (a), Michel purity is 0.972 → 0.928 (−0.045).
  The verdict is the same.
* **The STM pass is fragile.** One more A1 false positive (118 / 122 → 117 / 122) makes it −0.025.
* **Splits** (amendment 2):
  - untouched items: `is_stm` purity −0.001, efficiency −0.022; Michel purity −0.011, efficiency +0.028;
  - Michel purity per lever: K 0.880, S 0.906, A1 0.895.
* **One-sidedness.** Tier 1 can only clear a charged false positive. D1 is not reached, so amendment 2's TP-mover
  sample is not triggered.

### 10.4 What the PDHD Michel cost is

`103_michel_fp_pdhd_own103h.txt`: A0 has 4 Michel false positives and A1 has 9; 3 are shared. The A1-only ones:

| A1-only | Owner truth | The chain's Michel |
|---|---|---|
| 029107_19/111 | STM, detached dots | 0.2 cm attached, 1.1 MeV |
| 028084_10/109 | STM, detached dots | 1.1 cm attached, 2.4 MeV |
| 029107_23/41 | STM, no Michel | charge-only piece 0.8 cm away, 7.7 MeV |
| 029107_12/95 | STM, no Michel | 6.5 cm attached, 18.8 MeV |
| 029107_27/39 | STM, detached dots | 14.6 cm bridged, 5.2 cm from the stop, 37 MeV |
| 029107_28/109 | STM, detached dots | 19.4 cm attached, 64 MeV |

A0-only: 028084_2/116, 3.6 cm attached, 7.8 MeV.

* **Every one is a stopper the owner reads as having no visible Michel**, and the chain attaches a piece to it.
  - This is CheckSTM_Michel's Michel **admission**, not the KS / Bragg tests of sec 4.
  - The KS-denoise lever of sec 7 does not act on it.
* **Two are below 2.5 MeV**, so an energy floor would remove them. The other four carry 7–64 MeV and need a rule for
  what the attached piece is.

### 10.5 PDVD on the production lineage

* **Arms.** `d103v0` (production, no TLA) and `d103v1` (fit knobs + `charge_stepped`), on `p100flip`'s pctrees.
  - `d103v0` equals `p100flip` on 119 / 119 events; 039349_30 has no candidate.
  - Doc 102's g1 failure does not recur: 039349_78 keeps its candidate in `d103v1`.
* **First look** (`103_pdvd_prod_look.txt`; not pre-registered). Doc 100's carried, owner-corrected record covers
  451 of the 681 union candidates.

  | A0 → A1 | On the carried labels | Bound over the 230 unlabelled |
  |---|---|---|
  | `is_stm` purity | 0.963 → 0.961 (−0.002) | −0.141 … +0.008 |
  | `is_stm` efficiency | 0.794 → 0.759 (−0.034) | −0.023 … +0.091 |
  | Michel purity | 0.924 → 0.872 (−0.052) | −0.185 … +0.018 |
  | Michel efficiency | 0.763 → 0.716 (−0.047) | −0.048 … +0.098 |

  - A1 tags the unlabelled candidates far more often than A0 does (48 `is_stm` vs 8). That is sec 3's
    record-conditioning pattern.
  - The 14 A1-only Michel false positives are unadjudicated carried labels.
* **The round in progress** (amendment 4):
  - the carried record is the PDVD truth on this lineage;
  - blind agent labels (`smx11`) go on the unlabelled candidates, plus 20 calibration items;
  - then the owner's adjudication, `own103v`, on :5017.

### 10.6 What a flip takes

1. **PDVD:** the smx11 labels, the grade and the owner's adjudication.
2. **If Michel purity still fails** on either detector: a default-OFF Michel-admission knob aimed at sec 10.4's class,
   not the KS denoise, graded on these records.
3. **The flip as one unit, with the owner's go:**
   - the sampler: `figs/102r2_flip.patch`;
   - the fit knobs: the two keys in `pdhd_track_fitting.json` / `pdvd_track_fitting.json`. They cannot be
     key-suppressed, so the proof is an arm identical to the graded one;
   - the admission knob.

   Gates: the compiled-config proof, the production-configuration arm graded as above, resources and completeness.

### 10.7 Files (round 2)

* **Rules:** `figs/103_pred_amend2.txt` … `103_pred_amend5.txt`, each with its `.sha256`.
* **Scripts:**
  - new: `d103_owner_scan_set.py`, `d103_owner_scan_score.py`, `d103_michel_fp_list_own.py`,
    `d103_pdvd_prod_look.py`, `d103_pdvd_items.py`, and `figs/103_shoot_round.sh`;
  - extended: `d103_union_grade.py` (`D103_PDHD_RECORD`, `D103_PDVD_RECORD`, `D103_PDVD_CELLS`, `--owner-record`,
    `--stm-only-unset-negative`), `d103_audit.py` (a round name), `d103_scan_record.py`, `d103_fp_classes.py`,
    `d103_owner_queue.py`.
  - With the environment and the new options unset, the round-1 figures reproduce byte for byte:
    `103_union_grade_{pdhd,pdvd}`, `103_v1_fixed_pdvd`, `103_owner_queue_pdhd`.
  - The exception is `103_fp_classes_*`. Round 1 printed its Counter dicts in set-iteration order, which follows
    Python's hash seed. They are now printed sorted and regenerated; the counts are unchanged.
* **Records:**
  - `pdhd/docs/scan/pdhd_stm_michel_own103h_verdicts.json` (owner, 27 items);
  - `pdvd/docs/scan/pdvd_stm_michel_p99rwon_carried_corrected_verdicts.json`, a byte copy of doc 100 r2's corrected
    carry.
* **Figures:** `103_union_grade_pdhd_smx27.txt`, `103_fp_classes_pdhd_smx27.txt`, `103_own103h_pdhd.txt`,
  `103_union_grade_pdhd_own103h{,_stmonlyneg}.txt`, `103_michel_fp_pdhd_own103h.txt`, `103_pdvd_prod_look.txt`.
* **Still on smx22 truth** (round-1 diagnostics, not re-run): `103_separation*`, `103_bounds_pdhd`, `103_moves_pdhd`,
  `103_fp_path_pdhd`, `103_owner_queue_pdhd`.

## 11. Round 3 (2026-09-15): PDVD labels, grade and owner set on the production lineage

The grade here is provisional: the new items carry agent labels. The owner's adjudication `own103v` is in progress.
Nothing here changes a default.

### 11.0 Repro

```bash
# names from 10.0; amendment 4's cells and truth
export D103_PDVD_CELLS=d103v0,d103v1 D103_PDVD_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_p99rwon_carried_corrected_verdicts.json
N11=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx11_verdicts.json; RD=/home/xqian/tmp/d103/round_pdvd2
bash $F/103_shoot_round.sh pdvd round_pdvd2 3        # blind shots; then 11 agents on RUBRIC.md d760e223, each transcript audited
python3 $S/d103_scan_record.py --det pdvd --round $RD --items /home/xqian/tmp/d103/items/pdvd2_items.tsv --tag smx11 \
    --record-out $N11 > $F/103_scan_smx11_pdvd.txt
python3 $S/d103_union_grade.py --det pdvd --new-record $N11 > $F/103_union_grade_pdvd_prod.txt
python3 $S/d103_fp_classes.py  --det pdvd --new-record $N11 > $F/103_fp_classes_pdvd_prod.txt
python3 $S/d103_window_edge.py > $F/103_window_edge_pdvd_prod.txt
python3 $S/d103_owner_scan_set.py --det pdvd --new-record $N11 --out $O/set_pdvd_prod
(cd $IMG/pdhd/stm_michel_scan && ./serve_stm_michel_scan.sh 5017 --det pdvd --scan-tag own103v --manifest $O/set_pdvd_prod/manifest.tsv \
    --prepdir $O/set_pdvd_prod/prep --questions $O/set_pdvd_prod/questions.json --dead-points)
# 103_scan_reports_pdvd_prod.md is $RD/logs/report_w1_a{0..10}.md concatenated, each followed by its audit line
```

### 11.1 The blind labels, `smx11`

* **Items** (amendment 4 sec 3): the 230 unlabelled candidates of `d103v0` ∪ `d103v1`, plus 20 calibration items
  from the carried record. 239 have a record; the 11 without one are listed in `103_scan_smx11_pdvd.txt`.
* **Scan.**
  - 11 agents, 22 items each. Every record carries the rubric sha `d760e223`.
  - Every transcript was audited. The one flag (w1_a6) is a false positive: a grep inside its own output directory.
    It is adjudicated in `103_scan_reports_pdvd_prod.md`.
  - w1_a0 ended on an API error after writing all 22 of its records.
* **Verdicts.** THRU 87, STM_ONLY 51, STM_MICHEL 39, UNCLEAR 18, FRAG_THRU 12, MESSY 11, FRAG_STM_ONLY 1. Confidence:
  medium 131, high 74, low 14.
* **V2 calibration.** 19 items are judged on both sides. 2 differ on stopper class (11 %, limit 25 %): **pass**.

### 11.2 The provisional grade

`103_union_grade_pdvd_prod.txt`. Truth: the carried record first, then `smx11`. The population is 618: 428 items
from the record and 190 from `smx11`.

| A0 → A1 | Value | Change | Limit −0.02 |
|---|---|---|---|
| `is_stm` purity | 0.964 → 0.942 | **−0.022** | **fail** |
| `is_stm` efficiency | 0.623 → 0.678 | +0.055 | pass |
| Michel purity | 0.908 → 0.863 | **−0.044** | **fail** |
| Michel efficiency | 0.686 → 0.716 | +0.031 | pass |

* **Reading: D2 on both purities, provisional.**
  - The `is_stm` failure is one false positive. A1 at 15 instead of 16 gives 0.945 (−0.018).
  - Michel purity needs about 5 of A1's 20 A1-only false positives to clear.
* **Both efficiencies rise.** That is sec 3's swap, now that the gains are labelled.
* **False-positive classes** (`103_fp_classes_pdvd_prod.txt`):
  - **`is_stm`:** A0 has 9, A1 has 16. All 13 A1-only ones are THRU by hand: 7 new blind-agent labels, 5 carried
    agent labels, 1 owner.
  - **Michel:** A0 has 16, A1 has 26. The 20 A1-only ones split into two classes:
    - 12 on through-going tracks;
    - 8 on stoppers read without a visible Michel (4 detached dots, 4 no kind). This is sec 10.4's PDHD class.

    Sources: 13 carried agent labels, 5 new agent labels, 2 owner.

### 11.3 The readout-window edge

* **The gap.** Run 039349's recorded frame closes at slice ~1595 and opens at slice 0. The rubric has no rule for a
  fit end at either edge.
* **The class** (`103_window_edge_pdvd_prod.txt`): a hand-curated list, because a phrase search both over- and
  under-matches (the script's header says how).
  - 37 items whose scanner names the edge at the fit end: 32 at the end of the frame, 5 at its start.
  - 36 are on run 039349 and 1 on 039253.
* **The scanners split.**
  - 31 call THRU: the edge read as a dead region, "no measurement".
  - 6 call STM_ONLY: w1_a5 4 of its 6 ("rubric has no rule for this"), w1_a4 1, w1_a10 1.
* **It does not carry the purity cost.**
  - Neither arm tags any of the 6 stopper calls. Reading them as THRU leaves both purities unchanged and moves
    `is_stm` efficiency A1 − A0 by +0.001.
  - The class holds 3 A1-only false positives, all in `own103v`: 039349_53/28 and 039349_76/32 (`is_stm`) and
    039349_4/76 (Michel).
  - With the class removed: `is_stm` purity −0.015, Michel purity −0.040.
* **A ruling is still needed** so the record is consistent: is a fit end at the frame edge a dead region, or does
  rule 3 apply as written?

### 11.4 The owner set `own103v` (in progress)

* **Built by** amendment 2's rules in amendment 4's PDVD form:
  - tier 1 = judged, non-owner items that are a false positive in exactly one of A0 / A1: 32, under the 40 cap;
  - 8 controls (`random.Random(103)`);
  - 40 items, behind the same question panel as `own103h`;
  - shown on `d103v1`; items that are candidates only in A0 are shown on `d103v0`.
* **Served** on :5017. The sha256 of every other label tag is recorded before (`label_shas_before_own103v.txt`).
* **After the session:**
  - fold with `d103_owner_scan_score.py --det pdvd`;
  - re-grade with `--owner-record`, plus amendment 2's splits;
  - read the controls on both axes, stopper class and Michel call. On PDHD the Michel call moved on 2 of 8 controls.

### 11.5 What a flip takes (updates sec 10.6)

1. **PDVD `own103v`, then the re-grade.**
   - `is_stm` purity passes if one A1-only false positive clears.
   - Michel purity needs about 5.
2. **If Michel purity still fails on either detector:** the Michel-admission knob of sec 10.6.
   - On PDHD it targets sec 10.4's class.
   - On PDVD it targets the same stopper class (8), plus the Michels the chain finds on through-going tracks (12).
3. **The flip as one unit, with the owner's go.** Unchanged from sec 10.6.

### 11.6 Files (round 3)

* **Scripts:**
  - new: `d103_window_edge.py`;
  - extended:
    - `d103_owner_scan_set.py`: PDVD, tag `own103v`, tier-1 cap 40, PDVD Michel truth = verdict STM_MICHEL, an
      A0-only item shown on A0;
    - `d103_owner_scan_score.py`: `--det`;
    - `d103_fp_classes.py`: two-cell lineages.
  - With the new options unset, the committed PDHD and round-1 outputs reproduce byte for byte:
    - `103_fp_classes_{pdhd,pdvd,pdhd_smx27}`;
    - `103_own103h_pdhd` and the `own103h` record. Only the output-path line differs.
* **Record:** `pdvd/docs/scan/pdvd_stm_michel_smx11_verdicts.json`, 239 items, sha256 `5302d87c`.
* **Figures:** `103_scan_smx11_pdvd.txt`, `103_scan_reports_pdvd_prod.md`, `103_union_grade_pdvd_prod.txt`,
  `103_fp_classes_pdvd_prod.txt`, `103_window_edge_pdvd_prod.txt`.

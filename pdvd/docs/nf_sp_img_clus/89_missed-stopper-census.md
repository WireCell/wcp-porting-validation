# 89 — The missed stoppers, by the check that rejects each (doc 88 §9.5 item 1)

After docs 78–88, which were mostly about Michels, the largest gap in the PDVD chain is the stoppers it misses. The owner asked to start with doc 88 §9.5's first item: a census of the missed stoppers by the check that rejects each, on today's production and the record that includes smx6.

**Status (2026-09-11): DONE. Read-only: no toolkit, config or record change.**

- **Two denominators.**
  - §9.5's "45" is the 546 judged items that have a candidate: `is_stm` 233 / 7 / **45** (efficiency 0.838). That is `census_score`'s population.
  - On all 576 judged items it is 233 / 7 / **55** (0.809). An item with no candidate counts as `is_stm` 0; that is doc 77's convention.
  - So 55 = 45 missed by the chain + 10 the STM tagger never hands on.
  - Doc 77 had 58 on this convention. Three left the missed set since, and none entered (§6).
- **The partition of the 55 (§4) is the result.**

  | part | n | what it is |
  |---|---:|---|
  | (a1) the chain can act, 0 FP | **5** | two existing settings together clear them on the record with no false stopper: the P1 floors at 3 MeV / 3 cm, and `plateau_mip_hi` 2.0 |
  | (a2) a Michel object the floors do not reach at 0 FP | 6 | lengths 0.9–2.6 cm; every through-going item with a Michel-like arm is under 2.5 cm |
  | (b1) re-judge first | **22** | the shape tests only, no Michel object, a medium- or low-confidence single-scanner verdict |
  | (b2) no lever on this record | 4 | the same, high confidence, all STM_ONLY |
  | (c) closed by the owner's preference | 8 | fiducial, continuation, hadron guard |
  | (d) the tagger's, no candidate | 10 | never handed on |

- **The shape tests are not borderline.**
  - `shape_flat` misses sit a median 0.038 past the margin (25 of 31 by more than 0.02).
  - `no_bragg` misses reach a median 0.71 of the required contrast (20 of 23 under 0.9).
  - So no threshold nudge recovers them. `bragg_contrast_min` 0.5 buys 1 stopper for 15 through-going items.
- **The tagger's 10 (§3).** Six are its dQ/dx eval (status 3).
  - The tagger's own combined score does separate four of them from all 19 through-going status-3 items on the record, by a gap of 0.009.
  - As a rule applied to every status-3 cluster, that waiver hands the chain 12 to 34 clusters, nearly all unjudged, for at most 3 of the missed stoppers.
  - The record cannot grade that. It is not a flip candidate.
- **Recommendation.** Flip the (a1) bundle next: +5 stoppers, 0 FP, config only, no C++. The owner decides, because it lowers the P1 energy floor (§5.1).

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts; S=$IMG/pdvd/docs/scan
# production = p88vprod (doc 88 sec 9.4; toolkit a4bba314, wcp 6e722ad2).  Its payloads, in scratch:
cd $IMG/pdhd/stm_michel_scan && ./prep_stm_michel_scan.py --det pdvd --arm p88vprod \
    --outdir /home/xqian/tmp/p89/prep_p88vprod --sheetdir /home/xqian/tmp/p89/sheet_p88vprod \
    --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv
# (provenance: 585 / 585 payloads identical to doc 88's prep_p88v5fr once the arm label is masked;
#  /home/xqian/tmp/p89/provenance.txt)
STM_SCAN_RECORD=$S/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_verdicts.json python3 $X/d89_miss_census.py \
    --prep /home/xqian/tmp/p89/prep_p88vprod --arm p88vprod --cfg /home/xqian/tmp/p88/proofs/post.json \
    --prep-old /home/xqian/tmp/p75/prep_p75vprod --arm-old p75vprod --json /home/xqian/tmp/p89/census.json
```

`/home/xqian/tmp/p88/proofs/post.json` is the compiled production config (doc 88 §9.4's proof C). The script exits unless `STM_SCAN_RECORD` names the smx6 record; checked with the smx5 record, rc 1. Output: `/home/xqian/tmp/p89/census.txt`.

**Every threshold is read from the compiled config, or, where the key is absent there, from the C++ member initializer; the script prints the source of each** (§0 of its output).
- *Compiled:* `ks_margin` −0.02, the plateau window [0.6, 1.6] × 55000 e/cm, the tagger's `mip_dqdx` 55000 and `michel_res_length_cut` 6.5 cm.
- *C++ default:* `bragg_contrast_min` 0.6, `topology_michel_ke_min` 10 MeV, `topology_michel_len_min_cm` 3, the continuation window (20°, 3 cm, 0.7–1.3 MIP), and the tagger's `accept_guards` (off).

**What was seen before this doc's predictions.** While planning I re-ran doc 77's `d77_review.py` read-only on this arm and record, so the bucket counts were known before anything was predicted. `/home/xqian/tmp/p89/pred.txt` (13:07:42, before `d89_miss_census.py` existed) pre-registers only what had not been measured: the per-bit margins, the tagger's negative control, and the movement since doc 77. They are graded in §7.

## 1. The two denominators

| population | n | `is_stm` TP / FP / FN / TN | purity | efficiency | F1 |
|---|---:|---|---:|---:|---:|
| judged items with a candidate (§9.5's "45") | 546 | 233 / 7 / 45 / 261 | 0.971 | 0.838 | 0.900 |
| all judged items, no candidate = 0 (doc 77's) | 576 | 233 / 7 / 55 / 281 | 0.971 | 0.809 | 0.883 |

The ten extra misses on the second row are §3's. Every other count in this doc uses the second row.

## 2. The 45 with a candidate, by the deciding bits, with each bit's margin

Grouping is by the full set of reject bits: an item flips only when every bit clears.

| deciding bits | n | STM_MICHEL | a Michel object | margin (median, range) |
|---|---:|---:|---:|---|
| `no_bragg` + `shape_flat` | 17 | 11 | 6 | contrast 0.71 × required (0.26–0.97); ks 0.050 past the margin (0.005–0.076) |
| `shape_flat` | 10 | 4 | 3 | ks 0.036 past (0.013–0.059) |
| `plateau_off_mip` | 5 | 0 | 0 | plateau 0.25, 0.34, 0.41, 0.59 and 1.79 MIP (window 0.6–1.6) |
| `no_bragg` | 4 | 1 | 1 | contrast 0.24, 0.71, 0.79, 0.99 × required |
| `no_bragg` + `plateau_off_mip` | 1 | 1 | 0 | 0.97 × required; plateau 0.598 MIP |
| `stop_near_boundary` (± `shape_flat`) | 4 | 3 | 3 | stop outside the curved FV (tolerance −25 / −50 cm) |
| `continuation` | 2 | 1 | 1 | the arm is 11–16° (< 20°), 3.6–4.7 cm (> 3), 0.71–0.97 MIP |
| `continuation` + boundary + `shape_flat` | 1 | 1 | 1 | 11.2°, 4.2 cm, 1.07 MIP |
| `vertex_hadron` + shape | 1 | 1 | 1 | one interior arm classified hadron |

- **Over all 45:** `shape_flat` is set on 31, and 25 of them sit more than 0.02 past the margin (min +0.005, p25 +0.024, median +0.038, max +0.076). `no_bragg` is set on 23, and 20 of them reach under 0.9 of the required contrast (min 0.24, p25 0.60, median 0.71, max 0.99).
- **Which profile the bits are read on.** The shape bits are read on the anchored profile. Doc 75's geometric fallback, which lets the geometric reading stand, stood on **0** of the 45: none of them passes at the geometric origin either.
- **The stop against the scanner's pin.**
  - Among the 45, 5 carry a placed owner pin (smx4): |stop − pin| is 1.2, 2.3, 4.3, 4.7 and 6.5 cm.
  - 8 carry a smx1a `pin_rr`, the scanner's reading that the fit runs 3.4–9.0 cm past the real stop. On those 8 the anchor moved the stop back only 0.0–2.6 cm: `039252_9/101`, `039253_2/13`, `039349_34/48`, `039349_44/28`, `039349_51/29`, `039349_71/37`, `039349_81/25`, `039349_81/44`.
  - That is doc 78 §2's overshoot mechanism, still unresolved (doc 82's peak-then-drop mover is dead). All 8 are medium- or low-confidence verdicts, so they fall in (b1) (§4).
  - This is a column on 13 items, not a bucket.

## 3. The ten judged stoppers with no candidate: the STM tagger

`T_stm_pass` records why the tagger did not hand the cluster on. Status 3 is set only when every eval row rejects (`TaggerCheckSTM.cxx:4028`, `set_pass_status(flag_pass ? 5 : 3)`).

| status | n | items |
|---|---:|---|
| 3, the dQ/dx eval | 6 | `039252_1/109`, `039253_3/60`, `039253_7/30`, `039349_7/20` (STM_MICHEL, owner); `039349_50/55`, `039349_57/18` (STM_ONLY, owner) |
| 5, proton endpoint | 2 | `039252_8/72`, `039349_75/67` |
| 2, long leftover past the kink | 1 | `039252_9/49` |
| 7, a pass guard | 1 | `039349_33/60` |

The candidate cap (doc 79) now fires on 0 events.

### 3.1 The first eval condition each row fails

`T_stm_eval` holds 8–16 rows per rejected pass. These are the (peak_range, offset, compare_range) variants the tagger tries in turn (`:3904–3943`), and the pass is accepted if any row accepts. For each row the script re-runs `eval_stm_core_impl`'s tests in code order (`:2968–3017`) from the recorded `ks1`, `ks2`, `ratio1`, `ratio2`, `res_length` and `ave_res_dqdx`:
- **F1:** `ks1 − ks2 ≥ 0`, the muon template fits no better than flat.
- **F2:** near-flat, with the combined score `comb = ks1 − ks2 + (|r1−1| − |r2−1|)/1.5 × 0.3` > −0.02.
- **F3:** a straight residual. It needs `res_length1` / `res_dis1`, which are not recorded; the script says "F3 or" wherever F3 cannot be excluded.
- **F4:** the residual is not Michel-like.
- **F5:** no accept branch fires.

The recomputed verdict agrees with the recorded one on every row: 0 mismatches.

The tree's lengths are in cm: peak_range reads 40 / 20 and compare_range 35 / 15, the code's constants. A first pass read them as mm and produced exactly one mismatch, `039253_2/50`, a row that fails F4's 20 cm clause; with cm it matches.

**The best row per item.** This is the row that reaches the deepest test, ties broken by the smallest `ks1 − ks2`:

| | best-row stage | `ks1 − ks2` | comb |
|---|---|---|---|
| the 6 status-3 stoppers | F1 on 4, F2 on 2 | −0.033 … +0.024 | **−0.027, −0.020, −0.018, −0.012**, +0.036, +0.104 |
| the 19 status-3 THRU items on the record | F5 on 14, F1 on 3, F2 on 1, F4 on 1 | −0.111 … +0.047 | −0.003 … +0.101 |

On the first condition the two populations are the same: `ks1 − ks2` ranges overlap. The stoppers die earlier, at F1, while most through-going items get as far as F5 and fail to find an accept branch.

The combined score is a different matter. `039252_1/109`, `039253_3/60`, `039253_7/30` and `039349_57/18` read −0.027 to −0.012, and every negative reads −0.003 or above. That is a separation on this record, with a gap of 0.009 on 25 items. The code's own F5 accept branch is `comb < 0`, but F1 rejects these rows before they reach it.

### 3.2 The waiver the separation suggests, on the whole population

The candidate rule: F1 does not reject a row whose comb is below *t*, and the row must then pass F2–F5 as written. Applied to **every** cluster in the 120 events with a status-3 pass and no candidate: 695 clusters, of which 670 unjudged, 19 through-going and 6 stoppers.

| *t* | handed on (exact) | + unless the unrecorded F3 fires |
|---|---|---|
| −0.020 | 2 stoppers, 7 unjudged | 1 stopper, 1 THRU (`039253_2/50`), 14 unjudged |
| −0.010 | 2 stoppers, 10 unjudged | 1 stopper, 1 THRU, 20 unjudged |
| 0.000 | 2 stoppers, 12 unjudged | 1 stopper, 1 THRU, 27 unjudged |

- **The stoppers it reaches.** `039252_1/109` and `039253_3/60` exactly, and `039253_7/30` unless F3 fires. `039349_57/18`'s best row fails F2, not F1, so the waiver does not reach it.
- **The unjudged clusters dominate.** The judged separation (4 stoppers, 0 negatives) does not survive contact with the population: a waiver hands on 12–34 clusters, nearly all never scanned.
- **Only candidacy is known.** A handed-on cluster becomes a `CheckSTM_Michel` candidate, and the chain's own shape tests then decide `is_stm`. That verdict is not predictable offline.
- **Blast radius.** `TaggerCheckSTM` is shared with PDHD and SBND.

**This lever needs a blind scan of the new candidates plus an arm. It is not ranked for a flip.**

## 4. The partition of all 55

Parts are assigned in the order d, c, a, b.

- **(d) the tagger's (10).** §3.
- **(c) closed by the owner's preference (8).**
  - The fiducial rule (doc 44) and the continuation and hadron guards, which P1 does not clear by design (doc 70 §3.3, doc 77 §7.5):
    - `039252_2/103` (28.9 MeV Michel, stop outside the FV);
    - `039349_27/41`, `039349_35/30`, `039349_65/40` (boundary);
    - `039349_5/65` (34.7 MeV Michel, a 15.7° / 4.7 cm continuation arm);
    - `039349_51/21` (continuation);
    - `039349_76/23` (hadron);
    - `039349_78/22` (all three).
  - Six are owner-confirmed STM_MICHEL.
- **(a1) the chain can act at 0 FP (5).** The bundle of two existing settings clears them (§5):
  - `039349_19/52` (5.9 MeV / 4.2 cm), `039349_48/21` (8.7 / 3.9), `039349_63/55` (3.3 / 3.8), `039349_70/61` (5.9 / 4.1): owner-confirmed STM_MICHEL, attached Michels under the 10 MeV energy floor;
  - `039252_6/106`: STM_ONLY, high confidence, plateau 1.79 MIP.
- **(a2) a Michel object the 0-FP bundle does not reach (6).** `039252_12/90` (0.9 cm), `039253_12/93` (2.6 cm), `039253_8/64` (14.1 MeV, 2.5 cm), `039349_22/56` (13.6 MeV, 2.3 cm), `039349_38/57` (2.0 cm), `039349_68/63` (1.3 cm).
  - All six fail the 3 cm length floor.
  - Lowering it to 2 cm gains `039253_8/64`, `039349_22/56` and `039349_38/57`, but admits `039253_8/31` (2.4 cm) and `039349_0/68` (2.1 cm), both owner-judged THRU.
  - A floor at 2.5 cm would sit between them on this record; that is a one-item margin, and it is not proposed.
- **(b1) re-judge first (22): the shape tests only, no Michel object, medium or low confidence.** 10 STM_MICHEL and 12 STM_ONLY, all single-scanner smx1a verdicts. §2's eight overshoot items are among them.
  - The chain has no positive evidence on these: no Michel, a profile that reads flat or Bragg-less by the margins of §2.
  - Doc 68 §3 and doc 70 §9 are the precedents: the owner's re-judges overturned a third of such verdicts in both directions.
  - Before any fix is aimed at the 22, a blind owner re-judge says how many are stoppers.
- **(b2) no lever on this record (4).** High confidence, STM_ONLY, no Michel object:
  - `039252_16/108`: plateau 0.25 MIP, a charge-scale question (doc 77 §7.4);
  - `039349_2/38`, `039349_26/18`, `039349_55/24`: `shape_flat` 0.039 / 0.024 / 0.013 past the margin. `039349_55/24` comes back only at `ks_margin` −0.04, with 4 THRU.

**The five high-confidence misses are where the chain is most likely simply wrong.** They are `039252_6/106` (a1), `039252_16/108`, `039349_2/38`, `039349_26/18` and `039349_55/24` (b2). One of the five has a lever.

## 5. Levers, on all 576 judged items

| setting | exactness | `is_stm` TP / FP / FN | gains (record) | new FP |
|---|---|---|---|---|
| production | — | 233 / 7 / 55 | — | — |
| P1 floors 3 MeV / 3 cm | exact | 237 / 7 / 51 | `039349_19/52`, `039349_48/21`, `039349_63/55`, `039349_70/61` (owner STM_MICHEL) | — |
| P1 floors 0 MeV / 3 cm | exact | 237 / 7 / 51 | the same four: the energy floor does nothing below 10 MeV once the length floor is 3 cm | — |
| P1 floors 10 MeV / 2 cm | exact | 235 / 7 / 53 | `039253_8/64`, `039349_22/56` | — |
| P1 floors 3 MeV / 2 cm | exact | 241 / 9 / 47 | + `039253_12/93`, `039349_38/57` | `039253_8/31`, `039349_0/68` (THRU, owner) |
| plateau window [0.6, 2.0] | lower bound | 234 / 7 / 54 | `039252_6/106` (STM_ONLY, high) | — |
| **bundle: P1 3 / 3 + [0.6, 2.0]** | **lower bound** | **238 / 7 / 50** | the five of (a1) | **—** |
| plateau window [0.5, 1.6] | lower bound | 234 / 8 / 54 | `039349_7/65` | `039349_77/26` |
| `ks_margin` −0.03 / −0.04 | lower bound | 233 / 8 / 55; 234 / 11 / 54 | — ; `039349_55/24` | 1; 4 THRU |
| `bragg_contrast_min` 0.5 / 0.4 | indicative | 234 / 22 / 54; 236 / 26 / 52 | `039252_12/123`; + `039349_19/52`, `039349_3/47` | 15; 19 THRU |

How the exactness labels were earned:
- **The P1 floors are exact.** They have one consumer, `stm_michel_topology_clear` at `CheckSTM_Michel.cxx:3989`. It runs last, after every other bit is final, on recorded inputs: the Michel's found flag, connection type, `michel_ke_best` and `michel_len`.
- **The plateau window and `ks_margin` are exact lower bounds: every named gain is exact, and the totals are floors.**
  - The anchored reading sets both from recorded values.
  - Doc 75's geometric re-read (`:2803–2846`) runs only when the anchored reading still carries a shape bit, and it can only clear bits. So an item the anchored reading clears is gained, whatever the geometric reading says.
  - A looser setting can additionally let the geometric reading clear items whose anchored reading still fails. Its values are not recorded, so those extra gains cannot be counted; the script lists the 22 anchored shape-bit items where they could occur.
- **`039252_6/106`, the bundle's only high-confidence gain, is exact.**
  - Its only bit is `plateau_off_mip`, from the anchored plateau of 1.79 MIP.
  - The arm's log shows the geometric re-read ran on it: `anchor_geo_fallback: cluster 106 shift 0.66 cm peak/plateau 1.55 … geometric … bits 1024 -> anchored stands`.
  - Under `plateau_mip_hi` 2.0 the anchored reading sets no bit, so the re-read never runs.
- **No second consumer.** `plateau_mip_hi` is read only by `CheckSTM_Michel`: the anchored test at `:2729–2731` and the geometric re-read at `:2813–2815`. `topology_michel_ke_min` is read only at `:3992`. A grep over `clus/`, `cfg/` and both ProtoDUNE `wct-pr-perevt.jsonnet` finds no other reader; PDHD's file sets `plateau_mip_hi: 1.6` for its own chain and is not touched.
- **`bragg_contrast_min` is indicative.** R_NO_BRAGG is set at two sites (`:2726`, `:2812`). On this record it is closed anyway.

### 5.1 Why the bundle is the owner's call

- **It moves a threshold on the record it is graded on.** Doc 77 §7.2 raised the same point. The four P1 gains are the owner's own smx4 stoppers.
- **The energy floor does no work at 3 cm on this record.** The 0 MeV / 3 cm row gains the same four. It is the length floor that protects purity, since every through-going Michel-like arm is under 2.5 cm. A 3 MeV floor keeps a guard against tiny objects that this record cannot test.
- **Lowering `topology_michel_ke_min` below 10 MeV breaks an invariant.** Docs 72 and 84 argued that the moved-stop veto and its exemptions "cannot move `is_stm`", because a spared Michel is under `moved_stop_michel_ke_min` (10 MeV) and P1 needs ≥ `topology_michel_ke_min` (10 MeV). The code comment at `CheckSTM_Michel.cxx:664–678` states the same condition: "while that and topology_michel_ke_min stay equal". That argument no longer holds once the floor is lowered:
  - a future change to T2c would then move `is_stm`, and has to be graded on it;
  - the re-verdict above is still exact, because the recorded Michel fields already carry today's T2c outcome.
- **The P1 energy is `michel_ke_best`, the fit-based headline.** At 3–10 MeV it inherits item 9's dependence on the Michel's fit. The 3 cm length floor does not.
- **`plateau_mip_hi` 2.0** admits one high-confidence stopper and nothing else, as in doc 77 §7.4. It is independent of P1.

## 6. Movement since doc 77, on the same record

The census re-run on doc 77's production payloads (`p75vprod`) with the smx6 record: missed **58 → 55**.

| item | `p75vprod` | `p88vprod` | moved by |
|---|---|---|---|
| `039253_8/65` | no candidate | `is_stm` 1 | doc 79, `max_candidates` 64 |
| `039349_81/62` | no candidate | `is_stm` 1 | doc 79 |
| `039349_9/19` | `no_bragg` + `shape_flat` | `is_stm` 1 | doc 83, the near-stop arm, then P1 |

No item entered the missed set, and no non-stopper's `is_stm` moved.

## 7. Predictions (`/home/xqian/tmp/p89/pred.txt`), graded

| # | prediction | result |
|---|---|---|
| P1 | the THRU status-3 items and the six stoppers fail on the same first condition (F1) at overlapping margins, so the tagger lever is closed | **missed on both counts.** The first failing condition differs: stoppers stop at F1 (4) or F2 (2); THRU items mostly at F5 (14 of 19). `ks1 − ks2` overlaps, but the combined score separates 4 of 6 from 0 of 19. The lever is not closed on the judged items. It is un-gradeable on the population (§3.2), which is why it is still not ranked for a flip. |
| P2 | most `shape_flat` misses sit more than 0.02 past the margin | **held**: 25 of 31 |
| P3 | `no_bragg` contrasts well under the threshold (< 0.9 × required on most) | **held**: 20 of 23, median 0.71 |
| P4 | the movement since doc 77 is `039253_0/44`, the two doc 79 items, `039349_9/19`, and `039349_36/63`'s round trip | **partly missed.** The three that moved are exactly the doc 79 pair and 9/19, and 36/63 shows no net change. But `039253_0/44` was already a stopper on `p75vprod`, since doc 75 recovered it. My list confused its Michel (doc 87) with its stopper call. |

## 8. What the record cannot grade

- The 670 unjudged status-3 clusters, and so any tagger-side waiver (§3.2).
- What doc 75's geometric re-read would add under a looser plateau, `ks_margin` or contrast setting. That needs an arm.
- The chain's verdict on a cluster that does not reach it today.
- The 22 (b1) verdicts themselves: a single scanner at medium or low confidence.

## 9. Next, ranked

1. **Flip the (a1) bundle.**
   - The settings: `topology_michel_ke_min: 3.0` (or leave energy at 10 and accept 0 of the 4), and `plateau_mip_hi: 2.0`, both in `stm_michel_knobs`. The knobs exist; no C++.
   - Prediction: 233 / 7 / 55 → 238 / 7 / 50 on all judged items (238 / 7 / 40 with a candidate, efficiency 0.838 → 0.856). That is a lower bound; the 22 anchored items could add. The Michel census should be unchanged.
   - Bar: a bare-production arm, the compiled-config proofs, and the other 16 live jobs unchanged.
   - The two keys differ in kind:
     - `plateau_mip_hi` is already in the compiled config (1.6), so the flip is a value edit.
     - `topology_michel_ke_min` is absent (C++ default 10), so the flip adds a key. That is the inert-key trap of docs 58 and 61.
   - So the proofs are: PRE vs POST differs in exactly these two keys; POST with an override forcing 10.0 / 1.6 diffs to zero against PRE except for the added `topology_michel_ke_min` key itself, and the value-equality is shown.
   - Needs the owner's yes on the energy floor (§5.1).
2. **A blind owner re-judge of the 22 (b1) items (smx7).** This is the largest bucket, and the only one whose size is a record question. 8 of them carry a scanner-moved stop 3.4–9.0 cm back.
3. **Doc 88 §9.5 item 2**: the PDVD flips of docs 83–88 graded on PDHD's smx18 record.
4. **The tagger waiver (§3.2)**, only after a blind scan of its 12–34 new candidates and an arm on both ProtoDUNEs. Its ceiling is 3 stoppers.
5. **Doc 78 items 9 and 8**, as in doc 88 §9.5.

**Recommendation: 1.** It is the only item that adds stoppers without a scan, and it adds no FP. If the owner prefers to keep the 10 MeV floor, `plateau_mip_hi` 2.0 alone is +1, and the bundle waits for 2.

# 77 — The PDVD STM / Michel chain against the hand-scan record, after P1–P5

The owner asked, after P1b and P5, for a review of the PDVD stopping-muon situation: (1) against the hand-scan record, (2) whether there is room left to improve. This is that review. It is read-only: one bare-production arm, one script, no code change.

**Status (2026-09-10): DONE.** Measured on `p75vprod` (toolkit `567a7232`, PDVD production after doc 75's flip; the P5 twin `p76vsame` is output-identical, doc 76 §5). Against the merged smx1a + smx3 + smx4 record (601 items, 576 judged):
- **`is_stm` 230 / 7 / 58** on the 576 judged items (an item with no candidate counts as 0): purity 0.970, efficiency 0.799, F1 0.876. On the 544 judged items that have a candidate: 230 / 7 / 46.
- **`michel_found` 136 / 12 / 29**: purity 0.919, efficiency 0.824, F1 0.869.
- **Room, ranked (§7):** a candidate cap that drops two owner-confirmed Michel stoppers (+2, no scan needed); P1's floors at 3 MeV / 3 cm (+4 owner-judged stoppers, 0 FP on the record); the STM tagger's own dQ/dx eval, which never hands six owner-confirmed stoppers to the chain; the `plateau_mip_hi` bound (+1, 0 FP). The rest of the misses are either the tagger's domain, the fiducial rule's, or need a re-scan of medium-confidence items first.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
# the production arm's prep is doc 75's (/home/xqian/tmp/p75/prep_p75vprod); every table below is this one command
STM_SCAN_RECORD=$REC python3 $X/d77_review.py --prep /home/xqian/tmp/p75/prep_p75vprod --arm p75vprod --json /home/xqian/tmp/p77/review.json
```

Sections 8 and 9 of the script read, in addition, the arm's `wct_pr_*.log` lines (the candidate cap) and its `tracking-stm.root` `T_stm_pass` (the tagger's pass status); section 9's re-verdicts are exact for the three settings it sweeps, each of which has one consumer in `CheckSTM_Michel.cxx` (doc 67's method).

## 1. Where the chain stands

### 1.1 The census, and how it got here

| step | doc | record | `is_stm` TP / FP / FN | purity | eff | `michel_found` TP / FP / FN |
|---|---|---|---|---:|---:|---|
| first CheckSTM_Michel baseline (`d53v`) | 57 | smx1a, 549 scored | 144 / 9 / 125 | 0.941 | 0.535 | 111 / 39 / 41 |
| T1a–T3 (retreat, split, no-trajectory pieces, range-energy guard) | 57–62 | smx1a | 152 / 9 / 116 | 0.944 | 0.567 | 132 / 22 / 20 |
| the owner's operating points (T4–T8) | 67 | smx1a | 177 / 6 / 91 | 0.967 | 0.660 | — |
| 3 cm peak anchor | 68 | smx1a + smx3, 580 judged | 176 / 7 / 108 → 197 / 7 / 87 | 0.966 | 0.694 | 132 / 17 / 32 |
| P1 topology-first (+ `topology_clears_sparse`), P4 gamma collect at 50 cm | 70, 71 | + smx4, 544 with a candidate | 223 / 8 / 53 → 225 / 7 / 51 | 0.970 | 0.815 | 133 / 12 / 25 |
| P3b, P3 (attached-gate exemption; strict retreat tail) | 72, 74 | same | 225 / 7 / 51 | 0.970 | 0.815 | 136 / 12 / 22 |
| **P1b (the anchor's geometric fallback)** | **75** | same | **230 / 7 / 46** | **0.970** | **0.833** | 136 / 12 / 22 |
| the same, on all 576 judged items (no candidate = 0) | this doc | | 230 / 7 / 58 | 0.970 | 0.799 | 136 / 12 / 29 |

The three record versions are not the same population (smx3 added 45 owner re-judges of the hardest items, smx4 the 54 Michel-carrying items P1 was sized on), so the rows are not one curve; each row is against the record its doc used.

### 1.2 By slice (576 judged items, `is_stm` TP / FP / FN / TN)

| slice | n | `is_stm` | purity | eff | F1 | `michel_found` |
|---|---:|---|---:|---:|---:|---|
| smx1a (the original 569-item scan, minus re-judges) | 482 | 192 / 3 / 31 / 256 | 0.985 | 0.861 | 0.919 | 94 / 2 / 18 |
| smx3 (the owner's re-judge of doc 67's disagreements) | 44 | 9 / 4 / 12 / 19 | 0.692 | 0.429 | 0.529 | 2 / 0 / 11 |
| smx4 (the 54 Michel-carrying items, blind) | 50 | 29 / 0 / 15 / 6 | 1.000 | 0.659 | 0.795 | 40 / 10 / 0 |
| confidence high | 285 | 127 / 0 / 5 / 153 | 1.000 | 0.962 | 0.981 | 72 / 1 / 0 |
| confidence medium | 188 | 63 / 3 / 24 / 98 | 0.955 | 0.724 | 0.824 | 22 / 1 / 17 |
| confidence owner | 94 | 38 / 4 / 27 / 25 | 0.905 | 0.585 | 0.710 | 42 / 10 / 11 |
| run 039252 | 85 | 41 / 0 / 11 / 33 | 1.000 | 0.788 | 0.882 | 28 / 4 / 4 |
| run 039253 | 94 | 56 / 0 / 7 / 31 | 1.000 | 0.889 | 0.941 | 31 / 1 / 6 |
| run 039349 | 397 | 133 / 7 / 40 / 217 | 0.950 | 0.769 | 0.850 | 77 / 7 / 19 |

The reading: **on the items the first scanner was sure about, the chain is essentially done** (high confidence: 5 misses in 285, 0 false positives; F1 0.98). The remaining error is concentrated where the record itself is hardest — the owner's re-judges (smx3: 4 of the 7 false positives and 12 of the 58 misses live in 44 items) and the medium-confidence items. All seven false positives are in run 039349.

## 2. Every missed stopper, by mechanism (58)

| deciding mechanism | n | of which STM_MICHEL on the record | of which carry a Michel object | the doc that owns it |
|---|---:|---:|---:|---|
| **no candidate at all** (the STM tagger never handed the cluster on, or the cap dropped it) | 12 | 7 | — | §7.1, §7.3 |
| shape tests only: `no_bragg` + `shape_flat` | 17 | 11 | 6 | doc 65 / 68 / 75 (the anchor), doc 70 (P1) |
| shape tests only: `shape_flat` | 10 | 4 | 3 | same |
| shape tests only: `plateau_off_mip` | 5 | 0 | 0 | doc 48 §6.2, §7.4 |
| shape tests only: `no_bragg` | 4 | 1 | 1 | |
| shape only, other combinations (`no_bragg`+`plateau_off_mip`; `profile_sparse`+`shape_flat`) | 2 | 1 | 0 | |
| `stop_near_boundary` (± a shape bit) | 5 | 4 | 4 | the fiducial rule (doc 44); P1 deliberately does not clear it |
| `continuation` (± boundary) | 3 | 2 | 2 | doc 63 |
| `vertex_hadron` (+ shape) | 1 | 1 | 1 | doc 94-family guard |

Of the 38 misses decided by the shape tests alone, **16 carry a Michel object the chain found and attached**, all of them under P1's floors (10 MeV and 3 cm): `039252_12/90` (4.1 MeV, 0.9 cm), `039253_12/93` (4.6, 2.6 — the owner's "not fully identified"), `039253_8/64` (14.1, 2.5), `039349_19/52` (5.9, 4.2), `039349_22/56` (13.6, 2.3), `039349_38/57` (5.2, 2.0), `039349_48/21` (8.7, 3.9), `039349_63/55` (3.3, 3.8), `039349_68/63` (6.9, 1.3), `039349_70/61` (5.9, 4.1) and, on the boundary / continuation items, `039252_2/103` (28.9 MeV — kept out by the fiducial rule, correctly), `039349_27/41`, `039349_35/30`, `039349_5/65` (34.7 MeV, `continuation`), `039349_78/22`, `039349_76/23`.

The other 22 shape-only misses have no Michel object. Eleven are STM_MICHEL on the record, and the scan's `michel` tag tells where the Michel went: on `039252_8/102` and `039253_2/13` the tagged segment is a **role-2 delta at an interior vertex** (the Michel is in the chain, classified as a delta); on `039349_44/28` and `039349_81/44` it is **role 1** (fused into the muon chain, doc 74's mechanism, both with a chain of only two segments); on `039349_32/63` one tag is role 3 while the object was still not built (its 4.0 MeV, 5.7 cm reads as `michel_found` 0 — the T2c / T3c vetoes); on the remaining six the tagged segment has **no row at all** — it was never fitted as part of this candidate (`039252_9/101`, `039349_64/52` with five tagged segments, `039349_9/19`, and three items whose record carries no `michel` tag: `039349_51/29`, `039349_71/37`, `039349_81/25`).

## 3. The seven false stoppers

| item | record | anchor shift (cm) | contrast / expected | ks_flat − ks_mu | what holds it in |
|---|---|---:|---|---:|---|
| `039349_14/22` | THRU, owner | 2.80 | 1.22 / 1.96 | +0.019 | the anchor (doc 68's named FP) |
| `039349_24/23` | THRU, owner | 2.28 | 1.20 / 1.95 | −0.004 | the anchor; `shape_flat` misses by 0.016 of the −0.02 margin |
| `039349_30/21` | THRU, owner | 2.67 | 1.19 / 1.96 | −0.000 | the anchor; by 0.02 |
| `039349_76/25` | THRU, owner | 2.49 | 1.42 / 1.96 | −0.008 | the anchor; by 0.012 |
| `039349_38/60` | THRU, medium | 2.80 | 1.58 / 1.96 | −0.005 | the anchor; by 0.015 |
| `039349_13/56` | THRU, medium | 0.00 | 1.76 / 1.69 | +0.088 | a genuine Bragg-like end at the geometric origin, `stop_dis` 3.8 cm |
| `039349_59/14` | THRU, medium | 0.00 | 1.29 / 1.69 | +0.075 | a short profile (plateau window 10–20 cm) |

Five of the seven are the 3 cm anchor's, and four of those pass `shape_flat` by less than the −0.02 margin. Section 9 of the script shows the margin cannot be tightened for them: at −0.01 they stay and ten true stoppers leave (§7.5). None has a Michel object; none is cleared by P1. Three are medium-confidence single-scanner verdicts (§6).

## 4. The 29 missed Michels

| class | n | items |
|---|---:|---|
| (a) no candidate on the arm | 7 | `039252_1/109`, `039253_3/60`, `039253_7/30`, `039253_8/65`, `039349_33/60`, `039349_7/20`, `039349_81/62` — all owner-confirmed; §7.1 and §7.3 |
| (b) the candidate is rejected as a stopper, so the Michel search has no stop | 11 | the eleven STM_MICHEL items of §2's shape-only rows: `039252_8/102`, `039252_9/101`, `039253_2/13`, `039349_32/63`, `039349_44/28`, `039349_51/29`, `039349_64/52`, `039349_71/37`, `039349_81/25`, `039349_81/44`, `039349_9/19` |
| (c) found stopper, Michel search failed | 11 | `039252_2/79` (an 8.9 MeV, 9.3 cm object exists but `michel_found` 0), `039253_0/44` (just recovered as a stopper by doc 75; the owner's "Michel not identified"), `039253_3/61`, `039349_30/45`, `039349_43/66`, `039349_58/69`, `039349_60/40`, `039349_64/24`, `039349_64/65`, `039349_69/56`, `039349_72/11` |

In class (c), three items have the scanner's `michel` tag **inside the muon chain as role 1** — `039253_3/61` (61008), `039349_60/40` (40002), `039349_64/65` (65003) — doc 74's population: the fit runs through the Michel and only the strict tail retreat can separate them, which it does not on these three (`039253_3/61` needs the sub-live reading doc 74 left OFF; the other two are doc 74's "hot" items, not collapses). On the other eight the tagged segment has no row or the record carries no tag (`039252_2/79`, `039349_30/45`, `039349_43/66`, `039349_58/69`, `039349_72/11`): the Michel is either not clustered with the muon or never fitted — the imaging / clustering side, doc 96's territory, not the chain's.

## 5. The 12 spurious Michels

| kind | n | items (record; conn; KE MeV; length cm; kink °) |
|---|---:|---|
| attached arm on a THRU track (record: no Michel) | 6 | `039252_6/114` (1; 3.8; 1.7; 24), `039252_8/82` (1; 6.0; 1.8; 75), `039253_8/31` (1; 7.0; 2.4; 34), `039349_0/68` (1; 5.9; 2.1; 99), `039349_10/27` (1; 2.0; 1.1; 47), `039349_83/23` (1; 1.7; 0.8; 47) — all `is_stm` 0, owner-judged |
| bridged object on an STM_ONLY item | 6 | `039252_12/90` (2; 4.1; 0.9), `039252_16/98` (2; 23.8; 9.2, `is_stm` 1), `039349_20/32` (2; 1.0; 3.8, `is_stm` 1), `039349_22/56` (2; 13.6; 2.3), `039349_62/63` (2; 15.0; 5.5, `is_stm` 1, high confidence), `039349_81/51` (2; 12.9; 4.8, `is_stm` 1) |

The six THRU cases are small (≤ 7 MeV, ≤ 2.4 cm) attached arms at a kink — a delta or a short branch the classifier calls kMichel. The six STM_ONLY cases are bridged (conn 2) objects, four of them 12.9–23.8 MeV on stoppers the record calls STM_ONLY with "detached dots" or "none": those four are as likely a scan question as a reconstruction one (§6). A size gate on the object does not pay (§7.6).

## 6. The record itself

- **601 items, 576 judged; 25 MESSY / UNCLEAR; 6 FRAG_*; 10 low-confidence; 99 owner-judged** (smx3 + smx4).
- **`is_stm` disagreements: 65** — owner 31, medium 27, high 5, low 2. The five high-confidence disagreements are `039349_2/38`, `039349_55/24`, `039252_16/108`, `039252_6/106` (STM_ONLY, `shape_flat` / `plateau_off_mip`) and `039349_26/18`; the owner's 31 are the hard cases of smx3 by construction.
- **29 medium- or low-confidence disagreements** are re-judge candidates before any fix is aimed at them (listed by `d77_review.py` §6): a single scanner at medium confidence called them, and three of the seven false positives (`039349_13/56`, `039349_38/60`, `039349_59/14`) and 26 misses are among them. Doc 68 §3's precedent: three of the anchor's "new FPs" were stoppers on the owner's re-judge.
- **What the record cannot grade:** the 17 unjudged clusters the candidate cap drops (§7.1); the 30 candidates that only exist with a different fit configuration (doc 68 §4's `dx_norm_length` arm; P5 now lets such a study run on the STM side alone, doc 76); and the chain's objects on MESSY events.

## 7. Room to improve, ranked

Each entry is sized on `p75vprod`, offline and exact where the setting has one consumer, with the negatives named. "Scan need" says whether the record can grade it as it stands.

### 7.1 The candidate cap (`max_candidates` 8) — +2 owner-confirmed Michel stoppers, no scan needed

`CheckSTM_Michel` sorts the STM-flagged main clusters by cluster id and keeps the first 8 (`CheckSTM_Michel.cxx:1527-1533`, C++ default 8, not set in the PDVD config). On this arm the cap fires on **6 of 120 events** (9–13 flagged clusters), drops **19 tagger-accepted clusters**, and **two of them are owner-confirmed STM_MICHEL items**: `039253_8/65` and `039349_81/62` — both `T_stm_pass` status 0 (accepted by the tagger), both simply never reconstructed. The other 17 dropped clusters are unjudged (they were never candidates, so never scanned). Raising the cap is a config change (`max_candidates` in `stm_michel_knobs`); the arm must show that the 578 existing candidates are unchanged (the chain claims clusters and pieces per candidate, so order could matter) and what the 17 new candidates read; CPU cost is the only known price. **Recommendation: build first** — it is the only item that recovers owner-confirmed stoppers with no threshold and no scan.

### 7.2 P1's floors — +4 stoppers at 3 MeV / 3 cm, 0 FP on the record

P1 clears the shape bits for a Michel of ≥ 10 MeV and ≥ 3 cm. Exact re-verdict (script §9):

| floors (MeV / cm) | `is_stm` TP / FP / FN | gained | new FP |
|---|---|---|---|
| 10 / 3 (production) | 230 / 7 / 58 | — | — |
| 10 / 2 | 232 / 7 / 56 | `039253_8/64`, `039349_22/56` | — |
| 5 / 3 | 233 / 7 / 55 | `039349_19/52`, `039349_48/21`, `039349_70/61` | — |
| **3 / 3** | **234 / 7 / 54** | + `039349_63/55` | **—** |
| 5 / 2 | 236 / 9 / 52 | + `039253_8/64`, `039349_22/56`, `039349_38/57` | `039253_8/31`, `039349_0/68` (THRU, 2.1–2.4 cm) |
| 0 / 0 | 240 / 13 / 48 | all 16 | 6 THRU |

The length floor is what protects purity: every THRU item with a Michel-like arm is under 2.5 cm (§5). The energy floor does nothing at 3 cm on this record — the four gained at 3 MeV / 3 cm are owner-judged stoppers from the blind smx4 scan (`039349_19/52` 5.9 MeV, `039349_48/21` 8.7, `039349_63/55` 3.3, `039349_70/61` 5.9). Risk: a threshold moved on the record it is graded on; the negatives are 2.0–2.4 cm objects, 0.6 cm under the floor. A knob change (`topology_michel_ke_min` 3), graded like P1 was, with the four named; the owner may prefer to hold the 10 MeV (the T2c / T3c floor) and accept the four as the price.

### 7.3 The STM tagger's own eval — six owner-confirmed stoppers never reach the chain

Of the 12 judged stoppers with no candidate, 2 are the cap (§7.1) and **10 are the tagger's**: `T_stm_pass` status 3 (the tagger's dQ/dx KS eval rejected the pass) on `039252_1/109`, `039253_3/60`, `039253_7/30`, `039349_7/20` (STM_MICHEL, owner) and `039349_50/55`, `039349_57/18` (STM_ONLY, owner); status 5 (proton endpoint) on `039252_8/72`, `039349_75/67`; status 2 (long leftover past the kink) on `039252_9/49`; status 7 (a doc-63 guard) on `039349_33/60`. The same status-3 eval correctly rejects 19 THRU items. `TaggerCheckSTM::eval_stm_core_impl` is the doc-48-era shape test: geometric origin, no peak anchor, no topology — none of docs 65 / 68 / 70 / 75 touched it, because the tagger's verdict gates the whole chain (a candidate that does not exist cannot be re-judged downstream). Room: either a topology-first admission in the tagger (a cluster with a Michel-like arm at the kink is handed on regardless of its KS eval, and `CheckSTM_Michel` decides), or the tagger running the P1 / anchor logic itself. Either is a change to a production component with two other consumers (uBooNE's frozen reference, PDHD) and needs its own default-OFF knob and an arm on both ProtoDUNEs; the six items are the test. **Scan need:** none for the six; the 19 THRU are the negative control.

### 7.4 The plateau window — +1 at `plateau_mip_hi` 2.0, 0 FP

`plateau_off_mip` (window 0.6–1.6 × 55000 e/cm) is the only bit on 11 judged items: 5 stoppers and 6 THRU. Exact sweep (script §9): the **upper** bound at 2.0 admits `039252_6/106` (STM_ONLY, high confidence, plateau 1.79 MIP) and nothing else; the lower bound at 0.5 admits one stopper and one THRU. The four low-plateau stoppers (0.25–0.41 MIP, two with contrasts of 3–8) are tracks whose whole plateau reads a quarter to half of a MIP — a charge-scale question (dead planes, a low-gain region), not a shape one. Room: `plateau_mip_hi` 2.0 (+1, exact, 0 FP); the low side is closed on this record.

### 7.5 Closed on this record

- **`ks_margin`** (production −0.02): exact sweep — −0.01 loses 10 true stoppers and removes no FP; −0.03 adds a THRU FP and no TP. The four anchor-FPs of §3 sit inside 0.02 of the margin together with ten true stoppers.
- **A `michel_found` size gate**: the 12 spurious objects are small, but so are 24 true ones. Requiring ≥ 3 MeV costs 1 TP for 3 FP; ≥ 5 MeV costs 9 for 5; ≥ 2.5 cm costs 17 for 8. Not worth a knob.
- **The boundary / continuation / hadron misses (9)**: five carry a Michel of 4.7–34.7 MeV, but the fiducial rule (doc 44) and the continuation guard are the owner's stated preference over topology; P1 does not clear them by design (doc 70 §3.3).

### 7.6 What needs a re-scan before it can be measured

- The **29 medium / low-confidence disagreements** (§6), three of them false positives.
- The **four STM_ONLY items with a 13–24 MeV bridged object** (§5) — Michel or not.
- The **17 clusters the cap drops** (§7.1), once an arm reconstructs them.
- Any **fit-sampling study** (P5's purpose): a new candidate set, a new scan.

### 7.7 Not the chain's

- **Eight missed Michels whose tagged segment was never fitted with the candidate** (§4 c): clustering / imaging (doc 96).
- **The three Michels fused into the muon chain** (`039253_3/61`, `039349_60/40`, `039349_64/65`): doc 74's residue; a dead-channel-aware sub-live tail reading is the only lead.

## 8. Recommendation

*Follow-up (2026-09-10):* the owner's three observations after this review — the stop overshoot, the near-isolated Michel, unassociated segments — are sized on the same arm in doc 78, which orders its action items behind (1) and (2) below; the cap is doc 79.

In this order, each under the doc 56 bar: **(1)** the candidate cap — a config value, an arm, two named gains, no threshold; **(2)** P1's floors at 3 MeV / 3 cm — a config value, four named gains, the 2.0–2.4 cm THRU arms as the named negatives; **(3)** the tagger's admission — a default-OFF knob in `TaggerCheckSTM`, the six named stoppers as the target and the 19 status-3 THRU items as the control, gated on PDHD and PDVD and checked against the uBooNE reference; **(4)** `plateau_mip_hi` 2.0. Then a re-judge of the 29 medium-confidence disagreements, which is where the remaining error concentrates, before anything is aimed at them.

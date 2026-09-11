# 85 — Doc 78 action item 5: the gamma collect's rejections, and a capture gamma only on a stopper

The owner asked (2026-09-11, after doc 84) to proceed with doc 78's action item 5, which doc 84 §9.6 had recommended next, under the same bar as the rest of the campaign. Item 5 as written has two parts:
- **a census:** for the ~76 same-bundle gamma tags within 35 cm and ≤ 10 cm long on items *with* a Michel object, which P4 gate (doc 71, `michel_gamma_collect`) excluded them — the cone, the per-gamma cap or the body test;
- **a decision:** whether the collect should anchor on the stop when no Michel is found (the other 54 items). Doc 71's purity (0.933) is the bar.

**Status (2026-09-11): FLIPPED in PDVD production — `stop_gamma_require_stm: true`.**
- **The census (§1).** On items with a Michel, every P4 gate removes more delta/other than gamma. There is no gate to loosen.
- **The decision (§2).** The stop-anchored collect already exists: it is doc 51's capture-gamma stage (role 5). Widening it to P4's 50 cm is measured dead on the arm, at 15 good / 38 delta-other on the candidates it would serve.
- **What the census found instead.** The capture stage fires on candidates the verdict rejects. On the owner's record its clusters are **36 gamma / 0 delta-other on `is_stm` 1** candidates and **7 gamma / 33 delta-other on `is_stm` 0** ones, 27 of the latter on through-going muons.
- **The flip.** The new knob withholds the capture gamma from a rejected candidate. Role-5 clusters on judged items go 43 good / 33 delta-other / 1 untagged → **36 / 0 / 0**.
  - `is_stm`, `michel_found` and `reject_bits` are identical on all 596 candidates.
  - Every `is_stm`-1 candidate is bit-identical.
  - Every output that moves, moves only on the 30 events holding a withheld candidate, and only on the withheld objects (§7).
- **Gates.** The OFF path is byte-identical on both detectors. PDHD stays OFF.
- **Two corrections to doc 78 §4.3 (§1.3).** The segment ids of the scan arms drift against production's. At cluster level, item 5's "76" is 55 tags in 53 clusters on 38 items, and only 13 of those clusters, on 9 items, have a Michel. The "54 items without a Michel" are mostly capture events (STM_ONLY).

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
export STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
mkdir -p /home/xqian/tmp/p85
# the census, read-only, on production (doc 84's arm p84vr65 == p84vprod; after the arms, p85voff is the same arm)
cd $IMG && python3 $X/d85_gamma_census.py --arm p85voff --prep /home/xqian/tmp/p85/prep_p85voff > /home/xqian/tmp/p85/census_p85voff.txt
# the build: toolkit 77f4d2d6 + this round's hunks; then the full local/lib pin (572 *.so*)
cd /home/xqian/toolkit-dev/toolkit && ./build/clus/wcdoctest-clus          # 392 cases pass
mkdir -p /home/xqian/tmp/p85/libpin_p85 && cp -L /nfs/data/1/xqian/toolkit-dev/local/lib/*.so* /home/xqian/tmp/p85/libpin_p85/
# the arms (bare production + one key; the smoke arm adds the d53 survey bag), the gates, the flip proofs, the confirmation arm
WAVE=1 nohup bash $X/d85_arms.sh > /home/xqian/tmp/p85/arms_wave1.log 2>&1 < /dev/null & disown
bash $X/d85_gates.sh > /home/xqian/tmp/p85/gates.log 2>&1
cd $IMG/pdvd && PRE_EVT=/home/xqian/tmp/p85/pre_wct-pr-perevt.jsonnet VAL=true bash $X/d85_proofs.sh > /home/xqian/tmp/p85/proofs.txt
JOBS=20 WAVE=2 nohup bash $X/d85_arms.sh > /home/xqian/tmp/p85/arms_wave2.log 2>&1 < /dev/null & disown
# p85vprod == p85vwh: the zip / tree functions of d85_gates.sh, then the branch census
cd $IMG && sed -n '/^zipcmp() {/,/^G=/p' $X/d85_gates.sh | sed '$d' > /home/xqian/tmp/p85/confirm_funcs.sh && source /home/xqian/tmp/p85/confirm_funcs.sh
zipcmp pdvd p85vprod p85vwh; treecmp pdvd p85vprod p85vwh
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p85vwh" --after "pdvd/work/*_p85vprod" --before-arm p85vwh --after-arm p85vprod --pts --out /home/xqian/tmp/p85/g_confirm.txt
```

**Every tag join in this doc is at CLUSTER level.** P4 and the capture stage each decide per companion cluster, and the scan arms' PR segment ids sit one graph index off production's on some clusters (§1.3). A cluster is "gamma" if any of its tags is gamma, else "michel", else its first tag, else "untagged". "Good" means gamma or michel. Purity is good / (good + delta-other). Doc 71's 0.933 was a segment-level number on an earlier state of the record; the unit is stated wherever the two meet.

The predictions were written to `/home/xqian/tmp/p85/pred.txt` at 08:25:15, **before** any arm ran; the arms started at 08:25:22.

## 1. The census: what P4 rejects on items with a Michel

### 1.1 Every companion cluster P4 saw

P4 logs every companion cluster it gates:
- `michel-gamma-cl: … gate N`, where N is 0 pass, 1 radius, 2 length, 3 cone (cos < 0.5), 4 body (d_body ≤ d_mich), 5 blob energy or 6 non-finite;
- `michel-gamma-take: … take T live L` for every cluster that passed.

On production (`p85voff` ≡ `p84vprod`) that is 317 gate lines on the 145 candidates where P4 ran. Here is `d85_gamma_census.py` §A on the owner's STM_MICHEL items, by outcome:

| P4 outcome | gamma / michel | delta / other | untagged |
|---|---:|---:|---:|
| taken (role 4) | **93** | 6 | 2 |
| passed, then not taken (the Michel was vetoed after phase 1) | 2 | 1 | |
| 1 radius (> 50 cm from the final stop) | 2 | 1 | |
| 2 length (> 10 cm) | 0 | 3 | |
| **3 cone** (outside 60° of the Michel direction) | **12** | **29** | 4 |
| **4 body** (closer to the muon body than to the Michel) | **6** | **7** | |
| 5 blob energy (> 20 MeV) | 1 | 0 | |

**Every gate rejects at least as much delta/other as gamma.** The cone's rejects on all judged items, by how far outside they sit:

| cos (Michel direction, stop → blob) | gamma / michel | delta / other | untagged |
|---|---:|---:|---:|
| 0.3 – 0.5 (just outside 60°) | 7 | 9 | |
| 0 – 0.3 | 6 | 22 | 1 |
| < 0 (behind the Michel) | 10 | 63 | 3 |

Even the band just outside the cone is below 50 % gamma, so a wider cone buys gammas at worse than 1:1. The body test is 6 gamma to 7 delta/other, and the energy cap has one gamma. None of these is a loosening that holds doc 71's purity.

### 1.2 Item 5's own population, and which gate excluded it

Item 5 asks about the record's gamma tags in same-bundle clusters that carry no chain role, within 35 cm and ≤ 10 cm long. Re-derived at cluster level (`d85_gamma_census.py` §D), that is **55 tags in 53 clusters on 38 items**. Split by the chain's `michel_found`:
- **With a Michel object: 13 clusters on 9 items.** The C++'s own line for each:
  - cone (gate 3): 7 — `039252_2/39` c134, `039252_9/94` c299, `039253_8/62` c296 and c299, `039349_21/51` c113, `039349_22/56` c160, `039349_31/51` c131;
  - body (gate 4): 4 — `039253_13/39` c122, c123 and c124, `039253_8/62` c295;
  - blob energy (gate 5): 1 — `039349_29/45` c100, 8.4 cm long;
  - no gate line: 1 — `039252_9/52` c173, the Michel's own cluster or one another stage claimed.
- **Without: 40 clusters on 29 items.**
  - 32 of the 40 clusters sit on items the owner calls STM_ONLY, i.e. **capture events**, and 8 on STM_MICHEL items whose Michel the chain misses.
  - P4 never ran on 31 of them: no attached or bridged Michel.
  - On 9, P4 did run (the chain had an attached or bridged Michel at phase 1, vetoed later): the cone rejected 5, and 4 passed the gate but were not taken once the Michel was vetoed.

**The answer to the census half of item 5:** on the 9 items with a Michel, the cone and the body test exclude the owner's gammas, and §1.1 shows both are right to on this record. Loosening either costs purity.

### 1.3 Two corrections to doc 78 §4.3

- **Segment-id drift.** The record's tag keys are the scan arms' segment ids (cluster × 1000 + graph index). On some clusters production's graph index sits one off: tag 125009 is production's 125008. Joined by segment, **23 gamma tags read "no chain role" although their cluster carries one**: role 4 on 14, role 3 on 6, role 5 on 3. Doc 78's "161 gamma tags with no chain role" is inflated by exactly this.
- **The "76".** At cluster level, today, it is 55 tags / 53 clusters / 38 items, of which **13 clusters on 9 items have a Michel** (doc 78 said 76 tags, "on the other 33 items"). The "54 items without a Michel" are **29 items, and 32 of their 40 clusters are on STM_ONLY items**: those gammas are capture gammas, not the gammas of a missed Michel.

## 2. The decision: anchoring on the stop when no Michel is found

### 2.1 The stop-anchored collect already exists

Doc 51's capture-gamma stage (role 5) is the stop-anchored collect. It runs on **every** candidate, Michel or not. Its test:
- a same-bundle companion cluster in the ring michel_dot_radius (15 cm) < d ≤ `stop_gamma_radius_cm` (35 cm);
- ≤ 10 cm long;
- farther from everything the chain has claimed than from the stop (the body test);
- 0.2–20 MeV.

It is isotropic, as a nuclear-capture gamma is. On the 29 no-Michel items it already takes the separable part (§2.3).

**An offline twin of the stage** (`d85_gamma_census.py` §C) uses the image points of each fitted companion. Inside 15–35 cm it accepts **all 84** of production's role-5 clusters, 12 more that production refuses (the energy window, the cap, the Michel's own cluster), and agrees on the 296 it rejects. It is an optimistic bound.

### 2.2 Widening it to P4's radius: dead, on the arm

Item 5's anchor, read literally, is the capture ring out to P4's 50 cm, where P4 never runs. On the arm, `p85vsg50` is bare production plus `stop_gamma_radius_cm: 50`.
- **Admission is unchanged.** It is `max(15, 50)` either way, since P4 already admits to 50. No muon-profile branch moves on any candidate, and `is_stm` and `michel_found` flip 0.
- **What it adds to role 5, against `p85voff`** (`census_p85vsg50.txt` §E):

| candidates | gamma / michel | delta / other | unjudged |
|---|---:|---:|---:|
| P4 never ran, `is_stm` 1 | 11 | 9 | 4 |
| P4 never ran, `is_stm` 0 | 4 | 29 | |
| **P4 never ran (item 5's population)** | **15** | **38** | 4 |
| P4 ran (35–50 cm blobs the capture ring now claims first) | 6 | 3 | |

- **The anchored collect adds gammas at purity 0.28** on the candidates it is meant for. That is 0.55 even on stoppers alone, against a 0.933 bar.
- On P4's own candidates it steals four of P4's good blobs (role 4 → role 5 on `039252_0/82`, `039252_12/90`, `039253_5/32`, `039349_29/45`) and adds 3 delta/other.
- Overall role-5 purity goes 0.566 → 0.464. **Not flipped.**
- **The pre-registered bound was missed by two:** "≤ 16 good / ≥ 40 delta-other" came out 15 / 38. The twin over-accepts delta/other as well as gamma. The purity prediction (< 0.5) held.

### 2.3 What the census found instead: capture gammas on rejected candidates

Role-5 and role-4 clusters on judged items, by the candidate's verdict (`d85_gamma_census.py` §B on production):

| | `is_stm` 1 | `is_stm` 0 | all |
|---|---|---|---|
| role 5 (capture gamma) | **36 gamma / 0 delta-other** (+ 7 on unjudged items), purity 1.000 | **7 gamma / 33 delta-other / 1 untagged**, purity 0.175 | 43 / 33 / 1, 0.566 |
| role 4 (P4 blobs) | 97 / 10 / 2 untagged, 0.907 | 8 / 7, 0.533 | 105 / 17 / 2, 0.861 |

27 of the 33 delta/other capture gammas are on through-going muons. A capture gamma is the claim that the muon **stopped** and was captured, and on a candidate the verdict rejects, nothing stopped. The stage publishes before the verdict exists, so it cannot know this.

**The design that follows:** withhold the capture gamma once the verdict rejects the candidate. On today's record that should take role 5 to 36 / 0 / 0. It costs 7 gammas on stoppers the verdict misses, and those come back by themselves when the verdict is fixed.

P4's role 4 on `is_stm` 0 is a wash (8 good / 7 bad), and those blobs belong to a Michel object that exists whatever the verdict says. It is not gated (§9).

## 3. Design (toolkit `clus/`, default OFF)

- **Knob** `stop_gamma_require_stm` (bool, C++ default false) in `CheckSTM_Michel`: member, `configure`, `default_configuration` with a provenance comment, pinned in `doctest_check_stm_michel_defaults.cxx`.
- **Pure helpers** in `StmMichelFunctions.{h,cxx}`, doctested:
  - `stm_michel_stop_gamma_withhold(require, reject_bits)`: true exactly when the knob is on and a reject bit is set;
  - `stm_michel_rows_keep(roles, drop_role)`: the keep-mask for an order-preserving row erase.
- **The capture stage is unchanged.** With the knob on, before `set_pdg(sg, 11)`, it also saves each accepted segment's particle info and score and its distance to the stop.
- **The withhold step** runs after every reject bit is final: after the topology block, before P4's phase 2. Nothing below it sets a bit, and the coverage test above has already counted the rows. For a rejected candidate with a capture gamma it:
  - erases the role-5 rows from every parallel row array, keeping the order of the rest;
  - restores the segments' particle info and score;
  - drops the gamma showers from the published set (`tf->set_showers` again). `calculate_shower_kinematics` energised every shower on its own, so the Michel's numbers are the knob-off ones;
  - resets `stop_gamma_*` to their defaults and counts the withheld gammas in the new branch `n_stop_gammas_withheld`, persisted only when the knob is on;
  - logs a DEBUG line, `stop-gamma-withheld:`;
  - with the survey on, gives the withheld segments role-6 rows with a new code, **rej 16**, so the scan display still draws them.
- **What it keeps:** the claims, so P4 and the census see the knob-off pool.
- **Why `is_stm` cannot move:** the step runs after the verdict. **Why an accepted candidate cannot move:** the step does not fire on it. The only knob-on difference on such a candidate is the saved-info vector and the new branch at 0.
- **Viewer** (`pdhd/stm_michel_scan/stm_michel_viewer.py`, `REJ_NAMES`): names 16, and the two codes P4 already emitted without a name, 11 (the total-energy guard) and 12 (the Michel vetoed).

## 4. Built

- **Toolkit** `51adc923` (on `77f4d2d6`), five files:
  - `clus/inc/WireCellClus/StmMichelFunctions.h`, `clus/src/StmMichelFunctions.cxx`: the two helpers.
  - `clus/src/CheckSTM_Michel.cxx`: the knob, the saved info, the withhold step, the record field and its persistence.
  - `clus/test/doctest_stm_michel.cxx`: two cases — the predicate's truth table (knob off never withholds; any bit rejects), and the keep-mask (empty, one role dropped in order, a role not present).
  - `clus/test/doctest_check_stm_michel_defaults.cxx`: the default, `false`.
- `wcdoctest-clus`: **392 / 392 pass** (23 846 assertions). The first `./wcb build` hit the new-symbol link trap; one `install -k` and a `wcbuild` cleared it.
- **Freshness:** `libWireCellClus.so` 08:21:56, after the last source edit at 08:20:40.
- **Pin:** the **whole** `local/lib` (572 files) in `/home/xqian/tmp/p85/libpin_p85`, Clus md5 `60fdddac69d0`, unchanged before and after every arm (`md5sum -c`).
- **TLA proof** (`wcsonnet`, before launch): `stop_gamma_require_stm:true` and `stop_gamma_radius_cm:50.0` each add exactly their key to the compiled `CheckSTM_Michel` block. `michel_gamma_radius_cm` compiles to 50, so the 50 cm ring leaves admission where it is.
- **Viewer selftest** (`selftest_stm_michel_scan.py --det pdvd --quick`): 7303 checks pass and 2 fail. The two are a data check, the smx3 / smx4 label files against the tranche-1 sheet, which has not changed since 2026-09-08. They do not read `REJ_NAMES`.

## 5. Criteria and predictions

Arms (`d85_arms.sh`, a fork of `d84_arms.sh`): JOBS 5 × 5 arms, detached, on the pin. Another user's PDHD jobs held the box at load ~22.

| arm | det | key | purpose |
|---|---|---|---|
| `p85voff` | pdvd | — | OFF gate vs `p84vprod` |
| `p85hoff` | pdhd | — | OFF gate vs `p84hoff` |
| `p85vwh` | pdvd | `stop_gamma_require_stm: true` | **flip candidate** |
| `p85vsg50` | pdvd | `stop_gamma_radius_cm: 50.0` | item 5's stop anchor, config only |
| `p85vwhs` | pdvd | the knob + the d53 survey bag | smoke: rej-16 rows |

Predictions (`pred.txt`, with the named list `pred_withheld_list.txt`):
- **OFF gates.**
  - `p85voff` ≡ `p84vprod`: 596 × all 144 branches and every point row, 120 zips, 119 calib, 8 trees.
  - `p85hoff` ≡ `p84hoff`: 325 × 130, 61 zips, 8 trees.
- **`p85vwh`.**
  - **Verdicts:** `is_stm` / `michel_found` / `reject_bits` identical on 596. Every `is_stm`-1 candidate is bit-identical.
  - **Movers:** exactly the `is_stm`-0 candidates with a capture gamma, the 32 named, plus any candidate without a payload, named. On them only the `stop_gamma_*` fields and role-5 rows move, and every other row stays identical in value and order.
  - **Role-5 purity:** 43 / 33 / 1 → 36 / 0 / 0. The 7 withheld gammas are named.
  - **Zips and calib:** they move only on events holding a withheld candidate.
  - **Census:** unchanged (merged 141 / 12 / 19; `is_stm` 233 / 7 / 45); `--check` 0 / 14.
- **`p85vsg50`.**
  - Admission is unchanged: the muon-profile branches and every verdict are identical.
  - No role-5 cluster is lost.
  - On no-P4 candidates it adds ≤ 16 good / ≥ 40 delta-other.
  - On P4 candidates, blobs migrate role 4 → 5.
  - Dead; not flipped.
- **`p85vwhs`:** every withheld segment carries a role-6 rej-16 row, and no rejected candidate keeps a role-5 row.

**Flip bar:**
1. both OFF gates PASS;
2. the verdicts are identical, and every `is_stm`-1 candidate is bit-identical;
3. role-5 cluster purity on judged items ≥ 0.933, with every withheld cluster named;
4. `--check` 0 / 14;
5. the doctests pass.

## 6. Gates

`d85_gates.sh`, a fork of `d84_gates.sh` (`/home/xqian/tmp/p85/gates.log`; branch censuses `g_<arm>.txt`; the first pass, whose smoke section crashed on `039252_11`'s missing tree, is `gates_pass1.log`):
- every Bee zip member by hash, the calib json, every `tracking-pr.root` tree, and every `T_stm_michel` branch and point row;
- section 3 names every candidate on which any production branch or point-row set moved, and checks the non-role-5 rows for value **and order**;
- section 3b compares the ON arms' zips, calib and trees;
- section 4 preps and scores each arm on both records;
- section 5 runs the census per arm;
- section 6 is the smoke.

## 7. Results

All five wave-1 arms ran on the pin (Clus `60fdddac69d0` before and after; the whole pin unchanged). PDVD had 120 / 120 event dirs and 119 complete: `039252_11` has no STM candidate on every arm, as always. PDHD had 61 / 61. There were 0 loader deaths.

### 7.1 OFF gates: PASS on both detectors

| gate | zips (member hash) | calib json | trees | `T_stm_michel` branches | point rows |
|---|---|---|---|---|---|
| `p85voff` ↔ `p84vprod` (PDVD) | 120 / 120 | 119 same, 0 diff | 8 / 8 on every event | 596 / 596 × 144 | 596 / 596, 0 role moves |
| `p85hoff` ↔ `p84hoff` (PDHD) | 61 / 61 | 61 same, 0 diff | 8 / 8 | 325 / 325 × 130 | 325 / 325, 0 role moves |

### 7.2 `p85vwh`, by name: every prediction held

**Movers: 36 candidates, all `is_stm` 0 on production, and 45 capture gammas withheld.** They are the 32 named in `pred_withheld_list.txt` plus 4 candidates with no record entry: `039349_0/63`, `039349_21/27`, `039349_43/21` and `039349_64/73`.

**What moved, on them:**
- **Branches:** only the seven `stop_gamma_*` fields, on all 36.
- **Rows:** only the role-5 rows. The other rows are identical in value and order on all 596 candidates.
- **The new branch:** `n_stop_gammas_withheld` = the old `n_stop_gammas`.
- **Verdicts:** `is_stm` flips 0, `michel_found` flips 0, `reject_bits` unchanged.
- **Everyone else:** 560 / 596 candidates are bit-identical on all 144 shared branches.

**Role 5 on judged items**, cluster level (`census_p85vwh.txt`):

| arm | `is_stm` 1 | `is_stm` 0 | all judged: good / delta-other / untagged | purity |
|---|---|---|---|---:|
| `p85voff` (production) | 36 / 0 (+7 unjudged) | 7 / 33 / 1 | 43 / 33 / 1 | 0.566 |
| **`p85vwh`** | 36 / 0 (+7 unjudged) | — | **36 / 0 / 0** | **1.000** |

**Withheld:**
- 33 delta/other clusters: 29 on 21 through-going candidates (27 on 19 THRU, 2 on 2 FRAG_THRU), and one each on `039252_12/90` and `039349_51/21` c86 (STM_ONLY) and on `039349_37/33` and `039349_77/52` (MESSY). The per-candidate list is in `gates.log` §3.
- 1 untagged cluster: `039253_12/93` c315.
- 7 gammas, on stoppers the verdict misses: `039252_12/123` c386, `039253_8/27` c121, `039349_48/63` c226 and `039349_51/21` c85 (STM_ONLY); `039349_51/29` c107 and `039349_68/63` c209 (STM_MICHEL); and `039349_54/56` c147 (MESSY).

Role 4 does not move: P4 is untouched.

**The outputs that move, and only where they should.** `d85_gates.sh` §3b, traced item by item:
- **`mabc-pr.zip`** differs on **30 events: exactly the 30 events holding a withheld candidate**.
  - `mc.json` (the PF tree) loses exactly the 45 withheld `gamma` pseudo-nodes and their `e-` leaves. The renderer's synthetic node ids after them renumber down by one.
  - `shower_track-global.json` changes on 22 of the 30 events: the withheld segments' track/shower flag.
- **The calib json** differs on the same 30 events: the candidate's `showers` list loses the withheld gamma, and those segments' `particle_id` / `shower_id` / `particle_score` and point `flag_shower` go back to their pre-`set_pdg` values.
- **`T_rec_charge`** differs on 22 events, only in `particle_id`, and only on the 60 rows of withheld capture segments: 11 → 4 on 58 rows and 11 → 13 on 2. That is the PR's own classification, restored.
- **`T_stm_michel_pts`** differs on the 30 events (the role-5 rows). **`T_stm_michel`** differs on 119 events, because the new branch exists. `T_bad_ch`, `T_cluster`, `T_proj`, `T_proj_data` and `Trun` are identical everywhere.

**The census is unchanged on both records** (all four arms print identical score lines): merged record `is_stm` 233 / 7 / 45, `michel_found` 141 / 12 / 19; `census_score.py --check` 0 / 14.

**Smoke `p85vwhs`** (knob + survey bag): 45 withheld gammas, rej-16 role-6 rows on 36 candidates (152 rows), and 0 role-5 rows on any `is_stm`-0 candidate. This is the only configuration in which the erase also squeezes the `rej` / `d_stop` / `d_body` columns, and the columns stay aligned with `role` on every row of all 120 events:
- roles 1–5 and 7 carry only rej 0;
- role 6 carries only the survey's codes (2, 4, 5, 6, 8, 9, 11, 12, 16);
- role 8 carries only the census's codes (2, 3, 4, 13);
- the 152 rej-16 rows sit 16.0–35.1 cm from the stop (the capture ring), none of them in the candidate's own cluster.

### 7.3 `p85vsg50`: §2.2

61 candidates move.
- **Where:** 27 of them are `is_stm` 1.
- **Which branches:** only `stop_gamma_*` and, on the four migration candidates, the five P4 branches.
- **What does not move:** the muon profile and the verdicts, with `is_stm` and `michel_found` flips 0 and the census identical.
- **Outputs:** zips differ on 47 events.

Graded in §2.2: dead.

## 8. Flip

One key goes in `pdvd/wct-pr-perevt.jsonnet` `stm_michel_knobs`, directly after `michel_gamma_radius_cm: 50.0`: `stop_gamma_require_stm: true`. Its comment carries the C++ default, the physics argument, the 36 / 0 vs 7 / 33 split and the arm result. The file was edited with no PDVD job of this tree in flight.

`d85_proofs.sh` (`/home/xqian/tmp/p85/proofs.txt`):
- **A:** PRE + `-S stm_michel_extra={stop_gamma_require_stm:true}` (the arm's config) vs POST: **0 lines**.
- **B:** POST with the key forced to `false` vs PRE: exactly the one key, present `false` vs absent.
- **C:** PRE vs POST: exactly `"stop_gamma_require_stm": true`.
- **D:** PDHD: 0 lines of the key in `pdhd/wct-pr-perevt.jsonnet`.

**Confirmation arm `p85vprod`** (the flipped file, no TLA, same pin; `/home/xqian/tmp/p85/gates_confirm.log`): against `p85vwh`, **120 / 120 zips identical, calib 119 / 119, all 8 trees identical on every event, 596 / 596 candidates bit-identical on all 145 branches, and point geometry and role labels identical on 596 / 596**. The arm was 119 / 120 complete (`039252_11`, as always), and the pin was unchanged before and after.

PDVD production is now `p85vprod`. Its prep for the next round is `/home/xqian/tmp/p85/prep_p85vwh`, from the identical arm.

## 9. Observations, and what this leaves

1. **P4's gates are right on this record.** On items with a Michel the cone rejects 12 gammas for 29 delta/other and the body test 6 for 7, and even the band just outside the cone is 7 : 9. Item 5's census half needs no knob.
2. **The "no Michel" population is capture.** 32 of the 40 uncollected gamma clusters on items without a Michel are on STM_ONLY items. The chain already anchors a collect on the stop for them, the capture stage, and on stoppers it is pure (36 / 0). What it leaves behind, it leaves at 15 : 38 (§2.2). The body test is what separates, and it is doing so.
3. **P4's role-4 purity at cluster level on today's record reads 0.861**: 0.907 on `is_stm` 1 and 0.533 on `is_stm` 0. Doc 71's 0.933 was segment-level, on the record as it stood then (19 of its role-4 segments sat on items the record did not yet judge). A verdict gate for role 4 would withhold 8 good for 7 bad. It is not built: those blobs are members of a Michel object that exists regardless of the verdict, and whether a rejected candidate should carry any Michel daughters in the PF tree is the owner's call.
4. **The survey mislabels P4's rejects on the display.** With the survey on, a companion P4 examined and rejected, say by the cone at 40 cm, gets the capture stage's code on its role-6 row, e.g. 9, "survey only — neither stage was offered it". The C++ log has P4's own gate; the row does not. Carrying P4's code on the row would change scan-arm output, so it needs its own knob. It is a next step if the owner wants the display to show P4's reason.
5. **Segment-id joins across arms are unsafe.** The scan record's segment ids come from the arm the sheet was built on. A later arm can renumber a companion's segments by one, and a segment-level join then reads a collected cluster as "no role". Every join here is at cluster level. Earlier segment-level counts (doc 78 §4.3, and any tag-to-row join on a later arm) carry this error.
6. **What the scanner sees changes on rejected candidates.** On scan arms (survey on) a withheld capture gamma keeps its points, as a role-6 rej-16 row, but the PF panel no longer has a gamma node for that candidate. The 7 withheld gammas sit on stoppers the verdict misses: 5 STM_ONLY, 2 STM_MICHEL. That is the `is_stm` false-negative population doc 78 items 2, 4 and 6 work to recover, and the cost of this flip lands there. A scanner re-judging such a candidate sees the blob, not the gamma; when the verdict is fixed, the gamma comes back without any further change.
7. **One companion can be claimed by two candidates.** `039349_37/33` (MESSY) and `039349_37/52` (THRU) both took cluster 205 as a capture gamma. Both are rejected, and both are now withheld. It is noted, not addressed.
8. **Next: doc 78 item 6**, the hand-check of `039349_43/66`, `039349_72/11`, `039253_0/44` and `039349_30/45`, which have no fitted charge near the stop (doc 96's territory; no code). After it, doc 78's list is closed.

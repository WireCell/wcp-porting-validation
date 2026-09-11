# 84 — Doc 78 action item 3: T2c's margin, and a reach exemption for the moved-stop veto

The owner asked (2026-09-11, after doc 83) to proceed with doc 78's action item 3, which doc 83 §9.6 had recommended next. Item 3 as written: re-grade T2c (`moved_stop_michel_guard`, doc 61) on the current record. `039252_2/79` is owner-confirmed STM_MICHEL, and its 8.9 MeV / 9.3 cm Michel is demoted at a 59.58° kink against the 60° exemption of doc 72. "A value under 59.5° recovers the item — a one-item margin, to be declared as such."

**Status (2026-09-11): FLIPPED in PDVD production.** The knob is `moved_stop_michel_reach_min_cm` in `CheckSTM_Michel` (C++ default -1 = off), and PDVD production runs it at `6.5`. The moved-stop veto (T2c) now also spares an attached Michel whose arm plus far subtree reaches ≥ 6.5 cm. On production it recovers exactly one Michel, `039252_2/79`, the owner-confirmed item of doc 78 item 3:
- On the owner's merged record (smx1a + smx3 + smx4), `michel_found` goes 140 / 12 / 20 → **141 / 12 / 19** (TP / FP / FN), F1 0.897 → 0.901.
- `is_stm` is unchanged, and so is every other candidate on every production branch, every point row, every Bee zip and every calib dump.
- The frozen doc 55 record (smx1a), which the scorer reads by default, still calls this item THRU, so there it reads as one more FP (§7.3). That is the verdict the owner's smx3 scan overturned, and the reason item 3 asked for the re-grade.

The OFF path is byte-identical on both detectors. PDHD stays off (T2c itself is off there). Item 3's own value, a 59.1° kink exemption, gives the identical result and is **not** flipped: it sits in a 0.9° window beside an item whose kink moves 10.5° with the survey bag, a configuration that never touches T2c (§2).

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
mkdir -p /home/xqian/tmp/p84
# the re-grade and the cross-arm census, read-only
python3 $X/d84_t2c_census.py > /home/xqian/tmp/p84/t2c_census.txt
# the build: toolkit 5904f1d2 + this round's hunks; then the full local/lib pin
cd /home/xqian/toolkit-dev/toolkit && ./build/clus/wcdoctest-clus          # 390 cases pass
mkdir -p /home/xqian/tmp/p84/libpin_p84 && cp -L /nfs/data/1/xqian/toolkit-dev/local/lib/*.so* /home/xqian/tmp/p84/libpin_p84/
# the arms (bare production + one key), the gates, the flip proofs, the confirmation arm
WAVE=1 nohup bash $X/d84_arms.sh > /home/xqian/tmp/p84/arms_wave1.log 2>&1 < /dev/null & disown
bash $X/d84_gates.sh > /home/xqian/tmp/p84/gates.log 2>&1
PRE_EVT=/home/xqian/tmp/p84/pre_wct-pr-perevt.jsonnet VAL=6.5 bash $X/d84_proofs.sh > /home/xqian/tmp/p84/proofs.txt
WAVE=2 nohup bash $X/d84_arms.sh > /home/xqian/tmp/p84/arms_wave2.log 2>&1 < /dev/null & disown
# p84vprod == p84vr65: the zip / tree / branch comparisons of d84_gates.sh with PROD=p84vr65
```

`d84_gates.sh` scores every arm twice: on `census_score.py`'s default record (the frozen smx1a scan, which keeps `--check` meaningful) and with `STM_SCAN_RECORD` set to the merged record, `pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json` (§7.3). The merged-record lines were added to the script after the first gates pass, whose scores were taken by hand with the same command (`score_merged_<arm>.txt`).

The predictions were written to `/home/xqian/tmp/p84/pred.txt` at 07:03:33, **before** any arm ran; the first arm started at 07:03:39.

## 1. The re-grade on today's record

On production `p83vprod` (596 candidates), T2c fires on four candidates (`d84_t2c_census.py` §1):

| item | record | T2c | kink | KE | len + far | moved by |
|---|---|---|---:|---:|---:|---|
| `039252_2/79` | STM_MICHEL (owner) | **veto** | 59.58° | 8.92 MeV | 9.3 + 0 cm | split |
| `039252_4/55` | THRU | veto | 17.19° | 4.62 MeV | 5.1 + 0 cm | split |
| `039349_61/62` | THRU | veto | 58.68° | 5.99 MeV | 5.4 + 0 cm | split |
| `039349_48/21` | STM_MICHEL | kink-exempt (doc 72) | 132.64° | 8.67 MeV | 3.9 + 8.0 cm | 2 retreats |

**So T2c today removes 2 THRU false positives for 1 owner-confirmed true positive lost — not the 4 for 1 that item 3 anticipated.** Doc 61 built the veto against five "through-going" items on the smx1a record. Two of those five, `039252_2/79` and `039349_48/21`, are STM_MICHEL on the merged record (doc 68's owner scan). `039349_20/41` (THRU) no longer fires on production, and neither does doc 61's own lost true positive, `039349_36/63`: production has moved their stops or Michels since doc 61.

## 2. Every T2c instance on the record, and what separates them

`d84_t2c_census.py` §2 reads every T2c veto or exemption on 28 PDVD arms (d67v through p83vprod: bare production, trial arms, and the arms that carried the d53 survey bag — d67v, d68a3, d68d4 and doc 82's void wave p82v*). There are **12 distinct instances on 7 items**, every one judged THRU or STM_MICHEL:

| item | record | kink | KE (MeV) | reach = len + far (cm) | arms |
|---|---|---:|---:|---:|---|
| `039252_2/79` | STM_MICHEL | 59.58 | 8.92 | 9.3 | 22 (d67v, and production since d68a3) |
| `039252_2/79` | STM_MICHEL | 89.34 | 7.16 / 7.44 | 7.5 | d68d4 (survey + TrackFitting dx 0.4); the four `stop_tail_peak` arms p82btp, p82btp0, p82vtp, p82vtp0 |
| `039349_48/21` | STM_MICHEL | 132.64 | 8.67 | 11.9 | 25 |
| `039252_4/55` | THRU | 17.19 | 4.62 | 5.1 | 26 |
| `039349_20/41` | THRU | 43.47 | 5.73 | 4.9 | d67v, d68a3 |
| `039349_61/62` | THRU | 48.21 | 7.97 / 8.02 | 5.8 | the 8 survey-bag arms (d67v, d68a3, d68d4, p82vcs, p82vk10, p82voff, p82vtp, p82vtp0) |
| `039349_61/62` | THRU | 58.68 | 5.99 | 5.4 | 18, every bare arm from p79vcap to p83vprod |
| `039349_72/54` | THRU | 33.93 | 6.77 | 5.6 | the four `stop_tail_peak` arms |
| `039349_75/73` | THRU | 23.89 | 7.20 | 5.4 | the four `stop_tail_peak` arms |

(`039252_2/79` at 59.58° appears twice in the script's output, at `is_stm` 0 on d67v and 1 since; the table merges them.)

| separator | THRU | STM_MICHEL | verdict |
|---|---|---|---|
| kink | 17.19 – 58.68° | 59.58 – 132.64° | separates, but only in **(58.68°, 59.58°]**, a 0.9° window |
| KE | 4.62 – 8.02 MeV | 7.16 – 8.92 MeV | **does not separate** |
| reach (len + far_len) | 4.9 – 5.8 cm | 7.5 – 11.9 cm | separates in **(5.8, 7.5] cm** |

Three readings follow.

- **The kink window is real but is not a margin.** `039349_61/62`'s kink reads 58.68° on every bare arm and 48.21° on every arm that carried the survey bag, whose wider companion admission changes its fit through `preload_clusters` (doc 72 §2 found the same jump; doc 53 §6.2 the mechanism). Neither configuration touches T2c. A cut at 59.1° would sit 0.4° above an item whose kink has already moved about 25 times that far.
- **KE, doc 72 §9's first option, is dead.** On the survey-bag arms `039349_61/62` reads 7.97–8.02 MeV, above `039252_2/79`'s 7.16–7.44 on the stop-mover arms. On bare production alone a 7 MeV floor separates (5.99 vs 8.67); across the configurations on record it does not.
- **Reach, doc 72 §9's second option, separates every instance.** Every through-going arm is 4.9–5.8 cm; every owner Michel is ≥ 7.5 cm. Doc 72 floated 8 cm on its five points; the 7.5 cm variant of `039252_2/79` (the stop-mover arms and d68d4) puts 8 cm too close, and the midpoint of the window is 6.65 cm.

**The owner chose the reach exemption (this session), at 6.5 cm, with the kink value armed alongside and reported.** It is a length stand-in for the owner's own discriminator, the turn (doc 72 §1): it is chosen because on this record it is the separator that has not moved across its window.

## 3. Design (toolkit `clus/`)

- **A pure predicate** in `StmMichelFunctions.{h,cxx}`, `stm_michel_moved_stop_spare(kink_deg, reach_cm, kink_min_deg, reach_min_cm)` → `kVeto | kKink | kReach`.
  - The kink test is doc 72's expression verbatim and runs **first**, so its precedence and its counter `n_michel_veto_exempt` are unchanged; `039349_48/21` stays a kink exemption.
  - Then the reach test: `reach_min_cm >= 0 && reach_cm >= reach_min_cm`.
  - A threshold < 0 is off. An unmeasurable kink (-1) never spares by the kink. A NaN fails both comparisons and is vetoed.
- **The T2c site** in `CheckSTM_Michel.cxx` switches on the predicate with reach = `(michel_len + michel_far_len) / cm`. Both conn-1 paths (the attached path and doc 83's near-arm path) set `michel_len` and `michel_far_len` from the seed arm.
- **Knob** `moved_stop_michel_reach_min_cm` (C++ default -1 = off), with `configure` and `default_configuration`. **New branch** `n_michel_veto_reach_exempt`, written only when the knob is ≥ 0.
- **Why `is_stm` cannot move** (the doc 72 argument, unchanged): the veto writes only `michel_conn_type`. A spared Michel is under `moved_stop_michel_ke_min` (10 MeV), and `topology_stop_evidence` needs ≥ `topology_michel_ke_min` (10 MeV).
- **What else can move on a spared item.** The P4 gamma collect runs after T2c and reads `michel_found && conn ∈ {1, 2}`. A spared Michel may therefore collect its reserved gamma blobs: the gamma branches and role-4 rows may move on the spared candidate, and nowhere else.
- **A hazard named in the code.** The far subtree is the stop-arm classifier's walk, which can re-enter the muon chain through a side loop and read the muon as the arm's reach (doc 83 §9.4, `039349_32/63`, 206 cm). No T2c instance on this record has a loop-back.

## 4. Built

- **Toolkit** `77f4d2d6` (on `5904f1d2`), five files:
  - `clus/inc/WireCellClus/StmMichelFunctions.h`, `clus/src/StmMichelFunctions.cxx`: the enum and the predicate.
  - `clus/src/CheckSTM_Michel.cxx`: the knob, the switch at the T2c site, the record field and its persistence, the comments.
  - `clus/test/doctest_stm_michel.cxx`: three cases — precedence (kink first, then reach; either off), boundaries and non-finite inputs, and the record's 12 instances replayed at (60°, 6.5 cm), which spare exactly the STM_MICHEL ones.
  - `clus/test/doctest_check_stm_michel_defaults.cxx`: the default round-trip, `-1`.
- `wcdoctest-clus`: **390 / 390 pass** (23836 assertions). The first `wcbuild` hit the new-symbol link trap (the test linked against the installed lib); one install and a rebuild cleared it.
- Freshness: `libWireCellClus.so` 07:00:53, after the last source edit at 06:59:34. The **whole** `local/lib` (572 files) is pinned in `/home/xqian/tmp/p84/libpin_p84` (Clus md5 `955859a5…`).
- A `wcsonnet` proof before launch: the TLA `moved_stop_michel_reach_min_cm:6.5` adds exactly that key to the compiled config; `moved_stop_michel_kink_min:59.1` changes exactly 60 → 59.1.

## 5. Criteria and predictions

Arms (`d84_arms.sh`, a fork of `d83_arms.sh`): bare production plus one key, JOBS 5 × 6 arms, detached.

| arm | det | key | purpose |
|---|---|---|---|
| `p84voff` | pdvd | — | OFF gate vs `p83vprod` |
| `p84hoff` | pdhd | — | OFF gate vs `p83hoff` (T2c is off on PDHD) |
| `p84vr65` | pdvd | `moved_stop_michel_reach_min_cm: 6.5` | **flip candidate** |
| `p84vr8` | pdvd | `…: 8.0` | doc 72 §9's value |
| `p84vr5` | pdvd | `…: 5.0` | negative control: spares both THRU arms (5.1, 5.4 cm) — on production, T2c with nothing left to veto |
| `p84vk59` | pdvd | `moved_stop_michel_kink_min: 59.1` | item 3's own value, config only; reported, not flipped |

Predictions, verbatim from `pred.txt`:
- **`p84voff` ≡ `p83vprod`** on every branch and point row, 120/120 zips, 8/8 trees, calib; **`p84hoff` ≡ `p83hoff`** (325 × 130, 61/61, 8/8).
- **`p84vr65`: exactly one candidate moves, `039252_2/79`:** `michel_conn_type` 0→1, `michel_found` 0→1, `n_michel_veto` 1→0, `n_michel_veto_reach_exempt` 0→1; `is_stm` stays 1, `reject_bits` 0. The gamma branches and role-4 rows may move on it only. `039349_48/21` stays kink-exempt. Census: michel TP 136→137, FN 16→15, FP 18 (F1 0.889→0.893); `is_stm` 226/13/42 unchanged.
- **`p84vr8`** identical to `p84vr65`.
- **`p84vr5`**: `p84vr65` plus `039252_4/55` and `039349_61/62` michel_found 0→1, michel FP 18→20.
- **`p84vk59`**: the same change on `039252_2/79`, counted in `n_michel_veto_exempt` instead.
- **Pre-registered choice:** flip reach 6.5 if the bar holds. Kink 59.1 is reported, never flipped. Reach 8 is not flipped even if identical (0.5 cm from the 7.5 cm variant).

**Flip bar:** `039252_2/79` recovered; nothing else moves bar the declared gamma possibility on it; 0 new michel FP, 0 new `is_stm` FP, 0 TP lost; both OFF gates PASS; `census_score.py --check` 0/14.

## 6. Gates

`d84_gates.sh` (a fork of `d83_gates.sh`): every Bee zip member by hash, calib json, every `tracking-pr.root` tree, every `T_stm_michel` branch and point row (`d51g_branch_census.py --pts`). Section 3 names every candidate on which any production branch or point-row set moved. Section 4 preps each arm and scores it against the record, with a fresh `prep_p83vprod` under `/home/xqian/tmp/p84/`.

## 7. Results

All six wave-1 arms ran on the pinned lib (Clus md5 `955859a5…` before and after, the whole pin unchanged by `md5sum -c`). PDVD 120/120 event dirs each, 119 complete (`039252_11` is incomplete on every arm, production included, as in doc 83), PDHD 61/61; 0 loader deaths. Gates in `/home/xqian/tmp/p84/gates.log`; branch censuses `g_<arm>.txt`; scores `score_<arm>.json` (smx1a) and `score_merged_<arm>.json` (merged record).

### 7.1 OFF gates — PASS on both detectors

| gate | zips (member hash) | calib json | trees | `T_stm_michel` branches | point rows |
|---|---|---|---|---|---|
| `p84voff` ↔ `p83vprod` (PDVD) | 120 / 120 | 119 same, 0 diff | 8 / 8 on every event | 596 / 596 × 143 | 596 / 596, 0 role moves |
| `p84hoff` ↔ `p83hoff` (PDHD) | 61 / 61 | 61 same, 0 diff | 8 / 8 | 325 / 325 × 130 | 325 / 325, 0 role moves |

### 7.2 The ON arms, by name — every prediction held item for item

| arm | candidates moved | what moved | T2c vetoes / kink-exempt / reach-exempt |
|---|---|---|---|
| production `p83vprod` | — | — | 3 / 1 / 0 |
| `p84vr65` | **1**: `039252_2/79` | `michel_conn_type` 0→1, `michel_found` 0→1, `n_michel_veto` 1→0 (+ `n_michel_veto_reach_exempt` 1) | 2 / 1 / 1 |
| `p84vr8` | 1: the same | the same | 2 / 1 / 1 |
| `p84vr5` | 3: + `039252_4/55` (THRU, 5.1 cm), `039349_61/62` (THRU, 5.4 cm) | the same three branches on each | 0 / 1 / 3 |
| `p84vk59` | 1: `039252_2/79` | `michel_conn_type`, `michel_found`, `n_michel_veto`, `n_michel_veto_exempt` 0→1 | 2 / 2 / 0 |

- On every ON arm: `is_stm` flips 0, `reject_bits` unchanged, 0 point-row role moves, point geometry identical on 596 / 596, and **all 120 Bee zips identical** (`michel_found` does not feed the zip). The other seven trees are identical on every event.
- **The declared gamma possibility did not happen:** `039252_2/79` has no reserved gamma blob (`n_michel_gammas` 0), so no gamma branch or role-4 row moved.
- `039349_48/21` stays a kink exemption on every arm: the kink test runs first, as designed.
- `p84vr5` is production with nothing left for T2c to veto. It is the measured "T2c off" on today's record: +2 THRU Michels.

### 7.3 The census, on two records

| arm | merged record (owner): michel TP / FP / FN, F1 | frozen smx1a: michel TP / FP / FN, F1 | `is_stm` (merged / smx1a) |
|---|---|---|---|
| `p83vprod` | 140 / 12 / 20, 0.897 | 136 / 18 / 16, 0.889 | 233/7/45 · 226/13/42 |
| `p84vr65`, `p84vr8`, `p84vk59` | **141 / 12 / 19, 0.901** | 136 / 19 / 16, 0.886 | unchanged |
| `p84vr5` | 141 / 14 / 19, 0.895 | 136 / 21 / 16, 0.880 | unchanged |

(Merged: 546 judged of 570 payloads matched to 601 records; smx1a: 547 judged. `census_score.py --check` 0 / 14 differ.)

**A prediction slip, recorded.** `pred.txt` quoted the smx1a scorer's baseline (136 / 18 / 16) and predicted TP +1 on it. The scorer reads `pdvd_stm_michel_smx1a_verdicts.json` by default, doc 55's frozen record, where `039252_2/79` (and `039349_48/21`) are THRU. The owner re-judged both STM_MICHEL in smx3 (2026-09-10, "the current identified end point is OK"). On that record the gain is the predicted TP +1; on the frozen one it is FP +1. The item-level prediction — one candidate, these three branches — held exactly; only the table the headline was quoted from was the wrong one. The merged scores come from the same script with `STM_SCAN_RECORD` pointing at the merged verdicts (§0).

### 7.4 Kink 59.1°, reported

`p84vk59` is bit-identical to `p84vr65` in every scored quantity and moves the same three branches, counting the spare in the kink counter instead. It is not flipped, by the pre-registered choice: its window is (58.68°, 59.58°], and `039349_61/62`'s kink is the value that moves 10.5° with the survey bag. Reach 8 cm is also identical and also not flipped (0.5 cm from `039252_2/79`'s 7.5 cm variant).

## 8. Flip

One key in `pdvd/wct-pr-perevt.jsonnet` `stm_michel_knobs`, directly after `moved_stop_michel_kink_min: 60.0`: `moved_stop_michel_reach_min_cm: 6.5`, with its comment (the C++ default, the census, why the kink stays at 60, why KE is dead, the arm result). `d84_proofs.sh` (`/home/xqian/tmp/p84/proofs.txt`):
- **A** PRE + `-S stm_michel_extra={moved_stop_michel_reach_min_cm:6.5}` (the arm's config) vs POST: **0 lines**.
- **B** POST with the key forced to -1 vs PRE: exactly the one key, present at -1 vs absent.
- **C** PRE vs POST: exactly `"moved_stop_michel_reach_min_cm": 6.5`.
- **D** PDHD: 0 lines of the key in `pdhd/wct-pr-perevt.jsonnet`.

**Confirmation arm `p84vprod`** (the flipped file, no TLA, same pin; `/home/xqian/tmp/p84/gates_confirm.log`): against `p84vr65`, **120 / 120 zips identical, calib 119 / 119, 8 / 8 trees identical on every event, 596 / 596 candidates bit-identical on all 144 branches, point geometry and role labels identical on 596 / 596**. Production now is `p84vprod`; its prep for the next round is `/home/xqian/tmp/p84/prep_p84vr65` (identical arm).

## 9. Observations, and what this leaves

1. **T2c's value on today's record is two THRU Michels.** With the reach exemption it vetoes `039252_4/55` and `039349_61/62` and nothing the owner calls a Michel. `p84vr5` measures its absence: michel FP +2 on either record. Doc 61's other targets have left its population as production moved their stops.
2. **The reach is a length stand-in for the turn,** the owner's discriminator. It was chosen because it is the one separator whose window no instance has crossed on 28 arms. It has not been tested beyond these 12 instances, and a moved-stop Michel that is short and soft *and* barely turns would still be vetoed (none on this record).
3. **The loop-back hazard** (doc 83 §9.4): the far subtree can read the muon itself through a side loop (206 cm on `039349_32/63`). A T2c item with such a walk would be spared on reach. None exists on this record; the hazard is named in the code comment.
4. **The upper margin is `039252_2/79`'s own other reading.** On the stop-mover arms (the four `stop_tail_peak` arms) and d68d4 its arm is 7.5 cm, not 9.3; a later stop change could move it again. 6.5 cm keeps 1.0 cm to it and 0.7 cm to the longest THRU arm (5.8 cm).
5. **The scorer's default record is the frozen smx1a scan.** Every doc-56 number quoted from `census_score.py` without `STM_SCAN_RECORD` is graded on it, including doc 83's headline. For items the owner re-judged in smx3/smx4 the two records disagree; this round is the first where the difference decides the sign of a result. Reporting both is cheap and should be the habit.
6. **Next.** Doc 78's remaining items: **item 5**, the P4 rejection census (which gate excluded the 76 same-bundle gamma tags on items with a Michel), and **item 6**, the hand-check of four items with nothing fitted at the stop. Item 5 is next: doc 80's per-piece `rej` column already carries most of what it needs.

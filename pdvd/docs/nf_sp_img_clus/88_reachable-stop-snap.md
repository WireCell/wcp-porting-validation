# 88 — Doc 78 action item 7b: the stop snap only onto what the entry reaches

Doc 87 built a size floor on doc 62's stop-local residual keep. At 5 cm, with 5 terminals and 5 cm, it gained both of the owner's smx5 Michels (`039253_0/44`, `039349_30/45`) with 0 through-going items touched. It was not flipped because `039349_36/63` lost its stopper call. Doc 87 §6.2 traced the cause:
- the kept residual's endpoint lies 0.64 cm from the tagger's stop, nearer than the chain's own end (2.29 cm);
- the stop snap (`anchor_vertex` → `closest_cluster_vertex`) takes the cluster's nearest vertex with no test that the entry can reach it;
- the residual is disconnected by construction, so the chain comes back empty, `R_STOP_UNMATCHED` is set, and the chain is walked to the farthest vertex.

The owner asked to advance to that next step (item 7b) under the same bar: an md, a flip if good, commit and push. Their standing answer from doc 87 still holds: if `039253_15/36` is the only new Michel FP, hold the flip and show it to them first.


**Status (2026-09-11, later): FLIPPED in PDVD production (§9).** The owner judged `039253_15/36` on smx6: STM + MICHEL, attached, "clear image results near the end of the STM … no track trajectory fit, but clear in image" (the predicted `unfitted`). On the record that now includes smx6, the flip candidate is Michel 141 / 12 / 20 → **144 / 12 / 17** (0 FP added, 0 TP lost) with `is_stm` unchanged. The four keys are in `pdvd/wct-pr-perevt.jsonnet`; PDHD stays OFF.

*What follows through §8 was written while the flip was held (2026-09-11, before the owner's answer).*
- **The knob.** `stop_snap_reachable` (C++ default false) acts only where the stop-local keep fired. There, the stop snaps onto nothing the entry cannot reach. With the knob off, output is byte-identical on both detectors. With the knob on and no keep, it is byte-identical on every shared branch and row (§5).
- **The flip candidate** is `p88v5fr`: doc 87's floored keep (5 cm, 5 terminals, 5 cm) plus the knob.
  - The skip fires on `039349_36/63` alone, and 36/63 is a stopper with its Michel again.
  - Against its doc 87 twin (`p87v5f`), 595 of 596 candidates are bit-identical; the one that moved is 36/63.
  - Against production, the Michel census goes 141 / 12 / 19 → **143 / 13 / 17**: the owner's `039253_0/44` and `039349_30/45` are gained.
  - `is_stm` is **233 / 7 / 45, unchanged**. 0 THRU items are touched, and nothing moves on a candidate without a keep fire.
- **Why the flip is held.** The only new Michel FP is `039253_15/36` (record: STM_ONLY, smx1a, high). The owner's answer in doc 87 still stands: if it is the only FP, hold the flip and show it to them. The flip follows their verdict.
- **On the unfloored 5 cm keep** the knob also undoes doc 62's loss `039349_61/21` (stopper and Michel back). That arm still touches 4 THRU items and adds a THRU Michel FP, so the floor stays.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts; S=$IMG/pdvd/docs/scan
export STM_SCAN_RECORD=$S/pdvd_stm_michel_smx1a_smx3_smx4_smx5_verdicts.json
# the build: toolkit 314186f8 + this round's hunks; doctests; the full local/lib pin (572 *.so*)
cd /home/xqian/toolkit-dev/toolkit && wcbuild && ./build/clus/wcdoctest-clus        # 396 cases pass
mkdir -p /home/xqian/tmp/p88/libpin_p88 && cp -L /nfs/data/1/xqian/toolkit-dev/local/lib/*.so* /home/xqian/tmp/p88/libpin_p88/
# the arms (wave 1: OFF gates, the flip candidate, the knob alone; wave 2: the two informational arms)
WAVE=1 nohup bash $X/d88_arms.sh > /home/xqian/tmp/p88/arms_wave1.log 2>&1 < /dev/null & disown
WAVE=2 nohup bash $X/d88_arms.sh > /home/xqian/tmp/p88/arms_wave2.log 2>&1 < /dev/null & disown
# gates, movers by name, census on the merged record, each ON arm against its doc 87 twin
bash $X/d88_gates.sh > /home/xqian/tmp/p88/gates.log 2>&1
# the blind smx6 hand check of 039253_15/36, from PRODUCTION's payload (no chain Michel drawn)
cd $IMG/pdhd/stm_michel_scan && python3 $X/d88_build_smx6.py --prep /home/xqian/tmp/p87/prep_p85vprod \
    --outprep $PWD/prep-pdvd-smx6 --sheet $S/pdvd_stm_michel_smx6_sheet.tsv \
    --questions $S/pdvd_stm_michel_smx6_questions.json --key $S/pdvd_stm_michel_smx6_key.tsv
./serve_stm_michel_scan.sh 5017 --det pdvd --scan-tag smx6 --manifest $S/pdvd_stm_michel_smx6_sheet.tsv \
    --prepdir $PWD/prep-pdvd-smx6 --questions $S/pdvd_stm_michel_smx6_questions.json
```

The predictions were written to `/home/xqian/tmp/p88/pred.txt` at 11:30:39, **before** any arm ran. Both waves launched from the same command, immediately after the write. The mover set comes from the tree: `n_kept_near_stop_main` and `stop_snap_skipped`, the latter written only when the knob is on.

## 1. What 7b fixes

This is doc 87 §6.2 in brief. `anchor_vertex` calls `PatternAlgorithms::closest_cluster_vertex`, which returns the cluster's nearest vertex; the split fallback takes the nearest fitted segment. Neither asks whether the entry can reach it.

In production that never matters: the only disconnected pieces of a main cluster are ones PR itself left, and doc pdhd/03's `R_STOP_UNMATCHED` fallback is the deliberate reading for a tagger stop in a detached fragment. The stop-local keep changes that. It adds a disconnected piece next to the stop on purpose, so the snap can land on it.

## 2. Design

- **One new knob, default OFF: `stop_snap_reachable`.**
  - It acts only on candidates where the stop-local keep fired on the main cluster (`n_kept_near_stop_main > 0`). Every other candidate runs the legacy snap, textually unchanged.
  - Where it acts, the stop snaps with `anchor_vertex_reachable`, a fork by duplication of `anchor_vertex` (CLAUDE.md M10). The rule is the same: the nearest vertex within `stop_snap_tol_cm`, otherwise a split of the nearest fitted segment within tolerance, otherwise the nearest vertex. The difference is that a vertex the entry cannot reach, or a segment with such an endpoint, is never a candidate.
  - `anchor_vertex` itself is untouched.
- **Two pure helpers in `StmMichelFunctions`:**
  - `stm_michel_reachable_vertices(g, from)`: everything reachable from the entry along graph edges, sorted by graph index (it reuses the chain walk's Dijkstra);
  - `stm_michel_closest_vertex_of(cands, pt, accept)`: the nearest candidate by its wcpt, the same point `closest_cluster_vertex` measures.
- **A branch, `stop_snap_skipped`** (written only when the knob is on): 1 where the legacy snap's nearest vertex was unreachable from the entry, i.e. where the knob changed the snap's candidate set. A DEBUG line `stop-snap-reachable:` gives both distances.
- **Why not all candidates.** On candidates without a keep fire the graph is production's, and doc pdhd/03's `R_STOP_UNMATCHED` fallback (a tagger stop in a detached fragment) is a deliberate reading there. Gating on the keep confines the knob to the graphs the keep changed.

## 3. Built (toolkit, uncommitted at run time; libWireCellClus md5 `53ea5fe5408b`)

| file | change |
|---|---|
| `clus/inc/WireCellClus/StmMichelFunctions.h`, `clus/src/StmMichelFunctions.cxx` | `stm_michel_reachable_vertices`, `stm_michel_closest_vertex_of` (pure) |
| `clus/src/CheckSTM_Michel.cxx` | knob `stop_snap_reachable` (C++ default false; read and round-tripped in `default_configuration`); `anchor_vertex_reachable`; the gated stop snap; `stop_snap_skipped` (written only when on) |
| `clus/test/doctest_stm_michel.cxx` | 1 case on a synthetic graph, 036/63's geometry: the legacy nearest vertex is the detached residual's (0.6 cm), the reachable one is the chain end (2.3 cm); order-independence; the predicate; empty and null inputs |
| `clus/test/doctest_check_stm_michel_defaults.cxx` | the default pinned |

## 4. Predictions (`/home/xqian/tmp/p88/pred.txt`), graded

| prediction | result |
|---|---|
| `p88voff` ≡ production, `p88hoff` ≡ PDHD OFF | **yes** (§5) |
| `p88vr` (the knob, no keep) ≡ production on every shared branch, row, zip and calib | **yes**: 596 / 596, 120 / 120 zips, 119 / 119 calib; `T_stm_michel` gains only `stop_snap_skipped` (0 everywhere) |
| `p88v5fr`: keep fires on the same 7 candidates as `p87v5f`; skip on `039349_36/63` only | **yes** (36/63: legacy nearest vertex 0.64 cm, unreachable; the reachable snap lands 1.37 cm from the tagger's stop) |
| `p88v5fr`: the 6 keep-fire candidates without a skip are bit-identical to `p87v5f` | **yes**: 595 / 596 identical to the twin; the one mover is 36/63 |
| *hypothesis:* 36/63 gets its stopper back and keeps its Michel | **held**: `is_stm` 1, bits 0, Michel found (13.7 → 18.6 MeV, conn 1 → 2) |
| `p88v5fr` census 143 / 13 / 17 and `is_stm` 233 / 7 / 45; the only new FP 15/36 | **yes, exactly** |
| `p88v5r`: skips on 61/21, 28/36, 43/62, 16/56 and 36/63 (marginal) | 4 of 5. **28/36 did not skip.** The skip also fired on **`039349_28/52`** (MESSY), which I had called "not a snap capture" (§6.2). |
| `p88v5r` *hypothesis:* 61/21 gets its stopper and Michel back (doc 62's loss undone) | **held** (`is_stm` 1, Michel 12.0 → 33.4 MeV) |
| `p88v20fr`: no skip anywhere, bit-identical to `p87v20f` (the negative control) | **no.** It skipped on `039252_2/87` (THRU, stays `is_stm` 0) and `039349_64/65`, whose stopper returns. 594 / 596 identical to the twin (§6.2). |

## 5. Gates (labels: arms `p88voff p88hoff p88v5fr p88vr p88v5r p88v20fr`; logs `/home/xqian/tmp/p88/arm_<arm>.log`, gate log `/home/xqian/tmp/p88/gates.log`)

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | 396 / 396 cases, 23 885 assertions (1 new case + 1 default pin) |
| freshness | `local/lib/libWireCellClus.so` 11:27:58 is newer than the last source edit (11:27:05) |
| binary pin | full 572-lib snapshot `/home/xqian/tmp/p88/libpin_p88`: md5 before = after; Clus `53ea5fe5408b` before and after all 6 arms; 0 loader deaths |
| compiled-config proof | with the `p88v5fr` TLA the four keys appear in the compiled `CheckSTM_Michel` data; 0 without (`/home/xqian/tmp/p88/cfgproof/`) |
| **OFF, PDVD** `p88voff` vs `p85vprod` | zips 120 / 120, trees 8 / 8, calib 119 / 119; 596 / 596 candidates bit-identical on all 145 branches with identical point geometry; 0 flips. **PASS** |
| **OFF, PDHD** `p88hoff` vs `p85hoff` | zips 61 / 61, trees 8 / 8, calib 61 / 61; 325 / 325 bit-identical on all 130 branches. **PASS** |
| **knob alone** `p88vr` vs `p85vprod` | as above, plus the one new branch. **PASS**: the knob is inert without a keep. |
| ON, every arm vs production | **0 candidates moved without a keep fire.** On `p88v5fr` the zips, calib, `T_proj_data`, `T_rec_charge` and `T_stm_michel_pts` differ on exactly the 7 keep-fire events. |
| ON, each arm vs its doc 87 twin | `p88v5fr`/`p87v5f` 595 / 596 (zip 119 / 120); `p88v5r`/`p87v5` 592 / 596; `p88v20fr`/`p87v20f` 594 / 596. Every mover is a skip candidate. |
| `census_score.py --check` | 0 of 14 differ |
| SBND / uBooNE | no jsonnet of theirs touched. The change is confined to `CheckSTM_Michel` (a PDVD/PDHD-only module) and new pure helpers; `anchor_vertex` and every production path are untouched. |

## 6. Results

### 6.1 On the merged record (546 judged items with a payload)

| arm | keep fires | skips | THRU fired on | Michel TP / FP / FN | Michel F1 | `is_stm` TP / FP / FN |
|---|---:|---:|---:|---|---:|---|
| production `p85vprod` | — | — | — | 141 / 12 / 19 | 0.901 | 233 / 7 / 45 |
| `p87v5f` (doc 87's candidate) | 7 | — | 0 | 143 / 13 / 17 | 0.905 | 232 / 7 / 46 |
| **`p88v5fr`** 5 cm + floor + knob | 7 | 1 | 0 | **143 / 13 / 17** | **0.905** | **233 / 7 / 45** |
| `p88vr` knob alone | 0 | 0 | 0 | 141 / 12 / 19 | 0.901 | 233 / 7 / 45 |
| `p88v5r` 5 cm + knob | 16 | 5 | 4 | 143 / 14 / 17 | 0.902 | 233 / 7 / 45 |
| `p88v20fr` 20 cm + floor + knob | 18 | 2 | 3 | 144 / 13 / 16 | 0.909 | 233 / 8 / 45 |

By name, against production:
- **`p88v5fr`**: Michel TP gained `039253_0/44` and `039349_30/45`; new FP `039253_15/36` (STM_ONLY); nothing lost.
  - The unjudged `039253_8/81` goes `is_stm` 0 → 1 (`no_bragg` cleared after the refit, as on every doc 87 arm).
  - The unjudged `039349_64/80` gains a 23.9 MeV Michel.
- **`p88v5r`**: Michel FP `039253_15/36` and `039349_20/73` (THRU). Its skips restore `039349_61/21` and `039349_36/63`. The MESSY `039349_46/58` still goes `is_stm` 1 → 0 on `plateau_off_mip` (no skip there; a refit effect). The MESSY `039349_28/52` goes 0 → 1 with a Michel.
- **`p88v20fr`**: the THRU `039252_8/93` is still an `is_stm` FP (0 → 1 on `no_bragg`, no skip, as on `p87v20f`). The skip restores `039349_64/65` (§6.2).

### 6.2 What the skip does, candidate by candidate (the DEBUG `stop-snap-reachable:` lines)

| arm | candidate | legacy nearest vertex (unreachable) | reachable snap | outcome |
|---|---|---:|---:|---|
| `p88v5fr`, `p88v5r` | `039349_36/63` | 0.64 cm (the kept residual's end; 2.23 on `p88v5r`) | 1.37 cm (2.71) | stopper and Michel restored |
| `p88v5r` | `039349_61/21` | 0.38 cm | 2.69 cm | stopper and Michel restored (doc 62's loss) |
| `p88v5r` | `039349_16/56` (THRU) | 0.60 cm | 5.80 cm | stays `is_stm` 0 |
| `p88v5r` | `039349_43/62` (THRU) | 0.32 cm | 25.94 cm | stays `is_stm` 0 |
| `p88v5r` | `039349_28/52` (MESSY) | 0.79 cm | 18.29 cm | `is_stm` 0 → 1, Michel 24.1 MeV |
| `p88v20fr` | `039252_2/87` (THRU) | 6.27 cm | 7.06 cm | stays `is_stm` 0 |
| `p88v20fr` | `039349_64/65` | 6.25 cm | **18.71 cm** | `is_stm` 1, Michel 10.1 MeV |

**Two kinds of skip.**
- **The first is the capture doc 87 traced.** The kept residual's own endpoint is the nearest vertex, and the chain's end is still reachable a little farther away: 36/63, 61/21.
- **The second is new.** After the keep, the muon's own end region is no longer reachable from the entry, and the reachable snap lands far from the tagger's stop: `039349_64/65` at 18.71 cm, `039349_28/52` at 18.29 cm, `039349_43/62` at 25.94 cm. On 64/65 the kept residual is 18.7 cm from the stop, and the reachable snap lands on it. So in the 20 cm arm the stop moves 18.7 cm onto a residual. The census reads that as a stopper with a Michel, but **it is not a validated stop**.

The second kind is doc 87's "uncharacterised path" on 64/65, now partly characterised: the keep changes which part of the cluster the entry reaches. It does not occur on the flip candidate. That arm's only skip is 36/63, and its reachable stop is 1.37 cm from the tagger's.

## 7. Flip decision

The bar was written before the arms, and it is doc 87's:
1. both OFF gates PASS;
2. every candidate without a keep fire is bit-identical;
3. on the merged record, 0 Michel TP lost, no new `is_stm` FP, no `is_stm` TP lost, and 0 THRU touched;
4. `--check` 0 / 14.

The owner's standing clause, from their answer in doc 87: if `039253_15/36` is the only new Michel FP, hold the flip and show them 15/36.

`p88v5fr` meets 1–4. Its only new Michel FP is 15/36, so **the flip is held**, and 15/36 is served blind:
- **tag `smx6`** (a new tag: `pdvd/work/stm_michel_labels/smx6/`), on **:5017**. The older label files are unchanged (md5 before and after);
- sheet, questions and key are in `pdvd/docs/scan/pdvd_stm_michel_smx6_*`, built by `scripts/d88_build_smx6.py`;
- **blind.** The payload is production's (`p85vprod`, `michel_found` 0), so the chain's new Michel is not drawn; the panel carries neither the record's call nor the arm's reading. A browser check shows the item, the question and an empty labels file, with no page errors (`/home/xqian/tmp/p88/shot_smx6.png`);
- **the question** is smx5's vocabulary: `MECH: <fit-through / unfitted / other-cluster / dead-region / not-imaged / no-michel / unclear>; DEAD: <U/V/W/seam or none>;`, then the pin and the verdict. The key predicts `unfitted` (a 12-terminal, 8.78 cm residual PR drops at 3.9 cm; doc 86 §8.1's largest unfitted excess on an STM_ONLY item).

**What the owner's answer does:**
- **STM + MICHEL:** 15/36 becomes a TP. `p88v5fr` is then +2 Michel TP, 0 FP, 0 stoppers lost. Flip the four keys `stop_local_residual_cm: 5.0`, `stop_local_residual_min_points: 5`, `stop_local_residual_min_len_cm: 5.0` and `stop_snap_reachable: true` in `pdvd/wct-pr-perevt.jsonnet`. Then run the proofs (the doc 87 `d87_proofs.sh` extended to four keys), the `p88vprod` ≡ `p88v5fr` confirmation arm, and the merged-record fold.
- **STM, no Michel:** the candidate is +2 TP for 1 FP at 0 stopper cost. Purity goes 0.922 → 0.917 and F1 0.901 → 0.905. It would then be the owner's call (docs 83–84 flipped only at 0 new FP).

PDHD stays OFF either way (no record).

## 8. Observations

1. **The 20 cm arms are not a quieter alternative.** `p88v20fr` has the best Michel F1 (0.909) but keeps the THRU `is_stm` FP `039252_8/93`, and its 64/65 recovery rests on a stop moved 18.7 cm.
2. **0/44's recovered Michel reads 5.0 MeV** for a 10.6 cm, 20-terminal piece. The bridged object's energy is doc 81's question (the 2-D charge estimator, built, OFF); recorded, not scoped.
3. **The unjudged `039253_8/81`** turns `is_stm` 1 on every arm that keeps its residual. It would enter a later scan tranche.
4. **Next** (as written before the answer): the owner judges `039253_15/36` on :5017 (smx6); then the flip or not (§7); then item 8's blind re-judge (doc 86 §9).

## 9. The owner's answer, the fold, and the flip

### 9.1 Score (`scripts/d88_score_smx6.py`, `/home/xqian/tmp/p88/score_smx6.txt`)

| item | record before | the owner (smx6) | key's MECH | owner's MECH |
|---|---|---|---|---|
| `039253_15/36` | STM_ONLY (smx1a, high) | **STM_MICHEL, attached** | `unfitted` | `unfitted`, from the notes: "Yes, there are some clear image results near the end of the STM. They were not identified as Michel, so no track trajectory fit, but clear in image." |

- **The answer's form.** The owner answered in the notes, not in the `MECH:` line. The mapping to `unfitted` is mine, and the words are quoted beside it. DEAD was not answered.
- **The pin.** It was left on the fit end (not placed).
- **`revealed_before_label` is true on every label the viewer writes now.** The chain's answer on that panel was production's (`michel_found` 0), so the label was not steered toward the flip.
- **The smx1a call and the display's blind spot.** The smx1a scanner's high-confidence STM_ONLY rested on "the display draws exactly ONE object… nothing past the stop". That is what a residual PR drops looks like on that display, the same blind spot that made `039253_0/44`'s first call wrong.

### 9.2 The fold

`pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_verdicts.json` has 601 rows. It is a new file; the smx5 record is untouched. It differs from the smx5 record on one row: 15/36 goes STM_ONLY → STM_MICHEL (`michel_kind` attached, confidence owner, source smx6). The stopper class is unchanged, and the Michel class goes False → True. The labels are copied to `pdvd/docs/scan/pdvd_stm_michel_smx6_labels.json`. **From here on, scoring uses this record.**

### 9.3 The census on the new record (546 judged items with a payload)

| arm | Michel TP / FP / FN | purity | F1 | `is_stm` TP / FP / FN |
|---|---|---:|---:|---|
| production `p85vprod` | 141 / 12 / 20 | 0.922 | 0.898 | 233 / 7 / 45 |
| **`p88v5fr`** | **144 / 12 / 17** | **0.923** | **0.909** | 233 / 7 / 45 |

Michel TP gained: `039253_0/44`, `039253_15/36`, `039349_30/45`. Michel TP lost: none. FP new: none. `is_stm`: identical. THRU touched: 0. **Bars 1–4 of §7 hold with 0 new FP.**

### 9.4 The flip and its proofs

The flip is four keys at the end of `stm_michel_knobs` in `pdvd/wct-pr-perevt.jsonnet`, each with a comment giving the C++ default, the numbers and the owner's scans:
- `stop_local_residual_cm: 5.0`;
- `stop_local_residual_min_points: 5`;
- `stop_local_residual_min_len_cm: 5.0`;
- `stop_snap_reachable: true`.

The T3b comment's "T3a is NOT set" is updated to point here. The PRE copy is `/home/xqian/tmp/p88/pre_wct-pr-perevt.jsonnet` (md5 `5cfa3f86`).

| proof (`scripts/d88_proofs.sh`, `/home/xqian/tmp/p88/proofs.txt`) | result |
|---|---|
| A: PRE + `p88v5fr`'s TLA vs POST | **0 lines** |
| B: POST with the radius forced to 0 vs PRE | the four keys present (radius 0; floors 5 / 5; snap true), all inert at radius 0. The floors are read only inside `m_stop_local_residual_cm > 0`, and the snap is gated on a keep fire (`p88vr` ≡ production, §5). |
| C: PRE vs POST | exactly the four keys |
| D: PDHD | 0 lines |
| `abtest/compile_all_cfg.sh` before and after, `cmp_cfg.sh` | all 16 live SBND / PDHD / PDVD jobs NORMDIFF 0: **OVERALL PASS** (`/home/xqian/tmp/p88/cfg_{before,after}`, `cfg_cmp.txt`) |
| confirmation arm `p88vprod` (the flipped file, no TLA) vs `p88v5fr` | zips 120 / 120, trees 8 / 8 on every event, calib 119 / 119; 596 / 596 candidates bit-identical on all 147 branches, identical point geometry, 0 flips (`/home/xqian/tmp/p88/g_confirm.txt`). Same pinned binary (Clus `53ea5fe5408b`, md5 unchanged). **PASS**: production is `p88vprod`. |

### 9.5 Next, ranked (the owner asked for suggestions after this round)

1. **A census of the 45 missed stoppers** (`is_stm` efficiency 0.838), by the reject bit that decides each one, on today's production and the smx6 record. Items 1–8 were mostly about Michels; this is the largest gap left. It is read-only. *Done (doc 89, 2026-09-11): 55 on all judged items = 45 + 10 the tagger never hands on. 5 are cleared at 0 FP by P1 floors of 3 MeV / 3 cm plus `plateau_mip_hi` 2.0 (the owner's call), 22 need a re-judge first, 8 are closed by the owner's fiducial / continuation preference, and the tagger's waiver cannot be graded on this record.*
2. **The PDVD flips of docs 83–88 graded on PDHD.** PDHD now has a hand-scan record (`pdhd_stm_michel_smx18_verdicts.json`, doc pdhd/18). It was a verdict-blind agent scan, not the owner's, so any PDHD flip would say so.
3. **Doc 78 item 9, the Michel charge energy.** First, why the nominal association-based `michel_ke_charge` reads 0 on 94–96 % of found Michels on both detectors. Then the role-0 measurement: how much Michel charge doc 81's association-based cell selection leaves out, before the definition is changed. *Done (doc pdvd/95, 2026-09-11): the role-0 measurement says the association drops a median 1495 cells per candidate, and the region definition that replaces it is FLIPPED in PDVD production — found Michels 34.6 MeV against 4.9 where the owner says a stopper has no Michel, where `michel_ke_best` reads 23.23 and 0.00. The `michel_ke_charge == 0` half is answered for PDVD by doc 81 §5.1's drift-frame diagnosis and is NOT re-diagnosed for PDHD.*
4. **Item 8's blind re-judge** of the anchor-tail rule's 10 fires. Its ceiling is now about 4 Michels (30/45 is found).
5. **Housekeeping.** The unjudged candidates the flip changes (`039253_8/81` becomes a stopper, `039349_64/80` gains a Michel) go in the next scan tranche.

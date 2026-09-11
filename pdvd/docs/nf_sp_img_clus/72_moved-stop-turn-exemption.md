# 72 — P3b: the moved-stop veto spares a hard-turning Michel (`moved_stop_michel_kink_min`)

This is P3b of doc 70 §4.2 / §6. It is built, armed and graded together with P2 (doc 73): one toolkit build, one arm set. The reason is that P3b is P2's safety net. P2 turns bridged Michels into attached ones, some of them on moved stops, where the moved-stop veto would otherwise demote them. The owner asked for each proposal in its own file, so this is the P3b file.

**Status (2026-09-10): DONE and in PDVD production.** The confirmation criteria were pre-stated (§4) before any arm ran, and every one held (§6–§7).
- **On: `moved_stop_michel_kink_min: 60.0`** (§8). The moved-stop veto now spares `039349_48/21`, an owner-confirmed Michel turning 132.6°.
- **`michel_found`** 133 / 12 / 25 → **134 / 12 / 24**. Nothing else moved: every other branch, point row, zip and calib is unchanged, `is_stm` included.
- **90° is bit-identical.**
- **Correction to doc 70:** its 60° cut spares one Michel, not two (59.6° < 60°). Its through-going kinks were survey-arm numbers (§2).
- **Toolkit:** built and gated with doc 73.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
# the production re-measurement of sec 2 (read-only: the p4v50 / d68a3 / d71vsp / p4v35 / p4voff / p4v60 logs)
python3 $X/d72_sizing.py
# build (toolkit clus/), twice for the new-symbol doctest link, then the tests
cd /nfs/data/1/xqian/toolkit-dev/toolkit && ./wcb build -p; ./wcb install --notests -p; ./wcb build -p; ./wcb install --notests -p
./build/clus/wcdoctest-clus
cp -a ../local/lib /home/xqian/tmp/p72/libpin_p72
# wave 1 (7 arms, bare production config), gates, grading
$X/d72_arms.sh; $X/d72_gates.sh 2>&1 | tee /home/xqian/tmp/p72/gates.log
```

## 1. The question

The moved-stop veto is T2c, doc 61, in PDVD production since then. It demotes an attached Michel (`michel_conn_type` 1) when:
- the stop was moved this event (a retreat or a split fired), and
- the Michel's `michel_ke_best` is below 10 MeV.

Doc 61 built it for five through-going items, all of which picked up a spurious attached arm when T1a/T1c moved their stop. It also named its one real cost.

Doc 70 §4.1 found that the veto demotes owner-confirmed Michels. For example `039252_2/79`, where the owner said "the current identified end point is OK", and `039349_48/21`. Doc 70 §4.2 proposed sparing an attached Michel that **turns** hard, since a KE floor does not read the turn. The owner's own discriminator is the kink.

## 2. Re-measured on today's production

On production (arm `p4v50`, bare production config, doc 71 §11), the veto fires on exactly five items. Here are their stop-arm DEBUG lines on six arms. Kink in degrees, segment in brackets where it changes:

| item | record | KE (MeV) | d68a3 · d71vsp (60 cm survey) | p4voff · p4v35 (35 cm) | **p4v50 (50 cm, production)** | p4v60 (60 cm) |
|---|---|---:|---:|---:|---:|---:|
| `039252_2/79` | STM_MICHEL, attached | 8.92 | 59.6 | 59.6 | **59.6** | 59.6 |
| `039349_48/21` | STM_MICHEL, attached | 8.67 | 132.6 | 132.6 | **132.6** | 132.6 |
| `039252_4/55` | THRU | 4.62 | 17.2 | 17.2 | **17.2** | 17.2 |
| `039349_20/41` | THRU | 5.73 | 43.5 | 43.5 | **43.5** | 43.5 |
| `039349_61/62` | THRU | 5.99 | 48.2 (62023) | 58.7 (62017) | **58.7** (62020) | 48.2 (62023) |

Two corrections to doc 70 follow from this table.

1. **The table in doc 70 §6 read "+2 owner-confirmed Michels (59.6°, 132.6°)" at 60°.** That is internally inconsistent: 59.6 < 60, so a ≥ 60° cut spares **one**, `039349_48/21`.
2. **Doc 70's THRU kinks, 17 / 44 / 48°, were survey-arm numbers.** The survey widens companion admission to 60 cm, and that changes `039349_61/62`'s fit through `preload_clusters` (doc 53 §6.2). Bare production at 35 or 50 cm gives 58.7°.

The spare decision is the same for **any threshold in (59.6°, 132.6°]** on all six arms: it spares `039349_48/21` and nothing else. The only kink that moved between arms moved *away* from that interval, not across it. But 60° sits 1.3° above the nearest through-going item, whose kink has already moved 10° with a config change that never touched the classifier. So both ends of the interval are armed: 60°, the value pre-stated in doc 70, and 90°.

Nothing else separates the two Michels from the three THRU items with a margin that means anything. Five points will always offer some separator: KE 8.7–8.9 against 4.6–6.0 MeV, or reach (len + far_len) 9.3–12.0 against 4.9–5.4 cm. These are reported as owner options (§9), not built. `039252_2/79` at 59.6° stays demoted; sparing it would need a cut inside the 0.9° between it and `039349_61/62`.

## 3. Design (toolkit `clus/src/CheckSTM_Michel.cxx`, the T2c site)

- **New knob** `moved_stop_michel_kink_min`, in degrees. The C++ default is −1, which is off.
- **The rule.** When the knob is ≥ 0, a would-be T2c demotion is skipped if `michel_kink_deg` ≥ the knob.
  - `michel_kink_deg` is the seed arm's kink at the stop: the classifier's 15 cm window, the number on the DEBUG line.
  - An unmeasurable kink (−1) is never spared.
- **New branch** `n_michel_veto_exempt`, written only when the knob is on (the P1/P4 pattern), so the knob-off tree is unchanged.
- **Why `is_stm` cannot move:** the veto writes only `michel_conn_type`, never `reject_bits`. A spared Michel is below `moved_stop_michel_ke_min` (10 MeV), and P1's `topology_stop_evidence` needs ≥ `topology_michel_ke_min` (10 MeV). So it cannot reach P1.
  - That holds **only while those two floors are equal**; the dependency is written into the knob's comment.
  - It is verified on the arm (§4 criterion 3), not assumed.
- **Why the PF cannot move:** the veto sits after the Michel's `set_pdg` and Shower are built. It never undid them, so `mabc-pr.zip` and the calib JSON do not depend on it.
- **What does move:** P4 phase 2 reads `michel_found`, so a spared Michel gets its role-4 gamma blobs and P4 branches.

## 4. Pre-stated confirmation criteria (written before wave 1 launched)

Flip `moved_stop_michel_kink_min: 60.0` in PDVD production if all of these hold:

1. **OFF gate.** Knob off, P72 pin against the P4 pin (today's production binary):
   - PDVD `p72vleg` ↔ `p72voff`: `mabc-pr.zip` member content, calib md5, and every `T_stm_michel` branch and point row identical.
   - PDHD `p72hleg` ↔ `p72hoff`: the same.
   - Stale-baseline check: `p72vleg` identical to `p4v50`.
2. **Predicted neutrality**, `p72voff` ↔ `p72vb60`:
   - zip and calib identical on 120/120.
   - Every branch and point row identical on every candidate **except** `039349_48/21`:
     - `michel_conn_type` 0 → 1 and `michel_found` 0 → 1;
     - `n_michel_veto` 1 → 0;
     - its P4 fields and role-4 rows, if P4 finds blobs.
   - `n_michel_veto_exempt` exists: 1 on that item, 0 elsewhere.
3. **`is_stm` identical** on every candidate, the check that the two 10 MeV floors are equal.
4. **Plateau:** `p72vb60` ≡ `p72vb90` on every branch and row.
5. **Gain and cost:** ≥ 1 record Michel gained by `michel_found`, 0 THRU or other non-Michel items re-admitted, and `census_score.py --check` 0 of 14.

Any failure: no flip, the failure named, and a stop (CLAUDE.md §5.5/§5.7).

## 5. What was built (toolkit, one build with doc 73)

- **`CheckSTM_Michel.cxx`:**
  - the knob `moved_stop_michel_kink_min`, in configure, in `default_configuration` (its comment carries the two-10-MeV dependency), and as a member at −1;
  - `Record::n_michel_veto_exempt`;
  - at the T2c site, the knob is read before demoting;
  - the branch is written only when the knob is ≥ 0.
- **Tests.** `doctest_check_stm_michel_defaults.cxx` pins the default at −1. The rule is one `if` in `visit()` with no pure-function seam. Its test is the arm itself, against an exact per-item prediction (§4, criteria 2–4); that is stated rather than left implied.
- **Build.** Built twice (the new-symbol doctest link trap: the first build's doctest link fails against the old installed library, as expected). `wcdoctest-clus`: **366 / 366** (363 before, plus doc 73's three cases).
  - Freshness: `local/lib/libWireCellClus.so` 11:53:21, after the last source edit at 11:50:28.
  - Pin: `/home/xqian/tmp/p72/libpin_p72`, md5 `8aa45329`. The production pin `/home/xqian/tmp/p4/libpin_p4`, md5 `1af88622`, was re-verified.
- **Compiled-config proof** (`/home/xqian/tmp/p72/cfg/`): bare production carries none of the five new keys. `-S stm_michel_extra={moved_stop_michel_kink_min:60.0}` adds exactly that one leaf.
- **Prep:** `prep_stm_michel_scan.py` VERDICT_SCALARS gains `n_michel_veto_exempt`.

## 6. Gates (wave 1, 2026-09-10 12:01–12:19; `/home/xqian/tmp/p72/gates.log`)

| gate | pair | result |
|---|---|---|
| completeness, pins | all 7 arms | PDVD 120/120 per arm, PDHD 61/61; 0 loader deaths. Pins unchanged before and after every arm (P4 `1af88622`, P72 `8aa45329`) |
| **OFF, PDVD** | `p72vleg` ↔ `p72voff` | `mabc-pr.zip` 120 / 120 same; calib 119 / 119 same (one event has none on either arm); **578 / 578 candidates bit-identical on all 137 branches**; point geometry and roles identical on 578 / 578 |
| **OFF, PDHD** | `p72hleg` ↔ `p72hoff` | zip 61 / 61, calib 61 / 61, **325 / 325 × 130 branches**, points identical |
| stale baseline | `p4v50` ↔ `p72vleg` | zip 120 / 120, calib 119 / 119, 578 / 578 × 137 branches, points identical: production today is what doc 71 §11 graded |
| **P3b neutrality** | `p72voff` ↔ `p72vb60` | zip 120 / 120, calib 119 / 119; 577 / 578 candidates bit-identical, `is_stm` 0 flips; the one that differs is **`039349_48/21`**: `michel_conn_type` 0 → 1, `michel_found` 0 → 1, `n_michel_veto` 1 → 0, new `n_michel_veto_exempt` 1 (0 everywhere else); point rows identical (P4 found no gamma blob for it) |
| **plateau** | `p72vb60` ↔ `p72vb90` | zip 120 / 120, calib 119 / 119, **578 / 578 × 138 branches bit-identical**, points identical |
| record | `census_score.py --check` | 0 of 14 differ |

## 7. Result — every §4 criterion holds

| arm | `is_stm` TP / FP / FN | `michel_found` TP / FP / FN | moved-stop veto: fires / vetoed / spared |
|---|---|---|---|
| `p72voff` (= production) | 225 / 7 / 51 | 133 / 12 / 25 | 5 / 5 / 0 |
| `p72vb60` | 225 / 7 / 51 | **134 / 12 / 24** | 5 / 4 / **1: `039349_48/21`** (STM_MICHEL, 132.6°) |
| `p72vb90` | 225 / 7 / 51 | 134 / 12 / 24 | 5 / 4 / 1: the same |

- `michel_found` purity 0.917 → 0.918, efficiency 0.842 → 0.848, on the 544 judged items.
- The four still vetoed are `039252_2/79` (STM_MICHEL, 59.6°) and the three THRU items at 17.2°, 43.5° and 58.7°. The prediction of §2 held item for item.
- 1. OFF gate: holds on both detectors, plus the stale-baseline check.
- 2. Predicted neutrality: holds exactly. The one differing candidate differs in exactly the predicted fields; zip and calib are untouched.
- 3. `is_stm` is identical, confirming the two-10-MeV dependency on the arm.
- 4. The plateau holds: 60° and 90° are bit-identical.
- 5. One record Michel gained, zero non-Michels spared, and `--check` shows 0 of 14 differ.

## 8. The production flip (PDVD, 2026-09-10)

`pdvd/wct-pr-perevt.jsonnet` gains `moved_stop_michel_kink_min: 60.0`, after `michel_gamma_radius_cm`. Its comment carries the C++ default and the arm result.
- **Edited only after wave 2 had finished,** with no `wire-cell` or `wcsonnet` process running (checked with `ps`).
- **md5** `23bebb30` → `f596bedc`.

Compiled-config proofs (`/home/xqian/tmp/p72/flip/proofs.txt`, with `/home/xqian/tmp/p72/flip_proofs.sh`):

| proof | compared | result |
|---|---|---|
| A: flip-equivalence | pre-flip file + `-S stm_michel_extra={moved_stop_michel_kink_min:60.0}` vs the flipped file | **0 lines** |
| B: OFF path | both files with the key forced to −1 | **0 lines** |
| C: pre vs post | the two files as they stand | exactly `"moved_stop_michel_kink_min": 60` |

The first attempt used a `wcsonnet` flag that does not exist (`-J`; the right one is `-P`). Every compile failed, and the diffs of the missing files printed "0 lines". The script now aborts on a failed compile. The table is the re-run, over five compiled configs of 277 KB each.

By proof A, **production is now what arm `p72vb60` ran**:
- `is_stm` 225 / 7 / 51 and `michel_found` 134 / 12 / 24;
- zip and calib identical to production before the flip.

A **bare-production confirmation arm, `p72vprod`**, confirms it. It ran the flipped file on the P72 pin with no extra knobs (JOBS 16):
- 120 / 120 events, 0 loader deaths;
- against `p72vb60`: zip 120 / 120 and calib 119 / 119 identical, **578 / 578 candidates bit-identical on all 138 branches**, point rows identical;
- census `is_stm` 225 / 7 / 51, `michel_found` 134 / 12 / 24.

The production baseline prep is now `/home/xqian/tmp/p72/prep_p72vprod`.

**PDHD stays OFF.** There is no PDHD hand-scan record, and the moved-stop veto itself is not on in PDHD production, so the key would be inert there.

## 9. For the owner, and next

- **60° or 90°.** The two are bit-identical on every branch, row, zip and calib (§6). 60° is the value doc 70 pre-stated, and it sits 1.3° above the through-going `039349_61/62` (58.7°), whose kink has moved 10° between arms. 90° gives the same result with 31° of margin. It is a one-key change, and this arm already grades it.
- **`039252_2/79` (59.6°, owner: "the current identified end point is OK") stays demoted.** Sparing it would need a threshold in (58.7°, 59.6°], a 0.9° window, which I do not propose.
- **Other separators exist on these five items, but none was built:**
  - a 7 MeV KE floor, which would demote the THRU items at 4.6–6.0 MeV and spare the Michels at 8.7–8.9;
  - a reach test, len + far_len ≥ 8 cm (Michels 9.3–12.0, THRU 4.9–5.4).

  Both are fitted to the same five points, which will always offer some separator.

  *(Doc 84, 2026-09-11: on the 12 T2c instances of 28 arms the KE floor does not separate, and the reach test does, in (5.8, 7.5] cm. It was built as `moved_stop_michel_reach_min_cm` and flipped in PDVD production at 6.5 cm, not 8: `039252_2/79` reads 7.5 cm on the stop-mover arms.)*
- **Next:** P3, the collinear split on a confirmed chain (doc 70 §4.2), in its own doc.

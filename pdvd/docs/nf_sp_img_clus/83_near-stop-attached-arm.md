# 83 — Doc 78 action item 4: the attached arm near the stop

The owner asked (2026-09-11, after doc 82) to proceed with doc 78's action item 4, which doc 82 §9 had recommended next. As doc 80 re-read it, this is an attached-arm rule: the four Michels doc 78 §3.2 found missing on `michel_found`-0 stoppers are PR segments **attached to the muon chain** (doc 80: role 8, `rej 13`). No Michel test ever looks at them.

**Status (2026-09-11): FLIPPED in PDVD production.** The knob is `michel_near_stop_arm_cm` in `CheckSTM_Michel` (C++ default 0 = off), and PDVD production runs it at `5.0`. Two missed Michels are recovered, `039349_64/24` and `039349_9/19`, each with the scanner's own segment as the seed. On the record:
- `michel_found` true positives go 134 → 136.
- `is_stm` true positives go 225 → 226: the new Michel lets `topology_stop_evidence` clear `039349_9/19`.
- There are **0 new false positives and 0 true positives lost**.
- The other 594 candidates are **bit-identical on every production branch**.

The OFF path is byte-identical on both detectors. PDHD stays off.

*(Doc 84 §7.3, 2026-09-11: the counts above are `census_score.py`'s default record, the frozen smx1a scan. On the owner's merged record, the grading record since doc 68, the same arms read `michel_found` 138 / 12 / 22 → 140 / 12 / 20 and `is_stm` 232 / 7 / 46 → 233 / 7 / 45: the same +2 and +1, 0 FP. Both targets are STM_MICHEL on both records.)*

Of doc 78's four targets, two are refused by the production gate, which was not loosened for them. Doc 78's "as energy" half is doc 73's failure family and was not built (§2.4). This round also corrects one slip in doc 82's offline twin (§9.5).

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
mkdir -p /home/xqian/tmp/p83
# the sizing, read-only, on doc 80's census arm (production p79vprod + segment_census)
python3 $X/d83_sizing.py > /home/xqian/tmp/p83/sizing.txt
# the build: toolkit 082376c5 + this round's hunks; then the full local/lib pin
cd /home/xqian/toolkit-dev/toolkit && ./build/clus/wcdoctest-clus          # 387 cases pass
mkdir -p /home/xqian/tmp/p83/libpin_p83 && cp -L /nfs/data/1/xqian/toolkit-dev/local/lib/*.so* /home/xqian/tmp/p83/libpin_p83/
# the arms (bare production + the key), the gates, the flip proofs, the confirmation arm
WAVE=1 nohup bash $X/d83_arms.sh > /home/xqian/tmp/p83/arms_wave1.log 2>&1 < /dev/null & disown
bash $X/d83_gates.sh > /home/xqian/tmp/p83/gates.log 2>&1
PRE_EVT=/home/xqian/tmp/p83/pre_wct-pr-perevt.jsonnet VAL=5.0 bash $X/d83_proofs.sh > /home/xqian/tmp/p83/proofs.txt
WAVE=2 nohup bash $X/d83_arms.sh > /home/xqian/tmp/p83/arms_wave2.log 2>&1 < /dev/null & disown
```

The predictions were written to `/home/xqian/tmp/p83/pred.txt` at 05:56:08, **before** any arm ran; the first arm started at 05:56:32.

## 1. Where the four segments hang

The graph is rebuilt from each payload's `pf.seg` and `pf.vtx` (`d83_sizing.py` §1):

| item | segment | where it leaves the chain | distance before the stop | length | charge | turn | production reading |
|---|---|---|---:|---:|---:|---:|---|
| `039349_64/24` | s24003 | penultimate vertex (degree 3) | 2.5 cm | 14.4 cm | 1.11 MIP | 146° | interior kOther |
| `039349_9/19` | s19003 | penultimate vertex | 3.8 cm | 8.0 cm | 0.87 MIP | 106° | interior kOther |
| `039349_64/52` | s52018 | penultimate vertex | 6.5 cm | 7.8 cm (+ 34.6 cm shower subtree) | 0.55 MIP | 52° | interior kOther |
| `039349_69/56` | s56006 | penultimate vertex | 6.5 cm | 13.6 cm | 0.29 MIP | 90° | interior kOther |

All four items share one shape. The muon arrives at a vertex and leaves two branches. The chain takes one of them as its last segment, a 2.5–6.5 cm stub (on `039349_64/24` that stub is the Bragg peak itself, 137 k e/cm). The Michel is the other branch.

Because the stub is the chain's last segment, the stop-arm classifier (`stm_michel_classify_stop_arm`, run only at the stop vertex) never sees the Michel. None of the four has a `stop-arm:` log line. The interior-vertex classifier (`stm_michel_classify_chain_arm`) reads the Michel as neither a ≤ 8 cm terminal delta nor a > 1.4 MIP hadron. It files it as kOther, and nothing reads kOther. The doc 80 census row says the same thing in its own terms: `rej 13`, attached and not taken.

## 2. Sizing (`d83_sizing.py`, read-only)

### 2.1 What the offline twin is worth

The twin rebuilds each arm the way `measure_arm` measures one. It is validated against the C++'s own `stop-arm:` lines on the same arm (§0 of the script): 207 of 208 stop arms matched.

| quantity | twin − C++, p50 | p90 | p99 |
|---|---:|---:|---:|
| length | 0.000 cm | 0.000 | 0.000 |
| far subtree | 0.000 cm | 0.004 | 51 (where the cap bites) |
| charge | 0.005 MIP | **0.09** | 0.31 |
| turn | 5° | 18° | 58° |

The turn's error tail comes from short reference segments. The C++ reads a stop arm against `last`, which can be 2.4 cm long; the twin reads against 15 cm of muon rows. At an interior vertex the reference is the incoming chain segment, which is ≥ 35 cm on every target. So the twin's turn is sound here and its **charge is good to about 0.1 MIP**.

The twin has two blind spots, both measured on the arms (§7.3), and neither touches the targets:
- **The payload's `michel_conn_type` is the final value, after T2c and T3c.** The block runs before them. So "items with no Michel today" over-includes an item whose bridged Michel T3c demoted later; the block never runs there.
- **A chain segment with no role-1 point rows reads as an arm** when its points sit off the muon rows. Every such case on this record is at a degree-2 vertex or on an item whose `n_body_other` is 0 in C++.

The payload does not carry the shower flag, and 170 of the 207 stop arms are shower-flagged. For a flagged arm the turn bar is 15°, not 30°. No unclaimed near-stop arm is refused on the turn alone in [15°, 30°), so the flag decides nothing below.

### 2.2 The rule, and what holds its purity

The rule offers every unclaimed arm (role 7, role 8, or none) at an interior chain vertex within *G* of the stop the stop-arm gate, as production runs it: reach ≤ 25 cm, 0.3 < MIP < 2.0, turn ≥ 30° or shower-flagged. It applies only on items with no Michel today.

| *G* | fires | by class |
|---:|---:|---|
| 3 cm | 1 | `039349_64/24` |
| 5 cm | 2 | `039349_64/24`, `039349_9/19` |
| 7 cm | 2 | the same two |
| 10 cm | 5 | + `039252_12/114` (THRU, 9.9 cm), `039252_16/108` (STM_ONLY, 8.5 cm), `039349_37/33` (MESSY) |

Out to 7 cm, every fire is a target and no control item fires. The first control items arrive at 8.5–9.9 cm. **The distance is what holds the purity**, not a charge or turn threshold.

The other two targets are refused on production's own terms:
- **s56006** is at 0.29 MIP, 0.01 below the charge floor. That is inside the twin's charge error.
- **s52018's** reach is 7.8 cm plus a 34.6 cm shower subtree.

The gate is not loosened for either. A floor of 0.25 would admit s56006, and the THRU arm `039349_18/10` s10001 sits at 0.24 MIP, 2.4 cm from its stop. That item carries `no_bragg` only, so a Michel on it would also flip `is_stm`. It is the watch item named in advance.

### 2.3 The verdict channel

`topology_stop_evidence` is on in PDVD production. It clears `no_bragg`, `shape_flat` and `profile_sparse` for any conn-1 or conn-2 Michel of ≥ 10 MeV and ≥ 3 cm (`stm_michel_topology_clear`). `039349_9/19` carries exactly `no_bragg|shape_flat`, so its Michel should turn it `is_stm` 1. The two 10 cm control items carry `plateau_off_mip`, which topology does not clear.

### 2.4 Three variants, sized and not built

- **(a) Arms production already claimed as deltas (role 2) near the stop.** At 7 cm these add `039252_8/102` and `039253_2/13` (both record Michels) for **6 judged control fires** (4 THRU, 2 STM_ONLY), plus 2 MESSY. Dead.
- **(b) Doc 78's "the 21 stop-touching members on found items, as energy".** These are arms at the stop vertex itself that failed the gate: under 0.3 MIP, under 30°, or over the reach. Absorbing every unclaimed stop arm into a found Michel takes **19 scanner-tagged Michel pieces, 14 tagged muon/delta/gamma and 2 untagged**. The non-Michel pieces include those of `039349_76/23` and `039349_78/22`, the same two items that sank doc 73's three P2 sub-knobs. This is doc 73's family and was measured there. Dead.
- **(c) A stop arm whose reach walk loops back into the muon chain.** On `039349_32/63` s63006 the walk goes through 63007 and 63008 and back onto the chain 11.6 cm upstream, so it reads far = 206 cm. This is one item, reported in §9 as an observation.

## 3. Design

**A pure function** in `StmMichelFunctions.{h,cxx}`, `stm_michel_near_stop_arms(g, chain, chain_vtxs, max_dist, th, skip)`, graph-only and unit-tested:
1. It walks the chain's interior vertices from the stop backward. The along-chain distance is Σ `segment_track_length` of the segments past the vertex.
2. At each vertex it re-reads every non-chain, non-skipped arm with `stm_michel_classify_chain_arm` and leaves kDelta and kHadron alone, since production already acted on those.
3. It offers the rest `stm_michel_classify_stop_arm(g, chain[vi-1], arm, v, th)`: **the same gate a stop arm faces**, with the incoming chain segment as the kink reference and any P2 sub-knob that is on.
4. The nearest vertex with a kMichel wins, and its Michel arms are ordered longest first.

**In `CheckSTM_Michel`**, knob `michel_near_stop_arm_cm` (0 = off):
- **Placement and guard.** The block runs after the attached and companion stages, and only when `michel_conn_type` is still 0. So it can change no Michel that production already found.
- **The object.** On a hit it builds the object the way the attached path does (duplicated, M10), started at the arm's own vertex: role-3 rows, PDG 11, the Shower, `calculate_kinematics`, `michel_ke_core` and `michel_ke_range`.
- **Conn type.** It is **conn 1**. A new conn value would silently switch off both `stm_michel_topology_clear` and the gamma collect, which accept only 1 and 2.
- **`michel_dis_cm`** is set to the vertex's distance from the stop. This is the one conn-1 Michel whose `dis` is not 0, so the T2c comment that asserted "conn 1 ⟹ dis 0" is updated. Every consumer of `michel_dis_cm` was checked:
  - T3c reads it for conn 2/3 only.
  - The T2c veto does not read it.
  - `census_score.py`'s §15.1 what-if row ("drop dis > 3 cm & KE < 10 MeV") would drop a near-arm Michel under 10 MeV. Neither recovered Michel is under 10 MeV.
- **Design constraint, stated in the code the way T2c states its own: the block writes the Michel object and nothing else a verdict reads.**
  - A kContinuation answer from the classifier is ignored and never OR'd into `R_CONTINUATION`.
  - It does not touch `n_stop_arms`, `n_stop_other`, `n_body_other` or the `michel_guards_stop` demotion.
  - `topology_stop_evidence` is the one declared channel through which `is_stm` can then move.
- **New branches**, persisted only when the knob is on: `michel_near_arm`, `near_arm_dist_cm`, `n_near_arms_examined`. A DEBUG `near-arm:` line is written per Michel arm taken.

## 4. Built

- **Toolkit** (on `082376c5`), four files:
  - `clus/inc/WireCellClus/StmMichelFunctions.h` and `clus/src/StmMichelFunctions.cxx`: the struct and the function.
  - `clus/src/CheckSTM_Michel.cxx`: the knob, the block, the record fields and their persistence, and the T2c comment.
  - `clus/test/doctest_stm_michel.cxx`: four cases.
    1. The turned-back arm off the penultimate vertex is taken, its kink read against the incoming segment. Off, a one-segment chain, and the `skip` predicate each yield nothing.
    2. The distance gate: 9 cm is refused at 5 cm and taken at 10 cm.
    3. A delta and a hadron are skipped uncounted. A collinear MIP arm is offered and read as a continuation. A 0.25 MIP arm is offered and refused.
    4. The nearest vertex wins, longest arm first.
  - `clus/test/doctest_check_stm_michel_defaults.cxx` pins `michel_near_stop_arm_cm` at 0.
- **Build.** `wcbuild` needed a second pass: the first `wcdoctest-clus` link met the new-symbol trap. The result is `wcdoctest-clus` **387 / 387**. Freshness: `libWireCellClus.so` 05:51:55, after the last source edit at 05:49:52.
- **Pin.** `/home/xqian/tmp/p83/libpin_p83` is the full `local/lib` snapshot: 572 files, Clus `6af1f2eaaa74`, Root `46bf51057716`. The md5 was identical before and after every arm, with 0 loader deaths.

## 5. Criteria and predictions (from `pred.txt`, written before the arms)

**Flip bar.**
- At least 2 of the 4 targets recovered, each with the tagged segment as the seed. That bar equals the prediction: if the C++ recovered only one, the round would not flip, and the gate would not be loosened to reach two.
- 0 new `michel_found` false positives and 0 new `is_stm` false positives on judged items.
- 0 true positives lost.
- Every fire named.
- The OFF gate passes on both ProtoDUNEs.
- `census_score.py --check` differs on 0 of 14.

**Pre-registered choice between 5 and 7 cm.** Flip 7 only if it strictly dominates 5: the same false-positive set and at least one more true positive.

| arm | predicted | measured |
|---|---|---|
| `p83v5` | exactly `039349_64/24` s24003 and `039349_9/19` s19003; `039349_9/19` → `is_stm` 1; 0 FP | **exactly that** |
| `p83v7` | = `p83v5`, unless the C++ reads s56006 above 0.3 MIP | **= `p83v5`** (s56006 offered, refused) |
| `p83v10` | + `039252_16/108`, `039252_12/114` as `michel_found` FP; `is_stm` unchanged on both | **exactly that**; the twin's third 10 cm fire, `039349_37/33` (MESSY), is a chain segment the twin misread (§7.3) |
| watch item `039349_18/10` | FP at every *G* if the C++ reads ≥ 0.3 MIP | **offered, refused** at every *G* |

## 6. Gates (`d83_gates.sh` → `/home/xqian/tmp/p83/gates.log`)

| gate | result |
|---|---|
| **OFF, PDVD** `p83voff` vs `p82vprod` (production after doc 82) | Bee zip identical on 120/120 events (every member); calib json 119/119; **all eight `tracking-pr.root` trees identical on every event**; **596/596 candidates bit-identical on all 140 branches and every point row**; 0 flips |
| **OFF, PDHD** `p83hoff` vs `p82bhoff` | 61/61 zips; 61/61 calib; all eight trees; **325/325 × 130 branches and every point row** |
| completeness | 120/120 PDVD and 61/61 PDHD with `tracking-pr.root`; `039252_11` has no candidate line on every arm, exactly as on `p82vprod` (the event, not the arm) |
| `census_score.py --check` | 0 of 14 differ |

## 7. Result

### 7.1 `p83v5`, the candidate

Three kOther arms were offered the gate on the whole sample. Two were taken, and the third is the watch item, refused.

| item | record | segment (scanner's tag) | vertex before the stop | length | charge | turn (C++) | Michel KE | change |
|---|---|---|---:|---:|---:|---:|---:|---|
| `039349_64/24` | STM_MICHEL | s24003 (michel) | 2.51 cm | 14.4 cm | 1.13 MIP | 133° | 38.9 MeV | `michel_found` 0 → 1 |
| `039349_9/19` | STM_MICHEL | s19003 (michel) | 3.79 cm | 8.0 cm | 0.87 MIP | 93° | 16.8 MeV | `michel_found` 0 → 1; `is_stm` 0 → 1 (`no_bragg\|shape_flat` cleared by topology) |

Both arms are shower-flagged and terminal. The gamma collect, which runs for any attached Michel, took 2 blobs for `039349_9/19`'s new Michel (6.1 MeV, out to 19.7 cm; object total 23.0 MeV) and none for `039349_64/24`'s. The muon energy is unchanged on both, since the arm was never part of the muon. Every other candidate, 594 of 596, is bit-identical to production on all 140 branches and every point row. On the two that fired, what moved is the Michel, gamma-collect and energy branches, plus `reject_bits` and `topology_cleared_bits` on `039349_9/19`.

| census on the record (547 judged) | production `p82vprod` | `p83v5` |
|---|---:|---:|
| `is_stm` TP / FP / FN | 225 / 13 / 43 | **226** / 13 / **42** |
| `is_stm` purity / efficiency | 0.945 / 0.840 | 0.946 / 0.843 |
| `michel_found` TP / FP / FN | 134 / 18 / 18 | **136** / 18 / **16** |
| `michel_found` purity / efficiency / F1 | 0.882 / 0.882 / 0.882 | 0.883 / 0.895 / 0.889 |
| scanner-tagged Michel segments with role 3 | 234 | 236 |

The twin read both arms within its stated error. Length is exact, and charge is within 0.02 MIP. The turn reads 13° lower in C++ (133° vs 146°, 93° vs 106°), the fitted direction against the geometric one, and both clear the bar by 60° or more.

### 7.2 `p83v7`: identical to `p83v5`

Seven arms were offered and the same two taken. `039349_69/56` s56006 and `039349_64/52` s52018 were offered and refused, and so were `039253_1/105` and `039253_6/85`. The C++ logs only the arms it takes; the twin's readings of the four refusals are charge, reach, reach and reach. Per the pre-registration, 7 does not dominate 5, so **5 is flipped**.

### 7.3 `p83v10`, the negative control: the distance gate

Nine arms were offered and four taken: the two targets plus `039252_12/114` (THRU; s114008, tagged delta/other, 9.9 cm, 7.9 MeV) and `039252_16/108` (STM_ONLY; s108018, 8.5 cm, 8.4 MeV). `michel_found` FP rises 18 → 20 and purity falls to 0.872. `is_stm` is unchanged on both, since `plateau_off_mip` is not topology-clearable. This is the prediction to the item.

**The twin reconciled against what the C++ offered.** The twin's 10 cm list, restricted to items whose payload shows no Michel, holds 13 items; the C++ offered the gate on 9. The four the C++ skipped each have a reason read off the arm's own branches:
- `039349_19/24` had a **bridged Michel at block time**: `n_dots` 1 at 5.8 cm, demoted afterwards by T3c (`n_michel_range_veto` 1). The block runs only at conn 0, so it never ran there.
- `039349_15/18` (s18007), `039349_37/33` (s33006) and `039349_37/52` (s52007) are **chain segments the twin misread as arms**. Each sits at a degree-2 interior vertex, or on an item with `n_body_other` 0 in C++; none has a role-1 row that would tell the twin so.

On `039253_6/85` the twin lists two arms at one vertex and the C++ offered one, because s85018 is the same kind of chain segment. None of this touches the 5 or 7 cm result: the twin's fires there are the two targets, and the C++ took exactly those.

Both of these false positives are under 10 MeV at more than 3 cm from the stop, the kinematics doc 55 §15.1 calls impossible for a Michel. The `census_score.py` §15.1 row drops exactly them (FP 20 → 18). T3c applies that test only to conn 2/3 Michels, and this rule's Michels are conn 1 (§9.3).

## 8. Flip

This time the owner's go was for this knob: *"if things are good, flip on for PDVD"*. The bar of §5 holds on every line. One key goes into `pdvd/wct-pr-perevt.jsonnet` `stm_michel_knobs`, after `michel_range_energy_guard`, with a comment that names the items, the 7 and 10 cm arms, and the two refused targets:

```jsonnet
michel_near_stop_arm_cm: 5.0,
```

Proofs (`d83_proofs.sh` → `/home/xqian/tmp/p83/proofs.txt`):
- **A.** PRE plus `-S stm_michel_extra={michel_near_stop_arm_cm:5.0}` (the arm's config) vs POST unset: **0 lines**.
- **B.** POST plus the C++ default forced back (0.0) vs PRE: exactly the one key, present at 0 vs absent.
- **C.** PRE vs POST: exactly the one key, `"michel_near_stop_arm_cm" : 5`.
- **D.** `pdhd/wct-pr-perevt.jsonnet` carries 0 lines of it, so PDHD runs the C++ default 0.

Confirmation arm `p83vprod` (the flipped file, no TLA; `PROD=p83v5 VOFF=p83vprod bash d83_gates.sh` → `/home/xqian/tmp/p83/gates_confirm.log`) vs `p83v5`: Bee zip identical on **120/120** events, calib json 119/119, **all eight trees identical on every event**, **596/596 candidates bit-identical on all 143 branches** (the 140 plus the three new ones) and every point row. The pin was unchanged before and after.

Toolkit commit `5904f1d2` (pushed to `apply-pointcloud`).

## 9. Observations, and what this leaves

1. **One shipped path is exercised only by the doctest.** When two kMichel arms leave the winning vertex, both get role-3 rows and PDG 11, but the Shower is started from the longer one; the second adds energy only if `complete_structure` reaches it. That is the attached path's own behaviour, duplicated. On this record every winning vertex had exactly one kMichel arm, so the multi-arm case is covered by the fourth doctest, not by the sample.
2. **A one-item margin on each side, declared.** `039349_69/56` s56006 misses the charge floor by 0.01 MIP. `039349_18/10` s10001, a THRU arm 2.4 cm from a stop whose only objection is `no_bragg`, sits just below the same floor. The floor keeps its production value, and this record cannot tell a better one apart.
3. **The near-arm Michel has no kinematic guard of its own.** T3c's "too far and too soft" test runs for conn 2/3 only. At 5 cm the two recovered Michels are 17 and 39 MeV, so the question does not arise. If the distance is ever widened, the §15.1 row says T3c extended to `michel_near_arm` would remove exactly the two 10 cm false positives. That is a measured option, not a recommendation.
4. **The loop-back reach** (`039349_32/63` s63006): a stop arm's far walk can re-enter the muon chain through a side loop and read the muon as the arm's subtree (206 cm). One item on this record, `is_stm` 0, whose other piece is T3c-demoted. It would need a fence on the chain in the far walk, which is a change to production's stop-arm classification. Recorded, not built.
5. **A slip in doc 82's offline twin, corrected here.**
   - **The slip.** `d82_sizing.py` hard-coded `mip_dqdx` 50000 for the live-row floor, but the compiled PDVD value is 55000 (`wct-pr-perevt.jsonnet:578`). Re-run with 55000, now committed in the script:
     - Doc 82 §1's table moves by one item in places: production 5 / 11 → **4 / 9**; *f* 0.5 / 15° 16 / 11 / 60 → **15 / 10 / 60**; 0.5 / 25° 13 / 8 / 33 → **12 / 7 / 33**; 0.6 / 25° 13 / 8 / 42 → **12 / 7 / 42**; 0.7 / 15° unchanged at 16 / 10 / 88.
     - Its §2.3 twin validation moves 14 (2.5 %) → **13 (2.3 %)**.
     - Its side-sweep at `split_kink_min_deg` 10 reaches **2**, not 3: `039252_9/101` drops out.
   - **What does not change.** Doc 82 §3's operating point (8 signal, 6 within 2 cm, 2 control) is unchanged, and every C++ arm result of doc 82 is unaffected.
   - **One explanation that was wrong.** Doc 82 §7.4 put `039252_9/101`'s non-move down to the twin's graph over-count. With the right floor, the twin does not predict it either.
6. **Next.** Doc 78's remaining items:
   - **Item 3**, T2c's kink margin on `039252_2/79`, is a re-grade with no scan. Doc 82 noted that the item wants a smaller stop correction than doc 82 applied.
   - **Item 5** is the P4 rejection census.
   - **Item 6** is the hand-check of four items with nothing fitted at the stop.

   Item 3 is the natural next round: it is one number and the evidence is already on disk.

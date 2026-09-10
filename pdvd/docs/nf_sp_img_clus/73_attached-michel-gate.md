# 73 — P2: PDVD operating points for the attached Michel gate

This is P2 of doc 70 §4.2 / §6. It is built, armed and graded together with P3b (doc 72): one toolkit build, one arm set. Every P2 arm carries P3b, because P2 can turn a bridged Michel into an attached one on a moved stop, and P3b is what keeps the moved-stop veto from demoting it again. The owner asked for each proposal in its own file, so this is the P2 file.

**Status (2026-09-10): DONE — built, gated, and NOT flipped. All three sub-knobs stay OFF.** The criteria and the prediction were pre-stated (§4) before any arm ran.

As predicted, no sub-knob adds a `michel_found` true positive on a judged item. Each puts two owner-tagged `michel` segments into the Michel object at most, and each fails a pre-stated guard by attaching an arm the owner tagged as delta/other or muon (§7):
- **(a)** attaches two such arms on `039349_76/23`;
- **(b)** attaches two on `039349_78/22`, and also loses the stopper `039253_13/73`, whose new attached seed arm is too short for P1's 3 cm test;
- **(c)** attaches one on `039349_31/51`.

Two findings for later (§9):
- the production far-walk walks back through the stop into the muon on 5 arms;
- P1 reads the seed arm's length, not the object's.

**Toolkit `299d8bc4`,** one build with doc 72.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
python3 $X/d72_sizing.py          # sec 2, read-only on the production arm p4v50
# build + pin: doc 72 sec 0.  Wave 1, gates and grading:
$X/d72_arms.sh; $X/d72_gates.sh 2>&1 | tee /home/xqian/tmp/p72/gates.log
# wave 2, the (c) arm, chosen from the p72voff diagnostic (sec 6):
WAVE=2 ARMS2='p72v2c|moved_stop_michel_kink_min:60.0,michel_kink_window_cm:5.0' $X/d72_arms.sh
```

## 1. The question

Doc 70 §4.1 found found-stoppers whose Michel **is there as an arm at the stop and fails a Michel gate** (`kOther`). Doc 70 §4.2 proposed three PDVD operating points for the attached gate (`stm_michel_classify_stop_arm`). Each is a sub-knob defaulting to today's behaviour:

- **(a) `michel_mip_lo_turned`, 0.15 at `kink ≥ 60°`.** The diluted PDVD Michel sits under `michel_mip_lo` 0.3 MIP, and a lower floor applies only to an arm that turns hard. A Michel is admitted by its topology, never by low dQ/dx alone (the classifier's own rule since doc pdvd/42).
- **(b) `michel_far_len_shower_max_cm`.** A shower-flagged arm's far subtree (the Michel's own brems) is capped on its own, instead of counting in `len + far_len ≤ 25 cm`.
- **(c) the kink window.**

## 2. Re-measured on today's production — and two corrections to doc 70

The stop-arm DEBUG line carries every input the classifier reads: len, far_len, mip, kink, shower, terminal. So (a) and (b) can be re-judged exactly offline on production (`p4v50`; `d72_sizing.py`).

**Correction 1, (c):**
- The classifier's kink window is **15 cm** (`StmMichelArmThresholds::dir_window`, never overridden by CheckSTM_Michel), not the 5 cm doc 70 said. So the helpful direction is a *shorter* window, for the Michel turn test only.
- It **cannot be sized from the payloads.** Their segment points are not the fits: an offline reproduction of the 15 cm kink agrees with the C++ within 1° on 1 of 182 stop arms.
- So the build adds a log-only diagnostic (§3) and the knob-off arm sizes it.

**Correction 2, doc 70's four named items on production:**

| item | arm | production reading | what P2 would need |
|---|---|---|---|
| `039349_11/19` | 19004 | len 8.74, far_len 54.37, mip 0.22, kink 81.1°, shower-flagged | (a) *and* (b). The 54.37 is the capped walk's partial sum. The whole subtree is ~66 cm by payload endpoint matching, so it is **out under a 60 cm cap**. |
| `039349_69/56` | — | **no stop arm on production** | unreachable by any classifier change |
| `039253_3/61` | 61007 | len 7.80, far_len 7.08, mip 0.62, kink 18.0°, not shower | (c) only |
| `039349_32/63` | 63006 | far_len 206.63 | **out under any sane cap** (the over-clustered structure doc 70 warned about) |

**What (a) + (b) at the argued points admit on production.** At 0.15 MIP / 60°, and a 60 cm cap on shower-flagged arms, 13 kOther stop arms become Michels:

- **The one `michel_found = 0` item they touch** is `039252_13/66`, MESSY, not judged.
- **Items that already have a Michel** are the rest. Their bridged object (`conn_type` 2) becomes attached (1), with the arm itself in role 3:
  - record Michels `039252_14/37`, `039253_13/73`, `039253_3/79`, `039349_76/23`, `039349_78/22` (two arms), `039349_82/54`;
  - `039252_16/98` (STM_ONLY, already a `michel_found` false positive);
  - MESSY `039349_46/58` and `039349_81/54`.
- **Hazard.** `039349_76/23` sits on a moved stop (one retreat) with a 0.67 MeV bridged object today. As an attached Michel, the moved-stop veto would demote it unless P3b spares it; its kink is 125.6°, so it is spared at 60°. `039253_13/73` is also on a moved stop, but at 30.5 MeV the veto does not fire.

## 3. Design (toolkit `clus/`)

- **Knobs.** Four new `CheckSTM_Michel` knobs, all C++ default −1 = off except the turn angle:
  - `michel_mip_lo_turned`, with `michel_mip_lo_turned_kink_deg` (default 60);
  - `michel_far_len_shower_max_cm`;
  - `michel_kink_window_cm`.
- **The gate as a pure function.** They reach `StmMichelArmThresholds` through the new `stm_michel_michel_gate` (StmMichelFunctions.h).
  - When all three are off, the classifier runs its doc pdvd/48 expression **verbatim**; the doctest pins that the pure gate agrees with it on a 4320-point grid.
  - The continuation test, the shower-flag minimum kink, the Bragg-stub absorber and the persisted `michel_kink_deg` are not touched. P2 can only turn a kOther arm into a Michel.
  - The stop-extension loop uses the same classifier, so a new Michel can also stop an extension (`michel_guards_stop`). That is one route by which `is_stm` can move, and the census reports it.
- **(b) re-measures a shower-flagged arm's far subtree** up to its own cap with `stm_michel_far_subtree_len`:
  - it is the existing walk with the **stop vertex fenced off**;
  - at caps above 25 cm, the unfenced walk could loop back through the stop into the muon chain;
  - the doctest builds such a loop and checks the fence.
- **(c) adds `kink_w`**, the kink over the shorter window, to the Michel turn test.
- **Log-only diagnostic, on every arm.** The stop-arm DEBUG line gains three fields:
  - `kink5`: the kink over 5 cm;
  - `far_full`: the fenced walk to 100 cm;
  - `kink_w`: the (c) kink the arm was classified with, −1 when off.
  - The knob-off arm then sizes (b)'s cap and (c)'s window from the C++'s own numbers. Sanity check: 19004's `far_full` must read ~66 cm, not 200+.

## 4. Pre-stated criteria and prediction (written before wave 1 launched)

**Prediction for `p72v2ab`** (P3b 60 + (a) 0.15 + (b) 60 cm):
- **`michel_found`:** about **+0 TP** on judged items.
- **Where the gain shows:** in the object. `michel` tags land in role 3 on the converted items, and `michel_len` / `michel_kink_deg` / KE move on them.
- **Possible `is_stm` movers:** through `michel_guards_stop` in the extension loop, through the continuation demotion beside a Michel at a Bragg-confirmed stop, and through P1 reading the new `michel_len`. Each is named.

**Criteria, per sub-knob.** A sub-knob is flipped in PDVD production only if all of these hold:

1. The OFF gate passes (doc 72 §4 criterion 1; the OFF arm has every P2 knob off).
2. **Gain:** the role-3 recall of the owner's `michel` tags rises by ≥ 1 named segment, or `michel_found` gains ≥ 1 TP.
3. **Guards:**
   - 0 new `michel_found` FP on judged items;
   - 0 `is_stm` TP lost;
   - 0 new `is_stm` FP;
   - 0 new muon- or delta-tagged role-3 segments.
4. **Rule check:** the stop-arm DEBUG lines re-judged by `d72_score.py`'s twin give 0 mismatches.

Where the attribution is needed, `p72v2a` / `p72v2b` isolate (a) and (b). **(c) is armed at 5 cm**, the one window the diagnostic measures; its admits are listed from `p72voff` before its arm is read. A sub-knob that fails stays OFF, with the failure named (the T3a precedent, doc 62).

## 5. What was built (toolkit, one build with doc 72)

- **`StmMichelFunctions.{h,cxx}`:**
  - four `StmMichelArmThresholds` fields and `StmMichelArm::kink_w_deg`;
  - the pure `stm_michel_michel_gate`;
  - the fenced walk `stm_michel_far_subtree_len`;
  - `stm_michel_classify_stop_arm` keeps its doc pdvd/48 expression verbatim unless one of the three P2 fields is on.
  - `measure_arm`, the chain-arm classifier and `PRSegmentFunctions` are untouched.
- **`CheckSTM_Michel.cxx`:**
  - the four knobs, threaded into the classifier thresholds `th` (read by both the stop-extension loop and the stop-arm pass);
  - the stop-arm DEBUG line's `kink5 / far_full / kink_w` fields.
- **`doctest_stm_michel.cxx`, three new cases:**
  - the gate with P2 off equals the legacy expression on 4320 points, for each of the two shower-min-kink settings (0 mismatches);
  - the three operating points and their boundaries, on 19004- and 61007-like numbers;
  - a graph case: (a) turns a 90°, 0.2-MIP arm from kOther into a Michel. For (b), a shower arm carries a 40 cm branch plus a two-hop loop back into the stop. The fenced walk reads 40 cm + one hop; the unfenced walk goes through the stop into the muon (> 95 cm). With (b) at 60 cm the arm is a Michel, and at 40 cm it is not.
- **`doctest_check_stm_michel_defaults.cxx`** pins the four keys.
- **Build, tests, freshness and pins** are doc 72 §5.
- **Compiled-config proof:** the (a)+(b) TLA adds exactly its three leaves and the (c) TLA exactly its two (`/home/xqian/tmp/p72/cfg/`).
- **Early diagnostic check**, on the first 7 events finished on both `p72vleg` and `p72voff`:
  - every stop-arm line's pre-existing fields are identical to the production binary's;
  - on 8 non-terminal arms whose capped walk was exact, `far_full` equals `far_len`.

## 6. Gates (`/home/xqian/tmp/p72/gates.log` wave 1, `gates2.log` wave 2)

- **OFF gate, stale-baseline check, pins:** doc 72 §6. The OFF arm has every P2 knob off, and it is bit-identical to production on both detectors.
- **Wave 2** (`p72v2a`, `p72v2b`, `p72v2c`, JOBS 8 each): 120 / 120 each, 0 loader deaths, pin `8aa45329` unchanged before and after. Which sub-knob the (c) arm tests was fixed in §4 (5 cm); its admits were listed from `p72voff` (grading section F) before it was scored.
- **Against `p72voff`, P2 does reach the PF.** A converted arm becomes a Michel shower member.

  | arm | zips differing | calib JSONs differing | candidates bit-identical on all 137 branches | `is_stm` flips |
  |---|---:|---:|---:|---:|
  | `p72v2ab` | 10 | 11 | 566 / 578 | 1 |
  | `p72v2a` | 3 | 4 | 573 | 0 |
  | `p72v2b` | 6 | 6 | 571 | 1 |
  | `p72v2c` | 1 | 1 | 576 | 0 |

- **Rule check:** all 200 stop-arm lines per arm, re-judged by `d72_score.py`'s twin of the classifier using that arm's settings.
  - Every arm has one flagged line: `039349_10/58` arm 58020, printed mip 0.30. `p72v2ab` has one more: `039349_22/63` arm 63005, printed 0.15.
  - The persisted, unrounded values are 0.3032 and 0.1530. Both lie above their strict thresholds, as the C++ decided. **0 real mismatches.**
  - The cause is the twin's input, not the twin. It reads the DEBUG line, which prints mip with two decimals, so any arm whose mip prints as exactly 0.30 or 0.15 is flagged against a strict `>`. A later round reusing `d72_score.py` will see the same flag on such arms; check the persisted `michel_mip` before reading it as a defect.
- **The fence:** 19004's `far_full` is **63.68 cm**, against the payload estimate of ~66 cm, so there is no re-entry. On every non-terminal arm whose capped walk was exact, `far_full` equals `far_len`.

## 7. Result, by sub-knob (census on the 544 judged items)

| arm | settings | `is_stm` TP / FP / FN | `michel_found` TP / FP / FN | owner `michel` tags newly in role 3 | owner-tagged-bad segments newly in role 3 | pre-stated verdict |
|---|---|---|---|---|---|---|
| `p72voff` | production | 225 / 7 / 51 | 133 / 12 / 25 | — | — | baseline |
| `p72vb60` | P3b only | 225 / 7 / 51 | 134 / 12 / 24 | 0 | 0 | doc 72: flipped |
| `p72v2a` | P3b + (a) 0.15 MIP at ≥ 60° | 225 / 7 / 51 | 134 / 12 / 24 | **+2**: `039252_14/37`, `039253_3/79` | **2**: `039349_76/23` arm 23026 (delta / other) and arm 23061 (muon) | fails the tagged-bad guard |
| `p72v2b` | P3b + (b) shower far_len ≤ 60 cm | **224 / 7 / 52** | 134 / 12 / 24 | **+2**: `039349_7/4`, `039349_82/54` | **2**: `039349_78/22` arms 22007 and 22010 (delta / other) | fails: lost `is_stm` TP `039253_13/73`, and the tagged-bad guard |
| `p72v2c` | P3b + (c) 5 cm window | 225 / 7 / 51 | 134 / 12 / 24 | 0 | **1**: `039349_31/51` arm 51024 (muon) | fails: no gain, and the tagged-bad guard |
| `p72v2ab` | P3b + (a) + (b) | 224 / 7 / 52 | 134 / 12 / 24 | +5, the union | 4, the union | fails |

**The §4 prediction held.** In every column the one `michel_found` TP over production is P3b's `039349_48/21`. No P2 sub-knob adds a judged TP. (a) gives `michel_found` to `039252_13/66`, a MESSY item that is not judged.

- **Recall.** `michel` tags in role 3 go from 209 of 263 (79.5 %) to 211 with (a) or (b), and to 214 with both (81.4 %). The sum 209 + 2 + 2 would give 213. The extra one is `039349_22/63`'s arm 63005 (7.54 cm, far subtree 19.16 cm, 0.153 MIP, 69°, shower-flagged), which needs both knobs:
  - (a) alone lowers the charge floor but keeps the reach test, and 7.54 + 19.16 > 25 cm;
  - (b) alone relaxes the reach test but keeps the 0.3 MIP floor.

  Either way, the gain is small next to the contamination each knob brings.
- **Baselines.** `p72voff` is production *before* the P3b flip. Production *after* it is `p72vprod`, bit-identical to `p72vb60` (doc 72 §8). The one `michel_found` TP every P2 column adds over `p72voff` is P3b's. Future rounds grade against `/home/xqian/tmp/p72/prep_p72vprod`.
- **The lost stopper, `039253_13/73`.** The chain has a bridged Michel object (`conn_type` 2), 9.8 cm and 30.5 MeV. (b) makes arm 73015 (2.38 cm, 59°) the attached seed, so `michel_len` becomes 2.38. P1 needs `michel_len` ≥ 3 cm, so it no longer clears `shape_flat`, and `is_stm` goes 1 → 0. This is the P1 route §4 named.
- **The object where a sub-knob fires** (`conn_type` 2 → 1 unless noted):

  | knob | item | KE (MeV) | length (cm) | owner's tags on the converted arms |
  |---|---|---|---|---|
  | (a) | `039252_14/37` | 10.8 → 30.9 | | |
  | (a) | `039253_3/79` | 14.9 → 24.6 | | |
  | (a) | `039349_76/23` | 0.7 → 12.0 | | delta / other, muon |
  | (b) | `039349_82/54` | 27.1 → 45.8 | 22.1 → 4.0 | |
  | (b) | `039349_7/4` | 9.4 → 15.5 | | |
  | (b) | `039349_78/22` | 21.6 → 32.7 | | delta / other |
  | (c) | `039349_31/51` | 51.1 → 59.6 | | muon |

## 8. Production: no P2 sub-knob is flipped

All three stay at their C++ default, −1 = off, and none is set in `pdvd/wct-pr-perevt.jsonnet`. Production is P3b alone (doc 72 §8). The knobs, the pure gate and the diagnostic stay in the code: the DEBUG line keeps sizing (b) and (c) on every arm for free.

## 9. Observations, and next

- **The production far walk walks back through the stop into the muon.**
  - `segment_far_subtree_track_length` excludes only the stem, so a subtree that loops back to the stop continues into the muon chain.
  - On production, 5 stop arms read a far_len inflated this way. Capped walk, then fenced walk:

    | item | arm | capped `far_len` | fenced `far_full` |
    |---|---|---:|---:|
    | `039349_7/4` | 4009 | 217.44 | 8.63 |
    | `039349_7/4` | 4011 | 223.14 | 8.63 |
    | `039349_78/22` | 22007 | 50.28 | 16.95 |
    | `039349_78/22` | 22010 | 44.68 | 16.95 |
    | `039349_22/22` | 22010 | 81.40 | 72.16 |

  - All five are kOther today, partly because of it. A fence-only fix (keep the 25 cm budget, fence the walk) would admit `039349_7/4`'s 4009, which is `michel`-tagged, and `039349_78/22`'s two, which are delta-tagged. That is (b)'s split again, so it is not proposed. It is named because any future far_len test inherits the defect.
- **P1 reads the seed arm's length, not the object's.** `039253_13/73` shows that attaching a short seed un-clears a stopper P1 had cleared on the bridged object's length. A future change that attaches bridged Michels should make P1 read the object's length first.
- **Doc 70's (c) target cannot be reached.** `039253_3/61`'s arm 61007 turns **19.5° over the C++'s 5 cm window**, not the 40° of doc 70's offline chord. No window of 5 cm or more reaches it.
- **If the owner wants P2 back with another discriminator,** the tags already give the scan list:
  - where P2 is right: `039252_14/37`, `039253_3/79`, `039349_7/4`, `039349_82/54`;
  - where it attaches a non-Michel arm: `039349_76/23`, `039349_78/22`, `039349_31/51`.
- **Next:** P3, the collinear split on a confirmed chain (doc 70 §4.2), in its own doc.

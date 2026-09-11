# 87 — Doc 78 action item 7: a size floor on the stop-local residual keep

The owner scanned smx5 (doc 86). On `039253_0/44` and `039349_30/45` they saw "clear blob topology near the end of the stopping point… Those should be Michel electrons, but not identified by the PR chain at all. These should be improved." The blob is PR's own object. `find_other_segments` fits a residual at the stop and then drops it as isolated (`pr54 isolated-residual drop`). Doc 62's T3a (`stop_local_residual_cm`, default OFF) already keeps such a residual near the tagger's stop, and doc 62's T3b (production) then admits it into the Michel. T3a stayed OFF because at 20 cm it touched ~40 items and moved verdicts both ways. Doc 86 §8.1 sized a **size floor**: on today's production drop lines, 5 cm with ≥ 5 terminals and ≥ 5 cm touches 5 judged items and no through-going one. The owner asked for item 7 under the campaign's bar: a new md, a PDVD flip if good, commit and push.

**Status (2026-09-11): BUILT, measured, NOT flipped. The knob ships default OFF.**
- **Built.** `stop_local_residual_min_points` and `stop_local_residual_min_len_cm` floor T3a's keep (C++ default 0 / 0 = doc 62's keep unchanged). The OFF path is byte-identical on both detectors (§5).
- **The flip candidate** (`p87v5f`: 5 cm, 5 terminals, 5 cm) does what doc 86 predicted for the Michels:
  - Michel census 141 / 12 / 19 → **143 / 13 / 17**: the owner's two Michels are gained (`039253_0/44`, `039349_30/45`), and `039253_15/36` (STM_ONLY) is the one new FP;
  - 0 through-going items touched;
  - every candidate without a keep fire is bit-identical to production.
- **It fails the pre-registered bar on one stopper.** `039349_36/63` (STM_MICHEL, owner-grade record) loses its stopper call: `is_stm` 1 → 0 on `stop_unmatched`.
- **The mechanism** is the same one behind doc 62's lost TP `039349_61/21` (§6.2):
  - the kept residual's endpoint lies closer to the tagger's stop than the chain's own end;
  - the stop snap (`anchor_vertex` → `closest_cluster_vertex`) takes the nearest vertex of the cluster with no reachability test, so it snaps onto the residual, which is disconnected from the entry;
  - the chain is then empty, `R_STOP_UNMATCHED` is set, and the chain is walked to the farthest vertex.
- **The owner's 15/36 look is not needed yet.** They chose to hold the flip and see 15/36 if it were the only FP. It is the only Michel FP, but the lost stopper blocks the flip on its own, so no smx6 scan was built.
- **Next (§8): item 7b.** Make the stop snap ignore vertices that are not connected to the entry, behind its own default-OFF knob, then re-run this arm. If 36/63 comes back, 15/36 is the only FP left, and the owner's look at it decides the flip.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
export STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_verdicts.json
# the build: toolkit 51adc923 + this round's hunks; doctests; the full local/lib pin (572 *.so*)
cd /home/xqian/toolkit-dev/toolkit && wcbuild && ./build/clus/wcdoctest-clus        # 395 cases pass
mkdir -p /home/xqian/tmp/p87/libpin_p87 && cp -L /nfs/data/1/xqian/toolkit-dev/local/lib/*.so* /home/xqian/tmp/p87/libpin_p87/
# the arms: wave 1 = OFF gates + the flip candidate, wave 2 = the informational radius/floor arms
WAVE=1 nohup bash $X/d87_arms.sh > /home/xqian/tmp/p87/arms_wave1.log 2>&1 < /dev/null & disown
WAVE=2 nohup bash $X/d87_arms.sh > /home/xqian/tmp/p87/arms_wave2.log 2>&1 < /dev/null & disown
# gates, movers by name, census on the merged record, the keep census per arm
bash $X/d87_gates.sh > /home/xqian/tmp/p87/gates.log 2>&1
python3 $X/d87_keep_census.py --arm p87v5f --prep /home/xqian/tmp/p87/prep_p87v5f \
    --base p85vprod --base-prep /home/xqian/tmp/p87/prep_p85vprod      # -> /home/xqian/tmp/p87/keep_p87v5f.txt
```

The predictions were written to `/home/xqian/tmp/p87/pred.txt` at 10:31:26, **before** any arm ran (wave 1 started at 10:31:51). The 20 cm + floor row was counted from production's drop lines by `/home/xqian/tmp/p87/pred_count.py` (`pred_count.txt`), which is `d86_sizing.py`'s R section with that row added.

**The mover set comes from the tree, not the log.**
- `n_kept_near_stop_main` / `_comp` (doc 62, always persisted) define which candidates have a keep fire. `n_floored_near_stop` (new, written only when a floor is set) counts the residuals the floor refused.
- The log lines give each residual's terminals, fitted length and distance to the tagger's stop.
- The two sources agree on every arm: 0 candidates carry more own-cluster keep lines than the tree counts.

"Terminals" is the pr54 `n_points`, the number of Steiner terminals, as in `other_seg_keep_isolated_ok`.

## 1. The object, and why doc 62's grade is stale

When `find_other_segments` fits an isolated residual and the pr54 floors (`other_seg_keep_isolated`) refuse it, the residual is dropped: its vertices are removed from the graph (`NeutrinoOtherSegments.cxx`, the `pr54 isolated-residual drop` branch).

Doc 62's T3a adds one exception. It keeps the residual when either fitted endpoint lies within `stop_local_residual_cm` of the STM tagger's stop, and it applies no size floor. On the owner's two items, production's drop lines read:
- `039253_0/44`: 20 terminals, 10.61 cm, 3.1 cm from the fit end;
- `039349_30/45`: 6 terminals, 8.53 cm, at 1.4 cm, plus a 4-terminal, 5.47 cm residual at 1.7 cm.

Doc 62 graded T3a at 20 cm on the smx1a record of its day. It counted `039253_0/44` among "five extra michel FPs". The owner has since called that item a Michel twice (smx3, smx5). Doc 62's production (`d61v`) also predates docs 63–85. This round re-grades T3a on today's production and today's record (`p87v20`, §6.3).

## 2. Design

- **The floor lives where the keep is decided.**
  - It sits in `NeutrinoOtherSegments.cxx`, beside doc 62's anchor test, and it is the same kind of edit doc 62 made there: an in-place, default-0 extension of an already-knobbed block.
  - The neutrino path (`TaggerCheckNeutrino`) never sets an anchor, so it runs the identical branch.
  - A new pure helper, `other_seg_keep_anchor_ok`, returns: an anchor was measured, the nearer endpoint is inside the radius, `terminals ≥ min_points`, and `fitted length ≥ min_length`.
  - At 0 / 0 its first two terms are doc 62's `near_anchor` expression verbatim.
- **The floor's defaults stay 0, not 5.** Defaulting them to 5 would silently change what the existing `stop_local_residual_cm` does for anyone who sets it (CLAUDE.md §5.1).
- **The floors are inert without a radius.** `CheckSTM_Michel` copies them into the pattern algorithms only inside `if (m_stop_local_residual_cm > 0)`.

## 3. Built (toolkit, uncommitted at run time; libWireCellClus md5 `f5c3d3c9a7ec`)

| file | change |
|---|---|
| `clus/inc/WireCellClus/PRSegmentFunctions.h`, `clus/src/NeutrinoOtherSegments.cxx` | `other_seg_keep_anchor_ok` (pure). `near_anchor` now comes from it. A residual inside the radius that the floor refuses is counted (`m_other_seg_keep_anchor_floored`), logged at DEBUG (`pr54 keep-isolated near-anchor floored: …`), and dropped exactly as the legacy path drops it. |
| `clus/inc/WireCellClus/NeutrinoPatternBase.h` | `m_other_seg_keep_anchor_min_points{0}`, `m_other_seg_keep_anchor_min_length{0.0}`, `m_other_seg_keep_anchor_floored{0}` |
| `clus/src/CheckSTM_Michel.cxx` | knobs `stop_local_residual_min_points` (int) and `stop_local_residual_min_len_cm` (cm), both default 0 and round-tripped in `default_configuration`. `n_floored_near_stop` is written only when a floor is > 0. |
| `clus/test/doctest_other_seg_keep_isolated.cxx` | 3 cases (20 assertions): off → never keeps; 0/0 reproduces the radius test; the 5/5 floor is inclusive and AND-ed, and is checked on the named residuals (`039253_0/44` kept, 30/45's 6-terminal piece kept and its 4-terminal piece refused, `039349_61/21`'s 2-terminal stub refused, two THRU stubs refused). |
| `clus/test/doctest_check_stm_michel_defaults.cxx` | the two new defaults pinned |

## 4. Predictions (`/home/xqian/tmp/p87/pred.txt`), graded

| prediction | result |
|---|---|
| `p87voff` ≡ production, `p87hoff` ≡ PDHD OFF | **yes** (§5) |
| `p87v5f` keep fires on 0/44, 30/45, 15/36, 36/63 and 37/39 (a lower bound) | **yes**, plus 2 unjudged candidates (`039253_8/81`, `039349_64/80`) |
| `p87v5f` floored > 0 on 30/45 (its 4-terminal piece) | **no.** Once the 6-terminal piece is kept and refit, the 4-terminal residual is not re-found. Floored residuals land on 11 other candidates (18 residuals). |
| `michel_found` 0 → 1 on 0/44 and 30/45; 15/36 a named record-FP; census 143 / 13 / 17 | **yes, exactly** |
| 36/63 and 37/39 keep their Michel | **yes** (37/39: 29.5 → 31.1 MeV; 36/63: 13.7 → 10.0 MeV) |
| `is_stm` moves only on keep-fire items; 0 THRU touched | **yes**; but the bar's "no stopper TP lost" **fails** on 36/63 |
| `p87v5` (no floor): the 2-terminal stub on 61/21 is kept and doc 62's lost TP reappears | **yes**: 61/21 `is_stm` 1 → 0 and `michel_found` 1 → 0, as in doc 62. `p87v5f` refuses the same stub twice (floored 2) and leaves 61/21 untouched. |
| `p87v5`: 3 THRU stubs (17/91, 20/73, 47/24) | 4 THRU fires: 16/56, 20/73, 43/62, 47/24. The anchor is the tagger's stop, not the payload's fit end. |
| `p87v20`: ~31 judged items, 14 THRU | 42 judged items (45 candidates), 16 THRU |
| `p87v20f`: 11 judged items, 2 THRU (2/87, 33/45) | 17 judged (18 candidates), 3 THRU: 2/87, 33/45 and `039252_8/93` |

## 5. Gates (labels: arms `p87voff p87hoff p87v5f p87v5 p87v20 p87v20f`; logs `/home/xqian/tmp/p87/arm_<arm>.log`, gate log `/home/xqian/tmp/p87/gates.log`)

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | 395 / 395 cases, 23 870 assertions (3 new cases) |
| freshness | `local/lib/libWireCellClus.so` 10:27:59 is newer than the last source edit (10:26:37) |
| binary pin | the full 572-lib snapshot `/home/xqian/tmp/p87/libpin_p87`: md5 before = after (`libpin_md5_{before,after}.txt`); Clus `f5c3d3c9a7ec` before and after every arm; 0 loader deaths |
| compiled-config proof | with the `p87v5f` TLA the three keys appear in the compiled `CheckSTM_Michel` data; without it, 0 (`/home/xqian/tmp/p87/cfgproof/{on,off}.json`) |
| **OFF, PDVD** `p87voff` vs `p85vprod` | zips 120 / 120, trees 8 / 8 on every event, calib 119 / 119. 596 / 596 candidates bit-identical on all 145 branches, identical point geometry, 0 role labels moved, 0 flips (`g_p87voff.txt`). **PASS** |
| **OFF, PDHD** `p87hoff` vs `p85hoff` | zips 61 / 61, trees 8 / 8, calib 61 / 61. 325 / 325 bit-identical on all 130 branches. **PASS** |
| ON, every arm | **0 candidates moved without a keep fire**, on any branch or point row, in all four ON arms. On `p87v5f` the zips, calib, `T_proj_data`, `T_rec_charge` and `T_stm_michel_pts` differ on exactly the 7 keep-fire events. `T_stm_michel` differs on 119 events only by the new `n_floored_near_stop` branch (589 / 596 candidates identical on the 145 shared branches). |
| `census_score.py --check` | 0 of 14 differ |
| SBND / uBooNE | no jsonnet of theirs touched, no production config touched. The new pattern-algorithm members are never set by `TaggerCheckNeutrino`, so their path is unchanged by construction (doc 62's precedent). No claim for them rests on an arm. |

## 6. Results

### 6.1 Arm by arm, on the merged record (546 judged items with a payload)

| arm | keep fires (candidates) | THRU fired on | Michel TP / FP / FN | Michel F1 | `is_stm` TP / FP / FN |
|---|---:|---:|---|---:|---|
| production `p85vprod` | — | — | 141 / 12 / 19 | 0.901 | 233 / 7 / 45 |
| **`p87v5f`** 5 cm + floor | 7 | 0 | **143 / 13 / 17** | **0.905** | 232 / 7 / 46 |
| `p87v5` 5 cm | 16 | 4 | 141 / 14 / 19 | 0.895 | 231 / 7 / 47 |
| `p87v20` 20 cm (doc 62's A) | 45 | 16 | 143 / 15 / 17 | 0.899 | 231 / 8 / 47 |
| `p87v20f` 20 cm + floor | 18 | 3 | 144 / 13 / 16 | 0.909 | 232 / 8 / 46 |

The changes by name (`keep_<arm>.txt` §C):

| arm | Michel TP gained | Michel TP lost | Michel FP new | `is_stm` TP lost | `is_stm` FP new |
|---|---|---|---|---|---|
| `p87v5f` | 0/44, 30/45 | — | 15/36 (STM_ONLY) | **36/63** | — |
| `p87v5` | 0/44, 30/45 | 36/63, 61/21 | 15/36, 20/73 (THRU) | 36/63, 61/21 | — |
| `p87v20` | 0/44, 30/45, 64/65 | 61/21 | 15/36, 17/91 (THRU), 20/73 (THRU) | 61/21, 64/65 | `039252_8/93` (THRU) |
| `p87v20f` | 0/44, 30/45, 64/65 | — | 15/36 | 64/65 | `039252_8/93` (THRU) |

All item names are 0392xx_… as in the record (0/44 = `039253_0/44`; 15/36 = `039253_15/36`; 20/73, 30/45, 36/63, 61/21, 64/65 = `039349_…`; 17/91 = `039252_17/91`).

- **The floor does its job.** Against the bare radius at 5 cm it removes every THRU fire and the 61/21 loss, and it adds nothing. At 20 cm it cuts the THRU fires from 16 to 3 and the FPs from 3 to 1.
- **Every arm loses at least one stopper, and every loss carries `stop_unmatched`.**
- **Doc 62's T3a re-graded (`p87v20`).** On today's record it is +2 Michel TP net (3 gained, 1 lost) for 3 new FPs, and `is_stm` −2 TP / +1 FP. Doc 62's "extra FP" 0/44 is now a TP. Its two `is_stm` movers of that day read differently on today's production:
  - `039252_12/114` still fires but stays `is_stm` 0;
  - `039252_8/93` is `is_stm` 0 in today's production and turns 1 (a new THRU FP).
  T3a unfloored stays OFF.
- **Two unjudged candidates move in every arm that fires on them.** `039253_8/81` goes `is_stm` 0 → 1 (`no_bragg` cleared after the refit). `039349_64/80` gains a Michel (23.9 MeV).

### 6.2 The loss: the stop snaps onto the kept residual

`039349_36/63` on `p87v5f`, from the tree, the payload and the keep line:
- the kept residual has 17 terminals and is 15.06 cm long, and one endpoint is **0.64 cm** from the tagger's stop (38.8, 192.5, 252.9);
- the chain's own end vertex in production is **2.29 cm** from it (`stop_dis`).

`anchor_vertex` (`CheckSTM_Michel.cxx`) calls `closest_cluster_vertex`, which picks the cluster's nearest vertex with no test that the entry can reach it. So the stop snaps onto the residual's endpoint. The residual is disconnected by construction (it is what T3b admits as a disconnected piece). So:
- `stm_michel_shortest_chain` returns nothing;
- `R_STOP_UNMATCHED` is set;
- the chain is walked to the farthest vertex of the main cluster: 2 segments, muon length 332.4 → 340.7 cm, the stop now 6.1 cm from the tagger's;
- `is_stm` 1 → 0, and the Michel survives as a bridged object (conn 1 → 2).

The same path explains:
- **`039349_61/21`** (unfloored arms). A 2-terminal stub at 0.4 cm: `stop_dis` 1.1 → 6.5 cm, bits 0 → 14 (`stop_unmatched` + `no_bragg` + `shape_flat`), and the Michel is lost. This is doc 62 §4.3's loss, reproduced on today's production.
- **`039349_36/63` in `p87v5`.** A 3-terminal residual at 2.2 cm: `stop_dis` 2.3 → 11.3 cm, bits 0 → 14.

**`039349_64/65`** (both 20 cm arms) is different and not characterised here. Its kept residual is 18.7 cm from the stop, too far to capture the snap, yet `stop_dis` goes 3.8 → 18.2 cm with `stop_unmatched`. The keep changed the graph the chain is built on some other way.

The 5 cm arms never fire on 64/65, and the floor alone saves 61/21. The one loss left on the flip candidate is the snap on 36/63.

## 7. Flip decision

The bar, written before the arms:
1. both OFF gates PASS;
2. every candidate without a keep fire is bit-identical;
3. on the merged record, 0 Michel TP lost, no new `is_stm` FP, no `is_stm` TP lost, and 0 THRU touched;
4. `--check` 0 / 14.

Bars 1, 2 and 4 hold. Bar 3 fails on `039349_36/63` (`is_stm` TP lost). Under the plan's rule for that outcome (a lost TP → no flip, ship the knob OFF, name each mover), **nothing is flipped**. `pdvd/wct-pr-perevt.jsonnet` is unchanged. PDHD stays OFF regardless (no record).

The owner's option is `-S stm_michel_extra={stop_local_residual_cm:5.0,stop_local_residual_min_points:5,stop_local_residual_min_len_cm:5.0}`: +2 Michel TP, +1 Michel FP (15/36), −1 stopper TP (36/63).

## 8. Observations and next

1. **Next: item 7b, the stop snap within the entry's component.** When the stop-local keep has fired on a candidate, the stop's `anchor_vertex` should consider only vertices the entry can reach. Kept residuals are disconnected by construction, so this removes exactly the capture of §6.2. It needs a new default-OFF knob, evaluated only on keep-fire candidates (7 on `p87v5f`), so every other candidate stays bit-identical by construction. Re-run `p87v5f` with it.
   - Predicted: 36/63 keeps its stopper and its Michel, and the rest is unchanged.
   - If so, 15/36 is the only FP left, and the owner's blind look at it decides the flip. The smx6 builder is drafted and parked in `/home/xqian/tmp/p87/d87_build_smx6.py`; it serves production's payload, so the chain's kept-residual Michel is not drawn.
2. **The new Michels' energy.** 0/44's recovered Michel reads 5.0 MeV (`michel_ke_best`), low for a 10.6 cm, 20-terminal piece. The others read 23.2 (30/45), 16.1 (15/36) and 23.9 (64/80) MeV. The bridged object's energy is doc 81's question (the 2-D charge estimator, built, OFF). It is recorded here, not scoped.
3. **`039349_42/41`** (MESSY, unjudged) loses its payload on `p87v20`: the kept residual collapses its chain from 12.6 to 1.9 cm, and the prep drops it. It is a 20 cm-only pathology and is not in the census.
4. **15/36** also belongs in item 8's blind re-judge (doc 86 §9), which is still the gate on item 8.

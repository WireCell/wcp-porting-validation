# doc pdhd/28 — The wire-lookup fix for the region Michel energy, flipped into PDHD production

Doc pdhd/27 §2 found that the region Michel energy measured PDHD's induction-plane distances on the wrong wire. The
owner's go, 2026-09-12: "Yes, push and fix the bug". This doc is that fix, its gates, and the PDHD production flip.

## Headline

1. **Fixed, and on in PDHD production.**
   * **Code:** toolkit `f516b013` adds `michel_q2d_region_wire_lookup` (C++ default false) and a pure
     `stm_michel_pick_wire` with doctests.
   * **Config:** `pdhd/wct-pr-perevt.jsonnet` gains that one key; the PDVD file is unchanged.
   * **Production arm:** PDHD's is now `h28prod` (pin `libpin_h28`, Clus md5 `30113227`).
2. **With the key absent the new build is bit-identical to production on both detectors:**
   * PDHD `h28off` == `h26q2dprod`, 61/61 events and 341 candidates;
   * PDVD `p97voff` == `p96vprod`, 120/120 events and 596 candidates.
3. **With it on, only what should move moves.**
   * **What moved:** region and control branches, `michel_q2d_n_role0`, the new `michel_q2d_n_rewired`, and
     multi-wire U/V rows of `T_stm_michel_2d`.
   * **What didn't:**
     * 0 `is_stm` / `michel_found` / `reject_bits` changes;
     * every W row identical (PDHD 170 029, PDVD 270 470);
     * on PDVD every single-wire U/V row identical (354 620).
   * **The defect is gone:** PDHD U/V footprint cells beyond 100 cm fall from 49–74 % to **0.000** in every class,
     and controls with no U/V cell from 91 to **6** of 341.
4. **The corrected PDHD Michel energy reads higher, not closer to PDVD.**
   * **PDHD hand Michel region median:** 39.5 → **45.3 MeV**.
   * **Above the 52.8 MeV endpoint:** 12 → **20 of 44**, against 12 of 134 on PDVD.
   * **Body control:** 8.5 → 11.1 MeV.
   * **Not leakage.** After the fix the induction/collection plane ratios on APA1/APA3 (U/W 1.19, V/W 1.05) match
     APA2's (1.14 / 1.15) and PDVD's (1.10 / 1.11). The restored U+V charge carries only 1.5 MeV-eq unclaimed and
     0.0 cross-shared.
   * **What it is:** doc 27 §1.4's fit under-prediction, now counted on the planes that were missing, plus steeper
     muons on the APA1/APA3 side.
   * **What changed:** the estimator's inputs are now the right cells; its PDHD floor is unchanged in kind. No verdict
     reads it.
5. **PDVD is not flipped.** With the key on, 6 of 596 candidates move, by at most 0.2 MeV; hand Michel medians are
   unchanged (34.35 / 2.60 → 34.35 / 2.59).

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/docs/scan/d28
export STM_SCAN_RECORD=$IMG/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json
# build (the new symbol + its doctest needs the -k install once: feedback memory), freshness, tests
cd /home/xqian/toolkit-dev/toolkit && ./wcb build --notests -p -k; ./wcb install --notests -p -k
./wcb build --notests -p && ./wcb install --notests -p && ./build/clus/wcdoctest-clus          # 400/400
cp -a ../local/lib /home/xqian/tmp/h28/libpin_h28; md5sum /home/xqian/tmp/h28/libpin_h28/*.so* > $D/libpin_md5_before.txt
# arms (runner copied at $D/run_arm28.sh; pin libpin_h28, Clus md5 30113227)
bash $D/run_arm28.sh pdhd h28off;  bash $D/run_arm28.sh pdhd h28wl  $D/tla_wl.txt
bash $D/run_arm28.sh pdvd p97voff; bash $D/run_arm28.sh pdvd p97vwl $D/tla_wl.txt
cd $D
DET=pdhd MODE=identical  ARM=h28off  BASE=h26q2dprod bash d28_gates.sh > gate_h28off.txt   # G1
DET=pdvd MODE=identical  ARM=p97voff BASE=p96vprod   bash d28_gates.sh > gate_p97voff.txt  # G2
DET=pdhd MODE=wirelookup ARM=h28wl   BASE=h28off     bash d28_gates.sh > gate_h28wl.txt    # K1 K2
DET=pdvd MODE=wirelookup ARM=p97vwl  BASE=p97voff    bash d28_gates.sh > gate_p97vwl.txt   # P1 rule
python3 d28_measure.py --vd-off NONE --vd-on NONE > measure_pdhd.txt                      # K3 K4
python3 d28_measure.py --hd-off NONE --hd-on NONE > measure_pdvd.txt                      # P1
python3 d28_spectrum.py > spectrum.txt; python3 d28_planes.py > planes.txt                # sec 4
# the flip
STAGE=pre bash d28_proofs.sh > cfg_pre.txt      # before the edit
#   (edit pdhd/wct-pr-perevt.jsonnet: michel_q2d_region_wire_lookup: true)
STAGE=post bash d28_proofs.sh > cfg_proofs.txt  # after
bash run_arm28.sh pdhd h28prod
DET=pdhd MODE=identical ARM=h28prod BASE=h28wl bash d28_gates.sh > gate_h28prod.txt       # F
```

**Gate self-test.** On doc 26's confirmed pair `h26q2dprod` vs `h26q2d` the gate passes
(`selftest_identical_pass.txt`). On `h26q2dprod` vs `h26conf`, which differ by 49 branches, it fails
(`selftest_identical_fail.txt`). Pre-registration: `preregistered.txt`, written before any arm.

## 1. The defect and the fix

**Symptom** (doc 27 §2). On PDHD the region and body-control sums lost their induction planes: 91 of 336 candidates
had no U or V cell in the control. On APA1/APA3 the recorded `d_stop_cm` of the muon's own U/V footprint cells had a
median of about 150 cm.

**Root cause.** `michel_q2d_estimate` starts every cell at `fw.front()`, the readout channel's first (face, wire) in
`AnodePlane` order, and only a Michel/gamma cloud or blob match moves it. The distances were computed on that wire. On
PDHD every U/V channel wraps onto both faces, face 0 listed first, and 348 of 800 have two segments on one face.

**Why it hid.**
* **It was PDVD work.** The estimator was developed on PDVD, where no cell lands beyond 100 cm.
* **Doc pdvd/96's repair stopped short.** It iterated every (face, wire) for `own_blob`, but not for the distance.
* **Doc 26's gates couldn't see it.** They were additive, and its twin re-derived the arithmetic from the recorded
  distances.

**Fix** (`clus/src/CheckSTM_Michel.cxx`, `clus/src/StmMichelFunctions.cxx`; toolkit `f516b013`). With
`michel_q2d_region_wire_lookup` on:
* **Distances.** Each unplaced multi-wire cell is measured on every incarnation, and
  `stm_michel_pick_wire(dist, R, use_stop, use_ctl, own_of)` chooses among them:
  * the first in-radius incarnation its own blobs cover (`own_of` is called only in radius, and stops at the first
    covered one);
  * else the nearest in-radius one;
  * else the nearest overall.
* **Own blobs.** `own_blob` is then evaluated on that (face, wire) alone, at the scope's bits (1 main, 2 unfitted
  companion, 4 fitted companion). Walking every incarnation would let a far segment's coverage vouch for a near one.
* **Placed cells.** Cells a cloud or blob match placed keep their (face, wire).
* **The new branch.** `michel_q2d_n_rewired` counts the in-radius cells moved off the first incarnation; it is
  written only with the key on.
* **Tests.** Four doctest cases pin the rule: a single incarnation; a far face listed first; coverage before distance;
  disabled centres, unprojectable distances and ties. The defaults doctest pins the key at false.

**Known edge, not exercised.** The pick is called with radius `R` when the region is on and 0 when it is off. The
legacy control test uses `R` whatever it is. The two differ only for a negative `michel_q2d_region_cm`, which no
configuration sets (both productions: 10).

## 2. Verification

| gate | comparison | result |
|---|---|---|
| build | `wcdoctest-clus` | 400/400 (`stm_michel_pick_wire` 4 cases + defaults, 287 assertions); `libWireCellClus.so` 19:26 newer than both source edits (19:22, 19:23) |
| G1 | PDHD `h28off` (key absent, new pin) vs `h26q2dprod` (old pin) | **bit-identical**: 341/341 candidates, 198 branches, every point row, every tree incl. `T_stm_michel_2d`, zips, calib, census (`gate_h28off.txt`) |
| G2 | PDVD `p97voff` vs `p96vprod` | **bit-identical**: 596/596, 120/120 events (`gate_p97voff.txt`). `039252_11` has no candidate in either (incomplete on production too) |
| K1 | PDHD `h28wl` vs `h28off` | 245/341 candidates move, only registered branches; +1 branch `michel_q2d_n_rewired`; 0 verdict changes; point rows, other trees, zips, calib and census identical (`gate_h28wl.txt`) |
| K2 | `T_stm_michel_2d` rows, PDHD | W 170 029 identical; U/V (all multi-wire on PDHD): 177 169 identical, 94 199 placement moved, 85 117 role-0 rows added, 0 charge columns moved |
| P1 rule | PDVD `p97vwl` vs `p97voff` | 10/596 candidates move registered branches; 0 verdict changes (`gate_p97vwl.txt`) |
| P1 rows | `T_stm_michel_2d` rows, PDVD | W 270 470 and single-wire U/V 354 620 identical; multi-wire U/V 57 230 identical, 54 891 placement moved, 9 role-0 rows added |
| F proofs | compiled configs (`cfg_pre.txt`, `cfg_proofs.txt`) | pre: h28cfg0 → h28cfgT +1 key. A: h28cfgT → h28cfg 0/0/0, whole config identical after the tag rename. B: key at its initializer (false). C: +1 key. D: PDVD file unchanged. Production diff: one line |
| F confirm | PDHD `h28prod` (flipped file, no TLA) vs `h28wl` | **bit-identical** (`gate_h28prod.txt`) |

Every arm ran on `libpin_h28` (md5 `30113227` before and after each arm, no loader deaths). The manifest
`libpin_md5_before.txt` still matches the pin.

## 3. What the lookup changes (`measure_pdhd.txt`, `measure_pdvd.txt`, `spectrum.txt`)

**The distances.** Role-1 U/V cells, recorded `d_stop_cm` beyond 100 cm, by the channel's wires on its APA's active
face:

| class (PDHD) | knob off: n, beyond 100 cm | knob on: n, beyond 100 cm |
|---|---|---|
| one active-face wire, recorded on the other face | 38 727, 0.740 | 358, 0.000 |
| one active-face wire, recorded on the active face | 51 056, 0.000 | 89 425, 0.000 |
| two active-face wires, recorded on the other face | 30 383, 0.522 | 176, 0.000 |
| two active-face wires, recorded on the active face | 38 359, 0.492 | 68 566, 0.000 |

The few cells still recorded on the other face are ones a cloud match placed there.

**Scale.**
* **Controls with no U and no V cell:** 91 → 6 of 341.
* **`michel_q2d_n_rewired` > 0:** on all 56 APA1 and all 80 APA3 candidates (median 818 / 839 cells), 51 of 91 on
  APA2 (median 36), and 53 of 109 on APA0.

**The region sum, per plane, PDHD hand Michel (44), median.**

| | U | V | W | dropped plane (none / U / V / W) |
|---|---|---|---|---|
| region sum, off (MeV-eq) | 39.1 (153 cells) | 39.1 (149) | 46.9 (211) | 13 / 8 / 8 / 15 |
| region sum, on | 56.3 (262) | 54.8 (273) | 46.9 (211) | 9 / 16 / 15 / 4 |
| control sum, off → on | 0.0 → 15.9 | 0.0 → 15.2 | 11.9 → 11.9 | 15 / 12 / 9 / 8 → 2 / 15 / 15 / 12 |

The collection plane does not move. The restored induction planes now outweigh it, and the plane rule, which used to
drop W, now mostly drops U or V.
* **APA2:** region 37.4 → 37.4, control 6.3 → 7.0 (n 13).
* **APA1/APA3:** region 40.2 → 55.3, control 14.2 → 13.8 (n 31).
* **Per item:** the region shift p10/p50/p90 is +0.0 / +2.9 / +23.0 MeV; 8 of 44 are unchanged.

**Against doc 27's offline bounds (K4).**
* **Region median 45.3:** MISSED, just above the registered [38.9, 44.0] ± 1.
* **Control 11.1:** HELD, at the edge of [7.8, 10.2] ± 1.
* **Per candidate:** 121 of 140 values (both estimators) sit within [fixlo − 2, fix + 2] MeV, 0.864, HELD.

The miss is in the direction doc 27 §7 named. The offline correction could not add the role-0 cells the chain never
wrote; the C++ adds 85 117 of them, which doubles `michel_q2d_n_role0` from 310 to 655 at the hand-Michel median.

## 4. The corrected spectrum, and why it is not leakage (`spectrum.txt`, `planes.txt`)

| hand Michel items | n | region median [68 %] | p90 | above 52.8 MeV | control median |
|---|---|---|---|---|---|
| PDHD, knob off (was production) | 44 | 39.5 [37.4, 42.6] | 59.8 | 12 (0.273) | 8.5 |
| **PDHD, knob on (production now)** | 44 | **45.3 [42.7, 54.4]** | 70.2 | **20 (0.455)** | **11.1** |
| PDVD `p96vprod` | 134 | 34.3 [32.1, 35.8] | 51.5 | 12 (0.090) | 2.6 |

PDHD on against PDVD: KS p 0.0001, Mann-Whitney p 0.0001. `michel_ke_best` is untouched (25.3).

**Is the added charge real, or charge from another segment of a wrapped channel?**

| region sum per plane, median MeV-eq | n | U / V / W | per-item U/W, V/W | U+V by role: claimed / footprint / unclaimed / cross-shared |
|---|---|---|---|---|
| PDHD APA2, off | 13 | 40.7 / 42.2 / 37.1 | 1.05, 1.14 | 72.5 / 14.1 / 0.0 / 0.0 |
| PDHD APA2, on | 13 | 40.7 / 48.8 / 37.1 | 1.14, 1.15 | 72.5 / 15.0 / 0.0 / 0.0 |
| PDHD APA1/3, off | 31 | 37.2 / 37.9 / 54.5 | 0.88, 0.75 | 78.7 / 0.0 / 0.0 / 0.0 |
| PDHD APA1/3, on | 31 | 62.2 / 56.5 / 54.5 | **1.19, 1.05** | 78.3 / 21.3 / 1.5 / 0.0 |
| PDVD `p96vprod` (and `p97vwl`, identical) | 134 | 40.0 / 39.8 / 34.3 | **1.10, 1.11** | 61.3 / 10.8 / 0.4 / 0.0 |

* **The ratios agree.** On APA1/APA3 the corrected induction-to-collection ratios land where APA2's already were,
  and where PDVD's always were: induction planes read about 10–20 % above collection on both detectors.
* **The U/V gain is consistent with the muon's own footprint coming back inside the radius.**
  * **Footprint:** the per-item median moves 0.0 → 21.3 MeV-eq.
  * **The other roles barely move:** Michel-claimed cells stay flat (78.7 → 78.3), unclaimed charge adds a median
    1.5 and cross-shared none.

  These are medians over items, not a decomposition of the plane sums. They fit doc 27 §1.4's under-prediction by
  the muon fit (U/V measured/predicted 1.19–1.24 on PDHD) being counted again on the planes that had been dropped.
* **APA1/APA3's W sum is higher too** (54.5 against 37.1) and does not move with the fix. Those muons are steeper, and
  PDHD's vertical collection wires foreshorten them (doc 27 §1.2).
* **Conclusion.** The fix gives the estimator the right cells. It does not remove its PDHD floor, and was not meant
  to. A PDHD Michel energy closer to the endpoint needs item 2 or 3 of §7.

## 5. The pre-registration, graded

| id | prediction | outcome |
|---|---|---|
| G1 | `h28off` == `h26q2dprod` on everything | **HELD** |
| G2 | `p97voff` == `p96vprod` on everything | **HELD** |
| K1 | only the registered branches move, +1 new, 0 verdicts, rest identical | **HELD** (PDHD 245/341) |
| K2 | W and single-wire U/V rows identical; added/removed rows role 0 only | **HELD** (PDHD has no single-wire U/V; PDVD 354 620 identical) |
| K3 | `n_rewired` > 0 on every APA1/APA3 `is_stm` Michel with U/V cells; beyond 100 cm < 0.05 | **HELD** (all 136 APA1/APA3 candidates; 0.000) |
| K4 | region median in [38.9, 44.0] ± 1 | **MISSED** (45.3) |
| K4 | control median in [7.8, 10.2] ± 1 | **HELD** (11.1) |
| K4 | ≥ 80 % of per-candidate values within [fixlo − 2, fix + 2] | **HELD** (0.864) |
| P1 | PDVD candidates moved < 5 % | **HELD** (6/596, 0.010) |
| F | 1-key compiled diff, 0/0/0 vs the TLA config, PDVD unchanged, `h28prod` == `h28wl` | **HELD** |

## 6. What is NOT concluded

* **Not a better PDHD Michel energy scale.** The region estimator is now geometrically right on PDHD. It reads higher
  than before and further from PDVD, because the planes it recovered carry the shared fit's under-prediction. No
  energy truth exists.
* **Not a change to any selection.** Every verdict, reject bit and census is identical.
* **Not a PDVD change.** PDVD production stays off. With the key on it moves 6 candidates by at most 0.2 MeV; turning
  it on there would be a separate owner decision.
* **Not a validation of `stm_michel_pick_wire`'s coverage-first rule against truth.** The negative controls show it
  leaves everything else untouched; the choice among in-radius incarnations rests on own-blob coverage.

## 7. Next, ranked

1. **Refresh doc 26 §3's PDHD spectrum on `h28prod`** when the comparison is next quoted. Doc 26's numbers are
   production as of the previous flip.
2. **A control that cannot overlap the stop** (doc 27 §8 item 2), now on correct distances.
3. **The fit's prediction bias** (doc pdvd/96 §5, doc 27 §8 item 3). It now accounts for most of the PDHD floor on all
   three planes.
4. **The same fix on PDVD**, only if the owner wants the two detectors on one lookup: cosmetic there (6/596, ≤ 0.2 MeV).

## Files

| file | what |
|---|---|
| toolkit `f516b013` | `CheckSTM_Michel.cxx` (knob, pick block, own-blob helper, `michel_q2d_n_rewired`), `StmMichelFunctions.{h,cxx}` (`stm_michel_pick_wire`), `doctest_stm_michel_q2d.cxx`, `doctest_check_stm_michel_defaults.cxx` |
| `pdhd/wct-pr-perevt.jsonnet` | the one key, at the end of the region block, with the gate labels |
| `scan/d28/preregistered.txt` | written before any arm |
| `scan/d28/run_arm28.sh`, `tla_wl.txt`, `DONE_*`, `libpin_md5_before.txt` | arms and pin provenance |
| `scan/d28/d28_gates.sh` → `gate_*.txt`, `selftest_identical_{pass,fail}.txt` | G1 G2 K1 K2 P1 F |
| `scan/d28/d28_proofs.sh` → `cfg_pre.txt`, `cfg_proofs.txt` | compiled-config proofs |
| `scan/d28/d28_measure.py` → `measure_pdhd.txt`, `measure_pdvd.txt` | K3 K4 P1 |
| `scan/d28/d28_spectrum.py` → `spectrum.txt`; `d28_planes.py` → `planes.txt` | §3–4 |

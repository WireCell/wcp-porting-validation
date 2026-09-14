# 100 — The PDVD gain round: the owner's look at the frame-edge tags, the C refit, the two data-sized Michel thresholds, QtoL measured; round 2: the Michel movers and the readout-edge exemption

**Status (2026-09-13, round 2, §7): the gain-flip criterion PASSES on the owner-corrected record, and the gain bundle is FLIPPED into production (owner yes).**
- The owner judged all 65 chain movers between production and the candidate (`own100m`). On the corrected record Michel purity is
  0.938 (`p99wflip`) → 0.924 (`p100c`), −0.48σ; is_stm purity 0.965 → 0.963; both efficiencies rise (§7.2). The selection
  check (items no scan corrected) gives the same answer, −0.26σ (§7.3).
- Round 1's 3σ Michel drop was mostly the record: of its 21 "new false Michels" the owner calls 13 real Michels.
- **FLIPPED (§7.5, owner yes 2026-09-13):** production is now SP top gain 0.889 (imaging tag `pvdimg`) + C 0.8630.
- The readout-edge guard's Michel exemption is built default OFF (two knobs, toolkit `1e2447f2`) and measured: on the
  candidate it adds exactly the 9 predicted tags and removes none; the owner judged them 7 stoppers / 2 THRU, below the
  pre-registered bar, so it stays OFF (§7.4).

**Round 1 status (2026-09-13): NOT FLIPPED — stopped at the pre-registered criterion.**
- Production is unchanged: SP top gain OFF, C 0.7941, thresholds as before, QtoL 0.094.
- Built: a default-legacy `stm_recomb_C` knob (toolkit `6ac5fffc`). The compiled config is byte-identical at the default (G1).
- C refit measured: C = 0.8630 closes on the gain-ON arm, and the refit alone is purity-neutral.
- The two data-sized thresholds have nothing to refit.
- The round's pre-registered Michel-purity criterion fails as written, against a comparison confounded by the SP wire order (§4).
- The owner's look at the 19 frame-edge objects (§1) finds that the readout-edge guard, shipped in production with the real window (wcp f52a374e), removes about as many real stoppers as through-going tracks.

Owner, 2026-09-13: *"the gain round: flip top_gain_scale=0.889 together with the C refit (0.8630) and the Michel-threshold refits, gated against p99wflip. Before or alongside it, look at run 039349's removed tags … put it in display port 5017"*; *"we do not need blind scan"*.
Owner answers, the same day, on scope:
- Michel thresholds: **the two data-sized knobs only** (`topology_michel_ke_min`, `plateau_mip_hi`);
- QtoL: **measure, don't flip**.

Pre-registration: `d100/prereg.md`, written after the C measurement on `p99rwon` and before any refit arm ran.

## Repro
```bash
# run from pdvd/docs/nf_sp_img_clus unless noted; scratch /home/xqian/tmp/p100; S=scripts
# sec 1: the owner's look (payloads: doc 99 sec 6.4 preps in /home/xqian/tmp/p99scan, symlinked, no prep run)
python3 $S/d100_edge_scan_set.py --out /home/xqian/tmp/p100/scan
(cd ../../stm_michel_scan && ./serve_stm_michel_scan.sh 5017 --scan-tag own100 --manifest /home/xqian/tmp/p100/scan/manifest.tsv \
    --prepdir /home/xqian/tmp/p100/scan/prep --questions /home/xqian/tmp/p100/scan/questions.json --dead-points)
python3 $S/d100_edge_scan_score.py --set /home/xqian/tmp/p100/scan --record-out ../scan/pdvd_stm_michel_own100_verdicts.json > d100/edge_scan.txt
# sec 2: G1 (wcsonnet of pdvd/wct-pr-perevt.jsonnet with 039349_7_p99rwon's runner TLAs, before / after / -S stm_recomb_C=0.8630) -> d100/g1_compiled_config.txt
# sec 3: C
(cd ../../.. && for A in p99wflip p99rwon; do python3 pdhd/docs/scripts/d16_stm_energy_scales.py --det pdvd --arm "pdvd/work/*_$A" --chain-C 0.7941 --out /home/xqian/tmp/p100/cfit/c_refit_$A; done)   # d100/c_refit_$A.txt
WAVE=pr ARM=p100c SRC=p99rwon PR_TLA="-S stm_recomb_C=0.8630" $S/d100_arms.sh                                   # d100/arm_p100c.log
(cd ../../.. && python3 pdhd/docs/scripts/d16_stm_energy_scales.py --det pdvd --arm "pdvd/work/*_p100c" --chain-C 0.8630 --out /home/xqian/tmp/p100/cfit/c_refit_p100c)   # d100/c_refit_p100c.txt
python3 $S/d99rw_census.py --arms p99wflip p99rwon p100c > d100/census_p99wflip_p99rwon_p100c.txt
# records carried by geometry, verdicts only (L = ../scan/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json, X = the smx1a..smx9 record)
python3 $S/d100_carry_verdicts.py --record L --base-arm p98vonq  --arm p99rwon  --out /home/xqian/tmp/p100/carry/latest_on_p99rwon.json   # d100/carry_latest_on_p99rwon.txt
python3 $S/d100_carry_verdicts.py --record L --base-arm p98vonq  --arm p99wflip --out /home/xqian/tmp/p100/carry/latest_on_p99wflip.json  # d100/carry_latest_on_p99wflip.txt
python3 $S/d100_carry_verdicts.py --record X --base-arm p96vprod --arm p99wflip --out /home/xqian/tmp/p100/carry/smx_on_p99wflip.json     # d100/carry_smx_on_p99wflip.txt
python3 $S/d100_carry_verdicts.py --record X --base-arm p96vprod --arm p99rwon  --out /home/xqian/tmp/p100/carry/smx_on_p99rwon.json      # d100/carry_smx_on_p99rwon.txt
python3 $S/d99_grade.py --arms p99rwon:<latest_on_p99rwon> p100c:<latest_on_p99rwon> --movers p99rwon,p100c > d100/grade_p99rwon_p100c_latest.txt
python3 $S/d99_grade.py --arms p99wflip:<latest_on_p99wflip> p100c:<latest_on_p99rwon> --movers p99wflip,p100c > d100/grade_p99wflip_p100c_latest_record.txt
python3 $S/d99_grade.py --arms p99wflip:<smx_on_p99wflip> p100c:<smx_on_p99rwon> --movers p99wflip,p100c > d100/grade_p99wflip_p100c_smx_record.txt
# sec 3.3: the two data-sized thresholds
(cd ../../../pdhd/stm_michel_scan && ./prep_stm_michel_scan.py --det pdvd --arm p100c --redraw --outdir /home/xqian/tmp/p100/prep_p100c --sheetdir /home/xqian/tmp/p100/sheet_p100c)
python3 $S/d100_twin.py --prep /home/xqian/tmp/p100/prep_p100c --record /home/xqian/tmp/p100/carry/latest_on_p99rwon.json --arm p100c > d100/twin_p100c.txt
python3 $S/d100_plateau_michel_diag.py > d100/twin_plateau_binding_and_michel_fp.txt
# sec 5: QtoL, measured only
(cd ../../ql_light_calib && python3 fit_qtol_crossers.py --tag $A) > d100/qtol_crossers_$A.txt          # A in p99wflip p99rwon
python3 $S/d100_qtol_halves.py --tag p99wflip --tag p99rwon > d100/qtol_halves.txt
# ---- round 2 (sec 7; prereg d100/prereg_round2.md) ----
# sec 7.1-7.3: own100m, the owner's look at every chain mover p99wflip -> p100c, and the re-grade on the corrected record
python3 $S/d100_michel_scan_set.py --out /home/xqian/tmp/p100/mscan                              # d100/own100m_items.tsv
(cd ../../stm_michel_scan && ./serve_stm_michel_scan.sh 5017 --scan-tag own100m --manifest /home/xqian/tmp/p100/mscan/manifest.tsv \
    --prepdir /home/xqian/tmp/p100/mscan/prep --questions /home/xqian/tmp/p100/mscan/questions.json --dead-points)
python3 $S/d100_michel_scan_score.py --set /home/xqian/tmp/p100/mscan --record-out ../scan/pdvd_stm_michel_own100m_verdicts.json \
    --carry-out /home/xqian/tmp/p100/carry_r2 > d100/michel_scan.txt
python3 $S/d100_michel_scan_subsets.py > d100/michel_scan_subsets.txt                           # the selection check
# sec 7.4: the readout-edge exemption.  Toolkit knobs TaggerCheckSTM readout_edge_defer + CheckSTM_Michel
# readout_edge_require_michel (R_READOUT_EDGE); wcbuild; ./build/clus/wcdoctest-clus; pin /home/xqian/tmp/p100/libpin_p100b
# X0: the G1 wcsonnet recipe with -S stm_readout_edge_defer=true [-S 'stm_michel_extra={readout_edge_require_michel:true}']
setsid nohup bash $S/d100r2_arms.sh > /home/xqian/tmp/p100/r2_arms.log 2>&1 < /dev/null &   # p100boff p100bg p100bd p100bx p100bxp h100a h100b
python3 $S/d99rw_identity.py --arm p100boff --base p100c --nt all > d100/x2_identity_p100boff_p100c.txt
python3 $S/d99rw_identity.py --arm p100bd --base p100bg --nt all > d100/x4_identity_p100bd_p100bg.txt
python3 $S/d100_pdhd_identity.py --arm h100b --base h100a > d100/x3_pdhd_identity_h100b_h100a.txt
python3 $S/d100r2_exempt.py --arm p100bx --base p100c --control p100bg --defer p100bd --side latest \
    --carried /home/xqian/tmp/p100/carry/latest_on_p99rwon.json --unjudged-out /home/xqian/tmp/p100/exempt_unjudged_p100bx.tsv > d100/exempt_p100bx.txt
python3 $S/d100r2_exempt.py --arm p100bxp --base p99wflip --side production \
    --carried /home/xqian/tmp/p100/carry/latest_on_p99wflip.json --unjudged-out /home/xqian/tmp/p100/exempt_unjudged_p100bxp.tsv > d100/exempt_p100bxp.txt
python3 $S/d100_pdhd_identity.py --arm h100b --base h28prod > d100/x3b_pdhd_identity_h100b_h28prod.txt   # X3 attribution
(cd ../../../pdhd/stm_michel_scan && for A in p100bx p100bxp; do ./prep_stm_michel_scan.py --det pdvd --arm $A --redraw \
    --outdir /home/xqian/tmp/p100/prep_$A --sheetdir /home/xqian/tmp/p100/sheet_$A; done)
python3 $S/d100_exempt_scan_set.py --out /home/xqian/tmp/p100/xscan \
    --arm p100bx /home/xqian/tmp/p100/exempt_unjudged_p100bx.tsv /home/xqian/tmp/p100/prep_p100bx \
    --arm p100bxp /home/xqian/tmp/p100/exempt_unjudged_p100bxp.tsv /home/xqian/tmp/p100/prep_p100bxp
(cd ../../stm_michel_scan && ./serve_stm_michel_scan.sh 5017 --scan-tag own100x --manifest /home/xqian/tmp/p100/xscan/manifest.tsv \
    --prepdir /home/xqian/tmp/p100/xscan/prep --questions /home/xqian/tmp/p100/xscan/questions.json --dead-points)
python3 $S/d100_exempt_scan_score.py --set /home/xqian/tmp/p100/xscan --record-out ../scan/pdvd_stm_michel_own100x_verdicts.json > d100/exempt_scan.txt
# sec 7.5: the flip.  Compiled-config proof -> d100/flip_compiled_config.txt (SP entry and PR driver, before/after)
(cd ../../work && for s in *_p98von; do e=${s%_p98von}; mkdir ${e}_pvdimg; ln $s/clusters-apa-anode*-ms-*.tar.gz $s/img-provenance.txt ${e}_pvdimg/; done)
(cd ../../work && ls -d *_pvdimg | xargs -I{} sh -c 'ls {}/*' | xargs -P 16 -n 8 sha256sum | sort -k2) > d100/pvdimg_sha256_manifest.txt
PIN=/home/xqian/tmp/p100/libpin_p100b WAVE=flip ARM=p100flip $S/d100_arms.sh                     # F1: production, no overrides
python3 $S/d99rw_identity.py --arm p100flip --base p100c --nt all > d100/f1_identity_p100flip_p100c.txt
(cd ../.. && setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh -O _p100spflip 039252 0)               # F2: SP at the new default
python3 ../../../abtest/hash_archive.py ...  # F2 comparison against work/039252_0_p98von -> d100/f2_sp_frames_p100spflip_p98von.txt
```

## 1. The owner's look at the 19 frame-edge objects (`own100`, not blind)

**Why.** Production has used each run's real readout window since wcp f52a374e (doc 99 §4.5). On run 039349 (6400-tick frames) PR's `readout_edge_guard` untags any stop in the last 60 ticks.
- It removed **11** is_stm tags on production (`p96vprod` → `p99wflip`) and **14** on the latest configuration (`p98vonq` → `p99rwon`). Nothing entered the other way.
- Doc 99 said the removals were "mostly unjudged". That is true of the swap scan, but the smx record already judged 10 of production's 11, 8 of them as stoppers.
- Matched by geometry, the two lists are **19 objects**.

**Set** (`scripts/d100_edge_scan_set.py`, `d100/edge_scan_items.tsv`).
- Each object is shown on the arm that still tagged it: `p98vonq` for the 14 on the latest list (6 of them shared with production's), `p96vprod` for the 5 only on production's.
- Each item's question panel carries the guard's own log line (stop tick, ticks before the frame end) and every existing record verdict for either key.
- Tag `own100`, served on :5017. The other 9 label tags' sha256 are unchanged after the session.

**Result** (`scripts/d100_edge_scan_score.py` → `d100/edge_scan.txt`; record `../scan/pdvd_stm_michel_own100_verdicts.json`, 25 items, one per arm key):

| | objects | stoppers (owner) | not stoppers (owner THRU) |
|---|---|---|---|
| all | 19 | **12** (STM_ONLY 7, STM_MICHEL 5) | **7** |
| production's list (the change shipped) | 11 | **6 lost** | 5 removed correctly |
| the latest configuration's list | 14 | 9 | 5 |
| top / bottom | 12 / 7 | 9 / 3 | 3 / 4 |

**The guard window cannot be tuned to separate them** (§4 of the record). Of the 18 objects the guard removed (039349_32/48 left through a flash-match change instead):
- **Ticks before the frame end.** Stoppers sit at 12.1–55.5, non-stoppers at 20.1–53.8, fully interleaved.
- **Shrinking the window.** A 30-tick guard would still remove 3 stoppers and 2 non-stoppers; a 50-tick guard 10 and 5. So `stm_readout_edge_ticks` is not the lever.

**The chain's own Michel separates them** (§5 of the record):
- **Chain found a Michel:** 4 objects, all 4 owner stoppers (039349_40/48, 48/21, 67/35, 81/41), 0 non-stoppers.
- **No Michel found:** 14 objects, 8 stoppers and 6 non-stoppers.
- A found Michel at the stop is evidence, independent of the edge, that the muon stopped inside the frame. This is 4/4 on 18 objects: a candidate for a guard exemption, not a result. It is not changed here.

**The owner against the existing records** (§3 of the record):
- **Changed class:**
  - 039349_22/51 and 039349_26/23: THRU → STM_ONLY;
  - 039349_40/48: UNCLEAR → STM_MICHEL, pin at rr 3.6 cm;
  - 039349_69/44: UNCLEAR → STM_ONLY;
  - 039349_64/65: STM_MICHEL (low) → THRU;
  - 039349_82/52: STM_ONLY (high, twice) → THRU;
  - 039349_18/46: unjudged → STM_ONLY.
- **Changed Michel kind:** 039349_67/35 and 039349_75/63, STM_ONLY → STM_MICHEL.
- **Owner over owner — one re-judgement.**
  - `039349_72/11` (production key; `039349_72/9` on the latest arm) was STM_MICHEL in **smx5 (owner, 2026-09-11)** and is **THRU in own100 (owner, 2026-09-13)**.
  - The earlier owner rulings on 039349_48/21 (STM_MICHEL, smx3 2026-09-10) and 039349_76/25 (THRU, smx3 2026-09-10) are unchanged.
  - The own100 record carries both dates; the smx record is untouched (M13).

**What it means for production today.** The real window is correct, but on run 039349 its edge guard costs 6 of production's stoppers to remove 5 non-stoppers. The decision this feeds is the guard rule, not the window.

## 2. The knob (G1)

- toolkit `cfg/pgrapher/experiment/protodunevd/pr.jsonnet`: new arg `stm_recomb_C=0.7941`, used as `pdvd_stm_recomb.data.C`.
- `pdvd/wct-pr-perevt.jsonnet`: top-level `stm_recomb_C = 0.7941`, forwarded, so `PDVD_PR_TLA="-S stm_recomb_C=…"` sets it.
- **Gate** (`d100/g1_compiled_config.txt`): the compiled PR config (the runner's TLAs for `039349_7_p99rwon`, pipeline `-nu -stm-fit`) md5 is 447e0f65 before and 447e0f65 after at the default (**identical**). With `-S stm_recomb_C=0.8630` exactly one leaf differs: `PowerBoxRecombination pdvd_stm_recomb data.C` 0.7941 → 0.863.
- No C++ changed. `clus/test/doctest_stm_michel.cxx`'s literal 0.7941 is a unit fixture and stays.

## 3. The C refit and the two data-sized thresholds

### 3.1 C on the pre-refit candidate, with production as the null
`d16_stm_energy_scales.py --chain-C 0.7941` (`d100/c_refit_*.txt`):

| arm | fit C | chain reproduction | E_dQdx/E_range median at the fit, bottom / top |
|---|---|---|---|
| `p99wflip` (production, null) | 0.7923 ± 0.0051 (254 tracks) | 1.0017 | 1.1089 / 0.9716 (bottom/top 1.141) |
| `p99rwon` (gain ON, real window) | **0.8630 ± 0.0075** (259) | 1.0023 | 1.0389 / 0.9946 (1.045) |

- The null holds: production refits 0.35σ from the chain's 0.7941.
- On production, bottom stoppers read 11 % high because C was fitted mostly on low-reading top tracks. With the gain on, one C fits both volumes.

### 3.2 The refit arm `p100c`
PR only on `p99rwon`'s pctree with `-S stm_recomb_C=0.8630` (`scripts/d100_arms.sh WAVE=pr`, pin `libpin_p96` Clus 4e1db810 at start and end, `setarch -R`).

**Completion: 119/120.** `039349_30` has no candidate: the real window empties it on the latest arms (doc 99 §4.4), so PR prints no candidate line. Not an error.

**Closure, pre-registered** (`d100/c_refit_p100c.txt`, `--chain-C 0.8630`) — **all pass**:
- **C1:** the chain reproduction is 1.0024 (≤ 1.005).
- **C2:** the fit is 0.8630 ± 0.0075, the same as the candidate's.
- **C3:** bottom/top is 1.0389 / 0.9946 = 1.045 (< 5 %; 1.141 on production).

**What C moves in the tagger: almost nothing.**
- Census (`d100/census_p99wflip_p99rwon_p100c.txt`): `p100c` equals `p99rwon` in every count — is_stm 259, with Michel 160, candidates 540, guard firings 440.
- On the same record (`d100/grade_p99rwon_p100c_latest.txt`): 0 is_stm movers and 1 `michel_found` mover (one Michel FP fewer). Michel purity is 0.819 → 0.824, is_stm unchanged at 223/18/36 (purity 0.925, efficiency 0.861).
- C feeds `check_stm_michel`'s energies. Only its MeV cuts can respond, and on this sample they don't.

### 3.3 `topology_michel_ke_min` and `plateau_mip_hi` (doc 89's 0-FP rule, re-run)
`scripts/d100_twin.py` (fork of `d90_twin.py`, both directions) on `p100c`'s 533 candidate payloads, graded on the latest record carried to `p99rwon` keys (`d100/twin_p100c.txt`).
- **Twin reproduction:** at the arm's own 3.0 / 2.0 the twin reproduces `p100c` exactly (0 mismatches).
- **FN count:** it includes judged items with no candidate on the arm.
- **`topology_michel_ke_min`: live, nothing to gain.**
  - 3.5 or 4.0 MeV loses one stopper (039349_65/43).
  - 5.0 loses two stoppers and one non-stopper.
  - 2.0 and 2.5 move nothing.
  - By the rule, **keep 3.0**.
- **`plateau_mip_hi`: does not bind on this arm.**
  - Across 1.6–2.5 not one candidate moves.
  - `d100/twin_plateau_binding_and_michel_fp.txt` (a): 2 of 499 Bragg-valid candidates have plateau/MIP > 1.6, 1 > 1.8, maximum 1.998.
  - Every one of the 44 plateau rejections is the lower edge (< 0.6, `plateau_mip_lo`, not in this round's scope).
  - **"Keep 2.0" is vacuous, not a sizing.**
- **No threshold changes; no `p100thr` arm was needed.** The final candidate is `p100c`.

## 4. The pre-registered round criterion: FAILS as written, against a confounded comparison

Pre-registered (`d100/prereg.md` §2):
- Michel purity on the carried record returns to within 1.5σ of `p99wflip`'s.
- If it does not, the "thresholds fitted on the old scale" reading is wrong: report, don't retune.

**Same record on both arms** (carried by geometry, verdicts only; `d100/grade_p99wflip_p100c_*_record.txt`):

| record | arm (items carried) | is_stm TP/FP/FN, purity, eff | Michel (all judged) TP/FP/FN, purity, eff |
|---|---|---|---|
| latest (`p98vonq`-keyed) | `p99wflip` (613) | 224/5/25, 0.978, 0.900 | 136/9/14, **0.938**, 0.907 |
| latest | `p100c` (635) | 223/18/36, 0.925, 0.861 | 131/28/26, **0.824**, 0.834 |
| smx (`p96vprod`-keyed) | `p99wflip` (600) | 234/6/28, 0.975, 0.893 | 141/10/16, **0.934**, 0.898 |
| smx | `p100c` (555) | 176/12/36, 0.936, 0.830 | 102/25/23, **0.803**, 0.816 |

**Verdict: the criterion fails** (0.824 against 0.938, about 3σ). No threshold was retuned. Three things stand between this and "the gain costs Michel purity":

1. **The comparison mixes four changes.** `p99wflip` → `p100c` is gain + SP wire order v5 → v7 + (on 039349) the latest configuration's clustering + C.
   - Doc 99 §6 split the first two on the carried smx record: Michel purity 0.923 → 0.831 for the wire order alone, then 0.831 → 0.785 for the gain alone.
   - The C refit alone is neutral (§3.2).
2. **Both records were drawn from production candidates.**
   - The smx record was scanned with the chain's answer on screen, which under-calls what production did not tag (doc 99 §6.4 found 6/6 disagreements one way).
   - The owner's look in §1 reversed record classes on 7 of 19 objects, in both directions.
   - For is_stm, doc 99 §6.4's blind scan of both sides of the swap, the fair instrument, read the difference as purity-neutral (−0.061 [−0.174, +0.060]; −0.040 with the real window).
   - No such measurement exists for the Michel.
3. **The movers are top-heavy** (`michel_found` movers, latest record):

   | | new Michel FP (0 → 1) | Michel FP removed (1 → 0) | hand Michels gained | hand Michels lost |
   |---|---|---|---|---|
   | top | 17 (STM_ONLY 9, THRU 8) | 2 | 5 | 13 |
   | bottom | 4 | 1 | 1 | 3 |

   - **Size of the new top false Michels** (`d100/twin_plateau_binding_and_michel_fp.txt` (b)): they are small — `michel_ke_best` median 5.3 MeV [2.1, 17.4], length 5.5 cm; the 90 top true Michels have 23.2 MeV [12.8, 40.2], 7.7 cm.
   - **The admission window does not cut them:** their `michel_mip` sits inside it (median 0.55).
   - **Consistent with** small pieces crossing absolute charge thresholds after the ×1.125 (doc 99 §6.2: control-region energy ×1.43 on top), and equally with record under-calls. Not separated here.

**So the flip is not made.** The criterion named a comparison it cannot answer cleanly, and the Michel cost of the package is not yet known.

## 5. QtoL (measured only; production keeps 0.094)

`fit_qtol_crossers.py` (`d100/qtol_crossers_*.txt`) and its per-half fork `scripts/d100_qtol_halves.py` (`d100/qtol_halves.txt`):

| | anchors | pooled Σmeas/Σpred → QtoL | a_b (bottom half) | a_t (top half) |
|---|---|---|---|---|
| `p99wflip` | 207 | 0.893 → 0.0839 | 0.692 | 0.809 |
| `p99rwon` | 192 | 0.833 → 0.0783 | 0.673 | 0.746 |
| ratio | | 0.933 | 0.973 (predicted 1.000) | 0.921 (predicted 0.889) |

- **Directionally as predicted, but not a clean separation.** The top half drops more than the bottom.
- **Noise.** The per-anchor fits scatter from 0 to about 1.9, and the anchor sets are not the same crossers.
- **What production's own anchors say.** They already read 0.893, so 0.094 is about 11 % high on production itself: a calibration question independent of the gain.

## 6. Not concluded / next (round 1; items 2 and 3 were taken in the same round, §7)

1. **The flip decision (owner).** Measured: C refit closes (§3.2); the two data-sized thresholds have nothing to refit (§3.3); QtoL is measured (§5). Not measured: whether the Michel purity cost against production (§4) is real.
2. **Recommended next: the owner's look at the Michel movers on the gain-ON display.**
   - Take the 21 new Michel FPs and the 16 lost hand Michels from §4's table, on `p100c` with the production item beside it, in a fresh tag.
   - It answers whether the Michel drop is real or a record under-call, which is what the flip decision is waiting on.
   - Optional, to split cause from effect: a gain-OFF, v7-wire, real-window arm (`p99rwoff`, from `p98voff`'s complete imaging; clustering + PR only).
3. **The readout-edge guard (§1).** A default-OFF knob exempting a candidate whose chain found a Michel (4/4 stoppers, 0 non-stoppers here) is the next guard round, sized on a sample it was not chosen from. Shrinking the tick window is ruled out by §1.
4. **Doc 99 §10 item 3's re-scan list** on the arm eventually adopted.
5. **The flip itself, when decided:**
   - SP entry default 0.889;
   - a production imaging tag of its own (hard links of `p98von`'s imaging, never the study arm's name);
   - driver `stm_recomb_C` 0.8630;
   - gate: a production-staged arm with no overrides equal to `p100c` on 120/120 (`d100_arms.sh WAVE=flip`, written, not run).

## 7. Round 2 (same day): the owner's look at the Michel movers, and the readout-edge exemption

Owner, 2026-09-13, after §6: *"Let's go with your recommendation, I am happy to scan in display port 5017. I feel that we do
not need to have a separate round, but can do it in the same round for this round."* Pre-registration
`d100/prereg_round2.md`, written before `own100m` was served and before any round-2 arm ran.

### 7.1 `own100m` — every chain mover between production and the candidate (not blind)
- **Set** (`scripts/d100_michel_scan_set.py`, `d100/own100m_items.tsv`): every record object whose chain answer differs
  between `p99wflip` and `p100c` on the latest carried record, joined by original key as `d99_grade.py` lists movers.
  Tranche 1: the 46 `michel_found` movers, both directions. Tranche 2: the 19 is_stm-only movers. Shown on `p100c` with both
  chain answers and the record verdict in the question.
- **Labelled 65/65**: STM_MICHEL 42, STM_ONLY 16, THRU 4, MESSY 2, UNCLEAR 1. The other ten tags' `labels.json` sha256 are
  unchanged (`d100/label_shas_r2.txt`). Record `../scan/pdvd_stm_michel_own100m_verdicts.json`, one item per arm key (130).
- **The owner against the record** (`d100/own100m_counts.txt`): unchanged 35, Michel kind changed 14, class changed 16.
  Nine of the changed items carried an earlier **owner** verdict (drawn on production's display), in both directions:
  039349_32/63, 44/20, 77/45 and 039252_8/96 THRU → STM_MICHEL; 039349_22/59 STM_ONLY → STM_MICHEL; 039349_63/44 THRU → MESSY;
  039349_81/55 STM_ONLY → MESSY; 039349_70/58 and 039349_39/56 STM_MICHEL → THRU. With own100's 039349_72/11 (§1), a single
  hand verdict near the boundary is not a fixed point.

Tranche 1 by mover direction (descriptive only; the set is selected by the metric in dispute):

| record says | `michel_found` p99wflip → p100c | n | owner |
|---|---|---|---|
| Michel | 0 → 1 | 6 | STM_MICHEL 6 |
| Michel | 1 → 0 | 16 | STM_MICHEL 12, STM_ONLY 3, THRU 1 |
| no Michel | 0 → 1 | 21 | STM_MICHEL 13, STM_ONLY 6, UNCLEAR 1, MESSY 1 |
| no Michel | 1 → 0 | 3 | STM_MICHEL 2, MESSY 1 |

- Of round 1's "21 new Michel false positives" (§4), 13 are Michels the record had not called; 6 stay false positives.
- Of its "16 lost hand Michels", 12 are real losses. On these 46 the candidate still misses 14 Michels production finds and
  finds 19 production misses.

### 7.2 Re-grade on the corrected record — the pre-registered criterion PASSES
Corrected records (`/home/xqian/tmp/p100/carry_r2/`, new files): the latest carried record on each arm with the 65 own100m
verdicts on that arm's key, and own100's where it names a key (10 on `p99wflip`, 0 on `p100c`). Same grader, same criterion
as `prereg.md` §2 (`scripts/d100_michel_scan_score.py` → `d100/michel_scan.txt`):

| arm | is_stm TP/FP/FN, purity, eff | Michel TP/FP/FN, purity, eff | top Michel purity / eff | bottom Michel purity / eff |
|---|---|---|---|---|
| `p99wflip` | 220/8/35, 0.965 ± 0.012, 0.863 | 135/9/29, **0.938 ± 0.020**, 0.823 | 0.931 / 0.817 | 0.953 / 0.837 |
| `p100c` | 231/9/34, 0.963 ± 0.012, 0.872 | 145/12/26, **0.924 ± 0.021**, 0.848 | 0.918 / 0.842 | 0.936 / 0.863 |

- **Michel purity −0.014 = −0.48σ: PASS** (limit 1.5σ). is_stm purity −0.002 = −0.14σ. Both efficiencies rise.
- The owner judged on the gain-ON display; a Michel visible only there is still a Michel, so each verdict applies to the
  object on both arms (registered before the scan).

### 7.3 The selection check (`scripts/d100_michel_scan_subsets.py` → `d100/michel_scan_subsets.txt`)
own100m corrected exactly where the arms disagree, so the PASS could in principle be carried by where the corrections sit.
Both arms on the same corrected records, split:

| subset | n (p99wflip / p100c) | Michel purity | is_stm purity |
|---|---|---|---|
| all | 409 / 369 | 0.938 → 0.924 (−0.48σ) | 0.965 → 0.963 (−0.14σ) |
| **untouched** (no scan corrected it) | 347 / 307 | **0.966 → 0.960 (−0.26σ)** | 0.973 → 0.961 (−0.67σ) |
| common to both arms, untouched | 236 / 236 | 0.978 → 0.978 (identical by construction) | 0.986 → 0.986 |
| candidate on one arm only | 111 / 71 | 0.929 → 0.912 (−0.24σ) | 0.925 → 0.898 (−0.47σ) |

- The PASS does not rest on where the corrections were made. What difference remains lives in the candidate-set churn,
  still graded on production-display verdicts, and is inside 1σ.
- **Reading:** round 1's 3σ drop (§4) was mostly the record under-calling Michels the gain-ON display shows. On this sample
  the package (gain + v5→v7 wire order + real window + C 0.8630) is purity-neutral within errors for both is_stm and the
  Michel, with higher efficiency.

### 7.4 The readout-edge guard's Michel exemption
**Why it is not in the guard.** `readout_edge_guard` rejects inside TaggerCheckSTM before `Flags::STM` is set;
`michel_found` is CheckSTM_Michel's, which reads only STM-flagged clusters (`require_stm_flag`, default true). Between them
`protect_bundle` (`stm_only_bundles`, `open_convicted_bundles`, `skip_convicted`) opens an STM cluster's bundle and never
splits it. own100's 4/4 was measured on arms where the guard did not fire.

**Knobs, default OFF, keys omitted when off:**
- TaggerCheckSTM `readout_edge_defer`: a pass the guard fires on carries on as with the guard off; if that pass accepts,
  the cluster gets the scalar `stm_readout_edge` = 1 and a log line names the deferral.
- CheckSTM_Michel `readout_edge_require_michel`: `stm_readout_edge` with final `michel_found` 0 → new bit `R_READOUT_EDGE`
  (1u<<14), never cleared by the topology clear; free function `stm_michel_readout_edge_bits` + two doctest cases.
- toolkit `protodunevd/pr.jsonnet` `stm_readout_edge_defer` (emitted only with the guard on), forwarded by the driver; the
  CheckSTM_Michel key through `stm_michel_extra`. Defer without it = the guard effectively off.

**Gates** (labels = arm names; pin `libpin_p100b`, Clus ef0822ec, verified copy of the installed build):

| gate | what | result |
|---|---|---|
| X0 | compiled PDVD PR config (G1 recipe) at defaults; with defer; with defer + bag key | md5 447e0f65 = G1; +1 key `readout_edge_defer`; +1 more `readout_edge_require_michel`; guard off suppresses defer |
| X1 | `./build/clus/wcdoctest-clus` | 402 passed, 0 failed (2 new cases) |
| X2 | `p100boff` (new binary, no key, p100c's config) vs `p100c` (pin p96) | **identical 120/120** (`d100/x2_identity_p100boff_p100c.txt`) |
| X3 | PDHD `h100b` (pin p100b) vs `h100a` (pin p96), production config | in progress |
| X4 | `p100bd` (defer, no bag key) vs `p100bg` (guard off) | **identical 120/120**: the deferral is the guard-off flow (`d100/x4_identity_p100bd_p100bg.txt`) |

SBND sets neither key (its configs carry no readout-edge setting); by construction, not run.

**The exemption on the candidate: `p100bx` = `p100c` + defer + require_michel** (`scripts/d100r2_exempt.py` → `d100/exempt_p100bx.txt`):
- **Deferrals:** 440 (039252 70, 039253 65, 039349 305). 332 end with `R_READOUT_EDGE` and other bits, 37 with
  `R_READOUT_EDGE` only, 51 are not accepted by the tagger's later stages, 11 have a Michel but other bits reject, **9 become
  is_stm**.
- **E1 holds exactly:** gained 9 = the 9 deferred clusters the guard-off control tags with a Michel; lost 0.
- **E2 (own100's 14 latest-side objects):** restores **2**, both owner STM_MICHEL (039349_40/48, 67/35). The other 9 owner
  stoppers stay removed (`michel_found` 0 on the arm and on the control). The exemption recovers a small, clean subset of
  the guard's cost, not the cost.
- **The other 7 gains are unjudged, and on a different edge than own100 measured:** 4 at the late edge of the 10000-tick runs
  (ticks 9947.7–10040.0: 039252_14/104, 039252_15/74, 039253_15/98, 039253_16/103) and 3 at the early edge (ticks 4.0–6.8:
  039252_8/88, 039252_9/42, 039349_71/19). Owner look `own100x`: in progress.

**The exemption on production: `p100bxp` = `p99wflip`'s pctree, production config + defer + require_michel** (`d100/exempt_p100bxp.txt`):
- **Deferrals:** 369 (039252 62, 039253 64, 039349 243); 277 `R_READOUT_EDGE` + other bits, 30 `R_READOUT_EDGE` only,
  42 not accepted by the tagger's later stages, 14 have a Michel but other bits reject, **6 become is_stm**. Lost 0.
- **E2 (own100's 11 production-side objects):** restores **2 of the 6 lost owner stoppers** (039349_48/21, 81/41, both
  STM_MICHEL); 039349_67/37 (owner STM_MICHEL) stays removed because production's chain finds no Michel on it (the own100
  payload's Michel on 67/35 was the latest arm's). No owner non-stopper returns.
- **4 gains unjudged:** 039253_1/85 (early edge, tick 2.5, Michel 1.1 MeV), 039253_14/108 (late edge of 10000, tick 9954.2),
  039349_71/21 (early, 4.1), 039349_78/20 (early, 21.1). E1 was not run on this side (no control arms).

**`own100x`** (`scripts/d100_exempt_scan_set.py`, `/home/xqian/tmp/p100/xscan`): the 11 unjudged gains of both arms, 10
objects (039349_71/19 on `p100bx` is 039349_71/21 on `p100bxp`, shown once on the candidate). Each question names the edge:
5 at the early edge (frame start), 5 at the late edge of a 10000-tick frame; none at run 039349's 6400-tick late edge, the
only edge own100 measured. Payloads from `prep_stm_michel_scan.py --arm p100bx` / `--arm p100bxp` (scratch). Served on :5017.

**The adoption rule** (`scripts/d100_exempt_scan_score.py` → `d100/exempt_scan.txt`; record
`../scan/pdvd_stm_michel_own100x_verdicts.json`, 11 items; label sha256 of every other tag unchanged). Owner, 10/10 labelled:
STM_MICHEL 8, THRU 2.

| arm | gained | owner stoppers / non-stoppers | purity | bar (arm's is_stm purity, §7.2) | rule |
|---|---|---|---|---|---|
| `p100bx` (the gain-flip candidate + exemption) | 9 | 7 / 2 | 0.778 | 0.963 | **below the bar**: 7 stoppers for 2 non-stoppers, owner decides |
| `p100bxp` (production + exemption) | 6 | 6 / 0 | 1.000 | 0.965 | at the bar |

- By edge on `p100bx`: early edge 2 STM_MICHEL / 1 THRU (039252_9/42, Michel 2.3 MeV); 039349's 6400-tick late edge 2 / 0
  (own100); the 10000-tick late edge 3 / 1 (039253_15/98, Michel 1.3 MeV). Both non-stoppers carry a Michel object under
  2.5 MeV.
- Production is moving to the gain bundle (§7.5), so `p100bx` is the arm the rule reads for production: the exemption stays
  OFF unless the owner decides otherwise. `p100bxp`'s 6/6 describes the pre-flip production only.

### 7.5 The flip (owner, 2026-09-13: *"yes you can flip the gain bundle into production"*)
What production now is: **SP top gain 0.889 + v7-uvwfit wires (imaging `pvdimg`) + each run's real readout window + PR
`stm_recomb_C` 0.8630.** Thresholds unchanged (§3.3), QtoL 0.094 unchanged (§5), readout-edge exemption OFF (§7.4).

| change | file | proof |
|---|---|---|
| SP entry default `top_gain_scale` 1.0 → **0.889** (toolkit `sp.jsonnet` default stays 1.0) | `pdvd/wct-nf-sp-dnnroi.jsonnet`; runner help | compiled at the new default = the old file with `-S top_gain_scale=0.889` (md5 436c1480 = 436c1480); `-S top_gain_scale=1.0` = the old default (c16158bb = c16158bb) |
| PR driver default `stm_recomb_C` 0.7941 → **0.8630** (toolkit default stays 0.7941) | `pdvd/wct-pr-perevt.jsonnet` | compiled at the new default = the old file with `-S stm_recomb_C=0.8630` = round 1's G1 compile (md5 18971a22, all three) |
| production imaging tag **`pvdimg`**: hard links of `p98von`'s 16 imaging archives + `img-provenance.txt` per event | `pdvd/work/*_pvdimg` (120 dirs) | 1920 archives + 120 provenance, every inode equal to `p98von`'s; sha256 manifest `d100/pvdimg_sha256_manifest.txt` (2040 files) |
| staging default `SRC_TAG` `d27fresh` → **`pvdimg`** | `pdvd/scripts/stage_ql_tag.sh` | used by the `p100flip` gate below |

**Gates:**
- **F1 production chain, no overrides:** `p100flip` = `d100_arms.sh WAVE=flip` (stm/run_campaign.sh's staging from the new
  default tag, `run_clus_evt.sh -calib -save-pctree` with the window from `readout_window_ticks.txt`, `run_pr_evt.sh -nu
  -stm-fit` with no TLA; pin `libpin_p100b`) against `p100c` on 120/120 (pctree, tlas, every PR branch, mabc-pr):
  **IDENTICAL 120/120** (`d100/f1_identity_p100flip_p100c.txt`, `d100/arm_p100flip.log`). Staging 120, clustering 120,
  PR 119 + 039349_30 (no candidate on either arm, as §3.2). Cross-binary (pin p100b vs p96) like X2, which already showed
  the two binaries identical on this path; the pctree equality also proves the table window == the env window on the
  gain-ON side, and `pvdimg` == `p99rwon`'s imaging input.
- **F2 SP at the new default:** `run_nf_sp_dnnroi_evt.sh -O _p100spflip 039252 0` (no `--top-gain-scale`) against
  `p98von` (`--top-gain-scale 0.889`), frame archives by member content on all 8 anodes: **IDENTICAL 8/8**
  (`d100/f2_sp_frames_p100spflip_p98von.txt`; no "Top gain:" line in the run's log, so the jsonnet default applied).

`d27fresh` (gain-OFF imaging) stays on disk as the pre-flip production input; `p98von` stays as the study arm `pvdimg`
links. Retiring either removes nothing from `pvdimg`.

## Files

| file | what |
|---|---|
| toolkit `cfg/pgrapher/experiment/protodunevd/pr.jsonnet` (`6ac5fffc`); `pdvd/wct-pr-perevt.jsonnet` | `stm_recomb_C` knob, default 0.7941 (byte-identical) |
| `scripts/d100_edge_scan_set.py`, `d100_edge_scan_score.py` | §1 set (symlinked payloads, manifest, questions) and fold/score |
| `scripts/d100_arms.sh` | `WAVE=pr` PR-only arm on a source pctree with a TLA; `WAVE=flip` production with no overrides (not run) |
| `scripts/d100_carry_verdicts.py` | a record's verdicts carried to another arm by geometry (d99_match's matcher) |
| `scripts/d100_twin.py`, `d100_plateau_michel_diag.py` | §3.3 threshold twin (both directions) and the plateau-binding / Michel-FP diagnostic |
| `scripts/d100_qtol_halves.py` | §5 per-half QtoL |
| `d100/` | prereg, G1, C refits, census, grades, twin, diagnostics, QtoL, edge scan, carry logs, `arm_p100c.log` |
| `../scan/pdvd_stm_michel_own100_verdicts.json` | the owner's 19 verdicts, one item per arm key (25) |
| arms kept | `p100c` (PR only; pctree links into `p99rwon`) |
| toolkit (`1e2447f2`) `clus/src/TaggerCheckSTM.cxx`, `CheckSTM_Michel.cxx`, `StmMichelFunctions.{h,cxx}`, `clus/test/doctest_stm_michel.cxx`, `cfg/.../protodunevd/pr.jsonnet` | §7.4 readout-edge exemption knobs (`readout_edge_defer`, `readout_edge_require_michel`, `stm_readout_edge_defer`), default OFF |
| `pdvd/wct-nf-sp-dnnroi.jsonnet`, `pdvd/run_nf_sp_dnnroi_evt.sh`, `pdvd/wct-pr-perevt.jsonnet`, `pdvd/scripts/stage_ql_tag.sh` | §7.5 production defaults (top gain 0.889, C 0.8630, imaging `pvdimg`); driver forwards `stm_readout_edge_defer` (false) |
| `d100/prereg_round2.md` | round 2 pre-registration |
| `scripts/d100_michel_scan_set.py`, `d100_michel_scan_score.py`, `d100_michel_scan_subsets.py` | §7.1–7.3 own100m set, fold + re-grade, selection check |
| `scripts/d100r2_arms.sh`, `d100_pdhd_identity.py`, `d100r2_exempt.py`, `d100_exempt_scan_set.py`, `d100_exempt_scan_score.py` | §7.4 arms, PDHD gate, exemption analysis, own100x set and adoption rule |
| `d100/michel_scan*.txt`, `own100m_*`, `label_shas_r2.txt`, `x2_*`, `x3_*`, `x3b_*`, `x4_*`, `exempt_*.txt`, `r2_arms.log` | round 2 records |
| `d100/flip_compiled_config.txt`, `pvdimg_sha256_manifest.txt`, `f1_*`, `f2_*` | §7.5 flip proofs |
| `../scan/pdvd_stm_michel_own100m_verdicts.json`, `pdvd_stm_michel_own100x_verdicts.json` | the owner's 65 + 10 verdicts, one item per arm key (130, 11) |
| work tags kept | `pvdimg` (PRODUCTION imaging, hard links), `p100boff`, `p100bg`, `p100bd`, `p100bx`, `p100bxp`, `p100flip`; PDHD `h100a`, `h100b` |

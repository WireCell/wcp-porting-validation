# doc pdhd/25 — PDVD's STM/Michel configuration on PDHD, the Bragg-peak + Michel fraction on both detectors, and what is left for PDHD

**The owner's three questions (2026-09-12), APA0 excluded from every PDHD number:**

1. *Take the PDVD configuration and apply it to PDHD as a whole — what are the results?*
   **It carries little of PDVD's efficiency across, and the literal copy costs purity.** Taken
   literally (Arm A), stopper efficiency goes 0.733 → 0.771, but purity drops 1.000 → **0.964** (three
   false positives, the same three already measured on PDHD) and Michel efficiency drops
   0.810 → 0.746. Without the three keys already known to be negative on PDHD (Arm B), efficiency is
   0.733 → **0.752 at purity 1.000**, and Michel efficiency drops 0.810 → 0.762. Either way only
   0.02–0.04 of the 0.144 gap to PDVD (0.877) closes. The only key that buys stoppers for free is
   `compare_range_cm` 45.
2. *Taking the best results on each detector, how many true stopping muons does the chain accept on a
   clear Bragg peak **and** then find a good Michel for — and are the two comparable?* **Not
   comparable: PDHD 0.524 ± 0.063 against PDVD 0.671 ± 0.037** of hand-scanned STM+Michel items
   (a 2.0 σ gap). About two thirds of the gap is the Bragg-peak reading (0.619 vs 0.727), one third
   the Michel given a Bragg-path accept (0.846 vs 0.923).
3. *How to raise PDHD further?* PDHD's 28 remaining misses are **25 shape-test rejections** (`no_bragg` /
   `shape_flat`), and PDHD's shape quantities separate stoppers from through-goers far worse than
   PDVD's (**AUC 0.73 vs 0.88**). Two PDHD-only threshold moves — `ks_margin` −0.10 and the P1 Michel
   floors at 5 MeV / 1.5 cm — were sized offline at +13 stoppers for 0 false positives, and **a real arm
   graded as one unit (§5) delivers exactly that, item by item: 77/0/28/69 → 90/0/15/69, efficiency
   0.733 → 0.857 at purity 1.000**, Michel census unchanged, golden fraction 0.524 → 0.587. PDVD sits at
   0.877. `compare_range_cm` 45 alone gives +3 (0.762) and nothing on top of the lever. On PDVD the same
   moves cost 22 false positives, so they are PDHD-specific. The 13 unreadable profiles and 2 plateau
   misses are what is left. **The blind re-judge (§6): 13 of 14 recoveries are stoppers on two further
   blind looks; one splits (stopper / unreadable), so by the pre-registered rule lever 1 is not free
   (worst case 89/1/15/69, purity 0.989), while `compare_range_cm` 45 is. The owner's ruling on that
   item, and on an owner-ruled control both agents disagreed with, is what stands before a flip.**

**Read-only for production.** No C++ change, no production jsonnet edit, no record or label touched.
Six new arms on a pinned binary, new tags only.

## 0. Repro

```sh
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img ; H=/home/xqian/tmp/h25 ; X=$I/pdhd/docs/scan/h25
PIN=/home/xqian/tmp/p96/libpin_p96        # toolkit 81ff37d7, libWireCellClus md5 4e1db810
S=$I/pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh

# 1. the delta, from the COMPILED configs (compile-only; each work dir holds symlinks to one pctree)
(cd $I/pdhd && PDHD_PR_COMPILE_ONLY=1 ./run_pr_evt.sh -nu -stm-fit -s h25cfg0 29107 17)
(cd $I/pdvd && PDVD_LIGHT_SUFFIX=_keep PDVD_PR_COMPILE_ONLY=1 ./run_pr_evt.sh -nu -stm-fit -s h25vcfg 39252 0)
python3 $I/pdhd/docs/scan/h22/d22_cfg_keys.py \
    $I/pdhd/work/029107_17_h25cfg0/.wct-pr_h25cfg0.json $I/pdvd/work/039252_0_h25vcfg/.wct-pr_h25vcfg.json
#    (arms A, B, K, R, KR compiled the same way with PDHD_PR_TLA="$(cat $X/tla_<arm>.txt)": cfg_proofs.txt;
#     a compile-only dir needs the pctree symlinks of 029107_17_d51hclus)

# 2. the arms (61 events, ~5 min each), launched by $X/run_arm.sh
ARM=h25base DET=pdhd SRC=d16hnu JOBS=8 PIN=$PIN LOGD=$H/arm_h25base bash $S
ARM=h25va   DET=pdhd SRC=d16hnu JOBS=8 PIN=$PIN LOGD=$H/arm_h25va PR_TLA="$(cat $X/tla_A.txt)" bash $S
ARM=h25vb   DET=pdhd SRC=d16hnu JOBS=8 PIN=$PIN LOGD=$H/arm_h25vb PR_TLA="$(cat $X/tla_B.txt)" bash $S

# 3. gates and grading
python3 $I/pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py --before "$I/pdhd/work/*_h23conf" \
    --after "$I/pdhd/work/*_h25base" --before-arm h23conf --after-arm h25base --pts --out $X/g_h25base.txt
#    (same for h25base -> h25va / h25vb)
python3 $I/pdhd/docs/scan/h23/d23_grade.py h25base h25va h25vb      # all APAs, readings A and B
python3 $I/pdhd/docs/scan/h23/d23_apa.py   h25base h25va h25vb      # per APA, APA0 excluded
python3 $X/d25_movers.py h25base h25va h25vb                        # movers by name and mechanism
python3 $X/d25_bragg_michel.py --pdhd h23conf,h25va,h25vb --pdvd p96vprod   # section 2
python3 $X/d25_misses.py                                            # section 3 (offline, production payload)

# 4. section 5: the levers as real arms (pre-registration and frozen re-judge controls written first)
python3 $X/d25_controls.py > $X/controls_frozen.txt                 # BEFORE any lever arm
bash $X/run_arm.sh h25k $X/tla_K.txt ; bash $X/run_arm.sh h25r $X/tla_R.txt
python3 $X/d25_misses.py --pdhd h25r > $X/q3_misses_on_h25r.txt      # h25kr's twin, written before h25kr runs
bash $X/run_arm.sh h25kr $X/tla_KR.txt
#    branch gates g_h25k/g_h25r/g_h25kr.txt as in 3 (--before h25base); then
python3 $X/d25_twin_check.py --arm h25k  --ks -0.10 --ke 5 --len 1.5 > $X/twin_h25k.txt
python3 $X/d25_twin_check.py --arm h25r  --named 028084_20/116,028084_5/115,029107_24/33 > $X/twin_h25r.txt
python3 $X/d25_twin_check.py --arm h25kr --ks -0.10 --ke 5 --len 1.5 > $X/twin_h25kr.txt
python3 $I/pdhd/docs/scan/h23/d23_grade.py h25base h25k h25r h25kr ; python3 $I/pdhd/docs/scan/h23/d23_apa.py h25base h25k h25r h25kr
python3 $X/d25_movers.py h25base h25k h25r h25kr > $X/movers_h25k.txt
python3 $X/d25_bragg_michel.py --pdhd h23conf,h25k,h25r,h25kr --pdvd p96vprod > $X/q2_levers.txt

# 5. section 6: the blind re-judge, tag smx25 (R=/home/xqian/tmp/h25r, the round dir)
(cd $I/pdhd/stm_michel_scan && ./prep_stm_michel_scan.py --det pdhd --arm h25base --ctx-cells \
    --outdir $PWD/prep-pdhd-h25base --sheetdir $R/prep_sheets --pin-tranche ../docs/scan/smx18/pdhd_stm_michel_scan_sheet.tsv)
python3 $X/d25_build_smx25.py --arms h25k,h25r,h25kr --round $R --outdir $I/pdhd/docs/scan/smx25 \
    --prep $I/pdhd/stm_michel_scan/prep-pdhd-h25base
#    frames: scan_harness.py shots --det pdhd --blind --hide-selection --prepdir .../prep-pdhd-h25base \
#            --manifest docs/scan/smx18/pdhd_stm_michel_scan_sheet.tsv --items-file $R/chunk_<n>.txt (2 processes,
#            private blank labeldirs); campaign/mkzoom.py $R/shots; check_shots.py $R/shots
#    scanners rv5_a1..a6: $R/AGENT_TASK.md + $R/items_a<i>.txt -> $R/v_parts/rv5_a<i> (rubric v5 frozen in $R/RUBRIC.sha)
python3 $X/d25_audit.py --selftest ; python3 $X/d25_audit.py <session>/subagents/agent-<id>.jsonl rv5_a<i>
python3 $I/pdhd/stm_michel_scan/campaign/mkowner_record.py $I/pdhd/docs/scan/pdhd_stm_michel_smx23_verdicts.json \
    $I/pdhd/docs/scan/smx25/rulings_empty.json $R $I/pdhd/docs/scan/pdhd_stm_michel_smx25_verdicts.json \
    $I/pdhd/docs/scan/smx25/provenance.json --provenance-json $R/provenance_in.json --stopper-split --arm h25base
python3 $X/d25_score_smx25.py --key $I/pdhd/docs/scan/smx25/key_smx25.tsv --round $R \
    --record $I/pdhd/docs/scan/pdhd_stm_michel_smx25_verdicts.json --reading-b-out $I/pdhd/docs/scan/smx25/record_readingB.json
STM_SCAN_RECORD=<smx25 record | record_readingB.json> D25_GATES="<printed by the scorer>" \
    python3 $I/pdhd/docs/scan/h23/d23_grade.py h25base h25k h25r h25kr     # likewise d23_apa.py, d25_bragg_michel.py
```

Every grader self-gates before printing: `d23_*` on `p82bhoff` = 61/0/87/108 (smx23);
`d25_bragg_michel.py` on PDHD production (`h23conf`) = 96/1/52/107 all APAs, 79/1/30/70 majority,
**77/0/28/69 strict**, Michel 69/2/18 / 52/2/12 / 51/2/12, and on PDVD production (`p96vprod`) =
242/8/34 and Michel 144/12/20 (doc pdvd/93, 96); `d25_misses.py` additionally reproduces
production's reject bits exactly (below).

**Conventions.** PDHD truth = the `smx23` record with owner precedence (`owner_review` >
`owner_smx1` > agent), population = the committed 303-item key. PDVD truth = the merged
smx1a…smx9 record, population = items with a candidate (546). APA0 exclusion follows doc pdhd/23 §7:
**strict** drops any candidate with a role-1 fit point in APA0 (x<0, z<231 cm), **majority** drops the
APA0-majority ones; strict is the headline, majority is reported alongside. Efficiencies are on the
chain's own candidate pool on both detectors — neither is an absolute efficiency.

## 1. Q1 — PDVD's whole STM/Michel configuration on PDHD

### 1.1 What "PDVD's configuration" is — read from the compiled configs, not the jsonnet

The two production bags were compiled and diffed structurally (`cfg_proofs.txt`). Beyond what PDHD
already carries, PDVD production sets **11 keys PDHD does not have and 1 at a different value**, plus
**3 keys in the STM tagger**:

| key | PDVD | PDHD production | what it was on PDHD before |
|---|---|---|---|
| `absorb_bragg_stub` | true | absent (false) | broke `029107/1` cluster 113 (doc pdhd/03 §6.8) |
| `topology_clears_sparse` | true | absent | cost purity (doc pdhd/22, `h22p2`) |
| `topology_michel_ke_min` | 3.0 | absent (10.0) | cost purity (doc pdhd/22, `h22p3`) |
| `compare_range_cm` | 45 | absent (35) | never run on PDHD |
| `michel_range_energy_dis_cm` | 3.0 | absent (5.0) | never run on PDHD |
| `moved_stop_michel_guard` | true | **absent** | never run on PDHD — see the finding below |
| `michel_q2d`, `_cells`, `_region_cm` 10, `_region_ctl_cm` 35, `_region_scope` 1 | set | absent | energy-only by construction (doc pdvd/95–96) |
| `plateau_mip_hi` | 2.0 | 1.6 | recovered 0 (doc pdhd/21 §5) |
| `TaggerCheckSTM` `kink_asym_enable` / `_entry_mip` / `_far_mip` | true / 1.2 / 0.5 | absent | never run on PDHD |

Kept **PDHD's own** in every arm, because they are detector constants, not choices: `mip_dqdx` /
`_median`, `fv_tolerance`, the fiducial and detector-volume objects, `pc_transforms`, the
recombination model and the track-fitting file.

**Finding (dead config, reported, not fixed):** PDHD production sets `moved_stop_michel_kink_min` 60 and
`moved_stop_michel_reach_min_cm` 6.5, but not `moved_stop_michel_guard`. Both are *exemptions from*
that guard and are read only inside `if (m_moved_stop_michel_guard && …)` (`CheckSTM_Michel.cxx:4335`),
so on PDHD they are inert. They only write the two `n_michel_veto_*exempt` counters.

A text grep of the jsonnet had also listed `stop_local_residual_*` and `stop_snap_reachable` as
"PDHD-only". The compiled diff shows **PDVD sets them too, to the same values**
(`pdvd/wct-pr-perevt.jsonnet:783-786`). That is why the delta was taken from compiled configs.

Two arms, both on top of the PDHD production file through `stm_michel_extra` and one top-level `-S`
(never `-S stm_michel_knobs=`, which replaces the bag):

* **`h25va` — Arm A, literal:** all of the table.
* **`h25vb` — Arm B:** Arm A minus the three keys already measured negative on PDHD
  (`absorb_bragg_stub`, `topology_clears_sparse`, `topology_michel_ke_min`).

### 1.2 Gates

| gate | result |
|---|---|
| compiled config, production → Arm A | **11 added, 0 removed, 1 changed** (`plateau_mip_hi`) |
| compiled config, Arm A → PDVD production | **0 added, 0 removed, 8 changed — all detector constants** |
| compiled config, Arm A → Arm B | **0 added, 3 removed, 0 changed** |
| negative control (a config against itself) | 0 / 0 / 0 |
| whole config, production → A and → B, all 51 components | only `CheckSTM_Michel` bag, `TaggerCheckSTM` kink_asym trio, and output paths differ (`cfg_wholecfg_diff.txt`) |
| **`h25base`** (production file, no TLA, new pin) vs **`h23conf`** (production on `libpin_p65`) | **341/341 bit-identical on all 149 branches, point geometry 341/341, 0 flips** — the pin change is inert, so production is the baseline |
| every arm | 61/61 complete, rc=0, pin md5 `4e1db810` before and after |

### 1.3 Results

`is_stm` (TP/FP/FN/TN, purity, efficiency) and `michel_found` on hand stoppers, on `smx23`:

| population | production `h25base` | Arm A `h25va` (literal PDVD) | Arm B `h25vb` (minus the 3 negatives) |
|---|---|---|---|
| **APA0 excluded, strict** | 77/0/28/69 — **1.000 / 0.733** | 81/3/24/65 — **0.964 / 0.771** | 79/0/26/68 — **1.000 / 0.752** |
| APA0 excluded, majority | 79/1/30/70 — 0.988 / 0.725 | 83/4/26/66 — 0.954 / 0.761 | 81/1/28/69 — 0.988 / 0.743 |
| all four APAs (reference) | 96/1/52/107 — 0.990 / 0.649 | 106/4/42/103 — 0.964 / 0.716 | 101/1/47/106 — 0.990 / 0.682 |
| Michel, strict | 51/2/12/39 — 0.962 / **0.810** | 47/1/16/40 — 0.979 / **0.746** | 48/1/15/40 — 0.980 / **0.762** |
| candidates | 341 | 340 (`028084_16/106`, an owner THRU, left the pool) | 340 (the same item left) |

For comparison, **PDVD production** is 242/8/34, purity 0.968 / efficiency **0.877**, Michel efficiency
0.882 (same definition).

### 1.4 Arm A, by item (`movers_h25va.txt`)

APA0 strict: **+5 stoppers, −1 stopper, +3 false positives; −4 true Michels, −1 false Michel.**

| item | truth | effect | the key that did it (traced, not assumed) |
|---|---|---|---|
| `028084_20/116`, `028084_5/115`, `029107_24/33` | stopper | `shape_flat` cleared | **`compare_range_cm` 45** — it moves `ks_mu`/`ks_flat` on 286 of 340 candidates; the KS gap on these goes −0.039→+0.038, −0.029→+0.039, −0.062→−0.014 |
| `028084_2/116`, `029107_3/99` | stopper | P1 clears | **`topology_michel_ke_min` 3** (Michels at 7.8 and 5.4 MeV) |
| `028084_18/17`, `029107_19/123`, `029107_7/45` | **THRU** | **new false positives** | **`topology_clears_sparse`** (`topology_cleared_bits` = 512) — **the exact three items pre-registered by name** from `h22p2`/`h22p3` |
| `029107_26/88` | stopper | **lost** (`shape_flat` set) | **`kink_asym`** — the tagger's kink moved (`kink_num` 123→117), the stop moved 2.4 cm, the profile read flat |
| `028084_26/109`, `028084_7/61`, `029107_12/108` | hand Michel | **Michel lost** | **`michel_range_energy_dis_cm` 3** (`n_michel_range_veto` 0→1; Michels of 3.7 / 1.8 / 7.9 MeV) |
| `029107_1/85` | hand Michel | **Michel lost** | **`absorb_bragg_stub`** (`n_stub_absorb` 1) swallowed the Michel stub — the doc pdhd/03 §6.8 failure mode again |

### 1.5 Arm B, by item

APA0 strict: **+3 stoppers, −1 stopper, 0 false positives; −3 true Michels, −1 false Michel**
(`movers_h25vb.txt`). Branch gate (`g_h25vb.txt`): 40/340 candidates bit-identical, `ks_mu`/`ks_flat`
moved on 286, point geometry changed on only 7 (the `kink_asym` stop moves).

| item | truth | effect | key |
|---|---|---|---|
| `028084_20/116`, `028084_5/115`, `029107_24/33` | stopper | `shape_flat` cleared | `compare_range_cm` 45 — the same three as Arm A |
| `029107_26/88` | stopper | **lost** | `kink_asym` — as in Arm A |
| `028084_26/109`, `028084_7/61`, `029107_12/108` | hand Michel | **Michel lost** | `michel_range_energy_dis_cm` 3 — as in Arm A |
| `028084_23/114` | STM_ONLY | false Michel removed | `michel_range_energy_dis_cm` 3 |

**Arm B is Arm A with exactly the three negative keys' effects removed.** On APA0 strict, A − B =
+2 stoppers (`028084_2/116`, `029107_3/99`, from `topology_michel_ke_min` 3), +3 false positives (from
`topology_clears_sparse`) and −1 Michel (`029107_1/85`, from `absorb_bragg_stub`). That matches the
per-item traces of §1.4 exactly.

### 1.6 The pre-registration, graded

Written before either arm launched (`preregistered.txt`):

| # | expectation | outcome |
|---|---|---|
| base | `h25base` ≡ `h23conf` 341/341 | **HELD** |
| 1 | 61/61, rc 0, pin unchanged | **HELD** (all three arms) |
| 3 | `kink_asym` moves a few candidates in/out of the pool | **HELD** — 1 out (`028084_16/106`) in A; the same single item in Arm B |
| 4 | Arm A's `clears_sparse` + `ke_min` 3 bring back `h22p3`'s FPs `028084_18/17`, `029107_19/123`, `029107_7/45` | **HELD BY NAME** — all three, nothing else |
| 5 | `029107_1/113` cannot move the census | **HELD** — not a candidate in either arm |
| 6 | `plateau_mip_hi` 2.0 buys ~0 | **HELD on APA0-excluded** (0); one APA0 item (`028084_3/23`) cleared `plateau_off_mip`, in both arms |
| 7 | PDVD's config does not close the gap: strict efficiency stays < ~0.78, falsifier ≥ 0.80 | Arm A **HELD** (0.771); Arm B **HELD** (0.752) |

### 1.7 Answer to Q1

**PDVD's configuration, taken whole, does not carry PDVD's efficiency to PDHD.**

| APA0 strict | `is_stm` purity | `is_stm` efficiency | Michel efficiency | GOLDEN (§2) |
|---|---|---|---|---|
| PDHD production | 1.000 | 0.733 | 0.810 | 33/63 = 0.524 |
| + PDVD config, literal (Arm A) | 0.964 | 0.771 | 0.746 | 31/63 = 0.492 |
| + PDVD config minus the 3 known negatives (Arm B) | **1.000** | **0.752** | 0.762 | 31/63 = 0.492 |
| *PDVD production* | *0.968* | *0.877* | *0.882* | *108/161 = 0.671* |

The one key in PDVD's delta that buys stoppers for free on PDHD is **`compare_range_cm` 45** (+3, all
`shape_flat`). Widening the KS window acts as a shape relaxation — the same direction as §3.4's lever
1. Every other key is neutral or costs something on PDHD:

* `topology_clears_sparse` brings back its three named false positives.
* `topology_michel_ke_min` 3 buys two stoppers, but only together with those false positives in Arm A.
* `michel_range_energy_dis_cm` 3 vetoes three true, small PDHD Michels (1.8–7.9 MeV).
* `kink_asym` loses a stopper.
* `absorb_bragg_stub` swallows a Michel.

The GOLDEN fraction goes *down* on both arms (0.524 → 0.492), because the Michel losses land on items
the Bragg path already accepts (Michel given a Bragg-path accept falls 0.846 → 0.775).

Per-key attribution here comes from the mechanism fields each mover carries (`topology_cleared_bits`,
`n_michel_range_veto`, `n_stub_absorb`, `kink_num`) and from the A − B difference, **not** from
single-key arms.

## 2. Q2 — a clear Bragg peak and then a good Michel

### 2.1 Definitions (the owner's choice: the chain's own path, scored on hand truth)

* **hand Michel item** — a hand stopper (`STM_MICHEL`/`STM_ONLY`) whose `michel_kind` is attached or both.
* **Bragg-path accept** — `is_stm = 1` **and `topology_cleared_bits = 0`**: the stopper was accepted by the
  dQ/dx shape tests themselves (`no_bragg`, `shape_flat` never set), not rescued by P1 on the strength
  of a Michel. *Anchor-helped* = Bragg-path accepts where the geometric fallback or the wide anchor
  fired.
* **GOLDEN** — a hand Michel item with a Bragg-path accept **and** `michel_found = 1`.

`d25_bragg_michel.py`, production arms. PDHD production is labelled `h23conf` in the script's output
and `h25base` in §1; the two are the same chain, bit-identical on 341/341 candidates (§1.2). PDVD
production is `p96vprod`.

| | PDHD APA0 strict | PDHD APA0 majority | PDHD all APAs | **PDVD** |
|---|---|---|---|---|
| hand stoppers | 105 | 109 | 148 | 276 |
| accepted (`is_stm` efficiency) | 0.733 ± 0.043 | 0.725 ± 0.043 | 0.649 ± 0.039 | **0.877 ± 0.020** |
| **Bragg-path accept**, all stoppers | **0.657 ± 0.046** | 0.651 ± 0.046 | 0.568 ± 0.041 | **0.772 ± 0.025** |
| … of which anchor-helped | 0.038 | 0.037 | 0.041 | 0.040 |
| topology-rescued, all stoppers | 0.076 | 0.073 | 0.081 | 0.105 |
| Bragg-path accept, hand `STM_ONLY` | 29/41 = 0.707 ± 0.071 | 30/44 = 0.682 | 38/60 = 0.633 | 95/112 = **0.848 ± 0.034** |
| hand Michel items | 63 | 64 | 87 | 161 |
| Bragg-path accept, hand Michel items | 39/63 = **0.619 ± 0.061** | 40/64 = 0.625 | 45/87 = 0.517 | 117/161 = **0.727 ± 0.035** |
| Michel found, given a Bragg-path accept | 33/39 = **0.846 ± 0.058** | 34/40 = 0.850 | 38/45 = 0.844 | 108/117 = **0.923 ± 0.025** |
| **GOLDEN**, over hand Michel items | **33/63 = 0.524 ± 0.063** | 34/64 = 0.531 ± 0.062 | 38/87 = 0.437 ± 0.053 | **108/161 = 0.671 ± 0.037** |
| GOLDEN without anchor help | 0.476 | 0.484 | 0.379 | 0.640 |
| GOLDEN, over all hand stoppers | 33/105 = 0.314 | 34/109 = 0.312 | 38/148 = 0.257 | 108/276 = 0.391 |
| the chain's own GOLDEN selection | 36 items: 33 hand Michel, 1 STM_MICHEL of another kind, 2 STM_ONLY, **0 THRU** | 38 (1 THRU) | 42 (1 THRU) | 110: 108 hand Michel, 2 STM_ONLY, **0 THRU** |

### 2.2 Are they comparable?

**No — PDHD's golden fraction is 0.147 ± 0.073 lower (2.0 σ), and the gap is mostly the Bragg read.**
GOLDEN factorises as (Bragg-path accept on a hand Michel item) × (Michel found | Bragg-path accept):

* PDHD 0.619 × 0.846 = 0.524; PDVD 0.727 × 0.923 = 0.671.
* In log terms **65 % of the gap is the Bragg-peak reading, 35 % the Michel**.

What *is* comparable:

* **The chain's golden selection is pure on both.** Zero hand-THRU items in it on PDHD strict (36) and
  PDVD (110). When both tests pass, the item is a stopper on both detectors.
* **The help rates are the same.** Anchor help is 4 % on both. Topology rescue is 8 % vs 10 %. PDHD is
  not leaning on the rescue paths more than PDVD — it simply passes the shape tests less often.
* **The Bragg peak itself looks the same size.** Median contrast on hand `STM_MICHEL` is 1.52 on PDHD
  against 1.56 on PDVD. PDHD's difference is in the *spread* (§3.2), not the centre.

The Michel factor is not a finder problem alone. On PDHD, 12 of the 63 hand Michel items have no
`michel_found` at all (0.810), against 19 of 161 on PDVD (0.882).

**Caveats that apply to this comparison and are not quantified:** PDVD's base scan was not verdict-blind
and PDHD's was (doc pdhd/23 §4 point 1), which inflates PDVD. The Michel census has two definitions in
use — hand stoppers vs `michel_kind` (PDHD's graders, used here on both) and all judged vs
`STM_MICHEL` (`census_score` §14.2, PDVD's 144/12/20). Both are printed by the script. Both
denominators are candidate pools.

With lever 1 as a real arm (§5), PDHD's golden fraction is **37/63 = 0.587 ± 0.062**, still below
PDVD's 0.671: the lever lifts the Bragg-path accept on hand Michel items 0.619 → 0.714, while Michel
found given a Bragg-path accept moves 0.846 → 0.822 (the new accepts carry fewer found Michels).

## 3. Q3 — what is left on PDHD, and what would move it

### 3.1 The misses are shape-test rejections, and only shape-test rejections

`d25_misses.py`, production:

| | PDHD APA0 strict (28 misses) | PDVD (34 misses) |
|---|---|---|
| `no_bragg` + `shape_flat` | 17 | 9 |
| `shape_flat` only | 8 | 9 |
| `no_bragg` only | 0 | 4 |
| `plateau_off_mip` | 2 | 3 (+1 with `no_bragg`) |
| `shape_flat` + `profile_sparse` | 1 | 0 |
| geometry / topology bits (`stop_near_boundary`, `continuation`, `vertex_hadron`) | **0** | 8 |
| carry `michel_found = 1` | 10 | 11 |

PDHD has already cleared every non-shape rejection site PDVD still has. **All of PDHD's remaining
inefficiency is "the chain cannot read the Bragg rise."**

### 3.2 PDHD's shape quantities separate stoppers from through-goers much worse

The chain's own two shape quantities, hand stoppers against hand THRU (AUC: 0.5 = no separation):

| | PDHD APA0 strict | PDVD |
|---|---|---|
| AUC, KS gap (`ks_flat − ks_mu`) | **0.732** | **0.877** |
| AUC, contrast / expected | **0.735** | **0.880** |
| stoppers, KS gap p25 / p50 / p75 | **−0.042** / 0.015 / 0.062 | −0.009 / 0.031 / 0.066 |
| THRU, KS gap | −0.091 / −0.066 / −0.030 | −0.086 / −0.073 / −0.047 |
| stoppers, contrast / expected | **0.61** / 0.78 / 0.94 | 0.70 / 0.84 / 0.96 |
| THRU, contrast / expected | 0.39 / 0.49 / **0.68** | 0.43 / 0.50 / 0.57 |
| stoppers, plateau / MIP | 0.96 / 1.05 / 1.13 | 0.93 / 1.02 / 1.11 |
| THRU, plateau / MIP | 0.67 / 0.86 / 1.01 | 0.79 / 0.94 / 1.04 |

Both populations are broader on PDHD. Stoppers have a lower tail reaching into THRU territory, and
through-goers reach up into stopper territory. The medians barely differ.

### 3.3 The profile noise is not larger on PDHD — it is slower

Over the muon plateau (residual range 20–60 cm, points above the chain's 0.15-MIP live cut):

| | robust scatter (1.48·MAD/median) p50 | median neighbour-to-neighbour jump / median |
|---|---|---|
| PDHD stoppers accepted / missed / THRU | 0.191 / 0.231 / 0.280 | 0.072 / 0.069 / 0.092 |
| PDVD stoppers accepted / missed / THRU | 0.182 / 0.200 / 0.231 | 0.102 / 0.102 / 0.109 |

The amplitude is similar, but PDHD's point-to-point jumps are **30 % smaller** at that amplitude, so its
fluctuations are correlated over several points. This matches the independent measurement of doc
pdvd/65 §3: plateau lag-1 autocorrelation **0.60 on PDHD against 0.42 on PDVD**. It also matches the
periodic dQ/dx wave of doc pdhd/24, whose correlation length is *not* the wire-crossing cell (doc
pdhd/24 §0, **a local commit not yet on the remote**). A correlated wave does not average out over the
3 cm Bragg tail window or the 35 cm KS range: it can fake a rise on a through-goer or bury one on a
stopper. That is consistent with §3.2's two-sided broadening. **Consistent with, not demonstrated as,
the mechanism** — nothing here isolates it.

### 3.4 The levers, sized offline on production's payload

**Model.** The model re-derives every bit from the payload with the C++'s own gates: `no_bragg` =
contrast < `bragg_contrast_min` × expected (`:3083`); `shape_flat` = `ks_mu` + `ks_margin` ≥ `ks_flat`
(`:3121`); P1 last, on final bits (`:4440`, `StmMichelFunctions.cxx:772`). Before any sweep it
**reproduces production exactly**:
* pre-P1 bits on 170/170 modelled PDHD items and 534/534 PDVD items (the 4 + 12 where the geometric
  fallback or the wide anchor fired are carried unchanged);
* `topology_cleared_bits` on 174/174 and 546/546;
* `is_stm` ≡ (`reject_bits` = 0) on all items.

**These are predictions for single threshold moves on a fixed payload.** They cannot see a knob that
moves the fit or the pool, and they are in-sample. **The combined lever and `compare_range_cm` 45 were
then run as real arms: §5.**

| lever (all else production) | PDHD APA0 strict | PDHD all APAs | PDVD |
|---|---|---|---|
| production | 77/0/28/69 — 0.733 | 96/1/52/107 — 0.649 | 242/8/34 — 0.877 |
| **`ks_margin` −0.05** | 81/0 — 0.771 (+4 / 0 FP) | +5 / 0 | +5 / **+6 FP** |
| **`ks_margin` −0.10** (saturates; −0.20 identical) | **85/0 — 0.810 (+8 / 0 FP)** | 109/1 — 0.736 (+13 / 0) | +9 / **+19 FP** |
| `bragg_contrast_min` 0.5 / 0.4 | +0 / 0; +0 / +1 FP | — | +2 / +14 FP |
| **P1 floors `ke_min` 5 MeV, `len_min` 1.5 cm** | **83/0 — 0.790 (+6 / 0 FP)** | 104/1 — 0.703 (+8 / 0) | +3 / **+3 FP**, −1 |
| P1 floors 0 / 0 | +8 / **+4 FP** | — | +5 / +6 FP |
| `topology_clears_sparse` | +0 / **+2 FP** | — | 0 |
| **`ks_margin` −0.10 + P1 5 MeV / 1.5 cm** | **90/0 — 0.857 (+13 / 0 FP)** | 115/1 — 0.777 (+19 / 0) | +10 / **+22 FP** |
| *hypothetical, no such knob:* a Michel ≥ 20 MeV and ≥ 10 cm also clears `plateau_off_mip` | +2 / 0 | +2 / 0 | 0 |
| *all three together* | *92/0 — 0.876* | *117/1 — 0.791* | +10 / +22 FP |

Three readings:

1. **On PDHD, `shape_flat` is not what keeps through-goers out.** Every `shape_flat`-only miss comes back
   by `ks_margin` −0.10, and nothing moves after that, even at −0.20.
   * **PDHD, APA0 strict:** 55 of the 69 hand through-goers carry `shape_flat`. With it removed
     entirely, **0 of the 55** would be accepted. Each still carries another bit: `no_bragg` alone on 18,
     `no_bragg` + `stop_near_boundary` on 14, `stop_near_boundary` alone on 8, the rest `plateau_off_mip`
     or `profile_sparse`. Their KS gaps span −0.170 to −0.026, so saturation comes from those other
     bits, not from a KS-gap floor (`q3_misses.txt` §6).
   * **PDVD:** the opposite. **19 of its 227** `shape_flat`-rejected through-goers carry no other bit —
     exactly the 19 false positives the −0.10 sweep predicts there. The `ks_margin` −0.02 PDHD inherited from PDVD (doc pdhd/21) is a PDVD-tuned
   operating point.
2. **The P1 Michel floors are too high for PDHD's Michels.** Six misses have an attached or bridged
   Michel that the chain found, blocked only by 10 MeV or 3 cm. Doc pdvd/92 §7 found PDVD's last five
   reachable stoppers blocked by the same 3 cm floor. On PDVD, though, lowering it costs purity; on
   PDHD, on this record, it does not.
3. **Lowering `bragg_contrast_min` does nothing useful on either detector.** `no_bragg` is not a
   near-miss: PDHD's `no_bragg` misses read contrast/expected 0.17–0.52 against a cut at 0.60.

**Arms A and B already show lever 1 working through a different key.** Its three `shape_flat` recoveries
(`028084_20/116`, `028084_5/115`, `029107_24/33`) all come from `compare_range_cm` 45 moving the KS
inputs, and all three are in the offline `ks_margin` recovery list.

### 3.5 The 28 misses, item by item (`q3_misses.txt` §5)

| class | n | items |
|---|---|---|
| `shape_flat` only → `ks_margin` ≤ −0.10 | 8 | `028084_10/46`, `028084_20/116`, `028084_5/115`, `029107_12/118`, `029107_12/95`, `029107_21/65`, `029107_24/33`, `029107_28/109` |
| Michel found, below the P1 floor → 5 MeV / 1.5 cm | 6 (1 overlaps above) | `028084_2/116`, `028084_2/49`, `028084_20/116`, `029107_1/85`, `029107_3/99`, `029107_7/85` |
| `plateau_off_mip` with a strong Michel (41–44 MeV, 15–19 cm, contrast 1.8 × expected) | 2 | `028084_24/49`, `028084_28/110` |
| **unreadable — no threshold reaches them** | **13** | `028084_11/130`, `028084_14/83`, `028084_22/57`, `028084_23/53`, `028084_27/107`, `028084_28/106`, `028084_29/115`, `028084_8/120`, `029107_14/89`, `029107_19/120`, `029107_23/116`, `029107_29/119`, `029107_8/104` |

The 13 read contrast/expected **0.17–0.52**, and 12 of 13 have no Michel found. Five are hand
`STM_MICHEL`, and the chain finds no Michel on four of them; the fifth, `029107_19/120`, has an 18 MeV
Michel only 0.4 cm long. Seven sit in APA3, four in APA1 and two in APA2.

### 3.6 Recommendations, ranked, each with the measurement that would test it

1. **A PDHD-specific shape operating point: `ks_margin` −0.10 plus the P1 floors at 5 MeV / 1.5 cm,
   graded as one unit.** **Measured (§5): +13 stoppers at 0 false positives, 0.733 → 0.857 (strict),
   the offline twin held item by item on every population; the Michel census does not move.** It needs
   no C++. What still stands between it and a flip:
   * **the blind re-judge (§6) is done:** 13 of 14 recoveries are stoppers on two further blind looks,
     and one (`029107_21/65`) splits stopper / unreadable. By the pre-registered rule the lever is
     therefore **not free** (reading B 89/1/15/69, purity 0.989). Zero false positives on 69
     through-goers still bounds the rate only below ~4 % (95 %), and both thresholds were chosen on
     this record;
   * the owner's ruling on `029107_21/65`, and a look at `028084_18/17`, where both blind agents called
     the owner's THRU a stopper;
   * the owner's decision.

   Do not port it to PDVD: there it costs 22 false positives.
2. **From PDVD's delta, take only what Arm B shows is free.** On this record that is **`compare_range_cm` 45 alone**:
   **measured (§5), +3 stoppers at 0 false positives (0.762), free on the re-judged record (§6), and nothing on top of lever 1** — the
   combined arm `h25kr` is census-identical to `h25k`, because its three recoveries are a subset of lever 1's. Keep `absorb_bragg_stub`,
   `topology_clears_sparse` and `topology_michel_ke_min` 3 off. Hold `michel_range_energy_dis_cm` 3: it
   vetoed three hand-confirmed small PDHD Michels. Hold `kink_asym`: it lost `029107_26/88`.
3. **A strong-Michel override for `plateau_off_mip`** (+2, new C++ behind a default-OFF knob).
   Doc pdhd/21 §5 showed no plateau window separates these charge-scale misses from through-goers. A
   41–44 MeV, 15–19 cm attached Michel does. Size its null population (THRU with a strong Michel and a
   low plateau) before building.
4. **The 13 unreadable misses are a profile problem, not a verdict problem.** Measure first, offline and
   read-only, whether a correlation-aware reading of the tail lifts PDHD's shape AUC (0.73) toward
   PDVD's (0.88). Two candidates: averaging the profile over its ~1.2 cm correlation length before
   the KS and contrast tests (doc pdvd/65 §3), or a test that fits the doc pdhd/24 wave as a nuisance.
   Only a lifted AUC would justify a knob. In parallel, look at why the chain finds no Michel on the
   four hand `STM_MICHEL` items among the 13 that lack one, and at the APA3 concentration (7 of 13).
5. **APA0 is structural** (plane-2 hardware, doc pdvd/50 §§10–17, doc pdhd/10). Excluded here as asked.
   For the record, Arm A lifts APA0-only efficiency 0.436 → 0.590, the largest per-APA gain.

## 4. What is NOT concluded

* **Not** a flip. Nothing here changes production; §3.4's thresholds are offline predictions, and §5's
  arms measure them without turning anything on.
* **Not** that PDVD's efficiency advantage is physics. PDVD's scan was not blind; both denominators are
  candidate pools; the two records differ in size (105 vs 276 stoppers on the compared populations).
* **Not** that the correlated wave causes PDHD's weaker shape separation — §3.3 is consistent with it,
  no more.
* **Not** attributed within Arm B beyond what its movers name — it is still a multi-key unit.
* **Not** a statement about APA0, excluded throughout at the owner's request.
* **Not** that §5's +13 are owner-confirmed stoppers. All 13 were agent calls; the blind re-judge (§6)
  is a second agent pass (13 confirmed, 1 split), and on the one owner-ruled control it disagreed with
  the owner, in the lever's direction.

## 5. The levers as real arms (the owner's request of 2026-09-12)

Three arms on the pin and source of §1, each on top of the production file through
`stm_michel_extra` only, **pre-registered before launch** (`preregistered.txt`) with the re-judge
controls frozen before any of them ran (`controls_frozen.txt`, §6):

* **`h25k` — lever 1, one unit:** `ks_margin` −0.10, `topology_michel_ke_min` 5, `topology_michel_len_min_cm` 1.5.
* **`h25r` — `compare_range_cm` 45 alone.**
* **`h25kr` — both**, pre-registered from `d25_misses.py` run on `h25r`'s own payload after `h25r` finished.

### 5.1 Gates

| gate | result |
|---|---|
| compiled config, production → K / R / KR | **2 added + `ks_margin` −0.02 → −0.1 / 1 added / 3 added + 1 changed**; K → KR = `compare_range_cm` only; negative control 0/0/0 (`cfg_proofs.txt`) |
| every arm | 61/61 complete, rc 0, pin md5 `4e1db810` before and after |
| candidate pool | **341 in every arm** (none in, none out) |
| `h25base` → `h25k` branch gate (`g_h25k.txt`) | 174/341 bit-identical; point geometry **338/341** |
| `h25base` → `h25r` | 41/341 bit-identical (`ks_mu`/`ks_flat`/`comp_*` on 287); point geometry **341/341** |
| `h25base` → `h25kr` | 29/341 bit-identical; point geometry **338/341** |
| `h25kr`'s twin input: `d25_misses.py` on `h25r` | model self-check 170/170 bits, 174/174 P1 clears, 174/174 `is_stm` |

### 5.2 Results, on `smx23`

| APA0 strict unless stated | production `h25base` | **`h25k`** lever 1 | `h25r` `compare_range_cm` 45 | `h25kr` both |
|---|---|---|---|---|
| **`is_stm`, strict** | 77/0/28/69 — 1.000 / 0.733 | **90/0/15/69 — 1.000 / 0.857** | 80/0/25/69 — 1.000 / 0.762 | 90/0/15/69 — 1.000 / 0.857 |
| `is_stm`, majority | 79/1/30/70 — 0.988 / 0.725 | 93/1/16/70 — 0.989 / 0.853 | 82/1/27/70 — 0.988 / 0.752 | 93/1/16/70 — 0.989 / 0.853 |
| `is_stm`, all four APAs | 96/1/52/107 — 0.990 / 0.649 | 115/1/33/107 — 0.991 / 0.777 | 101/1/47/107 — 0.990 / 0.682 | 115/1/33/107 — 0.991 / 0.777 |
| all APAs, owner + high-confidence truth (reading B) | 66/1/5/58 — 0.985 / 0.930 | 67/1/4/58 — 0.985 / 0.944 | 67/1/4/58 | 67/1/4/58 |
| Michel, strict | 51/2/12/39 — 0.810 | **unchanged** | unchanged | unchanged |
| Bragg-path accept, all hand stoppers | 69/105 = 0.657 | **79/105 = 0.752** | 71/105 = 0.676 | 79/105 = 0.752 |
| **GOLDEN** (§2) | 33/63 = 0.524 | **37/63 = 0.587 ± 0.062** | 34/63 = 0.540 | 37/63 = 0.587 |
| per APA (all-APA population) | APA1 0.621, APA2 0.743, APA3 0.778, APA0 0.436 | APA1 0.828, APA2 0.886, APA3 0.844, APA0 0.564 | APA2 0.771, APA3 0.822, APA0 0.487 | as `h25k` |

*PDVD production for comparison: 0.968 / 0.877, golden 0.671.* Lever 1 brings PDHD's APA0-strict
efficiency to within 0.02 of PDVD's at higher purity; the golden fraction still trails (0.587 vs 0.671),
because the lever's new accepts are the shape-marginal ones and carry fewer found Michels.

The 15 strict misses left after `h25k` are exactly §3.5's 13 unreadable profiles plus the 2
`plateau_off_mip` items with strong Michels.

Outside the scored 303-item population, `h25k` also flips 4 candidates: two cap extras that the record
calls stoppers (`028084_7/124`, `029107_5/98`) and two record-`UNCLEAR` items in APA0 (`029107_18/45`,
`029107_8/44`). None enters any number above.

### 5.3 The pre-registration, graded

| arm | expectation | outcome |
|---|---|---|
| `h25k` | 61/61, rc 0, pin unchanged, pool 341 | **HELD** |
| `h25k` | the offline twin, item by item: strict +13 by name, majority +14, all +19; 0 new FP, 0 lost | **HELD EXACTLY** on all three populations (`twin_h25k.txt`) — the offline model's per-item prediction is the arm's result |
| `h25k` | Michel census unchanged | **HELD** |
| `h25k` | geometry 341/341 and contrast / KS inputs unchanged on every candidate ("both keys only gate bits") | **FAILED on 6 candidates, with no verdict consequence.** Three newly accepted stoppers (`028084_2/49`, `028084_4/29`, `029107_5/32`) gained stop gammas — `stop_gamma_require_stm` keys gamma collection on `is_stm`, so an accept feeds back into the stop's companions (role-5 points 31 → 39). Three already-accepted stoppers (`028084_12/36`, `028084_17/97`, `029107_20/57`) switched their Bragg-anchor geometric-fallback reading, which is decided on whether the anchored profile rejects. The keys do more than gate the final bits; the census is untouched |
| `h25r` | 61/61, pool 341, geometry 341/341, Michel unchanged | **HELD** |
| `h25r` | `ks_mu`/`ks_flat` move on ~286 | **HELD** (287) |
| `h25r` | `no_bragg` unchanged on every candidate (contrast does not read `compare_range_cm`) | **HELD** — 0 movers; `shape_flat` newly set on 13 already-rejected candidates, cleared on 16 |
| `h25r` | strict +3 by name, 0 FP, 0 lost | **HELD** (strict and majority, `twin_h25r.txt`). On all four APAs it adds two APA0 stoppers (`028084_21/17`, `028084_30/34`), a population that was not pre-registered |
| `h25kr` | the twin from `h25r`'s payload: the same 13 / 14 / 19 as `h25k`, 0 FP, Michel unchanged; saturation survives the wider window (0 of 55 `shape_flat` THRU pass with it removed) | **HELD EXACTLY** (`twin_h25kr.txt`); census identical to `h25k` in every population |

### 5.4 What the arms say

* **Lever 1 is what the offline model said it is, on every item.** That is the strongest form of
  prediction this campaign has had: the naive twins of `h21f`, `h21z` and `h22c` all broke (docs
  pdhd/21–22); this one, built item by item on the arm's own payload, held on all three populations,
  and again for `h25kr`.
* **`compare_range_cm` 45 is free on its own and redundant with lever 1.** If lever 1 is taken, adding it
  changes no verdict on this record (it does move the KS inputs on 287 candidates, which is a reason to
  leave it out rather than add a key that buys nothing).
* **Neither arm touches the Michel side.** The golden fraction rises only through the Bragg-path accept.
* **What the arms cannot say:** whether the 13 are true stoppers. All 13 are agent calls on `smx23`
  (none owner-labelled), and the thresholds were sized on this very record. That is §6.

## 6. The blind re-judge (`smx25`)

**Result in one line.** 13 of the 14 recoveries were called stoppers by both blind scanners. One,
`029107_21/65`, split between STM_ONLY and UNCLEAR. By the pre-registered rule, that split makes lever 1
**not free**: reading B has 89/1/15/69, purity 0.989. `compare_range_cm` 45 alone **is free**. The
controls held 4/4 on stoppers and 7/8 on through-goers. The eighth is the tranche's only owner-ruled
item, and both agents called the owner's THRU a stopper. A defect in the fold tool, which would have
let that call overwrite the owner's ruling, was found and fixed before any number was computed (§6.3).

### 6.1 Design (frozen before the arms)

**Owner's choice:** agents now, blind. The design, the controls and the outcome rule were written to
`preregistered.txt` **before any lever arm ran**; only the decision set itself comes from the arms.

* **Decision set — 14 items:** every `is_stm` mover of `h25k`/`h25r`/`h25kr` on the APA0-majority-excluded
  303 population (the 13 strict recoveries plus the majority-only `028084_29/53`). All 14 read like
  production's misses (`is_stm` 0, shape bits only). **All 14 are agent calls in `smx23`**; none is
  owner-labelled.
* **Two strata**, because a Michel the chain found is visible on the display even with `--blind`:
  M (production `michel_found` 1) and N (0) — 7 decision items each.
* **Controls — 12 items, frozen by `d25_controls.py` before the arms:** per stratum, the 4 hand-THRU
  items nearest the new operating point (fewest bits left with `shape_flat` off and P1 at 5 / 1.5, then
  the Michel's shortfall to the floor, then contrast/expected nearest 0.60) and 2 hand-stopper misses
  that no arm moves (seed 25). No arm moved any control, so no replacement was needed.
  * THRU M: `028084_18/122`, `029107_5/28`, `028084_18/17` (owner THRU), `029107_19/123`
  * THRU N: `028084_24/114`, `028084_0/108`, `029107_4/118`, `028084_1/28`
  * stoppers M: `028084_24/49`, `029107_19/120`; N: `028084_8/120`, `028084_23/53`
* **Scan:** 26 items shuffled (seed 25) into three groups; **two verdict-blind scans per item** by two of
  six agents over two waves, rubric v5 (sha `750751ea…`, the same text as docs pdhd/19 §8–9), frames shot
  blind on today's production (`prep-pdhd-h25base`, with the grey context cells). Transcripts audited
  for forbidden reads (`d25_audit.py`, negative control 12/12 flagged including globbed, recursive and
  `find` reads of another scanner's records, 0/5 false alarms).
* **What each set can do**, fixed before the arms: a decision item re-judged not-a-stopper turns an arm
  TP into an **FP** (the only items that can cost purity); a THRU control re-judged a stopper is
  TN → FN (efficiency, not purity); a stopper control that moves measures the re-judge itself.
* **Fold:** `mkowner_record.py --stopper-split` into a new record `smx25` (smx23 is never written):
  confirmed / adopted / split. **Reading B** (worst case for the lever): a split decision item counts
  as THRU. **Decision rule:** the lever is *free on the re-judged record* iff 0 new FP on APA0 strict
  under reading B. Thresholds are never re-selected on this tranche.

### 6.2 Integrity

| check | result |
|---|---|
| frames | `prep-pdhd-h25base` with `--ctx-cells` (the `h23conf` prep has no `proj_ctx`, so no grey cells); Canvas2D; `check_shots` clean (0 blank, `c_3d_stop` min 2246 unique colours); `context.json` carries no chain-verdict key on 26/26 (`smx25/check_shots.txt`) |
| scans | 26 items × 2 = **52 records, each item scanned by exactly its two assigned scanners**; rubric sha `750751ea…` on all 52 |
| order | each item's second scan started only after its first scanner had finished (`a6` after `a1`, `a4` after `a2`, `a5` after `a3`) |
| audit | 6 transcripts, 87 / 86 / 78 / 92 / 77 / 87 tool calls, **0 flagged** (`smx25/audit.txt`); the audit flags 12/12 synthetic forbidden reads and 0/5 clean ones. The first version matched only literal paths, so a globbed (`v_parts/*/…`), recursive-grep or `find` read of another scanner's records would have passed. It was widened after the scan (any `v_parts` not followed by the scanner's own dir, any `find`, any `grep -r`, any wildcard under the round dir) and re-run: still 0 |
| independence | the 26 pairs' evidence texts share at most 0.25 of their characters in order (`difflib` ratio; median about 0.16). Two pairs pinned the same 0.1 cm (`029107_19/120` at 10.5, `029107_1/85` at 7.4); pins snap to chain points, and both pairs' evidence texts differ (ratio 0.07 and 0.10) |
| display note | `f_meas`'s time axis is the payload's absolute slice index. `a1` asked; on `029107_4/118` the track's charge runs down to slice 5, so its THRU at the readout-window start is read correctly |
| labels artifact | **no labels tag was written for `smx25`.** Doc pdhd/19 §9.4 wrote one for its adopted items. Here the only adopted item, `028084_18/17`, keeps the owner's `owner_review` (THRU), which supersedes the adopted verdict in every census, so a labels tag would carry a call no number uses. The record `pdhd_stm_michel_smx25_verdicts.json` is the round's artifact |

### 6.3 A defect in the fold tool, found and fixed before scoring

**Symptom.** The first fold of `smx25` turned the THRU control `028084_18/17` into STM_MICHEL, with outcome
*adopted* (both agents called it a stopper), and the record lost the item's `owner_review` block. That block
is the owner's own THRU ruling from doc pdhd/19 §8 ("#46 not likely STM, I guess").

**Root cause.** `stm_michel_scan/campaign/mkowner_record.py` rebuilds an *adopted* item from the chosen
scan. It carried `review`, `owner_smx1` and `calibration` across, but not an `owner_review` written by an
**earlier** round. The census reads `owner_review` first, so it would have scored the agents' call over
the owner's.

**Why it hid.** No item ever adopted before carried an `owner_review`: 2 were adopted in smx20 and 4 in
smx21, none of them owner-ruled. The builder gate (an empty pass must rebuild smx23) cannot reach the
adopted branch.

**Fix.** The adopted branch now carries `owner_review` as well.

**Verification.**
* Rebuilding **smx21** with the fixed tool, from smx20, the `own19` rulings and the h21 round's own
  52 scans, reproduces the committed record **317/317**, with only the path field `owner_review.source`
  masked. Its provenance is byte-identical.
* The empty pass still rebuilds smx23 identically.
* The fixed smx25 differs from the first build **only** in restoring that `owner_review`, which is equal
  to smx23's. Provenance is identical.
* **No committed record is affected**, and the first build was never committed.

### 6.4 Outcomes (`score_smx25.txt`, judged on stopper-or-not, the fold's own rule)

| group | n | confirmed by both scans | split | against the truth in force |
|---|---|---|---|---|
| decision, stratum M | 7 | 6 stoppers | **1: `029107_21/65`** (STM_ONLY / UNCLEAR) | 6 agree |
| decision, stratum N | 7 | 7 stoppers | 0 | 7 agree |
| THRU controls, M | 4 | 3 not-stopper; **`028084_18/17` both STM_MICHEL** (adopted into the agent verdict; the owner's THRU governs) | 0 | 3 agree, **1 disagrees with the owner** |
| THRU controls, N | 4 | 4 not-stopper | 0 | 4 agree |
| stopper controls, M / N | 2 / 2 | 4 stoppers | 0 | 4 agree |

* **13 of 14 recoveries** are stoppers on two further blind looks. Four of the 13 rest on at least one
  *low*-confidence scan: `028084_2/49`, `029107_7/85` and `028084_29/53` on one, `029107_3/99` on both.
* The split item `029107_21/65` is a near-isochronous APA2 track. Its scanners wrote "the end rise may
  be just another bump of a wavy … profile" and "a ~14 cm periodic wave". That is the doc pdhd/24 wave on
  a shape-marginal accept, exactly the population §3.3 describes.
* Inside agreed stoppers, the Michel **kind** moved on some items (for example `029107_7/85`, record
  STM_MICHEL, both scans STM_ONLY; `028084_29/53` the other way). The fold rule keeps the base record for a
  kind difference, so `michel_found` scoring is not re-graded here.

### 6.5 The census on the re-judged record

Reading A is the record rule. Reading B is the pre-registered worst case: the split decision item counts as
THRU. The last row is **not pre-registered** and is shown only as a sensitivity: neither scanner called
the split item THRU, so it is counted as unscored there. All numbers come from the independent recount in
`d25_score_smx25.py`, cross-checked 4/4 on smx23, and are reproduced by the committed graders with the
derived gates (`census_smx25_A.txt`, `census_smx25_B.txt`, `q2_smx25_A/B.txt`).

| APA0 strict | production `h25base` | **`h25k` lever 1** (= `h25kr`) | `h25r` `compare_range_cm` 45 |
|---|---|---|---|
| smx23 = smx25 reading A | 77/0/28/69 — 1.000 / 0.733 | **90/0/15/69 — 1.000 / 0.857** | 80/0/25/69 — 1.000 / 0.762 |
| **smx25 reading B** | 77/0/27/70 — 1.000 / 0.740 | **89/1/15/69 — 0.989 / 0.856** | 80/0/24/70 — 1.000 / 0.769 |
| *split item unscored (sensitivity)* | *77/0/27/69 — 0.740* | *89/0/15/69 — 1.000 / 0.856* | *80/0/24/69 — 0.769* |
| majority, reading B | 79/1/29/71 — 0.988 / 0.731 | 92/2/16/70 — 0.979 / 0.852 | 82/1/26/71 — 0.988 / 0.759 |
| all four APAs, reading B | 96/1/51/108 — 0.990 / 0.653 | 114/2/33/107 — 0.983 / 0.776 | 101/1/46/108 — 0.990 / 0.687 |
| GOLDEN, reading A / B | 33/63 = 0.524 / 33/62 = 0.532 | **37/63 = 0.587 / 36/62 = 0.581** | 34/63 = 0.540 / 34/62 = 0.548 |
| Michel, strict, A / B | 51/2/12/39 / 50/2/12/39 | unchanged | unchanged |

Reading A equals smx23 item for item. The only adopted item is owner-governed, so it moves no truth, and
the split keeps the base record.

### 6.6 The pre-registered decision

| arm | new false positives, APA0 strict, reading B | verdict by the rule |
|---|---|---|
| `h25r` `compare_range_cm` 45 | none | **free on the re-judged record** |
| `h25k` lever 1 | `029107_21/65` | **not free** |
| `h25kr` both | `029107_21/65` | **not free** |

The rule was fixed before the arms, and it is applied as written. Lever 1's case now rests on one item.
Two blind scanners saw that item's stop as a stopper or as unreadable; neither saw a through-going muon.
Thresholds are not re-selected on this tranche.

### 6.7 What the re-judge may and may not conclude

**It may conclude:**
* Lever 1's efficiency gain is not one agent's artefact. 13 of 14 recoveries hold on two further blind
  looks.
* The stopper null held 4/4 and the agent-governed THRU null held 7/7.
* `compare_range_cm` 45 passes its pre-registered test.

**It may not conclude:**
* **That these are the owner's verdicts.** This is an agent pass.
* **That agent confirmation settles it.** The tranche's one owner-ruled item is the calibration point,
  and there both blind agents called the owner's THRU a stopper. `028084_18/17` is also one of the three
  `topology_clears_sparse` false positives of §1.4. On the only item where the owner's answer is known,
  the agents err toward "stopper", which is the lever's own direction. So 13/14 confirmations are weaker
  evidence than they look. Doc pdvd/92 found the same instability near a shape threshold, where the owner's
  own call flipped on a blind re-look.
* **A flip.** What stands between lever 1 and a flip:
  1. the owner's ruling on `029107_21/65` (the split);
  2. the owner's look at `028084_18/17` as a calibration probe of the agents;
  3. optionally, the low-confidence confirmations (`029107_3/99` first);
  4. the owner's decision.

**Rubric points the scanners flagged**, logged and not folded, since the rubric stayed frozen:
* Unfitted `C` rows never carry a dqdx, which conflicts with the "no dqdx ⇒ degenerate row" clause. Five
  of the six scanners raised this.
* The ~10 cm michel/gamma edge (`029107_24/33` sits at 10.02 cm).
* Whether a straight-on stub counts as the Michel once rule 3 has established the stop.
* A fit end at the readout-window edge.
* No threshold for "near-isochronous".

## Files

| what | where |
|---|---|
| compiled-config proofs, whole-config diff | `docs/scan/h25/cfg_proofs.txt`, `cfg_wholecfg_diff.txt` |
| arm TLAs and launcher | `docs/scan/h25/tla_A.txt`, `tla_B.txt`, `tla_K.txt`, `tla_R.txt`, `tla_KR.txt`, `run_arm.sh` |
| pre-registration (written before each launch) | `docs/scan/h25/preregistered.txt` |
| branch gates | `docs/scan/h25/g_h25base.txt`, `g_h25va.txt`, `g_h25vb.txt`, `g_h25k.txt`, `g_h25r.txt`, `g_h25kr.txt` |
| censuses | `docs/scan/h25/census_h25.txt` (§1), `census_h25k.txt` (§5) (d23_grade + d23_apa) |
| movers by name | `docs/scan/h25/d25_movers.py`, `movers_h25va.txt`, `movers_h25vb.txt`, `movers_h25k.txt` |
| §2, the Bragg + Michel fraction | `docs/scan/h25/d25_bragg_michel.py`, `q2_bragg_michel.txt`, `q2_levers.txt` (§5) |
| §3, misses, separation, noise, offline sweeps | `docs/scan/h25/d25_misses.py`, `q3_misses.txt`, `q3_misses_on_h25r.txt` (`h25kr`'s twin) |
| §5, item-level twins | `docs/scan/h25/d25_twin_check.py`, `twin_h25k.txt`, `twin_h25r.txt`, `twin_h25kr.txt` |
| §6, frozen controls | `docs/scan/h25/d25_controls.py`, `controls_frozen.txt` |
| §6, the tranche | `docs/scan/h25/d25_build_smx25.py`; `docs/scan/smx25/key_smx25.tsv` (never shown to a scanner), `items_smx25.txt`, `items_a1.txt`…`items_a6.txt`, `AGENT_TASK.md`, `rubric_v5.sha` |
| §6, frames and audit | `stm_michel_scan/prep-pdhd-h25base` (gitignored payloads), `docs/scan/smx25/check_shots.txt`, `docs/scan/h25/d25_audit.py`, `docs/scan/smx25/audit.txt` |
| §6, the record | `docs/scan/pdhd_stm_michel_smx25_verdicts.json` (317 records, `review_v5` on the 26), `smx25/provenance.json`, `smx25/rulings_empty.json`, `smx25/record_readingB.json` |
| §6.3, the fold-tool fix | `stm_michel_scan/campaign/mkowner_record.py` (the adopted branch keeps `owner_review`) |
| §6, scoring | `docs/scan/h25/d25_score_smx25.py`, `score_smx25.txt`, `census_smx25_A.txt`, `census_smx25_B.txt`, `q2_smx25_A.txt`, `q2_smx25_B.txt`, `sens_smx25_split_unscored.txt`; grader overrides `STM_SCAN_RECORD` / `D25_GATES` in `h23/d23_grade.py`, `h23/d23_apa.py`, `h25/d25_bragg_michel.py` |

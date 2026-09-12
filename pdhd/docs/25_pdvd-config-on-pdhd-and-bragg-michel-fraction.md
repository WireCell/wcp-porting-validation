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
   PDVD's (**AUC 0.73 vs 0.88**). Offline, on production's own payload, two PDHD-only threshold moves
   — `ks_margin` −0.10 and the P1 Michel floors at 5 MeV / 1.5 cm — would recover **+13 stoppers
   for 0 false positives** (0.733 → 0.857). On PDVD the same moves cost 22 false positives, so they
   are PDHD-specific. The last 13 misses are unreadable profiles that no threshold reaches.

**Read-only for production.** No C++ change, no production jsonnet edit, no record or label touched.
Three new arms on a pinned binary, new tags only.

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
#    (arms A and B compiled the same way with PDHD_PR_TLA="$(cat $X/tla_A.txt)" / tla_B.txt: cfg_proofs.txt)

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
moves the fit or the pool, and they are in-sample.

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
   graded as one unit.** **Predicted — not measured —** +13 stoppers at 0 false positives, 0.733 → 0.857
   (strict), which is PDVD's 0.877 territory. The +13 is an offline re-verdict of production's payload.
   The two levers overlap (`028084_20/116` is in both lists). The campaign's own precedent cuts both
   ways: `h21f` was predicted at +17 and delivered +19 (doc pdhd/21 §4). Only an arm gives the number. Both keys exist; this needs no C++. It does need three things:
   * a real arm, since combined arms have broken naive twins before (docs pdhd/21–23);
   * the item list pre-registered;
   * a **blind re-judge** of the 13 recoveries plus the THRU items nearest the cut. Zero false positives
     on 69 through-goers bounds the rate only below ~4 % (95 %), and both thresholds were chosen on
     this record.

   Do not port it to PDVD: there it costs 22 false positives.
2. **From PDVD's delta, take only what Arm B shows is free.** On this record that is **`compare_range_cm` 45 alone** (+3 stoppers, 0 false positives, Arm B). It
should get its own arm before anything is flipped. Its three recoveries are a subset of lever 1's, so
if both are taken they must be graded together. Keep `absorb_bragg_stub`,
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

* **Not** a flip. Nothing here changes production; every threshold in §3.4 is an offline prediction.
* **Not** that PDVD's efficiency advantage is physics. PDVD's scan was not blind; both denominators are
  candidate pools; the two records differ in size (105 vs 276 stoppers on the compared populations).
* **Not** that the correlated wave causes PDHD's weaker shape separation — §3.3 is consistent with it,
  no more.
* **Not** attributed within Arm B beyond what its movers name — it is still a multi-key unit.
* **Not** a statement about APA0, excluded throughout at the owner's request.

## Files

| what | where |
|---|---|
| compiled-config proofs, whole-config diff | `docs/scan/h25/cfg_proofs.txt`, `cfg_wholecfg_diff.txt` |
| arm TLAs and launcher | `docs/scan/h25/tla_A.txt`, `tla_B.txt`, `run_arm.sh` |
| pre-registration (written before each launch) | `docs/scan/h25/preregistered.txt` |
| branch gates | `docs/scan/h25/g_h25base.txt`, `g_h25va.txt`, `g_h25vb.txt` |
| censuses | `docs/scan/h25/census_h25.txt` (d23_grade + d23_apa) |
| movers by name | `docs/scan/h25/d25_movers.py`, `movers_h25va.txt`, `movers_h25vb.txt` |
| §2, the Bragg + Michel fraction | `docs/scan/h25/d25_bragg_michel.py`, `q2_bragg_michel.txt` |
| §3, misses, separation, noise, offline sweeps | `docs/scan/h25/d25_misses.py`, `q3_misses.txt` |

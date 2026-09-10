# 67 — The owner's operating points: `ks_margin`, the 3 cm range-energy veto and `compare_range_cm`, confirmed and flipped

**Status (2026-09-10). Three thresholds moved in PDVD production, each
confirmed on a real arm against the 569-item `smx1a` record; no C++ changed.**
`ks_margin: -0.02`, `michel_range_energy_dis_cm: 3.0` and `compare_range_cm:
45.0` join the `stm_michel_knobs` bag in `pdvd/wct-pr-perevt.jsonnet`.
Against production (`d66vleg`): `is_stm` TP 152 / FP 9 / FN 116 → **177 / 6 / 91**
(purity 0.944 → 0.967, F1 0.709 → 0.785),
`michel_found` TP 132 / FP 22 / FN 20 → **132 / 17 / 20** (F1 0.863 → 0.877).
26 stoppers gained, 3 false positives removed, 0 new
false positives, and **one stopper lost** (`039349_48/63`, STM_ONLY, medium
confidence) — the price the owner's rule accepts for the 45 cm window (§3). PDHD stays OFF (no PDHD scan record). SBND/uBooNE untouched
(`CheckSTM_Michel` is not in their chains; 16 live jobs' compiled config
unchanged).

This is doc 66 §6's owner list, answered. The owner's rule (2026-09-10): "if
the results lead to better reconstructed STM + Michel w.r.t. our scan
results, they would be good to be updated", then "proceed, follow your
recommendation". Two of the seven items turned out to need no arm at all to
be measured exactly (§1); one needed arms, and they reversed doc 65's reading
of it (§3); only one genuinely needs a new scan (§5).

Companion docs: pdvd/65 §2 (the shape-test tables this replaces with real
arms), pdvd/62 §4.4 (the 3 cm row), pdvd/66 §6 (the list), pdvd/56 §8.

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=$HOME/tmp/d67; PIN=$HOME/tmp/d66/libpin            # libWireCellClus md5 c83d6227aca6 (doc 66's binary)
S='survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0'
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
X=pdvd/docs/nf_sp_img_clus/scripts
ARM=d67ks   DET=pdvd SRC=d16vnu JOBS=10 PIN=$PIN PR_TLA="-S stm_michel_extra={$S,ks_margin:-0.02,michel_range_energy_dis_cm:3.0}" $R
ARM=d67cr45 DET=pdvd SRC=d16vnu JOBS=10 PIN=$PIN PR_TLA="-S stm_michel_extra={$S,compare_range_cm:45.0}" $R
ARM=d67cr60 DET=pdvd SRC=d16vnu JOBS=10 PIN=$PIN PR_TLA="-S stm_michel_extra={$S,compare_range_cm:60.0}" $R
ARM=d67v    DET=pdvd SRC=d16vnu JOBS=16 PIN=$PIN PR_TLA="-S stm_michel_extra={$S,ks_margin:-0.02,michel_range_energy_dis_cm:3.0,compare_range_cm:45.0}" $R
cd pdhd/stm_michel_scan
for a in d67ks d67cr45 d67cr60 d67v; do
  ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
      --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv
  python3 census_score.py --prep $W/prep_$a --baseline $HOME/tmp/d66/prep_d66vleg --arm $a
  python3 ../../$X/d63_by_name.py $HOME/tmp/d66/prep_d66vleg $W/prep_$a
done
python3 ../../$X/d67_exact_offline.py $HOME/tmp/d66/prep_d66vleg $HOME/tmp/d65/prep_d65v3   # sec 1: exact re-verdicts
python3 ../../$X/d67_confirm.py $HOME/tmp/d66/prep_d66vleg $W/prep_d67ks  -0.02 3.0       # sec 1: arm == prediction
python3 ../../$X/d67_ks_sweep.py $HOME/tmp/d66/prep_d66vleg $W/prep_d67cr45 $W/prep_d67cr60 $HOME/tmp/d65/prep_d65v3  # sec 3
python3 ../../$X/d67_confirm.py $W/prep_d67cr45 $W/prep_d67v -0.02 3.0                     # sec 4: combined arm == prediction
python3 census_score.py --check                                                           # 0 of 14 differ
```

`prep_d66vleg` is doc 66's leg arm: today's production bag + the survey TLA
on the same pinned binary — the baseline for every row below.

---

## 1. Two knobs that can be measured exactly without an arm

`ks_margin` is read at exactly one place, `CheckSTM_Michel.cxx:1850`
(`if (ks_mu + ks_margin >= ks_flat) reject_bits |= R_SHAPE_FLAT`), and
nothing downstream of the verdict reads `reject_bits` except the persisted
`is_stm = (reject_bits == 0)` and the candidate count (the only earlier reads,
`:1617`/`:1654`, test `R_STOP_UNMATCHED` before the verdict is formed).
`ks_mu`, `ks_flat` and `reject_bits` are all in the payload. So on ANY arm,
the verdict at another margin `m` is exact: `is_stm = 1` iff no other bit is
set and (bit 3 unset or `ks_flat − ks_mu > m`). The same holds for
`michel_range_energy_dis_cm` (read only at `:2715`; `michel_found =
conn_type > 0` at `:2728`): on an arm run at 5 cm, a 3 cm veto adds exactly
the items with `conn_type ∈ {2,3}`, `3 < dis ≤ 5 cm`, `ke_best < 10 MeV`.

Gates for the instrument: at `m = 0` it reproduces every published `is_stm`
on the 547 judged items of `d66vleg` and of `d65v3` (0 mismatches each); at
5 cm the veto rule finds 0 items on the 5 cm arm.

`ks_margin` on production (exact, `d67_exact_offline.py`):

| `ks_margin` | TP | FP | FN | purity | eff | F1 | vs production |
|---:|---:|---:|---:|---:|---:|---:|---|
| **0 (shipped)** | 152 | 9 | 116 | 0.944 | 0.567 | 0.709 | — |
| −0.01 | 160 | 9 | 108 | 0.947 | 0.597 | 0.732 | +8 TP |
| **−0.02** | **171** | **9** | 97 | 0.950 | 0.638 | 0.763 | **+19 TP, 0 new FP** |
| −0.03 | 176 | 10 | 92 | 0.946 | 0.657 | 0.775 | +24 TP, +1 FP |
| −0.05 | 186 | 12 | 82 | 0.939 | 0.694 | 0.798 | +34 TP, +3 FP |

Relaxing the margin can only clear `R_SHAPE_FLAT`, so it never loses a TP;
the question is only where the first FP enters. **−0.02 is the largest
relaxation that adds no false positive** — the bar every flip in docs 57–63
met — which is why it is the operating point, not the best-F1 cell (−0.05).
Both run halves improve at −0.02 (039349: TP 90 → 102, FP 7 → 7;
039252+039253: TP 62 → 69, FP 2 → 2) — supporting, not a hold-out. The 19
recovered stoppers are 10 high- and 9 medium-confidence scan labels:
`039252_17/104`, `039252_2/55`, `039253_0/44`, `039253_1/79`,
`039253_13/73`, `039253_15/45`, `039253_8/28`, `039349_15/23`, `039349_2/38`,
`039349_3/46`, `039349_31/39`, `039349_31/51`, `039349_36/63`,
`039349_53/51`, `039349_54/30`, `039349_55/24`, `039349_58/21`,
`039349_64/24`, `039349_81/41`. The mechanism predates this sweep (doc 55
§15.2, doc 65 §2.1): a stopper carrying a Michel has a weaker fitted rise, and
the KS-vs-muon comparison is the test that rise fails first.

`michel_range_energy_dis_cm` on production (exact):

| dis | TP | FP | FN | purity | F1 | change |
|---:|---:|---:|---:|---:|---:|---|
| **5 cm (shipped)** | 132 | 22 | 20 | 0.857 | 0.863 | — |
| 4 cm | 132 | 18 | 20 | 0.880 | 0.874 | −4 FP |
| **3 cm** | 132 | **17** | 20 | 0.886 | **0.877** | **−5 FP, 0 TP lost** |
| 2 cm | 130 | 16 | 22 | 0.890 | 0.872 | −6 FP, **−2 TP** (`039252_0/75`, `039253_1/36`) |

3 cm is the last row with no TP cost. The five removed objects are 3.2–4.8 cm
from the stop carrying 0.1–8 MeV: `039253_4/91`, `039349_0/28`,
`039349_3/26`, `039349_34/48`, `039349_7/57` (the same five doc 62 §4.4
found on C alone).

**Confirmation arm `d67ks`** (production bag + both keys, pinned binary):
item for item against the prediction — **0 `is_stm` mismatches, 0
`michel_found` mismatches on 566 common items, and no other verdict field
moved** (`d67_confirm.py`, every payload field except the seven the two
knobs own). `is_stm` 152 / 9 / 116 → 171 / 9 / 97; `michel_found` 132 / 22 /
20 → 132 / 17 / 20. The instrument is exact; the numbers above are arm
numbers.

## 2. The whole decision table

Baseline `d66vleg`; "exact" rows are the §1 re-verdict applied to a real
arm's payloads, one confirmation arm away from the same standing as `d67ks`.

| option | how measured | `is_stm` TP / FP / FN | purity | F1 | vs production |
|---|---|---|---:|---:|---|
| production | arm `d66vleg` | 152 / 9 / 116 | 0.944 | 0.709 | — |
| `ks −0.02` + `dis 3` | **arm `d67ks`** | 171 / 9 / 97 | 0.950 | 0.763 | +19 TP; michel −5 FP |
| `compare_range_cm 45` | **arm `d67cr45`** | 160 / 5 / 108 | 0.970 | 0.739 | +11 / −3 TP, −4 FP, 0 new |
| `cr 45` + `ks −0.02` (+ `dis 3`) | exact on `d67cr45`; **confirmed by arm `d67v`** (§4) | 177 / 6 / 91 | 0.967 | 0.785 | +26 / −1 TP, −3 FP, 0 new |
| `compare_range_cm 60` | **arm `d67cr60`** | 165 / 4 / 103 | 0.976 | 0.755 | +18 / −5 TP, +1 / −6 FP |
| `cr 60` + `ks −0.02` | exact on `d67cr60` | 181 / 4 / 87 | 0.978 | 0.799 | +33 / −4 TP, +1 / −6 FP |
| anchor 3 cm | arm `d65v3` (doc 65 §5.1) | 168 / 10 / 100 | 0.944 | 0.753 | +23 / −7 TP, +4 / −3 FP |
| anchor 3 cm + `ks −0.02` | exact on `d65v3` | 192 / 11 / 76 | 0.946 | 0.815 | +43 / −3 TP, +5 / −3 FP |

`compare_range_cm` and the anchor leave `michel_found` identical.

## 3. `compare_range_cm`: the arms reverse doc 65's reading

Doc 65 §2.2 scored the KS window offline on the shape tests alone (F1 0.726 →
0.763 at 60 cm) and flagged it as a three-consumer change: the KS window
(`:1839`), `do_track_comp` (`:1857`, `R_NOT_MUON_PID`) and the
dead-fraction diagnostic (`:1757`). The arms measure all three together, and
the result is not an efficiency trade but a **purity** gain — the only lever
on the list that raises it: 0.944 → 0.970 (45 cm) / 0.976 (60 cm).

- **45 cm alone**: 11 stoppers gained, 4 FPs removed (`039253_14/84`,
  `039349_21/26`, `039349_22/45`, `039349_6/62`), 0 new FPs, 3 stoppers lost
  — `039252_8/32` (STM_MICHEL, high), `039349_48/63` (STM_ONLY, medium),
  `039349_66/78` (STM_ONLY, medium; one of doc 63's three `absorb_bragg_stub`
  recoveries).
- **45 cm + `ks −0.02`**: the relaxed margin returns two of the three; the
  single remaining cost is **`039349_48/63`** (STM_ONLY, medium). 26 gained,
  3 FPs removed (`039349_21/26` comes back as an FP at −0.02).
- **60 cm**: two more FPs removed (`039349_26/51`, `039349_38/60`) but one
  new (`039349_6/61`, high) and five stoppers lost (`039252_1/30`,
  `039253_16/109`, `039349_48/63`, `039349_82/52`, `039349_9/20`); with
  `ks −0.02`, four remain lost. Not taken: the 45 cm row meets the bar
  better and is the smaller move from the shipped 35.

The choice of 45 cm + `ks −0.02` over `ks −0.02` alone is the owner's
principle applied plainly: +7 stoppers and −3 FPs for one medium-confidence
STM_ONLY item.

## 4. The combined arm (`d67v`)

Production bag + all three keys, on the same pinned binary. The prediction
for it is the §1 re-verdict applied to the real `d67cr45` arm (which already
carries the 45 cm window), and it lands **item for item: 0 `is_stm`
mismatches, 0 `michel_found` mismatches on 566 common items, and no other
verdict field moved** (`d67_confirm.py prep_d67cr45 prep_d67v -0.02 3.0`).

| | `is_stm` TP | FP | FN | purity | eff | F1 | `michel_found` TP / FP / FN | F1 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| production (`d66vleg`) | 152 | 9 | 116 | 0.944 | 0.567 | 0.709 | 132 / 22 / 20 | 0.863 |
| **`d67v`** | **177** | **6** | 91 | **0.967** | 0.660 | **0.785** | **132 / 17 / 20** | **0.877** |

`is_stm` moves on 30 of 566 items. Gained (26): `039252_17/104`,
`039252_2/55`, `039253_0/44`, `039253_1/79`, `039253_1/98`, `039253_13/73`,
`039253_15/45`, `039253_17/115`, `039253_4/79`, `039253_8/28`,
`039349_15/23`, `039349_18/33`, `039349_29/45`, `039349_3/46`,
`039349_31/39`, `039349_31/51`, `039349_36/63`, `039349_5/64`,
`039349_53/51`, `039349_54/30`, `039349_58/21`, `039349_60/40`,
`039349_64/24`, `039349_76/25`, `039349_77/63`, `039349_81/41`
(`039349_18/33` is one of doc 63's three class-F stubs left over the 20°
angle — recovered by the window, not by the stub). FPs removed (3):
`039253_14/84`, `039349_22/45` (the THRU item doc 66's guard also removed),
`039349_6/62`. New FPs: none. Lost (1): `039349_48/63`. `michel_found`: the
five 3–5 cm objects of §1 removed, nothing else moved; Michel attachment
(role 3 225, no role 41) unchanged. `census_score.py --check` 0 of 14.

This is the bag now in `pdvd/wct-pr-perevt.jsonnet` and the baseline for any
later round (`prep_d67v`).

## 5. What is not flipped, and what each would need

| item | status | what it needs |
|---|---|---|
| `bragg_peak_anchor` 3 cm (doc 65 §5.1) | **FLIPPED by doc pdvd/68** on the owner's scan (`is_stm` 176/7 → 197/7 on the merged record); the rest of this row is superseded | it swaps the FP set (+5 / −3 even with `ks −0.02`) and reaches the same profiles as §1 and §3; if wanted, a combined arm on top of this production bag, and a look at `039252_2/79`, `039349_14/22`, `039349_24/23`, `039349_43/66`, `039349_72/11` (new FPs) and `039253_3/66`, `039253_6/85`, `039349_76/75` (lost) first |
| `dx_norm_length` 4 mm (doc 65 §4) | measured on the owner's scan (doc pdvd/68 §4): better than this doc's production alone, **worse on top of the anchor** (`is_stm` F1 0.807 → 0.792); OFF — its file is also shared with `TaggerCheckNeutrino` | **the one item that needs a scan**: it changes which clusters the tagger flags, 35 candidates the record never saw |
| `stop_local_residual_cm` (doc 62 §4.3) | **dropped by the owner (2026-09-10)** | measured worse on Michel purity (+2 / −1 TP for +5 FP on top of B); off the decision list. The C++ knob stays, default OFF |
| `publish_other_arms` in production (doc 64 §5) | unchanged | a display choice (+3 % `stm_michel_pts` rows, no verdict effect); not a physics decision |

## 6. Gates

| gate | result |
|---|---|
| C++ | untouched — every arm ran doc 66's binary, whose byte-identical gates stand |
| binary pin | `libWireCellClus.so` md5 `c83d6227aca6` before and after all four arms |
| compiled-config proof (each arm's TLA) | `CheckSTM_Michel` config differs from the leg's only by the arm's own key(s) |
| exact-instrument gates (§1) | m = 0 reproduces every published `is_stm` (0 of 547 on `d66vleg`, `d65v3`, `d67cr45`, `d67cr60`); the 5 cm veto rule finds 0 on the 5 cm arm |
| arm == prediction | `d67ks` vs `d66vleg` and `d67v` vs `d67cr45`: 0 / 0 mismatches on 566, no other verdict field moved |
| `census_score.py --check` | 0 of 14 differ |
| flip-equivalence (compiled JSON: edited production vs pre-flip file + `d67v`'s override) | **0 lines** |
| true OFF path (the same force-off override — `ks_margin 0`, `dis 5`, `compare_range_cm 35` — on the pre-flip and the edited file) | **0 lines** |
| `abtest/compile_all_cfg.sh` + `cmp_cfg.sh`, before vs after the flip (`$W/cfg_all_pre`, `$W/cfg_all_post`) | 16 live jobs NORMDIFF 0, **OVERALL PASS** (SBND/uBooNE untouched) |
| event `039252_11` | reported INCOMPLETE by the runner on every arm: it has no STM candidate on the baseline either (no `CheckSTM_Michel: … candidate(s)` line) |

## 7. Doc 56 update

Status block and Order paragraph: ten knobs now PDVD production; the owner's
list reduced to the §5 table. Scripts committed: `scripts/d67_exact_offline.py`,
`scripts/d67_confirm.py`, `scripts/d67_ks_sweep.py`.

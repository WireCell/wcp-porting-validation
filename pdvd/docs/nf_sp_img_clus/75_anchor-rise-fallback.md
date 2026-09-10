# 75 — P1b: the Bragg rise the peak anchor threw away

This is P1b of doc 70 §3.4 / §6. The owner asked for each proposal in its own file; this is the P1b file.

**Status (2026-09-10): DONE; in PDVD production.** §1–§4 were written before any arm launched; the results are in §5–§9.
- **On: `bragg_anchor_geo_fallback: true`** (rise 1.5, the C++ default). When the 3 cm peak anchor rejects a profile on a shape bit although its own peak is prominent, and the shape tests pass at the geometric origin, the geometric reading stands.
  - `is_stm` 225 / 7 / 51 → **230 / 7 / 46**: all four of doc 70 §3.4's targets (`039253_0/44`, `039253_6/85`, `039349_15/23`, `039349_76/75`) and `039349_36/46`. 0 new FP, 0 lost TP.
  - `michel_found` (136 / 12 / 22), every stop, every pin and every Michel tag identical.
  - Without the threshold (`p75vfb0`): the same five plus 3 THRU false positives.
- **Correction to doc 65 §5.1 / doc 70 §3.4 (§2.2):** the one-sided "rise precondition" is backwards on production; the anchor's gains have the less prominent peaks. The working rule is two-sided.
- **Toolkit `567a7232`.** Confirmation arm `p75vprod` bit-identical to the graded arm (§8).

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
# the production re-measurement of sec 2 (read-only: doc 74's bare-production arm p74vprod2) -> /home/xqian/tmp/p75/sizing.txt
python3 $X/d75_sizing.py --json /home/xqian/tmp/p75/pred.json
# build (toolkit clus/), the tests, the pin
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild && ./build/clus/wcdoctest-clus
cp -a ../local/lib /home/xqian/tmp/p75/libpin_p75
# wave 1 (6 arms, bare production config; detached), gates, grading
nohup bash $X/d75_arms.sh > /home/xqian/tmp/p75/arms_wave1.log 2>&1 < /dev/null & disown
$X/d75_gates.sh 2>&1 | tee /home/xqian/tmp/p75/gates.log
```

## 1. The question

Doc 68 §3 put the 3 cm peak anchor (T7, doc 65) into PDVD production: the verdict's two dQ/dx shape tests read the live profile from a residual-range origin set at the peak of a 5-point running mean, `end_L = L[peak] + 0.2 cm`, rows past the peak dropped. It gained 28 stoppers and swapped three false positives for three, and it **lost 7 true stoppers** the owner confirmed: `039253_0/44`, `039253_13/73`, `039253_3/66`, `039253_6/85`, `039349_15/23`, `039349_36/46`, `039349_76/75`. Doc 65 §5.1 named the refinement it did not build: a "rise precondition" — re-origin the profile only when the anchored peak sits above the plateau by some factor, otherwise the geometric origin stands.

Doc 70 §3.4 scoped it: three of the seven carry a Michel and are inside P1's population (`039253_13/73` and `039253_3/66` are back through P1; `039349_36/46`'s Michel is 4.6 MeV, under P1's 10 MeV floor). **The four without topology — `039253_0/44`, `039253_6/85`, `039349_15/23`, `039349_76/75` — are P1b's target.** Build criterion, stated there: size it offline from the payload profiles first; build only if it recovers at least 2 of the 4 without a new false positive.

## 2. Re-measured on today's production

Measured on `p74vprod2`, doc 74's bare-production confirmation arm (toolkit `54e6ab99`, every knob at its production value). `d75_sizing.py` is an offline twin of the verdict-stage shape tests: the 0.15 MIP live filter, the T7 anchor, `stm_michel_bragg_contrast`, the plateau window, and `kslike_compare` with `ks_margin −0.02` over `compare_range_cm 45`; the other reject bits and P1's `topology_cleared_bits` are taken from the payload.

**Validation.** The twin reproduces the published `is_stm` on **564 of 568** candidates and the anchor shift to 0.01 cm on 543. The four it misses are named — `039252_17/80`, `039349_10/58`, `039349_32/56`, `039349_76/75` — and have one cause: the payload's `L` is rounded to 0.01 cm, so a row that sits at exactly rr = 3.0, 20 or 40 cm can fall on the other side of a window edge than in the C++ (`039349_76/75`: the anchored plateau window reads 0.63 MIP in the twin and 0.58 in the C++, one row apart, and `plateau_off_mip` decides it). This is also the likely cause of the 15 "unexplained" geometric-contrast outliers doc 74 §2 / §9 reported. The twin is the build/no-build instrument; the arm decides the named four.

### 2.1 What happens on the four

At the geometric origin all four pass every shape test; the anchored reading fails `shape_flat` on three and `plateau_off_mip` on `039349_76/75`:

| item | anchor shift (cm) | rows dropped | anchored peak / anchored plateau | dropped rows' median / peak | last row / peak | anchored contrast / expected | anchored ks_flat − ks_mu | geometric ks_flat − ks_mu |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `039253_0/44` | 1.76 | 3 | 1.71 | 1.03 | 0.55 | 1.48 / 1.96 | −0.037 | −0.016 |
| `039253_6/85` | 2.80 | 5 | 1.87 | 0.60 | 0.60 | 1.24 / 1.72 | −0.030 | +0.016 |
| `039349_15/23` | 1.63 | 3 | 1.81 | 0.98 | 1.01 | 1.62 / 1.96 | −0.059 | −0.018 |
| `039349_76/75` | 2.66 | — | 1.87 | — | — | 1.65 / 1.96 (plateau 0.58 MIP) | +0.070 | +0.066 |
| `039349_36/46` (P1-side) | 1.60 | 3 | 2.02 | 1.06 | 0.59 | 1.71 / 1.96 | −0.023 | +0.028 |

(`shape_flat` fires when ks_flat − ks_mu ≤ −0.02.) The pattern: **the Bragg rise runs to the fit's last row.** The end row is a partial step (0.76 cm on `039253_0/44`) with a low dQ/dx (0.55–0.60 of the peak on three of them), and it pulls the 5-point running mean's maximum 2–3 rows back from the end. The anchor then drops the true last 3–5 rows — the top of the rise, with a median at or above the "peak" — and the anchored profile reads flatter than the muon template. The geometric origin keeps those rows and passes.

### 2.2 Doc 65's precondition as written is backwards on production

"Anchor only when the anchored peak sits above the plateau by a factor" is the opposite of what separates these items. Among judged candidates whose `is_stm` differs between the two origins:

| population | n | anchored peak / anchored plateau: min / p10 / p50 / p90 / max |
|---|---:|---|
| anchor accepts, geometric rejects (the anchor's gains) | 34 | 0.60 / 0.76 / 1.22 / 1.76 / 3.41 |
| anchor rejects, geometric accepts — the 5 stoppers above | 5 | 1.68–2.02 |
| anchor rejects, geometric accepts — the 3 THRU the anchor removed (`039252_8/93`, `039349_21/26`, `039349_26/51`) | 3 | 1.26 / 1.32 / 1.47 |

The lost stoppers have the **most** prominent peaks. Applied one-sidedly ("keep the anchor only if prominent"), the precondition removes the anchor from its own gains: at 1.5× it recovers none of the four and loses 26 true stoppers (`d75_sizing.py` §3a). The other one-sided readings do the same — anchor only if what it dropped is a real fall (`drop_med / peak ≤ F`): F = 0.8 recovers 3 and loses 9; anchor only if the last row is below the peak: X = 0.5 recovers 4, loses 21, adds 3 FPs (§3b, §3c). The anchor's gains and its losses have the same distribution of every one-sided measure tried. Two modified anchors (skip the end row in the running mean; a 3-point mean) move 200–300 items' shifts and lose 8–16 stoppers.

### 2.3 What does separate: a two-sided reading

Keep the anchored verdict; when it **rejects on a shape bit** and the **geometric reading passes**, let the geometric reading stand **if the anchored peak is prominent** (winning 5-point mean ≥ T × anchored plateau median). On this record:

| T | `is_stm` TP / FP / FN | gained | new FP | lost |
|---:|---|---|---|---|
| 0 (no threshold) | 230 / 12 / 46 | the 5 stoppers | `039252_8/93`, `039349_21/26`, `039349_26/51` | — |
| 1.3 | 230 / 11 / 46 | the 5 | `039252_8/93`, `039349_26/51` | — |
| 1.4 | 230 / 10 / 46 | the 5 | `039349_26/51` | — |
| **1.5–1.6** | **230 / 9 / 46** | `039253_0/44`, `039253_6/85`, `039349_15/23`, `039349_36/46` (+ `039349_10/58`, a twin artefact: its production anchor did not fire) | **none** | **none** |
| 1.7 | 229 / 9 / 47 | four | none | none |
| 2.0 | 226 / 9 / 50 | `039349_36/46` | none | none |

Production reads 225 / 9 / 51 on the twin against `census_score.py`'s 225 / 7 / 51 on the same arm: the difference is exactly the twin's four named misses (two THRU items the twin passes, `039252_17/80` and `039349_32/56`, plus `039349_10/58` and `039349_76/75` swapping places between TP and FN). The movers are what matter, and the twin is read for those. The rule can only clear bits, so it loses nothing by construction. **The gap between the last THRU (1.47) and the first stopper (1.68) is 0.2 and was found on this record with three negatives**: the threshold's default, 1.5, is chosen here, and `p75vfb0` (T = 0) measures exactly what it buys. `039349_76/75` is twin-uncertain (§2): if the C++ anchored reading sets `plateau_off_mip`, as the payload says, the fallback fires on it (prominence 1.87, geometric plateau 0.73 MIP) — the fifth gain.

Doc 70's criterion (≥ 2 of 4, 0 new FP) is met: **build**. The mechanism is not doc 65's precondition; doc 70 §3.4 gets a correction, as §4.2 did for P3.

## 3. Design (toolkit `clus/src/CheckSTM_Michel.cxx`)

Two knobs, both inert unless the first is on:
- **`bragg_anchor_geo_fallback`** (bool, C++ default false).
- **`bragg_anchor_rise_min`** (double, 1.5): the anchored peak's winning 5-point mean, in units of the anchored plateau median.

The block runs right after the KS test, inside the `!prof.empty()` branch, before doc 66's profile geometry. Conditions, all required: the knob is on; the muon table exists; the anchor fired (`bragg_anchor_shift > 0`); the anchored Bragg reading is valid with a positive plateau; `peak_mean ≥ rise_min × plateau_med`; and `reject_bits` has any of `R_NO_BRAGG | R_SHAPE_FLAT | R_PLATEAU_OFF_MIP | R_PROFILE_SPARSE`. Then the contrast, the plateau window and the KS pair are computed again on the **geometric** live profile (a copy taken before the anchor block), with the very same expressions, duplicated in place (CLAUDE.md M10). If that reading sets none of the four bits: the four bits are cleared, `rec.bragg`, `ks_mu`, `ks_flat`, `ratio_mu`, `ratio_flat` take the geometric values (so the branches, doc 66's `plateau_med` reader and the continuation guard all see the reading that produced the verdict), and **`bragg_anchor_fallback`** = 1 — a new `T_stm_michel` branch written only when the knob is on (the `stop_move_p3_bits` pattern), also added to `prep_stm_michel_scan.py`. A DEBUG line prints both readings on every evaluation.

Deliberately unchanged: `do_track_comp` (the template PID) and the dead-volume probe keep the anchored profile; the chain walk never saw the anchor; `michel_found` cannot move (nothing between this block and the Michel search reads the four bits; P1's topology rule runs later and only clears). Off ⇒ a copied profile and an unused double, and the block never runs: byte-identical.

## 4. Pre-stated criteria and predictions (written before wave 1 launched)

Arms (bare production config, `d75_arms.sh`, detached; JOBS 4, a peer had 6 wire-cell jobs live):
- `p75vleg` / `p75hleg`: the P74 pin (production, toolkit `54e6ab99`, `libWireCellClus` md5 `566dc517`).
- `p75voff` / `p75hoff`: the P75 pin (`02557b8d`), knob off.
- `p75vfb`: `bragg_anchor_geo_fallback: true` (rise 1.5).
- `p75vfb0`: `bragg_anchor_geo_fallback: true, bragg_anchor_rise_min: 0` — the two-sided rule with no threshold.

**Gates** (all must hold before any flip):
1. OFF gate, byte-identical: `p75vleg` ↔ `p75voff` (120 events) and `p75hleg` ↔ `p75hoff` (61): zip member content, calib md5, every `T_stm_michel` branch and point row. Stale-baseline: `p74vprod2` ≡ `p75vleg`.
2. Compiled-config proof: the keys appear in the ON arms only (`/home/xqian/tmp/p75/cfg/`).
3. Every `bragg_anchor_fallback` fire named and explained against `d75_sizing.py`'s prediction; a fire the twin did not predict is explained from the C++ DEBUG line or the round stops (CLAUDE.md §5.5).

**Predictions** (from §2; `/home/xqian/tmp/p75/pred.json`):
- **`p75vfb`:** fires on `039253_0/44`, `039253_6/85`, `039349_15/23`, `039349_36/46`, and probably `039349_76/75`; **not** on `039349_10/58` (its production anchor did not fire). `is_stm` **+4 (or +5) TP, 0 new FP, 0 lost**; `michel_found` bit-identical; every stop and pin identical (the rule is verdict-only).
- **`p75vfb0`:** the same fires plus `039252_8/93`, `039349_21/26`, `039349_26/51` (THRU → **3 new FPs**) and `039349_27/41` (fires, `is_stm` stays 0: `stop_near_boundary`).
- *Correction to the fire lists, 2026-09-10, after the first DEBUG lines of `p75vfb` and before any grading:* the C++ evaluates the fallback on the raw anchored bits, before P1's topology clearance, so it also fires on items P1 already accepts (`039253_3/66` at rise 1.5; `039253_13/73` too at rise 0) — there it changes the recorded reading and nothing else. `d75_sizing.py`'s fire list now uses the same semantics (`pred.json`; the first version is kept as `pred_v1.json`). The `is_stm` predictions above are unchanged.

**Flip criteria** (docs 71–74's bar):
- **Gain:** ≥ 2 of the 4 named stoppers recovered as `is_stm` TPs.
- **Guards:** 0 new `is_stm` FP on judged items; 0 `is_stm` TP lost; `michel_found` bit-identical; no pin moved; every fire explained; OFF gate PASS on both detectors; `census_score.py --check` 0 of 14.
- `p75vfb0` is not a flip candidate; it is reported as the measurement of the threshold.
- A flip is one key in `pdvd/wct-pr-perevt.jsonnet` with the three compiled-config proofs, then a bare-production confirmation arm. PDHD stays OFF (no PDHD scan record).

## 5. What was built (toolkit `567a7232`)

`clus/src/CheckSTM_Michel.cxx`, exactly §3: the two knobs (configure, `default_configuration`, members), a copy of the geometric live profile and the anchor's winning mean hoisted out of the anchor block, the fallback block after the KS test, the `bragg_anchor_fallback` record field and its knob-gated branch, a DEBUG line per evaluation. `doctest_check_stm_michel_defaults.cxx` pins both keys; `wcdoctest-clus` passes (23665 assertions). `prep_stm_michel_scan.py` carries the new scalar. Freshness proof: `libWireCellClus.so` 14:27:53 against the source's 14:27:14; the P75 pin is `02557b8d`, the P74 pin `566dc517` (today's production, `54e6ab99`).

## 6. Gates

`d75_gates.sh` → `/home/xqian/tmp/p75/gates.log`. Completeness: all six arms finished under their DONE markers (`arms_wave1.log`, `ALL_DONE`), 120/120 PDVD and 61/61 PDHD events with both tracking files, 0 loader deaths, both pins unchanged before and after. The PDVD runners' rc=1 is the known `039252_11`, which writes no candidate on every arm.

| gate | result |
|---|---|
| OFF, PDVD (`p75vleg` P74 pin ↔ `p75voff` P75 pin) | `mabc-pr.zip` member content 120/120 same; calib json 119/119 same; `T_stm_michel` **578/578 candidates bit-identical on all 139 branches**, 578/578 identical point rows |
| OFF, PDHD (`p75hleg` ↔ `p75hoff`) | 61/61 zips; 61/61 calib; **325/325 × 130 branches**, points identical |
| stale baseline (`p74vprod2` ↔ `p75vleg`; `p74hoff` ↔ `p75hleg`) | identical on both detectors |
| compiled config | `bragg_anchor_geo_fallback` absent from the OFF compile, present in `fb` and `fb0` (`/home/xqian/tmp/p75/cfg/`) |
| ON arms vs `p75voff` | zips and calib identical on all 120 events (the rule touches `T_stm_michel` only); `p75vfb` 572/578 bit-identical, `p75vfb0` 567/578; the moved branches are the reading's own (`contrast`, `contrast_expected`, `plateau_med`, `ks_*`, `ratio_*`, `reject_bits`, `is_stm`, `n_tail`, `n_plateau`) plus doc 66's `tail_med` and `chain_support_min`, which read `rec.bragg`; **no point geometry and no role label moved** on any candidate |
| `census_score.py --check` | 0 of 14 |

The C++'s own account of every fire in `p75vfb` (20 evaluations, 6 with the geometric reading standing):

```
039253_0  cluster 44 shift 1.76 cm peak/plateau 1.72 | anchored contrast 1.48/1.96 ks 0.069/0.032 bits 8    | geometric contrast 1.76/1.98 ks 0.060/0.044 bits 0 -> geometric stands
039253_3  cluster 66 shift 1.73 cm peak/plateau 2.35 | anchored contrast 1.84/1.96 ks 0.148/0.092 bits 8    | geometric contrast 2.41/2.00 ks 0.107/0.137 bits 0 -> geometric stands
039253_6  cluster 85 shift 2.80 cm peak/plateau 1.87 | anchored contrast 1.24/1.72 ks 0.104/0.074 bits 8    | geometric contrast 1.43/1.71 ks 0.078/0.094 bits 0 -> geometric stands
039349_15 cluster 23 shift 1.63 cm peak/plateau 1.82 | anchored contrast 1.63/1.96 ks 0.100/0.041 bits 8    | geometric contrast 1.77/2.05 ks 0.082/0.064 bits 0 -> geometric stands
039349_36 cluster 46 shift 1.60 cm peak/plateau 2.02 | anchored contrast 1.71/1.96 ks 0.088/0.065 bits 8    | geometric contrast 2.17/2.06 ks 0.066/0.094 bits 0 -> geometric stands
039349_76 cluster 75 shift 2.66 cm peak/plateau 2.04 | anchored contrast 1.80/1.96 ks 0.074/0.143 bits 1024 | geometric contrast 1.38/1.87 ks 0.069/0.135 bits 0 -> geometric stands
039253_0  cluster 40 shift 2.20 cm peak/plateau 1.54 | anchored contrast 1.82/1.96 ks 0.119/0.187 bits 1024 | geometric contrast 0.85/2.22 ks 0.110/0.174 bits 1028 -> anchored stands
```

(bits: 8 = `shape_flat`, 1024 = `plateau_off_mip`, 4 = `no_bragg`.) The last line is the rule refusing: a prominent peak whose geometric reading is worse on both tests.

## 7. Result, by name

### `p75vfb` (rise 1.5): passes every criterion

`is_stm` on the judged items **225 / 7 / 51 → 230 / 7 / 46** (`census_score.py` on the 544 payload items: efficiency 0.815 → 0.833, purity 0.970 unchanged). The five gains, all owner- or medium-confidence stoppers, and nothing else moved:

| item | record | anchored reading (bits) | geometric reading | `is_stm` |
|---|---|---|---|---|
| `039253_0/44` | STM_MICHEL, owner ("Michel not identified") | contrast 1.48 / 1.96, ks_mu 0.069 ≥ ks_flat 0.032 − 0.02 → `shape_flat` | 1.76 / 1.98, ks 0.060 / 0.044, clean | 0 → 1 |
| `039253_6/85` | STM_ONLY, owner | 1.24 / 1.72, ks 0.104 / 0.074 → `shape_flat` | 1.43 / 1.71, ks 0.078 / 0.094, clean | 0 → 1 |
| `039349_15/23` | STM_ONLY, owner | 1.63 / 1.96, ks 0.100 / 0.041 → `shape_flat` | 1.77 / 2.05, ks 0.082 / 0.064, clean | 0 → 1 |
| `039349_36/46` | STM_MICHEL, owner (Michel 4.6 MeV, under P1's floor) | 1.71 / 1.96, ks 0.088 / 0.065 → `shape_flat` | 2.17 / 2.06, ks 0.066 / 0.094, clean | 0 → 1 |
| `039349_76/75` | STM_ONLY, owner | plateau 0.58 MIP → `plateau_off_mip` | plateau 0.73 MIP, contrast 1.38 / 1.87, ks 0.069 / 0.135, clean | 0 → 1 |
| `039253_3/66` | STM_MICHEL, owner (already `is_stm` 1 through P1) | 1.84 / 1.96, ks 0.148 / 0.092 → `shape_flat` (P1 had cleared it) | 2.41 / 2.00, ks 0.107 / 0.137, clean | 1 (the recorded reading changes, `topology_cleared_bits` 8 → 0) |

- **Guards.** 0 new `is_stm` FP; 0 lost TP; `michel_found` 136 / 12 / 22 identical, no `michel_conn_type` transition, no T2c change; the owner's `michel` tags in role 3 stay 215 of 263 and the role-3 contamination set is identical; the 25 pins identical (median 2.67 cm, 9 within 2 cm); every stop, chain and Michel-object field identical on all 578 candidates (`d75_score.py` §A's non-verdict check: 0).
- **Prediction vs arm** (`d75_score.py` §G): 6 fires predicted, 6 fired. Both twin-uncertain items resolved as §2 said they would: `039349_10/58` did not fire (its production anchor never fired: shift 0), `039349_76/75` fired (the C++'s anchored plateau is 0.58 MIP, the twin's 0.63). The `is_stm` gain list is the pre-launch prediction with `10/58` swapped for `76/75` — four of the four targets, plus `039349_36/46`.

### `p75vfb0` (rise 0): the threshold's measurement

212 evaluations, 11 fires. The same five gains, plus exactly the three predicted THRU false positives — `039252_8/93` (peak / plateau 1.32, `no_bragg` anchored), `039349_21/26` (1.26, `no_bragg`+`shape_flat`), `039349_26/51` (1.47, `shape_flat`) — so `is_stm` 230 / **10** / 46; `039349_27/41` fires and keeps `stop_near_boundary` (`is_stm` stays 0); `039253_13/73` changes its recorded reading only. `michel_found`, tags and pins identical. **The threshold buys the three THRU items and costs nothing on this record**; the gap it sits in (1.47 → 1.72 on the arm's own peak / plateau values) is the fragility named in §2.3.

## 8. The flip

One key in `pdvd/wct-pr-perevt.jsonnet`, after `bragg_peak_search_cm: 3.0,`: `bragg_anchor_geo_fallback: true,` with a comment naming the five items and the threshold's provenance. `bragg_anchor_rise_min` is left at the C++ default 1.5 (so the key is inert to add; the comment records it). Proofs (`/home/xqian/tmp/p75/flip/proofs.txt`, `flip_proofs.sh` with `-P`, aborting on a failed compile): **A** flip-equivalence PRE + `-S stm_michel_extra={bragg_anchor_geo_fallback:true}` vs POST **0 lines**; **B** OFF path PRE + `{…:false}` vs POST + `{…:false}` **0 lines**; **C** pre vs post differ by exactly `"bragg_anchor_geo_fallback": true`.

**What production now writes that it did not before:** the `bragg_anchor_fallback` branch on every PDVD candidate (a byte gate against a pre-flip baseline will list it under "NEW branches"), and on the six items above the recorded `contrast`, `contrast_expected`, `plateau_med`, `ks_*`, `ratio_*`, `tail_med` and `chain_support_min` are the geometric reading. Nothing in the point rows, the Bee zip or the calib json moves. PDHD stays OFF (knob available, no PDHD scan record).

**Confirmation arm `p75vprod`** (the flipped file, no TLA, P75 pin; `/home/xqian/tmp/p75/vprod_check.txt`): 120/120 events, 0 loader deaths, pin unchanged; against `p75vfb` **120/120 `mabc-pr.zip` member-identical, calib json identical, 578/578 candidates bit-identical on all 140 `T_stm_michel` branches, every point row identical**; `census_score.py` reads the same 230 / 7 / 46 and 136 / 12 / 22; `--check` 0 of 14. The production baseline prep for the next round is `/home/xqian/tmp/p75/prep_p75vprod`.

## 9. Observations and next

- **The anchor's weak point is the end row.** The fit's last row is a partial step (the chain's total length is not a multiple of 0.6 cm), and its dQ/dx is systematically low: on the anchor-fired judged items the last row reads p50 0.68 of the winning peak mean. On a rise that runs to the end this pulls the 5-point maximum back and the anchor throws the rise away. Excluding the end row from the running mean would be the general fix, but it re-anchors 295 of 568 candidates and, sized offline, loses 8 stoppers for 5 (§2.2); the two-sided rule reaches the same five without touching the rest. Not built.
- **The offline twin's boundary trap.** The payload's `L` is rounded to 0.01 cm; a row at exactly rr = 3.0 / 20 / 40 cm falls on the wrong side of a window edge in about 1 % of verdicts (4 of 568 here) and in far more contrasts (the 15 doc 74 could not explain). Any future twin should carry the rows' side of each edge from the C++, or read the C++'s own DEBUG line.
- **The two-sided rule has a threshold found on this record with three negatives** (§2.3). `p75vfb0` shows exactly what it buys. A future scan that adds THRU items with a prominent end peak (peak / plateau > 1.5, `shape_flat` anchored, clean geometric) is the test that would move it.
- **What production now writes:** the `bragg_anchor_fallback` branch; on six items the recorded reading is the geometric one (§8).
- **Next: P5** (doc 70 §2.3: `stm_trackfitting_config_file`, jsonnet only), then the owner's review of the PDVD STM chain against the record (doc 77).

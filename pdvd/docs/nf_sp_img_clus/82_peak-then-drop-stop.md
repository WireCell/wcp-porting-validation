# 82 — Doc 78 action item 2: the peak-then-drop stop

The owner asked (2026-09-10, after docs 79 and 80) to proceed with doc 78's action item 2 — *"a peak-then-drop stop mover: after a Bragg peak ≥ 1.4 × plateau, a tail whose median falls to ≤ f × the peak (not the plateau) with a bend ≥ 15° at the drop row marks the stop"*. Target: the 14 hot-tail items of doc 78 §2.3, behind which sit 13 missed Michels and 8 missed stoppers.

**Status (2026-09-10). The knob is built, gated and NOT flipped: measured on the arm, the peak-relative tail reading costs 7 new `is_stm` false positives and 4 true positives (two owner-confirmed) for 5 recovered stoppers. The mechanism is reported dead, with the reason — on through-going tracks a large bend at a hot-tail row is common, so no bend threshold separates. Doc 78's side-sweep `split_kink_min_deg: 10.0` IS now PDVD production: four stops move, the verdict census is identical on every class, and the mean distance from the scanner's own pin falls 4.76 → 4.33 cm (within 1 cm: 3 → 5). Byte-identical OFF gates PASS on both ProtoDUNEs (PDVD 596/596 × 140 branches and every point row, PDHD 325/325 × 130). Doc 78 item 2 is corrected in three places.**

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
mkdir -p /home/xqian/tmp/p82
# the sizing, read-only on the production arm (doc 79's confirmation prep)
python3 $X/d82_sizing.py --prep /home/xqian/tmp/p79/prep_p79vprod > /home/xqian/tmp/p82/sizing.txt
# the arms (toolkit 125d9f57 + this round's edit; full local/lib pin)
PIN=/home/xqian/tmp/p82/libpin_p82 $X/d82_arms.sh
# the gates
bash $X/d82_gates.sh | tee /home/xqian/tmp/p82/gates.log
```

Predictions were written to `/home/xqian/tmp/p82/pred.txt` **before** any arm ran.

## 1. The rule as doc 78 wrote it is not a collapse test

Production's two stop movers (doc 57 retreat, doc 58 split) call a tail collapsed when its median falls below `collapse_frac` = 0.5 × the **plateau**. Doc 78 item 2 proposed judging it against the **peak** instead. Sized on the production arm (`d82_sizing.py` §1, 585 payloads, the smx1a+smx3+smx4 record):

| *f* (tail ≤ *f* × peak) | bend ≥ | pin_rr items reached (of 21) | within 2 cm of the pin | THRU control fires (of 259) |
|---|---:|---:|---:|---:|
| production (plateau only) | 15° | 6 | 5 | 17 |
| 0.5 | 15° | 16 | 11 | 63 |
| 0.6 | 25° | 13 | 8 | 45 |
| 0.7 | 15° | 16 | 10 | 91 |

The arithmetic is the reason. With peak/plateau in 1.4–3 — which is what the peak gate already requires — `tail ≤ 0.7 × peak` is `tail ≤ 0.98–2.1 × plateau`. **A muon that simply keeps going passes it.** The literal rule is not a collapse test at all; it is a threshold that rises with the peak it is meant to be anchored against. Doc 75 hit the same shape of problem with doc 65's anchor precondition, and the answer is the same: build, but not the literal rule.

## 2. What actually keeps the stop where it is

Two measurements, both read off the same production arm.

### 2.1 On 14 of the 21 items, neither mover was ever tried

`CheckSTM_Michel.cxx:2270-2273` and `:2328-2332` run the retreat and the split **only** when the chain is *not* already Bragg-confirmed — unless `michel_collinear_split` (doc 74's P3 knob) is on, and it is **OFF** in PDVD production. `bragg_confirmed` is the contrast test alone (`:2210`), read at the geometric origin.

Of the 21 `pin_rr` items production did not move, **14 are Bragg-confirmed**. Doc 78 §2.3 reads the collapse test at the scanner's pin on all 21 and reports "tail NOT collapsed" as the cause; on those 14 the collapse test was never reached. It is a symptom, not the blocker. Doc 78 §2.3 is corrected accordingly.

The same applies to doc 78's suggested `retreat_peak_frac` 1.25 for `039253_3/61`: that item is Bragg-confirmed, so the retreat is never tried on it and the peak fraction is not what blocks it. **No arm; the suggestion is moot.**

### 2.2 Bragg confirmation is a usable control gate

| scanner verdict | payloads | Bragg-confirmed |
|---|---:|---:|
| STM_ONLY | 117 | 110 (94 %) |
| STM_MICHEL | 159 | 129 (81 %) |
| MESSY | 23 | 7 (30 %) |
| **THRU** | **264** | **40 (15 %)** |

Requiring the chain to be Bragg-confirmed before the relaxed tail test may fire shrinks the through-going control **6.5×**, from 259 items to 40, while keeping 14 of the 21 signal items in reach. It converts the risk from "new false positives across a 259-item control, uncheckable offline" into "moves on a bounded, named set of items the record already grades".

### 2.3 What the offline twin is worth

A fired mover cannot be re-read from a payload — the dropped tail is not in the published profile. The converse is checkable, and it is the validation: on the **565 payloads where no production mover fired**, this reading of the production rule says one would have fired on **14 (2.5 %)** — 4 of 288 Bragg-confirmed (where production never tried), 10 of 277 not. Production's split fired on 5 of that second group. So the twin **over-counts by about 2×**, from the graph constraints it cannot see (`fits() >= 4`, `break_segment` success, `stop_split_max` 1, an on-chain vertex for the retreat). Every control number in this doc carries that factor.

One asymmetry is matched explicitly: production runs `retreat_tail_strict: true`, so the retreat's tail excludes the boundary row, while the split has no such flag and includes it. `d82_sizing.py --strict` picks the reading; the default is the split's, since the split is the mechanism that reaches a row with no vertex on it.

## 3. The design

**A peak-relative alternative for the collapsed-tail test, carried by the row's own bend, on a Bragg-confirmed chain.**

Two knobs, both default OFF, threaded into the same two pure functions the retreat and the split already share (`StmMichelFunctions.{h,cxx}` — extended, not forked, following doc 74's `tail_strict`):

- `stop_tail_peak_frac` (double, **0 = off**): a tail also counts as collapsed when `tail_med <= frac * peak`, the peak being the surviving profile's, the same one the existing Bragg-rise gate reads.
- `stop_tail_peak_kink_min_deg` (double, **25.0**): when that reading is what admits the row — the plateau test having failed — the row's own trajectory bend (`stm_michel_row_kink_deg`, doc 58's discriminator) must reach it. A row the plateau test already accepts never sees this bar, which is what leaves the doc 57/58/74 path untouched.

The bend is not decoration. Charge shape alone cannot separate a Michel continuing past the Bragg peak from a muon that keeps going — both leave 0.5–1.9 × plateau behind the peak (the owner's own kink discriminator, `feedback_owner_kink_discriminator`). The bend is what carries the looser tail.

Byte-identical **by construction**, not by re-derivation: the legacy test is the first clause and exits at the same point it always did.

```cpp
const bool collapsed_plateau = tail_med < th.collapse_frac * plateau;
if (!collapsed_plateau && !(th.tail_peak_frac > 0)) break;      // the doc 57/58 reading, verbatim
… the existing peak computation, unchanged …
if (!(peak >= th.peak_frac * plateau)) break;                   // unchanged
double kink = -1;                                               // reached with collapsed_plateau
if (!collapsed_plateau) {                                       // false only when the knob is on
    kink = stm_michel_row_kink_deg(prof, boundary_i, th.dir_window);
    if (!(tail_med <= th.tail_peak_frac * peak && kink >= th.tail_peak_kink_min)) break;
}
```

(`continue` in `stm_michel_stop_split`.) The result struct is written only after every test, so nothing depends on which test exits first. The retreat gained the boundary **index** (it kept only `boundary_L`) and a `dir_window`, so the bend can be measured at the vertex it lands on; both movers share one bend definition (`split_dir_window_cm`).

`stop_move_p3_bits` gains **bit 2** = "the peak-relative reading is what admitted this move", with a DEBUG line carrying tail, peak, plateau and bend. It is persisted whenever any of the four knobs that can set it is on.

The knob reaches the 14 Bragg-confirmed items only in company with `michel_collinear_split: true` — the flip is therefore two keys, and doc 74's measured cost of that knob (one lost `is_stm` TP) is inherited and declared up front.

### 3.1 The operating point, chosen offline

Peak-relative test, Bragg-confirmed chains only, rows inside the chain's last segment:

| *f* | bend ≥ | signal (of 14 Bragg-confirmed) | within 2 cm of the pin | control (of 40 Bragg-confirmed THRU) |
|---|---:|---:|---:|---:|
| 0.5 | 15° | 10 | 8 | 8 |
| 0.5 | 20° | 8 | 6 | 5 |
| **0.5** | **25°** | **8** | **6** | **2** |
| 0.5 | 30° | 6 | 5 | 2 |
| 0.6 | 25° | 8 | 6 | 7 |
| 0.7 | 25° | 8 | 6 | 7 |

`f = 0.5`, bend ≥ 25° is where the control falls to 2 without costing a signal item. Above 25° the signal starts paying; below it the control triples by 15°.

## 4. Built

Toolkit, on `125d9f57` (doc 81, the peer session's round, landed first and is in this binary):

- `clus/inc/WireCellClus/StmMichelFunctions.h`, `clus/src/StmMichelFunctions.cxx`: `tail_peak_frac`, `tail_peak_kink_min` and (retreat only) `dir_window` in both threshold structs; `by_tail_peak` and `last_kink_deg` in the results.
- `clus/src/CheckSTM_Michel.cxx`: `stop_tail_peak_frac`, `stop_tail_peak_kink_min_deg` in `configure` / `default_configuration` / members, both movers wired, bit 2 and the DEBUG lines.
- `clus/test/doctest_stm_michel.cxx`: **7 new cases** — a hot tail (0.9 × plateau = 0.45 × peak) refused with the knob off and taken with it on, on the retreat and on the split; the bend gate refusing the straight version and a bar above the bend it has; a tail above `frac × peak` still refused; and the doc 57/58 collapse cases proved untouched (`by_tail_peak` false, `tail_peak_kink_min` set absurdly high).
- `clus/test/doctest_check_stm_michel_defaults.cxx`: the two new keys pinned.

`wcdoctest-clus` **383/383 pass**. Freshness proof: `local/lib/libWireCellClus.so` 21:06 against sources at 21:03.

The pin is a full `local/lib` snapshot, `/home/xqian/tmp/p82/libpin_p82` (572 files, Clus `5f2c3ede7076`, Root `46bf51057716`) — doc 80's lesson, where a Clus-only pin mixed with a rebuilt Root library killed every event with rc 139.

## 5. Criteria and predictions

Written to `/home/xqian/tmp/p82/pred.txt` before the arms; §8 grades against it item by item.

| arm | extra | purpose |
|---|---|---|
| `p82voff` / `p82hoff` | — | OFF gate vs `p80boff` / `p80bhoff` |
| `p82vcs` | `michel_collinear_split:true` | doc 74's knob alone — the baseline the peak test is added to |
| `p82vtp` | `michel_collinear_split:true, stop_tail_peak_frac:0.5, stop_tail_peak_kink_min_deg:25.0` | **the candidate** |
| `p82vtp0` | the same two doc-82 keys, **no** Bragg gate | negative control — what the Bragg gate buys |
| `p82vk10` | `split_kink_min_deg:10.0` | existing knob, no code (doc 78's other side-sweep) |

Predicted movers on `p82vtp`, at most these 8, all Bragg-confirmed, all in the scanner's 25 `pin_rr` set:

| item | pin_rr | predicted stop | today |
|---|---:|---:|---|
| `039253_17/77` | 10.1 | 4.2 (6.0 cm short) | is_stm 1, Michel found |
| `039253_7/80` | 9.2 | 3.0 (6.2 cm short) | is_stm 1, Michel found |
| `039349_60/40` | 6.1 | 5.6 ✓ | is_stm 1, **no Michel** |
| `039349_7/4` | 5.9 | 4.8 ✓ | is_stm 1, Michel found |
| `039349_30/45` | 3.8 | 4.0 ✓ | is_stm 1, **no Michel** |
| `039252_16/110` | 3.6 | 4.2 ✓ | is_stm 1, **no Michel** |
| `039253_3/29` | 3.5 | 3.0 ✓ | is_stm 1, Michel found |
| `039253_2/13` | 3.4 | 4.6 ✓ | **is_stm 0** |

Predicted **losses, by name rather than discovered**: `039349_68/65` (doc 74 measured `michel_collinear_split` losing this `is_stm` TP), and the list may grow — the relaxed test moves more stops than doc 74's plateau test did. Every moved stop also exposes its Michel to T2c (`moved_stop_michel_guard`), so `michel_found` can lose items that pass today.

Predicted FP risk: `039252_12/114` and `039349_30/41`, the two Bragg-confirmed THRU items that fire offline; ≈1 after the 2× twin correction.

**Not reachable, stated up front**: 7 of the 8 `is_stm`-0 `pin_rr` items are not Bragg-confirmed, so the gated rule cannot touch them (`039349_51/29`, `81/44`, `71/37`, `81/25`, `34/48`, `039252_9/101`, `039349_44/28`). `p82vk10` reaches 3 of them.

**Flip bar** (the owner authorised the PDVD flip on 2026-09-10 if the validations hold): ≥ 3 of the 8 named recovered with the stop within 2 cm of the pin; 0 new `is_stm` FP on judged items; 0 `is_stm` TP lost beyond `039349_68/65` declared-and-accepted, and that one only if the round's net is clearly positive; `michel_found` net ≥ 0 with every loss named; every fire named; OFF gate PASS on both ProtoDUNEs; `census_score.py --check` 0/14. PDHD stays OFF.

## 6. Gates

Every arm is **bare production** plus this round's keys — no d53 survey bag. Wave 1 (`p82v*`) carried the survey and is **void as a gate**: the OFF baselines on disk (`p80boff`, `p80bhoff`, `p79vprod`) are bare, so the comparison read the survey's own effect (it fits extra clusters, so `T_rec_charge` and `T_proj_data` move, and doc 53 measured a 20–25 % mover rate on the muon's own profile branches) as if it were this round's. Its work dirs are kept as a record (CLAUDE.md M13); wave 2 (`p82b*`) is the graded set.

Completeness: 120/120 PDVD and 61/61 PDHD dirs with a `tracking-pr.root` on every arm, 0 loader deaths, `libWireCellClus` md5 `5f2c3ede7076` identical before and after each arm.

**OFF gate, PDVD — `p82boff` vs `p80boff`:** 120/120 Bee zips identical member-by-member (sha256), 119/119 calib json identical, all eight `tracking-pr.root` trees identical on every event, **596/596 candidates bit-identical on all 140 shared branches**, no shared branch moved on any candidate, 0 `is_stm` flips, 596/596 with identical point geometry and no role label moved.

**OFF gate, PDHD — `p82bhoff` vs `p80bhoff`:** 61/61 zips, 61/61 calib json, all eight trees, **325/325 candidates × 130 branches**, 0 flips, 325/325 identical point geometry.

`p80boff` and `p80bhoff` ran on the **pre-doc-81** binary, so this gate re-proves doc 81's OFF path as well as this round's.

## 7. Result

### 7.1 `p82btp`, the candidate: the mechanism does not separate

37 candidates get a different stop move, 36 of them carrying the doc-82 bit. Against the record:

| | production `p79vprod` | `p82btp` |
|---|---:|---:|
| `is_stm` false positives | 13 | **20** |
| `is_stm` false negatives | 43 | 41 |
| `michel_found` false positives | 18 | **30** |
| `michel_found` false negatives | 18 | 15 |
| `michel_found` purity / F1 | 0.882 / 0.882 | 0.820 / 0.859 |
| mean \|stop − pin\| over the 25 overshoot items | 4.76 cm | 3.79 cm |

The stop *is* more accurate — 8 of the 25 within 2 cm of the pin against production's 4. It is paid for with **7 new `is_stm` false positives and 12 new `michel_found` false positives**.

**Gained** (5 stoppers, all judged): `039252_12/123` (STM_ONLY), `039349_51/29` (STM_MICHEL, pin 9.0, stop 1.8 cm from it), `039349_63/55`, `039349_68/63`, `039349_71/37` (pin 6.5, 3.7 cm) — plus `039349_81/54` (MESSY) and `039253_6/94` (unjudged). Michels recovered on `039349_51/29`, `039349_60/40`, `039349_71/37` and `039252_2/79`.

**Lost** (4 stoppers, two at owner confidence):

| item | verdict | conf | what happened |
|---|---|---|---|
| `039252_2/79` | STM_MICHEL | **owner** | the split's row changed, 9.26 → 7.46 cm at a 78.2° bend; `reject_bits` 0 → 4, `is_stm` 1 → 0 (its Michel *is* now found) |
| `039349_48/54` | STM_MICHEL | **owner** | move grew 5.12 → 10.52 cm; `reject_bits` 0 → 32 |
| `039253_12/41` | STM_ONLY | medium | a new 16.50 cm move; `reject_bits` 0 → 1056 |
| `039349_68/65` | STM_ONLY | medium | the doc 74 loss, **predicted by name** — a 20.40 cm move from `michel_collinear_split` alone |

**The eight new false positives, and why no threshold saves the rule.** They are through-going tracks whose stop moved 6–24 cm, and each then assembled a "Michel" of 12–58 MeV:

| item | bend at the row | tail / plateau | the fabricated Michel |
|---|---:|---:|---|
| `039349_45/52` | 68.7° | 1.28 | 45.9 MeV, 24.9 cm |
| `039349_44/66` | 57.0° | 0.18 | 17.1 MeV, 24.0 cm |
| `039349_73/72` | 52.6° | 1.10 | 26.6 MeV, 14.4 cm |
| `039349_30/41` | 52.1° | 0.60 | 38.3 MeV, 16.8 cm |
| `039349_23/28` | 39.5° | 0.98 | 57.5 MeV, 22.2 cm |
| `039349_25/43` | 32.9° | 0.83 | 24.1 MeV, 17.4 cm |
| `039349_32/56` | 27.2° | 2.17 | 28.9 MeV, 10.2 cm |
| `039252_17/90` | (a retreat, 6.03 cm) | 1.28 | 11.7 MeV, 6.0 cm |

Their bends are **27–69°**, straddling and exceeding the signal items' 15–57°. Raising `stop_tail_peak_kink_min_deg` removes the gains before it removes these. That is the round's finding, and the sizing predicted it: over the through-going control the largest bend per item has p50 15°, p90 47°. **On a through-going fit, a sharp bend next to a hot tail is ordinary** — scattering, a delta ray, a crossing track — so the bend cannot carry a tail test that is itself loose. Doc 58's discriminator works precisely because it is paired with a *genuine* collapse; decoupled from it, it discriminates nothing.

### 7.2 `p82btp0`, the negative control: what the Bragg gate buys

30 different moves against the candidate's 37, and `is_stm` FP 19 / michel FP 29 — essentially the same damage. The gate's effect is not protective here: without `michel_collinear_split` the 14 Bragg-confirmed signal items are skipped, so it removes gains (`039349_60/40`, `039349_68/65`, `039349_81/44`) more than fires. **The Bragg gate is not the missing ingredient; the tail test is simply wrong.**

### 7.3 `p82bcs`, `michel_collinear_split` alone: doc 74 reproduced exactly

One changed move in 596 candidates — `039349_68/65`, `is_stm` 1 → 0, a 20.40 cm retreat on a Bragg-confirmed chain. No gains. Census `is_stm` FN 43 → 44, everything else identical. Doc 74's measurement stands on today's production; the knob stays OFF.

### 7.4 `p82bk10`, `split_kink_min_deg: 10` — clean

Four stops move; the verdict census is **identical to production on every class** (`is_stm` FP 13 / FN 43, michel FP 18 / FN 18; 0 `is_stm` flips, 0 `michel_found` flips):

| item | verdict | pin_rr | move | bend | effect |
|---|---|---:|---:|---:|---|
| `039349_34/48` | STM_ONLY | 5.5 | 5.24 cm | 12.1° | **0.3 cm from the pin** (was 5.5 cm off) |
| `039349_7/4` | STM_MICHEL | 5.9 | 5.40 cm | 10.4° | **0.5 cm from the pin**; Michel energy 9.4 → 15.5 MeV |
| `039349_26/40` | STM_MICHEL | — | 6.00 cm | 13.2° | verdict and Michel energy unchanged |
| `039349_17/62` | THRU | — | 7.20 cm | 10.7° | `is_stm` stays 0 — the negative control holds |

Mean \|stop − pin\| over the 25 overshoot items **4.76 → 4.33 cm**, within 2 cm 4 → 6, **within 1 cm 3 → 5**. `039349_7/4`'s Michel gains 6 MeV because the charge past the old stop now joins the Michel instead of the muon — which is the physics doc 78 §2 was after.

Two of doc 78 §2.3's four "straight gap-bridge" items are recovered this way; `039349_81/25` (12°) and `039252_9/101` (13°) did not move on the real graph, the offline twin's 2× over-count showing up exactly where §2.3 predicted it would.

## 8. Flip

**Flipped in PDVD production** (`pdvd/wct-pr-perevt.jsonnet`, one key in `stm_michel_knobs`): `split_kink_min_deg: 10.0`. The owner authorised the flip for confirmed improvements on 2026-09-10; §7.4 is the confirmation.

The note doc 58 left at this site says `split_kink_min_deg` was deliberately left **unset** so it could not appear as an inert key. 10 is not its C++ default, so it is a real value and belongs in the file — the OFF proof for it is an override back to 15.0, not the key's absence. Proofs (`d82_proofs.sh`, `/home/xqian/tmp/p82/proofs.txt`):

- **A** PRE + `-S stm_michel_extra={split_kink_min_deg:10.0}` vs POST unset: **0 lines** — the flipped file compiles to exactly what the graded arm ran.
- **B** POST + `-S stm_michel_extra={split_kink_min_deg:15.0}` vs PRE: exactly one line, the key present at 15 vs absent.
- **C** PRE vs POST: exactly one line, the key at 10.
- **D** the doc-82 knobs are absent from the shipped config: `stop_tail_peak_frac` 0 keys, `stop_tail_peak_kink_min_deg` 0 keys, `michel_collinear_split` not present.

**Confirmation arm** `p82vprod` — the flipped file, no TLA at all — is bit-identical to `p82bk10`: 120/120 Bee zips identical member-by-member, all eight `tracking-pr.root` trees identical on all 120 events.

**Not flipped**: `stop_tail_peak_frac` / `stop_tail_peak_kink_min_deg` (§7.1), `michel_collinear_split` (§7.3). PDHD unchanged — the split does not run there at all.

## 9. Observations, and what this leaves

1. **The predictions missed in a way worth recording.** The twin predicted at most 8 movers, all Bragg-confirmed `pin_rr` items; the arm moved 37, of which only 5 were on the list. Two mechanisms the twin cannot see: the knob **changes an existing move** (a newly admitted row can win the split's largest-bend contest, which is what cost `039252_2/79` and `039349_48/54`), and it reaches candidates with no `pin_rr` entry at all. A twin built on the published profile can predict *whether a row qualifies*; it cannot predict *which row wins* or what a moved stop does downstream. Docs 57/58 met the same limit from the other side (probe 7 → real 2).
2. **`039252_2/79` is now doubly interesting.** Doc 78 §3.1 wanted its Michel, which T2c demotes at a 59.58° kink against a 60° exemption. On `p82btp` the Michel *is* found — and the stopper is lost. Doc 78 item 3 (the T2c margin) should be re-read with that in hand: the item wants a *smaller* correction than this round applied.
3. **A hot tail is not a Michel signature, and this round measured the cost of pretending otherwise.** What remains for doc 78's 13 missed Michels behind the stop is a topological handle, not a charge one — which is what item 4 (the attached-arm rule) and doc 80's census are for. The census's rej-13 population is the natural next place to look: a MIP-like piece attached to the chain a few cm inside the fit's end says the muon stopped there in a way charge shape cannot.
4. **`retreat_peak_frac` and the four straight bridges are settled.** The peak fraction is moot (§2.1); two of the four bridges are now handled by the flipped `split_kink_min_deg`, and the other two (`039349_81/25`, `039252_9/101`) need the graph, not the threshold.
5. **Next**: doc 78 item 4 (the attached-arm rule, now gradeable with doc 80's census) ahead of item 3 (the T2c margin, re-read per observation 2) and item 5 (the P4 rejection census).

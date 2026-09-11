# 91 — A second, wider Bragg-peak read for the fit-through stoppers: sized offline, not built

**Status (2026-09-11): SIZED, read-only.** No toolkit, config or record change, and no new arm. This is doc 90 §8 item 1.
- **The rule.** Where production's reading still fails a shape test, look for the Bragg peak up to W cm back instead of 3, read the four shape tests from there, and clear them only if that reading passes and two charge guards hold.
- **The result.** Its best pre-registered point, W 8 cm with guards R 1.3 and D 0.8, gains **5 stoppers at 0 judged THRU**: `is_stm` 238 / 7 / 47 → 243 / 7 / 42, efficiency 0.835 → 0.853, purity 0.971 → 0.972. All 5 are owner-judged STM_MICHEL, and no MESSY or unjudged candidate moves.
- **The twin is exact.** It reads the unrounded fit rows from the arm's ROOT trees and reproduces production on 585 of 585 candidates, bit for bit (§4). On the payload's rounded rows it had put the anchor one row off on 25, because the rows sit on exact 0.6 cm steps and nominal edges like rr 3.0 cm are decided by the last bit.
- **Scope.** The rule recovers the stopper call, not the Michel: `michel_found` does not move, so the 5 gains become `is_stm` 1 / `michel_found` 0. Of doc 90's five pinned fit-through Michels it gains 2 (`039349_71/37`, `039349_44/28`).
- **Why it is the owner's call, not an automatic flip.**
  - W, R and D were chosen on this record: the third anchor window tried, after doc 65's 10 cm and doc 68's 3 cm.
  - The R guard keeps out an owner-judged THRU (`039349_81/25`) by 0.02 (§6).
  - A build is the next step: a new default-OFF knob. `bragg_peak_search_cm` is not touched.

## 0. Repro

```bash
S=/home/xqian/toolkit-dev/wcp-porting-img/pdvd/docs
export STM_SCAN_RECORD=$S/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json
# inputs: doc 90's production arm p90vprod -- its prep (585 payloads), its tracking-pr.root trees, its compiled config
# pre-registration (grid, selection rule, bar, predictions): /home/xqian/tmp/p91/pred.txt, mtime 15:00:44; first run 15:02:25
python3 $S/nf_sp_img_clus/scripts/d91_wide_anchor.py --prep /home/xqian/tmp/p90/prep_p90vprod \
    --cfg /home/xqian/tmp/p90/proofs/post.json --root-arm p90vprod \
    --json /home/xqian/tmp/p91/wide_root.json > /home/xqian/tmp/p91/wide_root.txt 2>&1; echo rc=$?     # rc=0
# the same on the payload's rounded rows (section 1: 583 / 585, 30 items flagged) -- the first run, kept for §4
python3 $S/nf_sp_img_clus/scripts/d91_wide_anchor.py --prep /home/xqian/tmp/p90/prep_p90vprod \
    --cfg /home/xqian/tmp/p90/proofs/post.json --json /home/xqian/tmp/p91/wide.json > /home/xqian/tmp/p91/wide.txt 2>&1; echo rc=$?   # rc=0
# the record guard: any record but smx7 is refused (rc=1)
STM_SCAN_RECORD=$S/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_verdicts.json python3 $S/nf_sp_img_clus/scripts/d91_wide_anchor.py \
    --prep /home/xqian/tmp/p90/prep_p90vprod --cfg /home/xqian/tmp/p90/proofs/post.json; echo rc=$?
```

The thresholds are read from the compiled production config and printed with their source in section 0: the anchor, the fallback, P1 at 3 MeV / 3 cm clearing sparse, `plateau_mip_hi` 2.0, `ks_margin` −0.02, `compare_range_cm` 45. Where a key is absent (contrast 0.6, tail 0.5–3 cm, plateau 20–40 cm, `bragg_anchor_rise_min` 1.5), they come from the C++ member initializers.

## 1. The question, and what production does

Doc 90 §6 found that on 5 of the owner's 13 missed Michels the track fit runs through the Michel, and the owner's pin sits 3.6–7.2 cm back from the fit end.

Production reads the Bragg peak at a residual-range origin (`CheckSTM_Michel.cxx:2687–2847`):
- **The anchor** (doc 65 T7, doc 68). The origin is the row with the largest 5-point running mean of dQ/dx on the live profile, searched within `bragg_peak_search_cm` 3 of the fit end.
- **The fallback** (doc 75). The four shape tests are re-read at the geometric end when the anchored reading keeps a shape bit and its peak is ≥ 1.5 × its plateau.

Doc 90 §8 asked whether a wider search would reach the peak without the purity cost doc 65 measured: 10 cm unconditionally gave +38 / −8 TP and +19 FP.

**The precondition, first.** A wider maximum search reaches the peak only where the peak's 5-point mean beats every mean nearer the end.
- On `039349_29/40` and `039349_44/28` the fit's *last* row is the hottest row (2.26 and 2.23 × plateau). That is why the 3 cm anchor stays at the end. A wider search does pass it, because the 5-point mean at the end is diluted by the low rows before it.
- The pattern on these items is a Bragg rise, a low stretch (the Michel), then a hot last row. §7 gives where the wide origin lands on each of the five.

## 2. The rule, and where it would sit

A candidate for a **new** default-OFF knob, placed after the fallback and before P1 (`stm_michel_topology_clear`, `:3989`):
- **When it runs.** Only where the reading that stands still carries a shape bit (`no_bragg | shape_flat | plateau_off_mip | profile_sparse`).
- **What it reads.** It re-anchors at the 5-point maximum within **W** cm of the geometric end, with the same recipe on the same live rows, and reads the same four tests there.
- **When it clears.** It clears all four bits only if that reading sets none of them, and:
  - **R:** the wide peak's 5-point mean is ≥ R × the wide plateau median (R 0 = off);
  - **D:** at least 3 live rows lie past the wide peak, and their median is ≤ D × the wide plateau median (D 0 = off). This is doc 90 §8's "the tail past the peak reads below the plateau".
- **What it cannot do.**
  - It never sets a bit, so it cannot lose a stopper (checked: 0 `is_stm` 1 → 0 anywhere on the grid).
  - PID (`do_track_comp`), the dead-volume probe, `n_live_pts` and `dead_frac_cmp` keep the anchored profile, so every non-shape bit is the payload's.
  - P1 is re-applied afterwards at the compiled floors.
  - It makes no Michel object.

**Pre-registered before the first run** (`/home/xqian/tmp/p91/pred.txt`):
- **Grid:** 60 points: W ∈ {4, 5, 6, 8, 10} cm, R ∈ {off, 1.3, 1.5, 1.8}, D ∈ {off, 1.0, 0.8}.
- **Selection:** among the points that admit 0 judged THRU, the most judged stoppers gained. Ties go to fewer MESSY or unjudged movers, then the smaller W, the larger R, the stricter D.
- **Bar:** at least 1 stopper gained at 0 THRU, and at least one one-step neighbour also gaining at 0 THRU (a plateau, not a spike).

## 3. The population

Production on the 576 judged items of the smx7 record reads 238 / 7 / 47 (efficiency 0.835, purity 0.971). A clears-only rule can touch only candidates that are `is_stm` 0 and fail a shape test:

| production reading | total | stopper | THRU | MESSY / UNCLEAR | unjudged |
|---|---:|---:|---:|---:|---:|
| `is_stm` 0 with a shape bit before P1 | 322 | 34 | 263 | 19 | 6 |
| … and no other bit left after P1 (the ceiling) | 290 | 29 | 238 | 18 | 5 |

The ceiling's 29 stoppers are 29 of doc 89's missed set. The THRU side is 8 times larger, so purity is the whole question.

## 4. The twin: exact once it reads the unrounded rows

**First pass, payload rows.** Section 1 of `wide.txt` checks production's own reading (anchor 3 cm, fallback, P1) against the published verdict:
- `is_stm` agrees on 583 of 585, and (`reject_bits`, `topology_cleared_bits`) on 579.
- The anchor shift agrees within 0.01 cm on only 560: on 25 candidates the twin reads 2.80 cm where production reads 2.20, one row further back.

**Cause.** The fit rows near the stop sit on exact 0.6 cm steps. On `039252_16/110` the rows read rr = 6.6, 6.0, 5.4, …, 3.0, 2.4 in the tree's doubles, so the row at nominal rr 3.0 lies exactly on the 3 cm search edge.
- The prep rounds L to 0.01 cm, and the twin's rr is rebuilt as total − L. Whether that row sits a hair inside or outside the edge is decided by the last bit, which the payload does not carry.
- The same holds at rr 6.0 (the W 6 edge) and at 20.0 (the anchored plateau's lower edge, since the anchored rr = 0.2 + 0.6 k).
- This is doc 75's "4 of 568 rounding misses". It is small on `is_stm`, but a window sweep puts the edges on the step grid.
- `wide.txt` was written by the script before the `--root-arm` option was added. The committed script without `--root-arm` reproduces it line for line (diff empty), so both runs in §0 are this script's.

**Second pass, the tree's rows** (`--root-arm p90vprod`). The twin reads each candidate's role-1 rows (L, rr, q as 8-byte doubles) from `T_stm_michel_pts` in the arm's `tracking-pr.root`:
- **The join.** 170,765 rows on 585 candidates, row for row with the payload; max |L − payload L| is 0.0050 cm, the rounding.
- **The windows.** Edges are compared with no tolerance.
- **Agreement:**
  - `is_stm` 585 / 585;
  - bits 585 / 585;
  - anchor fired 348 / 348;
  - shift within 0.01 cm 585 / 585;
  - fallback flag 585 / 585.
- **So every number below is a prediction, not an estimate.** An arm built from the knob should match it item for item.
- **One caveat.** The twin's KS sums differ from `kslike_compare`'s in accumulation order. The difference is about 1e-16 and matters only at an exact tie.

## 5. The grid (`wide_root.txt` section 3)

| W (cm) | R | D | fires | +stoppers | +THRU | `is_stm` TP / FP / FN | purity |
|---:|---:|---:|---:|---:|---:|---|---:|
| 4 | off | off | 6 | 1 | 5 | 239 / 12 / 46 | 0.952 |
| 5 | off | off | 18 | 5 | 12 | 243 / 19 / 42 | 0.927 |
| 6 | off | off | 21 | 6 | 14 | 244 / 21 / 41 | 0.921 |
| 8 | off | off | 31 | 11 | 18 (+1 unjudged) | 249 / 25 / 36 | 0.909 |
| 10 | off | off | 36 | 13 | 20 (+2 unjudged) | 251 / 27 / 34 | 0.903 |
| 8 | 1.3 | 1.0 | 12 | 7 | 4 | 245 / 11 / 40 | 0.957 |
| 8 | off | 0.8 | 7 | 5 | 2 | 243 / 9 / 42 | 0.964 |
| **8** | **1.3** | **0.8** | **5** | **5** | **0** | **243 / 7 / 42** | **0.972** |
| 8 | 1.5 | 0.8 | 3 | 3 | 0 | 241 / 7 / 44 | 0.972 |
| 6 | 1.3 | 0.8 | 3 | 2 | 0 | 240 / 7 / 45 | 0.972 |
| 10 | 1.3 | 0.8 | 4 | 3 | 1 | 241 / 8 / 44 | 0.968 |

**Unguarded, the rule is doc 65's cost in miniature.** From W 4 up it admits THRU items with a hot stretch a few cm back: `039252_17/80`, `039253_10/93`, `039349_24/58`, `039349_42/43`, `039349_51/44` at W 4, and 20 at W 10.

**Guarded, 0 THRU is a thin ridge along D 0.8.** The 13 points that gain at 0 THRU:
- **W 5–6:** R 1.3 or 1.5 with D 0.8, and R 1.5 with D 1.0: +2 (`039252_9/101`, `039349_44/28`). R 1.8: +1.
- **W 8:** R 1.3 with D 0.8: **+5**. R 1.5 with D 0.8: +3. R 1.8: +3 or +2.
- **W 10:** R 1.5 with D 0.8: +3. R 1.8 with D 0.8: +2.

D 0.8 with R ≥ 1.5 is 0-THRU at every W from 5 to 10; loosening D to 1.0 lets THRU in at W 8 and 10. Only W 8 with R 1.3 reaches 5.

**The pre-registered selection is W 8, R 1.3, D 0.8.**
- **Gains, all owner-judged STM_MICHEL:** `039252_9/101`, `039349_44/28`, `039349_51/29`, `039349_71/37` (smx7) and `039253_12/93` (smx4).
- **No other movement:** 0 THRU, 0 MESSY, 0 unjudged.
- **Neighbours:**
  - W 6 → +2, 0 THRU;
  - W 10 → +3, +1 THRU (`039349_81/25`);
  - R off → +5, +2 THRU (`039252_4/55`, `039349_81/25`);
  - R 1.5 → +3, 0 THRU;
  - D 1.0 → +7 (adds `039252_8/102` and `039349_3/47`), +4 THRU (`039349_17/62`, `039349_24/58`, `039349_50/46`, `039349_51/44`).
- **The bar is met:** two neighbours also gain at 0 THRU.

## 6. The margins at the selected point

At W 8, every candidate whose wide reading passes the four tests with p ≥ 1.3, sorted by the D quantity. Here p is the wide peak's 5-point mean over the plateau, and d is the median of the rows past the peak over the plateau:

| item | record | p | d | at D 0.8 |
|---|---|---:|---:|---|
| `039349_71/37` | owner STM_MICHEL | 1.81 | 0.54 | gained |
| `039349_51/29` | owner STM_MICHEL | 1.37 | 0.72 | gained |
| `039349_44/28` | owner STM_MICHEL | 1.75 | 0.77 | gained |
| `039252_9/101` | owner STM_MICHEL | 1.97 | 0.78 | gained |
| `039253_12/93` | owner STM_MICHEL (smx4) | 1.42 | 0.79 | gained |
| `039349_3/47` | owner STM_MICHEL | 1.44 | 0.81 | out by 0.01 |
| `039252_8/102` | owner STM_MICHEL | 2.13 | 0.90 | out |
| `039349_51/44` | THRU, smx1a medium | 1.30 | 0.93 | out |
| `039349_24/58` | THRU, smx1a medium | 1.72 | 0.94 | out |
| `039349_17/62` | THRU, smx1a medium | 1.54 | 0.96 | out |
| `039349_50/46` | THRU, smx1a medium | 1.48 | 0.97 | out |
| `039349_28/57` | THRU, smx1a high | 2.15 | 1.00 | out |
| … 8 more THRU at d 1.03–1.26, 2 owner Michels and 1 stopper at 1.11–1.42 | | | | out |

- **D (0.8) sits in a gap.** The gained stoppers read d ≤ 0.79, and the nearest THRU d ≥ 0.93, all four of them medium-confidence smx1a calls.
- **R (1.3) is thin against an owner THRU.** `039349_81/25`, judged THRU by the owner in smx7, reads p **1.28** and d 0.55 at W 8. It is kept out by 0.02 on R alone. At W 10 it reads p 1.38 and is admitted.
- **In charge, that item is indistinguishable from a fit-through Michel:** a peak about 7 cm back, then a low tail. This is the owner's kink discriminator again: charge shape cannot name the particle after the peak.
- **R 1.5 is the conservative point.** It keeps `039349_81/25` out by 0.22 and gains 3 (`039252_9/101`, `039349_44/28`, `039349_71/37`), losing `039349_51/29` (p 1.37) and `039253_12/93` (p 1.42).

## 7. Doc 90's five pinned fit-through Michels

| item | owner pin (cm) | wide origin W 5 / 6 / 8 / 10 (cm) | at the selected point |
|---|---:|---|---|
| `039349_71/37` | 5.4 | – / – / 7.6 / 8.8 | **gained** (origin +2.2 cm upstream of the pin) |
| `039349_44/28` | 4.3 | 4.7 / 5.3 / 5.3 / 5.3 | **gained** (+1.0) |
| `039349_29/40` | 3.6 | 4.6 / 5.8 / 5.8 / 5.8 | not: d 1.11. The hot last row (2.26 ×) and the rows after the peak read at plateau |
| `039349_3/47` | 7.2 | 4.6 / 5.8 / 7.6 / 9.4 | not: d 0.81, 0.01 over D |
| `039252_12/123` | 7.2 | – / – / – / 9.4 | not: reached only at W 10, where d 0.96 |

The wide origin is the top of the Bragg rise, so it lands 0.4–2.2 cm upstream of the owner's stop. That is the right origin for the shape tests; it is not a stop position, and the stop movers are untouched.

## 8. The predictions, graded

- **P1** (the payload twin reproduces production's `is_stm` on ≥ 580, the misses are window-edge rounding, `039349_82/25`'s shift among them): **held.** 583, and 30 items flagged, `039349_82/25` among them. The tree's rows remove all 30 (§4).
- **P2** (no `is_stm` 1 → 0): **held**, by construction.
- **P3** (reach: `29/40` and `44/28` from W 5–6, `71/37` and `3/47` only from W 8, `12/123` only at W 10; ≥ 2 of the 5 gained at a 0-THRU point): **mostly held.** `3/47`'s wide reading already passes at W 6 (origin 5.8 cm, 1.4 short of its pin), so "only from W 8" missed for it. Exactly 2 are gained.
- **P4** (unguarded THRU from W 5 up, ≥ 3 at W 10; `039349_81/25` admitted by no point with R ≥ 1.3): **missed twice.** THRU enter already at W 4 (5). `039349_81/25` is admitted at 3 points with R ≥ 1.3, all at W 10.
- **P5** (the selected point carries a guard, W 6 or 8, gains 3–10): **held.** W 8, R 1.3, D 0.8, +5.

## 9. What this does not settle

- **The Michel.** The 5 gains would read `is_stm` 1 / `michel_found` 0, which is doc 78 item 8's population. Item 8's anchor-tail rule reads the 3 cm anchor's shift; with the wide origin it would see a 5.3–7.6 cm tail on these items. That is a separate rule and needs item 8's blind re-judge.
- **The other 8 owner Michels** (doc 90 §8 item 2, off-fit charge). The rule does not touch them.
- **Out-of-record behaviour.** The thresholds were chosen on this record, so no build can prove purity away from it. The margins in §6 are what the owner weighs.
- **The bar counts every judged THRU alike.**
  - A medium-confidence smx1a THRU weighs as much as an owner verdict, although smx7 overturned 5 of 30 agent calls of that class.
  - The D 1.0 neighbour is where this bites: +7 owner-judged Michels against 4 THRU (`039349_17/62`, `039349_24/58`, `039349_50/46`, `039349_51/44`), all medium-confidence smx1a.
  - This is a limit of the bar, not a reason to move the selection.

## 10. Next, ranked

> **Update (doc pdvd/92, same day).** Item 1 is **done**: the rule is built behind
> `bragg_wide_anchor_cm` / `_rise_min` / `_tail_max`, OFF is byte-identical on both detectors, and
> both arms reproduce the twin item by item. But the blind re-judge doc 92 ran alongside it
> (`smx8`) **overturned the owner's own smx7 verdict on `039349_71/37`**, from STM_MICHEL to THRU.
> That item is one of the five gains below, at both R 1.3 and R 1.5. So §4's headline — *+5 stoppers
> at 0 judged THRU* — does not survive: on the corrected record the selected point gains **4
> stoppers and 1 false positive** (eff 0.832 → 0.846, purity 0.971 → 0.968), and R 1.3 dominates
> R 1.5. The numbers in §4–§7 below are as measured on the smx7 record and are left unchanged as
> that round's record; read doc 92 §6 for the corrected census.

1. **Build the rule as a new default-OFF knob:** e.g. `bragg_wide_anchor_cm` (0 = off), `bragg_wide_anchor_rise_min`, `bragg_wide_anchor_tail_max`.
   - Placement: after the fallback, all four bits or none, with a persisted flag and the wide shift, and a DEBUG line per re-read.
   - Order: the C++ must keep the twin's order, or the exact twin stops being exact. The fallback runs first; the wide read runs only where the fallback left a shape bit, so both can run on one candidate, and a fallback that stood leaves nothing for the wide read.
   - Gates: the OFF gate on both detectors, then arms at the two operating points (R 1.3: +5; R 1.5: +3). The exact twin predicts both item by item.
   - The flip is the owner's pick between them (the doc 68 precedent). PDHD stays OFF.
2. **Item 8 on the wide shift:** the Michel object for the gained fit-through stoppers. Blind re-judge first.
3. **Doc 90 §8 item 2:** a discriminator for the 8 unfitted Michels.
4. **Doc 88 §9.5 item 2:** grade the PDVD flips on PDHD's smx18 record.
5. **Doc 78 item 9:** the Michel charge energy.

**Recommendation: 1.**
- The twin is exact, so the arm is a confirmation, not a measurement.
- The OFF path is byte-identical by construction.
- Both operating points gain owner-judged stoppers at 0 judged THRU.
- The owner then decides between +5 at a 0.02 margin against an owner THRU, and +3 at 0.22.
- **What settles that choice is `039349_81/25`.**
  - It is the only owner-judged THRU near the boundary, and in charge it reads like the gained items.
  - Its earlier record called it STM_MICHEL (medium, pin 6.0 cm) before the owner's smx7 call.
  - If the owner wants the +5 point, a second look at it, first in the next scan tranche, is what decides between R 1.3 and R 1.5.

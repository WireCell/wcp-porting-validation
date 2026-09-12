# doc pdhd/20 — the dQ/dx panel drew discarded points as measurements; the owner's re-judge, and three pre-registered tests

**Status:** display fix shipped default-OFF and gated. Record `smx22`. Two of three
pre-registered predictions FAILED and are reported as such. No production code, config or
arm is touched; no A/B gate is owed.

## Repro

```sh
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
S=$I/pdhd/stm_michel_scan ; X=$I/pdhd/docs/scan ; C=$S/campaign ; H=/home/xqian/tmp/h22
K="--key $X/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv \
   --key-extras $X/smx18/pdhd_stm_michel_scan_key_h18b.tsv --shots /home/xqian/tmp/h19/shots"

# 1. the display fix, default OFF (patches in $H/patch_viewer.py, patch_viewer2.py, patch_harness.py)
#    gate: flag OFF must reproduce the PRE-patch render byte for byte
$S/scan_harness.py shots --det pdhd --tag smx21 --prepdir $S/prep-pdhd-smx19 \
    --labeldir $H/lbl_off2 --items "028084_17/38,029107_25/27,028084_15/31,028084_0/108" --out $H/shots_off2
$S/scan_harness.py shots ... --dead-points --out $H/shots_on2          # the ON arm

# 2. served for the owner on :5017 under their OWN empty tag, with the fix on
$S/serve_stm_michel_scan.sh 5017 --det pdhd --scan-tag own20 \
    --manifest $H/manifest_own20.tsv --prepdir $S/prep-pdhd-smx19 --dead-points

# 3. their 12 rulings -> the smx22 record (no v5 pass this round, hence --skip-v5)
python3 $C/mkowner_record.py $X/pdhd_stm_michel_smx21_verdicts.json $X/smx22/owner_rulings_own20.json \
    $H $X/pdhd_stm_michel_smx22_verdicts.json $X/smx22/provenance.json --skip-v5
cd $S && ./verify_scan_record.py --det pdhd --tag smx21 --record ../../pdhd/docs/scan/pdhd_stm_michel_smx22_verdicts.json
python3 $I/pdhd/docs/scripts/d18_census.py --record $X/pdhd_stm_michel_smx22_verdicts.json $K  # = $X/smx22/census_smx22.txt
```

## 1. Why this round

After doc 19 §9 the owner judged 8 items and said: *"All of these cases, the dQ/dx fit are
not good enough. Difficult to judge."* Three lines pointed the same way — those 8, the 4
unsettled smx21 splits, and four of six h21 scanners objecting that their stopper calls
rested on a rule with no owner example. The owner chose to **fix the display first**, then
re-judge, then measure.

## 2. Root cause: the panel's live mask was inert

`stm_michel_viewer.py:2524` computed the dQ/dx panel's mask as `live = Q > 0`. Its comment
targets roles 2/3/4, which carry negative dQ/dx where the fit found no charge — but
`muon_arrays(pay)` returns the **muon chain only**, and no chain point in any payload has
`q <= 0` (PDHD min **7.0** e/cm over 317 payloads; PDVD min 27.4 over 569). On the chain the
mask was inert by construction.

The chain's real live cut is `dQ/dx >= profile_min_dqdx_frac * mip_dqdx`. Points below it
are dropped from **every** verdict metric, yet were drawn at full opacity on the same Turbo
scale as real measurements. Consequence on the last 20 cm: dead-fraction median 0.000,
q90 0.324, **max 1.000**; **38 of 317 items at >= 25 % dead, 16 at >= 50 %** — at least one
item with no measured point near the stop that looked fully measured.

**The cut is not a guessed constant.** Per detector, from that detector's own production
config, and validated against the chain's own `verdict.n_dead_pts`:

| det | frac | mip_dqdx | cut | source | agreement |
|---|---|---|---|---|---|
| pdhd | 0.15 | 56000 | **8400** | `pdhd/wct-pr-perevt.jsonnet:243, :310` | **317/317** |
| pdvd | 0.15 | 55000 | **8250** | `pdvd/wct-pr-perevt.jsonnet:246, :696` | **569/569** |

Wrong constants fail loudly: PDVD at 8100 gives 498/569, at 8400 gives 516/569. Note
`census_lib.MIP = 54000` is the *display's reference-curve plateau*, **not** `mip_dqdx`;
using it would mislabel 71 PDVD items.

## 3. The fix, and why `live` was left alone

A **second** scatter series draws sub-cut points hollow grey, on no colour scale. It joins
`QSCAT` only when the flag is on, so with the flag off the glyph list, the clear shape
(`blank()`, :1585) and the tap wiring (:3094, :3101) are exactly as before. A per-item
agreement check compares the count below the cut with `verdict.n_dead_pts` and, on
mismatch, **greys nothing** and says so in the title — a wrong constant must never dress a
real measurement as a hole.

`live` is deliberately **not** redefined: it is the row filter at :2538 and sets the y-axis
maximum at :2562, so narrowing it would have *deleted* dead points from the panel — hiding
the problem instead of showing it.

Flags, all default OFF: viewer `--dead-points`, passed through
`serve_stm_michel_scan.sh` and `scan_harness.py`.

### Gates (all re-run from scratch after a mid-course style change, never assumed)

| gate | result |
|---|---|
| flag OFF byte-identical to the pre-patch render (4 items × `g_dqdx.png` + `context.json`) | **PASS 8/8** |
| flag ON changes the three dead-ended items | **PASS** |
| zero-dead control `028084_0/108` identical ON vs OFF (predicted in advance) | **PASS** |
| cut reproduces `n_dead_pts` on the gate items | **PASS** 17/17, 15/15, 24/24, 0/0 |
| label rows written by any shoot | **none** |

## 4. A defect four passing gates could not catch

Reading the rendered frame as a scanner would showed two faults: the panel **title is
already clipped** at the figure width, so the count note the first patch *appended* to it
was never visible — and it duplicated the badge div's "N live / M dead pts" anyway; and the
marker was too faint at native size. Both fixed; the title now carries only the failure
case, prepended so it survives the clip.

**Lesson:** a byte gate proves a render *changed*; it cannot prove the change is legible or
that a message reaches the reader. Looking at the picture is a separate check.

What the fixed panel then revealed on `028084_15/31`: **the troughs of the periodic dQ/dx
wave are the dead points** — the oscillation's minima fall below the cut, so the chain keeps
the peaks and discards the valleys.

## 5. The owner's re-judge (`own20`) — 3 of 8 changed

Served on :5017 under their own empty tag (the pattern that works: never a record tag).
They judged all 12 — their 8 own19 items plus the 4 splits — and reported *"I finished
scan, still hard to judge"*.

**"Still hard to judge" is NOT "nothing changed", and was first written up here as if it
were. That was an error, corrected.**

| item | own19 | own20 |
|---|---|---|
| `029107_15/26` | STM_MICHEL attached | **THRU** |
| `029107_24/47` | THRU | **UNCLEAR** |
| `029107_5/93` | THRU | **UNCLEAR** |
| `028084_16/106` (split) | — | THRU |
| `029107_20/24` (split) | — | UNCLEAR |
| `029107_21/43` (split) | — | STM_MICHEL attached, **pin at rr 13.42 cm** |
| `029107_4/72` (split) | — | STM_ONLY |

Five unchanged. The pin on `029107_21/43` accompanies a verdict, so it is the owner's stop,
not the stray unconfirmed kind seen on smx19; it is kept as `owner_review.app_edit`.

`revealed_before_label` is true on 12/12 (and 8/8 in own19): **these owner looks are not
blind.** A standing property of the judge queue, recorded so the write-up does not imply
otherwise.

All other label files (smx1, smx18, smx19, smx20, smx21, own19) are byte-unchanged by the
session, before and after the viewer was stopped.

## 6. Census: smx21 -> smx22

Owner ruling 2026-09-11: the four splits **re-enter** the grading, being no longer
unsettled. `mkowner_record.py` clears `owner_queue` on any item the owner has since ruled,
so no exclusion list exists or is needed.

| production (303) | TP | FP | FN | TN | purity | efficiency |
|---|---|---|---|---|---|---|
| smx21 | 61 | 0 | 89 | 110 | 1.000 | 0.407 |
| **smx22** | 61 | 0 | **86** | 110 | **1.000** | **0.415** |

Hand stoppers **160 -> 157**; Michel FP 12 -> 11, purity 0.824 -> 0.836. Verified per item
against the records rather than by arithmetic: net **-3** stoppers, net **0** Michels.

`verify_scan_record` reports the single known mismatch (`029107_15/26`, the stray smx19
pin), unchanged from smx21. Labels stay at tag `smx21`: this round adopts no item, so no
label row changes.

**A trap worth recording:** the committed censuses of this lineage use
`--shots /home/xqian/tmp/h19/shots` (94 item dirs). Running with `h18/shots` (317) changes
**only** the per-APA breakdown — 91 resolved + 212 unresolved becomes ~235 resolved — and
no headline number. Both are kept (`census_smx21_baseline_h18shots.txt`) so the difference
is documented rather than rediscovered.

## 7. Three pre-registered predictions: two failed, one survived

Each was written down before it was computed (`$H/design.md`, `$H/prereg_fit_quality.md`).

**1. FAILED — coverage at the stop.** Predicted the owner's items would be unusually dead
near the stop. **0 of 8** above a 0.25 dead fraction, against a 12.0 % base rate — below it.

**2. FAILED — wave troughs vs coverage holes.** Predicted their dead points would be
isolated (oscillation troughs). Isolated share **0.067** against a **0.085** base; their
runs are long (27, 15, 7). Their items have coverage holes, just **not at the stop**. The
trough reading is true of `028084_15/31`, where it was seen, and does not generalise.

**3. SURVIVED — the fit quality, which is what the owner actually named.** On the 305
items excluding the owner's 12, with production's own thresholds: family A = `plateau_med`
outside the 0.6–1.6 × mip window; family B = `contrast < 0.6` inside it.

| set | A ∪ B | rate |
|---|---|---|
| null floor, all 305 | 132/305 | 0.433 |
| HARD (confidence low/medium) | 104/185 | 0.562 |
| EASY (high) | 28/120 | 0.233 |

Naively z = +6.19 — **but that is inflated**: A ∪ B is a *subset* of "production rejected
it" by construction (0/65 among accepted items, since plateau-outside-window **is**
`plateau_off_mip`). Within the rejected stratum: 0.627 vs 0.378, z = +3.66. And hard items
are half the length of easy ones (median 188 vs 382 profile points), with
plateau-outside-window falling 0.500 -> 0.287 -> 0.225 across length terciles. Stratified
over length, **MH odds ratio 2.58, CMH χ² 9.33, z = 3.05, p ≈ 0.002 — it survives.**

## 8. What is NOT concluded

* **Not** that the display fix explains the owner's difficulty. It changed 3 of 8 verdicts,
  and they still call the items hard.
* **Not** causation in §7.3. `confidence` is a scanner's judgement formed from the same
  picture the chain measures; for family B "looks flat" and `contrast < 0.6` are nearly the
  same statement. Family A carries the signal, which may be exactly because it is the less
  directly visible one.
* **Not** anything about the owner's 12 — excluded by construction, being where the pattern
  was spotted.
* **Not** a knob. Nothing here licenses moving `plateau_mip_lo/hi`. The obvious
  confirmatory step — does a length-aware plateau window recover stoppers without adding
  false ones — has **not** been run.
* The periodic-wave/wire-crossing item stays open and separate: median 1.43 cm per W wire
  over the last 20 cm, q95 6.60, 43 of 315 items at >= 4 cm/wire. A tail population, not the
  explanation for 38 dead-ended items.

## 9. What is left

1. Test whether a **length-aware plateau window** recovers stoppers without adding false
   ones — the confirmatory step §7.3 does not itself justify.
2. Port `census_score.py` / `census_lib.py` to `--det pdhd` (its PDVD constants are at
   `census_lib.py:38-51`); `d18_census.py` is already PDHD.
3. Grade PDVD's production knobs on PDHD against `smx22` (P1 `topology_stop_evidence`).
4. Then update the PDHD production chain.

# h22 — render the chain's dead dQ/dx points as dead (doc pdhd/20 step 1)

Written **before** the patch and before any render, so the fix's intent, its gate and
its failure mode are fixed in advance. Owner OK 2026-09-11 ("fix display, you re-judge").

## Symptom

The owner judged 8 items on :5017 (tag `own19`) and reported: *"All of these cases, the
dQ/dx fit are not good enough. Difficult to judge."* Three independent lines point at the
same place: those 8, the 4 unsettled smx21 splits, and four of six h21 scanners flagging
that their stopper calls rest on a rule with no owner example behind it.

## Root cause (verified, not inferred)

`stm_michel_viewer.py:2524` computes the dQ/dx panel's mask as

    live = Q > 0

Its own comment says this is meant for roles 2/3/4, which carry negative dQ/dx where the
fit found no charge. But `muon_arrays(pay)` returns the **muon chain only**, and no chain
point in any payload has `q <= 0`:

* PDHD `prep-pdhd-smx19`, 317 payloads: global min `muon.q` = **7.0 e/cm**, points with
  `q <= 0` = **0**.
* PDVD `prep-pdvd`, 569 payloads: min over the first 60 items = **27.4 e/cm**.

So on the muon chain the mask is **inert by construction**: every point is drawn, at full
opacity, on the same Turbo scale. The chain's own verdict, however, is computed on a live
subset defined by `dQ/dx >= profile_min_dqdx_frac * mip_dqdx`. Points below that cut are
excluded from every verdict metric but are drawn identically to measured points.

Measured consequence on the last 20 cm of `rr` (PDHD, 317 items): dead fraction median
0.000, q90 0.324, **max 1.000**; **38 items at >= 25 % dead, 16 at >= 50 %**. At least one
item has no measured point at all near the stop and looks fully measured.

This is the rubric's `NO_MEASUREMENT:` case, which scanners are currently asked to catch by
eyeballing the separate 2-D `f_meas` panels instead of reading it off the profile itself.

## The cut, and why it is not a guessed constant

Per detector, from each detector's own production config:

| det  | `profile_min_dqdx_frac` | `mip_dqdx` | cut (e/cm) | source |
|------|------------------------|-----------|-----------|--------|
| pdhd | 0.15                   | 56000     | **8400**  | `pdhd/wct-pr-perevt.jsonnet:243`, `:310` |
| pdvd | 0.15                   | 55000     | **8250**  | `pdvd/wct-pr-perevt.jsonnet:246`, `:696` |

**Verified against the chain's own answer**, by counting payload points below the cut and
comparing with `verdict.n_dead_pts`:

* PDHD cut 8400: **317 / 317** items agree.
* PDVD cut 8250: **569 / 569** items agree.
* Wrong constants fail loudly and are therefore excluded: PDVD at 8100 gives 498/569,
  PDVD at 8400 gives 516/569.

`census_lib.MIP = 54000` must NOT be used here: its own comment says it is the *display's
reference-curve plateau*, not the tagger's `mip_dqdx`. Using it would have mislabelled 71
PDVD items.

## What the patch does

1. A per-detector cut table with the config lines above cited inline.
2. A **second** scatter series for points below the cut, rendered visibly dead (grey, no
   Turbo fill), while the existing series keeps the Turbo colouring for live points.
3. The new series joins `QSCAT` **only when the flag is on**, so it gets the correct clear
   shape (`:1585`), mutual deselection (`:3094`) and the tap callback (`:3101`) — a scanner
   can click a dead point and locate it, which is the whole purpose.
4. A **per-item agreement check**: the count below the cut must equal
   `verdict.n_dead_pts`. On mismatch the panel draws **no** dead series and says so in the
   title. A wrong constant must never silently grey real measurements.

Explicitly NOT done: `live` is not redefined. It is the row filter at `:2538` and sets the
y-axis maximum at `:2562`; narrowing it would *delete* dead points from the panel, hiding
the problem instead of showing it. The y-range is computed from the same `live` in both
modes so ON and OFF frame identically.

## Default and gate

Default **OFF**. Flag follows the existing `parse_args` idiom (`:125`), passed through
`serve_stm_michel_scan.sh` beside `--det`/`--scan-tag`.

**Gate:** with the flag off, `g_dqdx.png` from `scan_harness.py shots` must be
byte-identical to the pre-patch render on the same items, same browser, same backend.
Payloads are not touched at all — this is a rendering change only, so no arm is re-run and
no A/B on reconstruction output is owed.

**Falsifiable prediction, recorded now:** the 38 high-dead items should be
over-represented among the owner's 8 and the 4 splits relative to the 317 base rate. If the
items the owner called unjudgeable are *not* unusually dead, then dead-point rendering is
not the explanation for their difficulty and this fix does not address their complaint —
that outcome gets reported as plainly as the other.

## What this does not claim

* It does not claim the dead points are a reconstruction defect. Why coverage is lost at
  APA0/APA2 upper ends is a separate, still-open question.
* It does not touch the periodic-wave/wire-crossing item, which the h21 measurement shows
  is a tail population (median 1.43 cm per W wire over the last 20 cm, q95 6.60, 43 of 315
  items at >= 4 cm/wire) and NOT the explanation for 38 dead-ended items. The two findings
  are independent; conflating them would overclaim.
* No verdict, record or label is changed by this patch.

## OUTCOME OF THE PRE-REGISTERED PREDICTION — IT FAILED

Computed 2026-09-11, before the patch was applied and before any render.

Base rate over the 317: **12.0 %** of items have dead fraction >= 0.25 over the last
20 cm; median dead fraction **0.000**.

| group | >= 0.25 | median |
|---|---|---|
| all 317 | 12.0 % | 0.000 |
| the owner's 8 own19 items | **0 / 8 (0 %)** | 0.111 |
| the 4 smx21 splits | 1 / 4 | 0.065 |

The prediction was that the owner's 8 and the splits would be **over**-represented among
high-dead items. They are not: the owner's 8 are at 0 %, *below* the 12 % base rate.

**Therefore: coverage loss at the stop is NOT the explanation for "the dQ/dx fit are not
good enough. Difficult to judge."** The fix is still correct and still worth making — the
mask is provably inert, and 38 items do have >= 25 % dead ends — but it must not be
reported as the answer to the owner's complaint, and this document does not do so.

Post-hoc, and labelled as post-hoc rather than folded into the prediction: all 8 of the
owner's items have *some* dead points (median fraction 0.111 against a base median of
0.000; `029107_15/26` and `029107_15/40` carry 53 and 52 dead points over the whole
track). That is a weaker and differently-shaped signal than the one predicted. It is a
hypothesis for the measurement round, not a result.

**Consequence for the round:** the readability question is NOT answered by the dead-point
fix, so the measurement round carries the weight, and its metric must not be built out of
last-20 cm coverage alone — the statistic that just failed. In particular the moderate-band
enrichment above is NOT a licence to rebuild the same metric at a lower threshold: that
would be a threshold hunt on the variable that already failed its test. The owner's words
were about the **fit**, not the coverage, so shape and wave are the untested axes.

## Gate assertions, fixed BEFORE the renders are made

Baseline (pre-patch, viewer at git `5fd0ef20`): `/home/xqian/tmp/h22/shots_base`, shas in
`baseline_shas.txt`, 4 items, canvas2d, private empty labeldir (0 rows written).

1. **Flag OFF must reproduce the baseline exactly.** All four `g_dqdx.png` shas equal, and
   all four `context.json` shas equal. Any difference fails the gate outright.
2. **Flag ON must change the three dead-ended items**: `028084_17/38` (dead frac 1.000),
   `029107_25/27` (1.000), `028084_15/31` (0.447). A render identical to OFF on these would
   mean the dead series never drew, i.e. the patch is inert.
3. **Flag ON on the zero-dead control `028084_0/108` (n_dead 0): predicted byte-identical
   to its OFF render.** With no dead points the mask is all-false, `shown == live`, and the
   only difference is an extra renderer carrying zero rows. Whether Bokeh emits identical
   PNG bytes with an empty renderer present is an EMPIRICAL question, not a logical one.
   Recorded now so the answer is not rationalised afterwards: if the control's ON render
   differs, that is **not** silently accepted as fine — it is reported, and the difference
   must be shown to be confined to the empty glyph and to change no plotted point.
4. **Per-item agreement check**: on all four items the count below the cut must equal
   `verdict.n_dead_pts`, so no title reads "DEAD-POINT CHECK FAILED".

## A defect in the first patch, found by LOOKING at the render

All four gates passed, and the patch still had a fault none of them could catch. Reading
the ON frame as a scanner would:

* **The panel title is already clipped** at the figure's width — the unpatched title
  ends "…(no pin y" mid-word. The count note the first patch *appended* to it was
  therefore never visible to anyone. It was also redundant: the badge div has always
  reported "chain L cm, N live / M dead pts".
* The hollow grey was **too faint at native size** for the one job it has.

Fixed in a follow-up patch: the title now carries only what the div cannot say — that the
cut disagreed with the chain, so nothing was greyed and the panel is the old misleading
one — **prepended** so it survives the clip; and the dead marker is larger with a darker
edge. All gates are re-run after the change, not assumed to still hold.

**Lesson, recorded:** a byte-identity gate proves a render *changed*; it cannot prove the
change is legible, or that a message reaches the reader. Looking at the picture is a
separate check and it caught something four passing gates did not.

## What the fixed panel revealed — the dead points ARE the wave troughs

Seen on `028084_15/31` (dead fraction 0.447) by comparing the OFF and ON frames:

* OFF: the muon side (arc 0 to ~35 cm) is a continuous oscillating band of filled points
  — the periodic dQ/dx wave.
* ON: **the troughs of that oscillation are hollow grey.** The wave's minima fall below
  the live cut, so the chain discards them from every verdict metric, while the peaks
  stay.

This is not the coverage story the failed prediction tested, and it is a better-shaped
hypothesis for the owner's complaint: the profile is hard to read not because charge is
*missing near the stop*, but because an oscillation is being **half-discarded**, leaving a
sparse, peaky remnant for the Bragg and plateau tests to work on.

It also links the two items the h21 measurement had kept apart — the wave (task 24, a
wire-crossing candidate, a tail population) and the dead points. If the wave's troughs are
what the live cut removes, then wave amplitude and dead-point pattern are the SAME
phenomenon measured two ways, on the items where the wave is large.

**Test, stated before it is run,** so this does not become a second story fitted after the
fact: dead points caused by an oscillation are *isolated* — a single sub-cut point with
live neighbours on both sides. Dead points caused by a genuine coverage hole come in
*runs*. The prediction is that the owner's 8 and the 4 splits are enriched in ISOLATED
dead points rather than in runs. If they are enriched in runs instead, this reading is
wrong too and the coverage story returns.

### OUTCOME: THIS PREDICTION FAILED TOO

| group | dead pts | isolated | isolated share | median longest run |
|---|---|---|---|---|
| all 317 with any dead | 4570 | 388 | **0.085** | 5 |
| the owner's 8 | 149 | 10 | **0.067** | 4 |
| the 4 splits | 80 | 16 | 0.200 | 9 |

The owner's 8 are at 0.067 against a 0.085 base — **below** it, not enriched. Their dead
points come in RUNS, not isolated dips: `029107_15/26` longest run 27, `029107_15/40`
run 15, `029107_18/59` run 7. The splits' 0.200 is an artefact of `029107_4/72`, which has
exactly 2 dead points, both isolated — a denominator too small to mean anything.

So the wave-trough reading is true of `028084_15/31`, where it was seen, and does NOT
characterise the owner's items. Combined with the first failure this says something
specific: the owner's items DO have coverage holes (long runs), but those holes are NOT
concentrated in the last 20 cm — which is why test 1 found nothing at the stop and test 2
found runs rather than troughs.

## THE OWNER'S RE-JUDGE ON THE FIXED PANEL — 3 of 8 CHANGED

Tag `own20`, 12 items, saved 2026-09-11 17:11 (sha `bb768440`). The owner said "I finished
scan, still hard to judge". **That is not the same as "nothing changed", and it was first
written up here as if it were — an error, corrected.** Read off the real `label` field
(the schema key is `label`/`choice`, NOT `verdict` — two earlier diffs in this session read
a key that does not exist and reported false "unchanged" rows):

| item | own19 | own20 |
|---|---|---|
| `029107_15/26` | STM_MICHEL attached | **THRU** |
| `029107_24/47` | THRU | **UNCLEAR** |
| `029107_5/93` | THRU | **UNCLEAR** |
| `028084_15/28`, `029107_1/36`, `029107_15/40`, `029107_18/59`, `029107_21/64` | — | unchanged |
| `028084_16/106` (split) | not judged | THRU |
| `029107_20/24` (split) | not judged | UNCLEAR |
| `029107_21/43` (split) | not judged | STM_MICHEL attached, **pin placed rr 13.42, moved 10.0 cm** |
| `029107_4/72` (split) | not judged | STM_ONLY |

So the display fix **did** change judgements — 3 of the 8 re-judged items — and let the
owner settle all four splits that two blind scanners could not. It did not make the items
*easy*: they still report them as hard, with "hard to judge" notes on four.

Two further facts for the record:
* a pin WAS placed this time, on `029107_21/43`, accompanying an STM_MICHEL call — that is
  a genuine owner stop, not the stray `app_edit` kind seen on smx19;
* `revealed_before_label` is true on 12/12 (and was on 8/8 in own19): **these owner looks
  are not blind.** A standing property of the judge queue, not something new, but it
  belongs in the write-up.

**Two pre-registered tests, two failures. Stop here.** Both used the same payload
variable — where the sub-cut points are. A third hypothesis fitted to the same data would
be a fishing expedition, and the honest statement is: *it is not yet known what makes
these items hard to judge.* The owner's words were "the dQ/dx **fit** are not good
enough", and the fit-quality quantities — Bragg contrast, `ks_mu`, `plateau_med`,
`bragg_valid` — have not been looked at at all. That is the untested axis, and the cheapest
next evidence is the owner themselves, now that the panel shows which points the chain
kept.

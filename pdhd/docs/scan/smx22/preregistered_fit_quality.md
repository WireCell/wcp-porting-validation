# Pre-registration — the fit-quality axis (doc pdhd/20 step 3)

Written BEFORE the test is computed. Two earlier pre-registered predictions on the
coverage axis both failed (see `design.md`); this document exists so the third attempt
cannot quietly become a fishing expedition on the same data.

## Why this axis

The owner's words, twice: *"the dQ/dx **fit** are not good enough. Difficult to judge."*
The coverage axis is now measured and negative:

* last-20 cm dead fraction: owner's 8 at **0/8** above 0.25, against a 12.0 % base;
* isolated (wave trough) vs run (coverage hole) structure: isolated share **0.067** for
  the owner's 8 against a **0.085** base, their runs long (27, 15, 7).

And the decisive one: after the display fix showed exactly which points the chain
discards, the owner re-judged all 12 and reported **still hard to judge**. Whatever makes
these items hard, it is not that dead points were drawn as measurements.

## The observation this test is about — and why the owner's 12 CANNOT test it

Looking at the owner's 12 (orientation, not a test), their production reject reasons fall
into two opposite families:

| family | plateau_med | contrast | reject_names |
|---|---|---|---|
| A: "rise, but the plateau is too low" | **below** the 33600 floor | high, 1.27–3.90 | `plateau_off_mip` |
| B: "plateau fine, but no rise" | inside 33600–89600 | low, 0.17–0.75 | `no_bragg` + `shape_flat` |

4 of the owner's 8 sit outside the plateau window, against 27 % of all 317.

**This pattern was found by looking at those 12.** Any test of it on those same 12 is
circular and is not performed. The test below uses the other 305 and a hardness signal
that does not come from the owner.

## The test, fixed now

**Held-out set.** The 305 items that are NOT among the owner's 12.

**Hardness proxy, defined without the owner.**

*Amended 2026-09-11, BEFORE the test was computed and before any result was seen.* The
first wording was internally inconsistent: it let HARD come from confidence alone while
requiring two scanners for EASY, and it then excluded one-scan items — which would discard
nearly the whole held-out set, since only the 26 smx21 and 5 smx20 items were scanned
twice and 4 of those sit in the owner's 12. Resolving that ambiguity after seeing a result
would be indefensible, so it is resolved here:

* **HARD** = the surviving record carries `confidence` low or medium.
* **EASY** = the surviving record carries `confidence` high.
* Confidence exists on every item, so nothing is excluded for scan count.
* **Secondary, reported separately:** the same comparison restricted to the
  double-scanned subset, where scanner disagreement is also available. Small n; reported
  as a consistency check, never as the headline.

Class counts are reported before any comparison.

**Statistic.** For family A, `plateau_med` outside the production window
(0.6–1.6 × mip_dqdx). For family B, `contrast < 0.6` (`bragg_contrast_min`, the chain's
own threshold) with `plateau_med` inside the window. Family membership is computed from
the payload `verdict` block only.

**Prediction.** HARD items are enriched in A ∪ B relative to EASY items.

**What would falsify it.** If the A ∪ B rate among HARD items is within the binomial
spread of the EASY rate, the fit-quality families do not explain hardness either, and the
honest report is that three axes have now been tested and none accounts for the owner's
difficulty — at which point the next step is to ask the owner directly what they are
looking for and failing to find, rather than to test a fourth variable on the same
payloads.

**Null floor, required.** The comparison is reported with the base rate of A ∪ B over all
305 held-out items, so an enrichment claim has a floor to stand against. No threshold in
this document may be adjusted after the result is seen; the two thresholds used (the
plateau window and `bragg_contrast_min` 0.6) are production's own, not chosen by me.

## RESULT — the prediction is SUPPORTED

Computed 2026-09-11 on the 305 held-out items, thresholds exactly as fixed above.

Confidence spread: medium 175, high 120, low 10. HARD n=185, EASY n=120, none unclassified.

| set | A ∪ B | rate |
|---|---|---|
| **null floor, all 305** | 132/305 | **0.433** |
| HARD (low/medium) | 104/185 | **0.562** |
| EASY (high) | 28/120 | **0.233** |

Difference +0.329, SE 0.053, **z = +6.19**. Family A carries it (0.368 HARD vs 0.108 EASY,
a 3.4× enrichment); family B is weak (0.195 vs 0.125).

So after two failures on the coverage axis, the fit-quality axis does separate hard items
from easy ones, and it is **the plateau level, not the Bragg rise**, that does the work.

## Why this result is weaker than its z-score

Stated now, not after someone else points it out:

1. **Shared-cause circularity.** `confidence` is a scanner's own judgement, formed while
   looking at the very profile the chain measures. A flat profile makes both the scanner
   unsure and `contrast` small, so family B is close to tautological. Family A is less so
   — the absolute plateau level is only indirectly visible against the reference curves —
   which may be exactly why A carries the signal and B does not.
2. **Possible restatement of production's verdict.** `plateau_off_mip` is a reject reason.
   If hard items are simply the ones production rejects, the enrichment says nothing new.
3. A z-score on a proxy is not a measurement of the owner's difficulty.

**Confound check, declared BEFORE it is run:** repeat the HARD/EASY comparison *within*
each stratum of the chain's own `is_stm` verdict, and check `confidence` against
`muon_len_cm` and `n_profile_pts`. If the enrichment vanishes inside the strata, finding 1
is that hard items are the rejected ones, and the plateau result is a restatement. If it
survives in both strata, the plateau level carries information about judgeability that
production's accept/reject bit does not.

## CONFOUND CHECK RESULTS — they cut the headline down

**1. A ∪ B is a SUBSET of "production rejected it", by construction.** Among the 65
held-out items the chain accepts (`reject_bits == 0`), A ∪ B is true on **0 of 65** — and it
must be, because plateau-outside-window *is* `plateau_off_mip` and `contrast < 0.6` *is*
`no_bragg`. An accepted item cannot be in either family. So the z = +6.19 above was
inflated by the accept/reject split doing part of the work.

The honest comparison is **within the rejected stratum** (n=240), where there is variance:

| | A ∪ B | rate |
|---|---|---|
| HARD | 104/166 | 0.627 |
| EASY | 28/74 | 0.378 |

difference +0.248, **z = +3.66 — the enrichment SURVIVES.** So A ∪ B is not merely a
restatement of the reject bit: among items production already rejects, it still separates
the ones scanners found hard.

**2. Track length is a serious confound.** HARD items have median `n_profile_pts` **188**
against EASY's **382** — hard items are roughly half the length. A shorter track gives a
noisier plateau median, which is mechanically likelier to fall outside a fixed window. This
was declared as a check in advance; it came back positive, and it means the plateau result
and a track-length result are not yet distinguishable.

(`muon_len_cm` is a payload top-level field, not part of the `verdict` block, so it did not
print; `n_profile_pts` carries the same information here.)

**Decisive follow-up, a direct continuation of check 2:** repeat HARD vs EASY inside bands
of `n_profile_pts`, within the rejected stratum. If the enrichment disappears inside the
bands, the finding is about track length, not the plateau, and must be reported that way.

### Length-stratified result — the finding weakens but does not vanish

Rejected stratum (n=240), split into terciles of `n_profile_pts`:

| tercile | npts range | HARD | EASY | diff | z |
|---|---|---|---|---|---|
| 1 | 20–133 | 47/67 = 0.701 | 5/13 = 0.385 | +0.317 | **+2.17** |
| 2 | 133–355 | 32/57 = 0.561 | 8/23 = 0.348 | +0.214 | +1.79 |
| 3 | 364–1326 | 25/42 = 0.595 | 15/38 = 0.395 | +0.201 | +1.83 |

The confound is confirmed outright: plateau-outside-window falls **0.500 → 0.287 → 0.225**
from the shortest to the longest tercile, so short tracks really are twice as likely to
land outside the window.

The enrichment is positive in **all three** bands at a similar size (+0.20 to +0.32), but
only the shortest reaches z > 2 on its own. Three underpowered bands must NOT be reported
one at a time — that invites quoting whichever suits. The stratified (Mantel–Haenszel)
pooling over the three bands is the single honest summary:

> **pooled MH odds ratio 2.58; CMH χ² = 9.33; z = 3.05; p ≈ 0.002 two-sided.**
> The enrichment SURVIVES adjustment for track length.

### What may and may not be claimed from this

**May:** among the items production already rejects, and at equal track length, the ones
scanners found hard are about 2.6× more likely (in odds) to sit in family A ∪ B — and
family A, the plateau level, carries it. This is the first of three pre-registered tests
to survive.

**May not:**
* *Causation.* `confidence` is a scanner's own judgement formed from the same picture the
  chain measures, so a shared cause is not excluded — least of all for family B, where
  "looks flat" and `contrast < 0.6` are nearly the same statement.
* *Anything about the owner's 12.* They were excluded by construction, being the items
  the pattern was spotted in. The result does not explain their difficulty; it says a
  similar structure exists in data they never touched.
* *A knob.* Nothing here licenses moving `plateau_mip_lo/hi`. The plateau window's
  interaction with short tracks is a hypothesis to test against production, not a
  conclusion, and the obvious confirmatory step — does a length-aware plateau window
  recover stoppers without adding false ones — has not been run.

**Standing caveat:** the z is computed on 305 items scanned mostly once, with confidence
assigned by different scanners under a rubric that changed between rounds (v4 → v5).
That heterogeneity is not modelled here.

## Scope

Read-only. No arm is re-run, no payload, record, label or production config is touched.

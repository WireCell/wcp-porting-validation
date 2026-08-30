# pr/130 item 4 — what the q_miss pool actually is

**Status: MEASURED. The peer's GO stands on concentration, but the census
re-aims the round.** 74% of the addressable q_miss is small cross-cluster EM
fragments assigned to the wrong object — the same defect Part 6 found from the
q_extra side, not a separate under-clustering problem. A fifth of that is
mis-attributed *inside* the main cluster and therefore **not lost at all**
under the owner's own pr/128 precedent, which puts the recoverable figure at
**58.7% of kept q_miss**, not 74%. The one class a threshold could reach (F12,
our oldest shipped walk-add guard) is **measured dead**. The recommended scan
set is not the raw q_miss top-10 — the two largest under-clustered showers turn
out to be main-cluster re-attribution.

Follow-on to `pr130-qmiss-refresh.md` (which answered go/no-go) and
`pr130-qextra-98set.md` Part 6 (the q_extra half). Scoring and census only:
**no knob, no C++, no config, no arm was run.**

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
scripts/pr130_qmiss_census.py > docs/pr/pr130-qmiss-census.txt   # + .tsv
scripts/pr130_qmiss_f12.py    > docs/pr/pr130-qmiss-f12.txt
```

Both import `em_display/em117_score.py` rather than reimplementing it, so every
charge weight is the same number that produced `pr130-{98,141}-score-prod.tsv`
and the totals reconcile by construction (checked: 5.9870e7 vs 5.9869e7, a
rounding delta of 6.8e2). Read-only over `em_labels/`, the calib dumps and
`emprep-pr130q{98,141}/` (M13). Adjudicated events (318769, 415278, 283515,
179369) are carried in the TSV with `adj=True` and excluded from every total.

## Part 1 — first cut: how much of the pool is even addressable

Before asking *why* a segment is missing, ask whether the row describes a
shower the reconstruction built at all. When the labelled shower's **own root
segment** is in the miss list and `--cross-run` matched the row to a shower in
a different cluster, the reco never built that object; the matcher paired two
different things by charge overlap. That row's q_miss is a **seeding** failure
and no admission-time predicate can reach it.

| | 98-set | 141-set | combined |
|---|---|---|---|
| kept q_miss | 4.1864e7 | 1.8005e7 | 5.9869e7 |
| **REROOT rows** | 2 (142421, 409634) | 2 (52044, 397630) | 4 |
| **REROOT q_miss** | 6.5597e6 = **15.7%** | 3.6468e6 = **20.3%** | **17.0%** |

Two independent, disjoint sets agree to within 5 points. **142421 is in it**,
which is an independent confirmation of Part 6's finding that two of its three
condemned showers are *rooted* on a condemned segment — that result was reached
from the q_extra side and shows up here, unprompted, from the q_miss side.

The two `matched = -1` rows (no reco shower overlapped the target at all) are
179369 and 283515 — both already owner-adjudicated, so the kept pool has none.

## Part 2 — the mechanism census of the remaining 83%

For each missing segment, four mutually-exclusive outcomes, from the sidecar
`absorb` tape plus the reconstructed membership:

| class | n | q_miss | share | what it means |
|---|---|---|---|---|
| REROOT | 4 rows | 1.0206e7 | 17.0% | reco never built the shower (Part 1) |
| **DECLINED** | 4 seg | 4.0353e6 | **6.7%** | a shipped guard refused it (Part 3) |
| **SIBLING** | 8 seg | 1.2559e6 | **2.1%** | double-counted with another labelled row (Part 4) |
| **SPLIT** | 63 seg | 1.9383e7 | **32.4%** | built as its own shower, never merged |
| **STOLEN** | 108 seg | 2.3259e7 | **38.9%** | absorbed into a *different* shower |
| UNTOUCHED | 3 seg | 1.7297e6 | 2.9% | no reco shower holds it, no tape names it |

`SPLIT + STOLEN + UNTOUCHED` = **174 segments, 4.4372e7, 74.1% of kept q_miss.**
That is the pool the round is actually about.

## Part 3 — DECLINED is measured dead

The four DECLINED segments are all `SHOWER_ABSORB EXCLUDE`, i.e. the **F12
walk-add track guard**, `guard_excludes` at `PRShower.cxx:827-836`
(`absorb_track_guard`, doc pr/40 round 6). Worth stating precisely: F12 is the
**ancestor** of the pr/123→pr/130 seat family, not one of its additions. None
of this round's flips is implicated.

The denominator is what kills it. Across both manifests the same guard declines
**69 distinct segments carrying 2.852e8 of charge**, and the scanner wants
**1.4% of it**. (Counted by *distinct segment*: a segment can be excluded at
more than one site — 409888 seg 13001, 1.989e7, is taped twice on the 98-set —
so counting records double-bills the denominator.)

| set | exclude records | distinct seg | charge declined | wanted | wanted share |
|---|---|---|---|---|---|
| 98 | 27 | 25 | 8.835e7 | 1 | **1.9%** |
| 141 | 44 | 44 | 1.968e8 | 4 | **2.6%** |
| **both** | 71 | **69** | **2.852e8** | 5 | **2.4%** |

Four of those five are the kept DECLINED rows (4.0353e6 = **1.4%** of the
declined charge); the fifth is 318769, which the owner has already ruled a
*correct* decline.

And no feature separates the wanted from the rest. Ordered by length, the four
wanted declines (14.1, 15.9, 16.5, 16.9 cm) sit *inside* a dense band of
unwanted ones spanning 10.1–26.9 cm. The decisive pair:

| length | set | event | seg | pdg | site | scanner |
|---|---|---|---|---|---|---|
| 16.5 cm | 98 | 176502 | 146220 | 13 | from_vertices | **does not want it** |
| 16.5 cm | 141 | 54341 | 34015 | 13 | from_vertices | **wants it** |

Same length, same PID, same site, opposite verdict. pdg does not separate
(wanted spans 13 and 2212; unwanted spans 13, 211 and 2212); site does not
separate (all four wanted sites are heavily represented among the unwanted).
Above the band the guard is declining 355.2 cm and 413.7 cm muons — any
loosening that admits the wanted 4 admits those too.

**The one decline in this population that carries an owner ruling says the
guard was right**: 318769 seg 19005, a 21.1 cm proton — the pr/129 owner
*reject*, and the longest of the wanted-looking ones.

This is the third time in this round that an admission-side feature search has
come back interleaved (item C's `pass4_angle`, Part 5's ten features, now F12).
Recording it as a pattern rather than a surprise.

## Part 4 — SIBLING reconciles with Part 6, and it is one event

8 segments / 1.2559e6 on the 98-set, **0 on the 141-set**. This is the same
number Part 6 reported for its mirror (98-set 8/138 = 3.7%, 141-set 0/35) and
it is the same single event, **269774**, shower 97197 — the pr/121 ex1-dedup
shape. Two scripts written from opposite directions land on the same 8
segments, which is the check that the definition is stable.

It matters because this charge is **not missing**: `em117_score.py` scores each
labelled shower independently, so a segment the scanner re-homed A→B is
counted in A's `miss` *and* in B's `extra`. Correcting it moves both halves.

## Part 5 — the shape of the 74%, and why it is the same defect as Part 6

The fragment pool (SPLIT + STOLEN + UNTOUCHED), 174 segments / 4.4372e7:

| property | value |
|---|---|
| in a **different cluster** from the reco shower | **158/174 seg (90.8%)**, 73.8% by charge |
| length | median **2.9 cm**; q1 0.6, q3 6.0, max 52.4 |
| under 5 cm | 122/174 seg, 33.0% of the pool charge |
| PID | **116 of 174 are pdg-11**, carrying **75.1%** of the charge |
| SPLIT holders | 26 of 63 are single-segment showers; most ≤ 4 segments |
| STOLEN tape | 99 of 108 carry an absorb record — `pass4_angle` 39, `from_vertices` 20, `in_other_clusters_seg_cone` 18 |

Read together: **the missing charge is small electron-PID fragments, in
clusters other than the one the reco shower lives in, that were either left as
their own one-to-four-segment shower (SPLIT) or actively absorbed into some
other, unlabelled shower (STOLEN).** It is not charge that nothing reached —
99 of 108 STOLEN segments were reached by an absorber, which put them
somewhere else.

*What tape-absence does and does not prove.* The `absorb` tape records
**admissions and F12 excludes only**. It carries no entry for a candidate that
an absorber weighed and dropped on distance, angle or cone. So "55 of 63 SPLIT
segments have no tape" establishes that nothing *admitted or F12-excluded*
them — it does **not** establish that nothing ever considered them, and no
claim in this doc rests on the stronger reading.

That aside, the mechanism is the same one Part 6 named on the q_extra side: *EM
absorbing EM*, mis-partition of one shower's fragments across several objects,
80.5% pdg-11. The 75.1% pdg-11 here is that number seen from the other
direction. **The two halves of the charge error are not two problems. They are
one partition problem, double-billed** — every fragment that lands on the wrong
object is q_miss for the object it left and q_extra for the object it joined.

This has a direct consequence for any knob that comes out of this round, and it
sharpens the peer's warning that the gate must watch q_extra: on this pool a
looser merge does not *add* charge to the event, it *moves* charge between
objects. A q_miss-only score would show a win for any change that redistributes
fragments toward the labelled showers, including changes that are wrong.

## Part 5b — is the charge lost to the CANDIDATE, or only to the labelled shower?

`q_miss` is defined per labelled shower, so it books intra-candidate
re-attribution as loss by construction. The owner's pr/128 metric is
lose / double-count / far-away, and its 105074 precedent settles the test:
**main-cluster membership is a sufficient admission rule; vertex reachability
is not required.** So a fragment held by another node *inside the main cluster*
is already the candidate's energy — nothing is lost, only the node-level
attribution differs.

Measured on `is_main_cluster` from the dump segment and the holder shower's
`conn` class:

| | seg | charge | of the fragment pool | of kept q_miss |
|---|---|---|---|---|
| **in the main cluster** — already the candidate's | 12 | 9.2364e6 | **20.8%** | **15.4%** |
| outside the main cluster — the live loss question | 162 | 3.5136e7 | **79.2%** | **58.7%** |
| holder is conn-4 (cluster >80 cm from the candidate) | 22 | 2.7126e6 | 6.1% | 4.5% |

Two things follow, and they cut in opposite directions:

- **A fifth of the pool is not lost at all.** It is mis-attributed *within* the
  candidate, which the owner's rule already says counts. Any headline built on
  the raw 74.1% overstates the recoverable charge; the honest figure is
  **58.7% of kept q_miss**.
- **It is not a far-away problem either.** Only 6.1% of the pool sits with a
  conn-4 holder, so the "do not count far-away activity" half of the owner's
  rule is not what is binding here. The donors are near.

The split is very uneven between the two classes, and this is what re-aims the
scan: **SPLIT is 36.3% main-cluster by charge, STOLEN only 5.6%.** The largest
SPLIT rows — 284206 (2.271e6) and 314838 (2.117e6), which a raw ranking would
put at the top of any scan set — are *both* main-cluster, i.e. already counted.

## Part 6 — what to do with the round

The peer's go/no-go stands as written: on **concentration** (top-10 holds 78.7%
and 81.9%), a hand-scan is worth a scanner's time. What this census changes is
the question the scan should be asked to answer, and which events it should be
run on.

**Not worth a scan:** "which segments are missing" — that is now enumerated per
segment in `pr130-qmiss-census.tsv`, with mechanism, cluster, main-cluster flag
and holder connectivity attached.

**Not worth a knob:** DECLINED (Part 3, interleaved on every feature) and
REROOT (Part 1, a seeding failure by construction).

**Not worth scanning as loss:** the 20.8% of the pool that is main-cluster
(Part 5b) — under the 105074 precedent that charge is already the candidate's.
If node-level attribution inside the candidate matters for some *other* reason
(PID, pi0 pairing), that is a separate question and should be asked separately.

**The one question the owner's eye can answer and no script can:** for the
SPLIT class **outside the main cluster** — 54 segments, 1.235e7 — *should*
those one-to-four-segment neighbours be one object with the labelled shower, or
is the label store over-marking fragments a physicist would leave separate?
That verdict decides whether the 58.7% is a real defect or a scanning
convention, and everything downstream depends on which.

Recommended scan set, ranked by **outside-main-cluster SPLIT charge** (not by
raw q_miss, and not by raw SPLIT charge — both put main-cluster events on top):

| event | set | charge | seg |
|---|---|---|---|
| 122660 | 98 | 1.788e6 | 2 |
| 54332 | 98 | 1.475e6 | 7 |
| 463565 | 98 | 1.025e6 | 2 |
| 469665 | 98 | 7.605e5 | 2 |
| 76346 | 98 | 7.173e5 | 5 |
| 181050 | 141 | 5.718e5 | 2 |

Five of six are 98-set, which is itself informative: this is where the 98-set's
larger q_miss actually lives.

**Recommendation.** Run the scan on that question and that set, and hold any
merge-side knob until it is answered — a knob designed before it would be tuned
against a metric that cannot tell a fix from a redistribution, and would be
credited for 15.4% of kept q_miss that was never lost. If the owner would
rather not spend scanner time, the honest fallback is that the q_miss half is
now characterised and closed the way item C was: measured, negative on every
addressable class, with the residue pointing at partition and seeding.

## Numbers to carry forward

- Charge error on the 141-set is still **q_extra-dominated** (48.4% q_miss);
  the 98-set's 72.7% is 98-set-only. Do not re-derive; see `pr130-qmiss-refresh.md`.
- **17.0%** of kept q_miss is a shower the reco never built (seeding).
- **6.7%** is F12 declining, and F12 declines 2.852e8 over 69 distinct segments
  to get 4 of them wrong.
- **74.1%** is cross-cluster EM fragment mis-assignment, 75.1% of it pdg-11 —
  but **15.4% of kept q_miss is main-cluster re-attribution that is not lost**,
  so the recoverable figure is **58.7%**.
- **2.1%** is not missing at all — it is double-counted with a sibling row.
- Only **6.1%** of the fragment pool sits with a conn-4 (>80 cm) holder: this
  is a near-candidate partition problem, not a far-away one.

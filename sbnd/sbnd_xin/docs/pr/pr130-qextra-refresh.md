# pr/130 item 1b — the other half of the charge error

**Status: MEASURED. This is the recommended next round.** Scoring only: no
knob, no C++, no config, nothing shipped, no arm launched.

Companion to [`pr130-qmiss-refresh.md`](pr130-qmiss-refresh.md), which asked
"is a q_miss hand-scan worth a scanner's time", answered **GO** on
concentration, and found the premise fails out of sample — on the 141-set
`q_extra` (2.514e7) is the larger half. This doc asks the question that
finding raises and does not answer: **is the other half concentrated, and is
it even a physics quantity?**

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
scripts/pr130_qextra_rank.py > docs/pr/pr130-qextra-rank.txt
```

Reads item 1's two score tables, `em_labels/` (read-only, M13) and the calib
dumps. Nothing was written over; `pr130-qextra-rank.txt` is a fresh file.

## The trap this doc exists to avoid

`q_miss` and `q_extra` are **not symmetric** in `em117_score.py` (lines
188-202):

```python
target = (members | ins) - outs   # members = shower membership AT SCAN TIME
miss   = target - have            # reco dropped it
extra  = have  - target           # reco holds it
```

`miss` is anchored on things the scanner actually saw. `extra` is a
**complement**: a segment lands in it either because the scanner marked it
`out` — an affirmative over-clustering complaint — or merely because it was
not in that shower when the scanner looked. The second class includes every
segment a **later, correct merge** added. Ranking on raw `q_extra` would
score this campaign's own shipped merges as if they were errors.

So the decomposition below is mandatory before any q_extra claim, and
`q_miss` is held to the matching standard (a `miss` carrying an explicit `in`
mark) so the comparison is like-for-like rather than strict on one side only.

## The labels are two-sided — the concern does not void the pool

`emscan-0828-agent5` (the 141-set scan) carries **83 explicit OUT marks
against 59 IN marks**, on 30 of 57 marked showers. Scanners did affirmatively
condemn over-clustering; OUT marks are ordinary, not rare.

| 141-set, kept pool | charge | segments |
|---|---|---|
| q_miss **affirmative** (explicit IN, reco dropped it) | 1.345e7 | 35 |
| q_miss weak (scan-time member, reco dropped it) | 4.554e6 | 22 |
| **q_extra affirmative** (explicit OUT, reco still holds) | **1.731e7** | **22** |
| q_extra weak (never judged, absent at scan time) | 7.827e6 | 151 |
| **affirmative-only split** | **q_miss 43.7% / q_extra 56.3%** | |

**The split gets sharper under the stricter standard, not softer** — q_miss
falls from 48.4% to 43.7%. Over-clustering is the larger half of the 141-set
charge error, and the decomposition strengthens rather than rescues that.

The 98-set says the opposite, also more sharply: **q_miss 82.9% / q_extra
17.1%** (3.429e7 vs 7.056e6). Both are true. The two manifests are disjoint
and genuinely disagree about which failure dominates; the 141-set is the
larger and out-of-sample one.

Note the shape of the weak half: 151 segments for 7.827e6 — a long tail of
small unjudged fragments, which is what an aging label set looks like. The
affirmative half is 22 segments for 1.731e7.

## Concentration — the tightest target list this campaign has had

**Ten events carry all of the affirmative q_extra, and the top four hold 74%
of it in five segments.**

| event | shower | q_aff | nseg | condemned segment(s) |
|---|---|---|---|---|
| 100222 | 113236 (cl113) | 5.973e6 | 1 | 14003 (cl14, **110 cm, pdg 13**) |
| 175896 | 17044 (cl17) | 3.531e6 | 2 | 66037 (cl66, 6 cm, p), 66041 (cl66, 18 cm, e) |
| 489327 | 19005 (cl19) | 1.804e6 | 1 | **19005** (cl19, 23 cm, e) — the shower's own root |
| 499577 | 13009 (cl13) | 1.456e6 | 1 | 95059 (cl95, 7 cm, p) |
| 286655 | 79023 (cl79) | 1.356e6 | 4 | incl. 82063 (cl82, 21 cm, e) |
| 69232 | 20021 (cl20) | 1.166e6 | 1 | 20021 (cl20, 27 cm, e) — own root |
| 350354 | 18092 (cl18) | 1.140e6 | 2 | 18008, 18015 (cl18, own cluster) |
| 278420 | 61027 (cl61) | 6.303e5 | 7 | seven clusters, all 0–5 cm |
| 72786 | 16017 (cl16) | 1.855e5 | 2 | 9009 (cl9, 2 cm), 31033 (cl31, 1 cm) |
| 400504 | 62014 (cl62) | 7.242e4 | 1 | 21003 (cl21, 1 cm) |

**18 of the 22 condemned segments sit in a cluster other than the shower's
own; 4 sit in the shower's own cluster.** The dominant morphology is an EM
shower reaching into a *foreign* cluster and taking a segment the scanner says
does not belong — which is the owner's own third pr/128 metric term ("don't
count far-away over-clustering") measured from the label side.

The single largest item in the whole pool is one segment: a **110 cm pdg-13
track from cluster 14 absorbed into a cluster-113 shower** (evt 100222),
worth 5.973e6, or 34% of the affirmative q_extra by itself.

**Segment ids encode `cluster*1000 + index`** — verified against
`seginfo["cluster"]` for every row above. Id adjacency in the `extra` column
therefore means *different clusters*, not "one cluster, sequential index".
An earlier reading of these lists as within-cluster "contiguous chains" was
wrong and is recorded here so it is not repeated.

## Why this is the round to run

1. **It is the larger half** on the out-of-sample manifest (56.3%), and the
   affirmative decomposition widens the margin instead of closing it.
2. **The targets are already adjudicated.** All 22 segments carry a scanner
   OUT mark. A q_miss round needs a fresh hand-scan before it can start; this
   one starts from judgements already on disk.
3. **It is disjoint from worked ground.** The top-10 q_extra and top-10
   q_miss event lists share **zero** events on the 141-set.
4. **Concentration is extreme**: 10 events, 22 segments, top-4 = 74%.
5. **It is a different question from the last five rounds.** pr/119, pr/128,
   pr/129 and both halves of pr/130 all asked "which candidate should the
   absorber admit" and all came back measured-dead on admission-time
   geometry. This asks what the absorber *did*, against truth, with the
   answer key already written.
6. Three targets connect to open items: **278420** (the parked
   "contiguous far chains" complaint), **286655** (one of the eight
   `stem_backfill_back_guard` firing candidates), and **72786** (the pr/128
   CONTROL sentinel — 1.855e5 of condemned cosmic charge is still inside its
   shower).

## What is NOT established

- **The mechanism.** Which absorber put those 22 segments in is unmeasured.
  That is the first measurement of the round, not a claim of this doc — the
  `pr93_absorb_dbg()` census already tapes every absorb site and would answer
  it on ten events.
- **The 98-set does not support this.** There, under-clustering dominates
  82.9/17.1. A round aimed at q_extra must gate on both manifests.
- **Attribution of the −6.01e6 q_extra drop** noted in the companion doc
  (94392, 52693) still needs a knob-off arm.

Related: [`pr130-qmiss-refresh.md`](pr130-qmiss-refresh.md),
[`130_guard-freed-overcount.md`](130_guard-freed-overcount.md) Part 4-5.

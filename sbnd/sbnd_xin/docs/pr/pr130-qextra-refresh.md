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

## The cross-set comparison is confounded — the two scans mark differently

Before reading "43.7% vs 82.9%" as a statement about the reconstruction:

| label tag | marked showers | IN marks | OUT marks | OUT share |
|---|---|---|---|---|
| `emscan-0827` (98-set) | 33 | 246 | 29 | **11%** |
| `emscan-0828-agent5` (141-set) | 57 | 59 | 83 | **58%** |

These are different marking habits, not a physics difference: the 98-set scan
was IN-heavy (list what belongs), the 141-set scan was OUT-heavy (list what
does not). So **the affirmative split partly measures which scan labelled
which manifest**, and the claim has to be stated as a claim about a label
set:

> On the 141-set's labels, affirmative over-clustering (1.731e7) exceeds
> affirmative under-clustering (1.345e7).

It is **not** established that over-clustering exceeds under-clustering in
the detector. That would need one scan protocol applied to both manifests.

**What survives the confound intact** is the absolute pool, which needs no
share at all: 22 segments carrying 1.731e7 of charge that a scanner
explicitly condemned and the reconstruction still holds. The 98-set adds a
further 22 segments / 7.056e6 on its own labels. Those 44 segments are the
target list, and reasons 2-6 below do not depend on any percentage.

## Concentration — the tightest target list this campaign has had

**Ten events carry all the affirmative q_extra on the 141-set; the top four
hold 73.7% of it in five segments.** (Not "top-10 = 100%" — with ten
contributing events that is true by construction, the same empty statistic as
the old "top-25 = 100%".)

| event | shower | q_aff | nseg | condemned segment(s) | absorber |
|---|---|---|---|---|---|
| 100222 | 113236 (cl113) | 5.973e6 | 1 | 14003 (cl14, **110 cm, pdg 13**) d=38 a=4° | `pass4_proximity` |
| 175896 | 17044 (cl17) | 3.531e6 | 2 | 66037 (6 cm, p), 66041 (18 cm, e) d≈34 a≈10° | `pass3_cone` |
| 489327 | 19005 (cl19) | 1.804e6 | 1 | **19005** (23 cm, e) — the shower's own root, a=172° | own root |
| 499577 | 13009 (cl13) | 1.456e6 | 1 | 95059 (cl95, 7 cm, p) d=30 a=22° | `pass3_cone` |
| 286655 | 79023 (cl79) | 1.356e6 | 4 | four clusters, d=68–86, **a=137–150°** | `pass4_angle` |
| 69232 | 20021 (cl20) | 1.166e6 | 1 | 20021 (27 cm, e) — own root | own root |
| 350354 | 18092 (cl18) | 1.140e6 | 2 | 18008, 18015 (own cluster) | `conn3_unreachable`, `pass3_cluster_map` |
| 278420 | 61027 (cl61) | 6.303e5 | 7 | seven clusters, **d=98–125 cm**, a=3–11° | `pass4_angle` |
| 72786 | 16017 (cl16) | 1.855e5 | 2 | 9009, 31033 (1–2 cm) | `pass4_angle` |
| 400504 | 62014 (cl62) | 7.242e4 | 1 | 21003 (1 cm) | `pass4_angle` |

**Two failure modes, not one.** By charge, **76% (1.320e7, 18 segs)** is the
shower reaching into a *foreign* cluster; **24% (4.111e6, 4 segs)** sits in
the shower's own cluster — mis-rooting or over-extent, which is a different
bug and must not be scoped away. Quoting only "18 of 22" hides that.

The single largest item is one segment: a **110 cm pdg-13 track from cluster
14 absorbed into a cluster-113 shower** (evt 100222), 5.973e6, 34.5% of the
affirmative pool by itself.

**Segment ids encode `cluster*1000 + index`** — verified against
`seginfo["cluster"]` on every row. Id adjacency in the `extra` column
therefore means *different clusters*, not "one cluster, sequential index". An
earlier reading of these lists as within-cluster "contiguous chains" was
wrong and is recorded here so it is not repeated.

## The mechanism is on disk, not a guess

`marks_detail[shower]["marked"][seg]["absorbed_by"]` records which absorber
placed each condemned segment. Attribution by charge:

| absorber | charge | share | segs |
|---|---|---|---|
| `pass4_proximity` (direct) | 5.973e6 | 34.5% | 1 |
| `pass3_cone` (direct) | 4.987e6 | 28.8% | 3 |
| *(none — shower's own root/extent)* | 2.970e6 | 17.2% | 2 |
| `pass4_angle` (direct) | 2.244e6 | 13.0% | 14 |
| `conn3_unreachable` (walk_add) | 6.351e5 | 3.7% | 1 |
| `pass3_cluster_map` (direct) | 5.054e5 | 2.9% | 1 |

Two leads fall straight out of the `dist`/`angle` columns:

- **`pass4_angle` admits backward.** All four of 286655's segments come in at
  **137–150°** — beyond the 110° that `stem_backfill_back_guard` (pr/120)
  declines on. The backward test exists in one absorber and not in this one.
  286655 is also one of the eight events that guard fires on, so the same
  event is being fixed by one path and re-broken by another.
- **`pass4_angle` tier 2 admits far.** All seven of 278420's segments arrive
  at **98–125 cm** with angles of 3–11° — well-aligned but distant. That is
  the parked "contiguous far chains" complaint and the owner's pr/128 "don't
  count far-away over-clustering" term, now with an absorber name on it.

Note the count/charge inversion: `pass4_angle` placed 14 of the 22 segments
but only 13% of the charge, while `pass4_proximity` and `pass3_cone` placed 4
segments and 63%. A round that optimises for segment count would work on the
wrong absorber.

## Why this is the round to run

1. **The pool is pre-adjudicated.** All 22 segments carry a scanner OUT mark.
   A q_miss round needs a fresh hand-scan before it can start; this one
   starts from judgements already on disk.
2. **It is disjoint from worked ground** — the top-10 q_extra and top-10
   q_miss event lists share **zero** events on the 141-set.
3. **Concentration**: 10 events, 22 segments, top-4 = 73.7%, and one segment
   is 34.5%.
4. **The mechanism is already named** (above), so the round opens on a fix
   hypothesis rather than a census.
5. **It is a different question from the last five rounds.** pr/119, pr/128,
   pr/129 and both halves of pr/130 all asked "which candidate should the
   absorber admit" and all came back measured-dead on admission-time
   geometry. This asks what the absorber *did*, against truth, with the
   answer key already written.
6. Three targets connect to open items: **278420** (parked far-chain
   complaint), **286655** (a `stem_backfill_back_guard` firing candidate),
   **72786** (the pr/128 CONTROL sentinel — 1.855e5 of condemned cosmic
   charge still inside its shower).

## What is NOT established

- **That over-clustering dominates the detector.** See the confound above.
  The 141-set claim is about the 141-set's labels.
- **The 98-set does not support a q_extra-first reading of the split**
  (82.9 / 17.1 there), though it contributes 22 more condemned segments.
  A round aimed at q_extra must gate on both manifests.
- **`absorbed_by` is the label store's record, not a re-run.** It should be
  reconfirmed against a live `pr93_absorb_dbg()` census before a knob is
  designed on it.
- **Attribution of the −6.01e6 q_extra drop** noted in the companion doc
  (94392, 52693) still needs a knob-off arm.

Related: [`pr130-qmiss-refresh.md`](pr130-qmiss-refresh.md),
[`130_guard-freed-overcount.md`](130_guard-freed-overcount.md) Part 4-6.

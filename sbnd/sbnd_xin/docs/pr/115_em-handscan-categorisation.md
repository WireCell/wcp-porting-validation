# pr/115 — categorising the em_display hand scan into work buckets

**Scope.** Read the 97 hand-scan labels the owner wrote on port 5017 (scan tag
`emscan-0827`, display doc [pr/114](114_em-pi0-handscan-display.md), sample
audit [pr/113](113_em-shower-pi0-long-muon-coverage-audit.md)) and divide the
events into the groups asked for: **1 EM-shower under-clustering**, **2 EM-shower
over-clustering**, **3 π⁰ with a known vertex**, **4 π⁰ without a known vertex**,
plus **good events** and **events with an incorrect ν vertex**.

**No code is changed.** No C++, no jsonnet, no change to the display — this
document and its classifier only *read* what is already on disk, and the labels
are opened read-only (CLAUDE.md M13). **No A/B gate is owed.**

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
./em_display/em114_categorize.py                                  # the table below
./em_display/em114_categorize.py --tsv docs/pr/pr115-handscan-buckets.tsv
```

Inputs, both read-only:

* `sbnd_xin/em_labels/emscan-0827/labels-evt*.json` — 97 label files, written
  2026-08-28 01:26 → 15:51 UTC.
* `sbnd_xin/em_display/em114-manifest.tsv` — the 98-event sample, including the
  `scan_note` column from the pr/113 survey pass.

Row table: [`pr115-handscan-buckets.tsv`](pr115-handscan-buckets.tsv), one line
per event with every field the rule looked at plus the reason it fired.

---

## 1. What was scanned

98 events in the manifest; **97 carry a label**. **`evt282909` was never
opened** — it is an `mcp2k` NCπ⁰ event with a reco π⁰ of 669.7 MeV, i.e. one of
the more interesting ones, and it is the single gap in the scan. Every count
below is out of 97 unless it says otherwise.

| sample | origin | n |
|---|---|---|
| `nuecc48` | νe CC | 48 |
| `ncpi0` | NCπ⁰ | 19 |
| `mcp2k` | NCπ⁰-enriched | 17 |
| `mcp1k` | NCπ⁰-enriched | 10 |
| `mcp1k` | owner-picked | 4 |

## 2. How an event is categorised — and why this needs a rule at all

**The display's own verdict control was used once.** `em.verdict` is set on
**1 of 97** records (`evt64591`, "correct") and `confidence` on one other
(`evt169356`, "certain"). The scanner recorded judgements as **free-text notes**
(65 events) and as **segment marks** (25 events) instead. So every bucket below is
*inferred*, and the inference rule — not the table — is the thing to review.
It lives in `em_display/em114_categorize.py:classify`, and its precedence is:

0. **Never scanned wins.** Absence of a correction is not evidence of
   correctness.
1. **A named direction outranks everything else.** If the note says
   "overclustering" or "not clustered together", that decides the bucket, even
   when the note also complains about the vertex or says the event is hard.
2. **Then an abandoned scan**: a vertex complaint *and* no marks *and* no π⁰
   block. The conjunction is the point — `evt54332`'s note says the vertex is
   wrong and the scanner corrected the clustering anyway, so a vertex complaint
   on its own must not send an event to the not-actionable pile.
3. **Then the marks**, counting only the ones that are not no-ops.
4. **Then "good"**, only for a note that is exactly *good / OK / very good /
   pretty good* with no correction attached.

### 2.1 Three places where the naive reading is backwards

**(a) The note governs direction, not the mark count.**

* `evt463565` carries **26 IN marks** and the note *"overclustering of two
  closely gamma, not sure if one can separate them, but one can at least
  include all the pieces."* The owner is recording that the right fix is
  **separation** and that including the pieces was the achievable thing
  instead. Mark direction alone files this under under-clustering, which is the
  opposite of what was written.
* `evt84229` carries **8 IN marks**, **7 of them on segments shower 69134
  already owned**, and the note *"69134 overclustered another gamma in (segment
  of 69150 and connected pieces)"*. The scanner was using IN as a **highlighter**
  to point at the intruder, not as a correction.

**(b) A mark is not a mark.** `marks_detail[shw].marked[seg].owner` records
which shower owned the segment at scan time, and it splits the **246 IN marks**
into four different pieces of work (there are 29 OUT marks, tabulated below):

| IN marks | n | what it means |
|---|---|---|
| from another multi-segment shower (`merge`) | 168 | one shower was split in two |
| from a one-segment shower of its own (`absorb-orphan`) | 63 | an orphan stub never absorbed |
| already a member (`highlight`) | 14 | no-op — the scanner was pointing |
| from nobody (grey) | 1 | a truly unowned segment |

| OUT marks | n | |
|---|---|---|
| on a member | 18 | the real "remove this" statement |
| on a non-member | 11 | see (c) |
| — of which on the shower's own seed | 4 | `evt47212`, `evt54332`, `evt76346`, `evt314838` |

**(c) OUT on a non-member is a no-op for the energy but not for the intent.**
`marks_energy` skips it (`if kind == "out" and sid not in mem: continue`). On
`evt142421` the scanner selected a **one-segment stub, shower 7010**, and used
it as a scratch pad: 33 segments IN and 10 OUT, **all 43 owned by shower
108104**. The note — *"The OUT segments can form a separate gamma, and then they
could be a π⁰"* — says exactly what that means: shower 108104 is **over**-clustered
and the OUT set is the second gamma. Counting IN marks would file it under
group 1. Tagged `split-by-proxy`; `evt47212` is the other one.

### 2.2 Known vertex vs not (groups 3 and 4)

`vertex_how == "main_vertex"` is **the widget default** (`vtx_mode_group …
active=0`), so it means "the scanner built a π⁰ and did not override the
vertex" — weaker evidence than a positive act. An event is put in **group 4**
only on positive evidence:

* the scanner set the **`no_vertex_ncpi0`** flag — `evt169626`, `evt347129`,
  `evt76346`; or
* the π⁰ vertex came from **back-projecting the two gamma rays** (same three); or
* the scanner **placed the vertex by hand far from the reco's**. The distances
  decide, and they separate cleanly:

| event | hand-placed π⁰ vertex − reco main vertex | reading |
|---|---|---|
| `evt64591` | **177.0 cm** | vertex rejected → group 4 |
| `evt54332` | **67.9 cm** | vertex rejected → group 4 |
| `evt281165` | **47.9 cm** | vertex rejected → group 4 |
| `evt47212` | **3.5 cm** | vertex *confirmed* by hand → group 3 |

The threshold in the script is 15 cm — far outside vertex resolution, far
inside the smallest rejection actually seen.

---

## 3. The result

97 scanned + 1 unscanned = 98.

| bucket | n | π⁰ vertex usable | π⁰ vertex not usable | no π⁰ |
|---|---|---|---|---|
| **1 under-clustered** | **18** | 8 | 3 | 7 |
| **2 over-clustered** | **7** | 1 | 1 | 5 |
| **1+2 both** | **3** | 1 | 1 | 1 |
| good (no major change) | 37 | 0 | 0 | 37 |
| vertex-bad (not actionable) | 18 | 0 | 0 | 18 |
| undecidable / too busy | 1 | 0 | 0 | 1 |
| scanned, no clustering correction | 13 | 10 | 1 | 2 |
| not scanned | 1 | 0 | 0 | 1 |

**π⁰ axis, cutting across the above: 26 events carry a π⁰ the scanner built —
20 in group 3, 6 in group 4.**

The clustering bucket and the π⁰ group are **orthogonal**, which is why they are
two columns and not one list: `evt76346` is a no-vertex π⁰ *and* has 16 IN and
4 OUT marks; `evt54332` says the vertex is wrong, is over-clustered, and carries
two π⁰ candidates.

Sample composition is not uniform and says something on its own:

| bucket | nuecc48 | ncpi0 | mcp1k | mcp2k |
|---|---|---|---|---|
| 1 under-clustered | 7 | 2 | 7 | 2 |
| 2 over-clustered | 0 | 3 | 0 | 4 |
| 1+2 both | 1 | 1 | 0 | 1 |
| good | **34** | 3 | 0 | 0 |
| vertex-bad | 6 | 4 | 3 | 5 |
| no clustering correction | 0 | 6 | 4 | 3 |

**34 of the 37 "good" events are νe CC**, and **every over-clustered event is
NCπ⁰**. Single-shower νe CC is the case the clustering handles; two overlapping
gammas is where it breaks.

---

## 4. Group 1 — EM-shower under-clustering (18 events)

Ordered by how much the scanner added relative to what the reco had. The
`+n/−n` counts exclude no-ops.

| event | sample | π⁰ | what was marked | kind |
|---|---|---|---|---|
| `evt347129` | mcp1k | 4 | shw 11000 (**1** mem) **+9** | stub |
| `evt409634` | mcp1k | 3 | shw 69032 (**2** mem) **+10**, 9 of them from shw 27015 | stub |
| `evt281485` | mcp1k | 3 | shw 88090 (**2** mem) +5 | stub |
| `evt342199` | nuecc48 | 3 | shw 74101 (**3** mem) +7, 4 from shw 25109 | stub |
| `evt444187` | nuecc48 | — | shw 19079 (**3** mem) +4, 3 from shw 19082 | stub |
| `evt105946` | ncpi0 | 3 | shw 55063 (16 mem) +5 ; shw 56056 (3 mem) +2 | stub, multi |
| `evt469665` | nuecc48 | — | shw 15003 (18 mem) +12 | tail |
| `evt168596` | nuecc48 | — | shw 14153 (41 mem) **+20, 19 of them from shw 14058** | tail |
| `evt284235` | mcp1k | 3 | shw 6004 (7 mem) +1 ; shw 74027 (6 mem) +2 | tail, multi |
| `evt423981` | nuecc48 | — | shw 12095 (18 mem) +5, 4 from shw 12038 | tail |
| `evt122660` | nuecc48 | — | shw 9110 (37 mem) +10, 6 from shw 47050 | tail |
| `evt173093` | mcp2k | 3 | shw 12007 (13 mem) +3 | tail, multi |
| `evt415278` | mcp2k | — | shw 23012 (37 mem) +7 ; shw 23037 (43 mem) +5 | tail, multi |
| `evt21073` | ncpi0 | 3 | shw 60081 (38 mem) +7, 6 from shw 31023 | tail |
| `evt169626` | mcp1k | 4 | shw 53069 (30 mem) +4 | tail |
| `evt166870` | mcp1k | 3 | shw 87058 (10 mem) +1 — note: *"85045 should be an EM shower, part of π⁰"* | tail, **pid** |
| `evt64591` | mcp1k | 4 | shw 83044 (17 mem) +1 | tail |
| `evt235435` | nuecc48 | — | **no marks**; note: *"The entire segments are one EM shower from the main nu vertex, not clustered together"* | no-marks |

Two different algorithmic failures live here and should be worked separately:

* **stub (6 of the 18)** — the reco produced a seed of 1–3 segments and the scanner
  attached the object to it. The shower was **never grown**. `evt347129`
  (1 member → +9) and `evt409634` (2 → +10) are the extreme cases.
* **tail (11 of the 18)** — a real shower, 6–43 members, that **lost pieces**;
  the acceptance gate stopped too early.

And across both, **merging beats absorbing orphans by mark count: 168 marks
take a segment from a real neighbouring shower against 63 that pick up a
one-segment stub** (20 and 23 events respectively — the same events usually do
some of each). Most of what is missing is not loose fragments, it is **a second
shower**. `evt168596` is the pure case — 19 of the 20 marks come
from **one** other shower, 14058. "These two showers are one object" is a
different repair from "pick up 20 loose fragments", and the record says which.

`evt166870` is a **PID** case rather than a pure clustering one: the note asks
for an object the reco did not call EM to be promoted to a gamma. It is the
event that motivated the `is an EM shower (reco PID wrong)` verdict in pr/114.

---

## 5. Group 2 — EM-shower over-clustering, needs separating (7 events)

| event | sample | π⁰ | evidence |
|---|---|---|---|
| `evt142421` | ncpi0 | — | 43 of shower **108104**'s segments partitioned through a stub: 33 one gamma, 10 the other. *"The OUT segments can form a separate gamma, and then they could be a π⁰"* |
| `evt314838` | ncpi0 | — | shw 110088 (26 mem): **7 marked OUT**, a contiguous block 110083–111092 including the seed. *"Overclustering, The OUT segments should be a separate gamma cluster, then form a π⁰ likely"* |
| `evt54332` | mcp2k | 4 | *"overclustering a track ... 16014, also nu vertex is wrong"* — shower 16014's **own seed** marked OUT |
| `evt84229` | ncpi0 | — | *"69134 overclustered another gamma in (segment of 69150 and connected pieces)"* — 50-member shower, 7 IN marks used as a highlighter |
| `evt47212` | mcp2k | 3 | segment 2103 marked OUT of its **own 25-member shower** twice, from both showers. Both reco π⁰ gammas are PID'd **proton** and the reco π⁰ mass is 764 MeV |
| `evt176502` | mcp2k | — | *"Significant EM shower overclustering, not sure if one can improve ... not easily label"* — diagnosis only, no marks |
| `evt281567` | mcp2k | — | *"nu vertex is wrong ... , 95128 has an overclustering issue of a EM shower isolated with a main cluster segment"* — diagnosis only, no marks |

The last two are kept **here** rather than in the vertex/give-up piles: they are
the only over-clustering diagnoses the scanner recorded for those events, and
burying them loses the information. They are tagged `no-marks` — there is
nothing at segment level to fit against.

**`seed-out` (4 events: 47212, 54332, 76346, 314838)** is a distinct signal: the
shower's own **seed segment** was rejected. That is not "one member is wrong",
it is "this shower was started from the wrong thing" — usually a track stem
(`evt54332`: *"overclustering a track"*; `evt47212`: gammas PID'd as protons).

---

## 6. Both directions in one event (3 events)

| event | sample | π⁰ | |
|---|---|---|---|
| `evt269774` | nuecc48 | 3 | **the cleanest transfer in the sample**: 5 segments OUT of shower 13237 (48 mem), 8 IN to shower 97197 (8 mem) — **5 of those 8 come from 13237**. One boundary drawn in the wrong place. Reco π⁰ mass 1444 MeV |
| `evt76346` | mcp2k | 4 | shw 14059 (7 mem) +11/−3 ; shw 40030 (**1** mem) +5/−1 including its own seed. Both starts moved, axis aimed by hand |
| `evt463565` | ncpi0 | — | 26 IN on shw 13001 (15 mem), 15 from shower 115088; note says the two gammas are **merged** and separation is the right fix but was not attempted. Counted here because the note names both |

---

## 7. Group 3 — π⁰ with a usable vertex (20 events)

`evt21073` `evt47212` `evt71372` `evt99838` `evt105946` `evt166870` `evt169356`
`evt173093` `evt269774` `evt278794` `evt281485` `evt281639` `evt284235`
`evt285567` `evt342199` `evt359980` `evt399052` `evt409634` `evt506114`
`evt506746`

Ten of these sit in the **"no clustering correction"** bucket — the scanner
built the π⁰, did not mark a single segment and did not write a note. On **six** of
them (`evt99838` `evt169356` `evt278794` `evt281639` `evt359980` `evt399052`)
the hand-built mass reproduces the reco's `kine_pio_mass` to the decimal, i.e.
**the scanner confirmed the reconstruction's own pairing**. That is a positive
result, and it is what group 3 is mostly made of. Of the remaining four,
`evt71372` agrees to 3 MeV, and `evt285567`, `evt506114` and `evt506746` do
**not** — the reco puts them at 19.1, 87.9 and 10.1 MeV against hand-built
147.9, 158.9 and 142.6, so the pairing was rebuilt even though no segment was
re-marked.

The other ten carry a clustering correction as well, and appear in §4–§6.

## 8. Group 4 — π⁰ without a usable vertex (6 events)

| event | sample | how the vertex was got | hand mass (axis / vertex) | reco π⁰ mass |
|---|---|---|---|---|
| `evt169626` | mcp1k | back-projection, flagged | 144.6 / 145.4 | 145.8 |
| `evt347129` | mcp1k | back-projection, flagged | 130.7 / 131.1 | **0.0** |
| `evt76346` | mcp2k | back-projection, flagged | 40.8 / 40.8 | 158.1 |
| `evt64591` | mcp1k | hand-placed, 177 cm off | 174.0 / 149.6 | 149.6 |
| `evt54332` | mcp2k | hand-placed, 68 cm off | 147.1 / 111.9 | 133.7 |
| `evt281165` | mcp2k | hand-placed, 48 cm off | 137.4 / 148.1 | 148.1 |

Only three of the six were flagged `no_vertex_ncpi0` in the display. The other
three are recognisable only from the hand-placed vertex being tens of cm away
from the reco's — **the flag under-counts this class by a factor of two**, which
is a finding about the display as much as about the events. If more no-vertex
NCπ⁰ scanning is planned, the flag should be set (or offered) whenever the
manual vertex lands far from the main vertex.

---

## 9. Good events (37) — and an honest caveat

34 νe CC, 3 NCπ⁰; notes *good / very good / pretty good / OK*, no marks, no π⁰
built. These are the events that need no major change.

**They were a fast pass.** All 37 were saved between **14:47 and 15:51 UTC**,
26 of them inside the 15:38:05–15:51:41 block — 26 events in under 14 minutes. That
does not make them wrong: an event with one clean shower off a good vertex is
decidable at a glance, and that is what the νe CC sample mostly is. But "good"
here means *nothing jumped out*, not *verified segment by segment*, and it
should not be quoted as a per-segment efficiency.

**Three of them contradict the pr/113 survey note** and the contradiction is
left standing rather than averaged away:

| event | pr/113 survey note | hand scan |
|---|---|---|
| `evt256587` | "track arm got ided as electron" | "good" |
| `evt90055` | "shower stem got ided as proton" | "good" |
| `evt389538` | "capture second neutrinos …" | "good" |

The most likely reconciliation: **this display asks about segment membership**,
so "good" is scoped to *clustering* and is silent on PID. Both records are then
right about different questions. Tagged `note-conflict`; they are the first
place to look if a PID-focused pass is run.

## 10. Not actionable — wrong ν vertex (18) and too busy (1)

`evt30504` `evt38856` `evt52672` `evt56982` `evt76350` `evt111412` `evt114446`
`evt116962` `evt163543` `evt165157` `evt172942` `evt174752` `evt176986`
`evt180801` `evt259542` `evt281781` `evt394532` `evt475096` — notes ranging from
*"incorrect nu vertex"* to *"wrong nu vertex, give up"*, no marks, no π⁰. The
shower question starts at the vertex, so with the wrong starting point the
in/out judgement is unanswerable; this is the escape the scan was given up
front and 18 of 97 events used it.

`evt396222` — *"very busy events, difficult to scan"*.

**These 19 are not a clustering result and must not be counted as failures of
it.** They are a *vertex* result: 19 % of a sample chosen for EM content is out
of reach of shower scanning because of vertex finding. That is the number to
carry into vertex work (pr/104–pr/112), not into shower work.

## 11. What the π⁰ masses say

Over the 26 π⁰ events, comparing the scanner's π⁰ with the reconstruction's own
`kine_pio_mass` (vertex convention where present):

| | in 100–180 MeV | median \|m − 135\| |
|---|---|---|
| hand-built | **22 / 26** | **14.3 MeV** |
| reco `kine_pio_mass` | 15 / 26 | 26.6 MeV |

**This is not a head-to-head between two independent estimators, and must not
be quoted as one.** The scanner chose the pairing *while watching the mass
update live* — that is what the two mass conventions in the π⁰ panel are for —
so the hand column is tuned against the target and its median is not a
resolution. The defensible reading is the first column: **on 22 of 26 events a
pairing consistent with a π⁰ exists in the event and the scanner found it; the
reconstruction found one on 15.** That is a statement about whether the pieces
are there to be paired, which is exactly the question the clustering work has
to answer.

The reco's failures need no such caveat — they are wrong on their own terms: **1444 MeV** (`evt269774`), **764**
(`evt47212`), **330** (`evt166870`), **315** (`evt342199`), and at the other end
**0.0** (`evt347129`), **10.1** (`evt506746`), **17.1** (`evt173093`), **19.1**
(`evt285567`). Every one of those events is in a correction bucket above — the
mass is the symptom, the clustering or the pairing is the cause.

### 11.1 The four hand-built masses that are still far from 135

`evt76346` (40.8), `evt281485` (67.2), `evt409634` (83.4), `evt21073` (218.5).
Three of the four are **`stub`** events, and the reason is visible in the
record: **the scanner marked the missing segments but did not switch their
charge into the gamma energy.** 22 of 52 gamma slots have
`energy_includes_marks` on; the rest carry the delta un-applied. Applying each
stored `energy_marks_delta` at the stored opening angle:

| event | E₁ | Δ₁ | E₂ | Δ₂ | θ | mass stored | mass + Δ |
|---|---|---|---|---|---|---|---|
| `evt76346` | 246.7 | −9.4 | **5.0** | **+98.9** | 70.7° | **40.8** | **181.8** |
| `evt409634` | 180.6 | 0 | **39.1** | **+105.1** | 59.6° | **83.4** | **160.3** |
| `evt269774` | 916.1 | −77.9 | 250.8 | +90.3 | 18.5° | 154.3 | 172.1 |
| `evt54332` | 182.5 | +52.5 | 65.0 | 0 | 61.9° | 111.9 | 127.0 |
| `evt284235` | 133.6 | +1.6 | 87.7 | +16.7 | 85.2° | 146.5 | 160.8 |
| `evt169626` | 537.1 | +23.6 | 107.9 | 0 | 35.2° | 145.4 | 148.6 |
| `evt64591` | 298.1 | +4.1 | 50.7 | 0 | 74.9° | 149.6 | 150.6 |

This is a **first-order** estimate and is labelled as one: it applies the
recorded charge delta at the *recorded* opening angle, and does not re-fit the
shower start or axis after the segments are added — θ would move too. It is
enough to show that on `evt76346` a 5 MeV gamma is missing 99 MeV of marked
charge, and that these masses are not evidence against the hand scan.

---

## 12. Caveats carried, not resolved

* **`evt282909` was never scanned.** One of the more interesting NCπ⁰ events
  (reco π⁰ 669.7 MeV) and the only gap.
* **The verdict radio and the confidence radio went unused** (1/97 each). If
  the buckets here are to be maintained rather than re-inferred, the cheapest
  fix is to record them at scan time.
* **The `no_vertex_ncpi0` flag under-counts by 2×** (§8).
* **"Good" is a fast pass** (§9) and is scoped to clustering, not PID (§9 table).
* **One mark in the sample is a pure no-op**: `evt47212` marks segment 2103 OUT
  while shower 70038 is selected, and 2103 was never a member of 70038. The
  same statement is recorded correctly against shower 2103, so nothing is lost.
* **The one verdict on record disagrees with its own marks.** `evt64591` is
  marked `correct` and also carries one orphan segment marked IN, which is why
  it appears in group 1. A one-segment addition to a 17-member shower is a
  small enough correction that both readings are defensible; it is listed under
  under-clustering because a mark is a positive act and the radio may simply
  have been left set.
* **`evt21073` and `evt269774` have gammas in the slots but no stored π⁰
  candidate** — their masses come from the live pairing, not from
  `pio.candidates`.

## 13. Where to start

1. **Split showers (`merge` tag, 20 events, most of group 1)** — the largest
   and most uniform class, and the one with a clean signature: most of a
   shower's missing charge sits in **one** neighbouring shower. `evt168596`
   (19 of 20 marks from shower 14058) is the reference case.
2. **Stub showers (`stub` tag, 8 events — 6 in group 1, plus `evt142421` and
   `evt76346`)** — a seed of 1–3 segments that was never grown. A different
   failure, small and self-contained, and it is what is holding three of the
   four bad hand-built π⁰ masses (§11.1).
3. **NCπ⁰ over-clustering, 7 events** — two gammas merged into one shower.
   Hardest, but it is where the reco π⁰ mass fails worst, and `evt314838` and
   `evt142421` both come with the intended partition already marked.
4. **`seed-out`, 4 events** — showers started from a track stem. Cheap to test
   for, and `evt47212` shows it can put PID and the π⁰ mass 600 MeV out.
5. **Vertex, 18+1 events** — not shower work. Hand to pr/104–pr/112.

## 14. Files

| file | |
|---|---|
| `sbnd_xin/em_display/em114_categorize.py` | the classifier — the reviewable rule |
| `sbnd_xin/docs/pr/pr115-handscan-buckets.tsv` | one row per event, every field the rule read |
| `sbnd_xin/docs/pr/115_em-handscan-categorisation.md` | this document |

## 15. Round 2 — a plan of attack for EM clustering and π⁰ pairing

Sections 1–14 say *what* is broken. This one says *where in the code*, *what
prior art already covers it*, and — the part that was missing — *how we would
know a change helped*. It is a **plan only**: no C++, no jsonnet, no knob flips.

**Scope, as set by the owner.** In: groups 1–4 (EM clustering and π⁰ pairing).
Out: PID and ν-vertex finding, hence the 18 wrong-vertex events (§10), the
too-busy event, and the 3 `note-conflict` events (§9). **In despite sounding
like vertex work:** group 4 — back-projecting two shower axes to a *decay*
vertex is π⁰ reconstruction, not ν-vertex finding.

### Repro

Every line number below was re-checked against the working tree immediately
before this section was written. The file has grown to 5295 lines, so any
citation from an older doc will not land.

```
cd toolkit/clus/src
grep -n 'PatternAlgorithms::' NeutrinoShowerClustering.cxx     # the 16 methods
awk 'NR>=4375 && NR<=5295' NeutrinoShowerClustering.cxx \
  | grep -n '^ *\(shower_clustering\|examine_\|id_pi0\|stem_backfill\|if (m_\)'
grep -n 'shower_\|pi0_\|pio_' ../inc/WireCellClus/NeutrinoPatternBase.h
grep -n '= true,\|= false,' ../../cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet
```

### 15.1 Three things established first, each of which changed this plan

**(a) The toolkit's own review doc is stale, and its bug list is closed.**
`clus/docs/patternrecognition/shower_clustering_review.md` lists B.1–B.5 + L.1
against a 3310-line file. Verified in the current tree:

| ID | claim | state today |
|---|---|---|
| B.1 | `calculate_num_daughter_showers(…, false)` | superseded by pr/33 F1 — both counts computed, knob selects (`:805`, `:3314`); `daughter_count_proto_examine_showers` is ON in SBND |
| B.2 | `continue` should be `break` | fixed, `:4255` |
| B.3 | missing `pdg==11` | fixed, `:4135` |
| B.4 | missing post-merge update | fixed, `:2420-2421` |
| B.5 | direction from the shower's own start vertex | fixed, `:3810-3817` |
| L.1 | hard-coded `0.511 * units::MeV` | **survives**, now at `:1400` (not `:203`) |

None of B.1-B.5 is actionable work. L.1 is cosmetic -- the value is correct,
only the provenance differs from the dependency-injection pattern used
elsewhere -- and is carried in §15.10 rather than treated as a finding.
Reported so the next reader does not chase them; the doc is not repaired here.

**(b) The scan ran on an already heavily-tuned operating point.**
`wct-pr-perevt.jsonnet` turns on ~45 shower knobs — `shower_stem_backfill`,
`shower_cone_absorb_guard`, `shower_absorb_unreachable_main`,
`shower_detach_track_stem`, `shower_ghost_member_drop`, `shower_hadronic_tag`,
`shower_nv_bridge_track`, `shower_dedup_start_seg`, … — from ~97 `doc pr/`
citations in `NeutrinoPatternBase.h`. **The 28 failing events are the residual
of that entire campaign.** Every proposal below therefore has to say why the
knobs already on do not reach it.

**(c) Two of the four buckets already have named prior art; one has a built,
un-flipped lever.**

| §13 item | prior art | state |
|---|---|---|
| 1 · split showers (`merge`, 20) | **pr/91 §7 P2 + P3** | designed, explicitly **not implemented**; P2 blocked on a measurement, P3 gated behind P2. pr/113 §7 still lists both open |
| 2 · stub (8) | none found | **uncovered class** |
| 3 · NCπ⁰ over-clustering (7) | pr/53 r6/r7 shipped; pr/55; pr/56; **pr/57 §14** | pr/57 §14's `relaxed_strict_img_2d_rescue` is *TOOLKIT SHIPPED, DEFAULT NOT SELECTED*, fitted against 899 owner labels over 230 events |
| 4 · `seed-out` (4) | none found | **uncovered class** |

So this round's contribution to items 1 and 3 is to **supply the measurement
each was blocked on**, not to re-design them. Its original content is items 2
and 4. The running ledger of claimed-vs-open for this stage is **pr/113 §7**.

### 15.2 The pass sequence, as it actually runs

Entry `TaggerCheckNeutrino.cxx:2502` → `shower_clustering_with_nv`
(`NeutrinoShowerClustering.cxx:4375`):

| # | pass | line | note |
|---|---|---|---|
| 0 | `shower_absorb_unreachable_main` | 4419 | knob |
| 1 | `…_with_nv_in_main_cluster` | 4465 | BFS from main vertex; muon→electron re-type |
| 2 | `…_connecting_to_main_vertex` | 4474 | conn-1 showers at the main vertex |
| 3 | `…_with_nv_from_main_cluster` | 4481 | conn-2; direction cone |
| — | `sccc_bridge_body` replay | 4488 | pr/93 bridge edges land here |
| 4 | `…_with_nv_from_vertices` | 4534 | largest pass; conn-3; `nv_bridge_track` |
| 5 | `collect_charge_maps` → `calculate_shower_kinematics` | 4545 / 4551 | energy |
| 6 | `examine_merge_showers` | 4556 | **conn-1 × conn-2 only, angle < 10°** |
| 7 | `shower_clustering_in_other_clusters` | 4564 | three sub-passes |
| 8 | `stem_backfill` | 4578 | knob, pr/93 |
| 9 | `calculate_shower_kinematics` | 4588 | |
| 10 | `examine_showers` → `examine_shower_1` | 4593 / 3586 | the cut ladders |
| 11 | `merge_showers_sharing_start_segment` | 4605 | knob, pr/84 |
| 12 | `shower_detach_track_stem` | 4633 | knob, pr/93 |
| 13 | `shower_ghost_member_drop` | 4743 | knob, pr/99 r2 |
| 14 | `shower_hadronic_tag` | 4960 | knob, pr/99 r3 |
| 15 | `id_pi0_with_vertex` | 5189 | |
| 16 | `id_pi0_without_vertex` | 5199 | |

**The structural finding: every pass here grows, absorbs, re-types, or drops.
None partitions one shower into two.** `detach_track_stem` peels a *prefix*;
`drop_ghost_member` removes a *leaf*. Group 2 has no shower-level lever at all
today — which is why §15.5 begins with a question about *which level* it fails
at, rather than with a new pass.

### 15.3 Step 1 — the scoring harness, before any C++

The owner's instruction — *"for the algorithm improvement and validation, we
want to use the hand scan events to help"* — and CLAUDE.md §5's "propose the
checkable success criterion first" point at the same first deliverable.
**Nothing else starts until it exists.**

The bridge is already present on both sides: `PrDisplayDump::dump_showers()`
(`PrDisplayDump.h:148`) writes per-shower member lists into
`calib-evt<ID>-group*.json` keyed by the same display id `cluster_id*1000 +
seg_id` that `em_labels/<tag>/labels-evt<N>.json` records in
`em.marks_by_shower` / `em.marks_detail[shw].marked{}`. So for every marked
event we have **target** membership (the marks, with `owner` saying where each
segment came from) and **actual** membership (the dump).

`em_display/em115_score.py`, beside `em114_categorize.py`:

* per hand-scanned shower — completeness `|A∩T|/|T|`, purity `|A∩T|/|A|`, and
  **charge-weighted** variants, because charge is what the π⁰ mass consumes and
  segment counting alone under-reports a missing long member;
* per event — one scalar (charge-weighted F of the best-matching shower) and,
  for π⁰ events, `|m − 135|` of the best pairing;
* `--baseline` / `--compare`, so a knob change is a two-column delta table
  rather than a re-read of 97 files.

**Objective, two-sided, with the bias stated:** improve the **25 marked**
events, hold the **37 "good"** ones flat. The 25 were selected for failure and
are few; the 37 are the regression set. **This is not MC truth** — the toolkit
has no truth-matching machinery — it measures *agreement with the hand scan*,
and any result must be quoted in those words.

Also in step 1, cheaply: scan **`evt282909`**, the one event of 98 never opened
(§1), so the roster is complete before it becomes a benchmark.

### 15.4 Step 2 — group 1, under-clustering (18 + 3 both)

§2.1 already split this by `owner`: **168 IN marks take a segment from a real
neighbouring shower** (`merge`) vs **63 pick up a never-absorbed stub**
(`orphan`). Different failures, different repairs.

**(a) `merge` — 20 events, `evt168596` the reference.** Owner: pass 6
`examine_merge_showers` (`:2079`), narrow in three independent ways — it pairs
**only conn-1 with conn-2** (`:2118-2120`), the direction cut is a hard-coded
**10°** (`:2150`), and both directions are taken **from the main vertex**, the
wrong pivot for two fragments that meet away from it.

pr/91 §7 P2 — measure the conn-3 admission distance to the parent shower's
**charge** rather than its start segment — is exactly this class, and pr/91's
own verdict was *"blast radius: not census-boundable; needs a full A/B plus a
hand scan."* **We now have the hand scan, so the deliverable here is pr/91's
missing measurement, not a new design.** Run both distance definitions over the
20 `merge` events and the 37 good ones and report the distributions;
`marks_detail[shw].member_span {dist_min, dist_max, angle_min, angle_max}` is a
*measured* record of how far and how wide the intended members actually sit —
the calibration input pr/91 lacked.

*(pr/91 illustrates P2 with evt174752, 4.914 → 1.704 cm. That number is pr/91's,
from before this scan; §10 has since put evt174752 in the wrong-vertex bucket.
It is P2's provenance, not one of our 20.)*

Only if the measurement supports it: ship P2 under pr/91's own proposed name
`ex_shower1_conn3_dis_to_shower`, plus knobs on the hard-coded merge limits —
`shower_merge_angle_deg` (C++ default 10.0), `shower_merge_conn11` (allow
conn-1 × conn-1 under a proximity requirement), `shower_merge_local_pivot`
(directions from closest approach, not the main vertex). pr/91's P3 stays where
pr/91 put it: not attempted until P2 is measured.

**(b) `orphan` / `stub` — 8 events. No prior art.** Owner: `stem_backfill` and
`shower_cone_absorb_guard`, **both already ON** — so this is a *reach* problem
in machinery that exists, not a missing pass. First deliverable is a
distribution: the passes' effective reach against `member_span`. Only if those
separate does a reach knob (`shower_absorb_cone_deg` / `shower_absorb_max_dist`)
follow. This is also the prerequisite for π⁰ energy work (§15.6).

**(c) `seed-out` — 4 events, `evt47212`. No prior art.** Owner: pass 12
`shower_detach_track_stem`, **already ON**, so these are its residual. Item:
run the existing pr/93 probe (`WCT_SHOWER_ABSORB_DEBUG`,
`NeutrinoShowerClustering.cxx:80`) on the 4 and report *why it declines* before
touching anything.

### 15.5 Step 3 — group 2, over-clustering (7 + 3 both)

**First establish at which level each of the 7 fails. This is not optional.**
"Over-clustered" is ambiguous across two stages, and the prior art lives at the
*other* one:

* **image / graph level** — the two gammas sit in one *cluster* that should have
  been separated. pr/53's territory, and **pr/57 §14 already has a lever built
  and validated for it**, awaiting only an owner flip of `protect_graph_name`.
* **shower-assembly level** — clusters are fine, but `shower_clustering_*`
  walked both gammas into one `Shower`. Nothing in §15.2 can undo this.

So: for each of the 7, check whether the two gammas' segments already live in
distinct clusters. If they do, the event is a pr/57 §14 candidate and belongs in
*that* decision. Only the residual needs a new pass. Doing this first is what
prevents building new machinery for events an already-validated lever fixes.

**For the residual — a shower-level split.** The bookkeeping already exists
twice, written to one contract: `Shower::detach_track_prefix` (`PRShower.h:311`,
pr/93) and `Shower::drop_ghost_member` (`:336`, pr/99 r2). Between them they
spell out what any member-removal owes — rebuild the named point clouds from the
survivors (`kine_charge` reads the clouds), erase walked vertex marks,
invalidate caches, clear `flag_kinematics`, and leave the caller to re-run
`update_particle_type` / `calculate_kinematics` / `set_kine_charge` /
`update_shower_maps` — plus the trap: refuse if any survivor would be
**stranded** from the start segment under view-restricted connectivity.

1. `Shower::split_at(members_to_move)` — **forked by duplication** from
   `detach_track_prefix` (the production method stays byte-untouched), returning
   a second `Shower` under the same contract. Unit-testable with no event.
2. A new pass choosing where to cut, run after `examine_showers` and **before
   the π⁰ finders** so pairing sees two gammas. `evt314838` and `evt142421`
   already carry the intended partition in their marks — `evt142421`'s
   `split-by-proxy` marks (33 IN + 10 OUT, all 43 owned by shower 108104) are a
   hand-drawn answer key.

Knob `shower_split_pi0`, default OFF. This is the only new pass in the plan.

### 15.6 Step 4 — π⁰ reconstruction (groups 3 and 4)

**Clustering is upstream of mass, in two directions, and both must be respected
before any π⁰ number is quoted as a π⁰ improvement:** a charge-starved gamma
gives the wrong mass under a perfect pairing (§11.1 — three of the four worst
hand masses are `stub` events), and a shower holding *both* gammas gives the
wrong opening angle by construction (see (c) below).

**(a) The finding that reframes §11 — `kine_pio_mass` is not the identified π⁰.**
Each finder runs **two independent selections over the same shower pairs**:

| | BDT-feature fill → `T_kine kine_pio_*` | π⁰ identification → `pi0_showers`, `map_pio_id_*` |
|---|---|---|
| with-vertex | `:3777-3834` | `:3837-3916` |
| without-vertex | `:4260-4298` | `:4300-4370` |
| criterion | **highest summed `kine_charge`, no mass window at all** | **mass window** + greedy best-match |

`pio_kine.flag = 1` is assigned at `:3818`, *inside the fill*, before the
identification loop runs. So `kine_pio_mass` is the highest-energy pair whatever
its mass, and `kine_pio_flag` records **which fill ran**, not that a π⁰ was
identified — despite the `NeutrinoTaggerInfo.h:43` comment reading "0=not
found".

**§11 must be read in that light.** Its "reco π⁰ mass" column is the
energy-ranked pair, so the 22-vs-15 comparison there was never against the
identification. §11 stands as the record of what was believed; this is the
correction. **The first π⁰ item is therefore a measurement, not a change** —
re-extract both quantities on the 26 π⁰ events and redo the comparison. It may
show the identification already agrees with the hand scan far better than §11
suggests, which would move effort out of (b) entirely.

**(b) Pairing recall — the real cuts.** Both finders use
`m = √(4·E₁·E₂·sin²(θ/2))` (`:3771`, `:4199`) over `get_kine_charge()`.

* `id_pi0_with_vertex` **never pairs conn-1 × conn-1** (`:3766`) — two showers
  both directly attached to the same vertex are excluded, which is precisely the
  NCπ⁰-at-a-vertex topology of group 3. Highest-value lever here:
  `pi0_pair_conn11`.
* Its window is **(100, 160) MeV** (`:3854`), with a **6 MeV bonus** for
  both-conn-2 pairs (`tmp_penalty`, `:3850` — named penalty, applied as a
  bonus). Both hard-coded; exposing them is what lets the harness scan them.
* Its greedy loop **does** emit multiple π⁰ (`:3909-3915`).
  `id_pi0_without_vertex` emits **at most one**, window ±60 MeV (`:4300-4322`).
* `id_pi0_without_vertex` requires `conn_type == 3` (`:4125`), demands the
  back-projected midpoint sit within **25°** of both axes (`:4196`), and when
  **both** showers are ≤ 15 cm it `break`s the inner loop (`:4255`), abandoning
  every remaining partner for that shower. That `break` is prototype-faithful
  and was deliberately restored, so relaxing it is a knob
  (`pi0_pair_short_break`), never a silent edit.

**Flagged, not proposed:** on success `id_pi0_without_vertex` **moves the main
vertex** to the reconstructed π⁰ decay point (`:4337-4338`). That is π⁰ code
writing the ν vertex — worth knowing before any group-4 change, even though
vertex work is out of scope.

**(c) Direction — why an over-clustered shower cannot give a good mass.**
`shower_cal_dir_3vector` (`PRShowerFunctions.cxx:132-185`) is **not a PCA**: it
averages every trajectory-fit point within 15 cm of `p` and returns
`(centroid − p).norm()`. For a shower that swallowed both gammas the centroid is
a blend of two axes, so the opening angle is wrong by construction however good
the pairing. **§15.5 is the π⁰ fix for those 7 events**, not anything in §15.6.

**(d) Multi-π⁰ is not representable.** `Pi0KineFeatures`
(`NeutrinoPatternBase.h:149`) holds one candidate; `evt21073` has two and cannot
be expressed, even though the greedy loop finds both and `map_pio_id_showers` /
`map_pio_id_mass` carry them. Only the output struct and the 12 `T_kine`
branches (`root/src/UbooneTaggerOutputVisitor.cxx:1172-1183`) collapse them. The
fix is additive — new branches behind a knob, legacy branches byte-identical.

**(e) Vertex routing.** §8 found the `no_vertex_ncpi0` flag catches only 3 of 6;
the same ambiguity plausibly mis-routes events between the two finders.
Instrument which finder claims each of the 26 π⁰ events and compare with the
group-3/group-4 column — a measurement, no code change, and it says whether
recall is lost in (b)'s cuts or in the routing.

### 15.7 Order of work

Every measurement lands before the change that depends on it.

| # | work | kind |
|---|---|---|
| 1 | scoring harness (§15.3); scan `evt282909` | tooling |
| 2 | `kine_pio_mass` vs the identified π⁰ on the 26 (§15.6a) | measurement |
| 3 | cluster-level vs shower-level for the 7 (§15.5) | measurement |
| 4 | pr/91 P2's missing distance measurement on the 20 (§15.4a) | measurement |
| 5 | stub/orphan reach vs `member_span` (§15.4b) | measurement |
| 6 | which finder claims each π⁰ event (§15.6e) | measurement |
| 7 | π⁰ pairing knobs (§15.6b) | C++, default OFF |
| 8 | merge knobs / pr/91 P2 — only if 4 supports it | C++, default OFF |
| 9 | absorb-reach knobs — only if 5 supports it | C++, default OFF |
| 10 | `Shower::split_at` + split pass, for 3's residual only | C++, default OFF |
| 11 | multi-π⁰ output branches (§15.6d) | C++, default OFF |

Items 1–6 carry no reconstruction risk. Nothing in 7–11 starts before its
measurement.

### 15.8 How each C++ item ships

Note the wiring: `NeutrinoShowerClustering.cxx` contains **no
`get(config, …)` at all** — knobs are `PatternAlgorithms` members read there and
set in `TaggerCheckNeutrino::configure()`, reached from the
`tagger_check_neutrino` node in `cfg/pgrapher/common/clus.jsonnet` via its
`knobs={}` parameter. A new knob is therefore three edits: member + default in
`NeutrinoPatternBase.h`, the `get()` in `TaggerCheckNeutrino::configure()`, and
the jsonnet.

Then, per item: default OFF so the absent key is byte-identical; the
key-suppression idiom in `sbnd/wct-pr-perevt.jsonnet` **only**, commented with
the C++ default; a new pin in `clus/test/doctest_clus_knob_defaults.cxx`, which
exists precisely to catch a moved default; `./build/clus/wcdoctest-clus` green;
a byte-identical A/B gate with its label quoted; and a knob-on smoke run on the
named event. **Nothing here changes an existing default or a constant** — that
is a stop-and-ask.

### 15.9 One decision this section surfaces but does not take

pr/57 §14's `relaxed_strict_img_2d_rescue` is built and validated against 899
owner labels, and deliberately left unselected. Flipping `protect_graph_name` is
a production default change and therefore the owner's call. §15.5's measurement
will say how many of the 7 it plausibly reaches; that number is offered as
input to the decision, not as a recommendation — 7 events is not a basis for
moving a production default.

### 15.10 Noticed while reading, not acted on

* `clus/docs/patternrecognition/shower_clustering_review.md` — line numbers
  stale by ~2000 lines; B.1–B.5 all closed, only L.1 survives (now `:1400`).
* `clus/docs/shower_clustering.md` — worse than stale. Its description of
  `shower_clustering_with_nv_from_vertices` ("runs `find_proto_vertex()`, then
  calls `shower_clustering_with_nv_in_main_cluster` on each satellite cluster")
  describes neither what that function does nor what it calls. It also still
  says "~3,310 lines" and omits all eight post-`examine_showers` passes. Worth
  its own round.

**No code changed in this round — no C++, no jsonnet, no config. No A/B gate is
owed.**

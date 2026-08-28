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

# 62 — STM tagger baseline from the owner's hand scan, and the proton list

**Status.** The owner hand-scanned the 97-event :5012 display (doc 61 §5c–§5e) and
adjudicated **72 in-beam bundles** by comment. Those verdicts are **truth** here —
they supersede the doc-61 sub-agent scan wherever the two differ. This document is
the baseline for improving the STM tagger.

Two classes are excluded at the owner's instruction: **negative dQ/dx** (mostly
neutrons, not a tagger problem) and the **TGM labels** (ruled fine — doc 61 §5e).

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# the owner's adjudication, as saved by their own :5012 viewer instance
ls work-mcp1kall-d59k/nusel_labels/s61mis/nusel-labels-evt*.json      # 77 events
# the two products of this doc
column -ts$'\t' scan-d59k/stm-baseline.tsv | less -S                  # 72 bundles
column -ts$'\t' scan-d59k/proton-list.tsv | less -S                   # 13 protons
```

Bundles with **no** owner comment are ones where the tagger's label and the doc-61
scan already agreed, and the owner confirmed by silence; they are not in the
baseline.

## 1. The answer to the count

| | n |
|---|---|
| **ν misidentified as STM by the code** (tagger STM, owner: not a stopping muon) | **15** |
| **STM missed by the code** (tagger untagged, owner: it is an STM) | **1** |
| tagger STM and the owner agrees | 36 |
| tagger untagged and the owner agrees | 20 |
| **total adjudicated** | **72** |

So of the **51 STM tags the owner adjudicated, 15 are wrong — 29 %** — and the
tagger missed **1** STM among the 21 untagged bundles he looked at.

**Scope, so these are not over-read.** :5012 was built from the doc-61 scan's
*claimed* errors plus its pathology classes, not by random sampling. The owner
therefore adjudicated 51 of the code's 150 in-beam STM tags in the 393-event set;
**the other 99 are unaudited**, so 15 is a **floor** on the false-STM count, not a
measurement of the rate over the whole sample. The one claimed mistake he did not
comment on is `56463:12` (tagger STM, scan said nu) — unadjudicated.

## 2. The correction table — what to fix in the tagger

These 16 rows are the whole fix list. `class` and the owner's own words are in
`scan-d59k/stm-baseline.tsv`.

### 2a. Tagger says STM, owner says it is not (15) — false positives

| event:main | t (µs) | len (cm) | owner's reason | failure mode |
|---|---|---|---|---|
| 285366:13 | 1.102 | 95.1 | "very high dQ/dx at the end point, neutrino vertex" | **vertex mistaken for a Bragg peak** |
| 289832:10 | 0.629 | 214.2 | "good nu, vertex high dQ/dx" | same |
| 290294:12 | 1.752 | 28.2 | "vertex activity" | same |
| 409546:8 | 0.936 | 82.0 | "high vertex dQ/dx" | same |
| 278662:1 | 0.678 | 269.0 | "the high dQ/dx could be a vertex activity" | same |
| 278794:7 | 0.695 | 203.1 | "multiple shower like activities, negative dQ/dx is sign of gaps" | **several objects fitted as one** |
| 285443:7 | 1.105 | 178.9 | "clearly other tracks/showers" | same |
| 349241:15 | 2.003 | 137.1 | "other tracks" | same |
| 353223:15 | 1.072 | 67.6 | "two tracks" | same |
| 402330:1 | 1.472 | 56.3 | "multiple tracks, neutrino" | same |
| 72586:17 | 1.175 | 301.6 | "there is a small track near the bottom" | same |
| 392200:27 | 0.683 | 102.0 | "flat track" | **no Bragg at all — accepted anyway** |
| 48895:17 | 1.442 | 235.5 | "Not STM" (the original calibration event) | same |
| 321107:13 | 0.755 | 226.9 | "Not STM" | (unstated) |
| 321371:18 | 0.950 | 308.7 | "Not STM" | (unstated) |

Grouped, the tagger's false STMs are **5 vertex-as-Bragg**, **6 multi-object
clusters**, **2 flat tracks with no Bragg** (+2 unstated). The two big modes are
both *"is this one stopping particle at all?"* questions, not dQ/dx-shape
questions:

* **vertex-as-Bragg** — high dQ/dx at the *end* of the fitted track is accepted as
  a stopping point when it is really a neutrino interaction vertex. A Bragg peak
  and a vertex both raise dQ/dx at one end; what separates them is whether *other
  prongs leave that same point*.
* **multi-object** — the fit runs over a cluster holding several tracks/showers,
  so "one end enters, the other stops" is a statement about the merged blob. Same
  root cause as the owner's doc-61 §5e TGM ruling: establish one-objectness first.
* **flat-track acceptance** — `eval` PASSes with no Bragg rise at all (48895 is
  the reference; 392200 the second instance).

### 2b. Tagger missed an STM (1) — false negative

| event:main | t (µs) | len (cm) | owner |
|---|---|---|---|
| 62613:17 | 1.041 | 301.4 | "This looks like a STM missed. AI scan is correct." |

The tagger left it `nu-candidate`; it has an upstream-wall entry at z≈−0.2 and a
green-curve Bragg, and was rejected on a single 242 ke/cm end point. **One missed
STM in 21 adjudicated untagged bundles: efficiency is not the tagger's problem —
purity is.**

## 3. The proton list

`scan-d59k/proton-list.tsv`, 13 bundles the owner identified as protons.

| event:main | tagger | STM-tag OK? | owner |
|---|---|---|---|
| 59377:7 | STM | yes | "Proton candidate, IDed as STM is fine" |
| 61313:18 | STM | yes | "Proton candidate, treated as STM is fine." |
| 72828:7 | STM | yes | "Proton, IDed as STM is fine." |
| 168388:6 | STM | yes | "Proton, ided as STM is fine." |
| 174488:3 | STM | yes | "Proton, ided as STM is fine." |
| 289343:9 | STM | yes | "Proton candidate, can be viewed as STM." |
| 291345:12 | STM | yes | "Proton, treated as STM is fine." |
| 386838:16 | STM | yes | "Proton track, can treat as STM" |
| 389544:13 | STM | yes | "Proton track, can be treated as STM." |
| 389962:5 | STM | yes | "Proton, can be treated as STM" |
| 409084:12 | STM | yes | "Proton, can be viewed as STM." |
| **397920:8** | nu-candidate | **no** | "Not STM, clear proton at vertex." |
| **404684:9** | nu-candidate | **no** | "Proton candidate, Not STM" |

**The rule the owner is applying:** a *standalone* proton is not a neutrino
candidate, so the tagger removing it as STM is acceptable (11 of 13). A proton **at
a vertex** is the neutrino signature itself, so those two must stay in the ν pool —
and the tagger already leaves them untagged, correctly.

### 3a. This answers the open dQ/dx question from doc 61 §5d

Doc 61 flagged a class needing an owner ruling: `dqdx-normalisation`, 15 bundles
whose dQ/dx sits **1.5–2× above the muon table over the whole track with the
correct shape**, and the scan had to invent a discriminator ("does the far end
return to the 56 ke/cm MIP line?").

**12 of those 15 are on this proton list.** That class *is* the proton signature —
not a normalisation or dx artifact, which is what the scan suspected. The three
that are not protons: 281632:8 (owner: STM is fine), 285366:13 (a neutrino vertex —
in the false-STM table above), 63603:13 (a ν candidate with multiple tracks).

For the tagger this is directly actionable: a track sitting uniformly ~2× MIP with
a muon-shaped profile should be **identified as a proton**, and then the standalone
/ at-a-vertex distinction decides whether it may be removed.

### 3b. Confirmed from the charge — [55](55_dqdx-vs-rr-three-bundles.md) §11

The 13 tracks' fitted dQ/dx-vs-residual-range profiles were pulled out of
`work-mcp1kall-d59k` and put on the SBND expectation curves. **Twelve of the 13
land at k_muon = 1.69–1.93 and k_proton = 1.02–1.14** — a coherent population at
~1.9 × the muon expectation, agreeing with the doc-55 §7g proton curve (fitted
before this list existed) to **0.991 ± 4.3 %**. The identification here was made
by eye, so that agreement is a measurement rather than a definition, and §3a's
reading is confirmed from the charge side.

The one exception is **397920:8** — but for a geometric reason, not a charge one:
its fitted main runs 278.9 cm over 453 points, i.e. the fit spans the whole
multi-prong object the owner saw the proton *inside*, not the proton. The other
at-a-vertex proton, 404684:9, has a 78.8 cm proton-like main and behaves like the
rest.

This list also closes doc 55 §9 item 3 ("a proton population"), which is why that
doc's §11 exists.

## 4. How the automated doc-61 scan actually did (calibration, not truth)

Worth recording because it sets how much the *unaudited* 99 STM tags can be
trusted, and the answer is: not much.

| doc-61 claim | adjudicated | upheld | rejected | precision |
|---|---|---|---|---|
| "tagger misidentified this as STM" | 43 | 14 | 29 | **33 %** |
| "tagger missed this STM" | 18 | 1 | 17 | **6 %** |

Agreement over all 72 adjudicated bundles: **25 agree / 47 disagree**.

The systematic is one-directional and clear from the owner's wording. The scan
demanded a *textbook* muon Bragg and read anything else at the stopping end as
disqualifying; the owner accepts as "STM is fine":

* **a Michel electron at the end** — **10** of the rejections say exactly that,
  and the scan overturned the tag on every one of them: 62473:12, 172596:15,
  277958:13, 280092:6, 285795:11, 390028:5, 392184:18, 393542:22, 394906:16,
  411790:13. The scan read the Michel as "extra tracks at the stopping end" or as
  a dQ/dx tail that spoiled the Bragg. **10 of the 29 false alarms are this one
  mistake** — teaching it costs nothing and removes a third of them.
* **a proton** — 11 more (§3): "IDed as STM is fine".
* **imperfect but plausible fits** — "May be a STM, OK", "Can be a STM, decay in
  flight?", "treated as STM is probably fine".

In other words the owner is judging *"is tagging this STM acceptable?"*, while the
scan judged *"is this strictly a stopping muon?"*. The 17 wrong promotions come
from the same place in reverse: a clean Bragg plus a wall entry was read as an STM
the tagger missed, when the object was several tracks or a ν vertex.

**Consequence for doc 61:** its §5b headline numbers (43 overturned / 18 promoted)
do not survive contact with the owner's scan; the surviving truth is the 15 + 1 of
§2 here. One code error the scan *also* missed — 278794:7, where both the tagger
and the scan said STM — shows the scan was not merely over-strict.

## 5. Files

| path | what |
|---|---|
| `scan-d59k/stm-baseline.tsv` | **the baseline**: 72 owner-adjudicated bundles, 4 classes, `owner_verdict` is truth |
| `scan-d59k/proton-list.tsv` | the 13 protons + whether an STM tag is acceptable on each |
| `dqdx_rr_sample/proton_index.tsv` | §3b: the same 13 with their fitted dQ/dx diagnostics (`k_muon`, `k_proton`, shape residual, drift), and the one named exclusion |
| `dqdx_rr_sample/proton_points.tsv` | §3b: their 789 fitted dQ/dx points |
| `work-mcp1kall-d59k/nusel_labels/s61mis/` | the owner's raw comments (their record — read only) |
| `scan-d59k/stm-disagreements.tsv` | the doc-61 scan's *claims* that :5012 was built from (superseded by the baseline) |
| `docs/61_nusel-handscan-key.md` | the scan key, the sub-agent method, and §5e's one-objectness ruling |

# doc pr/141 §16 — scan brief: the μ-typed PID set (18 objects)

**Viewer**: `http://localhost:5022/em_display_viewer` — note the path, the old
tab may still point at `/split_viewer` (that 404s).
**Tag**: `pidmu-0906-owner` (fresh; nothing existing is written).
**Arm**: `work-pr140r2-off-*` — the shipped production configuration.

## The question, per object

> **Is this object an EM shower (electron or γ), or a track (μ/π/p)?**

Nothing else. Not "does it belong to a π⁰", not "should it be split".

## Why it matters

Each object is typed `pdg = 13` by the reconstruction, so its energy is priced
under the **track** hypothesis. If it is really EM, `kine_reco_Enu` is low on it
by the exact global factor

    (0.87 × 0.95) / (0.58 × 0.86) = 1.657

i.e. **0.657 × the stored energy is missing** from the neutrino energy. doc
pr/141 §9.5 found one such object by hand (122660's `54071`, an owner-called
electron). This set asks how common it is.

## The 18, in the order the viewer shows them (by stored energy, descending)

| # | sample | event | object | stored E (MeV) | E added if EM |
|---|---|---|---|---|---|
| 1 | mcp1k | 283713 | 17007 | 874.1 | 574 |
| 2 | mcp2k | 99838 | 14004 | 795.3 | 522 |
| 3 | mcp2k | 499577 | 13031 | 554.5 | 364 |
| 4 | mcp2k | 77328 | 16017 | 529.2 | 348 |
| 5 | mcp2k | 318769 | 65037 | 353.4 | 232 |
| 6 | mcp2k | 294174 | 16028 | 327.2 | 215 |
| 7 | mcp2k | 292524 | 9018 | 318.8 | 209 |
| 8 | mcp2k | 397401 | 16012 | 275.1 | 181 |
| 9 | mcp2k | 281165 | 16006 | 236.0 | 155 |
| 10 | mcp1k | 286681 | 72040 | 205.9 | 135 |
| 11 | mcp2k | 392901 | 127030 | 160.3 | 105 |
| 12 | ncpi0 | 259542 | 81053 | 138.6 | 91 |
| 13 | mcp2k | 396323 | 10009 | 130.4 | 86 |
| 14 | mcp1k | 282979 | 41018 | 122.4 | 80 |
| 15 | mcp2k | 176502 | 20005 | 85.1 | 56 |
| 16 | mcp1k | 348691 | 51080 | 79.8 | 52 |
| 17 | nuecc48 | 235435 | 2024 | 79.3 | 52 |
| 18 | mcp2k | 170098 | 70030 | 51.0 | 34 |

**Stop wherever you like** — the list is ranked by what is at stake, so every
object you do retires the largest remaining error.

## How to answer

Either save in the viewer, or just reply with a list — the same shape as the
μ-typed verdicts you gave in §9.5:

```
283713/17007 --> track
318769/65037 --> electron shower
294174/16028 --> overclustered muon
...
```

`gamma` / `electron` / `shower` all read as EM; `muon` / `track` / `proton` read
as track. Add "unclear" freely — an unclear is a usable answer here, a guess is
not.

## What is deliberately NOT on this sheet

There is a **pre-registered prediction** for each of the 18 (doc pr/141 §16.2,
committed in `scripts/pr141_pidset.py` before the set was served). It is not
printed here on purpose: printing the predictor's answer on the sheet you then
judge makes the agreement number circular. Four of the 18 are predicted-track
controls.

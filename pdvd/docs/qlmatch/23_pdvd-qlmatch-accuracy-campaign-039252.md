# 23 — PDVD Q/L matching accuracy campaign (run 039252)

Running doc for the correct-match-rate improvement campaign that follows the
doc-22 non-match explanation round. Owner-approved plan; one section per
phase, each phase ends with a commit (toolkit + wcp-porting-img) and push.

## Repro block

```sh
# Baseline = nm4b runner defaults (wcp 98950c0, toolkit knobs OFF).
# Scorecard (both owner metrics) on all 18 events of 039252:
cd pdvd
python ql_display/ql_agree_score.py --tag <TAG>        # metric 1: agreement vs frozen doc-19 truth
python ql_display/unmatched_census.py --tag <TAG>      # metric 2: missed long clusters
# Byte-identical-off gate (knobs OFF must reproduce nm4b):
python ../abtest/hash_archive.py work/039252_<idx>_<TAG>/calib-evt*.json \
       work/039252_<idx>_nm4b/calib-evt*.json          # idx 0/5/15 + mabc zips
```

## Owner metrics and baseline (fixed)

| metric | baseline (nm4b, 2026-07-17) |
|---|---|
| agreement with AI + owner hand scans (`ql_agree_score.py`, objective tiers) | **84.3%** (phantoms 137) |
| non-matched long tracks (missed) | **91** (66 wrong-flash, 25 unmatched — doc 22) |

Adoption rule per phase: flip runner defaults only if a metric improves and
neither worsens. All toolkit changes are default-OFF knobs, byte-identical
when off. Tags are fresh per phase (`ac1`, `ac2`, ...); nothing under an
existing tag is rewritten.

## Phase index

| phase | lever | status |
|---|---|---|
| 0 | bookkeeping: doc 22 artifacts + this skeleton committed | in progress |
| 1a | rescue blind-spot fix (4 PASSES_UNADOPTED clusters) | pending |
| 1b | clean-channel rescue ratio (saturation-inflated ratios, e.g. uid33) | pending |
| 2 | phantom-side precision gates (xtpc pin, wtrunc overpred, flash twins) | pending |
| 3 | amplitude-model residual fit on 751 agreed GT pairs (+ optional knob) | pending |
| 4 | joint-fit levers (cull keep-quality, cross-flash exclusivity) — contingent | pending |
| 5 | final validation + Bee sets for rescan | pending |

Background evidence: doc 20 (census + precull), doc 21 (relaxed tier sweep),
doc 22 (scan comparison; wrong-flash reframe; rescue blind spot), and the
re-ranking sims recorded in doc 22's follow-up (score-argmin breaks 43-50% of
the 751 agreed matches — per-bundle re-ranking is excluded as a lever).

---

## Phase 0 — bookkeeping

Committed the doc-22 deliverables (analysis only, no code):
`docs/qlmatch/22_pdvd-nonmatch-scan-comparison-039252.md`,
`ql_display/nonmatch_explain.py`, and the render context records
`ql_display/png-nonmatch-nm4b/evt*/context.jsonl`, plus this skeleton.
The PNGs themselves (18 renders, 4 events, ~25 MB) follow repo convention
(`*.png` gitignored) and stay untracked — regenerable from the nm4b dumps
with `ql_display/render_groups.py` per the doc-22 Repro block.

<!-- phase sections appended as the campaign proceeds -->

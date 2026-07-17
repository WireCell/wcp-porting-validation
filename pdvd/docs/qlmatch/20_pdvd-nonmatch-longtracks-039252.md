# 20 — Long UNMATCHED clusters (Bee "non-match" long tracks), PDVD run 039252

Status: IN PROGRESS (Phase 0–1 complete: baseline + census; rescue runs next).

Goal: reduce the number of LONG tracks (>= 25 cm or >= 100 pts) that QLMatch
leaves with **no matched flash** — the tracks the owner sees under the Bee
"non-match" button. Baseline (doc 19 operating point `tune_c2_cr`, current
runner defaults) misses **109 / 842** long objective scan-positives (12.9 %).

The decision metric is **correct** matches, not merely non-empty ones: a track
matched to the WRONG flash clears the non-match button but is physically wrong
(worse than a non-match downstream). So precision, not raw recall, governs the
adoption recommendation.

## Repro

```bash
cd pdvd
# 0. fresh current-defaults baseline with calib dumps (matching-only, ~20s/evt):
for i in $(seq 0 17); do ./scripts/stage_ql_tag.sh 39252 $i nm0; done
env PDVD_LIGHT_SUFFIX=_keep PDVD_MAX_JOBS=6 ./run_clus_evt.sh -calib 39252 all -s nm0
python ql_display/ql_agree_score.py --tag nm0            # -> work/ql_scores/nm0/scores.md
# 1. census of the long unmatched clusters + offline rescue what-if:
python ql_display/unmatched_census.py --tag nm0          # -> .../unmatched_census.md
```

Truth = the doc-19 frozen reference (gold hand scan
`work/ql_labels/wfresc/labels-evt298567.json` + AI scan
`ql_display/decisions-cathxa/decisions-evt*.jsonl`, objective tiers
gold/high/med). Join key (event, flash time ±0.5 µs, main-cluster uid).

## 1. Baseline `nm0`

`nm0` (current runner defaults, matching-only from `_keep`) reproduces the
adopted `tune_c2_cr` scorecard **exactly** — agree 733, phantom 137, 84.3 %,
missed 109 (12.9 %), miss-flashcut 0 — confirming no drift from the runner
changes that landed after tune_c2_cr (cathode-connect tip-touch `651a0aa`) on
this run, and giving a clean current-state baseline.

`miss-flashcut 0` means **every** missed positive has an admitted flash within
tol of its scan time: the true flash is never cut by flash admission. The
misses are matching failures, not admission failures.

## 2. Census — why the 109 are unmatched

`ql_display/unmatched_census.py` reads the calib dumps (which hold only
CONTAINED bundles, QLMatching.cxx:3457) and places each missed long positive
in one mechanism class:

| class | count | meaning |
|---|---|---|
| A anchored-elsewhere | 0 | rides a matched group anchor — not a real non-match |
| B no-bundle | 0 | containment-culled at every T0 — would need new C++ |
| C gate-fail (rescue-reachable) | 105 | a contained candidate bundle exists near the true time |
| D wrong-time-only | 4 | bundles exist but none near the true time — photon-model |

Two structural findings:

- **Containment is already solved.** Zero class-B: the robust-endpoint trim
  family (doc 15, ON in the runner) already rescues the drift-extent-inflation
  cases that stranded clus 34 / clus 97. No new containment C++ is needed here.
- **105 of 109 are rescue-reachable**: a contained, light-examined candidate
  bundle for that cluster exists near the true flash time. Of these, **68**
  have a candidate that already PASSES the current rescue gates
  (ks<.25, chi2/ndf<15, .3<pred/meas<3) but was **culled from the LASSO
  snapshot pool by `cull_inconsistent`** before the fit. That is exactly the
  gap `cluster_rescue_precull` closes (draw the rescue pool from the pre-cull
  `all_bundles` universe): PDHD hard-ON, **PDVD never threaded it**.

## 3. Why PDHD's precull does NOT port cleanly — wrong-flash risk

Naively enabling precull with the current (base) gates is a **trap** for PDVD.
The offline what-if replays the C++ rescue `accept()`/`score()`
(QLMatching.cxx:2893-2910) over the precull pool and classifies each adoption
against the frozen truth — crucially splitting out **wrong-flash** adoptions
(the cluster IS a known positive but is adopted at a DIFFERENT flash, dt≫tol):
these are provably wrong yet **invisible to the scorer** (they land in
`unknown`, so the phantom count stays flat and falsely reassuring).

| gate set (ks/c2ndf/ratio) | recovered | phantom | wrongflash | unlabeled | precision |
|---|---|---|---|---|---|
| base .25/15/.3-3 (current) | 22 | 1 | **24** | 102 | 0.47 |
| tight .18/4/.5-2 | 21 | 0 | 19 | 54 | 0.53 |
| **tight .15/3/.5-2** | **21** | 0 | **12** | 31 | 0.64 |
| tight .12/2/.6-1.7 | 16 | 0 | 7 | 12 | 0.70 |
| tight .10/2/.6-1.6 | 12 | 0 | 4 | 3 | 0.75 |

The many-flash PDVD regime (shared all-PD flash, ~190 flashes/event) means the
per-bundle rescue `score` (ks·√(c2ndf)+|log ratio|) cannot reliably tell a
track's true flash from a rival: the correct and wrong-flash candidates have
**overlapping** ks/chi2/ratio/score distributions (recovered ks 0.02–0.22 vs
wrong-flash 0.04–0.25; scores overlap 0.1–0.8). There is no clean global score
cut. The only lever that improves precision is **tightening** the gates, which
trades a little recall for markedly fewer wrong-flash and unlabeled adoptions.

The current shared rescue (base gates, snapshot pool) adopts only **15**
clusters across all 18 events — nearly inert — so tightening its gates costs
almost nothing, while precull enlarges the reachable pool.

## 4. Ceiling (honest)

This lever cannot empty the non-match button. Of the 109 misses, only
~12–21 are **cleanly** recoverable (the tight-gate points); the remaining
~84 are flash-ambiguous (light metrics don't distinguish the true flash) or
class-D photon-model. Driving recall higher re-introduces wrong-flash errors.
A larger, non-byte-identical lever — not culling the correct-time bundle in
`cull_inconsistent` before the LASSO can place it — is noted for the owner but
NOT built here.

## 5. Plan for the rescue runs (next)

- Thread `cluster_rescue_precull` into PDVD jsonnet + runner, default OFF,
  byte-identical when off (standard knob recipe).
- `nm1` = precull ON + **unchanged** base gates: validates the census tool
  against real output (expect the base-row pattern: ~+22 recovered / +1
  phantom / wf~24 in the full scorecard).
- `nm2*` = precull ON + tight gates along the frontier above.
- Score every tag vs the frozen truth; report the full agree/phantom/missed
  scorecard (not the incremental deltas) since tightening moves the baseline.

Because the rescue adds matches the frozen truth cannot judge, and because
the non-match / wrong-match trade is a physics call (escalation rule 7), the
**operating point is the owner's to pick** from the frontier, with a rescan of
the chosen tag before adoption — this section presents the frontier, it does
not bake a point.

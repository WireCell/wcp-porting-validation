# 20 — Long UNMATCHED clusters (Bee "non-match" long tracks), PDVD run 039252

Status: COMPLETE + **ADOPTED** (baseline + census + rescue frontier + additive
knob + owner adoption 2026-07-17).  The `nm3` operating point is now the PDVD
runner default (§6); the toolkit C++/jsonnet knobs still ship default OFF
(byte-identical).  §8 characterises the tracks that remain non-matched after nm3.

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

## 5. Rescue runs — full scorecards (real output, not the what-if)

Every tag is a full matching-only reprocess of the 18 events, scored vs the
frozen truth. `recov`/`regr` are computed against `nm0`'s missed set (clusters
`nm0` matched that a tag now misses = regressions). `adopt` = QLclusrescue log
adoptions.

| tag | rescue config | agree | phantom | missed | unknown | recov | regr | adopt |
|---|---|---|---|---|---|---|---|---|
| nm0 | baseline (snapshot pool, .25/15/.3-3) | 733 | 137 | 109 | 98 | – | – | 15 |
| nm1 | precull, base .25/15/.3-3 | 746 | 137 | 96 | 147 | 18 | 5 | 82 |
| nm2a | precull, tight .15/3/.5-2 | 746 | 137 | 96 | 118 | 18 | 5 | 50 |
| nm2b | precull, tight .12/2/.6-1.7 | 740 | 136 | 102 | 110 | 14 | 7 | 34 |
| **nm3** | **precull+additive, tight .15/3/.5-2** | **747** | **137** | **95** | **117** | **18** | **4** | 50 |

Reading the frontier:

- **Precull works and recovers ~18 real non-matches** (the cull_inconsistent
  victims the census predicted), phantom flat throughout.
- **Base gates (nm1) are imprecise**: 82 adoptions, unknown +49 — the
  many-flash score adopts many clusters at unlabeled/wrong flashes.
- **Tightening (nm2a) keeps the full recovery at far higher precision**:
  same missed 96, unknown 118 vs 147, adopt 50 vs 82. Tighter still (nm2b) is
  over-tight — it drops genuine recoveries (missed back to 102).
- **Census-tool faithfulness (advisor reconciliation)**: the what-if predicted
  +22 recovered for base-gate precull; the real run recovered **18** and
  **regressed 5**, net −13 missed. The gap is the incremental sim not modelling
  the full rescue re-run; direction and magnitude hold, and the relative
  ordering of gate sets (which drives the sweep) is faithful.

### The wrong-flash regressions and the additive fix

Pure precull replaces the rescue pool wholesale, so it re-decides clusters the
shipped snapshot rescue already matched. In PDVD's many-flash regime the
per-bundle score cannot always tell a track's true flash from a rival, so a few
correctly-rescued tracks get **switched to the wrong flash** — e.g. nm2a evt298777
uid67 (truth 49.9 µs) → 676 µs, and uid4000103 (truth −1799 µs) → −366 µs. These
are physically wrong yet invisible to the scorer (they land in `unknown`).

`cluster_rescue_precull_additive` (new default-OFF C++ knob) fixes this: the
snapshot pool is the PRIMARY (its decisions are kept exactly), and the pre-cull
pool is a per-cluster FALLBACK only for clusters the snapshot cannot rescue — so
precull can add, never re-switch. **nm3** = additive + tight is the best point:
missed **95** (−14 vs nm0, −13 %), phantom flat, and it eliminates the worst
switch (uid4000103's 1432 µs error). Its 4 residual regressions are 1 cosmetic
(1.3 µs, same flash-coincidence group), 2 losses where the tight gate drops a
rescue nm0 made only under the looser base gates (a recall/precision tradeoff),
and 1 irreducible wrong-flash (uid67: the wrong flash has ks 0.07, genuinely
better light than the true flash — no light gate can separate it).

### Ceiling reached

nm3 leaves **95** long positives missed. Of the original 109, ~14 were cleanly
recoverable; the rest are flash-ambiguous (the true flash is not the
light-best) or class-D photon-model. Emptying the non-match button further
would require an upstream, non-byte-identical change (not culling the
correct-time bundle in `cull_inconsistent` before the LASSO can place it) —
noted for the owner, not built here.

## 6. Recommendation and adoption (owner decision)

Recommended operating point: **precull + additive + tight gates** (the `nm3`
config), enabled in the runner by
```bash
PDVD_QL_CRESCUE_PRECULL=1 PDVD_QL_CRESCUE_PRECULL_ADD=1 \
  PDVD_QL_CRESCUE_KS=0.15 PDVD_QL_CRESCUE_C2N=3 \
  PDVD_QL_CRESCUE_RLO=0.5 PDVD_QL_CRESCUE_RHI=2.0
```
Effect on 039252: **13 fewer non-matched long tracks** (missed 109→95, −13 %),
phantom flat, at the cost of **+19 new long matches the frozen truth cannot
judge** (unknown 98→117) and 4 minor regressions (1 cosmetic, 2 gate-tradeoff,
1 irreducible wrong-flash).

This is a production behavior change (NOT byte-identical). Per escalation rules
1 & 7 the toolkit C++/jsonnet defaults stay OFF (byte-identical), and the owner
made the adoption call.

**ADOPTED 2026-07-17 (owner request).** The `nm3` operating point is now the
PDVD **runner default** in `run_clus_evt.sh` (the `PDVD_QL_CLUSTER_RESCUE`
block): `PDVD_QL_CRESCUE_PRECULL=1`, `PDVD_QL_CRESCUE_PRECULL_ADD=1`, gate
defaults `KS=0.15 / C2N=3 / RLO=0.5 / RHI=2.0`. Verified the flip reproduces
`nm3` exactly: a default run (no rescue env) with `PDVD_KEEP_CFG=1` on idx 0
emits the six rescue keys at the nm3 values in the compiled config, and its
`calib-evt298567.json` is **byte-identical** to the explicit-`nm3` dump. Revert
to the pre-nm3 baseline with `PDVD_QL_CRESCUE_PRECULL=0 PDVD_QL_CRESCUE_KS=0.25
PDVD_QL_CRESCUE_C2N=15 PDVD_QL_CRESCUE_RLO=0.3 PDVD_QL_CRESCUE_RHI=3.0`; the
config-only fallback (no additive C++ lib) is `PDVD_QL_CRESCUE_PRECULL_ADD=0`
(= nm2a). The `nm3` Bee zips (`work/039252_*_nm3/mabc-all-apa.zip`) remain
available for the rescan of the +19 unverified new matches.

**Config-only fallback (`nm2a`, no C++ needed):** `nm2a` is one cluster worse
(missed 96 vs 95) and keeps the two gross wrong-flash switches the additive knob
removes, but it needs ONLY the pre-existing `cluster_rescue_precull` threading
(`PDVD_QL_CRESCUE_PRECULL=1` + the tight gates) — no new C++ lib. If the owner
prefers the zero-cross-detector-risk path, `nm2a` is the clean choice.

**What the owner will see in Bee:** the `missed` metric counts only the
*scanned* long positives; the non-match button shows *all* non-matched
clusters. Net there is ≈ −13 long tracks visible as newly matched, but the
rescan will also see the 2 tight-gate losses become non-matches and 1 track
(uid67) hidden behind a wrong-flash match — so eyeball those, not just the wins.

**Byte-identity of the shared C++ (both affected detectors):** the additive
commit edits `rescue_unmatched_clusters`, which **PDHD** runs in production
(`cluster_rescue_precull` hard-ON = the pure-precull branch). Gated on BOTH:
PDVD idx 0/5/15 (precull-off and pure-precull) and **PDHD run 029107 evt
983/991** (precull-ON, actively rescuing) — new lib vs the pre-additive lib,
all calib dumps + every mabc zip content-hash identical. The refactor changes
output only when `cluster_rescue_precull_additive` is explicitly on.

## 7. Decision numbers — baseline vs recommended (all 18 events)

Two complementary views of `nm0` (baseline) vs `nm3` (recommended), computed
from the calib dumps (`ql_display/ql_agree_score.py` + a direct long-cluster
count). The Bee non-match button is the *second* table (all long clusters);
the scan-agreement table is the subset the hand scan judged.

**A. Scan agreement — long tracks judged against the frozen hand scan**

| metric | baseline nm0 | recommended nm3 | change |
|---|---|---|---|
| scan positives (long) | 842 | 842 | — |
| matched (agree) | 733 | 747 | **+14** |
| **missed** | **109** | **95** | **−14** |
| missed % | 12.9 % | 11.3 % | −1.7 pt |
| phantom (scan-rejected) | 137 | 137 | **0** |
| agree % (agree / judged) | 84.3 % | 84.5 % | +0.2 pt |
| unlabeled new matches | 98 | 117 | +19 |

**B. Bee non-match button — ALL long tracks, not just the scanned subset**

| quantity | baseline nm0 | recommended nm3 | change |
|---|---|---|---|
| total long tracks (≥25 cm or ≥100 pts) | 1233 | 1233 | — |
| **left non-matched (no flash)** | **193** | **157** | **−36** |
| non-match fraction | 15.7 % | 12.7 % | −3.0 pt |

**How to read this for the decision.**

- The non-match button drops by **36 long tracks** (193 → 157). That is the
  number the owner sees directly in Bee, and it is larger than the scan
  table's −14 because the scan judged only ~840 of the 1233 long clusters.
- Of those 36 newly-matched tracks, **~14 are confirmed correct** against the
  hand scan (agree +14) and **~19 are new matches the scan never judged**
  (unknown +19) — physically plausible but unverified, hence the rescan
  requirement before adoption.
- **Purity does not degrade**: phantom (scan-confirmed wrong matches) stays at
  137, and agree % is unchanged (84.3 → 84.5 %). The gain is recall, not a
  precision trade against the scanned set.
- The residual cost is 4 regressions inside the −36 (1 cosmetic, 2 tracks the
  tight gate drops, 1 wrong-flash) — small next to the 36 recovered, but the
  reason the rescan matters.

Net: **−36 non-matched long tracks in Bee at flat phantom**, of which ~14 are
already scan-verified and ~19 await the owner's rescan. If the rescan confirms
most of the 19, the effective non-match reduction stands; if many are wrong,
fall back to the more conservative gates or to `nm2a`.

## 8. What remains non-matched after nm3 — ambiguity, not absence

Re-running the census on the **nm3** dumps
(`ql_display/unmatched_census.py --tag nm3`,
`work/ql_scores/nm3/unmatched_census.md`) classifies the 95 long positives that
are still non-matched:

| class | count | meaning |
|---|---|---|
| A anchored-elsewhere | 0 | rides a matched anchor — not a real non-match |
| B no-bundle (containment) | 0 | robust-trim already handles this |
| **C rescue-reachable but ambiguous** | **91** | a contained candidate bundle exists AT the true flash time |
| D wrong-time-only (photon-model) | 4 | many bundles (16–71) but NONE within tol of the true time |

**So the answer is: these tracks are almost all "cannot find a *reliable*
match", not "no match exists at all."**

- **91 of 95 (class C) DO have a candidate at the right flash.** For each, a
  contained, light-examined bundle sits within tolerance of the track's true
  flash time — the match is physically present. What's missing is
  *distinguishability*: in PDVD's ~190-flash-per-event shared-flash regime the
  light metrics (ks, chi²/ndf, pred/meas ratio) of the correct candidate are
  not separable from those of rival flashes at other times. The nm3 what-if
  makes this quantitative — pulling more of these 91 from the precull pool:

  | gate set | recovered | wrong-flash | unlabeled | precision |
  |---|---|---|---|---|
  | loose base .25/15/.3-3 | 8 | **13** | 91 | 0.36 |
  | tight .15/3/.5-2 (nm3) | 3 | 3 | 19 | 0.50 |
  | tighter .12/2 | 2 | 0 | 5 | 1.00 |

  At every operating point the wrong-flash count is ≈ the real-recovery count.
  That is the signature of **too many plausible matches**: the true flash is not
  the light-best one (e.g. evt298777 uid67 — the wrong flash has ks 0.07,
  genuinely better light than the true 49.9 µs flash). No acceptance gate on the
  per-bundle light score can separate them; loosening buys a wrong flash for
  every right one, tightening just leaves them unmatched. nm3 is essentially at
  the per-bundle-score recall ceiling.

- **4 of 95 (class D) are the genuine "no good match" case.** Bundles exist
  (16, 49, 66, 71 of them for the four) but *none* lands within tolerance of the
  true flash time — the correct-time light prediction is systematically off
  (photon-model / geometry), so there is no candidate at the right place to
  accept. This is the small minority.

**Why the lever is exhausted here.** Both residual causes are physics/instrument
limits, not matcher bugs: (C) the light yield + flash pileup make the true flash
un-rankable by per-bundle light alone, and (D) the photon model mis-predicts a
handful of true-time patterns. Reducing them further needs an *upstream* change,
not a rescue-gate change:
  1. **Don't cull the correct-time bundle in `cull_inconsistent` before the
     LASSO** — let the joint LASSO (which sees all flashes together) arbitrate
     instead of the per-bundle rescue score. Non-byte-identical, not built here.
  2. **Cut flash multiplicity / improve the photon model** so the true flash is
     genuinely the light-best — this attacks class C at its source and is also
     the only lever for class D.

Both are larger, non-byte-identical efforts scoped for a future round; nm3 is
the ceiling of the rescue-gate lever.

## Repro (rescue runs)

```bash
cd pdvd
for tag_env in \
  "nm1 PDVD_QL_CRESCUE_PRECULL=1" \
  "nm2a PDVD_QL_CRESCUE_PRECULL=1 PDVD_QL_CRESCUE_KS=0.15 PDVD_QL_CRESCUE_C2N=3 PDVD_QL_CRESCUE_RLO=0.5 PDVD_QL_CRESCUE_RHI=2.0" \
  "nm3 PDVD_QL_CRESCUE_PRECULL=1 PDVD_QL_CRESCUE_PRECULL_ADD=1 PDVD_QL_CRESCUE_KS=0.15 PDVD_QL_CRESCUE_C2N=3 PDVD_QL_CRESCUE_RLO=0.5 PDVD_QL_CRESCUE_RHI=2.0"; do
  set -- $tag_env; tag=$1; shift
  for i in $(seq 0 17); do ./scripts/stage_ql_tag.sh 39252 $i $tag; done
  env PDVD_LIGHT_SUFFIX=_keep PDVD_MAX_JOBS=6 "$@" ./run_clus_evt.sh -calib 39252 all -s $tag
  python ql_display/ql_agree_score.py --tag $tag
done
```
nm3 needs the additive C++ knob (toolkit `cluster_rescue_precull_additive`,
built + installed). Byte-identical-off verified: precull-OFF and pure-precull
calib dumps + mabc hashes identical to the pre-rebuild lib on idx 0/5/15.

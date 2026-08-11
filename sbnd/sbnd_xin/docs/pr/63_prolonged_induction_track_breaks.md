# doc pr/63 — do S5/S6/S7 split prolonged (dashed) induction-plane tracks?

Diagnosis + examples only. No code change. No production flip.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# today's production (S6+rescue+S7 all on), dumps+census on, over the doc pr/62
# mover-adjacent shortlist (37 events: 24 nueCC48 + 12 NCpi0-19 + 1 PR-data)
PR_OC56_SCAN_DUMP=1 WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=16 \
    ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr63-cur48 data <24 events>
PR_OC56_SCAN_DUMP=1 WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=16 \
    ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr63-cur19 data <12 events>
PR_OC56_SCAN_DUMP=1 WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=16 \
    ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr63-cur50 data 59179

# blind scan of the 555 PR-data events NOT already covered by any prior S6
# hand-scan campaign (out of the full 1000-event mcp1k pool)
PR_OC56_SCAN_DUMP=1 WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=32 \
    ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr63-scan300 data <300 events>

# priority verification, today's production, 3 events found by mining the
# EXISTING (free, no-rerun) 395-event round-4 dump work-pr57r4-scan395
PR_OC56_SCAN_DUMP=1 WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=3 \
    ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr63-verify3 data 348471 348691 314507

# Bee before (S6+rescue+S7 off) vs after (today's production), 5 events
SBND_PROTECT_GRAPH=relaxed_strict_img PR_JOBS=8 \
    ./run_pr_chain_batch.sh work-ncpi0-cb0805  work-pr63-pre19 data 71372
SBND_PROTECT_GRAPH=relaxed_strict_img PR_JOBS=8 \
    ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr63-pre48 data 388
SBND_PROTECT_GRAPH=relaxed_strict_img PR_JOBS=8 \
    ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr63-preX data 314507 348471 59179
python3 scripts/bee/make_pr_bee.py -q work-ncpi0-cb0805 -q work-nuecc48-cb0805 -q work-mcp1k-cb0805 \
    -p work-pr63-pre19 -p work-pr63-pre48 -p work-pr63-preX \
    -o bee/pr63/pr63-before.zip 71372 388 348471 314507 59179
python3 scripts/bee/make_pr_bee.py -q work-ncpi0-cb0805 -q work-nuecc48-cb0805 -q work-mcp1k-cb0805 \
    -p work-pr63-cur19 -p work-pr63-cur48 -p work-pr63-verify3 -p work-pr63-cur50 \
    -o bee/pr63/pr63-after.zip 71372 388 348471 314507 59179
./upload-to-bee.sh bee/pr63/pr63-before.zip
./upload-to-bee.sh bee/pr63/pr63-after.zip
```

## The question

Owner: a rare class of long tracks has signal on an induction plane (U or V)
stretched in *time* rather than broken in space — appearing as a long **dashed
line** on that plane's wire-vs-tick view because the track runs close to that
plane's wire direction (or drifts near-isochronously). Two asks: (1) would
S5/S6/S7 separate such a track, algorithmically; (2) find real examples in the
owner's own samples and show a Bee before/after.

## Algorithm side

`ClusteringProtectBundle` filters candidate edges between cluster components
through layered kill tests in `clus/src/connect_graph_relaxed_strict.cxx`.
Two senses of "prolonged" exist in this codebase and only one has protection:

| sense | mechanism | detected by | protected by |
|---|---|---|---|
| **topology** — path runs near-parallel to a plane's wires | geometric | `angle1`/`angle2 < 12.5°` -> `excuse_u`/`excuse_v` | S6 **and** S7, identical formula |
| **signal** (the owner's dashed line) — long low-amplitude same-wire pulse train drops hits | physical, hit-finding | `slope_*` (time/wire) + `coverage` (fraction of wires with any signal) | **S6's rescue only, and only `edge_dis <= 5.0 cm`** |

```cpp
// clus/src/connect_graph_relaxed_strict.cxx:481 (two_d_rescue_ok body)
if (in.dis_cm > 5.0) return false;
// :1135 (call site)
if (two_d_rescue && killed && edge_dis <= 5.0 * units::cm) { ... }
```

Distance-band summary:

| layer | band | prolonged-**signal** protection |
|---|---|---|
| S5 `relaxed_img_bad` | `[0,15)` cm | none — no per-plane gap notion at all |
| S6 `two_d_connectivity_bad` | `[1,30)` cm, kills on 1 unexcused gapped plane | rescue reaches only `<= 5cm` |
| S7 `long_corridor_bad` | `[30,inf)` cm | none, by design (`connect_graphs.h:105-109`: "never rescues one they killed") |

So the *nominal* ceiling is 5cm. **The empirical finding below is that this
ceiling rarely binds** — read the mechanism, not the number, as the answer.

## The mechanism that actually decides it

S6 evaluates **up to three independent candidates** per component pair —
`closest` (nearest 3D points between the two components), `dir1`, `dir2`
(points chosen along each component's own PCA direction) — each an
independent 3D distance `edge_dis`, each independently gap-tested and
independently rescue-eligible. **A component pair is connected if *any one*
candidate survives.** A track broken by a local induction-signal dropout is
still, physically, two pieces separated by a small 3D gap at the break point —
so the `closest` candidate is very often a few cm even when the two pieces
are 50-350cm long overall, and that candidate alone is what the rescue needs
to reach.

Verified directly against today's production dumps (`work-pr63-cur*`,
`work-pr63-verify3`) for four cases carrying a genuine unexcused gapped
induction plane, high time/wire slope, near-full wire coverage (the rescue's
own "prolonged" tiers):

```
evt   pair    candidate(s)                                      final today
59179  0-1    closest 1.13cm KEPT(rescued) | dir2 4.96cm KEPT(rescued) | dir1 10.46cm KILLED  -> TOGETHER
  388  2-3    closest 1.05cm KEPT(rescued) | dir2 4.85cm KEPT(rescued) | dir1  5.06cm KILLED   -> TOGETHER
348471 0-2    closest 1.66cm KEPT(rescued) | dir2 9.62cm KILLED                                 -> TOGETHER
71372 10-21   closest 1.73cm KEPT(rescued) | dir2 10.68cm KILLED                                -> TOGETHER
```

In every one of these, a *different* candidate for the *same* physical pair
exceeds 5cm and is killed un-rescued (proving the ceiling is real and live) —
but the pair stays connected anyway because `closest` was in reach. This
generalizes what doc pr/57's "closest-pair" MST design already relies on
elsewhere: the closest-pair distance, not the track length or the dir1/dir2
probe distance, is what usually governs.

**Where it does NOT hold — the same event's other pairs.** evt 71372 (the
owner's own canonical prolonged case, doc pr/56 §8) has, in the same cluster,
three pairs where **the only candidate that exists** exceeds 5cm and nothing
rescues it:

```
evt71372 10-13  dir1 only,    dis=5.07cm  gapUVW=010(V) unexcused  -> SPLIT (misses ceiling by 0.07cm)
evt71372  8-14  closest only, dis=15.15cm gapUVW=100(U) unexcused  -> SPLIT
evt71372  4-15  closest only, dis=13.97cm gapUVW=100(U) unexcused  -> SPLIT
```
`final` partition today: components {1,2,...,11,21} merge into one group,
but {10-13, 8-14, 4-15} above are NOT part of that merge — 12/22 components
of this event's cluster stay outside the main body, several by exactly this
mechanism.

A fifth case, evt 314507 3-5, has only one candidate (`closest`, 6.68cm,
**all three planes** gapped, not induction-only) and stays split today —
presented separately below since it is not a clean single-plane exhibit.

## Why the search stayed inside 37 events, then correctly stopped at 300

Owner: expand beyond the pr/62 117-event sample to the full 1000-event
PR-data pool, track cases only (not EM showers), 32 CPUs authorized.

**Track vs. shower filter.** Point density alone does not separate them well
(a genuine 141cm track sits at ~9-11 pts/cm, overlapping shower-junction
density). PCA linearity — the fraction of a component's point-cloud variance
along its principal axis — does, calibrated against known cases:

```
evt59179 comp0/comp1 (confirmed clean track):        linearity 0.79 / 0.90
evt122660,54095,142421 (visually confirmed shower/junction blobs): 0.63 - 0.73
```
Threshold `>=0.75` on both sides of a candidate pair. Applied to every
prolonged-signature candidate in the 37-event shortlist (`work-pr63-cur*`)
plus the free, already-dumped 395-event round-4 arm
(`work-pr57r4-scan395` — confirmed a genuine subset of the 1000-event
`work-mcp1k-cb0805` pool): **7 genuinely track-shaped candidates found in
total**, all reported above or below. Small doubly-filtered sample by
construction — report the mechanism, not this count, as the finding.

**300-event blind expansion, then stop.** Of the 1000-event PR-data pool,
445 were already S6-hand-scan-dumped by prior campaigns (`work-pr57r4-scan395`
+ `work-pr57r4-scan50`) — those events were pre-selected for having rich
reconstructed activity. The remaining 555 were never selected for anything.
32-CPU dump+census run on 300 of them: **14/300 produced a non-empty S6/S7
dump; 7 total candidate edges; 1 prolonged-signature candidate (any
morphology, not just tracks) in the whole batch.** Most of the leftover 555
events have essentially nothing for S6/S7 to evaluate — thin or
non-neutrino-selected events. Scanning further into this pool is not
expected to be productive; stopped here rather than continuing to scan
blindly.

## Exhibits (Bee before/after, 5 events)

Before = `SBND_PROTECT_GRAPH=relaxed_strict_img` (S1-S3+S5 only, S6/rescue/S7
all off — every one of these tracks split). After = today's SBND production
default (S6+rescue+S7 all on).

- **before**: https://www.phy.bnl.gov/twister/bee/set/52eb9c0f-ca49-4970-99df-67baea1dad4b/event/list/
- **after**: https://www.phy.bnl.gov/twister/bee/set/1bef3aa8-3e9f-4560-870a-3c8f4d8a617c/event/list/
- index: bee 0=evt71372, 1=evt388, 2=evt348471, 3=evt314507, 4=evt59179

Both S6 and S7 are off in "before", not just one — several of these events
also carry pr/62 S7 movers elsewhere in the same cluster, so the diff is not
isolated to the exhibited pair.

**evt 348471, comp0(1418pt)-comp2(472pt), dis=1.66cm — RESCUED today.**
Auto-scan label: `bad [R2 long-track break, no W gap]` (i.e. a real track
wrongly split by the naive test — the label this whole investigation is
about). U/W panels show clean continuous 2D connectivity across the break;
V shows the textbook dashed gap — a blank band with no fired cells at all
between a tight blue clump and a tight red clump, both otherwise dense and
linear (Lmax=55.4cm, both sides `linearity` 0.77/0.93). Kept together today
because its `closest` candidate is 1.66cm even though `dir2` for the same
pair is 9.62cm and stays killed.

**evt 71372, comp10(5015pt)-comp13(544pt), dis=5.07cm — SPLIT today.** Same
signature (V gap, U/W continuous, auto-label `bad [R2 long-track break, no W
gap]`, Lmax=59.8cm) but the *only* candidate ever generated for this pair is
`dir1` at 5.07cm — 0.07cm over the rescue ceiling, no `closest`/`dir2`
alternative exists to fall back on. This is the cleanest evidence the 5cm
ceiling is a real, live limit and not merely nominal.

**evt 314507, comp3(539pt)-comp5(77pt), dis=6.68cm — SPLIT today.** All
three planes gapped (`gapUVW=111`), auto-label `bad [R2w thin collinear pair
across a W gap]`. Two nearly-straight collinear segments meeting near the
event vertex, small angle kink; visually one continuous thin track. Only one
candidate exists (`closest`, 6.68cm), no rescue possible. Presented
separately from the induction-only cases above since W is also gapped here.

**evt 59179 / evt 388** — reproduce the same rescued pattern as 348471
(`closest`/`dir2` in reach, one longer candidate for the same pair killed
un-rescued); evt 59179 comp0-comp1 is a single clean 141cm straight track
with the V-plane gap visible over its full length in all three probe
distances (1.13/4.96/10.46cm), all three showing the identical `gap=[F,T,F]`
signature.

## S7 census cross-check (from the pr/62 campaign, no new run)

`OC62CENSUS-S7` lines already logged in `work-pr62-cen48`/`cen19` (189 S7-
evaluated candidates, all `>=30cm`): **179/184 kills carry no topology
excusal at all** — the excuse mechanism essentially never fires at this
range. S7 has no signal-prolonged rescue of any kind. No confirmed example
of this actually biting a real long track was found this round (S7-band
candidates were not among the 37-event shortlist or the 300-event scan's
hits) — this is reported as an **open, unmeasured structural gap**, not a
demonstrated risk.

## Incidental correction to doc pr/62

pr/62 states both S7 operating points (`min_gapped_planes=1` vs `=2`) gave
the identical 117-event mover set *because* "whenever S7 fires at all in this
sample, it fires with >=2 non-excused gapped planes, not just exactly 1".
That mechanism claim is **false**, found while building the S7 census table
above: of the 184 S7 kills, **49 fire with exactly 1 voting plane**. The
event-level mover sets are still identical (verified this round, not done in
pr/62 which only compared each on-arm against off):

```
work-pr62-on48  vs work-pr62-on2-48:  ARCHIVE 1/48 events differ, nusel 0/48
work-pr62-on19  vs work-pr62-on2-19:  ARCHIVE 0/19 events differ, nusel 0/19
```
So the flip is not affected (nusel unchanged both ways), but pr/62's stated
*reason* for the identical mover sets needs correcting — see that doc for the
edit.

## What a fix would need (not built this round)

- The 5cm ceiling is a *reach* limit on `two_d_rescue_ok`, not a fitted
  physics boundary — its own tiers (slope/ext_med/coverage) carry no
  distance term. Raising it is mechanically direct.
- But it needs owner labels beyond 5cm to refit against: this round's
  evidence is 4 confirmed-rescued + 3 confirmed-still-split cases, far short
  of S6's 899-label campaign. A rescue-style relaxation round (pr/57 round
  6's path) is the natural follow-on if the owner wants this reach extended,
  scoped to real 5-30cm labels.
- S7 (>=30cm) has no rescue mechanism to extend — it would need one built
  from scratch, and no confirmed example motivates it yet.

## Fix — none shipped

No code change. No production flip. `git status cfg/ clus/` shows nothing
from this round beyond the doc pr/62 correction noted there.

## Cross-links

- [[project_pr62_long_edge_corridor]] — S7, the >=30cm sibling this doc's S7
  census cross-check reuses; the mechanism-claim correction lives there
- doc pr/56 §8 — the original prolonged-signal report (evt 71372) and the
  rescue's tier constants this doc verifies the reach of, not the tiers
  themselves
- doc pr/57 §14 — the 899-label hand-scan campaign; `348471`/`348691` here
  are drawn from its `claude-scan223` machine-labelled (not owner-corrected)
  set, cross-checked against the raw dump directly, not taken on the label
  alone

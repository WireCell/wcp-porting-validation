# doc pr/64 — evt 314507 is a collection-plane break; where are the induction analogues?

Diagnosis + examples only. No code change. No production flip.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# Q1/Q2: three-flavor attribution on the two headline events
E="314507 170814"
SBND_PROTECT_GRAPH=relaxed_strict_img          PR_OC56_SCAN_DUMP=1 WCT_RELAXED_EDGE_CENSUS=1 \
  PR_JOBS=4 ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr64-att-before data $E
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue PR_OC56_SCAN_DUMP=1 WCT_RELAXED_EDGE_CENSUS=1 \
  PR_JOBS=4 ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr64-att-mid data $E
PR_OC56_SCAN_DUMP=1 WCT_RELAXED_EDGE_CENSUS=1 \
  PR_JOBS=4 ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr64-att-after data $E

# full 1000-event PR-data pool at today's production, dumps + S7 census
PR_OC56_SCAN_DUMP=1 WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=32 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr64-scan1k data      # all 1000 ql_evt

# induction-analogue exhibits, before vs after
SBND_PROTECT_GRAPH=relaxed_strict_img PR_JOBS=4 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr64-ind-before data 408534 286681
# "after" reuses work-pr64-scan1k (bare production, same arm as the full scan)

# stratified mining (new script, imports oc56_autoscan/oc56_render_pair -- does not fork them)
python3 scripts/analysis/pr64/mine_pr64.py work-pr64-scan1k --ratio-min 6.0

# Bee
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-cb0805 -p work-pr64-att-before  -o bee/pr64/pr64-attr-before.zip 314507 170814
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-cb0805 -p work-pr64-att-mid    -o bee/pr64/pr64-attr-mid.zip    314507
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-cb0805 -p work-pr64-att-after  -o bee/pr64/pr64-attr-after.zip  314507 170814
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-cb0805 -p work-pr64-ind-before -o bee/pr64/pr64-ind-before.zip  408534 286681
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-cb0805 -p work-pr64-scan1k     -o bee/pr64/pr64-ind-after.zip   408534 286681
./upload-to-bee.sh bee/pr64/<name>.zip     # x5

# S7-top set: every event in the 1000-sample with >=1 killed S7 (long_gap_kill) edge
# at today's production, ranked by OC62CENSUS-S7 killed-edge count
grep -l "OC62CENSUS-S7" work-pr64-scan1k/pr_evt*/wct_pr_evt*.log   # 21/1000 events have any S7 candidate, 17 have a kill
E7="288859 314507 169356 286681 276836 168614 174224 288287 292027 315497 319039 349461 393042 394642 395654 400504 408534"
SBND_PROTECT_GRAPH=relaxed_strict_img PR_JOBS=8 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr64-s7top-before data $E7   # evt287007 dropped: no PR candidate selected in either flavor
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-cb0805 -p work-pr64-s7top-before -o bee/pr64/pr64-s7top-before.zip $E7
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-cb0805 -p work-pr64-scan1k       -o bee/pr64/pr64-s7top-after.zip  $E7
./upload-to-bee.sh bee/pr64/pr64-s7top-before.zip
./upload-to-bee.sh bee/pr64/pr64-s7top-after.zip

# worked-example panels (reuses oc56_render_pair.render(), not forked)
python3 -c "
import sys; sys.path.insert(0, 'scripts/analysis/pr57')
import oc56_autoscan as A, oc56_render_pair as R
prm = A.parse_params('')
for arm, evt, jk, kw in [('work-pr64-scan1k','174224',(1,2),{}), ('work-pr64-scan1k','174224',(2,3),{}),
                          ('work-pr64-scan1k','276836',(1,3),{'want_shown_only': False})]:
    for evt2, path in A.select_events(arm, 0, ''):
        if evt2 != evt: continue
        for pk, p in sorted(A.pair_table(arm, evt, path, **kw).items()):
            if (p['j'], p['k']) == jk:
                print(R.render(arm, evt, path, p, prm, 'docs/pr/pr64-panels'))
"

# Round 3: is a gated W-plane exception possible? Investigation + prototype only.
python3 scripts/analysis/pr64/wgate_sweep.py owner-labels     # R2w/R2d vs raw 899-label truth, no graph replay
python3 scripts/analysis/pr64/wgate_sweep.py blast-radius     # fresh work-pr64-scan1k population, no cache
python3 scripts/analysis/pr64/wgate_sweep.py sweep            # local-vs-global axis; STALE graph cache, see Caveat 2
```

## The questions

Owner, following up on one of doc pr/63's five exhibit events (evt 18255-**314507**,
where "before" shows one track and "after" today's production shows it broken):

1. Is the signal inefficiency on the **collection** plane, not induction?
2. Is **S7** responsible for the break?
3. Then: find the **induction-plane** analogue — long tracks with prolonged (dashed)
   signal in U or V — searching more broadly across the 1000-event PR-data pool.

## Q1 — yes, it is the collection plane, verified three independent ways

Event 314507 has 6 components in one cluster; 7 S6 candidate edges, **all killed**
today; `connectivity.final = [0,1,2,3,4,5]` — every component separate.

**(a) The seed clouds are prolonged specifically on W.** Per-plane tick-span /
wire-span of the exact cells fed to the corridor search:

| pair | dis cm | U ratio | V ratio | **W ratio** | gap UVW | excuse UV |
|---|---|---|---|---|---|---|
| 2-4 | 0.98 | 1.3 / 2.9 | 1.6 / 2.5 | **4.0 / 20.0** | `..T` | `T.` |
| 2-5 | 24.59 | 1.3 / 2.9 | 1.6 / 2.9 | **4.0 / 100.0** | `..T` | `..` |
| 4-5 | 26.78 | 3.1 / 2.9 | 3.1 / 2.9 | **148.0 / 100.0** | `..T` | `..` |
| 3-5 | 6.68 | 3.4 / 3.0 | 3.6 / 3.0 | **18.0 / 64.0** | `TTT` | `..` |

Confirmed against SBND's own geometry, not assumed: the W plane's wires are at
**0deg from vertical in both TPCs** (`(y,z)` wire direction `(1,0)`; U/V are at
`+-60deg`, mirrored between TPC0/TPC1) — `clus/docs/clustering/parallel-prolong-
analysis.md:24-27` (D1), independently rederived from
`wire-cell-data/sbnd-wires-larsoft-v1.json.bz2`. The W wire index stays pinned at
**456-458 across ~464 ticks** on these edges while U/V sweep ~50 wires over the same
span — this is exactly the isochronous/prolonged geometric signature, and it sits on
the collection plane.

**(b) The W dropout is a real hit-finding gap, and U/V are solid in the identical tick
window** (cells between the two seed clumps, same time range):

| pair | between-clump window | **W fired** | U fired | V fired |
|---|---|---|---|---|
| 4-5 | 216 ticks / 54 slices | **9 cells, 7/54 slices** | 377 cells, 53/54 | 273 cells, 53/54 |
| 2-5 | 196 ticks / 49 slices | **4 cells, 4/49 slices** | 344 cells, 48/48 | 247 cells, 48/48 |
| 2-4 | 12 ticks / 3 slices | **0 cells** | 15 cells, 2/2 | 11 cells, 2/2 |

A dashed W line against a fully continuous U and V line — the owner's signature,
inverted onto the collection plane.

**(c) It is not a search artifact.** `budget=[false,false,false]` on all 7 edges, so
S6's fail-closed circuit breaker (`connect_graph_relaxed_strict.cxx:209-212`) never
fired here; these are real corridor-search gaps, not budget exhaustion.

**Why this kills, structurally: W has no excusal channel at all.**
```cpp
// connect_graph_relaxed_strict.cxx:471   (S6 kill predicate)
return (gap_u && !excuse_u) || (gap_v && !excuse_v) || gap_w;   // gap_w is bare
// :549                                    (S7 kill predicate)
const bool excused[3] = {in.excuse_u, in.excuse_v, false};  // W never excused
```
The W analogue of the excuse angle, `angle1p`, **is computed**
(`:1651-1655`, against `af_W_dir`) and is consumed by S1-S3
(`:1678,1696,1716,1863,1996`) — but it is never passed to S6 or S7 (call sites
`:1752-1754`, `:1767-1769` pass only `angle1, angle2`). The stated rationale
(`Graphs.h:265-271`) is that prolongation is "an induction-plane-only physical
effect"; evt 314507 is a direct counterexample. The rescue reaches W through only two
narrow gates (`:500-507`: a dead-W-wire band, or tiny-closure+collinear+substantial),
both capped at `<=5cm` same as everywhere else.

## Q2 — both layers fire, cleanly separable and now verified by direct comparison

`OC62CENSUS-S7` on 314507 records 6 more candidates, all killed, unexcused,
39.9-49.2cm: pairs **1-2, 1-4, 2-3, 3-4** — every pair *inside* the {1,2,3,4}
sub-group. Since S6 (`<30cm`) and S7 (`>=30cm`) are provably mutually exclusive at the
30cm seam (`connect_graph_relaxed_strict.cxx:854` / `:1393`), any difference below
30cm is 100% S6 and any difference at/above 30cm is 100% S7.

**Three-flavor `connectivity.final` on 314507, run this round (prediction matched
exactly):**

| flavor | S6 | S7 | `final` | groups |
|---|---|---|---|---|
| before (`relaxed_strict_img`) | off | off | `[0,0,0,0,0,0]` | 1 (whole cluster) |
| mid (`relaxed_strict_img_2d_rescue`) | **on** | off | `[0,1,1,1,1,2]` | 3: `{0}`, `{1,2,3,4}`, `{5}` |
| after (today's production) | on | **on** | `[0,1,2,3,4,5]` | 6: everything separate |

So: **S6 already isolates comp0 (14pt) and comp5 (77pt)** via its W-only-gap kills
(`0-3`, `1-5`, `2-5`, `3-5`, `4-5` at 0.98-26.78cm — the population from Q1). The
`1-2/1-4/2-3/3-4` edges connecting the backbone {1,2,3,4} (comp3=539pt/38.4cm,
comp4=125pt/20.1cm, comp1/comp2=10/20pt) are all `>=30cm`, invisible to S6, and
survive S1-S3 alone in the "mid" flavor. **S7 alone breaks that backbone down to
singletons.** Answer: **both, at two different pairs of components in the same
cluster** — not a single mechanism doing all the damage.

Cross-checked: `work-pr64-att-after`'s S6 edge list on 314507 is byte-identical to
`work-pr63-verify3`'s (same 7 edges, same distances/gaps/excuses/killed flags), and
`mabc-pr.zip` + `pctree-pr-evt314507.tar.gz` hash member-identical
(`abtest/hash_archive.py`) between the two arms — the dump/census machinery is
log-only and "after" really is bare production.

## Part 3 — the induction search inverts the naive expectation

Mining the full **1000-event PR-data pool at today's production** (`work-pr64-scan1k`,
310/1000 non-empty S6/S7 dumps, 233 candidate rows with usable seed clouds — most of
the other 690 events have thin/non-selected clusters with nothing for S6/S7 to
evaluate, consistent with doc pr/63's earlier 300-event sample), stratified by
"prolonged on plane p (seed-cell tick/wire ratio >= 6) AND gapped on that same plane",
split by whether the geometric excuse fires:

| plane | excusal | n | killed | kill rate |
|---|---|---|---|---|
| U | excused | 74 | 5 | **7%** |
| U | NOT excused | 29 | 24 | **83%** |
| V | excused | 51 | 14 | **27%** |
| V | NOT excused | 49 | 35 | **71%** |
| **W** | no excuse channel exists | 30 | 28 | **93%** |

**The induction plane is the protected case; collection is the unprotected one.**
U/V have an excuse channel (7-tier rescue loop, `:511-536`) that drops the kill rate
from 71-93% to 7-27% — same order of magnitude as W's flat 93% with no protection
mechanism at all. So the population the owner is actually looking for — a real long
track, dashed on U or V, still split today — lives specifically in the **unexcused**
34-cm-region 5-30cm slice of that table, which is exactly why it is rarer than the
collection-plane class: most genuinely prolonged induction tracks *are* caught by the
12.5deg displacement-angle excuse.

Note the excuse is computed from the **p1->p2 displacement direction**
(`:1636-1649`), not from either component's own point-cloud shape, so
"cloud-prolonged" and "displacement-excused" are related but not identical — most of
the unprotected residue above is small fragments (a handful of points) peeling off a
large track, not two large track pieces breaking apart. Filtering the 56
unexcused-and-still-split induction candidates to pairs where **both** sides are a
real track (PCA linearity >= 0.75 both sides, calibrated in doc pr/63: 0.79-0.90
confirmed tracks vs 0.63-0.73 confirmed shower blobs; length >= 15cm) leaves exactly
**one clean large-vs-large exhibit**: evt 408534, presented below.

**S7 band (>=30cm) is dump-blind by construction** — `two_d_gap_kill`/`long_gap_kill`
return before the dump write for any `>=30cm` candidate not entering S6, so seed
clouds and render panels are unavailable there; this is a stated limitation, not
engineered around. Using `OC62CENSUS-S7`'s `excuse_u`/`excuse_v` as the topology proxy
across the full 1000-event scan (superseding doc pr/63's smaller 117-event-derived
number): **160 S7-band candidates, 63 killed (39%), of which 36 fully unexcused and 27
killed via one plane despite the other excusing** — the excuse channel does fire
sometimes at this range (unlike doc pr/63's earlier, smaller-sample claim of
"179/184 unexcused"), but a majority of kills are still unexcused. The largest
fully-unexcused S7 kill by component size is evt **286681** comp10(103pt)/comp11(1922pt)
at 31.35cm — included below as a secondary exhibit, with a caveat: comp11's own
point-cloud linearity is 0.54 (shower-range, not confirmed track-shaped), so unlike
408534 this is not a clean track-vs-track break.

## Exhibits

### Collection-plane class (Q1/Q2), before / after, plus a mid arm for attribution

- **pr64-attr-before** (314507, 170814, S6/rescue/S7 all off): https://www.phy.bnl.gov/twister/bee/set/31f2f600-1c3f-42b2-9954-b8ce153d07a5/event/list/
- **pr64-attr-mid** (314507 only, S6+rescue on / S7 off — shows the {1,2,3,4} backbone still connected): https://www.phy.bnl.gov/twister/bee/set/4ad67170-0016-46b8-a640-f1e83e3b6b26/event/list/
- **pr64-attr-after** (314507, 170814, today's production): https://www.phy.bnl.gov/twister/bee/set/e78560db-d87b-4495-85c0-24ffe11456fc/event/list/
- index: bee 0=evt314507, 1=evt170814 (all three zips); pr64-attr-mid has evt314507 only at index 0

**evt 314507** — see Q1/Q2 above; today fully separated into 6 pieces, `bad [R2w thin
collinear pair across a W gap]` machine label on 3-5.

**evt 170814** — a *rescued* W-prolonged counterexample, included for contrast: pair
0-1 has two candidates, `closest` at 1.00cm (W-only gap, `killed_pre_rescue=True`,
**rescued=True today**) and `dir1` at 5.21cm (W-only gap, un-rescued, but irrelevant
since `closest` already keeps the pair connected). `final` stays `[0,0]` in all three
flavors. Shows the same `<=5cm` reach that protects induction cases also protects W
when the gap happens to be close enough — 314507's W gaps (6.68-26.78cm) are simply
all outside that reach.

### Induction-plane analogue (Part 3), before / after

- **pr64-ind-before** (S6/rescue/S7 all off): https://www.phy.bnl.gov/twister/bee/set/fcc641fb-40a7-4459-8ccc-31bb0f3339cc/event/list/
- **pr64-ind-after** (today's production): https://www.phy.bnl.gov/twister/bee/set/150b5b11-1971-449b-b96e-d2e60696c81d/event/list/
- index: bee 0=evt408534, 1=evt286681

**evt 408534, comp0(757pt,86.0cm)-comp2(668pt,51.2cm), dis=19.05cm.** Auto-scan label
`bad [R2 long-track break, no W gap]`. Panel confirms a visually straight track
running through the reconstructed vertex in all three 3D projections. `gapUVW=110`:
**both U and V gapped, unexcused** (`excuse=00`); W panel shows full continuous
coverage (`matrix` all 1s). V panel shows the clearest dashed signature — a blank
band with essentially no fired cells between the two dense, linear point clumps.
`linA=0.96 linB=0.79`, both above the track threshold. Only one candidate (`closest`)
was ever generated for this pair and it is 19.05cm — outside the rescue's `<=5cm`
reach regardless of tier, so it is not merely un-rescued, it was never rescue-eligible.
This is the induction-plane mirror of evt 314507: same auto-label family, same
"real track killed by an unexcused single-plane-class gap, out of rescue reach"
mechanism, opposite plane. The same event also has a **separate S7 kill** at
0-1 (comp0 vs. an 11pt fragment, 46.07cm, fully unexcused) — a second, independent cut
in the same cluster, structurally parallel to 314507's dual S6+S7 story.

**evt 286681, comp10(103pt)-comp11(1922pt), dis=31.35cm, S7-only** (no S6 edge exists
for this pair — the distance is already outside S6's 30cm cap, so S6 never evaluated
it). Fully unexcused S7 kill (`excuse_u=false excuse_v=false`), the largest by
component size in the full 1000-event unexcused-S7 population. Presented as a
secondary example with an explicit caveat: comp11's point-cloud PCA linearity is
**0.54** — shower range, not confirmed track-shaped by the doc pr/63 threshold
(`>=0.75`) — and no seed-cell panel exists for this pair (S7-band dump-blind, see
Part 3). Included because it is the largest available S7-only fully-unexcused case,
not because its track morphology has been independently confirmed the way 408534's
has; inspect the Bee link directly before treating it as a track example.

### S7-kill census set (pr64-s7top), before / after

Every event in the full 1000-sample with >=1 killed S7 (`long_gap_kill`, >=30cm) edge
today — the *entire* population, not a top-N cut: only 21/1000 events ever generate an
S7 candidate, 17 of those have a kill. evt287007 (an S7 kill, but zero PR candidate
selected in either flavor — `TaggerCheckNeutrino` picks nothing) dropped as
unevaluable. Index and per-event kill counts: `docs/pr/pr64-s7top-bee.index.txt`.

- **pr64-s7top-before** (S6+S7 both off): https://www.phy.bnl.gov/twister/bee/set/297de299-3bf8-4209-acf3-95025d44e12f/event/list/
- **pr64-s7top-after** (today's production): https://www.phy.bnl.gov/twister/bee/set/47d8a0ee-329a-4a60-b88e-53db7fca4d72/event/list/
- ranked by killed-S7-edge count: **288859** (36/44, dominant outlier), 314507 (6/6),
  169356 (3/10), **286681** (3/3), **276836** (2/2), 168614 (1/3), then 11 events at
  1/1 each (174224, 288287, 292027, 315497, 319039, 349461, 393042, 394642, 395654,
  400504, 408534).
- evt 276836 (bee idx 4) and evt 174224 (bee idx 6) are the two worked examples below.

## Worked examples from the S7-kill census: evt 276836 and evt 174224

Two follow-up questions on the `pr64-s7top` set (every 1000-sample event with >=1
killed S7 edge today, `docs/pr/pr64-s7top-bee.index.txt`): for evt 276836, which
induction plane (U or V) is the gap on, and is collection all connected? For evt 174224,
a close-to-parallel track, which plane is its gap on? Both answered directly from the
`work-pr64-scan1k` dump (`oc56scan-evt*.jsonl` for S6, `OC62CENSUS-S7` lines in
`wct_pr_evt*.log` for S7) plus rendered panels
(`docs/pr/pr64-panels/pair-evt{174224,276836}-*.png`, reusing
`oc56_render_pair.render()` unchanged — Repro block above).

**A semantics correction first, established via an Explore pass over
`connect_graph_relaxed_strict.cxx` before trusting any number below:** `gap_cm[p]` at
`:1492` is `(1 - reach) * dis`, where `reach` is the *furthest fractional progress*
the S7 corridor BFS made from the **A side only**. It is an **upper bound on the
unreached tail**, not a measured hole length, and it is asymmetric (the B side never
contributes). `gap_cm[p] == dis` means "blocked immediately next to the A seeds", not
"the whole corridor is empty" — a single impassable ring can produce it regardless of
what the rest of the corridor contains. The only thing that actually decides a kill is
`gap[p]` (pure BFS reachability, no cm threshold at all — the census's own `gap_floor_cm`
default is 0.0, permanently inactive) combined with `excused[3] = {excuse_u, excuse_v,
false}` (`:549` — **W is structurally never excusable**; `angle1p`, the W-frame
prolongation angle, is computed at `:1651-1655` but never reaches S6/S7's call sites).
Both examples below are read off `gap[]`/`excuse[]`/the dump's own `fired`/`dead` cell
lists, not `gap_cm`.

### evt 276836 — not "U or V": a drift-parallel track, gapped on all three planes, W is the sole non-excused voter

Components 1/2/3/10 (`final=[0,1,2,2,3,3,3,3,3,3,4]`, cluster splits into 5 pieces) all
run along **±X, the drift direction**, at nearly constant (y,z) — comp1 dir
`(+0.99,-0.02,-0.14)`, comp10 (884 pt / 89.1 cm / lin 0.97) dir `(-0.99,+0.14,+0.09)`:
one long track crossing the cathode, travelling parallel to the drift axis. A track
along X sweeps almost no wires on *any* plane and spreads entirely in time — seed-cloud
tick/wire ratios are 16-56 on **U, V, and W alike** — so `excuse_u=excuse_v=true` on
every edge in this event (topology excuse correctly fires; nothing wrong there).

**So the answer is: it is not specifically U or V, it is a genuine multi-plane time
dropout, and collection is NOT all connected — W is gapped on every killed edge and,
because it alone cannot be excused, is the sole plane that votes.** Decisive edge
`closest 1-3`, dis=5.97cm, `gap=TTT excuse=TT`, killed (rendered panel: bottom row, all
three wire-vs-slice matrices are `0000000000000000` in the shown window — a shared,
plane-independent time gap):

| plane | seed wire boxes A / B | wire gap | fired cells in the 76-tick window between the clouds |
|---|---|---|---|
| U | [1175,1177] / [1176,1179] | 0 (touching) | **0 — completely blank** |
| V | [1001,1006] / [993,999] | 1 wire | 12 in 6/19 slices |
| W | [1028,1034] / [1023,1027] | **0 (touching)** | 14 in 6/19 slices |

U is actually the *worst* plane by raw occupancy (zero charge in the window) and V is no
better than W — yet U and V are excused and only W convicts. Same pattern on `5-10`/
`6-10` (W wire-gap 0-1, U/V excused). `dead=0` on every killed edge in this event: these
are live channels with dropped signal, not dead-channel gaps. The pair that *survives*
in the same cluster, `2-3`/`6-7`, has `gap=.T.` — V-only gapped (V genuinely blank) while
U and W both stay connected — so W's own connectivity, not the excuse mechanism, is what
separates the surviving pairs from the killed ones here. The S7 edge `4-10` (36.64cm,
census) repeats the story with the inversion stated plainly: `gap_cm=[U 4.42, V 36.62,
W 4.40]`, `excuse=TT` — V's corridor never got past its own A-side seeds (`gap_cm==dis`
territory) yet is forgiven; W's BFS got 88% of the way across and is the one that kills.

### evt 174224 — not a prolongation case at all: a plain 6-wire W charge hole, and the same cluster shows the asymmetry with distance held fixed

Components 1 (2164 pt / **175.0cm** / lin 0.96), 2 (1295 pt / 84.4cm / 0.93), 3 (1325 pt
/ 66.0cm / 0.87) are collinear along **±Z** (dirs `(-0.23,+0.15,+0.96)`,
`(+0.18,-0.10,-0.98)`, `(+0.15,-0.11,-0.98)`, z: 188→501cm) — one ~325cm track running
close to parallel to the wire planes rather than to the drift axis. Seed tick/wire
ratios are ~0.7-2.7 (not prolonged), and `excuse_u=excuse_v=false` correctly — there is
nothing here for the excuse to forgive. `final=[0,0,1,1]`: the track breaks into
**{comp1} | {comp2,comp3}**.

**The gap is on W only.** Break edge `closest 1-2`, dis=**1.65cm**, `gap=..T`, killed,
not rescued (panel confirms: U and V wire-vs-slice matrices are solid `1111...1111`
overlapping ranges; W's matrix is `0000...0000` with a genuine 6-wire hole and the
orange dead-channel band sitting *inside* one cloud, not in the hole):

| plane | A wire box | B wire box | verdict |
|---|---|---|---|
| U | [1087,1105] | [1083,1094] | overlap (1087-1094 shared) — no gap |
| V | [1159,1171] | [1150,1165] | overlap (1159-1165 shared) — no gap |
| W | [1108,1118] | [1091,1102] | **6-wire hole (1103-1107), 0 fired cells** |

All three S6 candidates for this pair (`closest` 1.65, `dir1` 5.01, `dir2` 4.92cm) are
W-only gapped and all three are killed — the rescue never reaches it.

**This is the sharpest single-event demonstration of the pr/64 asymmetry found so far**:
the very next pair in the same cluster, `closest 2-3`, has a **V-only** gap at
**1.21cm** — 0.44cm *closer* than the W gap above — and is `killed=False, rescued=True`
(panel: V matrix `0000000011111111`, a genuine gap, but the rescue keeps comp2 and comp3
joined). Same event, same cluster, same graph, comparable 3D distance, opposite outcome,
with the plane as the only thing that differs.

### Why the two look similar but aren't

Both events die on W, for opposite reasons: **276836** is a real, shared, multi-plane
time dropout where W is convicted only because U/V happen to have an excuse channel and
W structurally cannot; **174224** has no prolongation anywhere — W genuinely has a charge
hole U/V do not — and the failure is that the rescue's reach does not extend to a 1.65cm
W-only gap while it reaches a 1.21cm V-only gap 0.44cm away.

Lattice sanity (per the census's own known trap — a nonzero `(s2-s1) % slice_step`
would flag a stride-residue artifact rather than a real BFS block): 276836's S7 edge
`4-10`, `1852-1388=464`, `464 % 4 == 0`; 174224's `1596-1408=188`, `188 % 4 == 0`. Both
clean. `budget=[false,false,false]` on every edge quoted above — the fail-open circuit
breaker is not involved in either verdict.

## Round 3 — is a gated W-plane exception possible without degrading separation?

Owner's follow-up after the two worked examples above: S6/S7 never excuse a W gap —
only U/V get a topology excuse and a signal rescue. Three real long tracks (314507,
276836, 174224) break because of it. But W is also doing a great deal of *correct*
separation work (Q1 above; R3's blanket "W gap ⇒ good" is the majority case). Can a
**narrow** exception fix the first without damaging the second? **Investigation and
prototype only — no C++ or config change, both explicitly out of scope this round.**
Script: `scripts/analysis/pr64/wgate_sweep.py` (imports `oc56_autoscan`/`oc56_fit`/
`oc56_conn`, does not fork them), three subcommands used below.

### Headline: yes, a specific validated gate exists — it's already fitted, just never applied to W

The needed rule is not new. `oc56_autoscan.classify()`'s **R2w** ("thin collinear pair
across a W gap", `oc56_autoscan.py:283-291`) is exactly this exception: `gw and
Lmin>6 and npmin>=50 and Tmax<2.0 and angle<25`, using the **whole-component PCA**
angle (a genuine "is this one straight track" test, not a local wiggle at the break
point — this distinction matters, see Caveat 1). It was fitted against the owner's
899-label hand scan for a different purpose (general W-gap over-splitting) and has
never been applied as a rescue.

**Scored against the raw 899 owner labels** (`wgate_sweep.py owner-labels`, pure
feature/label tabulation — no graph replay, see Caveat 2 for why that distinction
matters), restricted to the 378 W-gapped pairs in `docs/pr/pr57r6-truth.tsv`:

| verdict | n | R2w fires | R2d (dead-W) fires | either |
|---|---|---|---|---|
| bad (want caught) | 12 | 6 | 2 | **8** |
| good (want left alone) | 133 | 1 | 0 | **1** |
| OK | 233 | 5 | 1 | 6 |

**8 of 12 genuine owner-labeled W-gap track breaks recovered for the cost of exactly 1
of 133 correct separations** (evt122660 pair 13-16, `Lmin=10.5, Tmax=1.73, npmin=170,
angle=7.6°` — a real near-miss, worth an owner eyeball before any deployment). The 4
uncaught bad pairs (137238, 170814, 286241, 288287) are all fat (`Tmax` 2.1-14.1) or
kinked (18.7-48.3°) — R2w correctly leaves them alone; loosening further to catch them
was already tried in pr/57 and rejected (`oc56_autoscan.py:88-92`: "recovers evt137238
… costs 5 of 59 good recall").

### Of the owner's 3 named events, only one has direct label confirmation — R2w recovers it, and generalizes out-of-sample to the other two

314507 and 276836 are **not** in the 230-event pr/57 hand scan (they came from this
session's separate, newer 1000-event PR-data pool) — no owner ground truth exists for
them specifically. 174224 pair 1-2 **is** one of the 12 labeled-bad W-gap pairs, and
R2w fires on it cleanly: `Lmin=84.4, Tmax=1.38, npmin=1295, angle=4.2°`.

Applying the *same, un-retuned* rule out-of-sample to the other two: 276836 pairs 1-3
(`angle=10.1°`) and 6-10 (`angle=3.8°`), and 314507 pair 4-5 (`angle=0.9°`) — all
fire, comfortably inside every threshold, consistent with generalizing rather than
being fitted to these specific events.

**Fresh full-sample cross-check** (`wgate_sweep.py blast-radius`, today's production,
`work-pr64-scan1k`, not the label cache — reproduces `classify()`'s verdict directly
from this round's own dumps): 389 killed S6 edges, of which **97 have W as the sole
voting plane** — the complete population any W exception could touch (a W excuse is
monotone toward *more* connection, so this set can only shrink, never grow, with a
gate). Of the 86 fully-killed, W-sole-voter-revivable **pairs**, `classify()` calls 82
`good [R3]` and exactly the same **4** `bad [R2w]` as above (174224, 276836×2, 314507).
Of the 82 good, only 7 have `Lmin>10cm`, and **every one is vetoed by R2w's own angle
cut** — nearest correct-separation angle is 24.6°, vs. the 4 targets' 0.9-10.1°. Same
wide margin, two independent checks (owner labels; fresh full sample).

### S7 needs no exception in this sample

Of 63 killed S7 (≥30cm) edges, 25 are W-sole-voter; **0 pass any reasonable gate**
(`Lmin>10 & Tmax<2 & angle<15` → 0/25). The only substantial candidate, 276836's
`4-10` (`Lmin=54.2, npmin=421`), fails on collinearity at **40.3°** — a real kink, not
an artifact. Stated plainly: even a shipped W-exception would only *partially* repair
276836 (its two S6 pairs reconnect; its S7 break at 36.64cm does not), and would not
touch 314507's S7 edges at all (`U,V,W` all vote there — no W-only case exists in that
event's S7 census). This piece of the question is a clean "no, and the sample already
proves it" — recommend leaving S7 untouched.

### Caveat 1 — local vs. global collinearity is not a detail

The production S6 rescue (doc pr/57 round 6, `oc56_fit.rescue()`, shipped) already has
a narrow W branch (`w_tiny`: gap closes within a 3-wire stencil + coverage≥0.8;
`collw`: **local** break-point PCA angle <15°, computed via fresh radius queries at
the break — `ab_local`). Checked against 174224 pair 1-2's real feature record:
`close_mx_w=5` (misses `cw≤3` by 2 wires) **and** `ab_local=20.67°` (misses `collw<15`
— even though the track's **whole-component** angle is 4.2°). `wgate_sweep.py sweep`
confirms this is not a rounding issue: widening `cw` alone up to 8 never recovers
174224 (`cvw=0.786<0.8` also fails), and widening the **local** angle threshold to
catch it (~20°) costs **8** good pairs elsewhere (69314, 122660×2, 18625, 60669,
174752, 348691, 394642). Using the **global** axis instead (R2w's feature) at the same
widened angle recovers *more* bad pairs (118→122/127 as the threshold opens) while
breaking *fewer* good ones (152-151/156) than the local-axis sweep ever achieves — a
strictly better trade, not a matter of degree. A genuinely straight long track can
have a noisy local angle right at one break point while its whole-track PCA axis stays
tight; local break-point axis is the wrong feature for this question.

### Caveat 2 — the existing rescue-evaluation harness replays a stale, pre-S7-flip graph; demonstrated, not just asserted

`oc56_fit.DEFAULT_ARMS` are all `work-pr57r4-*` — round-4 dumps that predate doc
pr/62's S7 production flip (2026-08-11) and everything shipped since. Concretely: the
harness's own replay scores 174224 pair 1-2 as a **correct hit** — but not because any
W rule fired (`F.rescue()` returns `False` on all 3 of its candidate edges, checked
directly). The round-4 conn record for this graph_call contains a stray **82.7cm MST
edge (comp1–comp3)** that no longer exists in today's bare production
(`work-pr64-scan1k`'s own connectivity record has only the 0-1 and 2-3 edges — the 1-3
edge is gone, almost certainly killed once S7 started running). The replay walks
comp1→comp3 via that phantom edge, then a *separately*-rescued V-gap edge (2-3, dis
1.21cm) bridges the rest — the pair "passes" for a reason unrelated to W, and does not
reflect today's actual graph, where this pair is genuinely broken (`final=[0,0,1,1]`,
confirmed in the worked-example section above).

**What this does and doesn't invalidate**: pure feature/label rule-firing (the 8/12
vs. 1/133 table, and R2w applied directly to fresh `work-pr64-scan1k` records) does
**not** touch graph replay and is unaffected — trustworthy as reported.
`wgate_sweep.py sweep`'s numbers (used above only to demonstrate the local-vs-global
axis point, a *relative* comparison) inherit the staleness risk for any *absolute*
claim and should be treated as indicative pending a fresh, post-pr/62 label-arm rerun
— out of scope this round. This is a real, pre-existing limitation of
`scripts/analysis/pr57/oc56_fit.py`'s validation tooling, independent of the W
question, worth fixing before anyone trusts its replay output for future rescue work.

### C++ feasibility — the needed features already exist, cached, one step from the call sites

- `s6_comp_stat(comp)` (`connect_graph_relaxed_strict.cxx:676-709`) already computes a
  full 3D covariance + `Eigen::SelfAdjointEigenSolver` per component, **memoized** in
  `s6_comp_stats` (`:667`) — O(1) after first touch, not recomputed per pair. This is
  `Lmin`/`npmin` already; the eigenvalues computed at `:696` are discarded — a
  transverse-RMS "thin" feature (`Tmax`) is ~2 lines away.
- The **global** principal-axis direction is not currently kept (only the extent is
  returned); the existing collinearity check (`s6_local_axis`, `:714-733`) recomputes
  a **local** axis via fresh radius queries per candidate — the feature Caveat 1 shows
  is the wrong one here. Extending `s6_comp_stat`'s cached return to also keep the top
  eigenvector (already computed, just discarded) gives the global axis **for free** —
  cheaper than the local approach already in production, which redoes a radius-query
  PCA per candidate.
- No call-site signature change needed: `two_d_gap_kill`/`long_gap_kill` are
  `[&]`-capturing lambdas defined after these helpers; `two_d_gap_kill` already calls
  `s6_comp_stat`/`s6_local_axis` directly at `:1140-1149`.
- Composition precedent (`Graphs.h:323-326`): the existing rescue is purely
  subtractive, `killed && !two_d_rescue_ok(...)`. A W-exception should follow the same
  shape — **not** touch the pinned `excused[3]={excuse_u,excuse_v,false}` inside
  `long_corridor_bad`/`two_d_connectivity_bad` themselves (doctest-pinned).
- Knob plumbing precedent (template `long_check`): jsonnet `protect_graph_name` →
  `ClusteringProtectBundle.cxx` `graph_name` (default `"relaxed"`) → **four** dispatch
  sites in `Facade_Cluster.cxx` (`:2842,2902,3043,3121` — miss one and it silently
  falls through to the wrong graph) → `make_graphs.cxx` factory → new bool param
  defaulting OFF.
- Undecided, not blocking: `s6_comp_stat`'s 20000-point cap was fitted for the ≤5cm
  band; a ≥30cm exception (not recommended above anyway) would need an explicit call
  on whether to keep it.

### What is not validated — reported, not papered over

- **S7 band**: zero owner labels exist for any W-only S7 kill; recommendation to leave
  it alone is based on the fresh sample showing no gate would fire there, not on
  direct owner confirmation that none *should*.
- **evt122660's one false positive** is a real near-miss (`angle=7.6°`, well inside a
  25° threshold) — worth an owner eyeball, not auto-accepted.
- **This is entirely offline dump/label analysis, not a measured production A/B.** No
  C++ touched, no config flipped, no chain rerun with any exception active. Cascade
  effects through MST edge selection, downstream clustering, and nusel are
  unmeasured. Real deployment needs: default-OFF knob, byte-identical off-gate, full
  A/B on the standard manifest, and ideally fresh owner labels in the 5-30cm W-gap
  band specifically (mirrors the gap doc pr/63 already flagged for U/V).
- **Caveat 2's stale-harness finding** should be fixed (regenerate `oc56_fit.py`'s
  label arms post-pr/62) before trusting its replay output for any future rescue
  work — a pre-existing tooling gap, surfaced by this round, not created by it.

## What a fix would need (not built this round)

- **Collection plane**: superseded by Round 3 below — `angle1p` was the first-guess
  fix candidate but turns out not to be the right feature (it is the *displacement*
  angle, same family as `angle1`/`angle2`; Round 3 found the *whole-component PCA*
  angle is what actually separates the owner-labeled bad cases from the good ones,
  with a wide margin). Round 3 has a concrete, owner-label-validated gate (R2w) and a
  C++ feasibility path using already-cached per-component statistics — still needs the
  full default-OFF knob + byte-identical gate treatment and a real production A/B
  before deployment, but the open question changed from "is there a candidate
  feature" to "run the standard deployment checklist on this specific one."
- **Induction plane**: the rescue's `<=5cm` reach ceiling (doc pr/63) is the binding
  constraint for evt 408534 specifically (19.05cm, no closer candidate exists) — same
  open item doc pr/63 already flagged, now with a second confirmed exhibit.
- **S7 band**: no rescue mechanism exists at all; 36/160 (1000-event count) fully
  unexcused kills is a larger, more representative number than doc pr/63's earlier
  117-event-derived one, but still no dump/panel evidence for what those kills'
  cloud shapes actually look like — that blind spot (`OC56CENSUS`/`OC62CENSUS-S7`
  return before the dump write for `>=30cm` candidates) is structural to the current
  logging, not just a gap in this round's search.

## Fix — none shipped in Rounds 1-3

Rounds 1-3: no code change, no production flip (investigation + prototype script
only, per explicit scope). Round 4 below ships the fix.

## Round 4 — implementation: `two_d_w_track_ok`, the harness fix, and deployment

Owner request (2026-08-11): implement the Round 3 gate in C++, fix the stale
validation harness (Caveat 2), correct the narrow-W-branch deficiency (Caveat 1),
with the explicit success criterion "no regression against my existing scan, only
recover these three events and similar cases — narrow, W-plane long-track". Owner
selected the **tightened operating point** when offered the trade:

> base R2w (`W sole voting plane && Lmin > 6cm && npmin >= 50 && Tmax < 2cm &&
> angle < 25°`) **plus** (`Tmax < 1.7cm` OR `angle < 6°`)

The tightening excludes evt122660's good pair 13-16 (`Tmax=1.73, angle=7.6°`) —
zero good-pair regressions on the 899-label scan — at the cost of the two marginal
recoveries 64959/172656 (angles 21.9°/18.4°, not clean-long-track topology).

### Repro

```bash
# harness fix: regenerate the two label arms that predate the pr/62 S7 flip
cd wcp-porting-img/sbnd/sbnd_xin
PR_OC56_SCAN_DUMP=1 PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 \
    work-pr64r4-scan48 data $(cat valfast/events-nuecc48-cb0805.txt)
PR_OC56_SCAN_DUMP=1 PR_JOBS=32 ./run_pr_chain_batch.sh work-ncpi0-cb0805 \
    work-pr64r4-scan19 data $(cat valfast/events-ncpi0-cb0805.txt)
python3 scripts/analysis/pr57/oc56_fit.py features --out /home/xqian/tmp/pr64r4_edges.jsonl
python3 scripts/analysis/pr64/wgate_sweep.py fresh-labels     # owner labels vs FRESH features
# gates (byte-identity = member-content hash, never raw archive cmp)
python3 scripts/analysis/pr64/pr64_gate.py work-pr64r4-scan48 work-pr64r4-off48
python3 scripts/analysis/pr64/pr64_gate.py work-pr64r4-scan19 work-pr64r4-off19
python3 scripts/analysis/pr64/pr64_gate.py work-pr64-scan1k   work-pr64r4-off1k
# on-arms: SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue_long_wtrack + same env
# doctest
cd toolkit && ./build/clus/wcdoctest-clus -tc='s6 w-track*'
```

### The harness fix (Caveat 2, closed)

`oc56_truth.py DEFAULT_ARMS` now points at post-pr/62 bare-production dumps:
`work-pr64r4-scan48/19` (regenerated 2026-08-11, 66/66 events rc=0) +
`work-pr64-scan1k` (the full 1000-event PR-data pool, superset of the old
scan50+scan395 events). The archived `pr57r6-truth.tsv` stays untouched
(scientific record; `--arm` selects the old arms explicitly). The hardcoded
population map at `oc56_fit.py` (which silently degraded any unknown arm name to
its raw basename) now routes through `oc56_truth.POPULATION`.

Demonstrated fixed: evt174224 graph_call 0's fresh connectivity record contains
only the `0-1` and `2-3` MST edges — the phantom 82.7cm `1-3` edge that made the
old replay score this pair as "already handled" is gone from today's production,
exactly as Round 3's Caveat 2 predicted.

Fresh-feature label scoring (`wgate_sweep.py fresh-labels`, all 899 labels join
today's production, **0 orphans**): of 378 W-gapped labelled pairs — bad 12,
good 133, OK 233. The shipped tightened gate fires on **7/12 bad** (60017,
60669×2 via R2d, 172656 — whose fresh `Tmax` lands just inside 1.7 — 174224,
287517, 407280), **0/133 good**, and 5 OK pairs (54095 5-11 via R2d, 122660 4-6
and 5-6, 48367 9-10, 409634 2-3 — owner-free verdicts, all expected movers).

### The C++ (Caveat 1 corrected as designed: additive global-axis branch, shipped local branch untouched)

- **`Graphs::two_d_w_track_ok(S6WTrackInput)`** (`Graphs.h`,
  `connect_graph_relaxed_strict.cxx`): pure predicate, R2d (dead-W band,
  `w_gap && dead_w>=3 && npmin>=20 && dis<3cm`, deliberately NOT sole-vote-gated —
  the fitted evt60669 population has an induction plane also voting) then
  tightened R2w. All thresholds `constexpr` in one block, in lockstep with
  `wgate_sweep.py r2w_tight_fires()`.
- **`s6_comp_stat` cache widened** to `S6CompStat{npoints, extent_cm, axis,
  trms_cm}` — the top eigenvector and transverse RMS were already computed by the
  existing per-component eigen-decomposition and discarded; keeping them is free
  and strictly cheaper than the local-axis radius-query PCA the shipped rescue
  uses. `trms` uses the (n-1) normalization to match the Python fit's `np.cov`.
- **Composition**: after the existing rescue block, `if (w_track_excuse &&
  killed)` — revive-only, no <= 5cm cap (276836's 5.97cm pair is in scope), sole
  vote = `gap[2] && !(gap[0]&&!excuse_u) && !(gap[1]&&!excuse_v)` (identical to
  the offline voter definition). The pinned `excused[3]={excuse_u,excuse_v,false}`
  and the shipped `two_d_rescue_ok` W branch are byte-for-byte untouched. Same
  dir-MST emission repair as the rescue (evt286180 class).
- **Flavor**: `relaxed_strict_img_2d_rescue_long_wtrack` =
  `_rescue_long` + `w_track_excuse=true`; new factory + all four
  `Facade_Cluster.cxx` dispatch sites. C++ default stays OFF everywhere.
- **Dump provenance**: `w_track`/`w_track_revived`/`v3{w_gap,w_sole,npmin,lmin,
  tmax,ab_global,dead_w}` — absent unless the flavor enables it, so every
  existing flavor's dump records are unchanged.
- **Doctest** `clus/test/doctest_w_track.cxx`: 6 cases / 33 assertions — the
  three target events, the protected 122660 pair, the forgone marginals, every
  R2w boundary, the tightening's two branches, R2d boundaries + its `w_gap`
  gate, sentinel behavior. Full `wcdoctest-clus`: 170 cases / 1783 assertions
  pass.
- **S7 untouched** — Round 3's null result stands (0/25 W-sole S7 kills pass any
  gate).

### Gates (labels reported; `pr64_gate.py` = member-content hashes of mabc-pr.zip + pctree tar.gz + exact-byte nusel tsv)

- Freshness proof: `libWireCellClus.so` 2026-08-11 14:39 > last source edit
  14:36; `wcbuild` + `./wcb install` rc=0.
- **Off-gate** (new binary, knob off, vs pre-change same-day baselines):
  - `work-pr64r4-scan48` vs `work-pr64r4-off48`: **47/47 identical, 0 movers**
  - `work-pr64r4-scan19` vs `work-pr64r4-off19`: **19/19 identical, 0 movers**
  - `work-pr64-scan1k` vs `work-pr64r4-off1k`: **990/1000 identical, 10 movers —
    ALL attributed to doc pr/65's production flip, not this change** (pr/65's
    cfg flip landed 13:53 PDT, after scan1k's 09:43 generation; every mover's
    diff is confined to `mc.json` ± `shower_track-global.json` — exactly
    pr/65's PF/shower-absorb footprint — with `absorb_unreachable_main` /
    `orphan_audit_only` active in every off1k log and absent from every scan1k
    log; pctree + nusel byte-identical in all 1000; knob-off dumps carry zero
    `w_track` keys). Same situation as pr/53's "no pre-existing off arm is
    today's bare"; `work-pr64r4-off1k` is today's bare baseline, and the
    binary-level identity of THIS change is the 0/66 same-config result above.
- **On-arm** (`SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue_long_wtrack`,
  same binary, same config, vs the off arms — the clean knob isolation):
  - `work-pr64r4-off1k` vs `work-pr64r4-on1k`: **997/1000 identical; the 3
    movers are EXACTLY the target events** — 174224, 276836, 314507
    (mabc + pctree only; nusel byte-identical in all 1000).
  - `work-pr64r4-off48` vs `work-pr64r4-on48`: 46/47 identical; the 1 mover is
    evt122660 (see below — the protected pair is NOT the thing that moved).
  - `work-pr64r4-off19` vs `work-pr64r4-on19`: **19/19 identical, 0 movers.**
- **Mover census / causal match** (every `w_track_revived` record in the on
  arms): 6 events fire on the 1k sample — 174224 (1-2, closest+dir1+dir2),
  276836 (1-3 at 5.97cm and 6-10 at 7.91cm, both beyond the old rescue's 5cm
  reach), 314507 (4-5 at 26.78cm), plus 172656/287517/407280 whose pairs are
  already connected in today's production (dir-block revives, no membership
  change — which is why they are not movers). All revives are `w_sole=True`;
  C++ full-cloud features match the Python strided-cloud fit to the printed
  precision (174224: tmax 1.38 / ab 4.2 both sides). **R2d fired zero times**
  in 1067 events — the shipped rescue's dead-W branch already handles that
  band (evt60669's candidates never reach the exception), so R2d is a dormant
  safety net, kept for its label fit but currently exercised only by the
  doctest.
- **Membership deltas** (connectivity records, off → on):
  - 174224: `[0,0,1,1] → [0,0,0,0]` — the track is whole again.
  - 276836: 5 groups → 3 (pairs 1-3, 6-10 rejoin; the 40° S7 kink at 36.6cm
    stays split by design).
  - 314507: 6 groups → 5 (pair 4-5; its all-plane-voter S7 breaks remain by
    design).
  - 122660: components 4/5/6 merge (pairs 4-6/5-6 owner-verdict OK, 4-5
    unlabelled, all three thin-collinear, ab 3.7-5.6°); **the owner-labelled
    good pair 13-16 stays split on-arm** (final groups 9 vs 10) — verified in
    the connectivity record, not inferred.

### Bee before/after (`docs/pr/pr64r4-bee.index.txt`)

- **pr64r4-before** (today's bare, knob off):
  https://www.phy.bnl.gov/twister/bee/set/fa4f8d6b-3e78-46bc-bb1c-5ec774217fdc/event/list/
- **pr64r4-after** (W exception on):
  https://www.phy.bnl.gov/twister/bee/set/e6e78f04-7e3a-4af3-bf77-b1af847d7630/event/list/
  — idx 0 **174224**, 1 **276836**, 2 **314507**, 3 **64959** (forgone case,
  must look identical).
- **pr64r4-48-before**:
  https://www.phy.bnl.gov/twister/bee/set/e60f2d53-0810-4718-82c6-a4ecdc727903/event/list/
- **pr64r4-48-after**:
  https://www.phy.bnl.gov/twister/bee/set/1da5bb71-c3bf-47dc-b7c9-8e6019175c12/event/list/
  — **122660**, the one non-target mover: the protected good pair stays split;
  what merges is the 4/5/6 component chain.

### Production flip — SBND ON (owner pre-authorized "if the validation passes")

`wct-pr-perevt.jsonnet` `protect_graph_name =
'relaxed_strict_img_2d_rescue_long_wtrack'`. Cfg-only; C++ default stays OFF
(doc 68 single-source). Proofs (full production `pipeline_names` incl.
`protect_bundle` + `reality` TLA set):
- post-flip bare compiled JSON **byte-exact `cmp`** == on-arm
  `-A protect_graph_name=..._wtrack` config (the validated arms above ARE the
  flipped production);
- `-A protect_graph_name=relaxed_strict_img_2d_rescue_long` sole-key legacy
  escape **byte-exact `cmp`** == pre-flip bare (restores pre-pr/64 production);
- compiled-config proof: `"graph_name" : "..._rescue_long_wtrack"` appears
  exactly once post-flip, zero pre-flip.

## Cross-links

- [[project_pr63_prolonged_dashed_tracks]] — the induction-only search this doc
  supersedes with the fuller 1000-event scan and the collection-plane counterexample;
  the `<=5cm` rescue ceiling and PCA-linearity track filter this doc reuses
- [[project_pr62_long_edge_corridor]] — S7 mechanics and the original (smaller-sample)
  S7 census this doc's 1000-event S7 numbers supersede
- doc pr/56 sec 8 — the original prolonged-signal report and rescue tier constants
- `clus/docs/clustering/parallel-prolong-analysis.md` — independent confirmation of
  SBND's W=0deg wire-angle convention, from an earlier/different clustering stage
  (`clustering_parallel_prolong`) that already reads all three plane angles including W

## Round 5 (2026-08-11) — evt 18259-18625: "still missing clustering points at (142.1, 78.3, 176.5)"

**Status: investigation + fix proposal only. No code change, no production
flip.** The owner reports that after the round-4 flip a track segment in evt
18259-18625 still shows missing clustering points near
`(142.1, 78.3, 176.5)` cm. Round-4 declared 18625 a protected *good* pair
(Caveat 1) rather than a target, so this is worth tracing precisely rather
than assuming the fix is incomplete.

### Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# fresh bare-production single-event rerun (cfg default = round-4 flavor,
# same QL hub the *-cb0805 prid-maps were built from; fresh name per M13):
PR_OC56_SCAN_DUMP=1 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr64r5-verify data 18625

# the read-only probe that produces every number below (arms + the uploaded
# Bee set + the S6 edge dump + the owner's own label file):
python3 scripts/analysis/pr64/probe_18625.py \
  --arm work-pr64r5-verify/pr_evt18625 \
  --bee bee/prod0811/ncpi0-prod0811.zip

# owner's hand-scan verdict on the exact edge:
grep -A2 '144.02,78.92,173.63' overclustering_labels/labels-evt18625.json
```

### Symptom, and four literal readings ruled out

| candidate reading | measured | verdict |
|---|---|---|
| img charge with no clustering point nearby (a true imaging/clustering hole) | 0 orphan img points within 10 cm of the target; nearest clustering point **0.047 cm**, nearest img point 0.286 cm | ruled out |
| a fitted trajectory with no charge under it (the pr/61 "phantom edge" class) | **0 of 662** track_fit points event-wide sit >2 cm from a clustering point | ruled out |
| a PF segment created with zero associated (`shower_track`) points (the pr/59 class) | 25/37 segments in this event, but **65.6 % (764/1165) across the full 19-event NCpi0 arm** — the norm for this arm, not an 18625-specific defect | not event-specific, real but out of scope here (see "Separate observation" below) |
| the neutrino loses the shower's reconstructed energy | `T_kine`: `kine_reco_Enu = 1499.34 MeV`, `kine_energy_particle` includes **717.24 MeV** (`particle_type=11`, `kine_energy_included=1`) against a truth γ→e⁻ of 717 MeV — matched to 0.01 MeV | ruled out |

### Root cause — an S6 kill at the photon-conversion gap, and pr/64's gate is structurally inapplicable

`ClusteringProtectBundle` (S6, doc pr/56/57) splits the neutrino's 6996-point
image cluster into PR clusters **11** (3390 pts, muon+proton+vertex) and
**126** (3169 pts, the converted-photon shower). The target coordinate is
0.047 cm from a cluster-126 point, 2.4 cm from the split. The only bridge is
killed at a **1.876 cm, pure-drift-direction** gap where **W is the sole
voting plane** (`oc56scan-evt18625.jsonl`, `j=0 k=1 blk=closest graph_call=0`,
reproduced identically across three independent runs this round — the two
round-4 gate arms and the fresh rerun above):

```
dis=1.876cm  gap=[U,V,W]=[T,T,T]  excuse=[T,T]  dead_w=0
killed_pre_rescue=true  rescued=false          <- pr/57 rescue declined
w_track=true  w_track_revived=false            <- pr/64 declined
v3: npmin=3062  lmin=52.48cm  tmax=12.75cm  ab_global=43.39deg
```

pr/64's tightened gate needs `tmax<2.0cm` (or `<1.7`) and `ab_global<25deg`
(or `<6`). This pair misses by **6.4×** on thinness and **1.7×** on
collinearity — not a near miss at the boundary, but a different population:
`two_d_w_track_ok` is a thin/long/collinear-track rule (fitted to 174224-class
tracks, `npmin` in the tens to low hundreds), and this pair has `npmin=3062`
with a transverse spread an order of magnitude too wide to be a track. The
dumped W fired-pixel map (below, 4-tick columns) shows the real gap: a clean
~20-tick (~1.7 cm) break, same wire column (562-569 vs 578-599), `dead_w=0` —
not a dead-channel artifact.

```
    562 ####..####................
    563 ####..####................
    ...
    569 #.....####................
    570 ......####................        <- gap opens: U/V bridge, W does not
    ...
    577 ......####................
    578 .......###......#.........        <- second W blob begins (comp 126)
    579 ......####.....####.......
```

### The blocking conflict — the owner's own label calls this cut correct

`overclustering_labels/labels-evt18625.json` carries `verdict: "good"` on this
exact edge (`144.02,78.92,173.63|145.89,78.92,173.63`, `dis=1.8756`) and on
its `dir1` sibling, and this doc's own Caveat 1 (round 4) already names 18625
as one of the 8 good pairs the tightened gate deliberately protects. This is
the NC-π⁰ sample: a real 9.76 cm photon-conversion gap is *expected* physics
here, not necessarily a defect. A nearest-axis test on the charge between the
vertex and the conversion point splits 47/46 between the γ-axis and the
27°-away µ-axis — genuinely ambiguous, not a clean call either way. **No rule
change is proposed as a "fix" below without the owner first deciding whether
this verdict should flip.**

### An independent, literal match to the symptom — a Bee-layer inconsistency in the uploaded set

Probing the actual uploaded Bee set the owner is looking at
(`bee/prod0811/ncpi0-prod0811.zip`, produced today after the round-4 flip)
turns up something that was **not** reproduced in any of this round's three
fresh reruns, and is a more literal match to "a track segment missing its
clustering points":

- `clustering-global` in that zip has **cluster 11 alone, 6996 points** — no
  cluster 126 at all; the entire 126-family of `real_cluster_id`s is absent.
- `track_fit-global`, `shower_track-global`, and `vertices-global` in the
  *same zip* all still reference **15 distinct `126xxx` segments**, including
  `126042` (the e⁻ 717 MeV segment nearest the reported coordinate, which
  still has 2614 associated points in `shower_track-global`).

So in this specific uploaded set, a fitted, associated 126-family trajectory
is drawn with **no underlying charge cloud at all** in the layer Bee uses for
cluster coloring/selection — a stronger and more literal defect than the S6
split itself, which (per `nusel_extract.py`'s own documented convention,
`clustering-global` is filtered to `switch_scope`'s active/in-scope clusters)
may be legitimate scope-filtering of a non-"main"-flagged shard rather than a
bug in itself.

**This could not be pinned down further this round**: the three fresh reruns
here (`run_pr_chain_batch.sh`, same QL hub, same cfg default) all show cluster
126 present in `clustering-global` with 3169 points, consistent with each
other and with `track_fit`/`shower_track`. Whatever generated
`ncpi0-prod0811.zip` is not `run_pr_chain_batch.sh` (its driver script was not
found in this repo), and the difference between the two could be a genuine
`switch_scope` behavior difference, a different beam-window/manifest option,
or something else entirely. **Recommend as a next step**: identify the exact
script/config that produced `ncpi0-prod0811.zip` (it landed in the 2026-08-11
17:07 retirement-round commit, `e86a81b`, "delivered three fresh Bee scan sets
at current production") and diff its `switch_scope` configuration against
`run_pr_chain_batch.sh`'s.

### Fix proposal — two families, nothing shipped

**Family 1 — a shower-aware S6 branch (needs the label reversed).** A third
branch of `Graphs::two_d_w_track_ok` (`clus/src/connect_graph_relaxed_strict.cxx`),
reusing the `s6_comp_stat` memo round 4 already extended
(`npoints`/`extent_cm`/`axis`/`trms_cm`): revive iff `w_sole_vote && dis <
2.5cm && npmin >= 500 && dead_w == 0` **and** the two endpoint cells share the
same W wire (`|Δwire| <= 1`) — i.e. the components are not separated in wire
space at all, only in drift/tick. That last term is the discriminator that
keeps this from becoming "revive every big pair" and is computable from data
S6 already caches. Would ship as a new default-OFF flavor, never by widening
the shipped R2w thresholds (which would silently reopen the validated
round-4 operating point and its 0/133-good-pairs guarantee).

**Family 2 — no code change (compatible with the existing "good" label).**
The separation stands; the PF chain already claims the shower as its own
particle and the energy is already counted (see above). The only genuine
per-point residual inside the whole neutrino bundle is small: **17 of 6996**
cluster-11 clustering points (in the merged-view `bee/prod0811` zip) have no
PF-association point within 2 cm, concentrated in a 12-point blob at
`x[142.8,143.7] y[78.3,79.8] z[172.9,174.7]` — 2.36 cm from the reported
coordinate.

**Recommendation: Family 2 (no action needed on the split) now; Family 1 only
if the owner flips the 18625 label.** If flipped, Family 1 must go through the
same process round 4 did before any flip — re-score all 899 labels against
the new branch, a 1k-sample census of how many currently-killed S6 edges
satisfy the shape, a byte-identical off-gate, and an on-arm mover census —
not a threshold fitted to one event. Separately, the clustering-global/
track_fit layer inconsistency above should be tracked down regardless of the
label decision: it is a display-correctness question, not a physics-tuning
one.

### Separate observation — not part of this round

Zero-association PF segments (the pr/59 class: a segment created after its
cluster's only `clustering_points` association pass) are common in this arm:
**65.6 % (764/1165) of all fitted segments across the 19-event NCpi0
manifest** have zero associated points, evt18625 included (25/37) but not
unusually so. Worth its own round; not fixed here.

### Files

- `scripts/analysis/pr64/probe_18625.py` (new) — the read-only probe behind
  every number above.

## Round 6 (2026-08-11) — the seam gap at (142.1, 78.3, 176.5): root cause found, fix proposed

**Status: root cause identified, fix proposed. No code change, no production
flip.** Supersedes Round 5's "Family 2" paragraph, which filed this as a minor
17-point footnote under the S6 discussion. On closer look (owner pointed
directly at the Bee set, comparing `img-global` vs `shower_track-global`
layers), it is the precise, literal, and now fully explained answer to "a
track segment missing its clustering points" — a real algorithmic gap,
independent of the S6 split story, small in absolute size (12-18 points, no
detectable energy impact) but cleanly diagnosed and worth fixing on its own
merits.

### Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# confirms which Bee set/event the owner is looking at:
for f in bee/prod0811/*.url; do echo "$f: $(cat $f)"; done
head -2 bee/prod0811/ncpi0-prod0811.index.txt   # bee_idx 0 = evt 18625

# img-global vs shower_track-global, direct, in the exact uploaded zip
# (ad-hoc probe this round -- not added as a script, see "Files" below):
python3 - <<'PYEOF'
import zipfile, json, numpy as np
from scipy.spatial import cKDTree
z = zipfile.ZipFile('bee/prod0811/ncpi0-prod0811.zip')
L = lambda n: json.loads(z.read(f'data/0/0-{n}.json'))
im, st = L('img-global'), L('shower_track-global')
Pi = np.stack([im['x'], im['y'], im['z']], 1)
Ps = np.stack([st['x'], st['y'], st['z']], 1)
d, _ = cKDTree(Ps).query(Pi)
T = np.array([142.1, 78.3, 176.5])
orphan = Pi[d > 1.5]
near = orphan[np.linalg.norm(orphan - T, axis=1) < 5]
print(len(near), 'orphan img points within 5cm of the reported coordinate')
print('bbox', near.min(0), near.max(0))
PYEOF
```

### Root cause

`shower_track-global` is filled directly from each PF segment's
`dpcloud("associate_points")` (`clus/src/MultiAlgBlobClustering.cxx:884-899`,
`use_associate_points` mode). That cloud is built once, for every segment in
the event, by `clustering_points_segments`
(`clus/src/PRSegmentFunctions.cxx:2742`), a three-stage Voronoi partition of
each cluster's own point cloud:

- **Stage A (terminal seeding, :2894-2916)**: each segment seeds Voronoi
  "terminals" only from its **interior** fit points —
  `for (size_t i = 1; i+1 < fits.size(); i++)` (:2896) explicitly skips the
  first and last fit point of every segment's trajectory. A segment's own
  endpoints never seed a terminal.
- **Stage B (:2951)**: a graph-shortest-path Voronoi partition
  (`Graphs::Weighted::voronoi`) assigns every point in the cluster to its
  nearest terminal — a full partition, no gaps yet.
- **Stage C (ghost removal, :2961-3060)**: each point's Voronoi-assigned
  segment (`main_sg`) is re-checked against the true 2D wire-plane-projection
  nearest segment (via `global_kd2d`, F17's global KD-tree spanning **all**
  input segments, deliberately including other clusters' segments too, for
  cross-cluster ghost detection). If `main_sg` doesn't win that check
  (`flag_change`, :3017, stays `true`), the point is **dropped** — never
  reassigned to whichever segment actually does win. `map_segment_points`
  (filled at :3056-3057, read at :3070 by `create_segment_point_cloud` →
  `associate_points`) only ever gains entries from points that passed this
  check.

**Traced to the exact spot**: the orphan blob (12-18 points, both in the
uploaded zip and independently in a fresh rerun this round, same location) is
centered at `(143.3, 79.0, 174.1)`, **0.1-0.7 cm from segment 126042's own
trajectory endpoint** `(143.57, 79.25, 173.47)` — the reconstructed
photon-conversion vertex (also present in `vertices-global`, and the same
point named `real_cluster_id=-1` in `track_fit-global`'s vertex-marker rows).
Because that endpoint is explicitly excluded from terminal-seeding
(Stage A), and no other segment has a fit point close enough to win the
Stage-C check there either (segment 11065, the muon track, is 5 cm away at
its nearest; the other cluster-126 sub-segments are 15-19 cm away), the
charge immediately around this vertex gets Voronoi-assigned to some distant
terminal, fails the geometric re-check, and is dropped rather than handed
to 126042 (the segment it actually belongs to) — even though 126042 is
present in `segs` for this cluster and is the obvious rightful owner.

This is a structural blind spot at **every** segment endpoint, most visible
at vertices like this one where a shower trunk terminates and nothing else
picks up the slack nearby. It is unrelated to the S6 split discussed in
Round 5: 126042 is entirely within one PR cluster (126); this is a
within-cluster association gap, not a cross-cluster one.

### Fix proposal — two options, neither implemented

**Option B (recommended) — make Stage C reassign, not just drop.** The F17
global KD-tree query already identifies, per plane, the single input point
that achieves the 2D-projection minimum (`res[0].first`, the flat index into
the KD-tree's own arrays) — currently only its *distance* is kept
(`min_2d_dis2`), the identity is discarded. Cheap, additive change: alongside
`xs`/`ys` when building `global_kd2d` (:2864-2874), also append a parallel
per-point owning-segment id; when a point fails the `main_sg` check
(`flag_change == true`), look up which segment achieved the global 2D
minimum via that parallel array, and if — and only if — that segment is a
member of the **same cluster's** `segs` (never a foreign cluster's segment;
cross-cluster ghost rejection is unaffected and stays exactly as strict as
today), assign the point there instead of dropping it. This directly closes
the seam-gap class in general (any two adjacent same-cluster segments whose
graph-Voronoi and 2D-projection nearest-segment disagree near a shared
vertex), not just the endpoint-starvation case specifically. O(1) extra work
per already-computed KD-tree query; no new pass.

**Option A (narrower alternative) — seed terminals at true dead-end
endpoints.** Only seed a terminal at a segment's own first/last fit point
when no other segment's fit point already exists within a small radius of it
(i.e. it is a genuine trajectory dead end, like 126042's shower-terminating
end here) — a shared vertex where two segments meet, each already covered by
the other's interior terminals, would keep the current exclusion. Narrower
in scope than Option B (only fixes dead-end starvation, not general seam
disagreement), and needs a radius threshold choice to fit and validate.

**Recommendation: Option B.** It fixes the general mechanism (Option A only
patches one manifestation of it), needs no new tunable threshold, and cannot
regress cross-cluster ghost rejection by construction (foreign-cluster
winners are still dropped exactly as today — only same-cluster
reassignment is new behavior). Both would ship as a new default-OFF boolean
parameter threaded through `clustering_points_segments`'s signature
(`clus/inc/WireCellClus/PRSegmentFunctions.h:548`) and its three call sites
(`NeutrinoTrackShowerSep.cxx:112,166`, `NeutrinoVertexFinder.cxx:600`),
byte-identical when off, per this repo's standard knob convention — same
bar as pr/64 round 4's validation process (fit against a labeled sample if
one exists or can be built for this class, byte-identical off-gate, on-arm
mover/energy census) before any flip. No labels exist yet for this specific
gap class; building a small hand-scan (a handful of events, comparing
`img-global` vs `shower_track-global` orphan blobs at PF-graph vertices) is
the natural first step of implementation, not skippable.

### Why this doesn't move Round 5's S6/energy conclusions

`kine_reco_Enu` and `kine_energy_particle` (Round 5) are computed from
`T_rec_charge`/segment-level charge sums, not from the `associate_points`/
`shower_track` display cloud — the 12-18 dropped points (a few percent of
segment 126042's own ~2600 associated points) have no measurable effect on
the quoted 717.24 MeV. This is a display/association-completeness defect,
not an energy-reconstruction one.

### Files

No new script this round — the probe was a short ad-hoc snippet (reproduced
in the Repro block above) rather than a reusable tool; `probe_18625.py`
(Round 5) remains the reusable one for the S6/energy/label checks.

## Round 7 (2026-08-11) — `assoc_reassign_orphans` implemented, validated, SBND PRODUCTION ON; the specific 18625 blob is NOT fixed by it (root cause is different, correcting Round 6)

**Status: Option B from Round 6 is implemented, demonstrated, validated on the
48-event nueCC sample, and flipped ON for SBND production** (owner
pre-authorization: "if validation passed, just turn on this knob"). **But**
digging into *why* this event's specific reported blob survives the fix
found that Round 6 named the wrong mechanism for it — corrected below.

### Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# unit test (clus/test/doctest_pattern_recognition.cxx, new TEST_CASE):
./build/clus/wcdoctest-clus -tc="pattern_recognition*assoc_reassign*"

# log-only census (channel classification, no behavior change):
WCT_PR64_ORPHAN_CENSUS=1 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr64r7-census18625 data 18625
grep "pr64 orphan-census tally" work-pr64r7-census18625/pr_evt18625/wct_pr_evt18625.log

# evt 18625 demonstration, off vs on:
PR_JOBS=1 ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr64r7-off18625 data 18625
PR_JOBS=1 SBND_ASSOC_REASSIGN_ORPHANS=true \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr64r7-on18625-v2 data 18625

# 48-event nueCC validation, off vs on (event list = work-nuecc48-cb0805's 48 ql_evt* dirs):
SBND_MAX_JOBS=6 PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr64r7-off48 data <48 evts>
SBND_MAX_JOBS=6 PR_JOBS=6 SBND_ASSOC_REASSIGN_ORPHANS=true \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr64r7-on48 data <48 evts>
diff work-pr64r7-off48/nusel-table.tsv work-pr64r7-on48/nusel-table.tsv   # empty
```

### Implementation

New default-OFF bool `reassign_orphans` (cfg key `assoc_reassign_orphans`),
threaded through `clustering_points_segments`'s signature
(`clus/inc/WireCellClus/PRSegmentFunctions.h:548`) exactly like Round 6's
"Option B" proposal, mirroring the existing `assoc_full_recluster` (pr/59)
knob's plumbing pattern end to end (`NeutrinoPatternBase.h`,
`TaggerCheckNeutrino.h`/`.cxx`, `cfg/pgrapher/common/clus.jsonnet`,
`cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`). Threaded through
**both** live call sites in `NeutrinoTrackShowerSep.cxx` — the main
`clustering_points` pass (`:112`) **and** the pr/59 `reassociate_cluster_orphans`
recluster path (`:166`), which is live in SBND production
(`assoc_full_recluster=true`) — so a rescued segment gets the same
association rule as the main pass, not a stale default.

Mechanism, in `clus/src/PRSegmentFunctions.cxx` (Stage C, `:2961-3076`): a
**second, independent pass** over the same graph vertices, run only when the
knob or the `WCT_PR64_ORPHAN_CENSUS` diagnostic is set — it never touches the
primary loop, so the knob-off path is byte-for-byte unchanged by construction
(M10). For every vertex the primary pass left unclaimed, it:
1. Recomputes the point's global per-plane 2D minimum via the existing F17
   `global_kd2d` KD-trees, this time also capturing **which segment** achieves
   it (a new parallel `global_kd2d_owner` table — the KD-tree's `knn(1,q)`
   already returns the winning flat index, previously discarded).
2. Restricts candidates to segments in the **same cluster** — a point whose
   true winner is in a different cluster stays dropped, so cross-cluster
   ghost rejection is unaffected by construction.
3. Re-checks the candidate against a **duplicated** copy of the Stage-C
   acceptance rule (`accept_for`, not shared with the primary chain — M10),
   and assigns the point to the first same-cluster candidate that passes it,
   in deterministic graph-index order.

### Unit test

New `TEST_CASE("pattern_recognition clustering_points assoc_reassign_orphans [A]")`
in `clus/test/doctest_pattern_recognition.cxx`, alongside the existing pr/59
`reassociate_cluster_orphans [A]` test it's modeled on. Checks two safety
properties against real fixture geometry (no manufactured orphan needed):
(1) knob off is a deterministic no-op — re-running `clustering_points`
explicitly off reproduces the exact baseline per-segment counts; (2) knob on
is monotonic-only-additive — every segment's `associate_points` count can
only stay the same or grow. On fixture [A] this round's diagnostic run
measured **7523 -> 8149 total associate_points (+626, +8.3%)** — the fix
does something material even on the small unit-test cluster.
`./build/clus/wcdoctest-clus`: **175/175 pass** (171 pre-existing/concurrent
+ 4 from an unrelated in-progress `doctest_nu_band_veto.cxx` a concurrent
session is adding to this shared tree, per M9 — not committed by this round).

### The census: confirms Round 6's *general* mechanism, corrects its claim about *this event's specific coordinate*

`WCT_PR64_ORPHAN_CENSUS` classifies every point the primary pass drops into
`no_terminal` (no Voronoi terminal reaches it) or `ghost_drop` (it had a
candidate but lost Stage C), and tallies whether a same-cluster winner
exists. On evt 18625, cluster 126 (the split shower cluster from Round 5):

```
pr64 orphan-census tally: cluster 126 no_terminal=0 ghost_drop=828 ghost_drop_same_cluster_winner=828 rescued=0
```

**All 828 ghost-dropped points in cluster 126 have a same-cluster winner** —
the general Round-6 mechanism (Stage C drops instead of reassigning) is real
and fires 828 times in this one cluster alone. This is what the knob fixes,
and does fix: evt 18625's `shower_track-global` point count goes
**6129 -> 6335 (+206 points)** on, cluster 126 drops from **13 -> 12**
orphans (proxy: clustering-global points with no `shower_track-global` point
within 2 cm), and `kine_reco_Enu` moves **1499.340 -> 1499.871 MeV**
(+0.531 MeV — small on this event specifically, consistent with Round 6's
"no measurable effect on this event" claim; see the 48-event numbers below
for the aggregate effect elsewhere).

**But none of the 828 rescued points, nor any of the remaining 12 orphans, are
within 8 cm of the owner-reported (142.1, 78.3, 176.5).** The literal blob
the owner pointed to in Bee is **not explained by Stage-C ghost removal at
all** — a direct correction to Round 6's claim. Tracing it (a targeted,
temporary per-point trace, not shipped — see Files) found:

- The 12 target-proximate points **are** claimed by the primary pass, at the
  cluster's single `clustering_points_segments` call, by segment with
  graph-index **47** (not 42, as Round 6 assumed from proximity alone).
- Segment 47 **does not exist** in the final PF output: `track_fit-global`
  has zero `real_cluster_id=126047` rows, and cluster 126 ends the pipeline
  with only **15** surviving segments (`126039,041,042,044-046,049,050,
  055-057,060-063`) — 47 is gone.
- `reassociate_cluster_orphans` (pr/59, live for cluster 126 —
  confirmed by an entry trace, `n_segments=15` at the time it runs) does
  **not** re-cluster cluster 126, because its trigger
  (`any_orphan`: does any *current* segment have zero `associate_points`
  points?) is false — all 15 surviving segments already have *some* points
  of their own. The check has no way to notice that a *different*, no-longer-
  existing segment's points were never redistributed.

So the actual mechanism for this specific blob is: **a segment legitimately
wins points during clustering, is later removed from the cluster's graph by
some downstream restructuring step (not yet identified precisely — between
the single clustering pass and the final PF/Bee dump), and nothing
re-associates its orphaned points to a surviving neighbor.** This is a
different bug from Round 6's Stage-C diagnosis, in a different place
(segment-lifecycle / re-association-trigger gap, not Voronoi-vs-2D
disagreement), and this round does **not** fix it.

### 48-event nueCC validation

Off-gate: **`nusel-table.tsv` byte-identical, 48/48 events** — the knob
never moves selection variables. On-arm mover census: `mabc-pr.zip` content
hash differs on **47/48** events (the 48th, evt116962, has no PF
reconstruction / no `T_kine` tree at all on either arm — genuinely
unaffected, not a gate miss). Every one of the 47 moved events'
`kine_reco_Enu` moves **up** (never down — expected, since the rescue only
ever adds a previously-dropped point, never removes one): mean **+10.7 MeV**,
max **+51.9 MeV** (evt 46363: 1987.5 -> 2039.4 MeV). Top movers:

| event | Enu before | Enu after | delta |
|---|---|---|---|
| 46363 | 1987.505 | 2039.386 | +51.881 |
| 90055 | 2588.634 | 2638.482 | +49.848 |
| 42280 | 2529.677 | 2578.654 | +48.977 |
| 214469 | 2364.883 | 2400.845 | +35.962 |
| 239794 | 2911.144 | 2939.670 | +28.526 |

This is a **systematic, one-directional correction** consistent with fixing
an under-count (points silently discarded by Stage C previously contributed
zero charge to any segment's dQ/dx sum; now they're counted by their
rightful segment), not noise — but it is a real, non-trivial shift in
reconstructed energy across nearly the whole sample, reported here in full
rather than summarized away.

### Bee links (owner-requested before/after, evt 18625 + top 3 movers)

- Before: https://www.phy.bnl.gov/twister/bee/set/18a01f59-73e0-4f6f-b3c9-1db8fddd1fbf/event/list/
- After: https://www.phy.bnl.gov/twister/bee/set/47257fd1-a79e-4fb8-b3fe-aebca9861927/event/list/
- bee_idx 0 = evt 18625 (the demonstration event; blob still visible, unresolved by this fix), 1 = evt 46363 (largest Enu mover, +51.9 MeV), 2 = evt 90055 (+49.8 MeV), 3 = evt 42280 (+49.0 MeV).

### Flip

Per owner pre-authorization, since the off-gate held cleanly (nusel
untouched, 48/48) and the on-arm behaved as designed (monotonic, no crashes,
no selection-variable movement): `assoc_reassign_orphans = true` in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`, threaded to
`clus.pr(...)`. C++ default stays `false`. Legacy escape:
`-A assoc_reassign_orphans=false` (or `SBND_ASSOC_REASSIGN_ORPHANS=false`).
Compile proof: `wcsonnet` on the bare production config with
`pipeline_names` including `tagger_check_neutrino` (the pr/64-memory-recorded
trap — a proof that doesn't instantiate the component is vacuous) shows the
key present exactly once (`"assoc_reassign_orphans" : true`) post-flip,
absent with the escape TLA, and the flipped bare compile is **byte-identical**
to the explicit-TLA config this round's validated `work-pr64r7-on48`/
`work-pr64r7-on18625-v2` arms actually used.

### Still open

- **The literal 18259-18625 blob at (142.1, 78.3, 176.5) remains unfixed.**
  Its root cause (a legitimately-won segment removed from the graph after
  clustering with no re-association pass) needs its own investigation round:
  which step removes segment 47 (candidates: `determine_main_vertex`,
  `improve_vertex`, deghosting, or a shower-absorption pass — not yet
  isolated), and whether the fix is broadening `reassociate_cluster_orphans`'s
  trigger (whole-cluster point-coverage check, not just per-segment
  zero-count) or something narrower at the removal site itself.
- The Round 5 open item (uploaded Bee zip's clustering-global layer merging
  cluster 11/126 where every rerun this round split them, same as Round 5)
  remains unexplained, unchanged by this round.

### Files

- `clus/src/PRSegmentFunctions.cxx`, `clus/inc/WireCellClus/PRSegmentFunctions.h`,
  `clus/inc/WireCellClus/NeutrinoPatternBase.h`,
  `clus/inc/WireCellClus/TaggerCheckNeutrino.h`, `clus/src/TaggerCheckNeutrino.cxx`,
  `clus/src/NeutrinoTrackShowerSep.cxx` — the knob implementation.
- `clus/test/doctest_pattern_recognition.cxx`,
  `clus/test/doctest_clus_knob_defaults.cxx` — new/updated tests.
- `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` — cfg plumbing + flip.
- `run_pr_chain_batch.sh` (`SBND_ASSOC_REASSIGN_ORPHANS` env-to-TLA wiring).
- No new analysis script this round (ad-hoc python + the existing
  `probe_18625.py`); the "still open" segment-lifecycle trace used a
  temporary, unshipped debug log (added, used, then reverted) rather than a
  committed diagnostic — worth building into a reusable probe if the
  follow-on round happens.

## Round 8 (2026-08-12) — root cause of the Round 7 "still open" 18625 blob
found and fixed: `assoc_clear_on_merge`, implemented + validated, DEFAULT NOT
SELECTED for SBND production (toolkit uncommitted this turn, wcp uncommitted
this turn)

### Repro block

```
# root-cause trace (temporary instrumentation, added/used/reverted this round):
cd sbnd_xin
setarch x86_64 -R env WCT_PR64_BBOX_TRACE=1 WCT_PR64_TARGET_GIDX=<freshly-found> \
  PR_JOBS=1 ./run_pr_chain_batch.sh work-ncpi0-cb0805 <tag> data 18625

# implementation validation:
PR_JOBS=1 ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr64r8-off18625 data 18625
PR_JOBS=1 SBND_ASSOC_CLEAR_ON_MERGE=true \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr64r8-on18625 data 18625
SBND_MAX_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr64r8-off48 data <48 evts>
SBND_MAX_JOBS=6 SBND_ASSOC_CLEAR_ON_MERGE=true \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr64r8-on48v2 data <48 evts>
```

### Root cause

Traced the full removal chain for cluster 126/evt 18625 with temporary,
env-gated, fully-reverted probes: a checkpoint scan across every stage of
`TaggerCheckNeutrino::visit()` (built into `detg_dump`), a bbox-targeted
trace inside `clustering_points_segments`' Stage-C loop
(`PRSegmentFunctions.cxx`), and per-site removal probes in
`NeutrinoStructureExaminer.cxx`. All built, run under `setarch x86_64 -R`
for a self-consistent single-process trace, then `git checkout --`
reverted — nothing from the trace itself shipped.

Note: commit `095df966` (doc pr/66 round 2, also SBND PRODUCTION ON) landed
between Round 7 and this investigation and changes every segment's global
`graph_index` for a given event — Round 7's "graph-idx 47" is not reusable
across sessions/commits and had to be rediscovered fresh via the bbox trace
(it happened to still be 47 this time, a coincidence).

**Confirmed chain** (cluster 126 here is an "other" beam-flash cluster, not
`main_cluster`, but the code path is identical for both):

1. `pattern_algos.clustering_points()` (the only call for this cluster) runs
   Stage-C ghost removal and legitimately assigns 33 `associate_points` to a
   short (6-wcpt) segment, including all 8 raw points inside the reported
   bbox — confirmed **kept**, not dropped, by the primary Stage-C loop this
   round. This contradicts Round 6/7's "ghost-drop" framing even further:
   under current production code this specific segment is never even a
   ghost-removal casualty.
2. `pattern_algos.determine_main_vertex()` unconditionally calls
   `examine_structure_final()`, which chains `_1`/`_1p`/`_2`/`_3`
   (`NeutrinoStructureExaminer.cxx`). `_1p` (~line 2891) fires when a
   cluster's main_vertex has exactly 2 connected, near-collinear (angle
   > 175°) segments and one is short (< 6 cm): it merges the short
   segment's `wcpts` into its neighbor, rebuilds ONLY the neighbor's "main"
   point cloud (`create_segment_point_cloud(..., "main")`), then
   `remove_segment` deletes the short segment outright. Confirmed by a
   removal-site probe: `site=final_1p:short_sg2_deleted cid=126 gidx=47
   nwcpts=6 nassoc=33` — the exact segment from step 1, with its full 33
   associate_points, deleted here.
3. **The bug**: `_1p` (and `_1`'s duplicate-segment-removal branch, and
   `_3`'s self-loop-removal + main-connector-removal branches — same
   pattern, not individually traced but fixed by construction below) never
   touched `associate_points` on either side of the merge. The deleted
   segment's associate_points (the actual charge/blob points) were simply
   discarded; the survivor's associate_points was left exactly as it was
   BEFORE the merge — stale, since it didn't cover the newly-absorbed
   geometry.
4. `pattern_algos.reassociate_cluster_orphans()` (pr/59, live in SBND prod
   via `assoc_full_recluster=true`) runs right after `determine_main_vertex`
   specifically to catch segments created/modified too late for the first
   `clustering_points` pass — but its trigger is `any_orphan`: ANY segment
   in the cluster has a NULL or EMPTY associate_points cloud. Here the
   survivor already had SOME points from its own earlier association, so
   `any_orphan` is false and the whole cluster's re-clustering never fired.
   This is why the 12-point blob (a subset of the 33 lost points, closest to
   the merge vertex) survived all the way to the final PF/Bee dump with no
   segment owning it.

This is unconditional, always-on production code — `examine_structure_final`
has no knob and always runs inside `determine_main_vertex` for every
cluster, main or other. It is not specific to pr/64's `assoc_reassign_orphans`
mechanism (rounds 6/7) at all; those rounds' "ghost-drop" framing was a red
herring for this specific event.

### Implementation: `assoc_clear_on_merge`

New default-OFF bool, threaded exactly like `assoc_reassign_orphans`:
`PatternAlgorithms::m_assoc_clear_on_merge` (`NeutrinoPatternBase.h`),
`TaggerCheckNeutrino::m_assoc_clear_on_merge` (config get/default_configuration/
push), `cfg/pgrapher/common/clus.jsonnet` key-suppression idiom,
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` (`clus_pr` + the exported `pr()`
wrapper), `wct-pr-perevt.jsonnet` (added as a genuine top-level function
parameter — its parameter list runs from line 43 to line ~1400, and
`assoc_reassign_orphans`/`assoc_full_recluster` are already parameters deep
inside it, not locals; a first attempt that only threaded the TLA plumbing in
`run_pr_chain_batch.sh` without adding the parameter itself failed every
event with `RUNTIME ERROR: function has no parameter assoc_clear_on_merge`,
caught and fixed before the real validation run), `run_pr_chain_batch.sh`
(`SBND_ASSOC_CLEAR_ON_MERGE` env-to-TLA wiring).

Mechanism: a new helper `pr64_clear_survivor_on_merge(enabled, loser,
survivor)` in `NeutrinoStructureExaminer.cxx`, called at every segment-delete
site inside `examine_structure_final_1`/`_1p`/`_3` (the duplicate-removal
branch, both short-merge branches, and `_3`'s self-loop + main-connector
deletions, using the segments extended toward the merge point as the
survivor set). When the segment being deleted has a non-empty
`associate_points` cloud, the designated survivor's `associate_points` is
cleared (set to null) — nothing else. This makes pr/59's existing
`any_orphan` trigger correctly see a gap and re-derive a real,
geometry-consistent Voronoi+ghost-removal association for the whole cluster
— no new competition logic, reuses the already-validated pr/59 machinery.
Byte-identical off by construction (the helper no-ops unless
`m_assoc_clear_on_merge` is true).

### Unit test

New `pattern_recognition determine_main_vertex assoc_clear_on_merge [A]` in
`doctest_pattern_recognition.cxx`, run against real fixture geometry with two
independently-built contexts (`determine_main_vertex` mutates the graph
structurally, so unlike `clustering_points` it cannot be safely re-run on the
same graph for an A/B comparison). Fires naturally on fixture [A] with no
manufactured scenario needed: `nsegs 16 -> 16, null-cloud segments 0 -> 2,
total associate_points 7516 -> 7291`. Safety property checked: turning the
knob on can only ever ADD null-associate_points segments relative to knob
off (`CHECK(n_null_on >= n_null_off)`) — the mechanism only clears a stale
cloud, never fabricates or removes real association data itself.
`wcdoctest-clus` 176/176 (was 175 + this new one; 1 concurrent-session
doctest still not committed here, unrelated — M9).

### Demonstration on evt 18625

Off-gate: `work-pr64r8-off18625`'s `mabc-pr.zip` is member-hash identical to
an independently-built current-HEAD baseline (my 8 changed files stashed,
rebuilt, run, hash compared, then unstashed) — proves the knob-off path is
truly inert on top of everything that landed since Round 7 (incl. pr/66),
not just inert relative to a stale Round 7 baseline.

Knob on (`work-pr64r8-on18625`): `shower_track-global` bbox points (the
reported blob) **0 -> 8** — all 8 recovered, matching the traced segment's
33-point associate_points cloud (of which 8 fall inside the reported bbox);
total `shower_track-global` points 6335 -> 6368 (+33, the whole segment's
cloud). `kine_reco_Enu` 1499.871 -> 1499.980 MeV (+0.108). **The literal
18259-18625 blob at (142.1,78.3,176.5) reported by the owner is now
resolved.**

### 48-event nueCC validation

`nusel-table.tsv` byte-identical 48/48 (selection untouched, sorted-diff
0 lines). `mabc-pr.zip` moves on **3/48** (54095, 174637, 389538) — a much
narrower footprint than Round 7's Stage-C mechanism (47/48), consistent with
this being a rarer trigger (a specific structural-merge geometry, not every
event's Stage-C competition). Every mover's `kine_reco_Enu` goes UP only:

| event | off (MeV) | on (MeV) | delta |
|---|---|---|---|
| 54095 | 2991.456 | 2995.833 | +4.377 |
| 174637 | 1017.618 | 1019.864 | +2.246 |
| 389538 | 1744.944 | 1745.546 | +0.602 |

Monotonic, small-magnitude, consistent with correcting an under-count (the
mechanism only ever clears a cloud that pr/59 then re-derives with
strictly-equal-or-more points, never fewer — same safety property the unit
test checks directly).

### Bee before/after (evt18625 + the 3 Enu movers)

before https://www.phy.bnl.gov/twister/bee/set/a5dc3258-abc5-4f4a-85bc-43087d0df806/event/list/,
after https://www.phy.bnl.gov/twister/bee/set/da10ab0c-e6a7-4e1e-bd4b-a7899c0cd9f7/event/list/
(bee_idx 0=18625 demo/now-resolved, 1=54095, 2=174637, 3=389538). Index:
`docs/pr/pr64r8-bee.index.txt`.

### Status: DEFAULT NOT SELECTED for SBND production

The C++ knob default stays `false`; `wct-pr-perevt.jsonnet` adds
`assoc_clear_on_merge` as a genuine top-level function parameter, default
`false` this round (unlike Round 7, no production flip was requested or
made). Override for A/B: `-A assoc_clear_on_merge=true` (or
`SBND_ASSOC_CLEAR_ON_MERGE=true`).

### Files

- `clus/src/NeutrinoStructureExaminer.cxx` — the fix (4 call sites) + the
  `pr64_clear_survivor_on_merge` helper.
- `clus/inc/WireCellClus/NeutrinoPatternBase.h`,
  `clus/inc/WireCellClus/TaggerCheckNeutrino.h`,
  `clus/src/TaggerCheckNeutrino.cxx` — the knob plumbing.
- `clus/test/doctest_pattern_recognition.cxx`,
  `clus/test/doctest_clus_knob_defaults.cxx` — new/updated tests.
- `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` — cfg plumbing (no flip).
- `run_pr_chain_batch.sh` (`SBND_ASSOC_CLEAR_ON_MERGE` env-to-TLA wiring).
- `docs/pr/pr64r8-bee.index.txt` — Bee set index.

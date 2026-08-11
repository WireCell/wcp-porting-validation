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

## Fix — none shipped

No code change. No production flip. `git status cfg/ clus/` shows nothing from this
round, including Round 3 (investigation + prototype script only, per explicit scope).

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

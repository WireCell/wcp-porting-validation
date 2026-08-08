# doc pr/51 — near-vertex PR graph robustness: duplicated corridors, charge-less bridges, micro-stubs (131357 / 268067 / 360535)

Investigation only — **no code is changed in this round**.  The deliverable
is the per-event root-cause analysis and the proposed fix design; the
implementation is a future session (owner instruction 2026-08-08).

## Repro

```
# arms (existing, read-only): work-pr49-off48d (pr/48-equivalent baseline),
#   work-pr50-snap48a (== flipped production, toolkit ba5bbe59)
cd wcp-porting-img/sbnd/sbnd_xin
python3 scripts/analysis/pr51/vtx_struct.py  work-pr50-snap48a 131357 6   # near-vertex segments+vertices
python3 scripts/analysis/pr51/fit_ghost.py   work-pr50-snap48a 268067 8   # fit-layer image support + dQ/dx
python3 scripts/analysis/pr51/seg_overlap.py work-pr50-snap48a 360535 15 1.4  # corridor-overlap matrix
python3 scripts/analysis/pr51/vtx_plot.py    work-pr50-snap48a 268067 25 out.png  # near-vertex scatter
# per-stage segment tables are already in every arm's pr_evt<ID>/stdout.log
# ("After first round of main cluster PR" .. "After shower clustering with NV",
#  print_segs_info: index, length cm, type, dir, pdg, mass, KE MeV, flags)
```

dQ/dx numbers below are the Bee display charge of `track_fit-global`
(`make_pr_bee.py` header: q = dQ ×0.1 − 1000); they are used only as ratios.
Single-MIP segments in these events read ~1600–3300; the C++ fix must use
the existing detector-scaled reference `m_mip_dqdx_median`
(NeutrinoVertexFinder.cxx:2241 idiom, 43000 e/cm at the uBooNE default).

## Symptom (owner hand-scan, 2026-08-08, screenshots in
`sbnd_xin/docs/pics/Screenshot 2026-08-08 at 2.4*.png`)

The pr/50 snap fixed the 172230 fit-miss class but three events remain
wrong **at the PR-structure level near the vertex** — not fit-trajectory
errors:

1. **18259-131357** — a 2-track vertex now displays 3 tracks (a stub
   appeared at the vertex).
2. **18255-268067** — should be a 3-track vertex with a connected shower;
   instead the reconstruction routes into the shower and the vertex region
   holds multiple overlapping tracks.
3. **18255-360535** — a 3-track vertex with an extra **ghost track**.

## Findings

### Which rounds introduced what

| evt | off48d (knob-off) vs snap48a (production) | class |
|---|---|---|
| 131357 | **differ** — pr/49 `fit_blob_coverage` partition reshuffle is the trigger | regression (pr/49) |
| 268067 | **byte-identical** — predates pr/49 entirely | long-standing PR weakness |
| 360535 | near-vertex structure identical (only far-field churn from pr/49) | long-standing PR weakness |

So this is one *symptom class* with two *origins*; the fix must be an
outcome-level graph audit, not another input-level tweak to the pr/49 knob
(that lesson is doc pr/50's four falsified input-level gates).

### 131357 — displaced vertex + carved stub (3 prongs from 2)

Baseline (off48d): main vertex at the image corner `(43.69,177.12,136.68)`,
exactly two prongs — shower trunk (rcid 12056) + track (12078, 17 cm).
One PR vertex within 6 cm.

Production (snap48a ≡ on48d; the snap does NOT fire here):

- Main vertex `(42.30,178.00,137.48)` — **1.83 cm up the shower arm** from
  the true corner (the pr/49 census shift).
- THREE vertices within 3.2 cm (extras at d=1.54 and d=3.18 cm), and a new
  third prong: **segment 70, a 1.56 cm 9-point Track stub** spanning main
  vertex → `(43.7,178.0,136.4)` ≈ the true corner.  Final PID calls it a
  2212 (proton) at 8.9 MeV.
- Stage attribution (stdout `print_segs_info` tables): segment 70 does
  **not** exist "After first round of main cluster PR" (max index 55); it
  exists by "After improve vertex".  I.e. `determine_main_vertex` picked
  the displaced candidate, then the improve_vertex/examine_structure
  machinery carved the leftover gap between the chosen vertex and the true
  corner into its own micro-segment.
- The final graph retains a q=0 vertex **0.77 cm from the true corner**
  (`(43.24,177.61,136.33)`) — the right answer survives in the graph, the
  display simply has one prong too many and the star on the wrong spot.
- Why the pr/50 snap correctly declines: the fitted trajectory FOLLOWS the
  image through the corner (stub + track cover it; no fit-miss ≥ 0.35 cm).
  The snap's G5 guard exists precisely to keep it out of this class — this
  is a *graph-shape* error, not a *fit-vs-image* error.
- Corridor overlap: stub 12070's 4 fit points lie 75% / 50% within 0.6 cm
  of the neighboring prongs 12029 / 12049 — it lives in the crotch of the
  vertex, not on its own charge.

### 268067 — near-vertex cycle through the shower + charge-less bridge

Byte-identical in off48d and production; all key segments already exist
"After first round of main cluster PR" (segment 5 is a 10.7 cm S_traj from
round 1).  Final near-vertex mini-graph (fit-layer endpoints, every end
lands exactly on a PR vertex):

```
MAIN (-77.47,-66.29,279.19)
 ├─ 15001  82 cm  med dQ/dx 4984   real proton track            [OK]
 ├─ 15050  0.5 cm med 12789        micro-stub at the vertex     [pathology c]
 └─ 15003  12.1 cm med 4902        → V_A (-86.52,-61.43,285.60)  [pathology a]
      V_A ─ 15008  9.9 cm med 8408 (Bragg)  real prong into the shower fan
      V_A ─ 15005  9.5 cm med 202  → V_B    charge-less bridge   [pathology b]
           V_B (-78.15,-65.27,283.17) ─ 15015  18 cm med 2696  middle track
```

- **15003 duplicates the proton corridor**: 86% of its fit points lie
  within 0.6 cm of 15001's fit points (13° opening angle); it "reaches"
  V_A — a junction sitting in the shower fan — by riding 12 cm of the
  proton's charge.  This is the owner's "the snap [reconstruction] just
  totally go to the shower".
- **15005 carries no charge** (median dQ/dx 202 ≈ 0.1 of the single-MIP
  band): a pure connectivity bridge across the gap between the shower fan
  and V_B.
- **V_B is 4.2 cm from MAIN but not connected to it** — the middle track
  15015 dangles off V_B and reaches the vertex only through the roundabout
  cycle MAIN→15003→V_A→15005→V_B.  The correct topology (owner) is
  15015 connected at/near MAIN directly: a 3-track vertex (15001, 15015,
  V_A-direction prong) with the shower attached beyond it.
- Net display: four trajectory ribbons + a stub inside a 5 cm ball around
  the vertex — "multiple tracks in the vertex region".

### 360535 — parallel duplicated connection = the ghost track

Near-vertex structure identical with the knob off.  Endpoints:

```
7060  73 pts  med 3282   MAIN (-185.72,-111.85,81.40) → far   real track 1
7067  13 pts  med  934   MAIN ↔ V2 (-184.01,-110.14,87.35)    ┐ parallel pair,
7020  15 pts  med 1219   V3 (-186.20,-110.49,80.18) ↔ V2      ┘ ~1 cm apart
7018  48 pts  med 9382   V2 → far (Bragg)                     real track 2
```

- 7067 and 7020 are **two nearly-parallel paths covering the same physical
  track trunk** between the vertex region and junction V2: 0% mutual
  overlap at 0.6 cm tolerance but 77–80% at 1.4 cm — separated by ~1 cm,
  i.e. two distinct fitted ribbons through one charge corridor.
- The dQ/dx association **splits the single track's charge between them**:
  934 + 1219 = 2153, squarely inside this event's single-MIP band
  (1602–3282) — each member reads roughly half-MIP.  Whichever ribbon lies
  off the true ridge is the owner's ghost; the split is why the ghost has
  the tell-tale low-dQ/dx (dark blue) look in Bee.
- V3 (1.89 cm from MAIN) is the same satellite-vertex pattern as 131357's
  displaced pair: the graph carries a triangle MAIN/V3/V2 where the truth
  is a single edge.

## Common mechanism

All three displays fail because the **near-vertex pattern graph** contains
structures the fitter then faithfully draws:

- (a) **duplicated corridors** — two segments over one charge ribbon
  (268067's 15003-on-15001; 360535's 7067/7020 parallel pair);
- (b) **charge-less bridges** — segments whose median dQ/dx is a small
  fraction of MIP, existing only to connect components (268067's 15005 at
  ~0.1 MIP);
- (c) **micro-stubs at the vertex** — sub-2 cm terminal fragments that
  inflate the prong count (131357's 12070 = 1.56 cm; 268067's 15050 =
  0.5 cm);
- (d) **wrong vertex among near-degenerate candidates** — the star sits on
  a displaced candidate while a candidate at the true corner survives
  0.77–1.9 cm away (131357; 360535's V3).

Every one of these is invisible to point-level fit metrics (fit-vs-image
means/maxes are clean — the ribbons DO follow charge), which is why the
pr/49/pr/50 census machinery scored these events as unremarkable and why
they surface only in hand-scans.  Conversely, every one of them is
**cheaply measurable post-fit**: corridor overlap fraction, median dQ/dx
vs `m_mip_dqdx_median`, terminal-segment length, and parallel-connection
detection are all a few lines each on `fits()` + the local graph.

Why existing passes cannot catch them:

- `vertex_kink_snap` (pr/50) is scoped by its fit-miss guard to the
  fit-detours-from-image class; all three events pass that guard honestly.
- `fit_blob_coverage_defer` (pr/50, OFF) only reverts pr/49-induced
  partition reshuffles — it would fix nothing in 268067/360535 (pre-pr/49)
  and in 131357 merely restores the old partition rather than making the
  selection robust.
- `improve_vertex` is a local position optimizer; it actively *creates*
  the 131357 stub while polishing the wrong candidate.
- `examine_structure_final_2/_3` merge at 2.0/2.5 cm but run before the
  main-vertex determination settles and do not look at charge sharing or
  corridor overlap at all.

## Proposed fix (future session): `main_vertex_graph_audit`

One new default-OFF pass in the same empty pipeline window the snap uses
(after `determine_overall_main_vertex[_DL]` — after the snap, before the
final `improve_vertex`), operating only on the main cluster's local graph
within `mvga_radius` (~15 cm) of the main vertex.  Four ordered
operations, each individually gated and sentinel-logged:

1. **Duplicate-corridor merge** (fixes 268067-15003, 360535-7067/7020):
   for each segment pair in scope, compute the fraction of the shorter
   segment's fit points within `mvga_dup_tol` (~1.2 cm — must be wider
   than the fitter's ribbon separation, cf. 360535's 1 cm) of the longer
   one's.  Above `mvga_dup_frac` (~0.7): delete the *worse* member (lower
   integrated charge; tie-break shorter length), then reconnect its
   orphaned far vertex to the survivor's nearest endpoint if they are not
   already the same vertex (direct edge via `crawl_segment`-style path
   rebuild).  Re-run the dQ/dx association afterward so the survivor
   recovers the full corridor charge (360535: one ribbon at full MIP
   instead of two at half).
2. **Charge-less-bridge removal** (fixes 268067-15005): a segment with
   median dQ/dx < `mvga_bridge_mip` (~0.33) × `m_mip_dqdx_median`, not
   Bragg-terminal, and length > stub scale is deleted *iff* the local
   graph stays connected or the disconnected side can be reconnected by a
   shorter direct edge to a vertex within `mvga_reconnect` (~5 cm).  In
   268067 this deletes 15005 and reconnects V_B (i.e. the middle track
   15015) directly toward MAIN — 4.2 cm, well inside the window — which
   simultaneously collapses the cycle and detaches the vertex from the
   shower fan.
3. **Micro-stub absorption + vertex re-seat** (fixes 131357-12070,
   268067-15050): a terminal segment at the main vertex shorter than
   `mvga_stub` (~2 cm) is absorbed into its most-collinear neighbor when
   its corridor-overlap with the neighbors exceeds `mvga_dup_frac`
   (131357: 75%).  When the stub is the vertex-ward continuation of a
   longer track (the absorbed pair spans the corner), the main vertex is
   re-seated at the stub's far end — for 131357 that is 0.9 mm from the
   true corner, restoring both the 2-prong count and the star position.
   `kProtectedBreak` vertices are exempt (snap precedent G1).
4. **One local `do_multi_tracking` refit** of the main cluster, exactly
   the snap's post-edit contract, so dQ/dx and the display layers reflect
   the audited graph.

Guard rails carried over from the pr/50 round: all thresholds are config
knobs with C++ defaults OFF (`mvga_*` = 0 disables each operation
independently); every edit prints one sentinel line with the measured
quantities (overlap fraction, dQ/dx ratio, stub length, re-seat distance)
so censuses stay mechanical; degree-2 pass-through vertices and
protected-break vertices are never deleted; the pass runs once, no
recursion.

### Expected per-event outcome (the acceptance test for the fix session)

| evt | operations firing | display outcome |
|---|---|---|
| 131357 | 3 (stub absorb + re-seat) | 2-prong vertex, star at the image corner (≤1 mm) |
| 268067 | 1 (15003 dedup), 2 (15005 removal + V_B reconnect), 3 (15050 absorb) | 3-track vertex (15001, 15015, V_A prong), shower attached beyond V_A, single ribbon per track |
| 360535 | 1 (7067/7020 dedup + reconnect) | single full-MIP connection MAIN↔V2, ghost gone |
| 172230 / 57441 / snap movers | none (no duplicate corridors, no sub-MIP bridges, no stubs) | byte-identical — to be proven in the off-gate + census |

### Open questions for the fix session

- 131357: was the true-corner candidate present at
  `determine_main_vertex` time or created later by improve_vertex?  Needs
  a `WCT_DET_DEBUG=1` probe (detg vertex tables at
  `main:determine_main_vertex`).  If present-but-outscored, a companion
  look at the scorer's near-degenerate handling may be warranted; the
  audit fixes the display either way.
- Whether operation 1's re-run of the dQ/dx association can reuse the
  existing `do_multi_tracking` wholesale (simplest, matches snap) or
  needs a narrower re-association to keep runtime negligible.
- Interaction ordering with the snap: audit runs *after* snap so a
  snapped corner is never re-judged as a stub (the snap's break is
  `kProtectedBreak`, which operation 3 already exempts).
- Whether operations 1+2 also help the defer-only events (342199,
  469665) — if yes, the audit may deliver most of
  `fit_blob_coverage_defer`'s benefit without its 57441 cost.  To be
  measured in the fix round's census.

## Verification (this round)

- No toolkit or config change was made; production (`vertex_kink_snap`
  ON, toolkit `ba5bbe59`) is untouched.
- All numbers above reproduce with the four committed scripts
  (`scripts/analysis/pr51/`) against the existing read-only arms
  `work-pr49-off48d` and `work-pr50-snap48a`; stage attribution reads the
  arms' `stdout.log` tables — no reruns were needed.

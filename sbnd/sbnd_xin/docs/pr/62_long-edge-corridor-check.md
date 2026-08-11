# doc pr/62 — S7: long-edge corridor connectivity (close the >=30cm blind spot)

**SBND PRODUCTION ON, owner flip 2026-08-11** (toolkit commit below). Operating
point `min_gapped_planes=1` (flavor `relaxed_strict_img_2d_rescue_long`) — see
"Owner flip" section near the end.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
M50=$(awk 'NR>1{print $2}' docs/pr/mcp1k-50-cb0805.index.txt)

# pristine-HEAD baselines (toolkit 55b234e5)
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr62-base48 data
PR_JOBS=32 ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr62-base19 data
PR_JOBS=32 ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr62-base50 data $M50

# ---- build the S7 change (toolkit commit below), wcbuild, wcdoctest-clus ----

# off-gate: new binary bare == pristine baseline
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr62-off48 data
PR_JOBS=32 ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr62-off19 data
PR_JOBS=32 ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr62-off50 data $M50
python3 scripts/analysis/pr49/on_compare.py work-pr62-base48 work-pr62-off48   # 0/48
python3 scripts/analysis/pr49/on_compare.py work-pr62-base19 work-pr62-off19   # 0/19
python3 scripts/analysis/pr49/on_compare.py work-pr62-base50 work-pr62-off50   # 0/50

# on-arms, both operating points
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue_long  PR_JOBS=32 \
    ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr62-on48 data     # + on19, on50
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue_long2 PR_JOBS=32 \
    ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr62-on2-48 data   # + on2-19, on2-50
python3 scripts/analysis/pr49/on_compare.py work-pr62-off48 work-pr62-on48     # movers

# causal-attribution census on the movers only (log-only)
MOV48="42280 52672 54095 90055 163543 196649 214469 239794 246579 256587 271851 433451 489330"
MOV19="142421 314838 399860 506114"
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue_long WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=16 \
    ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr62-cen48 data $MOV48
SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue_long WCT_RELAXED_EDGE_CENSUS=1 PR_JOBS=8 \
    ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr62-cen19 data $MOV19

# Bee before/after over the 17 movers
python3 scripts/bee/make_pr_bee.py -q work-nuecc48-cb0805 -q work-ncpi0-cb0805 \
    -p work-pr62-off48 -p work-pr62-off19 -o bee/pr62/pr62-before.zip $MOV48 $MOV19
python3 scripts/bee/make_pr_bee.py -q work-nuecc48-cb0805 -q work-ncpi0-cb0805 \
    -p work-pr62-on48  -p work-pr62-on19  -o bee/pr62/pr62-after.zip  $MOV48 $MOV19
./upload-to-bee.sh bee/pr62/pr62-before.zip
./upload-to-bee.sh bee/pr62/pr62-after.zip
```

## Symptom / motivating gap

`ClusteringProtectBundle` decides where a `Cluster` is cut, using the graph
flavor named by `protect_graph_name` (SBND production =
`relaxed_strict_img_2d_rescue` since toolkit `fce53e2c`, 2026-08-10).
Candidate edges between closely-components are filtered by layered kill
tests in `clus/src/connect_graph_relaxed_strict.cxx`:

| layer | predicate | distance range actually tested |
|---|---|---|
| S1-S3 | `Graphs::relaxed_strict_bad` | **all** distances |
| S5 | `Graphs::relaxed_img_bad` | `< 15 cm` (`dis_cap_cm`) |
| S6 | `two_d_gap_kill` -> `Graphs::two_d_connectivity_bad` | `< 30 cm` (`s6_dis_cap`) |

Above 30 cm the *only* surviving test is S1-S3, and S1-S3 has a known
defect: `test_good_point` (`Facade_Grouping.cxx:585-613`) checks U/V/W
independently via three per-plane 2D kd radius queries, never intersected
in 3D. Three planes can each see charge from a *different* nearby track and
every 1cm sample reads "good" with nothing physically there. S5 and S6
exist specifically to patch that; neither reaches past 30cm.

**Measured, evt 142421 cluster 7** (21 components, C(21,2)=210 candidate
pairs, arm `work-pr61-census142421`, doc pr/61):

```
210 pairs total
 26 survive S1-S3
 12 of those <30cm   -> S6 evaluates them
 14 of those >=30cm  -> NOTHING checked 2D contiguity before this change
    30.03 31.01 31.76 32.66 33.94 34.13 36.87 37.39 40.61 42.48 46.56 71.23 72.21 74.80
```

Three of those (30.03, 36.87, 37.39 cm) are the MST bridges that reconnect
a 17-point island to the rest of cluster 7 — the case the owner scanned in
Bee and asked about (doc pr/61). They carry per-plane 2D charge (S1-S3
passes them) but no 3D-contiguous support.

**Why `s6_dis_cap` cannot simply be raised.** S6's flood fill roams a 2D
*rectangle* whose linear size tracks `edge_dis` (`seed_radius = edge_dis +
2cm`). For a wide/dense object the fill explores *area* -> ~O(D^2). It is
bounded by `cell_budget = 20000`, but that breaker **fails closed**
(`s6_planes_connected` returns "gap" = kill on exhaustion). Raising the cap
trades missed gaps for silent false kills of large sparse real objects.

## Root cause / design

**S7** is a new, separate layer running only on the band S6 skips
(`edge_dis >= 30cm`), using a **corridor-restricted** search — cost O(D)
instead of O(D^2), so the cell budget stops being a live constraint.

- **Corridor anchored in the plane's own lattice, not a 3D projection.**
  `q1`/`q2` are cloud members, so their native lattice cells are available
  by the existing S6 seed construction (`cluster.wire_index(gi,plane)` +
  `blob->slice_index_min()`). The corridor per plane runs between those two
  cells — a capsule (band of half-width `hw=3.0` + endpoint disks of radius
  `cap=8.0`, lattice units). `Grouping::convert_3Dpoint_time_ch` exists and
  is the analytic inverse of the ctpc keying, but is a *different*
  implementation of the wire/tick map than the one that produced
  `cluster.wire_index` (disagree by +-1 wire at tie boundaries) — kept out
  of the verdict path entirely.
- **Fixed 2cm seed radius** (not `edge_dis`-scaled like S6's) — the change
  that removes the O(D^2) driver.
- **A required correctness fix, not a refinement: `s7_cell_passable`.**
  S6's exact-key `wire_charge_row` lookup is safe at its own <30cm range
  because its `edge_dis`-scaled seed radius pulls in enough blobs to span
  several residues mod `slice_step`. S7's fixed 2cm radius collapses each
  side toward a single blob, so an exact-key lookup would read a residue
  mismatch as "no charge" independent of whether charge is actually there —
  a silent, systematic false-kill on every long candidate. Fixed by probing
  a half-open window of exactly `slice_step` ticks centred on the cell
  (residue-blind; widens acceptance by <= half a slice, ~1.6mm).
- **The circuit breaker fails OPEN**, the opposite of S6's fail-closed
  posture. At S7's 30cm-and-up range, an exhausted budget on a large sparse
  real shower must abstain, not silently become a kill.
- **Kill rule**: `Graphs::long_corridor_bad` — >= `min_gapped_planes`
  non-excused, non-abstaining planes show a corridor gap; W never excused;
  U/V excused when quasi-parallel to that plane's wires (same test S6
  reuses). `min_gapped_planes` is a defaulted parameter, not hardcoded,
  because **this distance band carries no hand-scan labels** (unlike S6's
  899-label fit) — there was nothing to fit an operating point against, so
  two flavors ship instead of one guessed value:
  - `relaxed_strict_img_2d_rescue_long` — `min_gapped_planes=1` (mirrors
    S6's owner rule)
  - `relaxed_strict_img_2d_rescue_long2` — `min_gapped_planes=2`
    (conservative, S1-S5's "at least 2 views" convention)

Both flavors build on `relaxed_strict_img_2d_rescue` (today's production
flavor), so movers are directly attributable against the current operating
point. Plumbing: `Graphs::connect_graph_relaxed_strict` gained trailing
`bool long_check = false, int long_min_planes = 1`; call sites mirror S6's
in all three candidate blocks (closest, dir1, dir2), each guarded on
"not already killed" so S7 can only *add* kills, never rescue one. All four
`Facade_Cluster.cxx` dispatch chains register both new flavor names.
**No jsonnet change** — `SBND_PROTECT_GRAPH` passes straight through as
`--tla-str protect_graph_name=...` and jsonnet has no whitelist, so both
flavors are reachable for the A/B with production cfg completely untouched.

Instrumentation: `OC62CENSUS-S7` log lines, gated on the existing
`WCT_RELAXED_EDGE_CENSUS` env var (no new knob — a second gate risks a
half-populated census across an A/B). One self-contained line per
candidate carrying per-plane `has`/`gap`/`budget_hit`/`gap_cm`, the corridor
operating point, the excusals, and the verdict.

## Verification

**Off-gate PASS**, new binary bare == pristine baseline:

```
work-pr62-base48 vs work-pr62-off48: ARCHIVE-LEVEL 0/48, nusel 0/48
work-pr62-base19 vs work-pr62-off19: ARCHIVE-LEVEL 0/19, nusel 0/19
work-pr62-base50 vs work-pr62-off50: ARCHIVE-LEVEL 0/50, nusel 0/50
```

**`./build/clus/wcdoctest-clus`**: 161/161 test cases, 1730/1730 assertions
(new `doctest_long_corridor.cxx`: 8/8 cases, 110/110 assertions — seam at
30cm incl. the measured 30.03cm pair; fails-OPEN on `budget_hit` (a named
regression guard against "harmonizing" it with S6's fail-closed breaker);
no-seed-plane abstention; W never excused; `min_gapped_planes` strictness;
monotonicity incl. `budget_hit`/`!has_plane` only ever removing a kill;
`gap_floor_cm` inactive at its shipped default).

**Smoke test, evt 142421** (`SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue_long
WCT_RELAXED_EDGE_CENSUS=1`, arm `work-pr62-smoke142421`): all three known
island-reconnecting bridges killed —

```
j=0  k=16 dis=30.03cm gap=[true,true,false]  killed=true
j=11 k=16 dis=37.39cm gap=[true,true,true]   killed=true
j=13 k=16 dis=36.87cm gap=[true,true,false]  killed=true
```
13 of 14 S7-evaluated candidates in that cluster killed at `min_gapped_planes=1`.

**Movers over 117 events, both operating points give the IDENTICAL mover
set:**

```
nueCC48 (off48 vs on48 / on2-48):  ARCHIVE 13/48, nusel 0/48
ncpi0-19 (off19 vs on19 / on2-19): ARCHIVE  4/19, nusel 0/19
PR-data50 (off50 vs on50 / on2-50): ARCHIVE  0/50, nusel 0/50
```

**Correction (doc pr/63):** this doc originally attributed the identical
mover sets to "S7 always fires with >=2 non-excused gapped planes in this
sample, not just exactly 1" — that claim is **false**. The `OC62CENSUS-S7`
census shows 49 of the 184 S7 kills fire with exactly 1 voting plane. The
event-level mover *sets* are still identical between `min_gapped_planes=1`
and `=2` (confirmed by directly comparing the two on-arms, not done here
originally: `work-pr62-on48` vs `work-pr62-on2-48` ARCHIVE 1/48 differ,
nusel 0/48; `on19` vs `on2-19` ARCHIVE 0/19, nusel 0/19) — so the flip below
is unaffected — but the *reason* given was wrong. No re-validation of the
flip itself was needed since nusel is unchanged both ways.
17/117 events move; **nusel (final neutrino-selection scores/tags) is
byte-identical in every event, both operating points** — only the
`clustering-global`/`shower_track-global`/`track_fit-global`/`mc` display
layers differ.

Movers: nueCC48 = `42280 52672 54095 90055 163543 196649 214469 239794
246579 256587 271851 433451 489330`; ncpi0-19 = `142421 314838 399860
506114` (index at `docs/pr/pr62-bee.index.txt`).

**Census log-only proof**: `WCT_RELAXED_EDGE_CENSUS=1` rerun on all 17
movers (`work-pr62-cen48`, `work-pr62-cen19`) hashes member-identical to
the non-census `on48`/`on19` arms — `0/17` events differ.

**S7 kill-rate census, the 17 movers**: **189 candidates evaluated, 184
killed (97.4%)**. Per-event breakdown ranges 93-100% (evt 142421 13/14 =
93%, matching the smoke test above; all other 16 events 100%). This is
consistent with — and higher than — doc pr/56 sec 8.4's finding that S6's
own kill rate already rises with distance (53% at 0-1cm to 89% at
15-30cm): **S7 should be read as an aggressive operating point at
`min_gapped_planes=1`, not a conservative one.**

**Fragmentation delta** (`ClusteringProtectBundle`'s own `retained N + M
fragment(s)` log line, off vs on, every affected cluster — one cluster per
mover event, both operating points identical):

```
evt   42280 cid 8:  1750 blobs, retained 1705->1654 (+3 frag)
evt   52672 cid 9:   600 blobs, retained  481->477  (+1 frag)
evt   54095 cid 17: 3105 blobs, retained 2987->2964 (+5 frag)
evt   90055 cid 11: 2119 blobs, retained 1878->1847 (+3 frag)
evt  163543 cid 14:  898 blobs, retained  808->752  (+1 frag)
evt  196649 cid 11: 1765 blobs, retained 1758->1703 (+1 frag)
evt  214469 cid 16: 1750 blobs, retained 1328->1306 (+1 frag)
evt  239794 cid 2:  1930 blobs, retained 1781->1776 (+1 frag)
evt  246579 cid 19: 1387 blobs, retained 1290->1254 (+1 frag)
evt  256587 cid 11: 4244 blobs, retained 4099->4013 (+7 frag)
evt  271851 cid 23: 1108 blobs, retained  837->801  (+4 frag)
evt  433451 cid 4:  1742 blobs, retained 1649->1340 (+1 frag)  <- largest single move
evt  489330 cid 4:  1223 blobs, retained 1182->755  (+1 frag)  <- largest single move
evt  142421 cid 7:  1781 blobs, retained 1468->1351 (+3 frag)
evt  314838 cid 13:  659 blobs, retained  310->164  (+1 frag)  <- already heavily fragmented pre-change
evt  399860 cid 17:  800 blobs, retained  417->386  (+1 frag)
evt  506114 cid 19: 1510 blobs, retained 1304->1268 (+1 frag)
```

No blobs are lost (`nblobs` unchanged everywhere) — mass moves from the
main body into fragment(s). Three events stand out for the *size* of the
single move rather than the fragment count: **433451** (-309 blobs, ~18% of
the cluster), **489330** (-427 blobs, ~35% of the cluster), and **314838**
(retained drops from 47% to 25% of an already-small 659-blob cluster,
already at 8 fragments pre-change). These three specifically deserve visual
inspection before any flip decision — the Bee links above make that
comparison directly (before = `work-pr62-off*`, after = `work-pr62-on*`).

## Fix — what shipped, what did not (round 1, DEFAULT NOT SELECTED)

**Shipped, DEFAULT NOT SELECTED**: the S7 corridor check, the two new
flavors, the pure predicate + doctest, the census instrumentation. SBND
production cfg (`protect_graph_name = 'relaxed_strict_img_2d_rescue'`) was
completely untouched at this point — verified by `git status cfg/` showing
zero diff. *(Superseded below — round 2 flips production.)*

**Not shipped / explicitly out of scope this round**:
- No production flip yet. The 97.4% kill rate and the three large
  single-cluster moves above are reasons for caution, not reasons to
  withhold the measurement — this doc's job was to hand the owner the Bee
  before/after and the numbers first, not to decide unilaterally.
- `min_gapped_planes` is unfitted (no labels exist in this band); the doc
  reports both `=1` and `=2` because they turned out identical on this
  sample, not because either is validated as correct.
- `gap_floor_cm` (censused, in `S7CorridorInput::gap_cm`) ships inactive
  (default 0.0) — no scan yet justifies a nonzero value.
- No `hwmin`/`wres`/`tres` fitting instrumentation (corridor half-width
  sweep, cross-check against `convert_3Dpoint_time_ch`) — scoped out of
  this round to keep the change reviewable; the census fields that exist
  (`gap_cm`, `budget_hit`, the corridor operating point echoed per line)
  are sufficient to causally attribute every mover above, which was the
  round's actual deliverable. Follow-on work if the operating point needs
  refitting.

## Owner flip, 2026-08-11 — SBND PRODUCTION ON

Owner reviewed the 17-event Bee before/after (built from the
`min_gapped_planes=1` arm, `work-pr62-on48`/`work-pr62-on19` — see Repro
block) and instructed the flip to production default at that operating
point, with the `=2` flavor kept as a legacy escape rather than default.

**Change**: `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`,
`protect_graph_name` default `'relaxed_strict_img_2d_rescue'` ->
`'relaxed_strict_img_2d_rescue_long'`. No other file changed; the round-1
C++/doctest is untouched, this is a cfg-only flip in the same file/line the
doc pr/57 round-6 flip used (doc 68: SBND operating point lives only in
this file).

**Compiled-config proof** (M6; `wcsonnet`, no other TLAs — `graph_name` is
the actual JSON key `protect_graph_name` threads to):

```
pre-edit bare:                                 graph_name = relaxed_strict_img_2d_rescue
post-edit bare (no override):                  graph_name = relaxed_strict_img_2d_rescue_long
post-edit + -A protect_graph_name=..._rescue:  graph_name = relaxed_strict_img_2d_rescue   (0-line diff vs pre-edit bare)
pre-edit + -A protect_graph_name=..._long:     byte-identical to post-edit bare (0-line diff)
```

So the flip is exactly the explicit override already validated in the
117-event on-arms above (`work-pr62-on48/19/50`) — a bare run of the new
production cfg reproduces those on-arms' compiled config byte-for-byte, and
the legacy escape (`-A protect_graph_name=relaxed_strict_img_2d_rescue`,
env `SBND_PROTECT_GRAPH=relaxed_strict_img_2d_rescue`) reproduces the
pre-flip production graph byte-for-byte. No new A/B run needed: the on-arms
already run above *are* the production-flip validation, since the flip
changes no code, only which pre-validated flavor name the bare default
points at.

**Carried forward from round 1, now describing production behavior**:
17/117 movers, nusel byte-identical everywhere, 97.4% S7 kill rate on
evaluated candidates (aggressive — see the round-1 verification section
above), fragmentation deltas quoted above including the three flagged
large-magnitude moves (433451, 489330, 314838). These numbers do not change
with the flip; the flip only changes which flavor a bare production run
selects by default. The `=2` conservative flavor and the pre-pr/62 graph
both remain one `-A protect_graph_name=...` away (legacy escapes above).

`min_gapped_planes`/`gap_floor_cm` remain UNFITTED against hand-scan labels
(round-1 caveat, unchanged by the flip) — this band has no equivalent of
S6's 899-label fit yet. A follow-on scan in this style (pr/57 round 6's
path) is the natural next round if the operating point needs refitting
against real labels rather than the owner's visual Bee review.

## Cross-links

- [[project_pr57_separation_scan]] — S6's own operating point, the
  `oc56_truth`/`oc56_fit` hand-scan machinery this band lacks
- [[project_pr53_overclustering_investigation]] — S5's `dis_cap_cm`
  precedent for a distance-based test needing an explicit boundary
- doc pr/61 — the owner's original "phantom trajectory" report that traced
  to this exact untested band for evt 142421 (diagnosis only, no fix); this
  doc is the fix for that band
- doc pr/56 sec 8.4 — the S6 kill-rate-vs-distance census this doc's own
  97.4% number is read against

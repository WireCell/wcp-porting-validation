# 30 — Event 298595, cluster 86: why the STM-fit point reads "cluster 0" in Bee, and why `track_fit_global` looks like a different event there

## Repro block

```
# work dir (fresh v450-production smoke-test run, doc pdvd/29):
W=/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_2_v450

# Bee dump: stm_fit / track_fit / clustering(raw) / vertices global layers
unzip -j $W/mabc-pr.zip \
  "data/0/0-stm_fit-global.json" "data/0/0-track_fit-global.json" \
  "data/0/0-clustering-global.json" "data/0/0-vertices-global.json" -d <outdir>

# PR log (verdict + fit provenance lines quoted below)
grep -n "cluster 86" $W/wct_pr_039252_2.log

# calib dump (per-candidate main_vertex / kine)
python3 -c "import json; d=json.load(open('$W/calib-pr-evt298595.json')); print(d['candidates'])"

# Q1 fix verification: rerun PR under a fresh tag (pctree copied read-only,
# M13 never overwrites _v450) with the fixed clus/src/MultiAlgBlobClustering.cxx
# installed, then hash-compare mabc-pr.zip member-by-member against _v450.
W2=/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_2_stmrcidfix
cd /home/xqian/toolkit-dev/wcp-porting-img
diff <(python3 abtest/hash_archive.py --members $W/mabc-pr.zip) \
     <(python3 abtest/hash_archive.py --members $W2/mabc-pr.zip)

# Q2 mechanism trace: rerun with the WCT_DQDX_DROP_DEBUG instrumentation
# (toolkit e3dee831 + f622161e + ed035408) to attribute form_map_graph's
# zero-charge point drop to a specific segment, check its pre-drop shape,
# and trace curr_pts's size/source at organize_segments_path_3rd's entry.
W3=/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_2_currptsprobe
mkdir -p "$W3" && cp $W/pctree-evt298595.tar.gz $W/pctree-evt298595.tlas "$W3"/
cd /home/xqian/toolkit-dev/wcp-porting-img/pdvd
WCT_DQDX_DROP_DEBUG=1 ./run_pr_evt.sh -s currptsprobe 39252 2
grep "gi=2 cluster=86" "$W3/wct_pr_039252_2.log"
```

Run 39252, event 298595, cluster 86 (production defaults, doc pdvd/29: E=450 V/cm,
DL/DT=4.1307/7.9135, wires v7-uvwfit).

## Question 1: point (306.5, -243.4, 102.0) reads `real_cluster_id = 0` in Bee — why?

**Not a per-point or per-event anomaly.** The queried point matches
`stm_fit-global` index 4507 almost exactly (306.551, -243.426, 101.918 —
0.1 mm off), with `cluster_id = 86` but `real_cluster_id = 0`. Checking
*every* point in this event's `stm_fit-global` layer (7308 points, 22
distinct `cluster_id` values) shows `real_cluster_id == 0` for **all** of
them — not just cluster 86, not just this point.

Root cause is in the writer, `MultiAlgBlobClustering::fill_bee_points_from_cluster`
(`clus/src/MultiAlgBlobClustering.cxx:2859-2876`), the `pcname == "stm_fit"`
branch:

```cxx
for (size_t i = 0; i < fx.size(); ++i) {
    bpts.append(Point(fx[i], fy[i], fz[i]), fdQ[i]*dQdx_scale + dQdx_offset, clid, 0);
}
```

`Bee::Points::append(point, q, clid, real_clid)` (`util/inc/WireCellUtil/Bee.h:127`)
is called with the literal `0` as the fourth argument for every STM-fit point,
regardless of `clid`. The genuine `cluster_id` (86) IS written correctly and
is what the raw (unfit) `clustering-global` layer also reports at this exact
coordinate (121 raw points within 10 cm, all `cluster_id = 86`, nearest raw
point 0.42 cm away) — so "cluster 86" is the real answer; "0" is a Bee
display-field the `stm_fit` writer never populates. Compare: the `track_fit`
branch of the same function (general scoped-view path, further down in the
same function) *does* propagate a real `real_cluster_id` when the cluster
carries a `"real_cluster_id"/"perblob"` pcarray (see cluster 86's
`track_fit-global` points below: `86002`, `86003`, `86005`, `86009`,
`86010`, `-1` — genuinely varying). The `stm_fit` branch is the only one of
the three (`stm_fit` / `steiner_pc` / everything-else) that hard-codes it.

This is a **display-layer gap in the diagnostic `save_stm_fit` dump**, not a
reconstruction defect — `save_stm_fit` is default OFF in production (doc
pdvd/25 sec 7b) and exists to let a human inspect *why* TaggerCheckSTM tagged
a cluster, so nothing downstream consumes `stm_fit`'s `real_cluster_id`.

### Fixed (toolkit `ab0762c6`)

```cxx
// before
bpts.append(Point(fx[i], fy[i], fz[i]), fdQ[i]*dQdx_scale + dQdx_offset, clid, 0);
// after
bpts.append(Point(fx[i], fy[i], fz[i]), fdQ[i]*dQdx_scale + dQdx_offset, clid, clid);
```

`clid` matches the same backward-compatible fallback the general scoped-view
branch already documents ("When absent ... the Bee real_cluster_id stays ==
clid"), so `stm_fit` now follows the same convention instead of a value that
reads in Bee as "genuinely cluster 0" and can collide with any cluster whose
real id actually is 0.

Verification (per CLAUDE.md §4's C++-change bar):

* `./build/clus/wcdoctest-clus`: 264/264 test cases, 2887/2887 assertions
  passed (rc=0).
* Freshness proof: `local/lib/libWireCellClus.so` mtime postdates the source
  edit (`wcbuild`, both build+install rc=0).
* Reran the PR stage for this same event under a fresh tag
  (`work/039252_2_stmrcidfix`, pctree copied read-only from `_v450`, M13) and
  hash-compared `mabc-pr.zip` against the pre-fix `_v450` run with
  `abtest/hash_archive.py --members`: **20 of 21 members byte-identical**;
  the only member that differs is `data/0/0-stm_fit-global.json`, and the
  only key that differs inside it is `real_cluster_id` (now `== cluster_id`
  for every point, confirmed for the queried point: `cluster_id=86
  real_cluster_id=86`). No other Bee layer, the calib dump, or either ROOT
  file changed.
* This is unconditionally safe for production: the branch's loop body only
  ever executes when `save_stm_fit=true` and the cluster's `stm_fit` PC is
  non-empty; with `save_stm_fit=false` (the production default) the change
  is dead code, so every existing config stays byte-identical.

## Question 2: this point looks like an STM, but `track_fit_global` shows something else there, and its Michel trajectory doesn't match

### The two layers are two independently-fit, independently-fired algorithms

`stm_fit-global` and `track_fit-global` are Bee dumps of **two different
grouping-level `TrackFitting` slots**, filled by **two different
components**, dumped at **two different points** in the PR pipeline
(`cfg/pgrapher/experiment/protodunevd/pr.jsonnet`'s `bee_points_sets`):

| Bee layer | slot | filled by | dumped after (`visitor:`) |
|---|---|---|---|
| `stm_fit` | `"stm"` (named) | `TaggerCheckSTM`'s own internal trial fit (`m_track_fitter`), `clus/src/TaggerCheckSTM.cxx:614-634` | `TaggerCheckSTM:pr` |
| `track_fit` | default/unnamed | `TaggerCheckNeutrino`'s PR-graph fit for its own candidate | `TaggerCheckNeutrino:pr` |

Nothing in the pipeline requires (or checks) that TaggerCheckNeutrino's PR
graph, built independently afterward, reproduces the trajectory
TaggerCheckSTM used to justify its STM tag.

### What each one actually did with cluster 86 (log evidence, `wct_pr_039252_2.log`)

```
component_extreme_wcps: cluster 86 3 component(s), 1 above 10.0 cm -> 6 extreme group(s)
check_tgm: cluster 86 CASE-B pair (1,5) rejected: rescued end, straight chord 57.7 cm has an unsupported run > 30.0 cm
visit: TaggerCheckTGM: cluster 86 -> TGM=false
check_other_tracks: cluster 86 seg 1/1: len=3.5cm medQ=0.38MIP lenThr=0.0cm straight=0.989 front=(292.8,-206.4,45.3)cm
visit: TaggerCheckSTM: cluster 86 -> STM=1 TGM=0
persist_stm_fit: cluster 86 stmfit pass=0 status=0 kink=97 exit_L=61.8 left_L=28.2 npts=141
...
TaggerCheckNeutrino: [nu_per_bundle] gid 192: candidate main cluster 86 (t0 6484.400 us, L 70.6 cm, 0 associated)
dual_chain: OFF pass (snap) vertex=true (290.69,-231.98,80.53) cm cluster 86 5 candidates, 0 voxels
sgp guard: cluster 86 VETO maxsep=4.119 cap=3.000 n_gap=135 n_base=128 detour=6.446 base_cm=129.831
  first=(283.45,-228.11,93.75) last=(343.28,-253.19,92.73)     <- repeats several times, same pair
TaggerCheckNeutrino: [nu_per_bundle] ROW 1 gid 192 cluster 86 vertex (293.0316,-205.2276,45.2573) cm Enu 909.5364 acts 1
```

Cluster 86's raw (pre-fit) 3D imaging footprint (`clustering-global`, 598
points) is a single connected but **branchy** object: 3 components / 6
"extreme" (far) points, bounding box x=[281.7,343.3] y=[-253.8,-205.4]
z=[44.8,104.5], with a dense (~200-point) cluster right at x=[282,285] —
a junction — and two "arms" reaching away from it: one toward
(344.0,-252.9,92.8), the other toward (293.0,-205.2,45.3) — a very
different direction (drops 48 cm in z).

* **TaggerCheckSTM** treated the junction as the muon-stop / Michel-decay
  point and fit a single, smooth, densely-sampled (141 pts, ~0.4 cm
  spacing, `q` ~2000-4000, MIP-like, no obvious Bragg rise) trajectory
  along the "toward-344" arm. The queried point (306.5,-243.4,102.0) sits
  mid-way along that arm, ~25 cm past the junction. This is the fit that
  produced `STM=1`. Its own second-track check (`check_other_tracks`) saw
  only a short, low-charge 3.5 cm stub near the *other* arm's end
  (292.8,-206.4,45.3) and did not treat it as competing with the main fit.
* **TaggerCheckNeutrino**, run afterward as an independent candidate search
  (`gid 192`, `nu_index 1` in the calib dump), explored a vertex right at
  the same junction (`dual_chain` snap, 290.7,-232.0,80.5 — matches STM's
  anchor to within a few cm) but ended up **selecting the opposite arm's
  end** as this candidate's `main_vertex`: (293.03,-205.23,45.26) — exactly
  matching `check_other_tracks`'s "irrelevant" 3.5 cm stub and matching
  vertex marker `86003` in `vertices-global`. Its own Steiner-graph path
  builder logged repeated `sgp guard: ... VETO` lines while trying to
  connect the junction to the "toward-344" endpoint (`maxsep=4.119 cm`
  against a `3.000 cm` cap, `n_gap=135` of `n_base=128` edges flagged gap)
  — i.e. it repeatedly found that route badly supported by its own graph
  and fell back to a different-flavor path rather than accept it outright.
* **TaggerCheckTGM**, a third, independent algorithm, separately flagged
  a straight chord near this same cluster as having "an unsupported run
  > 30.0 cm" and rejected a through-going-muon classification on that
  basis.

Net effect on the Bee dumps: `track_fit-global`'s allocation for cluster 86
is only 41 points (vs. 293 for the event's actual selected main-vertex
cluster, 40) — nearly all of them packed into the 2 cm-wide junction region,
plus a handful of isolated markers at the two arm extremities. **None are
within 20 cm of the queried point**, and the low-z arm end (the "Michel-like"
short stub) that *does* survive into `track_fit_global` sits ~63 cm away
from, and in a visually unrelated direction from, the junction/kink that
`stm_fit`'s trajectory treats as the decay point. That mismatch is exactly
what looks like "the Michel trajectory in track_fit_global is not consistent
with the STM fit."

### Follow-up: why is the long leg *missing*, not just sparse?

The "41 points" figure hides that the shortfall is concentrated in exactly
one place. Grouping `track_fit-global`'s 41 cluster-86 points by their
(now-fixed, still segment-encoded) id and measuring each group's x-span:

| segment | n points | x-span |
|---|---|---|
| 86002 | 2 | **59.9 cm** |
| 86003 | 2 | 9.2 cm |
| 86005 | 2 | 8.8 cm |
| 86009 | 2 | 0.4 cm |
| 86010 | 27 | 1.8 cm |
| -1 (unassociated) | 6 | 60.6 cm |

Segments 86003/86005/86009 are legitimately short (< 12 cm) — 2 points
(their own endpoints) is the *correct*, by-design output for a short segment
(see below). Segment 86010 is the densely-refit junction/vertex region.
**Segment 86002 is the one genuine anomaly**: at 59.9 cm it is squarely the
"toward-344" arm — the queried point's leg — and by construction it should
not have only 2 points.

I initially found that `TrackFitting::organize_segments_path_3rd`
(`clus/src/TrackFitting.cxx:1522-1671`, called from `do_multiple_tracking`
at `step_size = 0.6 cm`, not the 12 cm `low_dis_limit` I first assumed --
that value governs an earlier pass) SHOULD interpolate roughly
`round(dis1/step_size)` evenly-spaced points along any segment whose
endpoints are farther apart than `step_size`: for a 59.9 cm segment at
0.6 cm that is ~100 points, not 2. That ruled out "the resampler never ran"
as the explanation on its own and pointed at a later pruning stage.

**Confirmed by instrumentation** (toolkit `e3dee831`, doc pdvd/30
investigation continued): `do_multiple_tracking` already counts, in
aggregate, how many trajectory points its pre-dQ/dx `form_map_graph` call
drops for having zero measured charge in every 2D plane projection (an
existing, always-on DEBUG line: "pre-dQ/dx form_map_graph dropped N of M
trajectory point(s) with zero plane quantity") -- but only as an event-wide
total, with no way to attribute a drop to a specific segment. Added an
env-gated (`WCT_DQDX_DROP_DEBUG=1`, no config knob, no behavior change --
verified byte-identical output with and without it set) per-segment
before/after count and reran:

```
do_multi_tracking: WCT_DQDX_DROP_DEBUG segment gi=2 cluster=86 zero-quantity drop 108 -> 2 points
```

This is segment 86002 (`gi=2` = graph index 2, `cluster=86`) -- exactly the
59.9 cm "toward-344" leg. Of its ~108 interpolated points, **106 (98%) are
dropped for having zero charge in every plane**, leaving only its two
vertex endpoints, which is exactly what reaches the Bee `track_fit` dump.
This is the direct mechanism: the resampler *did* run and *did* produce a
dense trajectory; the charge-based point filter then removed nearly all of
it because those particular 3D coordinates carry no measured charge.

**Is that a straight-line artifact, or a real charge desert?** Extended the
instrumentation once more (toolkit `f622161e`) to report, for the pre-drop
point list, the perpendicular distance of every point from the straight
line through the segment's own first and last point -- distinguishing "the
resampler carried forward the segment's real (bent) shape and there's
genuinely no charge along it" from "the resampler laid a straight chord
across a bend and of course found nothing there". Result for segment 86002:

```
do_multi_tracking: WCT_DQDX_DROP_DEBUG segment gi=2 cluster=86 pre-drop shape: n=108 chord=64.18cm max_perp_dev=0.000cm mean_perp_dev=0.000cm
```

**`max_perp_dev = 0.000 cm`, for all 108 points, every one of five separate
fit passes.** This is not a real charge desert along the cluster's actual
shape -- it is a dead straight line, confirming
`organize_segments_path_3rd`'s degenerate-input fallback
(`clus/src/TrackFitting.cxx:1560-1567`, taken when a segment's
carried-forward point list has collapsed to just its two vertices) ran for
this segment. And that in turn explains itself once combined with the
NeutrinoID Steiner-graph evidence already in hand: the `sgp guard` log line
for this same general junction<->far-end connection reported
`detour=6.446` -- the real, graph-connectivity-supported path between
those two points is **6.4 times longer** than the straight-line chord
(64.18 cm chord here vs. the graph's own preferred route). A straight line
laid across a connection that actually detours 6.4x will, unsurprisingly,
spend nearly all of its length in real 3D space where the true (bent)
598-point raw cluster is *not* -- so `form_map_graph`'s per-point charge
lookup correctly finds nothing there and drops it. `TaggerCheckSTM`'s own
fit avoided this entirely: it evidently traced the actual bent point cloud
(141 points, all with real charge 2000-6000-ish) rather than interpolating
a chord between two endpoints.

So the full chain, now traced end to end and confirmed by data at each
link rather than inferred: cluster 86's junction-to-far-end connection is
real but bent (not straight); NeutrinoID's own Steiner path builder already
knew this (`detour=6.446`) but that knowledge did not reach
`organize_segments_path_3rd`'s fallback interpolation, which laid a
dead-straight, zero-deviation chord across the bend; that chord then lost
98% of its points to the (correct) charge-support check, leaving
`track_fit_global` with just two disconnected endpoints where
`stm_fit_global` shows a full, real, densely-fit trajectory. Four
independent, now-measured signals -- TGM's chord-continuity check, the
Steiner `sgp` guard's `detour=6.446`, this segment's 98% charge-drop, and
its `max_perp_dev=0.000cm` pre-drop shape -- all point at the same
underlying fact (this connection bends significantly, it is not a straight
chord), which only `TaggerCheckSTM`'s fitter, having no straight-line
assumption anywhere in its own path, was unaffected by.

### The last open link, closed: why does `fits()` collapse to 2 in the first place?

Traced backward from the collapse (toolkit `ed035408`): added an env-gated
log of `curr_pts`'s size and *source* (`segment->fits()` vs. `segment->wcpts()`)
right at the top of `organize_segments_path_3rd`, before `examine_end_ps_vec`
runs. For segment 86002 across ~10 calls (this cluster gets re-evaluated once
per flash-matched candidate-vertex trial -- 5 candidates per the earlier
`dual_chain` log line):

```
organize_segments_path_3rd: segment gi=2 cluster=86 pre-examine curr_pts=99 (from fits), fits_size=99, wcpts_size=128
organize_segments_path_3rd: segment gi=2 cluster=86 pre-examine curr_pts=2 (from fits), fits_size=2, wcpts_size=128
```

`wcpts_size=128` -- the segment's real, raw, assigned 3D imaging points --
is present and **identical in every single call**, whether `fits()` shows 99
or 2. The 128 real points, which trace the actual bent shape (the same one
`TaggerCheckSTM` fit successfully), are never missing or reassigned. What
changes is only `fits()`: sometimes 99 points survive from an earlier
resample/charge-check round, sometimes it has already been reduced to just
the two vertex endpoints by that same round's `form_map_graph` drop.

The code at `clus/src/TrackFitting.cxx:1546-1555` reads:

```cxx
if (!segment->fits().empty()) {
    for (const auto& fit : segment->fits()) curr_pts.push_back(fit.point);
} else {
    for (const auto& wcpt : segment->wcpts()) curr_pts.push_back(wcpt.point);
}
```

`fits().empty()` is false whenever there are 2 (or more) stale points --
it does not distinguish "a real fitted path" from "collapsed to just the
two vertex endpoints by the previous round's drop." So the branch always
takes the (degenerate) `fits()` when the collapse has already happened,
and **never falls back to the 128 real `wcpts()` sitting right there,
unused, in the very same object**.

That closes the loop completely. Root cause, traced end to end with data
at every link:

1. Segment 86002 is a real ~60 cm connection with 128 real, bent-path raw
   points (confirmed: `TaggerCheckSTM` fit it cleanly on real charge; the
   Steiner graph independently measured its true path detours 6.4x a
   straight chord).
2. At some round, its `fits()` collapses to exactly its 2 vertex
   endpoints (from the charge-check drop discussed above -- itself
   downstream of an earlier straight-line resample).
3. `organize_segments_path_3rd` sees `fits()` non-empty (2 points) and
   uses it as-is -- it has no test for "degenerate" vs. "real," so it
   never reaches for the 128 real points in `wcpts()` that would break
   the cycle.
4. It resamples those 2 points into a fresh, dead-straight ~108-point
   chord (confirmed: `max_perp_dev=0.000cm`).
5. The next `form_map_graph` charge check correctly finds no charge on
   106 of those 108 straight-line points (they are not on the real, bent
   path) and drops them back to 2.
6. Repeat. This is a **self-perpetuating fixed point**: nothing in this
   cycle ever looks at `wcpts()` again once step 2 has happened once, so
   it cannot self-correct, even though the data needed to correct it
   never left the object.

**What this means for a fix**: not in the charge-drop check (doing its job
correctly) and not in `TaggerCheckSTM` (fit the real geometry correctly).
The gap is in `organize_segments_path_3rd`'s `!fits().empty()` test, which
should distinguish a meaningfully-sized/shaped `fits()` from a degenerate
2-point collapse, and in the latter case prefer `wcpts()` (or the graph's
own already-computed Steiner rough-path, which independently agrees the
real path is not straight) over re-interpolating the same doomed chord.

### Is the STM leg real charge, or a fit artifact? (resolved)

Yes, it is real. The raw `clustering-global` points are not gapped in 3D
along x (5 cm bins from x=280 to x=340 all carry 21-42 points, no >30 cm
empty run) -- the physical charge is there, and `TaggerCheckSTM`'s fit,
which stays on real charge for all 141 of its points, follows it correctly.
What is *not* real is `organize_segments_path_3rd`'s fallback interpolation
of that same span as a straight chord: `chord_has_charge` (TGM) and the
Steiner `sgp` guard (NeutrinoID) both flagged, independently and ahead of
time, that a straight chord across this connection is a poor approximation
(`detour=6.446`) -- and the `max_perp_dev=0.000cm` measurement above
confirms `track_fit_global`'s sparse rendering is exactly that: a straight
chord, mostly discarded by the (correct) charge check, not a second read of
the same real trajectory STM found.

## Status

* **Question 1: FIXED**, toolkit `ab0762c6` — `stm_fit`'s Bee
  `real_cluster_id` now matches `cluster_id`. Verified byte-identical
  elsewhere (20/21 `mabc-pr.zip` members unchanged; the 21st differs only in
  the one intended field). Unconditionally inert when `save_stm_fit=false`
  (the production default), so no existing config's output changes.
* **Question 2: root cause fully traced.** Investigation only, nothing
  behavior-changing done (three env-gated, always-inert-by-default DEBUG
  instrumentation additions, `e3dee831`/`f622161e`/`ed035408`, all verified
  byte-identical when their env var/knob is unset -- `abtest/hash_archive.py
  --members` against the doc pdvd/29 `_v450` baseline). The `track_fit_global`
  gap is a self-perpetuating fixed point in `organize_segments_path_3rd`:
  once a segment's `fits()` has collapsed to its two vertex endpoints (by
  the pre-dQ/dx charge-support check dropping a bad resample), the
  function's `!fits().empty()` test treats those 2 stale points as good
  enough to resample from again -- producing another dead-straight chord
  (`max_perp_dev=0.000cm`) across a connection the pipeline's own Steiner
  graph had already measured as detouring 6.4x that chord (`detour=6.446`)
  -- which the charge check then drops right back to 2, forever, without
  ever falling back to the segment's 128 real, unused, bent-path `wcpts()`.
  The STM=1 tag itself is well-founded (`TaggerCheckSTM` fit the real
  geometry, all on real charge); `track_fit_global`'s sparse rendering is
  a resampling artifact, not competing evidence.

## Recommendation / next steps

1. **Candidate fix**, now precisely targeted and well-motivated by data:
   in `organize_segments_path_3rd` (`clus/src/TrackFitting.cxx:1546-1555`),
   change `if (!segment->fits().empty())` to also require the fitted path
   have real extent/count (e.g. `fits().size() > 2` or a minimum chord
   length check), falling back to `segment->wcpts()` — or better, to the
   graph's own already-computed Steiner rough-path, which independently
   agrees the real connection is not straight — whenever `fits()` has
   degenerated to just its two vertex endpoints. This targets the exact
   mechanism traced above and would let a genuinely bent, real connection
   escape the 2-point fixed point instead of being asked to resample the
   same doomed chord forever.
2. **Not implemented here**: this is a change to production PR-graph
   fitting behavior for every PDVD (and, since `TrackFitting` is shared,
   potentially SBND) event, not an opt-in diagnostic -- outside what a
   debugging session should do unilaterally. It needs the usual bar
   (§4): a default-off knob or an explicit owner decision that this is a
   universal improvement, a byte-identical-when-off gate, a knob-on smoke
   run showing the recovered trajectory, and ideally a check on whether
   any *other* long/branchy cluster in the existing validation manifests
   changes shape once escaped from this fixed point.
3. Whether cluster 86's TWO arms (the one just traced, and the separate
   low-z "Michel-like" arm toward the selected vertex) are one real
   particle each or reflect an imaging mis-merge remains a separate, open
   question worth checking against the raw 2D wire signals (a known PDVD
   failure mode; see docs 25-26 in this directory) — but it is no longer
   needed to explain *this* symptom, which is now fully accounted for by
   the mechanism above.

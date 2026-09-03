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

# ---- round 2 (2026-09-03) -------------------------------------------------
# Every arm below copies the pctree read-only into a FRESH tag first (M13).
cd /home/xqian/toolkit-dev/wcp-porting-img/pdvd

# M1: is the pre-dQ/dx drop the cause?  Turn on the existing SBND knob pr/107.
WCT_DQDX_DROP_DEBUG=1 PDVD_PR_TLA="-S dqdx_fit_keep_all_points=true" \
  ./run_pr_evt.sh -s keepall -stm-fit 39252 2
grep -c "pre-dQ/dx form_map_graph dropped" work/039252_2_keepall/wct_pr_039252_2.log   # 0
grep "organize_segments_path_3rd.*cluster=86" work/039252_2_keepall/wct_pr_039252_2.log | tail
# ... fits_size=2 persists on the 128-wcpt arm => the drop is a MASK, not the cause.

# M2: is examine_end_ps_vec draining the path?  (probe added in ed035408)
python3 -c "import json;p='/home/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json';\
d=json.load(open(p));d['traj_cover_probe']=1.0;json.dump(d,open('/tmp/tf_probe.json','w'),indent=2)"
WCT_DQDX_DROP_DEBUG=1 PDVD_PR_TLA="-A trackfitting_config=/tmp/tf_probe.json" \
  ./run_pr_evt.sh -s drainprobe -stm-fit 39252 2
grep -c "DRAINED TO EMPTY" work/039252_2_drainprobe/wct_pr_039252_2.log                # 0

# M3: WHICH form_map_graph stage collapses it?  (probe added this round)
WCT_DQDX_DROP_DEBUG=1 ./run_pr_evt.sh -s stageprobe -stm-fit 39252 2
grep -o "form_map_graph: WCT_DQDX_DROP_DEBUG stage=[0-9]*" \
  work/039252_2_stageprobe/wct_pr_039252_2.log | sort | uniq -c        # 216/201/200
grep "form_map_graph:.*cluster=86.*wcpts=128" work/039252_2_stageprobe/wct_pr_039252_2.log | head
# ... stage=1 goes 81->76 (healthy) then 81->2 right after a do_rough_path graph edit.

# the fix, OFF and ON (knob defaults off; compiled-config + identity proofs)
wcsonnet ... -o cfg_off.json wct-pr-perevt.jsonnet ; grep -c traj_degenerate_wcpts_fallback cfg_off.json  # 0
wcsonnet ... -S traj_degenerate_wcpts_fallback=true -o cfg_on.json ... ; grep -c ... cfg_on.json          # 1
./run_pr_evt.sh -s fixoff2 -stm-fit 39252 2
PDVD_PR_TLA="-S traj_degenerate_wcpts_fallback=true" ./run_pr_evt.sh -s fixon2 -stm-fit 39252 2
# acceptance instrument (max_perp must leave 0.001cm; median-to-stm_fit must fall):
python3 <scratch>/leg_check.py currptsprobe keepall fixoff2 fixon2
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

### Round 2 (2026-09-03): the round-1 chain was wrong in its first link — corrected here

Round 1 (above) ended by asserting that `fits()` collapses to 2 because the
**pre-dQ/dx** `form_map_graph` drops the zero-charge points, and recommended
teaching `organize_segments_path_3rd` to distinguish a degenerate `fits()`.
The recommendation survives. **The attribution did not.** Three measurements,
each a fresh work tag, took it apart:

**M1 — the pre-dQ/dx drop is a mask, not the cause.**
`dqdx_fit_keep_all_points` (doc sbnd_xin/docs/pr/107) is an existing,
default-OFF knob whose entire purpose is to stop exactly that drop
("the prototype never drops a trajectory point between the last trajectory
round and the dQ/dx fit"). Turned on for this event (`work/039252_2_keepall`):

| | pre-dQ/dx points dropped | segment gi/128-wcpt arm `fits()` |
|---|---|---|
| production (knob off) | 390 of 430 in one call | 2, every round |
| `dqdx_fit_keep_all_points=true` | **0** | **still 2, every round** |

So the drop was never what created the 2-point state.

**M2 — `examine_end_ps_vec` is not draining the path either.** Re-run with
`traj_cover_probe=1` (`work/039252_2_drainprobe`), whose probe was extended in
`ed035408` to fire on a total drain: **0 drain-to-empty events** in the whole
event, on any cluster.

**M3 — the collapse is at trajectory round 1, and it is exclusion contention.**
A new env-gated probe at the single drop site inside `form_map_graph` itself
(this round's toolkit commit; `WCT_DQDX_DROP_DEBUG=1`,
`work/039252_2_stageprobe`) attributes every drop to the stage that performs
it. For cluster 86's 128-wcpt arm:

```
48.424  stage=1  gi=2  zero-quantity drop  81 -> 76   (healthy, bent path)
48.428  organize_segments_path_3rd gi=2  curr_pts=99 (from fits)  wcpts=128
48.435  pr55 do_rough_path ... sgp guard VETO detour=6.446
48.440  pr55 do_rough_path ...                      <-- graph edit adds an overlapping segment
48.443  stage=1  gi=2  zero-quantity drop  81 -> 2    <-- the initiating collapse
48.449  stage=2  gi=2  zero-quantity drop 108 -> 2
48.455  stage=3  gi=2  zero-quantity drop 108 -> 2
48.476  stage=1  gi=4  zero-quantity drop  81 -> 2    <-- the duplicate, same 128 wcpts
```

The arm's charge association survives fine (81 → 76) until a mid-fit
`do_rough_path` graph edit adds a second segment over the same charge. With
`fit_exclusion` on, `update_association` strips every 2-D cell that is
(near-)equidistant from two segments — with two segments occupying one arm that
is *every interior cell* — so both copies lose all charge at once and collapse
to their endpoints. Stage counts for the whole event: 216 stage-1, 201 stage-2,
200 stage-3 drops, i.e. the loop runs through **all three** `form_map_graph`
calls, which is why a knob guarding only the third one cannot break it.

### The loop, corrected

1. Cluster 86's ~60 cm "toward-344" arm is a real, 128-point bent path
   (`wcpts()`), present and unchanged in **every** observed call.
2. A mid-fit graph edit adds an overlapping duplicate segment; `fit_exclusion`
   contention zeroes both copies' interior charge; **stage-1** `form_map_graph`
   drops `81 → 2`, leaving only the two endpoint vertices.
3. `organize_segments_path_3rd`'s `if (!segment->fits().empty())` accepts those
   2 points — the test distinguishes only *empty* from *non-empty*, never
   *degenerate* from *meaningful* — and so **never** consults the 128 real
   points sitting unused in the same object.
4. It resamples the 2 points into a fresh ~108-point **straight chord**
   (measured: `max_perp_dev = 0.000 cm` over all 108).
5. Stages 1, 2 and 3 each correctly find no charge on ~106 of those 108 points
   (the real path bends; NeutrinoID's own Steiner router independently reports
   `detour = 6.446` between the same endpoints) and drop them back to 2.
6. Self-perpetuating: nothing in the cycle re-reads `wcpts()`, so it cannot
   escape even though the data needed to escape never left the object.

The `else`-to-`wcpts()` branch that would have rescued step 3 fires **0 times
out of 686** in this event — not because it is unreachable, but because by the
time `_3rd` runs, `fits()` is never *empty*; it is exactly 2.

### Why the existing SBND knob is not the fix

Turning `dqdx_fit_keep_all_points` on does bring the leg back into
`track_fit-global` — and draws it in the wrong place:

| arm | points in the toward-344 box | max perp. deviation | median distance to `stm_fit` |
|---|---|---|---|
| production | 2 | — | — |
| `dqdx_fit_keep_all_points=true` | 80 | **0.001 cm** (dead straight) | **3.45 cm** |
| `stm_fit` (reference) | 64 | 1.158 cm (genuinely bent) | — |

It unmasks the straight chord instead of correcting it, painting a 3.45 cm-wrong
trajectory into production output. **Recommend against flipping it as a remedy
for this symptom** — that is a separate, still-open owner decision on its own
merits (pr/107 §7), and this event is not an argument for it.

This also answers the standing question of whether the SBND PR-chain rounds
already cover this. They cover the *mask* and nothing else: `dqdx_fit_keep_all_points`
is `false` in both `sbnd/wct-pr-perevt.jsonnet` and PDVD's, and the 11 knobs whose
values differ between the two detectors are all STM-tagger guards and
cosmic-skip flags, none of which touch path organization. pr/107's own census
measured the drop at 443 points over 47 nueCC48 events (~9/event, junction-local);
here it is 390 of 430 in a single call — a regime its evaluation never saw.

### The fix (toolkit `a8190d6e`, DEFAULT OFF)

`traj_degenerate_wcpts_fallback` (C++ `TrackFitting::Parameters`, double
sentinel 0 = legacy; jsonnet TLA in `pdvd/wct-pr-perevt.jsonnet` with the
key-suppression idiom). When on, `organize_segments_path_3rd` prefers
`segment->wcpts()` in exactly the degenerate state:

```cxx
const bool use_wcpts =
    segment->fits().empty() ||
    (m_params.traj_degenerate_wcpts_fallback > 0 &&
     segment->fits().size() <= 2 &&
     segment->wcpts().size() > 2 * segment->fits().size());
```

The discriminator is deliberately **not** `fits().size() > 2`: a genuinely short
segment legitimately has two fit points. It is the *mismatch* between a
shapeless `fits()` and a much longer `wcpts()`. Verified against the three real
states in this event's own log:

| segment | `fits()` | `wcpts()` | fires? | why |
|---|---|---|---|---|
| gi=9 | 2 | 2 | **no** | nothing to gain; a real 2-point segment |
| gi=7 | 2 | 12 | yes | raw path carries shape the chord does not |
| gi=13 | 2 | 128 | yes | the arm this doc is about |

**Prototype parity.** The prototype needs no such test because its
`ProtoSegment` constructor seeds `fit_pt_vec` — which is `get_point_vec()`, the
same field toolkit calls `fits()` (`clus/docs/porting/porting_dictionary.md:218-219`)
— from the **full** `path_wcps` (`ProtoSegment.cxx:39-46`), so its equivalent
input is never degenerate by construction. The toolkit leaves `m_fits` empty at
construction and relies on the organize passes to fill it. That divergence is
**not** listed in the porting dictionary; it is recorded here and the knob
restores the prototype's effective behaviour for the one state where the
difference bites.

**Scope: `organize_segments_path_3rd` only.** `organize_segments_path_2nd`
(`TrackFitting.cxx:1716`) carries the identical `!fits().empty()` pattern and is
deliberately left untouched: this round has no probe data on it, and its
resampling is driven by `low_dis_limit`/`end_point_limit` rather than a bare
step size. Fixing a site blind is how a knob-on run turns into an unexplained
gate diff. Named here so the next round does not have to rediscover it.

### Verification

- `wcdoctest-clus` DOCTEST_RESULT (adds a `traj_degenerate_wcpts_fallback`
  default + `set_parameter` round-trip case, and the config-key default check,
  following the `dqdx_fit_keep_all_points` pattern).
- **Compiled-config proof**: key count 0 with the knob off, 1 with it on; the
  OFF compiled JSON `cmp`-identical to the one from the committed jsonnet.
- **OFF-path identity, by construction**: with the parameter at its default 0
  the second disjunct short-circuits and `use_wcpts` reduces to exactly the
  legacy `segment->fits().empty()`, so the OFF path executes the same branch on
  the same data.
- **OFF-path empirical archive gate: BLOCKED, not run.** Both arms
  (`work/039252_2_fixoff`, `work/039252_2_fixon`) exited **rc=139 (SIGSEGV)**
  during teardown, after the physics and the tracking ROOT writer, leaving a
  truncated `mabc-pr.zip` that `hash_archive.py` rejects. The fault is **not**
  the knob: knob-OFF and knob-ON fail identically, and every earlier arm
  (`_keepall`, `_drainprobe`, `_stageprobe`) ran rc=0. The binary was built
  while a concurrent session in this shared working tree was running
  `git stash push`/`pop` cycles over `clus/inc/WireCellClus/NeutrinoPatternBase.h`
  and `clus/src/NeutrinoStructureExaminer.cxx` — stashing a **header** mid-build
  yields mixed object files and exactly this class of teardown crash (see
  memory `feedback_shared_tree_binary_pin`; pinning `LD_LIBRARY_PATH` protects
  against a mid-run swap, but not against a binary that was already raced when
  it was pinned). **Owed: re-run both arms on a clean build of a quiet tree and
  hash-compare `_fixoff*` against `_stageprobe`.** The two crashed tags are kept
  as the record (M13).
- Freshness proof (M1) done at 09:29 (lib newer than every edited source); the
  binary was pinned for both arms.
- **Knob-on effect: measurement owed**, blocked by the same crash. The
  acceptance criterion is **pre-registered here, before the run**, so it cannot
  be fitted afterwards: in the toward-344 box the fallback must move
  `max_perp` **off 0.001 cm toward the ~1.2 cm** the `stm_fit` reference shows,
  and must pull the **3.45 cm** median distance to `stm_fit` down. The leg
  coming back *straight* — i.e. present but still 0.001 cm — counts as the fix
  NOT working, however many points appear. Instrument:
  `scratchpad/leg_check.py` (see Repro).

> **Both items above were settled in round 3 (below), on a clean binary.**
> The **OFF-path archive gate PASSES**: `work/039252_2_fixoff3` (knob off, new
> binary carrying both `traj_degenerate_wcpts_fallback` and the
> `examine_end_ps_vec` guard) vs `work/039252_2_stageprobe` (pre-fix binary) —
> `abtest/hash_archive.py --members` on `mabc-pr.zip`, **all 22 members
> byte-identical**. That covers the `4776d637` guard as well.
> The **pre-registered acceptance criterion FAILS**: the knob-on arm leaves the
> box at 2 points. See round 3 for what actually causes the missing leg.

### Pre-existing and unrelated, found here — FIXED (toolkit `4776d637`)

`examine_end_ps_vec` read `ps_list.back()` at the top of its `flag_end` block.
The `flag_start` block above it can legitimately leave the list **empty** on a
non-empty input: it pops every point failing `is_good_point`, and its S1.7
else-branch re-inserts `temp_start` only when that point has a valid face. The
empty-**input** guard at the top of the function (doc pr/82 sec 12.7) does not
cover that drained state.

This is not undefined-but-harmless. The `std::list` sentinel garbage becomes
`temp_end`, and the **symmetric** S1.7 else-branch `push_back()`s `temp_end`
whenever `m_dv->contained_by()` reports a valid face for it — inventing an
out-of-detector point, exactly what S1.7 exists to prevent. Where it did not,
the function returned empty only by luck, which is why pr/82's evt 54629 stack
showed the crash in the *caller* (`organize_ps_path`) rather than here.

Fixed as `if (flag_end && !ps_list.empty())`, shipped **unconditionally** like
its sibling guard in the same function: the behaviour replaced is undefined,
not a legacy path to preserve. It can differ from the status quo only in the
case where the sentinel read landed in a valid face — i.e. only where the old
code invented a point. Flagged for the owner as a deliberate departure from the
default-OFF-knob rule on those grounds.

No reproducer: unlike the pr/82 cases, draining the list needs `m_dv`,
`m_pcts` and `m_grouping` live, so a null segment and a default-constructed
`TrackFitting` crash long before the drain. A real one needs geometry injected
via `TrackFittingTestHarness` (the pr/98 friend seam used by
`doctest_update_association.cxx`) plus a grouping whose `is_good_point()` is
false everywhere. Recorded in the test file as owed rather than faked with a
test that could not fail. Not reached in this event (0 drain-to-empty events
measured) and independent of `traj_degenerate_wcpts_fallback`.

### Round 3 (2026-09-03): the fallback was VALIDATED AND FOUND NOT TO FIX THIS SYMPTOM

Round 2 shipped `traj_degenerate_wcpts_fallback` reasoned but unvalidated (the
arms had segfaulted on a raced binary). Re-run on a sound binary, the knob-ON
arm completes cleanly — and **does not restore the leg**. The pre-registered
acceptance criterion (round 2, Verification) is the judge, and it fails.

Bee set, all four arms of the *same* physical event side by side (flip between
events 0-3): <https://www.phy.bnl.gov/twister/bee/set/966f429e-652c-405b-9a9d-dc520ce1f3ae/event/list/>

| Bee evt | arm | pts in toward-344 box | max perp | median dist to `stm_fit` |
|---|---|---|---|---|
| 0 | production | 2 | — | 7.07 cm |
| 1 | `dqdx_fit_keep_all_points=true` | 80 | 0.001 cm | 3.45 cm |
| 2 | `traj_degenerate_wcpts_fallback=true` | **2** | — | 3.04 cm |
| 3 | `fit_exclusion=false` | **71** | **1.197 cm** | **0.20 cm** |

(`stm_fit` reference in the same box: 64 points, max perp 1.158 cm.)

**The fallback works mechanically and is irrelevant to the outcome.** It fired
93 times, and on the arm in question exactly as designed:

```
organize_segments_path_3rd: segment gi=2 cluster=86 curr_pts=128 (from wcpts), fits_size=2, wcpts_size=128
```

The path it hands the fitter is now genuinely bent — `n=187 chord=60.67cm
max_perp_dev=5.158cm mean_perp_dev=2.559cm`, against production's dead-straight
`max_perp_dev=0.000cm`. So the geometry defect round 2 diagnosed is real and the
knob does correct it. **But `form_map_graph` then drops 187 → 2 anyway, at every
stage.** The charge check finds no charge on the real, bent, Steiner path
either.

**Round 2's causal chain was therefore wrong about sufficiency.** "Straight
chord ⇒ no charge ⇒ dropped" had the straightness as the cause; it is a
*symptom*. The arm has zero U+V+W association quantity in the PR fitter's charge
maps regardless of its shape.

### What actually causes it: `fit_exclusion` contention (2x2, isolated)

| `fit_exclusion` | fallback | pts in box | max perp | median dist to `stm_fit` |
|---|---|---|---|---|
| `true` (PDVD production) | off | 2 | — | 7.07 cm |
| `true` | on | 2 | — | 3.04 cm |
| `false` | off | **71** | 1.197 cm | **0.20 cm** |
| `false` | on | **71** | 1.197 cm | **0.20 cm** |

The bottom two rows are **identical** — `fit_exclusion=false` restores the leg
by itself, and the fallback adds nothing either alone or in combination (it
fires 0 times on this arm once exclusion is off, because `fits()` never
collapses in the first place). With exclusion off, stage 1's drop stays at the
healthy `81 → 76` on every pass and never collapses to 2.

And the restored trajectory is not merely present, it is *correct*: max perp
1.197 cm against the `stm_fit` reference's 1.158 cm, sitting a median **0.20 cm**
from the trajectory `TaggerCheckSTM` fits on the same charge. The two
independent fitters now agree to within 2 mm, which is the strongest evidence in
this whole investigation that the STM=1 tag was right all along.

Mechanism, consistent with round 2's stage-1 attribution: a mid-fit
`do_rough_path` graph edit puts a **second** segment over the same charge, and
`update_association` with exclusion on strips every 2-D cell that is
(near-)equidistant from two segments — with two segments on one arm that is
every interior cell, so both copies lose all charge at once. `mvga: op1
dup-merge` does fire on this cluster (removing a 69.53 cm segment in favour of a
46.21 cm survivor at overlap 0.99), but further `do_rough_path` calls re-add
overlapping segments afterwards; the dedup and the re-adding are mis-ordered.

**`fit_exclusion=false` is NOT the recommendation.** It is SBND production since
pr/98, PDVD production here, and doc pr/106 §9 measured it as globally
consequential. It is used here purely as the isolating instrument that proves
where the charge goes. The fix belongs at the duplicate segment.

## Status

- **Q1 — FIXED** (toolkit `ab0762c6`): the `stm_fit` Bee layer's hard-coded
  `real_cluster_id = 0` now carries the cluster id.
- **Q2 — root cause FOUND AND PROVEN, and it is not where rounds 1-2 put it.**
  The STM=1 tag is well-founded: with `fit_exclusion` off, the PR fitter's own
  trajectory lands a median **0.20 cm** from the one `TaggerCheckSTM` fits on the
  same charge. `track_fit_global`'s missing leg is caused by **`fit_exclusion`
  contention with a duplicate segment** placed over the same charge by a mid-fit
  `do_rough_path` graph edit — not by the straight-chord resample, which is a
  downstream symptom.
- **The shipped knob `traj_degenerate_wcpts_fallback` (toolkit `a8190d6e`) does
  NOT fix this symptom** and is superseded as an explanation. It is default OFF,
  byte-identical when off, and it does correct a real defect (it puts genuinely
  bent geometry back into the fitter, `max_perp_dev` 0.000 → 5.158 cm) — but the
  charge check drops the arm either way, so on this event it changes nothing
  observable. Kept as a latent robustness guard, explicitly not as the fix;
  retiring it is a reasonable owner call (cf. doc 77's knob ledger).
- **`examine_end_ps_vec` empty-list read — FIXED** (toolkit `4776d637`),
  unrelated and found in passing.

## Recommendation / next steps

1. **Fix the duplicate segment; that is the whole defect.** The exclusion
   contention exists only because two segments end up on one arm. `mvga: op1
   dup-merge` already fires on this cluster (removing a 69.53 cm segment at
   overlap 0.99) but later `do_rough_path` calls re-add overlapping segments —
   the dedup and the re-adding are mis-ordered. Either re-run the dedup after
   the last graph edit, or stop `do_rough_path` from adding a segment that
   duplicates an existing one. This is the next round.
2. **Do not flip `fit_exclusion` to false.** It is SBND production since pr/98
   and PDVD production here, and pr/106 §9 measured it as globally
   consequential. It is used in this doc purely as the isolating instrument
   that proves where the charge goes.
3. **Do not flip `dqdx_fit_keep_all_points` for this symptom** — it restores the
   leg in the wrong place (dead straight, 3.45 cm off; Bee event 1). It is a
   plausible-looking lever precisely because the track reappears, which is why
   this is recorded as a measured negative rather than left unsaid.
4. **Owner call on `traj_degenerate_wcpts_fallback`**: keep as a latent guard or
   retire it. It is inert on this event; the degenerate-resample state it
   targets is real but, here, always dominated by the charge drop. If kept, it
   still needs an SBND gate before any flip, since `TrackFitting` is shared.
5. `organize_segments_path_2nd` carries the same untested `!fits().empty()`
   pattern (round 2, Scope) — untouched, and now lower priority given item 1.
6. Whether cluster 86's two arms are one real particle each or an imaging
   mis-merge remains open, but is no longer needed to explain this symptom.

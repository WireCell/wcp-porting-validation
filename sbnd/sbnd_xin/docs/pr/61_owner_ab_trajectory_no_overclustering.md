# doc pr/61 — "a track connects A to B" in 18255-71372 / 18255-142421: neither point pair is one reconstructed Cluster with a phantom trajectory; the visual read comes from Bee's point-cloud rendering plus (event-specific) real code mechanisms unrelated to protect_bundle's S6 rescue

Status: **diagnosis only, this session. No code, no config, no knob.** Owner's
scope: understand what is going on; write it up.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 - <<'PY'
import zipfile, json, numpy as np
def load(evt,name,arm='work-pr57r6-scan19'):
    z=zipfile.ZipFile(f'{arm}/pr_evt{evt}/mabc-pr.zip')
    return json.loads(z.read(f'data/0/0-{name}-global.json'))
# every number in this doc is reproducible from these two zips + numpy/scipy;
# see sec 2-4 for the exact queries.
PY
```

Inputs: `work-pr57r6-scan19/pr_evt{71372,142421}/mabc-pr.zip` (post-flip
production, doc pr/57 round 6), `work-pr55-instr19b/pr_evt{71372,142421}/mabc-pr.zip`
(pre-flip, `relaxed_strict_img`, for the causation check in §2). Read-only —
no `work-*` content modified. Figures: `docs/pr/pics/59_71372_overview.png`,
`59_142421_overview.png`, `59_142421_isolated_rcid.png`, `61_71372_chain.png`.

## 1. Symptom

Owner scanned the post-flip Bee set `22ca051a-d032-4b2b-9454-69a7bf85630e`
(built from `work-pr57r6-scan{19,48,395}`, idx 1 = 71372, idx 4 = 142421) and
reported two fitted trajectories, read directly off the display:

- **18255-71372**: `(-155.3,-102.8,229.0)` → `(-151.9,-144.9,268.3)` (57.7 cm)
- **18255-142421**: `(111.4,-87.3,295.3)` → `(108.9,-54.1,249.8)` (56.4 cm)

Reasoning given: `protect_bundle`'s graph is what decides whether two
components stay one `Cluster`; pattern recognition and track fitting run on
whatever clustering already exists; therefore a fitted trajectory spanning
two points means those points were clustered together **at the step just
flipped on** (doc pr/57 round 6, `relaxed_strict_img_2d_rescue`).

The first two premises are correct in the code (§2). The conclusion does not
hold for either named pair, for two different reasons — one per event (§3,
§4) — and neither reason is the round-6 flip.

## 2. What is true in the code, and what it does not imply

`ClusteringProtectBundle.cxx:428,481` really does call `Grouping::separate()`
on the graph named by `protect_graph_name`, moving cut components into
distinct `Cluster` objects and purging cached graphs
(`ClusteringProtectBundle.cxx:514-533`). `do_rough_path` / `TrackFitting` /
`CreateSteinerGraph` are strictly per-`Cluster`
(`NeutrinoPatternBase.cxx:111-120`), so **no single fitted trajectory can
cross a protect_bundle cut.**

But two things break the chain from "fitted trajectory looks continuous" to
"same Cluster, cut by the flip":

1. **Bee never draws lines.** Every point layer (`img`, `clustering-global`,
   `track_fit-global`, `shower_track-global`, `vertices-global`) is rendered
   as a `THREE.Points` scatter cloud (`wire-cell-bee3/events/static/js/bee/physics/sst.js:196`,
   `initPointCloud` → `drawInsideBox`). There is no polyline, no
   `THREE.Line`/`LineSegments` connecting consecutive points of a segment or
   across segments. A "trajectory" is visually a dense (~0.6 cm-spaced)
   sequence of individual points that *looks* continuous, not a drawn edge —
   so a chain of several small, separately-clustered objects sitting close
   together can look exactly like one continuous track, whether or not they
   are one `Cluster`.
2. **Color aliasing.** `store.js:91-107` defines only **14** colors; the
   point color is `real_cluster_id % 14` (`sst.js:152-153`). With hundreds to
   thousands of `real_cluster_id` values in a busy event, most colors repeat
   many times — a coincidence of color is not evidence of one object.

Given (1), the operative question for each event is not "is there a drawn
line" but "is there a real, densely-spaced chain of points — charge or fit —
between A and B, and is it one `Cluster`." Both were tested directly.

**Causation check (both events, done first, decisive):** cluster membership
of A and B, pre-flip (`work-pr55-instr19b`, `relaxed_strict_img`) vs.
post-flip (`work-pr57r6-scan19`, `relaxed_strict_img_2d_rescue`):

| event | pre-flip | post-flip |
|---|---|---|
| 71372 | A cid 69 (14 pts) — B cid 19 (8541 pts): **different** | identical: A cid 69, B cid 19 (5865 pts) |
| 142421 | A cid 7 — B cid 7: **same** | identical: A cid 7, B cid 7 |

Membership at both points is byte-identical before and after the flip in
both events (the flip made both events *more* fragmented overall: 71372
116→132 clusters, 142421 101→103). **The round-6 rescue is not implicated in
either case.**

**No fitted trajectory has endpoints at the reported A/B in either event**
(best endpoint-pair match: 42.0 cm in 71372, 40.0 cm in 142421) — so the
"trajectory" being read is not a single segment's polyline either.

## 3. evt 71372 — genuinely different Clusters, correctly separated; the visual read comes from a busy shower's cross-cluster PF stitching

**Charge**: A's nearest charge is 0.041 cm away, in `cluster_id 69` (14
points, a ~1.3×0.5×0.6 cm blob). B's nearest charge is 0.037 cm away, in
`cluster_id 19` (5865 points, the event's big shower). Minimum charge-to-charge
distance between the two clusters: **42.9 cm**.

**No physical chain, even generously.** A proximity-graph BFS over the raw
`clustering-global` charge cloud (all points, all clusters, edges at hop
radius up to 5 cm) leaves A's 14-point component completely isolated — it
does not merge with anything, let alone reach B's component, at any radius
up to 5 cm. The same test on `track_fit`/`shower_track` fit points (which can
include unsupported points, so a more generous test) also finds A's
component and B's component disjoint at 1–3 cm hops.

**MC truth confirms the split is correct.** True interaction vertex
`(-167.5,-155.5,225.7)`; A matches to 0.30 cm the endpoint of a **118 MeV
electron** from a π⁰-gamma conversion (`(-168.0,-137.8,228.1)` →
`(-155.3,-103.1,229.0)`); B sits inside the **855 MeV electron** shower
(cluster 19). Different particles, correctly in different Clusters both
before and after the round-6 flip.

**What is visually near-continuous, and why.** `shower_track-global` (the PF
shower layer) does chain four *distinct* Cluster-level fragments from A's
neighborhood to B's neighborhood with small 3D gaps:

| shower_track object | underlying `cluster_id`s | closest to A | closest to B | gap to next |
|---|---|---|---|---|
| `97136` | 68, 69, 97 | 0.04 cm | — | 4.29 cm |
| `66101` | 56, 57, 66, 75 | — | — | 2.17 cm |
| `91112` | 19, 61, 63, 72, 91 | — | — | 0.35 cm |
| `19032` | 19, 60, 62 | — | 0.04 cm | — |

All four are, individually, small real-charge fragments (22–5865 points,
real centroids — verified in `clustering-global`, not phantom points; see
`docs/pr/pics/61_71372_chain.png`). They are **not one Cluster** — `97136`'s
members (68/69/97) never overlap `19032`'s (19/60/62) in `cluster_id`. What
strings them together at the display layer is
`NeutrinoShowerClustering.cxx:1217-1254` ("Add segments from other
clusters"): a shower absorbs a foreign segment when its direction agrees
within 12.5–25° and it is within 40–120 cm (radius depends on which angle
gate fires), independent of any `protect_bundle` graph. This is a real,
pre-existing code path — unrelated to S6/the rescue — that can make several
separately-clustered fragments read as one shower-level PF object without
ever merging their underlying `Cluster`s or point clouds.

In two projected views (`docs/pr/pics/61_71372_chain.png`) the four objects
do **not** actually form a single unbroken line — `97136`+`66101` sit in one
tight group near A, `91112`+`19032` in another near B, with a visible gap
between the groups in the chosen projections; only specific point pairs
(not the bulk of each object) achieve the sub-5 cm closest approach. In a
busy, fragment-heavy shower region a static screenshot, plus 14-color
aliasing, makes this kind of loose spatial proximity easy to read as "one
track" even though no single object, fit, or `Cluster` spans A to B.

**Bottom line for 71372: no over-clustering, nothing to fix.** The
separation the owner is asking about already exists and matches truth; the
"track" is a shower-level PF-stitching artifact of a busy multi-fragment
region, not a phantom trajectory inside one Cluster.

## 4. evt 142421 — A and B genuinely are the same Cluster (unaffected by the flip); the visual chain is real fitted structure, and a concurrent investigation (doc pr/59) has already found the deeper mechanism for a directly adjacent symptom in the same cluster

Unlike 71372, A and B **are** the same `cluster_id 7` (the main/neutrino
cluster), both before and after the round-6 flip (§2 table) — so the owner's
premise "clustered together" is correct here, just not attributable to the
flip: cluster 7's membership at these two points is identical in
`relaxed_strict_img` (round 7) and `relaxed_strict_img_2d_rescue` (round 6/production).

Cluster 7 is not internally one blob at fine radius: at a 1 cm charge-hop
radius it is 140 pieces; B sits on a 17-point island whose minimum gap to
the rest of cluster 7 is **9.21 cm** — beyond both the round-6 rescue's ≤5 cm
window and any reach `protect_bundle`'s graph has, so no protect_bundle
flavor (past, present, or a future S6 variant) could touch this junction.

**A real near-continuous chain of *fitted* segments exists.** Segment
`7111` (342 pts, 204.6 cm, 0.60 cm-spaced) passes 2.60 cm from A; segment
`7010` (102 pts, 63.9 cm) passes 0.90 cm from B; a third, `7019` (115 pts),
sits between them. Segment-to-segment closest approach:

```
A --2.60cm--> [7111] --4.63cm--> [7019] --10.04cm--> [7010] --0.90cm--> B
```

(`docs/pr/pics/59_142421_isolated_rcid.png`). All three segments belong to
cluster 7's own `do_rough_path` routing on `"steiner_graph"` — per doc
pr/55's already-documented finding, this graph carries **uncapped** MST
bridges and has no relationship to `protect_bundle`'s `graph_name` knob at
all (`NeutrinoPatternBase.cxx:111-119`). Segment `7010` alone is doc pr/55's
already-catalogued case **142421-G1**: a 22.2 cm ghost run peaking 10.77 cm
from any charge, rooted at B, and it is exactly the `other_seg_keep_isolated`
segment doc pr/54 shipped (`pr54 keep-isolated: cluster 7 n_points=564
length=64.03cm`) — its own fit does not follow the shower it was kept for.

**A separate, deeper mechanism for an adjacent symptom in this same event
has already been found and documented by a concurrent session**, doc pr/59
(`59_shower-track-associate-points-swap-gap.md`, toolkit `7001cd5b`, wcp
`202674d`, both already committed and pushed). It answers a related but
distinct owner report — a missing-`shower_track` complaint at
`(146.7,-46.1,239.0)`, a third point in this same busy cluster 7 — with an
instrumented, code-confirmed root cause: segment `7020` (a different segment
from the 7111/7019/7010 chain above) is created *after* cluster 7's only
`clustering_points_segments` association pass, and is then permanently
orphaned when `determine_overall_main_vertex_DL`'s rerank silently swaps
`main_cluster` from 7 to 106 (`swap_main_cluster`,
`NeutrinoPatternBase.cxx:2963`, previously un-logged) before the second
association pass runs. That doc's diagnosis is authoritative for that
specific symptom and is **not** re-derived here; it independently confirms,
via five separate arms including today's production flavor, that the hole
is unrelated to `protect_bundle`'s graph choice (its own §2).

Read together: cluster 7 in this event carries **at least two** distinct,
independent artifacts of the same general kind (a fitted/PF structure
extending into regions the image does not support) — segment `7020`'s total
association orphaning (pr/59) and the `7111`→`7019`→`7010` chain's steiner-routing
gaps (pr/55, confirmed fresh here). Neither is caused by, or fixable via,
`protect_bundle`'s `graph_name` — both live entirely downstream of it, inside
the PR/fitting stages that never consult that graph (doc pr/55 §2).

## 5. Summary

| event | are A, B the same Cluster? | caused by the round-6 flip? | mechanism |
|---|---|---|---|
| 71372 | No (cid 69 vs 19), correctly, per MC truth | No — identical pre/post flip | Bee's point-only rendering + `NeutrinoShowerClustering`'s cross-cluster foreign-segment absorption (`:1217-1254`) chaining 4 distinct real Clusters at the PF/display layer |
| 142421 | Yes (cid 7), both before and after | No — identical pre/post flip; the junction (9.21 cm) is beyond any protect_bundle graph's reach | `do_rough_path`'s uncapped-MST `"steiner_graph"` routing (doc pr/55) stringing together real but gapped fitted segments within one already-merged cluster; a related, deeper defect for an adjacent segment in the same cluster is doc pr/59 |

No code or config is changed by this doc. Nothing here contradicts or
supersedes doc pr/57's flip (§14.7) — the flip's own gates (compiled-config
proof, `wcdoctest-clus` 152/152, byte-identical off-arms) are unaffected,
since neither case traces to the flavor this doc examined.

## Gotchas carried forward

- Bee renders **every** point layer as `THREE.Points` — never assume a
  "trajectory" seen in the viewer is a drawn edge; it is always a dense
  point sequence, and dense-but-separate sequences from different objects
  can look identical to one continuous object, especially with only 14
  colors aliasing across hundreds of `real_cluster_id`s.
- `NeutrinoShowerClustering`'s foreign-segment absorption (`:1217-1254`,
  40–120 cm depending on angle) is a second, PF-level cross-cluster
  stitching mechanism, independent of `protect_bundle`'s graph and of doc
  pr/55's leak paths 1/2 (which are within-cluster). Any "why does this look
  connected" question should check both.
- Two Claude sessions were working this same Bee set concurrently this
  session; doc pr/59 (142421, missing-shower_track symptom) and this doc
  (both events, the "phantom connecting trajectory" symptom) were written
  independently and cross-link rather than duplicate. Always `ls docs/pr/`
  for the next free number before writing, and check `git log` on the
  candidate filename — a concurrent session may have already claimed it.

Related: [[project_pr57_separation_scan]] (the flip this investigation
clears of causation), [[project_pr55_fit_vs_image]] (leak paths 1-3, the
`steiner_graph` mechanism reused in §4), doc pr/59 (the deeper 142421
mechanism, cited not re-derived), [[feedback_concurrent_sessions_same_tree]].

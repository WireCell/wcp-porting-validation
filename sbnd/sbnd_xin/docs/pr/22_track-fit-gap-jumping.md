# Doc pr/22 — Why `track_fit` points fly over space the `track_shower` layer never covers

**Status: CLOSED — analysis + one runner-only fix (owner-approved 2026-08-02).**
No C++ or jsonnet changed.  `run_pr_evt.sh`'s tagger shorthands
(`-stm/-tgm/-nu/-dnn`) now carry the production chain's
`unmerge_bundle,unmerge_assoc` stages (§5.2, §6); production itself never
lacked them, so **no production baseline changes** — only future per-event
scan Bee sets.  Owner
question (2026-08-02): in Bee set
<https://www.phy.bnl.gov/twister/bee/set/29af12de-2cb2-43fb-b3d8-934000889c0d/event/1/>
(the doc pr/19 OLD-baseline scan set; event index 1 = **evt 386948**, a nueCC
candidate from the 1000-event data production), many `track_fit` points are
not covered by `track_shower` points — the track fitting "jumped gaps".  Gap
jumping across dead regions / SP failures is wanted, but this looks like too
much.  Is it the `improve_cluster` rerun, or something else?

**Answer in one paragraph.**  It is not `improve_cluster`.  The uncovered
trails are fitted trajectories of PR segments whose *paths* ride the steiner
graph, and the steiner graph connects every disconnected fragment of the
cluster with uncapped MST bridge edges (a faithful port of the prototype's
`Connect_graph`).  The excess in THIS Bee set is dominated by a
**scan-runner pipeline mismatch**: `run_pr_evt.sh -nu` (which produced the
set) omits the `unmerge_bundle,unmerge_assoc` stages that the production PR
chain runs, so the fit operated on the pre-unmerge flash bundle — every
bundle companion and isolated-grouping absorption collapsed into one
cluster, and the PR chain dutifully stitched them together across voids.
Re-running the same event with the production pipeline drops the in-void fit
points from 26.3% to **7.9%** and the total uncovered length from 146 cm to
**33 cm**, with the two 45–49 cm monster void bridges gone entirely.  The
residual 33 cm is short (≤ 10 cm) hops between genuine nueCC shower
fragments 1.6–3.7 cm off charge — the designed WCP behavior.  Zero overlap
with dead regions either way (this is MC-era data-prod with tiny dead area).

## 0. Repro

```bash
# probe a PR Bee dump (any mabc-pr.zip with track_fit/shower_track layers):
python3 sbnd_xin/scripts/analysis/misc/gapjump_probe.py \
    sbnd_xin/work-oc19scan-old/pr_evt386948/mabc-pr.zip 1.0

# the two fresh arms (toolkit HEAD fe6b7d90, production install):
cd sbnd_xin
mkdir -p work-pr22gap-input work-pr22gap-a work-pr22gap-b
ln -sf  $PWD/work-oc19scan-old/evt386948/sp-frames.tar.bz2 work-pr22gap-input/frames-dnn.tar.bz2
ln -sfn $PWD/work-oc19scan-old/ql_evt386948 work-pr22gap-a/ql_evt386948
ln -sfn $PWD/work-oc19scan-old/ql_evt386948 work-pr22gap-b/ql_evt386948
# arm A — replica of the scan set's pipeline (-nu, NO unmerge):
SBND_INPUT_DIR=$PWD/work-pr22gap-input SBND_WORK_ROOT=$PWD/work-pr22gap-a \
    ./run_pr_evt.sh data -nu 1
# arm B — the production fit-relevant prefix (unmerges before steiner):
SBND_INPUT_DIR=$PWD/work-pr22gap-input SBND_WORK_ROOT=$PWD/work-pr22gap-b \
    ./run_pr_evt.sh data -p \
    switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_neutrino \
    -bw 0.2,2.2 1
python3 scripts/analysis/misc/gapjump_probe.py work-pr22gap-a/pr_evt386948/mabc-pr.zip 1.0
python3 scripts/analysis/misc/gapjump_probe.py work-pr22gap-b/pr_evt386948/mabc-pr.zip 1.0
```

Fresh tags `work-pr22gap-{input,a,b}` (M13: nothing written into
`work-oc19scan-old`; its `ql_evt386948` pctree is read-only input).

## 1. What the two layers actually are

Both layers are dumped by the same PR job
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet` `bee_points_sets`,
visitor `TaggerCheckNeutrino:pr`):

- **`track_fit`** = the PR-graph segment **fit points** (plus vertex fit
  points, `real_cluster_id = -1`), i.e. the fitted trajectory
  (`MultiAlgBlobClustering.cxx` `fill_bee_points_from_pr_graph`).
- **`shower_track`** (displayed as *track_shower*) = each segment's
  **`associate_points`** cloud (`use_associate_points: true`): the cluster's
  own real 3-D points partitioned among segments by
  `clustering_points_segments` (`PRSegmentFunctions.cxx:1889`) — a Voronoi
  split of measured points, q=15000 shower / q=0 track.

So "fit points not covered by track_shower" = fitted trajectory passing
through space where the cluster has **no measured points at all**.  The
probe confirms this is not an association radius artifact: for every
uncovered fit point in every arm there is also *no `clustering`-layer charge
within 1 cm* (260/260 in the scan-set zip, 48/50 in the production arm).

## 2. Where a fit trajectory through empty space comes from

Chain, source-verified at toolkit HEAD `fe6b7d90`:

1. **Paths ride the steiner graph.**  Every PR rough path is a Dijkstra
   shortest path over `steiner_graph` on `steiner_pc`
   (`NeutrinoPatternBase.cxx:95-106` `do_rough_path`), and
   `find_other_segments` (`NeutrinoOtherSegments.cxx:31-240`) *explicitly*
   builds new segments spanning the connected components of the remaining
   terminals (Voronoi + component-connection over `steiner_graph`).
2. **The steiner graph bridges fragments with uncapped edges.**  The
   underlying `ctpc_ref_pid` graph is completed by
   `connect_graph{,_with_reference}`: between the point clouds of
   disconnected components an MST is formed and its edges added with **no
   distance cut** (`connect_graph.cxx:155-169` — the visible
   `> 5*units::cm` branch assigns the *same* distance in both arms; only
   the directional extras get ×1.2).  Faithful port of the prototype
   `PR3DCluster::Connect_graph` (`pid/src/PR3DCluster_graph.h`,
   `prim_minimum_spanning_tree`, no cap) — same mechanism doc 50
   established for the STM fit.  Do **not** "fix" it for parity (M15).
3. **The fit interpolates across any jump at 0.6 cm.**
   `TrackFitting::organize_segments_path` resamples the path with
   `npoints = round(dis / low_dis_limit)` interpolated points across
   *arbitrarily long* inter-waypoint jumps
   (`TrackFitting.cxx:1752-1763`, and again at `:1771-1782`;
   `low_dis_limit = 0.6 cm`).  That is why a 45 cm void crossing appears in
   Bee as a dense dotted trail (measured spacing: median 0.60 cm even
   inside voids) rather than one long invisible chord.

### 2b. `improve_cluster` is NOT the cause

The steiner stage does rerun the improve chain
(`steiner: cm.steiner(retiler=improve2)`, sbnd `clus.jsonnet:1020,1072`),
but:

- `ImproveCluster_2::mutate` injects fake activity **only along the single
  end-to-end trunk path** of the cluster (`hack_activity_improved`,
  `improvecluster_1.cxx:424-567`: a ±3-slice × ±3-wire disc of sentinel
  charge `(0, 1e12)` around under-covered path points, then re-tiling).
  The many trails in the event are per-segment paths, not the one trunk
  tube.
- The retiled/improved cluster is a **temporary**: `CreateSteinerGraph`
  builds `ctpc_ref_pid`, the trunk path, and the steiner tree on it, then
  transfers only `steiner_graph` + `steiner_pc` back to the original
  cluster and destroys the retiled child
  (`CreateSteinerGraph.cxx:174-260`).  Improve-created points never enter
  the cluster's real point cloud, so they cannot appear in `shower_track`
  — but steiner terminals *can* sit on them, which is the intended
  dead-region / SP-failure gap bridging along the trunk.

## 3. The quantification (evt 386948)

`scripts/analysis/misc/gapjump_probe.py`: for every `track_fit` point, distance to nearest
`shower_track` point and to nearest `clustering` (QL charge) point;
uncovered = no `shower_track` point within 1 cm; stretches = ≥ 2 consecutive
uncovered fit points of one segment.

| arm | pipeline | fit pts | uncovered | total stretch len | true-void stretches (> 3 cm from any charge) |
|---|---|---|---|---|---|
| scan-set zip (2026-08-01) | `-nu` (no unmerge) | 761 | 260 (**34.2 %**) | 188.5 cm / 16 | 4, **111.0 cm** |
| A: replica at HEAD | `-nu` (no unmerge) | 700 | 184 (**26.3 %**) | 146.3 cm / 16 | 4, 87.6 cm |
| B: production prefix | `+ unmerge_bundle,unmerge_assoc` | 634 | 50 (**7.9 %**) | **33.3 cm** / 7 | 1, **1.2 cm** |

(A vs the zip differs only because HEAD moved between 2026-08-01 and now —
the B0 cathode-kink veto and re-fits shift break points; the comparison that
matters is A vs B at the same HEAD.)

Attribution of the scan-set zip's 16 stretches:

- **13 of 16 bridge two *different* connected components** of the event's
  charge cloud (3 cm linkage: the fitted cluster spans **90 components**,
  top sizes 12306/11866/6918 pts — the pre-unmerge bundle).  The two
  monsters: seg 16015, 45.0 cm through void (median 8.6 cm from any
  charge) and seg 16050, 49.2 cm (median **11.7 cm**), which jumps from
  y=147 to y=192 to reach a 15-point clump.  Segments 16015/16050 are
  83 %/93 % uncovered — near-phantom segments that exist mostly to bridge.
- **0 of 16 overlap a dead region** (56 dead-area polygons in the dump;
  zero uncovered points inside any, both orientations tested).
- In arm B all seven remaining stretches are ≤ 10.2 cm, sit 1.0–3.7 cm
  from charge (i.e. hopping between *adjacent* shower fragments), and the
  single true-void stretch is 1.2 cm.  This is the designed
  shower-fragment stitching, same as uBooNE.

## 4. Why the scan set overstates it

`run_pr_evt.sh -nu` expands to
`switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_neutrino`
(`run_pr_evt.sh:125`) — **no unmerge stages** — while the production PR
chain runs
`switch_scope,unmerge_bundle,unmerge_assoc,steiner,...`
(`run_pr_chain_batch.sh:107`).  Without the unmerges the "main cluster" fed
to steiner + fitting is the Q/L flash-merged bundle (every member of the
matched bundle collapsed to one cluster, docs 45/50) *plus* the
isolated-grouping absorptions (docs 51/52).  The PR chain then does exactly
what it is built to do — span all of it with one connected graph — and the
fit draws 0.6 cm-spaced points along every bridge.  Doc 50 hit the same
epoch-mixing trap from the other side (the port-5010 viewer).

Production is *not* immune — doc 50's census: 69 % of fitted mains remain
multi-component at 5 cm linkage even after unmerge (clustering-chain merges
with no record to invert), 26 % of STM fits have a trajectory point > 5 cm
from their own cluster's charge.  But for this event and this question, the
production-config residual is the modest, intended behavior of §3 arm B.

## 5. Recommendation

1. **No C++ change** — the bridging is a faithful prototype port doing its
   job; the residual production-config jumping (7.9 % / 33 cm here) is the
   designed shower stitching plus SP-dropout hops, and dead-region jumps
   would use the same machinery.
2. **Scan-set runner alignment — DONE (owner-approved).**
   `run_pr_evt.sh`'s `-stm`, `-tgm`, `-nu`, `-dnn` shorthands now insert
   `unmerge_bundle,unmerge_assoc` after `switch_scope`, mirroring the
   production pipelines (`run_pr_chain_batch.sh:107`,
   `run_nusel_evt.sh:72` PIPELINE_FULL).  Verified: fresh tag
   `work-pr22gap-c`, updated `-nu` on evt 386948 reproduces arm B exactly
   (634 fit points, 50 uncovered = 7.9 %, same stretch table).  Explicit
   `-p` pipelines and the empty-pipeline round-trip identity gate are
   untouched.  Bee sets built before this change (docs pr/3–pr/21 era
   `-nu`/`-tgm`/`-stm` sets, incl. the pr/19 pair) show the pre-unmerge
   fit and must be read with this doc's caveat.
3. **If void-crossing fits ever need flagging** (doc 50 remedy 2): persist
   the max fit-point-to-charge distance per segment — makes the jumps
   visible in scans without changing any verdict.  Evidence-blocked;
   nothing here demands it.

## 6. Baseline impact of the runner change

- **Production: none, and nothing was ever missing there.**  The batch
  production runner was *born* with both unmerge stages
  (`run_pr_chain_batch.sh` created 2026-07-31 with
  `switch_scope,unmerge_bundle,unmerge_assoc,steiner,...` — doc pr/11's
  1071-event census and every mcp1kall arm since), and the nusel arm
  (`run_nusel_evt.sh` PIPELINE_FULL) likewise runs both.  All production
  score tables, verdict censuses, and hand-scan baselines are unaffected.
- **What was affected**: only per-event scan/Bee dumps made with
  `run_pr_evt.sh -nu/-tgm/-stm` (shorthands written 2026-07-10, before the
  unmerge visitors existed — docs 45/51-52 — and never updated).  Those
  Bee sets remain valid for what they were scanned for; their `track_fit`
  trails overstate gap jumping per §3-4.
- **No scripted consumer changes behavior**: nothing calls the shorthands
  from other scripts (`scripts/analysis/misc/compare_pr_roundtrip.py` uses the empty pipeline /
  `-p switch_scope`, both untouched).

## 7. Bee: before/after on evt 386948

Two-event set, same event, same HEAD, same Q/L input — only the PR pipeline
differs.  Index **0 = OLD** (`work-pr22gap-a`, pre-fix `-nu`, no unmerge),
index **1 = NEW** (`work-pr22gap-c`, post-fix `-nu`):

<https://www.phy.bnl.gov/twister/bee/set/81518c11-055d-4bad-b66e-96ecbd7b7227/event/list/>

Compare `track_fit` against `shower_track`: index 0 shows the long dotted
trails across empty space (incl. the 45 cm and 49 cm monsters), index 1 does
not.  Also visible in the `mc` particle-flow tree and in the remap log: the
OLD PR job fits **1** cluster (the whole pre-unmerge flash bundle), the NEW
one **21** (the unmerged pieces, each fit on its own).

Built with `scripts/bee/make_pr_bee.py -q work-oc19scan-old -p work-pr22gap-{a,c}`, the
two single-event zips merged into `data/{0,1}`.

## 8. The residual jumps: uboone's second graph examination is missing from the PR chain

Owner follow-up on the §7 Bee set: the NEW arm is much better but still has
fit points in gaps, and the trajectory visibly connects several photons —
didn't uboone have another round of graph examination before PR?  It did,
and it is indeed missing from the SBND toolkit PR chain.

### 8.1 What the residual actually is (arm C probe)

50/634 fit points uncovered (7.9%), 33.3 cm in 7 stretches, 48/50 through
genuinely charge-free space, zero dead-area overlap.  **Every stretch is
inside cluster 16** (the nu candidate).  Attributing each stretch's endpoints
to connected components of the charge cloud at 3 cm linkage:

| seg   | len (cm) | endpoint components (size) | verdict |
|-------|---------|----------------------------|---------|
| 16003 |     9.0 | 15(823) ↔ 12(814)          | bridges two fragments |
| 16011 |     3.9 | 12(814) ↔ 10(319)          | bridges two fragments |
| 16024 |     6.6 | 4(6918) ↔ 15(823)          | bridges two fragments |
| 16030 |    10.2 | 12(814) ↔ 8(2738)          | bridges two fragments |
| 16013/16022/16035 | 3.6 total | same component     | in-fragment stitching |

So 29.7 of the 33.3 cm are Dijkstra/MST bridges between **distinct charge
fragments that share cluster id 16** — the "several photons connected by one
trajectory" seen in Bee.  The remaining 3.6 cm is benign short stitching
within a fragment.

### 8.2 Is the initial clustering the origin?

Partly, and by design: the clustering tail deliberately merges
vertex-proximate fragments (photon showers) into the nu cluster.  The
ordering is prototype-faithful — `ToyClustering.cxx:374-395` runs deghost →
examine_x_boundary → `Clustering_protect_overclustering` (:388) →
`Clustering_neutrino` (:395), and `sbnd/clus.jsonnet:259-263` mirrors it
exactly (`cm.deghost, cm.examine_x_boundary, cm.protect_overclustering,
cm.neutrino`).  So the clustering-stage overclustering protection runs
*before* the neutrino merge in **both** codebases and never re-splits the
merged photons.  uboone tolerated this because of a second, later round:

### 8.3 The uboone pid-stage round: `WCPPID::Protect_Over_Clustering`

In the prototype nue app (`wire-cell-prod-nue.cxx:1313-1322`), after Q/L
matching and before NeutrinoID, every beam-window bundle is re-examined:

- `Protect_Over_Clustering` (`pid/src/ProtectOverClustering.cxx:6-160`)
  calls `PR3DCluster::Examine_graph(ct_point_cloud)` on **each** cluster of
  the bundle, including the main (`:56`).
- `Examine_graph` (`data/src/PR3DCluster.cxx:2311`) rebuilds the graph as
  close-connected edges + `Connect_graph_overclustering_protection(ct)` —
  inter-component bridges must pass `check_connectivity` against the 2D
  charge/dead-channel CT cloud (`pid/src/PR3DCluster_graph_overclustering.h`
  :105-183); failing bridges are dropped and the cluster splits into its
  surviving connected components.
- The largest component keeps the main cluster id; the rest become new
  clusters **in the same parent group** (`ProtectOverClustering.cxx:104-124`).
- `NeutrinoID(main_cluster, additional_clusters, ...)` then fits each piece
  on its own (`wire-cell-prod-nue.cxx:1345,1360`); shower clustering later
  re-associates them at the *shower* level, so the photons are still part of
  the interaction — but no track fit ever bridges the voids between them.

Our residual bridges (3.9–10.2 cm hops, 1.6–3.7 cm off charge, no dead
channels) are exactly what `check_connectivity` exists to cut.  (Caveat: the
test runs in 2D projections, so overlapping unrelated activity can keep a
3D-empty bridge alive; a definitive per-bridge verdict needs the port.)

### 8.4 The SBND toolkit PR chain has no counterpart

The production PR pipeline is `switch_scope, unmerge_bundle, unmerge_assoc,
steiner, ...` (`wct-pr-perevt.jsonnet:89`).  The unmerge stages restore the
clustering-era cluster boundaries (bundle → main + associated), which is why
§5's fix helped — but they cannot split *within* a cluster id, so the
fragments `Clustering_neutrino` merged into cluster 16 stay one graph, and
`connect_graph.cxx:155-169`'s uncapped MST does the rest.

The machinery to fix this is already ported: `Cluster::connected_blobs(dv,
pcts, "relaxed")` (`Facade_Cluster.h:531-534` — "used to be called
examine_graph()"; flavor "relaxed" *is* the overclustering protection) is
what `ClusteringProtectOverclustering` uses at the clustering stage.  The
missing piece is a PR-chain visitor mirroring `WCPPID::Protect_Over_Clustering`:
after `unmerge_assoc`, split each beam-bundle cluster by relaxed-graph
connected components, keep the largest as main, spin the rest off as
associated clusters so the existing per-cluster fitting handles them.  Per
house rules it would ship as a default-OFF stage (named in `pipeline_names`
only when wanted) and needs its own validation round — **not implemented
here**; this section records the diagnosis and the fix shape.

## Artifacts

- Probe: `sbnd_xin/scripts/analysis/misc/gapjump_probe.py` (this doc).
- Fresh arms: `sbnd_xin/work-pr22gap-{input,a,b,c}` (evt 386948 only; c = updated `-nu` verification).
- Scan-set input analyzed: `sbnd_xin/work-oc19scan-old/pr_evt386948/mabc-pr.zip`.
- Related: doc pr/19 (the Bee sets), doc 50 (STM gap-jumping + MST no-cap),
  docs 45/50/51/52 (unmerge stages), doc pr/3 (the PR Bee layers).

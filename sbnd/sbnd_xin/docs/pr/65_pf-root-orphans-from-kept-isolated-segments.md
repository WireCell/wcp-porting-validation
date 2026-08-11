# doc pr/65 — 18259-54095: particle-flow root gets orphan `mu-`/`e-` fragments that belong inside the main EM shower

Status: **diagnosis only, no fix**. Round 1: root cause identified and isolated
with a single-knob A/B. Round 2: traced *why* particle-flow formation misses
these segments (the prototype's own design intention), measured that its own
absorption algorithm would in fact claim every one of them, and recommends a
three-rung fix (extend the existing absorber to reach them; extend it further
in the prototype's own vocabulary if that's not enough; stop fabricating root
nodes for whatever is still left). No code changed either round. Bee link:
https://www.phy.bnl.gov/twister/bee/set/5f25ab42-db88-46d4-a53b-0730d1b531a7/event/2/

## Symptom (owner report)

SBND 18259-54095's particle flow: `e- 2334 MeV`, `proton 284 MeV`, plus a
`mu- 96 MeV` and five `e- 9-40 MeV` nodes hanging directly off the PF root next
to the neutrino vertex, none of them actually touching it. Two things are
wrong: (1) a particle that is not connected to the neutrino vertex cannot
legitimately be a PF-root daughter; (2) the muon (and the electrons) sit
spatially inside the 2334 MeV EM shower's footprint and should have been
absorbed into it, not left standalone. Owner has seen the same pattern in
other events and suspected a recent "clustering points" fix.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin   # symlink into wcp-porting-img/sbnd/sbnd_xin

# 1. Archive bisect (no run needed -- extracts 0-mc.json from every already-
#    archived pr_evt54095/mabc-pr.zip and lists root-level node text in mtime order):
python3 - <<'EOF'
import zipfile, json, os, glob, datetime
base = os.getcwd()
rows = []
for z in glob.glob(base + '/work-*/pr_evt54095/mabc-pr.zip'):
    lbl = z.split('/')[-3]
    try:
        with zipfile.ZipFile(z) as f:
            n = [x for x in f.namelist() if x.endswith('-mc.json')][0]
            d = json.loads(f.read(n))
    except Exception:
        continue
    rows.append((os.path.getmtime(z), lbl, [x['text'].strip() for x in d]))
for mt, lbl, top in sorted(rows):
    print(datetime.datetime.fromtimestamp(mt).strftime('%m-%d %H:%M'), lbl, ' | '.join(top))
EOF

# 2. Confirmation run -- WCT_BEE_PF_PRINT=1 shows the six nodes come from the
#    orphan safety net, with no ANCHOR line ever claiming them.
mkdir -p work-pr65-pfprint/ql_evt54095
cp work-nuecc48-cb0805/ql_evt54095/pctree-evt54095.tar.gz work-pr65-pfprint/ql_evt54095/
SBND_WORK_ROOT=$PWD/work-pr65-pfprint \
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod \
WCT_BEE_PF_PRINT=1 ./run_pr_evt.sh mc -nu 35
grep -n "ADD orphan-track-root\|ANCHOR orphan" work-pr65-pfprint/pr_evt54095/wct_pr_evt54095.log  # prints to stdout, not the file -- redirect stdout when re-running

# 3. Single-knob A/B: other_seg_keep_isolated=false, everything else at production.
#    (No SBND_OTHER_SEG_KEEP_ISOLATED env hook is wired into wct-pr-perevt.jsonnet
#    despite the comment at line 1305 -- the runner has no --tla passthrough for it.
#    Isolated the knob by a temporary literal edit at line 1309, ran, reverted
#    immediately; git diff on the file is empty before and after.)
mkdir -p work-pr65-osoff/ql_evt54095
cp work-nuecc48-cb0805/ql_evt54095/pctree-evt54095.tar.gz work-pr65-osoff/ql_evt54095/
SBND_WORK_ROOT=$PWD/work-pr65-osoff \
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod \
./run_pr_evt.sh mc -nu 35   # (with other_seg_keep_isolated temporarily false in wct-pr-perevt.jsonnet:1309)
grep -c "pr54 keep-isolated" work-pr65-osoff/pr_evt54095/wct_pr_evt54095.log   # 0

# 4. Generality scan across the archived nueCC48 set (read-only, no re-run):
#    root nodes whose 'start' != the most common start (the neutrino vertex).
```

Code cited below:
```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
sed -n '704,771p'  clus/src/NeutrinoShowerClustering.cxx   # distance-only absorber, main_cluster guard :723
sed -n '772,846p'  clus/src/NeutrinoShowerClustering.cxx   # angle+distance absorber, main_cluster guard :801
sed -n '845,874p'  clus/src/NeutrinoShowerClustering.cxx   # orphan-sibling adopt, main_cluster guard :860
sed -n '44,88p'    clus/src/NeutrinoShowerClustering.cxx   # update_shower_maps -- used_shower_clusters rebuild
sed -n '620,745p'  clus/src/NeutrinoOtherSegments.cxx      # other_seg_keep_isolated keep branch, add_segment :737
sed -n '2190,2245p' clus/src/PRSegmentFunctions.cxx        # segment_do_track_pid -- 20 cm electron-hypothesis cut
sed -n '1121,1145p' clus/src/MultiAlgBlobClustering.cxx    # fill_bee_pf_tree entry
sed -n '1961,2022p' clus/src/MultiAlgBlobClustering.cxx    # flat orphan safety net (pf_shower_vertex_barrier)
sed -n '1558,1620p' clus/src/MultiAlgBlobClustering.cxx    # pf_orphan_track_parentage anchoring pass
sed -n '1305,1311p' cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet  # other_seg_keep_isolated = true (SBND prod)
```

## Regression bisect

Extracting `0-mc.json` from every archived `work-*/pr_evt54095/mabc-pr.zip` and
listing root-level node text in mtime order (2026-08-05 → 08-11, over 100
arms) gives a clean transition:

| label (mtime) | root-level nodes |
|---|---|
| everything 08-05 → `work-pr54-off48a` 08-09 10:32 | `e- 2599 \| gamma 15 \| proton 284 \| pi+ 1` — clean |
| **`work-pr54-on48a` 08-09 10:55** | same 4 **+ 7 stray `e-`** |
| `work-pr56r3-final-cen48` 08-09 22:19 | 3 + **10 stray incl. `mu- 38`, `mu- 43`** |
| `work-pr62-on48` 08-10 22:13 → `work-pr63-cur48` (current production) | 3 + **`mu- 96` + 5 stray `e-`** ← the owner's Bee link |

`other_seg_keep_isolated` (doc pr/54, **SBND production ON since 2026-08-09**)
is the origin of this node class. Everything downstream of it in the bisect —
pr/56 round 3, pr/62's S7 long-edge corridor — only changes *which* fragments
end up stray, because they change upstream clustering/connectivity, not the
mechanism that strands them. They are not the cause.

Log confirmation, same archives: `work-pr54-off48a` has **zero**
`pr54 keep-isolated` lines for this event; `work-pr63-cur48` has **eight**,
all in cluster 17 (the main cluster). The first one —

```
cluster 17 n_points=93 length=31.95 cm v1=(198.3,29.1,267.7) v2=(199.8,27.5,289.1)
```

— matches the `mu- 96 MeV` node's endpoints exactly (`(198.58,28.74,266.96)` →
`(199.28,28.18,289.41)`, same segment after the subsequent track fit).

## Root cause: a broken graph-connectivity invariant, not a broken absorber

`other_seg_keep_isolated` (`clus/src/NeutrinoOtherSegments.cxx:729-745`) keeps
a residual segment that failed every other reattachment attempt and — in its
own comment — adds it "as a disconnected piece of this cluster's graph":

```cpp
// clus/src/NeutrinoOtherSegments.cxx:737
add_segment(graph, new_seg, v1, v2);
```

`v1`/`v2` are two vertices created fresh for this segment; nothing connects
them to the rest of cluster 17's PR graph. Every downstream shower-absorption
mechanism assumes the main cluster's graph is connected — true in the
prototype, which never had this keep — and this knob breaks that assumption
for exactly the segments it keeps. Four independent mechanisms then fail in
sequence for the same reason:

1. **Shower seeding never reaches them.** `shower_clustering_with_nv_in_main_cluster`
   (`NeutrinoShowerClustering.cxx:90`) BFSes out of `main_vertex` over the main
   cluster's graph; a disconnected component is unreachable, so no `Shower` is
   ever seeded on it — even though 5 of the 6 stray fragments in this event
   carry `pdg 11` (already shower-flagged) at this point.
2. **The two distance/angle absorbers exclude every main-cluster segment by
   construction**, not just the disconnected ones:
   `if (seg1->cluster() == main_cluster) continue;` at
   `NeutrinoShowerClustering.cxx:723` (pure-distance fallback, `<3.5 cm`) and
   `:801` (angle+distance, the `(θ<25°,d<80cm) ∨ …` cuts). That guard is
   prototype-faithful *because* the prototype's main-cluster segments are
   always flood-fill reachable; under pr/54 that is no longer true, so the
   guard now also excludes segments that genuinely need this rescue.
3. **The orphan-sibling adopt pass excludes them too** (`:860`, same
   `main_cluster` guard), and so does `shower_clustering_in_other_clusters` —
   cluster 17 is permanently in `used_shower_clusters` once any one of its
   segments is absorbed (`update_shower_maps`, `:44-88`, rebuilds
   `used_shower_clusters` from `map_segment_in_shower` on every call), so the
   whole-cluster promotion path in `shower_clustering_in_other_clusters` never
   even considers cluster 17.
4. **They survive to `fill_bee_pf_tree`** (`MultiAlgBlobClustering.cxx:1121`),
   fail the track BFS out of `main_vertex` (their component touches nothing
   reachable), fail `pf_orphan_track_parentage` anchoring (`:1558-1620` —
   neither endpoint is ever in `vtx_incoming_seg` or `vtx_to_parent_shower`,
   because the component is isolated by construction), and land in the
   **flat orphan safety net** (`:1961-2022`), which emits every BFS-unreached,
   same-cluster, `dirsign!=0` segment as a **root-level leaf**.

### Confirmation

`WCT_BEE_PF_PRINT=1` on this event (`work-pr65-pfprint`) prints exactly:

```
[fill_bee_pf_tree] ADD orphan-track-root  seg=17019  name=mu-  ke=96 MeV  cluster=17
[fill_bee_pf_tree] ADD orphan-track-root  seg=17022  name=e-   ke=23 MeV  cluster=17
[fill_bee_pf_tree] ADD orphan-track-root  seg=17050  name=e-   ke=40 MeV  cluster=17
[fill_bee_pf_tree] ADD orphan-track-root  seg=17059  name=e-   ke=13 MeV  cluster=17
[fill_bee_pf_tree] ADD orphan-track-root  seg=17060  name=e-   ke=13 MeV  cluster=17
[fill_bee_pf_tree] ADD orphan-track-root  seg=17068  name=e-   ke=9  MeV  cluster=17
```

with **zero** `ANCHOR orphan seg=...` lines for any of these ids — the
anchoring pass never finds a claimed neighbor for them, exactly as predicted
by the connectivity break.

**Single-knob A/B** (`work-pr65-osoff`, `other_seg_keep_isolated` forced
`false` for this one run, everything else at production, config reverted
immediately after): the six stray roots disappear entirely (mc.json root list
drops to `e- | gamma | proton`) and `pr54 keep-isolated` goes from 8 lines to
0. This isolates the mechanism to the knob directly, independent of the
chronological bisect. (The main shower's total reconstructed energy also
drops substantially in the off arm — expected and consistent with doc pr/54's
own stated purpose, injecting real charge that used to be silently discarded
into the reconstruction; not further investigated here, out of scope.)

## Answers to the owner's two questions

- **"How can it connect to the PF root if it's not connected to the neutrino
  vertex?"** It doesn't — the root placement is a *display fallback*, not a
  reconstructed relationship. `MultiAlgBlobClustering.cxx:1961`'s safety net
  reproduces the prototype's `mc_mother = 0` default for segments the
  mother-assignment BFS never reaches (`NeutrinoID.cxx:1485-1489`), and that
  fallback is itself intentional design (doc pr/38) for a case the prototype
  could only hit on genuinely-BFS-unreachable, but still main-cluster-graph-
  connected, segments. pr/54 puts main-cluster segments into the graph that
  are not graph-connected at all, a state the fallback's design never
  contemplated.
- **"Why isn't the muon absorbed into the 2334 MeV shower?"** Because every
  absorption path here is graph-connectivity-driven (BFS reachability or an
  explicit `cluster() == main_cluster` distance/angle sweep that assumes
  reachability), and pr/54's keep is disconnected from that graph by
  construction.

## Round 2 — why particle-flow formation misses them: the design intention

Round 1 stopped at "connectivity is assumed and pr/54 breaks it." The owner
asked a sharper question: *why does PF formation skip these segments at all,
by both the prototype's and the toolkit's own design* — because the answer
determines whether the right fix is to patch the display, patch the source,
or extend the clustering algorithm. This round reads the prototype's shower
clustering and mother-assignment code directly to answer that, then measures
whether its own absorber would in fact take these fragments.

Repro (archive-only, no wire-cell run — reconstructs the cluster-17 PR graph
and the absorption test purely from `0-track_fit-global.json` and
`0-vertices-global.json` already inside `work-pr65-pfprint/pr_evt54095/mabc-pr.zip`):

```bash
cd /home/xqian/tmp && mkdir -p pr65r2 && cd pr65r2
unzip -o -q .../work-pr65-pfprint/pr_evt54095/mabc-pr.zip -d cur
python3 - <<'EOF'
import json, math
from collections import defaultdict, deque
d = json.load(open('cur/data/0/0-track_fit-global.json'))
v = json.load(open('cur/data/0/0-vertices-global.json'))
rid = d['real_cluster_id']
segs = {}
for i, s in enumerate(rid):
    if s == -1: continue
    segs.setdefault(s, []).append((d['x'][i], d['y'][i], d['z'][i]))
V = [(v['x'][i], v['y'][i], v['z'][i], v['real_cluster_id'][i])
     for i in range(len(v['x'])) if v['real_cluster_id'][i] != -1]
c17 = sorted(s for s in segs if 17000 <= s < 18000)
adj, segv = defaultdict(set), {}
for s in c17:
    ev = [min((math.dist(e, (a,b,c)), vid) for a,b,c,vid in V)[1]
          for e in (segs[s][0], segs[s][-1])]
    segv[s] = tuple(ev)
    adj[ev[0]].add(s); adj[ev[1]].add(s)
seen, comps = set(), []
for s in c17:
    if s in seen: continue
    q, comp = deque([s]), set()
    while q:
        cur = q.popleft()
        if cur in comp: continue
        comp.add(cur); seen.add(cur)
        for vv in segv[cur]:
            for nb in adj[vv]:
                if nb not in comp: q.append(nb)
    comps.append(sorted(comp))
for i, c in enumerate(sorted(comps, key=len, reverse=True)):
    print(i, len(c), c)
EOF
```

### 1. The prototype's intention: "main cluster" is shorthand for "graph-reachable"

Shower absorption comes in exactly two flavours, partitioned on cluster
identity:

- **Graph-driven, for the main cluster.** `shower_clustering_with_nv_in_main_cluster`
  (`prototype_base/pid/src/NeutrinoID_shower_clustering.h:1654`, ported at
  `NeutrinoShowerClustering.cxx:90`) walks outward from `main_vertex` through
  `map_vertex_segments`, seeds a `WCShower` at the first shower-flagged
  segment reached, and `complete_structure_with_start_segment` absorbs the
  rest of that branch. **Reachability is the membership rule.**
- **Proximity/angle-driven, for everything else.** The 3.5 cm closest-distance
  fallback and the 25°/80 cm ∨ 12.5°/130 cm ∨ 5°/200 cm cone absorber in
  `shower_clustering_with_nv_from_main_cluster`
  (`NeutrinoID_shower_clustering.h:1870`, `:1901`), the same two tests again
  in `shower_clustering_with_nv_from_vertices` (`:1276`, `:1346`), and
  whole-cluster attachment in `shower_clustering_in_other_clusters` (`:1514`,
  which builds the shower with `connection_type = 3`).

**All five proximity paths open with**
**`if (seg1->get_cluster_id() == main_cluster->get_cluster_id()) continue;`.**
The toolkit reproduces every one of them:
`NeutrinoShowerClustering.cxx:723`, `:801` (the two already cited in round 1),
`:860` (orphan-sibling adopt, toolkit-only), `:1221`, `:1297`, `:1581`.

The guard is not a physics statement — it's redundancy elimination. A
main-cluster segment is *already* handled by the graph walk, so re-testing it
by distance would be wasted work and risks double-claiming it. The invariant
that makes skipping it safe is **the main cluster's PR graph is connected**.

### 2. The prototype enforces that invariant exactly where pr/54 relaxed it

`find_other_segments` runs inside `find_proto_vertex`
(`NeutrinoPatternBase.cxx:2496`), long before any shower clustering. Its
Step-9 isolated-residual branch (`NeutrinoOtherSegments.cxx:640-745`, cited in
round 1) has exactly two legal outcomes in the prototype: **attach** — snap
onto an existing vertex/segment within a strict isochronous window — or
**discard** — `remove_vertex` on both fresh endpoints. There is no third
outcome. pr/54 added one, in one line
(`add_segment(graph, new_seg, v1, v2)`, `:737`) — "add as a disconnected piece
of this cluster's graph", per its own comment.

### 3. The root node is prototype behaviour, not a toolkit invention

Once a main-cluster segment is graph-unreachable it falls into a gap no path
covers: too "main cluster" for every proximity absorber, unreachable for the
graph walk. It is never placed in any `WCShower`.

`fill_particle_tree` (`prototype_base/pid/src/NeutrinoID.cxx:1485-1489`) gives
**every** non-shower main-cluster segment a node with `mc_mother = 0` — the
prototype's own debug print hardcodes `parent=ROOT` — and only *afterwards*
does the mother-assignment BFS (`:1596-1626`) overwrite `mc_mother` for
segments it reaches. Anything the BFS never reaches keeps `mc_mother = 0`.
`MultiAlgBlobClustering.cxx:1961-2022`'s orphan safety net is a faithful port
of exactly this, and its own comment says so. The prototype has the identical
failure mode; it simply can never enter it, because `find_other_segments`
discards the only thing that could produce a disconnected main-cluster
component.

Corollary: the prototype's proper representation of "EM charge spatially near
but not graph-connected" already exists — a `connection_type = 2/3` shower,
rendered by `fill_particle_tree` as a **pseudo-gamma** carrier node under the
real parent (`NeutrinoID.cxx:1668-1690`; toolkit `append_pseudo_shower`,
`MultiAlgBlobClustering.cxx:1735`). The machinery to place these fragments
correctly is already built and already exercised elsewhere in this same
pipeline — the fragments produced by `other_seg_keep_isolated` are just never
routed into it.

### 4. The measurement: cluster 17's graph is shattered into 9 pieces, and every orphan is inside the shower cone

Reconstructing the cluster-17 PR graph directly from the archive (segment
endpoints matched to listed vertices — every match lands at **exactly
0.000 cm**, so the edge list is exact, not inferred) gives **40 segments in 9
connected components**:

| component | segs | fit pts | contains `main_vertex` (17007) | PF-visible strays (round 1) |
|---|---|---|---|---|
| 0 | 28 | 596 | **yes** | — |
| 1 | 4 (17058, 17059, 17060, 17068) | 50 | no | 17059, 17060, 17068 |
| 2 | 2 (17019, 17053) | 64 | no | 17019 (`mu- 96 MeV`) |
| 3-8 | 1 each (17022, 17023, 17050, 17051, 17052, 17062) | 120 | no | 17022, 17050 |

**8 orphan components — exactly one per `pr54 keep-isolated` log line** for
this event (8 lines, all cluster 17, round 1). Components 1 and 2 grew extra
segments because subsequent track-breaking split the originally-kept segment;
the children inherit the disconnection.

Two consequences round 1 didn't have:

- **The orphan set is twice what the PF tree shows.** 12 segments / 234 fit
  points (28% of cluster 17's 830) are graph-unreachable. Six surface as root
  nodes (round 1); the other six — 17023, 17051, 17052, 17053, 17058, 17062 —
  are dropped from the tree entirely by the orphan net's own filters
  (`dirsign()==0`, empty fits, KeepMC floors). Their charge isn't misplaced in
  the particle flow — it's absent from it.
- **Every one of the 12 satisfies the prototype's own cone absorber.**
  Neutrino vertex `(175.91, 26.89, 186.94)` cm (the 17038/17063 junction);
  shower axis taken shower-wide over 60 cm — the branch the prototype selects
  for `total_length > 100 cm`
  (`NeutrinoID_shower_clustering.h:1826-1830`) — → `(0.230, 0.002, 0.973)`;
  `angle_offset = 0` (axis sits 13° off drift-perpendicular, threshold 5°).

| seg | in PF tree | fit pts | min dist to component 0 | dist to vertex | angle to shower axis | absorbed? |
|---|---|---|---|---|---|---|
| 17019 `mu- 96 MeV` | yes | 49 | 1.34 cm | 83.84 cm | 2.6° | yes (12.5°/130cm) |
| 17022 `e- 23 MeV` | yes | 23 | 2.36 cm | 72.65 cm | 0.7° | yes |
| 17050 `e- 40 MeV` | yes | 20 | 2.68 cm | 65.68 cm | 8.0° | yes |
| 17059 `e- 13 MeV` | yes | 13 | 1.81 cm | 67.94 cm | 1.2° | yes |
| 17060 `e- 13 MeV` | yes | 16 | 3.06 cm | 64.03 cm | 6.7° | yes |
| 17068 `e- 9 MeV`  | yes | 12 | 1.41 cm | 63.21 cm | 2.1° | yes |
| 17023 | no | 18 | 4.76 cm | 48.38 cm | 7.5° | yes |
| 17051 | no | 26 | 2.18 cm | 56.99 cm | 10.9° | yes |
| 17052 | no | 18 | 2.44 cm | 60.09 cm | 5.3° | yes |
| 17053 | no | 15 | 2.00 cm | 83.84 cm | 2.6° | yes |
| 17058 | no | 9 | 2.01 cm | 67.89 cm | 0.7° | yes |
| 17062 | no | 15 | 0.55 cm | 50.45 cm | 2.7° | yes |

Every orphan is **0.55-4.76 cm** from the reachable graph and **0.7°-10.9°**
off the shower axis. The prototype's own absorber, with its own already-tuned
thresholds, would claim all twelve into the 2331 MeV shower. **The only thing
stopping it is `seg1->cluster() == main_cluster`.**

Robustness check: recomputing the axis from the first 15 cm of start segment
17038 instead of the 60 cm shower-wide average gives `(0.238, 0.008, 0.971)`
and shifts every angle by <2° — no row changes verdict, and `angle_offset`
stays 0 under both.

## Secondary finding: why an EM fragment is typed `mu-`

`segment_do_track_pid` (`clus/src/PRSegmentFunctions.cxx:2222-2243`) only
admits the electron hypothesis when `length < 20 cm`; the default particle
type is `13` (muon). The kept fragment matching the `mu- 96 MeV` node is
**31.95 cm** (`pr54 keep-isolated` log line above), so `e-` was never reachable
for it regardless of dQ/dx. Every `mu- NN MeV` stray node observed in this
class across the bisect (`mu- 38`, `mu- 43`, `mu- 96`, `mu- 44`, `mu- 5`) is an
isolated EM fragment whose fitted length happens to clear the 20 cm cut — a
PID artifact of the disconnection, not an independent bug.

## Generality (archived nueCC48 set, read-only — no fresh campaign run)

Root-level nodes whose `start` is not the neutrino vertex, across the 47/24
archived `pr_evt*` dirs in each labeled arm:

| arm | events | events with stray roots | stray nodes | `pr54 keep-isolated` lines |
|---|---|---|---|---|
| `work-pr54-off48a` (pre-pr/54) | 47 | **1** (2 `gamma`, a different pre-existing conn-2/3 case, see below) | 2 | 0 |
| `work-pr63-cur48` (current production) | 24 | **16** | 85 | 59 |

This confirms the owner's "I've seen this in other events": roughly two-thirds
of nueCC events develop this class of PF-root orphan under current production,
versus essentially none before pr/54. (These two arms differ in more than
`other_seg_keep_isolated` — bisect table above shows the class first appearing
at exactly the pr/54 transition, so the knob is the origin; other in-between
changes reshape which fragments end up stray without introducing or removing
the class.)

The `work-pr54-off48a` `gamma` stray-root pair is a **different, pre-existing**
case unrelated to this bug: a shower whose `start_vtx` is a legitimate
conn-type 2/3 attachment point that the graph-connectivity BFS itself does not
reach for an unrelated reason (`MultiAlgBlobClustering.cxx:1501-1523`, the
"fallback: start_vtx truly isolated" branch for *showers*, which is ungated by
any knob). Not counted against the `other_seg_keep_isolated` total above.

## Recommended fix (design only, NOT implemented) — round 2

The owner set two requirements on whatever fix this doc recommends:

> 1. we should not add arbitrary particles to the root node of particle flow.
> 2. they should show up in the particle flow the normal way through the
>    existing clustering algorithm. If that is not sufficient, we should
>    update the algorithm.

These are two separate defects, and the recommendation treats them as such:
the root emission is a **hygiene** bug (a fabricated particle-flow node), and
the non-absorption is the **algorithmic** bug (real charge that the existing
absorber would take but never gets offered). Fixing only the first hides the
second — a "clean" tree that silently drops 234 fit points would satisfy
requirement 1 while failing requirement 2. The recommendation is a **ladder**,
each rung behind its own default-OFF knob so a mover census can attribute
each rung's effect independently:

### Rung 1 — make the existing algorithm see them ("the normal way")

Relax the guard from cluster identity to reachability, which is what it was
written to mean (§1 above). Compute the `main_vertex`-connected component once
in `shower_clustering_with_nv`, thread it through, and change each of the six
sites from `if (seg1->cluster() == main_cluster) continue;` to
`if (seg1->cluster() == main_cluster && root_reachable.count(seg1)) continue;`.

No new code path, no new physics constant — the 3.5 cm and cone thresholds are
the prototype's own, already tuned. On a connected main cluster the predicate
is identical to today's, so **knob-off is byte-identical by construction**.
Measured reach on 18259-54095: **12 of 12** orphan segments claimed (§4
table), including the six the PF tree never showed at all. An absorbed
segment becomes a shower member — it carries its charge into the shower and
stops being a standalone node, which also moots the `mu-` typing of 17019
(next section).

### Rung 2 — if rung 1 leaves an orphan unclaimed, extend the algorithm in the prototype's own vocabulary

The prototype already has a representation for "charge that is spatially near
but not graph-connected" (§3 corollary): register the unclaimed component as
its own unit with a local main vertex, and let
`shower_clustering_in_other_clusters` attach it with `connection_type = 3`
(`NeutrinoID_shower_clustering.h:1442-1514`), which `fill_particle_tree`
renders as a pseudo-gamma carrier under the real parent — a *reconstructed*
relationship with a recorded connection type, not a placement of last resort.
Concretely: treat each remaining disconnected component of the main cluster as
an "other cluster" for that pass — give it a local main vertex via the same
`determine_main_vertex` call the real other-clusters loop already makes
(`TaggerCheckNeutrino.cxx:1069-1105`), and add it to the map the pass
iterates.

Whether rung 2 is needed at all is an empirical question, not assumed here: on
18259-54095, rung 1 alone already covers everything (§4 table, 12/12), so rung
2 is specified but **unexercised** by this event. A future implementing round
needs a case where the cone absorber genuinely rejects a kept-isolated
fragment (too far, or too far off-axis) to exercise it.

### Rung 3 — root hygiene: an unclaimed orphan must never become a root-level particle

The flat orphan net (`MultiAlgBlobClustering.cxx:1961-2022`) exists as doc
pr/38's "no-silent-drop complement of the barrier", and is a faithful port of
the prototype's `mc_mother = 0` default — but that default was only ever
reachable, in the prototype, for segments the mother-BFS could reach (§3);
porting it faithfully imported a guarantee the prototype never actually had to
honour. Fix: keep the *visibility* the net was added for, drop the
*fabrication*. Replace node emission with an unconditional counter plus a log
line naming each still-unclaimed segment (`PortAuditCounters`, `PRGraph.h:39-68`,
is the existing home for this kind of counter). No fabricated particle, no
silent drop — the residue shows up in the audit line instead of the physics
tree.

This makes rungs 1-2 **load-bearing**: applying rung 3 alone would make the
fragments simply vanish from the tree, satisfying requirement 1 while
regressing requirement 2 to worse than today (today at least shows the
charge, if mislabeled). This is why the three are presented as one ladder to
be applied together, not three independent alternatives to pick from.

### Open questions for the implementing round (stated as open, not assumed here)

- **Energy.** Is the orphan charge already inside the reported 2331 MeV
  figure? If yes, today double-counts (the six visible strays alone add
  194 MeV on top) and the fix removes a duplicate; if no, the fix *moves*
  charge into the shower and its reported energy rises. Both readings are
  live — the 0.55-4.76 cm proximity measured in §4 does not distinguish them.
  This decides whether rung 1 is a neutral cleanup or a shower-energy change,
  and must be checked before any flip.
- **Is `other_seg_keep_isolated` the only producer of disconnected
  main-cluster components?** Measured 8-for-8 against this event's
  keep-isolated log lines, and 0 orphan-net nodes in the 47 pre-pr/54 events
  (round 1's generality scan) — but that is one event and one sample. The
  claim "rung 1 is a no-op wherever nothing was kept isolated" is
  *conditional* on this holding generally.
- **Graph topology stability.** Confirm no shower-clustering step mutates PR
  graph topology between where the reachable component is computed and the
  last of the six guard sites — otherwise "compute once and thread through"
  needs a different anchor point.
- **`used_shower_clusters`.** Confirm rung 1 does not cause `main_cluster` to
  gain an entry in `used_shower_clusters` that changes
  `shower_clustering_in_other_clusters` behaviour for other clusters.

**Named consequence, not a caveat to bury:** the cone absorber tests
**geometry only** — there is no `flag_shower` check on `seg1`. Rung 1
therefore also absorbs a genuinely disconnected main-cluster *track* segment
that happens to lie in the cone, folding its charge into the shower. This is
the same rule already applied to every other (non-main) cluster today, so it
is *consistent*, not novel — but it is a real behaviour change beyond the six
segments in this event, and is exactly what a default-OFF knob plus a mover
census across the nueCC48/mcp1k50 manifests is for.

## Options considered and rejected

Round 1 listed three options neutrally. Round 2 records why two of them fail
the owner's stated requirements, and keeps them here as fallbacks rather than
recommendations:

- **`fill_bee_pf_tree` nearest-shower re-parent (display-only).** Satisfies
  requirement 1 (nothing left at the root) but **fails requirement 2**: the
  attachment is invented at write-out time, not reconstructed by the
  clustering algorithm — the shower's kinematics never change, and the six
  PF-invisible orphans (17023/17051/17052/17053/17058/17062) stay invisible.
  A better-looking hack, not an algorithm fix. Rejected as primary.
- **Source-side attach-or-drop restoration in `find_other_segments`.** Most
  conservative — literally restores the prototype's two-outcome invariant
  (§2) — but it forfeits pr/54's own gain exactly where no valid attachment
  exists, i.e. it re-deletes the same real trajectories the owner called
  good in the original pr/65 report. Rejected on that basis; kept as the
  fallback if the ladder's mover census turns out unacceptable.

## Status

Diagnosis only, both rounds. No code changed. No config changed (the one
jsonnet edit made for round 1's A/B was reverted before commit — `git diff` on
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` is empty). Round 2's
graph reconstruction and absorption-cone measurement are read-only, derived
entirely from data already inside `work-pr65-pfprint/pr_evt54095/mabc-pr.zip`
— no new work dir, no new run, nothing under `work/`, `abtest/snap/`, or
`decisions*` touched (M13). Nothing to revalidate; the recommended ladder
(rungs 1-3) is a design for a future implementing round, not shipped here.

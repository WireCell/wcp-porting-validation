# doc pr/65 — 18259-54095: particle-flow root gets orphan `mu-`/`e-` fragments that belong inside the main EM shower

Status: **diagnosis only, no fix**. Root cause identified and isolated with a
single-knob A/B; no code changed. Bee link:
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

## Fix options — NOT implemented (diagnosis only, per request)

Listed for the owner to choose from; each would ship as a default-OFF knob
per project policy, and would need its own byte-identical gate:

1. **Narrowest — relax the three main-cluster absorber guards.** Let
   `shower_clustering_with_nv_from_main_cluster`'s two absorbers (`:723`,
   `:801`) and the orphan-sibling adopt (`:860`) admit a main-cluster segment
   that is graph-disconnected from `main_cluster`'s BFS component, while
   keeping the existing guard for genuinely-connected segments (where it is
   prototype-faithful and must stay untouched). Fixes the root cause where it
   originates; touches the highest-traffic clustering code.
2. **Display-only — anchor unclaimed orphans by distance in the PF tree.**
   Have `fill_bee_pf_tree`'s `pf_orphan_track_parentage` pass (or a new knob
   alongside it) fall back to nearest-shower-by-distance when graph anchoring
   finds nothing, instead of the flat safety net. Does not change any
   reconstructed physics quantity (shower composition, kinematics), only how
   the existing tree is displayed/nested. Safest, narrowest blast radius, but
   leaves the underlying non-absorption (and its correctness for kinematics)
   unaddressed.
3. **Prevention at the source — bound `other_seg_keep_isolated` by proximity.**
   Only keep a residual segment when it lies within some distance of the
   existing graph component, instead of unconditionally. Prevents the
   disconnected-graph state from being created at all; narrows pr/54's own
   recovery (fewer segments kept) as a tradeoff.

No option above is implemented in this investigation.

## Status

Diagnosis only. No code changed. No config changed (the one jsonnet edit made
for the A/B in this investigation was reverted before commit — `git diff` on
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` is empty). Nothing to
revalidate.

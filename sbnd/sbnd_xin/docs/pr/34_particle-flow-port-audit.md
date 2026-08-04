# doc pr/34 — Particle flow: prototype ↔ toolkit fidelity audit

**Why.** Step 6 of the eight PR stages listed in doc pr/27 §0. The neutrino
vertex, the segments, the PID and the showers are all settled by now; this stage
turns them into a *parent–child particle tree* rooted at the neutrino vertex —
the thing the owner actually reads in the Bee particle-flow panel when judging an
event.

**Status.** **AUDIT ONLY. No code was changed.** Every item below is reported,
none is fixed. Ranked list in §8.

> **§10 added — owner filter, 14 → 5.** The owner asked for the P-list narrowed
> to *bugs and things missing from the port*, dropping everything where the
> toolkit improves on the prototype, with a proposed fix per survivor. **§10** is
> that filter, re-verified at toolkit **`407c5ba9`**. Survivors:
> **F1**=P1, **F2**=P2, **F3**=P3+P11 (one defect, not two), **F4**=P10,
> **F5**=P12 — **five findings, five knobs**. **P9 and P13 are RESOLVED, not
> merely dropped** (§10.7), closing loose ends §7.2, §7.4 and §7.6. Dropped as
> improvements: P4, P5, P6, P7, P8, P14. Still **zero code changes**; §10.6
> states the gate these knobs would need, which is **not** the gate the earlier
> rounds used. **§10.9 is RETRACTED** — the `enumerate_idents` no-op it reported
> was a false negative from grepping a member name instead of a config key.

**Headline.** The tree-*shaping* logic is a reasonable reimplementation, and two
of its departures from the prototype are repairs of genuine prototype bugs. But
the toolkit's track walk is **missing the prototype's main-cluster filter**
(P1) and **missing the prototype's shower-vertex barrier** (P2), so its track set
is a superset of the prototype's in two independent ways — visible in the evt-388
tree, which contains nodes from clusters 23, 28 and 81. Two further items change
which particle a shower hangs from (P3, P4), and one splits a π⁰ into two π⁰
nodes when its daughters land on different parents (P10).

**A severity note this round, unlike pr/28–pr/33.** Every earlier round's P-list
changed reconstruction output. **This stage is display-only.** Its single
non-display side effect in the prototype — `sg->set_kine_charge(...)` — writes to
a field that nothing in the prototype ever reads (§5.7). Nothing here feeds the
BDTs, the energy sum, or the selection. The §8 table therefore carries a
**class** column — *port defect* / *deliberate* / *prototype bug not reproduced*
/ *cosmetic* — because several of the strongest findings are the toolkit being
**better** than the prototype, and the P-numbering must not imply otherwise.

**The stage counterpart spans two prototype programs.** The toolkit's
`fill_bee_pf_tree` collapses what the prototype does in two places: the reco-tree
fill (`pid/src/NeutrinoID.cxx`) and the Bee JSON conversion
(`bee/WCReader.cc`). Both are audited. See §0.

**Provenance warning — read this before trusting any prototype citation, here or
in pr/28–pr/33.** `prototype_base/pid` is a git submodule checked out on branch
**`port`** at `53ca938`, which has diverged from its upstream merge-base
`a5fc0b9` by **+5833/−989 lines across 26 files**. That set includes **every
file this audit series cites** — `NeutrinoID.cxx` (204),
`NeutrinoID_shower_clustering.h` (162, pr/33), `NeutrinoID_track_shower.h` (78,
pr/31 and pr/33's P1), `NeutrinoID_proto_vertex.h` (287, pr/30),
`NeutrinoID_improve_vertex.h` (191, pr/32), `PR3DCluster_steiner.h` (294,
pr/29), `PR3DCluster_dQ_dx_fit.h` (297) and `PR3DCluster_multi_track_fitting.h`
(318, both pr/28), `ProtoSegment.cxx` (20, pr/33's P6).

Much of that is instrumentation — the `[fill_particle_tree] …` `std::cout`
blocks in the functions audited here are **our own**, not prototype code — but
not all of it is: `08f229a` ("fix T_tagger nu_x/y/z garbage values in prototype
port") and `3dc01ac` ("fix null-pointer crash in determine_overall_main_vertex")
are algorithm edits, and one restructuring is plainly visible in the diff (the
`flag_main_cluster` / `determine_overall_main_vertex` block lifted out of an
`if`). Either way **line numbers have shifted in every cited file**.

For *this* stage I diffed the four `fill_*` functions across `a5fc0b9..HEAD`
and confirmed **every changed line is a print or a brace adjusted around one**:
the algorithm is pristine and the citations below are sound. **That check has
not been done for any earlier round.** §7.7.

---

## Repro

```bash
# toolkit revision audited
git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse --short HEAD      # 01ff88b1
git -C /nfs/data/1/xqian/toolkit-dev/toolkit show HEAD:clus/src/MultiAlgBlobClustering.cxx \
    > /home/xqian/tmp/claude-25225/pr34/MultiAlgBlobClustering.cxx       # 2683 lines

# prototype revision + the provenance check
cd /nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base
git submodule status pid          # 53ca938... pid (heads/port)
git -C pid merge-base port origin/master                                 # a5fc0b9
git -C pid diff a5fc0b9..HEAD --stat | tail -3      # 26 files, +5833 / -989  <-- SERIES-WIDE
git -C pid diff a5fc0b9..HEAD --stat -- src/NeutrinoID.cxx               # 204 changed
git -C pid diff a5fc0b9..HEAD -- src/NeutrinoID.cxx \
  | sed -n '/fill_reco_tree/,$p' | grep '^[-+]' | grep -v 'std::cout\|<<\|fpt_'
                                  # -> only braces: the fill_* algorithm is untouched

# P1's load-bearing claim, mechanically
awk 'NR>=1063 && NR<=1679' /home/xqian/tmp/claude-25225/pr34/MultiAlgBlobClustering.cxx \
  | grep -n main_cluster            # 3 hits: decl + 2 inside the flag_print block

# the emitted artifact, on a real event
python3 - <<'EOF'
import zipfile, json
z = zipfile.ZipFile('/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/'
                    'work-dqdx388-base/pr_evt388/mabc-pr.zip')
def walk(ns, d=0):
    for n in ns:
        print('  '*d, n['id'], n['text']); walk(n.get('children', []), d+1)
walk(json.loads(z.read('data/0/0-mc.json')))
EOF
```

The evt-388 tree that command prints, quoted throughout below:

```
 23014  e-    797 MeV
 23032  e-   1143 MeV
 6      gamma  13 MeV
   81082  e-   13 MeV
 7      gamma  10 MeV
   28053  e-   10 MeV
 23034  mu-   19 MeV
```

---

## §0 Scope

### What this stage is

| # | prototype | toolkit |
|---|---|---|
| — | `NeutrinoID::fill_particle_tree` `NeutrinoID.cxx:1458` | `MultiAlgBlobClustering::fill_bee_pf_tree` `MultiAlgBlobClustering.cxx:1063` |
| 1 | `fill_reco_tree(ProtoSegment*, WCRecoTree&)` `:1182` | `build_seg_node` lambda `:1584` |
| 2 | `fill_reco_tree(WCShower*, WCRecoTree&)` `:1264` | `make_shower_leaf` lambda `:1464` |
| 3 | `fill_pi0_reco_tree` `:1322` | π⁰ node inside `append_showers` `:1555-1580` |
| 4 | `fill_psuedo_reco_tree` `:1370` | `append_pseudo_shower` lambda `:1502` |
| 5 | `find_incoming_segment` `:1763` | `vtx_incoming_seg` map `:1128`, filled by the BFS |
| 6 | BFS mother/daughter linkage `:1588-1627` | BFS `:1130-1171` |
| 7 | shower linkage `:1642-1747` | shower attach `:1270-1400` |
| 8 | `bee/WCReader.cc DumpMCJSON(int,ostream&)` `:431` | `make_node` `:1417` + recursion |
| 9 | `bee/WCReader.cc DumpMCJSON(ostream&)` `:480` | assembly `:1657-1676` |
| 10 | `bee/WCReader.cc KeepMC` `:505` | `keep_node` lambda `:1451` |
| 11 | `bee/WCReader.cc PDGName` `:529` | `pf_pdg_to_name` `:1024` |
| 12 | `bee/WCReader.cc KE(float*)` `:522` | *(no counterpart — see §2.4)* |

**The two-program point matters.** In the prototype, `fill_particle_tree` fills a
flat ROOT `WCRecoTree` (`mc_id`, `mc_pdg`, `mc_mother`, `mc_daughters`,
`mc_startMomentum`, …) that is written to the output file; a **separate ROOT
macro**, `bee/WCReader.cc`, later reads that file and emits the jsTree JSON that
Bee displays. The toolkit emits the jsTree JSON directly. So the toolkit's single
function is the counterpart of *both*, and an audit that stopped at
`fill_particle_tree` would miss the display floors, the name table, the KE
formula and the ordering — four of the fourteen findings below.

`DumpMCJSON` reads exactly seven `WCRecoTree` fields: `mc_id`, `mc_pdg`,
`mc_startMomentum`, `mc_startXYZT`, `mc_endXYZT`, `mc_mother`, `mc_daughters`.
Everything else the prototype computes in these functions — `mc_included`,
`mc_dir_weak`, `mc_kine_range`, `mc_kine_dQdx`, `mc_kine_charge`, `mc_length`,
`mc_stopped`, `mc_process`, `mc_endMomentum` — **never reaches the particle-flow
tree**. It is out of scope here, and §5.10 records why that is not a gap.

### Prototype functions in these files that are *not* this stage

- `fill_proto_main_tree` `:1786` and `fill_reco_simple_tree` `:1436` — **dead**.
  Commented out at all four production call sites
  (`pid/apps/wire-cell-prod-nue.cxx:3086`, `-nue-port.cxx:3100`,
  `-nue-mt.cxx:2673`, `-pi0.cxx:2657`). §5.8.
- `fill_skeleton_info` `:2004`, `fill_skeleton_info_magnify` `:1852`,
  `fill_point_info` `:2179` — the Magnify/point outputs, audited under doc pr/7.

### Not audited

`WCShower::fill_sets`, `PR::Shower::fill_sets`, `PR::sorted_out_edges`,
`PR::ordered_nodes`, `PR::find_other_vertex`, the `Bee::ParticleTree` writer, and
`WCReader`'s non-MC dumps (`DumpSpacePoints`, the op/flash blocks). The upstream
producers of `get_showers()`, `get_pi0_showers()`, `map_shower_pio_id` and
`map_pio_id_mass` are pr/33's subject, not this one.

---

## §1 Trust tiers

Carried unchanged from pr/28 §3b through pr/33.

- **Tier A — read and cross-checked line by line**, both sides, in this session:
  `fill_particle_tree` and its four `fill_*` helpers; `find_incoming_segment`;
  `WCReader::DumpMCJSON` (both overloads), `KeepMC`, `KE`, `PDGName`,
  `IsPrimary`; the whole of `fill_bee_pf_tree` including every lambda.
  Findings P1–P14 are Tier A.
- **Tier B — read for interface and control flow only**: `ParticleInfo`
  (`aux/inc/WireCellAux/ParticleInfo.h`), `TrackFitting`'s shower accessors,
  `PRShower.h`'s container comparators, `Facade::Cluster::get_cluster_id`.
  Claims resting on Tier B are marked where they occur.

---

## §2 What matches

Establishing this first, so the divergence list is read against a mostly-correct
port rather than as a verdict on the whole.

### §2.1 The two-pass shape

Both build the tree in the same two passes: **(a)** walk track segments outward
from the neutrino vertex, breadth-first, level by level, recording each
segment's parent; **(b)** then attach showers, each under the track segment (or
shower, or the root) indicated by its `start_connection_type`. Both use a
worklist-of-(vertex, arriving-segment) pairs, both mark segments used at the
moment they are claimed rather than when expanded, and both dedupe on the
*vertex* so a segment can be claimed only once. Prototype `:1596-1627`, toolkit
`:1134-1171`.

### §2.2 The connection-type ladder

`start_connection_type` is read identically on both sides:

| type | meaning | prototype | toolkit |
|---|---|---|---|
| 1 | direct connection | shower nests straight under its parent `:1651` | `direct = (conn_type == 1)` `:1288` |
| 2 | indirect, with a gap | pseudo-particle inserted first `:1666` | `append_pseudo_shower` `:1546` |
| 3 | association | same as 2 `:1666` | same as 2 `:1288` (`!direct`) |
| 4 | not clearly connected | dropped `:1297` | dropped `:1274` |

Type 4 is a trap and is treated in §5.1 — it *looks* like the prototype keeps it.

### §2.3 The pseudo-particle rule

Both insert an unseen neutral carrier between a parent and a gap-connected
shower, with the same PDG choice: **γ (22)** when the shower is EM
(`|pdg| == 11` or `22`), **neutron (2112)** otherwise. Prototype `:1375-1379`,
toolkit `:1503-1504`. Both give it the connection vertex as its start point and
the shower's start point as its end point (prototype `:1394-1403`, toolkit
`:1507-1508`), i.e. it spans exactly the gap it exists to explain.

### §2.4 The kinetic energies displayed are the same numbers — via very different arithmetic

This is the subtlest agreement in the round and it took algebra to confirm, so it
is recorded rather than assumed.

The prototype **never stores a KE**. It stores a 4-momentum, and `WCReader::KE`
(`:522`) recovers the kinetic energy as

```cpp
TLorentzVector particle(momentum);
return particle.E() - particle.M();     // M = sqrt(E^2 - p^2), from the 4-vector itself
```

— note `M` comes from the 4-vector, **not** from a PDG mass table. Working
through each node type:

| node | prototype 4-momentum | ⇒ displayed KE | toolkit |
|---|---|---|---|
| track segment | `E = 4mom(3)`, `p⃗ = 4mom(0..2)` `:1240-1247` | `4mom(3) − m` | `pi->kinetic_energy()` `:1595` |
| shower | `E = kine_best + m(sg1)`, <br>`|p| = sqrt((kine_best+m)² − m²)` `:1305-1310` | **`kine_best` exactly** | `get_kine_best()` `:1466` |
| pseudo (EM) | `E = kine_best`, `|p| = kine_best` `:1414-1416` | `kine_best` | `get_kine_best()` `:1509` |
| pseudo (n) | `E = kine_best + m_n`, `|p| = sqrt((·)² − m_n²)` `:1419-1422` | `kine_best` | `get_kine_best()` `:1509` |
| π⁰ | `E = mass + 135 MeV`, `|p| = sqrt(E² − 135²)` `:1352-1355` | **`mass`** | `map_pio_id_mass[id].first` `:1557` |

The shower row is the one worth pausing on: the prototype builds the 4-vector
using the **start segment's** mass `m(sg1)` while the node's PDG is the
**shower's** type, which need not agree. It does not matter — `M` is recomputed
from the 4-vector, so `m(sg1)` cancels exactly and the displayed KE is
`kine_best` whatever the start segment was PID'd as. The toolkit reaches the same
number directly.

The π⁰ row is the other: `map_pio_id_mass[id].first` is the *reconstructed
invariant mass* (`NeutrinoID_shower_clustering.h:691`, `:936`), and the prototype
files it as though it were a kinetic energy (`E = mass + 135 MeV`). Odd, but
`M` comes back as exactly 135 MeV and the displayed KE is exactly `mass`. The
toolkit prints `mass` directly. **Same number, same oddity.**

Integer truncation matches too: prototype `int e = KE(...)*1000` `:436`, toolkit
`static_cast<int>(energy / units::MeV)` `:1440` under `prototype_names`.

### §2.5 Node format

Both emit `{"id":…, "text":"<name>␣␣<KE> MeV", "data":{"start":[x,y,z],
"end":[x,y,z]}, "children":[…]}` with `"icon":"jstree-file"` added when and only
when `children` is empty (prototype `:449-462`, toolkit `:1420-1434`, `:1494`,
`:1524`, `:1578`, `:1653`). Two spaces between name and number, coordinates in
cm, energies in MeV — all match.

### §2.6 Root identification

Prototype `IsPrimary(i)` is `mc_mother[i] == 0` (`WCReader.h:74`); every node is
initialised with `mc_mother = 0` and only linkage sets it otherwise. The toolkit
uses `seg_parent[seg] == nullptr` `:1667`. Same predicate, different spelling.

### §2.7 The KeepMC particle classes

Prototype `:507-517`: `{22, 11, −11}` take the EM floor, `{2112, 2212}` and
`pdg > 1e9` (nuclei) take the nucleon floor, everything else is unconditionally
kept. Toolkit `:1454-1457`: `apdg ∈ {22, 11}`, `apdg ∈ {2112, 2212}` or
`apdg > 1000000000`, else keep. The `abs` is harmless — there is no −22 and
nuclear codes are positive.

### §2.8 The π⁰ → γ → shower nesting depth

Both render a π⁰ as a three-level chain: π⁰ node → pseudo-γ → the shower leaf.
Prototype `:1712-1718` (π⁰'s daughter is the pseudo, the pseudo's daughter is the
shower start segment); toolkit `:1571-1573` (`append_pseudo_shower` into the π⁰
node's children). Same depth, same order.

### §2.9 Determinism of the graph walk

The toolkit walks with `PR::ordered_nodes` `:1100` and `PR::sorted_out_edges`
`:1138`, `:1157` (`PRTrajectoryView.h:154`), and every container it keys on
segments, vertices or showers uses an index comparator
(`SegmentIndexCmp`, `VertexIndexCmp`, `ShowerIndexCmp`, `PRShower.h:216-225`).
The prototype iterates `map_segment_vertices` — a `std::map<ProtoSegment*, …>` —
at `:1485`, i.e. **its node order is pointer-dependent**. The toolkit is strictly
better here and this is not a divergence to port back. §6 has the residual.

---

## §3 Divergences

Fourteen. Each carries a class, per the header note.

### P1 — the track walk has no main-cluster filter — *port defect*

Prototype `:1487-1488`, the two guards that open the track loop:

```cpp
if (map_segment_in_shower.find(sg)!=map_segment_in_shower.end()) continue;
if (sg->get_cluster_id() != main_vertex->get_cluster_id() ) continue;   // <-- this one
```

The toolkit ports the first and **not** the second. There is no cluster test in
the BFS seed (`:1138-1148`), none in the expansion (`:1157-1168`), and none at
assembly (`:1666-1674`). `main_cluster` *is* computed, at `:1090`:

```cpp
const auto* main_cluster = main_vertex->cluster();
```

`grep -n main_cluster` over the whole 617-line function returns exactly three
hits: that declaration, and `:1609` / `:1620` — both inside a `flag_print`
block, where it feeds a diagnostic `in_main_cluster=` field that is printed and
discarded. It is never used to filter anything.

**Consequence.** Any track segment reachable through the PR graph from the
neutrino vertex enters the tree, whatever cluster it belongs to. The prototype
restricts the track walk to the main cluster and lets only *showers* cross
cluster boundaries (its shower loop at `:1643` has no cluster filter either — see
§5.9, this asymmetry is deliberate on the prototype's part).

**Reachability is not hypothetical.** The evt-388 tree quoted in the Repro block
carries ids `23014`, `23032`, `23034` (cluster 23, the main cluster) alongside
`81082` and `28053` (clusters 81 and 28). Those two arrive as showers, so they
are legitimate on both sides — but they demonstrate the graph does span clusters
at this point in the chain, which is the precondition for the track walk to leak.

Read this as a dropped guard. The second `continue` sits directly beneath the
first, in the same loop header, and one of the two was ported.

### P2 — the shower-vertex barrier is missing from the BFS — *port defect*

The prototype pre-seeds **both** collections from every shower before walking
(`:1591-1593`):

```cpp
for (auto it = showers.begin(); it!=showers.end();it++){
  (*it)->fill_sets(used_vertices, used_segments, false);
}
```

`used_segments` stops the walk from *claiming* a shower segment; `used_vertices`
stops it from *expanding through* a shower vertex, because the frontier check at
`:1611` is `if (used_vertices.find(curr_vtx)!=used_vertices.end()) continue;`.

The toolkit pre-seeds only the segment half (`:1131`):

```cpp
PR::IndexedSegmentSet used_segs = shower_segs;   // pre-mark showers as visited
```

and starts the vertex set at `:1133` with `visited_vtxs.insert(main_vertex);` —
main vertex alone. So a vertex that lies inside a shower but is also reachable
from the neutrino vertex by a non-shower segment **will be expanded**, and any
further track segments hanging off it are pulled into the track tree. The
prototype refuses to cross that boundary.

This is independent of P1: P1 admits foreign-cluster segments, P2 admits
segments behind a shower. Either alone makes the toolkit's track set a strict
superset; both apply.

*Aside, so it is not mistaken for a third instance:* the prototype's seed loop at
`:1597-1602` pushes **every** segment at `main_vertex` onto the frontier,
including shower segments, without filtering. That looks like a leak and is not
one — the far vertex of such a segment is itself a shower vertex, pre-marked at
`:1592`, so `:1611` rejects it on the next level and no mother is ever set from
it. The toolkit skips those segments one step earlier at `:1140`. Same outcome.

### P3 — inverted precedence when a shower picks its parent — *port defect*

When a shower's start vertex is not the main vertex, both trees must decide what
that vertex hangs from. They consult the same two sources in **opposite order**.

Prototype, at all three of its sites (`:1655-1660`, `:1680-1684`, `:1720-1724`),
identical text each time:

```cpp
if (map_vertex_in_shower.find(pair_vertex.first) != map_vertex_in_shower.end()){
  prev_sg = map_vertex_in_shower[pair_vertex.first]->get_start_segment();   // shower FIRST
}else{
  prev_sg = find_incoming_segment(pair_vertex.first);                       // track second
}
```

Toolkit, `:1310-1335`:

```cpp
auto it = vtx_incoming_seg.find(start_vtx);
if (it != vtx_incoming_seg.end()) {
    ... mp[it->second].push_back({shower, start_vtx});      // track FIRST
} else {
    if (root_reachable_vtxs.count(start_vtx)) { ... }       // shower second
```

For a vertex that is *only* in a shower, or *only* on the track path, the two
agree. For a vertex that is **both** — on the BFS track path and inside a shower
— the prototype hangs the new shower under the containing shower, the toolkit
hangs it under the track segment. Different parent, different subtree, different
picture.

Whether that vertex class is populated is unmeasured; §7 keeps it.

### P4 — the fallback parent is chosen by a different rule entirely — *port defect*

Even where P3's precedence agrees, the *track* branch selects a different
segment on each side.

The prototype's `find_incoming_segment` (`:1763-1783`) picks by **fitted
direction** — the segment whose direction points *into* the vertex:

```cpp
if (flag_start && current_sg->get_flag_dir()==-1
    || (!flag_start) && current_sg->get_flag_dir()==1 ){
  sg = current_sg;
  break;
}
```

The toolkit's `vtx_incoming_seg` (`:1128`, filled at `:1146` and `:1166`) records
**which segment first reached the vertex during the BFS** — a topological fact
about distance from the neutrino vertex, with no reference to `dirsign` at all.

These coincide only when the reconstructed directions happen to point outward
from the neutrino vertex along the BFS tree. Where a segment's direction was
flipped or left weak, they diverge, and the shower moves to a different parent.

**Two prototype bugs live in that function; neither should be ported.** First,
`bool flag_start;` at `:1768` is left **uninitialized** when the vertex index
matches neither endpoint — the `if`/`else if` at `:1769-1772` has no `else` — and
`:1774` then reads it. Second, the function returns `0` when no segment qualifies,
and all three call sites use the result unchecked: `map_sg_sgid[prev_sg]` with
`prev_sg == nullptr` inserts a default `0` into the map, so the shower's mother
is set to `0` (making it a root) *and* `mc_daughters->at(map_sgid_rtid[0])`
pushes the shower onto **whatever node happens to be at index 0**. The toolkit
cannot reproduce either, having no such function.

So P4 is one divergence with two different characters: the *selection rule* is a
port defect; the *unguarded nullptr* is a prototype bug the toolkit is right not
to have.

### P5 — the prototype silently drops direction-less segments; the toolkit keeps them — *prototype bug not reproduced*

`fill_reco_tree(ProtoSegment*, …)` at `:1214`:

```cpp
if (sg->get_flag_dir()==0) return; // no direction not plot
```

`rtree.mc_Ntrack++` is at `:1260`, **after** that return. So the slot at
`mc_Ntrack` was partially written and is then never committed — the next call
overwrites it. A segment with no reconstructed direction never appears in the
prototype's tree. The toolkit's `build_seg_node` has no equivalent test; it emits
a node for every BFS-reached segment.

Three consequences inside the prototype, all bugs:

1. The caller writes `rtree.mc_stopped[rtree.mc_Ntrack-1] = …` at `:1492-1496`
   immediately after — **stomping the previous entry's** `mc_stopped`. Harmless
   for the particle-flow tree, because `DumpMCJSON` never reads `mc_stopped`
   (§0), but it corrupts the ROOT tree for anyone who does.
2. The dropped segment is absent from `map_sgid_rtid` (built at `:1569-1573`
   from committed entries only), so the linkage pass at `:1618-1620` evaluates
   `map_sgid_rtid[map_sg_sgid[curr_sg]]` on a missing key. `std::map::operator[]`
   inserts `0`. The mother is written onto **entry 0** — the first node in the
   tree, whatever it is.
3. Its children lose their parent link and become roots.

The toolkit is right not to reproduce any of this. But the tree still differs:
the prototype omits direction-less segments, the toolkit shows them. Whether the
toolkit's behaviour is *preferable* is the owner's call — a direction-less
segment arguably has no place in a directed flow tree — and it is not obviously
the same call as "don't reproduce the entry-0 corruption".

### P6 — a below-threshold node takes its whole subtree with it in the prototype, not in the toolkit — *deliberate, diverges*

`WCReader::DumpMCJSON(int id, ostream&)` opens with (`:433-434`):

```cpp
int i = trackIndex[id];
if (!KeepMC(i)) return false;
```

and a parent filters its children the same way at `:444`. There is no
"but it has children" exception. **A 4 MeV γ with a 900 MeV electron beneath it
is dropped, and the electron with it.**

The toolkit's `keep_node` (`:1451-1453`) inserts exactly that exception:

```cpp
if (cfg.em_ke_min <= 0.0 && cfg.np_ke_min <= 0.0) return true;
if (!node["children"].empty()) return true;      // <-- no prototype counterpart
```

and the in-code comment at `:1447-1450` states the intent: *"Nodes with surviving
children are always kept so the flow hierarchy stays intact."* Deliberate, and
defensible — losing a high-energy daughter because its carrier is soft is
plainly a prototype defect. It still means the two trees differ whenever a
low-energy EM or nucleon node has surviving children, which is precisely the
pseudo-γ case that this stage manufactures.

Note the interaction with §4: SBND runs with `em_ke_min = 5 MeV`,
`np_ke_min = 10 MeV` — the same floors the prototype hard-codes — so the
*thresholds* match and only the *rule* differs.

### P7 — top-level order is inverted — *port defect (display)*

Prototype `DumpMCJSON(ostream&)` `:483-496` emits primaries in `mc_Ntrack` order,
i.e. **fill order**. The fill order is set by `fill_particle_tree`: the track
loop runs first (`:1485-1511`), the shower loop second (`:1523-1565`). So the
prototype's tree lists **tracks first, showers after**.

The toolkit assembles in the opposite order (`:1657-1674`):

```cpp
append_showers(particles, root_direct_showers, root_indirect_showers, main_vertex);   // :1661
...
for (auto& [seg, parent] : seg_parent) { ... particles.append(root_node); }           // :1666
```

Visible in the evt-388 tree: two e⁻ showers and two pseudo-γ come first, the
`mu-` track last.

Cosmetic in isolation. Recorded because the owner reads these trees side by side
against prototype output, and a reordered list reads as a different result.

### P8 — disconnected main-cluster segments are dropped — *deliberate, diverges*

The prototype's track loop (`:1485`) creates a node for **every** non-shower
main-cluster segment, whether or not the later BFS ever reaches it. Unreached
segments keep `mc_mother == 0` and therefore surface as additional **top-level
primaries**.

The toolkit only ever creates nodes for segments the BFS claimed —
`seg_parent` is populated exclusively at `:1144` and `:1163` — and the comment at
`:1663-1665` says so:

> Disconnected segments (orphaned fragments unreachable from main_vertex) are now
> skipped entirely to avoid adding zero-energy orphaned particles.

"now" dates this to a deliberate change. The commented-out diagnostic block at
`:1173-1197` is its residue — it was written to enumerate exactly these
segments.

Defensible: a flow tree that lists unconnected fragments as primary particles is
misleading. But it is a real difference in what the panel shows, and it
compounds with P1 and P2 in the opposite direction (those *add* segments the
prototype excludes; this one *removes* segments the prototype includes).

### P9 — pseudo and π⁰ node ids come from a different number space — *port defect*

The prototype allocates both from the **global segment-id counter**:

- pseudo-particle: `sg1->get_cluster_id()*1000 + acc_segment_id; acc_segment_id++`
  (`:1374`)
- π⁰: `sg1->get_cluster_id()*1000 + map_shower_pio_id[shower]` (`:1324`), and
  the π⁰ ids themselves come from the same `acc_segment_id` allocator.

`acc_segment_id` is a `NeutrinoID` member initialised once (`NeutrinoID.cxx:44`),
so every synthetic node lands in the same `cluster*1000 + n` space as the real
segments and cannot collide with one.

The toolkit uses a **function-local counter starting at 1** (`:1415`):

```cpp
int next_id = 1;  // fallback counter for nodes without a natural ID
```

used for pseudo nodes (`:1510`), π⁰ nodes (`:1561`) and as the shower-leaf
fallback (`:1469`). In evt 388 that produces the ids `6` and `7` seen in the
Repro tree.

**The collision condition.** A real id is `cluster_id*1000 + seg_id`. For any
cluster with `ident ≥ 1` that is ≥ 1000, so small counter values are safe. The
one unsafe case is **a cluster whose ident is 0**, which would put real ids in
`0..999` — squarely on top of `next_id`. `Cluster::get_cluster_id()` returns
`ident()` (`Facade_Cluster.cxx:148`) and nothing in this file constrains it. In
evt 388 the smallest ident present is 23 (44 distinct clusters, 23…94), so it
does not bite there. I did not establish whether ident 0 is reachable in general
— §7 keeps it.

This also directly inherits **pr/33 P3**: `acc_segment_id` is passed by value
from 0 into the toolkit's shower stage, so `map_shower_pio_id` values are already
not the prototype's. The two findings compound.

### P10 — one π⁰ node per π⁰, or one per parent — *port defect*

The prototype memoizes on a **member** map (`:1326`, `:1361`):

```cpp
if (map_pio_id_saved_pair.find( map_shower_pio_id[shower]) == map_pio_id_saved_pair.end()){
  ...create the pi0 node...
  map_pio_id_saved_pair[ map_shower_pio_id[shower] ] = std::make_pair(...);
  return ...;
}else{
  return map_pio_id_saved_pair[ map_shower_pio_id[shower] ];   // reuse
}
```

so a π⁰ node is created **once per event per π⁰ id**, and both daughter γ hang
under that single node regardless of where each attached.

The toolkit groups in a map that is **local to one `append_showers` call**
(`:1551-1553`):

```cpp
std::map<int, std::vector<std::pair<PR::ShowerPtr, PR::VertexPtr>>> pi0_groups;
```

`append_showers` is invoked once per parent — per track segment (`:1635`), per
parent shower (`:1489`), and once for the root (`:1661`). Two showers of the same
π⁰ that attach to **different** parents therefore land in two different
`pi0_groups` and produce **two π⁰ nodes**, each with one γ beneath it and each
labelled with the full π⁰ mass.

The prototype has explicit machinery for the split-parent case — the duplicate
guard at `:1736-1737`, which checks whether the π⁰'s id is already among the
parent's daughters before pushing it. That guard exists precisely because a
single π⁰ node can be reached from more than one parent. The toolkit's structure
cannot express that: a jsTree node lives in exactly one children array.

Consequence in the panel: one π⁰ shown twice, each copy claiming the pair mass,
each missing a daughter. §7 keeps the frequency.

### P11 — the shower-parent resolution is a reimplementation, with a different failure mode — *port defect (mechanism)*

The prototype has a maintained member map, `map_vertex_in_shower`
(`NeutrinoID.h:2032`), populated during shower clustering and covering **every**
vertex of **every** shower. At fill time the lookup at `:1656` is total.

The toolkit has no such map and reconstructs an equivalent by iterating showers
to a fixed point (`:1211-1251`), classifying each shower's vertices as either
root-reachable or track-attached:

```cpp
bool any_added = true;
while (any_added) { ... for (const auto& shower : showers) { ... } }
```

The comment at `:1199-1204` is candid that this is a replication of a guarantee
rather than the guarantee itself.

**Where it fails differently.** The fixed point only ever propagates *from*
showers whose own start vertex is already resolved. A shower whose start vertex
lies inside a shower that is *itself* unresolved never enters either set, falls
through `:1310` and `:1335`, and lands in the `:1377` branch — *"start_vtx truly
isolated from main_vertex → fallback to root"* — becoming a top-level particle.
The prototype's map has no such dependency and would have found its parent.

The fixed-point loop is also **first-claim-wins** on `vtx_to_parent_shower`
(`:1236-1240`), so when two showers share a vertex the parent assigned depends on
`showers` iteration order. That is `IndexedShowerSet` order, i.e. shower-id
order, i.e. construction order — see §6.

### P12 — `PDGName` falls back to the number; `pf_pdg_to_name` falls back to "particle" — *port defect (display)*

Prototype `:529-547` looks the code up in `TDatabasePDG`, and when that fails:

- `pdg > 1e9` → a nuclear name, `"Ar-40"` style, decoded from Z and A;
- otherwise → **the PDG number itself**, as a string.

Toolkit `pf_pdg_to_name` `:1024-1040` is a ten-entry `switch` returning
`"particle"` for everything else. Three concrete gaps:

- **111 (π⁰) is absent from the table.** Harmless for the π⁰ *node*, which uses
  the literal `"pi0  "` at `:1562` — but a *track segment* PID'd as π⁰ renders as
  `"particle"`.
- **Nuclei render as `"particle"`**, losing the species entirely. They are also
  the class the `np_ke_min` floor is written for (`:1456`), so they are expected
  to occur.
- **321/−321 (K±) are in the toolkit table** and are fine, but any other code —
  Λ, Σ, deuteron — degrades to `"particle"` where the prototype would have shown
  the number and so remained diagnosable.

Related and smaller: when `has_particle_info()` is false the toolkit emits
`"particle  0 MeV"` (`:1587-1588`); the prototype emits `PDGName(0)` — which
`TDatabasePDG` does not know, so the literal `"0"` — and `KE` of an all-zero
4-vector, also `0`. So `"0  0 MeV"`.

### P13 — the track node's KE comes from a different field, and pr/31 P1 makes that matter — *port defect*

The prototype fills the track node's 4-momentum **only under a guard**
(`:1249-1264`):

```cpp
if (sg->get_particle_4mom(3)>0){
   ...startMomentum from get_particle_4mom...
}else{
   ...all eight components set to 0...
}
```

An all-zero 4-vector gives `E = 0`, `M = 0`, so **`WCReader::KE` returns 0** and
the node reads `"<name>  0 MeV"`. The prototype is explicit: no 4-momentum, no
energy shown.

The toolkit reads a **separately stored scalar** (`:1595`):

```cpp
ke_str = format_mev(pi->kinetic_energy());
```

`ParticleInfo::kinetic_energy()` (`aux/inc/WireCellAux/ParticleInfo.h:48`) is an
independent member, set by `set_kinetic_energy` and by the constructors — it is
*not* derived from `four_momentum()` at read time. So a segment whose 4-momentum
was never computed can still carry a non-zero `m_kinetic_energy` and will display
it. (Tier B: I read `ParticleInfo`'s interface, not `update_kinematics`'s body.)

**This is where the round touches another round.** doc pr/31 P1 found the
prototype's `cal_4mom` guard dropped at 11 of 13 toolkit sites. Those two
findings point the same way: the prototype's tree shows `0 MeV` exactly where the
4-momentum was withheld, and the toolkit shows a number both because the guard is
gone upstream and because the display no longer routes through the 4-vector.
Neither change alone would be visible here; together they are.

### P14 — coordinates are written at full precision instead of one decimal — *cosmetic*

Prototype `:449`, immediately before writing the node:

```cpp
out << fixed << setprecision(1);
```

so `"start":[-178.9, 33.1, 475.3]`. The toolkit writes raw JSON doubles
(`:1425-1430`), so `"start":[-178.91503528618642, …]`.

No display consequence — Bee reads the numbers, not the text. Recorded because
byte comparison of a toolkit `mc.json` against a prototype one will differ on
every coordinate of every node, and it is better to know that in advance than to
discover it mid-comparison.

---

## §4 The SBND operating point

`cfg/pgrapher/experiment/sbnd/clus.jsonnet:1757-1774`:

```jsonnet
bee_pf: [
    {
        name: 'mc',
        visitor: 'TaggerCheckNeutrino:pr',
        grouping: 'live',
        [if std.member(pipeline_names, 'tagger_check_neutrino') then 'prototype_names']: true,
        [if std.member(pipeline_names, 'tagger_check_neutrino') then 'em_ke_min']: 5 * wc.MeV,
        [if std.member(pipeline_names, 'tagger_check_neutrino') then 'np_ke_min']: 10 * wc.MeV,
    },
],
```

Against the C++ defaults (`MultiAlgBlobClustering.h:170-175`:
`prototype_names{false}`, `em_ke_min{0.0}`, `np_ke_min{0.0}`):

| knob | C++ default | SBND | prototype |
|---|---|---|---|
| `prototype_names` | `false` | **`true`** | n/a — the prototype has only its own naming |
| `em_ke_min` | `0` (keep all) | **5 MeV** | 5 MeV, hard-coded `WCReader.cc:508` |
| `np_ke_min` | `0` (keep all) | **10 MeV** | 10 MeV, hard-coded `WCReader.cc:509` |

So **SBND runs at the prototype's operating point** on all three, and the
key-suppression idiom keeps the compiled config byte-identical when the PR
visitor is absent. This is the first stage in the series where the SBND
configuration *reduces* the divergence rather than widening it — contrast pr/33
§4, where `mip_dqdx_median = 48000` vs the prototype's `43e3` made every
threshold in that stage ~10 % harder to pass.

Two caveats:

- The floors match but the **pruning rule** does not (P6), so equal thresholds do
  not give equal trees.
- `prototype_names = true` selects the toolkit's ten-entry name table, which is
  not the prototype's `TDatabasePDG` (P12). The knob name promises more parity
  than it delivers.

---

## §5 Looks like a divergence, is not

Thirteen. Written out because each cost real time to settle, and because the
first one would otherwise be this document's headline — wrongly.

**§5.1 `conn_type == 4` — the prototype drops it too.** The prototype's shower
loop *appears* to keep type-4 showers: the diagnostic at `:1533` prints
"SKIP shower (conn_type=4)" and then `fill_reco_tree(*it, rtree)` is called
**outside** that `if`/`else`, at `:1554`, for every shower. But inside
`fill_reco_tree(WCShower*, …)` at `:1297`:

```cpp
if (pair_start_vertex.second ==4 ){
  rtree.mc_dir_weak[rtree.mc_Ntrack] = 1;
  return;                      // before mc_Ntrack++ at :1317
}
```

Same uncommitted-slot pattern as P5. The node is never committed and the next
fill overwrites it. **Net behaviour matches the toolkit's `continue` at `:1285`.**
Not a divergence.

Two residues, neither affecting `mc.json`: the caller's
`rtree.mc_stopped[rtree.mc_Ntrack-1] = 0;` at `:1555` stomps the previous entry
(`mc_stopped` is unread by `DumpMCJSON`); and a type-4 shower that is *also* in
`pi0_showers` takes the `else` branch at `:1707`, where
`map_sgid_rtid[map_sg_sgid[curr_sg]]` misses and `operator[]` writes the mother
onto entry 0. Whether conn4 ∩ pi0_showers is reachable is a question, not a
claim — §7.

**§5.2 Shower KE is `kine_best` on both sides** despite the prototype's
start-segment mass appearing in the 4-vector. It cancels; §2.4.

**§5.3 π⁰ KE is the reconstructed pair mass on both sides.** The prototype's
`E = mass + 135 MeV` looks like a unit error and is not, given `M` is recovered
from the 4-vector; §2.4.

**§5.4 Pseudo-particle KE is `kine_best` on both sides**, for both the γ and the
neutron flavour; §2.4.

**§5.5 The `kine_best == 0 → kine_charge` fallback is present in both.** The
prototype writes it out at `:1304-1305` and `:1405-1406`; the toolkit puts it
inside the getter (`PRShower.h:152-153`):
`if (data.kenergy_best != 0) return data.kenergy_best; else return data.kenergy_charge;`.
Same rule, different location.

**§5.6 Integer-MeV truncation matches** under `prototype_names`; §2.4.

**§5.7 The prototype's `set_kine_charge` write-back is dead code.**
`fill_reco_tree(ProtoSegment*, …)` at `:1219` does

```cpp
sg->set_kine_charge( rtree.mc_kine_charge[rtree.mc_Ntrack] * units::MeV);
```

which is a *mutation performed at output time* — exactly the kind of hidden
side effect that would make an output-stage port unsafe. It is not: a full-tree
grep for `get_kine_charge()` across `prototype_base` returns the declaration
(`ProtoSegment.h:132`), the `WCShower` counterpart, and **twenty-odd call sites
that are all on showers** (`NeutrinoID_cosmic_tagger.h`,
`NeutrinoID_nue_functions.h`, `NeutrinoID_shower_clustering.h`). No caller reads
a *segment's* `kine_charge`. The toolkit's `PR::Segment` has no such field at
all, and that is correct.

There is a **second** state mutation in the same family, and it is also inert:
`fill_psuedo_reco_tree:1374` advances `acc_segment_id`, a `NeutrinoID` member
(`NeutrinoID.cxx:44`), so the prototype's PF fill leaves the object changed. At
the call site (`wire-cell-prod-nue.cxx:3087`) the only things that follow are
`get_kine_info()` — a getter — and `TMC->Fill()`, and the counter is per
`NeutrinoID` instance so it cannot leak between neutrino candidates in the
enclosing loop. Nothing reads it afterwards.

With both mutations accounted for, **the stage is display-only**: those were the
only two candidates for a physics consequence and neither has a consumer.

**§5.8 `fill_proto_main_tree` and `fill_reco_simple_tree` are dead.** Commented
out at all four app call sites (§0). Do not "restore" them; do not audit them as
missing ports.

**§5.9 The prototype's shower loop has no cluster filter either.** Its track loop
does (`:1488`) and its shower loops (`:1520`, `:1643`) do not — showers may come
from any cluster on both sides. The asymmetry is the prototype's design, not an
omission, and P1 is specifically about the *track* half.

**§5.10 The `WCRecoTree` fields with no toolkit counterpart are out of scope.**
`mc_included`, `mc_dir_weak`, `mc_kine_range`, `mc_kine_dQdx`, `mc_kine_charge`,
`mc_length`, `mc_stopped`, `mc_process`, `mc_endMomentum` are computed by the
prototype's `fill_*` functions and **never read by `DumpMCJSON`** (§0). They
belong to the ROOT reco tree, whose toolkit counterpart is the `kine`/`tagger`
blocks dumped elsewhere — not to the particle-flow tree. Their absence here is
not a gap in this stage.

**§5.11 Shower-under-shower nesting resolves to the same node.** The prototype
attaches a child shower by setting its mother to
`map_vertex_in_shower[v]->get_start_segment()`'s id — and the *parent shower's
own node* is keyed by that same start segment (`:1268`). So the flat-tree
linkage produces the same nesting the toolkit builds structurally at `:1489`.
Same result, different representation.

**§5.12 `IsPrimary` and `seg_parent == nullptr` are the same predicate**; §2.6.

**§5.13 The toolkit's deterministic iteration is not a divergence to port back.**
The prototype's `map_segment_vertices` walk at `:1485` is pointer-keyed and
therefore run-to-run unstable in node order. The toolkit's index-ordered walk is
strictly better. Under this tree's determinism rules (M4, and the sweep in
`c05bc5f7`) reproducing the prototype's ordering would be a regression.

---

## §6 Determinism

**Verdict: the graph walk is deterministic; one input ordering is inherited and
not proven here.**

Clean:

- Node and edge iteration use `PR::ordered_nodes` (`:1100`) and
  `PR::sorted_out_edges` (`:1138`, `:1157`), never raw `boost::vertices` /
  `boost::out_edges`. Consistent with `c05bc5f7`'s sweep; zero raw calls remain
  in this function.
- Every keyed container uses an index comparator: `seg_parent`, `seg_children`,
  `seg_endpoints`, `vtx_incoming_seg` (`:1125-1128`), `seg_to_shower` (`:1106`),
  `vtx_to_nd` (`:1099`), the four shower maps (`:1257-1266`), and
  `vtx_to_parent_shower` (`:1212`).
- Root-track emission iterates `seg_parent` (`:1666`) — `SegmentIndexCmp`
  ordered.
- π⁰ grouping iterates `std::map<int, …>` keyed by π⁰ id (`:1551`).

The residual:

- `showers` is `PR::IndexedShowerSet`, ordered by `get_shower_id()`
  (`PRShower.h:216-221`), which `PRShower.h:135` documents as *"assigned at
  construction, unique per run"*. So the order is deterministic **iff shower
  construction order is** — a property of pr/33's stage, not this one. It matters
  here because the fixed-point loop at `:1215-1250` is **first-claim-wins** on
  `vtx_to_parent_shower` (`:1236-1240`), so a vertex shared by two showers is
  assigned to whichever is visited first. Inherited, not established.

No pointer-keyed container is iterated anywhere in `fill_bee_pf_tree`.

---

## §7 Loose ends

Seven, each a question this audit raises and does not answer.

1. **P1/P2 population.** How many events actually have a foreign-cluster or
   behind-a-shower track segment enter the tree? Measurable by re-running the
   valfast manifest with `WCT_BEE_PF_PRINT=1` and counting
   `ADD track-node … in_main_cluster=0`. The diagnostic field already exists
   (`:1620`) — it was written for exactly this and is currently discarded.
2. **P3's vertex class.** Is a vertex ever both on the BFS track path and inside
   a shower? If never, P3 is inert.
3. **P10's frequency.** How often do two showers of one π⁰ attach to different
   parents? A duplicate `pi0` label in the panel is the observable.
4. **P9's collision precondition.** Can a `Facade::Cluster` ident be 0? Nothing
   read here constrains it; evt 388's minimum is 23.
5. **§5.1's residue.** Is `conn_type == 4 ∩ pi0_showers` reachable? If so the
   prototype corrupts entry 0 of its own tree — worth knowing before any
   prototype-output comparison is trusted.
6. **`ParticleInfo::update_kinematics`** was not read (Tier B). P13 assumes
   `m_kinetic_energy` and `m_four_momentum` can disagree; reading that function
   would settle it.
7. **The provenance check is stage-local, and the exposure is now measured.**
   The `a5fc0b9..HEAD` diff was evaluated line-by-line only for the four
   `fill_*` functions. The branch-wide stat (header) shows **+5833/−989 across
   26 files, every one of the series' cited files among them**. Two commits on
   the branch are algorithm fixes, not instrumentation. So the exposure is real
   and it is not one file: at minimum the line anchors in pr/28–pr/33 need
   re-checking against `a5fc0b9`, and where a cited function falls inside a
   non-instrumentation hunk the *finding* needs re-deriving, not just the
   anchor. Triage command:
   `git -C prototype_base/pid diff a5fc0b9..HEAD -- src/<file>`.
   This is a substantial piece of work and it is the highest-value follow-up in
   the series.

---

## §8 Summary

Ranked by how much each changes the emitted tree.

| # | finding | class | prototype | toolkit |
|---|---|---|---|---|
| P1 | track walk has no main-cluster filter | port defect | `NeutrinoID.cxx:1488` | `MultiAlgBlobClustering.cxx:1138`, `:1157` (absent); `main_cluster` `:1090` used only at `:1609` |
| P2 | no shower-vertex barrier in the BFS | port defect | `:1591-1593`, `:1611` | `:1131`, `:1133` |
| P3 | shower-parent precedence inverted | port defect | `:1655-1660`, `:1680-1684`, `:1720-1724` | `:1310`, `:1335` |
| P4 | fallback parent by direction vs by BFS arrival | port defect | `find_incoming_segment` `:1763-1783` | `vtx_incoming_seg` `:1128`, `:1146`, `:1166` |
| P10 | π⁰ node per parent, not per π⁰ | port defect | `map_pio_id_saved_pair` `:1326`, `:1361` | `pi0_groups` `:1551` (call-local) |
| P11 | shower-parent resolution reimplemented | port defect | `map_vertex_in_shower` `NeutrinoID.h:2032` | fixed point `:1211-1251`; fallback `:1377` |
| P5 | direction-less segments dropped / kept | prototype bug not reproduced | `:1214` (returns before `:1260`) | no counterpart |
| P6 | below-threshold node takes its subtree / does not | deliberate | `WCReader.cc:434`, `:444` | `keep_node` `:1451-1453` |
| P8 | disconnected main-cluster segments dropped | deliberate | `:1485` (no reachability test) | `:1663-1674` |
| P9 | synthetic ids from a different space | port defect | `acc_segment_id` `:1374`, `:1324` | `next_id = 1` `:1415` |
| P13 | track KE from a stored scalar, not the 4-vector | port defect | `:1249-1264` + `WCReader.cc:522` | `:1595`; cf. **pr/31 P1** |
| P12 | PDG-name fallback `"particle"` vs the number | port defect (display) | `WCReader.cc:529-547` | `pf_pdg_to_name` `:1024-1040` |
| P7 | top-level order inverted | port defect (display) | `:1485` then `:1520` | `:1661` then `:1666` |
| P14 | coordinate precision | cosmetic | `WCReader.cc:449` | `:1425-1430` |

Six port defects change tree *shape*; two more change *labels*; two are the
toolkit deliberately improving on the prototype; one is a prototype bug correctly
not reproduced; one is cosmetic.

---

## §9 What is NOT claimed

- **No event was run for this audit.** The evt-388 tree quoted in the Repro block
  was read from an existing `mabc-pr.zip` under `work-dqdx388-base/`; nothing was
  regenerated, nothing was written into any existing output directory (M13).
- **No frequency is measured.** P1, P2, P3, P9 and P10 are established from
  source. How often each fires is §7, and the answer could be "never" for any of
  them — P3 and P9 in particular have preconditions I did not establish.
- **P5, P6 and P8 are divergences, not defects.** In each, the toolkit's
  behaviour is arguably the better one. They are listed because the trees differ,
  not to recommend reverting them. The class column exists so that distinction
  survives being skimmed.
- **P13 is a compound finding.** Its visible effect depends on pr/31 P1's
  dropped `cal_4mom` guards being real. If those are fixed, P13's practical
  consequence shrinks to the edge case where `m_kinetic_energy` and
  `m_four_momentum` genuinely disagree — which §7.6 leaves open.
- **`ParticleInfo` was read at interface level only** (Tier B), as were the
  `TrackFitting` shower accessors and `Facade::Cluster::get_cluster_id`.
- **`PRShower::fill_sets` and `WCShower::fill_sets` were not compared.** Both
  sides call them to enumerate a shower's vertices and segments, and P2 and P11
  both depend on what they return. If they disagree, those two findings change
  shape.
- **The prototype checkout is not pristine and I verified only this stage's
  functions.** §7.7. What is established: the `port` branch changes every file
  the series cites, and this stage's four functions are clean. What is **not**
  established: whether any *earlier* round's cited lines fall inside a changed
  hunk. That is a per-round question and this document does not answer it for
  any round but its own — the warning is a prompt to check, not a verdict that
  the earlier findings are wrong.
- **§6's verdict is conditional**, not proven: the walk is deterministic given a
  deterministic shower construction order, which belongs to pr/33's stage.
- **No recommendation is made on any item.** All fourteen are escalation rule 1
  — they change output unconditionally, in a display artifact the owner reads —
  and the call is the owner's.

---

## §10 Owner filter — 14 → 5

**The ask.** *"For all these listed behavior changes, we can skip the ones that
are improvements over the previous prototype, and only focus on the ones that
are bugs or missing from the port."* Same filter previously applied to pr/30
(14→4), pr/31 (15→9), pr/32 (12→4) and pr/33 (14→5).

**Re-verification basis.** Everything below was re-read at toolkit
**`407c5ba9`** — five commits past the `01ff88b1` of §3 — not taken from §3's
text. Two provenance results, both new and both clean:

```bash
# toolkit side: the audited file did not move
git -C toolkit log --oneline 01ff88b1..HEAD -- clus/src/MultiAlgBlobClustering.cxx   # (empty)
git -C toolkit diff --stat 01ff88b1..HEAD -- clus/src/MultiAlgBlobClustering.cxx     # (empty)
wc -l clus/src/MultiAlgBlobClustering.cxx                                            # 2683, both revs

# prototype side: the OTHER program of this stage is pristine upstream
cd toolkit/prototype_base/bee && git status -sb        # ## porting...origin/porting @ 58bc60a
git merge-base HEAD origin/master                      # 8811c60e
git diff 8811c60e..HEAD --stat -- '*WCReader*'         # (empty)  <-- WCReader.{cc,h} UNMODIFIED
```

**Every §3/§8 anchor in this document is still exact.** That is worth stating
plainly: pr/32's anchors were four commits stale within a day and pr/33's were
up to +19 lines stale with a *new* finding created underneath them. A clean
anchor check is a result, not an absence of one.

And §7.7's provenance proof now covers **both** prototype programs. §3 verified
only that `pid`'s four `fill_*` functions are algorithm-pristine on the diverged
`port` branch; `bee/WCReader.cc` — which carries P6's, P7's and P12's citations —
had not been checked. It is on branch `porting`, and its diff against the
`origin/master` merge-base `8811c60e` is **empty**. So this round's citations are
sound on both sides. §7.7's warning about *earlier* rounds is untouched.

### §10.1 The filter table

| # | class in §8 | verdict | why |
|---|---|---|---|
| **P1** | port defect | **KEEP → F1** | One of two adjacent `continue` guards was ported. A dropped guard is the definition of missing-from-the-port. |
| **P2** | port defect | **KEEP → F2** | The prototype's `used_vertices` pre-seed has no toolkit counterpart; the frontier check it feeds does. Half a barrier. |
| **P3** | port defect | **KEEP → F3** | Real, and its mechanism is sharper than §3 states — see §10.4. Merged with P11. |
| **P11** | port defect | **KEEP → F3** | **Not a separate finding.** P11's failure mode *is* P3's mechanism. §10.4. |
| **P10** | port defect | **KEEP → F4** | The prototype memoizes on a member; the toolkit's map is call-local. One π⁰ renders twice. |
| **P12** | port defect (display) | **KEEP → F5** | The name table is missing entries the prototype has, and its fallback destroys information the prototype preserves. |
| P4 | port defect | **DROP** | §10.7a — the prototype's selector reads an uninitialized `bool` and returns an unchecked `nullptr` that corrupts tree entry 0. Its answer cannot be called the correct one. |
| P5 | prototype bug not reproduced | **DROP** | The header already classes it so. |
| P6 | deliberate | **DROP** | Keeping a high-energy daughter whose carrier is soft is an improvement. |
| P7 | port defect (display) | **DROP** | §10.7b — pure top-level ordering in a display-only stage; not a dropped guard but an emergent consequence of assembly order. One-line reorder offered, not proposed. |
| P8 | deliberate | **DROP** | Not listing unconnected fragments as primary particles is an improvement. |
| P9 | port defect | **RESOLVED** | §10.7c — the collision is **unreachable**. Closes §7.4. |
| P13 | port defect | **RESOLVED** | §10.7d — the two fields **cannot disagree**. Closes §7.6. |
| P14 | cosmetic | **DROP** | Already classed cosmetic; §3 records it only as a warning about byte comparison. |

**Five findings, five knobs.** Unlike pr/33 (5 findings → 8 knobs) no finding
needs splitting here: each survivor moves one decision in one direction, and F3's
two code edits are shown in §10.4 to be **not separately observable**, so one
knob can still attribute.

**All five are `BeePFConfig` knobs.** The struct is
`MultiAlgBlobClustering.h:163-176`; each addition is a `bool …{false}` member
plus one key-suppressed jsonnet line in `sbnd/clus.jsonnet`'s `bee_pf` block
(§4), so the compiled config is byte-identical when off.

### §10.2 F1 = P1 — restore the main-cluster guard on the track walk

**Verified at HEAD.** `grep -n main_cluster` over `MultiAlgBlobClustering.cxx`
returns exactly four hits: the declaration `:1090`, `:1609` and `:1620` inside
the `flag_print` block of `build_seg_node`, and `:2574`, an unrelated comment in
another function. Confirmed: the variable is computed and never used to filter.

**Proposed knob:** `pf_track_main_cluster_only` (C++ default `false`).

**On:** skip a segment whose cluster is not the main vertex's, at both BFS sites:

| site | current | with the knob |
|---|---|---|
| `:1140` seed | `if (!seg \|\| used_segs.count(seg) \|\| conn4_skip_segs.count(seg)) continue;` | `… \|\| (cfg.pf_track_main_cluster_only && !same_cluster(seg)) ) continue;` |
| `:1159` expansion | same predicate | same addition |

**One subtlety that must not be got wrong.** The prototype's guard compares
**cluster idents**, not cluster objects:

```cpp
if (sg->get_cluster_id() != main_vertex->get_cluster_id()) continue;   // NeutrinoID.cxx:1488
```

and `Grouping::separate` deliberately gives **every split product the parent's
ident** (`Facade_Grouping.cxx:182`), as do the bundle splitters
(`ClusteringUnmergeBundle.cxx:405`, `ClusteringProtectBundle.cxx:545`,
`ClusteringRecoveringBundle.cxx:211`, all `parent*100+sub`). So ident-equality is
strictly weaker than pointer-equality, and the two are **not interchangeable**.
`same_cluster` must be written

```cpp
auto same_cluster = [&](PR::SegmentPtr s) {
    const auto* c = s->cluster();
    return main_cluster && c && c->get_cluster_id() == main_cluster->get_cluster_id();
};
```

**A consequence for the existing diagnostic.** `:1609` computes
`in_main_cluster` by **pointer** comparison
(`seg_cluster == main_cluster`). That is the stricter test, so the
`in_main_cluster=0` count that §7.1 proposes to measure P1's population with
**over-reports** relative to the prototype's guard. Fix the diagnostic to the
ident test in the same change, or §7.1's measurement will overstate F1.

**Do not touch the shower loops** (§5.9): the prototype's asymmetry is by design.

### §10.3 F2 = P2 — pre-seed the vertex set from every shower

**Proposed knob:** `pf_shower_vertex_barrier` (C++ default `false`).

**On:** at `:1130-1133`, seed `visited_vtxs` from the shower vertex sets, exactly
as `used_segs` is already seeded from `shower_segs` at `:1131`:

```cpp
PR::IndexedVertexSet visited_vtxs;
if (cfg.pf_shower_vertex_barrier) visited_vtxs = shower_vtxs;   // NEW; mirrors :1131
visited_vtxs.insert(main_vertex);
```

`shower_vtxs` costs nothing to build — the loop at `:1108-1118` already calls
`fill_sets(sv, ss, false)` per shower and currently **discards `sv`**. Collect it
alongside `shower_segs`.

**Where the barrier actually bites.** `visited_vtxs` is tested at `:1153`, at the
top of the expansion, i.e. *after* a vertex has been pushed onto the frontier.
Seeding it therefore stops expansion **through** a shower vertex while still
allowing the segment that reaches it to be claimed — which is precisely the
prototype's behaviour: its `:1611` check is also on `curr_vtx` at pop time, and
its `:1597-1602` seed loop deliberately pushes shower segments (§3's aside).
**Do not additionally filter at `:1140`/`:1159`** — that would be a second,
stricter barrier the prototype does not have.

**Interaction with F1 and F3, stated so the gate is readable.** F1 and F2 both
*shrink* the toolkit's track set; F3 *re-parents* showers. Turning F2 on shrinks
`vtx_incoming_seg`, which changes which branch each shower takes at `:1310`, so
F2 and F3 are not independent in their effect on the emitted tree. They are still
separately attributable because each is a distinct code edit with a distinct knob
— but a combined-on arm is not the sum of two single-on arms, and the gate table
should say so rather than imply additivity.

### §10.4 F3 = P3 **and** P11 — one defect: `vtx_to_parent_shower` is half-populated

§3 lists these as two findings. They are one, and the mechanism is more concrete
than either entry states. From the fixed-point loop, verbatim at `:1227-1247`:

```cpp
PR::SegmentPtr parent_seg   = at_track ? vtx_incoming_seg.at(start_vtx) : nullptr;
PR::ShowerPtr  parent_shower = (at_main || at_root) ? shower : nullptr;
...
for (const auto& vtx : sv) {
    if (vtx == main_vertex) continue;
    if (at_main || at_root) {
        if (!root_reachable_vtxs.count(vtx)) {
            root_reachable_vtxs.insert(vtx);
            vtx_to_parent_shower[vtx] = parent_shower;      // <-- ONLY here
            any_added = true;
        }
    } else {
        if (!vtx_incoming_seg.count(vtx) && !root_reachable_vtxs.count(vtx)) {
            vtx_incoming_seg[vtx] = parent_seg;             // <-- no parent-shower record
            any_added = true;
        }
    }
}
```

**`vtx_to_parent_shower` is written only in the root branch.** So:

- a shower nested inside a **root-attached** shower resolves correctly — `:1335`
  finds the vertex in `root_reachable_vtxs`, `:1337` finds its parent shower, and
  the child nests under the parent shower, matching the prototype;
- a shower nested inside a **track-attached** shower does **not**. Its start
  vertex was written into `vtx_incoming_seg` at `:1244` pointing at the *track
  segment*, so `:1310` matches first and the child is hung under the **track
  segment**, one level too shallow and under the wrong particle.

That second case is exactly P3's "a vertex that is both on the track path and
inside a shower" — and the toolkit **manufactures that vertex class itself** at
`:1244`. So §7.2's open question ("is that class ever populated?") is answered:
it is populated by construction whenever any shower is track-attached and has
more than its start vertex. **§7.2 is closed.**

And P11's stated failure mode — "a shower whose start vertex lies inside a
shower that is itself unresolved falls through to the root" — is the same
half-population seen from the other side. One defect, one knob.

**Proposed knob:** `pf_shower_parent_precedence` (C++ default `false`). Two
edits, which must ship together:

- **(a)** in the `else` branch at `:1243`, also record the parent shower:
  ```cpp
  if (!vtx_incoming_seg.count(vtx) && !root_reachable_vtxs.count(vtx)) {
      vtx_incoming_seg[vtx] = parent_seg;
      if (cfg.pf_shower_parent_precedence) vtx_to_parent_shower[vtx] = shower;
      any_added = true;
  }
  ```
- **(b)** at `:1310`, test the parent shower **first**, matching the prototype's
  `map_vertex_in_shower`-first order at `:1655`, `:1680`, `:1720`:
  ```cpp
  auto ps_it = cfg.pf_shower_parent_precedence
             ? vtx_to_parent_shower.find(start_vtx) : vtx_to_parent_shower.end();
  if (ps_it != vtx_to_parent_shower.end()) { /* attach to ps_it->second */ }
  else { auto it = vtx_incoming_seg.find(start_vtx); ... }   // unchanged
  ```

**Why one knob is still attributable — the pr/33 §10.2 criterion, checked.**

- **(a) alone is inert.** Without (b), `:1310` still tests `vtx_incoming_seg`
  first and never consults `vtx_to_parent_shower` for those vertices. Zero
  observable change.
- **(b) alone is a regression.** With `vtx_to_parent_shower` still empty for
  track-attached shower vertices, inverting the test sends nothing new down the
  shower branch and can only reach `:1358`'s "no parent found" → **root**, i.e.
  hoisting showers *further* from their true parent than today.

Neither half is independently meaningful, so a single knob cannot fail to
attribute. This is the opposite finding from pr/33's F1/F2 and it is reached by
the same test.

**One implementation hazard, and why it is already handled.** `fill_sets`
(`PRShower.cxx:410-419`) inserts **every** vertex in the shower's view including
the shower's own start vertex — `flag_exclude_start_segment` filters *segments*
only. A naive (a) would therefore make a shower its own parent. It does not,
because the guard the line sits inside (`!vtx_incoming_seg.count(vtx)`) already
excludes the start vertex: `at_track` is true precisely *because* that vertex is
in `vtx_incoming_seg`. The root branch is safe for the mirror reason
(`!root_reachable_vtxs.count(vtx)` at `:1236`, plus the `vtx == main_vertex`
skip at `:1233`). **Write (a) inside the existing guard, not before it.**

**Not proposed:** the first-claim-wins ordering dependence in the same loop
(`:1236-1240`) is real but is P11's determinism residual, not this defect. It is
inherited from shower construction order (§6) and belongs to pr/33's stage.

### §10.5 F4 = P10 — one π⁰ node per π⁰, not per parent

**Proposed knob:** `pf_pi0_node_per_id` (C++ default `false`).

**On:** hoist the π⁰ grouping out of `append_showers`. `pi0_groups` is declared
at `:1551` inside the lambda, which `:1489`, `:1635` and `:1661` invoke once per
parent. Promoting it to a `fill_bee_pf_tree`-scope map keyed by π⁰ id, filled in
a pass over `showers` before assembly, gives the prototype's
`map_pio_id_saved_pair` semantics (`:1326`, `:1361`).

**The structural obstacle, stated rather than waved at.** The prototype's flat
`WCRecoTree` lets one π⁰ node be a daughter of two mothers; its explicit
duplicate guard at `:1736-1737` exists for exactly that. A jsTree node lives in
exactly one `children` array, so the toolkit **cannot** reproduce the shared
node. The knob must therefore choose a single home for the split π⁰. The
prototype's own choice is *first writer wins* — `fill_pi0_reco_tree` creates the
node on the first daughter it reaches and every later daughter reuses it, so the
π⁰ lands under the first daughter's parent in iteration order. Reproducing that
gives prototype parity; choosing the higher-energy daughter's parent would be
better physics and **would not** be parity.

**This is escalation rule 4 (M15): both readings, no pick.** The knob ships with
whichever the owner selects; the recommendation is *first-daughter parity*,
because this stage's whole purpose is matching the panel the prototype draws.

### §10.6 F5 = P12 — the PDG name table

**Proposed knob:** `pf_pdg_name_prototype_fallback` (C++ default `false`).

Corrected count first: `pf_pdg_to_name` (`:1024-1040`) is an **eleven**-entry
switch — 11, −11, 13, −13, 22, 211, −211, 2212, 2112, 321, −321 — not the
"ten-entry" of §3's P12 and §4.

**On:** three additions, all inside that one function.

| gap | prototype | fix |
|---|---|---|
| **111 (π⁰) absent** | `TDatabasePDG` → `"pi0"` | `case 111: return "pi0";` — a *track segment* PID'd as π⁰ currently renders `"particle"`. The π⁰ *node* is unaffected: it uses the literal at `:1562`. |
| **nuclei → `"particle"`** | `pdg > 1e9` → `"Ar-40"` from Z and A (`WCReader.cc:529-547`) | decode Z/A the same way. These are the class `np_ke_min` exists for (`:1456`), so they are expected. |
| **everything else → `"particle"`** | the PDG number as a string | `return std::to_string(pdg);` — Λ, Σ, deuteron stay diagnosable instead of collapsing to one label. |

**Why a new knob rather than folding this into `prototype_names`.** The obvious
home is the existing `prototype_names`, whose whole promise is prototype-style
naming — but that knob is **already SBND-ON** (§4), so widening it changes SBND
output with no off-state. That is escalation rule 1. A separate default-OFF knob
keeps the A/B possible.

**Leak check — the pr/33 F3 lesson, applied and cleared.** Before widening any
shared helper, grep every other caller. `pf_pdg_to_name` has exactly three:
`:1471`, `:1505`, `:1594`, all inside `fill_bee_pf_tree`. **No consumer outside
this stage**, so the change cannot leak the way pr/33's F3 would have leaked into
`ssm_tagger`.

**Also worth fixing in the same change, and not knob-worthy:** the no-PID node
text. The toolkit emits `"particle  0 MeV"` (`:1587-1588`); the prototype emits
`PDGName(0)`, which `TDatabasePDG` does not know, hence the literal `"0"`, and
`KE` of an all-zero 4-vector, hence `"0  0 MeV"`. Under the same knob this is one
more line.

### §10.7 The gate these knobs need is a different artifact

**Read this before reusing any earlier round's gate language.** Every prior round
in this series gated on `pctree-pr-evt*.tar.gz` member-content hashes via
`abtest/hash_archive.py`, driven by `pr32_cmp.py`. **That gate is vacuous here.**
The particle-flow tree is not in `pctree-pr`; all five knobs, on or off, leave
that archive bit-identical. Quoting "48/48 byte-identical" from such a run would
report a gate that tested nothing.

The artifact under test is `mabc-pr.zip::data/0/0-mc.json` — the same path the
Repro block already reads. The gate should be **two-sided**, which buys something
this document currently cannot claim:

| arm | `mc.json` | `pctree-pr` | what it establishes |
|---|---|---|---|
| knob **OFF** vs baseline | identical | identical | the byte-identical bar (§1 of CLAUDE.md) |
| knob **ON** vs baseline | **differs** | **identical** | the **first empirical proof of the display-only claim** |

The second row matters. The display-only verdict in this document's header rests
entirely on the §5.7 source argument — that the prototype's two state mutations
(`set_kine_charge`, `acc_segment_id++`) have no consumer. A knob-on arm that
moves `mc.json` while leaving `pctree-pr` byte-identical converts that argument
into a measurement.

`mc.json` is plain JSON, so the comparison is a content hash of the extracted
member — not `cmp` on the zip (M2). Note P14 in passing: coordinates are raw
doubles, so toolkit-vs-toolkit comparison is exact; only toolkit-vs-*prototype*
comparison would differ on every coordinate.

**Nothing was run.** No arm exists; this section specifies the gate, it does not
report one.

### §10.8 Dropped and resolved, with reasons

**a) P4 — DROPPED as an improvement, with an M15 residual.** The prototype's
`find_incoming_segment` (`:1763-1783`) selects by fitted direction; the toolkit's
`vtx_incoming_seg` records BFS arrival. The prototype's function declares
`bool flag_start;` at `:1768` with an `if`/`else if` at `:1769-1772` that has
**no `else`**, then reads it at `:1774`; and it returns `0` on no match, which
all three call sites use unchecked, writing the shower's mother onto tree entry 0
via `std::map::operator[]`. A selector that reads an uninitialized bool and
corrupts an unrelated node cannot be called the correct answer, so "port it back"
is not a coherent fix.

*The residual, which is a real open question and not a proposed change:* on a BFS
tree rooted at the neutrino vertex the two rules **coincide** whenever segment
directions point outward along that tree, and diverge only where a direction was
flipped or left weak. The prototype's answer then follows the reconstructed
physics flow; the toolkit's follows topological distance from the vertex. Which
is wanted is a physics call, undocumented in `porting_dictionary.md`, and per
escalation rule 4 it is presented, not picked.

**b) P7 — DROPPED as cosmetic, with a one-line option.** Top-level ordering in a
display-only stage, and not a dropped guard: the prototype's order is an emergent
consequence of `fill_particle_tree` running its track loop (`:1485`) before its
shower loop (`:1523`), and the toolkit's is the emergent consequence of `:1661`
preceding `:1666`. If the owner wants side-by-side parity when reading toolkit
and prototype panels together, swapping those two blocks is a one-line change —
offered here so the option is on the record, deliberately **not** seated beside
F1 as a peer.

**c) P9 — RESOLVED. The collision is unreachable. §7.4 CLOSED.**

§3 states the unsafe case as "a cluster whose ident is 0". Three checks retire
it, and a fourth retires a sharper variant §3 did not consider:

1. **The unset default is `−1`, not 0.** `ident(int def = -1)`
   (`Facade_Mixins.h:115-118`). A cluster whose ident was never written gives
   `seg_display_id = -1*1000 + sid`, i.e. a **negative** id, which cannot collide
   with `next_id ≥ 1`.
2. **The root allocator starts at 1.**
   `enumerate_idents(sort_order="tree", int start=1)`
   (`Facade_Grouping.h:142`, body `Facade_Grouping.cxx:125-143`).
3. **Every other ident writer is ≥ 1 or inherits.** The complete set, from
   `grep -rn "set_ident(\|set_cluster_id(" --include=*.cxx --include=*.h .`:
   `Facade_Grouping.cxx:141` (`id++` from 1), `:182` (`separate`, inherits),
   `ClusteringFuncs.cxx:323` (inherits), `retile_cluster.cxx:611` (inherits),
   `ClusteringUnmergeBundle.cxx:405`, `ClusteringProtectBundle.cxx:545`,
   `ClusteringRecoveringBundle.cxx:211` (all `parent*100 + sub`, so ≥ 100 for
   parent ≥ 1), `QLMatching.cxx:1312` (a later stage, after this fill),
   `doctest_facades.cxx:21` (a test). **No path yields 0.**
4. **The sharper variant, checked and also closed.** `seg_display_id` (`:1408`)
   has a null-cluster branch that returns a **bare `sid`** — which *is* in
   `next_id`'s range, and is a better collision candidate than ident 0. But every
   `PR::Segment` construction site sets the cluster:
   `NeutrinoPatternBase.cxx:197`, `TaggerCheckSTM.cxx:2460`,
   `NeutrinoStructureExaminer.cxx:323`, `:467`, `:688` (all
   `.cluster(&cluster)`), and `PRSegmentFunctions.cxx:544-545`
   (`seg1->cluster(seg->cluster())`, inherits). The branch is defensive-only.

P9's *fidelity* observation stands — the toolkit's synthetic ids are not the
prototype's `cluster*1000 + acc_segment_id` — but with no collision it is a
cosmetic difference in a display artifact, and it is dropped on that basis. Its
inherited half (pr/33 P3's `acc_segment_id` by value) remains pr/33's F3.

**d) P13 — RESOLVED. The two fields cannot disagree. §7.6 CLOSED.**

§7.6 left open whether `ParticleInfo::m_kinetic_energy` and `m_four_momentum` can
disagree. Reading `aux/src/ParticleInfo.cxx` end to end: they cannot.

| mutator | what it maintains |
|---|---|
| `set_four_momentum` `:85-88` | `m_kinetic_energy = e() − m_mass` |
| `set_kinetic_energy` `:90-111` | rebuilds the 4-vector at `E = KE + mass` |
| `set_momentum` `:80-83` → `update_kinematics` `:113-118` | `E = √(p²+m²)`, `KE = E − m` |
| `set_mass` (`ParticleInfo.h:70`) → `update_kinematics` | same |
| comprehensive ctor `:24-41` | `E = KE + mass` |
| 4-momentum ctor `:45-60` | `KE = e() − mass` |
| default ctor `:12-20` | both zero |

So `E = KE + mass > 0` whenever `KE > 0`, and the prototype's guard
`if (sg->get_particle_4mom(3) > 0)` (`:1249`) would **always pass** on toolkit
data. P13's stated premise — "a segment whose 4-momentum was never computed can
still carry a non-zero `m_kinetic_energy`" — is therefore true for a different
reason than §3 gives: not that the fields drift apart, but that **the toolkit has
no *uncomputed* state to test for at all.** `ParticleInfo` sets the 4-vector the
moment a KE exists.

That is a real fidelity gap, but its entire observable frequency is "how often
does the prototype skip `cal_4mom`" — which is **pr/31 P1's** territory, not this
stage's, and pr/31 P1 is already tracked there. There is no fix that belongs in
`fill_bee_pf_tree`: the display cannot test a state the data model does not
carry. Dropped here, and §7.6 is closed rather than left open.

**e) P5, P6, P8, P14 — dropped as already classed.** §3's own class column has
them as prototype-bug-not-reproduced (P5), deliberate improvements (P6, P8) and
cosmetic (P14). The filter changes nothing about them. P14 in particular is not a
finding at all — it is a warning that toolkit-vs-prototype byte comparison will
differ on every coordinate, which §10.7 restates where it is actionable.

### §10.9 RETRACTED — the per-visitor `enumerate_idents` does run

**This section originally reported that `enumerate_idents` at `:2445` is a no-op
in the SBND production path. That was wrong and is withdrawn.** It is kept here,
struck, because the way it was wrong is a reusable trap.

**What was claimed.** `m_clusters_id_order` (`MultiAlgBlobClustering.h:418`) is
an uninitialized `std::string`; `enumerate_idents` returns early on
`sort_order.empty()` (`Facade_Grouping.cxx:127`); a grep of `cfg/` found no SBND
setting; therefore `:2445` never runs and five in-tree comments rest on a false
invariant.

**Why it was wrong.** The **member** is `m_clusters_id_order` (plural
*clusters*); the **config key it is read from is `cluster_id_order`** (singular),
at `MultiAlgBlobClustering.cxx:149`. The `cfg/` grep was run on the member's
spelling and returned a false negative. SBND sets the key at **three** sites:

```
cfg/pgrapher/experiment/sbnd/clus.jsonnet:372   cluster_id_order: 'tree',
cfg/pgrapher/experiment/sbnd/clus.jsonnet:599   cluster_id_order: 'tree',
cfg/pgrapher/experiment/sbnd/clus.jsonnet:1725  cluster_id_order: 'tree',
```

So `m_clusters_id_order == "tree"`, `enumerate_idents` runs after every visitor
as documented, and **all five comments are correct**:
`MultiAlgBlobClustering.h:260` and `:269`, `ClusteringFuncs.h:142`,
`TrackFitting.cxx:1140`, `prov_check.cxx:22`. Each was re-read with context;
none is a passing mention — `TrackFitting.cxx:1136-1142` in particular rests a
safety argument on it ("it runs only between visitors, so this cannot fire").
Doc 53's measured 31% `real_cluster_id` epoch overlap is likewise consistent
with the renumbering running, not with it being dead.

**Effect on §10.** None on the filter. It **strengthens** §10.8c: idents are
actively re-enumerated from 1 immediately before each Bee dump, so P9's
collision is unreachable by an even shorter argument than the four checks given
there. No survivor, no drop and no proposed fix depended on this section.

**The reusable trap.** A WCT component's config key and the member it lands in
are routinely spelled differently. Grepping `cfg/` for the *member* name proves
nothing. Take the key from the `cfg["..."]` expression at the read site — here
`cfg["cluster_id_order"]` at `:149` — and grep for that.

### §10.10 What §10 does not claim

- **No code was written, no build was run, no event was run.** The document
  remains audit-only; §10 adds a filter and five proposed fixes, nothing more.
- **No frequency is measured for any survivor.** §9's statement stands
  unchanged: F1, F2, F3 and F4 are established from source. §7.1 and §7.3 remain
  open. What §10 *does* close is §7.2 (F3's vertex class is populated by
  construction, §10.4), §7.4 (P9, §10.8c) and §7.6 (P13, §10.8d).
- **The knob names are proposals.** Each is stated with its exact edit site so
  the owner can reject the name without rejecting the finding.
- **F3's merge of P3 and P11 is a claim about mechanism**, verified from
  `:1227-1247` and `:1310-1398` at `407c5ba9`. If `PRShower::fill_sets` and
  `WCShower::fill_sets` return different vertex sets — never compared, §9 — the
  merge holds but the population changes.
- **§10.5's π⁰ home is unresolved by design** (M15). The knob cannot ship until
  the owner picks first-daughter parity or highest-energy-daughter.
- **The provenance result is stage-local, again.** `bee/WCReader.{cc,h}` being
  pristine says nothing about the `pid` submodule's `port` branch, whose
  +5833/−989 exposure across 26 files (§7.7) is untouched and remains the
  series' highest-value follow-up.
- **§10.7's gate has not been run.** It is a specification. Any future "byte
  identical" claim for these knobs must name `mc.json`, not `pctree-pr`.

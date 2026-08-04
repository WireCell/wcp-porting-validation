# doc pr/29 — Steiner graph building: prototype-fidelity audit

**Why.** Owner request, following doc pr/28: *"do a similar audit for the Steiner
Graph Building, I want to see the divergence between Prototype code and
toolkit."*

**Status.** **Audit only. No code is changed and none is proposed.** The owner's
instruction was explicit — *"Please do not change any code yet."* Every finding
below carries an anchor on both sides and a severity verdict; nothing carries a
patch. The three behaviour-changing items (**D1**, **D2**, **D3**) would all
alter production output unconditionally, i.e. CLAUDE.md **escalation rule 1** —
they are reported and stop there.

**Headline.** The **algorithm is the same** — the toolkit vendors the same PAAL
routines and its Voronoi/Dijkstra block is a character-for-character port (§2).
The divergences are in the *inputs* to that algorithm, and the two largest are:

* **D3** — the steiner-graph same-blob edge pass groups points by the **retiled
  cluster's own blob** instead of the prototype's **original-cluster mcell found
  by strict wire containment**, and the prototype's exclusion of points that
  land in *no* original mcell has no counterpart. This changes which edges exist
  in the delivered steiner graph.
* **D1** — one toolkit function (`check_wire_ranges_match`) serves **two**
  prototype call sites that use **different wire tolerances**, and it implements
  the wrong one for the Steiner terminal filter. Not an M7 `<`/`<=` issue — see
  the row.

**A documentation finding that frames all the rest:** the porting dictionary's
Steiner Tree section (`clus/docs/porting/porting_dictionary.md:266`) is an empty
`⚠ xxx` placeholder. **No divergence in this file family is documented
anywhere.** Under CLAUDE.md §5 rule 4 that means none of the items below may be
silently "corrected" in either direction; this document therefore presents
readings and evidence, not recommendations.

---

## Repro

There is no run behind this document — it is a source audit, so the **greps are
the reproducibility**.

```bash
# Trees this doc was read against
cd /nfs/data/1/xqian/toolkit-dev/toolkit          && git rev-parse --short HEAD   # ea1a7e3d
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img  && git rev-parse --short HEAD   # 408cca8
# prototype_base -> /nfs/data/1/xqian/prototype-dev/wire-cell   (package pid/, WCPPID)

cd /nfs/data/1/xqian/toolkit-dev/toolkit

# The two files that ARE the Steiner port
wc -l prototype_base/pid/src/PR3DCluster_steiner.h            # 1056
wc -l clus/src/SteinerGrapher.cxx clus/src/CreateSteinerGraph.cxx   # 1084 + 313

# The prototype's same-mcell edge passes live in the graph file, not the steiner file
sed -n '10,130p' prototype_base/pid/src/PR3DCluster_graph.h

# Solver provenance (§2)
grep -n "paal\|steiner_tree_greedy\|kruskal" prototype_base/pid/src/PR3DCluster_steiner.h
grep -rn "PAAL" clus/src/Graphs.cxx clus/src/PAAL.h | head

# Determinism sweep (§6)
grep -n "out_edges\|std::set<.*\*>\|std::map<.*\*" \
     clus/src/SteinerGrapher.cxx clus/src/SteinerGrapher_helpers.cxx \
     clus/src/CreateSteinerGraph.cxx clus/src/Graphs.cxx

# The empty dictionary section
sed -n '266p' clus/docs/porting/porting_dictionary.md
```

---

## §0 Scope — what this audit covers, and what it deliberately does not

**In scope: the Steiner tree construction and everything it directly consumes.**

| # | prototype | toolkit |
|---|---|---|
| 1 | `create_steiner_graph` — `PR3DCluster_steiner.h:10-75` | `CreateSteinerGraph::visit`'s `process_cluster_steiner` lambda — `CreateSteinerGraph.cxx:172-288` |
| 2 | `Create_steiner_tree` — `:182-709` | `Steiner::Grapher::create_steiner_tree` — `SteinerGrapher.cxx:20-123` + `create_enhanced_steiner_graph` `:900-1082` |
| 3 | `find_steiner_terminals` — `:711-729` | `SteinerGrapher.cxx:520-543` |
| 4 | `find_peak_point_indices` — `:733-953` | `SteinerGrapher.cxx:308-337` (blob overload) / `:339-517` (live overload) |
| 5 | `calc_charge_wcp` — `:955-1035` | `Facade_Cluster.cxx:1031-1112` |
| 6 | `form_cell_points_map` — `:1038-1056` | `SteinerGrapher.cxx:547-581` |
| 7 | `establish_/remove_same_mcell_steiner_edges` — `PR3DCluster_graph.h:10-130` | `SteinerGrapher.cxx:583-706` / `:709-740`, and `:844-896` for the steiner-graph pass |
| 8 | `get_extreme_wcps` — `PR3DCluster_path.h:80-240` | `Facade_Cluster.cxx:3082-3241` |
| 9 | `recover_steiner_graph` — `:77-180` | **no counterpart** — see **D11** |

**Out of scope, and NOT claimed clean:**

* **The base-graph construction** — prototype `Create_graph` /
  `Establish_close_connected_graph` / `Connect_graph`
  (`PR3DCluster_graph.h:132-1518`, ~1400 lines) ↔ toolkit
  `make_graphs.cxx` + `connect_graph{,_closely,_ctpc,_relaxed}.cxx`
  (~3200 lines). This is the *input* to Steiner, not Steiner itself. It is
  large enough to deserve its own audit and is entirely unread here.
* **The retile step** — prototype `Improve_PR3DCluster_2` +
  `calc_sampling_points` + `Create_point_cloud`
  (`PR3DCluster_steiner.h:13-20`) ↔ toolkit `RetileCluster`
  (`m_grapher_config.retile->mutate`, `CreateSteinerGraph.cxx:191`). A separate
  component with its own porting history; the two are assumed equivalent here,
  not verified.
* Everything downstream that *reads* `steiner_graph` / `steiner_pc`.

---

## §1 How far to trust each finding

Following doc pr/28 §3b's convention, because these are **not** equally solid
and anyone deciding what to fix must know which is which.

**Tier 1 — independently read in both trees by the author.** The
`create_steiner_graph` ↔ `process_cluster_steiner` orchestration (§3), the
whole of `Create_steiner_tree` ↔ `create_steiner_tree` +
`create_enhanced_steiner_graph`, `find_steiner_terminals`,
`find_peak_point_indices`, `calc_charge_wcp` (both branches),
`form_cell_points_map`, both `establish_same_mcell_steiner_edges` flags,
`get_extreme_wcps`, the Dijkstra/Voronoi block, and both graph type
declarations. Findings **D1, D2 (mechanism), D3, D4 (the fact), D5, D6, D7,
D11** and every row of §5 rest on this tier.

**Tier 2 — single-pass observation, stated as such.** **D2's *reach*** (how
many real SBND points the two `calc_charge_wcp` branches actually separate),
**D8's** practical frequency, **D9**, **D10**. The anchors are correct; the
consequences are inference. Treat these as leads to confirm, not as established
the way D1/D3 are.

pr/28's first draft carried a `saved_skip` "divergence" that a second read
disproved. Re-read before acting on anything in tier 2.

---

## §2 The algorithm itself — **MATCH**, and this is the load-bearing check

Before any constant comparison matters, the question is whether the toolkit
solves the same problem with the same method. It does.

The prototype includes `WCPPaal/steiner_tree_greedy.h`
(`PR3DCluster_steiner.h:2`) and hand-inlines the *Mehlhorn* construction:
Dijkstra from all terminals at once with a nearest-terminal recorder → a
terminal-distance graph → the induced edge set. The toolkit vendors the same
PAAL pieces in `clus/src/PAAL.h` and its `Weighted::voronoi`
(`clus/src/Graphs.cxx:142-176`) is the prototype's block character for
character:

| | prototype `:454-458` | toolkit `Graphs.cxx:163-174` |
|---|---|---|
| multi-source Dijkstra | `boost::dijkstra_shortest_paths(*graph, terminals.begin(), terminals.end(), …)` | identical |
| comparator | `paal::utils::less()` | `PAAL::less()` |
| combine / inf / zero | `closed_plus<Weight>()`, `numeric_limits<Weight>::max()`, `0` | identical |
| visitor | `paal::detail::make_nearest_recorder(nearest_terminal_map, last_edge, on_edge_relaxed{})` | `PAAL::make_nearest_recorder(…, boost::on_edge_relaxed{})` |
| terminal seeding | `nearest_terminal_map[t] = t` for each terminal | identical |

The charge-weighting formula is also exact. Prototype `:615`

```cpp
float temp_dis = dis * (factor1 + factor2 * (0.5*Q0/(Qs+Q0) + 0.5*Q0/(Qt+Q0)));
```

toolkit `SteinerGrapher.cxx:1067-1070`, with `Q0 = 10000`, `factor1 = 0.8`,
`factor2 = 0.4` set at `:86-88` and defaulted identically in
`Graphs.h:98-103`. The commented-out `nsteiner`-dependent factor variants
(prototype `:600-611`) are absent from the toolkit — correctly, they are dead in
the prototype too.

**So every finding below is about what is fed to this solver, not about the
solver.**

---

## §3 Orchestration — step for step

| # | prototype `create_steiner_graph` | toolkit `process_cluster_steiner` | verdict |
|---|---|---|---|
| 1 | `Improve_PR3DCluster_2` + `calc_sampling_points` + `Create_point_cloud` `:13-20` | `retile->mutate(*src->node())` `CreateSteinerGraph.cxx:191-196` | out of scope (§0) |
| 2 | `new_cluster->Create_graph(ct_point_cloud, point_cloud)` `:21` — base graph built with **both** the CTPC and the *original* cluster's cloud as reference | `new_cluster.find_graph("ctpc_ref_pid", *src, …)` `:199` | MATCH in shape; the graph builders themselves are out of scope |
| 3 | `establish_same_mcell_steiner_edges(gds, /*disable_dead_mix_cell=*/false)` `:27` | `sg.establish_same_blob_steiner_edges("ctpc_ref_pid", false)` `:210` | **MATCH**, including the `false` |
| 4 | `get_two_boundary_wcps()` `:30` | `new_cluster.get_two_boundary_wcps()` `:215` | MATCH |
| 5 | `dijkstra_shortest_paths(first)` + `cal_shortest_path(second)` `:34-35` | `graph_algorithms("ctpc_ref_pid").shortest_path(first, second)` `:224` | MATCH — and in both trees the path is computed **while** the same-blob edges are in the graph |
| 6 | `remove_same_mcell_steiner_edges()` `:39` | `sg.remove_same_blob_steiner_edges("ctpc_ref_pid")` `:229` | MATCH |
| 7 | `Create_steiner_tree(…, old_mcells = this->mcells, flag_path = true, disable_dead_mix_cell = false)` `:46` | `sg.create_steiner_tree(src, path_point_indices, "ctpc_ref_pid", "steiner_graph", false, "steiner_pc")` `:234` | MATCH at the call — the `false` **is** passed. What happens to it inside is **D2** |
| 8 | `establish_same_mcell_steiner_edges(gds, true, /*flag=*/2)` `:47` — on the *original* cluster | `establish_same_blob_steiner_edges_steiner_graph(steiner_result, m_cluster)` `SteinerGrapher.cxx:98` | same position in the sequence; the grouping inside is **D3** |
| 9 | build `point_cloud_steiner_terminal` `:59-68` | none — a `flag_steiner_terminal` column on `steiner_pc` `SteinerGrapher.cxx:1037-1039` | **not a defect** — see §5.6 |

Inside `Create_steiner_tree` the phase order also matches:

| phase | prototype | toolkit |
|---|---|---|
| find terminals | `:186` | `SteinerGrapher.cxx:32` |
| filter against the original cluster's blobs | `:255-347` (`flag_remove`) | `:46` `filter_by_reference_cluster` |
| filter against the path | `:351-371`, gated on `(!flag_remove)` | `:57` `filter_by_path_constraints`, applied to the survivors |
| add extreme points | `:392-403` | `:66-67` |
| build the tree | `:409-621` | `:92` `create_enhanced_steiner_graph` |

The prototype's `if ((!flag_remove) && flag_path)` composition is exactly the
toolkit's sequential filtering — a terminal already dropped by the blob check is
never path-checked. **Equivalent, verified.**

---

## §4 Divergences

### D1 — the terminal filter lost the prototype's ±1 wire tolerance — **behaviour-changing, OPEN**

**This is a class of port bug, not a constant mismatch: one toolkit function
serves two prototype call sites that use different tolerances, and it implements
the wrong one for the Steiner terminal filter.**

The prototype tests point-in-reference-blob **twice**, with **different**
tolerances:

```cpp
// (A) Create_steiner_tree terminal filter -- PR3DCluster_steiner.h:285-290
//     (repeated verbatim at :310-315 for time_slice+1 and :336-341 for -1)
if (cloud.pts[*it].index_u <= u1_high_index + 1 &&
    cloud.pts[*it].index_u >= u1_low_index  - 1 && …)      // <-- ONE WIRE OF SLACK

// (B) get_extreme_wcps filter -- PR3DCluster_path.h:111-116
if (cloud.pts[i].index_u <= u1_high_index &&
    cloud.pts[i].index_u >= u1_low_index && …)             // <-- NO SLACK
```

The toolkit has **one** implementation,
`Cluster::check_wire_ranges_match` (`Facade_Cluster.cxx:3323-3363`), with no
tolerance — and the removal was deliberate:

```cpp
// Facade_Cluster.cxx:3344-3345
// NO tolerance added - use exact wire ranges like prototype
// Removed: u_min = u_min - 1; u_max = u_max + 1; etc.
```

It is called from `is_point_spatially_related_to_time_blobs`
(`:3243-3320`), which both the terminal filter (`flag_nearby_timeslice = true`,
via `SteinerGrapher.cxx:291`) and `get_extreme_wcps`
(`flag_nearby_timeslice = false`, `:3106`) use. So the toolkit is **correct for
(B) and one wire too strict on each side of all three planes for (A)**.

> **This is not the M7 half-open trap, and a reader who knows M7 will otherwise
> dismiss it.** `check_wire_ranges_match` uses `>= u_min && < u_max`
> (`:3352-3354`). Prototype `get_uwires().back()->index()` is the **last** wire,
> inclusive; toolkit `u_wire_index_max()` is exclusive. So `<= high` → `< max`
> is the **correct** M7 translation and must not be "fixed". The divergence is
> the missing `+1` / `-1`, which would translate to `< u_max + 1` and
> `>= u_min - 1`.

**Effect:** terminals sitting on the edge of, or one wire outside, every
reference blob in slices *t*, *t±1* are dropped by the toolkit and kept by the
prototype. Terminals are the seeds of the whole tree, so a dropped terminal
removes a branch of the skeleton.

The ±1 time-slice half of the prototype's check **is** ported
(`:3297-3313`, the `{-1, +1}` loop) — only the wire tolerance is missing.

### D2 — `disable_dead_mix_cell` is dropped on the edge-weight charge path — **mechanism confirmed, reach unverified, OPEN**

`create_steiner_tree` is called with `disable_dead_mix_cell = false`
(`CreateSteinerGraph.cxx:234`) and correctly forwards it to
`find_steiner_terminals` (`SteinerGrapher.cxx:32`). It then calls

```cpp
// SteinerGrapher.cxx:92-93   -- no disable_dead_mix_cell argument
auto steiner_result = Graphs::Weighted::create_enhanced_steiner_graph(
    base_graph, steiner_terminals, original_pc, m_cluster, charge_config);
```

and the parameter defaults to **`true`** (`SteinerGrapher.h:341`). It reaches
`calculate_vertex_charges` (`SteinerGrapher.cxx:1021-1027`) → `calc_charge_wcp`,
whose result is `Qs`/`Qt` in the edge-weight formula.

The prototype computes the same charges with **`false`**
(`PR3DCluster_steiner.h:514` and `:521`, inheriting `Create_steiner_tree`'s
`disable_dead_mix_cell` parameter, which `create_steiner_graph:46` passes as
`false`).

So the toolkit takes the *other* branch of `calc_charge_wcp` when weighting
steiner edges. The two branches (`Facade_Cluster.cxx:1063-1102`) differ as:

| branch | which planes enter the RMS |
|---|---|
| `true` | all three, then **dead** planes are subtracted — dead = `charge_uncertainty(pt, plane) > 1e10` (`:1027-1029`) |
| `false` | only planes with `charge_value != 0` |

**Reach not measured.** These predicates are independent: a live plane can read
`charge_value == 0`, and a dead plane can carry a nonzero `charge_value`. Every
point where the two predicates disagree gets a different `Qs`/`Qt` and hence a
different edge weight. **How often that happens on real SBND points was not
measured** — the confirmed claim is the mechanism and the wrong argument, not a
count of affected edges. That measurement is one instrumented run, not a source
read, and none was made for this doc.

Note also that this puts the toolkit on the branch that depends on **D10**'s
unverified dead-plane representation — the two compound.

### D3 — the steiner-graph same-blob pass groups by the wrong blobs, and drops the prototype's exclusion — **behaviour-changing, OPEN**

The prototype's flag-2 pass needs each selected steiner point to know which
**original-cluster** mcell contains it. `Create_steiner_tree:537-581` computes
that explicitly, with **strict** wire bounds (no ±1 — this is call site (B)'s
tolerance, distinct from D1's (A)):

```cpp
// PR3DCluster_steiner.h:542-563
WCP::SlimMergeGeomCell *mcell = 0;
int time_slice = cloud.pts[index].mcell->GetTimeSlice();
if (old_time_mcells_map.find(time_slice) != old_time_mcells_map.end()) {
  for (…each old mcell in that slice…) {
    if (index_u <= u1_high_index && index_u >= u1_low_index && …) { mcell = mcell1; break; }
  }
}
point_cloud_steiner->AddPoint(p, temp_indices, mcell);      // mcell MAY BE NULL
```

and the flag-2 pass then **skips every point whose mcell is null**:

```cpp
// PR3DCluster_graph.h:90-91
for (size_t i=0;i!=cloud.pts.size();i++){
  if (cloud.pts.at(i).mcell==0) continue;                   // <-- the exclusion
```

The toolkit's counterpart groups by `majs[old_index]` — the point's blob in the
**retiled** cluster, which is the cluster the Steiner tree was built on, not the
original:

```cpp
// SteinerGrapher.cxx:854-861
for (const auto& [old_index, new_index] : result.old_to_new_index) {
    size_t blob_node_idx = majs[old_index];
    …
    cell_points_map[blob_node_idx].insert(new_index);
}
```

Two consequences, both real:

1. **Different grouping.** Retiled blobs and original-cluster blobs are not the
   same partition of space — that is the entire point of the retile step. Points
   grouped together by one are not necessarily grouped by the other, so a
   different set of intra-blob edges is added to the delivered steiner graph.
2. **No exclusion.** Every selected point is grouped in the toolkit. The
   prototype silently omits points that fall in no original mcell — precisely
   the points the retile invented — so those points get **no** same-blob edges
   there and do get them here.

There is no compensating guard anywhere in the toolkit path.

Everything *else* about the flag-2 pass matches: plain geometric distance with
**no** 0.8/0.9 factor (prototype `:115`/`:119`, toolkit `:888`), the
at-least-one-terminal condition (§5.2), and the duplicate-edge guard (§5.3).

### D4 — edge weights are `float` in the prototype, `double` in the toolkit — **fidelity**

```cpp
// prototype  PR3DCluster.h:30       using EdgeProp = boost::property<boost::edge_weight_t, float>;
// toolkit    Graphs.h:22            using edge_weight_type = double;
```

The type difference is **verified** (both declarations read). The consequence is
**inference**, stated as pr/28 T8 states its own: **fidelity, not determinism —
it means the toolkit cannot be bit-identical to the prototype even with
everything else in this document fixed.** Dijkstra distances, the
`total_distance` comparisons that pick the best inter-terminal edge, and the
charge-weight product all accumulate at a different precision, so tie-breaks
near equality can land differently.

### D5 — the graph types differ in out-edge storage — **structural, compensated**

```cpp
// prototype  PR3DCluster.h:33   adjacency_list<setS,  vecS, undirectedS, VertexProp, EdgeProp>
// toolkit    Graphs.h:23-29     adjacency_list<vecS,  vecS, undirectedS, …>
```

`setS` forbids parallel edges and makes `add_edge` return `.second == false` on
a duplicate; `vecS` permits them and always reports success. The toolkit
compensates with an explicit `boost::edge(a, b, g).second` test before **every**
`add_edge` on this family — `SteinerGrapher.cxx:657`, `:682`, `:889`, `:1072` —
so the resulting edge sets agree. Where the prototype relies on `.second` to
decide whether to record an edge for later removal (`PR3DCluster_graph.h:75-80`),
the toolkit's `if (ok)` (`:662`, `:688`) is preceded by the same test, so the
bookkeeping agrees too.

`setS` also means the prototype's `boost::out_edges` iterates in **pointer**
order; the toolkit's `vecS` iterates in insertion order. That is the same hazard
pr/28 §9.1 documents for `PRTrajectoryView`, and here it lands on the
**toolkit's** side of the ledger — see **D6**.

### D6 — edge deduplication: pointer order vs vertex order — **toolkit is deterministic, prototype is not**

The prototype collects the induced tree edges and dedups them by edge
descriptor:

```cpp
// PR3DCluster_steiner.h:505-506
boost::sort(tree_edges);
auto unique_edges = boost::unique(tree_edges);
```

`boost::detail::edge_desc_impl::operator<` compares the **edge-property
pointer**, so both the sort and the subsequent insertion order into
`graph_steiner` are heap-address dependent.

The toolkit deliberately replaced this, and says so:

```cpp
// SteinerGrapher.cxx:969-970
// Store as (vertex_pair, edge_type) so we can deduplicate and sort by stable vertex-index
// values rather than by edge_type (which embeds a pointer — non-deterministic across runs).
```

sorting and uniquing on `vertex_pair` (`:992-997`) and inserting in that order
(`:1046`).

**This makes the toolkit reproducible; it does not make the two agree.** Sorting
by vertex pair is *an* order, the prototype's pointer order is another, and no
claim is made that they coincide. Since neither graph permits parallel edges in
practice (§D5), the *content* is the same set of edges — only the insertion
order, and hence every downstream `out_edges` traversal, differs.

### D7 — path-skeleton sampling density — **behaviour-changing at the threshold**

Both trees resample the shortest path at 0.6 cm before using it as the
path-proximity reference cloud. They step differently:

```cpp
// prototype PR3DCluster_steiner.h:214-226
int num_steps = dis/step_dis;                       // floor, no +1  => spacing >= 0.6 cm
for (int qx = 0; qx != num_steps; qx++) { …add interpolant (qx+1)/num_steps… }
temp_pcloud.AddPoint(p, wire_index, 0);             // endpoint added a SECOND time
```
```cpp
// toolkit DynamicPointCloud.cxx:744-756
int num_points = int(dis / step) + 1;               // => spacing < 0.6 cm
for (int k = 0; k != num_points; k++) { …add interpolant (k+1)/num_points… }   // endpoint once
```

The duplicated endpoint is harmless (a KD-tree of duplicate points returns the
same nearest distance). The **density** is not: the toolkit samples strictly
finer, so `dis_3d` and all three `dis_2d` can only be smaller or equal. The
filter is `close_in_2d && dis_3d > 6 cm` with a 1.8 cm 2-D threshold
(`SteinerGrapher.cxx:213-237`, prototype `:366-370` — **thresholds match
exactly**), and finer sampling pushes *both* sides of that conjunction down, so
the net sign is empirical. Terminals sitting near either threshold can flip.

`step_dis = 0.6 cm` is identical on both sides (`SteinerGrapher.cxx:202`,
prototype `:206`).

### D8 — reference containment additionally requires the same APA **and** face — **forced by multi-APA, reach unverified**

`is_point_spatially_related_to_time_blobs` keys on
`apa → face → time_slice` and returns `false` if either the APA or the face of
the point's own blob is absent from the reference map
(`Facade_Cluster.cxx:3262-3266`). The prototype's `old_time_mcells_map` is keyed
by **time slice alone** (`PR3DCluster_steiner.h:238-249`) — uBooNE has one TPC,
so the concept does not exist there.

This is a necessary adaptation, not a defect. It is listed because it is
**behaviour-changing in principle at APA/face boundaries**: a point whose
reference blob lies in the neighbouring face is matched by the prototype's
logic-as-written and rejected here. How often that configuration arises on SBND
was not measured. Same treatment as pr/28 §4.3's `paf` guard row.

### D9 — extreme points make an index → 3-D point → index round trip — **benign, latent**

The prototype inserts extreme points as terminal indices directly:

```cpp
// PR3DCluster_steiner.h:395-396
if (steiner_terminal_indices.find(extreme_wcps.at(i).at(j).index) == …end())
    steiner_terminal_indices.insert(extreme_wcps.at(i).at(j).index);
```

`get_extreme_wcps` returns `WCPoint`s, which carry `.index`. The toolkit's
returns bare `geo_point_t` (`Facade_Cluster.cxx:3082`), so the index has to be
recovered by a KD query:

```cpp
// SteinerGrapher.cxx:268
auto closest_idx = find_closest_vertex_to_point(point);   // -> get_closest_point_index
```

The extreme point *is* a cluster point, so the query should return distance 0
and the same index. It can differ if two points coincide exactly (the KD tie is
arbitrary) or if `get_closest_point_index` searches a different scope than the
one `get_extreme_wcps` scanned. Neither was checked. Listed as latent rather
than as a defect.

### D10 — the dead-plane representation differs — **forced, reach unverified**

| | prototype | toolkit |
|---|---|---|
| what "bad plane" means | a **per-mcell** list, `mcell->get_bad_planes()` (`PR3DCluster_steiner.h:984`) | a **per-point-per-plane** uncertainty sentinel, `charge_uncertainty(pt, plane) > 1e10` (`Facade_Cluster.cxx:1027-1029`) |
| where it is used | `calc_charge_wcp`'s `disable_dead_mix_cell == true` branch only | same |

A blob-level bad plane and a point-level dead wire are not the same set. This is
forced by the data model (the toolkit has no per-blob bad-plane list), but it is
**the same unread question as D2's reach** — both rows rest on whether
`charge_uncertainty > 1e10` picks out the points the prototype's bad-plane list
would. One measurement settles both; neither is settled here.

### D11 — `recover_steiner_graph` is not ported — **gap, and it is the only MST in the prototype**

`PR3DCluster_steiner.h:77-180` reruns the Mehlhorn construction on the
*already-built* steiner graph and — unlike `Create_steiner_tree` — actually runs
the minimum spanning tree:

```cpp
// :155-156                                     (cf. :489-490, COMMENTED OUT in Create_steiner_tree)
boost::kruskal_minimum_spanning_tree(terminal_graph, std::back_inserter(terminal_edge));
```

producing `steiner_graph_selected_terminal_indices`. The toolkit has no such
function; `SteinerGrapher.h:299-300` records the omission:

> `m_steiner_graph_terminal_indices` removed — not populated; reserved for
> `recover_steiner_graph()` if that function is ported in the future.

**Whether this matters depends on whether any ported consumer needs
`steiner_graph_selected_terminal_indices`** — that consumer sweep was not done
here and belongs with the downstream audit, not this one.

---

## §5 Things that look like divergences and are not

Each of these cost a wrong conclusion on the first pass. They are recorded so
the next reader does not re-derive them.

**5.1 — The missing MST is *not* missing.** `Create_steiner_tree`'s call to
`boost::kruskal_minimum_spanning_tree` is **commented out** (`:489-490`), and
the line immediately above it (`:486`) pushes *every* best-inter-terminal edge
into `terminal_edge`. So the prototype's Steiner "tree" is not a tree — it keeps
all of them. The toolkit does exactly the same
(`SteinerGrapher.cxx:962-964`). **MATCH.** The MST lives only in the unported
`recover_steiner_graph` (D11).

**5.2 — The flag-2 if/else arms are identical in the prototype.**
`PR3DCluster_graph.h:114-122` branches on `flag_index1 && flag_index2` versus
`flag_index1 || flag_index2` and the two bodies are **character-identical** —
plain distance, same graph. The toolkit's single
`if (flag_steiner_terminal[i1] || flag_steiner_terminal[i2])`
(`SteinerGrapher.cxx:887`) is the correct collapse, not a dropped case.

**5.3 — The toolkit's duplicate-edge guards are not extra logic.** They restore
what `setS` gave the prototype for free (D5).

**5.4 — `Create_graph()` inside `Create_steiner_tree` is a no-op.**
`PR3DCluster_steiner.h:183` calls it, but both overloads open with
`if (graph != 0) return;` (`PR3DCluster_graph.h:516-517`, `:544-545`) and the
graph was built at `create_steiner_graph:21`. The toolkit's plain
`get_graph(graph_name)` (`SteinerGrapher.cxx:80`) is equivalent. **Do not read
this as a missing rebuild.**

**5.5 — `find_peak_point_indices`'s N×N vs upper-triangle loop.** The prototype
loops `j, k` over all pairs including `j == k` (`:883-890`); the toolkit uses
`k = j + 1` (`:464-470`) and says why. For an undirected graph with no
self-loops these build the same connectivity graph. **MATCH.**

**5.6 — `point_cloud_steiner_terminal` is built and never read.** The prototype
constructs a terminal-only KD cloud (`:59-68`) and exposes it
(`PR3DCluster.h:58`), but the only two references in the entire prototype tree
are **commented out** (`apps/wire-cell-graph.cxx:525`,
`apps/wire-cell-tracking.cxx:681`). The toolkit's `flag_steiner_terminal` column
on `steiner_pc` is therefore a faithful drop of dead output, not a lost product.

**5.7 — The 1.8 cm 2-D threshold means the same thing on both sides.** The
prototype interpolates *wire indices* linearly along the path (`:220-223`),
which invites the conclusion that its 2-D distance is in wire units. It is not:
`ToyPointCloud::get_closest_2d_dis` (`data/src/ToyPointCloud.cxx:461-472`)
projects as `(x, cos(angle)·z − sin(angle)·y)` in **cm** — the identical formula
to the toolkit's `add_proj` (`DynamicPointCloud.cxx:736-738`). The interpolated
wire indices never enter the test.

**5.8 — `get_extreme_wcps` matches, including the 5 cm grouping.** Same PCA main
axis with the `y < 0` flip, same 8 extremes, same "within 5 cm joins the
existing group" tail (prototype `:214-236` ↔ toolkit
`Facade_Cluster.cxx:3215-3238`), same `excluded_points` skip. And its reference
filter genuinely has **no** wire tolerance and **no** ±1 time slice on both
sides — which is why D1 is a mis-shared function rather than a wrong constant.

**5.9 — `calc_charge_wcp` is otherwise a faithful port.** The RMS-of-squares
construction, the `ncharge > 1` gate that returns 0 otherwise, and the
`flag_charge_u && flag_charge_v && flag_charge_w` return are line-for-line
(`prototype :1028-1033` ↔ `Facade_Cluster.cxx:1105-1111`). The `4000` charge cut
is the prototype's default (`PR3DCluster.h:129`) and is passed explicitly on the
toolkit side at every call (`SteinerGrapher.cxx:353`, `:1025`). Only the
dead-plane test differs (D10).

**5.10 — `form_cell_points_map` and `find_steiner_terminals` match.** Prototype
loops `mcells` and calls `find_peak_point_indices` on one mcell at a time
(`:719-726`); the toolkit loops the blob→points map (`:536-540`). Same
partition, same per-blob peak finding, same union.

---

## §6 Determinism — clean

The mechanical sweep CLAUDE.md §2 requires, over `SteinerGrapher.cxx`,
`SteinerGrapher_helpers.cxx`, `CreateSteinerGraph.cxx` and `Graphs.cxx`:

| what | verdict |
|---|---|
| `boost::out_edges` calls | **none** in this family |
| pointer-keyed containers | exactly one — `std::map<const Facade::Blob*, size_t> blob_to_node_idx` (`SteinerGrapher.cxx:320`). **Lookup-only, never iterated**, and it lives in the dead overload (§7). Benign |
| blob→points maps | keyed by **blob node index** (`size_t`), with the reason stated at `SteinerGrapher.h:66-70` and `:561-564`. Deterministic |
| tree-edge dedup | vertex-pair ordered, not pointer ordered — **D6**. Deterministic |
| `m_added_edges_by_graph` | a `std::vector`, with the rationale at `SteinerGrapher.h:230-232`; removal is order-insensitive. Benign |
| cluster processing order | `grouping.children()` order, and the beam-window gate preserves it (`CreateSteinerGraph.cxx:124-126`). Deterministic |

**No determinism defect found in the Steiner-building family.** This is a
meaningfully better result than pr/28 got for `improve_vertex`, and it is worth
noting that the toolkit here is *more* deterministic than the prototype (D6).

The `connect_graph*` family that feeds this is **not** covered by that sweep and
is not claimed clean (§0).

---

## §7 Pre-existing dead code (mentioned, not touched)

Per CLAUDE.md's tie-breaker — report, do not fix in the same change:

* **`find_peak_point_indices(const std::vector<const Facade::Blob*>&, …)`**
  (`SteinerGrapher.cxx:308-337`, declared `SteinerGrapher.h:138-139`) has **no
  callers** anywhere in `clus/`. It is the literal transcription of the
  prototype signature (`SMGCSelection mcells`); the live path is the
  `vertex_set` overload at `:339`. This resolves the "two toolkit overloads
  against one prototype function" question: the divergence is a dead duplicate,
  not a behavioural split.
* **`improve_grapher{,_1,_2}`** (`SteinerFunctions.h:16-19`) are **declared and
  never defined**, with a header comment saying so. They are reserved stubs for
  the unported single-cluster `ImprovePR3DCluster` path.

---

## §8 Summary table

| # | what | severity | prototype | toolkit | tier |
|---|---|---|---|---|---|
| **D3** | steiner same-blob edges grouped by retiled blob, and the null-mcell exclusion is absent | **behaviour-changing** | `PR3DCluster_steiner.h:542-563` + `PR3DCluster_graph.h:90-91` | `SteinerGrapher.cxx:854-861` | 1 |
| **D1** | terminal filter lost the ±1 wire tolerance (one function, two prototype tolerances) | **behaviour-changing** | `PR3DCluster_steiner.h:285-290, 310-315, 336-341` | `Facade_Cluster.cxx:3344-3354` | 1 |
| **D2** | `disable_dead_mix_cell` dropped on the edge-weight charge path | **behaviour-changing** (reach unverified) | `PR3DCluster_steiner.h:514, 521` | `SteinerGrapher.cxx:92` / `SteinerGrapher.h:341` | 1 / 2 |
| **D7** | path-skeleton sampled finer (`+1` vs floor) | behaviour-changing at the threshold | `PR3DCluster_steiner.h:214-226` | `DynamicPointCloud.cxx:744` | 1 |
| **D8** | containment additionally requires same APA and face | forced (multi-APA); reach unverified | no counterpart | `Facade_Cluster.cxx:3262-3266` | 2 |
| **D10** | dead plane = per-blob list vs per-wire uncertainty sentinel | forced; reach unverified, **same question as D2** | `PR3DCluster_steiner.h:984` | `Facade_Cluster.cxx:1027-1029` | 2 |
| **D4** | edge weight `float` → `double` | fidelity — blocks bit-identicality | `PR3DCluster.h:30` | `Graphs.h:22` | 1 (fact) |
| **D6** | tree-edge dedup by vertex pair, not edge pointer | **toolkit fixes prototype nondeterminism** | `PR3DCluster_steiner.h:505-506` | `SteinerGrapher.cxx:969-997` | 1 |
| **D5** | `setS` → `vecS` out-edge storage | structural, compensated | `PR3DCluster.h:33` | `Graphs.h:23-29` | 1 |
| **D9** | extreme points round-trip index → point → index | benign, latent | `PR3DCluster_steiner.h:395` | `SteinerGrapher.cxx:268` | 2 |
| **D11** | `recover_steiner_graph` not ported (the only MST) | gap; consumer impact unassessed | `PR3DCluster_steiner.h:77-180` | none | 1 |

---

## §9 What is NOT claimed

* **No measurement.** No event was run for this document. Every statement is a
  source read, and every "this changes the output" is a mechanism argument, not
  an observed diff. D1, D2, D3 and D7 all deserve a single-event before/after
  before anyone acts on them — pr/28 §7.4 and §8.4 show why: the calib-JSON
  noise floor on this chain has been 268 and 356 leaves in two different
  sessions, so a same-binary repeat is mandatory before attributing anything.
* **No ranking against physics.** D3 is ranked first because it changes the most
  about the delivered graph, not because it is known to move a vertex, a score,
  or an efficiency.
* **No recommendation.** The porting dictionary has no Steiner section
  (`porting_dictionary.md:266` is an empty `⚠ xxx` placeholder), so under
  CLAUDE.md §5 rule 4 every divergence here is undocumented. Both readings are
  presented; the choice is the owner's. Filling that dictionary section is the
  natural companion to whatever is decided.
* **The base-graph builders and the retile step are unaudited and not claimed
  clean** (§0). D3 in particular sits on the retile/original-cluster boundary,
  so a decision about it may need that audit first.
* **Downstream consumers of `steiner_graph` / `steiner_pc` were not swept.**
  That is what D11 and §5.6 both hand off.

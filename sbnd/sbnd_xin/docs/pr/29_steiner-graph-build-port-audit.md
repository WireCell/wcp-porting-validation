# doc pr/29 — Steiner graph building: prototype-fidelity audit

**Why.** Owner request, following doc pr/28: *"do a similar audit for the Steiner
Graph Building, I want to see the divergence between Prototype code and
toolkit."*

**Status.** **Audit only. No code is changed and none is proposed.** The owner's
instruction was explicit — *"Please do not change any code yet."* Every finding
below carries an anchor on both sides and a severity verdict; nothing carries a
patch. The behaviour-changing items would all alter production output
unconditionally, i.e. CLAUDE.md **escalation rule 1** — they are reported and
stop there.

> **2026-08-04 — D1 and D12 are FIXED, SBND DEFAULT ON; see §11.** Both ship as
> config knobs whose off state is the historical behaviour (compiled config
> byte-identical, md5 in §11.5), with 7 new doctests pinning the two convention
> traps. On the event served on port 5017 (SBND evt 388) the legacy filter was
> discarding **47.7%** of all Steiner terminals and leaving **24** clusters
> below the two-terminal minimum; with both knobs on that is 20.0% and 3.
> `numu_score` moves −3.199 → −2.166, the event label does not move, and the
> pi0 candidate disappears — reported, not tuned. **Both are SBND PRODUCTION
> DEFAULTS as of the owner flip on 2026-08-04** (*"these are improvements, or
> bug fix, for SBND, right? They should be on"*): a bare run is now the fixed
> chain.

> **2026-08-04 second pass — D2 is FIXED and SBND-ON; D7 and D3 are now
> DISMISSED. §10.1 is down to one open row. See §12 and §10.2.8/§10.2.9.**
> Owner: *"I feel like D7 is not a problem, D2 should be fixed, is D3 a problem?
> I assume toolkit is better right?"* and, on the knob, *"it should also be
> default on. Since we are doing validation and improvements."*
> * **D2 fixed** — `edge_charge_forward_dead_mix`, C++ default OFF (so no other
>   detector moves), **SBND ON**. On evt 388 it alone moves segments 85 → 88,
>   vertices 129 → 133, `numu_score` −0.905 → −0.728, `kine_reco_Enu` 2816.1 →
>   2811.1 MeV; the event label does not move (§12.4). 4 new doctests pin the
>   claim that the two `calc_charge_wcp` branches are not interchangeable.
> * **D7 dismissed** — the toolkit resamples the path skeleton *finer* at the
>   same 0.6 cm target. That is a strictly more accurate nearest-distance, not a
>   lost behaviour (§10.2.8).
> * **D3 dismissed, and the owner's reading is confirmed by the prototype
>   itself** — the prototype groups by the **retiled** blob one step earlier, at
>   flag=1, and only reaches for original mcells at flag=2 because that is the
>   field which survives `delete new_cluster`. The toolkit's `steiner_pc` is a
>   Dataset with no pointer to dangle, so it has no such constraint (§10.2.9).
>   **But the dismissal comes with a coupling the owner should see: D1/D12,
>   now ON, move this axis further from the prototype** — §10.2.9's last
>   paragraph.
>
> **§13 collects everything still open** — the single-event measurement debt
> behind all three flips, the two unaudited stages D3's dismissal leans on,
> three source observations reported-not-fixed, and the still-empty porting
> dictionary section.

> **2026-08-04 revision — read §10 first.** The owner asked for the list to be
> **filtered**: *"we can skip the ones that are improvements over the previous
> prototype … only focus on the one that are bugs or missing from the port,"*
> with a warning that the toolkit's half-open `[min,max)` wire convention
> against the prototype's inclusive `[min,max]` *"may trick some code change."*
> Acting on that, every row was re-read against the live tree. Result: **§10**
> carries the filtered list — **five** rows survive as port defects, **seven**
> are dismissed with the evidence that dismisses them, and **one new defect
> (D12) was found by taking the convention warning seriously and applying it to
> the *time* axis as well as the wire axis.** D12 also **falsifies a sentence in
> D1's original write-up**, corrected in place and flagged there.

**Headline.** The **algorithm is the same** — the toolkit vendors the same PAAL
routines and its Voronoi/Dijkstra block is a character-for-character port (§2).
The divergences are in the *inputs* to that algorithm, and the largest are:

* **D12** — the ±1 **time-slice** fallback in the terminal filter is **dead
  code**. `slice_index_min()` is in **ticks**, blobs are one slice = `tick_span`
  ticks apart (SBND `nticks_live_slice: 4`), so `current_time_slice ± 1` can
  never name a real slice. Every other adjacent-slice lookup in the toolkit —
  including the one in the retile step that *produces* this very cluster — uses
  `± tick_span`. Found only on the 2026-08-04 pass.
* **D3** — the steiner-graph same-blob edge pass groups points by the **retiled
  cluster's own blob** instead of the prototype's **original-cluster mcell found
  by strict wire containment**, and the prototype's exclusion of points that
  land in *no* original mcell has no counterpart. This changes which edges exist
  in the delivered steiner graph.
* **D1** — one toolkit function (`check_wire_ranges_match`) serves **two**
  prototype call sites that use **different wire tolerances**, and it implements
  the wrong one for the Steiner terminal filter. Not an M7 `<`/`<=` issue — see
  the row.

**D1 and D12 land on the same filter.** `flag_nearby_timeslice = true` is passed
from exactly one place (`SteinerGrapher.cxx:291`, the terminal filter);
`get_extreme_wcps` passes `false`. So the two independent tightenings — no wire
slack, no adjacent-slice fallback — compound on the Steiner terminal set and
nothing else.

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

# 2026-08-04 re-read (the sec.10 pass) was done at toolkit 23bd6783.  Every line
# anchor in this document was re-validated first, and NONE had drifted -- this
# command prints nothing, i.e. not one Steiner-family source changed between the
# two commits:
git diff --stat ea1a7e3d..HEAD -- \
    clus/src/SteinerGrapher.cxx clus/src/SteinerGrapher.h \
    clus/src/Facade_Cluster.cxx clus/src/CreateSteinerGraph.cxx \
    clus/src/DynamicPointCloud.cxx clus/inc/WireCellClus/Graphs.h

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

# --- sec.10 / D12: the tick-vs-slice unit proof (all source, no run needed) ---
grep -n "unit: tick"            clus/inc/WireCellClus/Facade_Blob.h    # :33 slice_index_min
grep -n "slice_index_min"       aux/src/SamplingHelpers.cxx            # :90  islice->start()/tick
grep -n "nticks_live_slice"     cfg/pgrapher/experiment/sbnd/clus.jsonnet  # :147  -> 4
grep -n "tick_span"             clus/src/improvecluster_1.cxx          # :295, :313, :656, :697
grep -n "aligned_tick\|tick_span" clus/src/retile_cluster.cxx          # :256, :508, :528
grep -n "offset : {-1, 1}"      clus/src/Facade_Cluster.cxx            # :3302  <-- the defect

# --- sec.10 / D11: is recover_steiner_graph reachable in the ported chain? ---
grep -rn "recover_steiner_graph" prototype_base/pid/apps/ | grep -v "//"

# --- sec.10.2.9 / D3: the prototype groups by the RETILED blob at flag=1 ---
grep -n "establish_same_mcell_steiner_edges" prototype_base/pid/src/*.h prototype_base/pid/src/*.cxx
sed -n '30,46p'  prototype_base/pid/src/PR3DCluster_graph.h    # flag==1: point_cloud, retiled
sed -n '84,100p' prototype_base/pid/src/PR3DCluster_graph.h    # flag==2: point_cloud_steiner
sed -n '50,58p'  prototype_base/pid/src/PR3DCluster_steiner.h  # delete new_cluster / temp_holder
# the ONE other consumer of the steiner cloud's mcell, and its toolkit replacement
sed -n '355,380p' prototype_base/pid/src/PR3DCluster_path.h    # get_two_boundary_wcps(2)
sed -n '3423,3435p' clus/src/Facade_Cluster.cxx                # regular-PC scoring + terminal snap
```

**§12 (D2) is a code change and does have runs behind it.** Its repro:

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild && ls -la ../local/lib/libWireCellClus.so     # M1 freshness proof
./build/clus/wcdoctest-clus -tc="pr29 D2*"            # 4 cases / 12 assertions

# compiled-config gate (sec.12.5): pre-D2 from HEAD vs working tree forced off
mkdir -p /home/xqian/tmp/pr29d2cfg/base2
git archive HEAD cfg | tar -x -C /home/xqian/tmp/pr29d2cfg/base2
B=/home/xqian/tmp/pr29d2cfg/base2/cfg
WIRECELL_PATH=$B:/nfs/data/1/xqian/toolkit-dev/wire-cell-data \
  wcsonnet $B/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet | md5sum
wcsonnet --tla-code steiner_edge_charge_forward_dead_mix=false \
  sbnd_xin/wct-pr-perevt.jsonnet | md5sum          # must match the line above

# the matched pair + the noise floor (sec.12.4)
cd sbnd_xin
SBND_STEINER_EDGE_DEAD_MIX=0 PR_EXTRA_STAGES=pr_display PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr29-388-d2off  data 388
SBND_STEINER_EDGE_DEAD_MIX=0 PR_EXTRA_STAGES=pr_display PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr29-388-d2off2 data 388
PR_EXTRA_STAGES=pr_display PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr29-388-d2on   data 388
md5sum work-pr29-388-d2off{,2}/pr_evt388/calib-pr-evt388.json   # equal => zero floor
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

**D12 is tier 1 and then some** — it is the one row in this document whose
*mechanism* is closed by four mutually independent source facts (§4 D12, points
1–4), two of which are the toolkit contradicting itself in the same call chain.
Its unmeasured half is only the *count* of terminals affected. **D11's
dismissal** (§10.2.4) is likewise tier 1: a complete grep of
`prototype_base/pid/apps/`, not a sample.

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

### D1 — the terminal filter lost the prototype's ±1 wire tolerance — **behaviour-changing, FIXED**

> **FIXED 2026-08-04, SBND DEFAULT ON — see §11.** Kept as the pre-fix
> record; line anchors below are pre-fix.

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

> ~~The ±1 time-slice half of the prototype's check **is** ported
> (`:3297-3313`, the `{-1, +1}` loop) — only the wire tolerance is missing.~~
>
> **CORRECTION, 2026-08-04 — that sentence was wrong.** The `{-1, +1}` loop is
> *present* but **inert**: it steps the map key by one **tick**, and the key is a
> slice start, which advances by `tick_span` (4 on SBND). It never resolves. So
> the toolkit drops **both** halves of the prototype's slack, not one. Written up
> as **D12**; the two are independent defects on one filter and are listed
> separately so either can be fixed alone.
>
> How the error was made, since it is the instructive part: the loop was read
> for *shape* (`{-1, +1}` over `current_time_slice`) and the shape matched the
> prototype exactly. What was never asked is what the loop variable is **counted
> in**. That is the same question the wire ranges force — and it was asked there
> and not here.

### D2 — `disable_dead_mix_cell` is dropped on the edge-weight charge path — **mechanism confirmed, reach unverified, FIXED**

> **FIXED 2026-08-04, SBND DEFAULT ON — see §12.** The entry below is the
> pre-fix record; its `SteinerGrapher.cxx:92` anchor is the pre-fix line.

`create_steiner_tree` is called with `disable_dead_mix_cell = false`
(`CreateSteinerGraph.cxx:234`) and correctly forwards it to
`find_steiner_terminals` (`SteinerGrapher.cxx:32`). It then calls

```cpp
// SteinerGrapher.cxx:92-93 (PRE-FIX; now :105-113)  -- no disable_dead_mix_cell argument
auto steiner_result = Graphs::Weighted::create_enhanced_steiner_graph(
    base_graph, steiner_terminals, original_pc, m_cluster, charge_config);
```

and the parameter defaults to **`true`** (`SteinerGrapher.h:392`). It reaches
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

### D3 — the steiner-graph same-blob pass groups by a different blob, and drops the prototype's exclusion — **DISMISSED**

> **DISMISSED 2026-08-04 — see §10.2.9.** The two source readings below are
> correct and are kept. What changes the verdict is context they lack: the
> prototype groups by the RETILED blob one step earlier (flag=1, scored a
> MATCH in §3), and reaches for original mcells at flag=2 only because
> `point_cloud_steiner` outlives `delete new_cluster`. The heading's "wrong
> blobs" is therefore too strong.

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

### D7 — path-skeleton sampling density — **DISMISSED (improvement)**

> **DISMISSED 2026-08-04 — see §10.2.8.** The toolkit samples the same path
> finer than the prototype's own 0.6 cm target; a denser sample is a more
> accurate nearest-distance, not a lost behaviour.

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

> **2026-08-04 — the sweep was done, and it closes this row.** See §10.2.4:
> `recover_steiner_graph()` is **never called** by the production apps this port
> targets. D11 is dismissed.

### D12 — the ±1 time-slice fallback is dead code: the key is in **ticks**, not slices — **behaviour-changing, FIXED** *(added 2026-08-04)*

> **FIXED 2026-08-04, SBND DEFAULT ON — see §11.** Kept as the pre-fix
> record; line anchors below are pre-fix.

**This is the same class of bug the owner's warning names, applied to the time
axis instead of the wire axis: a loop that is shape-correct and unit-wrong.**

The prototype's terminal filter, having failed to find a containing mcell in the
point's own time slice, retries in slice *t+1* and then *t−1*
(`PR3DCluster_steiner.h:297-347`). Its map is keyed by slice number, so `+1` is
the next slice.

The toolkit reproduces the shape faithfully:

```cpp
// Facade_Cluster.cxx:3302-3305
for (int offset : {-1, 1}) {
    int adjacent_time_slice = current_time_slice + offset;
    auto time_it_adj = face_it->second.find(adjacent_time_slice);
```

but its map is **not** keyed by slice number. It is keyed by
`blob->slice_index_min()` (`Facade_Cluster.cxx:323`), and that field is
documented in the struct as **`// unit: tick`** (`Facade_Blob.h:33`), produced as
`islice->start()/tick` (`aux/src/SamplingHelpers.cxx:90`). Slice starts therefore
advance by one **`tick_span`**, not by one. On SBND `tick_span` is **4**
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet:147` `nticks_live_slice: 4`, and
`img.jsonnet:133` `span=4`).

`find(t ± 1)` asks for a slice starting one tick off the grid. **No blob starts
there.** Both lookups miss on every point of every event — *given
`tick_span ≥ 2`*, which is the honest statement of the precondition. Only a
detector configured with one tick per slice would make `± 1` accidentally
correct; SBND is at 4. That precondition is also exactly why a fix must read
`get_nticks_per_slice()` rather than hard-code a stride (§10.1).

**Four independent confirmations, all source-level — no run is needed:**

1. **The unit is declared.** `Facade_Blob.h:33`, `// unit: tick`, on the field
   that becomes the map key.
2. **The blob spans a whole slice.** `slice_index_max() - slice_index_min()` is
   used across the tree as `ntime_ticks` — e.g. `TrackFitting.cxx:2713`,
   `NeutrinoStructureExaminer.cxx:1210`, whose comment reads *"children()[0]
   spans exactly one readout bin."* A half-open `[min, max)` tick interval one
   slice wide means consecutive starts differ by exactly that width.
3. **The toolkit's own idiom for "adjacent slice" is `± tick_span`, everywhere
   else.** `improvecluster_1.cxx:656` `find(time_slice + tick_span)`, `:697`
   `find(time_slice - tick_span)`, `:313` `for (…; time_slice += tick_span)`,
   with the span taken from the canonical accessor
   `m_grouping->get_nticks_per_slice().at(apa).at(face)` (`:295`).
   `Facade_Cluster.cxx:3302` is the **only** `± 1` of its kind in the family.
4. **The producer of this very cluster agrees.** `retile_cluster.cxx:256` snaps
   retiled blobs onto the grid — `aligned_tick = round(t/tick_span)*tick_span` —
   and `:508`/`:528` do *their* adjacent-slice lookups as
   `time_slice ∓ tick_span`. The retile step and the filter that consumes its
   output disagree about what "next slice" means, inside one call chain.

Point 4 also settles the one thing that could have rescued the loop: the
retiled and reference clusters share a grid, so it is not that the ±1 lands on
some other valid key — there is no key one tick away at all.

**Effect.** The prototype keeps a terminal if it is contained in a reference
mcell in slice *t*, *t+1* **or** *t−1*. The toolkit keeps it only for *t*.
Terminals near a blob's time boundary — exactly the ones a slice-boundary
straddle would displace — are dropped. Same direction as D1, same filter, and
the two stack: a terminal needs to satisfy a wire test that is one wire tighter
on all six bounds *and* has lost two of its three chances to pass it.

**Blast radius is exactly one call site.** `flag_nearby_timeslice = true` is
passed only from `SteinerGrapher.cxx:291`
(`is_point_spatially_related_to_reference` → `filter_by_reference_cluster`).
`get_extreme_wcps` passes `false` (`Facade_Cluster.cxx:3106`) and so never
enters the dead loop — it does not want the fallback anyway, matching
`PR3DCluster_path.h:99-121`, which has no ±1 either. **So D12 cannot be
dismissed as "compensated elsewhere," and equally it cannot leak into
`get_extreme_wcps` when fixed.**

**Unmeasured:** how many terminals actually sit at a slice boundary and would be
rescued. The mechanism is proven; the count is not. Same standard as D2's reach.

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

The `filter` column was added on 2026-08-04: **IN** = survives as a port defect
(§10.1), **OUT** = dismissed, with the reason in §10.2. The `fixed` column was
added on the 2026-08-04 second pass.

| # | what | severity | prototype | toolkit | tier | filter | fixed |
|---|---|---|---|---|---|---|---|
| **D12** | the ±1 time-slice fallback steps **ticks**, never resolves — dead code | **behaviour-changing** | `PR3DCluster_steiner.h:297-347` | `Facade_Cluster.cxx:3302` | 1 | **IN** | **6ea51a3b**, SBND ON |
| **D3** | steiner same-blob edges grouped by retiled blob, and the null-mcell exclusion is absent | **behaviour-changing** | `PR3DCluster_steiner.h:542-563` + `PR3DCluster_graph.h:90-91` | `SteinerGrapher.cxx:854-861` | 1 | **OUT** — see §10.2.9 | n/a |
| **D1** | terminal filter lost the ±1 wire tolerance (one function, two prototype tolerances) | **behaviour-changing** | `PR3DCluster_steiner.h:285-290, 310-315, 336-341` | `Facade_Cluster.cxx:3344-3354` | 1 | **IN** | **6ea51a3b**, SBND ON |
| **D2** | `disable_dead_mix_cell` dropped on the edge-weight charge path | **behaviour-changing** (reach unverified) | `PR3DCluster_steiner.h:514, 521` | `SteinerGrapher.cxx:92` (pre-fix) / `SteinerGrapher.h:392` | 1 / 2 | **IN** | **§12**, SBND ON |
| **D7** | path-skeleton sampled finer (`+1` vs floor) | behaviour-changing at the threshold | `PR3DCluster_steiner.h:214-226` | `DynamicPointCloud.cxx:744` | 1 | **OUT** — see §10.2.8 | n/a |
| **D8** | containment additionally requires same APA and face | forced (multi-APA); reach unverified | no counterpart | `Facade_Cluster.cxx:3262-3266` | 2 | OUT — forced | n/a |
| **D10** | dead plane = per-blob list vs per-wire uncertainty sentinel | forced; reach unverified, **same question as D2** | `PR3DCluster_steiner.h:984` | `Facade_Cluster.cxx:1027-1029` | 2 | OUT — see D2 | n/a |
| **D4** | edge weight `float` → `double` | fidelity — blocks bit-identicality | `PR3DCluster.h:30` | `Graphs.h:22` | 1 (fact) | OUT — better | n/a |
| **D6** | tree-edge dedup by vertex pair, not edge pointer | **toolkit fixes prototype nondeterminism** | `PR3DCluster_steiner.h:505-506` | `SteinerGrapher.cxx:969-997` | 1 | OUT — better | n/a |
| **D5** | `setS` → `vecS` out-edge storage | structural, compensated | `PR3DCluster.h:33` | `Graphs.h:23-29` | 1 | OUT — equal | n/a |
| **D9** | extreme points round-trip index → point → index | benign, latent | `PR3DCluster_steiner.h:395` | `SteinerGrapher.cxx:268` | 2 | OUT — benign | n/a |
| **D11** | `recover_steiner_graph` not ported (the only MST) | gap; consumer impact unassessed | `PR3DCluster_steiner.h:77-180` | none | 1 | OUT — unreachable | n/a |

---

## §9 What is NOT claimed

> **Scope note added 2026-08-04.** §9 describes the **audit** (§0–§8, §10). It
> is *not* superseded by the fixes: §11 and §12 are separate sections with their
> own evidence, their own gates, and their own "what this does not settle"
> (§11.7, §12.7). Concretely, the first bullet's "no event was run" is still
> true of the audit — the runs live in §11.4 and §12.4 and are the only measured
> claims in this file. The third bullet's "no recommendation" also still holds
> of the audit: what changed is that the **owner** made three decisions on top
> of it, each quoted where it was acted on.

* **No measurement.** No event was run for this document. Every statement is a
  source read, and every "this changes the output" is a mechanism argument, not
  an observed diff. D1, D2, D3 and D7 all deserve a single-event before/after
  before anyone acts on them — pr/28 §7.4 and §8.4 show why: the calib-JSON
  noise floor on this chain has been 268 and 356 leaves in two different
  sessions, so a same-binary repeat is mandatory before attributing anything.
* **No ranking against physics.** The order in §8 and §10.1 — **D12** first,
  then D1, D3, D2, D7 — reflects how much of the delivered graph each row moves
  and how certain the mechanism is, **not** any knowledge that one moves a
  vertex, a score, or an efficiency more than another. (Before the 2026-08-04
  revision this bullet ranked D3 first; D12 displaced it because a branch that
  never executes is a larger and more certain defect than one that groups edges
  differently.)
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
* **§10's dismissals are dismissals of *this* stage.** D8 and D10 in particular
  are dismissed because of what they do (or cannot do) *here*; §10.2 says where
  each remains live elsewhere.

---

## §10 Filtered list — bugs and gaps only *(added 2026-08-04)*

**Owner request.** *"For all these listed behavior change, we can skip the ones
that are improvements over the previous prototype … and only focus on the one
that are bugs or missing from the port … Note there are some convention
difference between the toolkit and prototype, for wire index, the toolkit's
convention is `[)`, in which the higher side is not included, and the prototype
convention is `[]`, both end included. This may trick some code change."*

Every D-row was re-read against the live tree at toolkit `23bd6783`. No anchor
had drifted (Repro block). **Five rows survive; seven are dismissed; one new
defect — D12 — was found, and it is the highest-severity row in the document.**

### §10.1 IN — the five port defects

> **Superseded by the 2026-08-04 second pass.** Of these five, **D12 and D1 are
> fixed and SBND-ON** (§11, toolkit `6ea51a3b`), **D2 is fixed and SBND-ON**
> (§12), and **D7 and D3 are dismissed** — §10.2.8 and §10.2.9, both at the
> owner's reading. **Nothing in §10.1 is open.** The section is kept verbatim
> below because the traps it names are the reason the three fixes look the way
> they do, and because a later reader deserves the pre-fix reasoning rather than
> a rewritten one. Read the `fixed` column of §8 for current status.

Ranked by how much of the delivered product each moves. Each carries the trap a
fix has to clear; **none of these is a patch, and no code was changed.**

| rank | # | one line | fix is confined to |
|---|---|---|---|
| 1 | **D12** | the ±1 adjacent-slice fallback counts **ticks**, so it never resolves — the branch is dead on every point of every event | `Facade_Cluster.cxx:3302` |
| 2 | **D1** | the terminal filter lost the prototype's ±1 **wire** slack on all six bounds | `Facade_Cluster.cxx:3344-3354`, but **not** in place — see below |
| 3 | **D3** | the steiner-graph same-blob pass groups by the **retiled** blob, and drops the prototype's "no original mcell ⇒ no edges" exclusion | `SteinerGrapher.cxx:854-861` |
| 4 | **D2** | `disable_dead_mix_cell` is not forwarded, so the charge branch flips from the prototype's `false` to the default `true` | `SteinerGrapher.cxx:92` |
| 5 | **D7** | the path skeleton is resampled **finer** than the prototype (`int(dis/step)+1` vs `int(dis/step)`) | `DynamicPointCloud.cxx:744`, **shared** — see below |

**D12 and D1 are one filter, two independent tightenings.** Both live in
`is_point_spatially_related_to_time_blobs`, which is reached with
`flag_nearby_timeslice = true` from exactly one caller
(`SteinerGrapher.cxx:291` → `filter_by_reference_cluster`). A terminal must now
clear a wire test that is one wire tighter on each of six bounds, and it gets
one chance instead of three to clear it. Both drop terminals; neither can drop
fewer. **They are listed separately because either can be fixed alone**, and
because they will need separate before/after counts to attribute anything.

**The trap the owner named, and where it actually bites.** The half-open
convention is *already correctly applied* in `check_wire_ranges_match` —
`>= u_min && < u_max` is the right translation of the prototype's
`>= low && <= high`, because `u_wire_index_max()` is exclusive (proof:
`Facade_Blob.cxx:156` writes `b.u_wire_index_max()-1` to recover the last
wire). **That is exactly why D1 is easy to mis-fix.** Restoring the prototype's
±1 is *not* `<= u_max + 1`:

```
prototype (inclusive high)   index <= high + 1     with  high = u_max - 1
=> toolkit (exclusive max)   index <  u_max + 1        NOT  index <= u_max + 1
   and the low side          index >= u_min - 1        (unchanged in form)
```

A literal transcription of the `+1` would be **two** wires loose on the high
side and one on the low — an asymmetry no one would notice in a diff. And a
second trap on top: `check_wire_ranges_match` has **two** live callers with
**different** tolerances — the terminal filter wants ±1, `get_extreme_wcps`
(`Facade_Cluster.cxx:3106`, `flag_nearby_timeslice = false`) wants **none**, and
matches the prototype today (`PR3DCluster_path.h:111-119`). Editing the shared
function in place would silently change `get_extreme_wcps` too. **D3 would want
a third caller, also with no tolerance** (`PR3DCluster_steiner.h:553-558` uses
the strict bounds). So the shape that satisfies all three is a tolerance
parameter defaulting to 0 — stated here as the constraint a fix must meet, not
as a proposal.

**D12's fix has the same shape of trap, on the other axis.** The stride is not
a literal 4: it is per-apa/face, and the toolkit already has the accessor —
`m_grouping->get_nticks_per_slice().at(apa).at(face)`, used at
`improvecluster_1.cxx:295`. Hard-coding 4 would work on SBND and break
elsewhere.

**D7's fix is not local.** `make_points_cluster_skeleton` has **four** callers:
`SteinerGrapher.cxx:204` (this one) and `clustering_deghost.cxx:274`, `:785`,
`:796`. **They resolve to the same body** — checked, because the call sites differ
in arity (6 args here, 5 there) and two bodies would have made this a non-issue:
`DynamicPointCloud.h:196-201` declares the function **once**, with
`flag_wrap = false` and `step = 0.6*units::cm` defaulted, and
`DynamicPointCloud.cxx:679` is its only definition. So the density formula at
`:744` **is** shared with deghosting, and changing `+1` in place would change the
deghost skeleton too — a stage this document did not audit. A fix must be a
parameter or a Steiner-local copy.

**D2 is the one row where the fix is a single argument** — pass the
`disable_dead_mix_cell` that `create_steiner_tree` already holds down to
`create_enhanced_steiner_graph` (`SteinerGrapher.cxx:92`), instead of letting
`SteinerGrapher.h:341`'s `= true` default win. Note it also **removes D10 from
this path** (§10.2.5).

### §10.2 OUT — the seven dismissed, and what dismisses each

#### §10.2.1 D6 — dedup by vertex pair, not edge pointer → **improvement, keep**

The prototype sorts edge descriptors, whose `operator<` compares the
edge-property **pointer** (`PR3DCluster_steiner.h:505-506`), so its insertion
order into the steiner graph is heap-address dependent. The toolkit sorts on
`vertex_pair` (`SteinerGrapher.cxx:969-997`) and says why in a comment. This is
the toolkit being **reproducible where the prototype is not** — precisely the
category the owner asked to skip. Reverting it would reintroduce an M4-class
nondeterminism.

#### §10.2.2 D4 — `float` → `double` edge weights → **improvement, keep**

`double` is strictly more precise than the prototype's `float`
(`PR3DCluster.h:30` ↔ `Graphs.h:22`). Its only cost is that the two trees can
never be **bit**-identical — which matters for *validating* a port, not for the
physics. Dismissed as an item to fix; **retained as a standing caveat**: no
future D1/D2/D3/D7/D12 comparison should be scored on exact equality, only on
counts and distributions.

#### §10.2.3 D5 — `setS` → `vecS` out-edge storage → **verified equivalent**

`setS` gave the prototype duplicate-edge rejection for free; the toolkit
restores it with an explicit `boost::edge(a,b,g).second` test before **every**
`add_edge` in the family — `SteinerGrapher.cxx:657`, `:682`, `:889`, `:1072`,
four sites, all four checked. Resulting edge *sets* agree. Not a defect.

#### §10.2.4 D11 — `recover_steiner_graph` not ported → **not reachable in the ported chain**

This row was left open in the original audit pending a consumer sweep. **The
sweep was done and it closes the row.** A complete grep of
`prototype_base/pid/apps/` for live (non-commented) calls returns four hits, in
two files:

* `wire-cell-graph.cxx:475, :479`
* `wire-cell-tracking.cxx:635, :639`

and it is **commented out** in every STM app (`wire-cell-prod-stm.cxx:833`,
`-port.cxx:962`, `wire-cell-stm.cxx:704`). The production apps this port
targets — `wire-cell-prod-nue.cxx` and `wire-cell-prod-nue-port.cxx`, the latter
being the one the SBND runners invoke (doc pr/36 §2.4) — **never call it.** So
the toolkit is not missing a step the ported chain performs. `SteinerGrapher.h`'s
"reserved for `recover_steiner_graph()` if that function is ported" comment is
accurate and should stay. Dismissed as a gap; it is a correctly-scoped omission.

#### §10.2.5 D10 — dead plane: per-blob list vs per-point sentinel → **not an independent item here**

The divergence is real (`mcell->get_bad_planes()` ↔
`charge_uncertainty(pt, plane) > 1e10`) and forced — the toolkit has no
per-blob bad-plane list. But it lives **only** in `calc_charge_wcp`'s
`disable_dead_mix_cell == true` branch, and **in the Steiner-build path the
prototype never takes that branch**: `create_steiner_graph` passes `false` at
both `PR3DCluster_steiner.h:27` and `:46`, and the toolkit passes `false` at
`CreateSteinerGraph.cxx:210` and `:234`. The only way the `true` branch is
reached here is **D2's dropped argument**. Fix D2 and D10 stops being reachable
in this family — so it is not a separate thing to fix, it is D2's shadow.

**Where D10 *is* live, stated so the dismissal is not read too broadly:** the
ImproveCluster path genuinely uses `true` on both sides — prototype
`ImprovePR3DCluster.cxx:19` takes the header default
(`PR3DCluster.h:112`, `disable_dead_mix_cell = true`) and toolkit
`improvecluster_2.cxx:107` passes `true` explicitly. That call is **faithful**,
and the per-blob-vs-per-point semantic difference is real there. That path is
out of scope for this document (§0) and cannot be closed without a data-model
change.

#### §10.2.6 D8 — containment also requires same APA and face → **forced, and correct**

`Facade_Cluster.cxx:3262-3266` returns false if the point's APA or face is
absent from the reference map. The prototype has no such test because uBooNE has
one TPC — `old_time_mcells_map` is keyed by time slice alone. Matching a point
against a reference blob in a *different* drift volume would be wrong physics on
SBND. This is a required adaptation, not a lost check.

#### §10.2.7 D9 — extreme-point index → point → index round trip → **benign; the one risk was checked and is absent**

The stated risk was that `get_closest_point_index` might search a different
scope than `get_extreme_wcps` scanned. **It does not:** `get_extreme_wcps`
iterates `npoints()` and `get_closest_point_index` queries `kd3d()`
(`Facade_Cluster.cxx:1354-1362`), both the default scope of the same cluster,
and both are called on `m_cluster` (`SteinerGrapher.cxx:261`, `:268`). The
extreme point is by construction a cluster point, so the query returns distance
0. The only residual is a tie between two exactly-coincident points, which would
pick a different index of the same location. Not a defect.

#### §10.2.8 D7 — path skeleton resampled finer → **improvement, keep** *(added on the second pass)*

Owner: *"I feel like D7 is not a problem."* Agreed, and the arithmetic says why.
Both sides resample the shortest path at the same 0.6 cm target
(`SteinerGrapher.cxx:202`, prototype `:206`) and differ only in how they round:

| | count | spacing |
|---|---|---|
| prototype `:214-226` | `num_steps = floor(dis/step)` | `dis / floor(dis/step)` — **≥** 0.6 cm |
| toolkit `DynamicPointCloud.cxx:744` | `num_points = floor(dis/step) + 1` | `dis / (floor(dis/step)+1)` — **<** 0.6 cm |

The prototype *undershoots* its own target: asking for 0.6 cm spacing and
delivering up to 1.2 cm on a segment just short of two steps. The toolkit never
exceeds it. Since this cloud exists only to answer "how far is this terminal
from the path", a denser sample is a strictly more accurate answer to that
question — the reference curve is the same curve. **Nothing was lost in the
port; the port rounds the better way.**

The doubled endpoint (prototype adds it once in the loop and again at `:226`) is
inert: a KD-tree returns the same nearest distance whether a point appears once
or twice.

What remains true, and stays as a caveat rather than a defect: the filter is
`close_in_2d && dis_3d > 6 cm` with a 1.8 cm 2-D threshold
(`SteinerGrapher.cxx:213-237`; **the thresholds themselves match the prototype
exactly**), and finer sampling pushes *both* sides of that conjunction down. So
a terminal sitting on either threshold can flip either way. That is the ordinary
consequence of a more accurate distance, not evidence of a lost behaviour — and
it is the same caveat that would attach to any resolution change.

**This also retires the "fix is not local" problem.** §10.1 recorded that
`make_points_cluster_skeleton` is shared with `clustering_deghost.cxx:274`,
`:785`, `:796` through a single definition (`DynamicPointCloud.h:196-201`
declares it once with defaults, `DynamicPointCloud.cxx:679` is the only body),
so any change to `:744` would have silently altered the deghost skeleton too.
Dismissing D7 means that shared body is left alone — the safest outcome
available, and it happens to be the correct one.

#### §10.2.9 D3 — same-blob grouping key → **the toolkit's choice is the prototype's own, one step earlier** *(added on the second pass)*

Owner: *"is D3 a problem? I assume toolkit is better right?"* The evidence says
yes, on both halves of the row, and the strongest argument is not a judgement
call — it is what the prototype does at the *previous* call site.

**(a) The grouping key.** The row's complaint was that the toolkit groups
selected points by the blob they occupy in the **retiled** cluster
(`majs[old_index]`, `SteinerGrapher.cxx:933-939`, on `m_cluster` — the retiled
cluster, `CreateSteinerGraph.cxx:231`) while the prototype's flag=2 pass groups
by an **original-cluster mcell** resolved by strict wire containment
(`PR3DCluster_steiner.h:542-563`). Both statements are correct. What the
original row missed is that **the prototype itself groups by the retiled blob at
flag=1**:

```cpp
// PR3DCluster_steiner.h:27   -- called ON new_cluster, the retiled cluster
new_cluster->establish_same_mcell_steiner_edges(gds, false);
// PR3DCluster_graph.h:34,40  -- flag==1 branch
WCP::WCPointCloud<double>& cloud = point_cloud->get_cloud();   // the RETILED cloud
map_mcell_all_indices[cloud.pts.at(i).mcell]                   // the RETILED mcell
```

and §3 step 3 already scores that call a **MATCH**. So retiled-blob grouping is
not a toolkit invention the prototype would reject; it is the prototype's own
choice wherever it has a live retiled cluster to ask.

**(b) Why flag=2 is different, and it is not a physics reason.**
`point_cloud_steiner` is a persistent member of the **original** cluster, while
the retiled cluster and the holder that owns its mcells are destroyed
immediately after the tree is built:

```cpp
// PR3DCluster_steiner.h:53-56
new_cluster->Del_graph();  new_cluster->Del_point_cloud();
delete new_cluster;        delete temp_holder;
```

Storing retiled `SlimMergeGeomCell*` on a cloud that outlives them would dangle.
The prototype's `WCPoint` has exactly one `mcell` field, so the original-mcell
resolution at `:542-563` is what that field *has to* hold — a lifetime
constraint, not a claim about which partition is right. The toolkit's
`steiner_pc` is a `Dataset` of arrays with **no blob pointer at all**, so the
constraint does not exist and the grouping is computed while the retiled cluster
is alive. **Nothing was dropped; the thing that forced the prototype's hand is
absent.**

**(c) The missing exclusion is the same decision, not a second one.**
`PR3DCluster_graph.h:90-91` skips every point whose `mcell` is null. Under the
toolkit's key there is no null case — every selected point occupies exactly one
retiled blob by construction. The exclusion was never a rule about which points
deserve edges; it was the failure branch of a lookup that no longer happens.

**(d) The one consumer that does read the association, and how the toolkit
replaces it.** Sweeping the prototype for other readers of the *steiner* cloud's
`mcell` turns up exactly one outside flag=2:

```cpp
// PR3DCluster_path.h:356-379   get_two_boundary_wcps(flag=2) -> point_cloud_steiner
if (cloud.pts[i].mcell == 0) continue;
if (cloud.pts[i].mcell->Estimate_total_charge() < 1500) continue;
```

called from `NeutrinoID.cxx:1152`, `:1170`, `NeutrinoID_proto_vertex.h:426`,
`Cosmic_tagger.h:745,852,2999,3106`, `ToyFiducial.cxx:433,541`. The toolkit's
`get_two_boundary_steiner_graph_idx` (`Facade_Cluster.cxx:3393-3460`) replaces
both cuts deliberately: it scores boundaries on the **regular** PC, where blob
charge and dead-wire counts are available, then snaps each to the nearest
Steiner **terminal** — and its own comment names the substitution
(`:3430-3435`, *"Terminals are the original cluster data points (mcell != null
in the prototype)"*). So this is a known, documented structural substitution,
**not** an unexamined gap. It is recorded here rather than opened as a row
because it belongs to the vertex/boundary stage, which this document does not
audit (§0); doc pr/30 is its home.

**(e) The coupling the owner should see, and it points the other way.** In the
prototype the terminal filter admits with **±1 wire slack and a t±1 slice
fallback** (`:283-347`) while the flag=2 resolution is **strict and same-slice**
(`:551-558`). A terminal admitted *only* by the slack or the adjacent slice
therefore resolves to `mcell = 0` and is orphaned from same-blob edges. The
prototype admits loosely and then orphans; the toolkit — now that D1 and D12 are
ON — admits loosely and grafts. **The two fixes shipped in §11 widen this axis
rather than narrow it.** That is not a reason to revisit them: each is the right
call on its own filter, and (b) shows the orphaning is an artifact. But it means
D3's dismissal is a dismissal *with* a stated interaction, and it is the reason
the population run in §11.7 matters more than a single-event check.

**What is still not claimed.** The two partitions genuinely differ wherever the
retile subdivides an original blob (toolkit groups *fewer* points together) or
merges/extends one into a dead region (toolkit groups *more*). Nobody measured
which dominates on SBND, and this dismissal does not assert that the edge sets
are close — only that the toolkit's rule is principled, is the prototype's own
rule where the prototype is free to use it, and loses nothing that was
deliberate.

### §10.3 What this revision does and does not claim

> **Written before any code changed, and kept that way.** Its "still no
> measurement / no recommendation / no patch" bullets describe the §10
> pass. Three of the five §10.1 rows have since been acted on — D1 and D12
> in §11, D2 in §12 — each with a measurement and a gate; the other two
> were dismissed in §10.2.8/§10.2.9. `porting_dictionary.md:266` is still
> an empty `⚠ xxx`, so the CLAUDE.md §5 rule 4 point below still stands
> for the rows nobody has touched.

* **Still no measurement.** §9 stands unchanged. D12 is proven at source level
  four ways, but *how many terminals it costs* is unmeasured, as are D1's,
  D2's, D3's and D7's reaches. Any fix needs a before/after on the same binary
  first — pr/28 §7.4's 268-vs-356-leaf noise floor is why.
* **The dismissals are evidence-backed, not judgement calls** — §10.2.4 rests on
  a complete app-directory grep, §10.2.7 on reading both scope accessors,
  §10.2.3 on all four guard sites. Where a dismissal is *scoped* rather than
  absolute (D10, D8), the scope is stated.
* **Still no recommendation and no patch.** `porting_dictionary.md:266` is still
  an empty `⚠ xxx`, so CLAUDE.md §5 rule 4 still applies to all twelve rows.
  §10.1's "fix is confined to" column names *where* a change would go and *which
  trap it must clear* — deliberately not *what to write*.
* **D12 is a correction to this document, not only an addition.** D1's original
  claim that the time-slice half "is ported" was wrong and is struck through in
  place. Anyone who read the earlier version took away that the toolkit lost one
  of the prototype's two slacks; it lost both.

---

## §11 D1 and D12 FIXED — SBND DEFAULT ON *(2026-08-04)*

**Owner request.** *"Can we fix these … first? I assume this is D1 and D12.
Please make the code more robust … please check on the event served in port
5017."*

**Status: SHIPPED. SBND PRODUCTION DEFAULT ON (owner flip, 2026-08-04).**

They shipped default-OFF first, because turning either on changes production
output unconditionally and that is CLAUDE.md §5 rule 1 — an owner decision, not
mine. The owner made it the same day, on the reasoning that these are **port
bugs and not tunable behaviour**: *"what do you mean by knob off, these two are
improvements, or bug fix, for SBND, right? They should be on."* That is correct
— the toolkit filter was strictly tighter than the prototype in two independent
ways, with nothing on the other side of the trade.

Mechanically the knobs remain, and that is deliberate: **the C++ defaults stay
OFF** so uBooNE / ICARUS / PDHD / PDVD are untouched, and the ON state lives in
the SBND operating point (`wct-pr-perevt.jsonnet`, doc 68 — a bare run *is*
production). Setting both back to 0/false reproduces the pre-fix chain
byte-for-byte at the config level (§11.5), which is what every legacy comparison
needs. D2, D3 and D7 were untouched by *this* commit; the second pass fixed
D2 (§12) and dismissed D3 and D7 (§10.2.9, §10.2.8).

### §11.1 What shipped

| layer | change |
|---|---|
| `clus/inc/WireCellClus/Facade_Cluster.h`, `clus/src/Facade_Cluster.cxx` | `check_wire_ranges_match(..., int wire_tol = 0)` and `is_point_spatially_related_to_time_blobs(..., int wire_tol = 0, int slice_stride = 1)`. Both defaults ARE the previous literals. |
| `clus/src/SteinerGrapher.h`, `.cxx` | `Config::terminal_wire_tol{0}`, `Config::terminal_adjacent_slice{false}`; new `nticks_per_slice_or_1(apa, face)`; `filter_by_reference_cluster` resolves the stride per (apa, face) through a memo. |
| `clus/src/CreateSteinerGraph.cxx` | reads `terminal_wire_tol` / `terminal_adjacent_slice`, round-trips both in `default_configuration()`. |
| `cfg/pgrapher/common/clus.jsonnet` | `cm.steiner()` gains both args, emitted with the key-suppression idiom. |
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | threaded through `clus_pr()` **and** `pr()`, applied to `steiner` **and** `steiner_refresh`. |
| `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` | **the SBND operating point: `steiner_terminal_wire_tol=1`, `steiner_terminal_adjacent_slice=true`.** |
| `sbnd_xin/run_pr_chain_batch.sh` | env `SBND_STEINER_WIRE_TOL` / `SBND_STEINER_ADJ_SLICE`; unset ⇒ no TLA ⇒ the cfg default ⇒ **both on**. Set both to `0` for the legacy arm. |
| `clus/test/doctest_steiner_terminal_filter.cxx` | **new**, 7 cases / 18 assertions. |

**Both fixes are confined to the Steiner terminal filter.** They travel as
arguments, not as edits to the shared helper's behaviour, so `get_extreme_wcps`
— the other caller of the same C++ function, which the prototype gives **no**
slack and **no** slice fallback — keeps taking the defaults and is untouched.
That was the whole reason §10.1 said an in-place edit was the wrong shape.

### §11.2 The robustness the owner asked for

The traps are now encoded where the next reader will hit them, not only in this
document:

* **The half-open arithmetic is derived in a comment at the point of use**
  (`Facade_Cluster.cxx`, `check_wire_ranges_match`): `index <= high + 1` with
  `high = max - 1` gives `index < max + 1`, so the implementation adds
  `wire_tol` to `u_max` and subtracts it from `u_min` and keeps the `<`. The
  comment states in words that `<= max + wire_tol` would be loose on the high
  side only. **A doctest fails if anyone writes it that way** (§11.3).
* **The stride is never a literal.** `nticks_per_slice_or_1()` reads
  `Grouping::get_nticks_per_slice().at(apa).at(face)`, the same accessor
  `improvecluster_1.cxx:295` uses. Hard-coding 4 would work on SBND and be wrong
  elsewhere — §10.1 named that risk and the code now cannot take it.
* **Three guards, because this runs per terminal in a hot loop:** an unknown
  apa or face returns 1 with a WARN instead of letting `std::map::at` throw out
  of the filter; a non-positive span returns 1 rather than making the `±offset`
  loop test the point's own slice twice; a negative `terminal_wire_tol` is
  clamped to 0 in `configure()` with a warning, since a negative tolerance would
  silently *shrink* the band.
* **The terminal-count log line now echoes both knob values**
  (`SteinerGrapher.cxx`, `create_steiner_tree`), so a log alone says which arm
  produced a number. That is what made §11.4's measurement possible.

### §11.3 Tests

`clus/test/doctest_steiner_terminal_filter.cxx` — 7 cases, 18 assertions, all
passing. They pin the boundary, not the feature:

* **the `<= max + tol` discriminator.** With `tol=1` against a reference blob
  covering wires 10–11, wire 12 is accepted and wire **13 is not**. The wrong
  implementation accepts 13 and the case fails.
* **symmetry.** The accepted set with `tol=1` is asserted to be *exactly*
  `{9, 10, 11, 12}`. A band loose on one side only passes both single-wire cases
  above and fails this one.
* **D12 both ways.** A reference blob one slice away (tick 4, span 4) is **not**
  found with the legacy step of 1 — pinning that the historical branch is dead —
  and **is** found with a step of 4.
* **the fallback stays one slice wide.** A blob two slices away is still not
  found, so the fix widens the search by one slice, not unboundedly.
* **composition.** A terminal that is one wire out *and* one slice away needs
  both knobs; either alone leaves it rejected.

`./build/clus/wcdoctest-clus`: **85 cases / 935 assertions, 0 failed.**

### §11.4 Evidence on the event served on port **5017** (SBND run 18255 evt 388)

Port 5017 is serving `work-tfix388-final/pr_evt388/calib-pr-evt388.json`, i.e.
this event. Two fresh arms were run from the same Q/L root
(`work-nuecc48-prod0803`) — new tags, nothing existing written (M13):

```bash
cd sbnd_xin
PR_JOBS=1 PR_EXTRA_STAGES=pr_display SBND_WCT_LOGLEVEL=trace \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr29-388-off  sim 388
PR_JOBS=1 PR_EXTRA_STAGES=pr_display SBND_WCT_LOGLEVEL=trace \
  SBND_STEINER_WIRE_TOL=1 SBND_STEINER_ADJ_SLICE=1 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr29-388-on   sim 388
```

**The filter itself, summed over every call in the event:**

| arm | calls | terminals in | kept | dropped | clusters left with <2 terminals |
|---|---|---|---|---|---|
| OFF (legacy) | 44 | 2901 | 1517 | **1384 (47.7%)** | **24** |
| ON (`wire_tol=1`, `adjacent_slice=true`) | 42 | 2900 | 2320 | 580 (20.0%) | 3 |

**The legacy filter was discarding nearly half of every cluster's Steiner
terminals on this event, and leaving 24 clusters below the two-terminal minimum
— i.e. with no Steiner tree at all.** That is the size of D1+D12 together.

**Downstream, same event:**

| quantity | OFF | ON |
|---|---|---|
| PR segments | 75 | **88** |
| PR vertices | 118 | **131** |
| steiner points (dumped) | 35 | 38 |
| showers | 13 | 12 |
| `nue_score` | 4.3009 | 4.3009 (unchanged) |
| `numu_score` | −3.1994 | **−2.1661** |
| `kine_reco_Enu` | 2900.5 MeV | 2865.6 MeV |
| `kine_pio_flag` | **1** (mass 132.1 MeV) | **0** |
| reconstructed particles | 13 | 11 |
| event label / TGM / STM / FC | nu-candidate, 0/0/0 | **identical** |

20 of 44 dumped tagger fields move. **The selection verdict does not**: the
`nusel-table.tsv` files are identical line for line, both arms.

**Reported, not tuned (§5 rule 7): the pi0 candidate disappears when the knobs
are on** — `kine_pio_flag` 1 → 0, and a 132.1 MeV pi0 mass with it. On a
knob that can only *add* terminals this is not self-evidently an improvement,
and it is exactly the kind of number this document is not authorised to chase.
It is one event.

### §11.5 Gates

* **The legacy arm is still reachable and still byte-identical.** Compiling the
  SBND PR job three ways — from a `git archive HEAD cfg` tree (pre-change), from
  the working tree with `-A steiner_terminal_wire_tol=0 -A
  steiner_terminal_adjacent_slice=false`, and from the working tree bare:

  | config | md5 |
  |---|---|
  | pre-change (HEAD) | `d90a93301f724b4592c082d9636a674a` |
  | post-change, knobs forced off | `d90a93301f724b4592c082d9636a674a` |
  | post-change, **bare = production** | `c9604b6c1d9aa1c4676ca7571d1c1dfb` |

  So the flip did not strand the old chain: forcing both to 0/false gives the
  pre-fix config **byte for byte**, which is what any pre-flip A/B needs.
* **Compiled-config proof (M6).** `terminal_wire_tol: 1` and
  `terminal_adjacent_slice: true` appear on **both** `CreateSteinerGraph` inodes
  (`pr` and `prrefresh`) in the bare/production config, and on neither in the
  forced-off one — 4 occurrences vs 0. Not a wrapper-level merge that vanishes.
* **The bare compiled config is identical to the arm-ON config** measured in
  §11.4, so the numbers there are production numbers, not a special arm.
* **Freshness proof (M1).** `local/lib/libWireCellClus.so` 10:49 > last source
  edit 10:48, and `nm` finds `nticks_per_slice_or_1` in the **installed** lib.
* **Unit tests.** `./build/clus/wcdoctest-clus` 85/85.
* **Determinism.** The knob-OFF arm was run **twice**
  (`work-pr29-388-off`, `work-pr29-388-off2`): the two `calib-pr-evt388.json`
  are **identical as JSON**, 0 of 44 tagger fields differing, same main vertex
  to the last digit, same 2901/1517 terminal totals. **The noise floor on this
  event is zero**, so every number in §11.4 is the knobs and nothing else.
* **Reproduced on a second binary, after the flip.** The concurrent session
  landed `22249ff4` and `026a7501` mid-session, so both arms were re-run
  end-to-end through the runner on the rebuilt tree — `work-pr29-388-b2prod`
  (bare = production) and `work-pr29-388-b2legacy`
  (`SBND_STEINER_WIRE_TOL=0 SBND_STEINER_ADJ_SLICE=0`). **Every figure in §11.4
  reproduces exactly**: 2901→1517 vs 2900→2320 terminals, 24 vs 3 starved
  clusters, 75→88 segments, 118→131 vertices, `numu_score` −3.1994 → −2.1661,
  `kine_reco_Enu` 2900.5 → 2865.6, pi0 flag 1 → 0, `nusel-table.tsv` identical.
  That run also proves the runner path end-to-end: with no env set, the driver
  now produces the fixed chain.

### §11.6 One gate that could NOT be run, stated rather than glossed

**A knob-OFF runtime A/B against the stored `work-tfix388-final` baseline is
not attributable and was not claimed as a PASS.** My knob-OFF arm already
differs from that baseline — 118 vs 119 vertices, 35 vs 31 dumped steiner
points, `numu_score` −3.199 vs −2.835, main vertex moved 0.6 mm.

That difference is **not** these knobs and **not** noise:

* knobs off ⇒ the compiled config is byte-identical (md5 above) and every new
  parameter defaults to the literal the code used before, so no instruction on
  the knob-off path changed;
* the same-binary repeat (§11.5) is byte-identical, so the chain is not drifting
  run to run on this event.

It is the binary. `work-tfix388-final` was produced at toolkit `23bd6783`; the
runs above load a `local/lib` built from **`22249ff4`** (*"make the fitted-2D-charge
merge and the back-to-back retype loop deterministic"*, doc pr/28 §14, run on
this very event) and **`026a7501`** (*"order the shower edge walk by graph
index — the PR display dump is now run-to-run identical"*, doc pr/28 §15) — two
commits landed by a concurrent session in this shared tree while this work was
in progress. Both are downstream of the Steiner stage and both, by their own
titles, change the PR display dump. That fully accounts for the baseline
difference and none of it is attributable here.

Attributing a runtime difference to *this* change would need a tree holding only
this change, which this working copy is not. **The valid measurement is arm-OFF
vs arm-ON in §11.4** — same binary, same minute, one variable — and the
same-binary repeat proves the comparison is not sitting on noise.

### §11.7 What this does not settle

* **One event.** 47.7% → 20.0% terminal loss is this event's number. A
  population run is what would justify a default flip, and none was made.
* **The two knobs were measured together, not separately.** §11.4 cannot say how
  much of the segment 75 → 88 is D1 and how much is D12. Three more arms would;
  they were not run.
* **Direction of "better" rests on fidelity, not on a measurement.** The case
  for ON is that these are port bugs: the prototype keeps these terminals, the
  toolkit dropped them through a units error and a missing tolerance, and there
  is nothing on the other side of the trade. That is the owner's reasoning and
  it is sound. What has **not** been shown is that SBND *reconstruction* is
  better — no efficiency, purity or vertex-resolution number moved in either
  direction here, because none was measured. **The pi0 loss on evt 388 is the
  one concrete counter-signal on record** (`kine_pio_flag` 1 → 0, a 132.1 MeV
  candidate) and it should be looked at on a population before anyone treats
  the flip as validated rather than merely correct.
* **The obvious next step is a population run.** Both arms are reachable from
  one runner (`SBND_STEINER_WIRE_TOL=0 SBND_STEINER_ADJ_SLICE=0` for legacy),
  and the 572-event valfast manifest is the natural target: how many events move
  label, how many gain or lose a pi0, and how the 24-starved-clusters number
  scales. None of that was run for this document.
* **D2, D3 and D7 remain open** exactly as §10.1 leaves them. In particular D2's
  one-argument fix was deliberately **not** bundled in: separate defect,
  separate knob, separate gate. *(Second pass: D2 is now fixed — §12 — and D3
  and D7 are dismissed, §10.2.9 and §10.2.8. The separate gate promised here is
  §12.5.)*

---

## §12 D2 FIXED — SBND DEFAULT ON *(2026-08-04, second pass)*

**Owner instruction.** *"D2 should be fixed"*, and on the knob: *"for this one,
it should also be default on. Since we are doing validation and improvements."*

### §12.1 The defect, restated in one paragraph

`create_steiner_tree` receives `disable_dead_mix_cell = false`
(`CreateSteinerGraph.cxx:262`) and correctly forwards it to
`find_steiner_terminals`. It then called `create_enhanced_steiner_graph`
**without the argument**, so that function's `= true` default
(`SteinerGrapher.h:392`) won and the `Qs`/`Qt` entering every steiner edge
weight were computed on the other branch of `calc_charge_wcp`. The prototype
computes them with the same `false` it was called with
(`PR3DCluster_steiner.h:514`, `:521`, inheriting `Create_steiner_tree`'s
parameter, passed `false` at `create_steiner_graph:46`). A dropped argument, not
a decision — nothing in the toolkit ever chose `true` here.

### §12.2 What shipped

| where | what |
|---|---|
| `clus/src/SteinerGrapher.h` | `Config::edge_charge_forward_dead_mix{false}` |
| `clus/src/SteinerGrapher.cxx` | `const bool edge_dead_mix = m_config.edge_charge_forward_dead_mix ? disable_dead_mix_cell : true;` passed to `create_enhanced_steiner_graph`; the value echoed in the entry TRACE line |
| `clus/src/CreateSteinerGraph.cxx` | `configure()` reads the key; `default_configuration()` round-trips it |
| `cfg/pgrapher/common/clus.jsonnet` | `cm.steiner(..., edge_charge_forward_dead_mix=false)` with the key-suppression idiom |
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | declared and forwarded in **both** `clus_pr()` and `pr()`, applied to `steiner` **and** `steiner_refresh` |
| `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` | **the SBND operating point: `steiner_edge_charge_forward_dead_mix = true`** |
| `sbnd_xin/run_pr_chain_batch.sh` | `SBND_STEINER_EDGE_DEAD_MIX=0` for the pre-fix arm; empty = ON |

**The C++ default stays `false`.** Turning it on unconditionally would move
uBooNE, ICARUS, PDHD and PDVD as well, which is CLAUDE.md §5 rule 1. The SBND
operating point lives in cfg (doc 68: a bare run **is** production), so on SBND
a bare run now uses the prototype's charge and every other detector is
byte-identical.

### §12.3 Why it is a knob at all, given it is a bug fix

Same reasoning as §11.1, plus one difference worth stating: **unlike D1 and D12,
this one moves weights in *either* direction.** Those two could only ever *add*
terminals — each restored a way to PASS a filter. D2 changes a charge that goes
into a denominator, so an edge can get cheaper or dearer and the tree can gain
or lose edges. That makes a reachable legacy arm more valuable here, not less.

### §12.4 Evidence on SBND evt 388 (the event served on port 5017)

Matched arms on one binary, `work-pr29-388-d2off` (`SBND_STEINER_EDGE_DEAD_MIX=0`)
and `work-pr29-388-d2on` (bare = production). **D1 and D12 are ON in both**, so
every number below is D2 and nothing else.

| | D2 off | D2 on |
|---|---|---|
| PR segments | 85 | **88** |
| PR vertices | 129 | **133** |
| showers / steiner blocks / proj panels | 12 / 33 / 6 | 12 / 33 / 6 |
| `numu_score` | −0.9050 | **−0.7282** |
| `nue_score` | unchanged | unchanged |
| `kine_reco_Enu` | 2816.1 MeV | 2811.1 MeV |
| main vertex | (−163.176, 31.469, 426.429) | (−163.100, 31.575, 426.320) |
| `nusel-table.tsv` | identical | identical |

16 of 44 tagger fields move, all of them BDT sub-scores; the event label does
not move and no pi0 flag changes. The main vertex shifts by **1.6 mm**.

**Noise floor: zero.** The OFF arm was run twice
(`work-pr29-388-d2off`, `work-pr29-388-d2off2`) and the two
`calib-pr-evt388.json` are **md5-identical**
(`190221b8bf42d23e0ad92a876942bdc0`). So the table above is the knob, not drift.

### §12.5 Gates

* **Legacy arm byte-identical.** Compiling the SBND PR job from a
  `git archive HEAD cfg` tree (pre-D2, D1/D12 already ON) and from the working
  tree with `-A steiner_edge_charge_forward_dead_mix=false`:

  | config | md5 |
  |---|---|
  | pre-D2 (HEAD), bare | `d1e49000c3b148d15d99d1beab2eae47` |
  | post-D2, D2 forced off | `d1e49000c3b148d15d99d1beab2eae47` |
  | pre-D2 (HEAD), all pr/29 knobs off | `9a52a3f4c69a9652f33e448abb98a00e` |
  | post-D2, all three forced off | `9a52a3f4c69a9652f33e448abb98a00e` |
  | post-D2, **bare = production** | `a4e2beeb04682731d446b18f5dad9cd4` |

  Both pre-existing arms survive the change **byte for byte**. (These md5s
  differ from §11.5's because the shared tree moved — see §12.7.)
* **Compiled-config proof (M6).** `edge_charge_forward_dead_mix: true` appears
  on **both** `CreateSteinerGraph` inodes (`pr` and `prrefresh`) in the
  production config alongside `terminal_wire_tol: 1` and
  `terminal_adjacent_slice: true`; 2 occurrences bare, 0 forced-off. Not a
  wrapper-level merge that vanishes (M6).
* **Freshness proof (M1).** `local/lib/libWireCellClus.so` 11:24 > last source
  edit 11:21.
* **Unit tests.** `./build/clus/wcdoctest-clus` **89 cases / 948 assertions**,
  all passing — 4 cases and 12 assertions of that are new
  (`clus/test/doctest_steiner_edge_charge.cxx`).

### §12.6 Tests

The knob itself is a ternary; what needs pinning is the claim underneath it —
that the two branches of `calc_charge_wcp` are **not** interchangeable. They
select planes by independent predicates: `true` sums all three then subtracts
**dead** planes (`charge_uncertainty > 1e10`), `false` sums only planes with a
**nonzero** charge value. `doctest_steiner_edge_charge.cxx` pins both ways that
can disagree and both ways it cannot:

| case | configuration | result |
|---|---|---|
| live plane reading zero | q = (0, 5000, 5000), all live | 4082.5 vs 5000.0 — **and the returned boolean flips**, which is the half the terminal selection reads |
| dead plane reading nonzero | q = (3000, 5000, 7000), U dead | 6082.8 vs 5259.9 |
| all live, all nonzero | q = (3000, 5000, 7000) | **equal** |
| dead plane also reading zero | q = (0, 5000, 7000), U dead | **equal** |

The last two matter as much as the first two: without them the file would read
as "this knob changes every point", which is not what §12.4 shows.

### §12.7 What this does not settle

* **One event, again.** Same limitation as §11.7, and the same remedy: the
  572-event valfast manifest. D2 now has its own env switch
  (`SBND_STEINER_EDGE_DEAD_MIX=0`), so a three-arm population run
  (all-off / D1+D12 / all-three) is one runner invocation each.
* **Reach was never measured, and still has not been.** §4's D2 entry said the
  confirmed claim was the mechanism, not a count of affected points. That is
  still true: nobody has counted how many SBND steiner points have a
  live-but-zero or dead-but-nonzero plane. The evt-388 deltas show the answer is
  not zero; they do not show what it is.
* **A knob-OFF A/B against §11's `work-pr29-388-b2prod` is NOT attributable**,
  for the same reason §11.6 gave. The concurrent session landed `397b1517` and
  carries uncommitted edits in `NeutrinoPatternBase.h` and
  `NeutrinoVertexFinder.cxx`, all of which my rebuild compiled in. `d2off`
  therefore does **not** reproduce `b2prod` byte for byte, and the difference is
  theirs, not D2's. This is why §12.4 uses a matched pair on one binary and a
  same-binary repeat, and why the §12.5 md5s were recomputed from `HEAD`
  rather than quoted from §11.5.
* **D2 removes D10 from this path.** §10.2.5 dismissed D10 (dead-plane
  representation) as "not an independent item here" because it only mattered
  through the branch D2 put the toolkit on. With D2 ON, SBND no longer takes
  that branch when weighting steiner edges, so D10's unverified
  per-blob-list-vs-per-point-sentinel question is now inert **for this stage**.
  It remains live in the ImproveCluster path, which this document does not
  audit.

---

## §13 Everything still open, in one place *(2026-08-04)*

Owner: *"For this md file, anything else that we missed?"* This section exists
so the answer is not scattered across §9, §10.3, §11.7 and §12.7. Nothing here
is new work that was done and hidden; it is work that was **not** done.

### §13.1 The measurement debt — the largest gap in this file

> **PARTLY PAID by §14** *(2026-08-04)* — 48 nueCC events, three arms
> (production / pr/29-off / repeat) at one binary. It answers the `event_label`
> question (0/48 move) and the starved-clusters scaling question (87 → 10
> lines, 23 → 5 events), and it separates pr/29 as a whole from the rest of
> the window. It does **not** separate D1+D12 from D2, does not cover the pi0
> question, and 48 nueCC events are not the 572-event valfast manifest. The
> text below stands as written for everything §14 does not reach.

**Every measured number in this document comes from one event (SBND evt 388).**
Three behaviour changes are now SBND production defaults on the strength of
port fidelity plus a single-event sanity check. The 572-event valfast manifest
is the natural target and all four arms are one env var each:

| arm | env |
|---|---|
| pre-pr/29 legacy | `SBND_STEINER_WIRE_TOL=0 SBND_STEINER_ADJ_SLICE=0 SBND_STEINER_EDGE_DEAD_MIX=0` |
| D1+D12 only | `SBND_STEINER_EDGE_DEAD_MIX=0` |
| D2 only | `SBND_STEINER_WIRE_TOL=0 SBND_STEINER_ADJ_SLICE=0` |
| production | *(bare)* |

What it would answer, none of which is known today: how many events change
`event_label`; how many gain or lose a pi0 (**evt 388 lost one** — §11.7, the
single counter-signal on record); how the 24-starved-clusters number scales;
and whether the D1+D12 and D2 effects add or partly cancel, since the two were
measured against *different* baselines on *different* binaries.

### §13.2 Audit coverage that was never claimed, and is now load-bearing

* **The retile step is unaudited** (§0). §10.2.9 dismisses D3 partly on how the
  retile relates original and retiled blobs. That dismissal is only as strong as
  an unaudited assumption about `RetileCluster` vs `Improve_PR3DCluster_2`.
* **The base-graph builders are unaudited** (§0). `find_graph("ctpc_ref_pid")`
  vs `Create_graph(ct_point_cloud, point_cloud)` is scored a MATCH *in shape*
  only (§3 step 2). Everything in §4 is about inputs to the solver; the biggest
  input of all was not opened.
* **Downstream consumers of `steiner_graph` / `steiner_pc` were never swept**
  (§5.6, D11). §10.2.9(d) opened exactly one of them — `get_two_boundary_wcps(2)`
  — and found a deliberate, in-code-documented substitution. **That one was
  found by looking; nobody has looked at the rest.**
* **D8 and D10 reaches are still unmeasured.** Both are dismissed as *forced*,
  which is a statement about why the toolkit differs, not about how much. D10 is
  now inert for this stage (§12.7) but not elsewhere.

### §13.3 Three source observations — **one fixed, two closed by verification**

Originally logged here as "reported, not fixed". The owner asked for them to be
improved, so each was taken as far as it can go. Only one of the three was ever
a toolkit defect; the other two are **prototype-side**, and `prototype_base/` is
a read-only porting reference (CLAUDE.md §0), so the improvement available there
is to prove the toolkit counterpart is immune and say so — which is what was
done, rather than leaving a vague note.

**1. No bounds guard on `majs[old_index]` — FIXED** (toolkit, this commit).
`establish_same_blob_steiner_edges_steiner_graph` guarded the *result*
(`blob_node_idx >= nodes.size()`) but not the *index*. The contrast that makes
this worth a guard rather than a comment: `form_cell_points_map`
(`SteinerGrapher.cxx:626`) walks `point_idx < skd.npoints()` and **cannot** go
out of range, while this loop's keys are base-graph vertex descriptors carried
through `result.old_to_new_index` — a different, unbounded provenance. They
should all be `sv3d` points by construction, so the guard is expected never to
fire; the reason to add it is that an out-of-range read of `major_indices()`
would be **silent**, corrupting a blob grouping rather than crashing.
`get_blob_for_vertex` (`:855`) already guards its index the same way, so this
makes the file self-consistent. **No behaviour change** — gated in §13.5.

**2. The prototype's `nullptr` mcell bucket at flag=1 — CLOSED, the toolkit is
structurally immune.** `PR3DCluster_graph.h:40` inserts into
`map_mcell_all_indices[cloud.pts.at(i).mcell]` with no null skip, unlike flag=2
which skips at `:88`. Any point with no mcell would be same-blob-connected to
*every other* such point — a spurious clique. Whether WCP ever has such a point
in a cluster's own cloud was not checked and cannot be fixed here.
**The toolkit cannot express the bug:** `form_cell_points_map` keys on a blob
**node index** obtained from `major_indices()`, checks `blob_node_idx >=
nodes.size()`, and then checks the facade is non-null — three independent
reasons there is no null bucket. Recorded as **K2/K5** in the porting dictionary
so nobody ports the null-keyed shape across.

**3. The prototype iterates pointer-keyed maps in both same-blob passes —
CLOSED, recorded as a dictionary rule.** `std::map<SlimMergeGeomCell*,
std::set<int>>` iterated at `PR3DCluster_graph.h:60` and `:99`. The edge *set*
is order-independent, but the insertion order into `same_mcell_steiner_edges`
and into the boost graph is not — the same class as D6's edge-descriptor sort,
which compares an edge-property **pointer**. §6 audits toolkit determinism and
never stated the prototype's; it does now, and the toolkit's `size_t`-keyed
`form_cell_points_map` is written into the dictionary as **K2** with the reason,
so the "simplify it to a `Blob*` key" refactor is pre-emptively answered.

### §13.4 The documentation debt — **PAID**

`clus/docs/porting/porting_dictionary.md:266` was an empty `⚠ xxx` placeholder
and the framing finding of this audit (§0). **It is now filled**: the Steiner
Tree section carries a 15-row WCP→WCT function/type mapping, **ten `K1`–`K10`
known-divergence entries** covering every row of §8, and a table of the three
config knobs with what each restores.

The five decisions this document produced are now written where a porter will
hit them:

| dictionary | decision | direction |
|---|---|---|
| **K3** | ±1 wire slack is per-call-site, and `<= max + 1` is the wrong translation | toward the prototype (D1) |
| **K4** | the adjacent-slice stride is `nticks_per_slice`, never a literal 4 | toward the prototype (D12) |
| **K8** | `disable_dead_mix_cell` must reach the edge-weight charges | toward the prototype (D2) |
| **K5** | flag=2 grouping by the retiled blob is correct; **do not reintroduce** the null-mcell exclusion | toward the toolkit (D3) |
| **K7** | the finer path resampling is correct; the prototype undershoots its own target | toward the toolkit (D7) |

K2 additionally carries §13.3's observations 2 and 3, and K6 records that
`get_two_boundary_wcps(2)`'s two cuts are *replaced*, not dropped — the question
§10.2.9(d) had to answer from scratch.

### §13.5 Gate for the §13.3 fix

The bounds guard is expected to be a no-op, and "expected" is not a gate.
Re-running SBND evt 388 through the production chain on the rebuilt binary
gives a `calib-pr-evt388.json` **md5-identical** to `work-pr29-388-d2on`'s
(`7c9a0488552aa0deb2b0d14a5d81cd21`, arm `work-pr29-388-guard`), i.e. the guard
never fired on this event and the D2 numbers in §12.4 stand unchanged.
`wcdoctest-clus` 89/89, 948 assertions. No config file was touched, so every
md5 in §12.5 is unaffected.

## §14 The 48-event nueCC re-processing — pr/29's share *(2026-08-04)*

> **OWNER VERDICT: ACCEPTED** *(2026-08-04)* — after scanning the two Bee sets,
> *"The new code is much better, we are good for now."* D1, D12 and D2 stay
> SBND production defaults ON, and this document is closed for this round. The
> verdict is a hand scan of 48 nueCC events; it does not clear §13.1's valfast
> debt, does not separate D1+D12 from D2, and does not individually clear the
> events that fell to the `br_filled` sentinel (§14.3). Full record: doc pr/28
> §16.9.

**Owner instruction.** *"Now, you can use the 48 nueCC events to do a round of
new processing with the latest code … 1. compared to before, do we see any major
changes? … 2. Any newly developed processing problems."*

The joint old-vs-new result for pr/28 + pr/29 together, the two gates it rests
on, and the two Bee links are in **doc pr/28 §16** — not repeated here. This
section covers only what is specific to this document: **how much of the change
is pr/29's**, and what the Steiner warnings did.

Method, gates and Bee links: doc pr/28 §16.1–§16.4, §16.7. The two facts this
section leans on:

- **clustering and Q/L are inert to the whole delta** — 96/96 archives
  member-content identical (§16.3), so the Steiner stage sees byte-identical
  input on both arms;
- **every per-bundle tagger verdict is identical too** — `nusel-table.tsv`
  and `nusel-events.tsv` `diff` empty, old vs new (§16.3b), so nothing the
  Steiner change does reaches TGM/STM/FC/LM;
- **the run-to-run noise floor is zero** — 0/48 events differ between two runs
  of the same binary over 17 columns (§16.4). Everything below is signal.

### §14.1 The attribution arm

The three pr/29 knobs are all reachable from the runner, so a third arm at
**the same HEAD binary** isolates them:

```sh
SBND_STEINER_WIRE_TOL=0 SBND_STEINER_ADJ_SLICE=0 SBND_STEINER_EDGE_DEAD_MIX=0 \
  PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-0804 work-vfnuecc48-0804-pr29off data
```

That splits the aggregate cleanly in two, on one binary:

| comparison | meaning | events changed | median \|Δ`E_ν`\| | max \|Δ`E_ν`\| | median Δvtx | max Δvtx |
|---|---|---|---|---|---|---|
| `0804` vs `0804-pr29off` | **pr/29 alone (D1+D12+D2)** | 47/48 | 40.2 MeV | 477 MeV | 0.54 cm | 121.7 cm |
| `0804-pr29off` vs `prod0803` | pr/28 + everything else in the window | 47/48 | 75.2 MeV | 802 MeV | 1.67 cm | 121.8 cm |

All three arms ran on **one binary**: `local/lib/libWireCellClus.so` mtime
`08-04 11:44:10`, unchanged before, between and after the three runs, and
newer than the newest source edit (`SteinerGrapher.cxx`, `11:41:30`) — M1, and
the thing whose absence made §12.7 unattributable.

**Neither half dominates — and on three events the two arms land on the same
vertex while the off-arm lands somewhere else.** The maximum vertex move within
each half is ~122 cm, while the *net* old→new maximum is 62 cm (§16.5). Per
event:

| evt | pr/29 alone | rest alone | net old→new |
|---|---|---|---|
| 469665 | 121.71 cm | 121.80 cm | **4.16 cm** |
| 122660 | 85.23 cm | 85.34 cm | **0.18 cm** |
| 46363 | 44.97 cm | 44.97 cm | **0.00 cm** |

**Read the equal pairs correctly**: these are not two continuous displacements
that happen to oppose. There are only **two** distinct vertex positions on each
of these events — the 08-03 baseline and HEAD-production pick the *same* one,
the pr/29-off arm picks a different candidate, and both columns are the same
distance because both are measuring the same gap. (46363 is exactly 0.00 cm;
that is identity, not cancellation.)

The outlier arm is **HEAD with pr/29 off** — a configuration that has never
been production. Read forward rather than backward: on these events pr/28's
changes *combined with the legacy terminal filter* select a vertex 45–122 cm
away, and **D1/D12/D2 select the original one**. The two fixes are coupled,
not additive, and shipping pr/28 without pr/29 would have been the worst of the
three states.

This is the concrete form of the warning in §12.7 that the §11 and §12 numbers
cannot be summed; it now also applies across documents. Any future attempt to
attribute a change to one document alone needs its own matched arm — the
subtraction is not valid.

### §14.2 The starved-terminal warnings collapse — D1/D12 doing exactly what §11 predicted

`create_steiner_tree: only N terminal(s) remain after filtering (need >=2),
returning empty graph` is the log line emitted when
`filter_steiner_terminals` throws away so many candidates that no tree can be
built. That is the failure mode D1 and D12 were about.

| | prod0803 | 0804 (HEAD) |
|---|---|---|
| lines | **87** | **10** |
| events affected | **23 / 48** | **5 / 48** |
| `produced no steiner_graph for assoc` | 1209 | 1133 |

**78 of 87 starved filters are gone, and the number of affected events drops
from 23 to 5.** §11 measured this on evt 388 as "24 starved clusters" and could
not say how it scaled; it scales. This is the strongest population-level
evidence in the document that D1/D12 were real defects and not a wash.

The cost is visible too, and it is the same mechanism seen from the other side:
more surviving terminals ⇒ larger Steiner graphs ⇒ **+12.1 % wall / +15.5 %
core** over the 48 events (§16.6). Peak RSS is flat (−0.4 %). No new WARN or
ERROR family appears, `rc` is 48/48, and there is no `DL vertex failed`.

### §14.3 Where pr/29 changes a verdict, in both directions

`nue_score = −15` is the `br_filled != 1` sentinel (mechanism in §16.5). The
three-arm view of the events that reach it is the sharpest per-event statement
this round produces:

| evt | Bee idx | prod0803 | HEAD, pr/29 **off** | HEAD, production |
|---|---|---|---|---|
| 122660 | 14 | +4.301 | **−15.000** | −15.000 |
| 268784 | 32 | +4.301 | **−15.000** | **+4.301** |
| 469665 | 46 | +4.129 | −4.301 | **−15.000** |

- **122660** loses its nue evaluation to the *other* half of the window; pr/29
  neither causes nor repairs it.
- **268784** is lost by that half and **recovered by pr/29** — with the Steiner
  knobs off it hits the sentinel, with them on it is back at the saturated
  maximum.
- **469665** is pushed down by both halves, pr/29 the further of the two.

Net over the sample, `nue_score > 0` counts: prod0803 **41**, HEAD-pr29off
**39**, HEAD-production **40** — i.e. **pr/29 is worth +1 event against the
rest of the window**, and the overall net is −1 vs the 08-03 baseline. On 48
events these are single-event differences and are quoted as such; §5 rule 7
applies (reported, not tuned).

### §14.4 What this pays of §13.1

§13.1 recorded that *"every measured number in this document comes from one
event (SBND evt 388)"* and named four questions. Status after this round:

| §13.1 question | status |
|---|---|
| how many events change `event_label` | **answered: 0 / 48** (§16.5) |
| how the 24-starved-clusters number scales | **answered: 87 → 10 lines, 23 → 5 events** (§14.2) |
| whether D1+D12 and D2 effects add or partly cancel | **partly answered** — pr/29 *as a whole* vs the rest partly cancels (§14.1). D1+D12 vs D2 individually is still unseparated; that needs a fourth arm. |
| how many gain or lose a pi0 | **still open** — `pr_scores_table.py` does not carry the pi0 block; it needs `T_kine` or the display dump. |

And the framing caveat stands: **48 nueCC events is not the 572-event valfast
manifest.** This sample is selected for containing a neutrino, so it cannot
speak to the cosmic-dominated population where a terminal-filter change is most
likely to create a false candidate. §13.1 is *reduced*, not closed.

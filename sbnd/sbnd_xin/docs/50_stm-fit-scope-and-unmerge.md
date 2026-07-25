# Doc 50 — What cluster does the STM fit actually see? (unmerge scope vs. clustering-chain merges)

**Status:** investigation only. **No code is changed.** Two candidate remedies
are described in §7 and neither is implemented — they are clustering-level
decisions for the owner.

## Repro

```bash
cd sbnd_xin

# the population census (needs the -stm-fit rounds; d49son = doc 49 knob-on)
python3 stm_main_connectivity.py work-mcp10-d49son work-mcp1000-d49son work-mcp1000b-d49son
python3 stm_main_connectivity.py --gap 3   work-mcp10-d49son work-mcp1000-d49son work-mcp1000b-d49son
python3 stm_main_connectivity.py --gap 15  work-mcp10-d49son work-mcp1000-d49son work-mcp1000b-d49son

# the two reported events
python3 stm_main_connectivity.py --detail 284657:27 work-mcp10-d49son
python3 stm_main_connectivity.py --detail 285185:21 work-mcp10-d49son

# what the un-merge actually did, from the run's own log
grep "UnmergeBundle:pr> cluster" work-mcp10-d49son/nusel_evt285185/wct_nusel_evt285185.log
grep "UnmergeBundle:pr> cluster" work-mcp10-d49son/nusel_evt284657/wct_nusel_evt284657.log
```

Sample: the 30-event MCP2025C scan, tags `work-{mcp10,mcp1000,mcp1000b}-d49son`
(doc 49 knob-on, both arms run under `setarch x86_64 -R`; see
`project_sbnd_pr_chain_aslr_nondeterminism`).

---

## 1. The three questions, answered

### Q1 — "Is what port 5010 shows after un-merge?"

**No.** The viewer's *charge* and the viewer's *fit* come from two different
jobs, on opposite sides of the un-merge:

| viewer layer | source file | written by | un-merge? |
|---|---|---|---|
| grey/red/blue/orange charge points | `ql_evt<ID>/mabc-all-apa.zip` | the **Q/L** job | **before** |
| black-cross STM trajectory | `nusel_evt<ID>/tracking-stm.root` | the **nusel** job | after |
| verdict row (`tgm/stm/fc`, `stmfit`) | `nusel_evt<ID>/nusel-evt<ID>.tsv` | the **nusel** job | after |

`run_nusel_evt.sh:344` inserts `unmerge_bundle` between `switch_scope` and
`steiner` inside the *nusel* job, which reads `ql_evt<ID>/pctree-evt<ID>.tar.gz`.
The Bee zip the viewer loads (`nusel_scan_viewer.py:386`) was written earlier, by
the Q/L job, and never sees the split.

So **the orange squares are pieces the tagger no longer has.** The
"main N (+k merge fragment(s), squares)" annotation is derived from
`real_cluster_id` in the pre-un-merge dump — the very provenance
`ClusteringUnmergeBundle` then uses to remove them. The display is showing
pre-un-merge charge with a post-un-merge trajectory drawn on top.

The fix is cheap and not applied here: `nusel_evt<ID>/mabc-pr.zip` is the
**post**-un-merge dump from the nusel job itself and preserves cluster idents
(verified: evt285185 cid 21 = 1171 pts there vs 1211 pre). Pointing the viewer's
geometry loader at it — or drawing both and labelling the difference — removes
the mismatch. Left for a separate change so this doc stays measurement-only.

### Q2 — "Does the un-merge fully separate main from associated?"

**It does exactly what it claims, and that is less than "fully separate".** It
undoes the Q/L *flash* merge and nothing else. Both reported clusters split
cleanly:

```
evt285185: cluster 21: 272 blobs -> main 264 + 1 associated cluster(s) holding 8  (real mode)
evt284657: cluster 27:  85 blobs -> main  69 + 1 associated cluster(s) holding 16 (real mode)
```

and each output main carries exactly **one** `real_cluster_id` — so there is no
flash-merge residue left. Confirmed per event:

| | pre-un-merge `cid` | post-un-merge `cid` |
|---|---|---|
| evt285185 cluster 21 | 1211 pts, rids **{10, 22}** | 1171 pts, rid **{22}** |
| evt284657 cluster 27 |  337 pts, rids **{21, 32}** |  300 pts, rid **{32}** |

**But the retained main is still not spatially connected**, and that residue is
*inside a single pre-merge `real_cluster_id`* — i.e. it was put there by the
clustering chain's all-APA merge passes, upstream of Q/L and outside the
un-merge's remit by construction:

| | detached components in the post-un-merge main | gap |
|---|---|---|
| evt285185 cluster 21 | 1163 pts + **8 pts** | 18.3 cm |
| evt284657 cluster 27 |  279 pts + **21 pts** | 20.3 cm |

This is the same root cause as doc 45's open item (evt18 cluster 80: four
detached clumps, gaps 52 and 32 cm, `n_bundle=1`, never flash-merged). §4 turns
that one-event anecdote into a measured population.

### Q3 — "Is the STM tagger designed for the main cluster only?"

**Yes, and the toolkit matches the prototype here.**
`TaggerCheckSTM.cxx:196` selects only clusters carrying
`Flags::main_cluster`; associated clusters are passed as a `std::vector<Cluster*>`
that is used *only* for counting, in `check_other_clusters()`
(`TaggerCheckSTM.cxx:2063`). The prototype does the same
(`wire-cell-prod-stm.cxx:815-855`: `main_cluster` gets
`create_steiner_graph`, `additional_clusters` go to `check_stm` as a companion
list). No divergence.

---

## 2. evt284657 grp 5 main 27 — the reported behaviour is real

```
POST-un-merge main (rid 32, 300 pts), components at 5 cm linkage:
   comp n=279  span 22.4 cm  x[ 0.67,11.92] y[23.84,40.29] z[-0.07,10.13]
   comp n= 21  span  1.9 cm  x[ 5.99, 7.55] y[39.43,39.95] z[26.33,27.23]
   min gap 20.26 cm

FIT pass 0: 54 pts, L = 32.29 cm, status = 3
   ends (1.01, 25.93, 1.27) -> (6.77, 39.49, 26.48)
   max step 0.91 cm  (no discontinuity -- the path is dense THROUGH the void)
   traj -> nearest charge of its own cluster: median 2.15, max 10.01 cm
   17 of 54 trajectory points are > 5 cm from ANY charge
   the far END is 0.22 cm from the 21-pt clump  <-- the fit terminates in it
```

Distance from the trajectory to charge as a function of arclength shows the
crossing explicitly: 0.3 cm at s=0, rising to **10.0 cm at s=21.6**, back to
0.6 cm at s=31.7. The fit walks in a straight line across 20 cm of nothing and
lands on the detached clump, which then supplies the exit point the STM logic
reasons about.

Mechanism (all four links verified in source):

```
TaggerCheckSTM.cxx:580   shortest_path() over "steiner_graph"
  <- CreateSteinerGraph.cxx:161  create_steiner_tree(... "ctpc_ref_pid" ...)
  <- CreateSteinerGraph.cxx:126  find_graph("ctpc_ref_pid")
  <- make_graphs.cxx:61          make_graph_ctpc_pid -> connect_graph_with_reference
  <- connect_graph.cxx:167       add_edge(gind1, gind2, dis, graph)   -- MST, NO distance cut
```

`connect_graph.cxx:161-166` looks like it special-cases gaps > 5 cm but both
arms assign the same `dis`; only the extra *directional* edges get a ×1.2 weight
penalty. So any number of detached components in one cluster is stitched into a
single graph, and a shortest path can then run from one to another.

**This is a faithful port, not a bug.** The prototype does exactly the same:
`PR3DCluster::Connect_graph(ref_point_cloud)`
(`pid/src/PR3DCluster_graph.h:1166`) builds a `prim_minimum_spanning_tree` over
the components and adds every MST edge with no maximum length. Whether WCP's
*main clusters* fragment as much as ours do is **not established here** — that
would need the same census run on the prototype, which was not done. Do not read
this doc as claiming a prototype/toolkit divergence (M15).

## 3. evt285185 grp 14 main 21 — does **not** show it

Measured, same tooling:

```
POST-un-merge main (rid 22, 1171 pts): 1163 pts + 8 pts, gap 18.27 cm
FIT pass 0: 234 pts, L = 145.74 cm, status = 0 (accepted)
   ends (201.04, -1.46, 22.80) -> (130.55, 42.50, 135.09)   = the body's own corners
   max step 0.93 cm;  traj -> own charge: median 0.29, MAX 1.52 cm
   0 trajectory points > 5 cm from charge
   the 8-pt clump is 18.66 cm OFF the trajectory, at interior arclength 108/146 cm
   the 40-pt flash-merge fragment (rid 10) is 224.9 cm from the trajectory
   no OTHER cluster has charge within 15 cm of the trajectory
```

So on this event the fit is charge-supported end to end and includes neither the
far fragment nor the 18 cm side clump. The reported visual has two candidate
causes, both unconfirmed:

- the 8-pt clump at (165.3, 20.3, 120.5) lies almost exactly on the track in the
  **x–y** projection (the track spans x 130→201, y −1→43) while being 18.7 cm
  off it in y–z and x–z — one panel out of three makes it look attached;
- the display draws the 225 cm-away `rid 10` fragment as part of red/orange
  main 21 at all, per Q1.

Naming the projection would settle which.

## 4. How common is this? (128 fitted mains, 30 events)

`stm_main_connectivity.py`, post-un-merge geometry (`mabc-pr.zip`) vs the fitted
trajectory (`tracking-stm.root`, last pass):

| | gap=3 cm | **gap=5 cm** | gap=10 cm | gap=15 cm |
|---|---|---|---|---|
| main NOT connected | 90 (70 %) | **88 (69 %)** | 82 (64 %) | 72 (56 %) |
| trajectory point > 5 cm from any own charge | 33 (26 %) | **33 (26 %)** | 33 (26 %) | 33 (26 %) |
| trajectory **END** inside a detached clump | 44 (34 %) | **36 (28 %)** | 33 (26 %) | 29 (23 %) |

The linkage threshold is a free parameter, so `ncomp` is the soft number — it
drops from 70 % to 56 % across 3→15 cm. The `stray` count (a fitted point far
from *any* charge of its own cluster) is threshold-free and constant at
**33 (26 %)**; that is the number to quote. The end-in-clump count is stable at
23–34 %.

Gap sizes for the 36 end-in-clump cases (5 cm linkage): 3 under 10 cm, 8 at
10–20, 7 at 20–40, **18 over 40 cm** (max 94.3).

**These are not dead-channel bridges.** `connect_graph_ctpc` exists precisely to
link charge across dead wires, and a gap inside a dead region would be the
algorithm working. Tested directly against the Bee `channel-deadarea` layers in
the same zip: the dead fraction of the bridging segment is **0 % for every one of
the 36**. In this MC sample the dead-area polygons total ~94 cm² of the
~200 000 cm² (y, z) plane per TPC, i.e. SBND MCP2025C is essentially
dead-channel-free. On data with real dead regions this test must be re-run before
reusing the conclusion.

### What it costs in verdicts

Joining the census to the 329 scan rows: of the **44 STM tags** in these three
tags, **12 (27 %) rest on a fit whose endpoint sits in a detached clump**, and in
**11 of those 12 the trajectory itself flew through the void** (`nfly > 0`):

| evt | cid | endgap | traj pts in void | len_main | in_beam |
|---|---|---|---|---|---|
| 287825 | 2 | 94.3 | 145 | 123.2 | 0 |
| 288397 | 2 | 60.4 | 98 | 71.0 | 0 |
| 286527 | 18 | 42.4 | 75 | 47.4 | 0 |
| 286021 | 15 | 38.6 | 49 | 238.6 | 0 |
| 290135 | 16 | 30.0 | 34 | 171.3 | 0 |
| 288859 | 7 | 18.6 | 16 | 98.9 | 0 |
| 286197 | 8 | 17.2 | 15 | 252.3 | 0 |
| 288639 | 10 | 17.1 | 12 | 203.0 | 0 |
| 288067 | 14 | 16.8 | 17 | 150.2 | 0 |
| 287517 | 8 | 14.8 | 9 | 331.1 | 0 |
| 285999 | 9 | 13.8 | 6 | 384.7 | 0 |
| 287099 | 19 | 9.6 | 0 | 58.7 | 0 |

All twelve are out-of-beam, all have an accepted fit (`status = 0`). Being
out-of-beam, none of them changes a neutrino candidate today — but a tag whose
exit point comes from a different physical object is right for the wrong reason,
and the same mechanism has already been seen to *remove* a tag: doc 49's single
STM loss, **evt284657 m27**, is precisely a case where the fit ends in a detached
clump (§2).

## 5. So: is the un-merge doing the correct thing?

**Yes, within its contract, and its contract is narrower than "hand the STM
tagger a connected track".**

- It restores the prototype's *data product* (`main_cluster` +
  `additional_clusters`) from the flash merge. Both reported clusters split with
  exact per-blob provenance, `4 exact/provenance, 0 proxy/component`, and each
  output main holds one `real_cluster_id`. Nothing left to undo at that layer.
- It cannot address clustering-chain merges: those pieces share one
  `real_cluster_id`, so there is no bookkeeping record to invert. Splitting them
  is a *clustering decision*, which is exactly why `mode "real"` refuses to fall
  back to `mode "component"` (doc 45 / commit 35f2b72a — the relaxed graph does
  not join the two halves of a cathode crosser).
- 69 % of fitted mains being multi-component at 5 cm is therefore not an
  un-merge failure. It is the state of the clustering output, now visible because
  the un-merge removed the other, larger source of composite mains.

## 6. What this does not say

- It does not say the 12 tags are wrong. A 20 cm gap can be a real track through
  a badly-reconstructed region; only a hand scan decides.
- It does not say gap-bridging should be removed. Both the toolkit and the
  prototype bridge unconditionally by design; a hard cut would break genuine
  tracks across sparse regions and cathode crossers.
- It does not measure prototype main-cluster fragmentation (§2).
- The 128-cluster denominator is itself run-dependent: a cluster that loses
  `steiner_pc` is never fitted, and `CreateSteinerGraph` varies run to run
  unless `setarch x86_64 -R` is used. These roots were, so the numbers are
  reproducible — but a different ASLR layout would fit a slightly different set.

## 7. Candidate remedies — described, NOT implemented

Ranked by how well each matches something already accepted in this codebase.

1. **An STM analogue of TGM's `main_component_pairs` (doc 36/38).** TGM already
   solved the same problem inside the tagger: require the endpoint pair to lie in
   the largest 30 cm-step path component, rather than trusting the cluster
   object. For STM the natural form is: build the trajectory only between
   extremes of the dominant path component, so a detached clump can neither
   supply the exit point nor be flown to. Doc 45 already names this as the
   suggested middle ground for evt18 cluster 80. Default-OFF knob, byte-identical
   when off.
2. **A gap flag on the fit rather than a change to it.** Persist, per fitted
   cluster, the largest trajectory-to-charge distance (already computed here from
   the dump) and the arclength fraction spent in void; expose it as a `stmfit`
   reason and in the scan table. This changes no verdict — it makes the 12 cases
   visible during a hand scan and gives a cut to calibrate later.
3. **A distance-limited bridging mode for the PR graph.** Most invasive and least
   recommended: cap MST edge length in `connect_graph_with_reference`. It touches
   the graph every tagger shares, deviates from the prototype, and would need its
   own full A/B.

Option 2 costs nothing in physics and would let the owner judge option 1 on
evidence. Neither is done here; both change reconstruction reasoning and are the
owner's call (escalation rule 7).

## 8. Open items

- Hand-scan the 12 tags in §4 and evt284657 m27, deciding case by case whether
  the bridged gap is one track. That is the input option 1 needs.
- Point the viewer at `mabc-pr.zip` (Q1), or draw both layers and label the
  removed pieces. Until then, every scan is reading pre-un-merge charge.
- Re-run the dead-region test on a data sample before reusing the "0 % dead"
  result.
- Doc 45's evt18 cluster 80 remains the reference single-event case; it is the
  same phenomenon as §2.

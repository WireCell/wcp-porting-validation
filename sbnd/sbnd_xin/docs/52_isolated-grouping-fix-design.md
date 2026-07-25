# Doc 52 — Design: preserve the main + associated grouping end to end

**Status: DESIGN ONLY. Nothing is implemented.** Every stage below changes what
production emits, so each needs its own default-OFF knob and gate, and the last
one is an owner decision (escalation rule 1). Follows doc 51, which located the
defect, and doc 50, which measured its cost.

## 0. First, a correction — `switch_scope` does exactly what you described

Your model is what the code does. `clustering_switch_scope.cxx`:

```
add_corrected_points(pcts, "T0Correction")  ->  per-blob filter (1 = in scope, 0 = out)
separate(cluster, filter_results, true)     ->  two sub-clusters: id=1 in-scope, id=0 out
for each part:  carve(src, aname)           ->  row-partition the array by the SAME filter,
                                                re-attach to the part           (:117-135)
```

Keep the arrays, change the scope, drop the rows that went out of scope — that is
literally the implemented algorithm. My "drops it" phrasing was misleading. The
one and only problem is that `carve()` is called on a **hardcoded two-name list**:

```cpp
carve(src_rcid,   "real_cluster_id");     // clustering_switch_scope.cxx:132
carve(src_rcmain, "real_cluster_main");   // :133
```

`isolated` is not in that list, so it is the one array that does not survive the
rebuild. A missing name, not a design difference. Same for the `has_pcarray`
snapshot at `:88-95` that feeds it.

## 1. The four defects

| # | where | defect |
|---|---|---|
| **D1** | `clustering_isolated.cxx:507` | the merge records the member partition as `0,1,2,…` — the main is never marked, so the consumer's main-preserving branch cannot fire |
| **D2** | `clustering_examine_bundles.cxx:243` | overwrites `("isolated","perblob")` with relaxed-graph connectivity; the member partition is replaced, by design |
| **D3** | `clustering_switch_scope.cxx:132-133` | `isolated` missing from the carve list (§0) |
| **D4** | `merge_clusters` (`ClusteringFuncs.cxx:78`) | a merge **rebuilds** provenance arrays; it cannot **carry** an existing one, so any per-blob array dies at the next merge |

D4 is the general one and the reason this keeps happening: the codebase can
*write* provenance (`orig_id_aname`/`orig_main_aname`) and *carve* it through a
split (`switch_scope`), but has no way to *carry* it through a merge. Doc 38 hit
this and solved it once, by hand, for `real_cluster_id`.

## 2. What already exists and can be reused

`merge_clusters` already implements per-blob pre-merge ident **plus a main
marker** — `clustering_isolated` simply does not ask for them:

```cpp
merge_clusters(g, grouping, aname, pcname, orig_id_aname, flags_from_longest, orig_main_aname);
//                                        ^^^^^^^^^^^^^^                      ^^^^^^^^^^^^^^^^
```

`orig_id` gets each blob's pre-merge cluster ident (`ClusteringFuncs.cxx:208`);
`orig_main` marks the representative member's blobs (`:223-236`) as
`rep_ident = have_flash ? best_flash_ident : best_any_ident`. **Verified: at the
per-APA isolated stage there are no flashes, so it falls to `best_any_ident`,
which is assigned on the no-flash path (`:184-189`, gated on `save_origmain`) as
the longest member** — which is exactly the prototype's `max_cluster`
(`ToyClustering_isolated.h:296-302`). So the semantics we want are already
implemented and already match the prototype; only the call site is missing.

## 3. Staged fix

Each stage is independently useful, independently gated, default OFF, and
byte-identical when off.

### Stage 0 — mark the main in the existing array (your item 1)

`clustering_isolated`: mark the longest member's blobs `-1` in
`("isolated","perblob")` instead of writing `0,1,2,…`.

Then `clustering_examine_bundles.cxx:181-205` — code that exists today and can
never run — starts firing: instead of `flag_largest` (re-guess the main as the
longest *connectivity component*), it computes the overlap between the incoming
main blobs and each new connectivity group and carries the main designation
across. That is the "right thing" you expected, and it is right.

Cheapest implementation: post-process `cc` in `clustering_isolated` (it knows the
group's longest member), or add a `mark_main_negative` option to `merge_clusters`.
Consumers affected: QLMatching `decompose_cluster_groups`, the Bee `isolated`
layer, `examine_bundles`.

**But Stage 0 does not fix the two showcase events, and it is important to say
why.** Both are flash-merged at the all-APA stage (272 → 264 + 8 blobs;
85 → 69 + 16). The flash merge is `merge_clusters` at
`clustering_examine_bundles.cxx:153`, and it destroys the incoming `isolated`
array (D4) — so a few lines later `old_cc_array` is empty, the size check fails,
and the main is re-guessed as longest anyway. Stage 0 fixes the main designation
for clusters that are *not* flash-merged and is a prerequisite for everything
else; it is not sufficient on its own.

### Stage 1 — write a dedicated, never-recycled provenance pair

At `clustering_isolated.cxx:507`:

```cpp
merge_clusters(g, live_grouping, "isolated", "perblob",
               "assoc_cluster_id", /*flags_from_longest=*/false, "assoc_cluster_main");
```

New arrays, so D2 cannot touch them: `("isolated","perblob")` stays the rolling
connectivity partition that `examine_bundles` and QLMatching own, while
`assoc_cluster_id` / `assoc_cluster_main` are the immutable record of "which
pre-merge cluster was this blob, and which member was the main" — the exact
analogue of `real_cluster_id` / `real_cluster_main` for the flash merge. No new
C++ machinery (§2).

### Stage 2 — make provenance survive: carry across merges, carve through splits

This is the systematic part, and it closes D3 and D4 for good rather than for one
array name.

- **`merge_clusters`: a `carry_arrays` list.** For each named per-blob array,
  concatenate the members' arrays in the *same order* `descs` is iterated and
  `take_children` is called, checking the running length against
  `fresh_cluster.nchildren()` at each step — the invariant
  `cc.resize(fresh_cluster.nchildren(), parent_id)` already relies on. Members
  lacking the array contribute a **distinct sentinel** (not `0`, not `-1`: both
  mean something) so a consumer can distinguish "never grouped" from "is the
  main". Fail open — drop the array with a warning on any size mismatch, exactly
  as `carve()` does.
- **`switch_scope`: make the carve list configurable** instead of the hardcoded
  two names (§0). Then adding an array to the chain is a config change, not a
  C++ change, and the next provenance array does not repeat this bug.
- **Tensor persistence:** the pctree save needs the same homogenization
  `save_real_cluster_id` does, with doc 38's two gotchas — TensorDM silently drops
  heterogeneous same-named PC keys (now warns), and `separate()`/`from()` drop
  node-local PCs.

The full survival path for the assoc pair, all five hops:

| hop | mechanism needed |
|---|---|
| per-APA `examine_bundles` (tr15) | none — `use_flash_t0=false`, so it does not merge; different array name, untouched |
| all-APA `switch_scope` (tr00) | carve (Stage 2) |
| all-APA `examine_bundles` flash merge (tr08) | **carry** (Stage 2) |
| pctree tensor save | homogenize, like `save_real_cluster_id` |
| nusel job `switch_scope` | carve (same code) |

### Stage 3 — consume it: un-merge the isolated grouping

Add a mode to `ClusteringUnmergeBundle` (doc 45) that splits on
`assoc_cluster_id`, main = `assoc_cluster_main`. Everything it needs already
exists there: `separate(remove=false)` re-carving, flag clearing, the "never fall
back to a connectivity proxy" rule, and the requirement to run **before the
steiner stage**.

Running both un-merges in order — flash first (outer), then isolated (inner) —
reproduces the prototype's layout exactly: one `main_cluster` plus
`additional_clusters` as separate objects, which is what
`wire-cell-prod-stm.cxx:816-828` hands `check_stm`. `check_other_clusters()`,
`TaggerCheckNeutrino`'s companion list and TGM all benefit at once, so this is
preferable to teaching `TaggerCheckSTM` to filter blobs itself.

### Stage 4 — the end state: stop merging (owner decision)

Have `clustering_isolated` leave the objects separate and write a parent-id
*cluster scalar* (the mechanism exists — `matched_flash_gid` is one), matching the
prototype's `parent_cluster_id`. Then main-only fitting falls out everywhere with
no un-merge at all, and Stages 1-3 become dead weight. This changes the cluster
count entering QLMatching, so it needs a full A/B and a check that
`decompose_cluster_groups` still finds its groups. Worth doing only after Stages
1-3 prove the physics is better.

## 4. Which "main"? — a decision to make explicitly

After doc 51's probe there are **two** definitions available at the PR stage, and
they are different partitions:

| | definition | source | property |
|---|---|---|---|
| **A** | the pre-merge member cluster | `assoc_cluster_id` (Stage 1) | prototype parity; **not guaranteed connected** |
| **B** | the largest relaxed-connected component | `connected_blobs`, already used by TGM `main_component_pairs` (doc 36/38) | guaranteed connected; can cut a genuine track |

Measured: relaxed connectivity separates both showcase clumps — cluster 21
272 → 255 + 4 components, cluster 27 85 → 63 + 3 (`-unmerge-comp` probe,
`work-mcp10-trace51`). So B would fix these two events on its own.

But A is what the prototype fits, and A alone does **not** fix doc 45's evt18
cluster 80: one `real_cluster_id`, four detached clumps with 52 and 32 cm gaps —
a single pre-merge member that is internally disconnected. Conversely B is the
partition doc 45 refused for the flash merge because relaxed connectivity breaks
cathode crossers.

Recommendation: **give STM definition A** (prototype parity, and it is the
structure every other consumer wants), and treat the residual
internally-disconnected mains as a separate, smaller problem to be handled inside
the fit by a B-style path-component restriction — the STM analogue of
`main_component_pairs` from doc 50 §7. A fixes the representation; B guards the
fit. Doing only one leaves a known hole either way. This is the choice I would
most want your call on before any code is written.

## 5. Validation plan

Per stage, in this order:

1. `./build/clus/wcdoctest-clus`.
2. Knob-off compiled-config byte-identical (`wcsonnet` + `cmp`), and knob-off
   **output** identical — for the clustering stage that means `hash_archive.py`
   on `mabc-*.zip` plus the pctree tarball, not `cmp` (M2).
3. Freshness proof before any A/B (M1); **both arms under `setarch x86_64 -R`**,
   because this chain is ASLR-non-deterministic at ±7 STM tags
   (`project_sbnd_pr_chain_aslr_nondeterminism`) — and Stage 0 in particular is
   expected to produce a *small* diff, which is exactly the regime where that
   noise floor fabricates or erases results.
4. Knob-on: quote the array contents, not just verdict counts — for Stage 1, the
   `assoc_cluster_id` partition of cluster 21 must be 1164 + 8 and of cluster 27
   279 + 21 (doc 51 §2). That is a direct check, independent of any verdict.
5. Re-run doc 50's `stm_main_connectivity.py`: the headline "26 % of fits stray
   >5 cm from their own charge" should fall. If it does not, the fix did not
   reach the fitter.
6. Then, and only then, a verdict A/B on the 30-event manifest, and a hand scan
   of what moves.

## 6. What I would not do

- **Do not cap the MST edge length** in `connect_graph_with_reference`. It is a
  faithful port of `PR3DCluster::Connect_graph`, it is shared by every tagger, and
  gap-bridging inside a genuine main is wanted.
- **Do not touch the 80 cm `small_big_dis_cut`.** Identical to the prototype
  (doc 51 §3); the criteria are not what is wrong.
- **Do not implement a late un-merge alone.** The information it needs does not
  exist at that point (doc 51 §4b); it requires Stages 1-2 first.

## 7. Open

- §4 is unresolved and is the gating decision.
- Stage counts assume the SBND per-APA + all-APA layout. PDHD/PDVD run the same
  `clustering_isolated`, so Stage 0's diff will not be SBND-only — those
  detectors' gates have to run too before anything defaults on.
- Whether uBooNE's mains are as fragmented as ours is still unmeasured
  (doc 50 §6); that number would tell us how much of this is a porting artifact
  versus a real detector difference.

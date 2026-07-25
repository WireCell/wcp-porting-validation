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

## 4. Which "main"? — RESOLVED: not a decision, two layers

**Correction (owner, 2026-07-25).** An earlier version of this section posed an
A-vs-B choice:

| | definition | source | property |
|---|---|---|---|
| **A** | the pre-merge member cluster | `assoc_cluster_id` (Stage 1) | prototype parity; **not guaranteed connected** |
| **B** | the largest relaxed-connected component | `connected_blobs`, already used by TGM `main_component_pairs` (doc 36/38) | guaranteed connected; can cut a genuine track |

That framing was wrong. It came from asking "what should `TaggerCheckSTM` treat as
the main", when the intended architecture is: **run `ClusteringUnmergeBundle`
first, then STM/PR on the resulting main** — the prototype's layout, where
`wire-cell-prod-stm.cxx:816-828` fits `main_cluster` and carries
`additional_clusters` alongside. Under that architecture the representation is
**A** by construction, and **B is nothing but `mode="component"`** — the proxy
that doc 45 (`35f2b72a`) already demoted to opt-in precisely because relaxed
connectivity breaks cathode crossers. There is no choice to make.

(TGM's `main_component_pairs` predates the un-merge and is a *different* layer,
see below — not the alternative representation.)

### So the whole problem is missing write-side information

Two pieces, and in both cases the **reader is already correct**:

1. **The isolated grouping writes no main marker.** `clustering_isolated.cxx:507`
   calls `merge_clusters(g, live_grouping, "isolated")` without `orig_main_aname`,
   so `groups_from_provenance` has nothing to split on (Stage 0). Marking it is
   necessary but not sufficient — the flash merge destroys the array before any
   reader sees it, hence the carry in Stage 2.

2. **`merge_clusters` marks exactly ONE main** — `ClusteringFuncs.cxx:227`,
   `rep_ident = have_flash ? best_flash_ident : best_any_ident`, then
   `orig_main[i] = (orig_id[i] == rep_ident)`. Every other member is therefore
   "associated", including a second member that is itself a main.

But `ClusteringUnmergeBundle` **already keeps the union of all main-marked
members** (`:211-219`): `others` is built from `rmain[i] == 0` only, and every
`rmain != 0` blob keeps `groups[i] = -1` and stays with the retained cluster. That
is exactly the crosser-safe rule §4a derives. The split machinery needs **no
change**; it is simply never handed more than one main to keep. This is the
mechanical reason evt288287 cluster 13 is split today (§4a): both halves are real
mains, one is marked, the other is indistinguishable from co-merged junk.

⇒ The fix is write-side only: **mark as main every member that is itself a main**,
and carry that marking across subsequent merges.

### The one residual — and TGM already ships the guard for it

`groups_from_provenance` returns `no_split` when `nmain == nb`
(`ClusteringUnmergeBundle.cxx:207`), deliberately: an internally-disconnected
*single* pre-merge member — doc 45's evt18 cluster 80, one `real_cluster_id`, four
detached clumps with 52 and 32 cm gaps — was never "merged in", and splitting it
on connectivity would undo the chain's intentional long-range merges
(`extend`/`regular`/`parallel_prolong`), which the prototype keeps inside one
`PR3DCluster`. So that class is out of the un-merge's remit **by design**, and no
choice of main representation touches it. It has to be handled either

- **upstream** — if the pass that merged it was not one of the deliberate
  gap-jumpers, that is a separate bug; `stm_merge_attribution.py` can now name the
  pass (doc 51 §8, unmeasured for evt18 clus 80), or
- **inside the fit** — a path-component restriction, the STM analogue of TGM's
  `main_component_pairs` (doc 36/38), which restricts endpoint pairs to a single
  relaxed component.

TGM's guard is therefore *not* redundant with the un-merge: it is the second
layer, for exactly this residual class, and STM currently has no equivalent. Two
layers, not two definitions.

## 4a. Interaction with cross-TPC (cathode-crosser) merging

The isolated grouping and the cross-TPC merge never meet, which is what makes
this safe — but only if the un-merge rule is stated correctly, and there is a
name-collision trap plus a **pre-existing** problem in the same area.

### Where each happens

`cm.isolated()` appears **only** in the per-APA pipeline
(`cfg/.../sbnd/clus.jsonnet:236`, `clus_per_face`); the all-APA pipeline
(`:341` onward) has no isolated pass. Each per-APA MABC sees one TPC, so **the
isolated grouping is intra-TPC by construction and can never group across the
cathode.** Every cross-TPC merge happens later, at all-APA: the generic passes
tr01-tr06 (`use_flash_t0=true`), `cathode_connect` at tr07, then the flash merge
inside `examine_bundles` at tr08.

### The rule that keeps crossers whole

The two halves of a cathode crosser are each the **main** of their own per-APA
isolated group. So after a cross-TPC merge the merged cluster carries **two
main-marked members**. The un-merge rule must therefore be:

> split off only members with `assoc_cluster_main == 0`; keep the union of all
> main-marked members together as the retained main.

| merged cluster contains | result |
|---|---|
| main (TPC0) + main (TPC1) — a crosser | stays whole ✓ |
| main + associated — doc 51's showcase | clump split off ✓ |
| main + main + associated | halves kept, clump split ✓ |

This works *by construction*: a cross-TPC merge joins two mains, whereas the
isolated merge joins a main to associated pieces. The distinction is already in
the data — it does not need a geometric test.

### Trap: per-APA idents collide across TPCs

`assoc_cluster_id` would hold **per-APA** cluster idents, and each per-APA MABC
renumbers 1..N independently (`cluster_id_order: 'tree'`). Doc 51's trace shows
the collision directly: apa0's body carried cids 2,6,8 while apa1's carried 2,7.
After a cross-TPC merge, TPC0's cluster 7 and TPC1's cluster 7 are
indistinguishable and the un-merge would fuse two unrelated members. This is why
`real_cluster_id` is unaffected — it is written at the all-APA stage, where idents
are already global. **The assoc id must be made globally unique at write time**
(encode the APA in the value, e.g. `apa*10000 + ident`), or the carry mechanism
must offset each member's values as it concatenates.

### Stage 2's carry must cover the cross-TPC merges, not just the flash merge

`cathode_connect` calls `merge_clusters(g, live_grouping)` with **no provenance
arguments at all** (`clustering_cathode_connect.cxx:607`), and so do the generic
all-APA passes. Those merges happen *before* `examine_bundles`, so without carry
the assoc arrays die at the first cross-TPC merge — earlier than doc 51 §4b's
chain suggests. Carry has to be a property of `merge_clusters` itself (as
designed in Stage 2), not something added at one call site.

### Pre-existing: the flash-merge un-merge already splits a real crosser

Measured over the 30-event manifest (328 bundles): **141 clusters are
flash-merged, and 101 of those have substantial members on both cathode sides.**
Filtering for a crosser signature — both members reaching |x| < 8 cm with a gap
< 15 cm — leaves 2:

| event | cluster | members | gap | \|x\|min | PCA \|cos\| |
|---|---|---|---|---|---|
| 288287 (mcp1000) | 13 | 1457 + 309 pts | 6.31 cm | 2.06 / 0.50 | **0.997** |
| 289805 (mcp1000b) | 21 | 102 + 129 pts | 2.64 cm | 1.49 / 0.55 | 0.670 |

The first is collinear to 0.997 with transverse continuity (dy 5.67, dz 1.06 cm)
and both halves ending at the cathode: a genuine crosser that `cathode_connect`
missed and only the flash merge joined. **`ClusteringUnmergeBundle` — default ON
since doc 45 — splits it, 526 blobs → main 445 + 81**, and the bundle is
currently tagged nothing at all (tgm/stm/fc = 0/0/0, out-of-beam, 159.8 cm main).
A through-going cathode-crossing cosmic losing its far end is exactly how a TGM
endpoint pair goes missing. Hand-scan candidate, not a proven mis-tag. The second
is only 48° collinear and both halves are short — ambiguous.

This is **not** introduced by anything above; it is the same two-mains issue in
the *flash* merge, where `real_cluster_main` marks only ONE representative (the
flash donor), so the other half of a crosser is indistinguishable from co-merged
junk. Stage 2's carry would actually fix it: with `assoc_cluster_main` carried
through, the flash un-merge could refuse to demote a member that is itself a
main. Worth treating as its own item — it affects production today, whereas
everything else in this doc is about a knob that does not exist yet.

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

- §4 is **resolved** (representation = A, un-merge first, then STM/PR). What is
  still open from it: whether the residual internally-disconnected-single-member
  class needs an STM `main_component_pairs` analogue, or whether the pass that
  merged evt18 cluster 80 is itself the bug — run `stm_merge_attribution.py` on
  that event to decide.
- Stage counts assume the SBND per-APA + all-APA layout. PDHD/PDVD run the same
  `clustering_isolated`, so Stage 0's diff will not be SBND-only — those
  detectors' gates have to run too before anything defaults on.
- Whether uBooNE's mains are as fragmented as ours is still unmeasured
  (doc 50 §6); that number would tell us how much of this is a porting artifact
  versus a real detector difference.

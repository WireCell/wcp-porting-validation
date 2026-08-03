# Doc 52 — Design: preserve the main + associated grouping end to end

**Status: Stages 1-3 IMPLEMENTED, default OFF — see §8.** Stage 0 was subsumed by
Stage 1 (dedicated arrays are strictly safer than mutating the shared `isolated`
array) and Stage 4 remains an owner decision (escalation rule 1). Follows doc 51,
which located the defect, and doc 50, which measured its cost.

**Repro (knob-off gate, all four checks in §8.3):**
```
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild && ./build/clus/wcdoctest-clus
mkdir -p /home/xqian/tmp/d52base && git archive HEAD~1 cfg | tar x -C /home/xqian/tmp/d52base
# then the two wcsonnet compiles of sbnd_xin/wct-clus-matching-perevt.jsonnet and
# wct-pr-perevt.jsonnet against $WIRECELL_PATH = baseline vs current cfg (§8.3)
```

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

(B ≡ `mode="component"` is an exact identity, not an approximation:
`ClusteringUnmergeBundle.cxx:167` defaults `m_graph_name{"relaxed"}`, the same
graph doc 51's `-unmerge-comp` probe used.)

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

#### Three things that rule makes load-bearing

**(a) Absent provenance ⇒ main.** A cluster arriving at all-APA may never have
been through `clustering_isolated` and so carries no assoc arrays at all. The
default must be "a standalone cluster is a main, not an associated fragment" —
defaulted the other way, the crosser fix silently fails (both halves of
evt288287 clus 13 reach the flash merge as plain clusters) and it would only
surface at gate time. This is what makes §4a's "Stage 2's carry would fix it"
actually follow.

**(b) The union at `:211-219` is the general protection; `nmain == nb` is only a
special case of it.** When two mains are merged and *nothing* is associated to
either, all blobs have `rmain != 0` ⇒ `nmain == nb` ⇒ `Prov::no_split` (`:207`),
so the cluster is not even visited. But that short-circuit is **not** what keeps
crossers whole — as soon as either half carries a delta ray, `nmain < nb` and the
cluster *is* split, with both mains retained by the union. Do not reason about
`nmain == nb` as load-bearing for crossers, and do not "fix" it to force a split:
its actual job is the residual exclusion described below.

**(c) `real_cluster_main` has other readers, and one is production.**
`TaggerCheckTGM.cxx:790` (`main_component_mode="real"`, doc 36/38) reads it as a
per-blob boolean — `real_main[bidx] != 0 ? 1 : 0` — so multi-marking cannot
corrupt its semantics. But it is not inert: it *widens* "end is in the main", so
TGM would stop rejecting endpoint pairs whose far end is a crosser's other half
(`main_pair_rejects`, `:795-806`). That is the correct direction, and it is
precisely the evt288287-class recovery — but it is a TGM behavior change and
belongs in the A/B expectations, not discovered at gate time. The other two
readers are presence-check / carve only
(`MultiAlgBlobClustering.cxx:2381`, `clustering_switch_scope.cxx:93`).

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
  gap-jumpers, that is a separate bug; `scripts/analysis/stm/stm_merge_attribution.py` can now name the
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

## 4b. Readiness review: the two invariants, TGM, PDHD/PDVD

Owner's two requirements, 2026-07-25:

> 1. the associated cluster merged by ClusteringIsolated would not be actually
>    merged to main cluster
> 2. The cathode crossing tracks merged should stay merged (main + main, or
>    main + associate) and not be separated after the Unmerge operation

### The taxonomy that satisfies both at once

Of the **12** `merge_clusters()` call sites in `clus/src/`, exactly one passes
provenance arguments (`clustering_examine_bundles.cxx:153`). Split them by intent:

| kind | call sites | prototype behavior | un-merge |
|---|---|---|---|
| **grouping** — bookkeeping, puts related objects in one bucket | `clustering_isolated.cxx:507`, `clustering_examine_bundles.cxx:153` (flash) | prototype does **not** merge: `Clustering_isolated` returns `main → [(assoc, dis)]` with `live_clusters` untouched; the flash grouping is ours | **undoable** |
| **connection** — a physics claim that charge is one particle | `extend`, `regular:523`, `parallel_prolong:366`, `close:243`, `connect:737,833`, `deghost:805`, `live_dead:359`, `neutrino:1011`, `cathode_connect:607` | prototype keeps them inside **one** `PR3DCluster` | **permanent** |

**The rule for a connection merge is: concatenate the members' existing roles,
overwriting nothing.** Not "promote every member to main" — that was considered
and is wrong: promoting would sweep up any delta ray that `clustering_isolated`
had grouped onto either half, and the isolated un-merge would then never split it
off. That is requirement 1 broken on exactly the cathode crossers, i.e. the
original bug resurfacing. Carrying roles unchanged gives a crosser with delta rays
on both halves `nmain < nb`: it **is** split, both mains are retained by the union
at `ClusteringUnmergeBundle.cxx:211-219`, and only the delta rays leave. Both
requirements hold simultaneously.

| requirement | what guarantees it |
|---|---|
| 1 — associated never really merged | Stage 0 marks the main + Stage 2 carries the array + Stage 3 splits on it |
| 2 — cross-cathode merge survives | "absent provenance ⇒ main" (rule (a) above) + the union at `:211-219`. A connection merge joins two clusters that are each a main-or-standalone, so both survive as main. |

**The one way requirement 2 still fails**, stated as a known limit: the crossing
tip on the far side was itself mis-grouped as *associated* to an unrelated main,
`cathode_connect` bridged through that tip, and the un-merge splits it off — the
retained main then keeps an unrelated cluster and loses the tip. This is a
pre-existing `clustering_isolated` mis-grouping, not something the un-merge
creates, and no provenance rule can repair it. Detectable by the same
`scripts/analysis/stm/stm_merge_attribution.py` route if it ever shows up in a scan.

### TGM: no improvement needed — `mode="real"` goes inert by itself

TGM already runs after the un-merge: SBND's `clus_pr` places `unmerge_bundle`
between `switch_scope` and `steiner` (`cfg/.../sbnd/clus.jsonnet:574`), taggers
after. So the interface is *already* unified in ordering; what is not unified is
that TGM carries its own main-detection from before the un-merge existed.

That resolves itself. `ClusteringUnmergeBundle.cxx:339` calls `carve(cluster, -1)`,
which keeps exactly the rows with `groups[i] == -1`, i.e. exactly the `rmain != 0`
rows (`:216-219`). So on the retained main **every** `real_cluster_main` entry is
non-zero, and `TaggerCheckTGM.cxx:787-793`'s `end_in_main` returns 1 for every
point ⇒ `main_pair_rejects` (`:795-806`) can never fire. `main_component_mode
= "real"` is **inert by construction** after a correct un-merge, not merely
relaxed.

Recommendation: **do not extend TGM.** Once the un-merge is complete, retire
`mode="real"` as redundant and keep `mode="component"` as the deliberate second
layer for the internally-disconnected-single-member residual (§4, `nmain == nb`
class) — the same second layer STM still lacks. One mechanism, one stated purpose,
no duplicate notion of "main".

### PDHD / PDVD: safer than SBND, and the reason is config-level

`cm.isolated()` is **commented out at per-face** (`pdhd:250`, `protodunevd:295`)
and active at **per-drift-group** scope (`pdhd:482`, `protodunevd:528`), where it
is the **last** pass of stage 3 — after `extend`/`regular:1,2`/
`parallel_prolong`/`close`/`extend_loop`/`separate`/`connect1`/`deghost`/
`examine_x_boundary`/`neutrino`.

The owner's specific worry — PDHD/PDVD also merge across TPCs on the **same** side
of the cathode — is therefore answered structurally: the per-group scope *is* the
same-side multi-APA scope, so **every same-side cross-APA merge happens upstream of
the provenance write** and is invisible to it. At the moment the isolated array is
written, a same-side crosser is already a single member. Nothing downstream can
undo it. Downstream of `cm.isolated()` there is exactly **one** merge in either
detector: `cathode_connect` in stage 4 (`pdhd:616`, `protodunevd:704`), preceded by
`switch_scope` (a split, already carve-capable). That is a strictly simpler
topology than SBND, where `isolated` runs per-APA and *all* cross-APA merging is
downstream.

Two caveats specific to these detectors:

- **No consumer exists yet.** Neither `examine_bundles` nor `unmerge_bundle` nor
  any tagger is active in `pdhd/clus.jsonnet` or `protodunevd/clus.jsonnet` (the
  only active use is `cm.isolated()`; the per-group `examine_bundles` is
  deliberately disabled, 2026-06-14). So defect D2 does not exist there — the
  isolated array is never overwritten — and the fix is **representation-only**:
  correct to land, but there is **no physics effect to validate**, and no gate
  will show anything beyond the array contents. Do not claim otherwise.
- **Byte-identicality needs a save knob.** Adding a perblob array changes the
  pctree tarball, so the write must be gated per detector on the
  `save_real_cluster_id` pattern (doc 38) to keep knob-off output identical.
- **Ident collision applies here too**, with 2 drift groups instead of N APAs:
  group-0 and group-1 idents collide, so the assoc id must encode the scope
  (§4a).

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
5. Re-run doc 50's `scripts/analysis/stm/stm_main_connectivity.py`: the headline "26 % of fits stray
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
  merged evt18 cluster 80 is itself the bug — run `scripts/analysis/stm/stm_merge_attribution.py` on
  that event to decide.
- Stage counts assume the SBND per-APA + all-APA layout. PDHD/PDVD run the same
  `clustering_isolated`, so Stage 0's diff will not be SBND-only — those
  detectors' gates have to run too before anything defaults on.
- Whether uBooNE's mains are as fragmented as ours is still unmeasured
  (doc 50 §6); that number would tell us how much of this is a porting artifact
  versus a real detector difference.

## 8. Implementation (landed 2026-07-25, default OFF)

### 8.1 Code map

| stage | file:line | what |
|---|---|---|
| 1 | `clus/src/clustering_isolated.cxx` | `save_assoc_id` knob; the merge at the end of the pass becomes `merge_clusters(g, live_grouping, "isolated", "perblob", "assoc_cluster_id", false, "assoc_cluster_main")` when on, and the historical `merge_clusters(g, live_grouping, "isolated")` when off |
| 2 | `clus/src/ClusteringFuncs.cxx` (`merge_clusters`) | a fixed registry `carry_pairs = {{"assoc_cluster_id","assoc_cluster_main"}}` carried across **every** merge: main concatenated verbatim, id rebased per member into a fresh dense range, member without the arrays ⇒ one fresh id + `main = 1` |
| 2 | `clus/src/clustering_switch_scope.cxx` | the hardcoded two-name carve list became one `carry_anames[]` of four (defect D3), and the carve loops over it |
| 2 | `clus/src/MultiAlgBlobClustering.cxx` + `.h` | `save_assoc_cluster_id` knob: the `save_real_cluster_id` homogenization, repeated for the assoc pair, in its own loop so either can be saved alone |
| 3 | `clus/src/ClusteringUnmergeBundle.cxx` | `id_aname` / `main_aname` config (defaults `real_cluster_id` / `real_cluster_main`), threaded into `groups_from_provenance`; everything else — the union retain rule, the carve, the flag clearing, the no-proxy-fallback rule — is reused unchanged |

Config: `cfg/pgrapher/common/clus.jsonnet` `isolated(save_assoc_id=false)` and
`unmerge_bundle(id_aname=null, main_aname=null)`, both key-suppressed;
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` threads `save_assoc_id` /
`save_assoc_cluster_id` and adds the `unmerge_assoc` visitor after
`unmerge_bundle`. Runners: `run_ql_evt.sh -save-assoc` (`SBND_SAVE_ASSOC=1`),
`run_nusel_evt.sh -unmerge-assoc` (`SBND_UNMERGE_ASSOC=1`, implies `-unmerge`).

### 8.2 Deliberate choices worth remembering

- **The carry is a property of `merge_clusters`, not of any call site.** Of the 12
  call sites only `examine_bundles:153` passes provenance arguments;
  `cathode_connect` and the generic all-APA passes pass none. Per-call-site carry
  would have missed them and would be re-missed by the next new pass.
- **Roles are concatenated, never rewritten** (§4b). This is what keeps a cathode
  crosser whole while its delta rays still split off.
- **Ids are rebased per member**, ascending by original id, because per-APA /
  per-drift-group idents collide across scopes (§4a trap).
- **A member with no arrays is a main.** The same sentinel appears twice: in the
  carry, and in MABC's save-time homogenization.
- **`nmain == nb ⇒ no_split` is untouched**, so a cluster the grouping never
  merged is left exactly alone.

### 8.3 Knob-off gate — PASS

| check | result |
|---|---|
| `./build/clus/wcdoctest-clus` | 41/41 cases, 518/518 assertions, rc=0 |
| freshness proof (M1) | `local/lib/libWireCellClus.so` 14:00:58 > last source edit 13:59:49 |
| QL compiled config, knob off vs pre-change `HEAD` | **byte-identical**, 50675 B |
| PR compiled config, knob off vs pre-change `HEAD` | **byte-identical**, 250166 B |
| knob-off **output**, 30 events, both stages | **210/210 archives identical** (§9.1) |

Knob-on compiled-config proof: `save_assoc_id: true` appears **twice** (one
`ClusteringIsolated` per APA) and `save_assoc_cluster_id: true` **three** times
(both per-APA MABCs and the all-APA one -- see §9.3 for why the per-APA nodes
need it); the PR pipeline becomes

```
ClusteringSwitchScope:pr -> ClusteringUnmergeBundle:pr -> ClusteringUnmergeBundle:prassoc
  -> CreateSteinerGraph:pr -> MakeFiducialUtils:pr -> TaggerCheck{TGM,STM,FC}:pr
```

with `prassoc` carrying `id_aname=assoc_cluster_id`,
`main_aname=assoc_cluster_main`.

One bug was caught by the knob-on proof and not by the knob-off one: `per_apa()`
accepted `save_assoc_id` without forwarding it to `clus_per_face`, so the key
never reached `ClusteringIsolated`. Byte-identicality when off is blind to that
class of mistake — always run the on-side proof too (M6's cousin).

## 9. Results on the 30-event MCP2025C scan

**Repro:**
```bash
cd sbnd_xin
./scripts/runners/run_d52_campaign.sh both          # ~25 min; both arms under setarch x86_64 -R
python3 scripts/analysis/stm/d52_ab_report.py
python3 scripts/analysis/stm/stm_main_connectivity.py work-mcp10-d52on work-mcp1000-d52on work-mcp1000b-d52on
```
Tags: **`work-{mcp10,mcp1000,mcp1000b}-d52off`** and **`-d52on`**. Imaging is
symlinked from `-d49son`, never regenerated.

### 9.1 Knob-off gate — PASS

**210/210** comparisons identical over all 30 events: per event the three
`mabc-*.zip`, `pctree-evt<ID>.tar.gz`, `mabc-pr.zip`,
`pctree-pr-evt<ID>.tar.gz` (member-content hash) and `nusel-evt<ID>.tsv` (text).
Baselines are the products already on disk from the pre-change binary
(`work-*-mainreal` for Q/L, `work-*-d49son` for nusel).

One false alarm worth recording: the first attempt reported a pctree difference
of exactly one tensor, `lm_flag`. That was **my invocation**, not the code — the
baseline was produced with `-lm` and I had omitted it. Always diff the member
*names* before believing a hash difference.

### 9.2 Knob-on effect — the mechanism works

| quantity | value |
|---|---|
| clusters split by the isolated un-merge | **200** |
| associated pieces produced | **785** |
| blobs moved out of a main | **4544** |
| Q/L stage identical between arms | **90/90** ⇒ every delta below is attributable to the un-merge alone |
| retained mains left with ≤3 blobs | 2 of 200 (evt285185 c12 `41 → 1 + 8`, evt286329 c6 `5 → 2 + 1`) |

Both un-merges fire in the designed order, e.g. evt285185 cluster 21:
`272 → 264 + 8` (flash, outer) then `264 → 255 + 9` (isolated, inner).

Verdict movement (both arms `setarch -R`; the chain's ASLR noise floor is ±7 STM
tags, doc 49 §4a, so the STM number is *at* that floor even though ASLR was
pinned):

| verdict | off | on | delta |
|---|---|---|---|
| tgm | 122 | 109 | **−13** |
| stm | 44 | 37 | **−7** |
| fc | 51 | 59 | **+8** |

329 bundles in both arms — the un-merge does not change the bundle count, only
what each bundle's main contains.

### 9.3 A bug the knob-off gate structurally could not catch

The first knob-on run split **nothing** — `unmerged 0 main cluster(s)`, with no
warning, because `groups_from_provenance` saw `nmain == nb` (every blob marked
main) and correctly returned `no_split`. Cause: the assoc pair is written at
**per-APA** scope, and `MultiAlgBlobClustering`'s homogenization was wired only
on the all-APA node. TensorDM drops a local-PC key absent from the first-seen
node (doc 38's gotcha), so the real arrays died at the per-APA → all-APA handoff
and the all-APA save-time fill-in then made every blob a main.

Fixing that exposed a second, sharper problem: `Dataset::append`
(`util/src/PointCloudDataset.cxx:261`) **raises** when an incoming per-blob
dataset lacks a key the target has, and the whole job died with
`missing keys in append: 3 missing: isolated real_cluster_id real_cluster_main`.
The assoc pair exists *before* the all-APA `switch_scope`, so its carve puts it
on the out-of-volume shards too — and `examine_bundles` then skips those (not in
scope), so they never receive `isolated`. The invariant is therefore not "gate on
`isolated`" (what the flash pair does) but **"any cluster that has a `perblob` PC
must have the full key set"**, which is what the code now enforces.

Neither failure is visible to a knob-off gate, and neither is visible to a
compiled-config proof. The only thing that caught them was running knob-on and
reading the visitor's own count.

### 9.4 What did NOT improve, and it matters

Doc 52 §5 step 5 set the success criterion: doc 50's "26 % of fits stray >5 cm
from their own charge" should fall. It falls, modestly:

| doc 50 metric | off | on |
|---|---|---|
| STM-fitted mains | 128 | 125 |
| not connected post-un-merge (>1 component at 5 cm) | 88 (69 %) | 79 (**63 %**) |
| fitted point >5 cm from own charge | 33 (26 %) | 29 (**23 %**) |
| trajectory END inside a detached clump | 36 (28 %) | 32 (**26 %**) |

**And the two showcase clumps are still there.** Measured directly:

| event | cluster | components, off | components, on | clump gap | fit ends in clump? |
|---|---|---|---|---|---|
| 285185 | 21 | 1163 + 8 pts | 1128 + **8** pts | 18.27 cm | no (max stray 1.52 cm) |
| 284657 | 27 | 279 + 21 pts | 245 + **21** pts | 22.55 cm | **yes**, 1 of 2 |

The clump is verifiably still inside cluster 21 after both un-merges (8 points
within 4 cm of its centroid, all `cluster_id == 21`). So those blobs carry
`assoc_cluster_main != 0` — they belong to the **main** pre-merge member, not to
an associated one.

That **contradicts doc 51 §2's attribution**, which named `ClusteringIsolated`
(tr14) as the pass that joined body and clump for exactly these two events. Both
cannot be right. Either the attribution tool mis-identified the joining pass (its
majority-id coordinate matching is approximate, and cluster ids are renumbered
every step), or the clump is inside the same pre-merge member because an earlier
pass merged it and `clustering_isolated` only re-merged the pair. Resolving it is
the next step and it is cheap: re-run `run_ql_evt.sh -trace-bee -save-assoc` on
these two events and check the assoc marking layer by layer.

Read plainly: **the provenance mechanism is correct and does what it was designed
to do (200 splits, 785 pieces, gate-clean), but the isolated grouping is not the
dominant source of fragmented STM mains.** 63 % of fitted mains are still
multi-component, which points at §4's residual — single pre-merge members that
are internally disconnected, deliberately outside the un-merge's remit — as the
bigger population.

### 9.5 Requirement 2 is satisfied for the isolated merge only

The owner's second requirement was that cathode crossers stay merged. For the
isolated grouping that holds by the union rule. For the **flash** merge it does
not, and this landing does not change that: evt288287 cluster 13 (PCA
|cos| 0.997, §4a) is still split by the default-ON flash un-merge,
`526 → 445 + 81`, and the isolated un-merge then trims 3 more blobs
(`445 → 442 + 3`). Its bundle moved from `tgm/stm/fc = 0/0/0` to `0/0/1`.

**§4a claimed Stage 2's carry would fix this. That claim was wrong and is
withdrawn.** Two independent reasons:

1. `assoc_cluster_main` cannot discriminate. It is ~all-ones by construction —
   7614 of 7849 blobs in evt285185 — so "never demote a member that is itself an
   assoc-main" would protect essentially every member and disable the flash
   un-merge entirely.
2. More fundamentally, the crosser's two halves are two *distinct* flash-merge
   members **precisely because `cathode_connect` declined to join them**. No
   provenance array can recover a merge decision that was never made.

So the flash-merge crosser case needs either a `cathode_connect` that catches
these (a tuning question, with its own gate) or an explicit geometric guard
inside the un-merge. It is a separate item, and it affects production today.

## 10. REGRESSION: the un-merge strips near-touching track charge (owner report)

**Status: knob-on is NET HARMFUL as it stands. Production is unaffected — every
knob is default OFF and nothing was shipped on.** Reported by the owner from the
scan: evt284349 bundle grp 6 (t = −238.770 µs, main 8) and grp 13
(t = 592.196 µs, main 10) were TGM in the previous scan and now fall through to
the STM fit.

**Repro:**
```bash
cd sbnd_xin
python3 scripts/analysis/stm/d52_ab_report.py                     # verdict deltas
grep "check_tgm: cluster 8 " work-mcp10-d52on/nusel_evt284349/wct_nusel_evt284349.log
grep "prassoc> cluster 8:"   work-mcp10-d52on/nusel_evt284349/wct_nusel_evt284349.log
```

### 10.1 It is a real regression, and the mechanism is exact

All **14** TGM losses (and the 1 gain) are clusters the isolated un-merge split.
`len_main_cm` does not change — the main loses no *length*, only charge — so the
verdict flip is not a geometry change. **10 of the 14** name one guard in the log:

```
check_tgm: cluster 8 CASE-B pair (0,3) rejected: rescued end,
           straight chord 286.1 cm has an unsupported run > 30.0 cm
```

That is `rescue_chord_check` (docs 29-39). It walks the straight chord between two
candidate endpoints and requires charge support along it — **charge belonging to
the cluster it is given**. The un-merge takes charge out of the main, so the walk
now finds gaps it did not find before, and every endpoint pair is rejected.

| event | cid | len_cm | blobs main→main | logged rejection |
|---|---|---|---|---|
| 284349 | 10 | 310.6 | 324 → 311 (−13) | chord-unsupported |
| 284349 | 8 | 424.9 | 682 → 649 (−33) | chord-unsupported |
| 284657 | 22 | 313.8 | 562 → 476 (−86) | chord-unsupported |
| 285185 | 15 | 476.8 | 1123 → 1058 (−65) | (none logged) |
| 286021 | 10 | 114.6 | 317 → 283 (−34) | (none logged) |
| 286197 | 10 | 451.4 | 890 → 852 (−38) | chord-unsupported |
| 286329 | 18 | 217.4 | 385 → 321 (−64) | (none logged) |
| 286527 | 11 | 238.4 | 504 → 455 (−49) | chord-unsupported |
| 286527 | 20 | 292.3 | 599 → 471 (−128) | chord-unsupported |
| 286681 | 15 | 424.9 | 725 → 676 (−49) | (none logged) |
| 286681 | 16 | 459.4 | 593 → 543 (−50) | chord-unsupported |
| 288067 | 13 | 142.5 | 1049 → 905 (−144) | chord-unsupported |
| 288397 | 11 | 477.3 | 1313 → 1266 (−47) | chord-unsupported |
| 290135 | 15 | 129.7 | 189 → 181 (−8) | chord-unsupported |

### 10.2 Root cause: "associated" conflates two populations

Distance from every removed point to the retained main, over all 200 split
clusters (22455 points):

| within | fraction |
|---|---|
| 1 cm | **18.2 %** |
| 2 cm | 31.1 % |
| 3 cm | **40.9 %** |
| 5 cm | 55.4 % |
| 10 cm | 74.1 % |
| 20 cm | 90.3 % |

median **4.07 cm**, 90th percentile 19.77 cm.

So **~41 % of the stripped charge is within 3 cm of the main** and 18 % is within
1 cm. That is not a detached clump — it is track charge broken up by
signal-processing / dead-channel inefficiency, precisely what the chain's
gap-jumping passes exist to bridge. Per-cluster examples:
evt284349 c8 median 1.71 cm (75 % within 3 cm), evt288067 c13 median 2.61 cm with
**max 5.31 cm** (the whole "associated" set is touching the main).

The cause is in `clustering_isolated` itself: "small" (< `length_cut`, SBND 15 cm)
merged to the nearest big cluster within `small_big_dis_cut = 80 cm`, **with no gap
test at all** (doc 51 §3). One label therefore covers both
(a) genuinely detached clumps 18-22 cm out — doc 50's problem, the thing we set
out to remove — and (b) fragments 0.3-3 cm out that are the same particle. The
provenance is faithful, so the un-merge removes **both**.

Combined with §9.4 this is decisive: what the un-merge removes is mostly (b), and
the (a) clumps of the two showcase events are **not** marked associated at all. The
knob as it stands pays a real TGM cost for the wrong population.

### 10.3 Why the prototype does not have this problem

`wire-cell-prod-stm.cxx:816-828` hands `check_stm(main_cluster,
additional_clusters, ...)` — the companions travel *with* the main and
`check_other_clusters()` uses them. Our chord/rescue guards
(`rescue_chord_check`, `tgm_chord_charge`, `component_rescue`,
`main_component_pairs`) were invented for the *merged* main (docs 29-39) and only
ever look at the one cluster they are handed. Splitting the main without also
handing over the companions is faithful in representation and unfaithful in use.

### 10.4 Options, for the owner to choose

1. **Give the guards the companions** (prototype-faithful, my recommendation).
   The chord/rescue charge-support tests search the main **plus** its associated
   companions (`flag_associated_cluster` + shared `matched_flash_gid`, which the
   un-merge already sets). Restores the 14 TGM tags without giving up the
   un-merge, and is what the prototype does. Touches TGM, so it needs its own
   knob and gate.
2. **Only split what is actually detached** — a `min_gap` on the un-merge
   (split a companion only when its closest approach to the retained main
   exceeds, say, 5 cm). One knob, cheap, and it cuts the stripped-charge
   population roughly in half by the table above. It does *not* fix §9.4, since
   the showcase clumps are not marked associated.
3. **Both** — 2 to stop stripping same-particle charge, 1 so that whatever is
   still split cannot silently break a charge-support test.

Not recommended: reverting Stages 1-3. The provenance mechanism is gate-clean and
correct; what is wrong is the un-merge's *criterion* (2) and the guards'
*inputs* (1).

### 10.5 Still open from §9.4

The two showcase clumps are marked `assoc_cluster_main != 0`, contradicting
doc 51 §2's attribution to `ClusteringIsolated`. Until that is resolved we do not
actually know which pass creates the doc-50 population, and therefore cannot say
whether any un-merge can remove it. `-trace-bee -save-assoc` on evt285185 and
evt284657 answers it.

## 11. The real bug: the provenance is corrupted in the per-APA → all-APA handoff

**Owner was right and §10's reading was wrong.** §10 concluded the un-merge was
stripping same-particle charge because `clustering_isolated` labels near-touching
fragments "associated". That is not what happened. The owner's read of the two
screenshots — *"it split a piece that is connected to the main cluster; the
isolated piece was NOT un-merged but the tip of the main was removed"* — is
exactly right, and it is a **data-corruption bug**, not a criterion problem.

**Repro:**
```bash
cd sbnd_xin
# per-step per-APA layers: where do the pieces live before the handoff?
SBND_WORK_ROOT=$PWD/work-mcp10-d52trace ./run_ql_evt.sh data 1 \
    -save-pctree -save-rcid -lm -save-assoc -trace-bee
# then track the three pieces through tr00..tr15 of mabc-apa1-face0.zip
```

### 11.1 Ground truth, from the per-step trace (evt284349, apa1)

| piece | tr00 … tr13 | tr14 `ClusteringIsolated` |
|---|---|---|
| trunk (130, −15, 478) | cluster 16→15→14→22→20→**19** | → 7 |
| **tip** (136, +7, 498) | cluster 16→15→14→22→20→**19** | → 7 |
| isolated piece (146, −8.5, 492) | cluster 17→16→15→13→12→**11** | → 7 |

The tip is in the **same cluster as the trunk from the very first step**. Only
the isolated piece is a separate cluster, and `ClusteringIsolated` is what joins
it. So the correct marking is trunk+tip = main, isolated piece = associated.

### 11.2 The write is correct; the array arrives corrupted

Self-check inserted immediately after `merge_clusters` in `clustering_isolated`
(every group must be a compact set of blobs — it is one pre-merge cluster):

```
[isoself] cluster 10 group 11 main=0 n=5 centroid=(53.72,-8.64,492.81) rmax=0.74cm   <- the isolated piece
[isoself] cluster 10 group 19 main=1 n=311 centroid=(19.64,-84.64,402.07) rmax=171.51cm
```

Correct: the 5-blob isolated piece is `main=0`. But the same check at the **entry
of the all-APA `switch_scope`**, before that pass does anything:

```
[ssentry] cluster  4 group  6 main=0 n=4  rmax=  0.49 cm     <- still fine
[ssentry] cluster 13 group  6 main=0 n=3  rmax= 38.92 cm     <- scrambled
[ssentry] cluster 13 group  8 main=0 n=4  rmax= 78.95 cm
[ssentry] cluster 13 group  9 main=0 n=10 rmax=260.38 cm
[ssentry] cluster 14 group 10 main=0 n=4  rmax= 37.81 cm
```

Early clusters are intact, later ones are progressively wrong. At the time this
was read as the signature of an accumulating row offset in the TensorDM
concatenate/re-split. **That guess was wrong** — §12 pins the actual mechanism
(QLMatching's decompose/recompose permutes the blobs, not the serializer; the
serialize→deserialize round trip is row-faithful, proven by the §12.1 checker's
`rtrip` checkpoints).

Everything else was cleared by direct test:

| suspect | verdict |
|---|---|
| `clustering_isolated`'s write | **correct** (§11.2 self-check) |
| `merge_clusters`' carry ordering | **not the cause** — re-keying the placement by blob pointer instead of row position produced byte-identical output |
| `clustering_switch_scope`'s carve | **not the cause** — already corrupt at its entry |
| the analysis tooling | **not the cause** — `real_cluster_main`'s 7 zeros have max-radius 1.0 cm and `isolated` value 3 is the isolated piece at (146.3, −8.5, 492.1), both read with the same node walk |

Why nothing caught this before: `orig_id` / `orig_main` (the flash pair) are
**constant within a member**, so any permutation of a member's rows is a no-op
for them. The assoc pair is the first per-blob array whose value *varies within a
member*, so it is the first to expose the handoff. `TaggerCheckTGM`'s
`main_component_mode="real"` reads an all-ones array after a correct un-merge
(§4), so it could not see it either.

### 11.3 Consequence for §9 and §10 — both are void

- §9.2's "200 clusters split, 785 pieces" counted **corrupted** splits.
- §9.4's "the showcase clumps are marked main, contradicting doc 51" was an
  artifact: doc 51's attribution to `ClusteringIsolated` is **confirmed correct**
  by the trace above.
- §10's whole "associated conflates two populations" analysis measured the
  corruption, not the criterion. The 41 %-within-3 cm number describes which
  blobs the *scrambled* array happened to point at.
- The 14 TGM losses are collateral of the corruption, not of a design choice.

### 11.4 State of the code, and what is next — SUPERSEDED by §12

`min_gap` was shipped here as a stop-gap and is now **REMOVED** (owner
directive): it masked the corruption rather than repairing it. The blob-node
(one-row local PC per blob) repair floated below was also **not** taken — the
owner judged it likely too slow, and §12 shows the cluster-level N-row array is
fine once the one operation that breaks its alignment is fixed.

## 12. Root cause found and fixed: QLMatching's decompose/recompose permutes the blobs out from under the arrays

**Repro (all numbers below):**
```bash
cd sbnd_xin
mkdir -p work-mcp10-d52chk && for d in work-mcp10-d49son/evt*; do \
    ln -sfn "$(readlink -f "$d")" "work-mcp10-d52chk/$(basename "$d")"; done
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
SBND_WORK_ROOT=$PWD/work-mcp10-d52chk WCT_PROV_CHECK=1 \
    setarch x86_64 -R ./run_ql_evt.sh data 1 -save-pctree -save-rcid -lm -save-assoc
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
SBND_WORK_ROOT=$PWD/work-mcp10-d52chk WCT_PROV_CHECK=1 \
    setarch x86_64 -R ./run_nusel_evt.sh data 1 -chord -rescue -rescue-chord \
    -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc
grep provchk work-mcp10-d52chk/ql_evt284349/*.log
```

### 12.1 The instrument: WCT_PROV_CHECK

New diagnostic-only checker `check_perblob_provenance()`
(`clus/src/prov_check.cxx`, declared in `ClusteringFuncs.h`), enabled ONLY by
env `WCT_PROV_CHECK=1` (a fast no-op otherwise). Per cluster with a "perblob"
PC it validates (a) every array has exactly `nchildren` rows and (b) every
`assoc_cluster_main==0` group is spatially compact (an associated group is one
small pre-merge cluster, ≤ 20 cm by construction; rmax > 25 cm from its own
centroid can only be rows attached to the wrong blobs). It runs at every
boundary: MABC load, after **every** visitor, at save plus a
serialize→deserialize **round-trip self-test**, and in PointTreeMerging
(inputs / merged / round-trip). 107 checkpoints per QL run.

### 12.2 The bracketing (evt284349, corrupt binary)

| checkpoint | verdict |
|---|---|
| per-face apa1, `post:ClusteringIsolated` … `post:ClusteringExamineBundles` | clean (worst rmax 5.6 cm — a real compact group) |
| per-face apa1/apa0 `save` AND `rtrip` (serialize→deserialize self-test) | **clean** — TensorDM is row-faithful; §11.2's handoff hypothesis is dead |
| all-APA MABC `load` (the first look after QLMatching) | **4 problems, rmax 38.9 / 79.0 / 260.4 / 37.8 cm** |

The corruption is born inside the one node between those two points:
**QLMatching**.

### 12.3 The mechanism

`QLMatching::decompose_cluster_groups()` splits every grouped cluster on its
"isolated" cc array via `Grouping::separate(cluster, cc)` — a matching-internal
device so bundle geometry anchors on the main. `recompose_cluster_groups()`
merges the pieces back (`grouping->merge` → `take_children`). The round trip
**permutes the blob children**: `separate()` keeps the `cc<0` blobs in place
(original relative order) and `merge()` appends each split's blobs at the END,
in ascending-gid order. Meanwhile the main's node-local "perblob" arrays are
untouched — full length, ORIGINAL row order. Blob count is restored, so every
length check passes; but row *i* no longer describes child *i*.

Why every earlier probe missed it:

- "isolated" is misaligned too, but the all-APA `examine_bundles` **rewrites**
  it from a fresh connected-components pass — so it always *reads* clean
  downstream. (It is consumed BEFORE the rewrite by that visitor's own
  main-overlap vote, and by `decompose` itself — see §12.6.)
- `real_cluster_*` is written fresh at the all-APA flash merge — after the
  permutation. Clean by luck of ordering.
- The assoc pair is written per-face, BEFORE QLMatching — the first array that
  actually has to *survive* the permutation, and the first whose value varies
  within a member, making the scramble visible.
- The accumulating-offset look of §11.2 was an artifact of which clusters had
  more sub-groups, not a serializer offset.

### 12.4 The fix: `realign_perblob` (QLMatching knob, default OFF)

`recompose_cluster_groups()` now reorders each split main's **whole "perblob"
dataset** by the same permutation the blobs underwent (`[rows cc<0, original
order] ++ per gid ascending [rows cc==gid]` — `Dataset::subset(rows)`, all keys
at once), restoring row *i* == child *i*. The cc used at `separate()` is saved
at decompose (`ApaRun::decompose_cc`). Knob `realign_perblob`:

- C++ default **false** ⇒ the historical (misaligned) behavior bit-for-bit.
- jsonnet: `qlmatching.jsonnet` `matching()` / `matching_joint()` arg with the
  key-suppression idiom; threaded from `wct-clus-matching-perevt.jsonnet`'s
  `save_assoc` TLA (the assoc provenance is unusable without it), in both the
  joint and per-APA+PointTreeMerging graph modes.

This fixes the origin: the cluster-level N-row array representation is fine —
the ONE operation that permuted blobs without their arrays now moves both
together. No blob-node-per-row representation needed.

`min_gap` (§11.4) is removed everywhere: `ClusteringUnmergeBundle`,
`cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/clus.jsonnet`
(`unmerge_assoc_min_gap` gone).

### 12.5 Verification (evt284349)

- **Checker**: with the fix on, **0 problems at all 107 checkpoints** of the QL
  run (worst assoc-group rmax 5.6 cm = a genuine compact group) and 0 through
  the whole PR run.
- **The owner's screenshot case** (`mabc-pr.zip` clustering-global layer, raw
  coords): tip (135.3, 6.9, 498.2) → stays in main cluster 10; trunk (130.3,
  −14.3, 477.8) → main cluster 10; isolated piece (146.9, −8.6, 492.5) → its
  own associated cluster. Exactly the doc-51 expectation — issue 1 fixed.
- **The TGM regression** (§10): `bundle grp 6 t=-238.770 us main 8` and
  `bundle grp 13 t=592.196 us main 10` are both **TGM=1 again** — with NO
  min_gap. The un-merge now removes only genuinely-detached pieces, so the
  chord-support walk keeps its endpoints — issue 2 fixed.
- Gates: off-arm compiled configs byte-identical to the pre-change baseline
  (QL and PR jobs, `wcsonnet` diff vs HEAD worktrees); knob-on emits
  `realign_perblob: true` on every QLMatching node (compiled-config proof);
  `wcdoctest-clus` 518/518, `wcdoctest-match` 36/36. 30-event A/B redo (tag
  `d52roff`/`d52ron`, `./scripts/runners/run_d52_campaign.sh both d52r`) — §12.7.

### 12.6 Pre-existing bug, NOT fixed here (reported, default path unchanged)

With `realign_perblob` off — i.e. **today's production** — the stale "isolated"
rows are consumed in two places before the all-APA rewrite:

1. `clustering_examine_bundles`' main-overlap vote (which new component
   inherits "main") reads the scrambled `-1` rows;
2. a hypothetical second decompose would split on scrambled rows.

Both predate doc 52 and are independent of `save_assoc`. Turning
`realign_perblob` on realigns "isolated" as well, which is why the knob cannot
default on (it would change production output). Flipping it on for production
needs its own validation round.

### 12.7 30-event A/B redo (tags d52roff / d52ron) — replaces the VOID §9/§10 numbers

**Repro:** `./scripts/runners/run_d52_campaign.sh both d52r` then
`python3 scripts/analysis/stm/d52_ab_report.py --off-tag d52roff --on-tag d52ron`
(both arms `setarch x86_64 -R`, imaging symlinked from `work-*-d49son`).

- **GATE (knob-off): 210/210 archive comparisons identical → PASS** — the
  realign fix, the min_gap removal and the checker leave the legacy path
  byte-identical (member-content hashes vs `work-*-mainreal` / `work-*-d49son`).
- **Q/L stage, on vs off: 90/90 identical** → every nusel delta is attributable
  to the un-merge (realign changes no Q/L physics; the assoc arrays are the
  only addition).
- **Effect:** 200 clusters split into 786 associated pieces (4677 blobs moved);
  **0 of 200** splits left a degenerate main (≤ 3 blobs).
- **Verdicts (30 events):**

  | verdict | off | on | delta |
  |---|---|---|---|
  | tgm | 122 | 121 | **−1** |
  | stm | 44 | 52 | **+8** |
  | fc  | 51 | 52 | +1 |

  Compare the corrupt round (§10): tgm −13, stm −7. The TGM regression is
  gone — with NO min_gap, the un-merge now removes only genuinely detached
  pieces, so `rescue_chord_check` keeps its endpoint support.  The +8 STM is
  the intended doc-50 effect: fits that used to walk into a detached clump
  (and fail the stopping-muon shape test on garbage dQ/dx) now end on the real
  track.  The single lost TGM is a hand-scan follow-up, not a blocker.
- Hand-scan display: `serve_nusel_scan.sh 5011 --tag d52ron --charge-src pr
  --prev ../work-*-d49son:d49son ../work-*-d52ron` (port 5011, amber = verdict
  changed vs the d49son scan).

### 12.8 Default flipped ON (owner decision, 2026-07-25)

Owner: *"This fix once demonstrated should be on naturally"* and *"we should
definitely fix the pre-existing bug you reported in 12.6."*  Both point at the
same action, taken here: **`m_realign_perblob` C++ default = true.**

Why this is safe per detector:

- **PDHD / PDVD: structural no-op.** `decompose_cluster_groups()` only splits a
  cluster whose "isolated" array contains a `-1` main marker, and only
  `examine_bundles` writes `-1` — which is DISABLED in both configs
  (`pdhd/clus.jsonnet:483`, `protodunevd/clus.jsonnet:529`; they run the bare
  `cm.isolated()`, whose cc is 0,1,2,…).  No decompose ⇒ `others.empty()` ⇒ the
  realign block is never reached.  The flip protects them for the day they
  adopt the SBND-style chain.
- **SBND: the intended change.** The recomposed "isolated" rows are now
  aligned, so the all-APA `examine_bundles` main-overlap vote (§12.6 consumer
  1) reads true rows.  Production output changes by design — quantified by the
  production-delta arm below.

Knob plumbing after the flip: jsonnet arg is a TRISTATE (`null` = inherit the
C++ default = on; explicit `false` reproduces the pre-fix behavior).  Runner:
`run_ql_evt.sh -no-realign` / `SBND_REALIGN=0` — A/B archaeology only.
Compiled-config proof: default emits no key; `-no-realign` emits
`"realign_perblob": false` on both QLMatching nodes.

**Production-delta arm (tag d52rp):** `./scripts/runners/run_d52_campaign.sh off d52rp` with
the flipped default — same flags as d52roff, the ONLY difference is realign
(work dirs `work-*-d52rpoff`; comparison script: session scratch
`d52rp_delta.py`, member-content hashes + per-main verdict tuples).

Result over the 30 events, d52rpoff (realign on) vs d52roff (old behavior):

- **archives: 120/120 member-content IDENTICAL** — CORRECTED §13.2: the
  first pass of `d52rp_delta.py` compared `hash_archive.py`'s whole output
  line, which embeds the FILE PATH, so all 120 pairs "differed" vacuously.
  Re-run comparing only the hash field: 0/120 differ.  This is stronger than
  first reported: without `-save-assoc` the assoc pair is never written, and
  the one array realign touches at recompose ("isolated") is rewritten from
  scratch by the all-APA examine_bundles flash merge before anything is
  saved — so the flip is fully byte-identical on the off-arm manifest.
- **nusel verdict tables: 30/30 row-for-row IDENTICAL** — per-main
  (tgm, stm, fc, stmfit) tuples unchanged on every bundle; totals
  tgm 121 / stm 43 / fc 50 in BOTH arms (per-main de-duplicated count).

So on this manifest the default flip is byte-identical end to end.  The
verdict-visible effect of correct provenance arrives only through the opt-in
assoc chain (`-save-assoc` / `-unmerge-assoc`, §12.7).  Doctests re-run
after the flip: clus 518/518, match 36/36.

## 13. Codebase audit: every other place that could repeat this bug (2026-07-25)

Owner directive: *"check clustering, PR ... I want to make sure this would not
show up again."*  Audited every site in the toolkit that can change a
cluster's blob-children order or count (`separate()`, `merge()`,
`take_children`, `sort_children`) against every writer/reader of the
`"perblob"` N-row Dataset.

**Repro (audit sweep):**
```
grep -rn "separate(\|take_children\|->merge(\|sort_children" clus/src match/src img/src
grep -rn "perblob" clus/src match/src
```
plus manual reads of each hit.  Full classification table now lives in the
toolkit at `clus/docs/perblob_invariant.md`; summary:

* **Two raw `merge()` calls exist in the whole tree**: QLMatching (fixed,
  §12.4) and `RetileCluster::mutate` (`clus/src/retile_cluster.cxx:584-714`)
  — the latter was an exact replica of the §12 bug: separate on the
  `isolated` cc, merge back (children re-appended in ascending-gid order),
  fresh `isolated` written for the new order, **all other perblob arrays
  left in the old row order** on the original cluster, which stays in the
  live grouping.  Latent, not live: its only route (`cm.retile`) is commented
  out in every config; production steiner is `CreateSteinerGraph` +
  `ImproveCluster_2`, which retile a scratch COPY and never mutate the source
  cluster's child set (verified — so `TaggerCheckTGM` mode "real" reads
  `real_cluster_main` against an unpermuted cluster).
* **Two more silent-permutation sites** in `clus/src/clustering_neutrino.cxx`
  (cluster1 ~415→450, cluster2 ~593→627): `Separate_2` gives every blob a
  non-negative component id, `separate(remove=false)` empties the cluster,
  then a total take-back re-appends children grouped by component id —
  length-preserving permutation.  Dormant: `neutrino` runs before
  `isolated` (the first perblob writer) in SBND (clus.jsonnet 230 vs 240),
  PDHD (481 vs 482), PDVD (527 vs 528), and is absent from the PR pipeline
  and the uBooNE chain.
* **Stale-length/loss sites** (would trip the WCT_PROV_CHECK length check,
  or lose provenance): `ClusteringRecoveringBundle` (cfg-defined, never
  instantiated), `GroupingHelper::process_groupings_helper` (dead code),
  the absorb/pool sites in `clustering_separate.cxx`, and
  `examine_x_boundary` / `protect_overclustering` (remove=true destroys the
  arrays) — all run pre-`isolated` in the QL chain only.
* **Safe by construction**: the 12 graph-merge visitors via
  `merge_clusters()` (pointer-keyed carry, Stage 2), `switch_scope` (carve),
  `ClusteringUnmergeBundle` (subset), QLMatching (fixed);
  `sort_children` has zero callers.
* Side finding (reported, not fixed): `protect_overclustering` would destroy
  a zero-point blob (groupid stays −1) together with the original cluster —
  silent blob loss, independent of perblob.

### 13.1 Hardening shipped (owner-approved options 1+3)

New shared helper `realign_perblob_after_regroup(cluster, cc)`
(`clus/inc/WireCellClus/ClusteringFuncs.h`, `clus/src/prov_check.cxx`):
re-applies the separate/merge-back permutation (kept `cc<0` rows first, then
ascending gid) to the WHOLE perblob Dataset via `Dataset::subset`; fail-open
no-op unless the Dataset exists and both its major size and the child count
equal `cc.size()`.  Applied at the three latent sites:

* `retile_cluster.cxx` — after the `merge()` restore, before the fresh
  `isolated` write;
* `clustering_neutrino.cxx` — after both take-back loops;
* `ClusteringRecoveringBundle.cxx` — carve instead (pieces leave for good):
  snapshot before `separate()`, `subset(rows)` per part, the
  ClusteringUnmergeBundle idiom.

No knobs: every call is a guarded no-op wherever the array does not exist,
and today it does not exist on any of these paths in any production config
(retile unreachable, recovering-bundle not instantiated, neutrino
pre-`isolated` everywhere).  Docs: `clus/docs/perblob_invariant.md` (rules +
audit table) and a pointer entry at the end of
`clus/docs/porting/porting_dictionary.md`.

### 13.2 Verification

* `wcbuild` rc=0; freshness proof (`local/lib/libWireCellClus.so` 17:06 >
  last source edit 17:04); `wcdoctest-clus` 518/518.
* Gate: 30-event off-arm redo under fresh tag `d53`
  (`./scripts/runners/run_d52_campaign.sh off d53`, `setarch x86_64 -R`), compared with
  member-content hashes against the current production baseline
  `work-*-d52rpoff` (realign_perblob default ON).  **GATE PASS: 120/120
  archives identical, 30/30 nusel verdict tables row-for-row identical
  (totals tgm 121 / stm 43 / fc 50 both arms).**  Comparison script:
  session scratch `d53_gate.py` (hash field only — see the §12.8
  correction: never compare `hash_archive.py`'s path-bearing full line).
* The same corrected comparison re-run on §12.8's d52rpoff-vs-d52roff pair
  showed those 120 archives are ALSO identical — the earlier "120/120
  differ" was an artifact of the path-bearing hash line, corrected above.

### 13.2a Tag collision and repair of the mcp10 third (2026-07-25 evening)

A concurrent session independently chose the tag `d53off` and wrote
`work-mcp10-d53off` starting 17:05, contesting the same directory this
section's gate read at ~17:29.  The dir now holds that session's run (30/40
archives differ vs d52rpoff — its rcid re-stamp signature), so the mcp10
third of the §13.2 PASS cannot be attributed to the §13 binary with
certainty and is VOID.  The mcp1000/mcp1000b thirds (80 archives) were
never contested and stand.  **Repair**: a frozen git worktree at the §13
commit `5e69191f` re-ran the campaign's mcp10 block under the fresh tag
`d53r` (`work-mcp10-d53r`, imaging symlinked from d49son, `setarch x86_64
-R`): **0/40 archives differ vs d52rpoff** — the §13.2 claim is restored,
now 40/40 + 80/80 = 120/120 on uncontested runs.  Process rule adopted:
`ls -d work-*` before choosing any tag.

### 13.3 Option 2 shipped: the Grouping primitives enforce the invariant

Owner directive: *"you can carefully do option 2, please pay attention to
running speed. And careful validation."*

**Design** (toolkit `clus/docs/perblob_invariant.md` has the full rules):

* `Grouping::separate()` snapshots the cluster's "perblob" Dataset and
  carves it across the survivor (kept cc<0 rows; entry erased when empty)
  and every split (its own rows); stale-length input is dropped with a
  warning.
* New `Grouping::merge()` overloads (map + iterator, HIDING the
  NaryTree::FacadeParent merges) concatenate the parts' Datasets onto the
  target in adoption order; impossible concatenations (stale rows, a
  child-bearing part without a Dataset, mismatched keys) drop the target's
  Dataset with a warning.  cc return unchanged.
* Dependent sites: QLMatching recompose's inline realign + its decompose_cc
  bookkeeping REMOVED (the primitives produce byte-identical output —
  kept-rows-then-ascending-gid; `realign_perblob` knob now vestigial, still
  parsed, false can no longer reproduce the corruption);
  `retile_cluster.cxx` realign call removed; `clustering_neutrino.cxx`
  take-back loops replaced by `Grouping::merge` (same op sequence);
  `clustering_switch_scope.cxx` erases the auto-carved Dataset per part
  before re-attaching its 4-array carry list (preserving its deliberate
  "isolated" drop byte-for-byte).  UnmergeBundle/RecoveringBundle keep
  their carves (defense in depth, same values).
* CORRECTION to §13/§13.1: ClusteringRecoveringBundle IS instantiated —
  qlport `uboone-mabc.jsonnet:1247` (the audit had only grepped
  `cfg/pgrapher/experiment/`).  It writes "isolated" itself (main = -1) and
  separates on it, so the uBooNE chain also exercises decompose/recompose in
  QLMatching.  Hence the uBooNE gate below.

**Validation** (all runs from FROZEN git worktrees with private install
prefixes and PATH/LD_LIBRARY_PATH redirection — the shared local/lib was
never rebuilt, isolating from the concurrently-evolving working tree;
plugin origin proven from /proc/<pid>/maps; cfg tree content hash identical
before and after the whole gate window):

* BASE = HEAD `64e4dfbc`, TEST = HEAD + the option-2 diff only.
* Doctests: clus 565/565 (8 new tests in
  `clus/test/doctest_perblob_primitives.cxx`: carve, both merge overloads,
  total-separation round trip, remove=true, stale drop, foreign-part drop,
  raw-primitive round trip + helper), match 36/36.
* SBND 30-event, `setarch x86_64 -R`, tags d55b{off,on} vs d55t{off,on}:
  **off arm 120/120 archives member-content identical, 30/30 nusel tables
  identical** (tgm 121 / stm 43 / fc 50 both); **on (assoc) arm 120/120 +
  30/30 identical** (tgm 120 / stm 51 / fc 51 both) — the assoc arm is the
  critical equivalence: decompose/recompose + un-merge run with arrays
  PRESENT, proving primitive carve/concat == the removed inline realign.
* PDHD+PDVD abtest (`snap/d54base` vs `snap/d54opt2`, clus stage):
  **OVERALL PASS** — pdhd 027305_0 (15 archives) + pdvd 039349_0/039252_5
  (27 each) all content-identical.  3 pdhd manifest events (027409_0,
  027980_3, 028084_18) failed IDENTICALLY in both arms ("no cluster
  tarballs found" — pre-existing input rot, not this change).
* uBooNE qlport sweep (35 events, labels d54base_ub vs d54opt2_ub,
  `setarch` via run_one.sh): **ZIPS 35/35 content-identical**; tagger-log
  residue 33 DIFF at exactly the historically accepted level (past accepted
  campaigns pre/post_ctpcmerge: 32, pre/post_stmfit: 32; same signature —
  PR pointer-order sensitivity across different binary layouts, multiset-
  level feature drift, Bee zips unaffected).
* WCT_PROV_CHECK smoke (1 assoc-chain event, QL -save-assoc + PR
  -unmerge-assoc, TEST binary): 73 checkpoints, all "0 problem(s)".
* Speed: uBooNE 35-event sweep wall 485s -> 484s, mean RSS 1664 -> 1667 MB
  (no drift; the enforcement fast path is one local_pcs map lookup per
  participant).  PDHD/PDVD re-timing on a quiet box: see gate log below.
* PDHD/PDVD re-timing on a quiet box (labels d54base_t2/d54opt2_t2):
  pdhd 027305_0 wall 167 s -> 164 s (RSS 1903 -> 1902 MB), pdvd 039349_0
  11 s -> 11 s (490 -> 490 MB) — no measurable cost.  (The first-pass
  +12-15% wall deltas were co-load contention from the concurrent SBND
  campaign and a nice'd worktree build; discarded.)

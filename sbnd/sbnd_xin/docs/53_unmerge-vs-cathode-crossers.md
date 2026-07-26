# 53 — Does the un-merge break the cathode-crossing merges?

Audit requested by the owner after the doc-52 fixes landed
(`882ad107`, `7c0108af`). The two `ClusteringUnmergeBundle` instances undo two
*bookkeeping* merges. The cathode-crossing joins are *real* merges — one
particle, two TPCs — and must survive both. This doc reports the measurement.

**Verdict: no cathode-crossing merge is broken.** 28 cross-cathode units across
the 30 events served on port 5011 (tag `d52ron`), all 28 retained whole as the
fitted main. Zero torn, zero provenance scatter, confirmed independently in the
delivered PR output tree.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 unmerge_crosser_audit.py work-mcp10-d52ron work-mcp1000-d52ron \
                                 work-mcp1000b-d52ron          # add -v for the tables
# rc=0 => every check passed.  Log-side sweep:
grep -h -E "no flash-merge provenance|no blob marked real_cluster_main|provenance size|BLOB LOSS in separate|erased a steiner_pc built before" \
     work-mcp*-d52ron/nusel_evt*/wct_nusel_evt*.log        # -> empty
grep -oh "([0-9]* exact/provenance, [0-9]* proxy/component)" \
     work-mcp*-d52ron/nusel_evt*/wct_nusel_evt*.log |
  awk -F'[(, ]' '{e+=$2; p+=$4} END{print "exact="e" proxy="p}'   # -> exact=329 proxy=0
```

Inputs are our own pipeline products: `ql_evt*/pctree-evt*.tar.gz` (post
cathode_connect + flash merge) and `nusel_evt*/pctree-pr-evt*.tar.gz` (post both
un-merges). Read-only — nothing was written into any `work-*` tag dir.

## Why the crossers are safe by construction

Pipeline order, SBND all-APA stage, after Q/L matching
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet:356-375`):

```
switch_scope -> extend/regular/parallel_prolong/close/extend_loop (use_flash_t0)
             -> cathode_connect            <- joins the two halves of a crosser
             -> examine_bundles(use_flash_t0=true, flags_from_longest=true)
                                           <- writes real_cluster_id/_main
```

Then the PR job: `switch_scope, unmerge_bundle, unmerge_assoc, steiner, …`.

Three properties make the crosser safe, and each was verified rather than
assumed:

1. **cathode_connect runs BEFORE the flash merge.** So a connected crosser
   enters `examine_bundles` as ONE member and its blobs all receive a single
   `real_cluster_id`. Pass 1 groups by that id, so the crosser moves as one
   piece whichever way it goes.
2. **cathode_connect CARRIES the isolated-grouping provenance verbatim.** Its
   call site is `merge_clusters(g, live_grouping)` — `clustering_cathode_connect.cxx:607`
   — i.e. `pcname` defaults to `"perblob"`, which switches on the carry block in
   `ClusteringFuncs.cxx`. The main array is concatenated unchanged, so both
   halves stay `assoc_cluster_main == 1`.
3. **Pass 2 retains the UNION of every main-marked member.** `groups_from_provenance`
   splits off only `main == 0` rows, so two main-marked halves are retained
   together.

`real_cluster_id`/`real_cluster_main` are NOT in `carry_pairs` — harmless here
only because cathode_connect precedes the writer. Any future merge pass placed
*after* `examine_bundles` would drop that pair; the visitor then skips the
cluster with a warning rather than mis-splitting it (mode `real` never falls
back to the component proxy), so the failure mode is safe but silent.

## Measurement

Method: `real_cluster_id` labels one pre-flash-merge member. A member whose
blobs sit in both SBND drift volumes (blob `wpid>>4` = 0 and 1) is by
construction the product of a cross-cathode merge — nothing else in the chain
can put charge from both TPCs into one cluster. cathode_connect logs no pair
list, so this is the crosser definition used. Both un-merges are then replayed
exactly (`final main = real_main != 0 AND assoc_main != 0`) and the result is
checked against the PR job's own output tree.

30 events, 491 Q/L clusters (`work-mcp10-d52ron`, `work-mcp1000-d52ron`,
`work-mcp1000b-d52ron`):

| check | result | required |
|---|---|---|
| cross-cathode pre-merge units | 28 | — |
| (i) kept whole as the fitted main | **28** | — |
| (ii) kept whole, demoted to associated | 0 | — |
| (iii) **torn across the cathode** | **0** | 0 |
| (iv) cluster skipped for want of `flag_main_cluster` | 0 | — |
| scatter: `real_cluster_id` group straddling the cathode | 0 | 0 |
| scatter: `assoc_cluster_id` group straddling the cathode | 0 | 0 |
| END STATE: cross-cathode units in the PR tree | 28 | — |
| END STATE: of those, spread over >1 PR cluster | **0** | 0 |
| visitor split MORE than the replay upper bound | 0 | 0 |
| `proxy/component` splits (mode leak) | 0 of 329 | 0 |
| visitor alarm lines in 30 nusel logs | 0 | 0 |

Per work dir: mcp10 10/10 whole, mcp1000 13/13, mcp1000b 5/5.

The end-state row is the one that does not trust the replay: it reads the
delivered `pctree-pr-evt*.tar.gz`, finds each cross-cathode unit's surviving
main content (`real_cluster_main != 0 AND assoc_cluster_main != 0`, which by
construction exists only inside a retained main, and a retained main keeps its
Q/L ident), and confirms it still spans both TPCs. It counts 28 — equal to the
replay's 28 — and it `continue`s whenever the Q/L ident is not found in the PR
tree, so **all 28 crossers were located by ident in the delivered product**;
none of them fell into the "cluster the visitor left alone" path below. That is
what rules out a torn crosser hiding among the skipped clusters.

Largest protected crossers (evt / cluster / blobs / drift-x span cm):

```
288397 cl11  1313  264.0      285185 cl15  1123  252.7      285185 cl17  1039  285.7
286241 cl12  1167  236.0      285185 cl16  1112  307.4      284657 cl28   995  137.4
288067 cl11  1145  252.4      289475 cl15  1053  302.7      286197 cl10   890  207.9
```

## Two things that look alarming and are not

**`real_cluster_id` is not globally unique.** A cluster that was never
flash-merged gets the default fill `rid = its own current ident`
(`MultiAlgBlobClustering.cxx:2418`), which can collide with a merged
neighbour's pre-merge member id. evt288397 and evt288727 each have two
different Q/L clusters carrying a unit labelled 11 / 4. A naive end-state check
keyed on `rid` alone reports these as split crossers; keyed on
`(cluster ident, rid)` they are two unrelated units, both intact. Anyone reading
this provenance downstream must key per cluster.

**47 clusters are left alone by the visitor** (0 further clusters have an ident
absent from the PR tree). These fail `require_in_scope` —
detector-edge shards (e.g. evt289805 cl8 and evt286681 cl2 sit at z ∈
[501.1, 501.7] cm, outside the active z ∈ [0, 501]) — or have `nb < 2`. Skipped
means *no split at all*, which cannot tear anything. The replay models a split
for them, so it is a strict UPPER BOUND on splitting: zero disagreements ran the
other way, so "whole in the replay" implies "whole in reality".

## PDHD / PDVD

The question's other half — detectors where the cathode merge happens *before*
Q/L matching — needs no measurement. `grep -rn "ClusteringUnmergeBundle\|unmerge_bundle" cfg/`
hits exactly two files: `cfg/pgrapher/common/clus.jsonnet` (the builder) and
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` (the only instantiation). PDHD's and
PDVD's `clus.jsonnet` call `cm.cathode_connect` but never the un-merge builder,
and `grep -rn unmerge` over `wcp-porting-img/pdhd/` and `pdvd/` is empty (both
trees exist and contain the runner scripts, so the empty result is real). Their
cathode_connect merges are never un-merged.

## Observation, not a defect: 98 flash-bundle-only cross-cathode clusters

98 Q/L clusters span both TPCs *only* because the flash-time bundle put two
different pre-merge clusters together. Separating those is exactly the
visitor's job, and the surviving main is correctly single-TPC. Example:
evt285185 cl19 spans x ∈ [−201, +60] before the un-merge; its TPC1 content is a
6-blob clump at x ≈ 59 with a different `real_cluster_id`, split off as PR
cl23, leaving the main at x ∈ [−201, −34].

Screening them for a genuine crosser that cathode_connect *missed* — both
halves reaching within 20 cm of the cathode and meeting within 10 cm there —
leaves **2 of 98**:

```
evt288287 cl13  526 blobs  gap 6.7 cm  gid 1000009  t0 -734.50 us
evt289805 cl21   79 blobs  gap 2.8 cm  gid      8   t0 -1428.17 us
```

These are a cathode_connect *recall* question, not an un-merge defect: if
cathode_connect had joined them they would be one unit and the un-merge would
keep them whole, as it does for the other 28. Worth a hand scan in the 5011
viewer; not fixed here.

## Status

No code changed. `d52ron` needs no revalidation on this account.

Sample size is 28 crossers over 30 events — thin. `unmerge_crosser_audit.py` is
read-only pure file parsing, so it can be pointed at any additional `work-*-<tag>`
dirs (`python3 unmerge_crosser_audit.py <dir> …`) for a larger sample without
re-running the chain.

# Over-clustering of a gamma blob into a cosmic muon — SBND_xin MC event 11, cluster 7

## Summary

In the SBND_xin MC run of **event 11 (idx 3)**, the per-APA **TPC0** display
(`work/evt11/mabc-apa0-face0.zip`, final cluster **id 7**) shows a cosmic-muon
track with a small, likely over-clustered, gamma-like EM blob attached at
**(x, y, z) = (-35.3, -141.5, 387.3)** cm (raw `x`, so TPC0).

The merge happens in **`clustering_isolated`** (the very last grouping pass before
`examine_bundles`), **not** in `clustering_neutrino`. It is the
`clustering_isolated` "**small → big**" branch
(`clus/src/clustering_isolated.cxx:238-266`): a short cluster is attached to the
nearest long cluster purely on a **80 cm closest-point distance cut, with no
angle/topology check**. The gamma blob sits 51 cm from the muon, well inside the
cut, so it is absorbed.

Important caveat: this "merge" is the MicroBooNE-style **grouping** tail. It
physically collapses the blobs into one cluster *in the tree* (hence one cluster
in the Bee display), but it also writes a recoverable per-blob provenance array
(`"isolated"/"perblob"`, main = longest sub-cluster), which group-aware QLMatching
and `ClusteringRecoveringBundle` can decompose downstream. So the over-clustering
is partly "by design" — the question is whether a 16 cm EM blob 51 cm away from a
through-going muon *should* be grouped with it.

## How this was determined

A temporary per-step probe was inserted in the MABC pipeline loop
(`clus/src/MultiAlgBlobClustering.cxx:2006-2008`) that, after every clustering
step, logs the cluster owning the point nearest `P = (-35.3, -141.5, 387.3)` (the
**gamma**) and the longest cluster (the **muon**), plus whether they are now the
same cluster. (All instrumentation described here is temporary and has been
reverted — no algorithm code is changed by this investigation.)

Per-APA **apa0** (TPC0) trace — the gamma matches `P` to within 0.02 cm:

| pipeline step | n clusters | gamma id / len / npts | muon id / len | same? |
|---|---|---|---|---|
| pointed … neutrino | 50 | **50** / 16.1 cm / 250 | 41 / 411 cm | **no** |
| **isolated** | 50 → **14** | **7** / 414 cm / 5678 | 7 / 414 cm | **YES** |
| examine_bundles | 14 | 7 / 414 cm | 7 | yes |

The gamma survives as a **separate 16 cm cluster all the way through
`clustering_neutrino`**, and only at **`clustering_isolated`** does it collapse
into the muon (final id 7, 414 cm, 5678 pts). `SAME` flips `0 → 1` precisely at
that step. (The earlier id changes — 50 → 36 → 48 etc. — are just the
`cluster_id_order: 'tree'` renumbering after passes that add/remove other
clusters; the gamma stays a distinct object until isolated.)

A second probe inside `clustering_isolated` confirms the exact branch and cut:

```
[ISO-CLASS] near-P cluster ident=50 max_range=41 (cut 150) len=16.0899cm (cut 20) -> SMALL  ctr=(-36.52,-143.07,385.19)
[ISO-SB]    near-P small ident=50 len=16.0899cm -> nearest big ident=41 big_len=410.961cm  min_dis=51.3545cm (cut 80) -> MERGE
```

So the gamma is classified **SMALL** (wire-range 41 ≪ 150 **and** length 16.1 cm
< 20 cm), its nearest **big** cluster is the muon (411 cm), the closest-point gap
is **51.35 cm < 80 cm**, and it is merged.

## The triggering logic

`clustering_isolated` (`clus/src/clustering_isolated.cxx`):

1. **Classification** (lines 141-223). A cluster is **small** if its max
   wire/time range `< range_cut = 150` **and** `get_length() < length_cut = 20 cm`
   (line 166); otherwise it is **big** (with an intermediate
   `JudgeSeparateDec_1`/`Separate_2` re-check for 20-60 cm clusters). The 16 cm
   gamma → small; the 411 cm muon → big.

2. **Small → big merge** (lines 238-266). For each small cluster, find the nearest
   big cluster by closest-point distance and, **if `min_dis < small_big_dis_cut =
   80 cm`, merge — unconditionally** (lines 261-265):

   ```cpp
   double small_big_dis_cut = 80 * units::cm;                 // line 238
   ...
   std::tuple<int,int,double> results = big_cluster->get_closest_points(*curr_cluster);
   double dis = std::get<2>(results);
   if (dis < min_dis) { min_dis = dis; min_dis_cluster = big_cluster; }
   ...
   if (min_dis < small_big_dis_cut) {                          // line 261
       to_be_merged_pairs.insert(std::make_pair(min_dis_cluster, curr_cluster));
       used_small_clusters.insert(curr_cluster);
   }
   ```

   There is **no angle, direction, vertex, or PCA/shower-vs-track gate** on this
   branch. This is the loosest merge cut in the whole per-APA chain — every other
   merge pass (`regular`, `parallel_prolong`, `neutrino`, …) applies angle/PCA/
   vertex consistency tests; `clustering_neutrino` in particular has elaborate
   `judge_vertex` / direction / PCA-ratio gates (`clustering_neutrino.cxx:725-857`)
   and, as the probe shows, **declined** to merge this same blob.

3. **Grouping & provenance** (lines 471-520). The merge pairs are unioned into
   groups (longest = `max_cluster`), and `merge_clusters(g, live_grouping,
   "isolated")` (line 520) physically merges each group into one cluster while
   writing the per-blob `"isolated"/"perblob"` array (main blobs tagged with the
   longest sub-cluster's parent id). This is the MicroBooNE-style tail consumed by
   group-aware QLMatching (see `docs/clustering.md`, `ClusteringRecoveringBundle`).

### Why the gamma triggers it
- It is geometrically **small** (16 cm, 250 pts) → enters the small pool.
- The cosmic muon is the dominant **big** cluster in TPC0 and the nearest one.
- Their closest-point gap (51 cm) is **well under the 80 cm cut**, and the branch
  asks nothing else — not whether the blob points at the muon, connects at an
  endpoint, or is track- vs shower-like. A transversely-displaced EM blob is
  absorbed exactly the same as a genuine broken-off track fragment would be.

## Ideas for improvement (no code changed yet)

Any change must be **jsonnet-togglable and default-OFF** so existing production
output stays bit-identical (repo convention).

1. **Add a topology/angle gate to the small → big branch** (most direct). Mirror
   the cheaper `clustering_neutrino` checks: only attach a small cluster to a big
   one if it connects near a big-cluster **end/vertex** (`judge_vertex`) or is
   roughly **collinear** with the local big-cluster direction at the contact
   point. A 16 cm blob 51 cm transverse from a through-going muon would then be
   rejected. This is the smallest, most targeted fix.

2. **Make the 80 cm cut size/geometry aware.** 80 cm closest-point is very loose
   for a 16 cm object. Options: scale the cut with the small cluster's length, or
   require the gap to be small relative to a plausible track-continuation length
   rather than an absolute 80 cm.

3. **Track-vs-shower discriminant.** If the small cluster is shower-like (PCA
   roundness, low `values[1]/values[0]` separation, charge spread) and not
   collinear with the big track, keep it separate. EM gammas are exactly the
   population this would protect.

4. **Treat it downstream instead of in clustering.** Because the per-blob
   provenance is preserved in `"isolated"/"perblob"`, an alternative is to leave
   `clustering_isolated` untouched and instead make the consumer (group-aware
   QLMatching split, and/or the Bee clustering display) keep the associated EM
   blob visually/logically distinct from the muon main. This avoids touching the
   merge logic at all if the only concern is the displayed over-clustering.

A reasonable first experiment is option (1) behind a default-OFF SBND flag,
validated on this event (gamma id 50 should remain separate) plus the standard
10-event MC sample to confirm no regressions on genuine track fragments.

## Reproduction

```bash
cd sbnd_xin
./run_clus_evt.sh mc 3          # event 11; inputs in work/evt11/icluster-apa0-*.npz
# inspect work/evt11/mabc-apa0-face0.zip -> cluster 7 (muon + gamma)
```

The per-step / branch findings above were obtained by temporarily instrumenting
`clus/src/MultiAlgBlobClustering.cxx` (per-step nearest-cluster probe) and
`clus/src/clustering_isolated.cxx` (classification + small→big merge prints),
rebuilding (`./wcb build && ./wcb install`), and re-running the same command. That
instrumentation was removed afterward.

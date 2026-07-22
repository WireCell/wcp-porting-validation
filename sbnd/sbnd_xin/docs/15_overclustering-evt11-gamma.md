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

**Status:** the `isolated` `length_cut` (and `range_cut`) classification
thresholds are now configurable (default unchanged at 20 cm / 150), and SBND opts
into **`length_cut = 15 cm`** so this 16 cm gamma is no longer classified "small"
and is left as its own cluster. See **Resolution implemented** below for the code
locations and the 20-event impact study.

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
   group-aware QLMatching (see `docs/5_clustering.md`, `ClusteringRecoveringBundle`).

### Why the gamma triggers it
- It is geometrically **small** (16 cm, 250 pts) → enters the small pool.
- The cosmic muon is the dominant **big** cluster in TPC0 and the nearest one.
- Their closest-point gap (51 cm) is **well under the 80 cm cut**, and the branch
  asks nothing else — not whether the blob points at the muon, connects at an
  endpoint, or is track- vs shower-like. A transversely-displaced EM blob is
  absorbed exactly the same as a genuine broken-off track fragment would be.

## Resolution implemented: configurable `length_cut`, SBND set to 15 cm

The gamma is absorbed only because its 16.1 cm length is **under the 20 cm
`length_cut`** that defines "small". The most direct, lowest-risk lever is to
tighten that threshold so a ~16 cm EM blob is no longer auto-classified small.

`length_cut` (and `range_cut`) were **hardcoded** in `clustering_isolated.cxx`.
They are now **configurable**, defaulting to the historical `20 cm` / `150`:

- `clus/src/clustering_isolated.cxx` — `configure()` reads `length_cut`
  (default `20*units::cm`) and `range_cut` (default `150`); both are threaded
  into the classification at the top of `clustering_isolated()`.
- `cfg/pgrapher/common/clus.jsonnet` — `isolated(... length_cut=null,
  range_cut=null)`; the keys are emitted **only when non-null**, so every config
  that omits them is byte-identical to before (the C++ defaults take over).
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet` — SBND opts in with
  `cm.isolated(length_cut=15 * wc.cm)` (range_cut left at 150).

Note: this SBND file is shared by the standalone dev chain *and* LArSoft
production (`wcls-img-clus.jsonnet` via `per_volume`), so the 15 cm value applies
to both. Other detectors are unaffected (they never pass the key → 20 cm).

### Impact of 20 cm → 15 cm (10 MC + 10 data events)

Metric = per-APA **sorted cluster size-vector** (point count per cluster), compared
across both APAs of all 20 events (40 per-APA outputs). **Exactly 2 of the 40
per-APA outputs change; the other 38 are byte-for-byte identical** (full
size-vector diff, not just cluster count).

| event | sample | APA | clusters 20cm → 15cm | what changed |
|---|---|---|---|---|
| evt11 | MC | apa0 | **14 → 15** | gamma **separated** from muon: muon 5678 → 5333 pts, new **355-pt** gamma cluster (was Bee cluster 7) |
| evt1720 | data | apa1 | **8 → 9** | a **48-pt** piece split off the 8702 → 8654-pt cluster |
| all 18 other events | 9 MC + 9 data | both APAs | unchanged | identical sorted size-vector in every APA |

Per-cluster sizes (descending), for the two affected APAs:

```
evt11  apa0  20cm: [5678, 2041, 819, 645, 198, 80, 68, 14, 14, 13, 13, 12, 12, 12]
evt11  apa0  15cm: [5333, 2031, 819, 645, 355, 198, 80, 68, 14, 14, 13, 13, 12, 12, 12]
evt1720 apa1 20cm: [17737, 8702, 3039, 2253, 1951, 733, 10, 2]
evt1720 apa1 15cm: [17737, 8654, 3039, 2253, 1951, 733, 48, 10, 2]
```

So the change is **surgical**: it only releases clusters whose length sits in the
15–20 cm window that were previously pulled into a nearby long track. For the
target (evt11) this is exactly the desired effect — the gamma blob is now its own
cluster instead of part of the cosmic muon.

### Caveats / things to watch
- This is a **global** classification threshold, not gamma-specific. The single
  data change (evt1720, a 48-pt fragment) is benign here, but on larger samples a
  tighter cut could keep genuine short broken-off **track** fragments separate
  that the 20 cm cut would have re-attached. Worth a wider-sample check before
  treating 15 cm as final.
- The underlying looseness is the **angle-less 80 cm small→big merge**
  (`clustering_isolated.cxx:238-266`). `length_cut` only changes *which* clusters
  enter that branch; it does not add a topology test. If short EM blobs longer
  than 15 cm still get absorbed, the more complete fix is to add an
  angle/vertex/PCA gate to the small→big merge (mirroring `clustering_neutrino`'s
  `judge_vertex`/direction checks), which can also be made a default-OFF knob.

## Reproduction

```bash
cd sbnd_xin
./run_clus_evt.sh mc 3          # event 11; inputs in work/evt11/icluster-apa0-*.npz
# 15 cm (current SBND config): work/evt11/mabc-apa0-face0.zip -> gamma is its own cluster
```

To A/B the threshold, edit `cm.isolated(length_cut=15 * wc.cm)` in
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` (drop the arg for the 20 cm default).
The impact numbers above were produced by running all 10 MC + 10 data events at
each setting and diffing the per-APA **sorted cluster size-vectors** (point count
per `cluster_id`) from the Bee `*-clustering-*.json` — exact equality for 38 of
40 per-APA outputs, the two changed ones listed above.

The per-step / branch findings earlier in this doc were obtained by temporarily
instrumenting `clus/src/MultiAlgBlobClustering.cxx` (per-step nearest-cluster
probe) and `clus/src/clustering_isolated.cxx` (classification + small→big merge
prints), rebuilding (`./wcb build && ./wcb install`), and re-running. That
instrumentation was removed afterward; the configurable `length_cut`/`range_cut`
described above is the only retained change.

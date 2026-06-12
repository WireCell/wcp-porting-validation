# SBND Pattern-Recognition Integration Plan (`PR_integration.md`)

> Status: living document — guidance for the next phase of SBND Wire-Cell work.
> For the already-integrated chain see **[sbnd.md](sbnd.md)** (imaging),
> **[clustering.md](clustering.md)** and **[ql-chain.md](ql-chain.md)** (charge-light matching).
> For geometry / timing constants see **[geometry-and-timing.md](geometry-and-timing.md)**.

## Scope and context

For SBND we have so far integrated three stages of the Wire-Cell chain:

1. **Imaging** — tiling + charge solving per APA (`wct-img-all.jsonnet`).
2. **Clustering** — per-APA MABC pipeline followed by an all-APA, flash-aware
   pipeline including `cathode_connect` (`cfg/pgrapher/experiment/sbnd/clus.jsonnet`).
3. **Charge-light matching** — `QLMatching` with the SBND-specific prefilter,
   cross-TPC consistency cull, PMT non-linearity, etc.
   (`cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet`).

The **remaining Wire-Cell chain is pattern recognition (PR)**, currently
exercised end-to-end only for MicroBooNE via
`qlport/uboone-mabc.jsonnet`. Its MABC pipeline (after clustering + matching)
is:

```jsonnet
local cm_pipeline = [
    cm.tagger_flag_transfer("tagger"),
    cm.clustering_recovering_bundle("recover_bundle", graph_name="relaxed_pid"),
    cm.switch_scope(),
    cm.steiner(retiler=improve_cluster_2, perf=perf),     // retiling + Steiner graph
    cm.fiducialutils(),
    // cm.tagger_check_stm(...),                          // STM/TGM tagger (available, currently off)
    cm.tagger_check_neutrino(...),                        // full PR chain incl. track fitting
]   + (if numu_weights_dir != "" then [numu_bdt_scorer] else [])
    + (if nue_weights_dir  != "" then [nue_bdt_scorer]  else [])
    + (if tracking_output  != "" then [tracking_visitor, tagger_output_visitor] else []);
```

(see `qlport/uboone-mabc.jsonnet:1238-1250`).

This document describes, stage by stage, how to integrate these tools into
the SBND chain. Each section gives:

- **Algorithm overview** — what the component does and where it lives.
- **Integration attention points** — configuration parameters that must be
  re-derived for SBND, and in particular how to handle the **two TPCs and the
  connection between them** (new relative to MicroBooNE's single TPC), plus
  the fact that SBND T0 comes from charge-light matching rather than a single
  beam flash.
- **Validation plan** — following the MicroBooNE approach: per-function review
  against the WCP prototype, then event-by-event numerical comparison
  (`qlport/run_5384.pl` + `check_tagger_5384.pl` style: `track_com_*.root`
  via `wire-cell-uboone-magnify-tracking-convert`, tagger logs), then Bee
  visual hand-scans.

Cross-cutting reminders:

- **MicroBooNE → SBND constant translation.** Drift speed 1.101 → 1.563 mm/µs,
  `time_offset` −1600 µs → −205 µs, FV box (x: 1–255 cm single drift) → two
  mirrored drift volumes (x: ±2.5 to ±201.05 cm), E-field 0.273 → 0.5 kV/cm
  (recombination!), readout 9595 → 3427 ticks. Never copy a MicroBooNE number
  without asking which detector property it encodes.
- **Two TPCs.** Every PR component that derives a drift direction, an
  anode/cathode x position, or a dead-region lookup from "the" anode face must
  instead resolve it per point via `wpid` / `DetectorVolumes::contained_by()`.
  After `cathode_connect` a single cluster can span both TPCs.
- **ML retraining.** The DL vertex SCN model and the NuMu/NuE BDTs are trained
  on MicroBooNE samples and must be retrained on SBND MC (see §4, §6); until
  then run with them disabled (`dl_weights=""`, weight dirs `""`).
- **Toggleability.** As for all previous SBND work: every new pass must be
  jsonnet-togglable and default OFF (or scoped to SBND) so existing
  production configs stay bit-identical.

Sections:

1. [Retiling](#1-retiling)
2. [Steiner graph building](#2-steiner-graph-building)
3. [STM tagger](#3-stm-tagger)
4. [Neutrino tagger](#4-neutrino-tagger)
5. [Track trajectory and dQ/dx fitting](#5-track-trajectory-and-dqdx-fitting)
6. [Various event-selection taggers](#6-various-event-selection-taggers)
7. [Running SBND simulation](#7-running-sbnd-simulation)

---

## 1. Retiling

### 1.1 Algorithm overview

Retiling re-derives a cluster's blobs *after* clustering and T0 correction,
so that the downstream PR stages (Steiner graph, track fitting) work on a
cleaner, better-localized blob set than the original imaging output:

1. For each cluster, collect the wire-plane **activities** (per-plane charge
   intervals per time slice) covered by its blobs.
2. Re-run WCT **RayTiling** on those activities to form new blobs — now with
   the cluster isolated from the rest of the event, so ghost solutions that
   imaging had to keep are no longer competitive.
3. Re-sample the new blobs with an `IBlobSampler` (`charge_stepped` strategy)
   to make new point clouds, and rebuild the cluster's PC-tree.
4. Keep an old→new blob "shadow" mapping so flags and per-blob arrays survive.

Implementation:

| piece | where |
|---|---|
| Core engine `RetileCluster` | `clus/src/retile_cluster.{h,cxx}` |
| Visitor wrapper `ClusteringRetile` (`cm.retile(...)`) | `clus/src/clustering_retile.cxx` |
| Steiner-aware variants `ImproveCluster_1/2` | `clus/src/improvecluster_1.{h,cxx}`, `improvecluster_2.cxx` |
| jsonnet constructors | `cfg/pgrapher/common/clus.jsonnet` (`retiler()`, `improve_cluster_1/2()`) |

In the MicroBooNE pipeline retiling is not run as a standalone pass
(`cm.retile` is commented out); instead **`ImproveCluster_2` is handed to
`CreateSteinerGraph` as its `retiler`** and is invoked per cluster from
inside the Steiner stage (§2). `ImproveCluster_*` extend `RetileCluster` by
modifying ("hacking") the activities using Steiner-tree information before
re-tiling, recovering charge in gaps and trimming spurious wings.

Key configuration (MicroBooNE values, `qlport/uboone-mabc.jsonnet`):

```jsonnet
improve_cluster_2: {
  type: "ImproveCluster_2",
  data: {
    anodes: [wc.tn(a) for a in anodes],
    samplers: [ {name: wc.tn(bs_live_no_dead_mix), apa: 0, face: 0} ],
    cut_time_low: 3*wc.us, cut_time_high: 5*wc.us,   // retile time window
    verbose: false,
    detector_volumes: ..., pc_transforms: ...,
  },
}
// sampler: BlobSampler, strategy charge_stepped, disable_mix_dead_cell=false,
// time_offset = -1600us + 6mm/drift_speed, drift_speed = 1.101 mm/us
```

### 1.2 Integration attention points for SBND

**Sampler list must cover every (apa, face).** MicroBooNE registers a single
sampler `{apa: 0, face: 0}`. SBND must register one per live face:
`{apa: 0, face: 0}` and `{apa: 1, face: 0}` (one drift face per APA in WCT's
SBND wires file). The sampler's `time_offset`/`drift_speed` must be the SBND
values already used by `ctpointcloud`/imaging: `drift_speed = 1.563 mm/us`,
`time_offset = -205 us` (see `geometry-and-timing.md`). Any mismatch shifts
retiled points in x relative to the original point cloud — the same class of
bug as the PDVD "behind the anode" T0 artifacts.

**Retiling happens in *detector* coordinates but tiling in *wire/tick*
coordinates.** The activities must be regenerated against the correct face:
for a cross-TPC cluster (post `cathode_connect`) the blobs carry two
different `wpid`s, and the retiler must group blobs per (apa, face) and tile
each group against its own face geometry, then re-assemble one cluster.
Verify `RetileCluster` does its internal bookkeeping per `wpid` and not per
"the first face of the cluster" — this is the recurring multi-TPC failure
mode (cf. the per-APA assumptions found in the STM tagger, §3).

**T0 must be applied before retiling.** Retiling maps points back to
(channel, tick) space; that mapping is only correct when the cluster's
drift coordinate is right. In SBND the T0 comes from QLMatching
(`cluster_t0` scalar) and is applied by `switch_scope()` — note the pipeline
order in §0: `switch_scope` runs *before* `steiner`/retiling. Clusters with
**no matched flash** have no T0; decide the policy (skip retiling, or retile
at nominal T0 and flag). MicroBooNE never faces this because every event
there has a single beam flash assumption with fixed `time_offset`.

- The `cut_time_low/high = [3, 5] us` window is a MicroBooNE-tuned guard on
  how far in time the re-tiled activities may extend; re-derive for SBND's
  tick/slice configuration (4 ticks per slice in SBND imaging as well, but
  different drift speed → different mm equivalent).

**Dead channels.** Retiling consults the anode's channel masks. SBND-specific
dead regions (the W-defect band handled by the dead-gap registry, see
`dead_blob.md`) must be visible to the retiler the same way they are to
imaging, or retiled blobs will "heal" across truly-dead wires and create fake
charge. Use the same `chndb`/masks input as `wct-img-all.jsonnet`.

**`disable_mix_dead_cell`.** MicroBooNE uses `false` for the
`improve_cluster_2` sampler (dead-cell mixing ON) but `true` for the plain
retile sampler. Start with the MicroBooNE choice; revisit once SBND dead
regions are in (two 2-view patches behave differently from MicroBooNE's
large shorted regions).

**Where it sits in the SBND graph.** Retiling/Steiner belongs in the
**all-APA** MABC instance (after `PointTreeMerging` + `switch_scope` +
`examine_bundles`), i.e. appended to the `clus.jsonnet` all-APA pipeline —
*not* the per-APA one — so it sees final, T0-corrected, possibly cross-TPC
clusters. This mirrors MicroBooNE where it runs after
`tagger_flag_transfer`/`recover_bundle`/`switch_scope`.

### 1.3 Validation plan (MicroBooNE-style)

1. **Prototype parity on MicroBooNE first.** Any code change made for SBND
   (multi-face support, per-wpid bookkeeping) must keep the MicroBooNE
   output identical: rerun `qlport/run_5384.pl` events and diff
   `track_com_5384_*.root` / `tagger_*.log` against the stored references.
2. **Blob-level checks on SBND.** For a sample of clusters compare
   pre-retile vs post-retile: blob count, total charge, point count, x/y/z
   envelope. Retiling should not move the cluster envelope by more than a
   slice thickness, and total charge should be approximately conserved
   (large drops ⇒ activities lost; large gains ⇒ ghost blobs re-introduced).
3. **Bee hand-scan.** Add a `retiled` Bee point set next to the existing
   clustering sets (same upload pattern as `run_clus_evt.sh`) and hand-scan
   ~10 events: cosmics through the cathode (cross-TPC clusters), tracks near
   the W-defect dead band, and tracks at high drift (T0-sensitive).
4. **Cross-TPC stress test.** Pick events with `cathode_connect`-merged
   clusters (the 150-tail sample from the cathode-crossing study) and verify
   the retiled cluster still spans both TPCs with sensible blobs on each side
   and no double counting at the cathode seam.
5. **Determinism.** Run the same event twice; retiled output must be
   byte-identical (the Steiner/retile code went through pointer-order
   determinism fixes — keep it that way).

---

## 2. Steiner graph building

### 2.1 Algorithm overview

The Steiner stage turns each cluster's unstructured point cloud into a
**branching geometric skeleton** — the input to everything downstream
(proto-vertex finding, track fitting, all taggers). It is implemented as the
`CreateSteinerGraph` ensemble visitor (`cm.steiner(retiler=improve_cluster_2)`):

| piece | where |
|---|---|
| Visitor `CreateSteinerGraph` | `clus/src/CreateSteinerGraph.{h,cxx}` |
| Core algorithm `Steiner::Grapher` | `clus/src/SteinerGrapher.{h,cxx}`, `SteinerGrapher_helpers.cxx` |
| Free helpers | `clus/src/SteinerFunctions.{h,cxx}` |
| Deep dive | `clus/docs/steiner_graph.md`; port review `clus/docs/patternrecognition/steiner_graph_review.md` |

Per cluster, the two-phase algorithm is:

1. **Retiling** (§1): `ImproveCluster_2` resamples the cluster into a more
   uniform point cloud and patches gaps near dead regions.
2. **Graph construction** (`Steiner::Grapher::create_steiner_tree()`):
   - Build a weighted graph over the resampled points (k-d-tree neighbor
     search, edge weight = 3D distance).
   - Find **Steiner terminals**: charge-significant points per blob
     (`find_steiner_terminals()`, charge weighting with Q₀ = 10000,
     factors 0.8/0.4, charge threshold 4000 — empirical MicroBooNE tuning;
     `disable_dead_mix_cell` controls whether dead-mixed cells may be
     terminals).
   - Add intra-blob edges with discounted weights (0.8× distance when both
     ends are terminals, 0.9× for one) so paths prefer charge-bearing routes.
   - Run an MST (Kruskal) over the graph as the practical Steiner-tree proxy
     and prune it down to the terminal-spanning skeleton.

Outputs stored on the cluster facade:

- `steiner_graph` — the reduced tree (named graph, Boost adjacency list);
- `steiner_pc` — point cloud of the tree's points (`x_t0cor, y, z`, charge);
- `flag_steiner_terminal` — marks which points are terminals.

Graph-degree semantics drive the PR chain: degree-1 vertices are
track/shower endpoints, degree-≥3 vertices are branch/vertex candidates.
The later PR graph (`PR::Vertex`/`PR::Segment`, §4) is built by tracing
paths *through* this skeleton.

The Bee output set `steiner` (`pcname: "steiner_pc"`) and `regular`
(`pcname: "3d"`) are both emitted by this visitor — that is how the skeleton
is hand-scanned.

### 2.2 Integration attention points for SBND

**Scope plumbing, not algorithm.** The Grapher works on whatever point cloud
scope `switch_scope()` selected. In SBND the corrected coordinates are
`['x_t0cor', 'y_cor', 'z_cor']` for data (transverse `pos_offset` per TPC)
vs `['x_t0cor', 'y', 'z']` for sim — make sure the Steiner stage and its Bee
sets use `common_corr_coords(pos_offset_on)` exactly as the all-APA
clustering does, or data clusters will be skeletonized in a frame 1 cm off
from the one used for merging.

**Cross-TPC clusters.** After `cathode_connect`, one cluster holds blobs
with both `wpid`s. Points near the cathode from the two TPCs are ~cm apart
after T0 correction, so the neighbor search will naturally bridge the seam —
that is desired (one muon → one skeleton). Attention:

- `form_cell_points_map()` keys points by blob; blobs from different
  faces must not be conflated. The blob-vertex map is keyed by node index
  (determinism fix), which is face-safe, but verify terminal finding uses
  per-(apa,face) dead-cell information, not a global "the face".
- The cathode dead zone (±~2.5 cm structure exclusion, see
  `cathode_fiducial.jsonnet`) means there is a genuine charge gap at the
  seam. Terminal-finding thresholds were tuned for MicroBooNE charge scales;
  check that the seam does not split the skeleton into two trees (the MST is
  global per cluster, so connectivity survives, but terminal pruning could
  drop the bridge if no terminals exist near the cathode on either side).

**Charge-threshold re-tuning.** Q₀ = 10000, threshold = 4000 and the
0.8/0.4 factors encode MicroBooNE's gain and SP normalization. SBND's
charge scale post-SP is similar but not identical — re-derive by comparing
the dQ/dx MIP peak (electrons/cm) between detectors before trusting terminal
selection, and prefer exposing these as jsonnet knobs rather than editing
constants (they are currently hardcoded in `SteinerGrapher.cxx`).

**Unmatched clusters.** Same policy question as §1: clusters with no
`cluster_t0` have undefined `x_t0cor`. MicroBooNE always has one; SBND must
either skip Steiner for unmatched clusters or run them at nominal T0 and
propagate a flag the taggers can see.

**Cost.** Steiner is graph-lifecycle heavy (same boost adjacency_list
build/destroy pattern that dominated imaging cost). `perf: true` prints
per-cluster timing — keep it on for the first SBND runs. SBND events are
cosmic-rich (10+ matched clusters/event vs MicroBooNE's ~1 main cluster);
consider restricting the Steiner+PR stages to **flash-matched clusters in
the beam window** (or the QL `matched` survivors) rather than every cluster,
which is also what the physics needs.

### 2.3 Validation plan (MicroBooNE-style)

1. **Port parity guard.** Multi-TPC generalizations must keep MicroBooNE
   byte-identical: rerun the `qlport` event set, diff `track_com_*.root` and
   tagger logs. The Steiner port already has a determinism review
   (`steiner_graph_review.md`); preserve its invariants (node-index keys,
   sorted edge iteration).
2. **Skeleton sanity metrics on SBND.** Per cluster: number of terminals,
   tree edge count, total tree length vs cluster length (`get_length`),
   number of connected components after pruning (must be 1). Aggregate over
   ~50 events; outliers are the debug sample.
3. **Bee hand-scan.** Upload `regular` + `steiner` sets (the existing
   `run_clus_evt.sh` upload path) and scan: straight cosmics (skeleton =
   single chain), cathode crossers (single chain across the seam), showers
   (branching), and tracks through the SBND W-defect dead band (skeleton
   should bridge it via the dead-gap-aware sampling, not terminate).
4. **Two-TPC seam check.** For each `cathode_connect`-merged cluster verify
   exactly one Steiner component, and that the bridging edge length ≈ the
   physical cathode gap (no multi-meter "shortcut" edges).
5. **Determinism + timing.** Two identical runs → identical `steiner_pc`;
   record per-cluster wall time on busy cosmic events to budget the chain.

---

*Sections 3-7 follow.*

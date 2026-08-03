# SBND Pattern-Recognition Integration Plan (`19_PR_integration.md`)

> Status: living document — guidance for the next phase of SBND Wire-Cell work.
> For the already-integrated chain see **[1_sbnd.md](1_sbnd.md)** (imaging),
> **[5_clustering.md](5_clustering.md)** and **[8_ql-chain.md](8_ql-chain.md)** (charge-light matching).
> For geometry / timing constants see **[2_geometry-and-timing.md](2_geometry-and-timing.md)**.

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
`time_offset = -205 us` (see `2_geometry-and-timing.md`). Any mismatch shifts
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
`6_dead_blob.md`) must be visible to the retiler the same way they are to
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

## 3. STM tagger

### 3.1 Algorithm overview

The STM tagger classifies a flash-matched cluster as a **stopped muon (STM)**
or **through-going muon (TGM)** — the two dominant cosmic topologies that
survive generic cosmic rejection and must be removed before neutrino
selection. It is the first "physics" tagger in the chain and exercises the
full track-fitting machinery (§5) on a single cluster.

| piece | where |
|---|---|
| Visitor `TaggerCheckSTM` (`cm.tagger_check_stm()`) | `clus/src/TaggerCheckSTM.cxx` (~2700 lines) |
| Geometry helpers | `clus/src/FiducialUtils.cxx` (`check_dead_volume`, `check_signal_processing`, `inside_fiducial_volume`), `Clustering_Util.cxx` (`cluster_fc_check`) |
| Port review | `clus/docs/patternrecognition/stm_tagger_review.md` |

Per cluster the logic is (two rounds of endpoint detection, then a forward
pass and — for ambiguous-direction tracks — a backward pass):

1. **Endpoint analysis**: find the track's extreme points; for each, test
   `inside_fiducial_volume`, `check_dead_volume` (walk along the track
   direction; fail if >~81% of steps land in dead regions) and
   `check_signal_processing` (fail if the apparent endpoint is explained by
   SP inefficiency rather than a real stop).
2. **Trajectory + dQ/dx fit** of the candidate muon path (rough path through
   the Steiner graph, then two tracking iterations).
3. **`find_first_kink`** — split candidate at kinks (delta rays, decays).
4. **TGM check** — both endpoints at detector boundaries (anode/cathode
   handling, incl. dead-volume-truncated variants) ⇒ `Flags::TGM`.
5. **`eval_stm` dQ/dx ladder** — compare the end-of-track dQ/dx profile
   against the muon Bragg-peak hypothesis (using the `ParticleDataSet`
   dE/dx-vs-residual-range tables + recombination model) at several
   thresholds ⇒ stopped-muon decision.
6. **`search_other_tracks` / `check_other_tracks` / `detect_proton`** —
   veto if additional activity at the stopping point looks like an
   interaction (e.g. a proton) rather than a clean muon stop ⇒ if clean,
   `Flags::STM`.

Configuration (jsonnet `cm.tagger_check_stm(...)`):

```jsonnet
type: "TaggerCheckSTM",
data: {
  grouping: "live",
  trackfitting_config_file: "",        // JSON preset (§5); "" = built-in defaults
  particle_dataset: ...,               // dE/dx + range tables
  recombination_model: ...,            // Box model (detector E-field!)
  detector_volumes: ..., pc_transforms: ...,
  shorted_y_w_range: [],               // e.g. [7135, 7264] = uboone shorted-Y W wires
}
```

**Status:** ported and reviewed, but currently *commented out* in
`uboone-mabc.jsonnet` (line 1246) — the neutrino tagger is the exercised
path. Per the review's fidelity table the port-time issues (visitor stub,
missing `Flags::STM`, TGM anode drift direction) are fixed; the one residual
multi-TPC item is below.

### 3.2 Integration attention points for SBND

**Re-enable and re-baseline on MicroBooNE first.** Since the STM pass is
off in the current qlport pipeline, step zero is to switch it on for
MicroBooNE, regenerate the `track_com_*.root`/tagger-log references, and
hand-check a few STM/TGM tags against the prototype. Only then port.

**Residual single-TPC assumption (MULTI-APA-1).** `check_stm_conditions`
derives `drift_dir` from the *first* `wpid` of the cluster
(`TaggerCheckSTM.cxx:2167-2174` at review time). For an SBND cross-TPC
cluster the two halves drift in **opposite x directions**, so any
endpoint-vs-drift angle logic is wrong for one half. Fix as the review
prescribes: per-endpoint `m_dv->face_dirx(wpid)` using the wpid containing
that endpoint. The helpers (`cluster_fc_check`, `check_dead_volume`,
`check_signal_processing`) are already per-point `contained_by()`-based and
multi-APA-safe.

**Boundary topology is genuinely new.** MicroBooNE: one anode (x≈0), one
cathode (x≈256 cm); a TGM enters/exits through the FV surface of a single
volume. SBND: two anodes at x ≈ ±201 cm and a **shared cathode at x ≈ 0**
that is *interior* to the detector:

- A cathode-crossing muon (post-`cathode_connect`, one cluster) is a TGM
  whose "middle" passes through the cathode — its endpoints are at the two
  *outer* boundaries. Boundary tests must use the **overall** FV for
  cross-TPC clusters and the **per-TPC** FV for single-TPC clusters — reuse
  the scope-aware FV selection pattern already in `clustering_separate`
  (`select_scope_fv`).
- A muon stopping *near the cathode* has its Bragg peak inside or behind the
  CPA structure-exclusion zones (`cathode_fiducial.jsonnet`: pads, tube
  lattice, knuckles). Treat the cathode like MicroBooNE treats dead regions:
  an endpoint inside the CPA exclusion volume is "unobservable", not "not a
  stop". The 3-D cathode fiducial built for QLMatching is the right helper.
- A track clipped at the **anode** at its matched T0 is a different failure
  (containment); the QL `require_containment` / `flag_at_x_boundary`
  machinery already computes this — transfer the flags via
  `tagger_flag_transfer` rather than recomputing.

**Per-flash "main cluster", many per event.** MicroBooNE runs STM on the
single beam-flash-matched main cluster. SBND has O(10) matched flash bundles
per event (cosmics) plus the beam candidate. Run the tagger per matched
bundle main cluster (`Flags::main_cluster` is set by `tagger_flag_transfer`
from the QL matching output), and budget CPU accordingly — or gate on the
beam window if the immediate goal is only neutrino selection.

**Constants to re-derive, not copy:**

- `shorted_y_w_range` — uboone-specific guard; for SBND either empty or
  mapped to the W-defect dead band (already in the dead-gap registry; prefer
  reading that registry over a hardcoded channel range).
- `eval_stm` dQ/dx ladder thresholds — defined relative to MicroBooNE's MIP
  dQ/dx normalization; revalidate after the SBND recombination model is in
  (E-field 0.5 kV/cm vs 0.273 kV/cm changes the Box-model recombination
  factor substantially, §5).
- `check_dead_volume`/`check_signal_processing` step (1 cm) and ratios
  (0.81/0.8) — geometry-neutral, keep; but confirm SBND dead regions are in
  the CTPC dead grouping the helpers consult.

### 3.3 Validation plan (MicroBooNE-style)

1. **Function-level review is done** (`stm_tagger_review.md`); keep it
   updated when the multi-TPC fixes land, same review format (logic / bugs /
   efficiency / determinism four-point checklist).
2. **MicroBooNE numerical parity** with the pass enabled: prototype vs
   toolkit STM/TGM decisions per event on the `qlport` sample (the
   `check_tagger_5384.pl` log-diff machinery), plus fitted-track ROOT
   comparison.
3. **SBND cosmic data is the natural testbench** (no MC needed for a first
   pass): select geometrically obvious stopping muons (one endpoint well
   inside FV) and through-goers from existing hand-scanned events; measure
   tag efficiency and check the dQ/dx ladder fires on a visible Bragg peak.
   The Bragg-peak shape in tagged STMs simultaneously validates the §5
   dQ/dx chain and the recombination constants.
4. **MC efficiency/mis-tag**: on the SBND simulation sample (§7), compute
   STM/TGM efficiency vs truth (stopping/crossing muons) and the mis-tag
   rate on true neutrino interactions — the MicroBooNE figures of merit.
5. **Cross-TPC regression set**: the cathode-crossing event list from the
   `cathode_connect` study doubles as the TGM-through-cathode regression
   sample; every one of those should end up TGM-tagged.

---

## 4. Neutrino tagger

### 4.1 Algorithm overview

`TaggerCheckNeutrino` (`cm.tagger_check_neutrino(...)`) is the port of the
WCP prototype's **NeutrinoID** — the full pattern-recognition chain that
turns a flash-matched cluster (+ its associated clusters) into a physics
result: an interaction vertex, a particle-flow tree of track/shower
segments, reconstructed kinematics, and the feature variables consumed by
the event-selection BDTs (§6).

| piece | where |
|---|---|
| Visitor entry | `clus/src/TaggerCheckNeutrino.cxx` |
| Proto-vertex/segments | `clus/src/NeutrinoPatternBase.cxx`, `NeutrinoOtherSegments.cxx` |
| Structure cleanup | `clus/src/NeutrinoStructureExaminer.cxx` |
| Track/shower separation | `clus/src/NeutrinoTrackShowerSep.cxx` |
| Vertex finding/fitting (+DL) | `clus/src/NeutrinoVertexFinder.cxx` |
| Deghosting | `clus/src/NeutrinoDeghoster.cxx` |
| Shower clustering | `clus/src/NeutrinoShowerClustering.cxx` |
| Energy reco / kinematics | `clus/src/NeutrinoEnergyReco.cxx`, `NeutrinoKinematics.cxx` |
| Prototype↔toolkit map | `clus/docs/porting/neutrino_id_function_map.md` |

Sub-steps (prototype call order preserved):

1. **`find_proto_vertex`** — seed vertices from the Steiner skeleton
   (degree-based), trace rough paths, fit initial segments
   (`init_point_segment`, `break_segments`, `find_other_segments`) →
   the PR graph (`PR::Vertex` / `PR::Segment`).
2. **`examine_structure`** — merge/break pathological segments.
3. **`separate_track_shower`** — per-segment track vs EM-shower
   classification (dQ/dx, topology).
4. **`determine_main_vertex` / `improve_vertex`** — pick and refit the
   interaction vertex; optionally **DL vertex** re-ranking (below).
5. **`deghosting`** — drop segments that are projections/ghosts of others.
6. **Shower clustering** — aggregate EM activity to the parent shower.
7. **`fill_kine_tree`** — per-particle energies (dQ/dx-, range-, or
   charge-based, via `ParticleDataSet` + recombination model), neutrino
   energy.
8. **`init_tagger_info`** — fill the ~200-variable `TaggerInfo` BDT feature
   block (cosmic_*, numu_*, nue_*, ssm_*, pio_*, ...).

Configuration (MicroBooNE values):

```jsonnet
cm.tagger_check_neutrino(
  trackfitting_config_file = "uboone_track_fitting.json",   // §5
  recombination_model = wc.tn(ub.uBooNE_box_recomb_model),
  particle_dataset    = wc.tn(ub.particle_dataset),
  dl_weights   = "uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth", // "" disables
  dQdx_scale   = 0.1,   dQdx_offset = -1000,   // dQ/dx → DL-input normalization
  clus_geom_helper = wc.tn(uboone_geom_helper), // SCE position corrections
)
// further knobs: dl_vtx_rerank=true, dl_vtx_top_k=5,
// dl_vtx_min_accept_score=4.0, dl_vtx_score_scale=1000.0
```

**DL vertex**: a SparseConvNet model (uboone `.pth`, 0.5 cm voxels) scores
candidate vertex positions from the dQ/dx-normalized point cloud; the
geometric candidates are re-ranked against the top-K DL voxels. With
`dl_weights=""` the chain falls back to pure geometric vertex finding — a
fully functional mode.

Bee outputs from this visitor: `track_fit` (fitted points, dQ/dx colored),
`shower_track` (track/shower classification), `vertices` (PR-graph vertices,
primary highlighted), and the `mc` particle-flow JSON (`bee_pf`).

### 4.2 Integration attention points for SBND

**What runs on what.** MicroBooNE runs NeutrinoID once per event on the
beam-flash bundle. For SBND, restrict to **beam-window matched bundles**
(flash time in the BNB gate) — running the full PR chain on every cosmic
bundle is wasted CPU and was never the MicroBooNE operating point. The
bundle structure (main + associated clusters via `perblob`/`isolated`
arrays, group-aware matching) is already in the SBND chain;
`tagger_flag_transfer` + `clustering_recovering_bundle` are the adapters
that hand it to PR — verify they understand SBND's per-APA group tail
(`real_cluster_id` from `examine_bundles`).

**Hardcoded frame conventions are safe; drift is not.** Beam = +z and
vertical = +y match MicroBooNE. But any sub-step using a drift direction or
anode-distance must be per-wpid (same audit as §3). Specific known places to
audit: direction-vs-drift angle cuts in track/shower separation,
`shower_to_wall`-type distances (wall = which wall?), and deghosting (which
compares segments along the drift axis — two clusters in *different* TPCs
can never ghost each other; make sure the pair loop respects that, both for
correctness and as a free 2× speedup).

**Vertex at/near the cathode.** A BNB vertex can sit near x≈0 where (a) the
CPA exclusion zones hide charge, and (b) the interaction's tracks may exit
into both TPCs (a *cross-TPC PR graph* — topology MicroBooNE never has).
The vertex finder must accept segments with mixed wpids attached to one
vertex. This is the qualitatively new case to test first (§4.3).

**Fiducial volumes.** MicroBooNE uses hand-drawn `PolyFiducial` XY/ZX
polygons (data + MC variants). SBND should start from simple per-TPC boxes
(the existing `DetectorVolumes` FV + margins) plus the cathode structure
exclusion, and only add polygon complexity if validation demands it. The
taggers consult `FiducialUtils` — wire SBND's fiducial composition into
`cm.fiducialutils()` (it takes a `fiducial:` reference; uboone passes
`uboone_mc_fid`).

**SCE / `clus_geom_helper`.** uboone passes a `SimpleClusGeomHelper` for
space-charge position corrections. SBND's SCE is smaller but not zero;
start with the helper disabled (`""`), record it as a known systematic, and
revisit once an SBND SCE map is available (kinematics vertex output has
`_corr` fields that expect it).

**ML retraining (required):** the SCN DL-vertex model is trained on
MicroBooNE topology, wire spacing and the uboone dQ/dx normalization
(`dQdx_scale=0.1`, `dQdx_offset=-1000` are part of the trained contract).
For SBND: run with `dl_weights=""` until an SBND training sample exists
(§7), then retrain (same SCN architecture; training input = fitted dQ/dx
point clouds + true vertex from MC) and re-derive scale/offset against the
SBND dQ/dx distribution. `qlport/dl_vtx_optimization/` holds the uboone
optimization logs as the template for the acceptance-threshold tuning
(`dl_vtx_min_accept_score`, `dl_vtx_top_k`).

**Recombination + particle data.** `ParticleDataSet` (dE/dx and range
tables) is detector-independent — reuse as is. The Box recombination model
is **not**: uboone uses A=1.0, B=0.255 at E=0.273 kV/cm; SBND operates at
0.5 kV/cm (use the ArgoNeuT/ICARUS-style modified-Box parameters at SBND's
field). This single constant feeds every energy number downstream — get it
in before any kinematics validation.

### 4.3 Validation plan (MicroBooNE-style)

1. **Function-level reviews exist** for every sub-component
   (`clus/docs/patternrecognition/*_review.md`: main_vertex, vertex_fitting,
   track_shower_sep, deghosting_kinematics, shower_clustering, ...). Any
   SBND-driven change re-opens the relevant review file; keep MicroBooNE
   numerically frozen (qlport `track_com_*.root` + `mc` bee_pf diffs).
2. **SBND MC truth metrics** (needs §7 simulation): vertex residual
   distribution (|reco−true|, fraction < 1 cm as the headline number),
   track/shower classification purity/efficiency vs truth, per-particle and
   neutrino energy closure (reco/true vs energy), all split by vertex-x to
   expose cathode/anode edge effects.
3. **Cross-TPC interaction test**: hand-pick (or truth-select) MC events
   with vertices within ~10 cm of the cathode and tracks entering both
   TPCs; verify a single PR graph with mixed-wpid segments and a vertex on
   the correct side.
4. **Bee hand-scans** of `vertices` + `track_fit` + `shower_track` + `mc`
   sets — the same scan protocol as the QL hand-scans (10-event batches,
   recorded verdicts), first on cosmic data (no vertexing expected — PR
   should degrade gracefully), then on BNB MC.
5. **Stability/determinism/cost**: identical reruns byte-identical;
   per-bundle wall time logged (`perf: true`) — NeutrinoID is the most
   expensive stage in the prototype, and SBND multiplicity makes the
   beam-window gating of §4.2 load-bearing.

---

## 5. Track trajectory and dQ/dx fitting

### 5.1 Algorithm overview

`TrackFitting` is the calorimetric core of the PR chain. It is **not a WCT
component** — it is a utility class instantiated by the taggers
(`TaggerCheckSTM`, `TaggerCheckNeutrino`) and passed into every
`PatternAlgorithms` function:

| piece | where |
|---|---|
| Core | `clus/src/TrackFitting.cxx` (~8350 lines), `TrackFitting_Util.cxx` |
| Parameters | `clus/inc/WireCellClus/TrackFitting.h` (`Parameters` struct), `TrackFittingPresets.h` |
| JSON preset (uboone) | `qlport/uboone_track_fitting.json` (~45 knobs) |
| Deep dive / reviews | `clus/docs/track_fitting.md`; `clus/docs/patternrecognition/do_single_tracking_review.md`, `do_multi_tracking_review.md` |

What it does, per segment (or jointly for all segments of a PR graph via
`do_multi_tracking`):

1. **Trajectory organization/fit** — order the segment's 3D points along
   the path (spacing control `low_dis_limit`, endpoint handling
   `end_point_limit`), fit a smooth trajectory.
2. **3D→2D projection** — for each trajectory point, predict its (wire,
   tick) signature on every plane, smearing by diffusion (`DL`, `DT`, drift
   distance) and the **SP software-filter widths** (`col_sigma_w_T`,
   `ind_sigma_u_T/v_T`, `add_sigma_L`).
3. **Charge measurement** — read measured charge ± uncertainty from cached
   per-event 2D charge maps keyed `(apa, face, plane) → (wire, tick)`;
   dead/close wires enter with reduced weight (`dead_ind/col_weight`,
   `close_ind/col_weight`); induction vs collection carry different
   uncertainties (`rel_uncer_ind/col`, `add_uncer_col` = 300 e⁻, ...).
4. **dQ/dx fit** — solve for the charge per trajectory point with a
   regularized least-squares (LASSO-style, `lambda = 0.0005`) such that the
   predicted 2D projections match the three planes' measurements — this
   shares charge correctly at overlaps and through dead regions.
5. **Calorimetry** — dE/dx via the recombination model; track KE by range
   (`ParticleDataSet` range tables) or by dE/dx integration; shower energy
   by charge integration.

The fitted result is written back to the segments (and to Bee as the
`track_fit` point set with dQ/dx coloring; ROOT output via the magnify
tracking visitor for the validation harness).

### 5.2 Integration attention points for SBND

**The defaults encode MicroBooNE's response — create
`sbnd_track_fitting.json`** (fork by duplication; do not edit the uboone
preset). Walk every knob and decide "geometry", "response" or "tuning":

- **Hardwired uboone drift constants.** `add_sigma_L = 1.428249 ×
  0.5505 mm / 0.5` literally contains uboone's tick-drift
  (1.101 mm/µs × 0.5 µs = 0.5505 mm). SBND: 1.563 mm/µs × 0.5 µs =
  0.7815 mm per tick. Audit every parameter with a length-per-tick or
  drift-speed flavor (`min_drift_time`, `time_tick_cut` are in ticks).
- **Diffusion**: uboone fit values `DL=6.4, DT=9.8 cm²/s`; SBND sim uses
  4.0/8.8 (`simparams.jsonnet`) and the data values should be measured.
  Note SBND's max drift (~201 cm) is shorter, so mis-set diffusion hurts
  less — but set it consciously.
  **DONE 2026-07-25, REVERTED 2026-07-27**: the fit was briefly moved to
  `DL=6.5781, DT=13.1349 cm²/s` (`sbnd_track_fitting.json`, and the sim was
  moved with it), then put back to sbndcode's **4.0/8.8** after the owner
  clarified with colleagues. Fit, sim and the official MC/data on disk now all
  assume one diffusion model. The retune is
  `47_stm-bragg-reference-sbnd-retune.md` §6a; the revert and its 1000-event
  validation are `66_diffusion-revert-validation.md` (**neither direction is
  bit-identical**).
- **SP filter widths** (`col_sigma_w_T`, `ind_sigma_u_T/v_T`): these mirror
  the smearing applied by *our* SP configuration. Re-derive from the SBND
  SP filter setup (same procedure as the uboone numbers; wire pitch is 3 mm
  in both detectors, and U/V at ±60°, so only the filter factors change).
- **Charge uncertainties** (`rel_uncer_*`, `add_uncer_col=300 e⁻`,
  `add_charge_uncer=600 e⁻`): tied to uboone noise and gain; revisit after
  comparing SBND SP charge resolution (the Q/L PE-error-style study, but on
  charge).
- **Recombination**: all dE/dx conversion goes through the model passed by
  the tagger — must be the SBND Box parameters at 0.5 kV/cm (§4.2).

**Two TPCs.**

- The 2D charge maps are already keyed by `(apa, face, plane)` — verify a
  cross-TPC segment gathers measurements from **both** APAs and that the
  per-point projection uses the wpid containing that point (drift sign,
  anode x, tick↔x mapping all flip between TPCs).
- A trajectory crossing the cathode traverses a few cm with **no
  measurements** (CPA structure + exclusion zones). The dead-region
  down-weighting machinery is the natural handler — confirm the cathode
  zone is represented as "dead" to the fitter, otherwise the LASSO will
  smear end-of-TPC charge into the gap.
- dQ/dx continuity across the seam is a brand-new closure check (same muon,
  two independent TPC calibrations): use it (§5.3).

**T0 correctness feeds dQ/dx directly.** The projection uses the cluster's
T0-corrected x to find the (wire, tick) of each point; a wrong `cluster_t0`
shifts the predicted ticks off the measured charge and the fit degrades
silently (low fitted charge, not an error). The QL-matching quality flags
should travel with the cluster so poorly-matched bundles can be excluded
from calorimetric studies.

**Units discipline.** The JSON preset values are in implicit units (mm,
ticks, electrons). When writing the SBND preset, comment every entry with
its unit and origin — this file will be the SBND calorimetry reference.

### 5.3 Validation plan (MicroBooNE-style)

1. **Parity harness already exists** — this is exactly what
   `qlport/run_5384.pl` + `wire-cell-uboone-magnify-tracking-convert` +
   `check_tagger_5384.pl` validate: per-event fitted trajectories and dQ/dx
   in `track_com_5384_*.root` compared prototype-vs-toolkit. Keep it green
   through any SBND-motivated refactor.
2. **Stopping-muon dQ/dx vs residual range** (data + MC): the canonical
   MicroBooNE calibration plot. Select STM-tagged (or geometrically
   selected) stopping cosmics, overlay fitted dQ/dx vs residual range on
   the muon prediction (tables + SBND recombination). Gets MIP scale,
   recombination and electronics gain right in one figure.
3. **Per-TPC and cross-TPC closure**: MIP dQ/dx peak separately for TPC0
   and TPC1 (must agree); dQ/dx step across the cathode for
   cathode-crossing muons (must be continuous within errors).
4. **Drift-dependence**: dQ/dx vs drift distance → electron lifetime
   consistency with the purity monitors; also exposes T0/diffusion
   mis-settings.
5. **MC truth closure** (§7 sample): fitted trajectory residuals vs true
   trajectory (transverse rms vs angle), fitted total charge vs true
   deposited charge per track, proton/muon KE by range vs truth.
6. **Determinism**: LASSO solves must stay run-to-run identical (same
   Eigen path; watch the compiler-FP tie lessons from QL matching).

---

## 6. Various event-selection taggers

### 6.1 Algorithm overview

After `TaggerCheckNeutrino` has built the PR graph, the **selection layer**
turns it into event-level decisions. Two parts:

**(a) Feature-filling taggers** (inside the `clus` PR chain, called from
`TaggerCheckNeutrino` / filling `TaggerInfo` ≈ 500 fields + `KineInfo`):

| tagger | file | purpose |
|---|---|---|
| Cosmic tagger | `clus/src/NeutrinoTaggerCosmic.cxx` | generic cosmic flags (direction, containment, solid angle) |
| NuMu tagger | `clus/src/NeutrinoTaggerNuMu.cxx` | νμ CC features (long muon, daughters) |
| NuE tagger | `clus/src/NeutrinoTaggerNuE.cxx` | νe CC features (shower dE/dx stem, gaps, MIP quality, π⁰ overlap, …) |
| Single-photon tagger | `clus/src/NeutrinoTaggerSinglePhoton.cxx` | NC γ features |
| SSM tagger | `clus/src/NeutrinoTaggerSSM.cxx` | short straight muon (KDAR-style) features |
| π⁰ kinematics | `clus/src/NeutrinoKinematics.cxx` (kine_pio_*) | two-shower invariant mass |

Per-tagger port reviews live in `clus/docs/tagger/` (one file per NuE
sub-block) and `clus/docs/patternrecognition/`.

**(b) BDT scorers + output** (in `root/`, optional pipeline entries):

| component | weights | output |
|---|---|---|
| `UbooneNumuBDTScorer` | 3 TMVA XMLs (`numu_tagger1/2/3`) + `cos_tagger_10` + XGBoost combiner (`numu_scalars_scores_0923.xml`) | `numu_score` + sub-scores |
| `UbooneNueBDTScorer` | ~29 TMVA XMLs (mipid, gap, hol_lol, cme_anc, br1/3/…, stem, …) + XGB combiner | `nue_score` + 30 sub-scores |
| `UbooneTaggerOutputVisitor` | — | writes `TaggerInfo`/`KineInfo` to ROOT |
| `UbooneMagnifyTrackingVisitor` | — | fitted-track ROOT (`track_com_*.root`) |

In `uboone-mabc.jsonnet` these are appended to the MABC pipeline only when
`numu_weights_dir` / `nue_weights_dir` / `tracking_output` are set — the
toggle pattern to keep.

### 6.2 Integration attention points for SBND

**All BDT weights are MicroBooNE-trained and must be retrained.** TMVA XML
+ XGBoost weights encode MicroBooNE feature distributions (energy scale,
cosmic rate, beam spectrum — BNB is shared, which helps, but the detector
isn't). The retraining needs the §7 MC (BNB ν + cosmics) plus
data cosmics, and a feature-extraction run of the toolkit chain itself —
i.e. **feature filling must be validated before any retraining starts**.
Plan of record:

1. Integrate and validate the feature-filling taggers with scorers OFF
   (weight dirs `""`).
2. Dump features with the tagger-output visitor; build the training sample
   from toolkit output (never from prototype/WCP output — the features must
   come from the same code that will run in production).
3. Retrain TMVA/XGBoost with the MicroBooNE recipe; version the weight
   files in `wire-cell-data` under `sbnd/`.

**Feature definitions that silently assume one TPC.** Audit (same per-wpid
checklist as §3/§4) the features that measure distances to detector
boundaries or use drift direction: `shower_to_wall`-type lengths (which
wall — the cathode is not a wall for a cross-TPC event), containment flags,
`gap_*` variables (wire-gap detection must consult SBND's dead-gap
registry, not uboone shorted-region lists), and cosmic-tagger
directionality (SBND is shorter in y and wider in z; the "enters from top"
priors differ). Each such feature should resolve geometry through
`DetectorVolumes`/`FiducialUtils`, never through constants.

**Naming/forking.** The scorers and output visitors are `Uboone*`
components in `root/` (some genuinely uboone-specific via input ROOT
formats). Follow the fork-by-duplication rule: create `Sbnd*` peers (or
detector-neutral renames where the code is already generic) rather than
threading SBND conditionals into the uboone components.

**Thresholds in feature code.** Many features apply hardcoded cuts in
physical units (energies in MeV, dQ/dx ratios). These travel with the
recombination/energy-scale work (§4.2, §5.2): wrong energy scale → silently
shifted features → garbage BDT input. Validate the energy scale *first*,
features second, BDTs last.

**Event-level plumbing.** SBND needs a decision per *beam-window bundle*,
not per event-with-one-candidate. Decide early where the per-bundle scores
land (cluster-scalar PCs next to `cluster_t0` is the natural place, so the
selection survives file I/O and reaches CAF-level consumers) — mirroring
how QL matching results are stored today.

### 6.3 Validation plan (MicroBooNE-style)

The MicroBooNE plan is written down in
`clus/docs/tagger/tagger_validation_plan.md`; SBND inherits its structure:

1. **Reasonableness (toolkit alone):** every `TaggerInfo`/`KineInfo` field
   within physical range, fill-gates consistent (variables only filled when
   their tagger ran). This is detector-neutral and runs on SBND immediately.
2. **Distributional parity (uboone):** since PR is functionally-equivalent
   but not bit-identical to the prototype, compare *distributions* of
   features and BDT scores prototype-vs-toolkit on the qlport sample — not
   per-event values. Keep this green while generalizing code for SBND.
3. **SBND truth-based ROC:** after retraining, efficiency vs cosmic
   rejection on MC (νμ CC, νe CC, NC, cosmics), the headline
   generic-neutrino-selection numbers MicroBooNE quotes (their generic
   selection: ~half the BNB νμ CC events at ~10⁻⁵ cosmic acceptance is the
   shape of the target, not the number to copy).
4. **Hand-scan tails:** Bee scans of highest-score cosmics (fakes) and
   lowest-score true ν (misses) — feature debugging happens here.
5. **Cross-check against the SBND production reco** (Pandora-based) on the
   same events: not an apples-to-apples metric, but large disagreement
   samples are efficient bug-finders in both directions.

---

## 7. Running SBND simulation

PR validation (track fitting above all) needs truth-known SBND MC. The
template is the ProtoDUNE two-stage **depo-file-split** recipe in
`DNN_ROI_SP/simulation/` — see `pdhd-sim-setup.md` (practised end-to-end,
1-event + 50-event batch) and its PDVD peers. Mirror that layout with
`stageA_sbnd/` + `stageB_sbnd/` peers.

### 7.1 Two routes to MC

**Route A — existing LArSoft-produced samples (already working).** The
`input-10files-mc` baseline (dnnsp frames + opflash tensors produced by
SBND production) runs through our full chain today via
`scripts/analysis/misc/build_mcbase_stage.py` + `scripts/runners/run_mcbase.sh` (event-id remap UID = 700000 +
file·1000 + evt, `SBND_INPUT_DIR` override). Best for: chain integration
tests, flash matching on MC, selection-level studies. Limitation: truth
lives back in the art files, so fit-vs-truth requires a separate truth
extraction pass.

**Route B — two-stage depo split (recommended for fitting validation).**

```
        ┌────── STAGE A (apptainer, stock sbndcode) ──────┐   ┌──── STAGE B (native, OUR toolkit) ────┐
GENIE/CORSIKA ─> LArG4 ─> SimEnergyDeposit ─> depos-evt<N>.tar.bz2 ─> wire-cell: drift + sim + SP ─> frames
```

Re-running Stage B (toolkit/config iteration) never re-runs GEANT4 — the
property that made the ProtoDUNE SP work fast, and exactly what track-fit
tuning needs. The depo file itself is the 3-D truth for trajectory/dQ/dx
comparison (DepoFluxSplat adds waveform-level truth if wanted, as in the
PDHD training-mode graph).

### 7.2 Stage A — sbndcode + depo extraction

Adapt `stageA_pdhd/` one-to-one:

1. `lar` chain inside the SL7 container: generator → g4. For the BNB
   physics sample use the standard sbndcode GENIE+cosmics (overlay) fcls;
   for a muon-only fit-validation sample, single-particle gen or CORSIKA
   cosmics suffice and are simpler.
2. Final step: a minimal fcl whose only producer is `WireCellToolkit`
   running the detector-agnostic 2-node graph
   (`wclsSimDepoSetSource → DepoFileSink`) — symlink
   `stageA/wcls-extract-depos.jsonnet` as PDHD does. **Verify the
   SimEnergyDeposit art tag in sbndcode** (PDHD/PDVD use
   `IonAndScint`; sbndcode's refactored LArG4 label may differ — check the
   reference detsim fcl the same way §A.4 of the PDHD recipe did).
3. **Light:** unlike the ProtoDUNE recipe (which disables PDFastSim for a
   ~50× stage speedup), SBND PR validation eventually needs flashes for
   QLMatching. Two-track approach:
   - *Fit-validation samples:* disable optical sim (fast), and inject
     truth T0 into the clusters (a small switch_scope-level knob: set
     `cluster_t0` from the true interaction time) so the PR chain runs
     without matching.
   - *Full-chain samples:* keep PDFastSim + the SBND opflash reco in
     Stage A, and convert the recob::OpFlash products into the
     `opflash_apa{N}.tar.gz` tensor format our flash loader reads
     (timestamp + per-PMT PE matrix + `frame_apply_at_caf` metadata; see
     `8_ql-chain.md`). This converter is new, small, and the only missing
     piece for MC flash matching.
4. Known Stage-A gotchas that will recur (all documented in
   `pdhd-sim-setup.md`): CORSIKA OSDF pre-warm **and** `ShowerInputFiles`
   pinning; the `ifdh` TMPDIR race under `xargs -P` batching (set per-job
   `TMPDIR`); resumability sentinels keyed to the *last*-written output.

### 7.3 Stage B — standalone wire-cell with our toolkit

The SBND configs already exist: `cfg/pgrapher/experiment/sbnd/`
{`params`, `simparams`, `sim`, `sp`, `nf`}.jsonnet (used by the
`wcls-sim-drift-*` in-LArSoft references). The Stage B graph is the PDHD
one with SBND imports: `DepoFileSource → DepoSetDrifter →
DepoSetFanout(2 APAs) →` per-APA `(sim.splusn → SP)`, plus an optional
DepoFluxSplat truth branch. Notes:

- **2 APAs (idents 0,1)**, tick 0.5 µs, 3427-tick readout, drift speed
  1.563 mm/µs — all inherited from `sbnd/simparams.jsonnet`; sim diffusion
  DL/DT = 4.0/8.8 cm²/s, lifetime 35 ms.
- **Check for required extVars** in `sbnd/params.jsonnet` before scripting
  (the PDHD recipe lost an evening to the no-default `elecGain` TLA).
- **NF:** skip for sim (matches the ProtoDUNE choice and our SP-input
  expectations); the sim digitizer output goes straight to SP.
- **SP output vs the chain's input convention.** Our imaging scripts
  consume `frame_dnnsp_<E>.npy` members (DNN-ROI SP from production). From
  standalone Stage B, start with **traditional SP (gauss)** packaged under
  the same member names — fine for PR validation, but record the
  difference; the production-faithful variant is to run SBND's DNN-ROI
  model in Stage B (needs the SBND `.ts` model + `dnnroi` plumbing like
  PDHD's) — that, plus `wirecell-gen morse-splat` smearing tuning if
  the truth branch is used, are the two known follow-ups.
- Package outputs per event under the `SBND_INPUT_DIR` conventions
  (`frames-dnn.tar.bz2`, optional `opflash_apa{0,1}.tar.gz`) so
  `run_img_evt.sh → run_clus_evt.sh → run_ql_evt.sh` (and the future PR
  step) run on MC unchanged — `scripts/analysis/misc/build_mcbase_stage.py` is the reference for
  the member naming and the event-uid remap.

### 7.4 Verification ladder

Mirror the PDHD practice-run discipline:

1. 1-event end-to-end practice first; record wall-clock, file sizes, and
   every hiccup in a `sbnd-sim-setup.md` peer of `pdhd-sim-setup.md`.
2. Plot check (a `make_pics`-style 2×N panel per APA): cosmic/track visible
   in raw + SP panels, truth aligned with gauss.
3. Push one practiced event through imaging → clustering → (matching or
   truth-T0) → Bee; hand-scan against truth depos.
4. Then the 50-event batch (`-P` with the TMPDIR fix), then the PR
   validation samples of §3–§6: stopping muons, BNB νμ CC, νe CC, NC,
   cathode-crossing topologies.

---

## Suggested order of work

1. **Simulation first** (§7): it gates every truth-based validation. Route
   A (existing MC) can start immediately; Route B in parallel.
2. **Retiling + Steiner** (§1, §2) in the all-APA MABC instance — they are
   pure infrastructure with crisp determinism/parity checks.
3. **Track fitting** (§5) with the SBND preset + recombination, validated
   on stopping cosmics from *data* while MC ramps up.
4. **STM tagger** (§3) — first physics consumer of the fit; cosmic data
   validation.
5. **Neutrino tagger** (§4) with `dl_weights=""`; truth metrics on MC.
6. **Selection taggers + retraining** (§6) last — they depend on everything
   above being stable.

At every step: keep the MicroBooNE qlport references byte-identical, keep
new behavior jsonnet-togglable (default OFF outside SBND), and extend this
document with what was actually observed.

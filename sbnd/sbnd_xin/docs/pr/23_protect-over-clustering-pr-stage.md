# doc pr/23 — PR-stage overclustering protection (`protect_bundle`): uboone's second graph examination, ported and made cathode-gap safe

**Status: IN PROGRESS.** Implementation landed (toolkit + runners, default OFF
everywhere); validation campaign V1-V5 and the production flip V6 pending.
**NOT bit-identical when enabled — that is the point of the stage** (doc
pr/22 §8 diagnosed its absence as the residual gap-jumping cause; the owner
requested the port 2026-08-02, accepting the result change).

## 0. Repro

```bash
# toolkit @ <commit set below>, wcp-porting-img @ <commit set below>
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild   # build+install

# compiled-config proofs (both PASS, see §2.3):
cd sbnd_xin
./compile_prjob_cfg.sh /nfs/data/1/xqian/toolkit-dev/toolkit/cfg /home/xqian/tmp/new.json
# vs the same at HEAD^ cfg => byte-identical (stage absent);
# add 'protect_bundle' after unmerge_assoc in the pipeline_names TLA and the
# compiled JSON gains one ClusteringProtectBundle:pr node (knob-on proof).

# enable in any runner (validation phase; production default still OFF):
SBND_PROTECT_BUNDLE=1 ./run_pr_chain_batch.sh <ql_root> <out_root> data <evt...>
./run_pr_evt.sh data -nu -protect <idx>
./run_nusel_evt.sh data -protect <idx>
# knob overrides: SBND_PROTECT_GRAPH=relaxed|relaxed_pid,
# SBND_PROTECT_REJOIN_XCUT/_DYZ/_DIS in cm (0 disables the cathode re-join).
```

## 1. What and why

Doc pr/22 §8: after the runner fix, 29.7 of 33.3 cm of evt 386948's residual
in-void `track_fit` trail is MST bridges between DISTINCT charge fragments
sharing cluster id 16 — the photons `Clustering_neutrino` merged into the nu
cluster. uboone never fit those bridges because a **second graph-examination
round** ran between Q/L matching and NeutrinoID:

- `WCPPID::Protect_Over_Clustering` (`pid/src/ProtectOverClustering.cxx:6-160`,
  called at `wire-cell-prod-nue.cxx:1322` on every beam-window bundle; the
  same `map_parentid_clusters` is what `wire-cell-prod-stm.cxx:815-830` reads).
- Per bundle member: `PR3DCluster::Examine_graph(ct_point_cloud)`
  (`data/src/PR3DCluster.cxx:2311`) rebuilds the graph keeping only
  inter-fragment bridges that pass `check_connectivity` against the 2D
  charge/dead-channel clouds, then splits at the surviving components.
- The main's largest component keeps the main cluster id
  (`ProtectOverClustering.cxx:57-121`); every other component becomes a new
  cluster in the same parent bundle (`:104-137`), fit separately by
  NeutrinoID (`wire-cell-prod-nue.cxx:1345,1360`), re-associated at shower
  level.

The SBND PR chain (`switch_scope, unmerge_bundle, unmerge_assoc, steiner, …`)
had no counterpart: the un-merges restore *pre-merge cluster boundaries* but
cannot split **within** a cluster id.

**The SBND-specific risk (owner's point 3): the cathode gap.** uboone had no
cathode plane inside the active volume; SBND's sits at x=0 (±0.45 cm physical
gap, ~4-5 cm apparent charge gap, ~1.1 cm transverse offset — doc pr/20).
The toolkit already states the consequence outright
(`ClusteringUnmergeBundle.cxx:292-297`): *"splitting on graph connectivity is
a clustering decision (it breaks cathode crossers, whose two halves the
relaxed graph does not join)"*. A verbatim port would undo the doc pr/20
cathode work (A1/A2 `cathode_connect` joins + B0 kink veto). Hence the one
deliberate divergence: a **cathode re-join pass**, knob-gated, default OFF in
C++ (prototype-faithful) and ON in the SBND config.

## 2. Implementation (landed, default OFF)

### 2.1 C++ — `clus/src/ClusteringProtectBundle.cxx` (new file)

`ClusteringProtectBundle` (`IConfigurable` + `Clus::IEnsembleVisitor`,
`NeedDV`+`NeedPCTS`), modeled on `ClusteringUnmergeBundle` (no production file
touched — fork-by-duplication stance). Per visit on the "live" grouping:

1. **Beam gate** (`beam_window_only` + `beam_window_low/high`, same keys and
   idiom as `CreateSteinerGraph.cxx:127-160`): gids of `Flags::main_cluster`
   clusters with `cluster_t0 ∈ [low, high)`; a member is any cluster sharing
   such a gid. Prototype scope: `to_be_checked` is built from beam-window
   flashes only (`wire-cell-prod-nue.cxx:1313-1320`), and **every member** of
   the bundle is examined, main and companions alike
   (`ProtectOverClustering.cxx:57-121` main, `:123-136` others).
2. **Component split**: `cluster->connected_blobs(m_dv, m_pcts, graph_name)`
   (the toolkit `Examine_graph`, `Facade_Cluster.cxx:3061-3078`).
3. **Cathode re-join pass** (the divergence; `cathode_rejoin_xcut <= 0`
   disables): per component pair, closest points via per-component
   `Simple3DPointCloud` (construction as `connect_graph_relaxed.cxx:71-82`);
   union the pair when both endpoints are within `cathode_rejoin_xcut` of
   `cathode_x`, 3D gap < `cathode_rejoin_dis`, transverse offset
   < `cathode_rejoin_dyz`. Union-find, lowest id kept.
4. Longest surviving component by `get_length()` keeps the retained cluster
   (prototype keeps largest by mcell count, `ProtectOverClustering.cxx:60-70`
   — recorded divergence #2, the established toolkit idiom from
   `ClusteringUnmergeBundle::groups_from_components`); fragments:
   `grouping.separate()`, `main_cluster` flag cleared, `associated_cluster`
   set, ident `alloc_ident(taken, main_ident*100 + sub_id)`, per-part
   `perblob` carve, blob-conservation error check — all the
   `ClusteringUnmergeBundle` idiom verbatim.

Determinism: no pointer-keyed iteration anywhere; component ids, `separate()`
maps, and the union-find are int-keyed; ties break to lowest id; MABC re-runs
`enumerate_idents('tree')` after every visitor.

Recorded prototype divergences (M15, decided by measurement in V1/V2):
- **graph flavor**: config `graph_name`, default `"relaxed"` — the documented
  `Examine_graph` mapping (`clus/docs/patternrecognition/examine_graph_review.md:47`)
  and what unmerge/recovering/examine_bundles use. `"relaxed_pid"` is the
  *structurally* closer port of the pid-stage graph (only it calls the ported
  `check_connectivity`, `connect_graph_relaxed.cxx:585-756`, with the ≤0.9 cm
  retry and `examine_middle_path`, no MST). V1 measures both.
- **main selection**: `get_length()` vs prototype mcell count (above).

### 2.2 Config

- `cfg/pgrapher/common/clus.jsonnet`: `protect_bundle(...)` builder next to
  `unmerge_bundle`, key-suppression on every null knob.
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet`: `cm_by_name.protect_bundle`
  (beam window threaded from the existing `beam_window` arg), new
  `protect_*` args on `clus_pr` and `pr()`; SBND `pr()` defaults =
  the operating point `protect_cathode_x=0`, `rejoin_xcut=5*wc.cm`,
  `rejoin_dyz=4*wc.cm`, `rejoin_dis=8*wc.cm` (**INTERNAL units** — unlike
  the cm-taking `cathode_kink_xcut` one block up; the doc pr/20 trap).
- `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`: matching TLAs +
  passthrough (both entry points set, the doc pr/20 "explicit null overrides
  the other file back to OFF" gotcha).
- **`pipeline_names` defaults unchanged everywhere** until V6: the stage acts
  only when named, so every existing pipeline is byte-identical.

### 2.3 Proofs at landing

- Build+install rc=0; `local/lib/libWireCellClus.so` mtime 2026-08-02 14:33 >
  source edit; factory symbol present (M1 freshness proof).
- `./build/clus/wcdoctest-clus`: **565/565 assertions, SUCCESS**.
- Compiled-config, stage absent: `compile_prjob_cfg.sh` (production 13-name
  pipeline) on HEAD-cfg vs edited cfg → **BYTE-IDENTICAL**.
- Compiled-config, stage named: pipeline gains exactly
  `ClusteringProtectBundle:pr` between `ClusteringUnmergeBundle:prassoc` and
  `CreateSteinerGraph:pr`; its `data` block carries
  `beam_window [200,2200)` + `graph_name relaxed` + rejoin 50/40/80 (internal
  = 5/4/8 cm).

### 2.4 Runner plumbing (this repo)

- `run_pr_chain_batch.sh`: `SBND_PROTECT_BUNDLE=1` inserts the stage after
  `unmerge_assoc`; `SBND_PROTECT_GRAPH`, `SBND_PROTECT_REJOIN_XCUT/_DYZ/_DIS`
  (cm, converted via `wirecell.jsonnet` to internal units) override knobs.
- `run_pr_evt.sh`: `-protect` flag (works with `-stm/-tgm/-nu/-dnn` or an
  explicit `-p` containing `unmerge_assoc`), same env overrides.
- `run_nusel_evt.sh`: `-protect` flag; refuses without the un-merges.
- Bare runs of all three remain the pre-pr/23 production chain until V6.

## 3. V1 — pilot on evt 386948 (pending)

## 4. V2 — cathode robustness and re-join tuning (pending)

## 5. V3 — nueCC48 track_fit vs shower_track census (pending)

## 6. V4 — broad impact (valfast 629) (pending)

## 7. Bee sets (pending)

## 8. V6 — production flip (pending)

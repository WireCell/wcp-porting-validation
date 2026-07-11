# Applying Wire-Cell pattern recognition to SBND

Working document for the campaign that takes SBND from "chain ends at charge-light
(Q/L) matching" to "full Wire-Cell pattern recognition (PR): taggers, neutrino
selection, track fitting".  This doc defines the intermediate file format written
after Q/L matching, the save/load wiring, and the step-by-step application of the
PR stages, and is updated (commit + push) at every milestone.

Companion docs (this doc is the working plan; those are background):
- `sbnd_xin/docs/PR_integration.md` — the 7-section survey of what needs porting.
- `sbnd/docs/qlport-to-sbnd-downstream-plan.md` — earlier v1 checklist for the
  downstream port; superseded where it disagrees with this doc.
- toolkit `clus/docs/pipeline_stages.md`, `clus/docs/pattern_recognition.md`,
  `clus/docs/patternrecognition/*_review.md` — per-algorithm prototype↔toolkit
  fidelity reviews.

## 1. Where we are, and the uBooNE analogy

Current SBND chain (per event, `sbnd_xin/run_ql_evt.sh` →
`wct-clus-matching-perevt.jsonnet`):

```
icluster-apa{0,1}-{active,masked}.npz     opflash_apa{0,1}.tar.gz
        │                                      │
  PointTreeBuilding ── per-APA MABC ── FlashTensorToOpticalPCs
        │                                      │
        └────────────── QLMatching ◄───────────┘        (per APA, or joint)
                            │
              [PointTreeMerging] ── all-APA MABC        (switch_scope, extend/regular,
                            │                            cathode_connect, examine_bundles;
                            │                            all with use_flash_t0)
                            ├── mabc-all-apa.zip (Bee)
                            └── TensorFileSink 'trash-all-apa.tar.gz' (dump_mode: NO-OP)
```

The fully-exercised PR chain exists only for uBooNE
(`qlport/uboone-mabc.jsonnet`).  There, the "intermediate file" is the WCP
prototype ROOT file (`nuselEval_*.root`: TC/TDC blobs, T_flash, T_match), i.e.
**final clusters + flash + cluster↔flash match**, and the toolkit job runs only
the PR tail as MABC pipeline visitors:

```
tagger_flag_transfer → clustering_recovering_bundle → switch_scope
  → steiner (retiler=improve_cluster_2) → fiducialutils
  → [tagger_check_stm]                 (STM/TGM tagger; slot before neutrino PR)
  → tagger_check_neutrino              (the full pattern recognition + taggers)
  → numu/nue BDT scorers → tracking/tagger output visitors
```

**SBND equivalent decided here**: the intermediate file captures the state
**after the all-APA MABC** (final merged, T0-corrected, flash-annotated
clusters — the same information content as uBooNE's WCP file), and a separate
"PR job" loads it and runs the PR-tail visitors.  A `TensorFileSink` already
sits at exactly this point in the graph (in `dump_mode`, writing nothing) — we
turn it into the real save.

## 2. The intermediate file format

### 2.1 Container

The Wire-Cell **Tensor-Data-Model (TensorDM) tar stream** written by
`TensorFileSink` and read back by `TensorFileSource` (package `sio`), holding
the point-cloud tree converted with `Aux::TensorDM::as_tensors()` /
`as_pctree()` (`aux/inc/WireCellAux/TensorDMpointtree.h`).  This is the same
representation that already flows in memory from QLMatching into the all-APA
MABC, so persisting it adds no new conversion code.

- File: `work/ql_evt<ID>/pctree-evt<ID>.tar.gz` (one file per event; `.tar.gz`
  compression chosen by extension).
- Members (prefix `clustering_`, ident = event number):
  - `clustering_tensorset_<ident>_metadata.json` — tensor-set metadata.
  - `clustering_tensor_<ident>_<idx>_metadata.json` + `..._array.npy` — one
    pair per tensor (tree nodes, point-cloud datasets, arrays).
- Datapaths inside the set: `pointtrees/<ident>/live` and
  `pointtrees/<ident>/dead`.

### 2.2 Required content (what PR needs)

Everything below is already present on the tree at the save point; the format
spec is the *inventory that must survive the round trip*:

| item | where it lives | consumer |
|---|---|---|
| live cluster/blob tree with `3d` sampled points (x,y,z, wire indices, charge) | `/live` tree, per-blob local PCs | everything |
| corrected coordinates (`x_t0cor`, `y_cor`, `z_cor` for data; `x_t0cor,y,z` sim) | per-point arrays added by switch_scope/pos-offset | PR runs in corrected scope |
| dead (2-view) tree | `/dead` tree | `FiducialUtils::inside_dead_region`, STM |
| `cluster_t0` (ns), `flash` (row idx), `matched_flash_gid` | `cluster_scalar` PC per cluster (written by QLMatching) | switch_scope, taggers, Bee op layer |
| `flag_main_cluster`, `flag_associated_cluster` | `cluster_scalar` (`set_flag` → `flag_*` scalar; QLMatching sets both) | STM/neutrino taggers pick the main cluster |
| other `flag_*` scalars (e.g. xtpc consistency) | `cluster_scalar` | tagger flag transfer / diagnostics |
| `opflash` PC (one row per flash×channel: gid, time, ch, pe) | root-node PC of the tree | Bee op layer, any light-aware tagger |
| run/subrun/event | tensor-set `ident` (event) + job TLAs (run/subrun) | Bee labeling, bookkeeping |

NOT in the file (verified at Milestone 1): the `flashpred` pcarray (per-PMT
predicted PE written by QLMatching on cluster nodes) is consumed by the
all-APA MABC's pre-pipeline Bee op dump and does not survive the T0-corrected
re-clustering.  Nothing downstream needs it; if a future tagger does, it must
be recomputed or carried explicitly.

Facade **flags are persistent by construction**: `set_flag(name)` writes a
`flag_<name>` entry into the cluster's scalar PC
(`clus/inc/WireCellClus/Facade_Mixins.h:164`), and scalar PCs are serialized
with the tree.  Runtime-only state that does **not** persist and must be
re-established by the PR job: the active *scope* (which coordinate arrays are
"default") and any attached utility objects (`FiducialUtils`, Steiner graphs).

### 2.3 Versioning

The tensor-set metadata JSON carries whatever we put in MABC's output; the PR
job must not assume more than the inventory above.  If the inventory grows
(e.g. per-cluster tagger scores), add a row to the table and note the date —
old files remain readable since TensorDM is self-describing.

## 3. Stage 1 — saving (Milestone 1)

Design (knob default-OFF, current outputs byte-identical):

1. toolkit `cfg/pgrapher/experiment/sbnd/clus.jsonnet`, `clus_all_apa(...)`:
   new argument `tensor_outname=''`.  Empty (default) keeps today's sink
   (`outname:'trash-all-apa.tar.gz', dump_mode:true` — a no-op).  Non-empty:
   `outname: tensor_outname, prefix:'clustering_', dump_mode:false`.
2. `sbnd_xin/wct-clus-matching-perevt.jsonnet`: new TLA `save_tensors=''`
   threaded into `clus_maker.all_apa(...)` in both the joint and per-APA
   branches.
3. `sbnd_xin/run_ql_evt.sh`: new `-save-pctree` flag →
   `--tla-str save_tensors=work/ql_evt<ID>/pctree-evt<ID>.tar.gz`.

Verification: run one **sim** and one **data** event with the flag on;
- inspect the tarball (member listing + a small python reader): live+dead
  datapaths present, `cluster_t0`/`flag_main_cluster` scalars present,
  `opflash` PC present;
- `mabc-all-apa.zip` content-identical to a run without the flag
  (`abtest/hash_archive.py`).

**Done (2026-07-10).** Implemented exactly as designed (`tensor_outname` in the
toolkit `clus_all_apa`, `save_tensors` TLA, `-save-pctree` flag).  Results on
sim evt12 (`run_ql_evt.sh mc -save-pctree 4`, joint) and data evt1258
(`data -save-pctree 2`):
- compiled config with the knob off is identical to before the change; with
  the knob on, only the sink block changes (`dump_mode:false`, real
  `outname`);
- `mabc-all-apa.zip` content hashes unchanged with the flag on
  (sim `bcd75f66…`, data `1426a3fc…`);
- tarballs `pctree-evt{12,1258}.tar.gz` (~1.3–1.4 MB): inventory checker
  `sbnd_xin/inspect_pctree.py` passes 10/10 on both — live+dead trees,
  `3d` x/y/z (18431 / 15981 points; data carries the extra `y_cor`/`z_cor`
  pos-offset arrays), `cluster_scalar` with `cluster_t0` (15/15 and 13/13
  clusters carrying a t0), `flash`, `matched_flash_gid`,
  `flag_main_cluster` (10 / 2 mains), `flag_associated_cluster`, root
  `opflash` PC (gid/time/ch/pe/apa/group).  Also carried: `ctpc_*`,
  `dead_winds_*` / `dead_gap_*`, `light`/`flash`/`flashlight`, `perblob`,
  `grouping_scalar` — i.e. the dead-region and optical context PR needs.

## 4. Stage 2 — loading + round-trip gate (Milestone 2)

New standalone PR job, `sbnd_xin/wct-pr-perevt.jsonnet` + driver
`sbnd_xin/run_pr_evt.sh`:

```
TensorFileSource(pctree-evt<ID>.tar.gz, prefix 'clustering_')
   → MultiAlgBlobClustering            (all-APA config: same anodes/DetectorVolumes/
      pipeline = []  (initially)        PCTransforms/Bee settings as clus_all_apa)
   → Bee zip  (+ optional re-save TensorFileSink)
```

Round-trip gate (both events):
1. The PR job's Bee zip is content-identical (`hash_archive.py`) to the
   original chain's `mabc-all-apa.zip` clustering + op layers — proves
   clusters, T0-corrected coordinates and flash association survived disk.
2. Re-saving the loaded tree and comparing member `.npy` content hashes to the
   original tarball's — proves the tensor round trip is lossless.
   (Never compare the `.tar.gz` bytes — tar members carry timestamps.)

Open items to confirm during implementation: MABC accepting an empty
`pipeline` list; Bee-layer coordinate selection when the corrected scope is
re-derived rather than inherited (re-run `switch_scope` if needed for gate 1 —
it recomputes deterministically from `cluster_t0`).

**Done (2026-07-10).** Implemented as `clus_pr` in the toolkit SBND clus
module (constructor `clus_maker.pr(anodes, pipeline_names, tensor_outname)`)
plus `sbnd_xin/{wct-pr-perevt.jsonnet, run_pr_evt.sh, compare_pr_roundtrip.py}`.
The gate needs **two runs** (both verified on sim evt12 + data evt1258):

- *Pass-through* (`./run_pr_evt.sh <mode> <idx>`, empty pipeline): re-saved
  tarball **member-content-identical to the input** (309 / 311 members) —
  the TensorDM serialization round trip is lossless.  (MABC accepts an empty
  pipeline fine.)  The Bee clustering layer is empty in this run — see the
  scope-filter finding below.
- *switch_scope* (`./run_pr_evt.sh <mode> -p switch_scope <idx>`): Bee
  clustering layer vs the Q/L job's `mabc-all-apa.zip`: `y/z/q/cluster_id`
  **exact** on all 18376 / 15903 points, dead-area layers byte-identical.
  `x` (= `x_t0cor`) differs on 819 / 176 points by ≤ 0.0084 / 0.0032 cm —
  only on flash-merged clusters, where the file carries the per-sub-cluster
  t0 correction from before `examine_bundles`' flash merge while the re-run
  applies the merged cluster's single t0.  Bounded by `flash_group_window` ×
  drift speed (80 ns × 1.563 mm/µs ≈ 0.013 cm); benign vs the 3 mm pitch.

Findings baked into the design:
1. **The per-cluster scope-filter flag is runtime-only.** The Bee dump's
   default `filter:1` keys on it, so a pass-through job dumps nothing; the
   PR pipeline must start with `switch_scope`, which re-establishes the
   corrected scope, the filter flags, and the default scope used by the
   per-point charge lookup (with the raw scope active, the dumped `q` values
   are wrong — nearest-neighbour lookups mix corrected coordinates with the
   raw-scope KD-tree).
2. **`Cluster::add_corrected_points` made idempotent** (toolkit
   `Facade_Cluster.cxx`: erase-before-add): on a reloaded tree the corrected
   arrays already exist and the plain `add()` threw.  No behavior change for
   any existing chain (the arrays never pre-exist there); gated by the
   uBooNE qlport byte-identity smoke.
3. **Only uniform per-cluster arrays survive serialization.** The
   concatenated named-PC encoding drops arrays present on only some clusters:
   `real_cluster_id` (written by `examine_bundles` on flash-merged clusters
   only, used by the Bee dump for pre-merge ids) is lost — the PR job's Bee
   layer shows `real_cluster_id == cluster_id` for merged clusters.  Display
   cosmetics only; no PR algorithm consumes it.  This is also why QLMatching
   materializes its flags on *every* cluster.

## 5. Stage 3 — first tagger: STM (Milestone 3)

`TaggerCheckSTM` (`clus/src/TaggerCheckSTM.cxx`) is functional on the toolkit
side: `visit()` calls `check_stm_conditions()` and sets `Flags::STM` (stopped
muon) or `Flags::TGM` (through-going muon) on the main cluster.  (The stub
warning in `clus/docs/patternrecognition/stm_tagger_review.md` is stale.)

PR pipeline for the demo, appended in `wct-pr-perevt.jsonnet` (uBooNE
ordering):

```
switch_scope → steiner (retiler=improve_cluster_2) → fiducialutils → tagger_check_stm
```

Prerequisites to assemble:
- **Recombination**: SBND `BoxRecombination` at E-field 0.5 kV/cm (uBooNE used
  0.273).
- **Particle dataset**: reuse `sbnd/particle_dataset.jsonnet`.
- **Retiler samplers**: `improve_cluster_2` needs per-(APA,face) samplers for
  SBND.
- **Track fitting parameters**: `sbnd_xin/sbnd_track_fitting.json` passed as
  `trackfitting_config_file` (see §6.2 — never rely on the C++ preset, which
  is uBooNE-hard-coded).
- `tagger_flag_transfer` / `clustering_recovering_bundle` are likely *not*
  needed initially: SBND QLMatching already sets `flag_main_cluster` /
  `flag_associated_cluster` (QLMatching.cxx:904-909), which is what
  TaggerCheckSTM keys off.  Revisit if the main-cluster lookup comes up empty.

Success criteria: PR job completes on the 2-TPC geometry for one sim + one
data event; STM/TGM flags appear on plausible clusters (log + Bee eyeball of
stopped-muon candidates); the existing Q/L chain outputs remain untouched.

**Done (2026-07-10).** `./run_pr_evt.sh <mode> -stm <idx>` runs
`switch_scope → steiner (retiler = ImproveCluster_2, SBND stepped samplers)
→ fiducialutils → tagger_check_stm` on the loaded tarball, with
`sbnd_track_fitting.json` (§6.2) and the SBND `BoxRecombination`
(E-field 0.5 kV/cm) + the shared `particle_dataset.jsonnet`.  Ran on sim
evt12 + data evt686/1258/1302/1346/1698/1720 — completes on the 2-TPC
geometry everywhere, every matched main cluster (up to ~13/event) gets its
own STM/TGM evaluation.

**First SBND STM tag: data evt1302 cluster 5** — a ~145 cm track entering
at the TPC1 anode boundary (x = 201.4 cm, outside the FV) and terminating
mid-detector at (105, 29, 281) cm; the full forward/backward dQ/dx
(Bragg-peak) fit ran (~0.5 s).  *(Correction at M5: originally recorded as
evt1346 — an idx↔event mix-up (data idx 3 = evt1302, idx 4 = evt1346); the
M5 re-runs reproduce the identical tag on evt1302 cluster 5.)*  Small mains exit early as fully-contained
("Mid Point: A") or lacking a steiner graph (steiner warns and skips
clusters yielding < 2 terminals — graceful).  `grep "TaggerCheckSTM:
cluster" work/pr_evt<ID>/wct_pr_evt<ID>.log` shows the per-main verdicts.

Three toolkit changes were needed (all default-preserving; uBooNE qlport
2-event smoke byte-identical, SBND Q/L zip hashes unchanged):
1. `TaggerCheckSTM::visit` now loops over **all** flagged main clusters,
   associating each with the sub-clusters sharing its `matched_flash_gid`
   (absent gid → all associated clusters, i.e. the uBooNE single-bundle
   behavior).  Also completed the removal of the `if(false)` dev-stub block
   that referenced the old single-main variable.
2. `CreateSteinerGraph` gained `require_beam_flash` (default true =
   uBooNE-only behavior).  SBND sets it false: QLMatching sets
   `main/associated_cluster`, never uBooNE's WCP-derived `beam_flash`, so
   with the default the steiner stage silently processed **nothing** —
   the flag-provenance gotcha materialized.
3. `ImproveCluster_1::remove_bad_blobs` no longer `.at()`-crashes when a
   cathode-crossing (two-APA) cluster retiles to a shadow cluster with
   blobs on only one APA (`std::out_of_range` on the (apa,face) lookup —
   a topology uBooNE could never produce).

Observation to revisit: TGM (through-going muon) fired on none of the ~50
mains scanned.  Likely real: the Q/L containment prefilter
(`require_containment`/`reject_overpred`) biases matched bundles against
full crossers, and TGM needs both boundary ends outside the FV.  Check
against hand-scanned crossers once a labeled sample exists.

Known 2-TPC risk points inside TaggerCheckSTM (audit if results look wrong):
- `dist_to_anode` falls back to `|x|` for points outside all volumes
  ("preserves UBooNE behaviour") — wrong side for TPC0 (−x drift) corners.
- Several kink-detection helpers hard-code `drift_dir_abs(1,0,0)` — fine for
  |drift|∥x, but sign-blind assumptions should be watched near the cathode.
- `shorted_y_w_range` is a uBooNE shorted-wire hack — leave unset for SBND.

Any toolkit C++ change made here must keep the uBooNE qlport smoke
byte-identical (events 6604/6821 vs the gate3 reference, `hash_archive.py`).

## 5.1 Runbook (per event)

```
./run_img_evt.sh  <mc|data> <idx>              # per-event imaging (existing)
./run_ql_evt.sh   <mc|data> -save-pctree <idx> # Q/L matching + pctree tarball
./run_pr_evt.sh   <mc|data> -stm <idx>         # PR job: STM tagger only (M3 gate)
./run_pr_evt.sh   <mc|data> -tgm <idx>         # PR job: TGM -> STM (M7)
./run_pr_evt.sh   <mc|data> -nu  <idx>         # PR job: TGM -> STM -> neutrino PR
./run_pr_evt.sh   <mc|data> -dnn <idx>         # -nu + SCN DL vertex (uBooNE wts)
# verdicts:  grep "TaggerCheckTGM: cluster" work/pr_evt<ID>/wct_pr_evt<ID>.log
#            grep "TaggerCheckSTM: cluster" work/pr_evt<ID>/wct_pr_evt<ID>.log
#            grep "TaggerCheckNeutrino:" work/pr_evt<ID>/wct_pr_evt<ID>.log
# display:   work/pr_evt<ID>/mabc-pr.zip (clustering + dead layers; with -nu also
#            track_fit / shower_track / vertices layers + the "mc" particle flow)
# gates:     ./run_pr_evt.sh <mode> <idx> && python3 compare_pr_roundtrip.py <ID> tar
#            ./run_pr_evt.sh <mode> -p switch_scope <idx> && python3 compare_pr_roundtrip.py <ID> bee
```

The pctree tarball is deleted whenever `run_ql_evt.sh` reruns without
`-save-pctree` (the Q/L job wipes its work dir) — regenerate before PR runs.

## 6. Later stages

### 6.1 Neutrino pattern recognition (`tagger_check_neutrino`) — DONE (M5, 2026-07-10)

Runs the full ported NeutrinoID on the beam-coincident bundle:
`find_proto_vertex → clustering_points → separate_track_shower →
determine_direction → shower clustering → determine_main_vertex → deghost →
improve_vertex → cosmic/numu/ssm/nue/singlephoton tagger features → FC check
→ track-fit charge assembly`.

**Wiring (M5).**  `clus_pr`'s `cm_by_name` gained `tagger_check_neutrino`
(sbnd_box_recomb, the shared particle dataset, `sbnd_track_fitting.json`,
`dl_weights=''`), plus the uBooNE-style Bee outputs keyed to the visitor's
full `type:name` (`'TaggerCheckNeutrino:pr'` — MABC matches the complete
pipeline-entry name, and clus_pr uses prefix `pr`): `track_fit`,
`shower_track`, `vertices` layers and the `mc` particle-flow JSON.  Driver:
`./run_pr_evt.sh <mode> -nu <idx>`
(= `-p switch_scope,steiner,fiducialutils,tagger_check_stm,tagger_check_neutrino`).

**Beam-bundle selection, not a multi-main loop.**  Unlike STM (independent
per-bundle verdicts), TaggerCheckNeutrino's downstream state is single-bundle
by construction (one PRGraph, one TrackFitting stored on the grouping, one
particle-flow tree).  The SBND generalization is therefore a *selection*:
new config `beam_window_low/high` (defaults 0/0 = gate disabled = uBooNE
single-main behavior, byte-identical).  Gate enabled: pick the flagged main
whose `cluster_t0` (= matched flash time) falls in the window — several →
longest wins, losers logged at INFO — and take as companions the associated
clusters sharing its `matched_flash_gid`.  No in-window bundle → clean skip
(logged).  The uBooNE `Flags::beam_flash` companion lookup is untouched.

**Beam window (empirical, from the 7 saved pctrees).**  `cluster_t0` is the
trigger-offset-corrected matched-flash time — NOT the raw opflash-npy
convention of `flash_t0_lan_reco2.py` (whose in-time peak sits at (−1.2, 0)
µs).  In-time matched bundles: MC evt12 +1.257 µs; data +1.38 / +1.69 /
+1.77 µs (evts 686/1698/1258).  Defaults in `run_pr_evt.sh`: **mc 0.5–2.0
µs, data 0.5–2.5 µs** (`-bw l,h` overrides).  Calibrate properly on a larger
sample before production; the multi-bundle "longest wins" rule is also an
open item (evt1698 exercises it: two mains share in-window gid 9).

**M5 results** (evt12 + 6 data events, all rc=0, no crashes on the 2-TPC
geometry; STM verdicts in the `-nu` pipeline unchanged, incl. the evt1302
cluster-5 tag):

| event | in-window bundle | outcome |
|---|---|---|
| mc 12 | cluster 4 (t0 1.257 µs, 5.7 cm) | full PR; 3 PR vertices, main vertex (−65.2, 141.9, 464.8) cm — in TPC0 (−x), machinery handles the negative-drift side |
| data 686 | cluster 7 (1.380 µs, 1.6 cm) | full PR on tiny in-time activity |
| data 1258 | cluster 8 (1.769 µs, 253.6 cm) | full PR on an in-window cosmic (expected: no topology gate) |
| data 1302 | none (13 mains) | clean skip |
| data 1346 | none (6 mains) | clean skip |
| data 1698 | cluster 3 (1.687 µs, 42 cm); cluster 2 (1.3 cm) logged as not selected | longest-wins rule exercised |
| data 1720 | none (10 mains) | clean skip |

Note evt12's beam bundle is small (5.7 cm, 0 associated) — its Q/L match was
already flagged as marginal (the xtpc-crosser mis-match study); vertex-truth
comparison belongs to the DNN/retraining milestone.

Gates: uBooNE 2-event smoke byte-identical vs gate3 (`m5smoke`); SBND Q/L
zip hash unchanged (`bcd75f66` sim); PR identity gates (tar member hashes
309/309, switch_scope Bee gate) pass; `-stm` verdicts identical to M3.

**Remaining opens for neutrino PR on SBND** (unchanged by M5):
- `dl_weights=''` for now (the SCN vertex network is uBooNE-trained; DNN
  demo + retraining plan = next milestones);
- audit of hard-coded axes: ~56 literal `(1,0,0)`/`(0,0,1)` direction vectors
  across `NeutrinoTagger{Cosmic,NuMu,NuE,SSM,SinglePhoton}.cxx`,
  `NeutrinoTrackShowerSep.cxx`, `NeutrinoVertexFinder.cxx`,
  `NeutrinoStructureExaminer.cxx`, `NeutrinoShowerClustering.cxx`,
  `NeutrinoOtherSegments.cxx`, `PRSegmentFunctions.cxx`.  Beam = +z holds for
  SBND; drift = +x does **not** (TPC0 drifts −x) — each use needs a per-wpid
  `face_dirx` replacement or a demonstration that only |projection| matters.
  Policy: fix per observed wrong verdict (M5 saw none), not wholesale;
- tagger *feature values* (numu/nue/ssm TaggerInfo) embed uBooNE-tuned
  constants (PID retune section, added at closeout) — they are filled but
  not quantitatively trustworthy yet;
- vertices near the cathode / cross-TPC clusters: the PR graph must not break
  at the x=0 seam (steiner bridging, dead-region handling at the CPA) — no
  in-window cathode-crosser appeared in the 7-event sample; test when one does.

### 6.2 Track-fitting smearing parameters (`sbnd_track_fitting.json`)

`TrackFitting` reads geometry (pitch, tick, wire angles, drift speed,
time offset) from DetectorVolumes/grouping at runtime — those need no port.
The **smearing sigmas are constants** with uBooNE factors baked in
(`clus/inc/WireCellClus/TrackFittingPresets.h`), overridable one-by-one via
the `trackfitting_config_file` JSON (45 keys; uBooNE template
`qlport/uboone_track_fitting.json`).

**Consistency requirement (key constraint): the smearing function must match
the software filter applied in signal processing.**  The track fit predicts
measured charge by smearing trajectory charge with the same effective
Gaussian that SP's deconvolution filters imprinted on the data.  The uBooNE
constants decode exactly to the uBooNE SP filter settings:

- `1.428249` µs = 1/(2π × 0.111408 MHz) — the uBooNE `Gaus_wide` time-domain
  filter width (charge extraction);
- `0.402993` = (1/√π)/1.4 and `0.188060` = (1/√π)/3.0 — the uBooNE
  `Wire_ind`/`Wire_col` wire-domain filter factors.

SBND's SP filters (`toolkit/cfg/pgrapher/experiment/sbnd/sp-filters.jsonnet`)
differ: `Gaus_wide` σ = 0.10 MHz, `Wire_ind` = (1/√π)×1.05, `Wire_col` =
(1/√π)×3.60.  So the SBND values are **derived from the SBND SP filters**,
not copied or pitch-scaled (values in WC internal units as in the JSON;
finalize + cross-check conventions against the prototype at Milestone 3):

| key | uBooNE value | formula (SP-filter-driven) | SBND value (candidate) |
|---|---|---|---|
| `DL` | 6.4e-7 | longitudinal diffusion | 6.2e-7 (SBND Q/L chain value, DL = 6.2 cm²/s) |
| `DT` | 9.8e-7 | transverse diffusion | 9.8e-7 |
| `add_sigma_L` | 1.5699937 | [1/(2π·σ_Gaus_wide)] × drift_speed | 1/(2π×0.10 MHz) × 1.563 mm/µs = 1.59155 × 1.563 = **2.4876** |
| `ind_sigma_u_T` | 0.3626937 | [(1/√π)/Wire_ind] × pitch_U × 0.3 | (0.564190/1.05) × 3 × 0.3 = **0.48359** |
| `ind_sigma_v_T` | 0.6044895 | [(1/√π)/Wire_ind] × pitch_V × 0.5 | (0.564190/1.05) × 3 × 0.5 = **0.80599** |
| `col_sigma_w_T` | 0.112836 | [(1/√π)/Wire_col] × pitch_W × 0.2 | (0.564190/3.60) × 3 × 0.2 = **0.09403** |
| `div_sigma` | 6.0 (0.6 cm) | Gaussian charge-division width | start at 6.0, revisit with fits |
| others (uncertainty/threshold/ratio knobs) | — | selection-tuned, not geometry | start with uBooNE values; retune on SBND fits |

The trailing per-plane factors (0.2/0.3/0.5) are empirical uBooNE tunings on
top of the filter width — keep them initially, revisit once SBND fit
residuals are available.  If the SBND SP filter settings change, this JSON
must be re-derived — note the dependency in any SP retune.

### 6.3 Selection BDTs and outputs

`UbooneNumuBDTScorer` / `UbooneNueBDTScorer` (TMVA/XGBoost weights trained on
uBooNE) and `UbooneMagnifyTrackingVisitor` / `UbooneTaggerOutputVisitor`
(update WCP-format ROOT trees) are uBooNE-specific by construction.  Plan:
skip them for SBND until (a) the PR feature variables are validated on SBND
MC, then (b) retrain BDTs on SBND simulation and fork `Sbnd*` output visitors
(fork-by-duplication, not shared helpers).  Truth-level validation needs the
SBND MC route documented in `PR_integration.md` §7.

### 6.4 DNN neutrino vertex (SCN) — demo DONE (M6, 2026-07-10)

The DL vertex path (`determine_overall_main_vertex_DL` →
`pyutil/python/SCN_Vertex.py`: torch state-dict `.pth`, sparseconvnet
`DeepVtx`, python embedded in the wire-cell process) needed **no C++ change**
for SBND: voxelization is min-subtracted at 0.5 cm resolution, so the network
input is translation-invariant and TPC0's negative x is harmless.  Only the
*charge* feature is detector-tuned (`dQdx_scale` 0.1 / `dQdx_offset` −1000,
uBooNE values).

**Environment repair.**  `import sparseconvnet` had rotted (built against an
older torch ABI; `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv`
vs torch 2.5.1+cu121) — every DL-configured run since had silently fallen
back to the geometric vertex via the try/catch (`W ... DL vertex failed:
SCN_Vertex: import failed` in the logs).  Fixed by rebuilding from source:
`pip install --no-build-isolation -e /nfs/data/1/xqian/toolkit-dev/SparseConvNet`.
If torch is ever upgraded, sparseconvnet must be rebuilt again (ABI pin).

**Byte-identity gate policy (important).**  The gate3 uBooNE reference was
produced while the import was broken, i.e. **gate3 is defined DL-off** — and
SCN/GPU inference is not guaranteed bit-stable anyway.  Therefore:
`qlport/uboone-mabc.jsonnet` now exposes `dl_weights` as a TLA (default = the
production uBooNE path; compiled config verified identical when unset), and
`scripts/run_one.sh` passes `-A dl_weights=` **empty by default** — gate
smokes stay DL-off and byte-comparable to gate3 forever (verified: m6smoke ==
gate3 with the working python env).  DL-on runs are functional demos:
`DL_WEIGHTS=uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth ./run_one.sh …`.

**uBooNE DL-on sanity** (evt 6604): SCN inference ran in-process (1119.7 ms,
matching the April `dl_vtx_optimization` timings; no failure warning) and the
re-ranked DL vertex agreed with the geometric one — Bee zip content hash
unchanged (`a17dd831`).

**SBND demo** (`./run_pr_evt.sh <mode> -dnn <idx>` = `-nu` + uBooNE weights):

| event | geometric main vertex (cm) | DL main vertex (cm) | note |
|---|---|---|---|
| mc 12 | (−65.2, 141.9, 464.8) | (−42.0, 136.4, 477.8) | DL accepted by re-rank; shifted ~27 cm within the PR graph |
| data 1258 | (−126.0, 158.4, 238.2) | (−35.6, 45.1, 328.4) | in-window 253-cm cosmic: DL picks a different spot ~180 cm along the track |

Both runs: SCN inference ~1.1 s, no import/inference failures.  **The
weights are uBooNE-trained and untuned on SBND — these vertex choices are a
plumbing demo, not physics.**  Production use needs: (a) SCN retraining on
SBND MC with truth vertices (route: `PR_integration.md` §7); (b)
`dQdx_scale`/`dQdx_offset` recalibrated to SBND charge scale; (c)
`dl_vtx_min_accept_score` revalidated.  Until then keep `dl_weights=''`
(geometric) as the SBND default — `-nu` does exactly that.

**LD_PRELOAD gotcha (SBND-only).**  The uBooNE qlport job loads
`WireCellRoot`, and ROOT pulls libpython in with global symbol visibility;
the SBND PR job loads no ROOT, so python C-extensions (`_ctypes`) fail with
`undefined symbol: PyTuple_Type`.  `run_pr_evt.sh` preloads
`libpython3.11.so.1.0` (from `sysconfig LIBDIR`) whenever DL is requested.

### 6.5 TGM tagger port (`tagger_check_tgm`) — DONE (M7, 2026-07-10)

The toolkit had **no dedicated through-going-muon tagger** — only the reduced
front/back-endpoint TGM branch inside `TaggerCheckSTM` (which, using the
no-inset `fiducial=dv`, misses crossers whose ends sit exactly on the active
boundary).  New component `clus/src/TaggerCheckTGM.cxx` ports the prototype
`WCPPID::ToyFiducial::check_tgm` (`prototype_base/pid/src/Cosmic_tagger.h:1331`,
walkthrough in `clus/docs/tgm/check_tgm.html`): pairwise extreme-point groups
(`get_extreme_wcps`), first-outside-FV point per group; CASE A both-ends-out
with ¼/½/¾ interior sampling (+ other-group waypoint re-check); CASE B
one-end-in with PCA gate, 30 cm local Hough direction, prolonged-signal
angles (<10°/10°/5° to U/V/W in the drift metric → `check_signal_processing`)
and `check_dead_volume`.  Pipeline entry `tagger_check_tgm`, placed
**after fiducialutils, before tagger_check_stm**; `TaggerCheckSTM` now skips
mains already flagged TGM (inert in all pre-existing pipelines).

Port decisions (all documented in the source header):
- **Both-TPC fiducial (the tricky bit).**  The FV in/out tests use a
  dedicated `BoxFiducial` `sbnd_pr_fv` spanning BOTH TPCs (the overall FV
  bounds, x −201.05..+201.05 cm) so a cathode-crossing track is not an
  "exiter" at x=0.  `fiducial=dv` is unusable here: `DetectorVolumes::
  contained()` is the union of per-face **sensitive** volumes, which excludes
  the CPA slab (|x| < 0.45 cm) and has no inset.  The metadata margins
  (2/2/2.5/2.5/3/3 cm) are applied via the tagger's `fv_tolerance`
  (FiducialUtils tolerance-vec convention, negative = inset).  Dead-region /
  signal-processing checks still use the grouping's FiducialUtils
  (per-(apa,face) logic, unchanged).
- `offset_x = 0`: runs post-`switch_scope`, coordinates already T0-corrected.
- The prototype's `main_flash->get_type()==2` beam protection → beam-window
  test on `cluster_t0` (same per-mode windows as §6.1).  The protected
  branches require the **unported** `check_neutrino_candidate()` → v1 never
  tags an in-beam-window bundle (conservative; port is a follow-up).
- U/V/W directions from `compute_wireplane_params` per (apa,face) — not the
  prototype's hard-coded ±60°; drift angle test uses |dir·x| so both SBND
  drift signs are handled.
- Multi-main loop from day one; `WCT_TGM_DEBUG=1` dumps per-pair CASE-A data.

**M7 results** (`./run_pr_evt.sh <mode> -tgm <idx>`; `-nu`/`-dnn` now run the
full chain `…,tagger_check_tgm,tagger_check_stm,tagger_check_neutrino`):

| event | TGM / mains | notes |
|---|---|---|
| mc 12 | 5/10 | nu bundle (cluster 4, in-window) protected; vertex from `-nu` identical to M5 |
| data 686 | 4/6 | |
| data 1258 | 0/2 | the 253-cm in-window cosmic is beam-protected (expected v1) |
| data 1302 | 7/13 | incl. 366-cm top→TPC0-anode and 370-cm top→bottom crossers |
| data 1346 | 4/6 | |
| data 1698 | 5/8 | |
| data 1720 | 7/10 | textbook 399-cm top→bottom crossers in both TPCs (clusters at x<0 and x>0) |

**The M3 "first STM tag" is reclassified.**  evt1302 cluster 5 — STM-tagged
in the STM-only pipeline — is per the TGM analysis a **366-cm through-going
track** entering at the top (y = 199.8 cm) and exiting at the TPC0 anode
(x = −201.3 cm), midpoint well inside.  `TaggerCheckSTM`'s own reduced TGM
check missed it because the no-inset `fiducial=dv` reads boundary points as
inside.  With M7's pipeline order the TGM stage vetoes it and STM correctly
does not fire — this is exactly why the prototype runs the cosmic tagger
before STM.  (The M3 §5 geometry description of this cluster could not be
reproduced from the surviving logs and should be disregarded.)  Across all 7
events the `-tgm` pipeline currently yields no STM tags; a genuine SBND
stopped-muon example awaits a larger sample.

Known v1 behaviors (prototype-faithful, revisit if they matter):
- Tiny anode fragments (0–12 cm mains whose few points all sit past the
  margin, e.g. from small t0 residuals) are tagged via the prototype's
  `ngroups==2 && both-ends-out && no-interior-point` branch.  Harmless for
  neutrino selection; a `min-length` knob would deviate from the prototype.
- No in-window TGM possible until `check_neutrino_candidate`
  (`2dtoy/src/ToyFiducial.cxx:1284`, Dijkstra + kink topology) is ported.
- No cathode-crossing main appeared among the 7 events' TGM candidates; the
  box-FV x=0 behavior is by construction but untested on a real crosser.

Gates: uBooNE 2-event smoke (m7smoke, DL-off) byte-identical to gate3; SBND
`-stm` pipeline reproduces the M3 verdicts exactly (incl. evt1302 cluster 5
STM=1 — the reclassification only happens when `tagger_check_tgm` is in the
pipeline).

### 6.6 PID retuning — DEFERRED (comments only, per request)

The PR/tagger code is functionally detector-generic but its **quantitative**
verdict thresholds and feature normalizations embed uBooNE calibrations as
literals.  Left as-is for now (M5–M7 demonstrate the machinery); retune when
SBND calibrated dQ/dx is available.  Inventory:

| constant | meaning | where (clus/src) |
|---|---|---|
| `43e3/units::cm` | MIP dQ/dx reference (uBooNE gain/E-field) | NeutrinoVertexFinder, PRSegmentFunctions, PRShower, NeutrinoTrackShowerSep, NeutrinoTagger{NuMu,NuE,SSM,Cosmic,SinglePhoton}, NeutrinoDeghoster, NeutrinoStructureExaminer, NeutrinoOtherSegments (~10 files, many sites each) |
| `50e3` | dQ/dx normalization in the STM kink finder | TaggerCheckSTM.cxx:827-836 |
| `0.8866 + 0.9533·(18 cm/L)^0.4234` | effective recombination-vs-length correction | NeutrinoTaggerNuE (4), NeutrinoTaggerNuMu (2), NeutrinoVertexFinder, NeutrinoTaggerSSM, NeutrinoTaggerCosmic |
| tagger feature cuts | numu/nue/ssm/cosmic TaggerInfo features feed uBooNE-trained BDTs | NeutrinoTagger*.cxx throughout |

What is already SBND-correct: `BoxRecombination` (Efield 0.5 kV/cm) for
charge→energy in track fitting; the SP-filter-derived smearing JSON (§6.2);
the particle dE/dx + range tables (detector-agnostic NIST/PDG).  The retune
plan: derive the SBND MIP dQ/dx scale from calibrated cosmics (ideally
expose `43e3`/`50e3` as one configurable scale), re-fit the length
correction, and only then trust the tagger features quantitatively (§6.3
BDT retraining depends on this).

**Hard-coded axis audit (status).**  ~56 literal `(1,0,0)`/`(0,0,1)` vectors
remain in the PR layer (files listed in §6.1).  Beam +z is correct for SBND.
Drift ±x: most uses are |projection| angle tests (sign-agnostic); the
per-cluster `face_dirx` pattern (`TaggerCheckSTM.cxx:1885`) is the fix
template where sign matters.  Policy: fix on observed wrong verdict — M5–M7
runs on both TPCs (vertices and TGM tags at x<0 and x>0) surfaced none.

### 6.7 Consolidated open items

1. `check_neutrino_candidate` port (Dijkstra path + kink topology,
   `2dtoy/src/ToyFiducial.cxx:1284`) — unlocks in-beam-window TGM tags.
2. Beam-window calibration on a larger sample (current values from 7 events;
   §6.1 provenance) + the multi-bundle "longest wins" selection rule.
3. SBND SCN vertex retraining + `dQdx_scale`/`dQdx_offset` recalibration
   (§6.4); SparseConvNet/torch ABI pin.
4. PID constants retune + BDT retraining + `Sbnd*` output visitors
   (fork-by-duplication) — §6.3/§6.6.
5. Cathode-crossing PR validation: a real x=0-spanning main through steiner/
   TGM/neutrino PR (none in the 7-event sample).
6. STM validation: find a genuine SBND stopped muon (the only tag so far
   reclassified as TGM).
7. Per-event Q/L match quality for MC evt12 (5.7 cm beam bundle) — affects
   any truth-vertex comparison.

## 7. Attention points (gotchas)

- **`dump_mode:true` writes nothing.** The `trash-*.tar.gz` `TensorFileSink`s
  in the shipped configs are no-ops; a real save needs `dump_mode:false`.
- **FiducialUtils is mandatory before any tagger.** `TaggerCheckSTM` and
  `TaggerCheckNeutrino` return silently (no-op) when
  `grouping.get_fiducialutils()` is null — always keep `fiducialutils` in the
  pipeline ahead of them.  Note even the uBooNE test config
  `clus/test/uboone-mabc.jsonnet` gets this wrong.
- **`inside_dead_region` needs the dead tree** — the intermediate file must
  carry `/dead`, and the PR job must load both trees.
- **Scope does not persist.** Re-run `switch_scope` at the head of the PR
  pipeline; it recomputes `x_t0cor` deterministically from `cluster_t0`.
  (Verified at M2: without it the Bee dump is empty — filter flags — and
  per-point charges resolve against the wrong KD scope.  On flash-merged
  clusters the recompute shifts `x_t0cor` by ≤ `flash_group_window` × drift
  ≈ 0.013 cm vs the saved per-sub-cluster values.)
- **Only uniform per-cluster PC arrays survive serialization** — an array
  present on some clusters only (e.g. `real_cluster_id` on flash-merged
  ones) is dropped by the named-PC concatenation.  Anything the PR job needs
  per cluster must be written on every cluster (QLMatching already does this
  for its flags).
- **Flag provenance.** `flag_*` scalars persist through serialization.  SBND
  QLMatching sets `main_cluster`/`associated_cluster`; it does **not** set
  uBooNE's WCP-derived event-type flags (`beam_flash`, `light_mismatch`, …).
  Anything that needs those must derive them (e.g. beam window on flash time)
  rather than expect them from the file.
- **tar.gz outputs are never byte-stable** (member timestamps).  All identity
  gates compare member/content hashes (`hash_archive.py` for zips, `.npy`
  member hashes for tensor tars).
- **RSE**: the standalone chain uses `rse_from_ident` (run/subrun = 0, event =
  tensor-set ident).  The PR job takes run/subrun/event TLAs for Bee labeling;
  do not try to recover run numbers from the tarball.
- **Smearing ⇔ SP-filter consistency.** The track-fitting smearing constants
  are not free parameters — they must match the software filters used in
  signal processing (§6.2).  Derive them from
  `sbnd/sp-filters.jsonnet` (`Gaus_wide`, `Wire_ind`, `Wire_col`) and
  re-derive whenever the SP filters are retuned.
- **uBooNE frame assumptions** in the PR layer (see §5/§6.1): `|x|` fallback,
  hard-coded `(1,0,0)` drift axes, `shorted_y_w_range`, and
  `examine_x_boundary`'s 257 cm default (already overridden by
  DetectorVolumes metadata in SBND configs).
- **Toolkit C++ gate**: any clus change must keep the uBooNE qlport outputs
  byte-identical (2-event smoke vs gate3; full 35-event sweep for anything
  touching shared algorithms).

## 8. Milestone log

| # | milestone | status | commits (toolkit / validation) |
|---|---|---|---|
| 0 | This document (format spec + roadmap) | DONE 2026-07-10 | — / 30f513f |
| 1 | Save: real `TensorFileSink` after all-APA MABC (`-save-pctree`) | DONE 2026-07-10 (sim evt12 + data evt1258, 10/10 inventory, Bee zips byte-identical) | 490b1b9a / a270eb6 |
| 2 | Load: `wct-pr-perevt.jsonnet` + round-trip identity gate | DONE 2026-07-10 (tar members identical; Bee y/z/q/id exact, x within flash-merge bound) | fd1522bf / bd601ed |
| 3 | STM tagger on loaded sim + data events | DONE 2026-07-10 (7 events; first STM tag: data evt1302 cluster 5 — corrected from evt1346 at M5 — anode-entering stopped-muon candidate) | 2b3d50df / 5535214 |
| 4 | Roadmap finalized for neutrino PR / BDT stages (§6) | DONE 2026-07-10 | — / eb3ea44 |
| 5 | `tagger_check_neutrino` wired (beam-bundle selection, geometric vertex, Bee PR layers) | DONE 2026-07-10 (7 events; §6.1 results table; all identity gates green) | 12e68f34 / 3a565f0 |
| 6 | DNN (SCN) vertex demo with uBooNE weights (§6.4) | DONE 2026-07-10 (sparseconvnet rebuilt; DL-off gate policy locked to gate3; evt12 + evt1258 DL-vs-geometric table; retraining required for physics) | — / d7f71c2 |
| 7 | TGM tagger port (`TaggerCheckTGM`, both-TPC box FV) (§6.5) | DONE 2026-07-10 (7 events; crossers tagged, beam bundles protected; M3 STM tag reclassified TGM; gates green) | 79edf84b / be0fa42 |
| 8 | Closeout: PID retune inventory (§6.6), axis-audit status, open items (§6.7) | DONE 2026-07-10 | — / (this commit) |

The full uBooNE chain now runs on SBND: pctree load → switch_scope →
steiner (retiled imaging) → fiducialutils → TGM → STM → neutrino PR
(geometric or DL vertex).  Remaining work is calibration/retraining, not
plumbing — see §6.7.

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

Known 2-TPC risk points inside TaggerCheckSTM (audit if results look wrong):
- `dist_to_anode` falls back to `|x|` for points outside all volumes
  ("preserves UBooNE behaviour") — wrong side for TPC0 (−x drift) corners.
- Several kink-detection helpers hard-code `drift_dir_abs(1,0,0)` — fine for
  |drift|∥x, but sign-blind assumptions should be watched near the cathode.
- `shorted_y_w_range` is a uBooNE shorted-wire hack — leave unset for SBND.

Any toolkit C++ change made here must keep the uBooNE qlport smoke
byte-identical (events 6604/6821 vs the gate3 reference, `hash_archive.py`).

## 6. Later stages (written down; NOT executed yet)

### 6.1 Neutrino pattern recognition (`tagger_check_neutrino`)

Runs the full ported NeutrinoID on the main cluster + associated clusters:
`find_proto_vertex → clustering_points → separate_track_shower →
determine_direction → shower clustering → determine_main_vertex → deghost →
improve_vertex → cosmic/numu/nue taggers`.  SBND prerequisites beyond §5:
- run with `dl_weights=''` (the SCN vertex network is uBooNE-trained);
- audit of hard-coded axes: ~56 literal `(1,0,0)`/`(0,0,1)` direction vectors
  across `NeutrinoTagger{Cosmic,NuMu,NuE,SSM,SinglePhoton}.cxx`,
  `NeutrinoTrackShowerSep.cxx`, `NeutrinoVertexFinder.cxx`,
  `NeutrinoStructureExaminer.cxx`, `NeutrinoShowerClustering.cxx`,
  `NeutrinoOtherSegments.cxx`, `PRSegmentFunctions.cxx`.  Beam = +z holds for
  SBND; drift = +x does **not** (TPC0 drifts −x) — each use needs a per-wpid
  `face_dirx` replacement or a demonstration that only |projection| matters;
- beam-window definition for SBND (BNB window on the flash time) so the
  "beam-flash-matched" gate selects the right bundles;
- vertices near the cathode / cross-TPC clusters: the PR graph must not break
  at the x=0 seam (steiner bridging, dead-region handling at the CPA).

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

| # | milestone | status | commits |
|---|---|---|---|
| 0 | This document (format spec + roadmap) | DONE 2026-07-10 | (this commit) |
| 1 | Save: real `TensorFileSink` after all-APA MABC (`-save-pctree`) | DONE 2026-07-10 (sim evt12 + data evt1258, 10/10 inventory, Bee zips byte-identical) | toolkit + validation, see below |
| 2 | Load: `wct-pr-perevt.jsonnet` + round-trip identity gate | — | |
| 3 | STM tagger on loaded sim + data events | — | |
| 4 | Roadmap finalized for neutrino PR / BDT stages | — | |

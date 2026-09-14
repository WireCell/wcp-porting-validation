# doc 108 — improving the SBND tracking-pr ROOT output: is the information there, and does truth need reco1?

**Status:** plan only (2026-09-14, owner request). **No code is changed.** Every
proposed change below is a default-OFF knob with a byte-identical knob-off gate
(CLAUDE.md §1); none is implemented. Follows [doc 107](107_mc-truth-and-tagger-consistency.md),
whose analysis had to reconstruct by hand what these changes would record.

## Repro block

```bash
cd sbnd_xin
python3 d108_cluster_id_census.py      > docs/108_logs/cluster_id_census.txt      # sec 2.1 (13 216 files, ~1 min)
python3 d108_tagger_branch_census.py   > docs/108_logs/tagger_branch_census.txt   # sec 4.4 (1 500-file sample)
python3 d108_bee_label_match.py        > docs/108_logs/bee_label_match.txt        # sec 3.4 (needs products/d107)
python3 d108_reco1_probe.py            > docs/108_logs/reco1_probe.txt            # sec 3.5 (needs products/d107)
W=/home/xqian/work/WC_FM_Sim/runs
python3 d108_reco1_probe.py --products-only $W/{nue,numu,nue_smeared,numu_smeared}/sp_*_reco1.root $W/nue/g4_nue.root \
                                       > docs/108_logs/reco1_probe_wcfmsim.txt    # sec 3.5 second set (+ prod10k* file counts appended)
root -l -b -q 'd108_reco1_root_probe.C("/nfs/data/1/yuhw/2025-fall-prod-sample/round2-patrec/mc_paths-v10_14_02_03-100files/reco1-detsim-g4-gen-Gen2_2026-a6a0-2395-0263-6d62.root")' \
                                       > docs/108_logs/reco1_root_probe.txt       # sec 3.5
python3 d107_tables.py products/d107                                              # sec 2.1 association split
```

Code citations are against toolkit `apply-pointcloud` @ `d3b398fc`, the production
entry `sbnd/wcls-img-clus-matching-xin.jsonnet` @ `8ca6866d` (this repo), and the
truth labeler `HaiwangYu/larwirecell` branch `dev-v10_14_02_02` @ `a02a1a4d`
(`larwirecell/aiml/TensorSetLabeler.{h,cxx}`, `docs/TensorSetLabeler-notes.md`; not
checked out locally, read through the GitHub API).

---

## 0. Answers

| Group (from the owner discussion) | Information available at write time? | Blocker / caveat |
|---|---|---|
| **1** record what happened (both cluster ids, full activity roster, `has_vertex`, per-bundle tree incl. no-candidate) | **Yes** — all inside `TaggerCheckNeutrino` | items (a) and (b) to verify before implementing (sec 2.5) |
| **2** fix wrong/mislabelled fields (T_cluster flags, `flash_id`, RSE per row) | **Yes** | `beam_flash` has no SBND source; derive or drop (sec 4.1) |
| **3** truth in ROOT | **Yes, without reco1** — the production chain already delivers truth *into* the PR node | a transport step to the ROOT visitors, and a trackid→interaction map for secondaries (sec 3.3); per-blob trackid survival through the PR splits still to verify (sec 2.5 c) |
| **4** self-describing files | **Mostly** | a compiled-config hash is not obtainable in-process; must be injected (sec 5) |

**Reco1 question.** The truth for this production does **not** need the reco1 files:
the entry jsonnet runs `clus_all_apa → labeler_truth → pr_node → labeler_tagger`, so
the node that writes `tracking-pr.root` already receives the GENIE interaction
metadata (signed `nu_pdg`, CC/NC, mode, Eν, vertex, Edep) and a per-blob G4 `trackid`
(sec 3.1–3.2). There are two reco1 sets on disk (sec 3.5).
- **yuhw's SBND set:** **10 of the 1 000** production inputs (104 of 13 217 events, all
  run 713, one true νeCC in the FV). It is a good **development and validation** sample
  for a lar run of the chain, with a ready reference (the production Bee and tracking-pr
  of the same 104 events). It is not a truth source for the sample, and it is not
  readable without LArSoft on this machine.
- **`/home/xqian/work/WC_FM_Sim`:** **DUNE FD-HD** GENIE-CC files (4 × 20 events). Wrong
  detector for this chain, but proof that lar runs on `wcgpu1` in the SL7 container.

**Interim, for the existing 13 216 events:** the production Bee zips already carry
per-blob truth labels; on 25 T_tagger candidates they give a clean charge-based match
(sec 3.4), so a Bee-based offline truth match can be done now without any rerun.

**Corrections to what I said in chat before this investigation:**
- `UbooneTaggerOutputVisitor` is shared with the **PDHD and PDVD** PR configs
  (`pdhd/pr.jsonnet:2010`, `protodunevd/pr.jsonnet:2038`), not with uBooNE's
  (no `cfg/pgrapher/experiment/uboone/` config uses it).
- `T_tagger.cluster_id` is not "wrong": it is where the PR result lives; the
  selected activity is a different question. Both are needed (sec 2.1).
- the 25-candidate label test: I said "23 of 25 matched to one interaction at
  88–100%". The logged tally is 24 of 25 neutrino-dominated (88–100%) plus 1
  dominated by non-neutrino labels (sec 3.4).

---

## 1. Context

### 1.1 Who writes what

| Tree | Writer | Rows |
|---|---|---|
| `Trun`, `T_bad_ch`, `T_cluster`, `T_rec_charge`, `T_proj_data` | `root/src/SbndPrMagnifyTrackingVisitor.cxx` (SBND only: `sbnd/clus.jsonnet:2472`) | per event / per cluster / per point |
| `T_tagger`, `T_kine` | `root/src/UbooneTaggerOutputVisitor.cxx` (`common/clus.jsonnet:813`; used by SBND, PDHD, PDVD) | one per bundle with a candidate |

Both are `IEnsembleVisitor`s inside the PR `MultiAlgBlobClustering` (`clus_pr`) and see
**only the Ensemble** (no tensor-set metadata, no other component's config).

### 1.2 Implementation constraints for every item

- Default-OFF knob, key-suppression idiom in jsonnet, compiled SBND/PDHD/PDVD
  configs byte-identical with the knob off (the tagger writer is shared).
- Knob-off gate on the SBND standard manifest; new code ships with doctests
  (`wcdoctest-clus`, `wcdoctest-root`).
- Anything in larwirecell (sec 3.3 option L) is outside this repo and needs the
  labeler owner.
- Noted, out of scope here: the two writers use one fixed filename (RECREATE vs
  UPDATE), so a multi-event lar process silently loses T_tagger/T_kine
  (`wcls-img-clus-matching-xin.jsonnet:254-271`); packaging is not part of this plan.

---

## 2. Group 1 — record what happened

### 2.1 Selected activity vs final cluster

Measured on all 6 236 T_tagger rows (`108_logs/cluster_id_census.txt`):

| `cluster_id` relation to `act_cluster_id[argmax(act_is_selected)]` | rows | `cluster_id` in T_cluster | reco vertex < 5 cm of a true vertex* | median distance* |
|---|---|---|---|---|
| same | 5 748 | main 5 553, associated 195 (demoted-main fallback) | 69.1% | 1.7 cm |
| another `act_*` entry | 109 | associated 109 | 43.1% | 12.3 cm |
| not in `act_*` | 379 | associated 379 | 49.3% | 5.4 cm |

\* from `d107_tables.py` (nearest-true-vertex association, doc 107 sec 4.5).

In all 488 differing rows `T_kine.cluster_id`/`nu_index`/vertex equal the T_tagger row,
and the cluster's T_cluster `flash_id` equals the row's `matched_flash_gid`: the vertex
search moved the main cluster onto an associated companion **of the same flash bundle**.

**Where the information is** (`clus/src/TaggerCheckNeutrino.cxx`):
- selection: `cand.main` from `pick()` (:2251-2299), `is_selected` stamped at :2305-2307;
  `candidates[nu_index].main` stays in scope; the loop copies it into the local
  `main_cluster` (:2463) and never reassigns the vector element.
- the move: `determine_overall_main_vertex_DL` takes `Cluster*& main_cluster` and calls
  `swap_main_cluster` with **no knob** (`NeutrinoVertexFinder.cxx:5391`); the traditional
  path swaps only under `main_vertex_swap_apply` (:3236-3250; SBND sets false,
  `wct-pr-perevt.jsonnet:2896`). SBND's `pr()` defaults `dl_weights` to the SCN net
  (`sbnd/clus.jsonnet:981`), so the DL path is the likely source — confirm on the
  production config before relying on it (sec 2.5).
- the write: `tagger_info.cluster_id = main_cluster->get_cluster_id()` (:3547, post-move).

**Proposal (knob `nu_row_provenance`, both writers):** add `sel_cluster_id`
(= `candidates[nu_index].main`), `vertex_moved_cluster` (`sel != final`); optional
`sel_vertex_{x,y,z}` (sec 2.5 item a).

### 2.2 Full activity roster

`act_*` lists only the bundle's in-window mains and demoted mains (`add_acts`, :2230-2247).
Companions are collected afterwards into `NuCandidate::others` (:2310-2324; decl :1909),
with TGM/STM companions ≥ `cosmic_companion_min_length` dropped. In 379 rows the final
cluster therefore has no roster entry.

**Proposal:** append companions to the roster with `act_role` (0 main, 1 demoted main,
2 companion, 3 companion dropped as cosmic), `act_is_final` (the post-move cluster), and
`act_evaluated = 0` where the Q-L taggers never ran. All values exist at :2310-2324 and
:3547.

### 2.3 `has_vertex`

`kine_nu_{x,y,z}_corr` default to 0 (`NeutrinoTaggerInfo.h:27-29`) and are filled only
`if (final_main_vertex)` (:3733-3741); no "vertex found" flag exists. (0,0,0) is a point
on the SBND cathode plane; 160 rows carry it (doc 107). **Proposal:** `has_vertex =
(final_main_vertex != nullptr)` in T_tagger and T_kine.

### 2.4 One row per flash bundle, including bundles with no candidate

**Why 7 078 events have no T_tagger/T_kine at all:** with no candidate,
`TaggerCheckNeutrino` returns at :2367-2371 before `set_track_fitting` (:3829-3832);
`UbooneTaggerOutputVisitor` then returns at :73-76 (`if (!tf) return`) before it opens
the file (:98).

**Rejection reasons known at the time:**

| Code | Reason | Where | Recorded today? |
|---|---|---|---|
| 0 | selected | :2305 | yes (row) |
| 1 | demoted-main fallback selected | :2297-2299 | only via `act_is_demoted` |
| 2 | every main/demoted cosmic-tagged (TGM / STM / `lm_flag>0`) | `pick()` :2254-2263 | log only |
| 3 | below `nu_per_bundle_min_length` and not the legacy winner | :2285-2291 | log only |
| 4 | not STM under `nu_per_bundle_stm_only` (PDVD mode) | :2271-2276 | log only |
| — | event: no main/demoted cluster, or none in the beam window | :2207-2211 | counters `n_main_clusters`, `n_in_beam_clusters`, log only |
| — | in-window cluster with `matched_flash_gid < 0` | :2213 | **silently dropped; no gid, no counter** |

`cand.gid` and the complete `cand.acts` exist before `pick()` runs; they are discarded at
the `continue` (:2303).

**Proposal (knob `nu_bundle_census`):**
1. `TaggerCheckNeutrino` keeps each gid's reason code plus roster, and event counters
   including a new count of in-window `gid<0` clusters. It publishes them as numeric
   arrays on a grouping- or ensemble-level local PC (`put_pcarray`, the `Mixins::Cached`
   helper the Ensemble also has, `Facade_Mixins.h:~300`, `Facade_Ensemble.h:37`), before
   the :2367 return.
2. The visitor's `!tf` guard is relaxed under the knob; it writes `T_bundle` (one row
   per gid: gid, flash time, reason code, roster arrays, the T_tagger row index when
   selected) and event counters into `Trun`.

**Expected knob-on check:** events with no T_tagger row = 7 078 on the production sample,
each explained by a `Trun` counter or `T_bundle` reason.

### 2.5 Verify before implementing

- **(a) Selection-time vertex.** `determine_main_vertex` writes
  `map_cluster_main_vertices[main_cluster]` (:3147) before the overall-vertex step, and
  the DL swap writes the new key after swapping (`NeutrinoVertexFinder.cxx:5391-5396`).
  Whether the original entry survives every later write was **not audited**. If it does
  not, capture the position in a local at :3147.
- **(b) Which path moved the 488 clusters.** Confirm from the compiled production config
  (`dl_weights` resolved, `main_vertex_swap_apply`) or a debug log of one moved event.
  This decides where `vertex_moved_cluster` gets set.
- **(c) Per-blob `trackid` survives into `clus_pr`'s clusters** (the premise of 3b,
  sec 3.3).
  - **Evidence today is secondhand:** the entry jsonnet's comment
    (`wcls-img-clus-matching-xin.jsonnet:343-351`). The production ROOT and Bee products
    cannot show it: the Bee label layer is written by `labeler_truth`, upstream of the PR
    splits.
  - **Check on the first lar validation run:** in the tracking visitor, count the final
    candidate cluster's blobs whose `scalar` PC has a `trackid` array with values ≠ -1.
    Expect the labelled fraction to match the event's label rate (notes §1: 59–97% of
    blobs).

---

## 3. Group 3 — truth in ROOT

### 3.1 The production chain already attaches truth upstream of the PR node

`sbnd/wcls-img-clus-matching-xin.jsonnet:336-458`:

```
clus_all_apa ─ labeler_truth ─ pr_node (clus_pr: taggers, BDT scorers, tracking_visitor, tagger_output) ─ labeler_tagger ─ tail_dump
```

- **`labeler_truth`** (:388-401): `label_blobs: true`, `bee_sets: ['truth','sed']`,
  `pf_metadata_key: 'bee_pf_truth'`, `pf_nu_only`/`truth_tracks_nu_only: true`,
  `pf_ke_min: 10 MeV`.
- **`labeler_tagger`** (:410-418): `label_blobs: false`, tagger Bee sets only.
- **Production log** (`logs/fail_*.log`): both instances configured, together with
  `MultiAlgBlobClustering:clus_pr`, `UbooneTaggerOutputVisitor:pr` and
  `SbndPrMagnifyTrackingVisitor:pr`.
- **Merged `mc.json`** in every production Bee zip (doc 107): it can exist only because
  `clus_pr` read the truth tree from its input metadata.
- **Why truth sits upstream:** the jsonnet comment (:343-351) says truth attached
  upstream is independent of every PR knob, and that per-blob `trackid` survives the PR
  splits, since `switch_scope` erases only cluster-level `perblob` and `separate()`
  moves blob nodes wholesale.

### 3.2 What reaches `clus_pr`, and what the ROOT visitors can see

| Truth item | Written by the labeler | Reaches `clus_pr`? | Visible to the ROOT visitors today? |
|---|---|---|---|
| **Event metadata** `n_nu`, `nu_idx`, `nu_pdg` (signed), `nu_ccnc`, `nu_int_type`, `nu_energy` [GeV], `nu_vtx_{x,y,z}` [cm], `nu_flavor`, `nu_edep` [GeV] — parallel arrays, one entry per beam-ν interaction | `TensorSetLabeler.cxx:534-583`, set at :1730-1740 | yes: `m_in_metadata = ints->metadata()` (`MultiAlgBlobClustering.cxx:3821`), forwarded at :4282 | **no**: private member; `set_scalar` on the Ensemble is single-valued (:3914-3916) |
| **`bee_pf_truth`**: jsTree of beam-ν particles with KE > 10 MeV. Node id = trackid (1e7/2e7 G4-instance offsets), top node 9000000+`nu_idx` | :1734-1737 | yes; merged into Bee `mc.json` (:3044-3081) | **no** (JSON) |
| **Per-blob `trackid`**: dominant G4 track per live blob, `abs()`-folded delta rays, SCE+drift+smear aware, -1 = ghost/unmatched | :1222-1238 into blob `scalar` PC | yes (generic pctree load) | **yes**, via each cluster's blob `local_pcs()["scalar"]`; nothing reads it yet |
| **`truth_per_track`** tensor [ntracks × 23] (`trackid`, `pdg`, mother, start/end 4-vectors, `nu_idx`, `process`); with `truth_tracks_nu_only` = beam-ν **primaries only** | :1708-1726 | in the input tensor set, but **dropped**: MABC loads only its named groupings (:3919-3920) and serializes only those (:4128-4282) | no |

### 3.3 Plan

**3a — `T_truth`, one row per interaction (knob `nu_truth_tree`).**
- **Publishing:** under the knob, MABC copies the `nu_*` parallel arrays from
  `m_in_metadata` into an ensemble-level PC dataset with its own PC name (for example
  `nu_truth`), because all arrays in one PC share a major size. It uses `put_pcarray`.
  Every field is numeric except `nu_flavor`, which is redundant with `nu_pdg`.
- **Writing:** the SBND tracking visitor writes `T_truth` with run/subrun/event, `nu_idx`,
  `nu_pdg`, `nu_ccnc`, `nu_int_type`, `nu_energy`, `nu_vtx_*`, `nu_edep`.
- **Data events:** they get an empty tree, since the labeler skips MC reads there.
- **Knob-on check:** equality with the Bee `mc.json` truth nodes for every interaction of
  the production sample. This also adds the neutrino sign, which the Bee text lacks.

**3b — per-candidate truth match (knob `nu_truth_match`).**
- **New T_tagger/T_kine branches:** matched `nu_idx`, completeness and purity (charge),
  labelled-charge fraction, and dominant trackid.
- **Algorithm:**
  - For the final cluster (and the selected one, sec 2.1), sum blob charge by blob
    `trackid`, then map trackid → `nu_idx`.
  - Purity = matched-interaction charge / cluster charge.
  - Completeness = matched-interaction charge in the cluster / that interaction's labelled
    charge over all live blobs of the event (all blobs are in the `live` grouping).
- **Owner decision needed: the trackid → `nu_idx` map for secondaries.**

  | Option | Source | Coverage | Cost |
  |---|---|---|---|
  | **T** | flatten `bee_pf_truth` (node id → top node) into numeric pairs in MABC | particles with KE > 10 MeV only; lower-energy daughters fall to "unmapped" | WCT only |
  | **P** | load `truth_per_track` in MABC | **primaries only** (`truth_tracks_nu_only`); every shower/daughter trackid unmapped, unless the labeler runs with `truth_tracks_nu_only: false` (full largeant table, larger) | WCT only (+ config) |
  | **L** | labeler stamps `nu_idx` next to `trackid` in each blob `scalar` PC (it already builds the beam-ν trackid set for the nugraph HDF5, notes §2.11) | exact | larwirecell change (outside this repo) |

  Recommendation: **T** now (WCT-only; the sec 3.4 test shows the unmapped fraction on
  selected clusters is small), **L** as the exact long-term source.
- **Caveats:**
  - Blob granularity: a blob's charge goes wholly to its dominant track.
  - Unlabelled blobs are mostly ghosts (notes §2.8).
  - Michel electrons keep their own trackid in the PC (the Bee merge is display-only).

**3c — per-particle truth tree:** needs option P (MABC reading `truth_per_track`); not
proposed now.

### 3.4 Interim: the production Bee zips already allow a charge-based match

`108_logs/bee_label_match.txt`:

- **Where the labels live.** Over 25 events, `truth_trackid_labeled` + `truth_unlabeled`
  points coincide (< 0.05 cm) with the `img-global` layer (blob points): 681 170 of
  681 170. Only 989 coincide with `clustering-pr-global`, which holds trajectory/sampled
  points. So an exact xyz join to PR points finds nothing, and the join has to be
  nearest-neighbour.
- **25 random T_tagger candidates** (log `summary` line):

  | Result | Count |
  |---|---|
  | all PR-cluster points within 1 cm of a label point (median 0.04–0.26 cm) | 25 / 25 |
  | dominant label is a beam-ν interaction | 24 / 25, dominant charge fraction 0.88–1.00 |
  | that interaction = doc 107's nearest-true-vertex interaction | 24 / 24 |
  | dominant label outside the ν truth tree (0.95; nearest true vertex 118 cm) | 1 / 25 |

  The label reading is stronger than nearest-vertex association: two of the 24 have their
  nearest true vertex 302 cm and 359 cm away, yet are 99–100% that interaction's charge.
  Doc 107's 5 cm vertex match would count them as background.
- **Limits of the Bee route:**
  - Out-of-window clusters are drawn t0-shifted, so a per-cluster x shift is needed before
    joining.
  - Label ids outside the `mc.json` tree (cosmics, beam-ν particles below 10 MeV KE)
    cannot be assigned to an interaction.
  - Flavour text is sign-blind.

### 3.5 Are the on-disk MC reco1 files sufficient?

`108_logs/reco1_probe.txt`, `108_logs/reco1_root_probe.txt`:

| Question | Finding |
|---|---|
| Which files are on disk | `/nfs/data/1/yuhw/2025-fall-prod-sample/round2-patrec/mc_paths-v10_14_02_03-100files/`: **10** files (2.8 GB), all production inputs; the `-gpvm.lst` names 100 `/pnfs` paths |
| Event coverage | **104 of 13 217** production events, all run 713; 44 T_tagger rows; 193 truth interactions, 50 in the FV (32 νμCC, 17 NC, **1 νeCC**) |
| The other 990 files | `/pnfs` is not mounted on `wcgpu1`, and there is no `xrdcp`/`ifdh` |
| Truth products present | `simb::MCTruth` (generator, corsika), `GTruth`, `MCFlux`, `MCParticle` (largeant + dropped), `MCParticle↔MCTruth` Assns, `ParticleAncestryMap`, `SimEnergyDeposit` (priorSCE), `SimChannel`, `MCTrack/MCShower`, plus the `simtpc2d:dnnsp` wires the chain consumes. More truth than the labeler uses (e.g. W, Q², target, flux parent) |
| Readable here without LArSoft? | **No.** uproot: memberwise-serialization `NotImplementedError` on `MCTruth`/`MCParticle`/`SimChannel`, and a deserialization error on `EventAuxiliary`. Bare ROOT: complete StreamerInfo, but the emulated read produces no rows: 40 `TBufferFile::CheckByteCount` errors, then abort (`free(): invalid pointer`) |
| What reading them would take | **Mirror classes:** the `wire-cell-sbnd-reco1` pattern, kept out of `libWireCellRoot` (toolkit issue #494), where the Assns is the hard part. **Or LArSoft:** the SL7 container (`/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest`) with cvmfs `sbndcode v10_14_02_03` |
| Can the lar chain run here? | Not as-is. No `/exp`; the only local `libWireCellAIML` (`/home/xqian/fdhd_dev`, larwirecell v10_03_05) predates the labeler. It needs a `dev-v10_14_02_02` larwirecell build, or yuhw's gpvm setup (`sbnd/docs/1-run-tests-sl7-local-builds-sbnd.md`) |
| **Second set: `/home/xqian/work/WC_FM_Sim/runs/{nue,numu,nue_smeared,numu_smeared}/sp_*_reco1.root`** (`108_logs/reco1_probe_wcfmsim.txt`) | **DUNE FD-HD** (dune10kt 1x2x6) truth-labelling study, built 2026-07-02 on `wcgpu1` (dunesw v10_20_08d00 + custom larwirecell `xn/trackid_pid_map`; `WC_FM_Sim/docs/00_overview.md`). 20 events each; GENIE `EventGeneratorList: CC`, single flavour, no cosmics. Carries `MCTruth`, `GTruth`, `MCParticle` + Assns, `SimChannel` (`tpcrawdecoder`) and dnnsp/gauss/wiener wires; `SimEnergyDeposit` only in `g4_*.root`. The `prod10k*` campaigns hold HDF5 extracts only (56 000 `.h5`, no art ROOT). **Not usable for the SBND chain** (other detector, no SBND wires/flash). What it does prove: the SL7-container lar route runs on this host, which bears on decision 3 in sec 6 |

**Verdict:**
- **Not needed** for truth-in-ROOT of this chain (sec 3.1–3.3).
- **Suitable** as the development/validation sample for 3a/3b. A lar run over these 104
  events with the knobs on can be compared against the same events' production Bee zip
  (`mc.json` interactions; sec 3.4 label match) and production tracking-pr (knob-off
  identity).
- **Insufficient** for any efficiency or selection statistics.
- **Not a practical offline truth source** on this machine.

---

## 4. Group 2 — fix wrong or mislabelled fields

### 4.1 `T_cluster.tgm/stm/fc/lm/beam_flash` (always 0 on SBND)

- **What the writer reads:** lowercase `Flags::tgm/short_track_muon/fully_contained/
  light_mismatch/beam_flash` (`SbndPrMagnifyTrackingVisitor.cxx:349-353`;
  `ClusteringFuncs.h:48-63`).
- **What the SBND taggers set:** `Flags::TGM` (`TaggerCheckTGM.cxx:327`,
  `TaggerCheckSTM.cxx:3861,3875`), `Flags::STM` (`TaggerCheckSTM.cxx:641`), `Flags::FC`
  (`TaggerCheckFC.cxx:216`); `ClusteringFuncs.h:85-94`.
- **Why nothing fills them:** the only lowercase setter is `ClusteringTaggerFlagTransfer`
  (`ClusteringTaggerFlagTransfer.cxx:85-105`), which is not configured anywhere in `cfg/`.

**Proposal (knob `sbnd_tagger_flags`):**
- **tgm/stm/fc:** read `Flags::TGM/STM/FC`.
- **lm:** read `get_scalar<int>("lm_flag", -1)` (set by `QLMatching.cxx:3703-3705`).
- **beam_flash:** no SBND flag exists. Either derive it as TaggerCheckNeutrino's test
  (`cluster_t0` in `[beam_window_low, beam_window_high)` && `matched_flash_gid ≥ 0`,
  :2210-2213), with the window passed to the visitor from the same jsonnet local, or
  drop the column under the knob. **Owner choice.**
- **Knob-on check:** for main clusters, T_cluster flags equal the `act_tgm/act_stm/act_fc`
  of the same cluster ids in every row.

### 4.2 `flash_id` holds the global flash gid

- **Setting:** SBND sets `flash_by_gid: true` (`sbnd/clus.jsonnet:955`), so `flash_id` =
  `get_matched_flash().ident()` = the gid (`SbndPrMagnifyTrackingVisitor.cxx:366-367`).
- **Measured:** it equals `matched_flash_gid` in 6 236/6 236 rows
  (`cluster_id_census.txt`).
- **Proposal:** add a `matched_flash_gid` branch under the knob and leave `flash_id`
  unchanged (a rename would break every reader).

### 4.3 Run/subrun/event on every row

- **Today:** `T_tagger`/`T_kine` have no RSE. `UbooneTaggerOutputVisitor` never reads it,
  and uses `ensemble.ident()` only for the filename (:26-30).
- **Trun is fine:** it is correct in 500 of 500 sampled files (run, subrun and event match
  the file tag). The production entry sets `rse_from_ident` and `rse_from_metadata`
  (`wcls-img-clus-matching-xin.jsonnet:117`), and MABC publishes both
  `ensemble.set_rse()` (:3903-3905) and the unconditional scalars
  `runNo/subRunNo/eventNo` (:3914-3916).
- **Proposal:** read the ensemble scalars in `UbooneTaggerOutputVisitor` under a knob and
  add `run/subrun/event` branches to both trees.
- **Caveat:** `SbndPrMagnifyTrackingVisitor::event_rse()` (:73-80) checks only
  `rse_valid()`, then falls back to its configured numbers. It is correct in this
  production, but a `rse_from_metadata`-only chain would fall back to the config. The
  scalars are the robust channel.

### 4.4 Branch census (context only, no action)

- **Compression:** files are ZLIB(1). `T_tagger` (1 229 branches, one entry) is stored
  effectively uncompressed, since the baskets are too small to compress.
- **Constant branches:** 86 of the 1 220 non-`act_*` branches are constant over the
  sampled files with a candidate (`tagger_branch_census.txt`).
- **Not proposed for pruning:** rare firing looks identical on a finite sample, and the
  BDT reads them.

---

## 5. Group 4 — self-describing files

| Item | Obtainable in the visitor? | Proposal |
|---|---|---|
| Toolkit version | yes: `WIRECELL_VERSION` in generated `WireCellUtil/BuildConfig.h` (e.g. `apply-pointcloud-0.35.0-1367-g772c753d`). Use the macro; `Version.h` defines a non-inline symbol included only by `apps/Main.cxx` | `Trun.wct_version` (string) |
| BDT weight files | **no**: they are config of `UbooneNumuBDTScorer` (`:48-52`) and `UbooneNueBDTScorer` (`:77-107`), and a component cannot read another's config (`IConfigurable` has no getter) | pass the same jsonnet locals to the visitor; compiled-config proof that both copies agree |
| Beam window, FV | no (TaggerCheckNeutrino config `beam_window_low/high`, :480-481) | same pattern as the weight files |
| Compiled-config hash, entry git sha, labeler sha | **no** | injected by the runner / fcl as a string (extVar → visitor key), recorded as-is |

All go under one knob (`provenance_strings`). With the knob off, `Trun` is byte-identical.

---

## 6. Owner decisions

1. **trackid → `nu_idx` source for 3b:** option T now, and/or L (larwirecell), sec 3.3.
2. **`beam_flash`:** derive from the beam window, or drop the column (sec 4.1).
3. **Where 3a/3b are developed:** a larwirecell `dev-v10_14_02_02` build on `wcgpu1`, or
   yuhw's gpvm setup. On `wcgpu1` the SL7 container + cvmfs are available and a
   dunesw/larwirecell build already ran lar here (WC_FM_Sim, sec 3.5); there is no
   `/exp`. The reco1 files decide nothing here; the lar environment does.
4. **Backfill the 13 216-event sample** offline from the Bee zips (sec 3.4) now, or
   wait for a rerun with the knobs on.

## 7. Recommended order

1. **Groups 1 + 2 (one SBND knob bag, WCT only, no lar needed):** develop and gate on
   the standalone SBND PR chain and its standard manifest. The knob-on checks are the
   numbers in sec 2.1, 2.3, 2.4 and 4.1–4.2.
2. **3a + 3b with option T:** needs one lar run over the 104 on-disk events to exercise
   the metadata and blob-trackid path; validate against their production Bee zips.
3. **Group 4 strings:** whenever the next production config is cut.
4. **In parallel, no code:** a Bee-based offline truth match for the existing sample
   (extend `d108_bee_label_match.py` from 25 rows to all 6 236, with the t0 shift).

# doc 108 — improving the SBND tracking-pr ROOT output: plan, with the owner's decisions folded in

**Status:** plan only. **No code is changed.** Groups 1, 2 and 4 are implemented in
doc 109 (`109_root-output-improvements.md`), which also settles this doc's "verify before"
items.
- **Revision 1** (2026-09-14): the information audit.
- **Revision 2** (same day): the owner's decisions (sec 0.1) are folded in. The truth
  part is redone for the **standalone** chain with reco1 input, which is how the owner
  runs, and revision 1's claim that reco1 truth is unreadable without LArSoft is
  corrected (sec 0.2).

Every proposed Wire-Cell change is a default-OFF knob with a byte-identical
knob-off gate (CLAUDE.md §1). The truth file is a standalone script and touches no
Wire-Cell code.

## Repro block

```bash
cd sbnd_xin
python3 d108_cluster_id_census.py      > docs/108_logs/cluster_id_census.txt      # sec 2.1 (13 216 files)
python3 d108_flash_census.py           > docs/108_logs/flash_census.txt           # sec 4.1 (needs products/d107)
python3 d108_tagger_branch_census.py   > docs/108_logs/tagger_branch_census.txt   # sec 4.4
python3 d108_bee_label_match.py        > docs/108_logs/bee_label_match.txt        # sec 3.7
python3 d108_reco1_probe.py            > docs/108_logs/reco1_probe.txt            # sec 3.6
W=/home/xqian/work/WC_FM_Sim/runs
python3 d108_reco1_probe.py --products-only $W/{nue,numu,nue_smeared,numu_smeared}/sp_*_reco1.root $W/nue/g4_nue.root \
                                       > docs/108_logs/reco1_probe_wcfmsim.txt    # sec 3.6 second set
F=/nfs/data/1/yuhw/2025-fall-prod-sample/round2-patrec/mc_paths-v10_14_02_03-100files/reco1-detsim-g4-gen-Gen2_2026-a6a0-2395-0263-6d62.root
for e in $(seq 0 9); do                                                           # sec 3.2-3.3, ONE ENTRY PER PROCESS
  ROOT_INCLUDE_PATH=<toolkit>/root/src root -l -b -q "d108_reco1_truth_vectors.C(\"$F\", $e, $e)"
done                                   > docs/108_logs/reco1_truth_vectors.txt    # (ROOT noise lines filtered)
python3 d108_truth_vs_bee.py docs/108_logs/reco1_truth_vectors.txt > docs/108_logs/truth_vs_bee.txt   # sec 3.3
python3 d107_tables.py products/d107                                              # sec 2.1 association split
```

`d108_reco1_root_probe.C` (`reco1_root_probe.txt`) is revision 1's failed read: no
`ROOT_INCLUDE_PATH`, several variables in one `Scan`. It is kept as the record of
what not to do.

Code citations are against toolkit `apply-pointcloud` @ `d3b398fc`, the production
entry `sbnd/wcls-img-clus-matching-xin.jsonnet` @ `8ca6866d` (this repo), and the
larwirecell truth labeler `HaiwangYu/larwirecell` `dev-v10_14_02_02` @ `a02a1a4d`
(read through the GitHub API).

---

## 0. Answers

### 0.1 Owner decisions (2026-09-14)

| # | Question | Decision | Where |
|---|---|---|---|
| 1 | How much truth | **Event-level.** For each candidate the key is the **neutrino vertex**; events can hold several neutrinos, so truth is stored **as vectors** and matched in the analysis | sec 3.4–3.5 |
| 2 | Where it goes | **A separate truth ROOT file** | sec 3.4 |
| 3 | Per-candidate charge-based truth match | **Not needed** (revision 1's "3b" is dropped) | — |
| 4 | `T_cluster.beam_flash` | **Derive it.** Handle multiple beam flashes | sec 4.1 |

### 0.2 Corrections to revision 1 and to the chat

- **"reco1 truth is not readable without LArSoft on this machine" — wrong.** The
  owner's own `scripts/root/mc_truth_muons.C`, `mc_nu_vertices.C` and `dump_truth_sed.C`
  already read it with bare ROOT (`TTree::Draw` on emulated classes, run with
  `ROOT_INCLUDE_PATH=<toolkit>/root/src`). My probe had omitted that variable. Every
  field the truth file needs reads correctly, and is validated (sec 3.2–3.3).
- **Revision 1's truth plan went through the LArSoft truth labeler.** That applies
  only to the lar production chain. The owner runs Wire-Cell standalone with reco1
  input, where no labeler runs.
- **Shared tagger writer:** `UbooneTaggerOutputVisitor` is shared with the PDHD and PDVD
  PR configs (`pdhd/pr.jsonnet:2010`, `protodunevd/pr.jsonnet:2038`), not with uBooNE's.
- **`T_tagger.cluster_id` is not wrong:** it is where the PR result lives. The selected
  activity is a different question (sec 2.1).
- **25-candidate Bee label test:** I said "23 of 25". The logged tally is 24 of 25
  neutrino-dominated plus 1 non-neutrino (sec 3.7).

### 0.3 Information availability

| Group | Available? | Caveat |
|---|---|---|
| **1** record what happened | **Yes**, all inside `TaggerCheckNeutrino` | two items to verify (sec 2.5) |
| **2** fix wrong/mislabelled fields, incl. `beam_flash` | **Yes** | TPC encoding of the flash gid, and completeness of the flash list, to verify (sec 4.1) |
| **3** event-level truth, standalone | **Yes**, from reco1 with bare ROOT; 15/15 interactions match production truth | the emulated read aborts after a few entries in one process; use one entry per process (sec 3.2) |
| **4** self-describing files | **Mostly** | a compiled-config hash must be injected (sec 5) |

---

## 1. Context

### 1.1 Who writes what

| Tree | Writer | Rows |
|---|---|---|
| `Trun`, `T_bad_ch`, `T_cluster`, `T_rec_charge`, `T_proj_data` | `root/src/SbndPrMagnifyTrackingVisitor.cxx` (SBND only: `sbnd/clus.jsonnet:2472`) | per event / cluster / point |
| `T_tagger`, `T_kine` | `root/src/UbooneTaggerOutputVisitor.cxx` (`common/clus.jsonnet:813`; SBND, PDHD, PDVD) | one per flash bundle with a candidate |

Both writers are `IEnsembleVisitor`s inside the PR `MultiAlgBlobClustering` and see
only the Ensemble.

**Two chains produce this output:**
- **Standalone (the owner's):** `run_reco1_dump.sh -mc` extracts the reco1 file into
  frames + opflash, then imaging → Q/L → PR → `tracking-pr.root`. It runs no truth
  labeler.
- **lar production** (the 13 216-event sample of doc 107): the chain additionally runs
  the larwirecell `wclsTensorSetLabeler`, whose truth reaches the Bee zips (sec 3.7).

### 1.2 Constraints for the Wire-Cell changes (groups 1, 2, 4)

- **Knobs:** default OFF; key-suppression in jsonnet; compiled SBND/PDHD/PDVD configs
  byte-identical with the knob off (the tagger writer is shared).
- **Gates and tests:** knob-off gate on the SBND standard manifest; doctests
  (`wcdoctest-clus`, `wcdoctest-root`).
- **Out of scope here:** the two writers use one fixed filename (RECREATE vs UPDATE), so a
  multi-event process silently loses T_tagger/T_kine
  (`wcls-img-clus-matching-xin.jsonnet:254-271`).

---

## 2. Group 1 — record what happened

### 2.1 Selected activity vs final cluster

All 6 236 T_tagger rows (`108_logs/cluster_id_census.txt`):

| `cluster_id` vs `act_cluster_id[argmax(act_is_selected)]` | rows | `cluster_id` in T_cluster | reco vertex < 5 cm of a true vertex* | median* |
|---|---|---|---|---|
| same | 5 748 | main 5 553, associated 195 (demoted-main fallback) | 69.1% | 1.7 cm |
| another `act_*` entry | 109 | associated 109 | 43.1% | 12.3 cm |
| not in `act_*` | 379 | associated 379 | 49.3% | 5.4 cm |

\* `d107_tables.py` (nearest-true-vertex association, doc 107 sec 4.5).

In all 488 differing rows, `T_kine` carries the same `cluster_id`, `nu_index` and vertex,
and the cluster's `flash_id` equals the row's `matched_flash_gid`. The vertex search
moved the main cluster onto an associated companion **of the same flash bundle**.

**Where the information is** (`clus/src/TaggerCheckNeutrino.cxx`):
- **Selection:** `pick()` (:2251-2299) chooses the activity and `is_selected` is stamped at
  :2305-2307. `candidates[nu_index].main` stays in scope; the loop copies it into the
  local `main_cluster` (:2463) and never reassigns the vector element.
- **The move:** `determine_overall_main_vertex_DL` takes `Cluster*& main_cluster` and swaps
  with **no knob** (`NeutrinoVertexFinder.cxx:5391`). The traditional path swaps only
  under `main_vertex_swap_apply` (:3236-3250; SBND false, `wct-pr-perevt.jsonnet:2896`).
  SBND's `pr()` defaults `dl_weights` to the SCN net (`sbnd/clus.jsonnet:981`), so the DL
  path is the likely source (sec 2.5 b).
- **The write:** `tagger_info.cluster_id = main_cluster->get_cluster_id()` (:3547), after
  the move.

**Proposal (knob `nu_row_provenance`):** add `sel_cluster_id` and `vertex_moved_cluster`;
optionally `sel_vertex_{x,y,z}` (sec 2.5 a).

### 2.2 Full activity roster

`act_*` lists only in-window mains and demoted mains (`add_acts`, :2230-2247). Companions
are collected afterwards into `NuCandidate::others` (:2310-2324), with TGM/STM
companions ≥ `cosmic_companion_min_length` dropped. So in 379 rows the final cluster has no
roster entry.

**Proposal:** append companions with:
- `act_role`: 0 main, 1 demoted main, 2 companion, 3 companion dropped as cosmic;
- `act_is_final`;
- `act_evaluated = 0` where the Q-L taggers never ran.

### 2.3 `has_vertex`

`kine_nu_{x,y,z}_corr` default to 0 (`NeutrinoTaggerInfo.h:27-29`) and are filled only
`if (final_main_vertex)` (:3733-3741). There is no "vertex found" flag, and (0,0,0) is a
point on the SBND cathode plane; 160 rows carry it (doc 107).

**Proposal:** `has_vertex` in T_tagger and T_kine.

### 2.4 One row per flash bundle, including bundles with no candidate

**Why 7 078 events have no T_tagger/T_kine:** with no candidate, `TaggerCheckNeutrino`
returns at :2367-2371 before `set_track_fitting` (:3829-3832). `UbooneTaggerOutputVisitor`
then returns at :73-76 before opening the file (:98).

| Code | Reason | Where | Recorded today? |
|---|---|---|---|
| 0 | selected | :2305 | yes (row) |
| 1 | demoted-main fallback selected | :2297-2299 | only via `act_is_demoted` |
| 2 | every main/demoted cosmic-tagged (TGM / STM / `lm_flag>0`) | `pick()` :2254-2263 | log only |
| 3 | below `nu_per_bundle_min_length`, not the legacy winner | :2285-2291 | log only |
| 4 | not STM under `nu_per_bundle_stm_only` (PDVD mode) | :2271-2276 | log only |
| — | event: no main/demoted cluster, or none in the beam window | :2207-2211 | counters, log only |
| — | in-window cluster with `matched_flash_gid < 0` | :2213 | **silently dropped** |

**Proposal (knob `nu_bundle_census`):**
1. `TaggerCheckNeutrino` publishes each gid's reason code, roster and the event counters
   (plus a new count of in-window `gid<0` clusters) as numeric arrays on a grouping- or
   ensemble-level PC (`put_pcarray`, `Facade_Mixins.h:~300`) before the :2367 return.
2. The visitor's `!tf` guard is relaxed under the knob; it writes `T_bundle` (one row per
   gid, joined to `T_flash` of sec 4.1) and the counters into `Trun`.

**Knob-on check:** the 7 078 no-row events are all explained.

### 2.5 Verify before implementing

- **(a) Selection-time vertex.** Does `map_cluster_main_vertices[main_cluster]` (:3147)
  survive the DL swap (`NeutrinoVertexFinder.cxx:5391-5396`) and later writes? Not
  audited. If not, capture the position in a local at :3147.
- **(b) Which path moved the 488 clusters.** Confirm from the compiled config
  (`dl_weights` resolved, `main_vertex_swap_apply`) or a debug log of one moved event.

---

## 3. Group 3 — event-level truth for the standalone chain

### 3.1 Why the production truth path does not apply

The lar production chain (`sbnd/wcls-img-clus-matching-xin.jsonnet:336-458`) runs
`clus_all_apa → labeler_truth → pr_node → labeler_tagger`. The labeler is
`wclsTensorSetLabeler`, a larwirecell/art module, and the standalone chain has no such
stage. So standalone truth comes **straight from the reco1 file** the chain already
takes as input. Since the owner wants event-level truth in a separate file (sec 0.1), no
Wire-Cell code is involved.

### 3.2 Reading reco1 truth without LArSoft

**Recipe** (from `scripts/root/mc_truth_muons.C`, extended in `d108_reco1_truth_vectors.C`):
- **Environment:** `ROOT_INCLUDE_PATH=<toolkit>/root/src`, so the toolkit rootmap does not
  hijack ROOT's autoloader.
- **Reading:** `TTree::Draw(expr, "", "goff", 1, entry)`, one expression per call; never
  `Events->GetEntry()` (~2 500 branches).
- **One reco1 entry per ROOT process.** Several entries in one process abort
  (`free(): invalid pointer/size`; revision-2 probe: 5 of 10 entries read, then abort),
  whereas each of the 10 entries read cleanly in its own process
  (`108_logs/reco1_truth_vectors.txt`). The standalone pipeline is already per event, so
  this costs nothing.
- **Noise:** `TBufferFile::CheckByteCount` errors (process-history metadata) are harmless.
- **uproot cannot read these products** (memberwise serialization,
  `108_logs/reco1_probe.txt`).

**Fields (all read in the probe):**

| Truth | Draw expression (prefix `simb::MCTruths_generator__GenieGen.obj`) | Units |
|---|---|---|
| run / subrun / event | `EventAuxiliary.id_.subRun_.run_.run_`, `.id_.subRun_.subRun_`, `.id_.event_` | — |
| number of beam-ν interactions | element count of any generator-MCTruth expression | — |
| ν PDG (**signed**) | `.fMCNeutrino.fNu.fpdgCode` | — |
| CC/NC, mode, interaction type | `.fMCNeutrino.fCCNC`, `.fMode`, `.fInteractionType` | 0 = CC |
| Eν | `.fMCNeutrino.fNu.ftrajectory.ftrajectory.second.fE` | GeV |
| vertex | `.fMCNeutrino.fNu.ftrajectory.ftrajectory.first.fP.fX/fY/fZ` | cm, true coordinates (no SCE) |
| time | `.fMCNeutrino.fNu.ftrajectory.ftrajectory.first.fE` (the 4-vector's time slot) | ns |
| **Edep per interaction** | `sim::SimEnergyDeposits_ionandscint_priorSCE_G4.obj.{trackID,edep}` → `simb::MCParticles_largeant__GenieGen.obj.ftrackId` → Assns `...Assns_largeant__GenieGen.obj.ptr_data_1_.second` (particle index) / `ptr_data_2_.first.id_.value_` + `ptr_data_2_.second` (MCTruth product, key) | MeV |

**Details of the Edep chain:**
- **Deposit ownership:** use `|trackID|`. A negative value means a dropped secondary of
  `|trackID|`; `origTrackID` would lose the delta rays (`dump_truth_sed.C` header).
- **Two MCTruth collections:** the Assns points into both the generator and the corsika
  MCTruths. The art ProductID is the **CRC32 of the product branch name**:
  `crc32("simb::MCTruths_generator__GenieGen.")` = 3130632973 and
  `crc32("simb::MCTruths_corsika__GenieGen.")` = 647339619, both seen in the file. So the
  generator's key is the ν index, with no guessing.
- **Unmatched deposits** (no MCParticle for `|trackID|`): 0.0 MeV in all 10 entries.

### 3.3 Validation against the production truth

Same 10 events (`r713_s59_*`, entries 0–9 of the file above); the production Bee
`mc.json` truth came from the lar labeler (`108_logs/truth_vs_bee.txt`):

| Interactions | Flavour | CC/NC | Mode | Eν (±0.1 MeV) | Vertex (±0.02 cm) | T (±0.001 µs) | **Edep (±0.1 MeV)** |
|---|---|---|---|---|---|---|---|
| 15 (in 10 events; **5 events have 2 neutrinos**) | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 | **15/15** |

Examples: `e21` Edep 184.6 = 184.6; `e22` two neutrinos, Edep 448.0 and 0.0; `e41`
νe CC + νμ NC. The reco1 read also gives what Bee's text lacks: **the PDG sign**, and
interaction type codes.

### 3.4 Design: the truth file

**One file per event**, named like the reco output: `truth_r<run>_s<subrun>_e<event>.root`
next to `tracking-pr_r<run>_s<subrun>_e<event>.root`. Tree **`T_truth`, one entry per
event**:

| Branch | Type | Content |
|---|---|---|
| `run`, `subrun`, `event` | int | art event id (the join key to tracking-pr `Trun`) |
| `n_nu` | int | number of generator MCTruths (beam-ν interactions, incl. dirt/rock) |
| `nu_pdg` | vector<int> | signed ν PDG |
| `nu_ccnc`, `nu_mode`, `nu_int_type` | vector<int> | GENIE codes (CCNC 0 = CC) |
| `nu_E` | vector<float> | Eν [GeV] |
| `nu_vtx_x`, `nu_vtx_y`, `nu_vtx_z` | vector<float> | vertex [cm], true coordinates |
| `nu_t` | vector<float> | interaction time [ns] |
| `nu_edep` | vector<float> | deposited energy [MeV], SED sum over the interaction's descendants (same definition as the labeler's `Edep`, validated sec 3.3) |

Vector index `i` is the generator-MCTruth index, identical to Bee's `9000000+i` node.

- **Writer:** grow `d108_reco1_truth_vectors.C` into a writer with the same reads, run
  one entry per process from the `run_reco1_dump.sh -mc` step (which already walks the
  file's entries).
- **Data events:** no file.
- **Optional extras, only if wanted later:** lepton PDG/energy (`fMCNeutrino.fLepton`, not
  yet probed) and GTruth final-state counts. Not proposed now.

### 3.5 Matching truth to candidates in the analysis (multiple neutrinos)

- **By vertex (owner's key):** for each T_tagger row, take the distance from
  `nu_x/y/z` (= `kine_nu_*_corr`, SCE-corrected) to every `nu_vtx` in the event's vector.
  The nearest one, within the analysis cut (doc 107: 5 cm), is the matched interaction.
  This works unchanged with several neutrinos.
- **By time (recommended companion):** the row's flash time minus the true `nu_t`
  (converted to µs) is a fixed **+0.136 µs**. Over 4 098 rows whose vertex is < 5 cm from
  the true one, the 5–95% range is 0.132–0.140 µs and 98.9% lie within 0.1 µs of the
  median (`108_logs/flash_census.txt` part C).
  - This names the neutrino that made the row's flash, **even when the reco vertex is
    misplaced**, and separates two neutrinos in the window by their times.
  - It needs the row's flash time: today via `T_cluster.flash_time_us` of the cluster
    whose `flash_id == matched_flash_gid`; sec 4.1 puts it on the row.

### 3.6 The reco1 files on disk

| Set | Content | Use |
|---|---|---|
| `/nfs/data/1/yuhw/2025-fall-prod-sample/round2-patrec/mc_paths-v10_14_02_03-100files/` | 10 production inputs (2.8 GB) = 104 of 13 217 events, run 713; 193 interactions, 50 in the FV (32 νμCC, 17 NC, 1 νeCC) (`reco1_probe.txt`); `/pnfs` for the other 990 is not mounted here | development and validation of the truth file and of standalone runs (their production Bee/tracking-pr are a ready reference); too few for statistics |
| `/home/xqian/work/WC_FM_Sim/runs/*/sp_*_reco1.root` | **DUNE FD-HD** (1x2x6) GENIE-CC studies, 4 × 20 events; `prod10k*` hold HDF5 only (`reco1_probe_wcfmsim.txt`) | not usable for SBND |

### 3.7 The existing lar-produced 13 216-event sample

Its Bee zips already carry the labeler's truth: `mc.json` nodes (sign-blind text) and
per-blob labels on the `img-global` layer.
- **Event-level truth** for that sample is therefore in hand (doc 107 used it).
- **Label test:** on 25 random candidates, all points of the PR cluster were within 1 cm
  of a label point; 24/25 were ν-dominated (charge fraction 0.88–1.00, and the same
  interaction as doc 107's nearest true vertex in 24/24), 1/25 non-ν
  (`bee_label_match.txt`).
- Kept for reference only, since the owner does not need a per-candidate charge match.

---

## 4. Group 2 — fix wrong or mislabelled fields

### 4.1 `beam_flash`, and multiple beam flashes

**Today:**
- `T_cluster.tgm/stm/fc/lm/beam_flash` are always 0 on SBND. The writer reads lowercase
  flags (`SbndPrMagnifyTrackingVisitor.cxx:349-353`), which only the unconfigured
  `ClusteringTaggerFlagTransfer` sets.
- The taggers set `Flags::TGM/STM/FC` (`TaggerCheckTGM.cxx:327`, `TaggerCheckSTM.cxx:641`,
  `TaggerCheckFC.cxx:216`).
- `lm_flag` is a cluster scalar (`QLMatching.cxx:3703-3705`).
- SBND has **no** beam-flash flag.

**Derive `beam_flash` (owner decision 4).** It is 1 for a cluster that has a matched flash
(`matched_flash_gid ≥ 0`) whose time is in the beam window `[0.2, 2.2)` µs. That is
exactly the tagger's own test: `cluster_t0` in `[beam_window_low, beam_window_high)`,
:1971 and :2210-2213. `cluster_t0` equals the matched flash time. The window
(`sbnd/clus.jsonnet:1003`) is passed to the visitor from the same jsonnet local, and the
compiled-config proof shows both copies agree. tgm/stm/fc are read from `Flags::TGM/STM/FC`,
lm from `lm_flag`.

**How many beam flashes there are** (`108_logs/flash_census.txt`, 3 000 random production
events; a flash's TPC from Bee `op.json` `apa`, a row's TPC = `gid // 1000000`):

| In-window flashes per event | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| events | 177 | 1 741 | 1 079 | 3 |

| The 1 079 two-flash events | events |
|---|---|
| different TPCs, \|Δt\| < 0.1 µs (**one flash seen by both TPCs' PDS**) | 841 (clusters on both 491, one 267, neither 83) |
| different TPCs, \|Δt\| ≥ 0.1 µs (distinct flashes) | 237 |
| same TPC | 1 |

- **\|Δt\| of different-TPC pairs (µs):**

  | 0–0.01 | 0.01–0.02 | 0.02–0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.5 | 0.5–2 |
  |---|---|---|---|---|---|---|
  | 492 | 204 | 111 | 34 | 35 | 73 | 129 |

  The one-flash peak ends near **0.05 µs**; above it the density is flat.
- **Events with two T_tagger rows (98):**
  - **24** come from one flash seen by both TPCs. In **22** of them the second row has
    **no vertex**: the weak side of the same light makes an empty row.
  - **74** come from distinct flashes. In **33** both rows are < 5 cm from **different**
    true neutrinos; these are real two-neutrino events.

**How to handle it.** The chain already does the right thing: one row per flash bundle
(gid), and `beam_flash` is well defined per cluster however many flashes there are. What
is missing is enough flash information for the analysis to tell the cases apart:
1. **`T_cluster`:** `beam_flash` (derived), `matched_flash_gid` (today `flash_id` holds
   it; kept for readers), `flash_tpc`.
2. **`T_tagger`/`T_kine` rows:** `flash_time_us`, `flash_pe`, `flash_tpc`, and
   **`flash_group`** = one id shared by different-TPC flashes within
   **\|Δt\| < `flash_pair_dt`** (visitor knob, proposed default 0.05 µs from the
   distribution above).
   - **Rows sharing a `flash_group`** are one physical flash; the empty second row of the
     24 events can be dropped.
   - **Rows in different groups** are candidate distinct interactions; match each to the
     truth vector (sec 3.5).
3. **`T_flash`, one row per flash per event:** gid, TPC, time, PE, `in_window`,
   `flash_group`, number of matched main clusters, T_tagger row index (-1 if none).
   - This shows in-window flashes that produced no candidate: 587 of the 3 000 events
     have in-window flashes but none with a matched cluster.
   - Together with the truth `nu_t` + 0.136 µs, it tells whether a missed neutrino made a
     flash at all.

**Verify before implementing:**
- **TPC from gid:** `gid = anode_ident * 1000000 + index` holds for SBND production
  (`Facade_Grouping.h:248-252`), but the `opflash_phys_gid` config setting changes it.
  Confirm it is unset in both chains.
- **Flash list completeness:** `T_flash` must be built from the merge-safe `opflash` PC
  (grouped by gid), **not** `Grouping::flashes()`, which reads the `flash` PC that keeps
  only the primary input after the merge. Also confirm `QLMatching::write_opflash_pc`
  writes **all** flashes, not only matched ones.

**Knob-on checks:**
- For main clusters, the T_cluster flags equal `act_tgm/act_stm/act_fc` of the same ids
  in every row.
- `beam_flash` equals "`cluster_t0_us` in window && gid ≥ 0".
- The two-flash census above is reproduced from `T_flash`.

### 4.2 `flash_id` holds the global flash gid

SBND sets `flash_by_gid: true` (`sbnd/clus.jsonnet:955`); `flash_id` =
`get_matched_flash().ident()` (`SbndPrMagnifyTrackingVisitor.cxx:366-367`). It equals
`matched_flash_gid` in 6 236/6 236 rows. Handled by the `matched_flash_gid` branch of
sec 4.1; `flash_id` is unchanged.

### 4.3 Run/subrun/event on every row

- **Today:** `T_tagger`/`T_kine` carry no RSE (`UbooneTaggerOutputVisitor` uses
  `ensemble.ident()` only for the filename, :26-30).
- **Trun is fine:** it is correct in 500/500 sampled files. MABC publishes `set_rse()`
  (:3903-3905) and the unconditional scalars `runNo/subRunNo/eventNo` (:3914-3916).
- **Proposal:** read the scalars, then add `run/subrun/event` to both trees. It is also
  the join key to the truth file.
- **Caveat:** `SbndPrMagnifyTrackingVisitor::event_rse()` (:73-80) checks only
  `rse_valid()`, which a `rse_from_metadata`-only chain would not set.

### 4.4 Branch census (context only, no action)

- **Compression:** files are ZLIB(1); `T_tagger` (1 229 branches, one entry) is stored
  effectively uncompressed.
- **Constant branches:** 86 of 1 220 non-`act_*` branches are constant over the sample
  (`tagger_branch_census.txt`).
- **Not pruned:** rare firing looks the same, and the BDT reads them.

---

## 5. Group 4 — self-describing files

| Item | In the visitor? | Proposal |
|---|---|---|
| Toolkit version | yes: `WIRECELL_VERSION` (generated `WireCellUtil/BuildConfig.h`) | `Trun.wct_version` |
| BDT weight files | no: config of `UbooneNumuBDTScorer` (:48-52) / `UbooneNueBDTScorer` (:77-107) | pass the same jsonnet locals; compiled-config proof |
| Beam window, FV, `flash_pair_dt` | no (TaggerCheckNeutrino `beam_window_low/high`, :480-481) | same pattern |
| Compiled-config hash, entry git sha | no | injected by the runner as a string |

All under one knob (`provenance_strings`); `Trun` is byte-identical with it off.

---

## 6. Decisions

**Resolved** (sec 0.1): truth scope and vectors, a separate truth file, no per-candidate
match, `beam_flash` derived.

**Still open, with recommendations:**

1. **`flash_pair_dt`:** 0.05 µs (recommended; end of the one-flash peak) or 0.1 µs
   (revision-1 census cut; takes in about 34 more pairs per 3 000 events).
2. **Truth file granularity:** one file per event (recommended; mirrors
   `tracking-pr_<tag>.root`) or one per reco1 file.
3. **Truth for the existing 13 216-event lar sample:** use the Bee `mc.json` truth as doc
   107 did (sign-blind; recommended for now), or rerun the truth writer over its reco1
   files, which needs the 990 `/pnfs` files.

---

## 7. Recommended order

1. **Truth file** (no Wire-Cell change): turn `d108_reco1_truth_vectors.C` into the
   `T_truth` writer (sec 3.4) and hook it into `run_reco1_dump.sh -mc`, one entry per
   process.
   - Validate on the 104 on-disk events with `d108_truth_vs_bee.py`; expect every field
     equal, as on the 10 here.
2. **Groups 1 + 2**, incl. `beam_flash`, the flash fields and `T_flash`, as one SBND knob
   bag on the two writers. Gate knob-off on the standalone SBND manifest; the knob-on
   checks are listed in sec 2 and 4.1.
3. **Group 4 strings.**
4. **Analysis:** extend doc 107's selection to the vector truth with the vertex and
   +0.136 µs time matches.

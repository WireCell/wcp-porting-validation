# 107 — SBND MC production: where the truth is, and whether the cosmic taggers agree with the neutrino PR

Doc number 107 (claimed 2026-09-14). Analysis only: no toolkit code or config is
changed, and the sample is read-only.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# sbnd_mc_data -> /nfs/data/1/xqian/sbnd_data/run (tracking-pr/, bee/, logs/, summary-merged.csv)
python3 d107_truth_tagger_consistency.py --run sbnd_mc_data --out products/d107 \
    > ~/tmp/d107_run.log 2>&1; echo rc=$?
# -> products/d107/{candidates.tsv, truth.tsv, summary.txt}
# rc=0, "events 13216 candidate rows 6236 truth interactions 23738"
```

The tables in sections 4 and 5 are computed from `candidates.tsv` and `truth.tsv`,
with the column names given in each table. The script refuses to write into an
existing output dir.

## 0. Answers in brief

| Question | Answer |
|---|---|
| Where is the reco? | `tracking-pr/tracking-pr_<tag>.root`. `T_tagger` holds `numu_score`, `nue_score`, `neutrino_type`, the ~1200 tagger/BDT variables and the `act_*` cosmic-tagger arrays. `T_kine` holds `kine_reco_Enu`. Both have one row per neutrino candidate. |
| Where is the truth (type, Enu, vertex)? | **Only in Bee**, in `bee/bee_<tag>.zip::data/0/0-mc.json`. The tracking-pr ROOT files carry **no** truth branches. |
| Does mc.json hold truth or reco particle flow? | **Both, in one tree.** The truth interaction nodes (id 9000000+) come from larwirecell's `TensorSetLabeler`. Our reco PF is grafted under a single `reco nu …` node (id 19999999, children ≥ 20000000), or a `no reco neutrino candidate (…)` marker replaces it. Each node is unambiguous by id and text (section 2.2). |
| Is the PR candidate the non-cosmic one? | **Yes, in 6236/6236 rows, by construction.** The per-bundle `pick()` skips any activity with TGM, STM or `lm_flag>0` (section 3). The informative numbers are the residuals in section 4. |

## 1. The sample

| Item | Value |
|---|---|
| Path | `sbnd_xin/sbnd_mc_data` → `/nfs/data/1/xqian/sbnd_data/run` |
| Input | 1000 reco1 files, `…/v10_14_02_03/prodgenie_corsika_proton_rockbox0p1_sbnd/Gen2_2026/CV/reco1/…` (BNB ν with rockbox + CORSIKA cosmics) |
| Chain | lar (`wcls-img-clus-matching-xin.jsonnet` + `pr-operating-point.jsonnet`): imaging, clustering, Q/L matching, PR, taggers, BDT scorers |
| Events | 13 217 in `summary-merged.csv`. 13 216 have rc=0 (1 087 were re-run after rc=126 in `summary.csv`); 1 (`r471_s18_e33`, retry:1) has rc=11 with no products and audit FAIL (`logs/audit_9040_r471_s18_e33.txt`: `tracking_visitor ran but tracking-pr.root is missing`). |
| Products | 13 216 `tracking-pr_*.root`, 13 216 `bee_*.zip`. `nugraph/` and `work/` are empty. |

The production log confirms the per-bundle selection ran. For example,
`logs/missing_10000_r717_s93_e26.log` contains
`TaggerCheckNeutrino: [nu_per_bundle] gid 1000006: candidate main cluster 14 …`.

## 2. Where things are

### 2.1 tracking-pr ROOT (reco only)

| Tree | Writer | Rows | Use |
|---|---|---|---|
| `T_tagger` | `root/src/UbooneTaggerOutputVisitor.cxx:107` | one per candidate (`nu<i>` slot, `:83-89`, filled `:1237`) | `numu_score`, `nue_score`, `neutrino_type`, `cosmict_*`, `nu_x/y/z`, `cluster_id`, `matched_flash_gid`, `nu_index`, `act_*` |
| `T_kine` | same, `:1160` | aligned 1:1 with `T_tagger` (`:1237-1238`) | `kine_reco_Enu`, `kine_nu_*_corr`, π⁰ and MCS energies |
| `T_cluster` | `root/src/SbndPrMagnifyTrackingVisitor.cxx` | one per cluster | `is_main`, `is_associated`, `flash_id`, `length_cm`, `cluster_t0_us`. **Its `tgm/stm/fc/lm/beam_flash` columns are always 0 (section 4.3).** |
| `T_rec_charge`, `T_proj_data`, `Trun`, `T_bad_ch` | `SbndPrMagnifyTrackingVisitor` | — | track-fit charge, projections, run info |

If an event has no candidate, `UbooneTaggerOutputVisitor` returns early (`:72-75`,
"no TrackFitting in grouping"). **Such a file then has no `T_tagger` or `T_kine`
at all.** That is the case for 7 078 of the 13 216 events.

### 2.2 Bee `0-mc.json`: truth and reco particle flow in one forest

Bee draws one particle tree per event, so the SBND chain grafts the reco flow onto
the truth tree. The graft is `MultiAlgBlobClustering::pf_set_particles`
(`clus/src/MultiAlgBlobClustering.cxx` ~3040-3081), enabled by
`cfg/pgrapher/experiment/sbnd/clus.jsonnet:2737-2743`
(`merge_metadata_key: 'bee_pf_truth'`, `merge_node_text: 'reco nu'`,
`emit_empty: true`). The result is the truth forest at top level plus one reco
summary node.

| Top-level node | id range | text | Written by |
|---|---|---|---|
| **truth interaction** | `9000000 + nu_idx` (children = G4 trackids ≥ 1e7) | `"<n> <flav> <QE\|RES\|DIS\|MEC\|COH\|NuEEL> <CC\|NC> Etot <E> MeV Edep <D> MeV T <t> us"`, `data.start` = vertex | larwirecell `aiml/TensorSetLabeler.cxx` |
| **reco summary** | exactly `19999999` (= `merge_id_offset - 1`); every descendant ≥ 20000000 | `"reco nu <Enu> MeV numu <s> nue <s>"`, with children `"nu <i> (gid <g>, cluster <c>)"` → particles | `pf_summary_node` (`MultiAlgBlobClustering.cxx` ~2994) |
| **no-candidate marker** | — | `"no reco neutrino candidate (no TrackFitting)"` | `MultiAlgBlobClustering.cxx` ~1387 |

**Full-sample check** (`summary.txt`):
- 6 138 events have exactly one reco node and 7 078 exactly one marker; there are 0 other top-level nodes.
- Reco-node presence equals `T_tagger` presence in 13 216/13 216 events.
- The reco node's Enu equals `T_kine` `kine_reco_Enu` of the primary slot in 6 138/6 138.
- The reco scores are the primary candidate's (`nu_index` 0). A second candidate's scores appear only in `T_tagger`.

Without the labeler (data, or the standalone `run_pr_chain_batch.sh` runs), the same
file holds only the reco node. That is the "mc.json stores reco PF" case.

**Truth node definitions**, from the labeler source (read-only fetch of
`HaiwangYu/larwirecell` branch `dev-v10_14_02_02` @ `a02a1a4d`; the text format
matches every node in the sample):

| Field | Definition | Source |
|---|---|---|
| `Etot` | true incoming neutrino energy, `MCTruth.GetNeutrino().Nu().Momentum(0).E()` × 1e3 | `TensorSetLabeler.cxx:726-729` |
| `Edep` | sum of `sim::SimEnergyDeposit::Energy()` over deposits whose track belongs to this interaction (visible energy) | `:678-705` |
| vertex | `Nu().Position(0)`, generator coordinates in cm (no SCE, no drift) | `:723-726`, `:848-852` |
| `T` | `Position(0).T()` × 1e-3, interaction time in µs (sample range 0.009-1.634) | `:730` |
| `CC/NC` | `MCNeutrino::CCNC()` | `:725`, `:842` |
| mode | `MCNeutrino::InteractionType()` name | `:121-124` |

Caveats on the truth tree:
- **Particle floor:** MCParticles need KE > 10 MeV (`pf_ke_min`).
- **Beam neutrinos only:** with `pf_nu_only` (default true) there are no cosmics at all.
- **Not a full interaction list:** an interaction gets a node only if it has ≥ 1 kept particle. Use it with care as an efficiency denominator.
- **Stale header:** `TensorSetLabeler.h:125-128` still says the node energy is Edep only. The code writes both Etot and Edep.

The labeler also puts `nu_pdg/nu_ccnc/nu_int_type/nu_energy/nu_vtx_*` into the tensor-set
metadata. That metadata is not persisted in these products, and the nugraph HDF5
output is empty here.

**Sample content:** 23 738 truth interactions in 13 216 events.
- Interactions per event: 1 in 5 634 events, 2 in 5 223, 3 in 1 858, 4 in 431, 5+ in 70.
- Flavour: numu 23 493, nue 245.
- Mode: QE 12 477, RES 5 971, MEC 3 214, DIS 1 990, COH 84, NuEEL 2.
- 24 events have every interaction at Edep = 0 (rockbox, outside the TPC).

## 3. How the neutrino candidate is chosen (production path)

Production knobs (`sbnd/pr-operating-point.jsonnet`):
- `nu_per_bundle=true` (:32), `nu_per_bundle_min_length=15` (:33)
- `evaluate_demoted_mains=true` (:27), `restore_demoted_mains=true` (:42)
- `neutrino_type_bitmask=true` (:31)
- `nu_skip_cosmic=true` (:154), `nu_skip_cosmic_bundle=true` (:155)
- `nu_fallback_demoted_mains=true` (:151), `nu_selected_as_main=true` (:152)
- `skip_cosmic_companions=true` (:239)

1. **Cosmic taggers.**
   - `TaggerCheckTGM` / `TaggerCheckSTM` / `TaggerCheckFC` evaluate in-scope, in-beam-window **main** clusters, and demoted mains when `evaluate_demoted_mains` is on. They set `Flags::TGM` / `STM` / `FC` (`TaggerCheckTGM.cxx:327`, `TaggerCheckSTM.cxx:641`, `TaggerCheckFC.cxx:216`).
   - The Q/L LM tagger stamps the scalar `lm_flag` (0 pass, 1 low energy, 2 light mismatch) on every cluster of a matched bundle (`match/src/QLMatching.cxx:3703-3705`).
   - Associated clusters are never tagger candidates.
2. **Per bundle** (`clus/src/TaggerCheckNeutrino.cxx:2117-2353`): activities are the in-window mains and demoted mains grouped by `matched_flash_gid`. Every activity is recorded in `act_*` (`:2230-2247`). Then `pick()` runs (`:2251-2296`):
   ```cpp
   if (m_nu_skip_cosmic) { if (tgm || stm || lm > 0) continue; }   // :2254-2263
   if (c != legacy_main && per_bundle_min_len > 0 && c->get_length() < per_bundle_min_len) continue;
   if (!best || c->get_length() > best->get_length()) best = c;    // longest wins
   ```
   `pick()` runs first over the bundle's mains. If none survives, it runs over the demoted mains (`:2297-2299`). If still none, the bundle yields **no row** (`:2300-2304`).
3. **Companions.** The associated clusters of that gid are the candidate's companions. TGM/STM-tagged companions ≥ 15 cm are dropped. A companion is never promoted to candidate.
4. **Neutrino PR.** Candidates are sorted longest-first (`nu_index` 0 = primary), and the full neutrino PR (vertex, particle ID, taggers, BDTs) runs once per candidate into slot `nu<i>`.

The owner's picture ("the main cluster is judged first; if it is cosmic, e.g. TGM,
look at the neutrino candidate") is right. It happens **within one Q-L bundle**
(one flash gid): a cosmic-tagged main is skipped, the longest untagged main of the
same bundle wins, and failing that the longest untagged demoted main. The saved
`T_tagger`/`T_kine` row describes that surviving candidate, not the cosmic.

`act_is_demoted = 1` means `Flags::demoted_main`: a piece restored by
`ClusteringUnmergeBundle` that was a main before the flash-time merge. **It does not
mean "tagged cosmic"**; a cosmic-tagged main is simply skipped and keeps its flags.

The prototype (`prototype_base/wire-cell/pid/apps/wire-cell-prod-nue.cxx:1335-1363`)
builds a `NeutrinoID` for every in-beam flash-TPC pair, with no TGM/STM/LM veto, no
length order and no demoted fallback. There the cosmic verdicts only reach later
analysis via `T_match.event_type`. The veto is a deliberate toolkit addition.

## 4. Consistency of the cosmic taggers with the PR output (full sample)

Rows = `T_tagger` rows: 6 236 in 6 138 events (6 040 events with 1 row, 98 with 2).
`act_*` covers 19 820 activities: 483 with TGM=1, 134 with STM=1, 10 415 with FC=1.

### 4.1 The selected activity is never cosmic-tagged, by construction

| Selected activity (`act_is_selected==1`) | Rows |
|---|---|
| exactly one selected per row | 6 236 / 6 236 |
| TGM=0, STM=0, `lm_flag`≤0 | **6 236 / 6 236** |

Zero tagged candidates is what `pick()` guarantees, not independent evidence that
the taggers and PR agree. The measurements are the residuals below.

### 4.2 Bundles where a cosmic tag changed the choice

| Bundle status (`bundle_has_vetoed`, `sel_demoted`) | Rows | Candidate vertex within 5 cm of a true ν vertex | median distance |
|---|---|---|---|
| no tagged activity in the bundle | 5 653 (5 502 with a vertex) | 0.684 | 1.7 cm |
| tagged activity vetoed → another **main** chosen | 371 (362) | 0.655 | 1.7 cm |
| tagged activity vetoed → **demoted-main fallback** chosen | 212 (212) | 0.472 | 7.7 cm |

After a veto, the other-main choice associates with a true neutrino vertex about as
often as an untouched bundle. The demoted-main fallback is visibly weaker (0.47,
with 34 % of rows > 50 cm from any true vertex).

**Example of a good fallback: `r471_s69_e12`.**
- Bundle gid 1000002 has 7 activities, one TGM/STM-vetoed.
- The fallback picked demoted main 13.
- It sits 2.1 cm from a true numu QE CC vertex (Etot 587.8, Edep 444.8 MeV).
- Reco: Enu 489.6 MeV, numu 2.57, `neutrino_type` 4.

Bundles where **every** activity was vetoed yield no row, so this production output
cannot count them. They are among the 7 078 no-candidate events.

### 4.3 `T_cluster` tagger columns are unusable (defect, reported not fixed)

`T_cluster.tgm/stm/fc/lm/beam_flash` are 0 in **all** 13 216 files, while `act_*`
has 483 TGM, 134 STM and 10 415 FC. The cause: `SbndPrMagnifyTrackingVisitor.cxx:349-353`
reads the lowercase `Flags::tgm / short_track_muon / fully_contained /
light_mismatch / beam_flash`. Only `ClusteringTaggerFlagTransfer.cxx:90,102`
(uBooNE verdict import, not in the SBND pipeline) sets those. The SBND taggers set
uppercase `Flags::TGM/STM/FC`, and LM is the scalar `lm_flag`, not a flag.

**Use `act_tgm/act_stm/act_fc/act_lm`**, not `T_cluster`, for tagger verdicts.
Changing the writer would change the ROOT output values, so it would need a
default-OFF knob and a gate. It is not done here.

### 4.4 The LM arm never fired

`act_lm` = 0 on all 19 820 activities. There is not a single −1 (absent), so the LM
tagger ran and passed every in-window activity. In this sample every veto in 4.2
came from TGM/STM; the LM condition of `pick()` never removed anything.

### 4.5 `T_tagger.cluster_id` is not always the candidate

| `cluster_id` vs selected `act_cluster_id` (`cid_relation`) | Rows | vertex within 5 cm |
|---|---|---|
| same | 5 748 | 0.691 |
| another activity of the bundle | 109 | 0.431 |
| not in `act_*` at all | 379 | 0.493 |

`cluster_id` is `main_cluster->get_cluster_id()` when the row is written
(`TaggerCheckNeutrino.cxx:3547`). The DL/traditional vertex step can repoint
`main_cluster` after selection (`swap_main_cluster`, comment at `:2413`). **For
"which activity did PR run on", use `act_cluster_id[argmax(act_is_selected)]`.**
The 488 repointed rows also associate worse with truth. Example: `r471_s25_e25`,
selected 11, `cluster_id` 29, vertex 279 cm from any true vertex.

### 4.6 `neutrino_type`, and the PR's own cosmic tagger

With `neutrino_type_bitmask`, bit 1 = cosmic (PR-level `NeutrinoTaggerCosmic`),
bit 2 = numu CC, bit 3 = NC, bit 5 = nue.

| `neutrino_type` | 4 | 8 | 6 | 10 | 0 | 40 | 36 |
|---|---|---|---|---|---|---|---|
| meaning | νμCC | NC | cosmic+νμCC | cosmic+NC | no vertex | nue+NC | nue+νμCC |
| rows | 3 399 | 1 838 | 576 | 244 | 160 | 18 | 1 |

- **No vertex:** the 160 `neutrino_type=0` rows are exactly the rows with the default vertex (0,0,0). The candidate got a row but no neutrino vertex.
- **Two cosmic flags agree:** `cosmict_flag` equals bit 1 of `neutrino_type` in 6 236/6 236 rows.

The PR-level cosmic tagger and TGM/STM measure different things, and they are
largely independent here:

| | `cosmict_flag`=0 | `cosmict_flag`=1 |
|---|---|---|
| bundle without a TGM/STM veto | 4 950 | 703 (12.4 %) |
| bundle with a TGM/STM veto | 466 | 117 (20.1 %) |

A bundle that already contained a tagged cosmic is somewhat more likely to have its
surviving candidate also called cosmic by the PR. The rate is 20 % vs 12 %, so most
fallback candidates pass the PR cosmic tagger.

## 5. Truth ↔ reco

**Method: association, not truth matching.** Each candidate is associated with the
**nearest** true interaction vertex in its event (3D distance, generator
coordinates vs `T_tagger` `nu_x/y/z`). In rockbox events with 1-5 interactions, a
large distance means the candidate is not at any true ν vertex (a cosmic, or a
secondary deposit). It is **not** a resolution tail.

The principled join, which is not done here, is Bee's `truth_trackid_labeled`
point set (per-point G4 trackid → interaction). Using it would turn "near a true
vertex" into "made of that interaction's charge".

### 5.1 Vertex association (6 076 rows with a vertex)

| < 3 cm | < 5 cm | < 20 cm | > 50 cm | median |
|---|---|---|---|---|
| 0.616 | 0.674 | 0.758 | 0.180 | 1.7 cm |

Every candidate within 5 cm is at an interaction with Edep > 0 (4 098/4 098). The
second candidate of a two-row event (`nu_index`=1, 72 rows) associates like the
primary (0.639 within 5 cm).

Spot check `r471_s0_e10`:
- truth `1 numu QE CC Etot 1345.8 MeV Edep 671.1 MeV` at (−31.19, −26.57, 265.75);
- reco candidate cluster 4 (248.6 cm) at (−30.98, −26.47, 265.81), 0.25 cm away;
- Enu 811.7 MeV, numu 5.58, nue −15, `neutrino_type` 4;
- the event's second true interaction (MEC NC, Edep 0) produced nothing.

### 5.2 Scores by true class (candidates within 5 cm)

| True class | Rows | numu_score median | numu>0 | nue_score = −15 | nue>0 | νμCC bit | nue bit | cosmic bit |
|---|---|---|---|---|---|---|---|---|
| νμ CC | 3 449 | 3.01 | 0.921 | 0.924 | 0.009 | 0.887 | 0.001 | 0.075 |
| NC | 617 | −1.03 | 0.190 | 0.930 | 0.008 | 0.177 | 0.003 | 0.044 |
| νe CC | 32 | 0.35 | 0.625 | 0.094 | 0.656 | 0.344 | 0.281 | 0.031 |
| not near any true vertex (≥ 20 cm) | 1 470 | −1.20 | 0.269 | 0.917 | 0.010 | 0.352 | 0.002 | 0.281 |

`nue_score = −15` is the scorer's sentinel for candidates the nue BDT does not
evaluate. The BDT weights are the uBooNE-trained XMLs (`clus.jsonnet:2414-2440`),
so the scores are uncalibrated for SBND. The νe CC row has only 32 entries; the
sample has 245 true νe interactions.

### 5.3 Reco energy (candidates within 5 cm, `reco_Enu > 0`)

| True class | Rows | reco Enu / Etot: median [Q1, Q3] | reco Enu / Edep: median [Q1, Q3] |
|---|---|---|---|
| νμ CC | 3 449 | 0.738 [0.530, 0.911] | 1.184 [1.044, 1.336] |
| NC | 617 | 0.295 [0.165, 0.439] | 0.954 [0.756, 1.097] |
| νe CC | 32 | 0.802 [0.612, 1.006] | 0.995 [0.872, 1.123] |

`Edep` is the deposited energy; it omits the muon/pion masses, neutrinos, neutrons
and energy deposited outside the active volume. So `reco/Edep > 1` for νμ CC is
expected (e.g. the muon mass is added back), and `reco/Etot` is the number relevant
to a neutrino-energy measurement.

### 5.4 Why 7 078 events have no candidate

Candidate presence vs the largest true `Edep` in the event:

| max Edep (MeV) | Events | with a candidate | fraction |
|---|---|---|---|
| 0 | 24 | 0 | 0.000 |
| (0, 20) | 3 491 | 114 | 0.033 |
| [20, 50) | 1 046 | 169 | 0.162 |
| [50, 100) | 801 | 251 | 0.313 |
| [100, 200) | 1 221 | 517 | 0.423 |
| [200, 500) | 3 127 | 2 130 | 0.681 |
| ≥ 500 | 3 506 | 2 957 | 0.843 |

Almost half of the no-candidate events (3 401 of 7 078) have < 20 MeV deposited by any
neutrino: the interaction was upstream in the rockbox or outside the TPC, so the
"no candidate" is correct. The 114 candidates in those events are cosmic or
secondary activity called a neutrino. Above 500 MeV, 84 % of events have a
candidate. Per truth interaction with Edep ≥ 500 MeV, a candidate vertex lies
within 5 cm for 63.5 % (44.9 % at 200-500, 19.9 % at 100-200 MeV).

This is presence, not efficiency: there are no beam-window, FV or cosmic-overlap
cuts on the truth, and the truth tree drops interactions with no particle above
10 MeV.

## 6. Open items (reported, not fixed)

1. **`T_cluster` tagger columns always 0** (section 4.3). They read the lowercase uBooNE-import flags. A fix needs a default-OFF knob in `SbndPrMagnifyTrackingVisitor` plus a gate.
2. **`T_tagger.cluster_id` ≠ selected activity** in 488/6 236 rows (section 4.5). The branch name suggests "the candidate". Either document that it is post-swap, or add the selected id as its own branch (schema change behind a knob).
3. **Labeler header comment stale** (`TensorSetLabeler.h:125-128`: "energy = Edep, not the neutrino total energy"). The code writes both. This lives in larwirecell, not this repo.
4. **BDT weights are uBooNE-trained**, so the SBND score cuts are uncalibrated (section 5.2).
5. **Demoted-main fallback** candidates associate with a true vertex in 47 % of cases vs 68 % for ordinary candidates (section 4.2). This is a candidate for a hand scan before anyone relies on those 212 rows.

## 7. Recommended next step

Replace the nearest-vertex association (section 5) with a charge-based truth match:
- read `truth_trackid_labeled` from the same Bee zip;
- assign each selected activity's points to G4 trackid → interaction;
- record the purity and the matched interaction per `T_tagger` row.

With that in hand, re-cut sections 4.2 and 5.2 into neutrino-signal vs
cosmic-contaminated candidates. Then a Bee scan set of the 212 demoted-main
fallback rows would show whether the fallback should keep its current gate.

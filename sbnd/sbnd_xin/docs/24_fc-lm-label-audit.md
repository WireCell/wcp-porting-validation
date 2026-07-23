# 24 — Are "fully contained" (FC) and "light mismatch" (LM) labelled anywhere in the SBND TGM/STM chain?

**Question.** The SBND neutrino-selection chain of doc 23 takes the Q/L matching
result and runs `TaggerCheckTGM` → `TaggerCheckSTM`, producing a per-bundle
label table. The WCP prototype attaches two more per-bundle labels at exactly
this point — **fully contained** and **light mismatch**. Are either of them
produced, stored, or surfaced in our chain?

**Answer, in one line each.**

| label | status in the SBND chain | nature of the gap |
|---|---|---|
| **fully contained (FC)** | **computed but not wired in** | The FC computation is fully ported and lives in `Facade::cluster_fc_check`. It is surfaced as `tagger_info.match_isFC` — but only by `TaggerCheckNeutrino`, which is deliberately *not* in the nusel pipeline. Inside `TaggerCheckSTM` the same result is computed and thrown away (TRACE log only). |
| **light mismatch (LM)** | **never implemented** | Nothing in the toolkit ports `check_LM` / `check_LM_cuts` / `check_LM_bdt`. The flag *string* `Flags::light_mismatch` exists, but its only producer anywhere in the repo is the uBooNE ROOT import path. |

So: neither label reaches `nusel-table.tsv` (its verdict columns are only
`tgm`, `stm`, `label` — `nusel_extract.py:54`), but the reasons are different
and the remedies are different in kind.

> **Scope.** This is a read-only audit. No code, config, or output was changed.
> Every claim below is a `file:line` citation you can re-check with `grep`.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit

# toolkit side — who sets / reads the two flags
grep -rn "Flags::fully_contained\|Flags::light_mismatch" clus/ match/ img/ --include=*.cxx --include=*.h
grep -rn "has_fully_contained\|has_light_mismatch" --include=*.cxx --include=*.h . | grep -v prototype_base
grep -rn "cluster_fc_check\|match_isFC" clus/ root/ --include=*.cxx --include=*.h
grep -rn "check_LM\|check_fully_contained" clus/ match/ img/ util/ aux/ --include=*.cxx --include=*.h   # → empty

# prototype side — where the two labels are produced
grep -n "check_fully_contained\|check_LM" \
  prototype_base/uboone_nusel_app/apps/prod-wire-cell-matching-nusel-port.cxx
sed -n '598,760p'  prototype_base/2dtoy/src/ToyFiducial.cxx     # check_LM / _cuts / _bdt
sed -n '816,902p'  prototype_base/2dtoy/src/ToyFiducial.cxx     # check_fully_contained
```

## 1. What the prototype does

Both labels are set in the same per-bundle loop of the neutrino-selection
block, right after the matched bundles exist —
`prototype_base/uboone_nusel_app/apps/prod-wire-cell-matching-nusel-port.cxx`.
That loop packs verdicts into an `event_type` bitmask
(header comment at `:875-881`):

```
bit 0 (=1)   intime flash present
bit 1 (10)   light mismatch          <-- LM
bit 2 (100)  fully contained         <-- FC   ("FC (stopping muon)")
bit 3 (1000) through-going muon      <-- TGM  (we have this)
bit 4 (10000) low energy
```

- **FC** — `:1002-1007`. `fid->check_fully_contained(bundle, offset_x,
  ct_point_cloud, old_new_cluster_map, &_fc_breakdown)` sets bit 2; if the
  current main cluster fails it retries with `flag=2` (the *original*,
  pre-merge main cluster). The `_fc_breakdown` out-parameter records *why* it
  failed: implementation at
  `prototype_base/2dtoy/src/ToyFiducial.cxx:816` sets
  `1U<<2` = outside the fiducial volume (FV),
  `1U<<1` = failed `check_signal_processing` (SP), and
  `1U` = failed `check_dead_volume` (DC).
  These are the `match_notFC_FV / match_notFC_SP / match_notFC_DC` branches
  of the downstream eval tree
  (`prototype_base/pid/apps/wire-cell-prod-stm-port.cxx:1725`).
  The per-event summary variable is `match_isFC`
  (`prototype_base/uboone_bdt_app/inc/WCPLEEANA/eval.h:23`), and it is a direct
  **BDT input variable** (`prototype_base/pid/src/NeutrinoID_numu_bdts.h:81`).
- **LM** — `:1013-1034`. Three independent implementations are evaluated per
  bundle and packed into `_LM_type`:
  `check_LM` (`ToyFiducial.cxx:598`, the historical cuts),
  `check_LM_cuts` (`:661`, retuned cuts), and
  `check_LM_bdt` (`:725`, BDT). Each returns `1` = low energy, `2` = light
  mismatch, `0` = pass. Only `check_LM` feeds `event_type` (bits 1 and 4); the
  other two only fill `_LM_type` for comparison. `light_mismatch` is likewise a
  branch of the eval tree
  (`prototype_base/uboone_bdt_app/inc/WCPLEEANA/eval.h:34`).

The important structural point: **LM is a Q/L-quality verdict, not a geometry
verdict.** `check_LM` reads `bundle->get_pred_pmt_light()`,
`flash->get_total_PE()`, `flash->get_PE(i)`, `bundle->get_ks_dis()`,
`get_chi2()`, `get_ndf()`, `get_flag_close_to_PMT()`,
`get_flag_at_x_boundary()` and the cluster's `uvwt` extent — i.e. it lives
where the `FlashTPCBundle` objects are still alive, which in the prototype is
the matching app itself. FC, by contrast, is pure geometry on the cluster's
extreme points and needs only the fiducial volume.

## 2. What the toolkit has — FC

The FC algorithm **is ported and is in use**:

- `Facade::cluster_fc_check(Cluster&, IDetectorVolumes::pointer)` —
  `clus/src/Clustering_Util.cxx:75`, declared at
  `clus/inc/WireCellClus/ClusteringFuncs.h:312`. Its header comment states the
  intent explicitly: *"Used by: TaggerCheckNeutrino to fill
  tagger_info.match_isFC; TaggerCheckSTM to drive STM / TGM classification."*
  It is the same three-test structure as the prototype — direct FV test →
  wire-angle gate → `check_signal_processing` → PCA-angle gate →
  `check_dead_volume` — run in two rounds over the steiner boundary
  (round 1 `flag_cosmic=true`, round 2 `flag_cosmic=false`).
- The FV / SP / DC primitives it calls are all present on `FiducialUtils`:
  `inside_fiducial_volume`, `check_signal_processing`, `check_dead_volume`
  (`clus/inc/WireCellClus/FiducialUtils.h:73,79,81`).

Two consumers exist, and **neither delivers a label to our chain**:

1. **`TaggerCheckNeutrino`** (`clus/src/TaggerCheckNeutrino.cxx:517-520`):
   ```cpp
   auto fc_result = Facade::cluster_fc_check(*main_cluster, m_dv);
   tagger_info.match_isFC = fc_result.is_fc ? 1.0f : 0.0f;
   ```
   `match_isFC` is a real field (`clus/inc/WireCellClus/NeutrinoTaggerInfo.h:1351`)
   and is consumed downstream by the ported BDT scorers
   (`root/src/UbooneNumuBDTScorer.cxx:508`,
   `root/src/UbooneNueBDTScorer.cxx:1659`). **But `tagger_check_neutrino` is
   deliberately excluded from the nusel pipeline** — doc 23 §1 says so, and
   `run_nusel_evt.sh:44` sets
   `PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm"`.
   It is available by name in
   `cfg/pgrapher/experiment/sbnd/clus.jsonnet:513` (`tagger_check_neutrino`),
   just not selected.
2. **`TaggerCheckSTM`** (`clus/src/TaggerCheckSTM.cxx:1890`) calls
   `cluster_fc_check` as the *first* step of `check_stm_conditions` and uses
   `fc_result.is_fc` purely as an early exit — an FC cluster cannot be a
   stopped muon, so it returns `false` after a `SPDLOG_LOGGER_TRACE`
   ("Mid Point: A"). **The verdict is discarded**: no flag is set, nothing is
   written to the cluster's scalar PC, and the line is TRACE (below the INFO
   level `nusel_extract.py` parses). Every bundle STM evaluates has therefore
   already had its FC status computed — we simply throw it away.

Note the prototype's `_fc_breakdown` (FV/SP/DC failure reason) has **no**
toolkit counterpart: `FCCheckResult`
(`clus/inc/WireCellClus/ClusteringFuncs.h:288`) carries `is_fc`,
`exit_wcps`, `exit_boundary_set`, `boundary_first/second` — the exit geometry
STM needs, but not a failure-mode bitmask. So `match_notFC_FV / _SP / _DC` are
unported too.

**Also note the definitional divergence.** The prototype's
`check_fully_contained` iterates the cluster's `get_extreme_wcps()` groups with
an `offset_x` derived from the flash time, and retries on the *original*
pre-merge main cluster. `cluster_fc_check` iterates the **steiner** boundary
groups (requiring a non-empty `steiner_pc`), applies no `offset_x` (points are
already T0-corrected by `switch_scope` — same convention documented in the
`TaggerCheckTGM` header, `clus/src/TaggerCheckTGM.cxx:15-17`), and has no
`flag=2` original-cluster retry. If FC were surfaced, it would be *an* FC
definition, not bit-comparable to the prototype's.

## 3. What the toolkit has — LM

Nothing computes it.

- `grep -rn "check_LM" clus/ match/ img/ util/ aux/` returns **no hits**.
  `check_LM`, `check_LM_cuts` and `check_LM_bdt` are unported.
- `Flags::light_mismatch` exists as a name only
  (`clus/inc/WireCellClus/ClusteringFuncs.h:56-57`), together with
  `Flags::low_energy` (`:54`), `Flags::fully_contained` (`:59`),
  `Flags::tgm` (`:51`), `Flags::short_track_muon` (`:62`) and
  `Flags::full_detector_dead` (`:65`).
- The **only** place any of those six is ever `set_flag`-ed is
  `clus/src/ClusteringTaggerFlagTransfer.cxx:84-107`, which does no physics: it
  reads a per-cluster local point cloud named `tagger_info` and copies the
  integer keys `has_beam_flash`, `has_tgm`, `has_low_energy`,
  `has_light_mismatch`, `has_fully_contained`, `has_short_track_muon`,
  `has_full_detector_dead` into cluster flags (all gated on
  `has_beam_flash`, mirroring the prototype's beam-flash gate).
- The **only** producer of that `tagger_info` PC in the whole repo is
  `root/src/UbooneClusterSource.cxx:704-758` — the uBooNE reader that imports
  verdicts already computed by a WCP job into a ROOT file. It is not part of
  any SBND path.
- Accordingly `TaggerUtils::is_light_mismatch` / `is_fully_contained`
  (`clus/src/ClusteringTaggerFlagTransfer.cxx:140,144`) have **zero callers**
  anywhere in `clus/`, `match/` or `img/`.

Net effect for SBND: `ClusteringTaggerFlagTransfer` is not in the SBND
pipeline, and even if it were it would find no `tagger_info` PC and return
immediately (`:49-52`). Both flags are permanently false.

### Why LM can't simply be bolted onto the nusel pipeline

The nusel PR job loads a **persisted point-cloud tree**
(`work/ql_evt<ID>/pctree-evt<ID>.tar.gz`) — the Q/L bundles themselves are
long gone. The observables `check_LM` needs live on `TimingTPCBundle`
(`match/inc/WireCellMatch/TimingTPCBundle.h`: `get_ks_dis():152`,
`get_chi2():148`, `get_flag_close_to_PMT():109`,
`get_flag_at_x_boundary():110`, plus predicted/measured PE) and exist only
during the `QLMatching` visit. What `QLMatching` actually persists onto
clusters is just the association: `set_flag("main_cluster")` /
`set_flag("associated_cluster")` (`match/src/QLMatching.cxx:1243,1248`) and the
scalars `flash` / `matched_flash_gid` (`:3405-3412`), plus `cluster_t0`
(`clus/src/Facade_Cluster.cxx:188`). None of the LM inputs survive.

So the prototype's home for LM — the matched-bundle loop of the matching app —
maps to `QLMatching`, not to a pctree-based PR visitor. That is a statement of
where the information is, not a proposal; no design work is done here.

## 4. Summary map (prototype → toolkit)

| prototype | where | toolkit counterpart | reaches SBND nusel chain? |
|---|---|---|---|
| `event_type` bit 3 — TGM, `check_tgm` | `ToyFiducial.cxx:904` | `TaggerCheckTGM` → `Flags::TGM` (`clus/src/TaggerCheckTGM.cxx:122`) | **yes** (`tgm` column) |
| STM, `check_stm` | `pid/.../ToyFiducial.h:81` | `TaggerCheckSTM` → `Flags::STM` (`clus/src/TaggerCheckSTM.cxx:165`) | **yes** (`stm` column) |
| `event_type` bit 2 — FC, `check_fully_contained` | `ToyFiducial.cxx:816` | `Facade::cluster_fc_check` (`clus/src/Clustering_Util.cxx:75`) → `tagger_info.match_isFC` (`TaggerCheckNeutrino.cxx:519`) | **no** — `tagger_check_neutrino` not in the pipeline; STM's copy is discarded |
| `_fc_breakdown` FV/SP/DC → `match_notFC_*` | `ToyFiducial.cxx:852,885,892` | *(none)* — `FCCheckResult` has no failure mask | no |
| `event_type` bit 1 — LM, `check_LM` | `ToyFiducial.cxx:598` | *(none)* | no |
| `_LM_type` bits 2-3 — `check_LM_cuts` | `ToyFiducial.cxx:661` | *(none)* | no |
| `_LM_type` bits 4-5 — `check_LM_bdt` | `ToyFiducial.cxx:725` | *(none)* | no |
| `event_type` bit 4 — low energy | `nusel-port.cxx:1015` | `Flags::low_energy` name only, uBooNE-import producer | no |
| `match_isFC` (BDT input) | `WCPLEEANA/eval.h:23` | `NeutrinoTaggerInfo::match_isFC` (`:1351`), read by `Uboone{Numu,Nue}BDTScorer` | not in this chain |

## 5. Bottom line

- **FC is a wiring question, not a porting question.** The verdict is already
  being computed for every bundle the STM tagger touches; it is simply not
  recorded. Two independent routes exist to surface it (promote the
  `TaggerCheckSTM` early-exit result, or add `tagger_check_neutrino` to the
  pipeline for its `match_isFC`), and neither requires new physics. Any such
  change must respect the prime directive — knob defaulting OFF, gate PASS —
  and the FC definition would be the steiner/no-offset toolkit one, not
  bit-comparable to the prototype (§2).
- **LM is a genuine porting gap** at the Q/L layer, not the PR layer. The three
  prototype implementations are all absent, and the inputs they need are not
  persisted past `QLMatching`. Adding it downstream of the pctree is not
  possible without first persisting the bundle-quality observables.

No changes were made. Next step is the owner's call.

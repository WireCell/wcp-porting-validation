# 2 — Neutrino PR chain on SBND: uBooNE gap analysis + validation plan + nueCC sample

**Status: PLAN (with executed smoke test), plus one worklist item since executed.** No
production default changed. The one new artifact besides docs is the 2-event smoke root
`work-nuecc48-prsmoke/` (§5.3). Everything else here is analysis of what exists and a
step-by-step plan for validating `tagger_check_neutrino` (and the rest of the uBooNE
pattern-recognition tail) on SBND.

**Update 2026-07-30** — the first §2e(i) correctness item is half-closed: the SSM
beam-line vectors are now config knobs (`ssm_target_dir` / `ssm_absorber_dir`,
defaults = the uBooNE numbers ⇒ nothing moved). C++ and jsonnet *were* touched for
that; see **§2e(i-a)** for the plumbing, the gate, and the caveats. The SBND *values*
remain unknown — this is a relocation, not a calibration.

**Update 2026-07-30 (second)** — the §2e(ii) MIP normalization, this worklist's
highest-priority tune, is **implemented and ON in SBND production**: two knobs
(`mip_dqdx` = 56000, `mip_dqdx_median` = 48000 e/cm) shipped in toolkit `3d71e111`,
designed and gated in doc `pr/8`. **§2e(ii-a)** records the current state, the residual
call sites that still fall back to the uBooNE literals, and the one place where the
threading is incomplete in a way that also disables `proton_dir_vote`. The `48000` is a
ratio-preserving placeholder, not an SBND measurement.

Companion docs: `1_beam-window-cosmic-vs-nu-division.md` (same-day census of the
cosmic/nu-candidate split on both data samples; it created the `work-nuecc48-nuf` arm
this doc builds on), `../../../docs/sbnd-pattern-recognition.md` (the M0–M8 port log
this extends), `../19_PR_integration.md` (the 7-section port survey),
`toolkit/clus/docs/tagger/tagger_validation_plan.md` (the uBooNE toolkit-vs-prototype
statistical method), `../59_full1k-production-scan.md` (the 1000-event data production),
`../61_nusel-handscan-key.md` / `../62_stm-baseline-and-protons.md` (hand-scan method +
truth-table pattern this plan reuses).

## Repro block

```bash
SB=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TK=/nfs/data/1/xqian/toolkit-dev/toolkit

# The two chains, side by side:
grep -n "cm_pipeline" $TK/qlport/uboone-mabc.jsonnet              # uBooNE stage list (lines ~1244-1256)
grep -n "pipeline_names=" $TK/cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet   # SBND production default

# SBND neutrino visitor is wired but not in the production pipeline:
grep -n "tagger_check_neutrino" $TK/cfg/pgrapher/experiment/sbnd/clus.jsonnet

# nu-candidate population of the 1000-event data production (503 events, §3.2):
awk 'NR>1{c[$NF]++} END{for(k in c) print k, c[k]}' $SB/scan-d59k/census.tsv

# nueCC 48-event sample state (§5):
awk 'NR>1{c[$7]++} END{for(k in c) print k, c[k]}' $SB/work-nuecc48-nuf/nusel-events.tsv
ls $SB/work-nuecc48-nuf/nusel_evt*/nusel-evt*.tsv | wc -l          # 48

# Smoke test (§5.3) — exact command in that section; evidence:
grep "selected main cluster" $SB/work-nuecc48-prsmoke/nupr_evt*/wct_nupr_evt*.log
unzip -l $SB/work-nuecc48-prsmoke/nupr_evt172230/mabc-pr.zip
```

---

## 1. The uBooNE reference chain (what "the full chain" means)

The uBooNE PR job is `qlport/uboone-mabc.jsonnet`: WCP `nuselEval_*.root` (blobs +
clusters + optical, already cosmic-tagged by the WCP prototype upstream) →
`UbooneClusterSource` → one `MultiAlgBlobClustering` node running an ordered visitor
pipeline → Bee zip + `track_com_<run>_<ev>.root`.

The `cm_pipeline` (uboone-mabc.jsonnet ~1244–1256):

| # | stage | C++ | role |
|---|---|---|---|
| 1 | `tagger_flag_transfer` | `ClusteringTaggerFlagTransfer` | import WCP's upstream STM/TGM/LM flags |
| 2 | `clustering_recovering_bundle` | `ClusteringRecoveringBundle` | rebuild main+associated bundles from flash-merge provenance |
| 3 | `switch_scope` | `ClusteringSwitchScope` | T0-corrected coordinates |
| 4 | `steiner` (retiler=`improve_cluster_2`) | `CreateSteinerGraph` | steiner point cloud/graph per cluster |
| 5 | `fiducialutils` | `MakeFiducialUtils` | FV utility (taggers silently no-op without it) |
| 6 | `tagger_check_neutrino` | `TaggerCheckNeutrino` | **the whole WCP NeutrinoID** (below) |
| 7 | `numu_bdt_scorer` | `UbooneNumuBDTScorer` | TMVA numu 1–3 + cosmic-10 + XGB combiner |
| 8 | `nue_bdt_scorer` | `UbooneNueBDTScorer` | ~30 nue sub-BDTs + XGB combiner |
| 9 | `tracking_visitor` | `UbooneMagnifyTrackingVisitor` | `track_com_*.root` projection/track trees |
| 10 | `tagger_output` | `UbooneTaggerOutputVisitor` | appends `T_tagger` (~1200 branches) + `T_kine` |

(`cm.tagger_check_stm` and `cm.retile` exist but are commented out — uBooNE gets those
verdicts from the prototype upstream. This is the single biggest structural difference
from SBND, where TGM/STM/FC/LM are computed in-toolkit.)

Inside stage 6, `TaggerCheckNeutrino::visit()` (`clus/src/TaggerCheckNeutrino.cxx:124–558`)
runs the full NeutrinoID sequence, every step instrumented with a
`TaggerCheckNeutrino timing: <stage> took .. ms` log line:

main-cluster selection → `preload_clusters` → `find_proto_vertex` (break, 2 rounds) →
`clustering_points` → `separate_track_shower` → `determine_direction` →
`shower_determining_in_main_cluster` → `determine_main_vertex` → same block over the
other beam-flash clusters → `deghosting` → `determine_overall_main_vertex[_DL]` →
`improve_vertex` → `examine_direction` → `shower_clustering_with_nv` (incl. pi0
pairing) → `init_tagger_info` → `cosmic_tagger` → `numu_tagger` → `ssm_tagger` →
`nue_tagger` → `singlephoton_tagger` → `cluster_fc_check` → `fill_kine_tree` →
finalize (vertices, PF tree, fitted charge).

Its regression gate (standing policy, `sbnd-pattern-recognition.md` §7): the 35-event
nue MC sweep `qlport/scripts/ab_check.sh <label> gate3` — Bee-zip member hashes +
`wire-cell-uboone-tagger-compare` against the WCP prototype ROOT files — run with ASLR
off (`setarch -R`) and DL/SCN off (`-A dl_weights=`). Any clus/ change made during the
SBND campaign must keep this PASS.

One porting-status caveat: `clus/docs/porting/neutrino_id_function_map.md` is dated
April and its status column is stale — e.g. it marks `singlephoton_tagger` "(EMPTY)"
while `NeutrinoTaggerSinglePhoton.cxx` is 2511 lines and called at
`TaggerCheckNeutrino.cxx:500`. Trust the code + the `*_review.md` docs
(`clus/docs/patternrecognition/`, `clus/docs/tagger/`) over the map's status column.

## 2. Gap analysis: SBND vs uBooNE

SBND production (`wct-pr-perevt.jsonnet` default `pipeline_names`, = the doc-59
1000-event production):
`switch_scope → unmerge_bundle → unmerge_assoc → steiner → fiducialutils →
tagger_check_tgm → tagger_check_stm → tagger_check_fc` (+ `stm_magnify` with
`-stm-fit`). LM is a QLMatching-side flag (`lm_tagger`, doc 34), not a clus_pr visitor.

### 2a. Present and production-ON in SBND

| uBooNE stage | SBND status |
|---|---|
| `switch_scope` | same (must re-run at PR-job head; scope doesn't persist through the pctree) |
| `steiner` | same + SBND beam-window gate (`beam_window_only`, doc 56); `require_beam_flash=false` (SBND has no WCP `beam_flash` flag) |
| `fiducialutils` | same (SBND FV = `sbnd_pr_fv` box) |
| — (upstream in WCP) | `tagger_check_tgm` / `tagger_check_stm` / `tagger_check_fc`: SBND-new in-toolkit ports (docs 24–39, 40–66), TGM→STM→FC order, FC verdict-neutral last |

### 2b. Present, demo-validated, NOT enabled in production — **the gap this plan closes**

| piece | state |
|---|---|
| `tagger_check_neutrino` | wired in `clus.jsonnet` `cm_by_name` (~line 909) with SBND args (`sbnd_track_fitting.json`, `ParticleDataSet`, `recombination_model=sbnd_box_recomb`, beam-window gate); milestone-M5 demo on 7 events (2026-07-10); **absent from production `pipeline_names`**; re-verified by the §5.3 smoke |
| DL (SCN) vertex | **DEFAULT ON for SBND since 2026-07-30 (doc `pr/4`)** — adopted on evt 18253/1/172230, where the geometric vertex sat at the end of a proton track and the DL vertex moved it 9.73 cm onto the true interaction point. Weights are still **uBooNE-trained** (G3 open); requires the libpython preload or it silently falls back (pr/4 §3); stays out of all identity gates, which keep passing `dl_weights=''` |

### 2c. Missing — deliberate simplification (do NOT port)

| uBooNE stage | why SBND doesn't need it |
|---|---|
| `tagger_flag_transfer` | transfers WCP-prototype upstream verdicts; SBND computes TGM/STM/FC/LM natively. Nothing to transfer. |
| `clustering_recovering_bundle` | rebuilds bundles from WCP flash-merge provenance; SBND's `unmerge_bundle` (doc 45) + `unmerge_assoc` (docs 50–52) solve the same problem from SBND's own `real_cluster_*` / `assoc_cluster_*` provenance, with the merge undone rather than recovered. |
| uBooNE single-main selection (`beam_window_low=high=0`) | SBND uses the real experiment beam window `[0.2, 2.2) µs` on `cluster_t0`; since doc 56 it also gates which bundles get steiner+taggers at all. This is a deliberate improvement, not a gap. |
| old TMVA `cal_bdts()` path | superseded by XGBoost even on uBooNE; never ported. |
| `pio_tagger` as a separate stage | pi0 pairing/mass is subsumed into `NeutrinoShowerClustering` (`pi0_showers`, `map_pio_id_mass`) and consumed by the nue/single-photon taggers. Same on SBND for free once `tagger_check_neutrino` runs. |

### 2d. Missing — real gaps, needed for full parity (deferred work items)

| # | gap | what it takes |
|---|---|---|
| G1 | **numu/nue BDT scorers** (stages 7–8) | `UbooneNumuBDTScorer`/`UbooneNueBDTScorer` load ~35 MicroBooNE-trained XMLs. Running them on SBND would produce numbers with uBooNE priors baked in. Needs: SBND feature extraction first (G2), then retraining. Until then the ~200-variable `TaggerInfo` block is computed (by the taggers inside `tagger_check_neutrino`) but never scored. |
| G2 | **`SbndTaggerOutputVisitor` / PR tracking output** (stages 9–10) | **CLOSED 2026-07-30 (doc `pr/3`)**: `SbndPrMagnifyTrackingVisitor` forked (2-APA ChanScheme + per-point `PR::Fit::paf`); `UbooneTaggerOutputVisitor` reused as-is (audited geometry-free) — `tracking-pr.root` with T_tagger/T_kine validated on evt 172230/444187.  The BDT scorers are also wired (uBooNE weights, uncalibrated — G1 retraining still open). |
| G3 | **SCN vertex retraining** + `sparseconvnet`/torch ABI pin | **STILL OPEN, but the net is now in use**: doc `pr/4` made the uBooNE-trained SCN vertex the SBND default (owner call, evt 172230). Defensible because `SCN_Vertex.py`'s voxelizer subtracts the point-cloud min, so the net sees only relative geometry + charge scale — but every PR vertex now carries an other-detector training caveat until this gap closes. |
| G4 | **dQ/dx calibration constants** | uBooNE passes `dQdx_scale=0.1, dQdx_offset=-1000` (uBooNE data calibration) to the neutrino tagger; SBND passes neither (C++ defaults apply). SBND has `mip_dqdx=56000` (doc 48) for STM, but the PR-internal dQ/dx→PID chain (`ParticleDataSet` tables, recombination) needs an SBND calibration decision. Flag: uBooNE corrects SCE via `clus_geom_helper=UbooneGeomHelper`; SBND passes no geom helper (no SCE correction applied — acceptable to first order for SBND, but must be a stated decision, not an accident). |
| G5 | **Tagger threshold recalibration** | cosmic/numu/ssm/nue/single-photon tagger cut values are uBooNE-tuned (drift length, wire pitch, PMT geometry priors). They will *run* on SBND; their verdicts are uncalibrated until validated (checklist V8). |

### 2e. uBooNE-derived input values and cuts inside the chain — the tuning worklist

The ported code carries MicroBooNE numbers at three levels. A code sweep
(2026-07-29, files = `TaggerCheckNeutrino` + all `Neutrino*.cxx` + `TrackFitting` +
`SteinerGrapher` + `ParticleDataSet` + the two track-fitting JSONs) found the
following. **Headline: the config side is already converted to SBND's E = 0.5 kV/cm
(recombination model + all five dE/dx tables), but the C++ side still normalizes every
dQ/dx to uBooNE's MIP value `43e3 e/cm` (0.273 kV/cm) at ~97 sites in 16 files — the
chain currently runs two mutually inconsistent field assumptions.**

**(i) Correctness items — wrong on SBND regardless of value, fix before tuning:**

| where | what | why wrong |
|---|---|---|
| `NeutrinoTaggerSSM.cxx:582-583` | `target_dir(0.46,0.05,0.885)`, `absorber_dir(0.33,0.75,-0.59)` | BNB-target and NuMI-absorber directions **in the uBooNE detector frame**; feed 8 `ssm_*_angle_{target,absorber}` BDT features. SBND needs its own target vector; the NuMI-absorber features may be meaningless there. **Half-closed 2026-07-30: the numbers are now config-fed (`ssm_target_dir` / `ssm_absorber_dir`), defaults unchanged — see §2e(i-a). The remaining gap is the SBND *value*, which nobody has yet.** |
| `TaggerCheckNeutrino.cxx:493` | `nue_tagger(..., apa=0, face=0, ...)` | Single-drift-volume assumption: any vertex in SBND APA 1 gets APA 0's mirrored wire geometry for `flag_prolong_u/v/w` / `flag_parallel`. `NeutrinoTaggerSinglePhoton.cxx:2186-2196` already derives apa/face from `dv->contained_by(vtx_pt)` — the nue path needs the same fix. |
| `NeutrinoTaggerSinglePhoton.cxx:1499-1505` | inline inverse Modified-Box with `0.273` kV/cm, `1.38` g/cm³, `23.6e-6`, `43e3` | Bypasses the configured `sbnd_box_recomb` (0.5 kV/cm) entirely for its `median_dedx`/`mean_dedx`. |

**(i-a) DONE 2026-07-30 — the SSM beam vectors are config-fed** (owner request; the
SBND values are still unknown, so this moves the numbers out of the source, it does
**not** calibrate them).

Repro:

```bash
cd sbnd_xin/work-nuecc48-prsmoke2
./run_ssm_arm.sh /home/xqian/tmp/ssmA                                    # knob off
./run_ssm_arm.sh /home/xqian/tmp/ssmB --tla-code 'ssm_target_dir=[0,0,1]'  # knob on
# cross-binary off-gate arm: build the pre-change clus lib into a scratch dir
# (./wcb build, no install) and
LD_LIBRARY_PATH=/home/xqian/tmp/ssmlib_old ./run_ssm_arm.sh /home/xqian/tmp/ssmA0
# compare: python3 sbnd_xin/ssm_tagger_ab.py <A>/tracking-pr.root <B>/tracking-pr.root
```

`run_ssm_arm.sh` is `run_pr3_evt.sh` for evt 172230 with extra TLAs forwarded
(`wct-pr-perevt.jsonnet` on `work-nuecc48-nuf/ql_evt172230/pctree-evt172230.tar.gz`,
`pipeline_names=[… tagger_check_neutrino, numu/nue_bdt_scorer, tracking_visitor,
tagger_output]`, `dl_weights=''`, `beam_window_us=[0.2,2.2]`).

Two new `TaggerCheckNeutrino` keys, both `[x, y, z]` arrays in the detector frame:

| key | C++ default | meaning |
|---|---|---|
| `ssm_target_dir` | `[0.46, 0.05, 0.885]` | BNB target as seen from MicroBooNE |
| `ssm_absorber_dir` | `[0.33, 0.75, -0.59]` | NuMI absorber as seen from MicroBooNE |

Plumbing: `PatternAlgorithms::m_ssm_target_dir` / `m_ssm_absorber_dir`
(`NeutrinoPatternBase.h`) ← `TaggerCheckNeutrino::configure()` ← `tagger_check_neutrino`
in `cfg/pgrapher/common/clus.jsonnet` (key-suppression idiom, key omitted when `null`)
← SBND `clus.jsonnet` `clus_pr(...)`/`pr(...)` ← `wct-pr-perevt.jsonnet` TLA. The two
consuming sites are unchanged: `NeutrinoTaggerSSM.cxx:908-909` (the 10 cm
initial-direction pair) and `:1124-1125` (the nu / con_nu / prim_nu / track momentum
pairs). SBND sets **neither** — every level defaults to `null`, so the SBND compiled
config and the uBooNE path are untouched.

Three things a future SBND value must account for:

1. **The prototype vectors are not unit vectors**: `|target| = 0.99866`,
   `|absorber| = 1.00970`. `safe_acos()` clamps the dot product, so the *absorber*
   angle saturates at exactly 0 for true angles out to ~7.9°. They are kept verbatim
   for parity; a properly normalized SBND replacement will shift these feature
   distributions even at the same physical direction.
2. A **malformed or zero** array is rejected with a warning and the default kept — a
   zero reference would make every `safe_acos(dot)` return exactly π/2 and silently
   pin all 8 features to 1.5708.
3. `dir_beam(0,0,1)` and `dir_vertical(0,1,0)` on the adjacent lines are frame
   conventions that hold for SBND too and were deliberately **not** lifted.

Verification (evt 18253/1/172230, nueCC48; `ssm_flag_st_kdar = 1`, so the tagger does
fire and none of this is vacuous):

| arm | library | `ssm_target_dir` | `ssm_nu_angle_target` | `ssm_nu_angle_z` |
|---|---|---|---|---|
| A0 | pre-change | (key absent) | 0.85479736 | 0.39195666 |
| A0k | pre-change | `[0,0,1]` | 0.85479736 | 0.39195666 |
| A | post-change | (key absent) | 0.85479736 | 0.39195666 |
| B | post-change | `[0,0,1]` | **0.39195666** | 0.39195666 |

- **Off-gate (A0 vs A, cross-binary).** 1209 of 1216 `T_tagger` branches identical,
  including every `ssm_*`. The 7 that move (`pio_2_v_{acc_length,angle2,dis2}`,
  `shw_sp_pio_2_v_*`, `shw_sp_lol_1_v_angle`) also move in an **A/A control** — same
  multiset, permuted order — so they are pre-existing run-to-run instability, not this
  change. The pre-change library is toolkit `3d71e111` exactly (built into
  `/home/xqian/tmp/ssmlib_old/` with `./wcb build`, no install, and loaded via
  `LD_LIBRARY_PATH`); arm A0k proves the swap really took, since that library ignores
  the key. Both arms therefore differ by this commit alone.
- **Knob-on identity.** With `ssm_target_dir=[0,0,1]` the target reference *is* the
  beam axis, so every `ssm_*_angle_target` must equal `ssm_*_angle_z` bit-for-bit. All
  five pairs do, but **only two of them are live on this event**: `nu` (0.85480 →
  0.39196) and `con_nu` (1.17490 → 0.69469), both from the `:1124-1125` site. The
  `angle_to` pair is π/2 == π/2 and `prim_nu`/`track` are −999 == −999 — trivially
  equal, so they prove nothing. **The `:908-909` site (`angle_to_target_10`) was
  therefore never exercised with a non-degenerate value**; it is verified by
  construction (same local name, same rebind, one line apart) rather than by output.
  A/B moves exactly those two branches; `ssm_*_absorber` is untouched, as it must be.
- **Blast radius.** Neither `UbooneNumuBDTScorer` nor `UbooneNueBDTScorer` reads any
  `ssm_*` feature (`grep -c ssm root/src/Uboone*BDTScorer.cxx` → 0/0), so setting an
  SBND vector moves the `T_tagger` branches only — no BDT score shifts underneath it.
  The 8 features reach the ROOT output through `UbooneTaggerOutputVisitor` and wait
  there for an SSM classifier.
- **Compiled config.** Knob null ⇒ `cmp`-identical to the pre-edit cfg tree (255625 B,
  both arms). Knob set ⇒ exactly one added key. `./build/clus/wcdoctest-clus`: 49/49.

Recorded, not fixed (CLAUDE.md tie-breaker):

- `ssm_angle_to_{z,target,absorber,vertical}` are **all exactly π/2** on this event, in
  every arm. That means `init_dir_10 = segment_cal_dir_3vector(ssm_sg, dir, 10, 0)`
  came back as the zero vector, so the 10 cm-direction quartet is degenerate here
  regardless of the reference vectors. Unrelated to this change; worth a look before
  anyone tunes those four features.
- The 7 run-to-run-unstable `pio`/`lol` branches above (order, not content) — the same
  class of pointer-order dependence as the `T_rec_charge` row order fixed in
  `dc9ba62b` (doc `pr/7`).

**(ii) The MIP normalization — highest-leverage single tune.** `43e3/units::cm`
appears at ~97 sites in 16 files (19× NuE, 13× Cosmic, 10× SSM, 10× VertexFinder, 9×
TrackShowerSep, 5× ShowerClustering, …), including as *default arguments* in
`clus/inc/WireCellClus/PRSegmentFunctions.h:27,104,117,119,120`. There is no named
constant. At 0.5 kV/cm the SBND MIP is ~54.7e3 e/cm (the doc-48 regenerated tables:
muon plateau 48879→54658), so every normalized-dQ/dx ratio reads ≈1.27 where uBooNE
read ≈1.0, silently shifting every ratio cut (0.75/0.95/1.2/1.3/1.4/1.5/1.6…).
Proposed tune: one config-fed `mip_dqdx`-style parameter (SBND already ships
`mip_dqdx=56000` for the STM tagger, doc 48) threaded through these sites, default
43e3 ⇒ uBooNE byte-identical.

**DONE 2026-07-30 — implemented as designed and ON in SBND production; see §2e(ii-a).**
(The census above is left as the record of what the audit found. Its site count and the
`PRSegmentFunctions.h` line numbers are now stale — the current citations are in
(ii-a).)

**(ii-a) DONE 2026-07-30 — the MIP scales are config-fed and SBND sets them.** Designed,
implemented and gated in doc `pr/8` (toolkit `3d71e111`); this section is the §2e
worklist's view of it. **This doc entry changes no code**, so no new gate is owed — the
byte-identity evidence below is `pr/8` §10's, quoted by label.

The audit's "one parameter" turned out to be **two**, because the literal carries two
unrelated roles:

| key | C++ default | role | SBND |
|---|---|---|---|
| `mip_dqdx` | `50000` e/cm | flat-template amplitude in the `do_track_comp` KS comparison; `segment_cal_4mom` scale; `segment_is_shower_trajectory` | **56000** (reuses the existing STM arg — one number, both taggers) |
| `mip_dqdx_median` | `43000` e/cm | the scale every median-dQ/dx ratio threshold (×1.2/1.3/1.4/1.75…) and BDT normalization is quoted against | **48000** |

Both live on `TaggerCheckNeutrino` → `PatternAlgorithms::m_mip_dqdx{,_median}`
(`NeutrinoPatternBase.h:68-69`) → member functions directly, free tagger helpers via
`ctx.self`, and the `PRSegmentFunctions` free functions via explicit arguments. Header
defaults (`PRSegmentFunctions.h:27,91,106,126,133,138-142`, `PRShower.h:189,192`) keep
the uBooNE numbers, so an omitted argument is legacy-identical. 16 files under
`clus/src/` now read a configured scale (116 median-role, 131 flat-role references).

**Why 48000 and not the ~54.7e3 of the census above.** They are different quantities.
54.7e3 e/cm is the doc-48 *muon-plateau* dQ/dx; `mip_dqdx_median` is the *reference
scale the ratio cuts are quoted against*, which in uBooNE sat at 43000 against a
flat-template 50000 — a ratio of 0.860. 48000 against SBND's 56000 is 0.857, i.e. the
round number nearest to carrying that internal ratio over (0.860 × 56000 = 48160). It is
a **placeholder pending an SBND median-MIP measurement** (owner, 2026-07-30). Nothing
here is a measurement of the SBND median.

Compiled-config proof (fresh, 2026-07-30, this doc entry) — the full command, verbatim.
`input`/`output_dir` are never opened by jsonnet, so any placeholder works:

```bash
export WIRECELL_PATH=$TK/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data
wcsonnet \
  --tla-str "input=x.tar.gz" --tla-code 'anode_indices=[0,1]' \
  --tla-str "output_dir=/home/xqian/tmp/z" \
  --tla-code run=18253 --tla-code subrun=1 --tla-code event=172230 \
  --tla-str reality=data --tla-code DL=4.0 --tla-code DT=8.8 \
  --tla-code lifetime=35 --tla-code driftSpeed=1.563 \
  --tla-code "pipeline_names=['tagger_check_stm','tagger_check_neutrino']" \
  --tla-str "trackfitting_config=$SB/sbnd_track_fitting.json" \
  --tla-str "dl_weights=" --tla-code 'beam_window_us=[0.2,2.2]' \
  $TK/cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  | grep -n 'mip_dqdx\|proton_dir_vote'
# 206:  "mip_dqdx" : 56000                      <- TaggerCheckSTM
# 238:  "mip_dqdx" : 56000                      <- TaggerCheckNeutrino
# 239:  "mip_dqdx_median" : 48000
# 244:  "proton_dir_vote" : true
```

(a 2-stage `pipeline_names`, not the production 13-stage job — enough to show both
components receive the keys.)

Byte-identity, from `pr/8` §10 — labels so any of it can be re-checked: `wcdoctest-clus`
565/565; 16/16 live production jobs compiled byte-identical, the nu-PR job differing by
exactly `+mip_dqdx +mip_dqdx_median +proton_dir_vote`, uBooNE `uboone-mabc.jsonnet`
unchanged; uBooNE off-gate sweep `mipvoteoff_ub` vs `dirweakon_ub` 35/35 zips
content-identical; SBND off-arm `nupr_evt172230_mipvote_off3` = `c5bfe4bf…`,
bit-identical to the `pr/6` reference; knob-on arm `65a64151…` reproduced by two
independent builds.

**Incomplete threading — recorded, not fixed (CLAUDE.md tie-breaker).** `pr/8` §9(a)
states the threading reached "every `determine_dir_track` call site". It did not:

- `PRSegmentFunctions.cxx:2396`, inside
  `segment_determine_shower_direction(..., double MIP_dQdx, ...)`, calls
  `segment_determine_dir_track(segment, 0, fits.size(), particle_data, recomb_model)`
  and `segment_is_shower_trajectory(segment)` with **neither the scale nor
  `pid_opts`**. Two consequences on that path, and the second is the larger one:
  `mip_dqdx_median` reverts to `43000`, **and `proton_dir_vote` is off even though SBND
  sets it true**. The fix is not a one-liner: `segment_is_shower_trajectory` wants the
  50k-role scale, which that signature does not carry. Gate condition for the branch is
  `total_length < 5 cm` in the not-shower-like arm; **how often it is taken has not been
  measured.**
- `PRSegmentFunctions.cxx:558,568` — the two `segment_cal_4mom` calls inside
  `break_segment()`, whose signature carries no MIP parameter, so they take the 50000
  default.

Residuals already documented in `pr/8` §9 and still true: the four SSM
`do_track_comp` calls (`NeutrinoTaggerSSM.cxx:64,65,96,98`) stay at the 50k default
(their dQ/dx vector uses a different unit convention — flagged for separate review); the
two inert ×1000 outlier clamps (`PRSegmentFunctions.cxx:1240,1271`, still literal
`43e3`, unreachable at either scale); SinglePhoton's Birks/field constants
`1.38`/`0.273`, which are §2e(i)'s third correctness item, not this one.

Usability note, same class as (i-a)'s: `mip_dqdx_median` is **not** a TLA of
`wct-pr-perevt.jsonnet` (only `mip_dqdx=56000` at `:103`; the median is defaulted inside
`pr()` at `clus.jsonnet:1294`), so isolating the median change in an A/B currently needs
a jsonnet edit rather than a `--tla-code`.

**(iii) Energy reconstruction (`NeutrinoEnergyReco.cxx`):**

| where | value | note |
|---|---|---|
| `:190,241,298` | `fudge_factor=0.95`, `recom_factor=0.7`; shower `0.5/0.8`; proton `0.35` | average recombination survival — field-dependent, uBooNE-tuned |
| `:163,171` | plane weights `{0.25,0.25,1.0}`, asymmetry switch `0.04` | reflects uBooNE induction-plane quality |
| `:175` | W-value `23.6 eV` duplicated (not read from the recomb model's `Wi`) | low risk, note only |
| `:14-35` | `cal_corr_factor()` is a **stub** returning 1.0 | no position/lifetime/SCE correction for either detector; pairs with the retained `0.85` factor in the SBND dE/dx tables ("degenerate with the missing electron-lifetime correction") |

**(iv) Detector-extent / absolute-charge literals:**

| where | value | uBooNE meaning → SBND effect |
|---|---|---|
| `NeutrinoTaggerCosmic.cxx:1175,1190` | `highest_y > 100/102/80 cm` → cosmic | "reaches the top": 16 cm below uBooNE's y=116.5 top; SBND top is y≈200, so 100 cm is **mid-detector** — cut near-meaningless as-is |
| `NeutrinoVertexFinder.cxx:875,2948` | vertex score penalty `(z−min_z)/(200 cm)` | upstream-z prior calibrated to uBooNE's 1037 cm (≈5 units end-to-end); on SBND's 500 cm it is ~2× weaker relative to the +0.25-per-track terms |
| `NeutrinoTaggerNuMu.cxx:191,296`, `NeutrinoVertexFinder.cxx:3340` | `0.8866+0.9533·(18cm/L)^0.4234` (×43e3) | empirical uBooNE muon dQ/dx-vs-length curve |
| `NeutrinoTaggerCosmic.cxx:536` | FV shrink `−1.5 cm` (6 faces) | absolute margin for vertex-outside-FV test |
| `SteinerGrapher.cxx:86-88,202-214,353` | `Q0=10000`, `charge_threshold=4000` e; distances 0.6/1.8/6.0 cm | prototype constants; gain/noise- and pitch-coupled |
| `NeutrinoEnergyReco.cxx:209,264,287` | 2D-hit match `0.6 cm` | pitch-coupled; SBND also 3 mm — likely transfers |

**(v) `sbnd_track_fitting.json` vs `uboone_track_fitting.json`:** same 44 keys; **6
already SBND-adapted** (`DL`, `DT`, `add_sigma_L`, `col_sigma_w_T`, `ind_sigma_u_T`,
`ind_sigma_v_T`), **38 verbatim uBooNE**. Detector-coupled among the 38:
`min_drift_time=50 µs` (uBooNE max drift ≈2.3 ms vs SBND ≈1.28 ms — re-derive what the
gate means), and the absolute-electron thresholds `default_dQ_dx=5000`,
`charge_cut=2000`, `add_charge_uncer=600`, `share_charge_err=8000`,
`default_charge_err=1000` (uBooNE noise/gain). `search_range=10` wires /
`time_tick_cut=20` are pitch/tick-coupled but SBND shares 3 mm / 0.5 µs. Reminder from
doc 66: this JSON is read at **runtime** — a byte-identical compiled jsonnet does NOT
prove the fit unchanged.

**(vi) Config-exposed knobs currently inheriting uBooNE defaults** (from
`cfg/pgrapher/common/clus.jsonnet` `tagger_check_neutrino(...)`): `dQdx_scale=0.1`,
`dQdx_offset=-1000` (SCN input scaling — inert while `dl_weights=''`, **live the day
SBND trains a net**), `dl_vtx_rerank/top_k/min_accept_score/score_scale` (uBooNE-tuned
re-rank, inert), `clus_geom_helper=''` (no SCE on SBND: `UbooneGeomHelper` doesn't even
exist in this tree — `kine_nu_{x,y,z}_corr` are raw positions, a *stated decision* per
§2d G4), and — new 2026-07-30 — `ssm_target_dir=[0.46,0.05,0.885]` /
`ssm_absorber_dir=[0.33,0.75,-0.59]` (§2e(i-a): reachable but **live and uncalibrated**,
unlike the inert SCN knobs — they feed 8 BDT features on every SSM-tagged event today).
`mip_dqdx` / `mip_dqdx_median` are deliberately *not* in this list: SBND **sets** them
(56000 / 48000, §2e(ii-a)) rather than inheriting the uBooNE defaults.

**(vii) Clean bills of health** (so nobody re-audits them): `ParticleDataSet.cxx` (all
curves from config; PDG masses only), `TrackFitting.cxx` geometry (pitch/tick/drift
all from `DetectorVolumes`/grouping), `FiducialUtils`/`sbnd_pr_fv` (real SBND bounds),
`improvecluster_2.cxx`, `SimpleClusGeomHelper.cxx` (FV from config),
`NeutrinoTaggerCosmic.cxx:1264-1287` (z_front derived, not hardcoded). The remaining
~1050 `units::cm`/angle/MeV literals across the taggers are **topology cuts**
(segment lengths, opening angles, shower-energy thresholds) encoding LAr shower
physics, not detector dimensions — expected to transfer; revisit only ones implicated
by scan failures. Hardcoded `dir_beam(0,0,1)`/`dir_drift(1,0,0)` literals are correct
for SBND (beam +z, drift ±x; consumers fold angles to |θ−90°|) — latent assumption,
noted.

**Tuning protocol:** every change from this worklist ships as a default-OFF knob
(uBooNE path byte-identical, Track A gate PASS), is tuned on the Track B/C samples,
and gets its own round in the doc-63 style. Priority order: (i) correctness items →
~~(ii) MIP normalization~~ **(done 2026-07-30, §2e(ii-a) — the knobs exist and SBND sets
them; what remains is a *measurement* to replace the 48000 placeholder, not plumbing)**
→ (iv) cosmic-y + vertex-z prior → (iii) energy reco → (v) fit-JSON thresholds; (vi)
waits for the SCN/BDT work (G1/G3).

Pre-existing oddity, noted not fixed (CLAUDE.md tie-breaker: report, don't fix):
`clus.jsonnet:398`-area `do_tracking()` delegates to `$.tagger_check_neutrino(...)` but
`tagger_check_neutrino` lives inside `clustering_methods()`, not at `$` — the path looks
dead/broken. Unused by both uBooNE and SBND jobs; do not rely on it.

**Answer to "did I miss anything": no silent misses at the stage level.** Every uBooNE
stage is accounted for: 5 running in production, 1 wired-but-not-enabled
(`tagger_check_neutrino` — the actual next step), 5 deliberately not applicable (§2c),
and 5 known deferred work items (§2d, all already flagged in
`sbnd-pattern-recognition.md` §6 — none newly discovered here, but G2/G4 are *blocking
for validation*, which is new emphasis: without an SBND tagger-variable dump (G2) the
tagger stages cannot be validated beyond "they ran"). **Below the stage level, §2e is
the new finding**: the ported code carries uBooNE constants — three outright
correctness items, a chain-wide 43e3 MIP normalization inconsistent with the SBND
recombination config, and a short list of detector-extent cuts — that must be tuned
(each behind a default-OFF knob) as part of this campaign.

## 3. Validation strategy

Three tracks, in dependency order. All arms: fresh work root per arm (M13), inputs
symlinked from existing productions (M11 — never regenerate imaging/Q-L), `setarch
x86_64 -R`, `dl_weights=''`, NJOBS ≤ 6, archives compared only via
`abtest/hash_archive.py` member hashes (M2).

### 3.1 Track A — port fidelity (uBooNE, existing gate)

Nothing new to build. The 35-event `ab_check.sh <label> gate3` sweep stays the
regression floor: **run it after every C++ change** made during this campaign, since
`clus/` is shared between detectors. For deeper toolkit-vs-prototype physics questions
the written method is `clus/docs/tagger/tagger_validation_plan.md` (Phase-1 sanity /
Phase-2 distributional KS + Pearson-r + working-point efficiency / Phase-3 Class A-B-C
outlier triage); it applies unchanged if uBooNE-side questions come back.

### 3.2 Track B — SBND data, neutrino-candidate subset of the 1000-event production

- **Event set**: from `scan-d59k/census.tsv` (998 events censused), every event whose
  in-beam labels include ≥1 `nu-candidate`: 479 pure + 24 mixed = **503 events**.
  (For comparison: 144+ STM-only, 153+ TGM, 10 LM, 187 no in-beam bundle.) The list is
  derived by rule, stated in the campaign doc, and frozen as
  `scan-prk/events.txt`-style artifacts before any scan.
- **Arm layout**: `work-nuprk-<tag>` with `evt<ID>`/`ql_evt<ID>` symlinked from
  `work-mcp1kall-d59k` (the doc-63 `stm_campaign/run_round.sh` pattern — only the PR
  tail re-executes, seconds/event); refuse to run if the root exists.
- **Pipeline**: production tail + `tagger_check_neutrino` appended (after
  `tagger_check_fc`, the order smoke-tested in §5.3; note uBooNE computes its own
  `match_isFC` inside the neutrino stage via `cluster_fc_check`, so SBND's prior FC
  verdict and the internal one should be compared once — checklist V8).
- **Cost expectation** (doc 20): `TaggerCheckNeutrino` ≈ 0 ms on events with no
  in-window candidate, 0.2–3.5 s when the full pattern build runs; ≤ 0.4 GB extra.
  ~503 events ≈ well under an hour at NJOBS=6.

### 3.3 Track C — the 48-event nueCC data candidate sample (highest scan value per event)

Same arm mechanics against `work-nuecc48-nuf` (§5): 45/48 events are `nu-candidate`
after the cosmic taggers, and these are *expected* neutrino interactions (Lynn's
selection), so vertex/track-shower/shower-clustering results are directly judgeable by
eye — scan Track C **first**, then the Track B data candidates.

### 3.4 MC with truth (later)

`19_PR_integration.md` §7 route A (existing LArSoft MC via `build_mcbase_stage.py` /
`run_mcbase.sh`) gives chain-integration MC today; route B (depo-split sim) + the
missing `recob::OpFlash → opflash_apa{N}.tar.gz` converter unlocks fit-vs-truth
(vertex resolution, energy scale). Named here so the checklist can reference truth
steps; not part of the first campaign.

## 4. Step-by-step validation checklist

**V0 — instrumentation first (the doc-40/41 lesson).** Before judging verdicts, make
the internals persistent and diffable:
- Bee: `track_fit` / `shower_track` / `vertices` layers + the `mc` particle-flow tree
  already come free with `tagger_check_neutrino` (verified §5.3) — the scan currency.
- Variables: **DONE (doc `pr/3`)** — `T_tagger`+`T_kine` now come from the
  `tracking_visitor`+`tagger_output` pipeline stages (G2 closed); the TSV
  extension of `nusel_extract.py` remains optional convenience.
  (Original plan text kept below for the record.)
- Variables: G2 fork (`SbndTaggerOutputVisitor`, fork-by-duplication) writing
  `T_tagger`+`T_kine`, or minimally a per-bundle TSV extension of `nusel_extract.py`
  with the main PR scalars (nu vertex x/y/z, n_vertices, n_segments, n_showers,
  numu/nue tagger primitive flags, kine energies). Recommendation: do the TSV extension
  immediately (cheap, diffable, feeds the existing viewer/report tooling) and the ROOT
  visitor fork when BDT work (G1) starts. Both default-OFF knobs (wct-knob pattern).
- PC dump: the PR graph state via a `save_pr_dump`-style knob mirroring `save_stm_fit`
  (doc 41) if offline diagnosis needs more than the TSV — decide after the first scan
  round, from actual failure modes.

Then, in order — each step names its observable, tool, and acceptance bar:

| # | check | how | pass when |
|---|---|---|---|
| V1 | **Off-gate** (harness changes nothing) | run the Track B arm with the *production* pipeline (no neutrino stage); `hash_archive.py` pctree/mabc members + byte-diff `nusel-evt*.tsv` vs d59k | 100% identical (mind the log-tearing phantom-TSV gotcha, doc 66 §12.5) |
| V2 | **Determinism** | 2–3 events × 4 runs with the neutrino stage ON, ASLR off AND on (`repeat_check.sh` pattern); compare mabc-pr.zip + pctree member hashes | 1 distinct hash per event per condition; if ASLR-on differs, record it and pin ASLR for all A/Bs (uBooNE PR is pointer-order sensitive; doc 60 found the SBND tail deterministic — must re-measure with the PR stage on) |
| V3 | **Robustness sweep** | full Track B (503) + Track C (48) arms; per-event rc + `.status` audit (doc-59 pattern) | rc=0 everywhere; every non-zero gets its own doc-60-style autopsy before proceeding |
| V4 | **Per-stage execution sanity** | grep the per-stage `TaggerCheckNeutrino timing:` + `selected main cluster` log lines; distributions of stage walls | every nu-candidate event selects a main and completes all stages; no stage silently skipped |
| V5 | **Main-vertex correctness (the core physics scan)** | Bee sets (chunked, index maps beside zips) from Track C then Track B `mabc-pr.zip` `vertices`+`track_fit` layers; hand-scan "vertex on the interaction point? y/n" under a FRESH tag; sub-agent batches attributable, seeded controls (doc-61 method) | owner-adjudicated truth table (doc-62 pattern) with vertex-correct fraction; failures classified (wrong cluster / right cluster wrong end / off-track) — this table is the campaign's baseline metric |
| V6 | **Track/shower separation + shower clustering** | same scan, `shower_track` layer + `mc` PF tree on the 48 nueCC events: is the primary EM shower found, attached to the vertex, one shower not five | qualitative pass + failure list; pi0 spot-checks where two showers pair |
| V7 | **Track fitting / dQ/dx** | reuse STM-fit infra: `save_stm_fit` PCs + viewer dQ/dx panel + `stm_ref_dqdx.json` reference on fitted muons in the PR sample; MIP scale consistency with `mip_dqdx=56000` | fitted-muon dQ/dx plateau consistent with the STM campaign's; no systematic offset vs doc-48 tables |
| V8 | **Tagger-variable Phase-1 sanity** (needs V0 variables) | `tagger_validation_plan.md` §4.1 adapted: no NaN/Inf (excluding −999 sentinels), flags ∈ {0,1}, fill-gate consistency, per-tagger fire fractions plausible (nothing 0% or 100%); compare internal `match_isFC` vs SBND `tagger_check_fc` verdict; check the dQ/dx-ratio features for the ≈1.27 offset predicted by §2e(ii) | all sanity rows pass; fire-fraction table published; disagreement between the two FC computations understood; §2e(ii) offset confirmed or refuted with data before any MIP retune ships |
| V9 | **Kine tree sanity** | energy reco outputs on the 45 nueCC candidates: ranges physical, no negative/absurd energies; muon/proton/shower KE spot-checks against hand-measured track lengths | ranges physical; outliers diagnosed |
| V10 | **Perf/RSS** | `timecmd.py` wall+RSS on both arms vs the V1 baseline; per-stage timing rollup (doc-20 format) | budget stated and met (expected: +0.2–3.5 s and ≤ +0.4 GB on candidate events); no pathological event >30 s unexplained |

**Acceptance bar for enabling `tagger_check_neutrino` in production `pipeline_names`**
(doc-63 style, checkable): V1–V4 green; V5 truth table exists with the vertex-correct
fraction published (no fixed threshold up front — the first scan *sets* the baseline);
V8 sanity all-green; V10 budget met; plus Track A (`ab_check.sh` vs `gate3`) PASS and,
if any C++ changed, the standard knob-off byte-identical gates on every affected
detector. Improvement rounds after the baseline follow doc-63 rules: one default-OFF
knob per round, "fixes ≥1, regresses 0", full-population flip census.

## 5. The nueCC 48-event sample (prepared)

### 5.1 Provenance chain (all pre-existing; recorded here so it is findable)

1. Lynn's 48 nueCC candidate RSEs: `../../samples/lynn-nuecc-rse.csv` (48 events, runs
   18253–18409; doc `../../samples/docs/1-…`).
2. RSE → reco1 art files via samweb (`sbnd.event_number_list %_<evt>_%` trick):
   `find_reco1_files.sh`; 48/48 found, 47 unique files
   (`lynn-nuecc-reco1-files.lst`, `.map.txt`; doc `../../samples/docs/2-…`).
3. `FilterEventID` (ported verbatim into larwirecell) + `filter-nuecc-rse.fcl` →
   one 48-event filtered reco1 file; then `run_frameshift.fcl` (Gen2-data requirement,
   doc `../../samples/docs/gen2-data-frameshift.md`) →
   `input_files_reco1/data_filtered_decoded_reco1-…_eventidfiltered_frameshift.root`.
4. Toolkit extraction (`run_reco1_dump.sh` route, doc 21) →
   `input_files_reco1/extracted-2025fall-48evt-fsprod/` (`frames-dnn.tar.bz2`,
   `opflash_apa{0,1}.tar.gz`).
5. Imaging + Q/L from doc 21 (2026-07-21 campaign, shared `work/`); cosmic-tagger
   tail run 2026-07-29 by doc `1_beam-window-cosmic-vs-nu-division.md`:
   - `work-nuecc48-base/` — partial earlier arm (48 evt, 24 ql_evt, 20 nusel_evt).
   - **`work-nuecc48-nuf/`** — complete, 48/48 rc=0: 48 evt (symlinks into the shared
     `work/` imaging) + 48 `ql_evt` (with `pctree-evt<ID>.tar.gz`) + 48 `nusel_evt` +
     merged `nusel-table.tsv` / `nusel-events.tsv`. Production NUF flag set
     (unmerge_assoc confirmed in-log: `ClusteringUnmergeBundle:prassoc`).

### 5.2 What the cosmic taggers say about the 48

Per `work-nuecc48-nuf/nusel-events.tsv`: **45 nu-candidate, 3 cosmic-tagged**.
In-beam bundles (`nusel-table.tsv`): 45 nu-candidate, 6 TGM — i.e. 11.8 % cosmic /
88.2 % nu-candidate over 51 in-beam bundles, **STM = 0, LM = 0** (doc `1_`'s census,
reproduced independently here). The 3 cosmic-tagged
events (and the TGM'd bundles inside candidate events) are themselves scan-worthy:
each is either a tagger false-positive on a real neutrino or a Lynn-list impurity —
adjudicate during V5.

### 5.3 Smoke test — the neutrino PR stage runs on this sample (DONE, 2026-07-29)

Fresh root `work-nuecc48-prsmoke/` (M13: `work-nuecc48-nuf` untouched, inputs read
in place). Binary: installed `local/lib` stack of 2026-07-27 (= HEAD `c0501d7e`,
doc-66 ship; `local/bin/wire-cell` explicitly, avoiding the build/-tree M1 trap).

```bash
cd $SB && SM=$PWD/work-nuecc48-prsmoke
WC=/nfs/data/1/xqian/toolkit-dev/local/bin/wire-cell
export WIRECELL_PATH=$TK/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
for EVT in 172230 444187; do OUT=$SM/nupr_evt$EVT; mkdir -p $OUT
  setarch x86_64 -R $WC -l stderr -l "$OUT/wct_nupr_evt$EVT.log:debug" -L debug \
    --tla-str "input=$PWD/work-nuecc48-nuf/ql_evt$EVT/pctree-evt$EVT.tar.gz" \
    --tla-code 'anode_indices=[0,1]' --tla-str "output_dir=$OUT" \
    --tla-code run=18253 --tla-code subrun=1 --tla-code event=$EVT \
    --tla-str reality=data \
    --tla-code DL=4.0 --tla-code DT=8.8 --tla-code lifetime=35 --tla-code driftSpeed=1.563 \
    --tla-code "pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','tagger_check_neutrino']" \
    --tla-str "trackfitting_config=$PWD/sbnd_track_fitting.json" \
    --tla-str "save_tensors=$OUT/pctree-pr-evt$EVT.tar.gz" \
    --tla-str "dl_weights=" --tla-code 'beam_window_us=[0.2,2.2]' \
    -c $PWD/wct-pr-perevt.jsonnet > $OUT/stdout.log 2>&1; echo "evt$EVT rc=$?"; done
```

(Everything not passed takes the module defaults, which since doc 64 ARE the
production operating point — unlike `run_pr_evt.sh -nu`, which pins pre-adoption
TGM/FC values for its historical A/B demos and reads pctrees only from the shared
`work/`. A campaign runner should extend `run_nusel_evt.sh` with a `-nu` flag instead.)

Results — both rc=0:

| evt | selected main | TaggerCheckNeutrino wall |
|---|---|---|
| 172230 | `selected main cluster 5 (t0 1.662 us, L 123.7 cm, 31 associated)` | 3255 ms (main-cluster initial PR 2987 ms) |
| 444187 | `selected main cluster 6 (t0 1.096 us, L 210.6 cm, 1 associated)` | 193 ms |

All NeutrinoID stages logged (`preload_clusters` … `shower_clustering_with_nv`), and
`mabc-pr.zip` carries the full PR layer set: `clustering-global`,
**`shower_track-global`, `track_fit-global`, `vertices-global`**, channel-deadarea,
and the **`mc`** particle-flow JSON. `pctree-pr-evt<ID>.tar.gz` re-saved for the
round-trip/off-gate tooling. This is the plumbing proof for Tracks B/C.

### 5.4 Work-root registration

`work-nuecc48-base`, `work-nuecc48-nuf`, and `work-nuecc48-prsmoke` are now registered
in `../work-tags.md` (they predate this doc but were unregistered — the doc-59
work-tags lesson). `work-nuecc48-nuf/evt<ID>` are symlinks into the shared `work/`
imaging: do not archive `work/` pieces without `relink_tags.py`.

## 6. Open items (unchanged from `sbnd-pattern-recognition.md` §6, plus this doc's)

- G1–G5 (§2d): BDT retraining, Sbnd output visitors, SCN retraining, dQ/dx + tagger
  calibration.
- **An SBND BNB-target direction** for `ssm_target_dir`, and a decision on what (if
  anything) `ssm_absorber_dir` should mean at SBND. The knobs exist as of 2026-07-30
  (§2e(i-a)); until they are set, the 8 `ssm_*_angle_{target,absorber}` features carry
  MicroBooNE geometry. Note the parity caveat about non-unit vectors before supplying
  a normalized one.
- **An SBND median-MIP dQ/dx measurement** to replace the `mip_dqdx_median=48000`
  placeholder, which today is only the round number nearest to carrying the uBooNE 43/50
  ratio over onto 56000 (48000/56000 = 0.857 vs uBooNE's 0.860 — §2e(ii-a)). The knob is
  live in production, so this number is currently shifting
  every ratio cut in the PR chain on a value nobody measured.
- **`PRSegmentFunctions.cxx:2396`** — the one `determine_dir_track` call site the pr/8
  threading missed; it drops both `mip_dqdx_median` (→ 43000) and `proton_dir_vote`
  (→ off) on the `total_length < 5 cm` non-shower-like path. Owner call whether to fix,
  and it needs the 50k-role scale plumbed into `segment_determine_shower_direction`
  first (§2e(ii-a)).
- The §2e tuning worklist, in its stated priority order: correctness items (SSM beam
  vectors — now half-closed, see above; nue apa/face, SinglePhoton inline box model) →
  ~~chain-wide MIP normalization knob~~ (done, §2e(ii-a); the open piece is the
  measurement above) → cosmic-y / vertex-z-prior rescaling → energy-reco factors +
  `cal_corr_factor` stub → fit-JSON absolute-charge thresholds.
- Track B/C campaign execution (§3) and the V0 instrumentation knobs (§4).
- Beam-window calibration on a larger sample; multi-bundle "longest wins" rule.
- MC-with-truth route B + the `recob::OpFlash` converter (§3.4).
- The `do_tracking()` `$`-scoping oddity (§2d note) — dead code, flag to owner.

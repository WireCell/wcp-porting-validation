# 2 — Neutrino PR chain on SBND: uBooNE gap analysis + validation plan + nueCC sample

**Status: PLAN (with executed smoke test), plus three worklist items since executed**
(§2e(i-a), §2e(i-b), §2e(ii-a)). The original plan changed no production default; two of
those three items since have — the MIP scales are ON in SBND production (§2e(ii-a)) and
the nue apa/face fix ships unconditionally (§2e(i-b)). The one new artifact besides docs
is the 2-event smoke root
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

**Update 2026-07-30 (third)** — the second §2e(i) correctness item is **closed**:
`nue_tagger` now receives the apa/face of the vertex's own drift volume instead of a
hard-coded `(0,0)` (toolkit `7902615c`). Per the owner this is a bug fix and ships
**without a knob**; uBooNE is unaffected and **§2e(i-b)** shows it two ways rather than
arguing it. That section also records the first measured effect (evt 444187, APA 1:
`mip_quality_flag`/`mip_quality_overlap` flip, `nue_score` unchanged) and a caveat about
`ab_check.sh`'s second gate, which an A/A control shows is non-discriminating on the
uBooNE manifest. §2e(i) is now fully closed except for the SBND `ssm_target_dir` *value*
and the SinglePhoton inline box model.

**Update 2026-07-30 (fifth, the pr/10 energy round)** — the SinglePhoton inline box
model is knob-closed (`sp_dedx_use_recomb_model` + `sp_mean_dedx_cut`, §2e(i-c)); the
§2e(iv) muon dQ/dx-vs-length curve — actually at **nine** sites, not the three listed
— is config-fed AND carries an SBND table-derived fit in production (§2e(iv-b)); the
three `kine_*` recombination survival factors now carry SBND transfer values
(§2e(iii-b)); and the doc-55 free-power recombination model exists in the toolkit as
`PowerBoxRecombination` behind `use_power_recomb` (OFF pending review). All in
`sbnd_xin/docs/pr/10`; toolkit `405a0f9a` + `21c31439` + `db625c81`.

**Update 2026-07-30 (fourth)** — the two **§2e(iv) detector-extent** rows are **closed**
and are the first worklist item whose SBND value is a real translation rather than a
placeholder: `cosmic_y_top_main/_top_strict/_top_loose/_small_piece` (uBooNE
100/102/80/50 → SBND 183/185/163/133 cm, the same offsets below a top face that moved
from y = +117 to +200) and `vertex_z_prior_scale` (200 → 100 cm, the upstream-z prior
scaled by detector length 1037 → 501 cm). **§2e(iv-a)** has the geometry table, why the
two rows translate by *different* rules, the gates (uBooNE byte-identical), and the
measured SBND effect: 46/48 nueCC events unchanged, 2 changed — one main-cluster switch
(evt 235435, needs a hand scan) and one 22.8 cm main-vertex move (evt 269774). It also
records that `nue_score = ±4.301` is the BDT **saturation clamp**, not a resolved score.

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
| `TaggerCheckNeutrino.cxx:493` | `nue_tagger(..., apa=0, face=0, ...)` | Single-drift-volume assumption: any vertex in SBND APA 1 gets APA 0's mirrored wire geometry for `flag_prolong_u/v/w` / `flag_parallel`. `NeutrinoTaggerSinglePhoton.cxx:2186-2196` already derives apa/face from `dv->contained_by(vtx_pt)` — the nue path needs the same fix. **CLOSED 2026-07-30, toolkit `7902615c` — see §2e(i-b).** (The `:493` line number is from the 2026-07-29 sweep; the call site is `:561` at `34f0abd8`.) |
| `NeutrinoTaggerSinglePhoton.cxx:1499-1505` | inline inverse Modified-Box with `0.273` kV/cm, `1.38` g/cm³, `23.6e-6`, `43e3` | Bypasses the configured `sbnd_box_recomb` (0.5 kV/cm) entirely for its `median_dedx`/`mean_dedx`. **CLOSED 2026-07-30: `sp_dedx_use_recomb_model` routes it through the configured model, `sp_mean_dedx_cut` exposes the coupled cut — §2e(i-c), pr/10 §5. ON for SBND (with `sp_mean_dedx_cut=2.23`, toolkit `6d0396a2`); default OFF = uBooNE byte-identical.** |

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

**(i-b) DONE 2026-07-30 — `nue_tagger` gets the vertex's own apa/face.** Toolkit
`7902615c`. Owner call: **this is a bug, so it ships unconditionally, with no knob** —
the usual default-OFF rule does not apply to a wrong value. uBooNE is nevertheless
unaffected, and that is shown below rather than argued.

Repro:

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild && ./build/clus/wcdoctest-clus

# uBooNE arm: 35-event manifest, new binary
qlport/scripts/sweep_5384.sh nueapa_ub 6
qlport/scripts/ab_check.sh   nueapa_ub mipvoteoff_ub
qlport/scripts/sweep_5384.sh nueapa_ub_aa 6            # A/A control, same binary
qlport/scripts/ab_check.sh   nueapa_ub_aa nueapa_ub
# what apa/face the fix actually derived, per event:
grep -h "nue_tagger volume" qlport/scripts/sweep/nueapa_ub/*/wct_*.log \
  | sed -E 's/.*volume: //; s/ \.\..*//' | sort | uniq -c

# SBND arms on evt 18253/1/444187 (a vertex that lands in APA 1)
cd sbnd_xin/work-nuecc48-prsmoke2
PROUT=$PWD/nupr_evt444187_nueapa      ./run_pr3_evt.sh 444187        # post-change
PROUT=$PWD/nupr_evt444187_nueapa_aa   ./run_pr3_evt.sh 444187        # A/A control
# pre-change arm: ./wcb build (NO install) at 34f0abd8 into /home/xqian/tmp/nueapa_oldlib/
LD_LIBRARY_PATH=/home/xqian/tmp/nueapa_oldlib:$LD_LIBRARY_PATH \
  PROUT=$PWD/nupr_evt444187_nueapa_base ./run_pr3_evt.sh 444187
python3 ../tagger_tree_ab.py nupr_evt444187_nueapa_base/tracking-pr.root \
                             nupr_evt444187_nueapa/tracking-pr.root
```

The fix, at the `nue_tagger` call site (`TaggerCheckNeutrino.cxx:561` as of
`34f0abd8`, `:493` when the §2e(i) sweep was taken):

```cpp
int nue_apa = 0, nue_face = 0;
if (m_dv) {
    const Point nue_vtx_pt = final_main_vertex->fit().valid()
                             ? final_main_vertex->fit().point
                             : final_main_vertex->wcpt().point;
    const auto nue_wpid = m_dv->contained_by(nue_vtx_pt);
    if (nue_wpid.apa() >= 0) { nue_apa = nue_wpid.apa(); nue_face = nue_wpid.face(); }
}
```

Three choices worth recording:

1. **Derived at the call site, not inside `nue_tagger`.** The `int apa, int face`
   parameters exist so the caller decides; deriving inside would make them dead.
   `singlephoton_tagger` derives internally only because it takes no such parameters.
2. **Guarded on `apa() >= 0`, not on `wpid.valid()`.** `DetectorVolumes::contained_by`
   returns `WirePlaneId(kUnknownLayer, -1, -1)` for an uncontained point, whose
   `apa()` is −1 but whose `face()` is **+1** (the packed layer bits make it so) — so
   apa and face must be taken together or not at all. `valid()` additionally demands a
   well-defined *layer*, which the wpid coming back from `m_faces` is not guaranteed to
   carry; using it risks a guard that never fires. `apa() >= 0` is the idiom already in
   this file at `NeutrinoTaggerNuE.cxx:2764`.
3. **Falling back to (0,0) when uncontained**, i.e. to the legacy value. Not obviously
   right physically, but `Grouping::wire_angles()` is `map.at(apa).at(face)` — an
   unguarded −1 throws.

The two consuming sites are unchanged: `NeutrinoTaggerNuE.cxx:2725-2726`
(`wire_angles` + `get_drift_dir()` → `flag_prolong_u/v/w`, `flag_parallel` inside
`gap_identification`) and `:1638` (`segment_get_closest_2d_distances` inside
`mip_quality`). `contained_by`'s apa numbering is the same one those point-cloud
queries key on — `:2766` already feeds `q_apa` straight from `contained_by` into
`grouping->get_closest_points(...)`, as do `NeutrinoVertexFinder.cxx:125,190,1847` and
`NeutrinoStructureExaminer.cxx:569,2551`.

**uBooNE off-arm.** Two independent lines of evidence:

- **The arguments are literally unchanged.** The added debug line reports
  `apa=0 face=0` on **35/35** events of the qlport manifest — exactly the values the
  hard-coded literals supplied. Nothing downstream can differ.
- **Gate:** `sweep/nueapa_ub` vs `sweep/mipvoteoff_ub` → **ZIPS 35/35
  content-identical**.

`ab_check.sh`'s *second* gate (tagger-compare verbose logs) reports `identical=3
diff=32`. **That gate does not discriminate on this manifest and did not before this
change.** An A/A control with the identical binary — `sweep/nueapa_ub_aa` vs
`sweep/nueapa_ub` — reports `identical=2 diff=33`, i.e. *more* differing events than
the A/B, and 1401 differing log lines against the A/B's 1363. Spot-checking ev 6505,
the A/A reproduces the very same diffs the A/B shows
(`numu_cc_1_particle_type` 2001↔2201, `shw_sp_lol_2_v_length` 79.3804↔80.8994,
permuted `pio_2_v_*` vectors). Every historical `ab_check_*.log` in `sweep/` shows the
same 32–35 DIFF count for every label pair back to `d54opt2_ub`, so this is long-standing
run-to-run instability in the uBooNE PR stage, not a regression. **The operative gate on
this manifest is the ZIPS line.**

**SBND on-arm — the fix is exercised and it moves output.** Evt 18253/1/444187 derives
`apa=1 face=0`; evt 172230 (this doc's usual event) derives `apa=0 face=0` and is
therefore *unaffected*, which is why the effect had to be measured elsewhere. Against a
cross-binary pre-change arm (`34f0abd8` built with `./wcb build`, no install, loaded via
`LD_LIBRARY_PATH`; the base arm emits no `nue_tagger volume` line, which proves the swap
took), of 1216 `T_tagger` branches:

| comparison | moved | order-only (same multiset) | value-changed |
|---|---|---|---|
| A/A, same binary | 13 | 12 | 1 — `shw_sp_lol_1_v_angle` |
| A/B, apa (0,0) → (1,0) | 9 | 6 | 3 — `mip_quality_flag`, `mip_quality_overlap`, `shw_sp_lol_1_v_angle` |

`shw_sp_lol_1_v_angle` moves in the A/A too, so it is noise. The signal is the two that
do **not** appear in the A/A: `mip_quality_flag` **0 → 1** and `mip_quality_overlap`
**1 → 0** — precisely the `mip_quality` helper that consumes `ctx.apa`/`ctx.face` at
`:1638`. Both are NuE BDT input features. `nue_score` is nevertheless **unchanged**
(4.300936 in all three arms), as is `numu_score` (0.35807034): on this event the flip
does not propagate to the score. So the correct claim is *"the fix changes NuE feature
values on APA-1 vertices"*, **not** *"it changes the selection"* — no event is yet known
where the score moves.

**How often it matters.** Over the nueCC48 sample, 38 events ran to `rc=0` and 37 of
those reached `nue_tagger` (evt 116962 found no main vertex): **22 derive apa=0, 15
derive apa=1** — so roughly 40% of this sample was being evaluated against the wrong
drift volume. The batch was cut short at 37/48 by an unrelated concurrent rebuild of
`libWireCellClus.so` mid-run (the remaining jobs died with `failed to load plugin`);
the 11 missing events were **not** re-run, because by then the shared tree carried
another session's uncommitted work and a rebuild would no longer have isolated this
change. The 15/37 figure is a sample count, not a calibrated rate.

`./build/clus/wcdoctest-clus`: 49/49 cases, 565/565 assertions. Freshness proof:
`local/lib/libWireCellClus.so` 11:16:04 vs source 11:15:39.

**Which binary those numbers describe.** All of them — doctest, both uBooNE sweeps
(11:11–11:13), and all three SBND arms (11:10–11:16:47) — come from a library built from
`34f0abd8` **plus this hunk and nothing else**. At 11:18:29 a concurrent session in the
same tree edited `TaggerCheckNeutrino.cxx` (an unrelated `endpoint_trim_retry` knob) and
rebuilt at 11:19:55, so the *currently installed* `libWireCellClus.so` is no longer that
binary and re-running `wcdoctest-clus` today does not re-test this change. `7902615c`
was committed by staging exactly the 27+/1− hunk that was built and gated at 11:16:04
(`git apply --cached` of the pre-contamination diff), leaving the other session's work in
the worktree; it therefore contains only what the evidence above covers, but it was never
re-compiled in isolation *after* being staged.

Recorded, not fixed (CLAUDE.md tie-breaker):

- **`singlephoton_tagger`'s own derivation has no `apa >= 0` guard**
  (`NeutrinoTaggerSinglePhoton.cxx:2186-2196`). It does not call `wire_angles`, so there
  is no `.at(-1)` throw today, but an uncontained main vertex hands it `apa=-1, face=1`,
  which then flows into its point-cloud queries. Same one-line guard would fix it; out of
  scope for this change.
- `cosmic_tagger`, `numu_tagger` and `ssm_tagger` were checked: none takes apa/face at
  all, so there is nothing analogous to fix in them.
- `qlport/scripts/ab_check.sh` gate 2 has been non-discriminating for many labels
  (above). It should either be scoped to the stable branches or retired in favour of the
  ZIPS gate plus a per-branch `tagger_tree_ab.py` diff; leaving it as-is invites a future
  reader to treat a PASS-shaped FAIL as meaningful.

**(i-c) KNOB-CLOSED 2026-07-30 — the SinglePhoton inline inverse Box routes through
the configured model** (pr/10 §5, toolkit `21c31439`). `sp_dedx_use_recomb_model`
(default **false** = the inline float formula, byte-identical; gate: qlport ZIPS 35/35
`energyoff_ub` vs `geomoff_ub`) sends `shw_sp_vec_{median,mean}_dedx` through
`m_recomb_model->dE()`; the coupled hard cut is `sp_mean_dedx_cut` (default 2.3,
compared as float ⇒ bit-identical). Threshold transfer: 2.3 on the inline
(A=1.0/B=0.255 @ 0.273) scale = 58 768 e/cm = **2.23 MeV/cm** physical
(`dqdx_rr_sample/derive_kine_recom_factors.py`). **ON for SBND since 2026-07-30
(owner approval after the pr/10 §7 review; toolkit `6d0396a2`), together with
`use_power_recomb` and `sp_mean_dedx_cut=2.23`.** uBooNE keeps the inline path.

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
| `:190,241,298` | `fudge_factor=0.95`, `recom_factor=0.7`; shower `0.5/0.8`; proton `0.35` | average recombination survival — field-dependent, uBooNE-tuned. **Half-closed 2026-07-30: config-fed as `kine_fudge_factor` / `kine_recom_factor` / `kine_shower_*` / `kine_proton_recom_factor`, defaults unchanged — §2e(iii-a). Later same day: the three RECOM factors carry SBND production values 0.87/0.58/0.51 (table-integrated ratio transfer, §2e(iii-b) → pr/10 §6); the fudge factors deliberately stay uBooNE.** |
| `:163,171` | plane weights `{0.25,0.25,1.0}`, asymmetry switch `0.04` | reflects uBooNE induction-plane quality. **Half-closed 2026-07-30: `kine_plane_weights` / `kine_plane_asym_switch` — §2e(iii-a).** |
| `:175` | W-value `23.6 eV` duplicated (not read from the recomb model's `Wi`) | low risk, note only. **Config-fed 2026-07-30 as `kine_w_value`; still duplicated rather than read from `Wi` — §2e(iii-a).** |
| `:14-35` | `cal_corr_factor()` is a **stub** returning 1.0 | no position/lifetime/SCE correction for either detector; pairs with the retained `0.85` factor in the SBND dE/dx tables ("degenerate with the missing electron-lifetime correction") |

**(iii-a) DONE 2026-07-30 — the charge→energy calibration constants are config-fed**
(owner request; **no SBND value exists for any of them**, so this moves the numbers out
of the source, it does **not** calibrate them). Toolkit commit: see below.

Repro:

```bash
# compiled-config proof (knob off is byte-identical, knob on emits the keys)
cp -r cfg /home/xqian/tmp/cfg-new && cp -r cfg /home/xqian/tmp/cfg-base
for f in pgrapher/common/clus.jsonnet pgrapher/experiment/sbnd/clus.jsonnet \
         pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet; do
  git show HEAD~1:cfg/$f > /home/xqian/tmp/cfg-base/$f; done
# same TLAs both sides (run/subrun/event/pipeline as in run_pr_evt.sh -nu), then
cmp base.json new_off.json                      # -> identical, 250897 bytes
wcsonnet ... --tla-code 'kine_recom_factor=0.6' | grep kine_   # -> keys present

# knob-ON effect, uBooNE evt 5384/132/6642 (all four particles are showers):
#   scratch copies of qlport/uboone-mabc.jsonnet, one with
#   cm.tagger_check_neutrino(kine_shower_recom_factor=0.25, ...)
python3 -c "import uproot;t=uproot.open('track_com_5384_6642.root')['T_kine'];\
print(t['kine_reco_Enu'].array(library='np'))"
```

What moved (all in `TaggerCheckNeutrino` → `PR::KineChargeOptions`, C++ defaults = the
uBooNE literals they replaced, so an absent key is byte-identical):

| key | default | role |
|---|---|---|
| `kine_recom_factor` / `kine_fudge_factor` | `0.7` / `0.95` | track-like object |
| `kine_shower_recom_factor` / `kine_shower_fudge_factor` | `0.5` / `0.8` | `flag_shower` object |
| `kine_proton_recom_factor` | `0.35` | `abs(pdg)==2212` (the fudge factor stays at the track one — prototype behaviour) |
| `kine_plane_weights` | `[0.25,0.25,1.0]` | `[U,V,W]` charge-average weights |
| `kine_plane_asym_switch` | `0.04` | (median,max) asymmetry above which the largest plane is dropped |
| `kine_w_value` | `23.6` | argon W-value in eV |

Threaded to a TLA of `wct-pr-perevt.jsonnet` (i.e. `--tla-code`-reachable, unlike
`mip_dqdx_median` — the usability note above), through `pr()` and `clus_pr()` in
`cfg/pgrapher/experiment/sbnd/clus.jsonnet`. **No detector sets any of them.** Two
guards were added with the knobs (unreachable at the defaults): an all-zero
`kine_plane_weights` is rejected in `configure()` with a WARN, and both plane-average
denominators are zero-guarded in `kine_charge_from_maps`.

Gates:

- **uBooNE `qlport` ZIPS: 35/35 content-identical**, `sweep/kineoff_ub` vs
  `sweep/f3coff_ub2` (`ab_check.sh kineoff_ub f3coff_ub2`). Gate 2 (tagger-compare
  logs) reported 33/35 DIFF, which is its known non-discriminating A/A behaviour
  (same 33/35 seen on an A/A pair) — only ZIPS is a gate here.
- `wcdoctest-clus`: 49 cases / 565 assertions pass.
- Freshness: `local/lib/libWireCellClus.so` 15:21 > last source edit 15:20.
- Knob-ON, uBooNE evt 5384/132/6642 (4 shower particles, all `kine_energy_info==2`
  = charge-derived): `kine_shower_recom_factor` 0.5 → 0.25 doubles every particle
  energy and `kine_reco_Enu` **858.04 → 1716.07 MeV** (852.06 → 1704.13 for the
  leading shower). The same event is *insensitive* to `kine_recom_factor` (0.7 →
  0.35 changes nothing) — correct, since the track branch is unused when every
  object is shower-flagged.

Caveats worth knowing before calibrating:

- `kine_charge` is **not persisted by the SBND standalone PR job**: it lives in the
  `T_kine` tree of the uBooNE `track_com_*.root` and, on SBND, only reaches the
  (uncalibrated, gap G1) BDT features. Any SBND-side tuning of these knobs needs a
  tagger-tree dump first.
- `NeutrinoTaggerSinglePhoton.cxx:1499-1505` keeps its own inline `23.6e-6` (with
  `1.38`/`0.273`); it is §2e(i)'s third correctness item, not this one, and was left
  untouched.
- The `0.6 cm` 2D-hit match distance in the same function is row (iv), not (iii),
  and stays a literal.

**(iii-b) DONE 2026-07-30 — the three recombination factors carry SBND values**
(pr/10 §6, toolkit `db625c81`): `kine_recom_factor` 0.7 → **0.87**,
`kine_shower_recom_factor` 0.5 → **0.58**, `kine_proton_recom_factor` 0.35 → **0.51**
— the uBooNE empiricals scaled by the table-integrated survival ratio
R_eff(SBND free-power fit, C excluded) / R_eff(official uBooNE Box @ 0.273), per-class
dE/dx profiles from `energy_loss/pion_travel/stopping.root`
(`dqdx_rr_sample/derive_kine_recom_factors.py`). The fudge factors deliberately stay
uBooNE (they absorb the gain/lifetime normalization that the fit's C carries).
Effect: `kine_reco_Enu` −12…−14 % on nuecc48 172230/235435/444187 (T_kine of the
pr/3-style `tracking-pr.root` — which **corrects the (iii-a) caveat**: kine IS
persisted there; only `mabc-pr.zip`/pctree are insensitive). Plane weights, asymmetry
switch and W-value still have no SBND value.

**(iv) Detector-extent / absolute-charge literals:**

| where | value | uBooNE meaning → SBND effect |
|---|---|---|
| `NeutrinoTaggerCosmic.cxx:1073,1175,1190,1191` | `pca.center.y > 50`, `highest_y > 100/102/80 cm` → cosmic | "reaches the top": 67/17/15/37 cm below uBooNE's top face, which is **y = +117 cm**, not the 116.5 this row claimed until 2026-07-30 (`prototype_base/pid/apps/wire-cell-prod-nue.cxx:417` passes the FV as `3, 117, -116, 0, 1037, 0, 256`). SBND's top is y ≈ +200, so 100 cm is **mid-detector** — cut near-meaningless as-is. **CLOSED 2026-07-30: four config knobs `cosmic_y_top_main/_top_strict/_top_loose/_small_piece`, SBND 183/185/163/133 cm — see §2e(iv-a).** |
| `NeutrinoVertexFinder.cxx:875,3001` | vertex score penalty `(z−min_z)/(200 cm)` | upstream-z prior calibrated to uBooNE's 1037 cm (≈5.2 units end-to-end); on SBND's 501 cm it is ~2× weaker relative to the +0.25-per-track terms. (`:2948` in the pre-2026-07-30 sweep; the second site is `compare_main_vertices_global`, the first is `compare_main_vertices` — **both** are exercised on SBND.) **CLOSED 2026-07-30: config knob `vertex_z_prior_scale`, SBND 100 cm — see §2e(iv-a).** |
| `NeutrinoTaggerNuMu.cxx:198,296`, `NeutrinoVertexFinder.cxx:3394`, `NeutrinoTaggerNuE.cxx:402,759,1721,3448`, `NeutrinoTaggerSSM.cxx:1034`, `NeutrinoTaggerCosmic.cxx:558` | `0.8866+0.9533·(18cm/L)^0.4234` (×`mip_dqdx_median`) | empirical uBooNE muon dQ/dx-vs-length curve — **NINE sites, not the three this row listed until 2026-07-30. CLOSED: one knob `muon_dqdx_curve`, SBND production carries the table-derived fit `[0.8826, 1.0587, 18, 0.4745]` — §2e(iv-b) → pr/10 §4.** |
| `NeutrinoTaggerCosmic.cxx:536` | FV shrink `−1.5 cm` (6 faces) | absolute margin for vertex-outside-FV test |
| `SteinerGrapher.cxx:86-88,202-214,353` | `Q0=10000`, `charge_threshold=4000` e; distances 0.6/1.8/6.0 cm | prototype constants; gain/noise- and pitch-coupled |
| `NeutrinoEnergyReco.cxx:209,264,287` | 2D-hit match `0.6 cm` | pitch-coupled; SBND also 3 mm — likely transfers |

**(iv-a) DONE 2026-07-30 — the two detector-extent rows are config-fed and SBND sets
them** (owner request: "compare the geometry between MicroBooNE and SBND's FV and
suggest a fix; pull them out to configuration, then add the SBND value"). Unlike
§2e(i-a)/(iii-a), this one **does change SBND output** — the values are a translation of
the uBooNE geometry, not a placeholder. uBooNE stays byte-identical. Toolkit commit
`cbd78820` (parent `f14bbce8` = the §2e(iii-a) `kine_*` commit).

Repro:

```bash
# compiled-config proof (uBooNE unchanged; SBND emits the 5 keys).  The "before"
# tree is a worktree at 66957283, the commit this work started from (the parent
# f14bbce8 landed mid-task from a second session and touches neither key):
#   git worktree add --detach /home/xqian/tmp/wt-head 66957283
qlport/scripts/compile_ub_cfg.sh /home/xqian/tmp/wt-head/cfg ub_head.json
qlport/scripts/compile_ub_cfg.sh $TK/cfg                     ub_now.json
diff ub_head.json ub_now.json                                    # empty (258945 B both)
sbnd_xin/compile_prjob_cfg.sh    $TK/cfg pr_geo_now.json
grep -E 'cosmic_y_|vertex_z_prior' pr_geo_now.json               # 133/163/183/185, 100
# the uBooNE arm is one flag pair (reproduces 100/102/80/50 and 200 exactly):
#   --tla-code pr_y_top=117 --tla-code vertex_z_prior_scale=200
# uBooNE off-gate (35 events, keys absent)
qlport/scripts/sweep_5384.sh geomoff_ub 6
diff sweep/f3coff_ub2/hashes.txt sweep/geomoff_ub/hashes.txt     # empty
# SBND two-arm effect, ONE binary, 48-event nueCC sample
sbnd_xin/geom_ab_batch.sh            # ARM=on = SBND defaults; ARM=off = uBooNE TLAs
sbnd_xin/geom_ab_summary.sh          # per-event mabc-pr.zip member-hash equality
                                     # -> docs/pr/2_geom_ab_summary.tsv
```

**The geometry.** uBooNE's active volume, as the prototype's own apps declare it
(`prototype_base/pid/apps/wire-cell-prod-nue.cxx:417`,
`wire-cell-prod-bee.cxx:304` — `ToyFiducial(..., 3, 117, -116, 0, 1037, 0, 256, ...)`):

| axis | uBooNE | SBND (sensitive box, from the `DetectorVolumes` note in `sbnd/clus.jsonnet`) | ratio |
|---|---|---|---|
| y (vertical) | −116 … **+117** cm (233 cm tall, centre +0.5) | −199.965 … **+199.965** cm (400 cm tall, centre 0) | top face +83 cm |
| z (beam) | 0 … **1037** cm | 0 … **501.0** cm | 2.07× shorter |
| x (drift) | 0 … 256 cm (single drift) | ±201.45 cm (two drifts, CPA at 0) | — |

**How each cut translates.** The two rows need *different* rules, which is why they get
different treatment:

- **The cosmic-`y` cuts are offsets below the top face, and do not scale.** A downward
  cosmic entering the detector roof puts its highest reconstructed point a fixed
  tolerance below the top face — set by reconstruction slop, not by detector height. So
  the uBooNE values are read as 117 − {17, 15, 37, 67} and re-anchored to SBND's +200:
  **183 / 185 / 163 / 133 cm**. Scaling by height instead (×200/117) would give
  171/174/137/85 — a *looser* cosmic tag than uBooNE's own, i.e. more false cosmic tags,
  which is the wrong direction for a cut whose failure mode on SBND today is firing at
  mid-detector.
- **The upstream-`z` prior is a penalty per cm, and the question is dynamic range.**
  Within one cluster the trade-off "how many cm downstream cancels one +0.25 track
  bonus" (50 cm at 200 cm scale) is detector-independent, and by that reading 200 cm
  transfers unchanged. But the site that matters most on SBND is
  `compare_main_vertices_global`, which ranks candidates from **different clusters of
  the beam bundle**, where separations run toward the full detector length: uBooNE gets
  ≈5.2 penalty units end-to-end, SBND at 200 cm would get only 2.5. Shipped value is the
  length-scaled one, **100 cm ≈ 200 × 501/1037 = 96.6** (rounded). The alternative
  reading is one word away — `vertex_z_prior_scale=null` restores 200 cm — and the
  measured effect below is the evidence for choosing between them.

**What shipped** (toolkit, single commit; C++ defaults are the uBooNE literals, so an
absent key is byte-identical):

| knob (cm) | C++ default | SBND | site |
|---|---|---|---|
| `cosmic_y_top_main` | 100 | 183 | `NeutrinoTaggerCosmic.cxx:1176` — main cluster's own highest point (relaxes the vertical-angle cut 20° → 30°) |
| `cosmic_y_top_strict` | 102 | 185 | `:1191` — event highest point, single-cosmic branch |
| `cosmic_y_top_loose` | 80 | 163 | `:1192` — event highest point, global gate on the whole `flagp_cosmic` decision |
| `cosmic_y_small_piece` | 50 | 133 | `:1074` — PCA centre of a <3 cm cluster counted as cosmic debris (`acc_small_length`, which feeds the gate above) |
| `vertex_z_prior_scale` | 200 | 100 | `NeutrinoVertexFinder.cxx:875` + `:3002` |

(All line numbers at `cbd78820`; the pre-change sweep quoted `:1073/:1175/:1190/:1191`
and `:875/:3001` — the block shifted by the added trace line.)

Members live on `PatternAlgorithms` (internal units) and are fed from
`TaggerCheckNeutrino` (cm → internal), the same three-site pattern as
`fit_vertex_min_seg_length`. `vertex_z_prior_scale ≤ 0` warns and keeps 200 (it is a
divisor). jsonnet: `cosmic_y_*` / `vertex_z_prior_scale` args on
`common/clus.jsonnet:tagger_check_neutrino` with the key-suppression idiom; SBND sets
them from a single anchor `local sbnd_y_top = 200.0` in `sbnd/clus.jsonnet` (change that
one number, not the four cuts). `wct-pr-perevt.jsonnet` gains TLAs `pr_y_top` (default
200) and `vertex_z_prior_scale` (default 100), so **the uBooNE arm of any A/B is one
flag**: `--tla-code pr_y_top=117 --tla-code vertex_z_prior_scale=200` reproduces
100/102/80/50 and 200 exactly (verified in the compiled JSON).

Deliberately **not** touched, though they are the same class: the `mv_pt.y() > 0` term in
the same `flagp_cosmic` expression is the detector **mid-plane** and is scale-free on
both detectors (uBooNE centre +0.5 cm, SBND 0); the `highest_y = -100*units::cm` seed at
`:1119` is a pure max-accumulator sentinel that every threshold sits far above (it can
only make `highest_y` *larger* than the true maximum, and −100 fails all four cuts on
both detectors); and `NeutrinoTaggerCosmic.cxx:536`'s −1.5 cm FV shrink, which is an
absolute margin, not an extent.

**Gates.**

| gate | result |
|---|---|
| uBooNE 35-event off-gate | **PASS** — `sweep/geomoff_ub/hashes.txt` == `sweep/f3coff_ub2/hashes.txt`, zero diff, all `rc=0` (the 16-line delta vs `gate3` is the pre-existing one from `7902615c`, identical in `f1off_ub`/`f3coff_ub2`) |
| uBooNE compiled config | **byte-identical**, 258945 B both |
| SBND compiled config (PR job) | the 5 keys appear with 183/185/163/133/100 and **nothing else changes** |
| SBND **production** compiled config | **byte-identical** both entry points that import `sbnd/clus.jsonnet` — `wcls-img-clus.jsonnet` 169809 B and `wct-clus-matching-standalone.jsonnet` 190598 B, zero diff (`sbnd_xin/compile_sbnd_prod.sh`): the new `pr()`/`clus_pr()` args do not leak where `tagger_check_neutrino` is not in `pipeline_names` |
| `wcdoctest-clus` | 565/565 |
| freshness | `local/lib/libWireCellClus.so` == `build/clus/libWireCellClus.so`, 350804144 B, 15:34, both newer than the last edit (the sweep loads from `build/`) |
| SBND arm | **NOT bit-identical by design** — characterized below, no gate label |

Caveat on the binary: a second session was mid-flight on the §2e(iii-a) `kine_*` work in
the same tree, so the gated binary contains **both** changes. The uBooNE PASS therefore
covers both; it does not isolate mine. The SBND two-arm comparison below is
same-binary, so that contamination cancels there.

**Measured effect — 48-event nueCC sample, one binary, both arms.** 46/48 events have
**identical** `mabc-pr.zip` member hashes; 2 differ; 3 events fail identically in both
arms (10550, 271851 throw the same `RuntimeError`; 433451 `bad_alloc` — pre-existing,
not knob-related). Both differences come from the **z-prior**; the cosmic-`y` re-anchor
changed no verdict on this sample (`flagp_cosmic=false` in all 44 events that reach
part D — expected: these are νe candidates that already survived cosmic rejection, and
production also runs the PR chain with `nu_skip_cosmic=true`). Reach, for context: 10 of
those 44 events have `highest_y > 163` cm and 6 sit in the 80–163 cm band where the
uBooNE cut passes the loose gate and the SBND cut does not — the re-anchor removes a
latent false-positive channel rather than fixing an observed one.

- **evt 235435 — main cluster switches, quotable mechanism.** The two candidates are
  37.7 cm apart in z (`z_norm=0.1884` at 200 cm). At the uBooNE scale cluster 2 wins by a
  hairline, 0.6866 vs 0.6250; doubling the prior costs cluster 2 exactly the extra
  0.1884, and cluster 24 takes the main slot:
  ```
  off: cluster 2 score_A=0.1866 z_norm=0.1884 -> score_E=0.6866   cluster 24 score_E=0.6250
  on : cluster 2 score_A=-0.0018 z_norm=0.3768 -> score_E=0.4982  cluster 24 score_E=0.6250
       check_switch_main_cluster: switch main cluster 2 -> 24
  ```
  Consequence downstream: `nue_score` 4.30 → −15, i.e. the nue tagger no longer fills its
  branch variables (`br_filled != 1`) for the new main cluster. Reported, not tuned
  (CLAUDE.md §5 rule 7). **Owner decision 2026-07-30, after seeing the Bee sets above
  and the DL-ON numbers below: keep 100 cm.** The event stays on the list as a scan
  item for the geometric fallback (both candidates exist in either arm — the question
  is which one is the neutrino vertex), not as a blocker on the value.
- **evt 269774 — main vertex moves within the main cluster**, (16.5, −93.0, 182.7) with 3
  legs → (39.7, −67.3, 159.9) with 2 legs, i.e. 22.8 cm upstream in z; `nue_score`
  0.497 → 1.844, `numu_score` −2.446 → −2.416. Both are real (non-sentinel) BDT values.

Bee sets for the two changed events (**geometric arm**, where both differences show;
uploaded 2026-07-30 from `/home/xqian/tmp/geomab/<evt>/<arm>/mabc-pr.zip`):

| event | uBooNE values (`off`) | SBND values (`on`) |
|---|---|---|
| 235435 | [set fdbdd87f](https://www.phy.bnl.gov/twister/bee/set/fdbdd87f-b839-459a-8a47-643641852fb3/event/list/) — main cluster **2**, vertex (−82.9, 181.9, 171.2) | [set 56ce0478](https://www.phy.bnl.gov/twister/bee/set/56ce0478-ff13-4cd9-a56c-7a927020b406/event/list/) — main cluster **24**, vertex (−101.9, 159.7, 133.5) |
| 269774 | [set e4b8be1a](https://www.phy.bnl.gov/twister/bee/set/e4b8be1a-8dfc-4023-b3da-59d5bf9169d4/event/list/) — cluster 13 vertex (16.5, −93.0, 182.7), 3 legs | [set a46e0d7a](https://www.phy.bnl.gov/twister/bee/set/a46e0d7a-56f7-4fa6-a0ea-b56239ec76f1/event/list/) — cluster 13 vertex (39.7, −67.3, 159.9), 2 legs |

Both candidate positions exist in both arms of 235435 — only the *ranking* changes, so
the scan question is "which of the two is the neutrino vertex", not "did a vertex move".

**Where these sites sit relative to the DL vertex — and the production-configuration
numbers.** The measurement above is the *geometric* arm (`dl_weights=''`); SBND
production defaults the SCN vertex **ON**, and the three sites are not equally exposed
to it (`TaggerCheckNeutrino.cxx:506-524`):

| site | when | DL ON |
|---|---|---|
| `compare_main_vertices` (`:875`) | inside `determine_main_vertex`, **per cluster, before any DL** — it produces the candidates DL re-ranks | **live** |
| `compare_main_vertices_global` (`:3002`) | inside `determine_overall_main_vertex`, called only `if (!flag_dl_changed)` | **bypassed** whenever DL moves the vertex |
| the four `cosmic_y_*` | `cosmic_tagger` at `:584`, after DL + `improve_vertex` + shower clustering | always after DL |

Re-running the same 48 events with DL ON (`sbnd_xin/run_pr_geom_arm_dl.sh`, one binary,
both arms) gives **47/48 identical, 1 changed**:

- **evt 235435's main-cluster switch disappears** — the arms are byte-identical
  (`8a45c044…`), because DL fired and `compare_main_vertices_global` never ran. So the
  `2 → 24` switch, and the hand-scan question it raises, belong to the **geometric
  fallback path** (the gate arm, plus any event where DL declines to move the vertex),
  not to production-with-DL.
- **evt 269774 still differs**, as expected from the pre-DL `:875` site: cluster 13's
  main vertex is (16.5, −93.0, 182.7) with 3 legs vs (39.7, −67.3, 159.9) with 2. But
  the **tagger outputs agree exactly** (`nue_score` 4.301, `numu_score` −0.600 in both);
  DL overrides the final vertex, so only the per-cluster bookkeeping in the Bee zip moves.

M4 control, since the DL vertex is not bit-stable in general: both arms were run twice
independently and each reproduced its own hash exactly (`geomab_dl` vs `geomab_dlfull`,
on **and** off), so the single DIFF is a real effect, not SCN noise. Two events fail in
both arms and both configurations for unrelated reasons (271851 the same `RuntimeError`;
one `bad_alloc` that moves between events with concurrency — load, not knob).

Net: with SBND's production defaults the shipped change is **almost inert today** (1 of
48 events, and that one keeps its verdicts); it becomes consequential exactly when the
DL vertex is off or declines to move — which is also the arm every identity gate uses.

**Side finding worth its own line (feeds gap G1).** `nue_score` is **saturated** on most
of this sample: `±4.3009 = ±log10(0.9999·2/0.0001)` is the clamp at
`UbooneNueBDTScorer.cxx:1920-1923`, i.e. the uBooNE-trained BDT output pinned at ±1.
28 of 44 events sit at +4.3009 and 4 at −4.3009; `−15` is the separate
"`br_filled != 1`" default at `:1655,1925`. So a quoted `nue_score = 4.301` on SBND
today means "pegged at the ceiling", **not** a resolved per-event score — one more
reason the uBooNE weights cannot rank SBND events until they are retrained (§2d G1).

**(iv-b) DONE 2026-07-30 — the muon dQ/dx-vs-length envelope is config-fed AND SBND
sets a table-derived fit** (pr/10 §4, toolkit `21c31439` + `db625c81`). One knob
`muon_dqdx_curve [c0, c1, pivot_cm, power]` replaces the literal at **nine** sites
(this table's row undercounted 3; also note the prototype's SSM copy
`NeutrinoID_ssm_tagger.h:743` is dimensionally wrong by ×100 and the toolkit port
silently normalized it — pr/10 §8, porting-dictionary item). Defaults = the uBooNE
refit, byte-identical, and **bit-identical even when passed explicitly** (same mabc
content hash on evt 172230). SBND production: **`[0.8826, 1.0587, 18, 0.4745]`** =
the SBND stopping-muon table median (0.5 kV/cm, /48000) × the uBooNE empirical/table
margin g(L)=1.16–1.32, same functional form (`dqdx_rr_sample/fit_muon_length_curve.py`,
rms 1.5e-3). Looser at short L (2.83 vs 2.53 × MIP at 5 cm), converges at long L.
Effect on nuecc48 172230/235435/444187 (with (iii-b), same arm): `numu_score` moves
(−0.455→0.121 / 0.426→−0.123 / 0.358→0.889), **no verdict flips**. Scales as
1/`mip_dqdx_median` — re-derive when the 48000 placeholder becomes a measurement.

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
unlike the inert SCN knobs — they feed 8 BDT features on every SSM-tagged event today)
and the eight `kine_*` charge→energy constants (§2e(iii-a): same status — live on every
reconstructed energy, inherited from uBooNE, nobody has an SBND value).
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
→ ~~(iv) cosmic-y + vertex-z prior~~ **(done 2026-07-30, §2e(iv-a) — knobs exist AND
SBND sets translated values; this is the first worklist item that changes SBND output,
values ADOPTED by the owner 2026-07-30; the residual is a scan of evt 235435 on the
geometric fallback, not plumbing and not a blocker)** → ~~(iii) energy reco~~ **(plumbing done 2026-07-30,
§2e(iii-a); the pr/10 round then calibrated the three recombination factors —
§2e(iii-b); still uBooNE: fudge factors, plane weights, asymmetry switch, W-value)**
→ ~~(iv) muon dQ/dx-vs-length curve~~ **(done 2026-07-30, §2e(iv-b) — nine sites, one
knob, SBND fit adopted)** → (v)
fit-JSON thresholds; (vi) waits for the SCN/BDT work (G1/G3).
**Owner-approved and ON for SBND, 2026-07-30 (toolkit `6d0396a2`):**
`use_power_recomb` (free-power recombination for both taggers) and
`sp_dedx_use_recomb_model` + `sp_mean_dedx_cut=2.23` (SinglePhoton stem dE/dx) —
the pr/10 §7 before/after on three nuecc48 events showed small persisted diffs and
no verdict flips; the ON-defaults compile is cmp-identical to that evidence arm.

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
- **`singlephoton_tagger`'s apa/face derivation is unguarded**
  (`NeutrinoTaggerSinglePhoton.cxx:2186-2196`): an uncontained main vertex gives it
  `apa=-1, face=1`. No throw today (it never calls `wire_angles`), but those values reach
  its point-cloud queries. The one-line `apa() >= 0` guard from §2e(i-b) applies verbatim.
- **No event is yet known where the nue apa/face fix moves `nue_score`** — on evt 444187
  it flips `mip_quality_flag`/`mip_quality_overlap` but the BDT score is unchanged
  (§2e(i-b)). Worth a pass over the 15 APA-1 events of the nueCC48 sample before claiming
  any selection impact.
- The §2e tuning worklist, in its stated priority order: correctness items (SSM beam
  vectors — now half-closed, see above; ~~nue apa/face~~ **done, §2e(i-b)**;
  SinglePhoton inline box model) →
  ~~chain-wide MIP normalization knob~~ (done, §2e(ii-a); the open piece is the
  measurement above) → cosmic-y / vertex-z-prior rescaling → energy-reco factors +
  `cal_corr_factor` stub → fit-JSON absolute-charge thresholds.
- Track B/C campaign execution (§3) and the V0 instrumentation knobs (§4).
- Beam-window calibration on a larger sample; multi-bundle "longest wins" rule.
- MC-with-truth route B + the `recob::OpFlash` converter (§3.4).
- The `do_tracking()` `$`-scoping oddity (§2d note) — dead code, flag to owner.

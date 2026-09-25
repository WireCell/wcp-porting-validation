# doc 120 — Beam-particle pattern recognition on the beam-flash-matched bundle (`CheckBeamParticle`)

**Status (2026-09-25):** implemented, gated, run on the 6 doc-118 kaon events. New component `CheckBeamParticle`
(clus) + `run_pr_evt.sh -beam`. Byte-identical for every existing pipeline (new component, null-default TLAs; the
`-stm` / `-nu` / `-nu-legacy` compiled configs are unchanged; hash gate d120a/d120b PASS). **Open for the owner:**
the beam entry's drift coordinate — the GDML beam-plug axis and the data disagree by ~100 cm (sec 1.3); the
defaults follow the data. **Installed in the shared tree 2026-09-25 14:25 (owner's go)** and smoke-tested: the
bare runner reproduces arm B on all six events (sec 8). **Bee set uploaded (owner's go):
<https://www.phy.bnl.gov/twister/bee/set/68caddae-7c7a-45b6-9312-54ddfdd5fe6d/event/list/>** (sec 6). Toolkit
`57f99628`, wcp `59e42b14` (+ this record).

Owner request: for PDVD the beam is a charged particle *entering* the detector, so (1) the main vertex is the beam
particle's ENTRY point, like the STM+Michel chain's entry; (2) the rest of the PR is the neutrino chain
(proto-segments, EM shower clustering, particle id, particle flow), not the STM+Michel chain; (3) no cosmic
taggers and no DL vertex; (4) outputs = the Bee particle-flow display and the ROOT files; run only on the
beam-flash-matched bundle. Decisions taken during the plan: a NEW component (TaggerCheckNeutrino untouched),
entry from the simulation geometry + the data, main cluster = the bundle cluster closest to the entry point,
ROOT = `tracking-pr.root` + `T_kine`.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/pdvd
# knob on: the six doc-118 kaon events (pctrees from run_clus_evt.sh -save-pctree, symlinked into new tag dirs;
# the light dirs with the doc-119 trigger metadata carry the _d119beam suffix)
for e in 157312 245576 317673 191916 2001 36591; do d=work/039305_${e}_d120beam10; mkdir -p $d
  ln -sfn ../039305_$e/pctree-evt$e.tar.gz $d/; ln -sfn ../039305_$e/pctree-evt$e.tlas $d/; done
PDVD_BEAM_LIGHT_SUFFIX=_d119beam PDVD_PR_TLA="-S dl_weights=''" ./run_pr_evt.sh -beam -s d120beam10 39305 157312
grep 'CheckBeamParticle:' work/039305_157312_d120beam10/wct_pr_039305_157312.log
python3 kaon/make_d120_bee_zip.py d120beam10 /home/xqian/tmp/d120_bee/d120-beam6.zip     # Bee set, 6 events
# knob off (hash gate) and the compiled-config proof: sec 4
# unit tests: ./build/clus/wcdoctest-clus -tc='*beam_particle*,*CheckBeamParticle*'
```

The runs above used the private arm-B install `/home/xqian/tmp/d120inst/B` (`PATH`/`LD_LIBRARY_PATH` pinned, as
`kaon/d119s_gates.sh` does); with the shared tree installed (sec 8) the pin is unnecessary.

## 1. The beam entry point

### 1.1 From the simulation geometry

`protodunevd_v5_ggd.gdml` (dunecore v10_2x, the `protodunevd_v5_geo` that `services_protodunevd.fcl` loads; the
beam volumes are identical in v10_20_08, v10_21 and v10_26, and in the wires/nowires/PNS variants). World frame =
cryostat frame + (20, 0, 149.65) cm, which is also the WCT frame (cathode centre x=0, anodes at ±341.55, +x up =
the top drift volume).

| point | world (cm) |
|---|---|
| beam-plug upstream cap (steel) | (248.20, 427.02, −277.37) |
| plug centre (G10, N₂-filled, r 12.7/13.97, L 386.78) | (229.77, 290.72, −141.07) |
| plug downstream cap — the beam leaves the plug into bare LAr | (211.34, 154.42, −4.77) |
| axis crossing the active face z=0.6 | **(210.61, 149.05, 0.60)** |

Direction (particles' travel) = downstream − upstream = **(−0.0952, −0.7039, +0.7039)**: 45° in y–z toward −y,+z,
5.46° below horizontal (the `rBeamWRev3` rotation 5.4611° and the cutTube normals agree). The axis extended to
y=0 hits (190.5, 0, 149.65), the detector's y–z centre. It enters the top volume in anode 6 face 0 (19 cm from
the y=168.5 seam) and would leave through the downstream face near (170, −149, 299) in anode 5.

The generator `protodunevd_triggeredbeam` (`dunesim protodunebeam.fcl`) starts at BeamX/Y/Z = (248, 427.2,
−277.2) = the upstream cap, with the comment "7.7 deg below horizontal, 45 degree between y/z". The SingleGen gun
(`gen_protodunevd_singlep.fcl`, `pdvd_1GeV_*`) starts at **(94.8, 142.6, 0.7)** with direction
(−0.134, −0.701, +0.701).

### 1.2 From the data

Read-only 3-D line fits of the doc-118 beam-flash-matched clusters (`work/039305_<evt>/calib-evt*.json`, x
T0-corrected with the beam flash time; the doc 119 side-panel check verified this placement against the
toolkit's `x_t0cor` to 0.0015 cm):

| event | cluster | direction (x, y, z) | line at z=0 (x, y) | first point | L (cm) |
|---|---|---|---|---|---|
| 157312 | 4000090 | (−0.085, −0.699, +0.710) | (108.2, 174.2) | (110, 190, −16) | 184 |
| 245576 | 4000151 | (−0.389, −0.426, +0.817) | (118.5, 173.3) | (118, 172, 2) | 167 |
| 245576 | 4000115 | (+0.344, −0.939, +0.005) | — | (148, 172, 51) | 125 |
| 317673 | 4000101 | (−0.210, −0.719, +0.662) | (102.0, 129.7) | (106, 143, −12) | 453 |
| 191916 | 4000106 | (+0.878, −0.175, −0.446) | — | (5, −184, 299) | 60 (KS 0.85: not the beam) |

The three beam tracks enter through the z≈0 face at **x ≈ 102–118 cm** (≈ 230 cm below the top anode, ≈ 1 m above
the cathode), **y ≈ 130–175 cm**, travelling toward −y, +z, slightly downward; mean direction ≈ (−0.23, −0.62,
+0.73).

### 1.3 Agreement and the one discrepancy (owner's ruling needed)

y, z and the y–z angle agree between the GDML axis and the data (y ≈ 149 vs 130–175; 45° vs 41–62°). **The drift
coordinate does not: GDML x = 210.6 at the face, data x ≈ 102–118 — ~100 cm apart**, from three independent tracks
in three events. The SingleGen gun's x = 94.8 sides with the data. This host cannot settle it (no survey; the
generator's `NP02XDrift`/`NP02Rotation` code ships only as a `.so`; the doc 118 §5.3 cert-slab axis (−0.840,
+0.542) was measured on halo tracks). Per CLAUDE.md §5.7 the number is reported, not tuned:

- defaults = the data-derived entry **(110, 159, 0.6) cm** with the GDML direction **(−0.095, −0.704, +0.704)**
  and an acceptance radius of **50 cm** (the three tracks scatter ≤ 16 cm in x and 45 cm in y);
- with the GDML x the same radius would select nothing on today's data (nearest track 100 cm away);
- both are TLAs (`beam_entry_point_cm`, `beam_dir`, `beam_entry_max_dist_cm`), so the owner's ruling is a one-line
  change and no code moves.

## 2. Design

### 2.1 Why a new component

The neutrino chain is one visitor, `TaggerCheckNeutrino` (`clus/src/TaggerCheckNeutrino.cxx`, 4849 lines,
SBND/uBooNE production). No config key lets a caller supply the main vertex, and its cosmic veto
(`nu_skip_cosmic`) would drop a crossing beam particle, which the TGM tagger flags. So, as CheckSTM_Michel did for
the STM+Michel chain (doc pdvd/48), `CheckBeamParticle` is a fork by duplication (CLAUDE.md M10): its own fitter
and graph, the same `PatternAlgorithms` free functions in the same order, TaggerCheckNeutrino untouched.

### 2.2 The three selection rules (pure functions, `clus/inc/WireCellClus/BeamParticleFunctions.h`)

1. **The beam bundle** — `beam_particle_pick_bundle`: among the `Flags::main_cluster` clusters with
   `beam_window_low <= cluster_t0 < beam_window_high` (the TaggerCheckNeutrino convention; `cluster_t0` is the RAW
   matched flash time), grouped by `matched_flash_gid`, the gid whose flash is the **brightest** — the same rule the
   Bee `op_beam` label uses (doc 119). No valid flash row / a tie ⇒ the gid holding the longest main, then the
   smallest gid. `low >= high` (the C++ default) means OFF and OFF means **nothing is selected**: this stage never
   falls back to "every bundle".
2. **The main cluster** — `beam_particle_pick_main`: the bundle member whose closest point to the nominal entry is
   nearest (ties → longer → smaller id), among members with `length >= min_main_length_cm` (sec 5.2 for why the
   floor exists). Every other member is a companion, exactly as the neutrino chain treats the bundle.
3. **The entry** — `beam_particle_choose_entry`: of the main cluster's two main-axis extreme points
   (`Cluster::get_main_axis_points`), the one nearer the nominal entry; within `entry_tie_tol_cm` (5) the end whose
   direction toward the other end has the larger dot with `beam_dir`. Accepted when its distance to the nominal is
   ≤ `beam_entry_max_dist_cm` (and, if `beam_dir_max_angle_deg` ≥ 0, within that angle of `beam_dir`; off by
   default). The entry is then snapped onto the PR graph: the nearest vertex within `entry_snap_tol_cm` (10), else
   the nearest segment is split there (`PR::break_segment`, the CheckSTM_Michel `anchor_vertex` recipe).

### 2.3 The chain (`CheckBeamParticle::visit`, copied line ranges of TaggerCheckNeutrino.cxx)

| step | what | source |
|---|---|---|
| fitter + graph | own `TrackFitting` seeded from the runtime JSON, `preload_clusters(main + companions)` | CheckSTM_Michel.cxx:2716-2735 |
| main cluster | `find_proto_vertex(true,2,true)`, `clustering_points`, `separate_track_shower`, `determine_direction`, `shower_determining_in_main_cluster`, `determine_main_vertex`, `reassociate_cluster_orphans` | TCN:3657-3699 |
| companions | the long/short branches, `deghosting` over the bundle | TCN:3707-3756 |
| **entry vertex** | rule 3 above; `kNeutrinoVertex`; `map[main] = entry` | replaces TCN:3765-3813 (DL + `determine_overall_main_vertex`) |
| refinement | `clustering_points`, `reassociate_cluster_orphans`, `examine_direction(entry, entry, …, final)` | TCN:3894-3930 (`improve_vertex` only with `improve_entry_vertex`, default off) |
| EM showers | `demote_cross_cluster_straight_stems`, `shower_clustering_with_nv` (incl. the pi0 maps), `reconcile_particle_flags` | TCN:3949-4001 |
| no taggers | `TaggerInfo` at `init_tagger_info` defaults + `match_isFC` | TCN:4322-4337 |
| kinematics | `fill_kine_tree` → `KineInfo` (cluster_id, gid, nu_index 0, has_vertex 1) | TCN:4348-4354 |
| publish | `set_pi0_data`, `set_main_vertex`, `set_showers`, `set_kine_info`, `set_tagger_info`, `assemble_fitted_charge_2d`, `grouping.set_track_fitting(tf)` + `"nu0"` | TCN:4362-4448 |

`determine_main_vertex` still runs — its graph work (`examine_structure_*`, `examine_vertices`) is wanted — but the
vertex it picks is only recorded (`geo_vtx`, "what the neutrino chain would have chosen") and then replaced.
Deliberately skipped: the DL/SCN vertex and dual chain, the kink/junction snaps, graph audit and stitch, the
long-muon fallbacks and cathode bridge, all five taggers, MCS, the BDT scorers.

Publication is identical to the neutrino stage's, so the Bee PR layers (`track_fit` / `shower_track` / `vertices`
/ the `mc` particle flow), `PdvdPrMagnifyTrackingVisitor`, `UbooneTaggerOutputVisitor` (`T_kine`) and
`PrDisplayDump` render it unchanged under visitor `CheckBeamParticle:pr`.

### 2.4 Record

One row per event on the main cluster, PC `beam_particle` → `T_beam_particle` in `tracking-pr.root`: gid, t0,
flash pe, n_in_window, n_gids, n_companions, main length, nominal / entry / exit / entry-vertex / geo-vertex
coordinates, entry distance and cos to the beam, `entry_ok`, snap distance and whether a segment was split,
n_showers, `kine_reco_Enu`, `match_isFC`. Cluster scalars `beam_particle_main`, `beam_particle_gid` on the bundle.
Log: one `CheckBeamParticle:` line each for the selection, every bundle member, the main pick, the entry and the
verdict.

## 3. Configuration, TLAs, runner

- C++ (`CheckBeamParticle::default_configuration()`, pinned by `doctest_check_beam_particle_defaults.cxx`):
  `beam_window_low/high` 0/0 (OFF), `beam_entry_point_cm` [110, 159, 0.6], `beam_dir` [−0.095, −0.704, 0.704],
  `beam_entry_max_dist_cm` 50, `beam_dir_max_angle_deg` −1, `entry_tie_tol_cm` 5, `entry_snap_tol_cm` 10,
  `min_main_length_cm` 0, `improve_entry_vertex` false, `entry_fail_fallback_geo` false, `publish_nu_slots` true,
  `mip_dqdx` 50000 / `mip_dqdx_median` 43000 e/cm, and the PR-partition keys by TaggerCheckNeutrino's names.
- jsonnet: builder `cm.check_beam_particle(...)` (`cfg/pgrapher/common/clus.jsonnet`), the PDVD instance
  `check_beam_particle` in `protodunevd/pr.jsonnet` (same fitter JSON, particle dataset and `pdvd_recomb` as
  `tagger_check_neutrino`, the shared `stm_michel_partition` PR knobs + `beam_pr_knobs`), a three-way `pr_visitor`.
  `wct-pr-perevt.jsonnet` TLAs: `beam_trigger_us`, `beam_tc_type`, `beam_tc_types` [14,15,20,21,22],
  `beam_window_rel_us` [−1.5, −0.3], `beam_entry_point_cm`, `beam_dir`, `beam_entry_max_dist_cm` (null = C++
  default), `beam_pr_knobs` (PDVD default `{min_main_length_cm: 10}`). When a beam trigger is given,
  `beam_window_eff_us = beam_trigger_us + beam_window_rel_us` REPLACES `beam_window_us` for every consumer
  (steiner, taggers, protect_bundle, tagger_check_neutrino, check_beam_particle), so only the beam bundle gets a
  Steiner graph.
- **The axis.** `beam_trigger_us` is the CTB trigger on the RAW flash-time axis, which `cluster_t0` lives on, so no
  side trigger offset is added. The Bee `op_beam` label (wct-clustering.jsonnet) adds `side_trigger_offset` because
  `op_t` is offset; the two windows are different numbers for the same flash (e.g. 157312: PR window
  [2792.084, 2793.284) µs, the flash at 2792.678; the pctree `.tlas` `trigger_offset_top_us` −2362.125 vs the
  light `offset_top_us` −2375.632 differ by the +13.507 µs `PDVD_QL_EXTRA_OFFSET`).
- Runner `run_pr_evt.sh -beam`: `PIPE_BEAM = switch_scope, flag_mains, unmerge_assoc, steiner, fiducialutils,
  check_beam_particle, tracking_visitor, tagger_output, pr_display`. It reads `trigger_us`/`tc_type` from the light
  archive named by the pctree `.tlas` `opflash_input=` line (with `PDVD_BEAM_LIGHT_SUFFIX` inserted after
  `_light<EVT>`; the doc-118 kaon light dirs without the suffix predate the doc-119 label), skips an event without
  a beam trigger or with a non-beam `tc_type`, and forwards the two numbers. `PDVD_BEAM_TRIGGER_US` /
  `PDVD_BEAM_TC_TYPE` override.

## 4. Gates

- **Tests first.** `clus/test/doctest_beam_particle_functions.cxx` (3 cases) and
  `doctest_check_beam_particle_defaults.cxx` were built before the implementation existed: the link failed with 17
  undefined references (`/home/xqian/tmp/d120_build_a.log`). With the implementation: 4 cases / 91 assertions pass;
  the full `wcdoctest-clus` 450/450 cases, 716841 assertions (`/home/xqian/tmp/d120_doctest_full.out`).
- **Compiled-config identity** (`/home/xqian/tmp/d120_cfgdiff.sh`, the runner's exact `wcsonnet` argument list on
  039305/157312's sidecar; A = clean 9de7fcae cfg in the worktree, B = this change): `-stm` 274364 B, `-nu` 279325 B,
  `-nu-legacy` 285167 B — all three byte-identical. `-beam` with `-S beam_trigger_us=2793.584 -S beam_tc_type=22`:
  `CheckBeamParticle:pr` present, `beam_window_low/high` = 2792.084 / 2793.284 µs on both `CreateSteinerGraph:pr`
  and the new stage, Bee visitors `CheckBeamParticle:pr` (+ `TaggerCheckSTM:pr` for the stm layers), `bee_pf` =
  `CheckBeamParticle:pr`.
- **Knob-off hash gate** (arms A = clean 9de7fcae, B = A + this change, private installs
  `/home/xqian/tmp/d120inst/{A,B}` built in the `/home/xqian/tmp/d119wt` worktree; `-nu` with
  `PDVD_PR_TLA="-S dl_weights=''"`, tags `d120a` / `d120b`):

  | event | mabc-pr.zip (`abtest/hash_archive.py`, 17 members) | calib-pr json sha256 | tracking-pr.root (280 branches, awkward lists) |
  |---|---|---|---|
  | 157312 | `41d05530…` = `41d05530…` | identical | 280/280 identical |
  | 317673 | `3f1701e1…` = `3f1701e1…` | identical | 280/280 identical |

  Null pair: arm A twice on 157312 (`d120a` vs `d120a2`) — mabc-pr.zip `41d05530…` both, ROOT 280/280. The only
  shared C++ touched is the one-line `T_beam_particle` hook in `root/src/PdvdPrMagnifyTrackingVisitor.cxx`
  (absent-PC no-op), which this gate covers.
- Freshness: `/home/xqian/tmp/d120inst/B/lib/libWireCellClus.so` 14:15 (B2, with the per-member log line) is newer
  than the last edit to `CheckBeamParticle.cxx`; the shared `local/lib` is NOT touched (sec 8).

## 5. Results on the doc-118 kaon events (arm B, `_d120beam10`)

### 5.1 Per event

| event | mains / in window | beam gid (pe) | t0 (µs) | main cluster (L) | entry (x, y, z) cm | d_nominal | cos_beam | snap | geo-vertex d | showers | Enu (MeV) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 157312 | 43 / 2 | 96 (41416) | 2792.678 | 30 (94 cm) + 20 comp. | (110.6, 168.6, 8.6) → vtx (110.9, 167.4, 8.3) | 12.5 | 0.99 | 0.3 | 76.8 | 14 | 693.1 | entry_ok |
| 245576 | 40 / 3 | 74 (30492) | 2694.863 | 12 (169 cm) + 20 comp. | (118.5, 173.1, 2.5) → (118.1, 171.9, 3.0) | 16.5 | 0.92 | 0.0 | 1.3 | 20 | 1006.7 | entry_ok |
| 317673 | 42 / 1 | 102 (36331) | 2815.838 | 33 (405 cm) + 6 comp. | (99.7, 126.3, 6.0) → (99.9, 125.9, 6.1) | 34.7 | 0.99 | 0.0 | 0.5 | 5 | 909.1 | entry_ok |
| 191916 | 49 / 1 | 84 (83592) | 2864.671 | 37 (61 cm) | axis end (57.2, −193.7, 272.9) | 448.7 | 0.26 | — | — | — | — | entry cut: not the beam particle, nothing published |
| 2001 | 43 / 0 | — | — | — | — | — | — | — | — | — | — | no beam bundle |
| 36591 | 49 / 0 | — | — | — | — | — | — | — | — | — | — | no beam bundle |

Every picked flash sits at trigger − 0.9 µs (0.906 / 0.913 / 0.930 / 0.881), i.e. the doc 118/119 beam flash, and
the three reconstructed mains are the doc-118 clusters 4000090 / 4000151 / 4000101 (ids differ after
`unmerge_assoc`). The entry-cut and no-bundle events are handled as designed: the PR layers stay empty
(`require_pr_graph`), `PrDisplayDump` writes no vertex, `tagger_output` has no row.

Particle flow (`mc` layer / `T_kine`), rooted at the entry:

- 157312: `nu → mu- 564 MeV` (entry → (99.7, 110.0, 58.1)), with two detached 76 / 42 MeV pieces and an 8.5 MeV
  proton stub at the entry.
- 245576: `nu → proton 641 MeV` as the primary, plus two 175 / 178 MeV muon-typed pieces further down the bundle.
- 317673: `nu → mu- 909 MeV`, the single 405 cm track.

The particle labels are the neutrino chain's (dQ/dx PID against the muon/proton templates; a beam kaon/pion is
not in its vocabulary), so "mu-"/"proton" here read as "MIP-like"/"heavy". The PF tree text "reco nu … (no BDT
scores)" is the Bee producer's fixed label.

### 5.2 The one rule change from the plan: a length floor on the main pick

With `min_main_length_cm = 0` (tag `_d120beam`) 317673 picked **cluster 140, a 2.2 cm fragment 27.8 cm from the
nominal entry**, over the 405 cm beam track (cluster 33, closest point (99.7, 126.3, 6.0), 34.7 cm away — the
track enters 33 cm below the nominal y). Its "entry" then had cos_beam 0.09 and the published flow was a 1.3 MeV
stub with four zero-energy showers. The full member list (log, `member cluster` lines):

| cluster | L (cm) | closest point | d_nominal (cm) |
|---|---|---|---|
| 33 | 405.2 | (99.7, 126.3, 6.0) | 34.7 |
| 140 | 2.2 | (123.1, 169.2, 22.9) | 27.8 |
| 137 | 2.2 | (130.5, 147.4, 53.0) | 57.4 |
| 139 | 3.0 | (13.8, 38.2, 46.3) | 161.0 |
| 138 | 6.1 | (69.2, 23.7, 101.9) | 173.8 |
| 136 | 9.8 | (12.7, −169.5, 282.6) | 443.7 |
| 135 | 2.4 | (17.7, −187.1, 285.9) | 457.9 |

"Closest to the entry" alone cannot separate a fragment near the face from the track; a length floor can, and
the C++ knob `min_main_length_cm` exists for it. The PDVD job default is now `beam_pr_knobs =
{min_main_length_cm: 10}`; 157312 and 245576 are unchanged by it (their mains are 94 / 169 cm and were already the
nearest). This is a selection operating point for a new stage, not a physics retune — flagged here for the
owner's veto; `-S 'beam_pr_knobs={}'` restores the bare rule (Bee set `d120-beam6-nofloor.zip`).

## 6. Bee

`/home/xqian/tmp/d120_bee/d120-beam6.zip` (66 members, 6 events in the doc-118 order 157312, 317673, 245576,
191916, 2001, 36591; layers `clustering`, `track_fit`, `shower_track`, `vertices`, `mc`, dead areas) and the
no-floor control `d120-beam6-nofloor.zip`. Offline check on 157312: the `vertices` layer holds one q=15000 point at
(110.9, 167.4, 8.3) = the entry vertex; the `mc` tree is rooted there.

**Uploaded 2026-09-25 (owner's go):** <https://www.phy.bnl.gov/twister/bee/set/68caddae-7c7a-45b6-9312-54ddfdd5fe6d/event/list/>
(`upload-to-bee.sh`, log `/home/xqian/tmp/d120_bee/upload.out`). Live check on the BNL page (Chromium under Xvfb,
`kaon/bee_pf_check.py`, screenshots `/home/xqian/tmp/d120_pw/set2_ev*.png`), PR layers selected:

| idx | event | track_fit / shower_track / vertices points | main vertex (q=15000) | PF tree root |
|---|---|---|---|---|
| 0 | 157312 | 468 / 1752 / 54 | (110.9, 167.4, 8.3) | reco nu 693.1 MeV |
| 1 | 317673 | 760 / 3470 / 17 | (99.9, 125.9, 6.1) | reco nu 909.1 MeV |
| 2 | 245576 | 967 / 2826 / 54 | (118.1, 171.9, 3.0) | reco nu 1006.7 MeV |
| 3 | 191916 | clustering only (no PR layers, no `mc`) | — | — (entry cut, as designed) |
| 4, 5 | 2001, 36591 | clustering only | — | — (no beam bundle) |

## 7. Open items

1. **Entry x: GDML 210.6 vs data ≈110 cm** (sec 1.3). Owner's ruling; both values are TLAs.
2. `improve_vertex` on a boundary entry is off (`improve_entry_vertex`); on, it would re-fit the entry position
   from the segments — measure before turning on.
3. `T_tagger` rows carry only `match_isFC`; every BDT feature is at its `init_tagger_info` default (cosmic_flag
   1, scores 0). Readers must not interpret them.
4. 245576's bundle holds three sizeable clusters; the two 175 / 178 MeV muon-typed pieces are companions in the
   flow, not separate candidates. A per-cluster mode ("run each as its own candidate") was discussed and not built.
5. The length floor (sec 5.2) is a selection operating point chosen on one event; the owner may prefer a
   distance-to-axis metric instead. The candidate table is in the log of every run.
6. The doc-118 light dirs without the `_d119beam` suffix have no trigger metadata; a production run needs
   `run_light_evt.sh` with `PDVD_BEAM_LABEL=1` (the default since doc 119), after which `-beam` needs no suffix.
7. Timing: the stage takes 0.8–1.8 s per event on these top-only pctrees (steiner runs on the beam bundle only).

## 8. Files, commits, install

- toolkit (branch `apply-pointcloud`): new `clus/inc/WireCellClus/BeamParticleFunctions.h`,
  `clus/src/BeamParticleFunctions.cxx`, `clus/src/CheckBeamParticle.cxx`,
  `clus/test/doctest_beam_particle_functions.cxx`, `clus/test/doctest_check_beam_particle_defaults.cxx`; edited
  `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/protodunevd/pr.jsonnet`,
  `cfg/pgrapher/experiment/protodunevd/wct-pr-perevt.jsonnet`, `root/src/PdvdPrMagnifyTrackingVisitor.cxx`.
- wcp-porting-img (`main`): this doc, `pdvd/run_pr_evt.sh` (`-beam`), `pdvd/kaon/make_d120_bee_zip.py`.
- **Installed 2026-09-25 14:25 (owner's go):** `./wcb build --notests -p --targets=WireCellClus,WireCellRoot`
  then `./wcb install --notests -p --targets=WireCellClus,WireCellRoot` (rc 0, `/home/xqian/tmp/d120_install.log`;
  no wire-cell process was running). Freshness: `local/lib/libWireCellClus.so` and `libWireCellRoot.so` 14:25 vs
  `CheckBeamParticle.cxx` 14:14 / `PdvdPrMagnifyTrackingVisitor.cxx` 14:08; `strings` 679 `CheckBeamParticle`,
  1 `T_beam_particle`. The target build also refreshed `libWireCellUtil.so` / `libWireCellAux.so` (14:25), which
  carry the same uncommitted peer edits (`util/src/LassoModel.cxx`, `aux/src/BlobShadow.cxx`) the owner's 13:15
  install already had; `libWireCellImg.so` untouched (13:15).
- **Smoke test, shared install (bare runner, no pin), tag `_d120prod`:** all six events rc 0; `mabc-pr.zip`
  member hashes equal arm B's `_d120beam10` on the three reconstructed events (157312 `f87a45ab…`, 245576
  `eadbb007…`, 317673 `4df742db…`, 13 members each) and every `CheckBeamParticle:` verdict line is identical on
  all six. The doctest binary in the shared build links against `local/lib`, so build it AFTER an install
  (doc 119's trap).

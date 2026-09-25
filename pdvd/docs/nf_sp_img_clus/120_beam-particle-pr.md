# doc 120 — Beam-particle pattern recognition on the beam-flash-matched bundle (`CheckBeamParticle`)

**Status (2026-09-25):** implemented, gated, run on the 6 doc-118 kaon events. New component `CheckBeamParticle`
(clus) + `run_pr_evt.sh -beam`. Byte-identical for every existing pipeline (new component, null-default TLAs; the
`-stm` / `-nu` / `-nu-legacy` compiled configs are unchanged; hash gate d120a/d120b PASS). **Owner ruling
2026-09-25: use the DATA-derived entry point (110, 159, 0.6) cm — not the GDML axis x — and KEEP the 10 cm
main-cluster length floor.** Both are the shipped defaults (sec 1.3, sec 5.2); nothing moves. **Installed in the shared tree 2026-09-25 14:25 (owner's go)** and smoke-tested: the
bare runner reproduces arm B on all six events (sec 8). **Bee set uploaded (owner's go):
<https://www.phy.bnl.gov/twister/bee/set/68caddae-7c7a-45b6-9312-54ddfdd5fe6d/event/list/>** (sec 6). Toolkit
`57f99628`, wcp `59e42b14` (+ this record).

**Round 2 (2026-09-25, sec 9) — owner's report: the particle flow misses pieces the fitted trajectories cover.**
Three causes found and fixed, all inside the beam stage or behind a default-off knob: (1) the stage read only the
79-key STM+Michel knob subset while the neutrino stage runs with ~250 production keys (now the neutrino node's
compiled data, one config two consumers); (2) the entry-rooted "long muon" pseudo-shower flood-filled the whole
event into one `mu-` node (now released to ordinary track segments, `entry_long_muon_absorb=false`); (3) every EM
shower's charge energy was 0 on PDVD — the 2-D charge cells are tested in the raw drift frame against t0-corrected
clouds, ~50 cm apart — so the Bee tree's energy floor hid them (`kine_charge_t0_frame`, a new
`KineChargeOptions` knob, default off everywhere but this stage). Bee nodes on 157312 / 245576 / 317673: 8 / 13 / 3
→ 22 / 14 / 14. Knob-off gates: `-nu` production and `-nu-legacy` byte-identical (sec 9.5); `-stm/-nu/-nu-legacy`
compiled configs unchanged; wcdoctest-clus 716888/716888. Shared tree re-installed 15:03. Round-2 Bee zip built
(`/home/xqian/tmp/d120_bee/d120-beam6-r2.zip`, tag `_d120r4`), **not uploaded** (owner's call).

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
# round 2 (sec 9): same command, tag d120r4, shared install of 15:03; WCT_BEE_PF_PRINT=1 prints the tree assembly
WCT_BEE_PF_PRINT=1 PDVD_BEAM_LIGHT_SUFFIX=_d119beam PDVD_PR_TLA="-S dl_weights=''" ./run_pr_evt.sh -beam -s d120r4 39305 157312
python3 kaon/make_d120_bee_zip.py d120r4 /home/xqian/tmp/d120_bee/d120-beam6-r2.zip
# the neutrino chain on the same bundle, for comparison (PDVD -nu-legacy gates candidates on the STM tag by default)
PDVD_PR_TLA="-S dl_weights='' -S beam_trigger_us=2793.584 -S beam_tc_type=22 -S nu_per_bundle_stm_only=false" \
  ./run_pr_evt.sh -nu-legacy -s d120nulegacy2 39305 157312
# knob-off gate of the shared NeutrinoEnergyReco change: /home/xqian/tmp/d120_r4cmp.sh (sec 9.5)
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

**Owner ruling (2026-09-25): the data-derived entry point stands.** The defaults above are production; the GDML
value is kept here as the record of the discrepancy, not as an alternative operating point.

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
nearest). This is a selection operating point for a new stage, not a physics retune; `-S 'beam_pr_knobs={}'`
restores the bare rule (Bee set `d120-beam6-nofloor.zip`). **Owner ruling (2026-09-25): keep the 10 cm floor.**

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

1. ~~Entry x: GDML 210.6 vs data ≈110 cm~~ — RULED 2026-09-25: the data-derived entry (sec 1.3). Why the GDML
   plug axis sits 100 cm higher in the drift coordinate is still an open question for the detector group, not for
   this chain.
2. `improve_vertex` on a boundary entry is off (`improve_entry_vertex`); on, it would re-fit the entry position
   from the segments — measure before turning on.
3. `T_tagger` rows carry only `match_isFC`; every BDT feature is at its `init_tagger_info` default (cosmic_flag
   1, scores 0). Readers must not interpret them.
4. 245576's bundle holds three sizeable clusters; the two 175 / 178 MeV muon-typed pieces are companions in the
   flow, not separate candidates. A per-cluster mode ("run each as its own candidate") was discussed and not built.
5. ~~The length floor (sec 5.2) is a selection operating point chosen on one event~~ — RULED 2026-09-25: keep
   the 10 cm floor. The candidate table is in the log of every run should a distance-to-axis metric ever be
   wanted.
6. The doc-118 light dirs without the `_d119beam` suffix have no trigger metadata; a production run needs
   `run_light_evt.sh` with `PDVD_BEAM_LABEL=1` (the default since doc 119), after which `-beam` needs no suffix.
7. Timing: the stage takes 0.8–1.8 s per event on these top-only pctrees (steiner runs on the beam bundle only).
8. (round 2) The beam track is rendered as its chain of track segments (sec 9.3), not as one node; on 157312 the
   11 cm entry segment 30011 is typed `e-` once the charge energies are real (sec 9.4; `mu-` with them at 0). The
   beam particle's identity is known from the beamline — a PID lock on the entry chain is the natural next knob.
9. (round 2) `kine_charge_t0_frame` is on in this stage only. TaggerCheckNeutrino on PDVD (`-nu-legacy`) has the
   same zero-charge defect (sec 9.4, measured on its own run) and does not read the key; wiring it there is a
   3-line default-off addition if a PDVD neutrino-chain reference is ever wanted.
10. (round 2) The neutrino chain's main-vertex-specific passes (`shower_clustering_connecting_to_main_vertex`,
    the pi0 finders' vertex preference) act at the ENTRY; the interaction vertex 77 cm in is an ordinary graph
    vertex to them. A two-vertex design (beam track to the interaction, the neutrino chain from there) was
    considered and not built.

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

## 9. Round 2 — why the particle flow missed what the trajectories covered

Owner's report on the sec 6 Bee set: the fitted trajectories cover the image, the particle flow shows the entry
muon and little else. Measured on 157312 (tag `_d120beam10`, the sec 5 output): the `track_fit` layer carries 20
clusters of the bundle (30, 36, 155–172, 204), `mc.json` holds 8 nodes — `mu- 563 MeV` from the entry, two
`neutron → proton` pieces (76, 41 MeV) and an 8.5 MeV proton. The same bundle through the neutrino chain
(`-nu-legacy`, `nu_per_bundle_stm_only=false`, tag `_d120nulegacy2`, its own vertex at (98.8, 110.8, 58.7) cm)
gives 12 nodes: `mu- 207`, `e- 77`, `e- 24`, `proton 166`, `proton 87` and the same companions. Three causes,
found in that order.

### 9.1 The knob set (config)

The compiled `-beam` config gave `CheckBeamParticle` 48 keys; `TaggerCheckNeutrino` in the same job gets 246. The
stage had forked CheckSTM_Michel's knob reader (the 79-key PR-partition subset: kinks, two-end break, isochronous
endpoints, steiner penalty) and `pr.jsonnet` forwarded only that subset, so shower clustering, the cross-cluster
bridges and continuations (`shower_nv_bridge_track`, `straight_cont_cross_cluster`, `conn3_stitch_max`,
`kine_count_near_cross_cluster`, `kine_count_conn4_near`), the pi0 finders, the pass-4 ownership fixes and the
kine accounting all ran at their uBooNE-legacy C++ defaults.

Fix: `CheckBeamParticle` now carries TaggerCheckNeutrino's members, `configure()` reads and the
`pattern_algos.m_* = m_*` copy block VERBATIM — 344 keys generated from the TCN source by
`pdvd/kaon/gen_beam_knobs.py` (fork by duplication) — minus the families this stage never runs: DL/dual chain,
candidate selection (`nu_*`), cosmic taggers/BDT features, MCS, the main-vertex kink/junction snaps (the entry
is a geometric fact), the SBND cathode bridge and the probes. Same names, units and C++ defaults
(`TaggerCheckNeutrino.h`), and the doctest pins a sample of each family plus the absence of the dropped ones.
`pr.jsonnet` hands the stage the NEUTRINO NODE's compiled `data` filtered by that same drop list (one config, two
consumers: 198 keys on the PDVD job), `beam_pr_knobs` merging last. The chain also gained the neutrino chain's
post-vertex passes it had skipped: `main_vertex_graph_audit`, `stitch_disconnected_main_cluster`, the
`swap_orphan_dup_audit` sweep, `fit_blob_coverage_defer` and the `long_muon_range_empty_chain_fallback`
recompute (each inert unless its key is on). Effect alone (tag `_d120r2`): cluster 170 (52 cm, 6 segments) became
its own shower and the tree gained nothing visible — 8 nodes still — because of 9.3 and 9.4.

### 9.2 What the tree assembly saw (`WCT_BEE_PF_PRINT=1`)

Every companion shower WAS added to the tree (12 `ADD shower-leaf` lines on 157312) and then pruned by the Bee
producer's energy floors (`em_ke_min 0.2 MeV`, `np_ke_min 3 MeV`): every electron-typed piece had `ke=0`, and the
main cluster's own daughters were not separate showers at all — the entry `mu-` node had `nsegments=21`.

### 9.3 The entry-rooted "long muon" (algorithm, beam stage only)

`examine_direction` searches for a long muon FROM THE MAIN VERTEX (`NeutrinoVertexFinder.cxx:1896-1955`): every
segment leaving the vertex with MIP dQ/dx is walked with `find_cont_muon_segment` through each junction it can
continue across, and a chain > 45 cm with a > 35 cm member becomes `segments_in_long_muon`.
`shower_clustering_with_nv_in_main_cluster` then seeds ONE pseudo-shower on the first chain segment and
`Shower::complete_structure_with_start_segment` flood-fills it — with no stopping rule for a type-13 shower
(`PRShower.cxx:838`, the absorb guard exempts long muons on purpose). For a neutrino the chain is the exiting muon
and the flood-fill reassembles a broken track; rooted at the ENTRY it is the whole event: 157312's chain ran from
the entry through the interaction vertex 77 cm in (two protons, two EM showers, the outgoing muon) to the far end,
one node, `mu- 532 MeV, 11 segments` (r2).

Fix (`entry_long_muon_absorb`, C++ default false = release): after `examine_direction` the stage clears
`segments_in_long_muon` / `vertices_in_long_muon` (the PID-13 stamps and the cleared shower flags the search left on
the beam segments stay). The seeder's BFS then descends through the beam segments as tracks, the daughters at the
interaction vertex seed their own showers or stay tracks, and `fill_bee_pf_tree`'s track BFS gives beam segment(s)
→ daughters. Logged per event: `released the entry-rooted long-muon chain (N segment(s), …)` — 2 segments on
157312, 3 on 317673, none on 245576 (its main cluster is a 2-segment `pi+ → proton`). Effect (tag `_d120r3`):
157312 8 → 17 nodes (`mu- 51 → e- 24, mu- 181 → proton 161, mu- 5.65 → e- 25, pi+ 106, proton 89, …`), 317673
3 → 8 (`mu- 444 → e- 17, mu- 78 → e- 2, mu- 469 → e- 18`: the muon broken at its deltas). `true` restores the
neutrino-chain behaviour for comparison.

### 9.4 Zero charge energy on PDVD (shared code, default-off knob)

Every shower's `kine_charge` was 0.0 — in the beam stage AND in TaggerCheckNeutrino's own PDVD run (sec 9 head:
its `e- 77` / `e- 24` are range-valued). `kine_charge_from_maps` (`NeutrinoEnergyReco.cxx`) takes each 2-D charge
cell's geometric point from `Grouping::convert_time_wire_2Dpoint` — the raw t0 = 0 drift frame — and asks the
shower's `associate_points`/`fit` cloud (the cluster's t0-CORRECTED frame) for its nearest point within 0.6 cm.
Trace on 157312 (`PDVD_LOG_LEVEL=trace PDVD_LOG_LOGGERS=clus.NeutrinoPattern:trace`): `hits total=5226
within_cut=0`, every distance 37–60 cm, e.g. cell (ts 3820, apa 6 face 0) at drift 58.7 cm against cloud points
at x ≈ 111.5 cm. A beam neutrino has t0 ≈ 0 so uBooNE/SBND never see it; a PDVD beam particle's flash t0 is
~2.8 ms after the readout origin. doc pdhd/16 found the identical defect for the Michel charge and
CheckSTM_Michel fixed it locally (`michel_q2d`, `CheckSTM_Michel.cxx:1882-1946`): per segment and (apa, face),
median of `backward(point, t0).x − point.x` over its fit points, subtracted from the cell's drift.

Fix: `KineChargeOptions::t0_frame` (`NeutrinoPatternBase.h`; config key `kine_charge_t0_frame`, C++ default
false). `NeutrinoEnergyReco.cxx` gains `kine_t0_shift()` (that rule over an object's segments, int-pair-keyed,
median of sorted vectors) and every consumer — `cal_kine_charge` (shower and segment overloads),
`calculate_shower_kinematics`, the pr/99 dedup scan's per-context clouds and its rebuild-only path — passes the
shift when the knob is on and a null pointer otherwise, so the off path is the legacy call. Cells of an (apa,
face) the object has no fit point in are skipped, as in the Michel rule. `CheckBeamParticle` reads its own key
with default TRUE (PDVD-only, new stage); TaggerCheckNeutrino does not read it (open item 9). Effect (tag
`_d120r4`): 157312 17 → 22 nodes, the 51 cm cluster-170 piece `e- 142 MeV`, `30008 e- 181 MeV`, Enu 443 → 492;
245576 10 → 14 (`e- 566 MeV` for the 140 cm shower that was `0.00 MeV`, Enu 820 → 1404); 317673 8 → 14, Enu 1165
→ 1181. Two proton pieces changed `kine_best` from range to charge-consistent values; and the entry segment on
157312 is now typed `e-` (open item 8).

### 9.5 Gates

- Compiled configs (`/home/xqian/tmp/d120_cfgdiff2.sh`, `/home/xqian/tmp/d120cfg2/`): `-stm`, `-nu`, `-nu-legacy`
  byte-identical to the pre-doc-120 reference (274364 / 279325 / 285167 B); `-beam` node 48 → 198 keys.
- Knob-off, shared `NeutrinoEnergyReco.cxx` change (`/home/xqian/tmp/d120_r4cmp.sh`): PDVD production `-nu`
  (CheckSTM_Michel) 157312 / 317673, tags `_d120b` (pre-change) vs `_d120nu4` (15:05 install): `mabc-pr.zip`
  member hashes `41d05530…` / `3f1701e1…` identical, `calib-pr` json identical, `tracking-pr.root` 280/280
  branches identical; `-nu-legacy` (TaggerCheckNeutrino, the one consumer that calls the changed functions
  with the knob off) `_d120nulegacy2` vs `_d120nulegacy4`: zips `c80a916d…` / `2590878d…` identical, calib
  identical, 1315/1315 branches identical. uBooNE/SBND: same functions, same null-pointer path (structural).
- `wcdoctest-clus` 716888/716888 after the final build (defaults doctest extended: 20 kept keys pinned, 9
  dropped families asserted absent, the two new stage keys).
- Freshness: `local/lib/libWireCellClus.so` 15:03:46 vs `CheckBeamParticle.cxx` / `NeutrinoEnergyReco.cxx`
  15:02; all six events rerun on the shared install (`_d120r4`), rc 0; 191916 / 2001 / 36591 unchanged (entry
  cut / no bundle).

### 9.6 Per event, round 2 (tag `_d120r4`)

| event | main | tree (top level → children) | Enu MeV | nodes r1 → r4 |
|---|---|---|---|---|
| 157312 | 30 | `e- 36 (30011) → mu- 181 → {mu- 5.65, proton 161}`; `gamma 181 → e- 181 → {gamma 142 → e- 142 (cluster 170), neutron 41 → proton 41, proton 89}`; `neutron 76 → proton 76`; `proton 8.5`; 2 sub-MeV e- | 492 | 8 → 22 |
| 245576 | 12 | `pi+ 48 → proton 619 → gamma 15 → e- 15 → …`; `gamma 566 → e- 566 (140 cm)`; `neutron 171 → mu- 171` | 1404 | 13 → 14 |
| 317673 | 33 | `mu- 444 → {e- 17, mu- 78 → {e- 2.3, mu- 469 → e- 18}}`; `gamma 14 → e- 14`; `gamma 1.5 → e- 1.5` | 1181 | 3 → 14 |

Bee zip `/home/xqian/tmp/d120_bee/d120-beam6-r2.zip` (same six events, same order as sec 6) — built, not
uploaded.

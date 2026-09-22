# The per-event ROOT file: `stm_michel_<det>_<run6>_<evtid>.root`

Plain ROOT TTrees written with uproot.  Every tree carries `run` and `event` so files can be
chained (`uproot.concatenate("events/*/stm_michel_*.root:T_stm")`, or a `TChain`).
Units: lengths **cm**, times **µs**, charge **electrons** (after signal processing: no
recombination, no lifetime correction), dQ/dx **e/cm**, energies **MeV**.  Coordinates are
the detector frame (x drift, y up, z beam).

| tree | one row per | what |
|---|---|---|
| `T_event` | file | provenance and the per-event constants |
| `T_stm` | STM candidate cluster | the tagger's verdict and every scalar it computed (≈ 200 branches) |
| `T_stm_pts` | point of the tagger's chain | the trajectory the verdict was read on: x,y,z, dQ/dx, residual range, role |
| `T_stm_fit` | point of the PR fit of a candidate cluster | 3-D fit points with their 2-D projections and dQ/dx |
| `T_stm_2d` | fitted 2-D cell of a candidate cluster | (plane, wire, time-slice) → measured charge, error, fit prediction |
| `T_michel_2d` | 2-D cell in the Michel / stop region | the cells behind the Michel charge estimator, with roles and the muon-only prediction |
| `T_flash` | optical flash of the event | time, PE per photon detector, which clusters matched it |
| `T_ophit` | OpHit of a flash matched to an STM candidate | per-PD pulse list |
| `T_opwf` | (STM-matched flash, readout channel) | raw + deconvolved waveform window |

The joins: `T_stm.cluster_id` ↔ `T_stm_pts.cluster_id`, `T_stm_fit.cluster_id`,
`T_stm_2d.cluster_id`, `T_michel_2d.cluster_id`, and the Bee `cluster_id`;
`T_stm.flash_id` ↔ `T_flash.flash_id` ↔ `T_ophit.flash_id` ↔ `T_opwf.flash_id`.

## 1. `T_event`

| branch | meaning |
|---|---|
| `det`, `run`, `subrun`, `event`, `event_index` | detector, run, subrun, DAQ event number, position in the run's event list |
| `arm`, `toolkit_commit`, `wcp_commit`, `pr_dir`, `light_dir`, `frames_dir`, `wires` | provenance: the production arm and the directories / commits the file was built from; the wire-geometry file |
| `dQdx_scale`, `dQdx_offset` | how `T_stm_fit.q` encodes the fitted charge (`dQ = (q - offset) / scale`) |
| `nticks_per_slice` | time slices of the 2-D cells are this many 0.5 µs ticks (4) |
| `readout_window_ticks` | charge readout length in ticks |
| `light_offset_us` | PDHD: the readout-vs-trigger offset stamped in the flash archive |
| `trigger_offset_us`, `trigger_offset_bot_us`, `trigger_offset_top_us` | **add to a light-axis time to land on the charge (drift) time axis** (PDHD: one value; PDVD: per crate, bottom / top drift volume) |
| `opflash_offset_bot_us`, `opflash_offset_top_us` | PDVD: the raw per-crate offsets in the flash archive (the charge chain adds a per-run residual on top; the `trigger_offset_*` values are the ones it used) |
| `nchan`, `n_flash`, `n_candidates`, `n_stm`, `n_michel`, `n_stm_flash` | counts |
| `wf_pre_us`, `wf_post_us` | the waveform window: [flash − pre, flash + post] |

## 2. `T_stm` — the candidates

One row per cluster the tagger examined.  The selection branches:

| branch | meaning |
|---|---|
| `cluster_id` | the cluster (same id everywhere in this file and in Bee) |
| `is_stm` | 1 = accepted stopping muon (`reject_bits == 0`) |
| `reject_bits` | bit mask of failed tests; names in `stm_release.REJECT_BITS` (no_chain, stop_unmatched, no_bragg, shape_flat, not_muon_pid, continuation, stop_near_boundary, vertex_hadron, short, profile_sparse, plateau_off_mip, stop_into_dead, cluster_not_track, profile_geometry, readout_edge) |
| `topology_cleared_bits` | bits a Michel of sufficient quality was allowed to clear (owner rule P1; 0 when off) |
| `michel_found`, `michel_conn_type` | Michel object found; 1 attached, 2 bridged, 3 unfitted dots |
| `in_fv` | the stop is inside the fiducial volume |
| `gid`, `t0_us`, `flash_id`, `flash_time_us`, `flash_total_pe` | the matched flash: `t0_us` is the cluster's t0 (= the flash time on the light axis); `flash_id` is the row of `T_flash`; `gid` is the matcher's internal flash key (not a `T_flash` row) |
| `cluster_length_cm`, `cluster_npoints` | the cluster's extent and size |

Geometry of the chain (cm):

| branch | meaning |
|---|---|
| `entry_x/y/z`, `entry_vtx_id` | where the muon enters (the chain starts) |
| `stop_x/y/z`, `stop_vtx_id` | the reconstructed stop (Bragg end) |
| `tagger_stop_x/y/z`, `stop_dis` | the cosmic tagger's stop estimate and its distance to the chain's stop |
| `muon_len` | chain length entry → stop |
| `michel_start_x/y/z`, `michel_seg_id`, `michel_parent_vtx_id`, `michel_dis_cm` | where the Michel starts; the segment it is; distance from the stop |
| `michel_len`, `michel_far_len`, `michel_kink_deg`, `michel_mip` | the Michel seed segment's length, reach, kink angle at the stop, MIP-ness |
| `n_chain_segs`, `n_profile_pts`, `n_live_pts`, `n_dead_pts`, `n_end_pts`, `n_plateau`, `n_tail` | how many segments / profile points the chain has and how many are live / dead / in the plateau and tail windows |
| `n_delta`, `delta_len`, `n_ext`, `ext_len`, `n_retreat`, `retreat_len`, `n_split`, `split_len`, `split_kink_deg` | delta rays off the body, chain extensions, and the stop-topology moves (retreat / split) |
| `n_body_hadron`, `n_body_other`, `n_stop_arms`, `n_stop_other`, `cont_len`, `cont_angle_deg`, `cont_mip` | prongs off the body, arms at the stop, the longest continuation |
| `end_arc_cm`, `end_span_cm`, `end_arc_span`, `unsupported_len_cm`, `unsupported_frac_min`, `n_unsupported_segs`, `chain_coverage`, `chain_support_min` | profile-geometry diagnostics (coiled end; fitted length the charge does not support) |
| `dead_ahead`, `dead_frac_cmp`, `n_cmp_live` | dead-region checks ahead of the stop |

The dQ/dx shape tests:

| branch | meaning |
|---|---|
| `plateau_med`, `tail_med` | median dQ/dx on the plateau (40–60 cm from the stop, halved on short tracks) and in the tail [e/cm] |
| `contrast`, `contrast_expected`, `bragg_valid` | Bragg contrast (peak over plateau) measured / expected from the table |
| `bragg_anchor_shift_cm`, `bragg_anchor_fallback`, `bragg_wide_fired`, `bragg_wide_shift_cm` | where the Bragg peak was anchored |
| `ks_mu`, `ks_flat`, `ratio_mu`, `ratio_flat` | KS-like distances of the profile to the muon template and to a flat template, and the corresponding ratios |
| `comp_fwd0..3`, `comp_bwd0..3` | PID comparisons forward / backward (muon vs proton / electron / ...) |
| `short_track`, `kink_num`, `pass`, `has_pass` | the accepted tagger pass and its kink index |

Energies (MeV) and momenta (MeV/c):

| branch | meaning |
|---|---|
| `muon_ke_range` | from `muon_len` through the CSDA range table (muon) |
| `muon_ke_dqdx` | calorimetric: the fitted dQ/dx summed through the recombination model (carries the gain × lifetime × recombination normalisation) |
| `muon_ke_mcs`, `muon_mcs_amb`, `muon_mcs_nsegs`, `muon_mcs_tracklen`, `muon_mcs_range_ke`, `muon_mcs_bad_path`, `muon_mcs_cathode_segs`, `muon_mcs_cathode_angles` | multiple-Coulomb-scattering energy of the trajectory (purely geometric) and its fit details: ambiguity, segments, the length it was fitted on, the range KE of that length, quality flags |
| `muon_ke_best` | the chain's choice (range on a stopper) |
| `muon_p_range`, `muon_p_dqdx`, `muon_p_mcs` | the corresponding momenta |
| `michel_ke_q2d_region` | **the production Michel energy**: every own-cluster 2-D cell within 10 cm of the stop, measured charge minus the muon fit's prediction, the three planes combined (0.25 / 0.25 / 1.0 weights, plane dropped on a 4 % asymmetry), through the recombination model. Per-plane charges `michel_q2d_region_{u,v,w}` [e], the muon predictions `michel_q2d_region_mu_*`, the cell counts `michel_q2d_region_n_*` / `_nd_*` (dead), `michel_q2d_region_dropped_plane` |
| `michel_ke_q2d_ctl`, `michel_q2d_ctl_*`, `michel_q2d_ctl_valid` | the **body control**: the same sum centred 35 cm back up the muon; what a region with no Michel reads |
| `michel_ke_q2d`, `michel_q2d_{u,v,w}`, `michel_q2d_mu_*`, `michel_q2d_raw_*`, `michel_q2d_n_*`, `michel_q2d_nx_*`, `michel_q2d_gamma`, `michel_ke_q2d_gamma`, `michel_ke_q2d_total`, `michel_q2d_valid`, `michel_q2d_reason`, `michel_q2d_dropped_plane`, `michel_q2d_n_role0`, `michel_q2d_n_rewired` | the association-based 2-D estimate (cells claimed by the Michel segments) and its gamma term |
| `michel_ke_best` | the chain's association estimate: fitted dQ/dx along the Michel path + the unfitted charge (`dots_*`) |
| `michel_ke_core`, `michel_ke_dqdx`, `michel_ke_charge`, `michel_ke_range`, `michel_ke_gamma`, `michel_ke_total` | the Michel core alone, the whole object by dQ/dx, the charge-based estimate, the range estimate, the gamma term, core + gammas |
| `michel_n_pieces`, `michel_n_clusters`, `michel_n_segs`... `n_michel_gammas`, `n_michel_gamma_cand`, `n_michel_gamma_capped`, `n_michel_veto*` | how the Michel object was assembled |
| `n_dots`, `n_dot_clusters_unfit`, `dots_charge_unfit`, `dots_ke_dqdx`, `dots_ke_unfit` | unfitted dot clusters near the stop and their charge / energy |
| `n_stop_gammas`, `stop_gamma_*` | capture-gamma candidates at the stop (µ⁻ capture) |

Everything else in the tree is a diagnostic counter of the tagger; the full list is in the
appendix and the source is the toolkit's `clus/src/CheckSTM_Michel.cxx` (the block that
fills the `stm_michel` point cloud).

## 3. `T_stm_pts` — the chain trajectory with dQ/dx

| branch | meaning |
|---|---|
| `cluster_id`, `run`, `event`, `is_stm`, `michel_found` | the candidate this point belongs to (verdict copied for convenience) |
| `x`, `y`, `z` | position [cm] |
| `q` | **dQ/dx at this point [e/cm]** |
| `L` | distance along the chain from the entry [cm] |
| `rr` | **residual range** = distance along the chain to the stop [cm]; −1 on rows that are not on the muon chain |
| `role` | 1 muon chain, 2 delta ray, 3 Michel, 4 gamma / dot attached to the Michel, 5 unfitted dot (see `stm_release.ROLE`) |
| `seg_id` | the segment: `cluster_id * 1000 + graph index` |
| `q_sup` | the owning segment's median dQ/dx over the chain's plateau median (charge support; −1 when no plateau) |

`dQ/dx vs rr` for the muon is `q` vs `rr` on `role == 1` rows (drop `rr < 0` and `q <= 0`,
which are vertex rows and dead-region points).  A vertex shared by two segments is written
once per segment.

## 4. `T_stm_fit` — the PR fit of the candidate cluster

The pattern-recognition fit of the whole candidate cluster (every segment, not only the
chain), with its projection into the three wire planes — the bridge between the 3-D
trajectory and the 2-D cells of `T_stm_2d`.

| branch | meaning |
|---|---|
| `x`, `y`, `z` | fitted point [cm] |
| `q`, `nq`, `dQ`, `dx`, `dqdx` | `q` encodes the fitted charge (`dQ = (q - dQdx_offset) / dQdx_scale` electrons), `nq` = `dx` the step [cm], `dqdx = dQ / dx` [e/cm] (−1 where dx = 0) |
| `pu`, `pv`, `pw` | the point projected onto the U / V / W plane, in the **global wire-rank coordinate** of `T_stm_2d.channel_rank_global` (fractional; subtract the plane base to get `chan_rank`, see §5) |
| `pt` | the point's time in **time slices** (× `nticks_per_slice` = ticks) |
| `rr` | residual range along its segment [cm]; −1 at a shared vertex |
| `flag_vertex`, `flag_shower` | 1 on vertex rows; 1 on shower-like segments |
| `cluster_id`, `real_cluster_id`, `sub_cluster_id`, `particle_id`, `ndf` | the cluster; `sub_cluster_id` = `cluster_id * 1000 + segment index` (joins `T_stm_pts.seg_id`); `particle_id` = PDG code of the segment; `ndf` repeats the cluster id (legacy) |
| `chi2`, `reduced_chi2` | fit quality |

## 5. `T_stm_2d` — the fitted 2-D charge cells of the candidate cluster

One row per (plane, wire, time-slice) cell the track fit used for the candidate cluster —
the **2-D charge measurement after signal processing**, merged over the fit's snapshots.

| branch | meaning |
|---|---|
| `cluster_id` | the candidate |
| `plane` | 0 U, 1 V, 2 W (collection) |
| `chan_rank` | the wire coordinate in this plane: the rank of the physical channel among all channels of that plane over the whole detector (PDHD: 0..3199 U/V, 0..3839 W; PDVD: 0..3807 / 0..4671) |
| `channel_rank_global` | `chan_rank` + plane base (`[0, nch_U, nch_U + nch_V]`) — the axis `T_stm_fit.pu/pv/pw` are on |
| `channel`, `apa`, `face`, `wire` | the physical channel id and the (APA, face, wire index) it reads (the first segment of a wrapped channel) |
| `time_slice`, `tick` | the time slice and its first tick (`tick = time_slice * nticks_per_slice`; 1 tick = 0.5 µs) |
| `charge`, `charge_err` | measured charge of the cell [e] and its uncertainty |
| `charge_pred` | the fit's predicted charge in the cell [e] |

## 6. `T_michel_2d` — the cells behind the Michel estimator

The per-cell table the tagger built for the Michel energy: every cell within the region
radius of the stop, the Michel's and gammas' own cells, and the muon's footprint near the
stop, with the muon-only and the full fit prediction.

| branch | meaning |
|---|---|
| `cluster_id`, `apa`, `face`, `plane`, `wire`, `channel` | where the cell is; `wire` is the wire index within (apa, face) — join to `T_stm_2d` through `channel` |
| `time`, `time_slice` | the tick and the time slice |
| `charge`, `charge_err`, `flag` | measured charge [e], its error, live (1) / dead-region filler (0) |
| `pred_mu` | the **muon-only** fit prediction in the cell [e] |
| `pred_all` | the full fit prediction (muon + Michel + everything fitted) [e] |
| `role` | 3 Michel (within the Michel cloud, or predicted / unfitted Michel charge), 4 gamma, 1 the muon's own footprint near the stop, 0 a cell inside the region (or the control) that no selection claimed |
| `sel` | bit mask: 1 the cell lies within the Michel's (or a gamma's) fitted point cloud, 2 the Michel part of the fit predicts charge here, 4 the cell belongs to an unfitted Michel piece |
| `shared`, `xshared` | the cell is shared by several fitted segments; `xshared` = shared with another cluster (its measurement cannot be attributed) |
| `own_blob` | bit 1: the cell is covered by the candidate's own cluster; bit 2: by an admitted unfitted companion cluster; bit 4: by a fitted companion |
| `d_stop_cm`, `d_ctl_cm` | 2-D distance of the cell to the stop and to the body-control centre [cm]; −1 when unprojectable |

The production estimator is reproduced from this table by
`stm_release.michel_region_sum(ev, cluster_id)`: cells with `0 <= d_stop_cm <= 10` and
`own_blob != 0`, contribution `charge - max(pred_mu, 0)`, or `max(pred_all - pred_mu, 0)` on
`xshared` cells; the per-plane sums equal `T_stm.michel_q2d_region_{u,v,w}` to 1e-6
(checked on every candidate when the file was built).

## 7. `T_flash`, `T_ophit`, `T_opwf` — the light

See [LIGHT.md](LIGHT.md).

## 8. `dqdx_ref.json`

The reconstruction's own expectation of dQ/dx vs residual range (Modified-Box recombination
at the detector field, × 0.85), for `muon`, `electron`, `pion`, `kaon`, `proton`: grid
`start + step * i` cm (0–100 cm in 0.25 cm), values in e/cm.  `stm_release.dqdx_ref()`
returns it as arrays; beyond 100 cm hold the last value (minimum ionising).

## 9. Derived columns (everything else is copied verbatim)

`T_stm.flash_id / flash_time_us / flash_total_pe / cluster_length_cm / cluster_npoints`
(joined from the PR stage's cluster table and the flash archive); `T_stm_pts.is_stm /
michel_found` (joined); `T_stm_fit.dQ / dx / dqdx` (from `q`, `nq`); `T_stm_2d.plane /
chan_rank / channel / apa / face / wire / tick` (decoded from the global rank with the wire
geometry file); `T_flash.time_charge_*` (time + trigger offset), `n_matched_clusters`,
`is_stm_flash`, `matched_cluster_ids`, `stm_cluster_ids`; all of `T_opwf` (windows cut from
the dumped frames); the `run` / `event` columns.

## Appendix: every `T_stm` branch

```
bragg_anchor_fallback               bragg_anchor_shift_cm               bragg_valid                         bragg_wide_fired                    bragg_wide_shift_cm                 chain_coverage
chain_support_min                   cluster_id                          comp_bwd0                           comp_bwd1                           comp_bwd2                           comp_bwd3
comp_fwd0                           comp_fwd1                           comp_fwd2                           comp_fwd3                           cont_angle_deg                      cont_len
cont_mip                            contrast                            contrast_expected                   dead_ahead                          dead_frac_cmp                       delta_len
dots_charge_unfit                   dots_ke_dqdx                        dots_ke_unfit                       end_arc_cm                          end_arc_span                        end_span_cm
entry_vtx_id                        entry_x                             entry_y                             entry_z                             ext_len                             gid
has_pass                            in_fv                               is_stm                              kink_num                            ks_flat                             ks_mu
michel_conn_type                    michel_dis_cm                       michel_far_len                      michel_found                        michel_gamma_dis_max                michel_ke_best
michel_ke_charge                    michel_ke_core                      michel_ke_dqdx                      michel_ke_gamma                     michel_ke_q2d                       michel_ke_q2d_ctl
michel_ke_q2d_gamma                 michel_ke_q2d_region                michel_ke_q2d_total                 michel_ke_range                     michel_ke_total                     michel_kink_deg
michel_len                          michel_mip                          michel_n_clusters                   michel_n_pieces                     michel_near_arm                     michel_parent_vtx_id
michel_q2d                          michel_q2d_ctl                      michel_q2d_ctl_dropped_plane        michel_q2d_ctl_n_u                  michel_q2d_ctl_n_v                  michel_q2d_ctl_n_w
michel_q2d_ctl_u                    michel_q2d_ctl_v                    michel_q2d_ctl_valid                michel_q2d_ctl_w                    michel_q2d_dropped_plane            michel_q2d_gamma
michel_q2d_mu_u                     michel_q2d_mu_v                     michel_q2d_mu_w                     michel_q2d_n_role0                  michel_q2d_n_u                      michel_q2d_n_v
michel_q2d_n_w                      michel_q2d_nx_u                     michel_q2d_nx_v                     michel_q2d_nx_w                     michel_q2d_raw_u                    michel_q2d_raw_v
michel_q2d_raw_w                    michel_q2d_reason                   michel_q2d_region                   michel_q2d_region_dropped_plane     michel_q2d_region_mu_u              michel_q2d_region_mu_v
michel_q2d_region_mu_w              michel_q2d_region_n_u               michel_q2d_region_n_v               michel_q2d_region_n_w               michel_q2d_region_nd_u              michel_q2d_region_nd_v
michel_q2d_region_nd_w              michel_q2d_region_u                 michel_q2d_region_v                 michel_q2d_region_w                 michel_q2d_u                        michel_q2d_v
michel_q2d_valid                    michel_q2d_w                        michel_seg_id                       michel_start_x                      michel_start_y                      michel_start_z
muon_ke_best                        muon_ke_dqdx                        muon_ke_mcs                         muon_ke_range                       muon_len                            muon_mcs_amb
muon_mcs_bad_path                   muon_mcs_cathode_angles             muon_mcs_cathode_segs               muon_mcs_nsegs                      muon_mcs_range_ke                   muon_mcs_tracklen
muon_p_dqdx                         muon_p_mcs                          muon_p_range                        n_body_hadron                       n_body_other                        n_chain_segs
n_cluster_pts                       n_cmp_live                          n_dead_pts                          n_delta                             n_dot_clusters_unfit                n_dots
n_end_pts                           n_ext                               n_floored_near_stop                 n_kept_near_stop_comp               n_kept_near_stop_main               n_live_pts
n_local_pieces                      n_michel_gamma_cand                 n_michel_gamma_capped               n_michel_gammas                     n_michel_range_veto                 n_michel_segs
n_michel_veto                       n_michel_veto_exempt                n_michel_veto_reach_exempt          n_near_arms_examined                n_other_published                   n_plateau
n_profile_pts                       n_retreat                           n_split                             n_stop_arms                         n_stop_gammas                       n_stop_gammas_withheld
n_stop_other                        n_stub_absorb                       n_tail                              n_unsupported_segs                  near_arm_dist_cm                    pass
plateau_med                         ratio_flat                          ratio_mu                            reject_bits                         retreat_len                         short_track
split_kink_deg                      split_len                           stop_dis                            stop_gamma_charge                   stop_gamma_dis_max                  stop_gamma_dis_min
stop_gamma_ke_max                   stop_gamma_ke_tot                   stop_gamma_n_unfit                  stop_gamma_seg_id                   stop_move_p3_bits                   stop_snap_skipped
stop_vtx_id                         stop_x                              stop_y                              stop_z                              t0_us                               tagger_stop_x
tagger_stop_y                       tagger_stop_z                       tail_med                            topology_cleared_bits               unsupported_frac_min                unsupported_len_cm
run                                 event                               flash_id                            flash_time_us                       flash_total_pe                      cluster_length_cm
cluster_npoints
```

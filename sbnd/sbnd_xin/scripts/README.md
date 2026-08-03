# `sbnd_xin/scripts/` — what lives where

Reorganized 2026-08-03 (see `docs/work-tags.md`, "TIDY ROUND 2026-08-03").
127 scripts moved out of the top level; the map that drove it is
`scripts/retire/tidy_map_20260803.tsv`.

## Still at the top level — the interface

Run these from `sbnd_xin/`. They were deliberately **not** moved: they
resolve their own root as `$(dirname $0)`, they are named in 100+ doc
citations, and three of them invoke the pinned helpers below.

- **runners** (13): `_runlib.sh`, `run_bee_img_evt.sh`, `run_clus_evt.sh`, `run_clust_QL_evt.sh`, `run_full1k_nusel.sh`, `run_img_evt.sh`, `run_nusel_evt.sh`, `run_pr_chain_batch.sh`, `run_pr_evt.sh`, `run_ql_evt.sh`, `run_reco1_dump.sh`, `run_select_evt.sh`, `run_sp_to_magnify_evt.sh`
- **job configs** (12): `cathode_fiducial.jsonnet`, `clus.jsonnet`, `magnify-sinks.jsonnet`, `ql_dump_scalar.jsonnet`, `qlmatching.jsonnet`, `wct-clus-matching-perevt.jsonnet`, `wct-clus-matching-standalone.jsonnet`, `wct-clustering.jsonnet`, `wct-img-all.jsonnet`, `wct-pr-perevt.jsonnet`, `wct-reco1-dump.jsonnet`, `wct-sp-to-magnify.jsonnet`
- **cross-campaign tools** (5): `merge_sel_archives.py`, `nusel_extract.py`, `pr_scores_table.py`, `relink_tags.py`, `wct-img-2-bee.py`
- **pinned / symlinks** (6): `Magnify-tracking-SBND`, `Woodpecker`, `input_files`, `sbnd_track_fitting.json`, `trash-all-apa.tar.gz`, `upload-to-bee.sh`

> `upload-to-bee.sh`, `wct-img-2-bee.py` and `merge_sel_archives.py` belong
> in `scripts/bee/` by topic but are **invoked by runners that stay**, so
> moving them would mean editing production runners. Pinned here instead.

## Moved

Invoke from `sbnd_xin/` as `python3 scripts/<sub>/<name>` — the scripts
resolve paths relative to the working directory or to `sbnd_xin` itself,
not to their own location.

### `products/` — loose data products produced by the analysis scripts

`flash_dump.csv`, `pmt_nonlin_params_v3r1.csv`, `pr20off.txt`, `pr20on.txt`, `ql_light_dump.csv`, `ql_perpmt_dump.csv`, `saturation_flash_dump.csv`, `saturation_perchan.csv`, `upload-d55ton-30evt-stmfit.stmid-map.txt`

### `scripts/analysis/cathode/` — cathode crossing: distortion map, nu census, kink probe, rescue sweeps

`scripts/analysis/cathode/cathode_distortion.py`, `scripts/analysis/cathode/cathode_nu_census.py`, `scripts/analysis/cathode/cathode_plots.py`, `scripts/analysis/cathode/cbr_sweep_compare.py`, `scripts/analysis/cathode/kink_probe.py`

### `scripts/analysis/geom/` — geometry / projection: magnify alignment, Y-Z coverage, TGM readout cut

`scripts/analysis/geom/pr_proj_align.py`, `scripts/analysis/geom/tgm_readout_cut.py`, `scripts/analysis/geom/yz_coverage.py`

### `scripts/analysis/light/` — light reco: flash coincidence/t0, PMT non-linearity + health, saturation

`scripts/analysis/light/flash_coincidence.py`, `scripts/analysis/light/flash_t0_lan_reco2.py`, `scripts/analysis/light/pmt_health_study.py`, `scripts/analysis/light/pmt_nonlinearity_curve.py`, `scripts/analysis/light/saturation_pe.py`

### `scripts/analysis/misc/` — cross-cutting probes that do not belong to one campaign

`scripts/analysis/misc/build_mcbase_stage.py`, `scripts/analysis/misc/compare_pr_roundtrip.py`, `scripts/analysis/misc/fc_speck_audit.py`, `scripts/analysis/misc/feat_census.py`, `scripts/analysis/misc/gapjump_probe.py`, `scripts/analysis/misc/inspect_pctree.py`, `scripts/analysis/misc/iso_sweep_compare.py`, `scripts/analysis/misc/mabc_step_totals.py`, `scripts/analysis/misc/nuecc48_detail.py`, `scripts/analysis/misc/nusel_scan_filter.py`, `scripts/analysis/misc/oc19_census_mcp1k.py`, `scripts/analysis/misc/oc19_census_nuecc48.py`, `scripts/analysis/misc/oc_stage_gap.py`, `scripts/analysis/misc/oc_stage_trace.py`, `scripts/analysis/misc/pair_eyeball.py`, `scripts/analysis/misc/pr_arm_compare.py`, `scripts/analysis/misc/pr_stage_totals.py`, `scripts/analysis/misc/ssm_tagger_ab.py`, `scripts/analysis/misc/stub_census.py`, `scripts/analysis/misc/tagger_tree_ab.py`, `scripts/analysis/misc/unmerge_crosser_audit.py`, `scripts/analysis/misc/vveto_sweep_compare.py`

### `scripts/analysis/pr11/` — doc pr/11 — 1071-event PR-chain population census + crash-fix arms

`scripts/analysis/pr11/pr11_analyze_census.py`, `scripts/analysis/pr11/pr11_br_filled_census.py`, `scripts/analysis/pr11/pr11_pick_events.py`, `scripts/analysis/pr11/pr11_seq_compare.py`

### `scripts/analysis/pr20/` — doc pr/20 — split cosmics at the cathode, demoted mains (Parts I/II)

`scripts/analysis/pr20/pr20_b03_census.py`, `scripts/analysis/pr20/pr20_b03_survivors.py`, `scripts/analysis/pr20/pr20_b1_population.py`, `scripts/analysis/pr20/pr20_census.py`, `scripts/analysis/pr20/pr20_edge_census.py`, `scripts/analysis/pr20/pr20_partI_census.py`, `scripts/analysis/pr20/pr20_partI_pftree.py`, `scripts/analysis/pr20/pr20_s7_crossers.py`, `scripts/analysis/pr20/pr20_scores_diff.py`, `scripts/analysis/pr20/pr20_wasmain_check.py`

### `scripts/analysis/pr23/` — doc pr/23 — protect-over-clustering PR stage

`scripts/analysis/pr23/pr23_cathprobe.py`, `scripts/analysis/pr23/pr23_fitcover_census.py`

### `scripts/analysis/pr24/` — doc pr/24 — isochronous EM shower trunk

`scripts/analysis/pr24/pr24_iso_probe.py`

### `scripts/analysis/pr25/` — doc pr/25 — cathode re-join / TGM veto / shower-topology

`scripts/analysis/pr25/pr25_rejoin_census.py`, `scripts/analysis/pr25/pr25_shower_topo_census.py`, `scripts/analysis/pr25/pr25_spread_census.py`

### `scripts/analysis/ql/` — Q/L matching: PE error, beam preference, prefilter, LM tuning, recipes

`scripts/analysis/ql/automask_prototype.py`, `scripts/analysis/ql/check_dead5.py`, `scripts/analysis/ql/lm_tune.py`, `scripts/analysis/ql/ql_arm_compare.py`, `scripts/analysis/ql/ql_beam_pref_score.py`, `scripts/analysis/ql/ql_beam_pref_tune.py`, `scripts/analysis/ql/ql_light_compare.py`, `scripts/analysis/ql/ql_nonlin_compare.py`, `scripts/analysis/ql/ql_pe_error.py`, `scripts/analysis/ql/ql_prefilter_parity.py`, `scripts/analysis/ql/ql_prefilter_tune.py`, `scripts/analysis/ql/ql_recipe_compare.py`, `scripts/analysis/ql/unmatched_census.py`

### `scripts/analysis/stm/` — STM track fit: censuses, A/B reports, dQ/dx reference, doc 52/60/66 arms

`scripts/analysis/stm/bw_label_census.py`, `scripts/analysis/stm/bwgate_report.py`, `scripts/analysis/stm/d52_ab_report.py`, `scripts/analysis/stm/d60_ab_report.py`, `scripts/analysis/stm/d66_flip_report.py`, `scripts/analysis/stm/d66_proton_sweep.py`, `scripts/analysis/stm/d66_scan_score.py`, `scripts/analysis/stm/stm_dqdx_reference.py`, `scripts/analysis/stm/stm_fv_census.py`, `scripts/analysis/stm/stm_main_connectivity.py`, `scripts/analysis/stm/stm_merge_attribution.py`, `scripts/analysis/stm/stmfit_mc_compare.py`, `scripts/analysis/stm/stmfit_particle_overlay.py`, `scripts/analysis/stm/stmfit_showcase.py`, `scripts/analysis/stm/stmon_stats.py`

### `scripts/bee/` — Bee set builders and upload helpers

`scripts/bee/bee_frame_probe.py`, `scripts/bee/make_pr_bee.py`, `scripts/bee/make_scan_bee.sh`, `scripts/bee/make_stmfit_bee.py`, `scripts/bee/merge_mabc_bee.py`, `scripts/bee/unzip.pl`, `scripts/bee/zip-upload.sh`

### `scripts/cfg/` — compiled-config helpers (wcsonnet wrappers)

`scripts/cfg/compile_prjob_cfg.sh`, `scripts/cfg/compile_sbnd_prod.sh`

### `scripts/perf/` — profiling and perf A/B reporting

`scripts/perf/clustering_timing_analysis.py`, `scripts/perf/p54_ab_report.py`, `scripts/perf/pr11_config_cost.py`, `scripts/perf/profile_pr11.sh`, `scripts/perf/profile_pr65.sh`

### `scripts/root/` — ROOT C macros

`scripts/root/dump_stopping_dqdx.C`, `scripts/root/dump_truth_sed.C`, `scripts/root/mc_find_rse.C`, `scripts/root/mc_nu_vertices.C`, `scripts/root/mc_truth_muons.C`, `scripts/root/pr_proj_binctl.C`, `scripts/root/pr_proj_guishot.C`

### `scripts/runners/` — campaign and legacy shell drivers (NOT the daily interface)

`scripts/runners/d66_flip_plots.sh`, `scripts/runners/geom_ab_batch.sh`, `scripts/runners/geom_ab_summary.sh`, `scripts/runners/run_d52_campaign.sh`, `scripts/runners/run_mcbase.sh`, `scripts/runners/run_perf54_nusel.sh`, `scripts/runners/run_pr_geom_arm.sh`, `scripts/runners/run_pr_geom_arm_dl.sh`, `scripts/runners/s4_chain.sh`, `scripts/runners/s4_nuecc48.sh`, `scripts/runners/s4_off.sh`, `scripts/runners/tmp_run_pr_chain_cfghead.sh`, `scripts/runners/tmp_run_pr_chain_geovtx.sh`, `scripts/runners/tmp_run_pr_chain_nblhead.sh`


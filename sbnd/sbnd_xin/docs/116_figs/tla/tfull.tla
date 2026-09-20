retile_sampler_strategy='charge_stepped'
steiner_blank_plane_mode='prefer3'
steiner_base_weight_blank_alpha=0.5
steiner_base_weight_scope='tree+path'
# tfull = csp3bw + the two lattice fit keys (fit_weight_pow 1.5, assoc_cont_center 1).  Those keys are
# NOT jsonnet: they live in the TrackFitting JSON read at runtime, reached through SBND_TRACKFIT_JSON
# (run_pr_chain_batch.sh:134-136 -> --tla-str trackfitting_config=...).  stageB_cell.sh exports it from
# the sibling file tfull.tfjson.  Without it this file is byte-identical to csp3bw.tla (doc pr/150 trap).

# doc 117 cell t1e -- the PDHD/PDVD trajectory (doc 116 `tfull`) + ONE-step vertex chain,
# fit exclusion ON.  The four trajectory keys are byte-identical to docs/116_figs/tla/tfull.tla;
# the two lattice fit keys of tfull (fit_weight_pow 1.5, assoc_cont_center 1) are NOT jsonnet and
# arrive through the sibling t1e.tfjson -> SBND_TRACKFIT_JSON (doc pr/150's trap, doc 116 sec 2).
retile_sampler_strategy='charge_stepped'
steiner_blank_plane_mode='prefer3'
steiner_base_weight_blank_alpha=0.5
steiner_base_weight_scope='tree+path'
dl_vtx_dual_chain=false

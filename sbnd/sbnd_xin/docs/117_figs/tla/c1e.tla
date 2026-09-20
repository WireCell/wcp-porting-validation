# doc 117 cell c1e -- CURRENT trajectory, ONE-step vertex chain, fit exclusion ON.
# The only change from SBND production: the dual chain (the second, exclusion-free PR pass that
# suggests the neutrino vertex, doc pr/112 sec 11; production operating point mode 'snap',
# transfer on, D = 2.0 cm) is off.  This is the documented production revert
# (wct-pr-perevt.jsonnet: "Revert with --tla-code dl_vtx_dual_chain=false").
# dual_chain_mode / _transfer / _transfer_max stay at their production values and are inert
# when the pass does not run -- the revert is exactly this one line, and the compiled config
# shows them unchanged (sec 2's node diff).
dl_vtx_dual_chain=false

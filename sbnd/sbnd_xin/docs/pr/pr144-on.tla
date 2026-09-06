# doc sbnd_xin/pr/144 -- the ON arm: the PDVD production pair.
# Consumed by run_pr_chain_batch.sh via PR_EXTRA_TLA (one key=value per line,
# appended LAST as --tla-code so these win over any SBND_* env hook).
# Both keys are default-OFF TLAs in cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet;
# with this file absent the arm compiles byte-identically to pre-change production
# (proof T0, md5 3bfd2a80d0201d22e9a1b5db37c774eb).
excl_t0_frame=true
kine_dqdx_skip_zero_dx=true

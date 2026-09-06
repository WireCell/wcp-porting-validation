# doc sbnd_xin/pr/144 -- the GUARD-ATTRIBUTION arm: excl_t0_frame alone.
# Same config as the ON arm minus kine_dqdx_skip_zero_dx, so a byte gate
# against work-*-d144on answers "did the kine dx<=0 guard change anything on
# SBND with the frame knob on?".  It cannot be answered from the score tables:
# the guard MASKS the NaN kine_reco_Enu that is its own fire signature.
# BOTH keys are stated explicitly so this file means the same thing before and
# after the 2026-09-06 default flip -- the guard key must be forced false, not
# left to the jsonnet default.
excl_t0_frame=true
kine_dqdx_skip_zero_dx=false

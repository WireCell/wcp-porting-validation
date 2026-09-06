# doc sbnd_xin/pr/144 sec 16 -- the LEGACY frame, restored explicitly.
# After the 2026-09-06 flip both keys default true, so a pre-flip arm needs
# these two lines.  Used for the 2x2 sentinel design: {frame on, frame off} x
# {fix on, fix off}, which is what says whether the frame patch made a shipped
# fix inert or whether it was already inert.
excl_t0_frame=false
kine_dqdx_skip_zero_dx=false

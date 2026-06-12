// Local (pdhd-running-directory) PRE-FLIP coherent-noise grouping override.
//
// Why this exists
// ---------------
// The toolkit's chndb-base.jsonnet now derives the coherent-noise groups
// directly from PD2HDChannelMap_WIBEth_visiblewires_v1.txt (duneprototypes
// c8f43809), which carries the 2025-06-30 sign-flip of the visible-wire +/-3
// rotation.  That is correct for files DECODED with the post-flip channel map.
//
// Our run-027409 frames on disk were decoded with the PRE-flip map (verified:
// APA0 U-FEMB10 = offline 3..42, no wrap).  For those files the OLD grouping
// (coh_group_shift=3 programmatic blocks + femb-negpulse-groups-shifted_v2)
// is the matching one; the new map would mis-group U/V channels by 3 at each
// FEMB boundary during coherent-noise removal.
//
// This file reproduces the OLD grouping EXACTLY (it is the chndb-base body that
// existed at toolkit 1b6de3ea).  It is selected at run time, per running
// directory, by the `coh_groups_preflip` toggle in the NF drivers
// (wct-nf-sp.jsonnet / wct-nf-sp-dnnroi.jsonnet), which the run scripts flip on
// when the local `.coh_preflip` sentinel file is present.  The toolkit cfg is
// left at the latest (post-flip) default, so colleagues are unaffected.
//
// When run-027409 (and friends) are re-decoded with a post-flip duneprototypes,
// delete the `.coh_preflip` sentinel and this override is bypassed.
{
  // coh_group_shift: cyclic offset (in offline channels) applied to U and V
  // group boundaries.  +3 on anodes 0 & 2, -3 on anodes 1 & 3 (pre-flip map).
  local coh_group_shift = 3,
  local shift(n) = if n == 0 || n == 2 then coh_group_shift
                   else if n == 1 || n == 3 then -coh_group_shift
                   else 0,
  local u_group(n, u) = std.map(function(j) n * 2560 + std.mod(40 * u + shift(n) + j, 800),
                                std.range(0, 39)),
  local v_group(n, v) = std.map(function(j) n * 2560 + 800 + std.mod(40 * v + shift(n) + j, 800),
                                std.range(0, 39)),
  local w_group(n, w) = std.range(n * 2560 + 1600 + 48 * w, n * 2560 + 1600 + 48 * (w + 1) - 1),

  // Per-anode coherent-noise groups (60 sub-groups: 20 U + 20 V + 20 W).
  groups(n):: [u_group(n, u) for u in std.range(0, 19)]
              + [v_group(n, v) for v in std.range(0, 19)]
              + [w_group(n, w) for w in std.range(0, 19)],

  // Matching pre-flip FEMB negative-pulse groups (still shipped in toolkit cfg;
  // same source the old chndb-base used).  Same list for every anode, exactly
  // as the old chndb-base set `femb_negpulse_groups: femb_params.groups`.
  negpulse_groups:: (import 'pgrapher/experiment/pdhd/femb-negpulse-groups-shifted_v2.jsonnet').groups,
}

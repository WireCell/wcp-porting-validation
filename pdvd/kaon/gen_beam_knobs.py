#!/usr/bin/env python3
"""doc pdvd/120 sec 9: generate CheckBeamParticle's neutrino-knob duplicate from
TaggerCheckNeutrino (fork by duplication, M10).  Three blocks, each a verbatim
copy of the TCN line filtered to the kept keys:
  MEMBERS   TaggerCheckNeutrino.h member declarations (with C++ defaults)
  READS     TaggerCheckNeutrino::configure  `m_x = get(config, "x", m_x);`
  PA        TaggerCheckNeutrino::visit      `pattern_algos.m_y = m_x ...;`
  DEFAULTS  `cfg["x"] = m_x;` for default_configuration()
Dropped families (not read by the beam stage) are listed in DROP_* below."""
import re, sys

H = open('/home/xqian/toolkit-dev/toolkit/clus/inc/WireCellClus/TaggerCheckNeutrino.h').read().splitlines()
C = open('/home/xqian/toolkit-dev/toolkit/clus/src/TaggerCheckNeutrino.cxx').read().splitlines()

DROP_EXACT = set('''grouping_name trackfitting_config_file perf mip_dqdx mip_dqdx_median
fit_blob_coverage dqdx_fit_keep_all_points excl_t0_frame beam_window_low beam_window_high
dQdx_scale dQdx_offset main_vertex_swap_apply rough_path_probe sgp_edge_probe vertex_scoreboard
skip_cosmic_companions cosmic_companion_min_length flash_pair_dt_us
sp_photon_flag sp_sce_correction sp_dedx_use_recomb_model sp_mean_dedx_cut
nue_sp_consistent_fv ssm_target_dir ssm_absorber_dir muon_dqdx_curve kine_plane_weights
tagger_ordered_segment_sets stem_endpoint_wcpt_parity broken_muon_cluster_id_count neutrino_type_bitmask
vertex_kink_snap vertex_junction_snap kine_continuation_debug fiducial fv_tolerance'''.split())
DROP_PREFIX = ('mcs_', 'dl_', 'dual_chain_', 'nu_', 'cosmic_', 'long_muon_cathode_bridge', 'vks_', 'vjs_')

def dropped(k):
    return k in DROP_EXACT or any(k.startswith(p) for p in DROP_PREFIX)

# ---- READS: configure() lines 147-826
reads = []
keys = []
rd = re.compile(r'^\s*(m_\w+)\s*=\s*get\(config,\s*"(\w+)",\s*(m_\w+)\);?\s*(//.*)?$')
for ln in C[146:826]:
    m = rd.match(ln)
    if not m: continue
    mem, key, mem2, cmt = m.groups()
    if mem != mem2 or not mem.startswith('m_'): continue
    if dropped(key): continue
    keys.append((key, mem))
    reads.append('        %s = get(config, "%s", %s);%s' % (mem, key, mem, ('  ' + cmt.strip()) if cmt else ''))
members_needed = {mem for _, mem in keys}

# ---- MEMBERS: header declarations
decl = re.compile(r'^\s*(bool|double|int|std::string|std::vector<double>)\s+(m_\w+)\s*\{([^}]*)\};\s*(//.*)?$')
members = []
seen = set()
for ln in H[39:1160]:
    m = decl.match(ln)
    if not m: continue
    typ, mem, dflt, cmt = m.groups()
    if mem not in members_needed or mem in seen: continue
    seen.add(mem)
    members.append('    %-20s %s{%s};%s' % (typ, mem, dflt, ('  ' + cmt.strip()) if cmt else ''))
missing = members_needed - seen
if missing:
    print('MISSING member decls:', sorted(missing), file=sys.stderr)

# ---- PA: visit() copy block 3103-3553
pa = []
pal = re.compile(r'^\s*pattern_algos\.(m_[\w.]+)\s*=\s*(.*);\s*(//.*)?$')
for ln in C[3102:3553]:
    m = pal.match(ln)
    if not m: continue
    target, rhs, cmt = m.groups()
    used = set(re.findall(r'\bm_\w+\b', rhs))
    # every member the rhs reads must be one we keep (m_dv/m_pcts/m_recomb_model/m_perf/m_mip_* are ours)
    ours = {'m_dv', 'm_pcts', 'm_recomb_model', 'm_perf', 'm_mip_dqdx', 'm_mip_dqdx_median', 'm_fiducial', 'm_use_fiducial', 'm_fv_tolerance'}
    if not used:  # constant rhs -- keep
        pass
    elif not used <= (members_needed | ours):
        continue
    if any(target.startswith(p) for p in ('m_cosmic', 'm_nue_', 'm_ssm', 'm_muon_dqdx', 'm_sp_', 'm_vtx_', 'm_vks', 'm_vjs', 'm_vertex_kink', 'm_vertex_junction', 'm_dl_', 'm_rough', 'm_sgp_edge', 'm_tagger_ordered', 'm_stem_endpoint_wcpt', 'm_broken_muon', 'm_neutrino_type', 'm_vertex_scoreboard')):
        continue
    if target in ('m_perf', 'm_mip_dqdx', 'm_mip_dqdx_median', 'm_sgp_dv', 'm_sgp_pcts', 'm_recomb_model'):
        continue  # written by the hand-written prologue
    if 'kine_charge.plane_weights' in target or 'kine_charge.continuation_debug' in target:
        continue
    pa.append('        pa.%s = %s;%s' % (target, rhs, ('  ' + cmt.strip()) if cmt else ''))

# ---- DEFAULTS
defaults = ['        cfg["%s"] = %s;' % (key, mem) for key, mem in keys]

out = []
out.append('// ==== BEGIN generated from TaggerCheckNeutrino (gen_beam_knobs.py) ====')
out.append('// MEMBERS (%d)' % len(members)); out += members
out.append('// READS (%d)' % len(reads)); out += reads
out.append('// PA (%d)' % len(pa)); out += pa
out.append('// DEFAULTS (%d)' % len(defaults)); out += defaults
out.append('// ==== END generated ====')
open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
print('members', len(members), 'reads', len(reads), 'pa', len(pa), 'defaults', len(defaults))
print('keys:', ' '.join(k for k, _ in keys))

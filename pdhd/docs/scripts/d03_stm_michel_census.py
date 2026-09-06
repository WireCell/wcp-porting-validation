#!/usr/bin/env python3
"""doc pdhd/03 -- census of the CheckSTM_Michel verdicts over one PR arm.

Reads T_stm_michel from every work/<RUN6>_<idx>_<tag>/tracking-pr.root of the
manifest (default stm/events.txt) and reports: candidates, the reject-bit
histogram (each bit counted, plus the exclusive "only this bit" count), the
Bragg-contrast distribution against the tabulated expectation, the Michel
census and energy quantiles, the dots census, the continuation fraction, and
the wall time per event from the run logs.  Report only -- nothing here tunes
a threshold.

Usage: d03_stm_michel_census.py <tag> [events.txt] [--tsv out.tsv]
"""
import sys, os, glob, re, statistics
try:
    import uproot
except ImportError:
    sys.exit("needs uproot")

# pdhd/docs/scripts/<this> -> pdhd/ (fork BY DUPLICATION of pdvd/docs/nf_sp_img_clus/scripts/d48_stm_michel_census.py)
PDHD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = os.path.join(PDHD, 'work')
argv = sys.argv[1:]
tsv_out = None
if '--tsv' in argv:
    i = argv.index('--tsv'); tsv_out = argv[i + 1]; del argv[i:i + 2]
args = [a for a in argv if not a.startswith('--')]
tag = args[0]
ev_file = args[1] if len(args) > 1 else os.path.join(PDHD, 'stm', 'events.txt')
if os.path.isfile(ev_file):
    events = ['%06d_%s' % (int(l.split()[0]), l.split()[1]) for l in open(ev_file) if l.strip() and not l.startswith('#')]
else:
    # PDHD default manifest: every work/029107_<idx>_<tag> dir of the arm
    events = sorted({os.path.basename(d)[:-len(tag) - 1] for d in glob.glob(os.path.join(W, '*_' + tag))},
                    key=lambda e: (e.split('_')[0], int(e.split('_')[1])))

BITS = [(1, 'no_chain'), (2, 'stop_unmatched'), (4, 'no_bragg'), (8, 'shape_flat'), (16, 'not_muon_pid'),
        (32, 'continuation'), (64, 'stop_near_boundary'), (128, 'vertex_hadron'), (256, 'short'),
        (512, 'profile_sparse'), (1024, 'plateau_off_mip'), (2048, 'stop_into_dead'), (4096, 'cluster_not_track')]

rows = []
n_missing = 0
walls = []
for e in events:
    d = os.path.join(W, '%s_%s' % (e, tag))
    f = os.path.join(d, 'tracking-pr.root')
    log = glob.glob(os.path.join(d, 'wct_pr_*.log'))
    if log:
        for line in open(log[0], errors='replace'):
            m = re.search(r'Timer: Total ([0-9.]+) wall-sec', line)
            if m: walls.append(float(m.group(1)))
    if not os.path.isfile(f):
        n_missing += 1
        continue
    try:
        with uproot.open(f) as rf:
            if 'T_stm_michel' not in rf:
                continue
            t = rf['T_stm_michel'].arrays(library='np')
    except Exception as ex:
        print('WARN', f, ex, file=sys.stderr)
        continue
    n = len(t['cluster_id'])
    for i in range(n):
        r = {k: (v[i].item() if hasattr(v[i], 'item') else v[i]) for k, v in t.items()}
        r['event'] = e
        rows.append(r)

def q(v, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = sorted(v)
    if not v: return 'n/a'
    return ' / '.join('%.2f' % v[min(len(v) - 1, int(p * len(v)))] for p in ps)

N = len(rows)
print('arm %s: %d events in manifest, %d without tracking-pr.root, %d STM candidates' % (tag, len(events), n_missing, N))
if walls:
    print('wall per event (s): median %.1f  p90 %.1f  max %.1f  (n=%d)' % (statistics.median(walls), sorted(walls)[int(0.9 * len(walls))], max(walls), len(walls)))
if N == 0:
    sys.exit(0)
n_stm = sum(1 for r in rows if r['reject_bits'] == 0)
print('pass every check (is_stm): %d / %d = %.1f %%' % (n_stm, N, 100.0 * n_stm / N))
print('reject bits (count carrying the bit / count where it is the ONLY bit):')
for b, name in BITS:
    c = sum(1 for r in rows if r['reject_bits'] & b)
    only = sum(1 for r in rows if r['reject_bits'] == b)
    print('  %-20s %4d  %5.1f %%   only: %d' % (name, c, 100.0 * c / N, only))
have_chain = [r for r in rows if r['n_chain_segs'] > 0]
print('chain: %d with a muon chain; segments per chain p10/25/50/75/90 = %s; muon length cm = %s' % (
    len(have_chain), q([r['n_chain_segs'] for r in have_chain]), q([r['muon_len'] for r in have_chain])))
print('stop vertex vs tagger stop (cm) = %s' % q([r['stop_dis'] for r in have_chain]))
if 'n_dead_pts' in rows[0]:
    dead = [r['n_dead_pts'] / max(1, r['n_profile_pts']) for r in have_chain]
    print('dead profile points (dQ/dx < profile_min_dqdx_frac x mip): fraction per chain p10/25/50/75/90 = %s; chains with dead_frac_cmp > 0.3 within the compare range: %d / %d' % (
        q(dead), sum(1 for r in have_chain if r['dead_frac_cmp'] > 0.3), len(have_chain)))
bv = [r for r in have_chain if r['bragg_valid']]
print('bragg: %d valid; contrast p10/25/50/75/90 = %s; expected = %s; contrast/expected = %s' % (
    len(bv), q([r['contrast'] for r in bv]), q([r['contrast_expected'] for r in bv]),
    q([r['contrast'] / r['contrast_expected'] for r in bv if r['contrast_expected'] > 0])))
n2 = sum(1 for r in bv if r['contrast'] >= 2.0)
print('  contrast >= 2.0: %d / %d (doc 25 sec 13.6 measured 51/538 on the tagger fits)' % (n2, len(bv)))
print('KS: ks_mu < ks_flat on %d / %d chains' % (sum(1 for r in have_chain if r['ks_mu'] < r['ks_flat']), len(have_chain)))
print('template PID forward gate=1 and muon best: %d / %d' % (
    sum(1 for r in have_chain if r['comp_fwd0'] > 0.5 and r['comp_fwd1'] < r['comp_fwd2'] and r['comp_fwd1'] < r['comp_fwd3']), len(have_chain)))
if 'n_ext' in rows[0]:
    ne = [r for r in have_chain if r['n_ext'] > 0]
    print('chain extended past the tagger stop (stop_extend_max): %d / %d chains; extensions per chain p10/25/50/75/90 = %s; total extension cm = %s; stop moved by (cm) = %s' % (
        len(ne), len(have_chain), q([r['n_ext'] for r in ne]), q([r['ext_len'] for r in ne]), q([r['stop_dis'] for r in ne])))
    print('dead region ahead of the visible end (dead_volume_check): %d flagged, %d clear, %d not evaluated' % (
        sum(1 for r in rows if r['dead_ahead'] == 1), sum(1 for r in rows if r['dead_ahead'] == 0), sum(1 for r in rows if r['dead_ahead'] < 0)))
nc = sum(1 for r in have_chain if r['reject_bits'] & 32)
print('continuation past the stop: %d / %d = %.1f %% (doc 42 sec 4.4: 26 %% leftover)' % (nc, len(have_chain), 100.0 * nc / max(1, len(have_chain))))
na = sum(1 for r in have_chain if r['n_stop_arms'] > 0)
nother = sum(1 for r in have_chain if r['n_stop_arms'] > 0 and not r['michel_found'] and not (r['reject_bits'] & 32))
print('stop arms: %d / %d chains have >= 1 arm at the stop vertex; %d of those have neither a Michel nor a continuation (arms classified Other)' % (na, len(have_chain), nother))
nd = sum(1 for r in have_chain if r['n_delta'] > 0)
print('delta rays: %d chains with >= 1 (total %d); body hadron arms on %d' % (nd, sum(r['n_delta'] for r in have_chain), sum(1 for r in have_chain if r['n_body_hadron'] > 0)))
mi = [r for r in rows if r['michel_found']]
print('michel: %d / %d candidates (%d among is_stm); len cm = %s; mip = %s; kink deg = %s; KE(dQ/dx) MeV = %s' % (
    len(mi), N, sum(1 for r in mi if r['reject_bits'] == 0), q([r['michel_len'] for r in mi]), q([r['michel_mip'] for r in mi]),
    q([r['michel_kink_deg'] for r in mi]), q([r['michel_ke_dqdx'] for r in mi])))
dots = [r for r in rows if r['n_dots'] > 0]
print('dots: %d candidates with >= 1 fitted dot (total %d, KE MeV = %s); unfitted dot clusters on %d' % (
    len(dots), sum(r['n_dots'] for r in dots), q([r['dots_ke_dqdx'] for r in dots]), sum(1 for r in rows if r['n_dot_clusters_unfit'] > 0)))
print('stop inside fiducial: %d / %d (unknown %d)' % (sum(1 for r in rows if r['in_fv'] == 1), N, sum(1 for r in rows if r['in_fv'] < 0)))

# particle-flow census from the Bee mc.json: is the flow rooted at the entry with the muon as the first child?
import zipfile, json
pf = {'roots': 0, 'mu_first': 0, 'e_under_root': 0, 'mu_with_e_child': 0, 'empty_root': 0}
for e in events:
    zp = os.path.join(W, '%s_%s' % (e, tag), 'mabc-pr.zip')
    if not os.path.isfile(zp): continue
    try:
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.endswith('mc.json')]
            if not names: continue
            mc = json.loads(z.read(names[0]))
    except Exception as ex:
        print('WARN', zp, ex, file=sys.stderr); continue
    for root in mc:
        pf['roots'] += 1
        ch = root.get('children', [])
        if not ch: pf['empty_root'] += 1; continue
        texts = [c.get('text', '') for c in ch]
        if texts[0].startswith('mu-'): pf['mu_first'] += 1
        if any(t.startswith('e-') for t in texts): pf['e_under_root'] += 1
        for c in ch:
            if c.get('text', '').startswith('mu-') and any(g.get('text', '').startswith('e-') for g in c.get('children', [])):
                pf['mu_with_e_child'] += 1; break
print('PF trees (mc.json): %d roots; first child mu- %d; an e- directly under the root %d; a mu- with an e- child %d; empty roots %d' % (
    pf['roots'], pf['mu_first'], pf['e_under_root'], pf['mu_with_e_child'], pf['empty_root']))
if tsv_out:
    keys = sorted(rows[0].keys())
    with open(tsv_out, 'w') as fo:
        fo.write('\t'.join(keys) + '\n')
        for r in rows:
            fo.write('\t'.join(str(r[k]) for k in keys) + '\n')
    print('wrote', tsv_out)

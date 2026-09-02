#!/usr/bin/env python3
"""doc 94 round 2 -- the entry-rise figure.

Left  : the dQ/dx profile from the BOUNDARY end for the labelled events, with
        each one's measured shoulder marked.  The owner's picture.
Right : the shoulder distribution over the STM-tagged data population, from
        the probe census TSV -- the go/no-go that decided whether the feature
        is a separator or a continuum.

Usage: doc94r2_shoulder_plot.py <stm_fit arm> <census.tsv> <out.png>
Read-only.
"""
import io, json, os, sys, tarfile
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MIP = 56000.0
# event : (label, cluster, measured shoulder cm, owner verdict)
EVENTS = [
    ('4',  '827-27-4',   '18', 8.4,  'NEUTRINO (target)'),
    ('28', '304-6-28',    '5', 19.4, 'neutrino (round 1)'),
    ('12', '707-18-12',   '7', 0.0,  'GENUINE STM'),
    ('22', '966-2-22',    '6', 0.0,  'neutrino (round 1)'),
    ('17', '36-77-17',   '20', 0.0,  'control, not STM'),
]

def load(evt, root):
    metas, arrs = {}, {}
    with tarfile.open(f'{root}/pr_evt{evt}/pctree-pr-evt{evt}.tar.gz') as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrs[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    byp = {md['datapath']: arrs.get(b) for b, md in metas.items() if 'datapath' in md}
    pre = f'pointtrees/{evt}/live/pointclouds/namedpcs/stm_fit/arrays/'
    if pre + 'L' not in byp:
        return None
    return byp[pre+'L']/10., byp[pre+'dQ']/(byp[pre+'dx']/10.+1e-9)/MIP

root, census, out = sys.argv[1], sys.argv[2], sys.argv[3]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))

for evt, lab, cid, sh, verdict in EVENTS:
    r = load(evt, root)
    if r is None:
        print(f'{lab}: no stm_fit'); continue
    L, dq = r
    m = L < 45
    hot = 'NEUTRINO' in verdict or 'GENUINE' in verdict
    p = ax[0].plot(L[m], dq[m], '-', lw=2.0 if hot else 1.2,
                   label=f'{lab}  shoulder {sh:.1f} cm  [{verdict}]')
    if sh > 0:
        ax[0].axvspan(0, sh, color=p[0].get_color(), alpha=0.12)
ax[0].axhline(1.0, color='0.5', ls='--', lw=1)
ax[0].set_xlabel('L from the boundary point [cm]')
ax[0].set_ylabel('dQ/dx [MIP]')
ax[0].set_title('The ENTRY end -- what no other STM predicate reads\n'
                'shaded = the measured contiguous elevated run')
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

rows = [l.rstrip('\n').split('\t') for l in open(census) if l.strip()]
h = rows[0]; i_sh, i_stm = h.index('shoulder'), h.index('stm')
sh_stm = [float(r[i_sh]) for r in rows[1:] if r[i_stm] == '1']
bins = np.arange(0, 55, 2.5)
ax[1].hist(sh_stm, bins=bins, color='0.55', edgecolor='k')
ax[1].axvline(8.4, color='crimson', lw=2, label='827-27-4 = 8.4 cm (the target)')
ax[1].set_yscale('symlog')
ax[1].set_xlabel('shoulder [cm]')
ax[1].set_ylabel('STM-tagged data bundles')
ax[1].set_title(f'Population, all SBND data events (n={len(sh_stm)} STM=1 with a probe)\n'
                'the go/no-go: bimodal, not a continuum')
ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(out, dpi=105); print('wrote', out)

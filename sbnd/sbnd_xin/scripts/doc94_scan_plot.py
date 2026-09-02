#!/usr/bin/env python3
"""doc 94 -- render one event for a hand scan: 3 projections + the STM fit.

Blind by construction: this draws the DETECTOR, not the verdict.  No guard
result, no tag, no cos_y is printed on the image -- the point is to judge the
picture and only then look up what the code said (feedback_blind_the_scan_sheet).

Usage: doc94_scan_plot.py <pr_evt_dir> <main_cluster_id> <out.png>
"""
import io, json, os, sys, tarfile, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

prdir, main_id, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
evt = os.path.basename(prdir).replace('pr_evt', '')

# ---- points from the Bee clustering layer ---------------------------------
with zipfile.ZipFile(os.path.join(prdir, 'mabc-pr.zip')) as z:
    name = [n for n in z.namelist() if n.endswith('-clustering-global.json')][0]
    d = json.load(io.BytesIO(z.read(name)))
x = np.array(d['x'], float); y = np.array(d['y'], float); z_ = np.array(d['z'], float)
cid = np.array(d.get('cluster_id', d.get('real_cluster_id', [])), int)
q = np.array(d.get('q', np.ones(len(x))), float)

# ---- the fitted STM trajectory from the stm_fit PC ------------------------
fitpts = None
tar = os.path.join(prdir, f'pctree-pr-evt{evt}.tar.gz')
if os.path.exists(tar):
    metas, arrs = {}, {}
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrs[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    byp = {md['datapath']: arrs.get(b) for b, md in metas.items() if 'datapath' in md}
    pre = f'pointtrees/{evt}/live/pointclouds/namedpcs/stm_fit/arrays/'
    if pre + 'x' in byp:
        fitpts = (byp[pre+'x']/10., byp[pre+'y']/10., byp[pre+'z']/10.)

sel = cid == main_id
fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))
views = [(z_, y, 'z [cm]', 'y [cm]'), (z_, x, 'z [cm]', 'x [cm]'), (x, y, 'x [cm]', 'y [cm]')]
fv = [(0.85, 500.15), (-201.05, 201.05), (-199.31, 199.31)]  # z, x, y
for ax, (a, b, la, lb) in zip(axes, views):
    ax.scatter(a[~sel], b[~sel], s=0.5, c='0.78', linewidths=0, label='other clusters')
    sc = ax.scatter(a[sel], b[sel], s=2.0, c=np.clip(q[sel], 0, np.percentile(q[sel], 98)),
                    cmap='viridis', linewidths=0)
    if fitpts is not None:
        fa = {'z [cm]': fitpts[2], 'x [cm]': fitpts[0], 'y [cm]': fitpts[1]}[la]
        fb = {'z [cm]': fitpts[2], 'x [cm]': fitpts[0], 'y [cm]': fitpts[1]}[lb]
        ax.plot(fa, fb, '-', color='red', lw=1.2, alpha=0.9)
        ax.plot(fa[0], fb[0], 'o', color='red', ms=7, mfc='none', mew=2)
        ax.plot(fa[-1], fb[-1], 's', color='red', ms=7)
    ax.set_xlabel(la); ax.set_ylabel(lb); ax.grid(alpha=0.25)
    lim = {'z [cm]': fv[0], 'x [cm]': fv[1], 'y [cm]': fv[2]}
    for v in lim[la]: ax.axvline(v, color='C3', ls=':', lw=0.8)
    for v in lim[lb]: ax.axhline(v, color='C3', ls=':', lw=0.8)
    ax.set_aspect('equal', adjustable='datalim')
axes[0].set_title(f'evt {evt}  cluster {main_id}   colour = charge; grey = rest of event')
axes[1].set_title('red line = fitted STM path;  open circle = boundary end, square = stop')
axes[2].set_title('dotted = fiducial box')
plt.tight_layout(); plt.savefig(out, dpi=105)
print('wrote', out, ' main pts', int(sel.sum()), ' total', len(x))

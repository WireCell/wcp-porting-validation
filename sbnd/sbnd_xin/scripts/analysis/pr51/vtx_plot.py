#!/usr/bin/env python3
"""doc pr/51 scoping: near-vertex scatter of track_fit-global (colored per
real_cluster_id, labeled) over the raw image (grey), plus PR vertices (x).
Two projections.  Usage: vtx_plot.py <ARM> <EVT> [RADIUS=25] [OUT.png]"""
import sys, os, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
arm, evt = sys.argv[1], int(sys.argv[2])
RAD = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
out = sys.argv[4] if len(sys.argv) > 4 else '%s_%d.png' % (arm.replace('work-', ''), evt)

z = zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip'))
load = lambda n: json.loads(z.read('data/0/0-%s.json' % n))
vtx, fit, img = load('vertices-global'), load('track_fit-global'), load('clustering-global')

vp = np.array([vtx['x'], vtx['y'], vtx['z']]).T
vq = np.array(vtx['q'])
mv = vp[vq == 15000][0]
fp = np.array([fit['x'], fit['y'], fit['z']]).T
fr = np.array(fit['real_cluster_id'])
fq = np.array(fit['q'], dtype=float)
ip = np.array([img['x'], img['y'], img['z']]).T

fm = np.linalg.norm(fp - mv, axis=1) < RAD
im = np.linalg.norm(ip - mv, axis=1) < RAD
vm = np.linalg.norm(vp - mv, axis=1) < RAD

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, (a, b, la, lb) in zip(axes, [(2, 1, 'z', 'y'), (2, 0, 'z', 'x')]):
    ax.scatter(ip[im][:, a], ip[im][:, b], s=4, c='0.85', label='image')
    for r in sorted(set(fr[fm])):
        m = fm & (fr == r)
        ax.scatter(fp[m][:, a], fp[m][:, b], s=14, label='rcid %d (med dQdx %d)' % (r, int(np.median(fq[m]))))
        c = fp[m][np.argmin(np.linalg.norm(fp[m] - mv, axis=1))]
        ax.annotate(str(r), (c[a], c[b]), fontsize=8)
    ax.scatter(vp[vm][:, a], vp[vm][:, b], marker='x', s=60, c='k')
    ax.scatter([mv[a]], [mv[b]], marker='*', s=200, c='red')
    ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_title('%s evt %d (%s-%s)' % (arm, evt, la, lb))
axes[0].legend(fontsize=7, loc='best')
plt.tight_layout()
plt.savefig(out, dpi=110)
print('wrote', out)

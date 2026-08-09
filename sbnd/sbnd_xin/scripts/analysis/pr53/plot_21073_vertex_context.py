#!/usr/bin/env python3
"""doc pr/53 round 3: is the 18345-21073 pair separable by tightening a
connectivity threshold? This plots the PR-level context around the
charge-contiguous path's turning point (the apex found by
plot_21073_gap.py, near (-33.7,26.2,365.0)): the reconstructed main vertex
(red star, vertices-global q=15000) and the shower_track-global fit points
of every real_cluster_id within range, colored separately, to show whether
multiple distinct PR segments converge there (a genuine vertex/kink region)
rather than one bridged gap.

Usage: plot_21073_vertex_context.py [OUT.png]
"""
import sys, os, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
PR = os.path.join(SB, 'work-bee-0809/pr_evt21073/mabc-pr.zip')
QL = os.path.join(SB, 'work-ncpi0-cb0805/ql_evt21073/mabc-all-apa.zip')
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SB, 'docs/pr/53_21073_vertex_context.png')

A = np.array([-36.5, 32.0, 367.5])
B = np.array([-31.2, 28.8, 369.6])
apex = np.array([-33.71, 26.24, 364.95])  # geodesic turning point, from plot_21073_gap.py

zpr = zipfile.ZipFile(PR)
load = lambda n: json.loads(zpr.read('data/0/0-%s.json' % n))
vtx, st = load('vertices-global'), load('shower_track-global')
vp = np.array([vtx['x'], vtx['y'], vtx['z']]).T
vq = np.array(vtx['q'])
main_vtx = vp[vq == 15000][0]

sp = np.array([st['x'], st['y'], st['z']]).T
sr = np.array(st['real_cluster_id'])

zql = zipfile.ZipFile(QL)
img = json.loads(zql.read('data/0/0-img-global.json'))
ip = np.c_[img['x'], img['y'], img['z']]
near_img = np.linalg.norm(ip - apex, axis=1) < 12

near = np.linalg.norm(sp - apex, axis=1) < 8
rcids = sorted(set(sr[near].tolist()))
cmap = plt.get_cmap('tab10')

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, (i0, i1, l0, l1) in zip(axes, [(2, 1, 'z (cm)', 'y (cm)'), (0, 2, 'x (cm)', 'z (cm)')]):
    ax.scatter(ip[near_img][:, i0], ip[near_img][:, i1], s=8, c='0.85', label='image charge (QL)', zorder=1)
    for k, rc in enumerate(rcids):
        m = near & (sr == rc)
        ax.scatter(sp[m][:, i0], sp[m][:, i1], s=16, color=cmap(k % 10), label='PR segment rcid %d' % rc, zorder=3)
    ax.scatter([main_vtx[i0]], [main_vtx[i1]], marker='*', s=320, c='red', edgecolors='black',
               zorder=5, label='PR main vertex')
    ax.scatter([apex[i0]], [apex[i1]], marker='D', s=90, c='lime', edgecolors='black',
               zorder=5, label='charge-path turning point (apex)')
    ax.scatter(*zip((A[i0], A[i1]), (B[i0], B[i1])), marker='*', s=180, c='black', zorder=6, label='A, B (owner pair)')
    ax.set_xlabel(l0); ax.set_ylabel(l1)
axes[0].set_title('(y,z) view')
axes[1].set_title('(x,z) view -- drift direction')
axes[0].legend(fontsize=7, loc='best')
plt.suptitle('18345-21073: the charge path\'s turn sits 2.65cm from the PR main vertex --\n'
             'multiple PR segments converge there, consistent with a real vertex/kink, not a bridged gap')
plt.tight_layout()
plt.savefig(out, dpi=130)
print('wrote', out)
print('main vertex - apex distance: %.2f cm' % np.linalg.norm(main_vtx - apex))
print('rcids near apex (<8cm):', rcids)

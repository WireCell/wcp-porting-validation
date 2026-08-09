#!/usr/bin/env python3
"""doc pr/54: 18255-142421 owner point (108.6,-71.2,220.9), "cluster 7021" --
visualize the separated ~2930-point component of PR cluster 7 that has no
fitted trajectory anywhere near it. Two-panel (y,z)/(x,z) figure: all of
cluster 7's image charge, the separated component highlighted, every
track_fit-global point belonging to cluster 7 (its 11 final segments), and the
owner's point.

Usage: plot_142421_blob.py [OUT.png]
"""
import sys, os, json, zipfile
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
PR = os.path.join(SB, 'work-bee-0809/pr_evt142421/mabc-pr.zip')
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SB, 'docs/pr/54_142421_blob.png')

CID = 7
T = np.array([108.6, -71.2, 220.9])

z = zipfile.ZipFile(PR)
load = lambda n: json.loads(z.read('data/0/0-%s.json' % n))
cg, ft = load('clustering-global'), load('track_fit-global')

C = np.c_[cg['x'], cg['y'], cg['z']]
cid = np.array(cg['cluster_id'])
X = C[cid == CID]

F = np.c_[ft['x'], ft['y'], ft['z']]
frid = np.array(ft['real_cluster_id'])
Fc = F[(frid // 1000) == CID]

t = cKDTree(X)
pr = np.array(list(t.query_pairs(1.6)))
g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
n, lab = connected_components(g, directed=False)
i0 = int(np.argmin(np.linalg.norm(X - T, axis=1)))
lt = lab[i0]
blob = X[lab == lt]
rest = X[lab != lt]

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, (i0_, i1_, l0, l1) in zip(axes, [(2, 1, 'z (cm)', 'y (cm)'), (0, 2, 'x (cm)', 'z (cm)')]):
    ax.scatter(rest[:, i0_], rest[:, i1_], s=6, c='0.82', label='cluster 7 image charge (rest)', zorder=1)
    ax.scatter(blob[:, i0_], blob[:, i1_], s=8, c='tab:orange',
               label='separated component (n=%d)' % len(blob), zorder=2)
    ax.scatter(Fc[:, i0_], Fc[:, i1_], s=14, c='tab:blue', marker='x',
               label='track_fit-global (cluster 7 final segments, n=%d)' % len(Fc), zorder=3)
    ax.scatter([T[i0_]], [T[i1_]], marker='*', s=260, c='red', edgecolors='black',
               zorder=5, label='owner point (108.6,-71.2,220.9)')
    ax.set_xlabel(l0); ax.set_ylabel(l1)
axes[0].set_title('(y,z) view')
axes[1].set_title('(x,z) view -- drift direction')
axes[0].legend(fontsize=7.5, loc='best')
plt.suptitle('18255-142421: separated EM-shower component of cluster 7 has no fitted\n'
             'trajectory anywhere near it (nearest track_fit point 7.30 cm away)')
plt.tight_layout()
plt.savefig(out, dpi=130)
print('wrote', out)
print('blob n=%d, rest n=%d, cluster-7 fit points n=%d' % (len(blob), len(rest), len(Fc)))

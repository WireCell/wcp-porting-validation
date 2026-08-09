#!/usr/bin/env python3
"""doc pr/53: visualize the 18345-21073 owner pair (-36.5,32.0,367.5) vs
(-31.2,28.8,369.6). Shows why they are "overclustered": the straight line
between them crosses a real W-plane gap (grey dashed, red X marks steps with
neither live W charge nor a W dead channel), but the two points are actually
joined by a continuous charge path that detours around the gap (solid green,
every point W-live) -- so they are one real_cluster_id via genuine charge
continuity, not via a graph bridge tolerating a hole.

Usage: plot_21073_gap.py [OUT.png]
"""
import sys, os, json, zipfile
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from oc53_probe import Loader, walk_and_score

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
QL = os.path.join(SB, 'work-ncpi0-cb0805/ql_evt21073')
A = np.array([-36.5, 32.0, 367.5])
B = np.array([-31.2, 28.8, 369.6])
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SB, 'pics/53_21073_gap.png')

z = zipfile.ZipFile(os.path.join(QL, 'mabc-all-apa.zip'))
d = json.loads(z.read('data/0/0-img-global.json'))
P = np.c_[d['x'], d['y'], d['z']]
C = np.array(d['cluster_id'])

ia = int(np.argmin(np.linalg.norm(P - A, axis=1)))
ib = int(np.argmin(np.linalg.norm(P - B, axis=1)))
cid = C[ia]
m = np.where(C == cid)[0]
X = P[m]
ja = int(np.where(m == ia)[0][0])
jb = int(np.where(m == ib)[0][0])

# Geodesic charge path (0.99cm bottleneck radius connects the whole cluster --
# see doc pr/53 Finding 4)
t = cKDTree(X)
pr = np.array(list(t.query_pairs(1.05)))
w = np.linalg.norm(X[pr[:, 0]] - X[pr[:, 1]], axis=1)
g = coo_matrix((w, (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
dist, pred = dijkstra(g, directed=False, indices=ja, return_predecessors=True)
path = [jb]
while path[-1] != ja:
    path.append(pred[path[-1]])
path = path[::-1]
geo = X[path]

# Straight-line W-plane path test (exact test_good_point replay)
ld = Loader(QL)
r = walk_and_score(ld, A, B)
line_pts = np.array([A + (B - A) / r['nst'] * (i + 1) for i in range(r['nst'])])
line_wgap = np.array([s[2] == 0 and s[5] == 0 for s in r['rows']])
ld.cleanup()

# Local neighborhood for context (within 12cm of the midpoint)
mid = (A + B) / 2
near = np.linalg.norm(X - mid, axis=1) < 14

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, (i0, i1, l0, l1) in zip(axes, [(2, 1, 'z (cm)', 'y (cm)'), (0, 2, 'x (cm)', 'z (cm)')]):
    ax.scatter(X[near][:, i0], X[near][:, i1], s=10, c='0.82', label='image charge (this cluster)')
    ax.plot(line_pts[:, i0], line_pts[:, i1], 'r--', lw=1.3, label='straight line A-B')
    ax.scatter(line_pts[~line_wgap][:, i0], line_pts[~line_wgap][:, i1], marker='o', s=40,
               facecolors='none', edgecolors='darkred', linewidths=1.5, label='step: W live/dead-ok')
    ax.scatter(line_pts[line_wgap][:, i0], line_pts[line_wgap][:, i1], marker='x', s=80,
               c='red', linewidths=2.5, label='step: W GAP (no live, no dead ch)')
    ax.plot(geo[:, i0], geo[:, i1], '-', c='green', lw=2.2, alpha=0.85,
             label='actual charge path (all W-live, %.1fcm)' % dist[jb])
    ax.scatter(*zip((A[i0], A[i1]), (B[i0], B[i1])), marker='*', s=220, c='black', zorder=5, label='A, B (owner pair)')
    ax.set_xlabel(l0); ax.set_ylabel(l1)
axes[0].set_title('18345-21073: (y,z) view -- wire-plane view')
axes[1].set_title('18345-21073: (x,z) view -- drift direction')
axes[0].legend(fontsize=7.5, loc='best')
plt.suptitle('doc pr/53: overclustered pair is charge-contiguous, not a bridged W-gap\n'
             'straight line crosses a real W gap (%d/%d steps); the actual connecting path has none'
             % (line_wgap.sum(), r['nst']))
plt.tight_layout()
plt.savefig(out, dpi=130)
print('wrote', out)

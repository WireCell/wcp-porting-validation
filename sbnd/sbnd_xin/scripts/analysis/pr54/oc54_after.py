#!/usr/bin/env python3
"""doc pr/54 round 3: before/after comparison for the other_seg_keep_isolated
knob on 18255-142421.

(1) Recovery: distances from the 2930-point separated component of cluster 7
    (doc pr/54 SS3) to the nearest track_fit-global point, before vs after.
(2) Superset check (the owner's question): per-cluster comparison of
    track_fit-global between the bare arm and the knob-on arm -- which
    clusters' fitted point sets are unchanged, which changed, what was added.

Usage: oc54_after.py [BARE_ZIP] [ON_ZIP]
"""
import sys, os, json, zipfile
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
BARE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SB, 'work-bee-0809/pr_evt142421/mabc-pr.zip')
ON = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SB, 'work-pr54-on142421/pr_evt142421/mabc-pr.zip')

CID = 7
T = np.array([108.6, -71.2, 220.9])


def load(zpath, name):
    with zipfile.ZipFile(zpath) as z:
        return json.loads(z.read('data/0/0-%s.json' % name))


def fit_points(zpath):
    ft = load(zpath, 'track_fit-global')
    P = np.c_[ft['x'], ft['y'], ft['z']]
    rc = np.array(ft['real_cluster_id'])
    return P, rc


# --- the separated component, from the bare arm's clustering-global (the
# image points do not depend on the knob; assert that below).
cg_b = load(BARE, 'clustering-global')
cg_o = load(ON, 'clustering-global')
Cb = np.c_[cg_b['x'], cg_b['y'], cg_b['z']]
Co = np.c_[cg_o['x'], cg_o['y'], cg_o['z']]
same_img = Cb.shape == Co.shape and np.allclose(np.sort(Cb, axis=0), np.sort(Co, axis=0))
print('clustering-global identical between arms: %s' % same_img)

cid = np.array(cg_b['cluster_id'])
X = Cb[cid == CID]
t = cKDTree(X)
pr = np.array(list(t.query_pairs(1.6)))
g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
_, lab = connected_components(g, directed=False)
blob = X[lab == lab[int(np.argmin(np.linalg.norm(X - T, axis=1)))]]
print('separated component n=%d' % len(blob))

for tag, zpath in (('bare', BARE), ('on', ON)):
    F, rc = fit_points(zpath)
    d, _ = cKDTree(F).query(blob)
    down = np.linalg.norm(F - T, axis=1).min()
    print('%-5s: fit points total=%5d | blob->fit min=%.2f cm median=%.2f cm frac>5cm=%.3f | owner-point->fit min=%.2f cm'
          % (tag, len(F), d.min(), np.median(d), (d > 5).mean(), down))

# --- superset check: per-cluster fitted point sets ---------------------------
Fb, rb = fit_points(BARE)
Fo, ro = fit_points(ON)
cb, co = rb // 1000, ro // 1000
clusters = sorted(set(cb.tolist()) | set(co.tolist()))
print('\nper-cluster track_fit-global comparison (bare -> on):')
unchanged = []
for c in clusters:
    A, B = Fb[cb == c], Fo[co == c]
    if A.shape == B.shape and len(A) and np.allclose(np.sort(A, axis=0), np.sort(B, axis=0), atol=1e-9):
        unchanged.append(c)
        continue
    dAB = cKDTree(B).query(A)[0] if len(A) and len(B) else np.array([np.inf])
    print('  cluster %3d: n %5d -> %5d | old-point max displacement to nearest new %.3f cm, median %.3f cm'
          % (c, len(A), len(B), dAB.max(), np.median(dAB)))
print('  unchanged clusters (%d): %s' % (len(unchanged), unchanged))

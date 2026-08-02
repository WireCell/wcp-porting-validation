#!/usr/bin/env python3
"""Quantify one stage's merge: which pre-stage clusters land in a given
post-stage cluster, with per-cluster npts / length / min distance to a chosen
"absorber" pre-stage cluster (doc pr/19, SBND run 18253 evt 444187).

Usage:
  ./oc_stage_gap.py <pre.json> <post.json> <post_cluster_id> <pre_absorber_id>

pre/post are consecutive 0-tr*.json trace layers (same point set).
"""
import json, sys
import numpy as np
from scipy.spatial import cKDTree

pre_f, post_f, post_id, big_id = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
pre = json.load(open(pre_f))
post = json.load(open(post_f))
Pa = np.c_[pre['x'], pre['y'], pre['z']].astype(float)
ca = np.asarray(pre['cluster_id'], int)
Pb = np.c_[post['x'], post['y'], post['z']].astype(float)
cb = np.asarray(post['cluster_id'], int)

d, j = cKDTree(Pb).query(Pa, k=1)
assert d.max() < 1e-6, f"point sets differ (max {d.max()})"
s_of = {}
for c in np.unique(ca):
    ids, cnt = np.unique(cb[j[ca == c]], return_counts=True)
    s_of[c] = int(ids[np.argmax(cnt)])
into = sorted(c for c, v in s_of.items() if v == post_id)
print(f"pre-stage clusters absorbed into post cluster {post_id}: {len(into)}")

Pbig = Pa[ca == big_id]
tbig = cKDTree(Pbig)
print(f"absorber {big_id}: {len(Pbig)} pts, "
      f"x[{Pbig[:,0].min():.1f},{Pbig[:,0].max():.1f}] "
      f"y[{Pbig[:,1].min():.1f},{Pbig[:,1].max():.1f}] "
      f"z[{Pbig[:,2].min():.1f},{Pbig[:,2].max():.1f}]")
print(f"{'clus':>5} {'npts':>5} {'len(cm)':>8} {'gap(cm)':>8}  extent")
for c in into:
    if c == big_id:
        continue
    P = Pa[ca == c]
    a = P[np.argmax(((P - P[0]) ** 2).sum(1))]
    ln = np.sqrt(((P - a) ** 2).sum(1).max())
    gap = tbig.query(P, k=1)[0].min()
    print(f"{c:>5} {len(P):>5} {ln:>8.1f} {gap:>8.1f}  "
          f"x[{P[:,0].min():.1f},{P[:,0].max():.1f}] "
          f"y[{P[:,1].min():.1f},{P[:,1].max():.1f}] "
          f"z[{P[:,2].min():.1f},{P[:,2].max():.1f}]")

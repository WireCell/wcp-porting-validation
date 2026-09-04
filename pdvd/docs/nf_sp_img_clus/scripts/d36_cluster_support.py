#!/usr/bin/env python3
"""doc pdvd/36 sec 10.4 -- the per-cluster table behind the hand-scan Bee sets.

For each named (run, idx, cluster) and each arm, report the cluster's own 3-D
charge extent, the fitted trajectory's extent, how much of that charge the fit
covers (within 2 cm), and how much of the fit sits away from ANY 3-D charge of
the event.  Both layers come from the same `mabc-pr.zip`, so they are in one
frame -- the frac-0 arm, whose fit lies on charge by construction, is the
control that proves it (median 0.4-0.5 cm).

Usage:
  cd <repo>/pdvd
  python3 docs/nf_sp_img_clus/scripts/d36_cluster_support.py \
      d36p000,d36off,d36on <<'IN'
  039252 2 109
  039349 48 53
  IN
"""
import json, os, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # <repo>/pdvd
W = os.path.join(PDVD, 'work')
ARMS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['d36p000', 'd36off', 'd36on']

def diam(X):
    """Two-pass farthest-point approximation of the point set's diameter."""
    if len(X) < 2:
        return 0.0
    b = X[np.argmax(np.linalg.norm(X - X[0], axis=1))]
    return float(np.linalg.norm(b - X[np.argmax(np.linalg.norm(X - b, axis=1))]))

cache = {}
def load(tag, run, idx):
    key = (tag, run, idx)
    if key not in cache:
        z = zipfile.ZipFile(os.path.join(W, '%s_%s_%s' % (run, idx, tag), 'mabc-pr.zip'))
        ft = json.loads(z.read('data/0/0-stm_fit-global.json'))
        cl = json.loads(z.read('data/0/0-clustering-global.json'))
        F = np.stack([ft['x'], ft['y'], ft['z']], 1).astype(float)
        P = np.stack([cl['x'], cl['y'], cl['z']], 1).astype(float)
        cache[key] = (F, np.asarray(ft['cluster_id'], dtype=int),
                      P, np.asarray(cl['cluster_id'], dtype=int), cKDTree(P))
    return cache[key]

print('%-10s %4s %-9s %6s %8s %8s %6s %7s %7s %8s' % (
    'event', 'cl', 'arm', 'fitpts', 'fit_ext', 'chg_ext', 'cov', '>2cm', '>10cm', 'max_d'))
for line in sys.stdin.read().split('\n'):
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    run, idx, cid = line.split()[0], line.split()[1], int(line.split()[2])
    for tag in ARMS:
        F, fc, P, pc, T = load(tag, run, idx)
        C = P[pc == cid]
        X = np.unique(np.round(F[fc == cid], 2), axis=0)
        if len(X) == 0:
            print('%-10s %4d %-9s %6d %8s %8.1f %6s %7s %7s %8s' % (
                run + '/' + idx, cid, tag, 0, '-', diam(C), '0%', '-', '-', '-'))
            continue
        d, _ = T.query(X)                       # fit point -> nearest charge
        dcov, _ = cKDTree(X).query(C)           # charge point -> nearest fit point
        print('%-10s %4d %-9s %6d %8.1f %8.1f %5.0f%% %6.0f%% %6.0f%% %8.2f' % (
            run + '/' + idx, cid, tag, len(X), diam(X), diam(C),
            100 * (dcov < 2).mean(), 100 * (d > 2).mean(), 100 * (d > 10).mean(), d.max()))
    print()

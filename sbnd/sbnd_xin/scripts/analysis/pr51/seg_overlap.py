#!/usr/bin/env python3
"""doc pr/51 scoping: pairwise corridor overlap between fitted segments near
the main vertex.  For each ordered pair (A,B) of rcids with fit points within
RADIUS of the main vertex: fraction of A's fit points having a B fit point
within TOL, plus A's median dQ/dx.  High overlap + low dQ/dx = duplicated /
charge-starved path.

Usage: seg_overlap.py <ARM> <EVT> [RADIUS=15] [TOL=0.6]
"""
import sys, os, json, zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
arm, evt = sys.argv[1], int(sys.argv[2])
RAD = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
TOL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6

z = zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip'))
load = lambda n: json.loads(z.read('data/0/0-%s.json' % n))
vtx, fit = load('vertices-global'), load('track_fit-global')
vp = np.array([vtx['x'], vtx['y'], vtx['z']]).T
mv = vp[np.array(vtx['q']) == 15000][0]
fp = np.array([fit['x'], fit['y'], fit['z']]).T
fq = np.array(fit['q'], dtype=float)
fr = np.array(fit['real_cluster_id'])

sel = {}
for r in sorted(set(fr)):
    if r < 0: continue
    m = fr == r
    if np.linalg.norm(fp[m] - mv, axis=1).min() < RAD:
        sel[r] = fp[m]

print('arm=%s evt=%d rad=%.0f tol=%.1f' % (arm, evt, RAD, TOL))
rs = sorted(sel)
for a in rs:
    med = np.median(fq[fr == a])
    row = []
    for b in rs:
        if a == b: continue
        frac = np.mean([np.linalg.norm(sel[b] - p, axis=1).min() < TOL for p in sel[a]])
        if frac > 0.25:
            row.append('%d:%.0f%%' % (b, 100 * frac))
    print('  rcid %-7d npts %4d  med_dqdx %6.0f  overlaps: %s' % (a, len(sel[a]), med, ', '.join(row) if row else '-'))

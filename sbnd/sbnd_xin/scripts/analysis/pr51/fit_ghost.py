#!/usr/bin/env python3
"""doc pr/51 scoping: per-segment image support of the FITTED trajectory
(track_fit-global) near the main vertex.  For each real_cluster_id with a
fit point within RADIUS of the main vertex: point count, closest approach,
max/mean nearest-image distance, count of points >0.6cm from any image
("ghost points"), dQ/dx summary (q = dQ*0.1 - 1000 per make_pr_bee docs),
and the ghost run's endpoints.

Usage: python3 fit_ghost.py <ARM_LABEL> <EVT> [RADIUS_CM]
"""
import sys, os, json, zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
arm, evt = sys.argv[1], int(sys.argv[2])
RAD = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

z = zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip'))
def load(name):
    return json.loads(z.read('data/0/0-%s.json' % name))

vtx = load('vertices-global')
fit = load('track_fit-global')
img = load('clustering-global')

vp = np.array([vtx['x'], vtx['y'], vtx['z']]).T
mv = vp[np.array(vtx['q']) == 15000][0]

fp = np.array([fit['x'], fit['y'], fit['z']]).T
fq = np.array(fit['q'], dtype=float)
fr = np.array(fit['real_cluster_id'])
ip = np.array([img['x'], img['y'], img['z']]).T

print('arm=%s evt=%d  main vertex (%.2f,%.2f,%.2f)  radius=%.1f' % (arm, evt, *mv, RAD))
print('%-8s %5s %7s %7s %7s %6s %9s  %s' % ('rcid', 'npts', 'dmin', 'imgmax', 'imgavg', 'nghost', 'dqdx_med', 'ghost-run endpoints'))
for r in sorted(set(fr)):
    m = fr == r
    pts = fp[m]
    dd = np.linalg.norm(pts - mv, axis=1)
    if dd.min() > RAD: continue
    dn = np.array([np.linalg.norm(ip - p, axis=1).min() for p in pts])
    ghost = dn > 0.6
    dqdx = (fq[m] * 0.1 - 1000) if False else fq[m]  # raw q; MIP-scale printed separately
    med = np.median(fq[m])
    gh = ''
    if ghost.any():
        gi = np.where(ghost)[0]
        a, b = pts[gi[0]], pts[gi[-1]]
        gh = '(%.1f,%.1f,%.1f)->(%.1f,%.1f,%.1f)' % (*a, *b)
    print('%-8d %5d %7.2f %7.2f %7.2f %6d %9.0f  %s' % (
        r, m.sum(), dd.min(), dn.max(), dn.mean(), ghost.sum(), med, gh))

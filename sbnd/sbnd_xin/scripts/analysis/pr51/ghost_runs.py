#!/usr/bin/env python3
"""doc pr/51 scoping: whole-event scan of track_fit-global for GHOST RUNS --
maximal runs of >=MIN_RUN consecutive fit points (in dump order per rcid)
each farther than THR from every image point.  These are trajectory
stretches drawn over empty space, the owner's "ghost track" signature.

Usage: python3 ghost_runs.py <ARM_LABEL> <EVT> [THR_CM=0.8] [MIN_RUN=3]
"""
import sys, os, json, zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
arm, evt = sys.argv[1], int(sys.argv[2])
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
MINR = int(sys.argv[4]) if len(sys.argv) > 4 else 3

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

print('arm=%s evt=%d thr=%.2f min_run=%d  main vertex (%.2f,%.2f,%.2f)' % (arm, evt, THR, MINR, *mv))
for r in sorted(set(fr)):
    m = np.where(fr == r)[0]
    pts = fp[m]
    dn = np.array([np.linalg.norm(ip - p, axis=1).min() for p in pts])
    bad = dn > THR
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j + 1 < len(bad) and bad[j + 1]: j += 1
            if j - i + 1 >= MINR:
                a, b = pts[i], pts[j]
                seglen = np.linalg.norm(b - a)
                dv = min(np.linalg.norm(a - mv), np.linalg.norm(b - mv))
                print('  rcid %-7d run %d pts (idx %d-%d/%d)  len=%.1f cm  maxmiss=%.2f  dqdx_med=%.0f  vtx_dis=%.1f  (%.1f,%.1f,%.1f)->(%.1f,%.1f,%.1f)' % (
                    r, j - i + 1, i, j, len(bad), seglen, dn[i:j+1].max(),
                    np.median(fq[m][i:j+1]), dv, *a, *b))
            i = j + 1
        else:
            i += 1

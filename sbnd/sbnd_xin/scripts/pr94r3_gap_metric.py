#!/usr/bin/env python3
"""doc pr/94 round 3: how far does the fitted trajectory stray from any imaged
charge?  Reads one pr_evt<ID>/mabc-pr.zip and reports, over ALL track_fit
points, the distance to the nearest clustering-global (real charge) point."""
import sys, json, zipfile, io
import numpy as np
from scipy.spatial import cKDTree

def layers(zp):
    z = zipfile.ZipFile(zp)
    out = {}
    for n in z.namelist():
        for tag in ('track_fit-global', 'clustering-global'):
            if n.endswith(tag + '.json'):
                out[tag] = json.loads(z.read(n))
    return out

for zp in sys.argv[1:]:
    L = layers(zp)
    tf, st = L['track_fit-global'], L['clustering-global']
    P = np.c_[tf['x'], tf['y'], tf['z']]
    C = np.c_[st['x'], st['y'], st['z']]
    d, _ = cKDTree(C).query(P)
    rc = np.array(tf['real_cluster_id'])
    worst = ''
    if (d > 3).any():
        seg = {}
        for k in np.where(d > 3)[0]:
            seg.setdefault(int(rc[k]), []).append(d[k])
        worst = '  worst segs: ' + ', '.join(
            f'{k}({len(v)}pts,max {max(v):.1f}cm)' for k, v in
            sorted(seg.items(), key=lambda kv: -max(kv[1]))[:3])
    print(f'{zp}\n  n={len(d):4d}  >3cm={int((d>3).sum()):3d}  >5cm={int((d>5).sum()):3d}  '
          f'max={d.max():5.2f} cm  mean={d.mean():.2f} cm{worst}')

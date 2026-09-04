#!/usr/bin/env python3
"""doc pdvd/38 -- does the gap-aware end trim amputate REAL track ends that
cross a dead-channel region?

A long run of points failing the strict three-plane good-point test is what a
detached island looks like -- and it is also what a genuine dead region looks
like, because that test allows no bad plane.  This counts how much of what the
trim removes sits inside a dead area.

Arm A is the no-gap-trim arm, arm B the gap-trim arm; a point is REMOVED if it
is more than 2 cm from every point of B's fit for the same cluster.  Dead areas
come from the Bee `channel-deadarea-apa*-face*` layers of A's own zip: they are
(y, z) polygons per TPC.  A point is counted as in a dead area when it falls
inside any polygon of any TPC, which is an UPPER BOUND -- the layers carry no x
extent, so a dead region of one drift volume also shadows the other.  Read a
small number as decisive ("the trim is not eating dead regions") and a large one
as a cue to look per TPC.

Usage:
  cd <repo>/pdvd
  python3 docs/nf_sp_img_clus/scripts/d38_deadarea_census.py d38off d38g3 out.tsv
"""
import glob, json, os, sys, zipfile
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from multiprocessing import Pool
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # <repo>/pdvd
W = os.path.join(PDVD, 'work')
A, B, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

def dead_paths(z):
    out = []
    for n in z.namelist():
        if 'channel-deadarea' not in n:
            continue
        j = json.loads(z.read(n))
        for poly in j.get('polygons', []):
            if len(poly) >= 3:
                out.append(MplPath(np.asarray(poly, dtype=float)))
    return out

def one(base):
    run, idx = base.rsplit('_', 1)
    pa = os.path.join(W, base + '_' + A, 'mabc-pr.zip')
    pb = os.path.join(W, base + '_' + B, 'mabc-pr.zip')
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    za, zb = zipfile.ZipFile(pa), zipfile.ZipFile(pb)
    fa = json.loads(za.read('data/0/0-stm_fit-global.json'))
    fb = json.loads(zb.read('data/0/0-stm_fit-global.json'))
    cl = json.loads(za.read('data/0/0-clustering-global.json'))
    P = np.stack([cl['x'], cl['y'], cl['z']], 1).astype(float)
    T = cKDTree(P)
    polys = dead_paths(za)
    FA = np.stack([fa['x'], fa['y'], fa['z']], 1).astype(float)
    FB = np.stack([fb['x'], fb['y'], fb['z']], 1).astype(float)
    ca = np.asarray(fa['cluster_id'], dtype=int)
    cb = np.asarray(fb['cluster_id'], dtype=int)
    rows = []
    for cid in np.unique(ca):
        XA = np.unique(np.round(FA[ca == cid], 2), axis=0)
        XB = np.unique(np.round(FB[cb == cid], 2), axis=0) if (cb == cid).any() \
             else np.zeros((0, 3))
        if len(XA) == 0:
            continue
        rem = XA if len(XB) == 0 else XA[cKDTree(XB).query(XA)[0] > 2.0]
        if len(rem) == 0:
            continue
        d, _ = T.query(rem)
        yz = rem[:, 1:3]
        indead = np.zeros(len(rem), dtype=bool)
        for pth in polys:
            indead |= pth.contains_points(yz)
        rows.append([run, idx, int(cid), len(XA), len(XB), len(rem),
                     float((d > 2).mean()), float((d > 10).mean()),
                     float(indead.mean()), int((indead & (d > 2)).sum())])
    return rows

bases = sorted({os.path.basename(d)[:-(len(A)+1)] for d in glob.glob(os.path.join(W, '*_'+A))})
tot = ind = offc = 0
with Pool(16) as pool, open(OUT, 'w') as f:
    f.write('\t'.join(['run', 'idx', 'cid', 'nA', 'nB', 'removed',
                       'rem_f2', 'rem_f10', 'rem_in_dead', 'rem_dead_and_off']) + '\n')
    for rows in pool.imap_unordered(one, bases):
        if not rows:
            continue
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
            tot += r[5]; ind += r[5] * r[8]; offc += r[5] * r[6]
if tot:
    print('removed points: %d;  %.1f %% are >2 cm from charge;  %.1f %% fall inside a '
          'dead-area polygon (upper bound, see the docstring)' % (tot, 100*offc/tot, 100*ind/tot))
else:
    print('no removed points')

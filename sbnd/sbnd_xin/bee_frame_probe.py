#!/usr/bin/env python3
"""doc pr/13: which coordinate frame does each Bee layer live in?

Read-only.  Answers, from the shipped zips alone, why `img-global` does not sit
on top of `shower_track-global` / `track_fit-global` in the pr/12 Bee sets.

Frames in play (SBND, reality='data'):
  raw        (x, y, z)                    -- drift-time x, uncorrected y,z
  corrected  (x_t0cor, y_cor, z_cor)      -- per-cluster T0 drift correction plus
                                             the per-TPC transverse pos_offset
                                             (cfg/.../sbnd/clus.jsonnet:70-74)

Sections (all four are independent tests; -s picks a subset):
  A  frame fingerprint -- exact coordinate overlap of every PR layer with the
     Q/L zip's img-global (raw) and clustering-global (corrected).  Works only
     for layers that re-emit existing point-cloud points (shower_track).
  B  img <-> clustering delta -- points matched by their unique q value, so the
     two frames are compared point-by-point: per-TPC (dy,dz) and per-cluster dx.
  C  track_fit frame, paired residual -- fitted points are NEW points, so A is
     unavailable.  For each fitted point take the centroid of the charge within
     R cm and subtract; do it against BOTH clouds.  Against the cloud the fit
     was made from the residual is ~0 with no step at the cathode; against the
     other frame it carries the pos_offset sign flip.  The paired form removes
     the estimator's bias.
  D  Q/L clustering-global vs PR clustering-global -- does cluster_t0 survive
     the pctree tarball into the PR job?

Usage:
  python3 bee_frame_probe.py -q work-mcp1kall-d59k -p work-mcp1kall-cath01 \
      -e 280972 -e 292384 -e 407280
  python3 bee_frame_probe.py -q ... -p ... --index docs/pr/cath_spanned.index.txt -n 20 -s C
"""
import argparse
import collections
import json
import os
import zipfile

import numpy as np
from scipy.spatial import cKDTree

XYZ = ('x', 'y', 'z')


def layer(path, name, idx=0):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read(f'data/{idx}/{idx}-{name}.json'))


def pts(d):
    return np.column_stack([np.asarray(d[k], dtype=float) for k in XYZ])


def ql_zip(root, evt):
    return os.path.join(root, f'ql_evt{evt}', 'mabc-all-apa.zip')


def pr_zip(root, evt):
    return os.path.join(root, f'pr_evt{evt}', 'mabc-pr.zip')


def qmatch(a, b):
    """Index pairs (ib, ia) for points whose q value is unique in both dumps."""
    qa, qb = np.asarray(a['q']), np.asarray(b['q'])
    ca, cb = collections.Counter(qa.tolist()), collections.Counter(qb.tolist())
    ia = {q: j for j, q in enumerate(qa.tolist()) if ca[q] == 1}
    m = [(j, ia[q]) for j, q in enumerate(qb.tolist()) if cb[q] == 1 and q in ia]
    return np.array([p[0] for p in m], int), np.array([p[1] for p in m], int)


def keyset(P):
    return set(map(tuple, np.round(P, 4).tolist()))


# --------------------------------------------------------------------- A
def sec_a(q, p, evt):
    img, clus = pts(layer(q, 'img-global')), pts(layer(q, 'clustering-global'))
    ki, kc = keyset(img), keyset(clus)
    ti, tc = cKDTree(img), cKDTree(clus)
    print(f'  [A] evt {evt}: img {len(img)} pts, clustering {len(clus)} pts')
    for name in ('shower_track-global', 'track_fit-global', 'vertices-global'):
        try:
            P = pts(layer(p, name))
        except KeyError:
            continue
        k = keyset(P)
        # Exact match settles the frame for layers that re-emit existing points;
        # the NN median catches the ones that only miss on 4-decimal rounding.
        print(f'      {name:22s} {len(k):6d} uniq'
              f'  exact-in-img {len(k & ki):6d}  exact-in-clustering {len(k & kc):6d}'
              f'  |  NN med to img {np.median(ti.query(P)[0]):.4f}'
              f'  to clustering {np.median(tc.query(P)[0]):.4f}')


# --------------------------------------------------------------------- B
def sec_b(q, p, evt):
    a, b = layer(q, 'img-global'), layer(q, 'clustering-global')
    jb, ja = qmatch(a, b)
    A, B = pts(a)[ja], pts(b)[jb]
    d = B - A
    print(f'  [B] evt {evt}: {len(jb)} of {len(a["q"])} img points matched by unique q')
    for lab, sel in (('TPC0 x<0', A[:, 0] < 0), ('TPC1 x>=0', A[:, 0] >= 0)):
        if sel.sum() == 0:
            continue
        print(f'      {lab}: n={sel.sum():6d}  dy {np.median(d[sel,1]):+.3f}'
              f'  dz {np.median(d[sel,2]):+.3f}   (dx is per cluster, below)')
    cid = np.asarray(a['cluster_id'])[ja]
    rows = []
    for c in np.unique(cid):
        s = cid == c
        if s.sum() < 50:
            continue
        rows.append((int(c), int(s.sum()), float(np.median(d[s, 0]))))
    rows.sort(key=lambda r: -r[1])
    print('      img_cluster   npts   median dx (cm)')
    for c, n, dx in rows:
        print(f'      {c:10d} {n:7d} {dx:12.2f}')
    dxs = np.array([r[2] for r in rows])
    w = np.array([r[1] for r in rows], float)
    near = np.abs(dxs) < 1
    print(f'      clusters >=50 pts: {len(rows)}   |dx|<1cm: {near.sum()}'
          f'   |dx|>10cm: {int((np.abs(dxs) > 10).sum())}'
          f'   points with |dx|<1cm: {w[near].sum()/w.sum():.0%}')


# --------------------------------------------------------------------- C
def sec_c(q, p, evt, R, agg):
    raw = pts(layer(q, 'img-global'))
    cor = pts(layer(p, 'clustering-global'))
    F = pts(layer(p, 'track_fit-global'))
    for lab, C in (('raw img', raw), ('corrected clustering', cor)):
        t = cKDTree(C)
        for pt in F:
            nb = t.query_ball_point(pt, R)
            if len(nb) >= 10:
                agg[(lab, 'x<0' if pt[0] < 0 else 'x>0')].append(C[nb].mean(axis=0) - pt)
    print(f'  [C] evt {evt}: {len(F)} fitted points folded in')


def sec_c_report(agg):
    print('\n  [C] local-charge centroid minus fitted point (median, cm)')
    for lab in ('raw img', 'corrected clustering'):
        out = {}
        for side in ('x<0', 'x>0'):
            v = agg[(lab, side)]
            if not v:
                continue
            m = np.median(np.array(v), axis=0)
            out[side] = m
            print(f'      vs {lab:22s} {side}  n={len(v):6d}'
                  f'  ({m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f})')
        if len(out) == 2:
            j = out['x>0'] - out['x<0']
            print(f'      vs {lab:22s} step across the cathode:'
                  f'  dy {j[1]:+.3f}  dz {j[2]:+.3f}')


# --------------------------------------------------------------------- D
def sec_d(q, p, evt):
    a, b = layer(q, 'clustering-global'), layer(p, 'clustering-global')
    jb, ja = qmatch(a, b)
    d = pts(b)[jb] - pts(a)[ja]
    moved = np.abs(d).max(axis=1) > 1e-3
    print(f'  [D] evt {evt}: QL {len(a["q"])} pts vs PR {len(b["q"])} pts;'
          f' {len(jb)} matched, {int(moved.sum())} differ by >1e-3 cm;'
          f' max |delta| = {np.abs(d).max():.4f} cm')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-q', '--ql-root', required=True)
    ap.add_argument('-p', '--pr-root', required=True)
    ap.add_argument('-e', '--event', action='append', default=[])
    ap.add_argument('--index', help='cath_*.index.txt; takes the event column')
    ap.add_argument('-n', '--nmax', type=int, default=0)
    ap.add_argument('-R', type=float, default=2.0, help='section C centroid radius (cm)')
    ap.add_argument('-s', '--sections', default='ABCD')
    args = ap.parse_args()

    evts = list(args.event)
    if args.index:
        evts += [l.split()[1] for l in open(args.index) if not l.startswith('#')]
    if args.nmax:
        evts = evts[:args.nmax]

    agg = collections.defaultdict(list)
    for evt in evts:
        qz, pz = ql_zip(args.ql_root, evt), pr_zip(args.pr_root, evt)
        if not (os.path.exists(qz) and os.path.exists(pz)):
            print(f'  -- evt {evt}: missing zip, skipped')
            continue
        if 'A' in args.sections:
            sec_a(qz, pz, evt)
        if 'B' in args.sections:
            sec_b(qz, pz, evt)
        if 'C' in args.sections:
            sec_c(qz, pz, evt, args.R, agg)
        if 'D' in args.sections:
            sec_d(qz, pz, evt)
    if 'C' in args.sections and agg:
        sec_c_report(agg)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""doc pr/149 Phase 0: is charge_stepped's all-wire rule even reachable on SBND?

ChargeStepped (clus/src/BlobSampler.cxx:1200) samples EVERY wire of the max and
min views only when N_max * N_min <= max_wire_product_threshold (C++ 2500).
Above it every candidate is a stepped "must" wire and the point set equals
Stepped's, merely reordered.  Its non-stepped wires additionally need charge
>= charge_threshold_* (C++ 4000) in all three planes.  So before spending an arm
we measure, on the production stage-A point-cloud trees (work-<s>-d102m), how
the blob wire products and the per-cell charges sit against those two cuts,
split by the cluster's drift angle.  The charge column is the smallest nonzero
of the three wire charges at each stepped sample point of the cluster.

PROXY, stated: the PR retile (ImproveCluster_2) re-tiles each cluster from its
own 2-D activity, so the blobs the retile sampler sees are not byte-for-byte the
imaging blobs read here.  The direct measurement is the Steiner point count in
the Stage 1 arms; this table only says whether the cuts are in range.

Blob wire bounds are half-open [min, max) (Facade_Blob.cxx:156, M7), so a view's
width is max - min, the same swidth() the sampler computes.

Tree walk (as pr20_wasmain_check.py / unmerge_crosser_audit.py): the lpcmaps
arrays are per-NODE row counts in node order; a nonzero cluster_scalar count
opens the next cluster, the scalar (blob) and 3d counts that follow are its.

theta_drift = angle between the cluster's principal axis (PCA of its 3d points)
and the drift axis x; 90 deg = isochronous.  "lin" = the leading eigenvalue
carries >= 0.8 of the variance (a track-like cluster).

Usage:
  blob_product_census.py --tag d102m --samples nuecc48 ncpi0 [--max-events N]
      [--tsv out.tsv]
Paths are relative to sbnd_xin (run from there).
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile

import numpy as np

BINS = [0, 30, 50, 65, 75, 85, 90.01]
PRODUCTS = [2500, 6400, 10000]
QCUTS = [2000, 4000, 6000]
MIN_PTS = 20


def load_tensors(fname):
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if f is None:
                continue
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    return {md['datapath']: (md, arrays.get(b))
            for b, md in metas.items() if 'datapath' in md}


def clusters_of(fname):
    """Yield dict per live cluster: ident, theta, lin, length, npts, blob widths,
    min-nonzero wire charge per sampled point."""
    bp = load_tensors(fname)
    live = [p for p in bp if re.fullmatch(r'pointtrees/\d+/live', p)]
    if len(live) != 1:
        raise RuntimeError(f'{fname}: live trees {live}')
    md = bp[live[0]][0]
    items = bp[md['pointclouds']][0]['items']
    lpc = bp[md['lpcmaps']][0]['arrays']

    def arr(pc, a):
        ds = bp[items[pc]][0]['arrays']
        return bp[ds[a]][1]

    m_cs = bp[lpc['cluster_scalar']][1].astype(int)
    m_bl = bp[lpc['scalar']][1].astype(int)
    m_3d = bp[lpc['3d']][1].astype(int)
    ident = arr('cluster_scalar', 'ident').astype(int)
    wmin = {p: arr('scalar', f'{p}_wire_index_min').astype(int) for p in 'uvw'}
    wmax = {p: arr('scalar', f'{p}_wire_index_max').astype(int) for p in 'uvw'}
    X, Y, Z = (arr('3d', c).astype(float) for c in 'xyz')

    # Per-point wire charges at the stepped sample points (the activity value the
    # sampler's get_wire_charge reads).  The ctpc PCs sit at the grouping node with
    # a cident that is not the cluster ident, so they cannot be split per cluster.
    QU, QV, QW = (arr('3d', f'{p}charge_val').astype(float) for p in 'uvw')

    spans_b, spans_p = [], []
    ci, brow, prow = -1, 0, 0
    for n in range(len(m_cs)):
        if m_cs[n]:
            ci += 1
            spans_b.append([])
            spans_p.append([])
        nb, np_ = int(m_bl[n]), int(m_3d[n])
        if ci >= 0:
            spans_b[ci].extend(range(brow, brow + nb))
            spans_p[ci].extend(range(prow, prow + np_))
        brow += nb
        prow += np_
    if brow != len(wmin['u']) or prow != len(X):
        raise RuntimeError(f'{fname}: walk mismatch blobs {brow}/{len(wmin["u"])} pts {prow}/{len(X)}')

    for k in range(len(spans_b)):
        pr = np.array(spans_p[k], dtype=int)
        br = np.array(spans_b[k], dtype=int)
        if len(pr) < MIN_PTS or len(br) == 0:
            continue
        P = np.stack([X[pr], Y[pr], Z[pr]], axis=1)
        P = P - P.mean(axis=0)
        w, v = np.linalg.eigh(P.T @ P / len(P))
        ax = v[:, -1]
        theta = float(np.degrees(np.arccos(min(1.0, abs(ax[0])))))
        lin = float(w[-1] / max(w.sum(), 1e-12))
        proj = P @ ax
        length = float(proj.max() - proj.min())
        widths = np.stack([wmax[p][br] - wmin[p][br] for p in 'uvw'], axis=1)
        # ChargeStepped drops a non-must pair when ANY of the three wire charges is
        # nonzero and below the cut (a zero reads as dead and is let through when
        # disable_mix_dead_cell is false).  q = the smallest NONZERO charge of the
        # three at each stepped point; all-zero points are excluded.
        Q = np.stack([QU[pr], QV[pr], QW[pr]], axis=1)
        Qm = np.where(Q > 0, Q, np.inf).min(axis=1)
        q = Qm[np.isfinite(Qm)]
        yield dict(ident=int(ident[k]), theta=theta, lin=lin, length=length,
                   npts=len(pr), widths=widths, q=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='d102m')
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--max-events', type=int, default=0)
    ap.add_argument('--tsv')
    ap.add_argument('--min-len', type=float, default=10.0, help='cm, cluster extent floor')
    a = ap.parse_args()

    nb = len(BINS) - 1
    acc = {sel: dict(nblob=np.zeros(nb), prod=np.zeros((nb, len(PRODUCTS))),
                     ncl=np.zeros(nb), ncell=np.zeros(nb),
                     qlow=np.zeros((nb, len(QCUTS))), prodvals=[[] for _ in range(nb)],
                     qvals=[[] for _ in range(nb)])
           for sel in ('all', 'lin')}
    nev = 0
    for s in a.samples:
        files = sorted(glob.glob(f'work-{s}-{a.tag}/ql_evt*/pctree-evt*.tar.gz'))
        if a.max_events:
            files = files[:a.max_events]
        for f in files:
            nev += 1
            for c in clusters_of(f):
                if c['length'] < a.min_len:
                    continue
                b = int(np.searchsorted(BINS, c['theta'], side='right') - 1)
                b = min(max(b, 0), nb - 1)
                srt = np.sort(c['widths'], axis=1)
                prod = srt[:, 2] * srt[:, 0]          # N_max * N_min
                for sel in ('all', 'lin'):
                    if sel == 'lin' and c['lin'] < 0.8:
                        continue
                    A = acc[sel]
                    A['ncl'][b] += 1
                    A['nblob'][b] += len(prod)
                    for j, t in enumerate(PRODUCTS):
                        A['prod'][b, j] += int((prod <= t).sum())
                    A['prodvals'][b].append(prod)
                    q = c['q']
                    A['ncell'][b] += len(q)
                    for j, t in enumerate(QCUTS):
                        A['qlow'][b, j] += int((q < t).sum())
                    A['qvals'][b].append(q)

    rows = []
    hdr = ['sel', 'theta_bin', 'n_clusters', 'n_blobs', 'prod_p50', 'prod_p90'] + \
          [f'frac_prod_le_{t}' for t in PRODUCTS] + ['n_pts', 'qmin_p10', 'qmin_p50'] + \
          [f'frac_q_lt_{t}' for t in QCUTS]
    for sel in ('all', 'lin'):
        A = acc[sel]
        for b in range(nb):
            pv = np.concatenate(A['prodvals'][b]) if A['prodvals'][b] else np.zeros(0)
            qv = np.concatenate(A['qvals'][b]) if A['qvals'][b] else np.zeros(0)
            nbl = max(A['nblob'][b], 1)
            ncl = max(A['ncell'][b], 1)
            rows.append([sel, f'{BINS[b]:g}-{min(BINS[b+1],90):g}', int(A['ncl'][b]), int(A['nblob'][b]),
                         f'{np.percentile(pv,50):.0f}' if len(pv) else 'nan',
                         f'{np.percentile(pv,90):.0f}' if len(pv) else 'nan'] +
                        [f'{A["prod"][b,j]/nbl:.3f}' for j in range(len(PRODUCTS))] +
                        [int(A['ncell'][b]),
                         f'{np.percentile(qv,10):.0f}' if len(qv) else 'nan',
                         f'{np.percentile(qv,50):.0f}' if len(qv) else 'nan'] +
                        [f'{A["qlow"][b,j]/ncl:.3f}' for j in range(len(QCUTS))])
    out = sys.stdout if not a.tsv else open(a.tsv, 'w')
    print(f'# blob_product_census tag={a.tag} samples={" ".join(a.samples)} events={nev} '
          f'min_len={a.min_len}cm min_pts={MIN_PTS}', file=out)
    print('\t'.join(hdr), file=out)
    for r in rows:
        print('\t'.join(str(x) for x in r), file=out)
    if a.tsv:
        out.close()
        print(open(a.tsv).read())


if __name__ == '__main__':
    main()

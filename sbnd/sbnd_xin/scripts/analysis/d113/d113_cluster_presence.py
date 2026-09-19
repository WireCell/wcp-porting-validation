#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 5 -- per-event comparison of a fix arm's stage-A pctree with production: blobs, clusters,
points, dead_winds rows, and the share of each production cluster's points still present (within 10 mm) in the
arm, over the events of the given groups.

Usage: d113_cluster_presence.py --sample mcp1k --arm d113cfqpg64 [--arm d113cfqpg64dg] --groups 4,5,...
"""
import argparse, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C


def pts(sample, arm, ev):
    pt = C.PCTree(f'{C.SX}/work-{sample}-{arm}/ql_evt{ev}/pctree-evt{ev}.tar.gz', ev)
    P = f'pointtrees/{ev}/live/pointclouds/namedpcs/3d/arrays/'
    x, y, z = [pt.bp[P + k][1] for k in 'xyz']
    m3 = pt.bp[f'pointtrees/{ev}/live/lpcmaps/arrays/3d'][1]; mcs = pt.bp[f'pointtrees/{ev}/live/lpcmaps/arrays/cluster_scalar'][1]
    cl = np.zeros(len(x), int); ci = -1; off = 0
    for i in range(len(m3)):
        if mcs[i]:
            ci += 1
        n = int(m3[i]); cl[off:off + n] = ci; off += n
    dw = sum(len(v[1]) for k, v in pt.bp.items() if 'dead_winds' in k and k.endswith('/arrays/wind'))
    return np.c_[x, y, z], cl, len(pt.b_smin), len(pt.cl_ident), dw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='mcp1k')
    ap.add_argument('--arm', action='append', required=True)
    ap.add_argument('--groups', required=True)
    ap.add_argument('--base', default='d102m')
    a = ap.parse_args()
    evs = []
    for g in a.groups.split(','):
        evs += C.group_events(f'{C.SX}/work-{a.sample}-{a.base}/g{g}')
    for arm in a.arm:
        R = []; lost_big = 0; nclus_big = 0
        for ev in evs:
            if not os.path.exists(f'{C.SX}/work-{a.sample}-{arm}/ql_evt{ev}/pctree-evt{ev}.tar.gz'):
                continue
            A, ca, nbA, ncA, dwA = pts(a.sample, a.base, ev); B, cb, nbB, ncB, dwB = pts(a.sample, arm, ev)
            d, _ = cKDTree(B).query(A); d2, _ = cKDTree(A).query(B)
            for c in range(ca.max() + 1):
                m = ca == c
                if m.sum() >= 200:
                    nclus_big += 1; lost_big += (np.mean(d[m] < 10) < 0.5)
            R.append((nbA, nbB, ncA, ncB, dwA, dwB, np.mean(d < 10), np.mean(d2 < 10), len(A), len(B)))
        R = np.array(R, float)
        print(f'{arm} vs {a.base}: {len(R)} events; blobs {np.median(R[:,0]):.0f} -> {np.median(R[:,1]):.0f} (ratio median {np.median(R[:,1]/R[:,0]):.2f}); '
              f'clusters {np.median(R[:,2]):.0f} -> {np.median(R[:,3]):.0f}; dead_winds rows {np.median(R[:,4]):.0f} -> {np.median(R[:,5]):.0f}; '
              f'points {np.median(R[:,8]):.0f} -> {np.median(R[:,9]):.0f}; production points still present (10 mm) median {np.median(R[:,6]):.3f}; '
              f'arm points that are new {1-np.median(R[:,7]):.3f}; production clusters >= 200 pts lost (< 50 % present): {lost_big} / {nclus_big}')


if __name__ == '__main__':
    main()

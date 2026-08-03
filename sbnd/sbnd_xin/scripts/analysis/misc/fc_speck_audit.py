#!/usr/bin/env python3
"""Audit why a main cluster does (or does not) pass the FC containment check.

Usage:
    fc_speck_audit.py <pctree.tar.gz> <event> <cluster_ident>

Reports, for the named cluster:
  1. grid-connectivity fragments (3 cm cells) with each fragment's gap to the
     nearest detector wall and how many of its points fall outside the inset
     fiducial volume;
  2. the 8 extreme points get_extreme_wcps() would hand cluster_fc_check
     (2 along the PCA main axis + 6 coordinate extremes), each tested against
     BOTH fiducial definitions in play.

Why both volumes.  cluster_fc_check's direct containment test is
FiducialUtils' when the caller passes no IFiducial -- TaggerCheckNeutrino's
match_isFC does exactly that -- and the caller's IFiducial + fv_tolerance
otherwise, which is what SBND's tagger_check_fc is given (cfg
pgrapher/experiment/sbnd/clus.jsonnet, sbnd_pr_fv + sbnd_pr_fv_margins, the
same objects tagger_check_tgm gets).  The two disagree in a ~3 cm shell at
every wall, so a fragment parked on a wall can be contained for one and an
exiter for the other.  See sbnd_xin/docs/49_stm-containment-fv-inconsistency.md
and docs/27_fc-tgm-consistent-fv.md.

Run it on the PRE-tagger (QL-stage) tarball and on the post-tagger one: the
interesting failures are grafts that unmerge_bundle strips out in between, so
they are invisible in the post-tagger tree the taggers actually saw.

Repro (SBND data run 18255, the two events colleagues flagged as not-FC):
    python3 fc_speck_audit.py work-nuecc48-nuf/ql_evt46363/pctree-evt46363.tar.gz 46363 19
    python3 fc_speck_audit.py work-nuecc48-nuf/ql_evt163543/pctree-evt163543.tar.gz 163543 14
"""
import sys

import numpy as np

from inspect_pctree import load

# FiducialUtils' volume: DetectorVolumes::contained(), the union of the
# per-(apa,face) IAnodeFace::sensitive() boxes -- no margin anywhere, and the
# CPA slab |x| < 0.45 cm is a hole between the two per-face boxes.
FU = dict(x=201.45, y=199.965, z=(0.0, 501.0), cpa=0.45)
# sbnd_pr_fv (one box spanning both TPCs) + sbnd_pr_fv_margins
# (x 2.5, y 3, z_hi 5, z_lo 3 cm insets).
FV = dict(x=201.05 - 2.5, y=199.312 - 3.0, z=(0.85 + 3.0, 500.15 - 5.0))


def in_fu(p):
    return (abs(p[0]) <= FU['x'] and abs(p[0]) >= FU['cpa']
            and abs(p[1]) <= FU['y'] and FU['z'][0] <= p[2] <= FU['z'][1])


def in_fv(p):
    return (abs(p[0]) <= FV['x'] and abs(p[1]) <= FV['y']
            and FV['z'][0] <= p[2] <= FV['z'][1])


def cluster_points(tar, evt, ident):
    """(N,3) array of the cluster's 3d points in cm, T0-corrected coords."""
    b = load(tar)
    L = f'pointtrees/{evt}/live/'
    cs = b[L + 'lpcmaps/arrays/cluster_scalar'][1]
    m3 = b[L + 'lpcmaps/arrays/3d'][1]
    ids = b[L + 'pointclouds/namedpcs/cluster_scalar/arrays/ident'][1]
    pc = L + 'pointclouds/namedpcs/3d/arrays/'
    x, y, z = b[pc + 'x_t0cor'][1], b[pc + 'y_cor'][1], b[pc + 'z_cor'][1]
    off = np.concatenate([[0], np.cumsum(m3)])
    cn = np.where(cs > 0)[0]
    k = int(np.where(ids == ident)[0][0])
    lo = off[cn[k]]
    hi = off[cn[k + 1]] if k + 1 < len(cn) else off[-1]
    return np.stack([x[lo:hi], y[lo:hi], z[lo:hi]], 1) / 10.0   # mm -> cm


def fragments(P, cell=3.0):
    """Union-find over occupied `cell`-cm grid cells and their 26 neighbours."""
    key = np.floor(P / cell).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    idx = {tuple(k): i for i, k in enumerate(map(tuple, uniq))}
    parent = list(range(len(uniq)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nb = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for k, i in idx.items():
        for d in nb:
            j = idx.get((k[0] + d[0], k[1] + d[1], k[2] + d[2]))
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    lab = np.array([find(i) for i in range(len(uniq))])[inv]
    out = [(int((lab == r).sum()), P[lab == r]) for r in np.unique(lab)]
    out.sort(key=lambda t: -t[0])
    return out


def extreme_points(P):
    """The 8 points get_extreme_wcps() picks (Facade_Cluster.cxx:3042)."""
    axis = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
    if axis[1] < 0:
        axis = -axis
    proj = P @ axis
    return axis, {
        'main-hi': int(proj.argmax()), 'main-lo': int(proj.argmin()),
        'y-hi': int(P[:, 1].argmax()), 'y-lo': int(P[:, 1].argmin()),
        'z-hi': int(P[:, 2].argmax()), 'z-lo': int(P[:, 2].argmin()),
        'x-hi': int(P[:, 0].argmax()), 'x-lo': int(P[:, 0].argmin()),
    }


def main(tar, evt, ident):
    P = cluster_points(tar, evt, ident)
    print(f'{tar}\nevt {evt} cluster {ident}: {len(P)} points')

    print('\nfragments (3 cm grid connectivity):')
    for n, F in fragments(P):
        gaps = (FU['x'] - np.abs(F[:, 0]).max(), FU['y'] - np.abs(F[:, 1]).max(),
                FU['z'][1] - F[:, 2].max(), F[:, 2].min() - FU['z'][0])
        nout = sum(not in_fv(p) for p in F)
        print(f'  n={n:6d}  x[{F[:,0].min():7.1f},{F[:,0].max():7.1f}]'
              f' y[{F[:,1].min():7.1f},{F[:,1].max():7.1f}]'
              f' z[{F[:,2].min():7.1f},{F[:,2].max():7.1f}]'
              f'   wall gaps |x|{gaps[0]:6.2f} |y|{gaps[1]:6.2f} zhi{gaps[2]:6.1f} zlo{gaps[3]:6.1f}'
              f'   outside inset FV: {nout}')

    axis, sel = extreme_points(P)
    print(f'\n8 extreme points (PCA main axis {np.round(axis, 3)}):')
    for name, i in sel.items():
        p = P[i]
        print(f'  {name:8s} ({p[0]:8.2f},{p[1]:8.2f},{p[2]:8.2f})'
              f'   FiducialUtils: {"in " if in_fu(p) else "OUT"}'
              f'   sbnd_pr_fv+margins: {"in " if in_fv(p) else "OUT"}')
    print('\n(An OUT in either column means cluster_fc_check\'s direct test fails on\n'
          ' that extreme-point group => exit_wcps non-empty => is_fc=false.)')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

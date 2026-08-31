#!/usr/bin/env python3
'''doc pr/112 -- does reading the SCN score field by SPATIAL POOLING instead of
a hard argmax buy stability, and does the stability buy accuracy?

pr/111 sec 10: "the direction with actual headroom is the voxelization and the
argmax readout -- a 0.5 cm lattice pinned to a floating origin, read out by a
single hard argmax".  pr/111 sec 7 measured why: an iid sigma = 0.05 cm jitter
relocates that argmax > 2 cm in 30 % of draws.  A neighbourhood-integrated
score cannot move discontinuously the way a winner-take-all argmax can, so it
should be stable by construction.  This measures both halves:

  (1) STABILITY  -- under the same pr/111 P1 jitter, how far does the winning
      CANDIDATE move when chosen by pooled score vs by snapped argmax?
  (2) ACCURACY   -- target-hit (pr/106 metric) for each readout.

Caveat stated up front, not buried: a pooled score is NOT what the net was
trained to emit (uboone-dl-vtx train1.py regresses a Gaussian-in-distance
target per voxel and the deployed selector consumes 1000*dl_score against
min_accept).  Pooling therefore changes the absolute scale the acceptance
threshold is calibrated on -- exactly the failure mode that killed ft2u live
at -40/473 after it looked fine on rank-based metrics (pr/79 sec 3).  So this
is a READOUT probe scored on RANK only; any deployment would have to clear
calib_guard.py first.  Nothing here is a proposal to ship.

Usage: ./pr112_pool.py --n 8 --tsv runs/pr112-pool-nuecc48.tsv
'''
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), 'dl_vtx_training')
sys.path.insert(0, TRAIN)
sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/toolkit/pyutil/python')
from scn_vtx import io as vio      # noqa: E402

WEIGHTS = ('/nfs/data/1/xqian/toolkit-dev/wire-cell-data/uboone/scn_vtx/'
           't48k-m16-l5-lr5d-res0.5-CP24.pth')
ARM = 'work-vtx106-harv-base-nuecc48'
TAGS = ['vtxscan-harv3-nuecc48']
RADII = (0.5, 1.0, 1.5, 2.0, 3.0)
SEED = 20260823


def n_vox(x, y, z, res=0.5):
    c = np.stack((x, y, z), axis=1)
    c = ((c - c.min(axis=0)) / res).astype(np.int64)
    return len(np.unique(c, axis=0))


def field(x, y, z, q):
    '''full per-voxel score map: (N,4) x,y,z,score.  top_k = every voxel.'''
    import SCN_Vertex as sv
    k = n_vox(x, y, z)
    raw = sv.SCN_Vertex(WEIGHTS, x.tobytes(), y.tobytes(), z.tobytes(),
                        q.tobytes(), dtype='float32', top_k=int(k))
    return np.frombuffer(raw, np.float32).reshape(-1, 4).astype(float)


def pooled(fld, xyz, R):
    '''per-candidate pooled response: max of the field within R cm.'''
    out = np.zeros(len(xyz))
    for i, p in enumerate(xyz):
        m = np.linalg.norm(fld[:, :3] - p, axis=1) <= R
        out[i] = fld[m, 3].max() if m.any() else -1.0
    return out


def snapped(fld, xyz):
    '''the production readout, rank-only: the global argmax snapped to the
    nearest candidate (NeutrinoVertexFinder.cxx:4472-4474 keeps the higher
    score per candidate; with a single global peak this is that peak).'''
    j = int(np.argmax(fld[:, 3]))
    return int(np.argmin(np.linalg.norm(xyz - fld[j, :3], axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--tsv', default=None)
    a = ap.parse_args()
    root = vio.default_sbnd_root()

    hit = {('snap', 0.0): 0}
    hit.update({('pool', R): 0 for R in RADII})
    moved = {('snap', 0.0): []}
    moved.update({('pool', R): [] for R in RADII})
    rows, n = [], 0

    for lab in vio.iter_labels(root, TAGS):
        e = int(lab['eventNo'])
        p = os.path.join(root, ARM, 'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
        if not os.path.exists(p):
            continue
        sb = (vio.load_calib(p).get('vertex_scoreboard') or {})
        c = sb.get('hv_cloud')
        if not c or not int(c.get('n_vertex_rows', 0)):
            continue
        nv = int(c['n_vertex_rows'])
        ids = c['vertex_ids'][:nv]
        xyz = np.array([c['x'][:nv], c['y'][:nv], c['z'][:nv]], float).T
        tr = np.asarray(lab['truth_xyz'], float)
        target = ids[int(np.argmin(np.linalg.norm(xyz - tr, axis=1)))]
        x = np.array(c['x'], np.float32); y = np.array(c['y'], np.float32)
        z = np.array(c['z'], np.float32); q = np.array(c['q'], np.float32)

        f0 = field(x, y, z, q)
        base = {('snap', 0.0): snapped(f0, xyz)}
        for R in RADII:
            base[('pool', R)] = int(np.argmax(pooled(f0, xyz, R)))
        for k, ci in base.items():
            hit[k] += int(ids[ci] == target)

        # stability of the CANDIDATE choice under the pr/111 P1 null
        rng = np.random.default_rng(SEED + e)
        picks = {k: [] for k in base}
        for _ in range(a.n):
            jx = (x + rng.normal(0, a.sigma, x.shape)).astype(np.float32)
            jy = (y + rng.normal(0, a.sigma, y.shape)).astype(np.float32)
            jz = (z + rng.normal(0, a.sigma, z.shape)).astype(np.float32)
            fj = field(jx, jy, jz, q)
            picks[('snap', 0.0)].append(snapped(fj, xyz))
            for R in RADII:
                picks[('pool', R)].append(int(np.argmax(pooled(fj, xyz, R))))
        for k, ci in base.items():
            moved[k].append(float(np.mean([pk != ci for pk in picks[k]])))
        rows.append(dict(evt=e, target=target,
                         snap=ids[base[('snap', 0.0)]],
                         pool1=ids[base[('pool', 1.0)]],
                         mv_snap=round(moved[('snap', 0.0)][-1], 3),
                         mv_pool1=round(moved[('pool', 1.0)][-1], 3)))
        n += 1

    print('events %d ; %d jitter draws at sigma=%.2f cm\n' % (n, a.n, a.sigma))
    print('%-16s %-12s %s' % ('readout', 'target-hit', 'P(chosen candidate flips under the null)'))
    for k in [('snap', 0.0)] + [('pool', R) for R in RADII]:
        name = 'argmax+snap' if k[0] == 'snap' else 'pooled R=%.1f cm' % k[1]
        print('%-16s %-12s mean %.3f   events ever flipping %d/%d'
              % (name, '%d/%d' % (hit[k], n), float(np.mean(moved[k])),
                 sum(1 for v in moved[k] if v > 0), n))
    if a.tsv:
        os.makedirs(os.path.dirname(a.tsv) or '.', exist_ok=True)
        cols = ['evt', 'target', 'snap', 'pool1', 'mv_snap', 'mv_pool1']
        with open(a.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in sorted(rows, key=lambda r: r['evt']):
                fh.write('\t'.join(str(r[c]) for c in cols) + '\n')
        print('\nwrote %s' % a.tsv)


if __name__ == '__main__':
    sys.exit(main())

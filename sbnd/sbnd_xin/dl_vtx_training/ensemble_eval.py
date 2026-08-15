#!/usr/bin/env python3
'''
doc pr/79 -- fold-ensemble and weight-soup check on the ft2u checkpoints.

*** ADVISORY SECOND LOCKBOX READ ***
The 95 lockbox events were read ONCE (runs/ft2u/lockbox.log) to arbitrate
ft2u; this script reads them again, which turns them into a selection set.
Owner-approved 2026-08-15 with that caveat stated: any candidate this run
prefers needs fresh labels or a real production A/B before deployment.

Evaluated on the lockbox (no other uncontaminated set exists -- every
train-pool event was in the training set of 5 of the 6 folds):
  baseline   the uBooNE CP24 production net
  deploy     runs/ft2u-deploy/fold0/CP9.pth (what Phase C ships)
  ensemble   per-voxel MEAN of the 6 fold-best checkpoints' scores
             (voxel lattice is model-independent, so scores are alignable;
             NOT deployable -- the C++ loads one .pth)
  soup       single net whose state_dict is the tensor-mean of the 6
             fold-best checkpoints (all fine-tuned from the same CP24 init,
             same basin) -- deployable as one file

Usage:
  python3 ensemble_eval.py --data data/full473 --run runs/ft2u \
      --deploy runs/ft2u-deploy/fold0/CP9.pth --soup-out runs/ft2u/soup.pth
'''
import argparse
import glob
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from scn_vtx.model import make_model, load_weights, predict_scores
from scn_vtx.voxelize import voxelize_event, top_k_voxels
from dataset import load_manifest
from evaluate import snap_hit


def fold_best_cps(run_dir):
    cps = []
    for fold_dir in sorted(glob.glob(os.path.join(run_dir, 'fold*'))):
        best = json.load(open(os.path.join(fold_dir, 'best.json')))['best_epoch']
        cps.append(os.path.join(fold_dir, 'CP%d.pth' % best))
    return cps


def make_soup(cps, out_path):
    """Tensor-mean of the checkpoints' state_dicts (float tensors averaged,
    integer buffers taken from the first -- they are BN counts)."""
    dicts = [torch.load(p, map_location='cpu') for p in cps]
    soup = {}
    for k in dicts[0]:
        t0 = dicts[0][k]
        if t0.is_floating_point():
            soup[k] = torch.stack([d[k].float() for d in dicts]).mean(0).to(t0.dtype)
        else:
            soup[k] = t0.clone()
    torch.save(soup, out_path)
    return out_path


def loaded(path):
    m = make_model('cpu')
    load_weights(m, path)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--run', required=True, help='kfold run dir (e.g. runs/ft2u)')
    ap.add_argument('--deploy', required=True, help='the Phase-C deployment CP')
    ap.add_argument('--baseline', default=vio.default_weights())
    ap.add_argument('--soup-out', required=True)
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--guard-tol', type=float, default=1.0)
    args = ap.parse_args()

    print('*** ADVISORY SECOND LOCKBOX READ (see header) ***')
    rows = [r for r in load_manifest(args.data, drop_lockbox=False)
            if str(r.get('lockbox')) == '1']
    print('lockbox events: %d' % len(rows))

    cps = fold_best_cps(args.run)
    print('fold-best checkpoints: %s' % ', '.join(
        os.path.relpath(p, args.run) for p in cps))
    soup_path = make_soup(cps, args.soup_out)
    print('soup written: %s' % soup_path)

    scorers = dict(
        baseline=[loaded(args.baseline)],
        deploy=[loaded(args.deploy)],
        ensemble=[loaded(p) for p in cps],
        soup=[loaded(soup_path)])

    corr = np.array([int(r['corrective']) for r in rows], dtype=bool)
    d_of = {name: [] for name in scorers}
    snap_of = {name: [] for name in scorers}
    for row in rows:
        with np.load(os.path.join(args.data, row['npz'])) as f:
            xyz = f['xyz'].astype(np.float32)
            q = f['q'].astype(np.float32)
            truth = f['truth_xyz'].astype(np.float32)
            calib_path = str(f['calib_path'])
        coords, ft, offset = voxelize_event(xyz, q)
        for name, models in scorers.items():
            preds = np.stack([predict_scores(m, coords, ft) for m in models])
            pred = preds.mean(axis=0)
            topk = top_k_voxels(pred, coords, offset, k=5)
            d_of[name].append(float(np.linalg.norm(topk[0, :3] - truth)))
            h = snap_hit(topk, truth, calib_path, args.tol)
            snap_of[name].append(-1 if h is None else int(h))

    base = np.array(d_of['baseline'])
    print('\n== lockbox summary (n=%d, tol %.1f cm) ==' % (len(rows), args.tol))
    for name in ('baseline', 'deploy', 'ensemble', 'soup'):
        d = np.array(d_of[name])
        s = np.array(snap_of[name])
        hits, ns = (s == 1).sum(), (s >= 0).sum()
        guard = int(((d - base > args.guard_tol) & ~corr).sum())
        print('%-9s snap-hit %2d/%2d (%+3d)  guard-fails %d  '
              'd_argmax p50=%6.2f p90=%7.2f  corrective p50=%6.2f'
              % (name, hits, ns, hits - (np.array(snap_of['baseline']) == 1).sum(),
                 guard, np.percentile(d, 50), np.percentile(d, 90),
                 np.percentile(d[corr], 50)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

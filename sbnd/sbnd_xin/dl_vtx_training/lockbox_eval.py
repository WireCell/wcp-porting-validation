#!/usr/bin/env python3
'''
doc pr/78 -- the ONE-TIME lockbox read.

The 95 lockbox events (manifest lockbox=1) were excluded from every
gradient, every fold, and every model-selection decision of rounds 2-3.
This script evaluates an arm's six fold-selected checkpoints (plus the CP24
baseline) on exactly those events and prints per-fold and pooled verdicts.
Run it ONCE, on the single arm chosen from out-of-fold metrics, and freeze
the output in runs/<arm>/lockbox.log.  Running it on several arms would
turn the lockbox into a selection set and void it.
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
from scn_vtx.model import make_model, load_weights
from dataset import load_manifest, VtxSamples
from train import run_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--baseline', default=vio.default_weights())
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--guard-tol', type=float, default=1.0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    rows = [r for r in load_manifest(args.data) if str(r.get('lockbox')) == '1']
    print('lockbox events: %d' % len(rows))
    va = VtxSamples(args.data, rows, train=False)
    crit = torch.nn.MSELoss()

    def dists_of(weights_path):
        model = make_model(device=args.device)
        load_weights(model, weights_path)
        _, _, d = run_epoch(model, va, args.device, crit)
        return np.array(d)

    base = dists_of(args.baseline)
    corr = np.array([int(r['corrective']) for r in rows], dtype=bool)
    print('baseline: snap-hit %d/%d  corrective p50=%.1f p90=%.1f'
          % ((base <= args.tol).sum(), len(base),
             np.percentile(base[corr], 50), np.percentile(base[corr], 90)))

    pooled = []
    for fold_dir in sorted(glob.glob(os.path.join(args.run, 'fold*'))):
        best = json.load(open(os.path.join(fold_dir, 'best.json')))
        cp = os.path.join(fold_dir, 'CP%d.pth' % best['best_epoch'])
        d = dists_of(cp)
        guard = int(((d - base > args.guard_tol) & ~corr).sum())
        print('%s (CP%d): snap-hit %d/%d (%+d)  guard-fails %d  '
              'corrective p50=%.1f p90=%.1f'
              % (os.path.basename(fold_dir), best['best_epoch'],
                 (d <= args.tol).sum(), len(d),
                 (d <= args.tol).sum() - (base <= args.tol).sum(), guard,
                 np.percentile(d[corr], 50), np.percentile(d[corr], 90)))
        pooled.append(d)
    P = np.stack(pooled)
    med = np.median(P, axis=0)   # per-event median across the 6 checkpoints
    guard = int(((med - base > args.guard_tol) & ~corr).sum())
    print('\npooled (per-event median over %d folds): snap-hit %d/%d (%+d), '
          'guard-fails %d' % (len(pooled), (med <= args.tol).sum(), len(med),
                              (med <= args.tol).sum() - (base <= args.tol).sum(),
                              guard))
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
'''
doc pr/77 round 2 (S8c) -- confident pseudo-labels from the production
vertex on not-yet-scanned PR events.

Step 1 (--precision): on the ALREADY-labeled events, measure
P(production main vertex within --tol of truth | cut) for a grid of
confidence cuts (route accept + margin >= M + dl_best >= B + snap_dis <= S).
The labels arbitrate which cut is trustworthy -- pseudo-labels only ever
enter TRAINING, never validation, so the measured precision is the
contamination rate we knowingly accept.

Step 2 (--build NAME --margin M --dl-best B --snap S): freeze a pseudo
snapshot data/NAME from the unlabeled events passing the chosen cut,
truth := production main vertex, manifest flags is_pseudo=1.  REFUSES to
overwrite (same rule as build_dataset.py).

Usage:
  python3 pseudo_labels.py --precision
  python3 pseudo_labels.py --build pseudo0 --margin 500 --dl-best 1000 --snap 0.5
'''
import argparse
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from scan_ranker import event_signals, ARM, TAG


def collect(sbnd_root, arm, tag, tol):
    labeled = {}
    for label in vio.iter_labels(sbnd_root, [tag]):
        labeled[label['eventNo']] = int(not vio.is_corrective(label, tol=tol))
    recs = []
    for path in sorted(glob.glob(os.path.join(
            sbnd_root, arm, 'pr_evt*', 'calib-pr-evt*.json'))):
        evt = int(os.path.basename(path).split('evt')[-1].split('.')[0])
        calib = vio.load_calib(path)
        sig = event_signals(calib)
        sig.update(evt=evt, path=path,
                   labeled=int(evt in labeled),
                   prod_ok=labeled.get(evt, -1))
        recs.append(sig)
    return recs


def passes(r, margin, dl_best, snap):
    return (r['route'] == 'dl-rerank-accept'
            and np.isfinite(r['dl_best']) and r['dl_best'] >= dl_best
            and (not np.isfinite(r['margin']) or r['margin'] >= margin)
            and np.isfinite(r['snap_dis']) and r['snap_dis'] <= snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--arm', default=ARM)
    ap.add_argument('--tag', default=TAG)
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--precision', action='store_true')
    ap.add_argument('--build', default=None, help='snapshot name under data/')
    ap.add_argument('--margin', type=float, default=500.0)
    ap.add_argument('--dl-best', type=float, default=1000.0)
    ap.add_argument('--snap', type=float, default=0.5)
    args = ap.parse_args()

    recs = collect(args.sbnd_root, args.arm, args.tag, args.tol)
    lab = [r for r in recs if r['labeled']]
    unl = [r for r in recs if not r['labeled']]
    base = np.mean([r['prod_ok'] for r in lab])
    print('labeled %d (production correct %.1f%%), unlabeled %d'
          % (len(lab), 100 * base, len(unl)))

    if args.precision:
        print('\n  margin>=  dl_best>=  snap<=  | labeled-pass  precision  unl-yield')
        for m in (0, 200, 500, 1000):
            for b in (0, 500, 1000, 2000):
                for s in (0.3, 0.5, 1.0, 99.0):
                    lp = [r for r in lab if passes(r, m, b, s)]
                    if len(lp) < 10:
                        continue
                    prec = np.mean([r['prod_ok'] for r in lp])
                    ny = sum(1 for r in unl if passes(r, m, b, s))
                    mark = ' <-- >=95%' if prec >= 0.95 else ''
                    print('  %7.0f  %8.0f  %6.1f  |     %3d       %5.1f%%     %4d%s'
                          % (m, b, s, len(lp), 100 * prec, ny, mark))

    if args.build:
        here = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(here, 'data', args.build)
        if os.path.exists(os.path.join(out_dir, 'manifest.tsv')):
            print('REFUSING to overwrite existing snapshot %s' % out_dir)
            return 1
        os.makedirs(out_dir, exist_ok=True)
        chosen = [r for r in unl if passes(r, args.margin, args.dl_best, args.snap)]
        lp = [r for r in lab if passes(r, args.margin, args.dl_best, args.snap)]
        prec = np.mean([r['prod_ok'] for r in lp]) if lp else float('nan')
        print('cut (margin>=%.0f, dl_best>=%.0f, snap<=%.1f): measured '
              'precision %.1f%% on %d labeled; building %d pseudo events'
              % (args.margin, args.dl_best, args.snap, 100 * prec, len(lp),
                 len(chosen)))
        cols = ['evt', 'tag', 'arm', 'runNo', 'subRunNo', 'n_cloud',
                'truth_x', 'truth_y', 'truth_z', 'corrective', 'sample',
                'lockbox', 'is_pseudo', 'npz']
        rows = []
        for r in chosen:
            calib = vio.load_calib(r['path'])
            xyz, q, info = vio.rebuild_cloud(calib)
            if len(q) == 0:
                continue
            mv = calib.get('main_vertex') or {}
            truth = np.array([mv['x'], mv['y'], mv['z']], dtype=np.float32)
            npz = 'evt%d.npz' % r['evt']
            np.savez_compressed(
                os.path.join(out_dir, npz), xyz=xyz, q=q, truth_xyz=truth,
                eventNo=r['evt'], arm=args.arm, is_pseudo=True,
                calib_path=str(r['path']))
            rows.append(dict(evt=r['evt'], tag='pseudo:' + args.tag,
                             arm=args.arm, runNo=info['runNo'],
                             subRunNo=info['subRunNo'], n_cloud=len(q),
                             truth_x='%.6f' % truth[0], truth_y='%.6f' % truth[1],
                             truth_z='%.6f' % truth[2], corrective=0,
                             sample='pseudo', lockbox=0, is_pseudo=1, npz=npz))
        with open(os.path.join(out_dir, 'manifest.tsv'), 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for row in rows:
                fh.write('\t'.join(str(row[c]) for c in cols) + '\n')
        # freeze the cut + precision alongside the data
        with open(os.path.join(out_dir, 'cut.json'), 'w') as fh:
            import json
            json.dump(dict(margin=args.margin, dl_best=args.dl_best,
                           snap=args.snap, tol=args.tol,
                           measured_precision=prec, n_labeled_pass=len(lp),
                           n_pseudo=len(rows)), fh, indent=1)
        print('wrote %d pseudo events -> %s' % (len(rows), out_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())

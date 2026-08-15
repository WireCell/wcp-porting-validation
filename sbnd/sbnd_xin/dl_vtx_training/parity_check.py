#!/usr/bin/env python3
'''
doc pr/77 -- inference-parity check: run the production CP24 weights on the
cloud REBUILT from calib-pr-evt<ID>.json and compare the top-5 voxels+scores
against what production recorded in vertex_scoreboard.voxels[].

This measures the post-refit approximation error the training set inherits
(the calib dump is written after snap_to_kink/improve_vertex; the net saw the
pre-refit graph).  It is the go/no-go for training off the calib JSON.

Usage:
  python3 parity_check.py --tags vtxscan-prod0813 vtxscan-prod0813-ncpi0 \
      [--tsv out.tsv] [--events 388 10550]

Compares on CPU (production runs CPU).  Match metric per event:
  d1     : |top-1 rebuilt - top-1 recorded| (cm)
  s1     : |score-1 rebuilt - score-1 recorded|
  dset   : max over recorded voxels of distance to NEAREST rebuilt top-5 (cm)
           (rank-insensitive: near-degenerate scores reorder freely)
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from scn_vtx.model import make_model, load_weights, predict_scores
from scn_vtx.voxelize import voxelize_event, top_k_voxels


def run_event(model, calib, k=5):
    xyz, q, info = vio.rebuild_cloud(calib)
    coords, ft, offset = voxelize_event(xyz, q)
    pred = predict_scores(model, coords, ft, device='cpu')
    return top_k_voxels(pred, coords, offset, k=k), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+',
                    default=['vtxscan-prod0813', 'vtxscan-prod0813-ncpi0'])
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--weights', default=vio.default_weights())
    ap.add_argument('--events', nargs='*', type=int, default=None,
                    help='restrict to these eventNos')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    model = make_model(device='cpu')
    load_weights(model, args.weights)
    model.eval()

    rows = []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        evt = label['eventNo']
        if args.events and evt not in args.events:
            continue
        calib = vio.load_calib(vio.calib_path_for_label(args.sbnd_root, label))
        sb = calib.get('vertex_scoreboard') or {}
        rec = sb.get('voxels') or []
        if not rec:
            print('evt%-8d NO recorded voxels (scoreboard off?) -- skipped' % evt)
            continue
        topk, info = run_event(model, calib, k=max(5, len(rec)))
        rec_xyz = np.array([[v['x'], v['y'], v['z']] for v in rec], dtype=np.float32)
        rec_s = np.array([v['dl_score'] for v in rec], dtype=np.float32)

        d1 = float(np.linalg.norm(topk[0, :3] - rec_xyz[0]))
        s1 = float(abs(topk[0, 3] - rec_s[0]))
        # rank-insensitive coverage of the recorded set by the rebuilt set
        dmat = np.linalg.norm(rec_xyz[:, None, :] - topk[None, :, :3], axis=2)
        dset = float(dmat.min(axis=1).max())
        rows.append(dict(tag=label['scan_tag'], evt=evt, n_pts=len(topk),
                         n_cloud=info['n_vtx_points'] + info['n_seg_points'],
                         n_invalid=info['n_invalid_fit'],
                         d1=d1, s1=s1, dset=dset,
                         s1_rebuilt=float(topk[0, 3]), s1_recorded=float(rec_s[0])))
        print('evt%-8d cloud=%-6d d1=%7.3f cm  s1=%7.4f  dset=%7.3f cm'
              % (evt, rows[-1]['n_cloud'], d1, s1, dset))

    if not rows:
        print('nothing compared'); return 1
    d1 = np.array([r['d1'] for r in rows]); dset = np.array([r['dset'] for r in rows])
    s1 = np.array([r['s1'] for r in rows])
    print('\n== parity over %d events ==' % len(rows))
    for name, a in (('d1(cm)', d1), ('dset(cm)', dset), ('|ds1|', s1)):
        print('  %-9s p50=%7.3f  p90=%7.3f  max=%7.3f' %
              (name, np.percentile(a, 50), np.percentile(a, 90), a.max()))
    exact = int(((d1 < 0.75) & (s1 < 0.05)).sum())  # within ~one voxel + score noise
    print('  top-1 within one voxel AND |ds|<0.05: %d/%d' % (exact, len(rows)))

    if args.tsv:
        import csv
        with open(args.tsv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
            w.writeheader(); w.writerows(rows)
        print('wrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

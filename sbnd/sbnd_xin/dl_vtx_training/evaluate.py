#!/usr/bin/env python3
'''
doc pr/77 -- evaluate baseline (CP24) vs fine-tuned weights on a snapshot,
with metrics that match deployment (doc pr/52 S3):

  d_argmax      argmax voxel -> truth (raw net quality)
  d_top5        best of the top-5 voxels -> truth (what rerank can reach)
  snap_hit      nearest candidate vertex to the BEST top-5 voxel is within
                --tol of truth (production snaps voxels to PR-graph
                candidates; candidates come from scoreboard rows[])
  guard         on confirming events (dis_to_main==0) the fine-tune must not
                degrade d_argmax by more than --guard-tol vs baseline

Out-of-fold discipline: with --run pointing at a kfold train.py run, each
event is evaluated with the checkpoint of the fold that did NOT train on it.
With --tuned <path>, one checkpoint evaluates everything (use for quick looks
only).  Baseline-only: omit both.

Usage:
  python3 evaluate.py --data data/practice66 --run runs/ft0 --tsv runs/ft0/eval.tsv
'''
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from scn_vtx.model import make_model, load_weights, predict_scores
from scn_vtx.voxelize import voxelize_event, top_k_voxels
from dataset import load_manifest, kfold_split


def eval_weights(model, row, snapshot_dir, k=5):
    with np.load(os.path.join(snapshot_dir, row['npz'])) as f:
        xyz = f['xyz'].astype(np.float32)
        q = f['q'].astype(np.float32)
        truth = f['truth_xyz'].astype(np.float32)
        calib_path = str(f['calib_path'])
    coords, ft, offset = voxelize_event(xyz, q)
    pred = predict_scores(model, coords, ft)
    topk = top_k_voxels(pred, coords, offset, k=k)
    d_argmax = float(np.linalg.norm(topk[0, :3] - truth))
    d_top5 = float(np.linalg.norm(topk[:, :3] - truth, axis=1).min())
    return topk, d_argmax, d_top5, truth, calib_path


def snap_hit(topk, truth, calib_path, tol):
    """Production analogue: snap the best-scored voxel to the nearest
    candidate vertex (scoreboard rows[]); hit if that candidate is within tol
    of truth."""
    calib = vio.load_calib(calib_path)
    rows = (calib.get('vertex_scoreboard') or {}).get('rows') or []
    if not rows:
        return None
    cand = np.array([[r['x'], r['y'], r['z']] for r in rows], dtype=np.float32)
    snapped = cand[np.linalg.norm(cand - topk[0, :3], axis=1).argmin()]
    return bool(np.linalg.norm(snapped - truth) < tol)


def fold_checkpoints(run_dir, rows, kfold, seed):
    """Reproduce train.py's split; map evt -> that fold's best checkpoint."""
    with open(os.path.join(run_dir, 'config.json')) as fh:
        cfg = json.load(fh)
    assert cfg['kfold'] == kfold and cfg['seed'] == seed, \
        'pass --kfold/--seed matching the training run (%s)' % run_dir
    out = {}
    for fold, _, val in kfold_split(rows, kfold, seed=seed):
        with open(os.path.join(run_dir, 'fold%d' % fold, 'best.json')) as fh:
            best = json.load(fh)['best_epoch']
        cp = os.path.join(run_dir, 'fold%d' % fold, 'CP%d.pth' % best)
        for r in val:
            out[r['evt']] = cp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--run', default=None, help='kfold train.py run dir (out-of-fold)')
    ap.add_argument('--tuned', default=None, help='single checkpoint (not out-of-fold)')
    ap.add_argument('--baseline', default=vio.default_weights())
    ap.add_argument('--kfold', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--tol', type=float, default=1.0, help='snap-hit tolerance (cm)')
    ap.add_argument('--guard-tol', type=float, default=1.0,
                    help='allowed d_argmax degradation on confirming events (cm)')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    rows = load_manifest(args.data)
    base = make_model('cpu'); load_weights(base, args.baseline); base.eval()

    cp_of = {}
    if args.run:
        cp_of = fold_checkpoints(args.run, rows, args.kfold, args.seed)
    tuned_single = None
    if args.tuned:
        tuned_single = make_model('cpu'); load_weights(tuned_single, args.tuned)
        tuned_single.eval()

    _model_cache = {}
    def tuned_for(evt):
        if tuned_single is not None:
            return tuned_single
        cp = cp_of.get(evt)
        if cp is None:
            return None
        if cp not in _model_cache:
            m = make_model('cpu'); load_weights(m, cp); m.eval()
            _model_cache[cp] = m
        return _model_cache[cp]

    out = []
    for row in rows:
        evt = row['evt']
        tb, db, d5b, truth, calib_path = eval_weights(base, row, args.data)
        hb = snap_hit(tb, truth, calib_path, args.tol)
        rec = dict(evt=evt, tag=row['tag'], corrective=row['corrective'],
                   manual=int(row['pick_kind'] == 'manual'),
                   d_argmax_base=db, d_top5_base=d5b,
                   snap_base='' if hb is None else int(hb))
        mt = tuned_for(evt)
        if mt is not None:
            tt, dt, d5t, _, _ = eval_weights(mt, row, args.data)
            ht = snap_hit(tt, truth, calib_path, args.tol)
            rec.update(d_argmax_tuned=dt, d_top5_tuned=d5t,
                       snap_tuned='' if ht is None else int(ht),
                       guard_fail=int(row['corrective'] == 0
                                      and dt - db > args.guard_tol))
        out.append(rec)
        print('evt%-8d corr=%d  base d=%7.2f d5=%7.2f%s'
              % (evt, row['corrective'], db, d5b,
                 ('  tuned d=%7.2f d5=%7.2f%s'
                  % (rec['d_argmax_tuned'], rec['d_top5_tuned'],
                     '  GUARD-FAIL' if rec.get('guard_fail') else '')
                  if 'd_argmax_tuned' in rec else '')))

    def summary(key, sel, label):
        a = np.array([r[key] for r in out if sel(r) and key in r])
        if len(a) == 0:
            return
        print('  %-34s n=%2d  p50=%7.2f  p90=%7.2f  max=%7.2f'
              % (label, len(a), np.percentile(a, 50), np.percentile(a, 90), a.max()))

    print('\n== summary (cm) ==')
    for key, tag in (('d_argmax_base', 'baseline'), ('d_argmax_tuned', 'tuned')):
        summary(key, lambda r: True, '%s d_argmax all' % tag)
        summary(key, lambda r: r['corrective'] == 1, '%s d_argmax corrective' % tag)
        summary(key, lambda r: r['corrective'] == 0, '%s d_argmax confirming' % tag)
    ng = sum(r.get('guard_fail', 0) for r in out)
    if any('guard_fail' in r for r in out):
        print('  confirmation guard failures (>%.1f cm worse): %d' % (args.guard_tol, ng))
    for key, tag in (('snap_base', 'baseline'), ('snap_tuned', 'tuned')):
        hits = [r[key] for r in out if r.get(key, '') != '']
        if hits:
            print('  %s snap-hit (tol %.1f cm): %d/%d' % (tag, args.tol, sum(hits), len(hits)))

    if args.tsv:
        cols = sorted({k for r in out for k in r}, key=lambda c: (c != 'evt', c))
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in out:
                fh.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
        print('wrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

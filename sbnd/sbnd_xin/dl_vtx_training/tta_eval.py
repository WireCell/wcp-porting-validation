#!/usr/bin/env python3
'''
doc pr/77 round 2 (S8a) -- test-time augmentation: ensemble CP24 over the x4
X/Y reflections (optionally + sub-voxel shifts), no training, no new labels.

Ensembling across views lives on the IDENTITY view's voxel lattice: each
view's cloud is transformed, voxelized and scored independently; then for
every identity voxel centre c we look up the view voxel containing T(c)
(searching the 3^3 neighbourhood for the nearest occupied voxel -- lattices
of different views have different boundary phases).  The TTA score of c is
the mean over views.  A 1-view TTA therefore IS plain inference (identity
check asserted on the first 3 events).

Disagreement signals per event (for the S8c active-learning ranker):
  tta_argmax_spread  RMS distance (cm) of per-view argmax positions (mapped
                     back to the original frame) around their centroid
  tta_score_std      std of the per-view max scores

Modes:
  --data data/<snapshot>   labeled events, metrics vs truth per sample
  --arm work-mcp1k-prod0813  sweep every pr_evt* calib (no truth): signals TSV

Usage:
  python3 tta_eval.py --data data/mix116 --tsv runs/tta-mix116.tsv
  python3 tta_eval.py --arm work-mcp1k-prod0813 --tsv runs/tta-signals-mcp1k.tsv
'''
import argparse
import glob
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from scn_vtx.model import make_model, load_weights, predict_scores
from scn_vtx.voxelize import (voxelize_event, voxel_center_cm, top_k_voxels,
                              RESOLUTION)
from augment import FLIPS
from dataset import load_manifest

NEIGH = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                  for k in (-1, 0, 1)], dtype=np.int64)


def view_transforms(shifts, rng):
    """List of (kind, param): x4 flips + optional random sub-voxel shifts of
    the identity orientation."""
    views = [('flip', np.array([sx, sy, 1], dtype=np.float32))
             for sx, sy in FLIPS]
    for _ in range(shifts):
        views.append(('shift',
                      rng.uniform(0.0, RESOLUTION, 3).astype(np.float32)))
    return views


def apply_view(xyz, view):
    kind, p = view
    return xyz * p if kind == 'flip' else xyz + p


def map_points(pts, view):
    """Original-frame points -> this view's frame (same transform as the cloud)."""
    return apply_view(pts, view)


def unmap_points(pts, view):
    kind, p = view
    return pts * p if kind == 'flip' else pts - p


def tta_scores(model, xyz, q, views, device='cpu'):
    """Returns (centers_id (V,3) cm, mean_scores (V,), per-view argmax in the
    original frame (nviews,3), per-view max score (nviews,))."""
    coords_id, ft_id, off_id = voxelize_event(xyz, q)
    centers_id = voxel_center_cm(coords_id, off_id)

    acc = np.zeros(len(coords_id), dtype=np.float64)
    cnt = np.zeros(len(coords_id), dtype=np.int32)
    argmaxes, maxscores = [], []
    for view in views:
        xyz_v = apply_view(xyz, view)
        coords_v, ft_v, off_v = voxelize_event(xyz_v, q)
        pred_v = predict_scores(model, coords_v, ft_v, device=device)
        centers_v = voxel_center_cm(coords_v, off_v)
        lut = {tuple(c): i for i, c in enumerate(coords_v)}

        am = centers_v[int(np.argmax(pred_v))]
        argmaxes.append(unmap_points(am[None, :], view)[0])
        maxscores.append(float(pred_v.max()))

        t = map_points(centers_id, view)
        base = np.floor((t - off_v) / RESOLUTION).astype(np.int64)
        for i in range(len(centers_id)):
            best_i, best_d = -1, 1e9
            for n in NEIGH:
                j = lut.get(tuple(base[i] + n))
                if j is None:
                    continue
                d = float(np.linalg.norm(centers_v[j] - t[i]))
                if d < best_d:
                    best_d, best_i = d, j
            if best_i >= 0:
                acc[i] += pred_v[best_i]
                cnt[i] += 1
    mean = (acc / np.maximum(cnt, 1)).astype(np.float32)
    return centers_id, mean, np.array(argmaxes), np.array(maxscores)


def signals(argmaxes, maxscores):
    cen = argmaxes.mean(axis=0)
    spread = float(np.sqrt(np.mean(np.sum((argmaxes - cen) ** 2, axis=1))))
    return spread, float(np.std(maxscores))


def top5_of(centers, scores):
    k = min(5, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    out = np.empty((k, 4), dtype=np.float32)
    out[:, :3] = centers[idx]
    out[:, 3] = scores[idx]
    return out


def snap_hit(topk, truth, calib, tol):
    rows = (calib.get('vertex_scoreboard') or {}).get('rows') or []
    if not rows:
        return None
    cand = np.array([[r['x'], r['y'], r['z']] for r in rows], dtype=np.float32)
    snapped = cand[np.linalg.norm(cand - topk[0, :3], axis=1).argmin()]
    return bool(np.linalg.norm(snapped - truth) < tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None, help='snapshot dir (labeled eval)')
    ap.add_argument('--arm', default=None, help='work dir sweep (signals only)')
    ap.add_argument('--weights', default=vio.default_weights())
    ap.add_argument('--shifts', type=int, default=0)
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()
    assert bool(args.data) != bool(args.arm), 'exactly one of --data/--arm'

    rng = np.random.default_rng(args.seed)
    views = view_transforms(args.shifts, rng)
    model = make_model('cpu'); load_weights(model, args.weights); model.eval()

    out = []
    if args.data:
        rows = load_manifest(args.data)
        if args.limit:
            rows = rows[:args.limit]
        for n, row in enumerate(rows):
            with np.load(os.path.join(args.data, row['npz'])) as f:
                xyz = f['xyz'].astype(np.float32)
                q = f['q'].astype(np.float32)
                truth = f['truth_xyz'].astype(np.float32)
                calib_path = str(f['calib_path'])
            centers, mean, ams, mss = tta_scores(model, xyz, q, views)
            if n < 3:  # identity check: 1-view TTA == plain inference
                c1, m1, _, _ = tta_scores(model, xyz, q, views[:1])
                coords, ft, off = voxelize_event(xyz, q)
                plain = predict_scores(model, coords, ft)
                assert np.allclose(m1, plain, atol=1e-6), 'identity check failed'
            spread, sstd = signals(ams, mss)
            calib = vio.load_calib(calib_path)
            t5_t = top5_of(centers, mean)
            coords, ft, off = voxelize_event(xyz, q)
            plain = predict_scores(model, coords, ft)
            t5_b = top_k_voxels(plain, coords, off, k=5)
            rec = dict(evt=row['evt'], sample=row.get('sample', ''),
                       corrective=row['corrective'],
                       d_base=float(np.linalg.norm(t5_b[0, :3] - truth)),
                       d5_base=float(np.linalg.norm(t5_b[:, :3] - truth, axis=1).min()),
                       d_tta=float(np.linalg.norm(t5_t[0, :3] - truth)),
                       d5_tta=float(np.linalg.norm(t5_t[:, :3] - truth, axis=1).min()),
                       snap_base=snap_hit(t5_b, truth, calib, args.tol),
                       snap_tta=snap_hit(t5_t, truth, calib, args.tol),
                       tta_argmax_spread=spread, tta_score_std=sstd)
            out.append(rec)
            print('evt%-8d %-7s base d=%7.2f  tta d=%7.2f  spread=%6.2f'
                  % (rec['evt'], rec['sample'], rec['d_base'], rec['d_tta'],
                     spread), flush=True)

        print('\n== TTA vs single view (cm) ==')
        samples = sorted({r['sample'] for r in out}) + ['ALL']
        for s in samples:
            sel = [r for r in out if s == 'ALL' or r['sample'] == s]
            if not sel:
                continue
            for key in ('d_base', 'd_tta'):
                a = np.array([r[key] for r in sel])
                print('  %-7s %-7s n=%3d p50=%6.2f p90=%7.2f max=%7.2f'
                      % (s, key, len(a), np.percentile(a, 50),
                         np.percentile(a, 90), a.max()))
            for key in ('snap_base', 'snap_tta'):
                h = [r[key] for r in sel if r[key] is not None]
                print('  %-7s %-9s %d/%d' % (s, key, sum(h), len(h)))
    else:
        paths = sorted(glob.glob(os.path.join(
            args.sbnd_root, args.arm, 'pr_evt*', 'calib-pr-evt*.json')))
        if args.limit:
            paths = paths[:args.limit]
        for path in paths:
            calib = vio.load_calib(path)
            xyz, q, info = vio.rebuild_cloud(calib)
            if len(q) == 0:  # empty PR graph (e.g. cosmic-tagged, no bundle)
                print('evt%-8s EMPTY cloud, skipped' % info['eventNo'], flush=True)
                continue
            _, _, ams, mss = tta_scores(model, xyz, q, views)
            spread, sstd = signals(ams, mss)
            out.append(dict(evt=info['eventNo'], tta_argmax_spread=spread,
                            tta_score_std=sstd, n_cloud=len(q)))
            print('evt%-8d spread=%7.2f score_std=%.4f' %
                  (info['eventNo'], spread, sstd), flush=True)

    if args.tsv:
        cols = sorted({k for r in out for k in r}, key=lambda c: (c != 'evt', c))
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in out:
                fh.write('\t'.join(
                    '' if r.get(c) is None else
                    ('%d' % r[c] if isinstance(r[c], (bool, np.bool_)) else
                     ('%.4f' % r[c] if isinstance(r[c], float) else str(r[c])))
                    for c in cols) + '\n')
        print('wrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

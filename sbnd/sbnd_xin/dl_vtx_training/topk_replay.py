#!/usr/bin/env python3
'''
doc pr/78 round 3 -- replay CP24 on candidate-missing events and simulate the
`dl_vtx_top_k` admission (NeutrinoVertexFinder.cxx: each of the top-k DL
voxels snaps to the NEAREST PR-graph vertex; snapped vertices are the DL
candidate rows).  Production uses top_k=5 (clus.jsonnet:1898); the recorded
scoreboard keeps only those 5 voxels, so whether rank 6..K would have reached
the truth vertex can only be answered by re-running the net -- which this
does, on the rebuilt input cloud (post-refit approximation, doc pr/77
parity: top-1 voxel exact on 39/66, median off by one voxel).

For each event: full voxel ranking by net score, then for k = 1..K the
snapped vertex set grows; record the smallest k at which a vertex within
--tol of truth is admitted (truth_k).  Summarize simulated admission
recovery at top_k = 5 / 10 / 20 / 50.

Usage: python3 topk_replay.py --taxonomy runs/taxonomy-20260815.tsv \
           --tsv runs/topk-replay-20260815.tsv
'''
import argparse
import csv
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from scn_vtx.model import make_model, load_weights, predict_scores
from scn_vtx.voxelize import voxelize_event, voxel_center_cm


def graph_vertices(calib):
    pts = []
    for v in calib.get('vertices', []):
        f = v.get('fit') or {}
        if f.get('x') is not None:
            pts.append([f['x'], f['y'], f['z']])
    return np.asarray(pts, dtype=float)


def truth_k_for(model, calib, truth, tol, kmax, device):
    xyz, q, _ = vio.rebuild_cloud(calib)
    if len(q) == 0:
        return -1, 0
    coords, ft, offset = voxelize_event(xyz, q)
    scores = predict_scores(model, coords, ft, device=device)
    centers = voxel_center_cm(coords, offset)
    vtx = graph_vertices(calib)
    if not len(vtx):
        return -1, len(scores)
    order = np.argsort(-scores)[:kmax]
    for rank, vi in enumerate(order):
        d = np.linalg.norm(vtx - centers[vi], axis=1)
        snapped = vtx[int(np.argmin(d))]
        if np.linalg.norm(snapped - truth) <= tol:
            return rank + 1, len(scores)
    return 0, len(scores)   # 0 = not admitted within kmax


def pick_sim(model, calib, truth, tol, ks, device):
    """Concrete full-policy simulation: candidate set = vertices snapped by
    the top-k voxels (dedup, max dl per vertex); pick = candidate with max
    dl_score.  Returns {k: correct} -- a LOWER bound on a composite that
    also uses geometry, but computable without the unrecorded terms."""
    xyz, q, _ = vio.rebuild_cloud(calib)
    if len(q) == 0:
        return {k: False for k in ks}
    coords, ft, offset = voxelize_event(xyz, q)
    scores = predict_scores(model, coords, ft, device=device)
    centers = voxel_center_cm(coords, offset)
    vtx = graph_vertices(calib)
    if not len(vtx):
        return {k: False for k in ks}
    order = np.argsort(-scores)
    out = {}
    best = {}   # vertex index -> best dl score among its snapping voxels
    ki = 0
    for k in sorted(ks):
        for rank in range(ki, min(k, len(order))):
            vi = order[rank]
            d = np.linalg.norm(vtx - centers[vi], axis=1)
            j = int(np.argmin(d))
            s = float(scores[vi])
            if j not in best or s > best[j]:
                best[j] = s
        ki = min(k, len(order))
        pick = max(best, key=best.get) if best else None
        out[k] = (pick is not None
                  and np.linalg.norm(vtx[pick] - truth) <= tol)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taxonomy', required=True)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--cls', default='candidate-missing')
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--kmax', type=int, default=50)
    ap.add_argument('--weights', default=vio.default_weights())
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--pick-sim', action='store_true',
                    help='run the dl-argmax pick policy over ALL labeled '
                         'events at k=5/10/20/50 (risk AND recovery)')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    with open(args.taxonomy) as fh:
        tax = {int(r['evt']): r for r in csv.DictReader(fh, delimiter='\t')}
    wanted = ({e for e, r in tax.items() if r['cls'] == args.cls}
              if not args.pick_sim else set(tax))

    model = make_model(device=args.device)
    load_weights(model, args.weights)
    model.train(False)

    KS = (5, 10, 20, 50)
    recs = []
    for lab in vio.iter_labels(args.sbnd_root, [
            'vtxscan-prod0813', 'vtxscan-prod0813-ncpi0',
            'vtxscan-prod0813-mcp1k']):
        evt = lab['eventNo']
        if evt not in wanted:
            continue
        calib = vio.load_calib(vio.calib_path_for_label(args.sbnd_root, lab))
        if args.pick_sim:
            ok = pick_sim(model, calib, np.asarray(lab['truth_xyz']),
                          args.tol, KS, args.device)
            recs.append(dict(evt=evt, sample=tax[evt]['sample'],
                             cls=tax[evt]['cls'],
                             **{'ok%d' % k: int(ok[k]) for k in KS}))
            continue
        tk, nvox = truth_k_for(model, calib, np.asarray(lab['truth_xyz']),
                               args.tol, args.kmax, args.device)
        recs.append(dict(evt=evt, sample=tax[evt]['sample'], truth_k=tk,
                         n_voxels=nvox))
        print('  evt%-8d %-7s truth_k=%3d  (n_voxels=%d)'
              % (evt, recs[-1]['sample'], tk, nvox))

    if args.pick_sim:
        print('\n== dl-argmax pick policy over %d labeled events (tol %.1f) =='
              % (len(recs), args.tol))
        prod_ok = sum(1 for r in recs if tax[r['evt']]['cls'] == 'correct')
        print('  production (composite, top_k=5): %d/%d' % (prod_ok, len(recs)))
        for k in KS:
            n = sum(r['ok%d' % k] for r in recs)
            print('  dl-argmax @ top_k=%-3d : %d/%d' % (k, n, len(recs)))
        if args.tsv:
            cols = ['evt', 'sample', 'cls'] + ['ok%d' % k for k in KS]
            with open(args.tsv, 'w') as fh:
                fh.write('\t'.join(cols) + '\n')
                for r in recs:
                    fh.write('\t'.join(str(r[c]) for c in cols) + '\n')
            print('wrote', args.tsv)
        return 0

    ks = [r['truth_k'] for r in recs]
    print('\n== %s (%d events, tol %.1f cm, replayed net, kmax %d) =='
          % (args.cls, len(recs), args.tol, args.kmax))
    for cut in (5, 10, 20, 50):
        n = sum(1 for k in ks if 0 < k <= cut)
        print('  admitted at top_k=%-3d : %2d / %d' % (cut, n, len(ks)))
    print('  never admitted (<=%d): %2d   [net ranking truly misses truth]'
          % (args.kmax, sum(1 for k in ks if k == 0)))
    print('  replay disagrees with production (admitted at k<=5): %d '
          '[rebuilt-cloud approximation]' % sum(1 for k in ks if 0 < k <= 5))

    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('evt\tsample\ttruth_k\tn_voxels\n')
            for r in recs:
                fh.write('%d\t%s\t%d\t%d\n'
                         % (r['evt'], r['sample'], r['truth_k'], r['n_voxels']))
        print('wrote', args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

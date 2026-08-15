#!/usr/bin/env python3
'''
doc pr/79 step 4c -- extract the FROZEN production net's penultimate
features at candidate voxels (the pr/78 S9a adapter direction, linear-head
screen).

R1 of the selector campaign showed the routing signal is absent from the
recorded scoreboard features (rank_sim.py oracle +25 vs every deployable
linear variant negative).  Before any C++ work, this screens whether the
CP24 net's internal 16-dim representation carries that signal: for every
usable candidate row of an arm's scoreboard, grab the penultimate feature
vector (sparseModel output, before `linear`) at the candidate's source
voxel, from a forward pass on the calib-REBUILT cloud.

CAVEAT (flagged wherever these features are consumed): the rebuilt cloud is
the post-refit graph, not the live pre-refit input (scn_vtx/io.py:11-17) --
this is a SCREEN; any deployment fit must use live-harvested features
(pr/79 lesson 2).  The match distance between the recorded live voxel
centre and the nearest rebuilt-cloud voxel centre is stored per candidate
so downstream fits can cut on it.

doc pr/81 A1 -- `--harvest` mode lifts the caveat: the cloud comes from the
calib's hv_cloud payload (the EXACT live SCN input, doc pr/79 §10), so the
forward pass reproduces live inference bit-for-bit and match_dis is exactly
0 for every candidate whose voxel_rank points into the recorded voxels[].

Output: <out>/evt<ID>.npz with
    cand16      (n_usable, 16) penultimate features at each candidate voxel
    cand_score  (n_usable,)    rebuilt-cloud net score at that voxel
    match_dis   (n_usable,)    | live voxel centre - rebuilt voxel centre | cm
    vox_rank    (n_usable,)    the row's voxel_rank back-pointer

Usage:
  OMP_NUM_THREADS=1 python3 extract_feats.py \
      --arm-roots vtxscan-prod0813=work-nuecc48-ma10-k20 \
                  vtxscan-prod0813-ncpi0=work-ncpi0-ma10-k20 \
                  vtxscan-prod0813-mcp1k=work-mcp1k-ma10-k20 \
      --out data/k20feats16-20260815 --jobs 24
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from scn_vtx import voxelize as vox
from taxonomy import ALL_TAGS

_MODEL = None


def get_model(weights):
    global _MODEL
    if _MODEL is None:
        import torch
        torch.set_num_threads(1)
        from scn_vtx import model as vmodel
        m = vmodel.make_model(device='cpu')
        vmodel.load_weights(m, weights)
        m.eval()
        _MODEL = m
    return _MODEL


def one_event(task):
    label_path, calib_path, out_path, weights, harvest = task
    import torch
    label = vio.load_label(label_path)
    calib = vio.load_calib(calib_path)
    sb = calib.get('vertex_scoreboard') or {}
    rows = sb.get('rows') or []
    usable = [r for r in rows
              if r.get('dl_snapped') and not r.get('skipped_by_swap_guard')]
    if not usable:
        return (label['eventNo'], 'no-usable')
    voxels = sb.get('voxels') or []
    if harvest:
        if not sb.get('harvest'):
            return (label['eventNo'], 'no-harvest-payload')
        c = sb['hv_cloud']
        xyz = np.stack([np.array(c['x'], np.float32),
                        np.array(c['y'], np.float32),
                        np.array(c['z'], np.float32)], axis=1)
        q = np.array(c['q'], np.float32)
    else:
        xyz, q, _info = vio.rebuild_cloud(calib)
    coords, ft, offset = vox.voxelize_event(xyz, q)
    model = get_model(weights)
    with torch.no_grad():
        f16 = model.sparseModel([torch.LongTensor(coords),
                                 torch.FloatTensor(ft)])
        pred = torch.sigmoid(model.linear(f16))
    f16 = f16.numpy()
    score = (pred[:, 1] - pred[:, 0]).numpy()
    centers = vox.voxel_center_cm(coords, offset)

    cand16, cscore, mdis, vrank = [], [], [], []
    for r in usable:
        vr = int(r.get('voxel_rank', -1))
        if 0 <= vr < len(voxels):
            live = np.array([voxels[vr]['x'], voxels[vr]['y'],
                             voxels[vr]['z']], np.float32)
        else:                       # fall back to the candidate position
            live = np.array([r['x'], r['y'], r['z']], np.float32)
        d = np.linalg.norm(centers - live, axis=1)
        i = int(d.argmin())
        cand16.append(f16[i])
        cscore.append(score[i])
        mdis.append(float(d[i]))
        vrank.append(vr)
    np.savez_compressed(out_path,
                        cand16=np.stack(cand16).astype(np.float32),
                        cand_score=np.array(cscore, np.float32),
                        match_dis=np.array(mdis, np.float32),
                        vox_rank=np.array(vrank, np.int32))
    return (label['eventNo'], 'ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=ALL_TAGS)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--arm-roots', nargs='+', required=True)
    ap.add_argument('--weights', default=vio.default_weights())
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=24)
    ap.add_argument('--harvest', action='store_true',
                    help='doc pr/81 A1: forward on the calib hv_cloud (exact '
                         'live input) instead of the rebuilt cloud')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit('refusing to write into non-empty %s (M13): '
                         'pick a fresh --out' % out)
    os.makedirs(out, exist_ok=True)

    roots = vio.parse_arm_roots(args.arm_roots, args.sbnd_root)
    tasks = []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        cp = vio.calib_path_in_roots(roots, label)
        if not os.path.exists(cp):
            raise FileNotFoundError(cp)
        tasks.append((label['label_path'], cp,
                      os.path.join(out, 'evt%d.npz' % label['eventNo']),
                      args.weights, args.harvest))

    import multiprocessing as mp
    with mp.Pool(args.jobs) as pool:
        results = pool.map(one_event, tasks, chunksize=4)
    n_ok = sum(1 for _, s in results if s == 'ok')
    skipped = [(e, s) for e, s in results if s != 'ok']
    print('extracted %d/%d; skipped: %s' % (n_ok, len(results), skipped))
    return 0


if __name__ == '__main__':
    sys.exit(main())

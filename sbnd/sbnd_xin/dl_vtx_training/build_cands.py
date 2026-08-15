#!/usr/bin/env python3
'''
doc pr/78 S9b -- candidate sidecar for the deployed-objective margin loss.

For every event of an existing frozen snapshot, extract the usable
scoreboard candidates (dl_snapped, not swap-guarded) from the calib JSON
the npz already points at, and write data/<name>-cands/evt<ID>.npz with

  cand_xyz  (N,3) float32 candidate positions (cm, untransformed)
  truth_i   int   index of the candidate within --tol of the label truth,
                  -1 if none (event then contributes no margin term)

The frozen snapshot itself is never touched (new sidecar dir; refuses to
overwrite an existing one).

Usage: python3 build_cands.py --data data/full473
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from dataset import load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tol', type=float, default=1.0)
    args = ap.parse_args()

    out_dir = args.data.rstrip('/') + '-cands'
    if os.path.exists(out_dir):
        print('REFUSING to overwrite existing sidecar %s' % out_dir)
        return 1
    os.makedirs(out_dir)

    n_elig = 0
    rows = load_manifest(args.data)
    for r in rows:
        with np.load(os.path.join(args.data, r['npz'])) as f:
            calib_path = str(f['calib_path'])
            truth = f['truth_xyz'].astype(np.float32)
        calib = vio.load_calib(calib_path)
        sb = calib.get('vertex_scoreboard') or {}
        usable = [x for x in (sb.get('rows') or [])
                  if x.get('dl_snapped') and not x.get('skipped_by_swap_guard')]
        cand = np.array([[x['x'], x['y'], x['z']] for x in usable],
                        dtype=np.float32).reshape(-1, 3)
        truth_i = -1
        if len(cand) >= 2:
            d = np.linalg.norm(cand - truth, axis=1)
            if d.min() <= args.tol:
                truth_i = int(d.argmin())
                n_elig += 1
        np.savez_compressed(os.path.join(out_dir, 'evt%d.npz' % r['evt']),
                            cand_xyz=cand, truth_i=truth_i)
    print('wrote %d sidecars (%d margin-eligible) -> %s'
          % (len(rows), n_elig, out_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())

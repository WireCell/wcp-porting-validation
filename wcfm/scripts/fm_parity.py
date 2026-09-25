#!/usr/bin/env python3
"""fm_parity.py -- the F3 gate (wcfm/docs/01 sec 7 F3, docs/04): the C++ FMFeatureExtract sidecar
against the F2 Python oracle on the same SP frames.

Per event/anode/plane: the coordinate SETS must be identical (same active pixels in the same
(channel, slice) order), then max |d feat| and the per-pixel cosine.  Gate: CPU f4 arm max |d| < 1e-4
and min cosine > 0.9999 over the whole manifest; the f16-stored arm and the GPU arm are reported.

    python3 scripts/fm_parity.py --oracle-dir /home/xqian/tmp/wcfm-f3 --suffix _fmf4 --out /home/xqian/tmp/wcfm-f3/parity_f4.json work/000001_*[0-9]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from wirecell.util import ario

WCFM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANE_NAME = {0: 'U', 1: 'V', 2: 'W'}


def read_sidecar(path):
    """TensorFileSink output -> {plane: (coords, feat f32, tensor md)}, set metadata."""
    f = ario.load(path)
    keys = list(f.keys())
    setmd = [k for k in keys if k.endswith('_metadata') and 'tensorset_' in k]
    if len(setmd) != 1:
        sys.exit(f'{path}: expected one tensorset metadata, found {setmd}')
    md = f[setmd[0]]
    ident = setmd[0].split('tensorset_')[1].split('_')[0]
    per = {}
    i = 0
    while f'fm_tensor_{ident}_{i}_array' in keys:
        tmd = f[f'fm_tensor_{ident}_{i}_metadata']
        arr = f[f'fm_tensor_{ident}_{i}_array']
        p = per.setdefault(int(tmd['plane']), {})
        name = tmd['name']
        if name == 'coords':
            p['coords'] = np.asarray(arr).reshape(-1, 2).astype(np.int64)
        elif name == 'feat':
            p['feat'] = np.asarray(arr, dtype=np.float32).reshape(-1, int(md['feature_dim']))
        elif name == 'feat_half':
            p['feat'] = np.asarray(arr).astype(np.uint16).view(np.float16).astype(np.float32).reshape(-1, int(md['feature_dim']))
        p['md'] = tmd
        i += 1
    return per, md


def compare(a, b):
    d = np.abs(a - b)
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    cos = (a * b).sum(axis=1) / np.maximum(na * nb, 1e-12)
    return dict(max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0,
                min_cos=float(cos.min()) if cos.size else 1.0, n=int(a.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workdirs', nargs='+', help='base work dirs (work/000001_<evt>); the sidecar is read from <workdir><suffix>')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--oracle-dir', required=True)
    ap.add_argument('--max-abs', type=float, default=1e-4)
    ap.add_argument('--min-cos', type=float, default=0.9999)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []
    worst = dict(max_abs=0.0, min_cos=1.0)
    coord_fail = 0
    for wd in args.workdirs:
        base = os.path.basename(wd.rstrip('/'))
        for fn in sorted(glob.glob(os.path.join(wd.rstrip('/') + args.suffix, 'fm-features-anode*.tar.gz'))):
            a = int(os.path.basename(fn).split('anode')[1].split('.')[0])
            per, md = read_sidecar(fn)
            orc = np.load(os.path.join(args.oracle_dir, f'oracle-{base}-anode{a}.npz'))
            for P, d in sorted(per.items()):
                name = PLANE_NAME[P]
                oc = orc[f'coords_{name}'].astype(np.int64)
                of = orc[f'feat_{name}']
                same = oc.shape == d['coords'].shape and np.array_equal(oc, d['coords'])
                row = dict(event=base, anode=a, plane=name, n_cpp=int(d['coords'].shape[0]), n_oracle=int(oc.shape[0]), coords_identical=bool(same))
                if same:
                    row.update(compare(of, d['feat']))
                    worst['max_abs'] = max(worst['max_abs'], row['max_abs'])
                    worst['min_cos'] = min(worst['min_cos'], row['min_cos'])
                else:
                    coord_fail += 1
                    so = set(map(tuple, oc)); sc = set(map(tuple, d['coords']))
                    row['only_oracle'] = len(so - sc); row['only_cpp'] = len(sc - so)
                rows.append(row)
                print(f"[parity] {base} anode {a} {name}: N={row['n_cpp']}/{row['n_oracle']} coords {'OK' if same else 'DIFF'}"
                      + (f" max|d|={row['max_abs']:.2e} mean|d|={row['mean_abs']:.2e} min_cos={row['min_cos']:.6f}" if same else
                         f" only_oracle={row['only_oracle']} only_cpp={row['only_cpp']}"), flush=True)
    ok = coord_fail == 0 and worst['max_abs'] < args.max_abs and worst['min_cos'] > args.min_cos
    res = dict(suffix=args.suffix, gate=dict(max_abs=args.max_abs, min_cos=args.min_cos), n=len(rows), coord_fail=coord_fail,
               worst=worst, **{'pass': ok}, rows=rows)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f"[parity] {'PASS' if ok else 'FAIL'}: {len(rows)} plane-sets, coord mismatches={coord_fail}, "
          f"max|d|={worst['max_abs']:.2e} min_cos={worst['min_cos']:.6f} -> {args.out}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

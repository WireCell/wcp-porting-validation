#!/usr/bin/env python3
"""fm_scale_check.py -- fix the FM input scale and active-mask threshold against the training packs
(wcfm/docs/01 sec 7 F2, docs/04 sec 1).

Static chain (verified by reading the sources, see doc 04): LArSoft wclsFrameSaver frame_scale 0.005
-> wclsCookedFrameSource frame_scale 50 -> Labelling2D rebin_time_tick 4 = SUM of the 4 ticks ->
HDF5FrameTap scale 1 -> sparsify_to_prodjay.py keeps NONZERO pixels.  So pack pixel = 0.25 x (sum of 4
integer-valued SP samples), i.e. every pack value is a multiple of 0.25 and the smallest is 0.25.

This script checks that on the packs themselves (a sample of rows per plane) and compares the pack
value distribution with the wcfm SP frames summed and scaled the same way:

    python3 scripts/fm_scale_check.py --out /home/xqian/tmp/wcfm-f3/scale_check.json
"""
import argparse
import glob
import json
import os

import numpy as np

WCFM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = '/home/xqian/toolkit-dev/WC_FM_DINO/training_data/packed_numu_truth_apa0_{P}_20k.npz'
PLANE_CH = {'U': (0, 800), 'V': (800, 1600), 'W': (1600, 2560)}
PCTS = [2, 10, 50, 90, 98, 99.999]


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    q = v * 4.0
    return dict(n=int(v.size), min=float(v.min()), max=float(v.max()),
                frac_quarter_multiple=float(np.mean(np.abs(q - np.round(q)) < 1e-6)),
                frac_integer=float(np.mean(np.abs(v - np.round(v)) < 1e-6)),
                pct={str(p): float(np.percentile(v, p)) for p in PCTS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=5_000_000)
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--frames', nargs='*', default=sorted(glob.glob(os.path.join(WCFM_DIR, 'work/000001_[0-9]*/sim-frames-anode*.tar.bz2'))))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    out = dict(scale=args.scale, pack={}, wcfm={})

    for P in 'UVW':
        z = np.load(PACK.format(P=P))
        f = z['features'][:, 0]
        o = z['offsets']
        idx = rng.choice(f.size, size=min(args.sample, f.size), replace=False)
        s = stats(f[idx])
        # per-event minimum over the first 2000 events (the threshold: is any pixel < 0.25 or == 0?)
        mins = [float(f[o[i]:o[i + 1]].min()) for i in range(min(2000, o.size - 1)) if o[i + 1] > o[i]]
        s['per_event_min_min'] = min(mins)
        s['n_pixels_total'] = int(f.size)
        out['pack'][P] = s
        print(f'[scale] pack {P}: n={f.size} sample={idx.size} min={s["min"]} quarter-multiples={s["frac_quarter_multiple"]:.4f} '
              f'integers={s["frac_integer"]:.4f} p50={s["pct"]["50"]:.1f} p2={s["pct"]["2"]:.2f} p99.999={s["pct"]["99.999"]:.0f}', flush=True)
        del z, f

    from wirecell.util import ario
    acc = {P: [] for P in 'UVW'}
    raw_int = []
    for fn in args.frames:
        a = int(os.path.basename(fn).split('anode')[1].split('.')[0])
        fr = ario.load(fn)
        k = [k for k in fr.keys() if k.startswith('frame_gauss')][0]
        arr = np.asarray(fr[k], dtype=np.float32)
        ch = np.asarray(fr[k.replace('frame', 'channels')])
        nz = arr[arr != 0]
        raw_int.append(float(np.mean(np.abs(nz - np.round(nz)) < 1e-6)))
        q = arr.reshape(arr.shape[0], -1, 4).astype(np.float64).sum(axis=2) * args.scale
        for P, (lo, hi) in PLANE_CH.items():
            rows = (ch - a * 2560 >= lo) & (ch - a * 2560 < hi)
            v = q[rows]
            acc[P].append(v[v > 0])
    out['wcfm']['frames'] = len(args.frames)
    out['wcfm']['sp_samples_integer_fraction'] = float(np.mean(raw_int))
    for P in 'UVW':
        v = np.concatenate(acc[P])
        s = stats(v)
        out['wcfm'][P] = s
        print(f'[scale] wcfm {P} x{args.scale}: n={v.size} min={s["min"]} quarter-multiples={s["frac_quarter_multiple"]:.4f} '
              f'p50={s["pct"]["50"]:.1f} p2={s["pct"]["2"]:.2f} p99.999={s["pct"]["99.999"]:.0f}', flush=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'[scale] wrote {args.out}')


if __name__ == '__main__':
    main()

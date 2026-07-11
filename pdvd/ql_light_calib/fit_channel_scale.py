#!/usr/bin/env python3
"""Data-driven per-channel PE scale (measured_pe_scale) fit for PDVD.

Answers pdvd-questions-dune.md sec 2 from the data side while the official
DAPHNE gain calibration is pending: the cathode X-ARAPUCA ADC-per-PE scale
cannot come from a resolved 1-PE peak (threshold-buried), but the RELATIVE
per-channel scale is measurable from vetted charge->light predictions:

    r_ch = median over vetted (flash, cluster-group) pairs of meas_ch/pred_ch

Ground-truth samples:
  - the 80 beam-flash GOLD pairs (all three runs; selection-free, but a
    single topology: bright beam showers), and
  - the hand-scan KEEP matches (tag labels under work/ql_labels/<tag>/,
    diverse topologies; NOTE they were selected by the 128 nm matcher, a
    mild circularity the gold subset cross-checks).

Per-flash prediction = sum over that flash's kept bundles of the recomputed
per-channel prediction (ablib_gold.predict -- validated exact vs the C++).
Fits are reported for BOTH v5 libraries (128 nm production, 175 nm Xe).

Output per library:
  - per-channel r_ch and n;
  - cathode XA channels normalized to the cathode-group median (relative
    in-group gains -- the main deliverable; group-level model systematics
    mostly cancel);
  - cathode group anchored to the membrane XAs (absolute transfer -- CAVEAT:
    memXA-top and memXA-bot disagree at group level, see stdout);
  - a cfg-ready measured_pe_scale vector (multiplies MEASURED PE;
    scale_ch = anchor/r_ch, masked/unfit channels 1.0), default-OFF knob;
  - train/test validation: fit on even events, apply to odd -> KS before/after.

Usage:
    python3 fit_channel_scale.py 'work/0392*_*/calib-evt*.json' 'work/0393*_*/calib-evt*.json' \
        [--labels ../work/ql_labels/claude] [--table ../data/jjo_triglight_offsets.txt] \
        [--lib 128nm|175nm|both]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablib_gold import GridLib, predict, ks_dis, VUV_EFF, CH_GROUP  # noqa: E402

MEMXA_LIVE = [0, 1, 2, 3, 12, 18, 19]
CATHXA = list(range(4, 12))


def gold_samples(paths, table, win=5.0, pe_min=500.0):
    """(event_key, dump, [bundle]) for the beam-flash gold pairs."""
    tbl = {}
    for line in open(table):
        if line.startswith('#'):
            continue
        f = line.split()
        tbl[(int(f[0]), int(f[2]))] = float(f[6]) - float(f[8])
    for p in sorted(paths):
        m = re.search(r'(\d{5,6})_\d+/calib-evt(\d+)\.json', p.replace('\\', '/'))
        if not m:
            continue
        t_exp = tbl.get((int(m.group(1)), int(m.group(2))))
        if t_exp is None:
            continue
        d = json.load(open(p))
        fl = d['flashes']
        ts = np.array([f['time'] for f in fl])
        i = int(np.abs(ts - t_exp).argmin())
        if abs(ts[i] - t_exp) > win or fl[i]['total_PE'] < pe_min:
            continue
        bs = [b for b in d['bundles'] if b['flash_gid'] == fl[i]['gid']
              and b['total_pred_light'] > 0]
        if not bs:
            continue
        b = max(bs, key=lambda x: x['total_pred_light'])
        yield int(m.group(2)), d, fl[i], [b]


def scan_samples(labels_dir):
    """(event_key, dump, flash, [kept bundles]) per flash from hand-scan labels."""
    for lp in sorted(glob.glob(os.path.join(labels_dir, 'labels-evt*.json'))):
        lab = json.load(open(lp))
        ev = int(lab['event'][len('evt'):])
        dumps = glob.glob(os.path.join(os.path.dirname(os.path.dirname(labels_dir)),
                                       '*_*', lab['source']))
        if not dumps:
            continue
        d = json.load(open(dumps[0]))
        by_gid = {f['gid']: f for f in d['flashes']}
        cl = {c['uid']: c for c in d['clusters']}
        per_flash = {}
        for e in lab['matches']:
            gid = e['flash_gid']
            want = (e['cluster_idents'][0], e['apa'])
            for b in d['bundles']:
                if (b['flash_gid'] == gid and b['apa'] == e['apa']
                        and cl[b['main_cluster']]['ident'] == want[0]):
                    per_flash.setdefault(gid, []).append(b)
                    break
        for gid, bs in per_flash.items():
            yield ev, d, by_gid[gid], bs


def fit(samples, lib, pred_min=0.5, meas_min=0.5):
    """samples -> per-channel ratio lists + per-pair (meas, pred, live, event)."""
    per_ch = [[] for _ in range(len(VUV_EFF))]
    flat = []
    for ev, d, fl, bundles in samples:
        live = np.array([od['active'] and not od['auto_masked'] for od in d['opdets']])
        pred = np.zeros(len(VUV_EFF))
        for b in bundles:
            pred += predict(d, b, lib, live)
        if pred.sum() <= 0:
            continue
        meas = np.array(fl['pe'])
        flat.append((ev, meas, pred, live))
        for j in np.where(live)[0]:
            if pred[j] > pred_min and meas[j] > meas_min:
                per_ch[j].append(meas[j] / pred[j])
    return per_ch, flat


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dumps', nargs='+')
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--labels', default=os.path.join(here, '..', 'work', 'ql_labels', 'claude'))
    ap.add_argument('--table', default=os.path.join(here, '..', 'data',
                                                    'jjo_triglight_offsets.txt'))
    ap.add_argument('--lib', default='both', choices=['128nm', '175nm', 'both'])
    ap.add_argument('--min-n', type=int, default=8,
                    help='min samples for a per-channel fit')
    args = ap.parse_args()

    paths = []
    for pat in args.dumps:
        paths += glob.glob(pat)
    libs = {'128nm': os.path.join(here, '..', 'photlib', 'pdvd-photlib-vis-v5-128nm.json'),
            '175nm': os.path.join(here, '..', 'photlib', 'pdvd-photlib-vis-v5-175nm.json')}
    if args.lib != 'both':
        libs = {args.lib: libs[args.lib]}

    gold = list(gold_samples(paths, args.table))
    scan = list(scan_samples(args.labels)) if os.path.isdir(args.labels) else []
    print(f'samples: {len(gold)} gold pairs + {len(scan)} scanned flashes '
          f'({args.labels if scan else "no labels dir"})')

    for name, meta in libs.items():
        lib = GridLib(meta)
        per_ch, flat = fit(gold + scan, lib)
        per_ch_gold, _ = fit(gold, lib)
        print(f'\n===== {name} =====')
        med = np.full(len(VUV_EFF), np.nan)
        print(f'{"ch":>3} {"group":>10} {"r=meas/pred":>11} {"n":>5}   {"gold-only":>9} {"n":>4}')
        for j in range(len(VUV_EFF)):
            if len(per_ch[j]) >= args.min_n:
                med[j] = np.median(per_ch[j])
                g = (f'{np.median(per_ch_gold[j]):9.3f} {len(per_ch_gold[j]):4d}'
                     if len(per_ch_gold[j]) >= 5 else f'{"-":>9} {len(per_ch_gold[j]):4d}')
                print(f'{j:>3} {CH_GROUP[j]:>10} {med[j]:>11.3f} {len(per_ch[j]):>5}   {g}')

        cath = med[CATHXA]
        cmed = np.nanmedian(cath)
        print(f'\ncathode XA in-group relative gains (r_ch / cathode median {cmed:.3f}):')
        for j, r in zip(CATHXA, cath):
            if np.isfinite(r):
                print(f'  ch{j:2d}: {r / cmed:6.2f}')
        print(f'  in-group spread max/min: {np.nanmax(cath) / np.nanmin(cath):.1f}')

        mem = med[MEMXA_LIVE]
        mem_med = np.nanmedian(mem)
        mtop = np.nanmedian(med[[0, 1, 2, 3]])
        mbot = np.nanmedian(med[[12, 18, 19]])
        print(f'membrane XA anchor: all {mem_med:.3f} (top {mtop:.3f} vs bottom {mbot:.3f} '
              f'-- x{max(mtop, mbot) / min(mtop, mbot):.1f} apart, absolute transfer caveat)')
        print(f'cathode-group scale vs membrane anchor: {cmed / mem_med:.3f}')

        # cfg-ready measured_pe_scale (multiplies MEASURED PE; anchor = membrane
        # XA median so the correction is ~1 there and QtoL keeps the residual)
        scale = np.where(np.isfinite(med), mem_med / med, 1.0)
        print('measured_pe_scale (cfg order, unfit/masked = 1.0):')
        print('  [' + ',\n   '.join(', '.join(f'{s:.3f}' for s in scale[k:k + 8])
                                    for k in range(0, len(scale), 8)) + ']')

        # train/test: fit on even events, apply to odd
        train = [s for s in gold + scan if s[0] % 2 == 0]
        test_flat = [f for f in flat if f[0] % 2 == 1]
        per_ch_tr, _ = fit(train, lib)
        med_tr = np.array([np.median(v) if len(v) >= args.min_n else np.nan
                           for v in per_ch_tr])
        sc_tr = np.where(np.isfinite(med_tr), np.nanmedian(med_tr[MEMXA_LIVE]) / med_tr, 1.0)
        ks0 = [ks_dis(meas, pred, live) for _, meas, pred, live in test_flat]
        ks1 = [ks_dis(meas * sc_tr, pred, live) for _, meas, pred, live in test_flat]
        print(f'train/test (fit even events, apply odd; n_test={len(test_flat)}): '
              f'KS median {np.median(ks0):.3f} -> {np.median(ks1):.3f}')


if __name__ == '__main__':
    main()

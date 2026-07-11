#!/usr/bin/env python3
"""Q2 revisit: can ONE uniform cathode-gain constant be fitted from data?

The per-channel measured_pe_scale fit was rejected (held-out KS neutral or
worse, fit_channel_scale.py). This asks the weaker question the PoF
front-end asymmetry actually implies: a SINGLE gain constant for the
cathode X-ARAPUCA group (ch4-11, Power-over-Fiber) relative to the membrane
XAs, under the now-default 175 nm/Xe model.

Method
- vetted samples = beam-flash GOLD pairs (all three runs; predictions are
  recomputed from cluster geometry, so dump vintage is irrelevant) +
  hand-scan KEEP matches (tag claude-xe, the 10 Xe-reprocessed events).
- prediction = ablib_gold.predict with the 175 nm grid + official eff_Xe
  (exactness re-validated against the stored pred_pe of one Xe dump).
- per-sample group ratio R = (meas_cath/pred_cath) / (meas_mem/pred_mem),
  kept only when BOTH groups are genuinely lit (pred >= pred_min each).
  QtoL and any global model scale cancel in R.
- deliverable 1: the constant -- median R, [16,84]% spread, gold-only and
  per-run cross-checks.
- deliverable 2: deployability -- scale ch4-11 measured PE by 1/R fitted
  on even events, KS before/after on odd events (gold+scan and gold-only).

Usage:
    python3 fit_uniform_gain.py 'work/0392*_*/calib-evt*.json' \
        'work/0393*_*/calib-evt*.json' \
        [--labels ../work/ql_labels/claude-xe] [--table <jjo table>]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import ablib_gold  # noqa: E402
from ablib_gold import GridLib, predict, ks_dis  # noqa: E402
from fit_channel_scale import gold_samples, scan_samples, MEMXA_LIVE, CATHXA  # noqa: E402


def xe_eff():
    m = json.load(open(os.path.join(here, '..', 'photlib',
                                    'pdvd-photlib-chanmap.json')))
    eff = np.zeros(40)
    for c in m['channels']:
        w = c.get('wct_opdet')
        if w is not None:
            eff[w] = c.get('eff_Xe') or 0.0
    return eff


def validate_against_dump(dump_path, lib, n_check=50):
    """Python predict must reproduce a Xe dump's stored pred_pe exactly."""
    d = json.load(open(dump_path))
    live = np.array([od['active'] and not od['auto_masked']
                     for od in d['opdets']])
    devs = []
    for b in d['bundles']:
        if b['total_pred_light'] <= 1.0:
            continue
        ours = predict(d, b, lib, live)
        ref = np.array(b['pred_pe'])
        s = ref[live].sum()
        if s > 0:
            devs.append(np.abs(ours - ref)[live].max() / s)
        if len(devs) >= n_check:
            break
    print(f'validation vs {os.path.basename(dump_path)}: '
          f'{len(devs)} bundles, worst rel dev {max(devs):.6f}')


def group_ratios(samples, lib, pred_min=5.0):
    """Per-sample R = (meas/pred)_cath / (meas/pred)_mem where both lit."""
    out, flat = [], []
    for ev, d, fl, bundles in samples:
        live = np.array([od['active'] and not od['auto_masked']
                         for od in d['opdets']])
        pred = np.zeros(40)
        for b in bundles:
            pred += predict(d, b, lib, live)
        if pred.sum() <= 0:
            continue
        meas = np.array(fl['pe'])
        flat.append((ev, meas, pred, live))
        ci = [j for j in CATHXA if live[j]]
        mi = [j for j in MEMXA_LIVE if live[j]]
        pc, pm = pred[ci].sum(), pred[mi].sum()
        mc, mm = meas[ci].sum(), meas[mi].sum()
        if pc >= pred_min and pm >= pred_min and mc > 0 and mm > 0:
            out.append((ev, (mc / pc) / (mm / pm)))
    return out, flat


def summ(rs):
    if not rs:
        return 'n=0'
    v = np.array([r for _, r in rs])
    lo, med, hi = np.percentile(v, [16, 50, 84])
    return f'median {med:.2f}  [16,84]% [{lo:.2f}, {hi:.2f}]  n={len(v)}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dumps', nargs='+')
    ap.add_argument('--labels', default=os.path.join(
        here, '..', 'work', 'ql_labels', 'claude-xe'))
    ap.add_argument('--table', default=os.path.join(
        here, '..', 'data', 'jjo_triglight_offsets.txt'))
    args = ap.parse_args()

    ablib_gold.VUV_EFF = xe_eff()          # predict() reads it at call time
    lib = GridLib(os.path.join(here, '..', 'photlib',
                               'pdvd-photlib-vis-v5-175nm.json'))

    paths = []
    for pat in args.dumps:
        paths += glob.glob(pat)

    xe_dumps = sorted(p for p in paths if '039252_' in p)
    if xe_dumps:
        validate_against_dump(xe_dumps[0], lib)

    gold = list(gold_samples(paths, args.table))
    scan = list(scan_samples(args.labels)) if os.path.isdir(args.labels) else []
    print(f'samples: {len(gold)} gold pairs + {len(scan)} scanned flashes')

    rs_gold, flat_gold = group_ratios(gold, lib)
    rs_scan, flat_scan = group_ratios(scan, lib)
    rs_all = rs_gold + rs_scan
    flat_all = flat_gold + flat_scan

    print('\ncathode/membrane uniform gain constant R:')
    print(f'  gold+scan : {summ(rs_all)}')
    print(f'  gold only : {summ(rs_gold)}')
    print(f'  scan only : {summ(rs_scan)}')
    for run in ('039252', '039253', '039349'):
        sub = [p for p in paths if f'{run}_' in p]
        rs_r, _ = group_ratios(list(gold_samples(sub, args.table)), lib)
        print(f'  gold {run}: {summ(rs_r)}')

    # deployability: 1/R on cathode measured PE, cross-validated across
    # independent sample sets (gold is a single topology and its R is
    # run-dependent -- see stdout -- so an even/odd event split would be
    # dominated by one run's beam topology).
    def ks_ab(name, R, flat):
        if not flat:
            return
        scale = np.ones(40)
        scale[CATHXA] = 1.0 / R
        ks0 = [ks_dis(m, p, lv) for _, m, p, lv in flat]
        ks1 = [ks_dis(m * scale, p, lv) for _, m, p, lv in flat]
        n_b = sum(1 for a, b in zip(ks0, ks1) if b < a - 1e-9)
        print(f'  {name}: R={R:.2f} -> KS median {np.median(ks0):.3f} -> '
              f'{np.median(ks1):.3f} (improved {n_b}/{len(ks0)})')

    print('\nheld-out validation (scale = 1/R on ch4-11 measured PE):')
    if rs_scan:
        R_scan = float(np.median([r for _, r in rs_scan]))
        ks_ab('fit scan -> test gold (all runs)', R_scan, flat_gold)
        evs = sorted({ev for ev, *_ in scan})
        first, last = set(evs[:len(evs) // 2]), set(evs[len(evs) // 2:])
        r_tr = [r for ev, r in rs_scan if ev in first]
        if r_tr:
            ks_ab(f'fit scan evts 1-{len(first)} -> test scan evts rest',
                  float(np.median(r_tr)),
                  [f for f in flat_scan if f[0] in last])
    if rs_gold:
        ks_ab('fit gold -> test scan',
              float(np.median([r for _, r in rs_gold])), flat_scan)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Fit the PDVD Q/L PE scale (QtoL) from beam-trigger GOLD pairs.

Ground truth without a hand scan: in the CTB beam-triggered runs the beam
flash's folded time is KNOWN per event (t = tc_us - charge_bde_us from the
trigoff tree / jjo table -- the same anchor check_trigger_flash.py validated
to -0.9 us median), and its charge partner is the beam interaction = the
bundle with the largest predicted light on that flash.  Each such pair is a
true (cluster, flash) match by construction, so

    QtoL = median over gold pairs of  sum(meas PE) / sum(pred PE at QtoL=1)

*** RESULT 2026-07-10 (120 DIAG=2 dumps, QtoL=1, 80 gold pairs): ***
    QtoL = 0.11,  [16,84]% = [0.03, 0.25]
  Per-channel-group medians are consistent (cathode XA 0.13, bottom PMT 0.16,
  z-PMT 0.12, top membrane XA 0.10) => the official VUVEfficiency values are
  NOT double-counted; the ~10x over-prediction is one global normalization.

*** DO NOT fit QtoL from auto-selected "good-KS" bundles. ***
  With a badly-scaled QtoL the LASSO is amplitude-inert (the per-flash
  background column absorbs the fit and bundle strengths stay ~ at their
  initial value), so auto-selection is dominated by accidental pairings whose
  flash is ~40x brighter than the prediction.  That route gave "QtoL = 40" --
  wrong by a factor ~350.  Symmetric trap to the crosser-scan offset fit
  (fit_trigger_offset.py): always anchor on external ground truth.

Also reported, for the record and for measured_pe_scale work later:
  - per-channel meas/pred medians (relative PE calibration): within the
    cathode-XA group alone they span x13 (ch10 0.68 vs ch7 0.05) -- the known
    "no resolvable 1-PE peak" cathode SPE calibration question.  A per-channel
    correction fitted here improves the gold-pair KS only 0.43 -> 0.33 median
    (train/test split), i.e. shape mismatch is NOT just channel gains (bright
    beam showers; possible saturation; single topology) -- so the per-channel
    scale is NOT baked into the cfg yet.  Refit on hand-scan GT.
  - gold-pair bundle KS distribution: median 0.45, 0% below the hc ladder
    KS ceilings (0.06-0.10) => on PDVD round 1 the ladder cannot pass true
    matches; selection is LASSO-amplitude-driven.

Usage:
    python3 fit_qtol_gold.py work/0392*_*/calib-evt*.json work/0393*_*/calib-evt*.json \
        [--table ../data/jjo_triglight_offsets.txt] [--win 5] [--pe-min 500]

Dumps must be QtoL=1 references (the script unscales by the dump's own
quality_params.QtoL otherwise).
"""
import argparse
import glob
import json
import re

import numpy as np

MASK = {13, 24, 27, 28, 29, 32, 34, 39}
CH_GROUP = {**{i: 'memXA-top' for i in (0, 1, 2, 3)},
            **{i: 'cathXA' for i in range(4, 12)},
            12: 'memXA-bot', 13: 'memXA-bot', 18: 'memXA-bot', 19: 'memXA-bot',
            **{i: 'zPMT' for i in (14, 15, 16, 17, 20, 21, 22, 23)},
            **{i: 'botPMT' for i in range(24, 40)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dumps', nargs='+', help='calib-evt*.json paths')
    ap.add_argument('--table', default='../data/jjo_triglight_offsets.txt')
    ap.add_argument('--win', type=float, default=5.0,
                    help='flash acceptance around tc_us - charge_bde_us [us]')
    ap.add_argument('--pe-min', type=float, default=500.0,
                    help='min beam-flash total PE (kills no-beam-flash events)')
    args = ap.parse_args()

    tbl = {}
    for line in open(args.table):
        if line.startswith('#'):
            continue
        f = line.split()
        tbl[(int(f[0]), int(f[2]))] = float(f[6]) - float(f[8])  # tc - charge_bde

    live = [i for i in range(40) if i not in MASK]
    gold, per_ch = [], [[] for _ in range(40)]
    for p in args.dumps:
        m = re.search(r'(\d{5,6})_\d+/calib-evt(\d+)\.json', p.replace('\\', '/'))
        if not m:
            continue
        t_exp = tbl.get((int(m.group(1)), int(m.group(2))))
        if t_exp is None:
            continue
        d = json.load(open(p))
        qtol_ref = d.get('quality_params', {}).get('QtoL', 1.0) or 1.0
        fl = d['flashes']
        ts = np.array([f['time'] for f in fl])
        i = int(np.abs(ts - t_exp).argmin())
        if abs(ts[i] - t_exp) > args.win or fl[i]['total_PE'] < args.pe_min:
            continue
        gid = fl[i]['gid']
        bs = [b for b in d['bundles'] if b['flash_gid'] == gid and b['total_pred_light'] > 0]
        if not bs:
            continue
        b = max(bs, key=lambda x: x['total_pred_light'])
        pe = np.array(fl[i]['pe'])
        pred = np.array(b['pred_pe']) / qtol_ref
        gold.append((pe[live].sum() / pred[live].sum(), b['ks_dis']))
        for j in live:
            if pred[j] > 1e-3 and pe[j] > 0.5:
                per_ch[j].append(pe[j] / pred[j])

    if not gold:
        raise SystemExit('no gold pairs found -- wrong dumps or table?')
    r = np.array([g[0] for g in gold])
    ks = np.array([g[1] for g in gold])
    print(f'gold pairs: {len(gold)}')
    print(f'QtoL = {np.median(r):.3f}   [16,84]% = [{np.percentile(r, 16):.3f}, '
          f'{np.percentile(r, 84):.3f}]')
    print(f'gold bundle KS: median {np.median(ks):.3f}, <0.10: {(ks < 0.10).mean():.0%}, '
          f'<0.20: {(ks < 0.20).mean():.0%}')
    print('\nper-channel meas/pred medians (relative PE calibration; n>=5):')
    groups = {}
    for j in live:
        v = per_ch[j]
        if len(v) < 5:
            continue
        med = float(np.median(v))
        groups.setdefault(CH_GROUP[j], []).append(med)
        print(f'  ch {j:2d} {CH_GROUP[j]:>10}: {med:7.2f}  (n={len(v)})')
    print('group medians:')
    for g_, ms in sorted(groups.items()):
        print(f'  {g_:>10}: {np.median(ms):7.2f}')


if __name__ == '__main__':
    main()

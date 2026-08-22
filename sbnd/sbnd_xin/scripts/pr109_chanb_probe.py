#!/usr/bin/env python3
"""doc pr/109 sec 9 -- decompose dU(ON-OFF) into "cells that lost prediction" vs
"cells whose prediction merely moved", read-only on arms that already exist.

Why this exists.  sec 9 established that fit_exclusion reaches charge_pred through
exactly two channels:

  Channel A  the trajectory POSITIONS move (form_map_graph passes 1-2 feed
             multi_trajectory_fit; trimmed associations change the least squares)
  Channel B  trajectory POINTS are deleted (pass 3 drops points whose plane
             quantity fell to zero, TrackFitting.cxx:3617), so no fit point
             remains to predict that charge at all

Channel B is logged and measured (TrackFitting.cxx:8909-8918): exclusion-driven on
5/6 SBND events and 0/6 uBooNE, but only 1-9 points of 600-5100.  This probe asks
whether that handful can carry the effect: a deleted JUNCTION point owns a
disproportionate share of near-vertex coupling, so a few deletions could strip
thousands of cells' worth of prediction.

Test.  Over the same near-vertex box the sec 4 metric uses, restricted to cells
present in BOTH arms, classify every cell by what happened to its prediction:

  lost      yhat_OFF > 0  and  yhat_ON == 0     <- the Channel-B signature
  gained    yhat_ON  > 0  and  yhat_OFF == 0
  moved     both > 0, |y-yhat| changed
  neither   both == 0 (uncovered in both arms; sec 7's population)

and compare  sum q(lost) - sum q(gained)  against the actual increase in the dU
numerator  sum|y-yhat|_ON - sum|y-yhat|_OFF.  If they are comparable, Channel B is
the answer.  If the numerator moves while lost-gained stays near zero, the effect
is Channel A -- the predictions moved rather than vanished.

Reuses pr109_2d_resid.py's loaders unchanged so the cell selection (single-owner,
q>0, live, in-box) is bit-for-bit the sec 4 selection.

Usage:
  pr109_chanb_probe.py --arm ON=a.root[:wct] --arm OFF=b.root[:wct] \
                       [--box-cm 3.0] [--max-junctions 3] [--tag evt] [--tsv out]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr109_2d_resid import load_arm, affine, regions


def classify(on, off, anchor, box_cm, anchor_arm):
    """Per-cell ON-vs-OFF prediction classes inside the anchor's 2-D box.

    The box is taken from `anchor_arm` so both arms are scored on the SAME
    window; sec 9's box-centre confound is reported by the caller.
    """
    pred, res, grad = affine(anchor_arm, anchor, 30.0)
    if pred is None:
        return None
    c = pred(anchor)
    half = grad * box_cm

    acc = dict(n=0, sy=0.0, sabs_on=0.0, sabs_off=0.0,
               n_lost=0, q_lost=0.0, n_gained=0, q_gained=0.0,
               n_moved=0, q_moved=0.0, dabs_moved=0.0,
               n_neither=0, q_neither=0.0)
    per_plane = {p: dict(q_lost=0.0, q_gained=0.0, dabs=0.0) for p in 'UVW'}

    for (ch, ts), e_on in on['cells'].items():
        pl = on['plane_of'](ch)
        j = 'UVW'.index(pl)
        if abs(ch - c[j]) > half[j] or abs(ts - c[3]) > half[3]:
            continue
        e_off = off['cells'].get((ch, ts))
        if e_off is None:
            continue                      # sec 3 control 1: cell sets are identical
        if e_on['q'] <= 0 or e_off['q'] <= 0:
            continue
        if on['is_dead'](ch, ts) or off['is_dead'](ch, ts):
            continue
        # single-owner only, matching the sec 4 selection on both sides
        if len({cid for cid, _ in e_on['preds']}) > 1:
            continue
        if len({cid for cid, _ in e_off['preds']}) > 1:
            continue

        y = e_on['q']
        p_on = e_on['preds'][0][1]
        p_off = e_off['preds'][0][1]
        a_on, a_off = abs(y - p_on), abs(y - p_off)

        acc['n'] += 1
        acc['sy'] += y
        acc['sabs_on'] += a_on
        acc['sabs_off'] += a_off
        per_plane[pl]['dabs'] += a_on - a_off

        if p_off != 0 and p_on == 0:
            acc['n_lost'] += 1;   acc['q_lost'] += y;   per_plane[pl]['q_lost'] += y
        elif p_on != 0 and p_off == 0:
            acc['n_gained'] += 1; acc['q_gained'] += y; per_plane[pl]['q_gained'] += y
        elif p_on == 0 and p_off == 0:
            acc['n_neither'] += 1; acc['q_neither'] += y
        else:
            acc['n_moved'] += 1; acc['q_moved'] += y; acc['dabs_moved'] += a_on - a_off

    acc['_center'] = c
    acc['_planes'] = per_plane
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True,
                    help='LABEL=path.root[:wct|wcp]; exactly two, labelled ON and OFF')
    ap.add_argument('--box-cm', type=float, default=3.0)
    ap.add_argument('--max-junctions', type=int, default=3)
    ap.add_argument('--ticks-per-slice', type=int, default=4)
    ap.add_argument('--tag', default='evt')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        label, rest = spec.split('=', 1)
        path, _, kind = rest.partition(':')
        arms[label] = load_arm(path, kind or 'wct', args.ticks_per_slice)
    if set(arms) != {'ON', 'OFF'}:
        sys.exit('need exactly --arm ON=... --arm OFF=...')
    on, off = arms['ON'], arms['OFF']

    rows = []
    print('# %s  box %.2f cm  anchors from OFF' % (args.tag, args.box_cm))
    print('%-9s %6s %10s %8s %10s %8s %10s %10s %10s' %
          ('region', 'ncell', 'sum_y', 'n_lost', 'q_lost', 'n_gain', 'q_gain',
           'd_numer', 'lost-gain'))
    tot = None
    for name, anchor in regions(off, args.max_junctions):
        a = classify(on, off, anchor, args.box_cm, off)
        if a is None or a['n'] == 0:
            continue
        dnum = a['sabs_on'] - a['sabs_off']
        net = a['q_lost'] - a['q_gained']
        print('%-9s %6d %10.3g %8d %10.3g %8d %10.3g %10.3g %10.3g' %
              (name, a['n'], a['sy'], a['n_lost'], a['q_lost'],
               a['n_gained'], a['q_gained'], dnum, net))
        rows.append((name, a, dnum, net))
        if tot is None:
            tot = {k: v for k, v in a.items() if not k.startswith('_')}
        else:
            for k in tot:
                tot[k] += a[k]

    if tot is not None:
        dnum = tot['sabs_on'] - tot['sabs_off']
        net = tot['q_lost'] - tot['q_gained']
        print('%-9s %6d %10.3g %8d %10.3g %8d %10.3g %10.3g %10.3g' %
              ('POOLED', tot['n'], tot['sy'], tot['n_lost'], tot['q_lost'],
               tot['n_gained'], tot['q_gained'], dnum, net))
        print('  U_ON  %.4f   U_OFF %.4f   dU %+.4f' %
              (tot['sabs_on'] / tot['sy'], tot['sabs_off'] / tot['sy'], dnum / tot['sy']))
        print('  moved-cell share of d_numer: %.3g of %.3g  (%.0f%%)' %
              (tot['dabs_moved'], dnum,
               100.0 * tot['dabs_moved'] / dnum if dnum else float('nan')))
        print('  cells: lost %d  gained %d  moved %d  uncovered-both %d  (q_neither %.3g)' %
              (tot['n_lost'], tot['n_gained'], tot['n_moved'], tot['n_neither'],
               tot['q_neither']))

    if args.tsv:
        new = not os.path.exists(args.tsv)
        with open(args.tsv, 'a') as f:
            if new:
                f.write('tag region ncell sum_y n_lost q_lost n_gain q_gain '
                        'd_numer lost_minus_gain dabs_moved n_moved n_neither q_neither\n')
            for name, a, dnum, net in rows:
                f.write('%s %s %d %.6g %d %.6g %d %.6g %.6g %.6g %.6g %d %d %.6g\n' %
                        (args.tag, name, a['n'], a['sy'], a['n_lost'], a['q_lost'],
                         a['n_gained'], a['q_gained'], dnum, net, a['dabs_moved'],
                         a['n_moved'], a['n_neither'], a['q_neither']))


if __name__ == '__main__':
    main()

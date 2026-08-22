#!/usr/bin/env python3
"""doc pr/109 sec 7 -- where does the UNPREDICTED near-vertex charge come from?

For every cell in the 2-D shadow of a sphere around each anchor (single-owner,
charge>0, not on a dead channel), classify it as covered (charge_pred != 0) or
uncovered (charge_pred == 0), then ask WHERE the uncovered ones sit relative to
the fitted trajectory of their own cluster:

  d_wire, d_slice = distance to the nearest projection of a fitted point of the
                    SAME cluster (pu/pv/pw, pt from T_rec_charge)

The dQ/dx system only inserts a response entry for cells inside a +-10 wire /
+-10 slice window of a sampled point (TrackFitting.cxx:6801-6872, search_range;
prototype PR3DCluster_multi_dQ_dx_fit.h:368-426).  So:
  uncovered AND outside the window  -> the cluster owns charge its trajectory
                                       never reaches (a pattern-recognition /
                                       coverage statement)
  uncovered AND inside the window   -> the coupling itself evaluated to zero
                                       (a fit-input statement)
"""
import argparse, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr109_2d_resid import load_arm, affine, regions


def probe(arm, box_cm, max_j):
    pts = arm['pts']
    out = []
    for rname, A in regions(arm, max_j):
        pred, res, grad = affine(arm, A, 30.0)
        if pred is None:
            continue
        c = pred(A); half = grad * box_cm
        for (ch, ts), e in arm['cells'].items():
            pl = arm['plane_of'](ch); j = 'UVW'.index(pl)
            if abs(ch - c[j]) > half[j] or abs(ts - c[3]) > half[3]:
                continue
            if e['q'] <= 0 or arm['is_dead'](ch, ts):
                continue
            cids = {cid for cid, _ in e['preds']}
            if len(cids) > 1:
                continue
            cid = e['preds'][0][0]
            m = pts['cl'] == cid
            if not m.any():
                dw = ds = np.inf
            else:
                dw = np.abs(pts['P'][m, j] - ch)
                ds = np.abs(pts['P'][m, 3] - ts)
                k = np.argmin(np.maximum(dw / 10.0, ds / 10.0))
                dw, ds = dw[k], ds[k]
            out.append((rname, pl, e['q'], e['preds'][0][1], dw, ds,
                        1 if cid == pts['main'] else 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True, help='LABEL=path[:kind]')
    ap.add_argument('--box-cm', type=float, default=3.0)
    ap.add_argument('--max-junctions', type=int, default=3)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    for spec in args.arm:
        lab, _, rest = spec.partition('=')
        path, _, kind = rest.partition(':')
        arm = load_arm(path, kind or 'wct', 4)
        r = probe(arm, args.box_cm, args.max_junctions)
        if not r:
            print('%s %s: no cells' % (args.tag, lab)); continue
        q = np.array([x[2] for x in r]); p = np.array([x[3] for x in r])
        dw = np.array([x[4] for x in r]); ds = np.array([x[5] for x in r])
        main_ = np.array([x[6] for x in r], bool)
        unc = p == 0
        inwin = (dw <= 10) & (ds <= 10)
        tot = q.sum()
        print('%-14s %-8s cells %5d  uncov cells %5.1f%%  uncov charge %5.1f%%  '
              'med q cov %6.0f unc %6.0f | of the UNCOVERED charge: in-window %5.1f%%, '
              'out-of-window %5.1f%% (med d_wire %5.1f d_slice %5.1f), on main cluster %5.1f%%'
              % (args.tag, lab, len(q), 100 * unc.mean(), 100 * q[unc].sum() / tot,
                 np.median(q[~unc]) if (~unc).any() else 0,
                 np.median(q[unc]) if unc.any() else 0,
                 100 * q[unc & inwin].sum() / max(q[unc].sum(), 1),
                 100 * q[unc & ~inwin].sum() / max(q[unc].sum(), 1),
                 np.median(dw[unc]) if unc.any() else 0,
                 np.median(ds[unc]) if unc.any() else 0,
                 100 * q[unc & main_].sum() / max(q[unc].sum(), 1)))


if __name__ == '__main__':
    main()

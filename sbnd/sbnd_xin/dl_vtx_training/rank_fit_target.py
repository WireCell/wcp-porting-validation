#!/usr/bin/env python3
'''doc pr/106 sec 4.2 -- joint (multi-dimensional) fit of the re-rank weights ON THE TARGET METRIC.
(Distinct from pr/77's rank_fit.py: that one ranks on 11 standardized features against the 1 cm ruler;
this one uses exactly the code's nine knobs and the pre-DL-candidate target of doc pr/106.)

The live scorer is linear in its knobs:
    total = scale*dl + w_snap*s_snap + w_fwd_z*s_fwd_z + w_clen*s_clen
          + w_isol*s_isol + w_main*s_main + w_fv*s_fv
          + [votes>=1] * w_topo * (frac - center)
so "which candidate wins" is a pairwise-ranking problem and "accept or fall
back" is a threshold on the winner.  Instead of scanning, fit all weights at
once: L2-regularised pairwise logistic regression (RankSVM-style) on every
(target, other admitted candidate) pair of the tuning set, shrunk toward the
production theta; then pick min_accept by a 1-D scan of the fitted scorer.
Exactly the same decision space as the code (vtx_target_eval.decide), so the
fitted theta is directly replayable and live-closable.  5-fold CV on the
tuning set reports generalisation before any lockbox read.

Repro: ./rank_fit_target.py <vtx_target_eval args: --carried-tags ... --orig-tags ...
                      --exclude-events runs/vtx106/lockbox.txt> [--lam 1.0]
'''
import argparse
import sys
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vtx_target_eval as V  # noqa: E402

# feature order == theta keys; the topo term is two features
# (frac*I, I) with w_topo = c_frac and center = -c_I / w_topo.
FEATS = ['scale', 'w_snap', 'w_fwd_z', 'w_clen', 'w_isol', 'w_main', 'w_fv', 'w_topo', 'c_topo_I']
PROD_VEC = np.array([1000.0, 1, 1, 1, 1, 1, 1, 0.0, 0.0])
# internal scaling so that one unit of each coefficient is comparable:
# dl_score is O(1) while scale is O(1000) -> feature dl is multiplied by 1000
# and the fitted coefficient is then 'scale/1000'.
FSCALE = np.array([1000.0, 1, 1, 1, 1, 1, 1, 1, 1])


def feat(r):
    I = 1.0 if r['votes'] >= 1 else 0.0
    return np.array([r['dl'], r['geo']['s_snap'], r['geo']['s_fwd_z'], r['geo']['s_clen'],
                     r['geo']['s_isol'], r['geo']['s_main'], r['geo']['s_fv'],
                     r['frac'] * I, I]) * FSCALE


def vec_to_theta(v):
    v = v / FSCALE
    th = dict(scale=float(v[0] * 1000.0), w_snap=float(v[1]), w_fwd_z=float(v[2]), w_clen=float(v[3]),
              w_isol=float(v[4]), w_main=float(v[5]), w_fv=float(v[6]))
    w_topo = float(v[7])
    th['w_topo'] = w_topo
    th['center'] = float(-v[8] / w_topo) if abs(w_topo) > 1e-9 else 0.0
    return th


def pairs(events):
    X = []
    for ev in events:
        if not ev.admitted5:
            continue
        rt = next(r for r in ev.rows if r['vid'] == ev.target)
        ft = feat(rt)
        for r in ev.rows:
            if r['vid'] != ev.target:
                X.append(ft - feat(r))
    return np.array(X)


def fit(X, lam, w0, iters=50):
    '''minimise sum log(1+exp(-w.x)) + lam/2 |w - w0|^2 by Newton's method.'''
    w = w0.copy()
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = -(X.T @ (1.0 - p)) + lam * (w - w0)
        S = p * (1.0 - p)
        H = (X * S[:, None]).T @ X + lam * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    return w


def best_threshold(events, th, grid=(0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)):
    best = None
    for m in grid:
        t = dict(th, min_accept=float(m))
        h = V.score(events, t)['hit']
        if best is None or h > best[0]:
            best = (h, m)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sbnd-root', default=V.ROOT)
    ap.add_argument('--carried-tags', nargs='*', default=[])
    ap.add_argument('--orig-tags', nargs='*', default=[])
    ap.add_argument('--harv-base', default='work-vtx106-harv-base-{sample}')
    ap.add_argument('--harv-rows', default='work-vtx106-harv-topo3-{sample}')
    ap.add_argument('--live-arms', nargs='*', default=V.DEFAULT_LIVE)
    ap.add_argument('--ipw-tsv', default=os.path.join(HERE, 'runs/ipw-vtx100-20260820.tsv'))
    ap.add_argument('--exclude-events')
    ap.add_argument('--only-events')
    ap.add_argument('--dmax', type=float, default=3.0)
    ap.add_argument('--lam', type=float, nargs='*', default=[0.1, 1.0, 10.0, 100.0],
                    help='L2 shrinkage toward production theta (units of pairs)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--eval-events', help='held-out event file to evaluate the final fit on (ONE read)')
    args = ap.parse_args()

    events, _ = V.load_events(args)
    prim = [ev for ev in events if ev.d_target <= args.dmax]
    X = pairs(prim)
    print('tuning events %d, admitted-target events %d, pairs %d' % (
        len(prim), sum(ev.admitted5 for ev in prim), len(X)))
    print('production: hit %d/%d' % (V.score(prim, V.PROD)['hit'], len(prim)))
    # pair-ranking accuracy of production
    print('pairwise accuracy of production theta: %.3f' % np.mean(X @ PROD_VEC > 0))

    rng = np.random.RandomState(106)
    fold = rng.randint(0, args.folds, size=len(prim))
    results = {}
    for lam in args.lam:
        cv_hit = 0
        for k in range(args.folds):
            tr = [ev for ev, f in zip(prim, fold) if f != k]
            te = [ev for ev, f in zip(prim, fold) if f == k]
            w = fit(pairs(tr), lam, PROD_VEC)
            th = vec_to_theta(w)
            h, m = best_threshold(tr, th)
            cv_hit += V.score(te, dict(th, min_accept=float(m)))['hit']
        w = fit(X, lam, PROD_VEC)
        th = vec_to_theta(w)
        h, m = best_threshold(prim, th)
        th['min_accept'] = float(m)
        results[lam] = (cv_hit, h, th)
        print('\nlam=%g  5-fold CV hit %d/%d  |  in-sample %d/%d (min_accept %g)  pairwise acc %.3f' % (
            lam, cv_hit, len(prim), h, len(prim), m, np.mean(X @ w > 0)))
        print('  theta: ' + ', '.join('%s=%.3g' % (k, th[k]) for k in
                                     ('scale', 'w_snap', 'w_fwd_z', 'w_clen', 'w_isol', 'w_main', 'w_fv', 'w_topo', 'center')))
        print(V.HDR)
        print(V.fmt_row('fit lam=%g' % lam, V.score(prim, th)))
    best_lam = max(results, key=lambda l: results[l][0])
    print('\nbest by CV: lam=%g -> CV %d, in-sample %d; theta %s' % (best_lam, results[best_lam][0], results[best_lam][1], results[best_lam][2]))
    if args.eval_events:
        a2 = argparse.Namespace(**vars(args))
        a2.exclude_events, a2.only_events = None, args.eval_events
        ev2, _ = V.load_events(a2)
        p2 = [ev for ev in ev2 if ev.d_target <= args.dmax]
        print('\nheld-out (%s): n=%d' % (args.eval_events, len(p2)))
        print(V.HDR)
        print(V.fmt_row('production', V.score(p2, V.PROD)))
        print(V.fmt_row('rank fit lam=%g' % best_lam, V.score(p2, results[best_lam][2])))


if __name__ == '__main__':
    main()

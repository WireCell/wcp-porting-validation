#!/usr/bin/env python3
'''
doc pr/79 step 4b -- ROUTER fit: the oracle-routing diagnostic (rank_sim.py)
showed the composite chooser under PERFECT routing reaches 383/473 (+25)
while every deployable chooser variant is net-negative -- the selection gap
is an acceptance-discrimination problem, not a ranking problem.

This fits a per-EVENT logistic router P(composite pick correct | scoreboard
features) on the live arm rows and replays deployment end-to-end with the
same live anchoring as rank_sim.py:

    route = accept  iff  router_score >= threshold   (nested-OOF tuned)
    chooser = composite argmax (unchanged)

A sigmoid is monotone, so the deployed rule is a LINEAR threshold on raw
event features -- the same knob shape as the rank weights (a weights vector
+ threshold), computable in C++ from quantities already in scope.

Event features (all from recorded usable rows; C++-computable):
    best_total          max composite total (the legacy acceptance quantity)
    margin12            top1-top2 composite total gap (0 if single row)
    dl_score_win        raw dl_score of the composite winner
    snap_dis_win        snap distance of the winner (cm)
    log1p_host_win      log1p(host cluster length, cm) of the winner
    s_main_win          main-cluster bonus of the winner (0 or 2)
    s_fv_win            FV bonus of the winner (0 or 0.5)
    trad_agree          composite winner is also the traditional winner
    n_usable            number of usable candidates
    best_dl             max raw dl_score over usable rows

Usage:
  python3 route_fit.py \
      --arm-roots vtxscan-prod0813=work-nuecc48-ma10-k20 \
                  vtxscan-prod0813-ncpi0=work-ncpi0-ma10-k20 \
                  vtxscan-prod0813-mcp1k=work-mcp1k-ma10-k20 \
      --tsv runs/routefit-k20-<date>.tsv
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from taxonomy import ALL_TAGS
from rank_fit import build_folds
from rank_sim import load_events, replay, evaluate, report, manifest_sets

RFEATURES = ['best_total', 'margin12', 'dl_score_win', 'snap_dis_win',
             'log1p_host_win', 's_main_win', 's_fv_win', 'trad_agree',
             'n_usable', 'best_dl']
X16_NAMES = ['x16_%02d' % i for i in range(16)] + ['x16_best_score']


def rfeat_of(ev):
    tot = ev['tot']
    order = np.argsort(tot)[::-1]
    win = ev['usable'][int(order[0])]
    margin = float(tot[order[0]] - tot[order[1]]) if len(tot) > 1 else 0.0
    base = np.array([
        float(tot.max()), margin, float(win['dl_score']),
        float(win['snap_dis']),
        np.log1p(max(0.0, float(win.get('host_length', 0.0)))),
        float(win.get('s_main', 0.0)), float(win.get('s_fv', 0.0)),
        float(bool(win.get('trad_winner'))), float(len(ev['usable'])),
        max(float(r['dl_score']) for r in ev['usable'])], np.float64)
    if 'cand16' in ev:
        # frozen-net penultimate features at the composite winner's voxel
        # (extract_feats.py; REBUILT-cloud screen, doc pr/79 step 4c)
        win16 = ev['cand16'][int(order[0])].astype(np.float64)
        base = np.concatenate([base, win16,
                               [float(ev['cand_score'].max())]])
    return base


def fit_logistic(X, y, l2, seed, iters=3000, lr=0.1):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, Z.shape[1] + 1)   # last = bias
    Zb = np.hstack([Z, np.ones((len(Z), 1))])
    for it in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Zb @ w)))
        g = Zb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * g
    return w, mu, sd


def rscore(ev, w, mu, sd):
    z = (rfeat_of(ev) - mu) / sd
    return float(z @ w[:-1] + w[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=ALL_TAGS)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--arm-roots', nargs='+', required=True)
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--l2', type=float, default=0.1)
    ap.add_argument('--kfold', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--grid-points', type=int, default=61)
    ap.add_argument('--manifest', default='data/full473/manifest.tsv')
    ap.add_argument('--extra-feats', default=None,
                    help='dir of extract_feats.py npz sidecars: append the '
                         'composite winner\'s frozen-net 16-dim voxel '
                         'features (+best rebuilt score) to the router')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    manifest = args.manifest if os.path.isabs(args.manifest) \
        else os.path.join(here, args.manifest)
    numu, lockbox = manifest_sets(manifest)
    roots = vio.parse_arm_roots(args.arm_roots, args.sbnd_root)
    evs = load_events(args.sbnd_root, args.tags, roots, args.tol, numu)
    n = len(evs)
    n_rec = sum(e['ok_rec'] for e in evs)

    feat_names = list(RFEATURES)
    if args.extra_feats:
        xdir = args.extra_feats if os.path.isabs(args.extra_feats) \
            else os.path.join(here, args.extra_feats)
        mdis = []
        for e in evs:
            if 'F' not in e:
                continue
            npz = np.load(os.path.join(xdir, 'evt%d.npz' % e['evt']))
            if len(npz['cand16']) != len(e['usable']):
                raise SystemExit('candidate count mismatch evt%d' % e['evt'])
            e['cand16'] = npz['cand16']
            e['cand_score'] = npz['cand_score']
            mdis.append(float(npz['match_dis'].max()))
        feat_names += X16_NAMES
        print('extra frozen-net features from %s; worst live<->rebuilt '
              'voxel match: median %.3g cm, p90 %.3g cm, max %.3g cm'
              % (args.extra_feats, np.median(mdis),
                 np.percentile(mdis, 90), max(mdis)))
        print('NOTE: rebuilt-cloud SCREEN only -- deployment needs '
              'live-harvested features (pr/79 lesson 2)')

    # router training set: every event with usable rows; label = composite
    # pick within tol of truth
    def pick_ok(ev):
        return ('F' in ev and bool(np.linalg.norm(
            ev['pos'][ev['comp_i']] - ev['truth']) <= args.tol))
    train_all = [e for e in evs if 'F' in e]
    for e in train_all:
        e['y'] = float(pick_ok(e))
    print('events %d; recorded correct %d; router-trainable %d '
          '(composite pick correct on %d)'
          % (n, n_rec, len(train_all), sum(int(e['y']) for e in train_all)))

    folds = build_folds(train_all, args.kfold, args.seed)

    def fit_on(sub):
        X = np.stack([rfeat_of(e) for e in sub])
        y = np.array([e['y'] for e in sub])
        return fit_logistic(X, y, args.l2, args.seed)

    # nested OOF: fold model + fold threshold from train, applied to val
    def replay_with(ev, w, mu, sd, thr):
        if 'F' not in ev:
            return ev['ok_rec'], ''
        acc = rscore(ev, w, mu, sd) >= thr
        return replay(ev, acc, ev['comp_i'], args.tol)

    oof_ok, thr_folds, oof_dec = 0, [], {}
    for i in range(args.kfold):
        val_evts = {e['evt'] for e in folds[i]}
        train = [e for e in train_all if e['evt'] not in val_evts]
        w, mu, sd = fit_on(train)
        scores = [rscore(e, w, mu, sd) for e in train]
        grid = np.unique(np.percentile(
            scores, np.linspace(0, 100, args.grid_points)))
        best_t, best_ok = None, -1
        for t in grid:
            okc = sum(replay_with(e, w, mu, sd, t)[0] for e in train)
            if okc > best_ok:
                best_ok, best_t = okc, t
        thr_folds.append(best_t)
        for e in evs:
            if e['evt'] in val_evts:
                ok, standin = replay_with(e, w, mu, sd, best_t)
                oof_ok += ok
                oof_dec[e['evt']] = (w, mu, sd, best_t)
    # events in no fold (no usable rows): recorded behavior
    infold = {e['evt'] for f in folds for e in f}
    for e in evs:
        if e['evt'] not in infold:
            oof_ok += e['ok_rec']
    print('\nrouter nested-OOF end-to-end: %d/%d (%+d vs arm)   '
          'per-fold thr %s'
          % (oof_ok, n, oof_ok - n_rec, ['%.3g' % t for t in thr_folds]))

    # full fit (deployment weights) + in-sample grid
    w, mu, sd = fit_on(train_all)
    scores = [rscore(e, w, mu, sd) for e in train_all]
    grid = np.unique(np.percentile(scores,
                                   np.linspace(0, 100, args.grid_points)))
    curve = []
    for t in grid:
        okc = 0
        for e in evs:
            okc += replay_with(e, w, mu, sd, t)[0]
        curve.append((t, okc))
    t_best, ok_best = max(curve, key=lambda c: c[1])

    def d_router(ev):
        if 'F' not in ev:
            return False, None
        return rscore(ev, w, mu, sd) >= t_best, ev['comp_i']
    res = evaluate(evs, d_router, args.tol)
    print('\n== router deployment (composite chooser) ==')
    report('router thr=%.4g (IN-SAMPLE)' % t_best, res, n,
           sum(replay(e, *d_router(e), args.tol)[0]
               for e in evs if e['evt'] in lockbox))
    print('    curve: %s' % '  '.join(
        '%.3g:%d' % c for c in curve[::max(1, len(curve) // 12)]))
    print('\nfixed: %s' % sorted(res['fixed']))
    print('regressed: %s' % sorted(res['regressed']))

    print('\nrouter weights (standardized+bias):')
    for name, wi in sorted(zip(feat_names + ['bias'], w),
                           key=lambda t: -abs(t[1])):
        print('  %-16s %+.3f' % (name, wi))
    w_raw = w[:-1] / sd
    b_raw = float(w[-1] - np.sum(w[:-1] * mu / sd))
    print('# raw-space: score = w.f + b; w = [%s]; b = %.10g; thr = %.10g'
          % (', '.join('%.10g' % v for v in w_raw), b_raw, t_best))

    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('evt\tsample\tok_rec\tok_oof\trouter_oof_thr\n')
            for e in evs:
                if e['evt'] in oof_dec:
                    wf, muf, sdf, tf = oof_dec[e['evt']]
                    ok, _ = replay_with(e, wf, muf, sdf, tf)
                else:
                    ok, tf = e['ok_rec'], float('nan')
                fh.write('%d\t%s\t%d\t%d\t%.4g\n'
                         % (e['evt'], e['sample'], e['ok_rec'], ok, tf))
        print('wrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

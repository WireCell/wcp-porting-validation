#!/usr/bin/env python3
'''
doc pr/77 round 2 (S8b) -- candidate-RANKING fit: the deployed decision is
"which scoreboard candidate wins", a choice among 2-10 discrete options, so
an ~11-parameter pairwise-logistic ranker is matched to an O(100)-label
budget in a way 7.2M-param heatmap regression is not.

Model: P(i beats j) = sigmoid(w . (f_i - f_j))  (RankNet), one weight per
feature, trained on all (truth-candidate, other-candidate) pairs of eligible
events.  Features per scoreboard row (all recorded -- nothing recomputed):
the 7 composite terms + raw dl_score + snap_dis + log1p(host_length) +
trad_winner.  Learned on standardized features, L2-regularized.

Eligible event: >=2 usable rows (dl_snapped, not skipped_by_swap_guard --
same filter as production's rerank/rerank_replay.py) AND the truth pick lies
within --tol of one of them.  Single-candidate events carry no ranking
information; candidate-missing events cannot be fixed at this stage
(taxonomy S8a).

Baselines on the same events: production recorded choice (main_vertex vs
truth) and total-argmax over usable rows.  Closure: scoring with the
production weights (1 on each of the 7 terms, 0 elsewhere) must reproduce
total-argmax exactly.

Out-of-fold via the same stratified kfold as train.py.

Usage:
  python3 rank_fit.py --tsv runs/rankfit-20260814.tsv
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from taxonomy import numu50_set, ALL_TAGS

FEATURES = ['s_dl', 's_snap', 's_fwd_z', 's_clen', 's_isol', 's_main', 's_fv',
            'dl_score', 'snap_dis', 'log1p_host_length', 'trad_winner']
PROD_W = np.array([1.0] * 7 + [0.0] * 4)  # the recorded composite


def feat_of(row):
    return np.array(
        [float(row[t]) for t in
         ('s_dl', 's_snap', 's_fwd_z', 's_clen', 's_isol', 's_main', 's_fv')]
        + [float(row['dl_score']), float(row['snap_dis']),
           np.log1p(max(0.0, float(row.get('host_length', 0.0)))),
           float(bool(row.get('trad_winner')))], dtype=np.float64)


def load_eligible(sbnd_root, tags, tol, roots=None):
    """roots: optional {scan_tag: arm_root} from vio.parse_arm_roots -- fit
    on an explicit arm's scoreboards (doc pr/79 step 4: the live k20 rows)
    instead of the label['source'] prod0813 arm."""
    here = os.path.dirname(os.path.abspath(__file__))
    numu = numu50_set(here)
    evs = []
    n_lab = n_multi = 0
    for label in vio.iter_labels(sbnd_root, tags):
        n_lab += 1
        path = vio.calib_path_in_roots(roots, label) if roots \
            else vio.calib_path_for_label(sbnd_root, label)
        if not os.path.exists(path):
            raise FileNotFoundError('calib missing for evt%d: %s'
                                    % (label['eventNo'], path))
        calib = vio.load_calib(path)
        rows = ((calib.get('vertex_scoreboard') or {}).get('rows')) or []
        usable = [r for r in rows
                  if r.get('dl_snapped') and not r.get('skipped_by_swap_guard')]
        if len(usable) < 2:
            continue
        n_multi += 1
        truth = label['truth_xyz']
        pos = np.array([[r['x'], r['y'], r['z']] for r in usable], np.float32)
        d = np.linalg.norm(pos - truth, axis=1)
        if d.min() > tol:
            continue  # candidate-missing at this stage
        mv = label.get('main_vertex') or {}
        prod = np.array([mv.get('x', np.nan), mv.get('y', np.nan),
                         mv.get('z', np.nan)], np.float32)
        evs.append(dict(
            evt=label['eventNo'], sample=vio.sample_of_label(label, numu),
            corrective=int(vio.is_corrective(label, tol=tol)),
            F=np.stack([feat_of(r) for r in usable]),
            truth_i=int(d.argmin()),
            totals=np.array([float(r['total']) for r in usable]),
            prod_correct=int(np.linalg.norm(prod - truth) <= tol)))
    print('labels %d; >=2 usable rows %d; eligible (truth reachable) %d'
          % (n_lab, n_multi, len(evs)))
    return evs


def fit_pairwise(evs, l2, seed, iters=3000, lr=0.05):
    """RankNet on standardized features; returns (w, mu, sd)."""
    allF = np.concatenate([e['F'] for e in evs])
    mu, sd = allF.mean(axis=0), allF.std(axis=0)
    sd[sd == 0] = 1.0
    diffs = []
    for e in evs:
        Z = (e['F'] - mu) / sd
        t = e['truth_i']
        for j in range(len(Z)):
            if j != t:
                diffs.append(Z[t] - Z[j])
    D = np.stack(diffs)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, D.shape[1])
    for it in range(iters):
        z = D @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = D.T @ (p - 1.0) / len(D) + l2 * w
        w -= lr * g
    return w, mu, sd


def build_folds(evs, kfold, seed):
    """Stratified (sample, corrective) folds -- identical construction (and
    rng consumption order) to the original inline version, so fold
    membership is reproducible across scripts."""
    rng = np.random.default_rng(seed)
    by_class = {}
    for e in evs:
        by_class.setdefault((e['sample'], e['corrective']), []).append(e)
    folds = [[] for _ in range(kfold)]
    for key in sorted(by_class):
        rows = by_class[key]
        for j, idx in enumerate(rng.permutation(len(rows))):
            folds[j % kfold].append(rows[idx])
    return folds


def feature_indices(spec):
    """--features spec -> column indices into FEATURES.
    'all' = 11 features; '7terms' = the composite terms; else comma names."""
    if spec == 'all':
        return list(range(len(FEATURES)))
    if spec == '7terms':
        return list(range(7))
    idx = []
    for name in spec.split(','):
        if name not in FEATURES:
            raise SystemExit('unknown feature %r (have %s)' % (name, FEATURES))
        idx.append(FEATURES.index(name))
    return idx


def fit_subset(train, idx, l2, seed):
    sub = [dict(F=e['F'][:, idx], truth_i=e['truth_i']) for e in train]
    return fit_pairwise(sub, l2, seed)


def choice_acc(evs, score_fn):
    per = {}
    for e in evs:
        ok = int(int(np.argmax(score_fn(e))) == e['truth_i'])
        per.setdefault(e['sample'], []).append(ok)
        per.setdefault('ALL', []).append(ok)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=ALL_TAGS)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--arm-roots', nargs='+', default=None,
                    help='tag=path pairs (or one bare path): fit on this '
                         'arm\'s scoreboards instead of label[source]')
    ap.add_argument('--features', default='all',
                    help="'all' (11), '7terms', or comma-separated names")
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--l2', type=float, default=0.1)
    ap.add_argument('--kfold', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--tsv', default=None)
    ap.add_argument('--export-weights', default=None,
                    help='TSV of per-feature mu/sd/w_std/w_raw from the '
                         'full fit (w_raw = w_std/sd folds standardization '
                         'into raw feature space; argmax-equivalent)')
    args = ap.parse_args()

    roots = vio.parse_arm_roots(args.arm_roots, args.sbnd_root) \
        if args.arm_roots else None
    fidx = feature_indices(args.features)
    evs = load_eligible(args.sbnd_root, args.tags, args.tol, roots=roots)

    # closure: production weights on raw terms == total-argmax
    for e in evs:
        s = e['F'][:, :7].sum(axis=1)
        assert int(np.argmax(s)) == int(np.argmax(e['totals'])), \
            'closure failed on evt%d' % e['evt']
    print('closure: sum-of-7-terms argmax == recorded total argmax on all %d'
          % len(evs))

    # out-of-fold CV, stratified by (sample, corrective) like dataset.kfold_split
    folds = build_folds(evs, args.kfold, args.seed)

    oof = []
    for i in range(args.kfold):
        val = folds[i]
        train = [e for j, f in enumerate(folds) if j != i for e in f]
        w, mu, sd = fit_subset(train, fidx, args.l2, args.seed + i)
        for e in val:
            Z = (e['F'][:, fidx] - mu) / sd
            oof.append((e, int(np.argmax(Z @ w)) == e['truth_i']))

    w_all, mu_all, sd_all = fit_subset(evs, fidx, args.l2, args.seed)
    names = [FEATURES[i] for i in fidx]
    print('\nlearned weights (standardized features, full fit, %s):'
          % args.features)
    for name, wi in sorted(zip(names, w_all), key=lambda t: -abs(t[1])):
        print('  %-18s %+.3f' % (name, wi))

    per_prod = choice_acc(evs, lambda e: -np.arange(len(e['totals']))
                          if False else e['totals'])
    per_ranked = {}
    for e, ok in oof:
        per_ranked.setdefault(e['sample'], []).append(int(ok))
        per_ranked.setdefault('ALL', []).append(int(ok))
    print('\n== candidate-choice accuracy on eligible events (tol %.1f cm) ==' % args.tol)
    print('%-8s %5s  %14s  %14s  %14s' %
          ('sample', 'n', 'prod-recorded', 'total-argmax', 'rankfit-oof'))
    for s in sorted(per_ranked, key=lambda s: (s == 'ALL', s)):
        n = len(per_ranked[s])
        prodrec = sum(e['prod_correct'] for e, _ in oof if e['sample'] == s or s == 'ALL')
        ta = sum(int(np.argmax(e['totals']) == e['truth_i'])
                 for e, _ in oof if e['sample'] == s or s == 'ALL')
        rf = sum(per_ranked[s])
        print('%-8s %5d  %10d/%-3d  %10d/%-3d  %10d/%-3d'
              % (s, n, prodrec, n, ta, n, rf, n))

    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('evt\tsample\tcorrective\tn_cand\ttruth_i\t'
                     'total_argmax_ok\trankfit_oof_ok\tprod_correct\n')
            for e, ok in oof:
                fh.write('%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n'
                         % (e['evt'], e['sample'], e['corrective'], len(e['totals']),
                            e['truth_i'],
                            int(np.argmax(e['totals']) == e['truth_i']),
                            int(ok), e['prod_correct']))
            fh.write('# weights: %s\n' % ', '.join(
                '%s=%.4f' % (n, w) for n, w in zip(names, w_all)))
        print('wrote %s' % args.tsv)

    if args.export_weights:
        # w_raw = w_std/sd: per-event constant -sum(w*mu/sd) is shared by
        # all rows, so raw-space argmax == standardized argmax.  The
        # constant does NOT cancel for a threshold -- tune any acceptance
        # threshold on raw rank scores (rank_sim.py), never lift it from
        # the standardized fit.
        w_raw_full = np.zeros(len(FEATURES))
        for k, i in enumerate(fidx):
            w_raw_full[i] = w_all[k] / sd_all[k]
        with open(args.export_weights, 'w') as fh:
            fh.write('feature\tmu\tsd\tw_std\tw_raw\n')
            for k, i in enumerate(fidx):
                fh.write('%s\t%.10g\t%.10g\t%.10g\t%.10g\n'
                         % (FEATURES[i], mu_all[k], sd_all[k], w_all[k],
                            w_raw_full[i]))
            fh.write('# w_raw_full11 = [%s]\n'
                     % ', '.join('%.10g' % v for v in w_raw_full))
            fh.write('# features=%s l2=%g seed=%d tol=%g arm_roots=%s\n'
                     % (args.features, args.l2, args.seed, args.tol,
                        args.arm_roots))
        print('wrote %s' % args.export_weights)
    return 0


if __name__ == '__main__':
    sys.exit(main())

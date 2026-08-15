#!/usr/bin/env python3
'''
doc pr/79 step 4 -- deployment simulation of the linear rank-weights
selector on LIVE arm scoreboards (the work-*-ma10-k20 rows: the exact
features the C++ computed, no rebuilt-cloud approximation).

Live-anchored replay: whenever the replayed decision matches the arm's
recorded behavior, the prediction is the arm's recorded final answer
(main_vertex -- includes post-selection refit and the live traditional
route).  Row-level stand-ins are used ONLY where the selector actually
changes something:
  - replay accepts a different candidate than the recorded winner -> the
    candidate row xyz (ignores post-refit drift; standin='row');
  - replay rejects where the arm accepted -> the trad_winner row xyz
    (pr/79 lesson 1: this UNDERSTATES the live traditional route, so
    predicted gains are conservative on those events; standin='trad').
Stand-in counts are reported next to every number they could bias.

Variants (deployment candidates, doc pr/79 step 4):
  baseline : composite argmax, legacy acceptance (sanity: must reproduce
             the recorded arm answers exactly, 0 stand-ins)
  A        : rank argmax, route FROZEN to the recorded accept/reject
             (structural do-no-harm on routing)
  B        : rank argmax + acceptance on the raw rank score (threshold
             tuned per-fold on train, applied to val = honest OOF; the
             full-fit grid curve is the deployed-threshold choice)
The feature set is --features (all=11 / 7terms), so "variant C" = B with
--features 7terms.

OOF discipline: eligible events are scored with their fold's weights
(same folds/seed as rank_fit.py); non-eligible events use the full fit
(they never contributed pairs).  Cross-arm evaluation (--fit-arm-roots
!= --arm-roots, e.g. k20-fitted weights on the k5 arm) uses the full
fit everywhere and is flagged in-sample.

Usage:
  python3 rank_sim.py \
      --arm-roots vtxscan-prod0813=work-nuecc48-ma10-k20 \
                  vtxscan-prod0813-ncpi0=work-ncpi0-ma10-k20 \
                  vtxscan-prod0813-mcp1k=work-mcp1k-ma10-k20 \
      --manifest data/full473/manifest.tsv \
      --tsv-prefix runs/ranksim-k20-<date>
'''
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from taxonomy import ALL_TAGS
from rank_fit import (FEATURES, feat_of, fit_subset, build_folds,
                      feature_indices)


def manifest_sets(path):
    numu, lockbox = set(), set()
    if path:
        with open(path) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                evt = int(r['evt'])
                if r.get('sample', '').startswith('numu'):
                    numu.add(evt)
                if r.get('lockbox') == '1':
                    lockbox.add(evt)
    return numu, lockbox


def load_events(sbnd_root, tags, roots, tol, numu):
    evs = []
    for label in vio.iter_labels(sbnd_root, tags):
        path = vio.calib_path_in_roots(roots, label)
        if not path or not os.path.exists(path):
            raise FileNotFoundError('calib missing for evt%d: %s'
                                    % (label['eventNo'], path))
        calib = vio.load_calib(path)
        sb = calib.get('vertex_scoreboard') or {}
        rows = sb.get('rows') or []
        usable = [r for r in rows
                  if r.get('dl_snapped') and not r.get('skipped_by_swap_guard')]
        mv = calib.get('main_vertex') or {}
        if not mv and sb.get('filled'):
            mv = dict(x=sb['final_x'], y=sb['final_y'], z=sb['final_z'])
        truth = label['truth_xyz']
        rec = np.array([mv.get('x', np.nan), mv.get('y', np.nan),
                        mv.get('z', np.nan)], np.float64)
        ev = dict(
            evt=label['eventNo'],
            sample=vio.sample_of_label(label, numu, 'numu'),
            corrective=int(vio.is_corrective(label, tol=tol)),
            truth=np.asarray(truth, np.float64), rec=rec,
            ok_rec=bool(np.linalg.norm(rec - truth) <= tol),
            route=sb.get('route', ''), usable=usable)
        ev['rec_accept'] = ev['route'] == 'dl-rerank-accept'
        if usable:
            ev['F'] = np.stack([feat_of(r) for r in usable])
            ev['tot'] = np.array([float(r['total']) for r in usable])
            pos = np.array([[r['x'], r['y'], r['z']] for r in usable],
                           np.float64)
            ev['pos'] = pos
            d = np.linalg.norm(pos - np.asarray(truth, np.float64), axis=1)
            ev['comp_i'] = int(ev['tot'].argmax())
            win = [i for i, r in enumerate(usable) if r.get('dl_winner')]
            ev['rec_winner_i'] = win[0] if win else None
            if len(usable) >= 2 and d.min() <= tol:
                ev['truth_i'] = int(d.argmin())     # rank_fit eligibility
        trad = [r for r in rows if r.get('trad_winner')]
        ev['trad_xyz'] = (np.array([trad[0]['x'], trad[0]['y'], trad[0]['z']],
                                   np.float64) if trad else None)
        evs.append(ev)
    return evs


def replay(ev, accept, chosen_i, tol):
    """-> (correct, standin) with live anchoring."""
    if 'F' not in ev:                       # selector cannot act
        return ev['ok_rec'], ''
    if accept:
        if ev['rec_accept'] and chosen_i == ev['rec_winner_i']:
            return ev['ok_rec'], ''
        pred = ev['pos'][chosen_i]
        return bool(np.linalg.norm(pred - ev['truth']) <= tol), 'row'
    if not ev['rec_accept']:
        return ev['ok_rec'], ''
    if ev['trad_xyz'] is None:
        return False, 'trad-none'
    return bool(np.linalg.norm(ev['trad_xyz'] - ev['truth']) <= tol), 'trad'


def evaluate(evs, decide, tol):
    """decide(ev) -> (accept, chosen_i).  Returns summary dict."""
    out = dict(ok=0, per={}, standins={}, fixed=[], regressed=[],
               acc2rej=0, rej2acc=0)
    for ev in evs:
        accept, ci = decide(ev)
        ok, standin = replay(ev, accept, ci, tol)
        out['ok'] += ok
        p = out['per'].setdefault(ev['sample'], [0, 0])
        p[0] += ok
        p[1] += 1
        if standin:
            out['standins'][standin] = out['standins'].get(standin, 0) + 1
        if ok and not ev['ok_rec']:
            out['fixed'].append(ev['evt'])
        if ev['ok_rec'] and not ok:
            out['regressed'].append(ev['evt'])
        if 'F' in ev:
            if ev['rec_accept'] and not accept:
                out['acc2rej'] += 1
            if accept and not ev['rec_accept']:
                out['rej2acc'] += 1
    return out


def report(name, res, n, lockbox_ok=None):
    per = '  '.join('%s %d/%d' % (s, v[0], v[1])
                    for s, v in sorted(res['per'].items()))
    print('%-24s %3d/%d  (fixed %d / regressed %d)  '
          'route acc->rej %d rej->acc %d  standins %s'
          % (name, res['ok'], n, len(res['fixed']), len(res['regressed']),
             res['acc2rej'], res['rej2acc'], res['standins'] or '{}'))
    print('    per-sample: %s' % per)
    if lockbox_ok is not None:
        print('    lockbox(advisory): %d/95' % lockbox_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=ALL_TAGS)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--arm-roots', nargs='+', required=True,
                    help='EVAL arm (tag=path pairs or one bare path)')
    ap.add_argument('--fit-arm-roots', nargs='+', default=None,
                    help='arm to FIT on (default: same as --arm-roots)')
    ap.add_argument('--features', default='all')
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--l2', type=float, default=0.1)
    ap.add_argument('--kfold', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--min-accept', type=float, default=10.0)
    ap.add_argument('--grid-points', type=int, default=41)
    ap.add_argument('--manifest', default='data/full473/manifest.tsv')
    ap.add_argument('--tsv-prefix', default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    manifest = args.manifest if os.path.isabs(args.manifest) \
        else os.path.join(here, args.manifest)
    numu, lockbox = manifest_sets(manifest)
    fidx = feature_indices(args.features)

    roots = vio.parse_arm_roots(args.arm_roots, args.sbnd_root)
    evs = load_events(args.sbnd_root, args.tags, roots, args.tol, numu)
    cross_arm = bool(args.fit_arm_roots)
    if cross_arm:
        fit_roots = vio.parse_arm_roots(args.fit_arm_roots, args.sbnd_root)
        fit_evs = load_events(args.sbnd_root, args.tags, fit_roots,
                              args.tol, numu)
    else:
        fit_evs = evs
    elig_fit = [e for e in fit_evs if 'truth_i' in e]
    n = len(evs)
    n_rec = sum(e['ok_rec'] for e in evs)
    print('events %d; recorded arm correct %d; fit-eligible %d%s'
          % (n, n_rec, len(elig_fit),
             ' (CROSS-ARM fit -- full-fit weights only)' if cross_arm else ''))

    # ---- fits: full + per-fold (raw-space weights, subset length)
    w_all, mu_all, sd_all = fit_subset(elig_fit, fidx, args.l2, args.seed)
    wraw_all = w_all / sd_all
    fold_w = {}                      # evt -> per-fold raw weights (OOF)
    thr_folds = []
    if not cross_arm:
        folds = build_folds(elig_fit, args.kfold, args.seed)
        for i in range(args.kfold):
            train = [e for j, f in enumerate(folds) if j != i for e in f]
            w, mu, sd = fit_subset(train, fidx, args.l2, args.seed + i)
            for e in folds[i]:
                fold_w[e['evt']] = w / sd

    def wraw_of(ev):
        return fold_w.get(ev['evt'], wraw_all)

    def rank_of(ev, w):
        return ev['F'][:, fidx] @ w

    # ---- baseline sanity: must reproduce the recorded arm exactly
    def d_baseline(ev):
        if 'F' not in ev:
            return False, None
        return ev['tot'].max() >= args.min_accept, ev['comp_i']
    res = evaluate(evs, d_baseline, args.tol)
    print('\n== variants (n=%d, tol %.1f cm, features=%s, l2=%g) ==' %
          (n, args.tol, args.features, args.l2))
    report('baseline(legacy@%g)' % args.min_accept, res, n)
    if res['ok'] != n_rec or res['standins']:
        print('    WARNING: baseline does not reproduce the recorded arm '
              '-- inspect before trusting anything below')

    def lockbox_count(decide):
        total = 0
        for e in evs:
            if e['evt'] not in lockbox:
                continue
            a, ci = decide(e)
            total += replay(e, a, ci, args.tol)[0]
        return total

    # ---- variant A: frozen route, rank argmax (OOF weights)
    def d_A(ev):
        if 'F' not in ev:
            return False, None
        ri = int(np.argmax(rank_of(ev, wraw_of(ev))))
        return ev['rec_accept'], ri
    resA = evaluate(evs, d_A, args.tol)
    report('A frozen-route rank%s' % ('(IN-SAMPLE)' if cross_arm else '(OOF)'),
           resA, n, lockbox_count(d_A))

    # ---- variant B: rank argmax + raw rank-score threshold
    # nested CV: threshold tuned on train folds, applied to val fold
    def best_over_grid(sub_evs, w_by_evt, grid):
        best_t, best_ok = None, -1
        for t in grid:
            okc = 0
            for ev in sub_evs:
                if 'F' not in ev:
                    okc += ev['ok_rec']
                    continue
                R = rank_of(ev, w_by_evt(ev))
                okc += replay(ev, bool(R.max() >= t),
                              int(np.argmax(R)), args.tol)[0]
            if okc > best_ok:
                best_ok, best_t = okc, t
        return best_t, best_ok

    Rbest_all = np.array([rank_of(e, wraw_all).max()
                          for e in evs if 'F' in e])
    grid = np.percentile(Rbest_all, np.linspace(0, 100, args.grid_points))
    grid = np.unique(grid)

    if not cross_arm:
        oofB_ok = 0
        for i in range(args.kfold):
            val_evts = {e['evt'] for e in folds[i]}
            # tune the threshold on everything OUTSIDE the val fold that has
            # rows (non-eligible events are in no fold; their accept/reject
            # outcome still constrains the threshold)
            train_evs = [e for e in evs if 'F' in e
                         and e['evt'] not in val_evts]
            # fold-i training weights = the OOF weights stored for any
            # val-fold event (fit_subset over all folds except i)
            tw_vec = fold_w[next(iter(val_evts))]
            t_i, _ = best_over_grid(train_evs, lambda ev: tw_vec, grid)
            thr_folds.append(t_i)
            for e in evs:
                if e['evt'] not in val_evts:
                    continue
                if 'F' not in e:
                    oofB_ok += e['ok_rec']
                    continue
                R = rank_of(e, tw_vec)
                oofB_ok += replay(e, bool(R.max() >= t_i),
                                  int(np.argmax(R)), args.tol)[0]
        # non-eligible events under OOF-B: full-fit weights + median thr
        t_med = float(np.median(thr_folds))
        for e in evs:
            if 'truth_i' in e:
                continue
            if 'F' not in e:
                oofB_ok += e['ok_rec']
                continue
            R = rank_of(e, wraw_all)
            oofB_ok += replay(e, bool(R.max() >= t_med),
                              int(np.argmax(R)), args.tol)[0]
        print('B rank+thr (nested OOF)   %3d/%d   per-fold thr %s'
              % (oofB_ok, n, ['%.3g' % t for t in thr_folds]))

    # full-fit grid curve (deployed-threshold choice; in-sample)
    t_best, ok_best = best_over_grid(evs, lambda ev: wraw_all, grid)
    def d_B(ev):
        if 'F' not in ev:
            return False, None
        R = rank_of(ev, wraw_all)
        return bool(R.max() >= t_best), int(np.argmax(R))
    resB = evaluate(evs, d_B, args.tol)
    report('B rank+thr=%.4g (IN-SAMPLE)' % t_best, resB, n, lockbox_count(d_B))
    print('    grid curve (thr: correct): %s'
          % '  '.join('%.3g:%d' % (t, best_over_grid(evs,
                lambda ev: wraw_all, [t])[1]) for t in grid[::4]))

    # ---- ORACLE-ROUTING diagnostics (not deployable; shapes the next
    # round): union of "arm already correct" with "chooser pick correct",
    # i.e. what a PERFECT accept/override discriminator would deliver for
    # each chooser.  U_truth = the admission ceiling on the same footing.
    def pick_ok(ev, i):
        return ('F' in ev and i is not None and
                bool(np.linalg.norm(ev['pos'][i] - ev['truth']) <= args.tol))
    u_rank = u_comp = u_truth = 0
    for ev in evs:
        ri = (int(np.argmax(rank_of(ev, wraw_of(ev))))
              if 'F' in ev else None)
        u_rank += ev['ok_rec'] or pick_ok(ev, ri)
        u_comp += ev['ok_rec'] or pick_ok(ev, ev.get('comp_i'))
        u_truth += ev['ok_rec'] or ('truth_i' in ev)
    print('\n== oracle-routing ceilings (diagnostic, n=%d) ==' % n)
    print('arm + oracle-routed RANK pick(OOF): %d' % u_rank)
    print('arm + oracle-routed COMPOSITE pick: %d' % u_comp)
    print('arm + truth-candidate-admitted:     %d' % u_truth)

    print('\nfixed A: %s' % sorted(resA['fixed']))
    print('regressed A: %s' % sorted(resA['regressed']))
    print('fixed B: %s' % sorted(resB['fixed']))
    print('regressed B: %s' % sorted(resB['regressed']))
    print('# weights (raw, full fit, order %s): [%s]'
          % (args.features, ', '.join('%.10g' % v for v in wraw_all)))

    if args.tsv_prefix:
        for tag, dec in (('A', d_A), ('B', d_B)):
            path = '%s-%s.tsv' % (args.tsv_prefix, tag)
            with open(path, 'w') as fh:
                fh.write('evt\tsample\tok_rec\tok_new\tstandin\troute_rec\t'
                         'accept_new\n')
                for ev in evs:
                    a, ci = dec(ev)
                    ok, standin = replay(ev, a, ci, args.tol)
                    fh.write('%d\t%s\t%d\t%d\t%s\t%s\t%d\n'
                             % (ev['evt'], ev['sample'], ev['ok_rec'], ok,
                                standin or '-', ev['route'], a))
            print('wrote %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())

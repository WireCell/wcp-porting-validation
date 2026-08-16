#!/usr/bin/env python3
'''
doc pr/77 round 2 (S8c) -- active learning: order the not-yet-scanned PR
events so the owner's remaining scan effort lands where labels are most
informative (corrective labels are the scarce resource).

Per-event signals (all from recorded scoreboards; --tta-tsv adds the
tta_argmax_spread disagreement signal from tta_eval.py's --arm sweep):

  margin      top-1 total minus top-2 total over usable rows (dl_snapped,
              not swap-guard-skipped); single-candidate events get +inf
  dl_best     scoreboard dl_best_score (rerank_total of the winner)
  snap_dis    winner's voxel->candidate snap distance
  route       dl-rerank-reject / dl-veto-protected flagged as high priority
  tta_spread  RMS spread of per-view argmax under x4 reflections (cm)

VALIDATION on the already-labeled events: for each signal, does its
"suspicious" tail enrich corrective labels vs the base rate?  The combined
rank is then applied to the unlabeled events, most-informative-first.

Usage:
  python3 scan_ranker.py --tta-tsv runs/tta-signals-mcp1k.tsv \
      --tsv runs/scan-ranking-20260814.tsv
'''
import argparse
import csv
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio

ARM = 'work-mcp1k-prod0813'
TAG = 'vtxscan-prod0813-mcp1k'


def event_signals(calib):
    sb = calib.get('vertex_scoreboard') or {}
    rows = sb.get('rows') or []
    usable = [r for r in rows
              if r.get('dl_snapped') and not r.get('skipped_by_swap_guard')]
    totals = sorted((float(r['total']) for r in usable), reverse=True)
    margin = (totals[0] - totals[1]) if len(totals) >= 2 else np.inf
    winner = max(usable, key=lambda r: float(r['total'])) if usable else None
    return dict(
        margin=margin,
        n_cand=len(usable),
        dl_best=float(sb.get('dl_best_score') or np.nan),
        snap_dis=float(winner['snap_dis']) if winner else np.nan,
        route=sb.get('route', ''))


def suspicion(sig, tta_spread):
    """Rank score: larger = more informative to scan.  Each term is a soft
    vote from one signal's suspicious direction; scales chosen from the
    labeled-set distributions (validated below, not tuned per event)."""
    s = 0.0
    if sig['route'] in ('dl-rerank-reject', 'dl-veto-protected'):
        s += 2.0
    m = sig['margin']
    if np.isfinite(m):
        s += 2.0 / (1.0 + m / 200.0)      # low margin -> ~2, high -> 0
    if np.isfinite(sig['dl_best']):
        s += 1.0 / (1.0 + max(0.0, sig['dl_best']) / 500.0)
    if np.isfinite(sig['snap_dis']):
        s += min(sig['snap_dis'], 5.0) / 5.0
    if tta_spread is not None:
        s += 2.0 * min(tta_spread, 20.0) / 20.0
    return s


def fit_score(lab, unl):
    """Logistic regression of `corrective` on the signals, fitted on the
    labeled events (5-fold CV AUC reported), applied to the unlabeled --
    the S8 principle: spend labels on selection, not hand-weighting.
    Returns per-event fitted P(corrective) for lab and unl."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    def X_of(recs):
        return np.array([[
            np.log1p(r['margin']) if np.isfinite(r['margin']) else 10.0,
            r['dl_best'] if np.isfinite(r['dl_best']) else 0.0,
            min(r['snap_dis'], 10.0) if np.isfinite(r['snap_dis']) else 10.0,
            float(r['route'] == 'dl-rerank-reject'),
            float(r['route'] == 'dl-veto-protected'),
            r['tta_spread'] if r['tta_spread'] is not None else 0.0,
        ] for r in recs])

    Xl, yl = X_of(lab), np.array([r['corrective'] for r in lab])
    mu, sd = Xl.mean(0), Xl.std(0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(C=1.0, max_iter=2000)
    p_cv = cross_val_predict(clf, (Xl - mu) / sd, yl, cv=5,
                             method='predict_proba')[:, 1]
    auc = roc_auc_score(yl, p_cv)
    clf.fit((Xl - mu) / sd, yl)
    names = ['log1p_margin', 'dl_best', 'snap_dis_cap10',
             'route_reject', 'route_veto', 'tta_spread']
    print('\n  fitted P(corrective): 5-fold CV AUC = %.3f; coefs:' % auc)
    for n, c in sorted(zip(names, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print('    %-16s %+.3f' % (n, c))
    p_unl = clf.predict_proba((X_of(unl) - mu) / sd)[:, 1] if unl else []
    return p_cv, np.asarray(p_unl), auc


def enrichment(vals, corr, frac=0.25, larger_is_suspicious=True):
    """Corrective rate in the top-`frac` suspicious tail vs overall."""
    v = np.asarray(vals, dtype=np.float64)
    c = np.asarray(corr, dtype=np.int32)
    order = np.argsort(-v if larger_is_suspicious else v)
    k = max(1, int(round(frac * len(v))))
    top = c[order[:k]]
    return top.mean(), c.mean(), k


def _emit(args, recs, ordered):
    """TSV + top-15 print for the doc pr/88 apply-only path (no labels)."""
    print('\n== top 15 events to scan next (rule-based `suspicion` score; '
          'no fitted P without labels) ==')
    for r in ordered[:15]:
        print('  evt%-8d score=%.2f margin=%s dl_best=%.0f snap=%.2f route=%s'
              % (r['evt'], r['score'],
                 'inf' if not np.isfinite(r['margin']) else '%.0f' % r['margin'],
                 r['dl_best'], r['snap_dis'], r['route']))
    if args.tsv:
        cols = ['evt', 'p_corrective', 'score', 'margin', 'n_cand', 'dl_best',
                'snap_dis', 'tta_spread', 'route', 'labeled', 'corrective']
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in ordered:
                fh.write('\t'.join(
                    ('' if r[c] is None
                     or (isinstance(r[c], float) and not np.isfinite(r[c]))
                     else ('%.4f' % r[c] if isinstance(r[c], float)
                           else str(r[c])))
                    for c in cols) + '\n')
        print('\nwrote %s (most suspicious first)' % args.tsv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--arm', default=ARM)
    ap.add_argument('--tag', default=TAG)
    ap.add_argument('--tta-tsv', default=None,
                    help='tta_eval.py --arm sweep output (evt, tta_argmax_spread)')
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    tta = {}
    if args.tta_tsv and os.path.exists(args.tta_tsv):
        with open(args.tta_tsv) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                tta[int(r['evt'])] = float(r['tta_argmax_spread'])

    # doc pr/88: applying a VALIDATED ranking to a brand-new arm is the whole
    # point of the tool, and such an arm has no labels at all -- `iter_labels`
    # raises FileNotFoundError on a missing tag dir, which made the apply-only
    # case impossible.  Validate on one arm, apply to another; the validation
    # block below degrades to "not measured here" rather than pretending.
    labeled = {}
    tagdir = os.path.join(args.sbnd_root, 'vertex_labels', args.tag or '')
    if args.tag and os.path.isdir(tagdir):
        for label in vio.iter_labels(args.sbnd_root, [args.tag]):
            labeled[label['eventNo']] = int(vio.is_corrective(label,
                                                              tol=args.tol))
    else:
        print('APPLY-ONLY: no labels for tag %r -- signal validation is NOT '
              'measured on this arm.  The ranking below rests on the '
              'validation run separately against a labelled arm; quote that '
              'number, not this run.' % args.tag)

    recs = []
    for path in sorted(glob.glob(os.path.join(
            args.sbnd_root, args.arm, 'pr_evt*', 'calib-pr-evt*.json'))):
        evt = int(os.path.basename(path).split('evt')[-1].split('.')[0])
        sig = event_signals(vio.load_calib(path))
        sig['evt'] = evt
        sig['tta_spread'] = tta.get(evt)
        sig['score'] = suspicion(sig, sig['tta_spread'])
        sig['labeled'] = int(evt in labeled)
        sig['corrective'] = labeled.get(evt, '')
        recs.append(sig)

    lab = [r for r in recs if r['labeled']]
    unl = [r for r in recs if not r['labeled']]
    print('PR events %d: labeled %d (corrective %d), unlabeled %d'
          % (len(recs), len(lab), sum(r['corrective'] or 0 for r in lab), len(unl)))

    # doc pr/88: with no labelled events every enrichment is 0/0.  Printing a
    # table of nan invites someone to read it as a measurement, so skip it and
    # fall back to the rule-based `suspicion` score, which carries fixed scales
    # taken from the labelled-set distributions and needs no fit.
    if not lab:
        for r in recs:
            r['p_corrective'] = ''
        _emit(args, recs, sorted(recs, key=lambda r: -r['score']))
        return 0

    print('\n== signal validation on labeled events (top-quartile corrective '
          'enrichment vs base rate) ==')
    corr = [r['corrective'] for r in lab]
    for name, vals, larger in (
            ('margin (low)', [-r['margin'] if np.isfinite(r['margin']) else -1e9
                              for r in lab], True),
            ('dl_best (low)', [-r['dl_best'] for r in lab], True),
            ('snap_dis (high)', [r['snap_dis'] for r in lab], True),
            ('tta_spread (high)', [r['tta_spread'] if r['tta_spread'] is not None
                                   else 0.0 for r in lab], True),
            ('combined score', [r['score'] for r in lab], True)):
        top, base, k = enrichment(vals, corr, larger_is_suspicious=larger)
        print('  %-18s top-25%% (n=%2d): %5.1f%% corrective  (base %4.1f%%,'
              ' enrich x%.1f)' % (name, k, 100 * top, 100 * base,
                                  top / base if base else np.nan))
    routes = {}
    for r in lab:
        routes.setdefault(r['route'], []).append(r['corrective'])
    for rt, cs in sorted(routes.items()):
        print('  route %-20s n=%3d corrective %4.1f%%'
              % (rt, len(cs), 100 * np.mean(cs)))

    # fitted ranking (out-of-fold on labeled; applied to unlabeled)
    p_cv, p_unl, auc = fit_score(lab, unl)
    top, base, k = enrichment(p_cv, corr)
    print('  %-18s top-25%% (n=%2d): %5.1f%% corrective  (base %4.1f%%,'
          ' enrich x%.1f)' % ('fitted P (oof)', k, 100 * top, 100 * base,
                              top / base if base else np.nan))
    for r, p in zip(unl, p_unl):
        r['p_corrective'] = float(p)
    for r, p in zip(lab, p_cv):
        r['p_corrective'] = float(p)

    unl.sort(key=lambda r: -r.get('p_corrective', r['score']))
    print('\n== top 15 unlabeled events to scan next ==')
    for r in unl[:15]:
        print('  evt%-8d P=%.2f margin=%s dl_best=%.0f snap=%.2f '
              'spread=%s route=%s'
              % (r['evt'], r.get('p_corrective', -1),
                 'inf' if not np.isfinite(r['margin']) else '%.0f' % r['margin'],
                 r['dl_best'], r['snap_dis'],
                 '-' if r['tta_spread'] is None else '%.1f' % r['tta_spread'],
                 r['route']))

    if args.tsv:
        cols = ['evt', 'p_corrective', 'score', 'margin', 'n_cand', 'dl_best',
                'snap_dis', 'tta_spread', 'route', 'labeled', 'corrective']
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in unl + sorted(lab, key=lambda r: -r.get('p_corrective', 0)):
                fh.write('\t'.join(
                    ('' if r[c] is None or (isinstance(r[c], float) and not np.isfinite(r[c]))
                     else ('%.4f' % r[c] if isinstance(r[c], float) else str(r[c])))
                    for c in cols) + '\n')
        print('\nwrote %s (unlabeled first, most informative first)' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

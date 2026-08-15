#!/usr/bin/env python3
'''
doc pr/78 round 3 -- do the two selection gains stack?

S4a found the fitted candidate ranking beats the composite (+15/156 OOF);
S4b found a stricter acceptance beats production (+15/470 replay).  Both
were measured independently against production.  This simulates the 2x2:

    chooser  = composite argmax   | rankfit OOF argmax
    acceptance min_accept = 4.0 (production) | 10.0 (S4b optimum)

per event: the DL winner is the chooser's argmax over usable rows
(dl_snapped, not swap-guarded); acceptance compares the COMPOSITE total of
the composite winner against min_accept exactly as production routes
(rerank_replay semantics); a rejected event falls back to the trad winner.
rankfit choices are OUT-OF-FOLD (same stratified folds and seed as
rank_fit.py); non-eligible events (single candidate / truth unreachable)
keep the composite choice, so rankfit can only act where it was fitted.

Usage: python3 stack_sim.py --tsv runs/stack-sim-20260815.tsv
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from rank_fit import ALL_TAGS, FEATURES, feat_of, fit_pairwise
from rerank_replay import TERMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=ALL_TAGS)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--l2', type=float, default=0.1)
    ap.add_argument('--kfold', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--min-accepts', nargs='+', type=float, default=[4.0, 10.0])
    ap.add_argument('--scale', type=float, default=1000.0)
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    # ---- load every labeled event with a scoreboard
    evs = []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        calib = vio.load_calib(vio.calib_path_for_label(args.sbnd_root, label))
        sb = calib.get('vertex_scoreboard') or {}
        rows = sb.get('rows') or []
        if not rows:
            continue
        usable = [r for r in rows
                  if r.get('dl_snapped') and not r.get('skipped_by_swap_guard')]
        truth = label['truth_xyz']
        ev = dict(evt=label['eventNo'], truth=truth, rows=rows, usable=usable,
                  sample=vio.sample_of_label(label),
                  corrective=int(vio.is_corrective(label, tol=args.tol)),
                  scale0=float(sb.get('dl_score_scale') or 1000.0))
        if len(usable) >= 2:
            pos = np.array([[r['x'], r['y'], r['z']] for r in usable], np.float32)
            d = np.linalg.norm(pos - truth, axis=1)
            if d.min() <= args.tol:          # rank_fit eligibility
                ev['F'] = np.stack([feat_of(r) for r in usable])
                ev['truth_i'] = int(d.argmin())
        evs.append(ev)
    elig = [e for e in evs if 'F' in e]
    print('events with scoreboard %d; rankfit-eligible %d' % (len(evs), len(elig)))

    # ---- rankfit OOF choices (same folds as rank_fit.py)
    rng = np.random.default_rng(args.seed)
    by_class = {}
    for e in elig:
        by_class.setdefault((e['sample'], e['corrective']), []).append(e)
    folds = [[] for _ in range(args.kfold)]
    for key in sorted(by_class):
        rows = by_class[key]
        for j, idx in enumerate(rng.permutation(len(rows))):
            folds[j % args.kfold].append(rows[idx])
    for i in range(args.kfold):
        val = folds[i]
        train = [e for j, f in enumerate(folds) if j != i for e in f]
        w, mu, sd = fit_pairwise(train, args.l2, args.seed + i)
        for e in val:
            e['rankfit_i'] = int(np.argmax(((e['F'] - mu) / sd) @ w))

    # ---- 2x2 replay
    def run(min_accept, chooser):
        n_ok = 0
        per = {}
        for e in evs:
            best_tot, comp_i = -np.inf, None
            for i, r in enumerate(e['usable']):
                s_dl = float(r['dl_score']) * args.scale
                tot = s_dl + sum(float(r[t]) for t in TERMS[1:])
                if tot > best_tot:
                    best_tot, comp_i = tot, i
            accepted = comp_i is not None and best_tot >= min_accept
            if accepted:
                ci = comp_i
                if chooser == 'rankfit' and 'rankfit_i' in e:
                    ci = e['rankfit_i']
                chosen = e['usable'][ci]
            else:
                trad = [r for r in e['rows'] if r.get('trad_winner')]
                chosen = trad[0] if trad else None
            ok = False
            if chosen is not None:
                d = np.linalg.norm(np.array(
                    [chosen['x'], chosen['y'], chosen['z']], np.float32)
                    - e['truth'])
                ok = bool(d < args.tol)
            n_ok += ok
            per.setdefault(e['sample'], [0, 0])
            per[e['sample']][0] += ok
            per[e['sample']][1] += 1
        return n_ok, per

    lines = []
    print('\n== stacked selection replay (%d events, tol %.1f cm, scale %.0f) =='
          % (len(evs), args.tol, args.scale))
    print('%-12s %-10s %10s' % ('min_accept', 'chooser', 'correct'))
    for ma in args.min_accepts:
        for chooser in ('composite', 'rankfit'):
            n_ok, per = run(ma, chooser)
            tag = ' <- production' if (ma == 4.0 and chooser == 'composite') else ''
            print('%-12.1f %-10s %6d/%d%s' % (ma, chooser, n_ok, len(evs), tag))
            lines.append((ma, chooser, n_ok, per))

    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('min_accept\tchooser\tcorrect\ttotal\tper_sample\n')
            for ma, chooser, n_ok, per in lines:
                fh.write('%.1f\t%s\t%d\t%d\t%s\n'
                         % (ma, chooser, n_ok, len(evs),
                            ';'.join('%s=%d/%d' % (s, v[0], v[1])
                                     for s, v in sorted(per.items()))))
        print('wrote', args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
'''
doc pr/77 -- offline replay of the DL rerank acceptance stage
(NeutrinoVertexFinder.cxx:3986-4211, doc pr/52 S3) from the scoreboard
rows[] recorded in calib-pr-evt<ID>.json, joined with hand-scan labels.

Two modes:

  --closure      recompute total = s_dl+s_snap+s_fwd_z+s_clen+s_isol+s_main
                 +s_fv for every DL-snapped row and compare against the
                 recorded total (proves the replay is faithful before any
                 grid is trusted).

  --grid         scan (dl_vtx_min_accept_score, dl_vtx_score_scale) --
                 doc pr/52 S5.2a/b.  For each point: recompute s_dl =
                 dl_score*scale, re-argmax, accept iff total >= min_accept;
                 the event's chosen vertex = DL winner if accepted else the
                 traditional winner (trad_winner row).  Correct = chosen
                 vertex within --tol of the label truth.  Objective:
                 #correct (equivalently corrected-minus-broken vs the
                 production operating point 4.0/1000).

Events whose truth is not any candidate (manual pick / pr/51 class) can
never be "correct" here; they are excluded from the objective and counted
separately.  W_* retuning: --w-main/--w-clen/... multiply the recorded
per-term values (the terms' internal shapes stay fixed; exposing the
constexpr internals needs the pr/52 S5.2c knob round).

Usage:
  python3 rerank_replay.py --tags vtxscan-prod0813 vtxscan-prod0813-ncpi0 --closure
  python3 rerank_replay.py --tags ... --grid --tsv runs/rerank-grid.tsv
'''
import argparse
import itertools
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio

TERMS = ['s_dl', 's_snap', 's_fwd_z', 's_clen', 's_isol', 's_main', 's_fv']


def load_events(sbnd_root, tags):
    evs = []
    for label in vio.iter_labels(sbnd_root, tags):
        calib = vio.load_calib(vio.calib_path_for_label(sbnd_root, label))
        sb = calib.get('vertex_scoreboard') or {}
        rows = sb.get('rows') or []
        if not rows:
            continue
        evs.append(dict(label=label, sb=sb, rows=rows))
    return evs


def closure(evs):
    worst = 0.0
    n = 0
    for ev in evs:
        for r in ev['rows']:
            if not r.get('dl_snapped'):
                continue
            tot = sum(float(r[t]) for t in TERMS)
            worst = max(worst, abs(tot - float(r['total'])))
            n += 1
    print('closure over %d DL-snapped rows: max |sum(terms) - total| = %.3g' % (n, worst))
    return worst < 1e-6


def replay_event(ev, min_accept, scale, wmul, tol):
    """Return (correct, route) for one event at this operating point."""
    label = ev['label']
    truth = label['truth_xyz']
    rows = ev['rows']
    sb = ev['sb']
    scale0 = float(sb.get('dl_score_scale') or 1000.0)

    best_tot, best_row = -np.inf, None
    for r in rows:
        if not r.get('dl_snapped') or r.get('skipped_by_swap_guard'):
            continue
        s_dl = float(r['dl_score']) * scale
        tot = s_dl * wmul['s_dl'] + sum(float(r[t]) * wmul[t] for t in TERMS[1:])
        if tot > best_tot:
            best_tot, best_row = tot, r

    if best_row is not None and best_tot >= min_accept:
        chosen, route = best_row, 'dl'
    else:
        trad = [r for r in rows if r.get('trad_winner')]
        chosen, route = (trad[0] if trad else None), 'trad'

    if chosen is None:
        return False, route
    d = np.linalg.norm(np.array([chosen['x'], chosen['y'], chosen['z']],
                                dtype=np.float32) - truth)
    return bool(d < tol), route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+',
                    default=['vtxscan-prod0813', 'vtxscan-prod0813-ncpi0'])
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--closure', action='store_true')
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--min-accepts', nargs='+', type=float,
                    default=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    ap.add_argument('--scales', nargs='+', type=float,
                    default=[250.0, 500.0, 1000.0, 2000.0, 4000.0])
    for t in TERMS:
        ap.add_argument('--w-%s' % t[2:].replace('_', '-'), type=float, default=1.0,
                        dest='w_%s' % t[2:])
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    wmul = {t: getattr(args, 'w_%s' % t[2:]) for t in TERMS}
    evs = load_events(args.sbnd_root, args.tags)
    reachable = [ev for ev in evs if not ev['label']['not_a_candidate']]
    print('loaded %d events with scoreboards (%d truth-reachable, %d not-a-candidate)'
          % (len(evs), len(reachable), len(evs) - len(reachable)))

    ok = True
    if args.closure or not args.grid:
        ok = closure(evs)
    if not args.grid:
        return 0 if ok else 1

    results = []
    for ma, sc in itertools.product(args.min_accepts, args.scales):
        n_ok = n_dl = 0
        for ev in reachable:
            correct, route = replay_event(ev, ma, sc, wmul, args.tol)
            n_ok += int(correct)
            n_dl += int(route == 'dl')
        results.append(dict(min_accept=ma, scale=sc, correct=n_ok,
                            total=len(reachable), dl_route=n_dl))

    prod = [r for r in results if r['min_accept'] == 4.0 and r['scale'] == 1000.0]
    base = prod[0]['correct'] if prod else None
    results.sort(key=lambda r: -r['correct'])
    print('\n min_accept  scale   correct/total  dl-route  vs-prod')
    for r in results[:15]:
        delta = '' if base is None else '%+d' % (r['correct'] - base)
        star = '  <- production' if (r['min_accept'] == 4.0 and r['scale'] == 1000.0) else ''
        print(' %9.1f  %6.0f   %3d/%3d        %3d      %4s%s'
              % (r['min_accept'], r['scale'], r['correct'], r['total'],
                 r['dl_route'], delta, star))

    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('min_accept\tscale\tcorrect\ttotal\tdl_route\n')
            for r in sorted(results, key=lambda r: (r['min_accept'], r['scale'])):
                fh.write('%.2f\t%.0f\t%d\t%d\t%d\n'
                         % (r['min_accept'], r['scale'], r['correct'],
                            r['total'], r['dl_route']))
        print('wrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

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


def replay_event(ev, min_accept, scale, wmul, tol,
                 topo=None, w_topo=0.0, topo_center=0.0, swap_guard=False):
    """Return (correct, route) for one event at this operating point."""
    label = ev['label']
    truth = label['truth_xyz']
    rows = ev['rows']
    sb = ev['sb']
    scale0 = float(sb.get('dl_score_scale') or 1000.0)
    evt = label['eventNo']

    # doc pr/89 Arm A: dl_vtx_swap_guard replay.  With the guard on, a
    # candidate whose cluster is not the first-round main cluster never
    # enters the acceptance (NeutrinoVertexFinder.cxx:4618-4626) -- in the
    # recorded rows that is exactly s_main == 0 when a main cluster exists.
    # Approximation stated, not hidden: the C++ recomputes min_z_set over
    # the SURVIVING snapped set, so s_fwd_z (max |0.25|) can differ when
    # the excluded candidate held the min-z; the recorded s_fwd_z is used.
    has_main = any(float(r.get('s_main') or 0.0) > 0.0 for r in rows
                   if r.get('dl_snapped'))

    best_tot, best_row = -np.inf, None
    for r in rows:
        if not r.get('dl_snapped') or r.get('skipped_by_swap_guard'):
            continue
        if swap_guard and has_main and float(r.get('s_main') or 0.0) <= 0.0:
            continue
        s_dl = float(r['dl_score']) * scale
        tot = s_dl * wmul['s_dl'] + sum(float(r[t]) * wmul[t] for t in TERMS[1:])
        if w_topo != 0.0 and topo is not None:
            # doc pr/89 Arm C: eighth term, rule-1 outgoing-prong fraction
            # (pr/80 sec 4).  A vertex with NO decisive prong contributes
            # exactly nothing (pr/88 P6: below ~5 cm both end-windows cover
            # the same points -- a missing vote must never read as frac=0).
            t = topo.get((evt, r['vertex_id']))
            if t is not None and t[1] >= 1:
                tot += w_topo * (t[0] - topo_center)
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
    ap.add_argument('--topo-file', default=None,
                    help='doc pr/89 Arm C: TSV (evt, vertex_id, frac, '
                         'n_votes) from vtx_rules/rule1_feature.py; joined '
                         'per (evt, vertex_id) as an eighth additive term '
                         's_topo = w_topo * (frac - topo_center), voting '
                         'vertices only')
    ap.add_argument('--w-topo', type=float, default=0.0,
                    help='weight of s_topo (0 = term absent, output '
                         'bit-identical to the 7-term replay)')
    ap.add_argument('--topo-center', type=float, default=0.0)
    ap.add_argument('--topo-sweep', nargs='+', type=float, default=None,
                    help='sweep these w_topo values at each grid point '
                         '(implies --grid uses them as a third axis)')
    ap.add_argument('--ipw-file', default=None,
                    help='doc pr/89 sec 1.2: TSV (evt, weight) of '
                         'inverse-propensity weights; adds a weighted-'
                         'correct column (unlisted events weight 1.0)')
    ap.add_argument('--prod-point', nargs=2, type=float, default=[4.0, 1000.0],
                    metavar=('MIN_ACCEPT', 'SCALE'),
                    help='reference operating point for the vs-prod column '
                         '(production has been 10.0 1000 since pr/79)')
    ap.add_argument('--exclude-events', default=None,
                    help='file of eventNo (one per line, # comments) to '
                         'exclude -- doc pr/89 sec 7: any grid/sweep that '
                         'SELECTS an operating point must not see the '
                         'sealed held-out events (heldout-pr89.txt)')
    ap.add_argument('--swap-guard', action='store_true',
                    help='doc pr/89 Arm A: replay with dl_vtx_swap_guard ON '
                         '(cross-cluster candidates barred from acceptance)')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    wmul = {t: getattr(args, 'w_%s' % t[2:]) for t in TERMS}
    topo = None
    if args.topo_file:
        import csv
        topo = {}
        with open(args.topo_file) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                topo[(int(r['evt']), int(r['vertex_id']))] = (
                    float(r['frac']), int(r['n_votes']))
        print('topo: %d (evt, vertex) rule-1 rows from %s'
              % (len(topo), args.topo_file))
    ipw = {}
    if args.ipw_file:
        import csv
        with open(args.ipw_file) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                ipw[int(r['evt'])] = float(r['weight'])
        print('ipw: %d event weights from %s' % (len(ipw), args.ipw_file))
    evs = load_events(args.sbnd_root, args.tags)
    if args.exclude_events:
        with open(args.exclude_events) as fh:
            excl = {int(l.split()[0]) for l in fh
                    if l.strip() and not l.startswith('#')}
        n0 = len(evs)
        evs = [ev for ev in evs if ev['label']['eventNo'] not in excl]
        print('excluded %d of %d events (%s)'
              % (n0 - len(evs), n0, args.exclude_events))
    reachable = [ev for ev in evs if not ev['label']['not_a_candidate']]
    print('loaded %d events with scoreboards (%d truth-reachable, %d not-a-candidate)'
          % (len(evs), len(reachable), len(evs) - len(reachable)))

    ok = True
    if args.closure or not args.grid:
        ok = closure(evs)
    if not args.grid:
        return 0 if ok else 1

    wsum = sum(ipw.get(ev['label']['eventNo'], 1.0) for ev in reachable)
    wts = args.topo_sweep or [args.w_topo]
    results = []
    for ma, sc, wt in itertools.product(args.min_accepts, args.scales, wts):
        n_ok = n_dl = 0
        w_ok = 0.0
        for ev in reachable:
            correct, route = replay_event(ev, ma, sc, wmul, args.tol,
                                          topo=topo, w_topo=wt,
                                          topo_center=args.topo_center,
                                          swap_guard=args.swap_guard)
            n_ok += int(correct)
            n_dl += int(route == 'dl')
            if correct:
                w_ok += ipw.get(ev['label']['eventNo'], 1.0)
        results.append(dict(min_accept=ma, scale=sc, w_topo=wt, correct=n_ok,
                            total=len(reachable), dl_route=n_dl,
                            wcorrect=w_ok, wtotal=wsum))

    pp = tuple(args.prod_point)
    prod = [r for r in results if (r['min_accept'], r['scale']) == pp
            and r['w_topo'] == 0.0]
    if not prod:   # w_topo=0 reference absent from the sweep: fall back
        prod = [r for r in results if (r['min_accept'], r['scale']) == pp]
    base = prod[0]['correct'] if prod else None
    results.sort(key=lambda r: -r['correct'])
    print('\n min_accept  scale  w_topo   correct/total  wcorrect%%  dl-route  vs-prod')
    for r in results[:20]:
        delta = '' if base is None else '%+d' % (r['correct'] - base)
        star = ('  <- reference' if (r['min_accept'], r['scale']) == pp
                and r['w_topo'] == 0.0 else '')
        print(' %9.1f  %6.0f  %6.2f   %3d/%3d        %6.2f     %3d      %4s%s'
              % (r['min_accept'], r['scale'], r['w_topo'], r['correct'],
                 r['total'], 100.0 * r['wcorrect'] / r['wtotal'],
                 r['dl_route'], delta, star))

    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('min_accept\tscale\tw_topo\tcorrect\ttotal\twcorrect\twtotal\tdl_route\n')
            for r in sorted(results,
                            key=lambda r: (r['min_accept'], r['scale'], r['w_topo'])):
                fh.write('%.2f\t%.0f\t%.3f\t%d\t%d\t%.3f\t%.3f\t%d\n'
                         % (r['min_accept'], r['scale'], r['w_topo'],
                            r['correct'], r['total'], r['wcorrect'],
                            r['wtotal'], r['dl_route']))
        print('wrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

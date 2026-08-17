#!/usr/bin/env python3
'''doc pr/89 round 5 (sec 13.4) -- joint retune of the DL rerank composite
(6 geometric term weights, w_topo, topo_center, min_accept, scale) on the
labelled 1014, scored with an OUTCOME-CALIBRATED ruler built from the three
live arms (pr89base / pr89topo / pr89swap) instead of the row-coordinate
proxy that sec 11.13 showed to be biased.

Ruler: for a parameter set theta, each event's (route, winner) is computed
from the recorded candidate rows (topo-arm dumps carry all 8 terms + raw
dl_score + topo frac/votes).  The event's outcome is then the LIVE final
vertex of whichever arm actually made that same decision:
  ('accept', winner_vid) -> final d of any arm whose rerank winner was that
      vid (downstream veto/snap/improve are deterministic per winner, so the
      arm's final position IS the outcome of choosing that winner);
  ('reject',)            -> final d of the base or topo arm when IT rejected
      (the traditional outcome; the swap arm's reject finals are excluded --
      dl_vtx_swap_guard also touches the reject-route global compare).
Decisions no arm ever took are UNCOVERED and count as INCORRECT for the
candidate (pessimistic floor: a reported gain can only be under-stated).

Closure: theta = production (all weights 1, w_topo 0, 10.0/1000) must
reproduce the live base arm's (route, winner) on every replayable event
before any tuned number is quoted.

Repro:
  python3 rerank_tune.py --search          # coordinate ascent + report
  python3 rerank_tune.py --eval w_snap=1 ... min_accept=10 scale=1000
'''
import os, sys, json, csv, argparse, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TAGS = ['vtxscan-mcp2k', 'vtxscan-mcp2k-auto', 'vtxscan-harv3-nuecc48',
        'vtxscan-harv3-ncpi0', 'vtxscan-harv3-mcp1k', 'vtxscan-harv3-delta']
ACCEPT_FAMILY = ('dl-rerank-accept', 'dl-veto-protected')
GEO = ['s_snap', 's_fwd_z', 's_clen', 's_isol', 's_main', 's_fv']
PROD = dict(w_snap=1.0, w_fwd_z=1.0, w_clen=1.0, w_isol=1.0, w_main=1.0,
            w_fv=1.0, w_topo=0.0, center=0.0, min_accept=10.0, scale=1000.0)
# Owner constraint (2026-08-17): the production point encodes physics --
# refine near it.  No term may be zeroed or sign-flipped; geometric weights
# stay within [0.5, 2]; threshold/scale move one step at most.
GRIDS = dict(
    w_snap=[0.5, 0.75, 1.0, 1.5, 2.0],
    w_fwd_z=[0.5, 0.75, 1.0, 1.5, 2.0],
    w_clen=[0.5, 0.75, 1.0, 1.5, 2.0],
    w_isol=[0.5, 0.75, 1.0, 1.5, 2.0],
    w_main=[0.5, 0.75, 1.0, 1.5, 2.0],
    w_fv=[0.5, 0.75, 1.0, 1.5, 2.0],
    w_topo=[0.0, 0.5, 1.0, 2.0, 3.0],
    center=[0.0, 0.25, 0.5],
    min_accept=[8.0, 10.0, 12.0],
    scale=[750.0, 1000.0, 1500.0],
)


def sample_of(lab):
    t = lab['scan_tag']
    if t in ('vtxscan-mcp2k', 'vtxscan-mcp2k-auto'):
        return 'mcp2k'
    for s in ('nuecc48', 'ncpi0', 'mcp1k'):
        if s in t:
            return s
    a = lab.get('arm') or ''
    return 'nuecc48' if 'nuecc48' in a else 'mcp1k'


def load_all():
    ipw = {}
    with open(os.path.join(HERE, 'runs/ipw-mcp2k-closed-20260816.tsv')) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            ipw[int(r['evt'])] = float(r['weight'])
    events, seen = [], set()
    for lab in vio.iter_labels(ROOT, TAGS):
        e = lab['eventNo']
        if e in seen:
            continue
        seen.add(e)
        s = sample_of(lab)
        truth = np.array(lab['truth_xyz'])
        arms = {}
        for arm in ('base', 'topo', 'swap'):
            p = '%s/work-%s-pr89%s/pr_evt%d/calib-pr-evt%d.json' % (ROOT, s, arm, e, e)
            if os.path.exists(p):
                arms[arm] = json.load(open(p))['vertex_scoreboard']
        if 'base' not in arms or 'topo' not in arms:
            continue
        # candidate rows from the TOPO arm (carries all 8 terms + frac/votes)
        rows = []
        for r in arms['topo'].get('rows', []):
            if not r.get('dl_snapped'):
                continue
            rows.append(dict(
                vid=r['vertex_id'], dl=float(r['dl_score']),
                geo={t: float(r.get(t) or 0.0) for t in GEO},
                frac=float(r.get('topo_frac', -1.0)),
                votes=int(r.get('topo_votes') or 0)))
        # outcome map from live finals
        out = {}
        for arm, sb in arms.items():
            d = float(np.linalg.norm(
                np.array([sb['final_x'], sb['final_y'], sb['final_z']]) - truth))
            win = next((r['vertex_id'] for r in sb.get('rows', [])
                        if r.get('dl_winner')), None)
            if sb['route'] in ACCEPT_FAMILY and win is not None:
                out[('accept', win)] = d
            elif arm in ('base', 'topo'):
                out[('reject',)] = d
        db = float(np.linalg.norm(np.array(
            [arms['base']['final_x'], arms['base']['final_y'],
             arms['base']['final_z']]) - truth))
        base_win = next((r['vertex_id'] for r in arms['base'].get('rows', [])
                         if r.get('dl_winner')), None)
        events.append(dict(
            evt=e, sample=s, rows=rows, out=out, base_d=db,
            base_route=arms['base']['route'], base_win=base_win,
            w=ipw.get(e, 1.0)))
    return events


def decide(ev, th):
    '''(route, winner_vid) for this event under theta; None if no candidates.'''
    best, bvid = -1e18, None
    for r in ev['rows']:
        tot = r['dl'] * th['scale'] + sum(
            r['geo'][t] * th['w_' + t[2:]] for t in GEO)
        if th['w_topo'] != 0.0 and r['votes'] >= 1:
            tot += th['w_topo'] * (r['frac'] - th['center'])
        if tot > best:
            best, bvid = tot, r['vid']
    if bvid is None:
        return None
    return ('accept', bvid) if best >= th['min_accept'] else ('reject',)


def evaluate(events, th, tol=1.0):
    n = correct = uncov = 0
    wsum = wcor = 0.0
    per = collections.Counter()
    for ev in events:
        n += 1
        wsum += ev['w']
        dec = decide(ev, th)
        if dec is None:
            d = ev['base_d']         # no snapped candidates: nothing changes
        elif dec in ev['out']:
            d = ev['out'][dec]
        else:
            uncov += 1
            d = None                 # pessimistic: uncovered counts wrong
        c = int(d is not None and d < tol)
        correct += c
        wcor += ev['w'] * c
        per[ev['sample'] + ('+' if c else '-')] += 1
    return dict(n=n, correct=correct, uncovered=uncov,
                ipw=100.0 * wcor / wsum, per=per)


def closure(events):
    bad = 0
    for ev in events:
        dec = decide(ev, PROD)
        if dec is None:
            continue
        base_accept = ev['base_route'] in ACCEPT_FAMILY
        if dec[0] == 'accept':
            if not (base_accept and dec[1] == ev['base_win']):
                bad += 1
        else:
            if base_accept:
                bad += 1
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--search', action='store_true')
    ap.add_argument('--eval', nargs='*', default=None,
                    metavar='K=V', help='evaluate one theta')
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    events = load_all()
    print('loaded %d events (rows from pr89topo, outcomes from base/topo/swap)'
          % len(events))
    ncand = sum(1 for ev in events if ev['rows'])
    print('events with snapped candidates: %d' % ncand)
    bad = closure(events)
    print('CLOSURE at production theta: %d mismatches vs live base arm'
          % bad + ('  [OK]' if bad == 0 else '  [FAIL -- stop]'))
    if bad:
        sys.exit(1)
    base = evaluate(events, PROD, args.tol)
    print('production: correct %d/%d  IPW %.2f%%  (uncovered %d)'
          % (base['correct'], base['n'], base['ipw'], base['uncovered']))

    if args.eval is not None:
        th = dict(PROD)
        for kv in args.eval:
            k, v = kv.split('=')
            th[k] = float(v)
        r = evaluate(events, th, args.tol)
        print('theta %s' % th)
        print('correct %d/%d (%+d)  IPW %.2f%%  uncovered %d'
              % (r['correct'], r['n'], r['correct'] - base['correct'],
                 r['ipw'], r['uncovered']))
        return

    if not args.search:
        return

    # coordinate ascent from production, objective = covered-correct
    th = dict(PROD)
    cur = evaluate(events, th, args.tol)
    print('\n== coordinate ascent (objective: exact-covered correct@%g) ==' % args.tol)
    hist = []
    for it in range(6):
        improved = False
        for k in ('w_topo', 'center', 'min_accept', 'scale',
                  'w_snap', 'w_fwd_z', 'w_clen', 'w_isol', 'w_main', 'w_fv'):
            best_v, best_r = th[k], cur
            for v in GRIDS[k]:
                if v == th[k]:
                    continue
                t2 = dict(th)
                t2[k] = v
                r = evaluate(events, t2, args.tol)
                if (r['correct'], -r['uncovered']) > (best_r['correct'],
                                                      -best_r['uncovered']):
                    best_v, best_r = v, r
            if best_v != th[k]:
                print('  iter %d: %s %g -> %g  correct %d (%+d)  uncov %d'
                      % (it, k, th[k], best_v, best_r['correct'],
                         best_r['correct'] - base['correct'],
                         best_r['uncovered']))
                th[k] = best_v
                cur = best_r
                improved = True
                hist.append((dict(th), best_r['correct'], best_r['uncovered']))
        if not improved:
            break
    print('\n== best theta ==')
    for k in sorted(th):
        print('  %-11s %g%s' % (k, th[k],
              '' if th[k] == PROD[k] else '   (prod %g)' % PROD[k]))
    print('correct %d/%d (%+d vs prod)  IPW %.2f%% (prod %.2f%%)  uncovered %d'
          % (cur['correct'], cur['n'], cur['correct'] - base['correct'],
             cur['ipw'], base['ipw'], cur['uncovered']))
    per = collections.Counter()
    for ev in events:
        dec = decide(ev, th)
        d = ev['base_d'] if dec is None else ev['out'].get(dec)
        c = int(d is not None and d < args.tol)
        cb = int(ev['base_d'] < args.tol)
        if c != cb:
            per[(ev['sample'], 'fix' if c else 'break')] += 1
    print('per-sample fix/break:', dict(per))
    if args.tsv:
        with open(args.tsv, 'w') as fh:
            wr = csv.writer(fh, delimiter='\t')
            wr.writerow(list(sorted(PROD)) + ['correct', 'uncovered'])
            for t, c, u in hist:
                wr.writerow([t[k] for k in sorted(PROD)] + [c, u])
        print('wrote %s' % args.tsv)


if __name__ == '__main__':
    main()

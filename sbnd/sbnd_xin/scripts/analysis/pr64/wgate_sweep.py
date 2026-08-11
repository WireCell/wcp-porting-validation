#!/usr/bin/env python3
"""doc pr/64 round 3: is a gated W-plane exception for S6/S7 possible without
degrading the many correct W-gap separations?

Three real long tracks (evt314507, evt276836, evt174224 -- see the pr/64 doc
body) are broken because W has no excuse channel at all: `excused[3] =
{excuse_u, excuse_v, false}` (connect_graph_relaxed_strict.cxx:549). This
script is the prototype/validation harness for a candidate W exception --
it does NOT change any C++ or config. Three modes:

  owner-labels   -- score R2w/R2d (oc56_autoscan.classify()'s existing,
                    owner-fitted rule) against the raw 899-label truth table
                    for W-gapped pairs. Pure feature/label tabulation, no
                    graph replay -- see the staleness note below for why
                    that matters.
  blast-radius   -- fresh full 1000-event population (today's production,
                    work-pr64-scan1k): how many killed S6/S7 edges have W as
                    the SOLE voting plane (the only ones any W exception
                    could touch -- a W excuse is monotone toward more
                    connection, so this is the complete exposed population),
                    and how classify() splits the revivable pairs.
  sweep          -- local-vs-global collinearity axis comparison. The
                    production S6 rescue (oc56_fit.rescue(), doc pr/57 round
                    6, shipped) already has a narrow W branch using the
                    LOCAL break-point PCA angle (`ab_local`); this sweep
                    shows why widening ITS thresholds does not recover
                    evt174224 and does cost real good pairs, whereas R2w's
                    GLOBAL whole-component angle recovers it cleanly.

*** STALENESS CAVEAT, read before trusting `sweep`'s numbers ***
`oc56_fit.DEFAULT_ARMS` are all round-4 dumps that predate doc pr/62's S7
production flip (2026-08-11). `sweep` uses `replay_separated()`, which walks
each label arm's OWN cached graph edges -- for evt174224 pair 1-2 this graph
still contains a stray 82.7cm MST edge that no longer exists in today's bare
production (almost certainly killed once S7 started running), so the
replay-based verdict for that specific pair does not reflect current
production topology. `owner-labels` and `blast-radius` do NOT use replay
(pure feature/label tabulation, or a fresh dump respectively) and are not
subject to this. Treat `sweep` output as indicative, not decisive, until
the label arms are regenerated post-pr/62 -- out of scope for this round.

Usage:
    wgate_sweep.py owner-labels
    wgate_sweep.py blast-radius [--arm work-pr64-scan1k]
    wgate_sweep.py sweep
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pr57'))
import oc56_autoscan as A       # noqa: E402
import oc56_conn                # noqa: E402
import oc56_fit as F            # noqa: E402

SBND = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TRUTH_TSV = os.path.join(SBND, 'docs', 'pr', 'pr57r6-truth.tsv')

# R2w / R2d thresholds, unchanged from oc56_autoscan.DEFAULT_PARAMS
L, N, TW, AW = 6.0, 50, 2.0, 25.0
WD, ND = 3, 20


def r2w_fires(row):
    Lmin, npmin, Tmax, ang = row['Lmin'], row['npmin'], row['Tmax'], row['angle']
    return Lmin > L and npmin >= N and Tmax < TW and ang is not None and ang < AW


def r2d_fires(row):
    return row['wdeadX'] >= WD and row['npmin'] >= ND and row['dis'] < 3.0


def cmd_owner_labels(args):
    """Score R2w/R2d against the raw 899-label owner truth for every W-gapped
    pair. No graph replay -- immune to the staleness caveat."""
    rows = list(csv.DictReader(open(TRUTH_TSV), delimiter='\t'))
    for r in rows:
        for k in ('Lmin', 'Lmax', 'Tmax', 'npmin', 'angle', 'dis', 'wdeadX', 'gw'):
            v = r[k]
            r[k] = float(v) if v not in ('', 'None') else None
        r['npmin'] = int(r['npmin']) if r['npmin'] is not None else 0
        r['wdeadX'] = int(r['wdeadX']) if r['wdeadX'] is not None else 0
        r['gw'] = bool(int(r['gw'])) if r['gw'] is not None else False

    gw_rows = [r for r in rows if r['gw']]
    print('W-gapped pairs in the 899-label truth table: %d' % len(gw_rows))
    cm = collections.defaultdict(lambda: [0, 0, 0])  # verdict -> [n, r2w, r2d]
    either = collections.Counter()
    fp = []
    for r in gw_rows:
        v = r['verdict']
        cm[v][0] += 1
        f2w = r2w_fires(r)
        f2d = r2d_fires(r)
        if f2w:
            cm[v][1] += 1
        if f2d:
            cm[v][2] += 1
        if f2w or f2d:
            either[v] += 1
            if v == 'good':
                fp.append(r)
    print('%-6s %5s %8s %8s %8s' % ('verd', 'n', 'R2w', 'R2d', 'either'))
    for v in ('bad', 'good', 'OK'):
        n, r2w, r2d = cm[v]
        print('%-6s %5d %8d %8d %8d' % (v, n, r2w, r2d, either[v]))
    print()
    print('false positives (good pairs a W exception would break):')
    for r in fp:
        print('  evt%-8s %s-%s Lmin=%.1f Tmax=%.2f npmin=%d angle=%.1f dis=%.2f'
              % (r['evt'], r['j'], r['k'], r['Lmin'], r['Tmax'], r['npmin'],
                 r['angle'], r['dis']))
    print()
    print('bad pairs recovered (R2w or R2d fires):')
    for r in gw_rows:
        if r['verdict'] == 'bad' and (r2w_fires(r) or r2d_fires(r)):
            print('  evt%-8s %s-%s Lmin=%.1f Tmax=%.2f npmin=%d angle=%.1f '
                  'R2w=%d R2d=%d' % (r['evt'], r['j'], r['k'], r['Lmin'],
                                      r['Tmax'], r['npmin'], r['angle'],
                                      int(r2w_fires(r)), int(r2d_fires(r))))
    print()
    print('bad pairs NOT recovered (residual, correctly left to future work):')
    for r in gw_rows:
        if r['verdict'] == 'bad' and not (r2w_fires(r) or r2d_fires(r)):
            print('  evt%-8s %s-%s Lmin=%.1f Tmax=%.2f npmin=%d angle=%s'
                  % (r['evt'], r['j'], r['k'], r['Lmin'], r['Tmax'], r['npmin'],
                     '%.1f' % r['angle'] if r['angle'] is not None else '-'))


def cmd_blast_radius(args):
    """Fresh, non-cached population from today's production. W-sole-voter
    edges are the entire population any W exception could touch."""
    arm = args.arm
    S6_killed = S6_wsole = 0
    S7_killed = S7_wsole = 0
    files = []
    for evt, path in A.arm_events(arm):
        files.append((evt, path))
    for evt, path in files:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d['type'] != 'edge' or not d['killed']:
                continue
            g = d['gap']
            exc = [d['excuse'][0], d['excuse'][1], False]
            voters = [p for p in range(3) if g[p] and not exc[p]]
            S6_killed += 1
            if voters == [2]:
                S6_wsole += 1
    print('arm=%s' % arm)
    print('S6 killed edges: %d   W-sole-voter (revivable): %d' % (S6_killed, S6_wsole))

    prm = A.parse_params('')
    n_pairs = n_good = n_bad = 0
    for evt, path in files:
        pt = A.pair_table(arm, evt, path, want_shown_only=False)
        for pk, p in pt.items():
            eds = p['edges']
            if not eds or not all(e['killed'] for e in eds):
                continue
            rev = False
            for e in eds:
                g = e['gap']
                exc = [e['excuse'][0], e['excuse'][1], False]
                if [q for q in range(3) if g[q] and not exc[q]] == [2]:
                    rev = True
            if not rev:
                continue
            n_pairs += 1
            v, cause, rule, conf = A.classify(p, prm)
            if v == 'good':
                n_good += 1
            elif v == 'bad':
                n_bad += 1
                print('  bad-by-classify: evt%-8s %d-%d Lmin=%.1f Lmax=%.1f Tmax=%.2f '
                      'npmin=%d angle=%.1f dis=%.2f rule=%s'
                      % (evt, p['j'], p['k'], p['Lmin'], p['Lmax'], p['Tmax'],
                         p['npmin'], p['angle'] if p['angle'] is not None else -1,
                         p['dis'], rule))
    print('fully-killed, W-sole-voter-revivable pairs: %d  (good=%d bad=%d)'
          % (n_pairs, n_good, n_bad))


def cmd_sweep(args):
    """Local (production rescue) vs global (R2w) collinearity axis, scored
    with graph replay against the round-4 label cache -- see the staleness
    caveat in the module docstring before trusting these numbers."""
    cache = args.features
    if not os.path.exists(cache):
        print('missing feature cache %s -- run:\n'
              '  python3 %s features --out %s' % (cache, os.path.join(HERE, '..', 'pr57', 'oc56_fit.py'), cache),
              file=sys.stderr)
        return
    edges, conns = F.load_features(cache)

    def coverage(f, p):
        return F.coverage(f, p)

    def rescue_variant(f, cw_max, ang_max, use_global_ang):
        if not f['killed'] or f['dis'] > 5.0:
            return False
        gu, gv, gw = f['gap']
        if not (gu or gv or gw):
            return False
        exc_u, exc_v = f['excuse']
        ab = f.get('ab_local')
        ang = f.get('axis_angle') if use_global_ang else ab
        np_, Lv = f['npmin'], f['Lmin']
        coll = ab is not None and ab < 20.0
        collw = ang is not None and ang < ang_max
        dead_w = f.get('nw_dead_w', 0) >= 3
        sub = Lv > 4.0 and np_ >= 50
        if not gw and f.get('close_mx_w') is not None \
                and f.get('ov_w', 0.0) >= 0.6 and np_ >= 50:
            for p, g in zip('uv', (gu, gv)):
                if not g and f.get('close_mx_%s' % p) is not None \
                        and f.get('ov_%s' % p, 0.0) >= 0.6:
                    return True
        if gw:
            cw = f.get('close_mx_w', 999)
            cvw = coverage(f, 'w')
            w_tiny = cw <= cw_max and cvw is not None and cvw >= 0.8
            w_ok = (dead_w and np_ >= 20) or (w_tiny and collw and sub)
            if not w_ok:
                return False
        unexc = [p for p, (g, e) in zip('uv', ((gu, exc_u), (gv, exc_v)))
                 if g and not e]
        if not unexc:
            return True
        for p in unexc:
            c = coverage(f, p)
            if c is None:
                return False
            sl = f.get('slope_%s' % p) or 0.0
            ex = f.get('ext_med_%s' % p) or 0.0
            cl = f.get('close_mx_%s' % p, 5)
            ok = False
            if dead_w and np_ >= 20 and c >= 0.65: ok = True
            if sub and coll and c >= 0.90: ok = True
            if (not gw) and np_ >= 55 and collw and c >= 0.55: ok = True
            if Lv > 5.5 and np_ >= 50 and (sl >= 5.0 or ex >= 50.0) and c >= 0.82: ok = True
            if np_ >= 15 and (sl >= 8.0 or ex >= 50.0) and c >= 0.82: ok = True
            if np_ >= 5 and collw and sl >= 15.0 and c >= 0.95: ok = True
            if (not gw) and np_ >= 150 and f['dis'] < 2.0 and cl <= 4 and c >= 0.90: ok = True
            if not ok:
                return False
        return True

    def evaluate(rescue_fn, label):
        rescued = collections.defaultdict(list)
        for f in edges:
            if rescue_fn(f):
                rescued[(f['evt'], f['call'])].append((f['j'], f['k']))
        pairs = {}
        for f in edges:
            if 'label' not in f:
                continue
            pk = (f['evt'], f['call'], f['j'], f['k'])
            pairs.setdefault(pk, dict(labels=[]))
            pairs[pk]['labels'].append(f['label'])
        bh = bm = gh = gm = 0
        good_broken = []
        for (evt, call, j, k), p in pairs.items():
            rec = conns.get((evt, call))
            if rec is None:
                continue
            verd = ('bad' if 'bad' in p['labels'] else
                    'good' if 'good' in p['labels'] else 'OK')
            before = oc56_conn.pair_status(rec, j, k)['separated']
            after = F.replay_separated(rec, rescued[(evt, call)], j, k)
            if verd == 'bad':
                if not after:
                    bh += 1
                else:
                    bm += 1
            elif verd == 'good':
                if not before:
                    continue
                if after:
                    gh += 1
                else:
                    gm += 1
                    good_broken.append((evt, call, j, k))
        print('%-38s bad %d/%d   good %d/%d  (good broken: %s)'
              % (label, bh, bh + bm, gh, gh + gm, good_broken))

    print('=== baseline: production rescue (cw<=3, LOCAL angle<15) ===')
    evaluate(lambda f: rescue_variant(f, 3, 15.0, False), 'baseline')
    print()
    print('=== widen cw alone (LOCAL angle<15) -- does not recover evt174224 ===')
    for cw in (3, 5, 8):
        evaluate(lambda f, cw=cw: rescue_variant(f, cw, 15.0, False), 'cw<=%d' % cw)
    print()
    print('=== widen LOCAL angle (cw<=5) -- recovers it but costs good pairs ===')
    for ang in (15, 20, 25):
        evaluate(lambda f, ang=ang: rescue_variant(f, 5, ang, False), 'cw<=5 ang<%d(local)' % ang)
    print()
    print('=== use GLOBAL axis instead (cw<=5) -- R2w\'s feature, much cheaper trade ===')
    for ang in (10, 15, 20):
        evaluate(lambda f, ang=ang: rescue_variant(f, 5, ang, True), 'cw<=5 ang<%d(GLOBAL)' % ang)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('owner-labels')
    p1.set_defaults(func=cmd_owner_labels)

    p2 = sub.add_parser('blast-radius')
    p2.add_argument('--arm', default='work-pr64-scan1k')
    p2.set_defaults(func=cmd_blast_radius)

    p3 = sub.add_parser('sweep')
    p3.add_argument('--features', default=os.path.expanduser('/home/xqian/tmp/pr57r6_edges2.jsonl'))
    p3.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

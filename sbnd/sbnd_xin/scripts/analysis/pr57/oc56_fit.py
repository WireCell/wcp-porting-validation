#!/usr/bin/env python3
"""doc pr/57 round 6: fit a RESCUE rule for S6-killed candidates against the
owner's full hand scan, with a faithful pair-level replay.

The owner's spec (2026-08-10):
  * the unit is the component PAIR -- connected if any edge survives;
  * bad classes: dead-W band (U/V distorted), direction-consistent tracks with
    prolonged-signal induction inefficiency, U/V allowed gap > W allowed gap;
  * don't break long tracks (use direction consistency to jump gaps);
  * when genuinely hard, prefer connectivity over separation.

Shape of the fix: S6 v2 = S6 with a rescue branch.  Every dumped killed
candidate is re-examined; if the rescue predicate fires, the candidate is NOT
killed.  v2 never kills anything S6 kept, so the only way to damage a
`good` pair is an explicit rescue -- which the replay below detects, including
transitive reconnection through third components.

Evaluation: per graph call, adjacency = the edges the code actually emitted
(connectivity record) + rescued candidates; a labelled pair scores
  bad  -> want connected,
  good -> want separated (only pairs the code separated are in this set;
          good labels on never-separated pairs are excluded upstream),
  OK   -> free (reported).

Usage:
    oc56_fit.py features --out /home/xqian/tmp/pr57r6_edges.jsonl
      # per-killed-candidate feature records over the four round-4 arms
      # (feature extraction is the slow part; cache it once)
    oc56_fit.py evaluate --features /home/xqian/tmp/pr57r6_edges.jsonl
      # apply the rescue rule, replay, print per-population confusion
      # + every disagreeing pair
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oc56_autoscan as A  # noqa: E402
import oc56_conn  # noqa: E402
from oc56_dump_check import pca_length_axis, angle_between_deg  # noqa: E402
from oc56_truth import load_labels, DEFAULT_ARMS, POPULATION  # noqa: E402

SBND = A.SBND


# ---------------------------------------------------------------------------
# per-candidate features
# ---------------------------------------------------------------------------

def matrix_min_close(m):
    """Smallest stencil that closes the plane, from the 16-char dw x ds
    matrix ('1' connected).  Returns (min_dw, min_ds, min_maxdwds) with 5
    meaning 'does not close even at (4,4)'."""
    if not m or len(m) != 16:
        return 5, 5, 5
    best_dw = best_ds = best_mx = 5
    for dw in range(1, 5):
        for ds in range(1, 5):
            if m[(dw - 1) * 4 + (ds - 1)] == '1':
                best_dw = min(best_dw, dw)
                best_ds = min(best_ds, ds)
                best_mx = min(best_mx, max(dw, ds))
    return best_dw, best_ds, best_mx


def local_dir(P, p, r=6.0):
    """Unit principal axis of the points of P within r cm of p (widened once
    if starved), or None."""
    p = np.asarray(p, dtype=float)
    for rr in (r, 2 * r):
        d = np.linalg.norm(P - p, axis=1)
        S = P[d < rr]
        if len(S) >= 5:
            C = S - S.mean(axis=0)
            w, v = np.linalg.eigh(np.cov(C.T))
            return v[:, -1] / np.linalg.norm(v[:, -1])
    return None


def acos_deg_abs(u, v):
    """Angle between two axes (sign-free), degrees."""
    if u is None or v is None:
        return None
    c = abs(float(np.dot(u, v)))
    return float(np.degrees(np.arccos(min(1.0, c))))


def plane_track_slope(cells):
    """|dslice/dwire| of a straight fit through plane cells; large = the track
    is near-isochronous in this plane view (prolonged-signal territory).
    None when degenerate."""
    if len(cells) < 3:
        return None
    c = np.asarray(cells, dtype=float)
    dw = c[:, 0].max() - c[:, 0].min()
    dsl = c[:, 1].max() - c[:, 1].min()
    if dw < 1:
        return 999.0
    return float(dsl / dw)


def fired_extent_stats(pl):
    """Per-wire fired-slice tick extent inside the window, restricted to wires
    between the two seed footprints (the gap region plus its shoulders).
    Returns (median_extent, max_extent, n_wires_with_signal, n_wires_between,
    dead_wires_between)."""
    sa = [c[0] for c in pl['seeds_a']]
    sb = [c[0] for c in pl['seeds_b']]
    if not sa or not sb:
        return None
    lo = min(min(sa), min(sb))
    hi = max(max(sa), max(sb))
    per_wire = collections.defaultdict(list)
    for w, s in pl['fired']:
        if lo <= w <= hi:
            per_wire[w].append(s)
    exts = [max(v) - min(v) for v in per_wire.values()]
    dead_between = len({d[0] for d in pl['dead'] if lo <= d[0] <= hi})
    return (float(np.median(exts)) if exts else 0.0,
            float(max(exts)) if exts else 0.0,
            len(per_wire), hi - lo + 1, dead_between)


def edge_features(evt, arm, rec, comps):
    call = rec['graph_call']
    j, k = min(rec['j'], rec['k']), max(rec['j'], rec['k'])
    Ap, Bp = comps.get((call, rec['j'])), comps.get((call, rec['k']))
    if Ap is None or Bp is None:
        return None
    p1, p2 = np.asarray(rec['p1']), np.asarray(rec['p2'])
    ev = p2 - p1
    nv = np.linalg.norm(ev)
    ev = ev / nv if nv > 0 else None
    La, axa = pca_length_axis(Ap)
    Lb, axb = pca_length_axis(Bp)
    da = local_dir(Ap, p1)
    db = local_dir(Bp, p2)

    f = dict(evt=evt, arm=os.path.basename(arm.rstrip('/')), call=call,
             j=j, k=k, blk=rec['blk'], dis=rec['dis'],
             killed=rec['killed'], below_floor=rec.get('below_floor', False),
             gap=[bool(g) for g in rec['gap']],
             excuse=[bool(e) for e in rec['excuse']],
             budget=[bool(b) for b in rec.get('budget', [0, 0, 0])],
             Lmin=min(La, Lb), Lmax=max(La, Lb),
             npmin=min(len(Ap), len(Bp)), npmax=max(len(Ap), len(Bp)),
             axis_angle=angle_between_deg(axa, axb),
             # direction consistency at the break: local axis at each endpoint
             # vs the candidate edge, and the two local axes vs each other.
             a_edge=acos_deg_abs(da, ev), b_edge=acos_deg_abs(db, ev),
             ab_local=acos_deg_abs(da, db))
    for pl in rec.get('planes', []):
        p = pl['plane']
        st = fired_extent_stats(pl)
        tag = 'uvw'[p]
        mdw, mds, mmx = matrix_min_close((rec.get('matrix') or ['', '', ''])[p])
        f['close_dw_%s' % tag] = mdw
        f['close_ds_%s' % tag] = mds
        f['close_mx_%s' % tag] = mmx
        # doc pr/57 round 6: local footprint overlap of the two components in
        # this plane view (seed wire ranges).  End-to-end breaks give adjacent
        # DISJOINT ranges; a fragment lying alongside / inside the other
        # component overlaps.  Fraction of the smaller range covered.
        sa = [c[0] for c in pl['seeds_a']]
        sb = [c[0] for c in pl['seeds_b']]
        if sa and sb:
            ov = min(max(sa), max(sb)) - max(min(sa), min(sb)) + 1
            wmin = min(max(sa) - min(sa), max(sb) - min(sb)) + 1
            f['ov_%s' % tag] = max(0.0, ov / float(wmin))
            ta = [c[1] for c in pl['seeds_a']]
            tb = [c[1] for c in pl['seeds_b']]
            ovs = min(max(ta), max(tb)) - max(min(ta), min(tb)) + 1
            smin = min(max(ta) - min(ta), max(tb) - min(tb)) + 1
            f['ovs_%s' % tag] = max(0.0, ovs / float(smin))
        if st:
            (f['ext_med_%s' % tag], f['ext_max_%s' % tag],
             f['nw_sig_%s' % tag], f['nw_span_%s' % tag],
             f['nw_dead_%s' % tag]) = st
        sl = plane_track_slope(pl['seeds_a'] + pl['seeds_b'])
        f['slope_%s' % tag] = sl
        f['step'] = rec['slice_step']
    return f


def cmd_features(args):
    arms = [os.path.join(SBND, a) for a in DEFAULT_ARMS]
    truth = load_labels(['overclustering_labels',
                         'overclustering_labels/claude-scan50',
                         'overclustering_labels/claude-scan223'])
    n = 0
    with open(args.out, 'w') as fh:
        for arm in arms:
            for evt, path in A.arm_events(arm):
                comps, edges, conn = A.load_event(path, want_conn=True)
                # connectivity records: needed whole for the replay
                for rec in conn.values():
                    fh.write(json.dumps(dict(
                        kind='conn', evt=evt,
                        arm=os.path.basename(arm.rstrip('/')),
                        rec=rec)) + '\n')
                for rec in edges:
                    f = edge_features(evt, arm, rec, comps)
                    if f is None:
                        continue
                    kk = A.edge_key(evt, rec)
                    if kk in truth:
                        f['label'] = truth[kk][0]
                    f['kind'] = 'edge'
                    fh.write(json.dumps(f) + '\n')
                    n += 1
    print('wrote %d edge feature records -> %s' % (n, args.out))


# ---------------------------------------------------------------------------
# the rescue rule (iterated by hand, round by round)
# ---------------------------------------------------------------------------

def coverage(f, p):
    """Fraction of wires between the two seed footprints (in plane p's view)
    that carry ANY fired cell in the window.  ~1.0 => the plane sees signal on
    every wire and only fails time-contiguity -- the prolonged/inefficiency
    artifact signature.  A genuine gap has wires with no signal at all."""
    s, n = f.get('nw_sig_%s' % p), f.get('nw_span_%s' % p)
    if s is None or not n:
        return None
    return s / float(n)


def rescue(f, prm=None):
    """True => this S6-killed candidate is kept in v2 (doc pr/57 round 6).

    Fitted against the owner's full 2026-08-10 hand scan (899 labels).  Each
    branch maps to a stated principle:
      * dead-W band (owner bad case a): dead channels mid-TPC distort the
        induction response -- coverage floor is LOW here (0.65) because the
        distortion itself makes holes;
      * W-robustness (case c): a W gap is only forgivable when tiny
        (closes at a (3,3) stencil), well-covered and the break is collinear,
        or dead-explained;
      * direction consistency (case b / "jump gaps on long tracks"):
        local PCA axes at the two break points agree;
      * prolonged signal (case b): the gapped induction plane's per-wire time
        extent or its track slope in (wire,time) is large;
      * co-location: the two components' seed footprints OVERLAP in W plus
        one more connected plane -- two views say one object, not end-to-end
        fragments (this is what the render round showed for the residuals).
    Uses only information available at the C++ kill site (no vertex, no
    whole-cluster density)."""
    if not f['killed'] or f['dis'] > 5.0:
        return False
    gu, gv, gw = f['gap']
    if not (gu or gv or gw):
        return False
    exc_u, exc_v = f['excuse']
    ab = f.get('ab_local')
    np_, L = f['npmin'], f['Lmin']
    coll = ab is not None and ab < 20.0
    collw = ab is not None and ab < 15.0
    dead_w = f.get('nw_dead_w', 0) >= 3
    sub = L > 4.0 and np_ >= 50

    # co-location: overlap in W (connected) + >=1 more connected plane
    if not gw and f.get('close_mx_w') is not None \
            and f.get('ov_w', 0.0) >= 0.6 and np_ >= 50:
        for p, g in zip('uv', (gu, gv)):
            if not g and f.get('close_mx_%s' % p) is not None \
                    and f.get('ov_%s' % p, 0.0) >= 0.6:
                return True

    if gw:
        cw = f.get('close_mx_w', 5)
        cvw = coverage(f, 'w')
        w_tiny = cw <= 3 and cvw is not None and cvw >= 0.8
        w_ok = (dead_w and np_ >= 20) or (w_tiny and collw and sub)
        if not w_ok:
            return False

    unexc = [p for p, (g, e) in zip('uv', ((gu, exc_u), (gv, exc_v)))
             if g and not e]
    if not unexc:
        return True  # W-side already vouched for; induction gaps all excused
    for p in unexc:
        c = coverage(f, p)
        if c is None:
            return False
        sl = f.get('slope_%s' % p) or 0.0
        ex = f.get('ext_med_%s' % p) or 0.0
        cl = f.get('close_mx_%s' % p, 5)
        ok = False
        if dead_w and np_ >= 20 and c >= 0.65:
            ok = True
        if sub and coll and c >= 0.90:
            ok = True
        if (not gw) and np_ >= 55 and collw and c >= 0.55:
            ok = True
        if L > 5.5 and np_ >= 50 and (sl >= 5.0 or ex >= 50.0) and c >= 0.82:
            ok = True
        if np_ >= 15 and (sl >= 8.0 or ex >= 50.0) and c >= 0.82:
            ok = True
        if np_ >= 5 and collw and sl >= 15.0 and c >= 0.95:
            ok = True
        if (not gw) and np_ >= 150 and f['dis'] < 2.0 and cl <= 4 \
                and c >= 0.90:
            ok = True
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# replay + score
# ---------------------------------------------------------------------------

def load_features(path):
    edges, conns = [], {}
    for line in open(path):
        r = json.loads(line)
        if r.get('kind') == 'conn':
            conns[(r['evt'], r['rec']['graph_call'])] = r['rec']
        else:
            edges.append(r)
    return edges, conns


def replay_separated(conn_rec, extra_pairs, j, k):
    """Pair separated after adding rescued (j,k) edges to the emitted set?"""
    adj = collections.defaultdict(set)
    for e in conn_rec.get('edges', []):
        adj[e['j']].add(e['k'])
        adj[e['k']].add(e['j'])
    for a, b in extra_pairs:
        adj[a].add(b)
        adj[b].add(a)
    seen = {j}
    q = collections.deque([j])
    while q:
        v = q.popleft()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    return k not in seen


def cmd_evaluate(args):
    edges, conns = load_features(args.features)
    prm = dict(A=25.0, L=6.0, N=50, DW=3)
    if args.params:
        for tok in args.params.split(','):
            kk, v = tok.split('=')
            prm[kk] = float(v)
    prm['N'] = int(prm['N'])
    prm['DW'] = int(prm['DW'])

    rescued = collections.defaultdict(list)   # (evt, call) -> [(j,k)]
    n_resc = 0
    for f in edges:
        if rescue(f, prm):
            rescued[(f['evt'], f['call'])].append((f['j'], f['k']))
            n_resc += 1
    n_killed = sum(1 for f in edges if f['killed'])
    print('rescued %d of %d killed candidates (params %s)'
          % (n_resc, n_killed, prm))

    # labelled pairs: verdict dominance (bad > good > OK) over member edges
    pairs = {}
    for f in edges:
        if 'label' not in f:
            continue
        pk = (f['evt'], f['call'], f['j'], f['k'])
        pairs.setdefault(pk, dict(arm=f['arm'], labels=[], feats=[]))
        pairs[pk]['labels'].append(f['label'])
        pairs[pk]['feats'].append(f)

    cm = collections.defaultdict(collections.Counter)
    fails = []
    ok_flips = collections.Counter()
    for (evt, call, j, k), p in sorted(pairs.items()):
        rec = conns.get((evt, call))
        if rec is None:
            continue
        verd = ('bad' if 'bad' in p['labels'] else
                'good' if 'good' in p['labels'] else 'OK')
        before = oc56_conn.pair_status(rec, j, k)['separated']
        after = replay_separated(rec, rescued[(evt, call)], j, k)
        pop = POPULATION.get(p['arm'], p['arm'])
        if verd == 'bad':
            want_conn = True
            hit = not after
            cm[pop]['bad_hit' if hit else 'bad_miss'] += 1
            if not hit:
                fails.append((pop, evt, call, j, k, 'bad', 'still separated',
                              p['feats']))
        elif verd == 'good':
            if not before:
                cm[pop]['good_excluded_nonSEP'] += 1
                continue
            hit = after
            cm[pop]['good_hit' if hit else 'good_miss'] += 1
            if not hit:
                fails.append((pop, evt, call, j, k, 'good', 'reconnected',
                              p['feats']))
        else:
            ok_flips['%s->%s' % ('SEP' if before else 'conn',
                                 'SEP' if after else 'conn')] += 1

    print('\nper-population (bad: want connected; good(SEP): want separated):')
    tot = collections.Counter()
    for pop in sorted(cm):
        c = cm[pop]
        print('  %-8s bad %d/%d   good %d/%d   (good excluded non-SEP: %d)'
              % (pop, c['bad_hit'], c['bad_hit'] + c['bad_miss'],
                 c['good_hit'], c['good_hit'] + c['good_miss'],
                 c['good_excluded_nonSEP']))
        tot.update(c)
    print('  %-8s bad %d/%d   good %d/%d' % (
        'TOTAL', tot['bad_hit'], tot['bad_hit'] + tot['bad_miss'],
        tot['good_hit'], tot['good_hit'] + tot['good_miss']))
    print('  OK pairs outcome: %s' % dict(ok_flips))

    print('\ndisagreements (%d):' % len(fails))
    for pop, evt, call, j, k, verd, what, feats in fails:
        kf = [f for f in feats if f['killed']]
        f0 = min(kf, key=lambda f: f['dis']) if kf else feats[0]
        print('  %-8s evt%s c%d %d-%d %-4s %-15s dis=%.2f gap=%s exc=%s '
              'L=%.1f/%.1f np=%d ang(a,b,ab)=%s clo(mx uvw)=%s deadw=%s '
              'extmed(uv)=%s slope(uv)=%s'
              % (pop, evt, call, j, k, verd, what, f0['dis'],
                 ''.join(str(int(g)) for g in f0['gap']),
                 ''.join(str(int(e)) for e in f0['excuse']),
                 f0['Lmin'], f0['Lmax'], f0['npmin'],
                 ','.join('%.0f' % f0[a] if f0.get(a) is not None else '-'
                          for a in ('a_edge', 'b_edge', 'ab_local')),
                 ','.join(str(f0.get('close_mx_%s' % t, '-'))
                          for t in 'uvw'),
                 f0.get('nw_dead_w', '-'),
                 ','.join(str(f0.get('ext_med_%s' % t, '-')) for t in 'uv'),
                 ','.join('%.1f' % f0['slope_%s' % t]
                          if f0.get('slope_%s' % t) is not None else '-'
                          for t in 'uv')))


def cmd_table(args):
    """The owner's acceptance criterion (doc pr/57 round 6): per labelled
    pair, owner verdict x what the NEW (rescue-flavor) arms actually did.
    Also cross-checks, per rescued/killed candidate, that the C++ predicate
    agrees with this script's rescue() on this script's own features."""
    import oc56_conn
    truth = load_labels(['overclustering_labels',
                         'overclustering_labels/claude-scan50',
                         'overclustering_labels/claude-scan223'])
    # pre-rescue pair state, from the round-4 truth table (was this pair
    # separated by the code the owner scanned?)
    pre = {}
    tsv = os.path.join(SBND, 'docs', 'pr', 'pr57r6-truth.tsv')
    if os.path.isfile(tsv):
        import csv
        for r in csv.DictReader(open(tsv), delimiter='\t'):
            pre[(r['evt'], int(r['call']), int(r['j']), int(r['k']))] = r['pair']

    cm = collections.defaultdict(collections.Counter)
    fails, model_mism, nchk = [], [], 0
    npair = 0
    for arm in args.arm:
        base = os.path.basename(arm.rstrip('/'))
        pop = {'work-pr57r6-scan48': 'nueCC48', 'work-pr57r6-scan19': 'NCpi0'}.get(
            base, 'PR-data' if 'scan' in base else base)
        for evt, path in A.arm_events(arm):
            comps, edges, conn = A.load_event(path, want_conn=True)
            pairs = {}
            for rec in edges:
                # candidate-level python==C++ check
                if rec.get('two_d_rescue') and rec.get('killed_pre_rescue') \
                        and rec['dis'] <= 5.0:
                    f = edge_features(evt, arm, rec, comps)
                    if f is not None:
                        f['killed'] = True
                        nchk += 1
                        if rescue(f) != bool(rec['rescued']):
                            model_mism.append((evt, rec['graph_call'],
                                               rec['j'], rec['k'], rec['blk'],
                                               bool(rec['rescued']), f,
                                               rec.get('v2')))
                kk = A.edge_key(evt, rec)
                if kk in truth:
                    pk = (rec['graph_call'], min(rec['j'], rec['k']),
                          max(rec['j'], rec['k']))
                    pairs.setdefault(pk, []).append(truth[kk][0])
            for (call, j, k), labs in sorted(pairs.items()):
                verd = ('bad' if 'bad' in labs else
                        'good' if 'good' in labs else 'OK')
                st = oc56_conn.pair_status(conn.get(call), j, k)
                if st['separated'] is None:
                    cm[pop]['%s_nostatus' % verd] += 1
                    continue
                ach = 'SEP' if st['separated'] else 'conn'
                was = pre.get((evt, call, j, k), '?')
                npair += 1
                if verd == 'bad':
                    ok = ach == 'conn'
                    cm[pop]['bad_%s' % ('hit' if ok else 'MISS')] += 1
                    if not ok:
                        fails.append((pop, evt, call, j, k, 'bad',
                                      'still separated'))
                elif verd == 'good':
                    if was != 'SEP':
                        cm[pop]['good_preconn'] += 1  # never separated before
                        continue
                    ok = ach == 'SEP'
                    cm[pop]['good_%s' % ('hit' if ok else 'MISS')] += 1
                    if not ok:
                        fails.append((pop, evt, call, j, k, 'good',
                                      'reconnected'))
                else:
                    cm[pop]['OK_%s' % ach] += 1

    print('=== FINAL TABLE: owner verdict vs achieved pair state '
          '(%d labelled pairs joined) ===' % npair)
    tot = collections.Counter()
    for pop in sorted(cm):
        c = cm[pop]
        tot.update(c)
        print('  %-8s bad %3d/%-3d  good(SEP) %3d/%-3d  '
              '[good pre-connected: %d]  OK: %d SEP / %d conn' % (
                  pop, c['bad_hit'], c['bad_hit'] + c['bad_MISS'],
                  c['good_hit'], c['good_hit'] + c['good_MISS'],
                  c['good_preconn'], c['OK_SEP'], c['OK_conn']))
    print('  %-8s bad %3d/%-3d  good(SEP) %3d/%-3d' % (
        'TOTAL', tot['bad_hit'], tot['bad_hit'] + tot['bad_MISS'],
        tot['good_hit'], tot['good_hit'] + tot['good_MISS']))
    print('\ndisagreements (%d):' % len(fails))
    for t in fails:
        print('  %-8s evt%s c%d %d-%d %s -> %s' % t)
    print('\ncandidate-level python==C++ rescue check: %d checked, '
          '%d mismatches %s' % (nchk, len(model_mism),
                                'PASS' if not model_mism else 'FAIL'))
    for evt, call, j, k, blk, cxx, f, v2 in model_mism[:10]:
        print('  MISMATCH evt%s c%d %d-%d %s cxx_rescued=%s' %
              (evt, call, j, k, blk, cxx))
        if v2:
            print('    cxx v2: %s' % json.dumps(v2))
        print('    py: ab=%s np=%s L=%.2f cov=%s' % (
            f.get('ab_local'), f['npmin'], f['Lmin'],
            [f.get('nw_sig_%s' % t) for t in 'uvw']))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('features')
    s.add_argument('--out', required=True)
    s = sub.add_parser('evaluate')
    s.add_argument('--features', required=True)
    s.add_argument('--params', default='')
    s = sub.add_parser('table')
    s.add_argument('--arm', action='append', required=True)
    args = ap.parse_args()
    {'features': cmd_features, 'evaluate': cmd_evaluate,
     'table': cmd_table}[args.cmd](args)


if __name__ == '__main__':
    main()

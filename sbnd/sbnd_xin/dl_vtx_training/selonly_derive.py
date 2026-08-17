#!/usr/bin/env python3
'''doc pr/89 round 5 (sec 13) -- derive the selection-only s_topo variant's
LIVE outcome from the round-4 live arms, with zero new runs.

Basis (NeutrinoVertexFinder.cxx:4832-4847): the rerank winner is
argmax(composite) and acceptance compares the same composite to
min_accept.  The selection-only variant re-defines acceptance to use the
7-term max (identical to knob-off production), keeping the topo-informed
argmax.  Because the round-4 arms share candidate sets and s_topo >= 0 at
center 0 (acceptance can only flip reject->accept, never back):

  base acceptance passed (route dl-rerank-accept / dl-veto-protected)
      -> selection-only winner = the topo arm's winner and every
         downstream step (veto, snap, improve) sees exactly what the topo
         arm saw  -> final vertex = the TOPO arm's final vertex.
  base acceptance failed (any other route)
      -> selection-only acceptance fails too -> final = the BASE arm's.

The c05 variant (w=3, center=0.5) is NOT exactly derivable: s_topo' can be
negative, so acceptance and argmax both move and the resulting final
vertex may not have been observed in either arm.  We classify each event
and score only the exact subset, reporting the unknowns.

Repro:
  python3 selonly_derive.py            # all 1014 labelled events
'''
import os, sys, json, csv, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TAGS = ['vtxscan-mcp2k', 'vtxscan-mcp2k-auto', 'vtxscan-harv3-nuecc48',
        'vtxscan-harv3-ncpi0', 'vtxscan-harv3-mcp1k', 'vtxscan-harv3-delta']
ACCEPT_FAMILY = ('dl-rerank-accept', 'dl-veto-protected')
W_TOPO, C05 = 3.0, 0.5
MIN_ACCEPT = 10.0


def sample_of(lab):
    t = lab['scan_tag']
    if t in ('vtxscan-mcp2k', 'vtxscan-mcp2k-auto'):
        return 'mcp2k'
    for s in ('nuecc48', 'ncpi0', 'mcp1k'):
        if s in t:
            return s
    a = lab.get('arm') or ''
    return 'nuecc48' if 'nuecc48' in a else 'mcp1k'


def final_d(sb, truth):
    return float(np.linalg.norm(
        np.array([sb['final_x'], sb['final_y'], sb['final_z']]) - truth))


def main():
    ipw = {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'runs/ipw-mcp2k-closed-20260816.tsv')) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            ipw[int(r['evt'])] = float(r['weight'])

    per = collections.defaultdict(lambda: collections.Counter())
    wagg = collections.Counter()
    fixed, regressed, nearmiss = [], [], []
    c05_cls = collections.Counter()
    c05_net_exact = 0
    seen = set()
    for lab in vio.iter_labels(ROOT, TAGS):
        e = lab['eventNo']
        if e in seen:
            continue
        seen.add(e)
        s = sample_of(lab)
        truth = np.array(lab['truth_xyz'])
        b = json.load(open('%s/work-%s-pr89base/pr_evt%d/calib-pr-evt%d.json'
                           % (ROOT, s, e, e)))['vertex_scoreboard']
        t = json.load(open('%s/work-%s-pr89topo/pr_evt%d/calib-pr-evt%d.json'
                           % (ROOT, s, e, e)))['vertex_scoreboard']
        db, dt = final_d(b, truth), final_d(t, truth)
        # closure guard: s_topo >= 0 must never un-accept
        if b['route'] in ACCEPT_FAMILY and t['route'] not in ACCEPT_FAMILY:
            print('CLOSURE VIOLATION evt %d: base %s -> topo %s'
                  % (e, b['route'], t['route']))
        # -------- selection-only (exact) --------
        dsel = dt if b['route'] in ACCEPT_FAMILY else db
        w = ipw.get(e, 1.0)
        cb, cs = int(db < 1.0), int(dsel < 1.0)
        p = per[s]
        p['n'] += 1
        p['base'] += cb
        p['sel'] += cs
        p['base15'] += int(db < 1.5)
        p['sel15'] += int(dsel < 1.5)
        wagg['w'] += w
        wagg['wb'] += w * cb
        wagg['ws'] += w * cs
        if cs > cb:
            fixed.append((s, e, db, dsel))
        if cb > cs:
            regressed.append((s, e, db, dsel))
        if 1.0 <= dsel < 2.0:
            nearmiss.append((s, e, db, dsel))
        # -------- c05 (bounded) --------
        rows = [r for r in t.get('rows', []) if r.get('dl_snapped')]
        best_tot, best_vid = -1e18, None
        for r in rows:
            base7 = float(r['total']) - float(r.get('s_topo') or 0.0)
            st = (W_TOPO * (float(r['topo_frac']) - C05)
                  if int(r.get('topo_votes') or 0) >= 1 else 0.0)
            tot = base7 + st
            if tot > best_tot:
                best_tot, best_vid = tot, r['vertex_id']
        topo_win = next((r['vertex_id'] for r in t.get('rows', [])
                         if r.get('dl_winner')), None)
        base_win = next((r['vertex_id'] for r in b.get('rows', [])
                         if r.get('dl_winner')), None)
        if not rows:
            c05_cls['no-candidates(=base)'] += 1
            dc = db
        elif best_tot >= MIN_ACCEPT:
            if best_vid == topo_win and t['route'] in ACCEPT_FAMILY:
                c05_cls['accept, winner=topo-arm'] += 1
                dc = dt
            elif best_vid == base_win and b['route'] in ACCEPT_FAMILY:
                c05_cls['accept, winner=base-arm'] += 1
                dc = db
            else:
                c05_cls['UNKNOWN (accepted winner unobserved)'] += 1
                dc = None
        else:
            if b['route'] not in ACCEPT_FAMILY:
                c05_cls['reject(=base)'] += 1
                dc = db
            else:
                c05_cls['UNKNOWN (un-accept, trad unobserved)'] += 1
                dc = None
        if dc is not None:
            c05_net_exact += int(dc < 1.0) - int(db < 1.0)

    tot = collections.Counter()
    print('== selection-only (EXACT derivation), correct@1.0cm ==')
    print('%-8s %5s | base  sel   net | base@1.5 sel@1.5' % ('sample', 'n'))
    for s in ('mcp2k', 'nuecc48', 'ncpi0', 'mcp1k'):
        p = per[s]
        for k in p:
            tot[k] += p[k]
        print('%-8s %5d | %4d %4d  %+3d | %5d    %5d'
              % (s, p['n'], p['base'], p['sel'], p['sel'] - p['base'],
                 p['base15'], p['sel15']))
    print('%-8s %5d | %4d %4d  %+3d | %5d    %5d   <== TOTAL'
          % ('ALL', tot['n'], tot['base'], tot['sel'],
             tot['sel'] - tot['base'], tot['base15'], tot['sel15']))
    print('IPW: base %.2f%% -> sel %.2f%%'
          % (100 * wagg['wb'] / wagg['w'], 100 * wagg['ws'] / wagg['w']))
    print('\nfixed (%d):' % len(fixed))
    for s, e, db, ds in fixed:
        print('  %-8s evt%-8d %7.2f -> %6.2f' % (s, e, db, ds))
    print('regressed (%d):' % len(regressed))
    for s, e, db, ds in regressed:
        print('  %-8s evt%-8d %7.2f -> %6.2f' % (s, e, db, ds))
    print('near-miss 1.0<=d<2.0 in sel arm (%d):' % len(nearmiss))
    for s, e, db, ds in nearmiss:
        print('  %-8s evt%-8d %7.2f -> %6.3f' % (s, e, db, ds))
    print('\n== c05 (w=3, center=0.5), BOUNDED ==')
    for k, v in sorted(c05_cls.items()):
        print('  %-40s %d' % (k, v))
    print('  exact-subset net vs base: %+d' % c05_net_exact)


if __name__ == '__main__':
    main()

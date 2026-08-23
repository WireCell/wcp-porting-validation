#!/usr/bin/env python3
'''doc pr/112 sec 11 -- score a LIVE dual-chain arm against a baseline arm.

Objective (owner, 2026-08-23): the neutrino-vertex metric against the hand
scan.  pr/106 target metric, as pr112_dualchain_sim.py computes it: on each
arm's OWN pre-DL candidate set (hv_cloud vertex rows = the exact live SCN
input), TARGET = the row nearest the hand-scan click; HIT = the row nearest
the SHIPPED main_vertex is the target.  Candidate identity, so refinement-
and fit-epoch-immune.  The candidate set is built before the transfer in
every mode, so baseline and arm share it event by event (asserted).

Guards reported alongside: nue_score >= 4.3 count (the pr/106 sec 10 rule),
numu_score >= 0.9 count, movers (pick changed) with the 1 cm ADVERSE ruler
(shipped vertex moved from <= 1 cm of the click to > 1 cm), the agreement
flag census from the scoreboard's dual_chain block, OFF-pass wall.

Usage: ./pr112_dual_eval.py --sample nuecc48 --base work-pr112i-off-nuecc48 \\
           work-pr112i-snapD2-nuecc48 [more arms ...] [--tsv out.tsv]
'''
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), 'dl_vtx_training')
sys.path.insert(0, TRAIN)
from scn_vtx import io as vio      # noqa: E402

TAGS = {'nuecc48': ['vtxscan-harv3-nuecc48'], 'ncpi0': ['vtxscan-harv3-ncpi0'],
        'mcp1k': ['vtxscan-harv3-mcp1k'], 'numu100': ['vtxscan-harv3-mcp1k'],
        'mcp2k': ['vtxscan-mcp2k', 'vtxscan-mcp2k-auto', 'vtxscan-mcp2k-ragree']}
NUE_CUT, NUMU_CUT = 4.3, 0.9


def board(sb):
    c = sb.get('hv_cloud')
    if not c or not int(c.get('n_vertex_rows', 0)):
        return None, None
    n = int(c['n_vertex_rows'])
    return (list(c['vertex_ids'][:n]),
            np.array([c['x'][:n], c['y'][:n], c['z'][:n]], float).T)


def final_xyz(j):
    mv = j.get('main_vertex') or {}
    if mv.get('x') is not None:
        return np.array([mv['x'], mv['y'], mv['z']], float)
    return None


def read_arm(root, arm, evt):
    p = os.path.join(root, arm, 'pr_evt%d' % evt, 'calib-pr-evt%d.json' % evt)
    if not os.path.exists(p):
        return None
    j = vio.load_calib(p)
    sb = j.get('vertex_scoreboard') or {}
    ids, xyz = board(sb)
    fx = final_xyz(j)
    if ids is None or fx is None:
        return None
    t = j.get('tagger') or {}
    return dict(ids=ids, xyz=xyz, fx=fx, dual=sb.get('dual_chain'), route=sb.get('route'),
                nue=t.get('nue_score'), numu=t.get('numu_score'))


def score(root, sample, base, arm, only=None):
    rows = []
    for lab in vio.iter_labels(root, TAGS[sample]):
        e = int(lab['eventNo'])
        if only is not None and e not in only:
            continue
        b, a = read_arm(root, base, e), read_arm(root, arm, e)
        if b is None or a is None:
            continue
        if b['ids'] != a['ids']:
            print('WARN evt %d: candidate set differs between %s and %s (%d vs %d rows) -- scored on each arm\'s own'
                  % (e, base, arm, len(b['ids']), len(a['ids'])), file=sys.stderr)
        tr = np.asarray(lab['truth_xyz'], float)
        def pick(x):
            tgt = x['ids'][int(np.argmin(np.linalg.norm(x['xyz'] - tr, axis=1)))]
            pk = x['ids'][int(np.argmin(np.linalg.norm(x['xyz'] - x['fx'], axis=1)))]
            return tgt, pk, float(np.linalg.norm(x['fx'] - tr))
        tb, pb, db = pick(b)
        ta, pa, da = pick(a)
        d = a['dual'] or {}
        rows.append(dict(evt=e, hit_base=int(pb == tb), hit_arm=int(pa == ta), moved=int(pa != pb),
                         d_base=db, d_arm=da, adverse=int(db <= 1.0 and da > 1.0), rescued=int(db > 1.0 and da <= 1.0),
                         nue_b=b['nue'], nue_a=a['nue'], numu_b=b['numu'], numu_a=a['numu'],
                         agree=d.get('agree'), transferred=d.get('transferred'), dual_d=d.get('d'),
                         has_vertex=d.get('has_vertex'), off_ms=d.get('off_ms'), route=a['route']))
    return rows


def summarize(rows, base, arm):
    n = len(rows)
    hb, ha = sum(r['hit_base'] for r in rows), sum(r['hit_arm'] for r in rows)
    fix = sorted(r['evt'] for r in rows if r['hit_arm'] and not r['hit_base'])
    brk = sorted(r['evt'] for r in rows if r['hit_base'] and not r['hit_arm'])
    adv = sorted(r['evt'] for r in rows if r['adverse'])
    resc = sorted(r['evt'] for r in rows if r['rescued'])
    def cnt(k, cut):
        return sum(1 for r in rows if r[k] is not None and r[k] >= cut)
    print('\n== %s  vs base %s   n=%d' % (arm, base, n))
    print('  target-hit   base %d/%d   arm %d/%d   (%+d)   fixed %d %s   broken %d %s'
          % (hb, n, ha, n, ha - hb, len(fix), fix, len(brk), brk))
    print('  movers (pick changed) %d ; 1 cm ruler: ADVERSE %d %s  rescued %d %s'
          % (sum(r['moved'] for r in rows), len(adv), adv, len(resc), resc))
    print('  nue>=%.1f  base %d  arm %d    numu>=%.1f  base %d  arm %d'
          % (NUE_CUT, cnt('nue_b', NUE_CUT), cnt('nue_a', NUE_CUT), NUMU_CUT, cnt('numu_b', NUMU_CUT), cnt('numu_a', NUMU_CUT)))
    dual = [r for r in rows if r['agree'] is not None]
    if dual:
        ag = [r for r in dual if r['agree']]
        dis = [r for r in dual if not r['agree']]
        print('  dual: recorded %d  has_vertex %d  transferred %d  agree %d (arm right %d)  disagree %d (arm right %d)  off_ms median %.0f'
              % (len(dual), sum(1 for r in dual if r['has_vertex']), sum(1 for r in dual if r['transferred']),
                 len(ag), sum(r['hit_arm'] for r in ag), len(dis), sum(r['hit_arm'] for r in dis),
                 np.median([r['off_ms'] for r in dual if r['off_ms'] is not None] or [0])))
        from collections import Counter
        print('  routes: %s' % dict(Counter(r['route'] for r in dual)))
    return dict(n=n, hb=hb, ha=ha, adverse=len(adv), fixed=len(fix), broken=len(brk))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('arms', nargs='+')
    ap.add_argument('--only', help='file of event ids')
    ap.add_argument('--tsv')
    a = ap.parse_args()
    root = vio.default_sbnd_root()
    only = None
    if a.only:
        only = set(int(x) for x in open(a.only).read().split())
    elif a.sample == 'numu100':
        only = set(int(x) for x in open(os.path.join(HERE, 'pr112_numu100.txt')).read().split())
    allrows = []
    for arm in a.arms:
        rows = score(root, a.sample, a.base, arm, only)
        summarize(rows, a.base, arm)
        for r in rows:
            r['arm'] = arm
        allrows += rows
    if a.tsv:
        cols = ['arm', 'evt', 'hit_base', 'hit_arm', 'moved', 'd_base', 'd_arm', 'adverse', 'rescued',
                'nue_b', 'nue_a', 'numu_b', 'numu_a', 'agree', 'transferred', 'dual_d', 'has_vertex', 'off_ms', 'route']
        with open(a.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in allrows:
                fh.write('\t'.join('' if r[c] is None else str(r[c]) for c in cols) + '\n')
        print('\nwrote %s' % a.tsv)


if __name__ == '__main__':
    sys.exit(main())

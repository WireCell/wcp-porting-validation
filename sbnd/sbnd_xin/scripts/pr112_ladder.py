#!/usr/bin/env python3
'''doc pr/112 sec 11.6 -- the four-rung ladder that explains how much of the
exclusion-free chain's vertex advantage a TRANSFER can reach.

Owner (2026-08-23): "I recall that with no exclusion fit alone, the nu vertex
in all samples are significantly better than that of the exclusion fit.  I
would expect that once we combine both in the current implementation, we will
get something similar?  Why we did not?"

Rungs, all on the pr/106 target metric (candidate identity, epoch-immune):

  R0  production            base arm, hit on its OWN cloud
  R1  exclusion-free alone  nofitx arm, hit on ITS OWN cloud   <- pr/106 sec 9
  R2  transfer ceiling      OFF chain's shipped vertex snapped to the nearest
                            PRODUCTION candidate, scored on PRODUCTION's cloud
                            (the best any positional transfer can do)
  R3  delivered             the live dual-chain arm(s)

R1 -> R2 is the part of the advantage that lives in the exclusion-free
CANDIDATE SET rather than in its ranking: a transfer moves a position, so it
can only re-rank rows production already has.  Decomposed here per event.

Usage: ./pr112_ladder.py --sample mcp2k [--arms uniW0 snapD2] [--tsv out.tsv]
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
        'mcp1k': ['vtxscan-harv3-mcp1k'],
        'mcp2k': ['vtxscan-mcp2k', 'vtxscan-mcp2k-auto', 'vtxscan-mcp2k-ragree']}


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


def read(root, arm, evt):
    p = os.path.join(root, arm, 'pr_evt%d' % evt, 'calib-pr-evt%d.json' % evt)
    if not os.path.exists(p):
        return None
    j = vio.load_calib(p)
    ids, xyz = board(j.get('vertex_scoreboard') or {})
    fx = final_xyz(j)
    if ids is None or fx is None:
        return None
    return dict(ids=ids, xyz=xyz, fx=fx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', required=True)
    ap.add_argument('--base', default='work-pr112i-off-%s')
    ap.add_argument('--off', default='work-pr112i-nofitx-%s')
    ap.add_argument('--arms', nargs='*', default=['uniW0'])
    ap.add_argument('--tsv', default=None)
    a = ap.parse_args()
    root = vio.default_sbnd_root()
    s = a.sample
    base, off = a.base % s, a.off % s
    arms = ['work-pr112i-%s-%s' % (x, s) for x in a.arms]

    rows = []
    for lab in vio.iter_labels(root, TAGS[s]):
        e = int(lab['eventNo'])
        b, o = read(root, base, e), read(root, off, e)
        if b is None or o is None:
            continue
        tr = np.asarray(lab['truth_xyz'], float)
        db = np.linalg.norm(b['xyz'] - tr, axis=1)
        do = np.linalg.norm(o['xyz'] - tr, axis=1)
        tb, to = int(np.argmin(db)), int(np.argmin(do))          # target rows
        r = dict(evt=e,
                 d_target_base=round(float(db[tb]), 3),
                 d_target_off=round(float(do[to]), 3),
                 n_base=len(b['ids']), n_off=len(o['ids']))
        r['R0'] = int(int(np.argmin(np.linalg.norm(b['xyz'] - b['fx'], axis=1))) == tb)
        r['R1'] = int(int(np.argmin(np.linalg.norm(o['xyz'] - o['fx'], axis=1))) == to)
        dsnap = np.linalg.norm(b['xyz'] - o['fx'], axis=1)       # OFF vertex -> base cloud
        r['snap_cm'] = round(float(dsnap.min()), 3)
        r['R2'] = int(int(np.argmin(dsnap)) == tb)
        # is production's cloud even able to express the OFF answer?
        r['target_gap'] = round(float(np.linalg.norm(b['xyz'][tb] - o['xyz'][to])), 3)
        for nm, arm in zip(a.arms, arms):
            x = read(root, arm, e)
            r['R3_' + nm] = (None if x is None else
                             int(int(np.argmin(np.linalg.norm(x['xyz'] - x['fx'], axis=1)))
                                 == int(np.argmin(np.linalg.norm(x['xyz'] - tr, axis=1)))))
        rows.append(r)

    n = len(rows)
    g = lambda k: sum(r[k] for r in rows if r.get(k) is not None)   # noqa: E731
    print('=== sample %s   n=%d   base=%s  off=%s' % (s, n, base, off))
    print('  R0 production (own cloud)            %4d/%d  (%.1f %%)' % (g('R0'), n, 100.0 * g('R0') / n))
    print('  R1 exclusion-free alone (own cloud)  %4d/%d  (%+d vs R0)' % (g('R1'), n, g('R1') - g('R0')))
    print('  R2 transfer ceiling (base cloud)     %4d/%d  (%+d vs R0)' % (g('R2'), n, g('R2') - g('R0')))
    for nm in a.arms:
        k = 'R3_' + nm
        m = sum(1 for r in rows if r.get(k) is not None)
        print('  R3 delivered %-12s            %4d/%d  (%+d vs R0)' % (nm, g(k), m, g(k) - g('R0')))

    # R1 -> R2: which of the exclusion-free chain's WINS survive a transfer?
    win = [r for r in rows if r['R1'] and not r['R0']]
    kept = [r for r in win if r['R2']]
    lost = [r for r in win if not r['R2']]
    brk1 = [r for r in rows if r['R0'] and not r['R1']]
    print('\nR0 -> R1 churn: %d fix / %d break  (net %+d)' % (len(win), len(brk1), len(win) - len(brk1)))
    print('Events the exclusion-free chain gets right and production does not: %d' % len(win))
    print('  transfer KEEPS  %d' % len(kept))
    print('  transfer LOSES  %d' % len(lost))
    if lost:
        gap = np.array([r['target_gap'] for r in lost])
        far = [r for r in lost if r['target_gap'] > 2.0]
        print('    of those, production has NO candidate within 2 cm of the OFF target row: %d/%d'
              ' (candidate-set channel; target_gap median %.2f cm)'
              % (len(far), len(lost), float(np.median(gap))))
        print('    the rest snap to a production row that is not its target: %d' % (len(lost) - len(far)))
    lose_new = [r for r in rows if r['R0'] and not r['R2']]
    print('  transfer BREAKS production on %d events' % len(lose_new))

    # candidate-set quality, both clouds
    db = np.array([r['d_target_base'] for r in rows]); do = np.array([r['d_target_off'] for r in rows])
    print('\nCandidate-set quality (distance from the click to the NEAREST row):')
    print('  base   median %.2f cm ; no row within 2 cm on %d/%d' % (np.median(db), int((db > 2).sum()), n))
    print('  nofitx median %.2f cm ; no row within 2 cm on %d/%d' % (np.median(do), int((do > 2).sum()), n))

    if a.tsv:
        cols = list(rows[0].keys())
        with open(a.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in sorted(rows, key=lambda r: r['evt']):
                fh.write('\t'.join('' if r[c] is None else str(r[c]) for c in cols) + '\n')
        print('\nwrote %s' % a.tsv)


if __name__ == '__main__':
    sys.exit(main())

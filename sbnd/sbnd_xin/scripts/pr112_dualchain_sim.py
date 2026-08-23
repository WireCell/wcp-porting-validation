#!/usr/bin/env python3
'''doc pr/112 sec 5.2 -- the TRUE dual chain, simulated.

Owner correction (2026-08-23): "this dl_vtx_cloud_no_exclusion is not what I
want.  I thought that it only has the last step of fitting?  Note, the entire
PR is different if I keep the no exclusion fit off, multiple steps, right?"

Correct, and it is why sec 5.1's simulation understates idea 2.  m_fit_exclusion
reaches 34 do_multi_tracking call sites across FIVE stages --
NeutrinoStructureExaminer (15), NeutrinoVertexFinder (7), NeutrinoGraphAudit (5),
NeutrinoPatternBase/break_segments (4), NeutrinoOtherSegments (3) -- and those
stages EDIT THE GRAPH (break, merge, add and drop vertices).  So an
exclusion-free chain diverges from the production one at every one of them and
ends with a DIFFERENT graph.  dl_vtx_cloud_no_exclusion, by contrast, does
exactly ONE refit per cluster at DL time on the graph the exclusion chain
already built (NeutrinoVertexFinder.cxx:4796-4809), then restores it.  Last
step only.

This measures the real thing: run the whole PR chain with fit_exclusion=false
(the work-*-harv-nofitx arm IS that chain), take ITS final neutrino vertex, and
transfer that POSITION into the production chain by snapping it to the nearest
production candidate -- exactly "use the no exclusion fit to determine the
neutrino vertex on the exclusion fit".

Scored on the pr/106 target metric defined on the PRODUCTION candidate set,
because that is the chain that ships the answer.  Positions, never ids: the two
chains' graphs differ, so pr75 ids do not correspond (pr/111 sec 2 recorded ids
drifting even between same-graph arms).

Usage: ./pr112_dualchain_sim.py --sample nuecc48 --tsv runs/pr112-dual-nuecc48.tsv
'''
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), 'dl_vtx_training')
sys.path.insert(0, TRAIN)
from scn_vtx import io as vio      # noqa: E402

ON_ARM = 'work-vtx106-harv-base-%s'      # production: fit_exclusion ON everywhere
OFF_ARM = 'work-vtx106-harv-nofitx-%s'   # the WHOLE chain with fit_exclusion=false
TAGS = {'nuecc48': ['vtxscan-harv3-nuecc48'], 'ncpi0': ['vtxscan-harv3-ncpi0'],
        'mcp1k': ['vtxscan-harv3-mcp1k'],
        'mcp2k': ['vtxscan-mcp2k', 'vtxscan-mcp2k-auto', 'vtxscan-mcp2k-ragree']}


def board(sb):
    c = sb.get('hv_cloud')
    if not c or not int(c.get('n_vertex_rows', 0)):
        return None, None
    n = int(c['n_vertex_rows'])
    return (c['vertex_ids'][:n],
            np.array([c['x'][:n], c['y'][:n], c['z'][:n]], float).T)


def final_xyz(j):
    mv = j.get('main_vertex') or {}
    if mv.get('x') is not None:
        return np.array([mv['x'], mv['y'], mv['z']], float)
    sb = j.get('vertex_scoreboard') or {}
    if sb.get('filled'):
        return np.array([sb['final_x'], sb['final_y'], sb['final_z']], float)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='nuecc48')
    ap.add_argument('--on-arm', default=ON_ARM)
    ap.add_argument('--off-arm', default=OFF_ARM)
    ap.add_argument('--tsv', default=None)
    a = ap.parse_args()
    root = vio.default_sbnd_root()
    on_arm, off_arm = a.on_arm % a.sample, a.off_arm % a.sample

    rows = []
    for lab in vio.iter_labels(root, TAGS[a.sample]):
        e = int(lab['eventNo'])
        pon = os.path.join(root, on_arm, 'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
        pof = os.path.join(root, off_arm, 'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
        if not (os.path.exists(pon) and os.path.exists(pof)):
            continue
        jon, jof = vio.load_calib(pon), vio.load_calib(pof)
        ids_on, xyz_on = board(jon.get('vertex_scoreboard') or {})
        ids_of, xyz_of = board(jof.get('vertex_scoreboard') or {})
        if ids_on is None or ids_of is None:
            continue
        tr = np.asarray(lab['truth_xyz'], float)
        d = np.linalg.norm(xyz_on - tr, axis=1)
        target = ids_on[int(np.argmin(d))]

        # what production ships
        p_on = final_xyz(jon)
        # what the exclusion-free CHAIN ships
        p_of = final_xyz(jof)
        if p_on is None or p_of is None:
            continue
        pick_prod = ids_on[int(np.argmin(np.linalg.norm(xyz_on - p_on, axis=1)))]
        dsnap = np.linalg.norm(xyz_on - p_of, axis=1)
        pick_dual = ids_on[int(np.argmin(dsnap))]

        rows.append(dict(
            evt=e, target=target,
            n_on=len(ids_on), n_off=len(ids_of),
            id_overlap=len(set(ids_on) & set(ids_of)),
            hit_prod=int(pick_prod == target), hit_dual=int(pick_dual == target),
            transfer_cm=round(float(dsnap.min()), 3),
            pick_prod=pick_prod, pick_dual=pick_dual))

    n = len(rows)
    print('events %d   sample %s' % (n, a.sample))
    print('\nHOW DIFFERENT ARE THE TWO GRAPHS (candidate sets)?')
    no = np.array([r['n_on'] for r in rows]); nf = np.array([r['n_off'] for r in rows])
    ov = np.array([r['id_overlap'] for r in rows])
    print('  candidates: production median %d, exclusion-free chain median %d'
          % (np.median(no), np.median(nf)))
    print('  same candidate COUNT on %d/%d events' % (int((no == nf).sum()), n))
    print('  vertex-id overlap: median %.0f %% of the production set'
          % (100.0 * np.median(ov / np.maximum(no, 1))))
    print('  events with an IDENTICAL candidate id set: %d/%d'
          % (sum(1 for r in rows if r['id_overlap'] == r['n_on'] == r['n_off']), n))

    print('\nTRANSFER COST (exclusion-free chain vertex -> nearest production candidate):')
    t = np.array([r['transfer_cm'] for r in rows])
    print('  median %.3f cm   p90 %.2f   max %.1f cm ; beyond 2 cm on %d/%d events'
          % (np.median(t), np.percentile(t, 90), t.max(), int((t > 2).sum()), n))

    hp = sum(r['hit_prod'] for r in rows); hd = sum(r['hit_dual'] for r in rows)
    print('\nTARGET-hit on the PRODUCTION candidate set, n=%d' % n)
    print('  production (exclusion ON everywhere)      %d/%d' % (hp, n))
    print('  TRUE dual chain (OFF chain names the vtx) %d/%d   (%+d)' % (hd, n, hd - hp))
    fix = [r['evt'] for r in rows if r['hit_dual'] and not r['hit_prod']]
    brk = [r['evt'] for r in rows if r['hit_prod'] and not r['hit_dual']]
    print('  fixed  : %d  %s' % (len(fix), sorted(fix)))
    print('  broken : %d  %s' % (len(brk), sorted(brk)))
    if a.tsv:
        os.makedirs(os.path.dirname(a.tsv) or '.', exist_ok=True)
        cols = ['evt', 'target', 'n_on', 'n_off', 'id_overlap', 'hit_prod',
                'hit_dual', 'transfer_cm', 'pick_prod', 'pick_dual']
        with open(a.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in sorted(rows, key=lambda r: r['evt']):
                fh.write('\t'.join(str(r[c]) for c in cols) + '\n')
        print('\nwrote %s' % a.tsv)


if __name__ == '__main__':
    sys.exit(main())

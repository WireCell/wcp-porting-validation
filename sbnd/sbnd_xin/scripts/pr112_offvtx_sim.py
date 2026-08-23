#!/usr/bin/env python3
'''doc pr/112 -- simulate the owner's idea #2 offline, exactly.

Owner (2026-08-23): "use the same underlying graph ... we do not need anything
after the nu vtx determination in the no-exclusion fit case.  We only need it
to provide us a position of the vertex, so that we can use it to improve the
final nu vertex identification."

That constraint collapses onto machinery that already exists.
dl_vtx_cloud_no_exclusion (toolkit 14dd031d, DEFAULT OFF) already refits every
cluster exclusion-free through a child TrackFitting sharing the parent graph,
builds the net input from that fit, and restores bit-for-bit -- same graph,
candidate id lists identical to production on 47/47 (pr/106 sec 10).

It differs from the owner's design in exactly one place, which pr/111 sec 11
recorded as found-not-fixed (F4): the cloud's VERTEX ROWS are read at
NeutrinoVertexFinder.cxx:4818 while the no-exclusion refit is LIVE, but every
downstream snap target (:4933, :5054, :5109) is read AFTER the restore at
:4840-4848.  So today the net sees OFF-fit candidate positions and then snaps
to ON-fit ones.  The owner's design is that asymmetry resolved the other way.

Three selections, same graph, same ids, one net:
  A production      net(ON  cloud) -> snap to ON  candidate positions
  B cne knob today  net(OFF cloud) -> snap to ON  candidate positions   (F4)
  C owner's design  net(OFF cloud) -> snap to OFF candidate positions

Ruler = the pr/106 TARGET metric (target = pre-DL candidate nearest the click;
hit = the code PICKS it), which is immune to the fit epoch.  Candidate identity
is carried by pr75 vertex id, never by position, because the two arms' vertex
positions differ by construction -- that displacement is itself reported, since
it is the size of what F4 currently gets wrong.

Usage: ./pr112_offvtx_sim.py --tsv runs/pr112-offvtx-nuecc48.tsv
'''
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), 'dl_vtx_training')
sys.path.insert(0, TRAIN)
sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/toolkit/pyutil/python')
from scn_vtx import io as vio              # noqa: E402
from calib_guard import replay_rerank      # noqa: E402

WEIGHTS = ('/nfs/data/1/xqian/toolkit-dev/wire-cell-data/uboone/scn_vtx/'
           't48k-m16-l5-lr5d-res0.5-CP24.pth')
# Same-graph pair.  work-pr112-harv-* is proven bit-identical to the retired
# work-vtx106-harv-base-* (pr112_repro_gate.py 46/46).
ON_ARM = 'work-pr112-harv-%s'     # exclusion ON, harvest
OFF_ARM = 'work-pr112-cne-%s'     # SAME graph, exclusion-free cloud
TAGS = {'nuecc48': ['vtxscan-harv3-nuecc48'], 'ncpi0': ['vtxscan-harv3-ncpi0'],
        'mcp1k': ['vtxscan-harv3-mcp1k'],
        'mcp2k': ['vtxscan-mcp2k', 'vtxscan-mcp2k-auto', 'vtxscan-mcp2k-ragree']}


def net(c, top_k):
    import SCN_Vertex as sv
    a = [np.array(c[k], np.float32) for k in ('x', 'y', 'z', 'q')]
    raw = sv.SCN_Vertex(WEIGHTS, a[0].tobytes(), a[1].tobytes(), a[2].tobytes(),
                        a[3].tobytes(), dtype='float32', top_k=int(top_k))
    return np.frombuffer(raw, np.float32).reshape(-1, 4)


def cands(sb):
    c = sb['hv_cloud']
    n = int(c['n_vertex_rows'])
    return (c['vertex_ids'][:n],
            np.array([c['x'][:n], c['y'][:n], c['z'][:n]], float).T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='nuecc48')
    ap.add_argument('--on-arm', default=ON_ARM)
    ap.add_argument('--off-arm', default=OFF_ARM)
    ap.add_argument('--tsv', default=None)
    a = ap.parse_args()
    root = vio.default_sbnd_root()
    on_arm, off_arm = a.on_arm % a.sample, a.off_arm % a.sample

    rows, shifts = [], []
    nclos_b = 0
    for lab in vio.iter_labels(root, TAGS[a.sample]):
        e = int(lab['eventNo'])
        pon = os.path.join(root, on_arm, 'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
        pof = os.path.join(root, off_arm, 'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
        if not (os.path.exists(pon) and os.path.exists(pof)):
            continue
        sb_on = (vio.load_calib(pon).get('vertex_scoreboard') or {})
        sb_of = (vio.load_calib(pof).get('vertex_scoreboard') or {})
        if not sb_on.get('hv_cloud') or not sb_of.get('hv_cloud'):
            continue
        ids_on, xyz_on = cands(sb_on)
        ids_of, xyz_of = cands(sb_of)
        if ids_on != ids_of:                 # same graph is the whole premise
            rows.append(dict(evt=e, note='candidate-set-differs')); continue

        tr = np.asarray(lab['truth_xyz'], float)
        d = np.linalg.norm(xyz_on - tr, axis=1)
        target = ids_on[int(np.argmin(d))]
        d_target = float(d.min())

        # how far exclusion moves the candidate positions themselves
        shift = np.linalg.norm(xyz_on - xyz_of, axis=1)
        shifts.append(shift)

        k = int(sb_on.get('dl_top_k') or 5)
        vox_on = net(sb_on['hv_cloud'], k)
        vox_of = net(sb_of['hv_cloud'], k)

        A = replay_rerank(sb_on, vox_on)                 # production
        B = replay_rerank(sb_on, vox_of)                 # cne knob today (F4)
        C = replay_rerank(sb_of, vox_of)                 # owner's design
        if A is None or B is None or C is None:
            continue
        live_on = next((r['vertex_id'] for r in (sb_on.get('rows') or [])
                        if r.get('dl_winner')), None)
        live_of = next((r['vertex_id'] for r in (sb_of.get('rows') or [])
                        if r.get('dl_winner')), None)
        clos_a = int(live_on is None or A['best_vid'] == live_on)
        clos_b = int(live_of is None or B['best_vid'] == live_of)
        nclos_b += (1 - clos_b)
        rows.append(dict(evt=e, target=target, d_target=round(d_target, 3),
                         pickA=A['best_vid'], pickB=B['best_vid'], pickC=C['best_vid'],
                         hitA=int(A['best_vid'] == target), hitB=int(B['best_vid'] == target),
                         hitC=int(C['best_vid'] == target),
                         shift_med=round(float(np.median(shift)), 4),
                         shift_max=round(float(shift.max()), 3),
                         clos_a=clos_a, clos_b=clos_b, note=''))

    ok = [r for r in rows if not r.get('note')]
    n = len(ok)
    print('events %d (candidate-set-differs: %d)'
          % (n, sum(1 for r in rows if r.get('note'))))
    print('CLOSURE  A (production replay == live winner): %d/%d'
          % (sum(r['clos_a'] for r in ok), n))
    print('CLOSURE  B (cne replay == live cne winner)   : %d/%d   [tests F4: if this'
          % (sum(r['clos_b'] for r in ok), n))
    print('          passes, the live knob really does snap on ON positions]')
    allsh = np.concatenate(shifts) if shifts else np.zeros(1)
    print('\ncandidate position shift ON vs OFF fit, all candidates pooled (n=%d):'
          % len(allsh))
    print('  median %.4f cm   p90 %.4f   p99 %.3f   max %.3f cm'
          % (np.median(allsh), np.percentile(allsh, 90),
             np.percentile(allsh, 99), allsh.max()))
    print('\nTARGET-hit, n=%d' % n)
    for k, lab in (('hitA', 'A production          net(ON ) snap ON '),
                   ('hitB', 'B cne knob today      net(OFF) snap ON '),
                   ('hitC', "C owner's design      net(OFF) snap OFF")):
        print('  %s  %d/%d' % (lab, sum(r[k] for r in ok), n))
    fa = sum(r['hitA'] for r in ok)
    print('\n  B - A = %+d      C - A = %+d      C - B = %+d'
          % (sum(r['hitB'] for r in ok) - fa, sum(r['hitC'] for r in ok) - fa,
             sum(r['hitC'] for r in ok) - sum(r['hitB'] for r in ok)))
    fixB = [r['evt'] for r in ok if r['hitB'] and not r['hitA']]
    brkB = [r['evt'] for r in ok if r['hitA'] and not r['hitB']]
    print('\nB - A decomposition (the exclusion-free CLOUD, epoch-immune ruler):')
    print('  ON wrong -> OFF right (fix)  : %d  %s' % (len(fixB), sorted(fixB)))
    print('  ON right -> OFF wrong (break): %d  %s' % (len(brkB), sorted(brkB)))
    print('\nC vs A movers:')
    for r in ok:
        if r['hitC'] != r['hitA']:
            print('   evt %-8s %s  (target %s, A picked %s, C picked %s)'
                  % (r['evt'], 'FIXED ' if r['hitC'] else 'BROKEN',
                     r['target'], r['pickA'], r['pickC']))
    if a.tsv:
        os.makedirs(os.path.dirname(a.tsv) or '.', exist_ok=True)
        cols = ['evt', 'target', 'd_target', 'pickA', 'pickB', 'pickC',
                'hitA', 'hitB', 'hitC', 'shift_med', 'shift_max', 'clos_a', 'clos_b']
        with open(a.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in sorted(ok, key=lambda r: r['evt']):
                fh.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
        print('\nwrote %s' % a.tsv)


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
'''doc pr/112 sec 4.1 -- SUPERSEDED AS AN ACCURACY MEASUREMENT.  Read this first.

This script scores "right" as the net's argmax landing within TOL of the
hand-scan click.  That ruler is INVALID for comparing an exclusion-ON arm
against an exclusion-OFF one.  vtx_target_eval.py's docstring states why: the
labels were clicked on fit_exclusion-OFF reconstructions, so click distance is
biased toward the OFF arm -- pr/106 sec 9 measured the bias on this exact arm
(pr/105 got +135 on the click ruler, "mostly epoch").  Run here it reports
+7 ON-wrong-OFF-right with 0 breaks; the same quantity on the epoch-immune
target metric with candidate ids aligned (pr112_offvtx_sim.py, B - A) is
+2 with 4 fixes and 2 breaks.  Use THAT number.

What this script is still good for: the raw pair-DISAGREEMENT rate and the
median argmax displacement, which are same-quantity comparisons and carry no
epoch bias; and the confidence check re-measuring pr/111 sec 4.  It is kept,
with this warning, because the doc cites those.

Original docstring follows.

doc pr/112 -- the measured CEILING on exclusion-consistency fine-tuning.

Owner's idea #1: "use the O(1000) to fine tune the model so that they can work
with the result in exclusion fit.  Note, for each event we can get both images
with and without exclusion fit."

dl_vtx_training/train.py already carries a `consistency` term (label-free view
agreement, today fed the x4 reflection/jitter views).  The ON/OFF pair is a
drop-in second view, so the build is small.  The question this answers is what
such an objective could BUY, before anyone trains anything.

A consistency objective makes the two views AGREE; it does not make them agree
on the RIGHT answer.  So the honest ceiling is not "how often do they differ"
but the signed quantity

    gain = #(ON wrong AND OFF right)  -  #(ON right AND OFF wrong)

evaluated on the net's own answer, before the selector.  That is what is
printed here, together with the raw disagreement rate (the trainable signal).

Ruler: positions, never ids.  The two arms are DIFFERENT graphs (global
fit_exclusion=false re-fits and re-segments), so pr75 ids do not correspond --
pr/111 sec 2 recorded vertex ids drifting between arms (11002 ON == 11004 OFF,
0.18 cm apart).  "Right" therefore means the net's argmax lands within TOL of
the hand-scan click.

Caveat carried from pr/111 sec 4: there is NO systematic confidence gain from
removing exclusion (voxels[0].dl_score median ON 0.8923 vs OFF 0.8744, roughly
symmetric per event), which is evidence AGAINST the distribution-mismatch
story.  A net fine-tuned on exclusion clouds can be just as chaotic on its new
input distribution; only the consistency form targets the measured mechanism.

Usage: ./pr112_pair.py --tol 2.0 --tsv runs/pr112-pair-nuecc48.tsv
'''
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), 'dl_vtx_training')
sys.path.insert(0, TRAIN)
sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/toolkit/pyutil/python')
from scn_vtx import io as vio      # noqa: E402

WEIGHTS = ('/nfs/data/1/xqian/toolkit-dev/wire-cell-data/uboone/scn_vtx/'
           't48k-m16-l5-lr5d-res0.5-CP24.pth')
ON_ARM = 'work-vtx106-harv-base-nuecc48'
OFF_ARM = 'work-vtx106-harv-nofitx-nuecc48'
TAGS = ['vtxscan-harv3-nuecc48']


def peak(c):
    import SCN_Vertex as sv
    a = [np.array(c[k], np.float32) for k in ('x', 'y', 'z', 'q')]
    raw = sv.SCN_Vertex(WEIGHTS, a[0].tobytes(), a[1].tobytes(), a[2].tobytes(),
                        a[3].tobytes(), dtype='float32', top_k=2)
    v = np.frombuffer(raw, np.float32).reshape(-1, 4).astype(float)
    return v[0, :3], float(v[0, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol', type=float, default=2.0, help='cm, argmax-to-click')
    ap.add_argument('--tsv', default=None)
    a = ap.parse_args()
    root = vio.default_sbnd_root()
    rows = []
    for lab in vio.iter_labels(root, TAGS):
        e = int(lab['eventNo'])
        ps = [os.path.join(root, arm, 'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
              for arm in (ON_ARM, OFF_ARM)]
        if not all(os.path.exists(p) for p in ps):
            continue
        sbs = [(vio.load_calib(p).get('vertex_scoreboard') or {}) for p in ps]
        if not all(s.get('hv_cloud') and s['hv_cloud'].get('x') for s in sbs):
            continue
        tr = np.asarray(lab['truth_xyz'], float)
        (pon, son), (pof, sof) = [peak(s['hv_cloud']) for s in sbs]
        don = float(np.linalg.norm(pon - tr))
        dof = float(np.linalg.norm(pof - tr))
        rows.append(dict(evt=e, d_on=round(don, 3), d_off=round(dof, 3),
                         s_on=round(son, 4), s_off=round(sof, 4),
                         d_pair=round(float(np.linalg.norm(pon - pof)), 3),
                         on_ok=int(don <= a.tol), off_ok=int(dof <= a.tol)))
    n = len(rows)
    dis = [r for r in rows if r['d_pair'] > a.tol]
    fix = [r for r in rows if not r['on_ok'] and r['off_ok']]
    brk = [r for r in rows if r['on_ok'] and not r['off_ok']]
    print('events %d ; ruler: net argmax within %.1f cm of the click\n' % (n, a.tol))
    print('net argmax correct:  exclusion ON %d/%d      exclusion OFF %d/%d'
          % (sum(r['on_ok'] for r in rows), n, sum(r['off_ok'] for r in rows), n))
    print('pair disagreement (argmax moves > %.1f cm): %d/%d = %.0f %%   [the trainable signal]'
          % (a.tol, len(dis), n, 100.0 * len(dis) / max(n, 1)))
    print('  median |argmax_ON - argmax_OFF| = %.3f cm'
          % np.median([r['d_pair'] for r in rows]))
    print('\nCEILING on making ON behave like OFF:')
    print('  ON wrong & OFF right (recoverable) : %d   %s'
          % (len(fix), sorted(r['evt'] for r in fix)))
    print('  ON right & OFF wrong (would break) : %d   %s'
          % (len(brk), sorted(r['evt'] for r in brk)))
    print('  net gain                           : %+d / %d' % (len(fix) - len(brk), n))
    print('\nconfidence at the peak (pr/111 sec 4 re-measured on all %d):' % n)
    print('  median dl_score  ON %.4f   OFF %.4f'
          % (np.median([r['s_on'] for r in rows]), np.median([r['s_off'] for r in rows])))
    if a.tsv:
        os.makedirs(os.path.dirname(a.tsv) or '.', exist_ok=True)
        cols = ['evt', 'd_on', 'd_off', 's_on', 's_off', 'd_pair', 'on_ok', 'off_ok']
        with open(a.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in sorted(rows, key=lambda r: r['evt']):
                fh.write('\t'.join(str(r[c]) for c in cols) + '\n')
        print('\nwrote %s' % a.tsv)


if __name__ == '__main__':
    sys.exit(main())

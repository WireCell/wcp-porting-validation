#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73 round 2: did F3a fix 18255-57903?

The falsifiable PASS statement for the ISO case, measured per arm so it can be
re-run on any of them.  Everything is read off the Bee `track_fit-global` and
`clustering-global` layers of the arm's own mabc-pr.zip -- read-only.

Four primary numbers, all in the FIXED window z in [265.90, 314.03] cm (the
pre-round-5 segment's own extent, sec 4.3 -- fixing the window is what makes
the arms comparable, since they cut the corridor into segments at different
places and a per-segment ratio can inflate purely by absorbing a curved piece
from a neighbour):

                       pre-R5 (good)   R7 production      PASS
    path/chord             1.026       1.326 & 1.252     <= 1.05
    bow amplitude          1.90 cm     2.52 & 2.94 cm    <= 2.0 cm
    jitter rms             0.081 cm    0.342 & 0.273 cm  <= 0.10 cm
    segments in window     1 (83 pts)  2                 1, >= 80 pts

The segment-count clause is not cosmetic: production needs TWO rows, so
per-row ratios can look acceptable while the 27/69 deg hairpin survives.

bow/jitter come from zigzag_anatomy.bow_jitter at DEGREE 4.  Sec 7 measured
the split to be degree-dependent below 4 -- a quadratic cannot represent the
excursion at all and dumps it into "jitter", which would silently pass a
broken arm.  Degrees >= 4 agree.

DO-NO-HARM FLOOR.  Sec 4.3 found charge coverage flat across every arm
(44.2-44.9 %), so coverage cannot DEMONSTRATE success -- but sec 4.2 found the
straight chord captures LESS charge than the zigzag (23.1 % vs 36.6 % within
3 cm), because the ridge sits 3-5 cm off it.  So a revert to the base route
could improve every smoothness number while losing charge.  Hence, on the
fixed changed-region z in [249.64, 314.23] cm:

    charge within 1.5 cm of any fit point  >=  30.28 %  (pre-R5; prod 31.00 %)

NB on that 30.28.  Sec 4.3 prints the pre-round-5 coverage rounded, as
"30.3 %"; the underlying value is 30.2868 %.  Taking the bar from the rounded
figure makes the REFERENCE arm fail its own floor by 0.013 points, so the bar
is set from the measured value instead.  This is a transcription fix, not a
relaxation: the floor still says "do not lose charge relative to the arm the
doc names as correct".

SECONDARY.  Max run of q_bee == 0 in the corridor: 11 -> <= 2.  q_bee == 0
means dQ <= 1e4 e over a 0.6 cm step (sec 4.1), i.e. a real dQ/dx deficit --
sec 4.4 found eleven consecutive such points (i = 42..52, 6.6 cm) where the
trajectory walks off the ridge.

Usage:  f3a_57903_check.py <ARM> [ARM ...] [--evt 57903]
"""
import sys
import os
import json
import zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

# the fixed windows, sec 4.3 / sec 4.3-coverage.  Never derive these from the
# arm under test -- that is what makes the comparison like-for-like.
ZLO, ZHI = 265.90, 314.03          # the pre-round-5 segment's extent
CLO, CHI = 249.64, 314.23          # union of round-6 segs 14001+14007
CID = 14                           # the cluster carrying the corridor
DEG = 4

BAR = dict(ratio=1.05, bow=2.0, jit=0.10, nseg=1, npts=80, cover=30.28, q0run=2)
REF = ('pre-R5 good: ratio 1.026  bow 1.90  jit 0.081  1 seg / 83 pts | '
       'R7 prod: ratio 1.326 & 1.252  bow 2.52 & 2.94  jit 0.342 & 0.273  2 segs')


def bow_jitter(S, deg):
    """Verbatim from zigzag_anatomy.py -- the sec 4.7 bow/jitter split."""
    chord = np.linalg.norm(S[0] - S[-1])
    u = (S[-1] - S[0]) / chord
    t = (S - S[0]) @ u
    ex = np.array([1.0, 0.0, 0.0])
    e1 = ex - np.dot(ex, u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    perp = (S - S[0]) - np.outer(t, u)
    SM = S[0] + np.outer(t, u)
    res = np.zeros_like(S)
    for e in (e1, e2):
        c = perp @ e
        f = np.polyval(np.polyfit(t, c, deg), t)
        SM = SM + np.outer(f, e)
        res = res + np.outer(c - f, e)
    path = np.linalg.norm(np.diff(S, axis=0), axis=1).sum()
    bow = float(np.abs(np.linalg.norm(SM - (S[0] + np.outer(t, u)), axis=1)).max())
    return dict(chord=float(chord), ratio=float(path / chord), bow=bow,
                jit_rms=float(np.linalg.norm(res, axis=1).std()))


def layers(arm, evt):
    zp = os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip')
    z = zipfile.ZipFile(zp)
    fd = json.loads(z.read('data/0/0-track_fit-global.json'))
    im = json.loads(z.read('data/0/0-clustering-global.json'))
    F = (np.array([fd['x'], fd['y'], fd['z']]).T,
         np.array(fd['real_cluster_id']),
         np.array(fd['q'], dtype=float))
    I = (np.array([im['x'], im['y'], im['z']]).T,
         np.clip(np.array(im['q'], dtype=float), 0, None),
         np.array(im['cluster_id']))
    return F, I


def maxrun(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def check(arm, evt):
    (P, R, Q), (IP, IQ, IC) = layers(arm, evt)
    print('\n' + '=' * 78)
    print('ARM %s   event %d' % (arm, evt))
    print('=' * 78)

    # ---- primary: segments spanning the fixed window ------------------------
    rows, worst = [], None
    for s in sorted(set(R[R >= 0].tolist())):
        m = (R == s)
        S = P[m]
        inw = (S[:, 2] >= ZLO) & (S[:, 2] <= ZHI)
        if inw.sum() < 10:
            continue
        W = S[inw]
        bj = bow_jitter(W, DEG)
        q0 = maxrun(Q[m][inw] == 0)
        rows.append((s, int(inw.sum()), bj, q0))

    print('\n-- primary: corridor in the FIXED window z in [%.2f, %.2f] cm --'
          % (ZLO, ZHI))
    print('   %-8s %6s %9s %8s %9s %9s %8s' %
          ('seg', 'npts', 'chord', 'ratio', 'bow cm', 'jit rms', 'q0 run'))
    for s, n, bj, q0 in rows:
        print('   %-8d %6d %9.2f %8.3f %9.2f %9.3f %8d'
              % (s, n, bj['chord'], bj['ratio'], bj['bow'], bj['jit_rms'], q0))
    if not rows:
        print('   NO segment has >=10 points in the window -- cannot evaluate')
        return None

    nseg = len(rows)
    npts = max(r[1] for r in rows)
    ratio = max(r[2]['ratio'] for r in rows)
    bow = max(r[2]['bow'] for r in rows)
    jit = max(r[2]['jit_rms'] for r in rows)
    q0run = max(r[3] for r in rows)

    # ---- do-no-harm: charge coverage on the fixed changed region ------------
    mreg = (IC == CID) & (IP[:, 2] >= CLO) & (IP[:, 2] <= CHI)
    REG, REGQ = IP[mreg], IQ[mreg]
    fit = P[R >= 0]
    cover = {}
    for rad in (1.5, 3.0):
        if len(REG) and len(fit):
            d = np.sqrt(((REG[:, None, :] - fit[None, :, :]) ** 2).sum(-1)).min(1)
            cover[rad] = 100.0 * REGQ[d <= rad].sum() / REGQ.sum()
        else:
            cover[rad] = float('nan')
    print('\n-- do-no-harm: charge coverage, FIXED region z in [%.2f, %.2f] --'
          % (CLO, CHI))
    print('   %d image points, %.3e e' % (len(REG), REGQ.sum()))
    print('   within 1.5 cm of any fit point: %5.1f %%   (bar >= %.1f)'
          % (cover[1.5], BAR['cover']))
    print('   within 3.0 cm of any fit point: %5.1f %%' % cover[3.0])

    # ---- verdict ------------------------------------------------------------
    tests = [
        ('path/chord   <= %.2f' % BAR['ratio'], ratio, ratio <= BAR['ratio']),
        ('bow amp cm   <= %.1f' % BAR['bow'], bow, bow <= BAR['bow']),
        ('jitter rms   <= %.2f' % BAR['jit'], jit, jit <= BAR['jit']),
        ('segments     == %d' % BAR['nseg'], nseg, nseg == BAR['nseg']),
        ('max npts     >= %d' % BAR['npts'], npts, npts >= BAR['npts']),
        ('coverage %%    >= %.1f' % BAR['cover'], cover[1.5], cover[1.5] >= BAR['cover']),
        ('q0 run       <= %d' % BAR['q0run'], q0run, q0run <= BAR['q0run']),
    ]
    print('\n-- verdict --')
    allok = True
    for name, val, ok in tests:
        allok &= ok
        print('   %-22s  measured %8.3f   %s' % (name, val, 'PASS' if ok else 'FAIL'))
    print('\n   reference:  %s' % REF)
    print('   OVERALL: %s' % ('PASS' if allok else 'FAIL'))
    return allok


def main():
    argv = sys.argv[1:]
    evt, arms = 57903, []
    i = 0
    while i < len(argv):
        if argv[i] == '--evt':
            evt = int(argv[i + 1]); i += 2; continue
        if argv[i].startswith('--'):
            sys.exit('unknown flag %s\n\n%s' % (argv[i], __doc__))
        arms.append(argv[i]); i += 1
    if not arms:
        sys.exit(__doc__)
    for a in arms:
        check(a, evt)


if __name__ == '__main__':
    main()

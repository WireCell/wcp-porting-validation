#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73 sec 4.11: is the jitter IN THE SEED, or added by the fit?

Section 2 claims the trajectory fit never creates the zigzag, only fails to
remove it.  Section 4.10 proved that for 18255-57903's BOW -- the router
delivers it.  It was never checked for the small-amplitude JITTER that dominates
18255-53427 and 18255-54351 (sec 4.7), and this script checks it.

Compares, for the segment carrying each owner point:

  seed  = the do_rough_path route, dumped by the `sgp path sel:` sentinel
          (knob sgp_edge_probe); these are steiner-cloud points, so they are
          NOT uniformly spaced -- everything below is computed after resampling
          the seed polyline at the same 0.6 cm step the fit uses, which is what
          organize_segments_path_3rd does before fitting.
  fit   = Segment::fits() from the Bee track_fit-global layer.

and reports for each the jitter about its own degree-4 trend (the sec 4.7
split), plus how far the fitted points sit from the seed polyline.

If the seed is smooth and the fit is not, the fit ADDS the jitter and sec 2's
blanket statement needs qualifying.

The owner asked about the LOCAL jitter at his two points, not the long-track
curvature, so everything is reported twice: over the whole segment, and over a
window of +-WIN fitted points (default 25 = +-15 cm) centred on his point.

Read-only.  Usage:  seed_vs_fit.py <ARM> <EVT> <X> <Y> <Z> [--win N]
"""
import sys
import re
import json
import zipfile
import numpy as np

RXS = re.compile(r"sgp path sel: cluster (\d+) k=(\d+) idx=(\d+) "
                 r"\(([-\d.]+),([-\d.]+),([-\d.]+)\)")
DEG = 4

WIN = 25
argv = sys.argv[1:]
if '--win' in argv:
    i = argv.index('--win'); WIN = int(argv[i + 1]); del argv[i:i + 2]
if len(argv) < 5:
    sys.exit(__doc__)
arm, evt = argv[0], int(argv[1])
TGT = np.array([float(x) for x in argv[2:5]])


def seeds(path):
    """All dumped routes, in emission order, as (cluster, Nx3)."""
    out, cur, ccur = [], [], None
    for line in open(path, errors='replace'):
        m = RXS.search(line)
        if not m:
            continue
        c, k = int(m.group(1)), int(m.group(2))
        if k == 0 and cur:
            out.append((ccur, np.array(cur)))
            cur = []
        ccur = c
        cur.append([float(m.group(4)), float(m.group(5)), float(m.group(6))])
    if cur:
        out.append((ccur, np.array(cur)))
    return out


def resample(P, step=0.6):
    """Arc-length resample a polyline at a uniform step."""
    d = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
    if d[-1] < step:
        return P.copy()
    t = np.arange(0.0, d[-1], step)
    return np.stack([np.interp(t, d, P[:, k]) for k in range(3)], axis=1)


def split(S):
    """sec 4.7 bow/jitter split of a polyline."""
    chord = np.linalg.norm(S[0] - S[-1])
    u = (S[-1] - S[0]) / chord
    t = (S - S[0]) @ u
    ex = np.array([1.0, 0.0, 0.0])
    e1 = ex - np.dot(ex, u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    perp = (S - S[0]) - np.outer(t, u)
    res = np.zeros_like(S)
    SM = S[0] + np.outer(t, u)
    for e in (e1, e2):
        c = perp @ e
        f = np.polyval(np.polyfit(t, c, min(DEG, max(1, len(S) - 2))), t)
        SM = SM + np.outer(f, e)
        res = res + np.outer(c - f, e)
    path = np.linalg.norm(np.diff(S, axis=0), axis=1).sum()
    pb = np.linalg.norm(np.diff(SM, axis=0), axis=1).sum()
    st = np.diff(S, axis=0)
    n = np.linalg.norm(st, axis=1)
    ok = n > 1e-9
    tt = st[ok] / n[ok, None]
    ang = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', tt[:-1], tt[1:]), -1, 1)))
    return dict(n=len(S), chord=chord, ratio=path / chord,
                ratio_bow=pb / np.linalg.norm(SM[0] - SM[-1]), ratio_jit=path / pb,
                jit_rms=float(np.linalg.norm(res, axis=1).std()),
                jit_max=float(np.linalg.norm(res, axis=1).max()),
                fold=float((ang > 30).mean()) if len(ang) else 0.0,
                ang_max=float(ang.max()) if len(ang) else 0.0)


# --- the fitted segment carrying the owner point
z = zipfile.ZipFile('%s/pr_evt%d/mabc-pr.zip' % (arm, evt))
d = json.loads(z.read('data/0/0-track_fit-global.json'))
P = np.array([d['x'], d['y'], d['z']]).T
R = np.array(d['real_cluster_id'])
best = None
for s in sorted(set(R[R >= 0].tolist())):
    m = R == s
    if m.sum() < 10:
        continue
    dd = float(np.linalg.norm(P[m] - TGT, axis=1).min())
    if best is None or dd < best[0]:
        best = (dd, int(s), P[m])
_, seg, FIT = best

# --- the dumped route whose endpoints best match that segment
cands = seeds('%s/pr_evt%d/wct_pr_evt%d.log' % (arm, evt, evt))
score = lambda S: (min(np.linalg.norm(S[0] - FIT[0]), np.linalg.norm(S[0] - FIT[-1]))
                   + min(np.linalg.norm(S[-1] - FIT[0]), np.linalg.norm(S[-1] - FIT[-1])))
cands = [(c, S) for c, S in cands if len(S) >= 4]
if not cands:
    sys.exit('no dumped routes in the log -- was sgp_edge_probe on?')
cl, SEED = min(cands, key=lambda cs: score(cs[1]))
if np.linalg.norm(SEED[0] - FIT[-1]) < np.linalg.norm(SEED[0] - FIT[0]):
    SEED = SEED[::-1]

print('event %d, arm %s' % (evt, arm))
print('  fitted segment %d: %d points, chord %.2f cm' % (seg, len(FIT), np.linalg.norm(FIT[0] - FIT[-1])))
print('  matched rough path: cluster %d, %d steiner points, endpoint mismatch %.2f cm'
      % (cl, len(SEED), score(SEED)))
step = np.linalg.norm(np.diff(SEED, axis=0), axis=1)
print('  seed step length: min %.3f median %.3f max %.3f cm (steiner spacing, NOT uniform)'
      % (step.min(), np.median(step), step.max()))
print()

SEED_R = resample(SEED)
print('%-34s %5s %8s %8s %10s %10s %9s %9s %8s'
      % ('', 'n', 'chord', 'ratio', 'ratio_bow', 'ratio_jit', 'jit_rms', 'jit_max', 'maxturn'))
for lab, S in (('seed, as routed (steiner pts)', SEED),
               ('seed, resampled at 0.6 cm', SEED_R),
               ('FITTED trajectory (0.6 cm)', FIT)):
    r = split(S)
    print('%-34s %5d %8.2f %8.3f %10.3f %10.3f %9.3f %9.3f %8.0f'
          % (lab, r['n'], r['chord'], r['ratio'], r['ratio_bow'], r['ratio_jit'],
             r['jit_rms'], r['jit_max'], r['ang_max']))

# how far do the fitted points sit from the seed polyline?
def dist_to_poly(pts, poly):
    A, B = poly[:-1], poly[1:]
    AB = B - A
    L2 = np.einsum('ij,ij->i', AB, AB)
    L2[L2 == 0] = 1e-12
    out = []
    for p in pts:
        t = np.clip(np.einsum('ij,ij->i', p - A, AB) / L2, 0, 1)
        proj = A + t[:, None] * AB
        out.append(np.linalg.norm(proj - p, axis=1).min())
    return np.array(out)


dfs = dist_to_poly(FIT, SEED)
# ---------------------------------------------------------------- local window
j = int(np.linalg.norm(FIT - TGT, axis=1).argmin())
a, b = max(0, j - WIN), min(len(FIT), j + WIN + 1)
FIT_L = FIT[a:b]
# clip the seed to the same arc span, with a small margin
lo = FIT_L[0]
hi = FIT_L[-1]
dl = np.linalg.norm(SEED - lo, axis=1).argmin()
dh = np.linalg.norm(SEED - hi, axis=1).argmin()
i0, i1 = (dl, dh) if dl <= dh else (dh, dl)
SEED_L = SEED[max(0, i0 - 1):min(len(SEED), i1 + 2)]
print()
print('LOCAL window: fitted points %d..%d (+-%d, %.1f cm of track) around the owner point'
      % (a, b - 1, WIN, np.linalg.norm(FIT_L[0] - FIT_L[-1])))
if len(SEED_L) >= 4:
    print('%-34s %5s %8s %8s %10s %10s %9s %9s %8s'
          % ('', 'n', 'chord', 'ratio', 'ratio_bow', 'ratio_jit', 'jit_rms', 'jit_max', 'maxturn'))
    for lab, S in (('seed (steiner pts), local', SEED_L),
                   ('seed resampled 0.6 cm, local', resample(SEED_L)),
                   ('FITTED trajectory, local', FIT_L)):
        r = split(S)
        print('%-34s %5d %8.2f %8.3f %10.3f %10.3f %9.3f %9.3f %8.0f'
              % (lab, r['n'], r['chord'], r['ratio'], r['ratio_bow'], r['ratio_jit'],
                 r['jit_rms'], r['jit_max'], r['ang_max']))
else:
    print('  (only %d steiner points span this window -- the seed is coarser than'
          ' the window)' % len(SEED_L))

print()
print('  fitted points vs the seed polyline: median %.3f cm, p90 %.3f, max %.3f'
      % (np.median(dfs), np.percentile(dfs, 90), dfs.max()))
print('  (the fit is allowed to move off the seed; what matters is whether the')
print('   SEED is already jittery, i.e. compare jit_rms of rows 2 and 3 above)')

#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73: how zigzagged is a fitted trajectory, and does it
track isochronicity?

For every PR segment in one or more arms, measured on the Bee `track_fit-global`
layer (= Segment::fits(), the polyline whose local_dx feeds dQ_dx_multi_fit):

    chord      |first - last|
    path       sum of the 0.6 cm steps
    ratio      path / chord           <- the zigzag measure
    iso        angle of the chord OUT of the drift-perpendicular plane, deg
               (0 = isochronous, i.e. the whole segment at one drift time)
    fold       fraction of interior points whose turn angle between the two
               adjacent 0.6 cm steps exceeds --fold-cut (default 30 deg)
    rms_dr     rms of the transverse residual about the chord, along the
               drift direction
    rms_iso    ... and along the remaining (in-plane) transverse direction
    q0         fraction of points with q_bee == 0

`q_bee` is `fit.dQ * dQdx_scale + dQdx_offset` clipped at 0
(MultiAlgBlobClustering.cxx:955-957).  SBND uses 0.1 / -1000
(cfg/pgrapher/experiment/sbnd/clus.jsonnet:1768), so q_bee == 0 means
dQ <= 1e4 electrons over that 0.6 cm step -- a real dQ/dx deficit, NOT the
absence of charge.

Read-only: consumes each arm's existing mabc-pr.zip, runs nothing.

Usage:
    zigzag_census.py <ARM> [ARM ...] [--tsv OUT.tsv] [--min-pts N]
                     [--min-chord CM] [--fold-cut DEG] [--top N]
"""
import sys
import os
import json
import glob
import zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'

opt = dict(tsv=None, min_pts=10, min_chord=5.0, fold_cut=30.0, top=12)
arms = []
argv = sys.argv[1:]
i = 0
while i < len(argv):
    a = argv[i]
    if a == '--tsv':
        opt['tsv'] = argv[i + 1]; i += 2; continue
    if a == '--min-pts':
        opt['min_pts'] = int(argv[i + 1]); i += 2; continue
    if a == '--min-chord':
        opt['min_chord'] = float(argv[i + 1]); i += 2; continue
    if a == '--fold-cut':
        opt['fold_cut'] = float(argv[i + 1]); i += 2; continue
    if a == '--top':
        opt['top'] = int(argv[i + 1]); i += 2; continue
    if a.startswith('--'):
        sys.exit('unknown flag %s\n\n%s' % (a, __doc__))
    arms.append(a); i += 1
if not arms:
    sys.exit(__doc__)


def segments(arm):
    """Yield one dict per qualifying segment of one arm."""
    for d in sorted(glob.glob(os.path.join(SB, arm, 'pr_evt*'))):
        evt = int(os.path.basename(d)[6:])
        zp = os.path.join(d, 'mabc-pr.zip')
        if not os.path.exists(zp):
            continue
        try:
            z = zipfile.ZipFile(zp)
            fd = json.loads(z.read('data/0/0-track_fit-global.json'))
        except Exception as e:
            print('  [skip] %s evt %d: %s' % (arm, evt, e), file=sys.stderr)
            continue
        P = np.array([fd['x'], fd['y'], fd['z']]).T
        R = np.array(fd['real_cluster_id'])
        Q = np.array(fd['q'], dtype=float)
        for s in sorted(set(R[R >= 0].tolist())):
            m = R == s
            S = P[m]
            if len(S) < opt['min_pts']:
                continue
            chord = float(np.linalg.norm(S[0] - S[-1]))
            if chord < opt['min_chord']:
                continue
            step = np.linalg.norm(np.diff(S, axis=0), axis=1)
            path = float(step.sum())
            u = (S[-1] - S[0]) / chord
            # transverse residual about the chord, split into the drift
            # direction and the remaining in-plane direction
            r = S - S[0]
            perp = r - np.outer(r @ u, u)
            ex = np.array([1.0, 0.0, 0.0])
            e_dr = ex - np.dot(ex, u) * u
            e_dr /= np.linalg.norm(e_dr)
            e_iso = np.cross(u, e_dr)
            # turn angle between consecutive steps
            ok = step > 1e-6
            t = np.diff(S, axis=0)[ok] / step[ok, None]
            ang = np.degrees(np.arccos(
                np.clip(np.einsum('ij,ij->i', t[:-1], t[1:]), -1.0, 1.0)))
            yield dict(arm=arm, evt=evt, seg=int(s), n=int(m.sum()),
                       chord=chord, path=path, ratio=path / chord,
                       iso=float(np.degrees(np.arcsin(min(1.0, abs(u[0]))))),
                       fold=float((ang > opt['fold_cut']).mean()) if len(ang) else 0.0,
                       rms_dr=float((perp @ e_dr).std()),
                       rms_iso=float((perp @ e_iso).std()),
                       max_iso=float(np.abs(perp @ e_iso).max()),
                       q0=float((Q[m] == 0).mean()))


rows = [r for arm in arms for r in segments(arm)]
if not rows:
    sys.exit('no segments passed the selection')

hdr = ['arm', 'evt', 'seg', 'n', 'chord', 'path', 'ratio', 'iso', 'fold',
       'rms_dr', 'rms_iso', 'max_iso', 'q0']
if opt['tsv']:
    with open(opt['tsv'], 'w') as fh:
        fh.write('\t'.join(hdr) + '\n')
        for r in rows:
            fh.write('%s\t%d\t%d\t%d\t%.3f\t%.3f\t%.4f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n'
                     % tuple(r[k] for k in hdr))
    print('wrote', opt['tsv'])

print('arms: %s' % ', '.join(arms))
print('segments with >= %d points and chord >= %.0f cm: %d'
      % (opt['min_pts'], opt['min_chord'], len(rows)))
print()

# --- the two-bin split.  Deliberately two bins: a finer binning produced
# --- cells of n=4 that cannot support a trend.
print('%-16s %5s %10s %10s %11s %13s'
      % ('bin', 'n', 'med ratio', 'p90 ratio', 'med fold%', 'med q_bee=0%'))
for lab, sel in [('iso < 10 deg', lambda r: r['iso'] < 10.0),
                 ('iso >= 10 deg', lambda r: r['iso'] >= 10.0)]:
    ss = [r for r in rows if sel(r)]
    if not ss:
        continue
    print('%-16s %5d %10.3f %10.3f %10.1f%% %12.1f%%'
          % (lab, len(ss),
             np.median([r['ratio'] for r in ss]),
             np.percentile([r['ratio'] for r in ss], 90),
             100 * np.median([r['fold'] for r in ss]),
             100 * np.median([r['q0'] for r in ss])))
print()


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


fold = [r['fold'] for r in rows]
rat = [r['ratio'] for r in rows]
iso = [-r['iso'] for r in rows]          # negated: larger = more isochronous
q0 = [r['q0'] for r in rows]
print('Spearman rank correlation with the fraction of q_bee == 0 points:')
print('   fold fraction (turn > %.0f deg) : %+.2f' % (opt['fold_cut'], spearman(fold, q0)))
print('   path/chord ratio               : %+.2f' % spearman(rat, q0))
print('   isochronicity (90 - |angle|)   : %+.2f' % spearman(iso, q0))
print('Spearman with the path/chord ratio: fold %+.2f, isochronicity %+.2f'
      % (spearman(fold, rat), spearman(iso, rat)))
print()

print('worst %d segments by fold fraction:' % opt['top'])
print('%-22s %8s %7s %5s %8s %7s %7s %7s %7s'
      % ('arm', 'event', 'seg', 'n', 'chord', 'ratio', 'isoDeg', 'fold%', 'q0%'))
for r in sorted(rows, key=lambda r: -r['fold'])[:opt['top']]:
    print('%-22s %8d %7d %5d %8.2f %7.3f %7.1f %6.1f%% %6.1f%%'
          % (r['arm'], r['evt'], r['seg'], r['n'], r['chord'], r['ratio'],
             r['iso'], 100 * r['fold'], 100 * r['q0']))
print()
print('worst %d segments by path/chord ratio:' % opt['top'])
print('%-22s %8s %7s %5s %8s %7s %7s %7s %7s'
      % ('arm', 'event', 'seg', 'n', 'chord', 'ratio', 'isoDeg', 'fold%', 'q0%'))
for r in sorted(rows, key=lambda r: -r['ratio'])[:opt['top']]:
    print('%-22s %8d %7d %5d %8.2f %7.3f %7.1f %6.1f%% %6.1f%%'
          % (r['arm'], r['evt'], r['seg'], r['n'], r['chord'], r['ratio'],
             r['iso'], 100 * r['fold'], 100 * r['q0']))

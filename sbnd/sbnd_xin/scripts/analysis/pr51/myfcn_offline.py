#!/usr/bin/env python3
"""Offline reproduction of MyFCN::AddSegment + FitVertex for one PR vertex.

Reads an arm's mabc-pr.zip -- track_fit-global (= Segment::fits(), the layer
MyFCN actually reads; shower_track-global is associate_points and must NOT be
used here) plus vertices-global q=15000 = main vertex -- and redoes
the vertex fit exactly as clus/src/MyFCN.cxx does, to expose WHICH directions
the fit actually constrains.

NOTE: the trajectories in the zip are the FINAL ones (post the last
do_multi_tracking), not the ones present at fit_vertex time.  This is a
characterization of the fit's stiffness at the converged vertex, not a
bit-level replay.

Usage: myfcn_offline.py <ARM> <EVT> [rcid ...]   (LAYER env overrides the layer)
doc sbnd_xin/docs/pr/51, round-5 addendum follow-up.
"""
import sys, os, json, zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
arm, evt = sys.argv[1], int(sys.argv[2])
want = [int(x) for x in sys.argv[3:]] or None

z = zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip'))
load = lambda n: json.loads(z.read('data/0/0-%s.json' % n))

vtx = load('vertices-global')
st = load(os.environ.get('LAYER','track_fit-global'))

vp = np.array([vtx['x'], vtx['y'], vtx['z']]).T
main = vp[np.array(vtx['q']) == 15000]
assert len(main) == 1, main
V = main[0]
print('main vertex (cm): (%.3f, %.3f, %.3f)' % tuple(V))

sp = np.array([st['x'], st['y'], st['z']]).T
rc = np.array(st['real_cluster_id'])

# MyFCN constructor from fit_vertex(): flag_vtx_constraint=true,
# vtx_constraint_range=0.43, vertex_protect_dis=1.5,
# vertex_protect_dis_short_track=0.9, fit_dis=6.0   (all cm)
RANGE, PROT, PROT_SHORT, FITDIS = 0.43, 1.5, 0.9, 6.0
FLOOR = 0.15 ** 2  # cm^2, added to every PCA eigenvalue

if want is None:
    d = np.linalg.norm(sp - V, axis=1)
    want = sorted(set(rc[d < FITDIS].tolist()))
print('legs:', want)

A = np.zeros((3, 3))
b = np.zeros(3)
npoints = 0
info = []
for r in want:
    pts = sp[rc == r]
    # AddSegment's "length" is |front - back| of the WHOLE segment
    length = np.linalg.norm(pts[0] - pts[-1])
    cut = PROT if length > 3.0 else PROT_SHORT
    d = np.linalg.norm(pts - V, axis=1)
    sel = pts[(d >= cut) & (d <= FITDIS)]
    if len(sel) < 2:
        info.append((r, length, len(sel), None, None, None))
        continue
    # center = the kept point closest to the vertex
    dsel = np.linalg.norm(sel - V, axis=1)
    center = sel[np.argmin(dsel)]
    cov = np.zeros((3, 3))
    for p in sel:
        dd = p - center
        cov += np.outer(dd, dd)
    cov /= len(sel)
    w, vecs = np.linalg.eigh(cov)          # ascending
    lam = w[::-1] + FLOOR                  # descending + floor
    ax = vecs[:, ::-1].T                   # rows: leading, mid, minor
    npoints += len(sel)
    R = np.zeros((3, 3))                   # row 0 zeroed by construction
    R[1] = np.sqrt(lam[0] / lam[1]) * ax[1]
    R[2] = np.sqrt(lam[0] / lam[2]) * ax[2]
    A += R.T @ R
    b += R.T @ (R @ center)
    info.append((r, length, len(sel), lam, ax, center))

print()
print('%-8s %8s %6s %10s %10s %10s %9s %9s' %
      ('rcid', 'len_cm', 'npts', 'lam0', 'lam1', 'lam2', 'w1=sq(l0/l1)', 'w2'))
dirs = {}
for r, length, n, lam, ax, c in info:
    if lam is None:
        print('%-8d %8.2f %6d   (<2 points in annulus -> NOT fittable)' % (r, length, n))
        continue
    dirs[r] = ax[0]
    print('%-8d %8.2f %6d %10.4f %10.4f %10.4f %9.2f %9.2f' %
          (r, length, n, lam[0], lam[1], lam[2],
           np.sqrt(lam[0] / lam[1]), np.sqrt(lam[0] / lam[2])))

ks = list(dirs)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        c = np.clip(np.dot(dirs[ks[i]], dirs[ks[j]]), -1, 1)
        print('opening angle %d vs %d : %.1f deg  (gate: >15)' %
              (ks[i], ks[j], np.degrees(np.arccos(c))))

print()
print('npoints (sum over fittable legs) =', npoints)
ew, ev = np.linalg.eigh(A)
print('A (no prior) eigenvalues [cm^-2]:', np.array2string(ew, precision=2))
print('  softest direction:', np.array2string(ev[:, 0], precision=3),
      ' stiffness %.3f' % ew[0])
print('  stiffest direction:', np.array2string(ev[:, 2], precision=3),
      ' stiffness %.3f' % ew[2])

prior = npoints / RANGE ** 2
print('prior stiffness (npoints/range^2) = %.2f cm^-2  (isotropic)' % prior)
print('  -> prior/softest = %.1f x ,  prior/stiffest = %.2f x' %
      (prior / ew[0] if ew[0] > 0 else float('inf'), prior / ew[2]))

Ac = A + np.eye(3) * prior
bc = b + V * prior
sol = np.linalg.solve(Ac, bc)
print()
print('solution WITH prior : (%.3f, %.3f, %.3f)  |move| = %.3f cm'
      % (*sol, np.linalg.norm(sol - V)))
sol0 = np.linalg.solve(A, b)
print('solution NO prior   : (%.3f, %.3f, %.3f)  |move| = %.3f cm'
      % (*sol0, np.linalg.norm(sol0 - V)))
mv = sol0 - V
if len(ks) >= 1:
    t = dirs[ks[0]]
    print('  no-prior move projected on leg %d axis: %.3f cm  (transverse %.3f cm)'
          % (ks[0], np.dot(mv, t), np.linalg.norm(mv - np.dot(mv, t) * t)))

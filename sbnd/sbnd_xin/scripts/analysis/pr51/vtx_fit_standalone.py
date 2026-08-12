#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/51 round-5 addendum: standalone re-implementation of the
WCT neutrino-vertex fit (MyFCN) for one event, so the constraint range and the
fit annulus can be swept outside the toolkit.

Answers two questions the in-toolkit fit cannot be asked directly:
  (a) how far would the vertex move if vtx_constraint_range were relaxed?
  (b) do the legs' FAR shoulders point at the same corner as the 1.5-6 cm
      shoulders the production fit uses?  (If not, the near-vertex trajectory
      is distorted and no amount of constraint-relaxing helps.)

Inputs are an arm's mabc-pr.zip:
  track_fit-global    = Segment::fits(), the layer MyFCN reads
  clustering-global   = the imaged 3D charge (the "image")
  vertices-global     = PR vertices, q=15000 marks the main (neutrino) vertex

Usage: vtx_fit_standalone.py <ARM> <EVT> <legA> <legB> [--png OUT.png]
"""
import sys, os, json, zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'

arm, evt = sys.argv[1], int(sys.argv[2])
legs = [int(a) for a in sys.argv[3:5]]
png = None
if '--png' in sys.argv:
    png = sys.argv[sys.argv.index('--png') + 1]

z = zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip'))
load = lambda n: json.loads(z.read('data/0/0-%s.json' % n))

vtxd = load('vertices-global')
vp = np.array([vtxd['x'], vtxd['y'], vtxd['z']]).T
V0 = vp[np.array(vtxd['q']) == 15000][0]

fitd = load('track_fit-global')
FP = np.array([fitd['x'], fitd['y'], fitd['z']]).T
FR = np.array(fitd['real_cluster_id'])

imgd = load('clustering-global')
IP = np.array([imgd['x'], imgd['y'], imgd['z']]).T
IQ = np.array(imgd['q'], dtype=float)

FLOOR = 0.15 ** 2


def leg_pca(pts, V, r1, r2):
    """MyFCN::AddSegment for one leg: keep fit points in the (r1, r2] shell
    around V, PCA about the kept point closest to V, eigenvalues + (0.15cm)^2."""
    d = np.linalg.norm(pts - V, axis=1)
    sel = pts[(d >= r1) & (d <= r2)]
    if len(sel) < 2:
        return None
    center = sel[np.argmin(np.linalg.norm(sel - V, axis=1))]
    cov = np.zeros((3, 3))
    for p in sel:
        dd = p - center
        cov += np.outer(dd, dd)
    cov /= len(sel)
    w, vecs = np.linalg.eigh(cov)
    lam = w[::-1] + FLOOR
    ax = vecs[:, ::-1].T
    return dict(n=len(sel), lam=lam, ax=ax, center=center)


def myfcn_solve(pcas, V, rng):
    """MyFCN::FitVertex normal equations.  rng=None -> no vertex constraint."""
    A = np.zeros((3, 3)); b = np.zeros(3); npts = 0
    for pc in pcas:
        lam, ax, c = pc['lam'], pc['ax'], pc['center']
        npts += pc['n']
        R = np.zeros((3, 3))                       # row 0 (track dir) zeroed
        R[1] = np.sqrt(lam[0] / lam[1]) * ax[1]
        R[2] = np.sqrt(lam[0] / lam[2]) * ax[2]
        A += R.T @ R
        b += R.T @ (R @ c)
    if rng is not None:
        k = npts / rng ** 2
        A += np.eye(3) * k
        b += V * k
    return np.linalg.solve(A, b), A, npts


WINDOWS = [(1.5, 6.0), (1.5, 10.0), (3.0, 12.0), (6.0, 18.0), (10.0, 30.0)]
RANGES = [0.43, 1.0, 2.0, 5.0, None]

print('arm=%s evt=%d  legs=%s' % (arm, evt, legs))
print('production vertex: (%.3f, %.3f, %.3f)' % tuple(V0))
print()
print('%-14s %-9s %-9s %-7s   %-28s %s' %
      ('window(cm)', 'npts A', 'npts B', 'angle', 'unconstrained fit', '|move| by constraint range'))
print('%-14s %-9s %-9s %-7s   %-28s %s' %
      ('', '', '', '', '(x, y, z) cm', '  '.join('%5s' % (r if r else 'inf') for r in RANGES)))

rows = []
for r1, r2 in WINDOWS:
    pcas = []
    ok = True
    for L in legs:
        pc = leg_pca(FP[FR == L], V0, r1, r2)
        if pc is None:
            ok = False
            break
        pcas.append(pc)
    if not ok:
        print('%-14s  (a leg has <2 points in this shell)' % ('(%.0f,%.0f]' % (r1, r2)))
        continue
    ang = np.degrees(np.arccos(np.clip(abs(np.dot(pcas[0]['ax'][0], pcas[1]['ax'][0])), -1, 1)))
    moves = []
    sols = {}
    for rng in RANGES:
        s, A, npts = myfcn_solve(pcas, V0, rng)
        sols[rng] = s
        moves.append(np.linalg.norm(s - V0))
    sinf = sols[None]
    print('%-14s %-9d %-9d %6.1f    (%7.3f,%7.3f,%7.3f)   %s' %
          ('(%.0f,%.0f]' % (r1, r2), pcas[0]['n'], pcas[1]['n'], 180 - ang if ang > 90 else ang,
           sinf[0], sinf[1], sinf[2],
           '  '.join('%5.2f' % m for m in moves)))
    rows.append(((r1, r2), pcas, sols))

print()
print('NOTE: "angle" is between the two legs\' leading PCA axes (sign-free).')
print('      |move| is the distance from the production vertex.')

# ---------------------------------------------------------------- plot
if png:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    RAD = 25.0
    di = np.linalg.norm(IP - V0, axis=1)
    m = di < RAD
    ip, iq = IP[m], IQ[m]

    # leg trajectories
    legpts = {L: FP[FR == L] for L in legs}

    # basis for the "leg plane": bisector e1, in-plane perpendicular e2
    pc_nom = dict(zip(legs, rows[0][1]))
    d0 = pc_nom[legs[0]]['ax'][0].copy()
    d1 = pc_nom[legs[1]]['ax'][0].copy()
    # orient both away from the vertex
    for L, d in ((legs[0], d0), (legs[1], d1)):
        far = legpts[L][np.argmax(np.linalg.norm(legpts[L] - V0, axis=1))]
        if np.dot(far - V0, d) < 0:
            d *= -1
    e1 = d0 + d1
    e1 /= np.linalg.norm(e1)
    e2 = d0 - np.dot(d0, e1) * e1
    e2 /= np.linalg.norm(e2)

    views = [
        ('z (cm)', 'y (cm)', lambda P: (P[:, 2], P[:, 1])),
        ('z (cm)', 'x (cm)  [drift]', lambda P: (P[:, 2], P[:, 0])),
        ('along leg bisector (cm)', 'in leg plane (cm)',
         lambda P: ((P - V0) @ e1, (P - V0) @ e2)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0))
    cols = {legs[0]: '#d62728', legs[1]: '#1f77b4'}
    for ax, (xl, yl, proj) in zip(axes, views):
        X, Y = proj(ip)
        ax.scatter(X, Y, s=5, c=np.clip(iq, 0, np.percentile(iq, 98)),
                   cmap='Greys', alpha=0.55, linewidths=0, zorder=1)
        for L in legs:
            P = legpts[L]
            P = P[np.linalg.norm(P - V0, axis=1) < RAD]
            x, y = proj(P)
            ax.plot(x, y, '-', color=cols[L], lw=1.6, zorder=3,
                    label='seg %d fit()' % L)
        # production vertex
        x, y = proj(V0[None, :])
        ax.plot(x, y, '*', ms=20, color='#2ca02c', mec='k', mew=0.7, zorder=6,
                label='production vertex')
        # unconstrained fits, one per window
        mk = ['o', 's', '^', 'D', 'v']
        for i, ((r1, r2), pcas, sols) in enumerate(rows):
            s = sols[None]
            x, y = proj(s[None, :])
            ax.plot(x, y, mk[i % len(mk)], ms=7, mfc='none', mew=1.6,
                    color=plt.cm.viridis(i / max(1, len(rows) - 1)), zorder=5,
                    label='unconstr. fit, shell (%.0f,%.0f]' % (r1, r2))
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25, lw=0.4)

    axes[2].set_xlim(-8, 8); axes[2].set_ylim(-8, 8)
    axes[0].legend(fontsize=7.5, loc='best', framealpha=0.9)
    fig.suptitle('evt %d, arm %s — neutrino vertex region (image = grey, legs = fit())  '
                 'production vertex (%.2f, %.2f, %.2f) cm'
                 % (evt, arm, *V0), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png, dpi=140)
    print('\nwrote', png)

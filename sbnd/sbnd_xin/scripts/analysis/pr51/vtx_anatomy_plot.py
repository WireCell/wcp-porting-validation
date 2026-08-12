#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/51 addendum follow-up: near-vertex anatomy of one event.

Two rows.  Top = 25 cm context, bottom = 6 cm zoom, in three projections
(z-y, z-x, and the plane spanned by the two legs).  Overlays:
  grey      imaged 3D charge (clustering-global)
  coloured  each leg's fitted trajectory (track_fit-global = Segment::fits())
  dashed    each leg's FAR-shoulder PCA axis, extrapolated back to the vertex
            region -- if a leg's dashed line misses the star, that leg's
            near-vertex trajectory is bent relative to its own straight body
  star      production neutrino vertex
  markers   standalone MyFCN solutions for several fit annuli

Usage: vtx_plot2.py <ARM> <EVT> <legA> <legB> <OUT.png>
"""
import sys, os, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
arm, evt = sys.argv[1], int(sys.argv[2])
legs = [int(sys.argv[3]), int(sys.argv[4])]
out = sys.argv[5]

z = zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip'))
load = lambda n: json.loads(z.read('data/0/0-%s.json' % n))

vd = load('vertices-global')
vp = np.array([vd['x'], vd['y'], vd['z']]).T
V0 = vp[np.array(vd['q']) == 15000][0]
fd = load('track_fit-global')
FP = np.array([fd['x'], fd['y'], fd['z']]).T
FR = np.array(fd['real_cluster_id'])
idd = load('clustering-global')
IP = np.array([idd['x'], idd['y'], idd['z']]).T
IQ = np.array(idd['q'], dtype=float)

FLOOR = 0.15 ** 2


def pca(P, V, r1, r2):
    d = np.linalg.norm(P - V, axis=1)
    s = P[(d >= r1) & (d <= r2)]
    if len(s) < 2:
        return None
    c = s[np.argmin(np.linalg.norm(s - V, axis=1))]
    cov = sum(np.outer(p - c, p - c) for p in s) / len(s)
    w, e = np.linalg.eigh(cov)
    return dict(n=len(s), lam=w[::-1] + FLOOR, ax=e[:, ::-1].T, center=c)


def solve(pcs, V, rng):
    A = np.zeros((3, 3)); b = np.zeros(3); n = 0
    for pc in pcs:
        lam, ax, c = pc['lam'], pc['ax'], pc['center']
        n += pc['n']
        R = np.zeros((3, 3))
        R[1] = np.sqrt(lam[0] / lam[1]) * ax[1]
        R[2] = np.sqrt(lam[0] / lam[2]) * ax[2]
        A += R.T @ R; b += R.T @ (R @ c)
    if rng is not None:
        k = n / rng ** 2
        A += np.eye(3) * k; b += V * k
    return np.linalg.solve(A, b)


legpts = {L: FP[FR == L] for L in legs}
NOM, FAR = (1.5, 6.0), (6.0, 18.0)
pc_nom = {L: pca(legpts[L], V0, *NOM) for L in legs}
pc_far = {L: pca(legpts[L], V0, *FAR) for L in legs}

# image-suggested vertex: centroid of the backmost 1% of the TRACK leg's own
# associate points, measured along the backward bisector (see doc).
sd = load('shower_track-global')
SP = np.array([sd['x'], sd['y'], sd['z']]).T
SR = np.array(sd['real_cluster_id'])


def _out(L, r1, r2):
    d = pca(legpts[L], V0, r1, r2)['ax'][0].copy()
    far = legpts[L][np.argmax(np.linalg.norm(legpts[L] - V0, axis=1))]
    return d if np.dot(far - V0, d) > 0 else -d


_e1 = _out(legs[0], 1.5, 10) + _out(legs[1], 1.5, 10)
_e1 /= np.linalg.norm(_e1)
_mu = SP[SR == legs[1]]
_s = (_mu - V0) @ _e1
TREF = _mu[_s < np.percentile(_s, 1.0)].mean(axis=0)

pc_r6 = {L: pca(legpts[L], V0, 6.0, 18.0) for L in legs}
cands = [
    ('production vertex', V0, '*', '#2ca02c', 20),
    ('MyFCN as shipped: annulus 1.5-6 cm, prior 0.43 cm  (moves 0.14 cm)',
     solve([pc_nom[L] for L in legs], V0, 0.43), 'o', '#9467bd', 9),
    ('same annulus, prior REMOVED  (moves 0.34 cm -- the prior is not the blocker)',
     solve([pc_nom[L] for L in legs], V0, None), 's', '#8c564b', 9),
    ('annulus 6-18 cm, prior 1.0 cm  (moves 2.45 cm)',
     solve([pc_r6[L] for L in legs], V0, 1.0), 'D', '#ff7f0e', 10),
    ('image-suggested vertex (start of the track leg\'s own charge)',
     TREF, 'P', '#e377c2', 13),
]

# leg-plane basis (from the far axes, oriented outward)
dirs = {}
for L in legs:
    d = pc_far[L]['ax'][0].copy()
    far = legpts[L][np.argmax(np.linalg.norm(legpts[L] - V0, axis=1))]
    if np.dot(far - V0, d) < 0:
        d = -d
    dirs[L] = d
e1 = dirs[legs[0]] + dirs[legs[1]]; e1 /= np.linalg.norm(e1)
e2 = dirs[legs[0]] - np.dot(dirs[legs[0]], e1) * e1; e2 /= np.linalg.norm(e2)

views = [
    ('z (cm)', 'y (cm)', lambda P: (P[:, 2], P[:, 1])),
    ('z (cm)', 'x (cm)  [drift]', lambda P: (P[:, 2], P[:, 0])),
    ('along leg bisector (cm)', 'in leg plane (cm)',
     lambda P: ((P - V0) @ e1, (P - V0) @ e2)),
]
cols = {legs[0]: '#d62728', legs[1]: '#1f77b4'}
names = {legs[0]: 'seg %d (shower)' % legs[0], legs[1]: 'seg %d (track)' % legs[1]}

fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.0))
for row, RAD in enumerate((25.0, 6.0)):
    di = np.linalg.norm(IP - V0, axis=1)
    ip, iq = IP[di < RAD], IQ[di < RAD]
    for col, (xl, yl, proj) in enumerate(views):
        ax = axes[row, col]
        if len(ip):
            X, Y = proj(ip)
            ax.scatter(X, Y, s=9 if row else 5,
                       c=np.clip(iq, 0, np.percentile(iq, 98)),
                       cmap='Greys', alpha=0.6, linewidths=0, zorder=1)
        for L in legs:
            P = legpts[L]
            P = P[np.linalg.norm(P - V0, axis=1) < RAD * 1.4]
            x, y = proj(P)
            ax.plot(x, y, '-', color=cols[L], lw=1.8, zorder=3,
                    label=names[L] + ' fit()' if (row == 0 and col == 0) else None)
            # far-shoulder axis, extrapolated through the vertex region
            c, d = pc_far[L]['center'], dirs[L]
            t = np.linspace(-11, 22, 2)[:, None]
            E = c[None, :] + d[None, :] * t
            x, y = proj(E)
            ax.plot(x, y, '--', color=cols[L], lw=1.2, alpha=0.85, zorder=4,
                    label=(names[L] + ' 6-18 cm axis, extrapolated'
                           if (row == 0 and col == 0) else None))
        for lab, P, mk, cc, ms in cands:
            x, y = proj(np.atleast_2d(P))
            ax.plot(x, y, mk, ms=ms, color=cc, mec='k', mew=0.7,
                    mfc=cc if mk == '*' else 'none', zorder=6,
                    label=lab if (row == 0 and col == 0) else None)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25, lw=0.4)
        if col == 2:
            ax.set_xlim(-RAD, RAD); ax.set_ylim(-RAD, RAD)
        else:
            xs, ys = proj(np.atleast_2d(V0))
            ax.set_xlim(xs[0] - RAD, xs[0] + RAD)
            ax.set_ylim(ys[0] - RAD, ys[0] + RAD)
    axes[row, 0].set_title('%s  (+-%g cm)' % ('context' if row == 0 else 'ZOOM', RAD),
                           fontsize=10, loc='left')

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, fontsize=9, loc='lower center', ncol=2, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.005))
fig.suptitle("evt %d (%s) — neutrino-vertex anatomy.  The track leg's 6-18 cm axis misses the "
             "reconstructed vertex by 2.4 cm (the shower's misses by 0.3 cm),\n"
             "and charge assigned to the track leg extends 1.9 cm behind it.  Removing the 0.43 cm "
             "prior buys only 0.34 cm; excluding the bent inner 6 cm buys 2.45 cm." % (evt, arm),
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.075, 1, 0.96])
fig.savefig(out, dpi=140)
print('wrote', out)
for lab, P, _, _, _ in cands:
    print('%-66s (%.3f, %.3f, %.3f)  |move| %.2f cm  d(image-suggested) %.2f cm'
          % (lab, *P, np.linalg.norm(P - V0), np.linalg.norm(P - TREF)))

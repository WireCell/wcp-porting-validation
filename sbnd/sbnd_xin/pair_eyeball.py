#!/usr/bin/env python3
"""Eyeball every NEW edge a candidate cathode_connect relaxation would add.

Input: (event, p1, p2) from the connector's own [feat] line.  The two clusters
are recovered from the Q/L clustering-global layer by the real_cluster_id of the
point nearest each tip -- real_cluster_id is the post-connector, pre-flash-merge
geometric ident, so a REJECTED pair still reads as two distinct ids.

For each pair we print, and plot, the evidence that decides "one track or two":
  * both halves' length and their PCA opening angle
  * the near-cathode arm direction of each half (last ARM cm of points)
  * the perpendicular residual when half A's arm is extrapolated to half B's tip
    (and vice versa) -- a real crosser continues onto its partner
  * transverse offset dyz across the gap (doc 14: data cathode offset ~1.1 cm)
"""
import os, json, zipfile, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARM_CM = 20.0


def load(arm, evt):
    z = zipfile.ZipFile(os.path.join(arm, f'ql_evt{evt}', 'mabc-all-apa.zip'))
    d = json.loads(z.read('data/0/0-clustering-global.json'))
    P = np.stack([np.asarray(d['x'], float), np.asarray(d['y'], float),
                  np.asarray(d['z'], float)], 1)
    return P, np.asarray(d['real_cluster_id'])


def pca_axis(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0]


def coll(a, b):
    return np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b))))))


def arm_dir(P, tip, r=ARM_CM):
    """direction of the near-cathode arm, pointing AWAY from the tip."""
    m = np.linalg.norm(P - tip, axis=1) < r
    if m.sum() < 5:
        return None, int(m.sum())
    c, ax = pca_axis(P[m])
    v = c - tip
    if np.dot(v, ax) < 0:
        ax = -ax
    return ax, int(m.sum())


def analyse(arm, evt, p1, p2, ax=None, label=''):
    P, rc = load(arm, evt)
    t = cKDTree(P)
    i1 = t.query(np.asarray(p1))[1]
    i2 = t.query(np.asarray(p2))[1]
    r1, r2 = int(rc[i1]), int(rc[i2])
    A, B = P[rc == r1], P[rc == r2]
    ta = cKDTree(A)
    d, j = ta.query(B, k=1)
    k = int(np.argmin(d))
    q1, q2, dis = A[j[k]], B[k], float(d[k])
    ca, aa = pca_axis(A)
    cb, ab = pca_axis(B)
    La = float(np.ptp((A - ca) @ aa))
    Lb = float(np.ptp((B - cb) @ ab))
    da, na = arm_dir(A, q1)
    db, nb = arm_dir(B, q2)
    res = {}
    if da is not None:
        # extrapolate A's arm BACK through its tip to B's tip
        v = q2 - q1
        res['perp_A->B'] = float(np.linalg.norm(v - np.dot(v, -da) * (-da)))
    if db is not None:
        v = q1 - q2
        res['perp_B->A'] = float(np.linalg.norm(v - np.dot(v, -db) * (-db)))
    out = dict(evt=evt, rcid=(r1, r2), n=(len(A), len(B)),
               L=(round(La, 1), round(Lb, 1)), dis=round(dis, 2),
               tips=(round(float(q1[0]), 2), round(float(q2[0]), 2)),
               dX=round(abs(float(q1[0] - q2[0])), 2),
               dyz=round(float(np.hypot(q1[1] - q2[1], q1[2] - q2[2])), 2),
               ttP=round(coll(aa, ab), 1),
               arm_ang=None if (da is None or db is None) else round(coll(da, db), 1),
               narm=(na, nb),
               **{k: round(v, 2) for k, v in res.items()})
    if ax is not None:
        c = (q1 + q2) / 2
        w = 60
        for Q, col in ((A, 'tab:blue'), (B, 'tab:red')):
            m = (np.abs(Q[:, 0] - c[0]) < w) & (np.abs(Q[:, 2] - c[2]) < w)
            ax.plot(Q[m, 0], Q[m, 2], '.', ms=1.2, color=col, alpha=.5)
        ax.plot([q1[0], q2[0]], [q1[2], q2[2]], 'k-', lw=1)
        ax.axvline(0, color='gray', ls='--', lw=.8)
        ax.set_xlim(c[0] - w, c[0] + w)
        ax.set_ylim(c[2] - w, c[2] + w)
        ax.set_title(f'{label}evt {evt}  rc{r1}/{r2}\n'
                     f'dis={out["dis"]} ttP={out["ttP"]} arm={out["arm_ang"]}',
                     fontsize=7)
        ax.tick_params(labelsize=6)
    return out


if __name__ == '__main__':
    arm = sys.argv[1]
    pairs = json.load(open(sys.argv[2]))
    n = len(pairs)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, pr in enumerate(pairs):
        r = analyse(arm, pr['evt'], pr['p1'], pr['p2'], axes[i])
        print(json.dumps(r))
    for j in range(n, len(axes)):
        axes[j].axis('off')
    fig.suptitle('candidate NEW cathode_connect edges — x (cm) vs z (cm), '
                 'cathode at x=0', fontsize=9)
    fig.tight_layout()
    fig.savefig(sys.argv[3], dpi=140)
    print('wrote', sys.argv[3])

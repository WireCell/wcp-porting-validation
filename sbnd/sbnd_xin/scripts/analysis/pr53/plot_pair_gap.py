#!/usr/bin/env python3
"""doc pr/53 round 4/5: general per-plane gap visualization for an
owner-flagged "over-clustering" point pair. Corrects an overclaim earlier in
this investigation ("422851/521075 are charge-connected, no gap") -- a direct
per-plane replay of connect_graph_relaxed's own test shows a REAL,
measurable gap (missing live charge AND no dead-channel excuse) at the true
closest-approach neck for every one of the 5 owner pairs, not just the two
family-B (71372) ones already known to fail the test outright. This script
marks EACH plane's gap status separately (a step can be U-bad, V-bad, W-bad,
or several at once) rather than cherry-picking one plane, and reports the
relaxed-graph's own kill decision (REJECT=True/False) plus how close nb1[0]
sits to the ">2" floor (doc pr/53 Finding 3 / F1-F2).

When A and B are already in the SAME 1.5cm-proximity component (21073's
case), the straight A-B line is not the path that actually joins them in
real_cluster_id -- the charge-contiguous geodesic (0.99cm-bottleneck Dijkstra,
same construction as plot_21073_gap.py) is overlaid too, so this figure does
not appear to contradict the existing two-path 53_21073_gap.png: the direct
line's gap is real, but it is not the path connectivity is via.

Usage: plot_pair_gap.py <ql_dir> <Ax> <Ay> <Az> <Bx> <By> <Bz> <label> <out.png>
"""
import sys, os, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from oc53_probe import Loader, walk_and_score, proximity_components, true_neck, img_load

PLANE_STYLE = {
    0: dict(marker='^', color='tab:orange', label='U gap'),
    1: dict(marker='s', color='tab:green', label='V gap'),
    2: dict(marker='D', color='tab:red', label='W gap'),
}


def make_plot(ql_dir, A, B, label, out):
    A = np.array(A, float); B = np.array(B, float)
    P, C = img_load(ql_dir)
    ia = int(np.argmin(np.linalg.norm(P - A, axis=1)))
    ib = int(np.argmin(np.linalg.norm(P - B, axis=1)))
    cid = C[ia]
    assert C[ib] == cid, "A and B are not in the same cluster in this QL output"

    m, X, lab = proximity_components(P, C, ia, radius=1.5)
    ja = int(np.where(m == ia)[0][0])
    jb = int(np.where(m == ib)[0][0])
    neck = true_neck(X, lab, ja, jb)

    ld = Loader(ql_dir)
    geo_pts = None
    if neck is None:
        # Already in the same 1.5cm-proximity component (21073's case): the
        # straight A-B line is NOT the path that actually connects them in
        # real_cluster_id (see plot_21073_gap.py / doc pr/53 sec 7) -- walk
        # it anyway for the per-plane gap markers, but also find and overlay
        # the charge-contiguous geodesic so this panel does not look like it
        # contradicts 53_21073_gap.png.
        p1, p2, neck_dis = A, B, float(np.linalg.norm(B - A))
        note = "SAME proximity component -- direct A-B line (real gap) + actual geodesic (no gap) both shown"
        t = cKDTree(X)
        pr = np.array(list(t.query_pairs(1.05)))
        w = np.linalg.norm(X[pr[:, 0]] - X[pr[:, 1]], axis=1)
        g = coo_matrix((w, (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
        dist, pred = dijkstra(g, directed=False, indices=ja, return_predecessors=True)
        path = [jb]
        while path[-1] != ja:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        geo_pts = X[path]
    else:
        p1, p2, neck_dis, na, nb_ = neck
        note = "true closest-approach neck between proximity components"
    r = walk_and_score(ld, p1, p2)
    ld.cleanup()

    pts = np.array([p1 + (p2 - p1) / r['nst'] * (i + 1) for i in range(r['nst'])])
    # s = [u_live, v_live, w_live, u_dead, v_dead, w_dead]
    gap = {pi: np.array([s[pi] == 0 and s[pi + 3] == 0 for s in r['rows']]) for pi in range(3)}
    any_gap = gap[0] | gap[1] | gap[2]

    mid = (p1 + p2) / 2
    near = np.linalg.norm(X - mid, axis=1) < max(14, neck_dis * 0.6 + 6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, (i0, i1, l0, l1) in zip(axes, [(2, 1, 'z (cm)', 'y (cm)'), (0, 2, 'x (cm)', 'z (cm)')]):
        ax.scatter(X[near][:, i0], X[near][:, i1], s=10, c='0.82', label='image charge (this cluster)')
        ax.plot(pts[:, i0], pts[:, i1], 'k--', lw=1.1, alpha=0.6, label='tested path (%.2fcm)' % neck_dis)
        ax.scatter(pts[~any_gap][:, i0], pts[~any_gap][:, i1], marker='o', s=35,
                   facecolors='none', edgecolors='0.4', linewidths=1.2, label='step: all 3 planes ok')
        for pi, style in PLANE_STYLE.items():
            sel = gap[pi]
            if sel.any():
                ax.scatter(pts[sel][:, i0], pts[sel][:, i1], s=90, linewidths=2.0,
                           facecolors='none', edgecolors=style['color'], marker=style['marker'],
                           label='step: %s (no live, no dead ch)' % style['label'])
        ax.scatter(*zip((A[i0], A[i1]), (B[i0], B[i1])), marker='*', s=220, c='black', zorder=5,
                   label='A, B (owner pair)')
        if neck is not None:
            ax.scatter(*zip((p1[i0], p1[i1]), (p2[i0], p2[i1])), marker='+', s=140, c='blue',
                       linewidths=2.5, zorder=5, label='p1, p2 (true neck endpoints)')
        if geo_pts is not None:
            ax.plot(geo_pts[:, i0], geo_pts[:, i1], '-', color='tab:blue', lw=2.0, alpha=0.85,
                     label='actual charge geodesic (%.1fcm, real_cluster_id path, no gap)' % (
                         np.sum(np.linalg.norm(np.diff(geo_pts, axis=0), axis=1))))
        ax.set_xlabel(l0); ax.set_ylabel(l1)
    axes[0].set_title('%s: (y,z) view -- wire-plane view' % label)
    axes[1].set_title('%s: (x,z) view -- drift direction' % label)
    axes[0].legend(fontsize=7, loc='best')
    plt.suptitle('%s -- %s\nsteps: %d/%d U-gap, %d/%d V-gap, %d/%d W-gap; relaxed-graph nb1[0]=%d '
                 '(kill floor >2) branch=%s REJECT=%s'
                 % (label, note, gap[0].sum(), r['nst'], gap[1].sum(), r['nst'], gap[2].sum(), r['nst'],
                    r['nb1'][0], r['branch'], r['kill']))
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print('wrote', out)
    print('  neck_dis=%.2fcm n=%d branch=%s nb1[0]=%d (U=%d V=%d W=%d bad steps)  REJECT=%s'
          % (neck_dis, r['nst'], r['branch'], r['nb1'][0], gap[0].sum(), gap[1].sum(), gap[2].sum(), r['kill']))


if __name__ == '__main__':
    ql_dir = sys.argv[1]
    A = tuple(float(v) for v in sys.argv[2:5])
    B = tuple(float(v) for v in sys.argv[5:8])
    label = sys.argv[8]
    out = sys.argv[9]
    make_plot(ql_dir, A, B, label, out)

#!/usr/bin/env python3
"""doc pr/53 round 7: find and route "fitted trajectory with no 3D image"
over-clustering cases -- the owner's hand-scan class that survives round 6's
relaxed_strict flavor.

The owner's criterion, made measurable: a PR track_fit/shower_track point is
"ghost" if it sits farther than d0 (default 1cm) from the nearest final-output
image point (mabc-pr.zip 0-clustering-global.json). A contiguous run of ghost
points inside one segment's own fit is the signature they hand-scanned in Bee
-- "fitted track trajectory connecting the points, however these fitted
trajectory points are not with 3D images".

Each run is then ROUTED to explain *why* protect_bundle's relaxed_strict graph
still bridges it, using the existing WCT_RELAXED_EDGE_CENSUS logs (OC53CENSUS-S
prefix, protect_bundle-only by construction, doc pr/53 round 6 sec 17.1):
  (a) closest-pair edge  -- an emitted OC53CENSUS-S "edge" whose closest-pair
      p1-p2 segment passes within routing_tol of the run
  (b) none-of-the-above  -- reported as an unrouted residual, NOT guessed at;
      the doc must say so rather than claim a fix for it (round 6 split_exam
      v1/v2 lesson: never invent an attribution the data doesn't support)

For every routed (and unrouted) run this also computes the ghost step count
along the census edge's OWN p1/p2 (round 7 S5 candidate metric): at each 1cm
step of the C++'s own closest-pair test, the 3D distance to the nearest image
point in this cluster's own pctree (oc53_probe.Loader.img_closest_dis), so the
"is 3D image support required" rule can be scanned offline against real data
before any C++ is written.

Usage:
  fitgap_exam.py <on_arm> <cen_arm> <outdir> <evt> [evt...] \\
      [--d0 CM] [--min-run N] [--route-tol CM] [--img-radius CM]

  <on_arm>  the production/on PR-stage arm, e.g. work-pr53-on19a (holds
            pr_evt<N>/{mabc-pr.zip,pctree-pr-evt<N>.tar.gz})
  <cen_arm> the matching WCT_RELAXED_EDGE_CENSUS-enabled rerun of the SAME
            flavor as <on_arm>, e.g. work-pr53-cen19 (holds
            pr_evt<N>/wct_pr_evt<N>.log with OC53CENSUS-S lines)

Repro (doc pr/53 round 7):
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
    python3 scripts/analysis/pr53/fitgap_exam.py \\
        work-pr53-on19a work-pr53-cen19 pics 142421 269774 463565 71372
"""
import sys, os, re, json, zipfile, argparse
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from oc53_probe import Loader, walk_and_score, angles  # noqa: E402
from split_exam import parse_blocks  # noqa: E402  (OC53CENSUS-S parser, round 6)

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'

FIT_LAYERS = ('0-track_fit-global.json', '0-shower_track-global.json')


def _z(arm, evt, name):
    with zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%s' % evt, 'mabc-pr.zip')) as f:
        return json.loads(f.read('data/0/%s' % name))


def load_image(arm, evt):
    d = _z(arm, evt, '0-clustering-global.json')
    P = np.c_[d['x'], d['y'], d['z']].astype(float)
    rcid = np.array(d['real_cluster_id'])
    return P, rcid


def load_fit(arm, evt):
    """Concatenate track_fit-global + shower_track-global. Per WCT's own
    Bee-layer schema (mabc-pr.zip), each entry's 'cluster_id' is the final
    output cluster, 'real_cluster_id' here is actually the per-segment id
    (e.g. 7009, 43021 -- large, segment-granularity, NOT the same namespace
    as clustering-global's real_cluster_id). We keep WCT's field name but
    call it 'seg' throughout this script to avoid confusion with the image
    cluster id used in load_image()."""
    xs, ys, zs, cid, seg, src = [], [], [], [], [], []
    for lay in FIT_LAYERS:
        d = _z(arm, evt, lay)
        n = len(d['x'])
        xs += d['x']; ys += d['y']; zs += d['z']
        cid += d['cluster_id']; seg += d['real_cluster_id']
        src += [lay] * n
    F = np.c_[xs, ys, zs].astype(float)
    return F, np.array(cid), np.array(seg), np.array(src)


def find_gap_runs(F, cid, seg, src, img_tree, d0=1.0, min_run=2):
    """Contiguous-in-storage-order runs (per cluster_id, seg, src -- fit
    points are written in trajectory order within one segment) where every
    point is farther than d0 from the nearest final image point."""
    dist, _ = img_tree.query(F)
    runs = []
    order = np.lexsort((np.arange(len(F)), src, seg, cid))
    keys = list(zip(cid[order], seg[order], src[order]))
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and keys[j] == keys[i]:
            j += 1
        # keys[i:j] is one (cluster_id, seg, src) group in original order
        idxs = order[i:j]
        bad = dist[idxs] > d0
        k = 0
        while k < len(bad):
            if not bad[k]:
                k += 1
                continue
            k2 = k
            while k2 < len(bad) and bad[k2]:
                k2 += 1
            if k2 - k >= min_run:
                run_idx = idxs[k:k2]
                runs.append(dict(cluster_id=int(cid[idxs[0]]), seg=int(seg[idxs[0]]),
                                  src=str(src[idxs[0]]), idx=run_idx,
                                  pts=F[run_idx], dist=dist[run_idx],
                                  n=k2 - k, maxdist=float(dist[run_idx].max())))
            k = k2
        i = j
    runs.sort(key=lambda r: -r['maxdist'])
    return runs


def _point_seg_dist(p, a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); p = np.asarray(p, float)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - p))


def route_run(run, census_edges, route_tol=5.0):
    """Route a gap run to the nearest emitted OC53CENSUS-S closest-pair edge
    by point-to-segment distance from the run's centroid to the edge's own
    p1-p2 line. No edge within route_tol => unrouted (reported, not guessed)."""
    centroid = run['pts'].mean(axis=0)
    best, best_d = None, 1e9
    for e in census_edges:
        c = e.get('closest')
        if c is None:
            continue
        d = _point_seg_dist(centroid, c['p1'], c['p2'])
        if d < best_d:
            best, best_d = e, d
    if best is not None and best_d < route_tol:
        return best, best_d
    return None, best_d


def emitted_closest_edges(cen_arm, evt):
    """All OC53CENSUS-S emitted edges (protect_bundle's relaxed_strict graph,
    round 6) with their closest-pair record attached, across every
    protect_bundle cluster block in the event."""
    log = os.path.join(SB, cen_arm, 'pr_evt%s' % evt, 'wct_pr_evt%s.log' % evt)
    blocks = parse_blocks(log, 'OC53CENSUS-S')
    out = []
    for b in blocks:
        for jk, e in b['edges'].items():
            rec = dict(e)
            rec['jk'] = jk
            rec['sig'] = b['sig']
            rec['closest'] = b['closest'].get(jk)
            out.append(rec)
    return out


def ghost_steps_along_edge(ld, c, img_radius=1.0):
    """Round-7 S5 candidate metric, evaluated at the C++'s OWN p1/p2 for an
    emitted closest-pair edge: per interior 1cm step (excludes the final
    step, which is always p2 -- S1's endpoint-honest convention, doc pr/53
    round 6 sec 16.3), the 3D distance to the nearest '3d' image point in
    this cluster's own pctree, and whether that step is dead-channel-excused
    by the SAME per-plane test the C++ already runs (test_good_point's
    scores[3..5] -- reused via walk_and_score's rows)."""
    r = walk_and_score(ld, c['p1'], c['p2'])
    p1 = np.array(c['p1'], float); p2 = np.array(c['p2'], float)
    nst = r['nst']
    n_ghost = 0
    n_ghost_unexcused = 0
    maxgap = 0.0
    for ii in range(nst - 1):  # interior steps only; last step is p2 itself
        tp = p1 + (p2 - p1) / nst * (ii + 1)
        d = ld.img_closest_dis(tp)
        s = r['rows'][ii]
        dead_excused = (s[3] + s[4] + s[5]) > 0
        if d > img_radius:
            n_ghost += 1
            maxgap = max(maxgap, d)
            if not dead_excused:
                n_ghost_unexcused += 1
    return dict(n_interior=nst - 1, n_ghost=n_ghost, n_ghost_unexcused=n_ghost_unexcused,
                maxgap=maxgap, walk=r)


def make_plot(P, rcid, run, best, best_d, ghost, out, label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    F = run['pts']
    mid = F.mean(axis=0)
    rad = max(18.0, run['maxdist'] * 2 + 10)
    near = np.linalg.norm(P - mid, axis=1) < rad
    Pn, Rn = P[near], rcid[near]
    for ax, (i0, i1, l0, l1) in zip(axes, [(2, 1, 'z (cm)', 'y (cm)'), (0, 2, 'x (cm)', 'z (cm)')]):
        for rid in np.unique(Rn):
            sel = Rn == rid
            ax.scatter(Pn[sel][:, i0], Pn[sel][:, i1], s=8, alpha=0.6, label='image rcid=%d' % rid)
        good = run['dist'] <= 1.0
        ax.plot(F[:, i0], F[:, i1], '-', color='k', lw=1.0, alpha=0.5, label='fitted trajectory (this run)')
        ax.scatter(F[:, i0], F[:, i1], s=45, marker='x', c='red',
                   label='fit pt, no image within 1cm (ghost)')
        if best is not None:
            c = best['closest']
            pts = np.array([c['p1'] + (c['p2'] - c['p1']) / ghost['walk']['nst'] * (i + 1)
                            for i in range(ghost['walk']['nst'])])
            ax.plot(pts[:, i0], pts[:, i1], 'b--', lw=1.3,
                    label='routed census edge (dis=%.2fcm, route_d=%.1fcm)' % (c['dis'], best_d))
            gap = {pi: np.array([s[pi] == 0 and s[pi + 3] == 0 for s in ghost['walk']['rows']])
                   for pi in range(3)}
            style = {0: ('^', 'tab:orange', 'U'), 1: ('s', 'tab:green', 'V'), 2: ('D', 'tab:purple', 'W')}
            for pi, (mk, col, nm) in style.items():
                if gap[pi].any():
                    ax.scatter(pts[gap[pi]][:, i0], pts[gap[pi]][:, i1], s=70, linewidths=1.6,
                               facecolors='none', edgecolors=col, marker=mk,
                               label='census step: %s gap (2D)' % nm)
        ax.set_xlabel(l0); ax.set_ylabel(l1)
    axes[0].set_title('(y,z) view'); axes[1].set_title('(x,z) view -- drift direction')
    axes[0].legend(fontsize=6.5, loc='best')
    rtxt = ('routed: j=%d k=%d dis=%.2fcm nb1=%s  ghost(3D,>1cm)=%d/%d unexcused=%d/%d maxgap=%.1fcm'
            % (best['jk'][0], best['jk'][1], best['closest']['dis'], best['closest']['nb1'],
               ghost['n_ghost'], ghost['n_interior'], ghost['n_ghost_unexcused'], ghost['n_interior'],
               ghost['maxgap'])) if best is not None else ('UNROUTED (nearest census edge %.1fcm away)' % best_d)
    plt.suptitle('%s\nfit run: cid=%d seg=%d n=%d maxdist=%.1fcm src=%s\n%s'
                 % (label, run['cluster_id'], run['seg'], run['n'], run['maxdist'], run['src'], rtxt),
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print('  wrote', out)


def exam_event(on_arm, cen_arm, outdir, evt, d0=1.0, min_run=2, route_tol=5.0, img_radius=1.0):
    P, rcid = load_image(on_arm, evt)
    F, cid, seg, src = load_fit(on_arm, evt)
    img_tree = cKDTree(P)
    runs = find_gap_runs(F, cid, seg, src, img_tree, d0=d0, min_run=min_run)
    print('=== evt %s [%s]: %d fit-gap run(s) (d0=%.1fcm, min_run=%d) ===' % (evt, on_arm, len(runs), d0, min_run))
    if not runs:
        return []
    census_edges = emitted_closest_edges(cen_arm, evt)
    ld = Loader(os.path.join(SB, on_arm, 'pr_evt%s' % evt))
    out_rows = []
    try:
        for i, run in enumerate(runs):
            best, best_d = route_run(run, census_edges, route_tol=route_tol)
            if best is not None:
                ghost = ghost_steps_along_edge(ld, best['closest'], img_radius=img_radius)
                print('  run%d cid=%d seg=%d n=%d maxdist=%.1fcm -> ROUTED j=%d k=%d dis=%.2fcm '
                      'route_d=%.1fcm  nb1=%s  ghost(3D>%.1fcm)=%d/%d unexcused=%d/%d'
                      % (i, run['cluster_id'], run['seg'], run['n'], run['maxdist'],
                         best['jk'][0], best['jk'][1], best['closest']['dis'], best_d,
                         best['closest']['nb1'], img_radius, ghost['n_ghost'], ghost['n_interior'],
                         ghost['n_ghost_unexcused'], ghost['n_interior']))
                out = os.path.join(SB, outdir, '53_r7_%s_cid%d_seg%d.png' % (evt, run['cluster_id'], run['seg']))
                make_plot(P, rcid, run, best, best_d, ghost, out,
                          '%s cid=%d seg=%d (ROUTED)' % (evt, run['cluster_id'], run['seg']))
                out_rows.append(dict(evt=evt, run=run, routed=best, route_d=best_d, ghost=ghost, fig=out))
            else:
                print('  run%d cid=%d seg=%d n=%d maxdist=%.1fcm -> UNROUTED (nearest census edge %.1fcm away)'
                      % (i, run['cluster_id'], run['seg'], run['n'], run['maxdist'], best_d))
                out = os.path.join(SB, outdir, '53_r7_%s_cid%d_seg%d_unrouted.png' % (evt, run['cluster_id'], run['seg']))
                make_plot(P, rcid, run, None, best_d, None, out,
                          '%s cid=%d seg=%d (UNROUTED)' % (evt, run['cluster_id'], run['seg']))
                out_rows.append(dict(evt=evt, run=run, routed=None, route_d=best_d, ghost=None, fig=out))
    finally:
        ld.cleanup()
    return out_rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('on_arm')
    ap.add_argument('cen_arm')
    ap.add_argument('outdir')
    ap.add_argument('evts', nargs='+')
    ap.add_argument('--d0', type=float, default=1.0)
    ap.add_argument('--min-run', type=int, default=2)
    ap.add_argument('--route-tol', type=float, default=5.0)
    ap.add_argument('--img-radius', type=float, default=1.0)
    a = ap.parse_args()
    all_rows = []
    for evt in a.evts:
        all_rows += exam_event(a.on_arm, a.cen_arm, a.outdir, evt, d0=a.d0, min_run=a.min_run,
                                route_tol=a.route_tol, img_radius=a.img_radius)
    routed = [r for r in all_rows if r['routed'] is not None]
    print('\n%d fit-gap run(s) total: %d routed to a census edge, %d unrouted'
          % (len(all_rows), len(routed), len(all_rows) - len(routed)))

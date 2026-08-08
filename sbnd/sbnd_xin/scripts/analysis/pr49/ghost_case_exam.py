#!/usr/bin/env python3
"""doc pr/49: per-mover exam for the fit_blob_coverage knob round.

For every event whose mabc-pr.zip differs between a knob-off arm (BASE) and
a knob-on arm (NEW), report:

  1. the knob's own sentinel evidence -- counts of the
     "fit_blob_coverage: deweighted foreign live cells" debug lines in the
     NEW arm's wct log (the knob only deweights LIVE cells outside the
     fitted cluster's own blob coverage that sit inside a 3D-distant
     foreign cluster's, so a nonzero count IS the projection-ghost
     signature, per-plane);
  2. the fit-vs-image consistency metric before/after: for each
     real_cluster_id present in the fitted trajectory, the max over fit
     points of the distance to the nearest raw-image point of the SAME
     cluster (the pr/49 symptom was fit points detouring ~3.7 cm from their
     own image).  A genuine ghost fix moves this DOWN for the affected
     cluster and leaves other clusters at baseline.

Usage: python3 ghost_case_exam.py [BASE_LABEL NEW_LABEL]
       (defaults: work-pr49-off48 vs work-pr49-on48)
"""
import sys, os, glob, json, re, zipfile
import numpy as np

sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest')
import hash_archive as ha

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
BASE = os.path.join(SB, sys.argv[1] if len(sys.argv) > 2 else 'work-pr49-off48')
NEW = os.path.join(SB, sys.argv[2] if len(sys.argv) > 2 else 'work-pr49-on48')

# WCT log lines can tear mid-write: match the line HEAD only.
SENTINEL = re.compile(r'fit_blob_coverage: deweighted foreign live cells u=(\d+) v=(\d+) w=(\d+)')


def movers(base, new):
    out = []
    base_evts = set(int(os.path.basename(p).replace('pr_evt', ''))
                    for p in glob.glob(os.path.join(base, 'pr_evt*')))
    new_evts = set(int(os.path.basename(p).replace('pr_evt', ''))
                   for p in glob.glob(os.path.join(new, 'pr_evt*')))
    for evt in sorted(base_evts & new_evts):
        bp = os.path.join(base, 'pr_evt%d' % evt, 'mabc-pr.zip')
        np_ = os.path.join(new, 'pr_evt%d' % evt, 'mabc-pr.zip')
        if not (os.path.exists(bp) and os.path.exists(np_)):
            continue
        a = dict(ha.members(bp))
        b = dict(ha.members(np_))
        dm = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        if dm:
            out.append((evt, dm))
    return out


def sentinel_counts(root, evt):
    """(nlines, sum_u, sum_v, sum_w) of sentinel deweights in the arm's log."""
    hits = [0, 0, 0, 0]
    for log in glob.glob(os.path.join(root, 'pr_evt%d' % evt, 'wct_pr_evt%d.log' % evt)):
        with open(log, errors='replace') as f:
            for line in f:
                m = SENTINEL.search(line)
                if m:
                    hits[0] += 1
                    for i in range(3):
                        hits[i + 1] += int(m.group(i + 1))
    return hits


def fit_vs_image(root, evt):
    """{real_cluster_id: max over fit points of distance (cm) to the nearest
    raw-image point of the same cluster}."""
    zp = os.path.join(root, 'pr_evt%d' % evt, 'mabc-pr.zip')
    with zipfile.ZipFile(zp) as z:
        tf = json.loads(z.read('data/0/0-track_fit-global.json'))
        cg = json.loads(z.read('data/0/0-clustering-global.json'))
    fx, fy, fz = (np.array(tf[k]) for k in 'xyz')
    fr = np.array(tf['real_cluster_id']) // 1000  # 20000 -> 20
    ix, iy, iz = (np.array(cg[k]) for k in 'xyz')
    ir = np.array(cg['real_cluster_id'])
    out = {}
    for cid in sorted(set(fr.tolist())):
        fm = fr == cid
        im = ir == cid
        if not im.any() or not fm.any():
            continue
        P = np.stack([fx[fm], fy[fm], fz[fm]], axis=1)
        Q = np.stack([ix[im], iy[im], iz[im]], axis=1)
        # chunked brute force; both sets are O(1e3-1e4)
        dmax = 0.0
        for i in range(0, len(P), 256):
            d = np.sqrt(((P[i:i+256, None, :] - Q[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            dmax = max(dmax, float(d.max()))
        out[cid] = dmax
    return out


def main():
    mv = movers(BASE, NEW)
    print('base:', BASE)
    print('new :', NEW)
    print('movers: %d' % len(mv))
    for evt, dm in mv:
        nl, du, dv, dw = sentinel_counts(NEW, evt)
        b = fit_vs_image(BASE, evt)
        n = fit_vs_image(NEW, evt)
        print('\n=== evt %d ===' % evt)
        print('  members changed: %s' % dm)
        print('  sentinel: %d fit points deweighted cells (u=%d v=%d w=%d)' % (nl, du, dv, dw))
        cids = sorted(set(b) | set(n))
        for cid in cids:
            db, dn = b.get(cid, float('nan')), n.get(cid, float('nan'))
            tag = ''
            if not (np.isnan(db) or np.isnan(dn)):
                if dn < db - 0.05:
                    tag = '  IMPROVED'
                elif dn > db + 0.05:
                    tag = '  WORSE'
            print('  cid %-4d fit-vs-image max: %.2f -> %.2f cm%s' % (cid, db, dn, tag))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""doc pdvd/40 round 2 -- is the fabricated-Steiner-point defect PDVD-only, or
does SBND have it too?

Measures, per detector, the fraction of a cluster's `steiner_pc` points that
have NO live 3D point within N cm -- the doc pdvd/40 statistic -- but from the
PrDisplayDump calib JSON rather than the Bee steiner layer, because:

  * the calib `steiner` block carries EVERY cluster that has a steiner_pc
    (87 on PDVD evt 298595) while the Bee layer carries only the STM-fitted or
    STM-tagged subset (9-15), and
  * SBND dumps no Steiner Bee layer at all, so the calib file is the only
    cross-detector source.  No detector config is touched by this script.

Two controls, both printed, both must pass before any number here is read:

  FRAME   `flag_terminal` marks the charge-selected Steiner terminals, which by
          construction sit ON real charge.  Their distance to the nearest live
          point is the probe that the calib block and the Bee clustering layer
          share a coordinate frame.  Median ~0 => they do.
  ROUTE   on PDVD, where both sources exist, the calib `steiner` points and the
          Bee `steiner_graph` layer must be the SAME points (--route-control).

**The confound this script exists to control.** SBND arms are neutrino MC and
PDVD arms are cosmics; a 38 599-point cosmic crossing two drift volumes has
gaps a 2 000-point neutrino cluster never has, so a raw detector-vs-detector
fraction measures the event sample, not the code.  Every table is therefore
also reported CONDITIONED on the cluster being multi-component at a 6 cm link
-- the population in which the mechanism (a Steiner path bridging a gap that
has no charge) can fire at all.

Usage:
    d40_steiner_void_xdet.py [--route-control] \
        PDVD:'/path/pdvd/work/*_d40nu' \
        SBND:'/path/sbnd_xin/work-ncpi0-doc25d38new/pr_evt*'

Each directory must hold calib-pr-evt*.json and mabc-pr.zip from the same run.
"""
import argparse, glob, json, os, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SENTINEL_CM = 1e4   # QLMatching sentinel-T0 points, see d40_steiner_void_census.py
LINK_CM = 6.0
FAR = (3, 10, 30)


def read_event(d):
    cal = glob.glob(os.path.join(d, 'calib-pr-evt*.json'))
    zps = glob.glob(os.path.join(d, 'mabc-pr.zip'))
    if not cal or not zps:
        return None
    st = json.load(open(cal[0])).get('steiner', [])
    if not st:
        return None
    z = zipfile.ZipFile(zps[0])
    keys = [k for k in z.namelist() if k.endswith('clustering-global.json')]
    if not keys:
        return None
    j = json.loads(z.read(keys[0]))
    cx = np.array(j['x'], float); cy = np.array(j['y'], float); cz = np.array(j['z'], float)
    cc = np.array(j['cluster_id'])
    m = np.abs(cx) < SENTINEL_CM
    return {'st': st, 'C': np.stack([cx[m], cy[m], cz[m]], 1), 'cc': cc[m], 'z': z, 'dir': d}


def ncomp(P):
    if len(P) < 2:
        return 1
    pr = cKDTree(P).query_pairs(LINK_CM, output_type='ndarray')
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P), len(P)))
    return connected_components(g, directed=False)[0]


def census(tag, dirs, route_control=False):
    rows = []          # per cluster: (ncomp, nsteiner, [far counts], nlive)
    term_d = []
    nev = 0
    for d in sorted(dirs):
        ev = read_event(d)
        if ev is None:
            continue
        nev += 1
        T = cKDTree(ev['C'])
        for e in ev['st']:
            S = np.stack([np.array(e['x'], float), np.array(e['y'], float),
                          np.array(e['z'], float)], 1)
            if not len(S):
                continue
            cid = e['cluster_id']
            P = ev['C'][ev['cc'] == cid]
            d_any, _ = T.query(S)
            F = np.array(e.get('flag_terminal', []), bool)
            if F.size == len(S) and F.any():
                term_d.append(d_any[F])
            rows.append((ncomp(P) if len(P) else 0, len(S),
                         [int((d_any > t).sum()) for t in FAR], len(P)))
        if route_control and 'data/0/0-steiner_graph-global.json' in ev['z'].namelist():
            j = json.loads(ev['z'].read('data/0/0-steiner_graph-global.json'))
            bx = np.array(j['x']); by = np.array(j['y']); bz = np.array(j['z'])
            bc = np.array(j['cluster_id'])
            byid = {x['cluster_id']: x for x in ev['st']}
            worst = 0.0; nsh = 0
            for cid in sorted(set(bc.tolist())):
                e = byid.get(cid)
                if e is None:
                    continue
                B = np.stack([bx[bc == cid], by[bc == cid], bz[bc == cid]], 1)
                A = np.stack([np.array(e['x']), np.array(e['y']), np.array(e['z'])], 1)
                dd, _ = cKDTree(A).query(B)
                worst = max(worst, float(dd.max())); nsh += 1
            print('   ROUTE CONTROL %s: %d shared clusters, worst bee->calib %.5f cm'
                  % (os.path.basename(d), nsh, worst))

    if not rows:
        print('== %-6s no usable events' % tag); return
    nc = np.array([r[0] for r in rows])
    ns = np.array([r[1] for r in rows])
    far = np.array([r[2] for r in rows])
    td = np.concatenate(term_d) if term_d else np.array([0.0])
    print('== %-6s %d events, %d clusters with a steiner_pc, %d steiner points'
          % (tag, nev, len(rows), ns.sum()))
    print('   FRAME CONTROL terminals (n=%d): median %.3f  p90 %.3f  max %.2f cm'
          % (len(td), np.median(td), np.percentile(td, 90), td.max()))
    def block(label, sel):
        if not sel.any():
            print('   %-28s (no clusters)' % label); return
        n = ns[sel].sum()
        print('   %-28s %4d clusters, %7d steiner pts' % (label, sel.sum(), n), end='')
        for i, t in enumerate(FAR):
            print('  >%2dcm %6d (%5.2f%%)' % (t, far[sel, i].sum(),
                                              100. * far[sel, i].sum() / n), end='')
        print()
    block('ALL clusters', np.ones(len(rows), bool))
    block('1-component clusters', nc <= 1)
    block('multi-component (>=2)', nc >= 2)
    block('  of those, >=5 comps', nc >= 5)
    frac = 100. * (nc >= 2).sum() / len(nc)
    print('   EXPOSURE: %d/%d clusters (%.1f%%) are multi-component at %.0f cm; '
          'component counts p50 %d p90 %d max %d'
          % ((nc >= 2).sum(), len(nc), frac, LINK_CM,
             int(np.median(nc)), int(np.percentile(nc, 90)), int(nc.max())))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--route-control', action='store_true')
    ap.add_argument('specs', nargs='+', help='DET:glob')
    a = ap.parse_args(argv[1:])
    for spec in a.specs:
        tag, pat = spec.split(':', 1)
        census(tag, glob.glob(pat), a.route_control)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

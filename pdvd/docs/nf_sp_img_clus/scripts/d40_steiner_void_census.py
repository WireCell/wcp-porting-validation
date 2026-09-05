#!/usr/bin/env python3
"""doc pdvd/40 -- Steiner-graph points that sit where there is no 3D image.

The owner pointed at two points in the doc pdvd/39 round-2 Bee set for
039252/2 (art event 298595) that have "clearly no image associated with them":

    P1 = (39.0, 75.7, 201.8)      P2 = (275.5, 12.1, 8.8)     [cm]

This script measures, from the Bee zips only:

  frame   -- the PREMISE.  Every claim below is a distance between the
             steiner_graph layer and the clustering layer, so the two must
             share a coordinate frame.  steiner_terminals is a charge-selected
             SUBSET of the Steiner cloud, so its distance to the nearest
             clustering point is the frame probe: median ~0 => frames agree.
             A systematic offset here would mean the "fabricated" points are a
             display artifact and every other number is meaningless.
  probe   -- where each named point lands in each arm's steiner_graph.
  void    -- per arm: Steiner points further than N cm from ANY live 3D point
             of ANY cluster, event-wide and per cluster.
  groups  -- the far points of one cluster grouped at 3 cm, with the live
             connected components (6 cm link) each group sits between.
  null    -- chance floor for the "(y,z) matches a live point" statistic.
             In a busy (y,z) window a 2 cm transverse match is nearly
             guaranteed; without this the statistic reads as evidence when it
             is not.  Reported so the doc can say the observation is at chance.
  sentinel-- live points at |x| > 1e4 cm.  QLMatching stamps every cluster with
             cluster_t0 = -1e12 (QLMatching.cxx:1351) and overwrites it only
             for flash-matched clusters; an unmatched cluster keeps the
             sentinel and its x_t0cor lands ~1.48e8 cm away (drift speed
             1.48073 mm/us x 1e12 ns).  Those points are in the clustering
             layer the owner looks at.

Usage:
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    docs/nf_sp_img_clus/scripts/d40_steiner_void_census.py \
        work/039252_2_d40rep work/039252_2_d40aniso0

Every arm dir must hold mabc-pr.zip.  The first is the reference.
"""
import zipfile, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PROBES = {'P1': (39.0, 75.7, 201.8), 'P2': (275.5, 12.1, 8.8)}
SENTINEL_CM = 1e4      # |x| beyond this is not a detector coordinate
FAR = (3, 5, 10, 20, 30)
LINK_CM = 6.0          # live connected-component link, as in d39_unmerge_census
GRP_CM = 3.0           # grouping of the far Steiner points themselves


def layers(zp):
    z = zipfile.ZipFile(zp)
    out = {}
    for n in z.namelist():
        if not n.endswith('-global.json'):
            continue
        name = os.path.basename(n)[2:-len('-global.json')]
        d = json.loads(z.read(n))
        x = np.array(d['x'], float); y = np.array(d['y'], float); zz = np.array(d['z'], float)
        cid = np.array(d.get('cluster_id', [-1] * len(x)))
        out[name] = (x, y, zz, cid)
    return out


def sane(t):
    """Drop the sentinel-T0 points: they are not detector coordinates and a
    kd-tree built with them still answers correctly, but every bbox does not."""
    x, y, z, c = t
    m = np.abs(x) < SENTINEL_CM
    return np.stack([x[m], y[m], z[m]], 1), c[m]


def report_arm(d, ref_far=None):
    tag = os.path.basename(d.rstrip('/'))
    L = layers(os.path.join(d, 'mabc-pr.zip'))
    if 'clustering' not in L:
        print('%-28s no clustering layer' % tag); return None
    C, cc = sane(L['clustering'])
    T = cKDTree(C)
    print('=' * 78)
    print('ARM %s   layers: %s' % (tag, ', '.join(sorted(L))))
    print('  live %d pts in %d clusters (sentinel-x dropped: %d)'
          % (len(C), len(set(cc.tolist())),
             (np.abs(L['clustering'][0]) >= SENTINEL_CM).sum()))

    # ---- frame check (the premise) -------------------------------------
    print('  -- frame check: distance from each layer to the nearest clustering point')
    for name in ('steiner_terminals', 'stm', 'stm_tagged', 'stm_fit', 'steiner_graph'):
        if name not in L: continue
        P, _ = sane(L[name])
        if not len(P): continue
        dd, _ = T.query(P)
        print('     %-18s n=%6d  median %6.3f  p90 %6.3f  max %8.2f cm'
              % (name, len(P), np.median(dd), np.percentile(dd, 90), dd.max()))
    print('     (steiner_terminals is a subset of the live cloud by construction;')
    print('      its median is the frame probe -- ~0 means the frames agree)')

    if 'steiner_graph' not in L:
        print('  -- no steiner_graph layer in this arm'); return None
    S, sc = sane(L['steiner_graph'])
    ST = cKDTree(S)
    for pname, p in PROBES.items():
        dd, ii = ST.query(np.array(p))
        near = 'cid=%s at (%.2f,%.2f,%.2f)' % (sc[ii], S[ii][0], S[ii][1], S[ii][2])
        print('  -- probe %s %-22s nearest steiner pt %7.3f cm  %s'
              % (pname, str(p), dd, near if dd < 1 else '(NOT in this layer)'))

    d_any, _ = T.query(S)
    print('  -- void census: steiner pts with no live 3D point within N cm')
    print('     %6d steiner pts in %d clusters' % (len(S), len(set(sc.tolist()))))
    for t in FAR:
        print('     > %2d cm : %5d (%5.2f%%)' % (t, (d_any > t).sum(), 100. * (d_any > t).mean()))
    rows = []
    for cid in sorted(set(sc.tolist())):
        m = sc == cid
        rows.append((cid, int(m.sum()), int((d_any[m] > 10).sum()),
                     100. * (d_any[m] > 10).mean(), float(d_any[m].max())))
    rows.sort(key=lambda r: -r[2])
    print('     per cluster (only those with any point > 10 cm):')
    print('        cid  nSteiner   nFar>10   pctFar   maxDist')
    for r in rows:
        if r[2] == 0: break
        print('        %4d %9d %9d %7.1f%% %9.1f' % r)
    return {'far10': int((d_any > 10).sum()), 'n': len(S),
            'per': {r[0]: r[2] for r in rows}, 'C': C, 'cc': cc, 'S': S, 'sc': sc}


def groups(state, cid):
    """Group one cluster's far Steiner points and name the live components they
    sit between.  A group whose ends are two different components is a bridge
    across a gap the cluster genuinely has; a group with one end is a spur."""
    C, cc, S, sc = state['C'], state['cc'], state['S'], state['sc']
    P = C[cc == cid]; Q = S[sc == cid]
    if not len(P) or not len(Q): return
    T = cKDTree(C); d, _ = T.query(Q); F = Q[d > 10]
    Tp = cKDTree(P)
    pr = Tp.query_pairs(LINK_CM, output_type='ndarray')
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P), len(P)))
    nc, lab = connected_components(g, directed=False)
    print('  cluster %d: %d live pts / %d components at %.1f cm ; %d far steiner pts'
          % (cid, len(P), nc, LINK_CM, len(F)))
    if not len(F): return
    fp = cKDTree(F).query_pairs(GRP_CM, output_type='ndarray')
    gf = coo_matrix((np.ones(len(fp)), (fp[:, 0], fp[:, 1])), shape=(len(F), len(F)))
    nf, flab = connected_components(gf, directed=False)
    for k in range(nf):
        G = F[flab == k]
        if len(G) < 3: continue
        dd, ii = Tp.query(G)
        print('     grp %d n=%4d span=%5.1f cm  x[%7.1f,%7.1f] y[%7.1f,%7.1f] z[%6.1f,%6.1f]'
              '  between live comps %s  worst %.1f cm'
              % (k, len(G), float(np.linalg.norm(G.max(0) - G.min(0))),
                 G[:, 0].min(), G[:, 0].max(), G[:, 1].min(), G[:, 1].max(),
                 G[:, 2].min(), G[:, 2].max(), sorted(set(lab[ii].tolist())), dd.max()))


def null_floor(state, cid, seed=7):
    """A (y,z) match to a live point is only evidence if the same window does
    not hand it to random points too."""
    C, S, sc = state['C'], state['S'], state['sc']
    T = cKDTree(C); Q = S[sc == cid]
    d, _ = T.query(Q); F = Q[d > 10]
    if len(F) < 3: return
    T2 = cKDTree(C[:, 1:3])
    d2, _ = T2.query(F[:, 1:3]); rate = float((d2 < 2.0).mean())
    lo, hi = F[:, 1:3].min(0), F[:, 1:3].max(0)
    R = np.random.default_rng(seed).uniform(lo, hi, size=(20000, 2))
    dn, _ = T2.query(R); null = float((dn < 2.0).mean())
    print('  cluster %3d: %4d far pts; (y,z)-match<2cm = %.3f   NULL(uniform, same bbox) = %.3f'
          % (cid, len(F), rate, null))


def sentinel(d):
    L = layers(os.path.join(d, 'mabc-pr.zip'))
    x, y, z, c = L['clustering']
    bad = np.abs(x) >= SENTINEL_CM
    print('  %-28s %d/%d live pts at |x|>%g cm, in %d clusters'
          % (os.path.basename(d.rstrip('/')), bad.sum(), len(x), SENTINEL_CM,
             len(set(c[bad].tolist()))))
    if not bad.sum(): return
    mixed = sum(1 for cid in set(c[bad].tolist()) if ((c == cid) & ~bad).sum())
    print('     distinct |x| values: %s' % sorted(set(np.abs(np.round(x[bad], 0)).tolist())))
    print('     clusters with SOME sentinel and SOME sane points: %d (0 = the whole cluster moves)' % mixed)
    print('     sign split: x>0 %d, x<0 %d (the two drift directions)'
          % ((x[bad] > 0).sum(), (x[bad] < 0).sum()))
    sz = sorted(((c == cid).sum() for cid in set(c[bad].tolist())))
    print('     affected cluster sizes: min %d median %d max %d' % (sz[0], sz[len(sz)//2], sz[-1]))


def main(argv):
    dirs = argv[1:]
    if not dirs: raise SystemExit(__doc__)
    states = {}
    for d in dirs:
        st = report_arm(d)
        if st: states[os.path.basename(d.rstrip('/'))] = st
    print()
    print('=' * 78)
    print('FAR-POINT GROUPS (reference arm, then every other arm)')
    for tag, st in states.items():
        print('-- %s' % tag)
        for cid in [c for c, n in st['per'].items() if n]:
            groups(st, cid)
    print()
    print('=' * 78)
    print('CHANCE FLOOR for the "(y,z) matches live charge" statistic')
    for tag, st in states.items():
        print('-- %s' % tag)
        for cid in [c for c, n in st['per'].items() if n]:
            null_floor(st, cid)
    print()
    print('=' * 78)
    print('SENTINEL-T0 LIVE POINTS (QLMatching.cxx:1351 cluster_t0 = -1e12)')
    for d in dirs:
        sentinel(d)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

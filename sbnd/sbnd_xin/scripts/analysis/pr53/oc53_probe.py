#!/usr/bin/env python3
"""doc pr/53 scoping: classify an owner-flagged "overclustering" point pair
(x,y,z),(x,y,z) and measure whether the toolkit's overclustering-protection
path test (Graphs::connect_graph_relaxed / test_good_point,
clus/src/connect_graph_relaxed.cxx, Facade_Grouping.cxx) would have rejected
the bridge between them, using the EVENT'S OWN ctpc_a<A>f0p<U|V|W> and
dead_winds_a<A>f0p<U|V|W> point clouds pulled from its pctree archive -- not a
generic radius search on img-global.

Two questions this answers, per pair:
  1. Family A ("graph") vs Family B ("clustering_isolated"): does the far
     endpoint sit in a *different* associated (non-main) sub-cluster written
     by ClusteringIsolated's save_assoc_id ("assoc_cluster_id"/"assoc_cluster_main",
     perblob PC)?  If so the merge is clustering_isolated's angle-less 80 cm
     small->big absorb (clus/src/clustering_isolated.cxx:297,325), which runs
     AFTER protect_overclustering (cfg/pgrapher/experiment/sbnd/clus.jsonnet:300
     vs :320) and is untouched by any graph-tightening fix.
  2. For family A: the true closest-approach point pair between the two
     img-global proximity components containing A and B (NOT A/B themselves,
     which are usually deep inside their component), replayed exactly through
     test_good_point's per-1cm-step scoring using this event's real ctpc/dead
     arrays, reporting num_bad[0..3] / num_bad1[0] / branch (strong / parallel /
     prolonged-U/V/W) / whether connect_graph_relaxed's
     "num_bad>7 || (num_bad>2 && num_bad>=0.75*num_steps)" rejection test fires.
  3. A direct connect_graph_closely check (overlap_fast, offset=2 wires, on the
     ACTUAL blob pair nearest A and B) confirming the two endpoints are not
     already linked by the short-range wire-overlap graph before the bridging
     test is even reached.

Usage:
    oc53_probe.py <ql_evt_dir> <Ax> <Ay> <Az> <Bx> <By> <Bz> [label]

<ql_evt_dir> is a QL-stage event directory containing mabc-all-apa.zip and
pctree-evt<N>.tar.gz (e.g. work-ncpi0-cb0805/ql_evt21073). Coordinates in cm.

Repro (doc pr/53):
    cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
    python3 ../wcp-porting-img/sbnd/sbnd_xin/scripts/analysis/pr53/oc53_probe.py \\
        work-ncpi0-cb0805/ql_evt21073 -36.5 32.0 367.5 -31.2 28.8 369.6 18345-21073
"""
import sys, os, json, math, zipfile, tarfile, tempfile, glob, shutil

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ---- SBND face-0 plane pitch angles (from wire-cell-data/sbnd-wires-geometry-v0206):
# U at +60 deg, V at -60 deg, W at 0 deg from +y, all measured toward +z.
_S, _C = math.sin(math.radians(60)), math.cos(math.radians(60))
def _y2d(pi, y, z):
    """2D pitch coordinate matching Grouping::get_closest_points's
    y = cos(angle)*z - sin(angle)*y convention, calibrated per-plane against
    the event's own ctpc arrays at load time (see Loader.calibrate)."""
    cy, cz = ((-_S, _C), (_S, _C), (0.0, 1.0))[pi]
    return cy * y + cz * z


class Loader:
    """Extracts one event's pctree tar.gz and indexes every pcarray by
    datapath, then builds the k-d trees / dead-wind lookups needed to
    reproduce Facade::Grouping::test_good_point / is_good_point exactly."""

    def __init__(self, ql_evt_dir):
        self.dir = ql_evt_dir
        base = os.path.basename(ql_evt_dir.rstrip('/'))
        evt = base.replace('ql_evt', '').replace('pr_evt', '')
        self.evt = evt
        # QL-stage dirs (ql_evt<N>) hold pctree-evt<N>.tar.gz; PR-stage dirs
        # (pr_evt<N>, round 7) hold pctree-pr-evt<N>.tar.gz -- same schema
        # (pointtrees/<evt>/live/{3d,scalar,perblob,ctpc_*,dead_*,steiner_pc,...}),
        # just a different filename. Try both, QL name first (existing callers).
        for cand in ('pctree-evt%s.tar.gz' % evt, 'pctree-pr-evt%s.tar.gz' % evt):
            tgz = os.path.join(ql_evt_dir, cand)
            if os.path.exists(tgz):
                break
        else:
            raise FileNotFoundError('no pctree-{,pr-}evt%s.tar.gz under %s' % (evt, ql_evt_dir))
        self.tmp = tempfile.mkdtemp(prefix='oc53_')
        with tarfile.open(tgz) as t:
            t.extractall(self.tmp)
        self.idx = {}
        for f in glob.glob(os.path.join(self.tmp, '*_metadata.json')):
            m = json.load(open(f))
            if m.get('datatype') == 'pcarray':
                self.idx[m['datapath']] = f.replace('_metadata.json', '_array.npy')
        self.pre = 'pointtrees/%s/live/' % evt
        self.tree, self.wfit, self.dead, self.gap = {}, {}, {}, {}
        for a in (0, 1):
            for pi, pl in enumerate('UVW'):
                b = 'pointclouds/namedpcs/ctpc_a%df0p%s/arrays/' % (a, pl)
                x, y, w = self.A(b + 'x'), self.A(b + 'y'), self.A(b + 'wind')
                self.tree[(a, pi)] = cKDTree(np.c_[x, y])
                self.wfit[(a, pi)] = np.polyfit(y, w, 1)  # wind = m*y2d + c, exact (mm units)
                self.dead[(a, pi)] = self._winds(a, pl)
            self.gap[a] = self._gap(a)

    def A(self, path):
        return np.load(self.idx[self.pre + path])

    def _winds(self, a, pl):
        try:
            b = 'pointclouds/namedpcs/dead_winds_a%df0p%s/arrays/' % (a, pl)
            return dict(zip(self.A(b + 'wind').tolist(),
                             zip(self.A(b + 'xbeg').tolist(), self.A(b + 'xend').tolist())))
        except KeyError:
            return {}

    def _gap(self, a):
        try:
            b = 'pointclouds/namedpcs/dead_gap_a%df0pW/arrays/' % a
            return dict(zip(self.A(b + 'wind').tolist(),
                             zip(self.A(b + 'xbeg').tolist(), self.A(b + 'xend').tolist())))
        except KeyError:
            return {}

    def wind(self, apa, pi, y2d_mm):
        f = self.wfit[(apa, pi)]
        return int(round(np.polyval(f, y2d_mm)))

    def scores(self, pt_cm, ch_range=1, radius_cm=0.6):
        """pt_cm: (x,y,z) in cm. Returns test_good_point's 6-vector
        [liveU,liveV,liveW,deadU,deadV,deadW]. Reproduces
        Facade_Grouping.cxx:585-615, including the in_dead_gap short-circuit."""
        x, y, z = (v * 10.0 for v in pt_cm)  # cm -> mm (ctpc/wind arrays are mm)
        apa = 0 if x < 0 else 1
        wg = self.wind(apa, 2, _y2d(2, y, z))
        for ch in range(wg - ch_range, wg + ch_range + 1):
            if ch in self.gap[apa]:
                lo, hi = self.gap[apa][ch]
                if lo <= x <= hi:
                    return [0, 0, 0, 1, 1, 1]
        out = [0] * 6
        r = radius_cm * 10.0
        for pi in range(3):
            c = _y2d(pi, y, z)
            if len(self.tree[(apa, pi)].query_ball_point([x, c], r)) > 0:
                out[pi] = 1
            else:
                w = self.wind(apa, pi, c)
                for ch in range(w - ch_range, w + ch_range + 1):
                    if ch in self.dead[(apa, pi)]:
                        lo, hi = self.dead[(apa, pi)][ch]
                        if lo <= x <= hi:
                            out[pi + 3] = 1
                            break
        return out

    def img_closest_dis(self, pt_cm):
        """3D distance (cm) from pt_cm to the nearest '3d' image point in this
        cluster's pctree -- the Python analogue of the C++'s
        Simple3DPointCloud::get_closest_dis(test_p) over the union of all
        closely-components' point clouds (round 7 S5 candidate). Cached
        cKDTree built lazily over the SAME '3d' PC blob_scalars()/assoc() use
        (mm->cm), so 'has 3D image support' here matches what the graph
        builder's own pt_clouds see -- not img-global (post-clustering,
        wrong stage) and not a generic radius search."""
        if not hasattr(self, '_img3d_tree'):
            g = 'pointclouds/namedpcs/3d/arrays/'
            P = np.c_[self.A(g + 'x'), self.A(g + 'y'), self.A(g + 'z')] / 10.0  # mm->cm
            self._img3d_tree = cKDTree(P) if len(P) else None
        if self._img3d_tree is None:
            return 1e9
        d, _ = self._img3d_tree.query(np.array(pt_cm, float))
        return float(d)

    def blob_scalars(self, target_pt_cm):
        """Return the scalar PC row for the blob nearest target_pt_cm, plus
        the 3d-point array/blob-index map (for component/neck searches)."""
        g = 'pointclouds/namedpcs/3d/arrays/'
        P = np.c_[self.A(g + 'x'), self.A(g + 'y'), self.A(g + 'z')] / 10.0  # mm->cm
        n = self.A('pointclouds/namedpcs/scalar/arrays/npoints')
        blob_of_point = np.repeat(np.arange(len(n)), n)
        i = int(np.argmin(np.linalg.norm(P - np.array(target_pt_cm), axis=1)))
        j = int(blob_of_point[i])
        fields = ['u_wire_index_min', 'u_wire_index_max', 'v_wire_index_min', 'v_wire_index_max',
                  'w_wire_index_min', 'w_wire_index_max', 'slice_index_min', 'slice_index_max', 'wpid']
        row = {f: int(self.A('pointclouds/namedpcs/scalar/arrays/' + f)[j]) for f in fields}
        row['blob'] = j
        row['pt_actual'] = P[i]
        return row, P, blob_of_point

    def assoc(self, target_pt_cm):
        """perblob assoc_cluster_id/assoc_cluster_main at the blob nearest
        target_pt_cm -- the ClusteringIsolated::save_assoc_id provenance."""
        g = 'pointclouds/namedpcs/3d/arrays/'
        P = np.c_[self.A(g + 'x'), self.A(g + 'y'), self.A(g + 'z')] / 10.0
        n = self.A('pointclouds/namedpcs/scalar/arrays/npoints')
        blob_of_point = np.repeat(np.arange(len(n)), n)
        i = int(np.argmin(np.linalg.norm(P - np.array(target_pt_cm), axis=1)))
        j = int(blob_of_point[i])
        aid = self.A('pointclouds/namedpcs/perblob/arrays/assoc_cluster_id')[j]
        amain = self.A('pointclouds/namedpcs/perblob/arrays/assoc_cluster_main')[j]
        rid = self.A('pointclouds/namedpcs/perblob/arrays/real_cluster_id')[j]
        return int(aid), int(amain), int(rid)

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def overlap_fast(b1, b2, offset=2):
    if b1['wpid'] != b2['wpid']:
        return False, [('wpid', b1['wpid'], b2['wpid'])]
    checks = []
    ok = True
    for pl in ('u', 'v', 'w'):
        lo1, hi1 = b1[pl + '_wire_index_min'], b1[pl + '_wire_index_max']
        lo2, hi2 = b2[pl + '_wire_index_min'], b2[pl + '_wire_index_max']
        good = not (lo1 > hi2 - 1 + offset or lo2 > hi1 - 1 + offset)
        checks.append((pl, lo1, hi1, lo2, hi2, good))
        ok = ok and good
    return ok, checks


def img_load(ql_evt_dir):
    evt = os.path.basename(ql_evt_dir).replace('ql_evt', '')
    z = zipfile.ZipFile(os.path.join(ql_evt_dir, 'mabc-all-apa.zip'))
    d = json.loads(z.read('data/0/0-img-global.json'))
    return np.c_[d['x'], d['y'], d['z']], np.array(d['cluster_id'])


def proximity_components(P, C, ia, radius=1.5):
    """1.5cm-radius connected components within the img-global cluster
    containing point ia. This is a PROXY for connect_graph_closely (which
    links on wire-index intervals, not a metric radius) -- used only to find
    candidate proximity groups for the true closest-approach search below;
    the direct overlap_fast() check on the actual blob pair is the exact
    test and is what the doc cites for the close-graph claim."""
    m = np.where(C == C[ia])[0]
    X = P[m]
    t = cKDTree(X)
    pr = np.array(list(t.query_pairs(radius)))
    if len(pr) == 0:
        return m, X, np.zeros(len(X), int)
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
    _, lab = connected_components(g, directed=False)
    return m, X, lab


def true_neck(X, lab, ja, jb):
    """Closest-approach point pair between the proximity components
    containing indices ja and jb (into X/lab). Returns None if already the
    same component."""
    if lab[ja] == lab[jb]:
        return None
    Qa, Qb = X[lab == lab[ja]], X[lab == lab[jb]]
    ta = cKDTree(Qa)
    dd, ii = ta.query(Qb)
    kb = int(np.argmin(dd))
    ka = int(ii[kb])
    return Qa[ka], Qb[kb], float(dd[kb]), len(Qa), len(Qb)


def angles(p1, p2):
    d = np.array(p2, float) - np.array(p1, float)
    dyz = math.hypot(d[1], d[2])
    du, dv, dw = (_y2d(pi, d[1], d[2]) for pi in range(3))
    a1 = math.degrees(math.atan2(abs(du), abs(d[0]))) if d[0] else 90.0
    a2 = math.degrees(math.atan2(abs(dv), abs(d[0]))) if d[0] else 90.0
    a1p = math.degrees(math.atan2(abs(dw), abs(d[0]))) if d[0] else 90.0
    a3 = math.degrees(math.atan2(dyz, abs(d[0]))) if d[0] else 90.0
    return a1, a2, a1p, a3


def walk_and_score(ld, p1, p2, step_cm=1.0):
    """Exact replay of connect_graph_relaxed.cxx's per-edge path test
    (lines ~150-303): 1cm steps from p1 to p2 (inclusive of p2, matching
    `(ii+1)/num_steps`), test_good_point at each, num_bad[]/num_bad1[]
    accumulation, branch selection, and the final invalidate_distance() test."""
    p1 = np.array(p1, float)
    p2 = np.array(p2, float)
    dis = np.linalg.norm(p2 - p1)
    nst = int(dis / step_cm) + 1
    nb = [0, 0, 0, 0]
    nb1 = [0, 0, 0, 0]
    rows = []
    for ii in range(nst):
        tp = p1 + (p2 - p1) / nst * (ii + 1)
        s = ld.scores(tp)
        if s[0] + s[3] + s[1] + s[4] + (s[2] + s[5]) * 2 < 3:
            nb[0] += 1
        if s[0] + s[3] == 0:
            nb[1] += 1
        if s[1] + s[4] == 0:
            nb[2] += 1
        if s[2] + s[5] == 0:
            nb[3] += 1
        if s[0] + s[3] + s[1] + s[4] + (s[2] + s[5]) < 3:
            nb1[0] += 1
        rows.append(s)

    a1, a2, a1p, a3 = angles(p1, p2)
    wire_tol, perp_tol, perp = 12.5, 10.0, 90.0
    strong = True
    if abs(a3 - perp) < perp_tol:
        strong = None  # would need vhough_transform at 15cm to resolve; not replayed here
    elif a1 < wire_tol or a2 < wire_tol or a1p < wire_tol:
        strong = False

    def exceeds(v, ratio=0.75):
        return v >= ratio * nst

    if strong is not False and strong is not None:
        branch = 'strong'
        kill = nb1[0] > 7 or (nb1[0] > 2 and exceeds(nb1[0]))
    elif strong is None:
        branch = 'isochronous(needs-vhough, treated as strong)'
        kill = nb1[0] > 7 or (nb1[0] > 2 and exceeds(nb1[0]))
    else:
        parallel = (a1 < wire_tol and a2 < wire_tol) or (a1p < wire_tol and a1 < wire_tol) or (a1p < wire_tol and a2 < wire_tol)
        if parallel:
            branch = 'parallel'
            kill = nb[0] > 7 or (nb[0] > 2 and exceeds(nb[0]))
        elif a1 < wire_tol:
            branch = 'prolonged-U'
            sb = nb[2] + nb[3]
            kill = sb > 9 or (sb > 2 and exceeds(sb)) or nb[3] >= 3
        elif a2 < wire_tol:
            branch = 'prolonged-V'
            sb = nb[1] + nb[3]
            kill = sb > 9 or (sb > 2 and exceeds(sb)) or nb[3] >= 3
        elif a1p < wire_tol:
            branch = 'prolonged-W'
            sb = nb[2] + nb[1]
            kill = sb > 9 or (sb > 2 and exceeds(sb))
        else:
            branch = 'strong-fallthrough'
            kill = nb[0] > 7 or (nb[0] > 2 and exceeds(nb[0]))

    return dict(dis=dis, nst=nst, nb=nb, nb1=nb1, angles=(a1, a2, a1p, a3),
                branch=branch, kill=kill, rows=rows)


def classify(ld, A, B):
    aidA, amainA, ridA = ld.assoc(A)
    aidB, amainB, ridB = ld.assoc(B)
    same_real = ridA == ridB
    family = 'B-clustering_isolated' if (aidA != aidB and (amainA == 0 or amainB == 0)) else 'A-graph'
    return dict(aidA=aidA, amainA=amainA, ridA=ridA, aidB=aidB, amainB=amainB, ridB=ridB,
                same_real_cluster=same_real, family=family)


def main():
    if len(sys.argv) < 8:
        print(__doc__)
        sys.exit(1)
    ql_dir = sys.argv[1]
    A = tuple(float(v) for v in sys.argv[2:5])
    B = tuple(float(v) for v in sys.argv[5:8])
    label = sys.argv[8] if len(sys.argv) > 8 else os.path.basename(ql_dir)

    P, C = img_load(ql_dir)
    ia = int(np.argmin(np.linalg.norm(P - np.array(A), axis=1)))
    ib = int(np.argmin(np.linalg.norm(P - np.array(B), axis=1)))
    print('=== %s ===' % label)
    print('A=%s -> img cid=%d d=%.2f cm (img pt %s)' % (A, C[ia], np.linalg.norm(P[ia] - A), P[ia]))
    print('B=%s -> img cid=%d d=%.2f cm (img pt %s)' % (B, C[ib], np.linalg.norm(P[ib] - B), P[ib]))
    if C[ia] != C[ib]:
        print('NOT SAME CLUSTER -- this pair is not overclustering under current-HEAD output. Stop.')
        return

    ld = Loader(ql_dir)
    try:
        cl = classify(ld, A, B)
        print('classification: %s  (A: assoc_id=%d assoc_main=%d real_id=%d | B: assoc_id=%d assoc_main=%d real_id=%d)'
              % (cl['family'], cl['aidA'], cl['amainA'], cl['ridA'], cl['aidB'], cl['amainB'], cl['ridB']))

        bA, _, _ = ld.blob_scalars(A)
        bB, _, _ = ld.blob_scalars(B)
        ok, checks = overlap_fast(bA, bB, offset=2)
        print('direct blob overlap_fast(offset=2): %s  (blobA slice %d-%d, blobB slice %d-%d, dslice=%d)'
              % (ok, bA['slice_index_min'], bA['slice_index_max'], bB['slice_index_min'], bB['slice_index_max'],
                 bB['slice_index_min'] - bA['slice_index_max']))
        for c in checks:
            print('   ', c)

        m, X, lab = proximity_components(P, C, ia, radius=1.5)
        ja = int(np.where(m == ia)[0][0])
        jb = int(np.where(m == ib)[0][0])
        neck = true_neck(X, lab, ja, jb)
        if neck is None:
            print('A and B already in the SAME 1.5cm-proximity component -- likely a charge-contiguous'
                  ' path, not a graph bridge. Report the geodesic separately (not computed here).')
            return
        p1, p2, d, na, nb_ = neck
        print('true closest-approach between proximity components: %s <-> %s  d=%.2f cm (compA n=%d, compB n=%d)'
              % (np.round(p1, 2), np.round(p2, 2), d, na, nb_))
        r = walk_and_score(ld, p1, p2)
        print('walk: n=%d branch=%s  num_bad1[0]=%d num_bad[0]=%d (U=%d V=%d W=%d)  -> relaxed-graph REJECT=%s'
              % (r['nst'], r['branch'], r['nb1'][0], r['nb'][0], r['nb'][1], r['nb'][2], r['nb'][3], r['kill']))
        print('   angles: U-prolong=%.1f V-prolong=%.1f W-prolong=%.1f drift=%.1f' % r['angles'])
        print('   per-step live/dead UVW:', ' '.join(''.join(str(v) for v in s) for s in r['rows']))
    finally:
        ld.cleanup()


if __name__ == '__main__':
    main()

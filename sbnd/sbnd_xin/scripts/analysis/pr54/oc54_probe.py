#!/usr/bin/env python3
"""doc pr/54: 18255-142421, owner point (108.6,-71.2,220.9) -- a visually
separate EM-shower component of PR cluster 7 that carries no fitted
trajectory. Two things this script answers, both read-only replays of the
actual toolkit primitives against this event's own products (no code run):

1. Geometry: is the owner's point in a spatially separated connected
   component of cluster 7's image charge, and how far is it from any fitted
   point / any of cluster 7's final segments? (clus/src/MultiAlgBlobClustering.cxx
   fill_bee_points_from_pr_graph iterates the PR graph's segments only, so a
   region with none is structurally invisible in Bee -- this measures the gap.)

2. Mechanism (M-A vs M-B): replay clus/src/NeutrinoOtherSegments.cxx's Step-1
   "already covered" tagging (lines ~102-114) for every steiner point that
   sits inside the separated component, using the run's FINAL segment set as
   the "existing_segments" fit clouds and the event's own dead-channel winds
   for the get_closest_dead_chs() disjunct:

       u_ok = min_2d_dist_u < scaling_2d*search_range (0.8*1.5cm=1.2cm)
              OR get_closest_dead_chs(..., plane=U)
       (same for v_ok, w_ok; tagged = u_ok and v_ok and w_ok)

   If most of the region's steiner points tag as "already covered" (M-A: the
   candidate component never formed / never survived to be considered a
   NEW segment candidate), report which plane's disjunct (2D-proximity or
   dead-channel) is doing the covering. If untagged points DO form a
   component but it still produced no segment, that is M-B (Step-8 quality
   cut / special_A==SIZE_MAX skip) and this script says so explicitly --
   settling M-B numerically needs a TRACE-level rerun, not this replay.

The 2D projection matches DynamicPointCloud::get_closest_2d_point_info
(clus/src/DynamicPointCloud.cxx:344-382): distance is Euclidean in
(drift-x, cos(angle)*z - sin(angle)*y), U=+60deg, V=-60deg, W=0deg
(clus/src/oc53_probe.py::_y2d, same convention, re-derived here standalone).

Usage:
    oc54_probe.py <pr_arm_dir> <ql_evt_dir> <cluster_id> <Tx> <Ty> <Tz>

    <pr_arm_dir>: a pr_evt<N> directory with mabc-pr.zip and
                  calib-pr-evt<N>.json (needs PR_EXTRA_STAGES=pr_display).
    <ql_evt_dir>: the matching QL-stage ql_evt<N> dir (pctree-evt<N>.tar.gz),
                  for dead_winds_a<A>f0p<U|V|W>.

Repro (doc pr/54):
    cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
    python3 ../wcp-porting-img/sbnd/sbnd_xin/scripts/analysis/pr54/oc54_probe.py \\
        work-ncpi0-cb0805/pr_evt142421 work-ncpi0-cb0805/ql_evt142421 \\
        7 108.6 -71.2 220.9
"""
import sys, os, json, math, zipfile, tarfile, tempfile, glob, shutil

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ---- SBND face-0 plane pitch angles: U=+60deg, V=-60deg, W=0deg from +y
# toward +z, matching DynamicPointCloud::get_closest_2d_point_info's
# projected_y = cos(angle)*z - sin(angle)*y.
_S, _C = math.sin(math.radians(60)), math.cos(math.radians(60))


def _y2d(pi, y, z):
    cy, cz = ((-_S, _C), (_S, _C), (0.0, 1.0))[pi]
    return cy * y + cz * z


SEARCH_RANGE = 1.5     # cm, NeutrinoOtherSegments.cxx default arg
SCALING_2D = 0.8        # cm, NeutrinoOtherSegments.cxx default arg
TAG_THRESH = SCALING_2D * SEARCH_RANGE  # 1.2 cm


class DeadChans:
    """dead_winds_a<A>f0p<U|V|W> reader + wind(y2d) conversion, same idiom as
    scripts/analysis/pr53/oc53_probe.py::Loader (independent copy so pr/54
    does not depend on pr/53's script surviving unmodified)."""

    def __init__(self, ql_evt_dir):
        evt = os.path.basename(ql_evt_dir).replace('ql_evt', '')
        tgz = os.path.join(ql_evt_dir, 'pctree-evt%s.tar.gz' % evt)
        self.tmp = tempfile.mkdtemp(prefix='oc54_')
        with tarfile.open(tgz) as t:
            t.extractall(self.tmp)
        self.idx = {}
        for f in glob.glob(os.path.join(self.tmp, '*_metadata.json')):
            m = json.load(open(f))
            if m.get('datatype') == 'pcarray':
                self.idx[m['datapath']] = f.replace('_metadata.json', '_array.npy')
        self.pre = 'pointtrees/%s/live/' % evt
        self.wfit, self.dead = {}, {}
        for a in (0, 1):
            for pi, pl in enumerate('UVW'):
                b = 'pointclouds/namedpcs/ctpc_a%df0p%s/arrays/' % (a, pl)
                y = self.A(b + 'y')
                w = self.A(b + 'wind')
                self.wfit[(a, pi)] = np.polyfit(y, w, 1)  # wind = m*y2d_mm + c
                self.dead[(a, pi)] = self._winds(a, pl)

    def A(self, path):
        return np.load(self.idx[self.pre + path])

    def _winds(self, a, pl):
        try:
            b = 'pointclouds/namedpcs/dead_winds_a%df0p%s/arrays/' % (a, pl)
            return dict(zip(self.A(b + 'wind').tolist(),
                             zip(self.A(b + 'xbeg').tolist(), self.A(b + 'xend').tolist())))
        except KeyError:
            return {}

    def wind(self, apa, pi, y2d_mm):
        f = self.wfit[(apa, pi)]
        return int(round(np.polyval(f, y2d_mm)))

    def is_dead(self, apa, pi, x_cm, y2d_cm, ch_range=1):
        """get_closest_dead_chs(point,1,apa,face=0,pind=pi) replay
        (Facade_Grouping.cxx)."""
        x_mm, y2d_mm = x_cm * 10.0, y2d_cm * 10.0
        w = self.wind(apa, pi, y2d_mm)
        d = self.dead[(apa, pi)]
        for ch in range(w - ch_range, w + ch_range + 1):
            if ch in d:
                lo, hi = d[ch]
                if lo <= x_mm <= hi:
                    return True
        return False

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def load_pr(pr_arm_dir, cluster_id):
    evt = os.path.basename(pr_arm_dir).replace('pr_evt', '')
    z = zipfile.ZipFile(os.path.join(pr_arm_dir, 'mabc-pr.zip'))
    load = lambda n: json.loads(z.read('data/0/0-%s.json' % n))
    cg = load('clustering-global')
    ft = load('track_fit-global')
    C = np.c_[cg['x'], cg['y'], cg['z']]
    cid = np.array(cg['cluster_id'])
    F = np.c_[ft['x'], ft['y'], ft['z']]
    frid = np.array(ft['real_cluster_id'])
    # track_fit-global real_cluster_id = cluster_id*1000 + segment graph index
    fmask = (frid // 1000) == cluster_id
    calib_path = os.path.join(pr_arm_dir, 'calib-pr-evt%s.json' % evt)
    calib = json.load(open(calib_path)) if os.path.exists(calib_path) else None
    return C[cid == cluster_id], F[fmask], calib


def blob_component(X, target, radius=1.6):
    t = cKDTree(X)
    pr = np.array(list(t.query_pairs(radius)))
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
    n, lab = connected_components(g, directed=False)
    i0 = int(np.argmin(np.linalg.norm(X - target, axis=1)))
    lt = lab[i0]
    return X[lab == lt], n, lab


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        sys.exit(1)
    pr_dir, ql_dir, cid = sys.argv[1], sys.argv[2], int(sys.argv[3])
    T = np.array([float(v) for v in sys.argv[4:7]])

    X, F, calib = load_pr(pr_dir, cid)
    print('cluster %d: %d image pts, %d fit pts (final segments)' % (cid, len(X), len(F)))

    blob, ncomp, lab = blob_component(X, T)
    print('components at 1.6cm: %d ; target-component n=%d' % (ncomp, len(blob)))
    tF = cKDTree(F) if len(F) else None
    if tF is not None:
        d, _ = tF.query(blob)
        print('blob -> nearest fit pt: min=%.2f cm median=%.2f cm  (n>5cm frac=%.3f)'
              % (d.min(), np.median(d), (d > 5).mean()))
    dT = np.linalg.norm(blob - T, axis=1)
    print('owner point %s: nearest blob pt %.2f cm away' % (T.tolist(), dT.min()))

    if calib is None:
        print('\nNo calib-pr-evt%s.json (PR_EXTRA_STAGES=pr_display) in %s -- '
              'cannot replay Step-1 tagging (needs steiner_pc + flag_terminal). Stop.' % (
                  os.path.basename(pr_dir).replace('pr_evt', ''), pr_dir))
        return

    st = [s for s in calib['steiner'] if s['cluster_id'] == cid]
    if not st:
        print('\nNo steiner entry for cluster %d in calib json -- steiner_pc empty/absent, '
              'find_proto_vertex would have returned at NeutrinoPatternBase.cxx:2381/2384. Stop.' % cid)
        return
    st = st[0]
    S = np.c_[st['x'], st['y'], st['z']]
    FT = np.array(st.get('flag_terminal', [0] * len(S)))
    tb = cKDTree(blob)
    ds, _ = tb.query(S)
    inblob = ds < 1.0
    print('\nsteiner pts total=%d ; inside target component (<=1cm)=%d ; '
          'of those, terminals=%d' % (len(S), inblob.sum(), FT[inblob].sum()))

    Sin = S[inblob]
    if len(Sin) == 0:
        print('No steiner points fall inside the separated component -- unexpected, stop.')
        return

    # apa/face: cluster 7 in this event is entirely x>0 (apa=1); generalize
    # per-point in case a future case spans both.
    apa_of = lambda x: 1 if x >= 0 else 0
    dc = DeadChans(ql_dir)
    try:
        if len(F) == 0:
            print('\nNo fitted points at all for cluster %d in this arm -- 2D proximity is '
                  'vacuously 1e9 everywhere (empty-tree sentinel, NeutrinoOtherSegments.cxx:85-92); '
                  'tagging can only fire via the dead-channel disjunct.' % cid)
        # Precompute per-(apa,plane) 2D coordinate arrays of the fit cloud.
        Fapa = np.array([apa_of(x) for x in F[:, 0]]) if len(F) else np.zeros(0, int)
        F2d = {}
        for a in (0, 1):
            m = Fapa == a
            for pi in range(3):
                if m.sum():
                    y2d = _y2d(pi, F[m, 1], F[m, 2])
                    F2d[(a, pi)] = np.c_[F[m, 0], y2d]
                else:
                    F2d[(a, pi)] = np.zeros((0, 2))

        n_tag = 0
        n_tag_dead_only = {0: 0, 1: 0, 2: 0}  # per-plane: dead disjunct carried it (2D too far/empty)
        n_ok = {0: 0, 1: 0, 2: 0}
        rows = []
        for p in Sin:
            a = apa_of(p[0])
            plane_ok = [False, False, False]
            plane_via_dead = [False, False, False]
            for pi in range(3):
                y2d = _y2d(pi, p[1], p[2])
                cloud = F2d[(a, pi)]
                if len(cloud):
                    dd = np.min(np.hypot(cloud[:, 0] - p[0], cloud[:, 1] - y2d))
                else:
                    dd = 1e9  # empty-tree sentinel (NeutrinoOtherSegments.cxx:85-92 guard: treated as far, not near)
                near = dd < TAG_THRESH
                dead = dc.is_dead(a, pi, p[0], y2d)
                plane_ok[pi] = near or dead
                plane_via_dead[pi] = dead and not near
                if plane_ok[pi]:
                    n_ok[pi] += 1
                if plane_via_dead[pi]:
                    n_tag_dead_only[pi] += 1
            tagged = all(plane_ok)
            if tagged:
                n_tag += 1
            rows.append((p, plane_ok, plane_via_dead, tagged))

        print('\nStep-1 tagging replay on the %d in-blob steiner points (final segment set, this arm):'
              % len(Sin))
        print('  tagged (u_ok and v_ok and w_ok) = %d / %d  (%.1f%%)'
              % (n_tag, len(Sin), 100.0 * n_tag / len(Sin)))
        for pi, pl in enumerate('UVW'):
            print('  plane %s: ok=%d/%d (%.1f%%), of which dead-channel-only (no 2D proximity)=%d'
                  % (pl, n_ok[pi], len(Sin), 100.0 * n_ok[pi] / len(Sin), n_tag_dead_only[pi]))

        if n_tag > 0.5 * len(Sin):
            print('\nVERDICT: M-A -- a majority of the region tags as "already covered" in Step 1 '
                  'and so never becomes an independent connected-component candidate at all.')
        else:
            print('\nVERDICT: mostly UNTAGGED (%.1f%% tagged) -- Step 1 does not explain the '
                  'missing segment; the component should have reached Step 8 as a candidate. '
                  'This points to M-B (the quality cut at NeutrinoOtherSegments.cxx:415-423 or the '
                  'special_A==SIZE_MAX skip at :310-313) and needs a TRACE-level rerun '
                  '(NeutrinoOtherSegments.cxx:117-126 is commented out) to read the actual '
                  'number_not_faked / length / max_dis_u,v,w values -- not settled by this replay.'
                  % (100.0 * n_tag / len(Sin)))
    finally:
        dc.cleanup()


if __name__ == '__main__':
    main()

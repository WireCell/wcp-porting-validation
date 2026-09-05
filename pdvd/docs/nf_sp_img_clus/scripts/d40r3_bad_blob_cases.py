#!/usr/bin/env python3
"""doc pdvd/40 round 3 -- collect the fabricated-Steiner cases across arms and
classify each by which hole of remove_bad_blobs let it through.

Two sources per event dir (work/<run6>_<evt>_<arm>/):
  * wct_pr_*.log     BADBLOB census lines (ImproveCluster_1, bad_blob_report=true):
                     one per (cluster, apa, face) with the legacy component count,
                     the per-blob support count, the legacy vote, and the
                     unsupported RUNS (blob count, slices, span, center).
  * calib-pr-evt*.json + mabc-pr.zip   the 3D census (as d40_steiner_void_xdet.py):
                     Steiner points > 3 cm from any live point of the event,
                     linked at 3 cm into groups; group span and center.

Every 3D group above --min-span is matched to the BADBLOB run of the same
cluster whose center is nearest (any apa/face), and classified:
  (a) single component      ncomp == 1 (legacy filter never looks)
  (b) mixed component       ncomp > 1, the run sits in a component that survived the
                            first-blob vote (a fabricated column attached to a real track)
  (c) wire-supported        the run was NOT reported unsupported: the blobs overlap an
                            original blob in wire space although their 3D points are far
  (d) no run                no BADBLOB run within --match cm of the group center.  In a
                            knob-OFF arm this is dominated by the STALE-CACHE faces: the
                            filter (and so the census) never ran on a multi-face cluster's
                            second face.  Use --runs-from <knob-ON tag> to classify those.

Usage:
  d40r3_bad_blob_cases.py [--min-span 10] [--match 15] [--top 30] work/*_d41base
  d40r3_bad_blob_cases.py --spans work/*_d41base     # blob-space vs 3D run-span distributions
"""
import argparse, glob, json, os, re, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SENTINEL_CM = 1e4
SKIPPED = []   # (event, cluster_id, nsteiner) of sentinel-T0 clusters left out of the 3D census
FAR_CM = 3.0
LINK_CM = 3.0

HEAD = re.compile(r"BADBLOB cid=(\d+) ident=(\d+) apa=(\d+) face=(\d+) nnew=(\d+) norig=(\d+) ncomp=(\d+) ncomp_ss=(\d+) nsup=(\d+) legacy_rm=(\d+) run_rm=(\d+) nruns=(\d+) maxrun_cm=([\d.]+)")
# run centers are in the RAW drift frame (blob center_pos); the 3D census is in
# x_t0cor.  The two differ by a per-(cluster, apa, face) constant along x, so
# matching is done on (y, z) only.
RUN = re.compile(r"run (\d+): nb=(\d+) nslices=(\d+) span_cm=([\d.]+) craw=\(([-\d.]+),([-\d.]+),([-\d.]+)\)(?: bb=\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\))?")


def parse_log(d):
    logs = glob.glob(os.path.join(d, 'wct_pr_*.log'))
    rows = []
    if not logs:
        return rows
    for line in open(logs[0], errors='replace'):
        m = HEAD.search(line)
        if not m:
            continue
        r = dict(cluster=int(m[1]), ident=int(m[2]), apa=int(m[3]), face=int(m[4]), nnew=int(m[5]), norig=int(m[6]),
                 ncomp=int(m[7]), ncomp_ss=int(m[8]), nsup=int(m[9]), legacy_rm=int(m[10]),
                 run_rm=int(m[11]), nruns=int(m[12]), maxrun=float(m[13]))
        r['runs'] = [dict(nb=int(x[2]), nslices=int(x[3]), span=float(x[4]),
                          c=np.array([float(x[5]), float(x[6]), float(x[7])]),
                          bb=(np.array([float(x[k]) for k in range(8, 14)]) if x[8] is not None else None))
                     for x in RUN.finditer(line)]
        rows.append(r)
    return rows


def groups_3d(d):
    """per cluster: list of (span, center, npts) of Steiner groups > FAR_CM from live."""
    cal = glob.glob(os.path.join(d, 'calib-pr-evt*.json'))
    zp = os.path.join(d, 'mabc-pr.zip')
    if not cal or not os.path.exists(zp):
        return None
    st = json.load(open(cal[0])).get('steiner', [])
    z = zipfile.ZipFile(zp)
    keys = [k for k in z.namelist() if k.endswith('clustering-global.json')]
    j = json.loads(z.read(keys[0]))
    C = np.stack([j['x'], j['y'], j['z']], 1).astype(float)
    cc = np.asarray(j['cluster_id'])
    live_ok = set(cc[np.abs(C[:, 0]) < SENTINEL_CM].tolist())   # clusters with in-detector live points
    C = C[np.abs(C[:, 0]) < SENTINEL_CM]
    T = cKDTree(C)
    out = {}
    tot = 0
    for e in st:
        S = np.stack([e['x'], e['y'], e['z']], 1).astype(float)
        if not len(S):
            continue
        if e['cluster_id'] not in live_ok:
            SKIPPED.append((os.path.basename(d), e['cluster_id'], len(S)))   # sentinel-T0 cluster (doc 40 sec 8)
            continue
        tot += len(S)
        dd, _ = T.query(S)
        F = S[dd > FAR_CM]
        if not len(F):
            continue
        if len(F) > 1:
            pr = cKDTree(F).query_pairs(LINK_CM, output_type='ndarray')
            g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(F), len(F)))
            n, lab = connected_components(g, directed=False)
        else:
            n, lab = 1, np.zeros(1, int)
        gl = []
        for k in range(n):
            P = F[lab == k]
            lo, hi = P.min(0), P.max(0)
            gl.append(dict(span=float(np.linalg.norm(hi - lo)), c=(lo + hi) / 2, n=len(P)))
        out[e['cluster_id']] = gl
    return out, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-span', type=float, default=10.0)
    ap.add_argument('--match', type=float, default=15.0)
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--spans', action='store_true')
    ap.add_argument('--runs-from', default=None,
                    help='take the BADBLOB runs from the sibling arm with this tag instead of the census arm '
                         '(the run census is computed BEFORE removal, so a knob-ON arm reports the runs the '
                         'knob-OFF arm never saw on its cache-hidden second faces)')
    ap.add_argument('dirs', nargs='+')
    a = ap.parse_args()

    cases = []
    OVR = dict(rm=0, calls=0, calls_rm=0, run_rm=0)
    blob_spans, d3_spans = [], []
    nev = 0
    nsteiner = 0
    for d in sorted(a.dirs):
        rd = d
        if a.runs_from:
            rd = re.sub(r'_[^_/]+/?$', '_' + a.runs_from, d.rstrip('/'))
        rows = parse_log(rd)
        g3 = groups_3d(d)
        if g3 is None or not rows:
            continue
        g3, tot = g3
        nev += 1
        nsteiner += tot
        ev = os.path.basename(d)
        byc = {}
        for r in rows:
            byc.setdefault(r['cluster'], []).append(r)
            OVR['calls'] += 1; OVR['rm'] += r['legacy_rm']; OVR['run_rm'] += r['run_rm']
            OVR['calls_rm'] += 1 if r['legacy_rm'] else 0
            for run in r['runs']:
                blob_spans.append(run['span'])
        for cid, gl in g3.items():
            for g in gl:
                d3_spans.append(g['span'])
                if g['span'] < a.min_span:
                    continue
                # The BADBLOB ident is the retile-time ident, which is NOT the
                # dump-time Bee/calib cluster id (idents drift as later stages
                # split clusters), so runs are matched over the WHOLE event on
                # (y, z) -- the x frames differ too (raw vs x_t0cor).
                best = None
                for r in rows:
                    for run in r['runs']:
                        dist = float(np.linalg.norm(run['c'][1:] - g['c'][1:]))   # (y, z) only, see RUN
                        if best is None or dist < best[0]:
                            best = (dist, r, run)
                if best is None or best[0] > a.match:
                    cls = 'd'
                    r = {}
                    detail = 'no run within %.0f cm in (y,z) (nearest %s)' % (a.match, '%.1f' % best[0] if best else 'none')
                    run = None
                else:
                    dist, r, run = best
                    if r['ncomp'] == 1:
                        cls = 'a'
                    else:
                        cls = 'b'
                    # (c): the 3D group is much longer than the matched run => most of it was "supported"
                    if run['span'] < 0.5 * g['span']:
                        cls = 'c'
                    detail = 'run nb=%d nsl=%d span=%.1f dyz=%.1f; ident %d apa%d f%d ncomp=%d nsup=%d/%d legacy_rm=%d' % (
                        run['nb'], run['nslices'], run['span'], dist, r['ident'], r['apa'], r['face'], r['ncomp'], r['nsup'], r['nnew'], r['legacy_rm'])
                cases.append(dict(ev=ev, cluster=cid, span3d=g['span'], n=g['n'], c=g['c'], cls=cls, detail=detail,
                                  apa=r.get('apa', -1), face=r.get('face', -1)))

    print('== %d events, %d steiner points, %d 3D groups >%.0f cm, %d BADBLOB runs' % (
        nev, nsteiner, len(cases), a.min_span, len(blob_spans)))
    print('   sentinel-T0 clusters excluded from the 3D census (doc 40 sec 8): %d clusters, %d steiner points' % (
        len(SKIPPED), sum(k[2] for k in SKIPPED)))
    print('   legacy vote: %d blobs removed in %d (cluster,apa,face) calls of %d; run bound would remove %d' % (
        OVR['rm'], OVR['calls_rm'], OVR['calls'], OVR['run_rm']))
    cls_n = {k: sum(1 for c in cases if c['cls'] == k) for k in 'abcd'}
    cls_p = {k: sum(c['n'] for c in cases if c['cls'] == k) for k in 'abcd'}
    print('   class census (groups / steiner points):  a single-component %d / %d   b mixed-component %d / %d   c wire-supported %d / %d   d no-run %d / %d' % (
        cls_n['a'], cls_p['a'], cls_n['b'], cls_p['b'], cls_n['c'], cls_p['c'], cls_n['d'], cls_p['d']))
    cases.sort(key=lambda c: -c['span3d'])
    print('   worst %d groups:' % a.top)
    print('   %-20s %5s %3s %8s %6s %-28s %s' % ('event', 'clus', 'cls', 'span_cm', 'npts', 'center (x,y,z) cm', 'blob-space run'))
    for c in cases[:a.top]:
        print('   %-20s %5d  %s  %8.1f %6d (%7.1f,%7.1f,%7.1f) %s' % (
            c['ev'], c['cluster'], c['cls'], c['span3d'], c['n'], c['c'][0], c['c'][1], c['c'][2], c['detail']))
    if a.spans or True:
        def q(v, ps=(50, 75, 90, 95, 99)):
            v = np.array(v)
            return ' '.join('p%d=%.1f' % (p, np.percentile(v, p)) for p in ps) + ' max=%.1f n=%d' % (v.max(), len(v)) if len(v) else 'none'
        print('   3D group spans   : ' + q(d3_spans))
        print('   blob-space run spans (BADBLOB, every reported run): ' + q(blob_spans))
        for t in (10, 20, 30):
            print('   groups >%2d cm: 3D %5d (%5.1f%%)   blob-space %5d (%5.1f%%)' % (
                t, sum(1 for s in d3_spans if s > t), 100. * sum(1 for s in d3_spans if s > t) / max(1, len(d3_spans)),
                sum(1 for s in blob_spans if s > t), 100. * sum(1 for s in blob_spans if s > t) / max(1, len(blob_spans))))


if __name__ == '__main__':
    main()

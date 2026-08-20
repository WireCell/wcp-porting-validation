#!/usr/bin/env python3
"""doc pr/96: imaged charge with NO fitted trajectory over it -- the inverse of
scripts/pr94r3_gap_metric.py.

Forked from pr94r3_gap_metric.py (same two Bee layers, same zip loader) because
that script measures fit -> charge (a fitted point straying into a void, doc
pr/94 sec 9.10) and this round needs charge -> fit (imaged charge that no
trajectory covers, the owner's "missing a vertex track").  Everything past the
loader is new: connected grouping of the uncovered points, a straightness /
bulk discriminator to separate a missing TRACK from ordinary shower spread, and
an optional join against the `pr54 isolated-residual drop` lines already present
in every arm's log at -L debug.

Layers are read from pr_evt<ID>/mabc-pr.zip.  "Charge" is ALWAYS
clustering-global, never img-global: img-global is the one raw layer and carries
a per-cluster drift-x offset (doc pr/13, pr/67 sec 1), so mixing the two can
land you in a different object entirely.

Usage:
  pr96_uncover_census.py <arm_root|pr_evt_dir> [...] [options]

    --thr CM        uncovered if farther than this from every fit point [3.0]
    --link CM       single-link radius for grouping uncovered points  [2.0]
    --npts N        track-like: minimum points in the group           [40]
    --len CM        track-like: minimum PCA extent                    [5.0]
    --rms CM        track-like: maximum transverse rms (straightness) [0.8]
    --qfrac F       track-like: minimum fraction of cluster charge    [0.03]
    --dvtx CM       track-like: group must start within this of the
                    neutrino main vertex -- this is what separates a missing
                    PRONG from a shower rind running alongside a trajectory
                    far from the vertex (calibration, doc pr/96 sec 4) [15.0]
    --all-clusters  score every cluster that has fit points, not just the
                    neutrino cluster (the one holding the q=15000 main vertex)
    --tsv PATH      write one row per group
    --quiet         summary lines only
"""
import sys, os, json, glob, zipfile, math, re
import numpy as np
from scipy.spatial import cKDTree

MAIN_VTX_Q = 15000.0     # MultiAlgBlobClustering marks the main vertex with q=15000

def layers(zp, tags=('clustering-global', 'track_fit-global',
                     'shower_track-global', 'vertices-global')):
    z = zipfile.ZipFile(zp)
    out = {}
    for n in z.namelist():
        for tag in tags:
            if n.endswith(tag + '.json'):
                out[tag] = json.loads(z.read(n))
    return out

def xyz(d, mask=None):
    P = np.c_[d['x'], d['y'], d['z']]
    return P if mask is None else P[mask]

def groups_single_link(P, radius):
    """Connected components of P under a `radius` single-link rule.  KD-tree
    ball queries, so this stays usable on the 1000+-point shower cases."""
    n = len(P)
    lab = -np.ones(n, int)
    tree = cKDTree(P)
    g = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = g
        while stack:
            j = stack.pop()
            for k in tree.query_ball_point(P[j], radius):
                if lab[k] < 0:
                    lab[k] = g
                    stack.append(k)
        g += 1
    return lab, g

def pca_axis(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0]

def parse_pr54_drops(logpath):
    """`pr54 isolated-residual drop: cluster N n_points=.. length=.. cm
    dir_mag=.. cm v1=(x,y,z) v2=(x,y,z) cm` -- unconditional DEBUG line, so it
    is present in any arm run at -L debug with no probe knob."""
    pat = re.compile(r'pr54 isolated-residual drop: cluster (\d+) n_points=(\d+) '
                     r'length=([-\d.]+) cm dir_mag=([-\d.]+) cm '
                     r'v1=\(([-\d.]+),([-\d.]+),([-\d.]+)\) '
                     r'v2=\(([-\d.]+),([-\d.]+),([-\d.]+)\)')
    out = []
    try:
        with open(logpath, 'r', errors='replace') as f:
            for line in f:
                m = pat.search(line)
                if not m:
                    continue
                g = m.groups()
                rec = dict(cid=int(g[0]), npts=int(g[1]), length=float(g[2]),
                           dir_mag=float(g[3]),
                           v1=np.array([float(g[4]), float(g[5]), float(g[6])]),
                           v2=np.array([float(g[7]), float(g[8]), float(g[9])]))
                out.append(rec)     # the line repeats once per round; uniq'd below
    except OSError:
        pass
    # de-duplicate on value (dict compare above fails on ndarray)
    uniq = []
    for r in out:
        if not any(r['cid'] == u['cid'] and r['npts'] == u['npts']
                   and abs(r['length'] - u['length']) < 1e-6
                   and np.allclose(r['v1'], u['v1']) for u in uniq):
            uniq.append(r)
    return uniq

def parse_pr30audit(logpath):
    try:
        with open(logpath, 'r', errors='replace') as f:
            for line in f:
                if 'PR30AUDIT' in line:
                    d = {}
                    for kv in line.split():
                        if '=' in kv and not kv.startswith('knobs'):
                            k, v = kv.split('=', 1)
                            if v.replace('.', '', 1).replace('-', '', 1).isdigit():
                                d[k] = v
                    return d
    except OSError:
        pass
    return {}

def event_dirs(args):
    out = []
    for a in args:
        if os.path.basename(a.rstrip('/')).startswith('pr_evt'):
            out.append(a.rstrip('/'))
        else:
            out += sorted(glob.glob(os.path.join(a, 'pr_evt*')))
    return out

def main(argv):
    opt = dict(thr=3.0, link=2.0, npts=40, length=5.0, rms=0.8, qfrac=0.03, dvtx=15.0,
               all_clusters=False, tsv=None, quiet=False)
    roots, i = [], 1
    while i < len(argv):
        a = argv[i]
        if a == '--thr':            opt['thr'] = float(argv[i+1]); i += 2
        elif a == '--link':         opt['link'] = float(argv[i+1]); i += 2
        elif a == '--npts':         opt['npts'] = int(argv[i+1]); i += 2
        elif a == '--len':          opt['length'] = float(argv[i+1]); i += 2
        elif a == '--rms':          opt['rms'] = float(argv[i+1]); i += 2
        elif a == '--qfrac':        opt['qfrac'] = float(argv[i+1]); i += 2
        elif a == '--dvtx':         opt['dvtx'] = float(argv[i+1]); i += 2
        elif a == '--all-clusters': opt['all_clusters'] = True; i += 1
        elif a == '--tsv':          opt['tsv'] = argv[i+1]; i += 2
        elif a == '--quiet':        opt['quiet'] = True; i += 1
        elif a in ('-h', '--help'): print(__doc__); return 0
        else:                       roots.append(a); i += 1
    if not roots:
        print(__doc__); return 2

    rows = []
    for ed in event_dirs(roots):
        evt = os.path.basename(ed).replace('pr_evt', '')
        zp = os.path.join(ed, 'mabc-pr.zip')
        if not os.path.exists(zp):
            print(f'{evt}\tNO_ZIP'); continue
        try:
            L = layers(zp)
        except Exception as e:
            print(f'{evt}\tBAD_ZIP {e}'); continue
        if 'track_fit-global' not in L or 'clustering-global' not in L:
            print(f'{evt}\tMISSING_LAYER'); continue

        cl, tf = L['clustering-global'], L['track_fit-global']
        vv = L.get('vertices-global', {'x': [], 'y': [], 'z': [], 'q': [],
                                       'cluster_id': []})
        drops = parse_pr54_drops(os.path.join(ed, f'wct_pr_evt{evt}.log'))
        audit = parse_pr30audit(os.path.join(ed, f'wct_pr_evt{evt}.log'))

        # neutrino cluster = the one holding the q=15000 main vertex
        nucid, nuvtx = None, None
        for k in range(len(vv['x'])):
            if float(vv['q'][k]) == MAIN_VTX_Q:
                nucid = int(vv['cluster_id'][k])
                nuvtx = np.array([vv['x'][k], vv['y'][k], vv['z'][k]])
        tf_cid = np.array(tf['cluster_id'])
        tf_rc = np.array(tf['real_cluster_id'])
        cl_cid = np.array(cl['cluster_id'])
        cl_q = np.array(cl['q'], float)

        if opt['all_clusters']:
            cids = sorted(set(int(c) for c in tf_cid[tf_rc > 0]))
        elif nucid is None:
            print(f'{evt}\tNO_MAIN_VERTEX'); continue
        else:
            cids = [nucid]

        ev_flag = 0
        for cid in cids:
            fm = (tf_cid == cid) & (tf_rc > 0)      # rcid<0 = PR-graph vertex points
            cm = cl_cid == cid
            if fm.sum() == 0 or cm.sum() == 0:
                continue
            F, Fr = xyz(tf, fm), tf_rc[fm]
            C, Q = xyz(cl, cm), cl_q[cm]
            d, near = cKDTree(F).query(C)
            unc = d > opt['thr']
            if unc.sum() == 0:
                if not opt['quiet']:
                    print(f'{evt} cid {cid:4d}: {cm.sum():5d} chg / {fm.sum():4d} fit  '
                          f'uncovered 0.0% q 0.0%  -- no group')
                continue
            U = C[unc]
            lab, ng = groups_single_link(U, opt['link'])
            order = sorted(range(ng), key=lambda g: -(lab == g).sum())
            hits = []
            for g in order:
                gm = lab == g
                G, GQ = U[gm], Q[unc][gm]
                c, ax = pca_axis(G) if len(G) > 2 else (G.mean(0), np.array([1., 0, 0]))
                proj = (G - c) @ ax
                plen = float(proj.max() - proj.min())
                perp = np.linalg.norm((G - c) - np.outer(proj, ax), axis=1)
                prms = float(perp.std())
                qfrac = float(GQ.sum() / Q.sum())
                maxd = float(d[unc][gm].max())
                dvtx = float(np.linalg.norm(G - nuvtx, axis=1).min()) if nuvtx is not None else -1.0
                # which existing segment is nearest, and at what angle
                nsegs = Fr[near[unc][gm]]
                seg = int(np.bincount(nsegs - nsegs.min()).argmax() + nsegs.min()) if len(nsegs) else -1
                ang = -1.0
                sm = Fr == seg
                if sm.sum() >= 3:
                    S = F[sm]
                    loc = S[np.linalg.norm(S - c, axis=1) < 15.0]
                    if len(loc) >= 3:
                        _, sax = pca_axis(loc)
                        ang = math.degrees(math.acos(min(1, abs(float(sax @ ax)))))
                # a pr54-dropped candidate landing in this group is a direct hit
                hit54 = ''
                for r in drops:
                    mid = 0.5 * (r['v1'] + r['v2'])
                    dd = min(np.linalg.norm(G - r['v1'], axis=1).min(),
                             np.linalg.norm(G - r['v2'], axis=1).min(),
                             np.linalg.norm(G - mid, axis=1).min())
                    if dd < 3.0:
                        hit54 = f"pr54drop(cid{r['cid']},npts{r['npts']},{r['length']:.1f}cm,d{dd:.1f})"
                        break
                tracklike = (gm.sum() >= opt['npts'] and plen >= opt['length']
                             and prms <= opt['rms'] and qfrac >= opt['qfrac']
                             and 0.0 <= dvtx <= opt['dvtx'])
                if tracklike:
                    ev_flag += 1
                rows.append(dict(evt=evt, cid=cid, grp=g, npts=int(gm.sum()),
                                 qfrac=qfrac, plen=plen, prms=prms, maxd=maxd,
                                 dvtx=dvtx, seg=seg, ang=ang,
                                 track=int(tracklike), hit54=hit54,
                                 cen='%.1f,%.1f,%.1f' % tuple(c)))
                if tracklike:
                    hits.append(rows[-1])
            if not opt['quiet']:
                print(f'{evt} cid {cid:4d}: {cm.sum():5d} chg / {fm.sum():4d} fit  '
                      f'uncovered {100*unc.mean():4.1f}% q {100*Q[unc].sum()/Q.sum():4.1f}%  '
                      f'{ng} grp  TRACKLIKE {len(hits)}'
                      + (f"  oseg_iso_drop={audit.get('oseg_iso_drop','?')}"
                         f" oseg_reject={audit.get('oseg_reject','?')}" if audit else ''))
                for h in hits:
                    print(f'    -> grp{h["grp"]}: n={h["npts"]:4d} q={100*h["qfrac"]:4.1f}% '
                          f'len={h["plen"]:5.1f} rms={h["prms"]:4.2f} maxd={h["maxd"]:4.1f} '
                          f'dvtx={h["dvtx"]:5.1f} seg={h["seg"]} ang={h["ang"]:5.1f} '
                          f'cen=({h["cen"]}) {h["hit54"]}')

    if opt['tsv']:
        cols = ['evt', 'cid', 'grp', 'npts', 'qfrac', 'plen', 'prms', 'maxd',
                'dvtx', 'seg', 'ang', 'track', 'hit54', 'cen']
        with open(opt['tsv'], 'w') as f:
            f.write('\t'.join(cols) + '\n')
            for r in rows:
                f.write('\t'.join(str(r[c]) for c in cols) + '\n')
        print(f'[wrote {opt["tsv"]}: {len(rows)} groups]')
    nt = sum(r['track'] for r in rows)
    print(f'[total: {len(rows)} uncovered groups, {nt} track-like, '
          f'{len(set(r["evt"] for r in rows if r["track"]))} events flagged]')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))

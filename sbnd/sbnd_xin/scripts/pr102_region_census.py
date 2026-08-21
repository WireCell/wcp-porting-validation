#!/usr/bin/env python3
"""doc pr/102: region-classified uncovered-charge census -- the pr/96 metric
with the blanket --dvtx cut replaced by an ownership/region classifier, so the
hadronic-shower population (which pr/96's --dvtx 15 deliberately discarded to
dodge the EM-rind confound) becomes measurable without flooding the metric
with EM-shower interiors.

Forked from scripts/pr96_uncover_census.py (frozen round record -- never edit
that file).  Everything up to and including the uncovered-group shape features
is the pr/96 code unchanged; new here is the calib-dump typing join and the
classifier.  --pr96-compat bypasses the classifier and reproduces the pr/96
track-like rule and output rows verbatim (fork-integrity proof).

Inputs per event dir pr_evt<ID>/:
  mabc-pr.zip          -- charge (clustering-global), fits (track_fit-global),
                          vertices-global (q==15000 marks the main vertex).
                          Charge is ALWAYS clustering-global, never img-global
                          (the one raw layer; per-cluster drift-x offset, doc
                          pr/13, pr/67 sec 1).
  calib-pr-evt<ID>.json -- typing source (PR_EXTRA_STAGES=pr_display only):
                          track_shower.particle_id is the OWNING SEGMENT id
                          (verified segments[].id join, doc pr/102), segments[]
                          carry pdg/flag_shower/shower_id, showers[].id is the
                          stem segment id and showers[].particle_id the pdg
                          (211 = A5 hadronic re-type, doc pr/99 r3).
  wct_pr_evt<ID>.log    -- `pr54 isolated-residual drop` join (pr/96) and,
                          on post-pr/99 arms, `A5 hadronic census:` lines.

Region classes for a shape-passing group (priority order, first match wins):
  NEARVTX  dvtx <= --dvtx-near               always flagged (owner scope)
  HAD-A5   owned by / adjacent to a pdg-211 (A5 hadronic) shower     flagged
  HAD-ADJ  within --dhad of a hadron TRACK segment (pdg 2212/211/321,
           flag_shower false; pdg 13 excluded -- that is corridor) or of a
           secondary >=2-track vertex; EM-owned fraction >= 0.8 overrides
           to EM-INT (the evt283713 rind canary)                     flagged
  EM-INT   matched points majority-owned (>= --ftype) by pdg-11 shower
           members                             reported, NEVER flagged
  TRK-COR  majority track-owned, group axis within 20 deg of the nearest
           segment (the 279955 fit-collapse family)     flagged (severity 2)
  OTHER    typed but none of the above                  flagged (hand-look)
  UNTYPED  matched fraction f_m < 0.5 (charge the PF never associated --
           itself the cross-cluster CUT signal)         flagged (hand-look)

Usage:
  pr102_region_census.py <arm_root|pr_evt_dir> [...] [options]

    --thr CM         uncovered if farther than this from every fit point [3.0]
    --link CM        single-link radius for grouping                     [2.0]
    --npts N         shape: minimum points in the group                  [40]
    --len CM         shape: minimum PCA extent                           [5.0]
    --rms CM         shape: maximum transverse rms                       [0.8]
    --qfrac F        shape: minimum fraction of cluster charge           [0.03]
    --dvtx-near CM   NEARVTX radius (pr/96's --dvtx, unchanged)          [15.0]
    --dmatch CM      zip point -> calib track_shower point match radius  [1.0]
    --ftype F        majority-ownership fraction for EM-INT / TRK-COR    [0.6]
    --dhad CM        adjacency radius to hadron segments / 2nd vertices  [10.0]
    --pr96-compat    reproduce pr/96 rows verbatim (no classifier)
    --all-clusters   score every cluster with fit points, not just the
                     neutrino cluster (the one holding the q=15000 vertex)
    --tsv PATH       per-group rows
    --summary-tsv PATH  per-event rows
    --quiet          summary lines only
"""
import sys, os, json, glob, zipfile, math, re
import numpy as np
from scipy.spatial import cKDTree

MAIN_VTX_Q = 15000.0     # MultiAlgBlobClustering marks the main vertex with q=15000
HAD_TRK_PDGS = (2212, 211, 321)   # proton / charged pion / kaon track segments
ANG_ALONG = 20.0                  # deg; group parallel to nearest segment => corridor
EM_OVERRIDE = 0.8                 # EM-owned fraction that overrides HAD-ADJ
F_MATCH_MIN = 0.5                 # below this matched fraction => UNTYPED
FRAME_TOL = 0.1                   # cm; zip q=15000 vertex vs calib main_vertex

# ---------------------------------------------------------------------------
# pr/96 code, unchanged (loader, grouping, features, pr54/pr30 log joins)
# ---------------------------------------------------------------------------

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
                out.append(rec)
    except OSError:
        pass
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

# ---------------------------------------------------------------------------
# new: A5 hadronic-census log join (post-pr/99 arms only; absent => empty)
# ---------------------------------------------------------------------------

def parse_a5_census(logpath):
    """`A5 hadronic census: shower id=.. pdg=.. conn=.. nseg=.. smax=..cm
    growth=.. n_early=.. n_late=.. dqdx_trunk=.. dqdx_term=.. bragg=..
    stem=.. verdict=..` -- one DEBUG line per evaluated claimed-EM shower."""
    out = {}
    pat = re.compile(r'A5 hadronic census: shower id=(\d+)\s+(.*)')
    try:
        with open(logpath, 'r', errors='replace') as f:
            for line in f:
                m = pat.search(line)
                if not m:
                    continue
                d = {}
                for kv in m.group(2).split():
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        d[k] = v.rstrip('cm')
                out[int(m.group(1))] = d
    except OSError:
        pass
    return out

# ---------------------------------------------------------------------------
# new: calib-dump typing
# ---------------------------------------------------------------------------

def load_calib(ed, evt):
    p = os.path.join(ed, f'calib-pr-evt{evt}.json')
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

class Typing:
    """Per-event typing index built from the calib dump."""
    def __init__(self, dump):
        ts = dump.get('track_shower', {})
        self.P = np.c_[ts.get('x', []), ts.get('y', []), ts.get('z', [])] \
            if ts.get('x') else np.zeros((0, 3))
        self.own = np.array(ts.get('particle_id', []), int)
        self.tree = cKDTree(self.P) if len(self.P) else None

        self.seg = {}
        for s in dump.get('segments', []):
            pts = np.array([[q['x'], q['y'], q['z']] for q in s.get('points', [])]) \
                if s.get('points') else np.zeros((0, 3))
            self.seg[int(s['id'])] = dict(
                pdg=int(s.get('particle_id') or 0),
                flag_shower=bool(s.get('flag_shower')),
                shower_id=int(s.get('shower_id', -1)),
                cluster_id=int(s.get('cluster_id', -1)),
                length=float(s.get('length', 0.0)), poly=pts)

        # showers[].id is the STEM SEGMENT id; segments[].shower_id joins it.
        # The join keeps one shower per segment (PrDisplayDump.cxx:432), so an
        # overlapped shower reads as under-populated -- accepted; exhibits that
        # hinge on membership escalate to WCT_SHOWER_CONTENT_DEBUG (doc pr/91).
        self.shower = {}
        self.ord2stem = {}   # A5 census log "shower id" is the ORDINAL shower_id
        for s in dump.get('showers', []):
            self.shower[int(s['id'])] = dict(
                pdg=int(s.get('particle_id') or 0),
                nseg=int(s.get('num_segments', 0)),
                total_length=float(s.get('total_length', 0.0)))
            self.ord2stem[int(s.get('shower_id', -1))] = int(s['id'])

        mv = dump.get('main_vertex') or {}
        self.main_vertex = (np.array([mv['x'], mv['y'], mv['z']])
                            if 'x' in mv else None)

        # hadron TRACK polylines (proton/pion/kaon, not shower-flagged, mu excluded)
        self.had_pts, self.had_ids = self._polyset(
            lambda s: (not s['flag_shower']) and abs(s['pdg']) in HAD_TRK_PDGS)
        # pdg-211 (A5 hadronic) shower stems + their member segments
        a5_stems = {sid for sid, sh in self.shower.items() if sh['pdg'] == 211}
        self.a5_member_segs = {i for i, s in self.seg.items()
                               if s['shower_id'] in a5_stems} | a5_stems
        self.a5_pts, self.a5_ids = self._polyset(
            lambda s, ids=self.a5_member_segs: True, only=self.a5_member_segs)
        # EM shower member segment ids (pdg-11 shower membership, NOT pdg 11
        # alone: (pdg 11, flag_shower=False) fit stems exist)
        em_stems = {sid for sid, sh in self.shower.items() if abs(sh['pdg']) == 11}
        self.em_segs = {i for i, s in self.seg.items()
                        if s['shower_id'] in em_stems} | em_stems

        # secondary vertices where >=2 TRACK segments meet (not the main vtx)
        self.sec_vtx = []
        seg_by_vtx = {}
        for s in dump.get('segments', []):
            if bool(s.get('flag_shower')):
                continue
            for vkey in ('start_vertex_id', 'end_vertex_id'):
                vid = int(s.get(vkey, -1))
                seg_by_vtx.setdefault(vid, []).append(int(s['id']))
        for v in dump.get('vertices', []):
            vid = int(v.get('id', -1))
            if v.get('is_main'):
                continue
            if len(seg_by_vtx.get(vid, [])) < 2:
                continue
            fit = v.get('fit') or {}
            if 'x' in fit:
                self.sec_vtx.append(np.array([fit['x'], fit['y'], fit['z']]))
        self.sec_vtx = np.array(self.sec_vtx) if self.sec_vtx else np.zeros((0, 3))
        self.sec_tree = cKDTree(self.sec_vtx) if len(self.sec_vtx) else None

    def _polyset(self, pred, only=None):
        pts, ids = [], []
        for i, s in self.seg.items():
            if only is not None and i not in only:
                continue
            if only is None and not pred(s):
                continue
            if len(s['poly']) == 0:
                continue
            pts.append(s['poly'])
            ids += [i] * len(s['poly'])
        if pts:
            return np.vstack(pts), np.array(ids, int)
        return np.zeros((0, 3)), np.array([], int)

    def match(self, G, dmatch):
        """G (n,3) group points -> (owner seg id array over matched, f_m)."""
        if self.tree is None or len(G) == 0:
            return np.array([], int), 0.0
        d, idx = self.tree.query(G)
        m = d <= dmatch
        return self.own[idx[m]], float(m.mean())


def classify_group(G, c, ax, dvtx, ty, opt):
    """-> (cls, sev, evidence dict).  Priority order per the module docstring."""
    ev = {}
    owners, f_m = ty.match(G, opt['dmatch'])
    ev['f_m'] = f_m
    # ownership fractions
    n = max(1, len(owners))
    f_em = sum(1 for o in owners if int(o) in ty.em_segs) / n
    f_trk = sum(1 for o in owners
                if int(o) in ty.seg and not ty.seg[int(o)]['flag_shower']) / n
    f_a5 = sum(1 for o in owners if int(o) in ty.a5_member_segs) / n
    ev['f_em'], ev['f_trk'], ev['f_a5'] = f_em, f_trk, f_a5
    # owner pdg histogram (top-3)
    from collections import Counter
    cnt = Counter(ty.seg[int(o)]['pdg'] for o in owners if int(o) in ty.seg)
    ev['own_pdg'] = ';'.join(f'{p}:{k}' for p, k in cnt.most_common(3))
    # hadron adjacency
    d_had, hadseg = -1.0, -1
    if len(ty.had_pts):
        d, i = cKDTree(ty.had_pts).query(G)
        j = int(np.argmin(d))
        d_had, hadseg = float(d[j]), int(ty.had_ids[i[j]])
    ev['d_had'], ev['hadseg'] = d_had, hadseg
    d_a5 = -1.0
    if len(ty.a5_pts):
        d_a5 = float(cKDTree(ty.a5_pts).query(G)[0].min())
    ev['d_a5'] = d_a5
    d_sec = -1.0
    if ty.sec_tree is not None:
        d_sec = float(ty.sec_tree.query(G)[0].min())
    ev['d_sec'] = d_sec

    if 0.0 <= dvtx <= opt['dvtx_near']:
        return 'NEARVTX', 1, ev
    if len(owners) and f_m >= F_MATCH_MIN:
        if f_a5 >= 0.2 or (0.0 <= d_a5 <= opt['dhad']):
            return 'HAD-A5', 1, ev
        had_adj = (0.0 <= d_had <= opt['dhad']) or (0.0 <= d_sec <= opt['dhad'])
        if had_adj and f_em < EM_OVERRIDE:
            return 'HAD-ADJ', 1, ev
        if f_em >= opt['ftype']:
            return 'EM-INT', 0, ev
        if f_trk >= opt['ftype']:
            return 'TRK-COR', 2, ev
        return 'OTHER', 2, ev
    return 'UNTYPED', 2, ev

# ---------------------------------------------------------------------------

def main(argv):
    opt = dict(thr=3.0, link=2.0, npts=40, length=5.0, rms=0.8, qfrac=0.03,
               dvtx_near=15.0, dmatch=1.0, ftype=0.6, dhad=10.0,
               compat=False, all_clusters=False, tsv=None, sumtsv=None,
               quiet=False)
    roots, i = [], 1
    while i < len(argv):
        a = argv[i]
        if a == '--thr':            opt['thr'] = float(argv[i+1]); i += 2
        elif a == '--link':         opt['link'] = float(argv[i+1]); i += 2
        elif a == '--npts':         opt['npts'] = int(argv[i+1]); i += 2
        elif a == '--len':          opt['length'] = float(argv[i+1]); i += 2
        elif a == '--rms':          opt['rms'] = float(argv[i+1]); i += 2
        elif a == '--qfrac':        opt['qfrac'] = float(argv[i+1]); i += 2
        elif a == '--dvtx-near':    opt['dvtx_near'] = float(argv[i+1]); i += 2
        elif a == '--dmatch':       opt['dmatch'] = float(argv[i+1]); i += 2
        elif a == '--ftype':        opt['ftype'] = float(argv[i+1]); i += 2
        elif a == '--dhad':         opt['dhad'] = float(argv[i+1]); i += 2
        elif a == '--pr96-compat':  opt['compat'] = True; i += 1
        elif a == '--all-clusters': opt['all_clusters'] = True; i += 1
        elif a == '--tsv':          opt['tsv'] = argv[i+1]; i += 2
        elif a == '--summary-tsv':  opt['sumtsv'] = argv[i+1]; i += 2
        elif a == '--quiet':        opt['quiet'] = True; i += 1
        elif a in ('-h', '--help'): print(__doc__); return 0
        else:                       roots.append(a); i += 1
    if not roots:
        print(__doc__); return 2

    CLASSES = ['NEARVTX', 'HAD-A5', 'HAD-ADJ', 'EM-INT', 'TRK-COR', 'OTHER',
               'UNTYPED']
    rows, evrows = [], []
    for ed in event_dirs(roots):
        evt = os.path.basename(ed).replace('pr_evt', '')
        zp = os.path.join(ed, 'mabc-pr.zip')
        if not os.path.exists(zp):
            print(f'{evt}\tNO_ZIP')
            evrows.append(dict(evt=evt, status='NO_ZIP')); continue
        try:
            L = layers(zp)
        except Exception as e:
            print(f'{evt}\tBAD_ZIP {e}')
            evrows.append(dict(evt=evt, status='BAD_ZIP')); continue
        if 'track_fit-global' not in L or 'clustering-global' not in L:
            print(f'{evt}\tMISSING_LAYER')
            evrows.append(dict(evt=evt, status='MISSING_LAYER')); continue

        cl, tf = L['clustering-global'], L['track_fit-global']
        vv = L.get('vertices-global', {'x': [], 'y': [], 'z': [], 'q': [],
                                       'cluster_id': []})
        logp = os.path.join(ed, f'wct_pr_evt{evt}.log')
        drops = parse_pr54_drops(logp)
        audit = parse_pr30audit(logp)
        a5log = {} if opt['compat'] else parse_a5_census(logp)

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
            print(f'{evt}\tNO_MAIN_VERTEX')
            evrows.append(dict(evt=evt, status='NO_MAIN_VERTEX')); continue
        else:
            cids = [nucid]

        # typing (classifier mode only)
        ty, ty_status = None, ''
        if not opt['compat']:
            dump = load_calib(ed, evt)
            if dump is None:
                ty_status = 'NO_CALIB'
            else:
                ty = Typing(dump)
                if ty.main_vertex is not None and nuvtx is not None:
                    if np.linalg.norm(ty.main_vertex - nuvtx) > FRAME_TOL:
                        ty_status = 'FRAME_MISMATCH'
                        ty = None

        ev_flag = 0
        ev_cls = {c: 0 for c in CLASSES}
        ev_cls_flag = {c: 0 for c in CLASSES}
        ev_cls_q = {c: 0.0 for c in CLASSES}
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
            # other-cluster fit-less charge (the cross-cluster CUT detector)
            fitless_cids = [c2 for c2 in set(int(x) for x in cl_cid)
                            if c2 != cid and ((tf_cid == c2) & (tf_rc > 0)).sum() == 0]
            XC = xyz(cl, np.isin(cl_cid, fitless_cids)) if fitless_cids else np.zeros((0, 3))
            xc_tree = cKDTree(XC) if len(XC) else None
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
                hit54 = ''
                for r in drops:
                    mid = 0.5 * (r['v1'] + r['v2'])
                    dd = min(np.linalg.norm(G - r['v1'], axis=1).min(),
                             np.linalg.norm(G - r['v2'], axis=1).min(),
                             np.linalg.norm(G - mid, axis=1).min())
                    if dd < 3.0:
                        hit54 = f"pr54drop(cid{r['cid']},npts{r['npts']},{r['length']:.1f}cm,d{dd:.1f})"
                        break
                shape_ok = (gm.sum() >= opt['npts'] and plen >= opt['length']
                            and prms <= opt['rms'] and qfrac >= opt['qfrac'])

                if opt['compat']:
                    tracklike = shape_ok and 0.0 <= dvtx <= opt['dvtx_near']
                    if tracklike:
                        ev_flag += 1
                    rows.append(dict(evt=evt, cid=cid, grp=g, npts=int(gm.sum()),
                                     qfrac=qfrac, plen=plen, prms=prms, maxd=maxd,
                                     dvtx=dvtx, seg=seg, ang=ang,
                                     track=int(tracklike), hit54=hit54,
                                     cen='%.1f,%.1f,%.1f' % tuple(c)))
                    if tracklike:
                        hits.append(rows[-1])
                    continue

                # ---- classifier mode ----
                cls, sev, ev = '', 0, {}
                xcd, xcn = -1.0, 0
                if xc_tree is not None:
                    dx = xc_tree.query(G)[0]
                    xcd = float(dx.min())
                    xcn = int((dx < 5.0).sum())
                if shape_ok:
                    if ty is not None:
                        ev['ang_near'] = ang if ang >= 0 else 90.0
                        cls, sev, ev2 = classify_group(G, c, ax, dvtx, ty, opt)
                        ev.update(ev2)
                    else:
                        cls, sev = ('NEARVTX', 1) if 0.0 <= dvtx <= opt['dvtx_near'] \
                            else ('UNTYPED', 2)
                        ev = dict(f_m=-1.0, f_em=-1, f_trk=-1, f_a5=-1,
                                  own_pdg=ty_status or 'NO_TYPING',
                                  d_had=-1.0, hadseg=-1, d_a5=-1.0, d_sec=-1.0)
                flagged = int(bool(cls) and cls != 'EM-INT')
                if cls:
                    ev_cls[cls] += 1
                    ev_cls_q[cls] += qfrac
                    if flagged:
                        ev_cls_flag[cls] += 1
                        ev_flag += 1
                # A5 log context: nearest evaluated shower stem within dhad
                a5note = ''
                if a5log and ty is not None:
                    best = None
                    for oid, rec in a5log.items():
                        sid = ty.ord2stem.get(oid, -1)
                        if sid in ty.seg and len(ty.seg[sid]['poly']):
                            dd = float(np.min(np.linalg.norm(ty.seg[sid]['poly'] - c, axis=1)))
                            if best is None or dd < best[0]:
                                best = (dd, sid, rec)
                    if best and best[0] <= opt['dhad']:
                        r = best[2]
                        a5note = (f"a5(seg{best[1]},d{best[0]:.1f},"
                                  f"g{r.get('growth','?')},b{r.get('bragg','?')},"
                                  f"v{r.get('verdict','?')})")
                rows.append(dict(evt=evt, cid=cid, grp=g, npts=int(gm.sum()),
                                 qfrac=qfrac, plen=plen, prms=prms, maxd=maxd,
                                 dvtx=dvtx, seg=seg, ang=ang,
                                 cls=cls or 'sub', sev=sev, flag=flagged,
                                 f_m=round(ev.get('f_m', -1.0), 2),
                                 f_em=round(ev.get('f_em', -1.0), 2),
                                 f_trk=round(ev.get('f_trk', -1.0), 2),
                                 own_pdg=ev.get('own_pdg', ''),
                                 d_had=round(ev.get('d_had', -1.0), 1),
                                 hadseg=ev.get('hadseg', -1),
                                 d_a5=round(ev.get('d_a5', -1.0), 1),
                                 d_sec=round(ev.get('d_sec', -1.0), 1),
                                 xcd=round(xcd, 1), xcn=xcn,
                                 hit54=hit54, a5=a5note,
                                 cen='%.1f,%.1f,%.1f' % tuple(c)))
                if flagged:
                    hits.append(rows[-1])

            if not opt['quiet']:
                head = (f'{evt} cid {cid:4d}: {cm.sum():5d} chg / {fm.sum():4d} fit  '
                        f'uncovered {100*unc.mean():4.1f}% q {100*Q[unc].sum()/Q.sum():4.1f}%  '
                        f'{ng} grp  ')
                if opt['compat']:
                    head += f'TRACKLIKE {len(hits)}'
                else:
                    head += 'FLAGGED %d [%s]' % (len(hits), ' '.join(
                        f'{k}:{ev_cls[k]}' for k in CLASSES if ev_cls[k]))
                if audit:
                    head += (f"  oseg_iso_drop={audit.get('oseg_iso_drop','?')}"
                             f" oseg_reject={audit.get('oseg_reject','?')}")
                print(head)
                for h in hits:
                    if opt['compat']:
                        print(f'    -> grp{h["grp"]}: n={h["npts"]:4d} q={100*h["qfrac"]:4.1f}% '
                              f'len={h["plen"]:5.1f} rms={h["prms"]:4.2f} maxd={h["maxd"]:4.1f} '
                              f'dvtx={h["dvtx"]:5.1f} seg={h["seg"]} ang={h["ang"]:5.1f} '
                              f'cen=({h["cen"]}) {h["hit54"]}')
                    else:
                        print(f'    -> grp{h["grp"]} {h["cls"]:8s}: n={h["npts"]:4d} '
                              f'q={100*h["qfrac"]:4.1f}% len={h["plen"]:5.1f} '
                              f'rms={h["prms"]:4.2f} dvtx={h["dvtx"]:5.1f} '
                              f'f_m={h["f_m"]:.2f} f_em={h["f_em"]:.2f} '
                              f'd_had={h["d_had"]:.1f} xc={h["xcn"]} '
                              f'cen=({h["cen"]}) {h["hit54"]} {h["a5"]}')

        st = 'OK' if not ty_status else ty_status
        er = dict(evt=evt, status=st, flagged=ev_flag)
        for c in CLASSES:
            er[c] = ev_cls.get(c, 0)
            er[c + '_flag'] = ev_cls_flag.get(c, 0)
            er[c + '_q'] = round(ev_cls_q.get(c, 0.0), 4)
        evrows.append(er)

    if opt['tsv']:
        if opt['compat']:
            cols = ['evt', 'cid', 'grp', 'npts', 'qfrac', 'plen', 'prms', 'maxd',
                    'dvtx', 'seg', 'ang', 'track', 'hit54', 'cen']
        else:
            cols = ['evt', 'cid', 'grp', 'npts', 'qfrac', 'plen', 'prms', 'maxd',
                    'dvtx', 'seg', 'ang', 'cls', 'sev', 'flag', 'f_m', 'f_em',
                    'f_trk', 'own_pdg', 'd_had', 'hadseg', 'd_a5', 'd_sec',
                    'xcd', 'xcn', 'hit54', 'a5', 'cen']
        with open(opt['tsv'], 'w') as f:
            f.write('\t'.join(cols) + '\n')
            for r in rows:
                f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
        print(f'[wrote {opt["tsv"]}: {len(rows)} groups]')
    if opt['sumtsv'] and not opt['compat']:
        cols = ['evt', 'status', 'flagged'] + sum(
            ([c, c + '_flag', c + '_q'] for c in CLASSES), [])
        with open(opt['sumtsv'], 'w') as f:
            f.write('\t'.join(cols) + '\n')
            for r in evrows:
                f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
        print(f'[wrote {opt["sumtsv"]}: {len(evrows)} events]')

    if opt['compat']:
        nt = sum(r['track'] for r in rows)
        print(f'[total: {len(rows)} uncovered groups, {nt} track-like, '
              f'{len(set(r["evt"] for r in rows if r["track"]))} events flagged]')
    else:
        scored = [r for r in evrows if r['status'] in
                  ('OK', 'NO_CALIB', 'FRAME_MISMATCH')]
        print(f'[scored {len(scored)} events; '
              f'{sum(1 for r in evrows if r["status"] == "NO_ZIP")} NO_ZIP, '
              f'{sum(1 for r in evrows if r["status"] not in ("OK", "NO_ZIP"))} other-status]')
        for c in CLASSES:
            ne = sum(1 for r in scored if r.get(c + '_flag', 0) > 0) \
                if c != 'EM-INT' else sum(1 for r in scored if r.get(c, 0) > 0)
            ng2 = sum(r.get(c, 0) for r in scored)
            tag = 'flagged' if c != 'EM-INT' else 'present (never flagged)'
            print(f'  {c:8s}: {ng2:4d} groups, {ne:4d} events {tag}')
        nf = sum(1 for r in scored if r.get('flagged', 0) > 0)
        print(f'[events with any flagged group: {nf} / {len(scored)}]')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))

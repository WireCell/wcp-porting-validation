#!/usr/bin/env python3
"""Measure how much more permissive TaggerCheckSTM's containment box is than FC/TGM's.

Why this exists (docs/49_stm-containment-fv-inconsistency.md): TaggerCheckSTM is
the only one of the three taggers that is never handed a `fiducial` /
`fv_tolerance`, so its `Facade::cluster_fc_check(cluster, m_dv)` call falls back
to `FiducialUtils::inside_fiducial_volume` -> `DetectorVolumes::contained()` =
`contained_by(p).valid()` = the union of per-face `IAnodeFace::sensitive()`
bounding boxes, with NO margin.  tagger_check_tgm and tagger_check_fc both get
the single inset `sbnd_pr_fv` box.  A cluster whose end lands between the two
boundaries is "fully contained" to STM (no fit is ever attempted) and an exiter
to FC, on the same event, through the same function.

Two boxes, SBND (cm):

  STM     |x| in [0.45, 201.45], |y| <= 199.965, z in [0, 501.0]
          -- read off the run log's own geometry dump, not assumed:
             <AnodePlane:apa1> face:1 ... sensvol: [(4.5 -1999.65 0) --> (2014.5 1999.65 5010)]
             <AnodePlane:apa0> face:0 ... sensvol: [(-2014.5 -1999.65 0) --> (-4.5 1999.65 5010)]
          AnodePlane.cxx builds each face's sensitive box as x in [anode_x,
          cathode_x] intersected over the three planes, so it runs to the W
          plane, NOT to the FV_x* metadata.  The |x| < 0.45 cm CPA slab is a
          hole in the union.
  FC/TGM  |x| <= 198.55, |y| <= 196.312, z in [3.85, 497.15]
          -- sbnd_pr_fv (201.05 / 199.312 / 0.85..500.15) inset by
             sbnd_pr_fv_margins (x 2.5, y 3, z 3 both faces at -fvz 5 -fvzi 3,
             where only the ENDPOINT test uses 3).  ONE box, so no CPA hole.

For every bundle whose `stmfit` column reads `contained`, this reports the
largest distance by which one of the cluster's 6 axis-extreme points -- the
points `Cluster::get_extreme_wcps()` feeds the check -- lies OUTSIDE the FC/TGM
box.  A positive number is a cluster the STM tagger declined to fit and the FC
tagger called an exiter.

    python3 stm_fv_census.py                       # the three dq48v3 scan tags
    python3 stm_fv_census.py work-mcp10-dq48v3          # bare tag still resolves
    python3 stm_fv_census.py --detail 285185:21    # one cluster, per merge component
    python3 stm_fv_census.py --margins 2.5,3,3,3 work-mcp10-dq48v3   # other flags

IMPORTANT: the FC/TGM box below is the one built by `-fvx 2.5 -fvy 3 -fvz 5
-fvzi 3` -- the flag set every `dq48*` tag was produced with, where only the
ENDPOINT test uses the 3 cm downstream inset.  A scan tag run with different
margin flags needs `--margins x,y,zlo,zhi` or the numbers are silently wrong.

The --detail mode reads the Bee zip instead, because per-point
`real_cluster_id` (the pre-merge component each blob came from) exists ONLY
there -- it does not survive into the pctree.  That matters for merged clusters:
`get_extreme_wcps` runs over the whole cluster, so a grafted fragment can own
the axis extremes and the real track's ends are never tested.
"""
import os
import re
import sys
import glob
import json
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nusel_extract as ne  # noqa: E402

MM = 10.0                     # pctree arrays are WCT internal units (mm)
DEFAULT_TAGS = ['work-mcp10-dq48v3', 'work-mcp1000-dq48v3', 'work-mcp1000b-dq48v3']


def resolve_tag(tag, root):
    """Accept a bare tag dir name whether it sits at the top level or under
    archive/<campaign>/.  The doc-29..49 tags were moved there on 2026-07-25;
    see docs/work-tags.md for the tag -> location map."""
    if os.path.isdir(os.path.join(root, tag)):
        return tag
    import glob as _glob
    hits = _glob.glob(os.path.join(root, 'archive', '*', os.path.basename(tag)))
    return os.path.relpath(hits[0], root) if hits else tag

# STM box: union of per-face sensitive volumes (cm).
S_XLO, S_XHI, S_Y, S_ZLO, S_ZHI = 0.45, 201.45, 199.965, 0.0, 501.0
# FC/TGM box: sbnd_pr_fv inset by sbnd_pr_fv_margins (cm).  See --margins.
B_X, B_Y, B_ZLO, B_ZHI = 201.05, 199.312, 0.85, 500.15      # un-inset sbnd_pr_fv
F_X, F_Y, F_ZLO, F_ZHI = 198.55, 196.312, 3.85, 497.15      # + margins 2.5/3/3/3

WALLS = ('anode-x', 'y', 'z-up', 'z-down')


def outside_fc(p):
    """(distance outside the FC/TGM box, which wall).  Distance 0 => inside."""
    x, y, z = p
    d = (abs(x) - F_X, abs(y) - F_Y, F_ZLO - z, z - F_ZHI)
    k = int(np.argmax(d))
    return max(d[k], 0.0), (WALLS[k] if d[k] > 0 else '-')


def outside_box(p):
    """Distance by which p lies outside the UN-INSET sbnd_pr_fv box (0 if inside).

    Separates the two halves of the STM/FC gap: a positive value here means the
    endpoint leaves the fiducial volume itself, so the disagreement is not merely
    about margins.
    """
    x, y, z = p
    return max(abs(x) - B_X, abs(y) - B_Y, B_ZLO - z, z - B_ZHI, 0.0)


def inside_stm(p):
    """Is p inside the STM box (union of per-face sensitive volumes)?"""
    x, y, z = p
    return (S_XLO <= abs(x) <= S_XHI and abs(y) <= S_Y and S_ZLO <= z <= S_ZHI)


def cluster_points(fname):
    """{cluster_ident: (npts,3) array of DEFAULT-SCOPE points, cm}."""
    bp = ne.load_pctree(fname)
    live = [p for p in bp if re.fullmatch(r'pointtrees/\d+/live', p)][0]
    md = bp[live][0]
    items = bp[md['pointclouds']][0]['items']
    lpc = bp[md['lpcmaps']][0]['arrays']

    def ds(p):
        return bp[items[p]][0]['arrays']

    def arr(p, a):
        return bp[ds(p)[a]][1]

    ident = arr('cluster_scalar', 'ident').astype(int)
    map_cs = bp[lpc['cluster_scalar']][1].astype(int)
    map_3d = bp[lpc['3d']][1].astype(int)
    d3 = ds('3d')
    # The PR pipeline's default scope is the T0/pos-offset-corrected one
    # (common_corr_coords), which is what get_extreme_wcps sees via point3d().
    xn = 'x_t0cor' if 'x_t0cor' in d3 else 'x'
    yn = 'y_cor' if 'y_cor' in d3 else 'y'
    zn = 'z_cor' if 'z_cor' in d3 else 'z'
    x, y, z = arr('3d', xn), arr('3d', yn), arr('3d', zn)

    starts, ci, pos = {}, -1, 0
    for n in range(len(map_cs)):
        if map_cs[n]:
            ci += 1
            starts[ci] = [pos, 0]
        if map_3d[n]:
            if ci >= 0:
                starts[ci][1] += int(map_3d[n])
            pos += int(map_3d[n])
    out = {}
    for k, cid in enumerate(ident):
        s, n = starts.get(k, (0, 0))
        out[int(cid)] = np.c_[x[s:s + n], y[s:s + n], z[s:s + n]] / MM
    return out


def axis_extremes(P):
    """The 6 coordinate-axis extreme points get_extreme_wcps() collects.

    The 2 PCA-axis extremes it also collects are omitted, as are the 2 steiner
    boundary points cluster_fc_check appends.  For the outsideness that is safe:
    they are points of the same cluster, so including them could only make the
    number LARGER -- what is printed is a lower bound.  The `inside_stm` check is
    the opposite direction and NOT covered by this subset; it does not need to be,
    because the tagger's own is_fc=true verdict already tested every group.
    """
    idx = set()
    for col in range(3):
        idx.add(int(np.argmax(P[:, col])))
        idx.add(int(np.argmin(P[:, col])))
    return [P[j] for j in sorted(idx)]


def census(tags, root):
    want = {}
    for tag in tags:
        hdr, rs = ne.read_table(os.path.join(root, tag, 'nusel-table.tsv'), 'run')
        i = {c: n for n, c in enumerate(hdr)}
        for r in rs:
            if r[i['main_id']] == '-1' or r[i['stmfit']] != 'contained':
                continue
            want.setdefault((tag, int(r[i['event']])), []).append(
                (int(r[i['main_id']]), int(r[i['fc']]),
                 float(r[i['len_main_cm']]), int(r[i['n_frag']])))

    out = []
    for (tag, ev), cs in sorted(want.items()):
        pt = glob.glob(os.path.join(root, tag, f'ql_evt{ev}', f'pctree-evt{ev}.tar.gz'))
        if not pt:
            print(f'WARNING: no pctree for {tag} evt{ev}', file=sys.stderr)
            continue
        pts = cluster_points(pt[0])
        for cid, fc, length, nfrag in cs:
            P = pts.get(cid)
            if P is None or len(P) == 0:
                continue
            worst, wall, dbox, stmin = 0.0, '-', 0.0, True
            for p in axis_extremes(P):
                d, w = outside_fc(p)
                if d > worst:
                    worst, wall = d, w
                # Maximised INDEPENDENTLY of the inset-box worst point: the point
                # furthest outside the inset box need not be the one furthest
                # outside the un-inset box (different walls can dominate).
                dbox = max(dbox, outside_box(p))
                if not inside_stm(p):
                    stmin = False
            out.append((tag, ev, cid, fc, length, nfrag, worst, wall, stmin, dbox))

    print(f'{"tag":22s} {"event":>7s} {"cid":>4s} {"fc":>3s} {"len_cm":>7s} '
          f'{"nfr":>4s} {"outside_FC_cm":>13s} {"outside_FV_cm":>13s}  wall')
    for t, ev, cid, fc, L, nf, w, wall, _, db in sorted(out, key=lambda r: -r[6]):
        print(f'{t:22s} {ev:7d} {cid:4d} {fc:3d} {L:7.1f} {nf:4d} {w:13.2f} '
              f'{db:13.2f}  {wall}')

    n = len(out)
    band = [r for r in out if r[6] > 0]
    print(f'\n{n} "contained" clusters over {len(tags)} tag(s); {len(band)} have an '
          f'extreme point OUTSIDE the FC/TGM box ({100.0 * len(band) / max(n, 1):.0f}%)')
    if band:
        w = np.array([r[6] for r in band])
        print(f'   outsideness cm: median {np.median(w):.2f}  '
              f'p90 {np.percentile(w, 90):.2f}  max {w.max():.2f}')
        print(f'   by wall: ' + '  '.join(
            f'{k}={sum(1 for r in band if r[7] == k)}' for k in WALLS))
        # Split the gap: how many leave the FV box itself, margins aside?
        nb = sum(1 for r in band if r[9] > 0)
        print(f'   also outside the UN-INSET sbnd_pr_fv box (margins aside): '
              f'{nb} of {len(band)}')
        print(f'      by wall: ' + '  '.join(
            f'{k}={sum(1 for r in band if r[7] == k and r[9] > 0)}'
            f'/{sum(1 for r in band if r[7] == k)}' for k in WALLS))
    # Independent cross-check: the FC tagger's own flag on the same clusters.
    # Geometry and flag must agree 1:1 -- if they do, the box is the ONLY thing
    # discriminating the two verdicts (no signal-processing / dead-volume check
    # is involved).
    fc0 = {(r[0], r[1], r[2]) for r in out if r[3] == 0}
    geo = {(r[0], r[1], r[2]) for r in band}
    print(f'   FC=0 (not contained) on these: {len(fc0)};  '
          f'geometry says outside: {len(geo)};  agree: {len(fc0 & geo)}, '
          f'FC-only: {len(fc0 - geo)}, geometry-only: {len(geo - fc0)}')
    n_out_stm = sum(1 for r in out if not r[8])
    print(f'   extreme point outside the STM box too: {n_out_stm} '
          f'(must be 0 -- otherwise the tagger would have found an exit)')


def detail(spec, tags, root):
    ev, cid = (int(v) for v in spec.split(':'))
    for tag in tags:
        z = os.path.join(root, tag, f'ql_evt{ev}', 'mabc-all-apa.zip')
        if os.path.isfile(z):
            break
    else:
        sys.exit(f'ERROR: no mabc-all-apa.zip for evt{ev} under {tags}')
    with zipfile.ZipFile(z) as zf:
        name = [k for k in zf.namelist() if k.endswith('-clustering-global.json')][0]
        d = json.loads(zf.read(name))
    x, y, zc = (np.array(d[k], float) for k in ('x', 'y', 'z'))   # Bee is already cm
    cl = np.array(d['cluster_id'], int)
    rid = (np.array(d['real_cluster_id'], int) if 'real_cluster_id' in d
           else np.zeros_like(cl))
    m = cl == cid
    if not m.any():
        sys.exit(f'ERROR: cluster {cid} not in {z}')
    print(f'{z}\ncluster {cid}: {m.sum()} points, '
          f'merge components {sorted(set(rid[m].tolist()))}')
    for r in sorted(set(rid[m].tolist())):
        s = m & (rid == r)
        P = np.c_[x[s], y[s], zc[s]]
        a = int(np.argmax(((P - P[0]) ** 2).sum(1)))
        b = int(np.argmax(((P - P[a]) ** 2).sum(1)))
        print(f'\n  component real_cluster_id={r}: {s.sum()} pts, '
              f'farthest-pair {np.linalg.norm(P[a] - P[b]):.1f} cm')
        print(f'    x [{P[:, 0].min():8.2f},{P[:, 0].max():8.2f}]  '
              f'y [{P[:, 1].min():8.2f},{P[:, 1].max():8.2f}]  '
              f'z [{P[:, 2].min():8.2f},{P[:, 2].max():8.2f}]')
        for lab, p in (('endA', P[a]), ('endB', P[b])):
            dfc, wall = outside_fc(p)
            print(f'    {lab} ({p[0]:8.2f},{p[1]:8.2f},{p[2]:8.2f})  '
                  f'inside_STM={inside_stm(p)}  outside_FC={dfc:5.2f} cm ({wall})')


def set_margins(spec):
    """--margins x,y,zlo,zhi (cm, positive = inset) -> recompute the FC/TGM box.

    The defaults match `-fvx 2.5 -fvy 3 -fvz 5 -fvzi 3`, the flags every dq48*
    tag was built with.  Anything else MUST be passed here.
    """
    global F_X, F_Y, F_ZLO, F_ZHI
    mx, my, mzlo, mzhi = (float(v) for v in spec.split(','))
    F_X, F_Y = B_X - mx, B_Y - my
    F_ZLO, F_ZHI = B_ZLO + mzlo, B_ZHI - mzhi
    print(f'# FC/TGM box from margins {spec}: |x|<={F_X}, |y|<={F_Y}, '
          f'z in [{F_ZLO}, {F_ZHI}]')


if __name__ == '__main__':
    argv = sys.argv[1:]
    root = os.path.dirname(os.path.abspath(__file__))
    if argv and argv[0] == '--margins':
        set_margins(argv[1])
        argv = argv[2:]
    if argv and argv[0] == '--detail':
        detail(argv[1], [resolve_tag(t, root) for t in (argv[2:] or DEFAULT_TAGS)], root)
    else:
        census([resolve_tag(t, root) for t in (argv or DEFAULT_TAGS)], root)

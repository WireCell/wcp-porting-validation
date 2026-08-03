#!/usr/bin/env python3
"""doc pr/25: ClusteringProtectBundle cathode re-join census (SBND evt 489327
and population).  Answers: after protect_bundle splits a bundle cluster at
the cathode, which (retained, fragment) component pairs fail ONLY the
cathode_rejoin_dyz cut (dyz >= 4 cm) while passing dis/xcut, and does a
direction-agreement test (local PCA collinearity + perpendicular offset)
separate genuine oblique crossers from same-side junk stubs?

Usage:
    python3 pr25_rejoin_census.py scan   [--arm work-vfmcp1k-prodon]
    python3 pr25_rejoin_census.py direct [--arm work-vfmcp1k-prodon]  # after scan

`scan` reads every wct_pr_evt*.log + pctree-pr-evt*.tar.gz in the arm,
recomputes the four cathode_rejoin cuts (SBND operating point:
xcut=5cm, dyz=4cm, dis=8cm, cathode_x=0) on the fragments protect_bundle
created, and reports how many pairs fail ONLY dyz. Pickles the row list to
/home/xqian/tmp/pr25_scan_rows.pkl for `direct` to reuse.

`direct` recomputes local-PCA collinearity (dir-dir angle) and perpendicular
tip offset for every dyz-only-fail pair, at several Hough radii, plus a
dis 8-20cm control set (pairs that must stay rejected because dis fails) to
confirm the direction test does not admit unrelated pairs if dis alone were
relaxed.

See sbnd_xin/docs/pr/25_cathode-rejoin-direction.md for the full writeup.
"""
import argparse
import glob
import io
import json
import os
import pickle
import re
import sys
import tarfile

import numpy as np
from scipy.spatial import cKDTree

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
ROWS_PKL = "/home/xqian/tmp/pr25_scan_rows.pkl"

# SBND operating point, cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet
XCUT, DYZ, DIS = 5.0, 4.0, 8.0   # cm

RE_SPLIT = re.compile(
    r'cluster (\d+)( \(main\))?: (\d+) blobs -> retained (\d+) \+ (\d+) fragment\(s\) holding (\d+)')
RE_SUM = re.compile(
    r'split (\d+) bundle cluster\(s\) into (\d+) extra cluster\(s\) \((\d+) cathode re-join')


def _load_tensors(fname):
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    return {md['datapath']: (md, arrays.get(b)) for b, md in metas.items() if 'datapath' in md}


def clusters(fname, ev):
    """Return {ident: {idx: point-row indices, nb: nblobs}}, x, y, z (mm, pctree frame)."""
    bp = _load_tensors(fname)
    P = f'pointtrees/{ev}/live'
    m3 = bp[P + '/lpcmaps/arrays/3d'][1]
    mcs = bp[P + '/lpcmaps/arrays/cluster_scalar'][1]
    A = lambda n: bp[P + '/pointclouds/namedpcs/3d/arrays/' + n][1]
    x, y, z = A('x'), A('y'), A('z')
    ident = bp[P + '/pointclouds/namedpcs/cluster_scalar/arrays/ident'][1]
    out, nb = {}, {}
    ci, off = -1, 0
    for i in range(len(m3)):
        if mcs[i]:
            ci += 1
            out[ci] = []
            nb[ci] = 0
        n = m3[i]
        if n:
            out[ci].append((off, off + n))
            nb[ci] += 1
        off += n
    res = {}
    for ci, r in out.items():
        idx = np.concatenate([np.arange(a, b) for a, b in r]) if r else np.array([], int)
        res[int(ident[ci])] = dict(idx=idx, nb=nb[ci])
    return res, x, y, z


def cmd_scan(arm):
    rows, bad = [], []
    for d in sorted(glob.glob(os.path.join(BASE, arm, 'pr_evt*'))):
        ev = os.path.basename(d)[len('pr_evt'):]
        log = f'{d}/wct_pr_evt{ev}.log'
        try:
            txt = open(log, errors='ignore').read()
        except OSError:
            continue
        sp = RE_SPLIT.findall(txt)
        su = RE_SUM.search(txt)
        if not sp or not su:
            continue
        nfrag_tot, nrejoin = int(su.group(2)), int(su.group(3))
        parents = [int(m[0]) for m in sp]
        tf = f'{d}/pctree-pr-evt{ev}.tar.gz'
        if not os.path.exists(tf):
            continue
        try:
            res, x, y, z = clusters(tf, ev)
        except Exception as e:
            bad.append((ev, str(e)))
            continue
        ids = sorted(k for k in res if len(res[k]['idx']) > 0)
        frags = ids[-nfrag_tot:] if nfrag_tot <= len(ids) else []
        # sanity: fragment ident allocation is main*100+sub (ClusteringProtectBundle::alloc_ident);
        # total fragment blobs must equal the sum of "holding N" in the log.
        want_nb = sum(int(m[5]) for m in sp)
        got_nb = sum(res[k]['nb'] for k in frags)
        ok = (want_nb == got_nb)

        def P(k):
            i = res[k]['idx']
            return np.stack([x[i], y[i], z[i]], 1) / 10.0  # mm -> cm

        for fk in frags:
            B = P(fk)
            best = None
            for pk in parents:
                if pk not in res or len(res[pk]['idx']) == 0:
                    continue
                A = P(pk)
                t = cKDTree(A)
                dd, ii = t.query(B)
                j = int(np.argmin(dd))
                if best is None or dd[j] < best[0]:
                    best = (dd[j], pk, A[ii[j]], B[j])
            if best is None:
                continue
            dis, pk, p1, p2 = best
            dyz = float(np.hypot(p1[1] - p2[1], p1[2] - p2[2]))
            c1, c2, c3, c4 = dis < DIS, abs(p1[0]) < XCUT, abs(p2[0]) < XCUT, dyz < DYZ
            rows.append(dict(ev=ev, parent=pk, frag=fk, npts=len(B), dis=float(dis),
                             x1=float(p1[0]), x2=float(p2[0]), dyz=dyz,
                             pass_dis=c1, pass_x1=c2, pass_x2=c3, pass_dyz=c4,
                             nrejoin=nrejoin, sane=ok))
    pickle.dump(rows, open(ROWS_PKL, 'wb'))
    n_events = len(set(r['ev'] for r in rows))
    n_insane = sum(1 for r in rows if not r['sane'])
    print(f'events scanned with splits: {n_events}  fragment pairs: {len(rows)}  '
          f'insane: {n_insane}  errors: {len(bad)}')
    allpass = [r for r in rows if r['pass_dis'] and r['pass_x1'] and r['pass_x2'] and r['pass_dyz']]
    dyzonly = [r for r in rows if r['pass_dis'] and r['pass_x1'] and r['pass_x2'] and not r['pass_dyz']]
    print(f'pairs passing all four (would rejoin as shipped): {len(allpass)}')
    print(f'pairs failing ONLY dyz                          : {len(dyzonly)}')
    for r in sorted(dyzonly, key=lambda r: -r['npts']):
        print(f"  evt {r['ev']} parent {r['parent']:>3} frag {r['frag']:>4} npts={r['npts']:>5} "
              f"dis={r['dis']:5.2f} x1={r['x1']:7.2f} x2={r['x2']:7.2f} dyz={r['dyz']:6.2f}")
    return rows


def _local_dir(P, p, R):
    """Local PCA principal axis of P within radius R of p, oriented away from p."""
    m = np.linalg.norm(P - p, axis=1) < R
    Q = P[m]
    if len(Q) < 5:
        return None, 0
    c = Q.mean(0)
    _, s, vt = np.linalg.svd(Q - c, full_matrices=False)
    v = vt[0]
    if np.dot(p - c, v) < 0:
        v = -v
    return v, len(Q)


def pair_direction_metrics(arm, ev, pk, fk, radius=15.0):
    """dir-dir angle (deg) and max perpendicular tip offset (cm) for a
    (parent, fragment) pair, using local PCA within `radius` cm of each tip."""
    res, x, y, z = clusters(os.path.join(BASE, arm, f'pr_evt{ev}', f'pctree-pr-evt{ev}.tar.gz'), ev)
    P = lambda k: (lambda i: np.stack([x[i], y[i], z[i]], 1) / 10.0)(res[k]['idx'])
    A, B = P(pk), P(fk)
    t = cKDTree(A)
    d, i = t.query(B)
    j = int(np.argmin(d))
    p1, p2 = A[i[j]], B[j]
    v1, n1 = _local_dir(A, p1, radius)
    v2, n2 = _local_dir(B, p2, radius)
    if v1 is None or v2 is None:
        return None
    ang = np.degrees(np.arccos(np.clip(abs(np.dot(v1, v2)), -1, 1)))
    w = p2 - p1
    perp1 = np.linalg.norm(w - np.dot(w, v1) * v1)
    perp2 = np.linalg.norm((p1 - p2) - np.dot(p1 - p2, v2) * v2)
    drift_angle = np.degrees(np.arccos(abs(v1[0])))
    return dict(dis=float(d[j]), ang=float(ang), perp=float(max(perp1, perp2)),
                n1=n1, n2=n2, drift_angle=float(drift_angle))


def cmd_direct(arm):
    rows = pickle.load(open(ROWS_PKL, 'rb'))
    cand = [r for r in rows if r['pass_dis'] and r['pass_x1'] and r['pass_x2'] and not r['pass_dyz']]
    ctrl = [r for r in rows if r['pass_x1'] and r['pass_x2'] and not r['pass_dyz']
            and not r['pass_dis'] and r['dis'] < 20]
    print('=== CANDIDATES: fail dyz only (newly admissible under a direction fallback) ===')
    print(f"{'evt':>7} {'par':>4} {'frg':>4} {'npts':>6} {'dis':>6} {'dyz':>6} | "
          f"{'dir-ang':>8} {'perp':>6} {'th_drift':>8}")
    for r in sorted(cand, key=lambda r: -r['npts']):
        m = pair_direction_metrics(arm, r['ev'], r['parent'], r['frag'])
        s = (f"{m['ang']:8.1f} {m['perp']:6.2f} {m['drift_angle']:8.1f}" if m
             else "   (too few pts for a direction)")
        print(f"{r['ev']:>7} {r['parent']:>4} {r['frag']:>4} {r['npts']:>6} "
              f"{r['dis']:6.2f} {r['dyz']:6.2f} | {s}")
    print(f'\n=== CONTROLS: |x| ok, dyz fail, dis 8-20cm (n={len(ctrl)}) '
          f'-- must stay rejected by dis<{DIS} ===')
    for r in sorted(ctrl, key=lambda r: -r['npts'])[:12]:
        m = pair_direction_metrics(arm, r['ev'], r['parent'], r['frag'])
        s = f"{m['ang']:8.1f} {m['perp']:6.2f}" if m else "  (few pts)"
        print(f"{r['ev']:>7} {r['parent']:>4} {r['frag']:>4} {r['npts']:>6} "
              f"{r['dis']:6.2f} {r['dyz']:6.2f} | {s}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['scan', 'direct'])
    ap.add_argument('--arm', default='work-vfmcp1k-prodon')
    args = ap.parse_args()
    if args.cmd == 'scan':
        cmd_scan(args.arm)
    else:
        cmd_direct(args.arm)

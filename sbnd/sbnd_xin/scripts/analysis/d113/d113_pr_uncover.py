#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 6 -- the PR-stage missing-particle census, in 3-D, from the production PR products
(pr_evt<ID>/mabc-pr.zip + calib-pr-evt<ID>.json + tracking-pr.root; no rerun).

Two ways a particle that IS in the 3-D image can be missing from the neutrino candidate's reconstruction:
  A  UNHELD  -- a group of the candidate's own image points (the PR-stage pctree's 3d cloud = the stage-A image
               with the PR job's cluster ids, x_t0cor; the PR Bee clustering layer holds only the PR's re-sampled
               points, so it cannot serve) that no
               reconstructed particle holds: farther than --thr from every associated point (shower_track-global,
               the points the PR assigned to its tracks and showers) AND from every fitted point (track_fit-global).
               Shower bodies are held through their associated points, so a shower rind is NOT unheld here (the
               pr/96 instrument used the fit alone and had to veto rinds by distance to the vertex).
  B  NEARBY  -- a cluster the bundle does not contain whose points come within --near cm of the candidate's points:
               a companion of the same flash (COMPANION), a cluster of another flash (OTHER) or an unmatched one
               (UNMATCHED, rescue gid >= 1e6), read from the stage-A pctree's cluster_scalar.
Per event: the candidate's charge, the unheld groups (single-link --link) with npts, charge, Qfrac, PCA extent,
transverse rms, distance to the main vertex (q = 15000 in vertices-global), the nearest held point, and the nearby
clusters with distance, size, class.  Selection is done by d113_pr_summary.py.

Usage: d113_pr_uncover.py [--samples S...] [--jobs 8] [--outdir docs/113_figs/pr_census] [--thr 3.0] [--link 2.0] [--near 5.0]
"""
import argparse, collections, csv, glob, json, os, sys, zipfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C

MAIN_VTX_Q = 15000.0
GKEYS = ['sample', 'event', 'kind', 'gid', 'npts', 'Q', 'Qfrac', 'ext', 'rms', 'dvtx', 'dheld', 'cx', 'cy', 'cz', 'cluster', 'cl_class', 'cl_npts', 'cl_gid', 'dcand', 'n_cand_pts', 'Qcand', 'f_absorbed', 'dfit', 'dfit_max', 'dfit_p90', 'f_far12']


def layers(zp):
    z = zipfile.ZipFile(zp); out = {}
    for n in z.namelist():
        for tag in ('clustering-global', 'track_fit-global', 'shower_track-global', 'vertices-global'):
            if n.endswith(tag + '.json'):
                out[tag] = json.loads(z.read(n))
    return out


def single_link(P, r):
    tree = cKDTree(P); n = len(P); lab = -np.ones(n, int); k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]; lab[i] = k
        while stack:
            j = stack.pop()
            for m in tree.query_ball_point(P[j], r):
                if lab[m] < 0:
                    lab[m] = k; stack.append(m)
        k += 1
    return lab, k


def image_points(pt, ev):
    """stage-A image: per point (x_t0cor, y, z) in cm and a charge (the collection-plane value, the mean of the
    live planes where W is 0), with the cluster ident of each point (lpcmaps walk as in pr25_rejoin_census.py)."""
    P = f'pointtrees/{ev}/live'
    A = lambda n: pt.bp[P + '/pointclouds/namedpcs/3d/arrays/' + n][1]
    x, y, z = A('x_t0cor') / 10.0, A('y') / 10.0, A('z') / 10.0
    qs = np.c_[A('ucharge_val'), A('vcharge_val'), A('wcharge_val')].astype(float)
    qw = qs[:, 2].copy(); m = qw <= 0; qw[m] = qs[m][:, :2].mean(axis=1) if m.any() else 0
    m3 = pt.bp[P + '/lpcmaps/arrays/3d'][1]; mcs = pt.bp[P + '/lpcmaps/arrays/cluster_scalar'][1]
    cl = np.zeros(len(x), int); ci = -1; off = 0
    for i in range(len(m3)):
        if mcs[i]:
            ci += 1
        n = int(m3[i]); cl[off:off + n] = ci; off += n
    ident = pt.cl_ident[cl]
    return np.c_[x, y, z], np.clip(qw, 0, None), ident


HELD_MODE = 'fit+shower'
STAGEB = 'pr150s0'


def held_points(L, mode='fit+shower'):
    """The points the PR RECONSTRUCTED.  'fit+shower' (the owner's missing-prong definition, 2026-09-19): the fitted
    trajectory points (track_fit-global) plus the points a SHOWER holds (shower_track-global with q = 15000; a
    shower's energy is the charge of its associated points).  Points a TRACK merely associated (q = 0) do not count:
    a track's energy comes from its fitted trajectory, so a prong absorbed into a neighbouring track's association
    is lost.  'asc' = every associated point (the lenient reading used first in doc 113 sec 6)."""
    held = []
    if 'track_fit-global' in L and len(L['track_fit-global'].get('x', [])):
        t = L['track_fit-global']; held.append(np.c_[t['x'], t['y'], t['z']])
    if 'shower_track-global' in L and len(L['shower_track-global'].get('x', [])):
        t = L['shower_track-global']; X = np.c_[t['x'], t['y'], t['z']]
        if mode == 'asc':
            held.append(X)
        else:
            q = np.asarray(t['q'], float); held.append(X[q >= 15000.0 - 1e-6])
    return np.vstack(held) if held else np.zeros((0, 3))


def one_event(args):
    sample, ev, thr, link, near = args
    d = f'{C.SX}/work-{sample}-{STAGEB}/pr_evt{ev}'
    rows = []
    try:
        B = C.load_bundle(d, ev)
        if B is None:
            return rows, dict(sample=sample, event=ev, has=0)
        L = layers(f'{d}/mabc-pr.zip')
        held = held_points(L, HELD_MODE)
        vtx = None
        if 'vertices-global' in L:
            v = L['vertices-global']; m = np.asarray(v['q'], float) == MAIN_VTX_Q
            if m.any():
                vtx = np.c_[v['x'], v['y'], v['z']][m][0]
        pt = C.PCTree(f'{d}/pctree-pr-evt{ev}.tar.gz', ev)   # the PR-stage pctree: same 3d cloud, the PR job's cluster ids (after unmerge)
        P, q, ident = image_points(pt, ev)
        ok = np.abs(P[:, 0]) < 1e4                      # x_t0cor sentinel guard (doc pdvd/33 trap)
        U = set(B['union']); inb = np.isin(ident, list(U)) & ok
        gid_of = dict(zip(pt.cl_ident.tolist(), pt.cl_gid.tolist()))
        main_gid = gid_of.get(B['main'], -1)
        Qcand = float(q[inb].sum()); ncand = int(inb.sum())
        summ = dict(sample=sample, event=ev, has=1, main=B['main'], n_union=len(U), n_cand_pts=ncand, Qcand=Qcand, n_held=len(held))
        Pc = P[inb]; qc = q[inb]
        # track-ASSOCIATED points (shower_track q = 0): a group sitting on them was absorbed into a track's
        # association (its charge is not in the energy); a group far from them is held by nothing at all
        asc0 = np.zeros((0, 3)); fitp = np.zeros((0, 3))
        if 'shower_track-global' in L and len(L['shower_track-global'].get('x', [])):
            t = L['shower_track-global']; X = np.c_[t['x'], t['y'], t['z']]; asc0 = X[np.asarray(t['q'], float) < 1.0]
        if 'track_fit-global' in L and len(L['track_fit-global'].get('x', [])):
            t = L['track_fit-global']; fitp = np.c_[t['x'], t['y'], t['z']]
        t_asc0 = cKDTree(asc0) if len(asc0) else None; t_fit = cKDTree(fitp) if len(fitp) else None
        if len(held) and len(Pc):
            # self-check: the held points must sit on the image (coordinate frames agree)
            dm, _ = cKDTree(Pc).query(held); summ['f_held_on_image'] = float((dm < 1.0).mean())   # the PR re-samples on a half-pitch lattice: held-to-image median 0.5 cm
            dh, _ = cKDTree(held).query(Pc)
            unh = dh > thr
            summ['f_unheld_pts'] = float(unh.mean()); summ['f_unheld_q'] = float(qc[unh].sum() / max(1.0, Qcand))
            if unh.sum() >= 3:
                lab, k = single_link(Pc[unh], link)
                Pu, qu, du = Pc[unh], qc[unh], dh[unh]
                for g in range(k):
                    m = lab == g
                    if m.sum() < 5:
                        continue
                    X = Pu[m]; Qg = float(qu[m].sum())
                    c = X.mean(axis=0); Y = X - c
                    _, sv, _ = np.linalg.svd(Y, full_matrices=False)
                    ext = float(2 * sv[0] / np.sqrt(len(X))) if len(X) > 1 else 0.0
                    rms = float(np.sqrt((sv[1] ** 2 + sv[2] ** 2) / len(X))) if len(X) > 2 else 0.0
                    dv = float(np.min(np.linalg.norm(X - vtx, axis=1))) if vtx is not None else float('nan')
                    f_abs = float((t_asc0.query(X)[0] <= 3.0).mean()) if t_asc0 is not None else 0.0
                    if t_fit is not None:
                        df = t_fit.query(X)[0]; dfit = float(df.min()); dfit_max = float(df.max()); dfit_p90 = float(np.percentile(df, 90)); f_far12 = float((df > 12.0).mean())
                    else:
                        dfit = dfit_max = dfit_p90 = float('nan'); f_far12 = 1.0
                    rows.append(dict(sample=sample, event=ev, kind='UNHELD', gid=g, npts=int(m.sum()), Q=round(Qg, 1), Qfrac=round(Qg / max(1.0, Qcand), 4),
                                     ext=round(ext, 1), rms=round(rms, 2), dvtx=round(dv, 1), dheld=round(float(du[m].min()), 1),
                                     cx=round(c[0], 1), cy=round(c[1], 1), cz=round(c[2], 1), cluster=-1, cl_class='OWN', cl_npts=0, cl_gid=0, dcand=0.0,
                                     n_cand_pts=ncand, Qcand=round(Qcand, 1), f_absorbed=round(f_abs, 3), dfit=round(dfit, 1), dfit_max=round(dfit_max, 1), dfit_p90=round(dfit_p90, 1), f_far12=round(f_far12, 3)))
        # nearby clusters (B): image clusters the bundle does not contain, within `near` of the candidate's image
        if len(Pc):
            tc = cKDTree(Pc)
            for c_id in np.unique(ident[~inb & ok]).tolist():
                m = (ident == c_id) & ok
                X = P[m]
                dd, _ = tc.query(X)
                dmin = float(dd.min())
                if dmin > near:
                    continue
                g = gid_of.get(int(c_id), -2)
                klass = 'UNMATCHED' if g >= 1000000 else ('COMPANION' if g == main_gid else ('OTHER' if g >= 0 else 'UNKNOWN'))
                dv = float(np.min(np.linalg.norm(X - vtx, axis=1))) if vtx is not None else float('nan')
                cc = X.mean(axis=0); Y = X - cc
                if len(X) > 2:
                    _, sv, _ = np.linalg.svd(Y, full_matrices=False)
                else:
                    sv = np.zeros(3)
                rows.append(dict(sample=sample, event=ev, kind='NEARBY', gid=-1, npts=int(m.sum()), Q=round(float(q[m].sum()), 1), Qfrac=round(float(q[m].sum()) / max(1.0, Qcand), 4),
                                 ext=round(float(2 * sv[0] / np.sqrt(len(X))) if len(X) > 1 else 0.0, 1), rms=round(float(np.sqrt((sv[1] ** 2 + sv[2] ** 2) / len(X))) if len(X) > 2 else 0.0, 2),
                                 dvtx=round(dv, 1), dheld=float('nan'), cx=round(cc[0], 1), cy=round(cc[1], 1), cz=round(cc[2], 1), cluster=int(c_id), cl_class=klass,
                                 cl_npts=int(m.sum()), cl_gid=g, dcand=round(dmin, 1), n_cand_pts=ncand, Qcand=round(Qcand, 1)))
        return rows, summ
    except Exception as e:
        return rows, dict(sample=sample, event=ev, has=0, error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', default=list(C.SAMPLES))
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--outdir', default=f'{C.SX}/docs/113_figs/pr_census')
    ap.add_argument('--thr', type=float, default=3.0)
    ap.add_argument('--link', type=float, default=2.0)
    ap.add_argument('--near', type=float, default=5.0)
    ap.add_argument('--stageB', default='pr150s0')
    ap.add_argument('--held', default='fit+shower', choices=['fit+shower', 'asc'])
    a = ap.parse_args()
    global HELD_MODE, STAGEB; HELD_MODE = a.held; STAGEB = a.stageB
    os.makedirs(a.outdir, exist_ok=True)
    for s in a.samples:
        evs = sorted(int(p.rstrip('/').split('pr_evt')[1]) for p in glob.glob(f'{C.SX}/work-{s}-{a.stageB}/pr_evt*/'))
        jobs = [(s, e, a.thr, a.link, a.near) for e in evs]
        rows, summ = [], []
        with ProcessPoolExecutor(a.jobs) as ex:
            for r, sm in ex.map(one_event, jobs, chunksize=8):
                rows.extend(r); summ.append(sm)
        with open(f'{a.outdir}/{s}.tsv', 'w') as f:
            w = csv.DictWriter(f, fieldnames=GKEYS, delimiter='\t', extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        json.dump(summ, open(f'{a.outdir}/{s}.events.json', 'w'))
        print(f'{s}: {len(evs)} events, {sum(1 for x in summ if x.get("has"))} with a candidate, {len(rows)} rows', flush=True)


if __name__ == '__main__':
    main()

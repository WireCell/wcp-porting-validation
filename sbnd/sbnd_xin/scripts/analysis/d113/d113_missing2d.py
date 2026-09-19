#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 2 -- the missing-charge census: 2-D SP charge that no 3-D object covers, per event, per
stage, on the production products (no rerun).  Conventions verified by d113_selftest.py (K=4 slice index,
half-open wire bounds, T_proj_data rank = base + apa*nwire + wip, charge rounded).

Per event, per (apa, plane), grids over (wire-in-plane x 4-tick slice bin):
  S0    L0 charge: the dnnsp frame summed over the 4 ticks of each slice (bad-mask channels zeroed)
  A     L1 activity: MaskSlices' thresholding re-applied (== ctpc on the slices that have a blob)
  cov2  covered by a stage-A live blob (any cluster)          -- the 3-D image
  covD  covered by a dead-fork blob (masked 2-view, 500-tick)  -- dead regions, never counted as missing
  cov3  covered by a live blob of an IN-SCOPE cluster (cluster_t0 set) -- what clustering/PR can see
  cov4  a cell of the PR bundle's T_proj_data (union of its stage clusters)  -- what the candidate holds
Uncovered cells (S0 > floor_p, not bad, not covD) are labelled into 8-connected components at three levels:
  level 2: ~cov2                      -> imaging classes  THRESH / NOBLOB-SLICE / NOBLOB-INSLICE
  level 3: cov2 & ~cov3               -> SCOPE   (imaged, but the cluster is out of scope: no t0 / out of volume)
  level 4: cov3 & ~cov4               -> BUNDLE  (imaged and in scope, but not in the PR candidate)
floor_p = 0.5 x median S0 over the event's ctpc-active cells of plane p (data-driven; reported per event).
Each component carries: ncell, Q, wire/slice extent, bbox, the L1-active fraction, the no-blob-slice fraction,
the zero-summary fraction, the number of planes with activity in its slices, its distance to the nearest PR
bundle cell of the same (apa, plane) and to the nearest in-scope blob, and the bundle's own L0 charge in that
plane (for Qfrac).  Selection (near the candidate, >= 2-plane same-slice coincidence, size) is applied later by
d113_summary.py so the cuts can be swept.

Usage: d113_missing2d.py extract --sample S [--groups 0,1] [--jobs 8] [--out docs/113_figs/census/<S>.tsv]
"""
import argparse, collections, csv, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C

NSL = (C.NTICK + C.TICK_SPAN - 1) // C.TICK_SPAN     # 857 slice bins
MIN_CELL = 6                                        # extraction floor; the summary applies the real cuts
FIELDS = ['sample', 'event', 'level', 'cls', 'apa', 'plane', 'comp', 'ncell', 'Q', 'wmin', 'wmax', 'smin', 'smax',
          'wext', 'sext', 'cw', 'cs', 'f_act', 'f_noblobslice', 'f_zsum', 'nplane_act', 'd_bundle', 'd_scope', 'Qbundle_plane',
          'floor', 'has_bundle', 'pca_rms', 'x_apa', 'qmax', 'f_topwire', 'f_topslice', 'nwire_q', 'f_cov4', 'fq_cov4']


def blob_grid(pt, apa, plane, sel, live=True):
    g = np.zeros((C.NWIRE[plane], NSL), dtype=bool)
    if live:
        idx = np.nonzero((pt.b_apa == apa) & sel)[0]
        wmin, wmax, smin, smax = pt.b_wmin[idx, plane], pt.b_wmax[idx, plane], pt.b_smin[idx], pt.b_smax[idx]
    else:
        d = pt.dead
        if d is None:
            return g
        idx = np.nonzero(d['apa'] == apa)[0]
        wmin, wmax, smin, smax = d['wmin'][idx, plane], d['wmax'][idx, plane], d['smin'][idx], d['smax'][idx]
    for lo, hi, s0, s1 in zip(wmin.tolist(), wmax.tolist(), smin.tolist(), smax.tolist()):
        if hi <= lo:
            continue
        b0, b1 = s0 // C.TICK_SPAN, s1 // C.TICK_SPAN
        g[max(0, lo):min(C.NWIRE[plane], hi), max(0, b0):min(NSL, b1 + 1)] = True
    return g


def one_group(args):
    sample, gid, outdir, stageA, stageB = args
    gdir = f'{C.SX}/work-{sample}-{stageA}/g{gid}'
    rows, evrows = [], []
    t0 = time.time()
    for ev, fr in C.iter_frames(gdir):
        try:
            rows_e, evrow = one_event(sample, ev, fr, stageA, stageB)
            rows.extend(rows_e); evrows.append(evrow)
        except Exception as e:  # keep the group going; the summary reports the failures
            evrows.append(dict(sample=sample, event=ev, error=repr(e)))
    out = f'{outdir}/{sample}-g{gid}.tsv'
    with open(out, 'w') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter='\t', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    with open(f'{outdir}/{sample}-g{gid}.events.json', 'w') as f:
        json.dump(evrows, f)
    return sample, gid, len(evrows), len(rows), time.time() - t0


def one_event(sample, ev, fr, stageA='d102m', stageB='pr150s0'):
    bad = C.mask_frame(fr)
    A, thr, zero = C.recompute_activity(fr)
    F = fr['frame']; ch = fr['channels']
    nch, nq = F.shape
    pad = NSL * C.TICK_SPAN - nq
    S0 = np.pad(F, ((0, 0), (0, pad))).reshape(nch, NSL, C.TICK_SPAN).sum(axis=2)
    apa_c, plane_c, wip_c = C.plane_of_channel(ch)
    pt = C.PCTree(f'{C.SX}/work-{sample}-{stageA}/ql_evt{ev}/pctree-evt{ev}.tar.gz', ev)
    inscope = pt.in_scope()
    scope_sel = inscope[pt.blob_cluster]
    B = C.load_bundle(f'{C.SX}/work-{sample}-{stageB}/pr_evt{ev}', ev)
    has_bundle = int(B is not None and bool(B['cells']))
    # ctpc slice presence per apa, and the per-plane activity planes per (apa, slice) for the tiling reading
    ctpc_slices = {apa: set() for apa in range(C.NAPA)}
    for (apa, p), d in pt.ctpc.items():
        ctpc_slices[apa].update((d['slice_index'] // C.TICK_SPAN).tolist())
    rows = []
    ev_summary = dict(sample=sample, event=ev, has_bundle=has_bundle, main=(B or {}).get('main'), floor={},
                      Q0={}, Qunc2={}, Qunc3={}, Qunc4={}, Qbundle={}, nblob=int(len(pt.b_smin)), nclus=int(len(pt.cl_ident)),
                      nclus_inscope=int(inscope.sum()))
    for apa in range(C.NAPA):
        # activity planes per slice for this apa (from the recomputed activity, which includes no-blob slices)
        rows_apa = apa_c == apa
        act_planes = np.zeros((3, NSL), dtype=bool)
        for p in range(3):
            m = rows_apa & (plane_c == p)
            act_planes[p] = (A[m] > 0).any(axis=0)
        nplane_act = act_planes.sum(axis=0)                      # per slice bin
        noblob_slice = np.array([b not in ctpc_slices[apa] for b in range(NSL)])
        for p in range(3):
            m = rows_apa & (plane_c == p)
            idx = np.nonzero(m)[0]
            w = wip_c[idx]
            G0 = np.zeros((C.NWIRE[p], NSL)); G0[w] = S0[idx]
            GA = np.zeros((C.NWIRE[p], NSL), dtype=bool); GA[w] = A[idx] > 0
            Gbad = np.zeros(C.NWIRE[p], dtype=bool); Gbad[w] = bad[idx]
            Gz = np.zeros(C.NWIRE[p], dtype=bool); Gz[w] = zero[idx]
            cov2 = blob_grid(pt, apa, p, np.ones(len(pt.b_smin), bool))
            covD = blob_grid(pt, apa, p, None, live=False)
            cov3 = blob_grid(pt, apa, p, scope_sel)
            cov4 = np.zeros_like(cov2)
            Qb = 0.0; tree = None
            if has_bundle and (apa, p) in B['cells']:
                bw, bs, bq = B['cells'][(apa, p)]
                ok = (bw >= 0) & (bw < C.NWIRE[p]) & (bs >= 0) & (bs < NSL)
                cov4[bw[ok], bs[ok]] = True
                Qb = float(G0[bw[ok], bs[ok]].sum())
                tree = cKDTree(np.c_[bw[ok], bs[ok]].astype(float))
            # data-driven floor: ctpc-active cells of this plane
            act_cells = G0[GA]
            floor = 0.5 * float(np.median(act_cells)) if len(act_cells) else 0.0
            ev_summary['floor'][f'a{apa}p{p}'] = floor
            live = (G0 > floor) & ~Gbad[:, None] & ~covD
            U2 = live & ~cov2
            U3 = live & cov2 & ~cov3
            U4 = live & cov3 & ~cov4
            ev_summary['Q0'][f'a{apa}p{p}'] = float(G0[live].sum())
            ev_summary['Qunc2'][f'a{apa}p{p}'] = float(G0[U2].sum())
            ev_summary['Qunc3'][f'a{apa}p{p}'] = float(G0[U3].sum())
            ev_summary['Qunc4'][f'a{apa}p{p}'] = float(G0[U4].sum())
            ev_summary['Qbundle'][f'a{apa}p{p}'] = Qb
            # in-scope blob cells for the distance-to-image reading
            sc_tree = None
            sw, ss = np.nonzero(cov3)
            if len(sw):
                sc_tree = cKDTree(np.c_[sw, ss].astype(float))
            struct = np.ones((3, 3), dtype=int)
            for level, U in ((2, U2), (3, U3), (4, U4)):
                lab, n = ndimage.label(U, structure=struct)
                if n == 0:
                    continue
                objs = ndimage.find_objects(lab)
                for k, sl in enumerate(objs, start=1):
                    if sl is None:
                        continue
                    cm = lab[sl] == k
                    ncell = int(cm.sum())
                    if ncell < MIN_CELL:
                        continue
                    q = G0[sl][cm]
                    Q = float(q.sum())
                    ww, ss_ = np.nonzero(cm)
                    ww = ww + sl[0].start; ss_ = ss_ + sl[1].start
                    f_act = float(GA[ww, ss_].mean())
                    f_nbs = float(noblob_slice[ss_].mean())
                    f_z = float(Gz[ww].mean())
                    npl = float(np.median(nplane_act[ss_]))
                    P = np.c_[ww, ss_].astype(float)
                    d_b = float(tree.query(P)[0].min()) if tree is not None else float('nan')
                    d_s = float(sc_tree.query(P)[0].min()) if sc_tree is not None else float('nan')
                    cw = float(np.average(ww, weights=q)); cs = float(np.average(ss_, weights=q))
                    # hot-channel / stripe readings: charge concentration on one wire or one slice
                    qw = np.bincount(ww - ww.min(), weights=q); qs = np.bincount(ss_ - ss_.min(), weights=q)
                    f_topwire = float(qw.max() / Q) if Q > 0 else 0.0
                    f_topslice = float(qs.max() / Q) if Q > 0 else 0.0
                    nwire_q = int((qw > 0.05 * Q).sum())
                    qmax = float(q.max())
                    # the PR stage re-tiles the candidate from ctpc: the share of this component that the
                    # candidate's T_proj_data holds anyway (recovered downstream, so not an energy loss)
                    c4 = cov4[ww, ss_]
                    f_cov4 = float(c4.mean()); fq_cov4 = float(q[c4].sum() / Q) if Q > 0 else 0.0
                    if ncell >= 3:
                        X = P - P.mean(axis=0)
                        _, sv, _ = np.linalg.svd(X, full_matrices=False)
                        pca_rms = float(sv[-1] / np.sqrt(ncell))
                    else:
                        pca_rms = 0.0
                    if level == 2:
                        if f_act < 0.5:
                            cls = 'THRESH'
                        elif f_nbs >= 0.5:
                            cls = 'NOBLOB-SLICE'
                        else:
                            cls = 'NOBLOB-INSLICE'
                    elif level == 3:
                        cls = 'SCOPE'
                    else:
                        cls = 'BUNDLE'
                    rows.append(dict(sample=sample, event=ev, level=level, cls=cls, apa=apa, plane=p, comp=k, ncell=ncell,
                                     Q=round(Q, 1), wmin=int(ww.min()), wmax=int(ww.max()), smin=int(ss_.min()), smax=int(ss_.max()),
                                     wext=int(ww.max() - ww.min() + 1), sext=int(ss_.max() - ss_.min() + 1), cw=round(cw, 1), cs=round(cs, 1),
                                     f_act=round(f_act, 3), f_noblobslice=round(f_nbs, 3), f_zsum=round(f_z, 3), nplane_act=npl,
                                     d_bundle=round(d_b, 2), d_scope=round(d_s, 2), Qbundle_plane=round(Qb, 1), floor=round(floor, 1),
                                     has_bundle=has_bundle, pca_rms=round(pca_rms, 2), x_apa=apa,
                                     qmax=round(qmax, 1), f_topwire=round(f_topwire, 3), f_topslice=round(f_topslice, 3), nwire_q=nwire_q,
                                     f_cov4=round(f_cov4, 3), fq_cov4=round(fq_cov4, 3)))
    return rows, ev_summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--sample', required=True)
    e.add_argument('--groups', default=None, help='comma list of group ids (default all)')
    e.add_argument('--jobs', type=int, default=8)
    e.add_argument('--outdir', default=f'{C.SX}/docs/113_figs/census')
    e.add_argument('--stageA', default='d102m', help='stage-A arm suffix (work-<s>-<stageA>); the frames come from d102m groups either way')
    e.add_argument('--stageB', default='pr150s0', help='stage-B arm suffix holding tracking-pr.root / calib dumps')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    gdirs = C.group_dirs(a.sample)   # frames-dnn.tar.bz2 live in the production d102m groups (symlinked into any d113 stage-A arm)
    gids = [int(g.rsplit('/g', 1)[1]) for g in gdirs]
    if a.groups:
        want = {int(x) for x in a.groups.split(',')}
        gids = [g for g in gids if g in want]
    jobs = [(a.sample, g, a.outdir, a.stageA, a.stageB) for g in gids]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        for sample, gid, nev, nrow, dt in ex.map(one_group, jobs):
            print(f'{sample} g{gid}: {nev} events, {nrow} components, {dt:.0f} s', flush=True)
    print(f'done {len(jobs)} groups in {time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()

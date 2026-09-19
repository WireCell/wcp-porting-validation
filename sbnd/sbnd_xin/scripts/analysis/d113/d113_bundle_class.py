#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 6 -- the PR-stage loss: for every BUNDLE flag (imaged charge near the candidate that the
candidate's T_proj_data does not hold), which stage-A cluster owns the blobs under it?
  OWN        the blobs belong to a cluster of the candidate's own bundle (T_proj_data union U): the PR had the
             cluster and left this part out (no fitted segment covers it)
  COMPANION  a cluster with the same matched_flash_gid as the main cluster that the bundle does not include
  OTHER      a cluster of another flash gid (a cosmic or another interaction in time), or unmatched (gid >= 1e6)
Per flag: the cell fractions by class, the dominant cluster (ident, npoints, gid, t0), and the candidate's main
cluster id / gid.  Cells = bbox cells covered by a live blob and not in T_proj_data (the census's level-4 mask,
without the charge floor; counts, not charge).

Usage: d113_bundle_class.py [--flags docs/113_figs/113_flags.tsv] [--out docs/113_figs/113_bundle_class.tsv] [--jobs 8]
"""
import argparse, collections, csv, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C
from d113_missing2d import NSL, blob_grid

KEYS = ['sample', 'event', 'cls', 'apa', 'plane', 'comp', 'ncell', 'Q', 'Qfrac', 'd_bundle', 'main', 'main_gid', 'n_union',
        'f_own', 'f_companion', 'f_other', 'f_unmatched', 'bclass', 'dom_cluster', 'dom_npts', 'dom_gid', 'dom_t0', 'dom_in_union']


def one_event(args):
    sample, ev, rows = args
    try:
        pt = C.PCTree(f'{C.SX}/work-{sample}-d102m/ql_evt{ev}/pctree-evt{ev}.tar.gz', ev)
        B = C.load_bundle(f'{C.SX}/work-{sample}-pr150s0/pr_evt{ev}', ev)
    except Exception as e:
        return [dict(r, bclass=f'ERR {e!r}') for r in rows]
    if B is None:
        return [dict(r, bclass='NO-BUNDLE') for r in rows]
    U = set(B['union']); main = B['main']
    ident = pt.cl_ident.tolist(); gid = pt.cl_gid.tolist(); t0 = pt.cl_t0.tolist()
    idx_of = {c: i for i, c in enumerate(ident)}
    main_gid = gid[idx_of[main]] if main in idx_of else -1
    npts = np.bincount(pt.blob_cluster, weights=pt.b_npts, minlength=len(ident))
    out = []
    grids = {}
    for r in rows:
        apa, p = int(r['apa']), int(r['plane'])
        if (apa, p) not in grids:
            # per-cluster coverage id grid: last blob wins (blobs rarely overlap in one plane/slice)
            g = np.full((C.NWIRE[p], NSL), -1, dtype=np.int64)
            sel = np.nonzero(pt.b_apa == apa)[0]
            for i in sel.tolist():
                lo, hi = int(pt.b_wmin[i, p]), int(pt.b_wmax[i, p])
                if hi <= lo:
                    continue
                b0, b1 = int(pt.b_smin[i]) // C.TICK_SPAN, int(pt.b_smax[i]) // C.TICK_SPAN
                g[max(0, lo):min(C.NWIRE[p], hi), max(0, b0):min(NSL, b1 + 1)] = pt.blob_cluster[i]
            cov4 = np.zeros((C.NWIRE[p], NSL), dtype=bool)
            if (apa, p) in B['cells']:
                bw, bs, _ = B['cells'][(apa, p)]
                ok = (bw >= 0) & (bw < C.NWIRE[p]) & (bs >= 0) & (bs < NSL)
                cov4[bw[ok], bs[ok]] = True
            grids[(apa, p)] = (g, cov4)
        g, cov4 = grids[(apa, p)]
        w0, w1, s0, s1 = int(r['wmin']), int(r['wmax']) + 1, int(r['smin']), int(r['smax']) + 1
        sub = g[w0:w1, s0:s1]; c4 = cov4[w0:w1, s0:s1]
        cells = (sub >= 0) & ~c4
        n = int(cells.sum())
        rec = dict(r); rec.update(main=main, main_gid=main_gid, n_union=len(U))
        if n == 0:
            rec.update(f_own=0, f_companion=0, f_other=0, f_unmatched=0, bclass='EMPTY', dom_cluster=-1, dom_npts=0, dom_gid=-1, dom_t0=0, dom_in_union=0)
            out.append(rec); continue
        cl = sub[cells]
        cnt = collections.Counter(cl.tolist())
        f_own = f_comp = f_oth = f_unm = 0.0
        for ci, k in cnt.items():
            cid = ident[ci]; gg = gid[ci]
            if cid in U: f_own += k
            elif gg >= 1000000: f_unm += k
            elif gg == main_gid: f_comp += k
            else: f_oth += k
        f_own /= n; f_comp /= n; f_oth /= n; f_unm /= n
        dom = max(cnt, key=cnt.get)
        bclass = max((('OWN', f_own), ('COMPANION', f_comp), ('OTHER', f_oth), ('UNMATCHED', f_unm)), key=lambda kv: kv[1])[0]
        rec.update(f_own=round(f_own, 3), f_companion=round(f_comp, 3), f_other=round(f_oth, 3), f_unmatched=round(f_unm, 3), bclass=bclass,
                   dom_cluster=ident[dom], dom_npts=int(npts[dom]), dom_gid=gid[dom], dom_t0=round(t0[dom], 1), dom_in_union=int(ident[dom] in U))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flags', default=f'{C.SX}/docs/113_figs/113_flags.tsv')
    ap.add_argument('--out', default=f'{C.SX}/docs/113_figs/113_bundle_class.tsv')
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    fl = [r for r in csv.DictReader(open(a.flags), delimiter='\t') if r['cls'] == 'BUNDLE']
    by = collections.defaultdict(list)
    for r in fl:
        by[(r['sample'], int(r['event']))].append(r)
    jobs = [(s, e, rs) for (s, e), rs in sorted(by.items())]
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for out in ex.map(one_event, jobs, chunksize=4):
            rows.extend(out)
    with open(a.out, 'w') as f:
        w = csv.DictWriter(f, fieldnames=KEYS, delimiter='\t', extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    # tables
    ev_cls = collections.defaultdict(set); q_cls = collections.defaultdict(float)
    for r in rows:
        ev_cls[r['bclass']].add((r['sample'], r['event'])); q_cls[r['bclass']] += float(r['Q'])
    nev = len(by)
    print(f'BUNDLE flags {len(rows)} in {nev} events -> {a.out}')
    print('class: flags, events, charge share')
    tot = sum(q_cls.values())
    for k in ('OWN', 'COMPANION', 'OTHER', 'UNMATCHED', 'EMPTY'):
        n = sum(1 for r in rows if r['bclass'] == k)
        print(f'  {k:10s} {n:5d} flags  {len(ev_cls[k]):4d} events  {100*q_cls[k]/max(1,tot):5.1f} % of the flagged charge')
    own = [r for r in rows if r['bclass'] == 'OWN']
    if own:
        qf = np.array([float(r['Qfrac']) for r in own])
        print(f'OWN: Qfrac median {np.median(qf):.3f}, >= 0.10 in {(qf>=0.1).sum()} flags / {len({(r["sample"],r["event"]) for r in own if float(r["Qfrac"])>=0.1})} events')


if __name__ == '__main__':
    main()

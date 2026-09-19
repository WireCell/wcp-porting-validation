#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 4 -- which imaging step lost each flagged component?  Reads the counterfactual imaging
arms (work-<s>-d113cf<arm>/g<K>/icluster-apa<A>-active.npz, whole-group npz keyed by cluster_<evt>_*) and,
for every flagged imaging-class component, the fraction of its cells that the arm's blobs cover:
  cf0     today's binary, no knob  -> > 0 means the imaging DID make the blob and clustering deleted it (CLUSDEL)
  pdoff   ProjectionDeghosting dryrun
  isdoff  InSliceDeghosting dryrun
  dt0     data-driven zero-summary fallback threshold
  th0     2.5 sigma slicing threshold
Component cells are rebuilt from the production products as in the census (bbox cells with S0 > floor that no
production blob covers).  Verdict per component = the first arm in the order cf0 > isdoff > pdoff > dt0 > th0
that covers >= 50 % of the cells; TILING if none does.

Usage: d113_counterfactual.py [--flags docs/113_figs/113_flags.tsv] [--out docs/113_figs/113_counterfactual.tsv]
"""
import argparse, collections, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C
from d113_missing2d import NSL, blob_grid

ARMS = ('cf0', 'isdoff', 'pdoff', 'dt0', 'th0')


def npz_cov(npz, ev, plane):
    """coverage grid (wires x slice bins) of the imaging npz blobs of one event/plane (b rows: sliceid col 5,
    bounds cols 8-13 half-open, verified by the self-test)"""
    b = npz[f'cluster_{ev}_bnodes']
    g = np.zeros((C.NWIRE[plane], NSL), dtype=bool)
    lo, hi = b[:, 8 + 2 * plane].astype(int), b[:, 9 + 2 * plane].astype(int)
    sb = b[:, 5].astype(int)
    for l, h, s in zip(lo.tolist(), hi.tolist(), sb.tolist()):
        if h > l and 0 <= s < NSL:
            g[max(0, l):min(C.NWIRE[plane], h), s] = True
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flags', default=f'{C.SX}/docs/113_figs/113_flags.tsv')
    ap.add_argument('--out', default=f'{C.SX}/docs/113_figs/113_counterfactual.tsv')
    ap.add_argument('--arms', default=','.join(ARMS))
    a = ap.parse_args()
    arms = a.arms.split(',')
    fl = [r for r in csv.DictReader(open(a.flags), delimiter='\t') if r['cls'] != 'BUNDLE']
    by_ev = collections.defaultdict(list)
    for r in fl:
        by_ev[(r['sample'], int(r['event']))].append(r)
    # group of each event
    gof = {}
    for s in {k[0] for k in by_ev}:
        for g in C.group_dirs(s):
            gid = int(g.rsplit('/g', 1)[1])
            for e in C.group_events(g):
                gof[(s, e)] = gid
    rows = []
    groups = collections.defaultdict(list)
    for (s, ev), rs in by_ev.items():
        groups[(s, gof[(s, ev)])].append(ev)
    for (s, gid), evs in sorted(groups.items()):
        gdir = f'{C.SX}/work-{s}-d102m/g{gid}'
        npzs = {}
        for arm in arms:
            d = f'{C.SX}/work-{s}-d113cf{arm}/g{gid}'
            npzs[arm] = {}
            for apa in range(C.NAPA):
                f = f'{d}/icluster-apa{apa}-active.npz'
                try:
                    npzs[arm][apa] = np.load(f) if os.path.exists(f) else None
                except Exception as e:   # an arm still being written
                    print(f'WARN unreadable {f}: {e!r}', flush=True); npzs[arm][apa] = None
        for ev, fr in C.iter_frames(gdir, wanted=set(evs)):
            C.mask_frame(fr)
            F, ch = fr['frame'], fr['channels']
            pad = NSL * C.TICK_SPAN - F.shape[1]
            S0 = np.pad(F, ((0, 0), (0, pad))).reshape(len(ch), NSL, C.TICK_SPAN).sum(axis=2)
            apa_c, plane_c, wip_c = C.plane_of_channel(ch)
            pt = C.PCTree(f'{C.SX}/work-{s}-d102m/ql_evt{ev}/pctree-evt{ev}.tar.gz', ev)
            for r in by_ev[(s, ev)]:
                apa, p = int(r['apa']), int(r['plane'])
                idx = np.nonzero((apa_c == apa) & (plane_c == p))[0]
                G0 = np.zeros((C.NWIRE[p], NSL)); G0[wip_c[idx]] = S0[idx]
                cov2 = blob_grid(pt, apa, p, np.ones(len(pt.b_smin), bool))
                w0, w1, s0, s1 = int(r['wmin']), int(r['wmax']) + 1, int(r['smin']), int(r['smax']) + 1
                cells = (G0[w0:w1, s0:s1] > float(r['floor'])) & ~cov2[w0:w1, s0:s1]
                n = int(cells.sum()); q = float(G0[w0:w1, s0:s1][cells].sum())
                out = dict(sample=s, event=ev, cls=r['cls'], apa=apa, plane=p, comp=r['comp'], ncell=n, Q=round(q, 1),
                           wmin=w0, wmax=w1 - 1, smin=s0, smax=s1 - 1)
                verdict = 'TILING'
                for arm in arms:
                    z = npzs[arm][apa]
                    if z is None or f'cluster_{ev}_bnodes' not in z:
                        out[f'f_{arm}'] = float('nan'); continue
                    cv = npz_cov(z, ev, p)
                    f = float(cv[w0:w1, s0:s1][cells].sum() / max(1, n))
                    fq = float(G0[w0:w1, s0:s1][cells & cv[w0:w1, s0:s1]].sum() / max(1.0, q))
                    out[f'f_{arm}'] = round(f, 3); out[f'fq_{arm}'] = round(fq, 3)
                    if verdict == 'TILING' and fq >= 0.5:
                        verdict = arm
                out['verdict'] = verdict
                rows.append(out)
            print(s, ev, len(by_ev[(s, ev)]), 'components', flush=True)
    keys = ['sample', 'event', 'cls', 'apa', 'plane', 'comp', 'ncell', 'Q', 'wmin', 'wmax', 'smin', 'smax'] + \
           [f'f_{a}' for a in arms] + [f'fq_{a}' for a in arms] + ['verdict']
    with open(a.out, 'w') as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter='\t', extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    # table: verdict counts by class, charge-weighted
    print('\nverdict (first arm covering >= 50 % of the component charge) by class: n components / charge share')
    for cls in ('NOBLOB-SLICE', 'NOBLOB-INSLICE', 'THRESH'):
        rs = [r for r in rows if r['cls'] == cls]
        if not rs:
            continue
        tot = sum(r['Q'] for r in rs)
        c = collections.Counter(r['verdict'] for r in rs); cq = collections.defaultdict(float)
        for r in rs:
            cq[r['verdict']] += r['Q']
        print(f'  {cls:15s} n={len(rs):4d}  ' + '  '.join(f'{k}: {c[k]} ({100*cq[k]/max(1,tot):.0f} %)' for k in ['cf0'] + [x for x in arms if x != 'cf0'] + ['TILING']))
    print('mean covered-charge fraction per arm (all flagged components):')
    for arm in arms:
        v = np.array([r.get(f'fq_{arm}', np.nan) for r in rows], float)
        print(f'  {arm:7s} {np.nanmean(v):.3f}  (>= 0.5 in {(v >= 0.5).sum()} / {np.isfinite(v).sum()})')


if __name__ == '__main__':
    main()

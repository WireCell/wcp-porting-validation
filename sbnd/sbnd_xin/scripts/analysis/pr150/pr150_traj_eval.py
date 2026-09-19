#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 3 -- the neutrino-PR trajectory fit against the 2-D measurements and the 3-D image,
per event, on SBND production output (no STM dump, no trace needed).

Port by duplication of the doc pdvd/111 / 115 instruments (d111_eval.py Ridge / P1-P4, d115_proj_resid.py
2-D closure) onto the SBND PR fit: tracking-pr.root T_rec_charge (the main cluster's PR fit rows, x/y/z in cm,
fractional global channel ranks pu/pv/pw, slice pt), T_proj_data (the cluster's measured 2-D cells with charge
and the fit's predicted charge), T_bad_ch (dead channels x tick ranges), the Bee clustering layer of mabc-pr.zip
(the 3-D image points with charge), and calib-pr-evt<ID>.json (main_vertex.cluster_id).  The pdvd scripts stay
untouched; the definitions are copied verbatim and cited.

The SBND "main cluster" of the PR stage is a BUNDLE: its fit rows carry real_cluster_id = <stage cluster>*1000 +
<running index> over every stage cluster the bundle merged (evt 175896: 13 of them; the main id alone covers 34 %
of the fit's slices, the union 100 %).  The measured 2-D cells and the 3-D image are therefore taken over the
UNION U = {main_vertex.cluster_id} + {real_cluster_id // 1000} of the fit rows, both in T_proj_data (keyed by the
stage cluster id in the PR job) and in the Bee clustering layer (same ids).

Per event (main cluster bundle), row = one fit point:
  rows        fit rows; len_cm the polyline length
  R2D_<p>     share of LIVE rows whose fractional projection is > 1 wire from the nearest charged cell of the
              cluster on the row's rounded slice, plane p in U V W  (d115_proj_resid.py 'gt1'; R2D = W)
  R2Dpm1_<p>  the same over slices s-1, s, s+1 (the smallest |d|)
  med_d_<p>   median |d_exact| over live rows with a cell on the slice (wires)
  noslice_<p> share of live rows whose cluster has no charged cell on the row's slice in that plane
  dead_<p>    share of rows whose rounded channel is dead at the row's tick (T_bad_ch)
  P1          share of rows > 1 cm from the image ridge (d111_stage_attrib.Ridge: charge-weighted local image
              axis, 1 cm voxels, 3 cm neighbourhood); P1_2 the > 2 cm share
  P2          share of rows off-charge in >= 1 live plane: no measured cell within +-1 wire on the row's own slice
  P3          Bee-visible holes: runs of >= 3 consecutive rows with q < 0, per 10 m of fit
  P4          image coverage: main-cluster image charge within 1.5 cm of the fit polyline / all its image charge
  qneg        share of rows with q < 0
  wig1        share of rows whose 3-row wiggle > 1 cm (d111_eval.wiggle)
  uncov       pr/149 M3: charge in cells predicted < 10 % of measured / measured (T_proj_data of the main id only,
              as pr/149 defined it); uncov_union = the unweighted mean over the bundle's stage clusters

Usage:
  pr150_traj_eval.py extract --arm TAG --samples S... [--out docs/pr/150_figs/traj/<TAG>-<S>.tsv] [--jobs 8]
  pr150_traj_eval.py compare --a TAG --b TAG --samples S... [--strata 149_figs/149_stage2_selection.tsv]
"""
import argparse, collections, csv, glob, json, math, os, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import uproot
from scipy.spatial import cKDTree

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
OUT = os.path.join(SX, 'docs/pr/150_figs/traj')
BASE = (0, 3968, 7936)            # SBND global channel-rank plane bases (calib meta 'base'; d42_proj2d_resid.py:40)
THR, RLOC, VOX = 1.0, 3.0, 1.0    # d111_stage_attrib.py:47-49
TICKS_PER_SLICE = 4               # calib meta nticks_per_slice (both APAs)
PLANES = 'UVW'; KEYS = ('pu', 'pv', 'pw')


class Ridge:  # verbatim d111_stage_attrib.Ridge
    def __init__(self, P, q):
        self.ok = len(P) >= 5
        if not self.ok:
            return
        w = np.clip(q, 1, None)
        key = np.floor(P / VOX).astype(np.int64)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        W = np.bincount(inv, weights=w)
        C = np.c_[[np.bincount(inv, weights=w * P[:, k]) for k in range(3)]].T / W[:, None]
        tree = cKDTree(C)
        self.cen = np.zeros_like(C); self.ax = np.zeros_like(C)
        for j, nb in enumerate(tree.query_ball_point(C, RLOC)):
            ww = W[nb]; X = C[nb]
            c = np.average(X, axis=0, weights=ww)
            if len(nb) >= 2:
                _, _, vt = np.linalg.svd((X - c) * np.sqrt(ww / ww.sum())[:, None], full_matrices=False)
                a = vt[0]
            else:
                a = np.array([1.0, 0, 0])
            self.cen[j] = c; self.ax[j] = a
        self.tree = tree

    def offset(self, X):
        X = np.atleast_2d(X)
        if not self.ok or len(X) == 0:
            return np.full(len(X), np.nan)
        dv, j = self.tree.query(X)
        V = X - self.cen[j]
        t = np.sum(V * self.ax[j], axis=1)
        perp = np.linalg.norm(V - t[:, None] * self.ax[j], axis=1)
        return np.where(dv > RLOC, np.maximum(perp, dv), perp)


def holes(q, minrows=3):  # verbatim d111_eval.holes
    n, i, k = 0, 0, len(q)
    while i < k:
        if q[i] < 0:
            j = i
            while j < k and q[j] < 0:
                j += 1
            n += (j - i) >= minrows; i = j
        else:
            i += 1
    return n


def wiggle(F, k=3):  # verbatim d111_eval.wiggle
    out = np.zeros(len(F))
    for i in range(k, len(F) - k):
        a, b = F[i - k], F[i + k]; nn = np.linalg.norm(b - a)
        if nn > 0:
            u = (b - a) / nn; v = F[i] - a; out[i] = np.linalg.norm(v - (v @ u) * u)
    return out


def dist_to_polyline(P, X):
    """distance of each X to the polyline P (segment-wise, via the two segments at the nearest vertex)."""
    if len(P) == 0 or len(X) == 0:
        return np.full(len(X), np.inf)
    if len(P) == 1:
        return np.linalg.norm(X - P[0], axis=1)
    tree = cKDTree(P)
    _, j = tree.query(X)
    best = np.full(len(X), np.inf)
    for k in (-1, 0):
        i0 = np.clip(j + k, 0, len(P) - 2); i1 = i0 + 1
        A, B = P[i0], P[i1]; AB = B - A; L2 = np.sum(AB * AB, axis=1)
        t = np.where(L2 > 0, np.sum((X - A) * AB, axis=1) / np.where(L2 > 0, L2, 1), 0.0)
        t = np.clip(t, 0, 1)
        Q = A + t[:, None] * AB
        best = np.minimum(best, np.linalg.norm(Q - X, axis=1))
    return best


def nearest_d(chans, w):  # verbatim d115_proj_resid.nearest_d
    if chans is None or len(chans) == 0:
        return np.nan
    k = np.searchsorted(chans, w)
    best = np.inf
    for j in (k - 1, k):
        if 0 <= j < len(chans):
            d = w - chans[j]
            if abs(d) < abs(best):
                best = d
    return best


def load_cells(f):
    """cluster -> plane -> slice -> sorted channel array (charged cells only); PR job: plain cluster ids."""
    pd = f['T_proj_data'].arrays(library='np')
    cells = collections.defaultdict(lambda: [collections.defaultdict(list) for _ in range(3)])
    uncov = {}
    if len(pd['cluster_id']) == 0:
        return {}, uncov
    for cid, ch, ts, q, qp in zip(pd['cluster_id'][0], pd['channel'][0], pd['time_slice'][0], pd['charge'][0], pd['charge_pred'][0]):
        ch = np.asarray(ch).astype(int); ts = np.asarray(ts).astype(int); qa = np.asarray(q, float); m = qa > 0
        pl = np.searchsorted(np.array(BASE[1:]), ch, side='right')
        cl = int(cid)
        for p in range(3):
            mm = m & (pl == p)
            for c, s in zip(ch[mm].tolist(), ts[mm].tolist()):
                cells[cl][p][s].append(c)
        if qa.sum() > 0:
            uncov[cl] = float(qa[np.asarray(qp, float) < 0.1 * qa].sum() / qa.sum())
    out = {cl: [{s: np.array(sorted(set(v))) for s, v in pl.items()} for pl in planes] for cl, planes in cells.items()}
    return out, uncov


def dead_lookup(f):
    b = f['T_bad_ch'].arrays(library='np')
    d = collections.defaultdict(list)
    for c, t0, t1 in zip(b['chid'].tolist(), b['start_time'].tolist(), b['end_time'].tolist()):
        d[int(c)].append((t0, t1))
    return d


def one_event(args):
    d, ev = args
    out = dict(event=ev, has=0)
    calib = f'{d}/calib-pr-evt{ev}.json'; root = f'{d}/tracking-pr.root'; zp = f'{d}/mabc-pr.zip'
    if not (os.path.exists(calib) and os.path.exists(root) and os.path.exists(zp)):
        return out
    cal = json.load(open(calib))
    mv = cal.get('main_vertex') or {}
    cid = mv.get('cluster_id')
    if cid is None:
        return out
    f = uproot.open(root)
    t = f['T_rec_charge'].arrays(library='np')
    m = t['cluster_id'] == cid
    n = int(m.sum())
    if n < 2:
        return out
    out['has'] = 1; out['main_cid'] = cid; out['rows'] = n
    F = np.c_[t['x'][m], t['y'][m], t['z'][m]]; q = t['q'][m]
    step = np.linalg.norm(np.diff(F, axis=0), axis=1)
    out['len_cm'] = float(step.sum())
    rc = t['real_cluster_id'][m]
    U = sorted({int(cid)} | {int(r) // 1000 for r in rc.tolist() if r > 0})
    out['n_union'] = len(U)
    cells, uncov = load_cells(f)
    out['uncov'] = uncov.get(cid, float('nan'))                      # pr/149 M3 stays main-id-only (comparable)
    uq = [uncov[c] for c in U if c in uncov]
    out['uncov_union'] = float(np.mean(uq)) if uq else float('nan')  # unweighted mean over the bundle's clusters
    pl = [{}, {}, {}]                                                # union of the bundle's cells per plane/slice
    for c in U:
        for p in range(3):
            for sl, arr in cells.get(c, [{}, {}, {}])[p].items():
                pl[p][sl] = np.array(sorted(set(pl[p].get(sl, np.array([], int)).tolist()) | set(arr.tolist())))
    dead = dead_lookup(f)
    s0 = np.round(t['pt'][m]).astype(int)
    tick = t['pt'][m] * TICKS_PER_SLICE
    off_any = np.zeros(n, bool); live_any = np.zeros(n, bool)
    for p, key in enumerate(KEYS):
        w = t[key][m]
        de = np.full(n, np.nan); dp = np.full(n, np.nan); isdead = np.zeros(n, bool)
        for i in range(n):
            ch = int(round(w[i]))
            isdead[i] = any(t0 <= tick[i] <= t1 for (t0, t1) in dead.get(ch, ()))
            de[i] = nearest_d(pl[p].get(s0[i]), w[i])
            best = np.inf
            for ds in (-1, 0, 1):
                dd = nearest_d(pl[p].get(s0[i] + ds), w[i])
                if np.isfinite(dd) and abs(dd) < abs(best):
                    best = dd
            dp[i] = best if np.isfinite(best) else np.nan
        live = ~isdead
        P = PLANES[p]
        out[f'dead_{P}'] = float(isdead.mean())
        out[f'nlive_{P}'] = int(live.sum())
        has = live & np.isfinite(de)
        out[f'noslice_{P}'] = float((live & ~np.isfinite(de)).sum() / max(live.sum(), 1))
        out[f'R2D_{P}'] = float(((np.abs(de) > 1) & has).sum() / max(live.sum(), 1))
        out[f'R2Dpm1_{P}'] = float(((np.abs(dp) > 1) & live & np.isfinite(dp)).sum() / max(live.sum(), 1))
        out[f'med_d_{P}'] = float(np.median(np.abs(de[has]))) if has.any() else float('nan')
        off_any |= live & ~(np.isfinite(dp) & (np.abs(dp) <= 1.0))   # P2: no cell within +-1 wire on the slice
        live_any |= live
    out['P2'] = float(off_any.sum() / n)
    # image ridge from the Bee clustering layer (stage-A 3-D cloud, per point charge)
    z = zipfile.ZipFile(zp)
    name = [k for k in z.namelist() if k.endswith('-clustering-global.json')]
    if name:
        c = json.loads(z.read(name[0]))
        cl = np.asarray(c['cluster_id']); mi = np.isin(cl, U)
        Pimg = np.c_[c['x'], c['y'], c['z']][mi]; Wimg = np.clip(np.asarray(c['q'], float)[mi], 1, None)
        R = Ridge(Pimg, Wimg)
        if R.ok:
            dd = R.offset(F)
            out['P1'] = float((dd > 1).mean()); out['P1_2'] = float((dd > 2).mean())
            out['ridge_med'] = float(np.median(dd))
            near = dist_to_polyline(F, Pimg) < 1.5
            out['P4'] = float(Wimg[near].sum() / Wimg.sum()) if Wimg.sum() > 0 else float('nan')
            out['img_pts'] = int(mi.sum())
    out['P3'] = float(holes(q) / max(out['len_cm'] / 1000.0, 1e-9))
    out['holes'] = int(holes(q))
    out['qneg'] = float((q < 0).mean())
    out['wig1'] = float((wiggle(F) > 1).mean())
    return out


FIELDS = ['event', 'has', 'main_cid', 'n_union', 'rows', 'len_cm', 'uncov', 'uncov_union', 'P1', 'P1_2', 'ridge_med', 'P2', 'P3', 'holes', 'P4', 'img_pts',
          'qneg', 'wig1'] + [f'{k}_{P}' for P in PLANES for k in ('R2D', 'R2Dpm1', 'med_d', 'noslice', 'dead', 'nlive')]


def extract(a):
    os.makedirs(OUT, exist_ok=True)
    for s in a.samples:
        d0 = f'{SX}/work-{s}-{a.arm}'
        evs = sorted(int(os.path.basename(p)[6:]) for p in glob.glob(f'{d0}/pr_evt*'))
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(one_event, [(f'{d0}/pr_evt{e}', e) for e in evs]))
        p = f'{OUT}/{a.arm}-{s}.tsv'
        with open(p, 'w') as fh:
            fh.write('\t'.join(FIELDS) + '\n')
            for r in rows:
                fh.write('\t'.join(('%.5g' % r[k] if isinstance(r.get(k), float) else str(r.get(k, ''))) for k in FIELDS) + '\n')
        print(f'{a.arm} {s}: {len(rows)} events, {sum(r["has"] for r in rows)} with a main-cluster fit -> {p}')


def load(arm, samples):
    R = {}
    for s in samples:
        for r in csv.DictReader(open(f'{OUT}/{arm}-{s}.tsv'), delimiter='\t'):
            if r['has'] == '1':
                R[(s, int(r['event']))] = {k: (float(v) if v not in ('', 'nan') else float('nan')) for k, v in r.items() if k != 'event'}
    return R


def sign_p(k, n):
    if n == 0:
        return float('nan')
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


def compare(a):
    A, B = load(a.a, a.samples), load(a.b, a.samples)
    common = sorted(set(A) & set(B))
    strata = {}
    if a.strata:
        for r in csv.DictReader(open(a.strata), delimiter='\t'):
            strata.setdefault(r['stratum'], set()).add((r['sample'], int(r['event'])))
    groups = [('all', set(common))] + [(k, v & set(common)) for k, v in sorted(strata.items())]
    print(f'# pr150 traj compare A={a.a} B={a.b} samples={a.samples} A={len(A)} B={len(B)} common={len(common)}')
    print('| group | n | metric | A | B | delta (row-weighted) | paired better / worse (|d|>0.005) | sign p |')
    print('|---|---|---|---|---|---|---|---|')
    for g, keys in groups:
        keys = sorted(keys)
        if not keys:
            continue
        for met, better in (('R2D_W', 'lower'), ('R2D_U', 'lower'), ('R2D_V', 'lower'), ('R2Dpm1_W', 'lower'), ('P1', 'lower'), ('P1_2', 'lower'),
                            ('P2', 'lower'), ('P3', 'lower'), ('P4', 'higher'), ('uncov', 'lower'), ('qneg', 'lower'), ('wig1', 'lower'),
                            ('med_d_W', 'lower'), ('noslice_W', 'lower'), ('len_cm', None), ('rows', None)):
            va = np.array([A[k].get(met, np.nan) for k in keys]); vb = np.array([B[k].get(met, np.nan) for k in keys])
            wa = np.array([A[k]['rows'] for k in keys]); wb = np.array([B[k]['rows'] for k in keys])
            ok = np.isfinite(va) & np.isfinite(vb)
            if met in ('len_cm', 'rows'):
                ma, mb = float(np.nansum(va)), float(np.nansum(vb)); dl = mb - ma
                print(f'| {g} | {ok.sum()} | {met} (sum) | {ma:.1f} | {mb:.1f} | {dl:+.1f} | | |')
                continue
            ma = float(np.sum(va[ok] * wa[ok]) / np.sum(wa[ok])); mb = float(np.sum(vb[ok] * wb[ok]) / np.sum(wb[ok]))
            d = vb[ok] - va[ok]
            if better == 'lower':
                bet, wor = int((d < -0.005).sum()), int((d > 0.005).sum())
            else:
                bet, wor = int((d > 0.005).sum()), int((d < -0.005).sum())
            p = sign_p(min(bet, wor), bet + wor)
            print(f'| {g} | {ok.sum()} | {met} | {ma:.4f} | {mb:.4f} | {mb - ma:+.4f} | {bet} / {wor} | {p:.3g} |')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract'); e.add_argument('--arm', required=True); e.add_argument('--samples', nargs='+', required=True); e.add_argument('--jobs', type=int, default=8)
    c = sub.add_parser('compare'); c.add_argument('--a', required=True); c.add_argument('--b', required=True); c.add_argument('--samples', nargs='+', required=True)
    c.add_argument('--strata', default=None)
    a = ap.parse_args()
    extract(a) if a.cmd == 'extract' else compare(a)

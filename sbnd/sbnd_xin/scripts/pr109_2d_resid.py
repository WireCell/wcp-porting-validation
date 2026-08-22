#!/usr/bin/env python3
"""doc pr/109 -- near-vertex measured-vs-predicted 2-D charge, exclusion fit ON vs OFF.

The dQ/dx system minimises |data - R x|^2 over the measured 2-D charge, and BOTH
implementations already persist the per-cell answer:

  tree T_proj_data   cluster_id / channel / time_slice / charge / charge_err / charge_pred
     toolkit  m_fitted_charge_2d  (TrackFitting.cxx:7386-7392 -> root/src/*MagnifyTracking*.cxx)
     prototype proj_data_{u,v,w}_map (PR3DCluster_multi_dQ_dx_fit.h:842-863
                                      -> pid/apps/wire-cell-prod-nue-port.cxx:3249-3320)

so this script needs no new instrumentation -- it reads the standard outputs of
the arms that already exist.

Metric, per (event, region, anchor, arm), over the cells of the 2-D shadow of a
sphere of radius --box-cm around the anchor:

  N      cells with charge > 0, not on a dead channel (T_bad_ch)
  U      sum|y-yhat| / sum y      <- the headline: unexplained charge fraction
  B      (sum yhat - sum y)/sum y <- signed bias
  chi2   sum((y-yhat)/sigma)^2 with ONE canonical sigma per cell (--sigma-from arm),
         sigma = sqrt(charge_err^2 + (charge*rel_uncer)^2 + add_uncer^2), the fit's
         own whitening (rel 0.075 ind / 0.05 col, add 0 / 300 -- identical in
         qlport/uboone_track_fitting.json and clus TrackFittingPresets.h:31-34)
  uncov  sum y over cells with yhat == 0, / sum y
  Ncol   fit points within the sphere -- the dof asymmetry, reported, never divided out

U and B are parameter-count-insensitive; chi2/N is not (exclusion changes the number
of trajectory points near the vertex, and the regulariser means ndf != N - Ncol), so
chi2 is printed BESIDE N and Ncol rather than as chi2/ndf.

Cells owned by more than one cluster are reported separately and excluded from the
primary numbers: on the toolkit side T_proj_data comes from the MERGED
m_fitted_charge_2d, whose pred_charge is last-writer-wins over a pointer-keyed map
(PrDisplayDump.cxx:1130-1150), while the prototype's map is genuinely per-cluster.

Frames (verified on uBooNE 5384 6505, both sides): T_rec_charge pu/pv/pw ARE global
channel numbers and pt IS the T_proj_data time_slice, scale 1; the prototype carries
the display +0.5 on all four (removed here).  T_bad_ch start/end are TICKS
(--ticks-per-slice, default 4).

Usage:
  pr109_2d_resid.py --arm ON=path.root:wct --arm OFF=path2.root:wct \
                    [--anchors-from OFF] [--box-cm 3.0] [--tsv out.tsv] [--tag evt]
"""
import argparse, sys, os
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr108_fit_point_compare import junctions  # the pr/108 junction definition, reused

REL = {'U': 0.075, 'V': 0.075, 'W': 0.05}
ADD = {'U': 0.0, 'V': 0.0, 'W': 300.0}


def load_arm(path, kind, ticks_per_slice):
    g = uproot.open(path)
    keys = {k.split(';')[0] for k in g.keys()}

    # ---- points -------------------------------------------------------------
    br = ['x', 'y', 'z', 'q', 'pu', 'pv', 'pw', 'pt', 'flag_vertex', 'cluster_id', 'sub_cluster_id']
    a = g['T_rec_charge'].arrays(br, library='np')
    off = 0.5 if kind == 'wcp' else 0.0
    pts = dict(X=np.stack([a['x'], a['y'], a['z']], 1).astype(float),
               P=np.stack([a['pu'] - off, a['pv'] - off, a['pw'] - off, a['pt'] - off], 1).astype(float),
               cl=a['cluster_id'].astype(int), sc=a['sub_cluster_id'].astype(int),
               fv=a['flag_vertex'].astype(bool), q=a['q'].astype(float))
    vals, cnt = np.unique(pts['cl'], return_counts=True)
    pts['main'] = int(vals[cnt.argmax()])

    # ---- cells --------------------------------------------------------------
    p = g['T_proj_data'].arrays(library='np')
    cids = [int(c) for c in p['cluster_id'][0]]
    cells = {}          # (ch, ts) -> dict(q, qe, preds=[(cid, pred)])
    for i, c in enumerate(cids):
        ch = np.asarray(list(p['channel'][0][i]), dtype=np.int64)
        ts = np.asarray(list(p['time_slice'][0][i]), dtype=np.int64)
        q = np.asarray(list(p['charge'][0][i]), dtype=float)
        qe = np.asarray(list(p['charge_err'][0][i]), dtype=float)
        qp = np.asarray(list(p['charge_pred'][0][i]), dtype=float)
        for k in range(len(ch)):
            key = (int(ch[k]), int(ts[k]))
            e = cells.get(key)
            if e is None:
                cells[key] = dict(q=q[k], qe=qe[k], preds=[(c, qp[k])])
            else:
                e['preds'].append((c, qp[k]))

    # ---- plane of a channel, learned from the point projections -------------
    ch2pl = {}
    for j, pl in enumerate('UVW'):
        for v in np.unique(np.rint(pts['P'][:, j]).astype(int)):
            ch2pl[int(v)] = pl
    known = np.array(sorted(ch2pl)) if ch2pl else np.array([0])

    def plane_of(ch):
        pl = ch2pl.get(ch)
        if pl is None:                       # nearest known channel
            pl = ch2pl[int(known[np.argmin(np.abs(known - ch))])]
        return pl

    # ---- dead channels ------------------------------------------------------
    dead = {}
    if 'T_bad_ch' in keys:
        b = g['T_bad_ch'].arrays(['chid', 'start_time', 'end_time'], library='np')
        for c, s, e in zip(b['chid'], b['start_time'], b['end_time']):
            dead.setdefault(int(c), []).append((float(s) / ticks_per_slice, float(e) / ticks_per_slice))

    def is_dead(ch, ts):
        for s, e in dead.get(ch, ()):
            if s - 0.5 <= ts <= e + 0.5:
                return True
        return False

    # ---- neutrino vertex (space-charge corrected; both sides carry it) ------
    nuv = None
    if 'T_kine' in keys:
        k = g['T_kine'].arrays(['kine_nu_x_corr', 'kine_nu_y_corr', 'kine_nu_z_corr'], library='np')
        if len(k['kine_nu_x_corr']):
            nuv = np.array([k['kine_nu_x_corr'][0], k['kine_nu_y_corr'][0], k['kine_nu_z_corr'][0]], float)

    return dict(path=path, kind=kind, pts=pts, cells=cells, plane_of=plane_of,
                is_dead=is_dead, nuv=nuv)


def affine(arm, anchor, radius=30.0):
    """Local affine map (x,y,z) -> (pu,pv,pw,pt) fitted on this arm's own points.

    Fitted within `radius` of the anchor so per-APA / per-t0 differences cannot
    leak in.  Returns (predict(P3) -> 4-vector, max residual in wires/slices,
    gradient norm per output = units per cm).
    """
    X, P = arm['pts']['X'], arm['pts']['P']
    d = np.linalg.norm(X - anchor, axis=1)
    m = d < radius
    if m.sum() < 8:
        m = d < 3 * radius
    if m.sum() < 8:
        return None, None, None
    A = np.column_stack([np.ones(m.sum()), X[m]])
    coef, *_ = np.linalg.lstsq(A, P[m], rcond=None)      # (4 x 4): rows 1..3 = gradient
    res = np.abs(A @ coef - P[m]).max(axis=0)
    grad = np.linalg.norm(coef[1:], axis=0)              # units per cm, per output
    return (lambda p: np.concatenate([[1.0], p]) @ coef), res, grad


def metrics(arm, anchor, box_cm, sigma_lookup, main_only=True, keep=None):
    """Per-plane metrics inside the 2-D shadow of a sphere of radius box_cm."""
    pred, res, grad = affine(arm, anchor, 30.0)
    if pred is None:
        return None
    c = pred(anchor)
    half = grad * box_cm
    out = {}
    for j, pl in enumerate('UVW'):
        out[pl] = dict(N=0, sy=0.0, sp=0.0, sabs=0.0, chi2=0.0, uncov=0.0,
                       nshared=0, ndead=0, sshared=0.0)
    for (ch, ts), e in arm['cells'].items():
        pl = arm['plane_of'](ch)
        j = 'UVW'.index(pl)
        if abs(ch - c[j]) > half[j] or abs(ts - c[3]) > half[3]:
            continue
        o = out[pl]
        if keep is not None and (ch, ts) not in keep:
            continue
        if e['q'] <= 0:
            continue
        if arm['is_dead'](ch, ts):
            o['ndead'] += 1
            continue
        shared = len({cid for cid, _ in e['preds']}) > 1
        yhat = e['preds'][0][1]
        if shared:
            o['nshared'] += 1
            o['sshared'] += e['q']
            if main_only:
                continue
        y = e['q']
        o['N'] += 1
        o['sy'] += y
        o['sp'] += yhat
        o['sabs'] += abs(y - yhat)
        if yhat == 0:
            o['uncov'] += y
        s = sigma_lookup(ch, ts, pl, y, e['qe'])
        o['chi2'] += ((y - yhat) / s) ** 2
    d = np.linalg.norm(arm['pts']['X'] - anchor, axis=1)
    ncol = int((d < box_cm).sum())
    for pl in 'UVW':
        o = out[pl]
        o['U'] = o['sabs'] / o['sy'] if o['sy'] > 0 else float('nan')
        o['B'] = (o['sp'] - o['sy']) / o['sy'] if o['sy'] > 0 else float('nan')
    tot = dict(N=sum(out[p]['N'] for p in 'UVW'), sy=sum(out[p]['sy'] for p in 'UVW'),
               sp=sum(out[p]['sp'] for p in 'UVW'), sabs=sum(out[p]['sabs'] for p in 'UVW'),
               chi2=sum(out[p]['chi2'] for p in 'UVW'), uncov=sum(out[p]['uncov'] for p in 'UVW'),
               nshared=sum(out[p]['nshared'] for p in 'UVW'), ndead=sum(out[p]['ndead'] for p in 'UVW'))
    tot['U'] = tot['sabs'] / tot['sy'] if tot['sy'] > 0 else float('nan')
    tot['B'] = (tot['sp'] - tot['sy']) / tot['sy'] if tot['sy'] > 0 else float('nan')
    out['ALL'] = tot
    out['_ncol'] = ncol
    out['_res'] = res
    out['_center'] = c
    return out


def sigma_from(arm):
    """Canonical sigma: this arm's charge_err put through the fit's whitening."""
    def look(ch, ts, pl, y, qe_fallback):
        e = arm['cells'].get((ch, ts))
        qe = e['qe'] if e is not None else qe_fallback
        q = e['q'] if e is not None else y
        return max(np.sqrt(qe ** 2 + (q * REL[pl]) ** 2 + ADD[pl] ** 2), 1e-6)
    return look


def regions(arm, max_j):
    """(name, xyz) list: the main vertex plus every junction of the main cluster."""
    d = dict(X=arm['pts']['X'], fv=arm['pts']['fv'], cl=arm['pts']['cl'], sc=arm['pts']['sc'])
    out = []
    if arm['nuv'] is not None:
        m = arm['pts']['fv'] & (arm['pts']['cl'] == arm['pts']['main'])
        if m.any():
            i = np.where(m)[0][np.argmin(np.linalg.norm(arm['pts']['X'][m] - arm['nuv'], axis=1))]
            out.append(('mainvtx', arm['pts']['X'][i]))
    J = junctions(d)
    for k, j in enumerate(J[:max_j]):
        out.append(('J%d' % k, j))
    return out



def control_same_data(a, b, la, lb):
    """Control 1: the measured charge must be identical between two arms of the
    same implementation (same imaging input).  Reports over ALL cells."""
    ka, kb = set(a['cells']), set(b['cells'])
    common = ka & kb
    dmax = 0.0; nd = 0
    for k in common:
        d = abs(a['cells'][k]['q'] - b['cells'][k]['q'])
        if d > 0: nd += 1
        dmax = max(dmax, d)
    print('# control1 %s vs %s: cells %d/%d common %d, max|dy|=%.6g on %d cell(s), '
          'only-%s %d, only-%s %d' % (la, lb, len(ka), len(kb), len(common), dmax, nd,
                                      la, len(ka - kb), lb, len(kb - ka)))
    return dmax

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True, help='LABEL=path.root[:wct|wcp]')
    ap.add_argument('--box-cm', type=float, default=3.0)
    ap.add_argument('--ticks-per-slice', type=int, default=4)
    ap.add_argument('--sigma-from', default=None, help='arm label whose charge_err defines sigma')
    ap.add_argument('--anchors-from', action='append', default=None, help='arm label(s) supplying anchors')
    ap.add_argument('--max-junctions', type=int, default=4)
    ap.add_argument('--common-pred', action='store_true',
                    help='restrict to cells predicted > 0 by EVERY arm (separates coverage from shape)')
    ap.add_argument('--shared', action='store_true', help='include cells owned by >1 cluster')
    ap.add_argument('--tag', default='')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        lab, _, rest = spec.partition('=')
        path, _, kind = rest.partition(':')
        arms[lab] = load_arm(path, kind or 'wct', args.ticks_per_slice)
    labs = list(arms)
    sig = sigma_from(arms[args.sigma_from or labs[-1]])
    anchor_labs = args.anchors_from or labs

    keep = None
    if args.common_pred:
        sets = []
        for lab in labs:
            sets.append({k for k, e in arms[lab]['cells'].items()
                         if e['preds'][0][1] > 0 and len({c for c, _ in e['preds']}) == 1})
        keep = set.intersection(*sets)
        print('# common-pred cells (pred>0 and single-owner in every arm): %d' % len(keep))

    rows = []
    if len(labs) >= 2:
        control_same_data(arms[labs[0]], arms[labs[1]], labs[0], labs[1])
    print('# tag=%s box=%.1fcm sigma-from=%s shared=%s' %
          (args.tag, args.box_cm, args.sigma_from or labs[-1], args.shared))
    for al in anchor_labs:
        for rname, A in regions(arms[al], args.max_junctions):
            print('== anchor %s %s  xyz=(%.2f %.2f %.2f)' % (al, rname, A[0], A[1], A[2]))
            print('   %-10s %6s %6s %10s %8s %8s %9s %7s %6s %6s' %
                  ('arm', 'N', 'Ncol', 'sum_y', 'U', 'B', 'chi2/N', 'uncov', 'shar', 'dead'))
            for lab in labs:
                m = metrics(arms[lab], A, args.box_cm, sig, main_only=not args.shared, keep=keep)
                if m is None:
                    print('   %-10s (no local points)' % lab); continue
                t = m['ALL']
                print('   %-10s %6d %6d %10.3g %8.4f %8.4f %9.3g %7.4f %6d %6d' %
                      (lab, t['N'], m['_ncol'], t['sy'], t['U'], t['B'],
                       t['chi2'] / max(t['N'], 1), t['uncov'] / max(t['sy'], 1e-9),
                       t['nshared'], t['ndead']))
                for pl in 'UVW':
                    o = m[pl]
                    print('      %-7s %6d %6s %10.3g %8.4f %8.4f %9.3g' %
                          (pl, o['N'], '', o['sy'], o['U'], o['B'], o['chi2'] / max(o['N'], 1)))
                    rows.append([args.tag, al, rname, '%.2f' % A[0], '%.2f' % A[1], '%.2f' % A[2],
                                 args.box_cm, lab, pl, o['N'], o['sy'], o['U'], o['B'],
                                 o['chi2'] / max(o['N'], 1), m['_ncol'], o['nshared'], o['ndead']])
                rows.append([args.tag, al, rname, '%.2f' % A[0], '%.2f' % A[1], '%.2f' % A[2],
                             args.box_cm, lab, 'ALL', t['N'], t['sy'], t['U'], t['B'],
                             t['chi2'] / max(t['N'], 1), m['_ncol'], t['nshared'], t['ndead']])
                if lab == labs[0]:
                    print('      affine residual (u,v,w,t) = %s' % np.round(m['_res'], 4).tolist())
    if args.tsv:
        new = not os.path.exists(args.tsv)
        with open(args.tsv, 'a') as fh:
            if new:
                fh.write('tag\tanchor_arm\tregion\tax\tay\taz\tbox\tarm\tplane\tN\tsum_y\tU\tB\tchi2_per_N\tncol\tnshared\tndead\n')
            for r in rows:
                fh.write('\t'.join(str(x) for x in r) + '\n')


if __name__ == '__main__':
    main()

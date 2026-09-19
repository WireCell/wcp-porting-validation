#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 3.3 -- the pr/149 sec 13.8 bow check: does the 3-D image follow the fitted drift-x bow?

On long isochronous (ISO) muons the pr/149 zig-zag metric is dominated by a ~1.2 cm smooth drift-x excursion of
the fitted segment about its chord (bow_x), which no sampler or fit key moved.  Two readings were left open:
physical (space charge / scattering: the image itself bows) or a fit artefact (the fit bows toward pinned segment
ends while the image is straight).  This measures both on the same rows, in the drift-time coordinate:
  per main-cluster (bundle) segment with length >= 10 cm, >= 10 fit rows, chord angle to drift >= 75 deg
  (the pr/149 zzi selection), each calib fit point is matched to its T_rec_charge row (nearest, < 0.2 cm) to get
  its W-plane projection (pw, global channel rank) and time slice (pt).  The IMAGE time at that wire is the
  charge-weighted mean time_slice of the bundle's measured W cells (T_proj_data, union of the bundle's stage
  clusters) on the rounded wire, restricted to |slice - pt| <= 8 (other tracks on the same wire are excluded).
    fit_bow   = rms of (pt - chord(pt)) after a 9-point running mean          [slices; the fit's own bow]
    res_bow   = rms of (pt - t_img)     after a 9-point running mean          [slices; what the image does NOT do]
    img_bow   = rms of (t_img - chord(t_img)) after a 9-point running mean    [slices; the image's own bow]
  1 slice = 4 ticks = 2 us ~ 0.32 cm of drift (SBND drift speed 0.16 cm/us), so 1.2 cm ~ 3.7 slices.
  If the image follows the bow: img_bow ~ fit_bow and res_bow << fit_bow  (physical).
  If the image is straight:     img_bow << fit_bow and res_bow ~ fit_bow   (fit artefact).
Length-weighted per event; reported per arm and paired arm - base.
Usage: pr150_bow.py --a pr150ts0 --b pr150tcsp3bw --samples mcp1k mcp2k [--events <manifest dir>] [--tsv out]
"""
import argparse, glob, json, math, os
import numpy as np
import uproot
from scipy.spatial import cKDTree

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
WBASE = 7936; K = 9; DT = 8


def smooth_rms(v, k=K):
    v = np.asarray(v, float); m = np.isfinite(v)
    if m.sum() < k:
        return float('nan')
    sm = np.convolve(v[m], np.ones(k) / k, mode='valid')
    return float(np.sqrt(np.mean(sm ** 2)))


def chord_resid(t, P):
    """residual of t about the straight line through the segment's ends, parameterised by arc length."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if s[-1] <= 0:
        return np.full(len(t), np.nan)
    line = t[0] + (t[-1] - t[0]) * s / s[-1]
    return t - line


def one(d, ev):
    calib = f'{d}/calib-pr-evt{ev}.json'; root = f'{d}/tracking-pr.root'
    if not (os.path.exists(calib) and os.path.exists(root)):
        return None
    cal = json.load(open(calib)); mv = cal.get('main_vertex') or {}; cid = mv.get('cluster_id')
    if cid is None:
        return None
    f = uproot.open(root); t = f['T_rec_charge'].arrays(library='np'); m = t['cluster_id'] == cid
    if m.sum() < 10:
        return None
    F = np.c_[t['x'][m], t['y'][m], t['z'][m]]; PW = t['pw'][m]; PT = t['pt'][m]; RC = t['real_cluster_id'][m]
    U = {int(cid)} | {int(r) // 1000 for r in RC.tolist() if r > 0}
    tree = cKDTree(F)
    pd = f['T_proj_data'].arrays(library='np')
    cells = {}
    if len(pd['cluster_id']):
        for c, ch, ts, q in zip(pd['cluster_id'][0], pd['channel'][0], pd['time_slice'][0], pd['charge'][0]):
            if int(c) not in U:
                continue
            ch = np.asarray(ch).astype(int); ts = np.asarray(ts).astype(int); q = np.asarray(q, float)
            for w, s, qq in zip(ch.tolist(), ts.tolist(), q.tolist()):
                if w >= WBASE and qq > 0:
                    cells.setdefault(w, []).append((s, qq))
    out = []
    for seg in cal.get('segments', []):
        if seg.get('cluster_id') != cid:
            continue
        P = np.array([[p['x'], p['y'], p['z']] for p in seg.get('points', [])], float)
        if len(P) < 10:
            continue
        c = P[-1] - P[0]; chord = np.linalg.norm(c)
        if chord <= 0:
            continue
        theta = math.degrees(math.acos(min(1.0, abs(c[0] / chord))))
        length = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        if theta < 75 or length < 10:
            continue
        dd, j = tree.query(P)
        ok = dd < 0.2
        if ok.sum() < 10:
            continue
        pt = PT[j]; pw = PW[j]
        timg = np.full(len(P), np.nan)
        for i in np.where(ok)[0]:
            cl = cells.get(int(round(pw[i])), [])
            cl = [(s, q) for s, q in cl if abs(s - pt[i]) <= DT]
            if cl:
                timg[i] = sum(s * q for s, q in cl) / sum(q for s, q in cl)
        good = ok & np.isfinite(timg)
        if good.sum() < 10:
            continue
        Pg = P[good]; ptg = pt[good]; tig = timg[good]
        out.append(dict(length=length, theta=theta, n=int(good.sum()), cover=float(good.sum() / len(P)),
                        fit_bow=smooth_rms(chord_resid(ptg, Pg)), img_bow=smooth_rms(chord_resid(tig, Pg)),
                        res_bow=smooth_rms(ptg - tig), res_rms=float(np.sqrt(np.mean((ptg - tig) ** 2))),
                        res_med=float(np.median(ptg - tig))))
    if not out:
        return None
    w = np.array([o['length'] for o in out])
    ev_out = {k: float(np.sum(w * np.array([o[k] for o in out])) / w.sum()) for k in ('fit_bow', 'img_bow', 'res_bow', 'res_rms', 'res_med', 'cover')}
    ev_out['nseg'] = len(out); ev_out['len'] = float(w.sum())
    return ev_out


def load(arm, samples, manifest):
    R = {}
    for s in samples:
        evs = None
        if manifest and os.path.exists(f'{manifest}/{s}.txt'):
            evs = {int(l.split()[0]) for l in open(f'{manifest}/{s}.txt') if l.strip() and not l.startswith('#')}
        for d in sorted(glob.glob(f'{SX}/work-{s}-{arm}/pr_evt*')):
            ev = int(os.path.basename(d)[6:])
            if evs is not None and ev not in evs:
                continue
            r = one(d, ev)
            if r:
                R[(s, ev)] = r
    return R


def q(x, p):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(np.percentile(x, p)) if len(x) else float('nan')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--events'); ap.add_argument('--tsv')
    a = ap.parse_args()
    A, B = load(a.a, a.samples, a.events), load(a.b, a.samples, a.events)
    common = sorted(set(A) & set(B))
    print(f'# pr150 bow check A={a.a} B={a.b}: events with >= 1 ISO segment A {len(A)} B {len(B)} common {len(common)}; units = time slices (1 slice ~ 0.32 cm drift)')
    print('| arm | events | ISO length (m) | fit_bow p50 / p90 | img_bow p50 / p90 | res_bow p50 / p90 | res_rms p50 | res_med p50 | image cover p50 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for lab, R in ((a.a, A), (a.b, B)):
        g = lambda k, p: q([r[k] for r in R.values()], p)
        print(f"| {lab} | {len(R)} | {sum(r['len'] for r in R.values()) / 100:.1f} | {g('fit_bow', 50):.2f} / {g('fit_bow', 90):.2f} | {g('img_bow', 50):.2f} / {g('img_bow', 90):.2f} | "
              f"{g('res_bow', 50):.2f} / {g('res_bow', 90):.2f} | {g('res_rms', 50):.2f} | {g('res_med', 50):+.2f} | {g('cover', 50):.2f} |")
    for k in ('fit_bow', 'img_bow', 'res_bow'):
        d = np.array([B[c][k] - A[c][k] for c in common]); d = d[np.isfinite(d)]
        print(f'paired {k} (B - A) on {len(d)}: median {np.median(d):+.3f}, lower {int((d < -0.05).sum())} / higher {int((d > 0.05).sum())}')
    r = np.array([[A[c]['fit_bow'], A[c]['img_bow'], A[c]['res_bow']] for c in common]); r = r[np.isfinite(r).all(1)]
    if len(r):
        print(f'on {a.a}: median img_bow / fit_bow = {np.median(r[:, 1] / r[:, 0]):.2f}; median res_bow / fit_bow = {np.median(r[:, 2] / r[:, 0]):.2f}; '
              f'events where the image follows the fit (res_bow < 0.5 fit_bow): {int((r[:, 2] < 0.5 * r[:, 0]).sum())} of {len(r)}; image straight (img_bow < 0.5 fit_bow): {int((r[:, 1] < 0.5 * r[:, 0]).sum())}')
    if a.tsv:
        with open(a.tsv, 'w') as fh:
            fh.write('sample\tevent\tarm\tnseg\tlen_cm\tfit_bow\timg_bow\tres_bow\tres_rms\tres_med\tcover\n')
            for lab, R in ((a.a, A), (a.b, B)):
                for (s, e), r in sorted(R.items()):
                    fh.write(f"{s}\t{e}\t{lab}\t{r['nseg']}\t{r['len']:.1f}\t{r['fit_bow']:.3f}\t{r['img_bow']:.3f}\t{r['res_bow']:.3f}\t{r['res_rms']:.3f}\t{r['res_med']:.3f}\t{r['cover']:.3f}\n")


if __name__ == '__main__':
    main()

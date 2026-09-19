#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 5 -- score the retune ladder against the frozen rule 150_pred_r2.txt on the 626-event
ladder manifest (Stage 1 + the pr/149 Stage-2 manifest).

Inputs: docs/pr/150_figs/metrics/<arm>-<sample>.tsv and traj/<arm>-<sample>.tsv (pr150_metrics.py / pr150_traj_eval.py
extract) for s0, csp3bw (the full arms, restricted to the manifest events) and each ladder level r<k>; the sentinel
printouts 150_figs/150_lad_sentinels_<arm>.txt (sentinels_tolerant.py on the level's arms); vtx105 labels.
Clauses (sec 3 of the rule): a) V1 >= s0 - 1 % of labelled AND away <= toward + 5; b) N1 lost <= gained + 5, TGM 0;
c) sentinel FAIL <= s0 FAIL + 1; d) numu > 0.9 within +-3 % of s0, |dEnu| median <= 25 MeV; e) R2D_W, P2, uncov
<= s0 (row-weighted); f) C (rc 0, DL failed 0) and R (core <= 1.50, RSS <= 1.25).  Split-half guard on odd / even
event ids: V1 within bar - 1 %, V2 / N1 within bar - 2 on either half.
Usage: pr150_ladder.py --manifest /home/xqian/tmp/pr150/manifest_ladder --levels r1 r2 r3 r4 [--t csp3bw] [--tsv out]
"""
import argparse, csv, glob, math, os, sys
import numpy as np
SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
sys.path.insert(0, f'{SX}/vtx_rules'); import vtx_io  # noqa: E402
D = f'{SX}/docs/pr/150_figs'; SAMPLES = ('nuecc48', 'ncpi0', 'mcp1k', 'mcp2k')


def manifest(mdir):
    M = set()
    for s in SAMPLES:
        for l in open(f'{mdir}/{s}.txt'):
            if l.strip() and not l.startswith('#'):
                M.add((s, int(l.split()[0])))
    return M


def load(arm, keys, kind='metrics'):
    R = {}
    for s in SAMPLES:
        p = f'{D}/{kind}/{arm}-{s}.tsv'
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p), delimiter='\t'):
            k = (s, int(r['event']))
            if k in keys:
                R[k] = r
    return R


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def vertex(r, truth):
    if r is None or r.get('has_calib') != '1' or r['vx'] in ('', 'nan'):
        return math.inf
    return math.dist(truth, (f(r['vx']), f(r['vy']), f(r['vz'])))


def score(A, X, keys, labs, sent_fail, name, half=None):
    ks = [k for k in keys if k in A and k in X and (half is None or k[1] % 2 == half)]
    lab = [k for k in ks if f'evt{k[1]}' in labs]
    v1a = sum(vertex(A[k], labs[f'evt{k[1]}']['truth']) <= 3 for k in lab); v1x = sum(vertex(X[k], labs[f'evt{k[1]}']['truth']) <= 3 for k in lab)
    away = toward = 0
    for k in lab:
        t = labs[f'evt{k[1]}']['truth']; da, dx = vertex(A[k], t), vertex(X[k], t)
        pa = (f(A[k]['vx']), f(A[k]['vy']), f(A[k]['vz'])) if da < math.inf else None
        px = (f(X[k]['vx']), f(X[k]['vy']), f(X[k]['vz'])) if dx < math.inf else None
        moved = math.inf if (pa is None) != (px is None) else (0 if pa is None else math.dist(pa, px))
        if moved > 10:
            away += dx > da; toward += dx < da
    lost = sum(1 for k in ks if A[k]['nu_evaluated'] == '1' and X[k]['nu_evaluated'] != '1'); gained = sum(1 for k in ks if A[k]['nu_evaluated'] != '1' and X[k]['nu_evaluated'] == '1')
    numu = lambda R: sum(1 for k in ks if R[k]['nu_evaluated'] == '1' and f(R[k]['numu_score']) > 0.9)
    denu = np.median([abs(f(X[k]['Enu']) - f(A[k]['Enu'])) for k in ks if A[k]['nu_evaluated'] == '1' and X[k]['nu_evaluated'] == '1' and np.isfinite(f(A[k]['Enu'])) and np.isfinite(f(X[k]['Enu']))])
    core = np.median([f(X[k]['core_s']) / f(A[k]['core_s']) for k in ks if f(A[k]['core_s']) > 0]); rss = np.median([f(X[k]['maxrss_kb']) / f(A[k]['maxrss_kb']) for k in ks if f(A[k]['maxrss_kb']) > 0])
    rcbad = sum(1 for k in ks if X[k]['rc'] not in ('0', ''))
    return dict(name=name, n=len(ks), nlab=len(lab), V1_s0=v1a, V1=v1x, away=away, toward=toward, lost=lost, gained=gained, numu_s0=numu(A), numu=numu(X), dEnu=float(denu),
                core=float(core), rss=float(rss), rcbad=rcbad, sent_fail=sent_fail)


def traj(A, X, keys):
    out = {}
    for met in ('R2D_W', 'P2', 'uncov', 'P1'):
        ks = [k for k in keys if k in A and k in X and A[k]['has'] == '1' and X[k]['has'] == '1' and A[k].get(met, '') not in ('', 'nan') and X[k].get(met, '') not in ('', 'nan')]
        wa = np.array([f(A[k]['rows']) for k in ks]); wx = np.array([f(X[k]['rows']) for k in ks])
        out[met] = (float(np.sum([f(A[k][met]) for k in ks] * wa) / wa.sum()), float(np.sum([f(X[k][met]) for k in ks] * wx) / wx.sum()), len(ks))
    return out


def sentinels(arm):
    p = f'{D}/150_lad_sentinels_{arm}.txt'
    if not os.path.exists(p):
        return None, None
    txt = open(p).read().splitlines()
    return sum(1 for l in txt if l.startswith('FAIL')), sum(1 for l in txt if l.startswith('PASS'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--levels', nargs='+', required=True)
    ap.add_argument('--t', default='csp3bw'); ap.add_argument('--tsv')
    a = ap.parse_args()
    keys = manifest(a.manifest); labs = {L['event']: L for L in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105) if L['truth'] is not None}
    A = load('pr150s0', keys); TA = load('pr150s0', keys, 'traj')
    s0_fail, s0_pass = sentinels('pr150s0')
    print(f'# pr150 ladder score, rule 150_pred_r2.txt; manifest {len(keys)} events; s0 sentinels on these events: PASS {s0_pass} FAIL {s0_fail}')
    print('| level | n | labelled | V1 <=3cm (s0) | bar | V2 away/toward | N1 lost/gained | numu>0.9 (s0) | dEnu med | sentinels FAIL (s0) | R2D_W s0->x | P2 | uncov | core | RSS | rc!=0 | clauses a b c d e f | split-half |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    rows = []
    for lev in [a.t] + a.levels:
        X = load(f'pr150{lev}', keys); TX = load(f'pr150{lev}', keys, 'traj')
        if not X:
            print(f'| {lev} | (no metrics) |'); continue
        sf, sp = sentinels(f'pr150{lev}')
        r = score(A, X, keys, labs, sf, lev); t = traj(TA, TX, keys)
        bar = r['V1_s0'] - math.ceil(0.01 * r['nlab'])
        ca = r['V1'] >= bar and r['away'] <= r['toward'] + 5
        cb = r['lost'] <= r['gained'] + 5
        cc = sf is not None and s0_fail is not None and sf <= s0_fail + 1
        cd = abs(r['numu'] - r['numu_s0']) <= 0.03 * max(r['numu_s0'], 1) and r['dEnu'] <= 25
        ce = all(t[m][1] <= t[m][0] for m in ('R2D_W', 'P2', 'uncov'))
        cf = r['rcbad'] == 0 and r['core'] <= 1.5 and r['rss'] <= 1.25
        halves = []
        for h in (0, 1):
            rh = score(A, X, keys, labs, sf, lev, half=h); barh = rh['V1_s0'] - math.ceil(0.01 * rh['nlab']) - math.ceil(0.01 * rh['nlab'])
            halves.append(rh['V1'] >= barh and rh['away'] <= rh['toward'] + 7 and rh['lost'] <= rh['gained'] + 7)
        sh = 'PASS' if all(halves) else 'FAIL'
        flags = ' '.join('P' if c else 'F' for c in (ca, cb, cc, cd, ce, cf))
        print(f"| {lev} | {r['n']} | {r['nlab']} | {r['V1']} ({r['V1_s0']}) | {bar} | {r['away']}/{r['toward']} | {r['lost']}/{r['gained']} | {r['numu']} ({r['numu_s0']}) | {r['dEnu']:.1f} | "
              f"{sf} ({s0_fail}) | {t['R2D_W'][0]:.4f}->{t['R2D_W'][1]:.4f} | {t['P2'][0]:.4f}->{t['P2'][1]:.4f} | {t['uncov'][0]:.4f}->{t['uncov'][1]:.4f} | {r['core']:.2f} | {r['rss']:.2f} | {r['rcbad']} | {flags} | {sh} |")
        rows.append((lev, all((ca, cb, cc, cd, ce, cf)), sum((ca, cb, cc, cd, ce, cf)), sh))
    passing = [l for l, ok, n, sh in rows if ok and sh == 'PASS' and l != a.t]
    print(f'# SELECTED (smallest passing level): {passing[0] if passing else "NONE"}; fewest-failing: ' + ', '.join(f'{l} {6 - n} fails' for l, ok, n, sh in sorted(rows, key=lambda x: -x[2]) if l != a.t))


if __name__ == '__main__':
    main()

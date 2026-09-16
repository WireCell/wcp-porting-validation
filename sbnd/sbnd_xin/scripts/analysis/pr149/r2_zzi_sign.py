#!/usr/bin/env python3
"""doc pr/149 round 2: the pre-registered PRIMARY endpoint (149_pred_r2.txt sec 3) and its
attribution table.

For a comparison B vs A over a stratum, paired over events with a calib dump in both arms and a
finite ISO-segment zig-zag (zzi_*: main-cluster segments >= 10 cm, >= 10 points, theta_drift >= 75
deg, pr149_metrics.py):
    median delta zzi_rms_dr (cm -- the calib dump length unit), median delta zzi_ratio (path/chord)
    improved = delta zzi_rms_dr < -0.01 (cm), worsened = delta > +0.01
    exact two-sided sign test on improved vs worsened
IMPROVES iff median d_rms < 0 AND median d_ratio < 0 AND improved > worsened AND p < 0.05.

Strata:
    stage 2  iso (ISO-strong, the long-ISO muon stratum; PRIMARY), vtx_iso, control
             -- 149_stage2_selection.tsv
    stage 1  ISO = d102mpr iso_len_any75 >= 10 cm with a calib dump (nuecc48 + ncpi0; second reading)

Usage: r2_zzi_sign.py --stage {1,2} --pairs B:A [B:A ...]    (arm names without the metrics/ prefix,
       e.g. pr149s2r2rscs:pr149s2r2s0)
"""
import argparse
import csv
import math
import os

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
M = os.path.join(SX, 'docs/pr/149_figs/metrics')
fnum = lambda x: float(x) if x not in ('', 'nan', None) else float('nan')


def load(arm, samples):
    d = {}
    for s in samples:
        p = os.path.join(M, f'{arm}-{s}.tsv')
        if not os.path.exists(p):
            raise SystemExit(f'missing metrics table {p}')
        for r in csv.DictReader(open(p), delimiter='\t'):
            d[(s, int(r['event']))] = r
    return d


def sign_test_p(k, n):
    """Exact two-sided binomial sign test, H0 p=1/2."""
    if n == 0:
        return float('nan')
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def median(v):
    v = sorted(x for x in v if math.isfinite(x))
    if not v:
        return float('nan')
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def compare(A, B, keys):
    both = [k for k in keys if A.get(k, {}).get('has_calib') == '1' and B.get(k, {}).get('has_calib') == '1'
            and math.isfinite(fnum(A[k]['zzi_rms_dr'])) and math.isfinite(fnum(B[k]['zzi_rms_dr']))]
    d_rms = [fnum(B[k]['zzi_rms_dr']) - fnum(A[k]['zzi_rms_dr']) for k in both]
    d_ratio = [fnum(B[k]['zzi_ratio']) - fnum(A[k]['zzi_ratio']) for k in both]
    d_zz = [fnum(B[k]['zz_rms_dr']) - fnum(A[k]['zz_rms_dr']) for k in both]
    d_unc = [fnum(B[k]['uncov']) - fnum(A[k]['uncov']) for k in both]
    imp = sum(1 for x in d_rms if x < -0.01)
    wor = sum(1 for x in d_rms if x > 0.01)
    p = sign_test_p(imp, imp + wor)
    m_rms, m_ratio = median(d_rms), median(d_ratio)
    verdict = (m_rms < 0 and m_ratio < 0 and imp > wor and p < 0.05)
    return dict(n=len(keys), paired=len(both), m_rms=m_rms, m_ratio=m_ratio, imp=imp, wor=wor, p=p,
                m_zz=median(d_zz), m_unc=median(d_unc),
                base_rms=median([fnum(A[k]['zzi_rms_dr']) for k in both]),
                verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=[1, 2], required=True)
    ap.add_argument('--pairs', nargs='+', required=True)
    a = ap.parse_args()
    if a.stage == 2:
        samples = ['mcp1k', 'mcp2k']
        st = {}
        for r in csv.DictReader(open(os.path.join(SX, 'docs/pr/149_figs/149_stage2_selection.tsv')), delimiter='\t'):
            st.setdefault(r['stratum'], []).append((r['sample'], int(r['event'])))
        strata = [('iso (PRIMARY)', st['iso']), ('vtx_iso', st['vtx_iso']), ('control', st['control'])]
    else:
        samples = ['nuecc48', 'ncpi0']
        R = load('d102mpr', samples)
        strata = [('ISO (stage-1 reading)', sorted(k for k in R if R[k]['has_calib'] == '1'
                                                   and fnum(R[k]['iso_len_any75']) >= 10))]
    print(f'# doc pr/149 round 2 primary endpoint (149_pred_r2.txt sec 3), stage {a.stage}')
    print('# d = B - A, paired; zzi_rms_dr in CM (calib-dump length unit; round-1 doc and 149_pred_r2.txt label it mm -- a label error, the threshold 0.01 is applied in cm); improved/worsened at |d zzi_rms_dr| > 0.01; exact sign test')
    for pair in a.pairs:
        b, base = pair.split(':')
        A, B = load(base, samples), load(b, samples)
        for name, keys in strata:
            r = compare(A, B, keys)
            print(f'{b} vs {base} [{name}] n={r["n"]} paired={r["paired"]}  base median zzi_rms_dr {r["base_rms"]:.4f}  '
                  f'median d zzi_rms_dr {r["m_rms"]:+.4f}  median d zzi_ratio {r["m_ratio"]:+.5f}  '
                  f'improved {r["imp"]} worsened {r["wor"]} sign-test p={r["p"]:.3g}  '
                  f'| median d zz_rms_dr (all segs) {r["m_zz"]:+.4f}  median d uncov {r["m_unc"]:+.5f}  '
                  f'=> {"IMPROVES" if r["verdict"] else "does not improve"}')


if __name__ == '__main__':
    main()

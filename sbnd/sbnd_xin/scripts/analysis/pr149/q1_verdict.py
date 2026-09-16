#!/usr/bin/env python3
"""doc pr/149: the pre-registered Q1 verdict (149_pred.txt sec 4) and best-cell rule (sec 6).

For each cell X (vs s0 of the same stage), per stage and pooled over Stage 1 + Stage 2 ISO:
  (a) median per-event delta zz_rms_dr < 0 AND median delta zz_ratio <= 0
  (b) median per-event delta uncov < 0
  (c) toward > away (> 1 cm, labelled ISO events) AND n(<= 3 cm) does not fall
  (d) sum n_stub rises by no more than max(2, 10 % of the stratum's events)
Deltas are paired over events with a calib dump in both arms; the vertex terms use the fixed
labelled denominator (a missing vertex is a failure).

Stage 1 ISO   = d102mpr iso_len_any75 >= 10 cm (nuecc48, ncpi0)
Stage 2 ISO   = strata iso + vtx_iso of 149_stage2_selection.tsv (mcp1k, mcp2k)

Usage: q1_verdict.py --cells cs:s2cs csq2000:s2csq2000 ...   (stage-1 arm : stage-2 arm, prefix pr149)
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
sys.path.insert(0, SX)
from vtx_rules import vtx_io  # noqa: E402

M = os.path.join(SX, 'docs/pr/149_figs/metrics')
f = lambda x: float(x) if x not in ('', 'nan', None) else float('nan')


def load(arm, samples):
    d = {}
    for s in samples:
        p = os.path.join(M, f'{arm}-{s}.tsv')
        if os.path.exists(p):
            for r in csv.DictReader(open(p), delimiter='\t'):
                d[(s, int(r['event']))] = r
    return d


def strata():
    R = load('d102mpr', ['nuecc48', 'ncpi0'])
    s1 = [k for k in R if R[k]['has_calib'] == '1' and f(R[k]['iso_len_any75']) >= 10]
    s2 = []
    for r in csv.DictReader(open(os.path.join(SX, 'docs/pr/149_figs/149_stage2_selection.tsv')), delimiter='\t'):
        if r['stratum'] in ('iso', 'vtx_iso'):
            s2.append((r['sample'], int(r['event'])))
    return s1, s2


def pieces(A, B, keys, lab):
    both = [k for k in keys if A.get(k, {}).get('has_calib') == '1' and B.get(k, {}).get('has_calib') == '1']
    d_rms = [f(B[k]['zz_rms_dr']) - f(A[k]['zz_rms_dr']) for k in both]
    d_ratio = [f(B[k]['zz_ratio']) - f(A[k]['zz_ratio']) for k in both]
    d_unc = [f(B[k]['uncov']) - f(A[k]['uncov']) for k in both]
    stubA = sum(int(f(A[k]['n_stub'])) for k in both)
    stubB = sum(int(f(B[k]['n_stub'])) for k in both)

    def dist(r, e):
        if r is None or r.get('has_calib') != '1':
            return float('inf')
        return math.dist((f(r['vx']), f(r['vy']), f(r['vz'])), lab[e])
    lk = [k for k in keys if k[1] in lab]
    da = np.array([dist(A.get(k), k[1]) for k in lk])
    db = np.array([dist(B.get(k), k[1]) for k in lk])
    return dict(n=len(keys), both=len(both), d_rms=d_rms, d_ratio=d_ratio, d_unc=d_unc, stubA=stubA, stubB=stubB,
                tow=int((db < da - 1).sum()), away=int((db > da + 1).sum()), n3A=int((da <= 3).sum()),
                n3B=int((db <= 3).sum()), nlab=len(lk))


def merge(p, q):
    out = {}
    for k in p:
        out[k] = p[k] + q[k]
    return out


def med(x):
    x = [v for v in x if math.isfinite(v)]
    return float(np.median(x)) if x else float('nan')


def verdict(P):
    a = med(P['d_rms']) < 0 and med(P['d_ratio']) <= 0
    b = med(P['d_unc']) < 0
    c = P['tow'] > P['away'] and P['n3B'] >= P['n3A']
    d = (P['stubB'] - P['stubA']) <= max(2, 0.10 * P['n'])
    return dict(a=a, b=b, c=c, d=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', required=True)
    a = ap.parse_args()
    lab = {d['eventNo']: d['truth'] for d in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105) if d['truth']}
    s1, s2 = strata()
    A1 = load('pr149s0', ['nuecc48', 'ncpi0'])
    A2 = load('pr149s2s0', ['mcp1k', 'mcp2k'])
    print('# doc pr/149 Q1 verdict (149_pred.txt sec 4); criteria a b c d; PASS = all four')
    rows = []
    for cell in a.cells:
        c1, c2 = cell.split(':')
        B1 = load(f'pr149{c1}', ['nuecc48', 'ncpi0'])
        B2 = load(f'pr149{c2}', ['mcp1k', 'mcp2k'])
        P1 = pieces(A1, B1, s1, lab)
        P2 = pieces(A2, B2, s2, lab) if B2 else None
        for name, P in (('stage1', P1), ('stage2', P2), ('pooled', merge(P1, P2) if P2 else None)):
            if P is None:
                print(f'{c1:10s} {name}: (no stage-2 arm)')
                continue
            v = verdict(P)
            npass = sum(v.values())
            print(f'{c1:10s} {name}: n={P["n"]} paired={P["both"]}  d_rms {med(P["d_rms"]):+.4f}  d_ratio {med(P["d_ratio"]):+.4f}  '
                  f'd_uncov {med(P["d_unc"]):+.4f}  vtx lab {P["nlab"]} toward {P["tow"]} away {P["away"]} '
                  f'<=3cm {P["n3A"]}->{P["n3B"]}  stubs {P["stubA"]}->{P["stubB"]}  '
                  f'a={int(v["a"])} b={int(v["b"])} c={int(v["c"])} d={int(v["d"])}  passed {npass}/4'
                  + ('  PASS' if npass == 4 else ''))
            if name == 'pooled':
                rows.append((npass, c1))
    if rows:
        best = max(rows, key=lambda r: r[0])[0]
        print(f'# most criteria passed (pooled): {best}/4 by {[c for n, c in rows if n == best]} '
              '(ties -> fewer Q2 adjudicated degradations -> cs, 149_pred.txt sec 6)')


if __name__ == '__main__':
    main()

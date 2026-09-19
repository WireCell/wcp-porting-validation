#!/usr/bin/env python3
# doc sbnd_xin/pr/150: fork by duplication (CLAUDE.md M10) of scripts/analysis/pr149/vertex_tolerance.py (untouched): metrics and
# outputs under docs/pr/150_figs; the pr/149 strata file 149_stage2_selection.tsv and the d102mpr metrics are read
# from 149_figs on purpose (the epoch-reference strata, never re-drawn).
"""doc pr/149: vertex accuracy vs the vtx105 clicks at several tolerances, and large movers.

Why beyond pr149_metrics.py's 1/3 cm: the vtx105 clicks are rank-1 picks of PR vertex
candidates of the (stepped) arm the scan was taken on, so a click sits within ~0 cm of s0's
own vertex (b1).  Any refit moves every vertex by a few mm-cm and costs the 1 cm count
without a topological change.  A >10 cm move is a different vertex choice and is immune
to that anchoring.  Fixed denominator: every labelled event of the strata; a missing main
vertex counts as a failure.

Usage: vertex_tolerance.py --stage {1,2,3} --base ARM --arms ARM [ARM ...]
(stage 3 = full mcp1k + mcp2k with the stage-1 strata rule applied to d102mpr)
"""
import argparse, csv, math, os, sys
import numpy as np
SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
sys.path.insert(0, SX)
from vtx_rules import vtx_io
M = os.path.join(SX, 'docs/pr/150_figs/metrics')
f = lambda x: float(x) if x not in ('', 'nan', None) else float('nan')

def load(arm, samples):
    d = {}
    for s in samples:
        p = os.path.join(M, f'{arm}-{s}.tsv')
        if os.path.exists(p):
            for r in csv.DictReader(open(p), delimiter='\t'):
                d[(s, int(r['event']))] = r
    return d

ap = argparse.ArgumentParser()
ap.add_argument('--stage', type=int, required=True, choices=[1, 2, 3])
ap.add_argument('--base', required=True)
ap.add_argument('--arms', nargs='+', required=True)
ap.add_argument('--samples', nargs='+', help='stage 3 only: restrict to these samples (default mcp1k mcp2k)')
a = ap.parse_args()
lab = {d['eventNo']: d['truth'] for d in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105) if d['truth']}
if a.stage in (1, 3):
    # stage 3 = the full mcp1k + mcp2k samples, strata by the same d102mpr rule as stage 1
    samples = ['nuecc48', 'ncpi0'] if a.stage == 1 else (a.samples or ['mcp1k', 'mcp2k'])
    R = load('d102mpr', samples)
    strata = {'ISO': [k for k in R if R[k]['has_calib'] == '1' and f(R[k]['iso_len_any75']) >= 10],
              'control': [k for k in R if R[k]['has_calib'] == '1' and f(R[k]['iso_len_any75']) == 0]}
    strata['all'] = list(R)
else:
    samples = ['mcp1k', 'mcp2k']
    st = {}
    for r in csv.DictReader(open(os.path.join(SX, 'docs/pr/149_figs/149_stage2_selection.tsv')), delimiter='\t'):
        st.setdefault(r['stratum'], []).append((r['sample'], int(r['event'])))
    strata = {'ISO(iso+vtx_iso)': st['iso'] + st['vtx_iso'], 'ISO-strong(iso)': st['iso'], 'control': st['control']}
B = load(a.base, samples)

def dist(r, e):
    if r is None or r.get('has_calib') != '1':
        return float('inf')
    v = (f(r['vx']), f(r['vy']), f(r['vz']))
    return math.dist(v, lab[e]) if all(map(math.isfinite, v)) else float('inf')

print(f'# vertex_tolerance stage {a.stage} base {a.base}; clicks vtx105; fixed denominator (missing vertex = inf)')
for name, keys in strata.items():
    keys = [k for k in keys if k[1] in lab]
    d0 = np.array([dist(B.get(k), k[1]) for k in keys])
    print(f'[{name}] labelled {len(keys)}  base b1: dist<0.05cm {int((d0 < 0.05).sum())}')
    for arm in [a.base] + a.arms:
        X = load(arm, samples)
        d = np.array([dist(X.get(k), k[1]) for k in keys])
        big_away = int(((d > d0 + 10)).sum()); big_tow = int(((d < d0 - 10)).sum())
        print(f'   {arm:16s} <=1cm {int((d <= 1).sum()):4d}  <=3cm {int((d <= 3).sum()):4d}  <=10cm {int((d <= 10).sum()):4d}  '
              f'lost {int(np.isinf(d).sum()):3d}  | vs base: moved>10cm away {big_away:3d} toward {big_tow:3d}  '
              f'median shift of kept(<=3cm in both) {np.median(np.abs(d - d0)[(d <= 3) & (d0 <= 3)]) if ((d <= 3) & (d0 <= 3)).any() else float("nan"):.2f} cm')

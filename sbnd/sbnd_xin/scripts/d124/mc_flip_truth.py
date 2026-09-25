#!/usr/bin/env python3
"""doc sbnd_xin/124 S3 -- grade the MC flips of flip_anatomy.py with truth: a candidate is SIGNAL when its reco
vertex is in the FV (5<|x|<190, |y|<190, 10<z<450, vertex_default=='0') and within 5 cm of a true numuCC vertex
(the doc 107/115 definition).  Per class: gains that are signal / background, losses that were signal / background.

usage: mc_flip_truth.py <anatomy.tsv> <products_A> <products_B>
"""
import csv, sys
from collections import Counter
def best(p):
    o = {}
    for r in csv.DictReader(open(p + '/candidates.tsv'), delimiter='\t'):
        k = (int(r['run']), int(r['subrun']), int(r['event']))
        if k not in o or float(r['numu_score']) > float(o[k]['numu_score']): o[k] = r
    return o
def sig(r):
    if r is None: return None
    fv = 5 < abs(float(r['nu_x'])) < 190 and abs(float(r['nu_y'])) < 190 and 10 < float(r['nu_z']) < 450 and r['vertex_default'] == '0'
    d = r['t_dist_cm']
    return bool(fv and r['t_flav'] == 'numu' and r['t_ccnc'] == 'CC' and d not in ('', 'nan') and float(d) < 5)
A, B = best(sys.argv[2]), best(sys.argv[3])
C = Counter()
for r in csv.DictReader(open(sys.argv[1]), delimiter='\t'):
    k = (int(r['run']), int(r['subrun']), int(r['event']))
    sa, sb = sig(A.get(k)), sig(B.get(k))
    grp = 'QL-main' if r['class'] == 'QL-main' else 'churn(%s)' % r['class']
    s = sb if r['kind'] == 'GAINED' else sa
    C[('QL-main' if r['class'] == 'QL-main' else 'churn', r['kind'], 'signal' if s else 'bkg')] += 1
    print('%s/%s/%s %-6s %-9s numu %7.2f -> %7.2f dvtx %6s  signal A=%s B=%s  t=%s/%s dA=%s dB=%s' % (k + (r['kind'], r['class'], float(r['numu_A']), float(r['numu_B']), r['dvtx_cm'], sa, sb,
          (A.get(k) or B.get(k))['t_flav'], (A.get(k) or B.get(k))['t_ccnc'], A[k]['t_dist_cm'] if k in A else '-', B[k]['t_dist_cm'] if k in B else '-')))
for k in sorted(C): print('  %-8s %-6s %-6s %d' % (k + (C[k],)))

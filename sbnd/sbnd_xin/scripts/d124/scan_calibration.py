#!/usr/bin/env python3
"""doc sbnd_xin/124 sec 4.1 -- calibrate the blind scan on MC truth.  For each MC flip the TRUTH-RIGHT arm is the
one whose numu > 0.9 verdict is correct: pass if its candidate is signal (reco vertex in the FV, vertex_default
0, within 5 cm of a true numuCC vertex), fail otherwise.  The scan's preferred arm (unblinded through the map)
is scored against it, per class.

usage: scan_calibration.py <verdicts.tsv> <blind_map_mc.tsv> <anatomy_r3cv.tsv> <products_A> <products_B> <pr_root_A>
"""
import csv, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from flip_anatomy import index_pr
V = {r['png'].strip(): r for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')}
M = {(r['tag'], r['event']): r for r in csv.DictReader(open(sys.argv[2]), delimiter='\t')}
def best(p):
    o = {}
    for r in csv.DictReader(open(p + '/candidates.tsv'), delimiter='\t'):
        k = (int(r['run']), int(r['subrun']), int(r['event']))
        if k not in o or float(r['numu_score']) > float(o[k]['numu_score']): o[k] = r
    return o
def sig(r):
    if r is None: return False
    fv = 5 < abs(float(r['nu_x'])) < 190 and abs(float(r['nu_y'])) < 190 and 10 < float(r['nu_z']) < 450 and r['vertex_default'] == '0'
    d = r['t_dist_cm']
    return bool(fv and r['t_flav'] == 'numu' and r['t_ccnc'] == 'CC' and d not in ('', 'nan') and float(d) < 5)
PA, PB = best(sys.argv[4]), best(sys.argv[5]); IA = index_pr(sys.argv[6])
C = Counter()
for r in csv.DictReader(open(sys.argv[3]), delimiter='\t'):
    k = (int(r['run']), int(r['subrun']), int(r['event']))
    f = os.path.basename(os.path.dirname(IA[k])); tag = 'mc' + f; png = '%s_%s.png' % (tag, k[2])
    v = V.get(png)
    if v is None:
        print('no verdict', png); continue
    passer = PB.get(k) if r['kind'] == 'GAINED' else PA.get(k)
    right = ('hits' if r['kind'] == 'GAINED' else 'reco1') if sig(passer) else ('reco1' if r['kind'] == 'GAINED' else 'hits')
    m = M[(tag, str(k[2]))]
    pref = {'A': m['A'], 'B': m['B']}.get(v['verdict'].strip())
    pref = {'first': 'reco1', 'second': 'hits'}.get(pref, v['verdict'].strip())
    score = 'agree' if pref == right else ('disagree' if pref in ('reco1', 'hits') else pref)
    cls = 'QL-main' if r['class'] == 'QL-main' else 'churn'
    C[(cls, score)] += 1; C[('all', score)] += 1
    C[('conf%s' % v['confidence'].strip(), score)] += 1
    print('%-18s %-6s %-9s truth-right %-5s scan %-7s c%s %-8s -> %s' % (png, r['kind'], r['class'], right, pref, v['confidence'].strip(), v['object'].strip(), score))
for k in sorted(C): print('  %-8s %-9s %d' % (k + (C[k],)))

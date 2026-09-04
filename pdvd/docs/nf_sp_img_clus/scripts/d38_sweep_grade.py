#!/usr/bin/env python3
"""doc pdvd/38 -- the sweep table: two axes plus the harm the two axes hide.

Reads the TSV written by d36_fit_twoaxis_scan.py and reports, per arm:
  coverage   fraction of every cluster's own 3-D charge within 2 cm of its fit
  >2 / >10   fraction of fit points that far from ANY 3-D charge of the event
  trimmed    clusters whose fit lost points against the baseline arm
  lost>5pt   clusters that lost more than 5 coverage points
  destroyed  clusters left with NO fit at all
The last two matter because an aggregate that improves on both axes can still
be amputating a minority of clusters outright -- coverage is charge-weighted,
so a short cluster's total loss barely moves it.

Usage:
  python3 d36_fit_twoaxis_scan.py d38off,d38h2,d38h3,d38h5 sweep.tsv
  python3 d38_sweep_grade.py sweep.tsv d38off        # baseline arm first
"""
import csv, sys
R = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
BASE = sys.argv[2]
arms = [k[:-2] for k in R[0] if k.endswith('_n')]
arms = [BASE] + [a for a in arms if a != BASE]
nq = sum(int(r['nq']) for r in R)
print('clusters with >=50 charge points and a fit in some arm: %d (%d charge points)' % (len(R), nq))
print('\n%-10s %8s %10s %9s %8s %8s %8s %9s %10s' % (
    'arm', 'clusters', 'coverage', 'fit pts', '>2cm', '>10cm', 'trimmed', 'lost>5pt', 'destroyed'))
for t in arms:
    n = sum(int(r[t+'_n']) for r in R)
    cov = sum(int(r['nq'])*float(r[t+'_cov']) for r in R)
    f2 = sum(int(r[t+'_n'])*float(r[t+'_f2']) for r in R)
    f10 = sum(int(r[t+'_n'])*float(r[t+'_f10']) for r in R)
    nf = len([r for r in R if int(r[t+'_n']) > 0])
    if t == BASE:
        tr = lost = dead = 0
    else:
        tr = len([r for r in R if int(r[t+'_n']) < int(r[BASE+'_n'])])
        lost = len([r for r in R if float(r[BASE+'_cov']) - float(r[t+'_cov']) > 0.05])
        dead = len([r for r in R if int(r[BASE+'_n']) > 0 and int(r[t+'_n']) == 0])
    print('%-10s %8d %9.1f%% %9d %7.1f%% %7.1f%% %8s %9s %10s' % (
        t, nf, 100*cov/nq, n, 100*f2/n, 100*f10/n,
        '-' if t == BASE else tr, '-' if t == BASE else lost, '-' if t == BASE else dead))
print('\nworst coverage losses per arm (baseline %s):' % BASE)
for t in arms[1:]:
    S = sorted(R, key=lambda r: -(float(r[BASE+'_cov']) - float(r[t+'_cov'])))[:5]
    print('  %s:' % t)
    for r in S:
        d = float(r[BASE+'_cov']) - float(r[t+'_cov'])
        if d <= 0.0: continue
        print('    %s/%-3s cl %-4s fit %5s -> %-5s cov %3.0f%% -> %3.0f%%  >2cm %3.0f%% -> %3.0f%%' % (
            r['run'], r['idx'], r['cid'], r[BASE+'_n'], r[t+'_n'],
            100*float(r[BASE+'_cov']), 100*float(r[t+'_cov']),
            100*float(r[BASE+'_f2']), 100*float(r[t+'_f2'])))

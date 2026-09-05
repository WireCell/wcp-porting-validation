#!/usr/bin/env python3
"""doc pdvd/39 -- what the STM fit residual actually IS, after the doc-38 trim.

Doc 38 graded the trim by two population aggregates (coverage, ghost).  Both are
charge/point weighted, so both are dominated by a handful of very large objects
and neither says whether the residual is a broad property of the fits or a
minority of pathological clusters.  This script answers that from the same TSV:

  * the MEDIAN cluster, next to the weighted mean -- the two disagree sharply
  * concentration: what share of all uncovered charge / all ghost points sits in
    the worst 10 / 50 / 100 clusters
  * whether low coverage and high ghost are the SAME clusters (they are), which
    decides whether this is one defect or two
  * the fits the trim removed outright, with the ghost fraction they had

Input is the TSV written by d36_fit_twoaxis_scan.py.

Usage:
  python3 d36_fit_twoaxis_scan.py d38off,d38h5,d38h10,d38h20,d38h40 sweep.tsv
  python3 d39_residual_profile.py sweep.tsv d38h20 d38off
       #                          <tsv>      <arm> <baseline for the removed-fit table>
"""
import csv, sys
import numpy as np

TSV, ARM = sys.argv[1], sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else None
ALL = list(csv.DictReader(open(TSV), delimiter='\t'))
R = [r for r in ALL if int(r[ARM + '_n']) > 0]

nq = np.array([int(r['nq']) for r in R], float)
cov = np.array([float(r[ARM + '_cov']) for r in R])
f2 = np.array([float(r[ARM + '_f2']) for r in R])
n = np.array([int(r[ARM + '_n']) for r in R], float)
unc = nq * (1 - cov)
ghost = n * f2

print('arm %s: %d clusters with a fit, %d charge points, %d fit points'
      % (ARM, len(R), nq.sum(), n.sum()))
print('\ncoverage  charge-weighted %.1f%%   MEDIAN cluster %.1f%%   mean %.1f%%'
      % (100 * (nq * cov).sum() / nq.sum(), 100 * np.median(cov), 100 * cov.mean()))
print('ghost     point-weighted  %.1f%%   MEDIAN cluster %.1f%%   mean %.1f%%'
      % (100 * ghost.sum() / n.sum(), 100 * np.median(f2), 100 * f2.mean()))

print('\nconcentration (the aggregates are a few objects, not a population trend):')
ou, og = np.argsort(-unc), np.argsort(-ghost)
print('  %8s %26s %24s' % ('worst k', 'share of uncovered charge', 'share of ghost points'))
for k in (10, 50, 100, max(1, int(0.1 * len(R)))):
    print('  %8d %25.1f%% %23.1f%%'
          % (k, 100 * unc[ou[:k]].sum() / unc.sum(), 100 * ghost[og[:k]].sum() / ghost.sum()))

print('\none defect or two?  pearson r(coverage, ghost) = %.3f' % np.corrcoef(cov, f2)[0, 1])
print('  %-10s %6s %13s %18s' % ('coverage', 'n', 'mean ghost', 'median charge pts'))
for lo, hi, lab in ((0, .5, '<50%'), (.5, .8, '50-80%'), (.8, .95, '80-95%'), (.95, 1.01, '>=95%')):
    m = (cov >= lo) & (cov < hi)
    if m.sum():
        print('  %-10s %6d %12.1f%% %18.0f' % (lab, m.sum(), 100 * f2[m].mean(), np.median(nq[m])))
print('  clean on both axes (cov>=90%% and ghost<=5%%): %d of %d (%.1f%%)'
      % (((cov >= .9) & (f2 <= .05)).sum(), len(R), 100 * ((cov >= .9) & (f2 <= .05)).mean()))
print('  cov<50%%: %d   ghost>25%%: %d   BOTH: %d'
      % ((cov < .5).sum(), (f2 > .25).sum(), ((cov < .5) & (f2 > .25)).sum()))

print('\nthe worst objects -- are these single tracks at all?')
print('  %-13s %-5s %8s %6s %6s %9s %9s %s'
      % ('event', 'cl', 'charge', 'cov', 'ghost', 'extent', 'uncov_d', 'separation'))
import json, os, zipfile
from scipy.spatial import cKDTree
W = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), 'work')
for r in sorted(R, key=lambda r: -int(r['nq']) * (1 - float(r[ARM + '_cov'])))[:10]:
    base, cid = '%s_%s' % (r['run'], r['idx']), int(r['cid'])
    zp = '%s/%s_%s/mabc-pr.zip' % (W, base, ARM)
    if not os.path.exists(zp):
        continue
    z = zipfile.ZipFile(zp)
    cl = json.loads(z.read('data/0/0-clustering-global.json'))
    P = np.stack([cl['x'], cl['y'], cl['z']], 1).astype(float)
    pc = np.asarray(cl['cluster_id'], dtype=int)
    ft = json.loads(z.read('data/0/0-stm_fit-global.json'))
    F = np.stack([ft['x'], ft['y'], ft['z']], 1).astype(float)
    fc = np.asarray(ft['cluster_id'], dtype=int)
    C = P[pc == cid]
    X = np.unique(np.round(F[fc == cid], 2), axis=0)
    d, _ = cKDTree(X).query(C)
    un, cv = C[d >= 2], C[d < 2]
    # extent      : diagonal of the cluster's bounding box (cm)
    # uncov_d     : median distance of UNCOVERED charge from the fit (cm)
    # separation  : centroid of uncovered charge to centroid of covered charge
    ext = float(np.linalg.norm(C.max(0) - C.min(0)))
    md = float(np.median(d[d >= 2])) if len(un) else 0.0
    sep = float(np.linalg.norm(un.mean(0) - cv.mean(0))) if len(un) and len(cv) else 0.0
    print('  %-13s %-5d %8d %5.0f%% %5.0f%% %8.0fcm %8.0fcm %8.0fcm'
          % (base, cid, len(C), 100 * float(r[ARM + '_cov']),
             100 * float(r[ARM + '_f2']), ext, md, sep))

if BASE:
    D = [r for r in ALL if int(r[BASE + '_n']) > 0 and int(r[ARM + '_n']) == 0]
    print('\nfits %s removed outright vs %s: %d' % (ARM, BASE, len(D)))
    print('  %-13s %-5s %8s %8s %9s %11s' % ('event', 'cl', 'fit pts', 'charge', 'coverage', 'GHOST >2cm'))
    for r in sorted(D, key=lambda r: -int(r[BASE + '_n'])):
        print('  %-13s %-5s %8s %8s %8.0f%% %10.0f%%'
              % (r['run'] + '_' + r['idx'], r['cid'], r[BASE + '_n'], r['nq'],
                 100 * float(r[BASE + '_cov']), 100 * float(r[BASE + '_f2'])))
    if D:
        g = sorted(float(r[BASE + '_f2']) for r in D)
        print('  median ghost fraction of the removed fits: %.0f%%  (min %.0f%%, max %.0f%%)'
              % (100 * g[len(g) // 2], 100 * g[0], 100 * g[-1]))

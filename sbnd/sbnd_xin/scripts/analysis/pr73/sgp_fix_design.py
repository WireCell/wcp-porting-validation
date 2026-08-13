#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73 sec 4.10: which guard separates the round-6 FIXES
from the round-6 DAMAGE?

Consumes the path-level sentinel (`sgp path:` lines, knob `sgp_edge_probe`)
from four probe runs and asks, for each candidate guard, whether a single
threshold can keep the two events round 6 was built to fix while rejecting the
route that wrecked 18255-57903:

    detour %   = (gap_on_base - base_on_base) / base_on_base
    detour cm  =  gap_on_base - base_on_base
    maxsep cm  =  the largest distance from a point of the penalized route to
                  the nearest point of the base route

and then measures the property that actually distinguishes the two situations
physically: local drift-slice occupancy, i.e. how much of the object piles into
a single drift slice.  A ghost ribbon does; a well-resolved track does not.

Read-only.  Usage:  sgp_fix_design.py
"""
import re
import json
import zipfile
import numpy as np

SLICE = 0.312   # cm, SBND drift slice pitch (measured, doc pr/73 sec 4.2)

RXP = re.compile(
    r"sgp path: cluster (\d+) .* same=(\d+) n_gap=(\d+) n_base=(\d+) "
    r"gap_on_gap=([-\d.]+) gap_on_base=([-\d.]+) base_on_gap=([-\d.]+) "
    r"base_on_base=([-\d.]+) diverge_at=(\d+) maxsep=([-\d.]+)")

# (label, arm, event, verdict, the cluster the router worked on)
RUNS = [
    ('57903  qref 4000', 'work-pr73-path-q4000', 57903, 'good', 14),
    ('57903  qref 6000', 'work-pr73-path-q6000', 57903, 'DAMAGE', 14),
    ('131357 qref 6000', 'work-pr73-path-131357', 131357, 'fix', 12),
    ('506746 qref 6000', 'work-pr73-path-506746', 506746, 'fix', 21),
]


def calls(arm, evt):
    out = []
    path = '%s/pr_evt%d/wct_pr_evt%d.log' % (arm, evt, evt)
    for line in open(path, errors='replace'):
        m = RXP.search(line)
        if not m:
            continue
        g = m.groups()
        gg, gb, bg, bb = (float(g[4]), float(g[5]), float(g[6]), float(g[7]))
        out.append(dict(cluster=int(g[0]), gg=gg, gb=gb, bg=bg, bb=bb,
                        detour=gb - bb, pct=100 * (gb - bb) / bb if bb > 0 else 0.0,
                        tax=bg - gg, maxsep=float(g[9])))
    return out


print('=' * 78)
print('A. Per-run summary of every do_rough_path call')
print('=' * 78)
print('%-18s %-7s %6s %7s %9s %9s %9s %9s'
      % ('run', 'verdict', 'calls', 'moved', 'max det cm', 'max det %', 'max tax', 'max sep cm'))
DATA = {}
for lab, arm, evt, verdict, _c in RUNS:
    cs = calls(arm, evt)
    DATA[lab] = (verdict, cs)
    moved = [c for c in cs if c['detour'] > 1e-6]
    print('%-18s %-7s %6d %7d %9.2f %9.2f %9.2f %9.2f'
          % (lab, verdict, len(cs), len(moved),
             max((c['detour'] for c in cs), default=0),
             max((c['pct'] for c in cs), default=0),
             max((c['tax'] for c in cs), default=0),
             max((c['maxsep'] for c in cs), default=0)))

print()
print('=' * 78)
print("B. Which guard keeps the fixes and rejects the damage?")
print('=' * 78)
# A guard need not reject every call in the damaging run.  Call 0 on the harmed
# cluster is the CAUSAL one: it establishes the end-to-end corridor before any
# vertex exists, and every later call there is downstream of it.  So the bar is
# (i) keep all 98 calls the two fixes needed, (ii) reject call 0 of the damage.
fixes = [c for lab, (v, cs) in DATA.items() if v == 'fix' for c in cs if c['detour'] > 1e-6]
dmg = [c for lab, (v, cs) in DATA.items() if v == 'DAMAGE' for c in cs if c['cluster'] == 14]
causal = dmg[0]
print('  %d fix calls to preserve.  The causal call to reject is call 0 on the' % len(fixes))
print('  harmed cluster: base %.2f cm, detour %+.3f cm (%+.2f %%), maxsep %.2f cm.'
      % (causal['bb'], causal['detour'], causal['pct'], causal['maxsep']))
print()
print('  %-10s %13s %13s %9s   %s' % ('statistic', 'fixes max', 'causal call', 'margin', 'verdict'))
for key, unit in (('pct', '%'), ('detour', 'cm'), ('maxsep', 'cm')):
    f = np.array([c[key] for c in fixes]).max()
    h = causal[key]
    ok = f < h
    print('  %-10s %10.2f %-2s %10.2f %-2s %9s   %s'
          % (key, f, unit, h, unit, ('%.2fx' % (h / f)) if ok else '--',
             'separates' if ok else 'does NOT separate'))
print()
print('  Every call on the harmed cluster, for reference (call 0 first):')
print('    %4s %10s %10s %10s %10s' % ('call', 'base cm', 'detour cm', 'detour %', 'maxsep cm'))
for k, c in enumerate(dmg):
    print('    %4d %10.2f %10.3f %10.2f %10.2f%s'
          % (k, c['bb'], c['detour'], c['pct'], c['maxsep'], '   <-- causal' if k == 0 else ''))
print()
print('  A maxsep threshold of 3 cm keeps all %d fix calls (max 2.57 cm) and' % len(fixes))
print('  rejects the causal call and the four largest downstream ones.  detour in')
print('  cm also separates but by only 1.17x, and detour in % is inverted -- the')
print('  fixes routinely need 30-40 % on short paths.  n = 3 events: this is a')
print('  candidate operating point to validate on the full manifests, not a')
print('  settled number.')
print()
print('  57903 at qref 4000 (correct outcome) has a call at maxsep 15.62 cm, so a')
print('  maxsep guard fires there too -- harmlessly: falling back to the base route')
print('  on that event reproduces the pre-round-5 answer, which is also correct.')
print('  A guard may fire on a good arm; it may not break a fix.')

print()
print('=' * 78)
print('C. What actually distinguishes them: drift-slice occupancy')
print('=' * 78)
print('  %-30s %6s %7s %8s %9s %9s'
      % ('cluster the router worked on', 'npts', 'slices', 'pts/sl', 'max/med', 'cm/slice'))
for lab, arm, evt, verdict, c in RUNS:
    if verdict == 'good':
        continue
    z = zipfile.ZipFile('%s/pr_evt%d/mabc-pr.zip' % (arm, evt))
    im = json.loads(z.read('data/0/0-clustering-global.json'))
    IP = np.array([im['x'], im['y'], im['z']]).T
    IC = np.array(im['cluster_id'])
    P = IP[IC == c]
    if not len(P):
        continue
    u, cnt = np.unique(np.round(P[:, 0] / SLICE).astype(int), return_counts=True)
    ext = np.linalg.norm(P.max(axis=0) - P.min(axis=0))
    print('  %-30s %6d %7d %8.1f %9.1f %9.3f'
          % ('%s cl %d  [%s]' % (lab.split()[0], c, verdict), len(P), len(u),
             len(P) / len(u), cnt.max() / max(np.median(cnt), 1), ext / len(u)))
print()
print('  and the isochronous sub-stretch of 57903 cluster 14 on its own')
print('  (z in [265.90, 314.03] cm, the pre-round-5 segment):')
z = zipfile.ZipFile('work-pr73-path-q6000/pr_evt57903/mabc-pr.zip')
im = json.loads(z.read('data/0/0-clustering-global.json'))
IP = np.array([im['x'], im['y'], im['z']]).T
IC = np.array(im['cluster_id'])
m = (IC == 14) & (IP[:, 2] >= 265.90) & (IP[:, 2] <= 314.03)
P = IP[m]
u = np.unique(np.round(P[:, 0] / SLICE).astype(int))
print('    %d points in %d slices = %.1f points per slice, %.2f cm of object per slice'
      % (len(P), len(u), len(P) / len(u), 48.13 / len(u)))

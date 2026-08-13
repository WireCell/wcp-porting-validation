#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73 sec 4.8: WHERE are the steiner_gap_penalty edges?

Parses the per-edge sentinel emitted by PatternAlgorithms::ensure_steiner_gap_graph
when the default-OFF knob `sgp_edge_probe` is true (doc pr/73):

    sgp edge: cluster C s=I t=J mid=(x,y,z) w=W bad=B qa=QA qb=QB deficit=D

and answers the question doc pr/73 sec 4.8 left open: the round-6 weak-charge
term broke 18255-57903 while the round-5 unsupported-fraction term did not, and
the hypothesis was that round 6 is the first penalty able to re-route the path
*inside* the isochronous ghost ribbon.  That predicts:

  (a) round 5's `bad` is ~0 inside the ribbon (a ghost ribbon is fully supported
      in 2-D by construction) and non-zero only outside it;
  (b) the edges that become weak when qref goes 4000 -> 6000 -- the ~120 edges
      that flip this event's outcome -- are concentrated inside the ribbon.

Refuted if (b) fails: if the flipping edges sit outside the ribbon, the round-6
damage is a global re-route, not a within-ribbon one.

The ribbon is defined geometrically and independently of any fit: the drift-slice
range of the pre-round-5 isochronous segment.  Default z window below.

Read-only.  Usage:
    sgp_edge_map.py <LOGFILE> [--cluster N] [--zlo CM] [--zhi CM]
                              [--qlo CHARGE] [--qhi CHARGE] [--tsv OUT.tsv]
                              [--routes ARM_A ARM_B]
"""
import sys
import re
import numpy as np

RX = re.compile(
    r"sgp edge: cluster (\d+) s=(\d+) t=(\d+) mid=\(([-\d.]+),([-\d.]+),([-\d.]+)\) "
    r"w=([\d.]+) bad=([\d.]+) qa=([\d.]+) qb=([\d.]+) deficit=([\d.]+)")

opt = dict(cluster=14, zlo=265.90, zhi=314.03, qlo=4000.0, qhi=6000.0, tsv=None,
           routes=None)
args = []
argv = sys.argv[1:]
i = 0
while i < len(argv):
    a = argv[i]
    if a in ('--cluster',):
        opt['cluster'] = int(argv[i + 1]); i += 2; continue
    if a in ('--zlo', '--zhi', '--qlo', '--qhi'):
        opt[a[2:]] = float(argv[i + 1]); i += 2; continue
    if a == '--tsv':
        opt['tsv'] = argv[i + 1]; i += 2; continue
    if a == '--routes':
        opt['routes'] = (argv[i + 1], argv[i + 2]); i += 3; continue
    if a.startswith('--'):
        sys.exit('unknown flag %s\n\n%s' % (a, __doc__))
    args.append(a); i += 1
if not args:
    sys.exit(__doc__)

rows = []
for line in open(args[0], errors='replace'):
    m = RX.search(line)
    if not m:
        continue
    c, s, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if c != opt['cluster']:
        continue
    rows.append((s, t, float(m.group(4)), float(m.group(5)), float(m.group(6)),
                 float(m.group(7)), float(m.group(8)),
                 float(m.group(9)), float(m.group(10)), float(m.group(11))))
if not rows:
    sys.exit('no "sgp edge:" lines for cluster %d in %s' % (opt['cluster'], args[0]))

A = np.array(rows)
z = A[:, 4]
w, bad, qa, qb, deficit = A[:, 5], A[:, 6], A[:, 7], A[:, 8], A[:, 9]
qmin = np.minimum(qa, qb)
inrib = (z >= opt['zlo']) & (z <= opt['zhi'])

print('log      : %s' % args[0])
print('cluster  : %d,  %d scanned edges' % (opt['cluster'], len(A)))
print('ribbon   : z in [%.2f, %.2f] cm  (the pre-round-5 isochronous segment)'
      % (opt['zlo'], opt['zhi']))
print('           %d edges inside, %d outside' % (inrib.sum(), (~inrib).sum()))
print()


def frac(mask, sel):
    n = (mask & sel).sum()
    d = sel.sum()
    return n, d, (100.0 * n / d if d else 0.0)


print('(a) ROUND 5 -- the unsupported-fraction term, bad > 0:')
b5 = bad > 0
for lab, sel in (('inside the ribbon', inrib), ('outside', ~inrib)):
    n, d, f = frac(b5, sel)
    print('    %-20s %4d of %4d edges penalized  (%5.1f %%)' % (lab, n, d, f))
print('    mean bad: inside %.4f, outside %.4f'
      % (bad[inrib].mean() if inrib.any() else 0, bad[~inrib].mean() if (~inrib).any() else 0))
print()

print('(b) ROUND 6 -- weak-charge term, deficit > 0 at the logged qref:')
b6 = deficit > 0
for lab, sel in (('inside the ribbon', inrib), ('outside', ~inrib)):
    n, d, f = frac(b6, sel)
    print('    %-20s %4d of %4d edges weak       (%5.1f %%)' % (lab, n, d, f))
print()

print('(c) THE FLIPPING EDGES -- weak at qref=%.0f but NOT at qref=%.0f,'
      % (opt['qhi'], opt['qlo']))
print('    i.e. min(qa, qb) in [%.0f, %.0f):' % (opt['qlo'], opt['qhi']))
flip = (qmin >= opt['qlo']) & (qmin < opt['qhi'])
print('    %d such edges in total' % flip.sum())
for lab, sel in (('inside the ribbon', inrib), ('outside', ~inrib)):
    n, d, f = frac(flip, sel)
    print('    %-20s %4d of %4d edges flip       (%5.1f %%)' % (lab, n, d, f))
if flip.sum():
    share_in = 100.0 * (flip & inrib).sum() / flip.sum()
    base_in = 100.0 * inrib.sum() / len(A)
    print()
    print('    share of flipping edges that are inside the ribbon : %5.1f %%' % share_in)
    print('    share of ALL scanned edges  that are inside the ribbon : %5.1f %%' % base_in)
    print('    enrichment factor : %.2fx' % (share_in / base_in if base_in else float('nan')))
print()

print('edge charge (min of the two endpoints), by region:')
print('    %-20s %6s %8s %8s %8s %8s' % ('', 'n', 'q10', 'q25', 'q50', 'q75'))
for lab, sel in (('inside the ribbon', inrib), ('outside', ~inrib)):
    if not sel.any():
        continue
    print('    %-20s %6d %8.0f %8.0f %8.0f %8.0f'
          % (lab, sel.sum(), *np.percentile(qmin[sel], [10, 25, 50, 75])))
print()

print('z profile of the flipping edges (2 cm bins, ribbon marked *):')
lo, hi = np.floor(z.min()), np.ceil(z.max())
for b in np.arange(lo, hi, 5.0):
    m = (z >= b) & (z < b + 5)
    if not m.sum():
        continue
    star = '*' if (b + 2.5 >= opt['zlo'] and b + 2.5 <= opt['zhi']) else ' '
    print('  %s z %6.1f-%6.1f  n=%4d  bad>0 %3d  weak %3d  flip %3d'
          % (star, b, b + 5, m.sum(), (b5 & m).sum(), (b6 & m).sum(), (flip & m).sum()))

print()
print('THE PRICE ACTUALLY PAID.  Mean weight multiplier w\'/w by region,')
print('at gap scale 2.0 / weak scale 5.0 (the SBND operating point).')
print('What steers a shortest path is the RATIO outside:inside -- a term that')
print('taxes the non-ribbon part of the cluster makes the ribbon relatively')
print('cheaper, whichever way its own absolute number goes.')


def deficit_at(qref):
    return 0.5 * (np.maximum(0.0, 1.0 - qa / qref) + np.maximum(0.0, 1.0 - qb / qref))


GAP, WEAK = 2.0, 5.0
print('  %-34s %9s %9s %9s' % ('', 'inside', 'outside', 'out/in'))
for lab, mult in (
        ('round 5            1 + 2.0*bad', 1.0 + GAP * bad),
        ('round 6 qref=4000  + 5.0*def', 1.0 + GAP * bad + WEAK * deficit_at(opt['qlo'])),
        ('round 6 qref=6000  + 5.0*def', 1.0 + GAP * bad + WEAK * deficit_at(opt['qhi']))):
    mi = mult[inrib].mean()
    mo = mult[~inrib].mean()
    print('  %-34s %9.4f %9.4f %9.4f' % (lab, mi, mo, mo / mi))
print()
print('  (deficit at qref=6000 is recomputed here from the logged qa/qb and must')
print('   reproduce the logged deficit column: max |diff| = %.2e)'
      % np.abs(deficit_at(opt['qhi']) - deficit).max())

# --------------------------------------------------------------------------
# The decisive test: price the two COMPETING ROUTES, not the two regions.
# Route A = the corridor the qref=4000 run produced (one smooth segment).
# Route B = the corridor the qref=6000 run produced (the hairpin pair).
# An edge is attributed to a route if its midpoint is within RCUT of that
# route's polyline and NOT within RCUT of the other -- the shared spine
# carries no information about which route wins.
if opt.get('routes'):
    import json
    import os
    import zipfile

    RCUT = 1.5   # cm

    def route(arm):
        z = zipfile.ZipFile(os.path.join(arm, 'pr_evt57903', 'mabc-pr.zip'))
        d = json.loads(z.read('data/0/0-track_fit-global.json'))
        P = np.array([d['x'], d['y'], d['z']]).T
        R = np.array(d['real_cluster_id'])
        return np.vstack([P[R == s] for s in sorted(set(R[R >= 0].tolist()))
                          if (R == s).sum() >= 10])

    RA, RB = route(opt['routes'][0]), route(opt['routes'][1])
    M = A[:, 2:5]
    dA = np.min(np.linalg.norm(M[:, None, :] - RA[None, :, :], axis=2), axis=1)
    dB = np.min(np.linalg.norm(M[:, None, :] - RB[None, :, :], axis=2), axis=1)
    onlyA = (dA < RCUT) & (dB >= RCUT)
    onlyB = (dB < RCUT) & (dA >= RCUT)
    print()
    print('THE DECISIVE TEST -- price the two competing ROUTES.')
    print('  route A = %s  (qref 4000 outcome, one smooth segment)' % opt['routes'][0])
    print('  route B = %s  (qref 6000 outcome, the hairpin pair)' % opt['routes'][1])
    print('  edges within %.1f cm of exactly one route: A-only %d, B-only %d'
          % (RCUT, onlyA.sum(), onlyB.sum()))
    if onlyA.sum() and onlyB.sum():
        print()
        print('  %-34s %9s %9s %9s' % ('', 'A-only', 'B-only', 'B/A'))
        for lab, mult in (
                ('round 5            1 + 2.0*bad', 1.0 + GAP * bad),
                ('round 6 qref=4000  + 5.0*def', 1.0 + GAP * bad + WEAK * deficit_at(opt['qlo'])),
                ('round 6 qref=6000  + 5.0*def', 1.0 + GAP * bad + WEAK * deficit_at(opt['qhi']))):
            ma, mb = mult[onlyA].mean(), mult[onlyB].mean()
            print('  %-34s %9.4f %9.4f %9.4f' % (lab, ma, mb, mb / ma))
        print()
        print('  B/A < 1 means route B (the hairpin) is the cheaper one per unit')
        print('  length under that pricing.  If the hypothesis is right this ratio')
        print('  must fall as qref goes 4000 -> 6000, i.e. round 6 at qref 6000 is')
        print('  what makes the hairpin route competitive.')

if opt['tsv']:
    with open(opt['tsv'], 'w') as fh:
        fh.write('s\tt\tx\ty\tz\tw\tbad\tqa\tqb\tdeficit\tin_ribbon\tflip\n')
        for r, ir, fl in zip(A, inrib, flip):
            fh.write('%d\t%d\t%.2f\t%.2f\t%.2f\t%.3f\t%.4f\t%.0f\t%.0f\t%.4f\t%d\t%d\n'
                     % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                        int(ir), int(fl)))
    print()
    print('wrote', opt['tsv'])

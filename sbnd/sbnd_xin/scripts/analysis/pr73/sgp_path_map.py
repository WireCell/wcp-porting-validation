#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73 sec 4.10: WHY does the re-pricing change the route?

Parses the path-level sentinel emitted by PatternAlgorithms::do_rough_path when
the default-OFF knob `sgp_edge_probe` is true (doc pr/73):

    sgp path: cluster C first=(...) last=(...) same=S n_gap=N n_base=M
              gap_on_gap=..  gap_on_base=..  base_on_gap=..  base_on_base=..
              diverge_at=K maxsep=..
    sgp path pt: cluster C which=gap|base k=.. idx=.. (x,y,z)

For each do_rough_path call the sentinel routes BOTH on the penalized
"steiner_graph_gap" flavor (what production uses) and on the untouched
"steiner_graph" base flavor, then prices each of the two routes under each of
the two weightings.  Optimality forces base_on_base <= gap_on_base and
gap_on_gap <= base_on_gap, so the two meaningful numbers are

    detour = gap_on_base - base_on_base    the extra TRUE length the penalty
                                           talked the router into accepting
    tax    = base_on_gap - gap_on_gap      the penalty the base route would
                                           have paid had it been kept

`detour` is the geometric damage; `tax` is the pressure that caused it.

Read-only.  Usage:
    sgp_path_map.py <LOGFILE> [LOGFILE ...] [--cluster N] [--call I]
                    [--zlo CM] [--zhi CM]
"""
import sys
import re
import os
import numpy as np

RXP = re.compile(
    r"sgp path: cluster (\d+) first=\(([-\d.]+),([-\d.]+),([-\d.]+)\) "
    r"last=\(([-\d.]+),([-\d.]+),([-\d.]+)\) same=(\d+) n_gap=(\d+) n_base=(\d+) "
    r"gap_on_gap=([-\d.]+) gap_on_base=([-\d.]+) base_on_gap=([-\d.]+) "
    r"base_on_base=([-\d.]+) diverge_at=(\d+) maxsep=([-\d.]+)")
RXT = re.compile(
    r"sgp path pt: cluster (\d+) which=(gap|base) k=(\d+) idx=(\d+) "
    r"\(([-\d.]+),([-\d.]+),([-\d.]+)\)")

opt = dict(cluster=14, call=0, zlo=265.90, zhi=314.03)
logs = []
argv = sys.argv[1:]
i = 0
while i < len(argv):
    a = argv[i]
    if a == '--cluster':
        opt['cluster'] = int(argv[i + 1]); i += 2; continue
    if a == '--call':
        opt['call'] = int(argv[i + 1]); i += 2; continue
    if a in ('--zlo', '--zhi'):
        opt[a[2:]] = float(argv[i + 1]); i += 2; continue
    if a.startswith('--'):
        sys.exit('unknown flag %s\n\n%s' % (a, __doc__))
    logs.append(a); i += 1
if not logs:
    sys.exit(__doc__)


def parse(path):
    """Return the list of calls for the requested cluster, each with its two
    routes attached (routes are present only when the two differ)."""
    calls, cur = [], None
    for line in open(path, errors='replace'):
        m = RXP.search(line)
        if m:
            g = m.groups()
            if int(g[0]) != opt['cluster']:
                cur = None
                continue
            cur = dict(first=tuple(float(x) for x in g[1:4]),
                       last=tuple(float(x) for x in g[4:7]),
                       same=int(g[7]), n_gap=int(g[8]), n_base=int(g[9]),
                       gg=float(g[10]), gb=float(g[11]),
                       bg=float(g[12]), bb=float(g[13]),
                       diverge=int(g[14]), maxsep=float(g[15]),
                       gap=[], base=[])
            calls.append(cur)
            continue
        m = RXT.search(line)
        if m and cur is not None and int(m.group(1)) == opt['cluster']:
            cur[m.group(2)].append([float(m.group(5)), float(m.group(6)), float(m.group(7))])
    return calls


print('%-26s %4s %5s %5s %10s %10s %10s %10s %9s %8s'
      % ('log', 'call', 'ngap', 'nbase', 'gap_on_gap', 'gap_on_base',
         'base_on_gap', 'base_on_base', 'detour', 'tax'))
allcalls = {}
for lg in logs:
    calls = parse(lg)
    allcalls[lg] = calls
    tag = os.path.basename(os.path.dirname(os.path.dirname(lg))) or lg
    for k, c in enumerate(calls):
        print('%-26s %4d %5d %5d %10.3f %10.3f %10.3f %10.3f %9.3f %8.3f'
              % (tag if k == 0 else '', k, c['n_gap'], c['n_base'],
                 c['gg'], c['gb'], c['bg'], c['bb'],
                 c['gb'] - c['bb'], c['bg'] - c['gg']))
print()
print('detour = gap_on_base - base_on_base  (extra TRUE length the penalty bought)')
print('tax    = base_on_gap - gap_on_gap    (penalty the base route would have paid)')
print()

# --- the head-to-head on the first (end-to-end) call, which is the one that
# --- establishes the corridor everything downstream inherits
print('=' * 78)
print('CALL %d -- the end-to-end rough path, head to head' % opt['call'])
print('=' * 78)
ref = None
for lg in logs:
    calls = allcalls[lg]
    if opt['call'] >= len(calls):
        print('  %s: no call %d' % (lg, opt['call']))
        continue
    c = calls[opt['call']]
    tag = os.path.basename(os.path.dirname(os.path.dirname(lg))) or lg
    det, tax = c['gb'] - c['bb'], c['bg'] - c['gg']
    print('  %-26s detour %+6.3f cm (%+5.2f %% of %.2f)   tax %6.3f (%5.2f %%)   maxsep %6.3f cm'
          % (tag, det, 100 * det / c['bb'], c['bb'], tax, 100 * tax / c['gg'], c['maxsep']))
    print('  %-26s first=(%.2f,%.2f,%.2f) last=(%.2f,%.2f,%.2f)  n_gap=%d n_base=%d'
          % ('', *c['first'], *c['last'], c['n_gap'], c['n_base']))
    if ref is None and c['base']:
        ref = np.array(c['base'])

# --- where, in z, does the penalized route leave the base route?
print()
print('WHERE the penalized route departs from the base route (per-point offset)')
print('ribbon = z in [%.2f, %.2f] cm' % (opt['zlo'], opt['zhi']))
for lg in logs:
    calls = allcalls[lg]
    if opt['call'] >= len(calls):
        continue
    c = calls[opt['call']]
    if not c['gap'] or not c['base']:
        print('  %s: routes identical, nothing to map' % lg)
        continue
    G, B = np.array(c['gap']), np.array(c['base'])
    d = np.min(np.linalg.norm(G[:, None, :] - B[None, :, :], axis=2), axis=1)
    tag = os.path.basename(os.path.dirname(os.path.dirname(lg))) or lg
    inrib = (G[:, 2] >= opt['zlo']) & (G[:, 2] <= opt['zhi'])
    print()
    print('  %s' % tag)
    print('    gap-route points off the base route by > 1 cm: %d of %d'
          % ((d > 1).sum(), len(G)))
    for lab, sel in (('inside the ribbon', inrib), ('outside', ~inrib)):
        if not sel.any():
            continue
        print('      %-20s n=%3d  mean offset %5.2f cm  max %5.2f cm'
              % (lab, sel.sum(), d[sel].mean(), d[sel].max()))
    far = G[d > 1]
    if len(far):
        print('    z span of the departing points: [%.1f, %.1f] cm'
              % (far[:, 2].min(), far[:, 2].max()))
        n_in = ((far[:, 2] >= opt['zlo']) & (far[:, 2] <= opt['zhi'])).sum()
        print('    of which inside the ribbon: %d of %d (%.0f %%)'
              % (n_in, len(far), 100.0 * n_in / len(far)))

#!/usr/bin/env python3
"""doc 97 -- which events does each arm change at stage A, and how do the sets relate?

Hashes ONE product per event -- ql_evt<ID>/mabc-all-apa.zip, the all-APA Q/L
output every downstream stage reads -- so it can cover all 3067 events in
minutes where the four-product gate takes half an hour.  Use d97_ql_gate.py when
the question is "is this byte-identical"; use this when the question is "which
events moved, and do two arms move the same ones".

  Usage: d97_stagea_sets.py [baseline-suffix] [arm-suffix ...]
Read-only.
"""
import glob, hashlib, os, sys, zipfile
from multiprocessing import Pool

BASE = sys.argv[1] if len(sys.argv) > 1 else 'd97off2'
ARMS = sys.argv[2:] or ['d97on', 'd97fv', 'grp0825']
SAMPLES = ['ncpi0', 'nuecc48', 'mcp1k', 'mcp2k']


def roll(p):
    h = hashlib.sha256()
    with zipfile.ZipFile(p) as z:
        for n in sorted(z.namelist()):
            if not n.endswith('/'):
                h.update(n.encode()); h.update(z.read(n))
    return h.hexdigest()


def one(job):
    smp, evt, arm = job
    pa = f'work-{smp}-{arm}/ql_evt{evt}/mabc-all-apa.zip'
    pb = f'work-{smp}-{BASE}/ql_evt{evt}/mabc-all-apa.zip'
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return (smp, evt, arm, None)
    return (smp, evt, arm, roll(pa) != roll(pb))


jobs, universe = [], []
for smp in SAMPLES:
    for d in sorted(glob.glob(f'work-{smp}-{BASE}/ql_evt*')):
        evt = os.path.basename(d)[len('ql_evt'):]
        universe.append((smp, evt))
        for arm in ARMS:
            jobs.append((smp, evt, arm))

with Pool(int(os.environ.get('D97_JOBS', 14))) as p:
    res = p.map(one, jobs)

sets = {a: set() for a in ARMS}
missing = {a: 0 for a in ARMS}
for smp, evt, arm, d in res:
    if d is None:
        missing[arm] += 1
    elif d:
        sets[arm].add((smp, evt))

print(f'baseline: work-*-{BASE}   universe: {len(universe)} events')
for a in ARMS:
    print(f'  {a:<12} changes stage A on {len(sets[a]):>5} events '
          f'({100.0 * len(sets[a]) / max(1, len(universe)):5.2f}%)'
          f'{"   [" + str(missing[a]) + " not comparable]" if missing[a] else ""}')

if len(ARMS) >= 2:
    print()
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            A, B = sets[a], sets[b]
            print(f'  {a} vs {b}: both {len(A & B)}, {a} only {len(A - B)}, '
                  f'{b} only {len(B - A)}')
for a in ARMS:
    if len(sets[a]) <= 60:
        print(f'\n  events changed by {a}:')
        for smp, evt in sorted(sets[a]):
            print(f'    {smp} {evt}')

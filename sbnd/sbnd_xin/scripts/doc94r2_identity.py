#!/usr/bin/env python3
"""doc 94 round 2 -- prove an arm moved NO verdict against a named baseline.

The probe runs entry_rise_guard=true with guard_entry_min_cm=1000, which is
above the feature's range, so no rejection is reachable.  This is the
empirical check of that claim: per-bundle (main_id, in_beam, tgm, stm, fc, lm,
label) must be identical event for event.

Baseline defaults to d94hadron -- the round-2 baseline, because
vertex_hadron_guard is SBND production as of ref/prod-2026-09-02.  Comparing
against prod0901b instead would re-attribute round 1's flips to this guard.

  Usage: doc94r2_identity.py <arm-suffix> [baseline-suffix] [sample ...]
Read-only.
"""
import glob, os, sys

def key(f):
    rows = [l.split() for l in open(f) if l.strip()]
    if len(rows) < 2:
        return None
    h = rows[0]
    ix = [h.index(k) for k in ('main_id', 'in_beam', 'tgm', 'stm', 'fc', 'lm', 'label')]
    return sorted(tuple(r[i] for i in ix) for r in rows[1:])

arm = sys.argv[1] if len(sys.argv) > 1 else 'r2probe'
base = sys.argv[2] if len(sys.argv) > 2 else 'd94hadron'
samples = sys.argv[3:] or ['ncpi0', 'nuecc48', 'mcp1k', 'mcp2k']

same = diff = miss = 0
bad = []
for s in samples:
    for p in sorted(glob.glob(f'work-{s}-{arm}/pr_evt*/nusel-evt*.tsv')):
        q = p.replace(f'-{arm}/', f'-{base}/')
        if not os.path.exists(q):
            miss += 1
            continue
        if key(p) == key(q):
            same += 1
        else:
            diff += 1
            bad.append(p)
print(f'arm work-*-{arm}  vs  baseline work-*-{base}')
print(f'events compared : {same + diff}')
print(f'  IDENTICAL     : {same}')
print(f'  DIFFER        : {diff}')
print(f'  no partner    : {miss}')
for b in bad[:10]:
    print('   ', b)
sys.exit(1 if diff else 0)

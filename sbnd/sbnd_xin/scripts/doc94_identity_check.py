#!/usr/bin/env python3
"""doc 94 -- prove the probe arm moved NO verdict against work-*-prod0901b.

The probe runs descent_guard=true with guard_descent_cos_y=1.01, which is
above the feature's range, so no rejection is reachable.  This is the
empirical check of that claim: per-bundle (main_id, in_beam, tgm, stm, fc, lm,
label) must be identical event for event.  Read-only.
"""
import glob, os, sys

def key(f):
    rows = [l.split() for l in open(f) if l.strip()]
    if len(rows) < 2:
        return None
    h = rows[0]
    ix = [h.index(k) for k in ('main_id', 'in_beam', 'tgm', 'stm', 'fc', 'lm', 'label')]
    return sorted(tuple(r[i] for i in ix) for r in rows[1:])

same = diff = miss = 0
bad = []
for arm in ('ncpi0', 'nuecc48', 'mcp1k', 'mcp2k'):
    for p in sorted(glob.glob(f'work-{arm}-d94probe/pr_evt*/nusel-evt*.tsv')):
        q = p.replace('-d94probe/', '-prod0901b/')
        if not os.path.exists(q):
            miss += 1
            continue
        if key(p) == key(q):
            same += 1
        else:
            diff += 1
            bad.append(p)
print(f'events compared : {same + diff}')
print(f'  IDENTICAL     : {same}')
print(f'  DIFFER        : {diff}')
print(f'  no partner    : {miss}')
for b in bad[:10]:
    print('   ', b)
sys.exit(1 if diff else 0)

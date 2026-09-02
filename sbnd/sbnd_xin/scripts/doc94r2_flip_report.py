#!/usr/bin/env python3
"""doc 94 round 2 -- per-bundle verdict flips between two named arms.

Generalized from doc94_flip_report.py (which hard-coded round 1's pair) so
round 2 can A/B against the CURRENT baseline: vertex_hadron_guard is SBND
production as of ref/prod-2026-09-02, so the round-2 OFF arm is
work-*-d94hadron, NOT work-*-prod0901b.  Diffing against prod0901b would
re-attribute round 1's three recoveries and one release to this guard.

  Usage: doc94r2_flip_report.py <on-suffix> [off-suffix] [sample ...]

Keyed
on (event, main_id) from nusel-evt<ID>.tsv -- never from tracking-pr.root's
T_cluster, whose tgm/stm/fc/lm branches are identically zero (doc 94 sec 8).
Bundles present in only one arm are reported separately, not silently dropped.
Read-only.
"""
import glob, os, sys

ON  = sys.argv[1] if len(sys.argv) > 1 else 'r2entry'
OFF = sys.argv[2] if len(sys.argv) > 2 else 'd94hadron'
SAMPLES = sys.argv[3:] or ['ncpi0', 'nuecc48', 'mcp1k', 'mcp2k']

def rows(f):
    if not os.path.exists(f):
        return None
    r = [l.split() for l in open(f) if l.strip()]
    if len(r) < 2:
        return {}
    h = r[0]
    return {x[h.index('main_id')]: dict(zip(h, x)) for x in r[1:]}

FIELDS = ('in_beam', 'tgm', 'stm', 'fc', 'lm', 'label')
flips, only_off, only_on, nev, nsame = [], [], [], 0, 0
for arm in SAMPLES:
    for on_p in sorted(glob.glob(f'work-{arm}-{ON}/pr_evt*/nusel-evt*.tsv')):
        evt = on_p.split('pr_evt')[1].split('/')[0]
        off = rows(on_p.replace(f'-{ON}/', f'-{OFF}/'))
        on = rows(on_p)
        if off is None:
            continue
        nev += 1
        for mid in sorted(set(off) | set(on)):
            a, b = off.get(mid), on.get(mid)
            if a is None:
                only_on.append((evt, mid)); continue
            if b is None:
                only_off.append((evt, mid)); continue
            if all(a[k] == b[k] for k in FIELDS):
                nsame += 1
            else:
                flips.append((evt, mid, a, b))
print(f'ON  arm : work-*-{ON}')
print(f'OFF arm : work-*-{OFF}')
print(f'events compared            : {nev}')
print(f'bundles identical          : {nsame}')
print(f'bundles FLIPPED            : {len(flips)}')
print(f'bundles only in OFF / ON   : {len(only_off)} / {len(only_on)}')
for evt, mid, a, b in flips:
    d = ' '.join(f'{k}:{a[k]}->{b[k]}' for k in FIELDS if a[k] != b[k])
    print(f"  evt {evt} main {mid}  len {float(a['len_main_cm']):.1f}cm  {d}")
for lbl, lst in (('OFF only', only_off), ('ON only', only_on)):
    for evt, mid in lst[:10]:
        print(f'  {lbl}: evt {evt} main {mid}')

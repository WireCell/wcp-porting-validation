#!/usr/bin/env python3
"""doc 94 -- per-bundle verdict flips, vertex_hadron_guard OFF -> ON.

OFF arm is work-*-prod0901b (production).  ON arm is work-*-d94hadron.  Keyed
on (event, main_id) from nusel-evt<ID>.tsv -- never from tracking-pr.root's
T_cluster, whose tgm/stm/fc/lm branches are identically zero (doc 94 sec 8).
Bundles present in only one arm are reported separately, not silently dropped.
Read-only.
"""
import glob, os, sys

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
for arm in ('ncpi0', 'nuecc48', 'mcp1k', 'mcp2k'):
    for on_p in sorted(glob.glob(f'work-{arm}-d94hadron/pr_evt*/nusel-evt*.tsv')):
        evt = on_p.split('pr_evt')[1].split('/')[0]
        off = rows(on_p.replace('-d94hadron/', '-prod0901b/'))
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

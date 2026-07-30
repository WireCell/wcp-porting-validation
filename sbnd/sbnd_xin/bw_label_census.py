#!/usr/bin/env python3
"""Beam-window label census over a full-1000 nusel arm.

Primary denominator: in-beam bundles (in_beam=1, label != 'no-bundle').
Also: per-event classes, integrity checks, fc split inside nu-candidate.
"""
import glob, os, sys
from collections import Counter

def read_tsv(path):
    with open(path, errors='replace') as f:
        rows = [ln.split() for ln in f.read().splitlines() if ln.strip()]
    if not rows:
        return []
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:] if len(r) == len(head)]

def census(root):
    tsvs = sorted(glob.glob(os.path.join(root, 'nusel_evt*', 'nusel-evt*.tsv')))
    lab = Counter()          # in-beam bundle labels
    fc_by_lab = Counter()    # (label, fc)
    nobundle = 0
    outofwindow = 0
    holes = 0                # in_beam=1, real bundle, tgm == -1
    lm_absent = 0            # lm == -1 rows (in-beam)
    ev_class = Counter()
    per_ev_labels = {}
    for p in tsvs:
        evid = os.path.basename(p)[len('nusel-evt'):-len('.tsv')]
        rows = read_tsv(p)
        inbeam = []
        for r in rows:
            if r['in_beam'] != '1':
                outofwindow += 1
                continue
            if r['label'] == 'no-bundle':
                nobundle += 1
                continue
            inbeam.append(r)
            lab[r['label']] += 1
            fc_by_lab[(r['label'], r['fc'])] += 1
            if r['tgm'] == '-1':
                holes += 1
            if r['lm'] == '-1':
                lm_absent += 1
        per_ev_labels[evid] = [r['label'] for r in inbeam]
        L = set(per_ev_labels[evid])
        if not inbeam:
            ev_class['no-in-beam-bundle'] += 1
        elif L <= {'TGM'}:
            ev_class['TGM only'] += 1
        elif L <= {'LM'}:
            ev_class['LM only'] += 1
        elif L <= {'TGM', 'LM'}:
            ev_class['TGM+LM'] += 1
        elif L & {'TGM', 'LM'}:
            ev_class['mixed (cosmic + keepable)'] += 1
        elif 'STM' in L:
            ev_class['STM (no TGM/LM)'] += 1
        else:
            ev_class['all nu-candidate'] += 1
    return dict(n_events=len(tsvs), lab=lab, fc_by_lab=fc_by_lab,
                nobundle=nobundle, outofwindow=outofwindow, holes=holes,
                lm_absent=lm_absent, ev_class=ev_class,
                per_ev_labels=per_ev_labels)

def show(root, c):
    tot = sum(c['lab'].values())
    print(f"=== {root}   events={c['n_events']}")
    print(f"integrity: in-beam rows with tgm==-1 (not evaluated): {c['holes']}")
    print(f"integrity: in-beam rows with lm==-1  (LM knob off)  : {c['lm_absent']}")
    print(f"out-of-window bundle rows: {c['outofwindow']}   "
          f"in-beam 'no-bundle' rows (flash, no qualifying bundle): {c['nobundle']}")
    print(f"\n-- per IN-BEAM BUNDLE (n={tot})")
    cos = 0
    for k in ('TGM', 'STM', 'LM', 'nu-candidate'):
        n = c['lab'][k]
        if k != 'nu-candidate':
            cos += n
        print(f"   {k:<14} {n:6d}   {100.0*n/tot:6.2f} %")
    for k in c['lab']:
        if k not in ('TGM', 'STM', 'LM', 'nu-candidate'):
            print(f"   {k:<14} {c['lab'][k]:6d}   (unexpected label)")
    print(f"   {'cosmic total':<14} {cos:6d}   {100.0*cos/tot:6.2f} %")
    print(f"\n-- fc split")
    for k in ('TGM', 'STM', 'LM', 'nu-candidate'):
        parts = {f: n for (l, f), n in c['fc_by_lab'].items() if l == k}
        s = sum(parts.values()) or 1
        print(f"   {k:<14} " + "  ".join(
            f"fc={f}: {n} ({100.0*n/s:.1f}%)" for f, n in sorted(parts.items())))
    print(f"\n-- per EVENT (n={c['n_events']})")
    for k, n in sorted(c['ev_class'].items(), key=lambda kv: -kv[1]):
        print(f"   {k:<28} {n:5d}   {100.0*n/c['n_events']:6.2f} %")
    print()

roots = sys.argv[1:]
cs = {}
for r in roots:
    cs[r] = census(r)
    show(r, cs[r])

if len(roots) == 2:
    a, b = roots
    print(f"=== delta {a} vs {b} (per-event in-beam label multiset)")
    A, B = cs[a]['per_ev_labels'], cs[b]['per_ev_labels']
    diff = [e for e in A if sorted(A[e]) != sorted(B.get(e, []))]
    print(f"events differing: {len(diff)}")
    for e in diff[:40]:
        print(f"   evt{e}: {sorted(A[e])}  ->  {sorted(B.get(e, []))}")

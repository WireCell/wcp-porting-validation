#!/usr/bin/env python3
"""doc 94 -- census of the descent feature (cos_y) over a PR arm.

Joins three sources per in-beam bundle:
  * the descent_guard DEBUG probe line in pr_evt<ID>/wct_pr_evt<ID>.log,
    which prints entry/stop/chord/cos_y for EVERY STM-evaluated bundle
    (cluster ident == nusel main_id, verified on work-stmfb8-probe evt4);
  * pr_evt<ID>/nusel-evt<ID>.tsv for the tagger verdict (space-padded, NOT
    tab-separated, despite the extension -- split on whitespace);
  * optionally scan-d59k/stm-baseline.tsv, the 72 bundles the OWNER
    adjudicated by hand (doc 62).  owner_verdict is truth.

Usage:
  doc94_descent_census.py --arm work-mcp1k-d94probe [--arm ...] \
      [--baseline] [--out PREFIX] [--cut -0.25]
Read-only.
"""
import argparse, glob, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))

RE_DESC = re.compile(
    r'descent_guard: cluster (\d+) entry=\(([-\d.]+),([-\d.]+),([-\d.]+)\)cm '
    r'stop=\(([-\d.]+),([-\d.]+),([-\d.]+)\)cm chord=([-\d.]+)cm cos_y=([-\d.]+)')
RE_REJ = re.compile(r'descent_guard: cluster (\d+) rejected')

ap = argparse.ArgumentParser()
ap.add_argument('--arm', action='append', required=True)
ap.add_argument('--baseline', action='store_true')
ap.add_argument('--cut', type=float, default=-0.25)
ap.add_argument('--out')
args = ap.parse_args()


def nusel(path):
    """{main_id: row-dict} for one event."""
    if not os.path.exists(path):
        return {}
    rows = [l.split() for l in open(path) if l.strip()]
    if len(rows) < 2:
        return {}
    hdr = rows[0]
    return {r[hdr.index('main_id')]: dict(zip(hdr, r)) for r in rows[1:]}


recs = []
for arm in args.arm:
    root = arm if os.path.isabs(arm) else os.path.join(BASE, arm)
    for log in sorted(glob.glob(os.path.join(root, 'pr_evt*', 'wct_pr_evt*.log'))):
        evt = re.search(r'pr_evt(\d+)', log).group(1)
        ns = nusel(os.path.join(os.path.dirname(log), f'nusel-evt{evt}.tsv'))
        rejected = set()
        seen = {}
        for line in open(log, errors='replace'):
            if 'descent_guard: cluster' not in line:
                continue
            m = RE_DESC.search(line)
            if m:
                cid = m.group(1)
                seen[cid] = dict(
                    ex=float(m.group(2)), ey=float(m.group(3)), ez=float(m.group(4)),
                    sx=float(m.group(5)), sy=float(m.group(6)), sz=float(m.group(7)),
                    chord=float(m.group(8)), cos_y=float(m.group(9)))
                continue
            r = RE_REJ.search(line)
            if r:
                rejected.add(r.group(1))
        for cid, d in seen.items():
            row = ns.get(cid, {})
            recs.append(dict(arm=os.path.basename(root), evt=evt, cid=cid,
                             run=row.get('run', '?'), subrun=row.get('subrun', '?'),
                             stm=row.get('stm', '?'), tgm=row.get('tgm', '?'),
                             fc=row.get('fc', '?'), label=row.get('label', '?'),
                             length=float(row.get('len_main_cm', 0) or 0),
                             rejected=int(cid in rejected), **d))

print(f'STM-evaluated bundles with a descent probe: {len(recs)}')
stm = [r for r in recs if r['stm'] == '1']
non = [r for r in recs if r['stm'] == '0']
print(f'  of which tagger STM=1 : {len(stm)}')
print(f'                  STM=0 : {len(non)}')

def hist(rs, title):
    if not rs:
        return
    edges = [-1.01, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.25, -0.2, -0.1, 0.0, 0.25, 0.5, 1.01]
    print(f'\n  {title}  (n={len(rs)})')
    for a, b in zip(edges, edges[1:]):
        n = sum(1 for r in rs if a <= r['cos_y'] < b)
        if n:
            print(f'    cos_y [{a:+.2f},{b:+.2f}) : {n:5d}  {"#"*min(60, max(1, n*60//max(1,len(rs))))}')
    for c in (-0.4, -0.3, -0.25, -0.2, -0.1, 0.0):
        n = sum(1 for r in rs if r['cos_y'] > c)
        print(f'    cos_y > {c:+.2f} : {n:5d}  ({100.0*n/len(rs):.1f}%)')

hist(stm, 'tagger STM=1')
hist(non, 'tagger STM=0 (evaluated, not tagged)')

if args.baseline:
    bl = os.path.join(BASE, 'scan-d59k', 'stm-baseline.tsv')
    rows = [l.rstrip('\n').split('\t') for l in open(bl) if l.strip() and not l.startswith('#')]
    hdr = rows[0]
    idx = {k: i for i, k in enumerate(hdr)}
    by = {(r[idx['event']], r[idx['main_id']]): r for r in rows[1:]}
    probe = {(r['evt'], r['cid']): r for r in recs}
    print(f'\n=== owner baseline join (doc 62, {len(by)} bundles) ===')
    tally = defaultdict(lambda: [0, 0, []])
    for key, r in sorted(by.items()):
        cls = r[idx['class']]
        p = probe.get(key)
        tally[cls][0] += 1
        if p is None:
            continue
        tally[cls][1] += 1
        tally[cls][2].append((p['cos_y'], key, float(r[idx['len_cm']])))
    for cls in ('code-STM-correct', 'code-FALSE-STM', 'code-MISSED-STM', 'code-not-STM-correct'):
        tot, res, vals = tally[cls]
        print(f'\n  {cls}: {tot} in baseline, {res} resolved in this arm')
        if not vals:
            continue
        vals.sort()
        over = [v for v in vals if v[0] > args.cut]
        print(f'    cos_y range [{vals[0][0]:+.3f}, {vals[-1][0]:+.3f}]   '
              f'above cut {args.cut:+.2f}: {len(over)} of {res}')
        for v, key, ln in vals:
            mark = '  <-- ABOVE CUT' if v > args.cut else ''
            print(f'      {key[0]}:{key[1]:>3s}  cos_y={v:+.3f}  len={ln:.1f}cm{mark}')

if args.out:
    keys = ['arm', 'evt', 'cid', 'run', 'subrun', 'stm', 'tgm', 'fc', 'label',
            'length', 'ex', 'ey', 'ez', 'sx', 'sy', 'sz', 'chord', 'cos_y', 'rejected']
    with open(args.out, 'w') as f:
        f.write('\t'.join(keys) + '\n')
        for r in sorted(recs, key=lambda r: (r['arm'], int(r['evt']), int(r['cid']))):
            f.write('\t'.join(str(r[k]) for k in keys) + '\n')
    print(f'\nwrote {args.out}  ({len(recs)} rows)')

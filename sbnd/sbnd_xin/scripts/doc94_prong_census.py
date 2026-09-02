#!/usr/bin/env python3
"""doc 94 -- census of the OTHER-TRACK PRONG feature over a PR arm.

check_other_tracks (TaggerCheckSTM.cxx:2872) already logs, at DEBUG, every
segment search_other_tracks hung off the fitted main:

  check_other_tracks: cluster N seg i/n: len=..cm medQ=..MIP lenThr=..cm
                      straight=.. front=(x,y,z)cm

Three of the five doc-94 symptom events carry a prong that is LONG (14.8-25.9
cm) and HEAVILY IONIZING (1.65-2.47 MIP) and CURVED (straightness 0.72-0.97),
which every acceptance clause in that function lets through -- :2909 skips a
hot curved segment by name.  This script measures how common such a prong is
on the STM population, which is the number that decides whether a
prong-based guard is selective or a blunt release.

Read-only.  Usage: doc94_prong_census.py --arm work-mcp1k-d94probe [--arm ...]
"""
import argparse, glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))
RE_SEG = re.compile(
    r'check_other_tracks: cluster (\d+) seg (\d+)/(\d+): len=([-\d.]+)cm '
    r'medQ=([-\d.]+)MIP lenThr=([-\d.]+)cm straight=([-\d.]+) '
    r'front=\(([-\d.]+),([-\d.]+),([-\d.]+)\)cm')

ap = argparse.ArgumentParser()
ap.add_argument('--arm', action='append', required=True)
ap.add_argument('--out')
args = ap.parse_args()


def nusel(path):
    if not os.path.exists(path):
        return {}
    rows = [l.split() for l in open(path) if l.strip()]
    if len(rows) < 2:
        return {}
    h = rows[0]
    return {r[h.index('main_id')]: dict(zip(h, r)) for r in rows[1:]}


segs, bundles = [], {}
for arm in args.arm:
    root = arm if os.path.isabs(arm) else os.path.join(BASE, arm)
    for log in sorted(glob.glob(os.path.join(root, 'pr_evt*', 'wct_pr_evt*.log'))):
        evt = re.search(r'pr_evt(\d+)', log).group(1)
        ns = nusel(os.path.join(os.path.dirname(log), f'nusel-evt{evt}.tsv'))
        for line in open(log, errors='replace'):
            if 'check_other_tracks: cluster' not in line:
                continue
            m = RE_SEG.search(line)
            if not m:
                continue
            cid = m.group(1)
            row = ns.get(cid, {})
            r = dict(evt=evt, cid=cid, stm=row.get('stm', '?'),
                     length=float(row.get('len_main_cm', 0) or 0),
                     i=int(m.group(2)), n=int(m.group(3)),
                     len_cm=float(m.group(4)), medQ=float(m.group(5)),
                     lenThr=float(m.group(6)), straight=float(m.group(7)),
                     fx=float(m.group(8)), fy=float(m.group(9)), fz=float(m.group(10)))
            segs.append(r)
            bundles.setdefault((evt, cid), dict(stm=r['stm'], segs=[]))['segs'].append(r)

stm_b = {k: v for k, v in bundles.items() if v['stm'] == '1'}
print(f'bundles with >=1 logged prong : {len(bundles)}   of which STM=1 : {len(stm_b)}')
print(f'prong rows                    : {len(segs)}')


def frac(pred, label):
    n = sum(1 for v in stm_b.values() if any(pred(s) for s in v['segs']))
    d = len(stm_b) or 1
    print(f'  {label:<52s} {n:5d} / {d}  ({100.0*n/d:5.1f}%)')

print('\nSTM=1 bundles with at least one prong satisfying:')
frac(lambda s: s['medQ'] > 1.5 and s['len_cm'] > 8,  'medQ>1.5 MIP  and len>8 cm')
frac(lambda s: s['medQ'] > 1.5 and s['len_cm'] > 12, 'medQ>1.5 MIP  and len>12 cm')
frac(lambda s: s['medQ'] > 1.5 and s['len_cm'] > 14, 'medQ>1.5 MIP  and len>14 cm')
frac(lambda s: s['medQ'] > 1.6 and s['len_cm'] > 14, 'medQ>1.6 MIP  and len>14 cm')
frac(lambda s: s['medQ'] > 2.0 and s['len_cm'] > 14, 'medQ>2.0 MIP  and len>14 cm')
frac(lambda s: s['medQ'] > 1.5 and s['len_cm'] > 14 and s['straight'] < 0.99,
     'medQ>1.5, len>14, straight<0.99 (the :2909 class)')

print('\nprong (len, medQ) joint distribution on STM=1 bundles:')
lb = [0, 4, 8, 12, 16, 20, 30, 1e9]
qb = [0, 0.4, 0.8, 1.2, 1.5, 2.0, 3.0, 1e9]
sstm = [s for s in segs if s['stm'] == '1']
print('      medQ->  ' + ''.join(f'{a:>7.1f}' for a in qb[:-1]))
for a, b in zip(lb, lb[1:]):
    row = [sum(1 for s in sstm if a <= s['len_cm'] < b and c <= s['medQ'] < d)
           for c, d in zip(qb, qb[1:])]
    tag = f'len {a:.0f}-{b:.0f}' if b < 1e9 else f'len >{a:.0f}'
    print(f'  {tag:<12s}' + ''.join(f'{v:7d}' for v in row))

if args.out:
    keys = ['evt', 'cid', 'stm', 'length', 'i', 'n', 'len_cm', 'medQ', 'lenThr',
            'straight', 'fx', 'fy', 'fz']
    with open(args.out, 'w') as f:
        f.write('\t'.join(keys) + '\n')
        for s in segs:
            f.write('\t'.join(str(s[k]) for k in keys) + '\n')
    print(f'\nwrote {args.out} ({len(segs)} rows)')

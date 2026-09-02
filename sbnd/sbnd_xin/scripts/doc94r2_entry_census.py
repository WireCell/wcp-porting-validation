#!/usr/bin/env python3
"""doc 94 round 2 -- census of the ENTRY-RISE feature over a PR arm.

Joins three sources per in-beam bundle:
  * the entry_rise DEBUG probe line in pr_evt<ID>/wct_pr_evt<ID>.log, which
    prints L_stop / body / ent / rise / shoulder / shoulder_nofirst / excess
    for EVERY STM-evaluated bundle whose muon is long enough for a body
    estimate (cluster ident == nusel main_id);
  * pr_evt<ID>/nusel-evt<ID>.tsv for the tagger verdict (space-padded, NOT
    tab-separated despite the extension -- split on whitespace);
  * optionally scan-d59k/stm-baseline.tsv, the 72 bundles the OWNER
    adjudicated by hand (doc 62).  owner_verdict is truth.

The go/no-go this exists to answer: is `shoulder` BIMODAL over the STM-tagged
population, or a continuum?  The positive side of the labelled set is n=1
(827-27-4), so a continuum means "no separator", not "tighten until the one
event survives" -- that is the n=1 fit doc 94 round 1 killed descent_guard and
the proton_muon_guard re-tune for.

Usage:
  doc94r2_entry_census.py --arm work-mcp1k-r2probe [--arm ...] \
      [--baseline] [--cut 5.0] [--maxcut 30.0] [--out FILE]
Read-only.
"""
import argparse, glob, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))

RE_ENT = re.compile(
    r'entry_rise: cluster (\d+) L_stop=([-\d.]+)cm body=([-\d.]+)MIP\((\d+)pts\) '
    r'ent=([-\d.]+)MIP\((\d+)pts\) rise=([-\d.]+) shoulder=([-\d.]+)cm '
    r'shoulder_nofirst=([-\d.]+)cm excess=([-\d.]+)cm')
RE_REJ = re.compile(r'entry_rise_guard: cluster (\d+) rejected')

ap = argparse.ArgumentParser()
ap.add_argument('--arm', action='append', required=True)
ap.add_argument('--baseline', action='store_true')
ap.add_argument('--cut', type=float, default=5.0)
ap.add_argument('--maxcut', type=float, default=30.0)
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
        rejected, seen = set(), {}
        for line in open(log, errors='replace'):
            if 'entry_rise' not in line:
                continue
            m = RE_ENT.search(line)
            if m:
                seen[m.group(1)] = dict(
                    L_stop=float(m.group(2)), body=float(m.group(3)), body_n=int(m.group(4)),
                    ent=float(m.group(5)), ent_n=int(m.group(6)), rise=float(m.group(7)),
                    shoulder=float(m.group(8)), sh_nofirst=float(m.group(9)),
                    excess=float(m.group(10)))
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

print(f'STM-evaluated bundles with an entry-rise probe: {len(recs)}')
stm = [r for r in recs if r['stm'] == '1']
non = [r for r in recs if r['stm'] == '0']
print(f'  of which tagger STM=1 : {len(stm)}')
print(f'                  STM=0 : {len(non)}')


def hist(rs, title, field='shoulder'):
    if not rs:
        return
    edges = [0.0, 0.01, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 1e9]
    print(f'\n  {title}  (n={len(rs)}), {field}')
    for a, b in zip(edges, edges[1:]):
        n = sum(1 for r in rs if a <= r[field] < b)
        if n:
            lab = f'== 0' if b == 0.01 else f'[{a:.1f},{b:.1f})'
            print(f'    {lab:>14s} : {n:5d}  {"#"*min(60, max(1, n*60//max(1,len(rs))))}')
    n_fire = sum(1 for r in rs if args.cut <= r[field] <= args.maxcut)
    print(f'    IN WINDOW [{args.cut},{args.maxcut}] : {n_fire} of {len(rs)}  ({100.0*n_fire/len(rs):.2f}%)')


hist(stm, 'tagger STM=1')
hist(non, 'tagger STM=0 (evaluated, not tagged)')
hist(stm, 'tagger STM=1', 'sh_nofirst')

fire = sorted([r for r in stm if args.cut <= r['shoulder'] <= args.maxcut],
              key=lambda r: -r['shoulder'])
print(f'\n=== STM=1 bundles the window would release: {len(fire)} ===')
for r in fire:
    print(f"  {r['arm'][5:]:<14s} evt{r['evt']:>7s}:{r['cid']:<4s} run {r['run']}-{r['subrun']} "
          f"len={r['length']:.1f}cm L_stop={r['L_stop']:.1f} body={r['body']:.2f} ent={r['ent']:.2f} "
          f"rise={r['rise']:.2f} sh={r['shoulder']:.1f} nofirst={r['sh_nofirst']:.1f} "
          f"excess={r['excess']:.1f}")

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
        tally[cls][2].append((p['shoulder'], key, float(r[idx['len_cm']]), p))
    for cls in ('code-STM-correct', 'code-FALSE-STM', 'code-MISSED-STM', 'code-not-STM-correct'):
        tot, res, vals = tally[cls]
        print(f'\n  {cls}: {tot} in baseline, {res} with an entry probe in this arm')
        if not vals:
            continue
        vals.sort()
        over = [v for v in vals if args.cut <= v[0] <= args.maxcut]
        print(f'    shoulder range [{vals[0][0]:.1f}, {vals[-1][0]:.1f}] cm   '
              f'IN WINDOW: {len(over)} of {res}')
        for v, key, ln, p in vals:
            mark = '  <-- WOULD FIRE' if args.cut <= v <= args.maxcut else ''
            print(f"      {key[0]}:{key[1]:>3s}  shoulder={v:5.1f}cm rise={p['rise']:.2f} "
                  f"body={p['body']:.2f} len={ln:.1f}cm{mark}")

if args.out:
    keys = ['arm', 'evt', 'cid', 'run', 'subrun', 'stm', 'tgm', 'fc', 'label', 'length',
            'L_stop', 'body', 'body_n', 'ent', 'ent_n', 'rise', 'shoulder', 'sh_nofirst',
            'excess', 'rejected']
    with open(args.out, 'w') as f:
        f.write('\t'.join(keys) + '\n')
        for r in sorted(recs, key=lambda r: (r['arm'], int(r['evt']), int(r['cid']))):
            f.write('\t'.join(str(r[k]) for k in keys) + '\n')
    print(f'\nwrote {args.out}  ({len(recs)} rows)')

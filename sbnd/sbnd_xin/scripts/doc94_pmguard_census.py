#!/usr/bin/env python3
"""doc 94 -- how often does proton_muon_guard overrule a proton endpoint?

doc 63 round 2 added proton_muon_guard to recover ONE missed STM (62613:17,
ks1 = 0.030).  It fires when detect_proton's end matches the muon hypothesis
better than the proton one: ks1 < guard_proton_ks1 (0.040) and ratio3 < 1.1.
On two of the five doc-94 symptom events (304-6-28 ks1=0.035, 146-60-31
ks1=0.031) that guard is what CANCELS a proton call and lets STM stand.

The question this answers: is the 0.040 bar a narrow, high-leverage lever, or
is the whole STM population sitting on top of it?

Log line (DEBUG, TaggerCheckSTM.cxx:1901):
  detect_proton: proton_muon_guard: end matches the muon hypothesis
  (ks1=0.035, ratio3=1.033); not a proton
Read-only.
"""
import argparse, glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))
RE_PM = re.compile(r'proton_muon_guard: end matches the muon hypothesis '
                   r'\(ks1=([-\d.]+), ratio3=([-\d.]+)\)')

ap = argparse.ArgumentParser()
ap.add_argument('--arm', action='append', required=True)
ap.add_argument('--out')
args = ap.parse_args()

rows = []
nev = 0
for arm in args.arm:
    root = arm if os.path.isabs(arm) else os.path.join(BASE, arm)
    for log in sorted(glob.glob(os.path.join(root, 'pr_evt*', 'wct_pr_evt*.log'))):
        nev += 1
        evt = re.search(r'pr_evt(\d+)', log).group(1)
        p = os.path.join(os.path.dirname(log), f'nusel-evt{evt}.tsv')
        stm_ids = set()
        if os.path.exists(p):
            r = [l.split() for l in open(p) if l.strip()]
            if len(r) > 1:
                h = r[0]
                stm_ids = {x[h.index('main_id')] for x in r[1:]
                           if x[h.index('stm')] == '1'}
        for line in open(log, errors='replace'):
            m = RE_PM.search(line)
            if m:
                rows.append(dict(arm=os.path.basename(root), evt=evt,
                                 ks1=float(m.group(1)), ratio3=float(m.group(2)),
                                 any_stm=int(bool(stm_ids))))

print(f'events scanned                      : {nev}')
print(f'proton_muon_guard fires (overrules) : {len(rows)}')
if rows:
    ks = sorted(r['ks1'] for r in rows)
    print(f'  ks1 range [{ks[0]:.4f}, {ks[-1]:.4f}]   median {ks[len(ks)//2]:.4f}')
    print(f'  (bar is guard_proton_ks1 = 0.040; the guard fires BELOW it)')
    for c in (0.010, 0.020, 0.025, 0.030, 0.031, 0.035, 0.040):
        n = sum(1 for k in ks if k < c)
        print(f'    fires that would SURVIVE a bar lowered to {c:.3f} : {n:4d} '
              f'({100.0*n/len(ks):.1f}% of fires)')
    print(f'  fires in an event with >=1 STM tag : '
          f'{sum(r["any_stm"] for r in rows)}')
if args.out:
    with open(args.out, 'w') as f:
        f.write('arm\tevt\tks1\tratio3\tany_stm\n')
        for r in rows:
            f.write(f"{r['arm']}\t{r['evt']}\t{r['ks1']}\t{r['ratio3']}\t{r['any_stm']}\n")
    print(f'\nwrote {args.out} ({len(rows)} rows)')

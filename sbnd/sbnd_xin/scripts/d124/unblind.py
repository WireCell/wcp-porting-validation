#!/usr/bin/env python3
"""doc sbnd_xin/124 -- unblind the hand scan: join the verdicts (A/B) with the shuffle map (A = 'first' arm =
reco1-flash arm, or 'second' = hit-flash arm) and the anatomy classes.  Prints one row per event and the tallies.

usage: unblind.py <verdicts.tsv> <blind_map.tsv> <anatomy.tsv> [...]
"""
import csv, sys
from collections import Counter
V = {r['png'].strip(): r for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')}
M = {(r['tag'], r['event']): r for r in csv.DictReader(open(sys.argv[2]), delimiter='\t')}
C = Counter(); rows = []
for p in sys.argv[3:]:
    for r in csv.DictReader(open(p), delimiter='\t'):
        png = '%s_%s.png' % (r['sample'], r['event'])
        v = V.get(png); m = M[(r['sample'], r['event'])]
        if v is None:
            print('no verdict for', png); continue
        arm = {'A': m['A'], 'B': m['B']}.get(v['verdict'].strip())
        pref = {'first': 'reco1', 'second': 'hits'}.get(arm, v['verdict'].strip())
        # does the preferred arm's verdict agree with what the flip did?  kind is relative to hits
        good = ('hits' if r['kind'] == 'GAINED' else 'reco1')      # the arm that passes numu > 0.9
        cls = 'QL-main' if r['class'] == 'QL-main' else 'churn'
        C[(cls, r['kind'], 'prefers ' + pref)] += 1
        rows.append((r['sample'], r['event'], r['kind'], r['class'], pref, v['confidence'], v['object'], v['reason'][:150]))
for x in rows:
    print('%-5s %6s %-6s %-8s scan->%-7s c%s %-8s %s' % x)
for k in sorted(C):
    print('  %-8s %-6s %-16s %d' % (k + (C[k],)))

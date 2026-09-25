#!/usr/bin/env python3
"""doc sbnd_xin/124 -- the per-event markdown table of the 46 flips: class, numu reco1 -> hits, the blind-scan
preference (unblinded), the t0-nudge outcome, and the lgop arm's numu (if its products exist).

usage: doc_table.py   (paths fixed to docs/124_flips and products/d123)
"""
import csv, os, re
D = 'docs/124_flips'
def best(p):
    o = {}
    if not os.path.exists(p + '/candidates.tsv'): return None
    for r in csv.DictReader(open(p + '/candidates.tsv'), delimiter='\t'):
        k = int(r['event'])
        if k not in o or float(r['numu_score']) > float(o[k]['numu_score']): o[k] = r
    return o
scan = {}
for l in open(D + '/124_scan_unblinded.txt'):
    m = re.match(r'(mcp\dk)\s+(\d+)\s+\S+\s+\S+\s+scan->(\S+)\s+c(\d)\s+(\S+)', l)
    if m: scan[(m.group(1), m.group(2))] = (m.group(3), m.group(4), m.group(5))
nud = {}
for l in open(D + '/124_t0_nudge.txt'):
    m = re.match(r'(\d+)\s+\S+\s+A\s+(\S+) \| nudged\s+(\S+) \((\S+),.*\| B\s+(\S+)\s+(.*)$', l)
    if m: nud[m.group(1)] = 'follows hits' if 'FOLLOWS' in m.group(6) else ('stays reco1' if 'stays' in m.group(6) else m.group(6))
bmap = {(r['tag'], r['event']): ('A' if r['A'] == 'first' else 'B') for r in csv.DictReader(open(D + '/124_blind_map.tsv'), delimiter='\t')}
print('| sample | event | flip | class | νμ reco1 → hits | light gate (lgop) | blind scan prefers (conf) | scan object | t0 nudge | reco1 is row |')
print('|---|---|---|---|---|---|---|---|---|---|')
for s in ('mcp1k', 'mcp2k'):
    L = best('products/d123/%s_lgop' % s)
    for r in csv.DictReader(open(D + '/124_anatomy_%s.tsv' % s), delimiter='\t'):
        e = r['event']
        lg = '' if L is None else ('%.2f' % float(L[int(e)]['numu_score']) if int(e) in L else 'none')
        fa = lambda x: 'none' if float(x) < -900 else '%.2f' % float(x)
        sc = scan.get((s, e), ('?', '?', '?'))
        nu = nud.get(e, '—' if r['class'] == 'QL-main' else 'not run')
        print('| %s | %s | %s | %s | %s → %s | %s | %s (%s) | %s | %s | %s |' % (s, e, r['kind'].lower(), r['class'], fa(r['numu_A']), fa(r['numu_B']), lg, sc[0], sc[1], sc[2], nu, bmap[(s, e)]))

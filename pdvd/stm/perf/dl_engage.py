#!/usr/bin/env python3
"""doc pdvd/28 sec 27: DL-vertex engagement census between two PR tags.
Usage: dl_engage.py <offTag> <onTag>
Per candidate (matched by nu_index and main cluster id): main-vertex distance
between the arms, the DL arm's scoreboard route, and 'DL vertex failed' counts."""
import sys, json, glob, os, re, math, collections
off, on = sys.argv[1], sys.argv[2]
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'work')
moved = []; same = 0; nomatch = 0; routes = collections.Counter(); failed = 0; nev = 0; cands = 0
for don in sorted(glob.glob(f'{W}/*_{on}')):
    ev = os.path.basename(don)[:-len(on)-1]
    doff = f'{W}/{ev}_{off}'
    fa = glob.glob(f'{doff}/calib-pr-evt*.json'); fb = glob.glob(f'{don}/calib-pr-evt*.json')
    if not fa or not fb: continue
    nev += 1
    for lg in glob.glob(f'{don}/wct_pr_*.log'):
        failed += sum(1 for l in open(lg, errors='replace') if 'DL vertex failed' in l)
    A = {(c['nu_index'], c['main_vertex']['cluster_id'] if c.get('main_vertex') else None): c for c in json.load(open(fa[0])).get('candidates') or []}
    for c in json.load(open(fb[0])).get('candidates') or []:
        cands += 1
        sb = c.get('vertex_scoreboard') or {}
        routes[str(sb.get('prod_route', sb.get('route', '?')))] += 1
        k = (c['nu_index'], c['main_vertex']['cluster_id'] if c.get('main_vertex') else None)
        a = A.get(k)
        if a is None or not a.get('main_vertex') or not c.get('main_vertex'): nomatch += 1; continue
        d = math.dist([a['main_vertex'][q] for q in 'xyz'], [c['main_vertex'][q] for q in 'xyz'])
        if d > 0.1: moved.append((ev, k, round(d, 1)))
        else: same += 1
print(f'events {nev} candidates {cands} same-vertex {same} moved(>0.1cm) {len(moved)} unmatched {nomatch} "DL vertex failed" lines {failed}')
print('routes (DL arm scoreboard):', dict(routes))
ds = sorted(m[2] for m in moved)
if ds: print('moved distances cm: median', ds[len(ds)//2], 'p90', ds[int(0.9*len(ds))], 'max', ds[-1])
for m in sorted(moved, key=lambda m: -m[2])[:15]: print('  ', m)

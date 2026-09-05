#!/usr/bin/env python3
"""Doc pdvd/41 sec 11 -- the EXACT version of "which contact did the flip remove?".

No PCA-end proxy: for every long TGM loss (and the kept control) take the cluster's
own points and compute the BAND -- points outside the OFF volume (box + 2.5/17.5/18)
that are inside the ON volume (curved surface + 2.5/3/3).  Those are exactly the
boundary contacts the curved fiducial removes.  Then ask whether they sit at a
readout-window plane (the raw-frame edges, doc pdvd/25 M5), whether the cluster
still touches the boundary elsewhere, and where the band lives.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_band.py
"""
import json, os, re, sys
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts')
from fv_curved_longloss import event_points, surfaces, RAW_LATE, RAW_EARLY
from fv_curved_surface import (build_polygons, CurvedFV, BoxFV, inside_with_margin,
                               BOX, BOX_MARGIN, NEW_MARGIN, fit_surface, choose, VOLS)
from fv_curved_map import XW, YW, ZLO, ZHI, CATH, WALLS, wall_dist

# the emitted surface (same parameters as curved_fiducial.jsonnet)
P = {("y+","bot"):(0.00,CATH,CATH), ("y+","top"):(2.76,CATH,126.82),
     ("y-","bot"):(9.22,114.79,176.19), ("y-","top"):(2.45,CATH,XW),
     ("z-","bot"):(11.15,CATH,200.95), ("z-","top"):(3.78,CATH,XW),
     ("z+","bot"):(17.66,CATH,205.14), ("z+","top"):(11.02,CATH,164.61)}
sur = {}
for (w,v),(dc,x1,x2) in P.items():
    sur.setdefault(w,{})[v] = dict(fv=dict(dc=dc,x1=x1,x2=x2))
xy, xz = build_polygons(sur, 0.0, 0.0, 0.0)
CURVED, BOXF = CurvedFV(xy, xz), BoxFV()

R = json.load(open('/home/xqian/tmp/doc41/ab_verdicts.json'))
G = {(r['run'],r['idx'],r['cid']): r for r in R['geometry']}
want = defaultdict(dict)
for k, r in G.items():
    if r['no_t0'] or r['len_cm'] < 200: continue
    if r['cat'] in ('tgm_lost','tgm_kept'):
        want[(k[0],k[1])][k[2]] = r['cat']

rows = []
for (run, idx), cids in sorted(want.items()):
    wd = f"/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/{run}_{idx}_d41fvoff"
    try:
        E = event_points(wd)
    except Exception as ex:
        print("skip", wd, ex); continue
    ph = E['phys']
    Pt, xr, sd, cid = E['P'], E['xraw'], E['side'], E['cid']
    io = inside_with_margin(BOXF, Pt[:,0], Pt[:,1], Pt[:,2], BOX_MARGIN['x'], BOX_MARGIN['y'], BOX_MARGIN['z'])
    inn = inside_with_margin(CURVED, Pt[:,0], Pt[:,1], Pt[:,2], NEW_MARGIN['x'], NEW_MARGIN['y'], NEW_MARGIN['z'])
    late = np.where(sd < 0, RAW_LATE, -RAW_LATE)
    early = np.where(sd < 0, -RAW_EARLY, RAW_EARLY)
    d_late, d_early = np.abs(xr - late), np.abs(xr - early)
    for c, cat in cids.items():
        m = (cid == c) & ph
        if m.sum() < 5: continue
        band = m & ~io & inn
        outon = m & ~inn
        # nearest surface of each band point
        walls = Counter(); dmin = []
        if band.any():
            B = Pt[band]
            for p in B:
                s = surfaces(p); w = min(s, key=s.get); walls[w] += 1; dmin.append(s[w])
        rows.append(dict(run=run, idx=idx, cid=int(c), cat=cat, n=int(m.sum()),
                         n_band=int(band.sum()), n_out_off=int((m & ~io).sum()),
                         n_out_on=int(outon.sum()),
                         band_wall=(walls.most_common(1)[0][0] if walls else None),
                         band_dist_med=(float(np.median(dmin)) if dmin else None),
                         band_at_late=int((band & (d_late < 5)).sum()),
                         band_at_early=int((band & (d_early < 5)).sum()),
                         clus_at_late=int((m & (d_late < 5)).sum()),
                         clus_at_early=int((m & (d_early < 5)).sum()),
                         band_x_med=(float(np.median(Pt[band][:,0])) if band.any() else None)))
json.dump(rows, open('/home/xqian/tmp/doc41/band.json','w'), indent=1)

for cat in ('tgm_lost','tgm_kept'):
    A = [r for r in rows if r['cat']==cat]
    n = len(A)
    print(f"\n=== {cat}: {n} clusters > 200 cm ===")
    nb = np.array([r['n_band'] for r in A])
    print(f"  band points (lost their at-boundary status): median {np.median(nb):.0f}, "
          f"clusters with none: {int((nb==0).sum())}")
    at_l = np.array([r['band_at_late'] for r in A]); at_e = np.array([r['band_at_early'] for r in A])
    print(f"  Q1 clusters with ANY band point at the window END plane:   {int((at_l>0).sum())} ({100*(at_l>0).mean():.0f} %)")
    print(f"     clusters with ANY band point at the window START plane: {int((at_e>0).sum())} ({100*(at_e>0).mean():.0f} %)")
    cl = np.array([r['clus_at_late'] for r in A]); ce = np.array([r['clus_at_early'] for r in A])
    print(f"     clusters TOUCHING a window plane at all (any point):    end {int((cl>0).sum())}, start {int((ce>0).sum())}")
    oo = np.array([r['n_out_on'] for r in A])
    print(f"  Q2 clusters still outside the ON volume somewhere: {int((oo>0).sum())} ({100*(oo>0).mean():.0f} %)")
    print("  band points' nearest wall:", Counter(r['band_wall'] for r in A).most_common())
    bd = np.array([r['band_dist_med'] for r in A if r['band_dist_med'] is not None])
    bx = np.array([r['band_x_med'] for r in A if r['band_x_med'] is not None])
    if len(bd):
        print(f"  band point distance to its wall: median {np.median(bd):.1f} cm; |x| median {np.median(np.abs(bx)):.0f} cm")

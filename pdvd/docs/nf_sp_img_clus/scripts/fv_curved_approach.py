#!/usr/bin/env python3
"""Doc pdvd/41 sec 11 -- does imaged charge that approaches a surface REACH it?

The endpoint counterpart of sec 5's density map, and the control that decides
whether a 12 cm gap means "the fiducial boundary is wrong" or "the track does not
go there".  The anode plane is the reference surface: a hard boundary with no
space-charge inset, so its closest-approach distribution is the instrument's own
endpoint behaviour.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_approach.py
"""
import json, os, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0,'/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts')
from fv_curved_longloss import event_points
from fv_curved_map import XW, YW, ZLO, ZHI, WALLS, wall_dist

R=json.load(open('/home/xqian/tmp/doc41/ab_verdicts.json'))
G={(r['run'],r['idx'],r['cid']):r for r in R['geometry']}
ev=defaultdict(list)
for k,r in G.items():
    if not r['no_t0'] and r['len_cm']>=200:
        ev[(k[0],k[1])].append((k[2], r['cat']))

acc=defaultdict(list)
for (run,idx),cl in sorted(ev.items()):
    wd=f"/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/{run}_{idx}_d41fvoff"
    try: E=event_points(wd)
    except Exception as e: continue
    P,cid,ph=E['P'],E['cid'],E['phys']
    for c,cat in cl:
        m=(cid==c)&ph
        if m.sum()<5: continue
        Q=P[m]
        cath = np.abs(Q[:,0])<170
        for w in WALLS:
            d=wall_dist(w,Q[:,1],Q[:,2])
            for half,sel in (('anode-half',~cath),('cathode-half',cath)):
                if sel.sum()<5: continue
                dm=float(d[sel].min())
                if dm<25: acc[(w,half)].append(dm)
        da=XW-np.abs(Q[:,0])
        dm=float(da.min())
        if dm<25: acc[('anode plane','both')].append(dm)

bins=[0,1,2,3,5,8,12,18,25]
print(f"{'surface':22s} {'n':>5s}  " + " ".join(f"{bins[i]}-{bins[i+1]:<3d}" for i in range(len(bins)-1)) + "   median")
for k in sorted(acc):
    v=np.array(acc[k]); h,_=np.histogram(v,bins=bins)
    print(f"{k[0]+' '+k[1]:22s} {len(v):5d}  " + " ".join(f"{x:5d}" for x in h) + f"   {np.median(v):6.1f}")

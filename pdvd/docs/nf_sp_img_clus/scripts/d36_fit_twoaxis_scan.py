#!/usr/bin/env python3
"""Two-axis grade over the 120-event manifest, per arm:
   COVERAGE  - what fraction of a cluster's own 3-D charge lies within 2 cm of its fit
   SUPPORT   - what fraction of the fit points lie >2 / >10 cm from any 3-D charge
Restricted to clusters that have a fit in at least one arm and >=50 charge points.
"""
import glob, json, os, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # <repo>/pdvd
W = os.path.join(PDVD, 'work')
ARMS=sys.argv[1].split(','); OUT=sys.argv[2]
def one(base):
    run,idx=base.rsplit('_',1); P=None; pc=None; out=[]
    fits={}
    for tag in ARMS:
        p=f'{W}/{base}_{tag}/mabc-pr.zip'
        if not os.path.exists(p): return None
        z=zipfile.ZipFile(p)
        ft=json.loads(z.read('data/0/0-stm_fit-global.json'))
        if P is None:
            cl=json.loads(z.read('data/0/0-clustering-global.json'))
            P=np.stack([cl['x'],cl['y'],cl['z']],1).astype(float)
            pc=np.asarray(cl['cluster_id'],dtype=int)
        F=np.stack([ft['x'],ft['y'],ft['z']],1).astype(float)
        fc=np.asarray(ft['cluster_id'],dtype=int)
        fits[tag]={int(c):np.unique(np.round(F[fc==c],2),axis=0) for c in np.unique(fc)}
    T=cKDTree(P)
    cids=set()
    for t in ARMS: cids|=set(fits[t])
    for cid in sorted(cids):
        C=P[pc==cid]
        if len(C)<50: continue
        row=[run,idx,cid,len(C)]
        for t in ARMS:
            X=fits[t].get(cid)
            if X is None or len(X)==0: row+=[0,0.0,0.0,0.0]; continue
            dcov,_=cKDTree(X).query(C); d,_=T.query(X)
            row+=[len(X),float((dcov<2).mean()),float((d>2).mean()),float((d>10).mean())]
        out.append(row)
    return out
bases=sorted({os.path.basename(d)[:-(len(ARMS[0])+1)] for d in glob.glob(f'{W}/*_{ARMS[0]}')})
with Pool(16) as pool, open(OUT,'w') as f:
    h=['run','idx','cid','nq']
    for t in ARMS: h+=[t+'_n',t+'_cov',t+'_f2',t+'_f10']
    f.write('\t'.join(h)+'\n')
    for rows in pool.imap_unordered(one,bases):
        if rows:
            for r in rows: f.write('\t'.join(str(x) for x in r)+'\n')

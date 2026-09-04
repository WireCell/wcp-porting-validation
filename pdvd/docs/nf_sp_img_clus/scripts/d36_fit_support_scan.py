#!/usr/bin/env python3
"""Global per-cluster fit-vs-charge scan over the 120-event PR manifest.

For every (event, cluster) with a persisted stm_fit in either arm, report the
deduped fit point count, approximate diameter, and the fraction of fit points
farther than 2 / 10 cm from ANY 3-D charge point of that event (Bee
clustering-global), in each arm.  Output: TSV.
"""
import glob, json, os, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # <repo>/pdvd
W = os.path.join(PDVD, 'work')
ARMS=sys.argv[1].split(',')
OUT=sys.argv[2]

def diam(X):
    if len(X)<2: return 0.0
    b=X[np.argmax(np.linalg.norm(X-X[0],axis=1))]
    c=X[np.argmax(np.linalg.norm(X-b,axis=1))]
    return float(np.linalg.norm(b-c))

def one(base):
    run,idx=base.rsplit('_',1)
    res={}
    for tag in ARMS:
        p=f'{W}/{base}_{tag}/mabc-pr.zip'
        if not os.path.exists(p): return None
        z=zipfile.ZipFile(p)
        ft=json.loads(z.read('data/0/0-stm_fit-global.json'))
        cl=json.loads(z.read('data/0/0-clustering-global.json'))
        P=np.stack([cl['x'],cl['y'],cl['z']],1).astype(float)
        T=cKDTree(P)
        F=np.stack([ft['x'],ft['y'],ft['z']],1).astype(float)
        fc=np.asarray(ft['cluster_id'],dtype=int)
        for cid in np.unique(fc):
            X=np.unique(np.round(F[fc==cid],2),axis=0)
            d,_=T.query(X)
            res.setdefault(int(cid),{})[tag]=(len(X),diam(X),float((d>2).mean()),float((d>10).mean()),float(d.max()))
    rows=[]
    for cid,per in sorted(res.items()):
        r=[run,idx,cid]
        for tag in ARMS:
            v=per.get(tag)
            r+= list(v) if v else [0,0.0,0.0,0.0,0.0]
        rows.append(r)
    return rows

bases=sorted({os.path.basename(d)[:-(len(ARMS[0])+1)] for d in glob.glob(f'{W}/*_{ARMS[0]}')})
print(len(bases),'events',file=sys.stderr)
with Pool(16) as pool, open(OUT,'w') as f:
    hdr=['run','idx','cid']
    for t in ARMS: hdr+=[f'{t}_n',f'{t}_diam',f'{t}_f2',f'{t}_f10',f'{t}_max']
    f.write('\t'.join(hdr)+'\n')
    for rows in pool.imap_unordered(one,bases):
        if not rows: continue
        for r in rows: f.write('\t'.join(str(x) for x in r)+'\n')
        f.flush()
print('done',file=sys.stderr)

#!/usr/bin/env python3
"""What does the metric ADD to (and REMOVE from) each fitted trajectory, and is
the added part on charge?

For each (event, cluster): take the deduped stm_fit point sets of arm A and arm
B.  "added"   = B points farther than 2 cm from every A point.
   "removed" = A points farther than 2 cm from every B point.
Each set is then measured against the event's 3-D charge (clustering-global).
Output TSV.
"""
import glob, json, os, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # <repo>/pdvd
W = os.path.join(PDVD, 'work')
A,B=sys.argv[1],sys.argv[2]; OUT=sys.argv[3]

def one(base):
    run,idx=base.rsplit('_',1)
    sets={}
    P=None
    for tag in (A,B):
        p=f'{W}/{base}_{tag}/mabc-pr.zip'
        if not os.path.exists(p): return None
        z=zipfile.ZipFile(p)
        ft=json.loads(z.read('data/0/0-stm_fit-global.json'))
        if P is None:
            cl=json.loads(z.read('data/0/0-clustering-global.json'))
            P=np.stack([cl['x'],cl['y'],cl['z']],1).astype(float)
        F=np.stack([ft['x'],ft['y'],ft['z']],1).astype(float)
        fc=np.asarray(ft['cluster_id'],dtype=int)
        sets[tag]={int(c):np.unique(np.round(F[fc==c],2),axis=0) for c in np.unique(fc)}
    T=cKDTree(P)
    rows=[]
    for cid in sorted(set(sets[A])|set(sets[B])):
        XA=sets[A].get(cid,np.zeros((0,3))); XB=sets[B].get(cid,np.zeros((0,3)))
        def novel(X,Y):
            if len(X)==0: return X
            if len(Y)==0: return X
            d,_=cKDTree(Y).query(X); return X[d>2.0]
        add=novel(XB,XA); rem=novel(XA,XB)
        def stat(X):
            if len(X)==0: return (0,0.0,0.0,0.0)
            d,_=T.query(X)
            return (len(X),float((d>2).mean()),float((d>10).mean()),float(d.max()))
        rows.append([run,idx,cid,len(XA),len(XB)]+list(stat(add))+list(stat(rem)))
    return rows

bases=sorted({os.path.basename(d)[:-(len(A)+1)] for d in glob.glob(f'{W}/*_{A}')})
with Pool(16) as pool, open(OUT,'w') as f:
    f.write('\t'.join(['run','idx','cid','nA','nB','add_n','add_f2','add_f10','add_max',
                       'rem_n','rem_f2','rem_f10','rem_max'])+'\n')
    for rows in pool.imap_unordered(one,bases):
        if not rows: continue
        for r in rows: f.write('\t'.join(str(x) for x in r)+'\n')
print('done',file=sys.stderr)

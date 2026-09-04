#!/usr/bin/env python3
"""Are PR-stage cluster ids the same object in both arms?

Charge point sets are identical between arms (same imaging input), so match
arm-A clusters to arm-B clusters by point overlap and check whether the id is
preserved.  Reports per-event id churn.
"""
import glob, json, os, sys, zipfile
import numpy as np
from multiprocessing import Pool
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # <repo>/pdvd
W = os.path.join(PDVD, 'work')
A,B=sys.argv[1],sys.argv[2]
MIN=int(sys.argv[3]) if len(sys.argv)>3 else 50

def load(p):
    z=zipfile.ZipFile(p)
    j=json.loads(z.read('data/0/0-clustering-global.json'))
    X=np.round(np.stack([j['x'],j['y'],j['z']],1).astype(float),3)
    c=np.asarray(j['cluster_id'],dtype=int)
    k=np.lexsort((X[:,2],X[:,1],X[:,0]))
    return X[k],c[k]

def one(base):
    try:
        Xa,ca=load(f'{W}/{base}_{A}/mabc-pr.zip'); Xb,cb=load(f'{W}/{base}_{B}/mabc-pr.zip')
    except Exception: return None
    if not np.array_equal(Xa,Xb): return (base,None)
    out=[]
    for cid in np.unique(ca):
        m=ca==cid
        if m.sum()<MIN: continue
        vals,cnt=np.unique(cb[m],return_counts=True)
        best=vals[np.argmax(cnt)]; frac=cnt.max()/m.sum()
        out.append((int(cid),int(best),float(frac),int(m.sum())))
    return (base,out)

bases=sorted({os.path.basename(d)[:-(len(A)+1)] for d in glob.glob(f'{W}/*_{A}')})
tot=same=0; bad=[]; nofr=[]
with Pool(16) as pool:
    for res in pool.imap_unordered(one,bases):
        if res is None: continue
        base,out=res
        if out is None: nofr.append(base); continue
        for cid,best,frac,n in out:
            tot+=1
            if cid==best and frac>0.90: same+=1
            else: bad.append((base,cid,best,frac,n))
print('events where the charge point set differs between arms:',len(nofr),nofr[:5])
print('clusters >= %d pts: %d;  id preserved with >90%% overlap: %d (%.2f%%)'%(MIN,tot,same,100*same/tot))
print('mismatches: %d'%len(bad))
for b in sorted(bad,key=lambda x:-x[4])[:20]:
    print('   %-12s cluster %4d -> best match %4d  overlap %.2f  (%d pts)'%b)

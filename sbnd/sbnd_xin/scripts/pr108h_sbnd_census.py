import sys, numpy as np
sys.path.insert(0,'/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/scripts')
from pr108_dqdx_cond import parse, pick, build
from pr108_fit_point_compare import load, junctions
W='/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/'
CASES=[(46363,'work-pr108-dqdump-on-nuecc48','/home/xqian/tmp/pr108_sbnd_on_46363.dump','ON'),
       (46363,'work-pr108-dqdump-off-nuecc48','/home/xqian/tmp/pr108_sbnd_off_46363.dump','OFF'),
       (360535,'work-pr108-dqdump2-on-nuecc48','/home/xqian/tmp/pr108_sbnd_on_360535.dump','ON'),
       (360535,'work-pr108-dqdump2-off-nuecc48','/home/xqian/tmp/pr108_sbnd_off_360535.dump','OFF')]
R=1.5
for ev,arm,dump,lab in CASES:
    d=load(f'{W}{arm}/pr_evt{ev}/tracking-pr.root','wct',False); J=junctions(d)
    C=parse(dump); rec=[]
    for j in J:
        c=pick(C,j,2.0)
        if c is None: continue
        A,b,x,P=build(C[c]); dist=np.linalg.norm(P-j,axis=1); sel=dist<=R
        if sel.sum()<2: continue
        rows=C[c]['rows']; nreg=sum(1 for k in np.where(sel)[0] if any(rows[k]['reg']))
        u=sel.astype(float); y=np.linalg.solve(A,u); xs=np.linalg.solve(A,b); q=xs[sel].sum()
        Cm=np.outer(y,xs)*A
        def kap(mask):
            M=(Cm+Cm.T)*mask
            var=(np.triu(M,1)**2).sum()+((np.diag(Cm)*np.diag(mask))**2).sum()
            return np.sqrt(var)/abs(q)
        loc=(sel[:,None]|sel[None,:]).astype(float)
        rec.append((A.shape[0],int(sel.sum()),nreg,q,kap(np.ones_like(A)),kap(loc)))
    kg=np.array([r[4] for r in rec]); kl=np.array([r[5] for r in rec])
    clean=np.array([r[2]==0 for r in rec])
    print(f'[{ev} {lab}] {len(rec)} junctions: kappa_glob med {np.median(kg):.1f} [{kg.min():.1f},{kg.max():.1f}] ; '
          f'kappa_loc med {np.median(kl):.1f} [{kl.min():.1f},{kl.max():.1f}] ; '
          f'reg-free {clean.sum()}/{len(rec)} -> kappa_loc med {np.median(kl[clean]) if clean.any() else float("nan"):.1f} '
          f'[{kl[clean].min() if clean.any() else float("nan"):.1f},{kl[clean].max() if clean.any() else float("nan"):.1f}] ; '
          f'flagged -> kappa_loc med {np.median(kl[~clean]) if (~clean).any() else float("nan"):.1f}')

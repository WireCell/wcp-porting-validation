import sys, numpy as np
sys.path.insert(0,'/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/scripts')
from pr108_dqdx_cond import parse, pick, build
S='/nfs/data/1/xqian/toolkit-dev/toolkit/qlport/scripts/sweep/'
def kappas(path,J,R=1.5):
    C=parse(path); c=pick(C,np.array(J),2.0)
    if c is None: return None
    A,b,x,P=build(C[c]); J=np.array(J)
    d=np.linalg.norm(P-J,axis=1); sel=d<=R
    rows=C[c]['rows']; nreg=sum(1 for k in np.where(sel)[0] if any(rows[k]['reg']))
    u=sel.astype(float); y=np.linalg.solve(A,u); xs=np.linalg.solve(A,b); q=xs[sel].sum()
    Cm=np.outer(y,xs)*A
    def kap(mask):
        M=(Cm+Cm.T)*mask
        var=(np.triu(M,1)**2).sum()+((np.diag(Cm)*np.diag(mask))**2).sum()
        return np.sqrt(var)/abs(q)
    return dict(n=A.shape[0],npt=int(sel.sum()),reg=nreg,kg=kap(np.ones_like(A)),
                kl=kap((sel[:,None]|sel[None,:]).astype(float)))
JS=[(6505,1,'pr108h','J0',(108.84,48.26,1029.27)),(6505,1,'pr108h','J1',(160.34,71.59,966.52)),
    (6532,6,'pr108h','J0',(151.19,-105.94,890.20)),(6650,16,'pr108h','J0',(185.21,-8.40,453.37)),
    (6650,16,'pr108h','J1',(186.29,-11.95,457.66)),(6805,22,'pr108h','J0',(47.93,69.44,430.40)),
    (6528,4,'pr108g','J0',(223.85,42.05,23.05)),(6528,4,'pr108g','J1',(223.43,42.63,22.22)),
    (6806,23,'pr108g','J0',(172.58,-96.35,732.97))]
print(f"{'evt/J':9s} {'arm':8s} {'n3d':>4s} {'npt':>3s} {'reg':>5s} {'kappa_glob':>10s} {'kappa_loc':>9s}")
for ev,idx,g,jn,J in JS:
    for side,arm in (('wcp','on'),('wct','on'),('wcp','off'),('wct','off')):
        f=f'{S}{g}_{side}_{arm}/dqdx_{ev if side=="wcp" else idx}.dump'
        r=kappas(f,J)
        if r: print(f"{str(ev)+'/'+jn:9s} {side+'-'+arm:8s} {r['n']:4d} {r['npt']:3d} {str(r['reg'])+'/'+str(r['npt']):>5s} {r['kg']:10.1f} {r['kl']:9.1f}")

#!/usr/bin/env python3
"""Two non-lattice candidates for the wave, tested on the same residual.

A. TRAJECTORY WOBBLE.  Doc pdhd/03 sec 9 item 2 saw a fit zig-zagging +-3 cm
   in x across a straight track.  If the wave is the trajectory, the residual
   must track the local transverse excursion of the fit about its own local
   straight line (window +- 3 cm).  Reported as the correlation of the dQ/dx
   residual with the signed and with the |excursion|, and with its curvature.

B. PR SEGMENT BOUNDARIES.  The chain is stitched from PR segments; a fixed
   fitter scale would put the wave's extrema at the joins.  Reported as the
   mean residual against distance to the nearest segment vertex.

Null for both: per-item circular shift of the residual.
"""
import json, glob, sys
import os
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")

from d24_fold import collect, arr, longest_run, RRMIN, MINPTS, JUMP

PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
rng=np.random.default_rng(20260912)
NNULL=500
HALF=5              # +- 5 points = +- 3 cm local straight-line window

def wobble_of(x,y,z,half=HALF):
    """signed transverse excursion of each point from the local chord."""
    n=len(x); w=np.full(n,np.nan)
    Pm=np.c_[x,y,z]
    for i in range(half,n-half):
        A=Pm[i-half]; B=Pm[i+half]
        d=B-A; L=np.linalg.norm(d)
        if L<1e-6: continue
        d=d/L
        v=Pm[i]-A
        perp=v-np.dot(v,d)*d
        w[i]=np.linalg.norm(perp)
    return w

def main():
    # re-collect with the geometry attached
    items=collect(PREP)
    bykey={it["key"]:it for it in items}
    rows=[]
    for f in sorted(glob.glob(PREP+"/smprep-*.json")):
        d=json.load(open(f)); key=d["event"]+"/"+str(d["cluster_id"])
        if key not in bykey: continue
        m=d["muon"]
        L,q,rr=arr(m["L"]),arr(m["q"]),arr(m["rr"])
        x,y,z=(arr(m[k]) for k in "xyz")
        P={k:arr(m[k]) for k in ("pw","pu","pv","pt")}
        ok=np.isfinite(q)&(q>0)&(rr>RRMIN)
        for k in P: ok&=np.isfinite(P[k])
        jump=np.zeros(len(L),bool)
        for k in ("pw","pu","pv"): jump[1:]|=np.abs(np.diff(P[k]))>JUMP
        ok&=~jump
        i0,i1=longest_run(ok)
        if i1-i0<MINPTS: continue
        s=slice(i0,i1)
        it=bykey[key]
        w=wobble_of(x[s],y[s],z[s])
        # vertices of the PR segments, for the boundary test
        vx=np.array(d["pf"]["vtx"]["x"],float); vy=np.array(d["pf"]["vtx"]["y"],float)
        vz=np.array(d["pf"]["vtx"]["z"],float)
        dv=np.full(i1-i0,np.inf)
        if vx.size:
            Pm=np.c_[x[s],y[s],z[s]]
            V=np.c_[vx,vy,vz]
            dv=np.sqrt(((Pm[:,None,:]-V[None,:,:])**2).sum(-1)).min(1)
        rows.append(dict(key=key, r=it["r"], w=w, dv=dv, L=it["L"]))
    print("items: %d"%len(rows))

    def corr(field, use_abs=False, dfun=None):
        num=[]; 
        for it in rows:
            a=it[field].copy()
            if dfun is not None: a=dfun(a)
            if use_abs: a=np.abs(a)
            g=np.isfinite(a)&np.isfinite(it["r"])
            if g.sum()<30: continue
            aa=a[g]-a[g].mean(); rr_=it["r"][g]-it["r"][g].mean()
            num.append((aa,rr_))
        obs=np.sum([np.sum(a*b) for a,b in num])/np.sqrt(
            np.sum([np.sum(a*a) for a,b in num])*np.sum([np.sum(b*b) for a,b in num]))
        nl=[]
        for _ in range(NNULL):
            tot=0.0
            for (a,b) in num:
                tot+=np.sum(a*np.roll(b,rng.integers(1,b.size)))
            nl.append(tot/np.sqrt(np.sum([np.sum(a*a) for a,b in num])*np.sum([np.sum(b*b) for a,b in num])))
        nl=np.array(nl)
        return obs, nl.mean(), nl.std(), ((np.abs(nl)>=abs(obs)).sum()+1)/(NNULL+1)

    out=[]
    for nm,field,ab,df in (("wobble  |transverse excursion|","w",True,None),
                           ("wobble  d|excursion|/dL","w",False,lambda a: np.r_[np.nan,np.diff(np.abs(a))]),
                           ("segment  distance to nearest vertex","dv",False,None)):
        o,m,s,p=corr(field,ab,df)
        out.append("%-38s r=%+.4f   null %+.4f +- %.4f   p(2-sided)=%.4f"%(nm,o,m,s,p))
    txt="\n".join(out); print(txt); open(os.path.join(FIGS, "24_wobble.txt"), "w").write(txt+"\n")

main()

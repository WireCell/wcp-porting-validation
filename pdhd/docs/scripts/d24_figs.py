#!/usr/bin/env python3
"""The two figures the write-up needs."""
import os
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from d24_fold import collect
PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
items=collect(PREP)

def profile(sel,key,nb=12):
    ph=np.concatenate([np.mod(it[key],1.0) for it in sel])
    r=np.concatenate([it["r"] for it in sel])
    idx=np.clip((ph*nb).astype(int),0,nb-1)
    x=(np.arange(nb)+0.5)/nb; m=np.zeros(nb); s=np.zeros(nb)
    for b in range(nb):
        g=idx==b
        if g.sum(): m[b]=r[g].mean(); s[b]=r[g].std()/np.sqrt(g.sum())
    return x,m,s

W=[it for it in items if it["pw_spc"]>=6 and it["pt_spc"]<6]
T=[it for it in items if it["pt_spc"]>=6 and it["pw_spc"]<6]
fig,ax=plt.subplots(1,4,figsize=(16,3.8),sharey=True)
for a,(sel,k,t) in zip(ax,[(W,"pw","W wire, RESOLVED (%d items)"%len(W)),
                           (W,"pt","time slice on the same items (control)"),
                           (T,"pt","time slice, RESOLVED (%d items)"%len(T)),
                           (T,"pw","W wire on the same items (control)")]):
    x,m,s=profile(sel,k)
    a.errorbar(x,m,yerr=s,fmt="o-",ms=4,color=("tab:blue" if "RESOLVED" in t else "tab:grey"))
    a.axhline(0,color="k",lw=0.6); a.set_title(t,fontsize=9); a.grid(alpha=0.3)
    a.set_xlabel("position within one cell")
ax[0].set_ylabel("dQ/dx residual [item sigma]")
fig.suptitle("dQ/dx is biased by where the trajectory point sits inside a 2-D cell -- "
             "on the plane the track resolves, and only there", fontsize=10)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(os.path.join(FIGS, "24_fig1_subcell_fold.png"),dpi=120)

# fig 2: correlation length vs cm per cell
MAXLAG=40
def lam_of(sel):
    acc=np.zeros(MAXLAG+1); cnt=np.zeros(MAXLAG+1)
    for it in sel:
        r=it["r"]-it["r"].mean(); v=np.var(r); n=r.size
        if v<=0: continue
        for k in range(MAXLAG+1):
            if n-k<20: break
            acc[k]+=np.mean(r[:n-k]*r[k:])/v*(n-k); cnt[k]+=(n-k)
    a=acc/np.maximum(cnt,1)
    b=np.where(a<1/np.e)[0]
    if b.size and b[0]>0: k=b[0]; return a, k-1+(a[k-1]-1/np.e)/(a[k-1]-a[k])
    return a, float(MAXLAG)
fig,ax=plt.subplots(1,2,figsize=(12,4))
for key,c,lab in (("pw_cm","tab:blue","collection wire"),("pt_cm","tab:orange","drift time slice")):
    xs=[];ys=[]
    edges=[(0,0.8),(0.8,1.5),(1.5,3.0),(3.0,8.0),(8.0,1e9)]
    for lo,hi in edges:
        sel=[it for it in items if lo<=it[key]<hi]
        if len(sel)<8: continue
        a,lam=lam_of(sel)
        xs.append(np.median([it[key] for it in sel])); ys.append(lam*0.6)
    ax[0].plot(xs,ys,"o-",color=c,label=lab)
xx=np.logspace(np.log10(0.4),np.log10(30),50)
ax[0].plot(xx,xx,"--",color="grey",label="if the scale were geometric (1 cell)")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("cm per cell (track geometry)"); ax[0].set_ylabel("dQ/dx correlation length [cm]")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
ax[0].set_title("the wave's scale does NOT follow the lattice", fontsize=10)
a,lam=lam_of(items)
ax[1].plot(np.arange(MAXLAG+1)*0.6, a, "o-", ms=3)
ax[1].axhline(1/np.e,color="grey",ls="--",lw=0.8); ax[1].axhline(0,color="k",lw=0.6)
ax[1].set_xlabel("lag [cm]"); ax[1].set_ylabel("ACF of the detrended residual")
ax[1].set_title("pooled: ACF(1)=%.3f, 1/e at %.2f cm (%.1f points)"%(a[1],lam*0.6,lam),fontsize=10)
ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "24_fig2_correlation_length.png"),dpi=120)
print("wrote", FIGS, "24_fig1_subcell_fold.png 24_fig2_correlation_length.png")

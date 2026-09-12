#!/usr/bin/env python3
"""Is the wave's correlation length a fixed fitter scale or a geometric one?

A lattice/geometry origin makes the correlation length scale with the track's
cm-per-cell; a regulariser scale makes it the same number of TRAJECTORY POINTS
on every stratum.  ACF is therefore reported per stratum in BOTH units.
"""
import os
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")

from d24_fold import collect
PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
items=collect(PREP)
MAXLAG=40

def acf_of(sel):
    acc=np.zeros(MAXLAG+1); cnt=np.zeros(MAXLAG+1)
    for it in sel:
        r=it["r"]-it["r"].mean(); v=np.var(r); n=r.size
        if v<=0: continue
        for k in range(MAXLAG+1):
            if n-k<20: break
            acc[k]+=np.mean(r[:n-k]*r[k:])/v*(n-k); cnt[k]+=(n-k)
    a=acc/np.maximum(cnt,1)
    below=np.where(a<1/np.e)[0]
    # linear interpolation of the 1/e crossing, in points
    if below.size:
        k=below[0]
        if k>0 and a[k-1]!=a[k]:
            lam=k-1+(a[k-1]-1/np.e)/(a[k-1]-a[k])
        else: lam=float(k)
    else: lam=float(MAXLAG)
    return a, lam

out=[]
def bucket(name, sel, cmcell):
    if len(sel)<8: return
    a,lam=acf_of(sel)
    med=np.median(cmcell)
    out.append("%-42s items=%3d  cm/cell med %6.2f   ACF1 %.3f   1/e at %.2f pts = %.2f cm"
               % (name,len(sel),med,a[1],lam,lam*0.6))

# stratify on the collection-wire cell
cw=np.array([it["pw_cm"] for it in items])
for lo,hi in ((0,0.8),(0.8,1.5),(1.5,3.0),(3.0,8.0),(8.0,1e9)):
    sel=[it for it in items if lo<=it["pw_cm"]<hi]
    bucket("cm per W wire in [%g,%g)"%(lo,hi), sel, [it["pw_cm"] for it in sel])
out.append("")
ct=np.array([it["pt_cm"] for it in items])
for lo,hi in ((0,0.5),(0.5,1.0),(1.0,2.0),(2.0,6.0),(6.0,1e9)):
    sel=[it for it in items if lo<=it["pt_cm"]<hi]
    bucket("cm per time slice in [%g,%g)"%(lo,hi), sel, [it["pt_cm"] for it in sel])
txt="\n".join(out); print(txt); open(os.path.join(FIGS, "24_acfstrat.txt"), "w").write(txt+"\n")

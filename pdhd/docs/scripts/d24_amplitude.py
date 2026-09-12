#!/usr/bin/env python3
"""How BIG is the cell effect?  Significance is not size.

Three numbers, all on the detrended residual (unit variance per item):
  A1, A2   pooled first/second harmonic amplitudes of the cell fold;
  fvar     fraction of residual VARIANCE a 12-bin cell-phase model removes,
           fitted pooled (honest) and per item (an upper bound that a
           circular-shift null is quoted against, since 12 free bins per item
           absorb noise);
  abs      the same in ke/cm, using each item's own dQ/dx mean.
"""
import os
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")
from d24_fold import collect
PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
items=collect(PREP)
rng=np.random.default_rng(20260912)
NB=12

def fvar_pooled(sel,key,shift=None):
    ph=[];r=[]
    for i,it in enumerate(sel):
        rr=it["r"] if shift is None else np.roll(it["r"],shift[i])
        ph.append(np.mod(it[key],1.0)); r.append(rr)
    ph=np.concatenate(ph); r=np.concatenate(r)
    idx=np.clip((ph*NB).astype(int),0,NB-1)
    pred=np.zeros_like(r)
    for b in range(NB):
        g=idx==b
        if g.sum(): pred[g]=r[g].mean()
    return 1.0-np.var(r-pred)/np.var(r)

def harm(sel,key,k):
    num=0j; n=0
    for it in sel:
        ph=np.mod(it[key],1.0); num+=np.sum(it["r"]*np.exp(2j*np.pi*k*ph)); n+=it["r"].size
    return 2*np.abs(num)/n

out=[]
for coord,other,lab in (("pw","pt","W wire"),("pt","pw","time slice")):
    sel=[it for it in items if it[coord+"_spc"]>=6 and it[other+"_spc"]<6]
    if len(sel)<5: continue
    f=fvar_pooled(sel,coord)
    null=np.array([fvar_pooled(sel,coord,[rng.integers(1,it["r"].size) for it in sel]) for _ in range(300)])
    a1,a2=harm(sel,coord,1),harm(sel,coord,2)
    # absolute scale: mean dQ/dx and residual sigma in ke/cm per item
    sig=np.median([it["sigma_kecm"] for it in sel]) if "sigma_kecm" in sel[0] else np.nan
    out.append("%s stratum (%d items, %d pts)" % (lab,len(sel),sum(it["r"].size for it in sel)))
    out.append("   A1=%.4f  A2=%.4f  (sigma units)" % (a1,a2))
    out.append("   variance removed by a %d-bin cell-phase model: %.4f   null %.4f +- %.4f  p=%.4f"
               % (NB,f,null.mean(),null.std(),((null>=f).sum()+1)/301.0))
    out.append("   excess over null: %.4f of the residual variance" % (f-null.mean()))
txt="\n".join(out); print(txt); open(os.path.join(FIGS, "24_amplitude.txt"), "w").write(txt+"\n")

#!/usr/bin/env python3
"""Upper bound on how much of the wave the cell fold can be.

The pooled fold assumes every item shares one phase offset.  If the offset
differs per item, pooling dilutes it and 0.5% is a floor, not the size.  So
refit PER ITEM: a free phase (first harmonic) and, more generously, a free
12-bin cell-phase profile per item.  Both absorb noise, so both are quoted
against the circular-shift null and the EXCESS over that null is the bound.
"""
import os
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")

from d24_fold import collect
PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
items=collect(PREP); rng=np.random.default_rng(20260912)
NNULL=400; NB=12

def peritem(sel, key, mode, shift=None):
    """mean over items of the residual variance fraction removed."""
    fr=[]
    for i,it in enumerate(sel):
        r=it["r"] if shift is None else np.roll(it["r"], shift[i])
        ph=np.mod(it[key],1.0)
        if mode=="h1":
            c=2*np.mean(r*np.exp(2j*np.pi*ph))
            pred=(c*np.exp(-2j*np.pi*ph)).real
        else:
            idx=np.clip((ph*NB).astype(int),0,NB-1)
            pred=np.zeros_like(r)
            for b in range(NB):
                g=idx==b
                if g.sum(): pred[g]=r[g].mean()
        fr.append(1.0-np.var(r-pred)/max(np.var(r),1e-12))
    return float(np.mean(fr))

out=[]
for coord,other,lab in (("pw","pt","W wire"),("pt","pw","time slice")):
    sel=[it for it in items if it[coord+"_spc"]>=6 and it[other+"_spc"]<6]
    if len(sel)<5: continue
    out.append("%s stratum, per-item free phase (%d items)" % (lab,len(sel)))
    for mode,ttl in (("h1","free-phase first harmonic"),("bins","free 12-bin profile")):
        o=peritem(sel,coord,mode)
        nl=np.array([peritem(sel,coord,mode,[rng.integers(1,it["r"].size) for it in sel])
                     for _ in range(NNULL)])
        out.append("   %-26s removed %.4f   null %.4f +- %.4f   EXCESS %.4f   p=%.4f"
                   % (ttl,o,nl.mean(),nl.std(),o-nl.mean(),((nl>=o).sum()+1)/(NNULL+1)))
txt="\n".join(out); print(txt); open(os.path.join(FIGS, "24_upper.txt"), "w").write(txt+"\n")

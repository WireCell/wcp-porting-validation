#!/usr/bin/env python3
"""How much of the detrended dQ/dx residual is coherent (a wave) at all?

Landau fluctuation, delta rays and imaging noise are point-to-point white; the
"wave" is the part that survives smoothing.  With r normalised to unit variance
and white noise uncorrelated at lag 1, ACF(1) is the coherent variance share,
and the lag at which ACF falls to 1/e is the wave's correlation length.  Every
"explains X% of the variance" number in this analysis is quoted against BOTH
denominators, because the white part was never something a mechanism had to
explain.
"""
import os
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")

from d24_fold import collect
PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
items=collect(PREP)
MAXLAG=60
acc=np.zeros(MAXLAG+1); cnt=np.zeros(MAXLAG+1)
steps=[]
for it in items:
    r=it["r"]; r=r-r.mean(); v=np.var(r)
    if v<=0: continue
    n=r.size
    steps.append(np.median(np.diff(it["L"])))
    for k in range(MAXLAG+1):
        if n-k < 20: break
        acc[k]+= np.mean(r[:n-k]*r[k:])/v * (n-k); cnt[k]+= (n-k)
acf=acc/np.maximum(cnt,1)
step=float(np.median(steps))
out=["median trajectory point spacing: %.3f cm" % step,
     "pooled ACF of the detrended residual (lag in points / cm):"]
for k in (1,2,3,4,5,8,10,15,20,30,50):
    if k<=MAXLAG: out.append("   lag %3d (%6.2f cm)   ACF %+.4f" % (k,k*step,acf[k]))
coh=acf[1]
ie=np.argmax(acf<1/np.e) if (acf<1/np.e).any() else MAXLAG
out.append("")
out.append("coherent variance share (= ACF at lag 1): %.3f" % coh)
out.append("correlation length (ACF -> 1/e): %d points = %.1f cm" % (ie, ie*step))
out.append("")
out.append("=> a mechanism removing f of the TOTAL residual variance removes")
out.append("   f/%.3f = %.1f x that share of the COHERENT (wave) part." % (coh,1.0/coh))
for nm,f in (("sub-cell fold, W wire (pooled)",0.0052),
             ("sub-cell fold, time slice (pooled)",0.0036),
             ("trajectory wobble (r=-0.1394)",0.1394**2)):
    out.append("   %-36s total %.4f  -> coherent %.3f" % (nm,f,f/coh))
txt="\n".join(out); print(txt); open(os.path.join(FIGS, "24_coherent.txt"), "w").write(txt+"\n")

#!/usr/bin/env python3
"""Does dQ/dx depend on WHERE INSIDE A WIRE CELL the trajectory point sits?

The most direct form of the wire-crossing hypothesis needs no period estimate:
if charge is modulated by wire crossings, the detrended residual folded on the
fractional part of the wire coordinate must show a modulation.  Folding is done
per plane (pu, pv, pw) and, as controls, on the time slice (pt) and on the
fractional part of L/pitch_equivalent (a coordinate with no readout meaning).

Statistic: the amplitude of the first harmonic of the fold (|<r e^{2 pi i phi}>|
* 2), compared with a permutation null that keeps the residuals and the phases
but destroys their pairing.  Pooled over the population AND per stratum of
samples-per-cell, because a cell the sampling cannot resolve cannot show a fold.
"""
import json, glob, sys
import os
import numpy as np

PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
RRMIN, MINPTS, JUMP = 30.0, 60, 5.0
NPERM = 1000

def arr(v): return np.array([np.nan if t is None else t for t in v], float)

def longest_run(mask):
    best=(0,0,0); i=0
    while i < len(mask):
        if mask[i]:
            j=i
            while j < len(mask) and mask[j]: j+=1
            if j-i > best[0]: best=(j-i,i,j)
            i=j
        else: i+=1
    return best[1], best[2]

def collect(prepdir):
    ref = json.load(open(prepdir + "/dqdx_ref_pdhd.json"))
    rg = np.arange(len(ref["muon"]))*0.25; refmu = np.array(ref["muon"], float)
    items=[]
    for f in sorted(glob.glob(prepdir+"/smprep-*.json")):
        d=json.load(open(f)); m=d["muon"]
        L,q,rr = arr(m["L"]), arr(m["q"]), arr(m["rr"])
        P={k:arr(m[k]) for k in ("pw","pu","pv","pt")}
        ok=np.isfinite(q)&(q>0)&(rr>RRMIN)
        for k in P: ok &= np.isfinite(P[k])
        jump=np.zeros(len(L),bool)
        for k in ("pw","pu","pv"): jump[1:] |= np.abs(np.diff(P[k]))>JUMP
        ok &= ~jump
        i0,i1=longest_run(ok)
        if i1-i0 < MINPTS: continue
        s=slice(i0,i1)
        Ls,qs,rrs=L[s],q[s],rr[s]
        r=qs/np.interp(rrs,rg,refmu)
        x=Ls-Ls[0]
        r=r-np.polyval(np.polyfit(x,r,2),x)
        r=r/(np.std(r)+1e-12)
        it=dict(key=d["event"]+"/"+str(d["cluster_id"]), r=r, L=x, n=i1-i0)
        for k in ("pw","pu","pv","pt"):
            p=P[k][s]
            it[k]=p
            # samples per cell = mean |dp| per step, inverted
            dd=np.abs(np.diff(p))
            rate=np.median(dd[dd>0]) if (dd>0).any() else 0.0
            it[k+"_spc"] = (1.0/rate) if rate>0 else np.inf
            it[k+"_cm"] = (x[-1]/max(np.nanmax(p)-np.nanmin(p),1e-9))
        items.append(it)
    return items

def h1(phi, r):
    return 2.0*np.abs(np.mean(r*np.exp(2j*np.pi*phi)))


def h1_items(sel, plane, shifts=None):
    """First-harmonic amplitude pooled over items, optionally with a per-item
    circular shift of the residual.  The shift keeps each item's residual
    autocorrelation EXACTLY and only breaks its alignment with the phase --
    the permutation null does not, and a red-noise series folded on a slowly
    varying phase beats a permutation null with no periodicity at all."""
    num = 0.0 + 0.0j
    n = 0
    for i, it in enumerate(sel):
        r = it["r"]
        if shifts is not None:
            r = np.roll(r, shifts[i])
        phi = np.mod(it[plane], 1.0) if plane != "_ctrl" else np.mod(it["L"]/0.4792, 1.0)
        num += np.sum(r*np.exp(2j*np.pi*phi))
        n += r.size
    return 2.0*np.abs(num)/n

def main(prepdir, out):
    rng=np.random.default_rng(20260912)
    items=collect(prepdir)
    print("items with a usable plateau stretch: %d" % len(items))
    lines=[]
    for plane in ("pw","pu","pv","pt"):
        for lo,hi,nm in ((0,np.inf,"all"),(3.0,np.inf,"spc>=3"),(6.0,np.inf,"spc>=6"),(1.0,3.0,"spc<3")):
            sel=[it for it in items if lo <= it[plane+"_spc"] < hi]
            if len(sel) < 5: continue
            npts=sum(it["r"].size for it in sel)
            a=h1_items(sel, plane)
            null=np.array([h1_items(sel, plane,
                                    [rng.integers(1, it["r"].size) for it in sel])
                           for _ in range(NPERM)])
            p=((null>=a).sum()+1)/(NPERM+1)
            lines.append("%-4s %-8s items=%3d pts=%6d  A1=%.4f  null95=%.4f  p=%.4f"
                         % (plane, nm, len(sel), npts, a, np.percentile(null,95), p))
    # control: a phase with no readout meaning -- L modulo the median W pitch in cm
    npts=sum(it["r"].size for it in items)
    a=h1_items(items, "_ctrl")
    null=np.array([h1_items(items, "_ctrl",
                            [rng.integers(1, it["r"].size) for it in items])
                   for _ in range(NPERM)])
    lines.append("CTRL L/0.4792  items=%3d pts=%6d  A1=%.4f  null95=%.4f  p=%.4f"
                 % (len(items), npts, a, np.percentile(null,95), ((null>=a).sum()+1)/(NPERM+1)))
    txt="\n".join(lines)
    print(txt); open(out,"w").write(txt+"\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

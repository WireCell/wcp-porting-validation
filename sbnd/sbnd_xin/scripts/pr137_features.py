#!/usr/bin/env python3
# doc pr/137 sec 13 -- the feature kernel: the owner's three factors, computed.
# READ-ONLY library, imported by pr137_trigger_bakeoff.py.
"""Every candidate trigger statistic, in one place, grouped by the owner's own
factors so the bake-off can report which of HIS factors carries the power.

  Family D  direction in theta-phi         (owner factor 1)
  Family S  size/distance vs the nearby large EM shower (owner factors 2+3)
  Family C  conversion structure           (owner factor 3, and the answer to 4)
  Family T  topology                       (owner factor 4)

Every Family-S feature is a PULL or a RATIO, never a raw cm -- doc pr/137 sec 12's
null model is what makes objects at different depths comparable.
"""
import math, collections
import numpy as np
import pr137_lib as L

W0, SLOPE = 3.575, 0.0283          # pr137_null_model.py fit, w_single(r) cm

def w_single(r):
    return W0 + SLOPE*np.asarray(r, float)

def _twomeans(r, iters=15):
    """doc pr/137 sec 3's kernel: segment-level charge-weighted ray 2-means.
    Measured in pr137_seed_split.py to be the better SPLIT kernel (median purity
    0.825 vs 0.734 for point-level seeded assignment), so it stays the kernel and
    the seeded density supplies the TRIGGER."""
    P,v,segs,ms = r['P'], r['v'], r['segs'], r['ms']
    dirs={}
    for s in segs:
        A=P.get(s)
        if A is None or not len(A): continue
        c=L.qw_centroid(A[:,:3],A[:,3]); u=c-v; n=np.linalg.norm(u)
        if n>0: dirs[s]=u/n
    segs=[s for s in segs if s in dirs]
    if len(segs)<2: return None,None
    b0=max(segs,key=lambda s: ms[s])
    b1=min(segs,key=lambda s: float(dirs[s]@dirs[b0]))
    D=[dirs[b0].copy(), dirs[b1].copy()]; asg={}
    for _ in range(iters):
        asg={s:(0 if dirs[s]@D[0] > dirs[s]@D[1] else 1) for s in segs}
        for c in (0,1):
            mem=[s for s in segs if asg[s]==c]
            if mem:
                vv=sum(dirs[s]*max(ms[s],0.0) for s in mem); n=np.linalg.norm(vv)
                if n>0: D[c]=vv/n
    return asg, D

def _bimodality(x, w):
    """1-D shape statistics of the charge-weighted projection onto the split axis.

    Returns (bimodality coefficient, valley depth, dBIC).  This is the
    lightweight stand-in for a dip test: no diptest dependency exists and the
    shipped C++ may not take one, so the statistics chosen are ones ~40 lines of
    C++ can reproduce.  dBIC = BIC(1 Gaussian) - BIC(2 Gaussians): positive means
    two components are preferred (the X-means / CMS-GMM model-selection rule)."""
    w=L.qwt(w); n=len(x)
    if n<12: return float('nan'), float('nan'), float('nan')
    mu=(x*w).sum()/w.sum(); var=((x-mu)**2*w).sum()/w.sum()
    if var<=0: return float('nan'), float('nan'), float('nan')
    sd=math.sqrt(var)
    g1=(((x-mu)/sd)**3*w).sum()/w.sum()
    g2=(((x-mu)/sd)**4*w).sum()/w.sum()-3.0
    bc=(g1*g1+1.0)/max(g2+3.0, 1e-9)
    hist,edges=np.histogram(x,bins=16,weights=w)
    if hist.max()<=0: return bc, float('nan'), float('nan')
    i=int(np.argmax(hist))
    j=int(np.argmax(np.where(np.arange(16)<i-2, hist, 0.0)))
    if hist[j]<=0: j=int(np.argmax(np.where(np.arange(16)>i+2, hist, 0.0)))
    if hist[j]<=0: valley=1.0
    else:
        a,b=min(i,j),max(i,j)
        valley=float(hist[a:b+1].min()/min(hist[i],hist[j]))
    # 1 vs 2 Gaussian BIC on the (weighted) 1-D projection
    ll1=-0.5*n*(math.log(2*math.pi*var)+1.0)
    thr=0.5*(edges[i]+edges[j]); m=x<thr
    if m.sum()<4 or (~m).sum()<4: return bc, valley, float('nan')
    ll2=0.0
    for sel in (m,~m):
        ww=w[sel]; xx=x[sel]
        mu_=(xx*ww).sum()/ww.sum(); v_=((xx-mu_)**2*ww).sum()/ww.sum()
        if v_<=0: return bc, valley, float('nan')
        ll2 += -0.5*len(xx)*(math.log(2*math.pi*v_)+1.0) + len(xx)*math.log(len(xx)/n)
    dbic = (ll2-ll1)*2 - 3*math.log(n)     # 3 extra params (mu2, var2, weight)
    return bc, valley, dbic

def features(r, pop_by_ev, sigma_fn):
    """all trigger features for one population row -> dict"""
    f={}
    P,v,segs,ms = r['P'], r['v'], r['segs'], r['ms']
    pts,q,dx = L.pack(P,segs)
    if pts is None or len(pts)<8: return None
    f['npts']=len(pts); f['nseg']=len(segs); f['Q']=r['Q']

    # ---- the split, by the better kernel (segment-level ray 2-means)
    asg,D = _twomeans(r)
    if asg is None: return None
    p0=[s for s in segs if asg.get(s)==0]; p1=[s for s in segs if asg.get(s)==1]
    q0=sum(max(ms[s],0.) for s in p0); q1=sum(max(ms[s],0.) for s in p1)
    if not p0 or not p1 or q0<=0 or q1<=0: return None
    f['balance']=min(q0,q1)/(q0+q1)
    f['angle']=math.degrees(math.acos(max(-1,min(1,float(D[0]@D[1])))))
    f['gap_cm']=L.min_gap(P,p0,p1)                      # EXACT (pr/137 sec 1d)

    # ---- Family D: direction in theta-phi (owner factor 1)
    M=L.angular_maxima(pts,q,v,sigma_fn,sep_scale=1.6,max_seeds=4)
    k=len(M['dirs']); f['n_seed']=k
    if k>=2:
        f['d2_over_d1']=float(M['dens'][1]/max(M['dens'][0],1e-12))
        f['valley']=float(M['valley'][0,1])
        f['seed_frac']=float(min(M['frac'][0],M['frac'][1]))
        f['seed_angle']=math.degrees(math.acos(max(-1,min(1,float(M['dirs'][0]@M['dirs'][1])))))
    else:
        f['d2_over_d1']=0.0; f['valley']=1.0; f['seed_frac']=0.0; f['seed_angle']=0.0
    U,rr = L.rays(pts,v)
    ax = D[0]-D[1]; n=np.linalg.norm(ax)
    if n>0:
        proj = U@(ax/n)
        bc,val1d,dbic = _bimodality(proj, q)
        f['bimodal_coef']=bc; f['valley_1d']=val1d; f['dBIC']=dbic
    else:
        f['bimodal_coef']=f['valley_1d']=f['dBIC']=float('nan')

    # ---- Family S: size/distance vs the nearby large EM shower (factors 2+3)
    w_obj=L.transverse_rms(pts,q,v)
    r_obj=float(np.linalg.norm(L.qw_centroid(pts,q)-v))
    f['w_pull']=(w_obj-float(w_single(r_obj)))/max(0.35*float(w_single(r_obj)),1e-6)
    f['w_over_expected']=w_obj/max(float(w_single(r_obj)),1e-6)
    f['sep_scaled']=2*r_obj*math.sin(math.radians(f['angle'])/2.0)/max(float(w_single(r_obj)),1e-6)
    for tag,pp in (('0',p0),('1',p1)):
        a,bq,_=L.pack(P,pp)
        if a is None: f['w_pull_%s'%tag]=float('nan'); continue
        ww=L.transverse_rms(a,bq,v); rrp=float(np.linalg.norm(L.qw_centroid(a,bq)-v))
        f['w_pull_%s'%tag]=(ww-float(w_single(rrp)))/max(0.35*float(w_single(rrp)),1e-6)
        f['r_%s'%tag]=rrp
    f['w_pull_min']=min(f.get('w_pull_0',np.nan), f.get('w_pull_1',np.nan))
    f['dr_parts']=abs(f.get('r_0',np.nan)-f.get('r_1',np.nan))     # owner factor 3
    dom=L.dominant_other(pop_by_ev.get(r['event'],[]), r)
    f['owner_ref']= 0.0 if dom is None else 1.0                    # 0 = fallback fired
    if dom is not None:
        dp,dq,_=L.pack(dom['P'],dom['segs'])
        if dp is not None:
            wd=L.transverse_rms(dp,dq,v); rd=float(np.linalg.norm(L.qw_centroid(dp,dq)-v))
            f['w_ratio']=w_obj/max(wd,1e-6); f['r_ratio']=r_obj/max(rd,1e-6)
            f['q_ratio']=r['Q']/max(dom['Q'],1e-6)
            f['w_at_r_ratio']=w_obj/max(wd*float(w_single(r_obj))/max(float(w_single(rd)),1e-6),1e-6)
    for kk in ('w_ratio','r_ratio','q_ratio','w_at_r_ratio'):
        f.setdefault(kk, float('nan'))

    # ---- Family C: conversion structure (owner factor 3, answer to factor 4)
    mip = r.get('mip') or 52137.8      # dqdx_ref electron plateau, per event
    ded=[]; ded15=[]; vg=[]; vf=[]
    for pp in (p0,p1):
        a,bq,bdx=L.pack(P,pp)
        if a is None:
            ded.append(np.nan); ded15.append(np.nan); vg.append(np.nan); vf.append(np.nan); continue
        ded.append(L.start_dedx(a,bq,bdx,v,span=3.0)/mip)
        ded15.append(L.start_dedx(a,bq,bdx,v,span=1.5)/mip)
        vg.append(L.vertex_gap(a,v)); vf.append(L.void_frac(a,bq,v))
    # in MIP units: ~1 = electron-like stem, ~2 = a photon conversion (e+e- pair)
    f['dedx0'],f['dedx1']=ded
    f['dedx15_min']=float(np.nanmin(ded15)) if np.any(~np.isnan(ded15)) else float('nan')
    f['dedx15_max']=float(np.nanmax(ded15)) if np.any(~np.isnan(ded15)) else float('nan')
    f['n_2mip']=float(sum(1 for x in ded15 if x==x and x>1.6))
    f['dedx_min']=float(np.nanmin(ded)) if np.any(~np.isnan(ded)) else float('nan')
    f['vgap_min']=float(np.nanmin(vg)) if np.any(~np.isnan(vg)) else float('nan')
    f['vgap_max']=float(np.nanmax(vg)) if np.any(~np.isnan(vg)) else float('nan')
    f['void_min']=float(np.nanmin(vf)) if np.any(~np.isnan(vf)) else float('nan')
    ref=float(np.nanmedian([d for d in ded if d==d])) if any(d==d for d in ded) else np.nan
    f['dedx_ratio']=(max(ded)/min(ded)) if all(d==d and d>0 for d in ded) else float('nan')

    # ---- Family T: topology (owner factor 4)
    f['bridge_gap']=f['gap_cm']
    f['gap_scaled']=f['gap_cm']/max(float(w_single(r_obj)),1e-6)

    # ---- diagnostic only, NEVER a trigger (doc pr/137 sec 11): the pi0 mass
    rec=r.get('rec') or {}
    kc=float(rec.get('kine_charge') or 0.0)
    tq=sum(max(ms[s],0.) for s in segs) or 1.0
    e0,e1 = kc*q0/tq, kc*q1/tq
    f['m_pi0']=math.sqrt(max(0.0,4*e0*e1))*math.sin(math.radians(f['angle'])/2.0)
    return f

FAMILY = {
 'D':['n_seed','d2_over_d1','valley','seed_frac','seed_angle','bimodal_coef','valley_1d','dBIC','angle'],
 'S':['w_pull','w_over_expected','sep_scaled','w_pull_min','dr_parts','w_ratio','r_ratio','q_ratio','w_at_r_ratio'],
 'C':['dedx0','dedx1','dedx_min','dedx_ratio','dedx15_min','dedx15_max','n_2mip','vgap_min','vgap_max','void_min'],
 'T':['gap_cm','gap_scaled','balance'],
 'X':['m_pi0'],
}

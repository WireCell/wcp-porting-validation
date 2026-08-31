#!/usr/bin/env python3
# doc pr/137 sec 12 -- the in-situ NULL MODEL for the split trigger.  READ-ONLY.
"""What does a SINGLE EM shower look like, as a function of depth?

Owner factor 2: "for a single shower, we do expect that the growth of the shower
to be bigger as the distance going further".  Owner factor 3: "it is possible
that one gamma convert at a much longer distance than the other one".  Both say
the same thing about the metric -- a raw angle or a raw width is not comparable
between objects at different depths, which is exactly why doc pr/137 sec 4's raw
angle/balance/gap cuts had such a broad null distribution.

So: fit w_single(r) and sigma_ang_single(r) on the SINGLE population and publish
them.  Every Family-D and Family-S feature in the bake-off is then a PULL against
this, not a raw number.  No PDG constant is used as a threshold anywhere -- LAr
X0 = 14.0 cm and R_M = 10.0 cm are quoted for scale only.

    scripts/pr137_null_model.py            # -> docs/pr/pr137-null-model.tsv
"""
import sys, os, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pr137_lib as L

OUT = 'docs/pr/pr137-null-model.tsv'

def main():
    rows=[]
    for ev in L.prep_events('emprep-136onV1c90'):
        rows += L.build_population(ev)
    single=[r for r in rows if r['cls']=='SINGLE']
    merged=[r for r in rows if r['cls']=='MERGED']
    print("population: %d rows  (%d SINGLE, %d MERGED, %d ORPHAN)"%(
        len(rows), len(single), len(merged), sum(1 for r in rows if r['cls']=='ORPHAN')))

    # ---- w(r): transverse RMS vs depth, pooled over the SINGLE population
    prof=collections.defaultdict(list)
    dedx=[]; wtot=[]; ang_rms=[]
    for r in single:
        pts,q,dx = L.pack(r['P'], r['segs'])
        if pts is None or len(pts)<8: continue
        rm, wm = L.width_profile(pts,q,r['v'])
        for a,b in zip(rm,wm):
            if a>0: prof[int(min(a//10,19))].append((a,b))
        wtot.append(L.transverse_rms(pts,q,r['v']))
        U,rr = L.rays(pts, r['v'])
        c = L.qw_centroid(pts,q); ax=c-r['v']; n=np.linalg.norm(ax)
        if n>0:
            cosang=np.clip(U@(ax/n),-1,1)
            w=L.qwt(q)
            ang_rms.append(math.degrees(math.sqrt(((np.arccos(cosang)**2)*w).sum()/w.sum())))
        d=L.start_dedx(pts,q,dx,r['v'])
        if d==d: dedx.append(d)

    lines=[]
    lines.append("# doc pr/137 sec 12 -- in-situ null model, calibrated on the SINGLE population")
    lines.append("# arms: work-pr136-onV1c90-* ; sidecars emprep-136off2 / emprep-136onV1c90")
    lines.append("# SINGLE=%d MERGED=%d ; LAr X0=%.1fcm R_M=%.1fcm quoted for SCALE ONLY"%(
        len(single),len(merged),L.X0_LAR,L.RM_LAR))
    lines.append("block\tkey\tn\tp10\tmedian\tp90\tmean")

    def emit(block,key,v):
        v=[x for x in v if x==x]
        if not v: return
        a=np.asarray(v,float)
        lines.append("%s\t%s\t%d\t%.4g\t%.4g\t%.4g\t%.4g"%(block,key,len(a),
            np.percentile(a,10),np.median(a),np.percentile(a,90),a.mean()))
        return np.median(a)

    print("\n== w_single(r): charge-weighted transverse RMS vs depth (SINGLE) ==")
    print("  %-14s %5s %8s %8s %8s"%("depth bin cm","n","p10","median","p90"))
    fit_r=[]; fit_w=[]
    for b in sorted(prof):
        v=np.asarray([x[1] for x in prof[b]]); rr=np.asarray([x[0] for x in prof[b]])
        if len(v)<8: continue
        med=float(np.median(v))
        print("  %-14s %5d %8.2f %8.2f %8.2f"%(("%d-%d"%(b*10,b*10+10) if b<19 else "190+"),len(v),
              np.percentile(v,10),med,np.percentile(v,90)))
        lines.append("w_single_vs_r\t%s\t%d\t%.4g\t%.4g\t%.4g\t%.4g"%(
            ("%d-%dcm"%(b*10,b*10+10)) if b<19 else "190+cm",len(v),np.percentile(v,10),med,np.percentile(v,90),v.mean()))
        fit_r.append(float(np.median(rr))); fit_w.append(med)

    # the one number the seeding kernel needs: sigma_ang(r) = w_scale / r
    fit_r=np.asarray(fit_r); fit_w=np.asarray(fit_w)
    w_scale = float(np.median(fit_w)) if len(fit_w) else 6.0
    # linear growth fit w(r) = w0 + slope*r  (owner factor 2, made quantitative)
    slope=w0=float('nan')
    if len(fit_r)>=3:
        slope, w0 = np.polyfit(fit_r, fit_w, 1)
    print("\n  linear fit  w_single(r) = %.3f + %.4f * r   (cm)"%(w0,slope))
    print("  -> owner factor 2 is CONFIRMED/DENIED by the sign of the slope: %s"%(
          "CONFIRMED, width grows with depth" if slope>0 else "DENIED, width does not grow"))
    lines.append("w_single_fit\tw0_cm\t%d\t\t%.4g\t\t"%(len(fit_r),w0))
    lines.append("w_single_fit\tslope_cm_per_cm\t%d\t\t%.4g\t\t"%(len(fit_r),slope))
    lines.append("w_single_fit\tw_scale_cm\t%d\t\t%.4g\t\t"%(len(fit_r),w_scale))

    print("\n== whole-object statistics (SINGLE) ==")
    m1=emit("single_obj","transverse_rms_cm",wtot);  print("  transverse RMS   median %.2f cm"%m1)
    m2=emit("single_obj","angular_rms_deg",ang_rms); print("  angular RMS      median %.1f deg"%m2)
    m3=emit("single_obj","start_dQdx",dedx);         print("  start dQ/dx      median %.4g"%m3)

    # the same, in the OWNER'S normalisation: ratio to the nearby large EM shower
    byev=collections.defaultdict(list)
    for r in rows: byev[r['event']].append(r)
    wr=[]; rr_=[]; qr=[]; nofb=0; nall=0
    for r in single:
        nall+=1
        dom=L.dominant_other(byev[r['event']], r)
        if dom is None: nofb+=1; continue
        p1,q1,_=L.pack(r['P'],r['segs']); p2,q2,_=L.pack(dom['P'],dom['segs'])
        if p1 is None or p2 is None: continue
        w1=L.transverse_rms(p1,q1,r['v']); w2=L.transverse_rms(p2,q2,r['v'])
        d1=np.linalg.norm(L.qw_centroid(p1,q1)-r['v']); d2=np.linalg.norm(L.qw_centroid(p2,q2)-r['v'])
        if w2>0: wr.append(w1/w2)
        if d2>0: rr_.append(d1/d2)
        if dom['Q']>0: qr.append(r['Q']/dom['Q'])
    print("\n== the OWNER's normalisation: ratio to the nearby large EM shower ==")
    print("  fallback fires (no other EM object above the floor): %d / %d = %.0f%%"%(
        nofb,nall,100*nofb/max(1,nall)))
    emit("owner_ratio","w_ratio",wr); emit("owner_ratio","r_ratio",rr_); emit("owner_ratio","q_ratio",qr)
    lines.append("owner_ratio\tfallback_frac\t%d\t\t%.4g\t\t"%(nall, nofb/max(1,nall)))
    for k,v in (("w_ratio",wr),("r_ratio",rr_),("q_ratio",qr)):
        if v: print("  %-9s median %.3f"%(k,float(np.median(v))))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT,'w').write("\n".join(lines)+"\n")
    print("\nwrote %s"%OUT)

if __name__=='__main__':
    main()

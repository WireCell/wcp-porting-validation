#!/usr/bin/env python3
# doc pr/137 sec 13a -- the SEEDED splitter kernel.  READ-ONLY.
"""Point-level seeding, segment-level assignment -- the shape every production
calorimeter splitter uses (ATLAS topo-cluster splitting, CMS particle-flow,
GARLIC, the 2025 Hough photon reconstruction).

The reframe (doc pr/137 sec 10): those algorithms do NOT run a splitter and then
veto it.  They seed on local maxima and let the SEED COUNT be the multiplicity
decision.  doc pr/137 sec 3-4 did the opposite -- a global 2-means that always
fires, then a hunt for an external accept test -- which is why the null
distribution was broad and the best purity was 27-36 %.

Two measurements here:
  (a) recovery -- given a KNOWN 2-way merge, does the seeded kernel find the OFF
      partition?  This is doc pr/137 sec 3's benchmark (0.920 median purity from
      a SEGMENT-level ray 2-means) recomputed with point-level seeding.
  (b) multiplicity -- how often is n_seed == 1 on SINGLE and >= 2 on MERGED?
      This is the trigger, measured directly.

    scripts/pr137_seed_split.py            # -> docs/pr/pr137-seed-split.tsv
"""
import sys, os, math, itertools, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pr137_lib as L

OUT='docs/pr/pr137-seed-split.tsv'

def purity(ms, truth, asg):
    """charge-weighted purity of the best k<->k assignment against the OFF partition"""
    tl=sorted({truth[s] for s in truth if truth[s] is not None})
    pl=sorted({asg[s] for s in asg})
    if len(tl)<2 or len(pl)<1: return float('nan')
    best=0.0; tot=sum(ms[s] for s in ms if truth.get(s) is not None) or 1
    for perm in itertools.permutations(pl, min(len(pl),len(tl))):
        m={t:p for t,p in zip(tl,perm)}
        g=sum(ms[s] for s in ms if truth.get(s) is not None and asg.get(s)==m.get(truth[s]))
        best=max(best,g)
    return best/tot

def segment_2means(r, iters=15):
    """doc pr/137 sec 3's kernel verbatim: 2-means on SEGMENT ray directions from
    the reference vertex, charge-weighted.  Recomputed here rather than quoted so
    the baseline and the challenger see the identical population."""
    P,v,segs,ms = r['P'], r['v'], r['segs'], r['ms']
    dirs={}
    for s_ in segs:
        A=P.get(s_)
        if A is None or not len(A): continue
        c=L.qw_centroid(A[:,:3],A[:,3]); u=c-v; n=np.linalg.norm(u)
        if n<=0: continue
        dirs[s_]=u/n
    segs=[s_ for s_ in segs if s_ in dirs]
    if len(segs)<2: return None
    b0=max(segs,key=lambda s_: ms[s_])
    b1=min(segs,key=lambda s_: float(dirs[s_]@dirs[b0]))
    D=[dirs[b0].copy(), dirs[b1].copy()]
    asg={}
    for _ in range(iters):
        asg={s_:(0 if dirs[s_]@D[0] > dirs[s_]@D[1] else 1) for s_ in segs}
        for c in (0,1):
            mem=[s_ for s_ in segs if asg[s_]==c]
            if mem:
                vv=sum(dirs[s_]*max(ms[s_],0.0) for s_ in mem)
                n=np.linalg.norm(vv)
                if n>0: D[c]=vv/n
    return asg

def main():
    rows=[]
    for ev in L.prep_events('emprep-136onV1c90'):
        rows += L.build_population(ev)
    single=[r for r in rows if r['cls']=='SINGLE']
    merged=[r for r in rows if r['cls']=='MERGED']
    print("population %d  (SINGLE %d, MERGED %d)"%(len(rows),len(single),len(merged)))
    print("NOTE doc pr/137 sec 4 published 33 MERGED / 354 SINGLE.  The gap is a JOIN")
    print("     defect, not a population change: sec 4 built the OFF owner map as")
    print("     {seg: shower} (last-writer-wins), so a segment held by two OFF showers")
    print("     lost one ancestor.  1.0 %% of OFF segments are shared, and the lossy")
    print("     join relabels 10 real MERGED objects as SINGLE.\n")

    lines=["# doc pr/137 sec 13a -- seeded splitter kernel",
           "# population: onV1c90 showers, q>1e6, >=3 segments, faithful OFF-ancestor join",
           "# SINGLE=%d MERGED=%d"%(len(single),len(merged)),
           "block\tvariant\tn\tvalue\tnote"]

    # ---------- (a) recovery, with k FORCED to 2 ---------------------
    # Fair comparison with doc pr/137 sec 3: that benchmark GAVE the kernel k=2.
    # Scoring "no second seed found" as purity 0 conflates the kernel with the
    # trigger, so recovery here always takes the top-2 angular maxima and the
    # trigger is measured separately in (b).
    two=[r for r in merged if r['nanc']==2]
    print("== (a) RECOVERY of the OFF partition on %d two-way merges (k forced to 2) =="%len(two))
    print("  %-44s %5s %8s %8s %8s"%("kernel","n","median","ge0.90","ge0.99"))
    for name, sig, sep in (
            ("seeded rays, profile sigma, sep1.6", L.profile_sigma_fn(), 1.6),
            ("seeded rays, profile sigma, sep1.0", L.profile_sigma_fn(), 1.0),
            ("seeded rays, profile sigma, sep2.5", L.profile_sigma_fn(), 2.5),
            ("seeded rays, flat sigma w=6cm, sep1.6", L.default_sigma_fn(6.0), 1.6),
            ("seeded rays, flat sigma w=4cm, sep1.0", L.default_sigma_fn(4.0), 1.0)):
        ps=[]; nofind=0
        for r in two:
            pts,q,dx=L.pack(r['P'],r['segs'])
            if pts is None: continue
            M=L.angular_maxima(pts,q,r['v'],sig,sep_scale=sep,max_seeds=2)
            if len(M['dirs'])<2: nofind+=1; continue
            asg,_=L.assign_segments(r['P'],r['segs'],r['v'],M['dirs'][:2])
            p_=purity(r['ms'],r['truth'],asg)
            if p_==p_: ps.append(p_)
        a=np.asarray(ps) if ps else np.zeros(1)
        print("  %-44s %5d %8.3f %8d %8d   (no 2nd max: %d)"%(name,len(ps),np.median(a),
              (a>=0.90).sum(),(a>=0.99).sum(),nofind))
        lines.append("recovery\t%s\t%d\t%.4f\tge0.90=%d ge0.99=%d nofind=%d"%(
            name,len(ps),np.median(a),(a>=0.90).sum(),(a>=0.99).sum(),nofind))
    # the pr/137 sec 3 kernel, recomputed on THIS population so the comparison is exact
    ps=[]
    for r in two:
        asg=segment_2means(r)
        if asg is None: continue
        p_=purity(r['ms'],r['truth'],asg)
        if p_==p_: ps.append(p_)
    a=np.asarray(ps)
    print("  %-44s %5d %8.3f %8d %8d"%("[pr/137 sec3 kernel: segment-level ray 2-means]",
          len(a),np.median(a),(a>=0.90).sum(),(a>=0.99).sum()))
    lines.append("recovery\t[pr137 sec3 segment-level ray 2-means, recomputed]\t%d\t%.4f\tge0.90=%d ge0.99=%d"%(
        len(a),np.median(a),(a>=0.90).sum(),(a>=0.99).sum()))

    # ---------- (b) the multiplicity trigger -----------------------------
    # The whole point of the reframe: the seed count IS the decision.  But a
    # bright patch inside ONE shower also makes a local maximum, so ATLAS's rule
    # is local-maxima-WITH-A-VALLEY -- the density must dip between them.  Here
    # the acceptance is (2nd peak >= dratio * 1st) AND (valley <= vmax * weaker
    # peak) AND (minor charge share >= fmin).
    print("\n== (b) MULTIPLICITY: n_seed as the trigger (no external accept test) ==")
    sig=L.profile_sigma_fn()
    cache={}
    for r in rows:
        if r['cls'] not in ('MERGED','SINGLE'): continue
        pts,q,dx=L.pack(r['P'],r['segs'])
        if pts is None: continue
        M=L.angular_maxima(pts,q,r['v'],sig,sep_scale=1.6,max_seeds=4)
        if len(M['dirs'])<2:
            cache[(r['event'],r['node'])]=(r['cls'],0.0,1.0,0.0); continue
        d=M['dens']; f=M['frac']; V=M['valley']
        cache[(r['event'],r['node'])]=(r['cls'], float(d[1]/max(d[0],1e-12)),
                                       float(V[0,1]), float(min(f[0],f[1])))
    print("  %-38s %-22s %-22s %8s %8s"%("accept rule","MERGED fires","SINGLE fires","enrich","purity"))
    best=None
    for dratio, vmax, fmin in ((0.20,1.00,0.00),(0.35,1.00,0.00),(0.50,1.00,0.00),
                               (0.20,0.90,0.05),(0.35,0.90,0.05),(0.35,0.80,0.10),
                               (0.50,0.80,0.10),(0.50,0.70,0.15),(0.65,0.70,0.15),
                               (0.65,0.60,0.20),(0.80,0.60,0.20),(0.80,0.50,0.25)):
        fm=nm=fs=ns=0
        for (cls,dr,vv,ff) in cache.values():
            fire = (dr>=dratio) and (vv<=vmax) and (ff>=fmin)
            if cls=='MERGED': nm+=1; fm+=fire
            else:             ns+=1; fs+=fire
        em=fm/max(1,nm); es=fs/max(1,ns); pur=fm/max(1,fm+fs)
        name="d2/d1>=%.2f valley<=%.2f frac>=%.2f"%(dratio,vmax,fmin)
        print("  %-38s %-22s %-22s %7.1fx %7.0f%%"%(name,
              "%d/%d (%.0f%%)"%(fm,nm,100*em),"%d/%d (%.0f%%)"%(fs,ns,100*es),
              min(em/max(1e-9,es),999.0),100*pur))
        lines.append("multiplicity\t%s\t%d\t%.4f\tmerged=%d/%d single=%d/%d enrich=%.2f"%(
            name,fm+fs,pur,fm,nm,fs,ns,em/max(1e-9,es)))
        if fm>=5 and (best is None or pur>best[0]): best=(pur,name,fm,nm,fs,ns)
    print("\n  best (>=5 merged fires): %s -> purity %.0f%% (%d merged, %d single)"%(
          best[1],100*best[0],best[2],best[4]) if best else "\n  no rule fired on >=5 merged")
    print("  [pr/137 sec 4 best across all three families: purity 27-36 %]")
    lines.append("multiplicity\t[pr137 sec4 best geometric]\t\t0.36\tthe number to beat")

    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    open(OUT,'w').write("\n".join(lines)+"\n")
    print("\nwrote %s"%OUT)

if __name__=='__main__': main()

#!/usr/bin/env python3
# doc pr/137 sec 14 -- the CURATED validation set: stratified sampler + contact sheets.
# READ-ONLY with respect to every existing record; writes only into a FRESH tag dir.
"""The owner asked for "a good set of curated data".  This builds it.

WHY STRATIFIED, AND WHY THE CONTROL STRATUM COMES FIRST.  Scanning only the
objects a trigger fires on measures purity and leaves efficiency unmeasured, and
it samples a population selected by a trigger we may be about to discard.  So the
random control stratum is drawn with a fixed seed BEFORE any feature is consulted,
and the enriched stratum is drawn afterwards.

  S1  random control   100   uniform over the q>1e6, >=3-seg onV1c90 population
  S2  known merges      44   every object with >=2 OFF ancestors (faithful join)
  S3  enriched          40   top by the bake-off's leading trigger, minus S1/S2

The owner scans ~50 of these as a CALIBRATION overlap (25 S1 / 15 S2 / 10 S3) so
agent-vs-owner agreement is measured across the whole range, not just on easy
objects.  The agent scans the rest.  If agreement is A, every agent-derived rate
carries a floor of (1-A) and no trigger may be claimed to beat it.

VERDICT VOCABULARY (doc pr/137 sec 15.6 -- TRIM was added after Scan A):
  KEEP    one object.  Do not touch it.
  SPLIT2  two objects.  Give the boundary: which segments go to part 0 / part 1.
  SPLIT3  three objects.  Same, three parts.
  TRIM    ONE object plus detached junk that does not belong to it.  Name the
          junk segments.  This is NOT a split -- there is no second object, and
          forcing it into KEEP or SPLIT is wrong either way.  Scan A found this
          is ~45 % of what the arm-difference proxy calls a 'merge' (463565,
          98844, 282909, 386948, 105946), and it survives the production prune
          passes, so it is a real and separate front.
  UNSURE  too sparse or ambiguous to call.  Preferred over a forced verdict.

    scripts/pr137_curate.py                 # -> docs/pr/pr137-curated-set.tsv
    scripts/pr137_curate.py --sheets        # + contact-sheet PNGs for the agent scan
"""
import sys, os, math, json, collections, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pr137_lib as L
import pr137_features as F

OUT='docs/pr/pr137-curated-set.tsv'
SHEETDIR='work/pr137_sheets'          # fresh dir; no existing record is touched
SEED=20260901

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sheets', action='store_true', help='render contact-sheet PNGs')
    ap.add_argument('--sheetdir', default=SHEETDIR)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--unblind', action='store_true',
                    help='print proxy class + features on the sheet (for the DOC, never for the scan)')
    a=ap.parse_args()

    sig=L.profile_sigma_fn()
    rows=[]
    for ev in L.prep_events('emprep-136onV1c90'):
        rows += L.build_population(ev)
    byev=collections.defaultdict(list)
    for r in rows: byev[r['event']].append(r)
    feat={}
    for r in rows:
        if r['cls'] not in ('MERGED','SINGLE'): continue
        f=F.features(r, byev, sig)
        if f: feat[(r['event'],r['node'])]=f
    pool=[r for r in rows if (r['event'],r['node']) in feat]
    print("pool: %d objects (%d MERGED, %d SINGLE)"%(len(pool),
          sum(1 for r in pool if r['cls']=='MERGED'), sum(1 for r in pool if r['cls']=='SINGLE')))

    rng=np.random.default_rng(SEED)
    key=lambda r:(r['event'],r['node'])
    # S1 -- FEATURE-INDEPENDENT, drawn first
    idx=rng.choice(len(pool), size=min(100,len(pool)), replace=False)
    S1={key(pool[i]) for i in idx}
    # S2 -- every known merge
    S2={key(r) for r in pool if r['cls']=='MERGED'}
    # S3 -- enriched by the bake-off's leading trigger (valley low & d2/d1 high)
    def score(r):
        # sec 15.2: valley_best, not valley -- the latter tests the two highest-
        # DENSITY maxima, which are often both inside the same lobe.
        f=feat[key(r)]
        return (1.0-min(f.get('valley_best',1.0),1.0)) + f.get('d2_best',0.0)
    rest=[r for r in pool if key(r) not in S1|S2]
    rest.sort(key=lambda r: -score(r))
    S3={key(r) for r in rest[:40]}
    strat={}
    for k in S1: strat[k]='S1'
    for k in S2: strat[k]='S2' if k not in S1 else 'S1+S2'
    for k in S3: strat.setdefault(k,'S3')
    sel=[r for r in pool if key(r) in strat]
    print("curated set: %d objects  (S1 %d, S2 %d, S3 %d; overlap S1&S2 %d)"%(
        len(sel),len(S1),len(S2),len(S3),len(S1&S2)))

    # the owner's ~50 calibration overlap, spread across strata, seeded
    def pick(cands,n):
        cands=sorted(cands)
        if len(cands)<=n: return set(cands)
        j=rng.choice(len(cands),size=n,replace=False)
        return {cands[i] for i in j}
    own = pick([k for k in strat if strat[k]=='S1'],25) \
        | pick([k for k in strat if strat[k] in ('S2','S1+S2')],15) \
        | pick([k for k in strat if strat[k]=='S3'],10)
    print("owner calibration subset: %d objects"%len(own))

    cols=['event','node','stratum','owner_scan','proxy_cls','Q','nseg','npts',
          'valley_best','d2_best','frac_best','angle_best',
          'valley','d2_over_d1','seed_frac','n_seed','angle','balance','gap_cm',
          'gap_scaled','w_pull','sep_scaled','vgap_min','dedx15_min','n_2mip',
          'r_ratio','q_ratio','m_pi0']
    lines=["# doc pr/137 sec 14 -- curated validation set for the shower-split trigger",
           "# VERDICTS: KEEP | SPLIT2 | SPLIT3 | TRIM | UNSURE  (TRIM added by sec 15.6:",
           "#   ONE object plus detached junk -- ~45 % of what the proxy calls a merge.",
           "#   Not a split.  For SPLIT give the segment->part boundary; for TRIM name the junk.)",
           "# arms work-pr136-onV1c90-* ; sidecars emprep-136off2 / emprep-136onV1c90",
           "# RNG seed %d (fixed; the set is re-derivable)"%SEED,
           "# strata: S1 random control (feature-INDEPENDENT, drawn first) | S2 all known",
           "#   merges (faithful OFF-ancestor join) | S3 enriched by valley+d2/d1",
           "# owner_scan=1 marks the ~50-object calibration overlap (25 S1 / 15 S2 / 10 S3)",
           "# proxy_cls is the ARM-DIFFERENCE proxy, NOT truth -- doc pr/137 sec 5",
           "\t".join(cols)]
    sel.sort(key=lambda r:(-r['Q'],))
    for r in sel:
        k=key(r); f=feat[k]
        vals=[r['event'],r['node'],strat[k],1 if k in own else 0,r['cls'],
              "%.4g"%r['Q'],len(r['segs']),f.get('npts',0)]
        for c in cols[8:]:
            v=f.get(c,float('nan'))
            vals.append("%.4g"%v if isinstance(v,float) else v)
        lines.append("\t".join(str(x) for x in vals))
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    open(OUT,'w').write("\n".join(lines)+"\n")
    print("wrote %s"%OUT)

    if a.sheets:
        render(sel, feat, strat, own, sig, a.sheetdir, a.limit, blind=not a.unblind)

def render(sel, feat, strat, own, sig, sheetdir, limit, blind=True):
    """Contact sheets for the agent visual scan.

    BLIND BY DEFAULT, and that is not decoration.  The proxy class (MERGED /
    SINGLE) is the very thing these labels exist to validate; printing it on the
    sheet would let it steer the judgement and the resulting agreement number
    would be circular.  So the blind sheet carries event/node/charge/segment
    count and nothing else -- no stratum, no proxy, no feature values.

    The theta-phi map is drawn RAW (uncoloured).  Only the side view shows the
    proposed 2-means partition, so the reader first sees the charge and only
    then a hypothesis about how to cut it -- the same order the owner's Bee scan
    presents.  Colouring the primary panel by the proposal biases toward SPLIT.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(sheetdir, exist_ok=True)
    n=0; manifest=[]
    for r in sel:
        if limit and n>=limit: break
        k=(r['event'],r['node']); f=feat[k]
        pts,q,dx=L.pack(r['P'],r['segs'])
        if pts is None: continue
        v=r['v']
        asg,D=F._twomeans(r)
        if asg is None: continue
        part=np.concatenate([np.full(len(r['P'][s]), asg.get(s,0))
                             for s in r['segs'] if s in r['P']])
        U,rr=L.rays(pts,v)
        c=L.qw_centroid(pts,q); ax=c-v; ax/=max(np.linalg.norm(ax),1e-9)
        e1=np.cross(ax,[0,0,1.0])
        if np.linalg.norm(e1)<1e-6: e1=np.cross(ax,[0,1.0,0])
        e1/=np.linalg.norm(e1); e2=np.cross(ax,e1)
        th=np.degrees(np.arccos(np.clip(U@ax,-1,1)))
        ph=np.arctan2(U@e2,U@e1)
        # gnomonic-ish: place each ray at radius theta, azimuth phi -> two lobes
        # of a merged object appear as two blobs, which is the whole of factor 1
        gx=th*np.cos(ph); gy=th*np.sin(ph)
        w=L.qwt(q); sz=8*np.sqrt(w/max(w.max(),1e-9))+1
        fig,axs=plt.subplots(2,2,figsize=(11,8.2))
        axs[0,0].scatter(gx,gy,s=sz,c=w,cmap='viridis',alpha=.75,lw=0)
        M=L.angular_maxima(pts,q,v,sig,sep_scale=1.6,max_seeds=4)
        for i_ in range(len(M['dirs'])):
            d_=M['dirs'][i_]
            t_=math.degrees(math.acos(max(-1,min(1,float(d_@ax)))))
            p_=math.atan2(float(d_@e2),float(d_@e1))
            axs[0,0].plot(t_*math.cos(p_), t_*math.sin(p_), 'rx', ms=11, mew=2)
        axs[0,0].set_aspect('equal'); axs[0,0].set_xlabel('deg'); axs[0,0].set_ylabel('deg')
        axs[0,0].set_title('theta-phi ray map from the vertex (x = angular maxima)')
        rm,wm=L.width_profile(pts,q,v)
        if len(rm):
            axs[0,1].plot(rm,wm,'o-',label='this object')
            axs[0,1].plot(rm,F.w_single(rm),'k--',label='single-shower null')
        axs[0,1].set_xlabel('depth along axis (cm)'); axs[0,1].set_ylabel('transverse RMS (cm)')
        axs[0,1].legend(fontsize=8); axs[0,1].set_title('width vs depth')
        dq=np.where(dx>0,q/np.maximum(dx,1e-9),0)/(r.get('mip') or 52137.8)
        axs[1,0].scatter(rr,dq,s=4,c='0.35',alpha=.45,lw=0)
        axs[1,0].axhline(1.0,color='k',ls=':',lw=1); axs[1,0].axhline(2.0,color='g',ls=':',lw=1)
        axs[1,0].set_ylim(0,4); axs[1,0].set_xlabel('distance from vertex (cm)')
        axs[1,0].set_ylabel('dQ/dx  [MIP]')
        axs[1,0].set_title('dE/dx vs depth  (1 = MIP, 2 = photon conversion)')
        d2=pts-v[None,:]
        for p_,col in ((0,'tab:blue'),(1,'tab:red')):
            m=part==p_
            if m.sum(): axs[1,1].scatter(d2[m]@e1, d2[m]@ax, s=sz[m], c=col, alpha=.6, lw=0)
        axs[1,1].plot([0],[0],'k*',ms=14); axs[1,1].set_xlabel('transverse (cm)')
        axs[1,1].set_ylabel('along axis (cm)')
        axs[1,1].set_title('side view: PROPOSED 2-way split (star = vertex)')
        if blind:
            head="evt%d node%d   Q=%.3g   nseg=%d   npts=%d"%(
                  r['event'],r['node'],r['Q'],len(r['segs']),f.get('npts',0))
        else:
            head=("evt%d node%d  %s  Q=%.3g nseg=%d  proxy=%s%s\n"
                  "valley=%.3f d2/d1=%.3f angle=%.1f bal=%.3f gap=%.1fcm w_pull=%.2f"%(
                  r['event'],r['node'],strat[k],r['Q'],len(r['segs']),r['cls'],
                  "  [OWNER SCAN]" if k in own else "",
                  f.get('valley',np.nan),f.get('d2_over_d1',np.nan),f.get('angle',np.nan),
                  f.get('balance',np.nan),f.get('gap_cm',np.nan),f.get('w_pull',np.nan)))
        fig.suptitle(head, fontsize=10)
        fig.tight_layout(rect=[0,0,1,0.94])
        fn='sheet-evt%d-node%d.png'%(r['event'],r['node'])
        fig.savefig(os.path.join(sheetdir,fn),dpi=95)
        plt.close(fig); n+=1
        manifest.append((fn,r['event'],r['node']))
    with open(os.path.join(sheetdir,'INDEX.txt'),'w') as fh:
        fh.write("# blind=%s -- proxy class and features are %s on these sheets\n"%(
            blind, "HIDDEN" if blind else "SHOWN"))
        for fn,ev,nd in manifest: fh.write("%s\t%d\t%d\n"%(fn,ev,nd))
    print("rendered %d contact sheets into %s (blind=%s)"%(n,sheetdir,blind))

if __name__=='__main__': main()

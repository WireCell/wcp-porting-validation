#!/usr/bin/env python3
"""doc 84 round 2 -- opening census, read-only over EXISTING arms.

Rates the two populations behind the owner's Bee-scan findings (doc 84 R1.6
scan, events 313847/281595 and 53793/177536/77978) on the CURRENT production
state (post e8b23aae flip; arms work-d84r1-flipchk98-* + work-d84r1-flip2-*):

  Pop A  chain-vs-membership truncation: showers typed |13| whose muon-typed
         member length exceeds the chain length recovered by inverting the
         shipped muon range->KE table on kine_range.  These are the events the
         proposed `long_muon_members_geometry` knob would move.

  Pop B  cathode-split muons: a muon shower with a member-trajectory end at
         |x| < xcut of the cathode (x=0), with a facing partner end (another
         shower's member, or a bare shower_id=-1 segment) on the other side,
         small 3D gap, collinear.  NB the halves do NOT share a PR cluster_id
         (53793: clus 37 vs 12) -- imaging-level stitching does not survive
         into PR cluster ids, so no same-cluster guard; recorded as a column.  NB segments[].shower_id
         holds the owning shower's `id` field, not its `shower_id`.  These are the candidates the
         proposed `long_muon_cathode_bridge` knob would join.  Guard-variable
         distributions (gap, angles, partner length) are recorded so knob
         defaults can be chosen against the three known-good cases.

CAVEAT (same as d84_longmu_census.py part 3): calib JSON is the FINAL PR
state; this rates "what a post-reconcile bridge/geometry pass would see",
which IS the proposed knobs' execution point (post shower_clustering, post
reconcile) -- unlike the P2 formation-time question, the replay here matches
the proposed code's vantage.

Usage: d84r2_census.py --arms DIR:SAMPLE [...] --out OUTDIR [--xcut 6]
                       [--gap-max 25] [--toolkit T]
"""
import argparse, collections, glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d84_longmu_census as d84   # range table, interp, find_toolkit

def vsub(a,b): return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
def vmag(a): return math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
def vang(a,b):
    ma,mb=vmag(a),vmag(b)
    if ma<=0 or mb<=0: return -1.0
    c=(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])/(ma*mb)
    return math.degrees(math.acos(max(-1.0,min(1.0,c))))
def pt(p): return (p["x"],p["y"],p["z"])

def seg_ends(seg):
    pts=seg.get("points") or []
    if len(pts)<2: return None
    return pt(pts[0]), pt(pts[-1])

def end_tangent(seg, which, lever_cm=5.0):
    """Direction pointing INTO the segment from the chosen end."""
    pts=[pt(p) for p in seg.get("points") or []]
    if len(pts)<2: return None
    if which=="last": pts=list(reversed(pts))
    p0=pts[0]; acc=0.0; pl=p0; far=pts[1]
    for q in pts[1:]:
        acc+=vmag(vsub(q,pl)); pl=q; far=q
        if acc>=lever_cm: break
    d=vsub(far,p0)
    return d if vmag(d)>0 else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arms",nargs="+",required=True,metavar="DIR:SAMPLE")
    ap.add_argument("--out",required=True)
    ap.add_argument("--xcut",type=float,default=6.0)
    ap.add_argument("--gap-max",type=float,default=25.0)
    ap.add_argument("--toolkit",default=None)
    a=ap.parse_args()
    os.makedirs(a.out,exist_ok=False)     # fresh tag only (M13)
    tk=d84.find_toolkit(a.toolkit)
    xs,ys=d84.load_muon_range(os.path.join(a.out,"muon_range.json"),tk)
    def ke_to_len(ke_mev):
        # invert monotone range[cm]->KE[MeV]
        if ke_mev<=ys[0]: return xs[0]
        if ke_mev>=ys[-1]: return xs[-1]
        import bisect
        i=bisect.bisect_left(ys,ke_mev)
        y0,y1,x0,x1=ys[i-1],ys[i],xs[i-1],xs[i]
        return x0+(x1-x0)*(ke_mev-y0)/(y1-y0) if y1>y0 else x0

    seen=set(); rowsA=[]; rowsB=[]; nevt=0; nmu=0
    for arm in a.arms:
        d,sample=arm.rsplit(":",1)
        for f in sorted(glob.glob(os.path.join(d,"pr_evt*","calib-pr-evt*.json"))):
            evt=os.path.basename(os.path.dirname(f))[len("pr_evt"):]
            if (sample,evt) in seen: continue
            seen.add((sample,evt)); nevt+=1
            try: J=json.load(open(f))
            except Exception as e:
                print("SKIP unreadable",f,e); continue
            segs=J.get("segments") or []; shws=J.get("showers") or []
            by_sid=collections.defaultdict(list)
            for s in segs: by_sid[s.get("shower_id",-1)].append(s)
            # ---------------- Pop A
            for sh in shws:
                if abs(sh.get("particle_id",0))!=13: continue
                if not sh.get("flag_kinematics"): continue
                nmu+=1
                kr=sh.get("kine_range",0.0)
                mem=[s for s in by_sid.get(sh.get("id"),[])
                     if abs(s.get("particle_id",0))==13]
                L_mem=sum(s.get("length",0.0) for s in mem)
                L_chain=ke_to_len(kr) if kr>0 else 0.0
                # farthest muon-member trajectory point from shower start
                st=pt(sh["start"]); far=0.0
                for s in mem:
                    e=seg_ends(s)
                    if not e: continue
                    far=max(far,vmag(vsub(e[0],st)),vmag(vsub(e[1],st)))
                arrow=vmag(vsub(pt(sh["end"]),st))
                if L_mem-L_chain>10.0:
                    rowsA.append((sample,evt,sh["shower_id"],len(mem),
                        "%.1f"%L_chain,"%.1f"%L_mem,"%.1f"%arrow,"%.1f"%far,
                        "%.1f"%kr,"%.1f"%sh.get("kine_best",0.0),
                        "%.1f"%sh.get("kine_charge",0.0),"%.1f"%sh.get("kine_dQdx",0.0)))
            # ---------------- Pop B
            # muon-shower ends near the cathode
            mu_ends=[]   # (shower, seg, end_xyz, tangent_into_seg, which)
            for sh in shws:
                if abs(sh.get("particle_id",0))!=13: continue
                for s in by_sid.get(sh.get("id"),[]):
                    if abs(s.get("particle_id",0))!=13: continue
                    e=seg_ends(s)
                    if not e: continue
                    for which,p,farp in (("first",e[0],e[1]),("last",e[1],e[0])):
                        if abs(p[0])<a.xcut:
                            mu_ends.append((sh,s,p,end_tangent(s,which),farp))
            if not mu_ends: continue
            # partner ends: bare segments and other showers' members
            partner_ends=[]  # (kind, key, seg, end_xyz, tangent, partner_len, partner_ke)
            for s in segs:
                e=seg_ends(s)
                if not e: continue
                sid=s.get("shower_id",-1)
                if sid==-1: kind,key,ke="bare",s.get("id"),-1.0
                else:
                    osh=[x for x in shws if x.get("id")==sid]
                    kind,key="shower",sid
                    ke=osh[0].get("kine_best",-1.0) if osh else -1.0
                for which,p,farp in (("first",e[0],e[1]),("last",e[1],e[0])):
                    if abs(p[0])<a.xcut:
                        partner_ends.append((kind,key,s,p,end_tangent(s,which),
                                             s.get("length",0.0),ke,farp))
            for sh,ms,mp,mt,mfar in mu_ends:
                for kind,key,s,p,t,plen,pke,pfar in partner_ends:
                    if s.get("id")==ms.get("id"): continue
                    if kind=="shower" and key==sh.get("id"): continue
                    same_clus=1 if s.get("cluster_id")==ms.get("cluster_id") else 0
                    # far ends (away from the seam) on opposite drift sides;
                    # near ends may both jitter onto one side (77978: partner
                    # crosses x=0, near end at +2.17 beside muon end +4.77)
                    if mfar[0]*pfar[0]>0: continue
                    gap=vmag(vsub(p,mp))
                    if gap>a.gap_max: continue
                    gv=vsub(p,mp)
                    # continuity: muon end tangent points INTO the muon; the
                    # continuation leaves along -mt; compare with gap vector
                    # and with the partner tangent (which points INTO partner,
                    # i.e. along the continuation direction).
                    a_gap = vang((-mt[0],-mt[1],-mt[2]),gv) if mt else -1.0
                    a_tan = vang((-mt[0],-mt[1],-mt[2]),(t[0],t[1],t[2])) if (mt and t) else -1.0
                    # partner tangent at ITS near end points away from the gap;
                    # continuation direction through partner = -t... both ends
                    # recorded; a_tan uses tangent INTO partner from near end,
                    # so collinear continuation => a_tan small.
                    rowsB.append((sample,evt,sh["shower_id"],same_clus,
                        "%.1f"%sh.get("kine_best",0.0),ms.get("id"),
                        "%.2f"%mp[0],kind,key,"%.1f"%plen,"%.1f"%pke,
                        "%.2f"%p[0],"%.2f"%mfar[0],"%.2f"%pfar[0],"%.1f"%gap,"%.1f"%a_gap,"%.1f"%a_tan,
                        "%.1f"%vmag(gv)))
    hdrA="sample evt shower_id n_mu_members L_chain_cm L_members_cm arrow_cm farthest_member_cm kine_range kine_best kine_charge kine_dQdx".split()
    hdrB="sample evt shower_id same_clus shower_ke mu_seg_id mu_end_x partner_kind partner_key partner_len partner_ke partner_x mu_far_x partner_far_x gap_cm angle_gap_deg angle_tan_deg gap2_cm".split()
    for name,hdr,rows in (("d84r2-members-gap.tsv",hdrA,rowsA),
                          ("d84r2-cathode-pairs.tsv",hdrB,rowsB)):
        with open(os.path.join(a.out,name),"w") as fp:
            fp.write("\t".join(hdr)+"\n")
            for r in rows: fp.write("\t".join(str(x) for x in r)+"\n")
    print("events scanned:",nevt," muon showers (flag_kinematics):",nmu)
    print("Pop A rows (L_mem-L_chain > 10 cm):",len(rowsA),
          " events:",len({(r[0],r[1]) for r in rowsA}))
    print("Pop B candidate pairs:",len(rowsB),
          " events:",len({(r[0],r[1]) for r in rowsB}))
    return 0

if __name__=="__main__": sys.exit(main())

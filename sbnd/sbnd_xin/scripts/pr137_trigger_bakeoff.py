#!/usr/bin/env python3
# doc pr/137 sec 13 -- the TRIGGER BAKE-OFF.  READ-ONLY.
"""Rank every candidate trigger statistic against two independent positive
classes, and kill the hopeless ones.

  class A (proxy)  onV1c90 showers that absorbed >=2 OFF showers, vs those that
                   absorbed exactly 1.  Large (44 vs 346) but CONTAMINATED --
                   doc pr/137 sec 5: "SINGLE" is not truth, the OFF point
                   over-clusters too, and the owner confirmed it.
  class B (labels) the pr/136 sec 10.1 hand marks: showers holding a member the
                   scanner marked OUT, vs showers the scanner marked with no OUT
                   at all.  Small but REAL, and at segment granularity -- the
                   same granularity the splitter acts at.

Agreement between the two rankings is the check that the proxy is not lying.
Disagreement is a result, not noise (doc pr/137 sec 14 stop-and-ask).

    scripts/pr137_trigger_bakeoff.py     # -> docs/pr/pr137-trigger-bakeoff.tsv
"""
import sys, os, json, glob, re, math, collections, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pr137_lib as L
import pr137_features as F

OUT='docs/pr/pr137-trigger-bakeoff.tsv'
LABEL_TAGS=['em_labels/emscan-0827','em_labels/emscan-0828-agent5']

# ------------------------------------------------------------ class B: labels

def load_marks():
    """{(event, node): {'ins':set, 'outs':set, 'verdict':str}} from the hand scan"""
    out={}
    for t in LABEL_TAGS:
        for f in sorted(glob.glob(t+'/*.json')):
            m=re.search(r'(\d+)', os.path.basename(f))
            if not m: continue
            ev=int(m.group(1))
            try: j=json.load(open(f))
            except Exception: continue
            em=(j.get('em') or {})
            mbs=em.get('marks_by_shower') or {}
            vd=em.get('verdict') or {}
            for sid,marks in mbs.items():
                ins={int(s) for s,v in marks.items() if v=='in'}
                outs={int(s) for s,v in marks.items() if v=='out'}
                out[(ev,int(sid))]=dict(ins=ins, outs=outs,
                                        verdict=(vd.get(str(sid)) if isinstance(vd,dict) else None))
    return out

def classB(marks, arm='f086probe', tag='emprep-136f086'):
    """positives = >=1 OUT mark that is a CURRENT member (actionable for a splitter);
    negatives = marked, but no OUT mark at all.

    The strict (node-id) join is used and its size is REPORTED, because
    em117_score matches showers by charge overlap instead and gets a bigger
    number -- doc pr/137 sec 14 requires the join to be named with the count."""
    rows=[]; stat=collections.Counter()
    byev=collections.defaultdict(dict)
    for (ev,node),m in marks.items(): byev[ev][node]=m
    for ev in sorted(byev):
        mem=L.prep(tag, ev)
        if mem is None: stat['no_sidecar']+=1; continue
        d=L.dump(ev, arm)
        if d is None: stat['no_dump']+=1; continue
        v=L.main_vertex(d)
        if v is None: stat['no_vertex']+=1; continue
        P=L.seg_pts(d); M=L.seg_meta(d); SR=L.shower_recs(d)
        for node,mk in byev[ev].items():
            if node not in mem: stat['node_unresolved']+=1; continue
            ms={s:qq for s,qq in mem[node].items() if s in P}
            if len(ms)<3: stat['too_few_segs']+=1; continue
            act = mk['outs'] & set(ms)
            if act and len(ms)-len(act) >= 1:
                cls='POS'; stat['pos']+=1
            elif not mk['outs']:
                cls='NEG'; stat['neg']+=1
            else:
                stat['out_not_member']+=1; continue
            rows.append(dict(event=ev,node=node,segs=list(ms),ms=ms,mip=L.mip_dqdx(d),
                             Q=sum(max(q,0.) for q in ms.values()),
                             cls=cls, v=v, P=P, M=M, rec=SR.get(node),
                             nanc=0, truth={}, act=act, verdict=mk['verdict']))
    return rows, stat

# ------------------------------------------------------------ scoring

def auc(pos, neg):
    pos=[x for x in pos if x==x]; neg=[x for x in neg if x==x]
    if len(pos)<3 or len(neg)<3: return float('nan'), 0, 0
    a=np.concatenate([pos,neg]); lab=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    o=np.argsort(a); ranks=np.empty(len(a)); ranks[o]=np.arange(1,len(a)+1)
    # average ranks for ties
    srt=a[o]; i=0
    while i<len(srt):
        j=i
        while j+1<len(srt) and srt[j+1]==srt[i]: j+=1
        if j>i: ranks[o[i:j+1]]=np.mean(ranks[o[i:j+1]])
        i=j+1
    n1=len(pos); n0=len(neg)
    u=ranks[lab==1].sum()-n1*(n1+1)/2.0
    return u/(n1*n0), n1, n0

def purity_at_eff(pos, neg, eff=0.50):
    """purity when the cut is placed to keep `eff` of the positives -- the number
    doc pr/137 sec 4 reported (27-36 %), so the comparison is apples to apples.
    Tries both cut directions and keeps the better."""
    pos=np.asarray([x for x in pos if x==x]); neg=np.asarray([x for x in neg if x==x])
    if len(pos)<5 or len(neg)<5: return float('nan'), float('nan')
    best=(float('nan'), float('nan'))
    for sign in (1,-1):
        p=sign*pos; n=sign*neg
        thr=np.percentile(p, 100*(1-eff))
        fp=(p>=thr).sum(); fn=(n>=thr).sum()
        if fp==0: continue
        pur=fp/(fp+fn)
        if not (best[0]==best[0]) or pur>best[0]: best=(pur, sign*thr)
    return best

def main():
    sig=L.profile_sigma_fn()
    # ---- class A
    rowsA=[]
    for ev in L.prep_events('emprep-136onV1c90'):
        rowsA += L.build_population(ev)
    byevA=collections.defaultdict(list)
    for r in rowsA: byevA[r['event']].append(r)
    featA={}
    for r in rowsA:
        if r['cls'] not in ('MERGED','SINGLE'): continue
        f=F.features(r, byevA, sig)
        if f: featA[(r['event'],r['node'])]=(r['cls'],f)
    nA=collections.Counter(c for c,_ in featA.values())
    print("class A (arm-difference proxy): %s"%dict(nA))

    # ---- class B
    marks=load_marks()
    rowsB, stat = classB(marks)
    byevB=collections.defaultdict(list)
    for r in rowsB: byevB[r['event']].append(r)
    featB={}
    for r in rowsB:
        f=F.features(r, byevB, sig)
        if f: featB[(r['event'],r['node'])]=(r['cls'],f)
    nB=collections.Counter(c for c,_ in featB.values())
    print("class B (hand marks, STRICT node-id join): %s"%dict(nB))
    print("  join census: %s"%dict(stat))

    # ---- sub-segment reachability check (doc pr/137 sec 1b-3)
    unreachable=sum(1 for r in rowsB if r['cls']=='POS' and not (set(r['segs'])-r['act']))
    print("\n== sec 1b-3 CHECK: hand-marked cases needing a SUB-SEGMENT cut ==")
    print("  positives where every member is marked OUT (nothing left to keep): %d"%unreachable)
    print("  -> the label space and the splitter's action space are both per-segment;")
    print("     %s"%("no case demands a cut the splitter cannot make." if unreachable==0
                     else "%d case(s) are out of reach and are reported as such."%unreachable))

    n1B_, n0B_ = nB.get('POS',0), nB.get('NEG',0)
    seB = math.sqrt((n1B_+n0B_+1)/(12.0*max(1,n1B_)*max(1,n0B_))) if n1B_ and n0B_ else float('nan')
    print("\n== how much can class B actually decide? ==")
    print("  POS=%d NEG=%d  ->  AUC standard error %.3f, so the 2-sigma band is 0.5 +- %.2f"%(
        n1B_,n0B_,seB,2*seB))
    print("  Any class-B AUC inside that band is consistent with pure noise.  This is the")
    print("  measurement that MOTIVATES the curated scan: the only real-label class we")
    print("  have today cannot confirm or refute the proxy's ranking.")

    allf=[k for fam in F.FAMILY.values() for k in fam]
    lines=["# doc pr/137 sec 13 -- trigger bake-off",
           "# class A = arm-difference proxy (MERGED %d vs SINGLE %d), CONTAMINATED"%(
               nA.get('MERGED',0),nA.get('SINGLE',0)),
           "# class B = hand marks, strict node-id join (POS %d vs NEG %d), REAL but small"%(
               nB.get('POS',0),nB.get('NEG',0)),
           "# purity@50%eff is the doc pr/137 sec 4 metric; its best across three families was 0.27-0.36",
           "family\tfeature\taucA\tpurA50\tnA_pos\taucB\tpurB50\tnB_pos"]

    res=[]
    for fam, keys in F.FAMILY.items():
        for k in keys:
            pA=[f[k] for c,f in featA.values() if c=='MERGED' and k in f]
            nAv=[f[k] for c,f in featA.values() if c=='SINGLE' and k in f]
            pB=[f[k] for c,f in featB.values() if c=='POS' and k in f]
            nBv=[f[k] for c,f in featB.values() if c=='NEG' and k in f]
            aA,n1A,_=auc(pA,nAv); uA,_=purity_at_eff(pA,nAv)
            aB,n1B,_=auc(pB,nBv); uB,_=purity_at_eff(pB,nBv)
            res.append((fam,k,aA,uA,n1A,aB,uB,n1B))
            lines.append("%s\t%s\t%.4g\t%.4g\t%d\t%.4g\t%.4g\t%d"%(fam,k,aA,uA,n1A,aB,uB,n1B))

    def show(title, key, rows_):
        print("\n== %s =="%title)
        print("  %-4s %-16s %7s %8s %6s %7s %8s %6s"%("fam","feature","AUC_A","pur@50A","nA+","AUC_B","pur@50B","nB+"))
        for fam,k,aA,uA,n1A,aB,uB,n1B in rows_:
            print("  %-4s %-16s %7.3f %8.3f %6d %7s %8s %6d"%(fam,k,
                  aA if aA==aA else float('nan'), uA if uA==uA else float('nan'), n1A,
                  ("%.3f"%aB) if aB==aB else "  -  ", ("%.3f"%uB) if uB==uB else "  -  ", n1B))
    ranked=sorted([r for r in res if r[2]==r[2]], key=lambda t: -max(t[2],1-t[2]))
    show("ALL FEATURES, ranked by |AUC-0.5| on class A", None, ranked)

    print("\n== per-family best (class A, purity at 50%% efficiency) ==")
    for fam in ('D','S','C','T','X'):
        cand=[r for r in res if r[0]==fam and r[3]==r[3]]
        if not cand: continue
        b=max(cand,key=lambda t:t[3])
        print("  %-2s  %-16s  purity %.0f%%   AUC %.3f"%(fam,b[1],100*b[3],b[2]))
        lines.append("family_best\t%s:%s\t%.4g\t%.4g\t\t\t\t"%(fam,b[1],b[2],b[3]))

    # ---- 2- and 3-feature cut scans on the strongest survivors -------------
    print("\n== 2-feature cut scan (class A) ==")
    top=[r[1] for r in ranked[:10]]
    keys=sorted(set(top)|{'valley','d2_over_d1','seed_frac','w_pull','gap_scaled','angle','balance'})
    P={k:np.array([f.get(k,np.nan) for c,f in featA.values() if c=='MERGED']) for k in keys}
    N={k:np.array([f.get(k,np.nan) for c,f in featA.values() if c=='SINGLE']) for k in keys}
    def fire(arr,k,thr,sign): return (sign*arr[k]>=sign*thr)
    combos=[]
    for k1,k2 in itertools.combinations(keys,2):
        for s1 in (1,-1):
            g=P[k1][~np.isnan(P[k1])]
            if len(g)<8: continue
            for t1 in np.percentile(g,[10,25,40,55,70]):
                for s2 in (1,-1):
                    g2=P[k2][~np.isnan(P[k2])]
                    if len(g2)<8: continue
                    for t2 in np.percentile(g2,[10,25,40,55,70]):
                        fm=int(np.nansum(fire(P,k1,t1,s1)&fire(P,k2,t2,s2)))
                        if fm<6: continue
                        fs=int(np.nansum(fire(N,k1,t1,s1)&fire(N,k2,t2,s2)))
                        pur=fm/max(1,fm+fs)
                        combos.append((pur,fm,fs,k1,s1,t1,k2,s2,t2))
    combos.sort(reverse=True)
    print("  %-46s %8s %8s %8s"%("rule (>=6 merged fires)","merged","single","purity"))
    seen=set()
    for pur,fm,fs,k1,s1,t1,k2,s2,t2 in combos[:200]:
        key=(k1,k2)
        if key in seen: continue
        seen.add(key)
        nm="%s%s%.3g & %s%s%.3g"%(k1,'>=' if s1>0 else '<=',t1,k2,'>=' if s2>0 else '<=',t2)
        print("  %-46s %8d %8d %7.0f%%"%(nm,fm,fs,100*pur))
        lines.append("combo2\t%s\t%d\t%.4f\tmerged=%d single=%d"%(nm,fm+fs,pur,fm,fs))
        if len(seen)>=8: break

    print("\n== 3-feature cut scan (class A), valley always included ==")
    k3=[k for k in keys if k!='valley']
    vthr=np.percentile(P['valley'][~np.isnan(P['valley'])],[40,55,70])
    out3=[]
    for k1,k2 in itertools.combinations(k3,2):
        for s1 in (1,-1):
            g1=P[k1][~np.isnan(P[k1])]
            if len(g1)<8: continue
            for t1 in np.percentile(g1,[20,40,60]):
                for s2 in (1,-1):
                    g2=P[k2][~np.isnan(P[k2])]
                    if len(g2)<8: continue
                    for t2 in np.percentile(g2,[20,40,60]):
                        for tv in vthr:
                            m=fire(P,k1,t1,s1)&fire(P,k2,t2,s2)&fire(P,'valley',tv,-1)
                            fm=int(np.nansum(m))
                            if fm<6: continue
                            fs=int(np.nansum(fire(N,k1,t1,s1)&fire(N,k2,t2,s2)&fire(N,'valley',tv,-1)))
                            out3.append((fm/max(1,fm+fs),fm,fs,k1,s1,t1,k2,s2,t2,tv))
    out3.sort(reverse=True); seen3=set()
    print("  %-58s %7s %7s %8s"%("rule (>=6 merged fires)","merged","single","purity"))
    for pur,fm,fs,k1,s1,t1,k2,s2,t2,tv in out3:
        key=(k1,k2)
        if key in seen3: continue
        seen3.add(key)
        nm="%s%s%.3g & %s%s%.3g & valley<=%.3g"%(k1,'>=' if s1>0 else '<=',t1,
                                                 k2,'>=' if s2>0 else '<=',t2,tv)
        print("  %-58s %7d %7d %7.0f%%"%(nm,fm,fs,100*pur))
        lines.append("combo3\t%s\t%d\t%.4f\tmerged=%d single=%d"%(nm,fm+fs,pur,fm,fs))
        if len(seen3)>=8: break

    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    open(OUT,'w').write("\n".join(lines)+"\n")
    print("\nwrote %s"%OUT)

if __name__=='__main__': main()

#!/usr/bin/env python3
# doc pr/137 -- shower-splitter feasibility probe.  READ-ONLY; reads the
# emprep-136off2 / emprep-136onV1c90 sidecars and the work-pr136-* dumps.
"""pr/136 -- feasibility probe for a SHOWER SPLITTER (owner request 2026-08-31).

Population: every onV1c90 shower that absorbed >=2 distinct OFF showers.
Target   : the OFF partition (a proxy for truth, not truth -- the owner has
           already said 314838's OFF partition was itself wrong).
Question : does a BLIND criterion recover that partition?
"""
import json, glob, os, collections, math, itertools

def prep(d, ev):
    p=os.path.join('em_display',d,'emprep-evt%d.json'%ev)
    if not os.path.exists(p): return None
    j=json.load(open(p))
    return {int(n): {int(m['seg']): float(m.get('dQ') or 0.0) for m in (e.get('members') or [])}
            for n,e in (j.get('showers') or {}).items()}

def dump(ev):
    for a in glob.glob('work-pr136-off2-*/pr_evt%d/calib-pr-evt%d.json'%(ev,ev)):
        return json.load(open(a))
    return None

def seg_pts(d):
    out={}
    for s in d.get('segments') or ():
        pts=[(p['x'],p['y'],p['z'],p.get('dQ') or 0.0) for p in (s.get('points') or ())]
        if pts: out[int(s['id'])]=pts
    return out

def mindist(A,B):
    m=1e9
    for ax,ay,az,_ in A:
        for bx,by,bz,_ in B:
            d=(ax-bx)**2+(ay-by)**2+(az-bz)**2
            if d<m: m=d
    return math.sqrt(m)

def purity(pred, truth, w):
    """charge-weighted purity of the best 2<->2 assignment"""
    keys=sorted(set(truth.values()))
    if len(keys)!=2: return None
    best=0.0; tot=sum(w.values()) or 1
    for perm in itertools.permutations(keys):
        good=sum(w[s] for s in w if pred.get(s) is not None and perm[pred[s]]==truth[s])
        best=max(best,good)
    return best/tot

evs=sorted(int(os.path.basename(f).split('evt')[1].split('.')[0])
           for f in glob.glob('em_display/emprep-136off2/emprep-evt*.json'))
cases=[]
for ev in evs:
    mo, mn = prep('emprep-136off2',ev), prep('emprep-136onV1c90',ev)
    if not mo or not mn: continue
    owner_off={s:n for n,ms in mo.items() for s in ms}
    for n,ms in mn.items():
        tot=sum(ms.values()) or 1
        c=collections.Counter()
        for s,q in ms.items(): c[owner_off.get(s,-1)]+=q
        big=[o for o,q in c.items() if o>=0 and q/tot>0.05 and q>1e5]
        if len(big)==2: cases.append((ev,n,ms,{s:owner_off.get(s,-1) for s in ms},big,tot))
print("2-way merge cases: %d over %d events"%(len(cases),len({c[0] for c in cases})))

GAPS=[2,4,6,8,12,20]
score={g:[] for g in GAPS}; kmeans=[]; ray=[]; sep=[]
for ev,n,ms,truth,big,tot in cases:
    d=dump(ev)
    if not d: continue
    P=seg_pts(d)
    segs=[s for s in ms if s in P]
    if len(segs)<2: continue
    tr={s:truth[s] for s in segs if truth[s] in big}
    w={s:ms[s] for s in tr}
    if len(set(tr.values()))!=2: continue
    # geometric separation of the two TRUE parts
    A=[p for s in tr if tr[s]==big[0] for p in P[s]]
    B=[p for s in tr if tr[s]==big[1] for p in P[s]]
    if not A or not B: continue
    gapAB=mindist(A[:400],B[:400])
    # centroid directions from the nu vertex
    mv=d.get('main_vertex') or {}
    vx,vy,vz=mv.get('x',0.0),mv.get('y',0.0),mv.get('z',0.0)
    def cdir(X):
        sw=sum(p[3] for p in X) or 1
        cx=sum(p[0]*p[3] for p in X)/sw; cy=sum(p[1]*p[3] for p in X)/sw; cz=sum(p[2]*p[3] for p in X)/sw
        u=(cx-vx,cy-vy,cz-vz); m=math.sqrt(sum(t*t for t in u)) or 1
        return (u[0]/m,u[1]/m,u[2]/m)
    ua,ub=cdir(A),cdir(B)
    ang=math.degrees(math.acos(max(-1,min(1,sum(ua[i]*ub[i] for i in range(3))))))
    sep.append((ev,n,gapAB,ang,tot))
    # (a) connected components at gap G
    for G in GAPS:
        parent={s:s for s in segs}
        def find(x):
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        for i in range(len(segs)):
            for j in range(i+1,len(segs)):
                if mindist(P[segs[i]][:200],P[segs[j]][:200])<G:
                    a,b=find(segs[i]),find(segs[j])
                    if a!=b: parent[a]=b
        comp=collections.defaultdict(list)
        for s in segs: comp[find(s)].append(s)
        order=sorted(comp.values(),key=lambda L:-sum(ms[s] for s in L))
        pred={s:(0 if s in set(order[0]) else 1) for s in segs} if len(order)>=2 else {s:0 for s in segs}
        p=purity(pred,tr,w)
        if p is not None: score[G].append(p)
    # (b) 2-means on charge-weighted segment centroids
    cent={s:tuple(sum(p[k]*p[3] for p in P[s])/(sum(p[3] for p in P[s]) or 1) for k in range(3)) for s in segs}
    a0=max(segs,key=lambda s: ms[s]); a1=max(segs,key=lambda s: sum((cent[s][k]-cent[a0][k])**2 for k in range(3)))
    C=[cent[a0],cent[a1]]
    for _ in range(12):
        asg={s:(0 if sum((cent[s][k]-C[0][k])**2 for k in range(3))<sum((cent[s][k]-C[1][k])**2 for k in range(3)) else 1) for s in segs}
        for c in (0,1):
            pts=[s for s in segs if asg[s]==c]
            if pts: C[c]=tuple(sum(cent[s][k]*ms[s] for s in pts)/(sum(ms[s] for s in pts) or 1) for k in range(3))
    p=purity(asg,tr,w)
    if p is not None: kmeans.append(p)
    # (c) 2-means on unit directions from the nu vertex (angular)
    dirs={}
    for s in segs:
        sw=sum(p[3] for p in P[s]) or 1
        c=[sum(p[k]*p[3] for p in P[s])/sw for k in range(3)]
        u=[c[0]-vx,c[1]-vy,c[2]-vz]; m=math.sqrt(sum(t*t for t in u)) or 1
        dirs[s]=[t/m for t in u]
    b0=max(segs,key=lambda s: ms[s])
    b1=min(segs,key=lambda s: sum(dirs[s][k]*dirs[b0][k] for k in range(3)))
    D=[dirs[b0][:],dirs[b1][:]]
    for _ in range(12):
        asg2={s:(0 if sum(dirs[s][k]*D[0][k] for k in range(3))>sum(dirs[s][k]*D[1][k] for k in range(3)) else 1) for s in segs}
        for c in (0,1):
            pts=[s for s in segs if asg2[s]==c]
            if pts:
                v=[sum(dirs[s][k]*ms[s] for s in pts) for k in range(3)]
                m=math.sqrt(sum(t*t for t in v)) or 1
                D[c]=[t/m for t in v]
    p=purity(asg2,tr,w)
    if p is not None: ray.append(p)

def rep(name,L):
    if not L: print("  %-28s (none)"%name); return
    L=sorted(L)
    med=L[len(L)//2]
    print("  %-28s n=%3d  median purity %.3f   >=0.90: %d (%.0f%%)   >=0.99: %d"%(
        name,len(L),med,sum(1 for x in L if x>=0.90),100*sum(1 for x in L if x>=0.90)/len(L),
        sum(1 for x in L if x>=0.99)))
print("\nBLIND SPLITTER RECOVERY OF THE OFF PARTITION (charge-weighted purity)")
for G in GAPS: rep("connected components %2dcm"%G, score[G])
rep("2-means on positions", kmeans)
rep("2-means on vertex rays", ray)
print("\nGEOMETRY OF THE TRUE 2-PART SPLIT (what a splitter would have to see)")
sep.sort(key=lambda x:-x[4])
print("  %-8s %-8s %9s %9s %10s"%("event","shower","gap_cm","angle_deg","q"))
for ev,n,g,a,t in sep[:18]:
    print("  %-8d %-8d %9.1f %9.1f %10.2e"%(ev,n,g,a,t))
gs=sorted(x[2] for x in sep); an=sorted(x[3] for x in sep)
if gs: print("\n  gap    median %.1f cm  (min %.1f, max %.1f);  frac >2cm: %.0f%%"%(gs[len(gs)//2],gs[0],gs[-1],100*sum(1 for x in gs if x>2)/len(gs)))
if an: print("  angle  median %.1f deg (min %.1f, max %.1f);  frac >10deg: %.0f%%"%(an[len(an)//2],an[0],an[-1],100*sum(1 for x in an if x>10)/len(an)))

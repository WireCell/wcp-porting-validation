#!/usr/bin/env python3
# doc pr/137 -- which reference point the direction split should use.  READ-ONLY.
"""Which REFERENCE POINT should the direction split use?  The owner's design says
"direction based from the targeted vertex"; the probe so far used the nu main
vertex.  Compare it against the shower's own start point and against the
shower's start VERTEX, on the same 44 two-way merges."""
import json, glob, os, collections, math, itertools
def prep(d, ev):
    p=os.path.join('em_display',d,'emprep-evt%d.json'%ev)
    if not os.path.exists(p): return None
    j=json.load(open(p))
    return {int(n): {int(m['seg']): float(m.get('dQ') or 0.0) for m in (e.get('members') or [])}
            for n,e in (j.get('showers') or {}).items()}
def dump(ev):
    for a in glob.glob('work-pr136-onV1c90-*/pr_evt%d/calib-pr-evt%d.json'%(ev,ev)): return json.load(open(a))
def seg_pts(d):
    out={}
    for s in d.get('segments') or ():
        pts=[(p['x'],p['y'],p['z'],p.get('dQ') or 0.0) for p in (s.get('points') or ())]
        if pts: out[int(s['id'])]=pts
    return out
def purity(pred,tr,w):
    ks=sorted(set(tr.values()))
    if len(ks)!=2: return None
    best=0.0; tot=sum(w.values()) or 1
    for perm in itertools.permutations(ks):
        best=max(best,sum(w[s] for s in w if perm[pred[s]]==tr[s]))
    return best/tot
def raysplit(P,ms,segs,ref):
    rx,ry,rz=ref; dirs={}
    for s in segs:
        sw=sum(p[3] for p in P[s]) or 1
        c=[sum(p[k]*p[3] for p in P[s])/sw for k in range(3)]
        u=[c[0]-rx,c[1]-ry,c[2]-rz]; m=math.sqrt(sum(t*t for t in u)) or 1
        dirs[s]=[t/m for t in u]
    b0=max(segs,key=lambda s:ms[s])
    b1=min(segs,key=lambda s:sum(dirs[s][k]*dirs[b0][k] for k in range(3)))
    D=[dirs[b0][:],dirs[b1][:]];asg={}
    for _ in range(15):
        asg={s:(0 if sum(dirs[s][k]*D[0][k] for k in range(3))>sum(dirs[s][k]*D[1][k] for k in range(3)) else 1) for s in segs}
        for c in (0,1):
            pts=[s for s in segs if asg[s]==c]
            if pts:
                v=[sum(dirs[s][k]*ms[s] for s in pts) for k in range(3)];m=math.sqrt(sum(t*t for t in v)) or 1
                D[c]=[t/m for t in v]
    return asg
evs=sorted(int(os.path.basename(f).split('evt')[1].split('.')[0])
           for f in glob.glob('em_display/emprep-136off2/emprep-evt*.json'))
res=collections.defaultdict(list)
for ev in evs:
    mo,mn=prep('emprep-136off2',ev),prep('emprep-136onV1c90',ev)
    if not mo or not mn: continue
    owner_off={s:n for n,ms in mo.items() for s in ms}
    d=None;P=None;shw={}
    for n,ms in mn.items():
        tot=sum(ms.values()) or 1
        c=collections.Counter()
        for s,q in ms.items(): c[owner_off.get(s,-1)]+=q
        big=[o for o,q in c.items() if o>=0 and q/tot>0.05 and q>1e5]
        if len(big)!=2: continue
        if d is None:
            d=dump(ev)
            if not d: break
            P=seg_pts(d); shw={int(x['id']):x for x in (d.get('showers') or [])}
        segs=[s for s in ms if s in P]
        tr={s:owner_off.get(s,-1) for s in segs if owner_off.get(s,-1) in big}
        w={s:ms[s] for s in tr}
        if len(set(tr.values()))!=2 or len(segs)<2: continue
        mv=d.get('main_vertex') or {}
        refs={'nu main vertex':(mv.get('x',0.),mv.get('y',0.),mv.get('z',0.))}
        st=shw.get(n,{}).get('start')
        if st: refs["shower's own start"]=(st['x'],st['y'],st['z'])
        # the charge-weighted centroid of the whole object, as a null reference
        allp=[p for s in segs for p in P[s]]
        sw=sum(p[3] for p in allp) or 1
        refs['object centroid']=tuple(sum(p[k]*p[3] for p in allp)/sw for k in range(3))
        for nm,r in refs.items():
            pr=purity(raysplit(P,ms,segs,r),tr,w)
            if pr is not None: res[nm].append(pr)
print("REFERENCE POINT for the direction split (44 two-way merges, charge-weighted purity)")
for nm,L in sorted(res.items(), key=lambda kv:-sorted(kv[1])[len(kv[1])//2]):
    L=sorted(L)
    print("  %-22s n=%3d  median %.3f   >=0.90 %2d (%3.0f%%)   >=0.99 %2d"%(
        nm,len(L),L[len(L)//2],sum(1 for x in L if x>=0.90),100*sum(1 for x in L if x>=0.90)/len(L),
        sum(1 for x in L if x>=0.99)))

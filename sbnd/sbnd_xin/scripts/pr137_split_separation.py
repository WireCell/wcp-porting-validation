#!/usr/bin/env python3
# doc pr/137 -- shower-splitter feasibility probe.  READ-ONLY; reads the
# emprep-136off2 / emprep-136onV1c90 sidecars and the work-pr136-* dumps.
"""The decisive test: does the ray-split statistic SEPARATE merged showers from
genuine single showers?  A splitter that fires on real showers is worse than no
splitter, so the accept test -- not the split -- is the whole problem."""
import json, glob, os, collections, math

def prep(d, ev):
    p=os.path.join('em_display',d,'emprep-evt%d.json'%ev)
    if not os.path.exists(p): return None
    j=json.load(open(p))
    return {int(n): {int(m['seg']): float(m.get('dQ') or 0.0) for m in (e.get('members') or [])}
            for n,e in (j.get('showers') or {}).items()}

def dump(ev, arm):
    for a in glob.glob('work-pr136-%s-*/pr_evt%d/calib-pr-evt%d.json'%(arm,ev,ev)):
        return json.load(open(a))
    return None

def stats(P, ms, vtx, segs):
    """ray 2-means -> (opening angle between the two ray centroids, charge balance,
    min-gap between the two parts)"""
    vx,vy,vz=vtx
    dirs={}; cen={}
    for s in segs:
        sw=sum(p[3] for p in P[s]) or 1
        c=[sum(p[k]*p[3] for p in P[s])/sw for k in range(3)]
        cen[s]=c
        u=[c[0]-vx,c[1]-vy,c[2]-vz]; m=math.sqrt(sum(t*t for t in u)) or 1
        dirs[s]=[t/m for t in u]
    b0=max(segs,key=lambda s: ms[s])
    b1=min(segs,key=lambda s: sum(dirs[s][k]*dirs[b0][k] for k in range(3)))
    D=[dirs[b0][:],dirs[b1][:]]
    asg={}
    for _ in range(15):
        asg={s:(0 if sum(dirs[s][k]*D[0][k] for k in range(3))>sum(dirs[s][k]*D[1][k] for k in range(3)) else 1) for s in segs}
        for c in (0,1):
            pts=[s for s in segs if asg[s]==c]
            if pts:
                v=[sum(dirs[s][k]*ms[s] for s in pts) for k in range(3)]
                m=math.sqrt(sum(t*t for t in v)) or 1
                D[c]=[t/m for t in v]
    q0=sum(ms[s] for s in segs if asg[s]==0); q1=sum(ms[s] for s in segs if asg[s]==1)
    if q0<=0 or q1<=0: return None
    ang=math.degrees(math.acos(max(-1,min(1,sum(D[0][k]*D[1][k] for k in range(3))))))
    bal=min(q0,q1)/(q0+q1)
    A=[p for s in segs if asg[s]==0 for p in P[s]][:400]
    B=[p for s in segs if asg[s]==1 for p in P[s]][:400]
    gap=1e9
    for ax,ay,az,_ in A:
        for bx,by,bz,_ in B:
            d=(ax-bx)**2+(ay-by)**2+(az-bz)**2
            if d<gap: gap=d
    return ang, bal, math.sqrt(gap), q0+q1

def seg_pts(d):
    out={}
    for s in d.get('segments') or ():
        pts=[(p['x'],p['y'],p['z'],p.get('dQ') or 0.0) for p in (s.get('points') or ())]
        if pts: out[int(s['id'])]=pts
    return out

evs=sorted(int(os.path.basename(f).split('evt')[1].split('.')[0])
           for f in glob.glob('em_display/emprep-136off2/emprep-evt*.json'))
MERGED=[]; SINGLE=[]
for ev in evs:
    mo, mn = prep('emprep-136off2',ev), prep('emprep-136onV1c90',ev)
    if not mo or not mn: continue
    owner_off={s:n for n,ms in mo.items() for s in ms}
    d=None; P=None
    for n,ms in mn.items():
        tot=sum(ms.values()) or 1
        if tot < 1e6 or len(ms) < 3: continue
        c=collections.Counter()
        for s,q in ms.items(): c[owner_off.get(s,-1)]+=q
        big=[o for o,q in c.items() if o>=0 and q/tot>0.05 and q>1e5]
        if d is None:
            d=dump(ev,'onV1c90')
            if not d: break
            P=seg_pts(d)
        segs=[s for s in ms if s in P]
        if len(segs)<3: continue
        mv=d.get('main_vertex') or {}
        st=stats(P, ms, (mv.get('x',0.),mv.get('y',0.),mv.get('z',0.)), segs)
        if not st: continue
        (MERGED if len(big)>=2 else SINGLE).append((ev,n)+st)

def q(L,i):
    v=sorted(x[i] for x in L); n=len(v)
    return v[n//10], v[n//2], v[9*n//10]
print("ray-split statistic on onV1c90 showers with q>1e6 and >=3 segments")
print("  MERGED (>=2 OFF showers inside): n=%d"%len(MERGED))
print("  SINGLE (1 OFF shower inside)   : n=%d"%len(SINGLE))
for name,L in (("MERGED",MERGED),("SINGLE",SINGLE)):
    if not L: continue
    a=q(L,2); b=q(L,3); g=q(L,4)
    print("  %-7s angle  p10/med/p90 = %5.1f / %5.1f / %5.1f deg"%(name,)+"" if False else
          "  %-7s angle  p10/med/p90 = %5.1f / %5.1f / %5.1f deg"%(name,a[0],a[1],a[2]))
    print("          balance      p10/med/p90 = %5.3f / %5.3f / %5.3f"%(b[0],b[1],b[2]))
    print("          gap cm       p10/med/p90 = %5.1f / %5.1f / %5.1f"%(g[0],g[1],g[2]))
print("\nSEPARATION POWER of simple accept tests (fires = would be split):")
for ang,bal,gap in ((10,0.10,0),(15,0.15,0),(20,0.15,0),(15,0.15,2),(20,0.20,2),(25,0.20,4),(30,0.25,4)):
    fm=sum(1 for x in MERGED if x[2]>ang and x[3]>bal and x[4]>gap)
    fs=sum(1 for x in SINGLE if x[2]>ang and x[3]>bal and x[4]>gap)
    print("  angle>%2d bal>%.2f gap>%dcm :  merged %2d/%2d (%3.0f%%)   single %3d/%3d (%3.0f%%)  ratio %.1f"%(
        ang,bal,gap,fm,len(MERGED),100*fm/max(1,len(MERGED)),fs,len(SINGLE),100*fs/max(1,len(SINGLE)),
        (fm/max(1,len(MERGED)))/max(1e-9,(fs/max(1,len(SINGLE))))))

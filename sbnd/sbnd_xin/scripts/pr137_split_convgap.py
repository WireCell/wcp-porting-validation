#!/usr/bin/env python3
# doc pr/137 -- shower-splitter feasibility probe.  READ-ONLY; reads the
# emprep-136off2 / emprep-136onV1c90 sidecars and the work-pr136-* dumps.
"""The remaining trigger lead: TWO CONVERSION GAPS.
Two photons from a pi0 each convert AWAY from the nu vertex, so BOTH parts of a
true two-gamma object start at a distance.  An artificial split of one real
shower puts the shower's own start in one part, so that part's vertex distance
is ~0.  Statistic: min over the two parts of (vertex -> nearest point of part)."""
import json, glob, os, collections, math

def prep(d, ev):
    p=os.path.join('em_display',d,'emprep-evt%d.json'%ev)
    if not os.path.exists(p): return None
    j=json.load(open(p))
    return {int(n): {int(m['seg']): float(m.get('dQ') or 0.0) for m in (e.get('members') or [])}
            for n,e in (j.get('showers') or {}).items()}
def dump(ev, arm):
    for a in glob.glob('work-pr136-%s-*/pr_evt%d/calib-pr-evt%d.json'%(arm,ev,ev)): return json.load(open(a))
    return None
def seg_pts(d):
    out={}
    for s in d.get('segments') or ():
        pts=[(p['x'],p['y'],p['z'],p.get('dQ') or 0.0) for p in (s.get('points') or ())]
        if pts: out[int(s['id'])]=pts
    return out

evs=sorted(int(os.path.basename(f).split('evt')[1].split('.')[0])
           for f in glob.glob('em_display/emprep-136off2/emprep-evt*.json'))
M=[];S=[]
for ev in evs:
    mo,mn=prep('emprep-136off2',ev),prep('emprep-136onV1c90',ev)
    if not mo or not mn: continue
    owner_off={s:n for n,ms in mo.items() for s in ms}
    d=None;P=None
    for n,ms in mn.items():
        tot=sum(ms.values()) or 1
        if tot<1e6 or len(ms)<3: continue
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
        vx,vy,vz=mv.get('x',0.),mv.get('y',0.),mv.get('z',0.)
        dirs={}
        for s in segs:
            sw=sum(p[3] for p in P[s]) or 1
            cc=[sum(p[k]*p[3] for p in P[s])/sw for k in range(3)]
            u=[cc[0]-vx,cc[1]-vy,cc[2]-vz];m=math.sqrt(sum(t*t for t in u)) or 1
            dirs[s]=[t/m for t in u]
        b0=max(segs,key=lambda s:ms[s])
        b1=min(segs,key=lambda s:sum(dirs[s][k]*dirs[b0][k] for k in range(3)))
        D=[dirs[b0][:],dirs[b1][:]];asg={}
        for _ in range(15):
            asg={s:(0 if sum(dirs[s][k]*D[0][k] for k in range(3))>sum(dirs[s][k]*D[1][k] for k in range(3)) else 1) for s in segs}
            for cc in (0,1):
                pts=[s for s in segs if asg[s]==cc]
                if pts:
                    v=[sum(dirs[s][k]*ms[s] for s in pts) for k in range(3)];m=math.sqrt(sum(t*t for t in v)) or 1
                    D[cc]=[t/m for t in v]
        q0=sum(ms[s] for s in segs if asg[s]==0);q1=sum(ms[s] for s in segs if asg[s]==1)
        if q0<=0 or q1<=0: continue
        dv=[]
        for cc in (0,1):
            best=1e9
            for s in segs:
                if asg[s]!=cc: continue
                for x,y,z,_ in P[s]:
                    dd=(x-vx)**2+(y-vy)**2+(z-vz)**2
                    if dd<best: best=dd
            dv.append(math.sqrt(best))
        ang=math.degrees(math.acos(max(-1,min(1,sum(D[0][j]*D[1][j] for j in range(3))))))
        bal=min(q0,q1)/(q0+q1)
        (M if len(big)>=2 else S).append((ev,n,min(dv),max(dv),ang,bal))
def qs(L,i):
    v=sorted(x[i] for x in L);n=len(v);return v[n//10],v[n//2],v[9*n//10]
print("TWO-CONVERSION-GAP statistic  (MERGED n=%d, SINGLE n=%d)"%(len(M),len(S)))
for nm,L in (("MERGED",M),("SINGLE",S)):
    a=qs(L,2);b=qs(L,3)
    print("  %-7s min(part vertex-gap) p10/med/p90 = %5.1f / %5.1f / %5.1f cm"%(nm,a[0],a[1],a[2]))
    print("          max(part vertex-gap) p10/med/p90 = %5.1f / %5.1f / %5.1f cm"%(b[0],b[1],b[2]))
print("\nACCEPT TESTS with the conversion-gap requirement:")
for dmin,ang,bal in ((3,10,0.10),(5,10,0.10),(5,15,0.15),(8,15,0.15),(10,15,0.10),(10,20,0.15),(15,10,0.10)):
    fm=sum(1 for x in M if x[2]>dmin and x[4]>ang and x[5]>bal)
    fs=sum(1 for x in S if x[2]>dmin and x[4]>ang and x[5]>bal)
    print("  min-gap>%2dcm angle>%2d bal>%.2f :  merged %2d/%2d (%3.0f%%)  single %3d/%3d (%3.0f%%)  ratio %4.1f  purity %3.0f%%"%(
        dmin,ang,bal,fm,len(M),100*fm/max(1,len(M)),fs,len(S),100*fs/max(1,len(S)),
        (fm/max(1,len(M)))/max(1e-9,fs/max(1,len(S))),100*fm/max(1,fm+fs)))

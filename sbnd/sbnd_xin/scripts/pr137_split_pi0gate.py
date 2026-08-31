#!/usr/bin/env python3
# doc pr/137 -- shower-splitter feasibility probe.  READ-ONLY; reads the
# emprep-136off2 / emprep-136onV1c90 sidecars and the work-pr136-* dumps.
"""Can a pi0-MASS gate be the splitter's trigger?  For every candidate 2-way ray
split, form m = sqrt(4 E1 E2) sin(theta/2) from the two parts' charge-weighted
centroid directions off the nu vertex, and ask how often it lands in (100,160).
A trigger is only useful if MERGED showers land there much more often than
genuine SINGLE ones."""
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

def seg_pts(d):
    out={}
    for s in d.get('segments') or ():
        pts=[(p['x'],p['y'],p['z'],p.get('dQ') or 0.0) for p in (s.get('points') or ())]
        if pts: out[int(s['id'])]=pts
    return out

evs=sorted(int(os.path.basename(f).split('evt')[1].split('.')[0])
           for f in glob.glob('em_display/emprep-136off2/emprep-evt*.json'))
M=[]; S=[]
for ev in evs:
    mo, mn = prep('emprep-136off2',ev), prep('emprep-136onV1c90',ev)
    if not mo or not mn: continue
    owner_off={s:n for n,ms in mo.items() for s in ms}
    d=None; P=None; kine={}
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
            kine={int(x['id']): float(x.get('kine_charge') or 0.0) for x in (d.get('showers') or [])}
        segs=[s for s in ms if s in P]
        if len(segs)<3: continue
        mv=d.get('main_vertex') or {}
        vx,vy,vz=mv.get('x',0.),mv.get('y',0.),mv.get('z',0.)
        dirs={};
        for s in segs:
            sw=sum(p[3] for p in P[s]) or 1
            cc=[sum(p[k]*p[3] for p in P[s])/sw for k in range(3)]
            u=[cc[0]-vx,cc[1]-vy,cc[2]-vz]; m=math.sqrt(sum(t*t for t in u)) or 1
            dirs[s]=[t/m for t in u]
        b0=max(segs,key=lambda s: ms[s])
        b1=min(segs,key=lambda s: sum(dirs[s][k]*dirs[b0][k] for k in range(3)))
        D=[dirs[b0][:],dirs[b1][:]]; asg={}
        for _ in range(15):
            asg={s:(0 if sum(dirs[s][k]*D[0][k] for k in range(3))>sum(dirs[s][k]*D[1][k] for k in range(3)) else 1) for s in segs}
            for cc in (0,1):
                pts=[s for s in segs if asg[s]==cc]
                if pts:
                    v=[sum(dirs[s][k]*ms[s] for s in pts) for k in range(3)]
                    m=math.sqrt(sum(t*t for t in v)) or 1
                    D[cc]=[t/m for t in v]
        q0=sum(ms[s] for s in segs if asg[s]==0); q1=sum(ms[s] for s in segs if asg[s]==1)
        if q0<=0 or q1<=0: continue
        k = (kine.get(n,0.0)/tot) if tot>0 and kine.get(n,0.0)>0 else 0.0
        if k<=0: continue
        E0,E1=k*q0,k*q1
        th=math.degrees(math.acos(max(-1,min(1,sum(D[0][j]*D[1][j] for j in range(3))))))
        mass=math.sqrt(4*E0*E1)*math.sin(math.radians(th)/2)
        (M if len(big)>=2 else S).append((ev,n,mass,th,min(E0,E1),max(E0,E1)))
def frac(L,lo,hi,emin=0.0):
    k=[x for x in L if lo<=x[2]<=hi and x[4]>=emin]
    return len(k),100.0*len(k)/max(1,len(L))
print("pi0-MASS GATE as the splitter trigger  (candidate ray split of every onV1c90 shower, q>1e6, >=3 seg)")
print("  MERGED n=%d   SINGLE n=%d"%(len(M),len(S)))
for lo,hi,emin,lab in ((100,160,0,"(100,160) MeV"),(100,160,15,"(100,160) + min E>15 MeV"),
                       (110,160,20,"(110,160) + min E>20 MeV"),(100,180,30,"(100,180) + min E>30 MeV")):
    fm,pm=frac(M,lo,hi,emin); fs,ps=frac(S,lo,hi,emin)
    print("  %-28s merged %2d/%2d (%3.0f%%)   single %3d/%3d (%3.0f%%)   ratio %.1f   purity %.0f%%"%(
        lab,fm,len(M),pm,fs,len(S),ps,pm/max(1e-9,ps),100*fm/max(1,fm+fs)))
print("\n  merged cases that PASS (100,160)+minE>15:")
for x in sorted([x for x in M if 100<=x[2]<=160 and x[4]>15],key=lambda y:-y[5])[:12]:
    print("    evt %-8d shw %-8d m=%6.1f  theta=%5.1f  E=%.0f/%.0f"%(x[0],x[1],x[2],x[3],x[5],x[4]))

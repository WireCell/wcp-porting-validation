import re,glob,sys,statistics as st
tag=sys.argv[1] if len(sys.argv)>1 else 'd28r2fp5'
rows=[]; dc=[]
for log in sorted(glob.glob(f'work/*_{tag}/wct_pr_*.log')):
    ev=log.split('/')[1].replace('_'+tag,'')
    cur=None
    for line in open(log,errors='replace'):
        m=re.search(r'selected main cluster (\d+) \(t0 ([\d.-]+) us, L ([\d.]+) cm, (\d+) associated',line)
        if m:
            cur={'ev':ev,'cid':int(m.group(1)),'L':float(m.group(3)),'assoc':int(m.group(4)),'off_ms':0,'prod_ms':0,'refine_ms':0,'npts':0}
            rows.append(cur); continue
        if cur is None: continue
        m=re.search(r'\[dual-off\] pass TOTAL took ([\d.]+) ms',line)
        if m: cur['off_ms']=float(m.group(1)); continue
        m=re.search(r'timing: main_cluster initial PR took ([\d.]+) ms',line)
        if m: cur['prod_ms']=float(m.group(1)); continue
        m=re.search(r'timing: improve_vertex \+ examine_direction took ([\d.]+) ms',line)
        if m: cur['refine_ms']=float(m.group(1)); continue
        m=re.search(r'dQ/dx: .*?(\d+) trajectory point',line)
        m=re.search(r'dual_chain: mode=(\w+) transfer=(\w+) off_vertex=(\w+) off_ms=(\d+) nearest_d=([\d.]+)cm agree=(\w+) transferred=(\w+) prod_route=(\w+)',line)
        if m: dc.append((ev,cur['cid'],m.group(5),m.group(6),m.group(7),m.group(8)))
n=len(rows)
off=sum(r['off_ms'] for r in rows)/1000; prod=sum(r['prod_ms'] for r in rows)/1000
print(f"candidates {n}; OFF-pass sum {off:.0f} s; production initial PR sum {prod:.0f} s; refine sum {sum(r['refine_ms'] for r in rows)/1000:.0f} s")
rows.sort(key=lambda r:-(r['off_ms']+r['prod_ms']))
print("top candidates: ev cid L_cm off_s prod_s")
for r in rows[:15]: print(f"  {r['ev']} {r['cid']} {r['L']:.0f} {r['off_ms']/1000:.1f} {r['prod_ms']/1000:.1f}")
# cost vs length bins
bins=[(0,100),(100,200),(200,400),(400,600),(600,900),(900,1e9)]
print("L bin: n, sum(off+prod) s, share, median s/cand")
tot=off+prod
for lo,hi in bins:
    sel=[(r['off_ms']+r['prod_ms'])/1000 for r in rows if lo<=r['L']<hi]
    if sel: print(f"  [{lo},{hi}) n={len(sel)} sum={sum(sel):.0f} share={sum(sel)/tot:.2f} median={st.median(sel):.1f} max={max(sel):.1f}")
print("dual chain decisions:",len(dc))
from collections import Counter
print(" agree:",Counter(d[3] for d in dc)," transferred:",Counter(d[4] for d in dc)," prod_route:",Counter(d[5] for d in dc))
ds=sorted(float(d[2]) for d in dc); 
if ds: print(" nearest_d cm: median",st.median(ds),"p90",ds[int(0.9*len(ds))],"max",max(ds), "n(d>2cm)",sum(1 for x in ds if x>2))
for d in dc:
    if d[4]=='true' or d[4]=='1': print("  TRANSFER",d)

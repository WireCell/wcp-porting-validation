#!/usr/bin/env python3
"""Per-cluster STM/TGM verdict flips between two PR arms (log-derived)."""
import glob, os, re, sys
from collections import Counter
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # <repo>/pdvd
PAT=re.compile(r"TaggerCheckSTM: cluster (\d+) . STM=(\d) TGM=(\d)")
def arm(tag):
    out={}
    for d in sorted(glob.glob(os.path.join(PDVD,'work','*_%s'%tag))):
        base=os.path.basename(d)[:-(len(tag)+1)]
        run,idx=base.rsplit('_',1)
        logs=glob.glob(os.path.join(d,'wct_pr_*.log'))
        if not logs: continue
        v={}
        for line in open(logs[0],errors='replace'):
            m=PAT.search(line)
            if m: v[int(m.group(1))]=(int(m.group(2)),int(m.group(3)))
        out[(run,idx)]=v
    return out
A,B=arm(sys.argv[1]),arm(sys.argv[2])
rows=[]
cnt=Counter()
for k in sorted(set(A)&set(B)):
    a,b=A[k],B[k]
    for cid in sorted(set(a)|set(b)):
        va,vb=a.get(cid),b.get(cid)
        if va==vb: continue
        rows.append((k,cid,va,vb))
        cnt[(va,vb)]+=1
print("flip census (A=%s -> B=%s), pairs are (STM,TGM); None = not evaluated"%(sys.argv[1],sys.argv[2]))
for k,v in sorted(cnt.items(),key=lambda x:-x[1]):
    print("  %-22s -> %-22s  %4d"%(k[0],k[1],v))
print("total changed clusters:",len(rows))
per=Counter((r[0]) for r in rows)
print("\ntop events by changed clusters:")
for k,v in per.most_common(12): print("   %s/%s  %d"%(k[0],k[1],v))
sel=sys.argv[3] if len(sys.argv)>3 else None
if sel:
    print("\nchanged clusters in %s:"%sel)
    run,idx=sel.split('/')
    for r in rows:
        if r[0]==(run,idx): print("   cluster %-4d %s -> %s"%(r[1],r[2],r[3]))

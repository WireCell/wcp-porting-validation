import os, re, collections, json
R="/home/xqian/toolkit-dev/wcp-porting-img"; W=f"{R}/pdvd/work"
ARM=re.compile(r"^\d{6}_\d+_(.+)$")
PROD={"d103vflip","d103vprod1","q29flip","q29stm","p100flip","pvdimg","p98von"}
OPEN={"p101q"}
PEER={"d113vbase","d113vnone","d111vst","d103v0","d103v1","d101vnew"}
SUB={"d27fresh","keep","d51vclus","d41prov","d39r2prov","(bare)"}
SCAN=set(json.load(open(f"{R}/pdhd/scripts/retire/scan_arms_20260916e.json")).get("pdvd",[]))
RATIO={"calib-pr":6.90,"calib-clus":6.16,"log":18.5,"csv":25.0}
CLS=[("calib-pr",r"^calib-pr-evt\d+\.json$"),("calib-clus",r"^calib-evt\d+\.json$"),("log",r"^wct_.*\.log$"),("csv",r"^(gpu_mem|pr_rss|clus_rss)_.*\.csv$")]
# inbound symlink targets (realpath, inode) from everywhere relevant
targets=set()
for root in (W, f"{R}/pdhd/work", f"{R}/sbnd/sbnd_xin", "/home/xqian/tmp"):
    for cur,subs,fs in os.walk(root):
        if root.endswith("/tmp") and cur[len(root):].count("/")>=4: subs[:]=[]
        for e in fs+subs:
            p=os.path.join(cur,e)
            if os.path.islink(p):
                try: st=os.stat(p); targets.add((st.st_dev,st.st_ino))
                except OSError: pass
        subs[:]=[s for s in subs if not os.path.islink(os.path.join(cur,s)) and s!="archive"]
agg=collections.defaultdict(lambda: collections.defaultdict(lambda:[0,0,0,0]))  # cat -> cls -> [files,bytes,linked,multilink]
famcat={}
for d in os.listdir(W):
    p=os.path.join(W,d)
    if os.path.islink(p) or not os.path.isdir(p): continue
    m=ARM.match(d); f=m.group(1) if m else "(bare)"
    cat=("production" if f in PROD else "open" if f in OPEN else "peer-held" if f in PEER else "substrate" if f in SUB
         else "scan-source" if f in SCAN else "other:"+f)
    famcat[f]=cat
    for cur,subs,fs in os.walk(p):
        for fn in fs:
            q=os.path.join(cur,fn)
            if os.path.islink(q): continue
            for k,pat in CLS:
                if re.match(pat,fn):
                    st=os.lstat(q); e=agg[cat][k]; e[0]+=1; e[1]+=st.st_size
                    if (st.st_dev,st.st_ino) in targets: e[2]+=1
                    if st.st_nlink>1: e[3]+=1
                    break
tot=0; save=0
for cat in sorted(agg):
    for k in agg[cat]:
        n,b,l,ml=agg[cat][k]; s=b-b/RATIO[k]
        print(f"{cat:14s} {k:10s} files {n:6d} {b/2**30:7.2f} GiB  linked-to {l:5d} nlink>1 {ml:5d}  est save {s/2**30:6.2f} GiB")
json.dump({f:c for f,c in famcat.items()}, open("/home/xqian/tmp/cleanup-20260916c/famcat.json","w"), indent=0)
print(sorted({c for c in famcat.values() if c.startswith("other")}))

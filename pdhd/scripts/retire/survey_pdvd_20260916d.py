import os, re, collections, json, time
W="/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
fam=collections.defaultdict(lambda:[0,0,0,0.0])   # dirs, bytes(first-link), files, newest mtime
kind=collections.defaultdict(lambda:collections.Counter())
seen=set()
ARM=re.compile(r"^\d{6}_\d+_(.+)$")
def cls(fn):
    for pat,k in [(r"frames.*\.tar\.bz2$","sp-frames"),(r"^clusters-apa.*\.tar\.gz$","img-archive"),(r"^icluster.*\.npz$","icluster-npz"),
                  (r"pctree.*\.tar\.gz$","pctree"),(r"^mabc.*\.zip$","mabc-zip"),(r"\.root$","root"),(r"^calib-evt.*\.json$","calib-dump"),
                  (r"\.log$","log"),(r"\.json$","json"),(r"\.zip$","zip"),(r"\.npz$","npz"),(r"\.tar\.gz$","tgz"),(r"\.tar\.bz2$","tbz2"),(r"\.png$","png")]:
        if re.search(pat,fn): return k
    return "other"
for d in os.listdir(W):
    p=os.path.join(W,d)
    if os.path.islink(p) or not os.path.isdir(p): 
        m=ARM.match(d); f=m.group(1) if m else "(top-level)"; fam[f][0]+=0; continue
    m=ARM.match(d); f=m.group(1) if m else "(bare:"+d+")" if not re.match(r"^\d{6}_\d+$",d) else "(bare)"
    fam[f][0]+=1
    for cur,subs,files in os.walk(p):
        for fn in files:
            q=os.path.join(cur,fn)
            try: st=os.lstat(q)
            except OSError: continue
            if not os.path.isfile(q) or os.path.islink(q): continue
            fam[f][2]+=1; fam[f][3]=max(fam[f][3],st.st_mtime)
            key=(st.st_dev,st.st_ino)
            if key in seen: continue
            seen.add(key); fam[f][1]+=st.st_size; kind[f][cls(fn)]+=st.st_size
rows=sorted(fam.items(), key=lambda x:-x[1][1])
json.dump({k:{"dirs":v[0],"bytes":v[1],"files":v[2],"newest":v[3],"kinds":dict(kind[k])} for k,v in rows}, open("/home/xqian/tmp/cleanup-20260916c/survey.json","w"))
tot=sum(v[1] for _,v in rows)
print(f"total {tot/2**30:.2f} GiB (first-link), families {len(rows)}")
for k,v in rows[:70]:
    top=", ".join(f"{a} {b/2**30:.1f}" for a,b in kind[k].most_common(3))
    print(f"{k:22s} {v[0]:5d} dirs {v[1]/2**30:7.2f} GiB  newest {time.strftime('%m-%d %H:%M',time.localtime(v[3]))}  [{top}]")

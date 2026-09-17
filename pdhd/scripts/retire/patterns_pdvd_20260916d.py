import os, re, collections
W="/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
ARM=re.compile(r"^\d{6}_\d+_(.+)$")
pat=collections.defaultdict(lambda:[0,0,set()])
seen=set()
for d in os.listdir(W):
    p=os.path.join(W,d)
    if os.path.islink(p) or not os.path.isdir(p): continue
    m=ARM.match(d); f=m.group(1) if m else "(bare)"
    for cur,subs,files in os.walk(p):
        rel=os.path.relpath(cur,p)
        for fn in files:
            q=os.path.join(cur,fn)
            if os.path.islink(q): continue
            try: st=os.lstat(q)
            except OSError: continue
            k=(st.st_dev,st.st_ino)
            if k in seen: continue
            seen.add(k)
            name=re.sub(r"\d+","N",("" if rel=="." else re.sub(r"\d+","N",rel)+"/")+fn)
            e=pat[name]; e[0]+=st.st_size; e[1]+=1; e[2].add(f)
for name,(b,n,fs) in sorted(pat.items(), key=lambda x:-x[1][0])[:45]:
    print(f"{b/2**30:7.2f} GiB {n:7d} files {len(fs):3d} fam  {name}   e.g. {sorted(fs)[:4]}")

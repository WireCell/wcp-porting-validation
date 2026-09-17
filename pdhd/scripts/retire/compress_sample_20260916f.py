import os, re, random, subprocess, collections, time
W="/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
ARM=re.compile(r"^\d{6}_\d+_(.+)$")
CLS=[("calib-pr",r"^calib-pr-evt\d+\.json$"),("calib-clus",r"^calib-evt\d+\.json$"),("log-pr",r"^wct_pr_\d+_\d+\.log$"),
     ("log-clus",r"^wct_clus_\d+_\d+\.log$"),("gpu-csv",r"^gpu_mem_.*\.csv$"),("tracking-root",r"^tracking-.*\.root$"),
     ("mabc-zip",r"^mabc.*\.zip$"),("pctree-tgz",r"^pctree-evt\d+\.tar\.gz$"),("img-tgz",r"^clusters-apa.*\.tar\.gz$"),("magnify-root",r"^magnify.*\.root$")]
files=collections.defaultdict(list)
for d in os.listdir(W):
    p=os.path.join(W,d)
    if os.path.islink(p) or not os.path.isdir(p): continue
    for fn in os.listdir(p):
        q=os.path.join(p,fn)
        if os.path.islink(q) or not os.path.isfile(q): continue
        for k,pat in CLS:
            if re.match(pat,fn): files[k].append(q); break
random.seed(1)
for k,_ in CLS:
    L=files[k]; S=random.sample(L,min(12,len(L)))
    raw=0; z3=0; z19=0; t3=0.0
    for q in S:
        b=open(q,"rb").read(); raw+=len(b)
        t=time.time(); z3+=len(subprocess.run(["zstd","-3","-c","-q"],input=b,capture_output=True).stdout); t3+=time.time()-t
        z19+=len(subprocess.run(["zstd","-19","-T8","-c","-q"],input=b,capture_output=True).stdout)
    if raw: print(f"{k:14s} n={len(L):5d} sample {len(S):2d} {raw/2**20:8.1f} MiB  zstd-3 x{raw/max(z3,1):5.2f} ({raw/2**20/max(t3,1e-9):6.0f} MiB/s)  zstd-19 x{raw/max(z19,1):5.2f}")

#!/usr/bin/env python3
"""Inventory of sbnd_xin work* dirs: size, records, dependents, citations."""
import os, re, subprocess, json, collections, sys
ROOT="/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin"
os.chdir(ROOT)
dirs=sorted(d for d in os.listdir('.') if d.startswith('work') and os.path.isdir(d) and not os.path.islink(d))

REC_EXT=('.tsv','.root','.log','.meta','.txt','.json.gz')
info={}
dep=collections.Counter()   # (src,tgt)
for d in dirs:
    tot=0; rec=0; nfile=0; nlink=0; labels=[]
    for cur,sub,files in os.walk(d, followlinks=False):
        # record symlinks
        for name in sub+files:
            p=os.path.join(cur,name)
            if os.path.islink(p):
                nlink+=1
                t=os.readlink(p)
                m=re.search(r'sbnd_xin/(work[^/]*)', t)
                if m and m.group(1)!=d: dep[(d,m.group(1))]+=1
                elif 'sbnd_xin/archive' in t: dep[(d,'archive')]+=1
        sub[:] = [s for s in sub if not os.path.islink(os.path.join(cur,s))]
        for f in files:
            p=os.path.join(cur,f)
            if os.path.islink(p): continue
            try: sz=os.path.getsize(p)
            except OSError: continue
            tot+=sz; nfile+=1
            if f.endswith(REC_EXT) or '.log' in f or f.endswith('.tsv'): rec+=sz
        if os.path.basename(cur).endswith('labels'): labels.append(cur)
    info[d]=dict(total=tot, rec=rec, nfile=nfile, nlink=nlink, labels=labels,
                 mtime=os.path.getmtime(d), entries=len(os.listdir(d)))

rdep=collections.defaultdict(list)
for (s,t),c in dep.items(): rdep[t].append((s,c))

# citations: bare tag (strip manifest prefix)
def bare(d):
    for pre in ('work-mcp1000b-','work-mcp1000-','work-mcp1kall-','work-mcp10-','work-stmcamp-','work-nuecc48-','work-mcsim-','work-r1ql-','work-r2patrec-','work-smoke-','work-'):
        if d.startswith(pre) and len(d)>len(pre): return d[len(pre):]
    return d
cites={}
for d in dirs:
    b=bare(d)
    if b in ('','work'): cites[d]=[]; continue
    try:
        out=subprocess.run(['grep','-rlE',r'\b'+re.escape(b)+r'\b','docs','nusel_display','ql_scan','stm_campaign','scripts','--include=*.md','--include=*.py','--include=*.sh','--include=*.jsonnet'],capture_output=True,text=True).stdout.split()
    except Exception: out=[]
    try:
        out2=subprocess.run(['bash','-c',f"grep -lE '\\b{re.escape(b)}\\b' *.py *.sh *.jsonnet 2>/dev/null"],capture_output=True,text=True).stdout.split()
    except Exception: out2=[]
    cites[d]=sorted(set(out+out2))

json.dump({'info':{k:{kk:(vv if kk!='labels' else vv) for kk,vv in v.items()} for k,v in info.items()},
           'dep':{f'{s}|{t}':c for (s,t),c in dep.items()},
           'rdep':{k:v for k,v in rdep.items()},
           'cites':cites}, open(os.environ.get('RETIRE_STATE','.')+'/inv.json','w'), indent=1)

tot=sum(v['total'] for v in info.values())
print(f"{len(dirs)} dirs, {tot/2**30:.1f} GiB real bytes")
print(f"{'dir':38s} {'MB':>8s} {'recMB':>7s} {'files':>6s} {'deps_in':>7s}  cites")
for d in sorted(dirs, key=lambda x:-info[x]['total']):
    v=info[d]
    di=sum(c for _,c in rdep.get(d,[]))
    print(f"{d:38s} {v['total']/2**20:8.0f} {v['rec']/2**20:7.1f} {v['nfile']:6d} {di:7d}  {len(cites[d])} {'LABELS' if v['labels'] else ''}")

#!/usr/bin/env python3
"""Define removal set R, compute archive footprint, dangling-link dry run."""
import os,re,json,sys,collections
ROOT="/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin"; os.chdir(ROOT)
dirs=sorted(d for d in os.listdir('.') if d.startswith('work') and os.path.isdir(d) and not os.path.islink(d))

def group(d):
    if d in ('work','work-mcp1000','work-mcp10'): return 'KEEP-BASE'
    if d=='work-mcp1kall-d59k': return 'KEEP-HUB'
    if d.startswith(('work-r1ql','work-r2patrec')): return 'KEEP-CURRENT'
    if d.startswith('work-nuecc48'): return 'KEEP-CURRENT'
    if re.match(r'work-stmcamp-(r\d|d64)',d): return 'R-doc63'
    if d.startswith('work-stmcamp-d66'): return 'R-doc66'
    if d.startswith('work-stmcamp-dbg'): return 'R-dbg'
    if d.startswith('work-mcp1kall-d60'): return 'R-doc60'
    if d=='work-smoke-d55pv': return 'R-docs52-57'
    if re.match(r'work-mcp(10|1000|1000b)-(d49son|d52|d53|d55|p54|p55|p56|d56bw|d57mip|m66|p65fin|trace51)',d): return 'R-docs52-57'
    return 'UNCLASSIFIED'

HEAVY=(re.compile(r'^pctree.*\.tar\.gz$'), re.compile(r'^mabc.*\.zip$'), re.compile(r'^calib-evt.*\.json(\.gz)?$'),
       re.compile(r'.*\.npz$'), re.compile(r'^clusters-apa.*\.tar\.gz$'))
def is_heavy(f): return any(p.match(f) for p in HEAVY)

R=[d for d in dirs if group(d).startswith('R-')]
K=[d for d in dirs if group(d).startswith('KEEP')]
assert not [d for d in dirs if group(d)=='UNCLASSIFIED'], [d for d in dirs if group(d)=='UNCLASSIFIED']

stat=collections.defaultdict(lambda: [0,0,0,0])  # tot, keepbytes, nkeepfiles, nheavy
per={}
for d in R:
    tot=keep=nk=nh=0; kfiles=[]
    for cur,sub,files in os.walk(d):
        sub[:]=[s for s in sub if not os.path.islink(os.path.join(cur,s))]
        for f in files:
            p=os.path.join(cur,f)
            if os.path.islink(p): continue
            try: sz=os.path.getsize(p)
            except OSError: continue
            tot+=sz
            if is_heavy(f): nh+=1
            else: keep+=sz; nk+=1; kfiles.append(p)
    per[d]=dict(tot=tot,keep=keep,nk=nk,nh=nh)
    g=group(d); s=stat[g]; s[0]+=tot; s[1]+=keep; s[2]+=nk; s[3]+=nh

print("=== ARCHIVE FOOTPRINT (all real files except pctree/mabc/calib/npz) ===")
for g in sorted(stat):
    t,k,nk,nh=stat[g]
    print(f"{g:14s} total {t/2**30:6.2f} GiB  archive {k/2**20:8.1f} MiB ({nk} files)  reclaim {(t-k)/2**30:6.2f} GiB")
T=sum(s[0] for s in stat.values()); Kb=sum(s[1] for s in stat.values())
print(f"{'TOTAL':14s} total {T/2**30:6.2f} GiB  archive {Kb/2**20:8.1f} MiB  reclaim {(T-Kb)/2**30:6.2f} GiB")

# dangling-link dry run: any symlink under a SURVIVING dir (or elsewhere in sbnd_xin) pointing into R?
Rset=set(R)
bad=collections.Counter()
scan_roots=K+['archive','scan-d59k','scan-r1ql','scan-r2patrec','nusel_display','ql_scan','pics','stm_campaign','docs','bee-d66','bee-nuecc48','showcase-stmfit-286241','showcase-stmfit-mc-evt18']
for root in scan_roots:
    if not os.path.exists(root): continue
    for cur,sub,files in os.walk(root):
        for name in sub+files:
            p=os.path.join(cur,name)
            if os.path.islink(p):
                t=os.readlink(p)
                m=re.search(r'sbnd_xin/(work[^/]*)',t)
                if m and m.group(1) in Rset: bad[(root,m.group(1))]+=1
print("\n=== DANGLING-LINK DRY RUN: surviving symlinks that point into the removal set ===")
if not bad: print("0 -- no surviving symlink resolves into any removal candidate")
else:
    for (s,t),c in bad.most_common(): print(f"  {c:6d}  {s} -> {t}")

json.dump({'R':R,'K':K,'per':per,'group':{d:group(d) for d in dirs}}, open(os.environ.get('RETIRE_STATE','.')+'/plan.json','w'), indent=1)
print(f"\nremoval set: {len(R)} dirs; keep: {len(K)} dirs")

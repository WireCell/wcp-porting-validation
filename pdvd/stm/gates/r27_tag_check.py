"""Compare the PR steiner cloud of the gid-127 (evt 53) / target cluster with the same tag's input clusters.
usage: tag_check.py <tag> <evt> <t0_us or ->  [pr_cid]"""
import sys,json,glob,zipfile,tarfile,io,numpy as np,collections
from scipy.spatial import cKDTree
import os
PD=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"work")
tag,ev=sys.argv[1],sys.argv[2]; t0want=None if sys.argv[3]=='-' else float(sys.argv[3]); cid=int(sys.argv[4]) if len(sys.argv)>4 else None
d=f"{PD}/039349_{ev}_{tag}"
# pctree
tf=tarfile.open(glob.glob(f"{d}/pctree-evt*.tar.gz")[0]); meta={}; arrs={}
names=tf.getnames()
for n in names:
    if n.endswith('_metadata.json'):
        m=json.load(tf.extractfile(n)); meta[m.get('datapath')]=n
def arr(dp): return np.load(io.BytesIO(tf.extractfile(meta[dp].replace('metadata.json','array.npy')).read()))
ev_no=[k for k in meta if k and k.startswith('pointtrees/')][0].split('/')[1]
b=f'pointtrees/{ev_no}/live/pointclouds/namedpcs/3d/arrays/'; c=f'pointtrees/{ev_no}/live/pointclouds/namedpcs/cluster_scalar/arrays/'; l=f'pointtrees/{ev_no}/live/lpcmaps/arrays/'
x,xc,y,z,wp=arr(b+'x')/10,arr(b+'x_t0cor')/10,arr(b+'y')/10,arr(b+'z')/10,arr(b+'wpid')
ident,gid,t0=arr(c+'ident'),arr(c+'matched_flash_gid'),arr(c+'cluster_t0')
m3,mc=arr(l+'3d'),arr(l+'cluster_scalar'); cl=np.zeros(len(x),int); off=0; ci=-1
for n in range(len(m3)):
    if mc[n]: ci+=1
    if m3[n]: cl[off:off+m3[n]]=ident[ci]; off+=m3[n]
print(f"[{tag} evt {ev}] pctree: {len(ident)} live clusters, {len(x)} pts; wpid face->y for top anodes 6/7:",
      {int(w):(round(float(y[wp==w].min())),round(float(y[wp==w].max()))) for w in (103,111,119,127) if (wp==w).any()})
# PR outputs
calib=json.load(open(glob.glob(f"{d}/calib-pr-evt*.json")[0]))
z0=zipfile.ZipFile(f"{d}/mabc-pr.zip"); bee=json.loads(z0.read('data/0/0-clustering-global.json'))
bx,by,bz,bc=map(np.array,(bee['x'],bee['y'],bee['z'],bee['cluster_id'])); ok=np.abs(bx)<1e6
tree=cKDTree(np.c_[bx[ok],by[ok],bz[ok]])
if t0want is not None:
    cands=[int(i) for i,t in zip(ident,t0) if abs(t/1000-t0want)<1]
    print("  pctree clusters with t0≈%.1f us:"%t0want, cands, "gids", [int(gid[list(ident).index(i)]) for i in cands])
# which PR steiner clusters to test: the given cid or all with >100 points
tests=[t for t in calib['steiner'] if (cid is None and len(t['x'])>100) or t['cluster_id']==cid]
for t in tests:
    sx,sy,sz=map(np.array,(t['x'],t['y'],t['z']))
    res=[]
    for dy in (0.0,168.4,-168.4):
        dd,ii=tree.query(np.c_[sx,sy+dy,sz]); res.append((dy,int((dd<3).sum()),collections.Counter(bc[ok][ii[dd<3]].tolist()).most_common(2)))
    best=max(res,key=lambda r:r[1])
    flag="" if best[0]==0.0 else "  <-- SHIFTED"
    print(f"  PR cid {t['cluster_id']:3d} n={len(sx):4d} on-charge: "+"; ".join(f"dy={r[0]:+.0f}:{r[1]}" for r in res)+flag)
    if cid is not None or best[0]!=0.0:
        top=best[2][0][0] if best[2] else None
        if top is not None:
            s=bc==top; e=np.arange(0,320,20); h1,_=np.histogram(bz[s],bins=e); h2,_=np.histogram(sz,bins=e)
            print(f"     Bee {top}: n={s.sum()} x[{bx[s].min():.1f},{bx[s].max():.1f}] y[{by[s].min():.1f},{by[s].max():.1f}] z[{bz[s].min():.1f},{bz[s].max():.1f}]")
            print("     Bee z hist:",h1.tolist()); print("     stn z hist:",h2.tolist())

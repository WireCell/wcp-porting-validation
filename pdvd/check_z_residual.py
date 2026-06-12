#!/usr/bin/env python
"""What does v5 actually change in the imaging?  For blobs in a tube around the
track locus, measure the signed z-residual (blob_z - nearest_locus_z) for v4 vs
v5.  If v5 corrects the registration, v4 blobs sit ~ +13/-10 mm off the locus and
v5 blobs sit ~0.  Also report per-drift-slice blob MULTIPLICITY (ghost blobs).
"""
import sys, json, glob, numpy as np
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd")
from wirecell.util.wires import persist
import pdvd_uvw_offset as O
V5="/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2"
PD="/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd"; TUBE=4.0

def locus(akey):
    cfg=O.CONFIGS[akey]; fn=O.MAGBASE.format(E=cfg["event"],A=cfg["anode"])
    _,res=O.centroids(fn,cfg["anode"],cfg); gm=O.good_mask(res)
    uc=res["U"]["cen"][gm]; vc=res["V"]["cen"][gm]
    st=persist.load(V5); _,tabs=O.find_face_planes(st,cfg["anode"],cfg["win"]["W"])
    return np.array([O.predict_w(tabs,u,v)[1] for u,v in zip(uc,vc)])/10.0
def blobs(tag,a):
    f=glob.glob(f"{PD}/peranodebee_{tag}/data/{a}/{a}-imaging-anode{a}.json")[0]
    d=json.load(open(f)); return np.array(d["x"]),np.array(d["y"]),np.array(d["z"])

for akey in (0,4):
    L=locus(akey)
    print(f"\n==== anode {akey} ====")
    for tag in ("v4baseline","v5"):
        bx,by,bz=blobs(tag,akey)
        d2=(by[:,None]-L[:,0])**2+(bz[:,None]-L[:,1])**2
        j=d2.argmin(1); dist=np.sqrt(d2.min(1)); m=dist<TUBE
        zres=(bz[m]-L[j[m],1])*10.0   # mm
        # ghost multiplicity: blobs per unique drift bin (0.5cm) within tube
        xb=np.round(bx[m]/0.5).astype(int)
        _,cnts=np.unique(xb,return_counts=True)
        print(f"  {tag:12s}: in-tube {m.sum():5d}  z-residual median={np.median(zres):+6.1f}mm "
              f"mean={zres.mean():+6.1f} rms={zres.std():4.1f}  "
              f"blobs/drift-slice mean={cnts.mean():.2f} max={cnts.max()}")

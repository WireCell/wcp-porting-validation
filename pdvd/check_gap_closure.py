#!/usr/bin/env python
"""Discriminating gap-closure check: restrict imaged blobs to a tube around the
calibration track's (y,z) locus, then compare how they populate the drift (x)
axis between the v4-baseline and v5 imaging.  Gaps = empty drift slices along
the track; v5 should fill slices that are empty in v4.

Track (y,z) locus is the per-tick U/cap V crossing (v5 geometry).  Bee blobs carry
(x,y,z) in cm; geometry crossings are in mm -> /10.
"""
import sys, json, glob, numpy as np
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd")
from wirecell.util.wires import persist
import pdvd_uvw_offset as O

V5 = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2"
TUBE_CM = 4.0          # transverse (y,z) tube half-width around the track locus
PD = "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd"

def track_locus_cm(akey):
    cfg = O.CONFIGS[akey]
    fn = O.MAGBASE.format(E=cfg["event"], A=cfg["anode"])
    _, res = O.centroids(fn, cfg["anode"], cfg)
    gm = O.good_mask(res)
    uc=res["U"]["cen"][gm]; vc=res["V"]["cen"][gm]
    store = persist.load(V5)
    _, tabs = O.find_face_planes(store, cfg["anode"], cfg["win"]["W"])
    yz = np.array([O.predict_w(tabs,u,v)[1] for u,v in zip(uc,vc)])  # (n,2) mm (y,z)
    return yz/10.0        # cm

def load_blobs(tag, anode):
    f = glob.glob(f"{PD}/peranodebee_{tag}/data/{anode}/{anode}-imaging-anode{anode}.json")[0]
    d = json.load(open(f))
    return np.array(d["x"]), np.array(d["y"]), np.array(d["z"])  # cm

def in_tube(by, bz, locus):
    # min distance in (y,z) from each blob to the locus polyline points
    # (vectorised: for each blob, nearest locus point)
    d2 = (by[:,None]-locus[:,0][None,:])**2 + (bz[:,None]-locus[:,1][None,:])**2
    return np.sqrt(d2.min(1)) < TUBE_CM

for akey in (0,4):
    locus = track_locus_cm(akey)
    ymin,ymax = locus[:,0].min(), locus[:,0].max()
    zmin,zmax = locus[:,1].min(), locus[:,1].max()
    print(f"\n==== anode {akey}: track locus y[{ymin:.0f},{ymax:.0f}] z[{zmin:.0f},{zmax:.0f}] cm, "
          f"{len(locus)} ticks ====")
    # common drift-x binning across both geometries (track is diagonal in x too)
    allx=[]
    sets={}
    for tag in ("v4baseline","v5"):
        bx,by,bz = load_blobs(tag, akey)
        m = in_tube(by,bz,locus)
        sets[tag]=(bx[m], m.sum())
        allx.append(bx[m])
    allx=np.concatenate(allx)
    if len(allx)==0:
        print("  no blobs in tube (?)"); continue
    lo,hi = allx.min(), allx.max()
    nb = 100
    edges = np.linspace(lo,hi,nb+1)
    for tag in ("v4baseline","v5"):
        bx,cnt = sets[tag]
        h,_ = np.histogram(bx, bins=edges)
        filled = int((h>0).sum())
        print(f"  {tag:12s}: {cnt:5d} blobs in tube,  drift-x span [{bx.min():.0f},{bx.max():.0f}]cm,  "
              f"{filled}/{nb} drift bins filled ({100*filled/nb:.0f}%)")

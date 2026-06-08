#!/usr/bin/env python
"""Validate the PDVD v5 wire file three independent ways (user request:
'use different ways to validate the shift is consistent with what you want').

  1. GEOMETRY FILE DIFF  v4 -> v5 : W points byte-identical; U,V points moved
     ONLY in z by exactly the per-CRP dz (x,y untouched).
  2. OFFSET REMEASURE on the real tracks: med dZ_W ~ 0 and +-1/2-pitch
     consistency 0% -> ~90/83% (the decisive sign+magnitude gate).
  3. W-SHIFT vs U/V-SHIFT EQUIVALENCE: v5 (U/V +dz, W fixed) and the W-shift file
     (W -dz, U/V fixed) must give the SAME internal U/cap V-vs-W consistency
     (differ only by a rigid global z-translation, which imaging is blind to).
"""
import sys, numpy as np
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd")
from wirecell.util.wires import persist
import pdvd_uvw_offset as O

WD = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data"
TMP = "/home/xqian/tmp"
V4 = f"{WD}/protodunevd-wires-larsoft-v4.json.bz2"
V5 = f"{WD}/protodunevd-wires-larsoft-v5.json.bz2"
WSHIFT = f"{TMP}/protodunevd-wires-larsoft-v4-Wshift.json.bz2"
DZ_UV = {0:+13.2,2:+13.2,5:+13.2,7:+13.2, 1:-9.8,3:-9.8,4:-9.8,6:-9.8}  # by type

# ---------- 1. geometry file diff ------------------------------------------
print("="*70); print("1. GEOMETRY FILE DIFF  v4 -> v5")
s4 = persist.load(V4); s5 = persist.load(V5)
# classify each point by the plane(s) that reference it, per anode
def plane_of_points(store):
    """point idx -> (anode_ident, plane_ident)"""
    pm = {}
    for an in store.anodes:
        for fi in an.faces:
            for pi in store.faces[fi].planes:
                pl = store.planes[pi]
                for wi in pl.wires:
                    w = store.wires[wi]
                    for p in (w.tail, w.head):
                        pm[p] = (an.ident, pl.ident)
    return pm
pm = plane_of_points(s4)
maxdx=maxdy=0.0; w_maxabs=0.0
uv_dz_err = {}            # anode_ident -> max |observed dz - expected|
uv_xy_max = 0.0
for i,(p4,p5) in enumerate(zip(s4.points, s5.points)):
    aid, pid = pm.get(i, (None,None))
    dx, dy, dz = p5.x-p4.x, p5.y-p4.y, p5.z-p4.z
    if pid == 2:                       # W must be identical
        w_maxabs = max(w_maxabs, abs(dx), abs(dy), abs(dz))
    elif pid in (0,1):                 # U/V: z shift only
        uv_xy_max = max(uv_xy_max, abs(dx), abs(dy))
        e = abs(dz - DZ_UV[aid])
        uv_dz_err[aid] = max(uv_dz_err.get(aid,0.0), e)
print(f"  W (collection) max |delta(x,y,z)|         = {w_maxabs:.3e} mm  (expect 0)")
print(f"  U/V max |delta x|,|delta y|               = {uv_xy_max:.3e} mm  (expect 0)")
for aid in sorted(uv_dz_err):
    print(f"  anode {aid}: U/V dz matches {DZ_UV[aid]:+.1f} mm to {uv_dz_err[aid]:.3e} mm")
ok1 = w_maxabs<1e-6 and uv_xy_max<1e-6 and max(uv_dz_err.values())<1e-6
print(f"  --> {'PASS' if ok1 else 'FAIL'}")

# ---------- 2 & 3. offset remeasure + equivalence --------------------------
print("="*70); print("2+3. OFFSET REMEASURE  &  W-shift vs U/V-shift EQUIVALENCE")
GEOMS = {"v4 (uncorrected)":V4, "v5 (U/V shift, W fixed)":V5,
         "v4-Wshift (W shift, U/V fixed)":WSHIFT}
half = O.W_PITCH_MM/2.0
for akey in (0,4):
    cfg = O.CONFIGS[akey]
    fn = O.MAGBASE.format(E=cfg["event"], A=cfg["anode"])
    ticks, res = O.centroids(fn, cfg["anode"], cfg)
    gm = O.good_mask(res)
    uc=res["U"]["cen"][gm]; vc=res["V"]["cen"][gm]; wc=res["W"]["cen"][gm]
    print(f"\n  {cfg['label']}  ({gm.sum()} usable ticks)")
    for gname,gpath in GEOMS.items():
        store = persist.load(gpath)
        fi, tabs = O.find_face_planes(store, cfg["anode"], cfg["win"]["W"])
        zc = np.array([O.predict_w(tabs,u,v)[1][1] for u,v in zip(uc,vc)])
        zw = O.w_chan_to_z(tabs["W"], wc)
        dzW = zc - zw
        med = float(np.median(dzW))
        frac = float(np.mean(np.abs(dzW-0.0) < half))   # consistency as-is (no extra centering)
        print(f"    {gname:34s}  med dZ_W = {med:+6.2f} mm   |dZ_W|<half: {frac*100:5.1f}%")
print("="*70)
print("Expect: v4 ~ -13/+10mm (0% consistent);  v5 AND v4-Wshift ~0mm (~90/83%).")

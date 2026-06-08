#!/usr/bin/env python
"""V-only fine optimization (user's proposal): take the common-mode shift as the
basis, keep U (and W) fixed there, and optimize ONLY an extra shift in V.

This is a 1-parameter fit, so it returns a unique delta_V -- but because the
objective depends on (dzU,dzV) only through the sum, optimizing V-only just
re-optimizes  dzU+dzV  by moving V; the result is "the residual sum correction,
attributed entirely to V."  Fixing U instead (move U-only) would give the mirror
answer; the data cannot tell them apart.  Shown on the clean single tracks.
"""
import sys
import numpy as np
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd")
import pdvd_uvw_offset as O
import pdvd_uvw_2dscan as S2
from wirecell.util.wires import persist

HALF = O.W_PITCH_MM / 2.0
store = persist.load(S2.V4)

print("V-only fine optimization on the clean single tracks (v4 geometry).")
for akey in (0, 4):
    cfg = O.CONFIGS[akey]
    fn = O.MAGBASE.format(E=cfg["event"], A=cfg["anode"])
    _, res = O.centroids(fn, cfg["anode"], cfg)
    gm = O.good_mask(res)
    uc = res["U"]["cen"][gm]; vc = res["V"]["cen"][gm]; wc = res["W"]["cen"][gm]
    fi, tabs = O.find_face_planes(store, cfg["anode"], cfg["win"]["W"])
    z0fn, cU, cV = S2.crossing_z_coeffs(tabs)
    base = z0fn(uc, vc) - O.w_chan_to_z(tabs["W"], wc)        # dZ_W at (0,0)

    def frac(dzU, dzV):
        return float(np.mean(np.abs(base + cU * dzU + cV * dzV) < HALF))

    g = np.arange(-25.0, 25.001, 0.1)
    # 1) basis = best COMMON-mode shift (dzU=dzV=s)
    s_star = float(g[int(np.argmax([frac(s, s) for s in g]))])
    # 2) fix U at s_star, optimize V-only:  dzV = s_star + delta
    fV = [frac(s_star, s_star + d) for d in g]
    dV = float(g[int(np.argmax(fV))])
    # mirror: fix V at s_star, optimize U-only
    fU = [frac(s_star + d, s_star) for d in g]
    dU = float(g[int(np.argmax(fU))])

    print(f"\n==== {cfg['label']} ====")
    print(f"  basis common-mode shift  s* = {s_star:+.2f} mm on BOTH U and V "
          f"(consistency {frac(s_star,s_star)*100:.1f}%)")
    print(f"  V-only optimum:  extra dV = {dV:+.2f} mm  -> (dzU,dzV)=({s_star:+.2f},{s_star+dV:+.2f})"
          f"  consistency {frac(s_star,s_star+dV)*100:.1f}%")
    print(f"  (mirror) U-only optimum:  extra dU = {dU:+.2f} mm  -> (dzU,dzV)=({s_star+dU:+.2f},{s_star:+.2f})"
          f"  consistency {frac(s_star+dU,s_star)*100:.1f}%")
    print(f"  --> both reach the SAME consistency by moving the SAME sum "
          f"(dV/2 = dU/2 = {dV/2:+.2f} mm of extra sum); the split U-vs-V is a free choice.")

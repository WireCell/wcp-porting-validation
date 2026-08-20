#!/usr/bin/env python3
"""Print segments from calib-pr-evt<ID>.json with any fit point within R cm of a point."""
import json, math, sys
arm_evt, px, py, pz, R = sys.argv[1], *map(float, sys.argv[2:6])
d = json.load(open(arm_evt))
mv = d["main_vertex"]
print(f"main_vertex=({mv['x']:.1f},{mv['y']:.1f},{mv['z']:.1f}) cid={mv['cluster_id']}")
for s in d["segments"]:
    pts = s["points"]
    P = [[q["x"], q["y"], q["z"]] if isinstance(q, dict) else q for q in pts]
    dmin = min(math.dist(p, (px, py, pz)) for p in P)
    if dmin <= R:
        L = sum(math.dist(P[i], P[i+1]) for i in range(len(P)-1))
        keys = {k: s[k] for k in s if k not in ("points",)}
        print(f"seg {keys.get('id','?')}: npts={len(P)} len={L:.1f}cm dmin={dmin:.1f} "
              f"front=({P[0][0]:.1f},{P[0][1]:.1f},{P[0][2]:.1f}) back=({P[-1][0]:.1f},{P[-1][1]:.1f},{P[-1][2]:.1f}) "
              + " ".join(f"{k}={v}" for k, v in keys.items() if k != "id"))

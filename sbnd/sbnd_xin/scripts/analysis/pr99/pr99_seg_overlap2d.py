#!/usr/bin/env python3
"""Per-wire-view 2D overlap (pr/83 op1-proj style) between two segments, SBND angles 0,+-60 deg."""
import json, math, sys
import numpy as np
from scipy.spatial import cKDTree
f, ida, idb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = json.load(open(f))
def pts(sid):
    for s in d["segments"]:
        if s.get("id", -999) == sid:
            return np.array([[q["x"], q["y"], q["z"]] if isinstance(q, dict) else q for q in s["points"]])
    raise SystemExit(f"{sid} not found")
A, B = pts(ida), pts(idb)
la = np.linalg.norm(np.diff(A, axis=0), axis=1).sum(); lb = np.linalg.norm(np.diff(B, axis=0), axis=1).sum()
S, L = (A, B) if la <= lb else (B, A)
for name, ang in (("U", math.radians(60)), ("V", math.radians(-60)), ("W", 0.0)):
    proj = lambda P: np.c_[P[:, 0], math.cos(ang) * P[:, 2] - math.sin(ang) * P[:, 1]]
    t = cKDTree(proj(L)); dd, _ = t.query(proj(S))
    print(f"view {name}: overlap@1.0cm={np.mean(dd < 1.0):.2f} @2.0cm={np.mean(dd < 2.0):.2f}")

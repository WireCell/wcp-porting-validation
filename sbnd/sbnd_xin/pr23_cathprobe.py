#!/usr/bin/env python3
"""doc pr/23 V2: detect cathode-straddling cluster splits in a PR arm.

For each fitted PR cluster (shower_track layer of mabc-pr.zip), report the
x-range and, for every cluster pair, the closest-point gap.  A pair whose
closest points sit on OPPOSITE sides of the cathode (x=0) with both endpoints
within `band` cm of it and a 3D gap below `gapcut` cm is a candidate broken
crosser -- the split the cathode re-join pass must prevent.

Usage: pr23_cathprobe.py <mabc-pr.zip> [band_cm=6] [gapcut_cm=10]
"""
import json, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree

zpath = sys.argv[1]
band = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
gapcut = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

with zipfile.ZipFile(zpath) as z:
    name = next(n for n in z.namelist() if n.endswith("-shower_track-global.json"))
    with z.open(name) as f:
        st = json.load(f)

xyz = np.column_stack([st["x"], st["y"], st["z"]])
cid = np.array(st["cluster_id"])

clusters = {}
for c in sorted(set(cid.tolist())):
    pts = xyz[cid == c]
    clusters[c] = pts
    print(f"cluster {c:>6}: {len(pts):>6} pts  x [{pts[:,0].min():8.2f}, {pts[:,0].max():8.2f}]"
          f"  y [{pts[:,1].min():8.2f}, {pts[:,1].max():8.2f}]"
          f"  z [{pts[:,2].min():8.2f}, {pts[:,2].max():8.2f}]"
          f"{'  STRADDLES x=0' if pts[:,0].min() < 0 < pts[:,0].max() else ''}")

ids = sorted(clusters)
print("\n--- cluster-pair closest approaches near the cathode ---")
print(f"{'a':>6} {'b':>6} {'gap_cm':>7} {'xa':>7} {'xb':>7} {'dyz_cm':>7}  verdict (band={band}, gapcut={gapcut})")
found = 0
for i, a in enumerate(ids):
    ka = cKDTree(clusters[a])
    for b in ids[i+1:]:
        d, idxa = ka.query(clusters[b])
        j = int(np.argmin(d))
        gap = float(d[j])
        if gap > 25.0:
            continue
        pa = clusters[a][idxa[j]]
        pb = clusters[b][j]
        near_cath = abs(pa[0]) < band and abs(pb[0]) < band
        straddle = (pa[0] < 0) != (pb[0] < 0)
        dyz = float(np.hypot(pa[1]-pb[1], pa[2]-pb[2]))
        verdict = ""
        if near_cath and gap < gapcut:
            verdict = "CATHODE-STRADDLE-SPLIT" if straddle else "cathode-band pair (same side)"
            found += 1
        print(f"{a:>6} {b:>6} {gap:>7.2f} {pa[0]:>7.2f} {pb[0]:>7.2f} {dyz:>7.2f}  {verdict}")
print(f"\ncathode-band pairs (gap<{gapcut}, both |x|<{band}): {found}")

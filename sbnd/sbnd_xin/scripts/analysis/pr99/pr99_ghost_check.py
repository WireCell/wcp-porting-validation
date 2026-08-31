#!/usr/bin/env python3
"""Fit points near a coordinate with distance to nearest imaged charge: ghost = fit in a void."""
import json, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree
zp, px, py, pz, R = sys.argv[1], *map(float, sys.argv[2:6])
z = zipfile.ZipFile(zp)
L = {}
for n in z.namelist():
    for tag in ("clustering-global", "track_fit-global"):
        if n.endswith(tag + ".json"):
            L[tag] = json.loads(z.read(n))
chg = np.c_[L["clustering-global"]["x"], L["clustering-global"]["y"], L["clustering-global"]["z"]]
fit = L["track_fit-global"]
F = np.c_[fit["x"], fit["y"], fit["z"]]
cid = np.array(fit["cluster_id"]); rcid = np.array(fit.get("real_cluster_id", cid))
t = cKDTree(chg)
sel = np.linalg.norm(F - (px, py, pz), axis=1) <= R
print(f"{sel.sum()} fit points within {R}cm of ({px},{py},{pz})")
if sel.sum():
    d, _ = t.query(F[sel])
    for cluster in np.unique(cid[sel]):
        m = sel & (cid == cluster)
        dd, _ = t.query(F[m])
        print(f"  fit cid={cluster} rcid={np.unique(rcid[m])} n={m.sum()} dist-to-charge min/med/max = {dd.min():.1f}/{np.median(dd):.1f}/{dd.max():.1f} cm")

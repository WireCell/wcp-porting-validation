#!/usr/bin/env python3
"""doc pr/99 -- along-trajectory track->shower transition scan (pr/93 sec 3's
named next measurement, owner's pi+- charge-exchange hypothesis).

For a claimed EM shower: walk its trajectory (start segment chain from the
shower start vertex), and in bins of arc length s report
  dqdx_med   median fit-point charge (T_rec_charge q) of the shower's own
             trajectory points in the bin (track-like: flat ~MIP)
  n_off      imaged charge points (clustering-global) within R_CYL of the
             axis bin center that are farther than R_CORE from the axis
  spread     rms transverse distance of imaged charge within R_CYL
Hadronic signature: flat MIP-ish dqdx + collimated (low spread) up to a
localized s*, then a step. EM signature: spread grows from small s.
Usage: transition_scan.py <arm> <evt> <shower_id> [label]
"""
import json, sys
import numpy as np
import uproot, zipfile
R_CYL, R_CORE, BIN = 8.0, 1.2, 3.0
arm, evt, shid = sys.argv[1], sys.argv[2], int(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else f"{evt}/{shid}"
d = json.load(open(f"{arm}/pr_evt{evt}/calib-pr-evt{evt}.json"))
a = uproot.open(f"{arm}/pr_evt{evt}/tracking-pr.root")["T_rec_charge"].arrays(library="np")
z = zipfile.ZipFile(f"{arm}/pr_evt{evt}/mabc-pr.zip")
chg = None
for n in z.namelist():
    if n.endswith("clustering-global.json"):
        c = json.loads(z.read(n)); chg = np.c_[c["x"], c["y"], c["z"]]
segs = {s["id"]: s for s in d["segments"] if s.get("shower_id") == shid}
if not segs: raise SystemExit(f"no members with shower_id {shid}")
# axis = ordered union of member fit points, walked greedily from the member
# whose endpoint is nearest the main vertex (good enough for a scan)
mv = np.array([d["main_vertex"][k] for k in "xyz"])
pts = []
for s in segs.values():
    P = np.array([[q["x"], q["y"], q["z"]] if isinstance(q, dict) else q for q in s["points"]])
    pts.append(P)
allp = np.vstack(pts)
# order by geodesic-ish: sort by distance from vertex along principal axis
d0 = np.linalg.norm(allp - mv, axis=1)
order = np.argsort(d0)
axis = allp[order]
s_arc = d0[order]          # radial arc proxy
qpt = []
for sid in segs:
    m = a["real_cluster_id"] == sid
    qpt.append(np.c_[a["x"][m], a["y"][m], a["z"][m], a["q"][m]])
qpt = np.vstack(qpt) if qpt else np.zeros((0, 4))
from scipy.spatial import cKDTree
taxis = cKDTree(axis)
print(f"# {label}: members={list(segs)} npts_axis={len(axis)} smax={s_arc.max():.1f}cm")
print(f"# {'s0':>5} {'s1':>5} {'dqdx_med':>8} {'spread':>6} {'n_in':>5} {'n_off':>5} {'offfrac':>7}")
for lo in np.arange(0, s_arc.max(), BIN):
    hi = lo + BIN
    axm = axis[(s_arc >= lo) & (s_arc < hi)]
    if len(axm) == 0: continue
    # fit charge in bin
    if len(qpt):
        dq = np.linalg.norm(qpt[:, None, :3] - axm[None, :, :], axis=2).min(axis=1) if len(qpt) < 3000 else cKDTree(axm).query(qpt[:, :3])[0]
        qb = qpt[dq < 2.0, 3]
    else:
        qb = np.array([])
    # imaged charge around bin
    dch = cKDTree(axm).query(chg)[0]
    inc = chg[dch < R_CYL]
    dc2 = cKDTree(axm).query(inc)[0] if len(inc) else np.array([])
    n_in = len(inc); n_off = int((dc2 > R_CORE).sum())
    spread = float(np.sqrt(np.mean(dc2**2))) if len(dc2) else 0.0
    print(f"{lo:6.0f} {hi:5.0f} {np.median(qb) if len(qb) else 0:8.0f} {spread:6.2f} {n_in:5d} {n_off:5d} {n_off/max(n_in,1):7.2f}")

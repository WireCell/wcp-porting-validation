#!/usr/bin/env python3
"""Quantify track_fit points not covered by shower_track / clustering charge.

Usage: gapjump_probe.py <mabc-pr.zip> [cover_cm]
For every track_fit point: distance to nearest shower_track point and to
nearest clustering (QL charge) point.  Group uncovered fit points into
per-segment stretches and print their geometry.
"""
import json, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree

zpath = sys.argv[1]
cover = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  # cm

def load(z, name):
    with z.open(name) as f:
        d = json.load(f)
    return d

with zipfile.ZipFile(zpath) as z:
    names = z.namelist()
    def find(layer):
        for n in names:
            if n.endswith(f"-{layer}-global.json"):
                return n
        return None
    tf = load(z, find("track_fit"))
    st = load(z, find("shower_track"))
    cl = load(z, find("clustering"))

def arr(d):
    return (np.column_stack([d["x"], d["y"], d["z"]]),
            np.array(d.get("real_cluster_id", d.get("cluster_id"))),
            np.array(d["cluster_id"]), np.array(d["q"]))

tf_xyz, tf_rid, tf_cid, tf_q = arr(tf)
st_xyz, st_rid, st_cid, st_q = arr(st)
cl_xyz, cl_rid, cl_cid, cl_q = arr(cl)

print(f"track_fit: {len(tf_xyz)} pts, shower_track: {len(st_xyz)} pts, clustering: {len(cl_xyz)} pts")
print(f"track_fit clusters: {sorted(set(tf_cid.tolist()))}")
print(f"track_fit segment ids (real_cluster_id): {len(set(tf_rid.tolist()))} distinct")

kd_st = cKDTree(st_xyz)
kd_cl = cKDTree(cl_xyz)
d_st, _ = kd_st.query(tf_xyz)
d_cl, _ = kd_cl.query(tf_xyz)

unc = d_st > cover
print(f"\nfit points with no shower_track point within {cover} cm: {unc.sum()}/{len(tf_xyz)}"
      f" ({100*unc.sum()/len(tf_xyz):.1f}%)")
print(f"  of those, also no clustering charge within {cover} cm: {(unc & (d_cl > cover)).sum()}")
print(f"  of those, clustering charge IS within {cover} cm (assoc miss, charge exists): {(unc & (d_cl <= cover)).sum()}")

# stretch finding: per segment (real_cluster_id encodes cluster*1000+seg for tracks),
# walk fit points in dump order, group consecutive uncovered points.
print("\n--- uncovered stretches (>=2 consecutive uncovered fit points per segment) ---")
print(f"{'seg':>8} {'npts':>5} {'len_cm':>7} {'step_cm':>7} {'x0':>8} {'x1':>8} {'d_charge_med':>12}")
order = np.arange(len(tf_xyz))
total_unc_len = 0.0
stretches = []
for seg in sorted(set(tf_rid.tolist())):
    idx = order[tf_rid == seg]
    run = []
    for i in idx:
        if unc[i]:
            run.append(i)
        else:
            if len(run) >= 2: stretches.append((seg, run))
            run = []
    if len(run) >= 2: stretches.append((seg, run))
for seg, run in stretches:
    pts = tf_xyz[run]
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    slen = steps.sum()
    total_unc_len += slen
    print(f"{seg:>8} {len(run):>5} {slen:>7.1f} {np.median(steps):>7.2f} "
          f"{pts[0][0]:>8.1f} {pts[-1][0]:>8.1f} {np.median(d_cl[run]):>12.1f}")
print(f"\ntotal uncovered stretch length: {total_unc_len:.1f} cm "
      f"across {len(stretches)} stretches")

# how much of the uncovered length has NO charge anywhere near (true void)
void = [np.linalg.norm(np.diff(tf_xyz[r], axis=0), axis=1).sum()
        for s, r in stretches if np.median(d_cl[r]) > 3.0]
print(f"stretches through true void (median charge dist > 3 cm): {len(void)}, length {sum(void):.1f} cm")

# fit-point spacing overall vs in uncovered regions
allsteps = []
for seg in sorted(set(tf_rid.tolist())):
    pts = tf_xyz[tf_rid == seg]
    if len(pts) > 1:
        allsteps.extend(np.linalg.norm(np.diff(pts, axis=0), axis=1).tolist())
allsteps = np.array(allsteps)
print(f"\nfit point spacing: median {np.median(allsteps):.2f} cm, p90 {np.percentile(allsteps,90):.2f} cm, max {allsteps.max():.2f} cm")

# --- attribution: connected components of the charge cloud at 3 cm linkage ---
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
link = 3.0
pairs = kd_cl.query_pairs(link, output_type='ndarray')
n = len(cl_xyz)
m = coo_matrix((np.ones(len(pairs)), (pairs[:,0], pairs[:,1])), shape=(n,n))
ncomp, lab = connected_components(m, directed=False)
sizes = np.bincount(lab)
print(f"\ncharge cloud: {ncomp} components at {link} cm linkage; top sizes {sorted(sizes.tolist())[::-1][:8]}")

_, near_idx = kd_cl.query(tf_xyz)
print("\n--- stretch endpoint attribution ---")
print(f"{'seg':>8} {'len_cm':>7} {'comp0(size)':>12} {'comp1(size)':>12} {'same?':>6} "
      f"{'end0 (x,y,z)':>24} {'end1 (x,y,z)':>24}")
for seg, run in stretches:
    pts = tf_xyz[run]
    slen = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    c0, c1 = lab[near_idx[run[0]]], lab[near_idx[run[-1]]]
    print(f"{seg:>8} {slen:>7.1f} {f'{c0}({sizes[c0]})':>12} {f'{c1}({sizes[c1]})':>12} "
          f"{'SAME' if c0==c1 else 'DIFF':>6} "
          f"({pts[0][0]:6.1f},{pts[0][1]:6.1f},{pts[0][2]:6.1f}) "
          f"({pts[-1][0]:6.1f},{pts[-1][1]:6.1f},{pts[-1][2]:6.1f})")

# --- dead-area test: do void stretch points project into dead y-z polygons? ---
from matplotlib.path import Path as MplPath
polys = []
with zipfile.ZipFile(zpath) as z:
    for nme in z.namelist():
        if "deadarea" in nme:
            with z.open(nme) as f:
                dd = json.load(f)
            for poly in dd.get("polygons", []):
                if len(poly) >= 3:
                    polys.append(MplPath(np.array(poly)))
print(f"\ndead-area polygons: {len(polys)}")
if polys:
    for seg, run in stretches:
        pts = tf_xyz[run]
        yz = pts[:, [2,1]]  # bee polygons are (z,y)? try both
        inside_zy = sum(any(p.contains_point(q) for p in polys) for q in yz)
        inside_yz = sum(any(p.contains_point(q) for p in polys) for q in pts[:, [1,2]])
        if inside_zy or inside_yz:
            print(f"  seg {seg}: {inside_zy} pts in dead (z,y), {inside_yz} in dead (y,z) of {len(pts)}")
    print("  (segments not listed: zero dead-area overlap)")

# --- per-segment coverage summary ---
print("\n--- per-segment fit coverage ---")
print(f"{'seg':>8} {'nfit':>5} {'len_cm':>7} {'unc%':>6}")
for seg in sorted(set(tf_rid.tolist())):
    m2 = tf_rid == seg
    pts = tf_xyz[m2]
    slen = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() if m2.sum() > 1 else 0
    print(f"{seg:>8} {m2.sum():>5} {slen:>7.1f} {100*unc[m2].mean():>6.1f}")

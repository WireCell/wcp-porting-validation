#!/usr/bin/env python3
"""doc pr/23 V3: batch census of track_fit points uncovered by shower_track.

For every pr_evt<ID>/mabc-pr.zip under each given arm root, classify every
uncovered stretch (>=2 consecutive fit points with no shower_track point
within `cover` cm, per segment) into:

  dead     - any stretch point inside a channel-deadarea polygon (z,y)
  cathode  - median |x - cathode_x| < 5 cm (the SBND cathode notch band)
  assoc    - clustering charge IS within `cover` cm (association miss)
  bridge   - endpoints attach to DIFFERENT charge components at 3 cm linkage
             (the doc pr/22 sec 8 pathology the protect_bundle stage removes)
  stitch   - same charge component (designed WCP shower stitching)

Priority: dead > cathode > assoc > bridge/stitch.

Usage: pr23_fitcover_census.py <arm_root> [<arm_root> ...] [-cover 1.0] [-o out.tsv]
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

args = sys.argv[1:]
cover = 1.0
out = None
roots = []
i = 0
while i < len(args):
    if args[i] == "-cover":
        cover = float(args[i + 1]); i += 2
    elif args[i] == "-o":
        out = args[i + 1]; i += 2
    else:
        roots.append(args[i]); i += 1

CATH_BAND = 5.0  # cm
LINK = 3.0       # cm charge-component linkage


def load_layer(z, suffix):
    for n in z.namelist():
        if n.endswith(suffix):
            with z.open(n) as f:
                return json.load(f)
    return None


def census_event(zpath):
    with zipfile.ZipFile(zpath) as z:
        tf = load_layer(z, "-track_fit-global.json")
        st = load_layer(z, "-shower_track-global.json")
        cl = load_layer(z, "-clustering-global.json")
        polys = []
        for n in z.namelist():
            if "deadarea" in n:
                with z.open(n) as f:
                    dd = json.load(f)
                for poly in dd.get("polygons", []):
                    if len(poly) >= 3:
                        polys.append(MplPath(np.array(poly)))
    if not tf or not st or not cl or not len(tf.get("x", [])):
        return None

    tf_xyz = np.column_stack([tf["x"], tf["y"], tf["z"]])
    tf_rid = np.array(tf.get("real_cluster_id", tf["cluster_id"]))
    st_xyz = np.column_stack([st["x"], st["y"], st["z"]])
    cl_xyz = np.column_stack([cl["x"], cl["y"], cl["z"]])

    kd_st = cKDTree(st_xyz)
    kd_cl = cKDTree(cl_xyz)
    d_st, _ = kd_st.query(tf_xyz)
    d_cl, near_idx = kd_cl.query(tf_xyz)
    unc = d_st > cover

    # charge components (for bridge/stitch attribution)
    pairs = kd_cl.query_pairs(LINK, output_type="ndarray")
    ncl = len(cl_xyz)
    m = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(ncl, ncl))
    _, lab = connected_components(m, directed=False)

    stretches = []
    order = np.arange(len(tf_xyz))
    for seg in sorted(set(tf_rid.tolist())):
        run = []
        for i in order[tf_rid == seg]:
            if unc[i]:
                run.append(i)
            else:
                if len(run) >= 2:
                    stretches.append((seg, run))
                run = []
        if len(run) >= 2:
            stretches.append((seg, run))

    res = {k: 0.0 for k in ("dead", "cathode", "assoc", "bridge", "stitch")}
    nstr = {k: 0 for k in res}
    for seg, run in stretches:
        pts = tf_xyz[run]
        slen = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        in_dead = any(any(p.contains_point(q) for p in polys) for q in pts[:, [2, 1]])
        if in_dead:
            cls = "dead"
        elif np.median(np.abs(pts[:, 0])) < CATH_BAND:
            cls = "cathode"
        elif np.median(d_cl[run]) <= cover:
            cls = "assoc"
        elif lab[near_idx[run[0]]] != lab[near_idx[run[-1]]]:
            cls = "bridge"
        else:
            cls = "stitch"
        res[cls] += slen
        nstr[cls] += 1

    return {
        "nfit": int(len(tf_xyz)),
        "nunc": int(unc.sum()),
        "pct": 100.0 * unc.sum() / len(tf_xyz),
        "len": {k: res[k] for k in res},
        "nstr": {k: nstr[k] for k in nstr},
        "total_len": sum(res.values()),
    }


rows = []
for root in roots:
    arm = os.path.basename(root.rstrip("/"))
    for zp in sorted(glob.glob(os.path.join(root, "pr_evt*", "mabc-pr.zip"))):
        evt = os.path.basename(os.path.dirname(zp)).replace("pr_evt", "")
        r = census_event(zp)
        if r is None:
            rows.append([arm, evt] + ["-"] * 13)
            continue
        rows.append([arm, evt, r["nfit"], r["nunc"], f"{r['pct']:.1f}",
                     f"{r['total_len']:.1f}"]
                    + [f"{r['len'][k]:.1f}" for k in ("dead", "cathode", "assoc", "bridge", "stitch")]
                    + [r["nstr"][k] for k in ("dead", "cathode", "assoc", "bridge", "stitch")])

hdr = ["arm", "event", "nfit", "nunc", "unc_pct", "unc_len_cm",
       "len_dead", "len_cathode", "len_assoc", "len_bridge", "len_stitch",
       "n_dead", "n_cathode", "n_assoc", "n_bridge", "n_stitch"]
lines = ["\t".join(str(x) for x in ([*hdr] if i < 0 else row))
         for i, row in enumerate(rows)]
text = "\t".join(hdr) + "\n" + "\n".join("\t".join(str(x) for x in row) for row in rows) + "\n"
if out:
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out} ({len(rows)} rows)")
else:
    sys.stdout.write(text)

# per-arm summary
print("\n--- per-arm totals ---", file=sys.stderr)
for root in roots:
    arm = os.path.basename(root.rstrip("/"))
    sel = [r for r in rows if r[0] == arm and r[2] != "-"]
    if not sel:
        continue
    tot_fit = sum(int(r[2]) for r in sel)
    tot_unc = sum(int(r[3]) for r in sel)
    sums = {k: sum(float(r[6 + j]) for r in sel)
            for j, k in enumerate(("dead", "cathode", "assoc", "bridge", "stitch"))}
    print(f"{arm}: {len(sel)} events, {tot_unc}/{tot_fit} uncovered "
          f"({100.0*tot_unc/max(tot_fit,1):.1f}%), stretch cm: "
          + ", ".join(f"{k}={v:.1f}" for k, v in sums.items()), file=sys.stderr)

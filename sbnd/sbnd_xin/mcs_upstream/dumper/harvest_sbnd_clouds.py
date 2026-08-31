#!/usr/bin/env python3
"""doc 80 round 0 step 0d: harvest SBND-shaped muon clouds for MCS fixtures.

Scans work-mcp1k-prod0825/pr_evt*/tracking-pr.root, picks events with a long
single track (largest particle_id group in the nu cluster by point count and
spatial extent), writes each as a text cloud for mcs_dump:
  line 1: start x y z   (first fit point of the track)
  line 2: end   x y z   (last fit point)
  then one 'x y z' per line (whole nu-cluster cloud, so trim_trajectory has
  real delta rays / other prongs to reject -- the ubreco-shaped situation).

Selection: nu cluster (T_kine.cluster_id), longest particle by extent >= MINLEN
cm, cluster cloud size capped at MAXPTS points.
"""
import glob
import os
import sys

import numpy as np
import uproot

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-mcp1k-prod0825"
OUT = os.path.dirname(os.path.abspath(__file__))
MINLEN = 100.0   # cm extent of the selected track
MAXPTS = 6000    # cap on cluster cloud size (runtime guard)
NWANT = 4

def main():
    picked = 0
    report = []
    for prdir in sorted(glob.glob(os.path.join(BASE, "pr_evt*"))):
        evt = prdir.rsplit("pr_evt", 1)[1]
        froot = os.path.join(prdir, "tracking-pr.root")
        if not os.path.exists(froot):
            continue
        try:
            f = uproot.open(froot)
            if "T_kine" not in f or "T_rec_charge" not in f:
                continue
            kine = f["T_kine"].arrays(["cluster_id"], library="np")
            if len(kine["cluster_id"]) == 0:
                continue
            nu_cid = int(kine["cluster_id"][0])
            rc = f["T_rec_charge"].arrays(["x", "y", "z", "cluster_id", "particle_id"], library="np")
        except Exception as e:
            print(f"pr_evt{evt}: skip ({e})", file=sys.stderr)
            continue
        sel = rc["cluster_id"] == nu_cid
        if sel.sum() < 100 or sel.sum() > MAXPTS:
            continue
        x, y, z, pid = rc["x"][sel], rc["y"][sel], rc["z"][sel], rc["particle_id"][sel]
        # longest particle (segment) group by extent
        best = None
        for p in np.unique(pid):
            if p < 0:
                continue  # -1 = unassigned scatter, not a track
            m = pid == p
            if m.sum() < 150:
                continue
            pts = np.stack([x[m], y[m], z[m]], axis=1)
            extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
            # contiguous ~0.6 cm-spaced trajectory: N*0.6 must roughly cover extent
            if extent <= 0 or not (0.7 <= m.sum() * 0.6 / extent <= 3.0):
                continue
            if best is None or extent > best[1]:
                best = (p, extent, pts)
        if best is None or best[1] < MINLEN:
            continue
        p, extent, pts = best
        # endpoints: first/last row of the selected particle (fit order)
        start, end = pts[0], pts[-1]
        cloud = np.stack([x, y, z], axis=1)
        out = os.path.join(OUT, f"sbnd_cloud_evt{evt}.txt")
        with open(out, "w") as fh:
            fh.write("%.17g %.17g %.17g\n" % tuple(start))
            fh.write("%.17g %.17g %.17g\n" % tuple(end))
            for r in cloud:
                fh.write("%.17g %.17g %.17g\n" % tuple(r))
        report.append((evt, int(sel.sum()), int(len(pts)), float(extent)))
        picked += 1
        print(f"pr_evt{evt}: cloud N={sel.sum()} track pid={p} npts={len(pts)} extent={extent:.1f} cm -> {out}")
        if picked >= NWANT:
            break
    if picked < NWANT:
        print(f"WARNING: only {picked}/{NWANT} clouds found", file=sys.stderr)

if __name__ == "__main__":
    main()

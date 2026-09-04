#!/usr/bin/env python3
"""doc pdvd/32 round 3b / doc pdvd/36: the SUPPORT axis over a PR manifest.

For every event dir `work/*_<tag>` holding a mabc-pr.zip, take each point of
the `stm_fit-global` Bee layer (the fitted STM trajectories) and measure its
distance to the nearest point of the `clustering-global` layer (all
reconstructed 3-D charge).  Report, per tag, the number of stm_fit points and
the fraction beyond 2 cm and 10 cm from any charge.

This is the manifest-scale companion of stm_endtrim_grade.py's support column:
the end trim exists to REMOVE unsupported trajectory points, so any change
that loosens it buys coverage and must be charged for what it keeps.  The
baseline is NOT zero (a fitted trajectory is smooth where the charge is
blobby), so read the DELTA between tags, never one absolute number.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/pr_arm_support120.py d32p000 d32p035 d36on d36on035
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree


def one(d):
    try:
        with zipfile.ZipFile(os.path.join(d, "mabc-pr.zip")) as z:
            cl = json.loads(z.read("data/0/0-clustering-global.json"))
            ft = json.loads(z.read("data/0/0-stm_fit-global.json"))
    except Exception:
        return None
    if not ft.get("x"):
        return None
    P = np.stack([cl["x"], cl["y"], cl["z"]], 1).astype(float)
    F = np.stack([ft["x"], ft["y"], ft["z"]], 1).astype(float)
    dd, _ = cKDTree(P).query(F)
    return len(F), int((dd > 2.0).sum()), int((dd > 10.0).sum())


def main(tags):
    print("  %-9s %6s  %9s  %-18s  %-18s" % ("tag", "events", "stm_fit", ">2 cm", ">10 cm"))
    for tag in tags:
        n = f = g = ev = 0
        for d in sorted(glob.glob("work/*_%s" % tag)):
            r = one(d)
            if r is None:
                continue
            ev += 1
            n += r[0]
            f += r[1]
            g += r[2]
        print("  %-9s %6d  %9d  %7d (%5.2f %%)  %7d (%5.2f %%)"
              % (tag, ev, n, f, 100.0 * f / max(n, 1), g, 100.0 * g / max(n, 1)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])

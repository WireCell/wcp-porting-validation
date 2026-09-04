#!/usr/bin/env python3
"""doc pdvd/32 round 3: grade a trajectory end-trim arm on BOTH axes.

`stm_trajectory_coverage.py` answers "how much of the track did the trajectory
reach".  On its own that is a one-sided score: the end trim exists to remove
trajectory points that have no charge under them, so any change that loosens it
buys coverage and must be charged for whatever unsupported trajectory it also
buys.  This script reports the two together, plus the STM verdicts, because a
longer trajectory that flips an STM tag is not automatically a better one.

Metrics per arm, from `mabc-pr.zip` and the PR log:

  * support -- distance from each `stm_fit-global` point to the nearest point
    of `clustering-global` (every cluster's blob samples, i.e. all the
    reconstructed 3-D charge in the event).  Reported as the fraction beyond
    1/2/5/10 cm.  NOTE the baseline is NOT zero: the STM fit legitimately
    crosses dead regions, which carry no blob points, so read the DELTA against
    the arm's own knob-off baseline and never the absolute number alone.
  * per-cluster split along a named axis (`--axis`), which is what separates
    "recovered the track's own end" from "grew a tail past it".
  * STM verdicts (`STM=0/1` per cluster) and the `persist_stm_fit` records.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/stm_endtrim_grade.py \
      base:work/039252_2_d32r3base fix060:work/039252_2_d32r3fix \
      --axis 109:343.1,140.1,195.3:221.4,196.8,253.7 --core-extent 149.4
"""
import argparse
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

CLUSTERING = "data/0/0-clustering-global.json"
STM_FIT = "data/0/0-stm_fit-global.json"
STM_VERDICT = re.compile(r"cluster (\d+) .*?STM=([01])")


def load(workdir):
    zp = os.path.join(workdir, "mabc-pr.zip")
    with zipfile.ZipFile(zp) as z:
        cl = json.loads(z.read(CLUSTERING))
        ft = json.loads(z.read(STM_FIT))
    P = np.stack([cl["x"], cl["y"], cl["z"]], 1).astype(float)
    F = np.stack([ft["x"], ft["y"], ft["z"]], 1).astype(float)
    return P, F, np.asarray(ft["cluster_id"]), np.asarray(cl["cluster_id"])


def verdicts(workdir):
    out = {}
    for log in glob.glob(os.path.join(workdir, "wct_pr_*.log")):
        for line in open(log, errors="replace"):
            m = STM_VERDICT.search(line)
            if m:
                out[int(m.group(1))] = int(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="TAG:workdir")
    ap.add_argument("--axis", action="append", default=[],
                    help="cid:x1,y1,z1:x2,y2,z2 -- profile this cluster along A->B")
    ap.add_argument("--core-extent", type=float, default=None,
                    help="cm along the axis where the cluster's own points stop; "
                         "trajectory beyond this is a tail, not coverage")
    args = ap.parse_args()

    base_v = None
    for spec in args.arms:
        tag, wd = spec.split(":", 1)
        P, F, fc, cc = load(wd)
        tree = cKDTree(P)
        d, _ = tree.query(F)
        print(f"\n=== {tag}  ({wd})")
        print(f"  {len(F)} stm_fit points over {len(set(fc.tolist()))} clusters")
        print("  unsupported (distance to the nearest reconstructed 3-D charge):")
        for thr in (1.0, 2.0, 5.0, 10.0):
            print(f"      > {thr:4.1f} cm : {int((d > thr).sum()):5d}  ({100*(d > thr).mean():5.1f} %)")

        v = verdicts(wd)
        n1 = sum(1 for x in v.values() if x == 1)
        print(f"  STM verdicts: {len(v)} clusters, STM=1 on {n1}")
        if base_v is None:
            base_v = (tag, v)
        elif v != base_v[1]:
            flips = sorted(c for c in set(v) | set(base_v[1])
                           if v.get(c) != base_v[1].get(c))
            print(f"  STM flips vs {base_v[0]}: " +
                  ", ".join(f"cluster {c} {base_v[1].get(c)}->{v.get(c)}" for c in flips))
        else:
            print(f"  STM verdicts identical to {base_v[0]}")

        for spec_ax in args.axis:
            cid, a, b = spec_ax.split(":")
            cid = int(cid)
            A = np.array([float(x) for x in a.split(",")])
            B = np.array([float(x) for x in b.split(",")])
            u = (B - A) / np.linalg.norm(B - A)
            m = fc == cid
            if not m.any():
                print(f"  cluster {cid}: no stm_fit")
                continue
            Ff = F[m]
            t = (Ff - A) @ u
            dd = d[m]
            print(f"  cluster {cid}: {m.sum()} fit pts, t = [{t.min():.1f}, {t.max():.1f}] cm")
            ce = args.core_extent
            bands = [(-1e9, 1e9, "whole trajectory")] if ce is None else [
                (-1e9, ce, "within the cluster's own extent"),
                (ce, 1e9, "past it (a tail, not coverage)")]
            for lo, hi, lbl in bands:
                k = (t >= lo) & (t < hi)
                if k.sum():
                    print(f"      {lbl:34s} n={int(k.sum()):4d}  "
                          f"median {np.median(dd[k]):5.2f} cm from charge, "
                          f"{100*(dd[k] > 2).mean():5.1f} % beyond 2 cm")


if __name__ == "__main__":
    main()

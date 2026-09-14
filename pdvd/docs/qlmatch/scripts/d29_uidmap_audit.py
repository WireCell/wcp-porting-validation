#!/usr/bin/env python3
"""doc qlmatch/29 -- is the scorer's geometric uid map sound across the imaging changes?  (read-only)

    python3 d29_uidmap_audit.py --truth-pairs keep:p99wflip keep:p100flip --cluster-pairs p99wflip:p100flip > ../d29/uidmap_audit.txt

--truth-pairs REF:TAG: the objective long truth entries (long in REF = keep, the truth's clustering) mapped REF -> TAG by
ql_agree_score.build_uid_map, per drift volume and y/z quadrant: mapped / unmapped, the mapped cluster's (y,z) centroid
distance from the REF cluster (median, p90), how many land > 100 cm away in y (a face-swap artefact would put whole anodes
there), and how many have any bundle at the truth flash.
--cluster-pairs A:B: the same census over ALL long clusters of A (no truth), as a control between two v7 arms.
"""
import argparse
import collections
import math

import numpy as np

import d29_common as C


def run(ref, tag, use_truth, T):
    stats = collections.defaultdict(collections.Counter)
    dists = collections.defaultdict(list)
    for idx in C.IDX:
        evt = C.S.evt_of_idx(idx)
        R, A = C.Arm(ref, idx, ref=ref), C.Arm(tag, idx, ref=ref)
        if use_truth:
            items = [(e["uid"], e) for e in T[evt] if e["conf"] in C.S.OBJECTIVE_TIERS]
        else:
            items = [(u, None) for u in R.clusters]
        for u, e in items:
            c = R.clusters.get(u)
            if c is None or not c.get("y") or not C.is_long(c):
                continue
            r = C.region(c)
            s = stats[r]
            s["entries" if use_truth else "clusters"] += 1
            nu = A.map.get(u)
            if nu is None:
                s["unmapped"] += 1
                continue
            s["mapped"] += 1
            (y0, z0), (y1, z1) = C.centroid(c), C.centroid(A.clusters[nu])
            dists[r].append(math.hypot(y1 - y0, z1 - z0))
            s["dy>100cm"] += abs(y1 - y0) > 100
            if use_truth:
                s["bundle at truth flash"] += any(abs(t - e["time"]) <= C.TOL for t, _ in A.bundles.get(nu, ()))
    print(f"== {'truth entries (objective, long in ' + ref + ')' if use_truth else 'all long clusters of ' + ref}: {ref} -> {tag}")
    tot = collections.Counter()
    for r in sorted(stats):
        tot.update(stats[r])
        d = np.array(dists[r]) if dists[r] else np.array([np.nan])
        print(f"  {r:16s} {dict(stats[r])}  centroid d(y,z) median {np.nanmedian(d):.1f} cm, p90 {np.nanpercentile(d, 90):.1f}")
    alld = np.array([x for v in dists.values() for x in v])
    print(f"  ALL {dict(tot)}  centroid d(y,z) median {np.median(alld):.1f} cm, p90 {np.percentile(alld, 90):.1f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-pairs", nargs="*", default=[])
    ap.add_argument("--cluster-pairs", nargs="*", default=[])
    a = ap.parse_args()
    T = C.truth()
    print("# doc qlmatch/29 -- uid-map audit (d29_uidmap_audit.py); run 039252, 18 events")
    for p in a.truth_pairs:
        run(*p.split(":"), True, T)
    for p in a.cluster_pairs:
        run(*p.split(":"), False, T)


if __name__ == "__main__":
    main()

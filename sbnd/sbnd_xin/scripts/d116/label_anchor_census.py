#!/usr/bin/env python3
"""doc sbnd_xin/116 sec 16.2: how much of doc pr/150's data vertex metric is anchored to the
configuration it was measuring against.

pr/150 scored the vertex on data as "the arm's own candidate vertex within 3 cm of the hand label"
(677 -> 609 of 823 for `tfull`).  A label in `vertex_labels/vtxscan-vtx105-*/labels-evt*.json` is
not a free 3-D click: `picks[]` are rows of the BASE arm's candidate list (`kind: "candidate"`,
with the base arm's own `vertex_id`), and the file also stores that arm's shipped `main_vertex`.
This measures the distance between the two, which is the size of the anchor: where the pick IS the
base arm's main vertex, the metric asks whether a new configuration still agrees with the old one,
not whether either is right.

Read-only; it touches nothing under vertex_labels/ (M13 -- these are scan records).

Usage: python3 scripts/d116/label_anchor_census.py [tag ...] > docs/116_figs/116_label_anchor.txt
       default tags = the numu-stream vtx105 sets pr/150's 823 events come from
"""
import collections
import glob
import json
import math
import os
import sys

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT = ["vtxscan-vtx105-mcp1k", "vtxscan-vtx105-mcp2k", "vtxscan-vtx105-mcp2k-auto",
           "vtxscan-vtx105-mcp2k-ragree", "vtxscan-vtx105-delta"]


def main(tags):
    print("# doc 116 sec 16.2 -- vtx105 label picks against the base arm's own main vertex")
    by = collections.defaultdict(list)
    kinds = collections.Counter()
    nfile = nopick = 0
    for t in tags:
        for f in sorted(glob.glob(os.path.join(SX, "vertex_labels", t, "labels-evt*.json"))):
            j = json.load(open(f))
            nfile += 1
            mv, ps = j.get("main_vertex") or {}, j.get("picks") or []
            if not ps:
                nopick += 1
                continue
            for p in ps:
                kinds[p.get("kind")] += 1
            if not mv:
                continue
            p = ps[0]
            by[j.get("label_source")].append(
                math.dist([p["x"], p["y"], p["z"]], [mv["x"], mv["y"], mv["z"]]))
    print(f"tags      : {' '.join(tags)}")
    print(f"label files: {nfile}   without a pick: {nopick}   pick kinds: {dict(kinds)}")
    print("\n| label source | n | pick IS the base arm's main vertex (0.000 cm) | within 3 cm | median |")
    print("|---|---:|---:|---:|---:|")
    allv = []
    for k in sorted(by):
        v = sorted(by[k])
        allv += v
        n = len(v)
        z = sum(1 for x in v if x == 0.0)
        t3 = sum(1 for x in v if x < 3.0)
        print(f"| {k} | {n} | {z} ({100.0*z/n:.1f} %) | {t3} ({100.0*t3/n:.1f} %) | {v[n//2]:.3f} cm |")
    v = sorted(allv)
    n = len(v)
    z = sum(1 for x in v if x == 0.0)
    t3 = sum(1 for x in v if x < 3.0)
    print(f"| all | {n} | {z} ({100.0*z/n:.1f} %) | {t3} ({100.0*t3/n:.1f} %) | {v[n//2]:.3f} cm |")
    print(f"\np90 {v[int(0.9*n)]:.2f} cm   max {v[-1]:.1f} cm")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)

#!/usr/bin/env python3
"""doc pr/91 round 1 -- "which PF object owns this 3-D point?"

The owner hands over a point read off a Bee display and asks what is there.
Answering it against segment ENDPOINTS is wrong and has misled us twice
(doc pr/84 sec 23, evt 285567): a shower's interior is drawn as a single
straight start->end line, so the point of interest is usually mid-segment.
This scans every FIT/TRAJECTORY point of every segment in the calib dump.

Output, nearest first: distance, segment display id, cluster, length, pdg,
flag_shower, the segment's `shower_id` join key (-1 = in NO shower), and the
index of the nearest fitted point within the segment.

Repro:
  scripts/pr91_point_owner.py work-pr91r1-dbg-mc/pr_evt169626/calib-pr-evt169626.json \
      -4.5 157.5 442.9
"""
import json
import math
import sys


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    path = sys.argv[1]
    px, py, pz = (float(a) for a in sys.argv[2:5])
    topn = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    d = json.load(open(path))

    shower_of = {s["id"]: s for s in d.get("showers", [])}
    rows = []
    for seg in d.get("segments", []):
        best, bi = 1e9, -1
        for i, p in enumerate(seg.get("points", [])):
            dd = math.dist((px, py, pz), (p["x"], p["y"], p["z"]))
            if dd < best:
                best, bi = dd, i
        if bi < 0:
            continue
        rows.append((best, seg, bi))
    rows.sort(key=lambda r: r[0])

    print(f"# point ({px}, {py}, {pz})   file {path}")
    print(f"# {'d_traj':>8} {'seg':>7} {'clus':>5} {'len':>8} {'pdg':>6} "
          f"{'shower':>7} {'flag_shower':>11} {'pt_idx':>7} {'npts':>5}  owner")
    for best, seg, bi in rows[:topn]:
        sid = seg.get("shower_id", -1)
        sh = shower_of.get(sid)
        owner = "IN NO SHOWER" if sid == -1 else (
            f"shower {sid} conn={sh['start_connection_type']} "
            f"{sh['kine_best']:.1f}MeV nseg={sh['num_segments']}" if sh else f"shower {sid}")
        print(f"  {best:8.3f} {seg['id']:>7} {seg['cluster_id']:>5} {seg['length']:8.3f} "
              f"{seg['particle_id']:>6} {sid:>7} {str(seg['flag_shower']):>11} "
              f"{bi:>7} {len(seg['points']):>5}  {owner}")

    # Nearest vertex too -- a point read off a Bee node's endpoint marker is
    # usually a vertex, and the vertices[] id is the join key into segments[].
    vbest = sorted(d.get("vertices", []),
                   key=lambda v: math.dist((px, py, pz),
                                           (v["fit"]["x"], v["fit"]["y"], v["fit"]["z"])))[:3]
    print("# nearest vertices")
    for v in vbest:
        dd = math.dist((px, py, pz), (v["fit"]["x"], v["fit"]["y"], v["fit"]["z"]))
        print(f"  {dd:8.3f} vtx {v['id']} cluster={v['cluster_id']} degree={v['degree']} "
              f"is_main={v['is_main']}")


if __name__ == "__main__":
    main()

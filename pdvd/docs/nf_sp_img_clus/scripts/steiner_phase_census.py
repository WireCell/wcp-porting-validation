#!/usr/bin/env python3
"""doc pdvd/31 sections 4.1 and 5.3: the per-phase Steiner terminal census.

Parses one PR log produced with WCT_STEINER_PHASE_DUMP=1 and `-L clus:trace`
and answers, for every `create_steiner_tree` call in the event:

  * how many terminals each phase kept (from the TRACE counts already in
    SteinerGrapher.cxx:39,55,68,78 -- no probe needed for these), and
  * WHERE they are (from the env-gated per-terminal dump added for doc 31),
    split by the two halves of the 039349/14 track.

The second question is the one a count cannot answer, and it is the whole
reason the dump exists: round 1 inferred the terminal population from blob
occupancy of the INPUT point cloud and got it wrong, because
CreateSteinerGraph RETILES -- the steiner stage runs on a different, denser
cluster (5705 points where the input cluster has 2348).  Only positions dumped
from inside the stage settle it.

The track (doc 26 section 7.5): V is where the steiner cloud stopped, A is the
far end of the starved 111 cm, U is the far end of the healthy control half.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/steiner_phase_census.py \
      work/039349_14_d31r3off/wct_pr_dump.log \
      work/039349_14_d31r3on/wct_pr_dump.log
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np

# doc 26 section 7.5, cm.
V_CM = np.array([273.26, -118.90, 86.61])
A_CM = np.array([196.86, -167.76, 151.48])   # far end of the starved half
U_CM = np.array([392.0, -83.0, 5.0])         # far end of the healthy half
R_CM = 3.0                                   # corridor radius about the axis

PT = re.compile(r"steiner_phase_pt: npts=(\d+) nterm=(\d+) phase=(\S+) "
                r"x=(-?[\d.]+) y=(-?[\d.]+) z=(-?[\d.]+)")
REF = re.compile(r"reference cluster filtering: (\d+) -> (\d+) terminals"
                 r" \(wire_tol=(\S+) adjacent_slice=(\S+)\)")
PATH = re.compile(r"path filtering: (\d+) -> (\d+) terminals")
FIND = re.compile(r"found (\d+) initial steiner terminals")

PHASES = ["P0_cluster", "P1_find", "P2_refcluster", "P3_path"]


def corridor(P, end):
    """Mask: within R_CM of the V->end segment, excluding the very ends."""
    if len(P) == 0:
        return np.zeros(0, bool)
    d = end - V_CM
    t = np.clip(((P - V_CM) @ d) / (d @ d), 0, 1)
    perp = np.linalg.norm(P - (V_CM + t[:, None] * d), axis=1)
    return (perp < R_CM) & (t > 0.02) & (t < 0.98)


def largest_gap(P, end):
    """Largest terminal-free gap (cm) along the V->end axis -- doc 31's metric."""
    m = corridor(P, end)
    if m.sum() < 2:
        return float("nan")
    d = end - V_CM
    L = np.linalg.norm(d)
    t = np.sort(((P[m] - V_CM) @ d) / (d @ d)) * L
    return float(np.max(np.diff(np.concatenate([[0.0], t, [L]]))))


START = re.compile(r"create_steiner_tree: starting with reference_cluster=")


def parse(path):
    """One dict per create_steiner_tree call.

    The call boundary is the "starting with reference_cluster=" TRACE line --
    the ONLY reliable delimiter.  An earlier version inferred boundaries from
    npts/phase transitions and silently merged calls, which made the corridor
    selection pick a cluster the track is not in.  Anchor on the delimiter the
    code actually emits.
    """
    calls = []
    cur = None
    counts = {"p2_removed": 0, "p2_total": 0, "p3_removed": 0, "p3_total": 0,
              "p3_calls": 0, "p3_nonzero": 0, "ncalls": 0}
    with open(path, errors="replace") as fp:
        for line in fp:
            if START.search(line):
                cur = {"npts": 0, "pts": defaultdict(list)}
                calls.append(cur)
                continue
            if "steiner_phase_pt:" in line:
                m = PT.search(line)
                if not m or cur is None:
                    continue
                cur["npts"] = int(m.group(1))
                cur["pts"][m.group(3)].append(
                    (float(m.group(4)), float(m.group(5)), float(m.group(6))))
                continue
            m = FIND.search(line)
            if m:
                counts["ncalls"] += 1
                if cur is not None:
                    cur["n_p1"] = int(m.group(1))
                continue
            m = REF.search(line)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                counts["p2_total"] += a
                counts["p2_removed"] += a - b
                counts["wire_tol"] = m.group(3)
                counts["adjacent_slice"] = m.group(4)
                continue
            m = PATH.search(line)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                counts["p3_calls"] += 1
                counts["p3_total"] += a
                counts["p3_removed"] += a - b
                if a != b:
                    counts["p3_nonzero"] += 1
    return calls, counts


def report(path):
    calls, c = parse(path)
    print(f"\n{'='*78}\n## {path}\n{'='*78}")
    print(f"create_steiner_tree calls: {c['ncalls']}")
    print(f"  phase 2 (reference cluster): removed {c['p2_removed']} of {c['p2_total']}"
          f"  (wire_tol={c.get('wire_tol')} adjacent_slice={c.get('adjacent_slice')})")
    print(f"  phase 3 (path constraints) : removed {c['p3_removed']} of {c['p3_total']}"
          f"  in {c['p3_calls']} calls, {c['p3_nonzero']} of which removed anything")

    # The call that builds our track's tree: the one whose retiled cluster is
    # biggest.  Resolved by SIZE, never by a hardcoded index -- a rerun
    # reorders calls.
    if not calls:
        print("  (no per-terminal dump in this log: was WCT_STEINER_PHASE_DUMP set?)")
        return None
    # Resolve OUR track's call by geometry -- the retiled cluster holding the
    # most points in the two corridors about V.  Never by index or by a
    # hardcoded cluster id: a rerun reorders calls and re-numbers clusters.
    def coverage(d):
        P = np.array(d["pts"].get("P0_cluster", []))
        if len(P) == 0:
            return -1
        return int(corridor(P, A_CM).sum()) + int(corridor(P, U_CM).sum())
    big = max(calls, key=coverage)
    print(f"\n  track's retiled cluster: {big['npts']} points, "
          f"{coverage(big)} of them in the two corridors "
          f"(of {len(calls)} calls dumped)")
    print(f"  {'phase':<16s} {'terminals':>9s} {'below V':>8s} {'above V':>8s} "
          f"{'gap below':>10s} {'gap above':>10s}")
    out = {}
    for ph in PHASES:
        if ph not in big["pts"]:
            continue
        P = np.array(big["pts"][ph])
        nb, na = int(corridor(P, A_CM).sum()), int(corridor(P, U_CM).sum())
        gb, ga = largest_gap(P, A_CM), largest_gap(P, U_CM)
        print(f"  {ph:<16s} {len(P):>9d} {nb:>8d} {na:>8d} {gb:>10.1f} {ga:>10.1f}")
        out[ph] = (len(P), nb, na, gb, ga)
    return out


def main():
    logs = sys.argv[1:] or ["work/039349_14_d31r3off/wct_pr_dump.log",
                            "work/039349_14_d31r3on/wct_pr_dump.log"]
    res = {}
    for path in logs:
        if not os.path.exists(path):
            print(f"## {path}: MISSING, skipped")
            continue
        res[path] = report(path)
    if len(res) == 2:
        (ka, a), (kb, b) = list(res.items())
        if a and b:
            print(f"\n{'='*78}\n## OFF -> ON, terminals below V (the starved half)\n{'='*78}")
            print(f"  {'phase':<16s} {'OFF':>8s} {'ON':>8s} {'gap OFF':>10s} {'gap ON':>10s}")
            for ph in PHASES:
                if ph in a and ph in b:
                    print(f"  {ph:<16s} {a[ph][1]:>8d} {b[ph][1]:>8d} "
                          f"{a[ph][3]:>10.1f} {b[ph][3]:>10.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pdvd/31 section 9.2: what charge does the RETILED cluster actually carry?

Round 2 named Phase 1 as the stage that fails to make terminals below the
vertex.  Round 3 fixed the charge bug that looked like the cause and measured
that it changed nothing.  Neither round could see the quantity Phase 1's gate
actually reads, because `CreateSteinerGraph` RETILES the cluster before building
the tree and the retiled point cloud is never persisted -- it exists only inside
the PR process.

This script reads it out of the env-gated dump (`WCT_STEINER_PHASE_DUMP=1`,
toolkit default OFF, log-only), which since round 4 appends the three per-plane
charges to every `steiner_phase_pt:` line.

Two questions, both from one log:

  1. per-plane charge of the retiled cluster's own points, split by the two
     halves of the 039349/14 track.  The number that matters is
     **n_nonzero_planes >= 2**, because `Cluster::calc_charge_wcp`
     (Facade_Cluster.cxx:1105) returns charge 0 unless more than one plane is
     non-zero -- and `find_peak_point_indices` then requires
     `charge > terminal_charge_threshold`, so a point with one live plane can
     never be a terminal however much charge that plane holds.

  2. the operands of `filter_by_path_constraints` (Phase 3), which round 2
     measured removes NOTHING on every call.  A count cannot say whether that is
     by design or because the operands are degenerate, and there is a specific
     way it could be degenerate: `DynamicPointCloud::get_closest_2d_point_info`
     returns a raw -1.0 when the (plane, face, apa) 2-D tree is empty
     (DynamicPointCloud.cxx:375-377), and -1 < 1.8 cm, so a sentinel would read
     as "very close" and collapse the test into a bare `dis_3d > 6 cm` cut.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/steiner_retile_charge_census.py \
      work/039349_14_d31r4dump/wct_pr_dump.log
"""
import re
import sys

import numpy as np

# doc 26 section 7.5, cm.  V is where the steiner cloud stops, A is the far end
# of the starved 111 cm, U the far end of the healthy control half.
V_CM = np.array([273.26, -118.90, 86.61])
A_CM = np.array([196.86, -167.76, 151.48])
U_CM = np.array([392.0, -83.0, 5.0])
R_CM = 3.0

# The charges are appended LAST so that older parsers, which match a prefix with
# re.search, keep working against a round-4 log.
PT = re.compile(r"steiner_phase_pt: npts=(\d+) nterm=(\d+) phase=(\S+) "
                r"x=(-?[\d.]+) y=(-?[\d.]+) z=(-?[\d.]+) "
                r"cu=(-?[\d.e+]+) cv=(-?[\d.e+]+) cw=(-?[\d.e+]+)")
P3 = re.compile(r"steiner_p3_dis: dis3d=(-?[\d.]+) du=(-?[\d.]+) dv=(-?[\d.]+) "
                r"dw=(-?[\d.]+) close2d=(\d) remove=(\d)")
START = re.compile(r"create_steiner_tree: starting with reference_cluster=")

PHASES = ["P0_cluster", "P1_find", "P2_refcluster", "P3_path"]


def corridor(P, end):
    """Mask: within R_CM of the V->end segment, excluding the very ends."""
    if len(P) == 0:
        return np.zeros(0, bool)
    d = end - V_CM
    t = np.clip(((P - V_CM) @ d) / (d @ d), 0, 1)
    perp = np.linalg.norm(P - (V_CM + t[:, None] * d), axis=1)
    return (perp < R_CM) & (t > 0.02) & (t < 0.98)


def parse(path):
    """One dict per create_steiner_tree call, plus the flat Phase-3 rows.

    The call boundary is the "starting with reference_cluster=" TRACE line --
    the only reliable delimiter.  Inferring boundaries from npts/phase
    transitions silently merges calls (that mistake once produced a wholly
    fictitious 206 -> 1095 terminal count).
    """
    calls, cur, p3 = [], None, []
    with open(path, errors="replace") as fp:
        for line in fp:
            if START.search(line):
                cur = {}
                calls.append(cur)
                continue
            if "steiner_phase_pt:" in line:
                m = PT.search(line)
                if not m or cur is None:
                    continue
                cur.setdefault(m.group(3), []).append(
                    [float(m.group(i)) for i in (4, 5, 6, 7, 8, 9)])
                continue
            if "steiner_p3_dis:" in line:
                m = P3.search(line)
                if m:
                    p3.append([float(m.group(i)) for i in (1, 2, 3, 4)]
                              + [int(m.group(5)), int(m.group(6))])
    return calls, np.array(p3) if p3 else np.zeros((0, 6))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 \
        else "work/039349_14_d31r4dump/wct_pr_dump.log"
    calls, p3 = parse(path)
    if not calls:
        print(f"{path}: no per-phase dump -- was WCT_STEINER_PHASE_DUMP=1 set?")
        return

    # Resolve OUR track's call by geometry -- the retiled cluster covering the
    # two corridors about V best.  Never by index or cluster id: a rerun
    # reorders the calls and renumbers the clusters.
    def coverage(d):
        P = np.array(d.get("P0_cluster", []))
        if len(P) == 0:
            return -1
        return int(corridor(P[:, :3], A_CM).sum()) \
            + int(corridor(P[:, :3], U_CM).sum())

    big = max(calls, key=coverage)
    print(f"{'='*88}\n## {path}\n{'='*88}")
    print(f"create_steiner_tree calls: {len(calls)};  track's retiled cluster: "
          f"{len(big.get('P0_cluster', []))} points, {coverage(big)} in the corridors")

    print(f"\n### 1. per-plane charge of the RETILED cluster "
          f"(what Phase 1's gate reads)\n")
    print(f"  {'phase':<14s} {'half':<8s} {'n':>5s} "
          f"{'U!=0':>6s} {'V!=0':>6s} {'W!=0':>6s} "
          f"{'0pl':>6s} {'1pl':>6s} {'>=2pl':>6s}   <- >=2 is the candidacy gate")
    for ph in PHASES:
        if ph not in big:
            continue
        P = np.array(big[ph])
        for name, end in (("below V", A_CM), ("above V", U_CM)):
            m = corridor(P[:, :3], end)
            Q = P[m][:, 3:6]
            if len(Q) == 0:
                print(f"  {ph:<14s} {name:<8s} {0:>5d}")
                continue
            nz = (Q != 0).sum(axis=1)
            print(f"  {ph:<14s} {name:<8s} {len(Q):>5d} "
                  f"{np.mean(Q[:, 0] != 0):>6.3f} {np.mean(Q[:, 1] != 0):>6.3f} "
                  f"{np.mean(Q[:, 2] != 0):>6.3f} "
                  f"{np.mean(nz == 0):>6.3f} {np.mean(nz == 1):>6.3f} "
                  f"{np.mean(nz >= 2):>6.3f}")

    Q = np.array(big["P0_cluster"])[:, 3:6]
    nz = (Q != 0).sum(axis=1)
    print(f"\n  whole retiled cluster (n={len(Q)}): U!=0 {np.mean(Q[:, 0] != 0):.3f} "
          f"V!=0 {np.mean(Q[:, 1] != 0):.3f} W!=0 {np.mean(Q[:, 2] != 0):.3f} "
          f">=2 planes {np.mean(nz >= 2):.3f}")
    print("  (the cluster spans several (apa, face) volumes.  Both corridors lie"
          " on a4f0 -- doc 31 section 5 matched their points to ctpc_a4f0p* 194/195"
          " and 709/721 -- and a4f0's U plane is the one whose 98 wrapped wires sit"
          " at wire index 0, so EVERY U index there is shifted by 98.  U is exactly"
          " 0 on all corridor points; elsewhere in the cluster it is not.)")

    # ---- the owner's three properties, section 6.1 --------------------------
    # "on the track or the vertex" / "not too many" / "not too few" is three
    # separate requirements, and doc 31 section 6.1 originally had a metric for
    # only one of them.  Measured here against the track's OWN two-segment
    # skeleton (V->A below, V->U above), so no corridor pre-cut biases the
    # localization number: every terminal of the call is assigned to whichever
    # of the two axes it is nearer.
    if "P3_path" in big:
        T = np.array(big["P3_path"])[:, :3]
        len_below = float(np.linalg.norm(A_CM - V_CM))
        len_above = float(np.linalg.norm(U_CM - V_CM))

        def perp_to(P, end):
            d = end - V_CM
            t = np.clip(((P - V_CM) @ d) / (d @ d), 0, 1)
            return np.linalg.norm(P - (V_CM + t[:, None] * d), axis=1)

        mb, ma = corridor(T, A_CM), corridor(T, U_CM)
        pb, pa = perp_to(T, A_CM), perp_to(T, U_CM)
        n_b, n_a = int(mb.sum()), int(ma.sum())
        print(f"\n### 3. the three properties of section 6.1, on this track "
              f"({len(T)} terminals in the call)\n")
        print(f"  not too few  -- terminals per cm of skeleton: "
              f"below V {n_b} over {len_below:.0f} cm = {n_b/len_below:.3f}/cm   "
              f"above V {n_a} over {len_above:.0f} cm = {n_a/len_above:.3f}/cm")
        print(f"  not too many -- the same numbers read the other way: "
              f"one terminal per {len_below/max(n_b,1):.0f} cm below V, "
              f"per {len_above/max(n_a,1):.1f} cm above V")
        for name, m, p in (("below V", mb, pb), ("above V", ma, pa)):
            if m.sum():
                print(f"  on the track ({name}, n={int(m.sum())}): perpendicular "
                      f"distance to the axis median {np.median(p[m]):.2f} cm, "
                      f"p90 {np.percentile(p[m], 90):.2f} cm, max {p[m].max():.2f} cm")
        print("  NOTE the localization row is bounded above by the 3 cm corridor"
              " that selects it, so it is an instrument for the SHAPE inside the"
              " corridor only.  The metric section 6.1 asks for is distance to the"
              " FITTED skeleton, which is a PR product this log does not carry;"
              " building it is named as owed work, not reported here.")

    if len(p3):
        d3, du, dv, dw = p3[:, 0], p3[:, 1], p3[:, 2], p3[:, 3]
        c2, rm = p3[:, 4], p3[:, 5]
        neg = (du < 0) | (dv < 0) | (dw < 0)
        second = np.sort(np.stack([du, dv, dw]), axis=0)[1]
        print(f"\n### 2. Phase 3 (filter_by_path_constraints) operands, "
              f"all {len(p3)} terminals tested event-wide\n")
        print(f"  2-D distance == the -1.0 empty-tree sentinel : "
              f"{int(neg.sum())} ({neg.mean():.4f})")
        print(f"  close_in_2d (two of three planes < 1.8 cm)   : "
              f"{int(c2.sum())} ({c2.mean():.4f})")
        print(f"  dis_3d > 6 cm                                : "
              f"{int((d3 > 6).sum())} ({(d3 > 6).mean():.4f})")
        print(f"  BOTH  (= removed)                            : "
              f"{int(((c2 > 0) & (d3 > 6)).sum())}   [reported removals: "
              f"{int(rm.sum())}]")
        print(f"  of the {int((d3 > 6).sum())} far-in-3D terminals, "
              f"2nd-smallest 2-D distance < 1.8 cm on "
              f"{int(((d3 > 6) & (second < 1.8)).sum())}")
        print(f"  dis_3d  p50={np.percentile(d3, 50):.2f} "
              f"p90={np.percentile(d3, 90):.2f} max={d3.max():.2f} cm")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pdvd/31 round 6: the terminal DENSITY census.

The owner's hand scan of Bee set A (039349/14, f4edc748) returned "the
steiner-global cloud is good, continuous; the terminals are too dense -- don't
we have some local maximum selection?".  We do.  This script answers both
halves of that with numbers rather than with a reading of the code:

  Section 1 -- WHY the peak finder cannot thin them.
      find_steiner_terminals (SteinerGrapher.cxx:681) calls
      find_peak_point_indices once PER BLOB, so the nlevel=1 suppression never
      crosses a blob boundary and a blob's LAST candidate can never be removed.
      The env-gated `steiner_p1_blobs` probe prints, per call,
      nblob / ncand_blob / nterm.  nterm == ncand_blob on every call is the
      signature of that floor.

  Section 2 -- WHAT the density is, on the arms that matter, using section
      9.4's three metrics (coverage / density / localization) over the same
      corridor about the doc 26 section 7.5 axis.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/steiner_density_census.py \
      work/039349_14_d31r5off work/039349_14_d31r6prod work/039349_14_d31r6sync
"""
import json
import os
import re
import sys

import numpy as np

# doc 26 section 7.5, cm.  V is the vertex at which the steiner cloud stopped in
# production; A is the far end of the starved half, U the far end of the control.
V_CM = np.array([273.26, -118.90, 86.61])
A_CM = np.array([196.86, -167.76, 151.48])
U_CM = np.array([392.0, -83.0, 5.0])
R_CM = 3.0

BLOBS = re.compile(r"<(\S+?):\S+?> steiner_p1_blobs: nblob=(\d+) ncand_blob=(\d+) nterm=(\d+)")


def corridor(P, end):
    """Mask + axial coordinate for points within R_CM of the V->end segment."""
    if len(P) == 0:
        return np.zeros(0, bool), np.zeros(0)
    d = end - V_CM
    t = np.clip(((P - V_CM) @ d) / (d @ d), 0, 1)
    perp = np.linalg.norm(P - (V_CM + t[:, None] * d), axis=1)
    return (perp < R_CM) & (t > 0.02) & (t < 0.98), t


def largest_gap(t_sorted, length):
    """Largest run along the axis with no point, in cm."""
    if len(t_sorted) == 0:
        return length
    edges = np.concatenate(([0.0], np.sort(t_sorted), [1.0]))
    return float(np.diff(edges).max() * length)


def perp_distance(P, end):
    d = end - V_CM
    t = np.clip(((P - V_CM) @ d) / (d @ d), 0, 1)
    return np.linalg.norm(P - (V_CM + t[:, None] * d), axis=1)


def blob_floor(workdir):
    """Section 1: does every candidate-bearing blob yield exactly one terminal?"""
    log = os.path.join(workdir, "wct_pr_dump.log")
    if not os.path.exists(log):
        print("    (no wct_pr_dump.log -- rerun with WCT_STEINER_PHASE_DUMP=1)")
        return
    per_comp = {}
    for line in open(log, errors="replace"):
        m = BLOBS.search(line)
        if not m:
            continue
        comp, nblob, ncand, nterm = m.group(1), *(int(x) for x in m.groups()[1:])
        d = per_comp.setdefault(comp, dict(calls=0, nblob=0, ncand=0, nterm=0, equal=0))
        d["calls"] += 1
        d["nblob"] += nblob
        d["ncand"] += ncand
        d["nterm"] += nterm
        d["equal"] += (ncand == nterm)
    for comp in sorted(per_comp):
        d = per_comp[comp]
        print(f"    {comp:22s} calls={d['calls']:4d}  blobs={d['nblob']:6d}  "
              f"candidate-bearing={d['ncand']:6d}  terminals={d['nterm']:6d}  "
              f"nterm==ncand on {d['equal']}/{d['calls']} calls")


def steiner_metrics(workdir):
    """Section 2: section 9.4's three metrics on the cluster owning the control half."""
    hits = [f for f in os.listdir(workdir) if f.startswith("calib-pr-") and f.endswith(".json")]
    if not hits:
        print("    (no calib-pr dump)")
        return
    entries = json.load(open(os.path.join(workdir, hits[0]))).get("steiner", [])
    # Resolve the cluster by OWNERSHIP of the control region -- re-clustering and
    # even re-tiling renumber it, and a hardcoded id silently selects nothing.
    best, best_n = None, -1
    for e in entries:
        P = np.stack([e["x"], e["y"], e["z"]], 1)
        n = corridor(P, U_CM)[0].sum()
        if n > best_n:
            best, best_n = e, n
    if best is None:
        return
    P = np.stack([best["x"], best["y"], best["z"]], 1)
    term = np.array(best["flag_terminal"], bool)
    print(f"    owner cluster_id={best['cluster_id']} main={best['is_main_cluster']} "
          f"points={len(P)} terminals={int(term.sum())}")
    for name, end in (("below V (restored)", A_CM), ("above V (control) ", U_CM)):
        length = float(np.linalg.norm(end - V_CM))
        m, t = corridor(P, end)
        npt, nterm = int(m.sum()), int((m & term).sum())
        gap = largest_gap(t[m], length)
        dperp = perp_distance(P[m & term], end)
        med = float(np.median(dperp)) if len(dperp) else float("nan")
        print(f"      {name} len={length:6.1f}cm  points={npt:5d} ({npt/length:5.2f}/cm)  "
              f"terminals={nterm:4d} ({nterm/length:5.2f}/cm, 1 per {length/nterm if nterm else float('inf'):5.2f}cm, "
              f"{nterm/npt if npt else 0:.3f}/point)  gap={gap:5.1f}cm  median_perp={med:4.2f}cm")


def main():
    for wd in sys.argv[1:]:
        print(f"== {os.path.basename(wd)}")
        print("  section 1: the per-blob floor")
        blob_floor(wd)
        print("  section 2: coverage / density / localization")
        steiner_metrics(wd)
        print()


if __name__ == "__main__":
    main()

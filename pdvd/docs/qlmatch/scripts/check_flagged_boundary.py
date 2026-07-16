#!/usr/bin/env python3
"""Self-consistency of QLMatching's OWN boundary-flagged selections vs the
anode position -- NO external truth (hand labels, production pairs) used.

Motivation (2026-07-12, after the FV anode moved to the U plane, toolkit
b8f7f3d6): during QLMatch tuning the correct T0 is unknown, so validated
pairs from the previous convention cannot arbitrate.  This script asks only:
for the CURRENT auto result, do the clusters QLMatching itself flagged as
boundary have ends consistent with the anode (u = 0) at their own selected
flash's T0?

Three views, in decreasing circularity:

  1. anode-facing PCA-end u of auto-selected bundles with the two_boundary
     flag (QLMatching.cxx:3722 compute_two_boundary_flag) and with the
     anode-window at_x_boundary flag (:3639, window [anode_ext1,anode_ext2]).
     CAVEAT: largely circular -- two_boundary is only granted when an end is
     within m_two_boundary_margin = 3 cm of a face AT THAT FLASH'S T0, so the
     flagged population self-selects flashes that place the end at u ~ 0.
  2. the same for ALL auto-selected bundles (no flag gate) -- the unflagged
     bulk shows where ends actually pile.
  3. SPAN CLOSURE for two_boundary bundles whose two faces are anode+cathode:
     span = u_cathode_end - u_anode_end is T0-INDEPENDENT (a uniform drift
     shift cancels), so span ~ u_cathode (338.51 cm) is a physical test no
     flash choice can fake.  span > u_cathode + 2 cm = impossible (pileup
     merge / junk); the flag tolerates these because its nearest-face test
     uses SIGNED distance <= margin, so an end arbitrarily far OUTSIDE the
     anode (u << 0) still counts as "at the edge" (pre-existing diagnostic
     limitation, reported not fixed).

PCA-endpoint caveat: endpoints here are SVD principal-axis extremes of the
calib-dump point cloud -- a proxy for Cluster::get_extreme_wcps groups [0]/[1]
used by the C++; they can differ for blobby/merged clusters (some dumped
two_boundary=true bundles recompute here with an end far from any face).

Per-side flash time: apa 0 uses f["time"] (BDE offset folded), apa 4 uses
f["time1"] (per-side key from toolkit e587f357), falling back to
f["time"] + (off_top - off_bot) for older dumps.

Usage (from this directory):
  python3 scripts/check_flagged_boundary.py --tag anodefix   # current U-plane reproc
  python3 scripts/check_flagged_boundary.py                  # production baseline
"""

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> qlmatch doc dir
WORK = os.path.join(HERE, "..", "..", "work")
FACE = {0: "anode", 1: "cathode", 2: "-y", 3: "+y", 4: "-z", 5: "+z"}


def pca_extremes(P):
    C = P - P.mean(0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    proj = C @ vt[0]
    return P[proj.argmax()], P[proj.argmin()]


def stats(a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return "n=0"
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return "n=%3d median %+6.2f MAD %5.2f mean %+6.2f rms %5.2f [%+.2f,%+.2f]" % (
        len(a), med, mad, a.mean(), a.std(), a.min(), a.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="work-dir tag (strict match), e.g. anodefix")
    args = ap.parse_args()
    pat = re.compile(r"^039252_\d+%s$" %
                     (re.escape("_" + args.tag) if args.tag else ""))

    acc = {}                       # (group, side) -> [u of anode-facing ends]
    pairs = {}                     # two_boundary face-pair composition
    counts = dict(auto=0, tb=0, axb=0, axb_anode=0)
    ac_rows = []                   # anode-cathode two_boundary span closures

    for f in sorted(glob.glob(os.path.join(WORK, "039252_*", "calib-evt*.json"))):
        if not pat.match(os.path.basename(os.path.dirname(f))):
            continue
        d = json.load(open(f))
        ev = os.path.basename(f)[len("calib-evt"):-len(".json")]
        fb = {x["gid"]: x for x in d["flashes"]}
        cb = {c["uid"]: c for c in d["clusters"]}
        geom = {int(k): g for k, g in d["geometry"].items()}
        drift = d["drift_speed"]
        offs = d.get("trigger_offsets_us") or [0.0, 0.0]
        delta = offs[1] - offs[0]
        seen = set()
        for b in d["bundles"]:
            if not b.get("auto_selected"):
                continue
            uid = b["main_cluster"]
            key = (uid, b["flash_gid"])
            if uid == 3999999 or uid not in cb or key in seen:
                continue
            seen.add(key)
            c = cb[uid]
            g = geom[c["apa"]]
            fl = fb.get(b["flash_gid"])
            if fl is None:
                continue
            counts["auto"] += 1
            t = fl["time"] if c["apa"] == 0 else \
                fl.get("time1", fl["time"] + delta)
            P = np.column_stack([np.asarray(c["x"], float),
                                 np.asarray(c["y"], float),
                                 np.asarray(c["z"], float)])
            if len(P) < 3:
                continue
            p_hi, p_lo = pca_extremes(P)
            xo = g["sign_offset"] * t * drift
            ends = []
            for p in (p_hi, p_lo):
                u = g["s"] * (p[0] + xo - g["anode_x"])
                dd = [u, g["u_cathode"] - u, p[1] - g["y_lo"],
                      g["y_hi"] - p[1], p[2] - g["z_lo"], g["z_hi"] - p[2]]
                fc = int(np.argmin(dd))
                ends.append((fc, dd[fc], u))
            side = "bot" if c["apa"] == 0 else "top"
            tb = b.get("two_boundary", False)
            axb = b.get("at_x_boundary", False)
            atc = b.get("at_cathode", False)
            if tb:
                counts["tb"] += 1
            if axb:
                counts["axb"] += 1
            if axb and not atc:
                counts["axb_anode"] += 1
            for fc, dist, u in ends:
                if fc != 0:
                    continue
                acc.setdefault(("all-auto", side), []).append(u)
                if tb:
                    acc.setdefault(("two_boundary", side), []).append(u)
                if axb and not atc:
                    acc.setdefault(("axb-anode-only", side), []).append(u)
            if tb:
                fp = tuple(sorted(FACE[e[0]] for e in ends))
                pairs[fp] = pairs.get(fp, 0) + 1
                if fp == ("anode", "cathode"):
                    ua = [e[2] for e in ends if e[0] == 0][0]
                    uc = [e[2] for e in ends if e[0] == 1][0]
                    ac_rows.append((ev, uid, side, b["flash_gid"], ua, uc,
                                    (uc - ua) - g["u_cathode"]))

    print("tag=%r  auto=%d  two_boundary=%d  at_x_boundary=%d (anode-only %d)"
          % (args.tag, counts["auto"], counts["tb"], counts["axb"],
             counts["axb_anode"]))
    print("\ntwo_boundary face-pair composition:")
    for fp, n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print("  %-22s %3d" % ("+".join(fp), n))

    print("\n== anode-facing PCA-end u (cm; 0 = FV anode edge, + = into "
          "volume), at the bundle's own selected-flash T0 ==")
    for grp, note in (("two_boundary", "CIRCULAR: 3 cm flag margin"),
                      ("axb-anode-only", "CIRCULAR: [-2,+4] advantage window"),
                      ("all-auto", "no flag gate")):
        print("  group %-15s (%s)" % (grp, note))
        for side in ("bot", "top"):
            print("    %s: %s" % (side, stats(acc.get((grp, side), []))))

    print("\n== SPAN CLOSURE (T0-independent): two_boundary anode+cathode "
          "bundles, (u_cath_end - u_anode_end) - u_cathode ==")
    print("   expect ~ -2..-9 cm (FR-domain anode gap + cathode shortfall); "
          "> +2 cm is physically impossible")
    n_phys = 0
    for ev, uid, side, gid, ua, uc, cl in sorted(ac_rows,
                                                 key=lambda r: abs(r[6])):
        ok = -12 <= cl <= 2
        n_phys += ok
        print("  ev %s uid %-8d %s gid %3d  u_anode %+7.2f  u_cath %7.2f  "
              "closure %+7.2f  %s" % (ev, uid, side, gid, ua, uc, cl,
                                      "PHYSICAL" if ok else "IMPOSSIBLE-SPAN"))
    print("  -> %d/%d physically consistent full-drift spans" %
          (n_phys, len(ac_rows)))


if __name__ == "__main__":
    main()

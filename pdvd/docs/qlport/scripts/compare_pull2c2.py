#!/usr/bin/env python3
"""18-event before/after census for the anode-pull + cathode-cushion study.

Read-only.  Compares each event's canonical (shield-FV, knobs OFF) Q-L calib
dump against the pull2c2 variant dump (both crates pulled +13.507 us toward the
anode = 2.0 cm at v=0.148073, AND cathode containment tolerance widened
cathode_ext1 1.2 -> 2.0 cm).  For every cluster it reads QLMatching's own final
selection (each bundle's auto_selected flag, the same field the ql_scan viewer
renders) and reports, per event and in aggregate:

  n_sel        number of auto_selected bundles (matched cluster<->flash pairs)
  n_moved      clusters whose auto_selected FLASH gid changed off->on
  n_bright     of those, how many moved to a BRIGHTER flash (higher total_PE)
  n_contained  candidate (flash,cluster) bundles marked contained (grows when
               the cushion + pull let more halves clear the cathode gate)

This is a case-check of the display, NOT a validation of the knobs: it shows
whether the target crossers converge onto their bright coincident flash, but it
does NOT price the global cost (a wider cushion can admit wrong bright flashes
elsewhere; a pull larger than the true offset can push anode-touchers past the
-3.0 cm anode floor).  Those need the census + full-manifest A/B.

Repro:
    cd wcp-porting-img/pdvd
    python3 docs/qlport/scripts/compare_pull2c2.py
"""
import sys
import glob
import json
import os

OFF_GLOB = "work/039252_%d/calib-evt*.json"
ON_GLOB = "work/039252_%d_pull2c2/calib-evt*.json"
NIDX = 18

# The two hand-scan target pairs in evt298567 (idx 0).
TARGETS = {50: "bot:50", 4000060: "top:60", 8: "bot:8", 4000063: "top:63"}


def load(path):
    d = json.load(open(path))
    fl = {f["gid"]: f for f in d["flashes"]}
    sel = {}          # cluster uid -> (flash_gid, total_PE)
    ncont = 0
    for b in d["bundles"]:
        if b.get("contained"):
            ncont += 1
        if not b.get("auto_selected"):
            continue
        for c in [b["main_cluster"]] + b.get("other_clusters", []):
            sel[c] = (b["flash_gid"], fl[b["flash_gid"]]["total_PE"])
    return d, fl, sel, ncont


def main():
    print("%-4s %-12s  %6s %6s   %6s %6s   %7s %7s" %
          ("idx", "event", "sel_off", "sel_on", "moved", "->bright", "cont_off", "cont_on"))
    tot = dict(moved=0, bright=0)
    for i in range(NIDX):
        offp = sorted(glob.glob(OFF_GLOB % i))
        onp = sorted(glob.glob(ON_GLOB % i))
        if not offp or not onp:
            print("%-4d  (missing dump)" % i)
            continue
        offp, onp = offp[0], onp[0]
        evt = os.path.basename(offp).replace("calib-evt", "").replace(".json", "")
        _, _, soff, coff = load(offp)
        _, _, son, con = load(onp)
        moved = bright = 0
        for c in set(soff) | set(son):
            a = soff.get(c)
            b = son.get(c)
            if a and b and a[0] != b[0]:
                moved += 1
                if b[1] > a[1]:
                    bright += 1
        tot["moved"] += moved
        tot["bright"] += bright
        print("%-4d %-12s  %6d %6d   %6d %6d   %7d %7d" %
              (i, evt, len(soff), len(son), moved, bright, coff, con))
    print("-" * 68)
    print("total clusters that changed matched flash: %d (of which -> brighter: %d)"
          % (tot["moved"], tot["bright"]))

    # The two target pairs, spelled out (idx 0 = evt298567).
    print("\nevt298567 target crossers (auto_selected flash: gid  t/us  PE):")
    _, floff, soff, _ = load(sorted(glob.glob(OFF_GLOB % 0))[0])
    _, flon, son, _ = load(sorted(glob.glob(ON_GLOB % 0))[0])
    for uid, name in TARGETS.items():
        go, _ = soff.get(uid, (None, None))
        gn, _ = son.get(uid, (None, None))
        so = "gid%-3d t=%8.1f PE=%6.0f" % (go, floff[go]["time"], floff[go]["total_PE"]) if go is not None else "(none)"
        sn = "gid%-3d t=%8.1f PE=%6.0f" % (gn, flon[gn]["time"], flon[gn]["total_PE"]) if gn is not None else "(none)"
        print("  %-8s  OFF %-32s  ON %-32s" % (name, so, sn))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc qlmatch/35 check C2 -- doc 12's crosser closure and doc 34's rail read-out on all three runs (120 events).

Doc 34 step A ran d34_rail_calib.py on run 039252 only.  This is the same harvest (fit_qtol_crossers.py, strict cathode
crossers, external geometric truth) and the same arm_report(), over every index dir of the three runs with the given
work-dir tags (fit_qtol_crossers.dump_paths), plus the two pre-registered read-outs of d35/prereg.md sec 2 C2:
  rail-excluded global median Sum(meas)/Sum(pred): candidate / control, STOP if |x - 1| > 0.05;
  rail-inclusive / rail-excluded on the candidate: STOP if > 1.10.

    python3 d35_crossers.py --cand q34tk1 --ctl q31ctl > ../d35/c2_crossers.txt
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "ql_light_calib"))
from fit_qtol_crossers import harvest, dump_paths   # noqa: E402
from d34_rail_calib import arm_report               # noqa: E402


def anchors_of(tag):
    paths = dump_paths(tag)
    anchors = harvest(paths)
    cache = {}
    for an in anchors:                  # railed = the dump's sat flag on this flash, statically kept channel (d34)
        if an["path"] not in cache:
            cache[an["path"]] = json.load(open(an["path"]))
        d = cache[an["path"]]
        f = next(f for f in d["flashes"] if f["gid"] == an["gid"])
        keep_static = np.array([od["active"] and not od["auto_masked"] for od in d["opdets"]], bool)
        an["rail"] = np.asarray(f.get("sat", [0] * 40), bool) & keep_static
    return paths, anchors


def flash_time(an, cache={}):
    if an["path"] not in cache:
        cache[an["path"]] = {f["gid"]: f["time"] for f in json.load(open(an["path"]))["flashes"]}
    return cache[an["path"]][an["gid"]]


def common(ca, ka, tol=1.5):
    """anchors present in both arms: same event, flash times within tol us (ToT moves bright flashes ~1 us)."""
    out = []
    for x in ca:
        tx = flash_time(x)
        for y in ka:
            if y["evt"] == x["evt"] and abs(flash_time(y) - tx) <= tol:
                out.append((x, y))
                break
    return out


def rail_excluded(anchors):
    v = [a["meas"][a["keep"]].sum() / a["pred"][a["keep"]].sum() for a in anchors
         if a["pred"][a["keep"]].sum() > 0 and a["meas"][a["keep"]].sum() > 0]
    return float(np.median(v)), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ctl", required=True)
    a = ap.parse_args()
    res, anc = {}, {}
    for tag in (a.cand, a.ctl):
        paths, anchors = anchors_of(tag)
        anc[tag] = anchors
        runs = sorted({os.path.basename(os.path.dirname(p)).split("_")[0] for p in paths})
        print(f"# arm {tag}: {len(paths)} dumps, runs {','.join(runs)}")
        _, gratio = arm_report(tag, anchors)
        res[tag] = (rail_excluded(anchors), gratio)
    (ce, cn), cg = res[a.cand]
    (ke, kn), _ = res[a.ctl]
    x = ce / ke
    print("\n# PRE-REGISTERED READ-OUT (d35/prereg.md sec 2 C2)")
    print(f"   rail-excluded global median: {a.cand} {ce:.3f} (n {cn})  {a.ctl} {ke:.3f} (n {kn})  ratio {x:.3f} "
          f"-> {'PASS' if abs(x - 1) <= 0.05 else 'STOP'} (|ratio - 1| <= 0.05)")
    print(f"   rail-inclusive / rail-excluded on {a.cand}: {cg:.3f} -> {'PASS' if cg <= 1.10 else 'STOP'} (<= 1.10)")
    pairs = common(anc[a.cand], anc[a.ctl])
    (pc, _), (pk, _) = rail_excluded([x for x, _ in pairs]), rail_excluded([y for _, y in pairs])
    print(f"\n# added after the read-out (informational): the same comparison on the {len(pairs)} anchors present in both "
          f"arms (same event, flash time within 1.5 us)")
    print(f"   rail-excluded global median: {a.cand} {pc:.3f}  {a.ctl} {pk:.3f}  ratio {pc / pk:.3f}")
    only = len(anc[a.cand]) - len(pairs)
    print(f"   {a.cand}-only anchors {only}: median flash total_PE {np.median([x['total'] for x in anc[a.cand] if not any(x is p for p, _ in pairs)]):.0f} "
          f"(harvest cut 1500 PE); common anchors' total_PE ratio {a.cand}/{a.ctl} median "
          f"{np.median([x['total'] / y['total'] for x, y in pairs]):.3f}")


if __name__ == "__main__":
    main()

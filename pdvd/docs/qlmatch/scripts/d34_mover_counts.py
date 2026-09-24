#!/usr/bin/env python3
"""doc qlmatch/34 -- pairwise mover counts over the 18 run-039252 events (no truth involved; sizing/diagnostic).

A mover between arms A and B is a long cluster (the scorer's filter, in both dumps) whose auto-selected flash set
differs.  When A is on control light and B on ToT light, A's picks are mapped forward with the doc 32 time map (the
doc 32/33 definition, which reproduces doc 33's 100 for q31ctl vs q32ti); same-light pairs compare directly.

    python3 d34_mover_counts.py --work-root /home/xqian/tmp/p31/wroot --time-map ../d32/time_map_ctl_to_q32ti.json \
        --pairs q31ctl:C:q32ti:T,q31ctl:C:q33tu:T,...
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d32_scan_items import calib_path, load, long_uids, auto_times, NEVT   # noqa: E402
from d33_scan_items import mapper, tset, same                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--time-map", required=True)
    ap.add_argument("--pairs", required=True)
    a = ap.parse_args()
    tmap = load(a.time_map)["events"]
    from d32_scan_items import EVT0, STEP
    for pr in a.pairs.split(","):
        ta, la, tb, lb = pr.split(":")
        assert not (la == "T" and lb == "C"), "put the control-light arm first"
        n = 0
        for idx in range(NEVT):
            evt = EVT0 + STEP * idx
            A, B = load(calib_path(a.work_root, idx, ta)), load(calib_path(a.work_root, idx, tb))
            fwd, _ = mapper(tmap.get(str(evt), []))
            f = fwd if (la, lb) == ("C", "T") else (lambda t: t)
            aa, bb = auto_times(A), auto_times(B)
            for u in long_uids(A) & long_uids(B):
                if not same(tset(f(t) for t in aa.get(u, [])), tset(bb.get(u, []))):
                    n += 1
        print(pr, n)


if __name__ == "__main__":
    main()

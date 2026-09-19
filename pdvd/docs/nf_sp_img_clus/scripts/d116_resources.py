#!/usr/bin/env python3
"""doc pdvd/116 -- the R clause: median per-event PR wall and peak-RSS ratio of each arm over a reference arm, from the
runner's pr_resource_*.txt files (wall_s=, peak_rss_gb=).  Read-only.  Fork of the resources block of
d113_nearfar_movers.py (untouched).

Usage: d116_resources.py --det pdhd --ref d115hp3bwp05 --arms d116hr1,d116hr2,d116hr3
"""
import argparse, glob, os, re
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"


def res(det, arm):
    out = {}
    for p in glob.glob(f"{IMG}/{det}/work/*_{arm}/pr_resource_*.txt"):
        t = open(p).read()
        ev = os.path.basename(os.path.dirname(p))[:-len(arm) - 1]
        w = re.search(r"wall_s=(\S+)", t); r = re.search(r"peak_rss_gb=(\S+)", t)
        if w and r:
            out[ev] = (float(w.group(1)), float(r.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--arms", required=True)
    a = ap.parse_args()
    b = res(a.det, a.ref)
    print(f"# {a.det}: reference {a.ref} ({len(b)} events with resources)")
    for arm in a.arms.split(","):
        x = res(a.det, arm)
        ev = sorted(set(x) & set(b))
        if not ev:
            print(f"{arm}: no common events"); continue
        w = np.array([x[e][0] / b[e][0] for e in ev]); r = np.array([x[e][1] / b[e][1] for e in ev])
        print(f"{arm:12s} events {len(ev)}: wall median ratio {np.median(w):.3f} (p90 {np.percentile(w, 90):.3f}; total {sum(x[e][0] for e in ev):.0f} vs {sum(b[e][0] for e in ev):.0f} s); "
              f"peak RSS median ratio {np.median(r):.3f} (max {max(x[e][1] for e in ev):.2f} vs {max(b[e][1] for e in ev):.2f} GB)")


if __name__ == "__main__":
    main()

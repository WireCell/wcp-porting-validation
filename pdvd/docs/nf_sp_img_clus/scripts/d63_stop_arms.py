#!/usr/bin/env python3
"""doc pdvd/63 -- the C++ stop-arm numbers (kind/len/mip/kink) for named items, from an arm's logs."""
import sys, re, glob, os
arm = sys.argv[1]; items = sys.argv[2:]
W = "/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
pat = re.compile(r"stop-arm: cluster (\d+) seg (\d+) kind (\d) len ([\d.]+) cm far_len ([\d.]+) cm mip ([\d.]+) kink ([-\d.]+) deg shower (\d) terminal (\d)")
KIND = {0: "other", 1: "michel", 2: "continuation", 3: "delta", 4: "hadron"}
for it in items:
    ev, cid = it.split("/")
    for lg in glob.glob(os.path.join(W, ev + "_" + arm, "wct_pr_*.log")):
        for line in open(lg, errors="replace"):
            m = pat.search(line)
            if m and m.group(1) == cid:
                g = m.groups()
                print("%-14s seg %-7s %-12s len %5.2f far %5.2f mip %.2f kink %6.1f shower %s terminal %s" % (it, g[1], KIND[int(g[2])], float(g[3]), float(g[4]), float(g[5]), float(g[6]), g[7], g[8]))

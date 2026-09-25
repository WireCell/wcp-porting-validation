#!/usr/bin/env python3
"""doc sbnd_xin/124 -- the TPC0-TPC1 beam-window flash pairing census of a PR arm: per event, the in-window
flashes (T_flash in_window, [0.2, 2.2) us) of each TPC, the smallest |t0 - t1| between them and whether the PR
paired them into one flash group; plus how many in-window flashes sit within 100 ns of a window edge
(0.2/2.2 us PR gate, 0.3/1.9 us Q/L window).

usage: beam_dt_census.py <pr_root> [...]     (prints a histogram per root)
"""
import glob, os, sys
from multiprocessing import Pool
import numpy as np, uproot

def one(d):
    try:
        t = uproot.open(os.path.join(d, "tracking-pr.root"))["T_flash"].arrays(["tpc", "time_us", "in_window", "flash_group", "pe"], library="np")
    except Exception:
        return None
    w = t["in_window"] == 1
    t0 = [(x, g, p) for x, g, p, c in zip(t["time_us"][w], t["flash_group"][w], t["pe"][w], t["tpc"][w]) if c == 0]
    t1 = [(x, g, p) for x, g, p, c in zip(t["time_us"][w], t["flash_group"][w], t["pe"][w], t["tpc"][w]) if c == 1]
    best = None
    for a in t0:
        for b in t1:
            dt = abs(a[0] - b[0]) * 1000
            if best is None or dt < best[0]:
                best = (dt, a[1] == b[1], min(a[2], b[2]))
    edge = sum(1 for x in t["time_us"][w] if min(abs(x - 0.2), abs(x - 2.2), abs(x - 0.3), abs(x - 1.9)) < 0.1)
    return (best, edge, int(w.sum()))

for root in sys.argv[1:]:
    ds = sorted(glob.glob(os.path.join(root, "pr_evt*")) + glob.glob(os.path.join(root, "f*", "pr_evt*")))
    with Pool(16) as p:
        R = [r for r in p.map(one, ds) if r is not None]
    pairs = [r[0] for r in R if r[0] is not None]
    dts = np.array([x[0] for x in pairs]); grp = np.array([x[1] for x in pairs])
    edges = [0, 10, 20, 30, 40, 50, 60, 80, 100, 200, 500, 2000]
    h = np.histogram(dts, bins=edges)[0]
    print("%s: events %d, with in-window flashes in both TPCs %d; paired (same group) %d" % (root, len(R), len(pairs), int(grp.sum())))
    print("   |t0-t1| ns bins " + " ".join("%d-%d:%d" % (edges[i], edges[i + 1], h[i]) for i in range(len(h))))
    print("   pairs with 40 < dt < 60 ns (the 50 ns pairing edge): %d; in-window flashes within 100 ns of a window edge: %d" % (
        int(((dts > 40) & (dts < 60)).sum()), sum(r[1] for r in R)))

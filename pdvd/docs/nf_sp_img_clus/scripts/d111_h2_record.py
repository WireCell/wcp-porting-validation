#!/usr/bin/env python3
"""doc pdvd/111 sec 7 -- is the h2 comparison like for like?  For PDHD 029107_16 cluster 106 in each arm: the persisted
record's span inside the +-24 cm window in the spot's image frame, how many rows sit on the side-branch side
(e2 < -3 cm), and, for traced arms, where the round-2 seed left the round-1 Steiner path (the crawl re-route).  Read-only.

Usage: d111_h2_record.py --arms d111hoff,d111hsr6,d111hsr10,d111hsr15 --traced d111hA1,d111hS,d111hsr10tr
"""
import argparse, os, sys
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d111_stage_attrib as A      # noqa: E402
from d111_spot_frame import frame   # noqa: E402

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
EV, CL, CLICK, R = "029107_16", 106, np.array((182.0, 537.4, 401.8)), 24.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--traced", default="")
    a = ap.parse_args()
    arms = a.arms.split(",")
    img = A.bee_layer(f"{IMG}/pdhd/work/{EV}_{arms[0]}/mabc-pr.zip", "clustering-global")
    mi = img[1] == CL; P = img[0][mi]; w = np.clip(img[2][mi], 1, None)
    near = np.linalg.norm(P - CLICK, axis=1) < R
    c, e1, e2, e3 = frame(P[near], w[near])
    print(f"# doc pdvd/111 h2 record check: {EV} cl{CL}, window {R} cm, frame of {arms[0]}'s image")
    for arm in arms + [x for x in a.traced.split(",") if x]:
        t = uproot.open(f"{IMG}/pdhd/work/{EV}_{arm}/tracking-stm.root")["T_rec_charge"].arrays(
            ["cluster_id", "pass", "x", "y", "z"], library="np")
        m = t["cluster_id"] == CL
        for p in np.unique(t["pass"][m]):
            mm = m & (t["pass"] == p)
            F = np.c_[t["x"][mm], t["y"][mm], t["z"][mm]]
            k = np.linalg.norm(F - CLICK, axis=1) < R
            if not k.any():
                continue
            V = F[k] - c; s = V @ e1; t2 = V @ e2
            L = np.sum(np.linalg.norm(np.diff(F[k], axis=0), axis=1))
            print(f"{arm}: pass {p}, rows {mm.sum()}, window rows {k.sum()}, window length {L:.1f} cm, "
                  f"s {s.min():.1f}..{s.max():.1f}, rows with e2 < -3 cm {(t2 < -3).sum()}")
    for arm in [x for x in a.traced.split(",") if x]:
        blocks, _ = A.parse_trace(f"/home/xqian/tmp/d111/arm_{arm}/evt_{EV}.log.gz")
        r1 = [b for b in blocks if b["cl"] == CL and b["rnd"] == "r1"]
        r2 = [b for b in blocks if b["cl"] == CL and b["rnd"] == "r2" and "final" in b["stages"]]
        if not r1 or not r2:
            continue
        s1, s2 = r1[0]["stages"]["seed"], r2[0]["stages"]["seed"]
        Q, _ = A.closest_on_polyline(s1, s2); dd = np.linalg.norm(Q - s2, axis=1); j = int(np.argmax(dd))
        print(f"{arm}: round-2 seed leaves the round-1 Steiner path by up to {dd.max():.2f} cm, "
              f"{np.linalg.norm(s2[j] - CLICK):.1f} cm from the h2 click")


if __name__ == "__main__":
    main()

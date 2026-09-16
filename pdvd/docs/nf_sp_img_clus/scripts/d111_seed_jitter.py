#!/usr/bin/env python3
"""doc pdvd/111 sec 7 -- what seed_recenter_sigma does to the seed itself, from a TRACED knob-on arm: organize_orig_path
points (stage org1) against the re-centred points (stage org1r) of every round-2 block.  Read-only.

  * move: |org1r - org1| per point;
  * 3-point wiggle: distance of point i from the chord i-1 -> i+1 (cm), before and after;
  * ridge offset (d111_stage_attrib.Ridge, image only) before and after.

Usage: d111_seed_jitter.py --det pdhd --arm d111hsr10tr --event 029107_16
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d111_stage_attrib as A   # noqa: E402

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"


def wig(P):
    out = []
    for i in range(1, len(P) - 1):
        u = P[i + 1] - P[i - 1]; n = np.linalg.norm(u)
        if n == 0:
            continue
        u = u / n; v = P[i] - P[i - 1]; out.append(np.linalg.norm(v - (v @ u) * u))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--event", required=True)
    a = ap.parse_args()
    blocks, _ = A.parse_trace(f"/home/xqian/tmp/d111/arm_{a.arm}/evt_{a.event}.log.gz")
    img = A.bee_layer(f"{IMG}/{a.det}/work/{a.event}_{a.arm}/mabc-pr.zip", "clustering-global")
    W0, W1, M, D0, D1 = [], [], [], [], []
    ridges = {}
    for b in blocks:
        if b["rnd"] != "r2" or "org1r" not in b["stages"]:
            continue
        o, r = b["stages"]["org1"], b["stages"]["org1r"]
        if len(o) != len(r) or len(o) < 5:
            continue
        W0.append(wig(o)); W1.append(wig(r)); M.append(np.linalg.norm(r - o, axis=1))
        if b["cl"] not in ridges:
            mi = img[1] == b["cl"]; ridges[b["cl"]] = A.Ridge(img[0][mi], img[2][mi])
        if ridges[b["cl"]].ok:
            D0.append(ridges[b["cl"]].offset(o)); D1.append(ridges[b["cl"]].offset(r))
    W0, W1, M, D0, D1 = map(np.concatenate, (W0, W1, M, D0, D1))
    print(f"# doc pdvd/111 seed jitter: {a.det} {a.arm} {a.event}")
    print(f"org1 -> org1r points {len(M)}: move p50 {np.median(M):.2f} p90 {np.percentile(M,90):.2f} max {M.max():.2f} cm; "
          f"moved {100*np.mean(M > 0):.1f} %")
    print(f"3-point wiggle: org1 p50 {np.median(W0):.2f} p90 {np.percentile(W0,90):.2f}, > 0.5 cm {100*np.mean(W0 > 0.5):.1f} % | "
          f"org1r p50 {np.median(W1):.2f} p90 {np.percentile(W1,90):.2f}, > 0.5 cm {100*np.mean(W1 > 0.5):.1f} %")
    print(f"ridge offset: org1 > 1 cm {100*np.mean(D0 > 1):.1f} % (p90 {np.percentile(D0,90):.2f}) | "
          f"org1r > 1 cm {100*np.mean(D1 > 1):.1f} % (p90 {np.percentile(D1,90):.2f})")


if __name__ == "__main__":
    main()

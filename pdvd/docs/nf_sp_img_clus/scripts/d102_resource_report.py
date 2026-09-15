#!/usr/bin/env python3
"""doc pdvd/102 sec 7.3 -- the pre-registered resource clause, per event, arm vs its baseline.

Per event (both dirs complete): wall_s and peak_rss_gb from pr_resource_<run>_<evt>.txt, and the MABC
per-visitor timings ("MABC timing: <visitor> took <ms>") from wct_pr_*.log.  The baseline arms ran on a
different night at a different machine load, so wall is load-contaminated.  'loaded live' (reading the
pctree, which no sampler setting can touch) is reported as a load control, and the Steiner stage
(CreateSteinerGraph:pr, where the retile sampler runs) is shown raw and divided by that control.

Clause (pred.txt): median per-event peak RSS ratio <= 1.25 and median wall ratio <= 1.50.

Usage: d102_resource_report.py det:BASE:ARM [det:BASE:ARM ...]
   eg  d102_resource_report.py pdhd:d101hkf:d102hcs pdvd:d101vkf:d102vcsall
"""
import glob, os, re, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
PAT = re.compile(r"MABC timing: (.+?) took ([0-9.]+) ms")


def visitors(log):
    d = {}
    for line in open(log, errors="replace"):
        m = PAT.search(line)
        if m:
            d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(2)) / 1000
    return d


def res(d):
    f = glob.glob(f"{d}/pr_resource_*.txt")
    if not f:
        return None
    s = open(f[0]).read()
    return int(re.search(r"wall_s=(\d+)", s).group(1)), float(re.search(r"peak_rss_gb=([0-9.]+)", s).group(1))


def main():
    for spec in sys.argv[1:]:
        det, base, arm = spec.split(":")
        W, R, S, L, T = [], [], [], [], {}
        for d in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}")):
            ev = os.path.basename(d)[:-(len(arm) + 1)]
            b = f"{IMG}/{det}/work/{ev}_{base}"
            ra, rb = res(d), res(b)
            if ra is None or rb is None:
                continue
            W.append(ra[0] / rb[0]); R.append(ra[1] / rb[1])
            A = visitors(glob.glob(f"{d}/wct_pr_*.log")[0]); B = visitors(glob.glob(f"{b}/wct_pr_*.log")[0])
            for k in B:
                if k in A:
                    T.setdefault(k, [0.0, 0.0]); T[k][0] += B[k]; T[k][1] += A[k]
            if B.get("CreateSteinerGraph:pr", 0) > 0.05 and B.get("loaded live", 0) > 0.05:
                S.append(A["CreateSteinerGraph:pr"] / B["CreateSteinerGraph:pr"]); L.append(A["loaded live"] / B["loaded live"])
        if not W:
            print(f"[{det}] {arm} vs {base}: no complete pairs"); continue
        wall_ok, rss_ok = np.median(W) <= 1.50, np.median(R) <= 1.25
        print(f"[{det}] {arm} vs {base}: {len(W)} events | wall ratio median {np.median(W):.2f} (p90 {np.percentile(W, 90):.2f}) -> {'PASS' if wall_ok else 'FAIL'} | "
              f"peak RSS ratio median {np.median(R):.3f} (max {max(R):.3f}) -> {'PASS' if rss_ok else 'FAIL'}")
        print(f"    Steiner stage ratio median {np.median(S):.2f}; load control 'loaded live' median {np.median(L):.2f}; "
              f"Steiner / control median {np.median(np.array(S) / np.array(L)):.2f}")
        for k, (b, a) in sorted(T.items(), key=lambda kv: -(kv[1][1] - kv[1][0]))[:6]:
            print(f"    {k[:45]:45s} total {b:8.1f} s -> {a:8.1f} s")


if __name__ == "__main__":
    main()

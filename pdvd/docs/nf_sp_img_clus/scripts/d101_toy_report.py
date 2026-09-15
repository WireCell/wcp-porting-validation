#!/usr/bin/env python3
"""doc pdvd/101 -- paired lever ranking from a d101_toy_scan.py run.

Every variant runs on the SAME (detector, theta, phi, seed) tracks, so the per-track difference
variant - base cancels the track-to-track spread.  For each (detector, variant, metric) this prints
the median paired difference and a 90 % bootstrap interval of that median (2000 resamples, fixed
seed), plus how many tracks improved.  Lower is better for every metric listed except dqdx_med
(|1 - x| is used).

Usage:
  d101_toy_report.py SCAN_DIR [--metrics res_t_p90,dev_p90,snap_w,dqdx_cv,dip,acf1] [--out FILE]
"""
import argparse, json, os, sys
import numpy as np

DEFAULT = "res_t_med,res_t_p90,dev_p90,snap_u,snap_v,snap_w,seed_p90,dqdx_cv,dip,dip_truth,acf1,r_res_q,len_ratio"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--metrics", default=DEFAULT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.scan, "rows.jsonl"))]
    rows = [r for r in rows if "error" not in r]
    key = lambda r: (r["det"], r["theta"], r["phi"], r["seed"])
    base = {key(r): r for r in rows if r["variant"] == "base"}
    mets = a.metrics.split(",")
    dets = sorted(set(r["det"] for r in rows))
    variants = [v for v in dict.fromkeys(r["variant"] for r in rows) if v != "base"]
    rng = np.random.default_rng(101)
    lines = ["# paired difference (variant - base): median [90% bootstrap CI of the median]  n_improved/n"]
    lines.append("det\tvariant\t" + "\t".join(mets))
    for d in dets:
        # base absolute values first
        bl = [r for r in base.values() if r["det"] == d]
        lines.append(f"{d}\tBASE(abs)\t" + "\t".join(f"{np.nanmedian([r.get(m, np.nan) for r in bl]):.4g}" for m in mets))
        for v in variants:
            cells = []
            vr = [r for r in rows if r["det"] == d and r["variant"] == v]
            for m in mets:
                diffs = []
                for r in vr:
                    b = base.get(key(r))
                    if b is None or m not in r or m not in b:
                        continue
                    x, y = r[m], b[m]
                    if m == "dqdx_med":
                        x, y = abs(1 - x), abs(1 - y)
                    if m == "r_res_q":
                        x, y = abs(x), abs(y)
                    if np.isfinite(x) and np.isfinite(y):
                        diffs.append(x - y)
                diffs = np.array(diffs)
                if len(diffs) < 5:
                    cells.append("n/a"); continue
                boots = np.median(rng.choice(diffs, size=(2000, len(diffs)), replace=True), axis=1)
                lo, hi = np.percentile(boots, [5, 95])
                cells.append(f"{np.median(diffs):+.3g} [{lo:+.2g},{hi:+.2g}] {int(np.sum(diffs < 0))}/{len(diffs)}")
            lines.append(f"{d}\t{v}\t" + "\t".join(cells))
    txt = "\n".join(lines)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

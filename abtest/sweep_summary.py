#!/usr/bin/env python3
"""Aggregate a sweep_events.sh results.tsv into the summary tables used by
clus/docs/imgclus-resource-profile.md (min/median/mean/max per detector and
per run), plus failures and top-N tails.

Usage: sweep_summary.py <results.tsv> [topN]
"""
import sys
import statistics as st
from collections import defaultdict

def fmt(vals):
    vals = sorted(vals)
    return "%d / %d / %d / %d" % (
        vals[0], st.median(vals), st.mean(vals), vals[-1])

def main(path, topn=10):
    rows = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 9:
                continue
            det, run, evt = p[0], p[1], p[2]
            img_s, img_kb, img_rc = int(p[3]), int(p[4]), int(p[5])
            clus_s, clus_kb, clus_rc = int(p[6]), int(p[7]), int(p[8])
            rows.append(dict(det=det, run=run, evt=evt,
                             img_s=img_s, img_mb=img_kb//1024, img_rc=img_rc,
                             clus_s=clus_s, clus_mb=clus_kb//1024, clus_rc=clus_rc))

    ok = [r for r in rows if r["img_rc"] == 0 and r["clus_rc"] == 0]
    bad = [r for r in rows if r["img_rc"] != 0 or r["clus_rc"] != 0]

    print("## Summary by detector\n")
    print("| Detector | N | Imaging time (s) | Imaging RSS (MB) | Clustering time (s) | Clustering RSS (MB) | img+clus time (s) |")
    print("|---|---|---|---|---|---|---|")
    for det in ("pdhd", "pdvd"):
        sel = [r for r in ok if r["det"] == det]
        if not sel:
            continue
        print("| %s | %d | %s | %s | %s | %s | %s |" % (
            det.upper(), len(sel),
            fmt([r["img_s"] for r in sel]),
            fmt([r["img_mb"] for r in sel]),
            fmt([r["clus_s"] for r in sel]),
            fmt([r["clus_mb"] for r in sel]),
            fmt([r["img_s"] + r["clus_s"] for r in sel])))

    print("\n## Summary by run\n")
    print("| Detector | Run | N | Imaging time (s) | Imaging RSS (MB) | Clustering time (s) | Clustering RSS (MB) |")
    print("|---|---|---|---|---|---|---|")
    byrun = defaultdict(list)
    for r in ok:
        byrun[(r["det"], r["run"])].append(r)
    for (det, run), sel in sorted(byrun.items()):
        print("| %s | %s | %d | %s | %s | %s | %s |" % (
            det.upper(), run, len(sel),
            fmt([r["img_s"] for r in sel]),
            fmt([r["img_mb"] for r in sel]),
            fmt([r["clus_s"] for r in sel]),
            fmt([r["clus_mb"] for r in sel])))

    if bad:
        print("\n## Failures\n")
        print("| Detector | Run | Evt | img rc | clus rc |")
        print("|---|---|---|---|---|")
        for r in bad:
            print("| %s | %s | %s | %d | %d |" % (
                r["det"], r["run"], r["evt"], r["img_rc"], r["clus_rc"]))

    for key, label in (("img_s", "imaging wall"), ("img_mb", "imaging RSS"),
                       ("clus_s", "clustering wall"), ("clus_mb", "clustering RSS")):
        print("\n## Top %d by %s\n" % (topn, label))
        for r in sorted(ok, key=lambda r: r[key], reverse=True)[:topn]:
            print("  %s %s/%s  img %ds/%dMB  clus %ds/%dMB" % (
                r["det"], r["run"], r["evt"],
                r["img_s"], r["img_mb"], r["clus_s"], r["clus_mb"]))

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)

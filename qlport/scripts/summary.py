#!/usr/bin/env python3
"""Population stats for a sweep label; optional before/after vs a base label.

Usage: summary.py <label> [<base-label>]
Reads scripts/sweep/<label>/results.tsv (idx ev subrun wall_s maxrss_kb rc
node_exec_s).  With a base label, prints a markdown before/after table and
the per-event wall deltas beyond the largest |delta| seen between two
identical-binary baselines (noise floor is printed, not enforced).
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(label):
    path = os.path.join(HERE, "sweep", label, "results.tsv")
    rows = {}
    with open(path) as f:
        header = f.readline().split()
        for line in f:
            d = dict(zip(header, line.split()))
            rows[int(d["idx"])] = {
                "ev": int(d["ev"]),
                "wall": float(d["wall_s"]),
                "rss_mb": float(d["maxrss_kb"]) / 1024.0,
                "rc": int(d["rc"]),
                "node": float(d["node_exec_s"]) if d["node_exec_s"] != "nan" else float("nan"),
            }
    return rows


def stats(vals):
    vs = sorted(vals)
    p90 = vs[min(len(vs) - 1, int(round(0.9 * (len(vs) - 1))))]
    return (min(vs), statistics.median(vs), statistics.mean(vs), p90, max(vs))


def fmt(label, rows):
    print(f"### {label}  (n={len(rows)}, failures={sum(1 for r in rows.values() if r['rc'])})")
    print("| metric | min | median | mean | p90 | max |")
    print("|---|---|---|---|---|---|")
    for key, name, f in (("wall", "wall_s", "%.0f"), ("node", "node_exec_s", "%.2f"),
                         ("rss_mb", "peak RSS MB", "%.0f")):
        s = stats([r[key] for r in rows.values()])
        print(f"| {name} | " + " | ".join(f % v for v in s) + " |")


def main():
    label = sys.argv[1]
    rows = load(label)
    fmt(label, rows)
    ev_by = sorted(rows, key=lambda i: rows[i]["node"])
    med = ev_by[len(ev_by) // 2]
    worst_w = max(rows, key=lambda i: rows[i]["wall"])
    worst_r = max(rows, key=lambda i: rows[i]["rss_mb"])
    print(f"\nrepresentative: median-node idx={med} ev={rows[med]['ev']}"
          f" | worst-wall idx={worst_w} ev={rows[worst_w]['ev']}"
          f" | worst-rss idx={worst_r} ev={rows[worst_r]['ev']}")

    if len(sys.argv) > 2:
        base = load(sys.argv[2])
        print(f"\n### {sys.argv[2]} -> {label}")
        print("| metric | base median | new median | base mean | new mean | base max | new max |")
        print("|---|---|---|---|---|---|---|")
        for key, name, f in (("wall", "wall_s", "%.0f"), ("node", "node_exec_s", "%.2f"),
                             ("rss_mb", "peak RSS MB", "%.0f")):
            b = stats([r[key] for r in base.values()])
            n = stats([r[key] for r in rows.values()])
            print(f"| {name} | {f % b[1]} | {f % n[1]} | {f % b[2]} | {f % n[2]} |"
                  f" {f % b[4]} | {f % n[4]} |")
        deltas = sorted(((rows[i]["wall"] - base[i]["wall"], i) for i in rows if i in base),
                        key=lambda t: abs(t[0]), reverse=True)
        print("\nlargest per-event wall deltas (new-base, s): "
              + ", ".join(f"idx{i}:{d:+.0f}" for d, i in deltas[:5]))


if __name__ == "__main__":
    main()

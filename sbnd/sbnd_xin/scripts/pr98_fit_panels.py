#!/usr/bin/env python3
"""doc pr/98 -- static before/after near-vertex fit panels.

For each event, reads calib-pr-evt<ID>.json (PrDisplayDump) from two arms and
renders a 2x3 PNG: rows = arm A (off) / arm B (fit_exclusion on), cols =
X-Y / X-Z / Y-Z, zoomed to a box around arm A's main vertex.  Charge =
track_shower points (gray, all clusters; main-vertex cluster darker); fits =
segments[].points polylines colored by segment; vertex = star (both arms').

The owner's pr/98 criterion is *is the fit visibly better near the vertex in
the multi-track region*, so this deliberately mirrors pr_display's row-1
panels rather than the wire-plane views (those stay available in the live
viewer via --wire-planes).

Usage: pr98_fit_panels.py <armA> <armB> <outdir> <evt> [evt ...] [--box CM]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(arm, evt):
    p = os.path.join(arm, f"pr_evt{evt}", f"calib-pr-evt{evt}.json")
    with open(p) as f:
        return json.load(f)


def panel(ax, d, c0, c1, box, title):
    ts = d["track_shower"]
    mv = d["main_vertex"]
    vx = (mv["x"], mv["y"], mv["z"])
    lo0, hi0 = vx["xyz".index(c0)] - box, vx["xyz".index(c0)] + box
    lo1, hi1 = vx["xyz".index(c1)] - box, vx["xyz".index(c1)] + box
    mv_cid = mv["cluster_id"]
    xs, ys, main = ts[c0], ts[c1], ts["cluster_id"]
    inbox_o = [(x, y) for x, y, c in zip(xs, ys, main)
               if lo0 <= x <= hi0 and lo1 <= y <= hi1 and c != mv_cid]
    inbox_m = [(x, y) for x, y, c in zip(xs, ys, main)
               if lo0 <= x <= hi0 and lo1 <= y <= hi1 and c == mv_cid]
    if inbox_o:
        ax.scatter(*zip(*inbox_o), s=2, c="0.8", linewidths=0, zorder=1)
    if inbox_m:
        ax.scatter(*zip(*inbox_m), s=2, c="0.55", linewidths=0, zorder=2)
    cmap = plt.get_cmap("tab10")
    for i, seg in enumerate(d["segments"]):
        pts = seg["points"]
        px = [p[ "xyz".index(c0)] for p in
              ([ [q["x"], q["y"], q["z"]] for q in pts ] if isinstance(pts[0], dict)
               else pts)]
        py = [p[ "xyz".index(c1)] for p in
              ([ [q["x"], q["y"], q["z"]] for q in pts ] if isinstance(pts[0], dict)
               else pts)]
        if not any(lo0 <= x <= hi0 and lo1 <= y <= hi1 for x, y in zip(px, py)):
            continue
        ax.plot(px, py, "-", color=cmap(i % 10), lw=1.4, zorder=3)
        ax.plot(px, py, ".", color=cmap(i % 10), ms=3.5, zorder=4)
    ax.plot(vx["xyz".index(c0)], vx["xyz".index(c1)], "r*", ms=14, zorder=5)
    ax.set_xlim(lo0, hi0)
    ax.set_ylim(lo1, hi1)
    ax.set_xlabel(f"{c0} [cm]")
    ax.set_ylabel(f"{c1} [cm]")
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armA")
    ap.add_argument("armB")
    ap.add_argument("outdir")
    ap.add_argument("evts", nargs="+")
    ap.add_argument("--box", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for evt in args.evts:
        try:
            da, db = load(args.armA, evt), load(args.armB, evt)
        except FileNotFoundError as e:
            print(f"evt {evt}: SKIP ({e})", file=sys.stderr)
            continue
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for row, (d, nm) in enumerate([(da, os.path.basename(args.armA)),
                                       (db, os.path.basename(args.armB))]):
            nseg = len(d["segments"])
            for col, (c0, c1) in enumerate([("x", "y"), ("x", "z"), ("y", "z")]):
                panel(axes[row][col], d, c0, c1, args.box,
                      f"{nm}  evt {evt}  {c0.upper()}-{c1.upper()}  ({nseg} segs)")
        fig.tight_layout()
        out = os.path.join(args.outdir, f"pr98_evt{evt}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()

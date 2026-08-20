#!/usr/bin/env python3
"""doc pr/99 -- static before/after fit panels zoomed on an OWNER-GIVEN point.

Fork of scripts/pr98_fit_panels.py (fork-by-duplication; that script stays the
pr/98 record).  The one behavioral change: the zoom box is centered on a
caller-supplied (x,y,z) -- the owner's complaint coordinate -- instead of arm
A's main vertex, so a symptom far from the vertex gets its own panel.  Rows =
arm A / arm B, cols = X-Y / X-Z / Y-Z; charge = track_shower points; fits =
segments[].points polylines; red star = each arm's main vertex (may be outside
the box).

Usage: pr99_point_panels.py <armA> <armB> <outdir> <evt> <x> <y> <z> [--box CM] [--tag T]
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


def panel(ax, d, c0, c1, center, box, title):
    ts = d["track_shower"]
    mv = d["main_vertex"]
    vx = (mv["x"], mv["y"], mv["z"])
    lo0, hi0 = center["xyz".index(c0)] - box, center["xyz".index(c0)] + box
    lo1, hi1 = center["xyz".index(c1)] - box, center["xyz".index(c1)] + box
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
        px = [p["xyz".index(c0)] for p in
              ([[q["x"], q["y"], q["z"]] for q in pts] if isinstance(pts[0], dict)
               else pts)]
        py = [p["xyz".index(c1)] for p in
              ([[q["x"], q["y"], q["z"]] for q in pts] if isinstance(pts[0], dict)
               else pts)]
        if not any(lo0 <= x <= hi0 and lo1 <= y <= hi1 for x, y in zip(px, py)):
            continue
        ax.plot(px, py, "-", color=cmap(i % 10), lw=1.4, zorder=3)
        ax.plot(px, py, ".", color=cmap(i % 10), ms=3.5, zorder=4)
    ax.plot(vx["xyz".index(c0)], vx["xyz".index(c1)], "r*", ms=14, zorder=5)
    ax.plot(center["xyz".index(c0)], center["xyz".index(c1)], "kx", ms=10, zorder=6)
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
    ap.add_argument("evt")
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("z", type=float)
    ap.add_argument("--box", type=float, default=15.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    center = (args.x, args.y, args.z)
    try:
        da, db = load(args.armA, args.evt), load(args.armB, args.evt)
    except FileNotFoundError as e:
        sys.exit(f"evt {args.evt}: {e}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (d, nm) in enumerate([(da, os.path.basename(args.armA)),
                                   (db, os.path.basename(args.armB))]):
        nseg = len(d["segments"])
        for col, (c0, c1) in enumerate([("x", "y"), ("x", "z"), ("y", "z")]):
            panel(axes[row][col], d, c0, c1, center, args.box,
                  f"{nm}  evt {args.evt}  {c0.upper()}-{c1.upper()}  ({nseg} segs)")
    fig.tight_layout()
    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(args.outdir, f"pr99_evt{args.evt}{tag}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

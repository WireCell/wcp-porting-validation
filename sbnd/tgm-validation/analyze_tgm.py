#!/usr/bin/env python3
"""Static XY / XZ / YZ sanity plots of cumulated TGM endpoints.

TGM endpoints are the exit points of through-going tracks; in SCE-corrected
true space they should populate the SURFACE of the fiducial-volume box, so the
three 2D projections should each look like the outline of a rectangle (points
concentrated on the edges / faces, with the FV box drawn for reference).

Usage:
  analyze_tgm.py <tgm_points.npz> [-o tgm_views.png] [--body]
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# FV box (cm), margin-shrunk, from clus.jsonnet dvm.overall (must match the cfg).
FV = dict(xlo=-201.05 + 2.0, xhi=201.05 - 2.0,
          ylo=-199.312 + 2.5, yhi=199.312 - 2.5,
          zlo=0.85 + 3.0, zhi=500.15 - 3.0)


def _panel(ax, a, b, alo, ahi, blo, bhi, alabel, blabel, body=None, pad=55.0):
    # View limits: FV box + padding (a handful of bad-t0 clusters produce
    # absurd x_t0cor; clip the VIEW so the box is visible -- count reported).
    axlim = (alo - pad, ahi + pad)
    bylim = (blo - pad, bhi + pad)
    if body is not None and body[0].size:
        ax.scatter(body[0], body[1], s=0.4, c="0.8", alpha=0.3, linewidths=0,
                   label="body")
    ax.scatter(a, b, s=4, c="crimson", alpha=0.6, linewidths=0, label="endpoints")
    ax.add_patch(Rectangle((alo, blo), ahi - alo, bhi - blo,
                           fill=False, ec="navy", lw=1.5, ls="--", label="FV box"))
    ax.set_xlim(*axlim)
    ax.set_ylim(*bylim)
    ax.set_xlabel("%s [cm]" % alabel)
    ax.set_ylabel("%s [cm]" % blabel)
    ax.set_title("%s vs %s" % (blabel, alabel))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("-o", "--out", default="tgm_views.png")
    ap.add_argument("--body", action="store_true",
                    help="also scatter the (grey) track body points")
    args = ap.parse_args()

    d = np.load(args.npz)
    ex, ey, ez = d["end_x"], d["end_y"], d["end_z"]
    body = None
    if args.body:
        body = (d["body_x"], d["body_y"], d["body_z"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # XY (z collapsed), XZ (y collapsed), YZ (x collapsed)
    _panel(axes[0], ex, ey, FV["xlo"], FV["xhi"], FV["ylo"], FV["yhi"],
           "x", "y", body=(body[0], body[1]) if body else None)
    _panel(axes[1], ex, ez, FV["xlo"], FV["xhi"], FV["zlo"], FV["zhi"],
           "x", "z", body=(body[0], body[2]) if body else None)
    _panel(axes[2], ey, ez, FV["ylo"], FV["yhi"], FV["zlo"], FV["zhi"],
           "y", "z", body=(body[1], body[2]) if body else None)
    axes[0].legend(loc="upper right", fontsize=8, markerscale=2)
    fig.suptitle("TGM tagged-track endpoints (%d points)" % ex.size)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("Wrote %s  (%d endpoints)" % (args.out, ex.size))


if __name__ == "__main__":
    main()

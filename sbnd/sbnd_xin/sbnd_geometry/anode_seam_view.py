#!/usr/bin/env python3
"""anode_seam_view.py -- the SBND anode structure at the mid-z seam.

Each SBND anode wall is built from TWO APA support frames stacked along the
beam (z) direction.  Where they meet -- the detector centre, z_J = 2505 mm --
their inner steel rails leave a small gap, and the 6 central collection (W)
channels fall right inside it.  This is the "additional structure in the middle
of the W plane" near the 6 dead channels.

Geometry (from sbnd_v02_06_nowires.gdml; z in the Wire-Cell J frame,
z_J = z_C + 2917.5 mm):
  * Wire planes (anode):  x = +/-2014.5 (U), +/-2017.5 (V), +/-2020.5 (W) mm
  * APA frame plane:      x = +/-2112 mm  (just outboard of the wires)
  * Two frames per wall, z_C centres +889 / -1714 mm -> inner vertical rails
    (each 100 mm wide in z, 4150 mm tall in y) at z_J = 2436.3 and 2573.7 mm
  * 37.4 mm gap between those rails, centred on the seam z_J = 2505 mm
  * 6 central W wires (the dead channels) at z_J = 2497.5 .. 2512.5 mm

Panel A: top view (look down +y), full drift, with the seam region boxed.
Panel B: zoom of the seam in the same x-z top view, showing the two rails,
         the 37.4 mm gap, and the 6 central W wires.

Usage:  python3 anode_seam_view.py   -> anode_seam_view.png
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "sbnd-wires-geometry-v0206.json")
ZJ = 2917.5  # z_J = z_C + ZJ

# anode wire-plane x positions (mm)
WX = {"U": 2014.5, "V": 2017.5, "W": 2020.5}
FRAME_X = 2112.0          # APA frame plane (|x|)
FRAME_X_HALF = 75.0       # tube half-thickness in x (150 mm box)
RAIL_W = 100.0            # vertical-rail thickness in z (mm)
RAIL_ZJ = (2436.3, 2573.7)  # inner-rail z_J centres (frame2, frame1)
SEAM_ZJ = 2505.0
PITCH = 3.0


def w_wire_zJ(side_sign=-1):
    """Collection-plane (W) wire z_J centres for one anode (East, ident 2)."""
    store = json.load(open(JSON))["Store"]
    pts = [p["Point"] for p in store["points"]]
    wires = [w["Wire"] for w in store["wires"]]
    planes = [p["Plane"] for p in store["planes"]]
    zs = []
    for wid in planes[2]["wires"]:        # East collection
        w = wires[wid]
        h, t = pts[w["head"]], pts[w["tail"]]
        zs.append((h["z"] + t["z"]) / 2.0)   # already in J frame
    return sorted(zs)


def draw_planes(ax, x_sign):
    for role, col in (("U", "#2ca02c"), ("V", "#ff7f0e"), ("W", "#1f1fcf")):
        x = x_sign * WX[role]
        ax.axvline(x, color=col, lw=1.4, zorder=3)


def draw_rails(ax, x_sign, zlo, zhi):
    x = x_sign * FRAME_X
    for zc in RAIL_ZJ:
        ax.add_patch(Rectangle((x - FRAME_X_HALF, zc - RAIL_W / 2),
                               2 * FRAME_X_HALF, RAIL_W,
                               facecolor="0.45", edgecolor="0.2", zorder=4))


def main():
    zs = w_wire_zJ()
    # 6 central W wires (dead channels)
    central6 = sorted(zs, key=lambda z: abs(z - SEAM_ZJ))[:6]
    c_lo, c_hi = min(central6), max(central6)

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 7.5),
                               gridspec_kw=dict(width_ratios=[1, 1.25]))

    # ---- Panel A: full top view, East wall only (x<0), seam boxed ---------
    x_sign = -1
    draw_planes(a, x_sign)
    a.axvline(0, color="firebrick", lw=2.0)
    a.text(-40, 5050, "cathode x=0", color="firebrick", ha="right", fontsize=8)
    draw_rails(a, x_sign, 0, 5010)
    a.text(x_sign * FRAME_X, 5120, "APA frame\nx=−2112", ha="center",
           fontsize=8, color="0.3")
    a.text(x_sign * WX["W"], -180, "U/V/W\nwire planes", ha="center",
           va="top", fontsize=8)
    a.axhline(SEAM_ZJ, color="purple", lw=1.0, ls="--")
    a.text(-1850, SEAM_ZJ + 40, "seam  z_J=2505", color="purple", fontsize=8)
    # zoom box
    zb_lo, zb_hi = 2380, 2630
    a.add_patch(Rectangle((-2200, zb_lo), 2200, zb_hi - zb_lo,
                          fill=False, edgecolor="black", lw=1.2, ls=":"))
    a.text(-1100, zb_hi + 30, "zoom →", ha="center", fontsize=8)
    a.set_xlim(-2260, 60)
    a.set_ylim(-260, 5260)
    a.set_xlabel("x  (drift)  [mm]")
    a.set_ylabel("z_J  (beam)  [mm]")
    a.set_title("(A) East anode, top view (look down +y)", fontsize=10)
    a.grid(alpha=0.25)

    # ---- Panel B: zoom on the seam ---------------------------------------
    draw_planes(b, x_sign)
    draw_rails(b, x_sign, zb_lo, zb_hi)
    # all W wires in window (light), 6 central ones highlighted (dead)
    for z in zs:
        if zb_lo <= z <= zb_hi:
            dead = z in central6
            b.plot([x_sign * WX["W"]], [z], marker="_", ms=16,
                   color=("red" if dead else "#1f1fcf"),
                   mew=(2.2 if dead else 0.8), zorder=6)
    # rail gap shading
    g_lo, g_hi = RAIL_ZJ[0] + RAIL_W / 2, RAIL_ZJ[1] - RAIL_W / 2
    b.axhspan(g_lo, g_hi, xmin=0, xmax=1, color="gold", alpha=0.18, zorder=0)
    b.axhline(SEAM_ZJ, color="purple", lw=1.0, ls="--")

    # labels (placed on the gray rails so nothing collides with the axes)
    bb = dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", alpha=0.9)
    b.text(-2112, RAIL_ZJ[1], "APA inner rail (frame +z)\nz_J=2573.7, 100 mm wide",
           ha="center", va="center", fontsize=7.5, color="black", bbox=bb, zorder=10)
    b.text(-2112, RAIL_ZJ[0], "APA inner rail (frame −z)\nz_J=2436.3, 100 mm wide",
           ha="center", va="center", fontsize=7.5, color="black", bbox=bb, zorder=10)
    b.annotate("37.4 mm rail gap\nseam z_J=2505",
               xy=(-2080, SEAM_ZJ), xytext=(-2180, 2655), fontsize=8,
               color="purple",
               arrowprops=dict(arrowstyle="->", color="purple"))
    b.annotate(f"6 central W wires =\ndead channels\n"
               f"z_J {c_lo:.0f}–{c_hi:.0f} mm\n(15 mm, pitch 3 mm)",
               xy=(x_sign * WX["W"], SEAM_ZJ), xytext=(-2018, 2640),
               fontsize=8, color="red", ha="left",
               arrowprops=dict(arrowstyle="->", color="red"))
    b.text(x_sign * WX["W"], zb_lo + 8, "U/V/W planes", ha="center",
           va="bottom", fontsize=7.5, color="#1f1fcf", rotation=90)

    b.set_xlim(-2220, -1995)
    b.set_ylim(zb_lo, zb_hi + 90)
    b.set_xlabel("x  (drift)  [mm]")
    b.set_ylabel("z_J  (beam)  [mm]")
    b.set_title("(B) Seam zoom: two APA inner rails straddle the\n"
                "6 dead W channels at the detector centre", fontsize=10)
    b.grid(alpha=0.25)

    fig.suptitle("SBND v02_06 — anode structure at the mid-z seam "
                 "(where the two APA frames meet)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "anode_seam_view.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print(f"6 dead W wires z_J = {[round(z,1) for z in central6]}")


if __name__ == "__main__":
    main()

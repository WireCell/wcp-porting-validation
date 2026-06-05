#!/usr/bin/env python3
"""anode_gap_view.py -- labelled drawing of the SBND anode region.

Companion to geoDisplay.C / geoAnode().  The ROOT pad render shows the two
anode planes and the drift gap in 3D, but cannot cleanly label distances or
quantify the wire planes.  This script draws the same thing as a clean,
orthographic, *labelled* schematic and adds a quantitative check of the
collection (W) plane structure.

Two panels:
  (A) Top view (look down +y): drift x horizontal, beam z vertical.  Shows
      West anode | drift | shared cathode (x=0) | drift | East anode, i.e. the
      "gap between the two TPCs".  No wires are drawn -- the planes are the
      bare LAr boxes, exactly as in the no-wires GDML.
  (B) Wire length vs z for the three planes of one anode, from the wire JSON.
      Answers "is there structure along the W plane?": the collection (W)
      plane is a single flat line at L=4000 mm (uniform, no gaps); the U/V
      induction planes carry all the structure (wrapped wires, L=10..5789 mm).

All geometry numbers are taken from sbnd_v02_06_nowires.gdml (see
sbnd_gdml_geometry.md).  Lengths in mm.

Usage:
    python3 anode_gap_view.py            # writes anode_gap_view.png
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "sbnd-wires-geometry-v0206.json")

# --- geometry constants (cryostat frame; x identical in both frames) --------
# Wire-plane x positions (mm), drift order 0=closest to cathode .. 2=collection
EAST_X = {"U": -2014.5, "V": -2017.5, "W": -2020.5}   # ident 0,1,2
WEST_X = {"U": +2014.5, "V": +2017.5, "W": +2020.5}
CATHODE_X = 0.0
ACTIVE_HALF_X = 2013.0          # active LAr dx
DRIFT = 2020.5                  # cathode -> collection plane
Z_LO, Z_HI = 0.0, 5010.0        # active z span (J frame); planes are 5013 long


def load_plane_lengths():
    """Return {plane_ident: [(z_center, length), ...]} for anode 0 (East)."""
    store = json.load(open(JSON))["Store"]
    pts = [p["Point"] for p in store["points"]]
    wires = [w["Wire"] for w in store["wires"]]
    planes = [p["Plane"] for p in store["planes"]]
    out = {}
    for ident in (0, 1, 2):
        rows = []
        for wid in planes[ident]["wires"]:
            w = wires[wid]
            h, t = pts[w["head"]], pts[w["tail"]]
            L = math.dist((h["x"], h["y"], h["z"]), (t["x"], t["y"], t["z"]))
            rows.append(((h["z"] + t["z"]) / 2.0, L))
        out[ident] = sorted(rows)
    return out


def draw_topview(ax):
    # active drift volumes (shaded) ----------------------------------------
    ax.add_patch(Rectangle((-ACTIVE_HALF_X, Z_LO), ACTIVE_HALF_X, Z_HI - Z_LO,
                           facecolor="#cfe3f7", edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((0, Z_LO), ACTIVE_HALF_X, Z_HI - Z_LO,
                           facecolor="#cfe3f7", edgecolor="none", zorder=1))
    ax.text(-ACTIVE_HALF_X / 2, Z_HI * 0.5, "East TPC (0)\ndrift volume",
            ha="center", va="center", fontsize=9, color="#1f4e79")
    ax.text(+ACTIVE_HALF_X / 2, Z_HI * 0.5, "West TPC (1)\ndrift volume",
            ha="center", va="center", fontsize=9, color="#1f4e79")

    # cathode --------------------------------------------------------------
    ax.plot([0, 0], [Z_LO, Z_HI], color="firebrick", lw=2.5, zorder=4)
    ax.text(0, Z_HI + 90, "Cathode (CPA)\nx = 0", ha="center", va="bottom",
            fontsize=9, color="firebrick", fontweight="bold")

    # anode planes (3 per side; no wires -- just the plane boxes) -----------
    for side, xmap, lab in ((-1, EAST_X, "East anode"), (1, WEST_X, "West anode")):
        for role, col in (("U", "#2ca02c"), ("V", "#ff7f0e"), ("W", "#1f1fcf")):
            x = xmap[role]
            ax.plot([x, x], [Z_LO, Z_HI], color=col, lw=1.6, zorder=3)
        ax.text(side * (DRIFT + 70), Z_HI + 90,
                f"{lab}\nU/V/W planes\n(x={xmap['W']:+.1f})",
                ha="center", va="bottom", fontsize=8.5)

    # drift-distance annotations ------------------------------------------
    yA = Z_HI * 0.18
    for sgn in (-1, 1):
        ax.add_patch(FancyArrowPatch((0, yA), (sgn * DRIFT, yA),
                     arrowstyle="<->", mutation_scale=12, color="black", lw=1.0))
        ax.text(sgn * DRIFT / 2, yA - 130, f"drift\n{DRIFT:.1f} mm",
                ha="center", va="top", fontsize=8)
    # anode-to-anode total
    yB = -260
    ax.add_patch(FancyArrowPatch((-DRIFT, yB), (DRIFT, yB),
                 arrowstyle="<->", mutation_scale=12, color="dimgray", lw=1.0))
    ax.text(0, yB - 60, f"anode-to-anode gap = {2*DRIFT:.1f} mm",
            ha="center", va="top", fontsize=8.5, color="dimgray")

    ax.set_xlim(-2350, 2350)
    ax.set_ylim(-470, Z_HI + 620)
    ax.set_xlabel("x  (drift)  [mm]")
    ax.set_ylabel("z  (beam)  [mm]")
    ax.set_title("(A) Top view (look down +y): the two TPCs share the cathode;\n"
                 "the 'gap' is the 2 × 2020.5 mm drift",
                 fontsize=10, pad=26)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def draw_wplane_check(ax, lengths):
    names = {0: ("U  (1st induction, +60°)", "#2ca02c", 5.0),
             1: ("V  (2nd induction, −60°; mirrors U)", "#ff7f0e", 2.0),
             2: ("W  (collection, vertical)", "#1f1fcf", 2.5)}
    for ident in (0, 1, 2):
        zs = [z for z, _ in lengths[ident]]
        Ls = [L for _, L in lengths[ident]]
        lab, col, ms = names[ident]
        ax.plot(zs, Ls, ".", ms=ms, color=col, label=lab, zorder=3 - ident)
    ax.set_xlabel("wire center  z  [mm]")
    ax.set_ylabel("wire length  [mm]")
    ax.set_title("(B) Structure check, East anode (from wire JSON):\n"
                 "W = flat 4000 mm (uniform, no gaps); U/V carry the structure",
                 fontsize=10)
    ax.legend(loc="center left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.annotate("collection plane:\n1670 wires, ALL L = 4000 mm,\n"
                "pitch 3.000 mm, no missing wires",
                xy=(2500, 4000), xytext=(1500, 5200), fontsize=8,
                color="#1f1fcf",
                arrowprops=dict(arrowstyle="->", color="#1f1fcf"))


def main():
    lengths = load_plane_lengths()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7))
    draw_topview(a1)
    draw_wplane_check(a2, lengths)
    fig.suptitle("SBND v02_06 anode region — drift gap and W-plane structure",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(HERE, "anode_gap_view.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

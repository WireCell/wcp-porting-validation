#!/usr/bin/env python3
"""ProtoDUNE-VD 3-D imaging + deghosting algorithm diagram (wide 16:9).

Counterpart of pdhd/pics/make_imaging_chain_diagram.py.  Draws the live per-CRP
imaging pipeline traced from
`cfg/pgrapher/experiment/protodunevd/img.jsonnet` (driven by
`pdvd/wct-img-all.jsonnet`, which forces `per_anode(anode, "multi")` with the
charge>0 slicing threshold nthreshold=1e-6):

  SP frame (gauss/wiener, per CRP; 8 CRPs image independently)
   -> pre-proc (CMMModifier . FrameMasking . ChargeErrorFrameEstimator)
   -> FrameFanout {active fork, masked fork}
   active fork:
     -> ① Slice   (MaskSlices: time-slice per plane, threshold on charge)
     -> ② Tile    (GridTiling / RayGrid, per face: fired U/V/W strips' triple
                   overlap -> 2-D blob per slice, incl. FALSE crossings = ghosts)
     -> ③ Solve & Deghost  (the centerpiece: BlobClustering stacks blobs across
                   slices into 3-D, then the asymmetric ladder
                     bc -> gd1 -> solve -> ld1 -> gd2 -> solve -> ld2 -> solve -> ld3 -> gc
                   ProjectionDeghosting x2 (global, cross-view) +
                   InSliceDeghosting x3 (local, charge-based) +
                   ChargeSolving x3)
     -> ④ ClusterFileSink -> clusters-apa-*-ms-active.tar.gz  (3-view live
                   blobs, post-deghosting)
  masked fork:
     -> 2-view blobs over dead regions (dummy plane, coarse span 1500),
        geometry only -> clusters-apa-*-ms-masked.tar.gz

Insets: the ghost concept illustrated with Fig. 8 of the WCT imaging note
(B. Viren, 2019) -- a toy 3-plane tiling where blobs at strip triple-overlaps
either surround points (real) or none (ghosts); a synthetic "slices stacked
along drift -> 3-D" schematic; and the full-event 3-D charge display (Bee,
deghosted result) for PDVD data run 039252 evt 298567.

Output: pdvd/pics/pdvd_imaging_chain.{png,pdf}
"""
import os
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from diagram_helpers import Canvas, INK

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "imaging_src")

C_IN = "#4a4a4a";    BG_IN = "#eef1f5"
C_SLICE = "#1f7a8c"; BG_SLICE = "#e3f2f4"
C_TILE = "#3d5a99";  BG_TILE = "#e8ecf7"
C_SOLVE = "#2e7d4f"; BG_SOLVE = "#e7f3ec"     # ChargeSolving
C_PD = "#7b1fa2"                              # ProjectionDeghosting (global)
C_ID = "#c2185b"                              # InSliceDeghosting (local)
C_BC = "#5a6572"                              # blob clustering / grouping
C_CLUS = "#b3671a"; BG_CLUS = "#fbeede"       # global geom clustering
C_OUT = "#2e7d4f";  BG_OUT = "#eaf5ee"
C_DEAD = "#8a8f96"; BG_DEAD = "#f0f1f3"       # dead/masked fork
BG_LADDER = "#f6eefb"                         # deghost band background
C_DEGHOST = "#7b1fa2"


def pill(c, cx, cy, w, h, text, fc, ec, fs=10.5, tc="white", bold=True):
    c.ov.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.10",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=8))
    c.ov.text(cx, cy, text, ha="center", va="center", fontsize=fs,
              color=tc, zorder=9, fontweight="bold" if bold else "normal")


def draw_3d_stack(c, cx, cy, color, s=1.0):
    """Synthetic schematic: 2-D transverse blobs at successive drift slices,
    offset along a drift axis and connected -> the 3-D image.  `s` scales it."""
    ax = c.ov
    dx, dy = 0.56 * s, 0.37 * s      # per-slice offset (drift direction)
    n = 4
    base = np.array([[-0.55, -0.34], [0.55, -0.30],
                     [0.62, 0.34], [-0.48, 0.38]]) * s
    prev = None
    for i in range(n):
        off = np.array([cx - (n - 1) * dx / 2 + i * dx,
                        cy - (n - 1) * dy / 2 + i * dy])
        quad = base * [1.0, 0.7] + off
        ax.add_patch(Polygon(quad, closed=True, facecolor=color,
                             edgecolor="#2b2b2b", alpha=0.55, linewidth=1.2,
                             zorder=5 + i))
        blob = off + np.array([0.02 * s, 0.0])
        ax.add_patch(Polygon(
            (base * [0.34, 0.26]) + blob, closed=True, facecolor="#12305c",
            edgecolor="#0a1c38", alpha=0.95, linewidth=0.9, zorder=5 + i))
        if prev is not None:
            ax.add_patch(FancyArrowPatch(prev, blob, arrowstyle="-",
                         color="#12305c", lw=1.5, alpha=0.8, zorder=20))
        prev = blob
    # drift axis arrow
    ax.add_patch(FancyArrowPatch(
        (cx - (n - 1) * dx / 2 - 0.9 * s, cy - (n - 1) * dy / 2 - 0.68 * s),
        (cx + (n - 1) * dx / 2 + 0.42 * s, cy + (n - 1) * dy / 2 - 0.24 * s),
        arrowstyle="-|>", mutation_scale=17, color="#555", lw=1.9, zorder=4))
    ax.text(cx + 1.05 * s, cy - 1.02 * s, "drift  (X = slice-time)",
            ha="center", va="center", fontsize=10.5, color="#555", rotation=17)


def main():
    c = Canvas()
    c.title("ProtoDUNE-VD Wire-Cell 3-D Imaging & Deghosting  ·  per CRP",
            "tomographic tiling: fired U/V/W wire strips coincide → blobs "
            "(+ false crossings) → charge-solve & deghost → 3-D clusters",
            my=8.62, sy=8.02, mfs=27, sfs=17.5)

    # ================= spine (high-level flow) ==========================
    ys = 6.95
    c.box(1.15, ys, 1.95, 1.55, "SP frames\nper CRP\ngauss / wiener\n(×8 CRPs)",
          BG_IN, C_IN, fs=12, tc=C_IN)

    c.algobox(4.45, ys, 3.35, 1.88, "① Slice", [
        "MaskSlices — time-slice",
        "per plane, charge > 0 (1e-6)",
        "→ fired U/V/W activity / slice",
        "multi-pass: UVW · UV · VW · UW",
    ], BG_SLICE, C_SLICE, title_fs=16, bullet_fs=11.8, dy=0.345)

    c.algobox(8.30, ys, 3.45, 1.88, "② Tile — RayGrid", [
        "GridTiling, per face 0/1",
        "U/V/W strips' triple overlap",
        "→ 2-D blob per slice",
        "false crossings ⇒ ghost blobs",
    ], BG_TILE, C_TILE, title_fs=16, bullet_fs=11.8, dy=0.345)

    c.algobox(12.25, ys, 3.75, 1.88, "③ Solve & Deghost", [
        "BlobClustering: stack blobs",
        "across slices → 3-D clusters",
        "then charge-solve ⇄ deghost",
        "ladder (3 rounds) ▼ below",
    ], BG_LADDER, C_DEGHOST, title_fs=16, bullet_fs=11.8, dy=0.345)

    c.box(15.15, ys, 1.78, 1.70, "④ ClusterFileSink\n→ clusters-apa-*\n"
          "-ms-active.tar.gz\n(3-view live blobs,\npost-deghost)",
          BG_OUT, C_OUT, fs=9.6, tc=C_OUT)

    # spine arrows
    c.arrow((2.13, ys), (2.75, ys), C_IN)
    c.arrow((6.14, ys), (6.55, ys), C_TILE)
    c.arrow((10.04, ys), (10.35, ys), C_DEGHOST)
    c.arrow((14.14, ys), (14.26, ys), C_OUT)
    # pre-proc note under the input box (folded into the input->slice arrow)
    c.ov.text(1.15, ys - 1.06, "pre-proc:  CMMModifier ·", ha="center",
              va="center", fontsize=9.2, color="#8a8a8a", style="italic")
    c.ov.text(1.15, ys - 1.30, "FrameMasking · ChargeError", ha="center",
              va="center", fontsize=9.2, color="#8a8a8a", style="italic")

    # dead/masked subordinate fork (FrameFanout) — one compact readable line
    c.ov.text(8.55, 5.72,
              "⊕ FrameFanout → dead / masked fork:  2-view blobs (dummy plane, "
              "span 1500), geometry only → clusters-apa-*-ms-masked.tar.gz",
              ha="center", va="center", fontsize=10, color="#6e747c",
              style="italic")
    c.ov.add_patch(FancyArrowPatch((4.45, ys - 0.94), (5.55, 5.86),
                   arrowstyle="-|>", mutation_scale=12, color=C_DEAD,
                   lw=1.5, ls=(0, (4, 2)), zorder=3))

    # ================= deghost ⇄ solve ladder band ======================
    lx0, lx1, lyb, lyt = 2.30, 14.55, 3.78, 5.42
    c.ov.add_patch(FancyBboxPatch(
        (lx0, lyb), lx1 - lx0, lyt - lyb,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.0, edgecolor=C_DEGHOST, facecolor=BG_LADDER, zorder=2))
    c.ov.text(lx0 + 0.18, lyt - 0.22,
              "inside ③  ·  charge-solve ⇄ deghost ladder  (uboone-solving) — "
              "note the asymmetry:  ProjectionDeghosting ×2,  ChargeSolving ×3,  "
              "InSliceDeghosting ×3",
              ha="left", va="center", fontsize=11, fontweight="bold",
              color=C_DEGHOST, zorder=3)

    # the exact asymmetric sequence, ending on the pipeline's own gc node:
    #   bc gd1 CS ld1 gd2 CS ld2 CS ld3 gc   (solving "full")
    cy = 4.66
    steps = [
        ("BC", C_BC), ("PD", C_PD), ("CS", C_SOLVE), ("ID₁", C_ID),
        ("PD", C_PD), ("CS", C_SOLVE), ("ID₂", C_ID), ("CS", C_SOLVE),
        ("ID₃", C_ID), ("GGC", C_CLUS),
    ]
    n = len(steps)
    x0, x1 = lx0 + 0.82, lx1 - 0.68
    xs = np.linspace(x0, x1, n)
    pw = 1.0
    for i, (lab, fc) in enumerate(steps):
        if i:
            c.ov.add_patch(FancyArrowPatch(
                (xs[i - 1] + pw / 2, cy), (xs[i] - pw / 2, cy),
                arrowstyle="-|>", mutation_scale=12, color="#8a8a8a",
                lw=1.6, zorder=7))
        pill(c, xs[i], cy, pw, 0.50, lab, fc, fc, fs=12.5, tc="white")

    # single-row legend for the pill keys, well below the pills
    leg = [
        ("BC", C_BC, "BlobClustering"),
        ("CS", C_SOLVE, "ChargeSolving ×3"),
        ("PD", C_PD, "ProjectionDeghosting ×2 (global)"),
        ("ID", C_ID, "InSliceDeghosting ×3 (local)"),
        ("GGC", C_CLUS, "GlobalGeomClustering"),
    ]
    lgx = [2.72, 4.42, 6.42, 9.85, 12.55]
    for (ab, col, txt), xx in zip(leg, lgx):
        pill(c, xx, 4.02, 0.52, 0.28, ab, col, col, fs=8.6)
        c.ov.text(xx + 0.36, 4.02, txt, ha="left", va="center", fontsize=10,
                  color=INK)

    # ================= insets (bottom row) ==============================
    # ghost concept (toy detector, Fig. 8 of the WCT imaging note) sits under
    # ② Tile; the 3-D result keeps a clean leader to the output box; the drift
    # schematic bridges them.  All three stay clear of the ladder band above.
    ybi = 0.74
    hg = c.place_image(os.path.join(SRC, "raygrid_fig8_ghost.png"), 2.95, 2.55,
                       ybi, "ghost concept — toy 3-plane tiling (Fig. 8)",
                       None, C_TILE, cap_fs=13)
    c.ov.text(2.95, ybi + hg + 0.12, "② tiling: false triple-overlaps ⇒ ghosts",
              ha="center", va="bottom", fontsize=11, color=C_TILE,
              style="italic")
    draw_3d_stack(c, 7.85, 2.15, "#9ec0e8", s=1.30)
    c.ov.text(7.85, 0.56, "stack 2-D blobs along drift → 3-D image",
              ha="center", va="top", fontsize=12.5, color=C_TILE,
              fontweight="bold")
    c.place_image(os.path.join(SRC, "img_event_3d.png"), 13.0, 3.42, ybi,
                  "full event 3-D charge display (Bee) — deghosted result",
                  (15.15, 6.10), C_OUT, box=(0.04, 0.10, 1.0, 0.99),
                  cap_fs=13.5)

    c.ov.text(8.0, 0.14, "Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/"
              "protodunevd/img.jsonnet + pdvd/wct-img-all.jsonnet  ·  ghost = "
              "B. Viren, Wire-Cell Toolkit Imaging (2019), Fig. 8;  3-D = data "
              "run 039252 evt 298567 (Bee img cloud)", ha="center",
              va="center", fontsize=9.5, color="#8a8a8a")
    c.save(os.path.join(HERE, "pdvd_imaging_chain"))


if __name__ == "__main__":
    main()

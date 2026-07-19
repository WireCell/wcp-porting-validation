#!/usr/bin/env python3
"""ProtoDUNE-HD 3-D imaging + deghosting algorithm diagram (wide 16:9).

Draws the live per-APA imaging pipeline traced from
`cfg/pgrapher/experiment/pdhd/img.jsonnet` (driven by `pdhd/wct-img-all.jsonnet`):

  SP frame (gauss/wiener, per APA)
   -> pre-proc (CMMModifier . FrameMasking . ChargeErrorFrameEstimator)
   -> ① Slice   (MaskSlices: time-slice per plane, threshold on charge)
   -> ② Tile    (GridTiling / RayGrid, per face: fired U/V/W strips' triple
                 overlap -> 2-D blob per slice, incl. FALSE crossings = ghosts)
   -> ③ Solve & Deghost  (the centerpiece: BlobClustering stacks blobs across
                 slices into 3-D, then the asymmetric ladder
                   bc -> gd1 -> solve -> ld1 -> gd2 -> solve -> ld2 -> solve -> ld3
                 ProjectionDeghosting x2 (global, cross-view) +
                 InSliceDeghosting x3 (local, charge-based) +
                 ChargeSolving x3)
   -> ④ Global cluster / write (GlobalGeomClustering -> ClusterFileSink)
   -> clusters-apa-*.tar.gz  (3-view live blobs, post-deghosting)

A subordinate dead/masked fork tiles 2-view blobs over dead regions (geometry
only, coarse span) -> clusters-apa-*-masked.tar.gz.

Insets: one time slice zoomed to a busy blob cluster (real vs flagged-ghost
blobs), the full-event Z-Y point cloud (deghosted result), and a synthetic
"slices stacked along drift -> 3-D" schematic.

Output: pdhd/pics/pdhd_imaging_chain.{png,pdf}
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


def draw_3d_stack(c, cx, cy, color):
    """Synthetic schematic: 2-D transverse blobs at successive drift slices,
    offset along a drift axis and connected -> the 3-D image."""
    ax = c.ov
    dx, dy = 0.46, 0.30      # per-slice offset (drift direction)
    n = 4
    base = np.array([[-0.55, -0.34], [0.55, -0.30], [0.62, 0.34], [-0.48, 0.38]])
    prev = None
    for i in range(n):
        off = np.array([cx - (n - 1) * dx / 2 + i * dx,
                        cy - (n - 1) * dy / 2 + i * dy])
        quad = base * [1.0, 0.7] + off
        ax.add_patch(Polygon(quad, closed=True, facecolor=color,
                             edgecolor="#2b2b2b", alpha=0.55, linewidth=1.1,
                             zorder=5 + i))
        blob = off + np.array([0.02, 0.0])
        ax.add_patch(Polygon(
            (base * [0.34, 0.26]) + blob, closed=True, facecolor="#12305c",
            edgecolor="#0a1c38", alpha=0.95, linewidth=0.8, zorder=5 + i))
        if prev is not None:
            ax.add_patch(FancyArrowPatch(prev, blob, arrowstyle="-",
                         color="#12305c", lw=1.3, alpha=0.8, zorder=20))
        prev = blob
    # drift axis arrow
    ax.add_patch(FancyArrowPatch(
        (cx - (n - 1) * dx / 2 - 0.75, cy - (n - 1) * dy / 2 - 0.55),
        (cx + (n - 1) * dx / 2 + 0.35, cy + (n - 1) * dy / 2 - 0.20),
        arrowstyle="-|>", mutation_scale=15, color="#555", lw=1.6, zorder=4))
    ax.text(cx + 0.9, cy - 0.85, "drift  (X = slice-time)", ha="center",
            va="center", fontsize=8.5, color="#555", rotation=17)


def main():
    c = Canvas()
    c.title("ProtoDUNE-HD Wire-Cell 3-D Imaging & Deghosting  ·  per APA",
            "tomographic tiling: fired U/V/W wire strips coincide → blobs "
            "(+ false crossings) → charge-solve & deghost → 3-D clusters")

    # ================= spine (high-level flow) ==========================
    ys = 7.05
    c.box(1.12, ys, 1.72, 1.30, "SP frames\nper APA\ngauss / wiener",
          BG_IN, C_IN, fs=10, tc=C_IN)

    c.algobox(4.05, ys, 3.05, 1.55, "① Slice", [
        "MaskSlices — time-slice",
        "per plane, threshold charge",
        "→ fired U/V/W activity / slice",
        "multi-pass: UVW · UV · VW · UW",
    ], BG_SLICE, C_SLICE, title_fs=12.5, bullet_fs=8.7, dy=0.255)

    c.algobox(7.70, ys, 3.20, 1.55, "② Tile — RayGrid", [
        "GridTiling, per face 0/1",
        "U/V/W strips' triple overlap",
        "→ 2-D blob per slice",
        "false crossings ⇒ ghost blobs",
    ], BG_TILE, C_TILE, title_fs=12.5, bullet_fs=8.7, dy=0.255)

    c.algobox(11.55, ys, 3.55, 1.55, "③ Solve & Deghost", [
        "BlobClustering: stack blobs",
        "across slices → 3-D clusters",
        "then charge-solve ⇄ deghost",
        "ladder (3 rounds) ▼ below",
    ], BG_LADDER, C_DEGHOST, title_fs=12.5, bullet_fs=8.7, dy=0.255)

    c.box(14.95, ys, 1.75, 1.45, "④ ClusterFileSink\n→ clusters-apa-*\n.tar.gz\n"
          "(3-view live blobs,\npost-deghost)",
          BG_OUT, C_OUT, fs=8.2, tc=C_OUT)

    # spine arrows
    c.arrow((1.98, ys), (2.52, ys), C_IN)
    c.arrow((5.58, ys), (6.10, ys), C_TILE)
    c.arrow((9.30, ys), (9.77, ys), C_DEGHOST)
    c.arrow((13.33, ys), (14.07, ys), C_OUT)
    # pre-proc note under the input box (folded into the input->slice arrow)
    c.ov.text(1.12, ys - 0.92, "pre-proc:  CMMModifier ·",
              ha="center", va="center", fontsize=7.2, color="#8a8a8a",
              style="italic")
    c.ov.text(1.12, ys - 1.15, "FrameMasking · ChargeError",
              ha="center", va="center", fontsize=7.2, color="#8a8a8a",
              style="italic")

    # dead/masked subordinate fork — one compact line off the slice stage
    c.ov.text(7.70, 6.02,
              "⊕ dead / masked fork:  2-view blobs (dummy plane, span 1500), "
              "geometry only → clusters-apa-*-masked.tar.gz",
              ha="center", va="center", fontsize=7.6, color="#6e747c",
              style="italic")
    c.ov.add_patch(FancyArrowPatch((4.05, ys - 0.78), (5.55, 6.10),
                   arrowstyle="-|>", mutation_scale=11, color=C_DEAD,
                   lw=1.3, ls=(0, (4, 2)), zorder=3))

    # ================= deghost ⇄ solve ladder band ======================
    lx0, lx1, lyb, lyt = 2.45, 14.30, 4.30, 5.62
    c.ov.add_patch(FancyBboxPatch(
        (lx0, lyb), lx1 - lx0, lyt - lyb,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.8, edgecolor=C_DEGHOST, facecolor=BG_LADDER, zorder=2))
    c.ov.text(lx0 + 0.16, lyt - 0.19,
              "inside ③  ·  charge-solve ⇄ deghost ladder  (uboone-solving) — "
              "note the asymmetry:  ProjectionDeghosting ×2,  ChargeSolving ×3,  "
              "InSliceDeghosting ×3",
              ha="left", va="center", fontsize=9.2, fontweight="bold",
              color=C_DEGHOST, zorder=3)
    # leader from spine box ③ down to the band
    c.ov.add_patch(FancyArrowPatch((11.55, ys - 0.78), (11.55, lyt + 0.01),
                   arrowstyle="-|>", mutation_scale=12, color=C_DEGHOST,
                   lw=1.6, zorder=3))

    # the exact asymmetric sequence, ending on the pipeline's own gc node:
    #   bc gd1 CS ld1 gd2 CS ld2 CS ld3 gc   (solving "full")
    cy = 5.02
    steps = [
        ("BC", C_BC), ("PD", C_PD), ("CS", C_SOLVE), ("ID₁", C_ID),
        ("PD", C_PD), ("CS", C_SOLVE), ("ID₂", C_ID), ("CS", C_SOLVE),
        ("ID₃", C_ID), ("GGC", C_CLUS),
    ]
    n = len(steps)
    x0, x1 = lx0 + 0.72, lx1 - 0.60
    xs = np.linspace(x0, x1, n)
    pw = 0.82
    for i, (lab, fc) in enumerate(steps):
        if i:
            c.ov.add_patch(FancyArrowPatch(
                (xs[i - 1] + pw / 2, cy), (xs[i] - pw / 2, cy),
                arrowstyle="-|>", mutation_scale=10, color="#8a8a8a",
                lw=1.4, zorder=7))
        pill(c, xs[i], cy, pw, 0.42, lab, fc, fc, fs=9.6, tc="white")

    # three-column legend for the pills
    leg = [
        ("BC", C_BC, "BlobClustering (stack across slices)"),
        ("CS", C_SOLVE, "ChargeSolving ×3 (grp→unif→lclus→ubn)"),
        ("PD", C_PD, "ProjectionDeghosting ×2 (global)"),
        ("ID", C_ID, "InSliceDeghosting ×3 (local, th300)"),
        ("GGC", C_CLUS, "GlobalGeomClustering (final node)"),
    ]
    col_x = [lx0 + 0.30, lx0 + 4.30, lx0 + 8.30]
    for i, (ab, col, txt) in enumerate(leg):
        cxl = col_x[i % 3]
        cyl = lyb + 0.46 - (i // 3) * 0.28
        pill(c, cxl + 0.18, cyl, 0.46, 0.23, ab, col, col, fs=6.8)
        c.ov.text(cxl + 0.48, cyl, txt, ha="left", va="center", fontsize=7.5,
                  color=INK)

    # ================= insets (bottom row) ==============================
    ybi = 1.02
    c.place_image(os.path.join(SRC, "img_slice_blobs.png"), 2.75, 2.95, ybi,
                  "one time slice — real blobs (navy) & flagged ghosts (magenta)",
                  (7.70, 6.20), C_TILE)
    draw_3d_stack(c, 8.05, 2.30, "#9ec0e8")
    c.ov.text(8.05, 0.82, "stack 2-D blobs along drift → 3-D image",
              ha="center", va="top", fontsize=10, color=C_TILE,
              fontweight="bold")
    c.ov.add_patch(FancyArrowPatch((8.05, 3.35), (8.55, 6.28), arrowstyle="-",
                   color=C_TILE, lw=1.3, alpha=0.8, ls=(0, (5, 3)), zorder=3))
    c.place_image(os.path.join(SRC, "img_event_zy.png"), 13.10, 3.80, ybi,
                  "full event, Z-Y — deghosted, charge-solved result",
                  (14.95, 6.30), C_OUT)

    c.footer("Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/pdhd/img.jsonnet + "
             "pdhd/wct-img-all.jsonnet  ·  data insets: ProtoDUNE-HD run 27305 "
             "evt 150 (img_plot cache)")
    c.save(os.path.join(HERE, "pdhd_imaging_chain"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ProtoDUNE-HD Wire-Cell 3-D clustering + charge-light (Q/L) matching diagram.

Counterpart of pdvd/pics/make_clus_chain_diagram.py (PD-VD original) with the
ProtoDUNE-HD configuration facts.

Wide 16:9 slide.  Two dataflow planes:
  - CHARGE / clustering (top spine): the 4-stage MultiAlgBlobClustering cascade
    per-face -> per-APA -> per-drift-group -> all-TPC, reading the live+dead
    imaging cluster tarballs.  PD-HD: 4 APAs, one imaging face each (even
    APAs image through face 0 / drift -x, odd through face 1 / drift +x; the
    opposite faces are non-imaging wall faces), drift groups {APA0,APA2} and
    {APA1,APA3}.
  - LIGHT / Q-L (lower band): the separate light-reco job (bridged by an
    opflash archive) feeding the joint QLMatching node, which pins a cluster T0
    that flows up into the all-TPC stage (switch_scope -> x_t0cor).

Two data insets (clus_chain_src/, committed): the clustering result 3-D cloud
coloured by cluster, and one flash's measured-vs-predicted light pattern.
Traced from cfg/pgrapher/experiment/pdhd/{clus,qlmatching,flash}.jsonnet
and pdhd/{wct-clustering,run_clus_evt,run_light_evt}.  Output:
pdhd/pics/pdhd_clus_ql_chain.{png,pdf}.
"""
import os
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from diagram_helpers import Canvas, W, H, INK

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "clus_chain_src")
OUT = os.path.join(HERE, "pdhd_clus_ql_chain")

# palette
C_IN = "#5a6572"       # input / passthrough
C_CLUS = "#2f6db5"     # clustering stages (charge)
C_LIGHT = "#d3781f"    # light reco
C_QL = "#7b1fa2"       # QLMatching
C_T0 = "#c0392b"       # the T0 connector
C_OUT = "#2e7d4f"      # output / clustering result
BG_CLUS = "#e9f1fb"
BG_QL = "#f4ecf8"


def main():
    c = Canvas()
    ov = c.ov
    ov.text(W / 2, 8.66, "ProtoDUNE Horizontal Drift — Wire-Cell 3-D "
            "Clustering  +  Charge–Light (Q/L) Matching Chain",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=INK)
    ov.text(W / 2, 8.22, "per-face → per-APA → per-drift-group → all-TPC "
            "clustering, with the joint QLMatching node pinning each cluster's "
            "drift T0", ha="center", va="center", fontsize=14, color="#444")

    # ================= CHARGE / clustering band =================
    c.group_bg(0.35, 6.18, 15.30, 1.80, C_CLUS, BG_CLUS,
               label="CHARGE   ·   3-D CLUSTERING   "
               "(MultiAlgBlobClustering cascade — live + dead blobs)",
               lx=5.55, ly=7.84, fs=13)

    cy = 7.00
    n_in = (1.92, cy)
    n_s1 = (4.78, cy)
    n_s2 = (7.28, cy)
    n_s3 = (10.15, cy)
    n_s4 = (13.42, cy)

    c.algobox(*n_in, 2.55, 1.50, "imaging clusters",
              ["ClusterFileSource ×4 APAs",
               "live (active) + dead (masked)",
               "→ PointTreeBuilding",
               "   (BlobSampler: live 3d, dead)"],
              "white", C_IN, tc=C_IN, title_fs=12, bullet_fs=8.9, dy=0.265)
    c.algobox(*n_s1, 2.45, 1.50, "① per-face  ×4",
              ["1 imaging face / APA",
               "pointed · live/dead · extend",
               "regular ×2 · prolong · close",
               "extend_loop · connect1"],
              "white", C_CLUS, tc=C_CLUS, title_fs=12, bullet_fs=9.0,
              dy=0.265)
    c.algobox(*n_s2, 2.05, 1.50, "② per-APA  ×4",
              ["merge faces",
               "deghost",
               "protect_overclustering"],
              "white", C_CLUS, tc=C_CLUS, title_fs=12, bullet_fs=9.0,
              dy=0.30)
    c.algobox(*n_s3, 3.28, 1.50, "③ per-drift-group  ×2",
              ["{APA0,2}·face0  /  {APA1,3}·face1",
               "extend·regular×2·prolong·close·ext_loop",
               "separate — 3-D track split + recover",
               "connect1·deghost·x_boundary·ν·isolated"],
              "white", C_CLUS, tc=C_CLUS, title_fs=12, bullet_fs=8.8,
              dy=0.265)
    c.algobox(*n_s4, 2.55, 1.50, "④ all-TPC  ×1",
              ["switch_scope → x_t0cor",
               "cathode_connect",
               "   (tip-touch, flash-T0 gate)",
               "→ mabc-all-apa.zip"],
              "#eaf5ee", C_OUT, tc=C_OUT, title_fs=12, bullet_fs=8.9,
              dy=0.265)

    def rt(node, w):
        return (node[0] + w / 2, node[1])

    def lf(node, w):
        return (node[0] - w / 2, node[1])

    c.arrow(rt(n_in, 2.55), lf(n_s1, 2.45), C_IN)
    c.arrow(rt(n_s1, 2.45), lf(n_s2, 2.05), C_CLUS)
    c.arrow(rt(n_s2, 2.05), lf(n_s3, 3.28), C_CLUS)

    # ================= LIGHT / Q-L band =================
    c.group_bg(0.35, 3.46, 15.30, 2.06, C_QL, BG_QL,
               label="LIGHT RECO  (separate wire-cell job)      →      "
               "CHARGE–LIGHT  (Q/L)  MATCHING",
               lx=6.7, ly=5.38, fs=13)

    qy = 4.42
    n_lr = (2.55, qy)
    n_ql = (9.75, qy)

    c.algobox(*n_lr, 3.95, 1.52, "LIGHT RECO — separate job",
              ["OpDecon — deconv, detect_saturation",
               "OpHitFinder — SPE-normalized hits",
               "OpFlashFinder — ≥5 PDs, ≥20 PE",
               "→ 160 flat-X-ARAPUCA flashes"],
              "white", C_LIGHT, tc=C_LIGHT, title_fs=12, bullet_fs=8.9,
              dy=0.275)
    c.algobox(*n_ql, 5.35, 1.66, "QLMatching — joint LASSO Q/L fit  "
              "(shared_flash · both drift volumes)",
              ["bundles = cluster × flash  ·  containment / fiducial prefilter",
               "joint LASSO charge↔light fit  ·  χ² + KS-shape scoring",
               "xTPC crosser pin · consistency ladder · cluster-centric "
               "rescue",
               "photon model: semi-analytical VUV · QtoL 1.0 · low-PE err "
               "inflation"],
              "white", C_QL, tc=C_QL, title_fs=11.6, bullet_fs=8.9, dy=0.295)

    # archive bridge between the two jobs
    bx = (5.55, qy)
    ov.add_patch(FancyBboxPatch(
        (bx[0] - 0.62, bx[1] - 0.30), 1.24, 0.60,
        boxstyle="round,pad=0.02,rounding_size=0.10", linewidth=1.4,
        edgecolor=C_IN, facecolor="#eef1f5", zorder=5))
    ov.text(bx[0], bx[1], "opflash\n*.tar.gz", ha="center", va="center",
            fontsize=8.4, color=C_IN, zorder=6)
    c.arrow(rt(n_lr, 3.95), (bx[0] - 0.62, bx[1]), C_LIGHT)
    c.arrow((bx[0] + 0.62, bx[1]), lf(n_ql, 5.35), C_LIGHT)

    # charge clusters ③ -> QLMatching (down), T0-matched clusters -> ④ (up)
    ov.annotate("", xy=(n_ql[0] - 2.2, qy + 0.83),
                xytext=(n_s3[0] - 0.35, cy - 0.75),
                arrowprops=dict(arrowstyle="-|>", color=C_CLUS, lw=2.4,
                                shrinkA=2, shrinkB=2,
                                connectionstyle="arc3,rad=0.12"))
    ov.text(7.9, 5.82, "clusters, both sides", ha="center", fontsize=8.8,
            color=C_CLUS, style="italic")
    ov.annotate("", xy=(n_s4[0] - 0.35, cy - 0.75),
                xytext=(n_ql[0] + 2.2, qy + 0.83),
                arrowprops=dict(arrowstyle="-|>", color=C_T0, lw=2.6,
                                shrinkA=2, shrinkB=2,
                                connectionstyle="arc3,rad=0.12"))
    ov.text(14.55, 5.62, "cluster T0 → x_t0cor", ha="center", fontsize=9.0,
            color=C_T0, fontweight="bold")

    # mode note (empty mid-gap, under ①/②)
    ov.text(4.55, 5.86, "runner operating point do_qlmatch = on:  ③ → "
            "QLMatching → ④    (no-QL: ③ → ④ direct merge)", ha="center",
            va="center", fontsize=8.4, color="#6a6f78", style="italic")

    # ================= insets =================
    c.place_image(os.path.join(SRC, "qlmatch_pattern.png"), 5.15, 4.30, 0.55,
                  "Q/L light pattern — flash 78 (measured vs semi-analytical "
                  "prediction)", (7.35, 3.60), C_QL, cap_fs=12)
    c.place_image(os.path.join(SRC, "clus_event_3d.png"), 12.75, 3.30, 0.55,
                  "clustering result — clusters coloured  (evt 983)",
                  (n_s4[0], 6.30), C_OUT, cap_fs=12)

    # legend (bottom-left margin, clear of the left inset)
    lx, ly, dy = 0.55, 2.62, 0.315
    for i, (col, lab) in enumerate([
            (C_IN, "input / imaging"), (C_CLUS, "clustering"),
            (C_LIGHT, "light reco"), (C_QL, "Q/L matching"),
            (C_T0, "T0 → x_t0cor")]):
        ov.add_patch(FancyArrowPatch(
            (lx, ly - i * dy), (lx + 0.46, ly - i * dy), arrowstyle="-|>",
            mutation_scale=12, lw=2.3, color=col, zorder=6))
        ov.text(lx + 0.58, ly - i * dy, lab, va="center", fontsize=9.2,
                color=INK)

    ov.text(W / 2, 0.13,
            "Wire-Cell Toolkit  ·  cfg/pgrapher/experiment/pdhd/"
            "{clus,qlmatching,flash}.jsonnet  ·  pdhd/{wct-clustering,"
            "run_clus_evt,run_light_evt}  ·  operating point ON via runner "
            "(jsonnet defaults byte-identical OFF)  ·  insets: run 029107 "
            "evt 983 (hand-scan reference)", ha="center", va="center",
            fontsize=9.0, color="#8a8a8a")

    c.save(OUT)


if __name__ == "__main__":
    main()

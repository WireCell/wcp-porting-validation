#!/usr/bin/env python3
"""Physics insets for the ProtoDUNE-HD clustering + Q/L-matching diagram.

Counterpart of pdvd/pics/make_clus_insets.py (PD-VD original).  Two insets,
both from the canonical PD-HD hand-scan reference event (run 029107 evt 983),
extracted to committed slim npz in clus_chain_src/ (by make_clus_srcdata.py)
so the master build is self-contained:

  clus_event_3d.png   - the clustering OUTPUT: the all-TPC, deghosted,
                        track-separated, T0-corrected 3-D point cloud
                        (`clustering-global`) coloured by cluster id.  Each
                        colour is one reconstructed cluster; this is the object
                        the Q/L matcher then pairs with a flash.
  qlmatch_pattern.png - the core of Q/L matching: measured vs
                        semi-analytical-predicted light pattern across the 160
                        flat X-ARAPUCA photon detectors for the brightest
                        cleanly-matched flash (gid 78).  KS compares the SHAPE
                        (0.032 here = excellent); chi2/ndf = 0.9.  PD-HD PDs:
                        ch 0-79 view drift +x, ch 80-159 view drift -x.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "clus_chain_src")


def make_event_3d(out="clus_event_3d.png", npz="event3d_clusters_983.npz"):
    """clustering-global coloured by cluster id -- the clustering result."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers '3d')
    d = np.load(os.path.join(SRC, npz))
    x, y, z, cid = d["x"], d["y"], d["z"], d["cid"]
    # stable per-cluster colour: rank clusters by size, cycle a 20-colour map
    uniq, counts = np.unique(cid, return_counts=True)
    order = uniq[np.argsort(-counts)]
    rank = {c: i for i, c in enumerate(order)}
    cmap = plt.get_cmap("tab20")
    col = np.array([cmap(rank[c] % 20) for c in cid])

    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(z, x, y, c=col, s=0.30, linewidths=0, depthshade=False,
               rasterized=True)
    ax.set_xlabel("Z  [cm]", fontsize=10, labelpad=2)
    ax.set_ylabel("X  drift  [cm]", fontsize=10, labelpad=2)
    ax.set_zlabel("Y  [cm]", fontsize=10, labelpad=2)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=18, azim=-70)
    fig.tight_layout()
    fig.savefig(os.path.join(SRC, out), dpi=200)
    plt.close(fig)
    print("wrote", out, "npts", len(x),
          "nclus", len(uniq), "run", int(d["run"]), "evt", int(d["event"]))


def make_ql_pattern(out="qlmatch_pattern.png", npz="qlmatch_flash78.npz"):
    """measured vs predicted light pattern over the 160 PDs for one flash."""
    d = np.load(os.path.join(SRC, npz))
    ch = d["ch"]
    meas = np.clip(d["meas"].astype(float), 0.5, None)
    pred = np.clip(d["pred"].astype(float), 0.5, None)
    masked = d["masked"].astype(bool)
    n = len(ch)
    xi = np.arange(n)

    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    w = 0.42
    ax.bar(xi - w / 2, meas, width=w, color="#2b6f9e", label="measured PE",
           zorder=3)
    ax.bar(xi + w / 2, pred, width=w, color="none", edgecolor="#c0392b",
           linewidth=0.9, label="predicted (semi-analytical)", zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(meas.max(), pred.max()) * 2.6)

    # drift-side split: ch 0-79 view +x, ch 80-159 view -x
    ax.axvline(79.5, color="0.45", lw=1.0, ls="--", zorder=2)
    ax.text(0.135, 0.92, "+x drift side (ch 0–79)", ha="center",
            fontsize=8.5, color="#5a6572", fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.75, 0.62, "−x drift side (ch 80–159)", ha="center",
            fontsize=8.5, color="#5a6572", fontweight="bold",
            transform=ax.transAxes)
    for i in np.where(masked)[0]:
        ax.plot(i, 0.62, marker="s", ms=3.0, color="#9aa0a6", zorder=6)

    ax.set_xlabel("photon-detector channel (0-159, flat X-ARAPUCA)",
                  fontsize=10)
    ax.set_ylabel("PE  (log)", fontsize=10)
    ax.set_xlim(-1, n)
    ax.tick_params(labelsize=8)
    ax.set_title(
        "flash %d  —  brightest matched flash   "
        "(KS=%.3f shape ✓,  χ²/ndf=%.1f)"
        % (int(d["gid"]), float(d["ks"]),
           float(d["chi2"]) / max(1, int(d["ndf"]))),
        fontsize=9.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(fc="#2b6f9e", label="measured PE"),
        Patch(fc="none", ec="#c0392b", label="predicted (semi-analytical)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#9aa0a6",
               markersize=6, label="masked channel"),
    ], fontsize=7.3, loc="upper right", framealpha=0.93, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(SRC, out), dpi=200)
    plt.close(fig)
    print("wrote", out, "totPE %.0f" % float(d["total_PE"]),
          "nmasked", int(masked.sum()))


def main():
    make_event_3d()
    make_ql_pattern()


if __name__ == "__main__":
    main()

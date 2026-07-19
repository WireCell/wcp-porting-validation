#!/usr/bin/env python3
"""Physics insets for the ProtoDUNE-VD clustering + Q/L-matching diagram.

Two insets, both from the canonical `keep`-round dump of run 039252 evt 298567
(the hand-scan reference event), extracted to committed slim npz in
clus_chain_src/ so the master build is self-contained:

  clus_event_3d.png   - the clustering OUTPUT: the all-TPC, deghosted,
                        track-separated, T0-corrected 3-D point cloud
                        (`clustering-global`) coloured by cluster id.  Each
                        colour is one reconstructed cluster; this is the object
                        the Q/L matcher then pairs with a flash.
  qlmatch_pattern.png - the core of Q/L matching: measured vs library-predicted
                        light pattern across the 40 photon detectors for one
                        bright cathode-crossing flash (gid 57).  KS compares the
                        SHAPE (0.034 here = excellent); chi2/ndf is inflated by
                        DAPHNE saturation on the brightest cathode channels
                        (kept-and-marked, not vetoed).

Sources (clus_chain_src/):
  event3d_clusters_298567.npz  - x,y,z,q,cid  (|x|<400 cm; unmatched-cluster
                                 sentinel points at x_t0cor=+/-1.48e8 dropped)
  qlmatch_flash57.npz          - per-PD ch,type,meas,pred,perr,sat,cov + scalars
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "clus_chain_src")


def make_event_3d(out="clus_event_3d.png", npz="event3d_clusters_298567.npz"):
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


def make_ql_pattern(out="qlmatch_pattern.png", npz="qlmatch_flash57.npz"):
    """measured vs predicted light pattern over the 40 PDs for one flash."""
    d = np.load(os.path.join(SRC, npz), allow_pickle=True)
    ch = d["ch"]
    meas = np.clip(d["meas"].astype(float), 0.5, None)
    pred = np.clip(d["pred"].astype(float), 0.5, None)
    sat = d["sat"].astype(float) > 0
    cov = d["cov"].astype(float) < 1
    n = len(ch)
    xi = np.arange(n)

    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    w = 0.40
    ax.bar(xi - w / 2, meas, width=w, color="#2b6f9e", label="measured PE",
           zorder=3)
    ax.bar(xi + w / 2, pred, width=w, color="none", edgecolor="#c0392b",
           linewidth=1.3, label="predicted (library)", zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(meas.max(), pred.max()) * 2.2)

    # shade the cathode-XArapuca group (ch 4-11) -- collects ~89% of the light
    ax.axvspan(3.5, 11.5, color="#f2c94c", alpha=0.16, zorder=0)
    ax.text(7.5, ax.get_ylim()[1] * 0.55, "cathode XA", ha="center",
            fontsize=8.5, color="#8a6d1a", fontweight="bold")

    top = ax.get_ylim()[1]
    for i in np.where(sat)[0]:
        ax.text(i, top * 0.80, "★", ha="center", va="center", fontsize=9,
                color="#c0392b", zorder=6)
    for i in np.where(cov)[0]:
        ax.plot(i, 0.62, marker="s", ms=3.2, color="#9aa0a6", zorder=6)

    ax.set_xlabel("photon-detector channel (0-39)", fontsize=10)
    ax.set_ylabel("PE  (log)", fontsize=10)
    ax.set_xlim(-1, n)
    ax.tick_params(labelsize=8)
    ax.set_title(
        "flash 57  —  cathode crosser, T0-pinned   "
        "(KS=%.3f shape ✓,  χ²/ndf=%.0f  sat-inflated)"
        % (float(d["ks"]), float(d["chi2"]) / float(d["ndf"])),
        fontsize=9.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(fc="#2b6f9e", label="measured PE"),
        Patch(fc="none", ec="#c0392b", label="predicted (library)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#c0392b",
               markersize=9, label="saturated (kept+marked)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#9aa0a6",
               markersize=6, label="no coverage"),
    ], fontsize=7.3, loc="upper right", framealpha=0.93, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(SRC, out), dpi=200)
    plt.close(fig)
    print("wrote", out, "totPE %.0f" % float(d["total_PE"]),
          "nsat", int(sat.sum()), "ncov<1", int(cov.sum()))


def main():
    make_event_3d()
    make_ql_pattern()


if __name__ == "__main__":
    main()

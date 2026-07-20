#!/usr/bin/env python3
"""Physics insets for the ProtoDUNE-VD 3-D imaging + deghosting diagram.

Counterpart of pdhd/pics/make_imaging_insets.py.  Two insets:

  raygrid_fig8_ghost.png - the ghost concept, illustrated with Fig. 8 of the
                         Wire-Cell imaging note (B. Viren, "Wire-Cell Toolkit
                         Imaging", 2019): a *toy* 3-plane detector where blobs
                         tiled at strip triple-overlaps either surround the
                         generated points (real) or none (ghosts).  Detector-
                         agnostic, so PDHD and PDVD share it.  Two leaders
                         (real / ghost) overlaid for slide legibility.  The
                         source page is committed as raygrid_fig8_page.png
                         (page 11 plot region, `pdftoppm -r 300`; see the doc
                         repro block); nothing of the original is altered.
  img_event_3d.png     - a full-event 3-D charge display (X drift, Y, Z)
                         coloured by charge -- a real Bee event, the deghosted
                         imaging result.  Built from a committed slim npz
                         extracted from our own mabc-all-apa.zip (PDVD data run
                         039252 evt 298567, the shared hand-scan reference
                         event), i.e. exactly the imaging charge cloud uploaded
                         to Bee -- "only charge, no clustering colour".
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "imaging_src")


def make_ghost_concept(out="raygrid_fig8_ghost.png",
                       page="raygrid_fig8_page.png"):
    """Fig. 8 of the WCT imaging note with real/ghost leaders overlaid."""
    im = np.asarray(Image.open(os.path.join(OUT, page)).convert("RGB"))
    H, W = im.shape[:2]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])

    def frac(fx, fy):
        return fx * W, (1 - fy) * H

    # ghost: an empty grey blob (no points inside), centre-right triangle
    ax.annotate("ghost blob\n(no points inside)", xy=frac(0.63, 0.55),
                xytext=(0.60 * W, 0.30 * H), fontsize=13, fontweight="bold",
                color="#b00020", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#b00020", lw=2.4),
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#b00020",
                          alpha=0.94))
    # real: a point-filled blob, top-left parallelogram
    ax.annotate("real blob\n(points inside)", xy=frac(0.30, 0.72),
                xytext=(0.14 * W, 0.40 * H), fontsize=13, fontweight="bold",
                color="#0d3b66", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#0d3b66", lw=2.4),
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#0d3b66",
                          alpha=0.94))
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, out), dpi=200)
    plt.close(fig)
    print("wrote", out)


def make_event_3d(out="img_event_3d.png", npz="event3d_039252_298567.npz"):
    """A full-event 3-D charge display (a real Bee event) from our own
    mabc-all-apa.zip img cloud, coloured by charge only."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')
    d = np.load(os.path.join(OUT, npz))
    x, y, z, q = d["x"], d["y"], d["z"], np.clip(d["q"], 1, None)
    if len(x) > 110000:
        i = np.random.default_rng(0).choice(len(x), 110000, replace=False)
        x, y, z, q = x[i], y[i], z[i], q[i]

    fig = plt.figure(figsize=(6.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(z, x, y, c=q, s=0.28, cmap="viridis",
                    norm=LogNorm(vmin=200, vmax=np.percentile(q, 99.5)),
                    linewidths=0, depthshade=False, rasterized=True)
    ax.set_xlabel("Z  [cm]", fontsize=10, labelpad=2)
    ax.set_ylabel("X  drift  [cm]", fontsize=10, labelpad=2)
    ax.set_zlabel("Y  [cm]", fontsize=10, labelpad=2)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=20, azim=-68)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.028, shrink=0.62)
    cb.set_label("charge", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=200)
    plt.close(fig)
    print("wrote", out, "npts", len(x),
          "run", int(d["run"]), "evt", int(d["event"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    make_ghost_concept()
    make_event_3d()


if __name__ == "__main__":
    main()

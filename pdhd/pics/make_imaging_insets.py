#!/usr/bin/env python3
"""Physics insets for the ProtoDUNE-HD 3-D imaging + deghosting diagram.

All insets are built from the preprocessed imaging cache the img_plot viewer
uses (`pdhd/img_plot/cache/evt0.npz`, real ProtoDUNE-HD data run 27305 evt 150).
That cache carries, per imaging blob: its Z-Y polygon (`blob_poly_*`), the fired
wire cells that formed it (`band_quad_yz`, keyed by `band_blob`), the solved
charge (`blob_val`; a kept `val==0` blob is a flagged zero-charge ghost), and the
downstream stepped sample points (`pts_*`).

Outputs (into imaging_src/):
  img_slice_blobs.png  - one time slice, zoomed to a busy blob cluster, Z-Y:
                         fired U/V/W wire strips + the blobs tiled at their
                         triple-overlaps.  Solid navy = charge-solved real blobs;
                         dashed magenta = surviving zero-charge ghost blobs (false
                         triple-coincidences the deghosting flagged).
  img_event_zy.png     - the whole event, Z-Y projection of the sampled blob
                         points coloured by charge (x-conversion independent) --
                         the deghosted, charge-solved imaging result.

(An X-Z / drift projection is deliberately NOT made from this cache: two of the
four APAs have a broken slice-time->x calibration on this event -- sidecar
residuals 382 / 252 cm vs 0.3 / 0.9 cm -- which smears the drift axis.  The
diagram draws the drift ("3-D") build-up as a clean synthetic schematic instead.)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "imaging_src")
CACHE = os.path.join(HERE, "..", "img_plot", "cache", "evt0.npz")

# fired-wire strip colours per plane (U,V,W)
PLANE_C = ["#e08a1e", "#1f9e6a", "#2f6fd0"]


def _load():
    return np.load(CACHE, allow_pickle=True)


def blob_polys(d):
    off, xy = d["blob_poly_off"], d["blob_poly_xy"]
    # xy columns are (y, z); return (z, y) so z is horizontal
    return [xy[off[i]:off[i + 1]][:, ::-1] for i in range(len(off) - 1)]


def make_slice_blobs(d, anode=1, face=1, sl=278,
                     zwin=(24, 58), ywin=(212, 262), out="img_slice_blobs.png"):
    """One time slice zoomed to a busy blob cluster: the tomographic-ambiguity
    picture where real (solid) and flagged ghost (dashed) blobs pack together."""
    ba, bf, bs, bv = d["blob_anode"], d["blob_face"], d["blob_sliceid"], d["blob_val"]
    sel = np.where((ba == anode) & (bf == face) & (bs == sl))[0]
    polys = blob_polys(d)

    # fired-wire cells that built these blobs
    bb, bq, bpl = d["band_blob"], d["band_quad_yz"], d["band_plane"]
    bmask = np.isin(bb, sel)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    # 1) fired wire strips (faint, per plane) -- the tomographic projections
    for pl in range(3):
        m = bmask & (bpl == pl)
        quads = bq[m][:, :, ::-1]  # (y,z)->(z,y)
        if len(quads):
            ax.add_collection(PolyCollection(
                quads, facecolors=PLANE_C[pl], edgecolors=PLANE_C[pl],
                alpha=0.13, linewidths=0.3, zorder=1))
    # 2) blobs: solid navy = real (charge>0), dashed magenta = zero-charge ghost
    real = [polys[i] for i in sel if bv[i] != 0]
    ghost = [polys[i] for i in sel if bv[i] == 0]
    if real:
        ax.add_collection(PolyCollection(
            real, facecolors="#2b3a67", edgecolors="#0d1836",
            alpha=0.9, linewidths=1.2, zorder=4))
    ax.add_collection(PolyCollection(
        ghost, facecolors="none", edgecolors="#e01ab0",
        linestyle=(0, (3, 1.5)), linewidths=1.4, zorder=5))

    ax.set_xlim(*zwin)
    ax.set_ylim(*ywin)
    ax.set_xlabel("Z  [cm]", fontsize=11)
    ax.set_ylabel("Y  [cm]", fontsize=11)
    ax.set_title("one slice, zoom — U/V/W strips cross → blobs + ghosts",
                 fontsize=10.5)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(fc=PLANE_C[0], alpha=0.5, label="U strip"),
        Patch(fc=PLANE_C[1], alpha=0.5, label="V strip"),
        Patch(fc=PLANE_C[2], alpha=0.5, label="W strip"),
        Patch(fc="#2b3a67", label="real blob"),
        Line2D([0], [0], color="#e01ab0", ls=(0, (3, 1.5)), lw=1.6,
               label="ghost blob (val=0)"),
    ], fontsize=7.5, loc="upper right", framealpha=0.92, ncol=1)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=200)
    plt.close(fig)
    print("wrote", out, "real", len(real), "ghost", len(ghost))


def make_event_zy(d, out="img_event_zy.png"):
    x, y, z, q = d["pts_x"], d["pts_y"], d["pts_z"], d["pts_q"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    qc = np.clip(q, 1, None)
    sc = ax.scatter(z, y, c=qc, s=0.4, cmap="viridis",
                    norm=LogNorm(vmin=200, vmax=np.percentile(qc, 99.5)),
                    linewidths=0, rasterized=True)
    ax.set_xlabel("Z  [cm]", fontsize=11)
    ax.set_ylabel("Y  [cm]", fontsize=11)
    ax.set_title("full event — sampled blob points, Z-Y (colour = charge)",
                 fontsize=10.5)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.046)
    cb.set_label("blob-point charge", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=200)
    plt.close(fig)
    print("wrote", out, "npts", len(x))


def main():
    os.makedirs(OUT, exist_ok=True)
    d = _load()
    make_slice_blobs(d)
    make_event_zy(d)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pdvd/115 round 2 sec 7.4 -- one cluster's image, Steiner vertices and persisted fit rows under several arms, in
the detector frame ((x, y) and (x, z) panels), for a stretch the doc-114 case tool cannot window (a fit that runs far
from the image).  Read-only.  Also prints, per arm, the fit rows farther than D_FAR from every image point of the
cluster and the Steiner vertices within that region.

Usage: d115r2_cluster_fig.py --det pdvd --event 039349_61 --cluster 67 --arms d115voff:production,d115vp3bwp025:alpha0.25,d115vp3bwp05:alpha0.5 \
           --out figs/115r2_case_pdvd_FIT_1_custom [--xr 0 100]
"""
import argparse, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import d111s_common as C
A = C.A
D_FAR = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--event", required=True); ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--arms", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--xr", type=float, nargs=2, help="x range of the zoom panels")
    a = ap.parse_args()
    arms = [t.split(":") for t in a.arms.split(",")]
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    L = []
    for k, (arm, lab) in enumerate(arms):
        d = f"{C.IMG}/{a.det}/work/{a.event}_{arm}"
        img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
        mi = img[1] == a.cluster
        P0 = img[0][mi]
        st = A.bee_layer(f"{d}/mabc-pr.zip", "steiner_graph-global")
        S0 = st[0][st[1] == a.cluster] if st is not None else np.zeros((0, 3))
        t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["x", "y", "z", "cluster_id", "pass", "rr"], library="np")
        m = t["cluster_id"] == a.cluster
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        kd = cKDTree(P0) if len(P0) else None
        dimg = kd.query(F)[0] if kd is not None and len(F) else np.full(len(F), np.nan)
        far = dimg > D_FAR
        ks = cKDTree(S0) if len(S0) else None
        dst = ks.query(F)[0] if ks is not None and len(F) else np.full(len(F), np.nan)
        L.append(f"{arm} ({lab}): image points {len(P0)}, Steiner vertices {len(S0)}, fit rows {len(F)}, rows > {D_FAR} cm from every image point {int(far.sum())}"
                 + (f" (x {F[far, 0].min():.1f}..{F[far, 0].max():.1f}, y {F[far, 1].min():.1f}..{F[far, 1].max():.1f}, z {F[far, 2].min():.1f}..{F[far, 2].max():.1f}; "
                    f"nearest image {np.median(dimg[far]):.1f} cm median, nearest Steiner vertex {np.median(dst[far]):.1f} cm median)" if far.any() else ""))
        if k == 0:
            for ax, (i, j) in zip(axs[:, 0], ((0, 1), (0, 2))):
                ax.scatter(P0[:, i], P0[:, j], s=2, c="lightgrey", label="image (base arm)")
            for ax, (i, j) in zip(axs[:, 1], ((0, 1), (0, 2))):
                ax.scatter(P0[:, i], P0[:, j], s=4, c="lightgrey", label="image (base arm)")
        col = ["black", "tab:blue", "tab:red", "tab:green", "tab:orange"][k]
        for ax, (i, j) in zip(axs[:, 0], ((0, 1), (0, 2))):
            ax.plot(F[:, i], F[:, j], "-", lw=0.8, c=col, label=f"fit {lab} ({len(F)} rows)")
        for ax, (i, j) in zip(axs[:, 1], ((0, 1), (0, 2))):
            ax.plot(F[:, i], F[:, j], ".-", ms=3, lw=0.8, c=col, label=f"fit {lab}")
            if len(S0):
                ax.scatter(S0[:, i], S0[:, j], s=6, marker="x", c=col, alpha=0.5, label=f"Steiner vertices {lab}")
    for ax, nm in zip(axs[:, 0], ("y", "z")):
        ax.set_xlabel("x (cm, drift)"); ax.set_ylabel(f"{nm} (cm)"); ax.legend(fontsize=7, loc="best")
    for ax, nm in zip(axs[:, 1], ("y", "z")):
        ax.set_xlabel("x (cm, drift)"); ax.set_ylabel(f"{nm} (cm)"); ax.legend(fontsize=7, loc="best")
        if a.xr:
            ax.set_xlim(*a.xr)
            # zoom the ordinate to the fit rows inside the x range
            ys = []
            for line in ax.get_lines():
                xd, yd = line.get_xdata(), line.get_ydata()
                sel = (np.asarray(xd) >= a.xr[0]) & (np.asarray(xd) <= a.xr[1])
                if sel.any():
                    ys.extend(np.asarray(yd)[sel].tolist())
            if ys:
                ax.set_ylim(min(ys) - 15, max(ys) + 15)
    axs[0, 0].set_title(f"{a.det} {a.event} cluster {a.cluster}: image, Steiner vertices and the persisted STM fit per arm")
    axs[0, 1].set_title("zoom" + (f" x {a.xr[0]:g}..{a.xr[1]:g}" if a.xr else ""))
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=110)
    open(a.out + ".txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

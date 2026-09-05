#!/usr/bin/env python3
"""doc pdvd/42 -- Magnify-style 2-D panels for one (cluster, pass) block of a
tracking-stm.root: per plane the measured charge, the predicted charge and the
fractional residual (yhat - y)/y, with the fitted trajectory overlaid -- the
Magnify-tracking pad layout (Data::DrawProj) rendered with matplotlib and
zoomed to the block, so the doc can show several blocks side by side without
an X server.  The measured/residual colour scales follow the GUI
(measured 500..20000 e, |residual| 0.01..1).

Usage:
  d42_proj2d_panels.py --det pdvd --block 1090 -o figs/42_panel_pdvd_298595_c109.png work/039252_2_d42fit/tracking-stm.root
"""
import argparse, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PLANES = {"pdvd": (0, 3808, 7616, 12288), "sbnd": (0, 3968, 7936, 11276)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--det", required=True, choices=PLANES)
    ap.add_argument("--block", type=int, required=True); ap.add_argument("-o", required=True)
    ap.add_argument("--pad", type=int, default=6, help="channels/slices of margin around the block")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    import uproot
    f = uproot.open(a.root)
    d = f["T_proj_data"].arrays(library="np")
    blocks = [int(c) for c in d["cluster_id"][0]]
    i = blocks.index(a.block)
    ch = np.asarray(list(d["channel"][0][i]), dtype=np.int64); ts = np.asarray(list(d["time_slice"][0][i]), dtype=np.int64)
    q = np.asarray(list(d["charge"][0][i]), dtype=float); qp = np.asarray(list(d["charge_pred"][0][i]), dtype=float)
    r = f["T_rec_charge"].arrays(["pu", "pv", "pw", "pt", "ndf", "status", "rr", "q", "nq"], library="np")
    m = r["ndf"] == a.block
    st = int(r["status"][m][0]) if m.sum() else -1
    edges = PLANES[a.det]
    fig, ax = plt.subplots(3, 3, figsize=(15, 12))
    for P in range(3):
        mp = (ch >= edges[P]) & (ch < edges[P + 1])
        proj = r[("pu", "pv", "pw")[P]][m]
        if mp.sum() == 0:
            for k_ in range(3): ax[k_, P].set_axis_off()
            continue
        # zoom to the TRAJECTORY's footprint (a cluster's own cells can sit far
        # from it -- another CRP, a fused branch; that charge is reported as the
        # out-of-view fraction in the title, and measured by f_off in
        # d42_proj2d_resid.py)
        if len(proj) == 0:
            for k_ in range(3): ax[k_, P].set_axis_off()
            continue
        c0 = max(int(proj.min()) - a.pad, edges[P]); c1 = min(int(proj.max()) + a.pad, edges[P + 1] - 1)
        t0 = max(int(r["pt"][m].min()) - a.pad, 0); t1 = int(r["pt"][m].max()) + a.pad
        nc, nt = int(c1 - c0 + 1), int(t1 - t0 + 1)
        H = np.full((nt, nc), np.nan); Hp = np.full((nt, nc), np.nan); R = np.full((nt, nc), np.nan)
        cc = ch[mp] - c0; tt = ts[mp] - t0
        ok = (cc >= 0) & (cc < nc) & (tt >= 0) & (tt < nt)
        H[tt[ok], cc[ok]] = q[mp][ok]; Hp[tt[ok], cc[ok]] = qp[mp][ok]
        with np.errstate(divide="ignore", invalid="ignore"):
            R[tt[ok], cc[ok]] = (qp[mp][ok] - q[mp][ok] + 0.01) / (q[mp][ok] + 0.01)
        ext = (c0 - 0.5, c1 + 0.5, t0 - 0.5, t1 + 0.5)
        im0 = ax[0, P].imshow(np.where(H > 0, H, np.nan), origin="lower", extent=ext, aspect="auto", cmap="viridis", norm=LogNorm(500, 2e4), interpolation="nearest")
        im1 = ax[1, P].imshow(np.where(Hp > 0, Hp, np.nan), origin="lower", extent=ext, aspect="auto", cmap="viridis", norm=LogNorm(500, 2e4), interpolation="nearest")
        im2 = ax[2, P].imshow(R, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        for k_ in range(3):
            ax[k_, P].plot(proj, r["pt"][m], "r.", ms=2 if k_ else 1.5, alpha=0.7)
            ax[k_, P].set_xlabel("channel (plane-concatenated)"); ax[k_, P].set_ylabel("time slice")
        Q = np.nansum(H); Qp = np.nansum(Hp); Qtot = q[mp].sum()
        ax[0, P].set_title("%s plane: measured charge in view %.2e e (%.0f%% of the cluster's %s-plane charge)" % ("UVW"[P], Q, 100 * Q / Qtot if Qtot else np.nan, "UVW"[P]), fontsize=8)
        ax[1, P].set_title("predicted charge (sum %.2e e, %.0f%% of measured)" % (Qp, 100 * Qp / Q if Q else np.nan), fontsize=9)
        u = np.nansum(np.abs(Hp - H)) / Q if Q else np.nan
        ax[2, P].set_title("(pred - meas)/meas, U = sum|y-yhat|/sum y = %.3f" % u, fontsize=9)
        fig.colorbar(im0, ax=ax[0, P], fraction=0.04); fig.colorbar(im1, ax=ax[1, P], fraction=0.04); fig.colorbar(im2, ax=ax[2, P], fraction=0.04)
    fig.suptitle("%s %s block %d (cluster %d pass %d, status %d, %d fit points, %.0f cm)  -- red: fitted trajectory" % (
        a.det.upper(), a.title, a.block, a.block // 10, a.block % 10, st, int(m.sum()), float(r["rr"][m].max()) if m.sum() else 0), fontsize=11)
    fig.tight_layout(); fig.savefig(a.o, dpi=100); plt.close(fig)
    print("wrote", a.o)


if __name__ == "__main__":
    sys.exit(main())

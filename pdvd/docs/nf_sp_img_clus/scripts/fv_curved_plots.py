#!/usr/bin/env python3
"""Doc pdvd/41 -- stage 3: figures from fv_curved_map.py's JSON (+ the point cache for
the occupancy maps).

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_plots.py /home/xqian/tmp/doc41/map20_result.json \
      --npz /home/xqian/tmp/doc41/points_d28dlfp.npz --figdir docs/nf_sp_img_clus/figs --prefix 41
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WALLS = ["y+", "y-", "z-", "z+"]
COL = {"y+": "#1f77b4", "y-": "#ff7f0e", "z-": "#2ca02c", "z+": "#d62728"}
VOL = ((0, "bottom (x<0)", "o", -1), (1, "top (x>0)", "s", +1))


def m1(xabs, dc, xk, cath=3.0):
    return dc * np.clip((xk - xabs) / (xk - cath), 0, 1)


def m2(xabs, dc, p, xw=339.91, cath=3.0):
    return dc * np.clip((xw - xabs) / (xw - cath), 0, 1) ** p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--npz")
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--prefix", default="41")
    a = ap.parse_args()
    R = json.load(open(a.json))
    xc = np.array(R["xcenter"]); G = R["geometry"]
    os.makedirs(a.figdir, exist_ok=True)
    xx = np.linspace(3, 339.9, 300)

    # ---- fig 1: d50(x) per wall, both volumes, + endpoint modes, + per-volume M1 fits
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, w in zip(axes.ravel(), WALLS):
        P = R["walls"][w]["profile"]; E = R["walls"][w]["endpoint"]
        for v, lab, mk, sgn in VOL:
            ax.errorbar(sgn * xc, P["d50"][v], yerr=P["d50_err"][v], fmt=mk, ms=4, color=COL[w],
                        alpha=1.0 if v else 0.55, label=f"half-density point d50, {lab}")
            ax.errorbar(sgn * xc, E["mode"][v], yerr=E["err"][v], fmt="x", ms=5, color="k", alpha=0.5,
                        label="through-going track ends (mode)" if v == 0 else None)
            M = R["walls"][w]["models_bot_d50" if v == 0 else "models_top_d50"]
            if "params" in M.get("M1", {}):
                ax.plot(sgn * xx, m1(xx, *M["M1"]["params"]), "-", color="gray", lw=1.5,
                        label="M1 flat + linear ramp (per volume)" if v == 0 else None)
            if "params" in M.get("M2", {}):
                ax.plot(sgn * xx, m2(xx, *M["M2"]["params"]), "--", color="purple", lw=1.2,
                        label="M2 power law (per volume)" if v == 0 else None)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.axhline(15, color="brown", lw=0.8, ls="-.", label="the flat 15 cm inset in use")
        ax.set_title(f"wall {w}"); ax.grid(alpha=0.3); ax.set_ylim(-6, 30)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("drift x (t0-corrected, cm); cathode at 0, anodes at ±339.9")
    for ax in axes[:, 0]:
        ax.set_ylabel("apparent wall inset (cm)")
    fig.suptitle("PDVD apparent side-wall inset vs drift -- 120 cosmic data events (d28dlfp), 20 cm bins")
    fig.tight_layout(); fig.savefig(os.path.join(a.figdir, f"{a.prefix}_edge_vs_x.png"), dpi=110); plt.close(fig)

    # ---- fig 2: factorization slices (d50, per volume)
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True, sharey=True)
    for i, w in enumerate(WALLS):
        P = R["walls"][w]["profile"]
        for v, lab, mk, sgn in VOL:
            ax = axes[i, v]
            ax.errorbar(xc, P["d50"][v], yerr=P["d50_err"][v], fmt="o", color="k", ms=4, label="all")
            for sname, S in R["walls"][w]["slices"].items():
                ax.errorbar(xc, S["d50"][v], yerr=S["d50_err"][v], fmt="-", lw=1.2, capsize=2, alpha=0.8,
                            label=f"{sname}  (chi2/ndf vs all, both vols: {S['chi2_vs_all_d50']:.0f}/{S['ndf_d50']})")
            ax.set_title(f"wall {w}, {lab}: d50 vs |x| in slices of the other coordinate", fontsize=9)
            ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_ylim(-6, 30)
    for ax in axes[3]:
        ax.set_xlabel("|x| (cm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("inset (cm)")
    fig.tight_layout(); fig.savefig(os.path.join(a.figdir, f"{a.prefix}_factorization.png"), dpi=100); plt.close(fig)

    # ---- fig 3: straight-line extrapolation of near-wall tracks (displacement vs loss)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, w in zip(axes.ravel(), WALLS):
        B = R["walls"][w]["bend"]; P = R["walls"][w]["profile"]
        for v, lab, mk, sgn in VOL:
            for k, (lo, hi) in enumerate(B["bands"]):
                if k == 2:
                    continue
                r = np.array(B["resid"][v])[:, k]; e = np.array(B["resid_err"][v])[:, k]
                nt = np.array(B["n_total"][v])[:, k]; nh = np.array(B["n_has"][v])[:, k]
                ax.errorbar(sgn * xc, r, yerr=e, fmt=mk, ms=4, color=COL[w] if k == 1 else "gray",
                            alpha=1.0 if v else 0.55,
                            label=f"line predicts {lo:.0f}-{hi:.0f} cm from the wall, {lab}: survive {nh.sum()}/{nt.sum()}")
            ax.plot(sgn * xc, P["d50"][v], ":", color="k", alpha=0.6, label="d50 (density)" if v == 0 else None)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"wall {w}: residual of near-wall tracks from their anode-side straight line", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=6.5); ax.set_ylim(-10, 30)
    for ax in axes[1]:
        ax.set_xlabel("x (cm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("measured minus predicted wall distance (cm); + = pushed inward")
    fig.tight_layout(); fig.savefig(os.path.join(a.figdir, f"{a.prefix}_bending.png"), dpi=110); plt.close(fig)

    # ---- fig 5: the suggested surface (M1 polygons per drift volume) in detector
    #             coordinates, overlaid with the measured d50 points
    if "polygons_M1_d50" in R:
        import matplotlib.gridspec as gridspec
        POLY = R["polygons_M1_d50"]
        XW, YW, ZLO, ZHI = G["XW"], G["YW"], G["ZLO"], G["ZHI"]
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1.6, 1, 1], hspace=0.32, wspace=0.18)

        def draw_overview(ax, plane):
            key = "boundary_xy" if plane == "xy" else "boundary_xz"
            lo, hi = (-YW, YW) if plane == "xy" else (ZLO, ZHI)
            ax.add_patch(plt.Rectangle((-XW, lo), 2 * XW, hi - lo, fill=False, ec="k", lw=1.0, label="sensvol envelope"))
            ax.add_patch(plt.Rectangle((-XW, lo + 15), 2 * XW, hi - lo - 30, fill=False, ec="brown", lw=1.0, ls="-.",
                                       label="flat 15 cm inset in use"))
            for vn, col in (("bottom_x_lt_0", "tab:blue"), ("top_x_gt_0", "tab:red")):
                V = np.array(POLY[vn][key])
                ax.add_patch(plt.Polygon(V, closed=True, fill=True, fc=col, alpha=0.12, ec=col, lw=2,
                                         label=f"suggested surface, {vn.replace('_', ' ')}"))
            ax.axvspan(-3, 3, color="gray", alpha=0.4, lw=0)
            ax.set_xlim(-360, 360); ax.set_ylim(lo - 25, hi + 25); ax.set_aspect("equal")
            ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)" if plane == "xy" else "z (cm)")
            ax.set_title(f"{'X-Y' if plane == 'xy' else 'X-Z'} plane, true scale (cathode slab shaded)")
            ax.legend(fontsize=7, loc="upper center", ncol=2)

        draw_overview(fig.add_subplot(gs[0, 0]), "xy")
        draw_overview(fig.add_subplot(gs[0, 1]), "xz")

        strips = [("y+", gs[1, 0], lambda d: YW - d, YW, -1), ("z+", gs[1, 1], lambda d: ZHI - d, ZHI, -1),
                  ("y-", gs[2, 0], lambda d: -YW + d, -YW, +1), ("z-", gs[2, 1], lambda d: ZLO + d, ZLO, +1)]
        for w, cell, to_abs, wall, sgn in strips:
            ax = fig.add_subplot(cell)
            P = R["walls"][w]["profile"]
            for v, lab, mk, s_ in VOL:
                d50, e = np.array(P["d50"][v]), np.array(P["d50_err"][v])
                ax.errorbar(s_ * xc, to_abs(d50), yerr=e, fmt=mk, ms=4, color=COL[w], alpha=1.0 if v else 0.55,
                            label=f"measured half-density point, {lab}")
            for vn, col in (("bottom_x_lt_0", "tab:blue"), ("top_x_gt_0", "tab:red")):
                key = "boundary_xy" if w[0] == "y" else "boundary_xz"
                V = np.array(POLY[vn][key])
                side = V[V[:, 1] * (1 if w[1] == "+" else -1) > (0 if w[0] == "y" else (ZLO + ZHI) / 2 * (1 if w[1] == "+" else -1))]
                side = side[np.argsort(side[:, 0])]
                ax.plot(side[:, 0], side[:, 1], "-", color=col, lw=2.2, label=f"suggested surface (M1), {vn.replace('_', ' ')}")
            ax.axhline(wall, color="k", lw=1.0, label="nominal wall (sensvol)")
            ax.axhline(wall + sgn * 15, color="brown", lw=1.0, ls="-.", label="flat 15 cm inset in use")
            ax.axvspan(-3, 3, color="gray", alpha=0.4, lw=0)
            lo, hi = sorted((wall - sgn * 6, wall + sgn * 30))
            ax.set_ylim(lo, hi); ax.set_xlim(-345, 345); ax.grid(alpha=0.3)
            ax.set_title(f"wall {w}: zoom on the last 30 cm", fontsize=10)
            ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)" if w[0] == "y" else "z (cm)")
            if w == "y+":
                ax.legend(fontsize=6.5, loc="lower center", ncol=2)
        fig.suptitle("PDVD suggested curved fiducial surface (M1 ramps on d50, per drift volume) over the measurement", y=0.995)
        fig.savefig(os.path.join(a.figdir, f"{a.prefix}_surface_overlay.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- fig 4: occupancy maps near each wall with d50 overlaid
    if a.npz:
        d = np.load(a.npz, allow_pickle=True)
        ph = d["phys"]; x, y, z = d["x"][ph], d["y"][ph], d["z"][ph]
        ok = np.abs(x) < 345
        x, y, z = x[ok], y[ok], z[ok]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        spec = {"y+": (G["YW"] - y, (z > 15) & (z < 283)), "y-": (G["YW"] + y, (z > 15) & (z < 283)),
                "z-": (z - G["ZLO"], (np.abs(y) < 321) & (np.abs(y) > 3)), "z+": (G["ZHI"] - z, (np.abs(y) < 321) & (np.abs(y) > 3))}
        for ax, w in zip(axes.ravel(), WALLS):
            dd, sel = spec[w]
            m = sel & (dd > -5) & (dd < 40)
            H, xe, ye = np.histogram2d(x[m], dd[m], bins=[np.arange(-345, 346, 5), np.arange(-5, 40.5, 1.0)])
            ax.pcolormesh(xe, ye, np.log10(H.T + 1), cmap="viridis")
            P = R["walls"][w]["profile"]
            for v, lab, mk, sgn in VOL:
                ax.errorbar(sgn * xc, P["d50"][v], yerr=P["d50_err"][v], fmt="o", ms=3, color="w", mec="r", ecolor="r")
            ax.set_title(f"wall {w}: log10(points+1), 5 cm x 1 cm; red = d50"); ax.set_xlabel("x (cm)")
            ax.set_ylabel("distance inside nominal wall (cm)")
        fig.tight_layout(); fig.savefig(os.path.join(a.figdir, f"{a.prefix}_occupancy.png"), dpi=110); plt.close(fig)
    print("figures in", a.figdir)


if __name__ == "__main__":
    main()

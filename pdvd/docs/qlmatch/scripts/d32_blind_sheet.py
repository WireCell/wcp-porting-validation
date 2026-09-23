#!/usr/bin/env python3
"""doc qlmatch/32 sec 4 -- BLIND scan sheet for one cluster in one light arm.

One PNG per (event, cluster, light): up to MAX_CAND candidate flashes for the cluster, one row each, ordered by flash
time and lettered A, B, ...  Each row draws the evidence of the pairing and nothing the matcher computed about it:
  - X-Z and X-Y projections of the event (grey) with this cluster (colour) and the bundle's other clusters (hollow)
    drift-corrected at THIS flash's time, with the drift boxes;
  - per-channel measured PE (bars) vs this cluster's predicted PE (line) on the cluster's volume side, railed
    (flash_sat) channels marked with a red "R" in every light alike;
  - measured and predicted 2-D light maps of that side.
Hidden: auto/selected marks, ks, chi2, strength, flags, flash gid, cluster uid, arm/light name.  The sheet title is
the random item id only.  Drawing helpers come read-only from pdvd/ql_display/render_groups.py.

Known leak (stated in the scan INDEX): under the ToT light, railed channels read higher than under twoside.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../ql_display"))
from render_groups import Event, draw_lightmap, MAX_PTS   # noqa: E402

MAX_CAND = 6
LETTERS = "ABCDEFGH"


def draw_proj(ax_xz, ax_xy, evt, bundle, dxs):
    """Event clusters grey, this cluster coloured, bundle's other clusters hollow; x shifted by this flash's dx."""
    main = bundle["main_cluster"]
    others = set(bundle["other_clusters"])
    for uid, c in evt.cluster_by_uid.items():
        if not c["x"]:
            continue
        x = np.asarray(c["x"]) + dxs[c["apa"]]
        y = np.asarray(c["y"])
        z = np.asarray(c["z"])
        if len(x) > MAX_PTS:
            s = np.linspace(0, len(x) - 1, MAX_PTS).astype(int)
            x, y, z = x[s], y[s], z[s]
        if uid == main:
            kw = dict(s=7, color="tab:red", alpha=1.0, lw=0, zorder=5)
        elif uid in others:
            kw = dict(s=4, color="tab:orange", alpha=0.8, lw=0, zorder=4)
        else:
            kw = dict(s=0.5, color="0.8", alpha=0.25, lw=0, zorder=1)
        ax_xz.scatter(z, x, **kw)
        ax_xy.scatter(y, x, **kw)
    for a in (0, 4):
        g = evt.geom[a]
        for ax, (h0, h1) in ((ax_xz, ("z_lo", "z_hi")), (ax_xy, ("y_lo", "y_hi"))):
            ax.plot([g[h0], g[h1], g[h1], g[h0], g[h0]],
                    [g["anode_x"], g["anode_x"], g["cathode_x"], g["cathode_x"], g["anode_x"]],
                    color="red", lw=0.6, alpha=0.6)
    ax_xz.set_title("X-Z at this flash's t0 (red = this cluster, orange = co-clusters)", fontsize=6)
    ax_xy.set_title("X-Y at this flash's t0; red box = drift volumes", fontsize=6)
    for ax, lab in ((ax_xz, "z"), (ax_xy, "y")):
        ax.set_xlabel(f"{lab} (cm)", fontsize=6)
        ax.set_ylabel("x (cm)", fontsize=6)
        ax.tick_params(labelsize=5)


def render_item(calib_path, uid, bundle_idx, item_id, outpath):
    """bundle_idx: the candidate bundles (this cluster as main) to draw, any order; drawn by flash time."""
    evt = Event(calib_path)
    cands = sorted(bundle_idx, key=lambda i: evt.flash_by_gid[evt.bundles[i]["flash_gid"]]["time"])[:MAX_CAND]
    n = len(cands)
    fig = plt.figure(figsize=(17, 3.0 * n + 0.8), dpi=80)
    gs = fig.add_gridspec(n, 6, width_ratios=[1.3, 1.3, 2.2, 1.0, 1.0, 1.3], hspace=0.55, wspace=0.3)
    c = evt.cluster_by_uid[uid]
    letters = {}
    for r, i in enumerate(cands):
        b = evt.bundles[i]
        f = evt.flash_by_gid[b["flash_gid"]]
        letters[LETTERS[r]] = i
        dxs = {a: evt.dx_cm(a, b["flash_gid"]) for a in (0, 4)}
        draw_proj(fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1]), evt, b, dxs)
        pe = np.asarray(f["pe"], float)
        pred = np.asarray(b["pred_pe"], float)
        sat = np.asarray(f.get("sat", [0] * evt.nchan), float) > 0
        side = np.where(evt.od_side[b["apa"]] & evt.od_active & ~evt.od_masked)[0]
        ax = fig.add_subplot(gs[r, 2])
        xs = np.arange(len(side))
        # y scale set by the UNRAILED channels and the prediction (rubric: R channels are a sanity check only);
        # a railed bar taller than that is clipped at the top and its value printed.
        unr_side = side[~sat[side]]
        top = max(float(pe[unr_side].max()) if len(unr_side) else 0.0, float(pred[side].max()), 1.0)
        ax.bar(xs, np.minimum(pe[side], top * 1.05), color="0.6", width=0.8, label="measured")
        ax.plot(xs, pred[side], color="tab:blue", lw=1.2, marker=".", ms=3, label="predicted (this cluster)")
        for k, ch in enumerate(side):
            if sat[ch]:
                lab = "R" if pe[ch] <= top * 1.05 else f"R\n{pe[ch]:.0f}"
                ax.text(k, top * 1.07, lab, color="red", fontsize=7, fontweight="bold", ha="center", va="bottom")
        ax.set_ylim(0, top * 1.35)
        ax.set_xticks(xs)
        ax.set_xticklabels(side, fontsize=6, rotation=90)
        ax.tick_params(labelsize=5)
        if r == 0:
            ax.legend(fontsize=5, loc="upper right")
        ax.set_title(f"channels on the {'bottom' if b['apa'] == 0 else 'top'} side; y scale = unrailed channels (R = railed, value printed if clipped)", fontsize=6)
        pe_map = pe.copy()                        # railed channels clipped to the unrailed maximum on the map
        if len(unr_side):
            pe_map[sat] = np.minimum(pe[sat], float(pe[unr_side].max()))
        draw_lightmap(fig.add_subplot(gs[r, 3]), evt, b["apa"], pe_map, "measured (R clipped)")
        draw_lightmap(fig.add_subplot(gs[r, 4]), evt, b["apa"], pred, "predicted")
        axt = fig.add_subplot(gs[r, 5])
        axt.axis("off")
        unr = side[~sat[side]]
        meas_u, pred_u = float(pe[unr].sum()), float(pred[unr].sum())
        txt = [f"CANDIDATE {LETTERS[r]}",
               f"flash t = {f['time']:.2f} us",
               f"railed ch (R): {int(sat[side].sum())} of {len(side)}",
               "UNRAILED channels, this side:",
               f"  meas {meas_u:.0f}  pred {pred_u:.0f} PE",
               f"  pred/meas {pred_u / meas_u:.2f}" if meas_u > 0 else "  pred/meas n/a",
               "ALL channels, this side:",
               f"  meas {float(pe[side].sum()):.0f}  pred {float(pred[side].sum()):.0f} PE",
               f"other clusters in hyp.: {len(b['other_clusters'])}"]
        g = evt.geom[c["apa"]]
        xs_c = np.asarray(c["x"]) + dxs[c["apa"]]
        lo_x, hi_x = sorted((g["anode_x"], g["cathode_x"]))
        txt += ["cluster x at this t0:",
                f"  [{xs_c.min():.1f}, {xs_c.max():.1f}] cm",
                f"  anode x {g['anode_x']:.1f}, cathode x {g['cathode_x']:.1f}",
                f"  outside drift box: {max(0.0, lo_x - xs_c.min()) + max(0.0, xs_c.max() - hi_x):.1f} cm",
                f"  to anode {min(abs(xs_c - g['anode_x'])):.1f} / to cathode {min(abs(xs_c - g['cathode_x'])):.1f} cm"]
        axt.text(0, 1, "\n".join(txt), va="top", fontsize=8, family="monospace", transform=axt.transAxes)
    length = evt.cluster_length(uid)
    fig.suptitle(f"item {item_id}   cluster: {'bottom' if c['apa'] == 0 else 'top'} volume, bbox length "
                 f"{length:.0f} cm, {c['npoints']} points   candidates A-{LETTERS[n - 1]} (by flash time)",
                 fontsize=9)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return {k: dict(bundle_idx=i, flash_gid=evt.bundles[i]["flash_gid"],
                    time=evt.flash_by_gid[evt.bundles[i]["flash_gid"]]["time"]) for k, i in letters.items()}

#!/usr/bin/env python3
"""doc qlmatch/33 -- LIGHT-NEUTRAL blind scan sheet for one cluster (fork of d32_blind_sheet.py).

One PNG per look at a cluster: up to MAX_CAND candidate flashes, one row each, lettered A, B, ... in the order the
caller passes them (the caller shuffles, so letters are NOT in flash-time order).  Everything is drawn from ONE calib
dump (the control arm's).  Differences from round 1 (doc 32):
  - railed channels carry NO measured value: a hatched full-height bar marked R ("bright enough to saturate").
    Unrailed PE is identical between the twoside and the ToT light (doc 32 sec 3, p99 1.003), so the sheet is the same
    whichever light an arm used;
  - the per-channel prediction is STACKED: the light predicted for the clusters that every frozen arm matches to this
    flash (co-matched, arm-neutral intersection, shaded) plus this cluster's own prediction (line on top).  A flash is
    lit by many clusters (owner procedure, ql-scan-criteria.md sec 3 point 5), and a stacked sum shows whether this
    cluster fits into what is left without a residual that would inherit the model's over-prediction;
  - a zoomed view of the cluster at this T0 with the anode and cathode planes, for short clusters.
Hidden: auto marks, ks, chi2, strength, flags, flash gid, cluster uid, arm/light name, flash time order.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../../ql_display"))
from render_groups import Event, draw_lightmap   # noqa: E402
from d32_blind_sheet import draw_proj            # noqa: E402

MAX_CAND = 5
LETTERS = "ABCDEFGH"


def draw_zoom(ax, evt, c, dx, geom):
    """This cluster only, x vs its longer transverse axis, window = cluster bbox +- 20 cm, planes drawn if inside."""
    x = np.asarray(c["x"]) + dx
    y, z = np.asarray(c["y"]), np.asarray(c["z"])
    h, hl = (z, "z") if np.ptp(z) >= np.ptp(y) else (y, "y")
    ax.scatter(h, x, s=4, color="tab:red", lw=0)
    pad = 20.0
    lo, hi = x.min() - pad, x.max() + pad
    planes = ((geom["anode_x"], "anode", "tab:green"), (geom["cathode_x"], "cathode", "tab:purple"))
    for xp, lab, col in planes:            # stretch to a plane only if it is within 60 cm of the cluster
        if x.min() - 60 <= xp < lo:
            lo = xp - 3
        elif hi < xp <= x.max() + 60:
            hi = xp + 3
    ax.set_ylim(lo, hi)
    away = []
    for xp, lab, col in planes:
        if lo <= xp <= hi:
            ax.axhline(xp, color=col, lw=1.2, ls="--")
            ax.text(0.02, (xp - lo) / (hi - lo), f" {lab}", color=col, fontsize=6, va="bottom",
                    transform=ax.transAxes)
        else:
            d = min(abs(xp - x.min()), abs(xp - x.max()))
            away.append(f"{lab} {d:.0f} cm {'below' if xp < lo else 'above'}")
    if away:
        ax.text(0.02, 0.02, "; ".join(away), fontsize=6, transform=ax.transAxes, color="0.3")
    hp = max(np.ptp(h), 5.0) * 0.15
    ax.set_xlim(h.min() - hp, h.max() + hp)
    ax.set_title(f"ZOOM: this cluster, x vs {hl}, planes dashed", fontsize=6)
    ax.set_xlabel(f"{hl} (cm)", fontsize=6)
    ax.set_ylabel("x (cm)", fontsize=6)
    ax.tick_params(labelsize=5)


def render_item(calib_path, uid, bundle_idx, co_bundles, item_id, outpath):
    """bundle_idx: candidate bundles (this cluster as main) in LETTER order; an entry is a bundle index into
    `calib_path`, or (other_calib_path, bundle index) for a candidate whose bundle exists only in another arm's dump
    (drawn from that dump; railed values stay hidden).
    co_bundles: {entry: [bundle indices, in the same dump, of the co-matched clusters on that flash]}."""
    evts = {calib_path: Event(calib_path)}
    for x in bundle_idx:
        if isinstance(x, tuple) and x[0] not in evts:
            evts[x[0]] = Event(x[0])
    cands = list(bundle_idx)[:MAX_CAND]
    n = len(cands)
    fig = plt.figure(figsize=(20, 3.1 * n + 0.9), dpi=80)
    gs = fig.add_gridspec(n, 7, width_ratios=[1.2, 1.2, 1.0, 2.2, 0.95, 0.95, 1.35], hspace=0.6, wspace=0.32)
    c = evts[calib_path].cluster_by_uid[uid]
    letters = {}
    for r, x in enumerate(cands):
        evt, i = (evts[x[0]], x[1]) if isinstance(x, tuple) else (evts[calib_path], x)
        b = evt.bundles[i]
        f = evt.flash_by_gid[b["flash_gid"]]
        letters[LETTERS[r]] = dict(src=evt.path, bundle_idx=i, flash_gid=b["flash_gid"], time=f["time"])
        dxs = {a: evt.dx_cm(a, b["flash_gid"]) for a in (0, 4)}
        draw_proj(fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1]), evt, b, dxs)
        g = evt.geom[c["apa"]]
        draw_zoom(fig.add_subplot(gs[r, 2]), evt, c, dxs[c["apa"]], g)
        pe = np.asarray(f["pe"], float)
        pred = np.asarray(b["pred_pe"], float)
        co = np.zeros_like(pred)
        for j in co_bundles.get(x, []):
            co += np.asarray(evt.bundles[j]["pred_pe"], float)
        tot = co + pred
        sat = np.asarray(f.get("sat", [0] * evt.nchan), float) > 0
        side = np.where(evt.od_side[b["apa"]] & evt.od_active & ~evt.od_masked)[0]
        unr = side[~sat[side]]
        ax = fig.add_subplot(gs[r, 3])
        xs = np.arange(len(side))
        top = max(float(pe[unr].max()) if len(unr) else 0.0, float(tot[side].max()), 1.0)
        meas_bar = np.where(sat[side], 0.0, pe[side])
        ax.bar(xs, meas_bar, color="0.6", width=0.8, label="measured (unrailed)")
        rx = xs[sat[side]]
        ax.bar(rx, [top * 1.05] * len(rx), color="none", edgecolor="red", hatch="///", width=0.8, lw=0.6,
               label="railed: bright, value not shown")
        for k in rx:
            ax.text(k, top * 1.07, "R", color="red", fontsize=7, fontweight="bold", ha="center", va="bottom")
        ax.fill_between(xs, 0, co[side], step="mid", color="tab:cyan", alpha=0.35, label="pred: co-matched clusters")
        ax.plot(xs, tot[side], color="tab:blue", lw=1.3, marker=".", ms=3, label="pred: co-matched + THIS cluster")
        ax.plot(xs, pred[side], color="tab:blue", lw=0.8, ls=":", label="pred: this cluster alone")
        ax.set_ylim(0, top * 1.35)
        ax.set_xticks(xs)
        ax.set_xticklabels(side, fontsize=6, rotation=90)
        ax.tick_params(labelsize=5)
        if r == 0:
            ax.legend(fontsize=5, loc="upper right")
        ax.set_title(f"channels, {'bottom' if b['apa'] == 0 else 'top'} side (R = railed, hatched, no value)",
                     fontsize=6)
        pe_map = np.where(sat, float(pe[unr].max()) if len(unr) else 0.0, pe)   # railed shown at the unrailed max
        draw_lightmap(fig.add_subplot(gs[r, 4]), evt, b["apa"], pe_map, "measured (R at max)")
        draw_lightmap(fig.add_subplot(gs[r, 5]), evt, b["apa"], tot, "predicted (co + this)")
        axt = fig.add_subplot(gs[r, 6])
        axt.axis("off")
        mu, pu, cu = float(pe[unr].sum()), float(pred[unr].sum()), float(co[unr].sum())
        xs_c = np.asarray(c["x"]) + dxs[c["apa"]]
        lo_x, hi_x = sorted((g["anode_x"], g["cathode_x"]))
        txt = [f"CANDIDATE {LETTERS[r]}",
               f"railed ch (R): {int(sat[side].sum())} of {len(side)}",
               "UNRAILED channels, this side:",
               f"  meas {mu:.0f} PE",
               f"  pred this {pu:.0f}, co-matched {cu:.0f}",
               f"  (co+this)/meas {(pu + cu) / mu:.2f}" if mu > 0 else "  (co+this)/meas n/a",
               f"  this/meas {pu / mu:.2f}" if mu > 0 else "  this/meas n/a",
               f"co-matched clusters: {len(co_bundles.get(x, []))}",
               "cluster x at this t0:",
               f"  [{xs_c.min():.1f}, {xs_c.max():.1f}] cm",
               f"  outside drift box: {max(0.0, lo_x - xs_c.min()) + max(0.0, xs_c.max() - hi_x):.1f} cm",
               f"  to anode {min(abs(xs_c - g['anode_x'])):.1f} / cathode {min(abs(xs_c - g['cathode_x'])):.1f} cm"]
        axt.text(0, 1, "\n".join(txt), va="top", fontsize=8, family="monospace", transform=axt.transAxes)
    fig.suptitle(f"item {item_id}   cluster: {'bottom' if c['apa'] == 0 else 'top'} volume, bbox length "
                 f"{evts[calib_path].cluster_length(uid):.0f} cm, {c['npoints']} points   candidates A-{LETTERS[n - 1]} "
                 "(letters in random order, not by time)", fontsize=9)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return letters

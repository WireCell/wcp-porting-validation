#!/usr/bin/env python
"""Headless per-flash-group evidence renderer for the PDVD Q/L hand scan.

Replicates the evidence panels of pdvd/ql_scan/ql_scan_viewer.py (drift-
corrected charge projections, measured/predicted 2-D light maps, per-channel
pred-vs-meas comparisons, candidate metrics) as one composite PNG per scan
group, so the scan can be judged without the interactive Bokeh app.  All
formulas follow the viewer: group numbering = ascending flash time (viewer
Event.__init__), dx = sign_offset * (t_flash + trigger_offset) * drift_speed
(Event.dx_cm), cluster length = 3-D bbox diagonal, opdet side membership by
opdet x with the cathode XAs (|x|<=10 cm) on both volumes.

Also writes a numeric context sidecar (JSONL, one line per nav group covering
ALL groups, rendered or not) with per-candidate metrics and the compare-table
evidence (competing flashes for the same main cluster).

Usage:
  python render_groups.py <calib-evt*.json> --outdir <dir> --context <file.jsonl>
                          [--mode auto|none] [--groups 12,40,...]

  --mode auto  (default) render groups with >=1 auto-selected bundle whose
               main cluster passes the viewer's 5 cm length filter
  --mode none  context JSONL only
  --groups     additionally render these group numbers (Pass-B escalation)
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MIN_CLUS_LEN = 5.0      # cm, viewer length filter
MAX_PANEL_CANDS = 8     # per-candidate mini-panels per PNG
MAX_PTS = 3000          # cluster downsample for projections


class Event:
    """Minimal re-implementation of the viewer's Event (load + geometry)."""

    def __init__(self, path):
        self.path = path
        with open(path) as fh:
            d = json.load(fh)
        self.nchan = d["nchan"]
        self.drift_speed = d["drift_speed"]                  # cm/us
        self.trigger_offset = d.get("trigger_offset", 0.0)   # us, matcher-applied
        self.qp = d.get("quality_params", {})
        self.geom = {int(k): dict(g) for k, g in d["geometry"].items()}
        flashes = list(d["flashes"])
        self.cluster_by_uid = {c["uid"]: c for c in d["clusters"]}
        self.bundles = d["bundles"]
        self._clen = {}
        for g, f in enumerate(sorted(flashes, key=lambda f: f["time"]), start=1):
            f["group"] = g
        self.flash_by_gid = {f["gid"]: f for f in flashes}
        od0 = d["opdets"]
        self.od_x = np.array([o["x"] for o in od0])
        self.od_y = np.array([o["y"] for o in od0])
        self.od_z = np.array([o["z"] for o in od0])
        self.od_active = np.array([o["active"] for o in od0], dtype=bool)
        self.od_masked = np.array([o.get("auto_masked", False) for o in od0],
                                  dtype=bool)
        cath = np.abs(self.od_x) <= 10.0
        self.od_side = {0: cath | (self.od_x < -10.0),
                        4: cath | (self.od_x > 10.0)}

    def cluster_length(self, uid):
        if uid not in self._clen:
            c = self.cluster_by_uid[uid]
            if c["x"]:
                d = sum((max(c[a]) - min(c[a])) ** 2 for a in ("x", "y", "z"))
                self._clen[uid] = math.sqrt(d)
            else:
                self._clen[uid] = 0.0
        return self._clen[uid]

    def dx_cm(self, apa, gid):
        g = self.geom[apa]
        t_us = self.flash_by_gid[gid]["time"] + self.trigger_offset
        return g["sign_offset"] * t_us * self.drift_speed


def c2n(b):
    return b["chi2"] / b["ndf"] if b["ndf"] else float("nan")


def flag_str(b):
    lab = [("consistent", "consist"), ("potential_bad_match", "badmatch"),
           ("close_to_PMT", "nearPD"), ("at_cathode", "atCATH"),
           ("at_x_boundary", "xbound"), ("spec_end", "specend"),
           ("window_truncated", "wtrunc"), ("two_boundary", "2bnd")]
    s = [short for k, short in lab if b.get(k)]
    if not b.get("contained", True):
        s.append("UNCONTAINED")
    return ",".join(s) or "-"


def group_candidates(evt, g):
    """Bundle indices in group g, viewer table order (longest main cluster
    first), length filter applied."""
    idx = [i for i, b in enumerate(evt.bundles)
           if evt.flash_by_gid[b["flash_gid"]]["group"] == g
           and evt.cluster_length(b["main_cluster"]) > MIN_CLUS_LEN]
    idx.sort(key=lambda i: -evt.cluster_length(evt.bundles[i]["main_cluster"]))
    return idx


def competing(evt, i, cap=8):
    """Compare-table evidence: other bundles sharing bundle i's main cluster,
    best-KS first (all flashes)."""
    uid = evt.bundles[i]["main_cluster"]
    alt = [j for j, b in enumerate(evt.bundles)
           if b["main_cluster"] == uid and j != i]
    alt.sort(key=lambda j: evt.bundles[j]["ks_dis"])
    out = []
    for j in alt[:cap]:
        b = evt.bundles[j]
        f = evt.flash_by_gid[b["flash_gid"]]
        out.append({"group": f["group"], "flash_gid": b["flash_gid"],
                    "time_us": round(f["time"], 3),
                    "ks": round(b["ks_dis"], 3), "chi2ndf": round(c2n(b), 1),
                    "strength": round(b["strength"], 3),
                    "predPE": round(b["total_pred_light"], 1),
                    "flashPE": round(f["total_PE"], 1),
                    "auto": bool(b["auto_selected"])})
    return out, len(alt)


def cand_record(evt, i):
    b = evt.bundles[i]
    comp, ncomp = competing(evt, i)
    return {"bundle_idx": i, "apa": b["apa"],
            "main_uid": b["main_cluster"],
            "ident": evt.cluster_by_uid[b["main_cluster"]]["ident"],
            "len_cm": round(evt.cluster_length(b["main_cluster"]), 1),
            "n_other": len(b["other_clusters"]),
            "ks": round(b["ks_dis"], 3), "chi2ndf": round(c2n(b), 1),
            "ndf": b["ndf"], "strength": round(b["strength"], 3),
            "predPE": round(b["total_pred_light"], 1),
            "auto": bool(b["auto_selected"]), "flags": flag_str(b),
            "n_competing": ncomp, "competing": comp}


def draw_lightmap(ax, evt, apa, pe, title):
    side = evt.od_side[apa]
    hi = max(float(np.max(pe[side])), 1e-6)
    for k in np.where(side)[0]:
        if not evt.od_active[k] or evt.od_masked[k]:
            ax.plot(evt.od_z[k], evt.od_y[k], "x", color="red", ms=6)
            continue
        r = 2.5 + 9.0 * math.sqrt(max(pe[k], 0.0) / hi)
        ax.add_patch(plt.Circle((evt.od_z[k], evt.od_y[k]), r * 2.2,
                                color=plt.cm.viridis(min(pe[k] / hi, 1.0)),
                                alpha=0.9))
        if abs(evt.od_x[k]) <= 10.0:      # cathode XA marker
            ax.add_patch(plt.Circle((evt.od_z[k], evt.od_y[k]), r * 2.2,
                                    fill=False, ec="orange", lw=0.8))
    ax.set_xlim(evt.od_z.min() - 30, evt.od_z.max() + 30)
    ax.set_ylim(evt.od_y.min() - 30, evt.od_y.max() + 30)
    ax.set_title("%s (max %.0f PE)" % (title, hi), fontsize=8)
    ax.tick_params(labelsize=6)


def draw_projections(axs, evt, gid, cand_idx, colors):
    """axs = (xz, xy, yz). All event clusters grey (shifted per volume with
    this flash's dx), candidate mains colored, candidate others same color
    hollow."""
    dx = {a: evt.dx_cm(a, gid) for a in (0, 4)}
    cand_uids = {}
    for ci, i in enumerate(cand_idx):
        b = evt.bundles[i]
        for u in [b["main_cluster"]] + b["other_clusters"]:
            cand_uids.setdefault(u, (ci, u == b["main_cluster"]))
    for uid, c in evt.cluster_by_uid.items():
        if not c["x"]:
            continue
        x = np.asarray(c["x"]) + dx[c["apa"]]
        y = np.asarray(c["y"]); z = np.asarray(c["z"])
        if len(x) > MAX_PTS:
            s = np.linspace(0, len(x) - 1, MAX_PTS).astype(int)
            x, y, z = x[s], y[s], z[s]
        if uid in cand_uids:
            ci, is_main = cand_uids[uid]
            kw = dict(s=2 if is_main else 1.2, color=colors[ci % len(colors)],
                      alpha=0.9 if is_main else 0.45, lw=0)
        else:
            kw = dict(s=0.8, color="0.75", alpha=0.35, lw=0)
        axs[0].scatter(z, x, **kw)
        axs[1].scatter(y, x, **kw)
        axs[2].scatter(z, y, **kw)
    # drift boxes + anode/cathode lines
    for a in (0, 4):
        g = evt.geom[a]
        for ax, horiz in ((axs[0], ("z_lo", "z_hi")), (axs[1], ("y_lo", "y_hi"))):
            ax.plot([g[horiz[0]], g[horiz[1]], g[horiz[1]], g[horiz[0]], g[horiz[0]]],
                    [g["anode_x"], g["anode_x"], g["cathode_x"], g["cathode_x"],
                     g["anode_x"]], color="red", lw=0.6, alpha=0.6)
            ax.axhline(g["anode_x"], color="red", lw=0.4, ls=":", alpha=0.5)
    g0, g4 = evt.geom[0], evt.geom[4]
    axs[2].plot([g0["z_lo"], g0["z_hi"], g0["z_hi"], g0["z_lo"], g0["z_lo"]],
                [g0["y_lo"], g0["y_lo"], g0["y_hi"], g0["y_hi"], g0["y_lo"]],
                color="red", lw=0.6, alpha=0.6)
    axs[0].set_xlabel("z (cm)", fontsize=7); axs[0].set_ylabel("x (cm)", fontsize=7)
    axs[1].set_xlabel("y (cm)", fontsize=7); axs[1].set_ylabel("x (cm)", fontsize=7)
    axs[2].set_xlabel("z (cm)", fontsize=7); axs[2].set_ylabel("y (cm)", fontsize=7)
    axs[0].set_title("X-Z (drift-corrected)", fontsize=8)
    axs[1].set_title("X-Y", fontsize=8)
    axs[2].set_title("Y-Z", fontsize=8)
    for ax in axs:
        ax.tick_params(labelsize=6)


def render_group(evt, g, gid, cand_idx, outpath):
    f = evt.flash_by_gid[gid]
    pe = np.asarray(f["pe"], dtype=float)
    colors = plt.cm.tab10.colors
    panel_cands = sorted(cand_idx,
                         key=lambda i: (not evt.bundles[i]["auto_selected"],
                                        -evt.cluster_length(
                                            evt.bundles[i]["main_cluster"])))
    panel_cands = panel_cands[:MAX_PANEL_CANDS]

    fig = plt.figure(figsize=(22, 13), dpi=100)
    gs = fig.add_gridspec(3, 8, height_ratios=[1.2, 0.9, 0.9],
                          hspace=0.35, wspace=0.35)
    # row 1: projections + text
    ax_xz = fig.add_subplot(gs[0, 0:2])
    ax_xy = fig.add_subplot(gs[0, 2:4])
    ax_yz = fig.add_subplot(gs[0, 4:5])
    ax_txt = fig.add_subplot(gs[0, 5:8]); ax_txt.axis("off")
    draw_projections((ax_xz, ax_xy, ax_yz), evt, gid, panel_cands, colors)

    # row 2: measured + predicted(auto combo) light maps per volume
    pred_auto = {a: np.zeros(evt.nchan) for a in (0, 4)}
    for i in cand_idx:
        b = evt.bundles[i]
        if b["auto_selected"]:
            pred_auto[b["apa"]] += np.asarray(b["pred_pe"], dtype=float)
    for col, (apa, arr, ttl) in enumerate(
            [(0, pe, "meas bottom"), (4, pe, "meas top"),
             (0, pred_auto[0], "pred(auto) bottom"),
             (4, pred_auto[4], "pred(auto) top")]):
        ax = fig.add_subplot(gs[1, col * 2:col * 2 + 2])
        draw_lightmap(ax, evt, apa, arr, ttl)

    # row 3: per-candidate channel comparison
    for k, i in enumerate(panel_cands):
        b = evt.bundles[i]
        side = np.where(evt.od_side[b["apa"]] & evt.od_active
                        & ~evt.od_masked)[0]
        ax = fig.add_subplot(gs[2, k])
        xs = np.arange(len(side))
        ax.bar(xs, pe[side], color="0.6", width=0.8)
        ax.plot(xs, np.asarray(b["pred_pe"], dtype=float)[side],
                color=colors[k % len(colors)], lw=1.2, marker=".", ms=3)
        ax.set_xticks(xs[::max(1, len(xs) // 8)])
        ax.set_xticklabels(side[::max(1, len(xs) // 8)], fontsize=5)
        ax.tick_params(labelsize=5)
        ax.set_title("[%d]%s %s c%d ks%.2f c2n%.0f st%.2f\n%s"
                     % (k, "*" if b["auto_selected"] else " ",
                        "bot" if b["apa"] == 0 else "top",
                        evt.cluster_by_uid[b["main_cluster"]]["ident"],
                        b["ks_dis"], c2n(b), b["strength"], flag_str(b)),
                     fontsize=6, color=colors[k % len(colors)])

    # text panel: header + full candidate table
    lines = ["group %d   flash gid=%d id=%d  t=%.2f us  totPE=%.0f  ncand=%d"
             % (g, gid, f["id"], f["time"], f["total_PE"], len(cand_idx)),
             "thresholds: ks<%.2f ndf>=%d strength>%.2f (PDVD round-1: "
             "KS/chi2 inflated, judge shape+amplitude+geometry)"
             % (evt.qp.get("highconsist_ks_max", 0.06),
                evt.qp.get("highconsist_min_ndf", 3),
                evt.qp.get("strength_cutoff", 0.05)),
             "",
             " #  auto vol clus   len   ks   c2n    str  predPE flags"]
    for k, i in enumerate(cand_idx):
        b = evt.bundles[i]
        mark = "[%d]" % panel_cands.index(i) if i in panel_cands else "   "
        lines.append("%s%s %s %s c%-4d %5.0f %5.2f %6.0f %6.2f %7.0f %s"
                     % (mark, "*" if b["auto_selected"] else " ",
                        "A" if b["auto_selected"] else " ",
                        "bot" if b["apa"] == 0 else "top",
                        evt.cluster_by_uid[b["main_cluster"]]["ident"],
                        evt.cluster_length(b["main_cluster"]),
                        b["ks_dis"], c2n(b), b["strength"],
                        b["total_pred_light"], flag_str(b)))
        if len(lines) > 40:
            lines.append("... (%d more)" % (len(cand_idx) - k - 1))
            break
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=6.5, transform=ax_txt.transAxes)

    fig.suptitle(os.path.basename(outpath), fontsize=9)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "none"])
    ap.add_argument("--groups", default="",
                    help="comma-separated extra group numbers to render")
    args = ap.parse_args()

    evt = Event(args.calib)
    base = os.path.basename(args.calib)
    event = base[len("calib-"):-len(".json")]   # evt<ID>, viewer event_label

    nav = {}
    for g in sorted({f["group"] for f in evt.flash_by_gid.values()}):
        gid = next(gg for gg, f in evt.flash_by_gid.items() if f["group"] == g)
        cands = group_candidates(evt, g)
        if cands:
            nav[g] = (gid, cands)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.context) or ".", exist_ok=True)

    extra = {int(x) for x in args.groups.split(",") if x.strip()}
    n_png = 0
    with open(args.context, "w") as ctx:
        for g, (gid, cands) in nav.items():
            f = evt.flash_by_gid[gid]
            rec = {"event": event, "group": g, "flash_gid": gid,
                   "flash_id": f["id"], "time_us": round(f["time"], 3),
                   "total_PE": round(f["total_PE"], 1),
                   "has_auto": any(evt.bundles[i]["auto_selected"] for i in cands),
                   "candidates": [cand_record(evt, i) for i in cands]}
            ctx.write(json.dumps(rec) + "\n")
            want = (args.mode == "auto" and rec["has_auto"]) or g in extra
            if want:
                out = os.path.join(args.outdir, "grp%03d_gid%d.png" % (g, gid))
                render_group(evt, g, gid, cands, out)
                n_png += 1
    print("%s: %d nav groups -> context %s, %d PNGs -> %s"
          % (event, len(nav), args.context, n_png, args.outdir))


if __name__ == "__main__":
    main()

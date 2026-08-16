#!/usr/bin/env python3
"""Overlay the PR segments on the 2-D MEASUREMENT near a vertex.

The 3-D panels (scankit.panel_zoom) draw only the reconstruction's own answer.
This one draws the image underneath it: `proj[]` in the calib dump is the
measured charge per (apa, plane) as (wire, slice, charge), and every fitted
point carries `pu/pv/pw` (fractional per-APA wire index) and `pt` (tick), so a
segment can be drawn in the panel's own coordinates with no wire geometry.

Coordinates and the zoom recipe follow pr_display_viewer.py:2370-2540 verbatim
so a picture here and port 5017 can be talked about in the same words:

  * `pt` is TICKS, `proj[].slice` is SLICES -- divide by
    meta.nticks_per_slice keyed on (apa, face).  SBND is 4.
  * points with `apa < 0` are dropped, never defaulted to APA 0 (doc pr/3).
  * dead bands are in slice units: `s0/s1`, not `t0/t1`.
  * the window grows x1,2,4,8 until >=2 fitted points are inside (doc pr/75) --
    an isolated micro-stub, which is this doc's whole population, has none
    within +-h by construction.

`proj[]` is NOT filtered by cluster_id: a mode-1 CUT prong lives in a different
cluster from the click (evt407280: click in 51, the 128.8 cm prong in 16), and
filtering would erase exactly the evidence this overlay exists to show.

  python3 pr85_panels2d.py <outdir> <event> [<event> ...]
  python3 pr85_panels2d.py <outdir> 38856:12106       # a chosen vertex id
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import Rectangle                         # noqa: E402

SX = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
sys.path.insert(0, os.path.join(SX, "vtx_rules"))
import baselines                                                 # noqa: E402
import scankit                                                   # noqa: E402
import vtx_io                                                    # noqa: E402

PLANE_NAME = ("U", "V", "W")
PAD = 10          # wires / slices, pr_display_viewer.py:2536
HALF = 6.0        # cm, the 3-D sphere the window is derived from


def nps_map(dump):
    """(apa, face) -> nticks_per_slice.  Keyed on BOTH, not on apa alone."""
    return {(r["apa"], r["face"]): r["nticks_per_slice"]
            for r in dump.get("meta", {}).get("nticks_per_slice", [])}


def click_vertex(dump, click):
    best, bd = None, None
    for v in dump.get("vertices", []):
        q = scankit.vertex_xyz(v)
        if q is None:
            continue
        dd = math.dist(q, click)
        if bd is None or dd < bd:
            best, bd = v, dd
    return best, bd


def vertex_apa(dump, c):
    """The APA whose fitted points sit nearest the vertex."""
    best, bd = None, None
    for s in dump.get("segments", []):
        for p in s.get("points") or []:
            if p.get("apa", -1) < 0:
                continue
            dd = math.dist((p["x"], p["y"], p["z"]), c)
            if bd is None or dd < bd:
                best, bd = (p["apa"], p["face"]), dd
    return best


def window(dump, apa, plane, c, nps):
    """Wire/slice span of the fitted points inside a growing 3-D sphere."""
    key = ("pu", "pv", "pw")[plane]
    for grow in (1.0, 2.0, 4.0, 8.0):
        hh = HALF * grow
        ws, ss = [], []
        for s in dump.get("segments", []):
            for p in s.get("points") or []:
                if p.get("apa", -1) != apa:
                    continue
                if (abs(p["x"] - c[0]) > hh or abs(p["y"] - c[1]) > hh
                        or abs(p["z"] - c[2]) > hh):
                    continue
                ws.append(p[key])
                ss.append(p["pt"] / nps.get((apa, p["face"]), 1))
        if len(ws) >= 2:
            return (min(ws) - PAD, max(ws) + PAD,
                    min(ss) - PAD, max(ss) + PAD), grow
    return None, None


def render(dump, out, vid, title_head, half_note=""):
    seg_idx = {s["id"]: i for i, s in enumerate(dump.get("segments", []))}
    seg_of = vtx_io.segments_of_vertex(dump)
    vmap = {v["id"]: v for v in dump.get("vertices", [])}
    v = vmap[vid]
    c = scankit.vertex_xyz(v)
    here = {s["id"] for s in seg_of.get(vid, [])}
    # Segments whose charge reaches the vertex but which are NOT incident on
    # it -- the pr/85 objects.  Drawn dashed magenta and labelled: they are
    # invisible as "grey background" and they are the whole point of the plot.
    near = set()
    for s in dump.get("segments", []):
        if s["id"] in here:
            continue
        for p in s.get("points") or []:
            if math.dist((p["x"], p["y"], p["z"]), c) <= 3.0:
                near.add(s["id"])
                break
    nps = nps_map(dump)
    apaface = vertex_apa(dump, c)
    if apaface is None:
        raise SystemExit("no fitted points with a valid APA in this event")
    apa = apaface[0]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for plane, ax in enumerate(axes):
        win, grow = window(dump, apa, plane, c, nps)
        # --- the measurement -------------------------------------------
        cells = [p for p in dump.get("proj", [])
                 if p["apa"] == apa and p["plane"] == plane]
        wx, wy, wq = [], [], []
        for p in cells:
            wx.extend(p["wire"]); wy.extend(p["slice"]); wq.extend(p["charge"])
        if win is not None:
            keep = [i for i in range(len(wx))
                    if win[0] <= wx[i] <= win[1] and win[2] <= wy[i] <= win[3]]
        else:
            keep = list(range(len(wx)))
        qs = sorted(wq[i] for i in keep) if keep else []
        qmax = qs[int(0.99 * (len(qs) - 1))] if qs else 1.0
        if keep:
            # vmin below zero on purpose: with vmin=0 a low-charge induction
            # cell renders white and the measurement looks ABSENT under the
            # polyline, which is the exact wrong conclusion this plot exists
            # to prevent.  Every measured cell must be visible as a cell.
            ax.scatter([wx[i] for i in keep], [wy[i] for i in keep], s=26,
                       c=[wq[i] for i in keep], cmap="Greys",
                       vmin=-0.30 * qmax, vmax=qmax, marker="s",
                       linewidths=0, zorder=2)
        # --- dead bands (SLICE units) ----------------------------------
        for dd in dump.get("dead", []):
            if dd["apa"] != apa or dd["plane"] != plane:
                continue
            s0 = dd.get("s0", dd["t0"]); s1 = dd.get("s1", dd["t1"])
            ax.add_patch(Rectangle((dd["wire"] - 0.5, s0), 1.0,
                                   max(1.0, s1 - s0), facecolor="#ffd9d9",
                                   edgecolor="none", zorder=1))
        # --- the PR segments -------------------------------------------
        key = ("pu", "pv", "pw")[plane]
        for s in dump.get("segments", []):
            runs = []
            cur = []
            for p in s.get("points") or []:
                if p.get("apa", -1) != apa:          # drop apa<0 AND other APAs
                    if cur:
                        runs.append(cur); cur = []
                    continue
                cur.append((p[key], p["pt"] / nps.get((apa, p["face"]), 1)))
            if cur:
                runs.append(cur)
            mine = s["id"] in here
            isnear = s["id"] in near
            col = (scankit._seg_color(seg_idx[s["id"]]) if mine
                   else "#d81b8c" if isnear else "#7fb3d5")
            ls = ":" if isnear else "-"
            for r in runs:
                if len(r) < 2:
                    if r:
                        ax.plot([r[0][0]], [r[0][1]], marker="o", ms=5,
                                color=col, zorder=7)
                    continue
                ax.plot([q[0] for q in r], [q[1] for q in r], color=col, ls=ls,
                        lw=2.6 if mine else 2.2 if isnear else 1.2,
                        alpha=0.95 if (mine or isnear) else 0.55,
                        zorder=6 if mine else 7 if isnear else 4)
            if (mine or isnear) and runs:
                mid = runs[0][len(runs[0]) // 2]
                ax.annotate(str(s["id"]), xy=mid, fontsize=7, color=col,
                            zorder=9,
                            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
        # --- the vertex ------------------------------------------------
        vp, vd = None, None
        for s in dump.get("segments", []):
            for p in s.get("points") or []:
                if p.get("apa", -1) != apa:
                    continue
                dd = math.dist((p["x"], p["y"], p["z"]), c)
                if vd is None or dd < vd:
                    vp, vd = (p[key], p["pt"] / nps.get((apa, p["face"]), 1)), dd
        if vp is not None:
            ax.plot([vp[0]], [vp[1]], marker="*", ms=17, mfc="#e8000b",
                    mec="black", mew=0.7, zorder=10)
        if win is not None:
            ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
        ax.set_xlabel("%s wire index (APA %d)" % (PLANE_NAME[plane], apa))
        ax.set_ylabel("slice")
        ax.set_title("%s plane%s" % (PLANE_NAME[plane],
                                     "" if grow in (None, 1.0)
                                     else "   (window x%g)" % grow))
        ax.grid(alpha=0.15, lw=0.5)
    fig.suptitle(title_head + half_note
                 + "\ngrey = measured charge cells   pink = dead   "
                   "star = the vertex   coloured solid = incident on it   "
                   "magenta dotted = charge within 3 cm but NOT incident   "
                   "pale blue = everything else",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=125)
    plt.close(fig)
    return out


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    labs = {L["eventNo"]: L for L in vtx_io.load_labels()
            if baselines.deployed_dump_path(L)}
    for spec in sys.argv[2:]:
        ev, _, want = spec.partition(":")
        ev = int(ev)
        L = labs[ev]
        with open(baselines.deployed_dump_path(L)) as fh:
            dump = json.load(fh)
        if want:
            vid, note = int(want), "  vertex %s (chosen)" % want
        else:
            v, gap = click_vertex(dump, L["truth"])
            vid = v["id"]
            note = "  clicked vertex %s (%.2f cm from click)" % (vid, gap)
        tag = "%s-v%s" % (ev, vid)
        p = render(dump, os.path.join(outdir, "evt%s-2d.png" % tag), vid,
                   "evt%d   PR segments over the 2-D measurement" % ev, note)
        print(p)


if __name__ == "__main__":
    main()

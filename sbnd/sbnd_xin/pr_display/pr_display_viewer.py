#!/usr/bin/env python3
"""SBND pattern-recognition (PR) event display (Bokeh server).

Stage 1: a read-only viewer for what the PR chain produced, built to drive
tuning of the PR code over the 572 valfast events.  See
sbnd_xin/docs/pr/26_pr-event-display.md.

INPUT is one self-contained JSON per event, written by the `pr_display` stage
(PrDisplayDump) of the PR chain:

    PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh <ql_root> <out> data 388
    -> <out>/pr_evt<ID>/calib-pr-evt<ID>.json

Nothing else is read.  In particular the display deliberately does NOT read
tracking-pr.root: that file's T_proj_data has only its `cluster_id` branch
(TTree::Branch refuses vector<vector<int> > without a compiled CollectionProxy
-- doc pr/26 sec 5), so the 2-D measurement is not in it.

LAYOUT

  Row 1 -- the three charge projections X-Y, Y-Z, X-Z, each carrying every
           layer below, individually toggleable.
  Row 2 -- six panels, two columns (TPC 0 | TPC 1) x three rows (T-U, T-V,
           T-W): the fitted 2-D charge as a heat map with the best-fit track
           drawn over it, i.e. the Magnify-tracking view for the neutrino
           interaction.

LAYERS (each a toggle)

  track fit      the PR graph's segments, drawn as polylines, coloured per
                 segment; this is the reconstructed particle trajectory set
  shower pts     associated 3-D points flagged shower
  track pts      associated 3-D points flagged track
  steiner        the Steiner skeleton (steiner_pc) of every cluster
  terminals      only the flag_steiner_terminal subset of that skeleton
  vertices       PR graph vertices; the neutrino vertex is drawn larger
  dead           dead-channel bands, 2-D panels only

ZOOM.  "zoom" reframes all nine panels to +-R around a centre.  The centre is
the identified neutrino vertex by default; type any (x, y, z) in cm into the
centre boxes (or press "vertex" to go back).  The 2-D panels follow the same
centre, projected onto each plane through the fitted points nearest it.

Coordinates follow doc pr/7: positions in cm, wire coordinates are FRACTIONAL
per-APA wire indices (integer = wire centre) and time is a slice index.
"""

import sys
import os
import glob
import json
import argparse
import math
from collections import defaultdict

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, Select, Button, Div, HoverTool,
                          CheckboxButtonGroup, TextInput, Toggle, Spacer,
                          ColorBar, LinearColorMapper, BasicTicker)
from bokeh.palettes import Viridis256, Category20_20
from bokeh.plotting import figure

HERE = os.path.dirname(os.path.abspath(__file__))

# SBND active volume (cm) -- same numbers nusel_display uses.
DET_BOX = dict(x=(-201.05, 201.05), y=(-199.312, 199.312), z=(0.85, 500.15))

DEFAULT_HALF = 30.0        # the "+-30 cm around the vertex" the display is for

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("specs", nargs="*", help="calib-pr-evt*.json paths or globs")
args = ap.parse_args(sys.argv[1:])

FILES = []
for spec in args.specs:
    FILES += sorted(glob.glob(spec)) if any(c in spec for c in "*?[") else [spec]
# Dedupe, keep order, keep only readable files.
seen, PATHS = set(), []
for p in FILES:
    rp = os.path.realpath(p)
    if rp in seen or not os.path.isfile(rp):
        continue
    seen.add(rp)
    PATHS.append(p)


def evt_label(path):
    """'calib-pr-evt388.json' -> 'evt388'; falls back to the parent dir name."""
    base = os.path.basename(path)
    if base.startswith("calib-pr-evt") and base.endswith(".json"):
        return "evt" + base[len("calib-pr-evt"):-len(".json")]
    return os.path.basename(os.path.dirname(path)) or base


EVENTS = {}          # label -> path
for p in PATHS:
    EVENTS.setdefault(evt_label(p), p)
LABELS = list(EVENTS)

state = dict(doc=None, label=None)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
proj_kw = dict(height=340, width=430, tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom")
f_xy = figure(title="X-Y", **proj_kw)
f_yz = figure(title="Y-Z", **proj_kw)
f_xz = figure(title="X-Z", **proj_kw)
f_xy.xaxis.axis_label, f_xy.yaxis.axis_label = "x (cm)", "y (cm)"
f_yz.xaxis.axis_label, f_yz.yaxis.axis_label = "z (cm)", "y (cm)"
f_xz.xaxis.axis_label, f_xz.yaxis.axis_label = "x (cm)", "z (cm)"
PROJ = ((f_xy, "x", "y"), (f_yz, "z", "y"), (f_xz, "x", "z"))

# The 3-D layers.  One CDS per layer, shared by all three projections: a point
# carries x/y/z and each figure picks the two it needs.  (Bokeh renders from
# whichever columns the glyph names, so no duplication is required.)
EMPTY3 = dict(x=[], y=[], z=[], c=[], tag=[])
shower_src = ColumnDataSource(data=dict(EMPTY3))
track_src = ColumnDataSource(data=dict(EMPTY3))
steiner_src = ColumnDataSource(data=dict(EMPTY3))
term_src = ColumnDataSource(data=dict(EMPTY3))
vtx_src = ColumnDataSource(data=dict(EMPTY3))
mainvtx_src = ColumnDataSource(data=dict(EMPTY3))
det_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                     xs_xz=[], ys_xz=[]))
# Segment polylines: one multi_line CDS per projection (the xs/ys pairs differ
# per projection, unlike the scatter layers).
seg_src = {k: ColumnDataSource(data=dict(xs=[], ys=[], c=[], sid=[], pid=[],
                                         cid=[], shower=[]))
           for k in ("xy", "yz", "xz")}

RENDER = defaultdict(list)          # layer name -> [renderers], for toggling

for f, hx, hy in PROJ:
    f.multi_line(xs="xs_%s" % ("xy" if f is f_xy else "yz" if f is f_yz else "xz"),
                 ys="ys_%s" % ("xy" if f is f_xy else "yz" if f is f_yz else "xz"),
                 source=det_src, line_color="#cc4444", line_width=1)

for f, hx, hy in PROJ:
    key = "xy" if f is f_xy else "yz" if f is f_yz else "xz"

    RENDER["steiner"].append(
        f.scatter(hx, hy, source=steiner_src, marker="circle", size=2,
                  fill_color="#b0b0b0", line_color=None, fill_alpha=0.5))
    RENDER["track"].append(
        f.scatter(hx, hy, source=track_src, marker="circle", size=3,
                  fill_color="#1f77b4", line_color=None, fill_alpha=0.55))
    RENDER["shower"].append(
        f.scatter(hx, hy, source=shower_src, marker="circle", size=3,
                  fill_color="#d62728", line_color=None, fill_alpha=0.45))
    RENDER["terminals"].append(
        f.scatter(hx, hy, source=term_src, marker="cross", size=7,
                  line_color="#ff7f0e", line_width=1.2, fill_color=None))
    r = f.multi_line(xs="xs", ys="ys", source=seg_src[key], line_color="c",
                     line_width=2.5, line_alpha=0.95)
    RENDER["trackfit"].append(r)
    f.add_tools(HoverTool(renderers=[r], tooltips=[
        ("segment", "@sid"), ("cluster", "@cid"),
        ("pdg", "@pid"), ("shower", "@shower")]))
    RENDER["vertices"].append(
        f.scatter(hx, hy, source=vtx_src, marker="circle", size=6,
                  fill_color=None, line_color="#111111", line_width=1.2))
    RENDER["vertices"].append(
        f.scatter(hx, hy, source=mainvtx_src, marker="star", size=20,
                  fill_color="#e377c2", line_color="#7b2d6b", line_width=1.5))

# --- the six 2-D panels -----------------------------------------------------
PLANE_NAME = ("U", "V", "W")
panel = {}          # (apa, plane) -> dict(fig, cell, fit, dead)
CMAP = LinearColorMapper(palette=Viridis256, low=0, high=1)

for apa in (0, 1):
    for pl in (0, 1, 2):
        f = figure(title="TPC %d   T vs %s" % (apa, PLANE_NAME[pl]),
                   width=560, height=250,
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   active_scroll="wheel_zoom")
        f.xaxis.axis_label = "%s wire index" % PLANE_NAME[pl]
        f.yaxis.axis_label = "time slice"
        cell = ColumnDataSource(data=dict(w=[], s=[], q=[], qp=[], cid=[]))
        dead = ColumnDataSource(data=dict(w=[], s=[], h=[]))
        fit = ColumnDataSource(data=dict(xs=[], ys=[], c=[], sid=[]))
        rd = f.rect(x="w", y="s", width=1.0, height="h", source=dead,
                    fill_color="#dddddd", line_color=None, fill_alpha=0.6)
        rc = f.rect(x="w", y="s", width=1.0, height=1.0, source=cell,
                    fill_color=dict(field="q", transform=CMAP),
                    line_color=None)
        rf = f.multi_line(xs="xs", ys="ys", source=fit, line_color="c",
                          line_width=2.0, line_alpha=0.95)
        f.add_tools(HoverTool(renderers=[rc], tooltips=[
            ("wire", "@w"), ("slice", "@s"),
            ("charge", "@q{0,0}"), ("pred", "@qp{0,0}"), ("cluster", "@cid")]))
        RENDER["dead"].append(rd)
        RENDER["trackfit2d"].append(rf)
        panel[(apa, pl)] = dict(fig=f, cell=cell, fit=fit, dead=dead)

cbar_fig = figure(width=110, height=250, toolbar_location=None,
                  outline_line_color=None)
cbar_fig.add_layout(ColorBar(color_mapper=CMAP, ticker=BasicTicker(desired_num_ticks=5),
                             label_standoff=6, title="charge (e)"), "right")
cbar_fig.xaxis.visible = cbar_fig.yaxis.visible = False
cbar_fig.grid.visible = False


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
event_select = Select(title="event", options=LABELS,
                      value=LABELS[0] if LABELS else "", width=200)
prev_btn = Button(label="< prev", width=80)
next_btn = Button(label="next >", width=80)

LAYERS = [("trackfit", "track fit"), ("shower", "shower pts"),
          ("track", "track pts"), ("steiner", "steiner"),
          ("terminals", "terminals"), ("vertices", "vertices"),
          ("dead", "dead (2-D)")]
# Steiner off by default: 6k points per event drawn under everything else is
# noise until you go looking for it.
LAYER_DEFAULT = [0, 1, 2, 5, 6]
layer_group = CheckboxButtonGroup(labels=[l for _, l in LAYERS],
                                  active=list(LAYER_DEFAULT))

zoom_btn = Toggle(label="zoom", active=False, width=90)
cx_in = TextInput(title="centre x (cm)", value="", width=110)
cy_in = TextInput(title="centre y (cm)", value="", width=110)
cz_in = TextInput(title="centre z (cm)", value="", width=110)
half_in = TextInput(title="half-width (cm)", value=str(DEFAULT_HALF), width=110)
vtx_btn = Button(label="centre on vertex", width=140)

status = Div(text="", width=1400)
info = Div(text="", width=1400)


def toggle_layers(attr, old, new):
    on = {LAYERS[i][0] for i in layer_group.active}
    for name, rs in RENDER.items():
        # the 2-D fit follows the 3-D "track fit" toggle
        key = "trackfit" if name == "trackfit2d" else name
        for r in rs:
            r.visible = key in on


layer_group.on_change("active", toggle_layers)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def seg_color(i):
    return Category20_20[i % 20]


def load(label):
    """Read one event's calib JSON and push every layer to its CDS."""
    path = EVENTS[label]
    with open(path) as fh:
        d = json.load(fh)
    state["label"] = label
    state["data"] = d

    meta = d["meta"]
    # (apa, face) -> ticks per slice, for pt -> slice
    nps = {(r["apa"], r["face"]): r["nticks_per_slice"]
           for r in meta.get("nticks_per_slice", [])}

    # --- detector box -------------------------------------------------------
    (xl, xh), (yl, yh), (zl, zh) = DET_BOX["x"], DET_BOX["y"], DET_BOX["z"]
    det_src.data = dict(
        xs_xy=[[xl, xh, xh, xl, xl], [0, 0]], ys_xy=[[yl, yl, yh, yh, yl], [yl, yh]],
        xs_yz=[[zl, zh, zh, zl, zl], []],     ys_yz=[[yl, yl, yh, yh, yl], []],
        xs_xz=[[xl, xh, xh, xl, xl], [0, 0]], ys_xz=[[zl, zl, zh, zh, zl], [zl, zh]])

    # --- associated points, split track / shower ----------------------------
    ts = d.get("track_shower", {})
    tx, ty, tz = ts.get("x", []), ts.get("y", []), ts.get("z", [])
    fl = ts.get("flag_shower", [])
    pid = ts.get("particle_id", [])
    sh = [i for i, f in enumerate(fl) if f]
    tr = [i for i, f in enumerate(fl) if not f]

    def pick(idx):
        return dict(x=[tx[i] for i in idx], y=[ty[i] for i in idx],
                    z=[tz[i] for i in idx], c=[""] * len(idx),
                    tag=[pid[i] if i < len(pid) else -1 for i in idx])

    shower_src.data = pick(sh)
    track_src.data = pick(tr)

    # --- steiner skeleton + its terminal subset -----------------------------
    sx, sy, sz, stag = [], [], [], []
    ux, uy, uz, utag = [], [], [], []
    for cl in d.get("steiner", []):
        ft = cl.get("flag_terminal") or []
        for i in range(len(cl["x"])):
            sx.append(cl["x"][i]); sy.append(cl["y"][i]); sz.append(cl["z"][i])
            stag.append(cl["cluster_id"])
            if i < len(ft) and ft[i]:
                ux.append(cl["x"][i]); uy.append(cl["y"][i]); uz.append(cl["z"][i])
                utag.append(cl["cluster_id"])
    steiner_src.data = dict(x=sx, y=sy, z=sz, c=[""] * len(sx), tag=stag)
    term_src.data = dict(x=ux, y=uy, z=uz, c=[""] * len(ux), tag=utag)

    # --- PR graph vertices --------------------------------------------------
    vx, vy, vz, vt = [], [], [], []
    for v in d.get("vertices", []):
        if v.get("is_main"):
            continue
        vx.append(v["fit"]["x"]); vy.append(v["fit"]["y"]); vz.append(v["fit"]["z"])
        vt.append(v["id"])
    vtx_src.data = dict(x=vx, y=vy, z=vz, c=[""] * len(vx), tag=vt)

    mv = d.get("main_vertex")
    if mv:
        mainvtx_src.data = dict(x=[mv["x"]], y=[mv["y"]], z=[mv["z"]],
                                c=[""], tag=[mv.get("cluster_id", -1)])
    else:
        mainvtx_src.data = dict(EMPTY3)

    # --- segments as polylines, 3-D ----------------------------------------
    segs = d.get("segments", [])
    cols = {k: dict(xs=[], ys=[], c=[], sid=[], pid=[], cid=[], shower=[])
            for k in ("xy", "yz", "xz")}
    for i, s in enumerate(segs):
        col = seg_color(i)
        px = [p["x"] for p in s["points"]]
        py = [p["y"] for p in s["points"]]
        pz = [p["z"] for p in s["points"]]
        for key, a, b in (("xy", px, py), ("yz", pz, py), ("xz", px, pz)):
            c = cols[key]
            c["xs"].append(a); c["ys"].append(b); c["c"].append(col)
            c["sid"].append(s["id"]); c["pid"].append(s["particle_id"])
            c["cid"].append(s["cluster_id"])
            c["shower"].append("yes" if s["flag_shower"] else "no")
    for key in cols:
        seg_src[key].data = cols[key]

    # --- 2-D panels ---------------------------------------------------------
    have = set()
    qmax = 1.0
    for p in d.get("proj", []):
        key = (p["apa"], p["plane"])
        if key not in panel:
            continue
        have.add(key)
        panel[key]["cell"].data = dict(w=p["wire"], s=p["slice"], q=p["charge"],
                                       qp=p["charge_pred"], cid=p["cluster_id"])
        if p["charge"]:
            qmax = max(qmax, float(np.percentile(p["charge"], 99)))
    for key in panel:
        if key not in have:
            panel[key]["cell"].data = dict(w=[], s=[], q=[], qp=[], cid=[])
    # 99th percentile, not the max: a handful of saturated cells otherwise
    # flatten every track to the bottom of the colour scale.
    CMAP.low, CMAP.high = 0.0, qmax

    # fitted trajectories, per (apa, plane), in the panel's own coordinates
    fitcols = {k: dict(xs=[], ys=[], c=[], sid=[]) for k in panel}
    for i, s in enumerate(segs):
        col = seg_color(i)
        # A segment can cross APAs; split its points by the (apa, face) each
        # was fitted in, and drop points with no recorded (apa,face) (-1) --
        # drawing those on APA 0 is exactly the overlay bug doc pr/3 fixed.
        runs = defaultdict(lambda: ([], [], []))
        for p in s["points"]:
            a, fc = p["apa"], p["face"]
            if a < 0:
                continue
            n = nps.get((a, fc), 1)
            u, v, w = runs[a]
            u.append((p["pu"], p["pt"] / n))
            v.append((p["pv"], p["pt"] / n))
            w.append((p["pw"], p["pt"] / n))
        for a, (u, v, w) in runs.items():
            for pl, pts in ((0, u), (1, v), (2, w)):
                key = (a, pl)
                if key not in fitcols or len(pts) < 2:
                    continue
                fitcols[key]["xs"].append([q[0] for q in pts])
                fitcols[key]["ys"].append([q[1] for q in pts])
                fitcols[key]["c"].append(col)
                fitcols[key]["sid"].append(s["id"])
    for key in panel:
        panel[key]["fit"].data = fitcols[key]

    # dead bands, in SLICE units (the dump carries both; s0/s1 are what the
    # cells above are keyed on)
    deadcols = {k: dict(w=[], s=[], h=[]) for k in panel}
    for dd in d.get("dead", []):
        key = (dd["apa"], dd["plane"])
        if key not in deadcols:
            continue
        s0 = dd.get("s0", dd["t0"])
        s1 = dd.get("s1", dd["t1"])
        deadcols[key]["w"].append(dd["wire"])
        deadcols[key]["s"].append(0.5 * (s0 + s1))
        deadcols[key]["h"].append(max(1.0, s1 - s0))
    for key in panel:
        panel[key]["dead"].data = deadcols[key]

    # --- centre + summary ---------------------------------------------------
    if mv:
        set_centre(mv["x"], mv["y"], mv["z"])
    else:
        set_centre(0.0, 0.0, 250.0)

    nsh = len(sh)
    ncell = sum(len(p["wire"]) for p in d.get("proj", []))
    nterm = len(ux)
    info.text = (
        "<b>run %s / subrun %s / event %s</b> &nbsp;|&nbsp; "
        "segments <b>%d</b> (%d fit points) &nbsp;|&nbsp; vertices <b>%d</b> "
        "&nbsp;|&nbsp; associated points <b>%d</b> (%d shower / %d track) "
        "&nbsp;|&nbsp; steiner <b>%d</b> points, %d terminals "
        "&nbsp;|&nbsp; 2-D cells <b>%d</b> &nbsp;|&nbsp; "
        "neutrino vertex %s"
        % (meta.get("runNo"), meta.get("subRunNo"), meta.get("eventNo"),
           len(segs), sum(len(s["points"]) for s in segs),
           len(d.get("vertices", [])), len(tx), nsh, len(tx) - nsh,
           len(sx), nterm, ncell,
           ("(%.1f, %.1f, %.1f) cm, cluster %s"
            % (mv["x"], mv["y"], mv["z"], mv.get("cluster_id"))) if mv
           else "<i>none found</i>"))
    status.text = "<code>%s</code>" % path
    apply_ranges()


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------
def set_centre(x, y, z):
    cx_in.value = "%.1f" % x
    cy_in.value = "%.1f" % y
    cz_in.value = "%.1f" % z


def centre():
    def num(w, dflt):
        try:
            return float(w.value)
        except (TypeError, ValueError):
            return dflt
    return num(cx_in, 0.0), num(cy_in, 0.0), num(cz_in, 250.0)


def half():
    try:
        h = float(half_in.value)
        return h if h > 0 else DEFAULT_HALF
    except (TypeError, ValueError):
        return DEFAULT_HALF


def apply_ranges():
    """Frame all nine panels: full detector, or +-half around the centre."""
    cx, cy, cz = centre()
    h = half()
    if zoom_btn.active:
        rng = {"x": (cx - h, cx + h), "y": (cy - h, cy + h), "z": (cz - h, cz + h)}
    else:
        rng = {k: DET_BOX[k] for k in "xyz"}
    for f, hx, hy in PROJ:
        f.x_range.start, f.x_range.end = rng[hx]
        f.y_range.start, f.y_range.end = rng[hy]

    # 2-D panels.  There is no closed-form (x,y,z) -> (wire, slice) here (that
    # needs the wire geometry, which this display deliberately does not load),
    # so derive the window from the FITTED POINTS near the centre: the ones
    # inside the same 3-D sphere define the wire/slice span to show.  Falls
    # back to the panel's full extent when nothing is near.
    d = state.get("data") or {}
    for (apa, pl), P in panel.items():
        cellw, cells = P["cell"].data["w"], P["cell"].data["s"]
        if not len(cellw):
            continue
        if not zoom_btn.active:
            P["fig"].x_range.start, P["fig"].x_range.end = min(cellw) - 5, max(cellw) + 5
            P["fig"].y_range.start, P["fig"].y_range.end = min(cells) - 5, max(cells) + 5
            continue
        ws, ss = [], []
        key = ("pu", "pv", "pw")[pl]
        nps = {(r["apa"], r["face"]): r["nticks_per_slice"]
               for r in d.get("meta", {}).get("nticks_per_slice", [])}
        for s in d.get("segments", []):
            for p in s["points"]:
                if p["apa"] != apa:
                    continue
                if (abs(p["x"] - cx) > h or abs(p["y"] - cy) > h
                        or abs(p["z"] - cz) > h):
                    continue
                ws.append(p[key])
                ss.append(p["pt"] / nps.get((apa, p["face"]), 1))
        if len(ws) >= 2:
            pad = 10
            P["fig"].x_range.start, P["fig"].x_range.end = min(ws) - pad, max(ws) + pad
            P["fig"].y_range.start, P["fig"].y_range.end = min(ss) - pad, max(ss) + pad
        else:
            P["fig"].x_range.start, P["fig"].x_range.end = min(cellw) - 5, max(cellw) + 5
            P["fig"].y_range.start, P["fig"].y_range.end = min(cells) - 5, max(cells) + 5


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_event(attr, old, new):
    load(new)


def step(delta):
    def _cb():
        if not LABELS:
            return
        i = (LABELS.index(event_select.value) + delta) % len(LABELS)
        event_select.value = LABELS[i]
    return _cb


def on_zoom(attr, old, new):
    apply_ranges()


def on_centre(attr, old, new):
    if zoom_btn.active:
        apply_ranges()


def on_vertex():
    mv = (state.get("data") or {}).get("main_vertex")
    if mv:
        set_centre(mv["x"], mv["y"], mv["z"])
        apply_ranges()


event_select.on_change("value", on_event)
prev_btn.on_click(step(-1))
next_btn.on_click(step(+1))
zoom_btn.on_change("active", on_zoom)
for w in (cx_in, cy_in, cz_in, half_in):
    w.on_change("value", on_centre)
vtx_btn.on_click(on_vertex)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
header = Div(text="<h2>SBND PR event display</h2>", width=1400)
controls = row(event_select, prev_btn, next_btn, Spacer(width=20),
               zoom_btn, cx_in, cy_in, cz_in, half_in, vtx_btn)

layout = column(
    header,
    row(f_xy, f_yz, f_xz),
    row(layer_group),
    controls,
    info,
    row(column(panel[(0, 0)]["fig"], panel[(0, 1)]["fig"], panel[(0, 2)]["fig"]),
        column(panel[(1, 0)]["fig"], panel[(1, 1)]["fig"], panel[(1, 2)]["fig"]),
        cbar_fig),
    status,
)

curdoc().add_root(layout)
curdoc().title = "SBND PR display"

toggle_layers(None, None, None)
if LABELS:
    load(LABELS[0])
else:
    status.text = ("<b>No calib-pr-evt*.json found.</b> Produce one with "
                   "<code>PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh "
                   "&lt;ql_root&gt; &lt;out&gt; data &lt;evt&gt;</code>, then pass "
                   "<code>&lt;out&gt;/pr_evt*/calib-pr-evt*.json</code> to "
                   "serve_pr_display.sh.")

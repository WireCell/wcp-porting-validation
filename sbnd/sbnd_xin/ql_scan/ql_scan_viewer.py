#!/usr/bin/env python3
"""SBND Q/L matching hand-scan event display (Bokeh server).

Reads the per-event calibration dumps written by QLMatching (run_ql_evt.sh -calib
-> work/ql_evt<ID>/calib-evt<ID>.json), one file per event holding BOTH TPCs:
every candidate (flash, cluster) bundle with its predicted vs measured light, the
matching metrics (ks/chi2/ndf/strength) and flags, the cluster geometry and the
detector box.

The tool lets a human pick the correct flash<->cluster match per cluster and saves
those labels (Save -> work/ql_evt<ID>/labels-evt<ID>.json) for later tuning of the
QLMatching chi2 / metric parameters and the PE_err model.

Selection rules (enforced live):
  * each cluster matches at most one flash (selecting a bundle drops the cluster's
    other candidate bundles);
  * one flash may match several clusters -> the predicted light shown for that flash
    is the element-wise SUM of the selected bundles' predictions (measured is the
    flash's own, unchanged);
  * both TPCs are shown together; once a flash is picked on one TPC, only the
    opposite-TPC bundles whose flash is in +-80 ns coincidence (same group id) stay
    available.

Launched by serve_ql_scan.sh; mirrors pdvd/sp_plot/filter_tune_viewer.py
(server-side callbacks, ColumnDataSource per figure).
"""

import sys
import os
import glob
import json
import math
from collections import defaultdict

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, DataTable, TableColumn, Select, Button,
                          Div, ColorBar, HoverTool, NumberFormatter)
from bokeh.transform import linear_cmap
from bokeh.plotting import figure

# Max charge points drawn per projection panel (stride-downsample above this).
MAX_DRAW_PTS = 5000

# ---------------------------------------------------------------------------
# Inputs: calib JSON paths from --args (globs already expanded by the shell).
# ---------------------------------------------------------------------------
def discover_files(argv):
    files = []
    for a in argv:
        if os.path.isdir(a):
            files += glob.glob(os.path.join(a, "calib-evt*.json"))
        elif any(ch in a for ch in "*?[") :
            files += glob.glob(a)
        elif os.path.isfile(a):
            files.append(a)
    # de-dup, keep existing, sort by the integer event id when derivable
    # (so evt2 < evt686 < evt1258, not lexicographic).
    def keyf(f):
        lab = event_label(f)
        digits = "".join(ch for ch in lab if ch.isdigit())
        return (int(digits) if digits else 0, f)
    return sorted(set(f for f in files if os.path.isfile(f)), key=keyf)


def event_label(path):
    base = os.path.basename(path)
    # calib-evt<ID>.json -> evt<ID>
    if base.startswith("calib-") and base.endswith(".json"):
        return base[len("calib-"):-len(".json")]
    return base


FILES = discover_files(sys.argv[1:])
LABELS = [event_label(f) for f in FILES]
FILE_OF = dict(zip(LABELS, FILES))


# ---------------------------------------------------------------------------
# Per-event model: load one calib JSON and build lookup indexes.
# ---------------------------------------------------------------------------
class Event:
    def __init__(self, path):
        with open(path) as fh:
            self.d = json.load(fh)
        self.path = path
        self.nchan = self.d["nchan"]
        self.drift_speed = self.d["drift_speed"]          # cm/us
        self.geom = {int(k): v for k, v in self.d["geometry"].items()}
        self.flash_by_gid = {f["gid"]: f for f in self.d["flashes"]}
        self.cluster_by_uid = {c["uid"]: c for c in self.d["clusters"]}
        self.bundles = self.d["bundles"]
        # opdet arrays (numpy) for fast light-pattern drawing
        od = self.d["opdets"]
        self.od_x = np.array([o["x"] for o in od])
        self.od_y = np.array([o["y"] for o in od])
        self.od_z = np.array([o["z"] for o in od])
        self.od_apa = np.array([o["apa"] for o in od])
        self.od_active = np.array([o["active"] for o in od], dtype=bool)
        self.qp = self.d.get("quality_params", {})

    def group_of(self, gid):
        return self.flash_by_gid[gid]["group"]

    def dx_cm(self, apa, gid):
        """T0 x-shift (cm) applied to apa's charge for the flash at gid."""
        g = self.geom[apa]
        t_us = self.flash_by_gid[gid]["time"]
        return g["sign_offset"] * t_us * self.drift_speed


# ---------------------------------------------------------------------------
# Global UI state.
# ---------------------------------------------------------------------------
state = {
    "evt": None,        # Event
    "focus": None,      # focused bundle index (into evt.bundles), or None
    "selected": set(),  # set of selected bundle indices (GLOBAL, all groups)
    "order": [],        # visible bundle indices in table display order
    "group": None,      # current coincidence group id (the scan unit)
    "groups": [],       # sorted group ids present in the event
}


# ----- coincidence-group helpers -------------------------------------------
def event_groups(evt):
    # Only groups that actually have a (contained) bundle to scan — after the
    # contained-only dump many coincidence groups hold no candidate, and paging
    # through empty stops would defeat the de-clutter goal.
    return sorted({evt.group_of(b["flash_gid"]) for b in evt.bundles})


def group_label(evt, g):
    """'grp G  (T0:.. T1:.. us)' — the group's per-TPC flash times, for orienting."""
    t = {0: None, 1: None}
    for f in evt.flash_by_gid.values():
        if f["group"] == g and t[f["apa"]] is None:
            t[f["apa"]] = f["time"]
    s = lambda x: ("%.1f" % x) if x is not None else "-"
    return "grp %d  (T0:%s T1:%s us)" % (g, s(t[0]), s(t[1]))


def group_flash_gid(apa):
    """The flash gid for TPC `apa` in the current group (None if none). When a
    group holds >1 flash on a TPC, prefer the focused bundle's flash, else first."""
    evt = state["evt"]
    g = state["group"]
    gids = sorted(gid for gid, f in evt.flash_by_gid.items()
                  if f["group"] == g and f["apa"] == apa)
    if not gids:
        return None
    idx = state["focus"]
    if idx is not None:
        b = evt.bundles[idx]
        if b["apa"] == apa and b["flash_gid"] in gids:
            return b["flash_gid"]
    return gids[0]


# ----- selection-rule helpers ----------------------------------------------
def cluster_eliminated(idx):
    """True if bundle idx's main cluster is already claimed by a *different*
    selected bundle (one cluster -> one flash). Applies across all groups."""
    evt = state["evt"]
    cu = evt.bundles[idx]["main_cluster"]
    return any(j != idx and evt.bundles[j]["main_cluster"] == cu
               for j in state["selected"])


def toggle_select(idx):
    evt = state["evt"]
    if idx in state["selected"]:
        state["selected"].discard(idx)
        return True, "deselected"
    # cluster uniqueness: selecting replaces any prior bundle for the same cluster
    b = evt.bundles[idx]
    dropped = [j for j in state["selected"]
               if evt.bundles[j]["main_cluster"] == b["main_cluster"]]
    for j in dropped:
        state["selected"].discard(j)
    state["selected"].add(idx)
    msg = "selected"
    if dropped:
        msg += " (replaced this cluster's previous match)"
    return True, msg


# ---------------------------------------------------------------------------
# Bokeh widgets / figures.
# ---------------------------------------------------------------------------
event_select = Select(title="Event", value=(LABELS[0] if LABELS else ""),
                      options=LABELS, width=180)
prev_evt_btn = Button(label="< prev evt", width=90)
next_evt_btn = Button(label="next evt >", width=90)
group_select = Select(title="coincidence group", value="", options=[], width=240)
prev_grp_btn = Button(label="< prev grp", width=90)
next_grp_btn = Button(label="next grp >", width=90)
select_btn = Button(label="Toggle match (focused)", button_type="primary", width=200)
clear_btn = Button(label="Clear selections", width=140)
save_btn = Button(label="Save labels", button_type="success", width=120)
status = Div(text="", width=1100)
metrics = Div(text="", width=560)
selsummary = Div(text="", width=540)

table_src = ColumnDataSource(data=dict())
table_cols = [
    TableColumn(field="state", title="state", width=90),
    TableColumn(field="auto", title="auto", width=45),
    TableColumn(field="apa", title="apa", width=35),
    TableColumn(field="flash_gid", title="flash", width=70),
    TableColumn(field="t_us", title="t(us)", width=70,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="grp", title="grp", width=40),
    TableColumn(field="cluster", title="clus", width=50),
    TableColumn(field="noth", title="+oth", width=40),
    TableColumn(field="ks", title="ks", width=55, formatter=NumberFormatter(format="0.000")),
    TableColumn(field="chi2ndf", title="chi2/ndf", width=70,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="strength", title="strength", width=70,
                formatter=NumberFormatter(format="0.000")),
    TableColumn(field="meas", title="measPE", width=70, formatter=NumberFormatter(format="0")),
    TableColumn(field="pred", title="predPE", width=70, formatter=NumberFormatter(format="0.0")),
    TableColumn(field="flags", title="flags", width=150),
]
table = DataTable(source=table_src, columns=table_cols, width=1100, height=300,
                  selectable=True, index_position=None)

# Light-pattern figures (Y vertical, Z horizontal): a 2x2 grid of measured vs
# predicted for TPC0 and TPC1. Positions are fixed, so the ranges are pinned to the
# detector box (no zoom). Active PMTs of the panel's TPC are faint outlines; the
# flash PMTs are sized by sqrt(PE).
def radii(vals, hi):
    # circle radius in cm, scaled by sqrt(PE/hi); 0-PE channels get a tiny dot
    return (1.5 + 6.0 * np.sqrt(np.clip(vals, 0, None) / hi)).tolist()


def make_light_fig(title):
    f = figure(title=title, height=260, width=420, tools="pan,reset,save")
    f.xaxis.axis_label = "z (cm)"
    f.yaxis.axis_label = "y (cm)"
    base = ColumnDataSource(data=dict(z=[], y=[]))
    src = ColumnDataSource(data=dict(z=[], y=[], pe=[], r=[]))
    f.scatter("z", "y", source=base, marker="circle", size=6,
              fill_color=None, line_color="#cccccc")
    g = f.circle("z", "y", source=src, radius="r",
                 fill_color=linear_cmap("pe", "Viridis256", 0, 1),
                 line_color="#333333", fill_alpha=0.85)
    f.add_layout(ColorBar(color_mapper=g.glyph.fill_color["transform"], title="PE"), "right")
    f.add_tools(HoverTool(renderers=[g], tooltips=[("PE", "@pe{0.0}")]))
    return dict(fig=f, base=base, src=src, glyph=g)


# LIGHT[apa] = {"meas": panel, "pred": panel}
LIGHT = {apa: {"meas": make_light_fig("TPC%d measured" % apa),
               "pred": make_light_fig("TPC%d predicted" % apa)}
         for apa in (0, 1)}

# Charge-projection figures (focused bundle's clusters, T0-shifted, in the fixed
# detector box; both TPC boxes drawn). XY, YZ (z horiz, y vert), XZ (x horiz, z vert).
proj_kw = dict(height=320, tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom")
f_xy = figure(title="X-Y", width=380, **proj_kw)
f_yz = figure(title="Y-Z", width=380, **proj_kw)
f_xz = figure(title="X-Z", width=380, **proj_kw)
f_xy.xaxis.axis_label, f_xy.yaxis.axis_label = "x (cm)", "y (cm)"
f_yz.xaxis.axis_label, f_yz.yaxis.axis_label = "z (cm)", "y (cm)"
f_xz.xaxis.axis_label, f_xz.yaxis.axis_label = "x (cm)", "z (cm)"

# focus cluster points, other selected clusters (context), and box outlines
foc_src = ColumnDataSource(data=dict(x=[], y=[], z=[]))
ctx_src = ColumnDataSource(data=dict(x=[], y=[], z=[]))
box_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                     xs_xz=[], ys_xz=[]))
f_xy.multi_line(xs="xs_xy", ys="ys_xy", source=box_src, line_color="#cc4444", line_width=1)
f_yz.multi_line(xs="xs_yz", ys="ys_yz", source=box_src, line_color="#cc4444", line_width=1)
f_xz.multi_line(xs="xs_xz", ys="ys_xz", source=box_src, line_color="#cc4444", line_width=1)
for f, hx, hy in ((f_xy, "x", "y"), (f_yz, "z", "y"), (f_xz, "x", "z")):
    f.scatter(hx, hy, source=ctx_src, marker="circle", size=2,
              fill_color="#bbbbbb", line_color=None, fill_alpha=0.4)
    f.scatter(hx, hy, source=foc_src, marker="circle", size=3,
              fill_color="#1f77b4", line_color=None, fill_alpha=0.7)


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def fmt_flags(b):
    parts = []
    if b["consistent"]:          parts.append("consist")
    if b["potential_bad_match"]: parts.append("badmatch")
    if b["close_to_PMT"]:        parts.append("nearPMT")
    if b["at_x_boundary"]:       parts.append("xbound")
    if b["spec_end"]:            parts.append("specend")
    if b["window_truncated"]:    parts.append("wtrunc")
    if not b["contained"]:       parts.append("UNCONTAINED")
    return ",".join(parts)


def rebuild_table():
    evt = state["evt"]
    g = state["group"]
    # Visible = bundles of the current coincidence group whose cluster is not
    # already claimed by another selected bundle (elimination spans all groups).
    visible = [i for i in range(len(evt.bundles))
               if evt.group_of(evt.bundles[i]["flash_gid"]) == g
               and (i in state["selected"] or not cluster_eliminated(i))]
    order = sorted(visible,
                   key=lambda i: (evt.bundles[i]["apa"],
                                  evt.bundles[i]["flash_gid"],
                                  evt.bundles[i]["main_cluster"]))
    state["order"] = order
    cols = defaultdict(list)
    for i in order:
        b = evt.bundles[i]
        ndf = b["ndf"] or 1
        cols["state"].append("SELECTED" if i in state["selected"] else "avail")
        cols["auto"].append("Y" if b["auto_selected"] else "")
        cols["apa"].append(b["apa"])
        cols["flash_gid"].append(b["flash_gid"])
        cols["t_us"].append(evt.flash_by_gid[b["flash_gid"]]["time"])
        cols["grp"].append(evt.group_of(b["flash_gid"]))
        cols["cluster"].append(evt.cluster_by_uid[b["main_cluster"]]["ident"])
        cols["noth"].append(len(b["other_clusters"]))
        cols["ks"].append(b["ks_dis"])
        cols["chi2ndf"].append(b["chi2"] / ndf)
        cols["strength"].append(b["strength"])
        cols["meas"].append(b["total_PE"])
        cols["pred"].append(b["total_pred_light"])
        cols["flags"].append(fmt_flags(b))
    table_src.data = dict(cols)
    # keep focus row highlighted
    if state["focus"] is not None and state["focus"] in order:
        table_src.selected.indices = [order.index(state["focus"])]
    else:
        table_src.selected.indices = []


def cluster_points(uid, apa, dx):
    c = state["evt"].cluster_by_uid[uid]
    x = np.array(c["x"]) + dx
    y = np.array(c["y"])
    z = np.array(c["z"])
    return x, y, z


def downsample(x, y, z):
    n = len(x)
    if n > MAX_DRAW_PTS:
        s = int(math.ceil(n / MAX_DRAW_PTS))
        return x[::s], y[::s], z[::s]
    return x, y, z


def render_light():
    """Four panels: measured + predicted for TPC0 and TPC1. Measured is anchored
    to the current group's per-TPC flash (a stable reference as rows are clicked);
    predicted sums the selected bundles on that flash (or previews the focus)."""
    evt = state["evt"]
    for apa in (0, 1):
        mp, pp = LIGHT[apa]["meas"], LIGHT[apa]["pred"]
        am = evt.od_active & (evt.od_apa == apa)
        chans = np.nonzero(am)[0]
        # faint outline of this TPC's active PMTs
        mp["base"].data = dict(z=evt.od_z[am].tolist(), y=evt.od_y[am].tolist())
        pp["base"].data = dict(z=evt.od_z[am].tolist(), y=evt.od_y[am].tolist())

        gid = group_flash_gid(apa)
        if gid is None:
            mp["src"].data = dict(z=[], y=[], pe=[], r=[])
            pp["src"].data = dict(z=[], y=[], pe=[], r=[])
            mp["fig"].title.text = "TPC%d measured  (no flash in group)" % apa
            pp["fig"].title.text = "TPC%d predicted" % apa
            continue
        flash = evt.flash_by_gid[gid]
        meas = np.array(flash["pe"])[chans]

        # predicted = sum over SELECTED bundles on THIS TPC's group-flash; if none,
        # preview the focused bundle when it sits on this flash. Never cross-TPC.
        share = [j for j in state["selected"]
                 if evt.bundles[j]["apa"] == apa and evt.bundles[j]["flash_gid"] == gid]
        previewed = False
        if not share:
            idx = state["focus"]
            if (idx is not None and evt.bundles[idx]["apa"] == apa
                    and evt.bundles[idx]["flash_gid"] == gid):
                share = [idx]
                previewed = True
        pred_full = np.zeros(evt.nchan)
        for j in share:
            pj = evt.bundles[j]["pred_pe"]
            if pj:
                pred_full += np.array(pj)
        pred = pred_full[chans]

        # Independent per-panel scales so the predicted *shape* stays readable even
        # when its absolute PE is tiny next to a bright measured flash.
        hi_meas = max(1.0, float(meas.max() if meas.size else 0))
        hi_pred = max(1.0, float(pred.max() if pred.size else 0))
        mp["glyph"].glyph.fill_color["transform"].high = hi_meas
        pp["glyph"].glyph.fill_color["transform"].high = hi_pred

        z = evt.od_z[chans].tolist()
        y = evt.od_y[chans].tolist()
        mp["src"].data = dict(z=z, y=y, pe=meas.tolist(), r=radii(meas, hi_meas))
        pp["src"].data = dict(z=z, y=y, pe=pred.tolist(), r=radii(pred, hi_pred))
        mp["fig"].title.text = ("TPC%d measured  gid %d (id %d)  t=%.1f us  totPE=%.0f"
                                % (apa, gid, flash["id"], flash["time"], flash["total_PE"]))
        if not share:
            lab = "none selected"
        elif previewed:
            lab = "preview: focused bundle"
        else:
            lab = "sum of %d selected cluster(s)" % len(share)
        pp["fig"].title.text = "TPC%d predicted  (%s)" % (apa, lab)


def box_lines(g):
    """Return the rectangle polylines for one TPC box in the three projections."""
    ax, cx = g["anode_x"], g["cathode_x"]
    ylo, yhi, zlo, zhi = g["y_lo"], g["y_hi"], g["z_lo"], g["z_hi"]
    xy = ([ax, cx, cx, ax, ax], [ylo, ylo, yhi, yhi, ylo])      # x-y
    yz = ([zlo, zhi, zhi, zlo, zlo], [ylo, ylo, yhi, yhi, ylo])  # z-y
    xz = ([ax, cx, cx, ax, ax], [zlo, zlo, zhi, zhi, zlo])       # x-z
    return xy, yz, xz


def render_projections():
    evt = state["evt"]
    # both TPC boxes always drawn (the fixed detector frame)
    xs_xy, ys_xy, xs_yz, ys_yz, xs_xz, ys_xz = [], [], [], [], [], []
    for apa in sorted(evt.geom):
        xy, yz, xz = box_lines(evt.geom[apa])
        xs_xy.append(xy[0]); ys_xy.append(xy[1])
        xs_yz.append(yz[0]); ys_yz.append(yz[1])
        xs_xz.append(xz[0]); ys_xz.append(xz[1])
    box_src.data = dict(xs_xy=xs_xy, ys_xy=ys_xy, xs_yz=xs_yz, ys_yz=ys_yz,
                        xs_xz=xs_xz, ys_xz=ys_xz)

    # context: all currently-selected matches (both TPCs), T0-shifted
    cx, cy, cz = [], [], []
    for j in state["selected"]:
        b = evt.bundles[j]
        dx = evt.dx_cm(b["apa"], b["flash_gid"])
        for uid in [b["main_cluster"]] + b["other_clusters"]:
            x, y, z = cluster_points(uid, b["apa"], dx)
            x, y, z = downsample(x, y, z)
            cx += x.tolist(); cy += y.tolist(); cz += z.tolist()
    ctx_src.data = dict(x=cx, y=cy, z=cz)

    # focus bundle clusters, T0-shifted
    idx = state["focus"]
    fx, fy, fz = [], [], []
    if idx is not None:
        b = evt.bundles[idx]
        dx = evt.dx_cm(b["apa"], b["flash_gid"])
        for uid in [b["main_cluster"]] + b["other_clusters"]:
            x, y, z = cluster_points(uid, b["apa"], dx)
            x, y, z = downsample(x, y, z)
            fx += x.tolist(); fy += y.tolist(); fz += z.tolist()
    foc_src.data = dict(x=fx, y=fy, z=fz)


def render_metrics():
    evt = state["evt"]
    idx = state["focus"]
    if idx is None:
        metrics.text = "<i>select a bundle row to inspect</i>"
        return
    b = evt.bundles[idx]
    qp = evt.qp
    ndf = b["ndf"] or 1
    others = ", ".join(str(evt.cluster_by_uid[u]["ident"]) for u in b["other_clusters"]) or "-"
    rows = [
        ("flash", "gid %d / id %d / TPC %d / group %d"
         % (b["flash_gid"], b["flash_id"], b["apa"], evt.group_of(b["flash_gid"]))),
        ("flash time", "%.2f us" % evt.flash_by_gid[b["flash_gid"]]["time"]),
        ("main cluster", str(evt.cluster_by_uid[b["main_cluster"]]["ident"])),
        ("assoc clusters", others),
        ("ks_dis", "%.4f  (consist if &lt; %.3f)" % (b["ks_dis"], qp.get("highconsist_ks_max", 0))),
        ("chi2 / ndf", "%.1f / %d = %.2f" % (b["chi2"], b["ndf"], b["chi2"] / ndf)),
        ("ndf", "%d  (consist if &ge; %d)" % (b["ndf"], qp.get("highconsist_min_ndf", 0))),
        ("strength", "%.4f  (cutoff %.3f)" % (b["strength"], qp.get("strength_cutoff", 0))),
        ("measured PE", "%.0f" % b["total_PE"]),
        ("predicted PE", "%.1f" % b["total_pred_light"]),
        ("flag_high_consistent", str(b["consistent"])),
        ("contained", str(b["contained"])),
        ("flags", fmt_flags(b) or "-"),
        ("auto_selected", str(b["auto_selected"])),
        ("hand-selected", str(idx in state["selected"])),
    ]
    html = "<table style='font-size:12px'>" + "".join(
        "<tr><td style='color:#555;padding-right:8px'>%s</td><td><b>%s</b></td></tr>" % r
        for r in rows) + "</table>"
    metrics.text = html


def render_summary():
    evt = state["evt"]
    if not state["selected"]:
        selsummary.text = "<b>0 matches selected.</b>"
        return
    lines = []
    by_flash = defaultdict(list)
    for j in state["selected"]:
        by_flash[evt.bundles[j]["flash_gid"]].append(j)
    for gid in sorted(by_flash):
        f = evt.flash_by_gid[gid]
        cls = []
        for j in by_flash[gid]:
            b = evt.bundles[j]
            cls += [evt.cluster_by_uid[u]["ident"]
                    for u in [b["main_cluster"]] + b["other_clusters"]]
        lines.append("flash %d (TPC%d, t=%.1fus, grp%d) &larr; clusters %s"
                     % (gid, f["apa"], f["time"], f["group"], cls))
    selsummary.text = ("<b>%d selected match(es):</b><br>" % len(state["selected"])
                       + "<br>".join(lines))


def refresh():
    rebuild_table()
    render_light()
    render_projections()
    render_metrics()
    render_summary()


# ---------------------------------------------------------------------------
# Callbacks.
# ---------------------------------------------------------------------------
def set_light_ranges(evt):
    """Pin each light panel to its TPC's detector box (fixed, no zoom)."""
    for apa in (0, 1):
        g = evt.geom.get(apa)
        if not g:
            continue
        pz = 0.05 * (g["z_hi"] - g["z_lo"])
        py = 0.05 * (g["y_hi"] - g["y_lo"])
        for panel in (LIGHT[apa]["meas"], LIGHT[apa]["pred"]):
            f = panel["fig"]
            f.x_range.start, f.x_range.end = g["z_lo"] - pz, g["z_hi"] + pz
            f.y_range.start, f.y_range.end = g["y_lo"] - py, g["y_hi"] + py


def load_event(label):
    evt = Event(FILE_OF[label])
    state["evt"] = evt
    state["focus"] = None
    state["selected"] = set()
    groups = event_groups(evt)
    state["groups"] = groups
    state["group"] = groups[0] if groups else None
    set_light_ranges(evt)
    # Populate the group selector (setting .value may re-fire on_group_change,
    # which just re-refreshes — harmless).
    group_select.options = [(str(g), group_label(evt, g)) for g in groups]
    group_select.value = str(groups[0]) if groups else ""
    n = len(evt.bundles)
    nsel = sum(b["auto_selected"] for b in evt.bundles)
    status.text = ("Loaded <b>%s</b>: %d contained bundles, %d flashes, %d clusters, "
                   "%d coincidence groups; %d auto-selected. Pick a group, click a row "
                   "to inspect, 'Toggle match' to select."
                   % (label, n, len(evt.flash_by_gid), len(evt.cluster_by_uid),
                      len(groups), nsel))
    refresh()


def on_event_change(attr, old, new):
    if new:
        load_event(new)


def step_event(d):
    if not LABELS:
        return
    i = LABELS.index(event_select.value)
    event_select.value = LABELS[(i + d) % len(LABELS)]


def on_group_change(attr, old, new):
    if not new or state["evt"] is None:
        return
    state["group"] = int(new)
    state["focus"] = None          # never carry focus across groups
    refresh()


def step_group(d):
    groups = state["groups"]
    if not groups or state["group"] is None:
        return
    i = groups.index(state["group"])
    group_select.value = str(groups[(i + d) % len(groups)])


def on_row_select(attr, old, new):
    if not new:
        return
    order = state["order"]
    pos = new[0]
    if 0 <= pos < len(order):
        state["focus"] = order[pos]
        render_light()
        render_projections()
        render_metrics()


def on_toggle():
    if state["focus"] is None:
        status.text = "Select a bundle row first."
        return
    ok, msg = toggle_select(state["focus"])
    b = state["evt"].bundles[state["focus"]]
    status.text = ("bundle [flash %d, cluster %d]: %s"
                   % (b["flash_gid"], state["evt"].cluster_by_uid[b["main_cluster"]]["ident"], msg))
    refresh()


def on_clear():
    state["selected"] = set()
    status.text = "Cleared all selections."
    refresh()


def on_save():
    evt = state["evt"]
    if not state["selected"]:
        status.text = "Nothing selected to save."
        return
    out = {"event": event_select.value, "source": os.path.basename(evt.path),
           "matches": []}
    for j in sorted(state["selected"]):
        b = evt.bundles[j]
        f = evt.flash_by_gid[b["flash_gid"]]
        out["matches"].append({
            "flash_gid": b["flash_gid"], "flash_id": b["flash_id"],
            "flash_time_us": f["time"], "apa": b["apa"], "group": f["group"],
            "cluster_idents": [evt.cluster_by_uid[u]["ident"]
                               for u in [b["main_cluster"]] + b["other_clusters"]],
            "op_pes": f["pe"], "op_pe_err": f["pe_err"], "pred_pes": b["pred_pe"],
            "metrics": {"ks_dis": b["ks_dis"], "chi2": b["chi2"], "ndf": b["ndf"],
                        "strength": b["strength"],
                        "total_PE": b["total_PE"], "total_pred_light": b["total_pred_light"]},
            "flags": {k: b[k] for k in ("consistent", "potential_bad_match",
                                        "close_to_PMT", "at_x_boundary", "spec_end",
                                        "window_truncated", "contained", "auto_selected")},
        })
    dest = os.path.join(os.path.dirname(evt.path),
                        "labels-%s.json" % event_select.value)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    status.text = "Saved %d match(es) -> %s" % (len(out["matches"]), dest)


event_select.on_change("value", on_event_change)
prev_evt_btn.on_click(lambda: step_event(-1))
next_evt_btn.on_click(lambda: step_event(+1))
group_select.on_change("value", on_group_change)
prev_grp_btn.on_click(lambda: step_group(-1))
next_grp_btn.on_click(lambda: step_group(+1))
table_src.selected.on_change("indices", on_row_select)
select_btn.on_click(on_toggle)
clear_btn.on_click(on_clear)
save_btn.on_click(on_save)


# ---------------------------------------------------------------------------
# Layout.
# ---------------------------------------------------------------------------
controls = row(event_select, prev_evt_btn, next_evt_btn,
               group_select, prev_grp_btn, next_grp_btn,
               select_btn, clear_btn, save_btn)
header = Div(text="<h2>SBND Q/L matching hand-scan</h2>", width=1100)
layout = column(
    header,
    controls,
    status,
    table,
    row(metrics, selsummary),
    row(LIGHT[0]["meas"]["fig"], LIGHT[0]["pred"]["fig"],
        LIGHT[1]["meas"]["fig"], LIGHT[1]["pred"]["fig"]),
    row(f_xy, f_yz, f_xz),
)

curdoc().add_root(layout)
curdoc().title = "Q/L hand-scan"

if LABELS:
    load_event(LABELS[0])
else:
    status.text = ("<b>No calib JSONs found.</b> Produce them with "
                   "run_ql_evt.sh &lt;mode&gt; &lt;idx&gt; -calib, then pass the "
                   "glob to serve_ql_scan.sh.")

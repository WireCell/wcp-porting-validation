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
                          Div, ColorBar, HoverTool, NumberFormatter, BasicTicker,
                          NumeralTickFormatter, Span, HTMLTemplateFormatter,
                          CheckboxGroup, Toggle)
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
    "nav_groups": [],   # groups with >=1 visible bundle (prev/next cycle these)
    "compare_cluster": None,  # uid of the cluster shown in the compare table, or None
    "compare_order": [],      # bundle indices listed in the compare table
    "filter_on": False,       # lock: forbid selecting a cluster already matched
    "sel_snapshot": [],       # last pushed checkbox column (to detect user edits)
    "suppress_edit": False,   # guard while we (re)write the table data ourselves
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


def cluster_ident(idx):
    evt = state["evt"]
    return evt.cluster_by_uid[evt.bundles[idx]["main_cluster"]]["ident"]


def set_selected(idx, want):
    """Add/remove bundle idx from the global selection. Returns (changed, message).
    When the filter is on, a bundle whose cluster is already claimed by *another*
    selected bundle is locked — it cannot be turned on (but stays visible)."""
    if want and idx not in state["selected"]:
        if state["filter_on"] and cluster_eliminated(idx):
            return False, ("cluster %d is locked by the filter (already matched to "
                           "another flash)" % cluster_ident(idx))
        state["selected"].add(idx)
        return True, "selected"
    if (not want) and idx in state["selected"]:
        state["selected"].discard(idx)
        return True, "deselected"
    return False, ""


def toggle_select(idx):
    return set_selected(idx, idx not in state["selected"])


def visible_count(g):
    """Bundles shown for group g (the whole group — rivals are no longer hidden).
    Groups with at least one bundle stay in the navigation."""
    evt = state["evt"]
    return sum(1 for b in evt.bundles if evt.group_of(b["flash_gid"]) == g)


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
compare_btn = Button(label="Compare cluster's flashes", width=200)
filter_btn = Toggle(label="Filter selected bundles: OFF", width=220)
status = Div(text="", width=1100)
metrics = Div(text="", width=560)
selsummary = Div(text="", width=380)
compare_div = Div(text="", width=1000)

# read-only green check (string field) for the compare table
check_fmt = HTMLTemplateFormatter(
    template='<span style="color:#2ca02c;font-weight:bold;font-size:15px"><%= value %></span>')

# Selection is driven by a real CheckboxGroup (reliable clickable boxes) beside the
# table — one box per current-group bundle, in table-row order. Ticking a box adds the
# bundle to the selection (its predicted light joins the per-flash sum); tick several
# to combine clusters. A 🔒 in the label marks a bundle the filter forbids selecting.
sel_group = CheckboxGroup(labels=[], active=[], width=380)
sel_group_title = Div(text="<b>select matches</b> (tick to add to predicted sum)", width=380)

table_src = ColumnDataSource(data=dict())
table_cols = [
    TableColumn(field="row", title="#", width=30),
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
table = DataTable(source=table_src, columns=table_cols, width=900, height=300,
                  selectable=True, index_position=None)

# Second table (request 4): every bundle whose main cluster matches the focused
# bundle's, across all flashes/groups, so one cluster's candidate flashes can be
# compared side by side. Populated on demand by the "Compare" button; click a row
# to jump the whole view (focus + group) to that candidate flash.
compare_src = ColumnDataSource(data=dict())
compare_cols = [
    TableColumn(field="sel", title="✓", width=34, formatter=check_fmt),
    TableColumn(field="auto", title="auto", width=45),
    TableColumn(field="flash_gid", title="flash", width=70),
    TableColumn(field="t_us", title="t(us)", width=70,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="grp", title="grp", width=40),
    TableColumn(field="ks", title="ks", width=55, formatter=NumberFormatter(format="0.000")),
    TableColumn(field="chi2ndf", title="chi2/ndf", width=70,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="strength", title="strength", width=70,
                formatter=NumberFormatter(format="0.000")),
    TableColumn(field="meas", title="measPE", width=70, formatter=NumberFormatter(format="0")),
    TableColumn(field="pred", title="predPE", width=70, formatter=NumberFormatter(format="0.0")),
    TableColumn(field="flags", title="flags", width=150),
]
compare_table = DataTable(source=compare_src, columns=compare_cols, width=1000,
                          height=200, selectable=True, index_position=None)

# Light-pattern figures (Y vertical, Z horizontal): a 2x2 grid of measured vs
# predicted for TPC0 and TPC1. Positions are fixed, so the ranges are pinned to the
# detector box (no zoom). Active PMTs of the panel's TPC are faint outlines; the
# flash PMTs are sized by sqrt(PE).
def radii(vals, hi):
    # circle radius in cm, scaled by sqrt(PE/hi); 0-PE channels get a small dot
    return (2.5 + 9.0 * np.sqrt(np.clip(vals, 0, None) / hi)).tolist()


def make_light_fig(title):
    f = figure(title=title, height=280, width=430, tools="pan,reset,save")
    f.xaxis.axis_label = "z (cm)"
    f.yaxis.axis_label = "y (cm)"
    base = ColumnDataSource(data=dict(z=[], y=[]))
    src = ColumnDataSource(data=dict(z=[], y=[], pe=[], r=[]))
    f.scatter("z", "y", source=base, marker="circle", size=6,
              fill_color=None, line_color="#cccccc")
    g = f.circle("z", "y", source=src, radius="r",
                 fill_color=linear_cmap("pe", "Viridis256", 0, 1),
                 line_color="#333333", fill_alpha=0.85)
    cbar = ColorBar(color_mapper=g.glyph.fill_color["transform"], title="PE",
                    ticker=BasicTicker(desired_num_ticks=6),
                    formatter=NumeralTickFormatter(format="0,0"),
                    width=14, padding=4, label_standoff=6,
                    title_text_font_size="11px", title_standoff=6,
                    major_label_text_font_size="10px")
    f.add_layout(cbar, "right")
    f.add_tools(HoverTool(renderers=[g], tooltips=[("PE", "@pe{0,0.0}")]))
    return dict(fig=f, base=base, src=src, glyph=g)


# LIGHT[apa] = {"meas": panel, "pred": panel}
LIGHT = {apa: {"meas": make_light_fig("TPC%d measured" % apa),
               "pred": make_light_fig("TPC%d predicted" % apa)}
         for apa in (0, 1)}


# 1-D per-channel comparison below the 2x2 light grid: for each TPC an overlay of
# measured vs predicted PE over that TPC's active PMTs, and the pred/meas ratio.
def make_hist_fig(title):
    f = figure(title=title, height=220, width=430,
               tools="pan,box_zoom,wheel_zoom,reset,save")
    f.xaxis.axis_label = "active PMT (channel index)"
    f.yaxis.axis_label = "PE"
    meas_src = ColumnDataSource(data=dict(x=[], pe=[]))
    pred_src = ColumnDataSource(data=dict(x=[], pe=[]))
    f.vbar(x="x", top="pe", width=0.9, source=meas_src,
           fill_color="#6baed6", line_color=None, fill_alpha=0.6,
           legend_label="measured")
    f.line("x", "pe", source=pred_src, line_color="#d62728", line_width=1.5,
           legend_label="predicted")
    f.scatter("x", "pe", source=pred_src, marker="circle", size=4,
              fill_color="#d62728", line_color=None, legend_label="predicted")
    f.legend.label_text_font_size = "9px"
    f.legend.padding = 2
    f.legend.location = "top_right"
    f.legend.background_fill_alpha = 0.6
    return dict(fig=f, meas=meas_src, pred=pred_src)


def make_ratio_fig(title):
    f = figure(title=title, height=220, width=430,
               tools="pan,box_zoom,wheel_zoom,reset,save")
    f.xaxis.axis_label = "active PMT (channel index)"
    f.yaxis.axis_label = "pred / meas"
    src = ColumnDataSource(data=dict(x=[], ratio=[]))
    f.scatter("x", "ratio", source=src, marker="circle", size=5,
              fill_color="#2ca02c", line_color=None, fill_alpha=0.8)
    f.add_layout(Span(location=1.0, dimension="width", line_color="#888888",
                      line_dash="dashed", line_width=1))
    return dict(fig=f, src=src)


# HIST[apa] = {"overlay": panel, "ratio": panel}
HIST = {apa: {"overlay": make_hist_fig("TPC%d meas vs pred" % apa),
              "ratio": make_ratio_fig("TPC%d pred/meas" % apa)}
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
    # Show every bundle of the current coincidence group (rivals are no longer
    # hidden); when the filter is on, those whose cluster is already matched get a
    # lock glyph and their checkbox is refused.
    visible = [i for i in range(len(evt.bundles))
               if evt.group_of(evt.bundles[i]["flash_gid"]) == g]
    order = sorted(visible,
                   key=lambda i: (evt.bundles[i]["apa"],
                                  evt.bundles[i]["flash_gid"],
                                  evt.bundles[i]["main_cluster"]))
    state["order"] = order
    cols = defaultdict(list)
    labels = []
    for r, i in enumerate(order):
        b = evt.bundles[i]
        ndf = b["ndf"] or 1
        cu_id = evt.cluster_by_uid[b["main_cluster"]]["ident"]
        cols["row"].append(r)
        cols["auto"].append("Y" if b["auto_selected"] else "")
        cols["apa"].append(b["apa"])
        cols["flash_gid"].append(b["flash_gid"])
        cols["t_us"].append(evt.flash_by_gid[b["flash_gid"]]["time"])
        cols["grp"].append(evt.group_of(b["flash_gid"]))
        cols["cluster"].append(cu_id)
        cols["noth"].append(len(b["other_clusters"]))
        cols["ks"].append(b["ks_dis"])
        cols["chi2ndf"].append(b["chi2"] / ndf)
        cols["strength"].append(b["strength"])
        cols["meas"].append(b["total_PE"])
        cols["pred"].append(b["total_pred_light"])
        cols["flags"].append(fmt_flags(b))
        locked = (state["filter_on"] and i not in state["selected"]
                  and cluster_eliminated(i))
        labels.append("%d: T%d fl%d c%d  ks%.2f pr%.0f%s"
                      % (r, b["apa"], b["flash_gid"], cu_id, b["ks_dis"],
                         b["total_pred_light"], "  🔒" if locked else ""))
    table_src.data = dict(cols)
    # the CheckboxGroup is the clickable selector: labels in table-row order, the
    # checked boxes = the selected bundles of this group. Guard the programmatic
    # `active` write so it does not re-fire on_sel_group.
    active = [r for r, i in enumerate(order) if i in state["selected"]]
    state["suppress_edit"] = True
    sel_group.labels = labels
    sel_group.active = active
    state["suppress_edit"] = False
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
        ho, hr = HIST[apa]["overlay"], HIST[apa]["ratio"]
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
            ho["meas"].data = dict(x=[], pe=[])
            ho["pred"].data = dict(x=[], pe=[])
            hr["src"].data = dict(x=[], ratio=[])
            ho["fig"].title.text = "TPC%d meas vs pred  (no flash in group)" % apa
            hr["fig"].title.text = "TPC%d pred/meas" % apa
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

        # 1-D per-channel comparison (same meas/pred over the active PMTs).
        xidx = np.arange(meas.size)
        ho["meas"].data = dict(x=xidx.tolist(), pe=meas.tolist())
        ho["pred"].data = dict(x=xidx.tolist(), pe=pred.tolist())
        mask = meas > 0     # ratio undefined where measured PE is 0 -> drop
        hr["src"].data = dict(x=xidx[mask].tolist(),
                              ratio=(pred[mask] / meas[mask]).tolist())
        ho["fig"].title.text = ("TPC%d meas vs pred  (gid %d, t=%.1f us)"
                                % (apa, gid, flash["time"]))
        hr["fig"].title.text = ("TPC%d pred/meas  (%d/%d chans meas>0)"
                                % (apa, int(mask.sum()), meas.size))


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


def rebuild_compare():
    """Fill the compare table with every bundle sharing the focused cluster's uid
    (one cluster -> naturally one TPC), across all flashes/groups. Empty when no
    cluster is being compared."""
    evt = state["evt"]
    cu = state["compare_cluster"]
    empty = dict(sel=[], auto=[], flash_gid=[], t_us=[], grp=[], ks=[],
                 chi2ndf=[], strength=[], meas=[], pred=[], flags=[])
    if cu is None or cu not in evt.cluster_by_uid:
        compare_src.data = empty
        state["compare_order"] = []
        compare_div.text = ("<i>focus a bundle, then 'Compare cluster's flashes' to "
                            "list every flash this cluster could match.</i>")
        compare_src.selected.indices = []
        return
    rows = [i for i in range(len(evt.bundles)) if evt.bundles[i]["main_cluster"] == cu]
    rows.sort(key=lambda i: evt.flash_by_gid[evt.bundles[i]["flash_gid"]]["time"])
    state["compare_order"] = rows
    cols = defaultdict(list)
    for i in rows:
        b = evt.bundles[i]
        ndf = b["ndf"] or 1
        cols["sel"].append("✔" if i in state["selected"] else "")
        cols["auto"].append("Y" if b["auto_selected"] else "")
        cols["flash_gid"].append(b["flash_gid"])
        cols["t_us"].append(evt.flash_by_gid[b["flash_gid"]]["time"])
        cols["grp"].append(evt.group_of(b["flash_gid"]))
        cols["ks"].append(b["ks_dis"])
        cols["chi2ndf"].append(b["chi2"] / ndf)
        cols["strength"].append(b["strength"])
        cols["meas"].append(b["total_PE"])
        cols["pred"].append(b["total_pred_light"])
        cols["flags"].append(fmt_flags(b))
    # Clear any stale selection *before* swapping the data: two different clusters can
    # share the same candidate flashes, so the visible left columns (flash/t/grp/meas)
    # are byte-identical and the grid won't repaint a still-selected row otherwise.
    compare_src.selected.indices = []
    compare_src.data = dict(cols)
    c = evt.cluster_by_uid[cu]
    compare_div.text = ("<b>cluster %d (TPC%d)</b> &mdash; %d candidate flash(es); "
                        "click a row to jump to that flash."
                        % (c["ident"], c["apa"], len(rows)))
    # highlight the focused row if it is in the list (programmatic; see on_compare_row)
    idx = state["focus"]
    if idx in rows:
        compare_src.selected.indices = [rows.index(idx)]


def sync_groups():
    """Update the group dropdown to the non-empty groups (plus the current one, kept
    valid even if it just emptied). prev/next cycle the non-empty set, so navigation
    skips groups with no remaining bundle. Never sets .value (would re-fire)."""
    evt = state["evt"]
    nonempty = [g for g in state["groups"] if visible_count(g) > 0]
    state["nav_groups"] = nonempty
    keep = set(nonempty)
    if state["group"] is not None:
        keep.add(state["group"])
    new_options = [(str(g), group_label(evt, g)) for g in sorted(keep)]
    if group_select.options != new_options:
        group_select.options = new_options


def refresh():
    rebuild_table()
    render_light()
    render_projections()
    render_metrics()
    render_summary()
    rebuild_compare()
    sync_groups()


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
    state["compare_cluster"] = None
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
                   "%d coincidence groups; %d auto-selected. Pick a group; tick the ✓ box "
                   "to select bundles (several per TPC add up in the predicted pattern); "
                   "'Filter selected bundles' locks reused clusters."
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
    g = int(new)
    state["group"] = g
    # keep focus only when the focused bundle belongs to the new group (so a compare
    # row-click can set focus then switch group without it being cleared here); a
    # plain group navigation drops the now-foreign focus.
    idx = state["focus"]
    if idx is not None and state["evt"].group_of(state["evt"].bundles[idx]["flash_gid"]) != g:
        state["focus"] = None
    refresh()


def step_group(d):
    nav = state["nav_groups"] or state["groups"]
    if not nav:
        return
    cur = state["group"]
    if cur in nav:
        group_select.value = str(nav[(nav.index(cur) + d) % len(nav)])
    else:
        group_select.value = str(nav[0])


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
        # once the compare table is open, follow the focused cluster so re-focusing a
        # different bundle refreshes it (no need to re-click Compare).
        if state["compare_cluster"] is not None:
            state["compare_cluster"] = state["evt"].bundles[state["focus"]]["main_cluster"]
            rebuild_compare()


def on_toggle():
    if state["focus"] is None:
        status.text = "Select a bundle row first."
        return
    ok, msg = toggle_select(state["focus"])
    b = state["evt"].bundles[state["focus"]]
    status.text = ("bundle [flash %d, cluster %d]: %s"
                   % (b["flash_gid"], cluster_ident(state["focus"]), msg or "no change"))
    refresh()


def on_sel_group(attr, old, new):
    """The user ticked/unticked a box in the selection CheckboxGroup. `new` is the
    list of checked positions (into state['order']). Diff against the current
    selection and apply each change through set_selected (which honours the filter
    lock); refresh re-renders the predicted sum and reverts any rejected tick."""
    if state["suppress_edit"] or state["evt"] is None:
        return
    order = state["order"]
    expected = {r for r, i in enumerate(order) if i in state["selected"]}
    nowset = set(new)
    msgs = []
    last = None
    for r in sorted(nowset ^ expected):
        if r >= len(order):
            continue
        want = r in nowset
        ok, msg = set_selected(order[r], want)
        if ok:
            last = order[r]
        elif msg:
            msgs.append(msg)
    if last is not None:
        state["focus"] = last           # show the just-toggled bundle in the panels
    if msgs:
        status.text = " / ".join(msgs)
    refresh()


def on_filter(attr, old, new):
    state["filter_on"] = bool(new)
    filter_btn.label = "Filter selected bundles: %s" % ("ON" if new else "OFF")
    filter_btn.button_type = "warning" if new else "default"
    status.text = ("Filter ON — bundles reusing an already-matched cluster are locked "
                   "(🔒) and cannot be selected." if new else
                   "Filter OFF — any bundle can be selected.")
    refresh()


def on_clear():
    state["selected"] = set()
    status.text = "Cleared all selections."
    refresh()


def on_compare():
    if state["focus"] is None:
        status.text = "Focus a bundle row first, then Compare."
        return
    cu = state["evt"].bundles[state["focus"]]["main_cluster"]
    state["compare_cluster"] = cu
    rebuild_compare()
    status.text = ("Comparing cluster %d's candidate flashes (table below)."
                   % state["evt"].cluster_by_uid[cu]["ident"])


def on_compare_row(attr, old, new):
    if not new:
        return
    order = state["compare_order"]
    pos = new[0]
    if not (0 <= pos < len(order)):
        return
    idx = order[pos]
    if idx == state["focus"]:
        return                      # programmatic re-highlight of the focused row
    state["focus"] = idx            # set before the group switch so it survives it
    g = state["evt"].group_of(state["evt"].bundles[idx]["flash_gid"])
    if group_select.value == str(g):
        refresh()
    else:
        group_select.value = str(g)  # fires on_group_change (keeps focus, then refresh)


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
sel_group.on_change("active", on_sel_group)
compare_src.selected.on_change("indices", on_compare_row)
select_btn.on_click(on_toggle)
clear_btn.on_click(on_clear)
save_btn.on_click(on_save)
compare_btn.on_click(on_compare)
filter_btn.on_change("active", on_filter)


# ---------------------------------------------------------------------------
# Layout.
# ---------------------------------------------------------------------------
controls = row(event_select, prev_evt_btn, next_evt_btn,
               group_select, prev_grp_btn, next_grp_btn,
               select_btn, clear_btn, save_btn, compare_btn, filter_btn)
header = Div(text="<h2>SBND Q/L matching hand-scan</h2>", width=1100)
layout = column(
    header,
    controls,
    status,
    row(table, column(sel_group_title, sel_group, selsummary)),
    metrics,
    compare_div,
    compare_table,
    row(LIGHT[0]["meas"]["fig"], LIGHT[0]["pred"]["fig"],
        LIGHT[1]["meas"]["fig"], LIGHT[1]["pred"]["fig"]),
    row(HIST[0]["overlay"]["fig"], HIST[0]["ratio"]["fig"],
        HIST[1]["overlay"]["fig"], HIST[1]["ratio"]["fig"]),
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

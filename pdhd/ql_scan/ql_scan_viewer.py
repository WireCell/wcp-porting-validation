#!/usr/bin/env python3
"""PDHD Q/L matching hand-scan event display (Bokeh server).

PDHD counterpart of sbnd_xin/ql_scan/ql_scan_viewer.py (faithful port). Reads the
per-event calibration dumps written by QLMatching (run_clus_evt.sh -calib). Unlike
SBND's single joint dump, PDHD wires one QLMatching node per drift side, so each
event has up to TWO per-group files:
    work/<run6>_<evt>/calib-evt<ID>-group02.json   (APAs 0+2, drift -x  -> side 0)
    work/<run6>_<evt>/calib-evt<ID>-group13.json   (APAs 1+3, drift +x  -> side 1)
The central cathode is opaque to VUV, so the two drift sides are physically
independent (no cross-side light coincidence). The viewer MERGES the 1-2 group
files of an event into one two-panel (side 0 / side 1) view; today usually only one
side carries light, and the unlit side's panels simply stay blank. Each file holds
every candidate (flash, cluster) bundle with its predicted vs measured light, the
matching metrics (ks/chi2/ndf/strength) and flags, the cluster geometry and the
detector box.

The tool lets a human pick the correct flash<->cluster match per cluster and saves
those labels (Save -> work/ql_labels/<tag>/labels-evt<ID>.json) for later tuning of
the QLMatching chi2 / metric parameters and the PE_err model. Labels and the
autosaved selection live in work/ql_labels/ (a sibling of the per-event
work/<run6>_<evt>/ workspace) so they are not lost when an event is reprocessed.
Pass `--tag mc` / `--tag data` to subdir them (work/ql_labels/mc/, .../data/) so the
two displays don't intermix.

Review mode: rather than scanning from scratch, click "Load auto-match" to seed the
selection from QLMatching's own result (each bundle's `auto_selected` flag, dumped
after matching finishes), then examine and correct it. The selection summary shows the
running diff vs the matcher (+added / -removed). Save then records both the corrected
`matches` and a `rejected_auto` list (auto-matches the human removed), so the
hand-vs-matcher correction is fully recoverable for tuning.

Selection rules (enforced live):
  * each cluster matches at most one flash (selecting a bundle drops the cluster's
    other candidate bundles);
  * one flash may match several clusters -> the predicted light shown for that flash
    is the element-wise SUM of the selected bundles' predictions (measured is the
    flash's own, unchanged);
  * both drift sides are shown together; the coincidence-group machinery is inherited
    from SBND but PDHD's opaque cathode means groups are single-side in practice.

Launched by serve_ql_scan.sh; mirrors sbnd_xin/ql_scan/ql_scan_viewer.py
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
    # (so evt2 < evt686 < evt1258, not lexicographic), then by side so group02
    # files sort before group13 of the same event.
    def keyf(f):
        lab = event_label(f)
        digits = "".join(ch for ch in lab if ch.isdigit())
        return (int(digits) if digits else 0, side_of(f), f)
    return sorted(set(f for f in files if os.path.isfile(f)), key=keyf)


def event_label(path):
    """calib-evt<ID>-group02.json / calib-evt<ID>-group13.json -> evt<ID> (the
    per-group suffix is stripped so the two drift-side files share one event)."""
    base = os.path.basename(path)
    if base.startswith("calib-") and base.endswith(".json"):
        stem = base[len("calib-"):-len(".json")]      # evt<ID>[-group02|-group13]
        for suf in ("-group02", "-group13"):
            if stem.endswith(suf):
                return stem[:-len(suf)]
        return stem
    return base


def side_of(path):
    """Drift side of a per-group calib file: group02 (APAs 0+2, drift -x) -> 0,
    group13 (APAs 1+3, drift +x) -> 1. Defaults to 0 for an un-suffixed file."""
    base = os.path.basename(path)
    if "group13" in base:
        return 1
    return 0


def extract_tag(argv):
    """Pull an optional `--tag NAME` out of argv; the rest are calib globs/paths.
    NAME namespaces the saved scan results into work/ql_labels/<NAME>/ so the MC and
    data displays (separate servers) keep their labels apart. Empty ⇒ top-level."""
    out, tag = [], ""
    it = iter(argv)
    for a in it:
        if a in ("--tag", "-t"):
            tag = next(it, "")
        else:
            out.append(a)
    return tag, out


SCAN_TAG, _ARGS = extract_tag(sys.argv[1:])
FILES = discover_files(_ARGS)
# Group the per-drift-side files by event: FILE_OF[label] = [path, ...] (1 or 2
# files, one per populated side). LABELS keeps first-seen (sorted) event order.
FILE_OF = defaultdict(list)
LABELS = []
for f in FILES:
    lab = event_label(f)
    if lab not in FILE_OF:
        LABELS.append(lab)
    FILE_OF[lab].append(f)


# ---------------------------------------------------------------------------
# Per-event model: load one calib JSON and build lookup indexes.
# ---------------------------------------------------------------------------
# Per-side id offset: gid / group / cluster-uid of side s get s*SIDE_OFF added so
# the two drift-side files never collide when merged. Larger than any apa*1e6+ident.
SIDE_OFF = 1_000_000_000


def disp_id(v):
    """Strip the per-side merge offset for display, so gid/group ids read exactly as
    the per-group dump wrote them (the side is shown separately)."""
    return v % SIDE_OFF


class Event:
    def __init__(self, paths):
        """`paths` is the list of 1-2 per-drift-side calib files for one event
        (group02 -> side 0, group13 -> side 1). They share the global OpDet table
        and detector frame; flashes/clusters/bundles are tagged with their side and
        id-offset so the merge is unambiguous, then concatenated."""
        if isinstance(paths, str):
            paths = [paths]
        self.paths = list(paths)
        self.path = self.paths[0]
        flashes, clusters, bundles = [], [], []
        self.geom = {}
        od0 = None
        active_any = None
        self.nchan = self.drift_speed = None
        self.qp = {}
        for p in self.paths:
            s = side_of(p)
            off = s * SIDE_OFF
            with open(p) as fh:
                d = json.load(fh)
            if self.nchan is None:
                self.nchan = d["nchan"]
                self.drift_speed = d["drift_speed"]       # cm/us
                self.qp = d.get("quality_params", {})
                od0 = d["opdets"]
            # OpDet activity is the OR across sides (each node marks only its own
            # side's OpDets active); positions/apa come from the first file.
            act = np.array([o["active"] for o in d["opdets"]], dtype=bool)
            active_any = act if active_any is None else (active_any | act)
            # union the side's per-APA boxes into one drift-side box
            self.geom[s] = self._merge_geom(d["geometry"])
            for f in d["flashes"]:
                f["gid"] += off; f["group"] += off; f["apa"] = s
                flashes.append(f)
            for c in d["clusters"]:
                c["uid"] += off; c["apa"] = s
                clusters.append(c)
            for b in d["bundles"]:
                b["apa"] = s
                b["flash_gid"] += off
                b["main_cluster"] += off
                b["other_clusters"] = [u + off for u in b["other_clusters"]]
                bundles.append(b)
        self.flash_by_gid = {f["gid"]: f for f in flashes}
        self.cluster_by_uid = {c["uid"]: c for c in clusters}
        self.bundles = bundles
        self._clen = {}                       # cached cluster lengths
        # opdet arrays (numpy) for fast light-pattern drawing. od_apa is the drift
        # SIDE (APA ident parity): APAs 0,2 -> side 0; APAs 1,3 -> side 1.
        self.od_x = np.array([o["x"] for o in od0])
        self.od_y = np.array([o["y"] for o in od0])
        self.od_z = np.array([o["z"] for o in od0])
        self.od_apa = np.array([o["apa"] % 2 for o in od0])
        self.od_active = active_any

    @staticmethod
    def _merge_geom(geom):
        """Union the per-APA boxes of one drift side into a single box. The APAs of
        a side share one drift volume (anode_x/cathode_x/sign), differing only in
        their y/z footprint, so take the shared drift params and the y/z envelope."""
        vals = list(geom.values())
        g = dict(vals[0])
        g["y_lo"] = min(v["y_lo"] for v in vals)
        g["y_hi"] = max(v["y_hi"] for v in vals)
        g["z_lo"] = min(v["z_lo"] for v in vals)
        g["z_hi"] = max(v["z_hi"] for v in vals)
        return g

    def group_of(self, gid):
        return self.flash_by_gid[gid]["group"]

    def cluster_length(self, uid):
        """Spatial extent of a cluster (cm): the diagonal of its 3-D bounding box.
        A cheap, stable size proxy for the roster (raw, un-shifted points)."""
        if uid not in self._clen:
            c = self.cluster_by_uid[uid]
            if c["x"]:
                d = sum((max(c[a]) - min(c[a])) ** 2 for a in ("x", "y", "z"))
                self._clen[uid] = math.sqrt(d)
            else:
                self._clen[uid] = 0.0
        return self._clen[uid]

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
    "filter_on": True,        # lock: forbid selecting a cluster already matched (default ON)
    "clus_pick": None,        # uid of the cluster clicked in the roster (drives Compare)
    "clus_order": [],         # roster row order (row index -> cluster uid)
    "sel_snapshot": [],       # last pushed checkbox column (to detect user edits)
    "suppress_edit": False,   # guard while we (re)write the table data ourselves
    "loaded_summary": "",     # the "Loaded <evt>: ..." sentence (echoed under the projections)
}


# ----- coincidence-group helpers -------------------------------------------
def flash_measured_on_side(evt, f):
    """Total measured PE of flash f over its own drift side's ACTIVE OpDets. The
    light panels mask to `od_active & (od_apa == side)`, so this is what a side
    actually shows. Zero for an uninstrumented-side flash."""
    am = evt.od_active & (evt.od_apa == f["apa"])
    if not am.any():
        return 0.0
    return float(np.asarray(f["pe"])[am].sum())


def event_groups(evt):
    # Only groups whose flash carries measured light on that side's active OpDets.
    # In single-sided optical runs (e.g. 27305) group_by_side is a no-op, so every
    # physical flash is duplicated into BOTH per-side nodes; the uninstrumented
    # side's copies have 0 PE on their (dark) active OpDets and would page as empty
    # measured panels (the duplicate "grp N" with the blank side). Skip them so the
    # scan lists only real, light-carrying groups. (Also drops the contained-only
    # dump's bundle-less groups, as before.) Falls back to all groups if the filter
    # would hide everything, so a fully-dark event is never blanked silently.
    lit = {evt.group_of(b["flash_gid"]) for b in evt.bundles
           if flash_measured_on_side(evt, evt.flash_by_gid[b["flash_gid"]]) > 0}
    if lit:
        return sorted(lit)
    return sorted({evt.group_of(b["flash_gid"]) for b in evt.bundles})


def lit_sides(evt):
    """Drift side(s) (0/1) whose flashes carry measured PE on their active OpDets.
    In single-sided optical runs (e.g. 27305) only the instrumented side qualifies;
    clusters on the dark side cannot be Q/L matched, so the roster hides them.
    Empty set ⇒ fully dark event (callers then fall back to showing all sides)."""
    s = getattr(evt, "_lit_sides", None)
    if s is None:
        s = {f["apa"] for f in evt.flash_by_gid.values()
             if flash_measured_on_side(evt, f) > 0}
        evt._lit_sides = s
    return s


def group_label(evt, g):
    """'grp G  (S0:.. S1:.. us)' — the group's per-side flash times, for orienting.
    The group id carries the side offset; show the bare id for readability."""
    t = {0: None, 1: None}
    for f in evt.flash_by_gid.values():
        if f["group"] == g and t[f["apa"]] is None:
            t[f["apa"]] = f["time"]
    s = lambda x: ("%.1f" % x) if x is not None else "-"
    return "grp %d  (S0:%s S1:%s us)" % (g % SIDE_OFF, s(t[0]), s(t[1]))


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
loadauto_btn = Button(label="Load auto-match", width=140)
clear_btn = Button(label="Clear selections", width=140)
save_btn = Button(label="Save labels", button_type="success", width=120)
compare_btn = Button(label="Compare cluster's flashes", width=200)
filter_btn = Toggle(label="Filter selected bundles: ON", width=220,
                    active=True, button_type="warning")
status = Div(text="", width=1100)
metrics = Div(text="", width=560)
selsummary = Div(text="", width=380)
compare_div = Div(text="", width=1000)
# Echo of the event / coincidence group / load-summary, shown beneath the three
# charge-projection plots so the context is visible without scrolling back up.
proj_info = Div(text="", width=1180)

# read-only green check (string field) for the compare table
check_fmt = HTMLTemplateFormatter(
    template='<span style="color:#2ca02c;font-weight:bold;font-size:15px"><%= value %></span>')

# Cell formatter that tints the whole row green when it is a SELECTED bundle, so the
# already-picked rows stand out persistently (distinct from the blue click-focus
# highlight and the default rows) the moment you land on a group. Reads a per-row
# `sel_bg` field (the colour, or "transparent") shared by every column; values are
# pre-formatted to strings in rebuild_table so the same formatter serves all columns.
def cell_fmt(align):
    return HTMLTemplateFormatter(template=(
        '<div style="background-color:<%= sel_bg %>;text-align:' + align
        + ';margin:-2px -5px;padding:2px 5px"><%= value %></div>'))
fmt_l = cell_fmt("left")
fmt_r = cell_fmt("right")
SEL_BG = "#cde6cd"   # light green for selected rows

# Selection is driven by a real CheckboxGroup (reliable clickable boxes) beside the
# table — one box per current-group bundle, in table-row order. Ticking a box adds the
# bundle to the selection (its predicted light joins the per-flash sum); tick several
# to combine clusters. A 🔒 in the label marks a bundle the filter forbids selecting.
sel_group = CheckboxGroup(labels=[], active=[], width=380)
sel_group_title = Div(text="<b>select matches</b> (tick to add to predicted sum)", width=380)

table_src = ColumnDataSource(data=dict())
table_cols = [
    TableColumn(field="row", title="#", width=30, formatter=fmt_l, sortable=False),
    TableColumn(field="auto", title="auto", width=45, formatter=fmt_l, sortable=False),
    TableColumn(field="apa", title="apa", width=35, formatter=fmt_l, sortable=False),
    TableColumn(field="flash_gid", title="flash", width=70, formatter=fmt_l, sortable=False),
    TableColumn(field="cluster", title="clus", width=50, formatter=fmt_l, sortable=False),
    TableColumn(field="noth", title="+oth", width=40, formatter=fmt_l, sortable=False),
    TableColumn(field="t_us", title="t(us)", width=70, formatter=fmt_r, sortable=False),
    TableColumn(field="grp", title="grp", width=40, formatter=fmt_l, sortable=False),
    TableColumn(field="ks", title="ks", width=55, formatter=fmt_r, sortable=False),
    TableColumn(field="chi2ndf", title="chi2/ndf", width=70, formatter=fmt_r, sortable=False),
    TableColumn(field="strength", title="strength", width=70, formatter=fmt_r, sortable=False),
    TableColumn(field="meas", title="measPE", width=70, formatter=fmt_r, sortable=False),
    TableColumn(field="pred", title="predPE", width=70, formatter=fmt_r, sortable=False),
    TableColumn(field="flags", title="flags", width=150, formatter=fmt_l, sortable=False),
]
table = DataTable(source=table_src, columns=table_cols, width=900, height=300,
                  selectable=True, index_position=None)

# Second table (request 4): every bundle whose main cluster matches the focused
# bundle's, across all flashes/groups, so one cluster's candidate flashes can be
# compared side by side. The dump only emits TPC-contained bundles, so the list is
# already the physically-feasible candidate set (cluster inside the box at that
# flash's T0). Populated on demand by the "Compare" button; click a row to jump the
# whole view (focus + group) to that candidate flash.
compare_src = ColumnDataSource(data=dict())
compare_cols = [
    TableColumn(field="sel", title="✓", width=34, formatter=check_fmt),
    TableColumn(field="auto", title="auto", width=45),
    TableColumn(field="apa", title="apa", width=35),
    TableColumn(field="flash_gid", title="flash", width=70),
    TableColumn(field="cluster", title="clus", width=50),
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

# Cluster roster (all clusters in the event, both drift sides): ident, side, #points,
# length (bbox diagonal, cm), and whether it has been matched (✓ + the flash gid) —
# so you can see at a glance which clusters are still unassigned.
clus_title = Div(text="<b>clusters</b> (✓ = matched)", width=330)
clus_src = ColumnDataSource(data=dict())
clus_cols = [
    TableColumn(field="sel", title="✓", width=30, formatter=check_fmt),
    TableColumn(field="cluster", title="clus", width=50),
    TableColumn(field="apa", title="side", width=40),
    TableColumn(field="flash_gid", title="→flash", width=60),
    TableColumn(field="npts", title="npts", width=55),
    TableColumn(field="length", title="len(cm)", width=65,
                formatter=NumberFormatter(format="0.0")),
]
clus_table = DataTable(source=clus_src, columns=clus_cols, width=330, height=300,
                       selectable=True, index_position=None)

# Light-pattern figures (Y vertical, Z horizontal): a 2x2 grid of measured vs
# predicted for drift side 0 and side 1. Positions are fixed, so the ranges are
# pinned to the detector box (no zoom). Active OpDets of the panel's side are faint
# outlines; the flash OpDets are sized by sqrt(PE).
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


# LIGHT[apa] = {"meas": panel, "pred": panel}  (apa is the drift side, 0 or 1)
LIGHT = {apa: {"meas": make_light_fig("side %d measured" % apa),
               "pred": make_light_fig("side %d predicted" % apa)}
         for apa in (0, 1)}


# 1-D per-channel comparison below the 2x2 light grid: for each side an overlay of
# measured vs predicted PE over that side's active OpDets, and the pred/meas ratio.
def make_hist_fig(title):
    f = figure(title=title, height=220, width=430,
               tools="pan,box_zoom,wheel_zoom,reset,save")
    f.xaxis.axis_label = "active OpDet (channel index)"
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
    f.xaxis.axis_label = "active OpDet (channel index)"
    f.yaxis.axis_label = "pred / meas"
    src = ColumnDataSource(data=dict(x=[], ratio=[]))
    f.scatter("x", "ratio", source=src, marker="circle", size=5,
              fill_color="#2ca02c", line_color=None, fill_alpha=0.8)
    f.add_layout(Span(location=1.0, dimension="width", line_color="#888888",
                      line_dash="dashed", line_width=1))
    return dict(fig=f, src=src)


# HIST[apa] = {"overlay": panel, "ratio": panel}  (apa is the drift side, 0 or 1)
HIST = {apa: {"overlay": make_hist_fig("side %d meas vs pred" % apa),
              "ratio": make_ratio_fig("side %d pred/meas" % apa)}
        for apa in (0, 1)}

# Charge-projection figures (focused bundle's clusters, T0-shifted, in the fixed
# detector box; both side boxes drawn). XY, YZ (z horiz, y vert), XZ (x horiz, z vert).
proj_kw = dict(height=320, tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom")
f_xy = figure(title="X-Y", width=380, **proj_kw)
f_yz = figure(title="Y-Z", width=380, **proj_kw)
f_xz = figure(title="X-Z", width=380, **proj_kw)
f_xy.xaxis.axis_label, f_xy.yaxis.axis_label = "x (cm)", "y (cm)"
f_yz.xaxis.axis_label, f_yz.yaxis.axis_label = "z (cm)", "y (cm)"
f_xz.xaxis.axis_label, f_xz.yaxis.axis_label = "x (cm)", "z (cm)"

# focus cluster points (blue), all selected clusters as context (gray), the selected
# clusters in the current group (green, on top of gray), plus the box outlines.
foc_src = ColumnDataSource(data=dict(x=[], y=[], z=[]))
ctx_src = ColumnDataSource(data=dict(x=[], y=[], z=[]))
ctxg_src = ColumnDataSource(data=dict(x=[], y=[], z=[]))
box_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                     xs_xz=[], ys_xz=[]))
f_xy.multi_line(xs="xs_xy", ys="ys_xy", source=box_src, line_color="#cc4444", line_width=1)
f_yz.multi_line(xs="xs_yz", ys="ys_yz", source=box_src, line_color="#cc4444", line_width=1)
f_xz.multi_line(xs="xs_xz", ys="ys_xz", source=box_src, line_color="#cc4444", line_width=1)
# gray = all SELECTED tracks (any group) as context; green = selected tracks IN THE
# CURRENT group (overlaid on gray); blue = the focused (clicked) bundle, on top.
for f, hx, hy in ((f_xy, "x", "y"), (f_yz, "z", "y"), (f_xz, "x", "z")):
    f.scatter(hx, hy, source=ctx_src, marker="circle", size=2,
              fill_color="#bbbbbb", line_color=None, fill_alpha=0.4)
    f.scatter(hx, hy, source=ctxg_src, marker="circle", size=3,
              fill_color="#2ca02c", line_color=None, fill_alpha=0.55)
    f.scatter(hx, hy, source=foc_src, marker="circle", size=3,
              fill_color="#1f77b4", line_color=None, fill_alpha=0.8)


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
    if b.get("two_boundary"):    parts.append("2bnd")
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
    # longest tracks first (by the main cluster's extent), then a stable tiebreak
    order = sorted(visible,
                   key=lambda i: (-evt.cluster_length(evt.bundles[i]["main_cluster"]),
                                  evt.bundles[i]["apa"],
                                  evt.bundles[i]["flash_gid"],
                                  evt.bundles[i]["main_cluster"]))
    state["order"] = order
    cols = defaultdict(list)
    labels = []
    for r, i in enumerate(order):
        b = evt.bundles[i]
        ndf = b["ndf"] or 1
        cu_id = evt.cluster_by_uid[b["main_cluster"]]["ident"]
        # values are pre-formatted to strings: the row colouring is done by an
        # HTMLTemplateFormatter (cell_fmt) shared across columns, which precludes the
        # numeric NumberFormatter, so we format here.
        cols["row"].append(str(r))
        cols["auto"].append("Y" if b["auto_selected"] else "")
        cols["apa"].append(str(b["apa"]))
        cols["flash_gid"].append(str(disp_id(b["flash_gid"])))
        cols["t_us"].append("%.1f" % evt.flash_by_gid[b["flash_gid"]]["time"])
        cols["grp"].append(str(disp_id(evt.group_of(b["flash_gid"]))))
        cols["cluster"].append(str(cu_id))
        cols["noth"].append(str(len(b["other_clusters"])))
        cols["ks"].append("%.3f" % b["ks_dis"])
        cols["chi2ndf"].append("%.1f" % (b["chi2"] / ndf))
        cols["strength"].append("%.3f" % b["strength"])
        cols["meas"].append("%.0f" % b["total_PE"])
        cols["pred"].append("%.1f" % b["total_pred_light"])
        cols["flags"].append(fmt_flags(b))
        # green tint persists for selected bundles (distinct from blue click-focus)
        cols["sel_bg"].append(SEL_BG if i in state["selected"] else "transparent")
        locked = (state["filter_on"] and i not in state["selected"]
                  and cluster_eliminated(i))
        labels.append("%d: S%d fl%d c%d  ks%.2f pr%.0f%s"
                      % (r, b["apa"], disp_id(b["flash_gid"]), cu_id, b["ks_dis"],
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
    """Four panels: measured + predicted for drift side 0 and side 1. Measured is
    anchored to the current group's per-side flash (a stable reference as rows are
    clicked); predicted sums the selected bundles on that flash (or previews focus)."""
    evt = state["evt"]
    for apa in (0, 1):
        mp, pp = LIGHT[apa]["meas"], LIGHT[apa]["pred"]
        ho, hr = HIST[apa]["overlay"], HIST[apa]["ratio"]
        am = evt.od_active & (evt.od_apa == apa)
        chans = np.nonzero(am)[0]
        # faint outline of this side's active OpDets
        mp["base"].data = dict(z=evt.od_z[am].tolist(), y=evt.od_y[am].tolist())
        pp["base"].data = dict(z=evt.od_z[am].tolist(), y=evt.od_y[am].tolist())

        gid = group_flash_gid(apa)
        if gid is None:
            mp["src"].data = dict(z=[], y=[], pe=[], r=[])
            pp["src"].data = dict(z=[], y=[], pe=[], r=[])
            mp["fig"].title.text = "side %d measured  (no flash in group)" % apa
            pp["fig"].title.text = "side %d predicted" % apa
            ho["meas"].data = dict(x=[], pe=[])
            ho["pred"].data = dict(x=[], pe=[])
            hr["src"].data = dict(x=[], ratio=[])
            ho["fig"].title.text = "side %d meas vs pred  (no flash in group)" % apa
            hr["fig"].title.text = "side %d pred/meas" % apa
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
        mp["fig"].title.text = ("side %d measured  gid %d (id %d)  t=%.1f us  totPE=%.0f"
                                % (apa, gid % SIDE_OFF, flash["id"], flash["time"], flash["total_PE"]))
        if not share:
            lab = "none selected"
        elif previewed:
            lab = "preview: focused bundle"
        else:
            lab = "sum of %d selected cluster(s)" % len(share)
        pp["fig"].title.text = "side %d predicted  (%s)" % (apa, lab)

        # 1-D per-channel comparison (same meas/pred over the active PMTs).
        xidx = np.arange(meas.size)
        ho["meas"].data = dict(x=xidx.tolist(), pe=meas.tolist())
        ho["pred"].data = dict(x=xidx.tolist(), pe=pred.tolist())
        mask = meas > 0     # ratio undefined where measured PE is 0 -> drop
        hr["src"].data = dict(x=xidx[mask].tolist(),
                              ratio=(pred[mask] / meas[mask]).tolist())
        ho["fig"].title.text = ("side %d meas vs pred  (gid %d, t=%.1f us)"
                                % (apa, gid % SIDE_OFF, flash["time"]))
        hr["fig"].title.text = ("side %d pred/meas  (%d/%d chans meas>0)"
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

    # context: ALL selected matches (any group, both TPCs), T0-shifted, drawn gray;
    # the selected matches IN THE CURRENT group are additionally drawn green on top.
    cx, cy, cz = [], [], []          # gray: all selected
    gx, gy, gz = [], [], []          # green: selected in current group
    for j in state["selected"]:
        b = evt.bundles[j]
        dx = evt.dx_cm(b["apa"], b["flash_gid"])
        in_grp = evt.group_of(b["flash_gid"]) == state["group"]
        for uid in [b["main_cluster"]] + b["other_clusters"]:
            x, y, z = cluster_points(uid, b["apa"], dx)
            x, y, z = downsample(x, y, z)
            cx += x.tolist(); cy += y.tolist(); cz += z.tolist()
            if in_grp:
                gx += x.tolist(); gy += y.tolist(); gz += z.tolist()
    ctx_src.data = dict(x=cx, y=cy, z=cz)
    ctxg_src.data = dict(x=gx, y=gy, z=gz)

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
        ("flash", "gid %d / id %d / side %d / group %d"
         % (disp_id(b["flash_gid"]), b["flash_id"], b["apa"],
            disp_id(evt.group_of(b["flash_gid"])))),
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


def auto_diff_line():
    """One-line diff of the current selection against the matcher's auto_selected
    bundles: how many the human added and how many auto-matches were removed."""
    evt = state["evt"]
    auto = {j for j, b in enumerate(evt.bundles) if b["auto_selected"]}
    sel = state["selected"]
    return ("<i>vs auto-match: +%d added, &minus;%d removed</i>"
            % (len(sel - auto), len(auto - sel)))


def render_summary():
    evt = state["evt"]
    if not state["selected"]:
        selsummary.text = "<b>0 matches selected.</b><br>" + auto_diff_line()
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
        lines.append("flash %d (side%d, t=%.1fus, grp%d) &larr; clusters %s"
                     % (disp_id(gid), f["apa"], f["time"], disp_id(f["group"]), cls))
    selsummary.text = ("<b>%d selected match(es):</b> %s<br>"
                       % (len(state["selected"]), auto_diff_line())
                       + "<br>".join(lines))


def rebuild_compare():
    """Fill the compare table with every bundle sharing the focused cluster's uid
    (one cluster -> naturally one TPC), across all flashes/groups. The dump only
    emits TPC-contained bundles, so these are exactly the cluster's physically
    feasible candidate flashes. Empty when no cluster is being compared."""
    evt = state["evt"]
    cu = state["compare_cluster"]
    empty = dict(sel=[], auto=[], apa=[], flash_gid=[], cluster=[], t_us=[], grp=[],
                 ks=[], chi2ndf=[], strength=[], meas=[], pred=[], flags=[])
    if cu is None or cu not in evt.cluster_by_uid:
        compare_src.data = empty
        state["compare_order"] = []
        compare_div.text = ("<i>focus a bundle, then 'Compare cluster's flashes' to "
                            "list every flash this cluster could match.</i>")
        compare_src.selected.indices = []
        return
    rows = [i for i in range(len(evt.bundles))
            if evt.bundles[i]["main_cluster"] == cu]
    rows.sort(key=lambda i: evt.flash_by_gid[evt.bundles[i]["flash_gid"]]["time"])
    state["compare_order"] = rows
    cols = defaultdict(list)
    for i in rows:
        b = evt.bundles[i]
        ndf = b["ndf"] or 1
        cols["sel"].append("✔" if i in state["selected"] else "")
        cols["auto"].append("Y" if b["auto_selected"] else "")
        cols["apa"].append(b["apa"])
        cols["flash_gid"].append(disp_id(b["flash_gid"]))
        cols["cluster"].append(evt.cluster_by_uid[b["main_cluster"]]["ident"])
        cols["t_us"].append(evt.flash_by_gid[b["flash_gid"]]["time"])
        cols["grp"].append(disp_id(evt.group_of(b["flash_gid"])))
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
    compare_div.text = ("<b>cluster %d (side%d)</b> &mdash; %d candidate flash(es); "
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


def rebuild_clusters():
    """Roster of every cluster in the event (both TPCs): ident, TPC, #points,
    length, and the flash it is matched to (✓) if any — so the unassigned clusters
    are obvious."""
    evt = state["evt"]
    matched = {}                            # cluster uid -> flash gid (via selection)
    for j in state["selected"]:
        b = evt.bundles[j]
        for u in [b["main_cluster"]] + b["other_clusters"]:
            matched[u] = b["flash_gid"]
    # Only clusters on a lit drift side: the other side has no flash to match
    # against (single-sided optical readout), so its clusters just clutter the
    # roster. Fall back to all sides if the event is fully dark.
    sides = lit_sides(evt)
    uids = [u for u in evt.cluster_by_uid
            if not sides or evt.cluster_by_uid[u]["apa"] in sides]
    # longest clusters first, then a stable tiebreak (TPC, ident)
    uids = sorted(uids, key=lambda u: (-evt.cluster_length(u),
                                       evt.cluster_by_uid[u]["apa"],
                                       evt.cluster_by_uid[u]["ident"]))
    state["clus_order"] = uids              # roster row index -> cluster uid (for Compare)
    clus_title.text = (
        "<b>clusters</b> (side %s only — lit side; ✓ = matched)"
        % "/".join(str(s) for s in sorted(sides)) if sides
        else "<b>clusters</b> (✓ = matched)")
    cols = defaultdict(list)
    for u in uids:
        c = evt.cluster_by_uid[u]
        cols["sel"].append("✔" if u in matched else "")
        cols["cluster"].append(c["ident"])
        cols["apa"].append(c["apa"])
        cols["npts"].append(c["npoints"])
        cols["length"].append(evt.cluster_length(u))
        cols["flash_gid"].append(matched.get(u, ""))
    clus_src.selected.indices = []          # force repaint on same-row-set change
    clus_src.data = dict(cols)


def render_proj_info():
    """Echo, beneath the projection plots, the same Event / coincidence group /
    load-summary shown by the controls + status at the top of the page."""
    evt = state["evt"]
    if evt is None:
        proj_info.text = ""
        return
    g = state["group"]
    grp = group_label(evt, g) if g is not None else "-"
    proj_info.text = (
        "<b>Event</b> %s &nbsp;&nbsp; <b>coincidence group</b> %s<br>%s"
        % (event_select.value, grp, state["loaded_summary"]))


def refresh():
    rebuild_table()
    render_light()
    render_projections()
    render_metrics()
    render_summary()
    rebuild_compare()
    rebuild_clusters()
    sync_groups()
    render_proj_info()


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


def labels_dir(evt):
    """Persistent scan-output dir: a sibling of the per-event work/<run6>_<evt>/
    workspace (work/ql_labels/, or work/ql_labels/<tag>/ when --tag is given). Kept
    OUT of the per-event workspace so reprocessing an event cannot delete saved scan
    labels / selections. The --tag subdir keeps separate displays' results apart
    (they share one work/ but run as separate servers)."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(evt.path))),
                     "ql_labels", SCAN_TAG)
    os.makedirs(d, exist_ok=True)
    return d


def state_file(evt):
    """Autosave path for the current selection (in the persistent ql_labels dir)."""
    return os.path.join(labels_dir(evt), ".scan_state-%s.json" % event_label(evt.path))


def save_state():
    """Persist the current selection to disk so it survives a page reload / restart /
    event switch. Keyed by (flash_gid, main_cluster) — stable across re-dumps."""
    evt = state["evt"]
    if evt is None:
        return
    keys = [[evt.bundles[j]["flash_gid"], evt.bundles[j]["main_cluster"]]
            for j in sorted(state["selected"])]
    try:
        with open(state_file(evt), "w") as fh:
            json.dump({"selected": keys}, fh)
    except OSError:
        pass


def load_state(evt):
    """Restore a previously-saved selection (set of bundle indices) for this event."""
    p = state_file(evt)
    sel = set()
    if not os.path.isfile(p):
        return sel
    try:
        with open(p) as fh:
            want = {tuple(k) for k in json.load(fh).get("selected", [])}
    except (OSError, ValueError):
        return sel
    for j, b in enumerate(evt.bundles):
        if (b["flash_gid"], b["main_cluster"]) in want:
            sel.add(j)
    return sel


def load_event(label):
    evt = Event(FILE_OF[label])
    state["evt"] = evt
    state["focus"] = None
    state["selected"] = load_state(evt)      # restore any saved picks for this event
    state["compare_cluster"] = None
    state["clus_pick"] = None
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
    summary = ("Loaded <b>%s</b>: %d contained bundles, %d flashes, %d clusters, "
               "%d coincidence groups; %d auto-selected. Pick a group; tick the ✓ box "
               "to select bundles (several per side add up in the predicted pattern); "
               "'Filter selected bundles' locks reused clusters."
               % (label, n, len(evt.flash_by_gid), len(evt.cluster_by_uid),
                  len(groups), nsel))
    status.text = summary
    state["loaded_summary"] = summary
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
        state["clus_pick"] = None       # focusing a bundle takes over from a roster pick
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
                   % (disp_id(b["flash_gid"]), cluster_ident(state["focus"]), msg or "no change"))
    save_state()
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
    save_state()
    refresh()


def on_filter(attr, old, new):
    state["filter_on"] = bool(new)
    filter_btn.label = "Filter selected bundles: %s" % ("ON" if new else "OFF")
    filter_btn.button_type = "warning" if new else "default"
    status.text = ("Filter ON — bundles reusing an already-matched cluster are locked "
                   "(🔒) and cannot be selected." if new else
                   "Filter OFF — any bundle can be selected.")
    refresh()


def on_load_auto():
    """Seed the selection from the matcher's result (auto_selected bundles), so the
    scan starts as a review of QLMatching rather than from scratch. Replaces the
    current selection."""
    evt = state["evt"]
    if evt is None:
        return
    state["selected"] = {j for j, b in enumerate(evt.bundles) if b["auto_selected"]}
    status.text = ("Loaded %d auto-matched bundle(s) from QLMatching. Examine and "
                   "correct, then Save." % len(state["selected"]))
    save_state()
    refresh()


def on_clear():
    state["selected"] = set()
    status.text = "Cleared all selections."
    save_state()
    refresh()


def on_clus_select(attr, old, new):
    """Remember the cluster clicked in the roster so the Compare button can list its
    candidate flashes (an alternative to focusing one of its bundles in the table)."""
    if not new:
        return
    order = state["clus_order"]
    pos = new[0]
    if 0 <= pos < len(order):
        state["clus_pick"] = order[pos]
        c = state["evt"].cluster_by_uid[order[pos]]
        status.text = ("cluster %d (side%d) picked — click 'Compare cluster's flashes'."
                       % (c["ident"], c["apa"]))


def on_compare():
    # Prefer a cluster clicked in the roster; otherwise fall back to the focused bundle.
    cu = state["clus_pick"]
    if cu is None:
        if state["focus"] is None:
            status.text = "Pick a cluster in the roster (or focus a bundle row), then Compare."
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


def _match_entry(evt, j):
    """The saved record for bundle j (shared by `matches` and `rejected_auto`)."""
    b = evt.bundles[j]
    f = evt.flash_by_gid[b["flash_gid"]]
    return {
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
    }


def on_save():
    evt = state["evt"]
    # auto-matches the human removed: a meaningful verdict, so an empty selection is
    # still worth saving as long as there were auto-matches to reject.
    rejected = sorted(j for j, b in enumerate(evt.bundles)
                      if b["auto_selected"] and j not in state["selected"])
    if not state["selected"] and not rejected:
        status.text = "Nothing selected (and no auto-match to reject) to save."
        return
    out = {"event": event_select.value, "source": os.path.basename(evt.path),
           "matches": [_match_entry(evt, j) for j in sorted(state["selected"])],
           "rejected_auto": [_match_entry(evt, j) for j in rejected]}
    dest = os.path.join(labels_dir(evt),
                        "labels-%s.json" % event_select.value)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    status.text = ("Saved %d match(es), %d rejected auto-match(es) -> %s"
                   % (len(out["matches"]), len(rejected), dest))


event_select.on_change("value", on_event_change)
prev_evt_btn.on_click(lambda: step_event(-1))
next_evt_btn.on_click(lambda: step_event(+1))
group_select.on_change("value", on_group_change)
prev_grp_btn.on_click(lambda: step_group(-1))
next_grp_btn.on_click(lambda: step_group(+1))
table_src.selected.on_change("indices", on_row_select)
sel_group.on_change("active", on_sel_group)
compare_src.selected.on_change("indices", on_compare_row)
clus_src.selected.on_change("indices", on_clus_select)
select_btn.on_click(on_toggle)
loadauto_btn.on_click(on_load_auto)
clear_btn.on_click(on_clear)
save_btn.on_click(on_save)
compare_btn.on_click(on_compare)
filter_btn.on_change("active", on_filter)


# ---------------------------------------------------------------------------
# Layout.
# ---------------------------------------------------------------------------
# Two layers: navigation on top, actions below (keeps the button strip from
# overflowing a narrow window).
controls_nav = row(event_select, prev_evt_btn, next_evt_btn,
                   group_select, prev_grp_btn, next_grp_btn)
controls_act = row(select_btn, loadauto_btn, clear_btn, save_btn, compare_btn, filter_btn)
controls = column(controls_nav, controls_act)
header = Div(text="<h2>PDHD Q/L matching hand-scan</h2>", width=1100)
layout = column(
    # Charge-projection views first so a small monitor shows the plots up top; all the
    # controls / tables / light panels follow below (top half = plots, bottom = operate).
    row(f_xy, f_yz, f_xz),
    proj_info,
    header,
    controls,
    status,
    row(table, column(sel_group_title, sel_group, selsummary), metrics,
        column(clus_title, clus_table)),
    compare_div,
    compare_table,
    row(LIGHT[0]["meas"]["fig"], LIGHT[0]["pred"]["fig"],
        LIGHT[1]["meas"]["fig"], LIGHT[1]["pred"]["fig"]),
    row(HIST[0]["overlay"]["fig"], HIST[0]["ratio"]["fig"],
        HIST[1]["overlay"]["fig"], HIST[1]["ratio"]["fig"]),
)

curdoc().add_root(layout)
curdoc().title = "Q/L hand-scan"

if LABELS:
    load_event(LABELS[0])
else:
    status.text = ("<b>No calib JSONs found.</b> Produce them with "
                   "run_clus_evt.sh -calib &lt;run&gt; &lt;evt&gt;, then pass the "
                   "work/&lt;run6&gt;_&lt;evt&gt;/calib-evt*-*.json glob to "
                   "serve_ql_scan.sh.")

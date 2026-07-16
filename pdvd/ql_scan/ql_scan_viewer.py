#!/usr/bin/env python3
"""PDVD Q/L matching hand-scan event display (Bokeh server).

PDVD counterpart of pdhd/ql_scan/ql_scan_viewer.py (full duplication, adapted).
Reads the per-event calibration dump written by QLMatching (run_clus_evt.sh
-calib). PDVD wires ONE JOINT shared-flash QLMatching node taking both drift
volumes, so each event has a single combined file:
    work/<run6>_<evtidx>/calib-evt<ID>.json   (apa=0 = bottom drift volume,
                                               anodes 0-3; apa=4 = top volume,
                                               anodes 4-7; cathode at x~0)
Unlike PDHD, PDVD has ONE all-PD flash list shared by both drift volumes: the
cathode X-ARAPUCAs (ch 4-11, ~89% of the light) are double-sided at x~0, so a
single physical flash is explained jointly by clusters from BOTH volumes (the
shared-flash LASSO). The viewer therefore has NO cross-side coincidence
machinery: every flash is one scan group, shown on BOTH panels at once (bottom
volume = cathode XAs + bottom membrane XAs + z-wall PMTs + bottom PMTs; top
volume = cathode XAs + top membrane XAs). The file holds every candidate
(flash, cluster) bundle with its predicted vs measured light, the matching
metrics (ks/chi2/ndf/strength) and flags (incl. the PDVD-only at_cathode), the
cluster geometry and the two drift boxes.

The tool lets a human pick the correct flash<->cluster match per cluster and
saves those labels (Save -> work/ql_labels/<tag>/labels-evt<ID>.json) for later
tuning of the QLMatching chi2 / metric parameters and the PE_err model. Labels
and the autosaved selection live in work/ql_labels/ (a sibling of the per-event
work/<run6>_<evtidx>/ workspace) so they are not lost when an event is
reprocessed. Pass `--tag mc` / `--tag data` to subdir them.

Review mode: rather than scanning from scratch, click "Load auto-match" to seed
the selection from QLMatching's own result (each bundle's `auto_selected` flag,
dumped after matching finishes), then examine and correct it. The selection
summary shows the running diff vs the matcher (+added / -removed). Save then
records both the corrected `matches` and a `rejected_auto` list (auto-matches
the human removed), so the hand-vs-matcher correction is fully recoverable.

Selection rules (enforced live):
  * each cluster matches at most one flash (selecting a bundle drops the
    cluster's other candidate bundles);
  * one flash may match several clusters -> the predicted light shown for that
    flash is the element-wise SUM of ALL selected bundles' predictions across
    BOTH volumes (measured is the flash's own, unchanged) -- the light panels
    are grouped by PD type (X-ARAPUCA / PMT), not drift volume, since the
    flash is one all-PD set;
  * both drift volumes are shown together in the charge-projection panels; a
    cathode-crossing cosmic is two bundles (one per volume) on the SAME flash
    -- tick both.

Launched by serve_ql_scan.sh (default port 5016); mirrors
pdhd/ql_scan/ql_scan_viewer.py (server-side callbacks, ColumnDataSource per
figure).
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
                          NumeralTickFormatter, Span, Label, HTMLTemplateFormatter,
                          CheckboxGroup, Toggle, LinearColorMapper)
from bokeh.plotting import figure

# Max charge points drawn per projection panel (stride-downsample above this).
MAX_DRAW_PTS = 5000

# Cluster length cutoff (cm, bbox-diagonal proxy): with the length filter on (default)
# clusters at or below this are hidden from the roster + bundle table, and a flash
# group whose every cluster is this short is dropped from the scan navigation. An event
# has many tiny fragments that otherwise swamp the display.
MIN_CLUS_LEN = 5.0

# The two drift volumes, keyed by the representative anode ident used as `apa` in
# the dump: 0 = bottom volume (anodes 0-3, anode at x=-335.8), 4 = top volume
# (anodes 4-7, anode at +335.8). The cathode sits at x~0 between them. Used for
# the cluster/bundle roster (a cluster belongs to one drift volume).
SIDES = (0, 4)
SIDE_NAME = {0: "bottom", 4: "top"}

# The light panels (2-D maps + 1-D histograms below) are grouped by PD TYPE, not
# drift volume: the flash is a single all-PD set (cathode X-ARAPUCAs are double-
# sided and belong to both volumes at once), so a bottom/top split either dupes
# or under-predicts them. Each type further splits into two spatially-disjoint
# sub-blocks (`type` = 0/1 in the dump's `opdets`):
#   XA  : wall/membrane XAs (|x| > 10)  |  cathode XAs (|x| <= 10)
#   PMT : z-wall PMTs (x > -320)        |  bottom/floor PMTs (x <= -320, all
#         PMTs live in the bottom volume; bottom-anode plane is x~-335.8)
GROUPS = ("XA", "PMT")
GROUP_NAME = {"XA": "X-ARAPUCA", "PMT": "PMT"}
SUBNAME = {"XA": ("wall", "cathode"), "PMT": ("wall", "bottom")}

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
    """calib-evt<ID>.json -> evt<ID>."""
    base = os.path.basename(path)
    if base.startswith("calib-") and base.endswith(".json"):
        return base[len("calib-"):-len(".json")]
    return base


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
# One combined file per event (the joint shared-flash node writes a single dump).
FILE_OF = {}
LABELS = []
for f in FILES:
    lab = event_label(f)
    if lab not in FILE_OF:
        LABELS.append(lab)
        FILE_OF[lab] = f


# ---------------------------------------------------------------------------
# Per-event model: load one calib JSON and build lookup indexes.
# ---------------------------------------------------------------------------
def od_jitter(x, y, z, scale=14.0):
    """Several PDs project onto the SAME (y,z) point in the z-y light maps --
    membrane XAs come in front/back pairs per drift volume (4-way stack at one
    (y,z)), z-wall PMTs come in front/back pairs (2-way) -- differing only in x
    (depth), which the 2-D scatter doesn't show. Without this they'd draw one
    marker on top of another and only the last-drawn channel would be visible.
    Spread same-(y,z) channels (ordered by x, so front/back stays a consistent
    left/right or ring layout run to run) onto a small ring around the true
    point; channels with a unique (y,z) get zero offset."""
    groups = defaultdict(list)
    for i in range(len(y)):
        groups[(round(y[i], 1), round(z[i], 1))].append(i)
    dz = np.zeros(len(z))
    dy = np.zeros(len(y))
    for idxs in groups.values():
        n = len(idxs)
        if n <= 1:
            continue
        idxs = sorted(idxs, key=lambda i: x[i])
        for k, i in enumerate(idxs):
            ang = 2 * math.pi * k / n
            dz[i] = scale * math.cos(ang)
            dy[i] = scale * math.sin(ang)
    return dz, dy


class Event:
    def __init__(self, path):
        """One combined calib file carries BOTH drift volumes (apa=0 and apa=4
        bundles/clusters) against ONE shared flash list (each physical flash is
        emitted once, gid = its row index). Every flash is its own scan group,
        numbered by ascending time."""
        self.path = path
        with open(path) as fh:
            d = json.load(fh)
        self.nchan = d["nchan"]
        self.drift_speed = d["drift_speed"]           # cm/us
        self.trigger_offset = d.get("trigger_offset", 0.0)  # us, matcher-applied
        self.qp = d.get("quality_params", {})
        od0 = d["opdets"]
        # geometry keyed by drift volume ("0" bottom / "4" top).
        self.geom = {int(k): dict(g) for k, g in d["geometry"].items()}
        flashes = list(d["flashes"])
        self.cluster_by_uid = {c["uid"]: c for c in d["clusters"]}
        self.bundles = d["bundles"]
        self._clen = {}                       # cached cluster lengths
        # One scan group per (shared) flash, ranked by ascending flash time so the
        # scan order matches Bee's time-ordered flash list.
        for g, f in enumerate(sorted(flashes, key=lambda f: f["time"]), start=1):
            f["group"] = g
        self.flash_by_gid = {f["gid"]: f for f in flashes}
        # opdet arrays (numpy) for fast light-pattern drawing. `type` (0 = XA,
        # 1 = PMT) drives the light-panel grouping (see GROUPS above); the
        # double-sided cathode XAs (x ~ 0) belong to the XA group's "cathode"
        # sub-block same as the wall XAs belong to its "wall" sub-block.
        self.od_x = np.array([o["x"] for o in od0])
        self.od_y = np.array([o["y"] for o in od0])
        self.od_z = np.array([o["z"] for o in od0])
        self.od_type = np.array([o["type"] for o in od0])
        self.od_active = np.array([o["active"] for o in od0], dtype=bool)
        self.od_auto_masked = np.array([o.get("auto_masked", False) for o in od0],
                                       dtype=bool)
        # z-y display offsets for PDs that share a projected position (see
        # od_jitter docstring); fixed detector geometry -> computed once.
        self.od_jitter_z, self.od_jitter_y = od_jitter(self.od_x, self.od_y, self.od_z)

    def group_mask(self, group):
        """(wall_mask, inner_mask) over all 40 opdets for a PD-type GROUPS key
        ("XA"/"PMT"), split into its two spatially-disjoint sub-blocks (see
        GROUPS comment for the boundaries)."""
        typ = self.od_type == (0 if group == "XA" else 1)
        if group == "XA":
            return typ & (np.abs(self.od_x) > 10.0), typ & (np.abs(self.od_x) <= 10.0)
        return typ & (self.od_x > -320.0), typ & (self.od_x <= -320.0)

    def group_channels(self, group):
        """Active channel idx for a PD-type group, ordered [wall..., inner...],
        and the boundary index between the two sub-blocks (for the separator)."""
        wall, inner = self.group_mask(group)
        aw = np.nonzero(self.od_active & wall)[0]
        ai = np.nonzero(self.od_active & inner)[0]
        return np.concatenate([aw, ai]), len(aw)

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
        """T0 x-shift (cm) applied to volume `apa`'s charge for the flash at gid.
        Includes the calibrated light<->charge trigger offset the matcher used."""
        g = self.geom[apa]
        t_us = self.flash_by_gid[gid]["time"] + self.trigger_offset
        return g["sign_offset"] * t_us * self.drift_speed


# ---------------------------------------------------------------------------
# Global UI state.
# ---------------------------------------------------------------------------
state = {
    "evt": None,        # Event
    "focus": None,      # focused bundle index (into evt.bundles), or None
    "selected": set(),  # set of selected bundle indices (GLOBAL, all groups)
    "order": [],        # visible bundle indices in table display order
    "group": None,      # current flash group id (the scan unit)
    "groups": [],       # sorted group ids present in the event
    "nav_groups": [],   # groups with >=1 visible bundle (prev/next cycle these)
    "compare_cluster": None,  # uid of the cluster shown in the compare table, or None
    "compare_order": [],      # bundle indices listed in the compare table
    "filter_on": True,        # lock: forbid selecting a cluster already matched (default ON)
    "len_filter_on": True,    # hide clusters <= MIN_CLUS_LEN (and all-short groups); default ON
    "clus_pick": None,        # uid of the cluster clicked in the roster (drives Compare)
    "clus_order": [],         # roster row order (row index -> cluster uid)
    "sel_snapshot": [],       # last pushed checkbox column (to detect user edits)
    "suppress_edit": False,   # guard while we (re)write the table data ourselves
    "loaded_summary": "",     # the "Loaded <evt>: ..." sentence (echoed under the projections)
}


# ----- flash-group helpers ---------------------------------------------------
def group_label(evt, g):
    """'grp G  (t=.. us, PE ..)' — the group's (single, shared) flash."""
    for f in evt.flash_by_gid.values():
        if f["group"] == g:
            return "grp %d  (t=%.1f us, PE %.0f)" % (g, f["time"], f["total_PE"])
    return "grp %d" % g


def group_flash_gid():
    """The (single, shared) flash gid of the current group (None if none)."""
    evt = state["evt"]
    g = state["group"]
    for gid, f in evt.flash_by_gid.items():
        if f["group"] == g:
            return gid
    return None


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
    selected bundle is locked — it cannot be turned on (and is hidden from the table;
    this is the safety net for the focus/toggle path)."""
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


def bundle_long_enough(i):
    """True if bundle i's main cluster passes the length filter (or it is off). The
    length is the cluster bbox-diagonal proxy (Event.cluster_length)."""
    if not state["len_filter_on"]:
        return True
    evt = state["evt"]
    return evt.cluster_length(evt.bundles[i]["main_cluster"]) > MIN_CLUS_LEN


def bundle_locked(i):
    """True if the match filter is on and bundle i's cluster is already claimed by a
    *different* selected bundle, so it is HIDDEN (not merely lock-glyphed) from the
    table and group navigation."""
    return (state["filter_on"] and i not in state["selected"]
            and cluster_eliminated(i))


def group_visible_bundles(g):
    """Indices of the bundles shown for flash group g under the active filters
    (the >MIN_CLUS_LEN length filter and, when the match filter is on, the lock that
    hides bundles whose cluster is already matched elsewhere). Drives both the bundle
    table and the group navigation, so a group whose clusters are all short, or all
    locked, shows/navigates as empty."""
    evt = state["evt"]
    return [i for i in range(len(evt.bundles))
            if evt.group_of(evt.bundles[i]["flash_gid"]) == g
            and bundle_long_enough(i) and not bundle_locked(i)]


def visible_count(g):
    """Bundles shown for group g under the active filters (length filter). A group with
    no length-passing bundle counts 0, so navigation + dropdown skip it."""
    return len(group_visible_bundles(g))


# ---------------------------------------------------------------------------
# Bokeh widgets / figures.
# ---------------------------------------------------------------------------
event_select = Select(title="Event", value=(LABELS[0] if LABELS else ""),
                      options=LABELS, width=180)
prev_evt_btn = Button(label="< prev evt", width=90)
next_evt_btn = Button(label="next evt >", width=90)
group_select = Select(title="flash group", value="", options=[], width=240)
prev_grp_btn = Button(label="< prev grp", width=90)
next_grp_btn = Button(label="next grp >", width=90)
select_btn = Button(label="Toggle match (focused)", button_type="primary", width=200)
loadauto_btn = Button(label="Load auto-match", width=140)
clear_btn = Button(label="Clear selections", width=140)
save_btn = Button(label="Save labels", button_type="success", width=120)
compare_btn = Button(label="Compare cluster's flashes", width=200)
filter_btn = Toggle(label="Filter selected bundles: ON", width=220,
                    active=True, button_type="warning")
len_filter_btn = Toggle(label="Hide clusters ≤5cm: ON", width=190,
                        active=True, button_type="warning")
status = Div(text="", width=1100)
metrics = Div(text="", width=560)
selsummary = Div(text="", width=380)
compare_div = Div(text="", width=1000)
# Echo of the event / flash group / load-summary, shown beneath the three
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
CATH_BG = "#f6e0c0"  # light orange for at-cathode bundles (cathode-hugging cluster)

# Selection is driven by a real CheckboxGroup (reliable clickable boxes) beside the
# table — one box per current-group bundle, in table-row order. Ticking a box adds the
# bundle to the selection (its predicted light joins the per-flash sum); tick several
# to combine clusters (e.g. BOTH halves of a cathode crosser on the shared flash).
sel_group = CheckboxGroup(labels=[], active=[], width=380)
sel_group_title = Div(text="<b>select matches</b> (tick to add to predicted sum)", width=380)

table_src = ColumnDataSource(data=dict())
table_cols = [
    TableColumn(field="row", title="#", width=30, formatter=fmt_l, sortable=False),
    TableColumn(field="auto", title="auto", width=45, formatter=fmt_l, sortable=False),
    TableColumn(field="apa", title="vol", width=45, formatter=fmt_l, sortable=False),
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

# Second table: every bundle whose main cluster matches the focused
# bundle's, across all flashes/groups, so one cluster's candidate flashes can be
# compared side by side. The dump only emits contained bundles, so the list is
# already the physically-feasible candidate set (cluster inside the drift box at that
# flash's T0). Populated on demand by the "Compare" button; click a row to jump the
# whole view (focus + group) to that candidate flash.
compare_src = ColumnDataSource(data=dict())
# sortable=False on every column (as the main table): a click reports a row index that
# we map through state["compare_order"] (build order). User-sorting a column reorders the
# view without reordering compare_order, so the index would resolve to the wrong bundle.
compare_cols = [
    TableColumn(field="sel", title="✓", width=34, formatter=check_fmt, sortable=False),
    TableColumn(field="auto", title="auto", width=45, sortable=False),
    TableColumn(field="apa", title="vol", width=45, sortable=False),
    TableColumn(field="flash_gid", title="flash", width=70, sortable=False),
    TableColumn(field="cluster", title="clus", width=50, sortable=False),
    TableColumn(field="t_us", title="t(us)", width=70,
                formatter=NumberFormatter(format="0.0"), sortable=False),
    TableColumn(field="grp", title="grp", width=40, sortable=False),
    TableColumn(field="ks", title="ks", width=55,
                formatter=NumberFormatter(format="0.000"), sortable=False),
    TableColumn(field="chi2ndf", title="chi2/ndf", width=70,
                formatter=NumberFormatter(format="0.0"), sortable=False),
    TableColumn(field="strength", title="strength", width=70,
                formatter=NumberFormatter(format="0.000"), sortable=False),
    TableColumn(field="meas", title="measPE", width=70,
                formatter=NumberFormatter(format="0"), sortable=False),
    TableColumn(field="pred", title="predPE", width=70,
                formatter=NumberFormatter(format="0.0"), sortable=False),
    TableColumn(field="flags", title="flags", width=150, sortable=False),
]
compare_table = DataTable(source=compare_src, columns=compare_cols, width=1000,
                          height=200, selectable=True, index_position=None)

# Cluster roster (all clusters in the event, both drift volumes): ident, volume,
# #points, length (bbox diagonal, cm), and whether it has been matched (✓ + the flash
# gid) — so you can see at a glance which clusters are still unassigned.
clus_title = Div(text="<b>clusters</b> (✓ = matched)", width=330)
clus_src = ColumnDataSource(data=dict())
# sortable=False on every column (as the main table): on_clus_select maps the clicked
# row index through state["clus_order"] (build order, longest-first). User-sorting a
# column (e.g. the "clus" header to find a cluster) reorders only the view, so the index
# would resolve to a neighbouring cluster.
clus_cols = [
    TableColumn(field="sel", title="✓", width=30, formatter=check_fmt, sortable=False),
    TableColumn(field="cluster", title="clus", width=50, sortable=False),
    TableColumn(field="apa", title="vol", width=40, sortable=False),
    TableColumn(field="flash_gid", title="→flash", width=60, sortable=False),
    TableColumn(field="npts", title="npts", width=55, sortable=False),
    TableColumn(field="length", title="len(cm)", width=65,
                formatter=NumberFormatter(format="0.0"), sortable=False),
]
clus_table = DataTable(source=clus_src, columns=clus_cols, width=330, height=300,
                       selectable=True, index_position=None)

# Light-pattern figures (Y vertical, Z horizontal): a 2x2 grid of measured vs
# predicted for the XA and PMT PD-type groups. The SAME shared flash lights every
# panel. Positions are fixed, so the ranges are pinned to the PD envelope (no zoom).
# Each group's two sub-blocks (see GROUPS comment) do not separate spatially (e.g.
# PMT wall/bottom z-ranges overlap), so they are drawn with distinct marker families
# instead: circle = wall sub-block, square = cathode/bottom sub-block. Flash OpDets
# are sized by sqrt(PE); a shared LinearColorMapper keeps both markers on one scale.
def radii(vals, hi):
    # circle radius in cm, scaled by sqrt(PE/hi); 0-PE channels get a small dot
    return (2.5 + 9.0 * np.sqrt(np.clip(vals, 0, None) / hi)).tolist()


def make_light_fig(title):
    f = figure(title=title, height=280, width=430, tools="pan,reset,save")
    f.xaxis.axis_label = "z (cm)"
    f.yaxis.axis_label = "y (cm)"
    base_wall = ColumnDataSource(data=dict(z=[], y=[], ch=[]))
    base_inner = ColumnDataSource(data=dict(z=[], y=[], ch=[]))
    dead_wall = ColumnDataSource(data=dict(z=[], y=[], ch=[]))
    dead_inner = ColumnDataSource(data=dict(z=[], y=[], ch=[]))
    src_wall = ColumnDataSource(data=dict(z=[], y=[], pe=[], r=[], ch=[]))
    src_inner = ColumnDataSource(data=dict(z=[], y=[], pe=[], d=[], ch=[]))
    # masked / dead OpDets of this group (static ch_mask + auto_mask), drawn first as
    # faint red x/+ marks so the live array reads against the FULL physical PD layout
    # (PDVD's static mask: no-WLS XA 13, dead PMTs 24/27/28/34, Ar-blind PMTs 29/32/39).
    b_dead_wall = f.scatter("z", "y", source=dead_wall, marker="circle_x", size=8,
                            line_color="#e0a0a0", line_width=1.3)
    b_dead_inner = f.scatter("z", "y", source=dead_inner, marker="square_x", size=8,
                             line_color="#e0a0a0", line_width=1.3)
    b_base_wall = f.scatter("z", "y", source=base_wall, marker="circle", size=8,
                            fill_color=None, line_color="#8a8a8a")
    b_base_inner = f.scatter("z", "y", source=base_inner, marker="square", size=7,
                             fill_color=None, line_color="#8a8a8a")
    cmapper = LinearColorMapper(palette="Viridis256", low=0, high=1)
    g_wall = f.circle("z", "y", source=src_wall, radius="r",
                      fill_color={"field": "pe", "transform": cmapper},
                      line_color="#333333", fill_alpha=0.85)
    g_inner = f.rect("z", "y", source=src_inner, width="d", height="d",
                     fill_color={"field": "pe", "transform": cmapper},
                     line_color="#333333", fill_alpha=0.85)
    # saturation-flagged OpDets of the current flash (DAPHNE rail): orange ring
    # drawn on top of the live marker. The measured PE there is the REPAIRED
    # value and the prediction is shown in full -- the flag only means the
    # channel is excluded from the chi2/KS fit.
    sat_wall = ColumnDataSource(data=dict(z=[], y=[], ch=[]))
    sat_inner = ColumnDataSource(data=dict(z=[], y=[], ch=[]))
    f.scatter("z", "y", source=sat_wall, marker="circle", size=16,
              fill_color=None, line_color="#ff7f0e", line_width=2.2)
    f.scatter("z", "y", source=sat_inner, marker="square", size=15,
              fill_color=None, line_color="#ff7f0e", line_width=2.2)
    # readout-uncovered OpDets of the current flash (self-trigger channels
    # with no snippet over the flash window): NO DATA -- the measured 0 is
    # not a measurement and the channel is excluded from the fit.
    cov_wall = ColumnDataSource(data=dict(z=[], y=[], ch=[]))
    cov_inner = ColumnDataSource(data=dict(z=[], y=[], ch=[]))
    f.scatter("z", "y", source=cov_wall, marker="circle_x", size=16,
              fill_color=None, line_color="#7f7f7f", line_width=1.8)
    f.scatter("z", "y", source=cov_inner, marker="square_x", size=15,
              fill_color=None, line_color="#7f7f7f", line_width=1.8)
    cbar = ColorBar(color_mapper=cmapper, title="PE",
                    ticker=BasicTicker(desired_num_ticks=6),
                    formatter=NumeralTickFormatter(format="0,0"),
                    width=14, padding=4, label_standoff=6,
                    title_text_font_size="11px", title_standoff=6,
                    major_label_text_font_size="10px")
    f.add_layout(cbar, "right")
    # channels sharing a projected position (see od_jitter) are spread into a
    # small ring around the true point -- "ch" in the tooltip is what actually
    # tells them apart there.
    f.add_tools(HoverTool(renderers=[g_wall, g_inner],
                          tooltips=[("ch", "@ch"), ("PE", "@pe{0,0.0}")]))
    f.add_tools(HoverTool(renderers=[b_base_wall, b_base_inner,
                                     b_dead_wall, b_dead_inner],
                          tooltips=[("ch", "@ch")]))
    return dict(fig=f, base_wall=base_wall, base_inner=base_inner,
               dead_wall=dead_wall, dead_inner=dead_inner,
               src_wall=src_wall, src_inner=src_inner, cmapper=cmapper,
               sat_wall=sat_wall, sat_inner=sat_inner,
               cov_wall=cov_wall, cov_inner=cov_inner)


# LIGHT[group] = {"meas": panel, "pred": panel}  (group = "XA" / "PMT")
LIGHT = {group: {"meas": make_light_fig("%s measured" % GROUP_NAME[group]),
                 "pred": make_light_fig("%s predicted" % GROUP_NAME[group])}
         for group in GROUPS}


# 1-D per-channel comparison below the 2x2 light grid: for each PD-type group an
# overlay of measured vs predicted PE over that group's active OpDets (wall
# sub-block first, then cathode/bottom, with a dashed separator + labels between
# them), and the pred/meas ratio.
def make_hist_fig(title):
    f = figure(title=title, height=220, width=430,
               tools="pan,box_zoom,wheel_zoom,reset,save")
    f.xaxis.axis_label = "active OpDet (channel)"
    f.yaxis.axis_label = "PE"
    meas_src = ColumnDataSource(data=dict(x=[], pe=[]))
    pred_src = ColumnDataSource(data=dict(x=[], pe=[]))
    sat_src = ColumnDataSource(data=dict(x=[], pe=[]))
    f.vbar(x="x", top="pe", width=0.9, source=meas_src,
           fill_color="#6baed6", line_color=None, fill_alpha=0.6,
           legend_label="measured")
    f.line("x", "pe", source=pred_src, line_color="#d62728", line_width=1.5,
           legend_label="predicted")
    f.scatter("x", "pe", source=pred_src, marker="circle", size=4,
              fill_color="#d62728", line_color=None, legend_label="predicted")
    # rail-flagged channels: repaired measured PE, shown but excluded from
    # the chi2/KS fit (and from the pred/meas ratio panel).
    f.scatter("x", "pe", source=sat_src, marker="triangle", size=9,
              fill_color="#ff7f0e", line_color=None,
              legend_label="saturated (excl. fit)")
    # readout-uncovered channels: no snippet over the flash window, the 0 is
    # NOT a measurement; marker drawn at the predicted PE.
    cov_src = ColumnDataSource(data=dict(x=[], pe=[]))
    f.scatter("x", "pe", source=cov_src, marker="circle_x", size=9,
              fill_color=None, line_color="#7f7f7f", line_width=1.5,
              legend_label="no data (excl. fit)")
    f.legend.label_text_font_size = "9px"
    f.legend.padding = 2
    f.legend.location = "top_right"
    f.legend.background_fill_alpha = 0.6
    sep, lbl_wall, lbl_inner = make_subblock_sep(f)
    return dict(fig=f, meas=meas_src, pred=pred_src, sat=sat_src, cov=cov_src,
               sep=sep, lbl_wall=lbl_wall, lbl_inner=lbl_inner)


def make_subblock_sep(f):
    """Dashed vertical separator + wall/inner sub-block labels, shared by the
    overlay and ratio figs; hidden until render_light positions them (the
    boundary is event-dependent -- it counts *active* wall channels)."""
    sep = Span(location=-1, dimension="height", line_color="#666666",
              line_dash="dashed", line_width=1.3, visible=False)
    f.add_layout(sep)
    lbl_wall = Label(x=0, y=203, y_units="screen", text="", text_font_size="9px",
                     text_color="#666666", visible=False)
    lbl_inner = Label(x=0, y=203, y_units="screen", text="", text_font_size="9px",
                      text_color="#666666", visible=False)
    f.add_layout(lbl_wall)
    f.add_layout(lbl_inner)
    return sep, lbl_wall, lbl_inner


def make_ratio_fig(title):
    f = figure(title=title, height=220, width=430,
               tools="pan,box_zoom,wheel_zoom,reset,save")
    f.xaxis.axis_label = "active OpDet (channel)"
    f.yaxis.axis_label = "pred / meas"
    src = ColumnDataSource(data=dict(x=[], ratio=[]))
    f.scatter("x", "ratio", source=src, marker="circle", size=5,
              fill_color="#2ca02c", line_color=None, fill_alpha=0.8)
    f.add_layout(Span(location=1.0, dimension="width", line_color="#888888",
                      line_dash="dashed", line_width=1))
    sep, lbl_wall, lbl_inner = make_subblock_sep(f)
    return dict(fig=f, src=src, sep=sep, lbl_wall=lbl_wall, lbl_inner=lbl_inner)


# HIST[group] = {"overlay": panel, "ratio": panel}
HIST = {group: {"overlay": make_hist_fig("%s meas vs pred" % GROUP_NAME[group]),
               "ratio": make_ratio_fig("%s pred/meas" % GROUP_NAME[group])}
        for group in GROUPS}

# Charge-projection figures (focused bundle's clusters, T0-shifted, in the fixed
# detector box; both volume boxes drawn). XY, YZ (z horiz, y vert), XZ (x horiz, z
# vert). In PDVD x is the VERTICAL drift coordinate (bottom anode -335.8, cathode ~0,
# top anode +335.8).
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
    if b["close_to_PMT"]:        parts.append("nearPD")
    if b.get("at_cathode"):      parts.append("atCATH")
    if b["at_x_boundary"]:       parts.append("xbound")
    if b["spec_end"]:            parts.append("specend")
    if b["window_truncated"]:    parts.append("wtrunc")
    if b.get("two_boundary"):    parts.append("2bnd")
    if not b["contained"]:       parts.append("UNCONTAINED")
    return ",".join(parts)


def rebuild_table():
    evt = state["evt"]
    g = state["group"]
    # Show the current group's bundles that pass the active filters (length filter
    # hides clusters <= MIN_CLUS_LEN; when the match filter is on, bundles whose cluster
    # is already matched elsewhere are hidden entirely — see group_visible_bundles).
    visible = group_visible_bundles(g)
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
        cols["apa"].append(SIDE_NAME[b["apa"]][:3])
        cols["flash_gid"].append(str(b["flash_gid"]))
        cols["t_us"].append("%.1f" % evt.flash_by_gid[b["flash_gid"]]["time"])
        cols["grp"].append(str(evt.group_of(b["flash_gid"])))
        cols["cluster"].append(str(cu_id))
        cols["noth"].append(str(len(b["other_clusters"])))
        cols["ks"].append("%.3f" % b["ks_dis"])
        cols["chi2ndf"].append("%.1f" % (b["chi2"] / ndf))
        cols["strength"].append("%.3f" % b["strength"])
        cols["meas"].append("%.0f" % b["total_PE"])
        cols["pred"].append("%.1f" % b["total_pred_light"])
        cols["flags"].append(fmt_flags(b))
        # green tint persists for selected bundles (distinct from blue click-focus);
        # else a light-orange tint marks cathode-hugging (at_cathode) candidates,
        # whose T0 is weakly constrained (a later T0 just slides them along drift).
        cols["sel_bg"].append(SEL_BG if i in state["selected"]
                              else CATH_BG if b.get("at_cathode") else "transparent")
        labels.append("%d: %s fl%d c%d  ks%.2f pr%.0f"
                      % (r, SIDE_NAME[b["apa"]][:3], b["flash_gid"], cu_id,
                         b["ks_dis"], b["total_pred_light"]))
    # Bokeh's DataTable (SlickGrid) repaints only when the row COUNT changes: replacing
    # source.data with the SAME number of rows leaves the old cells on screen. Force a
    # count change -- blank the source now (length -> 0, flushed at the end of this
    # callback) and refill on the next tick (0 -> N) -- so the grid always re-renders.
    new_data = dict(cols)
    sel_idx = ([order.index(state["focus"])]
               if state["focus"] is not None and state["focus"] in order else [])
    table_src.data = {k: [] for k in new_data}

    def _fill_table(data=new_data, idx=sel_idx):
        table_src.data = data
        table_src.selected.indices = idx
    curdoc().add_next_tick_callback(_fill_table)

    # the CheckboxGroup is the clickable selector: labels in table-row order, the
    # checked boxes = the selected bundles of this group. Guard the programmatic
    # `active` write so it does not re-fire on_sel_group.
    active = [r for r, i in enumerate(order) if i in state["selected"]]
    state["suppress_edit"] = True
    sel_group.labels = labels
    sel_group.active = active
    state["suppress_edit"] = False


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


def fill_group_2d(panel, evt, chans, boundary, vals, hi):
    """Split a group's active-channel values into the wall/inner sub-blocks and
    write the two ColumnDataSources (circle=wall, rect=cathode/bottom). Positions
    are jittered (od_jitter) so same-(y,z) channels (front/back XA or PMT rows)
    don't draw on top of each other."""
    wi, ii = chans[:boundary], chans[boundary:]
    rw = radii(vals[:boundary], hi)
    ri = radii(vals[boundary:], hi)
    panel["src_wall"].data = dict(z=(evt.od_z[wi] + evt.od_jitter_z[wi]).tolist(),
                                  y=(evt.od_y[wi] + evt.od_jitter_y[wi]).tolist(),
                                  pe=vals[:boundary].tolist(), r=rw, ch=wi.tolist())
    panel["src_inner"].data = dict(z=(evt.od_z[ii] + evt.od_jitter_z[ii]).tolist(),
                                   y=(evt.od_y[ii] + evt.od_jitter_y[ii]).tolist(),
                                   pe=vals[boundary:].tolist(),
                                   d=(np.array(ri) * 2.0).tolist(), ch=ii.tolist())


def set_subblock_sep(hpanel, group, boundary, n):
    """Position/show the dashed separator + wall/inner labels on a histogram
    panel, or hide them where a sub-block has no active channels (edge case:
    boundary == 0 or == n)."""
    wname, iname = SUBNAME[group]
    if 0 < boundary < n:
        hpanel["sep"].location = boundary - 0.5
        hpanel["sep"].visible = True
    else:
        hpanel["sep"].visible = False
    if boundary > 0:
        hpanel["lbl_wall"].x = (boundary - 1) / 2.0
        hpanel["lbl_wall"].text = wname
        hpanel["lbl_wall"].visible = True
    else:
        hpanel["lbl_wall"].visible = False
    if boundary < n:
        hpanel["lbl_inner"].x = boundary + (n - boundary - 1) / 2.0
        hpanel["lbl_inner"].text = iname
        hpanel["lbl_inner"].visible = True
    else:
        hpanel["lbl_inner"].visible = False


def render_light():
    """Two PD-type panels (X-ARAPUCA, PMT), each measured + predicted: the SAME
    shared flash lights every channel, so predicted is the sum of ALL selected
    bundles on the group flash across BOTH drift volumes (a cathode X-ARAPUCA's
    predicted light is not confined to one volume's clusters -- summing only one
    volume, as a bottom/top split would, under-counts it). Each group's wall and
    cathode/bottom sub-blocks (see GROUPS) are marked distinctly in the 2-D maps
    (circle vs square) and divided by a dashed separator + labels in the 1-D
    histograms."""
    evt = state["evt"]
    gid = group_flash_gid()

    pred_full = np.zeros(evt.nchan)
    lab = "none selected"
    if gid is not None:
        share = [j for j in state["selected"] if evt.bundles[j]["flash_gid"] == gid]
        previewed = False
        if not share:
            idx = state["focus"]
            if idx is not None and evt.bundles[idx]["flash_gid"] == gid:
                share = [idx]
                previewed = True
        for j in share:
            pj = evt.bundles[j]["pred_pe"]
            if pj:
                pred_full += np.array(pj)
        if share:
            lab = ("preview: focused bundle" if previewed
                   else "sum of %d selected cluster(s)" % len(share))

    for group in GROUPS:
        mp, pp = LIGHT[group]["meas"], LIGHT[group]["pred"]
        ho, hr = HIST[group]["overlay"], HIST[group]["ratio"]
        wall_m, inner_m = evt.group_mask(group)
        aw, ai = evt.od_active & wall_m, evt.od_active & inner_m
        dw, di = (~evt.od_active) & wall_m, (~evt.od_active) & inner_m
        # faint outline of this group's active OpDets; red x/+ for the masked ones
        # (jittered, same as the live markers, so the outline lines up with them).
        for panel in (mp, pp):
            panel["base_wall"].data = dict(
                z=(evt.od_z[aw] + evt.od_jitter_z[aw]).tolist(),
                y=(evt.od_y[aw] + evt.od_jitter_y[aw]).tolist(), ch=np.nonzero(aw)[0].tolist())
            panel["base_inner"].data = dict(
                z=(evt.od_z[ai] + evt.od_jitter_z[ai]).tolist(),
                y=(evt.od_y[ai] + evt.od_jitter_y[ai]).tolist(), ch=np.nonzero(ai)[0].tolist())
            panel["dead_wall"].data = dict(
                z=(evt.od_z[dw] + evt.od_jitter_z[dw]).tolist(),
                y=(evt.od_y[dw] + evt.od_jitter_y[dw]).tolist(), ch=np.nonzero(dw)[0].tolist())
            panel["dead_inner"].data = dict(
                z=(evt.od_z[di] + evt.od_jitter_z[di]).tolist(),
                y=(evt.od_y[di] + evt.od_jitter_y[di]).tolist(), ch=np.nonzero(di)[0].tolist())

        chans, boundary = evt.group_channels(group)

        if gid is None:
            for panel in (mp, pp):
                panel["src_wall"].data = dict(z=[], y=[], pe=[], r=[], ch=[])
                panel["src_inner"].data = dict(z=[], y=[], pe=[], d=[], ch=[])
                panel["sat_wall"].data = dict(z=[], y=[], ch=[])
                panel["sat_inner"].data = dict(z=[], y=[], ch=[])
                panel["cov_wall"].data = dict(z=[], y=[], ch=[])
                panel["cov_inner"].data = dict(z=[], y=[], ch=[])
            ho["sat"].data = dict(x=[], pe=[])
            ho["cov"].data = dict(x=[], pe=[])
            mp["fig"].title.text = "%s measured  (no flash in group)" % GROUP_NAME[group]
            pp["fig"].title.text = "%s predicted" % GROUP_NAME[group]
            ho["meas"].data = dict(x=[], pe=[])
            ho["pred"].data = dict(x=[], pe=[])
            hr["src"].data = dict(x=[], ratio=[])
            ho["fig"].title.text = "%s meas vs pred  (no flash)" % GROUP_NAME[group]
            hr["fig"].title.text = "%s pred/meas" % GROUP_NAME[group]
            for hpanel in (ho, hr):
                hpanel["sep"].visible = False
                hpanel["lbl_wall"].visible = False
                hpanel["lbl_inner"].visible = False
            continue
        flash = evt.flash_by_gid[gid]
        meas = np.array(flash["pe"])[chans]
        pred = pred_full[chans]
        # per-flash DAPHNE-rail flags (dump "sat" array; absent on pre-flag
        # dumps): measured is the repaired value, prediction is the FULL
        # prediction -- the flag only removes the channel from chi2/KS.
        sat = (np.array(flash["sat"])[chans].astype(bool)
               if flash.get("sat") else np.zeros(meas.size, bool))
        # readout-coverage fractions (dump "cov" array; absent on archives
        # without the emit_coverage chain): cov < 1 = the channel had no
        # (full) snippet over the flash window -- its 0 is NOT a measurement.
        nodata = (np.array(flash["cov"])[chans] < 1.0
                  if flash.get("cov") else np.zeros(meas.size, bool))

        # Independent per-panel scales so the predicted *shape* stays readable even
        # when its absolute PE is tiny next to a bright measured flash.
        hi_meas = max(1.0, float(meas.max() if meas.size else 0))
        hi_pred = max(1.0, float(pred.max() if pred.size else 0))
        mp["cmapper"].high = hi_meas
        pp["cmapper"].high = hi_pred

        fill_group_2d(mp, evt, chans, boundary, meas, hi_meas)
        fill_group_2d(pp, evt, chans, boundary, pred, hi_pred)
        sw = chans[:boundary][sat[:boundary]]
        si = chans[boundary:][sat[boundary:]]
        cw = chans[:boundary][nodata[:boundary]]
        ci = chans[boundary:][nodata[boundary:]]
        for panel in (mp, pp):
            panel["sat_wall"].data = dict(
                z=(evt.od_z[sw] + evt.od_jitter_z[sw]).tolist(),
                y=(evt.od_y[sw] + evt.od_jitter_y[sw]).tolist(), ch=sw.tolist())
            panel["sat_inner"].data = dict(
                z=(evt.od_z[si] + evt.od_jitter_z[si]).tolist(),
                y=(evt.od_y[si] + evt.od_jitter_y[si]).tolist(), ch=si.tolist())
            panel["cov_wall"].data = dict(
                z=(evt.od_z[cw] + evt.od_jitter_z[cw]).tolist(),
                y=(evt.od_y[cw] + evt.od_jitter_y[cw]).tolist(), ch=cw.tolist())
            panel["cov_inner"].data = dict(
                z=(evt.od_z[ci] + evt.od_jitter_z[ci]).tolist(),
                y=(evt.od_y[ci] + evt.od_jitter_y[ci]).tolist(), ch=ci.tolist())
        mp["fig"].title.text = ("%s measured  gid %d (id %d)  t=%.1f us  totPE=%.0f"
                                % (GROUP_NAME[group], gid, flash["id"], flash["time"],
                                   flash["total_PE"]))
        pp["fig"].title.text = "%s predicted  (%s)" % (GROUP_NAME[group], lab)

        # 1-D per-channel comparison (same meas/pred over the active PDs, wall block
        # first then cathode/bottom, separator between).
        xidx = np.arange(meas.size)
        ho["meas"].data = dict(x=xidx.tolist(), pe=meas.tolist())
        ho["pred"].data = dict(x=xidx.tolist(), pe=pred.tolist())
        ho["sat"].data = dict(x=xidx[sat].tolist(), pe=meas[sat].tolist())
        # no-data marker drawn at the PREDICTED PE (measured is a fake 0)
        ho["cov"].data = dict(x=xidx[nodata].tolist(), pe=pred[nodata].tolist())
        # ratio undefined where measured PE is 0; rail-flagged and uncovered
        # channels are excluded from the fit, so drop them here too.
        mask = (meas > 0) & ~sat & ~nodata
        hr["src"].data = dict(x=xidx[mask].tolist(),
                              ratio=(pred[mask] / meas[mask]).tolist())
        ho["fig"].title.text = ("%s meas vs pred  (gid %d, t=%.1f us)"
                                % (GROUP_NAME[group], gid, flash["time"]))
        hr["fig"].title.text = ("%s pred/meas  (%d/%d chans meas>0%s%s)"
                                % (GROUP_NAME[group], int(mask.sum()), meas.size,
                                   ", %d sat excl." % int(sat.sum())
                                   if sat.any() else "",
                                   ", %d no-data" % int(nodata.sum())
                                   if nodata.any() else ""))
        for hpanel in (ho, hr):
            set_subblock_sep(hpanel, group, boundary, meas.size)


def box_lines(g):
    """Return the rectangle polylines for one drift box in the three projections."""
    ax, cx = g["anode_x"], g["cathode_x"]
    ylo, yhi, zlo, zhi = g["y_lo"], g["y_hi"], g["z_lo"], g["z_hi"]
    xy = ([ax, cx, cx, ax, ax], [ylo, ylo, yhi, yhi, ylo])      # x-y
    yz = ([zlo, zhi, zhi, zlo, zlo], [ylo, ylo, yhi, yhi, ylo])  # z-y
    xz = ([ax, cx, cx, ax, ax], [zlo, zlo, zhi, zhi, zlo])       # x-z
    return xy, yz, xz


def render_projections():
    evt = state["evt"]
    # both drift boxes always drawn (the fixed detector frame)
    xs_xy, ys_xy, xs_yz, ys_yz, xs_xz, ys_xz = [], [], [], [], [], []
    for apa in sorted(evt.geom):
        xy, yz, xz = box_lines(evt.geom[apa])
        xs_xy.append(xy[0]); ys_xy.append(xy[1])
        xs_yz.append(yz[0]); ys_yz.append(yz[1])
        xs_xz.append(xz[0]); ys_xz.append(xz[1])
    box_src.data = dict(xs_xy=xs_xy, ys_xy=ys_xy, xs_yz=xs_yz, ys_yz=ys_yz,
                        xs_xz=xs_xz, ys_xz=ys_xz)

    # context: ALL selected matches (any group, both volumes), T0-shifted, drawn gray;
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
        ("flash", "gid %d / id %d / group %d"
         % (b["flash_gid"], b["flash_id"], evt.group_of(b["flash_gid"]))),
        ("flash time", "%.2f us" % evt.flash_by_gid[b["flash_gid"]]["time"]),
        ("volume", "%s (apa %d)" % (SIDE_NAME[b["apa"]], b["apa"])),
        ("main cluster", str(evt.cluster_by_uid[b["main_cluster"]]["ident"])),
        ("assoc clusters", others),
        ("ks_dis", "%.4f  (consist if &lt; %.3f)" % (b["ks_dis"], qp.get("highconsist_ks_max", 0))),
        ("chi2 / ndf", "%.1f / %d = %.2f" % (b["chi2"], b["ndf"], b["chi2"] / ndf)),
        ("ndf", "%d  (consist if &ge; %d)" % (b["ndf"], qp.get("highconsist_min_ndf", 0))),
        ("strength", "%.4f  (cutoff %.3f)" % (b["strength"], qp.get("strength_cutoff", 0))),
        ("measured PE", "%.0f" % b["total_PE"]),
        ("predicted PE", "%.1f" % b["total_pred_light"]),
        ("flag_high_consistent", str(b["consistent"])),
        ("at_cathode", str(bool(b.get("at_cathode")))),
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
            cls += ["%s:%d" % (SIDE_NAME[b["apa"]][:3], evt.cluster_by_uid[u]["ident"])
                    for u in [b["main_cluster"]] + b["other_clusters"]]
        lines.append("flash %d (t=%.1fus, grp%d) &larr; clusters %s"
                     % (gid, f["time"], f["group"], cls))
    selsummary.text = ("<b>%d selected match(es):</b> %s<br>"
                       % (len(state["selected"]), auto_diff_line())
                       + "<br>".join(lines))


def rebuild_compare():
    """Fill the compare table with every bundle sharing the focused cluster's uid
    (one cluster -> naturally one volume), across all flashes/groups. The dump only
    emits contained bundles, so these are exactly the cluster's physically
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
        cols["apa"].append(SIDE_NAME[b["apa"]][:3])
        cols["flash_gid"].append(b["flash_gid"])
        cols["cluster"].append(evt.cluster_by_uid[b["main_cluster"]]["ident"])
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
    compare_div.text = ("<b>cluster %d (%s)</b> &mdash; %d candidate flash(es); "
                        "click a row to jump to that flash."
                        % (c["ident"], SIDE_NAME[c["apa"]], len(rows)))
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
    """Roster of every cluster in the event (both volumes): ident, volume, #points,
    length, and the flash it is matched to (✓) if any — so the unassigned clusters
    are obvious."""
    evt = state["evt"]
    matched = {}                            # cluster uid -> flash gid (via selection)
    for j in state["selected"]:
        b = evt.bundles[j]
        for u in [b["main_cluster"]] + b["other_clusters"]:
            matched[u] = b["flash_gid"]
    uids = list(evt.cluster_by_uid)
    # length filter (default on): hide the many short fragments (<= MIN_CLUS_LEN cm)
    if state["len_filter_on"]:
        uids = [u for u in uids if evt.cluster_length(u) > MIN_CLUS_LEN]
    # longest clusters first, then a stable tiebreak (volume, ident)
    uids = sorted(uids, key=lambda u: (-evt.cluster_length(u),
                                       evt.cluster_by_uid[u]["apa"],
                                       evt.cluster_by_uid[u]["ident"]))
    state["clus_order"] = uids              # roster row index -> cluster uid (for Compare)
    cols = defaultdict(list)
    for u in uids:
        c = evt.cluster_by_uid[u]
        cols["sel"].append("✔" if u in matched else "")
        cols["cluster"].append(c["ident"])
        cols["apa"].append(SIDE_NAME[c["apa"]][:3])
        cols["npts"].append(c["npoints"])
        cols["length"].append(evt.cluster_length(u))
        cols["flash_gid"].append(matched.get(u, ""))
    clus_src.selected.indices = []          # force repaint on same-row-set change
    clus_src.data = dict(cols)


def render_proj_info():
    """Echo, beneath the projection plots, the same Event / flash group /
    load-summary shown by the controls + status at the top of the page."""
    evt = state["evt"]
    if evt is None:
        proj_info.text = ""
        return
    g = state["group"]
    grp = group_label(evt, g) if g is not None else "-"
    proj_info.text = (
        "<b>Event</b> %s &nbsp;&nbsp; <b>flash group</b> %s<br>%s"
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
    """Pin each light panel to its PD-type group's envelope (fixed, no zoom). PDVD's
    wall PDs sit OUTSIDE the drift box footprint (membrane XAs at y=±417.6, bottom
    PMTs out to z=455/-156), so the range is the PD-position envelope of the
    panel's channels, unioned with both drift boxes (the group may span both
    volumes), padded."""
    for group in GROUPS:
        wall_m, inner_m = evt.group_mask(group)
        sm = wall_m | inner_m
        if not sm.any():
            continue
        zlo, zhi = float(evt.od_z[sm].min()), float(evt.od_z[sm].max())
        ylo, yhi = float(evt.od_y[sm].min()), float(evt.od_y[sm].max())
        for g in evt.geom.values():
            zlo, zhi = min(zlo, g["z_lo"]), max(zhi, g["z_hi"])
            ylo, yhi = min(ylo, g["y_lo"]), max(yhi, g["y_hi"])
        pz = 0.06 * (zhi - zlo)
        py = 0.06 * (yhi - ylo)
        for panel in (LIGHT[group]["meas"], LIGHT[group]["pred"]):
            f = panel["fig"]
            f.x_range.start, f.x_range.end = zlo - pz, zhi + pz
            f.y_range.start, f.y_range.end = ylo - py, yhi + py


def labels_dir(evt):
    """Persistent scan-output dir: a sibling of the per-event work/<run6>_<evtidx>/
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
    groups = sorted({evt.group_of(b["flash_gid"]) for b in evt.bundles})
    state["groups"] = groups
    # With the length filter on (default) open on the first group that still has a
    # cluster longer than the cutoff, so the scan doesn't land on an all-short group.
    vis = [g for g in groups if visible_count(g) > 0]
    g0 = vis[0] if vis else (groups[0] if groups else None)
    state["group"] = g0
    set_light_ranges(evt)
    # Populate the group selector (setting .value may re-fire on_group_change,
    # which just re-refreshes — harmless).
    group_select.options = [(str(g), group_label(evt, g)) for g in groups]
    group_select.value = str(g0) if g0 is not None else ""
    n = len(evt.bundles)
    nsel = sum(b["auto_selected"] for b in evt.bundles)
    summary = ("Loaded <b>%s</b>: %d contained bundles, %d shared flashes, %d clusters; "
               "%d auto-selected. Pick a flash group; tick the ✓ box to select bundles "
               "(a cathode crosser is TWO bundles — bottom + top — on the same flash; "
               "tick both). 'Filter selected bundles' hides bundles reusing a matched "
               "cluster."
               % (label, n, len(evt.flash_by_gid), len(evt.cluster_by_uid), nsel))
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
        # The compare table stays pinned to the cluster from the last 'Compare cluster's
        # flashes' click — focusing a different bundle does NOT re-point it.


def on_toggle():
    if state["focus"] is None:
        status.text = "Select a bundle row first."
        return
    ok, msg = toggle_select(state["focus"])
    b = state["evt"].bundles[state["focus"]]
    status.text = ("bundle [flash %d, cluster %d]: %s"
                   % (b["flash_gid"], cluster_ident(state["focus"]), msg or "no change"))
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
    status.text = ("Filter ON — bundles reusing an already-matched cluster are hidden "
                   "from the table and navigation." if new else
                   "Filter OFF — all bundles shown; any can be selected.")
    if state["evt"] is None:
        return
    refresh()                         # re-trims the table and the group navigation
    nav = state["nav_groups"]
    if nav and state["group"] not in nav:
        group_select.value = str(nav[0])    # current group emptied -> jump (fires refresh)


def on_len_filter(attr, old, new):
    """Toggle the cluster-length filter. ON (default) hides clusters <= MIN_CLUS_LEN
    from the roster + bundle table and drops flash groups whose every cluster is
    that short from the navigation. If the current group just emptied, jump to the
    first group that still has a long-enough cluster."""
    state["len_filter_on"] = bool(new)
    len_filter_btn.label = "Hide clusters ≤5cm: %s" % ("ON" if new else "OFF")
    len_filter_btn.button_type = "warning" if new else "default"
    status.text = (("Length filter ON — clusters ≤ %g cm are hidden, and flash "
                    "groups with no longer cluster are skipped." % MIN_CLUS_LEN)
                   if new else "Length filter OFF — all clusters shown.")
    if state["evt"] is None:
        return
    refresh()                         # re-trims roster, table and the group navigation
    nav = state["nav_groups"]
    if nav and state["group"] not in nav:
        group_select.value = str(nav[0])    # current group emptied -> jump (fires refresh)


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
        status.text = ("cluster %d (%s) picked — click 'Compare cluster's flashes'."
                       % (c["ident"], SIDE_NAME[c["apa"]]))


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
        "flags": {k: b.get(k) for k in ("consistent", "potential_bad_match",
                                        "close_to_PMT", "at_cathode", "at_x_boundary",
                                        "spec_end", "window_truncated", "two_boundary",
                                        "contained", "auto_selected")},
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
len_filter_btn.on_change("active", on_len_filter)


# ---------------------------------------------------------------------------
# Layout.
# ---------------------------------------------------------------------------
# Two layers: navigation on top, actions below (keeps the button strip from
# overflowing a narrow window).
controls_nav = row(event_select, prev_evt_btn, next_evt_btn,
                   group_select, prev_grp_btn, next_grp_btn)
controls_act = row(select_btn, loadauto_btn, clear_btn, save_btn, compare_btn,
                   filter_btn, len_filter_btn)
controls = column(controls_nav, controls_act)
header = Div(text="<h2>PDVD Q/L matching hand-scan</h2>", width=1100)
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
    row(LIGHT["XA"]["meas"]["fig"], LIGHT["XA"]["pred"]["fig"],
        LIGHT["PMT"]["meas"]["fig"], LIGHT["PMT"]["pred"]["fig"]),
    row(HIST["XA"]["overlay"]["fig"], HIST["XA"]["ratio"]["fig"],
        HIST["PMT"]["overlay"]["fig"], HIST["PMT"]["ratio"]["fig"]),
)

curdoc().add_root(layout)
curdoc().title = "PDVD Q/L hand-scan"

if LABELS:
    load_event(LABELS[0])
else:
    status.text = ("<b>No calib JSONs found.</b> Produce them with "
                   "run_clus_evt.sh -calib &lt;run&gt; &lt;evt&gt;, then pass the "
                   "work/&lt;run6&gt;_&lt;evtidx&gt;/calib-evt*.json glob to "
                   "serve_ql_scan.sh.")

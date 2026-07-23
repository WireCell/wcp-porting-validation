#!/usr/bin/env python3
"""SBND neutrino-selection (nusel) hand-scan event display (Bokeh server).

Scans the per-bundle tagger verdicts of the PR chain (TGM / STM / FC) against
the same inputs Bee shows, so a Bee tab can be used side by side:

  * charge  : ql_evt<ID>/mabc-all-apa.zip -> data/0/0-clustering-global.json
              (the exact layer uploaded to Bee: T0-corrected x, per-point
              cluster_id / real_cluster_id)
  * light   : same zip -> data/0/0-op.json (op_pes measured, op_pes_pred
              predicted, per flash)
  * bundles : ql_evt<ID>/pctree-evt<ID>.tar.gz via nusel_extract.parse_pctree
              (cluster ident / matched_flash_gid / flag_main) + group_flashes
              (the 80 ns cross-APA physical-flash grouping the table uses)
  * rows    : nusel_evt<ID>/nusel-evt<ID>.tsv (nusel_extract output: one row
              per qualifying bundle + in-beam no-bundle flashes, with the
              tgm/stm/fc verdicts and the auto label)

Navigation: prev/next event buttons; within an event click a bundle row (or
use the < > bundle buttons) -- one row per physical flash group, like Bee's
> < flash stepper.  Two modes: all bundles, or in-beam flash bundles only.

Per bundle the display shows the three charge projections (X-Y, Y-Z, X-Z)
with the SBND detector box + the taggers' effective fiducial volume, the main
cluster in red and companion (same-flash) clusters in blue (or all bundle
points colored by charge), and the measured vs predicted light pattern of the
flash on both TPC sides.

Scan labels: TGM / STM / FC / LM (light mismatch), multi-select, plus a
free-text comment per bundle and per event.  Everything autosaves to
<work_root>/nusel_labels/<tag>/.scan_state-evt<ID>.json on every change and
"Save labels" writes the actionable per-event JSON
<work_root>/nusel_labels/<tag>/nusel-labels-evt<ID>.json.

Launched by serve_nusel_scan.sh; mirrors ql_scan/ql_scan_viewer.py.
"""

import sys
import os
import glob
import json
import math
import zipfile
from collections import defaultdict

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, DataTable, TableColumn, Select, Button,
                          Div, ColorBar, HoverTool, BasicTicker,
                          NumeralTickFormatter, HTMLTemplateFormatter,
                          CheckboxButtonGroup, Toggle, TextAreaInput)
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # sbnd_xin/ for nusel_extract
from nusel_extract import parse_pctree, group_flashes, read_table, COLUMNS  # noqa: E402

# Max charge points drawn for the gray full-event context (stride-downsampled).
MAX_CTX_PTS = 8000

# Scan labels offered (multi-select). LM = light mismatch (pred vs meas pattern).
SCAN_LABELS = ["TGM", "STM", "FC", "LM"]

# SBND PR fiducial: BoxFiducial sbnd_pr_fv (cfg/pgrapher/experiment/sbnd/
# clus.jsonnet) = the overall sensitive bounds spanning BOTH TPCs, and the
# fv_tolerance margins [-2,-2,-2.5,-2.5,-3,-3] cm that inset it into the
# effective FV the taggers test.  Cathode at x=0.
DET_BOX = dict(x=(-201.05, 201.05), y=(-199.312, 199.312), z=(0.85, 500.15))
FV_BOX = dict(x=(-199.05, 199.05), y=(-196.812, 196.812), z=(3.85, 497.15))

# Cross-APA physical-flash coincidence (us) -- matches nusel_extract FLASH_GROUP_NS.
OP_MATCH_TOL_US = 0.02


# ---------------------------------------------------------------------------
# Inputs: work roots (dirs holding nusel_evt*/ + ql_evt*/) from --args.
# ---------------------------------------------------------------------------
def extract_tag(argv):
    out, tag = [], ""
    it = iter(argv)
    for a in it:
        if a in ("--tag", "-t"):
            tag = next(it, "")
        else:
            out.append(a)
    return tag, out


def discover_events(argv):
    """[(label, tsv_path, work_root)] sorted by event id.  Args are work roots
    (dirs containing nusel_evt<ID>/nusel-evt<ID>.tsv) or explicit tsv paths."""
    found = []
    for a in argv:
        if os.path.isdir(a):
            found += glob.glob(os.path.join(a, "nusel_evt*", "nusel-evt*.tsv"))
        elif any(ch in a for ch in "*?["):
            found += glob.glob(a)
        elif os.path.isfile(a):
            found.append(a)
    out = []
    for f in sorted(set(found)):
        base = os.path.basename(f)                     # nusel-evt<ID>.tsv
        if not (base.startswith("nusel-") and base.endswith(".tsv")):
            continue
        label = base[len("nusel-"):-len(".tsv")]       # evt<ID>
        work_root = os.path.dirname(os.path.dirname(os.path.abspath(f)))
        out.append((label, f, work_root))
    def keyf(e):
        digits = "".join(ch for ch in e[0] if ch.isdigit())
        return (int(digits) if digits else 0, e[0])
    return sorted(out, key=keyf)


SCAN_TAG, _ARGS = extract_tag(sys.argv[1:])
EVENTS = discover_events(_ARGS)
LABELS = [e[0] for e in EVENTS]
EVENT_OF = {e[0]: e for e in EVENTS}


def find_opdets():
    """Opdet positions from wire-cell-data sbnd/photodet/semi-analytical-sbnd.json
    (list index = op channel index).  Resolved via SBND_OPDET_JSON or WIRECELL_PATH."""
    cands = [os.environ.get("SBND_OPDET_JSON", "")]
    for d in os.environ.get("WIRECELL_PATH", "").split(":"):
        if d:
            cands.append(os.path.join(d, "sbnd", "photodet", "semi-analytical-sbnd.json"))
    cands.append("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet/"
                 "semi-analytical-sbnd.json")
    for c in cands:
        if c and os.path.isfile(c):
            with open(c) as fh:
                od = json.load(fh)["OpDets"]
            return (np.array([o["x"] for o in od]),
                    np.array([o["y"] for o in od]),
                    np.array([o["z"] for o in od]))
    return None, None, None


OD_X, OD_Y, OD_Z = find_opdets()
OD_APA = None if OD_X is None else (OD_X > 0).astype(int)   # TPC0 = x<0, TPC1 = x>0


# ---------------------------------------------------------------------------
# Per-event model.
# ---------------------------------------------------------------------------
class Event:
    def __init__(self, label, tsv, work_root):
        self.label = label                              # evt<ID>
        self.work_root = work_root
        self.tsv = tsv
        evtid = label[len("evt"):]
        qldir = os.path.join(work_root, "ql_evt" + evtid)
        self.problems = []

        # --- scan rows (nusel_extract output) --------------------------------
        header, rows = read_table(tsv, COLUMNS[0])
        i = {c: n for n, c in enumerate(header or COLUMNS)}
        self.rows = []
        for r in rows:
            self.rows.append(dict(
                run=int(r[i["run"]]), subrun=int(r[i["subrun"]]),
                event=int(r[i["event"]]),
                main_id=int(r[i["main_id"]]), flash_gid=int(r[i["flash_gid"]]),
                flash_apa=int(r[i["flash_apa"]]), flash_grp=int(r[i["flash_grp"]]),
                t_us=float(r[i["flash_time_us"]]), pe=float(r[i["flash_pe"]]),
                pe_grp=float(r[i["flash_pe_grp"]]), in_beam=int(r[i["in_beam"]]),
                n_bundle=int(r[i["n_bundle"]]), npts_main=int(r[i["npts_main"]]),
                npts_bundle=int(r[i["npts_bundle"]]),
                len_cm=float(r[i["len_main_cm"]]), n_frag=int(r[i["n_frag"]]),
                tgm=int(r[i["tgm"]]), stm=int(r[i["stm"]]), fc=int(r[i["fc"]]),
                auto_label=r[i["label"]]))
        self.rows.sort(key=lambda r: r["t_us"])

        # --- bundle structure from the pctree --------------------------------
        self.clusters = []          # [{ident,gid,main,npts,...}]
        self.flashes = {}           # gid -> {t_us, pe, apa}
        self.fgrp = {}              # gid -> physical-flash group id
        pctree = os.path.join(qldir, f"pctree-{label}.tar.gz")
        if os.path.isfile(pctree):
            try:
                _, self.clusters, self.flashes = parse_pctree(pctree)
                self.fgrp = group_flashes(self.flashes)
            except Exception as e:          # keep scanning even if the tree is bad
                self.problems.append(f"pctree: {e}")
        else:
            self.problems.append(f"missing {pctree}")

        # --- Bee layers (charge points + light) ------------------------------
        self.pts = None             # dict of numpy arrays x,y,z,q,cid,rid
        self.op = None              # dict of lists (op.json)
        qlzip = os.path.join(qldir, "mabc-all-apa.zip")
        if os.path.isfile(qlzip):
            try:
                with zipfile.ZipFile(qlzip) as z:
                    names = z.namelist()
                    cg = [n for n in names if n.endswith("-clustering-global.json")]
                    opn = [n for n in names if n.endswith("-op.json")]
                    if cg:
                        d = json.loads(z.read(cg[0]))
                        self.pts = dict(
                            x=np.array(d["x"], float), y=np.array(d["y"], float),
                            z=np.array(d["z"], float), q=np.array(d["q"], float),
                            cid=np.array(d["cluster_id"], int),
                            rid=np.array(d.get("real_cluster_id",
                                               d["cluster_id"]), int))
                    else:
                        self.problems.append("no clustering-global layer in " + qlzip)
                    if opn:
                        self.op = json.loads(z.read(opn[0]))
                    else:
                        self.problems.append("no op layer in " + qlzip)
            except Exception as e:
                self.problems.append(f"{qlzip}: {e}")
        else:
            self.problems.append(f"missing {qlzip}")

    # --- bundle helpers -----------------------------------------------------
    def bundle_cluster_ids(self, r):
        """(main_ident, [companion idents]) for a scan row (peers = clusters
        matched to the same flash gid, exactly nusel_extract's definition)."""
        if r["main_id"] < 0:
            return None, []
        comps = [c["ident"] for c in self.clusters
                 if c["gid"] == r["flash_gid"] and c["ident"] != r["main_id"]]
        return r["main_id"], sorted(comps)

    def partner_gid(self, gid):
        """The opposite-APA flash gid in the same physical-flash group, if any."""
        g = self.fgrp.get(gid)
        if g is None or gid not in self.flashes:
            return None
        apa = self.flashes[gid]["apa"]
        for o, og in self.fgrp.items():
            if og == g and o != gid and self.flashes[o]["apa"] != apa:
                return o
        return None

    def op_index(self, gid):
        """Index into the op.json arrays for pctree flash `gid` (matched by
        APA + time; op.json carries no gid)."""
        if self.op is None or gid not in self.flashes:
            return None
        f = self.flashes[gid]
        best, bestdt = None, OP_MATCH_TOL_US
        for k in range(len(self.op["op_t"])):
            if int(self.op["apa"][k]) != f["apa"]:
                continue
            dt = abs(self.op["op_t"][k] - f["t_us"])
            if dt <= bestdt:
                best, bestdt = k, dt
        return best

    def op_pattern(self, k):
        """(meas[312], pred[312]) for op.json entry k (pred zeros if empty)."""
        meas = np.array(self.op["op_pes"][k], float)
        pred = self.op["op_pes_pred"][k]
        pred = np.array(pred, float) if pred else np.zeros_like(meas)
        return meas, pred


_EVENT_CACHE = {}


def get_event(label):
    if label not in _EVENT_CACHE:
        _EVENT_CACHE[label] = Event(*EVENT_OF[label])
    return _EVENT_CACHE[label]


# ---------------------------------------------------------------------------
# Global UI state.
# ---------------------------------------------------------------------------
state = {
    "evt": None,          # Event
    "focus": None,        # focused row index (into evt.rows), or None
    "order": [],          # visible row indices in table display order
    "inbeam_only": False, # mode: in-beam flash bundles only
    "color_mode": "role", # 'role' (main red / companions blue) or 'charge'
    "labels": {},         # row key -> [scan labels]
    "comments": {},       # row key -> str
    "event_comment": "",
    "suppress": False,    # guard programmatic widget writes
}


def row_key(r):
    return f"{r['main_id']}:{r['flash_gid']}"


# ---------------------------------------------------------------------------
# Widgets.
# ---------------------------------------------------------------------------
event_select = Select(title="Event", value=(LABELS[0] if LABELS else ""),
                      options=LABELS, width=180)
prev_evt_btn = Button(label="< prev evt", width=90)
next_evt_btn = Button(label="next evt >", width=90)
prev_bun_btn = Button(label="< prev bundle", width=110)
next_bun_btn = Button(label="next bundle >", width=110)
mode_btn = Toggle(label="Mode: ALL bundles", width=200, active=False)
color_select = Select(title="point color", value="role",
                      options=[("role", "main=red / companions=blue"),
                               ("charge", "charge (viridis)")], width=220)
save_btn = Button(label="Save labels", button_type="success", width=120)
status = Div(text="", width=1200)
metrics = Div(text="", width=430)
proj_info = Div(text="", width=1200)

label_title = Div(text="<b>scan labels</b> (click to tag the focused bundle; "
                       "multi-select; LM = light mismatch)", width=380)
label_group = CheckboxButtonGroup(labels=SCAN_LABELS, active=[], width=340)
clear_lbl_btn = Button(label="Clear labels (focused)", width=170)
comment_in = TextAreaInput(title="bundle comment (focused)", value="", rows=3,
                           width=380, max_length=2000)
evt_comment_in = TextAreaInput(title="event comment", value="", rows=3,
                               width=380, max_length=4000)

# Row tint: green once the bundle carries any scan label or comment (progress
# tracking); the focused row uses the table's native blue selection highlight.
def cell_fmt(align):
    return HTMLTemplateFormatter(template=(
        '<div style="background-color:<%= sel_bg %>;text-align:' + align
        + ';margin:-2px -5px;padding:2px 5px"><%= value %></div>'))
fmt_l, fmt_r = cell_fmt("left"), cell_fmt("right")
SEL_BG = "#cde6cd"

table_src = ColumnDataSource(data=dict())
table_cols = [
    TableColumn(field="row", title="#", width=30, formatter=fmt_l, sortable=False),
    TableColumn(field="grp", title="grp", width=40, formatter=fmt_l, sortable=False),
    TableColumn(field="t_us", title="t(us)", width=75, formatter=fmt_r, sortable=False),
    TableColumn(field="pe", title="PE(grp)", width=75, formatter=fmt_r, sortable=False),
    TableColumn(field="beam", title="beam", width=45, formatter=fmt_l, sortable=False),
    TableColumn(field="main", title="main", width=45, formatter=fmt_l, sortable=False),
    TableColumn(field="clusters", title="clusters", width=95, formatter=fmt_l, sortable=False),
    TableColumn(field="npts", title="npts", width=55, formatter=fmt_r, sortable=False),
    TableColumn(field="len", title="len(cm)", width=65, formatter=fmt_r, sortable=False),
    TableColumn(field="verdicts", title="tgm/stm/fc", width=80, formatter=fmt_l, sortable=False),
    TableColumn(field="auto", title="auto label", width=95, formatter=fmt_l, sortable=False),
    TableColumn(field="scan", title="scan", width=95, formatter=fmt_l, sortable=False),
    TableColumn(field="cmt", title="✎", width=25, formatter=fmt_l, sortable=False),
]
table = DataTable(source=table_src, columns=table_cols, width=900, height=290,
                  selectable=True, index_position=None)

# --- charge projections -----------------------------------------------------
proj_kw = dict(height=330, tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom")
f_xy = figure(title="X-Y", width=400, **proj_kw)
f_yz = figure(title="Y-Z", width=400, **proj_kw)
f_xz = figure(title="X-Z", width=400, **proj_kw)
f_xy.xaxis.axis_label, f_xy.yaxis.axis_label = "x (cm)", "y (cm)"
f_yz.xaxis.axis_label, f_yz.yaxis.axis_label = "z (cm)", "y (cm)"
f_xz.xaxis.axis_label, f_xz.yaxis.axis_label = "x (cm)", "z (cm)"

ctx_src = ColumnDataSource(data=dict(x=[], y=[], z=[]))       # gray: whole event
comp_src = ColumnDataSource(data=dict(x=[], y=[], z=[], c=[]))  # companions
main_src = ColumnDataSource(data=dict(x=[], y=[], z=[], c=[]))  # main cluster
det_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                     xs_xz=[], ys_xz=[]))
fv_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                    xs_xz=[], ys_xz=[]))
for f, sx, sy in ((f_xy, "xs_xy", "ys_xy"), (f_yz, "xs_yz", "ys_yz"),
                  (f_xz, "xs_xz", "ys_xz")):
    f.multi_line(xs=sx, ys=sy, source=det_src, line_color="#cc4444", line_width=1)
    f.multi_line(xs=sx, ys=sy, source=fv_src, line_color="#2ca02c",
                 line_width=1.5, line_dash="dashed")
for f, hx, hy in ((f_xy, "x", "y"), (f_yz, "z", "y"), (f_xz, "x", "z")):
    f.scatter(hx, hy, source=ctx_src, marker="circle", size=2,
              fill_color="#c8c8c8", line_color=None, fill_alpha=0.45)
    f.scatter(hx, hy, source=comp_src, marker="circle", size=3,
              fill_color="c", line_color=None, fill_alpha=0.8)
    f.scatter(hx, hy, source=main_src, marker="circle", size=3,
              fill_color="c", line_color=None, fill_alpha=0.9)


def box_polys(box, cathode=False):
    """Rectangle polylines for a {x:(lo,hi),y:..,z:..} box in the 3 projections;
    with cathode=True adds the x=0 plane line to X-Y and X-Z."""
    (xl, xh), (yl, yh), (zl, zh) = box["x"], box["y"], box["z"]
    xy = [([xl, xh, xh, xl, xl], [yl, yl, yh, yh, yl])]
    yz = [([zl, zh, zh, zl, zl], [yl, yl, yh, yh, yl])]
    xz = [([xl, xh, xh, xl, xl], [zl, zl, zh, zh, zl])]
    if cathode:
        # every projection needs the same polyline count (one shared CDS)
        xy.append(([0, 0], [yl, yh]))
        yz.append(([], []))
        xz.append(([0, 0], [zl, zh]))
    return xy, yz, xz


def set_boxes():
    for src, box, cath in ((det_src, DET_BOX, True), (fv_src, FV_BOX, False)):
        xy, yz, xz = box_polys(box, cathode=cath)
        src.data = dict(xs_xy=[p[0] for p in xy], ys_xy=[p[1] for p in xy],
                        xs_yz=[p[0] for p in yz], ys_yz=[p[1] for p in yz],
                        xs_xz=[p[0] for p in xz], ys_xz=[p[1] for p in xz])


set_boxes()

# --- light-pattern figures --------------------------------------------------
def radii(vals, hi):
    return (2.5 + 9.0 * np.sqrt(np.clip(vals, 0, None) / hi)).tolist()


def make_light_fig(title):
    f = figure(title=title, height=270, width=430, tools="pan,reset,save")
    f.xaxis.axis_label = "z (cm)"
    f.yaxis.axis_label = "y (cm)"
    base = ColumnDataSource(data=dict(z=[], y=[]))
    src = ColumnDataSource(data=dict(z=[], y=[], pe=[], r=[], ch=[]))
    f.scatter("z", "y", source=base, marker="circle", size=5,
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
    f.add_tools(HoverTool(renderers=[g], tooltips=[("ch", "@ch"), ("PE", "@pe{0,0.0}")]))
    return dict(fig=f, base=base, src=src, glyph=g)


LIGHT = {apa: {"meas": make_light_fig(f"TPC{apa} measured"),
               "pred": make_light_fig(f"TPC{apa} predicted")}
         for apa in (0, 1)}


def make_overlay_fig(title):
    f = figure(title=title, height=220, width=430,
               tools="pan,box_zoom,wheel_zoom,reset,save")
    f.xaxis.axis_label = "op channel"
    f.yaxis.axis_label = "PE"
    meas_src = ColumnDataSource(data=dict(x=[], pe=[]))
    pred_src = ColumnDataSource(data=dict(x=[], pe=[]))
    f.vbar(x="x", top="pe", width=0.9, source=meas_src, fill_color="#6baed6",
           line_color=None, fill_alpha=0.6, legend_label="measured")
    f.line("x", "pe", source=pred_src, line_color="#d62728", line_width=1.5,
           legend_label="predicted")
    f.scatter("x", "pe", source=pred_src, marker="circle", size=4,
              fill_color="#d62728", line_color=None, legend_label="predicted")
    f.legend.label_text_font_size = "9px"
    f.legend.padding = 2
    f.legend.location = "top_right"
    f.legend.background_fill_alpha = 0.6
    return dict(fig=f, meas=meas_src, pred=pred_src)


OVERLAY = {apa: make_overlay_fig(f"TPC{apa} meas vs pred") for apa in (0, 1)}


def set_light_ranges():
    for apa in (0, 1):
        for panel in (LIGHT[apa]["meas"], LIGHT[apa]["pred"]):
            f = panel["fig"]
            f.x_range.start, f.x_range.end = -25, 525
            f.y_range.start, f.y_range.end = -220, 220


set_light_ranges()


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def visible_rows(evt):
    idxs = list(range(len(evt.rows)))
    if state["inbeam_only"]:
        idxs = [i for i in idxs if evt.rows[i]["in_beam"] == 1]
    return idxs


def rebuild_table():
    evt = state["evt"]
    order = visible_rows(evt)
    state["order"] = order
    cols = defaultdict(list)
    for n, i in enumerate(order):
        r = evt.rows[i]
        main, comps = evt.bundle_cluster_ids(r)
        key = row_key(r)
        labs = state["labels"].get(key, [])
        cols["row"].append(str(n))
        cols["grp"].append(str(r["flash_grp"]))
        cols["t_us"].append(f"{r['t_us']:.3f}")
        cols["pe"].append(f"{r['pe_grp']:.0f}")
        cols["beam"].append("Y" if r["in_beam"] else "")
        cols["main"].append(str(main) if main is not None else "-")
        cols["clusters"].append("+".join(str(c) for c in ([main] + comps))
                                if main is not None else "-")
        cols["npts"].append(str(r["npts_bundle"]))
        cols["len"].append(f"{r['len_cm']:.1f}")
        cols["verdicts"].append(f"{r['tgm']}/{r['stm']}/{r['fc']}")
        cols["auto"].append(r["auto_label"])
        cols["scan"].append("+".join(labs))
        cols["cmt"].append("✎" if state["comments"].get(key) else "")
        cols["sel_bg"].append(SEL_BG if (labs or state["comments"].get(key))
                              else "transparent")
    table_src.data = dict(cols)
    if state["focus"] is not None and state["focus"] in order:
        table_src.selected.indices = [order.index(state["focus"])]
    else:
        table_src.selected.indices = []


def charge_colors(q):
    """Viridis colors for charge values (robust 99th-percentile scale)."""
    hi = max(1.0, float(np.percentile(q, 99))) if q.size else 1.0
    idx = np.clip(q / hi * 255, 0, 255).astype(int)
    return [Viridis256[k] for k in idx]


def render_projections():
    evt = state["evt"]
    # gray context: the whole event (downsampled), exactly the Bee clustering layer
    if evt.pts is not None:
        n = len(evt.pts["x"])
        s = max(1, int(math.ceil(n / MAX_CTX_PTS)))
        ctx_src.data = dict(x=evt.pts["x"][::s].tolist(),
                            y=evt.pts["y"][::s].tolist(),
                            z=evt.pts["z"][::s].tolist())
    else:
        ctx_src.data = dict(x=[], y=[], z=[])

    empty = dict(x=[], y=[], z=[], c=[])
    i = state["focus"]
    if i is None or evt.pts is None:
        main_src.data = dict(empty)
        comp_src.data = dict(empty)
        return
    r = evt.rows[i]
    main, comps = evt.bundle_cluster_ids(r)
    if main is None:
        main_src.data = dict(empty)
        comp_src.data = dict(empty)
        return
    cid = evt.pts["cid"]
    mm = cid == main
    cm = np.isin(cid, comps) if comps else np.zeros_like(mm)
    for src, mask, role_color in ((main_src, mm, "#d62728"),
                                  (comp_src, cm, "#1f77b4")):
        x, y, z = evt.pts["x"][mask], evt.pts["y"][mask], evt.pts["z"][mask]
        if state["color_mode"] == "charge":
            c = charge_colors(evt.pts["q"][mask])
        else:
            c = [role_color] * len(x)
        src.data = dict(x=x.tolist(), y=y.tolist(), z=z.tolist(), c=c)


def render_light():
    """Per TPC side: measured + predicted pattern of the physical flash of the
    focused bundle (own-side flash and its 80 ns cross-APA partner)."""
    evt = state["evt"]
    i = state["focus"]
    side_gid = {0: None, 1: None}
    if i is not None and evt.flashes:
        gid = evt.rows[i]["flash_gid"]
        if gid in evt.flashes:
            side_gid[evt.flashes[gid]["apa"]] = gid
            p = evt.partner_gid(gid)
            if p is not None:
                side_gid[evt.flashes[p]["apa"]] = p
    for apa in (0, 1):
        mp, pp = LIGHT[apa]["meas"], LIGHT[apa]["pred"]
        ov = OVERLAY[apa]
        if OD_APA is not None:
            am = OD_APA == apa
            chans = np.nonzero(am)[0]
            mp["base"].data = dict(z=OD_Z[am].tolist(), y=OD_Y[am].tolist())
            pp["base"].data = dict(z=OD_Z[am].tolist(), y=OD_Y[am].tolist())
        else:
            chans = None
        gid = side_gid[apa]
        k = evt.op_index(gid) if gid is not None else None
        if k is None:
            mp["src"].data = dict(z=[], y=[], pe=[], r=[], ch=[])
            pp["src"].data = dict(z=[], y=[], pe=[], r=[], ch=[])
            ov["meas"].data = dict(x=[], pe=[])
            ov["pred"].data = dict(x=[], pe=[])
            what = "no flash this side" if gid is None else "flash not in op layer"
            mp["fig"].title.text = f"TPC{apa} measured  ({what})"
            pp["fig"].title.text = f"TPC{apa} predicted"
            ov["fig"].title.text = f"TPC{apa} meas vs pred"
            continue
        meas_full, pred_full = evt.op_pattern(k)
        f = evt.flashes[gid]
        if chans is not None:
            meas, pred = meas_full[chans], pred_full[chans]
            hi_m = max(1.0, float(meas.max()))
            hi_p = max(1.0, float(pred.max()))
            mp["glyph"].glyph.fill_color["transform"].high = hi_m
            pp["glyph"].glyph.fill_color["transform"].high = hi_p
            zz, yy = OD_Z[chans].tolist(), OD_Y[chans].tolist()
            ch = chans.tolist()
            mp["src"].data = dict(z=zz, y=yy, pe=meas.tolist(),
                                  r=radii(meas, hi_m), ch=ch)
            pp["src"].data = dict(z=zz, y=yy, pe=pred.tolist(),
                                  r=radii(pred, hi_p), ch=ch)
            xidx = chans
        else:
            meas, pred = meas_full, pred_full
            xidx = np.arange(meas.size)
        mp["fig"].title.text = (f"TPC{apa} measured  gid {gid}  t={f['t_us']:.3f} us"
                                f"  totPE={meas_full.sum():.0f}")
        pp["fig"].title.text = (f"TPC{apa} predicted  totPE={pred_full.sum():.1f}"
                                + ("" if pred_full.any() else "  (no prediction)"))
        ov["meas"].data = dict(x=xidx.tolist(), pe=meas.tolist())
        ov["pred"].data = dict(x=xidx.tolist(), pe=pred.tolist())
        ov["fig"].title.text = f"TPC{apa} meas vs pred  (gid {gid}, t={f['t_us']:.3f} us)"


def render_metrics():
    evt = state["evt"]
    i = state["focus"]
    if i is None:
        metrics.text = "<i>click a bundle row to inspect</i>"
        return
    r = evt.rows[i]
    main, comps = evt.bundle_cluster_ids(r)
    key = row_key(r)
    partner = evt.partner_gid(r["flash_gid"])
    ptxt = (f"gid {partner} (t={evt.flashes[partner]['t_us']:.3f} us)"
            if partner is not None else "-")
    verdict = lambda v: {1: "YES", 0: "no", -1: "n/a"}.get(v, str(v))
    rows_ = [
        ("flash", f"gid {r['flash_gid']} / TPC {r['flash_apa']} / grp {r['flash_grp']}"),
        ("flash time", f"{r['t_us']:.3f} us" + ("  (IN BEAM)" if r["in_beam"] else "")),
        ("flash PE", f"{r['pe']:.1f}  (group {r['pe_grp']:.1f})"),
        ("cross-APA partner", ptxt),
        ("main cluster", str(main) if main is not None else "(no bundle)"),
        ("companions", ", ".join(str(c) for c in comps) or "-"),
        ("npts main / bundle", f"{r['npts_main']} / {r['npts_bundle']}"),
        ("len (main comp.)", f"{r['len_cm']:.1f} cm  ({r['n_frag']} merged frag.)"),
        ("TGM", verdict(r["tgm"])),
        ("STM", verdict(r["stm"])),
        ("FC", verdict(r["fc"])),
        ("auto label", f"<b>{r['auto_label']}</b>"),
        ("scan labels", "+".join(state["labels"].get(key, [])) or "-"),
        ("comment", state["comments"].get(key, "") or "-"),
    ]
    metrics.text = ("<table style='font-size:12px'>" + "".join(
        f"<tr><td style='color:#555;padding-right:8px'>{a}</td><td><b>{b}</b></td></tr>"
        for a, b in rows_) + "</table>")


def render_proj_info():
    evt = state["evt"]
    if evt is None:
        proj_info.text = ""
        return
    i = state["focus"]
    r0 = evt.rows[0] if evt.rows else None
    rse = f"run {r0['run']} subrun {r0['subrun']} event {r0['event']}" if r0 else ""
    foc = ""
    if i is not None:
        r = evt.rows[i]
        main, comps = evt.bundle_cluster_ids(r)
        foc = (f" &nbsp;&nbsp; <b>bundle</b> grp {r['flash_grp']} t={r['t_us']:.3f} us"
               f" main <span style='color:#d62728'><b>{main}</b></span>"
               + (f" + companions <span style='color:#1f77b4'><b>"
                  f"{','.join(str(c) for c in comps)}</b></span>" if comps else ""))
    nlab = sum(1 for r in evt.rows if state["labels"].get(row_key(r)))
    proj_info.text = (f"<b>{evt.label}</b> ({rse}) &mdash; "
                      f"{len(state['order'])} bundle(s) shown "
                      f"[{'IN-BEAM ONLY' if state['inbeam_only'] else 'all'}], "
                      f"{nlab} scanned{foc}"
                      + (f"<br><span style='color:#b00'>{'; '.join(evt.problems)}"
                         "</span>" if evt.problems else ""))


def sync_label_widgets():
    """Push the focused row's labels/comment into the widgets (guarded)."""
    state["suppress"] = True
    i = state["focus"]
    if i is None:
        label_group.active = []
        comment_in.value = ""
    else:
        key = row_key(state["evt"].rows[i])
        labs = state["labels"].get(key, [])
        label_group.active = [SCAN_LABELS.index(l) for l in labs if l in SCAN_LABELS]
        comment_in.value = state["comments"].get(key, "")
    evt_comment_in.value = state["event_comment"]
    state["suppress"] = False


def refresh():
    rebuild_table()
    render_projections()
    render_light()
    render_metrics()
    render_proj_info()
    sync_label_widgets()


# ---------------------------------------------------------------------------
# Persistence (autosave state + final labels JSON).
# ---------------------------------------------------------------------------
def labels_dir(evt):
    d = os.path.join(evt.work_root, "nusel_labels", SCAN_TAG)
    os.makedirs(d, exist_ok=True)
    return d


def state_file(evt):
    return os.path.join(labels_dir(evt), f".scan_state-{evt.label}.json")


def save_state():
    evt = state["evt"]
    if evt is None:
        return
    try:
        with open(state_file(evt), "w") as fh:
            json.dump({"labels": state["labels"], "comments": state["comments"],
                       "event_comment": state["event_comment"]}, fh)
    except OSError:
        pass


def load_state(evt):
    p = state_file(evt)
    if not os.path.isfile(p):
        return {}, {}, ""
    try:
        with open(p) as fh:
            d = json.load(fh)
        return (d.get("labels", {}), d.get("comments", {}),
                d.get("event_comment", ""))
    except (OSError, ValueError):
        return {}, {}, ""


def on_save():
    evt = state["evt"]
    if evt is None:
        return
    r0 = evt.rows[0] if evt.rows else {}
    bundles = []
    for r in evt.rows:
        key = row_key(r)
        main, comps = evt.bundle_cluster_ids(r)
        bundles.append({
            "main_id": r["main_id"], "flash_gid": r["flash_gid"],
            "flash_grp": r["flash_grp"], "flash_apa": r["flash_apa"],
            "flash_time_us": r["t_us"], "flash_pe_grp": r["pe_grp"],
            "in_beam": r["in_beam"],
            "clusters": ([main] + comps) if main is not None else [],
            "auto": {"tgm": r["tgm"], "stm": r["stm"], "fc": r["fc"],
                     "label": r["auto_label"]},
            "scan_labels": state["labels"].get(key, []),
            "comment": state["comments"].get(key, ""),
        })
    out = {"event": evt.label, "run": r0.get("run"), "subrun": r0.get("subrun"),
           "event_no": r0.get("event"), "source": os.path.basename(evt.tsv),
           "work_root": evt.work_root, "tag": SCAN_TAG,
           "event_comment": state["event_comment"], "bundles": bundles}
    dest = os.path.join(labels_dir(evt), f"nusel-labels-{evt.label}.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    nlab = sum(1 for b in bundles if b["scan_labels"] or b["comment"])
    status.text = f"Saved {len(bundles)} bundle(s) ({nlab} annotated) -> {dest}"


# ---------------------------------------------------------------------------
# Callbacks.
# ---------------------------------------------------------------------------
def load_event(label):
    evt = get_event(label)
    state["evt"] = evt
    state["labels"], state["comments"], state["event_comment"] = load_state(evt)
    order = visible_rows(evt)
    state["focus"] = order[0] if order else None
    n_in = sum(1 for r in evt.rows if r["in_beam"])
    status.text = (f"Loaded <b>{label}</b>: {len(evt.rows)} bundle row(s), "
                   f"{n_in} in-beam; charge/light from the Bee mabc-all-apa.zip "
                   "layers. Click a row, tag with the label buttons, Save when done.")
    refresh()


def on_event_change(attr, old, new):
    if new:
        load_event(new)


def step_event(d):
    if not LABELS:
        return
    i = LABELS.index(event_select.value)
    event_select.value = LABELS[(i + d) % len(LABELS)]


def step_bundle(d):
    order = state["order"]
    if not order:
        return
    cur = state["focus"]
    pos = order.index(cur) if cur in order else -d
    state["focus"] = order[(pos + d) % len(order)]
    refresh()


def on_row_select(attr, old, new):
    if not new:
        return
    order = state["order"]
    pos = new[0]
    if 0 <= pos < len(order):
        state["focus"] = order[pos]
        render_projections()
        render_light()
        render_metrics()
        render_proj_info()
        sync_label_widgets()


def on_mode(attr, old, new):
    state["inbeam_only"] = bool(new)
    mode_btn.label = "Mode: IN-BEAM only" if new else "Mode: ALL bundles"
    mode_btn.button_type = "warning" if new else "default"
    order_before = state["focus"]
    if order_before is not None and state["evt"] is not None:
        if new and state["evt"].rows[order_before]["in_beam"] != 1:
            state["focus"] = None
    if state["focus"] is None:
        vis = visible_rows(state["evt"]) if state["evt"] else []
        state["focus"] = vis[0] if vis else None
    refresh()


def on_color(attr, old, new):
    state["color_mode"] = new
    render_projections()


def on_labels(attr, old, new):
    if state["suppress"] or state["focus"] is None:
        return
    r = state["evt"].rows[state["focus"]]
    key = row_key(r)
    labs = [SCAN_LABELS[k] for k in sorted(new)]
    if labs:
        state["labels"][key] = labs
    else:
        state["labels"].pop(key, None)
    save_state()
    rebuild_table()
    render_metrics()
    render_proj_info()


def on_clear_labels():
    if state["focus"] is None:
        return
    key = row_key(state["evt"].rows[state["focus"]])
    state["labels"].pop(key, None)
    state["comments"].pop(key, None)
    save_state()
    refresh()


def on_comment(attr, old, new):
    if state["suppress"] or state["focus"] is None:
        return
    key = row_key(state["evt"].rows[state["focus"]])
    if new.strip():
        state["comments"][key] = new
    else:
        state["comments"].pop(key, None)
    save_state()
    rebuild_table()
    render_metrics()


def on_evt_comment(attr, old, new):
    if state["suppress"]:
        return
    state["event_comment"] = new
    save_state()


event_select.on_change("value", on_event_change)
prev_evt_btn.on_click(lambda: step_event(-1))
next_evt_btn.on_click(lambda: step_event(+1))
prev_bun_btn.on_click(lambda: step_bundle(-1))
next_bun_btn.on_click(lambda: step_bundle(+1))
table_src.selected.on_change("indices", on_row_select)
mode_btn.on_change("active", on_mode)
color_select.on_change("value", on_color)
label_group.on_change("active", on_labels)
clear_lbl_btn.on_click(on_clear_labels)
comment_in.on_change("value", on_comment)
evt_comment_in.on_change("value", on_evt_comment)
save_btn.on_click(on_save)


# ---------------------------------------------------------------------------
# Layout.
# ---------------------------------------------------------------------------
header = Div(text="<h2>SBND nusel (TGM/STM/FC) hand-scan</h2>", width=1200)
controls = column(
    row(event_select, prev_evt_btn, next_evt_btn, prev_bun_btn, next_bun_btn,
        mode_btn, save_btn),
    row(color_select))
layout = column(
    row(f_xy, f_yz, f_xz),
    proj_info,
    header,
    controls,
    status,
    row(table, column(label_title, label_group, clear_lbl_btn, comment_in,
                      evt_comment_in), metrics),
    row(LIGHT[0]["meas"]["fig"], LIGHT[0]["pred"]["fig"],
        LIGHT[1]["meas"]["fig"], LIGHT[1]["pred"]["fig"]),
    row(OVERLAY[0]["fig"], OVERLAY[1]["fig"]),
)

curdoc().add_root(layout)
curdoc().title = "nusel hand-scan"

if LABELS:
    load_event(LABELS[0])
else:
    status.text = ("<b>No nusel TSVs found.</b> Produce them with "
                   "run_nusel_evt.sh, then pass the work root (dir holding "
                   "nusel_evt*/ + ql_evt*/) to serve_nusel_scan.sh.")

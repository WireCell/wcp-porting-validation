#!/usr/bin/env python3
"""SBND pattern-recognition (PR) event display (Bokeh server).

A read-only viewer for what the PR chain produced, built to drive tuning of the
PR code over the 572 valfast events.  See sbnd_xin/docs/pr/26_pr-event-display.md.

INPUT, stage 1 -- one self-contained JSON per event, written by the `pr_display`
stage (PrDisplayDump) of the PR chain:

    PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh <ql_root> <out> data 388
    -> <out>/pr_evt<ID>/calib-pr-evt<ID>.json

INPUT, stage 2 -- the PARTICLE-FLOW tree, read from the Bee zip written beside
it (`mabc-pr.zip`, member `data/*/*-mc.json`) with stdlib zipfile.  That tree is
the canonical PF product -- MultiAlgBlobClustering::fill_bee_pf_tree writes it,
and it is the same one Bee shows -- so this display reads it rather than
rebuilding it and risking a second, disagreeing answer.  Its node ids are
`cluster_id*1000 + segment id`, exactly the calib JSON's segment ids; the
`shower_id` field on each segment carries the same encoding, which is how one
click on a shower node highlights all of that shower's segments.  No zip => the
PF panel is empty and everything else still works.

tracking-pr.root is still deliberately NOT read.  Its T_proj_data was empty of
everything but `cluster_id` when this display was designed, and has since been
repaired (doc pr/26 sec 5.1, toolkit 4c02b679) -- but it emits one row per
CLUSTER TAG, so a cell claimed by two clusters appears twice, and it keys on the
concatenated global channel rather than per-APA wire.  The dump's own `proj` is
already in the display's coordinates; reading ROOT would add a dependency to
re-derive what is here.

LAYOUT

  Row 1 -- the three charge projections X-Y, Y-Z, X-Z, each carrying every
           layer below, individually toggleable.
  Row 2 -- particle flow (click a particle to highlight it in all nine panels)
           next to the event's selection numbers.
  Row 3 -- six panels, two columns (TPC 0 | TPC 1) x three rows (T-U, T-V,
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

The BDT scores shown here are produced with the uBooNE-TRAINED weight XMLs (the
SBND config books those); they are availability and relative ranking only, not
calibrated SBND scores.  The panel says so, and the dump carries the same string
in `tagger.weights`.

THE COSMIC ANSWER is `cosmict_flag`: the OR of the cosmic tagger's ten tests
(NeutrinoTaggerCosmic.cxx).  The panel shows the verdict and which test fired,
plus whether each test ran at all -- on a neutrino-selected sample nearly every
test reads 0 and only that second column separates a quiet tagger from an
inactive one.  Two fields that look like cosmic answers are NOT:

  cosmic_flag     a BDT input feature, exactly !cosmict_flag_9, written only
                  inside the flag-9 block, with an in-class default of 1.  Its
                  polarity is the opposite of what the name suggests and 1 is
                  ambiguous between "never tested" and "rescued".
  cosmict_score   never computed, on either the toolkit or the prototype -- a
                  legacy slot of the uBooNE ntuple schema belonging to the
                  TMVA BDT path, which has no caller.  Not dumped, not shown.
"""

import sys
import os
import re
import glob
import json
import zipfile
import argparse
import math
import html
from collections import defaultdict

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, Select, Button, Div, HoverTool,
                          CheckboxButtonGroup, TextInput, Toggle, Spacer,
                          ColorBar, LinearColorMapper, BasicTicker,
                          DataTable, TableColumn, HTMLTemplateFormatter,
                          CDSView, AllIndices)
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
# The particle-flow highlight: the same polylines again, drawn as a fat halo
# UNDER the coloured segments so the colour still reads through it.
hl_src = {k: ColumnDataSource(data=dict(xs=[], ys=[]))
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
    # Halo first: created before the segment glyph, so it renders beneath it.
    # Amber, because it has to stay legible both over the dark associated-point
    # cloud here and over the viridis charge cells in the 2-D panels.
    f.multi_line(xs="xs", ys="ys", source=hl_src[key], line_color="#ffb000",
                 line_width=9, line_alpha=0.85)
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
        hl = ColumnDataSource(data=dict(xs=[], ys=[]))
        rd = f.rect(x="w", y="s", width=1.0, height="h", source=dead,
                    fill_color="#dddddd", line_color=None, fill_alpha=0.6)
        rc = f.rect(x="w", y="s", width=1.0, height=1.0, source=cell,
                    fill_color=dict(field="q", transform=CMAP),
                    line_color=None)
        # Same halo-under-the-line trick as the projections: created before the
        # fit polyline so the coloured track stays legible on top of it.
        f.multi_line(xs="xs", ys="ys", source=hl, line_color="#ffb000",
                     line_width=8, line_alpha=0.85)
        rf = f.multi_line(xs="xs", ys="ys", source=fit, line_color="c",
                          line_width=2.0, line_alpha=0.95)
        f.add_tools(HoverTool(renderers=[rc], tooltips=[
            ("wire", "@w"), ("slice", "@s"),
            ("charge", "@q{0,0}"), ("pred", "@qp{0,0}"), ("cluster", "@cid")]))
        RENDER["dead"].append(rd)
        RENDER["trackfit2d"].append(rf)
        panel[(apa, pl)] = dict(fig=f, cell=cell, fit=fit, dead=dead, hl=hl)

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

# ---------------------------------------------------------------------------
# Particle flow + event features
# ---------------------------------------------------------------------------
PDG_NAME = {11: "e-", -11: "e+", 13: "mu-", -13: "mu+", 22: "gamma",
            211: "pi+", -211: "pi-", 2212: "p", 2112: "n", 321: "K+",
            -321: "K-", 111: "pi0", 0: "?"}
# kine_energy_info: how that particle's energy was measured.
KINE_METHOD = {0: "dQ/dx", 1: "range", 2: "charge"}

PF_EMPTY = dict(row=[], label=[], id=[], kind=[], ke=[], nseg=[], length=[])
pf_src = ColumnDataSource(data=dict(PF_EMPTY))
# Bokeh 3.9 repaints a DataTable ONLY through its CDSView's change signal, and
# an all-rows Indices of the same length compares equal so nothing fires.  Keep
# two distinct AllIndices() and flip between them after every .data assignment
# (doc 58; nusel_display/nusel_scan_viewer.py carries the same fix).
VIEW_A, VIEW_B = AllIndices(), AllIndices()
_fmt = HTMLTemplateFormatter(template='<div style="font-family:monospace"><%= value %></div>')
pf_table = DataTable(
    source=pf_src, view=CDSView(filter=VIEW_A), width=620, height=260,
    index_position=None, selectable=True, sortable=False,
    columns=[TableColumn(field="label", title="particle", width=250, formatter=_fmt),
             TableColumn(field="kind", title="kind", width=70, formatter=_fmt),
             TableColumn(field="id", title="id", width=70, formatter=_fmt),
             TableColumn(field="ke", title="KE (MeV)", width=80, formatter=_fmt),
             TableColumn(field="nseg", title="nseg", width=55, formatter=_fmt),
             TableColumn(field="length", title="L (cm)", width=70, formatter=_fmt)])
pf_title = Div(text="<b>particle flow</b> "
                    "<span style='color:#666'>&mdash; click a row to highlight it "
                    "in all nine panels</span>", width=620)
pf_clear_btn = Button(label="clear highlight", width=140)
pf_note = Div(text="", width=620)

feat_div = Div(text="", width=760)
kine_div = Div(text="", width=760)
cos_div = Div(text="", width=760)
bdt_toggle = Toggle(label="BDT sub-scores", active=False, width=160)
bdt_div = Div(text="", width=760, visible=False)

# The cosmic tagger's ten tests, in the order they are evaluated in
# NeutrinoTaggerCosmic.cxx.  `filled` names the TaggerInfo field that says the
# test was actually evaluated (None = the test always evaluates, or its own
# convention applies -- see the notes on tests 9 and 10 below).
COSMIC_TESTS = [
    (1,  "vertex outside FV",          None,
     "main vertex outside the fiducial volume shrunk by 1.5 cm"),
    (2,  "single muon, wrong dir.",    "cosmict_2_filled",
     "muon at the vertex: <=2 muon tracks, <40 cm of shower, weak/steep "
     "direction or >40-60 deg off beam, and downward-going"),
    (3,  "long-muon chain, wrong dir.", "cosmict_3_filled",
     "same test applied to a long-muon shower chain instead of one segment"),
    (4,  "muon exits, >100 deg",       "cosmict_4_filled",
     "the muon's far end is outside the FV, >100 deg from the beam, with no "
     "connected showers"),
    (5,  "long muon exits, >100 deg",  "cosmict_5_filled",
     "same, for a long-muon chain"),
    (6,  "back-to-back secondary",     "cosmict_6_filled",
     "second muon track has a weak direction, exits the FV, and is >170 deg "
     "from the first"),
    (7,  "stopped muon + Michel",      "cosmict_7_filled",
     "stopped muon with a Michel-like or nearly back-to-back secondary, "
     "pointing downward/steep"),
    (8,  "muon + exiting back-track",  "cosmict_8_filled",
     "muon >100 cm, one track >165 deg from it leaving the FV, everything "
     "else <12 cm"),
    (9,  "vertical-track collection",  "cosmic_filled",
     "cluster-PCA: most of the event's length is vertical and reaches the top "
     "of the detector, and no neutrino-like shower at the vertex rescued it"),
    (10, "front-face beam-aligned",    None,
     "a vertex outside the FV within 15 cm of the upstream face, with a "
     "beam-aligned weak-direction track >10 cm"),
]


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


# ---------------------------------------------------------------------------
# Particle flow: read the Bee zip's mc.json and flatten it
# ---------------------------------------------------------------------------
def read_pf_tree(calib_path):
    """The jsTree node array from `mabc-pr.zip` beside the calib JSON.

    Returns (nodes, note).  `note` explains an empty result rather than leaving
    the panel silently blank.
    """
    zpath = os.path.join(os.path.dirname(os.path.abspath(calib_path)), "mabc-pr.zip")
    if not os.path.isfile(zpath):
        return [], "no <code>mabc-pr.zip</code> beside the calib JSON &mdash; no particle flow"
    try:
        with zipfile.ZipFile(zpath) as z:
            members = [n for n in z.namelist() if n.endswith("-mc.json")]
            if not members:
                return [], ("<code>%s</code> has no <code>*-mc.json</code> member: the PR "
                            "chain found no main vertex, so fill_bee_pf_tree wrote nothing"
                            % os.path.basename(zpath))
            return json.loads(z.read(sorted(members)[0])), ""
    except Exception as exc:                       # a corrupt zip must not kill the page
        return [], "could not read <code>%s</code>: %s" % (os.path.basename(zpath), exc)


def flatten_pf(nodes, depth=0, out=None):
    """Depth-first walk of the jsTree array -> flat rows carrying their depth."""
    if out is None:
        out = []
    for n in nodes:
        kids = n.get("children") or []
        out.append(dict(id=int(n.get("id", -1)), text=str(n.get("text", "")),
                        depth=depth, child_ids=[int(c.get("id", -1)) for c in kids]))
        flatten_pf(kids, depth + 1, out)
    return out


def pf_index(d, nodes):
    """The join tables every PF lookup needs, built ONCE per event.

    `pf_rows` needs a highlight count per row and `on_pf_select` needs the set
    itself; rebuilding these inside either would be (rows x segments) per event
    load, which only bites once the campaign starts stepping events.
    """
    by_shower = defaultdict(list)
    for s in d.get("segments", []):
        if s.get("shower_id", -1) >= 0:
            by_shower[s["shower_id"]].append(s["id"])
    return dict(seg_ids={s["id"] for s in d.get("segments", [])},
                by_shower=by_shower,
                flat={n["id"]: n for n in flatten_pf(nodes)})


def pf_rows(d, nodes, idx=None):
    """PF rows for the table, joined against the calib JSON's segments/showers."""
    idx = idx or pf_index(d, nodes)
    seg_ids, by_shower = idx["seg_ids"], idx["by_shower"]
    showers = {sh["id"]: sh for sh in d.get("showers", [])}

    # Every column, always -- an empty dict makes the client read 0 rows as 1
    # (doc 58 GOTCHA 1).
    cols = {k: [] for k in PF_EMPTY}
    flat = flatten_pf(nodes)
    for i, n in enumerate(flat):
        nid = n["id"]
        if nid in by_shower:
            kind = "shower"
        elif nid in seg_ids:
            kind = "track"
        else:
            # A pseudo-gamma node fill_bee_pf_tree inserts between a parent and
            # an indirectly-connected shower.  Its id comes from that function's
            # own counter and matches nothing here; it highlights via children.
            kind = "gamma"
        # nseg is what a click actually lights up -- for a gamma node that is
        # its children's segments, not zero.
        nseg = len(highlight_ids(d, nodes, nid, idx))
        sh = showers.get(nid)
        if sh is not None:
            ke, length = sh["kine_best"], sh["total_length"]
        else:
            m = re.search(r"([-+]?[0-9.]+)\s*MeV", n["text"])
            ke = float(m.group(1)) if m else float("nan")
            length = float("nan")
        # Non-breaking spaces: plain leading spaces collapse in the table's HTML.
        cols["row"].append(i)
        cols["label"].append("   " * n["depth"] +
                             ("└ " if n["depth"] else "") + n["text"])
        cols["id"].append(nid)
        cols["kind"].append(kind)
        cols["ke"].append("" if ke != ke else "%.1f" % ke)
        cols["nseg"].append(nseg)
        cols["length"].append("" if length != length else "%.1f" % length)
    return cols


def highlight_ids(d, nodes, node_id, idx=None):
    """Segment ids to light up for one PF node id (recursing through gammas)."""
    idx = idx or pf_index(d, nodes)
    seg_ids, by_shower, flat = idx["seg_ids"], idx["by_shower"], idx["flat"]

    picked, seen = set(), set()
    stack = [node_id]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        if nid in by_shower:
            picked.update(by_shower[nid])
        if nid in seg_ids:
            picked.add(nid)
        if nid not in by_shower and nid not in seg_ids:
            stack.extend(flat.get(nid, {}).get("child_ids", []))
    return picked


# ---------------------------------------------------------------------------
# Event features
# ---------------------------------------------------------------------------
def _f(v, fmt="%.2f", missing="&mdash;"):
    return missing if v is None else fmt % v


def fill_cosmic(tag):
    """The cosmic tagger, test by test.

    The point of the per-test breakdown is that `cosmict_flag` alone says an
    event was tagged but not why, and a bare 0 is ambiguous in a second way: it
    can mean the test ran and did not fire, or that its topology precondition
    was never met so it never ran at all.  Both are shown, because on a
    neutrino-selected sample almost every test reads 0 and only the `ran`
    column distinguishes a quiet tagger from an inactive one.
    """
    if not tag:
        cos_div.text = ""
        return

    v10 = tag.get("cosmict_flag_10") or []
    rows = []
    for num, name, filled_key, tip in COSMIC_TESTS:
        fired = tag.get("cosmict_flag_%d" % num)
        if num == 10:
            fired = tag.get("cosmict_flag_10_any")
            ran = "yes" if v10 else "no"
        elif num == 1:
            ran = "yes"          # test 1 has no precondition
        else:
            f = tag.get(filled_key)
            ran = "&mdash;" if f is None else ("yes" if f else "no")

        if fired:
            mark, style = "FIRED", "color:#a33;font-weight:bold"
        elif ran == "yes":
            mark, style = "no", "color:#333"
        else:
            mark, style = "no", "color:#aaa"     # never evaluated: greyed out

        # html.escape both, quote=True: a tooltip like "the muon's far end ..."
        # would otherwise close the title='' attribute on its apostrophe and
        # spill the rest of the string into the page as visible text.
        rows.append(
            "<tr title='%s' style='%s'>"
            "<td style='padding:1px 10px 1px 0'>%d</td>"
            "<td style='padding:1px 14px 1px 0'>%s</td>"
            "<td style='padding:1px 14px 1px 0'>%s</td>"
            "<td style='padding:1px 10px 1px 0'>%s</td></tr>"
            % (html.escape(tip, quote=True), style, num,
               html.escape(name, quote=True), mark, ran))

    verdict = tag.get("cosmict_flag")
    cos_div.text = (
        "<b>cosmic tagger</b> <span style='color:#666'>&mdash; "
        "cosmict_flag = <b>%s</b>, the OR of these ten. "
        "Greyed rows never ran (no matching topology at the vertex).</span>"
        "<table style='font-family:monospace;font-size:88%%;border-collapse:collapse'>"
        "<tr><th align=left style='padding:1px 10px 1px 0'>#</th>"
        "<th align=left style='padding:1px 14px 1px 0'>test</th>"
        "<th align=left style='padding:1px 14px 1px 0'>fired</th>"
        "<th align=left style='padding:1px 10px 1px 0'>ran</th></tr>%s</table>"
        % ("&mdash;" if verdict is None else "%.0f" % verdict, "".join(rows)))


def fill_features(d):
    """The selection numbers, next to the picture."""
    tag = d.get("tagger") or {}
    kine = d.get("kine") or {}
    mv = d.get("main_vertex")
    mvv = next((v for v in d.get("vertices", []) if v.get("is_main")), None)

    def chip(name, val, fmt="%.2f", tip=""):
        # escape the tooltip: an apostrophe in it closes title='' (see fill_cosmic)
        return ("<span title='%s' style='display:inline-block;padding:2px 8px;margin:2px;"
                "border:1px solid #ccc;border-radius:4px'>%s <b>%s</b></span>"
                % (html.escape(tip, quote=True), name, _f(val, fmt)))

    # The cosmic answer is cosmict_flag -- the OR of the tagger's ten tests.
    # There is no "cosmic score": the xgboost numu path folds every cosmic
    # feature into numu_score instead, and the legacy cosmict_score is never
    # computed (see the cosmic-tagger table below and doc pr/26 sec 7.5).
    cflag = tag.get("cosmict_flag")
    cos_chip = ("<span title='%s' style='display:inline-block;padding:2px 8px;"
                "margin:2px;border:1px solid %s;border-radius:4px;background:%s'>"
                "cosmic <b>%s</b></span>"
                % (html.escape("cosmict_flag -- the cosmic tagger's verdict, the "
                               "OR of its ten tests; see the table below for "
                               "which one fired", quote=True),
                   "#a33" if cflag else "#ccc",
                   "#fdd" if cflag else "transparent",
                   "&mdash;" if cflag is None else
                   ("TAGGED" if cflag else "not tagged")))

    scores = "".join([
        chip("nue_score", tag.get("nue_score"), "%.2f",
             "nueCC BDT log-odds; -15 is the background-like default, "
             "+4.30 the saturated signal-like one"),
        chip("numu_score", tag.get("numu_score"), "%.2f",
             "numuCC BDT log-odds; on the xgboost path this is also where "
             "every cosmic feature ends up"),
        cos_chip,
        chip("isFC", tag.get("match_isFC"), "%.0f", "fully contained"),
    ])
    ecal = "".join([
        chip("reco Enu", kine.get("kine_reco_Enu"), "%.0f MeV"),
        chip("add. energy", kine.get("kine_reco_add_energy"), "%.0f MeV",
             "rest masses + binding energies added to the KE sum"),
    ])
    if kine.get("kine_pio_flag"):
        ecal += chip("pi0 mass", kine.get("kine_pio_mass"), "%.0f MeV",
                     "kine_pio_flag %s" % kine.get("kine_pio_flag"))

    geom = "".join([
        chip("segments", len(d.get("segments", [])), "%d"),
        chip("showers", len(d.get("showers", [])), "%d"),
        chip("vertices", len(d.get("vertices", [])), "%d"),
    ])
    if mv:
        geom += ("<span style='display:inline-block;padding:2px 8px;margin:2px;"
                 "border:1px solid #ccc;border-radius:4px'>nu vertex "
                 "<b>(%.1f, %.1f, %.1f)</b> cluster <b>%s</b></span>"
                 % (mv["x"], mv["y"], mv["z"], mv.get("cluster_id")))
    if mvv is not None and "fit_distance" in mvv:
        geom += chip("fit moved", mvv["fit_distance"], "%.2f cm",
                     "distance from the seed point to the fitted vertex; "
                     "0 means the 3-D vertex fit did not run or was reverted")

    caveat = ""
    if tag:
        caveat = ("<div style='color:#a33;font-size:90%%;margin-top:2px'>BDT weights: %s</div>"
                  % tag.get("weights", "unknown"))
    if not tag and not kine:
        caveat = ("<div style='color:#a33'>no <code>tagger</code>/<code>kine</code> block in "
                  "this JSON &mdash; produced before doc pr/26 stage 2; re-run the "
                  "<code>pr_display</code> stage to get them</div>")

    feat_div.text = ("<b>selection</b><br>%s%s<br><b>energy</b><br>%s<br><b>topology</b><br>%s"
                     % (scores, caveat, ecal, geom))

    # --- per-particle energy breakdown ---
    ke = kine.get("kine_energy_particle") or []
    pdg = kine.get("kine_particle_type") or []
    inf = kine.get("kine_energy_info") or []
    inc = kine.get("kine_energy_included") or []
    if not ke:
        kine_div.text = ""
    else:
        td = "<td style='padding:0 14px 0 0'"
        rows = []
        for i in range(len(ke)):
            rows.append(
                "<tr>%s>%s</td>%s align=right>%.1f</td>%s>%s</td>%s align=center>%s</td></tr>"
                % (td, PDG_NAME.get(pdg[i] if i < len(pdg) else 0, str(pdg[i])),
                   td, ke[i],
                   td, KINE_METHOD.get(inf[i] if i < len(inf) else -1, "?"),
                   td, "&#10003;" if (i < len(inc) and inc[i] == 1) else ""))
        kine_div.text = (
            "<b>energy per particle</b> "
            "<span style='color:#666'>&mdash; &#10003; = counted in reco Enu "
            "(kine_energy_included == 1)</span>"
            "<table style='font-family:monospace;font-size:92%%;border-collapse:collapse'>"
            "<tr><th align=left style='padding:0 14px 0 0'>pdg</th>"
            "<th align=right style='padding:0 14px 0 0'>KE (MeV)</th>"
            "<th align=left style='padding:0 14px 0 0'>from</th>"
            "<th style='padding:0 14px 0 0'>in Enu</th></tr>%s</table>" % "".join(rows))

    # --- the cosmic tagger, test by test ---
    fill_cosmic(tag)

    # --- the sub-BDT decomposition, behind the toggle ---
    subs = [(k, v) for k, v in sorted(tag.items())
            if k.endswith("_score") and k not in ("nue_score", "numu_score")]
    if not subs:
        bdt_div.text = "<i>no sub-scores in this JSON</i>"
    else:
        per_row = 4
        rows = []
        for i in range(0, len(subs), per_row):
            cells = "".join(
                "<td style='padding:0 6px 0 0'>%s</td>"
                "<td align=right style='padding:0 18px 0 0'>%.2f</td>" % (k, v)
                for k, v in subs[i:i + per_row])
            rows.append("<tr>%s</tr>" % cells)
        bdt_div.text = ("<table style='font-family:monospace;font-size:88%%;"
                        "border-collapse:collapse'>%s</table>" % "".join(rows))


# ---------------------------------------------------------------------------
# Highlight
# ---------------------------------------------------------------------------
def set_highlight(seg_ids):
    """Draw the halo layer over the chosen segments, in all nine panels."""
    d = state.get("data") or {}
    segs = [s for s in d.get("segments", []) if s["id"] in seg_ids]

    cols = {k: dict(xs=[], ys=[]) for k in ("xy", "yz", "xz")}
    for s in segs:
        px = [p["x"] for p in s["points"]]
        py = [p["y"] for p in s["points"]]
        pz = [p["z"] for p in s["points"]]
        for key, a, b in (("xy", px, py), ("yz", pz, py), ("xz", px, pz)):
            cols[key]["xs"].append(a)
            cols[key]["ys"].append(b)
    for key in cols:
        hl_src[key].data = cols[key]

    # 2-D: split by the (apa, face) each point was fitted in, exactly as the
    # `fit` layer does -- drawing a point with no recorded apa on APA 0 is the
    # overlay bug doc pr/3 fixed.
    nps = {(r["apa"], r["face"]): r["nticks_per_slice"]
           for r in d.get("meta", {}).get("nticks_per_slice", [])}
    hlcols = {k: dict(xs=[], ys=[]) for k in panel}
    for s in segs:
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
                if key not in hlcols or len(pts) < 2:
                    continue
                hlcols[key]["xs"].append([q[0] for q in pts])
                hlcols[key]["ys"].append([q[1] for q in pts])
    for key in panel:
        panel[key]["hl"].data = hlcols[key]


def on_pf_select(attr, old, new):
    if not new:
        set_highlight(set())
        pf_note.text = ""
        return
    i = new[0]
    ids = pf_src.data["id"]
    if i >= len(ids):
        return
    nid = ids[i]
    d = state.get("data") or {}
    picked = highlight_ids(d, state.get("pf_nodes") or [], nid,
                           state.get("pf_index"))
    set_highlight(picked)
    pf_note.text = ("selected <b>%s</b> (id %d) &rarr; %d segment(s) highlighted"
                    % (pf_src.data["kind"][i], nid, len(picked)))


def on_pf_clear():
    pf_src.selected.indices = []
    set_highlight(set())
    pf_note.text = ""


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

    # --- particle flow (from the Bee zip) + event features -------------------
    nodes, note = read_pf_tree(path)
    state["pf_nodes"] = nodes
    state["pf_index"] = pf_index(d, nodes)
    pf_src.selected.indices = []
    pf_src.data = pf_rows(d, nodes, state["pf_index"])
    # The view-filter flip is the ONLY thing that repaints the grid; assigning
    # .data alone leaves the previous event's rows on screen (doc 58).
    pf_table.view.filter = VIEW_B if pf_table.view.filter is VIEW_A else VIEW_A
    pf_note.text = ("<span style='color:#a33'>%s</span>" % note) if note else ""
    set_highlight(set())
    fill_features(d)

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


def on_bdt(attr, old, new):
    bdt_div.visible = bool(new)


event_select.on_change("value", on_event)
prev_btn.on_click(step(-1))
next_btn.on_click(step(+1))
zoom_btn.on_change("active", on_zoom)
for w in (cx_in, cy_in, cz_in, half_in):
    w.on_change("value", on_centre)
vtx_btn.on_click(on_vertex)
pf_src.selected.on_change("indices", on_pf_select)
pf_clear_btn.on_click(on_pf_clear)
bdt_toggle.on_change("active", on_bdt)


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
    row(column(pf_title, pf_table, row(pf_clear_btn), pf_note),
        Spacer(width=20),
        column(feat_div, kine_div, cos_div, bdt_toggle, bdt_div)),
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

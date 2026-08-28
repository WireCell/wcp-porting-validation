"""doc pr/114 -- EM shower clustering and pi0 hand-scan display.

Two modes over one event view:

  EM    for a selected EM shower, mark every other segment IN or OUT of it, with
        the shower's own axis drawn and an acceptance plot that puts each
        candidate where the clustering's pass-1 gate would see it.
  PI0   build a pi0 from scratch: assign two gamma slots, set each gamma's start
        point, choose or back-project the decay vertex, and watch the mass.

Fork by duplication of pr_display/pr_display_viewer.py (CLAUDE.md sec 2 Code /
M10): that file is live for the neutrino-vertex scan and stays byte-untouched.
The projection block, the detector box, the reentrancy guards, the Bokeh-3.9
DataTable repaint workaround and the label-writer shape are copied and then
changed freely here.

Read-only with respect to the physics: the calib dumps and the probe sidecars
are never written.  Exactly one code path writes anything, `on_save`, and it
only ever touches ../em_labels/<tag>/.

Serve:  ./em_display/serve_em_display.sh 5021 --scan-tag <tag>
"""
import argparse
import copy
import datetime
import glob
import html
import json
import math
import os
import sys

import numpy

from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import (Button, CheckboxButtonGroup, CheckboxGroup,
                          ColumnDataSource, Div,
                          HoverTool, Select, TextInput, Toggle, RadioButtonGroup,
                          DataTable, TableColumn, CDSView, AllIndices, Range1d,
                          TapTool, Span, NumberFormatter, CustomJS, Tabs,
                          TabPanel, BoxSelectTool, WheelZoomTool, ResetTool,
                          SaveTool, MultiChoice, HTMLTemplateFormatter)
from bokeh.events import Tap, Pan, PanStart, PanEnd
from bokeh.palettes import Category20_20, Viridis256
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import em_geom as G  # noqa: E402
import em3d as D3  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)

# SBND active volume (cm) -- the same numbers pr_display and nusel_display use.
DET_BOX = dict(x=(-201.05, 201.05), y=(-199.312, 199.312), z=(0.85, 500.15))

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("specs", nargs="*", help="calib-pr-evt*.json paths or globs")
ap.add_argument("--manifest", default=os.path.join(HERE, "em114-manifest.tsv"),
                help="prep_em_scan.py output: event list, Bee links, stats. "
                     "When no explicit specs are given the manifest IS the "
                     "event list, which is what keeps the scan reproducible.")
ap.add_argument("--prepdir", default=os.path.join(HERE, "emprep"),
                help="probe sidecars (emprep-evt<ID>.json) from stage 2")
ap.add_argument("--scan-tag", default=None,
                help="label set: ../em_labels/<tag>/labels-evt<ID>.json.  Omit "
                     "and the viewer uses 'emscan1' but REFUSES to write into "
                     "it if it already holds labels (CLAUDE.md M13).  Pass the "
                     "tag explicitly to continue an existing scan.")
args = ap.parse_args(sys.argv[1:])

SCAN_TAG = args.scan_tag or "emscan1"
SCAN_TAG_EXPLICIT = args.scan_tag is not None

MANIFEST = {}          # event-id str -> manifest row dict
if os.path.exists(args.manifest):
    with open(args.manifest) as fh:
        _cols = fh.readline().rstrip("\n").split("\t")
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) == len(_cols):
                r = dict(zip(_cols, f))
                MANIFEST[r["event"]] = r


def evt_label(path):
    base = os.path.basename(path)
    if base.startswith("calib-pr-evt") and base.endswith(".json"):
        return "evt" + base[len("calib-pr-evt"):-len(".json")]
    return os.path.basename(os.path.dirname(path)) or base


PATHS = []
if args.specs:
    files = []
    for spec in args.specs:
        files += sorted(glob.glob(spec)) if any(c in spec for c in "*?[") else [spec]
    seen = set()
    for p in files:
        rp = os.path.realpath(p)
        if rp not in seen and os.path.isfile(rp):
            seen.add(rp)
            PATHS.append(p)
else:
    for e, r in sorted(MANIFEST.items(), key=lambda kv: (kv[1]["sample"], int(kv[0]))):
        p = os.path.join(SX, r["dump"])
        if os.path.isfile(p):
            PATHS.append(p)

EVENTS = {}
for p in PATHS:
    EVENTS.setdefault(evt_label(p), p)
LABELS = list(EVENTS)
if not LABELS:
    LABELS = ["(no events)"]
    EVENTS = {"(no events)": None}

# EM verdicts.  "vertex-bad" is not a nicety: the owner said up front that a
# wrong neutrino vertex can make the in/out question unanswerable, and without
# an explicit escape those events get silently mislabelled as merely wrong.
# APPEND ONLY.  A saved label stores the verdict STRING and is read back with
# EM_VERDICTS.index(), so appending is safe and re-ordering would silently
# re-label every record written before the change.
# "is an EM shower (reco PID wrong)" is the inverse of "not an EM shower" and it
# was missing: evt166870 had a muon-PID'd object the scanner wanted promoted to a
# gamma, and the only place to say so was the free-text note.
EM_VERDICTS = ["correct", "over-clustered", "under-clustered", "both",
               "vertex-bad (undecidable)", "not an EM shower",
               "is an EM shower (reco PID wrong)"]
# RETIRED as a control in round 5d, kept to READ labels written before it.
# The owner's workflow is "start from the code's reconstruction, then correct
# it", and the record already holds both sides independently: `pio.reco_groups`
# and `pio.reco_kine` are the code's answer, `pio.gammas`/`vertex`/`vertex_how`
# are the scanner's, and a mass is computed from each.  The verdict restated
# what the difference between them already says.  It was also unanchored -- the
# panel showed three pairings and the verdict named none of them.
# KNOWN LOSS, stated rather than papered over: "there is no pi0 in this event"
# cannot be said as a correction, because empty gamma slots are also what
# "not scanned" looks like.  No replacement is invented here.
PIO_VERDICTS_LEGACY = ["pi0 correct", "wrong pairing", "wrong start point",
                       "wrong vertex", "shower mis-grouped", "not a pi0"]
CONF = ["certain", "likely", "unclear"]

state = dict(label=None, data=None, prep=None,
             sel_shower=None,          # node id of the shower under scan
             marks={},                 # shower node -> {seg id -> "in"/"out"/"?"}
             excl=set(),               # shower nodes dimmed out of the way
             shorder={},               # shower node -> position, for the palette
             legacy_marks=None,        # (shower, n) when a round-4 file was read
             pio_verdict_legacy=None,  # a pi0 verdict from a pre-5d label
             acc_hidden=0,             # dots outside the zoomed acceptance range
             gamma={1: None, 2: None},  # slot -> node id
             gstart={1: None, 2: None},  # slot -> (x,y,z) override or None
             # Round 11.  Stored pi0 pairings, in the order they were added.
             # An event with two pi0 needs two masses in the record, and an
             # event where the pairing is uncertain needs the alternatives side
             # by side -- neither fits in one pair of slots.
             pio_cands=[],
             # Round 8.  The scanner's own start point and direction for an EM
             # shower, keyed BY SHOWER NODE because mark_metrics runs for every
             # marked shower, not just the selected one -- a flat pair would be
             # written from the selected shower and read back against another.
             em_start={},              # node -> (x,y,z) start override
             em_dir={},                # node -> (x,y,z) a point the axis goes THROUGH
             em_startvid={},           # node -> reconstructed vertex id it snapped to
             _axis_cache={},           # (node, start) -> recomputed dir15
             vtx_mode="main", vtx_manual=None,
             dirty=False, saved=None,
             cam=(math.radians(D3.PRESETS["iso"][0]),
                  math.radians(D3.PRESETS["iso"][1])),
             cam_c=(0.0, 0.0, 0.0), cam_R=100.0, cloud=None,
             _suspend=False, _guard=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Round 4 spreads the app across two columns instead of stacking everything in
# one 880-wide strip down the left of a wide screen.  CW is the 3-D control
# column, RW the right-hand column that carries the tables and the panels.
CW, RW = 310, 880

# 2 over 1 rather than 3 across (round 4).  Three 420-wide panels made the
# Tabs container 1260 wide -- wider than the 3-D tab it shares a Tabs with --
# and that width propagated into the page, shoving the right-hand column
# 200 px further right than the 3-D view needs and adding a horizontal
# scrollbar.  Stacked, the tab is 1040 wide and each panel is BIGGER.
proj_kw = dict(height=400, width=520, tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom")
# Range1d, never the figure() default DataRange1d: DataRange1d auto-refits to
# renderer data on every CDS push, which silently undoes an active zoom every
# time a table click updates the highlight source.
# `name=` so selftest_em3d_browser.py can read each panel's own width off the
# model instead of hardcoding 420 -- the pixel sizes move (round 4 added a size
# selector) and a canvas filter keyed to a stale constant silently matches
# nothing and passes vacuously.
f_xy = figure(name="f_xy", title="X-Y", x_range=Range1d(*DET_BOX["x"]),
              y_range=Range1d(*DET_BOX["y"]), **proj_kw)
f_yz = figure(name="f_yz", title="Y-Z", x_range=Range1d(*DET_BOX["z"]),
              y_range=Range1d(*DET_BOX["y"]), **proj_kw)
f_xz = figure(name="f_xz", title="X-Z", x_range=Range1d(*DET_BOX["x"]),
              y_range=Range1d(*DET_BOX["z"]), **proj_kw)
f_xy.xaxis.axis_label, f_xy.yaxis.axis_label = "x (cm)", "y (cm)"
f_yz.xaxis.axis_label, f_yz.yaxis.axis_label = "z (cm)", "y (cm)"
f_xz.xaxis.axis_label, f_xz.yaxis.axis_label = "x (cm)", "z (cm)"
PROJ = ((f_xy, "x", "y"), (f_yz, "z", "y"), (f_xz, "x", "z"))

# One CDS per layer, shared by all three projections: a point carries x/y/z and
# each figure names the two columns it needs.
EMPTY3 = dict(x=[], y=[], z=[], c=[], tag=[])
vtx_src = ColumnDataSource(data=dict(EMPTY3))
mainvtx_src = ColumnDataSource(data=dict(EMPTY3))
shwpt_src = ColumnDataSource(data=dict(EMPTY3))     # shower-flagged assoc points
gstart_src = ColumnDataSource(data=dict(EMPTY3))    # the two gamma start points
piovtx_src = ColumnDataSource(data=dict(EMPTY3))    # the pi0 vertex in use
# Round 8: the EM shower's start, and the point its axis is aimed through.
# emstart_src carries TWO points when an override is set -- the one in use and
# the reconstruction's own -- because "did my click take?" is otherwise
# unanswerable from the screen.
emstart_src = ColumnDataSource(data=dict(EMPTY3))
emdir_src = ColumnDataSource(data=dict(EMPTY3))
det_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                     xs_xz=[], ys_xz=[]))
# Polylines are per-projection: the xs/ys pairs genuinely differ.
# `a` is a per-segment line alpha rather than a fixed glyph alpha, so the "dim
# what is not in this shower" control can fade the rest without a second
# renderer and without re-pushing the geometry.
seg_src = {k: ColumnDataSource(data=dict(xs=[], ys=[], c=[], a=[], sid=[], pid=[],
                                         cid=[], owner=[], mark=[]))
           for k in ("xy", "yz", "xz")}


def _polymap():
    return {k: ColumnDataSource(data=dict(xs=[], ys=[])) for k in ("xy", "yz", "xz")}


mem_src = _polymap()
in_src = _polymap()
out_src = _polymap()
# What the next "mark" button press will hit.  Without this the 3-D pick surface
# is invisible (by design -- see r_pick) and a box-select gives no feedback at
# all until something is already marked, which is exactly backwards.
sel_src = _polymap()
# pi0 mode: the two gammas' own members, in their slot colours, so "which
# segments did I put in gamma 1" is answerable at a glance.
g1mem_src = _polymap()
g2mem_src = _polymap()
# Axis arrows: multi_line shaft + a rotated triangle head, NOT Bokeh's Arrow
# annotation -- Arrow is one model per arrow per panel and we draw several.
arrow_src = {k: ColumnDataSource(data=dict(xs=[], ys=[], c=[]))
             for k in ("xy", "yz", "xz")}
head_src = {k: ColumnDataSource(data=dict(x=[], y=[], angle=[], c=[]))
            for k in ("xy", "yz", "xz")}

# --- the 3-D siblings of every source above ---------------------------------
# Declared here, above the projection loop, because the halo helper below builds
# the 2-D and the 3-D stack from one description and therefore needs both.
#
# Sources come in three shapes, matching em3d.JS_REDRAW's three loops:
#   POINT  x, y, z  ->  u, v, al, sz
#   LINE   xs3, ys3, zs3  ->  xs, ys
#   HEAD   x, y, z + x0, y0, z0  ->  u, v, angle
# Python fills the projected columns too, so the first paint of an event is
# correct before any drag has happened; the JS then owns every later frame.
EMPTY3D = dict(x=[], y=[], z=[], c=[], tag=[], u=[], v=[], al=[], sz=[])
EMPTY3L = dict(xs3=[], ys3=[], zs3=[], xs=[], ys=[])

# The `name=` on these is not decoration: selftest_em3d_browser.py drives a real
# headless chromium and reaches them with `get_model_by_name`, which is how the
# CustomJS gets tested at all in a tree with no JS engine.
cam_src = ColumnDataSource(name="cam_src", data=dict(
    az=[state["cam"][0]], el=[state["cam"][1]], cx=[0.0], cy=[0.0], cz=[0.0],
    R=[100.0], az0=[0.0], el0=[0.0], xs0=[0.0], xe0=[0.0], ys0=[0.0], ye0=[0.0]))

cloud_src = ColumnDataSource(name="cloud_src",
                             data=dict(x=[], y=[], z=[], q=[], cid20=[],
                                       u=[], v=[], al=[], sz=[]))
shwpt3_src = ColumnDataSource(data=dict(EMPTY3D))
vtx3_src = ColumnDataSource(name="vtx3_src", data=dict(EMPTY3D))
mainvtx3_src = ColumnDataSource(name="mainvtx3_src", data=dict(EMPTY3D))
gstart3_src = ColumnDataSource(name="gstart3_src", data=dict(EMPTY3D))
piovtx3_src = ColumnDataSource(name="piovtx3_src", data=dict(EMPTY3D))
emstart3_src = ColumnDataSource(name="emstart3_src", data=dict(EMPTY3D))
emdir3_src = ColumnDataSource(name="emdir3_src", data=dict(EMPTY3D))
# Every fitted point of every segment, carrying its segment id.  This is the
# pick surface: Bokeh's own TapTool and BoxSelectTool hit-test it in screen space
# on the PROJECTED columns, so 3-D selection needs no JS at all -- and because a
# hit resolves to a segment id, a box in a rotated view marks whole segments, not
# a prism of loose points.
#
# It holds fitted points and NOTHING ELSE.  Vertices are tappable too (round 4)
# but they live in their own sources with their own handler: a vertex row in here
# would carry no segment id, and the first box-select that enclosed one would put
# a nonsense key straight into the saved label file.
pick_src = ColumnDataSource(name="pick_src",
                            data=dict(x=[], y=[], z=[], sid=[], u=[], v=[],
                                      al=[], sz=[]))
det3_src = ColumnDataSource(data=dict(EMPTY3L))
seg3_src = ColumnDataSource(name="seg3_src",
                            data=dict(xs3=[], ys3=[], zs3=[], xs=[], ys=[], c=[],
                                      a=[], sid=[], pid=[], cid=[], owner=[],
                                      mark=[]))
mem3_src = ColumnDataSource(data=dict(EMPTY3L))
in3_src = ColumnDataSource(name="in3_src", data=dict(EMPTY3L))
out3_src = ColumnDataSource(name="out3_src", data=dict(EMPTY3L))
sel3_src = ColumnDataSource(name="sel3_src", data=dict(EMPTY3L))
g1mem3_src = ColumnDataSource(data=dict(EMPTY3L))
g2mem3_src = ColumnDataSource(data=dict(EMPTY3L))
arrow3_src = ColumnDataSource(data=dict(xs3=[], ys3=[], zs3=[], xs=[], ys=[], c=[]))
head3_src = ColumnDataSource(data=dict(x=[], y=[], z=[], x0=[], y0=[], z0=[],
                                       u=[], v=[], angle=[], c=[]))
# The registry of every LINE-shaped 3-D source.  em3d.JS_REDRAW reprojects
# exactly what it is handed, so a source added above and forgotten here would
# keep drawing at the camera it was last filled from -- visible only as one
# halo sliding off its own segment mid-drag, which is a horrible thing to debug
# and a trivial thing to prevent.
_LINE3 = (det3_src, seg3_src, sel3_src, in3_src, out3_src, g1mem3_src,
          g2mem3_src, mem3_src, arrow3_src)

RENDER = {}


def _add(key, r):
    RENDER.setdefault(key, []).append(r)


# The halo stack, in draw order, and the order IS the message (round 4).
#
#   selection (cyan, 17)   what the next mark button will hit
#   your mark (13)         green IN / red OUT -- what YOU said
#   gamma members (11)     pi0 slot colours
#   reco members (9)       what the CLUSTERING said
#   the segment itself (2)
#   your mark again (4, dashed, ON TOP)
#
# Before round 4 the reco halo was drawn first and the mark halo over it, so
# marking a member ERASED the evidence that it was a member -- the one thing the
# owner asked to be able to see ("differentiate what initially have vs. what I
# clicked").  Widest underneath means the bands stay concentric and every state
# is readable at once: yellow ring inside a green ring = a member you confirmed,
# yellow inside red = a member you are removing, green with no yellow = a
# non-member you are adding.  The dashed repeat on top is the second, redundant
# channel for the same distinction, because a thin yellow band inside a thick
# green one is easy to miss on a laptop panel.
HALO = dict(line_cap="round")


def _halos(f, key2d, add):
    """The six halo renderers of one panel.  `key2d` is the projection key, or
    None for the 3-D panel where every source is the single 3-D sibling."""
    def src(m, s3):
        return m[key2d] if key2d else s3
    add("select", f.multi_line(xs="xs", ys="ys", source=src(sel_src, sel3_src),
                               line_color="#00b8d4", line_width=17, alpha=0.38,
                               **HALO))
    add("mark", f.multi_line(xs="xs", ys="ys", source=src(in_src, in3_src),
                             line_color="#2ca02c", line_width=13, alpha=0.42,
                             **HALO))
    add("mark", f.multi_line(xs="xs", ys="ys", source=src(out_src, out3_src),
                             line_color="#d62728", line_width=13, alpha=0.38,
                             **HALO))
    add("member", f.multi_line(xs="xs", ys="ys", source=src(g1mem_src, g1mem3_src),
                               line_color="#1f77b4", line_width=11, alpha=0.40,
                               **HALO))
    add("member", f.multi_line(xs="xs", ys="ys", source=src(g2mem_src, g2mem3_src),
                               line_color="#d62728", line_width=11, alpha=0.40,
                               **HALO))
    add("member", f.multi_line(xs="xs", ys="ys", source=src(mem_src, mem3_src),
                               line_color="#ffd27f", line_width=9, alpha=0.80,
                               **HALO))


def _mark_dashes(f, key2d, add):
    """Repeat of the marks ON TOP of the segment, dashed.  A dashed overlay reads
    as an annotation; a solid halo reads as part of the picture.  That is the
    whole point -- these two channels say 'you' and the yellow band says 'the
    reconstruction'."""
    for m, s3, col in ((in_src, in3_src, "#1a7d1a"), (out_src, out3_src, "#b01c1c")):
        add("mark", f.multi_line(xs="xs", ys="ys", source=(m[key2d] if key2d else s3),
                                 line_color=col, line_width=4, alpha=0.95,
                                 line_dash="dashed"))


for f, hx, hy in PROJ:
    # f_yz plots z on the horizontal axis, so its key is "yz" while its (hx, hy)
    # spells "zy" -- the one place the panel name and the column pair disagree.
    k = {"xy": "xy", "zy": "yz", "xz": "xz"}[hx + hy]
    _add("det", f.multi_line(xs="xs_" + k, ys="ys_" + k, source=det_src,
                             line_color="#cc4444", line_width=1, alpha=0.55))
    _add("shwpt", f.scatter(hx, hy, source=shwpt_src,
                            size=2, color="#8fbf8f", alpha=0.45))
    _halos(f, k, _add)
    r_seg = f.multi_line(xs="xs", ys="ys", source=seg_src[k], line_color="c",
                         line_width=2, line_alpha="a")
    _add("segments", r_seg)
    _mark_dashes(f, k, _add)
    f.add_tools(HoverTool(renderers=[r_seg], tooltips=[
        ("segment", "@sid"), ("cluster", "@cid"), ("pdg", "@pid"),
        ("in shower", "@owner"), ("mark", "@mark")]))
    _add("arrows", f.multi_line(xs="xs", ys="ys", source=arrow_src[k],
                                line_color="c", line_width=3, alpha=0.9))
    _add("arrows", f.scatter("x", "y", source=head_src[k], marker="triangle",
                             size=11, angle="angle", fill_color="c",
                             line_color="c", alpha=0.9))
    r_vtx = f.scatter(hx, hy, source=vtx_src, size=6, color="#7f7f7f", alpha=0.75)
    _add("vertices", r_vtx)
    r_mv = f.scatter(hx, hy, source=mainvtx_src, marker="star", size=20,
                     fill_color="#1f77b4", line_color="#08306b", alpha=0.95)
    _add("vertices", r_mv)
    # Bokeh fades every UNselected glyph on a tap; without this, tapping one
    # vertex visually erases the other hundred in the panel being scanned.
    for _r in (r_vtx, r_mv):
        _r.nonselection_glyph = _r.glyph
    _add("gamma", f.scatter(hx, hy, source=gstart_src, marker="diamond", size=18,
                            fill_color="c", line_color="#222222", alpha=0.95))
    _add("gamma", f.scatter(hx, hy, source=piovtx_src, marker="star", size=24,
                            fill_color="#e377c2", line_color="#7b3294", alpha=0.95))
    _add("emstart", f.scatter(hx, hy, source=emstart_src, marker="x", size=20,
                              line_color="c", line_width=4, alpha=0.95))
    _add("emstart", f.scatter(hx, hy, source=emdir_src, marker="triangle",
                              size=16, fill_color="c", line_color="#222222",
                              alpha=0.95))

# ---------------------------------------------------------------------------
# The 3-D panel (doc pr/114 rounds 3-4)
# ---------------------------------------------------------------------------
# Rotatable like Bee, but inside Bokeh so every label control keeps working off
# it.  The mechanics, the frame constraint and the honest limits are all in
# em3d.py's docstring; this block is only the wiring.  Its sources are declared
# with the 2-D ones above, because the halo stack is built from one description
# for both panels.
_wheel3 = WheelZoomTool(dimensions="both")
_tap3 = TapTool()
_box3 = BoxSelectTool()
f3d = figure(name="f3d", title="3-D  —  drag rotates, shift+drag pans, wheel zooms",
             width=760, height=760,
             x_range=Range1d(-100, 100), y_range=Range1d(-100, 100),
             tools=[_wheel3, _tap3, _box3, ResetTool(), SaveTool()],
             output_backend="webgl")
# Square figure + equal spans is what makes the view isotropic; do not let these
# two drift apart or a sphere stops looking like a circle.
f3d.toolbar.active_scroll = _wheel3
f3d.toolbar.active_tap = _tap3
# Explicitly None, NOT the "auto" default -- auto would make Box Select the
# active drag tool and a bare drag would box-select instead of rotating.  With
# it None, picking Box Select in the toolbar is exactly what suspends rotation.
f3d.toolbar.active_drag = None
f3d.xaxis.visible = False
f3d.yaxis.visible = False
f3d.xgrid.visible = False
f3d.ygrid.visible = False

_add("det", f3d.multi_line(xs="xs", ys="ys", source=det3_src,
                           line_color="#cc4444", line_width=1, alpha=0.45))
# Two renderers over one CDS: the colour mode is a visibility flip, so switching
# it never re-sends 25 000 points.
r_cloud_c = f3d.scatter("u", "v", source=cloud_src, size="sz", fill_alpha="al",
                        line_color=None,
                        fill_color=linear_cmap("cid20", Category20_20, 0, 19))
r_cloud_q = f3d.scatter("u", "v", source=cloud_src, size="sz", fill_alpha="al",
                        line_color=None,
                        fill_color=linear_cmap("q", Viridis256, 0.0, 40000.0))
r_cloud_q.visible = False
_add("cloud", r_cloud_c)
_add("cloud", r_cloud_q)
_add("shwpt", f3d.scatter("u", "v", source=shwpt3_src, size="sz",
                          color="#8fbf8f", fill_alpha="al", line_alpha="al"))
# Every 3-D point layer draws its size and alpha FROM THE COLUMNS, cued or not,
# so _PT_SIZE / _PT_ALPHA below are the single source of truth for both the
# Python fill and the JS frames.  A glyph with its own hardcoded size would drift
# from the JS the first time one of them was edited.
_halos(f3d, None, _add)
r_seg3 = f3d.multi_line(xs="xs", ys="ys", source=seg3_src, line_color="c",
                        line_width=2, line_alpha="a")
_add("segments", r_seg3)
_mark_dashes(f3d, None, _add)
f3d.add_tools(HoverTool(renderers=[r_seg3], tooltips=[
    ("segment", "@sid"), ("cluster", "@cid"), ("pdg", "@pid"),
    ("in shower", "@owner"), ("mark", "@mark")]))
_add("arrows", f3d.multi_line(xs="xs", ys="ys", source=arrow3_src,
                              line_color="c", line_width=3, alpha=0.9))
_add("arrows", f3d.scatter("u", "v", source=head3_src, marker="triangle",
                           size=11, angle="angle", fill_color="c",
                           line_color="c", alpha=0.9))
r_vtx3 = f3d.scatter("u", "v", source=vtx3_src, size="sz", color="#7f7f7f",
                     fill_alpha="al", line_alpha="al")
_add("vertices", r_vtx3)
r_mv3 = f3d.scatter("u", "v", source=mainvtx3_src, marker="star", size="sz",
                    fill_color="#1f77b4", line_color="#08306b", fill_alpha="al",
                    line_alpha="al")
_add("vertices", r_mv3)
_add("gamma", f3d.scatter("u", "v", source=gstart3_src, marker="diamond",
                          size="sz", fill_color="c", line_color="#222222",
                          fill_alpha="al", line_alpha="al"))
_add("gamma", f3d.scatter("u", "v", source=piovtx3_src, marker="star", size="sz",
                          fill_color="#e377c2", line_color="#7b3294",
                          fill_alpha="al", line_alpha="al"))
_add("emstart", f3d.scatter("u", "v", source=emstart3_src, marker="x", size="sz",
                            line_color="c", line_width=4, line_alpha="al"))
_add("emstart", f3d.scatter("u", "v", source=emdir3_src, marker="triangle",
                            size="sz", fill_color="c", line_color="#222222",
                            fill_alpha="al", line_alpha="al"))
# The pick surface is invisible (alpha 0) but still hit-tested -- Bokeh hit tests
# geometry, not paint.  Deliberately NOT registered in RENDER: a layer checkbox
# that silently disabled selection would be a trap.
r_pick = f3d.scatter("u", "v", source=pick_src, size="sz", fill_alpha="al",
                     line_alpha=0.0)
r_pick.nonselection_glyph = r_pick.glyph
# Tap reaches the vertices too (round 4: "click a vertex to say if this is pi0's
# vertex point"), but BOX SELECT DOES NOT.  A box is the bulk-marking gesture and
# its result is fed to `mark`, which keys state["marks"] by segment id; a vertex
# swept up by a box carries no segment id and would write a nonsense key into the
# saved label.  Keeping the two tools' renderer lists different is what makes
# that impossible rather than merely unlikely.
_tap3.renderers = [r_pick, r_vtx3, r_mv3]
_box3.renderers = [r_pick]
for _r in (r_vtx3, r_mv3, r_cloud_c, r_cloud_q):
    _r.nonselection_glyph = _r.glyph

# --- the CustomJS, spliced from em3d so there is one copy of the formula ------
#                     cloud shwpt  pick  vtx  mainvtx gstart piovtx
_PT_SRC = [cloud_src, shwpt3_src, pick_src, vtx3_src, mainvtx3_src, gstart3_src,
           piovtx3_src, emstart3_src, emdir3_src]
_PT_SIZE = [2.6, 2.0, 7.0, 6.0, 20.0, 18.0, 24.0, 20.0, 16.0]
_PT_ALPHA = [0.55, 0.45, 0.0, 0.75, 0.95, 0.95, 0.95, 0.95, 0.95]
_PT_CUE = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# The ONE table both mirrors read: Python's fill3_points looks a source up here,
# and the JS gets the same three lists through args.  Nothing else carries a
# base size or a base alpha for a 3-D point layer.
_PT_CFG = {s: (sz, al, cue > 0.5)
           for s, sz, al, cue in zip(_PT_SRC, _PT_SIZE, _PT_ALPHA, _PT_CUE)}
# Three parallel lists indexed like `pts`, so _PT_CFG above is the only place a
# base size or alpha is written.  See em3d.JS_REDRAW's point loop for the note on
# dict-shaped args (they are fine with string keys, and only with string keys).
_JS_ARGS = dict(cam=cam_src, pts=_PT_SRC, ptsize=_PT_SIZE, ptalpha=_PT_ALPHA,
                ptcue=_PT_CUE,
                # EVERY polyline source, or the ones left out stay frozen at the
                # camera they were filled from and drift off the rest of the
                # picture on the first drag.  selftest_em_display.py counts this
                # list against the module's own _LINE3 registry.
                lines=list(_LINE3),
                heads=[head3_src])
camtxt = TextInput(value="", visible=False)
_js_common = dict(_JS_ARGS, p=f3d, xr=f3d.x_range, yr=f3d.y_range, camtxt=camtxt)
# Named rather than inlined so selftest_em_display.py can lint them: with no JS
# engine in this tree a missing `args` entry would otherwise surface only as a
# ReferenceError in somebody's browser console, mid-scan.
js_panstart = CustomJS(args=_js_common, code=D3.JS_PANSTART)
js_rotate = CustomJS(args=_js_common, code=D3.JS_ROTATE)
js_panend = CustomJS(args=_js_common, code=D3.JS_PANEND)
js_apply = CustomJS(args=_JS_ARGS, code=D3.JS_APPLY)
f3d.js_on_event(PanStart, js_panstart)
f3d.js_on_event(Pan, js_rotate)
f3d.js_on_event(PanEnd, js_panend)
# Any Python-side push of cam_src.data re-runs the SAME projection in the
# browser, so the server's fill can never be the version left on screen.
cam_src.js_on_change("data", js_apply)

# --- the 3-D control column --------------------------------------------------
preset_btns, preset_js = [], []
for _name in D3.PRESET_ORDER:
    _az, _el = D3.PRESETS[_name]
    _b = Button(label=_name, width=70)
    _cb = CustomJS(
        args=dict(_JS_ARGS, camtxt=camtxt, az=math.radians(_az),
                  el=math.radians(_el)),
        code=("const d = cam.data; d.az[0] = az; d.el[0] = el;\n"
              + D3.JS_REDRAW
              + '\ncamtxt.value = az.toFixed(4) + "," + el.toFixed(4);'))
    _b.js_on_click(_cb)
    preset_btns.append(_b)
    preset_js.append(_cb)
refit_btn = Button(label="refit", width=70)
cam_div = Div(text="", width=CW)
cloud_layer = Select(title="charge cloud", value=D3.CLOUD_LAYERS[0],
                     options=D3.CLOUD_LAYERS + ["(none)"], width=170)
cloud_color = RadioButtonGroup(labels=["by cluster", "by charge"], active=0,
                               width=180)
cloud_max = Select(title="max points", value="25000",
                   options=["10000", "25000", "50000", "100000"], width=110)
# Default ON, and this is the round-4 headline.  The dump IS the neutrino
# candidate (NeutrinoID never sees anything but the main cluster and its bundle),
# while `clustering-global` is every cluster in the readout -- a measured median
# 81.4% of which is not the candidate.  See em3d.candidate_clusters.
cloud_scope = RadioButtonGroup(labels=["neutrino candidate", "all clusters"],
                               active=0, width=CW)
cloud_div = Div(name="cloud_div", text="", width=CW)
# One Select rather than a row of toggles: seven actions do not fit as radio
# buttons, and the scanner sets this once and then clicks the event, so a
# dropdown costs nothing per click.
TAP_SELECT = "select segment(s)"
TAP_IN, TAP_OUT, TAP_TOGGLE = "mark IN", "mark OUT", "toggle IN / OUT / clear"
TAP_CENTRE, TAP_XYZ, TAP_PIO = ("orbit around it", "fill x / y / z",
                                "make it the pi0 vertex")
TAP_START, TAP_DIR = ("make it this shower's START",
                      "aim this shower's AXIS through it")
TAP_ACTIONS = [TAP_SELECT, TAP_IN, TAP_OUT, TAP_TOGGLE, TAP_CENTRE, TAP_XYZ,
               TAP_PIO, TAP_START, TAP_DIR]
tap_action = Select(name="tap_action", title="a tap in 3-D does", value=TAP_SELECT,
                    options=TAP_ACTIONS, width=CW)
# Default "frame the shower" since round 5.  Doc pr/114 sec 12.7 left this open;
# the scanner then asked for a table click to bring the 3-D view up on the thing
# they clicked, which is that question answered.  The other two modes still must
# not re-frame on a table click -- only the DEFAULT moved.
fit_mode = RadioButtonGroup(
    labels=["frame the reco", "frame the cloud", "frame the shower"],
    active=2, width=CW)
view_size = Select(title="3-D panel size", value="760",
                   options=["620", "760", "900", "1100"], width=110)
# Default OFF on purpose.  The show_all_toggle comment further down is the record
# of this exact default going the wrong way on the owner once already: hiding or
# fading what is NOT in the shower hides the segments the scan is deciding about.
dim_toggle = Toggle(label="dim what is not in this shower", width=220,
                    active=False)


def _sw(col, w, dash=False):
    return ("<span style='display:inline-block;width:26px;border-top:%dpx %s %s;"
            "vertical-align:middle'></span>" % (w, "dashed" if dash else "solid",
                                                col))


legend_div = Div(width=CW, text=(
    "<span style='font-size:85%%;color:#444'><b>how to read a segment</b><br>"
    "%s &nbsp;in this shower, per the <b>reconstruction</b><br>"
    "%s &nbsp;<b>you</b> marked it IN &nbsp; %s &nbsp;you marked it OUT<br>"
    "%s &nbsp;selected &mdash; the next <i>mark</i> button hits this<br>"
    "%s &nbsp;gamma&nbsp;1 members &nbsp; %s &nbsp;gamma&nbsp;2 members "
    "(pi0 mode)<br>"
    "<i>Yellow inside green = a member you confirmed. Yellow inside red = a "
    "member you are taking out. Green with no yellow = something you are adding."
    "<br>A segment you mark IN is <b>repainted in that shower's colour</b> at "
    "once, and one you mark OUT drops back to grey &mdash; so the colours show "
    "the clustering as YOU are redefining it, while the <i>in shower</i> column "
    "keeps saying what the reconstruction did.</i></span>"
    % (_sw("#ffd27f", 6), _sw("#2ca02c", 6, True), _sw("#d62728", 6, True),
       _sw("#00b8d4", 6), _sw("#1f77b4", 4), _sw("#d62728", 4))))

# --- the acceptance plot ----------------------------------------------------
# Deliberately NOT a cone drawn over the projections.  A 3-D cone does not
# project to a cone, so any wedge drawn on X-Y would be decorative and would
# invite exactly the wrong reading.  This panel is the gate itself, exactly:
# distance and angle are the two quantities NeutrinoShowerClustering.cxx:1310
# actually tests, so a dot below a step is inside that tier and a dot above it
# is not.  Nothing is approximated here.
acc = figure(name="acc", title="pass-1 acceptance: angle to shower axis vs distance",
             height=330, width=520, x_range=Range1d(0, 220),
             y_range=Range1d(0, 90),
             tools="pan,wheel_zoom,box_zoom,reset,save,tap",
             active_scroll="wheel_zoom")
acc.xaxis.axis_label = "distance from shower start (cm)"
acc.yaxis.axis_label = "angle to shower axis (deg)"
tier_src = ColumnDataSource(data=dict(xs=[], ys=[]))
acc.multi_line(xs="xs", ys="ys", source=tier_src, line_color="#666666",
               line_width=2, line_dash="dashed", alpha=0.9)
cand_pt_src = ColumnDataSource(name="cand_pt_src", data=dict(x=[], y=[], c=[], sid=[], pid=[],
                                         length=[], tier=[], owner=[], site=[],
                                         mark=[], mk=[], sz=[]))
# Members are drawn as SQUARES, everything else as circles.  Colour alone was not
# enough: a member is orange, a mark is green or red, and on a 20-shower event
# the eye has to separate those from the palette hue of whatever is underneath.
r_cand = acc.scatter("x", "y", source=cand_pt_src, size="sz", marker="mk",
                     fill_color="c", line_color="#333333", alpha=0.85)
r_cand.nonselection_glyph = r_cand.glyph
acc.add_tools(HoverTool(renderers=[r_cand], tooltips=[
    ("segment", "@sid"), ("pdg", "@pid"), ("length", "@length{0.0} cm"),
    ("tier", "@tier"), ("now in shower", "@owner"),
    ("absorbed by", "@site"), ("your mark", "@mark")]))
# Default ON.  The gate box is 220 cm x 90 deg because tier 3 reaches that far,
# but the members of a real shower live in a corner of it: on evt64591's shower
# 78025 the two member dots that plot sit inside the first 8% of the axis, among
# 29 others.  "I do not see the points belonging to the existing EM shower" was a
# literal and accurate report of that, and a bigger marker does not fix a scale
# problem -- the range does.
acc_zoom = Toggle(name="acc_zoom",
                  label="zoom to this shower (off = the full gate box)",
                  width=330, active=True)
cmp_div = Div(name="cmp_div", text="", width=RW)
acc_note = Div(width=430, text=(
    "<span style='font-size:85%;color:#555'>Steps are the <b>pass-1</b> gate "
    "(<code>pass3_cone</code>, NeutrinoShowerClustering.cxx:1310-1312) &mdash; "
    "the largest single absorber, <b>41%</b> of all absorptions over this "
    "sample, but not the only one. Above every step &ne; rejected: "
    "<code>pass4_angle</code> (21%) and others use different constants. "
    "Below a step &ne; absorbed either: <code>shower_cone_absorb_guard</code> is "
    "SBND-ON and declines a confidently-PID'd non-electron straight track over "
    "50 cm (:1336-1351). <b>The <i>absorbed by</i> column is the authority.</b>"
    "</span>"))


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
mode_group = RadioButtonGroup(name="mode_group",
                              labels=["EM shower", "pi0"], active=0,
                              width=240)
event_select = Select(name="event_select", title="event", options=LABELS,
                      value=LABELS[0], width=190)
prev_btn = Button(label="< prev", width=80)
next_btn = Button(label="next >", width=80)
LAYERS = [("segments", "track fit"), ("member", "shower members"),
          ("mark", "your marks"), ("select", "selection"), ("arrows", "axes"),
          ("vertices", "vertices"), ("shwpt", "shower pts"),
          ("gamma", "gammas / pi0 vtx"), ("det", "volume"),
          ("cloud", "charge cloud (3-D)"), ("emstart", "shower start")]
LAYER_KEYS = [k for k, _ in LAYERS]
# Every key except "shower pts" (index 6), which stays off as before.  A renderer
# registered under a key that is NOT in this list is invisible forever, because
# apply_layers only ever turns on keys the checkbox group can name -- so adding a
# layer means adding it here too.
layer_group = CheckboxButtonGroup(labels=[t for _, t in LAYERS],
                                  active=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
# `name=` on these two so selftest_em3d_browser.py can read them out of the live
# document -- the banner's wording is a claim about the data ("built, not
# uploaded") and the hint must be provably on screen, so both are asserted
# against the RUNNING app, not just against the Python object.
banner = Div(name="banner", text="", width=RW + CW)
# The owner's own hint for this event, from the manifest's `scan_note` column
# (docs/pr/pr114-owner-adds.index.txt).  READ-ONLY, and deliberately not the same
# widget as `note_in`: note_in is the scanner's editable text and is what gets
# written into label["note"].  Loading the hint into note_in would mean the first
# save either overwrote the hint or recorded it as though the scanner had typed
# it -- and then a later reader could not tell the question from the answer.
scan_note_div = Div(name="scan_note_div", text="", width=RW + CW)
# Round 7: has THIS event already been scanned in this tag?  The scan is a long
# stop-start job across 98 events and the question "did I already do this one"
# was being answered by squinting at the n/98 counter, which cannot answer it.
# Read from the FILESYSTEM, not from state["saved"]: state["saved"] is a
# load-time snapshot, so a second tab open on the same tag would keep claiming
# "not scanned yet" after this tab wrote the file.  state["saved"] is used only
# for the timestamp, and dropped when it is not ours to quote.
#
# Deliberately DISK STATE ONLY.  Unsaved-edit state is already rendered by
# refresh_info() as [unsaved]; duplicating it here would give two indicators
# that can disagree.
scan_status = Div(name="scan_status", text="", width=RW + CW)
info = Div(text="", width=RW)

# Dim whole showers out of the way.  The scan question is always "does this
# piece belong to THAT shower", and on a busy event the other showers' segments
# are the noise in that judgement -- so this drives the same alpha column the
# 3-D and 2-D panels already read, and drops the excluded segments from the
# candidate table, rather than being a table-only filter.
excl_choice = MultiChoice(name="excl_choice", title="dim these showers away (3-D, projections and "
                                "the candidate table)", options=[], value=[],
                          width=RW - 10)
seg_color_mode = RadioButtonGroup(
    name="seg_color_mode",
    labels=["colour by shower", "colour by segment"], active=0, width=CW)

shower_src = ColumnDataSource(name="shower_src", data=dict(node=[], pdg=[], nseg=[], joined=[],
                                        E=[], kb=[], conn=[], pio=[], length=[],
                                        drift=[], flag=[], color=[]))
shower_view_a, shower_view_b = AllIndices(), AllIndices()
shower_view = CDSView(filter=shower_view_a)
# The swatch is the whole point of colouring by shower: without a key in the
# table the hues in the 3-D view say "these two are the same" but never "the same
# as WHICH row".
SWATCH = HTMLTemplateFormatter(template=(
    "<span style='display:inline-block;width:22px;height:11px;border:1px solid "
    "#555;background:<%= value %>'></span>"))
shower_tab = DataTable(source=shower_src, view=shower_view, width=RW, height=210,
                       index_position=None, columns=[
    TableColumn(field="color", title="", width=34, formatter=SWATCH),
    TableColumn(field="node", title="shower id", width=80),
    TableColumn(field="pdg", title="pdg", width=50),
    TableColumn(field="nseg", title="nseg", width=50),
    TableColumn(field="joined", title="joined", width=70),
    TableColumn(field="E", title="kine_charge", width=90,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="kb", title="kine_best", width=85,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="length", title="len cm", width=70,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="conn", title="conn", width=50),
    TableColumn(field="drift", title="axis-drift", width=80,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="pio", title="pio_id", width=60),
    TableColumn(field="flag", title="note", width=180)])

cand_src = ColumnDataSource(name="cand_src", data=dict(sid=[], cid=[], pdg=[], length=[], dist=[],
                                      angle=[], tier=[], metric=[], owner=[],
                                      site=[], mark=[]))
cand_view_a, cand_view_b = AllIndices(), AllIndices()
cand_view = CDSView(filter=cand_view_a)
cand_tab = DataTable(source=cand_src, view=cand_view, width=RW, height=250,
                     index_position=None, columns=[
    TableColumn(field="sid", title="segment", width=75),
    TableColumn(field="cid", title="cluster", width=60),
    TableColumn(field="pdg", title="pdg", width=50),
    TableColumn(field="length", title="len cm", width=65,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="dist", title="dist cm", width=70,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="angle", title="angle deg", width=75,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="tier", title="pass-1", width=60),
    TableColumn(field="metric", title="ellip", width=65,
                formatter=NumberFormatter(format="0.00")),
    TableColumn(field="owner", title="in shower", width=80),
    TableColumn(field="site", title="absorbed by", width=140),
    TableColumn(field="mark", title="mark", width=55)])

mark_in_btn = Button(label="mark IN", button_type="success", width=100)
mark_out_btn = Button(label="mark OUT", button_type="danger", width=100)
mark_q_btn = Button(label="mark ?", width=90)
mark_clear_btn = Button(label="unmark", width=90)
# Round 10.  A WHOLE shower at once.  The scan question is often "shower B is
# really part of shower A" -- evt172942 is the case that asked for it: shower
# 71022 (10 segments across 8 clusters, 96.9 MeV) is the same EM shower as 4002
# (6 segments, 368.8 MeV) -- and answering it one segment at a time is ten
# clicks that can each land on the wrong row.
#
# It SELECTS, it does not mark.  The selection is what the cyan halo draws and
# what `selected_cand_ids()` hands the mark buttons, so pressing this puts
# exactly the segments that are about to be marked on the screen first, and all
# four mark buttons then work on them unchanged -- including `mark OUT`, which
# makes "this whole shower is NOT part of the one I am scanning" one gesture
# too.  A direct "mark all IN" button would save one click and remove the look.
bulk_shower = Select(name="bulk_shower", title="whole shower", value="",
                     options=[""], width=330)
bulk_sel_btn = Button(name="bulk_sel_btn", label="select all its segments",
                      width=190)
bulk_add_btn = Button(name="bulk_add_btn", label="add to selection", width=150)
# Default ON.  The owner's question is symmetric -- "should this be inside or
# outside" -- and the `absorbed by` column only ever has something to say about a
# segment that WAS absorbed, i.e. a member.  Hiding members by default made that
# column read empty in the default view, which is exactly backwards: "why is this
# one IN" is the question the probe can actually answer.
show_all_toggle = Toggle(label="show members too (off = only segments outside "
                               "this shower)", width=380, active=True)
em_verdict = RadioButtonGroup(labels=EM_VERDICTS, active=None)
impact = Div(text="", width=RW)
# Every mark in the event, by shower.  With marks keyed per shower the halos can
# only ever show the shower being scanned, so without this the other showers'
# marks would be invisible until the table moved back to them.
marks_div = Div(name="marks_div", text="", width=RW)

# --- pi0 controls -----------------------------------------------------------
g1_btn = Button(label="selected shower -> gamma 1", width=210)
g2_btn = Button(label="selected shower -> gamma 2", width=210)
g_clear_btn = Button(label="clear gammas", width=120)
gstart_slot = RadioButtonGroup(labels=["gamma 1", "gamma 2"], active=0, width=160)
# Round 9.  Which recombination pair a gamma's CHARGE is converted with.
#
# kine_charge is charge / (recom * fudge), and WHICH pair the reconstruction used
# was decided by Shower::get_flag_shower() -- not by the slot a scanner later
# puts the object in.  A track- or proton-flagged object dropped into a gamma
# slot therefore carries a track's or a proton's energy, which is the wrong
# number for a photon: evt166870's shower 85045 is pdg 13, flag_shower False,
# and its 38.6 MeV becomes 64.2 MeV under the shower pair -- moving the pi0 mass
# from 116.1 to 149.7.
#
# DEFAULT IS "as reconstructed", deliberately.  Re-opening a record saved before
# this control existed must show the mass it was saved with; a default of "as EM"
# would silently re-price every past scan with no diff and no flag.  Flipping it
# is one click and both numbers are on screen.
EHYP_RECO = "as reconstructed"
EHYP_EM = "as EM shower (charge-inferred)"
g1_ehyp = Select(name="g1_ehyp", title="gamma 1 energy", value=EHYP_RECO,
                 options=[EHYP_RECO, EHYP_EM], width=250)
g2_ehyp = Select(name="g2_ehyp", title="gamma 2 energy", value=EHYP_RECO,
                 options=[EHYP_RECO, EHYP_EM], width=250)
# Round 12.  WHICH SEGMENTS a gamma's energy is summed over.
#
# `kine_charge` is the reconstruction's, over the segments the reconstruction
# put in the shower.  Marking a segment IN changes the membership and nothing
# else -- so on evt409634, merging shower 27015 (10 segments, 105.05 MeV) into
# shower 69032 (39.06 MeV) left the pi0 mass built on 39.06.  Round 10 made that
# merge one gesture, which is how the gap surfaced.
#
# DEFAULT IS THE RECONSTRUCTION'S MEMBERSHIP, and not as a matter of taste: six
# of the twelve records already on disk carry marks on a gamma's shower, so a
# default of "include them" would silently re-price six masses the scanner has
# already judged, with no diff and no flag.  The panel says out loud what the
# mass becomes; flipping it is one click and both numbers go in the record.
EMARK_RECO = "reco membership only"
EMARK_MARKS = "include my IN / OUT marks"
# Round 16b, and the owner's decision.  Round 12 made counting the scanner's own
# IN / OUT marks OPT-IN, and the consequence was measured in round 16: of the 18
# records on disk, every single one that carries marks was saved with
# `energy_includes_marks: false` on both gammas -- 14 segments marked into 14059
# on evt76346, 10 into 69032 on evt409634 (worth +105.1 MeV), 10 and 8 on
# evt54332 -- so no hand-made clustering fix reached ANY saved mass.  The marks
# exist because the reconstruction's membership is wrong; counting them is the
# answer the scan is for.  The default is therefore ON.
#
# This changes what a NEW save records.  It re-prices nothing already on disk:
# load_label below sets this switch from the record itself, absent still meaning
# off, so every existing record re-opens showing exactly the mass it was saved
# with.  Verified over all 18 labelled events, 0 differences (doc pr/114 24.7).
emark_mode = Select(name="emark_mode", title="gamma energy membership",
                    value=EMARK_MARKS, options=[EMARK_RECO, EMARK_MARKS],
                    width=290)
# Round 18, and the owner's instruction after round 17 showed them what was
# being dropped: *"when we added a particle into an EM shower, it should then be
# treated as an EM shower to do the calculation ... have a reasonable guess, it
# is fine for now."*  A marked segment the reconstruction put in NO shower has
# no per-segment E_est; this switch decides whether its charge is converted with
# the target shower's own factor and counted, or left out as rounds 12-17 left
# it.  196 of the sample's 5 830 segments (3.4%, on 84 of 98 events) are in that
# position, so it is not a corner.
#
# DELIBERATELY ITS OWN SWITCH, not a third state of `gamma energy membership`:
# that one means "marks with a known E_est" and its round-12/16b restore keys on
# `energy_includes_marks`.  Giving it a second meaning would re-price every
# record already saved with marks ON the moment it was re-opened.  Here, absent
# means OFF -- exactly what every record written before round 18 was saved with
# -- while a fresh event takes the default below.
EORPH_OUT = "leave them out"
EORPH_IN = "count as part of the shower"
orph_mode = Select(name="orph_mode",
                   title="marked segments no shower owns",
                   value=EORPH_IN, options=[EORPH_OUT, EORPH_IN], width=290)
# Round 11.  One event, several pi0 -- and, just as often, several ways to pair
# the gammas of one.  evt281485 is the case: 19 EM showers, the reconstruction
# found ONE group at 208.4 MeV, and the scanner reads more than one pi0 in it.
# Two slots hold one pairing, so until now a second mass could only be recorded
# by overwriting the first.
#
# A stored candidate is FROZEN NUMBERS, not a reference to the slots.  Every
# input a mass is built from stays editable afterwards -- `state["em_start"]` in
# particular is keyed by SHOWER and lives in EM mode -- so a candidate that
# merely named its showers would be silently re-priced by a later start
# correction.  Same discipline as `marks_detail`, measured at save time.
pio_add_btn = Button(name="pio_add_btn", label="store this pairing",
                     button_type="success", width=180)
pio_load_btn = Button(name="pio_load_btn", label="load into the slots",
                      width=170)
pio_del_btn = Button(name="pio_del_btn", label="remove", width=100)
# Round 16.  The exit from the freeze: a stored row can be replaced by today's
# answer deliberately, in one press, instead of being left stale or deleted and
# re-stored (which would lose its stored_utc and its place in the list).
pio_upd_btn = Button(name="pio_upd_btn", label="update to today's numbers",
                     width=215)
pio_cand_src = ColumnDataSource(name="pio_cand_src",
                                data=dict(n=[], g1=[], g2=[], e1=[], e2=[],
                                          th=[], mA=[], mB=[], how=[], flag=[]))
pio_cand_tab = DataTable(source=pio_cand_src, width=RW, height=170,
                         index_position=None, columns=[
    TableColumn(field="n", title="#", width=30),
    TableColumn(field="g1", title="gamma 1", width=75),
    TableColumn(field="g2", title="gamma 2", width=75),
    TableColumn(field="e1", title="E1 MeV", width=70,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="e2", title="E2 MeV", width=70,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="th", title="theta deg", width=75,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="mA", title="m axis", width=70,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="mB", title="m vertex", width=75,
                formatter=NumberFormatter(format="0.0")),
    TableColumn(field="how", title="vertex", width=90),
    TableColumn(field="flag", title="note", width=230)])
pio_cand_div = Div(name="pio_cand_div", text="", width=RW)
snap_btn = Button(label="snap start to nearest fit point", width=230)
# Round 8, EM mode.  Separate widgets from the pi0 ones on purpose: man_x/y/z
# and snap_btn belong to the pi0 vertex, the panels are switchable, and one pair
# of boxes meaning two different things depending on a radio elsewhere is how a
# scanner ends up moving the wrong point.
# Round 8.  An EVENT-level topology flag, not a per-shower one: it describes the
# whole event and it is what decides that the event needs different treatment
# downstream, so it must be readable without knowing which shower was selected
# when it was set.  A LIST rather than a boolean so the next class the scanner
# meets is a one-line data change here and in the doc, with no schema change to
# labels already on disk.
EVENT_FLAGS = [("no_vertex_ncpi0",
                "no-vertex \u03c0\u2070 (NC\u03c0\u2070) "
                "\u2014 needs separate treatment")]
EVENT_FLAG_KEYS = [k for k, _ in EVENT_FLAGS]
event_flag_group = CheckboxGroup(name="event_flag_group",
                                 labels=[t for _, t in EVENT_FLAGS], active=[],
                                 width=RW)
emstart_div = Div(name="emstart_div", text="", width=RW)
em_startv_btn = Button(label="start = nearest vertex", width=175)
em_startp_btn = Button(label="start = nearest fit point", width=175)
em_dirp_btn = Button(label="aim axis at nearest fit point", width=205)
em_start_reset = Button(label="reset start", width=105)
em_dir_reset = Button(label="reset axis", width=105)
em_sx = TextInput(name="em_sx", title="start x", value="", width=90)
em_sy = TextInput(name="em_sy", title="start y", value="", width=90)
em_sz = TextInput(name="em_sz", title="start z", value="", width=90)
em_setxyz_btn = Button(label="use these", width=95)
gstart_reset = Button(label="reset to reco start", width=170)
vtx_mode_group = RadioButtonGroup(
    name="vtx_mode_group",
    labels=["main vertex", "back-project the two gammas", "manual"], active=0)
# Round 14.  The back-projection is a mirror of id_pi0_without_vertex, and it
# built BOTH gamma rays from the reconstruction: `test_p` = the shower's fitted
# point closest to the main vertex, direction = dir15 evaluated there.  So a
# scanner who had corrected a shower's start and aimed its axis by hand watched
# the panel print "your corrected start from EM mode" and then back-project from
# the reco's geometry anyway.  Round 8b fixed exactly this disease for the two
# mass conventions and did not reach this function.
#
# Both readings are legitimate and they answer different questions -- "what
# vertex would the CODE compute here" versus "what vertex does MY geometry
# imply" -- so both are offered rather than one being chosen silently.  The
# default is the scanner's, because a start override or an aimed axis exists for
# exactly one reason.  Nothing on disk is re-priced by that default: a record
# written before this round restores to `reco` (see load_label), and a gamma the
# scanner never touched injects no ray at all, which is byte-for-byte the
# pre-round-14 mirror -- measured over all 2490 shower pairs of the 98 manifest
# events, 0 differences.
BPG_RECO = "the reconstruction's own rays (mirror)"
BPG_SCAN = "my corrected start / axis"
bp_geom = Select(name="bp_geom", title="back-projection geometry",
                 value=BPG_SCAN, options=[BPG_SCAN, BPG_RECO], width=310)
# `name=` so the browser test can read the boxes back out of the live page --
# the round-15 leak was a UI symptom and is pinned where it was seen.
man_x = TextInput(name="man_x", title="x", value="", width=90)
man_y = TextInput(name="man_y", title="y", value="", width=90)
man_z = TextInput(name="man_z", title="z", value="", width=90)
tap_toggle = Toggle(label="tap fills x/y/z", width=140)
kine_div = Div(name="kine_div", text="", width=RW)

conf_group = RadioButtonGroup(labels=CONF, active=None, width=240)
note_in = TextInput(title="note (optional)", value="", width=520)
save_btn = Button(label="Save event label", button_type="success", width=170)
save_note = Div(name="save_note", text="", width=RW)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def seg_color(i):
    return Category20_20[i % 20]


def labels_dir():
    return os.path.join(SX, "em_labels", SCAN_TAG)


def label_path(lbl):
    return os.path.join(labels_dir(), "labels-%s.json" % lbl)


def tag_has_labels():
    d = labels_dir()
    return os.path.isdir(d) and bool(glob.glob(os.path.join(d, "labels-*.json")))


def write_allowed():
    """M13: never append to somebody else's scan record by accident.  An
    explicitly-passed --scan-tag is consent; the implicit default may CREATE a
    tag but never add to one that already holds labels."""
    return SCAN_TAG_EXPLICIT or not tag_has_labels()


def done_count():
    return sum(1 for l in LABELS if os.path.exists(label_path(l)))


def cur_segments():
    return (state["data"] or {}).get("segments") or []


def cur_showers():
    return (state["data"] or {}).get("showers") or []


def shower_by_node(node):
    for sh in cur_showers():
        if sh.get("id") == node:
            return sh
    return None


def probe_members(node):
    """Non-lossy membership from the stage-2 sidecar, or None when there is no
    sidecar for this event.  This is the ONLY faithful source when two showers
    overlap -- see em_geom.join_completeness."""
    pr = state.get("prep")
    if not pr:
        return None
    e = (pr.get("showers") or {}).get(str(node))
    if not e:
        return None
    return [m["seg"] for m in e.get("members", [])]


def members_of(node):
    """Segment ids of a shower: the probe if we have it, else the dump join."""
    pm = probe_members(node)
    if pm is not None:
        return pm
    sh = shower_by_node(node)
    if not sh:
        return []
    return [s["id"] for s in G.shower_members(sh, cur_segments())]


def owner_map():
    """{segment id: the shower that owns it}, first owner wins.

    Rebuilt on demand rather than cached: membership comes from the probe
    sidecar, and an event without one falls back to the dump join, so the map is
    only as stable as `members_of` -- which is the property the callers want.
    """
    out = {}
    for sh in cur_showers():
        for s in members_of(sh.get("id")):
            out.setdefault(s, sh.get("id"))
    return out


def marks_for(node):
    """The marks recorded AGAINST one shower.

    Round 5.  Marks used to be one flat {segment: in/out} for the whole event,
    and the saved record named a single `em.shower` -- whichever row happened to
    be selected when Save was pressed.  A mark made while shower A was up and
    saved after the table had moved to B was therefore written against B, with
    nothing in the file to say otherwise, and `on_gamma` moves the table
    selection as a side effect of assigning a pi0 slot -- so the pi0 workflow
    reaches that state on its own.  Keying by shower removes the ambiguity at
    the source and lets one event hold marks for several showers at once.

    NOTE: the keys are showers that have been LOOKED at, not showers that have
    marks -- setdefault creates an entry on read, and `focus_points` reads.
    `marks_pruned` is what the record is built from.
    """
    if node is None:
        return {}
    return state["marks"].setdefault(node, {})


def marks_flat():
    """{segment id: (shower node, kind)} over every shower in the event."""
    out = {}
    for node, mk in state["marks"].items():
        for sid, kind in mk.items():
            out[sid] = (node, kind)
    return out


def marks_pruned():
    """The mark map with empty per-shower entries dropped.

    `marks_for` uses setdefault, so merely *looking* at a shower creates a key.
    Saving that would put empty objects in the record and make a shower look
    scanned when it was only glanced at."""
    return {n: dict(mk) for n, mk in state["marks"].items() if mk}


# Category20 is ordered as ten HUE PAIRS -- dark blue, light blue, dark orange,
# light orange ... -- so consecutive indices are two shades of one colour.  Taken
# raw that gave a pi0's two gammas #1f77b4 and #aec7e8, which is the one
# comparison that must not be ambiguous.  Walk the ten dark entries first and
# only then their light twins.
SHOWER_PALETTE = ([Category20_20[i] for i in range(0, 20, 2)]
                  + [Category20_20[i] for i in range(1, 20, 2)])
NO_SHOWER_COLOR = "#9aa5b1"


def shower_color(node):
    """One colour per shower, so two segments of the same shower look alike.

    Keyed on the shower's position in this event's own shower list, not on a hash
    of the id: stable while the event is open, and neighbouring rows in the table
    get neighbouring palette entries instead of colliding by chance.  Segments no
    shower claims stay neutral grey -- they are the ones the scan is deciding
    about, and giving them a hue of their own would read as membership.
    """
    if node is None or node == "-":
        return NO_SHOWER_COLOR
    order = state.get("shorder") or {}
    if node not in order:
        return NO_SHOWER_COLOR
    return SHOWER_PALETTE[order[node] % len(SHOWER_PALETTE)]


def effective_owner(sid, own=None):
    """Which shower a segment belongs to AFTER the scanner's marks.

    The colour is meant to answer "which pieces are one object", so once you say
    a piece belongs to a shower it has to LOOK like that shower -- otherwise the
    display keeps showing the reconstruction's answer while you are recording a
    different one, which is the confusing half of the picture.  A member marked
    OUT drops back to neutral for the same reason.  `owner_map` stays the
    reconstruction's own answer; this is yours.
    """
    if own is None:
        own = owner_map()
    for node, mk in state["marks"].items():
        if mk.get(sid) == "in":
            return node
    base = own.get(sid)
    if base is not None and state["marks"].get(base, {}).get(sid) == "out":
        return None
    return base


def excluded_segments():
    """Segments belonging to the showers the scanner has dimmed away."""
    out = set()
    for node in state["excl"]:
        out.update(members_of(node))
    return out


def absorb_site(sid):
    pr = state.get("prep")
    if not pr:
        return ""
    recs = (pr.get("absorb") or {}).get(str(sid)) or []
    if not recs:
        return ""
    last = recs[-1]
    s = last.get("site") or last.get("how") or ""
    return "%s (%s)" % (s, last.get("how", ""))


def shower_axis(node, use_override=True):
    """(dir, branch, source).  Prefers the probe's dir15 -- that is the C++'s own
    `shower_cal_dir_3vector(shower, start, 15cm)`, so it needs no reproduction
    caveat.  Without a sidecar, falls back to the Python `init_dir` mirror.

    Round 8, and this is the invariant the whole feature rests on: THE START AND
    THE AXIS MUST MOVE TOGETHER.  The probe's dir15 is anchored at the
    reconstruction's start.  If the scanner moves the start and this still
    returned dir15, seg_vs_shower would compute the angle between a direction
    anchored at the old start and a displacement measured from the new one --
    not a physical quantity, and one that looks entirely plausible in the
    acceptance plot and in the saved marks_detail.  So an overridden start
    invalidates the probe value, and the source string stops saying "probe".
    """
    dp = state["em_dir"].get(node) if use_override else None
    ov = state["em_start"].get(node) if use_override else None
    if dp is not None:
        base = ov if ov is not None else reco_start(node)
        if base is not None:
            d = G.vsub(dp, base)
            if G.vmag(d) > 0:
                # The scanner aimed it by eye at a second point: exact by
                # construction, no membership caveat at all.
                return G.vnorm(d), "two_point", "manual@override"
    if ov is not None:
        # Same formula the probe used -- shower_cal_dir_3vector at 15 cm -- just
        # evaluated at the new point.  Over the RECO's member set, deliberately:
        # letting marks in here would move two inputs at once and make the
        # before/after comparison uninterpretable.  Memoised because
        # mark_metrics calls seg_vs_shower once per segment and this walks every
        # member point.
        key = (node, ov)
        d = state["_axis_cache"].get(key)
        if d is None:
            segs = {sg.get("id"): sg for sg in cur_segments()}
            mem = [segs[i] for i in members_of(node) if i in segs]
            d = G.shower_cal_dir_3vector(mem, ov, 15.0)
            state["_axis_cache"][key] = d
        if G.vmag(d) > 0:
            # NOT "probe": em_geom:161 says the Python mirror is not bit-exact
            # (shower_ordered_edges vs fill_sets membership).
            return d, "dir15", "python@start_override"
    pr = state.get("prep")
    if pr:
        e = (pr.get("showers") or {}).get(str(node))
        if e and e.get("dir15"):
            d = tuple(e["dir15"])
            if G.vmag(d) > 0:
                # RE-NORMALISE.  The probe prints the components at %.3f, so the
                # vector arrives with |v| in 0.9994..1.0004 -- the C++ value is a
                # unit vector and this is print rounding, not a real magnitude.
                # ~1e-3 on the components is ~0.06 deg on any angle computed from
                # it, which is far below anything a hand scan turns on, but the
                # vector must still BE a unit vector for the arithmetic below.
                return G.vnorm(d), "dir15", "probe"
    sh = shower_by_node(node)
    if not sh:
        return (0.0, 0.0, 0.0), "none", "none"
    vb = {v["id"]: v for v in ((state["data"] or {}).get("vertices") or [])}
    d, br = G.shower_init_dir(sh, cur_segments(), vb)
    return d, br, "python"


def start_source(node, slot=None):
    """Which of the three starts shower_start() just returned.  One function so
    the pi0 panel, the EM readout and the saved record cannot disagree."""
    if slot is not None and state["gstart"].get(slot):
        return "gamma_slot_override"
    if state["em_start"].get(node) is not None:
        return "em_start_correction"
    return "reco"


def reco_start(node):
    """The reconstruction's OWN start for a shower, never the scanner's override.

    Kept separate from shower_start so the record can carry both and a later
    reader can see how far the hand scan moved it."""
    sh = shower_by_node(node)
    return G.pt(sh.get("start")) if sh else None


def shower_start(node, slot=None):
    """The start point the pass-1 gate is measured from.

    Round 8.  `slot is not None` is the pi0 path and is untouched: every pi0
    caller passes a slot, and the three bare calls -- the candidate table,
    seg_vs_shower and mark_metrics -- are exactly the EM gate's consumers, which
    is why the override plugs in here and nothing downstream needs to know.
    Putting the em_start lookup ahead of the slot test instead would silently
    move the gamma start points too."""
    if slot is not None and state["gstart"].get(slot):
        return state["gstart"][slot]
    # Round 8b.  An EM start correction belongs to the SHOWER, not to EM mode, so
    # it applies on the pi0 path too.  Before this it did not, and the result was
    # worse than a missing feature: shower_axis() takes no slot and so ALREADY
    # used the corrected start, while shower_start(node, slot) returned the
    # reco's -- so mass_axis_convention was computed from the scanner's geometry
    # and mass_vertex_convention from the reconstruction's, in the same record,
    # with nothing on screen saying so.
    #
    # Precedence, most specific first:
    #   gstart[slot]   a start set for THIS gamma slot, in pi0 mode  (above)
    #   em_start[node] the shower's corrected start                  (here)
    #   the dump's own start                                         (fallback)
    if state["em_start"].get(node) is not None:
        return state["em_start"][node]
    return reco_start(node)


def gamma_scan_ray(node, slot):
    """(ray, provenance) for ONE gamma of the back-projection, or (None, why).

    `None` is the load-bearing return value.  It means "the scanner corrected
    nothing about this gamma", and it makes `em_geom.pi0_backproject` fall
    through to its own `gamma_ray` -- which is byte-for-byte the pre-round-14
    function.  Rebuilding the un-overridden ray here instead would look
    equivalent and is not: this function resolves membership through
    `members_of` (the probe sidecar) while `gamma_ray` uses the dump join, and
    the two differ on exactly the lossy showers em_geom.join_completeness names
    -- evt347129 among them, which is one of the two records on disk that
    already used back-projection.  So the guarantee "an untouched gamma cannot
    move" is a property of returning None, not of the arithmetic agreeing.

    Precedence matches shower_start()/shower_axis() so the ray, the two mass
    conventions and the EM gate cannot disagree about what the scanner said.
    """
    if node is None:
        return None, "no gamma"
    dp = state["em_dir"].get(node)
    moved = start_source(node, slot) != "reco"
    if not moved and dp is None:
        return None, "reco@closest_to_anchor"

    if moved:
        p = shower_start(node, slot)
        psrc = ("gamma_slot_override" if state["gstart"].get(slot)
                else "em_start_correction")
    else:
        # Only the axis was aimed.  Keep the MIRROR's own origin -- the fitted
        # point closest to the main vertex.  Not the shower's `start`: those are
        # different points, and substituting one for the other would move the
        # vertex for a scanner who only touched the direction.
        sh = shower_by_node(node)
        anchor = G.pt((state["data"] or {}).get("main_vertex"))
        if sh is None or anchor is None:
            return None, "no anchor to build the reco start point from"
        _r, p, _d = G.gamma_ray(sh, cur_segments(), anchor)
        psrc = "reco_ray_point"
    if p is None:
        return None, "no fitted point"
    p = tuple(p)

    if dp is not None:
        # The ray goes through the point the scanner aimed at, measured FROM the
        # start this gamma is actually using.  Identical to shower_axis()'s
        # two_point branch whenever they share a start, which is every case
        # except a slot-scoped gstart -- and there this is the more faithful of
        # the two, because the ray it builds is the one drawn on screen.
        v = G.vsub(dp, p)
        if G.vmag(v) == 0:
            return None, psrc + " + axis point sits on the start"
        return (p, G.vnorm(v)), psrc + " + manual_axis"

    # Start moved, axis not aimed: the code's own recipe re-evaluated at the new
    # point -- shower_cal_dir_3vector at 15 cm, the same formula gamma_ray uses.
    # Over PROBE membership, deliberately: this is shower_axis()'s
    # `python@start_override` branch, and the panel shows that axis, so the ray
    # has to be built from the same member set.  It is a divergence from the
    # mirror's dump-join membership and is named in the provenance string.
    segs = {sg.get("id"): sg for sg in cur_segments()}
    mem = [segs[i] for i in members_of(node) if i in segs]
    dv = G.shower_cal_dir_3vector(mem, p, 15.0)
    if G.vmag(dv) == 0:
        return None, psrc + " + dir15 undefined (no member point within 15 cm)"
    return (p, dv), psrc + " + dir15@probe_members"


# kine_charge converts collected charge to MeV as
#     E = sum_p(w_p Q_p)/sum(w) / recom / fudge * w_value * 1e-6
# (`NeutrinoEnergyReco.cxx:188`), so E scales as 1/(recom*fudge).  WHICH pair is
# used is decided by `Shower::get_flag_shower()` -- NOT by the PDG:
#     kShowerTrajectory || kShowerTopology || |pdg| == 11,  on the START SEGMENT
# (`PRShower.cxx:1460-1464` and `:1578-1582`, `:204`).  A muon-PID'd object with
# neither shower flag therefore gets the TRACK pair even when a scanner is about
# to call it a gamma.  Values are the C++ defaults
# (`NeutrinoPatternBase.h:41-52`); SBND sets none of them
# (`wct-pr-perevt.jsonnet:674-689` documents them and leaves them at default).
KINE_TRACK = (0.7, 0.95)          # recom, fudge
KINE_SHOWER = (0.5, 0.8)
KINE_PROTON = (0.35, 0.95)        # fudge deliberately stays at the track value


def shower_is_em(node):
    """Mirror of `Shower::get_flag_shower()` off the dump.

    The dump's per-segment `flag_shower` is `kShowerTrajectory ||
    kShowerTopology` (`PrDisplayDump.cxx:469-470`); the third disjunct, |pdg|==11,
    is added here.  The shower's `id` IS its start segment's id
    (`PrDisplayDump.cxx:576`), so the start segment is a lookup, not a guess.
    Returns None when the start segment is not in the dump.
    """
    for s in cur_segments():
        if s.get("id") == node:
            return bool(s.get("flag_shower")) or abs(s.get("particle_id") or 0) == 11
    return None


def kine_hypothesis(node):
    """(label, (recom, fudge), alternative-energy) for one shower.

    The alternative is the SAME collected charge re-converted under the other
    recombination hypothesis -- what the energy would be if the object's
    track/shower flag were the other way.  Nothing is re-measured; it is one
    ratio, which is exactly why it is honest to show it."""
    sh = shower_by_node(node) or {}
    e = sh.get("kine_charge")
    em = shower_is_em(node)
    if em is None or e is None:
        return None, None, None
    if em:
        used, other, lbl = KINE_SHOWER, KINE_TRACK, "shower"
    elif abs(sh.get("particle_id") or 0) == 2212:
        used, other, lbl = KINE_PROTON, KINE_SHOWER, "proton"
    else:
        used, other, lbl = KINE_TRACK, KINE_SHOWER, "track"
    alt = e * (used[0] * used[1]) / (other[0] * other[1])
    return lbl, used, alt


def shower_energy(node):
    """Sum of member kine_charge is NOT available per segment in the dump, so the
    shower's own kine_charge is used -- and that is the right number anyway: it
    is exactly what the C++ mass formula reads (get_kine_charge(), see
    NeutrinoShowerClustering.cxx:3771)."""
    sh = shower_by_node(node)
    return (sh or {}).get("kine_charge")


def _est_map():
    """{segment id: (E_est MeV, owning shower)} from the probe sidecar.

    `E_est` is the probe's own per-segment decomposition of `kine_charge`, and
    it is EXACT: shower 69032's two members are 29.498 + 9.560 = 39.058, which
    is its kine_charge to the last digit.  So adding a marked segment's E_est is
    arithmetic on the same number the C++ mass formula reads
    (NeutrinoShowerClustering.cxx:3771), not a proxy for it.
    """
    pr = state.get("prep")
    out = {}
    if not pr:
        return out
    for node, e in (pr.get("showers") or {}).items():
        try:
            nd = int(node)
        except (TypeError, ValueError):
            continue
        for m in (e.get("members") or []):
            out.setdefault(m.get("seg"), (m.get("E_est"), nd))
    return out


PDG_SAYS = {11: "electron", -11: "electron", 13: "muon", -13: "muon",
            211: "pion", -211: "pion", 2212: "proton", 321: "kaon",
            -321: "kaon", 0: "no PID"}


def seg_dq(sid):
    """Collected charge on ONE segment: the sum of its fitted points' `dQ`, in
    electrons, straight out of the dump.

    This is the same quantity the probe reports per member segment.  Measured,
    not assumed: over the 5 700 member segments of all 98 manifest events the
    worst relative difference between the sidecar's `dQ` and this sum is
    5.9e-6, which is the sidecar printing one decimal.  That identity is the
    whole reason a segment the probe never saw can be put on the same scale as
    one it did WITHOUT inventing a charge-to-energy conversion for it.

    Deliberately NOT `refresh_impact`'s sum, which drops points with dQ < 0 or
    dx <= 0.  The probe keeps those (segment 53075 on evt169626 is -12 500 e)
    and the two differ by ~1% over a shower, so a fraction whose numerator and
    denominator came from different filters would not mean anything.  Noted
    rather than unified: `refresh_impact` is a different, older readout with
    its own wording, and changing its number is not this round's business.
    """
    for s in cur_segments():
        if s.get("id") == sid:
            return sum(p.get("dQ") or 0.0 for p in (s.get("points") or []))
    return None


def shower_dq(node):
    """Sum of collected charge over one shower's members.

    The probe's own numbers when there is a sidecar, so the denominator is
    exactly what `E_est` decomposes -- see `_est_map`.
    """
    e = ((state.get("prep") or {}).get("showers") or {}).get(str(node))
    if e:
        return sum(m.get("dQ") or 0.0 for m in (e.get("members") or []))
    tot = 0.0
    for sid in members_of(node):
        q = seg_dq(sid)
        if q is not None:
            tot += q
    return tot


def dump_shower_id(sid):
    """The reconstruction's OWN shower assignment for a segment, from the dump.

    -1 means it put the segment in no shower at all.  Read from the dump rather
    than inferred from the probe on purpose: on evt169626 the dump says
    `shower_id -1` for 53070 by itself, so the "in no shower" half of the
    message stands on an event with no sidecar.  None = not in the dump.
    """
    for s in cur_segments():
        if s.get("id") == sid:
            return s.get("shower_id")
    return None


def absorb_reason(sid, node=None):
    """Why the reconstruction did what it did with one segment, from the probe.

    `absorb_site` answers "where did this end up", so it takes `recs[-1]`.
    This answers the opposite question -- why is it NOT in the shower being
    scanned -- and a segment can carry several records (13023 on evt169626 has
    a `walk_add` then a `direct`), so the record naming `node` is picked
    EXPLICITLY.  Last-wins would name the wrong shower on the second event that
    hits this: 21 of the 98 manifest events carry a `walk_exclude`, and 9 of
    those 23 segments were absorbed somewhere else afterwards.

    Returns the record dict, or None when the probe has nothing to say.
    """
    recs = ((state.get("prep") or {}).get("absorb") or {}).get(str(sid)) or []
    if not recs:
        return None
    if node is not None:
        here = [r for r in recs if r.get("shower") == node]
        if here:
            return here[-1]
    return recs[-1]


def conv_span():
    """(min, max) MeV per collected electron over THIS event's showers.

    The concrete form of the reason no MeV is quoted for a segment no shower
    owns: on evt169626 the eight showers run 1.23e-5 to 7.20e-5, a factor of
    5.8, so "convert it like the others" is not a defined instruction.  Shown
    rather than asserted -- the scanner can see the spread that blocks it.
    """
    vals = []
    for sh in cur_showers():
        nd = sh.get("id")
        e, q = sh.get("kine_charge"), shower_dq(nd)
        if e is not None and q:
            vals.append(e / q)
    return (min(vals), max(vals)) if vals else (None, None)


def orphan_energy(node, sid, as_em=False):
    """What a segment no shower owns is worth if the shower it was marked into
    owns it -- (MeV, MeV per electron), or (None, None) when it cannot be had.

    Round 18, and the owner's instruction: *"when we added a particle into an EM
    shower, it should then be treated as an EM shower to do the calculation. Of
    course, it is possible this information is not available, in this case, have
    a reasonable guess, it is fine for now."*

    The guess that invents least is the TARGET shower's own
    `kine_charge / SigmaDQ(members)`, applied to the segment's own charge.  That
    is exactly the ratio the probe's `E_est` decomposition uses for a member
    (see `_est_map`), so an orphan counted this way is treated identically to a
    segment the reconstruction HAD put in that shower -- including the round-9
    `as_em` rescale its caller applies, which is what "treated as an EM shower"
    means when the target is track-flagged.

    What it is NOT, said here so the number is never mistaken for a
    measurement: the per-shower factor is an AVERAGE over that shower's members.
    It is constant within a shower only because `E_est` is `kine_charge` split
    in proportion to `dQ` -- constant by construction, not because
    charge-to-energy is flat.  Recombination runs with dQ/dx, so a dense
    track-like segment does not convert like a sparse shower, and between
    showers the factor spans 1.22e-5 to 7.20e-5 MeV per electron on evt169626
    alone.  Both the panel and the record say the number is an estimate and
    carry the factor used, so a later reader can undo it.
    """
    q = seg_dq(sid)
    e = shower_energy(node)
    tot = shower_dq(node)
    if q is None or e is None or not tot:
        return None, None
    f = e / tot
    return q * f, f


def unknown_mark_row(node, sid, kind):
    """Everything either panel needs to explain ONE marked segment that carries
    no per-segment energy.

    Measured in one place so the EM mark list and the pi0 panel say the same
    thing: round 17's complaint was that the mark was made in EM mode and the
    objection only turned up in pi0 mode, with no way to connect them.
    """
    seg = None
    for s in cur_segments():
        if s.get("id") == sid:
            seg = s
            break
    q = seg_dq(sid)
    tot = shower_dq(node)
    _e, _f = orphan_energy(node, sid)
    return dict(seg=sid, kind=kind, in_dump=seg is not None,
                pdg=(seg or {}).get("particle_id"),
                score=(seg or {}).get("particle_score"),
                length=(seg or {}).get("length"),
                shower_id=(seg or {}).get("shower_id"),
                dq=q, shower_dq=tot,
                dq_frac=(q / tot) if (q is not None and tot) else None,
                est_energy=_e, est_factor=_f,
                reason=absorb_reason(sid, node))


def _seg_says(r):
    """"a 14.1 cm segment the reconstruction PID'd proton (score 100)" -- what
    the marked segment IS, in the reconstruction's own terms."""
    if not r.get("in_dump"):
        return "a segment that is not in the dump at all"
    bits = []
    if r.get("length") is not None:
        bits.append("a %.1f cm segment" % r["length"])
    else:
        bits.append("a segment")
    pdg = r.get("pdg")
    if pdg not in (None, 0):
        bits.append("the reconstruction PID'd %s" % PDG_SAYS.get(pdg, "pdg %s" % pdg))
        if r.get("score") is not None:
            bits[-1] += " (score %.0f)" % r["score"]
    elif pdg == 0:
        bits.append("with no PID")
    return " ".join(bits)


def _placed_says(r):
    """Where the reconstruction put it -- the half that needs no probe."""
    if not r.get("in_dump"):
        return ""
    if r.get("shower_id") == -1:
        return (" and put in <b>no shower</b> (the dump gives it "
                "<code>shower_id -1</code>)")
    return " that no shower reports as a member"


def unknown_mark_why(node, r, brief=False):
    """The reconstruction's own decision about a marked segment it owns nowhere,
    in one clause -- or "" when the probe has nothing on it.

    The point of saying it at all: the code did not overlook this segment, it
    considered it for this shower and refused.  A hand scan overruling one
    named decision is a very different act from a hand scan working around a
    hole in the display, and until this round the panel read like the latter.

    `brief` drops the probe log line and the source citation: the EM mark list
    is a list of marks, not a reading of one, and the pi0 panel carries the
    full form.  Two renderings of one fact, not two facts.
    """
    rec = r.get("reason") or {}
    how, site = rec.get("how"), rec.get("site") or "?"
    if how == "walk_exclude":
        if brief:
            return ("the code considered it for this very shower and refused "
                    "&mdash; the straight-long-track guard "
                    "(<code>PRShower.cxx:722-727</code>)")
        return ("the code <b>considered it for this very shower and refused"
                "</b> &mdash; <code>SHOWER_ABSORB EXCLUDE shower_start_seg=%s "
                "seg=%s pdg=%s</code> at <code>%s</code>, the straight-long-"
                "track guard, which declines a confidently PID'd non-electron "
                "(<code>PRShower.cxx:722-727</code>, F12 / doc pr/40 round 6)"
                % (rec.get("shower"), r["seg"], rec.get("pdg"), site))
    if how in ("walk_add", "direct"):
        if brief:
            return ("the probe last records it going into shower %s, yet no "
                    "shower reports it as a member" % rec.get("shower"))
        return ("the probe last records it going <b>into shower %s</b> "
                "(<code>%s</code> at <code>%s</code>), yet no shower reports it "
                "as a member &mdash; that pair of facts is worth a look before "
                "the mark is trusted" % (rec.get("shower"), how, site))
    if how:
        return "the probe records <code>%s</code> at <code>%s</code>" % (how, site)
    return ""


def unknown_mark_note(node, r, counted=False):
    """One line for the EM mark list: said where the mark is MADE.

    Deliberately shorter than the pi0 block -- this is a list of marks, not a
    reading of one -- so it carries the fact, the size and the guard's name and
    leaves the full citation to the pi0 panel.
    """
    why = unknown_mark_why(node, r, brief=True)
    _on = counted and r.get("est_energy") is not None
    if r.get("dq_frac") is None:
        body = "it cannot move the &pi;&#8304; mass"
    elif _on:
        body = ("&Sigma;dQ <b>%.3g e</b>, <b>%.0f%%</b> of shower %s's own "
                "charge, counted into it at an <b>estimated %.1f MeV</b>"
                % (r["dq"], 100 * r["dq_frac"], node, r["est_energy"]))
    else:
        body = ("&Sigma;dQ <b>%.3g e</b>, <b>%.0f%%</b> of shower %s's own "
                "charge, will not reach the &pi;&#8304; mass"
                % (r["dq"], 100 * r["dq_frac"], node))
    lead = ("Your mark IS recorded and IS counted" if _on else
            "Your mark IS recorded, with the reason, but there is no "
            "per-segment energy for it")
    return ("<span style='color:%s'>&#8627; <b>%s</b> &mdash; %s%s. %s: %s.%s"
            "</span>"
            % ("#2ca02c" if _on else "#b58900", r["seg"], _seg_says(r),
               _placed_says(r), lead, body,
               (" Why: %s." % why) if why else ""))


def unknown_mark_lines(slot, node, r, e_reco, counted=False):
    """The pi0 panel's block for one such segment: what, why, and what it is worth.

    Three separate spans rather than one paragraph because they answer three
    different questions and the scanner reads them at different moments.  The
    third one changes with the switch: round 17 refused to name a MeV at all,
    round 18 names one and says in the same breath what it is worth as evidence.
    """
    lo, hi = conv_span()
    out = []
    _est = r.get("est_energy")
    _on = counted and _est is not None
    out.append(
        "<span style='font-size:88%%;color:%s'>%s <b>&gamma;%d "
        "&middot; %s</b> &mdash; %s%s. %s%s</span>"
        % ("#2ca02c" if _on else "#d62728", "&#10003;" if _on else "&#9888;",
           slot, r["seg"], _seg_says(r), _placed_says(r),
           ("It is <b>%s</b> anyway, as part of &gamma;%d, at an "
            "<b>estimated %.1f MeV</b>"
            % ("counted in" if r["kind"] == "in" else "taken off", slot, _est))
           if _on else
           ("There is no per-segment energy for it, so it is <b>not %s</b> and "
            "&gamma;%d stays at <b>%s MeV</b>"
            % ("added" if r["kind"] == "in" else "taken off", slot,
               "-" if e_reco is None else "%.1f" % e_reco)),
           "." if r.get("dq_frac") is None else
           " &mdash; its charge, &Sigma;dQ <b>%.3g e</b>, is <b>%.0f%%</b>"
           " of &gamma;%d's own %.3g e." % (r["dq"], 100 * r["dq_frac"],
                                            slot, r["shower_dq"])))
    why = unknown_mark_why(node, r)
    if why:
        out.append("<span style='font-size:88%%;color:#b58900'>&nbsp;&nbsp;"
                   "<b>why</b>: %s. Your mark overrules that one decision."
                   "</span>" % why)
    pair = ""
    if r.get("pdg") == 2212:
        pair = (" The reco PID'd it a proton, whose own pair is %.2f&times;%.2f"
                " against a shower's %.2f&times;%.2f &mdash; this uses "
                "&gamma;%d's, which is what <i>%s</i> means."
                % (KINE_PROTON[0], KINE_PROTON[1], KINE_SHOWER[0],
                   KINE_SHOWER[1], slot, EORPH_IN))
    elif r.get("pdg") not in (None, 0, 11, -11):
        pair = (" The reco PID'd it a track, whose own pair is %.2f&times;%.2f"
                " against a shower's %.2f&times;%.2f &mdash; this uses "
                "&gamma;%d's, which is what <i>%s</i> means."
                % (KINE_TRACK[0], KINE_TRACK[1], KINE_SHOWER[0],
                   KINE_SHOWER[1], slot, EORPH_IN))
    if _on:
        out.append(
            "<span style='font-size:88%%;color:#555'>&nbsp;&nbsp;<b>and it is "
            "an estimate, not a measurement</b> &mdash; %.2e MeV per electron, "
            "&gamma;%d's own average over the segments the reconstruction did "
            "give it. That average is flat within a shower only because "
            "<code>E_est</code> is <code>kine_charge</code> split in proportion "
            "to dQ; recombination really runs with dQ/dx, and this event's own "
            "showers span %.2e to %.2e (a factor of %.1f).%s The record keeps "
            "the factor and the per-segment working, so it can be undone."
            "</span>"
            % (r["est_factor"], slot, lo, hi, hi / lo if lo else 0, pair))
    else:
        out.append(
            "<span style='font-size:88%%;color:#555'>&nbsp;&nbsp;<b>your mark "
            "is kept either way</b> &mdash; <code>marks_by_shower</code> and "
            "<code>marks_detail</code> record it with that reason. To count its "
            "charge into &gamma;%d, set <i>marked segments no shower owns</i> "
            "to <i>%s</i>%s.%s</span>"
            % (slot, EORPH_IN,
               "" if _est is None else " &mdash; an estimated %.1f MeV" % _est,
               pair))
    return out


def marks_energy(node, as_em=False, with_orphans=False):
    """What the scanner's marks add to, or take off, one shower's energy.

    Returns dict(delta, rows, unknown).  `rows` is one entry per mark that has
    an EFFECT; a member marked IN is already inside `kine_charge` and a
    non-member marked OUT was never in it, so both are dropped here rather than
    double-counted -- reachable in one gesture since round 10's whole-shower
    select, with `show members too` on by default.

    A marked segment's E_est was converted with its OWNING shower's
    recombination pair.  Under the EM hypothesis the same round-9 ratio is
    applied per segment, reusing `kine_hypothesis` rather than keeping a second
    copy of the constants.

    There is NO fallback for a segment no shower owns: E_est/dQ is not constant
    across showers (6.05e-5 against 4.99e-5 MeV per electron on evt409634
    alone), so a dQ-derived number would be a different quantity in the same
    units.  Those segments are named in `unknown` and never summed.

    Round 17 left that decision exactly as it was and added `unknown_rows`
    beside it: the same segments, with what the reconstruction itself decided
    about them and how much charge is at stake.

    Round 18 makes counting them possible, on the owner's instruction that a
    segment added into an EM shower should be *treated as* part of that shower.
    `with_orphans` is the switch and it is PASSED IN, never read off a widget
    here, for the round-16 reason: a stored pairing has to be re-priced under
    the settings it was stored with.  With it off the arithmetic is exactly the
    round-12 one, byte for byte.  See `orphan_energy` for what the estimate is
    and, more importantly, what it is not.
    """
    mem = set(members_of(node))
    est = _est_map()
    delta = 0.0
    rows, unknown, unknown_rows = [], [], []
    for sid, kind in sorted(marks_for(node).items()):
        if kind == "in" and sid in mem:
            continue          # already inside kine_charge
        if kind == "out" and sid not in mem:
            continue          # never was
        if kind not in ("in", "out"):
            continue          # "?" is a question, not an answer
        e, owner = est.get(sid, (None, None))
        if e is None:
            unknown.append(sid)
            _ur = unknown_mark_row(node, sid, kind)
            unknown_rows.append(_ur)
            if with_orphans and _ur.get("est_energy") is not None:
                # Treated exactly as if the TARGET shower owned it: the rescale
                # below is the same line an owned segment takes six lines down,
                # with `node` standing in for the owner the segment has not got.
                # So an orphan counted here and a member of that shower are the
                # same arithmetic, which is what "part of the shower" has to
                # mean if the number is to be comparable with the rest of E1.
                _lbl, _used, _a = kine_hypothesis(node)
                _conv = _ur["est_energy"]
                if as_em and _lbl is not None and _lbl != "shower" and _used:
                    _conv = _conv * (_used[0] * _used[1]) / (KINE_SHOWER[0]
                                                             * KINE_SHOWER[1])
                delta += (1.0 if kind == "in" else -1.0) * _conv
                rows.append(dict(seg=sid, kind=kind, E_est=_ur["est_energy"],
                                 energy=_conv, owner=None, owner_kine=_lbl,
                                 estimated=True, est_factor=_ur["est_factor"],
                                 est_from=node))
            continue
        lbl, used, _alt = (kine_hypothesis(owner) if owner is not None
                           else (None, None, None))
        conv = e
        if as_em and lbl is not None and lbl != "shower" and used:
            conv = e * (used[0] * used[1]) / (KINE_SHOWER[0] * KINE_SHOWER[1])
        sign = 1.0 if kind == "in" else -1.0
        delta += sign * conv
        rows.append(dict(seg=sid, kind=kind, E_est=e, energy=conv, owner=owner,
                         owner_kine=lbl))
    return dict(delta=delta, rows=rows, unknown=unknown,
                unknown_rows=unknown_rows)


def ehyp_widget(slot):
    return g1_ehyp if slot == 1 else g2_ehyp


def energy_for(node, as_em, with_marks, with_orphans=False):
    """One shower's gamma energy with both switches PASSED IN.

    Round 16.  A stored pairing has to be re-priced under the settings IT was
    stored with -- compare it against a live chain whose switches sit somewhere
    else and the difference reported is the switch, not the physics.
    `gamma_energy` delegates here so the two can never drift apart, which is the
    same reason round 11 gave for lifting `gamma_record` out of `on_save`.

    `kine_hypothesis(node)[2]` is the SAME collected charge re-converted under
    the other pair, and for anything the reco did not flag as a shower that
    other pair IS the shower pair -- so it is exactly the charge-inferred EM
    energy, with no second copy of the ratio to drift from.  Reused rather than
    recomputed for that reason.
    """
    if node is None:
        return None
    e = shower_energy(node)
    if e is None:
        return None
    if as_em:
        lbl, _used, alt = kine_hypothesis(node)
        # lbl is None when the start segment is not in the dump: "unknown", not
        # "track", so nothing is re-converted on that path.
        if not (lbl is None or lbl == "shower" or alt is None):
            e = alt
    # Round 12.  The membership delta is applied LAST and to the already-
    # converted number, because the two questions are independent: the
    # hypothesis is which recombination pair, the marks are which segments.
    # `with_orphans` defaults to False so that any caller not updated in round
    # 18 keeps the round-12 number exactly.
    if with_marks:
        e += marks_energy(node, as_em=as_em, with_orphans=with_orphans)["delta"]
    return e


def gamma_energy(slot):
    """The energy the pi0 arithmetic uses for one gamma slot, off the widgets."""
    return energy_for(state["gamma"].get(slot),
                      ehyp_widget(slot).value == EHYP_EM,
                      emark_mode.value == EMARK_MARKS,
                      orph_mode.value == EORPH_IN)


def poly_for(sid):
    for s in cur_segments():
        if s.get("id") == sid:
            return G.seg_points(s)
    return []


def push_polys(srcmap, seg_ids, src3=None):
    dat = {k: dict(xs=[], ys=[]) for k in ("xy", "yz", "xz")}
    polys = []
    for sid in seg_ids:
        pts = poly_for(sid)
        if len(pts) < 2:
            continue
        polys.append(pts)
        dat["xy"]["xs"].append([p[0] for p in pts])
        dat["xy"]["ys"].append([p[1] for p in pts])
        dat["yz"]["xs"].append([p[2] for p in pts])
        dat["yz"]["ys"].append([p[1] for p in pts])
        dat["xz"]["xs"].append([p[0] for p in pts])
        dat["xz"]["ys"].append([p[2] for p in pts])
    for k in ("xy", "yz", "xz"):
        srcmap[k].data = dat[k]
    if src3 is not None:
        fill3_lines(src3, polys)


# ---------------------------------------------------------------------------
# 3-D fills
# ---------------------------------------------------------------------------
# Python writes both the 3-D columns and a projection of them, so the first paint
# of an event is right before any drag.  The JS then owns every later frame --
# and because any push of cam_src.data re-runs em3d.JS_REDRAW in the browser, the
# server's fill can never be the version left on screen even if the two mirrors
# of the formula ever drifted.


def _proj(pts):
    az, el = state["cam"]
    return D3.project(pts, az, el, state["cam_c"])


def fill3_points(src, pts, **extra):
    """Numeric columns go over the wire as **numpy arrays**, not Python lists.

    Bokeh serialises an ndarray as a binary buffer and a list as JSON numbers, so
    the charge cloud goes over the wire several times smaller and parses far
    faster this way -- and it is the one layer big enough for that to be felt on
    an ssh tunnel, which is how this display is always used.  float32 halves it
    again: at a 500 cm coordinate that is ~6e-5 cm of rounding, four orders below
    the ~0.3 cm point spacing.
    """
    base_size, base_alpha, cue = _PT_CFG[src]
    az, el = state["cam"]
    r, up, fw = D3.camera_basis(az, el)
    c = state["cam_c"]
    n = len(pts)
    xyz = numpy.asarray(pts, dtype="float64").reshape(n, 3) if n else \
        numpy.zeros((0, 3))
    px = xyz[:, 0] - c[0]
    py = xyz[:, 1] - c[1]
    pz = xyz[:, 2] - c[2]
    u = px * r[0] + py * r[1] + pz * r[2]
    v = px * up[0] + py * up[1] + pz * up[2]
    if cue:
        R = state["cam_R"] or 1.0
        t = numpy.clip(0.5 + 0.5 * (px * fw[0] + py * fw[1] + pz * fw[2]) / R,
                       0.0, 1.0)
        al = base_alpha * (0.30 + 0.70 * t)
        sz = base_size * (0.70 + 0.60 * t)
    else:
        al = numpy.full(n, float(base_alpha))
        sz = numpy.full(n, float(base_size))
    f32 = (lambda a: numpy.asarray(a, dtype="float32"))
    d = dict(x=f32(xyz[:, 0]), y=f32(xyz[:, 1]), z=f32(xyz[:, 2]),
             u=f32(u), v=f32(v), al=f32(al), sz=f32(sz))
    for k, val in extra.items():
        d[k] = f32(val) if k in ("q", "cid20") else val
    src.data = d


def fill3_lines(src, polys, **extra):
    az, el = state["cam"]
    c = state["cam_c"]
    d = dict(xs3=[], ys3=[], zs3=[], xs=[], ys=[])
    for pts in polys:
        d["xs3"].append([p[0] for p in pts])
        d["ys3"].append([p[1] for p in pts])
        d["zs3"].append([p[2] for p in pts])
        uv = D3.project(pts, az, el, c)
        d["xs"].append([q[0] for q in uv])
        d["ys"].append([q[1] for q in uv])
    d.update(extra)
    src.data = d


def fill3_heads(src, tips, tails, cols):
    """Arrow heads.  The head angle is the PROJECTED direction, so it has no
    3-D analogue and has to be recomputed on every camera change -- which is why
    the tail travels alongside the tip in the CDS."""
    az, el = state["cam"]
    c = state["cam_c"]
    ut = _proj(tips) if tips else []
    u0 = _proj(tails) if tails else []
    ang = []
    for a, b in zip(ut, u0):
        ang.append(math.atan2(a[1] - b[1], a[0] - b[0]) - math.pi / 2.0)
    src.data = dict(
        x=[p[0] for p in tips], y=[p[1] for p in tips], z=[p[2] for p in tips],
        x0=[p[0] for p in tails], y0=[p[1] for p in tails],
        z0=[p[2] for p in tails],
        u=[p[0] for p in ut], v=[p[1] for p in ut], angle=ang, c=list(cols))


def reco_points():
    """Everything the RECONSTRUCTION put in the event -- fit points, vertices,
    shower points.  This, not the charge cloud, is what frames the view by
    default: a cosmic-laden cloud spans the whole TPC and would shrink the
    neutrino to a speck."""
    pts = [(p[0], p[1], p[2]) for s in cur_segments() for p in G.seg_points(s)]
    d = state["data"] or {}
    for v in (d.get("vertices") or []):
        fp = v.get("fit") or {}
        if fp.get("x") is not None:
            pts.append((fp["x"], fp["y"], fp["z"]))
    ts = d.get("track_shower") or {}
    for i, fl in enumerate(ts.get("flag_shower") or []):
        if fl:
            pts.append((ts["x"][i], ts["y"][i], ts["z"][i]))
    return pts


def match_points():
    """The anchor set for the cloud-cluster match: every fitted point of every
    segment plus EVERY track_shower point.

    Deliberately not `reco_points()`, and the difference is not cosmetic in
    either direction.  Vertices are excluded (a vertex is a fitted position, not
    charge, so it can sit in a gap and match the wrong cluster), and the
    `flag_shower` filter reco_points applies is dropped -- track-flagged points
    are charge of the candidate too, and the whole set is what the 94-event
    validation in the doc was measured on.  Changing this set invalidates that
    measurement, so it lives in its own function to make that visible."""
    pts = [(p[0], p[1], p[2]) for s in cur_segments() for p in G.seg_points(s)]
    ts = (state["data"] or {}).get("track_shower") or {}
    for i in range(len(ts.get("x") or [])):
        pts.append((ts["x"][i], ts["y"][i], ts["z"][i]))
    return pts


def focus_points():
    """The segments the scan is actually about: the selected shower in EM mode,
    both assigned gammas in pi0 mode.  Empty when nothing is picked, and the
    caller falls back to the whole reconstruction rather than framing nothing."""
    nodes = ([state["sel_shower"]] if mode_group.active == 0
             else [state["gamma"][1], state["gamma"][2]])
    want = set()
    for n in nodes:
        if n is not None:
            want.update(members_of(n))
            # Round 5: what you MARKED into the shower is part of what you are
            # judging, so the frame has to reach it.  evt64591's mark sits 84 cm
            # out against a 35 cm shower -- framing members only put the green
            # halo off-screen at the exact moment it was placed.
            want.update(marks_for(n))
    if not want:
        return []
    # One pass over the segments, not one poly_for() linear scan per member:
    # this is on the on_shower_select path now, i.e. it runs on every table click
    # when "frame the shower" is on.
    pts = []
    for s in cur_segments():
        if s.get("id") in want:
            pts += [(p[0], p[1], p[2]) for p in G.seg_points(s)]
    return pts


def refit_camera(push=True):
    src = reco_points()
    if fit_mode.active == 1 and state.get("cloud"):
        cl = state["cloud"]
        src = list(zip(cl["x"], cl["y"], cl["z"])) or src
    elif fit_mode.active == 2:
        # An EM shower is a few tens of cm inside an event that spans the TPC:
        # framing the whole reconstruction leaves it a smudge (measured: R over
        # all reco has a median of 162 cm, over the main cluster alone 38 cm).
        src = focus_points() or src
    c, R = D3.bounding_sphere(src)
    state["cam_c"], state["cam_R"] = c, R
    f3d.x_range.start, f3d.x_range.end = -R, R
    f3d.y_range.start, f3d.y_range.end = -R, R
    if push:
        push_camera()


def push_camera():
    """Assigning cam_src.data is the one signal the browser needs: it syncs the
    camera AND fires em3d.JS_APPLY, which reprojects every registered source."""
    az, el = state["cam"]
    c, R = state["cam_c"], state["cam_R"]
    cam_src.data = dict(az=[az], el=[el], cx=[c[0]], cy=[c[1]], cz=[c[2]],
                        R=[R], az0=[az], el0=[el],
                        xs0=[f3d.x_range.start], xe0=[f3d.x_range.end],
                        ys0=[f3d.y_range.start], ye0=[f3d.y_range.end])
    cam_div.text = ("<span style='font-size:85%%;color:#555'>camera az %.0f&deg; "
                    "el %.0f&deg; &nbsp; centre (%.0f, %.0f, %.0f) &nbsp; "
                    "R %.0f cm</span>"
                    % (math.degrees(az), math.degrees(el), c[0], c[1], c[2], R))


def draw_cloud():
    """The Bee charge cloud, from the local zip.  Absent zip => skeleton only,
    said out loud rather than left as an empty panel."""
    evt = (state["label"] or "")[3:]
    row = MANIFEST.get(evt, {})
    if cloud_layer.value == "(none)":
        state["cloud"] = None
        cloud_src.data = dict(x=[], y=[], z=[], q=[], cid20=[], u=[], v=[],
                              al=[], sz=[])
        cloud_div.text = ("<span style='font-size:85%;color:#555'>charge cloud "
                          "off</span>")
        return
    cl = D3.load_bee_cloud(SX, row, evt, layer=cloud_layer.value,
                           max_pts=int(cloud_max.value),
                           reco=match_points(),
                           candidate_only=(cloud_scope.active == 0))
    state["cloud"] = cl
    if not cl:
        cloud_src.data = dict(x=[], y=[], z=[], q=[], cid20=[], u=[], v=[],
                              al=[], sz=[])
        zp = D3.bee_zip_path(SX, row)
        cloud_div.text = (
            "<span style='color:#b58900;font-size:85%%'>no charge cloud for this "
            "event &mdash; <code>%s</code> is not on disk. The zips are "
            "gitignored; rebuild with <code>prep_em_scan.py --bee-build "
            "bee/em114</code>. The skeleton below is unaffected.</span>"
            % html.escape(os.path.relpath(zp, SX) if zp else "bee/&lt;round&gt;.zip"))
        return
    fill3_points(cloud_src, list(zip(cl["x"], cl["y"], cl["z"])),
                 q=cl["q"], cid20=cl["cid20"])
    warn = ""
    if cl["layer"] == "img-global":
        warn = ("<br><span style='color:#c00'><b>raw frame.</b> img-global is "
                "dumped pre-pipeline, before the T0/pos corrections the "
                "reconstruction works in (doc pr/13). It can sit up to ~121 cm "
                "off the skeleton drawn over it, per cluster.</span>")
    # THREE numbers, not two.  With the filter on there are two reductions in
    # play -- the candidate cut and then the decimation budget -- and a readout
    # that reported only "kept of total" would hide which one bit.
    if cl["filtered"]:
        scope = ("<br><b>%s of %s clusters</b> carry the reconstruction "
                 "(ids %s) &rarr; %s of %s points are the neutrino candidate"
                 % (cl["ncluster_kept"], cl["ncluster"],
                    ", ".join(str(i) for i in cl["kept_ids"]),
                    "{:,}".format(cl["candidate"]), "{:,}".format(cl["total"])))
    elif cl["fallback"]:
        scope = ("<br><span style='color:#b58900'>candidate filter asked for but "
                 "not applied: %s.</span>" % html.escape(cl["fallback"]))
    else:
        scope = ("<br>all %s clusters &mdash; cosmics included"
                 % cl["ncluster"])
    cloud_div.text = (
        "<span style='font-size:85%%;color:#555'><code>%s</code> &mdash; drawing "
        "<b>%s</b> point%s%s%s</span>%s"
        % (html.escape(cl["layer"]), "{:,}".format(cl["kept"]),
           "" if cl["kept"] == 1 else "s",
           (" of %s (every %d)"
            % ("{:,}".format(cl["candidate"]),
               max(1, cl["candidate"] // max(1, cl["kept"]))))
           if cl["kept"] < cl["candidate"] else "",
           scope, warn))


def sync_cloud_vis():
    on = "cloud" in {LAYER_KEYS[i] for i in layer_group.active}
    r_cloud_c.visible = on and cloud_color.active == 0
    r_cloud_q.visible = on and cloud_color.active == 1


def flip(view, a, b):
    """Bokeh 3.9 repaints a DataTable only through its CDSView change signal, and
    an all-rows filter of the same length compares equal so nothing fires.  Flip
    between two AllIndices objects to force it."""
    view.filter = b if view.filter is a else a


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def draw_arrows():
    """Shower axes.  One per shower in EM mode (the selected one highlighted), or
    one per assigned gamma in pi0 mode."""
    dat = {k: dict(xs=[], ys=[], c=[]) for k in ("xy", "yz", "xz")}
    hdat = {k: dict(x=[], y=[], angle=[], c=[]) for k in ("xy", "yz", "xz")}
    shafts, tips, tails, cols = [], [], [], []
    todo = []
    if mode_group.active == 0:
        if state["sel_shower"] is not None:
            todo = [(state["sel_shower"], "#ff7f0e", None)]
    else:
        for slot, col in ((1, "#1f77b4"), (2, "#d62728")):
            if state["gamma"][slot] is not None:
                todo.append((state["gamma"][slot], col, slot))
    for node, col, slot in todo:
        p0 = shower_start(node, slot)
        d, _, _ = shower_axis(node)
        if p0 is None or G.vmag(d) == 0:
            continue
        L = 25.0
        p1 = G.vadd(p0, G.vscale(d, L))
        shafts.append([tuple(p0), tuple(p1)])
        tips.append(tuple(p1))
        tails.append(tuple(p0))
        cols.append(col)
        for k, (a, b) in (("xy", (0, 1)), ("yz", (2, 1)), ("xz", (0, 2))):
            x0, y0, x1, y1 = p0[a], p0[b], p1[a], p1[b]
            if abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6:
                continue        # degenerate in THIS projection only
            dat[k]["xs"].append([x0, x1])
            dat[k]["ys"].append([y0, y1])
            dat[k]["c"].append(col)
            hdat[k]["x"].append(x1)
            hdat[k]["y"].append(y1)
            # Bokeh's triangle marker points at +y, so subtract a quarter turn.
            hdat[k]["angle"].append(math.atan2(y1 - y0, x1 - x0) - math.pi / 2.0)
            hdat[k]["c"].append(col)
    for k in ("xy", "yz", "xz"):
        arrow_src[k].data = dat[k]
        head_src[k].data = hdat[k]
    # The 3-D arrow keeps every axis the projections had to drop: a shaft that is
    # degenerate in one projection is skipped there but is still drawn here.
    fill3_lines(arrow3_src, shafts, c=list(cols))
    fill3_heads(head3_src, tips, tails, cols)


def draw_gammas():
    gx, gy, gz, gc, gt = [], [], [], [], []
    # Which segments went into each gamma slot, in the slot's own colour.  In pi0
    # mode the shower table's own selection is usually pointed somewhere else
    # (you are about to assign the OTHER gamma), so without this there is nothing
    # on screen saying what slot 1 actually holds.
    for slot, m2, m3 in ((1, g1mem_src, g1mem3_src), (2, g2mem_src, g2mem3_src)):
        node = state["gamma"][slot]
        push_polys(m2, members_of(node) if (node is not None
                                            and mode_group.active == 1) else [], m3)
    for slot, col in ((1, "#1f77b4"), (2, "#d62728")):
        node = state["gamma"][slot]
        if node is None:
            continue
        p = shower_start(node, slot)
        if p is None:
            continue
        gx.append(p[0]); gy.append(p[1]); gz.append(p[2])
        gc.append(col); gt.append("gamma %d start" % slot)
    gstart_src.data = dict(x=gx, y=gy, z=gz, c=gc, tag=gt)
    fill3_points(gstart3_src, list(zip(gx, gy, gz)), c=list(gc), tag=list(gt))

    # Round 8.  EM mode only: in pi0 mode the gamma diamonds already own this
    # corner of the screen and two overlapping "start" glyphs would be a puzzle.
    ex, ey, ez, ec, et = [], [], [], [], []
    dx_, dy_, dz_, dc_, dt_ = [], [], [], [], []
    node = state["sel_shower"]
    if mode_group.active == 0 and node is not None:
        used = shower_start(node)
        ov = state["em_start"].get(node)
        if used:
            ex.append(used[0]); ey.append(used[1]); ez.append(used[2])
            ec.append("#ff7f0e" if ov else "#8c564b")
            et.append("start in use%s" % (" (yours)" if ov else " (reco)"))
        if ov:
            # The reco's own start stays on screen alongside, greyed: without it
            # there is nothing to judge the move against.
            rs = reco_start(node)
            if rs:
                ex.append(rs[0]); ey.append(rs[1]); ez.append(rs[2])
                ec.append("#999999"); et.append("reco start (replaced)")
        dp = state["em_dir"].get(node)
        if dp:
            dx_.append(dp[0]); dy_.append(dp[1]); dz_.append(dp[2])
            dc_.append("#2ca02c"); dt_.append("axis aimed through here")
    emstart_src.data = dict(x=ex, y=ey, z=ez, c=ec, tag=et)
    fill3_points(emstart3_src, list(zip(ex, ey, ez)), c=list(ec), tag=list(et))
    emdir_src.data = dict(x=dx_, y=dy_, z=dz_, c=dc_, tag=dt_)
    fill3_points(emdir3_src, list(zip(dx_, dy_, dz_)), c=list(dc_), tag=list(dt_))
    v = pio_vertex()[0] if mode_group.active == 1 else None
    if v is None:
        piovtx_src.data = dict(EMPTY3)
        fill3_points(piovtx3_src, [], c=[], tag=[])
    else:
        piovtx_src.data = dict(x=[v[0]], y=[v[1]], z=[v[2]], c=["#e377c2"],
                               tag=["pi0 vertex"])
        fill3_points(piovtx3_src, [(v[0], v[1], v[2])],
                     c=["#e377c2"], tag=["pi0 vertex"])


def draw_tiers():
    """The three pass-1 tiers as steps in the acceptance plot.  The offset is the
    shower's own isochronous bonus, so the steps move when a shower's axis is
    near-perpendicular to the drift."""
    off = 0.0
    if state["sel_shower"] is not None:
        d, _, _ = shower_axis(state["sel_shower"])
        off = G.cone_angle_offset(d)
    tiers = [(25.0 + off, 80.0), (12.5 + off * 8.0 / 5.0, 130.0),
             (5.0 + off * 2.0, 200.0)]
    xs, ys = [], []
    for ang, dis in tiers:
        xs.append([0.0, dis, dis])
        ys.append([ang, ang, 0.0])
    tier_src.data = dict(xs=xs, ys=ys)


def refresh_marks():
    """Halos for the marks on the shower being scanned -- and only those.

    A mark now belongs to a shower, so drawing every mark in the event at once
    would put a green halo on a segment that is IN for a DIFFERENT shower, which
    is the exact confusion the per-shower keying exists to end.  `marks_div`
    carries the rest of the event's marks in words."""
    here = marks_for(state["sel_shower"])
    push_polys(in_src, [s for s, m in here.items() if m == "in"], in3_src)
    push_polys(out_src, [s for s, m in here.items() if m == "out"], out3_src)
    refresh_mark_list()
    refresh_colors()
    refresh_dim()


def refresh_colors():
    """Repaint segments in the colour of the shower they belong to NOW.

    Patches the `c` column only, for the reason `refresh_dim` patches `a` only:
    assigning `.data` re-serialises every polyline in four sources on every
    mark, which since round 4 is every tap.  No-op in per-segment colour mode,
    where a segment's hue has nothing to do with membership."""
    if seg_color_mode.active != 0:
        return
    own = owner_map()
    for m in (seg_src["xy"], seg_src["yz"], seg_src["xz"], seg3_src):
        ids = list(m.data.get("sid") or [])
        want = [shower_color(effective_owner(i, own)) for i in ids]
        if list(m.data.get("c") or []) == want:
            continue
        m.patch({"c": [(slice(0, len(want)), want)]})


def refresh_mark_list():
    rows = []
    for node in sorted(state["marks"], key=lambda n: state["shorder"].get(n, 0)):
        mk = state["marks"][node]
        if not mk:
            continue
        sel = " &larr; scanning" if node == state["sel_shower"] else ""
        bits = ", ".join(
            "<span style='color:%s'>%s %s</span>"
            % ({"in": "#2ca02c", "out": "#d62728"}.get(k, "#666"), s, k.upper())
            for s, k in sorted(mk.items()))
        rows.append("<span style='display:inline-block;width:22px;height:9px;"
                    "border:1px solid #555;background:%s'></span> <b>%s</b>: %s%s"
                    % (shower_color(node), node, bits, sel))
        # Round 17.  Said HERE, where the mark is made, and not only in the pi0
        # panel: the scanner marked 53070 into 53069 in EM mode on evt169626 and
        # met the objection two modes later, with nothing to connect the two.
        # Same source as the pi0 block (`marks_energy`), so they cannot drift.
        _oc = orph_mode.value == EORPH_IN
        for _u in marks_energy(node)["unknown_rows"][:3]:
            rows.append("&nbsp;&nbsp;" + unknown_mark_note(node, _u, _oc))
    # A segment IN two showers at once is a contradiction, not an opinion, and
    # the pass-1 numbers that decide it are already computed -- so show them
    # side by side rather than just flagging the clash.
    for sid, nodes in sorted(mark_conflicts().items()):
        cells = []
        for nd in sorted(nodes, key=lambda n: state["shorder"].get(n, 0)):
            m = seg_vs_shower(nd, sid)
            cells.append(
                "<b>%s</b>: %s cm, %s&deg;, tier <b>%s</b>, ellip %s"
                % (nd,
                   "-" if m["dist"] is None else "%.1f" % m["dist"],
                   "-" if m["angle"] is None else "%.1f" % m["angle"],
                   m["tier"] if m["tier"] else "-",
                   "-" if m["ellip"] is None else "%.2f" % m["ellip"]))
        rows.append(
            "<span style='color:#d62728'><b>&#9888; %s is marked IN against %d "
            "showers</b></span> &mdash; it can only belong to one. %s. "
            "<i>Lower ellip is the code's own tie-break "
            "(NeutrinoShowerClustering.cxx:1314-1315); unmark it on the other."
            "</i>" % (sid, len(nodes), " &nbsp;|&nbsp; ".join(cells)))
    marks_div.text = (
        "" if not rows else
        "<span style='font-size:88%;color:#333'><b>marks in this event</b>, by "
        "shower &mdash; each one is recorded against the shower named here.<br>"
        + "<br>".join(rows) + "</span>")


def refresh_dim():
    """Rewrite the per-segment alpha column, and ONLY that column.

    `source.patch()` rather than assigning `.data`: an assignment re-serialises
    every `xs`/`ys`/`xs3` polyline in the source and ships it to the browser, and
    this runs on every single mark -- which since round 4 is every tap.  Over the
    four segment sources that is ~7 400 coordinates per tap down an ssh tunnel,
    for a change to one list of floats.  The early return covers the whole
    default path: with the toggle off the column is a constant and there is
    nothing to send at all.
    """
    keep = None
    if dim_toggle.active and state["sel_shower"] is not None:
        keep = set(members_of(state["sel_shower"]))
        keep |= {s for s, m in marks_for(state["sel_shower"]).items()
                 if m in ("in", "out")}
    # Excluded showers fade harder than "not in this shower" and they fade
    # whatever the dim toggle says: the scanner asked for them to be out of the
    # way, not merely de-emphasised.  A segment kept by `keep` still yields to an
    # explicit exclusion -- naming a shower is the stronger statement.
    excl = excluded_segments()
    for m in (seg_src["xy"], seg_src["yz"], seg_src["xz"], seg3_src):
        ids = list(m.data.get("sid") or [])
        want = []
        for i in ids:
            if i in excl:
                want.append(0.05)
            elif keep is None or i in keep:
                want.append(0.95)
            else:
                want.append(0.16)
        if list(m.data.get("a") or []) == want:
            continue
        m.patch({"a": [(slice(0, len(want)), want)]})


def refresh_selection():
    """The cyan halo: which segments the next mark button will hit.

    Reads the SAME resolver the mark buttons read, so what is drawn and what
    would be marked cannot disagree -- the failure this exists to prevent is a
    box-select that silently caught a segment behind the one being aimed at."""
    ids = selected_cand_ids()
    push_polys(sel_src, ids, sel3_src)
    return ids


def refresh_impact():
    """What the marks would cost.  Segment-level charge is not in the dump, so the
    honest proxy is fitted-point charge, sum(dQ) over the segment's own points --
    stated as such rather than dressed up as an energy."""
    if state["sel_shower"] is None:
        impact.text = ""
        return
    mem = set(members_of(state["sel_shower"]))

    def sumdq(sids):
        t = 0.0
        for s in cur_segments():
            if s.get("id") in sids:
                for p in (s.get("points") or []):
                    if (p.get("dx") or 0) > 0 and (p.get("dQ") or -1) >= 0:
                        t += p["dQ"]
        return t
    here = marks_for(state["sel_shower"])
    out_m = {s for s, m in here.items() if m == "out" and s in mem}
    in_m = {s for s, m in here.items() if m == "in" and s not in mem}
    sh = shower_by_node(state["sel_shower"])
    e = (sh or {}).get("kine_charge") or 0.0
    impact.text = (
        "<b>impact of your marks on shower %s</b> &mdash; reco kine_charge "
        "<b>%.1f MeV</b> over %d segments.  You marked <span style='color:#d62728'>"
        "%d member(s) OUT</span> (&Sigma;dQ %.3g e) and <span style='color:#2ca02c'>"
        "%d non-member(s) IN</span> (&Sigma;dQ %.3g e). "
        "<i>&Sigma;dQ is fitted-point charge, not a calibrated energy &mdash; it is "
        "the size of the change, not its MeV value.</i>"
        % (state["sel_shower"], e, len(mem), len(out_m), sumdq(out_m),
           len(in_m), sumdq(in_m)))


# ---------------------------------------------------------------------------
# pi0
# ---------------------------------------------------------------------------


def gamma_record(slot):
    """Everything the record keeps about ONE gamma slot, or None if it is empty.

    Round 11 lifted this out of `on_save` unchanged, so that a stored pi0
    candidate and the record's top-level pairing are built by the same code.
    Two builders would drift, and the fields that would drift first are exactly
    the ones that must not: `energy_hypothesis` and `energy_as_reconstructed`
    are per PAIRING, not per event, so a candidate that re-converted a
    track-flagged gamma has to be distinguishable from one that did not.
    """
    node = state["gamma"][slot]
    if node is None:
        return None
    mk = marks_energy(node, as_em=ehyp_widget(slot).value == EHYP_EM,
                      with_orphans=orph_mode.value == EORPH_IN)
    return dict(
        shower=node,
        start=list(shower_start(node, slot) or []),
        start_override=list(state["gstart"][slot]) if state["gstart"][slot] else None,
        # Round 8b.  Both mass conventions below are computed from this
        # point and this axis; naming their provenance is what makes the
        # two numbers comparable months later.
        start_source=start_source(node, slot),
        reco_start=list(reco_start(node) or []),
        em_start_correction=(list(state["em_start"][node])
                             if state["em_start"].get(node) else None),
        dir_point=(list(state["em_dir"][node])
                   if state["em_dir"].get(node) else None),
        axis_source=shower_axis(node)[2],
        energy=gamma_energy(slot),
        # Round 9.  Which pair the energy above was converted with, and
        # the reco's own number, so a later reader can recompute either
        # without going back to the dump.
        energy_hypothesis=("as_em_shower"
                           if ehyp_widget(slot).value == EHYP_EM
                           else "as_reconstructed"),
        energy_as_reconstructed=shower_energy(node),
        # Round 12.  Which SEGMENTS the energy above was summed over, what the
        # scanner's marks moved it by, and the per-segment arithmetic -- so a
        # later reader can recompute either number without the probe sidecar.
        energy_includes_marks=(emark_mode.value == EMARK_MARKS),
        energy_marks_delta=mk["delta"],
        energy_marks_detail=mk["rows"],
        energy_marks_unknown=mk["unknown"],
        # Round 18.  Which of those unowned segments were counted, at what
        # factor, and what they contributed -- kept separately from the marks
        # delta above because the two are different kinds of number: one is the
        # probe's exact decomposition, the other is an estimate.  Rows in
        # `energy_marks_detail` carrying `estimated: true` are the estimated
        # ones; absent means exact, which is every row written before round 18.
        energy_includes_orphans=(orph_mode.value == EORPH_IN),
        energy_orphan_delta=sum(
            (1.0 if r["kind"] == "in" else -1.0) * r["energy"]
            for r in mk["rows"] if r.get("estimated")),
        energy_orphan_detail=[
            dict(seg=r["seg"], kind=r["kind"], energy=r["energy"],
                 est_factor=r.get("est_factor"), est_from=r.get("est_from"))
            for r in mk["rows"] if r.get("estimated")],
        # The energy BEFORE the membership delta, always -- so the pair
        # (energy, energy_without_marks) reads the same way round 9's
        # (energy, energy_as_reconstructed) does, whichever way the switch was.
        energy_without_marks=(gamma_energy(slot) - mk["delta"]
                              if emark_mode.value == EMARK_MARKS
                              else gamma_energy(slot)),
        # Which recombination that energy used, and the same charge under
        # the other hypothesis.  A gamma slot filled with a track-flagged
        # object -- which is the whole point of "reco PID wrong" -- has an
        # energy 1.66x smaller than the identical charge in a
        # shower-flagged one, and the mass goes as sqrt(E1 E2).
        particle_id=(shower_by_node(node) or {}).get("particle_id"),
        flag_shower=shower_is_em(node),
        kine_hypothesis=kine_hypothesis(node)[0],
        energy_other_hypothesis=kine_hypothesis(node)[2],
        members=sorted(members_of(node)),
        axis=list(shower_axis(node)[0]))


def pio_pairing():
    """The pi0 currently in the slots: both gammas, the vertex, both masses.

    The record's top-level pi0 block and every stored candidate are this same
    dict, so the two can never carry different fields.
    """
    v, how, detail = pio_vertex()
    gam = {}
    for slot in (1, 2):
        g = gamma_record(slot)
        if g is not None:
            gam[str(slot)] = g
    e1 = gamma_energy(1)
    e2 = gamma_energy(2)
    thA = mA = thB = mB = None
    if state["gamma"][1] is not None and state["gamma"][2] is not None:
        thA = G.angle_deg(shower_axis(state["gamma"][1])[0],
                          shower_axis(state["gamma"][2])[0])
        mA = G.pi0_mass(e1, e2, thA)
        p1 = shower_start(state["gamma"][1], 1)
        p2 = shower_start(state["gamma"][2], 2)
        if v and p1 and p2:
            thB = G.angle_deg(G.vsub(p1, v), G.vsub(p2, v))
            mB = G.pi0_mass(e1, e2, thB)
    return dict(
        gammas=gam, vertex=list(v) if v else None, vertex_how=how,
        backproject=detail if how == "backproject" else None,
        # Round 14.  Which rays the back-projection used.  At the top level as
        # well as inside `backproject`, because load_label has to read it for a
        # record whose vertex_how is something else -- and because "absent means
        # the reconstruction's own rays" is the rule that keeps every record
        # written before this round exactly where it was.
        backproject_geometry=("handscan" if bp_geom.value == BPG_SCAN
                              else "reco"),
        mass_axis_convention=mA, theta_axis_convention=thA,
        mass_vertex_convention=mB, theta_vertex_convention=thB)


def _pair_sig(c):
    """What makes two stored pairings the SAME pairing.

    Not the shower ids alone: the same two gammas under a different vertex
    convention, a different energy hypothesis or a moved start are a different
    mass, and storing both is exactly what this list is for.  Rounded, because
    the numbers are re-derived floats and an exact compare would let a
    re-stored identical pairing through on the last bit.
    """
    g = c.get("gammas") or {}
    g1, g2 = g.get("1") or {}, g.get("2") or {}
    rnd = lambda v: tuple(round(float(x), 4) for x in (v or []))
    # Round 16.  The membership switch belongs here: without it a pairing
    # re-priced by the scanner's own marks had the SAME signature as the one
    # stored before them, so `store this pairing` refused the corrected version
    # as a duplicate -- measured on evt409634, where the stored row says 41.03
    # MeV (axis) and today's marks give 78.80, and the store was declined with
    # "same gammas, vertex, starts and energy hypotheses".  Through bool() so an
    # absent key -- every record written before round 12 -- reads as off and
    # does not make an old row look different from an identical new one.
    mk = lambda gg: bool(gg.get("energy_includes_marks"))
    # Round 18, for exactly the same reason and read exactly the same way:
    # absent is off, so no pre-round-18 row looks different from an identical
    # new one, and a pairing re-priced by counting an unowned segment can be
    # stored beside the one that did not count it.
    orp = lambda gg: bool(gg.get("energy_includes_orphans"))
    # Round 14's geometry, for the same reason, and only where it can matter.
    # Absent means the reconstruction's own rays, exactly as load_label reads it.
    bpg = ((c.get("backproject_geometry") or "reco")
           if c.get("vertex_how") == "backproject" else None)
    return (g1.get("shower"), g2.get("shower"), c.get("vertex_how"),
            rnd(c.get("vertex")), g1.get("energy_hypothesis"),
            g2.get("energy_hypothesis"), rnd(g1.get("start")),
            rnd(g2.get("start")), mk(g1), mk(g2), bpg, orp(g1), orp(g2))


def _mev(v):
    return "-" if v is None else "%.1f" % v


def cand_drift(c):
    """What has moved under a stored pairing since it was stored.

    Round 16.  A stored candidate is frozen on purpose -- it is the scanner's
    curated judgement, and `on_pio_load` has always reported rather than
    absorbed a disagreement.  What the freeze lacked was an EXIT: no way to see
    from the table that a row is no longer today's answer, and (until the
    signature was fixed above) no way to store the corrected one.

    Compared here are the candidate's own INPUTS against today's, never the live
    widgets: each gamma is re-priced under the hypothesis and membership switch
    IT was stored with, so what comes back is the physics that changed.  Plain
    text, because these phrases go into a DataTable cell.
    """
    out = []
    g = c.get("gammas") or {}
    for s in ("1", "2"):
        gg = g.get(s) or {}
        node = gg.get("shower")
        if node is None:
            continue
        e_was = gg.get("energy")
        e_now = energy_for(node, gg.get("energy_hypothesis") == "as_em_shower",
                           bool(gg.get("energy_includes_marks")),
                           bool(gg.get("energy_includes_orphans")))
        if e_was is not None and e_now is not None and abs(e_now - e_was) > 0.05:
            out.append("E%s %.1f -> %.1f MeV" % (s, e_was, e_now))
        # The start and the axis are per SHOWER once EM mode has corrected them,
        # so a correction made after the row was stored moves it even though the
        # row itself pinned nothing.  A slot override IS pinned, and does not.
        if not gg.get("start_override"):
            p_was, p_now = gg.get("start"), shower_start(node)
            if p_was and p_now:
                dd = G.vmag(G.vsub(tuple(p_was), p_now))
                if dd > 0.05:
                    out.append("g%s start moved %.1f cm" % (s, dd))
        a_was = gg.get("axis")
        a_now = shower_axis(node)[0]
        if a_was and G.vmag(tuple(a_was)) > 0 and G.vmag(a_now) > 0:
            da = G.angle_deg(tuple(a_was), a_now)
            if da is not None and da > 0.5:
                out.append("g%s axis turned %.1f deg" % (s, da))
    # The one vertex convention that can move on its own: the reconstruction's.
    if c.get("vertex_how") == "main_vertex":
        mv = G.pt((state["data"] or {}).get("main_vertex"))
        v = c.get("vertex")
        if mv and v and G.vmag(G.vsub(tuple(v), mv)) > 0.05:
            out.append("the main vertex moved")
    return out


def cand_unused_marks(c):
    """Marks that exist on this pairing's gammas but are NOT in its energies.

    Round 16, and the owner's actual question: a row stored with the membership
    switch off is internally consistent -- re-priced under its own settings it
    has not moved -- and would never show up in `cand_drift`.  It is still not
    their best number, because the EM clustering fix they made by hand is
    sitting outside it.  Said separately from a drift for exactly that reason:
    one is "this row went out of date", the other is "this row never counted
    what you corrected".  Same arithmetic the live panel warns with (:2624).
    """
    out = []
    g = c.get("gammas") or {}
    for s in ("1", "2"):
        gg = g.get(s) or {}
        node = gg.get("shower")
        if node is None or gg.get("energy_includes_marks"):
            continue
        d = marks_energy(node,
                         as_em=gg.get("energy_hypothesis") == "as_em_shower",
                         with_orphans=bool(gg.get("energy_includes_orphans"))
                         )["delta"]
        if abs(d) > 0.05:
            out.append("g%s ignores your marks (%+.1f MeV)" % (s, d))
    return out


def cand_unused_orphans(c):
    """Charge this pairing's gammas were marked with but that it does not count.

    Round 18's sibling of `cand_unused_marks`, and separate from it for the same
    reason that one is separate from `cand_drift`: a row stored with the unowned
    switch off is self-consistent and will never drift, it simply predates the
    switch or was stored with it off.  Reported as an ESTIMATE, because that is
    what it is -- see `orphan_energy`.
    """
    out = []
    g = c.get("gammas") or {}
    for s in ("1", "2"):
        gg = g.get(s) or {}
        node = gg.get("shower")
        if node is None or gg.get("energy_includes_orphans"):
            continue
        if not gg.get("energy_includes_marks"):
            continue          # cand_unused_marks already says the bigger thing
        _as = gg.get("energy_hypothesis") == "as_em_shower"
        d = (marks_energy(node, as_em=_as, with_orphans=True)["delta"]
             - marks_energy(node, as_em=_as)["delta"])
        if abs(d) > 0.05:
            out.append("g%s leaves out an unowned segment (est %+.1f MeV)"
                       % (s, d))
    return out


def cand_convention_note(c):
    """A row whose vertex convention is not the one this event was last saved with.

    Round 16, and the owner's second example -- "update of the pi0 vertex
    choice".  Like `cand_unused_marks` this is not staleness: the row is
    self-consistent, it was simply stored under a convention the scan has since
    moved off.  Compared against the SAVED record rather than against the live
    radio, so the note does not flicker while rows are being looked at -- moving
    the radio is how you look, not how you decide.
    """
    saved = ((state.get("saved") or {}).get("pio")) or {}
    top, mine = saved.get("vertex_how"), c.get("vertex_how")
    if not top or not mine:
        return []
    if mine != top:
        return ["vertex is %s; this event was last saved as %s" % (mine, top)]
    if mine == "backproject":
        # Round 14's two readings.  Absent means the reconstruction's own rays,
        # exactly as load_label reads it.
        _g = lambda d: (d.get("backproject_geometry") or "reco")
        if _g(c) != _g(saved):
            return ["back-projected from the %s rays; the event was last saved "
                    "with %s" % (_g(c), _g(saved))]
    return []


def refresh_pio_cands():
    """The stored pairings table, and what the set of them says together."""
    rows = dict(n=[], g1=[], g2=[], e1=[], e2=[], th=[], mA=[], mB=[], how=[],
                flag=[])
    used = {}
    drifted = {}
    for i, c in enumerate(state["pio_cands"]):
        g = c.get("gammas") or {}
        g1, g2 = g.get("1") or {}, g.get("2") or {}
        n1, n2 = g1.get("shower"), g2.get("shower")
        rows["n"].append(i + 1)
        rows["g1"].append(n1 if n1 is not None else "-")
        rows["g2"].append(n2 if n2 is not None else "-")
        rows["e1"].append(g1.get("energy") or 0.0)
        rows["e2"].append(g2.get("energy") or 0.0)
        for k, src in (("th", "theta_axis_convention"),
                       ("mA", "mass_axis_convention"),
                       ("mB", "mass_vertex_convention")):
            v = c.get(src)
            rows[k].append(-1 if v is None else v)
        rows["how"].append(c.get("vertex_how") or "-")
        bits = []
        if G.pi0_mass_accepted(c.get("mass_axis_convention")):
            bits.append("axis mass in the code's window")
        elif G.pi0_mass_accepted(c.get("mass_vertex_convention")):
            bits.append("vertex mass in the code's window")
        for sl, gg in (("1", g1), ("2", g2)):
            if gg.get("energy_hypothesis") == "as_em_shower":
                bits.append("g%s re-converted as EM" % sl)
            if gg.get("energy_includes_marks"):
                bits.append("g%s counts your marks" % sl)
            if gg.get("energy_includes_orphans"):
                bits.append("g%s counts an unowned segment (est)" % sl)
        _dr, _um = cand_drift(c), cand_unused_marks(c)
        _cv = cand_convention_note(c)
        _uo = cand_unused_orphans(c)
        drifted[i + 1] = (_dr, _um, _cv, _uo)
        if _uo:
            bits.insert(0, "; ".join(_uo))
        if _cv:
            bits.insert(0, "; ".join(_cv))
        if _um:
            bits.insert(0, "; ".join(_um))
        if _dr:
            bits.insert(0, "NOT today's numbers: " + "; ".join(_dr))
        rows["flag"].append("; ".join(bits))
        for nd in (n1, n2):
            if nd is not None:
                used.setdefault(nd, []).append(i + 1)
    pio_cand_src.data = rows

    if not state["pio_cands"]:
        pio_cand_div.text = (
            "<span style='font-size:88%;color:#666'>No pairing stored yet. "
            "Assign both gamma slots, then <b>store this pairing</b> &mdash; "
            "repeat for a second \u03c0\u2070, or for another way of pairing the "
            "same gammas. Every stored pairing keeps its own vertex, starts and "
            "energy hypotheses.</span>")
        return
    bits = ["<b>%d pairing(s) stored</b> for this event."
            % len(state["pio_cands"])]
    # Round 16.  A stored row is frozen numbers; the inputs under it are not.
    # Every EM-mode start correction and every mark made AFTER a row was stored
    # leaves that row behind, and until this round the table showed the old
    # numbers with nothing to say so.  Measured on the owner's own tag when this
    # was written: 8 of 14 stored pairings no longer matched.
    _stale = [(k, d) for k, (d, _u, _c, _o) in sorted(drifted.items()) if d]
    _unm = [(k, u) for k, (_d, u, _c, _o) in sorted(drifted.items()) if u]
    _cnv = [(k, v) for k, (_d, _u, v, _o) in sorted(drifted.items()) if v]
    _uno = [(k, o) for k, (_d, _u, _c, o) in sorted(drifted.items()) if o]
    if _stale:
        bits.append(
            "<span style='color:#d62728'><b>&#9888; %d of them are no longer "
            "today's numbers</b> &mdash; %s. Each row is re-priced under the "
            "hypothesis and membership switch <i>it</i> was stored with, so "
            "what is listed is what actually moved, not which switch is up. "
            "Press <b>load into the slots</b> to see today's, then <b>update to "
            "today's numbers</b> to replace the row &mdash; what it said before "
            "is kept inside it, never dropped.</span>"
            % (len(_stale), "; ".join("<b>#%d</b> (%s)" % (k, ", ".join(v))
                                      for k, v in _stale)))
    if _unm:
        bits.append(
            "<span style='color:#b58900'><b>&#9888; %d of them do not count "
            "marks you have made</b> on their gammas &mdash; %s. That is not a "
            "row going stale: it was stored with <i>%s</i>, and re-priced under "
            "its own setting it has not moved. If the marks are the correction "
            "you meant, load the row, switch <i>gamma energy membership</i> to "
            "<i>%s</i>, and either <b>store this pairing</b> to keep both "
            "readings or <b>update to today's numbers</b> to replace this "
            "one.</span>"
            % (len(_unm), "; ".join("<b>#%d</b> (%s)" % (k, ", ".join(v))
                                    for k, v in _unm),
               EMARK_RECO, EMARK_MARKS))
    if _uno:
        bits.append(
            "<span style='color:#b58900'><b>&#9888; %d of them leave out a "
            "marked segment no shower owns</b> &mdash; %s. Those numbers are "
            "estimates (the target shower's own MeV-per-electron applied to the "
            "segment's charge), which is why they are named separately from the "
            "exact marks above. To take them, load the row and set <i>marked "
            "segments no shower owns</i> to <i>%s</i>.</span>"
            % (len(_uno), "; ".join("<b>#%d</b> (%s)" % (k, ", ".join(v))
                                    for k, v in _uno), EORPH_IN))
    if _cnv:
        bits.append(
            "<span style='color:#b58900'><b>&#9888; %d of them use a different "
            "vertex convention</b> from the one this event was last saved with "
            "&mdash; %s. Alternatives are exactly what this list is for, so "
            "nothing here is wrong; but if the convention moved because you "
            "changed your mind rather than to compare, load the row and "
            "<b>update to today's numbers</b>.</span>"
            % (len(_cnv), "; ".join("<b>#%d</b> (%s)" % (k, ", ".join(v))
                                    for k, v in _cnv)))
    # A shower in two candidates is NOT a contradiction the way a segment marked
    # into two showers is -- it is how alternatives are expressed.  But two real
    # pi0 in one event need four distinct gammas, so which reading applies has
    # to be visible rather than inferred from the ids.
    shared = {nd: ns for nd, ns in used.items() if len(ns) > 1}
    if shared:
        bits.append(
            "<span style='color:#b58900'>%s</span> &mdash; those are "
            "<i>alternative pairings</i> of the same gamma, not two separate "
            "\u03c0\u2070. Two real \u03c0\u2070 in one event need four distinct showers."
            % "; ".join("shower <b>%s</b> is in candidates %s"
                        % (nd, " and ".join(str(x) for x in ns))
                        for nd, ns in sorted(shared.items())))
    else:
        bits.append("<span style='color:#2ca02c'>No shower is used twice, so "
                    "these read as separate \u03c0\u2070.</span>")
    # The forgotten pairing: the slots hold something that is not in the list,
    # and Save writes it only to the top-level block.  Nothing is lost either
    # way -- this is said, not auto-corrected, because silently appending to a
    # list the scanner curated is the worse failure.
    if state["gamma"][1] is not None and state["gamma"][2] is not None:
        live = _pair_sig(pio_pairing())
        if all(_pair_sig(c) != live for c in state["pio_cands"]):
            bits.append(
                "<span style='color:#b58900'>The pairing now in the slots "
                "(&gamma;1 %s + &gamma;2 %s) is <b>not one of them</b> &mdash; "
                "press <i>store this pairing</i> to add it. Saving without that "
                "still records it, but as the record's single top-level pairing "
                "rather than as one of the list.</span>"
                % (state["gamma"][1], state["gamma"][2]))
    pio_cand_div.text = ("<span style='font-size:88%;color:#333'>"
                         + "<br>".join(bits) + "</span>")


def pio_vertex():
    """(point, how, detail).  `how` is what goes in the saved record."""
    d = state["data"] or {}
    if vtx_mode_group.active == 0:
        return G.pt(d.get("main_vertex")), "main_vertex", {}
    if vtx_mode_group.active == 2:
        # `manual` with the x/y/z boxes empty yields None, which silently drops
        # the whole vertex convention from the record -- evt166870 was saved that
        # way.  The verdict detail carries the reason so refresh_kine can say it.
        if state["vtx_manual"] is None:
            return None, "manual", {"verdict": "no point set -- type x/y/z, or "
                                    "set 'a tap in 3-D does' to "
                                    "'make it the pi0 vertex' and click one"}
        return state["vtx_manual"], "manual", {}
    n1, n2 = state["gamma"][1], state["gamma"][2]
    if n1 is None or n2 is None:
        return None, "backproject", {"verdict": "need two gammas"}
    sh1, sh2 = shower_by_node(n1), shower_by_node(n2)
    anchor = G.pt(d.get("main_vertex"))
    if anchor is None:
        return None, "backproject", {"verdict": "no main vertex to anchor from"}
    a1, as1 = gamma_scan_ray(n1, 1)
    a2, as2 = gamma_scan_ray(n2, 2)
    use_scan = bp_geom.value == BPG_SCAN
    RECO_SRC = "reco@closest_to_anchor"

    def _run(r1, r2, s1, s2, tag):
        o = G.pi0_backproject(sh1, sh2, cur_segments(), anchor,
                              ray1=r1, ray2=r2)
        o["geometry"] = tag
        o["ray1_source"], o["ray2_source"] = s1, s2
        return o

    if use_scan:
        bp = _run(a1, a2, as1, as2, "handscan")
    else:
        bp = _run(None, None, RECO_SRC, RECO_SRC, "reco")
    # Both geometries, whenever the scanner actually corrected something -- so
    # the panel can say how far the correction moved the vertex and the saved
    # record carries the number it was NOT using.  Skipped when nothing was
    # corrected, because then the two are the same run.
    if a1 is not None or a2 is not None:
        bp["alt"] = (_run(None, None, RECO_SRC, RECO_SRC, "reco") if use_scan
                     else _run(a1, a2, as1, as2, "handscan"))
    return bp["vertex"], "backproject", bp


def refresh_kine():
    if mode_group.active != 1:
        kine_div.text = ""
        return
    d = state["data"] or {}
    n1, n2 = state["gamma"][1], state["gamma"][2]
    v, how, detail = pio_vertex()
    rows = []

    rows.append("<b>your pi0</b>")
    if n1 is None or n2 is None:
        rows.append("<i>assign both gamma slots to compute a mass.</i>")
    else:
        e1, e2 = gamma_energy(1), gamma_energy(2)
        p1, p2 = shower_start(n1, 1), shower_start(n2, 2)
        d1, _, s1 = shower_axis(n1)
        d2, _, s2 = shower_axis(n2)
        # Convention A: the showers' own axes.
        thA = G.angle_deg(d1, d2)
        mA = G.pi0_mass(e1, e2, thA)
        # Convention B: anchored on the chosen vertex.
        thB = mB = None
        if v is not None and p1 is not None and p2 is not None:
            thB = G.angle_deg(G.vsub(p1, v), G.vsub(p2, v))
            mB = G.pi0_mass(e1, e2, thB)
        esum = (e1 or 0) + (e2 or 0)
        asym = abs((e1 or 0) - (e2 or 0)) / esum if esum else None
        # Round 8b.  BOTH masses below depend on these points and axes, and the
        # scanner asked outright "which one was used?".  Said on screen, per
        # gamma, rather than left to be inferred from a number that moved.
        _SRC_SAYS = {
            "gamma_slot_override": "start set here in pi0 mode",
            "em_start_correction": "<b style='color:#ff7f0e'>your corrected "
                                   "start from EM mode</b>",
            "reco": "the reconstruction's start"}
        prov = []
        for _sl, _nd, _pp in ((1, n1, p1), (2, n2, p2)):
            _ss = start_source(_nd, _sl)
            prov.append("&gamma;%d: %s (%.1f, %.1f, %.1f), axis <code>%s</code>"
                        % (_sl, _SRC_SAYS[_ss],
                           _pp[0], _pp[1], _pp[2],
                           shower_axis(_nd)[2])
                        if _pp else "&gamma;%d: no start" % _sl)
        rows.append("<span style='font-size:88%%;color:#444'>%s</span>"
                    % " &nbsp;|&nbsp; ".join(prov))
        # Round 9.  Both masses below share E1*E2, so a hypothesis switch moves
        # both -- say per gamma which pair its energy used and what the other
        # would give, rather than leaving a number that silently changed.
        for _sl, _nd in ((1, n1), (2, n2)):
            _lbl, _u, _alt = kine_hypothesis(_nd)
            _em = ehyp_widget(_sl).value == EHYP_EM
            if _lbl is None:
                rows.append("<span style='font-size:88%%;color:#888'>&gamma;%d: "
                            "start segment not in the dump, so which "
                            "recombination pair was used is unknown.</span>" % _sl)
            elif _lbl == "shower":
                rows.append("<span style='font-size:88%%;color:#444'>&gamma;%d: "
                            "already charge-inferred as a shower; the switch "
                            "changes nothing.</span>" % _sl)
            elif _em:
                rows.append("<span style='font-size:88%%;color:#ff7f0e'>"
                            "<b>&gamma;%d: re-converted as an EM shower</b> "
                            "&mdash; %.1f MeV, not the reco's %.1f MeV "
                            "(it called this a %s).</span>"
                            % (_sl, _alt, shower_energy(_nd), _lbl))
            else:
                rows.append("<span style='font-size:88%%;color:#b58900'>"
                            "&#9888; &gamma;%d: the reco called this a %s, so "
                            "its %.1f MeV is a %s's energy. As an EM shower the "
                            "same charge gives <b>%.1f MeV</b> &mdash; switch "
                            "<i>gamma %d energy</i> above.</span>"
                            % (_sl, _lbl, shower_energy(_nd), _lbl, _alt, _sl))
        # Round 12.  The marks and the energy are two different questions, and
        # until this round the answer to the first never reached the second: a
        # segment marked into a gamma changed the membership and not the mass.
        _oc = orph_mode.value == EORPH_IN
        _mk = {sl: marks_energy(nd, as_em=ehyp_widget(sl).value == EHYP_EM,
                                with_orphans=_oc)
               for sl, nd in ((1, n1), (2, n2))}
        _on = emark_mode.value == EMARK_MARKS
        for _sl, _nd in ((1, n1), (2, n2)):
            _m = _mk[_sl]
            if _m["rows"]:
                _adds = [r for r in _m["rows"] if r["kind"] == "in"]
                _dels = [r for r in _m["rows"] if r["kind"] == "out"]
                _who = sorted({r["owner"] for r in _adds
                               if r["owner"] is not None})
                rows.append(
                    "<span style='font-size:88%%;color:%s'>%s&gamma;%d: your "
                    "marks move this energy by <b>%+.1f MeV</b> "
                    "(%d segment(s) in%s, %d out) &mdash; %s</span>"
                    % ("#2ca02c" if _on else "#b58900",
                       "" if _on else "&#9888; ", _sl, _m["delta"], len(_adds),
                       (" from shower %s" % ", ".join(str(w) for w in _who))
                       if _who else "", len(_dels),
                       "counted in the energies below."  if _on else
                       "<b>not counted below</b>. Switch <i>gamma energy "
                       "membership</i> to <i>%s</i>." % EMARK_MARKS))
            # Round 17.  The old one-liner here said only "marked but owned
            # by no shower", which reads like a hole in the display -- the
            # scanner's report was that it "is somewhat strange".  It is not a
            # hole: on evt169626 the reconstruction considered 53070 for THIS
            # shower and refused it by a named, knob-gated guard, the dump
            # records `shower_id -1` for it on its own, and its charge is 20%
            # of the gamma's.  All three were already on disk and none was on
            # screen.  The energy behaviour is unchanged -- `delta` still does
            # not include these segments -- only what the panel says about it.
            for _u in _m["unknown_rows"][:3]:
                rows.extend(unknown_mark_lines(_sl, _nd, _u,
                                               shower_energy(_nd), _oc))
            if len(_m["unknown_rows"]) > 3:
                rows.append(
                    "<span style='font-size:88%%;color:#d62728'>&#9888; "
                    "&gamma;%d: %d more marked segment(s) in no shower "
                    "(%s) &mdash; same story for each.</span>"
                    % (_sl, len(_m["unknown_rows"]) - 3,
                       ", ".join(str(r["seg"])
                                 for r in _m["unknown_rows"][3:9])))
        rows.append(
            "E1 <b>%.1f</b> MeV (shower %s) &nbsp; E2 <b>%.1f</b> MeV (shower %s)"
            " &nbsp; E&pi;&#8304; <b>%.1f</b> MeV &nbsp; asym %s"
            % (e1 or 0, n1, e2 or 0, n2, esum,
               "%.2f" % asym if asym is not None else "-"))
        rows.append(
            "&nbsp;&nbsp;<b>axis convention</b> (shower axes, %s/%s): "
            "&theta; %s &rarr; m = <b>%s MeV</b>%s"
            % (s1, s2, "%.1f&deg;" % thA if thA is not None else "-",
               "%.1f" % mA if mA is not None else "-",
               "  <span style='color:#2ca02c'>[in the code's accept window]</span>"
               if G.pi0_mass_accepted(mA) else ""))
        rows.append(
            "&nbsp;&nbsp;<b>vertex convention</b> (%s): &theta; %s &rarr; "
            "m = <b>%s MeV</b>%s%s"
            % (how, "%.1f&deg;" % thB if thB is not None else "-",
               "%.1f" % mB if mB is not None else "-",
               "  <span style='color:#2ca02c'>[in the code's accept window]</span>"
               if G.pi0_mass_accepted(mB) else "",
               ("  <span style='color:#d62728'>&#9888; %s</span>"
                % detail["verdict"]) if v is None and detail.get("verdict")
               else ""))
        if not _on and (_mk[1]["rows"] or _mk[2]["rows"]):
            _e1m = (e1 or 0) + _mk[1]["delta"]
            _e2m = (e2 or 0) + _mk[2]["delta"]
            _mAm = G.pi0_mass(_e1m, _e2m, thA)
            _mBm = G.pi0_mass(_e1m, _e2m, thB) if thB is not None else None
            rows.append(
                "&nbsp;&nbsp;<span style='color:#b58900'>With your marks "
                "counted the energies are <b>%.1f</b> / <b>%.1f</b> MeV and the "
                "mass becomes <b>%s MeV</b> (axis)%s. The angles do not move "
                "&mdash; a mark changes which charge belongs to the shower, not "
                "where it points.</span>"
                % (_e1m, _e2m, _mev(_mAm),
                   "" if _mBm is None else " / <b>%s MeV</b> (vertex)"
                   % _mev(_mBm)))
        rows.append(
            "<i>The two conventions are shown side by side on purpose: the code "
            "itself uses different direction recipes for the mass it stores "
            "(:3771) and the angle it stores (:3830), and they do not close.</i>")
        # Which recombination each gamma's energy was converted with, and what
        # the mass becomes if a track-flagged gamma is really an EM shower.  The
        # scanner can re-label an object as a shower; the ENERGY does not follow,
        # because kine_charge was fixed upstream off get_flag_shower().
        promoted = {}
        for slot, n, e in ((1, n1, e1), (2, n2, e2)):
            lbl, used, a = kine_hypothesis(n)
            if lbl is None:
                continue
            sh = shower_by_node(n) or {}
            rows.append(
                "&nbsp;&nbsp;<span style='font-size:90%%'>gamma&nbsp;%d "
                "(shower %s, reco pdg <b>%s</b>) converted as <b>%s</b> "
                "(recom %.2f, fudge %.2f). As a <b>%s</b> the same charge gives "
                "<b>%.1f MeV</b>.</span>"
                % (slot, n, sh.get("particle_id"), lbl, used[0], used[1],
                   "shower" if lbl != "shower" else "track", a))
            # Only the TRACK-flagged ones move.  Flipping both gammas is a
            # non-statement: the mass goes as sqrt(E1 E2), so one going up by
            # 1.66 and the other down by 1.66 cancels exactly.  The question a
            # PID correction actually asks is "what if the one the reco called a
            # track is really a shower".
            promoted[slot] = a if lbl != "shower" else e
        if promoted and (e1 or 0) > 0 and (e2 or 0) > 0:
            a1, a2 = promoted.get(1, e1), promoted.get(2, e2)
            if (a1, a2) != (e1, e2):
                for conv, th in (("axis", thA), ("vertex", thB)):
                    if th is None:
                        continue
                    m = G.pi0_mass(a1, a2, th)
                    if m is None:
                        continue
                    rows.append(
                        "&nbsp;&nbsp;<span style='color:#b58900;font-size:90%%'>"
                        "if every track-flagged gamma here is really an EM "
                        "shower, the %s-convention mass becomes <b>%.1f MeV</b> "
                        "(E %.1f + %.1f).</span>" % (conv, m, a1, a2))
                rows.append(
                    "&nbsp;&nbsp;<i style='font-size:88%'>Same collected charge, "
                    "re-scaled by 1/(recom&times;fudge) &mdash; nothing is "
                    "re-measured. <b>Re-labelling an object does NOT move its "
                    "energy</b>: kine_charge was fixed upstream from "
                    "<code>get_flag_shower()</code>, which reads the START "
                    "segment's shower flags and |pdg|==11 and nothing else "
                    "(NeutrinoEnergyReco.cxx:188, PRShower.cxx:1460). The factor "
                    "is <b>1.66&times;</b>.</i>")

    if how == "backproject" and detail:
        rows.append("<b>back-projection</b> (mirror of id_pi0_without_vertex, "
                    "NeutrinoShowerClustering.cxx:4158-4256): branch "
                    "<b>%s</b>, verdict <b>%s</b>%s" % (
                        detail.get("branch") or "-", detail.get("verdict"),
                        "" if detail.get("verdict") == "ok" else
                        " &mdash; <span style='color:#d62728'>the code would "
                        "REFUSE this pair here</span>"))
        # Round 14.  Which rays built that vertex, per gamma, in the scanner's
        # own words -- and, when a correction exists, the vertex the other
        # reading gives and how far apart the two are.
        _scan = detail.get("geometry") == "handscan"
        rows.append(
            "&nbsp;&nbsp;rays: <b>%s</b> &mdash; &gamma;1 <code>%s</code>, "
            "&gamma;2 <code>%s</code>"
            % ("your corrected start / axis" if _scan
               else "the reconstruction's own",
               detail.get("ray1_source") or "-",
               detail.get("ray2_source") or "-"))
        _alt = detail.get("alt")
        if _alt is None:
            if _scan:
                rows.append(
                    "&nbsp;&nbsp;<i style='font-size:88%'>Neither gamma has a "
                    "corrected start or an aimed axis, so this is the "
                    "reconstruction's own back-projection either way.</i>")
        else:
            _v, _av = detail.get("vertex"), _alt.get("vertex")
            _dd = (G.vmag(G.vsub(_v, _av)) if _v and _av else None)
            rows.append(
                "&nbsp;&nbsp;<span style='color:%s'>the other reading "
                "(<b>%s</b>) puts it at %s, verdict <b>%s</b> &mdash; %s</span>"
                % ("#2ca02c" if _scan else "#b58900",
                   "the reconstruction's own rays" if _scan
                   else "your corrected start / axis",
                   ("(%.1f, %.1f, %.1f)" % tuple(_av)) if _av else "no vertex",
                   _alt.get("verdict"),
                   ("<b>%.1f cm away</b>" % _dd) if _dd is not None
                   else "not comparable"))
            if not _scan:
                rows.append(
                    "&nbsp;&nbsp;<span style='color:#b58900'>&#9888; you have "
                    "corrected this pair's geometry and the vertex above is "
                    "<b>not using it</b>. Switch <i>back-projection "
                    "geometry</i> to <i>%s</i>.</span>" % BPG_SCAN)
        if detail.get("short_rerayed") is False:
            rows.append(
                "&nbsp;&nbsp;<i style='font-size:88%'>The short gamma was NOT "
                "re-rayed: the code re-derives a short shower's direction "
                "because it does not trust the stub's own, and you have stated "
                "it. The branch's own arithmetic is unchanged.</i>")
        if detail.get("branch") == "one_short":
            rows.append(
                "&nbsp;&nbsp;<i>one gamma is under 15 cm, so the code takes its "
                "re-ray branch (:4203-4247): the short gamma is re-rayed from the "
                "provisional midpoint and the vertex is the closest point on the "
                "LONG gamma's ray, not the midpoint.</i>")
        if detail.get("gap") is not None:
            rows.append(
                "&nbsp;&nbsp;closest-approach gap <b>%.2f cm</b>; conversion "
                "distances %.1f / %.1f cm; back-angles %.1f&deg; / %.1f&deg; "
                "(gate is 25&deg;); shower lengths %.1f / %.1f cm (gate 15 cm)"
                % (detail["gap"], detail.get("dis1") or -1, detail.get("dis2") or -1,
                   detail.get("angle1") or -1, detail.get("angle2") or -1,
                   detail.get("len1") or -1, detail.get("len2") or -1))
            # Round 14, said out loud because it looks like a bug otherwise: the
            # 15 cm branch test reads `showers[].total_length`, a reconstruction
            # number.  Marking segments in or out of a gamma moves its ENERGY
            # (round 12) and never its length, so a stub built up by hand stays
            # on the one-short branch.
            rows.append(
                "&nbsp;&nbsp;<i style='font-size:88%'>The 15 cm test reads the "
                "reco\'s total_length. Your IN / OUT marks move a gamma\'s "
                "energy, not its length, so they cannot change which branch "
                "runs.</i>")

    # --- what the reconstruction itself said, kept in TWO separate blocks -----
    grp = G.pi0_groups(cur_showers())
    rows.append("<hr style='margin:4px 0'><b>what the reconstruction paired</b> "
                "(showers[].pio_id &mdash; the accepted groups)")
    if not grp:
        rows.append("&nbsp;&nbsp;<i>no pi0 group in this event.</i>")
    for pid, shl in sorted(grp.items()):
        rows.append("&nbsp;&nbsp;group %s: showers %s &nbsp; mass <b>%.1f MeV</b>"
                    % (pid, " + ".join(str(s.get("id")) for s in shl),
                       (shl[0].get("pio_mass") or -1)))
    k = d.get("kine") or {}
    if k.get("kine_pio_flag"):
        m = float(k.get("kine_pio_mass") or -1)
        e1k = float(k.get("kine_pio_energy_1") or -1)
        e2k = float(k.get("kine_pio_energy_2") or -1)
        ang = float(k.get("kine_pio_angle") or -1)
        implied = G.pi0_mass(e1k, e2k, ang)
        rows.append(
            "<b>kine_pio_* (a BDT feature, NOT the pairing)</b>: mass %.1f, "
            "E1 %.1f, E2 %.1f, angle %.1f&deg;" % (m, e1k, e2k, ang))
        closes = implied is not None and m > 0 and abs(implied - m) <= 0.05 * m
        rows.append(
            "&nbsp;&nbsp;2&radic;(E1E2)&middot;sin(&theta;/2) = <b>%s</b> vs the "
            "stored %.1f &mdash; %s"
            % ("%.1f" % implied if implied else "-", m,
               "closes within 5%" if closes
               else "<span style='color:#d62728'>does NOT close</span>"))
        rows.append(
            "&nbsp;&nbsp;<i>This block is filled by a separate highest-energy scan "
            "over ALL candidate pairs, accepted or not, so it can name a pair no "
            "reconstruction ever accepted. Do not read it as the pairing.</i>")
    kine_div.text = "<br>".join(rows)
    # Round 16.  The stored-pairings table carries a live "is this still today's
    # answer" flag, and the inputs it compares against move with every start
    # correction and every mark.  Refreshed from here rather than from each of
    # the fifteen handlers that already end in refresh_kine, so a new handler
    # cannot forget it.  Measured rather than assumed: refresh_kine including
    # the table is 0.24 ms on evt281485 (3 stored rows, the most on disk) and
    # 0.48 ms on evt76346, whose back-projection runs twice for round 14's
    # `alt`.  The flag functions touch no widget and do no I/O.
    refresh_pio_cands()


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def fill_shower_table():
    d = state["data"] or {}
    segs = d.get("segments") or []
    rows = dict(node=[], pdg=[], nseg=[], joined=[], E=[], kb=[], conn=[],
                pio=[], length=[], drift=[], flag=[], color=[])
    nloss = 0
    for sh in sorted(cur_showers(), key=lambda s: -(s.get("kine_charge") or 0)):
        j, n = G.join_completeness(sh, segs)
        pm = probe_members(sh.get("id"))
        note = []
        if j != n:
            nloss += 1
            if pm is not None and len(pm) == n:
                note.append("join lossy %d/%d, REPAIRED by probe" % (j, n))
            else:
                note.append("<b>join incomplete %d/%d</b>" % (j, n))
        if sh.get("start_connection_type") == 4:
            note.append("conn 4: dropped by the PF tree")
        ax, br, srcname = shower_axis(sh.get("id"))
        a = G.angle_deg(ax, G.DRIFT_DIR)
        if a is not None:
            a = min(a, 180.0 - a)
        rows["node"].append(sh.get("id"))
        rows["pdg"].append(sh.get("particle_id"))
        rows["nseg"].append(n)
        rows["joined"].append("%d%s" % (len(pm) if pm is not None else j,
                                        "" if pm is not None else "*"))
        rows["E"].append(sh.get("kine_charge") or 0.0)
        rows["kb"].append(sh.get("kine_best") or 0.0)
        rows["conn"].append(sh.get("start_connection_type"))
        rows["pio"].append(sh.get("pio_id"))
        rows["length"].append(sh.get("total_length") or 0.0)
        rows["drift"].append(a if a is not None else -1)
        rows["flag"].append("; ".join(note))
        rows["color"].append(shower_color(sh.get("id")))
    shower_src.data = rows
    flip(shower_view, shower_view_a, shower_view_b)
    return nloss


def fill_cand_table():
    node = state["sel_shower"]
    rows = dict(sid=[], cid=[], pdg=[], length=[], dist=[], angle=[], tier=[],
                metric=[], owner=[], site=[], mark=[])
    pts = dict(x=[], y=[], c=[], sid=[], pid=[], length=[], tier=[], owner=[],
               site=[], mark=[], mk=[], sz=[])
    if node is None:
        cand_src.data = rows
        cand_pt_src.data = pts
        flip(cand_view, cand_view_a, cand_view_b)
        refresh_cmp()
        return
    start = shower_start(node)
    ax, _, _ = shower_axis(node)
    off = G.cone_angle_offset(ax)
    mem = set(members_of(node))
    mk_here = marks_for(node)
    owner_of = owner_map()
    excl = excluded_segments() - mem
    for s in cur_segments():
        sid = s.get("id")
        if sid in mem and not show_all_toggle.active:
            continue
        if sid in excl:
            continue
        if start is None or G.vmag(ax) == 0:
            dist = angle = None
        else:
            dist, q = G.segment_closest_point(s, start)
            angle = G.angle_deg(ax, G.vsub(q, start)) if q is not None else None
            if angle is None and dist is not None and dist < 1e-6:
                # The shower's own seed segment CONTAINS the start point, so the
                # vector start->closest is zero and angle_deg returns None.  Left
                # unhandled it fell through the `angle is not None` guard below
                # and the seed -- a member, and the one every other member is
                # measured against -- was silently absent from the plot.
                angle = 0.0
        tier = G.cone_tier(angle, dist if dist is not None else 1e9, off)
        met = G.cone_metric(angle, dist) if dist is not None else None
        rows["sid"].append(sid)
        rows["cid"].append(s.get("cluster_id"))
        rows["pdg"].append(s.get("particle_id"))
        rows["length"].append(s.get("length") or 0.0)
        rows["dist"].append(dist if dist is not None else -1)
        rows["angle"].append(angle if angle is not None else -1)
        rows["tier"].append(tier if tier else "-")
        rows["metric"].append(met if met is not None else -1)
        rows["owner"].append(owner_of.get(sid, "-"))
        rows["site"].append(absorb_site(sid))
        rows["mark"].append(mk_here.get(sid, ""))
        if dist is not None and angle is not None:
            mk = mk_here.get(sid, "")
            col = {"in": "#2ca02c", "out": "#d62728"}.get(mk)
            if col is None:
                col = "#ff7f0e" if sid in mem else "#7f9fbf"
            pts["x"].append(dist); pts["y"].append(angle); pts["c"].append(col)
            pts["sid"].append(sid); pts["pid"].append(s.get("particle_id"))
            pts["length"].append(s.get("length") or 0.0)
            pts["tier"].append(tier if tier else "-")
            pts["owner"].append(owner_of.get(sid, "-"))
            pts["site"].append(absorb_site(sid))
            pts["mark"].append(mk or "-")
            pts["mk"].append("square" if sid in mem else "circle")
            pts["sz"].append(12 if sid in mem or mk else 9)
    cand_src.data = rows
    cand_pt_src.data = pts
    flip(cand_view, cand_view_a, cand_view_b)
    fit_acc_ranges(mem)
    refresh_cmp()


def fit_acc_ranges(mem):
    """Scale the acceptance plot to what is being compared, not to the gate.

    The gate box is 220 cm x 90 deg because pass-1's third tier reaches that far.
    A real shower's members occupy a corner of it, so the default view answered
    "where is the gate" while the scanner was asking "how does this piece compare
    with the ones already in".  Zoomed, the range covers the members and any
    marked segment with 30% headroom; the tier steps are still drawn and simply
    run off the edge, which is honest -- and anything outside is counted out loud
    rather than silently cropped.
    """
    d = cand_pt_src.data
    if not acc_zoom.active or not d.get("x"):
        acc.x_range.start, acc.x_range.end = 0, 220
        acc.y_range.start, acc.y_range.end = 0, 90
        state["acc_hidden"] = 0
        return
    keys = [i for i, s in enumerate(d["sid"])
            if s in mem or (d["mark"][i] not in ("", "-"))]
    if not keys:                      # nothing to anchor on: show the gate
        acc.x_range.start, acc.x_range.end = 0, 220
        acc.y_range.start, acc.y_range.end = 0, 90
        state["acc_hidden"] = 0
        return
    xh = max(40.0, min(220.0, max(d["x"][i] for i in keys) * 1.3))
    yh = max(15.0, min(90.0, max(d["y"][i] for i in keys) * 1.3))
    acc.x_range.start, acc.x_range.end = 0, xh
    acc.y_range.start, acc.y_range.end = 0, yh
    state["acc_hidden"] = sum(1 for i in range(len(d["x"]))
                              if d["x"][i] > xh or d["y"][i] > yh)


def refresh_cmp():
    """Is the piece I marked like the ones already in this shower?

    The comparison the scanner is actually making, written out instead of left to
    be eyeballed off a scatter -- and in a form that aggregates over events,
    which a plot does not.
    """
    node = state["sel_shower"]
    if node is None:
        cmp_div.text = ""
        return
    d = cand_pt_src.data
    mem = set(members_of(node))
    mi = [i for i, s in enumerate(d["sid"]) if s in mem]
    bits = []
    if mi:
        xs = [d["x"][i] for i in mi]
        ys = [d["y"][i] for i in mi]
        bits.append("<b>already in shower %s</b>: %d segment(s) plotted &mdash; "
                    "distance <b>%.1f&ndash;%.1f cm</b>, angle "
                    "<b>%.1f&ndash;%.1f&deg;</b>"
                    % (node, len(mi), min(xs), max(xs), min(ys), max(ys)))
    else:
        bits.append("<b>shower %s</b> has no member plotted (no start point or "
                    "no axis)." % node)
    mk_here = marks_for(node)
    for sid, kind in sorted(mk_here.items()):
        if sid not in d["sid"]:
            bits.append("&nbsp;&nbsp;<b>%s</b> marked <b>%s</b> &mdash; not on "
                        "the plot (no distance/angle to this shower)."
                        % (sid, kind.upper()))
            continue
        i = d["sid"].index(sid)
        x, y = d["x"][i], d["y"][i]
        col = {"in": "#2ca02c", "out": "#d62728"}.get(kind, "#666")
        rel = []
        if mi:
            xs = [d["x"][j] for j in mi]
            ys = [d["y"][j] for j in mi]
            rel.append("angle <b>%s</b> the member spread"
                       % ("inside" if min(ys) <= y <= max(ys) else "outside"))
            far = max(xs)
            if far > 0:
                rel.append("distance <b>%.1f&times;</b> the furthest member"
                           % (x / far))
        site = absorb_site(sid) or "nothing"
        bits.append(
            "&nbsp;&nbsp;<span style='color:%s'><b>%s marked %s</b></span> "
            "&mdash; %.1f cm, %.1f&deg;, pass-1 tier <b>%s</b>, absorbed by "
            "<b>%s</b>. %s"
            % (col, sid, kind.upper(), x, y, d["tier"][i], site,
               "; ".join(rel)))
    if state.get("acc_hidden"):
        bits.append("<i>%d segment(s) sit outside the zoomed range &mdash; turn "
                    "the zoom off to see them.</i>" % state["acc_hidden"])
    cmp_div.text = ("<span style='font-size:88%;color:#333'>"
                    + "<br>".join(bits) + "</span>")


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------


def draw_segments():
    """Fill the segment polylines in both panels.  Returns the 3-D pick cloud.

    Split out of `load` in round 5 so the colour mode can be changed without
    re-reading the event.  Colouring by SHOWER is the default: `seg_color(i)`
    keys on the enumeration index, so two segments of the same shower came out
    two unrelated hues and the display never said which pieces were already
    considered one object -- which is the first thing the scan has to know.
    """
    d = state["data"] or {}
    segs = d.get("segments") or []
    dat = {k: dict(xs=[], ys=[], c=[], a=[], sid=[], pid=[], cid=[], owner=[],
                   mark=[])
           for k in ("xy", "yz", "xz")}
    d3 = dict(polys=[], c=[], a=[], sid=[], pid=[], cid=[], owner=[], mark=[])
    pick = dict(pts=[], sid=[])
    owner_of = owner_map()
    by_shower = (seg_color_mode.active == 0)
    for i, s in enumerate(segs):
        pts = G.seg_points(s)
        if len(pts) < 2:
            continue
        sid = s.get("id")
        own = owner_of.get(sid)
        c = (shower_color(effective_owner(sid, owner_of)) if by_shower
             else seg_color(i))
        for k, (a, b) in (("xy", (0, 1)), ("yz", (2, 1)), ("xz", (0, 2))):
            dat[k]["xs"].append([p[a] for p in pts])
            dat[k]["ys"].append([p[b] for p in pts])
            dat[k]["c"].append(c)
            dat[k]["a"].append(0.95)
            dat[k]["sid"].append(sid)
            dat[k]["pid"].append(s.get("particle_id"))
            dat[k]["cid"].append(s.get("cluster_id"))
            dat[k]["owner"].append(own if own is not None else "-")
            dat[k]["mark"].append("")
        d3["polys"].append([tuple(p) for p in pts])
        d3["c"].append(c)
        d3["a"].append(0.95)
        d3["sid"].append(sid)
        d3["pid"].append(s.get("particle_id"))
        d3["cid"].append(s.get("cluster_id"))
        d3["owner"].append(own if own is not None else "-")
        d3["mark"].append("")
        for p in pts:
            pick["pts"].append(tuple(p))
            pick["sid"].append(sid)
    for k in ("xy", "yz", "xz"):
        seg_src[k].data = dat[k]
    fill3_lines(seg3_src, d3.pop("polys"), **d3)
    return pick


def on_seg_color_mode(attr, old, new):
    if not state.get("data"):
        return
    draw_segments()
    refresh_dim()          # draw_segments resets the alpha column to 0.95
    refresh_mark_list()
    push_camera()


def load(lbl):
    path = EVENTS.get(lbl)
    state["label"] = lbl
    state["sel_shower"] = None
    state["marks"] = {}
    state["excl"] = set()
    state["shorder"] = {}
    state["legacy_marks"] = None
    state["pio_verdict_legacy"] = None
    state["acc_hidden"] = 0
    state["gamma"] = {1: None, 2: None}
    state["gstart"] = {1: None, 2: None}
    state["pio_cands"] = []
    state["_suspend"] = True
    try:
        # Event-level, so it must not leak across events: the next event's marks
        # are different marks.  Round 16b: the default this returns to is now
        # ON.  load_label puts a saved record's own setting back afterwards --
        # including OFF, which is what keeps every record already on disk at the
        # mass it was saved with.
        emark_mode.value = EMARK_MARKS
        # Round 18.  Same reason, same shape: load_label puts a saved record's
        # own setting back afterwards -- including OFF, which is what keeps
        # every record written before this round at the mass it was saved with.
        orph_mode.value = EORPH_IN
        # Round 14.  Same reason emark_mode is reset here: a Select keeps its
        # value across an event switch, and a geometry chosen for the last
        # event's corrections has no meaning for this one.  load_label puts a
        # saved record's own choice back afterwards.
        bp_geom.value = BPG_SCAN
    finally:
        state["_suspend"] = False
    # Node ids are per-event: a leaked override would land on a DIFFERENT
    # shower in the next event and move its start with no sign on screen.
    state["em_start"] = {}
    state["em_dir"] = {}
    state["em_startvid"] = {}
    state["_axis_cache"] = {}
    state["vtx_manual"] = None
    state["dirty"] = False
    if not path:
        return
    with open(path) as fh:
        state["data"] = json.load(fh)
    evt = lbl[3:] if lbl.startswith("evt") else lbl
    pp = os.path.join(args.prepdir, "emprep-evt%s.json" % evt)
    state["prep"] = None
    if os.path.exists(pp):
        with open(pp) as fh:
            state["prep"] = json.load(fh)

    d = state["data"]
    # Palette order and the exclusion menu, both in the SHOWER TABLE's own order
    # (energy, descending) rather than the dump's.  The scanner reads the table
    # top-down, so the biggest showers -- the ones a pi0 pairing is made of --
    # get the most widely separated hues instead of whatever the dump happened to
    # list first.  Must precede draw_segments: shower_color reads shorder.
    _byE = sorted(d.get("showers") or [],
                  key=lambda s: -(s.get("kine_charge") or 0))
    state["shorder"] = {sh.get("id"): i for i, sh in enumerate(_byE)}
    state["_suspend"] = True
    try:
        excl_choice.options = [
            "%s  (%.1f MeV, %d seg)"
            % (sh.get("id"), sh.get("kine_charge") or 0.0,
               len(members_of(sh.get("id"))))
            for sh in _byE]
        excl_choice.value = []
    finally:
        state["_suspend"] = False

    xl, xh = DET_BOX["x"]; yl, yh = DET_BOX["y"]; zl, zh = DET_BOX["z"]
    det_src.data = dict(
        xs_xy=[[xl, xh, xh, xl, xl], [0, 0]], ys_xy=[[yl, yl, yh, yh, yl], [yl, yh]],
        xs_yz=[[zl, zh, zh, zl, zl], []], ys_yz=[[yl, yl, yh, yh, yl], []],
        xs_xz=[[xl, xh, xh, xl, xl], [0, 0]], ys_xz=[[zl, zl, zh, zh, zl], [zl, zh]])
    # In 3-D the volume is 12 edges and the cathode is a PLANE, not a line -- the
    # x=0 line the projections draw is that plane seen edge-on.
    box = []
    for a, b in ((yl, zl), (yh, zl), (yh, zh), (yl, zh)):
        box.append([(xl, a, b), (xh, a, b)])
    for x0 in (xl, xh):
        box.append([(x0, yl, zl), (x0, yh, zl), (x0, yh, zh), (x0, yl, zh),
                    (x0, yl, zl)])
    box.append([(0.0, yl, zl), (0.0, yh, zl), (0.0, yh, zh), (0.0, yl, zh),
                (0.0, yl, zl)])
    fill3_lines(det3_src, box)

    pick = draw_segments()

    ts = d.get("track_shower") or {}
    sx, sy, sz = [], [], []
    for i, fl in enumerate(ts.get("flag_shower") or []):
        if fl:
            sx.append(ts["x"][i]); sy.append(ts["y"][i]); sz.append(ts["z"][i])
    shwpt_src.data = dict(x=sx, y=sy, z=sz, c=["#8fbf8f"] * len(sx),
                          tag=[""] * len(sx))

    vx, vy, vz, vt = [], [], [], []
    for v in (d.get("vertices") or []):
        fp = v.get("fit") or {}
        if fp.get("x") is None:
            continue
        vx.append(fp["x"]); vy.append(fp["y"]); vz.append(fp["z"])
        vt.append(str(v.get("id")))
    vtx_src.data = dict(x=vx, y=vy, z=vz, c=["#7f7f7f"] * len(vx), tag=vt)
    mv = G.pt(d.get("main_vertex"))
    mainvtx_src.data = (dict(x=[mv[0]], y=[mv[1]], z=[mv[2]], c=["#1f77b4"],
                             tag=["main vertex"]) if mv else dict(EMPTY3))

    # 3-D siblings of the point layers, then the cloud, then the framing.  Order
    # matters: refit_camera may frame the cloud, so the cloud has to exist first.
    draw_cloud()
    refit_camera(push=False)
    fill3_points(shwpt3_src, list(zip(sx, sy, sz)),
                 c=["#8fbf8f"] * len(sx), tag=[""] * len(sx))
    fill3_points(vtx3_src, list(zip(vx, vy, vz)),
                 c=["#7f7f7f"] * len(vx), tag=list(vt))
    fill3_points(mainvtx3_src, [tuple(mv)] if mv else [],
                 c=["#1f77b4"] if mv else [], tag=["main vertex"] if mv else [])
    fill3_points(pick_src, pick["pts"], sid=pick["sid"])
    # Replacing .data does not clear .selected, and a stale index list would make
    # the next "mark IN" apply to a segment of the PREVIOUS event.  Under
    # _suspend because on_pick/on_vtx_pick fire on the clear itself.
    state["_suspend"] = True
    try:
        for _s in (pick_src, vtx3_src, mainvtx3_src, cand_src, cand_pt_src):
            _s.selected.indices = []
    finally:
        state["_suspend"] = False
    if state.get("cloud"):
        cl = state["cloud"]
        fill3_points(cloud_src, list(zip(cl["x"], cl["y"], cl["z"])),
                     q=cl["q"], cid20=cl["cid20"])

    # load_label FIRST, then draw.  It restores sel_shower / marks / gammas from
    # disk, and until round 4 it ran last: re-opening a labelled event put the
    # marks back into state and then drew nothing from them, so the halos were
    # blank on exactly the events that already had an answer.
    load_label(lbl)
    nloss = fill_shower_table()
    state["_suspend"] = True
    try:
        _rows = list(shower_src.data.get("node") or [])
        # CLEARED FIRST, unconditionally.  Round 10, found by the browser test.
        # `shower_src.selected` survives an event switch, and Bokeh syncs a
        # property only when it CHANGES: an unlabelled event therefore opened
        # with row k of the PREVIOUS event still highlighted while
        # state["sel_shower"] was None, and clicking that very row did nothing
        # at all -- the scanner had to click some other row first.  The Python
        # test cannot see this; it assigns indices directly and had cleared them.
        shower_src.selected.indices = []
        if state["sel_shower"] is not None and state["sel_shower"] in _rows:
            shower_src.selected.indices = [_rows.index(state["sel_shower"])]
    finally:
        state["_suspend"] = False
    fill_cand_table()
    refresh_bulk_options()
    draw_tiers()
    draw_arrows()
    draw_gammas()
    push_polys(mem_src,
               members_of(state["sel_shower"]) if state["sel_shower"] is not None
               else [], mem3_src)
    refresh_marks()
    refresh_selection()
    refresh_impact()
    refresh_kine()
    refresh_pio_cands()
    refresh_emstart()
    set_banner(nloss)
    refresh_info()
    sync_cloud_vis()
    push_camera()


def set_banner(nloss):
    d = state["data"] or {}
    evt = (state["label"] or "")[3:]
    row = MANIFEST.get(evt, {})
    bits = []
    url = row.get("bee_url") or ""
    rnd = row.get("bee_round") or ""
    if url:
        bits.append("<a href='%s' target='_blank' style='font-weight:bold'>"
                    "open in Bee &#8599;</a> <span style='color:#666'>(%s)</span>"
                    % (html.escape(url), html.escape(rnd)))
    elif rnd:
        # bee_round and bee_url answer different questions: the round names the
        # LOCAL zip the 3-D cloud is read from, the url needs a server-minted
        # UUID that only an upload produces.  Saying "no Bee link" flatly here
        # would read as "no 3-D for this event", which is exactly wrong.
        bits.append("<span style='color:#b58900'>Bee set built but not uploaded"
                    "</span> <span style='color:#666'>(%s)</span> &mdash; the 3-D "
                    "cloud below IS this set; only the external link is missing"
                    % html.escape(rnd))
    else:
        bits.append("<span style='color:#999'>no Bee set for this event &mdash; "
                    "build one with <code>prep_em_scan.py --bee-build bee/em114"
                    "</code>, upload it yourself, then re-run prep</span>")
    if state["prep"]:
        bits.append("<span style='color:#2ca02c'>probe sidecar loaded</span> "
                    "(membership and absorb-site are the code's own)")
    else:
        bits.append("<span style='color:#b58900'>no probe sidecar</span> &mdash; "
                    "membership is the dump join and the <i>absorbed by</i> "
                    "column is empty")
    if nloss:
        rep = state["prep"] is not None
        bits.append("<span style='color:%s'>%d shower(s) have an incomplete "
                    "segments[].shower_id join%s</span>"
                    % ("#2ca02c" if rep else "#d62728", nloss,
                       " &mdash; repaired from the probe" if rep else
                       " &mdash; NOT repairable without a probe sidecar"))
    if state.get("legacy_marks"):
        node, n = state["legacy_marks"]
        bits.append(
            "<span style='color:#d62728'><b>this label predates per-shower "
            "marks</b></span> &mdash; its %d mark(s) carried no shower of their "
            "own, so they are shown against shower <b>%s</b> (the one the record "
            "named). If that is not the shower you meant, re-mark and save; "
            "nothing has rewritten the file." % (n, node))
    banner.text = " &nbsp;|&nbsp; ".join(bits)
    note = (row.get("scan_note") or "").strip()
    scan_note_div.text = (
        "<div style='background:#fff8e1;border-left:4px solid #f0ad4e;"
        "padding:6px 10px;margin:2px 0'><b>what you asked to look at here:</b> "
        "%s <span style='color:#888'>&mdash; your note from the scan list, not "
        "part of the record; the box at the bottom is the one that gets saved."
        "</span></div>" % html.escape(note)) if note else ""


def refresh_scan_status():
    lbl = state["label"] or ""
    if not lbl:
        scan_status.text = ""
        return
    if os.path.exists(label_path(lbl)):
        rec = state["saved"] or {}
        when = rec.get("saved_utc")
        scan_status.text = (
            "<div style='background:#e8f5e9;border-left:4px solid #2e7d32;"
            "padding:6px 10px;margin:2px 0'><b style='color:#2e7d32'>"
            "&#10004; you have already scanned this event</b>"
            "<span style='color:#555'> &mdash; a saved result exists in tag "
            "<code>%s</code>%s.</span></div>"
            % (html.escape(SCAN_TAG),
               ", saved %s" % html.escape(str(when)) if when else ""))
    else:
        scan_status.text = (
            "<div style='background:#f5f5f5;border-left:4px solid #9e9e9e;"
            "padding:6px 10px;margin:2px 0'><b style='color:#555'>"
            "not scanned yet</b><span style='color:#777'> &mdash; no saved "
            "result for this event in tag <code>%s</code>.</span></div>"
            % html.escape(SCAN_TAG))


def refresh_info():
    refresh_scan_status()
    d = state["data"] or {}
    m = d.get("meta") or {}
    evt = (state["label"] or "")[3:]
    row = MANIFEST.get(evt, {})
    warn = ""
    if not write_allowed():
        warn = ("<br><b style='color:#c00'>refusing to write: tag '%s' already "
                "holds labels and was not passed explicitly (CLAUDE.md M13). "
                "Restart with --scan-tag %s to continue that scan, or a new tag "
                "to start a fresh one.</b>" % (html.escape(SCAN_TAG),
                                               html.escape(SCAN_TAG)))
    info.text = (
        "run %s subrun %s event %s &nbsp;|&nbsp; sample <b>%s</b> (%s) "
        "&nbsp;|&nbsp; tag <code>%s</code>, %d/%d events labelled%s%s"
        % (m.get("runNo"), m.get("subRunNo"), m.get("eventNo"),
           row.get("sample", "?"), row.get("origin", "?"),
           html.escape(SCAN_TAG), done_count(), len(LABELS),
           "  <b style='color:#b58900'>[unsaved]</b>" if state["dirty"] else "",
           warn))


def load_label(lbl):
    p = label_path(lbl)
    state["saved"] = None
    state["_suspend"] = True
    try:
        em_verdict.active = None
        conf_group.active = None
        event_flag_group.active = []
        g1_ehyp.value = EHYP_RECO
        g2_ehyp.value = EHYP_RECO
        note_in.value = ""
        # Round 15.  Three more widgets that carry PER-EVENT answers and were
        # keeping the previous event's, because nothing here or in load() ever
        # put them back.  The radio is the one the owner reported: 9 of the 15
        # pi0-bearing records on disk were saved as `main vertex`, and the
        # restore below had a branch for `manual` and one for `backproject` and
        # none for that -- so re-opening any of them showed whichever mode the
        # PREVIOUS event had left.  Measured before the fix: 27 of 54 re-opens
        # wrong (repro in doc pr/114 sec 23.1).
        vtx_mode_group.active = 0
        # The manual boxes leak with it.  state["vtx_manual"] IS cleared in
        # load(), so a leaked `manual` mode reads as "no point set" rather than
        # as a foreign point -- but the boxes still held the last event's
        # numbers, and one keystroke in ONE of them made _manual_point() build a
        # vertex out of two coordinates from another event.
        man_x.value = man_y.value = man_z.value = ""
        # And the EM-mode start boxes, which are worse: _anchor_for_snap() reads
        # them FIRST and only falls back to the shower's own start when they do
        # not parse, so "snap to nearest vertex" and "aim axis at nearest fit
        # point" silently anchored on the previous event's typed coordinates.
        em_sx.value = em_sy.value = em_sz.value = ""
        if not os.path.exists(p):
            return
        with open(p) as fh:
            rec = json.load(fh)
        state["saved"] = rec
        em = rec.get("em") or {}
        if em.get("shower") is not None:
            state["sel_shower"] = em["shower"]
            if em.get("verdict") in EM_VERDICTS:
                em_verdict.active = EM_VERDICTS.index(em["verdict"])
        # Round 8.  Keyed by shower, so re-opening an event restores every
        # shower's start and direction, not only the selected one's -- the
        # saved marks_detail was measured against all of them.
        for _k, _dst in (("start_override_by_shower", "em_start"),
                         ("dir_point_by_shower", "em_dir")):
            for _nd, _p in (em.get(_k) or {}).items():
                try:
                    state[_dst][int(_nd)] = (float(_p[0]), float(_p[1]),
                                             float(_p[2]))
                except (TypeError, ValueError, IndexError):
                    continue
        for _nd, _v in (em.get("start_override_vertex_id_by_shower") or {}).items():
            try:
                state["em_startvid"][int(_nd)] = _v
            except (TypeError, ValueError):
                continue
        # Round 5 writes marks_by_shower and nothing else.  A round-4 file has a
        # flat map plus one `em.shower`, and the only defensible reading of it is
        # "they all belong to that shower" -- which may be wrong, so the read is
        # accepted and then SAID OUT LOUD by set_banner rather than absorbed
        # silently.  Nothing rewrites the old file; the scanner does that by
        # re-marking if the attribution is not what they meant.
        mbs = em.get("marks_by_shower")
        if isinstance(mbs, dict):
            state["marks"] = {int(n): {int(k): v for k, v in (mk or {}).items()}
                              for n, mk in mbs.items()}
        elif em.get("marks"):
            flat = {int(k): v for k, v in em["marks"].items()}
            if em.get("shower") is not None:
                state["marks"] = {em["shower"]: flat}
                state["legacy_marks"] = (em["shower"], len(flat))
        event_flag_group.active = [
            EVENT_FLAG_KEYS.index(f) for f in (rec.get("event_flags") or [])
            if f in EVENT_FLAG_KEYS]
        pio = rec.get("pio") or {}
        for slot in (1, 2):
            g = (pio.get("gammas") or {}).get(str(slot))
            if g:
                state["gamma"][slot] = g.get("shower")
                if g.get("start_override"):
                    state["gstart"][slot] = tuple(g["start_override"])
                # Absent on every record saved before round 9, and absent MUST
                # mean "as reconstructed" -- that is the number those records
                # were saved with, and re-opening one has to show it.
                ehyp_widget(slot).value = (
                    EHYP_EM if g.get("energy_hypothesis") == "as_em_shower"
                    else EHYP_RECO)
        # Round 12, and absent MUST mean off: records saved before this control
        # existed carry marks on a gamma's shower, and re-opening one has to
        # show the mass it was saved with.  Round 16b made this TOTAL rather
        # than one-way: with the default now ON, a record saved with the switch
        # OFF has to turn it back off, or every such record would be silently
        # re-priced the moment it was re-opened.  Only when the record carries
        # gammas -- a record with no pi0 block has no mass to protect and takes
        # the new default.
        if pio.get("gammas"):
            emark_mode.value = (
                EMARK_MARKS
                if any((pio["gammas"].get(str(_s)) or {}).get(
                    "energy_includes_marks") for _s in (1, 2))
                else EMARK_RECO)
            # Round 18, and TOTAL for the round-16b reason: the default is ON,
            # so a record saved with it off -- which is every record written
            # before this round -- has to turn it back off or its mass would be
            # silently re-priced by an ESTIMATE the moment it was re-opened.
            orph_mode.value = (
                EORPH_IN
                if any((pio["gammas"].get(str(_s)) or {}).get(
                    "energy_includes_orphans") for _s in (1, 2))
                else EORPH_OUT)
        # Round 11.  Absent MUST mean an empty list: a record saved before the
        # list existed stored exactly one pairing, at the top level, and has to
        # re-open showing precisely that and no phantom candidate.
        _cands = pio.get("candidates")
        state["pio_cands"] = list(_cands) if isinstance(_cands, list) else []
        # Round 5d: the control is gone, but a verdict written by an older
        # build is a past judgement on a scientific record -- carried through so
        # re-saving the event cannot silently destroy it.
        if pio.get("verdict") in PIO_VERDICTS_LEGACY:
            state["pio_verdict_legacy"] = pio["verdict"]
        # Round 15.  TOTAL, the way on_pio_load's mapping has always been --
        # every `vertex_how` the record can carry maps to a button, and an
        # unknown one falls back to `main vertex` rather than to whatever was on
        # screen.  Absent (`pio` null, or no record at all) leaves the reset
        # above standing, which is the same answer.
        _how = pio.get("vertex_how")
        if _how is not None:
            vtx_mode_group.active = {"main_vertex": 0, "backproject": 1,
                                     "manual": 2}.get(_how, 0)
        # `manual` with a null vertex stays on `manual` with the boxes empty --
        # that is what such a record was saved as, and refresh_kine says "no
        # point set" for it.  Falling back to `main vertex` here would show a
        # vertex the record does not carry.
        if _how == "manual" and pio.get("vertex"):
            state["vtx_manual"] = tuple(pio["vertex"])
            # Round 13: the BOXES too, not just the state.  A re-opened manual
            # vertex lived only in state["vtx_manual"] while x/y/z read empty,
            # so the first edit to any ONE box made `_manual_point()` return
            # None -- silently wiping the vertex and saving `vertex: null`
            # under `vertex_how: "manual"`.  evt64591 was one keystroke from it.
            # Written the same way `set_pio_vertex` writes them, and under
            # load_label's own _suspend so on_manual does not fire here.
            man_x.value, man_y.value, man_z.value = (
                "%.1f" % pio["vertex"][0], "%.1f" % pio["vertex"][1],
                "%.1f" % pio["vertex"][2])
        # Round 14.  A record written before this round has no such key and its
        # back-projection was built from the reconstruction's rays; restoring
        # this build's default would silently re-price a vertex already judged.
        # Of the two backproject records on disk, evt169626 moves 6.19 cm under
        # the new default and evt347129 does not move at all -- so the rule is
        # not academic, and it is applied by what the record SAYS, never by
        # whether the number happens to differ.
        _bpg = pio.get("backproject_geometry")
        if _bpg in ("reco", "handscan"):
            bp_geom.value = BPG_RECO if _bpg == "reco" else BPG_SCAN
        elif pio.get("vertex_how") == "backproject":
            bp_geom.value = BPG_RECO
        if rec.get("confidence") in CONF:
            conf_group.active = CONF.index(rec["confidence"])
        note_in.value = rec.get("note") or ""
    finally:
        state["_suspend"] = False


def seg_vs_shower(node, sid):
    """One segment measured against one shower: the pass-1 gate's own inputs."""
    segs = {s.get("id"): s for s in cur_segments()}
    s = segs.get(sid)
    start = shower_start(node)
    ax, _, _ = shower_axis(node)
    if s is None or start is None or G.vmag(ax) == 0:
        return dict(dist=None, angle=None, tier=None, ellip=None)
    dist, q = G.segment_closest_point(s, start)
    angle = G.angle_deg(ax, G.vsub(q, start)) if q is not None else None
    if angle is None and dist is not None and dist < 1e-6:
        angle = 0.0
    return dict(dist=dist, angle=angle,
                tier=G.cone_tier(angle, dist if dist is not None else 1e9,
                                 G.cone_angle_offset(ax)),
                ellip=G.cone_metric(angle, dist) if dist is not None else None)


def mark_conflicts():
    """Segments marked IN against MORE THAN ONE shower.

    A segment belongs to one shower or to none, so this is a contradiction in
    the record rather than a difference of opinion -- and it is reachable
    without noticing: a round-4 label's migrated mark stays attached to the
    shower the old file named while you mark the same segment against the one
    you actually meant.  evt64591 landed in exactly that state.
    """
    who = {}
    for node, mk in state["marks"].items():
        for sid, kind in mk.items():
            if kind == "in":
                who.setdefault(sid, []).append(node)
    return {sid: nodes for sid, nodes in who.items() if len(nodes) > 1}


def mark_metrics(node):
    """Everything a later fit needs about one shower's marks, measured now.

    The point of the scan is to tune the clustering, and a tuner wants the
    numbers the gate is cut on -- distance, angle, pass-1 tier, the ellipsoidal
    rank -- for each marked segment, next to the spread of the segments the
    reconstruction already put in that shower.  Recomputing those later means
    re-deriving the axis and the start from the dump and hoping they still match
    the probe's; measuring at save time makes each label self-contained.
    """
    start = shower_start(node)
    ax, br, axsrc = shower_axis(node)
    off = G.cone_angle_offset(ax)
    mem = set(members_of(node))
    segs = {s.get("id"): s for s in cur_segments()}
    own = owner_map()

    def one(sid):
        # One implementation of the gate's inputs, shared with the conflict
        # readout, so the record and the screen cannot drift apart.
        m = dict(seg_vs_shower(node, sid))
        s = segs.get(sid)
        m.update(length=(s or {}).get("length"), pdg=(s or {}).get("particle_id"),
                 cluster_id=(s or {}).get("cluster_id"),
                 absorbed_by=absorb_site(sid) or None, owner=own.get(sid))
        return m

    memm = [one(s) for s in sorted(mem)]
    ds = [m["dist"] for m in memm if m["dist"] is not None]
    as_ = [m["angle"] for m in memm if m["angle"] is not None]
    return dict(
        axis=list(ax), axis_branch=br, axis_source=axsrc,
        angle_offset_deg=off,
        start=list(start) if start else None,
        members=sorted(mem),
        member_span=dict(n=len(memm),
                         dist_min=min(ds) if ds else None,
                         dist_max=max(ds) if ds else None,
                         angle_min=min(as_) if as_ else None,
                         angle_max=max(as_) if as_ else None),
        marked={str(sid): dict(kind=kind, **one(sid))
                for sid, kind in sorted(marks_for(node).items())})


def on_save():
    lbl = state.get("label")
    if not lbl or not state.get("data"):
        return
    if not write_allowed():
        save_note.text = ("<b style='color:#c00'>refusing to write into tag '%s' "
                          "(M13). Restart with --scan-tag %s.</b>"
                          % (html.escape(SCAN_TAG), html.escape(SCAN_TAG)))
        return
    d = state["data"]
    m = d.get("meta") or {}
    evt = lbl[3:] if lbl.startswith("evt") else lbl
    mrow = MANIFEST.get(evt, {})

    marks_all = marks_pruned()
    em_block = None
    if state["sel_shower"] is not None or marks_all:
        node = state["sel_shower"]
        sh = shower_by_node(node) or {}
        j, n = G.join_completeness(sh, cur_segments()) if sh else (0, 0)
        ax, br, axsrc = shower_axis(node) if node is not None else ((0, 0, 0), "", "")
        rax, rbr, raxsrc = (shower_axis(node, use_override=False)
                            if node is not None else ((0, 0, 0), "", ""))
        em_block = dict(
            shower=node,
            # Round 5.  Keyed by shower, and the ONLY mark field written -- a
            # derived flat copy alongside it could disagree with this one, and
            # the ambiguity of the flat form is the bug being fixed.
            marks_by_shower={str(nd): {str(k): v for k, v in sorted(mk.items())}
                             for nd, mk in sorted(marks_all.items())},
            marks_detail={str(nd): mark_metrics(nd)
                          for nd in sorted(marks_all)},
            verdict=EM_VERDICTS[em_verdict.active] if em_verdict.active is not None else None,
            # the reco's own answer, copied in so a later fit never re-reads the dump
            reco=dict(members=sorted(members_of(node)) if node is not None else [],
                      membership_source="probe" if probe_members(node) is not None
                      else "dump-join",
                      join_complete=(j == n), num_segments=n, joined=j,
                      kine_charge=sh.get("kine_charge"),
                      kine_best=sh.get("kine_best"),
                      kine_dQdx=sh.get("kine_dQdx"),
                      kine_range=sh.get("kine_range"),
                      # What the reco thought this was, and therefore WHICH
                      # recombination its kine_charge used.  A verdict of "is an
                      # EM shower (reco PID wrong)" is only checkable later if
                      # the record says what the reco called it at the time.
                      particle_id=sh.get("particle_id"),
                      flag_shower=shower_is_em(node),
                      kine_hypothesis=kine_hypothesis(node)[0],
                      kine_charge_other_hypothesis=kine_hypothesis(node)[2],
                      start_connection_type=sh.get("start_connection_type"),
                      pio_id=sh.get("pio_id"),
                      axis=list(rax), axis_branch=rbr, axis_source=raxsrc),
            # Round 8.  The start the gate was measured from, the reco's own,
            # and the scanner's overrides -- all three, because "I moved the
            # start" is only checkable later against what it was moved FROM.
            # axis_source above already says whether the axis is the probe's,
            # recomputed at the new start, or aimed by hand at a second point.
            # What the gate ACTUALLY used.  Kept out of the `reco` block above:
            # that block is the reconstruction's own answer, and filing a
            # hand-aimed axis inside it would let a later reader attribute the
            # scanner's judgement to the reconstruction.
            axis_used=list(ax), axis_used_branch=br, axis_used_source=axsrc,
            start_used=list(shower_start(node) or []) if node is not None else None,
            reco_start=list(reco_start(node) or []) if node is not None else None,
            reco_start_vertex_id=sh.get("start_vertex_id"),
            start_override_by_shower={
                str(nd): list(p) for nd, p in sorted(state["em_start"].items())},
            start_override_vertex_id_by_shower={
                str(nd): v for nd, v in sorted(state["em_startvid"].items())},
            dir_point_by_shower={
                str(nd): list(p) for nd, p in sorted(state["em_dir"].items())})

    pio_block = None
    # Round 11: a stored candidate list keeps the pi0 block alive even after the
    # slots are cleared.  Without this clause, adding two pairings and then
    # pressing `clear gammas` before Save would drop both on the floor.
    if (state["gamma"][1] is not None or state["gamma"][2] is not None
            or state["pio_cands"]):
        pio_block = dict(
            pio_pairing(),
            **({"verdict": state["pio_verdict_legacy"]}
               if state.get("pio_verdict_legacy") else {}),
            # Round 11.  ALWAYS written, empty list included: a reader must be
            # able to tell "this scanner stored no alternatives" from "this
            # record predates the list", and an absent key cannot say either.
            candidates=[copy.deepcopy(c) for c in state["pio_cands"]],
            reco_groups={str(pid): dict(showers=[s.get("id") for s in shl],
                                        mass=shl[0].get("pio_mass"))
                         for pid, shl in G.pi0_groups(cur_showers()).items()},
            reco_kine={k: v2 for k, v2 in (d.get("kine") or {}).items()
                       if k.startswith("kine_pio_")})

    rec = dict(
        event=lbl, runNo=m.get("runNo"), subRunNo=m.get("subRunNo"),
        eventNo=m.get("eventNo"),
        sample=mrow.get("sample"), origin=mrow.get("origin"),
        source=os.path.realpath(EVENTS[lbl]),
        arm=os.path.basename(os.path.dirname(os.path.dirname(
            os.path.realpath(EVENTS[lbl])))),
        probe_sidecar=(state["prep"] or {}).get("source_log"),
        bee_url=mrow.get("bee_url") or None,
        scan_tag=SCAN_TAG,
        saved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        confidence=CONF[conf_group.active] if conf_group.active is not None else None,
        note=note_in.value or None,
        # Round 8.  Event-level, so it sits beside `em` and `pio` rather than
        # inside either: a later pass selecting "the no-vertex NCpi0 events"
        # reads one key and never has to open a shower block.
        event_flags=[EVENT_FLAG_KEYS[i] for i in sorted(event_flag_group.active)],
        main_vertex=d.get("main_vertex"),
        # The view the judgement was made from, so a later re-read can put the
        # event back on screen the way it was seen.
        camera=dict(az_deg=round(math.degrees(state["cam"][0]), 2),
                    el_deg=round(math.degrees(state["cam"][1]), 2),
                    centre=[round(v, 2) for v in state["cam_c"]],
                    R=round(state["cam_R"], 2),
                    cloud=(state["cloud"] or {}).get("layer"),
                    cloud_kept=(state["cloud"] or {}).get("kept"),
                    cloud_total=(state["cloud"] or {}).get("total"),
                    # Which clusters were on screen when the call was made.  A
                    # verdict of "under-clustered" means something different if
                    # four fifths of the charge was filtered out of the view, so
                    # the filter state belongs in the record, not just in the UI.
                    cloud_scope=("neutrino-candidate"
                                 if (state["cloud"] or {}).get("filtered")
                                 else "all-clusters"),
                    cloud_candidate=(state["cloud"] or {}).get("candidate"),
                    cloud_cluster_ids=(state["cloud"] or {}).get("kept_ids")),
        em=em_block, pio=pio_block)

    # Upsert: merge onto whatever is already on disk for this event so scanning
    # EM now and pi0 later does not silently drop the earlier half.
    path = label_path(lbl)
    old = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                old = json.load(fh)
        except ValueError:
            old = {}
    for k in ("em", "pio"):
        if rec.get(k) is None and old.get(k) is not None:
            rec[k] = old[k]
    os.makedirs(labels_dir(), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)       # atomic: never leave a half-written record
    state["saved"] = rec
    state["dirty"] = False
    # Saved either way -- the record is the scanner's, not ours to veto -- but a
    # segment claimed by two showers is said out loud AT the save, which is the
    # moment it would otherwise become a quiet contradiction on disk.
    warn = ""
    for sid, nodes in sorted(mark_conflicts().items()):
        warn += ("<br><span style='color:#d62728'><b>&#9888; %s is marked IN "
                 "against showers %s.</b> It can belong to only one &mdash; see "
                 "<i>marks in this event</i> above, unmark it on the other and "
                 "save again.</span>"
                 % (sid, " and ".join(str(n) for n in sorted(nodes))))
    if state["pio_cands"]:
        warn += ("<br><span style='color:#333'>%d stored \u03c0\u2070 pairing(s) "
                 "written to <code>pio.candidates</code>%s.</span>"
                 % (len(state["pio_cands"]),
                    "" if not (state["gamma"][1] is not None
                               and state["gamma"][2] is not None
                               and all(_pair_sig(c) != _pair_sig(pio_pairing())
                                       for c in state["pio_cands"]))
                    else ", plus the unstored pairing in the slots as the "
                         "record's top-level <code>pio.gammas</code>"))
    save_note.text = ("saved <code>%s</code> at %s%s"
                      % (html.escape(os.path.relpath(path, SX)),
                         rec["saved_utc"], warn))
    refresh_info()


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------


def touch():
    state["dirty"] = True
    refresh_info()


def on_mode(attr, old, new):
    fill_cand_table()
    draw_arrows()
    draw_gammas()
    refresh_kine()
    refresh_emstart()        # round 8b: otherwise the start readout goes stale
    apply_layers(None, None, None)


def on_event(attr, old, new):
    load(new)


def step(delta):
    def cb():
        if not LABELS:
            return
        i = (LABELS.index(state["label"]) + delta) % len(LABELS)
        event_select.value = LABELS[i]
    return cb


def on_shower_select(attr, old, new):
    if state["_suspend"] or not new:
        return
    i = new[0]
    try:
        node = shower_src.data["node"][i]
    except (KeyError, IndexError):
        return
    state["sel_shower"] = node
    fill_cand_table()
    refresh_bulk_options()   # the scanned shower drops out of the bulk menu
    draw_tiers()
    draw_arrows()
    draw_gammas()            # round 8: the start / direction markers are per shower
    push_polys(mem_src, members_of(node), mem3_src)
    refresh_marks()          # the halos follow the shower now that marks do
    refresh_impact()
    refresh_emstart()
    # Round 5: a table click brings the 3-D view up and puts the shower in it.
    # The scanner asked for exactly this -- picking a row and then having to find
    # and re-frame the thing by hand was the step that made the table and the
    # display feel like two unrelated tools.
    view_tabs.active = 0
    # "frame the shower" is the one mode where picking a row is also a camera
    # move; the other two must NOT re-frame under the scanner on a table click.
    if fit_mode.active == 2:
        refit_camera()
    else:
        push_camera()


def _sel_ids(src):
    idx = src.selected.indices or []
    col = list(src.data.get("sid") or [])
    return [col[i] for i in idx if i < len(col)]


def sync_selection(origin):
    """Linked brushing: one click, the same segments lit everywhere.

    The candidate table, the acceptance plot and the 3-D pick cloud are three
    views of one list of segments, and until round 5 a click in one left the
    other two showing something else.  The origin is authoritative -- the other
    two are rewritten from it rather than unioned with it, so a stale selection
    left in a view the scanner is not looking at cannot leak into what the mark
    buttons act on.

    `pick_src` is written under `_suspend` because `on_pick` is the MARKING path:
    without it, mirroring a table click into the 3-D cloud while `a tap in 3-D
    does` is set to `mark IN` would apply a mark nobody asked for.
    """
    if state["_guard"]:
        return
    ids = set(_sel_ids(origin))
    state["_guard"] = True
    state["_suspend"] = True
    try:
        for src in (cand_src, cand_pt_src, pick_src):
            if src is origin:
                continue
            col = list(src.data.get("sid") or [])
            src.selected.indices = [i for i, s in enumerate(col) if s in ids]
    finally:
        state["_suspend"] = False
        state["_guard"] = False
    refresh_selection()
    if ids:
        own = owner_map()
        bits = ", ".join("%s (in shower %s)" % (s, own.get(s, "-"))
                         for s in sorted(ids)[:6])
        save_note.text = ("selected %d segment(s): %s%s"
                          % (len(ids), bits, " ..." if len(ids) > 6 else ""))


def _excl_node(opt):
    """Shower id out of an `excl_choice` option label ("83044  (298.1 MeV...)")."""
    try:
        return int(str(opt).split()[0])
    except (ValueError, IndexError):
        return None


def on_excl(attr, old, new):
    if state["_suspend"]:
        return
    state["excl"] = {n for n in (_excl_node(o) for o in new) if n is not None}
    refresh_dim()
    fill_cand_table()
    push_camera()


def selected_cand_ids():
    # Deduped across the three views.  Before round 5 they held independent
    # selections and a segment could realistically be in only one of them; now
    # sync_selection puts the SAME segment in all three, so without this the
    # cyan halo was pushed twice per selected segment and any count taken off
    # this list read double.
    out, seen = [], set()
    for src in (cand_src, cand_pt_src):
        for i in (src.selected.indices or []):
            try:
                sid = src.data["sid"][i]
            except (KeyError, IndexError):
                continue
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    # A 3-D tap or box lands on fitted POINTS, but the labelling unit is the
    # segment, so it resolves to segment ids -- which is what makes a box in a
    # rotated view unambiguous where a lasso over a flat projection is not.
    for i in (pick_src.selected.indices or []):
        try:
            sid = pick_src.data["sid"][i]
        except (KeyError, IndexError):
            continue
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def refresh_bulk_options():
    """The `whole shower` menu: every shower in the event except the one being
    scanned, in the shower table's own order (energy, descending) so the table,
    the dim menu and this one all read the same way.

    The scanned shower is left out because every one of its segments is already
    a member: selecting it would offer a bulk `mark IN` that changes nothing and
    still writes one entry per segment into the saved record.
    """
    sel = state["sel_shower"]
    opts = []
    for sh in sorted(cur_showers(),
                     key=lambda s: state["shorder"].get(s.get("id"), 0)):
        nid = sh.get("id")
        if nid is None or nid == sel:
            continue
        opts.append("%s  (%.1f MeV, %d seg)"
                    % (nid, sh.get("kine_charge") or 0.0, len(members_of(nid))))
    keep = bulk_shower.value if bulk_shower.value in opts else (
        opts[0] if opts else "")
    state["_suspend"] = True
    try:
        bulk_shower.options = opts or [""]
        bulk_shower.value = keep
    finally:
        state["_suspend"] = False


def select_segments(ids, extend=False):
    """Put a list of SEGMENT IDS into the selection the mark buttons read.

    Written into all three views directly rather than into one and left to
    `sync_selection`, because sync's fan-out is keyed on the origin view's own
    `sid` column and the ids here need not appear in any single view: a dimmed
    shower is dropped from the candidate table but kept in `pick_src`, which
    `draw_segments` fills from every segment unfiltered.  Selecting through the
    table would silently return the segments that happened to be listed.

    Cleared before it is set.  Bokeh fires `selected.indices` only when the
    value CHANGES, so pressing the button twice for the same shower -- which is
    what a scanner does after an accidental click somewhere else -- would leave
    the browser showing the stale halo.  Same lesson as `on_tap_action`.

    `_guard` keeps `sync_selection` from re-entering through the on_change
    handlers; `_suspend` keeps `on_pick`, which since round 4 is a MARKING path,
    from applying a mark nobody asked for when `a tap in 3-D does` is set to one.

    Returns what is actually selected, resolved by the same function the mark
    buttons call -- never the list that was asked for.
    """
    want = list(dict.fromkeys(ids))
    if extend:
        want = list(dict.fromkeys(list(selected_cand_ids()) + want))
    wset = set(want)
    state["_guard"] = True
    state["_suspend"] = True
    try:
        for src in (cand_src, cand_pt_src, pick_src):
            col = list(src.data.get("sid") or [])
            src.selected.indices = []
            src.selected.indices = [i for i, s in enumerate(col) if s in wset]
    finally:
        state["_suspend"] = False
        state["_guard"] = False
    return refresh_selection()


def on_bulk(extend):
    def cb():
        node = _excl_node(bulk_shower.value)
        if node is None:
            save_note.text = ("<span style='color:#c00'>pick a shower in "
                              "<i>whole shower</i> first.</span>")
            return
        if state["sel_shower"] is None:
            save_note.text = ("<span style='color:#c00'>pick the shower you are "
                              "scanning in the table above first &mdash; this "
                              "selects segments <i>to be marked against</i> it, "
                              "so it needs one.</span>")
            return
        want = list(dict.fromkeys(members_of(node)))
        if not want:
            save_note.text = ("<span style='color:#c00'>shower %s has no "
                              "segments.</span>" % node)
            return
        got = set(select_segments(want, extend=extend))
        here = [s for s in want if s in got]
        lost = [s for s in want if s not in got]
        # Which of them the CANDIDATE TABLE lists is a different question from
        # which are selected, and both get answered.  A dimmed-away shower keeps
        # its fitted points in `pick_src`, so the selection and the mark are
        # complete while the table shows nothing -- and the table is where the
        # scanner looks to check, so silence there would read as a lost mark.
        listed = set(cand_src.data.get("sid") or [])
        unlisted = [s for s in here if s not in listed]
        msg = ("selected <b>%d of %d</b> segment(s) of shower %s%s &mdash; press "
               "<b>mark IN</b> to put them into shower %s, or <b>mark OUT</b> to "
               "rule them out of it."
               % (len(here), len(want), node,
                  " (kept what was already selected)" if extend else "",
                  state["sel_shower"]))
        if lost:
            msg += ("<br><span style='color:#c00'>%s not selectable &mdash; "
                    "fewer than two fitted points, so %s drawn in no view and a "
                    "mark cannot reach %s.</span>"
                    % (", ".join(str(s) for s in lost[:8]),
                       "it is" if len(lost) == 1 else "they are",
                       "it" if len(lost) == 1 else "them"))
        if unlisted:
            msg += ("<br><span style='color:#b58900'>%d of them %s not listed in "
                    "the candidate table (dimmed away, or hidden by <i>show "
                    "members too</i>) &mdash; %s still selected and a mark will "
                    "reach %s.</span>"
                    % (len(unlisted), "is" if len(unlisted) == 1 else "are",
                       "it is" if len(unlisted) == 1 else "they are",
                       "it" if len(unlisted) == 1 else "them"))
        msg += offframe_hint(here)
        save_note.text = msg
    return cb


def mark(kind):
    def cb():
        ids = selected_cand_ids()
        if not ids:
            save_note.text = ("<span style='color:#c00'>select one or more rows "
                              "(table, acceptance plot, or a tap/box in 3-D) "
                              "first.</span>")
            return
        if not apply_marks(ids, kind):
            return          # apply_marks has already said why
        save_note.text = ("marked %d segment(s) against shower %s &mdash; %s%s"
                          % (len(ids), state["sel_shower"],
                             ", ".join(str(s) for s in ids[:12])
                             + (" ..." if len(ids) > 12 else ""),
                             offframe_hint(ids)))
    return cb


def on_gamma(slot):
    def cb():
        if state["sel_shower"] is None:
            save_note.text = ("<span style='color:#c00'>select a shower in the "
                              "table first.</span>")
            return
        state["gamma"][slot] = state["sel_shower"]
        state["gstart"][slot] = None
        draw_arrows()
        draw_gammas()
        refresh_kine()
        refresh_pio_cands()
        if fit_mode.active == 2:
            refit_camera()
        touch()
    return cb


def on_pio_add():
    if state["gamma"][1] is None or state["gamma"][2] is None:
        save_note.text = ("<span style='color:#c00'>assign <b>both</b> gamma "
                          "slots first &mdash; a stored pairing is a mass, and "
                          "a mass needs two gammas.</span>")
        return
    cand = pio_pairing()
    cand["stored_utc"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")
    sig = _pair_sig(cand)
    for i, c in enumerate(state["pio_cands"]):
        if _pair_sig(c) == sig:
            save_note.text = (
                "<span style='color:#b58900'>that pairing is already stored as "
                "candidate <b>%d</b> &mdash; same gammas, vertex, starts and "
                "energy hypotheses, so it would be the same mass twice. Change "
                "one of them, or leave it.</span>" % (i + 1))
            return
    state["pio_cands"].append(cand)
    refresh_pio_cands()
    touch()
    save_note.text = (
        "stored as candidate <b>%d</b>: &gamma;1 shower %s + &gamma;2 shower "
        "%s &rarr; m = <b>%s</b> MeV (axis) / <b>%s</b> MeV (vertex, %s). "
        "%d pairing(s) now stored for this event."
        % (len(state["pio_cands"]), state["gamma"][1], state["gamma"][2],
           _mev(cand.get("mass_axis_convention")),
           _mev(cand.get("mass_vertex_convention")), cand.get("vertex_how"),
           len(state["pio_cands"])))


def on_pio_load():
    """Put a stored pairing back in the slots so it can be looked at again.

    Restores the slot-scoped `gstart`, never the per-shower `em_start`:
    `shower_start` consults gstart FIRST, and gstart is scoped to one slot, so
    loading candidate 2 cannot move candidate 1's geometry.  The stored record
    is never rewritten by a load -- what comes back on screen is recomputed
    live, and if it disagrees with what was stored the difference is reported
    rather than absorbed.
    """
    idx = list(pio_cand_src.selected.indices or [])
    if not idx:
        save_note.text = ("<span style='color:#c00'>pick a row in the stored "
                          "pairings table first.</span>")
        return
    i = idx[0]
    if i >= len(state["pio_cands"]):
        return
    c = state["pio_cands"][i]
    g = c.get("gammas") or {}
    state["_suspend"] = True
    try:
        for slot in (1, 2):
            gg = g.get(str(slot)) or {}
            node = gg.get("shower")
            state["gamma"][slot] = node
            # Pin the stored start ONLY where the live chain would not already
            # produce it.  Setting gstart unconditionally would pin every
            # candidate's start to a slot override, and `start_source` would
            # then report `gamma_slot_override` for a start that came from the
            # reconstruction -- a provenance the record would carry forever.
            state["gstart"][slot] = None
            st = gg.get("start")
            if st and node is not None:
                livep = shower_start(node, slot)
                if livep is None or G.vmag(G.vsub(tuple(st), livep)) > 1e-6:
                    state["gstart"][slot] = tuple(st)
            ehyp_widget(slot).value = (
                EHYP_EM if gg.get("energy_hypothesis") == "as_em_shower"
                else EHYP_RECO)
        # Round 16.  The candidate's own membership switch, the same way round
        # 9's hypothesis just above and round 14's geometry just below.  Absent
        # MUST mean off -- that is what every record written before round 12 was
        # saved with.  Without it a stored row was put on screen under whichever
        # switch the EVENT had been restored with, so the energies displayed
        # could differ from the ones in the very row that had just been loaded.
        emark_mode.value = (
            EMARK_MARKS
            if any((g.get(str(_s)) or {}).get("energy_includes_marks")
                   for _s in (1, 2))
            else EMARK_RECO)
        # Round 18, read the same way and for the same reason: without it a row
        # stored counting an unowned segment would come back priced by whatever
        # the EVENT had been restored with.
        orph_mode.value = (
            EORPH_IN
            if any((g.get(str(_s)) or {}).get("energy_includes_orphans")
                   for _s in (1, 2))
            else EORPH_OUT)
        how = c.get("vertex_how")
        vtx_mode_group.active = {"main_vertex": 0, "backproject": 1,
                                 "manual": 2}.get(how, 0)
        # Round 14: a stored pairing is frozen numbers, and the geometry its
        # vertex was built from is one of them.
        _cbpg = c.get("backproject_geometry")
        if _cbpg in ("reco", "handscan"):
            bp_geom.value = BPG_RECO if _cbpg == "reco" else BPG_SCAN
        elif how == "backproject":
            bp_geom.value = BPG_RECO
        if how == "manual" and c.get("vertex"):
            state["vtx_manual"] = tuple(c["vertex"])
            man_x.value, man_y.value, man_z.value = (
                "%.1f" % c["vertex"][0], "%.1f" % c["vertex"][1],
                "%.1f" % c["vertex"][2])
    finally:
        state["_suspend"] = False
    draw_arrows()
    draw_gammas()
    refresh_kine()
    refresh_pio_cands()
    if fit_mode.active == 2:
        refit_camera()
    else:
        push_camera()
    touch()
    live = pio_pairing()
    drift = []
    for k, lab in (("mass_axis_convention", "axis-convention mass"),
                   ("mass_vertex_convention", "vertex-convention mass")):
        a, b = c.get(k), live.get(k)
        if a is None and b is None:
            continue
        if a is None or b is None or abs(a - b) > 0.05:
            drift.append("%s %s &rarr; %s MeV" % (lab, _mev(a), _mev(b)))
    msg = ("loaded candidate <b>%d</b> into the slots: &gamma;1 %s + &gamma;2 "
           "%s, vertex <i>%s</i>."
           % (i + 1, state["gamma"][1], state["gamma"][2], c.get("vertex_how")))
    if drift:
        # The one way this happens: shower_axis reads the per-shower em_start,
        # which gstart cannot pin, so an EM-mode start correction made AFTER the
        # candidate was stored changes the axis-convention angle.  The stored
        # number stands; the screen shows today's.
        msg += ("<br><span style='color:#d62728'>&#9888; what is on screen is "
                "<b>not</b> what was stored &mdash; %s. The stored candidate is "
                "unchanged; something it was built from (most likely this "
                "shower's EM-mode start correction) has moved since. Press "
                "<i>store this pairing</i> if today's numbers are the ones you "
                "mean.</span>" % "; ".join(drift))
    save_note.text = msg


def on_pio_update():
    """Replace one stored pairing with what the chain computes today.

    Round 16.  The freeze stays -- nothing re-prices a stored row on its own --
    but it now has a deliberate exit.  Two guards make it a judgement rather
    than an accident: a row has to be SELECTED, and the slots have to be holding
    that row's own two gammas, which is what `load into the slots` does.  Delete
    -and-re-store would have done the same arithmetic and lost the row's
    `stored_utc` and its place in the list, and a note that named it by number
    would have gone stale.
    """
    idx = list(pio_cand_src.selected.indices or [])
    if not idx:
        save_note.text = ("<span style='color:#c00'>pick the row to update in "
                          "the stored pairings table first.</span>")
        return
    i = idx[0]
    if i >= len(state["pio_cands"]):
        return
    old = state["pio_cands"][i]
    og = old.get("gammas") or {}
    want = ((og.get("1") or {}).get("shower"), (og.get("2") or {}).get("shower"))
    if (state["gamma"][1], state["gamma"][2]) != want:
        save_note.text = (
            "<span style='color:#c00'>the slots hold &gamma;1 %s + &gamma;2 %s "
            "but candidate <b>%d</b> is %s + %s. Press <i>load into the slots</i> "
            "first &mdash; updating a row from a different pairing would rewrite "
            "it into something it never was.</span>"
            % (state["gamma"][1], state["gamma"][2], i + 1, want[0], want[1]))
        return
    moved = (cand_drift(old) + cand_unused_marks(old)
             + cand_convention_note(old) + cand_unused_orphans(old))
    cand = pio_pairing()
    cand["stored_utc"] = old.get("stored_utc")
    cand["updated_utc"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")
    # Round 5d discipline: a past judgement is carried, never destroyed.  A list
    # so that a row updated twice keeps both of its earlier readings, and each
    # entry is small enough that the record does not grow without bound.
    def _summary(c):
        g = c.get("gammas") or {}
        return dict(stored_utc=c.get("stored_utc"),
                    updated_utc=c.get("updated_utc"),
                    energies=[(g.get(s) or {}).get("energy") for s in ("1", "2")],
                    energy_includes_marks=[(g.get(s) or {}).get(
                        "energy_includes_marks") for s in ("1", "2")],
                    energy_includes_orphans=[(g.get(s) or {}).get(
                        "energy_includes_orphans") for s in ("1", "2")],
                    vertex=c.get("vertex"), vertex_how=c.get("vertex_how"),
                    mass_axis_convention=c.get("mass_axis_convention"),
                    mass_vertex_convention=c.get("mass_vertex_convention"),
                    theta_axis_convention=c.get("theta_axis_convention"))
    cand["supersedes"] = (old.get("supersedes") or []) + [_summary(old)]
    state["pio_cands"][i] = cand
    refresh_pio_cands()
    touch()
    save_note.text = (
        "candidate <b>%d</b> updated to today's numbers: m = <b>%s</b> &rarr; "
        "<b>%s</b> MeV (axis), <b>%s</b> &rarr; <b>%s</b> MeV (vertex, %s). %s "
        "The reading it replaced is kept in the row as <code>supersedes</code>."
        % (i + 1, _mev(old.get("mass_axis_convention")),
           _mev(cand.get("mass_axis_convention")),
           _mev(old.get("mass_vertex_convention")),
           _mev(cand.get("mass_vertex_convention")), cand.get("vertex_how"),
           ("What moved: %s." % "; ".join(moved)) if moved else
           "<span style='color:#b58900'>Nothing had moved &mdash; the row "
           "already was today's answer.</span>"))


def on_pio_del():
    idx = sorted(pio_cand_src.selected.indices or [], reverse=True)
    if not idx:
        save_note.text = ("<span style='color:#c00'>pick the row(s) to remove "
                          "in the stored pairings table first.</span>")
        return
    gone = []
    for i in idx:
        if i < len(state["pio_cands"]):
            gone.append(i + 1)
            state["pio_cands"].pop(i)
    state["_suspend"] = True
    try:
        pio_cand_src.selected.indices = []
    finally:
        state["_suspend"] = False
    refresh_pio_cands()
    touch()
    save_note.text = ("removed pairing %s. %d left &mdash; they are renumbered, "
                      "so a note that names one by number is now stale."
                      % (" and ".join(str(x) for x in sorted(gone)),
                         len(state["pio_cands"])))


def on_gamma_clear():
    """Clears the SLOTS.  The stored pairings are untouched -- they are the
    record, and a control labelled `clear gammas` must not delete them."""
    state["gamma"] = {1: None, 2: None}
    state["gstart"] = {1: None, 2: None}
    draw_arrows()
    draw_gammas()
    refresh_kine()
    refresh_pio_cands()
    touch()


def _manual_point():
    try:
        return (float(man_x.value), float(man_y.value), float(man_z.value))
    except ValueError:
        return None


def on_snap():
    """Snap the slot's start point to the nearest FITTED point of that gamma's
    own segments.  Snapping to real data rather than to a free 3-D position is
    what makes a two-panel tap usable at all: the tap fixes two coordinates and
    the snap resolves the third onto the trajectory."""
    slot = gstart_slot.active + 1
    node = state["gamma"][slot]
    if node is None:
        save_note.text = "<span style='color:#c00'>assign that gamma slot first.</span>"
        return
    target = _manual_point()
    if target is None:
        save_note.text = ("<span style='color:#c00'>type or tap an x/y/z to snap "
                          "toward.</span>")
        return
    best, bestd = None, None
    for sid in members_of(node):
        for s in cur_segments():
            if s.get("id") != sid:
                continue
            dd, q = G.segment_closest_point(s, target)
            if dd is not None and (bestd is None or dd < bestd):
                bestd, best = dd, q
    if best is None:
        save_note.text = "<span style='color:#c00'>that gamma has no fitted points.</span>"
        return
    state["gstart"][slot] = best
    save_note.text = ("gamma %d start snapped to (%.1f, %.1f, %.1f), %.2f cm from "
                      "your point" % (slot, best[0], best[1], best[2], bestd))
    draw_arrows()
    draw_gammas()
    refresh_kine()
    touch()


def on_gstart_reset():
    state["gstart"][gstart_slot.active + 1] = None
    draw_arrows()
    draw_gammas()
    refresh_kine()
    touch()


def on_bp_geom():
    if state["_suspend"]:
        return
    draw_gammas()
    refresh_kine()
    touch()


def on_vtx_mode(attr, old, new):
    if state["_suspend"]:
        return
    if new == 2:
        state["vtx_manual"] = _manual_point()
    draw_gammas()
    refresh_kine()
    touch()


def on_manual(attr, old, new):
    if state["_suspend"]:
        return
    if vtx_mode_group.active == 2:
        state["vtx_manual"] = _manual_point()
        draw_gammas()
        refresh_kine()
        touch()


def tap_fill(hx, hy):
    """Two projections show two of the three coordinates each, so a tap in two
    different panels pins a full 3-D position.  Gated by the toggle so ordinary
    taps do not clobber typed coordinates."""
    box = {"x": man_x, "y": man_y, "z": man_z}

    def cb(event):
        if not tap_toggle.active:
            return
        box[hx].value = "%.1f" % event.x
        box[hy].value = "%.1f" % event.y
    return cb


def apply_layers(attr, old, new):
    on = {LAYER_KEYS[i] for i in layer_group.active}
    for k, rs in RENDER.items():
        for r in rs:
            r.visible = k in on
    # The two cloud renderers share a CDS and a layer key but only one is ever
    # the live colour mode, so the checkbox alone must not turn both on.
    sync_cloud_vis()


def set_centre(p, why=""):
    """Orbit around `p` instead of around the bounding-sphere centre.

    The projection is relative to cam_c, so re-centring puts `p` at (0, 0) in
    view space; the ranges therefore have to be re-centred on zero as well, and
    they keep their CURRENT span so the user's zoom survives.  The ranges must be
    written BEFORE push_camera, which snapshots them into the pan anchor.

    cam_R is deliberately not touched: it is the zoom-independent scale the depth
    cue normalises by, and rewriting it here would silently rescale the fading.
    Known consequence, not a bug: orbiting far from the bounding-sphere centre
    flattens the depth cue, because the event no longer spans +-R about the new
    centre."""
    span_x = f3d.x_range.end - f3d.x_range.start
    span_y = f3d.y_range.end - f3d.y_range.start
    state["cam_c"] = (float(p[0]), float(p[1]), float(p[2]))
    f3d.x_range.start, f3d.x_range.end = -0.5 * span_x, 0.5 * span_x
    f3d.y_range.start, f3d.y_range.end = -0.5 * span_y, 0.5 * span_y
    push_camera()
    save_note.text = ("now orbiting around (%.1f, %.1f, %.1f)%s &mdash; drag to "
                      "rotate about it. <i>Refit</i> puts the centre back."
                      % (p[0], p[1], p[2], why))


def _em_redraw():
    """Everything the pass-1 gate feeds.  The start and the axis are inputs to
    all of it, so moving either has to sweep the same set a shower selection
    does -- otherwise the table says one thing and the plot another."""
    state["_axis_cache"] = {}
    fill_cand_table()
    draw_tiers()
    draw_arrows()
    draw_gammas()          # the start / direction markers ride along here
    refresh_marks()
    refresh_impact()
    refresh_emstart()
    touch()


def set_em_start(p, why="", vid=None):
    """Make a clicked point this shower's start."""
    node = state["sel_shower"]
    if node is None:
        save_note.text = ("<b>pick a shower first</b> &mdash; a start point has "
                          "to belong to one. Click a row in the shower table, "
                          "then click the point again.")
        return
    if mode_group.active != 0:
        mode_group.active = 0
    state["em_start"][node] = (float(p[0]), float(p[1]), float(p[2]))
    if vid is not None:
        state["em_startvid"][node] = str(vid)
    else:
        state["em_startvid"].pop(node, None)
    _em_redraw()
    rs = reco_start(node)
    moved = G.vmag(G.vsub(state["em_start"][node], rs)) if rs else None
    save_note.text = (
        "shower %s start set to (%.1f, %.1f, %.1f)%s%s &mdash; the candidate "
        "table, the acceptance plot and the saved metrics are all measured from "
        "it now. <b>The axis moved too</b>: %s."
        % (node, p[0], p[1], p[2], why,
           "" if moved is None else ", %.1f cm from the reco start" % moved,
           ("it is aimed through the point you picked"
            if state["em_dir"].get(node) is not None else
            "recomputed at 15 cm from the new start, since the probe's value "
            "was anchored at the old one")))


def set_em_dir(p, why=""):
    """Aim this shower's axis through a clicked point."""
    node = state["sel_shower"]
    if node is None:
        save_note.text = ("<b>pick a shower first</b> &mdash; a direction has to "
                          "belong to one.")
        return
    if mode_group.active != 0:
        mode_group.active = 0
    base = shower_start(node)
    if base is not None and G.vmag(G.vsub(p, base)) < 1e-6:
        save_note.text = ("that point IS the start &mdash; a direction needs a "
                          "second, different point to aim through.")
        return
    state["em_dir"][node] = (float(p[0]), float(p[1]), float(p[2]))
    _em_redraw()
    ax, _, _ = shower_axis(node)
    save_note.text = (
        "shower %s axis aimed through (%.1f, %.1f, %.1f)%s &mdash; direction "
        "now (%.3f, %.3f, %.3f), measured from the start in use."
        % (node, p[0], p[1], p[2], why, ax[0], ax[1], ax[2]))


def _nearest_fit_point(p):
    best, bd = None, None
    for sg in cur_segments():
        for q in G.seg_points(sg):
            d = G.vmag(G.vsub(q, p))
            if bd is None or d < bd:
                best, bd = q, d
    return best, bd


def _nearest_vertex(p):
    best, bd, bid = None, None, None
    for v in ((state["data"] or {}).get("vertices") or []):
        q = G.pt(v.get("fit"))
        if not q:
            continue
        d = G.vmag(G.vsub(q, p))
        if bd is None or d < bd:
            best, bd, bid = q, d, v.get("id")
    return best, bd, bid


def _anchor_for_snap():
    """What the buttons snap RELATIVE to: the typed boxes if they hold a point,
    else the start currently in use.  Without this the buttons would need a
    click in the 3-D view to mean anything, which is the trip they exist to
    save."""
    try:
        return (float(em_sx.value), float(em_sy.value), float(em_sz.value))
    except (TypeError, ValueError):
        pass
    node = state["sel_shower"]
    return shower_start(node) if node is not None else None


def on_em_startv():
    p = _anchor_for_snap()
    if p is None:
        return
    q, d, vid = _nearest_vertex(p)
    if q is None:
        save_note.text = "this event has no reconstructed vertices to snap to."
        return
    set_em_start(q, " (vertex %s, %.1f cm away)" % (vid, d), vid=vid)


def on_em_startp():
    p = _anchor_for_snap()
    if p is None:
        return
    q, d = _nearest_fit_point(p)
    if q is None:
        return
    set_em_start(q, " (fit point, %.1f cm away)" % d)


def on_em_dirp():
    p = _anchor_for_snap()
    if p is None:
        return
    q, d = _nearest_fit_point(p)
    if q is None:
        return
    set_em_dir(q, " (fit point, %.1f cm away)" % d)


def on_em_setxyz():
    try:
        p = (float(em_sx.value), float(em_sy.value), float(em_sz.value))
    except (TypeError, ValueError):
        save_note.text = "start x / y / z need three numbers."
        return
    set_em_start(p, " (typed)")


def clear_em_start():
    node = state["sel_shower"]
    if node is None:
        return
    state["em_start"].pop(node, None)
    state["em_startvid"].pop(node, None)
    _em_redraw()
    save_note.text = "shower %s start back to the reconstruction's." % node


def clear_em_dir():
    node = state["sel_shower"]
    if node is None:
        return
    state["em_dir"].pop(node, None)
    _em_redraw()
    save_note.text = "shower %s axis back to being computed, not aimed." % node


def refresh_emstart():
    """Say what the start and the axis currently ARE, and warn when marks were
    made before they moved."""
    node = state["sel_shower"]
    if node is None:
        emstart_div.text = ""
        return
    rs = reco_start(node)
    ov = state["em_start"].get(node)
    dp = state["em_dir"].get(node)
    ax, br, axsrc = shower_axis(node)
    bits = []
    if rs:
        bits.append("reco start (%.1f, %.1f, %.1f)" % rs)
    if ov:
        vid = state["em_startvid"].get(node)
        bits.append("<b style='color:#ff7f0e'>yours (%.1f, %.1f, %.1f)%s, "
                    "%.1f cm away</b>"
                    % (ov[0], ov[1], ov[2],
                       " = vertex %s" % vid if vid else "",
                       G.vmag(G.vsub(ov, rs)) if rs else float("nan")))
    bits.append("axis <code>%s</code> / <code>%s</code>" % (br, axsrc))
    if dp:
        bits.append("<b style='color:#2ca02c'>aimed through "
                    "(%.1f, %.1f, %.1f)</b>" % dp)
    warn = ""
    if (ov or dp) and marks_for(node):
        # Said out loud, not silently absorbed: mark_metrics recomputes tier,
        # angle and distance AT SAVE TIME, so marks made by eye before the move
        # get a record whose geometry the scanner never saw on screen.
        warn = ("<div style='color:#b58900;margin-top:3px'>&#9888; you already "
                "marked %d segment(s) on this shower. The saved tier / angle / "
                "distance are recomputed from the start and axis above, not "
                "from the ones you marked against &mdash; re-check the "
                "candidate table before saving.</div>" % len(marks_for(node)))
    emstart_div.text = (
        "<div style='background:#f7f7f7;border-left:4px solid #8c564b;"
        "padding:5px 9px;margin:2px 0;font-size:90%%'><b>start &amp; axis</b> "
        "&mdash; %s%s</div>" % (" &nbsp;|&nbsp; ".join(bits), warn))


def set_pio_vertex(p, why=""):
    """Make a clicked point the pi0 decay vertex.

    Order matters and this is a reentrancy edge: the x/y/z boxes are written
    first (under _suspend, so their on_change does not fire mid-update), THEN the
    mode radio is moved to `manual`, and the redraws are called explicitly.
    Flipping the radio first would run on_vtx_mode, which re-reads the boxes --
    and at that instant they still hold the PREVIOUS point."""
    state["_suspend"] = True
    try:
        man_x.value = "%.1f" % p[0]
        man_y.value = "%.1f" % p[1]
        man_z.value = "%.1f" % p[2]
        state["vtx_manual"] = (float(p[0]), float(p[1]), float(p[2]))
        vtx_mode_group.active = 2
    finally:
        state["_suspend"] = False
    if mode_group.active != 1:
        mode_group.active = 1           # nothing pi0 is visible in EM mode
    draw_gammas()
    refresh_kine()
    touch()
    save_note.text = ("pi0 vertex set to (%.1f, %.1f, %.1f)%s &mdash; the mass "
                      "block below now uses it." % (p[0], p[1], p[2], why))


def fill_xyz(p, why=""):
    state["_suspend"] = True
    try:
        man_x.value = "%.1f" % p[0]
        man_y.value = "%.1f" % p[1]
        man_z.value = "%.1f" % p[2]
    finally:
        state["_suspend"] = False
    on_manual(None, None, None)
    save_note.text = "picked (%.1f, %.1f, %.1f)%s" % (p[0], p[1], p[2], why)


def apply_marks(ids, kind):
    """`kind` is "in"/"out"/None, or "toggle" for the cycle.

    Refuses outright with no shower selected.  A mark is a statement ABOUT a
    shower -- "this piece belongs to that one" -- so there is no meaningful place
    to put one when no shower is named, and the round-4 behaviour of dropping it
    into a global dict is what let a mark end up filed against the wrong shower.
    """
    if not ids:
        return False
    node = state["sel_shower"]
    if node is None:
        save_note.text = ("<span style='color:#c00'>pick a shower in the table "
                          "first &mdash; a mark is recorded <i>against</i> a "
                          "shower, so it needs one to belong to.</span>")
        return False
    here = marks_for(node)
    for sid in ids:
        if kind == "toggle":
            nxt = {None: "in", "in": "out", "out": None}.get(here.get(sid), "in")
            if nxt is None:
                here.pop(sid, None)
            else:
                here[sid] = nxt
        elif kind is None:
            here.pop(sid, None)
        else:
            here[sid] = kind
    refresh_marks()
    fill_cand_table()
    refresh_impact()
    touch()
    return True


def offframe_hint(ids):
    """Warn when a mark landed outside the visible frame.

    `focus_points` includes marks, so *refit* will bring it in -- but the camera
    is deliberately NOT moved here: re-framing on every mark would throw away the
    scanner's zoom mid-judgement.  Saying it is the honest middle."""
    if not ids or not state.get("cam_R"):
        return ""
    half = min(f3d.x_range.end - f3d.x_range.start,
               f3d.y_range.end - f3d.y_range.start) / 2.0
    out = []
    for sid in ids:
        for p in poly_for(sid):
            u, v, _ = _proj([(p[0], p[1], p[2])])[0]
            if math.hypot(u, v) > half:
                out.append(sid)
                break
    if not out:
        return ""
    return ("  <span style='color:#b58900'>%s outside the current view &mdash; "
            "press <i>refit</i> to bring %s in.</span>"
            % (", ".join(str(s) for s in out[:4]),
               "it" if len(out) == 1 else "them"))


def on_pick(attr, old, new):
    """A tap or box on the 3-D pick surface.

    One gesture, seven jobs, chosen by `tap_action`, because a rotatable canvas
    gives a ray and not a point.  The marking actions are round 4's answer to
    "click the 3-D display to select things in and out": the tap IS the mark, no
    trip to a button.  The geometry actions resolve the ray by snapping to a real
    fitted point -- the same trick the two-panel tap needed the snap button for,
    done in one click."""
    if state["_suspend"] or not new:
        return
    act = tap_action.value
    sids = [pick_src.data["sid"][i] for i in new
            if i < len(pick_src.data.get("sid", []))]
    if act in (TAP_CENTRE, TAP_XYZ, TAP_PIO, TAP_START, TAP_DIR):
        i = new[0]
        try:
            p = (pick_src.data["x"][i], pick_src.data["y"][i],
                 pick_src.data["z"][i])
        except (KeyError, IndexError):
            return
        why = " on segment %s" % pick_src.data["sid"][i]
        {TAP_CENTRE: set_centre, TAP_XYZ: fill_xyz, TAP_PIO: set_pio_vertex,
         TAP_START: set_em_start, TAP_DIR: set_em_dir}[act](p, why)
        _clear_pick()
        return
    ids = sorted(set(sids))
    if act == TAP_SELECT:
        # Mirror the 3-D pick into the table and the acceptance plot, so a box
        # drawn in the view lights up the same segments in the numbers.
        sync_selection(pick_src)
        save_note.text = (
            "3-D selection: %d segment(s) &mdash; %s. Now press mark IN / OUT / ?."
            % (len(ids), ", ".join(str(s) for s in ids[:8])
               + (" ..." if len(ids) > 8 else "")))
        return
    # Only report success if the mark actually landed: apply_marks refuses when
    # no shower is selected, and overwriting its refusal with this line would
    # tell the scanner a mark was recorded when none was.
    if not apply_marks(ids, {TAP_IN: "in", TAP_OUT: "out",
                             TAP_TOGGLE: "toggle"}[act]):
        _clear_pick()
        return
    save_note.text = ("%s: %d segment(s) &mdash; %s%s"
                      % (act, len(ids), ", ".join(
                          "%s=%s" % (s, marks_for(state["sel_shower"]).get(s, "-"))
                          for s in ids[:8])
                         + (" ..." if len(ids) > 8 else ""),
                         offframe_hint(ids)))
    # Bokeh does not re-fire selected.indices when the SAME index is tapped
    # again, and "toggle" is defined by tapping the same segment repeatedly, so
    # the selection has to be re-armed by hand.  Only for the marking actions:
    # the mark itself is now the feedback, whereas TAP_SELECT's whole job is to
    # LEAVE a selection standing for the mark buttons to consume.
    _clear_pick()


def _clear_pick():
    # The table and the plot are cleared with the 3-D cloud now that the three
    # are brushed together: leaving a row highlighted after the mark landed would
    # say the selection still stands when the next mark button would find it
    # already consumed.  _guard as well as _suspend, so the clears do not each
    # bounce back through sync_selection.
    state["_suspend"] = True
    state["_guard"] = True
    try:
        for src in (pick_src, vtx3_src, mainvtx3_src, cand_src, cand_pt_src):
            src.selected.indices = []
    finally:
        state["_guard"] = False
        state["_suspend"] = False
    push_polys(sel_src, [], sel3_src)


def on_vtx_pick(src, what):
    """Tapping a reconstructed vertex.

    Vertices answer only the GEOMETRY actions -- a vertex has no segment id, so
    "mark IN" has nothing to mark and says so instead of guessing.  This is a
    separate handler on a separate source precisely so that a vertex can never
    reach state["marks"]; see the _tap3/_box3 renderer split above."""
    def cb(attr, old, new):
        if state["_suspend"] or not new:
            return
        i = new[0]
        try:
            p = (src.data["x"][i], src.data["y"][i], src.data["z"][i])
        except (KeyError, IndexError):
            return
        tag = (src.data.get("tag") or [""] * (i + 1))[i]
        why = " (%s %s)" % (what, tag)
        act = tap_action.value
        if act == TAP_CENTRE:
            set_centre(p, why)
        elif act == TAP_PIO:
            set_pio_vertex(p, why)
        elif act == TAP_XYZ:
            fill_xyz(p, why)
        elif act == TAP_START:
            # "change the start VERTEX": a tap that lands on a reconstructed
            # vertex records WHICH one, so the record can say "vertex 41" and
            # not just a bare point.
            set_em_start(p, why, vid=tag if what == "vertex" else None)
        elif act == TAP_DIR:
            set_em_dir(p, why)
        else:
            save_note.text = (
                "that is %s %s at (%.1f, %.1f, %.1f) &mdash; a vertex, not a "
                "segment, so <i>%s</i> has nothing to mark. Switch <b>a tap in "
                "3-D does</b> to <i>%s</i>, <i>%s</i> or <i>%s</i> to use it."
                % (what, tag, p[0], p[1], p[2], act, TAP_START, TAP_PIO,
                   TAP_CENTRE))
        _clear_pick()
    return cb


def on_camtxt(attr, old, new):
    """The browser reports the camera once per gesture (on panend), never per
    frame.  Python only needs it to put in the saved record and the readout."""
    try:
        az, el = (float(v) for v in new.split(","))
    except (ValueError, AttributeError):
        return
    state["cam"] = (az, el)
    c, R = state["cam_c"], state["cam_R"]
    cam_div.text = ("<span style='font-size:85%%;color:#555'>camera az %.0f&deg; "
                    "el %.0f&deg; &nbsp; centre (%.0f, %.0f, %.0f) &nbsp; "
                    "R %.0f cm</span>"
                    % (math.degrees(az), math.degrees(el), c[0], c[1], c[2], R))


def on_cloud_opt(attr, old, new):
    draw_cloud()
    sync_cloud_vis()
    if fit_mode.active == 1:
        refit_camera(push=False)
    push_camera()


def on_verdict(attr, old, new):
    if not state["_suspend"]:
        touch()


mode_group.on_change("active", on_mode)
event_select.on_change("value", on_event)
prev_btn.on_click(step(-1))
next_btn.on_click(step(+1))
shower_src.selected.on_change("indices", on_shower_select)
mark_in_btn.on_click(mark("in"))
mark_out_btn.on_click(mark("out"))
mark_q_btn.on_click(mark("?"))
mark_clear_btn.on_click(mark(None))
bulk_sel_btn.on_click(on_bulk(False))
bulk_add_btn.on_click(on_bulk(True))
show_all_toggle.on_click(lambda a: fill_cand_table())
g1_btn.on_click(on_gamma(1))
g2_btn.on_click(on_gamma(2))
g_clear_btn.on_click(on_gamma_clear)
pio_add_btn.on_click(on_pio_add)
pio_load_btn.on_click(on_pio_load)
pio_upd_btn.on_click(on_pio_update)
pio_del_btn.on_click(on_pio_del)
for _w in (g1_ehyp, g2_ehyp, emark_mode):
    _w.on_change("value", lambda a, o, n: (refresh_kine(), touch()))
# Not folded into the loop above, for the same kind of reason on_vtx_mode is
# not: this switch is the only one the EM-mode MARK LIST reads, so flipping it
# has to rebuild that too or the note under a mark keeps saying "will not reach
# the pi0 mass" about charge that is now being counted.  Caught in the live
# browser smoke, not in the headless test, which called both refreshes by hand.
orph_mode.on_change("value", lambda a, o, n: (refresh_kine(),
                                              refresh_mark_list(), touch()))
# Not folded into the loop above: this one MOVES the pi0 vertex, so the marker
# in the 3-D view has to be redrawn -- the same pair of calls on_vtx_mode makes
# -- and it must honour _suspend, because load()/load_label set it while an
# event is half-loaded.
bp_geom.on_change("value", lambda a, o, n: on_bp_geom())
snap_btn.on_click(on_snap)
em_startv_btn.on_click(on_em_startv)
em_startp_btn.on_click(on_em_startp)
em_dirp_btn.on_click(on_em_dirp)
em_setxyz_btn.on_click(on_em_setxyz)
em_start_reset.on_click(clear_em_start)
em_dir_reset.on_click(clear_em_dir)
gstart_reset.on_click(on_gstart_reset)
vtx_mode_group.on_change("active", on_vtx_mode)
for w in (man_x, man_y, man_z):
    w.on_change("value", on_manual)
layer_group.on_change("active", apply_layers)
em_verdict.on_change("active", on_verdict)
conf_group.on_change("active", on_verdict)
event_flag_group.on_change("active", lambda a, o, n: touch())
save_btn.on_click(on_save)
for _f, _hx, _hy in PROJ:
    _f.on_event(Tap, tap_fill(_hx, _hy))
pick_src.selected.on_change("indices", on_pick)
vtx3_src.selected.on_change("indices", on_vtx_pick(vtx3_src, "vertex"))
mainvtx3_src.selected.on_change("indices",
                                on_vtx_pick(mainvtx3_src, "the main vertex"))
# The acceptance plot and the candidate table feed the same resolver, so a
# selection made there lights the same cyan halo a 3-D box does.
cand_src.selected.on_change("indices", lambda a, o, n: sync_selection(cand_src))
cand_pt_src.selected.on_change("indices",
                               lambda a, o, n: sync_selection(cand_pt_src))
excl_choice.on_change("value", on_excl)
seg_color_mode.on_change("active", on_seg_color_mode)
acc_zoom.on_click(lambda a: fill_cand_table())
camtxt.on_change("value", on_camtxt)
cloud_layer.on_change("value", on_cloud_opt)
cloud_max.on_change("value", on_cloud_opt)
cloud_scope.on_change("active", on_cloud_opt)
cloud_color.on_change("active", lambda a, o, n: sync_cloud_vis())
fit_mode.on_change("active", lambda a, o, n: refit_camera())
refit_btn.on_click(lambda: refit_camera())
dim_toggle.on_click(lambda a: refresh_dim())


def on_tap_action(attr, old, new):
    """Changing what a tap does drops any standing selection.

    Not tidiness -- correctness.  Bokeh fires `selected.indices` only when the
    value CHANGES, so a selection left standing from the previous action makes
    the first gesture in the new one a no-op: box a region in "select", switch to
    "mark IN", box the same region again, and nothing happens because the index
    list is identical.  The real browser found this; the Python test could not,
    because it sets indices directly and had cleared them in between."""
    _clear_pick()


tap_action.on_change("value", on_tap_action)


def on_view_size(attr, old, new):
    """Bokeh re-lays out on a width/height change, and the JS pan handler reads
    p.inner_width live, so nothing else has to move.  The ranges are untouched:
    resizing must not re-frame the event under the scanner."""
    f3d.width = f3d.height = int(new)


view_size.on_change("value", on_view_size)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
header = Div(text="<h2 style='margin:2px 0'>EM shower &amp; &pi;&#8304; hand scan "
                  "<span style='font-size:60%;color:#666'>doc pr/114</span></h2>",
             width=RW)

em_panel = column(
    Div(text="<b>1.</b> pick a shower in the table above. "
             "<b>2.</b> select segments &mdash; candidate table, acceptance plot, "
             "or a tap / box straight in the 3-D view. <b>3.</b> mark them "
             "(or set <i>a tap in 3-D does</i> to mark on the click itself).",
        width=RW),
    row(mark_in_btn, mark_out_btn, mark_q_btn, mark_clear_btn, show_all_toggle),
    Div(text="<b>a whole shower at once</b> &mdash; pick it here, press "
             "<i>select all its segments</i>, then <b>mark IN</b>. That is the "
             "bulk answer to &ldquo;shower B is really part of shower A&rdquo;, "
             "and it beats clicking segments one after another because it takes "
             "the membership from the probe rather than from your aim. "
             "<i>add to selection</i> keeps what is already selected, so several "
             "fragments go in together.", width=RW),
    row(bulk_shower, bulk_sel_btn, bulk_add_btn),
    Div(text="<b>start &amp; direction</b> &mdash; set <i>a tap in 3-D does</i> "
             "to <i>%s</i> and click a vertex (or any fit point); then <i>%s</i> "
             "and click a second point to aim the axis. Everything below is "
             "measured from them." % (TAP_START, TAP_DIR), width=RW),
    row(em_startv_btn, em_startp_btn, em_dirp_btn),
    row(em_sx, em_sy, em_sz, em_setxyz_btn, em_start_reset, em_dir_reset),
    emstart_div,
    cand_tab,
    marks_div,
    Div(text="<b>verdict for this shower</b>", width=RW), em_verdict,
    impact)

pio_panel = column(
    row(g1_btn, g2_btn, g_clear_btn),
    Div(text="<b>energy hypothesis</b> &mdash; <code>kine_charge</code> was "
             "converted with the pair <code>get_flag_shower()</code> chose, not "
             "with the one the slot implies. If the reco called a gamma a track "
             "or a proton, switch it here and the mass follows.", width=RW),
    row(g1_ehyp, g2_ehyp, emark_mode),
    row(orph_mode),
    row(gstart_slot, snap_btn, gstart_reset),
    Div(text="<b>pi0 decay vertex</b> &mdash; or set <i>a tap in 3-D does</i> to "
             "<i>%s</i> and click the point straight in the 3-D view "
             "(reconstructed vertices are clickable too)." % TAP_PIO, width=RW),
    vtx_mode_group,
    row(bp_geom),
    row(man_x, man_y, man_z, tap_toggle),
    kine_div,
    Div(text="<b>stored \u03c0\u2070 pairings</b> &mdash; one event can hold "
             "more than one \u03c0\u2070, and one \u03c0\u2070 more than one "
             "plausible pairing. <b>store this pairing</b> freezes what is in "
             "the slots &mdash; both gammas, their starts, their energy "
             "hypotheses, the vertex and both masses &mdash; as its own entry. "
             "Pick a row and <b>load into the slots</b> to look at it again.",
        width=RW),
    row(pio_add_btn, pio_load_btn, pio_upd_btn, pio_del_btn),
    pio_cand_tab,
    pio_cand_div,
    Div(text="<span style='font-size:88%;color:#555'>No pi0 verdict: the record "
             "keeps the reconstruction's pairing (<code>reco_groups</code>, "
             "<code>reco_kine</code>) and yours (<code>gammas</code>, "
             "<code>vertex</code>) side by side, and the difference between them "
             "IS the judgement.</span>", width=RW))

_hr = (lambda: Div(text="<hr style='margin:6px 0'>", width=CW))
cam_panel = column(
    Div(text="<b>3-D view</b> &mdash; <b>drag</b> rotates, <b>shift+drag</b> "
             "pans, <b>wheel</b> zooms. Picking <i>Box Select</i> in the toolbar "
             "suspends rotation while it is on. Depth is shown by fading, not by "
             "perspective.", width=CW),
    row(*preset_btns), row(refit_btn, view_size), fit_mode,
    seg_color_mode,
    cam_div,
    _hr(),
    cloud_layer, cloud_scope, cloud_color, cloud_max, cloud_div,
    _hr(),
    tap_action, dim_toggle,
    _hr(),
    legend_div)

view_tabs = Tabs(tabs=[
    TabPanel(child=row(f3d, Spacer(width=12), cam_panel), title="3-D"),
    TabPanel(child=column(row(f_xy, f_yz), f_xz), title="2-D projections"),
], active=0, name="view_tabs")

# Two columns, not one 880-wide strip: the owner scans on a wide screen and
# round 3 left two thirds of it empty.  The 3-D view and its controls own the
# left, everything that is read or clicked while looking at it owns the right,
# and only the header strip spans both.
right_col = column(
    row(layer_group), info,
    shower_tab,
    excl_choice,
    em_panel,
    pio_panel,
    row(column(acc, row(acc_zoom), acc_note)),
    cmp_div,
    Div(text="<b>this event's topology</b> &mdash; event-level, saved as "
             "<code>event_flags</code> beside <code>em</code> and "
             "<code>pio</code>, so a later pass can select these events "
             "without opening a shower block.", width=RW),
    event_flag_group,
    row(conf_group, note_in, save_btn),
    save_note, width=RW)

layout = column(
    row(header, Spacer(width=20), event_select, prev_btn, next_btn,
        Spacer(width=20), mode_group),
    scan_status,
    banner,
    scan_note_div,
    row(column(view_tabs), Spacer(width=18), right_col))


def on_mode_layout(attr, old, new):
    em_panel.visible = (new == 0)
    pio_panel.visible = (new == 1)


mode_group.on_change("active", on_mode_layout)
on_mode_layout(None, None, mode_group.active)

curdoc().add_root(layout)
curdoc().title = "EM / pi0 hand scan"
apply_layers(None, None, None)
if LABELS:
    load(LABELS[0])

# The chip reads disk, but nothing was WAKING it while the scanner sits on one
# event: refresh_scan_status only fires from refresh_info, i.e. on load, save and
# touch.  So a save made in a second tab -- the case the disk read exists for,
# and a likely one, since every restart tells the owner to reload -- would not
# show here until they navigated away and back.  One stat every 5 s closes that,
# and re-assigning an unchanged Div.text syncs nothing.
curdoc().add_periodic_callback(refresh_scan_status, 5000)

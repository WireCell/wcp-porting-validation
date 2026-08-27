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
from bokeh.models import (Button, CheckboxButtonGroup, ColumnDataSource, Div,
                          HoverTool, Select, TextInput, Toggle, RadioButtonGroup,
                          DataTable, TableColumn, CDSView, AllIndices, Range1d,
                          TapTool, Span, NumberFormatter, CustomJS, Tabs,
                          TabPanel, BoxSelectTool, WheelZoomTool, ResetTool,
                          SaveTool)
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
EM_VERDICTS = ["correct", "over-clustered", "under-clustered", "both",
               "vertex-bad (undecidable)", "not an EM shower"]
PIO_VERDICTS = ["pi0 correct", "wrong pairing", "wrong start point",
                "wrong vertex", "shower mis-grouped", "not a pi0"]
CONF = ["certain", "likely", "unclear"]

state = dict(label=None, data=None, prep=None,
             sel_shower=None,          # node id of the shower under scan
             marks={},                 # seg id -> "in" / "out" / "?"
             gamma={1: None, 2: None},  # slot -> node id
             gstart={1: None, 2: None},  # slot -> (x,y,z) override or None
             vtx_mode="main", vtx_manual=None,
             dirty=False, saved=None,
             cam=(math.radians(D3.PRESETS["iso"][0]),
                  math.radians(D3.PRESETS["iso"][1])),
             cam_c=(0.0, 0.0, 0.0), cam_R=100.0, cloud=None,
             _suspend=False, _guard=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
proj_kw = dict(height=330, width=420, tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom")
# Range1d, never the figure() default DataRange1d: DataRange1d auto-refits to
# renderer data on every CDS push, which silently undoes an active zoom every
# time a table click updates the highlight source.
f_xy = figure(title="X-Y", x_range=Range1d(*DET_BOX["x"]),
              y_range=Range1d(*DET_BOX["y"]), **proj_kw)
f_yz = figure(title="Y-Z", x_range=Range1d(*DET_BOX["z"]),
              y_range=Range1d(*DET_BOX["y"]), **proj_kw)
f_xz = figure(title="X-Z", x_range=Range1d(*DET_BOX["x"]),
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
det_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                     xs_xz=[], ys_xz=[]))
# Polylines are per-projection: the xs/ys pairs genuinely differ.
seg_src = {k: ColumnDataSource(data=dict(xs=[], ys=[], c=[], sid=[], pid=[],
                                         cid=[], owner=[], mark=[]))
           for k in ("xy", "yz", "xz")}
mem_src = {k: ColumnDataSource(data=dict(xs=[], ys=[])) for k in ("xy", "yz", "xz")}
in_src = {k: ColumnDataSource(data=dict(xs=[], ys=[])) for k in ("xy", "yz", "xz")}
out_src = {k: ColumnDataSource(data=dict(xs=[], ys=[])) for k in ("xy", "yz", "xz")}
# Axis arrows: multi_line shaft + a rotated triangle head, NOT Bokeh's Arrow
# annotation -- Arrow is one model per arrow per panel and we draw several.
arrow_src = {k: ColumnDataSource(data=dict(xs=[], ys=[], c=[]))
             for k in ("xy", "yz", "xz")}
head_src = {k: ColumnDataSource(data=dict(x=[], y=[], angle=[], c=[]))
            for k in ("xy", "yz", "xz")}

RENDER = {}


def _add(key, r):
    RENDER.setdefault(key, []).append(r)


for f, hx, hy in PROJ:
    # f_yz plots z on the horizontal axis, so its key is "yz" while its (hx, hy)
    # spells "zy" -- the one place the panel name and the column pair disagree.
    k = {"xy": "xy", "zy": "yz", "xz": "xz"}[hx + hy]
    _add("det", f.multi_line(xs="xs_" + k, ys="ys_" + k, source=det_src,
                             line_color="#cc4444", line_width=1, alpha=0.55))
    _add("shwpt", f.scatter(hx, hy, source=shwpt_src,
                            size=2, color="#8fbf8f", alpha=0.45))
    # membership halos, drawn BEFORE the coloured segments so they sit beneath
    _add("member", f.multi_line(xs="xs", ys="ys", source=mem_src[k],
                                line_color="#ffd27f", line_width=9, alpha=0.55))
    _add("mark", f.multi_line(xs="xs", ys="ys", source=in_src[k],
                              line_color="#2ca02c", line_width=13, alpha=0.45))
    _add("mark", f.multi_line(xs="xs", ys="ys", source=out_src[k],
                              line_color="#d62728", line_width=13, alpha=0.40))
    r_seg = f.multi_line(xs="xs", ys="ys", source=seg_src[k], line_color="c",
                         line_width=2, alpha=0.95)
    _add("segments", r_seg)
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

# ---------------------------------------------------------------------------
# The 3-D panel (doc pr/114 round 3)
# ---------------------------------------------------------------------------
# Rotatable like Bee, but inside Bokeh so every label control keeps working off
# it.  The mechanics, the frame constraint and the honest limits are all in
# em3d.py's docstring; this block is only the wiring.
#
# Sources come in three shapes, matching em3d.JS_REDRAW's three loops:
#   POINT  x, y, z  ->  u, v, al, sz
#   LINE   xs3, ys3, zs3  ->  xs, ys
#   HEAD   x, y, z + x0, y0, z0  ->  u, v, angle
# Python fills the projected columns too, so the first paint of an event is
# correct before any drag has happened; the JS then owns every later frame.
EMPTY3D = dict(x=[], y=[], z=[], c=[], tag=[], u=[], v=[], al=[], sz=[])
EMPTY3L = dict(xs3=[], ys3=[], zs3=[], xs=[], ys=[])

# The `name=` on these four is not decoration: selftest_em3d_browser.py drives a
# real headless chromium and reaches them with `get_model_by_name`, which is how
# the CustomJS below gets tested at all in a tree with no JS engine.
cam_src = ColumnDataSource(name="cam_src", data=dict(
    az=[state["cam"][0]], el=[state["cam"][1]], cx=[0.0], cy=[0.0], cz=[0.0],
    R=[100.0], az0=[0.0], el0=[0.0], xs0=[0.0], xe0=[0.0], ys0=[0.0], ye0=[0.0]))

cloud_src = ColumnDataSource(name="cloud_src",
                             data=dict(x=[], y=[], z=[], q=[], cid20=[],
                                       u=[], v=[], al=[], sz=[]))
shwpt3_src = ColumnDataSource(data=dict(EMPTY3D))
vtx3_src = ColumnDataSource(data=dict(EMPTY3D))
mainvtx3_src = ColumnDataSource(data=dict(EMPTY3D))
gstart3_src = ColumnDataSource(data=dict(EMPTY3D))
piovtx3_src = ColumnDataSource(data=dict(EMPTY3D))
# Every fitted point of every segment, carrying its segment id.  This is the
# pick surface: Bokeh's own TapTool and BoxSelectTool hit-test it in screen space
# on the PROJECTED columns, so 3-D selection needs no JS at all -- and because a
# hit resolves to a segment id, a box in a rotated view marks whole segments, not
# a prism of loose points.
pick_src = ColumnDataSource(name="pick_src",
                            data=dict(x=[], y=[], z=[], sid=[], u=[], v=[],
                                      al=[], sz=[]))
det3_src = ColumnDataSource(data=dict(EMPTY3L))
seg3_src = ColumnDataSource(name="seg3_src",
                            data=dict(xs3=[], ys3=[], zs3=[], xs=[], ys=[], c=[],
                                      sid=[], pid=[], cid=[], owner=[], mark=[]))
mem3_src = ColumnDataSource(data=dict(EMPTY3L))
in3_src = ColumnDataSource(data=dict(EMPTY3L))
out3_src = ColumnDataSource(data=dict(EMPTY3L))
arrow3_src = ColumnDataSource(data=dict(xs3=[], ys3=[], zs3=[], xs=[], ys=[], c=[]))
head3_src = ColumnDataSource(data=dict(x=[], y=[], z=[], x0=[], y0=[], z0=[],
                                       u=[], v=[], angle=[], c=[]))

_wheel3 = WheelZoomTool(dimensions="both")
_tap3 = TapTool()
_box3 = BoxSelectTool()
f3d = figure(name="f3d", title="3-D  —  drag rotates, shift+drag pans, wheel zooms",
             width=660, height=660,
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
_add("member", f3d.multi_line(xs="xs", ys="ys", source=mem3_src,
                              line_color="#ffd27f", line_width=9, alpha=0.55))
_add("mark", f3d.multi_line(xs="xs", ys="ys", source=in3_src,
                            line_color="#2ca02c", line_width=13, alpha=0.45))
_add("mark", f3d.multi_line(xs="xs", ys="ys", source=out3_src,
                            line_color="#d62728", line_width=13, alpha=0.40))
r_seg3 = f3d.multi_line(xs="xs", ys="ys", source=seg3_src, line_color="c",
                        line_width=2, alpha=0.95)
_add("segments", r_seg3)
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
# The pick surface is invisible (alpha 0) but still hit-tested -- Bokeh hit tests
# geometry, not paint.  Deliberately NOT registered in RENDER: a layer checkbox
# that silently disabled selection would be a trap.
r_pick = f3d.scatter("u", "v", source=pick_src, size="sz", fill_alpha="al",
                     line_alpha=0.0)
r_pick.nonselection_glyph = r_pick.glyph
for _t in (_tap3, _box3):
    _t.renderers = [r_pick]
for _r in (r_vtx3, r_mv3, r_cloud_c, r_cloud_q):
    _r.nonselection_glyph = _r.glyph

# --- the CustomJS, spliced from em3d so there is one copy of the formula ------
#                     cloud shwpt  pick  vtx  mainvtx gstart piovtx
_PT_SRC = [cloud_src, shwpt3_src, pick_src, vtx3_src, mainvtx3_src, gstart3_src,
           piovtx3_src]
_PT_SIZE = [2.6, 2.0, 7.0, 6.0, 20.0, 18.0, 24.0]
_PT_ALPHA = [0.55, 0.45, 0.0, 0.75, 0.95, 0.95, 0.95]
_PT_CUE = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
                lines=[det3_src, seg3_src, mem3_src, in3_src, out3_src,
                       arrow3_src],
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
cam_div = Div(text="", width=330)
cloud_layer = Select(title="charge cloud", value=D3.CLOUD_LAYERS[0],
                     options=D3.CLOUD_LAYERS + ["(none)"], width=170)
cloud_color = RadioButtonGroup(labels=["by cluster", "by charge"], active=0,
                               width=180)
cloud_max = Select(title="max points", value="25000",
                   options=["10000", "25000", "50000", "100000"], width=110)
cloud_div = Div(text="", width=330)
pick_mode = RadioButtonGroup(labels=["tap selects segment", "tap fills x/y/z"],
                             active=0, width=330)
fit_mode = RadioButtonGroup(labels=["frame the reco", "frame the cloud"],
                            active=0, width=330)

# --- the acceptance plot ----------------------------------------------------
# Deliberately NOT a cone drawn over the projections.  A 3-D cone does not
# project to a cone, so any wedge drawn on X-Y would be decorative and would
# invite exactly the wrong reading.  This panel is the gate itself, exactly:
# distance and angle are the two quantities NeutrinoShowerClustering.cxx:1310
# actually tests, so a dot below a step is inside that tier and a dot above it
# is not.  Nothing is approximated here.
acc = figure(title="pass-1 acceptance: angle to shower axis vs distance",
             height=330, width=430, x_range=Range1d(0, 220),
             y_range=Range1d(0, 90),
             tools="pan,wheel_zoom,box_zoom,reset,save,tap",
             active_scroll="wheel_zoom")
acc.xaxis.axis_label = "distance from shower start (cm)"
acc.yaxis.axis_label = "angle to shower axis (deg)"
tier_src = ColumnDataSource(data=dict(xs=[], ys=[]))
acc.multi_line(xs="xs", ys="ys", source=tier_src, line_color="#666666",
               line_width=2, line_dash="dashed", alpha=0.9)
cand_pt_src = ColumnDataSource(data=dict(x=[], y=[], c=[], sid=[], pid=[],
                                         length=[], tier=[], owner=[], site=[],
                                         mark=[]))
r_cand = acc.scatter("x", "y", source=cand_pt_src, size=9, fill_color="c",
                     line_color="#333333", alpha=0.85)
r_cand.nonselection_glyph = r_cand.glyph
acc.add_tools(HoverTool(renderers=[r_cand], tooltips=[
    ("segment", "@sid"), ("pdg", "@pid"), ("length", "@length{0.0} cm"),
    ("tier", "@tier"), ("now in shower", "@owner"),
    ("absorbed by", "@site"), ("your mark", "@mark")]))
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
mode_group = RadioButtonGroup(labels=["EM shower", "pi0"], active=0, width=240)
event_select = Select(title="event", options=LABELS, value=LABELS[0], width=190)
prev_btn = Button(label="< prev", width=80)
next_btn = Button(label="next >", width=80)
LAYERS = [("segments", "track fit"), ("member", "shower members"),
          ("mark", "your marks"), ("arrows", "axes"), ("vertices", "vertices"),
          ("shwpt", "shower pts"), ("gamma", "gammas / pi0 vtx"), ("det", "volume"),
          ("cloud", "charge cloud (3-D)")]
LAYER_KEYS = [k for k, _ in LAYERS]
layer_group = CheckboxButtonGroup(labels=[t for _, t in LAYERS],
                                  active=[0, 1, 2, 3, 4, 6, 7, 8])
banner = Div(text="", width=880)
info = Div(text="", width=880)

shower_src = ColumnDataSource(data=dict(node=[], pdg=[], nseg=[], joined=[],
                                        E=[], kb=[], conn=[], pio=[], length=[],
                                        drift=[], flag=[]))
shower_view_a, shower_view_b = AllIndices(), AllIndices()
shower_view = CDSView(filter=shower_view_a)
shower_tab = DataTable(source=shower_src, view=shower_view, width=880, height=210,
                       index_position=None, columns=[
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

cand_src = ColumnDataSource(data=dict(sid=[], cid=[], pdg=[], length=[], dist=[],
                                      angle=[], tier=[], metric=[], owner=[],
                                      site=[], mark=[]))
cand_view_a, cand_view_b = AllIndices(), AllIndices()
cand_view = CDSView(filter=cand_view_a)
cand_tab = DataTable(source=cand_src, view=cand_view, width=880, height=250,
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
# Default ON.  The owner's question is symmetric -- "should this be inside or
# outside" -- and the `absorbed by` column only ever has something to say about a
# segment that WAS absorbed, i.e. a member.  Hiding members by default made that
# column read empty in the default view, which is exactly backwards: "why is this
# one IN" is the question the probe can actually answer.
show_all_toggle = Toggle(label="show members too (off = only segments outside "
                               "this shower)", width=380, active=True)
em_verdict = RadioButtonGroup(labels=EM_VERDICTS, active=None)
impact = Div(text="", width=880)

# --- pi0 controls -----------------------------------------------------------
g1_btn = Button(label="selected shower -> gamma 1", width=210)
g2_btn = Button(label="selected shower -> gamma 2", width=210)
g_clear_btn = Button(label="clear gammas", width=120)
gstart_slot = RadioButtonGroup(labels=["gamma 1", "gamma 2"], active=0, width=160)
snap_btn = Button(label="snap start to nearest fit point", width=230)
gstart_reset = Button(label="reset to reco start", width=170)
vtx_mode_group = RadioButtonGroup(
    labels=["main vertex", "back-project the two gammas", "manual"], active=0)
man_x = TextInput(title="x", value="", width=90)
man_y = TextInput(title="y", value="", width=90)
man_z = TextInput(title="z", value="", width=90)
tap_toggle = Toggle(label="tap fills x/y/z", width=140)
kine_div = Div(text="", width=880)
pio_verdict = RadioButtonGroup(labels=PIO_VERDICTS, active=None)

conf_group = RadioButtonGroup(labels=CONF, active=None, width=240)
note_in = TextInput(title="note (optional)", value="", width=520)
save_btn = Button(label="Save event label", button_type="success", width=170)
save_note = Div(text="", width=880)


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


def shower_axis(node):
    """(dir, branch, source).  Prefers the probe's dir15 -- that is the C++'s own
    `shower_cal_dir_3vector(shower, start, 15cm)`, so it needs no reproduction
    caveat.  Without a sidecar, falls back to the Python `init_dir` mirror."""
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


def shower_start(node, slot=None):
    if slot is not None and state["gstart"].get(slot):
        return state["gstart"][slot]
    sh = shower_by_node(node)
    return G.pt(sh.get("start")) if sh else None


def shower_energy(node):
    """Sum of member kine_charge is NOT available per segment in the dump, so the
    shower's own kine_charge is used -- and that is the right number anyway: it
    is exactly what the C++ mass formula reads (get_kine_charge(), see
    NeutrinoShowerClustering.cxx:3771)."""
    sh = shower_by_node(node)
    return (sh or {}).get("kine_charge")


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


def refit_camera(push=True):
    src = reco_points()
    if fit_mode.active == 1 and state.get("cloud"):
        cl = state["cloud"]
        src = list(zip(cl["x"], cl["y"], cl["z"])) or src
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
                           max_pts=int(cloud_max.value))
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
    cloud_div.text = (
        "<span style='font-size:85%%;color:#555'><code>%s</code> &mdash; showing "
        "<b>%s</b> of %s points%s</span>%s"
        % (html.escape(cl["layer"]), "{:,}".format(cl["kept"]),
           "{:,}".format(cl["total"]),
           " (every %d)" % max(1, cl["total"] // max(1, cl["kept"])) if
           cl["kept"] < cl["total"] else "", warn))


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
    ins = [s for s, m in state["marks"].items() if m == "in"]
    outs = [s for s, m in state["marks"].items() if m == "out"]
    push_polys(in_src, ins, in3_src)
    push_polys(out_src, outs, out3_src)


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
    out_m = {s for s, m in state["marks"].items() if m == "out" and s in mem}
    in_m = {s for s, m in state["marks"].items() if m == "in" and s not in mem}
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


def pio_vertex():
    """(point, how, detail).  `how` is what goes in the saved record."""
    d = state["data"] or {}
    if vtx_mode_group.active == 0:
        return G.pt(d.get("main_vertex")), "main_vertex", {}
    if vtx_mode_group.active == 2:
        return state["vtx_manual"], "manual", {}
    n1, n2 = state["gamma"][1], state["gamma"][2]
    if n1 is None or n2 is None:
        return None, "backproject", {"verdict": "need two gammas"}
    sh1, sh2 = shower_by_node(n1), shower_by_node(n2)
    anchor = G.pt(d.get("main_vertex"))
    if anchor is None:
        return None, "backproject", {"verdict": "no main vertex to anchor from"}
    bp = G.pi0_backproject(sh1, sh2, cur_segments(), anchor)
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
        e1, e2 = shower_energy(n1), shower_energy(n2)
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
            "m = <b>%s MeV</b>%s"
            % (how, "%.1f&deg;" % thB if thB is not None else "-",
               "%.1f" % mB if mB is not None else "-",
               "  <span style='color:#2ca02c'>[in the code's accept window]</span>"
               if G.pi0_mass_accepted(mB) else ""))
        rows.append(
            "<i>The two conventions are shown side by side on purpose: the code "
            "itself uses different direction recipes for the mass it stores "
            "(:3771) and the angle it stores (:3830), and they do not close.</i>")

    if how == "backproject" and detail:
        rows.append("<b>back-projection</b> (mirror of id_pi0_without_vertex, "
                    "NeutrinoShowerClustering.cxx:4158-4256): branch "
                    "<b>%s</b>, verdict <b>%s</b>%s" % (
                        detail.get("branch") or "-", detail.get("verdict"),
                        "" if detail.get("verdict") == "ok" else
                        " &mdash; <span style='color:#d62728'>the code would "
                        "REFUSE this pair here</span>"))
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


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def fill_shower_table():
    d = state["data"] or {}
    segs = d.get("segments") or []
    rows = dict(node=[], pdg=[], nseg=[], joined=[], E=[], kb=[], conn=[],
                pio=[], length=[], drift=[], flag=[])
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
    shower_src.data = rows
    flip(shower_view, shower_view_a, shower_view_b)
    return nloss


def fill_cand_table():
    node = state["sel_shower"]
    rows = dict(sid=[], cid=[], pdg=[], length=[], dist=[], angle=[], tier=[],
                metric=[], owner=[], site=[], mark=[])
    pts = dict(x=[], y=[], c=[], sid=[], pid=[], length=[], tier=[], owner=[],
               site=[], mark=[])
    if node is None:
        cand_src.data = rows
        cand_pt_src.data = pts
        flip(cand_view, cand_view_a, cand_view_b)
        return
    start = shower_start(node)
    ax, _, _ = shower_axis(node)
    off = G.cone_angle_offset(ax)
    mem = set(members_of(node))
    owner_of = {}
    for sh in cur_showers():
        for s in members_of(sh.get("id")):
            owner_of.setdefault(s, sh.get("id"))
    for s in cur_segments():
        sid = s.get("id")
        if sid in mem and not show_all_toggle.active:
            continue
        if start is None or G.vmag(ax) == 0:
            dist = angle = None
        else:
            dist, q = G.segment_closest_point(s, start)
            angle = G.angle_deg(ax, G.vsub(q, start)) if q is not None else None
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
        rows["mark"].append(state["marks"].get(sid, ""))
        if dist is not None and angle is not None:
            mk = state["marks"].get(sid, "")
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
    cand_src.data = rows
    cand_pt_src.data = pts
    flip(cand_view, cand_view_a, cand_view_b)


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------


def load(lbl):
    path = EVENTS.get(lbl)
    state["label"] = lbl
    state["sel_shower"] = None
    state["marks"] = {}
    state["gamma"] = {1: None, 2: None}
    state["gstart"] = {1: None, 2: None}
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

    segs = d.get("segments") or []
    dat = {k: dict(xs=[], ys=[], c=[], sid=[], pid=[], cid=[], owner=[], mark=[])
           for k in ("xy", "yz", "xz")}
    d3 = dict(polys=[], c=[], sid=[], pid=[], cid=[], owner=[], mark=[])
    pick = dict(pts=[], sid=[])
    owner_of = {}
    for sh in (d.get("showers") or []):
        for s in members_of(sh.get("id")):
            owner_of.setdefault(s, sh.get("id"))
    for i, s in enumerate(segs):
        pts = G.seg_points(s)
        if len(pts) < 2:
            continue
        c = seg_color(i)
        for k, (a, b) in (("xy", (0, 1)), ("yz", (2, 1)), ("xz", (0, 2))):
            dat[k]["xs"].append([p[a] for p in pts])
            dat[k]["ys"].append([p[b] for p in pts])
            dat[k]["c"].append(c)
            dat[k]["sid"].append(s.get("id"))
            dat[k]["pid"].append(s.get("particle_id"))
            dat[k]["cid"].append(s.get("cluster_id"))
            dat[k]["owner"].append(owner_of.get(s.get("id"), "-"))
            dat[k]["mark"].append("")
        d3["polys"].append([tuple(p) for p in pts])
        d3["c"].append(c)
        d3["sid"].append(s.get("id"))
        d3["pid"].append(s.get("particle_id"))
        d3["cid"].append(s.get("cluster_id"))
        d3["owner"].append(owner_of.get(s.get("id"), "-"))
        d3["mark"].append("")
        for p in pts:
            pick["pts"].append(tuple(p))
            pick["sid"].append(s.get("id"))
    for k in ("xy", "yz", "xz"):
        seg_src[k].data = dat[k]
    fill3_lines(seg3_src, d3.pop("polys"), **d3)

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
    # the next "mark IN" apply to a segment of the PREVIOUS event.
    pick_src.selected.indices = []
    if state.get("cloud"):
        cl = state["cloud"]
        fill3_points(cloud_src, list(zip(cl["x"], cl["y"], cl["z"])),
                     q=cl["q"], cid20=cl["cid20"])

    nloss = fill_shower_table()
    fill_cand_table()
    draw_tiers()
    draw_arrows()
    draw_gammas()
    refresh_marks()
    refresh_impact()
    refresh_kine()
    load_label(lbl)
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
    if url:
        bits.append("<a href='%s' target='_blank' style='font-weight:bold'>"
                    "open in Bee &#8599;</a> <span style='color:#666'>(%s)</span>"
                    % (html.escape(url), html.escape(row.get("bee_round", ""))))
    else:
        bits.append("<span style='color:#999'>no Bee link for this event &mdash; "
                    "build a set with <code>prep_em_scan.py --bee-build bee/em114"
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
    banner.text = " &nbsp;|&nbsp; ".join(bits)


def refresh_info():
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
        pio_verdict.active = None
        conf_group.active = None
        note_in.value = ""
        if not os.path.exists(p):
            return
        with open(p) as fh:
            rec = json.load(fh)
        state["saved"] = rec
        em = rec.get("em") or {}
        if em.get("shower") is not None:
            state["sel_shower"] = em["shower"]
            state["marks"] = {int(k): v for k, v in (em.get("marks") or {}).items()}
            if em.get("verdict") in EM_VERDICTS:
                em_verdict.active = EM_VERDICTS.index(em["verdict"])
        pio = rec.get("pio") or {}
        for slot in (1, 2):
            g = (pio.get("gammas") or {}).get(str(slot))
            if g:
                state["gamma"][slot] = g.get("shower")
                if g.get("start_override"):
                    state["gstart"][slot] = tuple(g["start_override"])
        if pio.get("verdict") in PIO_VERDICTS:
            pio_verdict.active = PIO_VERDICTS.index(pio["verdict"])
        if pio.get("vertex_how") == "manual" and pio.get("vertex"):
            state["vtx_manual"] = tuple(pio["vertex"])
            vtx_mode_group.active = 2
        elif pio.get("vertex_how") == "backproject":
            vtx_mode_group.active = 1
        if rec.get("confidence") in CONF:
            conf_group.active = CONF.index(rec["confidence"])
        note_in.value = rec.get("note") or ""
    finally:
        state["_suspend"] = False


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

    em_block = None
    if state["sel_shower"] is not None or state["marks"]:
        node = state["sel_shower"]
        sh = shower_by_node(node) or {}
        j, n = G.join_completeness(sh, cur_segments()) if sh else (0, 0)
        ax, br, axsrc = shower_axis(node) if node is not None else ((0, 0, 0), "", "")
        em_block = dict(
            shower=node,
            marks={str(k): v for k, v in sorted(state["marks"].items())},
            verdict=EM_VERDICTS[em_verdict.active] if em_verdict.active is not None else None,
            # the reco's own answer, copied in so a later fit never re-reads the dump
            reco=dict(members=sorted(members_of(node)) if node is not None else [],
                      membership_source="probe" if probe_members(node) is not None
                      else "dump-join",
                      join_complete=(j == n), num_segments=n, joined=j,
                      kine_charge=sh.get("kine_charge"),
                      kine_best=sh.get("kine_best"),
                      start_connection_type=sh.get("start_connection_type"),
                      pio_id=sh.get("pio_id"),
                      axis=list(ax), axis_branch=br, axis_source=axsrc))

    pio_block = None
    if state["gamma"][1] is not None or state["gamma"][2] is not None:
        v, how, detail = pio_vertex()
        gam = {}
        for slot in (1, 2):
            node = state["gamma"][slot]
            if node is None:
                continue
            gam[str(slot)] = dict(
                shower=node,
                start=list(shower_start(node, slot) or []),
                start_override=list(state["gstart"][slot]) if state["gstart"][slot] else None,
                energy=shower_energy(node),
                members=sorted(members_of(node)),
                axis=list(shower_axis(node)[0]))
        e1 = shower_energy(state["gamma"][1]) if state["gamma"][1] is not None else None
        e2 = shower_energy(state["gamma"][2]) if state["gamma"][2] is not None else None
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
        pio_block = dict(
            gammas=gam, vertex=list(v) if v else None, vertex_how=how,
            backproject=detail if how == "backproject" else None,
            mass_axis_convention=mA, theta_axis_convention=thA,
            mass_vertex_convention=mB, theta_vertex_convention=thB,
            verdict=PIO_VERDICTS[pio_verdict.active] if pio_verdict.active is not None else None,
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
        main_vertex=d.get("main_vertex"),
        # The view the judgement was made from, so a later re-read can put the
        # event back on screen the way it was seen.
        camera=dict(az_deg=round(math.degrees(state["cam"][0]), 2),
                    el_deg=round(math.degrees(state["cam"][1]), 2),
                    centre=[round(v, 2) for v in state["cam_c"]],
                    R=round(state["cam_R"], 2),
                    cloud=(state["cloud"] or {}).get("layer"),
                    cloud_kept=(state["cloud"] or {}).get("kept"),
                    cloud_total=(state["cloud"] or {}).get("total")),
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
    save_note.text = ("saved <code>%s</code> at %s"
                      % (html.escape(os.path.relpath(path, SX)), rec["saved_utc"]))
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
    draw_tiers()
    draw_arrows()
    push_polys(mem_src, members_of(node), mem3_src)
    refresh_impact()
    push_camera()


def selected_cand_ids():
    out = []
    for i in (cand_src.selected.indices or []):
        try:
            out.append(cand_src.data["sid"][i])
        except (KeyError, IndexError):
            pass
    for i in (cand_pt_src.selected.indices or []):
        try:
            out.append(cand_pt_src.data["sid"][i])
        except (KeyError, IndexError):
            pass
    # A 3-D tap or box lands on fitted POINTS, but the labelling unit is the
    # segment, so it resolves to segment ids -- which is what makes a box in a
    # rotated view unambiguous where a lasso over a flat projection is not.
    seen = set(out)
    for i in (pick_src.selected.indices or []):
        try:
            sid = pick_src.data["sid"][i]
        except (KeyError, IndexError):
            continue
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def mark(kind):
    def cb():
        ids = selected_cand_ids()
        if not ids:
            save_note.text = ("<span style='color:#c00'>select one or more rows "
                              "(table or acceptance plot) first.</span>")
            return
        for sid in ids:
            if kind is None:
                state["marks"].pop(sid, None)
            else:
                state["marks"][sid] = kind
        refresh_marks()
        fill_cand_table()
        refresh_impact()
        touch()
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
        touch()
    return cb


def on_gamma_clear():
    state["gamma"] = {1: None, 2: None}
    state["gstart"] = {1: None, 2: None}
    draw_arrows()
    draw_gammas()
    refresh_kine()
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


def on_pick(attr, old, new):
    """A tap or box on the 3-D pick surface.

    Two jobs on one gesture, chosen by pick_mode, because a rotatable canvas gives
    a ray and not a point: "select segment" is the labelling path (marks then work
    off it exactly as from the tables), "fill x/y/z" resolves the ray by snapping
    to a real fitted point -- which is the same trick the two-panel tap needed the
    snap button for, done in one click."""
    if state["_suspend"] or not new:
        return
    if pick_mode.active == 1:
        i = new[0]
        try:
            x, y, z = (pick_src.data["x"][i], pick_src.data["y"][i],
                       pick_src.data["z"][i])
        except (KeyError, IndexError):
            return
        state["_suspend"] = True
        try:
            man_x.value = "%.1f" % x
            man_y.value = "%.1f" % y
            man_z.value = "%.1f" % z
        finally:
            state["_suspend"] = False
        on_manual(None, None, None)
        save_note.text = ("picked fitted point (%.1f, %.1f, %.1f) on segment %s"
                          % (x, y, z, pick_src.data["sid"][i]))
        return
    ids = sorted({pick_src.data["sid"][i] for i in new
                  if i < len(pick_src.data.get("sid", []))})
    save_note.text = ("3-D selection: %d segment(s) &mdash; %s. Now press mark "
                      "IN / OUT / ?." % (len(ids), ", ".join(str(s) for s in ids[:8])
                                         + (" ..." if len(ids) > 8 else "")))


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
show_all_toggle.on_click(lambda a: fill_cand_table())
g1_btn.on_click(on_gamma(1))
g2_btn.on_click(on_gamma(2))
g_clear_btn.on_click(on_gamma_clear)
snap_btn.on_click(on_snap)
gstart_reset.on_click(on_gstart_reset)
vtx_mode_group.on_change("active", on_vtx_mode)
for w in (man_x, man_y, man_z):
    w.on_change("value", on_manual)
layer_group.on_change("active", apply_layers)
em_verdict.on_change("active", on_verdict)
pio_verdict.on_change("active", on_verdict)
conf_group.on_change("active", on_verdict)
save_btn.on_click(on_save)
for _f, _hx, _hy in PROJ:
    _f.on_event(Tap, tap_fill(_hx, _hy))
pick_src.selected.on_change("indices", on_pick)
camtxt.on_change("value", on_camtxt)
cloud_layer.on_change("value", on_cloud_opt)
cloud_max.on_change("value", on_cloud_opt)
cloud_color.on_change("active", lambda a, o, n: sync_cloud_vis())
fit_mode.on_change("active", lambda a, o, n: refit_camera())
refit_btn.on_click(lambda: refit_camera())


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
header = Div(text="<h2 style='margin:2px 0'>EM shower &amp; &pi;&#8304; hand scan "
                  "<span style='font-size:60%;color:#666'>doc pr/114</span></h2>",
             width=880)

em_panel = column(
    Div(text="<b>1.</b> pick a shower in the table above. "
             "<b>2.</b> select segments in the candidate table or the acceptance "
             "plot. <b>3.</b> mark them.", width=880),
    row(mark_in_btn, mark_out_btn, mark_q_btn, mark_clear_btn, show_all_toggle),
    cand_tab,
    Div(text="<b>verdict for this shower</b>", width=880), em_verdict,
    impact)

pio_panel = column(
    row(g1_btn, g2_btn, g_clear_btn),
    row(gstart_slot, snap_btn, gstart_reset),
    Div(text="<b>pi0 decay vertex</b>", width=880), vtx_mode_group,
    row(man_x, man_y, man_z, tap_toggle),
    kine_div,
    Div(text="<b>verdict for this pi0</b>", width=880), pio_verdict)

view_tabs = Tabs(tabs=[
    TabPanel(child=row(f3d, column(
        Div(text="<b>3-D view</b> &mdash; <b>drag</b> rotates, "
                 "<b>shift+drag</b> pans, <b>wheel</b> zooms. Picking "
                 "<i>Box Select</i> in the toolbar suspends rotation while it is "
                 "on. Depth is shown by fading, not by perspective.", width=330),
        row(*preset_btns), row(refit_btn, fit_mode),
        cam_div,
        Div(text="<hr style='margin:6px 0'>", width=330),
        cloud_layer, cloud_color, cloud_max, cloud_div,
        Div(text="<hr style='margin:6px 0'>", width=330),
        Div(text="<b>tap / box in 3-D</b>", width=330), pick_mode)),
             title="3-D"),
    TabPanel(child=row(f_xy, f_yz, f_xz), title="2-D projections"),
], active=0, name="view_tabs")

layout = column(
    header,
    banner,
    row(event_select, prev_btn, next_btn, Spacer(width=20), mode_group),
    view_tabs,
    row(column(acc, acc_note), Spacer(width=20), column(layer_group, info)),
    shower_tab,
    em_panel,
    pio_panel,
    row(conf_group, note_in, save_btn),
    save_note)


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

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
  Row 2 -- particle flow (click a particle to highlight it) next to the
           event's selection numbers.
  Row 3 -- dQ/dx panel (sbnd_xin/docs/pr/42): click a PF row above and its
           dQ/dx-vs-distance profile appears here against the muon/proton/
           pion/kaon reference curves (End mode, tracks) or the 1x/2x MIP
           lines (Start mode, showers).  Mode auto-picks by particle kind;
           override with the Start/End toggle, and use the segment dropdown
           to step through a multi-segment shower.
  Row 4 -- OFF BY DEFAULT (`--wire-planes` to restore): six panels, two
           columns (TPC 0 | TPC 1) x three rows (T-U, T-V, T-W), the fitted
           2-D charge as a heat map with the best-fit track drawn over it.

LAYERS (each a toggle)

  track fit      the PR graph's segments, drawn as polylines, coloured per
                 segment; this is the reconstructed particle trajectory set
  shower pts     associated 3-D points flagged shower
  track pts      associated 3-D points flagged track
  steiner        the Steiner skeleton (steiner_pc) of every cluster
  terminals      only the flag_steiner_terminal subset of that skeleton
  vertices       PR graph vertices; the neutrino vertex is drawn larger.
                 TAP one to select its row in the hand-scan table below (an
                 exact index join -- tapping empty space does nothing).  The
                 tap deliberately does NOT re-centre or force zoom the way
                 clicking a table row does: you are already looking at it.
  dead           dead-channel bands, 2-D panels only
  dQ/dx          the fitted track points coloured by their own measured
                 dQ/dx in e/cm (`segments[].points[].dQ / .dx`), on a FIXED
                 0-150000 ramp with a shared colour bar under the panels.
                 Fixed, not per-event autoscale: an autoscale makes every
                 event look alike and destroys cross-event comparability
                 during a scan.  Points with no measurement (PR::Fit
                 defaults dQ=-1, dx=0) are neutral grey, never the bottom of
                 the ramp.  With this layer on, the per-segment polylines
                 dim to 0.30 alpha so the charge ramp reads; toggling it off
                 restores them exactly.

                 NB this is NOT what wire-cell-bee3 shows for track_fit
                 points.  bee3 colours by a dx-UNnormalised
                 `q = dQ * 0.1 - 1000` (MultiAlgBlobClustering.cxx baking in
                 sbnd/clus.jsonnet's dQdx_scale/dQdx_offset), on a blue->red
                 HSL ramp clipped at 9333.  Because the fit step dx is ~0.6 cm
                 and roughly constant, bee3's colour only tracks dQ/dx
                 approximately.  Here it is the real ratio, in the same e/cm
                 as the 1-D dQ/dx panel and as meta.mip_dqdx_median.

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
                          CheckboxButtonGroup, RadioButtonGroup, TextInput, Toggle, Spacer,
                          ColorBar, LinearColorMapper, BasicTicker, Span,
                          DataTable, TableColumn, HTMLTemplateFormatter,
                          CDSView, AllIndices, Range1d, TapTool)
from bokeh.events import Tap
from bokeh.palettes import Viridis256, Turbo256, Category20_20
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
ap.add_argument("--wire-planes", action="store_true",
                help="show the six 2-D wire-plane panels (hidden by default -- "
                     "sbnd_xin/docs/pr/42: not useful for day-to-day PID work; "
                     "replaced by the dQ/dx panel. Code path unchanged, just "
                     "left out of the layout, so this flag brings it back.")
ap.add_argument("--scan-tag", default=None,
                help="neutrino-vertex hand-scan label set: labels land in "
                     "../vertex_labels/<tag>/labels-evt<ID>.json.  Omit and the "
                     "viewer uses 'scan1' but REFUSES to write into it if it "
                     "already holds labels -- CLAUDE.md M13, a scan record is "
                     "never appended to by accident.  Pass the tag explicitly "
                     "to continue an existing scan.")
args = ap.parse_args(sys.argv[1:])
SHOW_WIRE_PLANES = args.wire_planes
# Explicit tag => the user meant this scan set, so writing into it is allowed
# even when it already has labels.  Implicit => first write only.
SCAN_TAG = args.scan_tag or "scan1"
SCAN_TAG_EXPLICIT = args.scan_tag is not None

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

state = dict(doc=None, label=None,
             # neutrino-vertex hand scan (sbnd_xin/docs/pr/75)
             vrows=[], vpicks=[], vsaved=None, vdirty=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
proj_kw = dict(height=340, width=430, tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom")
# Explicit Range1d, not the figure()-default DataRange1d: DataRange1d
# auto-refits to renderer data on every CDS push (e.g. the PF-highlight
# source updated by on_pf_select/on_kine_select), which silently undid an
# active "zoom" toggle on any particle-flow/energy-table click. Range1d only
# ever moves when apply_ranges() sets it, so a toggled zoom now survives
# every other control.
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
# Hand-scan overlays (sbnd_xin/docs/pr/75).  Deliberately NOT part of the
# `vertices` layer toggle: turning the PR vertices off must not hide your own
# picks, or you cannot check a pick against the bare charge.
selvtx_src = ColumnDataSource(data=dict(EMPTY3))            # table row under the cursor
pick_src = ColumnDataSource(data=dict(x=[], y=[], z=[], c=[], tag=[]))  # your ranked picks
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
# Direction arrows (sbnd_xin/docs/pr/80 sec 9).  One arrow per segment, drawn
# at the COOLER end and pointing toward the hotter one -- i.e. the way the
# particle was travelling, since a charged particle deposits more as it slows.
# Computed from the fitted points, NOT from `dirsign`, so the arrow can and
# sometimes does disagree with the reconstruction's own direction verdict; that
# disagreement is the point.  A segment whose two ends are within 1.3x of each
# other gets NO arrow: "no opinion" must never be drawn as an opinion.
# multi_line + a rotated triangle rather than Bokeh's Arrow annotation, which
# is one model per arrow and would mean ~300 models per event across three
# panels.
arrow_src = {k: ColumnDataSource(data=dict(xs=[], ys=[]))
             for k in ("xy", "yz", "xz")}
arrowhead_src = {k: ColumnDataSource(data=dict(x=[], y=[], angle=[]))
                 for k in ("xy", "yz", "xz")}

# --- dQ/dx-coloured track-fit points ----------------------------------------
# The fitted trajectory POINTS, carrying their measured dQ/dx.  seg_src above
# flattens only x/y/z into per-segment polylines and throws dQ/dx away, and the
# two existing per-point scatter layers (shower_src/track_src) come from the
# `track_shower` block, which has no charge field at all -- so this is its own
# source rather than a column added to an existing one.
#
# dQ (electrons) and dx (cm) are both already in the dump, PrDisplayDump.cxx
# fit_json(): `j["dQ"] = fit.dQ; j["dx"] = fit.dx / cm;`.  dx is ALREADY divided
# by units::cm there, so points[].dQ / points[].dx is physical e/cm directly --
# the same quantity, same units, as the 1-D dQ/dx panel's y axis.  Do not divide
# by units::cm again (the writer's own comment flags that trap).
fitpt_src = ColumnDataSource(data=dict(x=[], y=[], z=[], dqdx=[], dQ=[], dx=[],
                                       rr=[], sid=[], cid=[], pid=[]))
# Fitted points with NO defined dQ/dx.  PR::Fit defaults are dQ=-1, dx=0
# (PRCommon.h) and the dump does not emit `index`, the only field Fit::valid()
# checks, so `dx > 0 and dQ >= 0` is the only client-side guard -- the same one
# _dqdx_valid_points() uses for the 1-D panel.  These are drawn NEUTRAL GREY and
# never at the bottom of the ramp: colouring "no measurement" as "low dQ/dx"
# would be a lie in the one panel being used to judge track direction.
fitpt_nodq_src = ColumnDataSource(data=dict(x=[], y=[], z=[], sid=[]))

# Its own mapper -- deliberately NOT the 2-D panels' CMAP, which is reassigned
# per event to the 99th percentile of proj[].charge.  A per-event autoscale
# would make every event look identical and destroy the cross-event
# comparability a hand scan depends on, so this range is FIXED and only ever
# moves when the operator types a new one.
# The top of the ramp is 150000 e/cm = 3.5x MIP, chosen from the measured
# distribution rather than picked round: 18601 fitted points over 34 events
# (25 nueCC48 + r1qlmc + r2mc, prod0813) give median 50123 e/cm (1.17 MIP),
# p75 1.74 MIP, p90 2.71 MIP, p95 3.62 MIP, with a long tail to 26 MIP.  At
# 3.5 MIP only 5.5% of points saturate -- the very tip of a Bragg peak, which
# should read as "hot" anyway -- while MIP lands at 29% of the ramp and 2 MIP
# at 57%, so the 1-vs-2 MIP shower-stem separation and the Bragg RISE that
# gives a track its direction both have real colour contrast.  f_dqdx's y_range
# (100000) would have clipped 14%.  Retype the max to re-scale live.
DQDX_LOW, DQDX_HIGH = 0.0, 150000.0     # e/cm
DQDX_CMAP = LinearColorMapper(palette=Turbo256, low=DQDX_LOW, high=DQDX_HIGH)

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
    # dQ/dx points, drawn AFTER the polylines so they sit on top of them.
    RENDER["dqdx"].append(
        f.scatter(hx, hy, source=fitpt_nodq_src, marker="circle", size=4,
                  fill_color="#9e9e9e", line_color=None, fill_alpha=0.6))
    r_dq = f.scatter(hx, hy, source=fitpt_src, marker="circle", size=6,
                     fill_color=dict(field="dqdx", transform=DQDX_CMAP),
                     line_color=None, fill_alpha=0.95)
    RENDER["dqdx"].append(r_dq)
    f.add_tools(HoverTool(renderers=[r_dq], tooltips=[
        ("dQ/dx (e/cm)", "@dqdx{0,0}"), ("dQ (e)", "@dQ{0,0}"),
        ("dx (cm)", "@dx{0.000}"), ("resid. range (cm)", "@rr{0.0}"),
        ("segment", "@sid"), ("cluster", "@cid"), ("pdg", "@pid")]))

    # Direction arrows, above the charge points so they stay readable over a
    # dense Bragg peak, below the vertices so they never hide a candidate.
    RENDER["arrows"].append(
        f.multi_line(xs="xs", ys="ys", source=arrow_src[key],
                     line_color="#1a7d32", line_width=2.0, line_alpha=0.9))
    RENDER["arrows"].append(
        f.scatter("x", "y", source=arrowhead_src[key], marker="triangle",
                  size=10, angle="angle", fill_color="#1a7d32",
                  line_color=None, fill_alpha=0.9))

    r_vtx = f.scatter(hx, hy, source=vtx_src, marker="circle", size=6,
                      fill_color=None, line_color="#111111", line_width=1.2)
    r_mainvtx = f.scatter(hx, hy, source=mainvtx_src, marker="star", size=20,
                          fill_color="#e377c2", line_color="#7b2d6b", line_width=1.5)
    RENDER["vertices"].append(r_vtx)
    RENDER["vertices"].append(r_mainvtx)
    # Tap a drawn vertex to select its row in the hand-scan table below.  Bound
    # to these two renderers ONLY: an exact index join, so the tap can never
    # answer with the wrong vertex, and a tap on empty space does nothing.
    f.add_tools(TapTool(renderers=[r_vtx, r_mainvtx]))
    # Bokeh fades every UNselected glyph by default.  Without this, tapping one
    # vertex visually erases the other 60-160 in the panel being scanned.
    for _r in (r_vtx, r_mainvtx):
        _r.nonselection_glyph = _r.glyph
    # Hand-scan: the selected candidate (hollow amber ring) and the ranked
    # picks (filled green, labelled with the rank).  Always visible.
    f.scatter(hx, hy, source=selvtx_src, marker="circle", size=22,
              fill_color=None, line_color="#ff8c00", line_width=2.5)
    f.scatter(hx, hy, source=pick_src, marker="diamond", size=17,
              fill_color="#2ca02c", fill_alpha=0.75,
              line_color="#12591b", line_width=1.5)
    f.text(x=hx, y=hy, text="tag", source=pick_src,
           text_color="#12591b", text_font_size="10pt",
           text_font_style="bold", x_offset=9, y_offset=-9)

# --- the six 2-D panels -----------------------------------------------------
PLANE_NAME = ("U", "V", "W")
panel = {}          # (apa, plane) -> dict(fig, cell, fit, dead)
CMAP = LinearColorMapper(palette=Viridis256, low=0, high=1)

for apa in (0, 1):
    for pl in (0, 1, 2):
        f = figure(title="TPC %d   T vs %s" % (apa, PLANE_NAME[pl]),
                   width=560, height=250,
                   x_range=Range1d(start=0, end=1), y_range=Range1d(start=0, end=1),
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

# --- the dQ/dx colour bar, under the three projections ----------------------
# One shared horizontal bar rather than one per panel: three 430 px panels have
# no room for a vertical bar each, and the scale is common to all three anyway.
dqdx_cbar_fig = figure(width=1290, height=64, toolbar_location=None,
                       min_border=0, outline_line_color=None)
dqdx_cbar_fig.add_layout(
    ColorBar(color_mapper=DQDX_CMAP, orientation="horizontal",
             ticker=BasicTicker(desired_num_ticks=6), label_standoff=5,
             title="track-fit dQ/dx (e/cm)   --   MIP 43000, 2x MIP 86000 "
                   "(fixed range, not per-event)"), "below")
dqdx_cbar_fig.xaxis.visible = dqdx_cbar_fig.yaxis.visible = False
dqdx_cbar_fig.grid.visible = False
# A figure carrying only a ColorBar layout has no glyph renderer, which Bokeh
# reports as W-1000 MISSING_RENDERERS on every session.  An empty scatter is a
# renderer and draws nothing, which keeps the served log clean.
dqdx_cbar_fig.scatter(x=[], y=[], size=0)
dqdx_lo_in = TextInput(title="dQ/dx min", value="%g" % DQDX_LOW, width=95)
dqdx_hi_in = TextInput(title="dQ/dx max", value="%g" % DQDX_HIGH, width=95)
dqdx_cbar_note = Div(text="", width=420)

# --- dQ/dx panel (sbnd_xin/docs/pr/42) --------------------------------------
# Click a particle-flow row (track or shower) and its measured dQ/dx appears
# here.  "End" mode plots residual range from the dumped `rr` (tracks, the
# stopping/Bragg end); "Start" mode plots distance from the shower's own
# start point (showers, the stem -- 1 vs 2 MIP separates e- from a converted
# photon).  Mode auto-picks by PF node kind on every click; the RadioButtonGroup
# only overrides it until the next click.  All dQ/dx here is e/cm, matching
# points[].dQ / points[].dx directly -- no unit conversion, unlike the
# reference-curve dump (PrDisplayDump::dump_dqdx_ref) which has its own trap.
dqdx_src = ColumnDataSource(data=dict(x=[], y=[]))
# The literal <=20 Shower::get_stem_dQ_dx samples the nue/single-photon
# taggers cut on, converted back from MIP units to e/cm so they sit on the
# same axis as the measured points.  Plotted at THIS segment's own point
# positions (index-matched, capped to whichever list is shorter) -- exact
# when the stem stayed within the start segment (the common case for a short
# shower stem); if get_stem_dQ_dx had to walk into a downstream segment for a
# short stem, the tail diamonds drift off the local x axis, which is a
# faithful (not broken) picture: those samples genuinely came from a
# different segment's points.
dqdx_stem_src = ColumnDataSource(data=dict(x=[], y=[]))
DQDX_REF_STYLE = {"muon": ("black", "solid"), "proton": ("saddlebrown", "solid"),
                  "pion": ("seagreen", "dashed"), "kaon": ("purple", "dashed"),
                  "electron": ("gray", "dotted")}
dqdx_ref_src = {name: ColumnDataSource(data=dict(x=[], y=[])) for name in DQDX_REF_STYLE}
# Both-ends mode draws every template TWICE -- once anchored at each physical
# end -- so the panel poses the question ("if this end were the stop, a proton
# would look like that") instead of answering it with the reconstruction's
# direction verdict.  This is the second anchor.
dqdx_ref2_src = {name: ColumnDataSource(data=dict(x=[], y=[]))
                 for name in DQDX_REF_STYLE}

f_dqdx = figure(title="dQ/dx", height=320, width=1150,
               x_range=Range1d(start=0, end=35), y_range=Range1d(start=0, end=100000),
               tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom")
f_dqdx.xaxis.axis_label = "residual range (cm)"
f_dqdx.yaxis.axis_label = "dQ/dx (e/cm)"

DQDX_REF_RENDER = {}
DQDX_REF2_RENDER = {}
for _name, (_color, _dash) in DQDX_REF_STYLE.items():
    _r = f_dqdx.line(x="x", y="y", source=dqdx_ref_src[_name], line_color=_color,
                     line_dash=_dash, line_width=1.6, legend_label=_name, visible=False)
    DQDX_REF_RENDER[_name] = _r
    # No legend entry: it is the same particle hypothesis, mirrored.
    DQDX_REF2_RENDER[_name] = f_dqdx.line(
        x="x", y="y", source=dqdx_ref2_src[_name], line_color=_color,
        line_dash=_dash, line_width=1.6, line_alpha=0.6, visible=False)
f_dqdx.legend.location = "top_right"
f_dqdx.legend.click_policy = "hide"
f_dqdx.legend.label_text_font_size = "8pt"
f_dqdx.legend.background_fill_alpha = 0.6

# Reference lines: do_track_comp's flat-MIP template (End mode) and the
# 1x/2x shower-stem MIP scale (Start mode, e- vs converted-photon).  Span,
# not a line CDS, so it spans the plot regardless of the current x_range.
mip_flat_span = Span(location=50000, dimension="width", line_color="#2ca02c",
                     line_dash="dashed", line_width=1.2, visible=False)
mip1_span = Span(location=43000, dimension="width", line_color="#9467bd",
                 line_dash="dotted", line_width=1.3, visible=False)
mip2_span = Span(location=86000, dimension="width", line_color="#9467bd",
                 line_dash="dashed", line_width=1.3, visible=False)
f_dqdx.add_layout(mip_flat_span)
f_dqdx.add_layout(mip1_span)
f_dqdx.add_layout(mip2_span)

f_dqdx.line(x="x", y="y", source=dqdx_src, line_color="#1f77b4", line_width=1.8)
dqdx_pts_r = f_dqdx.scatter(x="x", y="y", source=dqdx_src, marker="circle", size=5,
                            fill_color="#1f77b4", line_color=None)
f_dqdx.add_tools(HoverTool(renderers=[dqdx_pts_r], tooltips=[
    ("x (cm)", "@x{0.00}"), ("dQ/dx (e/cm)", "@y{0,0}")]))
f_dqdx.scatter(x="x", y="y", source=dqdx_stem_src, marker="diamond", size=9,
              fill_color="#ff7f0e", line_color="#7a3d00", line_width=1)

dqdx_title = Div(text="<b>dQ/dx</b> <span style='color:#666'>&mdash; click a particle-flow "
                      "row above; Start = distance from the shower's start point, "
                      "End = residual range from the stopping end</span>", width=1150)
dqdx_mode = RadioButtonGroup(labels=["Start (shower stem)", "End (track stopping)",
                                     "Both ends (no dirsign)"],
                             active=1, width=400)
dqdx_seg_sel = Select(title="segment", options=[], value="", width=160)
dqdx_caption = Div(text="", width=1150)

# --- every segment at once (sbnd_xin/docs/pr/80 sec 9) ----------------------
# The dQ/dx panel shows ONE segment, reached through a dropdown, so reading
# "which end does each particle stop at" for a 6-prong vertex meant six clicks
# and remembering six numbers.  Three of the five misses in the first blind AI
# scan were exactly that failure -- two Bragg ends in one cluster, only one of
# them noticed.  This table is the whole event's direction evidence on one
# screen.  dQ/dx here is the polarity-free 5 cm end mean, never `rr`.
segtab_src = ColumnDataSource(data=dict(sid=[], cid=[], pdg=[], length=[],
                                        v0=[], d0=[], v1=[], d1=[], verdict=[]))
segtab_view = CDSView(filter=AllIndices())
segtab_title = Div(text="<b>segments</b> <span style='color:#666'>&mdash; dQ/dx "
                        "at each end, 5 cm mean, computed from the fitted points "
                        "and NOT from dirsign/rr</span>", width=1180)
segtab = DataTable(
    source=segtab_src, view=segtab_view, width=1180, height=210,
    index_position=None, selectable=True, sortable=False,
    columns=[
        TableColumn(field="sid", title="seg", width=70),
        TableColumn(field="cid", title="clus", width=50),
        TableColumn(field="pdg", title="pdg", width=60),
        TableColumn(field="length", title="len cm", width=70),
        TableColumn(field="v0", title="vtx A", width=70),
        TableColumn(field="d0", title="dQ/dx @A", width=85),
        TableColumn(field="v1", title="vtx B", width=70),
        TableColumn(field="d1", title="dQ/dx @B", width=85),
        TableColumn(field="verdict", title="stops at", width=260),
    ])


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
          ("dead", "dead (2-D)"), ("dqdx", "dQ/dx"),
          # APPENDED, never inserted -- LAYER_DEFAULT indexes by key but the
          # assert below still pins the historical positions.
          ("arrows", "dQ/dx direction")]
LAYER_KEYS = [k for k, _ in LAYERS]
# Steiner off by default: 6k points per event drawn under everything else is
# noise until you go looking for it.
#
# By KEY, not by position.  This used to be the literal [0, 1, 2, 5, 6], which
# silently re-points to different layers the moment one is inserted rather than
# appended -- the trap that made "dQ/dx" an append-only addition above.  The
# five names below reproduce that literal exactly.
LAYER_DEFAULT = [LAYER_KEYS.index(k) for k in
                 ("trackfit", "shower", "track", "vertices", "dead", "dqdx",
                  "arrows")]
assert LAYER_DEFAULT[:5] == [0, 1, 2, 5, 6], LAYER_DEFAULT
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
# The inverse, plus both pf_pdg_to_name() naming conventions
# (MultiAlgBlobClustering.cxx ~1044: "proton"/"neutron" under prototype_names,
# "p"/"n" otherwise) -- the PF label text carries whichever the mc.json was
# written with, so both must resolve to the same pdg.
NAME_TO_PDG = {"e-": 11, "e+": -11, "mu-": 13, "mu+": -13, "gamma": 22,
              "pi+": 211, "pi-": -211, "proton": 2212, "p": 2212,
              "neutron": 2112, "n": 2112, "K+": 321, "K-": -321, "pi0": 111}
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

# The per-particle energy table (kine_energy_particle et al.): same
# click-to-highlight as the PF table above, joined to a PF node by energy +
# pdg (kine_pf_ids()) since the kine_* arrays carry no id of their own.  A row
# with no match (e.g. a sub-threshold particle the PF tree never gave a node)
# highlights nothing and says so, rather than guessing.
KINE_EMPTY = dict(row=[], pdg=[], ke=[], frm=[], inc=[], pf_id=[])
kine_src = ColumnDataSource(data=dict(KINE_EMPTY))
KINE_VIEW_A, KINE_VIEW_B = AllIndices(), AllIndices()
kine_table = DataTable(
    source=kine_src, view=CDSView(filter=KINE_VIEW_A), width=620, height=220,
    index_position=None, selectable=True, sortable=False,
    columns=[TableColumn(field="pdg", title="pdg", width=70, formatter=_fmt),
             TableColumn(field="ke", title="KE (MeV)", width=90, formatter=_fmt),
             TableColumn(field="frm", title="from", width=70, formatter=_fmt),
             TableColumn(field="inc", title="in Enu", width=60, formatter=_fmt)])
kine_title = Div(text="<b>energy per particle</b> "
                      "<span style='color:#666'>&mdash; &#10003; = counted in reco "
                      "Enu (kine_energy_included == 1); click a row to highlight it "
                      "in all nine panels</span>", width=620)
kine_note = Div(text="", width=620)

feat_div = Div(text="", width=760)
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


# ---------------------------------------------------------------------------
# Neutrino-vertex hand scan (sbnd_xin/docs/pr/75)
# ---------------------------------------------------------------------------
# doc pr/52 sec 5.4 asked the scan to record a 3-D true-vertex POSITION rather
# than a per-event correct/incorrect verdict, because the position is what both
# (a) route classification and (b) a future DeepVtx fine-tune need.  This panel
# produces exactly that.
#
# Candidate scores come from the `vertex_scoreboard` block of the calib dump
# (knob vertex_scoreboard, see the C++ side).  When that block is absent the
# table still works -- it just has no score columns to rank by, and the note
# says so.  An absent scoreboard means NO SCOREBOARD WAS TAKEN; it never means
# "this event has no candidates".

VSCAN_EMPTY = dict(idx=[], pick=[], vid=[], clus=[], x=[], y=[], z=[], deg=[],
                   main=[], cand=[], dl=[], snap=[], rerank=[], trad=[], dmain=[])
vscan_src = ColumnDataSource(data=dict(VSCAN_EMPTY))
# Same DataTable-refresh workaround the overclustering scan viewer needs
# (sbnd_xin/docs/pr/58): Bokeh will not repaint formatted cells when .data is
# swapped for a same-or-shorter row count, so the view filter is flipped after
# every assignment to force it.
vscan_view = CDSView(filter=AllIndices())
vscan_table = DataTable(
    source=vscan_src, view=vscan_view, width=1180, height=260,
    index_position=None, selectable=True, sortable=False,
    columns=[
        TableColumn(field="pick", title="pick", width=45,
                    formatter=HTMLTemplateFormatter(
                        template='<b style="color:#2ca02c"><%= value %></b>')),
        TableColumn(field="vid", title="vtx id", width=65),
        TableColumn(field="clus", title="clus", width=45),
        TableColumn(field="x", title="x", width=60),
        TableColumn(field="y", title="y", width=60),
        TableColumn(field="z", title="z", width=60),
        TableColumn(field="deg", title="deg", width=42),
        # Owner rule 1 as a column: "k/m" = of the m attached segments whose
        # ends are both measured, k get HOTTER going away from this vertex.
        # 86.5% of hand-scanned true vertices read m/m here, against 31.9% of
        # the other vertices in the same cluster (doc pr/80).  Evidence, not a
        # ranking -- the table is never sorted by it.
        TableColumn(field="outg", title="out/meas", width=75),
        TableColumn(field="main", title="main", width=45),
        TableColumn(field="cand", title="cand", width=45),
        TableColumn(field="dl", title="DL score", width=80),
        TableColumn(field="snap", title="snap cm", width=70),
        TableColumn(field="rerank", title="rerank", width=70),
        TableColumn(field="trad", title="trad", width=60),
        TableColumn(field="dmain", title="d(main) cm", width=85),
    ])

# An event has 60-160 PR-graph vertices and 40-120 of them are main-vertex
# CANDIDATES (measured on the nueCC48 dumps), so "candidates" is not a
# scannable set.  The default is the main cluster PLUS every DL-snapped vertex
# wherever it sits -- 4-36 rows on the same events, and by construction it can
# never hide the failure class where the main CLUSTER itself is wrong, because
# every vertex the DL pointed at is in the list regardless of cluster.
VSCAN_FILTERS = ["main cluster + DL", "candidates", "all vertices"]
vscan_filter = Select(title="show", options=VSCAN_FILTERS,
                      value=VSCAN_FILTERS[0], width=175)
VSCAN_SORTS = ["rerank total", "DL score", "trad score", "distance to main",
               "cluster, id"]
vscan_sort = Select(title="rank by", options=VSCAN_SORTS,
                    value=VSCAN_SORTS[0], width=160)

vscan_add_btn = Button(label="add pick", button_type="primary", width=100)
vscan_undo_btn = Button(label="remove last", width=105)
vscan_clear_btn = Button(label="clear picks", width=100)

# Requirement 5: the true vertex need not BE a candidate.  A manual pick is
# doc pr/52's Tier D -- an event no vertex-SELECTION tuning can fix, which must
# be excluded from an acceptance fit rather than fitted against.  Adding one
# sets not_a_candidate on the saved label automatically.
vman_x = TextInput(title="manual x", value="", width=88)
vman_y = TextInput(title="manual y", value="", width=88)
vman_z = TextInput(title="manual z", value="", width=88)
vman_add_btn = Button(label="add manual pick", button_type="warning", width=135)
vman_centre_btn = Button(label="from centre", width=105)
vman_tap = Toggle(label="tap fills coords", active=False, width=135)

VCONF = ["certain", "likely", "unclear"]
vconf_group = RadioButtonGroup(labels=VCONF, active=None, width=250)
vscan_save_btn = Button(label="Save event label", button_type="success", width=150)

vscan_title = Div(text="<b>neutrino vertex hand scan</b> "
                       "<span style='color:#666'>&mdash; click a row to zoom to that "
                       "candidate; <i>add pick</i> records it as your choice "
                       "(first pick = 1st choice)</span>", width=1180)
vscan_picks_div = Div(text="", width=1180)
vscan_note = Div(text="", width=1180)


def vscan_labels_dir():
    return os.path.join(HERE, "..", "vertex_labels", SCAN_TAG)


def vscan_label_path(label):
    return os.path.join(vscan_labels_dir(), "labels-%s.json" % label)


def vscan_tag_has_labels():
    d = vscan_labels_dir()
    return os.path.isdir(d) and bool(glob.glob(os.path.join(d, "labels-*.json")))


def vscan_write_allowed():
    """M13: never append to somebody else's scan record by accident.

    An explicitly-passed --scan-tag is consent.  The implicit default may only
    create a tag, never add to one that already holds labels.
    """
    return SCAN_TAG_EXPLICIT or not vscan_tag_has_labels()


def vscan_load_label(label):
    p = vscan_label_path(label)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def vscan_done_count():
    d = vscan_labels_dir()
    if not os.path.isdir(d):
        return 0
    return sum(1 for lb in LABELS if os.path.isfile(vscan_label_path(lb)))


def vscan_board(d):
    """The scoreboard block, or {} when the knob was off."""
    b = d.get("vertex_scoreboard")
    return b if isinstance(b, dict) and b.get("filled") else {}


def vscan_build_rows(d):
    """One row per PR-graph vertex, joined to the scoreboard on vertex id."""
    board = vscan_board(d)
    by_id = {r["vertex_id"]: r for r in board.get("rows", [])}
    mv = d.get("main_vertex") or {}
    mvp = (mv.get("x"), mv.get("y"), mv.get("z")) if mv else None

    rows = []
    for v in d.get("vertices", []):
        f = v["fit"]
        sb = by_id.get(v["id"])
        dm = (math.dist((f["x"], f["y"], f["z"]), mvp)
              if mvp and None not in mvp else None)
        _away, _meas, _n = vertex_outgoing(d, v["id"])
        rows.append(dict(
            vid=v["id"], clus=v["cluster_id"],
            x=f["x"], y=f["y"], z=f["z"],
            deg=v.get("degree", 0),
            # "-" when nothing attached to this vertex has two measured ends:
            # unmeasured is not zero, and must not read as "nothing points away".
            outg=("%d/%d" % (_away, _meas)) if _meas else "-",
            is_main=bool(v.get("is_main")),
            cand=bool(v.get("main_candidate")),
            # None, not 0: "the DL had no opinion" is not "the DL scored zero".
            dl=(sb["dl_score"] if sb and sb.get("dl_snapped") else None),
            snap=(sb["snap_dis"] if sb and sb.get("dl_snapped") else None),
            rerank=(sb["total"] if sb and sb.get("dl_snapped")
                    and not sb.get("skipped_by_swap_guard") else None),
            trad=(sb["trad_score"] if sb and sb.get("trad_scored") else None),
            dmain=dm,
        ))
    return rows


def _pick_key(p):
    return ("manual", round(p["x"], 3), round(p["y"], 3), round(p["z"], 3)) \
        if p["kind"] == "manual" else ("candidate", p["vertex_id"])


def vscan_refresh_table():
    rows = state.get("vrows") or []
    picks = state.get("vpicks") or []
    rank_of = {}
    for i, p in enumerate(picks):
        if p["kind"] == "candidate":
            rank_of[p["vertex_id"]] = i + 1

    if vscan_filter.value == VSCAN_FILTERS[0]:
        mv = (state.get("data") or {}).get("main_vertex") or {}
        mc = mv.get("cluster_id")
        shown = [r for r in rows
                 if (mc is not None and r["clus"] == mc)
                 or r["dl"] is not None or r["is_main"] or r["vid"] in rank_of]
    elif vscan_filter.value == VSCAN_FILTERS[1]:
        shown = [r for r in rows
                 if r["dl"] is not None or r["trad"] is not None
                 or r["cand"] or r["is_main"] or r["vid"] in rank_of]
    else:
        shown = list(rows)
    # A dump taken without the scoreboard AND without main_vertex_candidate_flag
    # would filter down to nothing; showing everything is more honest than
    # showing an empty table.
    if not shown:
        shown = list(rows)

    key = vscan_sort.value
    BIG = float("inf")
    if key == "rerank total":
        shown.sort(key=lambda r: -(r["rerank"] if r["rerank"] is not None else -BIG))
    elif key == "DL score":
        shown.sort(key=lambda r: -(r["dl"] if r["dl"] is not None else -BIG))
    elif key == "trad score":
        shown.sort(key=lambda r: -(r["trad"] if r["trad"] is not None else -BIG))
    elif key == "distance to main":
        shown.sort(key=lambda r: (r["dmain"] if r["dmain"] is not None else BIG))
    else:
        shown.sort(key=lambda r: (r["clus"], r["vid"]))

    def fmt(v, f="%.2f"):
        return "" if v is None else f % v

    state["vshown"] = shown
    vscan_src.data = dict(
        idx=list(range(len(shown))),
        pick=[str(rank_of.get(r["vid"], "")) for r in shown],
        vid=[r["vid"] for r in shown],
        clus=[r["clus"] for r in shown],
        x=[fmt(r["x"], "%.1f") for r in shown],
        y=[fmt(r["y"], "%.1f") for r in shown],
        z=[fmt(r["z"], "%.1f") for r in shown],
        deg=[r["deg"] for r in shown],
        main=["*" if r["is_main"] else "" for r in shown],
        cand=["y" if r["cand"] else "" for r in shown],
        dl=[fmt(r["dl"], "%.4f") for r in shown],
        snap=[fmt(r["snap"]) for r in shown],
        rerank=[fmt(r["rerank"]) for r in shown],
        trad=[fmt(r["trad"], "%.3f") for r in shown],
        dmain=[fmt(r["dmain"]) for r in shown],
    )
    # doc pr/58 refresh fix -- see the CDSView comment above.
    vscan_view.filter = AllIndices()


def vscan_refresh_picks():
    picks = state.get("vpicks") or []
    pick_src.data = dict(
        x=[p["x"] for p in picks], y=[p["y"] for p in picks],
        z=[p["z"] for p in picks], c=[""] * len(picks),
        tag=[str(i + 1) for i, _ in enumerate(picks)])
    if picks:
        bits = []
        for i, p in enumerate(picks):
            what = ("manual" if p["kind"] == "manual"
                    else "vtx %d (clus %d)" % (p["vertex_id"], p["cluster_id"]))
            bits.append("<b>%d.</b> %s at (%.1f, %.1f, %.1f)"
                        % (i + 1, what, p["x"], p["y"], p["z"]))
        manual = any(p["kind"] == "manual" for p in picks)
        extra = ("&nbsp; <span style='color:#b8860b'>[not_a_candidate -- doc "
                 "pr/52 Tier D, excluded from an acceptance fit]</span>"
                 if manual else "")
        vscan_picks_div.text = "picks: " + " &nbsp;|&nbsp; ".join(bits) + extra
    else:
        vscan_picks_div.text = ("picks: <i>none yet</i> &mdash; select a table row "
                                "and press <i>add pick</i>, or type a position and "
                                "press <i>add manual pick</i>.")


def vscan_refresh_note():
    d = state.get("data") or {}
    board = vscan_board(d)
    bits = []
    if board:
        bits.append("route <b>%s</b>" % html.escape(str(board.get("route", "?"))))
        if board.get("dl_ran"):
            bits.append("DL best %.3f vs accept %.2f"
                        % (board.get("dl_best_score", 0.0),
                           board.get("dl_min_accept_score", 0.0)))
        if board.get("weights_missing"):
            bits.append("<span style='color:#c00'>weights path not found</span>")
    else:
        bits.append("<span style='color:#c00'>no vertex_scoreboard in this dump</span> "
                    "(run with <code>SBND_VERTEX_SCOREBOARD=true</code>) &mdash; "
                    "no scoreboard was taken, which is not the same as "
                    "&ldquo;no candidates&rdquo;")
    saved = state.get("vsaved")
    if state.get("vdirty"):
        bits.append("<b style='color:#c60'>unsaved</b>")
    elif saved:
        bits.append("saved %s" % html.escape(str(saved.get("saved_utc", ""))))
    if not vscan_write_allowed():
        bits.append("<b style='color:#c00'>tag '%s' already holds labels; pass "
                    "--scan-tag explicitly to add to it (M13)</b>" % html.escape(SCAN_TAG))
    bits.append("tag <code>%s</code>, %d/%d events labelled"
                % (html.escape(SCAN_TAG), vscan_done_count(), len(LABELS)))
    vscan_note.text = " &nbsp;&middot;&nbsp; ".join(bits)


def vscan_load(label):
    """Called from load(): rebuild the table and restore any saved label."""
    d = state.get("data") or {}
    state["vrows"] = vscan_build_rows(d)
    state["vpicks"] = []
    state["vdirty"] = False
    selvtx_src.data = dict(EMPTY3)
    vscan_src.selected.indices = []
    # A marker selection left over from the previous event would otherwise point
    # at a vertex id that no longer exists.  Under _vsuspend so clearing it does
    # not re-enter on_vtx_tap.
    state["_vsuspend"] = True
    vtx_src.selected.indices = []
    mainvtx_src.selected.indices = []
    state["_vsuspend"] = False

    saved = vscan_load_label(label)
    state["vsaved"] = saved
    if saved:
        by_id = {r["vid"]: r for r in state["vrows"]}
        for p in saved.get("picks", []):
            q = dict(kind=p.get("kind", "candidate"),
                     vertex_id=p.get("vertex_id"),
                     cluster_id=p.get("cluster_id", -1),
                     x=p["x"], y=p["y"], z=p["z"])
            # Re-read the scores from TODAY's dump for a candidate pick: the
            # saved copy is the record of what was on screen then, and a
            # silent disagreement between the two is worth seeing rather than
            # papering over.
            if q["kind"] == "candidate" and q["vertex_id"] in by_id:
                r = by_id[q["vertex_id"]]
                q.update(cluster_id=r["clus"], x=r["x"], y=r["y"], z=r["z"])
            state["vpicks"].append(q)
        conf = saved.get("confidence")
        state["_vsuspend"] = True
        vconf_group.active = VCONF.index(conf) if conf in VCONF else None
        state["_vsuspend"] = False
    else:
        state["_vsuspend"] = True
        vconf_group.active = None
        state["_vsuspend"] = False

    vman_x.value = vman_y.value = vman_z.value = ""
    vscan_refresh_picks()
    vscan_refresh_table()
    vscan_refresh_note()


def on_vscan_select(attr, old, new):
    shown = state.get("vshown") or []
    if not new or new[0] >= len(shown):
        selvtx_src.data = dict(EMPTY3)
        return
    r = shown[new[0]]
    selvtx_src.data = dict(x=[r["x"]], y=[r["y"]], z=[r["z"]], c=[""], tag=[r["vid"]])
    # Requirement 2: clicking a candidate frames every panel on it.
    #
    # ...but NOT when the selection came from tapping the vertex's own marker in
    # a projection.  You are already looking at it; re-centring would move the
    # picture out from under you, and zoom_btn.active = True would override a
    # framing you set deliberately.  The amber ring above is the confirmation.
    # Clicking a table ROW keeps this behaviour unchanged.
    if state.get("_tap_select"):
        return
    set_centre(r["x"], r["y"], r["z"])
    zoom_btn.active = True
    apply_ranges()


def on_vtx_tap(src, main):
    """Tap a drawn vertex marker -> select its row in the hand-scan table.

    `src` is vtx_src (ordinary PR vertices, whose `tag` column carries the
    vertex id) or mainvtx_src (the single nu-vertex star).  The star's `tag` is
    the CLUSTER id, not a vertex id -- main_vertex has no `id` field at all --
    so `main` switches the lookup to the rows' own is_main flag instead.
    """
    def cb(attr, old, new):
        # Same two reentrancy guards the rest of the file uses.
        if state.get("_vsuspend") or state.get("_suppress_select"):
            return
        if not new:
            return
        rows = state.get("vrows") or []
        if main:
            want = next((r for r in rows if r["is_main"]), None)
        else:
            tags = src.data.get("tag") or []
            if new[0] >= len(tags):
                return
            vid = tags[new[0]]
            want = next((r for r in rows if r["vid"] == vid), None)
        if want is None:
            vscan_note.text = ("<b style='color:#c00'>tapped vertex is not in this "
                               "event's vertex table.</b>")
            return

        def row_of():
            shown = state.get("vshown") or []
            return next((i for i, r in enumerate(shown) if r["vid"] == want["vid"]), None)

        i = row_of()
        switched = False
        if i is None:
            # The default "main cluster + DL" filter shows only 4-36 of an
            # event's 60-160 vertices, so a tapped marker often has no row.  Open
            # the filter up rather than letting the tap do nothing -- and say so,
            # because this changes a control the operator set.
            vscan_filter.value = VSCAN_FILTERS[2]        # "all vertices"
            i = row_of()
            switched = True
        if i is None:
            vscan_note.text = ("<b style='color:#c00'>vertex %s has no row even with "
                               "the filter open.</b>" % want["vid"])
            return

        # Set-and-clear around the assignment, not in the callee: Bokeh fires
        # selected.on_change synchronously here, but it does NOT fire at all when
        # the tapped vertex is already the selected row -- and a flag cleared by
        # the callee would then leak True and silently cost the NEXT genuine
        # table-row click its reframe.
        state["_tap_select"] = True
        try:
            vscan_src.selected.indices = [i]
        finally:
            state["_tap_select"] = False

        shown = state.get("vshown") or []
        def f(v, spec="%.1f"):
            return "-" if v is None else spec % v
        vscan_note.text = (
            "<b>tapped vertex %s</b> &nbsp;&middot;&nbsp; cluster %s "
            "&nbsp;&middot;&nbsp; (%s, %s, %s) cm &nbsp;&middot;&nbsp; rerank %s "
            "&nbsp;&middot;&nbsp; DL %s &nbsp;&middot;&nbsp; <b>row %d of %d</b>%s"
            % (want["vid"], want["clus"], f(want["x"]), f(want["y"]), f(want["z"]),
               f(want["rerank"], "%.2f"), f(want["dl"], "%.4f"), i + 1, len(shown),
               " &nbsp;&middot;&nbsp; <span style='color:#c60'>switched 'show' to "
               "'all vertices' to reach it</span>" if switched else ""))
    return cb


def _vscan_stage(pick):
    picks = state.setdefault("vpicks", [])
    if any(_pick_key(p) == _pick_key(pick) for p in picks):
        return
    picks.append(pick)
    state["vdirty"] = True
    vscan_refresh_picks()
    vscan_refresh_table()
    vscan_refresh_note()


def on_vscan_add():
    sel = vscan_src.selected.indices
    shown = state.get("vshown") or []
    if not sel or sel[0] >= len(shown):
        return
    r = shown[sel[0]]
    _vscan_stage(dict(kind="candidate", vertex_id=r["vid"], cluster_id=r["clus"],
                      x=r["x"], y=r["y"], z=r["z"]))


def on_vman_add():
    try:
        x, y, z = float(vman_x.value), float(vman_y.value), float(vman_z.value)
    except (TypeError, ValueError):
        vscan_note.text = ("<b style='color:#c00'>manual pick needs all three of "
                           "x, y, z in cm.</b>")
        return
    _vscan_stage(dict(kind="manual", vertex_id=None, cluster_id=-1, x=x, y=y, z=z))


def on_vman_centre():
    cx, cy, cz = centre()
    vman_x.value, vman_y.value, vman_z.value = "%.1f" % cx, "%.1f" % cy, "%.1f" % cz


def on_vscan_undo():
    if state.get("vpicks"):
        state["vpicks"].pop()
        state["vdirty"] = True
        vscan_refresh_picks()
        vscan_refresh_table()
        vscan_refresh_note()


def on_vscan_clear():
    if state.get("vpicks"):
        state["vpicks"] = []
        state["vdirty"] = True
        vscan_refresh_picks()
        vscan_refresh_table()
        vscan_refresh_note()


def on_vconf(attr, old, new):
    # vscan_load() assigns .active while restoring a saved label, which fires
    # this callback; without the guard a freshly-loaded, fully-saved event
    # would immediately report itself unsaved.
    if state.get("_vsuspend"):
        return
    state["vdirty"] = True
    vscan_refresh_note()


def vscan_tap(hx, hy):
    """Tap-to-fill: a projection shows two of the three coordinates, so a tap
    in two different panels pins a full 3-D position (requirement 5)."""
    box = {"x": vman_x, "y": vman_y, "z": vman_z}

    def cb(event):
        if not vman_tap.active:
            return
        box[hx].value = "%.1f" % event.x
        box[hy].value = "%.1f" % event.y
    return cb


def on_vscan_save():
    label = state.get("label")
    d = state.get("data") or {}
    if not label:
        return
    if not vscan_write_allowed():
        vscan_note.text = ("<b style='color:#c00'>refusing to write: tag '%s' already "
                           "holds labels and was not passed explicitly (CLAUDE.md M13). "
                           "Restart with --scan-tag %s to continue that scan, or a new "
                           "tag to start a fresh one.</b>" % (html.escape(SCAN_TAG),
                                                             html.escape(SCAN_TAG)))
        return
    picks = state.get("vpicks") or []
    if not picks:
        vscan_note.text = "<b style='color:#c00'>nothing to save: no picks staged.</b>"
        return

    board = vscan_board(d)
    by_id = {r["vertex_id"]: r for r in board.get("rows", [])}
    rows_by_id = {r["vid"]: r for r in (state.get("vrows") or [])}
    mv = d.get("main_vertex") or {}
    mvp = (mv.get("x"), mv.get("y"), mv.get("z")) if mv else None
    meta = d.get("meta", {})

    out_picks = []
    for i, p in enumerate(picks):
        rec = dict(rank=i + 1, kind=p["kind"], vertex_id=p["vertex_id"],
                   cluster_id=p["cluster_id"], x=p["x"], y=p["y"], z=p["z"])
        if mvp and None not in mvp:
            rec["dis_to_main"] = math.dist((p["x"], p["y"], p["z"]), mvp)
        if p["kind"] == "candidate":
            r = rows_by_id.get(p["vertex_id"], {})
            sb = by_id.get(p["vertex_id"], {})
            # Copied INTO the label on purpose: a tuning fit then joins one
            # file per event and never has to re-read the dump.
            rec.update(is_main=r.get("is_main", False),
                       main_candidate=r.get("cand", False),
                       degree=r.get("deg", 0),
                       dl_score=sb.get("dl_score") if sb.get("dl_snapped") else None,
                       snap_dis=sb.get("snap_dis") if sb.get("dl_snapped") else None,
                       rerank_total=sb.get("total") if sb.get("dl_snapped") else None,
                       trad_score=sb.get("trad_score") if sb.get("trad_scored") else None,
                       dl_winner=sb.get("dl_winner", False),
                       trad_winner=sb.get("trad_winner", False))
        out_picks.append(rec)

    doc = dict(
        event=label,
        runNo=meta.get("runNo"), subRunNo=meta.get("subRunNo"),
        eventNo=meta.get("eventNo"),
        source=os.path.realpath(EVENTS[label]),
        arm=os.path.basename(os.path.dirname(os.path.dirname(
            os.path.realpath(EVENTS[label])))),
        scan_tag=SCAN_TAG,
        saved_utc=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        confidence=(VCONF[vconf_group.active]
                    if vconf_group.active is not None else None),
        # doc pr/52 Tier D: a manual pick means the true vertex was not in the
        # candidate set at all, so no vertex-SELECTION tuning can fix it.
        not_a_candidate=any(p["kind"] == "manual" for p in picks),
        main_vertex=(dict(x=mv.get("x"), y=mv.get("y"), z=mv.get("z"),
                          cluster_id=mv.get("cluster_id")) if mv else None),
        route=board.get("route"),
        dl_best_score=board.get("dl_best_score"),
        dl_min_accept_score=board.get("dl_min_accept_score"),
        dl_score_scale=board.get("dl_score_scale"),
        scoreboard_present=bool(board),
        picks=out_picks,
    )

    dirn = vscan_labels_dir()
    os.makedirs(dirn, exist_ok=True)
    path = vscan_label_path(label)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)          # atomic: never leave a half-written record
    state["vsaved"] = doc
    state["vdirty"] = False
    vscan_refresh_note()


vscan_src.selected.on_change("indices", on_vscan_select)
vscan_filter.on_change("value", lambda a, o, n: vscan_refresh_table())
vscan_sort.on_change("value", lambda a, o, n: vscan_refresh_table())
vscan_add_btn.on_click(on_vscan_add)
vscan_undo_btn.on_click(on_vscan_undo)
vscan_clear_btn.on_click(on_vscan_clear)
vman_add_btn.on_click(on_vman_add)
vman_centre_btn.on_click(on_vman_centre)
vconf_group.on_change("active", on_vconf)
vscan_save_btn.on_click(on_vscan_save)
for _f, _hx, _hy in PROJ:
    _f.on_event(Tap, vscan_tap(_hx, _hy))

# Marker tap -> table row.  The three projections share these two CDSs, so one
# binding each covers all of them.  This is additive: the figure-level Tap above
# still fills the manual x/y/z boxes whenever `tap fills coords` is on.
vtx_src.selected.on_change("indices", on_vtx_tap(vtx_src, False))
mainvtx_src.selected.on_change("indices", on_vtx_tap(mainvtx_src, True))


def toggle_layers(attr, old, new):
    on = {LAYERS[i][0] for i in layer_group.active}
    for name, rs in RENDER.items():
        # the 2-D fit follows the 3-D "track fit" toggle
        key = "trackfit" if name == "trackfit2d" else name
        for r in rs:
            r.visible = key in on
    # With the dQ/dx points on top, the per-segment polyline colour underneath
    # competes with the charge ramp for the same pixels, so dim it.  Restored on
    # toggle off, so the picture returns EXACTLY to its pre-dQ/dx appearance.
    # Set here rather than at construction because this is the one place layer
    # state is decided, and it is called once at startup (bottom of the file) --
    # which is also what keeps the alpha correct across an event change, since
    # it lives on the glyph and not in the CDS.
    dim = "dqdx" in on
    for r in RENDER["trackfit"]:
        r.glyph.line_alpha = 0.30 if dim else 0.95


layer_group.on_change("active", toggle_layers)


def on_dqdx_range(attr, old, new):
    """Re-scale the dQ/dx colour ramp.  Bad input leaves the old range alone."""
    try:
        lo, hi = float(dqdx_lo_in.value), float(dqdx_hi_in.value)
    except ValueError:
        dqdx_cbar_note.text = ("<b style='color:#c00'>dQ/dx range needs two "
                               "numbers &mdash; keeping %g to %g.</b>"
                               % (DQDX_CMAP.low, DQDX_CMAP.high))
        return
    if not hi > lo:
        dqdx_cbar_note.text = ("<b style='color:#c00'>dQ/dx max must exceed min "
                               "&mdash; keeping %g to %g.</b>"
                               % (DQDX_CMAP.low, DQDX_CMAP.high))
        return
    DQDX_CMAP.low, DQDX_CMAP.high = lo, hi
    dqdx_cbar_note.text = ("<span style='color:#666'>range %g to %g e/cm "
                           "(fixed; not per-event)</span>" % (lo, hi))


dqdx_lo_in.on_change("value", on_dqdx_range)
dqdx_hi_in.on_change("value", on_dqdx_range)


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


def kine_pf_ids(d, idx, pdgs, ke_list):
    """PF node id for each row of kine_energy_particle, or -1 if none matches.

    The kine_* arrays carry no id of their own -- they are pushed by
    NeutrinoKinematics.cxx's graph walk, a different traversal than the one
    that built the PF tree, so the join happens here rather than in the dump.
    A shower row's value is the exact same float as `showers[].kine_best`
    (both `kine_best / MeV`), so it matches to 0.05 MeV.  A track row's PF
    label instead TRUNCATES to an integer ("25.61 MeV" prints as "25 MeV",
    verified against several events -- not rounded, so `floor(ke) == label`
    is the right test, not nearest-integer).  pdg is required on both sides:
    without it, unrelated particles a few hundred keV apart collide (doc
    pr/26 pr_display click-highlight investigation, evt 489330).  Checked
    against all 48 nueCC48 + 8 loaded mcp1k/nuecc48 events: 0 collisions.
    """
    shower_cands = [(sh["id"], sh["kine_best"], sh.get("particle_id"))
                    for sh in d.get("showers", [])]
    track_cands = []
    flat = idx["flat"]
    for nid in idx["seg_ids"]:
        n = flat.get(nid)
        if not n:
            continue
        m = re.match(r"\s*([A-Za-z][A-Za-z+-]*)\s+(-?\d+)\s*MeV", n["text"])
        if not m:
            continue
        pdg = NAME_TO_PDG.get(m.group(1))
        if pdg is not None:
            track_cands.append((nid, int(m.group(2)), pdg))

    out = []
    for i, e in enumerate(ke_list):
        pdg = pdgs[i] if i < len(pdgs) else None
        matches = ([c for c in shower_cands if c[2] == pdg and abs(c[1] - e) <= 0.05] +
                   [c for c in track_cands if c[2] == pdg and math.floor(e) == c[1]])
        out.append(matches[0][0] if len(matches) == 1 else -1)
    return out


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
    kine_src.selected.indices = []
    if not ke:
        kine_src.data = dict(KINE_EMPTY)
    else:
        idx = state.get("pf_index") or pf_index(d, state.get("pf_nodes") or [])
        pf_ids = kine_pf_ids(d, idx, pdg, ke)
        cols = {k: [] for k in KINE_EMPTY}
        for i in range(len(ke)):
            cols["row"].append(i)
            cols["pdg"].append(PDG_NAME.get(pdg[i] if i < len(pdg) else 0,
                                            str(pdg[i]) if i < len(pdg) else "?"))
            cols["ke"].append("%.1f" % ke[i])
            cols["frm"].append(KINE_METHOD.get(inf[i] if i < len(inf) else -1, "?"))
            cols["inc"].append("✓" if (i < len(inc) and inc[i] == 1) else "")
            cols["pf_id"].append(pf_ids[i])
        kine_src.data = cols
    # The view-filter flip is the ONLY thing that repaints the grid (doc 58).
    kine_table.view.filter = KINE_VIEW_B if kine_table.view.filter is KINE_VIEW_A else KINE_VIEW_A
    kine_note.text = ""

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
# dQ/dx panel (sbnd_xin/docs/pr/42)
# ---------------------------------------------------------------------------
def dqdx_segment_by_id(d, sid):
    return next((s for s in d.get("segments", []) if s["id"] == sid), None)


def dqdx_segment_options(idx, nid):
    """Segment ids to offer in the dropdown for PF node `nid`, start first.

    A shower node's start segment id IS `nid` itself (PrDisplayDump::pf_node_id
    on start_segment(), the same encoding showers[].id and segments[].shower_id
    use) -- so `nid in seg_ids` finds it directly for both track and shower
    nodes; `by_shower` (built by pf_index) gives the rest of a shower's
    segments.  A gamma/pi0 pseudo-node matches neither -- return [].
    """
    ids = list(idx.get("by_shower", {}).get(nid, []))
    if nid in idx.get("seg_ids", set()) and nid not in ids:
        ids.append(nid)
    ids.sort(key=lambda sid: (sid != nid, sid))
    return ids


def _dqdx_valid_points(seg):
    """Points with a defined dQ/dx.  PR::Fit defaults are dQ=-1, dx=0
    (PRCommon.h) and the dump does not emit `index`, the only field
    Fit::valid() checks -- so dx>0 and dQ>=0 is the only client-side guard.
    """
    return [p for p in seg.get("points", [])
            if p.get("dx", 0) > 0 and p.get("dQ", -1) >= 0]


# --------------------------------------------------------------------------
# Polarity-free direction evidence (sbnd_xin/docs/pr/80 sec 9).
#
# Everything below deliberately avoids `dirsign` and `rr`.  examine_direction()
# runs LAST in TaggerCheckNeutrino.cxx:1460 and orients every segment relative
# to the main vertex it has already chosen, and PrDisplayDump.cxx:447-454
# reverses `rr` according to that verdict.  So "the Bragg peak is at the rr=0
# end" is the reconstruction's own answer being read back, and when the
# reconstruction has the direction wrong the End-mode panel shows the peak at
# the wrong end with nothing on screen to catch it.  Arc length recomputed from
# points[] has no convention to get backwards.
BRAGG_WINDOW = 5.0       # cm averaged at each end
MIN_POINTS_END = 3       # fewer than this and the end has NO opinion


def _arclen(pts):
    out, acc, prev = [], 0.0, None
    for p in pts:
        if prev is not None:
            acc += math.dist((p["x"], p["y"], p["z"]),
                             (prev["x"], prev["y"], prev["z"]))
        prev = p
        out.append(acc)
    return out


def seg_end_dqdx(seg, window=BRAGG_WINDOW):
    """Mean dQ/dx within `window` of each physical end of the segment.

    Returns (d_start_end, d_end_end, n0, n1) where "start"/"end" are the
    points[0] / points[-1] ends, i.e. the start_vertex_id / end_vertex_id ends.
    A mean is None when the end is UNMEASURED -- kept distinct from "low",
    because collapsing the two is how a Bragg test silently inverts on a short
    or badly fitted segment.
    """
    pts = seg.get("points") or []
    if len(pts) < 2:
        return (None, None, 0, 0)
    s = _arclen(pts)
    total = s[-1]
    lo, hi = [], []
    for si, p in zip(s, pts):
        if not (p.get("dx", 0) > 0 and p.get("dQ", -1) >= 0):
            continue
        if si <= window:
            lo.append(p["dQ"] / p["dx"])
        if total - si <= window:
            hi.append(p["dQ"] / p["dx"])
    d0 = sum(lo) / len(lo) if len(lo) >= MIN_POINTS_END else None
    d1 = sum(hi) / len(hi) if len(hi) >= MIN_POINTS_END else None
    return (d0, d1, len(lo), len(hi))


def vertex_outgoing(d, vid, ratio=1.3):
    """Owner rule 1 as a measurement: of the segments attached to this vertex,
    how many get HOTTER going away from it (i.e. stop somewhere else)?

    Returns (n_away, n_measured, n_attached).  Measured on 481 hand-scan labels
    (doc pr/80): 86.5% of the owner's vertices have every attached track
    pointing away, against 31.9% of the other vertices in the same cluster --
    the strongest single discriminator found.  It is shown as EVIDENCE, next to
    the other columns; it is not a ranking and the table is not sorted by it.
    """
    away = meas = n = 0
    for seg in d.get("segments", []):
        ends = (seg.get("start_vertex_id"), seg.get("end_vertex_id"))
        if vid not in ends:
            continue
        n += 1
        d0, d1, _, _ = seg_end_dqdx(seg)
        if d0 is None or d1 is None or d0 <= 0 or d1 <= 0:
            continue
        near, far = (d0, d1) if vid == ends[0] else (d1, d0)
        meas += 1
        if far / near >= ratio:
            away += 1
    return (away, meas, n)


def _dqdx_both_xy(seg):
    """Both-ends mode: dQ/dx vs raw arc length from the points[0] end.

    No `rr`, no `dirsign`.  The x axis runs from the start_vertex_id end to the
    end_vertex_id end, both named in the caption, so the reader decides which
    end is the stop instead of being told.
    """
    pts = seg.get("points") or []
    s = _arclen(pts)
    xs, ys = [], []
    for si, p in zip(s, pts):
        if p.get("dx", 0) > 0 and p.get("dQ", -1) >= 0:
            xs.append(si)
            ys.append(p["dQ"] / p["dx"])
    return xs, ys, ""


def _dqdx_end_xy(seg):
    """End mode: residual range from the dumped `rr`, oriented by dirsign.

    Trust the dump's `rr` for dirsign != 0 -- PrDisplayDump.cxx orients it
    correctly for BOTH +1 and -1 (a -1 direction means the stopping end sits
    at fits[0], so rr=L IS the residual range there; do not "correct" it by
    recomputing L.back()-L[i], that reverses the Bragg peak).  Only dirsign==0
    is genuinely ambiguous (never observed in practice, but handled rather
    than silently mis-plotted): fall back to raw arc length from fits()[0].
    """
    if seg.get("dirsign", 0) == 0:
        xs, ys = [], []
        acc, prev = 0.0, None
        for p in seg.get("points", []):
            if prev is not None:
                acc += math.dist((p["x"], p["y"], p["z"]), (prev["x"], prev["y"], prev["z"]))
            prev = p
            if p.get("dx", 0) > 0 and p.get("dQ", -1) >= 0:
                xs.append(acc)
                ys.append(p["dQ"] / p["dx"])
        return xs, ys, "direction undetermined (dirsign=0) -- x axis is raw arc length, not residual range"
    xs, ys = [], []
    for p in _dqdx_valid_points(seg):
        rr = p.get("rr", -1)
        if rr < 0:            # the -0.1 sentinel at a branching vertex end
            continue
        xs.append(rr)
        ys.append(p["dQ"] / p["dx"])
    return xs, ys, ""


def _dqdx_start_xy(d, seg):
    """Start mode: distance from the shower's own start point.

    Orient the segment's own point order (fits() order, unrelated to `rr`)
    by whichever end sits nearer showers[].start; recomputed from x/y/z here
    rather than reusing `rr`, since `rr` is defined toward the STOPPING end
    and Start mode wants the opposite end.
    """
    pts_all = seg.get("points", [])
    if not pts_all:
        return [], [], ""
    showers = {sh["id"]: sh for sh in d.get("showers", [])}
    sh = showers.get(seg["id"])
    note = ""
    reverse = False
    if sh and sh.get("start"):
        s = sh["start"]
        d0 = math.dist((pts_all[0]["x"], pts_all[0]["y"], pts_all[0]["z"]), (s["x"], s["y"], s["z"]))
        d1 = math.dist((pts_all[-1]["x"], pts_all[-1]["y"], pts_all[-1]["z"]), (s["x"], s["y"], s["z"]))
        reverse = d1 < d0
    else:
        note = "no shower row for this segment -- distance measured from its own fits()[0]"
    L = [0.0] * len(pts_all)
    acc = 0.0
    for i in range(1, len(pts_all)):
        acc += math.dist((pts_all[i]["x"], pts_all[i]["y"], pts_all[i]["z"]),
                         (pts_all[i - 1]["x"], pts_all[i - 1]["y"], pts_all[i - 1]["z"]))
        L[i] = acc
    if reverse:
        L = [L[-1] - v for v in L]
    xs, ys = [], []
    for i, p in enumerate(pts_all):
        if p.get("dx", 0) > 0 and p.get("dQ", -1) >= 0:
            xs.append(L[i])
            ys.append(p["dQ"] / p["dx"])
    return xs, ys, note


def clear_dqdx(msg=""):
    dqdx_src.data = dict(x=[], y=[])
    dqdx_stem_src.data = dict(x=[], y=[])
    for src in dqdx_ref_src.values():
        src.data = dict(x=[], y=[])
    for r in DQDX_REF_RENDER.values():
        r.visible = False
    for span in (mip_flat_span, mip1_span, mip2_span):
        span.visible = False
    dqdx_caption.text = ("<span style='color:#a33'>%s</span>" % msg) if msg else ""


def replot_dqdx():
    d = state.get("data") or {}
    sid = state.get("dqdx_seg_id")
    seg = dqdx_segment_by_id(d, sid) if sid is not None else None
    if seg is None:
        clear_dqdx("no segment selected")
        return

    is_start = (dqdx_mode.active == 0)
    is_both = (dqdx_mode.active == 2)
    meta = d.get("meta", {})
    mip_med = meta.get("mip_dqdx_median", 43000.0)
    mip_flat = meta.get("mip_dqdx_flat", 50000.0)

    if is_both:
        xs, ys, note = _dqdx_both_xy(seg)
    elif is_start:
        xs, ys, note = _dqdx_start_xy(d, seg)
    else:
        xs, ys, note = _dqdx_end_xy(seg)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs, ys = [xs[i] for i in order], [ys[i] for i in order]
    dqdx_src.data = dict(x=xs, y=ys)

    seg_len = _arclen(seg.get("points") or [])[-1:] or [0.0]
    seg_len = seg_len[0]
    ref = d.get("dqdx_ref")
    for name, src in dqdx_ref_src.items():
        show = (not is_start) and ref and name in ref
        DQDX_REF_RENDER[name].visible = bool(show)
        DQDX_REF2_RENDER[name].visible = bool(show and is_both)
        if show:
            grid = ref["grid"]
            gxs = [grid["start"] + i * grid["step"] for i in range(grid["n"])]
            if is_both:
                # Anchor one copy at each physical end, clipped to the segment.
                keep = [(g, v) for g, v in zip(gxs, ref[name]) if g <= seg_len]
                src.data = dict(x=[seg_len - g for g, _ in keep],
                                y=[v for _, v in keep])
                dqdx_ref2_src[name].data = dict(x=[g for g, _ in keep],
                                                y=[v for _, v in keep])
            else:
                src.data = dict(x=gxs, y=ref[name])
                dqdx_ref2_src[name].data = dict(x=[], y=[])
        else:
            src.data = dict(x=[], y=[])
            dqdx_ref2_src[name].data = dict(x=[], y=[])

    mip_flat_span.location = mip_flat
    mip_flat_span.visible = not is_start
    mip1_span.location = mip_med
    mip1_span.visible = is_start
    mip2_span.location = 2 * mip_med
    mip2_span.visible = is_start

    if is_start:
        stem = (next((sh for sh in d.get("showers", []) if sh["id"] == seg["id"]), {})
                or {}).get("stem_dqdx", [])
        n = min(len(stem), len(xs))
        dqdx_stem_src.data = dict(x=xs[:n], y=[v * mip_med for v in stem[:n]])
    else:
        dqdx_stem_src.data = dict(x=[], y=[])

    if is_both:
        f_dqdx.xaxis.axis_label = (
            "arc length from the vtx-%s end (cm)  ->  vtx-%s end   "
            "[recomputed, NOT residual range]"
            % (seg.get("start_vertex_id"), seg.get("end_vertex_id")))
        default_x = (0, max(seg_len * 1.02, 1.0))
    elif is_start:
        f_dqdx.xaxis.axis_label = "distance from start (cm)"
        default_x = (0, 20)
    else:
        f_dqdx.xaxis.axis_label = "residual range (cm)"
        default_x = (0, 35)
    f_dqdx.x_range.start, f_dqdx.x_range.end = default_x
    ymax_candidates = list(ys) + ([2.2 * mip_med] if is_start else [1.1 * mip_flat])
    if not is_start and ref:
        for name in ("muon", "proton"):
            if name in ref:
                ymax_candidates.append(max(ref[name]))
    ymax = max(ymax_candidates) if ymax_candidates else 100000.0
    f_dqdx.y_range.start, f_dqdx.y_range.end = 0, max(ymax, 1.0) * 1.15

    pdg = seg.get("particle_id")
    caption = (
        "segment <b>%d</b> &nbsp; pdg <b>%s</b> &nbsp; score <b>%s</b> &nbsp; "
        "dirsign <b>%s</b> &nbsp; dir_weak <b>%s</b> &nbsp; length <b>%s cm</b> "
        "&nbsp; n pts <b>%d</b>"
        % (seg["id"], PDG_NAME.get(pdg, str(pdg)), _f(seg.get("particle_score"), "%.2f"),
           seg.get("dirsign", "&mdash;"), seg.get("dir_weak", "&mdash;"),
           _f(seg.get("length"), "%.1f"), len(xs)))
    # The polarity-free reading, always shown -- including in End mode, where
    # it is the one line that can contradict `dirsign` on screen.
    _d0, _d1, _n0, _n1 = seg_end_dqdx(seg)
    caption += (
        "<br><span style='color:#333'>end dQ/dx (5 cm mean, no dirsign): "
        "vtx <b>%s</b> end <b>%s</b> (n=%d) &nbsp;|&nbsp; vtx <b>%s</b> end "
        "<b>%s</b> (n=%d) &nbsp;&rarr;&nbsp; %s</span>"
        % (seg.get("start_vertex_id"), _f(_d0, "%.0f"), _n0,
           seg.get("end_vertex_id"), _f(_d1, "%.0f"), _n1,
           ("hotter at the vtx-%s end" % seg.get("end_vertex_id")
            if _d0 and _d1 and _d1 / _d0 >= 1.3 else
            "hotter at the vtx-%s end" % seg.get("start_vertex_id")
            if _d0 and _d1 and _d0 / _d1 >= 1.3 else
            "no separation between the ends -- this segment has no opinion")))
    if note:
        caption += " &nbsp; <span style='color:#a33'>%s</span>" % note
    caption += (
        "<div style='color:#666;font-size:85%'>reference curves: dQ/dx (e/cm) from "
        "ParticleDataSet after Modified-Box recombination at 0.5 kV/cm, including the "
        "retained undocumented 0.85 scale factor; ElectronDeDx is held flat into "
        "rr&rarr;0 (particle_dataset.jsonnet). The dump's residual range differs from "
        "the PID's own by +0.15cm minus a 0-1cm offset -- negligible except right at "
        "the Bragg peak.</div>")
    dqdx_caption.text = caption


def set_dqdx_node(nid):
    """Point the panel at PF node `nid` (or clear it if None/unmatched)."""
    d = state.get("data") or {}
    idx = state.get("pf_index") or {}
    state["dqdx_node_id"] = nid
    if nid is None or nid < 0:
        dqdx_seg_sel.options = []
        dqdx_seg_sel.value = ""
        state["dqdx_seg_id"] = None
        clear_dqdx("")
        return
    opts = dqdx_segment_options(idx, nid)
    if not opts:
        dqdx_seg_sel.options = []
        dqdx_seg_sel.value = ""
        state["dqdx_seg_id"] = None
        clear_dqdx("no segment for this node (pseudo gamma/pi0 node -- no charge to plot)")
        return
    dqdx_seg_sel.options = [(str(sid), "%d%s" % (sid, "  (start)" if sid == nid else ""))
                            for sid in opts]
    default_sid = nid if nid in opts else opts[0]
    is_shower = nid in idx.get("by_shower", {})
    state["dqdx_seg_id"] = default_sid
    dqdx_mode.active = 0 if is_shower else 1
    dqdx_seg_sel.value = str(default_sid)
    replot_dqdx()


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
    if state.get("_suppress_select"):
        return
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
    set_dqdx_node(nid)
    pf_note.text = ("selected <b>%s</b> (id %d) &rarr; %d segment(s) highlighted"
                    % (pf_src.data["kind"][i], nid, len(picked)))
    # A click here supersedes any energy-table highlight; clear it without
    # bouncing back through on_kine_select and erasing what was just set.
    state["_suppress_select"] = True
    kine_src.selected.indices = []
    kine_note.text = ""
    state["_suppress_select"] = False


def on_pf_clear():
    pf_src.selected.indices = []
    kine_src.selected.indices = []
    set_highlight(set())
    set_dqdx_node(None)
    pf_note.text = ""
    kine_note.text = ""


def on_kine_select(attr, old, new):
    if state.get("_suppress_select"):
        return
    if not new:
        set_highlight(set())
        kine_note.text = ""
        return
    i = new[0]
    pf_ids = kine_src.data["pf_id"]
    if i >= len(pf_ids):
        return
    pf_id = pf_ids[i]
    if pf_id is None or pf_id < 0:
        set_highlight(set())
        set_dqdx_node(None)
        kine_note.text = ("row %d: no matching reconstructed particle-flow node "
                          "&mdash; nothing to highlight" % i)
    else:
        d = state.get("data") or {}
        picked = highlight_ids(d, state.get("pf_nodes") or [], pf_id,
                               state.get("pf_index"))
        set_highlight(picked)
        set_dqdx_node(pf_id)
        kine_note.text = ("selected <b>%s</b>, %s MeV (id %d) &rarr; %d segment(s) highlighted"
                          % (kine_src.data["pdg"][i], kine_src.data["ke"][i], pf_id, len(picked)))
    state["_suppress_select"] = True
    pf_src.selected.indices = []
    pf_note.text = ""
    state["_suppress_select"] = False


def fill_segtab(d):
    """Every segment's two-end dQ/dx on one screen (doc pr/80 sec 9)."""
    rows = sorted(d.get("segments", []),
                  key=lambda s: (-s.get("length", 0.0), s["id"]))
    sid, cid, pdg, ln, v0, d0c, v1, d1c, ver = [], [], [], [], [], [], [], [], []
    for s in rows:
        a, b, _, _ = seg_end_dqdx(s)
        A, B = s.get("start_vertex_id"), s.get("end_vertex_id")
        if a is None or b is None or a <= 0 or b <= 0:
            v = "unmeasured -- no opinion"
        elif b / a >= 1.3:
            v = "stops at vtx %s  (x%.2f)" % (B, b / a)
        elif a / b >= 1.3:
            v = "stops at vtx %s  (x%.2f)" % (A, a / b)
        else:
            v = "flat -- no separation"
        # Plain text, not _f(): these columns carry no HTML formatter, so an
        # "&mdash;" would render as those eight literal characters.
        def _n(x, fmt="%.0f"):
            return (fmt % x) if x is not None else "-"
        sid.append(s["id"]); cid.append(s["cluster_id"])
        pdg.append(PDG_NAME.get(s.get("particle_id"), str(s.get("particle_id"))))
        ln.append(_n(s.get("length"), "%.1f"))
        v0.append(A); d0c.append(_n(a))
        v1.append(B); d1c.append(_n(b))
        ver.append(v)
    segtab_src.data = dict(sid=sid, cid=cid, pdg=pdg, length=ln, v0=v0,
                           d0=d0c, v1=v1, d1=d1c, verdict=ver)
    # Same repaint workaround the hand-scan table needs (doc pr/58).
    segtab_view.filter = AllIndices()


def fill_arrows(d, frac=0.35, cap=12.0):
    """One arrow per segment, at its cooler end, pointing at its hotter end."""
    shafts = {k: dict(xs=[], ys=[]) for k in ("xy", "yz", "xz")}
    heads = {k: dict(x=[], y=[], angle=[]) for k in ("xy", "yz", "xz")}
    for s in d.get("segments", []):
        pts = s.get("points") or []
        if len(pts) < 3:
            continue
        a, b, _, _ = seg_end_dqdx(s)
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        if max(b / a, a / b) < 1.3:
            continue                    # no separation => no arrow, on purpose
        L = _arclen(pts)
        span = min(cap, max(frac * L[-1], 1.0))
        if b > a:                       # hotter at points[-1]: travel 0 -> -1
            tail = pts[0]
            head = next((p for si, p in zip(L, pts) if si >= span), pts[-1])
        else:
            tail = pts[-1]
            head = next((p for si, p in zip(reversed(L), reversed(pts))
                         if L[-1] - si >= span), pts[0])
        for key, hx, hy in (("xy", "x", "y"), ("yz", "z", "y"), ("xz", "x", "z")):
            x0, y0 = tail[hx], tail[hy]
            x1, y1 = head[hx], head[hy]
            if x0 == x1 and y0 == y1:
                continue                # degenerate in THIS projection only
            shafts[key]["xs"].append([x0, x1])
            shafts[key]["ys"].append([y0, y1])
            heads[key]["x"].append(x1)
            heads[key]["y"].append(y1)
            # Bokeh's triangle marker points at +y, so subtract a quarter turn.
            heads[key]["angle"].append(math.atan2(y1 - y0, x1 - x0)
                                       - math.pi / 2.0)
    for key in ("xy", "yz", "xz"):
        arrow_src[key].data = shafts[key]
        arrowhead_src[key].data = heads[key]


def load(label):
    """Read one event's calib JSON and push every layer to its CDS."""
    path = EVENTS[label]
    with open(path) as fh:
        d = json.load(fh)
    state["label"] = label
    state["data"] = d

    # Hand-scan panel first: it resets the picks, so a stale pick from the
    # previous event can never be saved against this one.
    vscan_load(label)

    meta = d["meta"]
    # (apa, face) -> ticks per slice, for pt -> slice
    nps = {(r["apa"], r["face"]): r["nticks_per_slice"]
           for r in meta.get("nticks_per_slice", [])}

    # --- per-segment direction table + on-canvas arrows ---------------------
    fill_segtab(d)
    fill_arrows(d)

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
    # The same walk also fills the dQ/dx point layer: one row per fitted POINT,
    # split into "has a measurement" and "does not" (see fitpt_src above).
    fp = dict(x=[], y=[], z=[], dqdx=[], dQ=[], dx=[], rr=[], sid=[], cid=[], pid=[])
    fn = dict(x=[], y=[], z=[], sid=[])
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
        for p in s["points"]:
            # Same validity guard as _dqdx_valid_points() -- kept as one
            # expression in both places on purpose, so the 3-D colouring and the
            # 1-D panel can never disagree about which points are measured.
            if p.get("dx", 0) > 0 and p.get("dQ", -1) >= 0:
                fp["x"].append(p["x"]); fp["y"].append(p["y"]); fp["z"].append(p["z"])
                fp["dqdx"].append(p["dQ"] / p["dx"])
                fp["dQ"].append(p["dQ"]); fp["dx"].append(p["dx"])
                # rr < 0 is the -0.1 sentinel at a branching vertex end.  It
                # disqualifies a point from the residual-range AXIS of the 1-D
                # panel, but says nothing about dQ/dx, so such points are
                # coloured normally here and carry the sentinel into the hover.
                fp["rr"].append(p.get("rr", -1))
                fp["sid"].append(s["id"]); fp["cid"].append(s["cluster_id"])
                fp["pid"].append(s["particle_id"])
            else:
                fn["x"].append(p["x"]); fn["y"].append(p["y"]); fn["z"].append(p["z"])
                fn["sid"].append(s["id"])
    for key in cols:
        seg_src[key].data = cols[key]
    fitpt_src.data = fp
    fitpt_nodq_src.data = fn

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
    set_dqdx_node(None)
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
        key = ("pu", "pv", "pw")[pl]
        nps = {(r["apa"], r["face"]): r["nticks_per_slice"]
               for r in d.get("meta", {}).get("nticks_per_slice", [])}
        # doc sbnd_xin/docs/pr/75: GROW the search box until two fitted points
        # are found rather than falling back to the panel's full extent.  The
        # old fallback made the 2-D view useless exactly where it is most
        # wanted -- on an isolated micro-stub candidate (the doc pr/51 class),
        # which by construction has no fitted points within +-h.
        ws, ss = [], []
        for grow in (1.0, 2.0, 4.0, 8.0):
            ws, ss = [], []
            hh = h * grow
            for s in d.get("segments", []):
                for p in s["points"]:
                    if p["apa"] != apa:
                        continue
                    if (abs(p["x"] - cx) > hh or abs(p["y"] - cy) > hh
                            or abs(p["z"] - cz) > hh):
                        continue
                    ws.append(p[key])
                    ss.append(p["pt"] / nps.get((apa, p["face"]), 1))
            if len(ws) >= 2:
                break
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


def on_dqdx_mode(attr, old, new):
    replot_dqdx()


def on_dqdx_seg(attr, old, new):
    state["dqdx_seg_id"] = int(new) if new else None
    replot_dqdx()


event_select.on_change("value", on_event)
prev_btn.on_click(step(-1))
next_btn.on_click(step(+1))
zoom_btn.on_change("active", on_zoom)
for w in (cx_in, cy_in, cz_in, half_in):
    w.on_change("value", on_centre)
vtx_btn.on_click(on_vertex)
pf_src.selected.on_change("indices", on_pf_select)
pf_clear_btn.on_click(on_pf_clear)
kine_src.selected.on_change("indices", on_kine_select)
bdt_toggle.on_change("active", on_bdt)
dqdx_mode.on_change("active", on_dqdx_mode)
dqdx_seg_sel.on_change("value", on_dqdx_seg)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
header = Div(text="<h2>SBND PR event display</h2>", width=1400)
controls = row(event_select, prev_btn, next_btn, Spacer(width=20),
               zoom_btn, cx_in, cy_in, cz_in, half_in, vtx_btn)

# The hand-scan panel (sbnd_xin/docs/pr/75) sits directly under the
# projections it drives: clicking a candidate reframes all nine panels, so the
# table and the pictures have to be visible at once.  It is a member of the
# LEFT column, not a full-width row beneath both columns -- as a full-width row
# it had to clear the much taller right column (PF/kine tables + the 320 px
# dQ/dx figure), which opened a screen-high empty band under the projections.
vscan_col = column(
    vscan_title,
    row(vscan_sort, vscan_filter, Spacer(width=20),
        vscan_add_btn, vscan_undo_btn, vscan_clear_btn),
    vscan_table,
    row(vman_x, vman_y, vman_z, vman_centre_btn, vman_tap,
        Spacer(width=15), vman_add_btn),
    row(Div(text="<b>confidence</b>", width=85), vconf_group,
        Spacer(width=25), vscan_save_btn),
    vscan_picks_div,
    vscan_note,
)

# Left column: the 3-D projections + run controls + the hand-scan panel.
# Right column: the particle-flow / kine tables, tagger info, and the dQ/dx
# panel -- side by side rather than stacked, so a wide browser window isn't
# left half-empty.
left_col = column(
    row(f_xy, f_yz, f_xz),
    dqdx_cbar_fig,
    row(dqdx_lo_in, dqdx_hi_in, Spacer(width=15), dqdx_cbar_note),
    row(layer_group),
    controls,
    info,
    vscan_col,
)
right_col = column(
    row(column(pf_title, pf_table, kine_title, kine_table,
              row(pf_clear_btn), pf_note, kine_note),
        Spacer(width=20),
        column(feat_div, cos_div, bdt_toggle, bdt_div)),
    column(dqdx_title, row(dqdx_mode, dqdx_seg_sel), f_dqdx, dqdx_caption),
    column(segtab_title, segtab),
)

_rows = [
    header,
    row(left_col, Spacer(width=30), right_col),
]
# Wire-plane panels: hidden by default (sbnd_xin/docs/pr/42 -- not useful for
# day-to-day PID work, replaced by the dQ/dx panel above), --wire-planes
# restores them.  Construction and data-filling are unchanged either way;
# this only decides whether the row is part of the served document.
if SHOW_WIRE_PLANES:
    _rows.append(
        row(column(panel[(0, 0)]["fig"], panel[(0, 1)]["fig"], panel[(0, 2)]["fig"]),
            column(panel[(1, 0)]["fig"], panel[(1, 1)]["fig"], panel[(1, 2)]["fig"]),
            cbar_fig))
_rows.append(status)
layout = column(*_rows)

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

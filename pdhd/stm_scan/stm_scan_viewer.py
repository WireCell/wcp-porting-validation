#!/usr/bin/env python3
"""PDHD stopping-muon (STM) hand-scan display -- doc pdhd/docs/stm-tagger-chain.md sec 13.

Forked BY DUPLICATION from pdhd/ql_scan/ql_scan_viewer.py (the PDHD Bokeh scan
app; pdvd/ql_display is a different, non-Bokeh analysis dir).  ql_scan_viewer.py
is untouched.

WHAT THIS IS FOR
  retile_wrapped_channel_activity changes the STM verdict on 224 of 2246
  clusters (doc sec 12).  This app shows each of those clusters as raw charge in
  three projections and records a human STM / THRU / UNCLEAR label.  The labels
  are then scored against both arms (score_stm_scan.py) to decide whether the
  knob should be flipped.

THE BLIND IS STRUCTURAL, NOT AN INSTRUCTION
  This process opens ONLY the 'clustering-global' and 'channel-deadarea' members
  of mabc-pr.zip.  Those two are byte-identical between the knob-off (stm0) and
  knob-on (stmw) arms -- verified by zip-member SHA-256 -- so the pixels cannot
  encode which arm you are looking at.  steiner_graph / steiner_terminals /
  stm_fit / stm_tagged are NEVER read: not loaded and hidden, simply not opened.
  The answer key (pdhd_retile_scan_key.tsv) is never read by this process
  either, and the stratum flag is deliberately absent from the UI -- you can see
  the point count, and labelling a fragment 'UNCLEAR' has to be your call, not a
  nudge from a label that says 'small'.

WHAT YOU ARE JUDGING
  From the charge alone: does this track enter the detector and STOP inside the
  active volume (STM), or does it pass through / exit a face (THRU)?  UNCLEAR is
  a real answer -- fragment, too sparse, overlapping, genuinely ambiguous.

  The GREY points are the rest of the event's charge, decimated for speed.  They
  are the reason this is a display and not a table: a track that continues into
  a neighbouring cluster is THRU even though the coloured cluster looks to stop.
  Grey is a charge-independent decimation of ALL other charge -- it is not
  "clusters the tagger considered", which would leak the answer.

  Panels default to the FULL detector volume with the active boundary drawn.
  A cluster auto-zoomed to its own extent looks contained in every projection,
  which would invert THRU/STM judgements; 'Zoom to cluster' is available but is
  not the default.

USAGE
  ./serve_stm_scan.sh 5017                       # then http://localhost:5017/stm_scan_viewer
  ./serve_stm_scan.sh 5017 --tag retile0         # namespace the saved labels
"""
import sys
import os
import json
import csv
import zipfile

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, Select, Button, Div, TextInput,
                          Toggle, ColorBar, LinearColorMapper)
from bokeh.plotting import figure

HERE = os.path.dirname(os.path.abspath(__file__))
PDHD = os.path.dirname(HERE)
WORK = os.path.join(PDHD, "work")
SHEET = os.path.join(PDHD, "docs", "scan", "pdhd_retile_scan_sheet.tsv")

# PDHD active volume (cm) -- the pr.jsonnet BoxFiducial, doc sec 7.1.
VOL = dict(x=(-357.985, 357.985), y=(7.61, 606.0), z=(0.234, 462.297))
CONTEXT_MAX = 8000          # decimated grey points per event, per the perf note
ARM_DIR = "stm0"            # either arm serves: the two layers read are identical


def parse_args(argv):
    tag = "retile0"
    if "--tag" in argv:
        i = argv.index("--tag")
        if i + 1 < len(argv):
            tag = argv[i + 1]
    return tag


SCAN_TAG = parse_args(sys.argv[1:])
LABEL_DIR = os.path.join(WORK, "stm_scan_labels", SCAN_TAG)
os.makedirs(LABEL_DIR, exist_ok=True)
LABEL_FILE = os.path.join(LABEL_DIR, "labels.json")


# ---------------------------------------------------------------------------
# the item list -- from the sheet, which carries no verdict from either arm
# ---------------------------------------------------------------------------
def load_items():
    rows = []
    with open(SHEET) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    for r in csv.DictReader(lines, delimiter="\t"):
        rows.append(dict(scan_id=int(r["scan_id"]), tranche=int(r["tranche"]),
                         event=r["event"], cluster=int(r["cluster"]),
                         npts=int(r["npts"]), length=float(r["length_cm"])))
    rows.sort(key=lambda r: r["scan_id"])
    return rows


ITEMS = load_items()
if not ITEMS:
    raise SystemExit("no scan items found in %s" % SHEET)


# ---------------------------------------------------------------------------
# charge, straight out of the Bee zip.  ONLY the two blind-safe members.
# ---------------------------------------------------------------------------
_cache = {}


def event_charge(event):
    """(x, y, z, q, cluster_id) for one event, from clustering-global only."""
    if event in _cache:
        return _cache[event]
    zp = os.path.join(WORK, "029107_%s_%s" % (event, ARM_DIR), "mabc-pr.zip")
    if not os.path.exists(zp):
        _cache[event] = None
        return None
    z = zipfile.ZipFile(zp)
    names = [n for n in z.namelist() if "clustering-global" in n]
    if not names:
        _cache[event] = None
        return None
    d = json.loads(z.read(names[0]))
    out = (np.asarray(d["x"], float), np.asarray(d["y"], float),
           np.asarray(d["z"], float), np.asarray(d["q"], float),
           np.asarray(d["cluster_id"], int))
    _cache[event] = out
    return out


# ---------------------------------------------------------------------------
# labels: one file keyed by "<event>/<cluster>", written on EVERY click
# ---------------------------------------------------------------------------
def item_key(it):
    return "%s/%d" % (it["event"], it["cluster"])       # cluster_id is per-event


def load_labels():
    if not os.path.isfile(LABEL_FILE):
        return {}
    try:
        with open(LABEL_FILE) as fh:
            return json.load(fh).get("labels", {})
    except (OSError, ValueError):
        return {}


LABELS = load_labels()          # append-only: an existing file is loaded, never truncated


def save_labels():
    tmp = LABEL_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"scan": "retile_wrapped_channel_activity",
                   "doc": "pdhd/docs/stm-tagger-chain.md sec 13",
                   "tag": SCAN_TAG, "sheet": os.path.relpath(SHEET, PDHD),
                   "labels": LABELS}, fh, indent=1)
    os.replace(tmp, LABEL_FILE)          # atomic: a crash mid-write keeps the old file


# ---------------------------------------------------------------------------
# figures -- built ONCE, sources updated on navigation (bokeh 3 note in the doc)
# ---------------------------------------------------------------------------
CTX = ColumnDataSource(dict(a=[], b=[]))
TGT = ColumnDataSource(dict(a=[], b=[], q=[]))
SRC = {}
FIGS = {}
PANELS = [("z", "y", "side view:  Z (beam) vs Y (vertical)"),
          ("z", "x", "top view:   Z (beam) vs X (drift)"),
          ("x", "y", "end view:   X (drift) vs Y (vertical)")]
cmap = LinearColorMapper(palette="Viridis256", low=0, high=1)

for ha, va, title in PANELS:
    ctx = ColumnDataSource(dict(a=[], b=[]))
    tgt = ColumnDataSource(dict(a=[], b=[], q=[]))
    f = figure(title=title, height=330, width=470, match_aspect=True,
               tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom",
               x_axis_label="%s [cm]" % ha.upper(), y_axis_label="%s [cm]" % va.upper())
    f.scatter("a", "b", source=ctx, size=1.5, color="#b9b9b9", alpha=0.45)
    f.scatter("a", "b", source=tgt, size=3.5,
              color={"field": "q", "transform": cmap}, alpha=0.95)
    # the active boundary, so "does it reach a face" is answerable by eye
    x0, x1 = VOL[ha]
    y0, y1 = VOL[va]
    f.line([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
           color="#d62728", line_width=1.2, line_dash="dashed")
    SRC[(ha, va)] = (ctx, tgt)
    FIGS[(ha, va)] = f


# ---------------------------------------------------------------------------
# widgets
# ---------------------------------------------------------------------------
def item_option(it):
    mark = LABELS.get(item_key(it), {}).get("label", "")
    return "%3d %s  evt %s cl %d  n=%d  %.0f cm%s" % (
        it["scan_id"], "*" if mark else " ", it["event"], it["cluster"],
        it["npts"], it["length"], ("   [%s]" % mark) if mark else "")


item_select = Select(title="Scan item", value=item_option(ITEMS[0]),
                     options=[item_option(i) for i in ITEMS], width=430)
prev_btn = Button(label="< prev", width=90)
next_btn = Button(label="next >", width=90)
next_unl_btn = Button(label="next unlabelled >>", width=150)
stm_btn = Button(label="STM  (stops inside)", button_type="success", width=200)
thru_btn = Button(label="THRU (through-going / exits)", button_type="primary", width=230)
uncl_btn = Button(label="UNCLEAR", button_type="warning", width=120)
clear_btn = Button(label="clear this label", width=130)
zoom_tog = Toggle(label="Zoom to cluster", width=140)
notes = TextInput(title="notes (optional)", width=430)
progress = Div(text="", width=430)
status = Div(text="", width=940)
header = Div(width=940, text="""
<b>PDHD stopping-muon hand scan</b> &mdash; gate on
<code>retile_wrapped_channel_activity</code> (doc stm-tagger-chain &sect;13).
<br>From the <b>charge alone</b>: does this track enter and <b>stop</b> inside the
active volume (<b>STM</b>), or does it pass through / exit a face (<b>THRU</b>)?
<b>UNCLEAR</b> is a real answer &mdash; fragment, too sparse, overlapping, ambiguous.
<br><span style="color:#555">Coloured = the cluster in question (colour is its charge).
Grey = all other charge in the event, decimated &mdash; use it to see whether the track
continues into a neighbouring cluster. Red dashed = the active boundary. Panels show the
full detector by default on purpose: a cluster zoomed to its own extent looks contained
in every view.</span>
""")

state = dict(idx=0)


def current():
    return ITEMS[state["idx"]]


def render():
    it = current()
    ch = event_charge(it["event"])
    if ch is None:
        status.text = ("<b style='color:#b00'>missing</b> %s -- no mabc-pr.zip for event %s"
                       % (item_key(it), it["event"]))
        for ha, va in SRC:
            SRC[(ha, va)][0].data = dict(a=[], b=[])
            SRC[(ha, va)][1].data = dict(a=[], b=[], q=[])
        return
    X, Y, Z, Q, C = ch
    m = C == it["cluster"]
    other = ~m
    # charge-independent decimation of the context: every Nth point, no cuts
    n_other = int(other.sum())
    step = max(1, n_other // CONTEXT_MAX)
    oi = np.flatnonzero(other)[::step]
    axes = dict(x=X, y=Y, z=Z)
    qt = Q[m]
    cmap.low = float(qt.min()) if qt.size else 0.0
    cmap.high = float(qt.max()) if qt.size else 1.0
    for ha, va in SRC:
        ctx, tgt = SRC[(ha, va)]
        ctx.data = dict(a=axes[ha][oi], b=axes[va][oi])
        tgt.data = dict(a=axes[ha][m], b=axes[va][m], q=qt)
        f = FIGS[(ha, va)]
        if zoom_tog.active and m.any():
            pad = 20.0
            f.x_range.start = float(axes[ha][m].min() - pad)
            f.x_range.end = float(axes[ha][m].max() + pad)
            f.y_range.start = float(axes[va][m].min() - pad)
            f.y_range.end = float(axes[va][m].max() + pad)
        else:
            f.x_range.start, f.x_range.end = VOL[ha][0] - 20, VOL[ha][1] + 20
            f.y_range.start, f.y_range.end = VOL[va][0] - 20, VOL[va][1] + 20
    rec = LABELS.get(item_key(it), {})
    notes.value = rec.get("notes", "")
    done = sum(1 for i in ITEMS if item_key(i) in LABELS)
    t1 = [i for i in ITEMS if i["tranche"] == 1]
    d1 = sum(1 for i in t1 if item_key(i) in LABELS)
    progress.text = ("<b>%d / %d</b> labelled overall &nbsp;|&nbsp; tranche 1: "
                     "<b>%d / %d</b> &nbsp;|&nbsp; this item: <b>%s</b>"
                     % (done, len(ITEMS), d1, len(t1), rec.get("label", "&mdash;")))
    status.text = ("event %s, cluster %d &mdash; %d points, %.0f cm, %d grey context "
                   "points (1 in %d)" % (it["event"], it["cluster"], it["npts"],
                                         it["length"], len(oi), step))


def refresh_options():
    opts = [item_option(i) for i in ITEMS]
    item_select.options = opts
    item_select.value = opts[state["idx"]]


def go(idx, keep_notes=False):
    if not keep_notes:
        pass
    state["idx"] = max(0, min(len(ITEMS) - 1, idx))
    refresh_options()
    render()


def set_label(lab):
    it = current()
    LABELS[item_key(it)] = dict(label=lab, notes=notes.value,
                                scan_id=it["scan_id"], event=it["event"],
                                cluster=it["cluster"], npts=it["npts"],
                                length_cm=it["length"])
    save_labels()                     # every click, not only on Save
    refresh_options()
    render()
    nxt = next((k for k in range(state["idx"] + 1, len(ITEMS))
                if item_key(ITEMS[k]) not in LABELS), None)
    if nxt is not None:
        go(nxt)


def clear_label():
    LABELS.pop(item_key(current()), None)
    save_labels()
    refresh_options()
    render()


def on_select(attr, old, new):
    if new in item_select.options:
        go(item_select.options.index(new))


def next_unlabelled():
    nxt = next((k for k in range(state["idx"] + 1, len(ITEMS))
                if item_key(ITEMS[k]) not in LABELS), None)
    if nxt is None:
        nxt = next((k for k in range(0, len(ITEMS))
                    if item_key(ITEMS[k]) not in LABELS), None)
    if nxt is None:
        status.text = "<b>every item is labelled.</b>"
    else:
        go(nxt)


item_select.on_change("value", on_select)
prev_btn.on_click(lambda: go(state["idx"] - 1))
next_btn.on_click(lambda: go(state["idx"] + 1))
next_unl_btn.on_click(next_unlabelled)
stm_btn.on_click(lambda: set_label("STM"))
thru_btn.on_click(lambda: set_label("THRU"))
uncl_btn.on_click(lambda: set_label("UNCLEAR"))
clear_btn.on_click(clear_label)
zoom_tog.on_click(lambda a: render())
notes.on_change("value", lambda a, o, n: None)

curdoc().add_root(column(
    header,
    row(item_select, prev_btn, next_btn, next_unl_btn),
    row(stm_btn, thru_btn, uncl_btn, clear_btn, zoom_tog),
    row(notes, progress),
    row(*[FIGS[(ha, va)] for ha, va, _ in PANELS]),
    status,
))
curdoc().title = "PDHD STM hand scan (%s)" % SCAN_TAG
go(0)

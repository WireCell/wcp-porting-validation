#!/usr/bin/env python3
"""PDHD wrapped-channel MOVER hand-scan display -- doc pdhd/docs/05_mover-scan-display.md.

Forked BY DUPLICATION from pdhd/stm_scan/stm_scan_viewer.py (doc pdhd/04 sec 13
of the older chain doc); that file is untouched and still serves its own scan.

WHAT THIS IS FOR
  Fixing the wrapped-strip channel lookup (`wrapped_channel_charge`, doc pdhd/04
  sec 9) moves the cosmic-tagger verdict on 39 objects across the six-event
  manifest (doc pdhd/04 sec 10.3).  This app shows each of those objects as raw
  charge in three projections and records a human label from the doc pdhd/04
  sec 11.3 alphabet.  The labels are then scored against the PRE-REGISTERED bar
  of doc pdhd/04 sec 11.5 -- by `docs/scripts/d04_movers_score.py`, which this
  directory does NOT re-implement, so "the bar was not re-tuned after the labels
  were seen" stays a checkable fact rather than a claim.

  It replaces the Bee set for this scan.  Bee shows the charge but makes you
  hunt for the object, remember the label alphabet, and write the answer into a
  TSV by hand; here the object is on screen with its ends marked and the eight
  choices are buttons.

THE BLIND IS STRUCTURAL, AND IT IS AN *ABSENCE* ARGUMENT -- SO IT IS PROVEN
  The older stm_scan blinded by byte-identity: the two layers it drew were
  SHA-256-identical between its two arms, so the pixels could not encode the
  arm.  **That argument does not hold here and is not used**: `q` differs on
  141 434 of 161 854 points in event 12.  This scan is blind for a different
  reason -- it draws ONE arm, so there is no "which arm is this" for the pixels
  to encode, and the thing being tested (which DIRECTION each verdict moved) is
  not in the charge at all.  It lives in `bee-pr-run029107-d04movers.KEY.tsv`,
  which this process never opens.

  That makes the blind an absence: no tagger layer is read.  `stm`, `stm_fit`,
  `stm_tagged`, `steiner_graph` and `steiner_terminals` are in the same zip and
  are simply never opened.  `selftest_d05_scan.py` asserts exactly that, by
  enumerating the members this module actually reads.

  The stratum (`>=1000` / `200-1000` / `<200` points) is deliberately absent
  from the UI, as it is in stm_scan.  You can see the point count; calling a
  small object UNCLEAR has to be your judgement, not a nudge from a label.

WHICH ARM IS DRAWN, AND THE ONE OBJECT WHERE IT MATTERS
  ARM = d05mON, the fixed (production) arm -- also the arm the scan sheet's
  npts came from, verified row by row.  For 565 of the 574 tagger-evaluated
  clusters the two arms' clusters are the same point set to the bit, so the
  choice is cosmetic.  For nine it is not, and **one of those nine is on this
  sheet**: event 20 cluster 31 has 11 432 points here and 11 065 in the
  pre-fix arm.  That row is judged as the ON-arm object.  (doc pdhd/05 sec 4.)

WHAT YOU ARE JUDGING
  From the charge alone, judge the WHOLE OBJECT -- the coloured cluster together
  with any grey charge that continues along the same trajectory:

    THRU       BOTH ends leave the active volume (or die in a dead region at it)
    STOP       ONE end enters, the other stops inside the active volume
    CONT       neither end reaches a boundary
    FRAG>...   the cluster is only PART of the object; the verdict is for the FULL object
    MESSY      several particles merged, or a shower: "does it stop" is ill-posed
    UNCLEAR    you cannot tell

  FRAG> carries the FULL object's verdict on purpose: a fragment of a
  through-goer and a fragment of a stopper are opposite physics truths, and one
  undivided FRAG bucket throws the second kind away.

  GREY = all other charge in the event.  With 'Dense context' on (the default)
  EVERY grey point within the context radius of the cluster is drawn and the
  rest of the event is thinned 1-in-N -- a track that continues into a
  neighbouring cluster is a fragment, not a stopper, and global decimation alone
  renders that continuation as a handful of dots.  Both selections are purely
  geometric over ALL other charge; neither is "what the tagger considered",
  which would leak the answer.

  The two CLOSE-UP panels show a window around each end with NOTHING thinned --
  every point in the window, cluster and grey alike.  A stopper's end is charge
  that simply ends; a fragment's end has grey charge carrying on along the same
  line.  Global decimation cannot show that difference, which is why it is a
  separate pair of panels rather than a zoom of the overview.

  Panels default to the FULL detector volume with the active boundary drawn.  A
  cluster auto-zoomed to its own extent looks contained in every projection,
  which would invert THRU/STOP judgements; 'Zoom to object' exists but is not
  the default.

USAGE
  ./serve_d05_scan.sh 5017                  # then http://localhost:5017/d05_scan_viewer
  ./serve_d05_scan.sh 5017 --tag pass2      # namespace the saved labels

  Port 5017 is SHARED with pdhd/stm_scan -- run one or the other, not both.
"""
import sys
import os
import json
import csv
import re
import zipfile

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, Select, Button, Div, TextInput,
                          Toggle, LinearColorMapper)
from bokeh.plotting import figure
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
PDHD = os.path.dirname(HERE)
WORK = os.path.join(PDHD, "work")
SHEET = os.path.join(PDHD, "bee-pr-run029107-d04movers.sheet.tsv")
RUN = "029107"
ARM = "d05mON"              # the fixed arm; see the module docstring
RE_DEAD = re.compile(r"\d+-channel-deadarea-[A-Za-z0-9]+-[A-Za-z0-9]+\.json")

# PDHD active volume (cm) -- the pr.jsonnet BoxFiducial, doc pdhd/04 sec 7.1.
VOL = dict(x=(-357.985, 357.985), y=(7.61, 606.0), z=(0.234, 462.297))
CONTEXT_MAX = 8000          # decimated grey points per event
DENSE_R_DEFAULT = 40.0      # cm: radius around the cluster for the dense context
DENSE_MAX = 25000           # cap on the dense set, decimated uniformly if over
END_PCT = 0.5               # percentile along the principal axis defining an "end"

# The scan alphabet.  These strings are written verbatim into the exported sheet
# and MUST match d04_movers_score.py's VALID set -- an unrecognised label makes
# that scorer exit 2 rather than fall through into a class, which is the point.
CHOICES = ["THRU", "STOP", "CONT",
           "FRAG>THRU", "FRAG>STOP", "FRAG>CONT",
           "MESSY", "UNCLEAR"]


def parse_args(argv):
    tag = "movers0"
    if "--tag" in argv:
        i = argv.index("--tag")
        if i + 1 < len(argv):
            tag = argv[i + 1]
    return tag


SCAN_TAG = parse_args(sys.argv[1:])
LABEL_DIR = os.path.join(WORK, "d05_scan_labels", SCAN_TAG)
os.makedirs(LABEL_DIR, exist_ok=True)
LABEL_FILE = os.path.join(LABEL_DIR, "labels.json")
FILLED_SHEET = os.path.join(LABEL_DIR, "filled_sheet.tsv")


# ---------------------------------------------------------------------------
# the item list -- from the BLIND sheet, which carries no direction
# ---------------------------------------------------------------------------
def load_items():
    with open(SHEET) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    rows = [dict(bee_idx=int(r["bee_idx"]), event=r["event"],
                 cluster=int(r["cluster"]), npts=int(r["npts"]))
            for r in csv.DictReader(lines, delimiter="\t")]
    rows.sort(key=lambda r: (int(r["event"]), r["cluster"]))
    for i, r in enumerate(rows):
        r["scan_id"] = i
    return rows


ITEMS = load_items()
if not ITEMS:
    raise SystemExit("no scan items found in %s" % SHEET)


# ---------------------------------------------------------------------------
# charge, straight out of the Bee zip.  ONLY the blind-safe members: the
# clustering layer and the dead-area polygons.  The tagger layers (stm,
# stm_fit, stm_tagged, steiner_graph, steiner_terminals) are in the same zip and
# are never opened -- selftest_d05_scan.py asserts it over MEMBERS_READ.
# ---------------------------------------------------------------------------
MEMBERS_READ = set()        # every zip member this process has opened, for the selftest
_cache = {}
_dead_cache = {}


def _zip_path(event):
    return os.path.join(WORK, "%s_%s_%s" % (RUN, event, ARM), "mabc-pr.zip")


def event_charge(event):
    """(x, y, z, q, cluster_id) for one event, from clustering-global only."""
    if event in _cache:
        return _cache[event]
    zp = _zip_path(event)
    if not os.path.exists(zp):
        _cache[event] = None
        return None
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist()
                 if os.path.basename(n).endswith("-clustering-global.json")]
        if not names:
            _cache[event] = None
            return None
        MEMBERS_READ.add(names[0])
        d = json.loads(z.read(names[0]))
    out = (np.asarray(d["x"], float), np.asarray(d["y"], float),
           np.asarray(d["z"], float), np.asarray(d["q"], float),
           np.asarray(d["cluster_id"], int))
    _cache[event] = out
    return out


def event_dead(event):
    """Dead-channel polygons as (ys, zs) lists.  Bee writes them as [y, z] pairs,
    so they overlay the SIDE view (Z vs Y) and only that one -- they carry no x
    extent, and faking one into the top/end views would invent information."""
    if event in _dead_cache:
        return _dead_cache[event]
    zp = _zip_path(event)
    if not os.path.exists(zp):
        _dead_cache[event] = ([], [])
        return _dead_cache[event]
    ys, zs = [], []
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            # exact-shape match, not a substring: this filter is the blind's
            # only moving part, so it must not glob wider than intended.
            if not RE_DEAD.fullmatch(os.path.basename(n)):
                continue
            MEMBERS_READ.add(n)
            for poly in json.loads(z.read(n)).get("polygons", []):
                ys.append([float(p[0]) for p in poly])
                zs.append([float(p[1]) for p in poly])
    _dead_cache[event] = (ys, zs)
    return _dead_cache[event]


# ---------------------------------------------------------------------------
# context selection.  BOTH paths are purely geometric over ALL other charge.
# ---------------------------------------------------------------------------
def context_index(P, mask, dense, radius):
    other = np.flatnonzero(~mask)
    if other.size == 0:
        return other, 1, 0
    step = max(1, other.size // CONTEXT_MAX)
    far = other[::step]                      # charge-independent global thinning
    if not dense:
        return far, step, 0
    tgt = P[mask]
    if tgt.size == 0:
        return far, step, 0
    d, _ = cKDTree(tgt).query(P[other], k=1, distance_upper_bound=radius)
    near = other[np.isfinite(d)]
    if near.size > DENSE_MAX:
        near = near[::max(1, near.size // DENSE_MAX)]
    return np.union1d(far, near), step, int(near.size)


# ---------------------------------------------------------------------------
# the object's two ends, and how far each is from the active boundary.
# This is the arithmetic a scanner would otherwise do by squinting at an axis;
# it is measured off the CHARGE, and it decides nothing -- THRU/STOP/CONT is
# still a judgement about the whole object including the grey continuation.
# ---------------------------------------------------------------------------
def ends_of(C):
    """C is (n,3) in (x,y,z).  Returns (A, B, length) or None."""
    if C.shape[0] < 2:
        return None
    mu = C.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(C - mu, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    t = (C - mu) @ Vt[0]
    lo, hi = np.percentile(t, [END_PCT, 100.0 - END_PCT])
    A = C[np.argmin(np.abs(t - lo))]
    B = C[np.argmax(t)] if hi <= lo else C[np.argmin(np.abs(t - hi))]
    return A, B, float(np.linalg.norm(B - A))


def wall_gap(p):
    """(distance to the nearest active-volume face, which face) for one point."""
    best, who = None, ""
    for i, ax in enumerate("xyz"):
        for j, edge in enumerate(VOL[ax]):
            d = abs(float(p[i]) - edge)
            if best is None or d < best:
                best, who = d, "%s%s" % (ax, "-" if j == 0 else "+")
    return best, who


def dead_gap(p, dead):
    """Distance in the (y, z) plane from a point to the nearest dead-channel
    polygon vertex, or None if the event has no dead area.  Reported, not acted
    on: 'the end is 2 cm from a dead region' is for the scanner to weigh."""
    ys, zs = dead
    if not ys:
        return None
    V = np.array([[y, z] for py, pz in zip(ys, zs) for y, z in zip(py, pz)])
    return float(np.min(np.hypot(V[:, 0] - p[1], V[:, 1] - p[2])))


# ---------------------------------------------------------------------------
# labels: one file keyed by "<event>/<cluster>", written on EVERY click
# ---------------------------------------------------------------------------
def item_key(it):
    return "%s/%d" % (it["event"], it["cluster"])


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
        json.dump({"scan": "wrapped_channel_charge movers",
                   "doc": "pdhd/docs/05_mover-scan-display.md",
                   "bar": "pdhd/docs/04_stm-tagger-scan.md sec 11.5",
                   "arm": ARM, "tag": SCAN_TAG,
                   "sheet": os.path.relpath(SHEET, PDHD),
                   "labels": LABELS}, fh, indent=1)
    os.replace(tmp, LABEL_FILE)          # atomic: a crash mid-write keeps the old file


def export_sheet(path=FILLED_SHEET):
    """Write a NEW sheet with the label column filled.  The committed blind sheet
    is never touched (M13: a record is not a scratch file)."""
    with open(SHEET) as fh:
        src = fh.readlines()
    out = []
    for line in src:
        if line.startswith("#") or line.startswith("bee_idx\t"):
            out.append(line)
            continue
        f = line.rstrip("\n").split("\t")
        while len(f) < 6:
            f.append("")
        rec = LABELS.get("%s/%s" % (f[1], int(f[2])), {})
        f[4] = rec.get("label", "")
        f[5] = (rec.get("notes", "") or "").replace("\t", " ")
        out.append("\t".join(f) + "\n")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.writelines(out)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# figures -- built ONCE, sources updated on navigation (bokeh 3: no JS callbacks)
# ---------------------------------------------------------------------------
SRC = {}
FIGS = {}
PANELS = [("z", "y", "side view:  Z (beam) vs Y (vertical)"),
          ("z", "x", "top view:   Z (beam) vs X (drift)"),
          ("x", "y", "end view:   X (drift) vs Y (vertical)")]
cmap = LinearColorMapper(palette="Viridis256", low=0, high=1)

for ha, va, title in PANELS:
    ctx = ColumnDataSource(dict(a=[], b=[]))
    tgt = ColumnDataSource(dict(a=[], b=[], q=[]))
    end = ColumnDataSource(dict(a=[], b=[], lab=[]))
    dead = ColumnDataSource(dict(xs=[], ys=[]))
    f = figure(title=title, height=360, width=480, match_aspect=True,
               tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom",
               x_axis_label="%s [cm]" % ha.upper(), y_axis_label="%s [cm]" % va.upper())
    if (ha, va) == ("z", "y"):
        f.patches("xs", "ys", source=dead, color="#7f7f7f", alpha=0.55,
                  line_color="#7f7f7f")
    f.scatter("a", "b", source=ctx, size=1.5, color="#b9b9b9", alpha=0.45)
    f.scatter("a", "b", source=tgt, size=3.5,
              color={"field": "q", "transform": cmap}, alpha=0.95)
    f.scatter("a", "b", source=end, size=16, marker="circle_x",
              line_color="#d62728", line_width=2.5, fill_alpha=0.0)
    # the active boundary, so "does it reach a face" is answerable by eye
    x0, x1 = VOL[ha]
    y0, y1 = VOL[va]
    f.line([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
           color="#d62728", line_width=1.2, line_dash="dashed")
    SRC[(ha, va)] = (ctx, tgt, end, dead)
    FIGS[(ha, va)] = f

# Two CLOSE-UPS, one per end, in the plane chosen below.  These are the panels
# that decide most of these labels: inside the window EVERY point is drawn --
# no global thinning, no dense-context radius -- so "the charge stops here" and
# "the charge carries on into the next cluster" look different.
ZOOM = {}
ZOOM_SRC = {}
for nm in ("A", "B"):
    ctx = ColumnDataSource(dict(a=[], b=[]))
    tgt = ColumnDataSource(dict(a=[], b=[], q=[]))
    end = ColumnDataSource(dict(a=[], b=[]))
    f = figure(title="end %s close-up" % nm, height=360, width=480, match_aspect=True,
               tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom")
    f.scatter("a", "b", source=ctx, size=4.0, color="#9a9a9a", alpha=0.75)
    f.scatter("a", "b", source=tgt, size=5.0,
              color={"field": "q", "transform": cmap}, alpha=0.95)
    f.scatter("a", "b", source=end, size=18, marker="circle_x",
              line_color="#d62728", line_width=2.5, fill_alpha=0.0)
    ZOOM[nm] = f
    ZOOM_SRC[nm] = (ctx, tgt, end)


# ---------------------------------------------------------------------------
# widgets
# ---------------------------------------------------------------------------
def item_option(it):
    mark = LABELS.get(item_key(it), {}).get("label", "")
    return "%3d %s  evt %-2s cl %-4d n=%-6d%s" % (
        it["scan_id"], "*" if mark else " ", it["event"], it["cluster"],
        it["npts"], ("   [%s]" % mark) if mark else "")


item_select = Select(title="Object", value="", options=[], width=430)
prev_btn = Button(label="< prev", width=90)
next_btn = Button(label="next >", width=90)
next_unl_btn = Button(label="next unlabelled >>", width=160)

BTN = {}
BTN["THRU"] = Button(label="THRU  (both ends leave)", button_type="primary", width=230)
BTN["STOP"] = Button(label="STOP  (one end stops inside)", button_type="success", width=250)
BTN["CONT"] = Button(label="CONT  (neither end reaches a wall)", button_type="success", width=280)
BTN["FRAG>THRU"] = Button(label="FRAG → THRU", button_type="primary", width=230)
BTN["FRAG>STOP"] = Button(label="FRAG → STOP", button_type="success", width=250)
BTN["FRAG>CONT"] = Button(label="FRAG → CONT", button_type="success", width=280)
BTN["MESSY"] = Button(label="MESSY (not one track)", button_type="warning", width=200)
BTN["UNCLEAR"] = Button(label="UNCLEAR", button_type="warning", width=130)

clear_btn = Button(label="clear this label", width=140)
export_btn = Button(label="export filled sheet", width=160)
zoom_tog = Toggle(label="Zoom to object", width=150)
dense_tog = Toggle(label="Dense context", width=140, active=True)
plane_sel = Select(title="", value="zy", width=190,
                   options=[("zy", "close-ups: Z vs Y (side)"),
                            ("zx", "close-ups: Z vs X (top)"),
                            ("xy", "close-ups: X vs Y (end)")])
win_sel = Select(title="", value="25", width=170,
                 options=[("15", "close-up ± 15 cm"), ("25", "close-up ± 25 cm"),
                          ("50", "close-up ± 50 cm"), ("100", "close-up ± 100 cm")])
radius_sel = Select(title="", value="40", width=150,
                    options=[("20", "context r = 20 cm"), ("40", "context r = 40 cm"),
                             ("80", "context r = 80 cm"), ("150", "context r = 150 cm")])
notes = TextInput(title="note (optional, free text -- e.g. which end is in a dead region)",
                  width=560)
progress = Div(text="", width=560)
status = Div(text="", width=1480)
ends_div = Div(text="", width=1480)
header = Div(width=1480, text="""
<b>PDHD wrapped-channel mover scan</b> &mdash;
39 objects whose cosmic-tagger verdict moved when the wrapped-strip charge lookup was
fixed (doc pdhd/04 &sect;10.3, &sect;11; display doc pdhd/05).
<br><b>Judge the whole object, including the grey continuation</b> &mdash; not just the
coloured points. Where does that object begin and end?
<br><span style="color:#555">Top row: the coloured cluster <i>is</i> the whole object.
Middle row: it is only a piece of one (under-clustering) &mdash; still say what the
<b>full</b> object does. <b>MESSY</b> = several particles or a shower, so the question is
ill-posed. <b>UNCLEAR</b> = you genuinely cannot tell.</span>
<br><span style="color:#555">Coloured = the cluster (colour is its charge, rescaled per
object). Grey = all other charge in the event; with <i>Dense context</i> on,
<b>every</b> grey point within the context radius is drawn, so a continuation off an end
is visible rather than thinned to a few dots. Red circles = the two ends along the
object's principal axis. Red dashed = the active boundary. Dark grey patches in the side
view = dead channels (they carry no drift extent, so they are shown there only).
Panels show the full detector by default on purpose: an object zoomed to its own extent
looks contained in every view.</span>
<br><span style="color:#777">This display never opens the tagger layers, and the
direction each verdict moved is not in this process. Labels are saved on every click to
<code>work/d05_scan_labels/%s/labels.json</code>.</span>
""" % SCAN_TAG)

state = dict(idx=0)


def current():
    return ITEMS[state["idx"]]


def render():
    it = current()
    ch = event_charge(it["event"])
    if ch is None:
        status.text = ("<b style='color:#b00'>missing</b> %s -- no mabc-pr.zip at %s"
                       % (item_key(it), _zip_path(it["event"])))
        for k in SRC:
            SRC[k][0].data = dict(a=[], b=[])
            SRC[k][1].data = dict(a=[], b=[], q=[])
            SRC[k][2].data = dict(a=[], b=[], lab=[])
        return
    X, Y, Z, Q, C = ch
    m = C == it["cluster"]
    P = np.c_[X, Y, Z]
    radius = float(radius_sel.value)
    oi, step, n_near = context_index(P, m, bool(dense_tog.active), radius)
    dead = event_dead(it["event"])
    axes = dict(x=X, y=Y, z=Z)
    qt = Q[m]
    cmap.low = float(qt.min()) if qt.size else 0.0
    cmap.high = float(qt.max()) if qt.size else 1.0
    e = ends_of(P[m])
    for ha, va in SRC:
        ctx, tgt, endsrc, deadsrc = SRC[(ha, va)]
        ctx.data = dict(a=axes[ha][oi], b=axes[va][oi])
        tgt.data = dict(a=axes[ha][m], b=axes[va][m], q=qt)
        ax3 = dict(x=0, y=1, z=2)
        if e is None:
            endsrc.data = dict(a=[], b=[], lab=[])
        else:
            endsrc.data = dict(a=[float(e[0][ax3[ha]]), float(e[1][ax3[ha]])],
                               b=[float(e[0][ax3[va]]), float(e[1][ax3[va]])],
                               lab=["A", "B"])
        if (ha, va) == ("z", "y"):
            deadsrc.data = dict(xs=dead[1], ys=dead[0])
        f = FIGS[(ha, va)]
        if zoom_tog.active and m.any():
            pad = 30.0
            f.x_range.start = float(axes[ha][m].min() - pad)
            f.x_range.end = float(axes[ha][m].max() + pad)
            f.y_range.start = float(axes[va][m].min() - pad)
            f.y_range.end = float(axes[va][m].max() + pad)
        else:
            f.x_range.start, f.x_range.end = VOL[ha][0] - 20, VOL[ha][1] + 20
            f.y_range.start, f.y_range.end = VOL[va][0] - 20, VOL[va][1] + 20
    # close-ups: every point inside the window, cluster and context alike
    ha2, va2 = plane_sel.value[0], plane_sel.value[1]
    half = float(win_sel.value)
    idx = dict(x=0, y=1, z=2)
    for nm, p in (("A", None if e is None else e[0]), ("B", None if e is None else e[1])):
        czt, tzt, ezt = ZOOM_SRC[nm]
        f = ZOOM[nm]
        f.xaxis.axis_label = "%s [cm]" % ha2.upper()
        f.yaxis.axis_label = "%s [cm]" % va2.upper()
        if p is None:
            czt.data = dict(a=[], b=[])
            tzt.data = dict(a=[], b=[], q=[])
            ezt.data = dict(a=[], b=[])
            continue
        win = (np.abs(X - p[0]) <= half) & (np.abs(Y - p[1]) <= half) & \
              (np.abs(Z - p[2]) <= half)
        czt.data = dict(a=axes[ha2][win & ~m], b=axes[va2][win & ~m])
        tzt.data = dict(a=axes[ha2][win & m], b=axes[va2][win & m], q=Q[win & m])
        ezt.data = dict(a=[float(p[idx[ha2]])], b=[float(p[idx[va2]])])
        f.x_range.start = float(p[idx[ha2]]) - half
        f.x_range.end = float(p[idx[ha2]]) + half
        f.y_range.start = float(p[idx[va2]]) - half
        f.y_range.end = float(p[idx[va2]]) + half

    rec = LABELS.get(item_key(it), {})
    notes.value = rec.get("notes", "")
    done = sum(1 for i in ITEMS if item_key(i) in LABELS)
    progress.text = ("<b>%d / %d</b> labelled &nbsp;|&nbsp; this object: <b>%s</b>"
                     % (done, len(ITEMS), rec.get("label", "&mdash;")))
    if e is None:
        ends_div.text = "<span style='color:#777'>too few points to define an axis.</span>"
    else:
        A, B, L = e
        parts = []
        for nm, p in (("A", A), ("B", B)):
            g, face = wall_gap(p)
            dg = dead_gap(p, dead)
            parts.append("end <b>%s</b> (x %.0f, y %.0f, z %.0f): <b>%.1f cm</b> from "
                         "the %s wall%s" % (nm, p[0], p[1], p[2], g, face,
                                            ("; %.1f cm from a dead channel in (y,z)" % dg)
                                            if dg is not None else ""))
        ends_div.text = ("straight-line extent <b>%.0f cm</b> &nbsp;|&nbsp; %s"
                         "<br><span style='color:#777'>measured off the charge; it does "
                         "not decide the label &mdash; an end can be at a wall and still "
                         "belong to a longer object that continues in grey.</span>"
                         % (L, " &nbsp;|&nbsp; ".join(parts)))
    status.text = ("event %s, cluster %d &mdash; %d points, %d grey context points "
                   "(1 in %d overall%s); arm %s"
                   % (it["event"], it["cluster"], it["npts"], len(oi), step,
                      ("; %d of them ALL charge within %.0f cm of the object"
                       % (n_near, radius)) if n_near else "", ARM))


def refresh_options():
    opts = [item_option(i) for i in ITEMS]
    item_select.options = opts
    item_select.value = opts[state["idx"]]


def go(idx):
    state["idx"] = max(0, min(len(ITEMS) - 1, idx))
    refresh_options()
    render()


def set_label(choice):
    it = current()
    LABELS[item_key(it)] = dict(label=choice, notes=notes.value,
                                scan_id=it["scan_id"], event=it["event"],
                                cluster=it["cluster"], npts=it["npts"], arm=ARM)
    save_labels()                     # every click, not only on Save
    export_sheet()                    # keep the filled sheet current too
    refresh_options()
    render()
    nxt = next((k for k in range(state["idx"] + 1, len(ITEMS))
                if item_key(ITEMS[k]) not in LABELS), None)
    if nxt is not None:
        go(nxt)


def clear_label():
    LABELS.pop(item_key(current()), None)
    save_labels()
    export_sheet()
    refresh_options()
    render()


def on_select(attr, old, new):
    if new in item_select.options:
        go(item_select.options.index(new))


def next_unlabelled():
    rng = list(range(state["idx"] + 1, len(ITEMS))) + list(range(0, state["idx"] + 1))
    nxt = next((k for k in rng if item_key(ITEMS[k]) not in LABELS), None)
    if nxt is None:
        status.text = "<b>every object is labelled.</b>"
    else:
        go(nxt)


def do_export():
    p = export_sheet()
    status.text = ("wrote <code>%s</code> (%d / %d labelled).  Score it with the "
                   "PRE-REGISTERED bar:<br><code>python3 docs/scripts/d04_movers_score.py "
                   "--sheet %s --key bee-pr-run029107-d04movers.KEY.tsv</code>"
                   % (p, sum(1 for i in ITEMS if item_key(i) in LABELS), len(ITEMS),
                      os.path.relpath(p, PDHD)))


item_select.on_change("value", on_select)
prev_btn.on_click(lambda: go(state["idx"] - 1))
next_btn.on_click(lambda: go(state["idx"] + 1))
next_unl_btn.on_click(next_unlabelled)
for _c in CHOICES:
    BTN[_c].on_click(lambda c=_c: set_label(c))
clear_btn.on_click(clear_label)
export_btn.on_click(do_export)
zoom_tog.on_click(lambda a: render())
dense_tog.on_click(lambda a: render())
radius_sel.on_change("value", lambda a, o, n: render())
plane_sel.on_change("value", lambda a, o, n: render())
win_sel.on_change("value", lambda a, o, n: render())

curdoc().add_root(column(
    header,
    row(item_select, prev_btn, next_btn, next_unl_btn),
    Div(text="<b>the cluster IS the whole object:</b>", width=1480),
    row(BTN["THRU"], BTN["STOP"], BTN["CONT"]),
    Div(text="<b>the cluster is only PART of the object</b> (under-clustered) &mdash; "
             "the verdict is for the FULL object:", width=1480),
    row(BTN["FRAG>THRU"], BTN["FRAG>STOP"], BTN["FRAG>CONT"]),
    row(BTN["MESSY"], BTN["UNCLEAR"], clear_btn, export_btn),
    row(zoom_tog, dense_tog, radius_sel, plane_sel, win_sel),
    row(notes, progress),
    ends_div,
    row(*[FIGS[(ha, va)] for ha, va, _ in PANELS]),
    Div(text="<b>the two ends, close up</b> &mdash; every point in the window is drawn, "
             "cluster and context alike. This is where <i>stops</i> and <i>continues into "
             "the next cluster</i> look different.", width=1480),
    row(ZOOM["A"], ZOOM["B"]),
    status,
))
curdoc().title = "PDHD mover scan (%s)" % SCAN_TAG
go(0)

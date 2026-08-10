#!/usr/bin/env python3
"""doc pr/57: overclustering-separation hand-scan display (Bokeh server).

Reads the per-edge JSONL dump written by connect_graph_relaxed_strict.cxx's
env-gated Oc56DumpWriter (WCT_OC56_SCAN_DUMP) -- one file per event,
<out>/pr_evt<ID>/oc56scan-evt<ID>.jsonl -- and shows a human, for every S6
2D-connectivity edge the algorithm evaluated (relaxed_strict_img_2d, still
default OFF everywhere in production), exactly the pixels that decided its
verdict: fired cells, dead-channel ranges, the two candidate components' own
seed footprints and the BFS search window, on the SAME lattice the BFS
walked (asserted, not just eyeballed -- see
scripts/analysis/pr57/oc56_dump_check.py, which this file imports for the
"is this really one long track broken in two" sort key).

Produce the dump:
    PR_JOBS=6 SBND_PROTECT_GRAPH=relaxed_strict_img_2d \\
    WCT_RELAXED_EDGE_CENSUS=1 PR_OC56_SCAN_DUMP=1 \\
      ./run_pr_chain_batch.sh <ql_root> <out> data <evt ...>
    # -> <out>/pr_evt<ID>/oc56scan-evt<ID>.jsonl

Serve:
    ./overclustering_display/serve_overclustering_scan.sh 5018 \\
      work-pr56r4b-scan*/pr_evt*/oc56scan-evt*.jsonl

LAYOUT

  Row 1 -- X-Y, Y-Z, X-Z. Component A (blue) and component B (red) that the
           selected edge candidates for a merge, the rest of the event's
           dumped points in grey for context, the edge itself as a black
           dashed line between its two closest-approach points.
  Row 2 -- U-T, V-T, W-T for the edge's own APA/face: fired cells (the
           real hits the BFS could step on), dead-channel ranges shaded,
           seeds A/B in the row-1 colours, the BFS search window outlined.
           This is the panel that answers "real gap vs. induction-plane
           signal inefficiency".
  Edge list -- one row per S6-evaluated edge in the current event: which
           planes gapped, distance, verdict, a near-miss badge read
           straight off the logged dw x ds connectivity matrix (e.g. "U
           closes @ dw=2" -- free, and turns a yes/no label into tuning
           data), and the long-track-break score (both pieces long by PCA,
           axes nearly collinear, facing endpoints close -- doc pr/56
           round 3's definition, computed in Python from the dumped points
           so revising it never costs a rerun). Defaults to "removed only",
           sorted by that score descending -- with up to ~170 removed
           edges in one event, the ordering is what makes the list usable.
  Labels -- good / OK / bad, a cause, and a free-text comment. Picking a
           verdict stages it in memory right away (no per-edge click); the
           table's label column shows staged-but-unsaved entries with a
           trailing "*". One "Save event labels" click flushes every staged
           edit for the current event to
           overclustering_labels/<tag>/labels-evt<ID>.json in a single
           write, keyed on geometry (event, blk, p1, p2 rounded to 0.01 cm),
           NOT on graph_call/j/k -- those are not reproducible across
           reruns (graph_call is a per-process atomic counter; j/k are
           cluster-local indices that collide across clusters, exactly the
           key doc pr/56 round 3 had to stop using for this reason).
           Switching events (or closing the browser tab) auto-flushes first,
           so nothing staged is ever silently lost.
  Pair   -- doc pr/57 round 4. `killed` is a verdict on ONE candidate edge;
           the scan's question is whether the two COMPONENTS should be apart.
           They are not the same: a pair gets up to three candidates
           (closest/dir1/dir2) and can also stay joined through a third
           component. The `pair` table column and the line under the
           projections answer the real question from the dump's
           "connectivity" record -- SEP / dir / via N / ? (pre-round-4 dump) --
           and the edges that still hold the pair together are drawn in green.
           "separated pairs only" narrows the table to the true separations.
"""
import argparse
import glob
import json
import os
import sys

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, Select, Button, Div, HoverTool,
                          RadioButtonGroup, TextInput, Toggle,
                          DataTable, TableColumn, HTMLTemplateFormatter,
                          Range1d)
from bokeh.plotting import figure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "analysis", "pr57"))
from oc56_dump_check import long_track_break_score  # noqa: E402
import oc56_conn as CONN  # noqa: E402

DET_BOX = dict(x=(-201.05, 201.05), y=(-199.312, 199.312), z=(0.85, 500.15))
DEFAULT_HALF = 30.0
PLANE_NAME = ("U", "V", "W")
COLOR_A, COLOR_B, COLOR_CTX, COLOR_EDGE = "#1f77b4", "#d62728", "#bbbbbb", "#111111"
COLOR_PATH = "#2ca02c"  # doc pr/57 round 4: edges that still connect the pair

CAUSES = ["induction inefficiency", "dead channel", "prolonged shower signal",
          "genuine separation", "other"]

# ---------------------------------------------------------------------------
# CLI / inputs
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("specs", nargs="*", help="oc56scan-evt*.jsonl paths or globs")
ap.add_argument("--tag", default="", help="labels subdir (overclustering_labels/<tag>/)")
args = ap.parse_args(sys.argv[1:])
SCAN_TAG = args.tag

FILES = []
for spec in args.specs:
    FILES += sorted(glob.glob(spec)) if any(c in spec for c in "*?[") else [spec]
seen, PATHS = set(), []
for p in FILES:
    rp = os.path.realpath(p)
    if rp in seen or not os.path.isfile(rp):
        continue
    seen.add(rp)
    PATHS.append(p)


def evt_label(path):
    base = os.path.basename(path)
    if base.startswith("oc56scan-evt") and base.endswith(".jsonl"):
        return "evt" + base[len("oc56scan-evt"):-len(".jsonl")]
    return os.path.basename(os.path.dirname(path)) or base


EVENTS = {}
for p in PATHS:
    EVENTS.setdefault(evt_label(p), p)
LABELS = sorted(EVENTS, key=lambda s: (len(s), s))

state = dict(evt=None, label=None, edges=[], rows_of=[], sel_edge=None,
            labels={}, pending={}, _suspend=False)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
class EventData:
    """One event's dump: every component point cloud (deduped by
    (graph_call, comp)) and every S6-evaluated edge, with the long-track-
    break score and near-miss badge precomputed."""

    def __init__(self, path):
        self.path = path
        self.components = {}
        self.edges = []
        conn_recs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["type"] == "component":
                    self.components[(rec["graph_call"], rec["comp"])] = rec
                elif rec["type"] == "edge":
                    self.edges.append(rec)
                elif rec["type"] == "connectivity":
                    conn_recs.append(rec)
        # doc pr/57 round 4. Absent in any pre-round-4 dump -- the viewer must
        # still open those, so every pair then reports "unknown" rather than
        # silently reading as "connected".
        self.conn = CONN.index_conn(conn_recs)
        for e in self.edges:
            ca = self.components.get((e["graph_call"], e["j"]))
            cb = self.components.get((e["graph_call"], e["k"]))
            e["_pts_a"] = ca["points"] if ca else []
            e["_pts_b"] = cb["points"] if cb else []
            score, detail = long_track_break_score(e["_pts_a"], e["_pts_b"], e["dis"])
            e["_score"] = score
            e["_detail"] = detail
            e["_near_miss"] = near_miss_badge(e)
            e["_key"] = edge_key(evt_label(path), e)
            e["_pair"] = CONN.pair_status(self.conn.get(e["graph_call"]),
                                          e["j"], e["k"])
        for e in self.edges:
            e["_siblings"] = CONN.siblings(self.edges, e["graph_call"],
                                           e["j"], e["k"], exclude=e)

    def context_points(self, exclude_keys, cap=20000):
        """All OTHER dumped components in this event (for the grey
        background), stride-subsampled if the total exceeds cap."""
        pts = []
        for key, c in self.components.items():
            if key in exclude_keys:
                continue
            pts.extend(c["points"])
        if len(pts) > cap:
            stride = (len(pts) + cap - 1) // cap
            pts = pts[::stride]
        return pts


def edge_key(evt_label_, e):
    """Geometric label key -- survives a rerun; graph_call/j/k do not (see
    module docstring)."""
    p1, p2 = e["p1"], e["p2"]
    return "%s|%s|%.2f,%.2f,%.2f|%.2f,%.2f,%.2f" % (
        evt_label_, e["blk"], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])


def near_miss_badge(e):
    names = PLANE_NAME
    parts = []
    for pi in range(3):
        if pi >= len(e.get("gap", [])) or not e["gap"][pi]:
            continue
        if pi >= len(e.get("matrix", [])):
            continue
        m = e["matrix"][pi]
        if len(m) != 16:
            continue
        near = []
        for dw in range(1, 5):
            if m[(dw - 1) * 4 + 0] == "1":
                near.append("dw=%d" % dw)
                break
        for ds in range(1, 5):
            if m[0 * 4 + (ds - 1)] == "1":
                near.append("ds=%d" % ds)
                break
        if near:
            parts.append("%s closes @ %s" % (names[pi], ",".join(near)))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Labels: good / OK / bad + cause + comment, keyed on geometry.
# ---------------------------------------------------------------------------
def labels_dir():
    d = os.path.join(HERE, "..", "overclustering_labels", SCAN_TAG)
    d = os.path.normpath(d)
    os.makedirs(d, exist_ok=True)
    return d


def labels_path(label):
    return os.path.join(labels_dir(), "labels-%s.json" % label)


def load_labels(label):
    p = labels_path(label)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p) as fh:
            return json.load(fh).get("labels", {})
    except (OSError, ValueError):
        return {}


def save_event_labels(label, entries):
    """Upsert a whole batch of edge labels into labels-evt<ID>.json in one
    write. Never touches or drops any other entry -- append-only in effect
    (M13: this is scan output, not something a later run should silently
    overwrite)."""
    p = labels_path(label)
    data = {"event": label, "source": os.path.basename(EVENTS[label])}
    if os.path.isfile(p):
        try:
            with open(p) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            pass
    data.setdefault("labels", {}).update(entries)
    with open(p, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    return p


# ---------------------------------------------------------------------------
# Figures -- row 1 (3-D projections)
# ---------------------------------------------------------------------------
proj_kw = dict(height=320, width=400, tools="pan,wheel_zoom,box_zoom,reset,save",
              active_scroll="wheel_zoom")
f_xy = figure(title="X-Y", x_range=Range1d(*DET_BOX["x"]), y_range=Range1d(*DET_BOX["y"]), **proj_kw)
f_yz = figure(title="Y-Z", x_range=Range1d(*DET_BOX["z"]), y_range=Range1d(*DET_BOX["y"]), **proj_kw)
f_xz = figure(title="X-Z", x_range=Range1d(*DET_BOX["x"]), y_range=Range1d(*DET_BOX["z"]), **proj_kw)
f_xy.xaxis.axis_label, f_xy.yaxis.axis_label = "x (cm)", "y (cm)"
f_yz.xaxis.axis_label, f_yz.yaxis.axis_label = "z (cm)", "y (cm)"
f_xz.xaxis.axis_label, f_xz.yaxis.axis_label = "x (cm)", "z (cm)"
PROJ = ((f_xy, "x", "y"), (f_yz, "z", "y"), (f_xz, "x", "z"))

EMPTY3 = dict(x=[], y=[], z=[])
ctx_src = ColumnDataSource(data=dict(EMPTY3))
a_src = ColumnDataSource(data=dict(EMPTY3))
b_src = ColumnDataSource(data=dict(EMPTY3))
edge_src = ColumnDataSource(data=dict(x=[], y=[], z=[]))  # 2-point line, p1->p2
# doc pr/57 round 4: the component edges that STILL connect this pair after
# every emit decision -- one multi_line segment per hop, per projection.
path_src = ColumnDataSource(data=dict(xs=[], ys=[], zs=[]))
det_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                     xs_xz=[], ys_xz=[]))

for f, hx, hy in PROJ:
    f.multi_line(xs="xs_%s" % ("xy" if f is f_xy else "yz" if f is f_yz else "xz"),
                 ys="ys_%s" % ("xy" if f is f_xy else "yz" if f is f_yz else "xz"),
                 source=det_src, line_color="#cc4444", line_width=1)
    f.scatter(hx, hy, source=ctx_src, marker="circle", size=2,
              fill_color=COLOR_CTX, line_color=None, fill_alpha=0.4)
    f.scatter(hx, hy, source=a_src, marker="circle", size=4,
              fill_color=COLOR_A, line_color=None, fill_alpha=0.85)
    f.scatter(hx, hy, source=b_src, marker="circle", size=4,
              fill_color=COLOR_B, line_color=None, fill_alpha=0.85)
    # surviving connection first, so the candidate under review draws on top
    f.multi_line(xs="%ss" % hx, ys="%ss" % hy, source=path_src,
                 line_color=COLOR_PATH, line_width=3, line_alpha=0.85)
    f.line(hx, hy, source=edge_src, line_color=COLOR_EDGE, line_width=2,
          line_dash="dashed")
    f.scatter(hx, hy, source=edge_src, marker="x", size=10,
              line_color=COLOR_EDGE, line_width=2)

# --- row 2: U-T, V-T, W-T for the selected edge's own apa/face -------------
panel = {}
for pi in range(3):
    f = figure(title="%s wire vs time slice" % PLANE_NAME[pi], width=560, height=280,
              tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom")
    f.xaxis.axis_label = "%s wire index" % PLANE_NAME[pi]
    f.yaxis.axis_label = "time slice"
    fired = ColumnDataSource(data=dict(w=[], s=[], h=[]))
    dead = ColumnDataSource(data=dict(w=[], s=[], h=[]))
    seeds_a = ColumnDataSource(data=dict(w=[], s=[]))
    seeds_b = ColumnDataSource(data=dict(w=[], s=[]))
    win = ColumnDataSource(data=dict(w=[], s=[]))
    rd = f.rect(x="w", y="s", width=1.0, height="h", source=dead,
               fill_color="#f2c14e", line_color=None, fill_alpha=0.5)
    rc = f.rect(x="w", y="s", width=1.0, height="h", source=fired,
               fill_color="#2a9d8f", line_color=None, fill_alpha=0.9)
    f.line("w", "s", source=win, line_color="#111111", line_width=1.5)
    f.scatter("w", "s", source=seeds_a, marker="circle", size=8,
             fill_color=None, line_color=COLOR_A, line_width=2)
    f.scatter("w", "s", source=seeds_b, marker="circle", size=8,
             fill_color=None, line_color=COLOR_B, line_width=2)
    f.add_tools(HoverTool(renderers=[rc], tooltips=[("wire", "@w"), ("slice", "@s")]))
    panel[pi] = dict(fig=f, fired=fired, dead=dead, seeds_a=seeds_a, seeds_b=seeds_b, win=win)

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
event_select = Select(title="event", options=LABELS, value=LABELS[0] if LABELS else "", width=220)
prev_evt_btn = Button(label="< prev evt", width=90)
next_evt_btn = Button(label="next evt >", width=90)
filter_toggle = Toggle(label="filter: removed / floor-blocked", active=True,
                       button_type="primary", width=220)
# owner: rejections beyond 5cm are reliably real gaps -- not worth scanning,
# so hide them by default and keep the scan focused on the close/ambiguous
# cases. Toggle off to see everything (e.g. to sanity-check a far edge).
far_filter_toggle = Toggle(label="hide dis > 5cm", active=True,
                           button_type="primary", width=140)
# doc pr/58: confirmation pass -- once edges are labeled, narrow the table to
# just the labeled verdict(s) under review, OK excluded either way. Both off
# is the unfiltered (pre-existing) behavior; both on shows good+bad, i.e.
# everything except OK/unlabeled.
show_good_toggle = Toggle(label="show good only", active=False,
                          button_type="success", width=130)
show_bad_toggle = Toggle(label="show bad only", active=False,
                         button_type="danger", width=130)
# doc pr/57 round 4: the hand scan judges whether two COMPONENTS should be
# separated, but a killed candidate often leaves them joined anyway (another
# candidate for the same pair, or a route through a third component). This
# narrows the table to the pairs the code really did separate -- the
# population the labels are about. Default OFF: today's behavior unchanged.
sep_filter_toggle = Toggle(label="separated pairs only", active=False,
                           button_type="warning", width=170)
sort_select = Select(title="sort by", options=[("score", "long-track-break score"),
                                               ("dis", "distance"), ("logged", "as logged")],
                    value="score", width=200)
status = Div(text="Loading...", width=1150)

edge_cols = ["idx", "blk", "j", "k", "planes", "dis", "killed", "pair",
             "near_miss", "score", "label"]
EDGE_EMPTY = {c: [] for c in edge_cols}
edge_table_src = ColumnDataSource(data=dict(EDGE_EMPTY))
edge_columns = [
    TableColumn(field="idx", title="#", width=35),
    TableColumn(field="blk", title="blk", width=55),
    TableColumn(field="j", title="j", width=30),
    TableColumn(field="k", title="k", width=30),
    TableColumn(field="planes", title="gap planes", width=80),
    TableColumn(field="dis", title="dis (cm)", width=65),
    TableColumn(field="killed", title="killed", width=60,
               formatter=HTMLTemplateFormatter(template="<b><%= value %></b>")),
    # doc pr/57 round 4: SEP = the two components really did end up apart;
    # dir = this same pair kept a direct edge anyway; via N = joined only
    # through N hops; ? = pre-round-4 dump with no connectivity record.
    TableColumn(field="pair", title="pair", width=60,
               formatter=HTMLTemplateFormatter(
                   template="<b style='color:<%= value == \"SEP\" "
                            "? \"#b03030\" : \"#2ca02c\" %>'><%= value %></b>")),
    TableColumn(field="near_miss", title="near-miss", width=200),
    TableColumn(field="score", title="break score", width=80),
    TableColumn(field="label", title="label", width=70,
               formatter=HTMLTemplateFormatter(template="<b><%= value %></b>")),
]
edge_table = DataTable(source=edge_table_src, columns=edge_columns, width=1150, height=300,
                       selectable=True, index_position=None)

# --- labeling panel ---------------------------------------------------------
# Picking a verdict stages it in memory (state["pending"]) immediately -- no
# per-edge save click. One "Save event labels" click at the end flushes every
# staged edit for the current event to disk in a single write. Switching
# events (or closing the server) auto-flushes first, so nothing is lost.
verdict_group = RadioButtonGroup(labels=["good", "OK", "bad"], active=None, width=300)
cause_group = RadioButtonGroup(labels=CAUSES, active=None, width=700)
comment_input = TextInput(title="comment", width=700)
save_btn = Button(label="Save event labels", button_type="success", width=160)
label_status = Div(text="", width=1150)
# doc pr/57 round 4: pair-level verdict for the selected edge -- separated or
# not, and if not, which emitted edges still hold the two components together.
conn_div = Div(text="", width=1150)

zoom_edge_btn = Button(label="zoom to edge", width=120)
zoom_event_btn = Button(label="whole event", width=120)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def progress_text():
    evt = state["evt"]
    if evt is None:
        return ""
    n_removed = sum(1 for e in evt.edges if e["killed"])
    labeled = len(set(state.get("labels", {})) | set(state.get("pending", {})))
    n_pending = len(state.get("pending", {}))
    unsaved = (" (<b>%d unsaved</b> -- click Save event labels)" % n_pending) if n_pending else ""
    return ("Loaded <b>%s</b>: %d S6-evaluated edges, %d removed (killed). "
            "%d labeled%s." % (state["label"], len(evt.edges), n_removed, labeled, unsaved))


def load_event(label):
    flush_pending(auto=True)
    evt = EventData(EVENTS[label])
    state["evt"] = evt
    state["label"] = label
    state["labels"] = load_labels(label)
    state["pending"] = {}
    state["sel_edge"] = None

    (xl, xh), (yl, yh), (zl, zh) = DET_BOX["x"], DET_BOX["y"], DET_BOX["z"]
    det_src.data = dict(
        xs_xy=[[xl, xh, xh, xl, xl], [0, 0]], ys_xy=[[yl, yl, yh, yh, yl], [yl, yh]],
        xs_yz=[[zl, zh, zh, zl, zl], []],     ys_yz=[[yl, yl, yh, yh, yl], []],
        xs_xz=[[xl, xh, xh, xl, xl], [0, 0]], ys_xz=[[zl, zl, zh, zh, zl], [zl, zh]])

    status.text = progress_text()
    refresh_table()
    clear_edge_panels()


def refresh_table(keep_selection=False):
    evt = state["evt"]
    if evt is None:
        edge_table_src.data = dict(EDGE_EMPTY)
        return
    labels = state.get("labels", {})
    pending = state.get("pending", {})

    def verdict_of(e):
        lab = pending.get(e["_key"]) or labels.get(e["_key"])
        return lab["verdict"] if lab else None

    # doc pr/58: "interesting" now also includes below-floor candidates
    # that S6's W-plane-only logic WOULD kill if the 1cm distance floor
    # didn't abstain first (killed_ignore_floor) -- exactly the case the
    # floor-override scan is meant to review (e.g. evt 21073 j=6/k=7).
    def interesting(e):
        removed = e["killed"] or e.get("killed_ignore_floor") or not filter_toggle.active
        if removed and far_filter_toggle.active and e["dis"] > 5.0:
            return False
        if not removed:
            return False
        # doc pr/57 round 4: only the pairs the code really separated.
        # Unknown (pre-round-4 dump) is kept -- hiding rows we cannot judge
        # would silently shrink the scan population.
        if sep_filter_toggle.active and e["_pair"]["separated"] is False:
            return False
        if show_good_toggle.active or show_bad_toggle.active:
            wanted = set()
            if show_good_toggle.active:
                wanted.add("good")
            if show_bad_toggle.active:
                wanted.add("bad")
            return verdict_of(e) in wanted
        return True
    pool = [e for e in evt.edges if interesting(e)]
    if sort_select.value == "score":
        pool.sort(key=lambda e: -e["_score"])
    elif sort_select.value == "dis":
        pool.sort(key=lambda e: e["dis"])
    state["rows_of"] = pool
    cols = {c: [] for c in edge_cols}
    for i, e in enumerate(pool):
        cols["idx"].append(i)
        cols["blk"].append(e["blk"])
        cols["j"].append(e["j"])
        cols["k"].append(e["k"])
        planes = "".join(PLANE_NAME[pi] for pi in range(3)
                         if pi < len(e.get("gap", [])) and e["gap"][pi])
        cols["planes"].append(planes or "-")
        cols["dis"].append("%.2f" % e["dis"])
        if e["killed"]:
            cols["killed"].append("yes")
        elif e.get("below_floor") and e.get("killed_ignore_floor"):
            cols["killed"].append("floor")  # below s6_dis_floor; W-plane-only logic would kill it
        else:
            cols["killed"].append("no")
        cols["pair"].append(CONN.short_token(e["_pair"]))
        cols["near_miss"].append(e["_near_miss"] or "-")
        cols["score"].append("%.1f" % e["_score"])
        if e["_key"] in pending:
            cols["label"].append(pending[e["_key"]]["verdict"] + " *")
        elif e["_key"] in labels:
            cols["label"].append(labels[e["_key"]]["verdict"])
        else:
            cols["label"].append("")
    edge_table_src.data = cols
    if not keep_selection:
        edge_table_src.selected.indices = []
    status.text = progress_text()


def clear_edge_panels():
    ctx_src.data = dict(EMPTY3)
    a_src.data = dict(EMPTY3)
    b_src.data = dict(EMPTY3)
    edge_src.data = dict(x=[], y=[], z=[])
    path_src.data = dict(xs=[], ys=[], zs=[])
    conn_div.text = ""
    for pi in range(3):
        panel[pi]["fired"].data = dict(w=[], s=[], h=[])
        panel[pi]["dead"].data = dict(w=[], s=[], h=[])
        panel[pi]["seeds_a"].data = dict(w=[], s=[])
        panel[pi]["seeds_b"].data = dict(w=[], s=[])
        panel[pi]["win"].data = dict(w=[], s=[])
    state["_suspend"] = True
    verdict_group.active = None
    cause_group.active = None
    comment_input.value = ""
    state["_suspend"] = False
    label_status.text = ""


def show_pair_connectivity(e):
    """doc pr/57 round 4: the pair-level answer for the selected candidate --
    are components j and k still connected after every emit decision, and if
    so by which edges. Those surviving edges are also drawn (green) in the
    three projections, so 'which edge connects them' is visible and not just
    asserted."""
    st = e["_pair"]
    xs, ys, zs = [], [], []
    for _, _, ce in st["path"]:
        q1, q2 = ce["p1"], ce["p2"]
        xs.append([q1[0], q2[0]])
        ys.append([q1[1], q2[1]])
        zs.append([q1[2], q2[2]])
    path_src.data = dict(xs=xs, ys=ys, zs=zs)

    color = "#b03030" if st["separated"] else ("#777777" if not st["known"] else "#2ca02c")
    parts = ["<span style='color:%s'><b>%s</b></span>"
             % (color, CONN.describe(st, e["j"], e["k"]))]
    sibs = e.get("_siblings", [])
    if sibs:
        parts.append("this pair's other S6 candidates: " + " &nbsp;|&nbsp; ".join(
            "%s killed=%s %.2fcm" % (s["blk"], "yes" if s["killed"] else "no", s["dis"])
            for s in sorted(sibs, key=lambda s: s["dis"])))
    else:
        parts.append("this pair's other S6 candidates: none "
                     "(this is the only one that reached the 2-D check)")
    conn_div.text = "<br>".join(parts)


def show_edge(e):
    stage_current()  # capture any in-progress edit on the edge we're leaving
    state["sel_edge"] = e
    evt = state["evt"]
    exclude = {(e["graph_call"], e["j"]), (e["graph_call"], e["k"])}
    ctx = evt.context_points(exclude)
    ctx_src.data = dict(x=[p[0] for p in ctx], y=[p[1] for p in ctx], z=[p[2] for p in ctx])
    a_src.data = dict(x=[p[0] for p in e["_pts_a"]], y=[p[1] for p in e["_pts_a"]],
                      z=[p[2] for p in e["_pts_a"]])
    b_src.data = dict(x=[p[0] for p in e["_pts_b"]], y=[p[1] for p in e["_pts_b"]],
                      z=[p[2] for p in e["_pts_b"]])
    p1, p2 = e["p1"], e["p2"]
    edge_src.data = dict(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]])
    show_pair_connectivity(e)

    by_plane = {p["plane"]: p for p in e.get("planes", [])}
    for pi in range(3):
        p = by_plane.get(pi)
        if p is None:
            panel[pi]["fired"].data = dict(w=[], s=[], h=[])
            panel[pi]["dead"].data = dict(w=[], s=[], h=[])
            panel[pi]["seeds_a"].data = dict(w=[], s=[])
            panel[pi]["seeds_b"].data = dict(w=[], s=[])
            panel[pi]["win"].data = dict(w=[], s=[])
            panel[pi]["fig"].title.text = "%s wire vs time slice (no data)" % PLANE_NAME[pi]
            continue
        h = p.get("native_step", 1)
        panel[pi]["fired"].data = dict(w=[c[0] for c in p["fired"]],
                                       s=[c[1] for c in p["fired"]],
                                       h=[h] * len(p["fired"]))
        dw, ds_, dh = [], [], []
        for wind, lo, hi in p["dead"]:
            dw.append(wind)
            ds_.append(0.5 * (lo + hi))
            dh.append(max(h, hi - lo + h))
        panel[pi]["dead"].data = dict(w=dw, s=ds_, h=dh)
        panel[pi]["seeds_a"].data = dict(w=[c[0] for c in p["seeds_a"]],
                                         s=[c[1] for c in p["seeds_a"]])
        panel[pi]["seeds_b"].data = dict(w=[c[0] for c in p["seeds_b"]],
                                         s=[c[1] for c in p["seeds_b"]])
        wlo, whi, slo, shi = p["win"]
        panel[pi]["win"].data = dict(w=[wlo, whi, whi, wlo, wlo], s=[slo, slo, shi, shi, slo])
        native = "" if h == 1 else (" (non-native dump, stride=%d)" % h)
        panel[pi]["fig"].title.text = ("%s wire vs time slice -- apa=%d face=%d%s"
                                       % (PLANE_NAME[pi], e["apa"], e["face"], native))

    pending_lab = state.get("pending", {}).get(e["_key"])
    lab = pending_lab or state.get("labels", {}).get(e["_key"])
    state["_suspend"] = True  # populating controls from a stored label is
                              # not a user edit -- don't re-stage it as pending
    if lab:
        verdict_group.active = ["good", "OK", "bad"].index(lab["verdict"]) \
            if lab.get("verdict") in ("good", "OK", "bad") else None
        cause_group.active = CAUSES.index(lab["cause"]) if lab.get("cause") in CAUSES else None
        comment_input.value = lab.get("comment", "")
    else:
        verdict_group.active = None
        cause_group.active = None
        comment_input.value = ""
    state["_suspend"] = False
    if lab:
        tag = "Unsaved" if pending_lab else "Saved"
        label_status.text = "%s label: <b>%s</b> / %s" % (tag, lab["verdict"], lab.get("cause", ""))
    else:
        label_status.text = "No label yet for this edge."
    apply_zoom_edge()


def apply_zoom_edge():
    e = state["sel_edge"]
    if e is None:
        return
    p1, p2 = e["p1"], e["p2"]
    cx = 0.5 * (p1[0] + p2[0]); cy = 0.5 * (p1[1] + p2[1]); cz = 0.5 * (p1[2] + p2[2])
    r = max(DEFAULT_HALF, e["dis"])
    f_xy.x_range.start, f_xy.x_range.end = cx - r, cx + r
    f_xy.y_range.start, f_xy.y_range.end = cy - r, cy + r
    f_yz.x_range.start, f_yz.x_range.end = cz - r, cz + r
    f_yz.y_range.start, f_yz.y_range.end = cy - r, cy + r
    f_xz.x_range.start, f_xz.x_range.end = cx - r, cx + r
    f_xz.y_range.start, f_xz.y_range.end = cz - r, cz + r


def apply_zoom_whole():
    f_xy.x_range.start, f_xy.x_range.end = DET_BOX["x"]
    f_xy.y_range.start, f_xy.y_range.end = DET_BOX["y"]
    f_yz.x_range.start, f_yz.x_range.end = DET_BOX["z"]
    f_yz.y_range.start, f_yz.y_range.end = DET_BOX["y"]
    f_xz.x_range.start, f_xz.x_range.end = DET_BOX["x"]
    f_xz.y_range.start, f_xz.y_range.end = DET_BOX["z"]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_event_change(attr, old, new):
    if new:
        load_event(new)


def step_event(d):
    if not LABELS:
        return
    i = LABELS.index(event_select.value)
    event_select.value = LABELS[(i + d) % len(LABELS)]


def on_filter_change(attr, old, new):
    refresh_table()
    clear_edge_panels()


def on_sort_change(attr, old, new):
    refresh_table()


def on_row_select(attr, old, new):
    if not new:
        return
    idx = new[0]
    rows = state.get("rows_of", [])
    if idx >= len(rows):
        return
    show_edge(rows[idx])


def stage_current():
    """Capture the verdict/cause/comment controls into state["pending"] for
    whichever edge is currently shown -- so picking a verdict for one edge
    and moving to the next never requires a click, and nothing is lost when
    switching edges, events, or hitting Save. A verdict is required; a
    comment/cause alone (no verdict) stages nothing, same as before."""
    e = state.get("sel_edge")
    if e is None or verdict_group.active is None:
        return
    verdict = ["good", "OK", "bad"][verdict_group.active]
    cause = CAUSES[cause_group.active] if cause_group.active is not None else ""
    entry = dict(verdict=verdict, cause=cause, comment=comment_input.value,
                blk=e["blk"], j=e["j"], k=e["k"], dis=e["dis"], killed=e["killed"],
                p1=e["p1"], p2=e["p2"], near_miss=e["_near_miss"], score=e["_score"])
    if state.get("labels", {}).get(e["_key"]) == entry:
        # identical to what's already on disk -- merely re-displaying a
        # saved edge is not an edit, so don't mark it pending again
        return
    state.setdefault("pending", {})[e["_key"]] = entry


def flush_pending(auto=False):
    """Write every staged (unsaved) label for the CURRENT event to disk in
    one batch upsert. `auto=True` is the silent flush done before switching
    events, so an in-progress scan is never silently discarded."""
    stage_current()
    pending = state.get("pending", {})
    label = state.get("label")
    if not pending or label is None:
        if not auto:
            label_status.text = "Nothing unsaved -- pick good / OK / bad for at least one edge first."
        return
    dest = save_event_labels(label, pending)
    state.setdefault("labels", {}).update(pending)
    n = len(pending)
    state["pending"] = {}
    prefix = "Auto-saved" if auto else "Saved"
    label_status.text = "%s %d label(s) for <b>%s</b> -> %s" % (prefix, n, label, dest)
    refresh_table(keep_selection=True)


def on_verdict_change(attr, old, new):
    if state.get("_suspend"):
        return
    stage_current()
    refresh_table(keep_selection=True)


def on_cause_change(attr, old, new):
    if state.get("_suspend"):
        return
    stage_current()
    refresh_table(keep_selection=True)


def on_comment_change(attr, old, new):
    if state.get("_suspend"):
        return
    stage_current()


event_select.on_change("value", on_event_change)
prev_evt_btn.on_click(lambda: step_event(-1))
next_evt_btn.on_click(lambda: step_event(+1))
filter_toggle.on_change("active", on_filter_change)
far_filter_toggle.on_change("active", on_filter_change)
show_good_toggle.on_change("active", on_filter_change)
show_bad_toggle.on_change("active", on_filter_change)
sep_filter_toggle.on_change("active", on_filter_change)
sort_select.on_change("value", on_sort_change)
edge_table_src.selected.on_change("indices", on_row_select)
verdict_group.on_change("active", on_verdict_change)
cause_group.on_change("active", on_cause_change)
comment_input.on_change("value", on_comment_change)
save_btn.on_click(lambda: flush_pending(auto=False))
zoom_edge_btn.on_click(apply_zoom_edge)
zoom_event_btn.on_click(apply_zoom_whole)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
controls = row(event_select, prev_evt_btn, next_evt_btn, filter_toggle, far_filter_toggle,
              sep_filter_toggle, show_good_toggle, show_bad_toggle, sort_select,
              zoom_edge_btn, zoom_event_btn)
proj_row = row(f_xy, f_yz, f_xz)
panel_row = row(panel[0]["fig"], panel[1]["fig"], panel[2]["fig"])
label_row = column(row(Div(text="<b>verdict:</b>", width=70), verdict_group),
                   row(Div(text="<b>cause:</b>", width=70), cause_group),
                   row(comment_input, save_btn),
                   label_status)

layout = column(status, controls, proj_row, conn_div, panel_row, edge_table, label_row)
curdoc().add_root(layout)
curdoc().title = "overclustering scan"
curdoc().on_session_destroyed(lambda session_context: flush_pending(auto=True))

if LABELS:
    load_event(LABELS[0])
else:
    status.text = ("No oc56scan-evt*.jsonl found. Produce the dump first "
                  "(WCT_OC56_SCAN_DUMP / PR_OC56_SCAN_DUMP=1) -- see the "
                  "module docstring and README.md.")

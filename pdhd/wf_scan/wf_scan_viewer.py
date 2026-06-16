"""Bokeh hand-scan viewer for the per-channel waveform plots.

Pages through ``pics/pd/wf_ch<NNN>.png`` (one PNG per optical channel, written
by ``pd_plot/spe_waveform_examples.py``) and records a Good / Bad verdict plus an
optional free-text comment per channel. Results accumulate in a single JSON file
keyed by channel, alongside the images, so a scan can be paused and resumed and
the verdicts are trivial to consume later.

Layout is a single scrollable column:

  - Channel dropdown + Prev / Next buttons (modulo wrap).
  - The current channel's PNG, embedded inline as a base64 data URI (no Bokeh
    static route needed, so it survives SSH port-forwarding unchanged).
  - Good / Bad buttons -> write the current channel's entry (verdict + comment +
    scanner + timestamp) into ``scan_results.json`` and auto-advance.
  - Comment box (free-form; saved with whichever verdict is pressed) and a
    scanner-name box.
  - Undo (one-deep, session-only) reverts the last write.
  - A summary table of every channel scanned so far, plus a counts line.

Run via the wrapper::

    ./pdhd/wf_scan/serve_wf_scan.sh 5016

To view from a remote laptop::

    ssh -L 5016:localhost:5016 user@wcgpu1
    # then open http://localhost:5016/wf_scan_viewer
"""
from __future__ import annotations

import base64
import datetime as dt
import glob
import json
import os
import re
import sys

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    DataTable,
    Div,
    Select,
    TableColumn,
    TextInput,
)

CH_RE = re.compile(r"wf_ch(\d+)\.png$")
RESULTS_NAME = "scan_results.json"


# ----- IO helpers ----------------------------------------------------------

def discover_images(spec):
    """Return [(channel_key, abspath), ...] sorted by channel number.

    ``spec`` is a directory (scanned for ``wf_ch*.png``) or a glob.
    """
    if os.path.isdir(spec):
        paths = glob.glob(os.path.join(spec, "wf_ch*.png"))
    else:
        paths = glob.glob(spec)
    out = []
    for p in paths:
        m = CH_RE.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), os.path.abspath(p)))
    out.sort(key=lambda t: t[0])
    return [("ch%03d" % n, p) for n, p in out]


def load_results(path):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_results(path, results):
    """Atomic full rewrite (small file, keyed-overwrite semantics)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def img_data_uri(path):
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return "data:image/png;base64," + b64


# ----- main ----------------------------------------------------------------

def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))

    # Optional --scanner NAME, then an optional image dir/glob.
    scanner_default = "anon"
    args = list(argv[1:])
    if args and args[0] in ("--scanner", "-s"):
        scanner_default = args[1] if len(args) > 1 else "anon"
        args = args[2:]
    spec = args[0] if args else os.path.join(here, "..", "pics", "pd")

    images = discover_images(spec)
    if not images:
        sys.exit("no wf_ch*.png found under %r" % spec)

    # The results file lives beside the images.
    img_dir = os.path.dirname(images[0][1])
    results_path = os.path.join(img_dir, RESULTS_NAME)
    results = load_results(results_path)
    undo_stack = []  # session-only snapshots of `results` before each write

    state = {"idx": 0}

    # ------------- widgets -------------------------------------------------
    keys = [k for k, _ in images]
    title = Div(text="<h2>PDHD wf_ch hand-scan</h2>", sizing_mode="stretch_width")
    channel_select = Select(title="Channel", value=keys[0], options=keys, width=160)
    prev_btn = Button(label="◀ Prev", width=90)
    next_btn = Button(label="Next ▶", width=90)
    image_div = Div(text="", sizing_mode="stretch_width")
    good_btn = Button(label="Good", button_type="success", width=120)
    bad_btn = Button(label="Bad", button_type="danger", width=120)
    comment_input = TextInput(title="Comment", placeholder="optional note",
                              sizing_mode="stretch_width")
    scanner_input = TextInput(title="Scanner", value=scanner_default, width=160)
    undo_btn = Button(label="Undo", width=90)
    status = Div(text="", sizing_mode="stretch_width")

    summary_src = ColumnDataSource(dict(channel=[], verdict=[], comment=[],
                                        scanner=[], scanned_at=[]))
    summary_table = DataTable(
        source=summary_src,
        columns=[
            TableColumn(field="channel", title="ch", width=70),
            TableColumn(field="verdict", title="verdict", width=80),
            TableColumn(field="comment", title="comment", width=320),
            TableColumn(field="scanner", title="scanner", width=100),
            TableColumn(field="scanned_at", title="scanned_at", width=170),
        ],
        autosize_mode="none", width=820, height=260,
    )

    # ------------- rendering -----------------------------------------------
    def refresh_summary():
        scanned = sorted(results.items())
        summary_src.data = dict(
            channel=[k for k, _ in scanned],
            verdict=[v.get("verdict", "") for _, v in scanned],
            comment=[v.get("comment", "") for _, v in scanned],
            scanner=[v.get("scanner", "") for _, v in scanned],
            scanned_at=[v.get("scanned_at", "") for _, v in scanned],
        )

    def counts_line():
        good = sum(1 for v in results.values() if v.get("verdict") == "good")
        bad = sum(1 for v in results.values() if v.get("verdict") == "bad")
        return "%d good, %d bad, %d unscanned (of %d)" % (
            good, bad, len(images) - len(results), len(images))

    def render():
        ck, path = images[state["idx"]]
        image_div.text = ('<img src="%s" style="max-width:100%%;height:auto;">'
                          % img_data_uri(path))
        if channel_select.value != ck:
            channel_select.value = ck
        entry = results.get(ck)
        comment_input.value = entry.get("comment", "") if entry else ""
        verdict = entry.get("verdict", "unscanned") if entry else "unscanned"
        status.text = ("<b>%s</b> (%d/%d) &mdash; current verdict: <b>%s</b> &mdash; %s"
                       % (ck, state["idx"] + 1, len(images), verdict, counts_line()))

    # ------------- callbacks -----------------------------------------------
    def goto(idx):
        state["idx"] = idx % len(images)
        render()

    def on_prev():
        goto(state["idx"] - 1)

    def on_next():
        goto(state["idx"] + 1)

    def on_select(_attr, _old, new):
        if new in keys:
            i = keys.index(new)
            if i != state["idx"]:
                goto(i)

    def do_label(verdict):
        ck = images[state["idx"]][0]
        undo_stack.append(json.loads(json.dumps(results)))  # cheap deep copy
        results[ck] = {
            "verdict": verdict,
            "comment": comment_input.value or "",
            "scanner": scanner_input.value or "anon",
            "image": os.path.basename(images[state["idx"]][1]),
            "scanned_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        save_results(results_path, results)
        refresh_summary()
        goto(state["idx"] + 1)  # auto-advance

    def on_undo():
        if not undo_stack:
            status.text = "nothing to undo this session"
            return
        prev = undo_stack.pop()
        results.clear()
        results.update(prev)
        save_results(results_path, results)
        refresh_summary()
        render()

    channel_select.on_change("value", on_select)
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    good_btn.on_click(lambda: do_label("good"))
    bad_btn.on_click(lambda: do_label("bad"))
    undo_btn.on_click(on_undo)

    # ------------- layout --------------------------------------------------
    layout = column(
        title,
        row(channel_select, prev_btn, next_btn),
        image_div,
        row(good_btn, bad_btn),
        comment_input,
        row(scanner_input, undo_btn),
        status,
        Div(text="<hr><h3>Scanned so far</h3>", sizing_mode="stretch_width"),
        summary_table,
        sizing_mode="stretch_width",
    )

    refresh_summary()
    render()
    curdoc().add_root(layout)
    curdoc().title = "PDHD wf_ch hand-scan"


main(sys.argv)

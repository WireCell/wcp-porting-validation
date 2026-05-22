"""Bokeh hand-scan viewer for Stage-A round-1 candidates.

Reads the artifacts written by ``code/inference/score_full_corpus.py`` and
serves a clickable review tool:

  - Threshold slider over the data-corpus score, filtering the scan pool.
  - Score histogram with markers at the threshold and current ROI.
  - Raw + decon waveforms (from consolidated data.h5) side-by-side.
  - Yes / No / Unsure buttons → append a row to handscan_round1.csv.
  - Labels tab with a DataTable of everything labeled so far.

Round-1 constraint: only train-split ROIs are scannable; val and test
rows are blocked at the viewer level to keep them genuinely held out for
round-2 evaluation. See docs/09 and the round-1 plan.

Run via the wrapper::

    ./pdhd/nf_plot/serve_handscan_viewer.sh \\
        experiments/stage_a_pu_round1/handscan

To view from a remote laptop::

    ssh -L 5008:localhost:5008 user@workstation
    # then http://localhost:5008/handscan_viewer
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BoxAnnotation,
    Button,
    CheckboxGroup,
    ColumnDataSource,
    DataTable,
    Div,
    HoverTool,
    NumericInput,
    Select,
    Slider,
    Span,
    TableColumn,
    TabPanel,
    Tabs,
    TextInput,
)
from bokeh.plotting import figure

# Make sibling imports work when launched via bokeh serve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "code", "data"))
from load_consolidated import ConsolidatedDataset  # noqa: E402


# ----- config -----------------------------------------------------------

LABEL_CSV_COLUMNS = [
    "row_idx", "run", "event", "plane", "channel",
    "start_tick", "end_tick", "score",
    "real", "type", "note",
    "scan_round", "model_version", "scan_stratum",
    "split", "holdout", "asym",
    "labeled_at_iso", "labeler",
]

TYPE_OPTIONS = [
    "(unspecified)",
    "L_long+asym",
    "L_long+fill_shape",
    "narrow_spike",
    "wide_plateau",
    "noise_like",
    "other",
]

PLANE_NAMES = ["U", "V", "W"]


# ----- IO helpers -------------------------------------------------------

def load_handscan_dir(handscan_dir: str) -> dict[str, Any]:
    """Load config + the two score CSVs."""
    cfg_path = os.path.join(handscan_dir, "handscan_config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"missing handscan_config.json under {handscan_dir} — "
                 f"run code/inference/score_full_corpus.py first")
    with open(cfg_path) as f:
        cfg = json.load(f)

    scores_path = os.path.join(handscan_dir, cfg["scores_full_csv"])
    pool_path = os.path.join(handscan_dir, cfg["scan_pool_csv"])
    labels_path = os.path.join(handscan_dir, cfg["labels_csv"])
    for p in (scores_path, pool_path):
        if not os.path.exists(p):
            sys.exit(f"missing CSV: {p}")

    scores_df = pd.read_csv(scores_path)
    pool_df = pd.read_csv(pool_path)
    return {
        "cfg": cfg,
        "handscan_dir": handscan_dir,
        "scores_df": scores_df,
        "pool_df": pool_df,
        "labels_path": labels_path,
    }


def load_labels(labels_path: str) -> pd.DataFrame:
    if not os.path.exists(labels_path):
        return pd.DataFrame(columns=LABEL_CSV_COLUMNS)
    df = pd.read_csv(labels_path)
    for c in LABEL_CSV_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[LABEL_CSV_COLUMNS]


def append_label_row(labels_path: str, row: dict) -> int:
    """Append one row, return the file offset BEFORE the write (for undo)."""
    write_header = not os.path.exists(labels_path)
    pos_before = os.path.getsize(labels_path) if os.path.exists(labels_path) else 0
    with open(labels_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LABEL_CSV_COLUMNS})
    return pos_before


def truncate_labels(labels_path: str, pos: int) -> None:
    """Truncate labels CSV to ``pos`` bytes (one-deep undo)."""
    with open(labels_path, "rb+") as f:
        f.truncate(pos)


# ----- main ---------------------------------------------------------------

def main(argv: list[str]) -> None:
    if len(argv) < 2:
        sys.exit("usage: bokeh serve handscan_viewer.py "
                 "--args <handscan_dir>")
    handscan_dir = os.path.abspath(argv[1])
    state = load_handscan_dir(handscan_dir)
    cfg = state["cfg"]
    scores_df: pd.DataFrame = state["scores_df"]
    pool_df: pd.DataFrame = state["pool_df"]
    labels_path: str = state["labels_path"]

    consolidated = ConsolidatedDataset(cfg["data_h5"], build_index=False)

    labels_df = load_labels(labels_path)
    # session-only stack of append offsets, used for undo
    undo_stack: list[int] = []

    # ------------- widgets -------------------------------------------------
    title_div = Div(
        text=(f"<h2>L1SP hand-scan — round 1</h2>"
              f"<i>model: <code>{cfg['model_version']}</code> &nbsp; "
              f"data: <code>{cfg['data_h5']}</code></i>"),
        sizing_mode="stretch_width",
    )

    pool_min = float(pool_df["score"].min())
    pool_max = float(pool_df["score"].max())
    slider_lo = max(cfg.get("score_floor", 0.005),
                    float(np.floor(pool_min * 1000) / 1000))
    slider_hi = float(np.ceil(pool_max * 100) / 100)
    threshold = Slider(
        title="threshold (data score)", start=slider_lo, end=slider_hi,
        step=0.001, value=0.020, sizing_mode="stretch_width",
    )

    filter_select = Select(
        title="filter", value="unlabeled",
        options=["unlabeled", "all", "labeled-yes", "labeled-no",
                 "labeled-unsure"],
        width=160,
    )

    stats_div = Div(text="", sizing_mode="stretch_width")

    prev_btn = Button(label="◀ Prev", width=90)
    next_btn = Button(label="Next ▶", width=90)
    yes_btn = Button(label="Yes ✓ (L1SP)", button_type="success", width=140)
    no_btn = Button(label="No ✗ (not L1SP)", button_type="danger", width=140)
    unsure_btn = Button(label="Unsure ?", width=110)
    undo_btn = Button(label="↶ Undo", button_type="warning", width=90)

    type_select = Select(title="type", value=TYPE_OPTIONS[0],
                         options=TYPE_OPTIONS, width=180)
    holdout_cb = CheckboxGroup(
        labels=["hold out from training (for blind rescan)"], active=[],
        width=320,
    )
    note_input = TextInput(title="note (optional)", value="",
                            sizing_mode="stretch_width")
    labeler_input = TextInput(title="labeler", value=os.environ.get(
        "USER", "anon"), width=160)

    jump_run = NumericInput(title="run", value=None, mode="int", width=90)
    jump_evt = NumericInput(title="event", value=None, mode="int", width=80)
    jump_chan = NumericInput(title="channel", value=None, mode="int",
                              width=90)
    jump_btn = Button(label="Jump", width=70)

    info_div = Div(text="", sizing_mode="stretch_width")
    warn_div = Div(text="", sizing_mode="stretch_width",
                   styles={"color": "#a00", "font-weight": "bold"})

    # ------------- figures -------------------------------------------------
    fig_kwargs = dict(height=180, sizing_mode="stretch_width",
                      tools="pan,wheel_zoom,box_zoom,reset,save",
                      active_scroll="wheel_zoom")
    f_raw = figure(title="raw ADC", **fig_kwargs)
    f_dec = figure(title="decon", x_range=f_raw.x_range, **fig_kwargs)
    for f in (f_raw, f_dec):
        f.xaxis.axis_label = "tick"
    f_raw.yaxis.axis_label = "ADC"
    f_dec.yaxis.axis_label = "amplitude"

    src_raw = ColumnDataSource(data=dict(x=[], y=[]))
    src_dec = ColumnDataSource(data=dict(x=[], y=[]))
    f_raw.line("x", "y", source=src_raw, line_width=1, color="#1f77b4")
    f_dec.line("x", "y", source=src_dec, line_width=1, color="#ff7f0e")
    for f, src in ((f_raw, src_raw), (f_dec, src_dec)):
        f.add_tools(HoverTool(tooltips=[("tick", "@x"), ("value", "@y")],
                              mode="vline"))
    roi_boxes = [BoxAnnotation(left=0, right=1, fill_color="#ccccff",
                                fill_alpha=0.25, line_color=None)
                 for _ in range(2)]
    f_raw.add_layout(roi_boxes[0])
    f_dec.add_layout(roi_boxes[1])

    # score-distribution histogram (built once over the full data corpus)
    full_scores = scores_df["score"].to_numpy()
    edges = np.geomspace(max(full_scores.min(), 1e-4),
                          max(full_scores.max(), 1.0), 101)
    counts, _ = np.histogram(full_scores, bins=edges)
    hist_src = ColumnDataSource(data=dict(
        left=edges[:-1], right=edges[1:],
        top=np.where(counts > 0, counts, 0.5),
        bottom=np.full_like(counts, 0.5, dtype=float),
    ))
    f_hist = figure(
        title="full-corpus data score distribution "
              "(log-log)  —  vertical lines: threshold (green), current ROI (red)",
        height=160, sizing_mode="stretch_width",
        x_axis_type="log", y_axis_type="log",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    f_hist.quad(left="left", right="right", top="top", bottom="bottom",
                source=hist_src, fill_color="#cccccc", line_color="#888888")
    thr_span = Span(location=threshold.value, dimension="height",
                    line_color="green", line_width=2)
    roi_span = Span(location=0.001, dimension="height",
                    line_color="red", line_dash="dashed", line_width=2)
    f_hist.add_layout(thr_span)
    f_hist.add_layout(roi_span)
    f_hist.xaxis.axis_label = "model score"
    f_hist.yaxis.axis_label = "count"

    # ------------- labels tab ----------------------------------------------
    labels_src = ColumnDataSource(data=labels_df.astype(str).to_dict("list"))
    labels_table = DataTable(
        source=labels_src,
        columns=[TableColumn(field=c, title=c) for c in
                 ["labeled_at_iso", "run", "event", "plane", "channel",
                  "score", "real", "type", "holdout", "note", "labeler"]],
        height=520, sizing_mode="stretch_width", autosize_mode="fit_columns",
    )
    labels_summary = Div(text="", sizing_mode="stretch_width")
    export_btn = Button(label="Export snapshot CSV", width=200)

    # ------------- state ---------------------------------------------------
    state_ = {
        "filtered": [],   # list of (rank, row_idx) tuples after filter
        "cursor": 0,      # index into filtered list
        "current_row_idx": None,
    }

    def labeled_set() -> set[int]:
        if len(labels_df) == 0:
            return set()
        return set(int(x) for x in labels_df["row_idx"].tolist())

    def labeled_by_real() -> dict[int, str]:
        """row_idx -> latest 'real' value."""
        out: dict[int, str] = {}
        for _, r in labels_df.iterrows():
            out[int(r["row_idx"])] = str(r["real"])
        return out

    def refilter() -> None:
        thr = float(threshold.value)
        labeled = labeled_set()
        last_real = labeled_by_real()
        mode = filter_select.value
        keep: list[tuple[int, int]] = []
        for _, r in pool_df.iterrows():
            ridx = int(r["row_idx"])
            score = float(r["score"])
            if score < thr:
                continue
            is_labeled = ridx in labeled
            if mode == "unlabeled" and is_labeled:
                continue
            if mode == "labeled-yes" and last_real.get(ridx) != "Yes":
                continue
            if mode == "labeled-no" and last_real.get(ridx) != "No":
                continue
            if mode == "labeled-unsure" and last_real.get(ridx) != "Unsure":
                continue
            keep.append((int(r["rank"]), ridx))
        state_["filtered"] = keep
        if state_["cursor"] >= len(keep):
            state_["cursor"] = max(0, len(keep) - 1)

    def update_stats() -> None:
        thr = float(threshold.value)
        n_above = int((pool_df["score"] >= thr).sum())
        n_lab = len(labels_df)
        remaining = max(n_above - n_lab, 0)
        n_filtered = len(state_["filtered"])
        cursor = state_["cursor"] + 1 if n_filtered > 0 else 0
        stats_div.text = (
            f"<b>pool above threshold:</b> {n_above:,} &nbsp; "
            f"<b>labeled (any):</b> {n_lab} &nbsp; "
            f"<b>filtered view:</b> {n_filtered:,} "
            f"(cursor {cursor}/{n_filtered}) &nbsp; "
            f"<b>train-only round; val + test reserved</b>"
        )

    def update_labels_tab() -> None:
        df = labels_df.iloc[::-1].copy()  # latest first
        labels_src.data = df.astype(str).to_dict("list")
        n = len(labels_df)
        if n == 0:
            labels_summary.text = "<i>no labels yet</i>"
            return
        c = labels_df["real"].value_counts().to_dict()
        pct = lambda k: f"{(c.get(k, 0) / n * 100):.1f}%"
        labels_summary.text = (
            f"<b>total:</b> {n} &nbsp; "
            f"<b>Yes:</b> {c.get('Yes', 0)} ({pct('Yes')}) &nbsp; "
            f"<b>No:</b> {c.get('No', 0)} ({pct('No')}) &nbsp; "
            f"<b>Unsure:</b> {c.get('Unsure', 0)} ({pct('Unsure')})"
        )

    def render_current() -> None:
        warn_div.text = ""
        thr_span.location = float(threshold.value)
        if not state_["filtered"]:
            info_div.text = "<i>no candidates match the current filter</i>"
            for src in (src_raw, src_dec):
                src.data = dict(x=[], y=[])
            for b in roi_boxes:
                b.left, b.right = 0, 1
            roi_span.location = 1e-4
            state_["current_row_idx"] = None
            return
        rank, ridx = state_["filtered"][state_["cursor"]]
        meta = scores_df.iloc[ridx]
        split = str(meta["split"])
        if split != "train":
            warn_div.text = (
                f"<u>split = '{split}'</u> — not scannable in round 1. "
                f"Move on or jump to a train-split ROI."
            )
        try:
            raw, dec = consolidated.waveforms(int(ridx))
        except Exception as e:
            info_div.text = f"<b>error loading waveform for row {ridx}: {e}</b>"
            return
        start = int(meta["start_tick"])
        end = int(meta["end_tick"])
        n = len(raw)
        x = np.arange(start, start + n)
        src_raw.data = dict(x=x, y=raw)
        src_dec.data = dict(x=x, y=dec)
        for b in roi_boxes:
            b.left, b.right = start, end
        roi_span.location = float(meta["score"])
        plane = int(meta["plane"])
        pname = PLANE_NAMES[plane] if 0 <= plane < 3 else str(plane)
        # Decon-storage diagnostic: ~9% of dumped ROIs (sim + data) have an
        # all-zero decon array in the source NPZ. Surface this explicitly so
        # the empty panel doesn't look like a viewer bug.
        dec_nz = int(np.count_nonzero(dec))
        if dec_nz == 0:
            f_dec.title.text = "decon — ALL ZERO IN STORAGE (upstream dump)"
            f_dec.title.text_color = "#a55"
            f_dec.y_range.start = -1.0
            f_dec.y_range.end = 1.0
            dec_note = ("&nbsp; <span style='color:#a55'>"
                         f"<b>decon all-zero</b> ({n} ticks, upstream "
                         f"dump artifact — model likely scoring the "
                         f"large raw asymmetry)</span>")
        else:
            f_dec.title.text = "decon"
            f_dec.title.text_color = "black"
            # let auto-range take over
            f_dec.y_range.start = None
            f_dec.y_range.end = None
            dec_note = (f"&nbsp; decon nonzero {dec_nz}/{n}")
        labels_now = labeled_by_real()
        prior_real = labels_now.get(int(ridx))
        prior_html = (f"<span style='color:#080'><b>prior label: "
                      f"{prior_real}</b></span><br>" if prior_real else "")
        info_div.text = (
            f"{prior_html}"
            f"<b>rank {rank} (cursor {state_['cursor']+1}/"
            f"{len(state_['filtered'])})</b> &nbsp; "
            f"run <b>{int(meta['run'])}</b> &nbsp; evt <b>{int(meta['event'])}</b>"
            f" &nbsp; APA <b>{int(meta['apa'])}</b> &nbsp; "
            f"plane <b>{pname}</b> &nbsp; ch <b>{int(meta['channel'])}</b>"
            f" &nbsp; ticks <b>[{start}, {end})</b>"
            f"{dec_note}<br>"
            f"score <b>{float(meta['score']):.6f}</b> &nbsp; "
            f"vae_kl {float(meta['vae_kl']):.3f} &nbsp; "
            f"vae_recon_mse {float(meta['vae_recon_mse']):.5f} &nbsp; "
            f"heuristic_label <b>{int(meta['heuristic_label'])}</b> &nbsp; "
            f"split <b>{split}</b>"
        )
        state_["current_row_idx"] = int(ridx)
        update_stats()

    def reload_view(advance: bool = False) -> None:
        refilter()
        if advance and state_["filtered"]:
            # leave cursor where it is — refilter has already clamped
            pass
        update_stats()
        update_labels_tab()
        render_current()

    # ------------- callbacks -----------------------------------------------
    def on_threshold(_attr, _old, _new):
        state_["cursor"] = 0
        reload_view()

    def on_filter(_attr, _old, _new):
        state_["cursor"] = 0
        reload_view()

    def on_prev():
        if not state_["filtered"]:
            return
        state_["cursor"] = (state_["cursor"] - 1) % len(state_["filtered"])
        render_current()

    def on_next():
        if not state_["filtered"]:
            return
        state_["cursor"] = (state_["cursor"] + 1) % len(state_["filtered"])
        render_current()

    def do_label(real: str) -> None:
        nonlocal labels_df
        ridx = state_.get("current_row_idx")
        if ridx is None:
            return
        meta = scores_df.iloc[int(ridx)]
        split = str(meta["split"])
        if split != "train":
            warn_div.text = (
                f"refusing to label split='{split}' ROI (round-1 hygiene)"
            )
            return
        thr = float(threshold.value)
        type_val = type_select.value if type_select.value != "(unspecified)" else ""
        holdout = bool(0 in holdout_cb.active)
        row = {
            "row_idx": int(ridx),
            "run": int(meta["run"]),
            "event": int(meta["event"]),
            "plane": int(meta["plane"]),
            "channel": int(meta["channel"]),
            "start_tick": int(meta["start_tick"]),
            "end_tick": int(meta["end_tick"]),
            "score": f"{float(meta['score']):.6f}",
            "real": real,
            "type": type_val,
            "note": note_input.value or "",
            "scan_round": int(cfg.get("scan_round", 1)),
            "model_version": cfg.get("model_version", ""),
            "scan_stratum": "top-score-thr-tunable",
            "split": split,
            "holdout": "True" if holdout else "False",
            "asym": f"{float(meta['score']) - thr:+.6f}",
            "labeled_at_iso": dt.datetime.now().isoformat(timespec="seconds"),
            "labeler": labeler_input.value or "anon",
        }
        pos_before = append_label_row(labels_path, row)
        undo_stack.append(pos_before)
        labels_df = load_labels(labels_path)
        # auto-reset per-label one-shot inputs
        holdout_cb.active = []
        note_input.value = ""
        type_select.value = TYPE_OPTIONS[0]
        # advance, then refilter (so 'unlabeled' filter hides this one)
        if filter_select.value == "unlabeled":
            reload_view()
            # cursor stayed put → next unlabeled is now at same position
        else:
            update_labels_tab()
            update_stats()
            state_["cursor"] = (state_["cursor"] + 1) % max(
                len(state_["filtered"]), 1)
            render_current()

    def on_yes():
        do_label("Yes")

    def on_no():
        do_label("No")

    def on_unsure():
        do_label("Unsure")

    def on_undo():
        nonlocal labels_df
        if not undo_stack:
            warn_div.text = "nothing to undo this session"
            return
        last_pos = undo_stack.pop()
        truncate_labels(labels_path, last_pos)
        labels_df = load_labels(labels_path)
        reload_view()

    def on_jump():
        if jump_run.value is None or jump_evt.value is None or jump_chan.value is None:
            warn_div.text = "fill run, event, and channel to jump"
            return
        mask = ((scores_df["run"] == int(jump_run.value)) &
                (scores_df["event"] == int(jump_evt.value)) &
                (scores_df["channel"] == int(jump_chan.value)))
        candidates = scores_df.index[mask].tolist()
        if not candidates:
            warn_div.text = "no ROI matches (run, event, channel)"
            return
        target = int(candidates[0])
        # find or insert this index into filtered list
        for i, (_, ridx) in enumerate(state_["filtered"]):
            if ridx == target:
                state_["cursor"] = i
                render_current()
                return
        # not in filtered list — append a one-off entry so we can show it
        state_["filtered"].append((0, target))
        state_["cursor"] = len(state_["filtered"]) - 1
        render_current()

    def on_export():
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        exp_dir = os.path.join(handscan_dir, "exports")
        os.makedirs(exp_dir, exist_ok=True)
        dst = os.path.join(exp_dir, f"handscan_round1_{ts}.csv")
        labels_df.to_csv(dst, index=False)
        warn_div.text = f"snapshot written: <code>{dst}</code>"

    threshold.on_change("value_throttled", on_threshold)
    filter_select.on_change("value", on_filter)
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    yes_btn.on_click(on_yes)
    no_btn.on_click(on_no)
    unsure_btn.on_click(on_unsure)
    undo_btn.on_click(on_undo)
    jump_btn.on_click(on_jump)
    export_btn.on_click(on_export)

    # ------------- layout --------------------------------------------------
    top_row = row(threshold, filter_select, sizing_mode="stretch_width")
    jump_row = row(jump_run, jump_evt, jump_chan, jump_btn,
                   sizing_mode="stretch_width")
    button_row = row(prev_btn, yes_btn, no_btn, unsure_btn, next_btn,
                     undo_btn, sizing_mode="stretch_width")
    meta_row = row(type_select, holdout_cb, labeler_input,
                   sizing_mode="stretch_width")
    scan_tab = TabPanel(
        title="Scan",
        child=column(
            title_div,
            top_row, stats_div, jump_row,
            f_hist,
            f_raw, f_dec,
            info_div, warn_div,
            button_row,
            meta_row, note_input,
            sizing_mode="stretch_width",
        ),
    )
    labels_tab = TabPanel(
        title="Labels",
        child=column(labels_summary, export_btn, labels_table,
                     sizing_mode="stretch_width"),
    )
    tabs = Tabs(tabs=[scan_tab, labels_tab])

    reload_view()

    curdoc().add_root(tabs)
    curdoc().title = "L1SP hand-scan round 1"


main(sys.argv)

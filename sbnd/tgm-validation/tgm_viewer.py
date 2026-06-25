#!/usr/bin/env python3
"""Interactive 3-panel (XY / XZ / YZ) viewer of TGM tagged tracks (Bokeh server).

Reads the .npz written by summarize_tgm.py and shows the cumulated
through-going-muon endpoints + track bodies in three linked-zoom 2D
projections, with the fiducial-volume box drawn for reference.  Endpoints
(q=10000) are large red dots; the track body (q=100) is small grey dots,
coloured per track.  An event selector restricts the display to one event or
shows all events overlaid (the box-shape sanity view).

Modelled on standalone-sample/w-gap/compare_wires_viewer.py (server-side
callbacks, run_app/main_cli split).

Launch:
    bokeh serve --show tgm_viewer.py --args tgm-validation/tgm_points.npz
CLI smoke test:
    python tgm_viewer.py tgm-validation/tgm_points.npz
"""
import sys

import numpy as np

# FV box (cm), margin-shrunk, from clus.jsonnet dvm.overall (keep in sync).
FV = dict(xlo=-201.05 + 2.0, xhi=201.05 - 2.0,
          ylo=-199.312 + 2.5, yhi=199.312 - 2.5,
          zlo=0.85 + 3.0, zhi=500.15 - 3.0)


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _palette_color(clid):
    # Deterministic per-track colour from cluster id.
    from bokeh.palettes import Category20
    pal = Category20[20]
    return [pal[int(c) % 20] for c in clid]


# ---------------------------------------------------------------------------
def run_app():
    from bokeh.io import curdoc
    from bokeh.layouts import column, row
    from bokeh.models import ColumnDataSource, Select, Div, BoxAnnotation, Range1d
    from bokeh.plotting import figure

    path = sys.argv[1] if len(sys.argv) > 1 else "tgm-validation/tgm_points.npz"
    data = load(path)

    events = np.unique(np.concatenate([data["end_evt"], data["body_evt"]])) \
        if data["end_evt"].size or data["body_evt"].size else np.array([], dtype=int)
    evt_options = ["all"] + [str(int(e)) for e in events]

    sel_evt = Select(title="event", value="all", options=evt_options, width=120)
    info = Div(text="", width=600)

    # endpoints only (q=10000); no body points
    src_end = ColumnDataSource(data=dict(x=[], y=[], z=[], clid=[], col=[]))

    pad = 12.0  # cm padding around a zoom window

    def make_panel(ax, ay, alo, ahi, blo, bhi, alabel, blabel, view_x=None, view_y=None):
        # view_x / view_y: (lo, hi) display window; default = FV box + padding.
        vx = view_x if view_x is not None else (alo - pad, ahi + pad)
        vy = view_y if view_y is not None else (blo - pad, bhi + pad)
        fig = figure(width=520, height=520, title="%s vs %s" % (blabel, alabel),
                     x_axis_label="%s [cm]" % alabel, y_axis_label="%s [cm]" % blabel,
                     x_range=Range1d(*vx), y_range=Range1d(*vy),
                     tools="pan,box_zoom,wheel_zoom,reset,hover,tap",
                     tooltips=[("clid", "@clid"), (alabel, "@" + ax), (blabel, "@" + ay)])
        fig.add_layout(BoxAnnotation(left=alo, right=ahi, bottom=blo, top=bhi,
                                     fill_alpha=0.0, line_color="navy", line_dash="dashed",
                                     line_width=2))
        fig.scatter(ax, ay, source=src_end, size=8, color="crimson",
                    line_color="black", line_width=0.5, alpha=0.9)
        return fig

    # YX: zoom the y-axis around the top Y boundary (~200 cm).
    fig_xy = make_panel("x", "y", FV["xlo"], FV["xhi"], FV["ylo"], FV["yhi"], "x", "y",
                        view_y=(FV["yhi"] - 30, FV["yhi"] + 15))
    # ZX: zoom the z-axis around the (top) Z boundary (~500 cm).
    fig_xz = make_panel("x", "z", FV["xlo"], FV["xhi"], FV["zlo"], FV["zhi"], "x", "z",
                        view_y=(FV["zhi"] - 30, FV["zhi"] + 15))
    # ZY: full box view.
    fig_yz = make_panel("y", "z", FV["ylo"], FV["yhi"], FV["zlo"], FV["zhi"], "y", "z")

    def refresh():
        ev = sel_evt.value
        if ev == "all":
            em = np.ones(data["end_evt"].shape, dtype=bool)
        else:
            em = data["end_evt"] == int(ev)

        src_end.data = dict(
            x=data["end_x"][em], y=data["end_y"][em], z=data["end_z"][em],
            clid=data["end_clid"][em], col=_palette_color(data["end_clid"][em]))

        ntrk = len(np.unique(data["end_clid"][em])) if em.any() else 0
        info.text = ("<b>event %s</b> &mdash; %d endpoints, %d tagged tracks "
                     "(endpoints only; YX zoomed to top y≈%.0f, "
                     "ZX zoomed to z≈%.0f)"
                     % (ev, int(em.sum()), ntrk, FV["yhi"], FV["zhi"]))

    sel_evt.on_change("value", lambda a, o, n: refresh())
    refresh()

    layout = column(row(sel_evt), info, row(fig_xy, fig_xz, fig_yz))
    curdoc().add_root(layout)
    curdoc().title = "TGM tagged tracks"


# ---------------------------------------------------------------------------
def main_cli():
    path = sys.argv[1] if len(sys.argv) > 1 else "tgm-validation/tgm_points.npz"
    d = load(path)
    nevt = np.unique(d["end_evt"]).size if d["end_evt"].size else 0
    print("%s: %d events, %d endpoints, %d body pts"
          % (path, nevt, d["end_x"].size, d["body_x"].size))
    for ax in "xyz":
        a = d["end_" + ax]
        if a.size:
            print("  endpoint %s [cm]: [%.1f, %.1f]" % (ax, a.min(), a.max()))


if __name__.startswith("bokeh_app"):
    run_app()
elif __name__ == "__main__":
    main_cli()

#!/usr/bin/env python3
"""Visualize an SBND "magnify" ROOT file (TH2F dump of channel x tick).

Defaults: top-to-bottom 3 2D panels for hw_orig0, hw_raw0, hw_gauss0 (the
W-plane / TPC 0 stages of the SBND signal-processing chain).  Below them a
1D waveform panel; clicking on any 2D panel overlays the clicked channel's
waveform from all three histograms on the 1D panel.  A control row below
the 1D panel exposes channel/tick range + step textboxes and reset buttons.

Override any panel via --panel HIST_NAME (give --panel up to three times),
or use the explicit --h1/--h2/--h3 aliases.

Batch mode: --out-prefix PREFIX dumps each 2D panel as PREFIX_<hname>.pdf
+ .npy with no GUI.

Example:
  python3 plot_magnify.py --input sbnd-data-check.root \\
      --channel-min 4000 --channel-max 4200 --tick-min 0 --tick-max 3427

  python3 plot_magnify.py --h1 hu_orig0 --h2 hu_raw0 --h3 hu_gauss0
"""

import argparse
import os
import sys

import numpy as np


DEFAULT_INPUT = "sbnd-data-check.root"
DEFAULT_PANELS = ["hw_orig0", "hw_raw0", "hw_gauss0"]
# Per-panel default color scale (vmin,vmax) matched to DEFAULT_PANELS by index.
# Pass "auto" via --vrange to fall back to the 99-percentile autoscale (and the
# signed-cmap auto-detection, if --auto-signed-cmap is set).
DEFAULT_VRANGES = ["600,1000", "0,400", "0,10000"]
_ARRAY_WARN_LIMIT = 500_000_000


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="magnify ROOT file")
    p.add_argument(
        "--panel",
        action="append",
        metavar="HIST",
        help=(
            "TH2F histogram name for a panel (use up to 3 times). "
            f"Defaults: {' '.join(DEFAULT_PANELS)}"
        ),
    )
    p.add_argument("--h1", metavar="HIST", help="Override panel 1 name (top).")
    p.add_argument("--h2", metavar="HIST", help="Override panel 2 name (middle).")
    p.add_argument("--h3", metavar="HIST", help="Override panel 3 name (bottom).")
    p.add_argument("--channel-min", type=int, default=None)
    p.add_argument("--channel-max", type=int, default=None)
    p.add_argument("--tick-min", type=int, default=None)
    p.add_argument("--tick-max", type=int, default=None)
    p.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.0,
        help="Percentile of nonzero |values| used for auto vmax (default 99.0)",
    )
    p.add_argument(
        "--cmap",
        default="YlOrRd",
        help="Colormap for unsigned panels (gauss/wiener-style, default YlOrRd).",
    )
    p.add_argument(
        "--cmap-signed",
        default="seismic",
        help=(
            "Colormap for signed panels (raw/orig-style with negative bins; "
            "default seismic). Only consulted when --auto-signed-cmap is set "
            "AND a panel has no --vrange (default panels carry explicit ranges)."
        ),
    )
    p.add_argument(
        "--auto-signed-cmap",
        action="store_true",
        help=(
            "Use --cmap-signed (symmetric ±) when a panel has negative content "
            "AND no explicit --vrange.  Off by default; with the default vranges "
            "all three panels use the unsigned --cmap."
        ),
    )
    p.add_argument(
        "--vrange",
        action="append",
        metavar="LO,HI",
        help=(
            "Color scale (vmin,vmax) for a panel; use up to 3 times, matched in "
            "order.  Pass 'auto' for that panel to fall back to percentile "
            f"autoscaling.  Defaults: {' '.join(DEFAULT_VRANGES)}"
        ),
    )
    p.add_argument(
        "--out-prefix",
        default=None,
        metavar="PREFIX",
        help="Batch mode: save PREFIX_<hist>.pdf and .npy per panel, no GUI.",
    )
    p.add_argument(
        "--initial-channel",
        type=int,
        default=None,
        help=(
            "Channel to show first on the 1D panel; defaults to the first "
            "channel with nonzero content on panel 1."
        ),
    )
    return p.parse_args()


def import_root():
    try:
        import ROOT  # noqa: PLC0415

        ROOT.gErrorIgnoreLevel = ROOT.kWarning
        return ROOT
    except ImportError:
        sys.exit("PyROOT not available — source your LArSoft/PyROOT setup first.")


def resolve_panels(args):
    panels = list(args.panel) if args.panel else []
    # Pad to 3 with defaults.
    while len(panels) < 3:
        panels.append(DEFAULT_PANELS[len(panels)])
    if len(panels) > 3:
        sys.exit("at most 3 --panel arguments are supported")
    if args.h1: panels[0] = args.h1
    if args.h2: panels[1] = args.h2
    if args.h3: panels[2] = args.h3
    return panels


def resolve_vranges(args, n_panels):
    """Return a list of (vmin,vmax) tuples or None (=auto) per panel."""
    raw = list(args.vrange) if args.vrange else []
    # Pad with DEFAULT_VRANGES, then "auto" beyond that.
    while len(raw) < n_panels:
        idx = len(raw)
        raw.append(DEFAULT_VRANGES[idx] if idx < len(DEFAULT_VRANGES) else "auto")
    out = []
    for s in raw[:n_panels]:
        s = s.strip()
        if s.lower() == "auto":
            out.append(None)
            continue
        try:
            lo_s, hi_s = s.split(",")
            lo, hi = float(lo_s), float(hi_s)
        except ValueError:
            sys.exit(f"--vrange expects 'LO,HI' (or 'auto'), got {s!r}")
        if not (lo < hi):
            sys.exit(f"--vrange needs LO < HI, got {s!r}")
        out.append((lo, hi))
    return out


def th2_to_numpy(h):
    """Return (arr, (ch_lo, ch_hi), (tick_lo, tick_hi)).

    arr shape is (nchannels, nticks) so arr[ch_index, tick_index] is one cell.
    ch_lo/ch_hi are the low edge of the first bin and high edge of the last
    bin on the X axis (channel); analogously for tick on the Y axis.  TH2
    storage is laid out as a 1D array of size (nx+2)*(ny+2) including
    overflow rows/cols; we use frombuffer to avoid a Python-level loop.
    """
    nx = h.GetNbinsX()
    ny = h.GetNbinsY()
    size = (nx + 2) * (ny + 2)
    # Determine the storage dtype from the histogram class.
    cn = h.ClassName()
    if cn.endswith("F"):
        dtype = np.float32
    elif cn.endswith("D"):
        dtype = np.float64
    elif cn.endswith("I"):
        dtype = np.int32
    elif cn.endswith("S"):
        dtype = np.int16
    else:
        dtype = np.float32  # fallback
    buf = np.frombuffer(h.GetArray(), dtype=dtype, count=size).copy()
    # ROOT stores: bin index = iy*(nx+2) + ix, with ix,iy in 0..nx+1 / 0..ny+1.
    buf = buf.reshape(ny + 2, nx + 2)
    # Strip overflow/underflow, return as (nx, ny) so axis 0 == channel.
    arr = np.ascontiguousarray(buf[1:ny + 1, 1:nx + 1].T)
    xax = h.GetXaxis()
    yax = h.GetYaxis()
    ch_lo = int(round(xax.GetBinLowEdge(1)))
    ch_hi = int(round(xax.GetBinUpEdge(nx)))
    tick_lo = int(round(yax.GetBinLowEdge(1)))
    tick_hi = int(round(yax.GetBinUpEdge(ny)))
    return arr, (ch_lo, ch_hi), (tick_lo, tick_hi)


def open_panels(ROOT, filename, names):
    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        sys.exit(f"Cannot open file: {filename}")
    panels = []
    for n in names:
        h = f.Get(n)
        if not h:
            sys.exit(f"No histogram named {n!r} in {filename}")
        if not h.ClassName().startswith("TH2"):
            sys.exit(f"{n!r} is {h.ClassName()}, not a TH2*")
        arr, ch_range, tick_range = th2_to_numpy(h)
        panels.append({
            "name": n,
            "arr": arr,
            "ch_range": ch_range,
            "tick_range": tick_range,
        })
        print(
            f"  loaded {n}: ch {ch_range[0]}..{ch_range[1]} ({arr.shape[0]} ch), "
            f"tick {tick_range[0]}..{tick_range[1]} ({arr.shape[1]} bins)",
            file=sys.stderr,
        )
    return f, panels


def clip_panel(panel, view_ch, view_tick):
    """Clip a panel array to the requested view window.

    Returns a NEW panel dict with the trimmed array and updated ranges, plus
    the (ch_lo_new, tick_lo_new) origin used for indexing into the original.
    """
    ch_lo, ch_hi = panel["ch_range"]
    tick_lo, tick_hi = panel["tick_range"]
    new_ch_lo = ch_lo if view_ch[0] is None else max(ch_lo, view_ch[0])
    new_ch_hi = ch_hi if view_ch[1] is None else min(ch_hi, view_ch[1])
    new_tick_lo = tick_lo if view_tick[0] is None else max(tick_lo, view_tick[0])
    new_tick_hi = tick_hi if view_tick[1] is None else min(tick_hi, view_tick[1])
    if new_ch_hi < new_ch_lo or new_tick_hi < new_tick_lo:
        sys.exit(
            f"{panel['name']}: requested view [ch {view_ch}, tick {view_tick}] "
            f"does not overlap histogram range "
            f"[ch {ch_lo}..{ch_hi}, tick {tick_lo}..{tick_hi}]"
        )
    ci_lo = new_ch_lo - ch_lo
    ci_hi = new_ch_hi - ch_lo + 1
    ti_lo = new_tick_lo - tick_lo
    ti_hi = new_tick_hi - tick_lo + 1
    return {
        "name": panel["name"],
        "arr": panel["arr"][ci_lo:ci_hi, ti_lo:ti_hi],
        "ch_range": (new_ch_lo, new_ch_hi),
        "tick_range": (new_tick_lo, new_tick_hi),
        "full_arr": panel["arr"],
        "full_ch_range": panel["ch_range"],
        "full_tick_range": panel["tick_range"],
    }


def warn_size(panel):
    nch, nt = panel["arr"].shape
    total = nch * nt
    if total > _ARRAY_WARN_LIMIT:
        mb = total * 4 / 1e6
        print(
            f"  WARNING: {panel['name']} array is {nch}x{nt} = {total:,} bins (~{mb:.0f} MB). "
            "Consider narrowing --channel-min/max or --tick-min/max.",
            file=sys.stderr,
        )


def is_signed(arr):
    """A panel is treated as 'signed' when its data carries negative bins
    (raw/orig waveforms, including bipolar induction-plane signals).  We
    detect this empirically rather than parsing the histogram name so users
    can drop unusual hist names into --panel without renaming the logic."""
    if arr.size == 0:
        return False
    # Cheap short-circuit: only inspect a sample if the array is huge.
    if arr.size > 10_000_000:
        sample = arr.ravel()[::max(1, arr.size // 1_000_000)]
        return bool((sample < 0).any())
    return bool((arr < 0).any())


def auto_vmax(arr, percentile):
    nonzero = np.abs(arr[arr != 0])
    if nonzero.size == 0:
        return 1.0
    return float(np.percentile(nonzero, percentile))


def imshow_kwargs(arr, cmap, cmap_signed, vmax_percentile,
                  vrange=None, auto_signed=False):
    # Explicit per-panel range always wins and always uses the unsigned cmap.
    if vrange is not None:
        vmin, vmax = vrange
        return dict(
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
    # No explicit range: percentile autoscale, optionally with signed-cmap
    # auto-detection (opt-in via --auto-signed-cmap).
    if auto_signed and is_signed(arr):
        v = auto_vmax(arr, vmax_percentile)
        return dict(
            aspect="auto",
            origin="lower",
            cmap=cmap_signed,
            vmin=-v,
            vmax=+v,
            interpolation="nearest",
        )
    return dict(
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=auto_vmax(arr, vmax_percentile),
        interpolation="nearest",
    )


def draw_2d_panel(ax, panel, cmap, cmap_signed, vmax_percentile, fig,
                  vrange=None, auto_signed=False):
    arr = panel["arr"]
    ch_lo, ch_hi = panel["ch_range"]
    tick_lo, tick_hi = panel["tick_range"]
    extent = [ch_lo - 0.5, ch_hi + 0.5, tick_lo - 0.5, tick_hi + 0.5]
    kw = imshow_kwargs(arr, cmap, cmap_signed, vmax_percentile,
                       vrange=vrange, auto_signed=auto_signed)
    im = ax.imshow(arr.T, extent=extent, **kw)
    fig.colorbar(im, ax=ax, pad=0.02)
    # Mark "(signed)" only when we actually selected the signed cmap.
    sign_tag = " (signed)" if (vrange is None and auto_signed and is_signed(arr)) else ""
    ax.set_title(panel["name"] + sign_tag)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Tick")
    return im


def save_static(panels, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = args.out_prefix
    vranges = resolve_vranges(args, len(panels))
    for panel, vr in zip(panels, vranges):
        fig, ax = plt.subplots(figsize=(14, 5))
        draw_2d_panel(
            ax, panel, args.cmap, args.cmap_signed, args.vmax_percentile, fig,
            vrange=vr, auto_signed=args.auto_signed_cmap,
        )
        fig.suptitle(f"{args.input}  {panel['name']}", fontsize=9)
        plt.tight_layout()
        pdf_path = f"{prefix}_{panel['name']}.pdf"
        npy_path = f"{prefix}_{panel['name']}.npy"
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        np.save(npy_path, panel["arr"])
        print(f"Saved {pdf_path}  {npy_path}", file=sys.stderr)


def first_nonzero_channel(panel):
    """Channel value (not index) of the first column with any nonzero bin."""
    col_has = np.any(panel["arr"] != 0, axis=1)
    idx = np.flatnonzero(col_has)
    if idx.size:
        return int(panel["ch_range"][0] + idx[0])
    return int(panel["ch_range"][0])


def channel_waveform(panel, ch):
    """Return (tick_axis, wave) for channel `ch` from this panel, or (None, None)
    if the channel is outside this panel's range."""
    ch_lo, ch_hi = panel["ch_range"]
    if ch < ch_lo or ch > ch_hi:
        return None, None
    idx = ch - ch_lo
    tick_lo, tick_hi = panel["tick_range"]
    n = panel["arr"].shape[1]
    ticks = np.arange(tick_lo, tick_lo + n)
    return ticks, panel["arr"][idx]


def show_interactive(panels, args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    # Outer gridspec:
    #   col 0 (wide): 2D panels + 1D panel + bottom control row
    #   col 1 (narrow): per-panel vmin/vmax controls aligned with each 2D row
    # The 1D and control rows span both columns so they keep their full width.
    fig = plt.figure(figsize=(16, 16))
    outer = fig.add_gridspec(
        5, 2,
        height_ratios=[2, 2, 2, 1.4, 1.1],
        width_ratios=[6, 1],
        hspace=0.55,
        wspace=0.05,
    )
    ax2d = [fig.add_subplot(outer[i, 0]) for i in range(3)]
    right_axes = [fig.add_subplot(outer[i, 1]) for i in range(3)]
    for rax in right_axes:
        rax.set_axis_off()
    ax_1d = fig.add_subplot(outer[3, :])
    ax_ctrl = fig.add_subplot(outer[4, :])
    ax_ctrl.set_axis_off()
    # right=0.86 leaves room for the third offset spine on the 1D twin axis
    # (which extends ~0.08 axes-coord beyond the host's right edge).
    fig.subplots_adjust(left=0.07, right=0.86, top=0.96, bottom=0.05)

    # Draw each 2D panel and remember the imshow artist so vmin/vmax textboxes
    # below can update the colormap range on the fly without redrawing.
    vranges = resolve_vranges(args, len(panels))
    for ax, panel, vr in zip(ax2d, panels, vranges):
        panel["im"] = draw_2d_panel(
            ax, panel, args.cmap, args.cmap_signed, args.vmax_percentile, fig,
            vrange=vr, auto_signed=args.auto_signed_cmap,
        )

    # Marker line on each 2D panel showing the currently selected channel.
    vlines = [
        ax.axvline(panels[0]["ch_range"][0], color="cyan", lw=0.8, ls="--", visible=False)
        for ax in ax2d
    ]

    # The view ranges to apply to all 2D panels.  Each panel knows its own
    # (broader) full range; the view state below is the intersection across
    # panels for display purposes.
    view_ch = list(panels[0]["ch_range"])
    view_tick = list(panels[0]["tick_range"])
    for panel in panels[1:]:
        view_ch[0] = min(view_ch[0], panel["ch_range"][0])
        view_ch[1] = max(view_ch[1], panel["ch_range"][1])
        view_tick[0] = min(view_tick[0], panel["tick_range"][0])
        view_tick[1] = max(view_tick[1], panel["tick_range"][1])

    state = {
        "channel": (
            args.initial_channel
            if args.initial_channel is not None
            else first_nonzero_channel(panels[0])
        ),
        "ch_min": view_ch[0],
        "ch_max": view_ch[1],
        "tick_min": view_tick[0],
        "tick_max": view_tick[1],
    }
    # Per-panel color scale state: initial values come from imshow_kwargs'
    # autoscale (so signed panels start symmetric); originals are kept around
    # for the "Reset colors" button.
    for i, panel in enumerate(panels):
        vmin, vmax = panel["im"].get_clim()
        state[f"vmin_{i}"] = float(vmin)
        state[f"vmax_{i}"] = float(vmax)
        state[f"vmin_{i}_orig"] = float(vmin)
        state[f"vmax_{i}_orig"] = float(vmax)

    # Extra y-axes created on the 1D panel for traces 2/3; tracked here so we
    # remove them cleanly before each update_1d redraw.
    twin_state = {"axes": []}

    status = ax_ctrl.text(0.01, 0.97, "", fontsize=9, transform=ax_ctrl.transAxes)

    def set_textbox(widget, val):
        # Without try/finally a stray exception from set_val (e.g. a draw
        # failure mid-update) would leave eventson=False and the widget
        # would silently stop responding to user input from then on.
        widget.eventson = False
        try:
            widget.set_val(str(val))
        finally:
            widget.eventson = True

    def update_1d(ch):
        # Tear down twin axes from the previous redraw before clearing ax_1d.
        for tax in twin_state["axes"]:
            try:
                tax.remove()
            except Exception:
                pass
        twin_state["axes"] = []
        ax_1d.cla()

        colors = ["steelblue", "forestgreen", "darkorange"]
        summary = []
        legend_handles, legend_labels = [], []
        any_drawn = False

        for i, (color, panel) in enumerate(zip(colors, panels)):
            ticks, wave = channel_waveform(panel, ch)
            if ticks is None:
                summary.append(f"{panel['name']}: ch {ch} OOR")
                continue

            # Each trace lives on its own y-axis pinned to that panel's
            # (vmin, vmax).  Trace 0 uses the host axis; trace 1 gets a
            # right twinx; trace 2 gets a further-offset right twinx.
            if i == 0:
                this_ax = ax_1d
            else:
                this_ax = ax_1d.twinx()
                twin_state["axes"].append(this_ax)
                if i >= 2:
                    this_ax.spines["right"].set_position(("axes", 1.0 + 0.08 * (i - 1)))

            mask = (ticks >= state["tick_min"]) & (ticks <= state["tick_max"])
            vis = wave[mask]
            integral = float(vis.sum()) if vis.size else 0.0
            (ln,) = this_ax.plot(
                ticks, wave, color=color, lw=0.9,
                label=f"{panel['name']} ch={ch}  ∫={integral:.4g}",
            )

            # Sync the y-axis range to the 2D panel's color limits.
            this_ax.set_ylim(state[f"vmin_{i}"], state[f"vmax_{i}"])
            this_ax.tick_params(axis="y", labelcolor=color)
            if i == 0:
                this_ax.spines["left"].set_color(color)
            else:
                this_ax.spines["right"].set_color(color)
            this_ax.set_ylabel(panel["name"], color=color, fontsize=8)

            legend_handles.append(ln)
            legend_labels.append(ln.get_label())
            any_drawn = True
            summary.append(f"{panel['name']} ∫={integral:.4g}")

        if any_drawn:
            ax_1d.axhline(0, color="black", lw=0.3, ls=":")
            ax_1d.set_xlim(state["tick_min"], state["tick_max"])
            ax_1d.legend(legend_handles, legend_labels, loc="upper right", fontsize=8)
        ax_1d.set_xlabel("Tick")
        ax_1d.set_title(f"Channel {ch}    " + "   ".join(summary))

        for vl in vlines:
            vl.set_xdata([ch, ch])
            vl.set_visible(True)

        state["channel"] = ch
        set_textbox(channel_box, ch)
        fig.canvas.draw_idle()

    def apply_view_ranges():
        for ax, panel in zip(ax2d, panels):
            ax.set_xlim(state["ch_min"] - 0.5, state["ch_max"] + 0.5)
            ax.set_ylim(state["tick_min"] - 0.5, state["tick_max"] + 0.5)
        # Re-plot 1D within the new tick window
        update_1d(state["channel"])

    def set_channel_range(ch_min, ch_max):
        if ch_min > ch_max:
            status.set_text("Channel min must be <= max.")
            fig.canvas.draw_idle()
            return
        state["ch_min"] = ch_min
        state["ch_max"] = ch_max
        set_textbox(ch_min_box, ch_min)
        set_textbox(ch_max_box, ch_max)
        apply_view_ranges()

    def set_tick_range(tk_min, tk_max):
        if tk_min > tk_max:
            status.set_text("Tick min must be <= max.")
            fig.canvas.draw_idle()
            return
        state["tick_min"] = tk_min
        state["tick_max"] = tk_max
        set_textbox(tick_min_box, tk_min)
        set_textbox(tick_max_box, tk_max)
        apply_view_ranges()

    def on_click(event):
        if event.inaxes not in ax2d:
            return
        if event.xdata is None:
            return
        ch = int(round(event.xdata))
        update_1d(ch)

    def submit_channel(text):
        try:
            ch = int(text.strip())
        except ValueError:
            status.set_text("Enter an integer channel number.")
            fig.canvas.draw_idle()
            return
        update_1d(ch)

    def step_channel(delta):
        update_1d(state["channel"] + delta)

    def submit_ch_range(_text):
        try:
            lo = int(ch_min_box.text.strip())
            hi = int(ch_max_box.text.strip())
        except ValueError:
            status.set_text("Enter integer Ch min/max.")
            fig.canvas.draw_idle()
            return
        set_channel_range(lo, hi)

    def submit_tick_range(_text):
        try:
            lo = int(tick_min_box.text.strip())
            hi = int(tick_max_box.text.strip())
        except ValueError:
            status.set_text("Enter integer Tick min/max.")
            fig.canvas.draw_idle()
            return
        set_tick_range(lo, hi)

    def reset_ch(_evt):
        full_lo = min(p["ch_range"][0] for p in panels)
        full_hi = max(p["ch_range"][1] for p in panels)
        set_channel_range(full_lo, full_hi)

    def reset_tick(_evt):
        full_lo = min(p["tick_range"][0] for p in panels)
        full_hi = max(p["tick_range"][1] for p in panels)
        set_tick_range(full_lo, full_hi)

    # Right column: per-panel [vmin] / [vmax] textboxes + tiny "↺" reset.
    # Each pair sits in its own right-column cell, aligned vertically with the
    # corresponding 2D panel.
    vmin_boxes = []
    vmax_boxes = []
    reset_color_btns = []

    def make_vrange_handler(idx):
        def handler(_text):
            try:
                vmin = float(vmin_boxes[idx].text.strip())
                vmax = float(vmax_boxes[idx].text.strip())
            except ValueError:
                status.set_text(f"P{idx+1} ({panels[idx]['name']}): enter numeric vmin/vmax.")
                fig.canvas.draw_idle()
                return
            if vmin >= vmax:
                status.set_text(f"P{idx+1} ({panels[idx]['name']}): vmin must be < vmax.")
                fig.canvas.draw_idle()
                return
            state[f"vmin_{idx}"] = vmin
            state[f"vmax_{idx}"] = vmax
            panels[idx]["im"].set_clim(vmin, vmax)
            update_1d(state["channel"])
        return handler

    def make_reset_color_handler(idx):
        def handler(_evt):
            v0 = state[f"vmin_{idx}_orig"]
            v1 = state[f"vmax_{idx}_orig"]
            state[f"vmin_{idx}"] = v0
            state[f"vmax_{idx}"] = v1
            panels[idx]["im"].set_clim(v0, v1)
            set_textbox(vmin_boxes[idx], v0)
            set_textbox(vmax_boxes[idx], v1)
            update_1d(state["channel"])
        return handler

    for i, rax in enumerate(right_axes):
        # Three rows inside the right-column cell:
        #   vmin textbox (top), vmax textbox (middle), reset button (bottom).
        vmin_ax = rax.inset_axes([0.20, 0.66, 0.75, 0.22])
        vmax_ax = rax.inset_axes([0.20, 0.36, 0.75, 0.22])
        rst_ax  = rax.inset_axes([0.20, 0.06, 0.75, 0.22])
        vmin_box = TextBox(vmin_ax, "vmin ")
        vmax_box = TextBox(vmax_ax, "vmax ")
        rst_btn  = Button(rst_ax, "auto")
        h = make_vrange_handler(i)
        vmin_box.on_submit(h)
        vmax_box.on_submit(h)
        rst_btn.on_clicked(make_reset_color_handler(i))
        vmin_boxes.append(vmin_box)
        vmax_boxes.append(vmax_box)
        reset_color_btns.append(rst_btn)
        set_textbox(vmin_box, state[f"vmin_{i}"])
        set_textbox(vmax_box, state[f"vmax_{i}"])

    # Bottom control row: two sub-rows inside ax_ctrl.
    # Row 1 (textboxes):  [Channel] [Ch min] [Ch max] [Tick min] [Tick max]
    # Row 2 (buttons):    [Prev]    [Next]   [Full Channel]      [Full Tick]
    channel_ax = ax_ctrl.inset_axes([0.07, 0.55, 0.10, 0.32])
    channel_box = TextBox(channel_ax, "Channel ")
    channel_box.on_submit(submit_channel)

    ch_min_ax = ax_ctrl.inset_axes([0.27, 0.55, 0.09, 0.32])
    ch_max_ax = ax_ctrl.inset_axes([0.42, 0.55, 0.09, 0.32])
    ch_min_box = TextBox(ch_min_ax, "Ch min ")
    ch_max_box = TextBox(ch_max_ax, "Ch max ")
    ch_min_box.on_submit(submit_ch_range)
    ch_max_box.on_submit(submit_ch_range)

    tick_min_ax = ax_ctrl.inset_axes([0.62, 0.55, 0.09, 0.32])
    tick_max_ax = ax_ctrl.inset_axes([0.78, 0.55, 0.09, 0.32])
    tick_min_box = TextBox(tick_min_ax, "Tick min ")
    tick_max_box = TextBox(tick_max_ax, "Tick max ")
    tick_min_box.on_submit(submit_tick_range)
    tick_max_box.on_submit(submit_tick_range)

    prev_ax = ax_ctrl.inset_axes([0.07, 0.10, 0.07, 0.32])
    next_ax = ax_ctrl.inset_axes([0.17, 0.10, 0.07, 0.32])
    reset_ch_ax = ax_ctrl.inset_axes([0.30, 0.10, 0.16, 0.32])
    reset_tk_ax = ax_ctrl.inset_axes([0.62, 0.10, 0.16, 0.32])
    prev_btn = Button(prev_ax, "Prev")
    next_btn = Button(next_ax, "Next")
    reset_ch_btn = Button(reset_ch_ax, "Full Channel")
    reset_tk_btn = Button(reset_tk_ax, "Full Tick")
    prev_btn.on_clicked(lambda _e: step_channel(-1))
    next_btn.on_clicked(lambda _e: step_channel(+1))
    reset_ch_btn.on_clicked(reset_ch)
    reset_tk_btn.on_clicked(reset_tick)

    fig._magnify_widgets = (
        *vmin_boxes, *vmax_boxes, *reset_color_btns,
        channel_box, ch_min_box, ch_max_box, tick_min_box, tick_max_box,
        prev_btn, next_btn, reset_ch_btn, reset_tk_btn,
    )

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.suptitle(
        f"{args.input}    panels: {', '.join(p['name'] for p in panels)}",
        fontsize=10,
    )

    # Initial view: honour CLI --channel-min/max --tick-min/max if given,
    # else show the union of panel ranges.
    init_ch_lo = args.channel_min if args.channel_min is not None else state["ch_min"]
    init_ch_hi = args.channel_max if args.channel_max is not None else state["ch_max"]
    init_tk_lo = args.tick_min if args.tick_min is not None else state["tick_min"]
    init_tk_hi = args.tick_max if args.tick_max is not None else state["tick_max"]
    set_textbox(channel_box, state["channel"])
    set_channel_range(init_ch_lo, init_ch_hi)
    set_tick_range(init_tk_lo, init_tk_hi)
    update_1d(state["channel"])

    plt.show()


def main():
    args = parse_args()
    names = resolve_panels(args)
    print(f"Opening {args.input} with panels {names}", file=sys.stderr)

    ROOT = import_root()
    f, panels = open_panels(ROOT, args.input, names)

    # In batch mode we save each (full or clipped) panel; in interactive mode
    # we let the user adjust the view at runtime, so the initial array stays
    # full and CLI --channel-min/max only seeds the textboxes.
    if args.out_prefix:
        view_ch = (args.channel_min, args.channel_max)
        view_tick = (args.tick_min, args.tick_max)
        clipped = [clip_panel(p, view_ch, view_tick) for p in panels]
        for p in clipped:
            warn_size(p)
        save_static(clipped, args)
    else:
        for p in panels:
            warn_size(p)
        show_interactive(panels, args)

    f.Close()


if __name__ == "__main__":
    main()

"""Bokeh viewer for PDVD top U-plane shield coupling NF diagnostic.

Reads the npz dumps written by PDVDShieldCouplingSub (when dump_path is set)
and the corresponding magnify ROOT files, then shows four stacked panels:

  1. Input waveform 2D (post-coherent-NF, pre-shield) with signal-mask overlay
  2. Signal mask 2D (1 = signal-protected, excluded from median pool)
  3. Median waveform 1D (the per-tick correction being subtracted)
  4. Output waveform 2D (after shield subtraction) vs reference hu_raw

Below the panels: a 1D wire cross-section showing all quantities for one wire,
plus a Python re-implementation of the algorithm with tunable parameters so
you can preview the effect of changing sig_factor / pad_bins / outlier_factor
without rerunning WCT.

Usage:
    bokeh serve --port 5006 shield_tune_viewer.py --args <dump_dir> [<magnify_dir>]

    dump_dir    directory containing shield_dump_ch*.npz files
    magnify_dir directory containing magnify-*.root files (optional;
                supplies the hu_raw reference and hu_orig pre-NF view)

Remote access:
    ssh -L 5006:localhost:5006 user@workstation
    open http://localhost:5006/shield_tune_viewer
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColorBar,
    ColumnDataSource,
    Div,
    LinearColorMapper,
    Range1d,
    RangeSlider,
    Select,
    Slider,
    Spinner,
    Toggle,
)
from bokeh.palettes import RdBu11, Viridis256
from bokeh.plotting import figure

# ─── constants ───────────────────────────────────────────────────────────────

# top_u_groups from chndb-base.jsonnet: anode ident → first channel of the group
ANODE_FIRST_CH = {4: 6144, 5: 6620, 6: 9216, 7: 9692}
GROUP_SIZE = 476

# ─── pure-Python re-implementation of the shield algorithm ───────────────────

def _rms_clean(arr):
    """RMS ignoring +200000 flag markers."""
    clean = arr[np.abs(arr) <= 100000.0]
    if len(clean) == 0:
        return 0.0
    return float(np.std(clean))


def python_signal_mask(wf_in_norm, sig_factor=4.0, pad_bins=70, bipolar=False):
    """Return boolean mask (nch, ntick): True = signal-protected.

    wf_in_norm: (nch, ntick) float array, strip-length-normalized waveforms
                WITHOUT +200000 sentinels — i.e. the clean normalized input.
    """
    from scipy.ndimage import binary_dilation
    nch, ntick = wf_in_norm.shape
    mask = np.zeros((nch, ntick), dtype=bool)
    kernel = np.ones(2 * pad_bins + 1, dtype=bool)
    for ich in range(nch):
        sig = wf_in_norm[ich]
        rms = _rms_clean(sig)
        thresh = sig_factor * rms
        flagged = (sig > thresh) & (sig < 16384.0)
        if bipolar:
            flagged |= (sig < -thresh)
        if flagged.any():
            mask[ich] = binary_dilation(flagged, structure=kernel)
    return mask


def python_compute_median(wf_in_norm, signal_mask, outlier_factor=5.0):
    """Return (medians, max_rms, pool_mask).

    medians:    (ntick,) float — per-tick median in strip-normalized units
    max_rms:    float — cross-channel mean RMS used for outlier rejection
    pool_mask:  (nch, ntick) bool — True = channel/tick enters the median pool
    """
    nch, ntick = wf_in_norm.shape
    # Per-channel RMS excluding masked samples and flag markers
    channel_rms = []
    for ich in range(nch):
        clean = wf_in_norm[ich, ~signal_mask[ich]]
        clean = clean[np.abs(clean) <= 100000.0]
        if len(clean) > 0:
            rms = float(np.std(clean))
            if rms > 0:
                channel_rms.append(rms)
    max_rms = float(np.mean(channel_rms)) if channel_rms else 1.0

    # Pool: not signal-masked AND below outlier threshold AND non-zero
    pool_mask = (
        ~signal_mask
        & (wf_in_norm < outlier_factor * max_rms)
        & (np.abs(wf_in_norm) > 0.001)
    )

    # Per-tick median over pool (vectorized via nanmedian)
    work = np.where(pool_mask, wf_in_norm, np.nan)
    medians = np.nanmedian(work, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)

    return medians, max_rms, pool_mask


def python_subtract(wf_in_raw, strip_lengths, medians_norm):
    """Apply strip-length normalization, subtract medians, restore.

    Returns wf_out_raw: (nch, ntick) in physical ADC units.
    """
    nch, ntick = wf_in_raw.shape
    sl = strip_lengths[:, np.newaxis]   # (nch, 1)
    normalized = wf_in_raw / sl
    subtracted = normalized - medians_norm[np.newaxis, :]
    # Zero-tolerance: skip exactly-zero samples (matches C++)
    subtracted = np.where(np.abs(normalized) > 0.001, subtracted, normalized)
    return subtracted * sl


# ─── data loading ────────────────────────────────────────────────────────────

def load_dump(dump_dir, anode_ident):
    """Load npz dump for a top anode.  Returns dict or None."""
    first_ch = ANODE_FIRST_CH.get(anode_ident)
    if first_ch is None:
        return None
    path = os.path.join(dump_dir, f"shield_dump_ch{first_ch}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def load_magnify_u(magnify_dir, anode_ident):
    """Load hu_orig and hu_raw 2D arrays from the magnify ROOT file.
    Returns (hu_orig, hu_raw, ch_centers, tick_centers) or None.
    """
    if not magnify_dir:
        return None
    pat = os.path.join(magnify_dir, f"*anode{anode_ident}.root")
    matches = glob.glob(pat)
    if not matches:
        return None
    try:
        import uproot
        f = uproot.open(matches[0])
        def _load(name):
            h = f[name]
            vals, xedges, yedges = h.to_numpy()
            # vals shape: (nch, ntick); X = channel, Y = tick
            ch_centers  = 0.5 * (xedges[:-1] + xedges[1:])
            tick_centers = 0.5 * (yedges[:-1] + yedges[1:])
            return vals, ch_centers, tick_centers
        orig_vals, ch_c, tick_c = _load(f"hu_orig{anode_ident}")
        raw_vals,  _,    _      = _load(f"hu_raw{anode_ident}")
        return dict(hu_orig=orig_vals, hu_raw=raw_vals,
                    ch_centers=ch_c, tick_centers=tick_c)
    except Exception as e:
        print(f"[magnify load] {e}")
        return None


# ─── helpers ─────────────────────────────────────────────────────────────────

def _sym_range(arr, pct=99.5):
    """Symmetric color range around zero at the given percentile."""
    v = float(np.nanpercentile(np.abs(arr), pct))
    return -v, v


def _make_image_fig(title, x_range=None, y_range=None, height=220):
    kw = dict(height=height, sizing_mode="stretch_width",
              tools="pan,wheel_zoom,box_zoom,reset,save",
              active_scroll="wheel_zoom")
    if x_range is not None:
        kw["x_range"] = x_range
    if y_range is not None:
        kw["y_range"] = y_range
    f = figure(title=title, **kw)
    f.xaxis.axis_label = "tick"
    f.yaxis.axis_label = "channel"
    return f


# ─── main ────────────────────────────────────────────────────────────────────

def main(argv):
    if len(argv) < 2:
        err = Div(text="<b>Usage:</b> bokeh serve shield_tune_viewer.py --args &lt;dump_dir&gt; [&lt;magnify_dir&gt;]",
                  width=800)
        curdoc().add_root(err)
        return

    dump_dir    = os.path.abspath(argv[1])
    magnify_dir = os.path.abspath(argv[2]) if len(argv) > 2 else ""

    # Discover which anodes have dumps.
    available_anodes = sorted(
        a for a in ANODE_FIRST_CH
        if os.path.exists(os.path.join(
            dump_dir, f"shield_dump_ch{ANODE_FIRST_CH[a]}.npz")))

    if not available_anodes:
        # Show error in browser rather than killing the server with sys.exit.
        print(f"No shield dump files found in {dump_dir}", file=sys.stderr)
        print(f"  sys.argv = {sys.argv}", file=sys.stderr)
        err = Div(text=f"<b>Error:</b> no shield_dump_ch*.npz files found in:<br><code>{dump_dir}</code>",
                  width=800)
        curdoc().add_root(err)
        curdoc().title = "Shield viewer — no data"
        return

    # ── state ──────────────────────────────────────────────────────────────
    state = dict(anode=available_anodes[0], dump=None, magnify=None,
                 py_mask=None, py_medians=None, py_pool_mask=None,
                 py_wf_out=None)

    def load_anode(anode_ident):
        state["anode"]   = anode_ident
        state["dump"]    = load_dump(dump_dir, anode_ident)
        state["magnify"] = load_magnify_u(magnify_dir, anode_ident)
        state["py_mask"] = state["py_medians"] = state["py_pool_mask"] = state["py_wf_out"] = None

    load_anode(available_anodes[0])

    # ── widgets ────────────────────────────────────────────────────────────
    anode_select   = Select(title="Anode", width=100,
                            options=[str(a) for a in available_anodes],
                            value=str(available_anodes[0]))
    sig_factor_sp  = Spinner(title="sig_factor", value=4.0, step=0.5, low=0.5, high=20.0, width=100)
    pad_bins_sp    = Spinner(title="pad_bins",   value=70,  step=5,   low=0,   high=200,  width=100)
    outlier_sp     = Spinner(title="outlier_factor", value=5.0, step=0.5, low=1.0, high=20.0, width=100)
    bipolar_tog    = Toggle(label="Bipolar mask (also mask negative signal)", active=False, width=250)
    apply_btn      = Button(label="Recompute (Python)", button_type="primary", width=180)
    info_div       = Div(text="", width=900)
    _first_ch0 = ANODE_FIRST_CH[available_anodes[0]]
    wire_sp        = Spinner(title="Channel", value=_first_ch0, step=1,
                             low=_first_ch0, high=_first_ch0 + GROUP_SIZE - 1, width=120)

    # ── color mappers ──────────────────────────────────────────────────────
    palette_div = RdBu11[::-1]   # blue=negative, red=positive for waveform

    mapper_in   = LinearColorMapper(palette=palette_div, low=-20, high=20)
    mapper_corr = LinearColorMapper(palette=palette_div, low=-20, high=20)
    mapper_out  = LinearColorMapper(palette=palette_div, low=-20, high=20)
    mapper_ref  = LinearColorMapper(palette=palette_div, low=-20, high=20)
    mapper_mask = LinearColorMapper(palette=["#333333", "#ff4444"], low=0, high=1)
    mapper_pool = LinearColorMapper(
        palette=["#888888", "#44cc44", "#ff4444"],  # grey / green=pool / red=protected
        low=0, high=2)

    # ── data sources ───────────────────────────────────────────────────────
    src_in    = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    src_mask  = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    src_pool  = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    src_corr  = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))  # 2D correction
    src_out   = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    src_ref   = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    src_med_c = ColumnDataSource(dict(x=[], y=[]))   # C++ median
    src_med_p = ColumnDataSource(dict(x=[], y=[]))   # Python median
    src_med_band = ColumnDataSource(dict(x=[], top=[], bottom=[]))

    # 1D wire cross-section sources
    src_w_in  = ColumnDataSource(dict(x=[], y=[]))
    src_w_out = ColumnDataSource(dict(x=[], y=[]))
    src_w_ref = ColumnDataSource(dict(x=[], y=[]))
    src_w_po  = ColumnDataSource(dict(x=[], y=[]))   # Python output
    src_w_med = ColumnDataSource(dict(x=[], y=[]))   # median (strip-scaled for this wire)

    # ── figures ─────────────────────────────────────────────────────────────
    f_in   = _make_image_fig("Input to shield filter (post-coherent NF, pre-shield)",  height=230)
    f_in.x_range = Range1d(5500, 6400)  # zoom to late-tick region where shield artifact is visible
    f_mask = _make_image_fig("Signal-protection mask (red=protected; green=in median pool)",
                             x_range=f_in.x_range, height=200)
    f_corr = _make_image_fig("Correction being subtracted: median × strip_length (ADC units)",
                             x_range=f_in.x_range, height=200)
    f_out  = _make_image_fig("Output: Python result (top) / C++ reference hu_raw (bottom)",
                             x_range=f_in.x_range, height=260)

    f_in.image  ("image", source=src_in,   x="x", y="y", dw="dw", dh="dh",
                 color_mapper=mapper_in)
    f_mask.image("image", source=src_pool, x="x", y="y", dw="dw", dh="dh",
                 color_mapper=mapper_pool)
    f_corr.image("image", source=src_corr, x="x", y="y", dw="dw", dh="dh",
                 color_mapper=mapper_corr)
    f_out.image ("image", source=src_out,  x="x", y="y", dw="dw", dh="dh",
                 color_mapper=mapper_out)
    f_out.image ("image", source=src_ref,  x="x", y="y", dw="dw", dh="dh",
                 color_mapper=mapper_ref)

    f_in.add_layout(ColorBar(color_mapper=mapper_in,   label_standoff=6, width=8), "right")
    f_corr.add_layout(ColorBar(color_mapper=mapper_corr, label_standoff=6, width=8), "right")
    f_out.add_layout(ColorBar(color_mapper=mapper_out,  label_standoff=6, width=8), "right")

    # Median 1D figure
    f_med = figure(title="Median waveform (strip-normalized units; dashed = Python recompute)",
                   height=180, sizing_mode="stretch_width",
                   x_range=f_in.x_range,
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   active_scroll="wheel_zoom")
    f_med.xaxis.axis_label = "tick"
    f_med.yaxis.axis_label = "ADC (norm)"
    f_med.varea("x", "bottom", "top", source=src_med_band,
                fill_color="gray", fill_alpha=0.2, legend_label="±max_rms")
    f_med.line("x", "y", source=src_med_c, color="steelblue", line_width=1.5,
               legend_label="C++ median")
    f_med.line("x", "y", source=src_med_p, color="tomato",    line_width=1.5,
               line_dash="dashed", legend_label="Python median")
    f_med.legend.click_policy = "hide"

    # Wire cross-section figure
    f_wire = figure(title="Wire cross-section (select wire with spinner below)",
                    height=220, sizing_mode="stretch_width",
                    x_range=f_in.x_range,
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    active_scroll="wheel_zoom")
    f_wire.xaxis.axis_label = "tick"
    f_wire.yaxis.axis_label = "ADC"
    f_wire.line("x", "y", source=src_w_in,  color="gray",      line_width=1,
                legend_label="input (wf_in_raw)")
    f_wire.line("x", "y", source=src_w_ref, color="steelblue", line_width=1,
                legend_label="hu_raw (full NF reference)")
    f_wire.line("x", "y", source=src_w_out, color="orange",    line_width=1.5,
                legend_label="C++ output (wf_out_raw)")
    f_wire.line("x", "y", source=src_w_po,  color="tomato",    line_width=1.5,
                line_dash="dashed", legend_label="Python output")
    f_wire.line("x", "y", source=src_w_med, color="green",     line_width=1,
                line_dash="dotted", legend_label="median × strip_len")
    f_wire.legend.click_policy = "hide"

    # ── update helpers ─────────────────────────────────────────────────────

    def _image_dict(arr2d, ticks, chs, half=None):
        """Build image ColumnDataSource dict from a (nch, ntick) array.

        half: None = full height; 'top' = upper half; 'bottom' = lower half.
        """
        nch, ntick = arr2d.shape
        t0, t1 = float(ticks[0]), float(ticks[-1])
        c0, c1 = float(chs[0]),  float(chs[-1])
        dw = t1 - t0 + 1.0
        full_dh = c1 - c0 + 1.0

        if half is None:
            img = arr2d
            y0, dh = c0 - 0.5, full_dh
        elif half == "top":
            mid = c0 + full_dh / 2.0
            img = arr2d[:nch // 2, :]
            y0, dh = mid, full_dh / 2.0
        else:   # "bottom"
            mid = c0 + full_dh / 2.0
            img = arr2d[nch // 2:, :]
            y0, dh = c0 - 0.5, full_dh / 2.0

        return dict(image=[img.astype(np.float32)],
                    x=[t0 - 0.5], y=[y0], dw=[dw], dh=[dh])

    def update_2d():
        dump = state["dump"]
        if dump is None:
            for src in (src_in, src_mask, src_pool, src_corr, src_out, src_ref):
                src.data = dict(image=[], x=[], y=[], dw=[], dh=[])
            return

        channels = dump["channels"].astype(int)    # (nch,)
        wf_in    = dump["wf_in_raw"]               # (nch, ntick)
        nch, ntick = wf_in.shape
        ticks = np.arange(ntick, dtype=float)

        # Determine C++ mask from signal_mask array
        cpp_mask = dump["signal_mask"].astype(np.uint8)  # (nch, ntick)

        # Compute pool map: 0=outlier/masked, 1=in-pool (green), 2=signal-protected (red)
        py_mask = state.get("py_mask")
        pool_map = np.zeros_like(cpp_mask, dtype=np.uint8)
        if py_mask is not None:
            pool_mask = state.get("py_pool_mask")
            if pool_mask is not None:
                pool_map[pool_mask]  = 1   # green: in pool
                pool_map[py_mask]    = 2   # red: signal-protected (overrides)
        else:
            # Fall back to C++ mask
            pool_map[cpp_mask == 0] = 1
            pool_map[cpp_mask == 1] = 2

        # Input image: (nch, ntick) → rows=channel (y), cols=tick (x), no transpose needed
        vlo, vhi = _sym_range(wf_in)
        mapper_in.update(low=vlo, high=vhi)
        src_in.data = _image_dict(wf_in, ticks, channels)

        # Pool/mask image: same orientation
        src_pool.data = _image_dict(pool_map.astype(np.float32), ticks, channels)

        # Correction image: median_norm[tick] × strip_length[ch] → ADC units per (ch, tick)
        med_c = dump["medians_norm"].astype(np.float32)          # (ntick,)
        sl    = dump["strip_lengths"].astype(np.float32)         # (nch,)
        corr  = sl[:, np.newaxis] * med_c[np.newaxis, :]         # (nch, ntick)
        vlo_c, vhi_c = _sym_range(corr)
        mapper_corr.update(low=vlo_c, high=vhi_c)
        src_corr.data = _image_dict(corr, ticks, channels)

        # Output: Python (top half) vs C++ reference (bottom half)
        wf_out_cpp = dump.get("wf_out_raw", wf_in)
        py_out = state.get("py_wf_out")
        if py_out is not None:
            top_img = py_out
            top_label = "Python output"
        else:
            top_img = wf_out_cpp
            top_label = "C++ output"

        # Use reference from magnify if available, else C++ dump output
        mag = state["magnify"]
        if mag is not None:
            ref_img = mag["hu_raw"]
        else:
            ref_img = wf_out_cpp

        vlo2, vhi2 = _sym_range(np.concatenate([top_img.ravel(), ref_img.ravel()]))
        mapper_out.update(low=vlo2, high=vhi2)
        mapper_ref.update(low=vlo2, high=vhi2)

        # Pack top (Python) and bottom (ref) into the same figure by stacking.
        # Use y offset: Python output occupies upper half, reference lower half.
        half_nch = nch // 2
        out_top = top_img[:half_nch, :] if top_img.shape[0] > half_nch else top_img
        ref_bot = ref_img[half_nch:, :] if ref_img.shape[0] > half_nch else ref_img

        src_out.data = _image_dict(out_top, ticks, channels[:half_nch])
        src_ref.data = _image_dict(ref_bot, ticks, channels[half_nch:])

        f_out.title.text = (
            f"{top_label} (upper wires) / C++ hu_raw reference (lower wires)"
        )
        info_div.text = (
            f"<b>Anode {state['anode']}</b> &nbsp; "
            f"channels {channels[0]}–{channels[-1]} &nbsp; "
            f"wires={nch} ticks={ntick} &nbsp; "
            f"wf_in ADC range [{wf_in.min():.1f}, {wf_in.max():.1f}]"
        )

    def update_median():
        dump = state["dump"]
        if dump is None:
            for src in (src_med_c, src_med_p, src_med_band):
                src.data = dict(x=[], y=[], top=[], bottom=[]) if "top" in src.data else dict(x=[], y=[])
            return

        ntick = dump["wf_in_norm"].shape[1]
        ticks = np.arange(ntick, dtype=float)

        med_c = dump["medians_norm"].astype(float)
        src_med_c.data = dict(x=ticks, y=med_c)

        py_med = state.get("py_medians")
        if py_med is not None:
            src_med_p.data = dict(x=ticks, y=py_med)
        else:
            src_med_p.data = dict(x=[], y=[])

        # RMS band estimate from the C++ median absolute deviation
        rms_est = float(np.median(np.abs(med_c))) * 1.4826
        src_med_band.data = dict(x=ticks,
                                  top=med_c + rms_est,
                                  bottom=med_c - rms_est)

    def update_wire():
        dump = state["dump"]

        if dump is None:
            for src in (src_w_in, src_w_out, src_w_ref, src_w_po, src_w_med):
                src.data = dict(x=[], y=[])
            return

        channels = dump["channels"].astype(int)
        nch, ntick = dump["wf_in_raw"].shape
        # Convert channel number → index within this group
        wire_idx = int(wire_sp.value) - int(channels[0])
        wire_idx = max(0, min(wire_idx, nch - 1))

        ticks = np.arange(ntick, dtype=float)
        sl = float(dump["strip_lengths"][wire_idx])

        src_w_in.data  = dict(x=ticks, y=dump["wf_in_raw"][wire_idx].astype(float))
        src_w_out.data = dict(x=ticks, y=dump["wf_out_raw"][wire_idx].astype(float))

        mag = state["magnify"]
        if mag is not None:
            src_w_ref.data = dict(x=ticks, y=mag["hu_raw"][wire_idx].astype(float))
        else:
            src_w_ref.data = dict(x=[], y=[])

        py_out = state.get("py_wf_out")
        if py_out is not None:
            src_w_po.data = dict(x=ticks, y=py_out[wire_idx].astype(float))
        else:
            src_w_po.data = dict(x=[], y=[])

        # Show the per-tick median scaled by strip length for this wire
        py_med = state.get("py_medians")
        med = py_med if py_med is not None else dump["medians_norm"].astype(float)
        src_w_med.data = dict(x=ticks, y=med * sl)

        f_wire.title.text = (
            f"Wire cross-section: wire idx={wire_idx} "
            f"(ch={dump['channels'][wire_idx]}, strip_len={sl:.3f})"
        )

    def full_update():
        update_2d()
        update_median()
        update_wire()

    # ── Python recompute ───────────────────────────────────────────────────

    def run_python_algo():
        dump = state["dump"]
        if dump is None:
            return

        wf_in_norm  = dump["wf_in_norm"].astype(np.float64)
        wf_in_raw   = dump["wf_in_raw"].astype(np.float64)
        strip_lens  = dump["strip_lengths"].astype(np.float64)

        sf  = float(sig_factor_sp.value)
        pb  = int(pad_bins_sp.value)
        of  = float(outlier_sp.value)
        bip = bipolar_tog.active

        mask     = python_signal_mask(wf_in_norm, sig_factor=sf, pad_bins=pb, bipolar=bip)
        medians, max_rms, pool_mask = python_compute_median(wf_in_norm, mask, outlier_factor=of)
        py_out   = python_subtract(wf_in_raw, strip_lens, medians)

        state["py_mask"]      = mask
        state["py_medians"]   = medians
        state["py_pool_mask"] = pool_mask
        state["py_wf_out"]    = py_out

        info_div.text = (
            info_div.text +
            f" &nbsp;|&nbsp; Python recompute: "
            f"sig_factor={sf} pad_bins={pb} outlier_factor={of} bipolar={bip} "
            f"max_rms={max_rms:.3f}"
        )

        full_update()

    # ── callbacks ──────────────────────────────────────────────────────────

    def on_anode_change(_attr, _old, new):
        load_anode(int(new))
        first_ch = ANODE_FIRST_CH[int(new)]
        wire_sp.low   = first_ch
        wire_sp.high  = first_ch + GROUP_SIZE - 1
        wire_sp.value = first_ch
        full_update()

    def on_wire_change(_attr, _old, _new):
        update_wire()

    def on_apply():
        run_python_algo()

    anode_select.on_change("value", on_anode_change)
    wire_sp.on_change("value", on_wire_change)
    apply_btn.on_click(on_apply)

    # ── layout ─────────────────────────────────────────────────────────────
    controls = row(
        anode_select,
        sig_factor_sp, pad_bins_sp, outlier_sp,
        bipolar_tog, apply_btn,
        sizing_mode="stretch_width",
    )

    layout = column(
        controls,
        info_div,
        f_in,
        f_mask,
        f_med,
        f_corr,
        f_out,
        row(wire_sp, sizing_mode="fixed"),
        f_wire,
        sizing_mode="stretch_width",
    )

    full_update()
    curdoc().add_root(layout)
    curdoc().title = "Shield coupling NF viewer"


main(sys.argv)

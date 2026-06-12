"""Bokeh viewer for tuning Wire / HF (Wiener / Gaus_wide) / LF (ROI_*_lf)
software filters on top of the bare deconvolved waveform.

Input frame: ``h{u,v,w}_rawdecon<ident>`` -- the per-channel deconvolved
waveform AFTER FR/ER division but BEFORE any software filter (Wire_ind,
Wire_col, Wiener_*, Gaus_wide, ROI_*_lf) and BEFORE any ROI mask.
Produced by a special-mode run of NF+SP+sp-to-magnify with the -R flag
(see ``OmnibusSigProc::decon_2D_init`` for the C++ tap point).

If ``rawdecon`` is missing in a magnify file, the tool falls back to
the production ``wiener`` frame (post-everything) and warns -- mostly
useful for archival files that predate the rawdecon tap.

Production overlay: ``h{u,v,w}_wiener<ident>`` is the production
post-Wiener+LF+ROI output, shown always-on (dotted green) for
comparing "your tuned filter" vs. production.

Filter forms (matching ``util/src/Response.cxx:435-444``):
    HF / Wire: H(f) = exp(-0.5 * (f/sigma)**power),  H[0]=0 if flag=true
    LF:        L(f) = 1 - exp(-(f/tau)**2)

The Wire filter operates across the WIRE axis of the plane, not time.
The viewer applies it as a numpy 2D-FFT multiplication on the cached
plane-array.  HF and LF act on the time-frequency axis per channel.

Run:
    bokeh serve --port 5007 filter_tune_viewer.py --args <spec> ...

where each <spec> is ``label|path|ident|detector`` (use the
``serve_filter_tune_viewer.sh`` launcher for the bundled defaults).

Remote viewing:
    ssh -L 5007:localhost:5007 user@workstation
    # then open http://localhost:5007/filter_tune_viewer
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import uproot
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColumnDataSource,
    Div,
    RadioButtonGroup,
    Select,
    Spinner,
    TextInput,
)
from bokeh.plotting import figure


TICK_US = 0.5
TICK_S = TICK_US * 1e-6

# OmnibusSigProc wire-axis padding: m_pad_nwires == m_wire_shift == 10 for both
# PDVD and PDHD (response file has 21 wire paths; enlargement = paths-1 = 20).
WIRE_PAD = 10


# ---- presets ---------------------------------------------------------------
# All σ/τ values taken verbatim from:
#   cfg/pgrapher/experiment/pdhd/sp-filters.jsonnet
#   cfg/pgrapher/experiment/protodunevd/sp-filters.jsonnet
# and cross-checked against pdvd/sp_plot/wiener_filter_construct.py:99-117.
#
# "(none)"  → no filter applied (bypass).
# "Custom"  → use the spinner widget values directly.
# any other → load preset into spinners and apply.

# PDHD: includes both the default Wiener_tight set (APA0/2/3, bare
# jsonnet names) and the APA1 override (Wiener_tight_*_APA1).
HF_PRESETS_PDHD = {
    "(none)":               None,
    "Custom":               "custom",
    "Gaus_wide":            dict(sigma=0.12,      power=2.0,     flag=True),
    "Wiener_tight U":       dict(sigma=0.221933,  power=6.55413, flag=True),
    "Wiener_tight V":       dict(sigma=0.222723,  power=8.75998, flag=True),
    "Wiener_tight W":       dict(sigma=0.225567,  power=3.47846, flag=True),
    "Wiener_tight U (APA1)": dict(sigma=0.203451, power=5.78093, flag=True),
    "Wiener_tight V (APA1)": dict(sigma=0.160191, power=3.54835, flag=True),
    "Wiener_tight W (APA1)": dict(sigma=0.125448, power=5.27080, flag=True),
    "Wiener_wide U":        dict(sigma=0.186765,  power=5.05429, flag=True),
    "Wiener_wide V":        dict(sigma=0.1936,    power=5.77422, flag=True),
    "Wiener_wide W":        dict(sigma=0.175722,  power=4.37928, flag=True),
}

# PDVD bottom CRP (anodes 0-3).  The _b and _t instances are identical
# today but kept separate to mirror the jsonnet convention.
HF_PRESETS_PDVD_BOTTOM = {
    "(none)":               None,
    "Custom":               "custom",
    "Gaus_wide_b":          dict(sigma=0.12,       power=2.0,     flag=True),
    "Wiener_tight U_b":     dict(sigma=0.148788,   power=3.76194, flag=True),
    "Wiener_tight V_b":     dict(sigma=0.1596568,  power=4.36125, flag=True),
    "Wiener_tight W_b":     dict(sigma=0.13623,    power=3.35324, flag=True),
    "Wiener_wide U_b":      dict(sigma=0.186765,   power=5.05429, flag=True),
    "Wiener_wide V_b":      dict(sigma=0.1936,     power=5.77422, flag=True),
    "Wiener_wide W_b":      dict(sigma=0.175722,   power=4.37928, flag=True),
}

# PDVD top CRP (anodes 4-7).  Currently byte-identical to _b.
HF_PRESETS_PDVD_TOP = {
    "(none)":               None,
    "Custom":               "custom",
    "Gaus_wide_t":          dict(sigma=0.12,       power=2.0,     flag=True),
    "Wiener_tight U_t":     dict(sigma=0.148788,   power=3.76194, flag=True),
    "Wiener_tight V_t":     dict(sigma=0.1596568,  power=4.36125, flag=True),
    "Wiener_tight W_t":     dict(sigma=0.13623,    power=3.35324, flag=True),
    "Wiener_wide U_t":      dict(sigma=0.186765,   power=5.05429, flag=True),
    "Wiener_wide V_t":      dict(sigma=0.1936,     power=5.77422, flag=True),
    "Wiener_wide W_t":      dict(sigma=0.175722,   power=4.37928, flag=True),
}

# LF τ values from sp-filters.jsonnet (PDHD and PDVD share the same
# loose τ; tight and tighter differ).
LF_PRESETS_PDHD = {
    "(none)":         None,
    "Custom":         "custom",
    "ROI_loose_lf":   dict(tau=0.002),
    "ROI_tight_lf":   dict(tau=0.016),
    "ROI_tighter_lf": dict(tau=0.08),
}

LF_PRESETS_PDVD = {
    "(none)":         None,
    "Custom":         "custom",
    "ROI_loose_lf":   dict(tau=0.002),
    "ROI_tight_lf":   dict(tau=0.014),
    "ROI_tighter_lf": dict(tau=0.06),
}

# Wire filter presets (from cfg/.../sp-filters.jsonnet).  σ_code is in
# cycles-per-wire units; larger σ_code → flatter in wire-freq → narrower
# spatial smearing.  See pdvd/sp_plot/compare_sp_filters.py.
WIRE_PRESETS_PDHD = {
    "(none)":      None,
    "Custom":      "custom",
    "Wire_ind":    dict(sigma=0.75 / np.sqrt(np.pi), power=2.0),    # ≈0.4231
    "Wire_col":    dict(sigma=10.0 / np.sqrt(np.pi), power=2.0),    # ≈5.642
}

WIRE_PRESETS_PDVD = {
    "(none)":      None,
    "Custom":      "custom",
    "Wire_ind":    dict(sigma=5.0 / np.sqrt(np.pi),  power=2.0),    # ≈2.821
    "Wire_col":    dict(sigma=10.0 / np.sqrt(np.pi), power=2.0),    # ≈5.642
}


def hf_kernel(f_mhz: np.ndarray, sigma: float, power: float, flag: bool) -> np.ndarray:
    # Guard tiny sigma so power(0) doesn't NaN.
    s = max(sigma, 1e-9)
    H = np.exp(-0.5 * (f_mhz / s) ** power)
    if flag:
        H[0] = 0.0
    return H


def lf_kernel(f_mhz: np.ndarray, tau: float) -> np.ndarray:
    t = max(tau, 1e-12)
    return 1.0 - np.exp(-((f_mhz / t) ** 2))


def wire_kernel(n_chan: int, sigma: float, power: float) -> np.ndarray:
    """Wire filter on numpy fft order (k = 0, 1, ..., n_chan-1).

    The HfFilter parametric form lives in normalised wire-frequency units
    (cycles/wire), max_freq = 0.5 (Nyquist).  Bins above N/2 are folded as
    negative frequencies, then |f|; the filter is even-symmetric so the
    inverse-FFT result is real.
    """
    s = max(sigma, 1e-9)
    k = np.arange(n_chan, dtype=np.float64)
    f = k / n_chan                       # 0 .. 1 cycles/wire (raw fft order)
    f = np.where(f > 0.5, f - 1.0, f)    # fold to [-0.5, 0.5]
    f = np.abs(f)
    return np.exp(-0.5 * (f / s) ** power)


# ---- file specs ------------------------------------------------------------

@dataclass
class FileSpec:
    label: str
    path: str
    ident: int
    detector: str   # 'pdhd' or 'pdvd'


def parse_specs(argv: list[str]) -> list[FileSpec]:
    """Parse positional ``label|path|ident|detector`` specs from argv."""
    specs: list[FileSpec] = []
    for raw in argv:
        parts = raw.split("|")
        if len(parts) != 4:
            print(f"Bad --args spec (need 'label|path|ident|detector'): {raw!r}",
                  file=sys.stderr)
            sys.exit(1)
        label, path, ident, det = (p.strip() for p in parts)
        if det not in ("pdhd", "pdvd"):
            print(f"Bad detector tag {det!r} in spec: {raw!r}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(path):
            print(f"Magnify file not found: {path}", file=sys.stderr)
            sys.exit(1)
        specs.append(FileSpec(label=label, path=path, ident=int(ident), detector=det))
    if not specs:
        print("Need at least one --args spec.", file=sys.stderr)
        sys.exit(1)
    return specs


# ---- magnify cache ---------------------------------------------------------

class PlaneData:
    """Cached data for one (file, plane).

    rawdecon: real (n_chan, n_tick) — bare deconvolved waveform (post FR/ER,
              pre Wire/HF/LF/ROI).  Fallback to wiener TH2 if rawdecon missing.
    wiener:   real (n_chan, n_tick) or None — production post-Wiener post-ROI output.
    F:        rfft along time of rawdecon, complex (n_chan, n_freq_t).
    ch_offset: global channel id of local row 0.
    used_fallback: True iff rawdecon was missing and we fell back to wiener.
    """
    def __init__(self, rawdecon, wiener, ch_offset, used_fallback):
        self.rawdecon = rawdecon
        self.wiener = wiener
        self.ch_offset = ch_offset
        self.used_fallback = used_fallback
        # Cache the time-axis rfft lazily on first render.
        self._F = None

    def F(self) -> np.ndarray:
        if self._F is None:
            self._F = np.fft.rfft(self.rawdecon.astype(np.float64), axis=1)
        return self._F


class MagnifyCache:
    """Lazy per-file loader for ``h{u,v,w}_rawdecon<ident>`` (preferred)
    and ``h{u,v,w}_wiener<ident>`` (production overlay)."""
    RD_PFX = {0: "hu_rawdecon", 1: "hv_rawdecon", 2: "hw_rawdecon"}
    WN_PFX = {0: "hu_wiener",   1: "hv_wiener",   2: "hw_wiener"}

    def __init__(self, specs: list[FileSpec]):
        self._specs = {s.label: s for s in specs}
        self._frames: dict[tuple[str, int], PlaneData] = {}

    def labels(self) -> list[str]:
        return list(self._specs.keys())

    def detector(self, label: str) -> str:
        return self._specs[label].detector

    def get_plane(self, label: str, plane: int) -> PlaneData:
        key = (label, plane)
        cached = self._frames.get(key)
        if cached is not None:
            return cached
        spec = self._specs[label]
        rd_name = f"{self.RD_PFX[plane]}{spec.ident}"
        wn_name = f"{self.WN_PFX[plane]}{spec.ident}"
        with uproot.open(spec.path) as f:
            file_keys = {k.split(';')[0] for k in f.keys()}
            used_fallback = False
            if rd_name in file_keys:
                src_h = f[rd_name]
            elif wn_name in file_keys:
                print(f"[warn] {spec.path}: missing {rd_name}, falling back to "
                      f"{wn_name} (post-filter post-ROI; rerun NF+SP with -R "
                      f"for the bare decon).", file=sys.stderr)
                src_h = f[wn_name]
                used_fallback = True
            else:
                print(f"[warn] {spec.path}: neither {rd_name} nor {wn_name} present.",
                      file=sys.stderr)
                self._frames[key] = PlaneData(
                    rawdecon=np.zeros((1, 6000), dtype=np.float32),
                    wiener=None, ch_offset=0, used_fallback=False)
                return self._frames[key]
            rawdecon = np.asarray(src_h.values(), dtype=np.float32)
            ch_offset = int(round(src_h.to_numpy()[1][0]))
            # Production wiener overlay (separate from the rawdecon fallback path).
            wiener = (np.asarray(f[wn_name].values(), dtype=np.float32)
                      if wn_name in file_keys and not used_fallback else None)
        self._frames[key] = PlaneData(rawdecon=rawdecon, wiener=wiener,
                                      ch_offset=ch_offset,
                                      used_fallback=used_fallback)
        return self._frames[key]


def best_channel(plane_data: "PlaneData") -> int:
    """Local row index of the channel with largest peak-to-peak."""
    data = plane_data.rawdecon
    if data.size == 0:
        return 0
    pp = data.max(axis=1) - data.min(axis=1)
    return int(np.argmax(pp))


# ---- main ------------------------------------------------------------------

def main(argv):
    specs = parse_specs(argv[1:])
    cache = MagnifyCache(specs)

    # ---- state ------------------------------------------------------------
    state = dict(
        wave_t=None,    # current channel rawdecon waveform (n_tick,)
        wiener_t=None,  # current channel production wiener waveform or None
        X=None,         # rfft of wave_t (n_tick//2 + 1,)
        f_mhz=None,     # rfft frequency grid in MHz
        n_tick=0,
        n_chan=0,
        local_idx=0,    # row index within the loaded plane array
        ch_global=0,    # ADC-channel id (offset + local index)
    )

    # ---- widgets ----------------------------------------------------------
    file_select = Select(title="File:", value=cache.labels()[0], options=cache.labels(),
                         width=220)
    plane_radio = RadioButtonGroup(labels=["U", "V", "W"], active=0)
    chan_input  = TextInput(title="Channel (global):", value="0", width=140)
    prev_btn    = Button(label="◀ Prev", width=80)
    next_btn    = Button(label="Next ▶", width=80)
    best_btn    = Button(label="Largest p-p", width=110)
    info_div    = Div(text="", width=900)
    status_div  = Div(text="<span style='color:#888'>idle</span>", width=400)
    update_btn  = Button(label="Update plots", button_type="primary", width=110)

    # HF column
    hf_preset  = Select(title="HF preset:", value="(none)",
                        options=list(HF_PRESETS_PDHD.keys()), width=220)
    hf_sigma   = Spinner(title="HF σ (MHz)",  low=0.001, high=2.0,  step=0.001,
                         value=0.20,  width=160)
    hf_power   = Spinner(title="HF power",    low=0.5,   high=30.0, step=0.1,
                         value=4.0,   width=160)
    hf_zero_dc = CheckboxGroup(labels=["Zero DC (flag=true)"], active=[0])

    # LF column
    lf_preset  = Select(title="LF preset:", value="(none)",
                        options=list(LF_PRESETS_PDHD.keys()), width=220)
    lf_tau     = Spinner(title="LF τ (MHz)", low=1e-5, high=1.0, step=0.001,
                         value=0.014, width=160)

    # Wire column (geometric wire-axis filter; production has Wire_ind for U/V,
    # Wire_col for W; user can override).
    wire_preset = Select(title="Wire preset:", value="(none)",
                         options=list(WIRE_PRESETS_PDHD.keys()), width=220)
    wire_sigma  = Spinner(title="Wire σ (cycles/wire)", low=0.001, high=20.0,
                          step=0.01, value=2.82, width=160)
    wire_power  = Spinner(title="Wire power", low=0.5, high=10.0, step=0.1,
                          value=2.0, width=160)

    # ---- Filter B widgets (orange; same structure as A) --------------------
    hf_preset_b  = Select(title="HF preset:", value="(none)",
                          options=list(HF_PRESETS_PDHD.keys()), width=220)
    hf_sigma_b   = Spinner(title="HF σ (MHz)",  low=0.001, high=2.0,  step=0.001,
                           value=0.20,  width=160)
    hf_power_b   = Spinner(title="HF power",    low=0.5,   high=30.0, step=0.1,
                           value=4.0,   width=160)
    hf_zero_dc_b = CheckboxGroup(labels=["Zero DC (flag=true)"], active=[0])

    lf_preset_b  = Select(title="LF preset:", value="(none)",
                          options=list(LF_PRESETS_PDHD.keys()), width=220)
    lf_tau_b     = Spinner(title="LF τ (MHz)", low=1e-5, high=1.0, step=0.001,
                           value=0.014, width=160)

    wire_preset_b = Select(title="Wire preset:", value="(none)",
                           options=list(WIRE_PRESETS_PDHD.keys()), width=220)
    wire_sigma_b  = Spinner(title="Wire σ (cycles/wire)", low=0.001, high=20.0,
                            step=0.01, value=2.82, width=160)
    wire_power_b  = Spinner(title="Wire power", low=0.5, high=10.0, step=0.1,
                            value=2.0, width=160)

    # Time-range
    tmin_input = TextInput(title="t-min (tick)", value="0", width=120)
    tmax_input = TextInput(title="t-max (tick)", value="6000", width=120)
    range_btn  = Button(label="Apply range", width=110)
    reset_btn  = Button(label="Reset range", width=110)

    # ---- figures ----------------------------------------------------------
    fig_kw_t = dict(height=300, sizing_mode="stretch_width",
                    active_scroll="wheel_zoom",
                    tools="pan,wheel_zoom,box_zoom,reset,save")

    # Color scheme: A = blue family, B = orange family, wiener = green
    CA_wave, CA_hf, CA_lf, CA_prod, CA_wire = "#1f77b4", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"
    CB_wave, CB_hf, CB_lf, CB_prod, CB_wire = "#ff7f0e", "#ff7f0e", "#8c564b", "#bcbd22", "#e377c2"

    f_time = figure(
        title="Waveform — Filter A (blue solid), Filter B (orange dashed), production wiener (green dotted)",
        **fig_kw_t)
    f_time.xaxis.axis_label = "tick (0.5 µs each)"
    f_time.yaxis.axis_label = "amplitude"

    f_filter = figure(
        title="Filter shapes — A: solid lines  |  B: dashed lines  |  HF, LF, HF×LF, Wire",
        height=260, sizing_mode="stretch_width",
        active_scroll="wheel_zoom",
        tools="pan,wheel_zoom,box_zoom,reset,save")
    f_filter.xaxis.axis_label = "frequency (MHz, time axis); Wire shown on rescaled common x"
    f_filter.yaxis.axis_label = "filter value (0 .. 1)"
    f_filter.x_range.start = 0.0
    f_filter.x_range.end = 1.0
    f_filter.y_range.start = -0.05
    f_filter.y_range.end = 1.05

    f_spec = figure(
        title="Channel spectrum |X(f)| — Filter A (blue), Filter B (orange dashed), production wiener (green dotted)",
        height=220, sizing_mode="stretch_width",
        active_scroll="wheel_zoom",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        y_axis_type="linear")
    f_spec.xaxis.axis_label = "frequency (MHz)"
    f_spec.yaxis.axis_label = "|X(f)|"
    f_spec.x_range.start = 0.0
    f_spec.x_range.end = 1.0

    # Filter A data sources
    src_filt    = ColumnDataSource(data=dict(x=[], y=[]))
    src_wiener  = ColumnDataSource(data=dict(x=[], y=[]))
    src_hf      = ColumnDataSource(data=dict(x=[], y=[]))
    src_lf      = ColumnDataSource(data=dict(x=[], y=[]))
    src_wire    = ColumnDataSource(data=dict(x=[], y=[]))
    src_prod    = ColumnDataSource(data=dict(x=[], y=[]))
    src_spec    = ColumnDataSource(data=dict(x=[], y=[]))
    src_spec_wn = ColumnDataSource(data=dict(x=[], y=[]))
    # Filter B data sources
    src_filt_b  = ColumnDataSource(data=dict(x=[], y=[]))
    src_hf_b    = ColumnDataSource(data=dict(x=[], y=[]))
    src_lf_b    = ColumnDataSource(data=dict(x=[], y=[]))
    src_wire_b  = ColumnDataSource(data=dict(x=[], y=[]))
    src_prod_b  = ColumnDataSource(data=dict(x=[], y=[]))
    src_spec_b  = ColumnDataSource(data=dict(x=[], y=[]))

    # Time-domain panel
    f_time.line("x", "y", source=src_filt,   line_width=1.5, color=CA_wave,
                legend_label="Filter A")
    f_time.line("x", "y", source=src_filt_b, line_width=1.5, color=CB_wave,
                line_dash="dashed", legend_label="Filter B")
    f_time.line("x", "y", source=src_wiener, line_width=1.5, color="#2ca02c",
                line_dash="dotted", legend_label="production wiener (post-ROI)")
    f_time.legend.location = "top_left"
    f_time.legend.click_policy = "hide"

    # Filter-shapes panel — A solid, B dashed
    f_filter.line("x", "y", source=src_hf,    line_width=1.5, color=CA_hf,
                  legend_label="A: HF")
    f_filter.line("x", "y", source=src_lf,    line_width=1.5, color=CA_lf,
                  legend_label="A: LF")
    f_filter.line("x", "y", source=src_prod,  line_width=2.0, color=CA_prod,
                  line_dash="dashed", legend_label="A: HF×LF")
    f_filter.line("x", "y", source=src_wire,  line_width=1.5, color=CA_wire,
                  legend_label="A: Wire")
    f_filter.line("x", "y", source=src_hf_b,  line_width=1.5, color=CB_hf,
                  line_dash="dashed", legend_label="B: HF")
    f_filter.line("x", "y", source=src_lf_b,  line_width=1.5, color=CB_lf,
                  line_dash="dashed", legend_label="B: LF")
    f_filter.line("x", "y", source=src_prod_b, line_width=2.0, color=CB_prod,
                  line_dash="dotted", legend_label="B: HF×LF")
    f_filter.line("x", "y", source=src_wire_b, line_width=1.5, color=CB_wire,
                  line_dash="dashed", legend_label="B: Wire")
    f_filter.legend.location = "center_right"
    f_filter.legend.click_policy = "hide"

    # Spectrum panel
    f_spec.line("x", "y", source=src_spec,    line_width=1,   color=CA_wave,
                legend_label="Filter A")
    f_spec.line("x", "y", source=src_spec_b,  line_width=1,   color=CB_wave,
                line_dash="dashed", legend_label="Filter B")
    f_spec.line("x", "y", source=src_spec_wn, line_width=1.5, color="#2ca02c",
                line_dash="dotted", legend_label="production wiener")
    f_spec.legend.location = "top_right"
    f_spec.legend.click_policy = "hide"

    # ---- helpers ----------------------------------------------------------
    def detector_group() -> str:
        """Return 'pdhd', 'pdvd_bottom', or 'pdvd_top' for the active file."""
        spec_label = file_select.value
        det = cache.detector(spec_label)
        if det == "pdhd":
            return "pdhd"
        ident = cache._specs[spec_label].ident
        return "pdvd_bottom" if ident < 4 else "pdvd_top"

    def hf_options() -> dict:
        g = detector_group()
        if g == "pdhd":
            return HF_PRESETS_PDHD
        if g == "pdvd_top":
            return HF_PRESETS_PDVD_TOP
        return HF_PRESETS_PDVD_BOTTOM

    def lf_options() -> dict:
        return LF_PRESETS_PDHD if detector_group() == "pdhd" else LF_PRESETS_PDVD

    def wire_options() -> dict:
        return WIRE_PRESETS_PDHD if detector_group() == "pdhd" else WIRE_PRESETS_PDVD

    def refresh_preset_lists():
        opts_hf   = list(hf_options().keys())
        opts_lf   = list(lf_options().keys())
        opts_wire = list(wire_options().keys())
        for w in (hf_preset, hf_preset_b):
            w.options = opts_hf
            if w.value not in opts_hf:
                w.value = "(none)"
        for w in (lf_preset, lf_preset_b):
            w.options = opts_lf
            if w.value not in opts_lf:
                w.value = "(none)"
        for w in (wire_preset, wire_preset_b):
            w.options = opts_wire
            if w.value not in opts_wire:
                w.value = "(none)"

    # Filter A kw helpers
    def hf_kw_from_widgets():
        if hf_preset.value == "(none)":
            return None
        return dict(sigma=float(hf_sigma.value),
                    power=float(hf_power.value),
                    flag=(0 in hf_zero_dc.active))

    def lf_kw_from_widgets():
        if lf_preset.value == "(none)":
            return None
        return dict(tau=float(lf_tau.value))

    def wire_kw_from_widgets():
        if wire_preset.value == "(none)":
            return None
        return dict(sigma=float(wire_sigma.value),
                    power=float(wire_power.value))

    # Filter B kw helpers (mirror of A)
    def hf_kw_from_widgets_b():
        if hf_preset_b.value == "(none)":
            return None
        return dict(sigma=float(hf_sigma_b.value),
                    power=float(hf_power_b.value),
                    flag=(0 in hf_zero_dc_b.active))

    def lf_kw_from_widgets_b():
        if lf_preset_b.value == "(none)":
            return None
        return dict(tau=float(lf_tau_b.value))

    def wire_kw_from_widgets_b():
        if wire_preset_b.value == "(none)":
            return None
        return dict(sigma=float(wire_sigma_b.value),
                    power=float(wire_power_b.value))

    def load_channel():
        """Read current (file, plane, channel) into state dict."""
        plane = plane_radio.active
        pdata = cache.get_plane(file_select.value, plane)
        n_chan, n_tick = pdata.rawdecon.shape
        ch_offset = pdata.ch_offset
        try:
            ch_global = int(chan_input.value)
        except ValueError:
            ch_global = ch_offset
        local = max(0, min(n_chan - 1, ch_global - ch_offset))
        ch_global = ch_offset + local
        chan_input.value = str(ch_global)

        wave = pdata.rawdecon[local].astype(np.float64)
        wiener_wave = (pdata.wiener[local].astype(np.float64)
                       if pdata.wiener is not None else None)
        f_mhz = np.fft.rfftfreq(n_tick, TICK_S) * 1e-6
        X = np.fft.rfft(wave)

        state["wave_t"] = wave
        state["wiener_t"] = wiener_wave
        state["X"] = X
        state["f_mhz"] = f_mhz
        state["n_tick"] = n_tick
        state["n_chan"] = n_chan
        state["local_idx"] = local
        state["ch_global"] = ch_global

    def _compute_filtered_waveform(plane_data: PlaneData,
                                   hf_kw, lf_kw, wire_kw,
                                   f_mhz, n_chan, n_tick, local_idx):
        """Apply Wire (axis 0) + HF (axis 1) + LF (axis 1) filters to the
        plane's rfft array and return the irfft'd waveform of channel
        local_idx as a 1-D numpy array."""
        H = (hf_kernel(f_mhz, **hf_kw) if hf_kw is not None
             else np.ones_like(f_mhz))
        L = (lf_kernel(f_mhz, **lf_kw) if lf_kw is not None
             else np.ones_like(f_mhz))
        HL = H * L

        if wire_kw is None:
            # Cheap path: per-channel rfft is already cached as plane_data.F[c, :].
            X = plane_data.F()[local_idx, :]
            Y = np.fft.irfft(X * HL, n=n_tick)
        else:
            # Wire filter at production-matching padded size (m_fft_nwires).
            # rawdecon arrives already wire-shifted on the padded grid then stripped
            # to m_nwires rows.  Reproduce production order: re-pad, undo shift,
            # FFT axis=0 at m_fft_nwires, multiply, IFFT, re-shift, strip.
            n_fft = n_chan + 2 * WIRE_PAD
            F_src = plane_data.F()               # (n_chan, n_freq)
            n_freq = F_src.shape[1]
            F_pad = np.zeros((n_fft, n_freq), dtype=np.complex128)
            F_pad[WIRE_PAD:WIRE_PAD + n_chan, :] = F_src
            F_pad = np.roll(F_pad, -WIRE_PAD, axis=0)  # undo production wire-shift
            W = wire_kernel(n_fft, wire_kw["sigma"], wire_kw["power"])
            G_filt = np.fft.fft(F_pad, axis=0) * W[:, None] * HL[None, :]
            F2 = np.fft.ifft(G_filt, axis=0)
            F2 = np.roll(F2, +WIRE_PAD, axis=0)[WIRE_PAD:WIRE_PAD + n_chan, :]
            Y = np.fft.irfft(F2[local_idx, :], n=n_tick)
        return H, L, HL, Y

    def _do_render():
        """Heavy work: filter math + push new ColumnDataSource data."""
        wave_t = state["wave_t"]
        f_mhz = state["f_mhz"]
        n_tick = state["n_tick"]
        n_chan = state["n_chan"]
        local_idx = state["local_idx"]
        if wave_t is None:
            status_div.text = "<span style='color:#888'>idle</span>"
            return
        x_ticks = np.arange(n_tick)

        t0 = time.perf_counter()
        plane = plane_radio.active
        pdata = cache.get_plane(file_select.value, plane)

        # Filter A
        hf_kw   = hf_kw_from_widgets()
        lf_kw   = lf_kw_from_widgets()
        wire_kw = wire_kw_from_widgets()
        H, L, HL, Y = _compute_filtered_waveform(
            pdata, hf_kw, lf_kw, wire_kw, f_mhz, n_chan, n_tick, local_idx)

        # Filter B
        hf_kw_b   = hf_kw_from_widgets_b()
        lf_kw_b   = lf_kw_from_widgets_b()
        wire_kw_b = wire_kw_from_widgets_b()
        Hb, Lb, HLb, Yb = _compute_filtered_waveform(
            pdata, hf_kw_b, lf_kw_b, wire_kw_b, f_mhz, n_chan, n_tick, local_idx)

        # Wire kernel for the freq-domain panel — show on a 0..1 rescaled x
        # so the line shares the same x-range as HF/LF (cosmetic only).
        def _wire_display(wkw):
            if wkw is not None:
                W = wire_kernel(n_chan, wkw["sigma"], wkw["power"])
                half = n_chan // 2 + 1
                return np.linspace(0.0, 1.0, half), W[:half]
            return np.array([0.0, 1.0]), np.array([1.0, 1.0])

        wire_x,  wire_y  = _wire_display(wire_kw)
        wire_xb, wire_yb = _wire_display(wire_kw_b)

        dt = (time.perf_counter() - t0) * 1000.0

        # Trim all time-domain data to the visible range so the y-axis
        # auto-scales to the signal inside the window, not the edge artifacts.
        try:
            t_lo = max(0, int(tmin_input.value))
            t_hi = min(n_tick, int(tmax_input.value))
        except ValueError:
            t_lo, t_hi = 0, n_tick
        if t_hi <= t_lo:
            t_lo, t_hi = 0, n_tick

        # Filter A time-domain
        src_filt.data   = dict(x=x_ticks[t_lo:t_hi], y=Y[t_lo:t_hi])
        # Filter B time-domain
        src_filt_b.data = dict(x=x_ticks[t_lo:t_hi], y=Yb[t_lo:t_hi])

        # Production wiener overlay — always shown when data is available.
        wiener_t = state["wiener_t"]
        if wiener_t is not None:
            src_wiener.data = dict(x=x_ticks[t_lo:t_hi], y=wiener_t[t_lo:t_hi])
        else:
            src_wiener.data = dict(x=[], y=[])

        # Spectrum panel
        src_spec.data   = dict(x=f_mhz, y=np.abs(np.fft.rfft(Y)))
        src_spec_b.data = dict(x=f_mhz, y=np.abs(np.fft.rfft(Yb)))
        if wiener_t is not None:
            src_spec_wn.data = dict(x=f_mhz, y=np.abs(np.fft.rfft(wiener_t)))
        else:
            src_spec_wn.data = dict(x=[], y=[])

        # Filter-shape panel — A
        src_hf.data    = dict(x=f_mhz, y=H)
        src_lf.data    = dict(x=f_mhz, y=L)
        src_prod.data  = dict(x=f_mhz, y=HL)
        src_wire.data  = dict(x=wire_x,  y=wire_y)
        # Filter-shape panel — B
        src_hf_b.data   = dict(x=f_mhz, y=Hb)
        src_lf_b.data   = dict(x=f_mhz, y=Lb)
        src_prod_b.data = dict(x=f_mhz, y=HLb)
        src_wire_b.data = dict(x=wire_xb, y=wire_yb)

        plane_name = ["U", "V", "W"][plane_radio.active]
        fb = " &nbsp; <b style='color:#d62728'>(rawdecon missing → using post-filter wiener)</b>" if pdata.used_fallback else ""
        info_div.text = (
            f"<b>{file_select.value}</b> &nbsp; plane=<b>{plane_name}</b> "
            f"&nbsp; channel=<b>{state['ch_global']}</b> &nbsp; "
            f"n_tick={n_tick} (window={n_tick * TICK_US:.1f} µs){fb}"
        )
        ts = time.strftime("%H:%M:%S")
        status_div.text = (
            f"<span style='color:#2ca02c'><b>done</b></span> &nbsp; "
            f"compute = {dt:.1f} ms &nbsp; @ {ts}"
        )

    def render():
        """Schedule a re-render: flash 'computing...' then run on next tick."""
        status_div.text = (
            "<span style='color:#d62728'><b>computing...</b></span>"
        )
        curdoc().add_next_tick_callback(_do_render)

    def reload_and_render():
        load_channel()
        render()

    def _wire_default_for_plane(plane: int) -> str:
        """U/V default to Wire_ind, W to Wire_col (matches production)."""
        return "Wire_col" if plane == 2 else "Wire_ind"

    # ---- callbacks --------------------------------------------------------
    def on_file_change(_a, _o, _n):
        refresh_preset_lists()
        # Reset channel to a useful one for the new file/plane.
        pdata = cache.get_plane(file_select.value, plane_radio.active)
        chan_input.value = str(pdata.ch_offset + best_channel(pdata))
        reload_and_render()

    def on_plane_change(_a, _o, _n):
        # Channel range varies by plane — pick the largest p-p in the new plane.
        pdata = cache.get_plane(file_select.value, plane_radio.active)
        chan_input.value = str(pdata.ch_offset + best_channel(pdata))
        # Auto-switch wire-preset default to match the plane (U/V→ind, W→col)
        # only when the user hasn't already picked '(none)' or 'Custom'.
        new_wire = _wire_default_for_plane(plane_radio.active)
        if wire_preset.value not in ("(none)", "Custom") and new_wire in wire_preset.options:
            wire_preset.value = new_wire
        reload_and_render()

    def on_chan_change(_a, _o, _n):
        reload_and_render()

    def step_chan(delta: int):
        try:
            ch = int(chan_input.value)
        except ValueError:
            ch = 0
        chan_input.value = str(ch + delta)

    def on_best_click():
        pdata = cache.get_plane(file_select.value, plane_radio.active)
        chan_input.value = str(pdata.ch_offset + best_channel(pdata))

    def on_hf_preset(_a, _o, new):
        kw = hf_options().get(new)
        if isinstance(kw, dict):
            # Push preset values into spinners without infinite-looping.
            hf_sigma.value = kw["sigma"]
            hf_power.value = kw["power"]
            hf_zero_dc.active = [0] if kw["flag"] else []
        render()

    def on_lf_preset(_a, _o, new):
        kw = lf_options().get(new)
        if isinstance(kw, dict):
            lf_tau.value = kw["tau"]
        render()

    def on_hf_spinner(_a, _o, _n):
        # If value diverges from the named preset, switch to "Custom".
        kw = hf_options().get(hf_preset.value)
        if isinstance(kw, dict):
            same = (abs(kw["sigma"] - hf_sigma.value) < 1e-9
                    and abs(kw["power"] - hf_power.value) < 1e-9
                    and (kw["flag"] == (0 in hf_zero_dc.active)))
            if not same:
                hf_preset.value = "Custom"
        render()

    def on_lf_spinner(_a, _o, _n):
        kw = lf_options().get(lf_preset.value)
        if isinstance(kw, dict) and abs(kw["tau"] - lf_tau.value) > 1e-12:
            lf_preset.value = "Custom"
        render()

    def on_wire_preset(_a, _o, new):
        kw = wire_options().get(new)
        if isinstance(kw, dict):
            wire_sigma.value = kw["sigma"]
            wire_power.value = kw["power"]
        render()

    def on_wire_spinner(_a, _o, _n):
        kw = wire_options().get(wire_preset.value)
        if isinstance(kw, dict):
            same = (abs(kw["sigma"] - wire_sigma.value) < 1e-9
                    and abs(kw["power"] - wire_power.value) < 1e-9)
            if not same:
                wire_preset.value = "Custom"
        render()

    # ---- Filter B callbacks (mirror of A) ---------------------------------
    def on_hf_preset_b(_a, _o, new):
        kw = hf_options().get(new)
        if isinstance(kw, dict):
            hf_sigma_b.value = kw["sigma"]
            hf_power_b.value = kw["power"]
            hf_zero_dc_b.active = [0] if kw["flag"] else []
        render()

    def on_lf_preset_b(_a, _o, new):
        kw = lf_options().get(new)
        if isinstance(kw, dict):
            lf_tau_b.value = kw["tau"]
        render()

    def on_hf_spinner_b(_a, _o, _n):
        kw = hf_options().get(hf_preset_b.value)
        if isinstance(kw, dict):
            same = (abs(kw["sigma"] - hf_sigma_b.value) < 1e-9
                    and abs(kw["power"] - hf_power_b.value) < 1e-9
                    and (kw["flag"] == (0 in hf_zero_dc_b.active)))
            if not same:
                hf_preset_b.value = "Custom"
        render()

    def on_lf_spinner_b(_a, _o, _n):
        kw = lf_options().get(lf_preset_b.value)
        if isinstance(kw, dict) and abs(kw["tau"] - lf_tau_b.value) > 1e-12:
            lf_preset_b.value = "Custom"
        render()

    def on_wire_preset_b(_a, _o, new):
        kw = wire_options().get(new)
        if isinstance(kw, dict):
            wire_sigma_b.value = kw["sigma"]
            wire_power_b.value = kw["power"]
        render()

    def on_wire_spinner_b(_a, _o, _n):
        kw = wire_options().get(wire_preset_b.value)
        if isinstance(kw, dict):
            same = (abs(kw["sigma"] - wire_sigma_b.value) < 1e-9
                    and abs(kw["power"] - wire_power_b.value) < 1e-9)
            if not same:
                wire_preset_b.value = "Custom"
        render()

    def on_apply_range():
        try:
            lo = int(tmin_input.value)
            hi = int(tmax_input.value)
        except ValueError:
            return
        if hi <= lo:
            return
        f_time.x_range.start = lo
        f_time.x_range.end = hi
        render()

    def on_reset_range():
        n = state["n_tick"] or 6000
        tmin_input.value = "200"
        tmax_input.value = str(n - 200)
        f_time.x_range.start = 200
        f_time.x_range.end = n - 200
        render()

    file_select.on_change("value", on_file_change)
    plane_radio.on_change("active", on_plane_change)
    chan_input.on_change("value", on_chan_change)
    prev_btn.on_click(lambda: step_chan(-1))
    next_btn.on_click(lambda: step_chan(+1))
    best_btn.on_click(on_best_click)

    # Filter A bindings
    hf_preset.on_change("value", on_hf_preset)
    lf_preset.on_change("value", on_lf_preset)
    wire_preset.on_change("value", on_wire_preset)
    for w in (hf_sigma, hf_power):
        w.on_change("value_throttled", on_hf_spinner)
    hf_zero_dc.on_change("active", on_hf_spinner)
    lf_tau.on_change("value_throttled", on_lf_spinner)
    for w in (wire_sigma, wire_power):
        w.on_change("value_throttled", on_wire_spinner)

    # Filter B bindings
    hf_preset_b.on_change("value", on_hf_preset_b)
    lf_preset_b.on_change("value", on_lf_preset_b)
    wire_preset_b.on_change("value", on_wire_preset_b)
    for w in (hf_sigma_b, hf_power_b):
        w.on_change("value_throttled", on_hf_spinner_b)
    hf_zero_dc_b.on_change("active", on_hf_spinner_b)
    lf_tau_b.on_change("value_throttled", on_lf_spinner_b)
    for w in (wire_sigma_b, wire_power_b):
        w.on_change("value_throttled", on_wire_spinner_b)

    update_btn.on_click(render)

    range_btn.on_click(on_apply_range)
    reset_btn.on_click(on_reset_range)

    # ---- initial state ----------------------------------------------------
    refresh_preset_lists()
    pdata0 = cache.get_plane(file_select.value, plane_radio.active)
    chan_input.value = str(pdata0.ch_offset + best_channel(pdata0))
    reload_and_render()
    on_reset_range()

    # ---- layout -----------------------------------------------------------
    sel_row = row(file_select, plane_radio, chan_input, prev_btn, next_btn, best_btn,
                  update_btn, status_div,
                  sizing_mode="stretch_width")

    label_a = Div(text="<b style='color:#1f77b4;font-size:13px'>▶ Filter A (blue)</b>"
                       " &nbsp; Wire: H(k)=exp(−0.5·(f_w/σ)^p) &nbsp;|&nbsp; "
                       "HF: H(f)=exp(−0.5·(f/σ)^p) &nbsp;|&nbsp; "
                       "LF: L(f)=1−exp(−(f/τ)²)")
    wire_box_a = column(
        Div(text="<b>Wire</b>"), wire_preset, wire_sigma, wire_power,
    )
    hf_box_a = column(
        Div(text="<b>HF</b>"), hf_preset, hf_sigma, hf_power, hf_zero_dc,
    )
    lf_box_a = column(
        Div(text="<b>LF</b>"), lf_preset, lf_tau,
    )

    label_b = Div(text="<b style='color:#ff7f0e;font-size:13px'>▶ Filter B (orange dashed)</b>"
                       " &nbsp; same filter forms as A")
    wire_box_b = column(
        Div(text="<b>Wire</b>"), wire_preset_b, wire_sigma_b, wire_power_b,
    )
    hf_box_b = column(
        Div(text="<b>HF</b>"), hf_preset_b, hf_sigma_b, hf_power_b, hf_zero_dc_b,
    )
    lf_box_b = column(
        Div(text="<b>LF</b>"), lf_preset_b, lf_tau_b,
    )

    range_row = row(tmin_input, tmax_input, range_btn, reset_btn)
    layout = column(
        sel_row,
        info_div,
        label_a,
        row(wire_box_a, hf_box_a, lf_box_a, sizing_mode="stretch_width"),
        label_b,
        row(wire_box_b, hf_box_b, lf_box_b, sizing_mode="stretch_width"),
        range_row,
        f_time,
        f_filter,
        f_spec,
        sizing_mode="stretch_width",
    )
    curdoc().add_root(layout)
    curdoc().title = "SP filter-tuning viewer"


main(sys.argv)

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

Production overlay: ``h{u,v,w}_gauss<ident>`` is the production
post-Wire+Wiener+LF+ROI output.  A toggle puts it on the time-domain
plot for direct visual comparison of "your tuned filter" vs.
"production".

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
    Slider,
    TextInput,
)
from bokeh.plotting import figure


TICK_US = 0.5
TICK_S = TICK_US * 1e-6


# ---- presets ---------------------------------------------------------------

HF_PRESETS_PDHD = {
    "(none)":         None,
    "Gaus_wide":      dict(sigma=0.12,     power=2.0,     flag=True),
    "Wiener_tight U": dict(sigma=0.221933, power=6.55413, flag=True),
    "Wiener_tight V": dict(sigma=0.222723, power=8.75998, flag=True),
    "Wiener_tight W": dict(sigma=0.225567, power=3.47846, flag=True),
    "Wiener_wide U":  dict(sigma=0.186765, power=5.05429, flag=True),
    "Wiener_wide V":  dict(sigma=0.1936,   power=5.77422, flag=True),
    "Wiener_wide W":  dict(sigma=0.175722, power=4.37928, flag=True),
}

HF_PRESETS_PDVD = {
    "(none)":         None,
    "Gaus_wide":      dict(sigma=0.12,     power=2.0,     flag=True),
    "Wiener_tight U": dict(sigma=0.15,     power=5.5,     flag=True),
    "Wiener_tight V": dict(sigma=0.15,     power=5.0,     flag=True),
    "Wiener_tight W": dict(sigma=0.25,     power=3.0,     flag=True),
    "Wiener_wide U":  dict(sigma=0.186765, power=5.05429, flag=True),
    "Wiener_wide V":  dict(sigma=0.1936,   power=5.77422, flag=True),
    "Wiener_wide W":  dict(sigma=0.175722, power=4.37928, flag=True),
}

LF_PRESETS_PDHD = {
    "(none)":         None,
    "ROI_loose_lf":   dict(tau=0.002),
    "ROI_tight_lf":   dict(tau=0.014),
    "ROI_tighter_lf": dict(tau=0.06),
}

LF_PRESETS_PDVD = {
    "(none)":         None,
    "ROI_loose_lf":   dict(tau=0.00175),
    "ROI_tight_lf":   dict(tau=0.0185),
    "ROI_tighter_lf": dict(tau=0.145),
}

# Wire filter presets (from cfg/.../sp-filters.jsonnet).  σ_code is in
# cycles-per-wire units; larger σ_code → flatter in wire-freq → narrower
# spatial smearing.  See pdvd/sp_plot/compare_sp_filters.py.
WIRE_PRESETS_PDHD = {
    "(none)":      None,
    "Wire_ind":    dict(sigma=0.75 / np.sqrt(np.pi), power=2.0),    # ≈0.4231
    "Wire_col":    dict(sigma=10.0 / np.sqrt(np.pi), power=2.0),    # ≈5.642
}

WIRE_PRESETS_PDVD = {
    "(none)":      None,
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
    gauss:    real (n_chan, n_tick) or None — production post-everything output.
    F:        rfft along time of rawdecon, complex (n_chan, n_freq_t).
    G:        fft along axis 0 of F, complex (n_chan, n_freq_t).  Used to
              apply Wire filter via 2D-frequency multiplication.
    ch_offset: global channel id of local row 0.
    used_fallback: True iff rawdecon was missing and we fell back to wiener.
    """
    def __init__(self, rawdecon, gauss, ch_offset, used_fallback):
        self.rawdecon = rawdecon
        self.gauss = gauss
        self.ch_offset = ch_offset
        self.used_fallback = used_fallback
        # Cache the FFTs lazily on first render.
        self._F = None
        self._G = None

    def F(self) -> np.ndarray:
        if self._F is None:
            self._F = np.fft.rfft(self.rawdecon.astype(np.float64), axis=1)
        return self._F

    def G(self) -> np.ndarray:
        if self._G is None:
            self._G = np.fft.fft(self.F(), axis=0)
        return self._G


class MagnifyCache:
    """Lazy per-file loader for ``h{u,v,w}_rawdecon<ident>`` (preferred)
    and ``h{u,v,w}_gauss<ident>`` (production overlay)."""
    RD_PFX = {0: "hu_rawdecon", 1: "hv_rawdecon", 2: "hw_rawdecon"}
    WN_PFX = {0: "hu_wiener",   1: "hv_wiener",   2: "hw_wiener"}
    GS_PFX = {0: "hu_gauss",    1: "hv_gauss",    2: "hw_gauss"}

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
        gs_name = f"{self.GS_PFX[plane]}{spec.ident}"
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
                    gauss=None, ch_offset=0, used_fallback=False)
                return self._frames[key]
            rawdecon = np.asarray(src_h.values(), dtype=np.float32)
            ch_offset = int(round(src_h.to_numpy()[1][0]))
            gauss = (np.asarray(f[gs_name].values(), dtype=np.float32)
                     if gs_name in file_keys else None)
        self._frames[key] = PlaneData(rawdecon=rawdecon, gauss=gauss,
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
        gauss_t=None,   # current channel gauss (production) waveform or None
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
    hf_preset = Select(title="HF preset:", value="(none)",
                       options=list(HF_PRESETS_PDHD.keys()), width=200)
    hf_sigma  = Slider(title="HF σ (MHz)", start=0.05, end=0.50, step=0.005,
                       value=0.20, width=320)
    hf_power  = Slider(title="HF power",   start=1.0,  end=10.0, step=0.1,
                       value=4.0, width=320)
    hf_zero_dc = CheckboxGroup(labels=["Zero DC (flag=true)"], active=[0])

    # LF column
    lf_preset   = Select(title="LF preset:", value="(none)",
                         options=list(LF_PRESETS_PDHD.keys()), width=200)
    lf_tau      = Slider(title="LF τ (MHz)", start=0.0005, end=0.20, step=0.0005,
                         value=0.014, width=320)
    lf_tau_text = TextInput(title="LF τ (typed, MHz):", value="0.014", width=140)

    # Wire column (geometric wire-axis filter; production has Wire_ind for U/V,
    # Wire_col for W; user can override).
    wire_preset = Select(title="Wire preset:", value="(none)",
                         options=list(WIRE_PRESETS_PDHD.keys()), width=200)
    wire_sigma  = Slider(title="Wire σ (cycles/wire)", start=0.1, end=10.0,
                         step=0.05, value=2.82, width=320)
    wire_power  = Slider(title="Wire power", start=1.0, end=6.0, step=0.1,
                         value=2.0, width=320)

    # Production-gauss overlay toggle.  Reads h{u,v,w}_gauss<id> from the
    # same magnify file and shows it dotted-green on the time-domain plot.
    gauss_overlay_chk = CheckboxGroup(labels=["Show production 'gauss' overlay"], active=[])

    # Time-range
    tmin_input = TextInput(title="t-min (tick)", value="0", width=120)
    tmax_input = TextInput(title="t-max (tick)", value="6000", width=120)
    range_btn  = Button(label="Apply range", width=110)
    reset_btn  = Button(label="Reset range", width=110)

    # ---- figures ----------------------------------------------------------
    fig_kw_t = dict(height=300, sizing_mode="stretch_width",
                    active_scroll="wheel_zoom",
                    tools="pan,wheel_zoom,box_zoom,reset,save")

    f_time = figure(title="Waveform — raw decon (dashed grey), filtered (blue), "
                          "production gauss (dotted green, optional)",
                    **fig_kw_t)
    f_time.xaxis.axis_label = "tick (0.5 µs each)"
    f_time.yaxis.axis_label = "amplitude"

    f_filter = figure(title="Filter shapes |H_wire(k)|, |H_HF(f)|, |L_LF(f)|, and HF × LF product",
                      height=240, sizing_mode="stretch_width",
                      active_scroll="wheel_zoom",
                      tools="pan,wheel_zoom,box_zoom,reset,save")
    f_filter.xaxis.axis_label = "frequency (MHz, time axis); Wire shown on rescaled common x"
    f_filter.yaxis.axis_label = "filter value (0 .. 1)"
    f_filter.x_range.start = 0.0
    f_filter.x_range.end = 1.0
    f_filter.y_range.start = -0.05
    f_filter.y_range.end = 1.05

    f_spec = figure(title="Channel spectrum |X(f)| (raw decon, log-y)",
                    height=220, sizing_mode="stretch_width",
                    active_scroll="wheel_zoom",
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    y_axis_type="log")
    f_spec.xaxis.axis_label = "frequency (MHz)"
    f_spec.yaxis.axis_label = "|X(f)|"
    f_spec.x_range.start = 0.0
    f_spec.x_range.end = 1.0

    src_raw   = ColumnDataSource(data=dict(x=[], y=[]))
    src_filt  = ColumnDataSource(data=dict(x=[], y=[]))
    src_gauss = ColumnDataSource(data=dict(x=[], y=[]))
    src_hf    = ColumnDataSource(data=dict(x=[], y=[]))
    src_lf    = ColumnDataSource(data=dict(x=[], y=[]))
    src_wire  = ColumnDataSource(data=dict(x=[], y=[]))
    src_prod  = ColumnDataSource(data=dict(x=[], y=[]))
    src_spec  = ColumnDataSource(data=dict(x=[], y=[]))

    f_time.line("x", "y", source=src_raw,  line_width=1, color="#808080",
                line_dash="dashed", legend_label="raw decon")
    f_time.line("x", "y", source=src_filt, line_width=1.5, color="#1f77b4",
                legend_label="filtered (Wire×HF×LF)")
    f_time.line("x", "y", source=src_gauss, line_width=1.5, color="#2ca02c",
                line_dash="dotted", legend_label="production gauss (post-everything)")
    f_time.legend.location = "top_left"
    f_time.legend.click_policy = "hide"

    f_filter.line("x", "y", source=src_hf,   line_width=1.5, color="#1f77b4",
                  legend_label="HF (time-freq, MHz)")
    f_filter.line("x", "y", source=src_lf,   line_width=1.5, color="#d62728",
                  legend_label="LF (time-freq, MHz)")
    f_filter.line("x", "y", source=src_prod, line_width=2.0, color="#2ca02c",
                  line_dash="dashed", legend_label="HF × LF")
    f_filter.line("x", "y", source=src_wire, line_width=1.5, color="#9467bd",
                  legend_label="Wire (wire-freq, x rescaled to 0..1)")
    f_filter.legend.location = "center_right"
    f_filter.legend.click_policy = "hide"

    f_spec.line("x", "y", source=src_spec, line_width=1, color="#666666")

    # ---- helpers ----------------------------------------------------------
    def detector() -> str:
        return cache.detector(file_select.value)

    def hf_options() -> dict:
        return HF_PRESETS_PDHD if detector() == "pdhd" else HF_PRESETS_PDVD

    def lf_options() -> dict:
        return LF_PRESETS_PDHD if detector() == "pdhd" else LF_PRESETS_PDVD

    def wire_options() -> dict:
        return WIRE_PRESETS_PDHD if detector() == "pdhd" else WIRE_PRESETS_PDVD

    def refresh_preset_lists():
        hf_preset.options = list(hf_options().keys())
        lf_preset.options = list(lf_options().keys())
        wire_preset.options = list(wire_options().keys())
        if hf_preset.value not in hf_preset.options:
            hf_preset.value = "(none)"
        if lf_preset.value not in lf_preset.options:
            lf_preset.value = "(none)"
        if wire_preset.value not in wire_preset.options:
            wire_preset.value = "(none)"

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
        gauss_wave = (pdata.gauss[local].astype(np.float64)
                      if pdata.gauss is not None else None)
        f_mhz = np.fft.rfftfreq(n_tick, TICK_S) * 1e-6
        X = np.fft.rfft(wave)

        state["wave_t"] = wave
        state["gauss_t"] = gauss_wave
        state["X"] = X
        state["f_mhz"] = f_mhz
        state["n_tick"] = n_tick
        state["n_chan"] = n_chan
        state["local_idx"] = local
        state["ch_global"] = ch_global

        # Spectrum panel — log-magnitude, replace zeros with eps for log axis.
        mag = np.abs(X)
        mag = np.where(mag <= 0, 1e-12, mag)
        src_spec.data = dict(x=f_mhz, y=mag)

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
            # Full-plane wire-direction FFT path.  G is cached lazily.
            W = wire_kernel(n_chan, wire_kw["sigma"], wire_kw["power"])
            G_filt = plane_data.G() * W[:, None] * HL[None, :]
            F2 = np.fft.ifft(G_filt, axis=0).real
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

        hf_kw   = hf_kw_from_widgets()
        lf_kw   = lf_kw_from_widgets()
        wire_kw = wire_kw_from_widgets()

        H, L, HL, Y = _compute_filtered_waveform(
            pdata, hf_kw, lf_kw, wire_kw, f_mhz, n_chan, n_tick, local_idx)

        # Wire kernel for the freq-domain panel — show on a 0..1 rescaled x
        # so the line shares the same x-range as HF/LF (cosmetic only).
        if wire_kw is not None:
            W = wire_kernel(n_chan, wire_kw["sigma"], wire_kw["power"])
            half = n_chan // 2 + 1
            wire_x = np.linspace(0.0, 1.0, half)
            wire_y = W[:half]   # take the positive-frequency half
        else:
            wire_x = np.array([0.0, 1.0])
            wire_y = np.array([1.0, 1.0])

        dt = (time.perf_counter() - t0) * 1000.0

        src_raw.data  = dict(x=x_ticks, y=wave_t)
        src_filt.data = dict(x=x_ticks, y=Y)

        # Production gauss overlay — only when the checkbox is on AND the
        # current channel has gauss data.  Empty source hides the line.
        if 0 in gauss_overlay_chk.active and state["gauss_t"] is not None:
            src_gauss.data = dict(x=x_ticks, y=state["gauss_t"])
        else:
            src_gauss.data = dict(x=[], y=[])

        src_hf.data   = dict(x=f_mhz, y=H)
        src_lf.data   = dict(x=f_mhz, y=L)
        src_prod.data = dict(x=f_mhz, y=HL)
        src_wire.data = dict(x=wire_x, y=wire_y)

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
        if wire_preset.value not in ("(none)",) and new_wire in wire_preset.options:
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
        if kw is not None:
            # Push preset values into sliders without infinite-looping.
            hf_sigma.value = kw["sigma"]
            hf_power.value = kw["power"]
            hf_zero_dc.active = [0] if kw["flag"] else []
        render()

    def on_lf_preset(_a, _o, new):
        kw = lf_options().get(new)
        if kw is not None:
            lf_tau.value = kw["tau"]
            lf_tau_text.value = f"{kw['tau']:.6g}"
        render()

    def on_hf_slider(_a, _o, _n):
        # Switch to "Custom" if value diverges from preset.
        kw = hf_options().get(hf_preset.value)
        if kw is not None:
            same = (abs(kw["sigma"] - hf_sigma.value) < 1e-9
                    and abs(kw["power"] - hf_power.value) < 1e-9
                    and (kw["flag"] == (0 in hf_zero_dc.active)))
            if not same:
                hf_preset.value = "(none)"  # custom — use sliders directly
        render()

    def on_lf_slider(_a, _o, _n):
        kw = lf_options().get(lf_preset.value)
        if kw is not None and abs(kw["tau"] - lf_tau.value) > 1e-12:
            lf_preset.value = "(none)"
        # Keep typed-text in sync (one-way: slider → text).
        if lf_tau_text.value != f"{lf_tau.value:.6g}":
            lf_tau_text.value = f"{lf_tau.value:.6g}"
        render()

    def on_lf_tau_text(_a, _o, new):
        try:
            v = float(new)
        except ValueError:
            return
        # Clamp to slider range, but allow values outside by widening the slider.
        if v < lf_tau.start:
            lf_tau.start = v
        if v > lf_tau.end:
            lf_tau.end = v
        if abs(lf_tau.value - v) > 1e-12:
            lf_tau.value = v   # triggers on_lf_slider → render

    def on_wire_preset(_a, _o, new):
        kw = wire_options().get(new)
        if kw is not None:
            wire_sigma.value = kw["sigma"]
            wire_power.value = kw["power"]
        render()

    def on_wire_slider(_a, _o, _n):
        kw = wire_options().get(wire_preset.value)
        if kw is not None:
            same = (abs(kw["sigma"] - wire_sigma.value) < 1e-9
                    and abs(kw["power"] - wire_power.value) < 1e-9)
            if not same:
                wire_preset.value = "(none)"
        render()

    def on_gauss_overlay(_a, _o, _n):
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

    def on_reset_range():
        n = state["n_tick"] or 6000
        tmin_input.value = "0"
        tmax_input.value = str(n)
        f_time.x_range.start = 0
        f_time.x_range.end = n

    file_select.on_change("value", on_file_change)
    plane_radio.on_change("active", on_plane_change)
    chan_input.on_change("value", on_chan_change)
    prev_btn.on_click(lambda: step_chan(-1))
    next_btn.on_click(lambda: step_chan(+1))
    best_btn.on_click(on_best_click)

    hf_preset.on_change("value", on_hf_preset)
    lf_preset.on_change("value", on_lf_preset)
    wire_preset.on_change("value", on_wire_preset)
    # Sliders fire render only on mouse-release (value_throttled), to avoid
    # spamming recomputes during drag.  The slider knob still moves visually.
    for w in (hf_sigma, hf_power):
        w.on_change("value_throttled", on_hf_slider)
    hf_zero_dc.on_change("active", on_hf_slider)
    lf_tau.on_change("value_throttled", on_lf_slider)
    lf_tau_text.on_change("value", on_lf_tau_text)
    for w in (wire_sigma, wire_power):
        w.on_change("value_throttled", on_wire_slider)
    gauss_overlay_chk.on_change("active", on_gauss_overlay)
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
    wire_box = column(
        Div(text="<b>Wire filter</b> — H(k) = exp(-0.5·(f_w/σ)^p), wire-axis"),
        wire_preset, wire_sigma, wire_power,
    )
    hf_box = column(
        Div(text="<b>High-frequency filter</b> — H(f) = exp(-0.5·(f/σ)^p)"),
        hf_preset, hf_sigma, hf_power, hf_zero_dc,
    )
    lf_box = column(
        Div(text="<b>Low-frequency filter</b> — L(f) = 1 − exp(−(f/τ)²)"),
        lf_preset, lf_tau, lf_tau_text,
    )
    range_row = row(tmin_input, tmax_input, range_btn, reset_btn,
                    gauss_overlay_chk)
    layout = column(
        sel_row,
        info_div,
        row(wire_box, hf_box, lf_box, sizing_mode="stretch_width"),
        range_row,
        f_time,
        f_filter,
        f_spec,
        sizing_mode="stretch_width",
    )
    curdoc().add_root(layout)
    curdoc().title = "SP filter-tuning viewer"


main(sys.argv)

"""Bokeh event display for PDHD light (photon-detector) reconstruction.

Reads the WCT-native optical products written per event into
``pdhd/work/<RUN6>_<EVT>/`` by the flash/ reconstruction chain:
  - ``opflash_pdhd-wct.tar.gz``  -> flash / ophit tensor set (design.md s3.4)
  - ``light-frames-wct.tar.bz2`` -> per-channel raw + deconvolved waveforms

Four linked panels:
  1. Flash list -- a table of the flashes in the event (+ a time-vs-PE overview);
     selecting a flash drives panels 2 and 3.
  2. Light geometry -- the 160 OpDet positions in three projections (Z-Y, X-Y,
     X-Z), the selected flash's OpDets coloured/sized by PE, the PDHD TPC boxes,
     and the flash y/z centroid.
  3. OpHit histogram -- PE per OpChannel for the selected flash's OpHits (a 1D
     bar chart) plus a table; selecting one OpHit drives panel 4.
  4. Waveform -- the selected OpHit channel's raw ADC and deconvolved snippet
     overlaid, with the hit start/peak time marked.

Run via the serve_pd_viewer.sh launcher:
    bokeh serve --port 5014 pd_viewer.py --args <workdir> [<opdet-geom.json>]
Remote viewing:  ssh -L 5014:localhost:5014 user@wcgpu1
                 then open http://localhost:5014/pd_viewer
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import tarfile

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BoxAnnotation, ColumnDataSource, DataTable, Div, HoverTool, LinearAxis,
    LinearColorMapper, Range1d, Select, Span, TableColumn,
)
from bokeh.plotting import figure
from bokeh.palettes import Viridis256

# Default OpDet geometry (mm); overridable as the 2nd --args token.
GEOM_DEFAULT = ("/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/"
                "experiment/pdhd/pdhd-opdet-geom.json")

# PDHD TPC boxes [cm], from wire-cell-bee3/docs/protodune_geometry.md.  Four APAs
# in a 2 (drift) x 2 (beam-z) layout, central cathode at x=0.  (xlo,xhi,zlo,zhi);
# y is common [7.61, 606.67].
TPC_BOXES = [
    (-353.202, -2.54, -0.10, 230.573),
    (2.54, 353.002, -0.10, 230.573),
    (-353.202, -2.54, 231.96, 462.633),
    (2.54, 353.002, 231.96, 462.633),
]
TPC_Y = (7.61, 606.67)


# ---- data ------------------------------------------------------------------

def scan_events(workdir):
    """[(label 'RUN/EVT', dirpath), ...] for dirs that have a WCT opflash."""
    out = []
    for name in sorted(os.listdir(workdir)):
        d = os.path.join(workdir, name)
        if not re.match(r"^\d{6}_\d+$", name):
            continue
        if os.path.isfile(os.path.join(d, "opflash_pdhd-wct.tar.gz")):
            run, evt = name.split("_")
            out.append((f"{run}/{evt}", d))
    return out


def _members(tf):
    return {m.name: m for m in tf.getmembers() if m.isfile()}


def _load_npy(tf, member):
    return np.load(io.BytesIO(tf.extractfile(member).read()))


class EventData:
    """All products for one event directory, loaded into memory."""

    def __init__(self, dirpath, geom):
        self.dirpath = dirpath
        self.geom = geom               # opdet -> (x,y,z) cm, shape (160,3)
        # --- opflash tensor set ---
        with tarfile.open(os.path.join(dirpath, "opflash_pdhd-wct.tar.gz")) as tf:
            mem = _members(tf)
            pat = re.compile(r"opflash_tensor_(\d+)_0_array\.npy$")
            evt = next(pat.search(n).group(1) for n in mem if pat.search(n))
            self.evt = int(evt)
            self.t0 = _load_npy(tf, f"opflash_tensor_{evt}_0_array.npy")  # [nf,161]
            self.t1 = _load_npy(tf, f"opflash_tensor_{evt}_1_array.npy")  # [nf,8]
            self.t2 = _load_npy(tf, f"opflash_tensor_{evt}_2_array.npy")  # [nh,9]
        # --- waveform frames ---
        with tarfile.open(os.path.join(dirpath, "light-frames-wct.tar.bz2")) as tf:
            mem = _members(tf)

            def pick(prefix):
                return next(n for n in mem if os.path.basename(n).startswith(prefix))
            self.f_raw = _load_npy(tf, pick("frame_raw_"))
            self.f_dec = _load_npy(tf, pick("frame_decon_"))
            self.ch_raw = _load_npy(tf, pick("channels_raw_")).astype(int)
            self.ch_dec = _load_npy(tf, pick("channels_decon_")).astype(int)
            self.ti_raw = _load_npy(tf, pick("tickinfo_raw_"))
            self.ti_dec = _load_npy(tf, pick("tickinfo_decon_"))
        self.row_raw = {int(c): i for i, c in enumerate(self.ch_raw)}
        self.row_dec = {int(c): i for i, c in enumerate(self.ch_dec)}

    def nflash(self):
        return self.t0.shape[0]

    def flash_time_us(self):
        return self.t0[:, 0] / 1000.0

    def flash_pe(self, fid):
        """PE per OpDet (len 160) for flash row fid."""
        return self.t0[fid, 1:161]

    def ophits_of(self, fid):
        """tensor2 rows whose flash_id == fid."""
        return self.t2[self.t2[:, 7].astype(int) == fid]


def snippet_window(row, t_ns, frame_time_ns, tick_ns, pad=64):
    """(lo, hi) sample indices of the nonzero snippet containing the hit time.

    row is the dense waveform (zeros between snippets).  Locate the contiguous
    nonzero block holding the global index of t_ns; if that index falls in a gap,
    fall back to a +/-512 window.  Returns clamped (lo, hi)."""
    n = len(row)
    idx = int(round((t_ns - frame_time_ns) / tick_ns))
    idx = max(0, min(n - 1, idx))
    nz = row != 0.0
    if not nz[idx]:
        return max(0, idx - 512), min(n, idx + 512)
    lo = idx
    while lo > 0 and nz[lo - 1]:
        lo -= 1
    hi = idx
    while hi < n - 1 and nz[hi + 1]:
        hi += 1
    return max(0, lo - pad), min(n, hi + 1 + pad)


def load_geom(path):
    g = json.load(open(path))
    arr = np.zeros((160, 3))
    for d in g["opdets"]:
        arr[int(d["opdet"])] = (d["x"] / 10.0, d["y"] / 10.0, d["z"] / 10.0)  # mm->cm
    return arr


# ---- app -------------------------------------------------------------------

def main(argv):
    if len(argv) < 2:
        print("usage: pd_viewer.py --args <workdir> [<opdet-geom.json>]",
              file=sys.stderr)
        sys.exit(1)
    workdir = argv[1]
    geom = load_geom(argv[2] if len(argv) > 2 else GEOM_DEFAULT)
    events = scan_events(workdir)
    if not events:
        print(f"no WCT opflash events under {workdir}", file=sys.stderr)
        sys.exit(1)

    state = dict(ev=None, fid=-1)

    # ---- sources ----------------------------------------------------------
    flash_src = ColumnDataSource(dict(fid=[], time_us=[], total_pe=[],
                                      nhits=[], y_cm=[], z_cm=[]))
    od_base = ColumnDataSource(dict(x=[], y=[], z=[], opdet=[]))   # all 160, grey
    od_pe = ColumnDataSource(dict(x=[], y=[], z=[], opdet=[], pe=[], size=[]))
    cen_src = ColumnDataSource(dict(y=[], z=[]))                   # flash centroid
    ophit_src = ColumnDataSource(dict(channel=[], pe=[], peak_us=[],
                                      amplitude=[], width_ns=[]))
    wav_src = ColumnDataSource(dict(t=[], raw=[], dec=[]))

    pe_mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)

    # ---- panel 1: flashes -------------------------------------------------
    flash_tbl = DataTable(
        source=flash_src, width=560, height=300, index_position=None,
        columns=[
            TableColumn(field="fid", title="flash"),
            TableColumn(field="time_us", title="t [us]"),
            TableColumn(field="total_pe", title="total PE"),
            TableColumn(field="nhits", title="nhits"),
            TableColumn(field="y_cm", title="y [cm]"),
            TableColumn(field="z_cm", title="z [cm]"),
        ])
    f_ov = figure(title="flashes: time vs total PE (tap to select)", width=560,
                  height=300, x_axis_label="t [us]", y_axis_label="total PE",
                  tools="pan,wheel_zoom,box_zoom,reset,save,tap",
                  active_scroll="wheel_zoom")
    f_ov.scatter("time_us", "total_pe", source=flash_src, size=7,
                 color="#1f77b4", alpha=0.8)

    # ---- panel 2: geometry projections ------------------------------------
    def add_boxes(p, cx, cy):
        for (xlo, xhi, zlo, zhi) in TPC_BOXES:
            rng = {"x": (xlo, xhi), "y": TPC_Y, "z": (zlo, zhi)}
            l, r = rng[cx]
            b, t = rng[cy]
            p.quad(left=l, right=r, bottom=b, top=t, fill_alpha=0.0,
                   line_color="#888888", line_width=1)

    def proj(title, xl, yl, cx, cy):
        p = figure(title=title, width=380, height=360, x_axis_label=xl,
                   y_axis_label=yl, tools="pan,wheel_zoom,box_zoom,reset,save",
                   active_scroll="wheel_zoom")
        add_boxes(p, cx, cy)
        p.scatter(cx, cy, source=od_base, size=4, color="#bbbbbb", alpha=0.5)
        r = p.scatter(cx, cy, source=od_pe, size="size",
                      fill_color={"field": "pe", "transform": pe_mapper},
                      line_color="black", line_alpha=0.3, alpha=0.95)
        p.add_tools(HoverTool(renderers=[r], tooltips=[("OpDet", "@opdet"),
                                                       ("PE", "@pe{0.0}")]))
        if (cx, cy) == ("z", "y"):
            p.scatter("z", "y", source=cen_src, size=16, marker="x",
                      line_color="red", line_width=3)
        return p
    f_zy = proj("Z-Y (light geometry)", "Z [cm]", "Y [cm]", "z", "y")
    f_xy = proj("X-Y", "X [cm]", "Y [cm]", "x", "y")
    f_xz = proj("X-Z", "X [cm]", "Z [cm]", "x", "z")

    # ---- panel 3: ophit histogram + table ---------------------------------
    f_hit = figure(title="OpHits of selected flash: PE per OpChannel "
                   "(tap to select)", width=760, height=300,
                   x_axis_label="OpChannel", y_axis_label="PE",
                   tools="pan,wheel_zoom,box_zoom,reset,save,tap",
                   active_scroll="wheel_zoom")
    hit_r = f_hit.vbar(x="channel", top="pe", width=0.7, source=ophit_src,
                       fill_color="#2ca02c", line_color="#2ca02c")
    f_hit.add_tools(HoverTool(renderers=[hit_r],
                              tooltips=[("ch", "@channel"), ("PE", "@pe{0.0}"),
                                        ("t", "@peak_us{0.000} us")]))
    hit_tbl = DataTable(
        source=ophit_src, width=360, height=300, index_position=None,
        columns=[
            TableColumn(field="channel", title="ch"),
            TableColumn(field="peak_us", title="peak [us]"),
            TableColumn(field="pe", title="PE"),
            TableColumn(field="amplitude", title="amp"),
            TableColumn(field="width_ns", title="width [ns]"),
        ])

    # ---- panel 4: waveform ------------------------------------------------
    raw_rng = Range1d(-1, 1)
    dec_rng = Range1d(-1, 1)
    f_wav = figure(title="OpHit waveform (raw vs decon)", width=1140, height=320,
                   x_axis_label="t [us]", y_axis_label="raw ADC",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   active_scroll="wheel_zoom", y_range=raw_rng)
    f_wav.extra_y_ranges = {"dec": dec_rng}
    f_wav.add_layout(LinearAxis(y_range_name="dec", axis_label="decon [SPE]"),
                     "right")
    f_wav.line("t", "raw", source=wav_src, line_color="#777777", line_width=1.2,
               legend_label="raw ADC")
    f_wav.line("t", "dec", source=wav_src, line_color="#d62728", line_width=1.5,
               y_range_name="dec", legend_label="decon")
    f_wav.legend.click_policy = "hide"
    wav_peak = Span(location=0, dimension="height", line_color="blue",
                    line_dash="dashed", line_width=1)
    wav_start = Span(location=0, dimension="height", line_color="blue",
                     line_alpha=0.4, line_width=1)
    f_wav.add_layout(wav_peak)
    f_wav.add_layout(wav_start)
    wav_note = Div(text="", width=1140)

    # ---- widgets ----------------------------------------------------------
    evt_select = Select(title="Event (run/evt)", value=events[0][0],
                        options=[lbl for lbl, _ in events], width=200)
    head = Div(text="", width=1140)
    status = Div(text="", width=1140)

    # ---- render -----------------------------------------------------------
    def render_geometry(fid):
        ev = state["ev"]
        cen_src.data = dict(y=[], z=[])
        if fid < 0:
            od_pe.data = dict(x=[], y=[], z=[], opdet=[], pe=[], size=[])
            return
        pe = ev.flash_pe(fid)
        sel = np.where(pe > 0)[0]
        if len(sel):
            pmax = float(pe[sel].max())
            pe_mapper.high = pmax
            od_pe.data = dict(
                x=ev.geom[sel, 0].tolist(), y=ev.geom[sel, 1].tolist(),
                z=ev.geom[sel, 2].tolist(), opdet=sel.tolist(),
                pe=pe[sel].tolist(),
                size=(6 + 22 * np.sqrt(pe[sel] / pmax)).tolist())
        else:
            od_pe.data = dict(x=[], y=[], z=[], opdet=[], pe=[], size=[])
        yc, zc = ev.t1[fid, 2] / 10.0, ev.t1[fid, 3] / 10.0
        cen_src.data = dict(y=[yc], z=[zc])

    def render_ophits(fid):
        ev = state["ev"]
        if fid < 0:
            ophit_src.data = dict(channel=[], pe=[], peak_us=[], amplitude=[],
                                  width_ns=[])
            return
        h = ev.ophits_of(fid)
        order = np.argsort(h[:, 0]) if len(h) else np.array([], int)
        h = h[order]
        ophit_src.data = dict(
            channel=h[:, 0].astype(int).tolist(),
            pe=h[:, 5].tolist(),
            peak_us=(h[:, 1] / 1000.0).tolist(),
            amplitude=h[:, 4].tolist(),
            width_ns=h[:, 2].tolist())
        ophit_src.selected.indices = []

    def render_waveform(hit_idx):
        ev = state["ev"]
        d = ophit_src.data
        if hit_idx is None or hit_idx >= len(d["channel"]):
            wav_src.data = dict(t=[], raw=[], dec=[])
            wav_note.text = ""
            return
        ch = int(d["channel"][hit_idx])
        peak_ns = d["peak_us"][hit_idx] * 1000.0
        if ch not in ev.row_raw or ch not in ev.row_dec:
            wav_src.data = dict(t=[], raw=[], dec=[])
            wav_note.text = f"<b>OpChannel {ch}</b>: no waveform snippet on file."
            return
        rrow = ev.f_raw[ev.row_raw[ch]]
        drow = ev.f_dec[ev.row_dec[ch]]
        ft, tick = ev.ti_raw[0], ev.ti_raw[1]
        lo, hi = snippet_window(rrow, peak_ns, ft, tick)
        t_us = (ft + np.arange(lo, hi) * tick) / 1000.0
        wav_src.data = dict(t=t_us.tolist(), raw=rrow[lo:hi].tolist(),
                            dec=drow[lo:hi].tolist())
        rmn, rmx = float(rrow[lo:hi].min()), float(rrow[lo:hi].max())
        dmn, dmx = float(drow[lo:hi].min()), float(drow[lo:hi].max())
        raw_rng.start, raw_rng.end = _pad(rmn, rmx)
        dec_rng.start, dec_rng.end = _pad(dmn, dmx)
        wav_peak.location = peak_ns / 1000.0
        wav_start.location = d["peak_us"][hit_idx] - d["width_ns"][hit_idx] / 2000.0
        wav_note.text = (f"<b>OpChannel {ch}</b> &mdash; flash {state['fid']}, "
                         f"PE {d['pe'][hit_idx]:.1f}, "
                         f"peak {d['peak_us'][hit_idx]:.3f} us")

    def render_flash(fid):
        state["fid"] = fid
        render_geometry(fid)
        render_ophits(fid)
        render_waveform(None)

    def load_event(label):
        dirpath = dict(events)[label]
        ev = EventData(dirpath, geom)
        state["ev"] = ev
        t_us = ev.flash_time_us()
        flash_src.selected.indices = []
        flash_src.data = dict(
            fid=list(range(ev.nflash())),
            time_us=np.round(t_us, 3).tolist(),
            total_pe=np.round(ev.t1[:, 1], 1).tolist(),
            nhits=ev.t1[:, 7].astype(int).tolist(),
            y_cm=np.round(ev.t1[:, 2] / 10.0, 1).tolist(),
            z_cm=np.round(ev.t1[:, 3] / 10.0, 1).tolist())
        od_base.data = dict(x=ev.geom[:, 0].tolist(), y=ev.geom[:, 1].tolist(),
                            z=ev.geom[:, 2].tolist(), opdet=list(range(160)))
        run = os.path.basename(dirpath).split("_")[0]
        head.text = (f"<h2>PDHD light viewer &mdash; run {int(run)} evt {ev.evt}</h2>"
                     f"<small>{dirpath}: {ev.nflash()} flashes, "
                     f"{ev.t2.shape[0]} ophits</small>")
        render_flash(-1)

    # ---- callbacks --------------------------------------------------------
    def on_event(attr, old, new):
        load_event(new)

    def on_flash(attr, old, new):
        if new:
            render_flash(int(flash_src.data["fid"][new[0]]))

    def on_ophit(attr, old, new):
        render_waveform(new[0] if new else None)

    evt_select.on_change("value", on_event)
    flash_src.selected.on_change("indices", on_flash)
    ophit_src.selected.on_change("indices", on_ophit)

    # ---- layout -----------------------------------------------------------
    curdoc().add_root(column(
        head,
        evt_select,
        Div(text="<b>1. Flashes</b> (select a row or tap the plot)"),
        row(flash_tbl, f_ov),
        Div(text="<b>2. Light geometry</b> (OpDets coloured/sized by PE)"),
        row(f_zy, f_xy, f_xz),
        Div(text="<b>3. OpHits</b> (tap a bar or table row to pick a hit)"),
        row(f_hit, hit_tbl),
        Div(text="<b>4. Waveform</b> (raw ADC left axis, decon right axis)"),
        wav_note, f_wav,
        status,
    ))
    curdoc().title = "PDHD light viewer"

    load_event(events[0][0])


def _pad(lo, hi, frac=0.08):
    if hi <= lo:
        hi = lo + 1.0
    m = (hi - lo) * frac
    return lo - m, hi + m


main(sys.argv)

"""Bokeh event display for PDVD imaging results.

Three linked views, all in the Bee frame (cm):
  1. 2D blob view (transverse Z-Y) at a selectable time slice: each fired U/V/W
     wire drawn as a +/- half-pitch cell, the blob outline (from imaging corners)
     on top, and the Bee sampling points of that slice overlaid.  Prev/Next step
     the slice; hovering a wire shows its channel; tapping it selects that channel
     for the waveform views.
  2. 3D point projections X-Y / Z-Y / X-Z (X = drift) of the Bee points, with
     X/Y/Z window filters; the current slice's points are highlighted.
  3. Waveforms for tapped channels: 1D per-channel + 2D U/V/W-vs-T, lazily read
     from the per-anode Magnify ROOT (degrades gracefully if the ROOT is absent).

Run via the serve_img_viewer.sh launcher:
    bokeh serve --port 5011 img_viewer.py --args <evt.npz> <magnify-template>
where <magnify-template> contains '{anode}', e.g.
    /.../039324_0/magnify-run039324-evt0-anode{anode}.root
Remote viewing:  ssh -L 5011:localhost:5011 user@wcgpu1
                 then open http://localhost:5011/img_viewer
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BoxAnnotation, Button, ColumnDataSource, Div, HoverTool, LinearColorMapper,
    RadioButtonGroup, Select, Spinner, TapTool, TextInput,
)
from bokeh.plotting import figure

TICK_NS = 500.0
PLANE_NAME = {0: "U", 1: "V", 2: "W"}
PLANE_COLOR = {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4"}

# Magnify TH2 frame tags written by run_sp_to_magnify_evt.sh / wct-sp-to-magnify.jsonnet:
#   orig/raw/gauss/wiener are always present; rawdecon/decon only in the -R mode.
MAGNIFY_FRAMES = ["gauss", "wiener", "raw", "orig", "rawdecon", "decon"]


def point_in_poly(y, z, poly):
    """Ray-cast point-in-polygon. poly: ordered (N,2) array of (y,z)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        yi, zi = poly[i]
        yj, zj = poly[j]
        if ((zi > z) != (zj > z)) and \
           (y < (yj - yi) * (z - zi) / (zj - zi + 1e-30) + yi):
            inside = not inside
        j = i
    return inside


# ---- data ------------------------------------------------------------------

class EventData:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.d = {k: d[k] for k in d.files}
        side = os.path.splitext(npz_path)[0] + ".json"
        self.meta = json.load(open(side)) if os.path.isfile(side) else {}
        b = self.d
        # (anode,face) -> sorted unique sliceids
        self.slice_lists = {}
        key = b["blob_anode"].astype(np.int64) * 100 + b["blob_face"]
        for a in range(8):
            for fc in (0, 1):
                m = (b["blob_anode"] == a) & (b["blob_face"] == fc)
                if m.any():
                    self.slice_lists[(a, fc)] = np.unique(b["blob_sliceid"][m])

    def anodes(self):
        return sorted({int(a) for a in self.d["blob_anode"]})

    def faces(self, anode):
        return sorted({fc for (a, fc) in self.slice_lists if a == anode})

    def slices(self, anode, face):
        return self.slice_lists.get((anode, face), np.array([], dtype=np.int64))


# ---- magnify lazy loader ---------------------------------------------------

class MagnifyCache:
    PFX = {0: "hu_", 1: "hv_", 2: "hw_"}

    def __init__(self, template):
        self.template = template            # contains '{anode}' (or empty)
        self._cache = {}                    # (anode,plane,frame) -> (values, ch_offset)

    def path(self, anode):
        return self.template.format(anode=anode) if self.template else ""

    def get(self, anode, plane, frame):
        key = (anode, plane, frame)
        if key in self._cache:
            return self._cache[key]
        path = self.path(anode)
        result = (None, 0, f"Magnify ROOT not found: {path}" if path
                  else "no --magnify-template given")
        if path and os.path.isfile(path):
            try:
                import uproot
                name = f"{self.PFX[plane]}{frame}{anode}"
                with uproot.open(path) as f:
                    keys = {k.split(';')[0] for k in f.keys()}
                    if name in keys:
                        h = f[name]
                        vals = np.asarray(h.values(), dtype=np.float32)
                        off = int(round(h.to_numpy()[1][0]))
                        result = (vals, off, "")
                    else:
                        result = (None, 0, f"{name} not in {os.path.basename(path)}")
            except Exception as e:                       # pragma: no cover
                result = (None, 0, f"error reading {os.path.basename(path)}: {e}")
        self._cache[key] = result
        return result


# ---- app -------------------------------------------------------------------

def main(argv):
    if len(argv) < 2:
        print("usage: img_viewer.py --args <evt.npz> [<magnify-template>]",
              file=sys.stderr)
        sys.exit(1)
    npz_path = argv[1]
    magnify_template = argv[2] if len(argv) > 2 else ""
    ev = EventData(npz_path)
    mag = MagnifyCache(magnify_template)
    b = ev.d

    state = dict(anode=ev.anodes()[0], face=0, sidx=0, channels=[])

    # ---- sources ----------------------------------------------------------
    src_bands = ColumnDataSource(dict(xs=[], ys=[], plane=[], channel=[], color=[]),
                                 name="src_bands")
    src_centers = ColumnDataSource(dict(xs=[], ys=[], color=[]))
    src_blob = ColumnDataSource(dict(xs=[], ys=[]))
    src_samp = ColumnDataSource(dict(z=[], y=[]))
    src_pts_all = ColumnDataSource(dict(x=[], y=[], z=[]))
    src_pts_hi = ColumnDataSource(dict(x=[], y=[], z=[]))
    src_wave = ColumnDataSource(dict(xs=[], ys=[], color=[], label=[]))

    # ---- view 1: blob (Z horizontal, Y vertical) --------------------------
    f_yz = figure(title="2D blob view (transverse)", width=560, height=560,
                  x_axis_label="Z [cm]", y_axis_label="Y [cm]",
                  tools="pan,wheel_zoom,box_zoom,reset,save,tap",
                  active_scroll="wheel_zoom")
    band_r = f_yz.patches("xs", "ys", source=src_bands, fill_color="color",
                          fill_alpha=0.18, line_color="color", line_alpha=0.5)
    f_yz.multi_line("xs", "ys", source=src_centers, line_color="color",
                    line_width=1.0, line_alpha=0.9)
    f_yz.patches("xs", "ys", source=src_blob, fill_alpha=0.0,
                 line_color="black", line_width=2)
    f_yz.scatter("z", "y", source=src_samp, size=4, color="orange",
                 alpha=0.9, marker="circle")
    f_yz.add_tools(HoverTool(renderers=[band_r],
                             tooltips=[("plane", "@plane"), ("channel", "@channel")]))

    # ---- view 2: projections ----------------------------------------------
    def proj(title, xl, yl):
        p = figure(title=title, width=370, height=300, x_axis_label=xl,
                   y_axis_label=yl, tools="pan,wheel_zoom,box_zoom,reset,save",
                   active_scroll="wheel_zoom")
        return p
    f_xy = proj("X-Y", "X [cm]", "Y [cm]")
    f_zy = proj("Z-Y", "Z [cm]", "Y [cm]")
    f_xz = proj("X-Z", "X [cm]", "Z [cm]")
    for p, cx, cy in ((f_xy, "x", "y"), (f_zy, "z", "y"), (f_xz, "x", "z")):
        p.scatter(cx, cy, source=src_pts_all, size=2, color="#888888", alpha=0.35)
        p.scatter(cx, cy, source=src_pts_hi, size=5, color="red", alpha=0.9)

    # ---- view 3: waveforms ------------------------------------------------
    f_wave = figure(title="1D waveforms (selected channels)", width=1120,
                    height=260, x_axis_label="tick", y_axis_label="ADC",
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    active_scroll="wheel_zoom")
    f_wave.multi_line("xs", "ys", source=src_wave, line_color="color",
                      line_width=1.3, legend_field="label")
    f_wave.legend.click_policy = "hide"

    img2d = {}
    img_src = {}
    img_band = {}
    for plane in (0, 1, 2):
        p = figure(title=f"{PLANE_NAME[plane]}-vs-T", width=370, height=300,
                   x_axis_label="channel", y_axis_label="tick",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   active_scroll="wheel_zoom")
        s = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
        cm = LinearColorMapper(palette="Viridis256", nan_color="white")
        p.image(image="image", x="x", y="y", dw="dw", dh="dh", source=s,
                color_mapper=cm)
        ba = BoxAnnotation(bottom=0, top=0, fill_color="red", fill_alpha=0.15,
                           line_color="red")
        ba.visible = False
        p.add_layout(ba)
        img2d[plane] = p
        img_src[plane] = s
        img_band[plane] = ba

    # ---- widgets ----------------------------------------------------------
    w_anode = Select(title="Anode", value=str(state["anode"]),
                     options=[str(a) for a in ev.anodes()], width=90, name="anode")
    w_face = RadioButtonGroup(labels=["face0", "face1"], active=0, width=140)
    b_prev = Button(label="◀ Prev slice", width=110)
    b_next = Button(label="Next slice ▶", width=110)
    w_slice = Spinner(title="slice idx", low=0, high=0, step=1, value=0, width=90)
    slice_div = Div(text="", width=560, name="slice_div")
    status = Div(text="", width=1120)

    win = {}
    for c in "xyz":
        win[c + "lo"] = Spinner(title=f"{c} min", value=0.0, step=5.0, width=95)
        win[c + "hi"] = Spinner(title=f"{c} max", value=0.0, step=5.0, width=95)
    b_apply = Button(label="Apply window", width=110, button_type="primary")
    b_reset = Button(label="Reset window", width=110)

    w_frame = Select(title="Magnify frame", value="gauss",
                     options=MAGNIFY_FRAMES, width=110)
    w_man_ch = TextInput(title="add channel", value="", width=110)
    w_man_pl = RadioButtonGroup(labels=["U", "V", "W"], active=2, width=120)
    b_add = Button(label="Add", width=60)
    b_clear = Button(label="Clear channels", width=120)
    chips = Div(text="<i>no channels selected</i>", width=1120, name="chips")

    # ---- render: view 1 + linked highlight --------------------------------
    def cur_slice_blobs():
        a, fc = state["anode"], state["face"]
        sl = ev.slices(a, fc)
        if sl.size == 0:
            return np.array([], dtype=np.int64), None
        state["sidx"] = int(np.clip(state["sidx"], 0, sl.size - 1))
        sid = int(sl[state["sidx"]])
        m = ((b["blob_anode"] == a) & (b["blob_face"] == fc)
             & (b["blob_sliceid"] == sid))
        return np.where(m)[0], sid

    def render_slice():
        a, fc = state["anode"], state["face"]
        sl = ev.slices(a, fc)
        blob_idx, sid = cur_slice_blobs()
        if sid is None:
            src_bands.data = dict(xs=[], ys=[], plane=[], channel=[], color=[])
            src_centers.data = dict(xs=[], ys=[], color=[])
            src_blob.data = dict(xs=[], ys=[])
            src_samp.data = dict(z=[], y=[])
            slice_div.text = f"<b>anode {a} face {fc}</b>: no blobs"
            return
        # bands of these blobs
        bsel = np.zeros(b["blob_anode"].size, dtype=bool)
        bsel[blob_idx] = True
        bmask = bsel[b["band_blob"]]
        bi = np.where(bmask)[0]
        quads = b["band_quad_yz"][bi]                  # (n,4,2) -> (y,z)
        planes = b["band_plane"][bi]
        colors = [PLANE_COLOR[int(p)] for p in planes]
        src_bands.data = dict(
            xs=[list(q[:, 1]) for q in quads],         # z horizontal
            ys=[list(q[:, 0]) for q in quads],         # y vertical
            plane=[PLANE_NAME[int(p)] for p in planes],
            channel=[int(c) for c in b["band_channel"][bi]],
            color=colors,
        )
        # wire center line = midline of the two +/- half-pitch edges
        # quad order = [tail-off, head-off, head+off, tail+off]
        tmid = 0.5 * (quads[:, 0] + quads[:, 3])       # tail center (y,z)
        hmid = 0.5 * (quads[:, 1] + quads[:, 2])       # head center (y,z)
        src_centers.data = dict(
            xs=[[float(tmid[i, 1]), float(hmid[i, 1])] for i in range(len(quads))],
            ys=[[float(tmid[i, 0]), float(hmid[i, 0])] for i in range(len(quads))],
            color=colors,
        )
        # blob outlines
        bxs, bys = [], []
        for k in blob_idx:
            poly = b["blob_poly_xy"][b["blob_poly_off"][k]:b["blob_poly_off"][k + 1]]
            bxs.append(list(poly[:, 1])); bys.append(list(poly[:, 0]))
        src_blob.data = dict(xs=bxs, ys=bys)
        # sampling points OF the displayed blobs: drift-side + slice x-window +
        # inside one of the displayed blob polygons (the same test Gate 1 uses).
        # This is shared by the blob view and the projection highlight so the two
        # views agree on "the data being viewed".
        xlo = float(b["blob_x_lo"][blob_idx].min())
        xhi = float(b["blob_x_hi"][blob_idx].max())
        g = 0 if a <= 3 else 1
        cand = np.where((b["pts_group"] == g)
                        & (b["pts_x"] >= xlo) & (b["pts_x"] <= xhi))[0]
        polys = [b["blob_poly_xy"][b["blob_poly_off"][k]:b["blob_poly_off"][k + 1]]
                 for k in blob_idx]
        sel = []
        for pi in cand:
            yy, zz = b["pts_y"][pi], b["pts_z"][pi]
            if any(point_in_poly(yy, zz, poly) for poly in polys):
                sel.append(pi)
        sel = np.asarray(sel, dtype=np.int64)
        src_samp.data = dict(z=list(b["pts_z"][sel]), y=list(b["pts_y"][sel]))
        # projection highlight = the same blob points, intersected with the window
        hi = sel[_window_mask()[sel]] if sel.size else sel
        src_pts_hi.data = dict(x=list(b["pts_x"][hi]), y=list(b["pts_y"][hi]),
                               z=list(b["pts_z"][hi]))
        # 2D waveform tick band
        t0 = b["blob_start_ns"][blob_idx][0] / TICK_NS
        t1 = (b["blob_start_ns"][blob_idx][0]
              + b["blob_span_ns"][blob_idx][0]) / TICK_NS
        for plane in (0, 1, 2):
            img_band[plane].bottom = float(min(t0, t1))
            img_band[plane].top = float(max(t0, t1))
            img_band[plane].visible = True
        slice_div.text = (f"<b>anode {a} face {fc}</b> &mdash; slice "
                          f"{state['sidx'] + 1}/{sl.size} (id={sid}), "
                          f"x&isin;[{xlo:.1f}, {xhi:.1f}] cm, {blob_idx.size} blobs")

    # ---- view 2 window ----------------------------------------------------
    def _window_mask():
        m = np.ones(b["pts_x"].size, dtype=bool)
        for c in "xyz":
            m &= (b["pts_" + c] >= win[c + "lo"].value)
            m &= (b["pts_" + c] <= win[c + "hi"].value)
        return m

    def apply_window():
        m = _window_mask()
        src_pts_all.data = dict(x=list(b["pts_x"][m]), y=list(b["pts_y"][m]),
                                z=list(b["pts_z"][m]))
        render_slice()   # refresh highlight under the new window

    def reset_window():
        for c in "xyz":
            arr = b["pts_" + c]
            win[c + "lo"].value = float(np.floor(arr.min()))
            win[c + "hi"].value = float(np.ceil(arr.max()))
        apply_window()

    # ---- view 3 waveforms -------------------------------------------------
    def render_waves():
        a = state["anode"]
        frame = w_frame.value
        xs, ys, color, label = [], [], [], []
        msgs = []
        for plane, ch in state["channels"]:
            vals, off, err = mag.get(a, plane, frame)
            if vals is None:
                if err:
                    msgs.append(err)
                continue
            local = ch - off
            if 0 <= local < vals.shape[0]:
                wf = vals[local]
                xs.append(list(range(len(wf)))); ys.append([float(v) for v in wf])
                color.append(PLANE_COLOR[plane])
                label.append(f"{PLANE_NAME[plane]}:{ch}")
            else:
                msgs.append(f"{PLANE_NAME[plane]}:{ch} outside ROOT range "
                            f"[{off},{off + vals.shape[0]})")
        src_wave.data = dict(xs=xs, ys=ys, color=color, label=label)
        render_images(frame)
        if msgs:
            status.text = "<span style='color:#c00'>" + " | ".join(
                dict.fromkeys(msgs)) + "</span>"
        else:
            status.text = ""

    def render_images(frame):
        a = state["anode"]
        by_plane = {0: [], 1: [], 2: []}
        for plane, ch in state["channels"]:
            by_plane[plane].append(ch)
        for plane in (0, 1, 2):
            s = img_src[plane]
            vals, off, err = mag.get(a, plane, frame)
            chs = by_plane[plane]
            if vals is None or not chs:
                s.data = dict(image=[], x=[], y=[], dw=[], dh=[])
                continue
            locs = [c - off for c in chs if 0 <= c - off < vals.shape[0]]
            if not locs:
                s.data = dict(image=[], x=[], y=[], dw=[], dh=[])
                continue
            c0 = max(min(locs) - 20, 0)
            c1 = min(max(locs) + 20, vals.shape[0])
            sub = vals[c0:c1, :].T                      # (ntick, nchanwin)
            s.data = dict(image=[sub], x=[off + c0], y=[0],
                          dw=[c1 - c0], dh=[vals.shape[1]])

    def refresh_chips():
        if not state["channels"]:
            chips.text = "<i>no channels selected</i> (tap a wire in view 1)"
            return
        parts = [f"<span style='background:{PLANE_COLOR[p]};color:white;"
                 f"padding:1px 5px;margin:2px;border-radius:3px'>"
                 f"{PLANE_NAME[p]}:{c}</span>" for p, c in state["channels"]]
        chips.text = "selected: " + " ".join(parts)

    def add_channel(plane, ch):
        if (plane, ch) not in state["channels"]:
            state["channels"].append((plane, ch))
            if len(state["channels"]) > 6:
                state["channels"] = state["channels"][-6:]
        refresh_chips()
        render_waves()

    # ---- callbacks --------------------------------------------------------
    def on_tap(attr, old, new):
        if not new:
            return
        i = new[0]
        data = src_bands.data
        pl = {v: k for k, v in PLANE_NAME.items()}[data["plane"][i]]
        add_channel(pl, int(data["channel"][i]))
        src_bands.selected.indices = []

    def set_slice_spinner():
        n = ev.slices(state["anode"], state["face"]).size
        w_slice.high = max(n - 1, 0)
        w_slice.value = state["sidx"]

    def on_anode(attr, old, new):
        state["anode"] = int(new)
        state["face"] = w_face.active if w_face.active in ev.faces(int(new)) else \
            (ev.faces(int(new))[0] if ev.faces(int(new)) else 0)
        w_face.active = state["face"]
        state["sidx"] = 0
        set_slice_spinner()
        render_slice()
        render_waves()

    def on_face(attr, old, new):
        state["face"] = int(new)
        state["sidx"] = 0
        set_slice_spinner()
        render_slice()

    def step(delta):
        state["sidx"] += delta
        sl = ev.slices(state["anode"], state["face"])
        state["sidx"] = int(np.clip(state["sidx"], 0, max(sl.size - 1, 0)))
        w_slice.value = state["sidx"]
        render_slice()

    def on_slice_spin(attr, old, new):
        state["sidx"] = int(new)
        render_slice()

    src_bands.selected.on_change("indices", on_tap)
    w_anode.on_change("value", on_anode)
    w_face.on_change("active", on_face)
    b_prev.on_click(lambda: step(-1))
    b_next.on_click(lambda: step(+1))
    w_slice.on_change("value", on_slice_spin)
    b_apply.on_click(apply_window)
    b_reset.on_click(reset_window)
    w_frame.on_change("value", lambda a, o, n: render_waves())
    b_clear.on_click(lambda: (state.update(channels=[]), refresh_chips(),
                              render_waves()))

    def on_add():
        try:
            ch = int(w_man_ch.value)
        except ValueError:
            return
        add_channel(int(w_man_pl.active), ch)
    b_add.on_click(on_add)

    # ---- layout -----------------------------------------------------------
    head = Div(text=f"<h2>PDVD imaging viewer &mdash; run {ev.meta.get('run')} "
                    f"evt {ev.meta.get('event')}</h2>"
                    f"<small>{npz_path}</small>", width=1120)
    controls = row(w_anode, w_face, b_prev, b_next, w_slice)
    winrow = row(win["xlo"], win["xhi"], win["ylo"], win["yhi"],
                 win["zlo"], win["zhi"], b_apply, b_reset)
    chrow = row(w_frame, w_man_pl, w_man_ch, b_add, b_clear)
    layout = column(
        head,
        controls, slice_div,
        row(f_yz, column(Div(text="<b>3D point projections</b>"),
                         row(f_xy, f_zy, f_xz), winrow)),
        Div(text="<hr><b>Waveforms</b> (tap a wire above to pick a channel)"),
        chrow, chips, f_wave,
        row(img2d[0], img2d[1], img2d[2]),
        status,
    )
    curdoc().add_root(layout)
    curdoc().title = "PDVD imaging viewer"

    # ---- initial render ---------------------------------------------------
    set_slice_spinner()
    reset_window()      # also calls apply_window -> render_slice
    render_slice()
    refresh_chips()


main(sys.argv)

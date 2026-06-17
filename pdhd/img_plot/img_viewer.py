"""Bokeh event display for PDHD imaging results.

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
    bokeh serve --port 5013 img_viewer.py --args <evt.npz> <magnify-template>
where <magnify-template> contains '{anode}', e.g.
    /.../027409_0/magnify-run027409-evt0-apa{anode}-dnnroi.root
Remote viewing:  ssh -L 5013:localhost:5013 user@wcgpu1
                 then open http://localhost:5013/img_viewer
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BoxAnnotation, Button, Checkbox, ColumnDataSource, Div, HoverTool,
    LinearColorMapper, RadioButtonGroup, Range1d, Select, Spinner, TapTool,
    TextInput,
)
from bokeh.plotting import figure

TICK_NS = 500.0
BLOB_PAD = 20.0   # cm padding around the displayed blob in the 2D view
# Stepped sampling points sit exactly at x = time2drift(slice start), one x per
# slice, 0.315 cm apart (4 ticks x 500 ns x 1.576 mm/us).  Match a point to a
# slice by |pts_x - x_start| < half that spacing (an exact-boundary window test
# drops them on float noise).
PTS_X_HALF = 0.1576
# A sliver blob's center-fallback point can land just outside the drawn
# polygon (the sampler's re-derived ray-grid corners differ slightly from the
# file's corners for degenerate slivers; observed offsets up to ~0.42 cm), so
# also accept points within this distance of a displayed blob's edge.
PTS_EDGE_TOL = 0.5
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


def dist_to_poly(y, z, poly):
    """Min distance from (y,z) to the polygon's edges. poly: (N,2) of (y,z)."""
    p = np.asarray(poly)
    a = p
    b = np.roll(p, -1, axis=0)
    d = b - a
    t = np.clip(((y - a[:, 0]) * d[:, 0] + (z - a[:, 1]) * d[:, 1])
                / (d[:, 0] ** 2 + d[:, 1] ** 2 + 1e-30), 0.0, 1.0)
    cy = a[:, 0] + t * d[:, 0]
    cz = a[:, 1] + t * d[:, 1]
    return float(np.min(np.hypot(y - cy, z - cz)))


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
        for a in range(4):
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
        self._bad = {}                      # anode -> {plane: set(global chid)}

    def path(self, anode):
        # use replace, not str.format -- a stray brace in the template must not
        # raise (it would crash the tap callback); a wrong path degrades to "not found".
        return self.template.replace("{anode}", str(anode)) if self.template else ""

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

    def bad(self, anode):
        """{plane: set(global chid)} from the Magnify T_bad<anode> dead-channel TTree
        (the authoritative dead list shipped in the ROOT).  Empty if absent."""
        if anode in self._bad:
            return self._bad[anode]
        out = {0: set(), 1: set(), 2: set()}
        path = self.path(anode)
        if path and os.path.isfile(path):
            try:
                import uproot
                with uproot.open(path) as f:
                    name = next((k.split(';')[0] for k in f.keys()
                                 if k.split(';')[0] == f"T_bad{anode}"), None)
                    if name:
                        # T_bad also carries start_time/end_time; here every entry
                        # spans the full readout (0..nticks) so we treat a listed
                        # chid as dead for all ticks.  Revisit if partial-bad lists appear.
                        arr = f[name].arrays(library="np")
                        for ch, pl in zip(arr["chid"], arr["plane"]):
                            if int(pl) in out:
                                out[int(pl)].add(int(ch))
            except Exception:                             # pragma: no cover
                pass
        self._bad[anode] = out
        return out


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

    state = dict(anode=ev.anodes()[0], face=0, sidx=0, channels=[],
                 pos_window=None,   # None or dict(x,y,z,pad): lock views to a region
                 slice_chan={},     # {plane: (cmin,cmax)} of the current slice's wires
                 render_lock=False) # suppress renders while programmatically retargeting

    # ---- sources ----------------------------------------------------------
    src_bands = ColumnDataSource(
        dict(xs=[], ys=[], plane=[], wip=[], channel=[], color=[],
             fill=[], falpha=[], status=[]),
        name="src_bands")
    src_centers = ColumnDataSource(dict(xs=[], ys=[], color=[]))
    src_blob = ColumnDataSource(dict(xs=[], ys=[], val=[]))
    src_ghost = ColumnDataSource(dict(xs=[], ys=[], val=[]))
    src_samp = ColumnDataSource(dict(z=[], y=[]))
    src_pts_all = ColumnDataSource(dict(x=[], y=[], z=[]))
    src_pts_hi = ColumnDataSource(dict(x=[], y=[], z=[]))
    src_wave = ColumnDataSource(dict(xs=[], ys=[], color=[], label=[]))

    # ---- view 1: blob (Z horizontal, Y vertical) --------------------------
    f_yz = figure(title="2D blob view (transverse)", width=560, height=560,
                  x_axis_label="Z [cm]", y_axis_label="Y [cm]",
                  tools="pan,wheel_zoom,box_zoom,reset,save,tap",
                  active_scroll="wheel_zoom",
                  x_range=Range1d(0, 300), y_range=Range1d(-350, 350))
    band_r = f_yz.patches("xs", "ys", source=src_bands, fill_color="fill",
                          fill_alpha="falpha", line_color="color", line_alpha=0.5)
    f_yz.multi_line("xs", "ys", source=src_centers, line_color="color",
                    line_width=1.0, line_alpha=0.9)
    blob_r = f_yz.patches("xs", "ys", source=src_blob, fill_alpha=0.0,
                          line_color="black", line_width=2)
    ghost_r = f_yz.patches("xs", "ys", source=src_ghost, fill_alpha=0.0,
                           line_color="magenta", line_width=2,
                           line_dash="dashed")
    f_yz.scatter("z", "y", source=src_samp, size=4, color="orange",
                 alpha=0.9, marker="circle")
    f_yz.add_tools(HoverTool(renderers=[band_r],
                             tooltips=[("plane", "@plane"), ("wire", "@wip"),
                                       ("channel", "@channel"),
                                       ("status", "@status")]))
    f_yz.add_tools(HoverTool(renderers=[blob_r, ghost_r],
                             tooltips=[("blob charge", "@val{0}")]))

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
                    active_scroll="wheel_zoom", x_range=Range1d(0, 6400),
                    y_range=Range1d(-100, 100))
    f_wave.multi_line("xs", "ys", source=src_wave, line_color="color",
                      line_width=1.3, legend_field="label")
    f_wave.legend.click_policy = "hide"

    img2d = {}
    img_src = {}
    img_band = {}
    img_dead = {}
    for plane in (0, 1, 2):
        p = figure(title=f"{PLANE_NAME[plane]}-vs-T", width=370, height=300,
                   x_axis_label="channel", y_axis_label="tick",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   active_scroll="wheel_zoom",
                   x_range=Range1d(0, 1), y_range=Range1d(0, 6400))
        s = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
        cm = LinearColorMapper(palette="Viridis256", nan_color="white")
        p.image(image="image", x="x", y="y", dw="dw", dh="dh", source=s,
                color_mapper=cm)
        # dead channels (Magnify T_bad) = full-height grey bars on top of the image
        sd = ColumnDataSource(dict(x=[], top=[], ch=[]))
        dr = p.vbar(x="x", top="top", bottom=0, width=1.0, source=sd,
                    fill_color="#888888", fill_alpha=0.45, line_alpha=0.0)
        p.add_tools(HoverTool(renderers=[dr],
                              tooltips=[("dead channel", "@ch")]))
        ba = BoxAnnotation(bottom=0, top=0, fill_color="red", fill_alpha=0.15,
                           line_color="red")
        ba.visible = False
        p.add_layout(ba)
        img2d[plane] = p
        img_src[plane] = s
        img_band[plane] = ba
        img_dead[plane] = sd

    # ---- widgets ----------------------------------------------------------
    w_anode = Select(title="Anode", value=str(state["anode"]),
                     options=[str(a) for a in ev.anodes()], width=90, name="anode")
    w_face = RadioButtonGroup(labels=["face0", "face1"], active=0, width=140)
    b_prev = Button(label="◀ Prev slice", width=110)
    b_next = Button(label="Next slice ▶", width=110)
    w_slice = Spinner(title="slice idx", low=0, high=0, step=1, value=0, width=90)
    w_ghosts = Checkbox(label="hide zero-charge blobs", active=False)
    slice_div = Div(text="", width=560, name="slice_div")
    chan_div = Div(text="", width=560, name="chan_div")
    status = Div(text="", width=1120)

    win = {}
    for c in "xyz":
        win[c + "lo"] = Spinner(title=f"{c} min", value=0.0, step=5.0, width=95)
        win[c + "hi"] = Spinner(title=f"{c} max", value=0.0, step=5.0, width=95)
    b_apply = Button(label="Apply window", width=110, button_type="primary")
    b_reset = Button(label="Reset window", width=110)

    # position -> window: type an X/Y/Z (cm), fill the window to position +/- pad
    w_pos_x = TextInput(title="pos X", value="", width=80)
    w_pos_y = TextInput(title="pos Y", value="", width=80)
    w_pos_z = TextInput(title="pos Z", value="", width=80)
    w_pos_pad = Spinner(title="± cm", value=20.0, step=5.0, low=1.0, width=80)
    b_center = Button(label="Center window on pos", width=160, button_type="primary")
    b_pos_here = Button(label="pos ← current slice", width=150)

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
        if state["render_lock"]:
            return
        a, fc = state["anode"], state["face"]
        sl = ev.slices(a, fc)
        blob_idx, sid = cur_slice_blobs()
        if sid is None:
            src_bands.data = dict(xs=[], ys=[], plane=[], wip=[], channel=[],
                                  color=[], fill=[], falpha=[], status=[])
            src_centers.data = dict(xs=[], ys=[], color=[])
            src_blob.data = dict(xs=[], ys=[], val=[])
            src_ghost.data = dict(xs=[], ys=[], val=[])
            src_samp.data = dict(z=[], y=[])
            state["slice_chan"] = {}
            slice_div.text = f"<b>anode {a} face {fc}</b>: no blobs"
            chan_div.text = ""
            render_waves()
            return
        # zero-charge ghosts: blobs the charge solver / deghosting zeroed but
        # left in the cluster file.  Optionally hidden; always counted.
        vals = b["blob_val"][blob_idx]
        nzero = int((vals == 0).sum())
        vis_idx = blob_idx[vals > 0] if w_ghosts.active else blob_idx
        # bands of the visible blobs
        bsel = np.zeros(b["blob_anode"].size, dtype=bool)
        bsel[vis_idx] = True
        bmask = bsel[b["band_blob"]]
        bi = np.where(bmask)[0]
        # a wire shared by several blobs is drawn once, else the overlapping
        # fills stack alpha and shared wires look darker than the rest
        if bi.size:
            keys = np.stack([b["band_plane"][bi], b["band_wip"][bi]])
            _, first = np.unique(keys, axis=1, return_index=True)
            bi = bi[np.sort(first)]
        quads = b["band_quad_yz"][bi]                  # (n,4,2) -> (y,z)
        planes = b["band_plane"][bi]
        colors = [PLANE_COLOR[int(p)] for p in planes]
        # dead channels (Magnify T_bad) drawn grey: a dead wire carries no
        # signal, so blobs split around it in the active-plane tiling passes
        # while a 2-view (masked-plane) pass re-covers it as its own blob.
        bad = mag.bad(a)
        chans = [int(c) for c in b["band_channel"][bi]]
        isdead = [c in bad.get(int(p), set()) for p, c in zip(planes, chans)]
        src_bands.data = dict(
            xs=[list(q[:, 1]) for q in quads],         # z horizontal
            ys=[list(q[:, 0]) for q in quads],         # y vertical
            plane=[PLANE_NAME[int(p)] for p in planes],
            wip=[int(w) for w in b["band_wip"][bi]],
            channel=chans,
            color=colors,
            fill=["#888888" if d else c for d, c in zip(isdead, colors)],
            falpha=[0.45 if d else 0.18 for d in isdead],
            status=["DEAD" if d else "live" for d in isdead],
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
        # blob outlines: charged solid black, zero-charge dashed magenta
        bxs, bys = [], []
        cd = dict(xs=[], ys=[], val=[])
        gd = dict(xs=[], ys=[], val=[])
        for k in vis_idx:
            poly = b["blob_poly_xy"][b["blob_poly_off"][k]:b["blob_poly_off"][k + 1]]
            v = float(b["blob_val"][k])
            dst = cd if v > 0 else gd
            dst["xs"].append(list(poly[:, 1]))
            dst["ys"].append(list(poly[:, 0]))
            dst["val"].append(v)
            bxs.append(list(poly[:, 1])); bys.append(list(poly[:, 0]))
        src_blob.data = cd
        src_ghost.data = gd
        # per-plane channel span of this slice's wires (drives the 2D ch-vs-T view
        # when a position is locked, and is the Y-Z-local channel set there).
        pl_arr = b["band_plane"][bi]
        ch_arr = b["band_channel"][bi]
        state["slice_chan"] = {pl: (int(ch_arr[pl_arr == pl].min()),
                                    int(ch_arr[pl_arr == pl].max()))
                               for pl in (0, 1, 2) if (pl_arr == pl).any()}
        chan_div.text = "fired-wire channels: " + " &nbsp; ".join(
            f"<span style='color:{PLANE_COLOR[pl]}'><b>{PLANE_NAME[pl]}</b>: "
            f"{lo}&ndash;{hi} ({hi - lo + 1} ch)</span>"
            for pl, (lo, hi) in sorted(state["slice_chan"].items())) \
            if state["slice_chan"] else "fired-wire channels: <i>none</i>"
        # blob view range: locked to the position window if set (Y-Z range = window,
        # X already centered on posX by the slice jump), else auto-zoom to the blob.
        pw = state["pos_window"]
        zz = [v for sub in bxs for v in sub]
        yy = [v for sub in bys for v in sub]
        if pw:
            f_yz.x_range.start = pw["z"] - pw["pad"]
            f_yz.x_range.end = pw["z"] + pw["pad"]
            f_yz.y_range.start = pw["y"] - pw["pad"]
            f_yz.y_range.end = pw["y"] + pw["pad"]
        elif zz:
            f_yz.x_range.start = min(zz) - BLOB_PAD
            f_yz.x_range.end = max(zz) + BLOB_PAD
            f_yz.y_range.start = min(yy) - BLOB_PAD
            f_yz.y_range.end = max(yy) + BLOB_PAD
        # sampling points OF the displayed blobs: drift-side + same slice +
        # inside one of the displayed blob polygons.  A slice's stepped points
        # all sit exactly at x = time2drift(slice start), so match on that x
        # within half the 0.313 cm point spacing -- an exact [x_lo,x_hi] window
        # test drops boundary points on ~1e-14 float noise.
        xlo = float(b["blob_x_lo"][blob_idx].min())
        xhi = float(b["blob_x_hi"][blob_idx].max())
        xstart = float(b["blob_x_start"][blob_idx[0]])
        g = 0 if a in (0, 2) else 1
        cand = np.where((b["pts_group"] == g)
                        & (np.abs(b["pts_x"] - xstart) < PTS_X_HALF))[0]
        polys = [b["blob_poly_xy"][b["blob_poly_off"][k]:b["blob_poly_off"][k + 1]]
                 for k in vis_idx]
        sel = []
        for pi in cand:
            yy, zz = b["pts_y"][pi], b["pts_z"][pi]
            if any(point_in_poly(yy, zz, poly)
                   or dist_to_poly(yy, zz, poly) < PTS_EDGE_TOL
                   for poly in polys):
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
        tklo, tkhi = min(t0, t1), max(t0, t1)
        ghost_note = (f" ({nzero} zero-charge"
                      + (", hidden" if w_ghosts.active and nzero else "") + ")"
                      if nzero else "")
        slice_div.text = (f"<b>anode {a} face {fc}</b> &mdash; slice "
                          f"{state['sidx'] + 1}/{sl.size} (id={sid}), "
                          f"x&isin;[{xlo:.1f}, {xhi:.1f}] cm, "
                          f"tick&isin;[{tklo:.0f}, {tkhi:.0f}], "
                          f"{blob_idx.size} blobs{ghost_note}")
        render_waves()   # keep the 1D + 2D-vs-T panels in sync with the slice

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
        state["pos_window"] = None        # release the position lock
        for c in "xyz":
            arr = b["pts_" + c]
            win[c + "lo"].value = float(np.floor(arr.min()))
            win[c + "hi"].value = float(np.ceil(arr.max()))
        apply_window()

    def xconv_of(anode, face):
        """(xorig_cm, xsign) of the MABC time2drift convention for (anode,face)."""
        xc = ev.meta.get("xconv", {}).get(f"{anode}_{face}")
        if xc:
            return float(xc["xorig_cm"]), int(xc["xsign"])
        return (-353.20, 1) if anode in (0, 2) else (353.00, -1)  # drift-side fallback

    def x_to_tick(anode, face, x_cm):
        """Invert the toolkit time2drift convention: drift x (cm) -> readout tick.

        time2drift: x_cm = xorig + xsign*(t_ns + toff)*drift*0.1  (0.1 == 1/units.cm)
        """
        drift = ev.meta.get("drift_mm_per_ns", 0.0016)
        # Fallback -250us covers npz caches preprocessed before the chain
        # zeroed its preset T0; new caches carry time_offset_ns = 0 in meta.
        toff = ev.meta.get("time_offset_ns", -250000.0)
        xorig, xsign = xconv_of(anode, face)
        t_ns = (x_cm - xorig) / (xsign * drift * 0.1) - toff
        return t_ns / TICK_NS

    def pos_tick_range():
        """Tick window for the current anode/face from the locked X +/- pad, or None."""
        pw = state["pos_window"]
        if not pw:
            return None
        a, fc = state["anode"], state["face"]
        t1 = x_to_tick(a, fc, pw["x"] - pw["pad"])
        t2 = x_to_tick(a, fc, pw["x"] + pw["pad"])
        lo, hi = sorted((t1, t2))
        return max(0.0, lo), hi

    def nearest_slice_idx(anode, face, posx):
        """Index (into the (anode,face) slice list) of the slice nearest drift posx."""
        sl = ev.slices(anode, face)
        if sl.size == 0:
            return 0
        idxs = np.where((b["blob_anode"] == anode) & (b["blob_face"] == face))[0]
        j = idxs[np.argmin(np.abs(b["blob_xc"][idxs] - posx))]
        w = np.where(sl == int(b["blob_sliceid"][j]))[0]
        return int(w[0]) if w.size else 0

    def blob_at_pos(posx, posy, posz):
        """(anode,face,sidx) of the blob whose drift-x window contains posx and whose
        Y-Z polygon contains (posy,posz); the x-closest such blob.  None if no match."""
        cand = np.where((b["blob_x_lo"] <= posx) & (posx <= b["blob_x_hi"]))[0]
        if cand.size == 0:
            return None
        cand = cand[np.argsort(np.abs(b["blob_xc"][cand] - posx))]
        for k in cand:
            poly = b["blob_poly_xy"][b["blob_poly_off"][k]:b["blob_poly_off"][k + 1]]
            if point_in_poly(posy, posz, poly):
                a = int(b["blob_anode"][k]); fc = int(b["blob_face"][k])
                sid = int(b["blob_sliceid"][k])
                sl = ev.slices(a, fc)
                w = np.where(sl == sid)[0]
                return a, fc, int(w[0]) if w.size else 0
        return None

    def center_on_pos():
        """Lock all views to (X,Y,Z) +/- pad: auto-select the anode/face/slice the
        point falls in, filter the projections, zoom the blob Y-Z view, and set the
        waveform tick range."""
        pad = float(w_pos_pad.value)
        try:
            pos = {c: float(w.value) for c, w in
                   (("x", w_pos_x), ("y", w_pos_y), ("z", w_pos_z))}
        except ValueError:
            status.text = ("<span style='color:#c00'>position X/Y/Z must be "
                           "numbers</span>")
            return
        state["pos_window"] = dict(x=pos["x"], y=pos["y"], z=pos["z"], pad=pad)
        for c in "xyz":
            win[c + "lo"].value = pos[c] - pad
            win[c + "hi"].value = pos[c] + pad
        # auto-detect the anode/face/slice that contains the point
        hit = blob_at_pos(pos["x"], pos["y"], pos["z"])
        state["render_lock"] = True
        if hit:
            ta, tfc, tsidx = hit
            state["anode"], state["face"], state["sidx"] = ta, tfc, tsidx
            w_anode.value = str(ta)        # callbacks fire but render is locked
            w_face.active = tfc
            set_slice_spinner()
            state["sidx"] = tsidx
            w_slice.value = tsidx
            status.text = (f"<span style='color:#070'>position in anode {ta} "
                           f"face {tfc}, slice idx {tsidx}</span>")
        else:
            state["sidx"] = nearest_slice_idx(state["anode"], state["face"], pos["x"])
            w_slice.value = state["sidx"]
            status.text = ("<span style='color:#c60'>no blob contains that point; "
                           "kept anode "
                           f"{state['anode']} face {state['face']}</span>")
        state["render_lock"] = False
        apply_window()   # filters projections + render_slice (-> blob view + waves)

    def pos_from_slice():
        """Fill the position boxes from the centroid of the displayed blobs."""
        blob_idx, sid = cur_slice_blobs()
        if sid is None or blob_idx.size == 0:
            return
        pts = np.concatenate([
            b["blob_poly_xy"][b["blob_poly_off"][k]:b["blob_poly_off"][k + 1]]
            for k in blob_idx])
        w_pos_y.value = f"{pts[:, 0].mean():.1f}"
        w_pos_z.value = f"{pts[:, 1].mean():.1f}"
        w_pos_x.value = f"{b['blob_xc'][blob_idx].mean():.1f}"

    # ---- view 3 waveforms -------------------------------------------------
    def plane_chan_window(plane, off, nchan):
        """Local-index channel window [c0,c1) shown for a plane: the locked Y-Z
        slice channels +/-15, else the tapped channels' neighborhood +/-20.  None
        if nothing to show."""
        if state["pos_window"]:
            rng = state["slice_chan"].get(plane)
            if not rng:
                return None
            return max(rng[0] - off - 15, 0), min(rng[1] - off + 15, nchan)
        locs = [c - off for p, c in state["channels"]
                if p == plane and 0 <= c - off < nchan]
        if not locs:
            return None
        return max(min(locs) - 20, 0), min(max(locs) + 20, nchan)

    def dead_channels(plane, off, c0, c1):
        """Global channels in window [c0,c1) listed dead in the Magnify T_bad tree."""
        bad = mag.bad(state["anode"]).get(plane, set())
        return [ch for ch in range(off + c0, off + c1) if ch in bad]

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
        # dead channels (Magnify T_bad tree) within the displayed channel window:
        # draw their ACTUAL waveform in grey, so a channel labelled dead that still
        # carries a real pulse (a mis-label) is visible against the flat truly-dead
        # ones.  Capped to keep the legend readable.
        ndead = 0
        sel = set(state["channels"])
        for plane in (0, 1, 2):
            vals, off, _ = mag.get(a, plane, frame)
            if vals is None:
                continue
            win = plane_chan_window(plane, off, vals.shape[0])
            if not win:
                continue
            for ch in dead_channels(plane, off, *win):
                if (plane, ch) in sel or ndead >= 24:
                    continue
                local = ch - off
                if not (0 <= local < vals.shape[0]):
                    continue
                wf = vals[local]
                xs.append(list(range(len(wf)))); ys.append([float(v) for v in wf])
                color.append("#999999"); label.append(f"dead {PLANE_NAME[plane]}:{ch}")
                ndead += 1
        src_wave.data = dict(xs=xs, ys=ys, color=color, label=label)
        # 1D waveform tick (x) range = the locked X window, else the full readout
        tr = pos_tick_range()
        if tr:
            f_wave.x_range.start, f_wave.x_range.end = tr
        else:
            f_wave.x_range.start, f_wave.x_range.end = 0, (len(ys[0]) if ys else 6400)
        # y range follows the signals inside the displayed tick window
        if ys:
            i0 = max(int(f_wave.x_range.start), 0)
            i1 = int(np.ceil(f_wave.x_range.end))
            lo = min(min(w[i0:i1] or w) for w in ys)
            hi = max(max(w[i0:i1] or w) for w in ys)
            pad = 0.08 * (hi - lo) or 1.0
            f_wave.y_range.start, f_wave.y_range.end = lo - pad, hi + pad
        render_images(frame)
        if msgs:
            status.text = "<span style='color:#c00'>" + " | ".join(
                dict.fromkeys(msgs)) + "</span>"
        else:
            status.text = ""

    def render_images(frame):
        """2D ch-vs-T panels.  With a locked position: channel (x) axis = the Y-Z-
        local wires of the current slice, tick (y) axis = the locked X window.  Else:
        the tapped channels' neighborhood with auto ranges."""
        a = state["anode"]
        pw = state["pos_window"]
        tr = pos_tick_range()
        for plane in (0, 1, 2):
            s = img_src[plane]
            p = img2d[plane]
            vals, off, err = mag.get(a, plane, frame)
            if vals is None:
                s.data = dict(image=[], x=[], y=[], dw=[], dh=[])
                img_dead[plane].data = dict(x=[], top=[], ch=[])
                continue
            win = plane_chan_window(plane, off, vals.shape[0])
            if not win:
                s.data = dict(image=[], x=[], y=[], dw=[], dh=[])
                img_dead[plane].data = dict(x=[], top=[], ch=[])
                continue
            c0, c1 = win
            if c1 <= c0:
                s.data = dict(image=[], x=[], y=[], dw=[], dh=[])
                img_dead[plane].data = dict(x=[], top=[], ch=[])
                continue
            sub = vals[c0:c1, :].T                      # (ntick, nchanwin)
            s.data = dict(image=[sub], x=[off + c0], y=[0],
                          dw=[c1 - c0], dh=[vals.shape[1]])
            # dead channels in the window as full-height grey bars
            # (image pixel for channel ch spans [ch, ch+1] -> bar center ch+0.5)
            dch = dead_channels(plane, off, c0, c1)
            img_dead[plane].data = dict(x=[ch + 0.5 for ch in dch],
                                        top=[vals.shape[1]] * len(dch),
                                        ch=dch)
            p.x_range.start, p.x_range.end = off + c0, off + c1
            p.y_range.start, p.y_range.end = (tr if (pw and tr)
                                              else (0, vals.shape[1]))

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
        render_slice()   # render_slice now refreshes the waveform panels too

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
    w_ghosts.on_change("active", lambda a, o, n: render_slice())
    b_apply.on_click(apply_window)
    b_reset.on_click(reset_window)
    b_center.on_click(center_on_pos)
    b_pos_here.on_click(pos_from_slice)
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
    head = Div(text=f"<h2>PDHD imaging viewer &mdash; run {ev.meta.get('run')} "
                    f"evt {ev.meta.get('event')}</h2>"
                    f"<small>{npz_path}</small>", width=1120)
    controls = row(w_anode, w_face, b_prev, b_next, w_slice, w_ghosts)
    winrow = row(win["xlo"], win["xhi"], win["ylo"], win["yhi"],
                 win["zlo"], win["zhi"], b_apply, b_reset)
    posrow = row(w_pos_x, w_pos_y, w_pos_z, w_pos_pad, b_center, b_pos_here)
    chrow = row(w_frame, w_man_pl, w_man_ch, b_add, b_clear)
    layout = column(
        head,
        controls, slice_div, chan_div,
        row(f_yz, column(Div(text="<b>3D point projections</b>"),
                         row(f_xy, f_zy, f_xz), winrow, posrow)),
        Div(text="<hr><b>Waveforms</b> (tap a wire above to pick a channel)"),
        chrow, chips, f_wave,
        row(img2d[0], img2d[1], img2d[2]),
        status,
    )
    curdoc().add_root(layout)
    curdoc().title = "PDHD imaging viewer"

    # ---- initial render ---------------------------------------------------
    set_slice_spinner()
    reset_window()      # also calls apply_window -> render_slice
    render_slice()
    refresh_chips()


main(sys.argv)

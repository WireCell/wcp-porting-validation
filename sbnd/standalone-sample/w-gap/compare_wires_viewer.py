#!/usr/bin/env python3
"""Compare recob::Wire products between two art ROOT files (Bokeh server).

Reads `recob::Wires_<module>_<tag>_<process>` from files A and B via PyROOT
(needs the sbndcode environment for dictionaries), builds dense (channel x
tick) arrays per event and shows:

  * Three linked, zoomable 2D panels (channel on Y, tick on X): A, B and
    A-B, all with a bipolar (blue-white-red) colormap centered at zero.
    Every pan/zoom re-renders server-side with sign-preserving max-|v|
    pooling so isolated single-tick spikes never disappear when zoomed out.
  * Tap a channel in any 2D panel -> 1D waveforms of that channel below:
    A, B overlaid and the signed difference A-B.
  * Controls: file paths for A and B (+ Load), product tag, previous/next
    event, and a table of the largest |A-B| entries (also printed to stdout).

Launched by serve_compare_wires.sh; mirrors the server-side-callback style of
ql_scan/ql_scan_viewer.py.

Usage (standalone test of the data layer):
    python compare_wires_viewer.py A.root B.root [tag]
"""

import sys

import numpy as np

DEFAULT_TAG = "dnnsp"
DEFAULT_MODULE = "simtpc2d"
MAX_IMG_W = 800    # rendered image resolution budget per panel (channel axis)
MAX_IMG_H = 600    # (tick axis)
NTOP = 5           # how many largest diffs to report

# SBND channel layout: per APA u(1984) + v(1984) + w(1670) = 5638 channels.
NCH_U, NCH_V, NCH_W = 1984, 1984, 1670
NCH_APA = NCH_U + NCH_V + NCH_W

# sim::SimChannel TDC -> readout tick alignment: tick = tdc + SIMCH_TICK_OFFSET.
SIMCH_TICK_OFFSET = -2990

# recob::Wire tags are in units of ~electrons/50 (SP DeconNorm); scale them up
# so gauss/wiener/dnnsp are directly comparable with simchannel electrons.
WIRE_SCALE = 50.0


def region_channels(apa, plane):
    """Channel window [lo, hi) for the APA / plane selection ('all' allowed)."""
    if apa == "all":
        return 0, 2 * NCH_APA
    base = int(apa) * NCH_APA
    lo, hi = {"all": (0, NCH_APA),
              "u": (0, NCH_U),
              "v": (NCH_U, NCH_U + NCH_V),
              "w": (NCH_U + NCH_V, NCH_APA)}[plane]
    return base + lo, base + hi


# ---------------------------------------------------------------------------
# Data layer (PyROOT)
# ---------------------------------------------------------------------------
class WireFile:
    """One art ROOT file giving dense per-event (channel x tick) arrays."""

    def __init__(self, path):
        import ROOT  # deferred so --help works without the env
        self.ROOT = ROOT
        self.path = path
        self.tfile = ROOT.TFile.Open(path)
        if not self.tfile or self.tfile.IsZombie():
            raise IOError(f"cannot open {path}")
        self.tree = self.tfile.Get("Events")
        if not self.tree:
            raise IOError(f"no Events tree in {path}")
        self.nevents = self.tree.GetEntries()
        self._cache = {}  # (entry, branch) -> ndarray

    def wire_branches(self):
        """All recob::Wires branch names in the file."""
        out = []
        for b in self.tree.GetListOfBranches():
            name = b.GetName()
            if name.startswith("recob::Wires_"):
                out.append(name)
        return out

    def branch_for_tag(self, tag, module=DEFAULT_MODULE):
        want = f"recob::Wires_{module}_{tag}_"
        cands = [n for n in self.wire_branches() if n.startswith(want)]
        if not cands:
            raise KeyError(f"no branch {want}* in {self.path}; have: {self.wire_branches()}")
        # the same tag can exist from several processes (e.g. dnnsp from both
        # DetSim and ReDetSim); prefer the re-processed one.
        for n in cands:
            if n.endswith("_ReDetSim."):
                return n
        return cands[0]

    def event_label(self, entry):
        br = self.tree.GetBranch("EventAuxiliary")
        br.GetEntry(entry)
        aux = getattr(self.tree, "EventAuxiliary")
        eid = aux.id()
        return f"run {eid.run()} subrun {eid.subRun()} event {eid.event()}"

    def simchannel_branch(self):
        cands = [b.GetName() for b in self.tree.GetListOfBranches()
                 if b.GetName().startswith("sim::SimChannels_")]
        if not cands:
            raise KeyError(f"no sim::SimChannels branch in {self.path}")
        for n in cands:
            if n.endswith("_ReDetSim."):
                return n
        return cands[0]

    def _dense_simchannel(self, entry):
        """(channel x tick) array of summed ionization electrons.

        tick = tdc + SIMCH_TICK_OFFSET; entries shifted below tick 0 are dropped.
        """
        bname = self.simchannel_branch()
        br = self.tree.GetBranch(bname)
        br.GetEntry(entry)
        scs = getattr(self.tree, bname).product()  # vector<sim::SimChannel>
        if scs is None:
            raise IOError(f"product missing for {bname} entry {entry}")
        nch = 0
        ntick = 0
        store = []  # (channel, [(tick, electrons), ...])
        for sc in scs:
            ch = sc.Channel()
            pairs = []
            for tdcide in sc.TDCIDEMap():
                tick = int(tdcide.first) + SIMCH_TICK_OFFSET
                if tick < 0:
                    continue
                pairs.append((tick, float(sum(i.numElectrons for i in tdcide.second))))
            store.append((ch, pairs))
            nch = max(nch, ch + 1)
            if pairs:
                ntick = max(ntick, max(t for t, _ in pairs) + 1)
        arr = np.zeros((nch, max(ntick, 1)), dtype=np.float32)
        for ch, pairs in store:
            for t, q in pairs:
                arr[ch, t] += q
        return arr

    def dense(self, entry, tag, module=DEFAULT_MODULE):
        """(channel x tick) float32 array for one event; channel index = channel number.

        tag "simchannel" reads sim::SimChannels (summed ionization electrons
        per channel/tdc) instead of recob::Wire.
        """
        key = (entry, tag, module)
        if key in self._cache:
            return self._cache[key]
        if tag == "simchannel":
            arr = self._dense_simchannel(entry)
            self._cache = {key: arr}
            return arr
        bname = self.branch_for_tag(tag, module)
        br = self.tree.GetBranch(bname)
        br.GetEntry(entry)
        wrapper = getattr(self.tree, bname)
        wires = wrapper.product()  # vector<recob::Wire>
        if wires is None:
            raise IOError(f"product missing for {bname} entry {entry}")
        nch = 0
        ntick = 0
        for w in wires:
            nch = max(nch, w.Channel() + 1)
            ntick = max(ntick, w.NSignal())
        arr = np.zeros((nch, ntick), dtype=np.float32)
        for w in wires:
            sig = np.asarray(w.Signal(), dtype=np.float32)
            arr[w.Channel(), : sig.size] = sig
        arr *= WIRE_SCALE  # -> electrons, comparable with simchannel
        # keep only the latest event per (tag, module) to bound memory
        self._cache = {key: arr}
        return arr


def aligned(a, b):
    """Zero-pad A and B to a common (channel, tick) shape."""
    nch = max(a.shape[0], b.shape[0])
    nt = max(a.shape[1], b.shape[1])
    if a.shape != (nch, nt):
        tmp = np.zeros((nch, nt), dtype=a.dtype)
        tmp[: a.shape[0], : a.shape[1]] = a
        a = tmp
    if b.shape != (nch, nt):
        tmp = np.zeros((nch, nt), dtype=b.dtype)
        tmp[: b.shape[0], : b.shape[1]] = b
        b = tmp
    return a, b


def maxpool2d_signed(arr, max_h, max_w):
    """Downsample by block, keeping the element with the largest |value| so
    spikes of either sign survive.  Returns (img, fy, fx)."""
    h, w = arr.shape
    fy = max(1, int(np.ceil(h / max_h)))
    fx = max(1, int(np.ceil(w / max_w)))
    if fy == 1 and fx == 1:
        return arr, 1, 1
    ph = (-h) % fy
    pw = (-w) % fx
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw)), constant_values=0)
    H, W = arr.shape
    blk = arr.reshape(H // fy, fy, W // fx, fx)
    bmax = blk.max(axis=(1, 3))
    bmin = blk.min(axis=(1, 3))
    img = np.where(bmax > -bmin, bmax, bmin)
    return img, fy, fx


def top_diffs(diff, n=NTOP):
    """[(value, channel, tick)] of the n largest |diff| entries."""
    flat = np.argpartition(np.abs(diff).ravel(), -n)[-n:]
    out = [(float(abs(diff.ravel()[i])), int(i // diff.shape[1]), int(i % diff.shape[1]))
           for i in flat]
    return sorted(out, reverse=True)


def bipolar_palette(n=256):
    """Blue-white-red hex palette (matplotlib RdBu_r if available)."""
    try:
        import matplotlib.cm as cm
        from matplotlib.colors import to_hex
        cmap = cm.get_cmap("RdBu_r")
        return [to_hex(cmap(i / (n - 1))) for i in range(n)]
    except Exception:
        out = []
        for i in range(n):
            t = i / (n - 1)
            if t < 0.5:  # blue -> white
                s = t * 2
                r, g, b = int(255 * s), int(255 * s), 255
            else:        # white -> red
                s = (t - 0.5) * 2
                r, g, b = 255, int(255 * (1 - s)), int(255 * (1 - s))
            out.append(f"#{r:02x}{g:02x}{b:02x}")
        return out


# ---------------------------------------------------------------------------
# Bokeh app
# ---------------------------------------------------------------------------
def run_app():
    from bokeh.io import curdoc
    from bokeh.layouts import column, row
    from bokeh.models import (Button, ColorBar, ColumnDataSource, Div,
                              LinearColorMapper, Range1d, Select, TextInput)
    from bokeh.events import Tap
    from bokeh.plotting import figure

    argv = sys.argv[1:]
    path_a = argv[0] if len(argv) > 0 else ""
    path_b = argv[1] if len(argv) > 1 else ""
    tag_a0 = argv[2] if len(argv) > 2 else DEFAULT_TAG
    tag_b0 = argv[3] if len(argv) > 3 else tag_a0

    state = {"A": None, "B": None, "entry": 0,
             "tag_a": tag_a0, "tag_b": tag_b0,
             "a": None, "b": None, "diff": None}

    # -- widgets ------------------------------------------------------------
    # A and B are each (file, tag); tags may differ, e.g. gauss vs dnnsp from
    # the same file.  tag "simchannel" reads sim::SimChannels (electrons).
    in_a = TextInput(title="file A", value=path_a, width=420)
    in_b = TextInput(title="file B", value=path_b, width=420)
    in_tag_a = TextInput(title="tag A (gauss/wiener/dnnsp/simchannel)", value=tag_a0, width=220)
    in_tag_b = TextInput(title="tag B", value=tag_b0, width=120)
    bt_load = Button(label="Load", button_type="primary", width=80)
    bt_prev = Button(label="< prev event", width=100)
    bt_next = Button(label="next event >", width=100)
    sel_apa = Select(title="APA", value="all", options=["all", "0", "1"], width=80)
    sel_plane = Select(title="plane", value="all", options=["all", "u", "v", "w"], width=80)
    # colormap limits; empty = auto (0.9 * view min/max).  One pair for the
    # A/B panels (shared scale), one for the diff panel.
    in_cmin_ab = TextInput(title="A/B cmap min", value="-5000", placeholder="auto", width=100)
    in_cmax_ab = TextInput(title="A/B cmap max", value="5000", placeholder="auto", width=100)
    in_cmin_d = TextInput(title="diff cmap min", value="-1000", placeholder="auto", width=100)
    in_cmax_d = TextInput(title="diff cmap max", value="1000", placeholder="auto", width=100)
    # diff panel mode: absolute (A-B) or relative (A-B)/(max(|A|,|B|)+eps).
    # eps regularizes the 0/0 empty regions and low-amplitude ROI edges.
    sel_dmode = Select(title="diff mode", value="abs",
                       options=[("abs", "A-B"), ("rel", "(A-B)/max(|A|,|B|)")], width=150)
    in_deps = TextInput(title="rel eps", value="100", width=80)
    info = Div(text="load files to begin", width=900)
    topdiv = Div(text="", width=420)

    # -- three linked 2D panels (A, B, A-B), bipolar colormap ----------------
    PAL = bipolar_palette()
    # sized so the three panels fill a 1920-wide screen
    PANEL_W, PANEL_H = 620, 540

    figs = {}
    img_srcs = {}
    mappers = {}

    def make_panel(key, title, shared=None):
        kw = dict(width=PANEL_W, height=PANEL_H,
                  x_axis_label="channel", y_axis_label="time tick",
                  tools="pan,box_zoom,wheel_zoom,reset,save",
                  active_scroll="wheel_zoom", title=title)
        if shared is not None:
            kw["x_range"] = shared.x_range
            kw["y_range"] = shared.y_range
        else:
            # explicit Range1d: DataRange1d would auto-fit to the rendered
            # image and silently undo the APA/plane selection jumps.
            kw["x_range"] = Range1d(0, 1)
            kw["y_range"] = Range1d(0, 1)
        fig = figure(**kw)
        mapper = LinearColorMapper(palette=PAL, low=-1, high=1)
        src = ColumnDataSource(data=dict(image=[np.zeros((2, 2), dtype=np.float32)],
                                         x=[0], y=[0], dw=[1], dh=[1]))
        fig.image(image="image", x="x", y="y", dw="dw", dh="dh",
                  color_mapper=mapper, source=src)
        fig.add_layout(ColorBar(color_mapper=mapper, width=10), "right")
        figs[key], img_srcs[key], mappers[key] = fig, src, mapper
        return fig

    fig_a = make_panel("a", "A")
    fig_b = make_panel("b", "B", shared=fig_a)
    fig_d = make_panel("d", "A - B", shared=fig_a)

    # -- 1D panels below ----------------------------------------------------
    fig1d = figure(title="channel waveform (tap a 2D panel)", width=700, height=300,
                   x_axis_label="time tick", y_axis_label="signal",
                   tools="pan,box_zoom,wheel_zoom,reset,save")
    src_a = ColumnDataSource(data=dict(x=[], y=[]))
    src_b = ColumnDataSource(data=dict(x=[], y=[]))
    fig1d.line("x", "y", source=src_a, color="#2ca02c", legend_label="A", line_width=1.2)
    # B drawn as dots ON TOP of the A line so overlapping samples stay visible
    fig1d.scatter("x", "y", source=src_b, color="#d62728", legend_label="B",
                  size=3, marker="circle")
    fig1d.legend.click_policy = "hide"

    figdf = figure(title="A - B", width=700, height=260,
                   x_axis_label="time tick", y_axis_label="A - B",
                   x_range=fig1d.x_range,
                   tools="pan,box_zoom,wheel_zoom,reset,save")
    src_d = ColumnDataSource(data=dict(x=[], y=[]))
    figdf.line("x", "y", source=src_d, color="#d62728", line_width=1.2)

    # -- rendering ----------------------------------------------------------
    render_pending = {"on": False}

    def view_window():
        """Current view as channel window (x axis) and tick window (y axis)."""
        diff = state["diff"]
        nch, nt = diff.shape
        xr, yr = fig_a.x_range, fig_a.y_range
        c0 = int(max(0, np.floor(xr.start if xr.start is not None else 0)))
        c1 = int(min(nch, np.ceil(xr.end if xr.end is not None else nch)))
        t0 = int(max(0, np.floor(yr.start if yr.start is not None else 0)))
        t1 = int(min(nt, np.ceil(yr.end if yr.end is not None else nt)))
        return c0, c1, t0, t1

    def render_view():
        """Redraw all three 2D images for the current axis ranges."""
        render_pending["on"] = False
        if state["diff"] is None:
            return
        c0, c1, t0, t1 = view_window()
        if c1 - c0 < 2 or t1 - t0 < 2:
            return

        # A and B share one color scale; diff gets its own.
        # Arrays are (channel, tick); display is channel on X, tick on Y,
        # so transpose the pooled block.
        suba = state["a"][c0:c1, t0:t1]
        subb = state["b"][c0:c1, t0:t1]
        if sel_dmode.value == "rel":
            try:
                eps = float(in_deps.value.strip())
            except ValueError:
                eps = 100.0
            subd = (suba - subb) / (np.maximum(np.abs(suba), np.abs(subb)) + max(eps, 1e-9))
            fig_d.title.text = "(A - B) / (max(|A|,|B|) + %.4g)" % eps
        else:
            subd = suba - subb
            fig_d.title.text = "A - B"
        imgs = {}
        for key, arr in (("a", suba), ("b", subb), ("d", subd)):
            img, _, _ = maxpool2d_signed(arr, MAX_IMG_W, MAX_IMG_H)
            imgs[key] = img.T.astype(np.float32)

        def cmap_limits(lo_widget, hi_widget, data_lo, data_hi):
            """Box value if parseable, else 0.9 * view min/max."""
            def parse(w, default):
                s = w.value.strip()
                if not s:
                    return default
                try:
                    return float(s)
                except ValueError:
                    return default
            lo = parse(lo_widget, 0.9 * data_lo)
            hi = parse(hi_widget, 0.9 * data_hi)
            if hi <= lo:
                hi = lo + 1e-9
            return lo, hi

        ab_lo = min(float(imgs["a"].min()), float(imgs["b"].min()))
        ab_hi = max(float(imgs["a"].max()), float(imgs["b"].max()))
        vlo, vhi = cmap_limits(in_cmin_ab, in_cmax_ab, ab_lo, ab_hi)
        mappers["a"].low, mappers["a"].high = vlo, vhi
        mappers["b"].low, mappers["b"].high = vlo, vhi
        dlo, dhi = cmap_limits(in_cmin_d, in_cmax_d,
                               float(imgs["d"].min()), float(imgs["d"].max()))
        mappers["d"].low, mappers["d"].high = dlo, dhi
        for key in ("a", "b", "d"):
            img_srcs[key].data = dict(image=[imgs[key]], x=[c0], y=[t0],
                                      dw=[c1 - c0], dh=[t1 - t0])

    def schedule_render(attr, old, new):
        if render_pending["on"]:
            return
        render_pending["on"] = True
        curdoc().add_timeout_callback(render_view, 150)

    for rng in (fig_a.x_range, fig_a.y_range):
        rng.on_change("start", schedule_render)
        rng.on_change("end", schedule_render)

    def show_channel(ch):
        a, b = state["a"], state["b"]
        if a is None or ch < 0 or ch >= a.shape[0]:
            return
        x = np.arange(a.shape[1])
        src_a.data = dict(x=x, y=a[ch])
        src_b.data = dict(x=x, y=b[ch])
        src_d.data = dict(x=x, y=a[ch] - b[ch])
        d = a[ch] - b[ch]
        k = int(np.argmax(np.abs(d)))
        fig1d.title.text = (f"channel {ch}:  max|A-B| = {abs(d[k]):.4g} @ tick {k}")

    def on_tap(event):
        show_channel(int(round(event.x)))

    for f in (fig_a, fig_b, fig_d):
        f.on_event(Tap, on_tap)

    # -- APA / plane region selection ----------------------------------------
    def update_top_table():
        """Largest |A-B| within the selected APA/plane channel window."""
        if state["diff"] is None:
            return []
        lo, hi = region_channels(sel_apa.value, sel_plane.value)
        hi = min(hi, state["diff"].shape[0])
        sub = state["diff"][lo:hi]
        tops = [(v, c + lo, t) for v, c, t in top_diffs(sub)]
        region = f"APA {sel_apa.value} / plane {sel_plane.value}"
        rows = "".join(f"<tr><td>{v:.5g}</td><td>{c}</td><td>{t}</td></tr>"
                       for v, c, t in tops)
        topdiv.text = (f"<b>largest |A-B|</b> ({region})"
                       "<table border=1 cellpadding=3><tr><th>|A-B|</th>"
                       f"<th>channel</th><th>tick</th></tr>{rows}</table>")
        print(f"[entry {state['entry']} {region}] largest |A-B|: "
              + ", ".join(f"{v:.5g}@(ch {c}, tick {t})" for v, c, t in tops),
              flush=True)
        return tops

    def apply_region():
        if state["diff"] is None:
            return
        lo, hi = region_channels(sel_apa.value, sel_plane.value)
        nch, nt = state["diff"].shape
        hi = min(hi, nch)
        fig_a.x_range.start, fig_a.x_range.end = lo, hi
        fig_a.y_range.start, fig_a.y_range.end = 0, nt
        # make the Reset tool come back to this region, not the startup range
        fig_a.x_range.reset_start, fig_a.x_range.reset_end = lo, hi
        fig_a.y_range.reset_start, fig_a.y_range.reset_end = 0, nt
        # range callbacks schedule the re-render

    def on_region(attr, old, new):
        apply_region()
        update_top_table()

    sel_apa.on_change("value", on_region)
    sel_plane.on_change("value", on_region)

    def on_cmap(attr, old, new):
        schedule_render(attr, old, new)

    for w in (in_cmin_ab, in_cmax_ab, in_cmin_d, in_cmax_d, in_deps):
        w.on_change("value", on_cmap)
    sel_dmode.on_change("value", on_cmap)

    # -- data loading -------------------------------------------------------
    def load_event():
        A, B = state["A"], state["B"]
        if A is None or B is None:
            return
        entry = state["entry"]
        tag_a, tag_b = state["tag_a"], state["tag_b"]
        try:
            a = A.dense(entry, tag_a)
            b = B.dense(entry, tag_b)
        except Exception as e:  # bad tag / entry: report, keep prior view
            info.text = f"<b>error:</b> {e}"
            return
        a, b = aligned(a, b)
        state["a"], state["b"] = a, b
        state["diff"] = a - b
        nch, nt = a.shape
        apply_region()
        render_view()

        tops = update_top_table()
        info.text = (f"entry <b>{entry}</b> / {min(A.nevents, B.nevents)-1} "
                     f"({A.event_label(entry)})&nbsp;&nbsp; "
                     f"shape: {nch} ch x {nt} ticks<br>"
                     f"A: {A.path} : <b>{tag_a}</b><br>"
                     f"B: {B.path} : <b>{tag_b}</b>")
        fig_a.title.text = f"A: {tag_a}"
        fig_b.title.text = f"B: {tag_b}"
        fig_d.title.text = "A - B"
        show_channel(tops[0][1])  # preload 1D with the worst channel

    def do_load():
        try:
            state["A"] = WireFile(in_a.value.strip())
            state["B"] = WireFile(in_b.value.strip())
        except Exception as e:
            info.text = f"<b>error:</b> {e}"
            return
        state["tag_a"] = in_tag_a.value.strip() or DEFAULT_TAG
        state["tag_b"] = in_tag_b.value.strip() or state["tag_a"]
        state["entry"] = 0
        load_event()

    def step(dn):
        if state["A"] is None:
            return
        nmax = min(state["A"].nevents, state["B"].nevents)
        state["entry"] = int(np.clip(state["entry"] + dn, 0, nmax - 1))
        load_event()

    bt_load.on_click(do_load)
    bt_prev.on_click(lambda: step(-1))
    bt_next.on_click(lambda: step(+1))

    controls = column(row(in_a, in_b),
                      row(in_tag_a, in_tag_b, bt_load, bt_prev, bt_next, sel_apa, sel_plane,
                          in_cmin_ab, in_cmax_ab, in_cmin_d, in_cmax_d, sel_dmode, in_deps),
                      info)
    layout = column(controls,
                    row(fig_a, fig_b, fig_d),
                    row(column(fig1d, figdf), topdiv))
    curdoc().add_root(layout)
    curdoc().title = "recob::Wire A-B compare"

    if path_a and path_b:
        do_load()


# ---------------------------------------------------------------------------
def main_cli():
    """Standalone smoke test of the data layer: print top diffs of entry 0."""
    a, b = WireFile(sys.argv[1]), WireFile(sys.argv[2])
    tag_a = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_TAG
    tag_b = sys.argv[4] if len(sys.argv) > 4 else tag_a
    print(f"A: {a.path} ({a.nevents} events)  branches: {a.wire_branches()}")
    print(f"B: {b.path} ({b.nevents} events)  tags: A={tag_a} B={tag_b}")
    da, db = aligned(a.dense(0, tag_a), b.dense(0, tag_b))
    print(f"shape {da.shape}; A sum {da.sum():.6g}  B sum {db.sum():.6g}")
    for v, c, t in top_diffs(da - db):
        print(f"  |A-B| {v:.5g} @ channel {c} tick {t}")


if __name__.startswith("bokeh_app"):
    run_app()
elif __name__ == "__main__":
    main_cli()

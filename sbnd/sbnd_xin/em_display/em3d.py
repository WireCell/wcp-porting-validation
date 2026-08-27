"""doc pr/114 round 3 -- the 3-D view: camera, its CustomJS, and the Bee cloud.

Bokeh 3.9 has no 3-D glyph, so this is net-new machinery.  What it is NOT is a
three.js embed: Bee runs three.js **r145 from a CDN**
(wire-cell-bee3/events/templates/events/event.html:299-306), the only copy on
disk is r71 (the wrong version, kept alive for the dead-area Web Worker's use of
the removed THREE.Geometry API), Bee's own bundle js/bee/dist/bee.js is
gitignored and not built in this checkout, there is no node/npm on this box, and
em_display is a single-script Bokeh app with no static/ dir.  Vendoring Bee is a
project, not a step.

What this is instead: an **orthographic trackball inside an ordinary Bokeh
figure**.  Every glyph carries 3-D columns and the projected 2-D pair the glyph
actually draws; a CustomJS recomputes the projection in the browser on each drag
frame and calls `source.change.emit()`.  Zero new dependencies, zero JS assets,
zero build step -- and, because the glyphs live in normal data space, Bokeh's own
tap, box-select and hover keep working, which is the whole point of putting the
3-D view inside em_display rather than beside it.

The design rests on one fact that was READ in the shipped bokehjs rather than
recalled (bokeh/server/static/js/bokeh.js, UIEventBus.__trigger): the tail of
that method calls `this._trigger_bokeh_event(plot_view, e)` **unconditionally**,
after the active-tool switch.  So Pan / PanStart / PanEnd / MouseWheel reach
`js_on_event` even when no pan or scroll tool is active, and PointEvent carries
`modifiers` (shift/ctrl) and cumulative `delta_x`/`delta_y`.  That is what lets a
bare drag mean "rotate" without a pan tool fighting it for the gesture.

And one more, from GlyphRendererView.connect_signals in the same file:
`this.connect(this.model.data_source.change, update)`.  So mutating
`source.data.<col>` IN PLACE and calling `source.change.emit()` repaints without
assigning `source.data`, which is the difference between a local repaint and
shipping 25 000 points back to the server on every frame of a drag.

HONEST LIMIT, stated once: there is no JS engine and no node in this tree, so
**the CustomJS below is not machine-tested**.  The Python mirrors here (project,
camera_basis, bounding_sphere) are tested, and they are what the selftest asserts
against; the JS is covered by the scripted manual check-list in
docs/pr/114 section 11.
"""
import json
import math
import os
import zipfile

try:                                    # optional accelerator, never required
    from scipy.spatial import cKDTree   # noqa: F401
except ImportError:                     # pragma: no cover - measured both ways
    cKDTree = None

# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------


def camera_basis(az, el):
    """Right / up / forward for an orthographic camera at azimuth `az` and
    elevation `el` (radians).  Orthonormal by construction -- which is what makes
    the framing in `bounding_sphere` rotation-invariant."""
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    right = (-sa, ca, 0.0)
    up = (-ca * se, -sa * se, ce)
    fwd = (ca * ce, sa * ce, se)
    return right, up, fwd


def project(pts, az, el, centre=(0.0, 0.0, 0.0)):
    """(u, v, d) for a sequence of (x, y, z).  `d` is depth along the view
    direction, used only for depth cueing.  The Python mirror of JS_PROJECT."""
    r, u, f = camera_basis(az, el)
    cx, cy, cz = centre
    out = []
    for p in pts:
        px, py, pz = p[0] - cx, p[1] - cy, p[2] - cz
        out.append((px * r[0] + py * r[1] + pz * r[2],
                    px * u[0] + py * u[1] + pz * u[2],
                    px * f[0] + py * f[1] + pz * f[2]))
    return out


def bounding_sphere(pts, pad=1.15, floor=30.0):
    """(centre, R) framing a point set.

    The frame is set from the 3-D bounding sphere and NEVER from the projected
    extent.  Because right/up/fwd is orthonormal, |(u,v)| <= |p - centre| <= R
    for every camera, so an elongated track swinging from broadside to end-on can
    neither balloon out of frame nor shrink to a dot -- zoom stays entirely the
    user's.  Framing off the projected extent would re-fit on every drag frame,
    which is the same failure the Range1d-not-DataRange1d comment in the viewer
    warns about, one level up.
    """
    if not pts:
        return (0.0, 0.0, 0.0), floor
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    c = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)),
         0.5 * (min(zs) + max(zs)))
    r = 0.0
    for p in pts:
        d = math.sqrt((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2)
        if d > r:
            r = d
    return c, max(floor, r * pad)


# Preset cameras, (azimuth, elevation) in degrees.  Three of them reproduce the
# 2-D panels EXACTLY, which is the point -- a scanner who loses their bearings in
# 3-D can step back to a view they already trust and then rotate out of it again.
# Worked through from camera_basis:
#   az=-90, el=+89  ->  right=+x, up=+y   : the X-Y panel
#   az=-90, el=  0  ->  right=+x, up=+z   : the X-Z panel
#   az=  0, el=  0  ->  right=+y, up=+z   : Y-Z with the axes swapped
# `right` always has rz == 0 (there is no roll), so a z-horizontal Y-Z view is
# not reachable; "z-y" is the honest name for what the third one shows.
PRESETS = {
    "x-y": (-90.0, 89.0),
    "x-z": (-90.0, 0.0),
    "z-y": (0.0, 0.0),
    "iso": (-55.0, 20.0),
}
PRESET_ORDER = ["iso", "x-y", "x-z", "z-y"]


# ---------------------------------------------------------------------------
# the Bee charge cloud
# ---------------------------------------------------------------------------
#
# FRAME, and it is a design constraint rather than a preference.  doc pr/13
# established that `img-global` is the ONLY raw-frame layer in a Bee zip; it is
# dumped pre-pipeline, before ClusteringSwitchScope creates the corrected arrays.
# `clustering-global` and the PR layers are in (x_t0cor, y_cor, z_cor), and the
# per-cluster T0 offset between the two runs to +-121 cm.
#
# Measured for THIS sample before any of this was designed (12 manifest events,
# calib dump vs the zip's own PR layers):
#
#     dump segments[].points[] -> track_fit-global : NN median 0.00043 cm
#     dump main_vertex         -> vertices-global  : 0.00007 .. 0.00059 cm
#
# i.e. the same numbers to JSON rounding.  The dump is in the PR-layer frame, and
# pr/13 pins the PR layers to clustering-global (NN median 0.0010 cm).  So
# clustering-global is the base layer and img-global is offered only with a
# label saying what it is.  (A naive check on img-global looks innocuous -- fit
# points sit ~0.34 cm from BOTH clouds, which is just the point spacing --
# precisely because fit points live on the in-beam cluster, whose T0 shift is
# ~0.  That near-miss is why the track_fit-global test is the one that decides.)
CLOUD_LAYERS = ["clustering-global", "img-global"]

# ---------------------------------------------------------------------------
# which of the cloud's clusters ARE the neutrino candidate (round 4)
# ---------------------------------------------------------------------------
#
# The calib dump is the neutrino candidate: WCPPID::NeutrinoID is built from the
# main cluster plus the other clusters of the same flash bundle, and nothing
# else ever enters it.  The Bee cloud is the opposite -- `clustering-global` is
# every cluster in the readout, and in this sample that is overwhelmingly cosmic
# muon.  Measured over all 94 manifest events, the candidate is a **median 18.5%
# of the cloud** (p90 40%, max 55%), so the other four fifths are the noise the
# scanner has to look through.
#
# The two numberings do NOT meet: the dump's `cluster_id` is WCP's PR sub-cluster
# index (17, 24, 25, 57..85 in evt 64591) and the cloud's `real_cluster_id` is
# WCT's clustering id (3, 13..20, 30).  The Bee PR layers do not bridge them
# either -- `track_fit-global` carries dump SEGMENT ids (17002, 81037), not
# cluster ids in either namespace.  So the link has to be made in space, and the
# direction that works is reco -> cloud: every reco point is anchored on real
# charge, so whichever cloud clusters the reconstruction lives on ARE the
# candidate's clusters.
#
# Rule: a cloud cluster is kept when at least `min_hits` reco points have their
# nearest cloud point (within `tol` cm) in it.  Measured over all 94 events with
# min_hits=5: the largest shower's fitted points stay >= 99.3% covered (min over
# the sample; median 99.9%), i.e. the filter essentially never eats charge the
# scan is about.  A relative threshold (>= 0.5% of matched points) was measured
# too and moves the kept fraction by 0.001 while dropping coverage to 97.1% --
# no reason to prefer it, so the flat, explainable number wins.


def match_cluster_ids(cx, cy, cz, cid, pts, tol=2.0, force_grid=False):
    """{cloud cluster id: how many of `pts` land on it}, nearest within `tol`.

    Two implementations of ONE answer.  scipy's cKDTree when it imports (median
    7 ms over the 94-event sample), and a uniform grid hash when it does not
    (median 164 ms, worst 951 ms -- correct, but a second per click is not a
    scan).  scipy is therefore an accelerator and never a requirement, and the
    fallback is not a guess: selftest_em_display.py runs BOTH over all 94 events
    and asserts the kept-cluster sets are identical (they are, 94/94).

    The grid is exact for this radius by construction: cells are `tol` on a side,
    so every cloud point within `tol` of a query lies in one of the 27 cells
    around the query's own cell.
    """
    out = {}
    n = len(cx)
    if n == 0 or not len(pts):
        return out
    if cKDTree is not None and not force_grid:
        import numpy
        C = numpy.column_stack([numpy.asarray(cx, dtype="float64"),
                                numpy.asarray(cy, dtype="float64"),
                                numpy.asarray(cz, dtype="float64")])
        d, i = cKDTree(C).query(numpy.asarray(pts, dtype="float64"),
                                distance_upper_bound=float(tol))
        ca = numpy.asarray(cid)
        for c in ca[i[numpy.isfinite(d)]].tolist():
            out[c] = out.get(c, 0) + 1
        return out
    inv = 1.0 / float(tol)
    grid = {}
    for i in range(n):
        k = (int(math.floor(cx[i] * inv)), int(math.floor(cy[i] * inv)),
             int(math.floor(cz[i] * inv)))
        g = grid.get(k)
        if g is None:
            grid[k] = [i]
        else:
            g.append(i)
    t2 = tol * tol
    for p in pts:
        kx = int(math.floor(p[0] * inv))
        ky = int(math.floor(p[1] * inv))
        kz = int(math.floor(p[2] * inv))
        best, bd = -1, t2
        for ax in (kx - 1, kx, kx + 1):
            for ay in (ky - 1, ky, ky + 1):
                for az in (kz - 1, kz, kz + 1):
                    for i in grid.get((ax, ay, az), ()):
                        d = ((cx[i] - p[0]) ** 2 + (cy[i] - p[1]) ** 2
                             + (cz[i] - p[2]) ** 2)
                        if d < bd:
                            bd, best = d, i
        if best >= 0:
            c = cid[best]
            out[c] = out.get(c, 0) + 1
    return out


def candidate_clusters(cx, cy, cz, cid, pts, tol=2.0, min_hits=5,
                       force_grid=False):
    """(kept ids as a set, the full hit histogram).  Empty set means "no anchor" --
    the caller must fall back to the whole cloud and SAY so, because a silently
    black panel reads as a broken display rather than as a filtered one."""
    hits = match_cluster_ids(cx, cy, cz, cid, pts, tol=tol, force_grid=force_grid)
    return {k for k, v in hits.items() if v >= min_hits}, hits


# Two tiny caches, because the scan clicks through events and then fiddles with
# the cloud controls on ONE of them.  Re-parsing 1.2 MB of JSON and re-running
# the match to change `max points` from 25 000 to 50 000 is pure latency on an
# ssh tunnel.  Capped and FIFO-evicted: an unbounded cache across 94 events would
# hold ~100 MB of cloud.
_CLOUD_CACHE, _MATCH_CACHE, _CACHE_MAX = {}, {}, 4


def _cache_put(store, key, val):
    if key not in store and len(store) >= _CACHE_MAX:
        del store[next(iter(store))]
    store[key] = val
    return val


def bee_zip_path(sx, row):
    rnd = (row or {}).get("bee_round") or ""
    if not rnd:
        return None
    return os.path.join(sx, "bee", rnd + ".zip")


def bee_event_index(sx, row, event):
    """Which data/<n>/ directory in the zip is this event.

    The sibling .index.txt is authoritative ("<bee_idx>\\tevent"); the URL's
    trailing /event/<n>/ is the fallback and agrees with it by construction
    (views.py:175 uses <n> directly as the on-disk directory name)."""
    rnd = (row or {}).get("bee_round") or ""
    if rnd:
        idxf = os.path.join(sx, "bee", rnd + ".index.txt")
        if os.path.exists(idxf):
            with open(idxf) as fh:
                for ln in fh:
                    if ln.startswith("#"):
                        continue
                    f = ln.split()
                    if len(f) >= 2 and f[1] == str(event):
                        return int(f[0])
    url = (row or {}).get("bee_url") or ""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


def load_bee_cloud(sx, row, event, layer="clustering-global", max_pts=25000,
                   reco=None, candidate_only=False, tol=2.0, min_hits=5):
    """The charge cloud Bee itself draws, straight out of the local zip.

    Returns None when the zip is not on disk -- bee/em114/*.zip is gitignored, so
    a fresh clone has the display but not the cloud, and the panel has to degrade
    to skeleton-only with a banner rather than crash.

    ORDER MATTERS, and getting it wrong is silent.  The candidate filter runs on
    the FULL arrays and decimation runs on what survives it.  Filtering after
    decimating would take the 54 477-point event down to 25 000, then down again
    to its ~2 100 candidate points' share of that -- about a thousand -- while the
    readout happily said "showing 25 000".  Half the thing being scanned would be
    gone with no sign on screen.

    Decimation walks a FRACTIONAL index rather than taking every k-th point.  It
    is still deterministic (no RNG) and still proportional per cluster -- the
    arrays arrive grouped by cluster, so an evenly-spaced pick keeps every cluster
    in proportion instead of dropping the small ones -- but it hits the requested
    budget exactly.  A plain `[::k]` stride cannot: at 25 586 points with a
    25 000 budget it takes k=2 and throws away half the event to save 586 points.
    """
    zp = bee_zip_path(sx, row)
    idx = bee_event_index(sx, row, event)
    if not zp or idx is None or not os.path.exists(zp):
        return None
    member = "data/%d/%d-%s.json" % (idx, idx, layer)
    ckey = (zp, member)
    d = _CLOUD_CACHE.get(ckey)
    if d is None:
        try:
            with zipfile.ZipFile(zp) as z:
                raw = z.read(member)
        except (KeyError, OSError, zipfile.BadZipFile):
            return None
        d = _cache_put(_CLOUD_CACHE, ckey, json.loads(raw))
    n = len(d.get("x") or [])
    if n == 0:
        return None
    cid = d.get("real_cluster_id") or d.get("cluster_id") or [0] * n
    q = d.get("q") or [0.0] * n
    x, y, z = d["x"], d["y"], d["z"]

    # --- 1. candidate filter, on the full arrays ------------------------------
    sel = None
    kept_ids, why = None, ""
    if candidate_only:
        if not reco:
            why = ("this event has no reconstructed points to anchor the match "
                   "on, so every cluster is shown")
        else:
            mkey = (zp, member, len(reco), tol, min_hits)
            ids = _MATCH_CACHE.get(mkey)
            if ids is None:
                ids = _cache_put(_MATCH_CACHE, mkey, candidate_clusters(
                    x, y, z, cid, reco, tol=tol, min_hits=min_hits)[0])
            if not ids:
                why = ("no cloud cluster carries the reconstruction, so every "
                       "cluster is shown")
            else:
                kept_ids = ids
                sel = [i for i in range(n) if cid[i] in ids]
    ncand = n if sel is None else len(sel)
    if sel is None:
        sel = range(n)

    # --- 2. dense colour index ------------------------------------------------
    # Re-index the surviving clusters 0..k-1 by descending size before the % 20,
    # so the biggest candidate cluster is always colour 0 and two kept clusters
    # cannot collide on a palette slot while fewer than 20 survive.  Sorted by
    # (-count, id): the id breaks size ties, so the colouring is deterministic.
    cnt = {}
    for i in sel:
        cnt[cid[i]] = cnt.get(cid[i], 0) + 1
    order = sorted(cnt, key=lambda c: (-cnt[c], c))
    dense = {c: j for j, c in enumerate(order)}

    # --- 3. decimate what survived -------------------------------------------
    keep = min(ncand, max(1, int(max_pts)))
    idxs = sel if keep >= ncand else [sel[i * ncand // keep] for i in range(keep)]
    out = dict(
        x=[x[i] for i in idxs], y=[y[i] for i in idxs], z=[z[i] for i in idxs],
        q=[float(q[i]) for i in idxs],
        cid20=[float(dense[cid[i]] % 20) for i in idxs],
        total=n, candidate=ncand, ncluster=len(set(cid)),
        ncluster_kept=len(cnt), kept_ids=sorted(kept_ids) if kept_ids else None,
        filtered=kept_ids is not None, fallback=why,
        layer=layer, member=member, zip=zp)
    out["kept"] = len(out["x"])
    return out


# ---------------------------------------------------------------------------
# the CustomJS
# ---------------------------------------------------------------------------
#
# One redraw function, defined once, reused by every handler that needs it.  It
# is spliced into each CustomJS body rather than duplicated by hand; the selftest
# asserts the viewer inlines these constants and does not carry a divergent copy.

JS_PROJECT = r"""
// --- camera --------------------------------------------------------------
// Mirror of em3d.camera_basis / em3d.project.  Orthonormal triple, so
// u^2 + v^2 + d^2 == |p - centre|^2 exactly (the selftest pins this in Python).
const _az = cam.data.az[0], _el = cam.data.el[0];
const _cx = cam.data.cx[0], _cy = cam.data.cy[0], _cz = cam.data.cz[0];
const _R  = cam.data.R[0] || 1.0;
const _ca = Math.cos(_az), _sa = Math.sin(_az);
const _ce = Math.cos(_el), _se = Math.sin(_el);
const rx = -_sa,       ry =  _ca,       rz = 0.0;
const ux = -_ca * _se, uy = -_sa * _se, uz = _ce;
const fx =  _ca * _ce, fy =  _sa * _ce, fz = _se;
"""

JS_REDRAW = JS_PROJECT + r"""
// --- point layers --------------------------------------------------------
// Depth CUEING, not depth sorting.  Bokeh draws in row order, so the only
// occlusion cue available without permuting every column on every frame is
// alpha and size falling off with depth -- and that is the cue that actually
// carries depth in a still frame.  Motion parallax covers the rest on a drag.
// ptalpha / ptsize / ptcue are three PARALLEL ARRAYS indexed the same way as
// `pts`, because the viewer keeps one table (_PT_CFG) that both the Python fill
// and these frames read -- neither mirror carries its own copy of a base size.
//
// (An array of {size, alpha, cue} objects would also work: Bokeh serialises a
// Python dict as {"type":"map", ...} but bokehjs's `_decode_map` returns a plain
// object whenever every key is a string.  It returns a real JS **Map** as soon
// as one key is not -- which is the shape trap worth remembering, and the one
// selftest_em_display.py guards.)
for (let s = 0; s < pts.length; s++) {
    const d = pts[s].data;
    const n = d.x.length;
    const a0 = ptalpha[s], s0 = ptsize[s], cue = ptcue[s] > 0.5;
    const u = new Float64Array(n), v = new Float64Array(n);
    const al = new Float64Array(n), sz = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const px = d.x[i] - _cx, py = d.y[i] - _cy, pz = d.z[i] - _cz;
        u[i] = px * rx + py * ry + pz * rz;
        v[i] = px * ux + py * uy + pz * uz;
        if (cue) {
            const t = 0.5 + 0.5 * ((px * fx + py * fy + pz * fz) / _R);
            const tc = t < 0 ? 0 : (t > 1 ? 1 : t);
            al[i] = a0 * (0.30 + 0.70 * tc);
            sz[i] = s0 * (0.70 + 0.60 * tc);
        } else {
            al[i] = a0;
            sz[i] = s0;
        }
    }
    d.u = u; d.v = v; d.al = al; d.sz = sz;
    // In-place mutation + change.emit(): repaints locally (GlyphRendererView
    // connects data_source.change -> update_data) without assigning .data,
    // which would ship every point back to the server on every drag frame.
    pts[s].change.emit();
}
// --- polyline layers -----------------------------------------------------
for (let s = 0; s < lines.length; s++) {
    const d = lines[s].data;
    const n = d.xs3.length;
    const xs = new Array(n), ys = new Array(n);
    for (let i = 0; i < n; i++) {
        const X = d.xs3[i], Y = d.ys3[i], Z = d.zs3[i], m = X.length;
        const a = new Float64Array(m), b = new Float64Array(m);
        for (let j = 0; j < m; j++) {
            const px = X[j] - _cx, py = Y[j] - _cy, pz = Z[j] - _cz;
            a[j] = px * rx + py * ry + pz * rz;
            b[j] = px * ux + py * uy + pz * uz;
        }
        xs[i] = a; ys[i] = b;
    }
    d.xs = xs; d.ys = ys;
    lines[s].change.emit();
}
// --- arrow heads ---------------------------------------------------------
// The head angle has no 3-D analogue: it is the projected direction, so it must
// be recomputed here alongside u/v.  Bokeh's triangle marker points at +y.
for (let s = 0; s < heads.length; s++) {
    const d = heads[s].data;
    const n = d.x.length;
    const u = new Float64Array(n), v = new Float64Array(n), an = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const px = d.x[i] - _cx, py = d.y[i] - _cy, pz = d.z[i] - _cz;
        const qx = d.x0[i] - _cx, qy = d.y0[i] - _cy, qz = d.z0[i] - _cz;
        const uu = px * rx + py * ry + pz * rz;
        const vv = px * ux + py * uy + pz * uz;
        const u0 = qx * rx + qy * ry + qz * rz;
        const v0 = qx * ux + qy * uy + qz * uz;
        u[i] = uu; v[i] = vv;
        an[i] = Math.atan2(vv - v0, uu - u0) - Math.PI / 2.0;
    }
    d.u = u; d.v = v; d.angle = an;
    heads[s].change.emit();
}
"""

JS_PANSTART = r"""
// Hammer reports deltas CUMULATIVE from the gesture start, so the handler must
// anchor on the state at panstart and add -- integrating per frame would drift.
const d = cam.data;
d.az0[0] = d.az[0];  d.el0[0] = d.el[0];
d.xs0[0] = xr.start; d.xe0[0] = xr.end;
d.ys0[0] = yr.start; d.ye0[0] = yr.end;
"""

JS_ROTATE = r"""
// A bare drag rotates.  The guard is the whole mode system: the moment the user
// picks Box Select (or Pan) in the toolbar, rotation steps aside for it -- no
// extra mode UI, and no gesture fought over by two handlers.  (Pan events reach
// js_on_event even with no pan tool active; see the module docstring.)
//
// The guard reads `toolbar.gestures.pan.active`, NOT `toolbar.active_drag`, and
// the difference is not cosmetic.  `active_drag` is the CONFIGURATION property
// ("auto" by default, set to null here); bokehjs's Toolbar._active_change writes
// the live state to `this.gestures[et].active` and never touches active_drag,
// so a guard on active_drag stays null forever and rotation would fight
// box-select on every drag.  `gestures.pan.active` is exactly what
// UIEventBus.__trigger itself consults.  BoxSelect, BoxZoom, Lasso and Pan all
// declare event_type "pan", so this one check covers every drag tool.
const _g = p.toolbar.gestures;
if (_g != null && _g.pan != null && _g.pan.active != null) { return; }
const d = cam.data;
const dx = cb_obj.delta_x || 0.0, dy = cb_obj.delta_y || 0.0;
const shift = cb_obj.modifiers ? cb_obj.modifiers.shift : false;
if (shift) {
    const W = d.xe0[0] - d.xs0[0], H = d.ye0[0] - d.ys0[0];
    const wpx = p.inner_width || p.width || 800;
    const hpx = p.inner_height || p.height || 640;
    const ox = -dx / wpx * W, oy = dy / hpx * H;
    xr.start = d.xs0[0] + ox; xr.end = d.xe0[0] + ox;
    yr.start = d.ys0[0] + oy; yr.end = d.ye0[0] + oy;
    return;
}
const K = 0.0075;                       // rad per pixel
let el = d.el0[0] - dy * K;
const lim = Math.PI / 2.0 - 0.02;       // never look exactly down the pole
if (el >  lim) el =  lim;
if (el < -lim) el = -lim;
d.az[0] = d.az0[0] + dx * K;
d.el[0] = el;
""" + JS_REDRAW

JS_PANEND = r"""
// One round trip per gesture, not per frame: Python only needs the camera to
// record it in the label, so it learns it when the drag ends.
const d = cam.data;
camtxt.value = (d.az[0]).toFixed(4) + "," + (d.el[0]).toFixed(4);
"""

# Set the camera from Python (preset buttons, sliders, event load) and redraw.
JS_APPLY = JS_REDRAW

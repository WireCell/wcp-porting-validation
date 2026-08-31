#!/usr/bin/env python3
# doc pr/138 Phase A -- the shower SPLIT scan tool.  Bokeh app.
"""Drag EM bundles and segments between groups; the 3-D view recolours live.

    ./split_display/serve_split_display.sh [PORT] --scan-tag splitscan-0901-owner

WHAT THIS IS FOR.  doc pr/137 sec 14 built a 172-object curated set and sec 15.6
added the TRIM verdict.  This is the tool that turns those objects into labels:
per object, how many things is it, and -- the part the old mark in/out mechanism
could not express -- WHICH segments belong to which one.

THE THREE LEVELS (owner's design, doc pr/138 sec A1):
    GROUP    a drop column.  "this is one object."  The verdict is READ OFF the
             columns, never typed: 1 non-empty group = KEEP, 2 = SPLIT2,
             3 = SPLIT3, anything in JUNK = TRIM.
      BUNDLE the draggable 'directory' -- a spatially connected set of segments.
             Drag it and its whole segment list moves.
        SEG  the 'file'.  Drag one out of its bundle when the true boundary is
             finer than the bundle, which doc pr/137 sec 15.5 says it sometimes is.

FORK, NOT AN EDIT (M10).  em_display/ is a production scan tool with committed
records behind it (emscan-0827, emscan-0828-agent5, pi0scan-0829-agent).  This is
a separate app on its own port writing its own tag.  It IMPORTS em3d for the 3-D
camera -- import is not modification, and duplicating 541 lines of trackball with
its own drift would be worse than the coupling.

M13.  --scan-tag names a FRESH directory.  The viewer refuses to write into any
tag it did not create, so a mis-typed tag cannot overwrite a scan record.
"""
import os, sys, json, math, argparse, collections, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'em_display'))
os.chdir(ROOT)

import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import (ColumnDataSource, CustomJS, Div, TextInput, Button,
                          Select, RadioButtonGroup, TapTool, HoverTool, PreText,
                          Range1d, WheelZoomTool, ResetTool, SaveTool, BoxZoomTool)
from bokeh.layouts import column, row as brow
from bokeh.events import Pan, PanStart, PanEnd, DocumentReady

import em3d
import split_model as SM
import split_tree_js as TJ
import bee_links as BL

DEFAULT_NGROUPS = 3          # owner: "3 can be the default though"
MAX_NGROUPS = len(SM.GROUP_COLORS)


def groups():
    """the live group list: 0 .. n-1 then JUNK.  A busy event can need more than
    three, so the count is state, not a constant."""
    return list(range(STATE['ngroups'])) + [SM.JUNK]


def group_name(g):
    return "JUNK (trim)" if g == SM.JUNK else "Group %d" % g


def group_color(g):
    return SM.JUNK_COLOR if g == SM.JUNK else SM.GROUP_COLORS[g % len(SM.GROUP_COLORS)]


# ----------------------------------------------------------------- args
ap = argparse.ArgumentParser()
ap.add_argument('--scan-tag', default='splitscan-0901-agent')
ap.add_argument('--set', default='docs/pr/pr137-curated-set.tsv')
ap.add_argument('--owner-only', action='store_true',
                help='restrict the worklist to owner_scan=1 (the 50)')
ap.add_argument('--gap', type=float, default=4.0, help='bundle linkage gap, cm')
args, _ = ap.parse_known_args()

LABEL_DIR = os.path.join('em_labels', args.scan_tag)


def worklist():
    rows = []
    with open(args.set) as fh:
        hdr = None
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if hdr is None:
                hdr = f; continue
            d = dict(zip(hdr, f))
            if args.owner_only and d.get('owner_scan') != '1':
                continue
            rows.append((int(d['event']), int(d['node']), float(d['Q']),
                         d.get('stratum', ''), d.get('proxy_cls', '')))
    rows.sort(key=lambda t: -t[2])
    return rows


WORK = worklist()

# Scanned once: bee/*/<name>.url joined to <name>.index.txt gives event -> set+index,
# and Bee addresses an event by its INDEX IN THE SET.  A set that was never
# uploaded has no .url and contributes nothing, which is why bee/pr137r2 supplies
# no links until its upload is authorised (CLAUDE.md sec 5.6).
try:
    BEE = BL.scan()
except Exception:
    BEE = None

# ----------------------------------------------------------------- state
STATE = dict(i=0, payload=None, group={}, bundles={}, row=None,
             ngroups=DEFAULT_NGROUPS)

pts = ColumnDataSource(dict(x=[], y=[], z=[], u=[], v=[], seg=[], bundle=[],
                            color=[], alpha=[], size=[], hl=[]))
cam = ColumnDataSource(dict(az=[0.6], el=[0.35], cx=[0.0], cy=[0.0], cz=[0.0],
                            R=[100.0], az0=[0.6], el0=[0.35],
                            xs0=[0.0], xe0=[1.0], ys0=[0.0], ye0=[1.0]))
vtx = ColumnDataSource(dict(x=[], y=[], z=[], u=[], v=[], al=[], sz=[]))
# The rest of the event, drawn faint.  The owner asked why the vertex star sits
# off the object: because the display holds ONE object and the event holds many
# (evt396222: 123 of the event's 180 segments are this shower, and its 14.5 cm
# vertex gap is a real photon conversion, start_connection_type=2).  Without the
# surroundings that reads as a bug rather than as physics.
ctx = ColumnDataSource(dict(x=[], y=[], z=[], u=[], v=[], al=[], sz=[]))
wprof = ColumnDataSource(dict(xs=[], ys=[], color=[], width=[], dash=[]))

gmap_box = TextInput(value='{}', visible=False)     # seg -> group, for the JS
cmap_box = TextInput(value='{}', visible=False)
hi_box = TextInput(value='', visible=False)         # highlight channel
moved_box = TextInput(value='', visible=False)      # drop channel
cam_box = TextInput(value='', visible=False)

bee = Div(text='', width=780, height=34)
tree = Div(text='', width=780, height=640)
info = Div(text='', width=780)
status = PreText(text='', width=780, height=54)

verdict_btn = RadioButtonGroup(labels=['KEEP', 'SPLIT2', 'SPLIT3', 'SPLIT4+',
                                       'TRIM', 'UNSURE'], active=0, width=560)
conf_btn = RadioButtonGroup(labels=['high', 'medium', 'low'], active=0, width=260)
note_box = TextInput(title='note (optional)', width=520)
centre_sel = Select(title='rotate about', width=210, value='nu vertex',
                    options=['nu vertex', 'object centroid', 'event centre'])


# ----------------------------------------------------------------- 3-D view
# Range1d, NOT the DataRange1d default: an auto range re-fits itself whenever a
# source changes, which silently undoes the user's zoom on every drop
# (em_display_viewer.py:196 warns about the same thing).  Square figure + equal
# spans is what keeps the projection isotropic -- do not let those drift apart.
_wheel3 = WheelZoomTool()
_tap3 = TapTool()
fig = figure(width=620, height=620,
             x_range=Range1d(-100, 100), y_range=Range1d(-100, 100),
             tools=[_wheel3, _tap3, BoxZoomTool(), ResetTool(), SaveTool()],
             toolbar_location='right', output_backend='webgl',
             title='drag rotates | shift-drag pans | wheel zooms | tap picks a segment')
fig.toolbar.active_scroll = _wheel3
fig.toolbar.active_tap = _tap3
fig.grid.visible = False
fig.axis.visible = False
# Explicitly None, not "auto": auto would make a drag tool active and a bare drag
# would box-select instead of rotating.
fig.toolbar.active_drag = None

r_ctx = fig.scatter('u', 'v', source=ctx, size=2, color='#c8c8c8', alpha=0.30,
                    line_color=None)
r_pts = fig.scatter('u', 'v', source=pts, size='size', color='color',
                    alpha='alpha', line_color=None)
r_vtx = fig.scatter('u', 'v', source=vtx, size=16, marker='star',
                    color='#000000', alpha=1.0)
fig.add_tools(HoverTool(renderers=[r_pts], tooltips=[('segment', '@seg'),
                                                     ('bundle', '@bundle')]))

# theta-phi ray map (owner factor 1).  It shares the SAME source as the 3-D view,
# so the drop recolour and the click highlight reach it for free -- no second
# colour channel to keep in step.
tp = figure(width=360, height=330, match_aspect=True,
            tools='pan,wheel_zoom,box_zoom,reset', toolbar_location=None,
            title='theta-phi ray map from the vertex (deg)')
tp.scatter('tx', 'ty', source=pts, size='size', color='color', alpha='alpha',
           line_color=None)
tp.grid.grid_line_alpha = 0.25

# width vs depth against the in-situ single-shower null (owner factor 2).
wp = figure(width=360, height=330, tools='pan,wheel_zoom,reset',
            toolbar_location=None,
            title='transverse RMS vs depth  (dashed = single-shower null)')
wp.multi_line('xs', 'ys', source=wprof, line_color='color',
              line_width='width', line_dash='dash')
wp.xaxis.axis_label = 'depth along axis (cm)'
wp.yaxis.axis_label = 'transverse RMS (cm)'


def _cam_js(extra=''):
    return CustomJS(args=dict(cam=cam, pts=[pts, vtx, ctx], lines=[], heads=[],
                              ptalpha=[1.0, 1.0, 0.30], ptsize=[4.0, 16.0, 2.0],
                              ptcue=[1.0, 0.0, 1.0], xr=fig.x_range, yr=fig.y_range,
                              p=fig, camtxt=cam_box),
                    code=em3d.JS_REDRAW + extra)


fig.js_on_event(PanStart, CustomJS(args=dict(cam=cam, xr=fig.x_range, yr=fig.y_range),
                                   code=em3d.JS_PANSTART))
fig.js_on_event(Pan, CustomJS(args=dict(cam=cam, pts=[pts, vtx, ctx], lines=[], heads=[],
                                        ptalpha=[1.0, 1.0, 0.30], ptsize=[4.0, 16.0, 2.0],
                                        ptcue=[1.0, 0.0, 1.0], xr=fig.x_range,
                                        yr=fig.y_range, p=fig),
                              code=em3d.JS_ROTATE))
fig.js_on_event(PanEnd, CustomJS(args=dict(cam=cam, camtxt=cam_box),
                                 code=em3d.JS_PANEND))
pts.selected.js_on_change('indices', CustomJS(args=dict(cloud=pts, hi=hi_box),
                                              code=TJ.JS_TAP))
hi_box.js_on_change('value', CustomJS(args=dict(cloud=pts, hi=hi_box),
                                      code=TJ.JS_HIGHLIGHT))
gmap_box.js_on_change('value', CustomJS(
    args=dict(cloud=pts, gmap=gmap_box, cmap=cmap_box, moved=moved_box, hi=hi_box),
    code=TJ.JS_RECOLOR))
# ARMING THE TREE.  Four earlier builds tried to arm the drag handlers from a
# property change -- div.text, then gmap_box.value -- and all four bound nothing.
# A js_on_change on a widget with visible=False does not reach the client at all,
# and a change made while the document is being BUILT is serialised as initial
# state rather than emitted as an event.  DocumentReady is the one channel that
# is neither: bokehjs fires it once, after the document exists, unconditionally.
curdoc().js_on_event(DocumentReady, CustomJS(
    args=dict(cloud=pts, gmap=gmap_box, cmap=cmap_box, moved=moved_box, hi=hi_box,
              pts=[pts, vtx, ctx], cam=cam, lines=[], heads=[],
              ptalpha=[1.0, 1.0, 0.30], ptsize=[4.0, 16.0, 2.0], ptcue=[1.0, 0.0, 1.0],
              xr=fig.x_range, yr=fig.y_range, p=fig),
    code=TJ.JS_SETUP + TJ.JS_RECOLOR_BODY + em3d.JS_REDRAW))

# THE INITIAL-LOAD TRAP, and it bit both of these.  Bokeh serialises the document's
# INITIAL state; no change event is emitted client-side for values set while the
# document is being built.  So neither the camera redraw nor the tree bind fires on
# first paint -- the cloud sits at u=v=0 and nothing is draggable.  The fix is two
# parts: register a js_on_change on cam.data (there was none: JS_REDRAW only ran
# from Pan), and bump both from a next-tick callback AFTER the document exists.
cam.js_on_change('data', _cam_js())


# ----------------------------------------------------------------- rendering
def render_tree():
    p = STATE['payload']
    if p is None:
        tree.text = TJ.CSS + "<div id='split-tree'>no object</div>"
        return
    grp = STATE['group']
    bysegs = {s['seg']: s for s in p['segs']}
    # bundle -> segs, and bundle -> its majority group
    bmem = collections.defaultdict(list)
    for s in p['segs']:
        bmem[s['bundle']].append(s['seg'])
    bgrp = {}
    for b, segs in bmem.items():
        c = collections.Counter()
        for s in segs:
            c[grp.get(s, 0)] += max(bysegs[s]['q'], 0.0)
        bgrp[b] = c.most_common(1)[0][0] if c else 0
    Qtot = p['Q'] or 1.0
    html = [TJ.CSS, "<div id='split-tree'><div class='cols'>"]
    for g in groups():
        bl = sorted([b for b in bmem if bgrp[b] == g],
                    key=lambda b: -sum(max(bysegs[s]['q'], 0.) for s in bmem[b]))
        qg = sum(max(bysegs[s]['q'], 0.) for b in bl for s in bmem[b])
        html.append("<div class='col' data-group='%d'>" % g)
        html.append("<div class='colhdr' style='background:%s'>%s &nbsp; %d bundle(s) "
                    "&nbsp; %.0f%% q</div>" % (group_color(g), group_name(g),
                                               len(bl), 100.0 * qg / Qtot))
        for b in bl:
            segs = sorted(bmem[b], key=lambda s: -max(bysegs[s]['q'], 0.))
            qb = sum(max(bysegs[s]['q'], 0.) for s in segs)
            html.append("<div class='bundle' draggable='true' data-drag='bundle' data-bundle='%d' "
                        "data-segs='%s'>" % (b, ",".join(str(s) for s in segs)))
            html.append("<div class='bhdr'>bundle %d &nbsp; %d seg &nbsp; "
                        "%.1f%% q</div>" % (b, len(segs), 100.0 * qb / Qtot))
            for s in segs:
                d = bysegs[s]
                # a segment whose own group differs from its bundle's is shown
                # in its own group's column instead -- so mark it here as moved
                if grp.get(s, 0) != g:
                    continue
                html.append("<div class='seg' draggable='true' data-drag='seg' data-seg='%d' "
                            "data-segs='%d'>#%d &nbsp;<span class='muted'>"
                            "%.1f%% &nbsp; %.0fp &nbsp; %.0fcm &nbsp; d%.0f</span>"
                            "</div>" % (s, s, s, 100.0 * max(d['q'], 0.) / Qtot,
                                        d['npts'], d['length'], d['dvtx']))
            html.append("</div>")
        # segments that were dragged here individually, away from their bundle
        loose = [s['seg'] for s in p['segs']
                 if grp.get(s['seg'], 0) == g and bgrp[s['bundle']] != g]
        if loose:
            html.append("<div class='bundle'><div class='bhdr'>moved segments</div>")
            for s in sorted(loose, key=lambda s: -max(bysegs[s]['q'], 0.)):
                d = bysegs[s]
                html.append("<div class='seg' draggable='true' data-drag='seg' data-seg='%d' "
                            "data-segs='%d'>#%d <span class='muted'>(from bundle %d, "
                            "%.1f%% q)</span></div>" % (s, s, s, d['bundle'],
                                                        100.0 * max(d['q'], 0.) / Qtot))
            html.append("</div>")
        html.append("</div>")
    html.append("</div></div>")
    STATE['rev'] = STATE.get('rev', 0) + 1
    # a distinct string every render: assigning an identical value emits no
    # change, so the re-bind would be skipped exactly when the tree was rebuilt
    html.append("<!--rev%d-->" % STATE['rev'])
    tree.text = "".join(html)


VERDICTS = ['KEEP', 'SPLIT2', 'SPLIT3', 'SPLIT4+', 'TRIM', 'UNSURE']


def n_parts():
    return len({g for g in STATE['group'].values() if g != SM.JUNK})


def derive_verdict():
    """the verdict is READ OFF the columns, never typed."""
    grp = STATE['group']
    if not grp:
        return VERDICTS.index('UNSURE')
    n = n_parts()
    junk = any(g == SM.JUNK for g in grp.values())
    if n >= 4:
        return VERDICTS.index('SPLIT4+')
    if n == 3:
        return VERDICTS.index('SPLIT3')
    if n == 2:
        return VERDICTS.index('SPLIT2')
    return VERDICTS.index('TRIM') if junk else VERDICTS.index('KEEP')


def refresh(recolor=True):
    p = STATE['payload']
    if p is None:
        return
    render_tree()                      # bumps STATE['rev']
    # The value must DIFFER every time or no change event reaches the client and
    # the recolour+setup callback is skipped -- which is exactly how the first
    # three builds shipped an unbound tree.  '_rev' is inert on the JS side: the
    # recolour looks up map[seg] with a numeric key and never sees it.
    cmap_box.value = json.dumps({str(g): group_color(g) for g in groups()})
    payload = {str(k): v for k, v in STATE['group'].items()}
    payload['_rev'] = STATE['rev']
    gmap_box.value = json.dumps(payload)
    _update_wprof()
    verdict_btn.active = derive_verdict()
    grp = STATE['group']
    c = collections.Counter(grp.values())
    info.text = ("<b>evt %d &nbsp; node %d</b> &nbsp; Q=%.3g &nbsp; %d seg &nbsp; "
                 "%d bundles<br><span style='color:#666'>proposal: %s</span><br>"
                 "<span style='color:#666'>groups: %s &nbsp;|&nbsp; derived verdict: "
                 "<b>%s</b></span>"
                 % (p['event'], p['node'], p['Q'], p['nseg'], p['nbundle'],
                    p['reason'],
                    ", ".join("%s=%d" % (group_name(g), n) for g, n in sorted(c.items())),
                    verdict_btn.labels[derive_verdict()]))
    g = STATE.get('vgap')
    if g == g and g is not None and g > 5.0:
        info.text += ("<br><span style='color:#a33'>note: the nearest charge of "
                      "this object is <b>%.1f cm</b> from the vertex star &mdash; "
                      "an unusually large gap (median over the owner set is 0.0 cm), "
                      "so the reference point is an extrapolation here.</span>" % g)


def _update_wprof():
    """one transverse-RMS-vs-depth curve per non-empty group, plus the in-situ null.

    doc pr/137 sec 12 fitted w_single(r) = 3.575 + 0.0283 r cm on 346 SINGLE
    showers, and doc pr/137 sec 15.4 found the practical consequence: every
    hand-labelled SPLIT sat 2-10x above this line and every KEEP at or below it.
    It did not win the AUC ranking; it is the fastest thing to READ."""
    row = STATE['row']
    if row is None:
        wprof.data = dict(xs=[], ys=[], color=[], width=[], dash=[])
        return
    curves, (lo, hi) = SM.group_width_profiles(row, STATE['group'])
    xs, ys, col, wid, dash = [], [], [], [], []
    for g, r, w in curves:
        xs.append(list(r)); ys.append(list(w))
        col.append(group_color(g)); wid.append(2.5); dash.append('solid')
    if lo is not None and hi is not None and hi > lo:
        nr = np.linspace(lo, hi, 12)
        xs.append(list(nr)); ys.append([float(x) for x in SM.w_single(nr)])
        col.append('#000000'); wid.append(1.5); dash.append('dashed')
    wprof.data = dict(xs=xs, ys=ys, color=col, width=wid, dash=dash)


def load(i):
    if not WORK:
        status.text = "worklist is empty"
        return
    i = max(0, min(i, len(WORK) - 1))
    STATE['i'] = i
    ev, nd, Q, strat, proxy = WORK[i]
    status.text = "loading evt%d node%d  (%d of %d)" % (ev, nd, i + 1, len(WORK))
    row = SM.load_object(ev, nd)
    if row is None:
        status.text = "evt%d node%d: not found in the arm" % (ev, nd)
        return
    p = SM.object_payload(row, gap=args.gap)
    STATE['row'] = row
    STATE['payload'] = p
    STATE['ngroups'] = DEFAULT_NGROUPS
    STATE['group'] = {s['seg']: s['group'] for s in p['segs']}
    # existing label wins over the proposal
    prev = read_label(ev, nd)
    if prev and prev.get('groups'):
        STATE['group'] = {int(k): int(v) for k, v in prev['groups'].items()}
        status.text = "evt%d node%d  (%d of %d)  -- existing label loaded" % (
            ev, nd, i + 1, len(WORK))
    else:
        status.text = "evt%d node%d  (%d of %d)  -- proposal" % (ev, nd, i + 1, len(WORK))
    # 3-D cloud
    P, v = row['P'], row['v']
    xs, ys, zs, sg, bd = [], [], [], [], []
    seg2b = {s['seg']: s['bundle'] for s in p['segs']}
    for s in sorted(row['segs']):
        A = P.get(s)
        if A is None or not len(A):
            continue
        for k in range(len(A)):
            xs.append(float(A[k, 0])); ys.append(float(A[k, 1])); zs.append(float(A[k, 2]))
            sg.append(int(s)); bd.append(int(seg2b.get(s, 0)))
    n = len(xs)
    tx, ty = SM.theta_phi(np.asarray(list(zip(xs, ys, zs))) if n else np.zeros((0, 3)),
                          np.asarray([1.0] * n), v)
    pts.data = dict(x=xs, y=ys, z=zs, u=[0.0] * n, v=[0.0] * n, seg=sg, bundle=bd,
                    tx=[float(t) for t in tx], ty=[float(t) for t in ty],
                    color=['#999999'] * n, alpha=[0.85] * n, size=[4.0] * n,
                    hl=[0.0] * n)
    # everything else in the event, faint: the answer to "why is the star not on
    # the object".  evt396222 holds 180 segments; this shower is 123 of them, and
    # its nearest point is 14.5 cm from the vertex because the photon converted
    # there (start_connection_type=2, a gap).
    mine = set(row['segs'])
    cx_, cy_, cz_ = [], [], []
    for s_, A in P.items():
        if s_ in mine or A is None or not len(A):
            continue
        for k in range(len(A)):
            cx_.append(float(A[k, 0])); cy_.append(float(A[k, 1])); cz_.append(float(A[k, 2]))
    m = len(cx_)
    ctx.data = dict(x=cx_, y=cy_, z=cz_, u=[0.0] * m, v=[0.0] * m,
                    al=[0.30] * m, sz=[2.0] * m)
    vtx.data = dict(x=[float(v[0])], y=[float(v[1])], z=[float(v[2])],
                    u=[0.0], v=[0.0], al=[1.0], sz=[16.0])
    # em3d.bounding_sphere takes a LIST of 3-tuples and guards with `if not pts`;
    # an ndarray there raises "truth value of an array is ambiguous".
    allp = list(zip(xs, ys, zs)) + list(zip(cx_, cy_, cz_)) + [tuple(float(t) for t in v)]
    (ecx, ecy, ecz), R = em3d.bounding_sphere(allp)
    STATE['event_centre'] = (ecx, ecy, ecz)
    # DEFAULT ORBIT POINT IS THE NU VERTEX, not the bounding-sphere centre: every
    # angle the proposal is built from is measured from the vertex, so rotating
    # about it is what makes the theta-phi structure read correctly by eye.
    cam.data = dict(az=[0.6], el=[0.35],
                    cx=[float(v[0])], cy=[float(v[1])], cz=[float(v[2])], R=[R],
                    az0=[0.6], el0=[0.35], xs0=[-R], xe0=[R], ys0=[-R], ye0=[R])
    fig.x_range.start, fig.x_range.end = -R, R
    fig.y_range.start, fig.y_range.end = -R, R
    centre_sel.value = 'nu vertex'
    # the vertex-to-charge gap, surfaced because it is occasionally the story
    STATE['vgap'] = float(np.min(np.linalg.norm(
        np.asarray(list(zip(xs, ys, zs))) - np.asarray(v)[None, :], axis=1))) if n else float('nan')
    # Bee deep links for THIS event, from the sets already uploaded.  The owner
    # asked for these so the whole event can be understood before the divide --
    # the split tool shows one object, Bee shows everything around it.
    bee.text = ("<b>Bee:</b> " + BL.links_html(BEE, ev)) if BEE is not None else ''
    hi_box.value = ''
    refresh()
    curdoc().add_next_tick_callback(lambda: _kick())


def set_centre(p, keep_zoom=True):
    """Orbit about `p`.

    The projection is relative to (cx, cy, cz), so re-centring puts `p` at (0, 0)
    in view space and the ranges have to be re-centred on zero too.  They keep
    their CURRENT span, so the user's zoom survives a centre change -- that is
    the whole reason this is not just a camera write.  R is deliberately left
    alone: it is the zoom-independent scale the depth cue normalises by.
    (Mirrors em_display_viewer.set_centre:4971.)"""
    d = dict(cam.data)
    d['cx'] = [float(p[0])]; d['cy'] = [float(p[1])]; d['cz'] = [float(p[2])]
    if keep_zoom:
        sx = fig.x_range.end - fig.x_range.start
        sy = fig.y_range.end - fig.y_range.start
    else:
        sx = sy = 2.0 * d['R'][0]
    fig.x_range.start, fig.x_range.end = -0.5 * sx, 0.5 * sx
    fig.y_range.start, fig.y_range.end = -0.5 * sy, 0.5 * sy
    d['xs0'] = [fig.x_range.start]; d['xe0'] = [fig.x_range.end]
    d['ys0'] = [fig.y_range.start]; d['ye0'] = [fig.y_range.end]
    cam.data = d


def centre_point(which):
    row = STATE['row']
    if row is None:
        return (0.0, 0.0, 0.0)
    if which == 'object centroid':
        pts, q, _ = L_pack(row)
        return tuple(float(x) for x in SM.L.qw_centroid(pts, q)) if pts is not None \
            else tuple(float(x) for x in row['v'])
    if which == 'event centre':
        return STATE.get('event_centre') or tuple(float(x) for x in row['v'])
    return tuple(float(x) for x in row['v'])          # 'nu vertex', the default


def L_pack(row):
    return SM.L.pack(row['P'], row['segs'])


def on_centre(attr, old, new):
    set_centre(centre_point(new))


centre_sel.on_change('value', on_centre)


def _kick():
    """Force one projection pass and one tree bind after the document exists.

    Runs from add_next_tick_callback, which is the earliest point at which a
    Python-side assignment reaches the client as a CHANGE rather than as part of
    the initial serialisation."""
    cam.data = dict(cam.data)
    refresh()


# ----------------------------------------------------------------- labels
def label_path(ev, nd):
    return os.path.join(LABEL_DIR, 'labels-evt%d.json' % ev)


def read_label(ev, nd):
    p = label_path(ev, nd)
    if not os.path.exists(p):
        return None
    try:
        j = json.load(open(p))
    except Exception:
        return None
    return (j.get('split_labels') or {}).get(str(nd))


def save_label():
    p = STATE['payload']
    if p is None:
        return
    os.makedirs(LABEL_DIR, exist_ok=True)
    guard = os.path.join(LABEL_DIR, '.split_display_tag')
    if not os.path.exists(guard):
        # M13: refuse to write into a directory this tool did not create.
        if os.path.isdir(LABEL_DIR) and any(
                f.endswith('.json') for f in os.listdir(LABEL_DIR)):
            status.text = ("REFUSED: %s already holds labels and carries no\n"
                           ".split_display_tag marker.  Use a fresh --scan-tag "
                           "(CLAUDE.md M13)." % LABEL_DIR)
            return
        open(guard, 'w').write('doc pr/138 split_display; safe to append here\n')
    ev, nd = p['event'], p['node']
    path = label_path(ev, nd)
    j = {}
    if os.path.exists(path):
        try:
            j = json.load(open(path))
        except Exception:
            j = {}
    j.setdefault('event', 'evt%d' % ev)
    j.setdefault('scan_tag', args.scan_tag)
    j.setdefault('source', 'split_display (doc pr/138 Phase A)')
    j.setdefault('split_labels', {})
    grp = STATE['group']
    parts = collections.defaultdict(list)
    for s, g in grp.items():
        parts[str(g)].append(int(s))
    j['split_labels'][str(nd)] = dict(
        verdict=verdict_btn.labels[verdict_btn.active],
        n_parts=n_parts(),        # SPLIT4+ is a bucket; keep the exact count
        n_groups=STATE['ngroups'],
        confidence=conf_btn.labels[conf_btn.active],
        comment=note_box.value,
        groups={str(k): int(v) for k, v in grp.items()},
        parts={k: sorted(v) for k, v in parts.items()},
        proposal=p['reason'],
        nseg=p['nseg'], Q=p['Q'],
        saved=datetime.datetime.now().isoformat(timespec='seconds'))
    json.dump(j, open(path, 'w'), indent=1)
    status.text = "saved %s  (verdict %s)" % (path, verdict_btn.labels[verdict_btn.active])


# ----------------------------------------------------------------- callbacks
def on_moved(attr, old, new):
    if not new:
        return
    try:
        raw, g, _ = new.split('|')
        g = int(g)
        segs = [int(x) for x in raw.split(',') if x != '']
    except Exception:
        return
    for s in segs:
        STATE['group'][s] = g
    refresh()


moved_box.on_change('value', on_moved)

btn_prev = Button(label='< prev', width=90)
btn_next = Button(label='next >', width=90)
btn_save = Button(label='SAVE', button_type='success', width=110)
btn_reset = Button(label='reset to proposal', width=150)
btn_prev.on_click(lambda: load(STATE['i'] - 1))
btn_next.on_click(lambda: load(STATE['i'] + 1))
btn_save.on_click(save_label)


def do_reset():
    if STATE['row'] is None:
        return
    p = SM.object_payload(STATE['row'], gap=args.gap)
    STATE['payload'] = p
    STATE['group'] = {s['seg']: s['group'] for s in p['segs']}
    refresh()


btn_reset.on_click(do_reset)

btn_addg = Button(label='+ group', width=90)
btn_delg = Button(label='- group', width=90)


def add_group():
    """Owner: 'for busy events, there may be many groups.'  Three is the default;
    this grows the column list on demand, up to the palette."""
    if STATE['ngroups'] >= MAX_NGROUPS:
        status.text = "at the maximum of %d groups (palette size)" % MAX_NGROUPS
        return
    STATE['ngroups'] += 1
    refresh()


def del_group():
    """Only ever removes an EMPTY trailing column, so a click cannot silently
    reassign segments the scanner has already placed."""
    if STATE['ngroups'] <= 1:
        return
    last = STATE['ngroups'] - 1
    if any(g == last for g in STATE['group'].values()):
        status.text = ("Group %d is not empty -- drag its bundles out first.\n"
                       "Removing a populated column would silently reassign them."
                       % last)
        return
    STATE['ngroups'] -= 1
    refresh()


btn_addg.on_click(add_group)
btn_delg.on_click(del_group)

jump = Select(title='object', width=300,
              options=[("%d" % i, "evt%d node%d  %s  Q=%.2g" % (e, n, st, q))
                       for i, (e, n, q, st, px) in enumerate(WORK)],
              value='0')
jump.on_change('value', lambda a, o, n: load(int(n)))

left = column(info, bee,
              brow(btn_prev, btn_next, jump, btn_save),
              brow(verdict_btn), brow(conf_btn), note_box,
              brow(btn_reset, btn_addg, btn_delg, centre_sel),
              tree, status,
              gmap_box, cmap_box, hi_box, moved_box, cam_box)
right = column(fig, brow(tp, wp))
curdoc().add_root(brow(left, right))
curdoc().title = 'split_display -- doc pr/138'
if WORK:
    load(0)

#!/usr/bin/env python3
# doc pr/138 Phase A -- the drag-and-drop tree, as browser-side JS.
"""HTML5 drag-and-drop over a Bokeh Div, wired to the 3-D point source.

WHY NOT A BOKEH WIDGET.  Bokeh 3.9 has no tree and no drag-and-drop widget, and
the owner asked for dragging specifically -- a "select rows, press a Move button"
table is what em_display's mark in/out already is, and the complaint about it is
that it is "not very handy".  So the tree is raw HTML in a Div and the wiring is
CustomJS.

THE BINDING TRAP, stated once.  `div.js_on_change('text', cb)` fires when Python
assigns new text, but bokehjs has not necessarily painted the new innerHTML by
the time the callback runs.  Every bind below is therefore deferred with
requestAnimationFrame; binding synchronously attaches listeners to the PREVIOUS
tree and silently does nothing.

HONEST LIMIT, same as em3d.py's: there is no JS engine in this tree, so this file
is not machine-tested.  selftest_split_display.py covers the Python mirrors --
the payload, the proposal, the verdict derivation and the label round-trip -- and
the JS is covered by the manual check-list in doc pr/138 section A1.
"""

# Bokeh 3 renders EVERY view inside an open shadow root, so
# `document.getElementById('split-tree')` finds nothing at all -- the first build
# of this file silently bound zero handlers and the tree simply was not
# draggable, with no console error to say so.  em_display's
# selftest_em3d_browser.py:136-150 documents the same trap for canvases.  Every
# top-level lookup below goes through this walk.
JS_FIND = r"""
const _walkAll = (sel) => {
    const out = [];
    const walk = (root) => {
        for (const el of root.querySelectorAll('*')) {
            if (el.matches && el.matches(sel)) out.push(el);
            if (el.shadowRoot) walk(el.shadowRoot);
        }
    };
    walk(document);
    return out;
};
const _tree = () => { const a = _walkAll('#split-tree'); return a.length ? a[0] : null; };
const _cols = () => { const t = _tree(); return t ? t.querySelector('.cols') : null; };
"""

# DELEGATION, not per-node binding.  Two earlier builds bound listeners to each
# card from `div.js_on_change('text', ...)` and bound ZERO both times: that
# callback is not a reliable channel for a Div whose text is set during document
# construction, and Bokeh 3's open shadow roots hide the nodes from
# document.getElementById on top of it.  So instead: ONE set of listeners on the
# document, attached once, that finds the card with composedPath()[0].closest().
# Drag and click events are composed:true, so they cross the shadow boundary --
# but `event.target` is RETARGETED to the shadow host, which is why the path is
# read instead of the target.  `draggable="true"` is emitted by Python into the
# HTML, so no DOM write is needed to make a card draggable.
# JS_FIND is PREPENDED, not assumed.  The drag auto-scroll below calls _cols(),
# which lives in JS_FIND -- and JS_SETUP was previously self-contained, so
# omitting this would have thrown ReferenceError on the first dragover with no
# other symptom than "drag stopped working".  selftest_split_display.py now
# lints every assembled blob for exactly that (helper used but not defined, and
# the mirror-image trap: the same `const` declared twice by double-concatenation).
JS_SETUP = JS_FIND + r"""
if (!window.__pr138_bound) {
    window.__pr138_bound = true;
    const src = (ev) => {
        const p = ev.composedPath ? ev.composedPath() : [ev.target];
        for (const n of p) {
            if (n && n.getAttribute && n.hasAttribute && n.hasAttribute('data-drag')) return n;
        }
        return null;
    };
    const zone = (ev) => {
        const p = ev.composedPath ? ev.composedPath() : [ev.target];
        for (const n of p) {
            if (n && n.hasAttribute && n.hasAttribute('data-group')) return n;
        }
        return null;
    };
    document.addEventListener('dragstart', (ev) => {
        const n = src(ev);
        if (!n) { return; }
        ev.dataTransfer.setData('text/plain', n.getAttribute('data-segs'));
        ev.dataTransfer.effectAllowed = 'move';
        n.classList.add('dragging');
        window.__pr138_payload = n.getAttribute('data-segs');
    }, true);
    document.addEventListener('dragend', (ev) => {
        const n = src(ev); if (n) n.classList.remove('dragging');
    }, true);
    document.addEventListener('dragover', (ev) => {
        // AUTO-SCROLL.  With the columns at a fixed width the row scrolls, so a
        // drop target can be off-screen -- and a drag that cannot reach its
        // target is worse than narrow columns.  Nudge the row whenever the
        // pointer is within 60 px of either edge.
        const row = _cols();
        if (row != null) {
            const r = row.getBoundingClientRect();
            if (ev.clientX > r.right - 60) row.scrollLeft += 24;
            else if (ev.clientX < r.left + 60) row.scrollLeft -= 24;
        }
        const z = zone(ev); if (!z) { return; }
        ev.preventDefault(); z.classList.add('over');
    }, true);
    document.addEventListener('dragleave', (ev) => {
        const z = zone(ev); if (z) z.classList.remove('over');
    }, true);
    document.addEventListener('drop', (ev) => {
        const z = zone(ev); if (!z) { return; }
        ev.preventDefault(); z.classList.remove('over');
        // Safari/webkit can hand back an empty dataTransfer across a shadow
        // boundary; the dragstart stash is the fallback.
        let raw = '';
        try { raw = ev.dataTransfer.getData('text/plain'); } catch (e) { raw = ''; }
        if (!raw) { raw = window.__pr138_payload || ''; }
        if (!raw) { return; }
        const g = parseInt(z.getAttribute('data-group'), 10);
        window.__pr138_moved.value = raw + '|' + g + '|' + Date.now();
    }, true);
    document.addEventListener('click', (ev) => {
        const n = src(ev); if (!n) { return; }
        const segs = n.getAttribute('data-segs').split(',').map(Number);
        window.__pr138_hi.value = JSON.stringify(segs) + '|' + Date.now();
    }, true);
}
window.__pr138_moved = moved;
window.__pr138_hi = hi;
"""

# Recolour the 3-D cloud, without a server round trip.
#
# THREE MODES, ONE CHANNEL.  The owner reported "the color of the group is gone,
# no red vs blue" -- measured, and it is not a rendering fault: 39 of the 50
# curated objects get a SINGLE-group proposal (doc pr/138 sec A1.5), so the group
# colouring has nothing to say and the cloud is uniformly group-0 blue.  The
# answer is to give the eye a second and third thing to look at: `bundle` (the
# unit the owner actually drags -- the tree lists 42 of them and they were
# visually identical) and `charge` (the valley the split criterion is looking
# for, which is a charge dip by construction).
#
# The mode arrives as a reserved "_mode" key inside the EXISTING cmap payload,
# not as a new widget callback.  Four earlier builds lost handlers to Bokeh 3
# binding traps; the rule now is that a working channel gets reused rather than
# duplicated.  "_mode" cannot collide with a group key -- those are str(int).
JS_RECOLOR_BODY = r"""
const d = cloud.data;
const map = JSON.parse(gmap.value || '{}');      // seg -> group
const col = JSON.parse(cmap.value || '{}');      // group -> css colour, + _mode
const mode = col['_mode'] || 'group';
const n = d.seg.length;
for (let i = 0; i < n; i++) {
    if (mode === 'bundle' && d.bcolor !== undefined) {
        d.color[i] = d.bcolor[i];
    } else if (mode === 'charge' && d.qcolor !== undefined) {
        d.color[i] = d.qcolor[i];
    } else {
        const g = map[d.seg[i]];
        d.color[i] = (g === undefined) ? '#999999' : (col[g] || '#999999');
    }
}
cloud.change.emit();
"""

# THE POINT STYLE, in one place.  Owner: "the transparency of the point can be
# less, and thicker in the 3D view, so that things can be viewed clearer."  These
# five numbers are the only copy -- split_viewer imports PT_ALPHA/PT_SIZE for the
# initial cloud and the highlight below formats itself from the same constants,
# because the previous build had the base values written twice (Python 0.85/4.0,
# JS 0.85/4.0) and nothing kept them in step.
PT_ALPHA     = 1.00    # was 0.85 -- opaque; overlapping points no longer wash out
PT_SIZE      = 6.0     # was 4.0
PT_HL_SIZE   = 10.0    # the tapped segment, was 7.0
PT_DIM_ALPHA = 0.18    # everything else while one is picked, was 0.12
PT_DIM_SIZE  = 3.0     # was 2.5

PT_ALPHA_FMT, PT_SIZE_FMT = repr(PT_ALPHA), repr(PT_SIZE)
PT_HL_SIZE_FMT = repr(PT_HL_SIZE)
PT_DIM_ALPHA_FMT, PT_DIM_SIZE_FMT = repr(PT_DIM_ALPHA), repr(PT_DIM_SIZE)

# Highlight one segment set: boost size and alpha, dim the rest.  Purely local.
JS_HIGHLIGHT = JS_FIND + ("""
const d = cloud.data;
const raw = (hi.value || '').split('|')[0];
let sel = [];
try { sel = JSON.parse(raw) || []; } catch (e) { sel = []; }
const S = new Set(sel);
const n = d.seg.length;
const on = S.size > 0;
for (let i = 0; i < n; i++) {
    const m = S.has(d.seg[i]);
    d.hl[i] = (on && m) ? 1.0 : 0.0;
    d.alpha[i] = on ? (m ? %(a)s : %(da)s) : %(a)s;
    d.size[i]  = on ? (m ? %(hs)s : %(ds)s)  : %(s)s;
}
cloud.change.emit();
// mirror the selection into the tree
const el = _tree();
if (el != null) {
    el.querySelectorAll('[data-drag]').forEach(node => {
        const segs = node.getAttribute('data-segs').split(',').map(Number);
        const hit = segs.some(s => S.has(s));
        node.classList.toggle('picked', hit);
    });
}
""" % dict(a=PT_ALPHA_FMT, da=PT_DIM_ALPHA_FMT, hs=PT_HL_SIZE_FMT,
           ds=PT_DIM_SIZE_FMT, s=PT_SIZE_FMT))

# Tap in the 3-D view -> select that segment (and scroll its card into view).
JS_TAP = JS_FIND + r"""
const inds = cloud.selected.indices;
if (inds == null || inds.length === 0) { return; }
const seg = cloud.data.seg[inds[0]];
hi.value = JSON.stringify([seg]) + '|' + Date.now();
requestAnimationFrame(() => {
    const el = _tree();
    if (el == null) { return; }
    const card = el.querySelector('[data-seg="' + seg + '"]');
    if (card != null) { card.scrollIntoView({block: 'nearest'}); }
});
"""

# TREE_W is the ONE place the tree's width is decided.  split_viewer passes the
# same number to the Div, so the HTML box and the box Bokeh reserves in the
# layout cannot drift apart -- and drifting apart is exactly the bug the owner
# hit: at 6 groups the columns grew the tree's content to 838 px while Bokeh had
# reserved 780 and had already placed the 3-D canvas at x=790, so the columns
# painted over the 3-D view.
TREE_W = 800
# how wide one drop column stays, however many there are.  FIXED, not flexible:
# `flex: 1 1 0` divided the 780 px among the columns and at 6 groups they were
# 115 px wide -- too narrow to read a segment row.  Now the columns keep a
# readable width and the ROW scrolls, which is what "the group can be a lot"
# actually needs.
COL_W = 186

CSS = ("""
<style>
/* box-sizing is load-bearing.  Each .col carries 1 px of border and 4 px of
   padding a side; under the default content-box those 10 px are added ON TOP of
   the flex basis, which is how seven columns overflowed an 800 px box. */
#split-tree, #split-tree * { box-sizing: border-box; }
#split-tree { font-family: system-ui, sans-serif; font-size: 10.5px;
              width: %dpx; max-width: %dpx; overflow: hidden; }
#split-tree .colhdr { font-size: 11px; }
/* the row scrolls; it never grows past the Div and never reaches the 3-D view */
#split-tree .cols { display: flex; gap: 5px; align-items: stretch;
                    flex-wrap: nowrap; overflow-x: auto; overflow-y: hidden;
                    padding-bottom: 4px; }
#split-tree .col  { flex: 0 0 %dpx; width: %dpx; min-width: %dpx;
                    border: 1px solid #ccc;
                    border-radius: 5px; padding: 4px; background: #fafafa;
                    max-height: 600px; overflow-y: auto; overflow-x: hidden; }""" %
       (TREE_W, TREE_W, COL_W, COL_W, COL_W) + """
#split-tree .col.over { background: #e8f0ff; border-color: #4a80d0; }
/* the horizontal scrollbar is the only cue that more columns exist, so it is
   forced visible rather than left to the browser's overlay behaviour */
#split-tree .cols { scrollbar-width: thin; scrollbar-color: #b0b0b0 #ececec; }
#split-tree .cols::-webkit-scrollbar { height: 11px; }
#split-tree .cols::-webkit-scrollbar-track { background: #ececec; border-radius: 6px; }
#split-tree .cols::-webkit-scrollbar-thumb { background: #b0b0b0; border-radius: 6px; }
#split-tree .colhdr { font-weight: 600; padding: 2px 4px; margin-bottom: 4px;
                      border-radius: 3px; color: #fff; }
#split-tree .bundle { border: 1px solid #bbb; border-radius: 4px; margin: 3px 0;
                      background: #fff; cursor: grab; }
#split-tree .bundle.dragging { opacity: 0.45; }
#split-tree .bundle.picked  { outline: 2px solid #111; }
#split-tree .bhdr { padding: 3px 5px; font-weight: 600; border-bottom: 1px solid #eee; }
#split-tree .seg  { padding: 2px 4px 2px 10px; cursor: grab;
                    border-top: 1px solid #f2f2f2; white-space: nowrap;
                    overflow: hidden; text-overflow: ellipsis; }
#split-tree .seg.picked { background: #ffe9a8; outline: 1px solid #111; }
#split-tree .seg.dragging { opacity: 0.45; }
#split-tree .muted { color: #888; }
</style>
""")

JS_RECOLOR = JS_SETUP + JS_RECOLOR_BODY

#!/usr/bin/env python3
"""Drive the nusel hand-scan viewer headlessly and grab per-bundle evidence.

For every in-beam bundle of every event served by the viewer on --port:
  <out>/<evt>-r<row>-ctx.png    the 3 charge projections, full-detector range
                                (containment / FV / where in the TPC)
  <out>/<evt>-r<row>-zoom.png   same 3 projections zoomed to the bundle's own
                                bounding box (topology: kinks, branches, ends)
  <out>/<evt>-r<row>-dqdx.png   the STM dQ/dx-vs-residual-range panel
  <out>/info.json               per bundle: the viewer's own text panels
                                (proj_info / status / metrics / stm_info),
                                the table row, and the bundle bbox

Read-only w.r.t. the scan record: it never clicks a label button, a comment box
or Save, so no nusel_labels/<tag>/ file is written (only label/comment handlers
call save_state()).  Point it at a viewer instance of your own anyway.
"""
import argparse
import json
import os
from playwright.sync_api import sync_playwright

AP = argparse.ArgumentParser()
AP.add_argument('--port', type=int, default=5019)
AP.add_argument('--out', required=True)
AP.add_argument('--events', nargs='*', help='subset of dropdown labels')
AP.add_argument('--pad-frac', type=float, default=0.12)
AP.add_argument('--min-span', type=float, default=25.0, help='cm')
args = AP.parse_args()
os.makedirs(args.out, exist_ok=True)

URL = f'http://localhost:{args.port}/nusel_scan_viewer'

# --- JS helpers -------------------------------------------------------------
JS_DIVS = """() => {
  const out = [];
  for (const m of Bokeh.documents[0]._all_models.values())
    if (m.type === 'Div') out.push(m.text || '');
  return out;
}"""

JS_BBOX = """() => {
  // union bbox of the focused bundle's own points: the three scatter sources
  // that carry 'cid' (main / main-fragments / companions), plus the STM
  // trajectory (x,y,z, no cid).  The gray whole-event context layer also has
  // (x,y,z) but is the FIRST such source, so it is skipped by cid-matching.
  let lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9], n = 0;
  for (const m of Bokeh.documents[0]._all_models.values()) {
    if (!(m.type === 'Figure' || m.type === 'Plot')) continue;
    if (!(m.title && m.title.text === 'X-Y')) continue;
    for (const r of m.renderers) {
      let d;
      try { d = r.data_source.data; } catch (e) { continue; }
      if (d.cid === undefined) continue;
      const xs = d.x || [], ys = d.y || [], zs = d.z || [];
      for (let i = 0; i < xs.length; i++) {
        const v = [xs[i], ys[i], zs[i]];
        for (let k = 0; k < 3; k++) {
          if (v[k] < lo[k]) lo[k] = v[k];
          if (v[k] > hi[k]) hi[k] = v[k];
        }
        n++;
      }
    }
  }
  return {lo: lo, hi: hi, n: n};
}"""

JS_SETRANGE = """([lo, hi, padf, minspan]) => {
  const axes = {'X-Y': [0, 1], 'Y-Z': [2, 1], 'X-Z': [0, 2]};   // [horiz, vert]
  for (const m of Bokeh.documents[0]._all_models.values()) {
    if (!(m.type === 'Figure' || m.type === 'Plot')) continue;
    const t = m.title && m.title.text;
    if (!(t in axes)) continue;
    const [ih, iv] = axes[t];
    const put = (rng, i) => {
      let a = lo[i], b = hi[i];
      const span = Math.max(b - a, minspan);
      const c = 0.5 * (a + b), pad = padf * span;
      rng.setv({start: c - 0.5 * span - pad, end: c + 0.5 * span + pad});
    };
    put(m.x_range, ih);
    put(m.y_range, iv);
  }
  return true;
}"""

JS_RESET = """() => {
  for (const m of Bokeh.documents[0]._all_models.values())
    if ((m.type === 'Figure' || m.type === 'Plot') &&
        m.title && ['X-Y', 'Y-Z', 'X-Z'].includes(m.title.text)) m.reset.emit();
  return true;
}"""

PROJ_CLIP = dict(x=0, y=0, width=1200, height=332)
DQDX_CLIP = dict(x=0, y=922, width=1170, height=302)   # f_dqdx + stm_info text


def rows_of(page):
    return [[c.inner_text().strip() for c in r.query_selector_all('.slick-cell')]
            for r in page.query_selector_all('.slick-row')]


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={'width': 1720, 'height': 1800},
                      device_scale_factor=2)
    errs = []
    page.on('pageerror', lambda e: errs.append(str(e)))
    page.goto(URL, timeout=180000)
    page.wait_for_selector('.slick-row', timeout=180000)
    page.wait_for_timeout(3000)

    labels = page.eval_on_selector('select >> nth=0',
                                   's => [...s.options].map(o => o.value)')
    print(f'{len(labels)} events in dropdown; first={labels[0]} last={labels[-1]}')
    todo = args.events or labels

    # IN-BEAM only mode (global viewer state, set once)
    page.click("button:has-text('Mode: ALL bundles')")
    page.wait_for_timeout(1500)
    assert page.query_selector("button:has-text('Mode: IN-BEAM only')"), 'mode toggle failed'

    out = []
    for evt in todo:
        page.select_option('select >> nth=0', evt)
        page.wait_for_timeout(2500)
        rows = rows_of(page)
        print(f'{evt}: {len(rows)} in-beam row(s)')
        for i in range(len(rows)):
            page.query_selector_all('.slick-row')[i].click()
            page.wait_for_timeout(2200)
            divs = page.evaluate(JS_DIVS)
            bbox = page.evaluate(JS_BBOX)
            base = f'{args.out}/{evt}-r{i}'
            page.screenshot(path=f'{base}-ctx.png', clip=PROJ_CLIP)
            if bbox['n']:
                page.evaluate(JS_SETRANGE, [bbox['lo'], bbox['hi'],
                                            args.pad_frac, args.min_span])
                page.wait_for_timeout(1200)
                page.screenshot(path=f'{base}-zoom.png', clip=PROJ_CLIP)
                page.evaluate(JS_RESET)
                page.wait_for_timeout(600)
            page.screenshot(path=f'{base}-dqdx.png', clip=DQDX_CLIP)
            out.append(dict(event=evt, row=i, table_row=rows_of(page)[i],
                            bbox=bbox, divs=divs))
            print(f'   r{i}: {rows[i][:6]} npts={bbox["n"]}')
    with open(f'{args.out}/info.json', 'w') as f:
        json.dump(dict(labels=labels, bundles=out), f, indent=1)
    print('js errors:', errs[:5])
    b.close()
print('wrote', args.out)

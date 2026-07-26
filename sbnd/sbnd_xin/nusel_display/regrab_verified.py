#!/usr/bin/env python3
"""Re-grab specific bundles, VERIFYING that the display actually focused them.

grab_scan_shots.py clicks row i and waits; on two events the click did not take
and both rows' screenshots showed the LAST row's bundle.  Here every row is
focused by clicking a *different* row first and then the target, and the bundle
info div is checked to name the expected main id before anything is written.

Usage: regrab_verified.py --port P --out DIR --want evt400174:8 evt62495:16
"""
import argparse
import json
import os
import re
from playwright.sync_api import sync_playwright

AP = argparse.ArgumentParser()
AP.add_argument('--port', type=int, required=True)
AP.add_argument('--out', required=True)
AP.add_argument('--want', nargs='+', required=True, help='evt<ID>:<main_id>')
args = AP.parse_args()

JS_DIVS = """() => {
  const out = [];
  for (const m of Bokeh.documents[0]._all_models.values())
    if (m.type === 'Div') out.push(m.text || '');
  return out;
}"""
# reuse the grabber's bbox / range JS verbatim, so the two agree by construction
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'grab_scan_shots.py')).read()
JS_BBOX = SRC.split('JS_BBOX = """')[1].split('"""')[0]
JS_SETRANGE = SRC.split('JS_SETRANGE = """')[1].split('"""')[0]
JS_RESET = SRC.split('JS_RESET = """')[1].split('"""')[0]
PROJ_CLIP = dict(x=0, y=0, width=1200, height=332)
DQDX_CLIP = dict(x=0, y=922, width=1170, height=302)

want = {}
for w in args.want:
    e, m = w.split(':')
    want.setdefault(e, []).append(m)

out = []
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={'width': 1720, 'height': 1800},
                      device_scale_factor=2)
    errs = []
    page.on('pageerror', lambda e: errs.append(str(e)))
    page.goto(f'http://localhost:{args.port}/nusel_scan_viewer', timeout=180000)
    page.wait_for_selector('.slick-row', timeout=180000)
    page.wait_for_timeout(3000)
    if page.query_selector("button:has-text('Mode: ALL bundles')"):
        page.click("button:has-text('Mode: ALL bundles')")
        page.wait_for_timeout(1500)

    for evt, mains in want.items():
        page.select_option('select >> nth=0', evt)
        page.wait_for_timeout(3000)
        rows = [[c.inner_text().strip() for c in r.query_selector_all('.slick-cell')]
                for r in page.query_selector_all('.slick-row')]
        print(f'{evt}: {len(rows)} rows, mains on screen '
              f'{[r[5] for r in rows if len(r) > 5]}')
        for main in mains:
            idx = [i for i, r in enumerate(rows) if len(r) > 5 and r[5] == main]
            assert idx, f'{evt}: no row with main {main}'
            i = idx[0]
            ok = False
            for attempt in range(4):
                other = (i + 1) % len(rows)
                if other != i:
                    page.query_selector_all('.slick-row')[other].click()
                    page.wait_for_timeout(1500)
                page.query_selector_all('.slick-row')[i].click()
                page.wait_for_timeout(3000)
                divs = page.evaluate(JS_DIVS)
                head = next((d for d in divs if 'bundle</b>' in d), '')
                m = re.search(r'main\s*<span[^>]*><b>(\d+)</b>', head) or \
                    re.search(r'main\s+(\d+)', head)
                got = m.group(1) if m else None
                print(f'   try {attempt}: row {i} -> focused main {got} '
                      f'(want {main})')
                if got == main:
                    ok = True
                    break
            assert ok, f'{evt} main {main}: focus never took'
            bbox = page.evaluate(JS_BBOX)
            base = f'{args.out}/{evt}-m{main}'
            page.screenshot(path=f'{base}-ctx.png', clip=PROJ_CLIP)
            if bbox['n']:
                page.evaluate(JS_SETRANGE, [bbox['lo'], bbox['hi'], 0.12, 25.0])
                page.wait_for_timeout(1200)
                page.screenshot(path=f'{base}-zoom.png', clip=PROJ_CLIP)
                page.evaluate(JS_RESET)
                page.wait_for_timeout(600)
            page.screenshot(path=f'{base}-dqdx.png', clip=DQDX_CLIP)
            out.append(dict(event=evt, main=main, bbox=bbox, divs=divs,
                            table_row=rows[i]))
            print(f'   wrote {base}-*.png  drawn points={bbox["n"]}')
    with open(f'{args.out}/info.json', 'w') as f:
        json.dump(dict(bundles=out), f, indent=1)
    print('js errors:', errs[:5])
    b.close()

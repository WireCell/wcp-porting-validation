"""doc pdvd/120: live check of the uploaded beam Bee set -- per event, the PR layers' point counts (after
selecting them), the q=15000 main vertex and the particle-flow tree root text as rendered by the BNL page.
Usage: xvfb-run -a python3 bee_pf_check.py <idx> [idx ...]   (SET hard-coded to the doc-120 set)"""
import asyncio, json, sys
from playwright.async_api import async_playwright
SET = "https://www.phy.bnl.gov/twister/bee/set/68caddae-7c7a-45b6-9312-54ddfdd5fe6d/event/"
JS = """() => {
  const L = window.bee.sst.list, out = {layers: {}, bee_keys: Object.keys(window.bee)};
  for (const n of Object.keys(L)) { const d = L[n].data; out.layers[n] = d && d.x ? d.x.length : (d ? 'nodata' : 'none'); }
  const v = L['vertices-global']; if (v && v.data && v.data.q) { out.main_vertex = []; for (let i = 0; i < v.data.q.length; i++) if (v.data.q[i] == 15000) out.main_vertex.push([v.data.x[i], v.data.y[i], v.data.z[i]]); }
  out.pf_text = Array.from(document.querySelectorAll('li, a, span, div')).map(e => e.textContent.trim()).filter(s => /^(reco nu|nu$|mu-|proton|gamma|e-|neutron|pi)/.test(s) && s.length < 60).slice(0, 12);
  return out;
}"""
async def run(idxs):
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False)
        pg = await b.new_page(ignore_https_errors=True, viewport={"width": 1600, "height": 900})
        for idx in idxs:
            await pg.goto(SET + f"{idx}/", wait_until="networkidle", timeout=120000)
            await pg.wait_for_function("() => window.bee && window.bee.sst && window.bee.sst.list", timeout=90000)
            for lay in ("track_fit-global", "shower_track-global", "vertices-global"):
                ok = await pg.evaluate("(n) => { const s = window.bee.sst.list[n]; if (!s) return false; s.selected(); return true; }", lay)
                if ok: await pg.wait_for_timeout(2500)
            # open the MC / particle-flow panel if the UI has one
            for sel in ("#mc-tree", ".mc-tree", "#pf-tree", "[id*=mc]"):
                try:
                    if await pg.locator(sel).count(): break
                except Exception: pass
            await pg.wait_for_timeout(1500)
            r = await pg.evaluate(JS)
            print(f"idx {idx}: {json.dumps(r)}", flush=True)
            await pg.screenshot(path=f"/home/xqian/tmp/d120_pw/set2_ev{idx}.png")
        await b.close()
asyncio.run(run(sys.argv[1:]))

"""doc pdvd/119 sec 8: check the Bee PDVD side panel on the live Bee page.
- PATCH=1 swaps in the LOCAL wire-cell-bee3 source of ProtoDUNEVD.detectorFrameCorrection,
  Experiment/ProtoDUNEVD.layerInDetectorFrame and SST.drawDetectorFrame (the served page is a
  parcel bundle, so the methods are replaced on the live objects, verbatim from the source files).
- ZIP=<zip> serves that zip's per-event json instead of the uploaded set's (route interception),
  so a not-yet-uploaded zip can be viewed.
Prints, per event: the side-panel x range of the beam flash's matched clusters on both layers,
and (all matched clusters) the img-layer side-panel x vs the clustering layer's x_t0cor.
Usage: [PATCH=1] [ZIP=...] python3 bee_side_check.py <idx> [idx ...]"""
import asyncio, json, os, re, sys, zipfile
from playwright.async_api import async_playwright
SET = os.environ.get("SET", "https://www.phy.bnl.gov/twister/bee/set/a6a48e04-e91e-4a82-8e92-5b1331a3137c/event/")
BEE = "/home/xqian/toolkit-dev/toolkit/wire-cell-bee3/events/static/js/bee/physics/"
PATCH = os.environ.get("PATCH") == "1"
os.makedirs("/home/xqian/tmp/d119_pw", exist_ok=True)
ZIP = os.environ.get("ZIP")

def method(src, sig, after=0):
    i = src.index(sig, after)
    j = src.index("{", i)
    d = 0
    for k in range(j, len(src)):
        d += {"{": 1, "}": -1}.get(src[k], 0)
        if d == 0:
            return src[j + 1:k]
    raise RuntimeError(sig)

exp_src = open(BEE + "experiment.js").read().replace("\r\n", "\n")
sst_src = open(BEE + "sst.js").read().replace("\r\n", "\n")
vd = exp_src.index("class ProtoDUNEVD extends Experiment")
PATCHES = {
    "dfc": method(exp_src, "detectorFrameCorrection(sst, op) {", vd),
    "lidf": method(exp_src, "layerInDetectorFrame(sst) {", vd),
    "ddf": method(sst_src, "drawDetectorFrame() {"),
}

JS_PATCH = """(P) => {
  const exp = window.bee.op.store.experiment;
  exp.detectorFrameCorrection = new Function('sst', 'op', P.dfc);
  exp.layerInDetectorFrame = new Function('sst', P.lidf);
  for (const n of Object.keys(window.bee.sst.list)) {
    Object.getPrototypeOf(window.bee.sst.list[n]).drawDetectorFrame = new Function(P.ddf);
  }
}"""

JS_MEASURE = """() => {
  const op = window.bee.op, exp = op.store.experiment, L = window.bee.sst.list;
  const img = L['img-global'], clu = L['clustering-global'];
  const res = {flash: op.currentFlash, status: $('#statusbar').text(),
               has_anodes: op.data.op_cluster_anodes != null};
  // 1. what the side panel drew for the current layer (after '/': the beam flash's clusters)
  const pc = window.bee.current_sst.pointCloudDetector;
  if (pc) { const a = pc.geometry.attributes.position.array; let mn = 1e9, mx = -1e9;
    for (let i = 0; i < a.length; i += 3) { mn = Math.min(mn, a[i]); mx = Math.max(mx, a[i]); }
    res.drawn = [window.bee.current_sst.name, a.length / 3, +mn.toFixed(1), +mx.toFixed(1)]; }
  // 2. every matched cluster: img-layer side-panel x vs clustering-layer x (point aligned?)
  if (img && img.data && clu && clu.data) {
    const n = img.data.x.length; res.aligned = (n === clu.data.x.length);
    if (res.aligned) for (let i = 0; i < n; i += 97) if (img.data.y[i] !== clu.data.y[i] || img.data.z[i] !== clu.data.z[i]) { res.aligned = false; break; }
    const corr = exp.detectorFrameCorrection(img, op);
    let nc = 0, npt = 0, worst = 0, bottom = 0, bad = [];
    if (corr && res.aligned) for (const [k, c] of corr) {
      nc++; if (exp.driftDir(c.tpc) > 0) bottom++;
      let w = 0;
      for (let i = 0; i < n; i++) { if (Number(img.data.cluster_id[i]) !== k) continue;
        const xs = img.data.x[i] - exp.driftVelocityForTPC(c.tpc) * c.t * exp.driftDir(c.tpc);
        w = Math.max(w, Math.abs(xs - clu.data.x[i])); npt++; }
      worst = Math.max(worst, w); if (w > 0.01) bad.push([k, c.tpc, +w.toFixed(2)]);
    }
    res.img_vs_clu = {clusters: nc, points: npt, bottom_volume: bottom, max_abs_dx_cm: +worst.toFixed(4), bad: bad.slice(0, 8)};
  }
  return res;
}"""

async def run(idxs):
    zf = zipfile.ZipFile(ZIP) if ZIP else None
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False)
        pg = await b.new_page(ignore_https_errors=True, viewport={"width": 1600, "height": 900})
        if zf:
            async def handler(route):
                m = re.search(r"/event/(\d+)/([^/]+)/$", route.request.url)
                if m and m.group(2) in ("op", "img-global", "clustering-global"):
                    name = f"data/{m.group(1)}/{m.group(1)}-{m.group(2)}.json"
                    return await route.fulfill(status=200, content_type="application/json", body=zf.read(name))
                await route.continue_()
            await pg.route(re.compile(r".*/event/\d+/[^/]+/$"), handler)
        for idx in idxs:
            await pg.goto(SET + f"{idx}/", wait_until="networkidle", timeout=120000)
            await pg.wait_for_timeout(6000)
            await pg.evaluate("(n) => { window.bee.sst.list[n].selected(); }", "img-global")
            await pg.wait_for_timeout(5000)
            if PATCH:
                await pg.evaluate(JS_PATCH, PATCHES)
            await pg.evaluate("() => { window.bee.op.store.config.op.sidePanel = true; }")
            await pg.mouse.click(700, 450)
            for lay in ("clustering-global", "img-global"):
                await pg.evaluate("(n) => { window.bee.sst.list[n].selected(); }", lay)
                await pg.wait_for_timeout(1500)
                await pg.keyboard.press("/"); await pg.wait_for_timeout(1500)
                r = await pg.evaluate(JS_MEASURE)
                print(f"idx {idx} layer {lay}: {json.dumps(r)}", flush=True)
                await pg.screenshot(path=f"/home/xqian/tmp/d119_pw/chk_{'P' if PATCH else 'L'}{'Z' if ZIP else ''}_ev{idx}_{lay}.png")
        await b.close()

asyncio.run(run(sys.argv[1:]))

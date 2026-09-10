#!/usr/bin/env python3
"""doc pdvd/69 -- the causal control for scan_harness.py's WebGL-loss record.

    ./selftest_webgl_loss.py [--det pdvd] [--items K1,K2,K3] [--work DIR]

Two `shots` runs over the same three items, each in its own browser, on its own
scratch port (never :5017), with labels in a scratch --labeldir:

  CONTROL  nothing forced.  OUT/_webgl_lost.txt must exist and name no item.
  FORCED   every WebGL context in the page is lost (WEBGL_lose_context) right
           after the harness switches to item 2 -- the exact failure five
           concurrent browsers produce, made on purpose.  Items 2 and 3 must be
           named (item 1 too, by the harness's one-item margin), and a loss that
           could not be forced is a FAIL, not a pass.

check_shots.py then runs on both dirs.  Its colour verdict is printed and not
asserted: it says whether the forced loss spoiled the pixels as well.

Both arms load the same init script (it records every WebGL context the page
creates, so the forced arm can reach Bokeh's) and reload, so the only
difference between them is the forcing.  Writes only under --work.

The browser is started WITHOUT $DISPLAY (see --keep-display).  On wcgpu1 an
unreachable forwarded DISPLAY stops chromium's SwiftShader from starting, and
Bokeh then draws the 3-D panel with Canvas2D: no WebGL, so nothing to lose and
nothing to test (doc pdvd/69 sec 8.3).

Exit 0 PASS, 1 FAIL, 3 CANNOT TEST (the 3-D panel was not on WebGL).
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_harness as sh                                    # noqa: E402

HOOK = """(() => {
  window.__glctx = [];
  const wrap = (proto) => {
    if (!proto || !proto.getContext) return;
    const orig = proto.getContext;
    proto.getContext = function (type, ...rest) {
      const c = orig.call(this, type, ...rest);
      if (c && /webgl/i.test(String(type)) && window.__glctx.indexOf(c) < 0)
        window.__glctx.push(c);
      return c;
    };
  };
  wrap(window.HTMLCanvasElement && HTMLCanvasElement.prototype);
  wrap(window.OffscreenCanvas && OffscreenCanvas.prototype);
})();"""

LOSE = """() => {
  let n = 0;
  for (const g of (window.__glctx || [])) {
    const e = g.getExtension('WEBGL_lose_context');
    if (e && !g.isContextLost()) { e.loseContext(); n++; }
  }
  return n;
}"""


def run(det, keys, out, labeldir, force):
    app = sh.App(det, "blank", labeldir)
    try:
        app.page.add_init_script(HOOK)
        app.page.reload(wait_until="networkidle", timeout=180000)
        app.page.wait_for_timeout(2500)
        forced = []
        if force:
            orig = app.goto

            def goto(key, *a, **k):
                orig(key, *a, **k)
                if key == keys[1] and not forced:
                    forced.append(app.page.evaluate(LOSE))
            app.goto = goto
        sh.do_shots(app, keys, out)
        return dict(forced=forced, suspect=list(app.gl_suspect),
                    backend=getattr(app, "backend_3d", None),
                    events=[e[:90] for e in app.gl_events[:3]],
                    n_ctx=app.page.evaluate("() => (window.__glctx || []).length"))
    finally:
        app.close()


def lost_file_keys(out):
    fp = os.path.join(out, sh.LOST_FILE)
    if not os.path.exists(fp):
        return None
    return [l.strip() for l in open(fp) if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--det", default="pdvd")
    ap.add_argument("--items", default="039252_0/77,039252_15/77,039252_15/91")
    ap.add_argument("--work", default=os.path.join(os.environ["HOME"], "tmp",
                                                   "selftest_webgl_%d" % os.getpid()))
    ap.add_argument("--keep-display", action="store_true",
                    help="leave $DISPLAY set for the browser.  On this host an "
                         "unreachable DISPLAY stops chromium's SwiftShader from "
                         "starting (xcb_connect fails), Bokeh falls back to "
                         "Canvas2D and there is no context to lose -- so by "
                         "default it is cleared, and this flag shows the fallback "
                         "(exit 3, CANNOT TEST)")
    a = ap.parse_args()
    if not a.keep_display:
        os.environ.pop("DISPLAY", None)     # the harness's chromium inherits this
    keys = [k for k in a.items.split(",") if k.strip()]
    if len(keys) != 3:
        sys.exit("need exactly three --items")
    os.makedirs(a.work, exist_ok=True)
    labeldir = os.path.join(a.work, "blank")

    fails = []
    res = {}
    for arm, force in (("control", False), ("forced", True)):
        out = os.path.join(a.work, "shots_" + arm)
        r = run(a.det, keys, out, labeldir, force)
        r["file"] = lost_file_keys(out)
        res[arm] = r
        print("%-8s contexts seen %s, forced %s, suspect %s, file %s"
              % (arm, r["n_ctx"], r["forced"], r["suspect"], r["file"]))
        for e in r["events"]:
            print("         signal: %s" % e)
        cs = subprocess.run([sys.executable, os.path.join(HERE, "check_shots.py"), out],
                            capture_output=True, text=True)
        for l in cs.stdout.splitlines():
            if l.startswith(("c_3d_stop", "VERDICT", "blank", "harness")):
                print("         check_shots: %s" % l)
        r["check_rc"] = cs.returncode

    c, f = res["control"], res["forced"]
    if c["file"] is None:
        fails.append("control: %s was not written" % sh.LOST_FILE)
    if c["suspect"] or c["file"]:
        fails.append("control: items named with nothing forced: %s" % c["suspect"])
    cannot = None
    if f["backend"] != "webgl":
        # Not a pass: the thing under test never ran.  Exit 3, apart from FAIL.
        cannot = ("the 3-D panel rendered with %s, not WebGL, so no context existed "
                  "to lose -- see doc pdvd/69 sec 8.3 (an unreachable DISPLAY "
                  "breaks chromium's SwiftShader)" % f["backend"])
    elif not f["forced"] or f["forced"][0] < 1:
        fails.append("forced: no WebGL context could be lost (%s) -- the control "
                     "controls nothing" % f["forced"])
    elif not set(keys[1:]) <= set(f["suspect"]):
        fails.append("forced: a context was lost before item 2 but the harness named %s"
                     % f["suspect"])
    if f["file"] != f["suspect"]:
        fails.append("forced: %s says %s, the run said %s"
                     % (sh.LOST_FILE, f["file"], f["suspect"]))
    verdict = "FAIL" if fails else ("CANNOT TEST" if cannot else "PASS")
    print("RESULT: %s" % verdict)
    for x in fails + ([cannot] if cannot else []):
        print("   " + x)
    return 1 if fails else (3 if cannot else 0)


if __name__ == "__main__":
    sys.exit(main())
